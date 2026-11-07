# semantic-product-matcher

Find the listings that are the same product as a query, however they are worded, without ever treating
"2 TB" and "1 TB" as the same thing. Works with any product catalog: give it listings as text, ask in plain words.

```
Query: "Samsung 990 PRO NVMe SSD 2 TB"

  ✅  Samsung 990 Pro 2000 GB M.2 NVMe drive          2000 GB = 2 TB
  ✅  Samsung 990 PRO with Heatsink 2TB NVMe SSD
  ❌  Samsung 990 PRO NVMe SSD 1TB                    capacity: 1 TB vs 2 TB
  ❌  Samsung 870 EVO 2TB SATA SSD                    model: 870 EVO; interface: SATA
```

## Why two stages

Sentence embeddings are good at "same meaning in different words". They are bad at numbers: a number is just another
token to them, not a quantity. For the query above, the embedding model scores the **wrong** drive higher than a
correct one:

| Listing | Cosine similarity (BAAI/bge-m3) |
|---|---|
| Samsung 990 PRO NVMe SSD 1TB ❌ | 0.906 |
| Samsung 990 Pro 2000 GB M.2 NVMe drive ✅ | 0.870 |
| Samsung 870 EVO 2TB SATA SSD ❌ | 0.710 |
| Logitech MX Master 3S wireless mouse | 0.380 |

Ranking alone can't fix that, so the engine runs two stages:

```
         listings ─┐
                   ▼
  ┌──────────────────────────────────┐
  │ 1. Embedding retrieval            │  BAAI/bge-m3 + FAISS, local, milliseconds
  │    cosine similarity to the query │  keeps everything ≥ SIMILARITY_THRESHOLD (0.55)
  └────────────────┬─────────────────┘
                   ▼ candidates
  ┌──────────────────────────────────┐
  │ 2. LLM verification               │  one structured call per candidate
  │    spec-by-spec check             │  numbers within ±NUMERIC_TOLERANCE (5%)
  └────────────────┬─────────────────┘
                   ▼
       matches, highest confidence first, each with a reason and the specs that matched / didn't
```

Stage 1 gives recall cheaply; stage 2 gives precision where it matters. There is no cap on the number of candidates:
everything above the threshold is verified, so nothing relevant is cut off by an arbitrary top-k.

## Install

```bash
pip install git+https://github.com/fellquent/semantic-product-matcher
# or, to work on it:
git clone https://github.com/fellquent/semantic-product-matcher && cd semantic-product-matcher
pip install -e ".[dev]"
```

Python 3.10+. The first run downloads the embedding model (BAAI/bge-m3, a few GB) into the Hugging Face cache.
The model is multilingual, so listings in other languages match too.

The verifier calls the Anthropic API: set `ANTHROPIC_API_KEY` in the environment or in a `.env` file
(see [.env.example](.env.example)).

## Usage

### Python

```python
from semantic_product_matcher import HybridMatcher, Listing

listings = [
    Listing(id="1", text="Samsung 990 Pro 2000 GB M.2 NVMe drive"),
    Listing(id="2", text="Samsung 990 PRO NVMe SSD 1TB"),
    Listing(id="3", text="Samsung 870 EVO 2TB SATA SSD"),
]

matcher = HybridMatcher(listings)          # builds the index once
for r in matcher.search("Samsung 990 PRO NVMe SSD 2 TB"):
    print(f"{r.confidence:.0%}  {r.listing_text}  — {r.reason}")
```

`search` returns only accepted listings. `matcher.debug` shows what each stage did: scores below the threshold,
listings the verifier rejected and why, and the timing of each stage.

### Command line

```bash
semantic-product-matcher "Samsung 990 PRO NVMe SSD 2 TB" --listings examples/sample_listings.json --verbose
semantic-product-matcher "USB-C charger 65 W" -l products.csv --threshold 0.6 --report results.csv
```

`--verbose` prints the rejected listings with reasons and the ones that never passed the similarity threshold.
Every run prints the number of API calls and tokens used.

### Listings files

- **JSON**: a list of `{"id": "...", "text": "...", "metadata": {...}}` objects
  ([example](examples/sample_listings.json)).
- **CSV**: a `text` column (required) and an optional `id` column; other columns are kept in `metadata`.
  Spreadsheets can be saved as CSV (UTF-8).

## Matching rules

The verifier extracts every attribute stated in the **query**, then checks each one against the listing:

| Query spec | Accepted when |
|---|---|
| Number with unit (TB, GB, W, V, mAh, kg, mm, ...) | within ±5% after unit conversion: 2 TB = 2000 GB, 1 TB is not 2 TB |
| Brand, model, interface, material, color, ... | same meaning: "990 PRO" is not "990 EVO Plus", "NVMe" is not "SATA" |
| Missing or unclear in the listing | **rejected**: a listing that doesn't state the capacity can't be confirmed as 2 TB |

Attributes the query doesn't mention are ignored. If any query spec fails, the listing is rejected with
`confidence = 0` and the failing specs in `mismatched_specs`.

## Configuration

Every setting can be passed in code (`MatcherConfig(similarity_threshold=0.6)`) or set as an environment variable /
`.env` entry:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | | API key for the verifier |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | model used for verification |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | any sentence-transformers model |
| `SIMILARITY_THRESHOLD` | `0.55` | minimum cosine similarity to reach stage 2 |
| `NUMERIC_TOLERANCE` | `0.05` | allowed relative difference between numbers |
| `REASON_LANGUAGE` | `English` | language of the `reason` text |
| `LLM_CONCURRENCY` | `8` | parallel verification requests |
| `MAX_RETRIES` | `5` | retries on rate limits, overload and network errors |
| `FORCE_TOOL_CHOICE` | `true` | force the structured `report_match` call; set `false` for models that reject a forced tool choice (e.g. `claude-opus-5-5`, `claude-sonnet-5-5`) |

**Threshold.** Lower it if relevant listings show up under "below the similarity threshold" in `--verbose`; raise it
to send fewer candidates to the LLM.

**Cost.** Stage 1 runs locally. Stage 2 makes one request per candidate above the threshold, so cost grows with the
number of candidates, not with the size of the catalog. The CLI prints input/output tokens for each run; multiply by
the model's [price](https://docs.anthropic.com/en/docs/about-claude/pricing) to see the cost.

## Extending

Both stages are replaceable:

```python
from semantic_product_matcher import EmbeddingRetriever, HybridMatcher, MatchResult

def my_encoder(texts):            # list[str] -> (n, dim) array with L2-normalized rows
    ...

class MyVerifier:                 # anything with verify_many(query, items) -> list[MatchResult]
    def verify_many(self, query, items):
        return [MatchResult(listing_id=i, listing_text=t, match=..., confidence=..., reason=...,
                            embedding_score=score) for i, t, score in items]

matcher = HybridMatcher(listings, retriever=EmbeddingRetriever(listings, encoder=my_encoder),
                        verifier=MyVerifier())
```

## Demo

[examples/demo.py](examples/demo.py) generates 1000 listings with known answers for the query
"Samsung 990 PRO NVMe SSD 2 TB": exact matches, the same drive in other capacities, other Samsung models
(870 EVO, T7, 980 PRO, 990 EVO Plus), other brands, listings without a capacity, and unrelated electronics.

```bash
python examples/demo.py --mock   # rule-based verifier, no API key needed
python examples/demo.py          # the real LLM verifier
```

It prints what each stage kept and dropped, and precision/recall against the known answers. In stage 1, 13 of the
20 most similar listings are the wrong drive, mostly the same 990 PRO in another capacity; all of them have to be
rejected in stage 2.

## Tests

```bash
pytest                    # fast, offline: fake encoder, mocked API
pytest -m integration     # with the real embedding model
```

## Layout

```
src/semantic_product_matcher/
├── config.py     settings (code, environment, .env)
├── models.py     Listing, MatchResult
├── embedder.py   stage 1: embeddings + FAISS index
├── verifier.py   stage 2: LLM verification, prompt and output schema
├── pipeline.py   HybridMatcher: both stages + debug info
├── io.py         listings from JSON/CSV, CSV reports
└── cli.py        command line
docs/             design notes
examples/         demo and sample data
```

More detail: [docs/architecture.md](docs/architecture.md), [docs/numeric_problem.md](docs/numeric_problem.md).

## License

MIT
