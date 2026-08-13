# Architecture

```
query ─┐
       ▼
┌────────────────────────────┐
│ EmbeddingRetriever         │  embedder.py
│ encode query, FAISS search │  inner product of L2-normalized vectors = cosine similarity
│ score every listing        │  index is built once per HybridMatcher
└────────────┬───────────────┘
             │ (index, score) for all listings
             ▼
   score ≥ similarity_threshold ?  ── no ──▶ debug.below_threshold
             │ yes
             ▼
┌────────────────────────────┐
│ Verifier                   │  verifier.py
│ one request per candidate  │  run in parallel (llm_concurrency)
│ structured verdict         │  report_match tool: match, confidence, reason, specs
└────────────┬───────────────┘
             ▼
   match ?  ── no ──▶ debug.llm_rejected
             │ yes
             ▼
   results, sorted by confidence
```

## Components

- **`EmbeddingRetriever`** takes any encoder: a function from texts to L2-normalized vectors. The default wraps a
  sentence-transformers model (BAAI/bge-m3: multilingual, 1024 dimensions), loaded once per process.
- **`Verifier`** is a protocol with one method, `verify_many(query, items)`. `ClaudeVerifier` is the default: it sends
  each candidate to the Anthropic API with a fixed system prompt and a `report_match` tool whose input schema is the
  verdict. Retries on rate limits, overload and network errors are left to the SDK (`max_retries`).
- **`HybridMatcher`** wires the two together, applies the threshold and keeps `SearchDebugInfo` for the last search.

## Why no top-k

The retriever scores every listing and the threshold is the only cut. A fixed top-k would silently drop relevant
listings from large catalogs; a threshold keeps recall independent of catalog size. The price is that the number of
verification calls depends on how many listings clear the threshold — that is what `similarity_threshold` tunes.

## Cost

Stage 1 is local. Stage 2 is one LLM request per candidate, so cost per query is roughly
`candidates × (input tokens × input price + output tokens × output price)`. `HybridMatcher` exposes `api_calls`,
`input_tokens` and `output_tokens`, and the CLI prints them after every run.
