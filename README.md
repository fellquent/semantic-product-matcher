# SpecMatch AI

A hybrid semantic product matching system that compares marketplace listings against a search query. It handles multilingual input (Ukrainian/Russian/English), abbreviations, and unit variations — while ensuring numeric specs like power ratings are never falsely matched.

## How It Works

```
Query: "Honda gasoline generator 10 kW"

     ┌─────────────────────────────────┐
     │  Stage 1: Embedding Retrieval   │
     │  BAAI/bge-m3 + FAISS            │
     │  Fast cosine similarity search  │
     │  → Top 50 candidates            │
     └───────────────┬─────────────────┘
                     │
     ┌───────────────▼─────────────────┐
     │  Stage 2: LLM Verification      │
     │  Claude Haiku 4.5               │
     │  Structured tool_use output     │
     │  Exact numeric spec check       │
     │  → match/no-match + reasoning   │
     └───────────────┬─────────────────┘
                     │
              Sorted Results
```

Embeddings handle semantic similarity (language, synonyms) while the LLM catches what embeddings miss — numeric spec mismatches like "10 kW" vs "15 kW". Numeric specs must match exactly (unit conversion is allowed: 10 kW = 10000 W).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

### Search against a local database (Excel/JSON)

```bash
# Search using local product database (tests/fixtures/marketplace_listings.json)
python -m src.cli "Honda gasoline generator 10 kW"

# Run with default demo query
python -m src.cli

# Show detailed analysis: which products were filtered at each stage
python -m src.cli --verbose "Honda gasoline generator 10 kW"
```

### Search with live Prom.ua scraping

```bash
# Scrape Prom.ua for the query, then run matching on results
python -m src.cli --prom "інвертор Deye 6 кВт"

# Scrape + verbose analysis
python -m src.cli --prom --verbose "інвертор Deye 6 кВт"
```

### Scrape Prom.ua only (no matching)

```bash
# Fetch products from Prom.ua and save to marketplace_listings.json
# Also saves screenshots of each page to screenshots/
python -m src.prom_parser "генератор 10 кВт"
```

### Load your own Excel database

```bash
# Convert test_data.xlsx (column A = product names) to JSON
python load_excel_to_json.py
```

### Sample Output

```
Query: Honda gasoline generator 10 kW

                    Match Results
┌───┬───────┬────────────┬──────────────────────────┬──────────────────────┐
│ # │ Match │ Confidence │ Listing                  │ Reason               │
├───┼───────┼────────────┼──────────────────────────┼──────────────────────┤
│ 1 │ ✅    │       95%  │ Honda gasoline generator │ Exact match: Honda,  │
│   │       │            │ 10 kW, model EU70is...   │ 10 kW, gasoline      │
├───┼───────┼────────────┼──────────────────────────┼──────────────────────┤
│ 2 │ ✅    │       93%  │ Honda EU70is 10 kW       │ Same product, specs  │
│   │       │            │ inverter generator...    │ match exactly        │
└───┴───────┴────────────┴──────────────────────────┴──────────────────────┘

Time: 4.2s | API calls: 5
```

## Loading Your Own Product Database

Place your product list in an Excel file (`test_data.xlsx`) with product names in column A, then run:

```bash
python load_excel_to_json.py
```

This converts the Excel file into `tests/fixtures/marketplace_listings.json`. The CLI will automatically use it for all searches.

## CSV Reports

Every search automatically saves a CSV report to the `reports/` folder:

```
reports/report_20260530_143025.csv
```

Each report contains: query, execution time, API call count, and a full results table with confidence scores, matched/mismatched specs, and embedding scores. Files are UTF-8 encoded and open correctly in Excel.

## Running Tests

```bash
# Unit tests only (no API calls)
pytest tests/ -v

# Include integration tests (requires API key)
pytest tests/ -v -m integration
```

## Cost Estimation

Using Claude Haiku 4.5 ($1 / $5 per 1M input/output tokens):

| Operation          | Tokens (approx) | Cost        |
|--------------------|------------------|-------------|
| 1 listing verify   | ~300 in, ~100 out| ~$0.0008    |
| 1 search (20 LLM)  | ~6K in, ~2K out  | ~$0.016     |
| 1000 searches/day  | ~6M in, ~2M out  | ~$16/day    |

## Project Structure

```
src/
├── config.py        — Settings from .env
├── models.py        — Pydantic data models
├── embedder.py      — FAISS index + sentence-transformers
├── llm_verifier.py  — Claude API verification
├── pipeline.py      — Hybrid matcher orchestration
└── cli.py           — Rich CLI interface
```
