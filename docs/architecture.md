# Architecture

## Hybrid Pipeline

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Stage 1: Embedding      │
│  BAAI/bge-m3 + FAISS     │
│  → Top-50 by cosine sim  │
│  (fast, ~10ms)           │
└──────────┬───────────────┘
           │ candidates with score ≥ 0.55
           ▼
┌──────────────────────────┐
│  Stage 2: LLM Verify     │
│  Claude Haiku 4.5        │
│  → match/no-match + why  │
│  (accurate, ~200ms each) │
└──────────┬───────────────┘
           │ match=true only
           ▼
     Sorted Results
```

## Why Two Stages?

Embeddings are fast and cheap but cannot reliably distinguish numeric specs
(e.g., "10 kW" vs "15 kW" score similarly in vector space). The LLM stage
catches these cases with explicit comparison logic.

## Cost Model

- Stage 1 is free (local computation)
- Stage 2 costs ~$0.001 per listing verified (Claude Haiku 4.5)
- With top-20 LLM verification: ~$0.02 per search query
