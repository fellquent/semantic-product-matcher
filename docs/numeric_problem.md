# The Numeric Specs Problem

## Why Embeddings Fail on Numbers

Sentence embeddings represent text as dense vectors in a high-dimensional space.
Similar meanings map to nearby points. This works well for semantic similarity:

- "gasoline generator" ≈ "бензиновий генератор" → high cosine similarity ✓

But embeddings treat numbers as just another token, not as quantities:

- "Honda 10 kW" ≈ "Honda 15 kW" → high cosine similarity ✗

The embedding model sees that both texts describe Honda generators — the numbers
10 and 15 are just different tokens in an otherwise identical context. The cosine
similarity might be 0.92+, which looks like a great match.

## Why This Matters

In product matching, numeric specs are often the most critical differentiator:
- A 10 kW generator cannot substitute for a 15 kW generator
- A 220V device is not the same as a 380V device
- A 5L capacity tank is different from a 10L tank

## Our Solution

We use a two-stage pipeline:
1. **Embeddings** find semantically relevant candidates (high recall)
2. **LLM verification** compares numeric specs explicitly (high precision)

The LLM prompt includes explicit rules:
- Numeric specs must match within ±5% tolerance
- Unit conversions are handled (10 кВт = 10000 Вт = 10 kW)
- Missing specs reduce confidence but don't auto-reject

This hybrid approach gives us both speed (embeddings filter thousands of listings
in milliseconds) and accuracy (LLM catches numeric mismatches that embeddings miss).
