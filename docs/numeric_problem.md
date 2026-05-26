# The numeric specs problem

## Why embeddings fail on numbers

Sentence embeddings place texts with similar meaning close together. That handles wording, abbreviations and
languages well:

- "Samsung 990 Pro 2000 GB M.2 NVMe drive" ≈ "Samsung 990 PRO NVMe SSD 2 TB" → high cosine similarity ✓

But a number is just another token to the model, not a quantity:

- "Samsung 990 PRO NVMe SSD 1TB" ≈ "Samsung 990 PRO NVMe SSD 2 TB" → high cosine similarity ✗

With BAAI/bge-m3 and the query "Samsung 990 PRO NVMe SSD 2 TB", the 1 TB drive scores **0.906** — higher than the
correct "2000 GB" listing at 0.870.
A different model line of the same capacity ("Samsung 870 EVO 2TB SATA SSD") still scores 0.710, far above the
default 0.55 threshold. An unrelated "Logitech MX Master 3S wireless mouse" scores 0.380.

## Why it matters

In product matching the number is often what decides it:

- a 1 TB drive is not a 2 TB drive;
- a 65 W charger is not a 100 W charger;
- a 20000 mAh power bank is not a 10000 mAh one.

## How the engine handles it

1. **Embeddings** find everything that is about the same kind of product (high recall). Wrong-capacity listings
   are expected to pass this stage — and to rank near the top.
2. **The verifier** extracts the specs stated in the query and checks each one explicitly (high precision):
   - numbers must agree within ±5% after unit conversion:
     2 TB = 2000 GB passes, 1 TB and 4 TB fail;
   - a spec the query states but the listing doesn't is a mismatch: "Samsung 990 PRO NVMe SSD" with no capacity
     is rejected for a "2 TB" query, because it cannot be confirmed;
   - any failed spec rejects the listing, with the reason and the failing specs in the result.

The demo in `examples/demo.py` has 120 wrong-capacity listings among 1000. In stage 1, 13 of the 20 most similar
listings are wrong drives; stage 2 must reject all of them.
