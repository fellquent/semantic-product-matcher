import pandas as pd
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

excel_file = Path("test_data.xlsx")
output_file = Path("tests/fixtures/marketplace_listings.json")

df = pd.read_excel(excel_file)

listings = []
for idx, row in df.iterrows():
    text = str(row.iloc[0]) if len(row) > 0 else ""
    if text.strip():
        listings.append({
            "id": str(idx + 1),
            "text": text,
            "metadata": {}
        })

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8-sig") as f:
    json.dump(listings, f, ensure_ascii=False, indent=2)

print(f"Done: Loaded {len(listings)} products")
print(f"Saved to: {output_file}")
print(f"\nFirst 5 products:")
for i, listing in enumerate(listings[:5], 1):
    print(f"{i}. {listing['text'][:80]}...")
