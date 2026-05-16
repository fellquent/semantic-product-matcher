"""Loading listings from files and writing result reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import Listing, MatchResult


def load_listings(path: str | Path) -> list[Listing]:
    """Listings from a JSON file: a list of {"id": ..., "text": ..., "metadata": {...}} objects."""
    with open(path, encoding="utf-8-sig") as f:
        return [Listing(**item) for item in json.load(f)]


def write_report(
    path: str | Path,
    query: str,
    results: Iterable[MatchResult],
    elapsed: float,
    api_calls: int,
) -> Path:
    """CSV report of matched listings; UTF-8 with BOM so Excel opens it correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Query", query])
        w.writerow(["Execution time (s)", f"{elapsed:.2f}"])
        w.writerow(["API calls", api_calls])
        w.writerow(["Matches", len(results)])
        w.writerow([])
        w.writerow(["#", "Listing ID", "Listing", "Confidence", "Embedding score",
                    "Reason", "Matched specs", "Mismatched specs"])
        for n, r in enumerate(results, 1):
            w.writerow([n, r.listing_id, r.listing_text, f"{r.confidence:.2f}", f"{r.embedding_score:.3f}",
                        r.reason, "; ".join(r.matched_specs), "; ".join(r.mismatched_specs)])
    return path
