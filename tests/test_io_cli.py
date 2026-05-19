import csv
import json

import pytest

from semantic_product_matcher import MatchResult, load_listings, write_report
from semantic_product_matcher import cli


def test_load_json(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps([{"id": "a", "text": "Samsung 990 PRO 2TB", "metadata": {"brand": "Samsung"}}]), encoding="utf-8")
    [listing] = load_listings(p)
    assert (listing.id, listing.text, listing.metadata) == ("a", "Samsung 990 PRO 2TB", {"brand": "Samsung"})


def test_report_round_trip(tmp_path):
    r = MatchResult(listing_id="1", listing_text="Samsung 990 PRO 2TB", match=True, confidence=0.9, reason="ok",
                    matched_specs=["capacity"], embedding_score=0.8)
    path = write_report(tmp_path / "out" / "r.csv", "Samsung 990 PRO 2 TB", [r], 1.5, 3)
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    assert rows[0] == ["Query", "Samsung 990 PRO 2 TB"] and rows[3] == ["Matches", "1"]
    assert rows[-1] == ["1", "1", "Samsung 990 PRO 2TB", "0.90", "0.800", "ok", "capacity", ""]


def test_cli_prints_matches_and_writes_report(tmp_path, monkeypatch, capsys):
    listings = tmp_path / "l.json"
    listings.write_text(json.dumps([{"id": "1", "text": "Samsung 990 PRO 2TB"}]), encoding="utf-8")

    class StubMatcher:
        api_calls, input_tokens, output_tokens = 1, 300, 80

        def __init__(self, listings, config):
            self.listings = listings

        def search(self, query, similarity_threshold=None):
            return [MatchResult(listing_id="1", listing_text="Samsung 990 PRO 2TB", match=True,
                                confidence=0.93, reason="same product", embedding_score=0.9)]

    monkeypatch.setattr(cli, "HybridMatcher", StubMatcher)
    cli.main(["Samsung 990 PRO NVMe SSD 2 TB", "-l", str(listings), "-r", str(tmp_path / "r.csv")])
    out = capsys.readouterr().out
    assert "93%" in out and "same product" in out and "tokens in/out: 300/80" in out
    assert (tmp_path / "r.csv").exists()
