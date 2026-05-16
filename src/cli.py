from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.models import Listing
from src.pipeline import HybridMatcher

DEFAULT_QUERY = "бензиновий генератор Honda 10 кВт"

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "marketplace_listings.json"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_listings.json"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_listings(path: Path | None = None) -> list[Listing]:
    if path is None:
        path = FIXTURES_PATH if FIXTURES_PATH.exists() else SAMPLE_PATH
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return [Listing(**item) for item in data]


def export_results_csv(query: str, results: list, elapsed: float, api_calls: int) -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = REPORTS_DIR / f"report_{timestamp}.csv"

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Product Matching Report"])
        writer.writerow([])
        writer.writerow(["Search Query", query])
        writer.writerow(["Execution Time (s)", f"{elapsed:.2f}"])
        writer.writerow(["API Calls", api_calls])
        writer.writerow(["Total Matches", len(results)])
        writer.writerow([])

        writer.writerow(["#", "Match", "Confidence", "Listing ID", "Listing Text", "Reason", "Matched Specs", "Mismatched Specs", "Embedding Score"])

        for i, result in enumerate(results, 1):
            writer.writerow([
                i,
                "✅" if result.match else "❌",
                f"{result.confidence:.0%}",
                result.listing_id,
                result.listing_text,
                result.reason,
                "; ".join(result.matched_specs) if result.matched_specs else "",
                "; ".join(result.mismatched_specs) if result.mismatched_specs else "",
                f"{result.embedding_score:.3f}"
            ])

    return str(filename)


def main() -> None:
    console = Console()

    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    console.print("[dim]Loading listings and building index...[/dim]")
    listings = load_listings()
    start = time.time()
    matcher = HybridMatcher(listings)

    console.print(f"[dim]Index built with {len(listings)} listings. Running search...[/dim]\n")
    results = matcher.search(query)
    elapsed = time.time() - start

    table = Table(title="Match Results", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Match", width=5)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Listing", max_width=60)
    table.add_column("Reason")

    for i, r in enumerate(results, 1):
        icon = "[green]✅[/green]" if r.match else "[red]❌[/red]"
        conf = f"{r.confidence:.0%}"
        text = r.listing_text[:60] + ("..." if len(r.listing_text) > 60 else "")
        table.add_row(str(i), icon, conf, text, r.reason)

    if not results:
        console.print("[yellow]No matches found.[/yellow]\n")
    else:
        console.print(table)

    console.print(f"\n[dim]Time: {elapsed:.1f}s | API calls: {matcher.api_calls}[/dim]")

    csv_file = export_results_csv(query, results, elapsed, matcher.api_calls)
    console.print(f"[green]✓[/green] Report saved: [bold]{csv_file}[/bold]\n")


if __name__ == "__main__":
    main()
