from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import settings
from .models import Listing
from .pipeline import HybridMatcher

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

    args = sys.argv[1:]
    use_prom = "--prom" in args
    verbose = "--verbose" in args
    args = [a for a in args if a not in ("--prom", "--verbose")]

    query = args[0] if args else DEFAULT_QUERY
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    if use_prom:
        from .prom_parser import fetch_and_save
        console.print("[dim]Fetching listings from prom.ua...[/dim]")
        listings = fetch_and_save(query, max_pages=3)
        if not listings:
            console.print("[red]No products found on prom.ua.[/red]\n")
            return
    else:
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

    if verbose:
        d = matcher.debug
        console.print(f"\n[bold cyan]── Verbose Debug ──────────────────────────────[/bold cyan]")
        console.print(f"  Total listings:       {d.total_listings}")
        console.print(f"  FAISS retrieved:      {d.faiss_retrieved}")
        console.print(f"  Above threshold ({settings.similarity_threshold}): {d.above_threshold}")
        console.print(f"  Sent to LLM:          {d.sent_to_llm}")
        console.print(f"  LLM accepted:         {len(d.llm_accepted)}")
        console.print(f"  LLM rejected:         {len(d.llm_rejected)}")

        if d.below_threshold:
            console.print(f"\n[yellow]  Відсіяно порогом схожості ({len(d.below_threshold)} шт):[/yellow]")
            below_table = Table(show_header=True, header_style="yellow", show_lines=False, box=None)
            below_table.add_column("Score", width=7, justify="right")
            below_table.add_column("Listing")
            for text, score in sorted(d.below_threshold, key=lambda x: x[1], reverse=True):
                below_table.add_row(f"{score:.3f}", text[:90])
            console.print(below_table)

        if d.skipped_llm_cap:
            console.print(f"\n[yellow]  Відсіяно лімітом LLM ({len(d.skipped_llm_cap)} шт):[/yellow]")
            for text, score in d.skipped_llm_cap:
                console.print(f"  [dim]{score:.3f}[/dim]  {text[:90]}")

        if d.llm_rejected:
            console.print(f"\n[red]  Відхилено Claude ({len(d.llm_rejected)} шт):[/red]")
            rej_table = Table(show_header=True, header_style="red", show_lines=False, box=None)
            rej_table.add_column("Conf", width=6, justify="right")
            rej_table.add_column("Embed", width=6, justify="right")
            rej_table.add_column("Listing", max_width=45)
            rej_table.add_column("Reason")
            for r in sorted(d.llm_rejected, key=lambda x: x.embedding_score, reverse=True):
                rej_table.add_row(
                    f"{r.confidence:.0%}",
                    f"{r.embedding_score:.3f}",
                    r.listing_text[:45],
                    r.reason[:80],
                )
            console.print(rej_table)

        console.print(f"[bold cyan]───────────────────────────────────────────────[/bold cyan]\n")

    console.print(f"\n[dim]Time: {elapsed:.1f}s | API calls: {matcher.api_calls}[/dim]")

    csv_file = export_results_csv(query, results, elapsed, matcher.api_calls)
    console.print(f"[green]✓[/green] Report saved: [bold]{csv_file}[/bold]\n")


if __name__ == "__main__":
    main()
