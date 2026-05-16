"""Command line: match one query against a listings file."""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from .config import MatcherConfig
from .io import load_listings, write_report
from .pipeline import HybridMatcher


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="semantic-product-matcher",
        description="Find the listings that match a product query: embedding retrieval + LLM spec verification.",
    )
    p.add_argument("query", help='what to look for, e.g. "Samsung 990 PRO NVMe SSD 2 TB"')
    p.add_argument("-l", "--listings", required=True, help="listings file: .json or .csv (see README)")
    p.add_argument("-r", "--report", help="also write the results to this CSV file")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    console = Console()
    config = MatcherConfig()

    listings = load_listings(args.listings)
    console.print(f"\n[bold]Query:[/bold] {args.query}")
    console.print(f"[dim]Indexing {len(listings)} listings with {config.embedding_model}...[/dim]")

    start = time.time()
    matcher = HybridMatcher(listings, config)
    results = matcher.search(args.query)
    elapsed = time.time() - start

    if results:
        table = Table(title="Matches", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Listing", max_width=60)
        table.add_column("Reason")
        for n, r in enumerate(results, 1):
            table.add_row(str(n), f"{r.confidence:.0%}", r.listing_text, r.reason)
        console.print(table)
    else:
        console.print("[yellow]No matches found.[/yellow]")

    console.print(
        f"[dim]Time: {elapsed:.1f}s | API calls: {matcher.api_calls} | "
        f"tokens in/out: {matcher.input_tokens}/{matcher.output_tokens}[/dim]"
    )
    if args.report:
        path = write_report(args.report, args.query, results, elapsed, matcher.api_calls)
        console.print(f"[green]✓[/green] Report saved: [bold]{path}[/bold]")
    console.print()


if __name__ == "__main__":
    main()
