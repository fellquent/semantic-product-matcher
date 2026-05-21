"""
Semantic Product Matcher - demo with 1000 listings.

Generates 1000 listings with known answers for one query and runs the whole
pipeline on them, showing what each stage keeps and drops.

Usage (after `pip install -e .`):
    python examples/demo.py          # real LLM verifier (needs ANTHROPIC_API_KEY)
    python examples/demo.py --mock   # offline: rule-based verifier, no API calls
"""

from __future__ import annotations

import random
import re
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from semantic_product_matcher import ClaudeVerifier, EmbeddingRetriever, Listing, MatchResult, MatcherConfig

settings = MatcherConfig()

# ---------------------------------------------------------------------------
# Listing generator
# ---------------------------------------------------------------------------

QUERY = "Samsung 990 PRO NVMe SSD 2 TB"
TARGET_TB = 2.0
TOLERANCE = 0.05  # same ±5% the LLM verifier uses by default

# fmt: off
EXACT_TEMPLATES = [
    "Samsung 990 PRO 2TB PCIe 4.0 NVMe M.2 Internal SSD",
    "Samsung 990 Pro 2 TB NVMe M.2 2280 solid state drive",
    "SSD Samsung 990 PRO 2TB, NVMe, PCIe Gen4 x4, up to 7450 MB/s",
    "Samsung 990 PRO NVMe SSD 2000 GB, M.2",
    "Samsung 990 PRO 2TB NVMe (MZ-V9P2T0BW)",
    "Samsung 990 PRO with Heatsink 2TB NVMe SSD for PS5",
    "Samsung 990 PRO 2 TB M.2 NVMe gaming SSD",
    "Samsung NVMe SSD 990 PRO, 2TB, V-NAND TLC",
    "990 PRO 2TB Samsung PCIe 4.0 NVMe M.2 2280",
    "Samsung 990 PRO 2TB internal NVMe drive, 5-year warranty",
    "Samsung 990 Pro NVMe M.2 2TB, read 7450 MB/s, write 6900 MB/s",
    "Samsung 990 PRO 2TB NVMe SSD, Gen4, low power",
    "Samsung 990 PRO 2.0 TB PCIe NVMe M.2",
    "Samsung 990 PRO NVMe 2TB SSD for desktop and laptop",
    "Samsung SSD 990 PRO 2TB M.2 NVMe, new, sealed",
]

WRONG_CAPACITY_TEMPLATES = [
    "Samsung 990 PRO {cap} PCIe 4.0 NVMe M.2 Internal SSD",
    "Samsung 990 Pro {cap} NVMe M.2 2280 solid state drive",
    "SSD Samsung 990 PRO {cap}, NVMe, PCIe Gen4 x4",
    "Samsung 990 PRO NVMe SSD {cap}, M.2",
    "Samsung 990 PRO with Heatsink {cap} NVMe SSD",
    "Samsung 990 PRO {cap} M.2 NVMe gaming SSD",
    "Samsung NVMe SSD 990 PRO, {cap}, V-NAND TLC",
    "Samsung 990 PRO {cap} internal NVMe drive",
]
WRONG_CAPACITIES = ["250GB", "500GB", "500 GB", "1TB", "1 TB", "1000 GB", "4TB", "4 TB", "4000 GB"]

# Same brand and capacity, different drive: other model line or interface.
WRONG_MODEL_TEMPLATES = [
    "Samsung 870 EVO 2TB SATA III 2.5\" Internal SSD",
    "Samsung 870 QVO 2TB SATA SSD",
    "Samsung T7 Shield 2TB portable SSD, USB 3.2",
    "Samsung T9 2TB external USB-C SSD",
    "Samsung 980 PRO 2TB NVMe M.2 SSD",
    "Samsung 990 EVO Plus 2TB NVMe M.2 SSD",
    "Samsung 870 EVO 2 TB 2.5-inch SATA solid state drive",
    "Samsung T7 2TB USB 3.2 Gen 2 portable SSD",
]

WRONG_BRAND_TEMPLATES = [
    "{brand} 2TB PCIe 4.0 NVMe M.2 SSD",
    "{brand} NVMe SSD 2TB, M.2 2280, Gen4",
    "{brand} 2 TB M.2 NVMe internal solid state drive",
    "{brand} 2TB NVMe SSD, up to 7300 MB/s",
    "{brand} 2000GB PCIe Gen4 NVMe M.2 SSD",
]
SSD_BRANDS = ["WD_BLACK SN850X", "Crucial T500", "Kingston KC3000", "Lexar NM790", "Sabrent Rocket 4 Plus",
              "Seagate FireCuda 530", "SK hynix Platinum P41", "Corsair MP600 PRO", "Solidigm P44 Pro", "TeamGroup MP44"]

MISSING_CAPACITY_TEMPLATES = [
    "Samsung 990 PRO NVMe M.2 SSD, PCIe 4.0",
    "Samsung 990 Pro internal NVMe SSD, up to 7450 MB/s",
    "Samsung 990 PRO NVMe solid state drive, M.2 2280",
    "Samsung 990 PRO PCIe Gen4 NVMe SSD, V-NAND",
    "Samsung 990 PRO NVMe SSD with heatsink",
    "Samsung 990 PRO M.2 NVMe, for gaming PCs",
    "Samsung 990 Pro NVMe internal drive, new",
    "Samsung 990 PRO NVMe SSD, 5-year warranty",
]

UNRELATED_CATEGORIES = {
    "monitor": [
        "{brand} 27\" 4K IPS monitor, 144 Hz",
        "{brand} 32-inch curved gaming monitor, 165 Hz",
        "{brand} 24\" Full HD office monitor",
    ],
    "keyboard": [
        "{brand} mechanical keyboard, RGB, hot-swap",
        "{brand} wireless keyboard, quiet keys",
        "{brand} 75% gaming keyboard, brown switches",
    ],
    "audio": [
        "{brand} wireless headphones, active noise cancelling",
        "{brand} gaming headset 7.1, USB",
        "{brand} true wireless earbuds, 30 h battery",
    ],
    "network": [
        "{brand} Wi-Fi 6 router AX3000, dual band",
        "{brand} mesh Wi-Fi system, 3-pack",
        "{brand} 8-port gigabit switch",
    ],
    "laptop": [
        "{brand} 15.6\" laptop, 16 GB RAM, 1 TB SSD",
        "{brand} 14\" ultrabook, 32 GB RAM, 2 TB SSD",
        "{brand} gaming laptop RTX 4060, 512 GB SSD",
    ],
    "power": [
        "{brand} power bank 20000 mAh, USB-C PD 65 W",
        "{brand} portable charger 10000 mAh",
    ],
    "mouse": [
        "{brand} wireless mouse, 4000 DPI",
        "{brand} ergonomic vertical mouse",
    ],
    "storage": [
        "{brand} 2TB external hard drive, USB 3.0",
        "{brand} 128 GB USB flash drive",
        "{brand} microSD card 256 GB, A2",
    ],
}

BRANDS = ["Logitech", "ASUS", "Dell", "HP", "Lenovo", "Sony", "Anker", "TP-Link",
          "Razer", "LG", "Acer", "MSI", "Xiaomi", "JBL", "Seagate", "Samsung"]
# fmt: on

CATEGORY_LABELS = {
    "exact_match": ("Exact match (990 PRO, NVMe, 2 TB)", "[green]match=true[/green]"),
    "wrong_capacity": ("Wrong capacity (990 PRO, ≠2 TB)", "[red]match=false[/red]"),
    "wrong_model": ("Wrong model (870 EVO, T7, 980 PRO, 990 EVO...)", "[red]match=false[/red]"),
    "wrong_brand": ("Wrong brand (other 2 TB NVMe drives)", "[red]match=false[/red]"),
    "missing_capacity": ("Capacity not stated (990 PRO, NVMe)", "[yellow]match=false (can't confirm)[/yellow]"),
    "unrelated": ("Unrelated products", "[red]match=false[/red]"),
}


def generate_listings(n: int = 1000, seed: int = 42) -> tuple[list[Listing], dict[str, list[str]]]:
    """Generate n listings with a controlled mix of matches and near misses.
    Returns (listings, category_map) where category_map[category] = [listing_ids]."""
    rng = random.Random(seed)
    listings: list[Listing] = []
    categories: dict[str, list[str]] = {c: [] for c in CATEGORY_LABELS}

    def add(text: str, cat: str):
        lid = str(len(listings) + 1)
        listings.append(Listing(id=lid, text=text, metadata={"category": cat}))
        categories[cat].append(lid)

    # exact matches (30)
    for t in EXACT_TEMPLATES:
        add(t, "exact_match")
    for t in rng.choices(EXACT_TEMPLATES, k=15):
        add(t + rng.choice([", free shipping", " - 10% off", " (3 left in stock)", ", ships today",
                            " + M.2 screwdriver"]), "exact_match")

    # same drive, wrong capacity (120)
    for _ in range(120):
        add(rng.choice(WRONG_CAPACITY_TEMPLATES).format(cap=rng.choice(WRONG_CAPACITIES)), "wrong_capacity")

    # same brand and capacity, different model or interface (60)
    for t in WRONG_MODEL_TEMPLATES:
        add(t, "wrong_model")
    for _ in range(52):
        add(rng.choice(WRONG_MODEL_TEMPLATES) + rng.choice([", free shipping", " - in stock", ", 3-year warranty",
                                                            " (new)"]), "wrong_model")

    # other brands, same capacity and interface (90)
    for _ in range(90):
        add(rng.choice(WRONG_BRAND_TEMPLATES).format(brand=rng.choice(SSD_BRANDS)), "wrong_brand")

    # right drive, capacity not stated (60)
    for t in MISSING_CAPACITY_TEMPLATES:
        add(t, "missing_capacity")
    for _ in range(52):
        add(rng.choice(MISSING_CAPACITY_TEMPLATES) + rng.choice([", price on request", " - used", ", negotiable",
                                                                 " (refurbished)"]), "missing_capacity")

    # unrelated products (the rest)
    while len(listings) < n:
        templates = UNRELATED_CATEGORIES[rng.choice(list(UNRELATED_CATEGORIES))]
        add(rng.choice(templates).format(brand=rng.choice(BRANDS)), "unrelated")

    rng.shuffle(listings)
    return listings, categories


# ---------------------------------------------------------------------------
# Mock verifier (offline mode)
# ---------------------------------------------------------------------------

class MockVerifier:
    """Rule-based stand-in for the LLM: checks the four specs of QUERY explicitly."""

    def __init__(self):
        self.api_calls = 0

    def verify(self, query: str, listing_id: str, listing_text: str, embedding_score: float) -> MatchResult:
        self.api_calls += 1
        text = listing_text.lower()
        matched, mismatched = [], []

        (matched if "samsung" in text else mismatched).append("brand: Samsung")
        (matched if "990 pro" in text else mismatched).append("model: 990 PRO")
        if "nvme" in text and not any(w in text for w in ("sata", "usb")):
            matched.append("interface: NVMe")
        else:
            mismatched.append("interface: NVMe")

        capacity = self._capacity_tb(text)
        if capacity is None:
            mismatched.append("capacity: not stated")
        elif abs(capacity - TARGET_TB) / TARGET_TB <= TOLERANCE:
            matched.append("capacity: 2 TB")
        else:
            mismatched.append(f"capacity: {capacity:g} TB vs 2 TB")

        is_match = not mismatched
        return MatchResult(
            listing_id=listing_id,
            listing_text=listing_text,
            match=is_match,
            confidence=round(0.9 + embedding_score * 0.08, 2) if is_match else 0.0,
            reason="All query specs match" if is_match else "Mismatch: " + "; ".join(mismatched),
            matched_specs=matched,
            mismatched_specs=mismatched,
            embedding_score=embedding_score,
        )

    @staticmethod
    def _capacity_tb(text: str) -> float | None:
        """Drive capacity in TB; ignores small GB values such as RAM."""
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", text):
            tb = float(value) if unit == "tb" else float(value) / 1000
            if unit == "tb" or float(value) >= 120:
                return tb
        return None


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool = False):
    console = Console(force_terminal=True)
    threshold = settings.similarity_threshold

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Semantic Product Matcher - Demo[/bold cyan]\n"
        f"[dim]Query:[/dim] [bold]{QUERY}[/bold]\n"
        f"[dim]Mode:[/dim]  {'[yellow]Mock verifier (offline)[/yellow]' if use_mock else '[green]LLM verifier (live)[/green]'}",
        border_style="cyan",
    ))
    console.print()

    # --- Step 1: listings ---
    console.print("[bold]Step 1:[/bold] Generating 1000 listings...")
    listings, categories = generate_listings(1000)
    category_of = {lid: cat for cat, ids in categories.items() for lid in ids}

    cat_table = Table(title="Dataset", show_edge=False, pad_edge=False)
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Expected")
    for cat, (label, expected) in CATEGORY_LABELS.items():
        cat_table.add_row(label, str(len(categories[cat])), expected)
    cat_table.add_row("[bold]Total[/bold]", f"[bold]{len(listings)}[/bold]", "")
    console.print(cat_table)
    console.print()

    # --- Step 2: index ---
    console.print(f"[bold]Step 2:[/bold] Building the embedding index ({settings.embedding_model})...")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task(f"Encoding {len(listings)} listings...", total=None)
        t0 = time.time()
        retriever = EmbeddingRetriever(listings, model_name=settings.embedding_model)
        t_embed = time.time() - t0
        progress.update(task, completed=1, total=1, description=f"Index built in {t_embed:.1f}s")
    console.print()

    # --- Step 3: retrieval ---
    console.print("[bold]Step 3:[/bold] Stage 1 - embedding retrieval")
    t0 = time.time()
    scored = retriever.retrieve(QUERY)
    t_search = time.time() - t0
    candidates = [(i, s) for i, s in scored if s >= threshold]
    console.print(f"  [dim]{len(scored)} scored in {t_search * 1000:.0f} ms, "
                  f"{len(candidates)} above the threshold ({threshold})[/dim]")
    console.print()

    styles = {"exact_match": "green", "missing_capacity": "yellow", "unrelated": "dim"}
    top = Table(title="Stage 1: top 20 by cosine similarity", show_lines=True)
    top.add_column("#", style="dim", width=4)
    top.add_column("Score", justify="right", width=7)
    top.add_column("Listing", max_width=62)
    top.add_column("Category", width=16)
    for rank, (idx, score) in enumerate(scored[:20], 1):
        cat = category_of[listings[idx].id]
        style = styles.get(cat, "red")
        marker = "●" if score >= threshold else "○"
        top.add_row(f"{marker}{rank}", f"{score:.3f}", listings[idx].text, f"[{style}]{cat.replace('_', ' ')}[/{style}]")
    console.print(top)
    console.print("  [dim]Red rows are near misses that embeddings rank as highly as the real matches - "
                  "stage 2 has to reject them.[/dim]")
    console.print()

    # --- Step 4: verification ---
    verifier = MockVerifier() if use_mock else ClaudeVerifier(settings)
    console.print(f"[bold]Step 4:[/bold] Stage 2 - verification "
                  f"({'mock' if use_mock else settings.claude_model}) of {len(candidates)} candidates...")
    results: list[MatchResult] = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Verifying...", total=len(candidates))
        t0 = time.time()
        for idx, score in candidates:
            listing = listings[idx]
            results.append(verifier.verify(query=QUERY, listing_id=listing.id,
                                           listing_text=listing.text, embedding_score=score))
            progress.advance(task)
        t_llm = time.time() - t0
    matches = sorted((r for r in results if r.match), key=lambda r: r.confidence, reverse=True)
    rejected = sorted((r for r in results if not r.match), key=lambda r: r.embedding_score, reverse=True)
    console.print(f"  [dim]{t_llm:.1f}s, API calls: {verifier.api_calls}[/dim]")
    console.print()

    match_table = Table(title=f"Matches ({len(matches)})", show_lines=True)
    match_table.add_column("#", style="dim", width=3)
    match_table.add_column("Conf.", justify="right", width=6)
    match_table.add_column("Embed", justify="right", width=6)
    match_table.add_column("Listing", max_width=60)
    for n, r in enumerate(matches[:15], 1):
        match_table.add_row(str(n), f"[green]{r.confidence:.0%}[/green]", f"{r.embedding_score:.3f}", r.listing_text)
    if len(matches) > 15:
        match_table.add_row("...", "", "", f"[dim]+{len(matches) - 15} more[/dim]")
    console.print(match_table)
    console.print()

    if rejected:
        rej_table = Table(title=f"Rejected in stage 2 ({len(rejected)}), highest similarity first", show_lines=True)
        rej_table.add_column("#", style="dim", width=3)
        rej_table.add_column("Embed", justify="right", width=6)
        rej_table.add_column("Listing", max_width=50)
        rej_table.add_column("Reason", max_width=40, style="red")
        for n, r in enumerate(rejected[:15], 1):
            rej_table.add_row(str(n), f"{r.embedding_score:.3f}", r.listing_text, r.reason)
        console.print(rej_table)
        console.print()

    # --- Summary ---
    matched_ids = {r.listing_id for r in matches}
    hits = {cat: len(matched_ids & set(ids)) for cat, ids in categories.items()}
    tp = hits["exact_match"]
    fp = sum(hits[c] for c in categories if c != "exact_match")
    fn = len(categories["exact_match"]) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = Text()
    summary.append("Pipeline\n\n", style="bold")
    summary.append(f"  Stage 1 (embeddings):  {len(listings)} → {len(candidates)} candidates\n")
    summary.append(f"  Stage 2 (verifier):    {len(candidates)} → {len(matches)} matches\n")
    summary.append(f"  API calls:             {verifier.api_calls}\n")
    summary.append(f"  Time:                  {t_embed + t_search + t_llm:.1f}s\n\n")
    summary.append("Matched, by category\n\n", style="bold")
    for cat, (label, _) in CATEGORY_LABELS.items():
        style = "green" if cat == "exact_match" else "red"
        summary.append(f"  {label:<48} {hits[cat]}/{len(categories[cat])}\n", style=style)
    summary.append("\nAgainst the known answers\n\n", style="bold")
    summary.append(f"  Precision:  {precision:.0%}\n  Recall:     {recall:.0%}\n  F1:         {f1:.0%}\n")
    console.print(Panel(summary, border_style="cyan", title="Summary", expand=False))
    console.print()

    if hits["wrong_capacity"] == 0:
        console.print(Panel(
            "[bold green]No wrong-capacity drive was accepted.[/bold green]\n"
            "[dim]'990 PRO 1TB' scores as high as the real 2 TB listings in stage 1 - "
            "embeddings can't tell the numbers apart; stage 2 does.[/dim]",
            border_style="green", expand=False,
        ))
    else:
        console.print(Panel(
            f"[bold red]Wrong-capacity drives accepted: {hits['wrong_capacity']}[/bold red]\n"
            "[dim]Check the verifier prompt and NUMERIC_TOLERANCE.[/dim]",
            border_style="red", expand=False,
        ))
    console.print()


if __name__ == "__main__":
    run_demo(use_mock="--mock" in sys.argv)
