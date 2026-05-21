"""
Semantic Product Matcher — Demo with 1000 listings.

Generates 1000 realistic marketplace listings with controlled match/mismatch
patterns, then runs the full hybrid pipeline to show how embedding retrieval
+ LLM verification work together.

Usage:
    python demo.py              # with real Claude API (needs ANTHROPIC_API_KEY)
    python demo.py --mock       # offline mode with rule-based verifier
"""

from __future__ import annotations

import random
import re
import sys
import time

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from src.config import settings
from src.embedder import EmbeddingRetriever
from src.models import Listing, MatchResult

# ---------------------------------------------------------------------------
# Listing generator
# ---------------------------------------------------------------------------

QUERY = "бензиновий генератор Honda 10 кВт"

# fmt: off
EXACT_TEMPLATES = [
    "Бензиновий генератор Honda EU70is, 10 кВт, інверторний",
    "Honda gasoline generator 10 kW, portable, inverter",
    "Генератор Honda 10 кВт бензин, електростарт",
    "Бензогенератор Honda 10000 Вт, тихий, для дому",
    "Honda EU7000is generator, 10 kW, gasoline, low noise",
    "Генератор бензиновий Honda, потужність 10 кВт, 220В",
    "Honda 10кВт генератор бензиновий з AVR стабілізатором",
    "Бензиновий генератор Honda EG10000, 10 кВт, промисловий",
    "Generator Honda 10 kW petrol / gasoline portable",
    "Генератор Honda бензин 10 кВт — ідеально для будинку",
    "Бензогенератор Honda EU70is 10kW inverter silent",
    "Honda генератор 10 кВт, бензин, мідна обмотка",
    "Генератор бензиновий Honda 10.0 кВт, модель EU7000",
    "Honda 10 кВт бензогенератор, гарантія 2 роки",
    "Бензиновий електрогенератор Honda, 10 кВт, новий",
]

WRONG_POWER_TEMPLATES = [
    "Бензиновий генератор Honda {power} кВт, інверторний",
    "Honda gasoline generator {power} kW, portable",
    "Генератор Honda {power} кВт бензин, електростарт",
    "Бензогенератор Honda {power_w} Вт, для дому",
    "Honda generator {power} kW, gasoline, low noise",
    "Генератор бензиновий Honda, потужність {power} кВт",
    "Honda {power}кВт генератор бензиновий з AVR",
    "Бензиновий генератор Honda, {power} кВт промисловий",
]

WRONG_FUEL_TEMPLATES = [
    "Дизельний генератор Honda 10 кВт, водяне охолодження",
    "Honda diesel generator 10 kW, heavy duty",
    "Генератор Honda 10 кВт дизель, електростарт",
    "Дизельний генератор Honda 10000 Вт, безшумний",
    "Honda 10 kW diesel generator, automatic start",
    "Генератор газовий Honda 10 кВт, на природному газі",
    "Honda 10 кВт генератор на газу (LPG/NG)",
    "Електрогенератор Honda 10 кВт на дизелі, 3 фази",
]

WRONG_BRAND_TEMPLATES = [
    "Бензиновий генератор {brand} 10 кВт, інверторний",
    "{brand} gasoline generator 10 kW, portable",
    "Генератор {brand} 10 кВт бензин, електростарт",
    "Бензогенератор {brand} 10000 Вт, для дому",
    "{brand} 10 кВт генератор бензиновий",
    "Генератор бензиновий {brand}, потужність 10 кВт",
]

PARTIAL_INFO_TEMPLATES = [
    "Генератор Honda, бензиновий, портативний, для дачі",
    "Honda generator gasoline portable for home",
    "Бензиновий генератор Honda, мідна обмотка, AVR",
    "Honda бензогенератор, тихий, економний",
    "Генератор Honda бензин, електростарт, колеса",
    "Honda portable gasoline generator, electric start",
    "Бензиновий генератор Honda, новий, в упаковці",
    "Генератор Honda бензиновий для дому та офісу",
]

UNRELATED_CATEGORIES = {
    "drill": [
        "Дриль акумуляторна {brand} 18В, безщіткова",
        "{brand} cordless drill 18V, brushless motor",
        "Перфоратор {brand} 800Вт, SDS-plus, 3 режими",
        "Шуруповерт {brand} 12В, 2 акумулятори",
        "{brand} impact driver 20V, 200 Nm torque",
    ],
    "pump": [
        "Насос {brand} циркуляційний 25-60, для опалення",
        "{brand} water pump 1.5 kW, submersible",
        "Мотопомпа {brand} для брудної води, 1000 л/хв",
        "Насосна станція {brand} 1.1 кВт, автоматична",
        "{brand} drainage pump 750W, stainless steel",
    ],
    "welder": [
        "Зварювальний апарат {brand} 200А, інверторний",
        "{brand} MIG welder 180A, gas/gasless",
        "Інверторний зварювальник {brand} 250А, IGBT",
        "Зварювання {brand} MMA 160A, електродна",
        "{brand} TIG welder 200A, AC/DC, pulse",
    ],
    "compressor": [
        "Компресор {brand} 50л, 1.5 кВт, поршневий",
        "{brand} air compressor 100L, 2.2 kW, belt drive",
        "Компресор {brand} безмасляний 24л, тихий",
        "{brand} compressor 8 bar, 200 L/min",
    ],
    "chainsaw": [
        "Бензопила {brand} 45 см, 2.5 кВт",
        "{brand} chainsaw 18 inch, 50cc",
        "Електропила {brand} 2 кВт, ланцюгова",
        "Пила ланцюгова {brand} акумуляторна 36В",
    ],
    "mower": [
        "Газонокосарка {brand} бензинова, 46 см",
        "{brand} lawn mower electric 1800W, 40 cm",
        "Тример {brand} бензиновий 2.5 кВт",
        "Косарка {brand} робот, автоматична, WiFi",
    ],
    "heater": [
        "Обігрівач {brand} інфрачервоний 2 кВт",
        "{brand} convector heater 1500W, thermostat",
        "Теплова гармата {brand} газова 15 кВт",
        "Котел {brand} газовий 24 кВт, двоконтурний",
    ],
    "lighting": [
        "Прожектор {brand} LED 50Вт, IP65",
        "{brand} floodlight 100W LED, motion sensor",
        "Лампа {brand} LED 15Вт, E27, 4000K",
        "Світильник {brand} вуличний на сонячній батареї",
    ],
}

BRANDS = ["Bosch", "Makita", "DeWalt", "Einhell", "Hyundai", "Forte",
           "Konner & Sohnen", "Vitals", "Sadko", "Sigma", "Зенит",
           "Dnipro-M", "Patriot", "Champion", "Huter", "STIHL"]

WRONG_POWERS = [1, 2, 2.5, 3, 5, 5.5, 6, 6.5, 7, 7.5, 8, 12, 15, 20, 25, 30]
# fmt: on


def generate_listings(n: int = 1000, seed: int = 42) -> tuple[list[Listing], dict[str, list[str]]]:
    """Generate n listings with controlled match/mismatch distribution.
    Returns (listings, category_map) where category_map[category] = [listing_ids].
    """
    rng = random.Random(seed)
    listings: list[Listing] = []
    categories: dict[str, list[str]] = {
        "exact_match": [],
        "wrong_power": [],
        "wrong_fuel": [],
        "wrong_brand": [],
        "partial_info": [],
        "unrelated": [],
    }

    idx = 0

    def add(text: str, cat: str, meta: dict | None = None):
        nonlocal idx
        idx += 1
        lid = str(idx)
        listings.append(Listing(id=lid, text=text, metadata=meta or {}))
        categories[cat].append(lid)

    # --- exact matches (~30) ---
    for t in EXACT_TEMPLATES:
        add(t, "exact_match", {"expected": True})
    for t in rng.choices(EXACT_TEMPLATES, k=15):
        suffix = rng.choice([", акція", " — знижка 10%", " (залишок 3 шт)",
                             ", безкоштовна доставка", " + масло в подарунок"])
        add(t + suffix, "exact_match", {"expected": True})

    # --- wrong power (~120) ---
    for _ in range(120):
        power = rng.choice(WRONG_POWERS)
        t = rng.choice(WRONG_POWER_TEMPLATES)
        text = t.format(power=power, power_w=int(power * 1000))
        add(text, "wrong_power", {"expected": False, "power_kw": power})

    # --- wrong fuel (~60) ---
    for t in WRONG_FUEL_TEMPLATES:
        add(t, "wrong_fuel", {"expected": False})
    for _ in range(52):
        t = rng.choice(WRONG_FUEL_TEMPLATES)
        suffix = rng.choice([", доставка по Україні", " — в наявності",
                             ", гарантія 1 рік", " (нове)"])
        add(t + suffix, "wrong_fuel", {"expected": False})

    # --- wrong brand (~90) ---
    for _ in range(90):
        brand = rng.choice(BRANDS)
        t = rng.choice(WRONG_BRAND_TEMPLATES)
        add(t.format(brand=brand), "wrong_brand", {"expected": False, "brand": brand})

    # --- partial info (~60) ---
    for t in PARTIAL_INFO_TEMPLATES:
        add(t, "partial_info", {"expected": "uncertain"})
    for _ in range(52):
        t = rng.choice(PARTIAL_INFO_TEMPLATES)
        suffix = rng.choice([", ціна договірна", " — б/в", ", торг", " (відновлений)"])
        add(t + suffix, "partial_info", {"expected": "uncertain"})

    # --- unrelated products (fill up to n) ---
    remaining = n - len(listings)
    cats = list(UNRELATED_CATEGORIES.keys())
    for _ in range(remaining):
        cat = rng.choice(cats)
        templates = UNRELATED_CATEGORIES[cat]
        t = rng.choice(templates)
        brand = rng.choice(BRANDS)
        add(t.format(brand=brand), "unrelated", {"expected": False, "type": cat})

    rng.shuffle(listings)
    return listings, categories


# ---------------------------------------------------------------------------
# Mock verifier (offline mode)
# ---------------------------------------------------------------------------

class MockVerifier:
    """Rule-based verifier for demo without API key."""

    def __init__(self):
        self.api_calls = 0

    def verify(self, query: str, listing_id: str, listing_text: str,
               embedding_score: float) -> MatchResult:
        self.api_calls += 1
        text_lower = listing_text.lower()

        has_honda = "honda" in text_lower
        has_gasoline = any(w in text_lower for w in
                          ["бензин", "gasoline", "petrol", "бензогенератор"])
        has_diesel = any(w in text_lower for w in ["дизел", "diesel"])
        has_gas = any(w in text_lower for w in ["газов", "газу", "lpg", "lng"])

        power_match = self._check_power(text_lower, target_kw=10.0)

        matched, mismatched = [], []
        if has_honda:
            matched.append("brand: Honda")
        else:
            mismatched.append("brand mismatch")

        if has_gasoline and not has_diesel and not has_gas:
            matched.append("fuel: gasoline")
        elif has_diesel:
            mismatched.append("fuel: diesel vs gasoline")
        elif has_gas:
            mismatched.append("fuel: gas vs gasoline")

        if power_match == "exact":
            matched.append("power: 10 kW")
        elif power_match == "missing":
            mismatched.append("power: not specified")
        else:
            mismatched.append(f"power: {power_match}")

        is_match = (has_honda and has_gasoline and not has_diesel
                    and not has_gas and power_match == "exact")

        if is_match:
            confidence = 0.90 + embedding_score * 0.08
        elif not mismatched:
            confidence = 0.45
        else:
            confidence = max(0.05, 0.3 - len(mismatched) * 0.1)

        confidence = round(min(confidence, 1.0), 2)

        if is_match:
            reason = "Exact match: Honda gasoline generator 10 kW"
        elif mismatched:
            reason = "Mismatch: " + "; ".join(mismatched)
        else:
            reason = "Insufficient spec information"

        return MatchResult(
            listing_id=listing_id,
            listing_text=listing_text,
            match=is_match,
            confidence=confidence,
            reason=reason,
            matched_specs=matched,
            mismatched_specs=mismatched,
            embedding_score=embedding_score,
        )

    @staticmethod
    def _check_power(text: str, target_kw: float) -> str:
        kw_patterns = re.findall(r"(\d+(?:\.\d+)?)\s*(?:квт|кв|kwt|kw)", text)
        watt_patterns = re.findall(r"(\d+(?:\.\d+)?)\s*(?:вт|w(?:att)?)\b", text)

        values_kw: list[float] = []
        for v in kw_patterns:
            values_kw.append(float(v))
        for v in watt_patterns:
            val = float(v)
            if val >= 100:
                values_kw.append(val / 1000)

        if not values_kw:
            return "missing"

        for v in values_kw:
            if abs(v - target_kw) / target_kw <= 0.05:
                return "exact"

        return f"{values_kw[0]} kW vs {target_kw} kW"


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(use_mock: bool = False):
    console = Console(force_terminal=True)

    # --- Header ---
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Semantic Product Matcher — Demo[/bold cyan]\n"
        f"[dim]Query:[/dim] [bold]{QUERY}[/bold]\n"
        f"[dim]Mode:[/dim]  {'[yellow]Mock verifier (offline)[/yellow]' if use_mock else '[green]Claude API (live)[/green]'}",
        border_style="cyan",
    ))
    console.print()

    # --- Step 1: Generate listings ---
    console.print("[bold]Step 1:[/bold] Generating 1000 listings...")
    listings, categories = generate_listings(1000)

    cat_table = Table(title="Dataset Composition", show_edge=False, pad_edge=False)
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Expected Result")
    cat_table.add_row("Exact match (Honda, gasoline, 10 kW)", str(len(categories["exact_match"])), "[green]match=true[/green]")
    cat_table.add_row("Wrong power (Honda, gasoline, ≠10 kW)", str(len(categories["wrong_power"])), "[red]match=false[/red]")
    cat_table.add_row("Wrong fuel (Honda, diesel/gas, 10 kW)", str(len(categories["wrong_fuel"])), "[red]match=false[/red]")
    cat_table.add_row("Wrong brand (≠Honda, gasoline, 10 kW)", str(len(categories["wrong_brand"])), "[red]match=false[/red]")
    cat_table.add_row("Partial info (Honda, gasoline, no power)", str(len(categories["partial_info"])), "[yellow]low confidence[/yellow]")
    cat_table.add_row("Unrelated products", str(len(categories["unrelated"])), "[red]match=false[/red]")
    cat_table.add_row("[bold]Total[/bold]", f"[bold]{len(listings)}[/bold]", "")
    console.print(cat_table)
    console.print()

    # --- Step 2: Build FAISS index ---
    console.print("[bold]Step 2:[/bold] Building embedding index (BAAI/bge-m3)...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding 1000 listings...", total=None)
        t0 = time.time()
        retriever = EmbeddingRetriever(listings)
        t_embed = time.time() - t0
        progress.update(task, completed=1, total=1,
                        description=f"Index built in {t_embed:.1f}s")

    console.print(f"  [dim]Embedding time: {t_embed:.1f}s for {len(listings)} listings[/dim]")
    console.print()

    # --- Step 3: Retrieve candidates ---
    console.print("[bold]Step 3:[/bold] Embedding retrieval (Stage 1 — FAISS search)...")
    t0 = time.time()
    candidates = retriever.retrieve(QUERY, k=settings.faiss_top_k)
    t_search = time.time() - t0

    above_threshold = [(i, s) for i, s in candidates if s >= settings.similarity_threshold]

    console.print(f"  [dim]Search time: {t_search*1000:.0f}ms[/dim]")
    console.print(f"  [dim]Top-{settings.faiss_top_k} retrieved, "
                  f"{len(above_threshold)} above threshold ({settings.similarity_threshold})[/dim]")
    console.print()

    # show top candidates
    cand_table = Table(title=f"Stage 1: Top {min(20, len(candidates))} Candidates by Cosine Similarity",
                       show_lines=True)
    cand_table.add_column("#", style="dim", width=3)
    cand_table.add_column("Score", justify="right", width=7)
    cand_table.add_column("Listing", max_width=70)
    cand_table.add_column("Category", width=14)

    for rank, (idx, score) in enumerate(candidates[:20], 1):
        lid = listings[idx].id
        cat = "?"
        for c, ids in categories.items():
            if lid in ids:
                cat = c.replace("_", " ")
                break
        cat_style = {"exact match": "green", "wrong power": "red",
                     "wrong fuel": "red", "wrong brand": "red",
                     "partial info": "yellow", "unrelated": "dim"}.get(cat, "")
        above = "●" if score >= settings.similarity_threshold else "○"
        text = listings[idx].text[:70]
        cand_table.add_row(
            f"{above}{rank}",
            f"{score:.3f}",
            text,
            f"[{cat_style}]{cat}[/{cat_style}]" if cat_style else cat,
        )

    console.print(cand_table)
    console.print()
    console.print(
        "  [dim]● = above similarity threshold (sent to LLM)  "
        "○ = below threshold (skipped)[/dim]"
    )
    console.print()

    # --- Step 4: LLM verification ---
    llm_candidates = above_threshold[:settings.llm_top_k]
    console.print(f"[bold]Step 4:[/bold] LLM verification (Stage 2 — "
                  f"{'Mock' if use_mock else 'Claude Haiku 4.5'}) "
                  f"on {len(llm_candidates)} candidates...")

    if use_mock:
        verifier = MockVerifier()
    else:
        from src.llm_verifier import LLMVerifier
        verifier = LLMVerifier()

    results: list[MatchResult] = []
    all_results: list[MatchResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Verifying...", total=len(llm_candidates))
        t0 = time.time()
        for idx, score in llm_candidates:
            listing = listings[idx]
            result = verifier.verify(
                query=QUERY,
                listing_id=listing.id,
                listing_text=listing.text,
                embedding_score=score,
            )
            all_results.append(result)
            if result.match:
                results.append(result)
            progress.advance(task)
        t_llm = time.time() - t0

    results.sort(key=lambda r: r.confidence, reverse=True)

    console.print(f"  [dim]LLM time: {t_llm:.1f}s | API calls: {verifier.api_calls}[/dim]")
    console.print()

    # --- Results ---
    console.print(Panel("[bold green]Final Results[/bold green]", expand=False, border_style="green"))
    console.print()

    # Matches table
    match_table = Table(title=f"Matched Listings ({len(results)} found)", show_lines=True)
    match_table.add_column("#", style="dim", width=3)
    match_table.add_column("ID", width=4)
    match_table.add_column("Conf.", justify="right", width=6)
    match_table.add_column("Embed", justify="right", width=6)
    match_table.add_column("Listing", max_width=55)
    match_table.add_column("Matched Specs", max_width=30)

    for i, r in enumerate(results[:25], 1):
        match_table.add_row(
            str(i),
            r.listing_id,
            f"[green]{r.confidence:.0%}[/green]",
            f"{r.embedding_score:.3f}",
            r.listing_text[:55] + ("..." if len(r.listing_text) > 55 else ""),
            ", ".join(r.matched_specs) if r.matched_specs else "-",
        )

    if len(results) > 25:
        match_table.add_row("...", "", "", "", f"[dim]+{len(results)-25} more[/dim]", "")

    console.print(match_table)
    console.print()

    # Rejections table (most interesting ones)
    rejections = [r for r in all_results if not r.match]
    rejections.sort(key=lambda r: r.embedding_score, reverse=True)

    if rejections:
        rej_table = Table(title=f"Rejected by LLM ({len(rejections)} filtered out)", show_lines=True)
        rej_table.add_column("#", style="dim", width=3)
        rej_table.add_column("ID", width=4)
        rej_table.add_column("Embed", justify="right", width=6)
        rej_table.add_column("Listing", max_width=50)
        rej_table.add_column("Reason", max_width=40, style="red")

        for i, r in enumerate(rejections[:15], 1):
            rej_table.add_row(
                str(i),
                r.listing_id,
                f"{r.embedding_score:.3f}",
                r.listing_text[:50] + ("..." if len(r.listing_text) > 50 else ""),
                r.reason,
            )
        if len(rejections) > 15:
            rej_table.add_row("...", "", "", f"[dim]+{len(rejections)-15} more[/dim]", "")

        console.print(rej_table)
        console.print()

    # --- Summary panel ---
    total_time = t_embed + t_search + t_llm

    # count how many from each category ended up matched
    matched_ids = {r.listing_id for r in results}
    cat_hits = {}
    for cat, ids in categories.items():
        cat_hits[cat] = len(matched_ids & set(ids))

    summary = Text()
    summary.append("Pipeline Summary\n\n", style="bold")
    summary.append(f"  Total listings:          {len(listings)}\n")
    summary.append(f"  Stage 1 (embedding):     {len(listings)} → {len(above_threshold)} candidates\n")
    summary.append(f"  Stage 2 (LLM):           {len(llm_candidates)} → {len(results)} matches\n")
    summary.append(f"  Filtered out by LLM:     {len(rejections)}\n")
    summary.append(f"  API calls:               {verifier.api_calls}\n")
    summary.append(f"  Total time:              {total_time:.1f}s\n\n")
    summary.append("Matches by Category\n\n", style="bold")
    summary.append(f"  ✅ Exact match:          {cat_hits['exact_match']}/{len(categories['exact_match'])}\n", style="green")
    summary.append(f"  ❌ Wrong power:          {cat_hits['wrong_power']}/{len(categories['wrong_power'])}\n", style="red")
    summary.append(f"  ❌ Wrong fuel:           {cat_hits['wrong_fuel']}/{len(categories['wrong_fuel'])}\n", style="red")
    summary.append(f"  ❌ Wrong brand:          {cat_hits['wrong_brand']}/{len(categories['wrong_brand'])}\n", style="red")
    summary.append(f"  ⚠️  Partial info:        {cat_hits['partial_info']}/{len(categories['partial_info'])}\n", style="yellow")
    summary.append(f"  ❌ Unrelated:            {cat_hits['unrelated']}/{len(categories['unrelated'])}\n", style="red")

    # accuracy
    true_positives = cat_hits["exact_match"]
    false_positives = sum(cat_hits[c] for c in ["wrong_power", "wrong_fuel", "wrong_brand", "unrelated"])
    false_negatives = len(categories["exact_match"]) - cat_hits["exact_match"]

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary.append(f"\nAccuracy Metrics (excl. partial)\n\n", style="bold")
    summary.append(f"  Precision:               {precision:.0%}\n")
    summary.append(f"  Recall:                  {recall:.0%}\n")
    summary.append(f"  F1 Score:                {f1:.0%}\n")

    console.print(Panel(summary, border_style="cyan", title="Summary", expand=False))
    console.print()

    # Key insight
    if cat_hits["wrong_power"] == 0:
        console.print(Panel(
            "[bold green]✅ Critical test passed:[/bold green] "
            "Zero wrong-power listings matched!\n"
            "[dim]e.g. '10 кВт vs 15 кВт' correctly returned match=false.\n"
            "This is the core problem embeddings alone cannot solve — "
            "the LLM stage catches numeric mismatches.[/dim]",
            border_style="green", expand=False,
        ))
    else:
        console.print(Panel(
            f"[bold red]⚠ Wrong-power matches: {cat_hits['wrong_power']}[/bold red]\n"
            "[dim]Some power mismatches slipped through. Check LLM prompt.[/dim]",
            border_style="red", expand=False,
        ))
    console.print()


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv
    run_demo(use_mock=use_mock)
