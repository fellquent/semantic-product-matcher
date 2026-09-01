from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

from .models import Listing

PROM_SEARCH_URL = "https://prom.ua/ua/search?search_term={query}&page={page}"
LISTINGS_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "marketplace_listings.json"

# Debug screenshots go next to whatever launched the program (run.bat does
# `cd /d "%~dp0"`, so that's the portable folder) — never inside the installed
# package, which in a portable build lives in venv\Lib\site-packages and has no
# business holding run output. Set PROM_SCREENSHOTS=0 to skip them entirely.
SCREENSHOTS_DIR = Path.cwd() / "screenshots"
SAVE_SCREENSHOTS = os.getenv("PROM_SCREENSHOTS", "1").strip().lower() not in ("0", "false", "no", "ні")


def _ensure_chromium_installed(playwright) -> None:
    """Playwright never auto-downloads its browser binary — it deliberately
    requires a separate `playwright install` step. A frozen .exe user has no
    Python/pip to run that with, so trigger the one-time download ourselves
    on first use instead of letting BrowserType.launch() fail.

    Takes the caller's Playwright instance on purpose. Opening a second
    sync_playwright() context just to read a path started a whole extra Node
    driver on every single scrape — wasted memory, and its teardown is what
    printed "Task was destroyed but it is pending!" before each run.
    """
    exe_path = playwright.chromium.executable_path
    if os.path.exists(exe_path):
        return

    print("Перше використання: завантажуємо браузер для скрапінгу (одноразово, потрібен інтернет)...")
    if getattr(sys, "frozen", False):
        # Re-invoke this same frozen executable with a hidden flag; the main
        # entry point routes it to playwright's own install routine and exits.
        subprocess.run([sys.executable, "--playwright-install"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


def _save_debug_screenshot(page, page_num: int) -> None:
    """Best-effort debug screenshot — must never abort a scrape.

    By the time this runs the products are already extracted, so a failure here
    costs nothing but used to raise straight out of search_prom and kill the
    whole run. full_page=True is the usual culprit: after lazy-loading a search
    page is tall enough to blow past Chromium's capture limits and come back as
    "Unable to capture screenshot", so fall back to the viewport when it does.
    """
    if not SAVE_SCREENSHOTS:
        return
    try:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOTS_DIR / f"page_{page_num}.png"
        try:
            page.screenshot(path=str(path), full_page=True)
        except Exception:
            page.screenshot(path=str(path))
        print(f"Screenshot saved: {path}")
    except Exception as e:
        print(f"Screenshot skipped for page {page_num}: {type(e).__name__}")


def _extract_items(page, query: str) -> list[dict]:
    """Extract product data from JSON-LD schema inside product cards."""
    raw = page.evaluate("""
        () => {
            const results = [];
            for (const block of document.querySelectorAll('[data-qaid="product_block"]')) {
                const script = block.querySelector('script[type="application/ld+json"]');
                const link = block.querySelector('a[data-qaid="product_link"]');
                if (!script || !link) continue;
                try {
                    const data = JSON.parse(script.textContent);
                    const name = data.name || '';
                    if (name.length < 5) continue;

                    const offers = data.offers || {};
                    const seller = offers.seller || {};

                    // Extract sales count from DOM (not in JSON-LD)
                    const salesEl = block.querySelector('[data-qaid="product_sales_count"]');
                    const sales = salesEl ? salesEl.innerText.trim() : '';

                    // Extract article/SKU from DOM
                    const skuEl = block.querySelector('[data-qaid="product_code"]');
                    const sku = skuEl ? skuEl.innerText.replace(/[^0-9a-zA-Zа-яА-Я\-]/g, '').trim() : (data.sku || '');

                    // CompanyID from product URL or data attribute
                    const companyId = block.getAttribute('data-company-id') || '';

                    // Shop name: prefer JSON-LD seller, fallback to DOM
                    let shopName = seller.name || '';
                    if (!shopName) {
                        const shopEl = block.querySelector('[data-qaid="company_name"]') ||
                                       block.querySelector('[data-qaid="shop_name"]') ||
                                       block.querySelector('a[data-qaid="company_link"]');
                        shopName = shopEl ? shopEl.innerText.trim() : '';
                    }

                    // Price: prefer JSON-LD, fallback to DOM
                    let price = offers.price || '';
                    if (!price) {
                        const priceEl = block.querySelector('[data-qaid="product_price"]') ||
                                        block.querySelector('[class*="price"]');
                        if (priceEl) {
                            const priceText = priceEl.innerText.replace(/[^\d.,]/g, '').trim();
                            if (priceText) price = priceText;
                        }
                    }

                    // Availability: prefer JSON-LD, fallback to DOM text
                    let availability = offers.availability || '';
                    if (!availability) {
                        const availEl = block.querySelector('[data-qaid="product_presence"]') ||
                                        block.querySelector('[data-qaid="availability"]');
                        if (availEl) {
                            const txt = availEl.innerText.trim().toLowerCase();
                            if (txt.includes('наявн')) availability = 'InStock';
                            else if (txt.includes('немає') || txt.includes('відсутн')) availability = 'OutOfStock';
                            else availability = availEl.innerText.trim();
                        }
                    }

                    results.push({
                        text: name,
                        href: link.href,
                        shop_name: shopName,
                        company_id: companyId,
                        price: price,
                        availability: availability,
                        sku: sku,
                        sales: sales,
                    });
                } catch(e) {}
            }
            return results;
        }
    """)

    return raw


def search_prom(query: str, max_pages: int = 3) -> list[Listing]:
    """Fetch product listings from prom.ua for the given search query."""
    listings: list[Listing] = []

    with sync_playwright() as p:
        _ensure_chromium_installed(p)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="uk-UA",
        )
        page = context.new_page()

        for page_num in range(1, max_pages + 1):
            url = PROM_SEARCH_URL.format(query=quote(query), page=page_num)

            for attempt in range(3):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    break
                except Exception:
                    if attempt == 2:
                        print(f"Page {page_num}: failed to load, skipping.")
                        continue
                    page.wait_for_timeout(2000)

            page.wait_for_timeout(2000)

            # Scroll to trigger lazy loading of all products
            for _ in range(8):
                page.evaluate("window.scrollBy(0, 800)")
                page.wait_for_timeout(400)
            page.wait_for_timeout(1000)

            items = _extract_items(page, query)

            _save_debug_screenshot(page, page_num)

            if not items:
                print(f"Page {page_num}: no products found, stopping.")
                break

            for item in items:
                listing_id = f"prom_{len(listings) + 1}"
                listings.append(Listing(
                    id=listing_id,
                    text=item["text"],
                    metadata={
                        "source": "prom.ua",
                        "url": item["href"],
                        "shop_name": item.get("shop_name", ""),
                        "company_id": item.get("company_id", ""),
                        "price": item.get("price", ""),
                        "availability": item.get("availability", ""),
                        "sku": item.get("sku", ""),
                        "sales": item.get("sales", ""),
                    },
                ))

            print(f"Page {page_num}: fetched {len(items)} products (total: {len(listings)})")
            time.sleep(1)

        browser.close()

    return listings


def fetch_and_save(query: str, max_pages: int = 3, output_path: Path | None = None) -> list[Listing]:
    """Search prom.ua, save results to JSON, return listings."""
    output_path = output_path or LISTINGS_PATH
    print(f"Searching prom.ua for: {query!r} ({max_pages} pages)...")

    listings = search_prom(query, max_pages=max_pages)

    if not listings:
        print("No products found.")
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": l.id, "text": l.text, "metadata": l.metadata} for l in listings]
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(listings)} products to {output_path}")
    return listings


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "генератор 10 кВт"
    listings = fetch_and_save(query, max_pages=3)
    print(f"\nFound {len(listings)} products:\n")
    for l in listings:
        print(f"[{l.id}] {l.text}")
        print(f"      {l.metadata.get('url', '')}")
        print()
