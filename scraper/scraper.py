"""
Rental listing scraper using OpenClaw + Gemini API.

Usage:
    python scraper.py                     # uses config.yaml defaults
    python scraper.py --config /path/to/config.yaml
    python scraper.py --url https://example.com --max-pages 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

try:
    import google.generativeai as genai  # type: ignore

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed – AI extraction disabled.")


def _build_gemini_model(model_name: str):
    """Initialise and return a Gemini generative model."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


EXTRACTION_PROMPT = """
You are an expert at extracting structured property listing data from HTML.

Extract ALL rental listings visible in the following HTML snippet and return a
JSON array where each element has these fields (use null if unavailable):

- title          : property name / headline (string)
- price          : monthly rent as a number (integer, HUF or local currency)
- location       : full address or district (string)
- size_m2        : floor area in square metres (number)
- num_rooms      : number of rooms/bedrooms (number)
- amenities      : list of features/amenities (array of strings)
- description    : full description text (string)
- image_urls     : list of image URLs (array of strings)
- contact_info   : phone / email / agent name (string)
- listing_url    : absolute URL of the listing detail page (string)

Return ONLY the JSON array, no markdown, no extra text.

HTML:
{html}
"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def extract_listings_with_gemini(html: str, model) -> list[dict[str, Any]]:
    """Send page HTML to Gemini and parse structured listing data."""
    prompt = EXTRACTION_PROMPT.format(html=html[:40_000])  # cap to avoid token limits
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip optional markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse Gemini response as JSON: {exc}")
        data = []

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    return data


# ---------------------------------------------------------------------------
# Fallback: BeautifulSoup heuristic extraction
# ---------------------------------------------------------------------------

from bs4 import BeautifulSoup  # noqa: E402


def extract_listings_heuristic(html: str, page_url: str) -> list[dict[str, Any]]:
    """
    Simple heuristic extraction used when Gemini is unavailable.
    Tries to find price, location, and size from common CSS patterns.
    """
    soup = BeautifulSoup(html, "lxml")
    listings: list[dict[str, Any]] = []

    # irisrent.hu uses article/div cards – adjust selectors as needed per site
    cards = (
        soup.select("div.listing-card")
        or soup.select("article.property-item")
        or soup.select("div.property-card")
        or soup.select("li.listing")
        or soup.select("div[class*='listing']")
    )

    for card in cards:
        title_el = card.select_one("h2, h3, .title, .listing-title")
        price_el = card.select_one(".price, .rent, [class*='price']")
        location_el = card.select_one(".address, .location, [class*='address']")
        size_el = card.select_one(".size, .area, [class*='size'], [class*='area']")
        link_el = card.select_one("a[href]")
        img_els = card.select("img[src]")

        listing: dict[str, Any] = {
            "title": title_el.get_text(strip=True) if title_el else None,
            "price": _parse_number(price_el.get_text(strip=True)) if price_el else None,
            "location": location_el.get_text(strip=True) if location_el else None,
            "size_m2": _parse_number(size_el.get_text(strip=True)) if size_el else None,
            "num_rooms": None,
            "amenities": [],
            "description": None,
            "image_urls": [img["src"] for img in img_els if img.get("src")],
            "contact_info": None,
            "listing_url": _make_absolute(
                link_el["href"] if link_el else None, page_url
            ),
        }
        listings.append(listing)

    return listings


def _parse_number(text: str) -> float | None:
    """Extract the first numeric value from a string."""
    match = re.search(r"[\d\s,.]+", text)
    if not match:
        return None
    raw = match.group().replace(" ", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _make_absolute(url: str | None, base: str) -> str | None:
    if not url:
        return None
    if url.startswith("http"):
        return url
    from urllib.parse import urljoin

    return urljoin(base, url)


# ---------------------------------------------------------------------------
# Browser / pagination
# ---------------------------------------------------------------------------

def _pagination_url(base_url: str, page: int) -> str:
    """
    Build a pagination URL. Supports common ?page=N and /page/N patterns.
    Override this function if the target site uses a different scheme.
    """
    if page <= 1:
        return base_url
    if "?" in base_url:
        return f"{base_url}&page={page}"
    return f"{base_url}?page={page}"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_page_html(page, url: str, timeout: int = 30) -> str:
    """Navigate to *url* with Playwright and return the rendered HTML."""
    try:
        page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=timeout * 1000)
    except PlaywrightTimeout:
        logger.warning(f"Timeout loading {url}, using partial content")
    return page.content()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "scraped_at",
    "title",
    "price",
    "location",
    "size_m2",
    "num_rooms",
    "amenities",
    "description",
    "image_urls",
    "contact_info",
    "listing_url",
]


def write_csv(listings: list[dict[str, Any]], output_path: Path, encoding: str = "utf-8") -> None:
    """Append listings to a CSV file, creating it with headers if needed."""
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding=encoding) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()
        for listing in listings:
            # Serialise list fields to JSON strings so CSV stays valid
            row = dict(listing)
            for key in ("amenities", "image_urls"):
                if isinstance(row.get(key), list):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            row["scraped_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            writer.writerow(row)
    logger.info(f"Wrote {len(listings)} listings to {output_path}")


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape(config: dict[str, Any]) -> None:
    scraper_cfg = config.get("scraper", {})
    gemini_cfg = config.get("gemini", {})
    output_cfg = config.get("output", {})

    target_urls: list[str] = scraper_cfg.get("target_urls", [])
    max_pages: int = scraper_cfg.get("max_pages", 5)
    request_delay: float = float(scraper_cfg.get("request_delay", 2.0))
    headless: bool = bool(scraper_cfg.get("headless", True))
    page_timeout: int = int(scraper_cfg.get("page_timeout", 30))

    model_name: str = gemini_cfg.get("model", "gemini-1.5-flash")

    out_dir = Path(os.environ.get("OUTPUT_DIR", output_cfg.get("directory", "./data")))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename: str = output_cfg.get("filename", "rental_listings")
    output_path = out_dir / f"{filename}_{timestamp}.csv"
    encoding: str = output_cfg.get("encoding", "utf-8")

    # Initialise Gemini model if available
    gemini_model = None
    if _GENAI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        try:
            gemini_model = _build_gemini_model(model_name)
            logger.info(f"Gemini model '{model_name}' initialised.")
        except Exception as exc:
            logger.warning(f"Could not initialise Gemini: {exc}. Falling back to heuristic extraction.")
    else:
        logger.info("Gemini not configured – using heuristic extraction.")

    total_listings: list[dict[str, Any]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        )
        pw_page = context.new_page()

        for base_url in target_urls:
            logger.info(f"Scraping: {base_url}")
            for page_num in range(1, max_pages + 1):
                url = _pagination_url(base_url, page_num)
                logger.info(f"  Page {page_num}: {url}")

                html = fetch_page_html(pw_page, url, timeout=page_timeout)

                if gemini_model:
                    try:
                        listings = extract_listings_with_gemini(html, gemini_model)
                    except Exception as exc:
                        logger.warning(f"Gemini extraction failed: {exc}. Using heuristic fallback.")
                        listings = extract_listings_heuristic(html, url)
                else:
                    listings = extract_listings_heuristic(html, url)

                if not listings:
                    logger.info(f"  No listings found on page {page_num} – stopping pagination.")
                    break

                logger.info(f"  Extracted {len(listings)} listings.")
                total_listings.extend(listings)

                if page_num < max_pages:
                    time.sleep(request_delay)

        browser.close()

    if total_listings:
        write_csv(total_listings, output_path, encoding=encoding)
        logger.success(f"Done. {len(total_listings)} listings saved to {output_path}")
    else:
        logger.warning("No listings were extracted. Check the target URL and selectors.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rental listing scraper")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: config.yaml next to this script)",
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="Override target URL(s) (can be repeated)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Override max pages to scrape",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.urls:
        config.setdefault("scraper", {})["target_urls"] = args.urls
    if args.max_pages is not None:
        config.setdefault("scraper", {})["max_pages"] = args.max_pages

    scrape(config)


if __name__ == "__main__":
    main()
