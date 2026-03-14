"""
ingestion/dynamic_scraper.py — Dynamic Company IR Scraper

Given any company name:
  1. Uses DuckDuckGo to find the Investor Relations page
  2. Scrapes the IR page for PDF links
  3. Downloads 3-5 most relevant documents
  4. Returns paths for indexing

No API key required — uses DuckDuckGo HTML search.
"""

import re
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote_plus
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Filing keywords to prioritize
PRIORITY_KEYWORDS = [
    "annual report", "annual-report", "integrated report",
    "quarterly results", "investor presentation", "earnings",
    "q1", "q2", "q3", "q4", "fy20", "fy21", "fy22", "fy23", "fy24",
    "concall", "analyst", "press release",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class DynamicIRScraper:
    """
    Fully dynamic IR scraper for any company worldwide.
    Uses DuckDuckGo HTML search — no API key required.
    """

    def __init__(self, company_name: str, max_pdfs: int = 4):
        self.company_name = company_name
        self.max_pdfs = max_pdfs
        self.slug = re.sub(r'[^a-z0-9]+', '_', company_name.lower()).strip('_')
        self.company_dir = DATA_DIR / self.slug
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.downloaded = []
        self.errors = []
        self._ensure_dirs()

    def _ensure_dirs(self):
        for folder in ["annual_report", "quarterly_briefing", "investor_presentation", "uploads"]:
            (self.company_dir / folder).mkdir(parents=True, exist_ok=True)

    # ── Step 1: DuckDuckGo search ──────────────────────────────────────────────
    def _ddg_search(self, query: str, max_results: int = 8) -> list[str]:
        """Search DuckDuckGo HTML and return result URLs."""
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        urls = []
        try:
            resp = self.session.get(search_url, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a.result__url"):
                href = a.get("href", "")
                if href and href.startswith("http"):
                    urls.append(href)
                    if len(urls) >= max_results:
                        break
            # Fallback: try result__a links
            if not urls:
                for a in soup.select("a.result__a"):
                    href = a.get("href", "")
                    if href and "duckduckgo" not in href and href.startswith("http"):
                        urls.append(href)
                        if len(urls) >= max_results:
                            break
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        logger.info(f"DDG search '{query}': {len(urls)} results")
        return urls

    # ── Step 2: Find IR page ───────────────────────────────────────────────────
    def find_ir_page(self) -> str | None:
        """Find the company's investor relations page via DuckDuckGo."""
        queries = [
            f"{self.company_name} investor relations annual report PDF",
            f"{self.company_name} IR page annual report download",
            f"{self.company_name} investors fillings PDF site:bseindia.com OR site:nseindia.com",
        ]

        ir_keywords = ["investor", "ir.", "/ir", "relations", "annual-report", "financials"]

        for query in queries:
            urls = self._ddg_search(query, max_results=10)
            for url in urls:
                url_lower = url.lower()
                # Direct PDF hit
                if url_lower.endswith(".pdf"):
                    return url
                # IR page hit
                if any(kw in url_lower for kw in ir_keywords):
                    logger.info(f"Found IR page: {url}")
                    return url
            time.sleep(1)

        # Return first result as fallback
        fallback = self._ddg_search(
            f"{self.company_name} annual report filetype:pdf", max_results=5
        )
        return fallback[0] if fallback else None

    # ── Step 3: Scrape page for PDF links ─────────────────────────────────────
    def scrape_pdfs_from_page(self, page_url: str) -> list[dict]:
        """Scrape a page and return scored PDF records."""
        if page_url.lower().endswith(".pdf"):
            return [{"url": page_url, "name": Path(page_url).name,
                     "filing_type": "annual_report", "score": 10}]

        try:
            resp = self.session.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning(f"Failed to scrape {page_url}: {e}")
            return []

        pdf_records = []
        seen_urls = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()

            if not href.lower().endswith(".pdf"):
                continue

            full_url = urljoin(page_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Score based on relevance keywords
            combined = (full_url + " " + text).lower()
            score = sum(2 if kw in combined else 0 for kw in PRIORITY_KEYWORDS)

            filing_type = self._classify_filing(combined)
            file_name = Path(urlparse(full_url).path).name or f"doc_{len(pdf_records)}.pdf"
            if not file_name.endswith(".pdf"):
                file_name += ".pdf"

            pdf_records.append({
                "url": full_url,
                "name": self._sanitize_filename(file_name),
                "filing_type": filing_type,
                "score": score,
            })

        # Sort by score descending
        pdf_records.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Found {len(pdf_records)} PDFs on {page_url}")
        return pdf_records

    # ── Also search directly for PDFs ─────────────────────────────────────────
    def search_direct_pdfs(self) -> list[dict]:
        """Use DDG to find direct PDF links."""
        queries = [
            f"{self.company_name} annual report 2023 2024 filetype:pdf",
            f"{self.company_name} quarterly results investor presentation pdf",
        ]
        records = []
        seen = set()

        for query in queries:
            urls = self._ddg_search(query, max_results=6)
            for url in urls:
                if url.lower().endswith(".pdf") and url not in seen:
                    seen.add(url)
                    combined = url.lower()
                    filing_type = self._classify_filing(combined)
                    score = sum(2 if kw in combined else 0 for kw in PRIORITY_KEYWORDS)
                    records.append({
                        "url": url,
                        "name": self._sanitize_filename(
                            Path(urlparse(url).path).name or f"doc_{len(records)}.pdf"
                        ),
                        "filing_type": filing_type,
                        "score": score + 5,  # Bonus for direct hits
                    })
            time.sleep(1)

        records.sort(key=lambda x: x["score"], reverse=True)
        return records

    # ── Step 4: Download PDF ───────────────────────────────────────────────────
    def _download_pdf(self, url: str, dest_path: Path) -> bool:
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            logger.info(f"  [SKIP] Already exists: {dest_path.name}")
            return True
        try:
            time.sleep(1.5)
            resp = self.session.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = dest_path.stat().st_size / 1024
            if size_kb < 5:
                dest_path.unlink()
                logger.warning(f"  [SKIP] Too small ({size_kb:.1f}KB): {dest_path.name}")
                return False
            logger.info(f"  [OK] {dest_path.name} ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            logger.error(f"  [FAIL] {url}: {e}")
            self.errors.append(str(e))
            if dest_path.exists():
                dest_path.unlink()
            return False

    # ── Main Entry ─────────────────────────────────────────────────────────────
    def run(self, progress_callback=None) -> dict:
        """
        Full pipeline: search → scrape → download.
        progress_callback(message: str) called at each step for UI updates.
        """
        def _progress(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _progress(f"🔍 Searching for {self.company_name} investor relations page...")

        # Combine IR page scraping + direct PDF search
        all_records = []

        # Strategy 1: Find IR page and scrape it
        ir_page = self.find_ir_page()
        if ir_page:
            _progress(f"📄 Found IR page: {ir_page[:80]}...")
            page_records = self.scrape_pdfs_from_page(ir_page)
            all_records.extend(page_records)

        # Strategy 2: Direct PDF search
        _progress("🔎 Searching for direct PDF links...")
        direct_records = self.search_direct_pdfs()
        all_records.extend(direct_records)

        # Deduplicate and sort
        seen_urls = set()
        unique_records = []
        for r in sorted(all_records, key=lambda x: x["score"], reverse=True):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_records.append(r)

        if not unique_records:
            _progress("⚠️ No PDFs found. Try uploading the document manually.")
            return {"company": self.company_name, "files": [], "errors": ["No PDFs found"]}

        _progress(f"📥 Found {len(unique_records)} PDFs. Downloading top {self.max_pdfs}...")

        # Download top N
        for record in unique_records[:self.max_pdfs]:
            filing_type = record["filing_type"]
            dest_path = self.company_dir / filing_type / record["name"]
            _progress(f"  Downloading: {record['name']}")
            success = self._download_pdf(record["url"], dest_path)
            if success:
                self.downloaded.append({
                    "company": self.company_name,
                    "slug": self.slug,
                    "filing_type": filing_type,
                    "filename": record["name"],
                    "path": str(dest_path),
                })

        _progress(f"✅ Downloaded {len(self.downloaded)} files for {self.company_name}")
        return {
            "company": self.company_name,
            "slug": self.slug,
            "files": self.downloaded,
            "errors": self.errors,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_filing(text: str) -> str:
        if any(kw in text for kw in ["annual", "integrated", "fy20", "fy21", "fy22", "fy23", "fy24"]):
            return "annual_report"
        if any(kw in text for kw in ["q1", "q2", "q3", "q4", "quarterly", "results", "earnings"]):
            return "quarterly_briefing"
        if any(kw in text for kw in ["investor", "presentation", "analyst"]):
            return "investor_presentation"
        return "annual_report"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        return name[:200] if len(name) > 200 else name
