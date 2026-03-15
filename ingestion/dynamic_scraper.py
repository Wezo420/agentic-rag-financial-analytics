"""
ingestion/dynamic_scraper.py — Dynamic Company IR Scraper (v3)

Strategy (in order of reliability):
  1. BSE India public API  → most reliable for Indian listed companies
  2. NSE India filings     → fallback for NSE-listed companies
  3. DuckDuckGo PDF search → fallback for global companies
  4. Direct IR page scrape → last resort
"""

import re
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.bseindia.com/",
}

PRIORITY_KEYWORDS = [
    "annual", "annual-report", "integrated", "quarterly",
    "q1", "q2", "q3", "q4", "fy20", "fy21", "fy22", "fy23", "fy24",
    "investor", "presentation", "earnings", "results",
]


class DynamicIRScraper:
    """Dynamic IR scraper — tries BSE/NSE APIs first, then web search."""

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

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 1: BSE India API
    # ══════════════════════════════════════════════════════════════════
    def _bse_search_company(self) -> str | None:
        try:
            url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active"
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            # Handle both list and dict responses
            companies = data if isinstance(data, list) else data.get("Table", [])
            name_lower = self.company_name.lower()
            for co in companies:
                co_name = co.get("Issuer_Name", "").lower()
                if name_lower in co_name or co_name in name_lower:
                    code = co.get("SCRIP_CD", "")
                    logger.info(f"BSE match: {co.get('Issuer_Name')} -> {code}")
                    return str(code)
        except Exception as e:
            logger.warning(f"BSE company search failed: {e}")
        return None

    def _bse_get_annual_reports(self, scrip_code: str) -> list:
        """Get annual report PDFs from BSE for a given scrip code."""
        records = []
        try:
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
                f"?scripcode={scrip_code}&type=AR"
            )
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            for item in data.get("Table", [])[:self.max_pdfs]:
                pdf_url = item.get("PDFLINKANNUALREPORT", "")
                if not pdf_url:
                    continue
                if not pdf_url.startswith("http"):
                    pdf_url = "https://www.bseindia.com" + pdf_url
                year = item.get("PERIOD", "")
                name = f"{self.slug}_annual_report_{year}.pdf"
                records.append({
                    "url": pdf_url,
                    "name": self._sanitize_filename(name),
                    "filing_type": "annual_report",
                    "score": 20,
                })
        except Exception as e:
            logger.warning(f"BSE annual reports failed: {e}")
        return records

    def _bse_get_filings(self, scrip_code: str) -> list:
        """Get recent investor presentations and quarterly reports from BSE."""
        records = []
        try:
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
                f"?scripcode={scrip_code}&type=IP"
            )
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            for item in data.get("Table", [])[:3]:
                pdf_url = item.get("PDFLINKANNUALREPORT", "")
                if not pdf_url:
                    continue
                if not pdf_url.startswith("http"):
                    pdf_url = "https://www.bseindia.com" + pdf_url
                name = f"{self.slug}_presentation_{item.get('PERIOD','')}.pdf"
                records.append({
                    "url": pdf_url,
                    "name": self._sanitize_filename(name),
                    "filing_type": "investor_presentation",
                    "score": 15,
                })
        except Exception as e:
            logger.warning(f"BSE filings failed: {e}")

        # BSE announcements
        try:
            url2 = (
                f"https://api.bseindia.com/BseIndiaAPI/api/Announcement/w"
                f"?strCat=Result&strPrevDate=&strScrip={scrip_code}"
                f"&strSearch=P&strToDate=&strType=C&subcategory=-1"
            )
            resp2 = self.session.get(url2, timeout=10)
            data2 = resp2.json()
            for item in data2.get("Table", [])[:3]:
                pdf_name = item.get("ATTACHMENTNAME", "")
                if pdf_name:
                    pdf_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{pdf_name}"
                    records.append({
                        "url": pdf_url,
                        "name": self._sanitize_filename(pdf_name),
                        "filing_type": "quarterly_briefing",
                        "score": 12,
                    })
        except Exception as e:
            logger.warning(f"BSE announcements failed: {e}")
        return records

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 2: NSE India
    # ══════════════════════════════════════════════════════════════════
    def _nse_search_and_get(self) -> list:
        """Search NSE for the company and get annual report links."""
        records = []
        try:
            nse_headers = {**HEADERS, "Referer": "https://www.nseindia.com/"}
            self.session.get("https://www.nseindia.com", timeout=10, headers=nse_headers)
            time.sleep(1)
            search_url = f"https://www.nseindia.com/api/search-autocomplete?q={quote(self.company_name)}"
            resp = self.session.get(search_url, timeout=10, headers=nse_headers)
            data = resp.json()
            symbols = data.get("symbols", [])
            if symbols:
                symbol = symbols[0].get("symbol", "")
                logger.info(f"NSE symbol: {symbol}")
                ar_url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
                ar_resp = self.session.get(ar_url, timeout=10, headers=nse_headers)
                ar_data = ar_resp.json()
                for item in ar_data.get("data", [])[:self.max_pdfs]:
                    pdf_url = item.get("fileName", "")
                    if pdf_url:
                        year = item.get("year", "")
                        name = f"{self.slug}_nse_annual_{year}.pdf"
                        records.append({
                            "url": pdf_url,
                            "name": self._sanitize_filename(name),
                            "filing_type": "annual_report",
                            "score": 18,
                        })
        except Exception as e:
            logger.warning(f"NSE search failed: {e}")
        return records

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 3: DuckDuckGo PDF Search
    # ══════════════════════════════════════════════════════════════════
    def _ddg_pdf_search(self) -> list:
        """Use duckduckgo_search to find PDFs directly."""
        records = []
        queries = [
            f"{self.company_name} annual report 2023 2024 filetype:pdf",
            f"{self.company_name} annual report PDF download",
            f"{self.company_name} investor presentation quarterly results PDF",
        ]
        seen = set()
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            for query in queries:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=6))
                    for r in results:
                        url = r.get("href", "")
                        if url and url not in seen and url.lower().endswith(".pdf"):
                            seen.add(url)
                            combined = (url + " " + r.get("title", "")).lower()
                            file_name = Path(urlparse(url).path).name or "document.pdf"
                            records.append({
                                "url": url,
                                "name": self._sanitize_filename(file_name),
                                "filing_type": self._classify_filing(combined),
                                "score": sum(2 for kw in PRIORITY_KEYWORDS if kw in combined) + 5,
                            })
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"DDG query failed: {e}")
        except ImportError:
            logger.warning("duckduckgo_search not installed")
        records.sort(key=lambda x: x["score"], reverse=True)
        return records

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 4: Direct IR page scrape
    # ══════════════════════════════════════════════════════════════════
    def _scrape_ir_page(self) -> list:
        """Try to find and scrape the company IR page directly."""
        records = []
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"{self.company_name} investor relations annual report",
                    max_results=5
                ))
            ir_keywords = ["investor", "/ir", "ir.", "relations", "annual"]
            ir_url = None
            for r in results:
                url = r.get("href", "")
                if any(kw in url.lower() for kw in ir_keywords):
                    ir_url = url
                    break
            if not ir_url and results:
                ir_url = results[0].get("href")
            if ir_url:
                logger.info(f"Scraping IR page: {ir_url}")
                resp = self.session.get(ir_url, timeout=15)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                seen = set()
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if not href.lower().endswith(".pdf"):
                        continue
                    full_url = urljoin(ir_url, href)
                    if full_url in seen:
                        continue
                    seen.add(full_url)
                    combined = (full_url + " " + a.get_text()).lower()
                    file_name = Path(urlparse(full_url).path).name or "doc.pdf"
                    records.append({
                        "url": full_url,
                        "name": self._sanitize_filename(file_name),
                        "filing_type": self._classify_filing(combined),
                        "score": sum(2 for kw in PRIORITY_KEYWORDS if kw in combined),
                    })
        except Exception as e:
            logger.warning(f"IR page scrape failed: {e}")
        records.sort(key=lambda x: x["score"], reverse=True)
        return records

    # ══════════════════════════════════════════════════════════════════
    # Download
    # ══════════════════════════════════════════════════════════════════
    def _download_pdf(self, url: str, dest_path: Path) -> bool:
        if dest_path.exists() and dest_path.stat().st_size > 5000:
            logger.info(f"  [SKIP] Already exists: {dest_path.name}")
            return True
        try:
            time.sleep(1.2)
            resp = self.session.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type and "pdf" not in content_type:
                logger.warning(f"  [SKIP] HTML response, not a PDF")
                return False
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = dest_path.stat().st_size / 1024
            if size_kb < 5:
                dest_path.unlink()
                logger.warning(f"  [SKIP] Too small ({size_kb:.1f}KB)")
                return False
            logger.info(f"  [OK] {dest_path.name} ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            logger.error(f"  [FAIL] {url}: {e}")
            self.errors.append(str(e))
            if dest_path.exists():
                dest_path.unlink()
            return False

    # ══════════════════════════════════════════════════════════════════
    # Main Entry
    # ══════════════════════════════════════════════════════════════════
    def run(self, progress_callback=None) -> dict:
        def _progress(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _progress(f"Searching for {self.company_name} filings...")
        all_records = []

        # Strategy 1: BSE
        _progress("Trying BSE India API...")
        bse_code = self._bse_search_company()
        if bse_code:
            _progress(f"Found on BSE: {bse_code}")
            bse_records = self._bse_get_annual_reports(bse_code)
            bse_records += self._bse_get_filings(bse_code)
            _progress(f"Found {len(bse_records)} BSE filings")
            all_records.extend(bse_records)

        # Strategy 2: NSE
        if len(all_records) < 2:
            _progress("Trying NSE India API...")
            nse_records = self._nse_search_and_get()
            _progress(f"Found {len(nse_records)} NSE filings")
            all_records.extend(nse_records)

        # Strategy 3: DDG
        if len(all_records) < 2:
            _progress("Searching web for PDF documents...")
            ddg_records = self._ddg_pdf_search()
            _progress(f"Found {len(ddg_records)} PDFs via web search")
            all_records.extend(ddg_records)

        # Strategy 4: IR page scrape
        if len(all_records) < 2:
            _progress("Scraping IR page directly...")
            ir_records = self._scrape_ir_page()
            _progress(f"Found {len(ir_records)} PDFs on IR page")
            all_records.extend(ir_records)

        # Deduplicate
        seen_urls = set()
        unique_records = []
        for r in sorted(all_records, key=lambda x: x["score"], reverse=True):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_records.append(r)

        if not unique_records:
            _progress("No PDFs found via any method.")
            return {"company": self.company_name, "files": [], "errors": ["No PDFs found"]}

        _progress(f"Downloading {min(self.max_pdfs, len(unique_records))} documents...")

        for record in unique_records[:self.max_pdfs]:
            dest_path = self.company_dir / record["filing_type"] / record["name"]
            _progress(f"Downloading: {record['name'][:60]}")
            success = self._download_pdf(record["url"], dest_path)
            if success:
                self.downloaded.append({
                    "company": self.company_name,
                    "slug": self.slug,
                    "filing_type": record["filing_type"],
                    "filename": record["name"],
                    "path": str(dest_path),
                })

        if self.downloaded:
            _progress(f"Downloaded {len(self.downloaded)} files successfully")
        else:
            _progress("All downloads failed — site may require login or block bots.")

        return {
            "company": self.company_name,
            "slug": self.slug,
            "files": self.downloaded,
            "errors": self.errors,
        }

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
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name[:200]
