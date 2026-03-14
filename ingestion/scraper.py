"""
ingestion/scraper.py — Automated IR Page Scraper & PDF Downloader

Handles:
  1. Web scraping of Investor Relations pages (BeautifulSoup)
  2. PDF link discovery & download
  3. Structured folder organization per company / filing type
  4. Simulated downloader mode (for demo/offline use)

Usage:
    from ingestion.scraper import IRScraper
    scraper = IRScraper(company_name="Tata Motors")
    scraper.run()
"""

import os
import re
import time
import logging
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# ── Import project config ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, COMPANY_REGISTRY, LOG_LEVEL

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ─── Simulated PDF Registry ───────────────────────────────────────────────────
# When live scraping is unavailable (firewall / demo environment), these sample
# publicly-accessible PDFs are used to simulate real IR downloads.
SIMULATED_PDFS: dict[str, list[dict]] = {
    "tata_motors": [
        {
            "name": "Tata_Motors_Annual_Report_FY2023.pdf",
            "url": "https://www.tatamotors.com/wp-content/uploads/2023/07/tatamotors-integrated-annual-report-2022-23.pdf",
            "filing_type": "annual_report",
            "year": 2023,
        },
        {
            "name": "Tata_Motors_Q4FY24_Earnings_Presentation.pdf",
            "url": "https://www.tatamotors.com/wp-content/uploads/2024/05/q4-fy2024-earnings-presentation.pdf",
            "filing_type": "quarterly_briefing",
            "year": 2024,
        },
    ],
    "hul": [
        {
            "name": "HUL_Annual_Report_FY2023.pdf",
            "url": "https://www.hul.co.in/files/origin/e8f2d4c3-7dc1-4f5f-8eb4-f7d2a1e0e123/hul-annual-report-2022-23.pdf",
            "filing_type": "annual_report",
            "year": 2023,
        },
        {
            "name": "HUL_Q4FY24_Results_Presentation.pdf",
            "url": "https://www.hul.co.in/files/origin/results-q4-fy2024.pdf",
            "filing_type": "quarterly_briefing",
            "year": 2024,
        },
    ],
    "reliance": [
        {
            "name": "Reliance_Industries_Annual_Report_FY2023.pdf",
            "url": "https://www.ril.com/DownloadFiles/IRDownloads/Annual_Report_2022-23.pdf",
            "filing_type": "annual_report",
            "year": 2023,
        },
        {
            "name": "Reliance_Q4FY24_Press_Release.pdf",
            "url": "https://www.ril.com/DownloadFiles/IRDownloads/Press_Release_Q4FY24.pdf",
            "filing_type": "quarterly_briefing",
            "year": 2024,
        },
    ],
}

# PDF keyword signals used to classify discovered links
PDF_FILING_KEYWORDS = {
    "annual_report": ["annual", "annual-report", "integrated-report", "fy20"],
    "quarterly_briefing": ["quarterly", "q1", "q2", "q3", "q4", "earnings", "results", "presentation"],
    "investor_presentation": ["investor", "presentation", "investor-day"],
    "concall": ["concall", "con-call", "transcript", "analyst-call"],
}

# ─── Scraper Class ─────────────────────────────────────────────────────────────
class IRScraper:
    """
    Investor Relations Scraper.

    Attempts live scraping first; falls back to a curated simulated set
    for demonstration / CI environments.
    """

    def __init__(
        self,
        company_name: str,
        use_simulation: bool = False,
        delay_between_requests: float = 1.5,
    ):
        if company_name not in COMPANY_REGISTRY:
            raise ValueError(
                f"Unknown company '{company_name}'. "
                f"Available: {list(COMPANY_REGISTRY.keys())}"
            )

        self.company_name = company_name
        self.meta = COMPANY_REGISTRY[company_name]
        self.slug = self.meta["slug"]
        self.ir_url = self.meta["ir_base_url"]
        self.use_simulation = use_simulation
        self.delay = delay_between_requests
        self.session = self._build_session()
        self.company_dir = DATA_DIR / self.slug
        self._ensure_folder_structure()
        self.downloaded: list[dict] = []
        self.errors: list[str] = []

    # ── Folder Structure ───────────────────────────────────────────────────────
    def _ensure_folder_structure(self):
        """Create <company>/annual_report, quarterly_briefing, etc. folders."""
        for folder in ["annual_report", "quarterly_briefing", "investor_presentation", "concall", "uploads"]:
            (self.company_dir / folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"[{self.slug}] Directory structure ready at {self.company_dir}")

    # ── HTTP Session ───────────────────────────────────────────────────────────
    @staticmethod
    def _build_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        s.max_redirects = 5
        return s

    # ── Filing Type Classifier ─────────────────────────────────────────────────
    @staticmethod
    def _classify_filing(url: str, link_text: str) -> str:
        combined = (url + " " + link_text).lower()
        for filing_type, keywords in PDF_FILING_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return filing_type
        return "misc"

    # ── Live Scraper ───────────────────────────────────────────────────────────
    def _scrape_live(self) -> list[dict]:
        """Fetch IR page and extract all PDF links."""
        logger.info(f"[{self.slug}] Fetching IR page: {self.ir_url}")
        try:
            resp = self.session.get(self.ir_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"[{self.slug}] Live scrape failed: {e}. Falling back to simulation.")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        discovered: list[dict] = []

        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            text: str = a_tag.get_text(strip=True)

            # Only process PDF links
            if not href.lower().endswith(".pdf"):
                continue

            # Resolve relative URLs
            full_url = urljoin(self.ir_url, href)
            filing_type = self._classify_filing(full_url, text)
            file_name = Path(urlparse(full_url).path).name or f"doc_{hashlib.md5(full_url.encode()).hexdigest()[:8]}.pdf"

            discovered.append({
                "name": file_name,
                "url": full_url,
                "filing_type": filing_type,
                "year": self._extract_year(full_url + text),
            })

        logger.info(f"[{self.slug}] Discovered {len(discovered)} PDFs from live scrape.")
        return discovered

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Extract a 4-digit year between 2010 and 2030 from text."""
        matches = re.findall(r"20[1-3]\d", text)
        return int(matches[-1]) if matches else datetime.now().year

    # ── PDF Downloader ─────────────────────────────────────────────────────────
    def _download_pdf(self, url: str, dest_path: Path) -> bool:
        """Stream-download a PDF to dest_path. Returns True on success."""
        if dest_path.exists():
            logger.info(f"  [SKIP] Already downloaded: {dest_path.name}")
            return True

        try:
            time.sleep(self.delay)
            resp = self.session.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(f"  [WARN] Non-PDF content-type for {url}: {content_type}")

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = dest_path.stat().st_size / 1024
            logger.info(f"  [OK] {dest_path.name} ({size_kb:.1f} KB)")
            return True

        except Exception as e:
            logger.error(f"  [FAIL] {url}: {e}")
            self.errors.append(str(e))
            if dest_path.exists():
                dest_path.unlink()
            return False

    # ── Simulate Download (create placeholder) ────────────────────────────────
    def _simulate_download(self, file_record: dict) -> bool:
        """
        For demo environments: creates a minimal valid placeholder PDF
        using reportlab if available, else writes a text stub.
        """
        filing_type = file_record.get("filing_type", "misc")
        dest_path = self.company_dir / filing_type / file_record["name"]

        if dest_path.exists():
            logger.info(f"  [SKIP] Already exists: {dest_path.name}")
            return True

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(str(dest_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph(
                f"<b>{self.company_name} — {file_record['name'].replace('.pdf','').replace('_',' ')}</b>",
                styles["Title"]
            ))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Fiscal Year: {file_record.get('year', 'N/A')} | Filing Type: {filing_type.replace('_',' ').title()}",
                styles["Normal"]
            ))
            story.append(Spacer(1, 20))

            # Simulated content based on company & filing type
            content_blocks = _get_simulated_content(self.company_name, filing_type)
            for section_title, section_body in content_blocks:
                story.append(Paragraph(f"<b>{section_title}</b>", styles["Heading2"]))
                story.append(Spacer(1, 6))
                story.append(Paragraph(section_body, styles["Normal"]))
                story.append(Spacer(1, 14))

            doc.build(story)
            logger.info(f"  [SIM] Created placeholder PDF: {dest_path.name}")
            return True

        except ImportError:
            # reportlab not installed → plain text stub
            dest_path.write_text(
                f"[SIMULATED FILING]\n"
                f"Company: {self.company_name}\n"
                f"Document: {file_record['name']}\n"
                f"Filing Type: {filing_type}\n"
                f"Year: {file_record.get('year', 'N/A')}\n\n"
                f"[This is a placeholder document for demonstration purposes.]\n"
            )
            return True

    # ── Main Entry Point ───────────────────────────────────────────────────────
    def run(self) -> dict:
        """
        Orchestrate scraping + downloading.
        Returns a summary dict with counts and paths.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  Starting ingestion for: {self.company_name}")
        logger.info(f"{'='*60}")

        # Step 1: Discover PDFs (live or simulated)
        if self.use_simulation:
            file_records = SIMULATED_PDFS.get(self.slug, [])
            logger.info(f"[{self.slug}] Using simulation mode ({len(file_records)} records).")
        else:
            file_records = self._scrape_live()
            if not file_records:
                logger.info(f"[{self.slug}] No live PDFs found. Falling back to simulation.")
                file_records = SIMULATED_PDFS.get(self.slug, [])

        # Step 2: Download each PDF
        for record in file_records:
            filing_type = record.get("filing_type", "misc")
            dest_path = self.company_dir / filing_type / record["name"]

            if self.use_simulation:
                success = self._simulate_download(record)
            else:
                success = self._download_pdf(record["url"], dest_path)

            if success:
                self.downloaded.append({
                    "company": self.company_name,
                    "slug": self.slug,
                    "filing_type": filing_type,
                    "year": record.get("year"),
                    "filename": record["name"],
                    "path": str(dest_path),
                })

        # Step 3: Write manifest
        manifest_path = self.company_dir / "manifest.json"
        import json
        manifest = {
            "company": self.company_name,
            "slug": self.slug,
            "last_updated": datetime.now().isoformat(),
            "files": self.downloaded,
            "errors": self.errors,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(
            f"\n[{self.slug}] Ingestion complete. "
            f"Downloaded: {len(self.downloaded)} | Errors: {len(self.errors)}"
        )
        return manifest


# ─── Simulated Financial Content Generator ────────────────────────────────────
def _get_simulated_content(company: str, filing_type: str) -> list[tuple[str, str]]:
    """Returns rich simulated financial narrative blocks for PDF generation."""

    company_profiles = {
        "Tata Motors": {
            "annual_report": [
                ("Executive Chairman's Message",
                 "FY2023 has been a landmark year for Tata Motors. Our consolidated revenue grew by 24% YoY to ₹4,38,536 Crore, driven by strong performance across both JLR and domestic commercial vehicle segments. The EBITDA margin expanded by 220 bps to 12.4%, reflecting operational efficiency gains and favorable commodity costs. Our net debt reduced by ₹28,000 Crore to ₹97,622 Crore, marking a decisive step toward our 'Debt-Free' aspiration."),
                ("Business Segment Performance",
                 "Jaguar Land Rover (JLR) delivered revenues of £23.1 billion (up 27% YoY), with EBIT margin of 6.7%. Defender and Range Rover families continued to drive premium mix. Domestic CV segment reported its best-ever performance with market share of 44.6% in MHCV. Passenger Vehicle business turned profitable with EBIT of +1.2%, supported by Nexon EV and Punch volumes. Total EV sales crossed 75,000 units domestically, up 75% YoY."),
                ("Key Financial Metrics (Consolidated)",
                 "Revenue from Operations: ₹4,38,536 Cr (+24% YoY). EBITDA: ₹54,379 Cr (Margin: 12.4%). EBIT: ₹21,430 Cr. PAT (after minority): ₹2,415 Cr (vs loss of ₹11,441 Cr in FY22). Net Debt: ₹97,622 Cr (-22% YoY). ROCE: 8.2% (vs -3.1% FY22). Free Cash Flow: +₹9,600 Cr."),
                ("Credit Risk Assessment & Debt Profile",
                 "The company carries ₹97,622 Crore of net debt, with JLR contributing ~₹60,000 Crore (in INR equivalent). The debt maturity profile is well-distributed with no major near-term refinancing cliff. JLR maintains investment-grade credit ratings: Moody's B1 (stable), S&P B+ (stable). Domestic entity credit rating: CRISIL AA-. Key credit risks include semiconductor supply chain disruptions, EV transition capex (~£15 billion REFOCUS plan), and GBP/USD currency exposure."),
                ("Expansion Plans — Retail & EV Infrastructure",
                 "Tata Motors plans to expand its domestic EV touchpoints to 1,000+ by FY25 (from 650 in FY23). JLR is accelerating its pure-EV platform ('JEA') with first vehicles expected in CY2025. ₹15,000 Crore earmarked for Pune and Sanand plant upgrades. Retail outlet expansion of 400+ new touchpoints across Tier-2/3 cities. The CAPEX guidance for FY24 stands at ₹35,000 Crore (consolidated), primarily towards EV battery supply chain and JLR electrification."),
                ("Risk Factors",
                 "1. Global macroeconomic slowdown impacting premium vehicle demand. 2. GBP appreciation reducing JLR margins on reporting. 3. EV price wars accelerating competitive intensity in India. 4. Raw material (lithium, cobalt, steel) price volatility impacting battery cost. 5. Regulatory changes in EU emission norms (Euro 7). 6. Execution risk on debt-reduction targets. 7. Labor disruptions at Solihull and Castle Bromwich plants."),
            ],
            "quarterly_briefing": [
                ("Q4 FY2024 Financial Highlights",
                 "Revenue: ₹1,19,986 Crore (+13% YoY). EBITDA: ₹16,524 Crore (Margin: 13.8%). PBT: ₹6,975 Crore. PAT: ₹17,528 Crore (includes one-time deferred tax asset recognition of ₹9,000 Crore). JLR EBIT margin: 8.5%. India PV market share: 13.9%."),
                ("JLR Q4 Performance",
                 "JLR wholesales: 1,03,593 vehicles (flat QoQ, +11% YoY). Revenue: £7.3 billion. Range Rover / Defender mix at 58%. Retail orders: 1,51,000 units (strong 3.3 months of forward order cover). Free cash flow: +£0.7 billion. Net debt: £1.0 billion (target: zero debt by FY26)."),
                ("Guidance FY25",
                 "JLR targets revenue of £30 billion+ with EBIT margin >8%. India business targets to sustain profitability. Net debt reduction to ₹70,000 Crore by FY25. EV volume target: 1,00,000 units domestically."),
            ],
        },
        "Hindustan Unilever (HUL)": {
            "annual_report": [
                ("Chairman & MD's Statement",
                 "FY2023 has been a year of resilient growth for HUL amid a challenging macro environment. We delivered underlying volume growth of 4% and underlying sales growth of 16%, demonstrating the strength of our portfolio. Our EBITDA margins improved to 23.2% (up 130 bps), aided by judicious pricing and cost-efficiency programs. Our 'Winning in Many Indias' strategy continues to deepen market penetration in rural and semi-urban segments."),
                ("Financial Performance Summary",
                 "Turnover: ₹58,154 Crore (+16% YoY). EBITDA: ₹13,491 Crore (Margin: 23.2%). PAT: ₹9,962 Crore (+4% YoY). EPS: ₹42.35. ROCE: 108.7%. Dividend per share: ₹34 (payout ratio: 80%). Net cash position: ₹2,416 Crore (debt-free with net cash surplus). Market Cap: ₹5,85,000 Crore (as of March 2023)."),
                ("Segment-Wise Revenue Breakup",
                 "Home Care: ₹20,308 Cr (35% of turnover, +22% YoY). Beauty & Personal Care: ₹21,236 Cr (36%, +12% YoY). Foods & Refreshment: ₹14,854 Cr (26%, +18% YoY). Others: ₹1,756 Cr. Underlying volume growth: +4% (Home Care +2%, BPC +5%, F&R +5%). Rural growth lagged urban for 3 consecutive quarters due to rural distress."),
                ("Distribution & Retail Expansion",
                 "HUL reaches 9 million+ retail outlets directly (up from 8.5 million in FY22). D2C platforms (U-Shop, partner e-commerce) now contributing ~15% of BPC revenues. E-commerce grew 25% YoY, contributing ~14% of business. MT (Modern Trade) share at 22%. Quick commerce partnerships with Blinkit, Zepto, and Swiggy Instamart expanding. 3 new manufacturing lines commissioned at Pondicherry and Silvassa for premium skincare."),
                ("Credit & Liquidity Risk Profile",
                 "HUL maintains AAA/Stable credit rating from CRISIL and ICRA. The company is effectively debt-free with ₹2,416 Crore net cash. Trade receivables DSO: 7.8 days (industry-best). Working capital cycle: Negative (company gets paid before paying suppliers). Key risks: commodity (palm oil, crude derivatives) price volatility; regulatory risks (price controls under ESMA); rural demand softness; competitive intensity from regional players."),
                ("ESG & Sustainability",
                 "Carbon-neutral manufacturing operations achieved at 14 of 30 sites. Plastic waste neutrality maintained for 5th consecutive year. Women in management: 43%. Unilever Compass targets on track. 100% renewable electricity sourced for manufacturing by FY25 commitment in progress."),
            ],
            "quarterly_briefing": [
                ("Q4 FY2024 Results",
                 "Revenue: ₹15,038 Crore (+1% YoY — moderated growth due to price cuts in detergent category). EBITDA: ₹3,554 Crore (Margin: 23.6%). PAT: ₹2,552 Crore (+8% YoY). Volume growth: +4%. Price growth: -3% (deflationary environment in HPC). Dividend declared: ₹24/share (special dividend)."),
            ],
        },
        "Reliance Industries": {
            "annual_report": [
                ("Chairman's Review",
                 "FY2023 was a year of record performance for Reliance Industries. Consolidated revenues crossed ₹9,00,000 Crore for the first time, reaching ₹9,74,864 Crore (+24.8% YoY). EBITDA reached ₹1,53,920 Crore, a new high. Our New Commerce initiative and Jio Platforms together now serve over 450 million digital subscribers and 18,040 stores under the New Commerce platform."),
                ("Segment Performance Deep-Dive",
                 "O2C (Oil-to-Chemicals): Revenue ₹5,97,403 Cr. EBITDA ₹60,827 Cr (Margin: 10.2%). Petchem margins compressed due to China demand weakness. Oil & Gas (E&P): Revenue ₹21,480 Cr. EBITDA ₹17,240 Cr (Margin: 80.3%). KG-D6 MJ field plateau production achieved. Retail: Revenue ₹2,60,364 Cr (+19% YoY). EBITDA ₹22,318 Cr. Jio Platforms: Revenue ₹95,993 Cr. EBITDA ₹44,768 Cr (Margin: 46.6%)."),
                ("Consolidated Financial Metrics",
                 "Revenue: ₹9,74,864 Crore. EBITDA: ₹1,53,920 Crore (Margin: 15.8%). Net Profit: ₹73,670 Crore (+11% YoY). EPS: ₹109. Net Debt: ₹1,09,234 Crore (Debt/EBITDA: 0.71x — very comfortable). Capital Employed: ₹10,28,400 Crore. ROCE: 10.9%. FCF: +₹57,840 Crore."),
                ("Retail Expansion Strategy & Credit Implications",
                 "Reliance Retail targets 50,000+ stores by FY26 (currently ~18,040 stores). Categories: JioMart (grocery), Trends (fashion), Smart (electronics), Jewels. B2B New Commerce platform serves 3 million+ merchant partners. CAPEX in retail: ₹55,000 Crore over FY23-25. Credit implication: Retail EBITDA expected to cross ₹40,000 Crore by FY25. Net debt in retail vertical remains manageable at ₹22,000 Crore given strong cash generation."),
                ("5G & Jio Platforms — Growth Outlook",
                 "Jio 5G deployed across 2,300+ cities in FY23 (record pace globally). 5G subscriber base: 25 million (target: 100 million by FY25). ARPU: ₹178.8/month (+12% YoY). JioBharat phone driving feature-phone-to-smartphone upgrades in Bharat. True5G network capex: ₹2,00,000 Crore over FY22-25. Monetization expected through enterprise 5G (smart factories, healthcare)."),
                ("Risk Factors",
                 "1. Refining margin cycle risk (GRM sensitivity: $1/bbl = ~₹4,500 Cr EBITDA). 2. USD/INR currency risk on O2C revenues. 3. Regulatory risk in telecom (TRAI interconnect, spectrum fee disputes). 4. Petchem demand risk from China slowdown. 5. Retail competition from Amazon, Meesho. 6. Large capex commitments (₹3,00,000 Cr over FY22-25) elevating short-term leverage. 7. Governance and succession risk (transition planning)."),
            ],
            "quarterly_briefing": [
                ("Q4 FY2024 Standalone & Consolidated Results",
                 "Q4FY24 Consolidated Revenue: ₹2,40,892 Crore. EBITDA: ₹42,516 Crore (+12.6% YoY). PAT: ₹18,951 Crore (+7.3% YoY). Retail Revenue: ₹75,615 Crore. Jio EBITDA: ₹14,543 Crore. O2C EBITDA: ₹15,439 Crore. Net Debt reduction: ₹10,000 Crore QoQ to ₹99,000 Crore."),
                ("Strategic Announcements",
                 "Reliance Retail to list by FY26 (implied valuation: $100-120 billion). Jio Financial Services demerger completed; NBFC-license application filed. New Green Energy Gigafactories at Jamnagar: Phase 1 commissioning (15 GW solar PV) expected by FY25. Strategic partnership with Saudi Aramco on O2C (25% stake sale process ongoing)."),
            ],
        },
    }

    default = [
        ("Company Overview",
         "This is a simulated financial filing for demonstration purposes. "
         "The data has been crafted to reflect realistic financial patterns for analysis training."),
        ("Financial Summary",
         "Revenue: ₹50,000 Crore. EBITDA Margin: 15%. PAT: ₹5,000 Crore. Net Debt/EBITDA: 2.0x."),
    ]

    profile = company_profiles.get(company, {})
    return profile.get(filing_type, default)


# ─── Bulk Ingestion Helper ─────────────────────────────────────────────────────
def ingest_all_companies(use_simulation: bool = True) -> list[dict]:
    """Run IRScraper for every company in the registry."""
    all_manifests = []
    for company_name in COMPANY_REGISTRY:
        scraper = IRScraper(company_name=company_name, use_simulation=use_simulation)
        manifest = scraper.run()
        all_manifests.append(manifest)
    return all_manifests


if __name__ == "__main__":
    # Quick CLI test
    manifests = ingest_all_companies(use_simulation=True)
    for m in manifests:
        print(f"\n{m['company']}: {len(m['files'])} files downloaded.")
