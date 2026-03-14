"""
ingestion/document_processor.py — Multi-Modal PDF Parser & Context Classifier

Capabilities:
  1. Text extraction from PDFs (pdfplumber preferred, PyPDF2 fallback)
  2. Table detection and structured extraction
  3. Context-aware classification:
     - LANGUAGE_CENTRIC: narrative, qualitative, forward-looking statements
     - FINANCIAL_OWNERSHIP: quantitative tables, figures, ratios, ownership data
  4. Smart chunking with metadata tagging

Classification drives separate indexing paths in the vector store for
higher retrieval precision on financial figures.
"""

import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHUNK_SIZE_TEXT, CHUNK_SIZE_FINANCIAL, CHUNK_OVERLAP, LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ─── Content Type Constants ────────────────────────────────────────────────────
LANGUAGE_CENTRIC = "language_centric"
FINANCIAL_OWNERSHIP = "financial_ownership"

# ─── Regex Patterns for Classification ────────────────────────────────────────
_FINANCIAL_PATTERNS = [
    r"₹\s*[\d,]+",                      # Indian Rupee amounts
    r"\$\s*[\d,]+",                      # USD amounts
    r"£\s*[\d,]+",                       # GBP amounts
    r"\d+\.\d+\s*%",                     # Percentages with decimal
    r"\b\d{1,3}(?:,\d{3})+\b",          # Large numbers with commas (e.g. 1,23,456)
    r"\b(?:crore|lakh|billion|million)\b",
    r"\b(?:ebitda|ebit|pat|roce|roe|eps|bps|cagr|fy\d{2,4})\b",
    r"\b(?:revenue|turnover|profit|loss|margin|debt|equity|capex)\b",
    r"(?:q[1-4]|fy)\s*20\d{2}",         # Quarter/FY references
    r"\b(?:ownership|shareholding|promoter|fii|dii|mf)\s*%",
    r"\btable|balance sheet|cash flow|income statement\b",
]

_LANGUAGE_PATTERNS = [
    r"\bstrategy\b|\bvision\b|\bexpansion\b|\boutlook\b",
    r"\bcommitted to\b|\bwe believe\b|\bour goal\b",
    r"\bsustainability\b|\besg\b|\bgovernance\b",
    r"\bmanagement discussion\b|\bmd&a\b",
    r"\brisk factor\b|\bopportunity\b|\bchallenge\b",
    r"\bmarket share\b|\bcustomer\b|\bproduct launch\b",
]

_FINANCIAL_RE = re.compile("|".join(_FINANCIAL_PATTERNS), re.IGNORECASE)
_LANGUAGE_RE = re.compile("|".join(_LANGUAGE_PATTERNS), re.IGNORECASE)


# ─── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class ParsedChunk:
    """A single text chunk with full provenance metadata."""
    text: str
    chunk_id: str
    content_type: str           # LANGUAGE_CENTRIC or FINANCIAL_OWNERSHIP
    source_path: str
    company: str
    filing_type: str
    year: Optional[int]
    page_numbers: list[int] = field(default_factory=list)
    has_table: bool = False
    financial_signal_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert chunk to ChromaDB-compatible metadata dict."""
        return {
            "chunk_id": self.chunk_id,
            "content_type": self.content_type,
            "source_path": self.source_path,
            "company": self.company,
            "filing_type": self.filing_type,
            "year": int(self.year) if isinstance(self.year, (int, float, str)) and self.year else 0,
            "page_numbers": str(self.page_numbers),
            "has_table": self.has_table,
            "financial_signal_score": float(self.financial_signal_score),
        }


@dataclass
class ParsedDocument:
    """Container for all chunks extracted from one PDF."""
    source_path: str
    company: str
    filing_type: str
    year: Optional[int]
    total_pages: int
    chunks: list[ParsedChunk] = field(default_factory=list)

    @property
    def language_chunks(self) -> list[ParsedChunk]:
        return [c for c in self.chunks if c.content_type == LANGUAGE_CENTRIC]

    @property
    def financial_chunks(self) -> list[ParsedChunk]:
        return [c for c in self.chunks if c.content_type == FINANCIAL_OWNERSHIP]


# ─── Core Document Processor ───────────────────────────────────────────────────
class DocumentProcessor:
    """
    Parses a PDF into classified, chunked content ready for vector indexing.

    Pipeline:
        PDF → raw text per page → table detection → classification →
        adaptive chunking → ParsedChunk list
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ── PDF Text Extraction ────────────────────────────────────────────────────
    def extract_pages(self, pdf_path: str) -> list[dict]:
        """
        Extract text and tables from each page.
        Returns list of {page_num, text, tables: list[str]}.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []

        # Primary: pdfplumber (best for tables)
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = []
                    for tbl in page.extract_tables():
                        if tbl:
                            table_str = self._table_to_text(tbl)
                            if table_str.strip():
                                tables.append(table_str)
                    pages.append({
                        "page_num": i + 1,
                        "text": text,
                        "tables": tables,
                    })
            logger.debug(f"pdfplumber: extracted {len(pages)} pages from {path.name}")
            return pages

        except ImportError:
            logger.warning("pdfplumber not installed. Falling back to PyPDF2.")

        # Fallback: PyPDF2
        try:
            import PyPDF2
            with open(str(path), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append({"page_num": i + 1, "text": text, "tables": []})
            return pages

        except ImportError:
            logger.warning("PyPDF2 not installed. Using basic text fallback.")

        # Last resort: read as text (for .txt stubs in simulation)
        content = path.read_text(errors="ignore")
        return [{"page_num": 1, "text": content, "tables": []}]

    @staticmethod
    def _table_to_text(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to tab-separated text."""
        lines = []
        for row in table:
            if row:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                lines.append("\t".join(cleaned))
        return "\n".join(lines)

    # ── Classification ─────────────────────────────────────────────────────────
    @staticmethod
    def classify_chunk(text: str) -> tuple[str, float]:
        """
        Determine if a text block is FINANCIAL_OWNERSHIP or LANGUAGE_CENTRIC.
        Returns (content_type, financial_signal_score).

        Scoring: count matches per 100 words, financial wins on tie.
        """
        words = max(len(text.split()), 1)
        fin_matches = len(_FINANCIAL_RE.findall(text))
        lang_matches = len(_LANGUAGE_RE.findall(text))

        fin_score = (fin_matches / words) * 100
        lang_score = (lang_matches / words) * 100

        # Heuristic: if the text contains a recognizable table structure,
        # always classify as financial regardless of score.
        has_table_structure = bool(re.search(r"(\t|\|){2,}", text))
        has_numeric_density = fin_matches >= 3

        if has_table_structure or has_numeric_density or fin_score >= lang_score:
            return FINANCIAL_OWNERSHIP, round(fin_score, 3)
        else:
            return LANGUAGE_CENTRIC, round(fin_score, 3)

    # ── Adaptive Chunker ───────────────────────────────────────────────────────
    def _chunk_text(
        self,
        text: str,
        content_type: str,
        page_num: int,
        has_table: bool,
    ) -> list[dict]:
        """
        Split text using appropriate chunk size based on content type.
        Uses sentence-aware splitting to avoid mid-sentence breaks.
        """
        chunk_size = CHUNK_SIZE_FINANCIAL if content_type == FINANCIAL_OWNERSHIP else CHUNK_SIZE_TEXT
        overlap = CHUNK_OVERLAP

        # Tokenize by sentence boundaries for cleaner splits
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        chunks_out = []
        current_tokens: list[str] = []
        current_len = 0

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            word_count = len(words)

            if current_len + word_count > chunk_size and current_tokens:
                chunk_text = " ".join(current_tokens).strip()
                if len(chunk_text) > 50:
                    chunks_out.append({
                        "text": chunk_text,
                        "page_num": page_num,
                        "has_table": has_table,
                    })
                # Overlap: keep last N words
                overlap_words = current_tokens[-overlap:] if len(current_tokens) > overlap else current_tokens[:]
                current_tokens = overlap_words + words
                current_len = len(current_tokens)
            else:
                current_tokens.extend(words)
                current_len += word_count

        # Flush remaining
        if current_tokens:
            chunk_text = " ".join(current_tokens).strip()
            if len(chunk_text) > 50:
                chunks_out.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "has_table": has_table,
                })

        return chunks_out

    # ── Main Process Method ────────────────────────────────────────────────────
    def process(
        self,
        pdf_path: str,
        company: str,
        filing_type: str,
        year: Optional[int] = None,
    ) -> ParsedDocument:
        """
        Full pipeline: PDF → ParsedDocument with classified chunks.
        """
        logger.info(f"Processing: {Path(pdf_path).name} ({company}, {filing_type})")
        pages = self.extract_pages(pdf_path)
        total_pages = len(pages)

        doc = ParsedDocument(
            source_path=pdf_path,
            company=company,
            filing_type=filing_type,
            year=year,
            total_pages=total_pages,
        )

        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page_num"]
            raw_text = page_data["text"]
            tables = page_data["tables"]

            # Process main text body
            if raw_text.strip():
                content_type, fin_score = self.classify_chunk(raw_text)
                for chunk_data in self._chunk_text(raw_text, content_type, page_num, has_table=False):
                    chunk = ParsedChunk(
                        text=chunk_data["text"],
                        chunk_id=f"{company.lower().replace(' ', '_')}_{filing_type}_p{page_num}_c{chunk_index}",
                        content_type=content_type,
                        source_path=pdf_path,
                        company=company,
                        filing_type=filing_type,
                        year=year,
                        page_numbers=[page_num],
                        has_table=False,
                        financial_signal_score=fin_score,
                    )
                    doc.chunks.append(chunk)
                    chunk_index += 1

            # Process tables separately → always FINANCIAL_OWNERSHIP
            for table_text in tables:
                if not table_text.strip():
                    continue
                _, fin_score = self.classify_chunk(table_text)
                for chunk_data in self._chunk_text(table_text, FINANCIAL_OWNERSHIP, page_num, has_table=True):
                    chunk = ParsedChunk(
                        text=chunk_data["text"],
                        chunk_id=f"{company.lower().replace(' ', '_')}_{filing_type}_p{page_num}_tbl{chunk_index}",
                        content_type=FINANCIAL_OWNERSHIP,
                        source_path=pdf_path,
                        company=company,
                        filing_type=filing_type,
                        year=year,
                        page_numbers=[page_num],
                        has_table=True,
                        financial_signal_score=fin_score,
                    )
                    doc.chunks.append(chunk)
                    chunk_index += 1

        logger.info(
            f"  → {total_pages} pages | "
            f"{len(doc.language_chunks)} language chunks | "
            f"{len(doc.financial_chunks)} financial chunks"
        )
        return doc

    # ── Batch Processor ────────────────────────────────────────────────────────
    def process_company_directory(self, company: str, company_slug: str) -> list[ParsedDocument]:
        """Process all PDFs in a company's data directory."""
        from config import DATA_DIR
        company_dir = DATA_DIR / company_slug
        documents = []

        for filing_type_dir in company_dir.iterdir():
            if not filing_type_dir.is_dir() or filing_type_dir.name in ["uploads"]:
                continue

            for pdf_file in filing_type_dir.glob("*.pdf"):
                year = self._extract_year_from_path(str(pdf_file))
                try:
                    doc = self.process(
                        str(pdf_file),
                        company=company,
                        filing_type=filing_type_dir.name,
                        year=year,
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {e}")

        return documents

    @staticmethod
    def _extract_year_from_path(path: str) -> Optional[int]:
        matches = re.findall(r"20[1-3]\d", path)
        return int(matches[-1]) if matches else None
