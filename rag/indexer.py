"""
rag/indexer.py — Document Indexing Orchestrator

Bridges the ingestion pipeline (DocumentProcessor) and the vector store.
Provides high-level indexing functions used by both the CLI and Streamlit UI.
"""

import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, COMPANY_REGISTRY, LOG_LEVEL
from ingestion.document_processor import DocumentProcessor
from rag.vector_store import VectorStoreManager

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """High-level indexer that coordinates parsing → embedding → storing."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.vs = VectorStoreManager()

    def index_company(self, company_name: str) -> dict:
        """
        Parse and index all downloaded PDFs for a company.
        Returns a summary dict.
        """
        meta = COMPANY_REGISTRY.get(company_name)
        if not meta:
            return {"error": f"Unknown company: {company_name}"}

        slug = meta["slug"]
        company_dir = DATA_DIR / slug

        if not company_dir.exists():
            return {"error": f"No data directory found for {company_name}. Run ingestion first."}

        total_chunks = 0
        total_docs = 0
        errors = []

        for filing_dir in company_dir.iterdir():
            if not filing_dir.is_dir() or filing_dir.name in ["uploads"]:
                continue
            filing_type = filing_dir.name

            for pdf_file in filing_dir.glob("*.pdf"):
                logger.info(f"Indexing: {pdf_file.name}")
                try:
                    doc = self.processor.process(
                        str(pdf_file),
                        company=company_name,
                        filing_type=filing_type,
                        year=self.processor._extract_year_from_path(str(pdf_file)),
                    )
                    added = self.vs.index_chunks(doc.chunks)
                    total_chunks += added
                    total_docs += 1
                    logger.info(f"  → {added} chunks indexed")
                except Exception as e:
                    logger.error(f"  Error indexing {pdf_file.name}: {e}")
                    errors.append({"file": pdf_file.name, "error": str(e)})

        return {
            "company": company_name,
            "documents_processed": total_docs,
            "chunks_indexed": total_chunks,
            "errors": errors,
        }

    def index_uploaded_file(
        self,
        file_path: str,
        company_name: str,
        filing_type: str = "uploads",
        year: Optional[int] = None,
    ) -> dict:
        """Index a user-uploaded PDF file."""
        try:
            doc = self.processor.process(
                file_path,
                company=company_name,
                filing_type=filing_type,
                year=year,
            )
            added = self.vs.index_chunks(doc.chunks)
            return {
                "success": True,
                "company": company_name,
                "chunks_indexed": added,
                "language_chunks": len(doc.language_chunks),
                "financial_chunks": len(doc.financial_chunks),
            }
        except Exception as e:
            logger.error(f"Upload indexing error: {e}")
            return {"success": False, "error": str(e)}

    def index_all_companies(self) -> list[dict]:
        """Index all companies in the registry."""
        results = []
        for company_name in COMPANY_REGISTRY:
            logger.info(f"\n{'='*50}")
            logger.info(f"Indexing: {company_name}")
            result = self.index_company(company_name)
            results.append(result)
        return results

    def get_index_stats(self) -> dict:
        stats = self.vs.get_stats()
        stats["indexed_companies"] = self.vs.get_indexed_companies()
        return stats
