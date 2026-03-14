"""
rag/vector_store.py — Dual-Collection ChromaDB Vector Store Manager

Design rationale:
  - Two separate Chroma collections per company:
      * "language_centric"   → qualitative narrative, strategic disclosures
      * "financial_ownership" → quantitative tables, ratios, ownership data
  - MMR (Maximal Marginal Relevance) retrieval to reduce redundancy
  - Supports both OpenAI embeddings and local HuggingFace fallback
  - Metadata filtering (by company, filing_type, year) for scoped retrieval
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_DB_DIR, CHROMA_COLLECTION_LANGUAGE, CHROMA_COLLECTION_FINANCIAL,
    OPENAI_API_KEY, EMBEDDING_MODEL, USE_LOCAL_EMBEDDINGS, LOCAL_EMBEDDING_MODEL,
    TOP_K_RETRIEVAL, LOG_LEVEL,
)
from ingestion.document_processor import ParsedChunk, LANGUAGE_CENTRIC, FINANCIAL_OWNERSHIP

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


# ─── Embedding Factory ─────────────────────────────────────────────────────────
def build_embedding_function():
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Gemini (free)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
            logger.info("Using Gemini embeddings: embedding-001")
            return GoogleGenerativeAiEmbeddingFunction(api_key=gemini_key)
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}")

    # Local fallback
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        logger.info(f"Using local HuggingFace embeddings: {LOCAL_EMBEDDING_MODEL}")
        return SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBEDDING_MODEL)
    except Exception as e:
        logger.warning(f"SentenceTransformer failed: {e}")

    logger.info("Using ChromaDB default embeddings.")
    return None


# ─── Vector Store Manager ──────────────────────────────────────────────────────
class VectorStoreManager:
    """
    Manages two ChromaDB collections for dual-pathway retrieval.
    Provides unified insert/query interface with metadata filtering.
    """

    def __init__(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.embed_fn = build_embedding_function()

        kwargs = {"embedding_function": self.embed_fn} if self.embed_fn else {}

        self._lang_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_LANGUAGE,
            metadata={"hnsw:space": "cosine"},
            **kwargs,
        )
        self._fin_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_FINANCIAL,
            metadata={"hnsw:space": "cosine"},
            **kwargs,
        )

        logger.info(
            f"VectorStore ready — "
            f"Language: {self._lang_collection.count()} docs | "
            f"Financial: {self._fin_collection.count()} docs"
        )

    def _get_collection(self, content_type: str):
        return (
            self._fin_collection
            if content_type == FINANCIAL_OWNERSHIP
            else self._lang_collection
        )

    # ── Indexing ───────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: list[ParsedChunk], batch_size: int = 100) -> int:
        """
        Index a list of ParsedChunk objects. Returns count of new docs added.
        Deduplicates by chunk_id to allow safe re-indexing.
        """
        if not chunks:
            return 0

        # Group by collection type
        groups: dict[str, list[ParsedChunk]] = {
            LANGUAGE_CENTRIC: [],
            FINANCIAL_OWNERSHIP: [],
        }
        for chunk in chunks:
            groups[chunk.content_type].append(chunk)

        total_added = 0
        for content_type, chunk_list in groups.items():
            if not chunk_list:
                continue
            collection = self._get_collection(content_type)

            # Batch insert
            for i in range(0, len(chunk_list), batch_size):
                batch = chunk_list[i : i + batch_size]

                # Check for existing IDs to avoid duplicates
                existing_ids = set()
                try:
                    existing = collection.get(ids=[c.chunk_id for c in batch])
                    existing_ids = set(existing["ids"])
                except Exception:
                    pass

                new_chunks = [c for c in batch if c.chunk_id not in existing_ids]
                if not new_chunks:
                    continue

                collection.add(
                    ids=[c.chunk_id for c in new_chunks],
                    documents=[c.text for c in new_chunks],
                    metadatas=[c.to_dict() for c in new_chunks],
                )
                total_added += len(new_chunks)

        logger.info(f"Indexed {total_added} new chunks.")
        return total_added

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        company: Optional[str] = None,
        content_type: Optional[str] = None,
        filing_type: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict]:
        """
        Retrieve relevant chunks. Queries both collections unless content_type
        is specified. Returns merged, scored list sorted by relevance distance.
        """
        where_filter = self._build_where_filter(company, filing_type, year)

        collections_to_query = []
        if content_type == FINANCIAL_OWNERSHIP:
            collections_to_query = [self._fin_collection]
        elif content_type == LANGUAGE_CENTRIC:
            collections_to_query = [self._lang_collection]
        else:
            # Query both and merge
            collections_to_query = [self._lang_collection, self._fin_collection]

        all_results: list[dict] = []

        for collection in collections_to_query:
            if collection.count() == 0:
                continue
            try:
                query_kwargs = {
                    "query_texts": [query_text],
                    "n_results": min(top_k, collection.count()),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_filter:
                    query_kwargs["where"] = where_filter

                results = collection.query(**query_kwargs)

                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    all_results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "relevance_score": round(1 - dist, 4),
                        "content_type": meta.get("content_type", "unknown"),
                    })
            except Exception as e:
                logger.warning(f"Query error on '{collection.name}': {e}")

        # Sort by ascending distance (most relevant first)
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:top_k]

    @staticmethod
    def _build_where_filter(
        company: Optional[str],
        filing_type: Optional[str],
        year: Optional[int],
    ) -> Optional[dict]:
        """Build a Chroma $and metadata filter from optional parameters."""
        conditions = []
        if company:
            conditions.append({"company": {"$eq": company}})
        if filing_type:
            conditions.append({"filing_type": {"$eq": filing_type}})
        if year:
            conditions.append({"year": {"$eq": year}})

        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── Stats & Utilities ──────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "language_centric_count": self._lang_collection.count(),
            "financial_ownership_count": self._fin_collection.count(),
            "total": self._lang_collection.count() + self._fin_collection.count(),
            "db_path": str(CHROMA_DB_DIR),
        }

    def get_indexed_companies(self) -> list[str]:
        """Return all unique companies present in the vector store."""
        companies = set()
        for collection in [self._lang_collection, self._fin_collection]:
            if collection.count() == 0:
                continue
            try:
                results = collection.get(include=["metadatas"])
                for meta in results["metadatas"]:
                    if "company" in meta:
                        companies.add(meta["company"])
            except Exception:
                pass
        return sorted(companies)

    def delete_company(self, company: str) -> int:
        """Remove all chunks for a company from both collections."""
        total_deleted = 0
        for collection in [self._lang_collection, self._fin_collection]:
            try:
                collection.delete(where={"company": {"$eq": company}})
                total_deleted += 1
            except Exception as e:
                logger.warning(f"Delete error: {e}")
        logger.info(f"Deleted all data for: {company}")
        return total_deleted
