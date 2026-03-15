"""
config.py — Central Configuration for Financial Analyst Intelligence Tool
Manage all environment variables, paths, and LLM settings here.
"""
import os
try:
    import streamlit as st
    for k, v in st.secrets.items():
        os.environ.setdefault(k, v)
except Exception:
    pass  # Local dev — use .env instead

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path


load_dotenv()

# ─── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "companies"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"

# Ensure critical directories exist
for d in [DATA_DIR, CHROMA_DB_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── LLM / Embedding Settings ─────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
USE_LOCAL_EMBEDDINGS = not bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fallback: use a local HuggingFace model if no OpenAI key is present
USE_LOCAL_EMBEDDINGS = not bool(OPENAI_API_KEY)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Chunking Strategy ─────────────────────────────────────────────────────────
CHUNK_SIZE_TEXT = 1000          # Tokens for language-centric (qualitative) content
CHUNK_SIZE_FINANCIAL = 600      # Tokens for financial tables (smaller = more precise)
CHUNK_OVERLAP = 150

# ─── Vector Store ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION_LANGUAGE = "language_centric"
CHROMA_COLLECTION_FINANCIAL = "financial_ownership"

# ─── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 6             # Documents fetched per retrieval step
RERANK_TOP_N = 3                # After MMR / reranking
MAX_AGENT_ITERATIONS = 5        # LangGraph self-correction loop cap
SIMILARITY_THRESHOLD = 0.35     # Below this → agent triggers re-query

# ─── Company Registry ──────────────────────────────────────────────────────────
# Each entry maps a display name → metadata used by the ingestion module
COMPANY_REGISTRY = {
    "Tata Motors": {
        "slug": "tata_motors",
        "ticker": "TATAMOTORS.NS",
        "ir_base_url": "https://www.tatamotors.com/investors/",
        "exchange": "NSE",
        "sector": "Automobile",
    },
    "Hindustan Unilever (HUL)": {
        "slug": "hul",
        "ticker": "HINDUNILVR.NS",
        "ir_base_url": "https://www.hul.co.in/investor-relations/",
        "exchange": "NSE",
        "sector": "FMCG",
    },
    "Reliance Industries": {
        "slug": "reliance",
        "ticker": "RELIANCE.NS",
        "ir_base_url": "https://www.ril.com/investors/",
        "exchange": "NSE",
        "sector": "Conglomerate",
    },
}

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"
