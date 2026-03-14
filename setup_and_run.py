"""
setup_and_run.py — One-Click Setup & Demo Runner

Performs:
  1. Simulates company filing downloads (reportlab placeholder PDFs)
  2. Indexes all documents into ChromaDB
  3. Runs a sample query to verify the pipeline
  4. Prints instructions to launch Streamlit

Usage:
    python setup_and_run.py
    python setup_and_run.py --skip-index   (if already indexed)
    python setup_and_run.py --query "What are the key risks?"
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     FINANCIAL ANALYST INTELLIGENCE TOOL — Setup Runner      ║
║     Agentic RAG | LangGraph | ChromaDB | Streamlit          ║
╚══════════════════════════════════════════════════════════════╝
"""


def step(n: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*60}")


def run_setup(args):
    print(BANNER)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if not args.skip_index:
        step(1, "Downloading Company Filings (Simulation Mode)")
        from ingestion.scraper import ingest_all_companies
        manifests = ingest_all_companies(use_simulation=True)
        for m in manifests:
            print(f"  ✓ {m['company']}: {len(m['files'])} files")

        # ── Step 2: Index ─────────────────────────────────────────────────────
        step(2, "Parsing & Indexing Documents into ChromaDB")
        from rag.indexer import DocumentIndexer
        indexer = DocumentIndexer()
        results = indexer.index_all_companies()

        total_chunks = sum(r.get("chunks_indexed", 0) for r in results)
        for r in results:
            print(
                f"  ✓ {r['company']}: "
                f"{r.get('chunks_indexed', 0)} chunks from "
                f"{r.get('documents_processed', 0)} documents"
            )
        print(f"\n  Total chunks indexed: {total_chunks}")

    # ── Step 3: Stats ─────────────────────────────────────────────────────────
    step(3, "Vector Store Statistics")
    from rag.vector_store import VectorStoreManager
    vs = VectorStoreManager()
    stats = vs.get_stats()
    print(f"  Language-Centric Chunks : {stats['language_centric_count']}")
    print(f"  Financial/Ownership     : {stats['financial_ownership_count']}")
    print(f"  Total                   : {stats['total']}")
    print(f"  Indexed Companies       : {', '.join(vs.get_indexed_companies())}")

    # ── Step 4: Test Query ────────────────────────────────────────────────────
    step(4, "Running Test Query")
    from rag.pipeline import FinancialRAGAgent

    test_query = args.query or (
        "What are the key credit risks and the debt reduction strategy "
        "for this company over the next two years?"
    )
    test_company = args.company or "Tata Motors"

    print(f"\n  Company : {test_company}")
    print(f"  Query   : {test_query}\n")

    agent = FinancialRAGAgent(vs)
    result = agent.query(question=test_query, company=test_company)

    print("\n" + "="*60)
    print("  AGENT RESPONSE PREVIEW")
    print("="*60)
    # Print first 800 chars of response
    preview = result.get("response", "")[:800]
    print(preview)
    if len(result.get("response", "")) > 800:
        print("  ... [truncated — see full output in Streamlit UI]")

    print(f"\n  Confidence : {result.get('confidence', 'N/A').upper()}")
    print(f"  Iterations : {result.get('iterations', 'N/A')}")
    print(f"  Risk Flags : {len(result.get('risk_flags', []))}")

    # ── Step 5: Launch ────────────────────────────────────────────────────────
    step(5, "Ready to Launch")
    print("""
  ✅ Setup complete! Launch the Streamlit dashboard with:

     streamlit run app.py

  Then open: http://localhost:8501

  Tips:
  • Set OPENAI_API_KEY in .env for full AI-powered responses
  • Without API key, the tool runs in demo mode with mock LLM
  • Use the sidebar to select companies and run ingestion
  • Try complex queries like:
    "What are the expansion plans for retail and associated credit risks?"
    "Analyze debt trajectory and free cash flow outlook."
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and run Financial Analyst Intelligence Tool")
    parser.add_argument("--skip-index", action="store_true", help="Skip ingestion and indexing steps")
    parser.add_argument("--query", type=str, default=None, help="Custom test query")
    parser.add_argument("--company", type=str, default=None, help="Company for test query")
    args = parser.parse_args()

    try:
        run_setup(args)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Tip: Run 'pip install -r requirements.txt' first.")
        sys.exit(1)
