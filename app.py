"""
app.py — Financial Analyst Intelligence Tool
Streamlit Dashboard for Chartered Accountants & Credit Analysts

Features:
  - Company selector with metadata display
  - One-click PDF ingestion + indexing pipeline
  - Document upload with auto-indexing
  - Agentic RAG Q&A with structured output
  - Risk flag visualization
  - Source citation explorer
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, COMPANY_REGISTRY, OPENAI_API_KEY,
    CHROMA_DB_DIR, LOG_FILE
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analyst AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global Imports ── */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #d4dbe8;
  }

  /* ── Main Header ── */
  .main-header {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 50%, #0f2036 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56, 161, 255, 0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38a1ff, #a78bfa, #38a1ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0 0 6px 0;
  }
  @keyframes shimmer {
    to { background-position: 200% center; }
  }
  .main-header p {
    color: #7b93b8;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 300;
  }
  .version-badge {
    display: inline-block;
    background: rgba(56, 161, 255, 0.12);
    border: 1px solid rgba(56, 161, 255, 0.3);
    color: #38a1ff;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 10px;
    vertical-align: middle;
  }

  /* ── Cards ── */
  .metric-card {
    background: linear-gradient(145deg, #111827, #1a2235);
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #38a1ff; }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #38a1ff;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #7b93b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* ── Company Info Card ── */
  .company-card {
    background: linear-gradient(145deg, #0d1625, #162030);
    border: 1px solid #1e3050;
    border-left: 3px solid #38a1ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }
  .company-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2eaf5;
  }
  .company-meta span {
    font-size: 0.75rem;
    background: rgba(56, 161, 255, 0.1);
    color: #38a1ff;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Section Headers ── */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #38a1ff;
    border-bottom: 1px solid #1e3050;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* ── Risk Flags ── */
  .risk-flag {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.25);
    border-left: 3px solid #ff6b6b;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #ffb3b3;
  }

  /* ── Source Citations ── */
  .source-cite {
    background: rgba(56, 161, 255, 0.05);
    border: 1px solid rgba(56, 161, 255, 0.15);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #7b93b8;
  }
  .source-cite .relevance {
    float: right;
    color: #38a1ff;
    font-weight: 500;
  }

  /* ── Response Box ── */
  .response-container {
    background: #0d1625;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 24px;
    line-height: 1.75;
  }
  .response-container h2, .response-container h3 {
    font-family: 'Syne', sans-serif;
    color: #e2eaf5;
  }

  /* ── Query Intent Badge ── */
  .intent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .intent-credit_risk   { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }
  .intent-expansion     { background: rgba(72,199,142,0.15);  color: #48c78e; border: 1px solid rgba(72,199,142,0.3); }
  .intent-financial     { background: rgba(56,161,255,0.15);  color: #38a1ff; border: 1px solid rgba(56,161,255,0.3); }
  .intent-general       { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }

  /* ── Confidence Indicator ── */
  .confidence-high   { color: #48c78e; font-weight: 600; }
  .confidence-medium { color: #fbbf24; font-weight: 600; }
  .confidence-low    { color: #ff6b6b; font-weight: 600; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #2563b0) !important;
    color: #e2eaf5 !important;
    border: 1px solid #38a1ff !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563b0, #1e3a6e) !important;
    box-shadow: 0 0 20px rgba(56, 161, 255, 0.25) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid #1e3050 !important;
  }
  [data-testid="stSidebar"] .section-header {
    color: #7b93b8;
  }

  /* ── Input Fields ── */
  .stTextArea textarea, .stSelectbox select, .stTextInput input {
    background: #0d1625 !important;
    border: 1px solid #1e3050 !important;
    color: #d4dbe8 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextArea textarea:focus {
    border-color: #38a1ff !important;
    box-shadow: 0 0 0 2px rgba(56,161,255,0.15) !important;
  }

  /* ── Status Messages ── */
  .stSuccess { background: rgba(72,199,142,0.1) !important; border: 1px solid rgba(72,199,142,0.3) !important; }
  .stWarning { background: rgba(251,191,36,0.1) !important; border: 1px solid rgba(251,191,36,0.3) !important; }
  .stError   { background: rgba(255,107,107,0.1) !important; border: 1px solid rgba(255,107,107,0.3) !important; }

  /* ── Pipeline Step Visualizer ── */
  .pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 16px 0;
    overflow-x: auto;
  }
  .pipeline-step {
    background: rgba(56,161,255,0.05);
    border: 1px solid rgba(56,161,255,0.2);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #7b93b8;
    white-space: nowrap;
  }
  .pipeline-step.active {
    background: rgba(56,161,255,0.15);
    border-color: #38a1ff;
    color: #38a1ff;
  }
  .pipeline-arrow {
    color: #1e3050;
    padding: 0 6px;
    font-size: 1rem;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #080c16; }
  ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #38a1ff; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "agent": None,
        "indexer": None,
        "selected_company": None,
        "query_history": [],
        "indexed_companies": [],
        "ingestion_status": {},
        "api_key_set": bool(OPENAI_API_KEY),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Lazy-Load Backend ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_backend():
    """Load vector store + agent once and cache across sessions."""
    try:
        from rag.vector_store import VectorStoreManager
        from rag.pipeline import FinancialRAGAgent
        from rag.indexer import DocumentIndexer
        vs = VectorStoreManager()
        agent = FinancialRAGAgent(vs)
        indexer = DocumentIndexer()
        stats = vs.get_stats()
        companies = vs.get_indexed_companies()
        return {
            "agent": agent,
            "indexer": indexer,
            "vs": vs,
            "stats": stats,
            "indexed_companies": companies,
            "error": None,
        }
    except Exception as e:
        return {
            "agent": None, "indexer": None, "vs": None,
            "stats": {}, "indexed_companies": [],
            "error": str(e),
        }


# ─── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(backend: dict):
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

        # API Key input
        api_key_display = "✅ Configured" if OPENAI_API_KEY else "⚠️ Not set (demo mode)"
        st.caption(f"OpenAI API: {api_key_display}")
        if not OPENAI_API_KEY:
            key_input = st.text_input("Enter OpenAI API Key", type="password", placeholder="sk-...")
            if key_input:
                os.environ["OPENAI_API_KEY"] = key_input
                st.success("API key set for this session.")

        st.markdown("---")
        st.markdown('<div class="section-header">🏢 Company Selection</div>', unsafe_allow_html=True)

        company_options = list(COMPANY_REGISTRY.keys()) + ["Upload Custom Document"]
        selected = st.selectbox(
            "Select Company",
            company_options,
            help="Select a company to analyze or upload your own document.",
        )
        st.session_state.selected_company = selected

        if selected != "Upload Custom Document":
            meta = COMPANY_REGISTRY[selected]
            st.markdown(f"""
            <div class="company-card">
              <div class="company-name">{selected}</div>
              <div class="company-meta" style="margin-top:8px;">
                <span>{meta['ticker']}</span>
                <span>{meta['sector']}</span>
                <span>{meta['exchange']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">📥 Data Ingestion</div>', unsafe_allow_html=True)

        if st.button("🔄 Ingest & Index Selected Company", use_container_width=True):
            if selected == "Upload Custom Document":
                st.warning("Select a company first to use this option.")
            else:
                _run_ingestion(selected, backend)

        if st.button("📚 Ingest ALL Companies", use_container_width=True):
            _run_ingestion_all(backend)

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Index Stats</div>', unsafe_allow_html=True)

        stats = backend.get("stats", {})
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Language", stats.get("language_centric_count", 0))
            with col2:
                st.metric("Financial", stats.get("financial_ownership_count", 0))
            st.caption(f"Total chunks: **{stats.get('total', 0)}**")

        indexed = backend.get("indexed_companies", [])
        if indexed:
            st.caption("Indexed companies:")
            for c in indexed:
                st.markdown(f"  ✓ {c}", unsafe_allow_html=False)

        st.markdown("---")
        st.caption("Financial Analyst Intelligence Tool v1.0.0")
        st.caption("Built with LangGraph · ChromaDB · Streamlit")


# ─── Ingestion Handlers ────────────────────────────────────────────────────────
def _run_ingestion(company: str, backend: dict):
    from ingestion.scraper import IRScraper

    with st.spinner(f"Downloading filings for {company}..."):
        try:
            scraper = IRScraper(company_name=company, use_simulation=True)
            manifest = scraper.run()
            st.sidebar.success(f"Downloaded {len(manifest['files'])} files")

            indexer = backend.get("indexer")
            if indexer:
                with st.spinner("Indexing documents..."):
                    result = indexer.index_company(company)
                    st.sidebar.success(
                        f"Indexed {result['chunks_indexed']} chunks from "
                        f"{result['documents_processed']} documents"
                    )
                    # Bust cache
                    load_backend.clear()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")


def _run_ingestion_all(backend: dict):
    from ingestion.scraper import ingest_all_companies

    with st.spinner("Downloading all company filings..."):
        try:
            ingest_all_companies(use_simulation=True)
            indexer = backend.get("indexer")
            if indexer:
                with st.spinner("Indexing all documents..."):
                    results = indexer.index_all_companies()
                    total = sum(r.get("chunks_indexed", 0) for r in results)
                    st.sidebar.success(f"Indexed {total} total chunks across {len(results)} companies")
                    load_backend.clear()
        except Exception as e:
            st.sidebar.error(f"Batch ingestion failed: {e}")


# ─── Main Dashboard ────────────────────────────────────────────────────────────
def render_main(backend: dict):
    # Header
    st.markdown("""
    <div class="main-header">
      <h1>Financial Analyst Intelligence Tool <span class="version-badge">v1.0.0</span></h1>
      <p>Agentic RAG · LangGraph · Multi-Modal PDF Parsing · Dual-Collection Vector Search</p>
    </div>
    """, unsafe_allow_html=True)

    if backend.get("error"):
        st.error(f"⚠️ Backend initialization error: {backend['error']}")
        st.info("Running in limited demo mode. Ensure ChromaDB and dependencies are installed.")

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Query Engine",
        "📄 Document Upload",
        "📈 Analytics Dashboard",
        "🏗️ System Architecture",
    ])

    with tab1:
        render_query_tab(backend)

    with tab2:
        render_upload_tab(backend)

    with tab3:
        render_analytics_tab(backend)

    with tab4:
        render_architecture_tab()


# ─── Tab 1: Query Engine ───────────────────────────────────────────────────────
def render_query_tab(backend: dict):
    selected_company = st.session_state.get("selected_company", list(COMPANY_REGISTRY.keys())[0])
    agent = backend.get("agent")

    col_q, col_opts = st.columns([3, 1])

    with col_q:
        st.markdown('<div class="section-header">💬 Intelligent Query</div>', unsafe_allow_html=True)

        # Query input
        query = st.text_area(
            "Ask a complex financial question",
            height=110,
            placeholder=(
                "e.g. 'What are the company's expansion plans for retail outlets "
                "and what is the associated credit risk?'\n\n"
                "e.g. 'Analyze the debt reduction trajectory and free cash flow outlook "
                "for the next 2 years.'"
            ),
            help="The AI agent will decompose your query, retrieve from the indexed documents, and self-correct if needed.",
        )

    with col_opts:
        st.markdown('<div class="section-header">🎛️ Options</div>', unsafe_allow_html=True)
        filing_filter = st.selectbox(
            "Filing Type",
            ["All", "annual_report", "quarterly_briefing", "investor_presentation"],
            help="Filter retrieval to a specific filing type.",
        )
        company_for_query = st.selectbox(
            "Company (override)",
            ["Use Sidebar Selection"] + list(COMPANY_REGISTRY.keys()),
        )
        show_sources = st.toggle("Show Source Citations", value=True)
        show_pipeline = st.toggle("Show Agent Pipeline", value=True)

    # Quick sample queries
    st.markdown("**Quick Queries:**")
    sample_queries = [
        "What are the key credit risks and debt maturity profile?",
        "Summarize revenue growth, EBITDA margins, and net profit trajectory.",
        "What are the expansion plans and capital expenditure commitments?",
        "Analyze shareholding pattern and promoter stake changes.",
    ]
    cols = st.columns(4)
    for i, sq in enumerate(sample_queries):
        with cols[i]:
            if st.button(sq[:45] + "...", key=f"sample_{i}", use_container_width=True):
                st.session_state["prefill_query"] = sq

    if "prefill_query" in st.session_state:
        query = st.session_state.pop("prefill_query")

    # Run Query
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        effective_company = (
            company_for_query
            if company_for_query != "Use Sidebar Selection"
            else (selected_company if selected_company != "Upload Custom Document" else list(COMPANY_REGISTRY.keys())[0])
        )
        ft_filter = None if filing_filter == "All" else filing_filter

        # Show agent pipeline steps
        if show_pipeline:
            steps = ["PLAN", "RETRIEVE", "GRADE", "SYNTHESIZE", "RESPOND"]
            pipeline_html = '<div class="pipeline-steps">'
            for j, step in enumerate(steps):
                pipeline_html += f'<div class="pipeline-step">{step}</div>'
                if j < len(steps) - 1:
                    pipeline_html += '<span class="pipeline-arrow">→</span>'
            pipeline_html += '</div>'
            st.markdown(pipeline_html, unsafe_allow_html=True)

        with st.spinner("🤖 Agent is analyzing..."):
            start_t = time.time()

            if agent:
                try:
                    result = agent.query(
                        question=query,
                        company=effective_company,
                        filing_type_filter=ft_filter,
                    )
                except Exception as e:
                    result = {
                        "response": f"**Query Error:** {e}\n\nRunning in demo mode — please ensure documents are indexed.",
                        "risk_flags": [],
                        "key_metrics": {},
                        "sources": [],
                        "confidence": "low",
                        "iterations": 0,
                    }
            else:
                result = _demo_response(query, effective_company)

            elapsed = time.time() - start_t

        # ── Display Results ────────────────────────────────────────────────────
        st.markdown(f"*Analysis completed in {elapsed:.2f}s*")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            conf_class = f"confidence-{result.get('confidence', 'medium')}"
            st.markdown(f"**Confidence** <span class='{conf_class}'>{result.get('confidence','medium').upper()}</span>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"**Iterations** `{result.get('iterations', 1)}`")
        with m3:
            st.markdown(f"**Sources** `{len(result.get('sources', []))}`")
        with m4:
            rf_count = len(result.get("risk_flags", []))
            color = "#ff6b6b" if rf_count > 2 else "#fbbf24" if rf_count > 0 else "#48c78e"
            st.markdown(f"**Risk Flags** <span style='color:{color};font-weight:600'>{rf_count}</span>", unsafe_allow_html=True)

        # Response layout
        col_resp, col_meta = st.columns([2, 1])

        with col_resp:
            st.markdown('<div class="section-header">📋 Analysis Report</div>', unsafe_allow_html=True)
            response_text = result.get("response", "No response generated.")

            # Safety net: if JSON leaked through, parse and format it
            import json, re
            clean = response_text.strip()
            # Find JSON block even if surrounded by other text
            json_match = re.search(r'\{[\s\S]*\}', clean)
            if json_match:
                try:
                    # Fix unescaped newlines inside JSON strings
                    raw_json = json_match.group()
                    # Remove actual newlines inside string values
                    fixed = re.sub(r'(?<=\w)\n(?=\s+"|\s+\w)', ' ', raw_json)
                    parsed = json.loads(fixed)
                    parts = ["## 📊 Financial Intelligence Report"]
                    if parsed.get("executive_summary"):
                        parts.append(f"### Executive Summary\n{parsed['executive_summary']}\n")
                    if parsed.get("analysis"):
                        # Restore paragraph breaks
                        analysis_text = parsed['analysis'].replace('\\n\\n', '\n\n').replace('\\n', '\n')
                        parts.append(f"### Detailed Analysis\n{analysis_text}\n")
                    if parsed.get("credit_assessment"):
                        parts.append(f"### 🏦 Credit Assessment\n{parsed['credit_assessment']}\n")
                    if parsed.get("expansion_outlook") and parsed["expansion_outlook"] not in [None, "null"]:
                        parts.append(f"### 🏗️ Expansion Outlook\n{parsed['expansion_outlook']}\n")
                    if parsed.get("risk_flags"):
                        parts.append("### ⚠️ Risk Flags")
                        for f in parsed["risk_flags"]:
                            parts.append(f"- {f}")
                        parts.append("")
                    if parsed.get("key_metrics"):
                        parts.append("### 📈 Key Metrics")
                        for k, v in parsed["key_metrics"].items():
                            parts.append(f"- **{k}:** {v}")
                        parts.append("")
                    if parsed.get("data_gaps"):
                        parts.append("### 🔍 Data Gaps")
                        for g in parsed["data_gaps"]:
                            parts.append(f"- _{g}_")
                    response_text = "\n".join(parts)
                except Exception:
                    # Nuclear fallback: strip JSON entirely and extract text values
                    texts = re.findall(r'"(?:executive_summary|analysis|credit_assessment|expansion_outlook)"\s*:\s*"([^"]*)"', clean)
                    if texts:
                        response_text = "\n\n".join(texts)

            with st.container(border=True):
                st.markdown(response_text)

        with col_meta:
            # Key Metrics
            metrics = result.get("key_metrics", {})
            if metrics:
                st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
                for k, v in metrics.items():
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:7px 0;
                    border-bottom:1px solid #1e3050;font-size:0.82rem;">
                      <span style="color:#7b93b8;">{k}</span>
                      <span style="color:#e2eaf5;font-family:'IBM Plex Mono',monospace;font-weight:500;">{v}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Risk Flags
            risk_flags = result.get("risk_flags", [])
            if risk_flags:
                st.markdown('<div class="section-header">⚠️ Risk Flags</div>', unsafe_allow_html=True)
                for flag in risk_flags:
                    st.markdown(f'<div class="risk-flag">⚠ {flag}</div>', unsafe_allow_html=True)

            # Source Citations
            if show_sources and result.get("sources"):
                st.markdown('<div class="section-header">📌 Source Citations</div>', unsafe_allow_html=True)
                for src in result["sources"][:6]:
                    st.markdown(f"""
                    <div class="source-cite">
                      <span class="relevance">↑{src.get('relevance', 0):.2f}</span>
                      {src.get('filing_type','?')} · {src.get('year','?')} · p.{src.get('page','?')}
                      <br><span style="color:#38a1ff;">{src.get('content_type','?')}</span>
                    </div>""", unsafe_allow_html=True)

        # Save to history
        st.session_state["query_history"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": query[:80],
            "company": effective_company,
            "confidence": result.get("confidence"),
        })

    # ── Query History ──────────────────────────────────────────────────────────
    if st.session_state.get("query_history"):
        with st.expander("📜 Query History", expanded=False):
            for item in reversed(st.session_state["query_history"][-10:]):
                st.markdown(
                    f"`{item['timestamp']}` | **{item['company']}** | "
                    f"_{item['query']}_ | conf: `{item['confidence']}`"
                )


# ─── Tab 2: Document Upload ────────────────────────────────────────────────────
def render_upload_tab(backend: dict):
    st.markdown('<div class="section-header">📤 Upload & Index Document</div>', unsafe_allow_html=True)
    st.info("Upload any financial PDF (Annual Report, Credit Note, Prospectus) to add it to the knowledge base.")

    col_up, col_meta = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Supports Annual Reports, Quarterly Results, Investor Presentations.",
        )

    with col_meta:
        upload_company = st.selectbox(
            "Associate with Company",
            list(COMPANY_REGISTRY.keys()) + ["Custom"],
            key="upload_company",
        )
        upload_type = st.selectbox(
            "Filing Type",
            ["annual_report", "quarterly_briefing", "investor_presentation", "concall", "other"],
            key="upload_type",
        )
        upload_year = st.number_input(
            "Fiscal Year",
            min_value=2010,
            max_value=2030,
            value=2024,
            key="upload_year",
        )

    if uploaded_file and st.button("📥 Index Document", type="primary"):
        indexer = backend.get("indexer")
        if not indexer:
            st.error("Backend not initialized. Check installation.")
            return

        with st.spinner(f"Parsing and indexing {uploaded_file.name}..."):
            # Save to company uploads folder
            company_slug = COMPANY_REGISTRY.get(upload_company, {}).get("slug", "custom")
            dest_dir = DATA_DIR / company_slug / "uploads"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / uploaded_file.name

            dest_path.write_bytes(uploaded_file.getvalue())

            result = indexer.index_uploaded_file(
                file_path=str(dest_path),
                company_name=upload_company,
                filing_type=upload_type,
                year=int(upload_year),
            )

        if result.get("success"):
            st.success(f"✅ Indexed **{result['chunks_indexed']}** chunks")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Language Chunks", result.get("language_chunks", 0))
            with col2:
                st.metric("Financial Chunks", result.get("financial_chunks", 0))
            load_backend.clear()
        else:
            st.error(f"Indexing failed: {result.get('error', 'Unknown error')}")

    # ── Filed Documents Browser ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📁 Downloaded Filings Browser</div>', unsafe_allow_html=True)

    for company_name, meta in COMPANY_REGISTRY.items():
        slug = meta["slug"]
        company_dir = DATA_DIR / slug
        if not company_dir.exists():
            continue

        files_found = list(company_dir.rglob("*.pdf"))
        if not files_found:
            continue

        with st.expander(f"📂 {company_name} ({len(files_found)} files)"):
            for f in files_found:
                relative = f.relative_to(DATA_DIR)
                size_kb = f.stat().st_size / 1024
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f.name)
                with col2:
                    st.caption(f"{size_kb:.1f} KB")
                with col3:
                    st.caption(f.parent.name)


# ─── Tab 3: Analytics Dashboard ───────────────────────────────────────────────
def render_analytics_tab(backend: dict):
    st.markdown('<div class="section-header">📈 Index Analytics</div>', unsafe_allow_html=True)

    stats = backend.get("stats", {})
    indexed_companies = backend.get("indexed_companies", [])

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    metric_data = [
        ("Total Chunks", stats.get("total", 0), "📊"),
        ("Language Chunks", stats.get("language_centric_count", 0), "📝"),
        ("Financial Chunks", stats.get("financial_ownership_count", 0), "💹"),
        ("Companies Indexed", len(indexed_companies), "🏢"),
    ]
    for col, (label, value, icon) in zip([col1, col2, col3, col4], metric_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>
              <div class="metric-value">{value:,}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chunk distribution chart
    if stats.get("total", 0) > 0:
        try:
            import plotly.graph_objects as go

            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.markdown("**Content Type Distribution**")
                fig = go.Figure(go.Pie(
                    labels=["Language Centric", "Financial / Ownership"],
                    values=[
                        stats.get("language_centric_count", 1),
                        stats.get("financial_ownership_count", 1),
                    ],
                    hole=0.5,
                    marker_colors=["#a78bfa", "#38a1ff"],
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#d4dbe8",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=280,
                    showlegend=True,
                    legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col_bar:
                st.markdown("**Filing Types Available**")
                filing_counts = {}
                for company_name, meta in COMPANY_REGISTRY.items():
                    slug = meta["slug"]
                    company_dir = DATA_DIR / slug
                    if company_dir.exists():
                        for sub_dir in company_dir.iterdir():
                            if sub_dir.is_dir():
                                count = len(list(sub_dir.glob("*.pdf")))
                                filing_counts[sub_dir.name] = filing_counts.get(sub_dir.name, 0) + count

                if filing_counts:
                    fig2 = go.Figure(go.Bar(
                        x=list(filing_counts.values()),
                        y=list(filing_counts.keys()),
                        orientation="h",
                        marker_color=["#38a1ff", "#48c78e", "#a78bfa", "#fbbf24"][:len(filing_counts)],
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#d4dbe8",
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280,
                        xaxis=dict(gridcolor="#1e3050"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        except ImportError:
            # Plotly not installed — show text summary
            st.info(f"Language Centric: {stats.get('language_centric_count',0)} | Financial: {stats.get('financial_ownership_count',0)}")

    if indexed_companies:
        st.markdown("---")
        st.markdown('<div class="section-header">✅ Indexed Companies</div>', unsafe_allow_html=True)
        for c in indexed_companies:
            st.markdown(f"&nbsp;&nbsp;✓ **{c}**")


# ─── Tab 4: Architecture ───────────────────────────────────────────────────────
def render_architecture_tab():
    st.markdown('<div class="section-header">🏗️ System Architecture Overview</div>', unsafe_allow_html=True)

    architecture_md = """
## LangGraph Agentic RAG — Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL ANALYST INTELLIGENCE TOOL              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  USER QUERY                                                         │
│     │                                                               │
│     ▼                                                               │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              LANGGRAPH STATE MACHINE                    │       │
│  │                                                         │       │
│  │  1. QUERY ANALYST NODE                                  │       │
│  │     • Decompose complex query → 3-5 sub-queries         │       │
│  │     • Classify intent (credit_risk / expansion / ...)   │       │
│  │                    │                                    │       │
│  │                    ▼                                    │       │
│  │  2. DUAL RETRIEVER NODE                                 │       │
│  │     • ChromaDB Collection A: Language-Centric           │       │
│  │       (Qualitative, Strategic, Narrative)               │       │
│  │     • ChromaDB Collection B: Financial/Ownership        │       │
│  │       (Quantitative, Tables, Ratios, Figures)           │       │
│  │     • MMR + cosine similarity scoring                   │       │
│  │                    │                                    │       │
│  │                    ▼                                    │       │
│  │  3. CONTEXT GRADER NODE                                 │       │
│  │     • LLM evaluates sufficiency of retrieved context    │       │
│  │     • Identifies missing aspects                        │       │
│  │        │                        │                       │       │
│  │    SUFFICIENT                NOT SUFFICIENT             │       │
│  │        │                        │                       │       │
│  │        │              4. SELF-CORRECTOR NODE            │       │
│  │        │                 • Generates refined queries    │       │
│  │        │                 • Re-enters Retriever          │       │
│  │        │                 • Max 5 iterations             │       │
│  │        │                        │                       │       │
│  │        ▼←───────────────────────┘                      │       │
│  │  5. KNOWLEDGE SYNTHESIZER NODE                          │       │
│  │     • Structured financial analysis                     │       │
│  │     • Separates financial figures from narrative        │       │
│  │     • Generates: executive summary, risk flags,         │       │
│  │       key metrics, credit assessment, expansion outlook │       │
│  │                    │                                    │       │
│  │                    ▼                                    │       │
│  │  6. FINAL RESPONDER NODE                                │       │
│  │     • Formats structured Markdown report                │       │
│  │     • Attaches source citations                         │       │
│  │     • Confidence level: HIGH / MEDIUM / LOW             │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  OUTPUT: Structured Analysis Report + Risk Flags + Metrics         │
└─────────────────────────────────────────────────────────────────────┘
```

## Ingestion Pipeline

```
IR Page (Tata Motors / HUL / Reliance)
        │
        ▼
  IRScraper (BeautifulSoup)
  • PDF link discovery
  • Filing type classification (annual / quarterly / concall)
  • Rate-limited downloading
        │
        ▼
  DocumentProcessor (pdfplumber / PyPDF2)
  • Per-page text extraction
  • Table detection + structured extraction
  • Context Classification:
    ├── LANGUAGE_CENTRIC  → qualitative narrative, strategy
    └── FINANCIAL_OWNERSHIP → tables, ratios, ownership
  • Adaptive chunking (1000 tokens / 600 tokens)
        │
        ▼
  VectorStoreManager (ChromaDB)
  ├── Collection: language_centric
  └── Collection: financial_ownership
  • OpenAI or HuggingFace embeddings
  • Metadata: company, year, filing_type, page, chunk_id
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | LangGraph (StateGraph) | Agentic loop with self-correction |
| LLM | GPT-4o / GPT-4-turbo | Query analysis, grading, synthesis |
| Embeddings | text-embedding-3-small | Semantic vector search |
| Vector DB | ChromaDB (persistent) | Dual-collection retrieval |
| PDF Parsing | pdfplumber + PyPDF2 | Text + table extraction |
| Scraping | requests + BeautifulSoup | IR page discovery |
| UI | Streamlit | Dashboard |
| Fallback | sentence-transformers | Local embeddings (no API key) |
    """

    st.markdown(architecture_md)


# ─── Demo Response (when no API key / no indexed docs) ────────────────────────
def _demo_response(query: str, company: str) -> dict:
    """Return a rich demo response for display without an API key."""
    return {
        "response": f"""## 📊 Financial Intelligence Report: {company}
**Query:** {query}

### Executive Summary
{company} demonstrates a resilient financial profile with strong revenue growth trajectory
and expanding EBITDA margins over the past two fiscal years. The credit outlook is broadly
positive, tempered by elevated capital expenditure commitments.

### Detailed Analysis
Based on the available filings, the company has delivered consistent top-line growth driven
by volume expansion and favorable pricing. EBITDA margins have expanded due to operating
leverage and cost optimization initiatives. The management's strategic focus on new business
segments and geographic diversification presents credible upside potential.

From a credit perspective, the net debt position, while elevated, is manageable relative
to EBITDA generation. The debt maturity profile is well-staggered with no near-term
refinancing cliff. Investment-grade credit ratings (CRISIL AA-) provide access to
competitive financing. Free cash flow generation is expected to improve as the high-capex
cycle plateaus.

*Note: This is a demonstration response. Index company documents and add an OpenAI API key
for AI-powered analysis.*

### 📈 Key Financial Metrics

- **Revenue Growth YoY:** ~20-24%
- **EBITDA Margin:** ~12-15%
- **Net Debt/EBITDA:** ~1.5-2.5x
- **ROCE:** ~8-11%
- **Credit Rating:** AA- (CRISIL)

### 🏦 Credit Assessment
**MODERATE** — The company's strong brand equity, diversified revenue streams, and clear
deleveraging trajectory support a moderate-to-strong credit assessment. The main watch
item is execution on CAPEX programs and commodity cost normalization.

### ⚠️ Risk Flags
- Elevated leverage during capex cycle
- Commodity input cost exposure
- Currency risk on international operations
- Competitive intensity in core markets

---
*Confidence: **MEDIUM** | Retrieval Iterations: 0 | Sources: 0*""",
        "risk_flags": [
            "Elevated leverage during capex cycle",
            "Commodity input cost exposure",
            "Currency risk on international operations",
        ],
        "key_metrics": {
            "Revenue Growth": "~20-24% YoY",
            "EBITDA Margin": "~12-15%",
            "Net Debt/EBITDA": "~2.0x",
            "Credit Rating": "AA- (CRISIL)",
        },
        "sources": [],
        "confidence": "medium",
        "iterations": 0,
    }


# ─── App Entry Point ───────────────────────────────────────────────────────────
def main():
    init_session_state()
    backend = load_backend()
    render_sidebar(backend)
    render_main(backend)


if __name__ == "__main__":
    main()
