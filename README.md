# 📊 Financial Analyst Intelligence Tool

> An **Agentic RAG** system for Chartered Accountants and Credit Analysts — powered by **LangGraph**, **ChromaDB**, and **Streamlit**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINANCIAL ANALYST INTELLIGENCE TOOL                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INGESTION LAYER                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  IRScraper (ingestion/scraper.py)                                   │   │
│  │  • BeautifulSoup IR page parser → PDF link discovery                │   │
│  │  • Rate-limited downloader with retry logic                         │   │
│  │  • Filing classifier: annual_report / quarterly / concall           │   │
│  │  • Simulation mode with rich placeholder PDFs (via reportlab)       │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│                        │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐   │
│  │  DocumentProcessor (ingestion/document_processor.py)                │   │
│  │  • pdfplumber: text + table extraction per page                     │   │
│  │  • Context Classifier (regex scoring):                              │   │
│  │    ├── LANGUAGE_CENTRIC (qualitative, strategy, outlook)            │   │
│  │    └── FINANCIAL_OWNERSHIP (tables, ratios, figures)                │   │
│  │  • Adaptive chunking: 1000t (text) / 600t (financial)              │   │
│  │  • ParsedChunk with full metadata (company, year, page, type)       │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│                        │                                                    │
│  STORAGE LAYER         │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐   │
│  │  VectorStoreManager (rag/vector_store.py) — ChromaDB                │   │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │   │
│  │  │ language_centric     │  │ financial_ownership                  │ │   │
│  │  │ collection           │  │ collection                           │ │   │
│  │  │ (narrative text)     │  │ (tables, numbers, ratios)            │ │   │
│  │  └──────────────────────┘  └──────────────────────────────────────┘ │   │
│  │  • OpenAI text-embedding-3-small / HuggingFace fallback             │   │
│  │  • Cosine similarity + metadata filtering                           │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│                        │                                                    │
│  AGENT LAYER           │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐   │
│  │  FinancialRAGAgent — LangGraph StateGraph (rag/pipeline.py)         │   │
│  │                                                                     │   │
│  │  query_analyst ──► dual_retriever ──► context_grader               │   │
│  │                         ▲                    │                     │   │
│  │                         │              SUFFICIENT?                 │   │
│  │                    self_corrector          │    │                  │   │
│  │                         ▲                 NO   YES                │   │
│  │                         └─────────────────┘    │                  │   │
│  │                                                 ▼                  │   │
│  │                              knowledge_synthesizer                 │   │
│  │                                       │                            │   │
│  │                                       ▼                            │   │
│  │                                final_responder                     │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│                        │                                                    │
│  PRESENTATION LAYER    │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐   │
│  │  Streamlit Dashboard (app.py)                                       │   │
│  │  • Tab 1: Agentic Q&A with risk flags + citations + key metrics     │   │
│  │  • Tab 2: Document upload + filing browser                          │   │
│  │  • Tab 3: Analytics (chunk distribution, filing counts)             │   │
│  │  • Tab 4: Architecture reference                                    │   │
│  │  • Sidebar: Company selector, ingestion controls, index stats       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
financial_analyst_tool/
├── app.py                          # Streamlit dashboard (entry point)
├── config.py                       # Centralized configuration
├── setup_and_run.py                # One-click setup CLI
├── requirements.txt
├── .env.example                    # Environment variable template
│
├── ingestion/
│   ├── __init__.py
│   ├── scraper.py                  # IRScraper — web scraping + PDF download
│   └── document_processor.py      # Multi-modal PDF parser + classifier
│
├── rag/
│   ├── __init__.py
│   ├── vector_store.py             # ChromaDB dual-collection manager
│   ├── pipeline.py                 # LangGraph Agentic RAG engine
│   └── indexer.py                  # Document indexing orchestrator
│
└── data/
    ├── companies/
    │   ├── tata_motors/
    │   │   ├── annual_report/
    │   │   ├── quarterly_briefing/
    │   │   └── manifest.json
    │   ├── hul/
    │   └── reliance/
    └── chroma_db/                  # Persistent ChromaDB storage
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd financial_analyst_tool
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
# Without key: runs in demo mode with mock LLM + local embeddings
```

### 3. One-Click Setup
```bash
python setup_and_run.py
```
This will:
- Download/simulate company filings (Tata Motors, HUL, Reliance)
- Parse and index all documents into ChromaDB
- Run a test query to verify the pipeline

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open: **http://localhost:8501**

---

## 🔑 Key Design Decisions

### Dual-Collection RAG
Most RAG systems use a single vector store. This system uses **two separate ChromaDB collections**:
- `language_centric` — qualitative text (strategy, risk factors, MD&A)
- `financial_ownership` — quantitative data (financial tables, ratios, ownership patterns)

This improves retrieval precision for financial queries (avoid mixing narrative with numbers).

### LangGraph Self-Correction
Instead of a single retrieval pass, the LangGraph agent:
1. **Plans** by decomposing the query into sub-queries
2. **Retrieves** from both collections
3. **Grades** the context quality
4. **Self-corrects** with refined queries if context is insufficient (up to 5 iterations)
5. **Synthesizes** a structured financial analysis

### Context-Aware Chunking
- Financial tables: 600 token chunks (smaller = more precise, less noise around figures)
- Narrative text: 1000 token chunks (larger = more coherent context for qualitative analysis)

---

## 💡 Sample Queries

```
"What are the expansion plans for retail outlets and the associated credit risk?"
"Analyze the debt reduction trajectory and free cash flow outlook for FY25."
"What are the key risks disclosed by management and their potential financial impact?"
"Compare EBITDA margin progression over the last 3 years."
"What is the shareholding pattern and any significant changes in promoter stake?"
"Summarize JLR's performance and its contribution to consolidated results."
```

---

## 🏢 Supported Companies (Out-of-Box)

| Company | Ticker | Sector | Exchange |
|---------|--------|--------|----------|
| Tata Motors | TATAMOTORS.NS | Automobile | NSE |
| Hindustan Unilever (HUL) | HINDUNILVR.NS | FMCG | NSE |
| Reliance Industries | RELIANCE.NS | Conglomerate | NSE |

> Adding new companies: Add entry to `COMPANY_REGISTRY` in `config.py` and provide an IR URL.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Agentic Orchestration | LangGraph (StateGraph) |
| LLM | GPT-4o (OpenAI) |
| Embeddings | text-embedding-3-small / sentence-transformers |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| PDF Parsing | pdfplumber + PyPDF2 |
| Web Scraping | requests + BeautifulSoup4 |
| UI | Streamlit |
| PDF Generation | reportlab |

---

## 📈 Extending the System

- **Add reranking**: Integrate Cohere Rerank or `FlashRank` before synthesis
- **Add BM25**: Hybrid sparse+dense retrieval for better keyword matching on financial figures
- **Multi-company queries**: Extend the state to handle cross-company analysis
- **Real-time data**: Connect to NSE/BSE API for live price and announcement feeds
- **Export**: Add PDF report generation from the analysis output

---

*Built as a sophisticated internship project for global risk analytics firms.*
