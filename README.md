# 📊 Financial Analyst Intelligence Tool

> An **Agentic RAG** system for Chartered Accountants and Credit Analysts — powered by **LangGraph**, **Groq (LLaMA 3.3 70B)**, **ChromaDB**, and **Streamlit**.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://agenticragfinance.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Wezo420/agentic-rag-financial-analytics)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINANCIAL ANALYST INTELLIGENCE TOOL                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INGESTION LAYER                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IRScraper (ingestion/scraper.py)                                   │    │
│  │  • BeautifulSoup IR page parser → PDF link discovery                │    │
│  │  • Rate-limited downloader with retry logic                         │    │
│  │  • Filing classifier: annual_report / quarterly / concall           │    │
│  │  • Simulation mode with rich placeholder PDFs (via reportlab)       │    │
│  │                                                                     │    │
│  │  DynamicIRScraper (ingestion/dynamic_scraper.py)                    │    │
│  │  • BSE India API → scrip code lookup → annual reports               │    │
│  │  • NSE India API → symbol lookup → annual reports                   │    │
│  │  • DuckDuckGo PDF search → direct PDF discovery                     │    │
│  │  • IR page scraping → BeautifulSoup PDF link extraction             │    │
│  └─────────────────────┬───────────────────────────────────────────────┘    │
│                        │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐    │
│  │  DocumentProcessor (ingestion/document_processor.py)                │    │
│  │  • pdfplumber: text + table extraction per page                     │    │
│  │  • Context Classifier (regex scoring):                              │    │
│  │    ├── LANGUAGE_CENTRIC (qualitative, strategy, outlook)            │    │
│  │    └── FINANCIAL_OWNERSHIP (tables, ratios, figures)                │    │
│  │  • Adaptive chunking: 1000t (text) / 600t (financial)               │    │
│  │  • ParsedChunk with full metadata (company, year, page, type)       │    │
│  └─────────────────────┬───────────────────────────────────────────────┘    │
│                        │                                                    │
│  STORAGE LAYER         │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐    │
│  │  VectorStoreManager (rag/vector_store.py) — ChromaDB                │    │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │    │
│  │  │ language_centric     │  │ financial_ownership                  │ │    │
│  │  │ collection           │  │ collection                           │ │    │
│  │  │ (narrative text)     │  │ (tables, numbers, ratios)            │ │    │
│  │  └──────────────────────┘  └──────────────────────────────────────┘ │    │
│  │  • ChromaDB built-in all-MiniLM-L6-v2 embeddings                    │    │
│  │  • Cosine similarity + metadata filtering                           │    │
│  └─────────────────────┬───────────────────────────────────────────────┘    │
│                        │                                                    │
│  AGENT LAYER           │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐    │
│  │  FinancialRAGAgent — LangGraph StateGraph (rag/pipeline.py)         │    │
│  │                                                                     │    │
│  │  query_analyst ──► dual_retriever ──► context_grader                │    │
│  │                         ▲                    │                      │    │
│  │                         │              SUFFICIENT?                  │    │
│  │                    self_corrector          │    │                   │    │
│  │                         ▲                 NO   YES                  │    │
│  │                         └─────────────────┘    │                    │    │
│  │                                                 ▼                   │    │
│  │                              knowledge_synthesizer                  │    │
│  │                                       │                             │    │
│  │                                       ▼                             │    │
│  │                                final_responder                      │    │
│  └─────────────────────┬───────────────────────────────────────────────┘    │
│                        │                                                    │
│  PRESENTATION LAYER    │                                                    │
│  ┌─────────────────────▼───────────────────────────────────────────────┐    │
│  │  Streamlit Dashboard (app.py) — 5 Tabs                              │    │
│  │  • Tab 1: Agentic Q&A — risk flags, citations, key metrics          │    │
│  │  • Tab 2: Document Upload — PDF upload + auto-indexing              │    │
│  │  • Tab 3: Analytics Dashboard — chunk stats, filing counts          │    │
│  │  • Tab 4: System Architecture — LangGraph flow diagram              │    │
│  │  • Tab 5: Research Any Company — dynamic web scraper + auto-query   │    │
│  │  • Sidebar: Company selector, ingestion controls, index stats       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 1. 🤖 Agentic RAG Engine (LangGraph)
A 6-node `StateGraph` that reasons across documents and self-corrects:

| Node | Role |
|------|------|
| `query_analyst` | Decomposes complex query into 3-5 sub-queries, classifies intent |
| `dual_retriever` | Fetches from both ChromaDB collections in parallel |
| `context_grader` | LLM evaluates if retrieved context is sufficient |
| `self_corrector` | Generates refined queries if insufficient (up to 5 iterations) |
| `knowledge_synthesizer` | Produces structured markdown financial analysis |
| `final_responder` | Formats final report with risk flags, metrics, credit assessment |

### 2. 📄 Context-Aware PDF Parsing
- **pdfplumber** for text + table extraction
- Regex-based classifier separates content into two types:
  - `LANGUAGE_CENTRIC` — qualitative narratives, strategy, MD&A
  - `FINANCIAL_OWNERSHIP` — quantitative tables, ratios, figures
- Adaptive chunking: **600 tokens** for financial tables, **1000 tokens** for narrative

### 3. 🗄️ Dual-Collection Vector Store
Two separate ChromaDB collections prevent narrative text from polluting quantitative queries — improving precision on financial figure retrieval.

### 4. 🌐 Research Any Company (Local Mode)
A dynamic scraper that:
1. Searches BSE India API for scrip code → fetches annual reports directly
2. Falls back to NSE India API
3. Falls back to DuckDuckGo PDF search
4. Falls back to direct IR page scraping
5. Auto-indexes downloaded PDFs and runs your query — all in one click

> **Note:** This feature works fully when running `app.py` locally. Streamlit Cloud restricts outbound network requests, so it is unavailable on the hosted demo.

### 5. 📊 Streamlit Dashboard
- Dark financial terminal aesthetic
- 5-tab layout with sidebar company selector
- Real-time ingestion progress
- Source citations with relevance scores
- Query history tracking

---

## 📁 Project Structure

```
agentic-rag-financial-analytics/
├── app.py                          # Streamlit dashboard (entry point)
├── config.py                       # Centralized configuration
├── setup_and_run.py                # One-click setup CLI
├── requirements.txt
├── render.yaml                     # Render deployment config
├── .env.example                    # Environment variable template
│
├── ingestion/
│   ├── __init__.py
│   ├── scraper.py                  # IRScraper — simulated IR downloads
│   ├── dynamic_scraper.py          # DynamicIRScraper — live web scraping
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

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/Wezo420/agentic-rag-financial-analytics.git
cd agentic-rag-financial-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\Activate    # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your-groq-api-key-here
```
Get a free Groq key at [console.groq.com](https://console.groq.com) — 14,400 free requests/day.

### 5. Run setup
```bash
python setup_and_run.py
```
This will:
- Download/simulate company filings (Tata Motors, HUL, Reliance)
- Parse and index all documents into ChromaDB
- Run a test query to verify the pipeline

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Open: **http://localhost:8501**

---

## 🌐 Deployment

### Streamlit Cloud
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agenticragfinance.streamlit.app)

Add this to Streamlit Cloud secrets:
```toml
GROQ_API_KEY = "your-groq-key-here"
```

### Render
The `render.yaml` file is included for one-click Render deployment. Add `GROQ_API_KEY` as an environment variable in the Render dashboard.

> **Note on Research Any Company:** Web scraping (BSE/NSE APIs + DuckDuckGo) is fully functional when running locally. Cloud platforms may restrict outbound requests — use the Document Upload tab as an alternative on hosted versions.

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

## 🏢 Pre-Indexed Companies

| Company | Ticker | Sector | Exchange |
|---------|--------|--------|----------|
| Tata Motors | TATAMOTORS.NS | Automobile | NSE |
| Hindustan Unilever (HUL) | HINDUNILVR.NS | FMCG | NSE |
| Reliance Industries | RELIANCE.NS | Conglomerate | NSE |

> Upload any PDF via the **Document Upload** tab or use **Research Any Company** (local mode) to add any company dynamically.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Agentic Orchestration | LangGraph (StateGraph) |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | ChromaDB built-in (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| PDF Parsing | pdfplumber + PyPDF2 |
| Web Scraping | requests + BeautifulSoup4 + DuckDuckGo Search |
| IR Data APIs | BSE India API + NSE India API |
| UI | Streamlit |
| PDF Generation | reportlab |
| Language | Python 3.10+ |

---

## 🔑 Key Design Decisions

### Dual-Collection RAG
Most RAG systems use a single vector store. This system uses two separate ChromaDB collections — `language_centric` and `financial_ownership` — ensuring narrative text never pollutes quantitative queries and vice versa.

### LangGraph Self-Correction Loop
Instead of a single retrieval pass, the agent decomposes queries, grades context quality, and self-corrects with refined queries for up to 5 iterations before synthesizing the final answer.

### Markdown-First LLM Output
The synthesizer prompt instructs the LLM to respond in structured markdown directly — avoiding JSON parsing fragility and ensuring clean, readable output every time.

---

## 📈 Potential Extensions

- **Hybrid BM25 + Dense Retrieval** for better keyword matching on financial figures
- **Cohere Rerank** integration before synthesis for higher precision
- **Multi-company cross-analysis** — compare two companies in one query
- **Live NSE/BSE price feed** integration for real-time data
- **PDF report export** — download the analysis as a formatted PDF

---

*Built as a sophisticated task/project for risk analytics — Dun & Bradstreet.*
