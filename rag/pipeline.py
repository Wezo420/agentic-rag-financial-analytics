"""
rag/pipeline.py — LangGraph Agentic RAG Engine

Architecture:
                ┌─────────────────────────────────────────────┐
                │            LangGraph State Machine           │
                │                                             │
  User Query ──►│  PLAN → RETRIEVE → GRADE → REASON → ANSWER │
                │           ↑___SELF-CORRECT___↑             │
                │         (if context insufficient)           │
                └─────────────────────────────────────────────┘

Nodes:
  1. query_analyst    → Decomposes complex query into sub-queries
  2. dual_retriever   → Fetches from both language & financial collections
  3. context_grader   → Evaluates if retrieved context is sufficient
  4. knowledge_synthesizer → Generates structured financial analysis
  5. self_corrector   → Re-queries with refined search terms if needed
  6. final_responder  → Formats the final structured answer

State: AgentState (TypedDict) flows through the graph.
"""

import json
import logging
from typing import TypedDict, Annotated, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    OPENAI_API_KEY, LLM_MODEL, MAX_AGENT_ITERATIONS,
    TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


# ─── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""
    # Input
    original_query: str
    company: str
    filing_type_filter: Optional[str]

    # Planning
    sub_queries: list[str]
    query_intent: str           # e.g., "credit_risk", "expansion", "financials"

    # Retrieval
    retrieved_context: list[dict]
    retrieval_iteration: int

    # Grading
    context_sufficient: bool
    insufficiency_reason: str

    # Generation
    synthesized_answer: str
    source_citations: list[dict]
    confidence_level: str       # "high" / "medium" / "low"

    # Final output
    final_response: str
    risk_flags: list[str]
    key_metrics: dict
    


# ─── LLM Factory ──────────────────────────────────────────────────────────────
def _build_llm(temperature: float = 0.1):
    """Build LLM — tries Groq, then Gemini, then Mock."""
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Groq (free, very fast)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            logger.info("Using Groq LLM: llama-3.3-70b-versatile")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                api_key=groq_key,
            )
        except Exception as e:
            logger.warning(f"Groq LLM failed: {e}")

    # Gemini fallback
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Using Gemini LLM: gemini-1.5-flash-latest")
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=temperature,
                google_api_key=gemini_key,
            )
        except Exception as e:
            logger.warning(f"Gemini LLM failed: {e}")

    logger.warning("No valid API key — using Mock LLM")
    return _MockLLM()



class _MockLLM:
    """
    Deterministic mock LLM for demo/testing when no OpenAI key is present.
    Provides contextually-rich fake responses based on query keywords.
    """

    def invoke(self, messages) -> "_MockResponse":
        # Extract the last user message text
        if hasattr(messages, "__iter__"):
            msg_list = list(messages)
            query_text = ""
            for m in msg_list:
                if hasattr(m, "content"):
                    query_text += str(m.content) + " "
                elif isinstance(m, dict):
                    query_text += str(m.get("content", "")) + " "
        else:
            query_text = str(messages)

        query_lower = query_text.lower()
        response = self._generate_mock_response(query_lower)
        return _MockResponse(content=response)

    @staticmethod
    def _generate_mock_response(query: str) -> str:
        if "sub_queries" in query or "decompose" in query or "intent" in query:
            return json.dumps({
                "sub_queries": [
                    "What are the company's expansion plans and new business segments?",
                    "What is the current debt level and credit rating?",
                    "What are the key financial metrics — revenue, EBITDA, PAT?",
                    "What risk factors does management highlight?"
                ],
                "intent": "credit_risk_and_expansion_analysis"
            })

        if "sufficient" in query or "grade" in query or "context" in query:
            return json.dumps({
                "sufficient": True,
                "reason": "Retrieved context contains relevant financial data and strategic disclosures."
            })

        # Default: rich financial analysis response
        return json.dumps({
            "analysis": (
                "Based on the retrieved filings, the company demonstrates a mixed credit profile. "
                "Revenue growth has been robust at ~20% YoY, and EBITDA margins are expanding, "
                "suggesting operational leverage. However, the net debt position requires monitoring, "
                "particularly in the context of elevated CAPEX for expansion. The company's expansion "
                "into new retail formats and digital channels presents upside potential but also "
                "execution risk given the scale of investment."
            ),
            "risk_flags": [
                "Elevated net debt-to-EBITDA ratio above 2x",
                "Large CAPEX commitments reducing near-term free cash flow",
                "Commodity cost exposure (palmoil, steel, crude)",
                "Currency risk on international operations",
            ],
            "key_metrics": {
                "Revenue Growth YoY": "~20%",
                "EBITDA Margin": "~15%",
                "Net Debt/EBITDA": "~2.0x",
                "ROCE": "~10%",
                "Credit Rating": "AA- (CRISIL)",
            },
            "confidence": "medium",
        })


class _MockResponse:
    def __init__(self, content: str):
        self.content = content


# ─── Prompt Templates ─────────────────────────────────────────────────────────
QUERY_ANALYST_PROMPT = """You are a senior credit analyst AI assistant. Decompose the following complex financial query into 3-5 specific sub-queries optimized for document retrieval. Also identify the primary intent.

Query: {query}
Company: {company}

Respond ONLY in JSON:
{{
    "sub_queries": ["sub-query 1", "sub-query 2", ...],
    "intent": "one of: credit_risk | expansion_plans | financial_performance | ownership_structure | risk_factors | general"
}}"""

CONTEXT_GRADER_PROMPT = """You are a financial document grader. Evaluate if the retrieved context is SUFFICIENT to answer the original query with confidence.

Original Query: {query}
Retrieved Context (summary):
{context_summary}

Is the context sufficient? Consider: Are specific financial figures present? Is the relevant business segment covered?

Respond ONLY in JSON:
{{
    "sufficient": true/false,
    "reason": "brief explanation",
    "missing_aspects": ["list of what's missing if not sufficient"]
}}"""

SYNTHESIZER_PROMPT = """You are a Senior Credit Analyst and Financial Analyst. Using the retrieved context below, provide a comprehensive, structured analysis.

CRITICAL: Respond with ONLY valid JSON. No real newlines inside string values. Use \\n\\n for paragraph breaks within strings. No trailing commas.

Company: {company}
Original Query: {query}
Query Intent: {intent}

Retrieved Context:
{context}

Provide analysis in this EXACT JSON structure:
{{
    "executive_summary": "2-3 sentence overview",
    "analysis": "Detailed analysis addressing the query. Use \\n\\n for paragraph breaks.",
    "risk_flags": ["Risk 1", "Risk 2"],
    "key_metrics": {{"Metric Name": "Value"}},
    "expansion_outlook": "if relevant, else null",
    "credit_assessment": "Strong/Moderate/Cautious/Weak with rationale",
    "confidence": "high/medium/low",
    "data_gaps": ["gaps if any"]
}}"""

SELF_CORRECTOR_PROMPT = """The previous retrieval was INSUFFICIENT for the query: {query}
Missing aspects: {missing_aspects}

Generate 2-3 alternative/refined search queries to recover the missing information:
Respond ONLY in JSON: {{"refined_queries": ["query 1", "query 2", "query 3"]}}"""


# ─── Agent Nodes ───────────────────────────────────────────────────────────────
class FinancialRAGAgent:
    """
    LangGraph-based Agentic RAG with self-correction for financial analysis.
    Falls back to a manual state machine if langgraph is not installed.
    """

    def __init__(self, vector_store):
        self.vs = vector_store
        self.llm = _build_llm(temperature=0.1)
        self._graph = None
        self._build_graph()

    def _build_graph(self):
        """Attempt to build LangGraph StateGraph; fall back to manual pipeline."""
        try:
            from langgraph.graph import StateGraph, END
            self._graph = self._compile_langgraph(StateGraph, END)
            logger.info("LangGraph StateGraph compiled successfully.")
        except ImportError:
            logger.warning("langgraph not installed. Using sequential fallback pipeline.")
            self._graph = None

    def _compile_langgraph(self, StateGraph, END):
        """Build and compile the full LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("query_analyst", self._node_query_analyst)
        graph.add_node("dual_retriever", self._node_dual_retriever)
        graph.add_node("context_grader", self._node_context_grader)
        graph.add_node("self_corrector", self._node_self_corrector)
        graph.add_node("knowledge_synthesizer", self._node_knowledge_synthesizer)
        graph.add_node("final_responder", self._node_final_responder)

        # Define edges
        graph.set_entry_point("query_analyst")
        graph.add_edge("query_analyst", "dual_retriever")
        graph.add_edge("dual_retriever", "context_grader")

        # Conditional: sufficient → synthesize, else → self-correct (up to max)
        graph.add_conditional_edges(
            "context_grader",
            self._route_after_grading,
            {
                "synthesize": "knowledge_synthesizer",
                "self_correct": "self_corrector",
                "force_synthesize": "knowledge_synthesizer",
            },
        )

        graph.add_edge("self_corrector", "dual_retriever")
        graph.add_edge("knowledge_synthesizer", "final_responder")
        graph.add_edge("final_responder", END)

        return graph.compile()

    # ── Node: Query Analyst ────────────────────────────────────────────────────
    def _node_query_analyst(self, state: AgentState) -> dict:
        logger.info("[Node: query_analyst]")
        prompt = QUERY_ANALYST_PROMPT.format(
            query=state["original_query"],
            company=state["company"],
        )
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except ImportError:
            response = self.llm.invoke(prompt)

        try:
            parsed = json.loads(response.content)
            sub_queries = parsed.get("sub_queries", [state["original_query"]])
            intent = parsed.get("intent", "general")
        except (json.JSONDecodeError, AttributeError):
            sub_queries = [state["original_query"]]
            intent = "general"

        return {
            "sub_queries": sub_queries,
            "query_intent": intent,
            "retrieval_iteration": 0,
        }

    # ── Node: Dual Retriever ───────────────────────────────────────────────────
    def _node_dual_retriever(self, state: AgentState) -> dict:
        logger.info(f"[Node: dual_retriever] iteration={state.get('retrieval_iteration', 0)}")

        all_chunks: list[dict] = []
        seen_ids = set()

        # Use sub_queries for first pass, refined_queries for subsequent
        queries = state.get("refined_queries", state.get("sub_queries", [state["original_query"]]))

        for q in queries:
            results = self.vs.query(
                query_text=q,
                company=state.get("company"),
                filing_type=state.get("filing_type_filter"),
                top_k=TOP_K_RETRIEVAL,
            )
            for r in results:
                uid = r["metadata"].get("chunk_id", r["text"][:50])
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    all_chunks.append(r)

        # Deduplicated, scored results
        all_chunks.sort(key=lambda x: x.get("distance", 1.0))

        return {
            "retrieved_context": all_chunks[:TOP_K_RETRIEVAL * 2],
            "retrieval_iteration": state.get("retrieval_iteration", 0) + 1,
        }

    # ── Node: Context Grader ───────────────────────────────────────────────────
    def _node_context_grader(self, state: AgentState) -> dict:
        logger.info("[Node: context_grader]")
        context = state.get("retrieved_context", [])

        if not context:
            return {
                "context_sufficient": False,
                "insufficiency_reason": "No documents retrieved.",
            }

        # Quick heuristic: if low relevance scores, flag as insufficient
        avg_relevance = sum(c.get("relevance_score", 0) for c in context) / len(context)
        if avg_relevance < SIMILARITY_THRESHOLD and state.get("retrieval_iteration", 1) <= 1:
            return {
                "context_sufficient": False,
                "insufficiency_reason": f"Average relevance score {avg_relevance:.3f} below threshold.",
            }

        # LLM-based grading
        context_summary = "\n---\n".join(
            f"[{c['content_type']}] {c['text'][:300]}..." for c in context[:5]
        )
        prompt = CONTEXT_GRADER_PROMPT.format(
            query=state["original_query"],
            context_summary=context_summary,
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except ImportError:
            response = self.llm.invoke(prompt)

        try:
            parsed = json.loads(response.content)
            sufficient = parsed.get("sufficient", True)
            reason = parsed.get("reason", "")
            missing = parsed.get("missing_aspects", [])
        except (json.JSONDecodeError, AttributeError):
            sufficient = True
            reason = "Could not parse grader response; proceeding."
            missing = []

        return {
            "context_sufficient": sufficient,
            "insufficiency_reason": reason,
            "missing_aspects": missing,
        }

    # ── Routing Function ───────────────────────────────────────────────────────
    def _route_after_grading(self, state: AgentState) -> str:
        iteration = state.get("retrieval_iteration", 0)
        if state.get("context_sufficient", True):
            return "synthesize"
        elif iteration >= MAX_AGENT_ITERATIONS:
            logger.warning(f"Max iterations ({MAX_AGENT_ITERATIONS}) reached. Forcing synthesis.")
            return "force_synthesize"
        else:
            return "self_correct"

    # ── Node: Self-Corrector ───────────────────────────────────────────────────
    def _node_self_corrector(self, state: AgentState) -> dict:
        logger.info("[Node: self_corrector]")
        missing = state.get("missing_aspects", [])
        prompt = SELF_CORRECTOR_PROMPT.format(
            query=state["original_query"],
            missing_aspects=", ".join(missing),
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except ImportError:
            response = self.llm.invoke(prompt)

        try:
            parsed = json.loads(response.content)
            refined = parsed.get("refined_queries", state["sub_queries"])
        except (json.JSONDecodeError, AttributeError):
            refined = state["sub_queries"]

        logger.info(f"  Refined queries: {refined}")
        return {"refined_queries": refined}

    # ── Node: Knowledge Synthesizer ───────────────────────────────────────────
    def _node_knowledge_synthesizer(self, state: AgentState) -> dict:
        logger.info("[Node: knowledge_synthesizer]")
        context = state.get("retrieved_context", [])

        fin_ctx = [c for c in context if c.get("content_type") == "financial_ownership"]
        lang_ctx = [c for c in context if c.get("content_type") == "language_centric"]

        context_str = "=== FINANCIAL DATA ===\n"
        for c in fin_ctx[:4]:
            meta = c.get("metadata", {})
            context_str += f"[{meta.get('filing_type','?')} | {meta.get('year','?')}]\n{c['text'][:600]}\n\n"

        context_str += "=== STRATEGIC NARRATIVE ===\n"
        for c in lang_ctx[:4]:
            meta = c.get("metadata", {})
            context_str += f"[{meta.get('filing_type','?')} | {meta.get('year','?')}]\n{c['text'][:600]}\n\n"

        prompt = SYNTHESIZER_PROMPT.format(
            company=state["company"],
            query=state["original_query"],
            intent=state.get("query_intent", "general"),
            context=context_str,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a Senior Credit Analyst. Ground your analysis in the provided documents. Be precise with financial figures."),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
        except ImportError:
            response = self.llm.invoke(prompt)

        raw = response.content if hasattr(response, "content") else str(response)

        # Try to parse JSON and format as markdown
        import json
        try:
            # Strip markdown code fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            parsed = json.loads(clean)

            parts = []
            if parsed.get("executive_summary"):
                parts.append(f"### Executive Summary\n{parsed['executive_summary']}\n")
            if parsed.get("analysis"):
                parts.append(f"### Detailed Analysis\n{parsed['analysis']}\n")
            if parsed.get("credit_assessment"):
                parts.append(f"### 🏦 Credit Assessment\n{parsed['credit_assessment']}\n")
            if parsed.get("expansion_outlook") and parsed["expansion_outlook"] != "null":
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

            formatted = "\n".join(parts)
            risk_flags = parsed.get("risk_flags", [])
            key_metrics = parsed.get("key_metrics", {})
            confidence = parsed.get("confidence", "medium")

        except Exception:
            # LLM returned plain text — use as-is
            formatted = raw
            risk_flags = []
            key_metrics = {}
            confidence = "medium"

        # Build source citations
        sources = []
        for c in context[:6]:
            m = c.get("metadata", {})
            src = {
                "company": m.get("company", ""),
                "filing_type": m.get("filing_type", ""),
                "year": m.get("year", ""),
                "page": m.get("page_numbers", "?"),
                "content_type": c.get("content_type", ""),
                "relevance": round(c.get("relevance_score", 0), 3),
            }
            if src not in sources:
                sources.append(src)

        return {
            "synthesized_answer": formatted,
            "source_citations": sources,
            "confidence_level": confidence if isinstance(confidence, str) else "medium",
            "risk_flags": risk_flags,
            "key_metrics": key_metrics,
        }

    # ── Node: Final Responder ──────────────────────────────────────────────────
    def _node_final_responder(self, state: AgentState) -> dict:
        logger.info("[Node: final_responder]")
        analysis = state.get("synthesized_answer", "")
        risk_flags = state.get("risk_flags", [])
        key_metrics = state.get("key_metrics", {})
        confidence = state.get("confidence_level", "medium")
        iterations = state.get("retrieval_iteration", 1)
        sources = state.get("source_citations", [])

        parts = []
        parts.append(f"## 📊 Financial Intelligence Report: {state['company']}")
        parts.append(f"**Query:** {state['original_query']}\n")
        parts.append(analysis)
        parts.append(
            f"\n---\n*Confidence: **{confidence.upper()}** | "
            f"Retrieval Iterations: {iterations} | "
            f"Sources: {len(sources)}*"
        )

        return {
            "final_response": "\n".join(parts),
            "risk_flags": risk_flags,
            "key_metrics": key_metrics,
            "confidence_level": confidence,
        }

    # ── Public Query Interface ─────────────────────────────────────────────────
    def query(
        self,
        question: str,
        company: str,
        filing_type_filter: Optional[str] = None,
    ) -> dict:
        """
        Main entry point. Runs the agent and returns a result dict.
        Compatible with both LangGraph and fallback sequential mode.
        """
        self.llm = _build_llm(temperature=0.1)
        initial_state: AgentState = {
            "original_query": question,
            "company": company,
            "filing_type_filter": filing_type_filter,
            "sub_queries": [],
            "query_intent": "general",
            "retrieved_context": [],
            "retrieval_iteration": 0,
            "context_sufficient": False,
            "insufficiency_reason": "",
            "synthesized_answer": "",
            "source_citations": [],
            "confidence_level": "medium",
            "final_response": "",
            "risk_flags": [],
            "key_metrics": {},
        }

        if self._graph:
            try:
                final_state = self._graph.invoke(initial_state)
            except Exception as e:
                logger.error(f"LangGraph execution error: {e}. Falling back to sequential.")
                final_state = self._run_sequential(initial_state)
        else:
            final_state = self._run_sequential(initial_state)

        return {
            "response": final_state.get("final_response", ""),
            "risk_flags": final_state.get("risk_flags", []),
            "key_metrics": final_state.get("key_metrics", {}),
            "sources": final_state.get("source_citations", []),
            "confidence": final_state.get("confidence_level", "medium"),
            "iterations": final_state.get("retrieval_iteration", 1),
        }

    def _run_sequential(self, state: AgentState) -> AgentState:
        """Sequential fallback pipeline (no langgraph dependency)."""
        state.update(self._node_query_analyst(state))
        state.update(self._node_dual_retriever(state))
        state.update(self._node_context_grader(state))

        # Allow up to 2 self-correction cycles
        for _ in range(2):
            if state.get("context_sufficient", True) or state.get("retrieval_iteration", 0) >= MAX_AGENT_ITERATIONS:
                break
            state.update(self._node_self_corrector(state))
            state.update(self._node_dual_retriever(state))
            state.update(self._node_context_grader(state))

        state.update(self._node_knowledge_synthesizer(state))
        state.update(self._node_final_responder(state))
        return state
