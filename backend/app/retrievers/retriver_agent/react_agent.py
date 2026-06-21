"""
react_agent.py
--------------
Public entry point for the agentic RAG pipeline.

Usage (from routes.py or any caller):

    from app.retrievers.retriver_agent.react_agent import run_agent

    answer, citations = run_agent("What is GraphSAGE?")

Internals:
  - Builds a LangGraph ReAct graph (agent_node ↔ tool_node loop).
  - Uses ChatGroq (LangChain wrapper) so the LLM can
    emit structured tool-call objects that ToolNode understands.
  - The retrieve tool internally calls the existing pipeline:
      retriever.py (Dense + BM25 + RRF) → CrossEncoderReranker
  - The compiled graph is cached as a module-level singleton.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.retrievers.graph_builder import build_graph, build_phase5_graph

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled graph singleton — built once, reused across all requests.
# ---------------------------------------------------------------------------
_graph: Any = None


def _get_graph() -> Any:
    """Lazy singleton: builds and compiles the Phase 5 Self-RAG graph."""
    global _graph
    if _graph is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )

        llm = ChatGroq(
            model="qwen/qwen3-32b",
            groq_api_key=api_key,
            temperature=0.6,           # Qwen3 thinking models require > 0
            reasoning_effort="none",   # disable thinking chain for tool-call reliability
        )

        logger.info("[react_agent] Building Phase 4 retrieval graph...")
        phase4_graph = build_graph(llm)

        logger.info("[react_agent] Building Phase 5 Self-RAG graph...")
        _graph = build_phase5_graph(llm, phase4_graph)
        logger.info("[react_agent] Phase 5 graph ready.")
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent(question: str) -> tuple[str, list[dict]]:
    """
    Run the Phase 5 Self-RAG graph for a given question.

    The pipeline:
      Phase 4 (ReAct)  — dense + BM25 + RRF + CrossEncoder retrieval
      generate_node    — first answer from retrieved chunks
      verify_node      — Self-RAG grounding check (SUPPORTED / PARTIALLY / UNSUPPORTED)
      regenerate_node  — rewrite using verifier feedback (up to 3 retries)

    Args:
        question: Natural-language question from the user.

    Returns:
        Tuple of:
          - answer    (str)        The final verified answer.
          - citations (list[dict]) Unique source citations from retrieved chunks.

    Raises:
        ValueError:   If GROQ_API_KEY is missing.
        RuntimeError: If the graph invocation fails unexpectedly.
    """
    graph = _get_graph()

    logger.info("[react_agent] Running agent for question: %r", question)

    # Phase 5 AgentState requires all fields to be initialised.
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question": "",
        "retrieved_chunks": [],
        "answer": "",
        "previous_answers": [],
        "verification_result": {},
        "feedback": "",
        "unsupported_claims": [],
        "missing_information": [],
        "retry_count": 0,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("[react_agent] Graph invocation failed: %s", exc)
        raise RuntimeError(f"Agent failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Extract answer — Phase 5 writes it directly to state["answer"]      #
    # ------------------------------------------------------------------ #
    answer: str = result.get("answer", "").strip()

    # Fallback: scan messages if answer field is somehow empty
    if not answer:
        from langchain_core.messages import AIMessage
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                if isinstance(content, list):
                    answer = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ).strip()
                else:
                    answer = str(content).strip()
                if answer:
                    break

    if not answer:
        answer = "I could not find relevant information in the uploaded documents."

    # ------------------------------------------------------------------ #
    # Build citations from retrieved_chunks (Phase 5 state field)          #
    # ------------------------------------------------------------------ #
    chunks: list[dict] = result.get("retrieved_chunks", [])
    if chunks:
        citations = _citations_from_chunks(chunks)
    else:
        # Fallback: try to parse ToolMessages from Phase 4 messages
        citations = _extract_citations(result.get("messages", []))

    # Log verification outcome
    vr = result.get("verification_result", {})
    logger.info(
        "[react_agent] Done. verdict=%s retries=%d answer_len=%d citations=%d",
        vr.get("verdict", "unknown"),
        result.get("retry_count", 0),
        len(answer),
        len(citations),
    )
    return answer, citations


def _citations_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    Build deduplicated citations from Phase 5 state["retrieved_chunks"].
    Each chunk already has source, page, text from retriever_tool.py.
    """
    seen: set[tuple[str, int]] = set()
    citations: list[dict] = []
    for chunk in chunks:
        source = chunk.get("source", "")
        page = chunk.get("page", 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page,
                "snippet": chunk.get("text", "")[:200],
            })
    return citations


def _extract_citations(messages: list) -> list[dict]:
    """
    Parse ToolMessages in the conversation to extract unique source citations.

    Each ToolMessage content is a JSON string produced by retriever_tool.py.
    We collect unique (source, page) pairs and return them as citation dicts.

    Args:
        messages: Full message list from the agent graph result.

    Returns:
        List of dicts: [{"source": str, "page": int, "snippet": str}, ...]
    """
    from langchain_core.messages import ToolMessage

    seen: set[tuple[str, int]] = set()
    citations: list[dict] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            chunks = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle error responses from the tool
        if isinstance(chunks, dict) and "error" in chunks:
            continue

        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            source = chunk.get("source", "")
            page = chunk.get("page", 0)
            key = (source, page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    {
                        "source": source,
                        "page": page,
                        "snippet": chunk.get("text", "")[:200],
                    }
                )

    return citations
