"""
retriever_tool.py
-----------------
Wraps the existing hybrid retrieval pipeline as a LangChain tool so
that LangGraph's ToolNode can invoke it automatically when the LLM
emits a tool-call.

What this file does:
  - Defines `retrieve` as a @tool-decorated function.
  - Internally calls the real pipeline:
      retrieve() → BM25+Dense+RRF (retriever.py)
      reranker   → CrossEncoderReranker (cross_encoder_reranker.py)
  - Returns a JSON string (ToolNode expects string tool outputs).

What this file does NOT do:
  - Modify any retrieval logic.
  - Duplicate DenseRetriever, BM25Retriever, Reranker logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.retrievers.retriever import retrieve as _pipeline_retrieve
from app.retrievers.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-encoder reranker singleton
# Loaded once at import time; shared across all tool calls.
# ---------------------------------------------------------------------------
_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    """Lazy singleton for the cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        logger.info("[retriever_tool] Initialising CrossEncoderReranker singleton...")
        _reranker = CrossEncoderReranker()
        logger.info("[retriever_tool] CrossEncoderReranker ready.")
    return _reranker


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@tool
def retrieve(query: str) -> str:
    """
    Search the research document corpus using hybrid retrieval.

    Runs Dense (Qdrant) + BM25 retrieval, fuses results with
    Reciprocal Rank Fusion, and reranks with a cross-encoder to
    return the top-5 most relevant chunks.

    Args:
        query: A natural-language search string. Be specific —
               e.g. "GraphSAGE inductive learning aggregation"
               rather than "graph neural networks".

    Returns:
        A JSON-encoded list of up to 5 chunk dicts, each with:
          - text         (str)   chunk text
          - source       (str)   document filename
          - page         (int)   page number within the source
          - rerank_score (float) cross-encoder confidence score
    """
    logger.info("[retriever_tool] retrieve() called — query: %r", query)

    try:
        # Step 1: Dense + BM25 + RRF (returns top_k=10 to give reranker more to work with)
        chunks, steps = _pipeline_retrieve(query, top_k=10)
        logger.info(
            "[retriever_tool] pipeline returned %d chunks in %d step(s).",
            len(chunks), steps,
        )
    except Exception as exc:
        logger.error("[retriever_tool] retrieval pipeline failed: %s", exc)
        return json.dumps({"error": str(exc), "chunks": []})

    if not chunks:
        logger.warning("[retriever_tool] no chunks returned for query %r", query)
        return json.dumps([])

    # Step 2: Cross-encoder reranking → trim to top 5
    try:
        reranker = _get_reranker()
        chunks = reranker.rerank(query, chunks, top_k=5)
        logger.info("[retriever_tool] reranked to %d chunks.", len(chunks))
    except Exception as exc:
        logger.warning("[retriever_tool] reranker failed, using raw RRF order: %s", exc)
        chunks = chunks[:5]

    # Normalise to the documented output schema
    output: list[dict[str, Any]] = [
        {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "page": chunk.get("page", 0),
            "rerank_score": chunk.get("rerank_score", 0.0),
        }
        for chunk in chunks
    ]

    logger.info("[retriever_tool] returning %d chunks.", len(output))
    return json.dumps(output, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
# graph_builder.py reads this list to wire up ToolNode and bind tools to
# the LLM. To add a future external tool (web_search, arxiv, etc.):
#
#   from app.retrievers.retriver_agent.web_search_tool import web_search
#   TOOL_REGISTRY = [retrieve, web_search, ...]
#
# No other file needs to change.

TOOL_REGISTRY: list = [retrieve]
