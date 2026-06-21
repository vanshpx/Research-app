"""
react_agent.py
--------------
Public entry point for the agentic RAG pipeline.

Usage (from routes.py or any caller):

    from app.retrievers.retriver_agent.react_agent import run_agent

    answer, citations = run_agent("What is GraphSAGE?")

Internals:
  - Builds a LangGraph ReAct graph (agent_node ↔ tool_node loop).
  - Uses ChatGoogleGenerativeAI (LangChain wrapper) so the LLM can
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
from langchain_google_genai import ChatGoogleGenerativeAI

from app.retrievers.retriver_agent.graph_builder import build_graph

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled graph singleton — built once, reused across all requests.
# ---------------------------------------------------------------------------
_graph: Any = None


def _get_graph() -> Any:
    """Lazy singleton: builds and compiles the LangGraph agent graph."""
    global _graph
    if _graph is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add it to your .env file."
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )

        logger.info("[react_agent] Building LangGraph agent graph...")
        _graph = build_graph(llm)
        logger.info("[react_agent] Graph ready.")
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent(question: str) -> tuple[str, list[dict]]:
    """
    Run the ReAct agent for a given question.

    The agent will:
      1. Reason about the question (Thought).
      2. Call `retrieve` with a targeted query (Action).
      3. Observe the returned chunks (Observation).
      4. Repeat up to MAX_STEPS times, then produce a final answer.

    Args:
        question: Natural-language question from the user.

    Returns:
        Tuple of:
          - answer   (str)        The agent's final answer text.
          - citations (list[dict]) Unique source citations extracted from
                                   tool-call results (source + page).

    Raises:
        ValueError:  If GOOGLE_API_KEY is missing.
        RuntimeError: If the graph invocation fails unexpectedly.
    """
    graph = _get_graph()

    logger.info("[react_agent] Running agent for question: %r", question)

    try:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as exc:
        logger.error("[react_agent] Graph invocation failed: %s", exc)
        raise RuntimeError(f"Agent failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Extract the final answer from the last AIMessage                     #
    # ------------------------------------------------------------------ #
    messages = result.get("messages", [])
    answer = ""
    for msg in reversed(messages):
        # Find the last AIMessage that has plain text content (no tool_calls)
        from langchain_core.messages import AIMessage
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            # Gemini can return content as a list of blocks: [{'type': 'text', 'text': '...'}]
            # or as a plain string depending on the SDK version. Handle both.
            if isinstance(content, list):
                answer = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            else:
                answer = str(content).strip()
            break

    if not answer:
        answer = "I could not find relevant information in the uploaded documents."

    # ------------------------------------------------------------------ #
    # Build citations from ToolMessage results                             #
    # ------------------------------------------------------------------ #
    citations = _extract_citations(messages)

    logger.info(
        "[react_agent] Done. answer_length=%d, citations=%d",
        len(answer), len(citations),
    )
    return answer, citations


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
