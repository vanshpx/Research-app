"""
retrievers/graph_builder.py
----------------------------
Single source of truth for ALL LangGraph graph assembly.

This file contains two graph factories:

  ┌─────────────────────────────────────────────────────────────────┐
  │  PHASE 4 — build_graph(llm)                                     │
  │                                                                 │
  │  ReAct retrieval agent:                                         │
  │    START → agent_node → tool_node (loop) → END                 │
  │                                                                 │
  │  Retrieval pipeline per tool call:                              │
  │    Dense (Qdrant) + BM25 → RRF fusion → CrossEncoder rerank    │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  PHASE 5 — build_phase5_graph(llm, phase4_graph)               │
  │                                                                 │
  │  Self-RAG verification wrapper around Phase 4:                  │
  │    START                                                        │
  │      → extract_question_node                                    │
  │      → phase4_retrieve_node  (runs Phase 4 as a black-box)     │
  │      → generate_node         (first answer from chunks)         │
  │      → verify_node           (SUPPORTED / PARTIALLY / UNSUP.)  │
  │      → regenerate_node  ──┐  (up to MAX_RETRIES = 3 loops)     │
  │           ↑───────────────┘                                     │
  │      → END                                                      │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    from app.retrievers.graph_builder import build_graph, build_phase5_graph

    llm = ChatGroq(model="qwen/qwen3-32b", ...)
    phase4 = build_graph(llm)
    phase5 = build_phase5_graph(llm, phase4)

Extending:
  Phase 6  – add new tools to TOOL_REGISTRY; ToolNode picks them up automatically.
  Phase 7  – inject memory_chunks into generate_node prompt.
  Phase 9  – prepend a planner_node before phase4_retrieve_node.
  Phase 10 – promote this graph to a sub-graph under a supervisor.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


from app.retrievers.retriver_agent.agent import AgentState as Phase4AgentState
from app.retrievers.retriver_agent.agent import build_agent_node
from app.retrievers.retriver_agent.retriever_tool import TOOL_REGISTRY
from app.retrievers.self_verfier.agent_state import AgentState as Phase5AgentState
from app.retrievers.self_verfier.answer_generator import (
    build_generate_node,
    build_regenerate_node,
)
from app.retrievers.self_verfier.schemas import Verdict
from app.retrievers.self_verfier.verifier import build_verifier_node

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 4 — ReAct Retrieval Agent
# =============================================================================


def should_continue(state: Phase4AgentState) -> Literal["tool_node", "__end__"]:
    """
    [Phase 4] Routing function called after every agent_node execution.

    Logic:
      - If the last AIMessage contains tool_calls → route to tool_node.
      - Otherwise (plain text answer) → route to END.

    The MAX_STEPS guard lives inside agent_node (agent.py), which forces
    a plain-text answer message when the step limit is reached — causing
    this function to naturally route to END.
    """
    messages = state["messages"]
    if not messages:
        return "__end__"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        logger.debug("[Phase4] should_continue → tool_node")
        return "tool_node"

    logger.debug("[Phase4] should_continue → __end__")
    return "__end__"


def build_graph(llm: Any) -> Any:
    """
    [Phase 4] Construct and compile the ReAct retrieval agent graph.

    Graph topology:
        START → agent_node ──(has tool_calls)──→ tool_node → agent_node (loop)
                           ──(plain answer)────→ END

    Args:
        llm: A bare LangChain chat model (e.g. ChatGroq).
             Tools are bound inside this function.

    Returns:
        Compiled LangGraph graph. Invoke with:
            graph.invoke({"messages": [HumanMessage(content=query)]})
    """
    # 1. Bind tools so the LLM can emit structured tool-call objects.
    llm_with_tools = llm.bind_tools(TOOL_REGISTRY)
    logger.info("[Phase4] LLM bound with tools: %s", [t.name for t in TOOL_REGISTRY])

    # 2. Build nodes.
    agent_node_fn = build_agent_node(llm_with_tools)
    tool_node = ToolNode(TOOL_REGISTRY)

    # 3. Assemble graph.
    graph = StateGraph(Phase4AgentState)
    graph.add_node("agent_node", agent_node_fn)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges(
        "agent_node",
        should_continue,
        {"tool_node": "tool_node", "__end__": END},
    )
    graph.add_edge("tool_node", "agent_node")

    # 4. Compile.
    compiled = graph.compile()
    logger.info("[Phase4] Graph compiled successfully.")
    return compiled


# =============================================================================
# PHASE 5 — Self-RAG Verification Wrapper
# =============================================================================

MAX_RETRIES: int = 3
"""Maximum verify → regenerate cycles before returning the best available answer."""


# ---------------------------------------------------------------------------
# Phase 4 wrapper node
# ---------------------------------------------------------------------------


def _build_phase4_retrieve_node(phase4_graph: Any):
    """
    [Phase 5] Wrap the Phase 4 compiled graph as a single LangGraph node.

    Invokes Phase 4 and extracts:
      - retrieved_chunks: from state["retrieved_chunks"] if Phase 5 AgentState
        was used, otherwise parsed from ToolMessage JSON (fallback).

    Args:
        phase4_graph: Compiled Phase 4 graph (from build_graph()).

    Returns:
        Callable[[Phase5AgentState], dict]
    """

    def phase4_retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        question: str = state["question"]
        logger.info("[Phase5] phase4_retrieve_node — running for: %r", question[:80])

        phase4_input = {"messages": [HumanMessage(content=question)]}
        try:
            phase4_output = phase4_graph.invoke(phase4_input)
        except Exception as exc:
            logger.error("[Phase5] Phase 4 graph failed: %s", exc)
            raise RuntimeError(f"Phase 4 retrieval failed: {exc}") from exc

        # Primary: Phase 4 state already has retrieved_chunks (Phase 5 schema).
        chunks: list[dict] = phase4_output.get("retrieved_chunks", [])

        # Fallback: parse JSON from the last ToolMessage.
        if not chunks:
            chunks = _extract_chunks_from_tool_messages(
                phase4_output.get("messages", [])
            )
            if not chunks:
                logger.warning(
                    "[Phase5] phase4_retrieve_node — no chunks found. "
                    "Proceeding with empty list."
                )

        logger.info("[Phase5] phase4_retrieve_node — %d chunks extracted.", len(chunks))
        return {"retrieved_chunks": chunks}

    return phase4_retrieve_node


def _extract_chunks_from_tool_messages(messages: list) -> list[dict]:
    """
    [Phase 5] Fallback: parse chunks from ALL ToolMessages in the conversation.

    Handles two tool output formats:
      - retrieve tool:    list of {"text", "source", "page", "rerank_score"}
      - tavily_search:    list of {"title", "content", "url"}
                         → normalised to {"text", "source", "page"} for Phase 5

    Merges results from every tool call so the generate_node has the
    complete evidence picture, not just the last retrieval round.
    """
    from langchain_core.messages import ToolMessage

    all_chunks: list[dict] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(data, list) or not data:
            continue

        first = data[0]
        if not isinstance(first, dict):
            continue

        # --- retrieve tool format: has "text" key ---
        if "text" in first:
            all_chunks.extend(data)

        # --- tavily_search format: has "content" + "url" keys ---
        elif "content" in first and "url" in first:
            for result in data:
                all_chunks.append({
                    "text": (result.get("content") or result.get("title") or "").strip(),
                    "source": (result.get("url") or "").strip(),
                    "page": 0,          # web results have no page number
                    "rerank_score": 0.0,
                })

    return all_chunks


# ---------------------------------------------------------------------------
# Extract question node
# ---------------------------------------------------------------------------


def _extract_question_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    [Phase 5] Pull the user's question from messages into state["question"].

    Runs once at graph entry so all downstream nodes can read
    state["question"] directly without scanning the messages list.
    """
    messages = state.get("messages", [])
    question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        logger.warning("[Phase5] extract_question_node — no HumanMessage found.")

    logger.info("[Phase5] extract_question_node — question: %r", question[:80])
    return {"question": question}


# ---------------------------------------------------------------------------
# Conditional edge: should_regenerate
# ---------------------------------------------------------------------------


def _should_regenerate(
    state: dict[str, Any],
) -> Literal["regenerate_node", "__end__"]:
    """
    [Phase 5] Route after every verify_node execution.

    → regenerate_node  if  retry_count < MAX_RETRIES  AND  verdict != SUPPORTED
    → END              otherwise (SUPPORTED, or retries exhausted)
    """
    retry_count: int = state.get("retry_count", 0)
    verdict_str: str = state.get("verification_result", {}).get(
        "verdict", Verdict.UNSUPPORTED.value
    )

    if verdict_str == Verdict.SUPPORTED.value:
        logger.info(
            "[Phase5] should_regenerate → END (verdict=SUPPORTED, retry=%d)", retry_count
        )
        return "__end__"

    if retry_count >= MAX_RETRIES:
        logger.warning(
            "[Phase5] should_regenerate → END (MAX_RETRIES=%d reached, verdict=%s)",
            MAX_RETRIES,
            verdict_str,
        )
        return "__end__"

    logger.info(
        "[Phase5] should_regenerate → regenerate_node (verdict=%s, retry=%d/%d)",
        verdict_str,
        retry_count,
        MAX_RETRIES,
    )
    return "regenerate_node"


# ---------------------------------------------------------------------------
# Phase 5 graph factory
# ---------------------------------------------------------------------------


def build_phase5_graph(llm: ChatGroq, phase4_graph: Any) -> Any:
    """
    [Phase 5] Construct and compile the Self-RAG verification graph.

    Graph topology:
        START
          → extract_question_node
          → phase4_retrieve_node   (Phase 4 ReAct agent as black-box)
          → generate_node          (first answer from retrieved chunks)
          → verify_node            (Self-RAG: SUPPORTED / PARTIALLY / UNSUPPORTED)
          → regenerate_node ──────→ verify_node  (loop, up to MAX_RETRIES)
          → END

    Args:
        llm:          Bare ChatGroq instance.
                      Shared across generate, regenerate, and verify nodes.
        phase4_graph: Compiled Phase 4 graph (from build_graph()).

    Returns:
        Compiled LangGraph graph. Invoke with:
            graph.invoke({
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
            })
    """
    # Build node functions from factories.
    phase4_retrieve_fn = _build_phase4_retrieve_node(phase4_graph)
    generate_fn = build_generate_node(llm)
    verify_fn = build_verifier_node(llm)
    regenerate_fn = build_regenerate_node(llm)

    # Assemble graph.
    graph = StateGraph(Phase5AgentState)

    graph.add_node("extract_question_node", _extract_question_node)
    graph.add_node("phase4_retrieve_node", phase4_retrieve_fn)
    graph.add_node("generate_node", generate_fn)
    graph.add_node("verify_node", verify_fn)
    graph.add_node("regenerate_node", regenerate_fn)

    # Linear entry path.
    graph.add_edge(START, "extract_question_node")
    graph.add_edge("extract_question_node", "phase4_retrieve_node")
    graph.add_edge("phase4_retrieve_node", "generate_node")
    graph.add_edge("generate_node", "verify_node")

    # Verification decision point.
    graph.add_conditional_edges(
        "verify_node",
        _should_regenerate,
        {"regenerate_node": "regenerate_node", "__end__": END},
    )

    # Regeneration always feeds back to verification.
    graph.add_edge("regenerate_node", "verify_node")

    compiled = graph.compile()
    logger.info("[Phase5] Self-RAG graph compiled successfully.")
    return compiled
