"""
retrievers/graph_builder.py
----------------------------
Single source of truth for ALL LangGraph graph assembly.

  ┌─────────────────────────────────────────────────────────────────┐
  │  build_graph(llm)  — ONE function, full pipeline                │
  │                                                                 │
  │  Phase 4 (internal) — ReAct retrieval agent:                   │
  │    agent_node → tool_node (loop)                               │
  │    Tools: retrieve / tavily_search / calculator                 │
  │    Retrieval: Dense (Qdrant) + BM25 → RRF → CrossEncoder       │
  │                                                                 │
  │  Phase 5 (returned) — Self-RAG verification wrapper:           │
  │    START                                                        │
  │      → extract_question_node                                    │
  │      → phase4_retrieve_node  (Phase 4 as an inner black-box)   │
  │      → generate_node         (first answer from chunks)         │
  │      → verify_node           (SUPPORTED / PARTIALLY / UNSUP.)  │
  │      → regenerate_node  ──┐  (up to MAX_RETRIES = 3 loops)     │
  │           ↑───────────────┘                                     │
  │      → END                                                      │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    from app.retrievers.graph_builder import build_graph

    llm = ChatGroq(model="qwen/qwen3-32b", ...)
    graph = build_graph(llm)          # returns compiled Phase 5 graph
    graph.invoke({...})

Extending:
  Add new tools  – update TOOL_REGISTRY in retriever_tool.py; picked up automatically.
  Phase 7        – inject memory_chunks into generate_node prompt.
  Phase 9        – prepend a planner_node before phase4_retrieve_node.
  Phase 10       – promote this graph to a sub-graph under a supervisor.
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


# ---------------------------------------------------------------------------
# Internal routing helper for the Phase 4 ReAct loop
# ---------------------------------------------------------------------------

def _should_continue(state: Phase4AgentState) -> Literal["tool_node", "__end__"]:
    """
    Route after every agent_node execution inside the Phase 4 ReAct loop.

      - Last message has tool_calls → route to tool_node (execute the tool).
      - Last message is plain text  → route to END (done retrieving).
    """
    messages = state["messages"]
    if not messages:
        return "__end__"

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        logger.debug("[Phase4] _should_continue → tool_node")
        return "tool_node"

    logger.debug("[Phase4] _should_continue → __end__")
    return "__end__"


# ---------------------------------------------------------------------------
# Phase 5 helpers — private, used only inside build_graph()
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3
"""Maximum verify → regenerate cycles before returning the best available answer."""


def _build_phase4_retrieve_node(phase4_graph: Any):
    """
    Wrap the compiled Phase 4 ReAct graph as a single Phase 5 node.

    Runs Phase 4 end-to-end, then extracts all retrieved chunks from the
    resulting ToolMessages (supports both `retrieve` and `tavily_search`
    output formats) and writes them to state["retrieved_chunks"].
    """

    def phase4_retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        question: str = state["question"]
        logger.info("[Phase5] phase4_retrieve_node — running for: %r", question[:80])

        try:
            phase4_output = phase4_graph.invoke(
                {"messages": [HumanMessage(content=question)]}
            )
        except Exception as exc:
            logger.error("[Phase5] Phase 4 graph failed: %s", exc)
            raise RuntimeError(f"Phase 4 retrieval failed: {exc}") from exc

        # Primary: explicit retrieved_chunks field (set if Phase 5 state was used).
        chunks: list[dict] = phase4_output.get("retrieved_chunks", [])

        # Fallback: parse ToolMessage JSON — covers the normal Phase 4 AgentState.
        if not chunks:
            chunks = _extract_chunks_from_tool_messages(
                phase4_output.get("messages", [])
            )
            if not chunks:
                logger.warning(
                    "[Phase5] phase4_retrieve_node — no chunks found; "
                    "proceeding with empty list."
                )

        logger.info("[Phase5] phase4_retrieve_node — %d chunks extracted.", len(chunks))
        return {"retrieved_chunks": chunks}

    return phase4_retrieve_node


def _extract_chunks_from_tool_messages(messages: list) -> list[dict]:
    """
    Parse chunks from every ToolMessage in the Phase 4 conversation.

    Handles two tool output formats:
      - retrieve tool:  list of {"text", "source", "page", "rerank_score"}
      - tavily_search:  list of {"title", "content", "url"}
                        → normalised to {"text", "source", "page"} for Phase 5

    Collects from ALL tool calls so the generator has the full evidence picture.
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

        if "text" in first:
            # retrieve tool format
            all_chunks.extend(data)
        elif "content" in first and "url" in first:
            # tavily_search format — normalise to retrieve format
            for result in data:
                all_chunks.append({
                    "text": (result.get("content") or result.get("title") or "").strip(),
                    "source": (result.get("url") or "").strip(),
                    "page": 0,
                    "rerank_score": 0.0,
                })

    return all_chunks


def _extract_question_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the user's question from messages into state["question"].

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


def _should_regenerate(
    state: dict[str, Any],
) -> Literal["regenerate_node", "__end__"]:
    """
    Route after every verify_node execution.

      → regenerate_node  if  retry_count < MAX_RETRIES  AND  verdict != SUPPORTED
      → END              if  SUPPORTED, or retries exhausted
    """
    retry_count: int = state.get("retry_count", 0)
    verdict_str: str = state.get("verification_result", {}).get(
        "verdict", Verdict.UNSUPPORTED.value
    )

    if verdict_str == Verdict.SUPPORTED.value:
        logger.info(
            "[Phase5] _should_regenerate → END (verdict=SUPPORTED, retry=%d)", retry_count
        )
        return "__end__"

    if retry_count >= MAX_RETRIES:
        logger.warning(
            "[Phase5] _should_regenerate → END (MAX_RETRIES=%d reached, verdict=%s)",
            MAX_RETRIES,
            verdict_str,
        )
        return "__end__"

    logger.info(
        "[Phase5] _should_regenerate → regenerate_node (verdict=%s, retry=%d/%d)",
        verdict_str,
        retry_count,
        MAX_RETRIES,
    )
    return "regenerate_node"


# =============================================================================
# Single public graph factory
# =============================================================================


def build_graph(llm: Any) -> Any:
    """
    Build and compile the complete RAG pipeline graph.

    Internally constructs two sub-graphs:
      Phase 4 — ReAct retrieval agent (agent_node ↔ tool_node loop).
                Tools: retrieve, tavily_search, calculator.
                This graph is compiled and captured as a local variable.
      Phase 5 — Self-RAG verification wrapper around Phase 4.
                Nodes: extract_question → phase4_retrieve → generate
                       → verify → (regenerate → verify)* → END

    Args:
        llm: A bare ChatGroq (or compatible) instance.
             Tool binding and structured-output wiring happen internally.

    Returns:
        Compiled Phase 5 LangGraph graph. Invoke with:
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
    # ------------------------------------------------------------------
    # Phase 4 — ReAct retrieval agent (internal, not returned)
    # ------------------------------------------------------------------
    llm_with_tools = llm.bind_tools(TOOL_REGISTRY)
    logger.info("[Phase4] LLM bound with tools: %s", [t.name for t in TOOL_REGISTRY])

    agent_node_fn = build_agent_node(llm_with_tools)
    tool_node = ToolNode(TOOL_REGISTRY)

    phase4_graph = StateGraph(Phase4AgentState)
    phase4_graph.add_node("agent_node", agent_node_fn)
    phase4_graph.add_node("tool_node", tool_node)
    phase4_graph.add_edge(START, "agent_node")
    phase4_graph.add_conditional_edges(
        "agent_node",
        _should_continue,
        {"tool_node": "tool_node", "__end__": END},
    )
    phase4_graph.add_edge("tool_node", "agent_node")

    compiled_phase4 = phase4_graph.compile()
    
    logger.info("[Phase4] ReAct graph compiled.")

    # ------------------------------------------------------------------
    # Phase 5 — Self-RAG verification wrapper (returned to caller)
    # ------------------------------------------------------------------
    phase4_retrieve_fn = _build_phase4_retrieve_node(compiled_phase4)
    generate_fn = build_generate_node(llm)
    verify_fn = build_verifier_node(llm)
    regenerate_fn = build_regenerate_node(llm)

    phase5_graph = StateGraph(Phase5AgentState)
    phase5_graph.add_node("extract_question_node", _extract_question_node)
    phase5_graph.add_node("phase4_retrieve_node", phase4_retrieve_fn)
    phase5_graph.add_node("generate_node", generate_fn)
    phase5_graph.add_node("verify_node", verify_fn)
    phase5_graph.add_node("regenerate_node", regenerate_fn)

    phase5_graph.add_edge(START, "extract_question_node")
    phase5_graph.add_edge("extract_question_node", "phase4_retrieve_node")
    phase5_graph.add_edge("phase4_retrieve_node", "generate_node")
    phase5_graph.add_edge("generate_node", "verify_node")
    phase5_graph.add_conditional_edges(
        "verify_node",
        _should_regenerate,
        {"regenerate_node": "regenerate_node", "__end__": END},
    )
    phase5_graph.add_edge("regenerate_node", "verify_node")

    compiled_phase5 = phase5_graph.compile()
    logger.info("[Phase5] Self-RAG graph compiled successfully.")
    return compiled_phase5