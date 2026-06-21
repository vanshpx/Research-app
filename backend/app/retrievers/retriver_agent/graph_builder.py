"""
graph/graph_builder.py
-----------------------
Assembles the LangGraph StateGraph for Phase 4 Agentic RAG.

Graph topology:

    START
      ↓
    agent_node          ← reasons; emits tool-calls or a final answer
      ↓ (conditional)
    ┌───────────────────────────┐
    │                           │
  tool_node (ToolNode)        END
  (runs retrieve, etc.)
      ↓
    agent_node  ←── loop until answer or MAX_STEPS

Key design choices:
  - ToolNode is used as specified.  It handles tool dispatch, result
    wrapping into ToolMessage, and error handling automatically.
  - The conditional edge `should_continue` is the only routing logic;
    it mirrors the standard LangGraph agent pattern.
  - The MAX_STEPS guard lives inside agent_node (agents/agent.py) so
    the graph topology itself stays simple and phase-agnostic.

Extending for future phases:
  Phase 5 – insert a `verify_node` between tool_node and agent_node.
  Phase 6 – add new tools to TOOL_REGISTRY; ToolNode picks them up
             automatically; no graph changes needed.
  Phase 9 – prepend a `planner_node` before agent_node.
  Phase 10 – wrap this graph as a sub-graph; supervisor routes to it.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.retrievers.retriver_agent.agent import AgentState, build_agent_node
from app.retrievers.retriver_agent.retriever_tool import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """
    Routing function called after every agent_node execution.

    Logic:
      - If the last AIMessage contains tool_calls → route to tool_node.
      - Otherwise (plain text answer) → route to END.

    This single function is the only routing logic in the graph.
    The MAX_STEPS guard is enforced inside agent_node by injecting a
    forced-answer instruction before the LLM call, which causes the LLM
    to return a plain text message with no tool_calls — so this function
    naturally routes to END without any extra logic here.

    Args:
        state: Current graph state.

    Returns:
        "tool_node" or "__end__"
    """
    messages = state["messages"]
    if not messages:
        return "__end__"

    last_message = messages[-1]

    # AIMessage with tool_calls → the agent wants to call a tool.
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        logger.debug("should_continue → tool_node")
        return "tool_node"

    # Any other message (plain AIMessage answer, error fallback) → done.
    logger.debug("should_continue → __end__")
    return "__end__"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(llm: Any) -> Any:
    """
    Construct and compile the Phase 4 LangGraph StateGraph.

    Args:
        llm: An uninstrumented LangChain chat model (e.g. ChatGoogleGenerativeAI).
             Tools are bound inside this function so the caller doesn't
             need to know about the tool registry.

    Returns:
        A compiled LangGraph graph (CompiledStateGraph) ready to call
        via `.invoke({"messages": [HumanMessage(content=query)]})`.

    Usage example:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from graph.graph_builder import build_graph

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        graph = build_graph(llm)
        result = graph.invoke({"messages": [HumanMessage(content="What is GraphSAGE?")]})
        print(result["messages"][-1].content)
    """

    # ---------------------------------------------------------------- #
    # 1. Bind tools to the LLM                                          #
    # ---------------------------------------------------------------- #
    # `bind_tools` tells the model what tools are available so it can
    # emit structured tool-call objects in its responses.
    llm_with_tools = llm.bind_tools(TOOL_REGISTRY)
    logger.info(
        "LLM bound with tools: %s",
        [t.name for t in TOOL_REGISTRY],
    )

    # ---------------------------------------------------------------- #
    # 2. Build nodes                                                     #
    # ---------------------------------------------------------------- #
    agent_node_fn = build_agent_node(llm_with_tools)

    # ToolNode automatically:
    #   - Inspects the last AIMessage for tool_calls.
    #   - Dispatches each call to the matching tool by name.
    #   - Wraps results in ToolMessage objects.
    #   - Handles per-tool errors gracefully.
    tool_node = ToolNode(TOOL_REGISTRY)

    # ---------------------------------------------------------------- #
    # 3. Assemble the graph                                             #
    # ---------------------------------------------------------------- #
    graph = StateGraph(AgentState)

    graph.add_node("agent_node", agent_node_fn)
    graph.add_node("tool_node", tool_node)

    # Entry point: always start with the agent.
    graph.add_edge(START, "agent_node")

    # After agent_node: decide whether to call a tool or finish.
    graph.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "__end__": END,
        },
    )

    # After tool_node: always return to the agent for the next reasoning step.
    graph.add_edge("tool_node", "agent_node")

    # ---------------------------------------------------------------- #
    # 4. Compile                                                         #
    # ---------------------------------------------------------------- #
    compiled = graph.compile()
    logger.info("LangGraph Phase 4 graph compiled successfully.")
    return compiled
