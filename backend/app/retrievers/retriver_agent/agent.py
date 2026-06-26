"""
agent.py
--------
Defines both the shared graph state (AgentState) and the ReAct agent
node (build_agent_node) for the LangGraph pipeline.

State + agent logic live together because they are tightly coupled:
the agent node is the only writer to the messages field, and the
state schema exists purely to serve this agent.

Extending for future phases:
  Phase 5  - add `verification_result: NotRequired[str]` to AgentState
  Phase 7  - add `memory_chunks: NotRequired[list[dict]]`
  Phase 9  - replace agent_node with a planner that injects subtasks first
  Phase 10 - promote to sub-agent; supervisor routes to it by name
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentState — shared graph state passed between every node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state passed between every node in the LangGraph graph.

    messages:
        Append-only list of LangChain BaseMessage objects.
        `add_messages` is a LangGraph reducer that merges lists on
        parallel branches and deduplicates by message ID, so nodes
        only ever need to return the *new* messages they produce.

    Future phases extend this TypedDict here — never in individual nodes —
    so the state schema stays as the single source of truth.
    """

    messages: Annotated[list, add_messages]

    # Phase 5  — Self-RAG verifier output
    verification_result: NotRequired[str]

    # Phase 7  — Long-term memory chunks
    memory_chunks: NotRequired[list[dict]]

    # Phase 8  — Research workspace identifier
    workspace_id: NotRequired[str]

    # Phase 9  — Decomposed subtask list from planner
    subtasks: NotRequired[list[str]]

    # Phase 10 — Active sub-agent in supervisor mode
    active_agent: NotRequired[str]


# ---------------------------------------------------------------------------
# Iteration guard
# ---------------------------------------------------------------------------

MAX_STEPS: int = 3
"""
Maximum number of tool-call rounds before the agent is forced to answer.
Prevents infinite retrieval loops on ambiguous or unanswerable queries.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


REACT_SYSTEM_PROMPT = """\
You are an expert research assistant with access to three tools:

1. retrieve(query)
2. tavily_search(query)
3. calculator(expression)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide accurate, evidence-based answers.

For technical, scientific, research, or document-related questions, rely on tool results rather than memory whenever possible.

For simple conversational questions such as greetings, introductions, or casual chat, you may answer directly without using tools.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

retrieve(query)

Use retrieve() when:

* answering questions about uploaded PDFs or indexed documents,
* explaining concepts, algorithms, datasets, methods, or models,
* comparing techniques,
* discussing machine learning, AI, mathematics, or research topics.

Examples:

* "What is LSTM?"
* "Explain GraphSAGE."
* "Compare BERT and RoBERTa."
* "What optimizer does the paper use?"

Default choice for technical questions: retrieve().

---

tavily_search(query)

Use tavily_search() when:

* recent information is required,
* external evidence is needed,
* retrieve() returned insufficient or irrelevant results,
* web validation is necessary.

Examples:

* "What GraphRAG papers appeared in 2025?"
* "Who won the Turing Award in 2024?"
* "Latest benchmark results for GPT-4o."

Prefer retrieve() first whenever reasonable.

---

calculator(expression)

Use calculator() for:

* arithmetic,
* percentages,
* statistics,
* logarithms,
* trigonometric functions,
* algebraic expressions.

Never perform calculations mentally.

---

directly llm usage when question is  greeting, introductions, or casual chat

Example when the query is like hi , hello , who are you?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thought:
What information do I need?

Action:
Call one appropriate tool.

Observation:
Read the returned information carefully.

Repeat if necessary.

Final Answer:
Synthesize all observations into a complete and well-supported response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* Do not answer technical or research questions solely from memory if evidence can be obtained through tools.

* If retrieved information is incomplete, gather more evidence before answering.

* Prefer specific queries over broad ones.

Bad:
retrieve("LSTM")

Better:
retrieve("definition and architecture of LSTM")

Bad:
tavily_search("GraphRAG")

Better:
tavily_search("GraphRAG papers published in 2025")

* Do not repeat the same query unnecessarily.

* Use information collected from previous tool calls.

* If evidence from multiple tools is useful, combine them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL ANSWER RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before producing the final answer, ensure:

1. Sufficient evidence has been collected.
2. Major aspects of the question are covered.
3. Claims are supported by observations.
4. Recent information is verified when necessary.
5. The answer is clear, concise, and well structured.

Keep your internal reasoning private.
Only reveal the final answer.

""".format(max_steps=MAX_STEPS)


# ---------------------------------------------------------------------------
# Agent node factory
# ---------------------------------------------------------------------------

def build_agent_node(llm_with_tools: Any):
    """
    Factory that closes over the tool-bound LLM and returns the
    `agent_node` function ready for graph.add_node().

    Using a factory (rather than a class or global) keeps the node
    function stateless and trivially testable in isolation.

    Args:
        llm_with_tools: A LangChain chat model with tools already bound
                        via `llm.bind_tools(tools)`.

    Returns:
        Callable[[AgentState], dict] — the node function.
    """

    def agent_node(state: AgentState) -> dict:
        """
        Single agent reasoning step.

        Flow:
          1. On the very first call inject the system prompt.
          2. Count tool messages in history to enforce MAX_STEPS.
          3. If limit reached, append a forced-answer instruction.
          4. Call the LLM and return the new AIMessage.

        Args:
            state: Current graph state (messages list).

        Returns:
            dict with key "messages" containing a list with one new AIMessage.
            LangGraph's `add_messages` reducer appends it to the existing list.
        """
        messages = state["messages"]

        # Step 1 — inject system prompt on first agent turn
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(messages)
            logger.debug("agent_node — system prompt injected.")

        # Step 2 — count completed tool-call rounds
        tool_call_rounds = sum(
            1
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        logger.info(
            "agent_node — tool_call_rounds so far: %d / %d",
            tool_call_rounds,
            MAX_STEPS,
        )

        # Step 3 — force answer if MAX_STEPS exceeded
        if tool_call_rounds >= MAX_STEPS:
            logger.warning(
                "agent_node — MAX_STEPS (%d) reached. Forcing final answer.", MAX_STEPS
            )
            messages = list(messages) + [
                HumanMessage(
                    content=(
                        f"You have already retrieved information {tool_call_rounds} times. "
                        "Do NOT call any more tools. "
                        "Produce your final answer now using only what you have collected."
                    )
                )
            ]

        # Step 4 — call the LLM
        try:
            response: AIMessage = llm_with_tools.invoke(messages)
            logger.info(
                "agent_node — LLM responded (tool_calls=%s).",
                bool(getattr(response, "tool_calls", None)),
            )
        except Exception as exc:
            logger.error("agent_node — LLM call failed: %s", exc)
            response = AIMessage(
                content=f"I encountered an error while processing your request: {exc}"
            )

        return {"messages": [response]}

    return agent_node
