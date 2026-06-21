"""
state/agent_state.py  (Phase 5 replacement — replace your Phase 4 version entirely)
--------------------------------------------------------------------------------------
Extends the Phase 4 messages-based state with all fields Phase 5 needs.

Phase 4 fields kept intact:
  messages        — LangGraph message list with add_messages reducer

Phase 5 fields added:
  question        — original user question (extracted from messages once)
  retrieved_chunks — flat list of chunks from the ReAct retrieval stage
  answer          — current best answer (mutated across retries)
  previous_answers — all prior answers kept for debugging / audit trail
  verification_result — last VerificationResult from the verifier node
  feedback        — actionable string extracted from verification_result
  unsupported_claims — list extracted from verification_result
  missing_information — list extracted from verification_result
  retry_count     — how many regeneration cycles have run

Future phases:
  Phase 7  — memory_chunks: NotRequired[list[dict]]
  Phase 8  — workspace_id: NotRequired[str]
  Phase 9  — subtasks:     NotRequired[list[str]]
  Phase 10 — active_agent: NotRequired[str]
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    # ------------------------------------------------------------------ #
    # Phase 4 — kept exactly as-is                                         #
    # ------------------------------------------------------------------ #
    messages: Annotated[list, add_messages]

    # ------------------------------------------------------------------ #
    # Phase 5 — verification pipeline state                                #
    # ------------------------------------------------------------------ #

    # Extracted from the initial HumanMessage so every node can read it
    # without scanning the messages list.
    question: str

    # Flat list of chunk dicts produced by the Phase 4 ReAct agent.
    # Schema per chunk: {text, source, page, rerank_score}
    retrieved_chunks: list[dict[str, Any]]

    # The current answer being evaluated / improved.
    answer: str

    # Every answer ever generated in this run — useful for debugging
    # and for Phase 9 planner comparisons.
    previous_answers: list[str]

    # Raw VerificationResult dict (serialised from the Pydantic model)
    # so it survives LangGraph state serialisation without issues.
    verification_result: dict[str, Any]

    # Flat fields surfaced from the latest verification_result for
    # convenient access by the regeneration node.
    feedback: str
    unsupported_claims: list[str]
    missing_information: list[str]

    # Counts completed verify → regenerate cycles.
    retry_count: int

    # ------------------------------------------------------------------ #
    # Future phase placeholders                                            #
    # ------------------------------------------------------------------ #
    memory_chunks: NotRequired[list[dict[str, Any]]]
    workspace_id: NotRequired[str]
    subtasks: NotRequired[list[str]]
    active_agent: NotRequired[str]
