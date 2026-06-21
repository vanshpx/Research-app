"""
generation/answer_generator.py
--------------------------------
Two LangGraph node factories for Phase 5 generation:

  build_generate_node()     — first-time answer from retrieved chunks
  build_regenerate_node()   — revised answer using verifier feedback

Both share the same underlying Groq LLM call pattern.
The separation keeps each node's prompt and responsibility distinct,
making it easy to tune them independently.

What these nodes do NOT do:
  - Run retrieval (Phase 4 already handles that).
  - Make routing decisions (the graph's conditional edge handles that).
  - Store anything beyond state fields they own.

Future phases:
  Phase 7 — generate_node can prepend memory_chunks to the evidence block.
  Phase 8 — workspace context can be injected into the system prompt.
  Phase 9 — planner can set a `generation_style` state field that
             these nodes read to adjust tone / depth.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.retrievers.self_verfier.generation_prompt import (
    GENERATION_SYSTEM_PROMPT,
    REGENERATION_SYSTEM_PROMPT,
    build_initial_generation_prompt,
    build_regeneration_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _call_llm(
    llm: ChatGroq,
    system_prompt: str,
    user_prompt: str,
    context_label: str,
) -> str:
    """
    Invoke the LLM with a system + user message pair.
    Returns the model's text response stripped of whitespace.
    """
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = str(response.content).strip()
        logger.info("%s — generated %d chars.", context_label, len(answer))
        return answer
    except Exception as exc:
        logger.error("%s — LLM call failed: %s", context_label, exc)
        raise RuntimeError(f"{context_label} generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# generate_node factory
# ---------------------------------------------------------------------------


def build_generate_node(llm: ChatGroq):
    """
    Factory returning the first-time answer generation node.

    Reads from state:
      - question
      - retrieved_chunks

    Writes to state:
      - answer           (new initial answer)
      - previous_answers (appends the new answer for the audit trail)

    Args:
        llm: Bare ChatGroq instance.

    Returns:
        Callable[[AgentState], dict]
    """

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        question: str = state.get("question", "")
        chunks: list[dict] = state.get("retrieved_chunks", [])

        logger.info(
            "generate_node — question length: %d | chunks: %d",
            len(question),
            len(chunks),
        )

        user_prompt = build_initial_generation_prompt(question, chunks)
        answer = _call_llm(llm, GENERATION_SYSTEM_PROMPT, user_prompt, "generate_node")

        # Seed the previous_answers list for the audit trail.
        previous_answers: list[str] = list(state.get("previous_answers", []))
        previous_answers.append(answer)

        return {
            "answer": answer,
            "previous_answers": previous_answers,
        }

    return generate_node


# ---------------------------------------------------------------------------
# regenerate_node factory
# ---------------------------------------------------------------------------


def build_regenerate_node(llm: ChatGroq):
    """
    Factory returning the answer regeneration node.

    Reads from state:
      - question
      - retrieved_chunks
      - answer           (the answer that failed verification)
      - feedback
      - unsupported_claims
      - missing_information
      - retry_count

    Writes to state:
      - answer           (improved answer replaces the previous one)
      - previous_answers (appends the new answer)
      - retry_count      (incremented by 1)

    Args:
        llm: Bare ChatGroq instance.

    Returns:
        Callable[[AgentState], dict]
    """

    def regenerate_node(state: dict[str, Any]) -> dict[str, Any]:
        question: str = state.get("question", "")
        chunks: list[dict] = state.get("retrieved_chunks", [])
        previous_answer: str = state.get("answer", "")
        feedback: str = state.get("feedback", "")
        unsupported_claims: list[str] = state.get("unsupported_claims", [])
        missing_information: list[str] = state.get("missing_information", [])
        retry_count: int = state.get("retry_count", 0)

        logger.info(
            "regenerate_node — retry %d | unsupported claims: %d",
            retry_count,
            len(unsupported_claims),
        )

        user_prompt = build_regeneration_prompt(
            question=question,
            chunks=chunks,
            previous_answer=previous_answer,
            feedback=feedback,
            unsupported_claims=unsupported_claims,
            missing_information=missing_information,
        )

        answer = _call_llm(
            llm, REGENERATION_SYSTEM_PROMPT, user_prompt, "regenerate_node"
        )

        # Append to audit trail and increment the retry counter.
        previous_answers: list[str] = list(state.get("previous_answers", []))
        previous_answers.append(answer)

        return {
            "answer": answer,
            "previous_answers": previous_answers,
            "retry_count": retry_count + 1,
        }

    return regenerate_node
