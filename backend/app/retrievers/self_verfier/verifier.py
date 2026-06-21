"""
verification/verifier.py
-------------------------
Self-RAG verifier node for Phase 5.

Responsibilities:
  - Call Groq (qwen/qwen3-32b) with structured output (VerificationResult).
  - Parse the result and write all verification fields into AgentState.
  - Be a pure LangGraph node: (state) → dict of updated state fields.

What this file does NOT do:
  - Make routing decisions (that is the conditional edge in phase5_graph.py).
  - Call the retriever or modify retrieved_chunks.
  - Know about retry limits (that is enforced in the graph).

Extending for future phases:
  Phase 9 — the planner can swap the verifier prompt per task type by
             injecting a `verification_strategy` field into state.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.retrievers.self_verfier.schemas import VerificationResult, Verdict
from app.retrievers.self_verfier.verifier_prompt import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_message,
)

logger = logging.getLogger(__name__)


def build_verifier_node(llm: ChatGroq):
    """
    Factory that returns the verifier_node function.

    Args:
        llm: A bare ChatGroq instance.
             `with_structured_output` is applied inside this factory
             so the caller passes the same raw LLM used everywhere else.

    Returns:
        Callable[[AgentState], dict] — LangGraph node function.
    """
    # Bind structured output once — reused across all invocations.
    structured_llm = llm.with_structured_output(VerificationResult)
    logger.info("Verifier structured LLM initialised (model: %s).", llm.model_name)

    def verifier_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        LangGraph node: verify the current answer against retrieved chunks.

        Reads from state:
          - question
          - retrieved_chunks
          - answer
          - retry_count (for logging only)

        Writes to state:
          - verification_result  (serialised VerificationResult dict)
          - feedback             (string for regeneration node)
          - unsupported_claims   (list for regeneration node)
          - missing_information  (list for regeneration node)

        Returns:
            Partial state dict — LangGraph merges it with existing state.
        """
        question: str = state.get("question", "")
        chunks: list[dict] = state.get("retrieved_chunks", [])
        answer: str = state.get("answer", "")
        retry_count: int = state.get("retry_count", 0)

        logger.info(
            "verifier_node — retry %d | answer length: %d chars | chunks: %d",
            retry_count,
            len(answer),
            len(chunks),
        )

        # ---------------------------------------------------------------- #
        # Guard: if there is no answer yet, mark as UNSUPPORTED immediately #
        # ---------------------------------------------------------------- #
        if not answer.strip():
            logger.warning("verifier_node — empty answer received, marking UNSUPPORTED.")
            fallback = VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                feedback="The answer is empty. Generate a substantive answer from the retrieved chunks.",
                unsupported_claims=[],
                missing_information=["A complete answer to the question."],
            )
            return _build_state_update(fallback)

        # ---------------------------------------------------------------- #
        # Build messages for the LLM                                        #
        # ---------------------------------------------------------------- #
        system_msg = SystemMessage(content=VERIFIER_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=build_verifier_user_message(question, chunks, answer)
        )

        # ---------------------------------------------------------------- #
        # Call Groq with structured output                                    #
        # ---------------------------------------------------------------- #
        try:
            result: VerificationResult = structured_llm.invoke([system_msg, user_msg])
            logger.info(
                "verifier_node — verdict: %s | unsupported claims: %d | confidence: %s",
                result.verdict.value,
                len(result.unsupported_claims),
                result.confidence_score,
            )
        except Exception as exc:
            logger.error("verifier_node — structured LLM call failed: %s", exc )
            # Fail safe: treat as PARTIALLY_SUPPORTED so the system retries
            # rather than either crashing or silently accepting a bad answer.
            result = VerificationResult(
                verdict=Verdict.PARTIALLY_SUPPORTED,
                feedback=(
                    f"Verification could not be completed due to an error: {exc}. "
                    "Please review the answer for unsupported claims."
                ),
                unsupported_claims=[],
                missing_information=[],
            )

        return _build_state_update(result)

    return verifier_node


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_state_update(result: VerificationResult) -> dict[str, Any]:
    """
    Convert a VerificationResult into the partial state dict that
    LangGraph will merge into the running AgentState.
    """
    return {
        "verification_result": result.to_state_dict(),
        "feedback": result.feedback,
        "unsupported_claims": result.unsupported_claims,
        "missing_information": result.missing_information,
    }
