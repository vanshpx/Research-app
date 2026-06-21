"""
verification/schemas.py
------------------------
Pydantic models for the Self-RAG verification layer.

These are the ONLY structured output contracts in Phase 5.
All other modules import from here — nothing defines its own schema.

Verdict semantics
-----------------
SUPPORTED
    Every material claim in the answer is directly backed by at least
    one retrieved chunk.  No hallucinations detected.

PARTIALLY_SUPPORTED
    Some claims are grounded but others are not, or evidence exists but
    is weak / indirect.  Regeneration is warranted.

UNSUPPORTED
    The answer contains significant hallucinations or makes claims that
    contradict or are entirely absent from the retrieved evidence.
    Regeneration is mandatory.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class VerificationResult(BaseModel):
    """
    Structured output produced by the LLM verifier.

    All list fields default to empty so the model can omit them when
    the answer is clean — avoids forcing the LLM to invent items.
    """

    verdict: Verdict = Field(
        description=(
            "Overall grounding verdict. "
            "SUPPORTED = fully grounded, "
            "PARTIALLY_SUPPORTED = some claims lack evidence, "
            "UNSUPPORTED = significant hallucinations present."
        )
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim or near-verbatim claims from the answer that are NOT "
            "backed by any retrieved chunk.  Empty when verdict is SUPPORTED."
        ),
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Topics or facts that would strengthen the answer but are absent "
            "from both the answer and the retrieved chunks.  These are gaps "
            "in the evidence, not hallucinations."
        ),
    )

    feedback: str = Field(
        description=(
            "Concise, actionable instructions for the answer regeneration node. "
            "Tell it exactly what to remove, what to qualify, and what to "
            "acknowledge as unknown.  Do NOT tell it to retrieve more information."
        )
    )

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional 0–1 confidence that the verdict is correct. "
            "1.0 = fully certain, 0.0 = highly uncertain."
        ),
    )

    def is_acceptable(self) -> bool:
        """Return True when no further regeneration is needed."""
        return self.verdict == Verdict.SUPPORTED

    def to_state_dict(self) -> dict:
        """
        Serialise to a plain dict safe for LangGraph state storage.
        Enums are converted to their string values.
        """
        return {
            "verdict": self.verdict.value,
            "unsupported_claims": self.unsupported_claims,
            "missing_information": self.missing_information,
            "feedback": self.feedback,
            "confidence_score": self.confidence_score,
        }
