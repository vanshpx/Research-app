"""
generation/generation_prompt.py
---------------------------------
All prompt text for answer generation and regeneration.

Two separate prompt builders:
  build_initial_generation_prompt()   — first-time generation from chunks
  build_regeneration_prompt()         — revision using verifier feedback

Keeping both here means changing generation behaviour requires touching
exactly one file — not scattered across graph nodes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared system prompt (used for both initial and regeneration calls)
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are an expert research assistant.
Your task is to answer a user question using ONLY the provided evidence chunks.

## Rules

1. Base every claim on the provided chunks.
2. If a topic is not covered by the chunks, explicitly say so.
3. Cite sources inline as [Source: <filename>, Page: <n>].
4. Be thorough but concise. Avoid padding.
5. Never invent information not present in the chunks.
6. Structure your answer with paragraphs; use markdown only if it aids clarity.
"""

# ---------------------------------------------------------------------------
# Initial generation (no prior answer exists)
# ---------------------------------------------------------------------------

INITIAL_GENERATION_TEMPLATE = """\
## Question
{question}

## Retrieved Evidence
{chunks_block}

Write a complete, well-cited answer to the question using only the evidence above.
If the evidence is insufficient to fully answer the question, say so clearly.
"""

# ---------------------------------------------------------------------------
# Regeneration (verifier feedback is available)
# ---------------------------------------------------------------------------

REGENERATION_SYSTEM_PROMPT = """\
You are an expert research assistant performing answer revision.
A previous answer was evaluated by a fact-checker and found to have issues.
Your task is to produce an improved answer that addresses all feedback.

## Rules

1. READ the feedback carefully before writing anything.
2. REMOVE every unsupported claim listed in the feedback.
3. QUALIFY uncertain statements with phrases like "according to the evidence" \
or "the retrieved chunks suggest".
4. ACKNOWLEDGE any gaps explicitly — do not fill them with invented content.
5. Use ONLY the provided evidence chunks. Do not add information from memory.
6. Cite sources inline as [Source: <filename>, Page: <n>].
7. Do not reference the fact-checking process in your answer.
"""

REGENERATION_TEMPLATE = """\
## Original Question
{question}

## Retrieved Evidence
{chunks_block}

## Previous Answer (contains issues)
{previous_answer}

## Verifier Feedback
{feedback}

## Unsupported Claims to Remove
{unsupported_claims_block}

## Missing Information (acknowledge these gaps)
{missing_information_block}

Write an improved answer that fully addresses the feedback above.
"""


# ---------------------------------------------------------------------------
# Shared helper: format chunks
# ---------------------------------------------------------------------------


def _format_chunks_block(chunks: list[dict]) -> str:
    """Render chunks into a numbered evidence block for prompt insertion."""
    if not chunks:
        return "No evidence chunks were retrieved."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Chunk {i} | Source: {source}, Page: {page}]\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_initial_generation_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the user-turn message for the first-time answer generation call.

    Args:
        question: Original user question.
        chunks:   Retrieved evidence chunks.

    Returns:
        Formatted prompt string.
    """
    return INITIAL_GENERATION_TEMPLATE.format(
        question=question,
        chunks_block=_format_chunks_block(chunks),
    )


def build_regeneration_prompt(
    question: str,
    chunks: list[dict],
    previous_answer: str,
    feedback: str,
    unsupported_claims: list[str],
    missing_information: list[str],
) -> str:
    """
    Build the user-turn message for answer regeneration.

    Args:
        question:             Original user question.
        chunks:               Retrieved evidence chunks (unchanged from Phase 4).
        previous_answer:      The answer that failed verification.
        feedback:             Actionable feedback string from VerificationResult.
        unsupported_claims:   List of specific claims to remove.
        missing_information:  List of gaps to acknowledge.

    Returns:
        Formatted prompt string.
    """
    unsupported_block = (
        "\n".join(f"- {c}" for c in unsupported_claims)
        if unsupported_claims
        else "None identified."
    )
    missing_block = (
        "\n".join(f"- {m}" for m in missing_information)
        if missing_information
        else "None identified."
    )
    return REGENERATION_TEMPLATE.format(
        question=question,
        chunks_block=_format_chunks_block(chunks),
        previous_answer=previous_answer,
        feedback=feedback,
        unsupported_claims_block=unsupported_block,
        missing_information_block=missing_block,
    )
