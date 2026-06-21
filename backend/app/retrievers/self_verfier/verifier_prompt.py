"""
verification/verifier_prompt.py
---------------------------------
All prompt text for the Self-RAG verifier lives here.
The verifier.py module imports these strings — nothing is hardcoded there.

Keeping prompts in a dedicated file means:
  - Prompt engineers can iterate without touching business logic.
  - Each prompt can be versioned / A-B tested independently.
  - Phase 9 planner can swap prompts per task type without changing nodes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = """\
You are a rigorous fact-checking assistant for a research RAG system.

Your job is to verify whether a generated answer is faithfully grounded \
in the provided retrieved evidence chunks.

## Your responsibilities

1. IDENTIFY UNSUPPORTED CLAIMS
   Flag any factual claim in the answer that cannot be directly traced \
to at least one retrieved chunk.
   Include statistics, named entities, relationships, and causal statements.

2. IDENTIFY HALLUCINATIONS
   A hallucination is a claim that contradicts or is entirely absent from \
the evidence.
   Be precise — quote or closely paraphrase the offending claim.

3. IDENTIFY MISSING EVIDENCE
   Note topics that are relevant to the question but absent from both \
the answer and the retrieved chunks.
   These are evidence gaps, not answer errors.

4. PRODUCE ACTIONABLE FEEDBACK
   Write clear, specific instructions for the answer regeneration step.
   Tell it what to REMOVE, what to QUALIFY with hedging language, and \
what to ACKNOWLEDGE as unknown.
   Do NOT tell it to search for more information — retrieval is already done.

## Critical rules

- Judge the answer ONLY against the provided chunks.
- If a claim is reasonable but not in the chunks, it is UNSUPPORTED.
- Prefer PARTIALLY_SUPPORTED over UNSUPPORTED when the answer is mostly correct.
- An empty answer or refusal to answer should be UNSUPPORTED.
- Be concise. Bullet points in feedback are encouraged.
"""

# ---------------------------------------------------------------------------
# User turn template
# ---------------------------------------------------------------------------

VERIFIER_USER_TEMPLATE = """\
## Original Question
{question}

## Retrieved Evidence Chunks
{chunks_block}

## Answer to Verify
{answer}

Evaluate the answer strictly against the evidence above.
"""


def format_chunks_block(chunks: list[dict]) -> str:
    """
    Render the retrieved chunks into a numbered block for the prompt.

    Args:
        chunks: List of chunk dicts with keys text, source, page, rerank_score.

    Returns:
        Multi-line string ready for insertion into VERIFIER_USER_TEMPLATE.
    """
    if not chunks:
        return "No chunks were retrieved."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Chunk {i} | Source: {source}, Page: {page}]\n{text}")

    return "\n\n".join(parts)


def build_verifier_user_message(
    question: str,
    chunks: list[dict],
    answer: str,
) -> str:
    """
    Assemble the user-turn message for the verifier LLM call.

    Args:
        question: The original user question.
        chunks:   Retrieved evidence chunks.
        answer:   The answer to be verified.

    Returns:
        Formatted string ready to pass as the user message.
    """
    chunks_block = format_chunks_block(chunks)
    return VERIFIER_USER_TEMPLATE.format(
        question=question,
        chunks_block=chunks_block,
        answer=answer,
    )
