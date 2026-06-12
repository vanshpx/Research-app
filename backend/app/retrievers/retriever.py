"""
Retrieval Agent Module — ReAct-style iterative retrieval agent.

The agent follows a Thought → Action → Observation loop:
1. Embeds and retrieves initial top-k chunks
2. Evaluates if the retrieved context sufficiently answers the question
3. If insufficient, reformulates the query and retrieves again (up to MAX_STEPS)
"""
import logging
from typing import List, Dict, Any, Tuple

from app.embeddings.embedder import embed_query
from app.vectordb.qdrant_store import similarity_search

logger = logging.getLogger(__name__)

MAX_STEPS = 3
MIN_CHUNKS_THRESHOLD = 2
MIN_SCORE_THRESHOLD = 0.3


def _evaluate_sufficiency(chunks: List[Dict[str, Any]], question: str) -> bool:
    """
    Evaluate whether the retrieved chunks are sufficient to answer the question.
    Uses simple heuristics: minimum number of chunks and score threshold.
    """
    if len(chunks) < MIN_CHUNKS_THRESHOLD:
        return False
    # Check if any chunk has a meaningful similarity score
    high_score_chunks = [c for c in chunks if c["score"] >= MIN_SCORE_THRESHOLD]
    return len(high_score_chunks) >= 1


def _reformulate_query(original: str, step: int) -> str:
    """
    Generate a reformulated query for additional retrieval steps.
    Uses simple keyword expansion strategies.
    """
    reformulations = [
        f"What does the document say about: {original}",
        f"Key findings related to: {original}",
    ]
    idx = min(step - 1, len(reformulations) - 1)
    return reformulations[idx]


def retrieve(
    question: str,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    ReAct-style retrieval agent that iteratively retrieves chunks.

    Args:
        question: The user's natural language question.
        top_k: Number of top chunks to retrieve per step.

    Returns:
        Tuple of (deduplicated_chunks, num_retrieval_steps).
    """
    all_chunks: List[Dict[str, Any]] = []
    seen_indices = set()
    steps = 0
    current_query = question

    for step in range(MAX_STEPS):
        steps += 1
        logger.info(f"[ReAct Step {step+1}] Query: '{current_query}'")

        # Action: embed and retrieve
        query_vector = embed_query(current_query)
        retrieved = similarity_search(query_vector, top_k=top_k)

        # Observation: add new unique chunks
        new_chunks = []
        for chunk in retrieved:
            uid = (chunk["source"], chunk["chunk_index"])
            if uid not in seen_indices:
                seen_indices.add(uid)
                new_chunks.append(chunk)
                all_chunks.append(chunk)

        logger.info(f"[ReAct Step {step+1}] Retrieved {len(new_chunks)} new chunks.")

        # Thought: evaluate sufficiency
        if _evaluate_sufficiency(all_chunks, question):
            logger.info(f"[ReAct] Sufficient context found after {steps} step(s).")
            break

        if step < MAX_STEPS - 1:
            # Reformulate for next iteration
            current_query = _reformulate_query(question, step + 1)
            logger.info(f"[ReAct] Reformulating query for step {step+2}: '{current_query}'")

    # Sort final chunks by relevance score descending
    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Return top_k best unique chunks
    return all_chunks[:top_k], steps
