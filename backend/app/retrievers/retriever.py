"""
Retrieval Agent Module — ReAct-style iterative retrieval agent.

The agent follows a Thought → Action → Observation loop:
1. Embeds and retrieves initial top-k chunks
2. Evaluates if the retrieved context sufficiently answers the question
3. If insufficient, reformulates the query and retrieves again (up to MAX_STEPS)
"""
from app.retrievers.bm25_retriever import BM25Retriever
from app.retrievers import rrf
from app.vectordb.qdrant_store import get_all_chunks, get_document_count
import logging
from typing import List, Dict, Any, Tuple, Optional

from app.embeddings.embedder import embed_query
from app.vectordb.qdrant_store import similarity_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 cache — built on first upload, rebuilt on every subsequent upload.
# Queries always read from the cache; never trigger a rebuild themselves.
# ---------------------------------------------------------------------------
_bm25_cache: Optional[BM25Retriever] = None
_bm25_chunk_count: int = 0


def rebuild_bm25() -> None:
    """
    Build (or rebuild) the BM25 index from the current Qdrant corpus.

    Call this immediately after every successful PDF upload so the index
    is always fresh before the next query arrives. Routes.py owns this call.
    """
    global _bm25_cache, _bm25_chunk_count

    all_chunks = get_all_chunks()
    _bm25_cache = BM25Retriever(all_chunks)
    _bm25_chunk_count = len(all_chunks)
    logger.info(
        "[BM25] Index built — %d chunks indexed.", _bm25_chunk_count
    )


def _get_bm25() -> BM25Retriever:
    """
    Return the cached BM25Retriever.

    If no upload has happened yet (cache is empty), builds a one-time
    fallback so queries don't crash on a fresh server start with pre-loaded data.
    """
    global _bm25_cache

    if _bm25_cache is None:
        logger.warning(
            "[BM25] Cache is empty — building fallback index. "
            "Upload a PDF to populate the index properly."
        )
        rebuild_bm25()

    return _bm25_cache

MAX_STEPS = 3
MIN_CHUNKS_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Retrieval depth constants — single source of truth for all depth limits.
# ---------------------------------------------------------------------------
DENSE_TOP_K: int = 15   # chunks fetched from Qdrant (dense) per step
BM25_TOP_K: int = 15    # chunks fetched from BM25 (sparse) per step
RRF_TOP_K: int = 10     # chunks kept after Reciprocal Rank Fusion


def _evaluate_sufficiency(chunks: List[Dict[str, Any]], question: str) -> bool:
    """
    Evaluate whether the retrieved chunks are sufficient to answer the question.
    Uses simple heuristics: minimum number of chunks and score threshold.
    """
    if len(chunks) < MIN_CHUNKS_THRESHOLD:
        return False
    # Check if any chunk has a meaningful fusion score
    high_score_chunks = [c for c in chunks if c.get("fusion_score", 0) > 0]
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
) -> Tuple[List[Dict[str, Any]], int]:
    """
    ReAct-style retrieval agent that iteratively retrieves chunks.

    Depth settings (enforced via module constants):
      Dense (Qdrant) : DENSE_TOP_K = {dense}
      BM25 (sparse)  : BM25_TOP_K  = {bm25}
      RRF output cap : RRF_TOP_K   = {rrf}

    Args:
        question: The user's natural language question.

    Returns:
        Tuple of (deduplicated_chunks, num_retrieval_steps).
        len(chunks) <= RRF_TOP_K ({rrf}).
    """.format(dense=DENSE_TOP_K, bm25=BM25_TOP_K, rrf=RRF_TOP_K)
    # Use cached BM25 index — only rebuilt when new docs are uploaded
    bm25 = _get_bm25()

    all_chunks: List[Dict[str, Any]] = []
    seen_indices = set()
    steps = 0
    current_query = question

    for step in range(MAX_STEPS):
        steps += 1
        logger.info("[ReAct Step %d] Query: %r", step + 1, current_query)

        # Action: embed and retrieve
        dense_results = similarity_search(
            embed_query(current_query),
            top_k=DENSE_TOP_K,
        )

        sparse_results = bm25.retrieve(
            current_query,
            top_k=BM25_TOP_K,
        )

        retrieved = rrf.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
        )
        # Cap RRF output immediately — only keep the top-ranked RRF_TOP_K chunks.
        retrieved = retrieved[:RRF_TOP_K]

        # Observation: add new unique chunks
        new_chunks = []
        for chunk in retrieved:
            uid = (chunk["source"], chunk["chunk_index"])
            if uid not in seen_indices:
                seen_indices.add(uid)
                new_chunks.append(chunk)
                all_chunks.append(chunk)

        logger.info("[ReAct Step %d] %d new chunks added.", step + 1, len(new_chunks))

        # Thought: evaluate sufficiency
        if _evaluate_sufficiency(all_chunks, question):
            logger.info("[ReAct] Sufficient context after %d step(s).", steps)
            break

        if step < MAX_STEPS - 1:
            current_query = _reformulate_query(question, step + 1)
            logger.info(
                "[ReAct] Reformulating query for step %d: %r", step + 2, current_query
            )

    # Return at most RRF_TOP_K unique chunks
    return all_chunks[:RRF_TOP_K], steps
