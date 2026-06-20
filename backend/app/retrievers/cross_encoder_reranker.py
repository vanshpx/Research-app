"""
reranker.py
===========
Production-ready Cross-Encoder reranker for an Agentic RAG pipeline.

Model   : BAAI/bge-reranker-base  (via sentence-transformers)
Backend : LangChain + LangGraph

Drop this file into your project as a module:

    from reranker import CrossEncoderReranker

Initialise **once** at application startup:

    reranker = CrossEncoderReranker()

Then call inside any retrieval node / chain:

    top_chunks = reranker.rerank(query, rrf_chunks, top_k=5)
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL    = "BAAI/bge-reranker-base"
_DEFAULT_TOP_K    = 5
_DEFAULT_BATCH    = 32
# bge-reranker-base is a BERT model; 512 is its hard token limit.
_DEFAULT_MAX_LEN  = 512


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder reranker backed by ``BAAI/bge-reranker-base``.

    The model is loaded **once** inside ``__init__``, so it is warm by the
    time the first request arrives.  Instantiate this class at application
    startup and inject the single instance wherever it is needed.

    Each chunk that flows through this class must be a ``dict`` with at
    least the following keys::

        {
            "text":         str,    # passage text that the model scores
            "metadata":     dict,   # arbitrary metadata — preserved as-is
            "fusion_score": float,  # upstream RRF score — preserved as-is
        }

    After ``rerank()`` runs, every returned chunk gains one extra key::

        "rerank_score": float   # higher → more relevant to the query

    The original input list and its dicts are **never modified** (shallow
    copy per chunk).

    Args:
        model_name (str):
            HuggingFace model identifier.
            Default: ``"BAAI/bge-reranker-base"``.
        device (str | None):
            Torch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
            When ``None`` the best available device is chosen automatically:
            CUDA → MPS → CPU.
        batch_size (int):
            Number of (query, passage) pairs forwarded in a single inference
            batch.  Raise for GPU; keep at 32 or lower for CPU.
            Default: ``32``.
        max_length (int):
            Maximum token length passed to the tokeniser.  Pairs longer than
            this are truncated.  Default: ``512``.

    Raises:
        RuntimeError: If the model cannot be loaded from HuggingFace Hub.

    Example::

        reranker = CrossEncoderReranker()       # load once at startup

        top_chunks = reranker.rerank(
            query="What is retrieval-augmented generation?",
            chunks=rrf_output_chunks,
            top_k=5,
        )
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH,
        max_length: int = _DEFAULT_MAX_LEN,
    ) -> None:
        self.model_name  = model_name
        self.device      = device or self._auto_device()
        self.batch_size  = batch_size
        self.max_length  = max_length

        logger.info(
            "Loading CrossEncoder '%s' on device '%s' …",
            self.model_name,
            self.device,
        )
        try:
            self.model: CrossEncoder = CrossEncoder(
                model_name_or_path=self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CrossEncoder '{self.model_name}'. "
                f"Check your internet connection and HuggingFace Hub access.\n"
                f"Original error: {exc}"
            ) from exc

        logger.info("CrossEncoder ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Score every (query, chunk) pair and return the ``top_k`` most
        relevant chunks, sorted by ``rerank_score`` descending.

        This method is **stateless** and **thread-safe** — it does not
        mutate any shared state between calls.

        Args:
            query (str):
                The user's search query.  Must be a non-empty string.
            chunks (list[dict]):
                Candidate chunks produced by the upstream RRF fusion step.
                Each element must be a ``dict`` containing at minimum the
                keys ``"text"`` (str), ``"metadata"`` (dict), and
                ``"fusion_score"`` (float).
            top_k (int):
                How many chunks to return.  If ``top_k`` exceeds the total
                number of chunks, all chunks are returned.
                Default: ``5``.

        Returns:
            list[dict]: Shallow copies of the top ``top_k`` input chunks,
            each with an added ``"rerank_score": float`` key, sorted
            by ``rerank_score`` descending.

        Raises:
            ValueError:
                If ``query`` is empty or ``top_k`` is less than 1.
            TypeError:
                If any element of ``chunks`` is not a ``dict``, or if a
                chunk is missing the required ``"text"`` key.
            RuntimeError:
                If the model's forward pass raises an unexpected error.

        Example::

            results = reranker.rerank(
                query="explain transformer attention mechanism",
                chunks=rrf_chunks,    # list[dict] from RRF step
                top_k=5,
            )
            for r in results:
                print(r["rerank_score"], r["text"][:80])
        """
        # ── input validation ──────────────────────────────────────────────
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "`query` must be a non-empty string; "
                f"got {type(query).__name__!r}: {query!r}."
            )

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(
                f"`top_k` must be a positive integer; got {top_k!r}."
            )

        if not chunks:
            logger.debug("rerank() received an empty chunk list; returning [].")
            return []

        self._validate_chunks(chunks)

        effective_top_k = min(top_k, len(chunks))

        # ── build (query, passage) pairs ──────────────────────────────────
        pairs: list[list[str]] = [
            [query, chunk["text"]] for chunk in chunks
        ]

        # ── model inference ───────────────────────────────────────────────
        try:
            raw_scores = self.model.predict(
                sentences=pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            # predict() returns a numpy ndarray; .tolist() gives plain floats
            scores: list[float] = raw_scores.tolist()
        except Exception as exc:
            raise RuntimeError(
                f"CrossEncoder.predict() failed for model '{self.model_name}'. "
                f"Original error: {exc}"
            ) from exc

        # ── attach rerank_score (shallow-copy each chunk) ─────────────────
        # Shallow copy preserves every existing key (text, metadata,
        # fusion_score, …) without touching the caller's original dicts.
        scored: list[dict] = []
        for chunk, score in zip(chunks, scores):
            new_chunk: dict = copy.copy(chunk)
            new_chunk["rerank_score"] = float(score)
            scored.append(new_chunk)

        # ── sort descending, slice top_k ──────────────────────────────────
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:effective_top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """
        Return the best torch device available on this machine.

        Priority: CUDA → MPS (Apple Silicon) → CPU.
        """
        if torch.cuda.is_available():
            logger.debug("Auto-selected device: cuda")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.debug("Auto-selected device: mps")
            return "mps"
        logger.debug("Auto-selected device: cpu")
        return "cpu"

    @staticmethod
    def _validate_chunks(chunks: list[dict]) -> None:
        """
        Raise ``TypeError`` for any malformed element in *chunks*.

        Checks:
        - Every element is a ``dict``.
        - Every dict contains a ``"text"`` key whose value is a ``str``.
        """
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise TypeError(
                    f"chunks[{idx}] must be a dict, "
                    f"got {type(chunk).__name__!r}."
                )
            if "text" not in chunk:
                raise TypeError(
                    f"chunks[{idx}] is missing the required 'text' key. "
                    f"Present keys: {list(chunk.keys())}."
                )
            if not isinstance(chunk["text"], str):
                raise TypeError(
                    f"chunks[{idx}]['text'] must be a str, "
                    f"got {type(chunk['text']).__name__!r}."
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model_name={self.model_name!r}, "
            f"device={self.device!r}, "
            f"batch_size={self.batch_size}, "
            f"max_length={self.max_length})"
        )


# ===========================================================================
# Application startup  ── initialise once, inject everywhere
# ===========================================================================
#
#   from reranker import reranker      # import the singleton
#
# reranker = CrossEncoderReranker()    # ← this line runs at import time;
#                                      #   the model loads once here.
#
# For larger apps you may prefer lazy initialisation or a DI container,
# but the singleton pattern is the simplest and most common approach.
# ===========================================================================


# ===========================================================================
# LangGraph node integration
# ===========================================================================
#
# Typical LangGraph state shape for your pipeline:
#
# from typing import TypedDict
#
# class RAGState(TypedDict):
#     query:            str
#     rrf_chunks:       list[dict]   # output of your RRF fusion step
#     reranked_chunks:  list[dict]   # filled in by rerank_node below
#     answer:           str
#
#
# def rerank_node(state: RAGState) -> RAGState:
#     """LangGraph node that runs the cross-encoder and trims to top 5."""
#     top_chunks = reranker.rerank(
#         query=state["query"],
#         chunks=state["rrf_chunks"],
#         top_k=5,
#     )
#     return {**state, "reranked_chunks": top_chunks}
#
#
# Wire it into your StateGraph:
#
# from langgraph.graph import StateGraph
#
# graph = StateGraph(RAGState)
# graph.add_node("dense_retrieve",  dense_retrieve_node)
# graph.add_node("bm25_retrieve",   bm25_retrieve_node)
# graph.add_node("rrf_fusion",      rrf_fusion_node)
# graph.add_node("rerank",          rerank_node)          # ← your new node
# graph.add_node("generate",        generate_node)
#
# graph.set_entry_point("dense_retrieve")
# graph.add_edge("dense_retrieve", "rrf_fusion")
# graph.add_edge("bm25_retrieve",  "rrf_fusion")
# graph.add_edge("rrf_fusion",     "rerank")
# graph.add_edge("rerank",         "generate")
# graph.set_finish_point("generate")
#
# app = graph.compile()
# ===========================================================================


# ===========================================================================
# LangChain RunnableLambda integration  (alternative to LangGraph node)
# ===========================================================================
#
# from langchain_core.runnables import RunnableLambda
#
# rerank_runnable = RunnableLambda(
#     lambda inputs: reranker.rerank(
#         query=inputs["query"],
#         chunks=inputs["rrf_chunks"],
#         top_k=5,
#     )
# )
#
# pipeline = rrf_chain | rerank_runnable | llm_chain
# ===========================================================================


# ===========================================================================
# Example usage  (python reranker.py to test without a full RAG stack)
# ===========================================================================

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # ── 1. Instantiate once (simulates application startup) ───────────────
    print("Loading reranker …")
    _reranker = CrossEncoderReranker()   # device auto-detected
    print(_reranker)

    # ── 2. Fake RRF output  (replace with your real retriever output) ──────
    _query = "How does reciprocal rank fusion work?"

    _rrf_chunks: list[dict] = [
        {
            "text": (
                "Reciprocal Rank Fusion (RRF) combines ranked lists from "
                "multiple retrieval systems by summing 1 / (k + rank) for "
                "each document across all lists."
            ),
            "metadata": {"source": "paper_A.pdf", "page": 3},
            "fusion_score": 0.91,
        },
        {
            "text": (
                "Dense retrievers encode queries and documents into vector "
                "embeddings and retrieve by cosine similarity."
            ),
            "metadata": {"source": "paper_B.pdf", "page": 1},
            "fusion_score": 0.78,
        },
        {
            "text": (
                "BM25 is a sparse retrieval algorithm based on term frequency "
                "and inverse document frequency."
            ),
            "metadata": {"source": "paper_C.pdf", "page": 7},
            "fusion_score": 0.65,
        },
        {
            "text": (
                "Cross-encoders jointly process the query and a candidate "
                "passage through a transformer, producing a relevance score."
            ),
            "metadata": {"source": "paper_D.pdf", "page": 2},
            "fusion_score": 0.55,
        },
        {
            "text": (
                "Python is a high-level, interpreted programming language "
                "known for its readability and large standard library."
            ),
            "metadata": {"source": "wiki_python.txt", "page": 0},
            "fusion_score": 0.20,
        },
    ]

    # ── 3. Rerank ──────────────────────────────────────────────────────────
    print(f"\nQuery: {_query!r}")
    print(f"Input chunks: {len(_rrf_chunks)}\n")

    _results = _reranker.rerank(query=_query, chunks=_rrf_chunks, top_k=3)

    # ── 4. Inspect results ─────────────────────────────────────────────────
    print("=== Top-3 reranked chunks ===")
    for _rank, _chunk in enumerate(_results, start=1):
        print(f"\n[{_rank}]  rerank_score={_chunk['rerank_score']:.4f}  "
              f"fusion_score={_chunk['fusion_score']:.2f}")
        print(f"     source : {_chunk['metadata']['source']}")
        print(f"     text   : {_chunk['text'][:100]} …")

    # ── 5. Confirm originals are untouched ─────────────────────────────────
    assert "rerank_score" not in _rrf_chunks[0], (
        "BUG: original chunk was mutated!"
    )
    print("\n✓ Original chunk list was not modified.")