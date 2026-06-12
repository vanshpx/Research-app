"""
Embedding Module — Generates dense vector embeddings using BAAI/bge-small-en-v1.5.
Embedding dimension: 384. Model is loaded once as a singleton.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

_model = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_model():
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float vectors (each of length 384).
    """
    model = get_model()
    # BGE models work best with a query prefix for retrieval tasks
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    BGE models recommend a prefix for queries during retrieval.

    Args:
        query: The user question string.

    Returns:
        Float vector of length 384.
    """
    model = get_model()
    # BGE retrieval query prefix
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
    return embedding[0].tolist()
