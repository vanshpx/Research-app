"""
Vector Store Module — Manages Qdrant in-memory vector database.
Stores chunk embeddings with metadata and performs cosine similarity search.
"""
import logging
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_docs"
EMBEDDING_DIM = 384

# Singleton Qdrant client (in-memory)
_client: QdrantClient = None


def get_client() -> QdrantClient:
    """Return or create an in-memory Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(":memory:")
        logger.info("Qdrant in-memory client initialized.")
    return _client


def ensure_collection():
    """Create the collection if it doesn't exist yet."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")


def upsert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Insert chunk documents with their embeddings into Qdrant.

    Args:
        chunks: List of chunk dicts (text, page, source, chunk_index).
        embeddings: Corresponding list of embedding vectors.
    """
    ensure_collection()
    client = get_client()

    points = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Upserted {len(points)} chunks into Qdrant.")


def similarity_search(
    query_vector: List[float],
    top_k: int = 5,
    source_filter: str = None,
) -> List[Dict[str, Any]]:
    """
    Perform cosine similarity search against stored chunks.

    Args:
        query_vector: Embedded query vector.
        top_k: Number of top results to return.
        source_filter: Optional filename to restrict search to one document.

    Returns:
        List of result dicts with text, page, source, chunk_index, score.
    """
    ensure_collection()
    client = get_client()

    query_filter = None
    

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

   

    

    hits = []
    for r in response.points:
        hits.append({
            "text": r.payload["text"],
            "page": r.payload["page"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score": r.score,
        })

    return hits


def get_document_count() -> int:
    """Return total number of indexed chunks."""
    try:
        ensure_collection()
        client = get_client()
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except Exception:
        return 0

def get_all_chunks() -> List[Dict[str, Any]]:
    """
    Retrieve all chunk payloads from Qdrant.
    """

    ensure_collection()
    client = get_client()

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100000,
        with_payload=True,
        with_vectors=False,
    )

    chunks = []

    for p in points:
        chunks.append({
            "text": p.payload["text"],
            "page": p.payload["page"],
            "source": p.payload["source"],
            "chunk_index": p.payload["chunk_index"],
        })

    return chunks