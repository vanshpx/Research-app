"""
Citation Builder Module — Formats retrieved chunks into structured citation objects.
"""
from typing import List, Dict, Any
from app.models.schemas import Citation


def build_citations(chunks: List[Dict[str, Any]]) -> List[Citation]:
    """
    Convert raw retrieved chunks into Citation objects.

    Args:
        chunks: List of chunk dicts (text, page, source, chunk_index, score).

    Returns:
        List of Citation objects with snippet previews.
    """
    citations = []
    seen = set()

    for chunk in chunks:
        # Create a unique key to avoid duplicate citations
        key = (chunk["source"], chunk["page"], chunk["chunk_index"])
        if key in seen:
            continue
        seen.add(key)

        # Trim snippet to ~200 characters for readability
        snippet = chunk["text"].strip()
        if len(snippet) > 250:
            snippet = snippet[:247] + "..."

        citations.append(
            Citation(
                page=chunk["page"],
                source=chunk["source"],
                snippet=snippet,
                chunk_index=chunk["chunk_index"],
            )
        )

    return citations
