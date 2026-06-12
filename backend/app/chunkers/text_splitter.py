"""
Chunking Module — Splits document text into overlapping chunks.
Uses RecursiveCharacterTextSplitter logic (1000 chars, 200 overlap).
Each chunk preserves its source page number and filename metadata.
"""
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively split text using separators, similar to LangChain's
    RecursiveCharacterTextSplitter.
    """
    def _merge(splits: List[str], separator: str) -> List[str]:
        chunks = []
        current = ""
        for s in splits:
            candidate = current + (separator if current else "") + s
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Handle single split that's too large
                if len(s) > chunk_size:
                    # Recursively split on next separator — handled below
                    chunks.extend(_split_with_fallback(s))
                else:
                    current = s
        if current:
            chunks.append(current)
        return chunks

    def _split_with_fallback(text: str) -> List[str]:
        for sep in SEPARATORS:
            if sep and sep in text:
                parts = text.split(sep)
                merged = _merge(parts, sep)
                return merged
        # Last resort: character-level split
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    if len(text) <= chunk_size:
        return [text]

    return _split_with_fallback(text)


def _add_overlap(chunks: List[str], chunk_overlap: int) -> List[str]:
    """
    Add overlap between chunks by prepending the tail of the previous chunk.
    """
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        overlap = chunks[i-1][-chunk_overlap:] if len(chunks[i-1]) > chunk_overlap else chunks[i-1]
        result.append(overlap + " " + chunks[i])
    return result


def chunk_pages(
    pages: List[Tuple[int, str]],
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split pages into overlapping chunks with metadata.

    Args:
        pages: List of (page_number, text) tuples.
        filename: Source filename for metadata.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with keys: text, page, source, chunk_index.
    """
    all_chunks = []
    global_index = 0

    for page_num, page_text in pages:
        raw_chunks = _split_text(page_text, chunk_size, chunk_overlap)
        overlapped = _add_overlap(raw_chunks, chunk_overlap)

        for chunk_text in overlapped:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            all_chunks.append({
                "text": chunk_text,
                "page": page_num,
                "source": filename,
                "chunk_index": global_index,
            })
            global_index += 1

    logger.info(f"Generated {len(all_chunks)} chunks from {len(pages)} pages of '{filename}'.")
    return all_chunks
