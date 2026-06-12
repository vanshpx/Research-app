"""
PDF Loader Module — Extracts text from PDF files using PyPDF.
Returns a list of (page_number, text) tuples.
"""
import logging
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Load a PDF file and extract text from each page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples (1-indexed page numbers).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages: List[Tuple[int, str]] = []
    reader = PdfReader(str(path))

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
            logger.debug(f"Loaded page {i} with {len(text)} characters.")

    logger.info(f"Loaded {len(pages)} pages from '{path.name}'.")
    return pages
