"""
Gemini LLM Module — Generates answers using Gemini 2.5 Flash.
Constrains output to only use information from the provided context chunks.
"""
import logging
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import google.genai as genai

logger = logging.getLogger(__name__)

_client = None


def get_gemini_client() -> genai.Client:
    """Lazy-initialize the Gemini client (singleton)."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini Client initialized.")
    return _client


def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer to the question using the retrieved context chunks.

    Args:
        question: The user's natural language question.
        context_chunks: List of chunk dicts with text, page, and source.

    Returns:
        The generated answer string.
    """
    client = get_gemini_client()

    # Format context with source info
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Context Passages:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer (based only on the context above):"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=(
                    "You are a precise research assistant. Answer questions ONLY based on the "
                    "provided context passages. Do not use any external knowledge. "
                    "If the answer cannot be found in the context, say: "
                    "'I could not find relevant information in the uploaded documents.' "
                    "Always be concise and factual."
                )
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        raise RuntimeError(f"Failed to generate answer: {str(e)}")
