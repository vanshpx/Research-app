"""
llm/groq_client.py (formerly gemini.py)
-----------------------------------------
Groq LLM module — wraps ChatGroq for use outside the LangGraph pipeline.

This module provides a simple get_answer() function for one-shot queries
that don't need the full agent graph. The LangGraph pipeline in react_agent.py
instantiates ChatGroq directly.

Model: qwen/qwen3-32b via Groq API
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)

_client: ChatGroq | None = None


def get_groq_client() -> ChatGroq:
    """Lazy-initialize the Groq LangChain client (singleton)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        _client = ChatGroq(
            model="qwen/qwen3-32b",
            groq_api_key=api_key,
            temperature=0,
        )
        logger.info("Groq ChatGroq client initialized (model: qwen/qwen3-32b).")
    return _client


def get_answer(question: str, context: str, system_prompt: str = "") -> str:
    """
    Generate an answer for a question given a context string.

    Args:
        question:      The user's natural-language question.
        context:       Retrieved evidence text to ground the answer.
        system_prompt: Optional custom system instruction.

    Returns:
        The model's answer as a plain string.
    """
    client = get_groq_client()

    system = system_prompt or (
        "You are an expert research assistant. "
        "Answer the question using only the provided context. "
        "Cite sources where possible. "
        "If the context does not contain sufficient information, say so clearly."
    )

    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        response = client.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])
        answer = str(response.content).strip()
        logger.info("Groq answer generated (%d chars).", len(answer))
        return answer
    except Exception as e:
        logger.error("Groq generation error: %s", e)
        raise
