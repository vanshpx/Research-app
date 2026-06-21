"""
tools/tavily_search_tool.py
----------------------------
Phase 6 — Tavily web search tool.

Use this tool when:
  - The user needs recent information not present in uploaded documents.
  - The question references events, papers, or benchmarks after the
    knowledge cutoff of the internal corpus.
  - External validation or supplementary context is required.

Do NOT use this tool when:
  - The question can be answered from uploaded PDFs (use retriever_tool).
  - The question requires mathematical computation (use calculator_tool).

Future phases:
  Phase 7 — results can optionally be saved to memory via memory_tool.
  Phase 9 — the planner can decide whether a web search sub-task is needed
             before invoking the agent.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESULTS: int = 3
"""Return at most this many results to keep context concise."""

_SEARCH_DEPTH: str = "basic"
"""
'basic' is fast and sufficient for most queries.
Switch to 'advanced' for deeper research if needed in a future phase.
"""

# ---------------------------------------------------------------------------
# Client — lazy singleton, initialised on first tool call
# ---------------------------------------------------------------------------

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """
    Return the shared TavilyClient, creating it on first use.

    Reads TAVILY_API_KEY from the environment at call time (not at import
    time) so the key can be injected via .env / secrets manager without
    requiring a restart.

    Raises:
        EnvironmentError: If TAVILY_API_KEY is not set.
        RuntimeError:     If the client cannot be initialised.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. "
            "Add it to your environment or .env file."
        )

    try:
        _client = TavilyClient(api_key=api_key)
        logger.info("TavilyClient initialised successfully.")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise TavilyClient: {exc}") from exc

    return _client


# ---------------------------------------------------------------------------
# Result normaliser
# ---------------------------------------------------------------------------


def _normalise_result(raw: dict[str, Any]) -> dict[str, str]:
    """
    Trim a raw Tavily result dict to the three fields the agent needs.

    Tavily returns many fields (score, raw_content, published_date, …).
    We keep only title / content / url to stay within context budgets.

    Args:
        raw: One entry from TavilyClient.search()["results"].

    Returns:
        {"title": str, "content": str, "url": str}
    """
    return {
        "title": (raw.get("title") or "").strip(),
        "content": (raw.get("content") or "").strip(),
        "url": (raw.get("url") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily and return the top 3 results.

    Use this tool when recent or external information is needed that is
    not present in the uploaded research documents.

    Args:
        query: A concise, specific search query string.
               Example: "GraphRAG survey papers 2024 NeurIPS"

    Returns:
        A JSON-encoded list of up to 3 result dicts, each with:
          - title   (str) page / article title
          - content (str) short summary snippet
          - url     (str) source URL

        On failure, returns a JSON object with an "error" key.
    """
    logger.info("tavily_search — query: %r", query)

    try:
        client = _get_client()
    except (EnvironmentError, RuntimeError) as exc:
        logger.error("tavily_search — client error: %s", exc)
        return json.dumps({"error": str(exc), "results": []})

    try:
        response: dict[str, Any] = client.search(
            query=query,
            search_depth=_SEARCH_DEPTH,
            max_results=MAX_RESULTS,
        )
    except Exception as exc:
        logger.error("tavily_search — API call failed: %s", exc)
        return json.dumps({"error": f"Tavily API error: {exc}", "results": []})

    raw_results: list[dict[str, Any]] = response.get("results", [])

    if not raw_results:
        logger.warning("tavily_search — no results returned for query %r", query)
        return json.dumps([])

    results = [_normalise_result(r) for r in raw_results[:MAX_RESULTS]]
    logger.info("tavily_search — returning %d results.", len(results))
    return json.dumps(results, ensure_ascii=False, indent=2)
