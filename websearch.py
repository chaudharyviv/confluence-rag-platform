"""
Live web search for the external-fallback route, via Tavily's search API.

Only graph.py's external_search_node calls this - and only after retrieval came
up weak and the router decided the question needs current information. Results
are untrusted third-party text: llm.generate_external_answer treats them as
data, never as instructions.

Fails soft: no key, a network error or a bad response all return [] so the
caller falls back to Claude's general knowledge (still labelled as external).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from config import settings

log = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class WebResult:
    title: str
    url: str
    content: str


def search_web(query: str) -> list[WebResult]:
    if not settings.tavily_api_key:
        return []
    try:
        resp = requests.post(
            _TAVILY_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={
                "query": query,
                "max_results": settings.web_search_max_results,
                "search_depth": "basic",
                "topic": "general",
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:
        # type name only: keep the request (and key) out of logs
        log.warning("Tavily search failed (%s)", type(exc).__name__)
        return []
    return [
        WebResult(title=r.get("title") or r["url"], url=r["url"], content=r.get("content", ""))
        for r in raw
        if r.get("url")
    ]
