"""
Minimal Confluence Cloud REST client: list pages in a space, fetch storage-
format body + version. Incremental sync means "only re-embed pages whose
version changed since last sync" - checked against db.get_synced_version(),
not a local JSON file (the original repos' approach, which isn't
concurrency-safe and leaves no audit trail).
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from config import settings


@dataclass
class ConfluencePage:
    page_id: str
    title: str
    version: int
    body_html: str
    url: str


class ConfluenceClient:
    def __init__(self) -> None:
        self._base = settings.confluence_base_url.rstrip("/")
        self._auth = (settings.confluence_email, settings.confluence_api_token)

    def list_pages(self) -> list[ConfluencePage]:
        pages: list[ConfluencePage] = []
        start = 0
        limit = 50
        while True:
            resp = requests.get(
                f"{self._base}/rest/api/content",
                auth=self._auth,
                params={
                    "spaceKey": settings.confluence_space_key,
                    "expand": "body.storage,version",
                    "start": start,
                    "limit": limit,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for result in data.get("results", []):
                pages.append(
                    ConfluencePage(
                        page_id=result["id"],
                        title=result["title"],
                        version=result["version"]["number"],
                        body_html=result["body"]["storage"]["value"],
                        url=f"{self._base}{result['_links']['webui']}",
                    )
                )
            if data.get("size", 0) < limit:
                break
            start += limit
        return pages
