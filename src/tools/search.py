"""Web search via Google Serper API."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from src.config import Settings
from src.tools.spec import SidekickTool

SERPER_URL = "https://google.serper.dev/search"


def run_serper_search(api_key: str, query: str, *, timeout: float = 30.0) -> str:
    payload = json.dumps({"q": query}).encode("utf-8")
    request = urllib.request.Request(
        SERPER_URL,
        data=payload,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as exc:
        return f"Search error: {exc}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return f"Search error: invalid JSON response"

    organic = data.get("organic") or []
    if not organic:
        return "No results returned."

    lines: list[str] = []
    for item in organic[:10]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines)


def search_tool(settings: Settings) -> SidekickTool | None:
    if not settings.serper_api_key:
        return None

    key = settings.serper_api_key

    def invoke(query: str) -> str:
        q = query.strip()
        if not q:
            return "Error: empty search query"
        return run_serper_search(key, q, timeout=float(settings.llm_timeout_seconds))

    return SidekickTool(
        name="web_search",
        description="Search the web via Google Serper. Input: natural language query string.",
        invoke=invoke,
    )
