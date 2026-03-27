"""Wikipedia summary lookup (REST API, no API key)."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

from src.tools.spec import SidekickTool

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def fetch_wikipedia_summary(title: str, *, timeout: float = 20.0) -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = WIKI_SUMMARY_URL.format(title=encoded)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BrowserSidekickAgent/1.0 (educational)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return f"No Wikipedia page found for: {title}"
        return f"Wikipedia error: HTTP {exc.code}"
    except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as exc:
        return f"Wikipedia error: {exc}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return "Wikipedia error: invalid response"

    extract = data.get("extract") or data.get("description") or ""
    page_title = data.get("title", title)
    content_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    if not extract:
        return f"No extract available for: {page_title}"
    return f"{page_title}\n{content_url}\n\n{extract}"


def wikipedia_tool() -> SidekickTool:
    def invoke(title: str) -> str:
        t = title.strip()
        if not t:
            return "Error: empty page title"
        return fetch_wikipedia_summary(t)

    return SidekickTool(
        name="wikipedia_summary",
        description="Fetch a short Wikipedia summary. Input: article title (e.g. 'LangGraph').",
        invoke=invoke,
    )
