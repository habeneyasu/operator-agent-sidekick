"""Browser automation via Playwright (sync API)."""

from __future__ import annotations

from typing import Any

from src.tools.spec import SidekickTool


class BrowserController:
    """Lifecycle wrapper around Playwright Chromium (sync)."""

    def __init__(self, headless: bool) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "playwright is required for browser tools. Install with "
                "`uv pip install playwright` and `playwright install chromium`."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._page is not None:
            self._page = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("Browser not started; call start() first")
        return self._page

    def navigate(self, url: str) -> str:
        p = self.page
        p.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url} (title: {p.title()!r})"

    def page_text_snapshot(self, max_chars: int = 12_000) -> str:
        p = self.page
        try:
            text = p.inner_text("body", timeout=5_000)
        except Exception as exc:  # noqa: BLE001 — surface page errors to the model
            return f"Could not read page text: {exc}"
        text = text.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n... (truncated)"
        return text or "(empty body)"


def browser_tools(controller: BrowserController) -> list[SidekickTool]:
    def navigate_tool(url: str) -> str:
        u = url.strip()
        if not u:
            return "Error: empty URL"
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        try:
            return controller.navigate(u)
        except Exception as exc:  # noqa: BLE001
            return f"Navigation error: {exc}"

    def snapshot_tool(_: str) -> str:
        try:
            return controller.page_text_snapshot()
        except Exception as exc:  # noqa: BLE001
            return f"Snapshot error: {exc}"

    return [
        SidekickTool(
            name="browser_navigate",
            description="Open a URL in the controlled browser. Input: full URL or hostname (https added if missing).",
            invoke=navigate_tool,
        ),
        SidekickTool(
            name="browser_page_text",
            description="Return visible text from the current page body (truncated). Input: ignored.",
            invoke=snapshot_tool,
        ),
    ]
