"""Register and compose Sidekick tools for the worker LLM."""

from __future__ import annotations

from src.config import Settings
from src.tools.browser import BrowserController, browser_tools
from src.tools.file_manager import file_manager_tools
from src.tools.notification import notification_tool
from src.tools.repl import repl_tool
from src.tools.search import search_tool
from src.tools.spec import SidekickTool
from src.tools.wikipedia import wikipedia_tool


def build_tools(
    settings: Settings,
    *,
    browser: BrowserController | None = None,
    include_browser: bool = False,
) -> list[SidekickTool]:
    """Assemble default tools. Browser tools are omitted unless ``include_browser`` is true and a started controller is passed."""
    tools: list[SidekickTool] = []
    tools.extend(file_manager_tools(settings.sandbox_dir))
    t_search = search_tool(settings)
    if t_search is not None:
        tools.append(t_search)
    tools.append(wikipedia_tool())
    tools.append(repl_tool())
    t_notify = notification_tool(settings)
    if t_notify is not None:
        tools.append(t_notify)
    if include_browser and browser is not None:
        tools.extend(browser_tools(browser))
    return tools


class ToolRegistry:
    """Holds settings and optional browser session, exposes ``get_tools()``."""

    def __init__(
        self,
        settings: Settings,
        *,
        browser: BrowserController | None = None,
        include_browser: bool = False,
    ) -> None:
        self._settings = settings
        self._browser = browser
        self._include_browser = include_browser

    def get_tools(self) -> list[SidekickTool]:
        return build_tools(
            self._settings,
            browser=self._browser,
            include_browser=self._include_browser,
        )

    @property
    def settings(self) -> Settings:
        return self._settings
