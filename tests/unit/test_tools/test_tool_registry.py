from dataclasses import replace

from src.config import Settings
from src.tools.tool_registry import ToolRegistry, build_tools
from src.tools.browser import BrowserController
from unittest.mock import MagicMock


def test_build_tools_excludes_search_without_key(minimal_settings: Settings):
    tools = build_tools(minimal_settings)
    names = {t.name for t in tools}
    assert "web_search" not in names
    assert "read_sandbox_file" in names
    assert "wikipedia_summary" in names
    assert "run_python" in names


def test_build_tools_includes_search_when_key_set(minimal_settings: Settings):
    s = replace(minimal_settings, serper_api_key="fake-key")
    tools = build_tools(s)
    names = {t.name for t in tools}
    assert "web_search" in names


def test_build_tools_notification_when_enabled(minimal_settings: Settings):
    s = replace(
        minimal_settings,
        enable_notifications=True,
        pushover_token="t",
        pushover_user="u",
    )
    names = {t.name for t in build_tools(s)}
    assert "send_push_notification" in names


def test_build_tools_skips_browser_by_default(minimal_settings: Settings):
    ctrl = MagicMock(spec=BrowserController)
    names = {t.name for t in build_tools(minimal_settings, browser=ctrl)}
    assert "browser_navigate" not in names


def test_build_tools_includes_browser_when_requested(minimal_settings: Settings):
    ctrl = MagicMock(spec=BrowserController)
    names = {
        t.name
        for t in build_tools(
            minimal_settings,
            browser=ctrl,
            include_browser=True,
        )
    }
    assert "browser_navigate" in names
    assert "browser_page_text" in names


def test_tool_registry_get_tools(minimal_settings: Settings):
    reg = ToolRegistry(minimal_settings)
    assert len(reg.get_tools()) >= 4
