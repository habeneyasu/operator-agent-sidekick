from unittest.mock import MagicMock

from src.tools.browser import browser_tools


def test_browser_tools_invoke_uses_controller():
    ctrl = MagicMock()
    ctrl.navigate.return_value = "ok-nav"
    ctrl.page_text_snapshot.return_value = "page-body"

    tools = browser_tools(ctrl)
    by_name = {t.name: t for t in tools}

    assert by_name["browser_navigate"].run("example.com") == "ok-nav"
    ctrl.navigate.assert_called_once()

    assert by_name["browser_page_text"].run("") == "page-body"
    ctrl.page_text_snapshot.assert_called_once()
