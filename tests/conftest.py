from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


@pytest.fixture
def minimal_settings(tmp_path: Path) -> Settings:
    sandbox = tmp_path / "sandbox"
    sess = tmp_path / ".sessions"
    return Settings(
        app_name="test-app",
        environment="test",
        log_level="INFO",
        openai_api_key=None,
        openai_model_worker="gpt-4o-mini",
        openai_model_evaluator="gpt-4o-mini",
        llm_timeout_seconds=30,
        max_agent_iterations=8,
        browser_headless=True,
        sandbox_dir=sandbox,
        session_store_dir=sess,
        enable_notifications=False,
        serper_api_key=None,
        pushover_token=None,
        pushover_user=None,
    )
