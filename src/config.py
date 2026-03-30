"""Application configuration for the browser sidekick agent.

This module provides a typed, environment-driven settings object that other
modules can import without duplicating `os.getenv` calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings loaded from environment variables."""

    app_name: str
    environment: str
    log_level: str
    openai_api_key: str | None
    openai_model_worker: str
    openai_model_evaluator: str
    llm_timeout_seconds: int
    max_agent_iterations: int
    browser_headless: bool
    sandbox_dir: Path
    session_store_dir: Path
    enable_notifications: bool
    serper_api_key: str | None
    pushover_token: str | None
    pushover_user: str | None
    tokens_per_run_limit: int
    history_char_limit: int
    openrouter_api_key: str | None
    openrouter_base_url: str | None
    openrouter_model_worker: str | None
    openrouter_model_evaluator: str | None
    openrouter_max_tokens: int | None
    ollama_base_url: str | None
    ollama_model_worker: str | None
    ollama_model_evaluator: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        root_dir = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
        sandbox_default = root_dir / "sandbox"
        sessions_default = root_dir / ".sessions"

        return cls(
            app_name=os.getenv("APP_NAME", "browser-sidekick-agent"),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model_worker=os.getenv("OPENAI_MODEL_WORKER", "llama3.1:8b"),
            openai_model_evaluator=os.getenv("OPENAI_MODEL_EVALUATOR", "llama3.1:8b"),
            llm_timeout_seconds=_as_int(os.getenv("LLM_TIMEOUT_SECONDS"), 30),
            max_agent_iterations=_as_int(os.getenv("MAX_AGENT_ITERATIONS"), 8),
            browser_headless=_as_bool(os.getenv("BROWSER_HEADLESS"), default=False),
            sandbox_dir=Path(os.getenv("SANDBOX_DIR", str(sandbox_default))).resolve(),
            session_store_dir=Path(
                os.getenv("SESSION_STORE_DIR", str(sessions_default))
            ).resolve(),
            enable_notifications=_as_bool(
                os.getenv("ENABLE_NOTIFICATIONS"), default=False
            ),
            serper_api_key=os.getenv("SERPER_API_KEY"),
            pushover_token=os.getenv("PUSHOVER_TOKEN"),
            pushover_user=os.getenv("PUSHOVER_USER"),
            tokens_per_run_limit=_as_int(os.getenv("TOKENS_PER_RUN_LIMIT"), 50000),
            history_char_limit=_as_int(os.getenv("HISTORY_CHAR_LIMIT"), 8000),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL"),
            openrouter_model_worker=os.getenv("OPENROUTER_MODEL_WORKER"),
            openrouter_model_evaluator=os.getenv("OPENROUTER_MODEL_EVALUATOR"),
            openrouter_max_tokens=_as_int(os.getenv("OPENROUTER_MAX_TOKENS"), 0) or None,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
            ollama_model_worker=os.getenv("OLLAMA_MODEL_WORKER"),
            ollama_model_evaluator=os.getenv("OLLAMA_MODEL_EVALUATOR"),
        )

    def ensure_runtime_dirs(self) -> None:
        """Create local directories required by tools/session storage."""
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.session_store_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached app settings for process-wide reuse."""
    settings = Settings.from_env()
    settings.ensure_runtime_dirs()
    return settings
