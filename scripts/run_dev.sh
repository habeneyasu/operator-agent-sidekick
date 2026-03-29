#!/usr/bin/env bash
set -euo pipefail

# Simple dev runner: executes the sidekick graph once using OpenAI-backed assembly.
# Requires: uv, OPENAI_API_KEY in environment, and langchain-openai/langgraph installed.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://github.com/astral-sh/uv" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

PROMPT=${1:-"Say hello and explain what this agent does"}
CRITERIA=${2:-"Provide a concise helpful response"}

uv run --with langchain-openai --with langgraph --with langchain-core python - <<'PYCODE'
import os
from src.ui.api import run_once_openai, run_once_openrouter

prompt = os.environ.get("PROMPT") or """Say hello and explain what this agent does"""
criteria = os.environ.get("CRITERIA") or "Provide a concise helpful response"

if os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_BASE_URL"):
    out = run_once_openrouter(user_messages=[prompt], success_criteria=criteria)
else:
    out = run_once_openai(user_messages=[prompt], success_criteria=criteria)
print("success_criteria_met:", out.get("success_criteria_met"))
print("messages:")
for m in out.get("messages", []):
    print("-", type(m).__name__, ":", getattr(m, "content", None))
PYCODE
