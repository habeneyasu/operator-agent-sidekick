#!/usr/bin/env bash
set -euo pipefail

# Launch the Gradio UI for manual testing.
# Requires: gradio/langgraph/langchain-openai/langchain-core
# Auth options:
#   - OPENAI_API_KEY
#   - or OPENROUTER_API_KEY and OPENROUTER_BASE_URL

if [ -z "${OPENAI_API_KEY:-}" ] && { [ -z "${OPENROUTER_API_KEY:-}" ] || [ -z "${OPENROUTER_BASE_URL:-}" ]; }; then
  echo "Please set either OPENAI_API_KEY, or OPENROUTER_API_KEY and OPENROUTER_BASE_URL." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://github.com/astral-sh/uv" >&2
  exit 1
fi

uv run --with gradio --with langgraph --with langchain-core --with langchain-openai python -m src.ui.gradio_app

