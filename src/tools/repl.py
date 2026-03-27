"""Run Python in a subprocess with timeout (never uses exec in-process)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from src.tools.spec import SidekickTool

MAX_CODE_CHARS = 16_384
MAX_OUTPUT_CHARS = 24_000
DEFAULT_TIMEOUT_SECONDS = 20


def run_python_code(code: str, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    stripped = textwrap.dedent(code).strip()
    if not stripped:
        return "Error: empty code"
    if len(stripped) > MAX_CODE_CHARS:
        return f"Error: code exceeds max length ({MAX_CODE_CHARS} chars)"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", stripped],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution exceeded {timeout_seconds}s timeout"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = []
    if out:
        combined.append(out)
    if err:
        combined.append(f"stderr:\n{err}")
    text = "\n".join(combined) if combined else f"(exit code {proc.returncode}, no output)"
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[: MAX_OUTPUT_CHARS - 20] + "\n... (truncated)"
    if proc.returncode != 0:
        return f"Exit {proc.returncode}\n{text}"
    return text


def repl_tool(timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> SidekickTool:
    def invoke(code: str) -> str:
        return run_python_code(code, timeout_seconds=timeout_seconds)

    return SidekickTool(
        name="run_python",
        description="Run Python in an isolated subprocess (stdout/stderr only; use print() for output). Input: Python source.",
        invoke=invoke,
    )
