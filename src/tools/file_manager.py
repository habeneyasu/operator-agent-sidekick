"""Sandboxed file read/write/list relative to a root directory."""

from __future__ import annotations

import json
from pathlib import Path

from src.tools.spec import SidekickTool


def _resolve_sandbox_path(root: Path, relative: str) -> Path:
    root_res = root.resolve()
    candidate = (root_res / relative.strip().lstrip("/")).resolve()
    if not candidate.is_relative_to(root_res):
        raise ValueError("Path escapes sandbox")
    return candidate


def read_sandbox_file(root: Path, relative_path: str) -> str:
    path = _resolve_sandbox_path(root, relative_path)
    if not path.is_file():
        return f"Error: not a file: {relative_path}"
    return path.read_text(encoding="utf-8", errors="replace")


def write_sandbox_file(root: Path, relative_path: str, content: str) -> str:
    path = _resolve_sandbox_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {relative_path}"


def list_sandbox_dir(root: Path, relative_path: str = ".") -> str:
    path = _resolve_sandbox_path(root, relative_path)
    if not path.is_dir():
        return f"Error: not a directory: {relative_path}"
    entries = sorted(path.iterdir(), key=lambda p: p.name)
    lines = [e.name + ("/" if e.is_dir() else "") for e in entries]
    return "\n".join(lines) if lines else "(empty)"


def file_manager_tools(sandbox_root: Path) -> list[SidekickTool]:
    root = sandbox_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def read_tool(payload: str) -> str:
        try:
            if payload.strip().startswith("{"):
                data = json.loads(payload)
                return read_sandbox_file(root, str(data["path"]))
            return read_sandbox_file(root, payload.strip())
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            return f"Error: {exc}"

    def write_tool(payload: str) -> str:
        try:
            data = json.loads(payload)
            path = str(data["path"])
            content = str(data.get("content", ""))
            return write_sandbox_file(root, path, content)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            return f"Error: {exc}. Expected JSON: {{\"path\": \"...\", \"content\": \"...\"}}"

    def list_tool(payload: str) -> str:
        try:
            rel = payload.strip() or "."
            return list_sandbox_dir(root, rel)
        except (ValueError, OSError) as exc:
            return f"Error: {exc}"

    return [
        SidekickTool(
            name="read_sandbox_file",
            description='Read a UTF-8 text file under the sandbox. Input: JSON {"path": "relative/path"} or a plain relative path string.',
            invoke=read_tool,
        ),
        SidekickTool(
            name="write_sandbox_file",
            description='Write a UTF-8 text file under the sandbox. Input: JSON {"path": "relative/path", "content": "..."}.',
            invoke=write_tool,
        ),
        SidekickTool(
            name="list_sandbox_dir",
            description="List entries in a sandbox directory. Input: relative path or empty for sandbox root.",
            invoke=list_tool,
        ),
    ]
