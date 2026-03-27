"""Shared tool types (keeps modules importable without cycles)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SidekickTool:
    """Minimal tool contract for documentation and future LangChain binding."""

    name: str
    description: str
    invoke: Callable[[str], str]

    def run(self, tool_input: str) -> str:
        """Run the tool; alias for invoke (clearer at call sites)."""
        return self.invoke(tool_input)
