"""Checkpoint saver wrappers."""

from __future__ import annotations

try:
    # Prefer the official in-memory saver from LangGraph for simplicity and testability
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except Exception:  # pragma: no cover
    MemorySaver = None  # type: ignore


def build_memory_saver():
    """Return an in-memory LangGraph checkpointer.

    This implementation is process-local and suitable for tests and dev.
    """
    if MemorySaver is None:  # pragma: no cover
        raise ImportError(
            "LangGraph checkpointing not available. "
            "Install with `uv add langgraph` to use build_memory_saver()."
        )
    return MemorySaver()

