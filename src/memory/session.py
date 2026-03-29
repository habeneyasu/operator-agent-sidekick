"""Session/thread helpers for LangGraph checkpointer usage."""

from __future__ import annotations

import uuid
from typing import Dict


def new_thread_id() -> str:
    """Generate a new unique thread identifier."""
    return str(uuid.uuid4())


def thread_config(thread_id: str) -> Dict[str, Dict[str, str]]:
    """Return the LangGraph config dict that carries the thread_id for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}

