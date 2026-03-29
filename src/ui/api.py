from __future__ import annotations

from typing import Any, Iterable, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from src.sidekick import (
    build_graph_with_openai,
    build_graph_with_openrouter,
    run_once_via_runnables,
)
from src.state import AgentState
from src.config import Settings, get_settings
from src.memory.saver import build_memory_saver
from src.memory.session import new_thread_id, thread_config


def run_once_openai(
    *,
    user_messages: Iterable[str] | Iterable[BaseMessage],
    success_criteria: str,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """One-shot helper using OpenAI-backed graph assembly."""
    s = settings or get_settings()
    graph = build_graph_with_openai(settings=s)
    hist: list[BaseMessage] = []
    for m in user_messages:
        if isinstance(m, BaseMessage):
            hist.append(m)
        else:
            hist.append(HumanMessage(content=str(m)))
    return graph.invoke(
        {
            "messages": hist,
            "success_criteria": success_criteria,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
            "thread_id": None,
        }
    )


def run_once_openrouter(
    *,
    user_messages: Iterable[str] | Iterable[BaseMessage],
    success_criteria: str,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """One-shot helper using OpenRouter-backed graph assembly."""
    s = settings or get_settings()
    graph = build_graph_with_openrouter(settings=s)
    hist: list[BaseMessage] = []
    for m in user_messages:
        if isinstance(m, BaseMessage):
            hist.append(m)
        else:
            hist.append(HumanMessage(content=str(m)))
    return graph.invoke(
        {
            "messages": hist,
            "success_criteria": success_criteria,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
            "thread_id": None,
        }
    )

def run_with_resume_openai(
    *,
    first_messages: Iterable[str],
    success_criteria: str,
    settings: Optional[Settings] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-step demonstration: first pass stops by cap, second pass resumes and finishes."""
    s = settings or get_settings()
    saver = build_memory_saver()
    graph = build_graph_with_openai(settings=s, checkpointer=saver)
    tid = new_thread_id()
    cfg = thread_config(tid)

    out1 = graph.invoke(
        {
            "messages": [HumanMessage(content=m) for m in first_messages],
            "success_criteria": success_criteria,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
            "thread_id": tid,
        },
        config=cfg,
    )
    out2 = graph.invoke({}, config=cfg)
    return out1, out2


def run_with_resume_openrouter(
    *,
    first_messages: Iterable[str],
    success_criteria: str,
    settings: Optional[Settings] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-step demonstration with OpenRouter."""
    s = settings or get_settings()
    saver = build_memory_saver()
    graph = build_graph_with_openrouter(settings=s, checkpointer=saver)
    tid = new_thread_id()
    cfg = thread_config(tid)

    out1 = graph.invoke(
        {
            "messages": [HumanMessage(content=m) for m in first_messages],
            "success_criteria": success_criteria,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
            "thread_id": tid,
        },
        config=cfg,
    )
    out2 = graph.invoke({}, config=cfg)
    return out1, out2

