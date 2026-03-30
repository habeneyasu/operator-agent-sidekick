"""Product-facing helpers to assemble and run the sidekick graph."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from src.agents.graph import create_sidekick_graph_from_settings
from src.config import Settings, get_settings
from src.state import AgentState, EvaluatorOutput
from src.tools.browser import BrowserController
import os


def build_graph_with_runnables(
    *,
    settings: Optional[Settings] = None,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Assemble the graph from provided runnables (no provider-specific imports)."""
    s = settings or get_settings()
    graph = create_sidekick_graph_from_settings(
        s,
        llm_worker=llm_worker,
        llm_evaluator_structured=llm_evaluator_structured,
        checkpointer=checkpointer,
        browser=browser,
        include_browser=include_browser,
    )
    return graph


def run_once_via_runnables(
    *,
    messages: Iterable[str] | Iterable[BaseMessage],
    success_criteria: str,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    settings: Optional[Settings] = None,
    checkpointer: Any | None = None,
):
    """Execute a single graph session given runnables and user messages.

    - If messages are strings, they will be wrapped as HumanMessage instances.
    """
    hist: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, str):
            hist.append(HumanMessage(content=m))
        elif isinstance(m, BaseMessage):
            hist.append(m)
        else:
            hist.append(HumanMessage(content=str(m)))

    graph = build_graph_with_runnables(
        settings=settings,
        llm_worker=llm_worker,
        llm_evaluator_structured=llm_evaluator_structured,
        checkpointer=checkpointer,
    )
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


def build_graph_with_openai(
    *,
    settings: Optional[Settings] = None,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Assemble the graph using ChatOpenAI for worker and evaluator (if available).

    Requires langchain and langchain_openai packages. Falls back with ImportError.
    """
    s = settings or get_settings()
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in integration
        raise ImportError(
            "build_graph_with_openai requires langchain_openai. "
            "Install with `uv add langchain-openai langchain-core langgraph`."
        ) from exc

    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    worker = ChatOpenAI(model=s.openai_model_worker, temperature=0.0, max_tokens=max_tokens)
    evaluator = ChatOpenAI(model=s.openai_model_evaluator, temperature=0.0, max_tokens=max_tokens).with_structured_output(EvaluatorOutput)  # type: ignore[attr-defined]
    # NOTE: We cannot directly pass the Pydantic class from state here via annotations
    # with static reference due to circular import risk; integration path will cover this.
    return create_sidekick_graph_from_settings(
        s,
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
        checkpointer=checkpointer,
        browser=browser,
        include_browser=include_browser,
    )


def build_graph_with_openrouter(
    *,
    settings: Optional[Settings] = None,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Assemble the graph using OpenRouter via ChatOpenAI (OpenAI-compatible API)."""
    s = settings or get_settings()
    api_key = s.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    base_url = s.openrouter_base_url or os.getenv("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("OPENROUTER_API_KEY and OPENROUTER_BASE_URL must be set.")
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "build_graph_with_openrouter requires langchain_openai. "
            "Install with `uv add langchain-openai langchain-core langgraph`."
        ) from exc
    worker_model = (s.openrouter_model_worker or os.getenv("OPENROUTER_MODEL_WORKER") or s.openai_model_worker)
    eval_model = (s.openrouter_model_evaluator or os.getenv("OPENROUTER_MODEL_EVALUATOR") or s.openai_model_evaluator)
    max_tokens_env = s.openrouter_max_tokens if s.openrouter_max_tokens not in (None, 0) else None
    max_tokens = int(max_tokens_env or os.getenv("OPENROUTER_MAX_TOKENS") or os.getenv("OPENAI_MAX_TOKENS", "512"))
    worker = ChatOpenAI(
        model=worker_model,
        temperature=0.0,
        openai_api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,  # type: ignore[arg-type]
    )
    evaluator = ChatOpenAI(
        model=eval_model,
        temperature=0.0,
        openai_api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,  # type: ignore[arg-type]
    ).with_structured_output(EvaluatorOutput)  # type: ignore[attr-defined]
    return create_sidekick_graph_from_settings(
        s,
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
        checkpointer=checkpointer,
        browser=browser,
        include_browser=include_browser,
    )


def build_graph_with_ollama(
    *,
    settings: Optional[Settings] = None,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Assemble the graph using a local Ollama server via ChatOllama.

    Tools are intentionally not bound — local 8B models reliably call tools
    even when they should answer directly, producing empty content responses.
    The auto-retrieve node handles factual grounding instead.
    """
    s = settings or get_settings()
    base_url = s.ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    worker_model = s.ollama_model_worker or os.getenv("OLLAMA_MODEL_WORKER") or "llama3.1:8b"
    eval_model = s.ollama_model_evaluator or os.getenv("OLLAMA_MODEL_EVALUATOR") or "llama3.1:8b"
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "build_graph_with_ollama requires langchain-ollama. "
                "Install with `uv add langchain-ollama langchain-core langgraph`."
            ) from exc
    worker = ChatOllama(
        model=worker_model,
        base_url=base_url,
        temperature=0.0,
        num_ctx=8192,
        num_predict=1024,
    )
    evaluator = ChatOllama(
        model=eval_model,
        base_url=base_url,
        temperature=0.0,
        num_ctx=4096,
        num_predict=256,
    )
    # Compile without tools — pass empty list so no ToolNode is added and
    # bind_tools is never called. The auto-retrieve node handles grounding.
    from src.agents.graph import compile_sidekick_graph
    return compile_sidekick_graph(
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
        sidekick_tools=[],
        checkpointer=checkpointer,
        max_iterations=s.max_agent_iterations,
        tokens_per_run_limit=s.tokens_per_run_limit,
        straight_line=True,
    )