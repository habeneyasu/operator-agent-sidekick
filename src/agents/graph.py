"""LangGraph construction: worker, optional ToolNode, evaluator, conditional edges."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel

from src.agents.evaluator import build_evaluator_chat_messages
from src.config import Settings
from src.state import AgentState, EvaluatorOutput
from src.tools.browser import BrowserController
from src.tools.spec import SidekickTool
from src.tools.tool_registry import build_tools
from src.utils.prompts import build_worker_system_message
from src.logger import get_logger

_log = get_logger("graph")


class SidekickGraphState(TypedDict, total=False):
    """LangGraph state for the sidekick loop (``add_messages`` on ``messages``)."""

    messages: Annotated[list[BaseMessage], add_messages]
    success_criteria: str
    feedback_on_work: str | None
    success_criteria_met: bool
    user_input_needed: bool
    iteration: int
    thread_id: str | None


def sidekick_tool_to_langchain(tool: SidekickTool) -> StructuredTool:
    """Wrap a :class:`SidekickTool` as a LangChain ``StructuredTool`` with a single string input.

    We standardize on a single argument named ``input`` to avoid ambiguity across tools.
    """

    class _SingleStringInput(BaseModel):
        input: str

    def _invoke(input: str) -> str:
        return tool.invoke(input)

    return StructuredTool.from_function(
        coroutine=None,
        func=_invoke,
        name=tool.name,
        description=tool.description,
        args_schema=_SingleStringInput,
    )


def lc_messages_to_agent_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to dicts compatible with :class:`AgentState` / evaluator."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": str(m.content or "")})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": str(m.content)})
    return out


def route_worker(state: SidekickGraphState) -> Literal["tools", "evaluator"]:
    # Guard: if there are no messages yet, go straight to evaluator
    if not state.get("messages"):
        _log.debug("route_worker: no messages -> evaluator")
        return "evaluator"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        _log.debug("route_worker: tool_calls detected -> tools")
        return "tools"
    _log.debug("route_worker: no tool_calls -> evaluator")
    return "evaluator"


def route_evaluator(state: SidekickGraphState) -> Literal["worker", "__end__"]:
    if state.get("success_criteria_met") or state.get("user_input_needed"):
        _log.debug(
            "route_evaluator: end (success=%s user_input=%s)",
            state.get("success_criteria_met"),
            state.get("user_input_needed"),
        )
        return "__end__"
    _log.debug("route_evaluator: continue -> worker")
    return "worker"


def _evaluator_lc_messages(agent_state: AgentState) -> list[BaseMessage]:
    cms = build_evaluator_chat_messages(agent_state)
    lc: list[BaseMessage] = []
    for cm in cms:
        if cm.role == "system":
            lc.append(SystemMessage(content=cm.content))
        elif cm.role == "user":
            lc.append(HumanMessage(content=cm.content))
        else:
            lc.append(HumanMessage(content=cm.content))
    return lc


def _worker_node_fn(
    state: SidekickGraphState,
    llm_with_tools: Runnable,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    sys_content = build_worker_system_message(
        success_criteria=state.get("success_criteria") or "",
        feedback_on_work=state.get("feedback_on_work"),
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    sys_msg = SystemMessage(content=sys_content)
    hist = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    _log.info("worker: invoking LLM (history=%d)", len(hist))
    response = llm_with_tools.invoke([sys_msg, *hist])
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    dt = (time.perf_counter() - t0) * 1000
    _log.info("worker: completed in %.1f ms", dt)
    return {"messages": [response]}


def _evaluator_node_fn(
    state: SidekickGraphState,
    evaluator_llm: Runnable,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    agent_state = AgentState(
        messages=lc_messages_to_agent_dicts(state["messages"]),
        success_criteria=state.get("success_criteria") or "",
        feedback_on_work=state.get("feedback_on_work"),
        success_criteria_met=state.get("success_criteria_met", False),
        user_input_needed=state.get("user_input_needed", False),
        iteration=state.get("iteration", 0),
        thread_id=state.get("thread_id"),
    )
    lc_eval = _evaluator_lc_messages(agent_state)
    _log.info("evaluator: invoking LLM")
    result = evaluator_llm.invoke(lc_eval)
    if not isinstance(result, EvaluatorOutput):
        raise TypeError(f"Evaluator must return EvaluatorOutput, got {type(result)}")
    feedback_line = f"Evaluator Feedback on this answer: {result.feedback}"
    dt = (time.perf_counter() - t0) * 1000
    _log.info(
        "evaluator: completed in %.1f ms (success=%s user_input=%s)",
        dt,
        result.success_criteria_met,
        result.user_input_needed,
    )
    return {
        # Append evaluator feedback as a SystemMessage so UIs don't show it as assistant text
        "messages": [SystemMessage(content=feedback_line)],
        "feedback_on_work": result.feedback,
        "success_criteria_met": result.success_criteria_met,
        "user_input_needed": result.user_input_needed,
        # Increment iteration after each evaluator pass
        "iteration": state.get("iteration", 0) + 1,
    }


def compile_sidekick_graph(
    *,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    sidekick_tools: list[SidekickTool],
    checkpointer: Any | None = None,
    max_iterations: int | None = None,
):
    """Compile the sidekick graph.

    ``llm_worker`` should be a chat model **with** ``bind_tools`` applied when
    ``sidekick_tools`` is non-empty. Pass a plain model when there are no tools.
    """
    from langgraph.prebuilt import ToolNode

    lc_tool_wrappers = [sidekick_tool_to_langchain(t) for t in sidekick_tools]

    def worker(state: SidekickGraphState) -> dict[str, Any]:
        return _worker_node_fn(state, llm_worker)

    def evaluator(state: SidekickGraphState) -> dict[str, Any]:
        return _evaluator_node_fn(state, llm_evaluator_structured)

    # Router with optional iteration cap (ends when iteration >= max_iterations)
    def _route_evaluator_with_cap(state: SidekickGraphState) -> Literal["worker", "__end__"]:
        if max_iterations is not None and state.get("iteration", 0) >= max_iterations:
            _log.info(
                "iteration cap reached: %s >= %s -> end",
                state.get("iteration", 0),
                max_iterations,
            )
            return "__end__"
        return route_evaluator(state)

    graph_builder = StateGraph(SidekickGraphState)
    graph_builder.add_node("worker", worker)
    graph_builder.add_node("evaluator", evaluator)

    if lc_tool_wrappers:
        graph_builder.add_node("tools", ToolNode(lc_tool_wrappers))
        graph_builder.add_edge(START, "worker")
        graph_builder.add_conditional_edges(
            "worker",
            route_worker,
            {"tools": "tools", "evaluator": "evaluator"},
        )
        graph_builder.add_edge("tools", "worker")
    else:
        graph_builder.add_edge(START, "worker")
        graph_builder.add_edge("worker", "evaluator")

    graph_builder.add_conditional_edges(
        "evaluator",
        _route_evaluator_with_cap,
        {"worker": "worker", "__end__": END},
    )

    return graph_builder.compile(checkpointer=checkpointer)


def create_sidekick_graph_from_settings(
    settings: Settings,
    *,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Build tools from settings and compile the graph."""
    tools = build_tools(
        settings,
        browser=browser,
        include_browser=include_browser,
    )
    lc_tools = tools
    # Bind tools only if the worker supports it; otherwise, fall back to no tools.
    if lc_tools and hasattr(llm_worker, "bind_tools"):
        llm_bound = getattr(llm_worker, "bind_tools")(
            [sidekick_tool_to_langchain(t) for t in lc_tools]
        )
    else:
        llm_bound = llm_worker
        if lc_tools and not hasattr(llm_worker, "bind_tools"):
            # Disable tools to avoid ToolNode path when worker cannot issue tool calls
            lc_tools = []
    return compile_sidekick_graph(
        llm_worker=llm_bound,
        llm_evaluator_structured=llm_evaluator_structured,
        sidekick_tools=lc_tools,
        checkpointer=checkpointer,
        max_iterations=settings.max_agent_iterations,
    )
