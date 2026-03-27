"""Tests for LangGraph wiring (routers, adapters, full compile + invoke)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from src.agents.graph import (
    SidekickGraphState,
    compile_sidekick_graph,
    lc_messages_to_agent_dicts,
    route_evaluator,
    route_worker,
    sidekick_tool_to_langchain,
)
from src.state import EvaluatorOutput
from src.tools.spec import SidekickTool


@tool
def _echo_fixture(input: str) -> str:
    """Echo input for ToolNode integration."""
    return f"tool:{input}"


def test_route_worker_no_tool_calls_goes_evaluator():
    state: SidekickGraphState = {
        "messages": [HumanMessage(content="h"), AIMessage(content="ok")],
        "success_criteria": "x",
    }
    assert route_worker(state) == "evaluator"


def test_route_worker_with_tool_calls_goes_tools():
    state: SidekickGraphState = {
        "messages": [
            HumanMessage(content="h"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "x",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        "success_criteria": "x",
    }
    assert route_worker(state) == "tools"


def test_route_evaluator_end_when_success():
    state: SidekickGraphState = {
        "messages": [],
        "success_criteria": "x",
        "success_criteria_met": True,
        "user_input_needed": False,
    }
    assert route_evaluator(state) == "__end__"


def test_route_evaluator_end_when_user_input():
    state: SidekickGraphState = {
        "messages": [],
        "success_criteria": "x",
        "success_criteria_met": False,
        "user_input_needed": True,
    }
    assert route_evaluator(state) == "__end__"


def test_route_evaluator_retry_worker():
    state: SidekickGraphState = {
        "messages": [],
        "success_criteria": "x",
        "success_criteria_met": False,
        "user_input_needed": False,
    }
    assert route_evaluator(state) == "worker"


def test_lc_messages_to_agent_dicts_skips_system():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="yo"),
        ToolMessage(content="t", tool_call_id="1"),
    ]
    d = lc_messages_to_agent_dicts(msgs)
    assert d == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "tool", "content": "t"},
    ]


def test_sidekick_tool_to_langchain_roundtrip():
    st = SidekickTool(name="t1", description="d", invoke=lambda s: s.upper())
    lc = sidekick_tool_to_langchain(st)
    assert lc.name == "t1"
    assert lc.invoke({"s": "ab"}) == "AB"


def test_graph_invoke_no_tools_single_pass():
    worker = RunnableLambda(lambda msgs: AIMessage(content="answer"))
    evaluator = RunnableLambda(
        lambda msgs: EvaluatorOutput(
            feedback="fine",
            success_criteria_met=True,
            user_input_needed=False,
        )
    )
    graph = compile_sidekick_graph(
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
        sidekick_tools=[],
    )
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="hello")],
            "success_criteria": "Respond",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
        }
    )
    assert out["success_criteria_met"] is True
    assert "Evaluator Feedback" in str(out["messages"][-1].content)


def test_graph_invoke_worker_evaluator_retry_then_success():
    worker_calls = {"n": 0}

    def worker_fn(msgs):
        worker_calls["n"] += 1
        return AIMessage(content=f"attempt {worker_calls['n']}")

    eval_calls = {"n": 0}

    def eval_fn(msgs):
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return EvaluatorOutput(
                feedback="improve",
                success_criteria_met=False,
                user_input_needed=False,
            )
        return EvaluatorOutput(
            feedback="done",
            success_criteria_met=True,
            user_input_needed=False,
        )

    graph = compile_sidekick_graph(
        llm_worker=RunnableLambda(worker_fn),
        llm_evaluator_structured=RunnableLambda(eval_fn),
        sidekick_tools=[],
    )
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="task")],
            "success_criteria": "finish",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
        }
    )
    assert worker_calls["n"] == 2
    assert eval_calls["n"] == 2
    assert out["success_criteria_met"] is True


def test_graph_invoke_with_tool_node():
    """Worker issues a tool call once, then replies with text; evaluator accepts."""
    st = SidekickTool(
        name="echo_fixture",
        description="echo",
        invoke=lambda s: _echo_fixture.invoke({"input": s}),
    )
    calls = {"w": 0}

    def worker_fn(msgs):
        calls["w"] += 1
        if calls["w"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_fixture",
                        "args": {"s": "ping"},
                        "id": "t1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="finished after tool")

    graph = compile_sidekick_graph(
        llm_worker=RunnableLambda(worker_fn),
        llm_evaluator_structured=RunnableLambda(
            lambda m: EvaluatorOutput(
                feedback="ok",
                success_criteria_met=True,
                user_input_needed=False,
            )
        ),
        sidekick_tools=[st],
    )
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="start")],
            "success_criteria": "use tool then answer",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
        }
    )
    assert calls["w"] == 2
    assert out["success_criteria_met"] is True
    types = [type(m).__name__ for m in out["messages"]]
    assert "ToolMessage" in types
