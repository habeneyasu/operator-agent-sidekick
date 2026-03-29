from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agents.graph import compile_sidekick_graph
from src.memory.saver import build_memory_saver
from src.memory.session import new_thread_id, thread_config
from src.state import EvaluatorOutput


def test_resume_with_thread_id_and_checkpointer():
    """First run stops due to iteration cap; second run resumes and succeeds."""
    # Worker returns a simple assistant message every time
    worker = RunnableLambda(lambda msgs: AIMessage(content="attempt"))

    # Evaluator: first call -> retry; second call -> success
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

    evaluator = RunnableLambda(eval_fn)

    saver = build_memory_saver()
    graph = compile_sidekick_graph(
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
        sidekick_tools=[],
        checkpointer=saver,
        max_iterations=1,  # force stop after first evaluator pass
    )

    tid = new_thread_id()
    cfg = thread_config(tid)

    # First invocation: will not meet success, ends due to iteration cap
    out1 = graph.invoke(
        {
            "messages": [HumanMessage(content="start")],
            "success_criteria": "finish",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
            "thread_id": tid,
        },
        config=cfg,
    )
    assert out1["success_criteria_met"] is False
    assert out1["iteration"] == 1

    # Second invocation: no new input; resumes from checkpoint and should succeed
    out2 = graph.invoke({}, config=cfg)
    assert out2["success_criteria_met"] is True
    # Ensure evaluator was called twice across both runs
    assert eval_calls["n"] == 2

