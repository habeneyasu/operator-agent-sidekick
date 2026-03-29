from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.sidekick import run_once_via_runnables
from src.state import EvaluatorOutput


def test_run_once_via_runnables_smoke():
    # Worker: reply once
    worker = RunnableLambda(lambda msgs: AIMessage(content="answer"))
    # Evaluator: mark as success in one pass
    evaluator = RunnableLambda(
        lambda msgs: EvaluatorOutput(
            feedback="ok",
            success_criteria_met=True,
            user_input_needed=False,
        )
    )
    out = run_once_via_runnables(
        messages=[HumanMessage(content="hello")],
        success_criteria="Respond",
        llm_worker=worker,
        llm_evaluator_structured=evaluator,
    )
    assert out["success_criteria_met"] is True
    assert "Evaluator Feedback" in str(out["messages"][-1].content)

