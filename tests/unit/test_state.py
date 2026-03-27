from src.state import AgentState, EvaluatorOutput, DEFAULT_SUCCESS_CRITERIA


def test_agent_state_defaults():
    state = AgentState()

    assert state.messages == []
    assert state.success_criteria == DEFAULT_SUCCESS_CRITERIA
    assert state.feedback_on_work is None
    assert state.success_criteria_met is False
    assert state.user_input_needed is False
    assert state.iteration == 0
    assert state.thread_id is None


def test_success_criteria_whitespace_falls_back_to_default():
    state = AgentState(success_criteria="   ")
    assert state.success_criteria == DEFAULT_SUCCESS_CRITERIA


def test_next_iteration_increments_and_keeps_feedback():
    state = AgentState(feedback_on_work="initial feedback", iteration=1)
    updated = state.next_iteration()

    assert updated.iteration == 2
    assert updated.feedback_on_work == "initial feedback"
    assert state.iteration == 1


def test_next_iteration_overrides_feedback_when_provided():
    state = AgentState(feedback_on_work="old")
    updated = state.next_iteration(feedback="new")

    assert updated.feedback_on_work == "new"
    assert updated.iteration == 1


def test_apply_evaluator_output_updates_state_flags():
    state = AgentState()
    result = EvaluatorOutput(
        feedback="Need more detail",
        success_criteria_met=False,
        user_input_needed=True,
    )

    updated = state.apply_evaluator_output(result)

    assert updated.feedback_on_work == "Need more detail"
    assert updated.success_criteria_met is False
    assert updated.user_input_needed is True


def test_evaluator_output_requires_non_empty_feedback():
    try:
        EvaluatorOutput(
            feedback="",
            success_criteria_met=True,
            user_input_needed=False,
        )
        assert False, "Expected validation error for empty feedback"
    except Exception:
        assert True
