from src.utils.prompts import (
    build_evaluator_system_message,
    build_evaluator_user_message,
    build_worker_system_message,
    evaluator_json_response_instruction,
    format_conversation_lines,
)


def test_worker_system_includes_success_criteria():
    text = build_worker_system_message(success_criteria="Must return OK only")
    assert "Must return OK only" in text
    assert "success criteria" in text.lower()


def test_worker_system_appends_feedback_when_present():
    base_only = build_worker_system_message(success_criteria="X")
    with_feedback = build_worker_system_message(
        success_criteria="X",
        feedback_on_work="Too vague",
    )
    assert "Too vague" in with_feedback
    assert len(with_feedback) > len(base_only)


def test_worker_system_includes_time_when_provided():
    text = build_worker_system_message(
        success_criteria="Y",
        current_time="2026-03-27 12:00:00",
    )
    assert "2026-03-27" in text


def test_eval_evaluator_user_includes_blocks():
    text = build_evaluator_user_message(
        conversation_text="User: hi",
        success_criteria="Be polite",
        last_assistant_response="Hello!",
    )
    assert "User: hi" in text
    assert "Be polite" in text
    assert "Hello!" in text


def test_evaluator_user_appends_prior_feedback():
    text = build_evaluator_user_message(
        conversation_text="",
        success_criteria="",
        last_assistant_response="",
        prior_feedback="Fix tone",
    )
    assert "Fix tone" in text


def test_evaluator_system_non_empty():
    assert len(build_evaluator_system_message()) > 50


def test_format_conversation_lines():
    out = format_conversation_lines(
        [("User", "Ping"), ("Assistant", "Pong")],
    )
    assert "User: Ping" in out
    assert "Assistant: Pong" in out


def test_evaluator_json_response_instruction_non_empty():
    text = evaluator_json_response_instruction()
    assert "JSON" in text
    assert "success_criteria_met" in text
