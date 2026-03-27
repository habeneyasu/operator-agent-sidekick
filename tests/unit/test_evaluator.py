import pytest

from src.agents.evaluator import (
    build_evaluator_request,
    last_assistant_content,
    messages_to_evaluator_turns,
    run_evaluator,
)
from src.llm.base import BaseLLMClient, LLMRequest, LLMResponse
from src.state import AgentState, EvaluatorOutput
from src.utils.parsing import EvaluatorParseError, parse_evaluator_output


class RecordingLLM(BaseLLMClient):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        model = request.model or "default-model"
        return LLMResponse(content=self.reply, model=model, raw=None)


def test_messages_to_evaluator_turns_maps_roles():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert messages_to_evaluator_turns(msgs) == [("User", "hi"), ("Assistant", "hello")]


def test_messages_to_evaluator_turns_skips_system():
    assert messages_to_evaluator_turns([{"role": "system", "content": "x"}]) == []


def test_last_assistant_content_ok():
    assert last_assistant_content(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    ) == "a"


def test_last_assistant_content_requires_assistant_last():
    with pytest.raises(ValueError):
        last_assistant_content([{"role": "user", "content": "only user"}])


def test_build_evaluator_request_includes_json_instruction():
    state = AgentState(
        success_criteria="Say OK",
        messages=[
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "OK"},
        ],
        feedback_on_work="Previous note",
    )
    req = build_evaluator_request(state, model="eval-model")
    assert req.model == "eval-model"
    assert len(req.messages) == 2
    assert req.messages[0].role == "system"
    assert req.messages[1].role == "user"
    body = req.messages[1].content
    assert "Say OK" in body
    assert "OK" in body
    assert "Previous note" in body
    assert "success_criteria_met" in body


def test_parse_evaluator_output_raw_json():
    raw = (
        '{"feedback":"good","success_criteria_met":true,"user_input_needed":false}'
    )
    out = parse_evaluator_output(raw)
    assert out == EvaluatorOutput(
        feedback="good",
        success_criteria_met=True,
        user_input_needed=False,
    )


def test_parse_evaluator_output_strips_markdown_fence():
    raw = (
        '```json\n{"feedback":"x","success_criteria_met":false,'
        '"user_input_needed":true}\n```'
    )
    out = parse_evaluator_output(raw)
    assert out.feedback == "x"
    assert out.success_criteria_met is False
    assert out.user_input_needed is True


def test_parse_evaluator_output_invalid_raises():
    with pytest.raises(EvaluatorParseError):
        parse_evaluator_output("not json")


def test_run_evaluator_updates_flags_and_appends_message():
    state = AgentState(
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ],
    )
    payload = (
        '{"feedback":"Needs work","success_criteria_met":false,"user_input_needed":false}'
    )
    llm = RecordingLLM(payload)
    new_state, result, response = run_evaluator(state, llm, model="m-eval")

    assert result.feedback == "Needs work"
    assert result.success_criteria_met is False
    assert result.user_input_needed is False
    assert new_state.feedback_on_work == "Needs work"
    assert new_state.success_criteria_met is False
    assert new_state.user_input_needed is False
    assert len(new_state.messages) == 3
    assert "Evaluator Feedback" in new_state.messages[-1]["content"]
    assert response.content == payload
    assert llm.last_request is not None
