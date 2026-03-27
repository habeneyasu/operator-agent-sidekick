from src.agents.worker import (
    build_worker_chat_messages,
    build_worker_request,
    history_to_chat_messages,
    message_to_chat_messages,
    run_worker,
)
from src.llm.base import BaseLLMClient, ChatMessage, LLMRequest, LLMResponse
from src.state import AgentState


class RecordingLLM(BaseLLMClient):
    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        model = request.model or "default-model"
        return LLMResponse(content=self.reply, model=model, raw=None)


def test_message_to_chat_messages_dict():
    assert message_to_chat_messages({"role": "user", "content": "hi"}) == [
        ChatMessage(role="user", content="hi")
    ]


def test_message_to_chat_messages_skips_system_dict():
    assert message_to_chat_messages({"role": "system", "content": "x"}) == []


def test_message_to_chat_messages_passes_chat_message():
    msg = ChatMessage(role="user", content="u")
    assert message_to_chat_messages(msg) == [msg]


def test_history_drops_system_and_keeps_turns():
    hist = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]
    rows = history_to_chat_messages(hist)
    assert rows == [
        ChatMessage(role="user", content="ping"),
        ChatMessage(role="assistant", content="pong"),
    ]


def test_build_worker_chat_messages_starts_with_system_and_criteria():
    state = AgentState(
        success_criteria="Must say DONE",
        messages=[{"role": "user", "content": "go"}],
    )
    msgs = build_worker_chat_messages(state)
    assert msgs[0].role == "system"
    assert "Must say DONE" in msgs[0].content
    assert msgs[1] == ChatMessage(role="user", content="go")


def test_build_worker_chat_messages_includes_evaluator_feedback():
    state = AgentState(
        success_criteria="X",
        feedback_on_work="Be more concise",
        messages=[],
    )
    msgs = build_worker_chat_messages(state)
    assert "Be more concise" in msgs[0].content


def test_build_worker_request_forwards_model_temperature():
    state = AgentState(messages=[])
    req = build_worker_request(
        state,
        model="gpt-custom",
        temperature=0.3,
        max_tokens=100,
    )
    assert req.model == "gpt-custom"
    assert req.temperature == 0.3
    assert req.max_tokens == 100


def test_run_worker_appends_assistant_and_returns_response():
    state = AgentState(
        success_criteria="Reply with OK",
        messages=[{"role": "user", "content": "ping"}],
    )
    llm = RecordingLLM(reply="OK")
    new_state, response = run_worker(state, llm, model="m1")

    assert response.content == "OK"
    assert len(new_state.messages) == 2
    assert new_state.messages[-1] == {"role": "assistant", "content": "OK"}
    assert llm.last_request is not None
    assert llm.last_request.model == "m1"
    assert len(state.messages) == 1
