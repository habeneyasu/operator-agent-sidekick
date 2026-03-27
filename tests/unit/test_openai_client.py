from types import SimpleNamespace

from src.llm.base import ChatMessage, LLMRequest
from src.llm.openai import OpenAIClient


class _FakeCompletions:
    def __init__(self):
        self.last_payload = None

    def create(self, **kwargs):
        self.last_payload = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="mocked output"))]
        )


class _FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_client_generate_maps_payload_and_response():
    fake_client = _FakeOpenAI()
    client = OpenAIClient(
        api_key="test-key",
        default_model="gpt-4o-mini",
        client=fake_client,
    )

    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ],
        temperature=0.2,
        max_tokens=128,
    )
    response = client.generate(request)
    payload = fake_client.chat.completions.last_payload

    assert payload["model"] == "gpt-4.1-mini"
    assert payload["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 128
    assert response.content == "mocked output"
    assert response.model == "gpt-4.1-mini"


def test_openai_client_uses_default_model_when_request_model_missing():
    fake_client = _FakeOpenAI()
    client = OpenAIClient(
        api_key="test-key",
        default_model="gpt-4o-mini",
        client=fake_client,
    )
    request = LLMRequest(messages=[ChatMessage(role="user", content="ping")])
    response = client.generate(request)

    assert fake_client.chat.completions.last_payload["model"] == "gpt-4o-mini"
    assert response.model == "gpt-4o-mini"


def test_openai_client_requires_api_key():
    try:
        OpenAIClient(api_key="", default_model="gpt-4o-mini", client=_FakeOpenAI())
        assert False, "Expected ValueError when api_key is missing"
    except ValueError:
        assert True
