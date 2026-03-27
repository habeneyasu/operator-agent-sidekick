from src.llm.base import ChatMessage, LLMRequest, LLMResponse


def test_llm_request_defaults():
    request = LLMRequest(messages=[ChatMessage(role="user", content="hello")])

    assert request.temperature == 0.0
    assert request.max_tokens is None
    assert request.model is None
    assert request.metadata == {}


def test_llm_response_shape():
    response = LLMResponse(content="ok", model="gpt-4o-mini", raw={"id": "1"})
    assert response.content == "ok"
    assert response.model == "gpt-4o-mini"
    assert response.raw == {"id": "1"}
