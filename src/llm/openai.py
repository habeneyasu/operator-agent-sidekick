"""OpenAI implementation of the provider-agnostic LLM client."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMClient, LLMRequest, LLMResponse


class OpenAIClient(BaseLLMClient):
    """Thin adapter around the official OpenAI Python client."""

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout_seconds: int = 30,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        if not default_model:
            raise ValueError("A default model is required")

        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install it with `uv add openai`."
            ) from exc

        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        completion = self._client.chat.completions.create(**payload)
        content = completion.choices[0].message.content or ""
        return LLMResponse(content=content, model=model, raw=completion)
