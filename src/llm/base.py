"""LLM abstraction layer shared by providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """Provider-agnostic chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """Unified request payload for chat completion."""

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Unified response payload for chat completion."""

    content: str
    model: str
    raw: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a model response for a chat request."""
        raise NotImplementedError
