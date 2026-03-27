"""Worker node: build prompts, normalize history, call the LLM, append assistant turn."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMClient, ChatMessage, LLMRequest, LLMResponse
from src.state import AgentState
from src.utils.prompts import build_worker_system_message


def _coerce_role(role: str) -> str:
    r = role.strip().lower()
    if r in {"human", "user"}:
        return "user"
    if r in {"ai", "assistant"}:
        return "assistant"
    if r == "system":
        return "system"
    return r


def message_to_chat_messages(message: Any) -> list[ChatMessage]:
    """Convert a single graph/message entry into zero or more ChatMessage rows."""
    if message is None:
        return []

    if isinstance(message, ChatMessage):
        return [message]

    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return []
        role_s = _coerce_role(str(role))
        if role_s == "system":
            return []
        return [ChatMessage(role=role_s, content=str(content))]

    role = getattr(message, "role", None) or getattr(message, "type", None)
    content = getattr(message, "content", None)
    if role is None or content is None:
        return []
    role_str = str(role).lower()
    if role_str in {"human", "user"}:
        norm = "user"
    elif role_str in {"ai", "assistant"}:
        norm = "assistant"
    elif role_str == "system":
        return []
    else:
        norm = str(role)
    return [ChatMessage(role=norm, content=str(content))]


def history_to_chat_messages(messages: list[Any]) -> list[ChatMessage]:
    """Flatten graph message list to provider chat rows; drop system turns (re-injected by worker)."""
    out: list[ChatMessage] = []
    for item in messages:
        out.extend(message_to_chat_messages(item))
    return out


def build_worker_chat_messages(
    state: AgentState,
    *,
    current_time: str | None = None,
) -> list[ChatMessage]:
    """Assemble [system, ...history] for the worker LLM call."""
    system_text = build_worker_system_message(
        success_criteria=state.success_criteria,
        feedback_on_work=state.feedback_on_work,
        current_time=current_time,
    )
    system = ChatMessage(role="system", content=system_text)
    return [system, *history_to_chat_messages(state.messages)]


def build_worker_request(
    state: AgentState,
    *,
    model: str | None = None,
    current_time: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMRequest:
    """Build the LLM request for one worker step (no tool binding yet)."""
    return LLMRequest(
        messages=build_worker_chat_messages(state, current_time=current_time),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def assistant_reply_message(content: str) -> dict[str, str]:
    """Standard shape for an assistant turn appended to graph state (LangGraph-friendly dict)."""
    return {"role": "assistant", "content": content}


def run_worker(
    state: AgentState,
    llm: BaseLLMClient,
    *,
    model: str | None = None,
    current_time: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[AgentState, LLMResponse]:
    """Invoke the worker LLM and return state with the new assistant message appended."""
    request = build_worker_request(
        state,
        model=model,
        current_time=current_time,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.generate(request)
    new_messages = [*state.messages, assistant_reply_message(response.content)]
    new_state = state.model_copy(update={"messages": new_messages})
    return new_state, response
