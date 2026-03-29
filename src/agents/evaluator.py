"""Evaluator node: score the worker's last reply and update graph flags."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLMClient, ChatMessage, LLMRequest, LLMResponse
from src.state import AgentState, EvaluatorOutput
from src.utils.parsing import parse_evaluator_output
from src.utils.prompts import (
    build_evaluator_system_message,
    build_evaluator_user_message,
    evaluator_json_response_instruction,
    format_conversation_lines,
)


def _label_and_content(message: Any) -> tuple[str, str] | None:
    if message is None:
        return None

    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return None
        r = str(role).lower()
        if r == "system":
            return None
        if r in {"human", "user"}:
            return ("User", str(content))
        if r in {"ai", "assistant"}:
            text = str(content) if content else "[Tools use]"
            return ("Assistant", text)
        return (str(role), str(content))

    role = getattr(message, "type", None) or getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None:
        return None
    r = str(role).lower()
    if r == "system":
        return None
    if r in {"human", "user"}:
        return ("User", str(content) if content is not None else "")
    if r in {"ai", "assistant"}:
        text = str(content) if content else "[Tools use]"
        return ("Assistant", text)
    return (str(role), str(content) if content is not None else "")


def messages_to_evaluator_turns(messages: list[Any]) -> list[tuple[str, str]]:
    """Map graph messages to (User|Assistant, text) pairs for the evaluator prompt."""
    turns: list[tuple[str, str]] = []
    for item in messages:
        pair = _label_and_content(item)
        if pair is not None:
            turns.append(pair)
    return turns


def last_assistant_content(messages: list[Any]) -> str:
    """Return the content of the last assistant message (required for evaluation)."""
    if not messages:
        raise ValueError("Evaluator requires a non-empty message list")

    last = messages[-1]
    if isinstance(last, dict):
        role = str(last.get("role", "")).lower()
        if role not in {"assistant", "ai"}:
            raise ValueError("Last message must be an assistant reply")
        return str(last.get("content", ""))

    role = getattr(last, "type", None) or getattr(last, "role", None)
    if role is None:
        raise ValueError("Last message must be an assistant reply")
    if str(role).lower() not in {"assistant", "ai"}:
        raise ValueError("Last message must be an assistant reply")
    content = getattr(last, "content", None)
    return str(content) if content is not None else ""


def build_evaluator_chat_messages(
    state: AgentState,
) -> list[ChatMessage]:
    """Build [system, user] messages for the evaluator LLM call."""
    turns = messages_to_evaluator_turns(state.messages)
    conversation_text = format_conversation_lines(turns)
    last_response = last_assistant_content(state.messages)

    user_body = build_evaluator_user_message(
        conversation_text=conversation_text,
        success_criteria=state.success_criteria,
        last_assistant_response=last_response,
        prior_feedback=state.feedback_on_work,
    )
    user_content = f"{user_body}\n\n{evaluator_json_response_instruction()}"

    return [
        ChatMessage(role="system", content=build_evaluator_system_message()),
        ChatMessage(role="user", content=user_content),
    ]


def build_evaluator_request(
    state: AgentState,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMRequest:
    """Build the LLM request for one evaluator step."""
    return LLMRequest(
        messages=build_evaluator_chat_messages(state),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def evaluator_feedback_message(feedback: str) -> dict[str, str]:
    """System-shaped message recording evaluator feedback (hidden from typical UIs)."""
    return {
        "role": "system",
        "content": f"Evaluator Feedback on this answer: {feedback}",
    }


def run_evaluator(
    state: AgentState,
    llm: BaseLLMClient,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[AgentState, EvaluatorOutput, LLMResponse]:
    """Call the evaluator LLM, parse structured output, update state and transcript."""
    request = build_evaluator_request(
        state,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.generate(request)
    result = parse_evaluator_output(response.content)

    with_flags = state.apply_evaluator_output(result)
    new_messages = [
        *state.messages,
        evaluator_feedback_message(result.feedback),
    ]
    new_state = with_flags.model_copy(update={"messages": new_messages})
    return new_state, result, response


__all__ = [
    "build_evaluator_chat_messages",
    "build_evaluator_request",
    "evaluator_feedback_message",
    "last_assistant_content",
    "messages_to_evaluator_turns",
    "run_evaluator",
]
