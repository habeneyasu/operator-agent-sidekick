"""Prompt templates for worker and evaluator nodes.

Keep all long-form system/user strings here so agent logic stays thin and prompts
can be versioned and tested in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable


def build_worker_system_message(
    *,
    success_criteria: str,
    feedback_on_work: str | None = None,
    current_time: str | None = None,
) -> str:
    """Build the worker system message (tools, success criteria, optional retry feedback)."""
    time_line = (
        f"The current date and time is {current_time}."
        if current_time
        else ""
    ).strip()

    base = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigate, and retrieve web pages.
You have a tool to run Python code; include print() when you need visible output.
{time_line}

This is the success criteria:
{success_criteria}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, reply clearly, for example:

Question: please clarify whether you want a summary or a detailed answer

If you have finished, reply with the final answer only; do not ask a question."""

    if not feedback_on_work:
        return base.strip()

    retry = f"""

Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{feedback_on_work}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""
    return (base + retry).strip()


def build_evaluator_system_message() -> str:
    """System prompt for the evaluator LLM (when not using structured output instructions from the provider)."""
    return """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""


def build_evaluator_user_message(
    *,
    conversation_text: str,
    success_criteria: str,
    last_assistant_response: str,
    prior_feedback: str | None = None,
) -> str:
    """User-side content for the evaluator: conversation, criteria, and last answer."""
    msg = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{conversation_text}

The success criteria for this assignment is:
{success_criteria}

And the final response from the Assistant that you are evaluating is:
{last_assistant_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

The Assistant has access to tools to write files. If the Assistant says they have written a file, you can assume they have done so.
Give the Assistant the benefit of the doubt when they claim they completed an action, but reject if more work is still needed."""

    if prior_feedback:
        msg += f"""

Also, note that in a prior attempt from the Assistant, you provided this feedback: {prior_feedback}
If you see the Assistant repeating the same mistakes, consider responding that user input is required."""

    return msg.strip()


def evaluator_json_response_instruction() -> str:
    """Tell the model to return only JSON matching :class:`EvaluatorOutput`."""
    return (
        "You must respond with a single JSON object only (no markdown fences, no prose). "
        'Use exactly these keys: "feedback" (string), '
        '"success_criteria_met" (boolean), "user_input_needed" (boolean).'
    )


def format_conversation_lines(
    turns: Iterable[tuple[str, str]],
) -> str:
    """Format (role_label, content) pairs as a plain-text block for the evaluator.

    role_label examples: "User", "Assistant".
    """
    lines: list[str] = ["Conversation history:", ""]
    for role, content in turns:
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
