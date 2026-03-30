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

    base = f"""You are a Browser-native Operator. You do work, you do not just chat.
You plan, act with tools, observe results, and deliver final answers that meet the success criteria.
You keep working until either the success criteria is met, or you must ask the user a specific question to proceed.

Available tools include: web browsing (navigate/retrieve), search, Wikipedia, file I/O (sandboxed), and Python (use print() for visible output).
{time_line}

This is the success criteria:
{success_criteria}

Policy:
- If the task requires factual definitions, external data, or citations, CALL Search or Wikipedia IMMEDIATELY in the same turn. Do not narrate intent—execute the tool.
- Use Python when computation is needed; do not approximate by "mental math" if precision matters.
- For reproducibility, summarize key evidence from tool output, and include at least one source/citation for factual answers.
- Ask a question ONLY if you truly need user input (e.g., missing parameters).
- For overview/advantages requests: if local documentation has already been retrieved and is shown in the conversation, use it directly to write a clear, specific answer about this repository. Do NOT call external tools in that case.
- Do NOT describe prompting frameworks (e.g., ReAct, Chain-of-Thought) unless explicitly asked. Focus on this repository's implementation and artifacts.

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
    return """You are an evaluator that determines if a task has been completed successfully by an Operator-style Assistant.
Assess ONLY the Assistant's last response relative to the success criteria and available evidence (including tool outputs).
Return feedback and decide whether the success criteria was met, and whether user input is needed."""


def build_evaluator_user_message(
    *,
    conversation_text: str,
    success_criteria: str,
    last_assistant_response: str,
    prior_feedback: str | None = None,
) -> str:
    """User-side content for the evaluator: conversation, criteria, and last answer."""
    msg = f"""You are evaluating a conversation between the User and an Operator-style Assistant. Judge if the last response truly completes the task.

The entire conversation with the assistant, with the user's original request and all replies, is:
{conversation_text}

The success criteria for this assignment is:
{success_criteria}

And the final response from the Assistant that you are evaluating is:
{last_assistant_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

The Assistant has access to tools to write files. If the Assistant says they have written a file, you can assume they have done so.
Give the Assistant the benefit of the doubt when they claim they completed an action, but REJECT if more work is still needed.

Anti-fluff rules (reject these cases):
- The response describes a plan to search but shows no evidence from any tool.
- The response provides a factual definition or external data without any citation or source.
- The response fails the explicit success criteria (format/length/“exactly N bullets”).
- The response asks vague follow-ups instead of a specific, necessary user question.
If you reject, give explicit, actionable feedback (e.g., “Call Search/Wikipedia and include at least one citation”)."""

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
