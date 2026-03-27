"""Typed state models for the Sidekick LangGraph workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


DEFAULT_SUCCESS_CRITERIA = "The answer should be clear and accurate."


class EvaluatorOutput(BaseModel):
    """Structured decision payload returned by the evaluator node."""

    feedback: str = Field(min_length=1, description="Feedback for the worker.")
    success_criteria_met: bool = Field(
        description="True when the worker result satisfies success criteria."
    )
    user_input_needed: bool = Field(
        description="True when the loop should pause for user clarification."
    )


class AgentState(BaseModel):
    """Graph state shared across worker, tools, and evaluator nodes."""

    messages: list[Any] = Field(default_factory=list)
    success_criteria: str = Field(default=DEFAULT_SUCCESS_CRITERIA, min_length=1)
    feedback_on_work: str | None = Field(default=None)
    success_criteria_met: bool = Field(default=False)
    user_input_needed: bool = Field(default=False)
    iteration: int = Field(default=0, ge=0)
    thread_id: str | None = Field(default=None)

    @field_validator("success_criteria")
    @classmethod
    def normalize_success_criteria(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return DEFAULT_SUCCESS_CRITERIA
        return cleaned

    def next_iteration(self, feedback: str | None = None) -> "AgentState":
        """Return a copied state advanced to the next loop iteration."""
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "feedback_on_work": feedback or self.feedback_on_work,
            }
        )

    def apply_evaluator_output(self, result: EvaluatorOutput) -> "AgentState":
        """Return a copied state with evaluator decisions applied."""
        return self.model_copy(
            update={
                "feedback_on_work": result.feedback,
                "success_criteria_met": result.success_criteria_met,
                "user_input_needed": result.user_input_needed,
            }
        )
