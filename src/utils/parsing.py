"""Parse model outputs into structured types."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.state import EvaluatorOutput


class EvaluatorParseError(ValueError):
    """Raised when the evaluator response cannot be parsed into :class:`EvaluatorOutput`."""


def parse_evaluator_output(raw: str) -> EvaluatorOutput:
    """Parse evaluator LLM text into :class:`EvaluatorOutput`.

    Accepts raw JSON or a JSON object inside a markdown code fence.
    """
    text = raw.strip()
    if not text:
        raise EvaluatorParseError("Empty evaluator response")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluatorParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise EvaluatorParseError("Evaluator JSON must be an object")

    try:
        return EvaluatorOutput.model_validate(data)
    except ValidationError as exc:
        raise EvaluatorParseError(str(exc)) from exc
