"""Prompt construction for rubric-based grading (the CRE pattern from ICER 2025).

CRE = Criteria-Referenced Evaluation: the model judges a student's reasoning
against a fixed list of *question-specific* criteria, returning a binary
pass/fail plus a verbatim evidence quote for each — and nothing else. Crucially
the model never computes the final score; that aggregation happens in Python (see
:func:`core.rubric_grader.grade_submission`), because LLMs are unreliable at
arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.rubric_grader import RubricCriterion

__all__ = ["GRADING_SYSTEM_PROMPT", "build_grading_user_prompt"]


GRADING_SYSTEM_PROMPT = (
    "You are an econometrics teaching assistant grading a student's written "
    "reasoning for a market-decision case. Evaluate the reasoning against EACH "
    "criterion independently and objectively.\n\n"
    "Rules:\n"
    "- Judge only the substance of the econometric/economic reasoning, never the "
    "writing style, grammar, length, or language.\n"
    "- For every criterion decide pass = true only if the reasoning clearly and "
    "correctly satisfies it; otherwise pass = false. There is no partial credit.\n"
    "- For every criterion provide a short verbatim quote from the student's text "
    "as 'evidence'. If nothing in the text is relevant, set pass = false and use "
    'an empty string for evidence.\n'
    "- Do NOT compute any total or score. Only return per-criterion judgements."
)


def build_grading_user_prompt(
    student_reasoning: str,
    rubric: Sequence[RubricCriterion],
) -> str:
    """Render the user prompt listing the rubric criteria and the student's text.

    Parameters
    ----------
    student_reasoning:
        The student's submitted justification.
    rubric:
        The criteria to grade against.

    Returns
    -------
    str
        A prompt enumerating each criterion (id + description) followed by the
        student's reasoning.
    """
    lines = ["Criteria to evaluate:"]
    for criterion in rubric:
        lines.append(f"- id={criterion.id!r}: {criterion.description}")
    lines.append("")
    lines.append("Student reasoning:")
    lines.append('"""')
    lines.append(student_reasoning.strip())
    lines.append('"""')
    return "\n".join(lines)
