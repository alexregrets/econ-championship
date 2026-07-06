"""Question-specific rubric grading (the CRE pattern from ICER 2025).

The LLM returns only a binary pass/fail plus an evidence quote per criterion;
the final numeric score is aggregated here in Python, never inside the model.
This keeps grading auditable and immune to LLM arithmetic mistakes.

The grader depends on the :class:`~llm.base.StructuredLLM` protocol, so it can be
unit-tested with a fake client and no network access.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from llm.base import StructuredLLM
from llm.prompts.grading import GRADING_SYSTEM_PROMPT, build_grading_user_prompt

__all__ = [
    "RubricCriterion",
    "RubricGrade",
    "GradingResult",
    "LLMGradingResponse",
    "grade_submission",
]


class RubricCriterion(BaseModel):
    """A single, question-specific grading criterion.

    Parameters
    ----------
    id:
        Stable machine identifier, e.g. ``"specification"`` or ``"vif_check"``.
    description:
        Human-readable statement of what must be present to pass.
    weight:
        Relative importance in ``[0, 1]``. Scores are normalized by the sum of
        weights, so the weights need not add up to exactly 1.
    binary:
        Whether the criterion is all-or-nothing (the only mode currently
        supported; kept explicit to mirror the ICER rubric design).
    """

    id: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    binary: bool = True


class RubricGrade(BaseModel):
    """The LLM's judgement for one criterion."""

    criterion_id: str
    passed: bool
    evidence: str = Field(
        default="",
        description="Verbatim quote from the student's answer supporting the call.",
    )


class LLMGradingResponse(BaseModel):
    """Raw structured payload returned by the model: one grade per criterion."""

    grades: list[RubricGrade]


class GradingResult(BaseModel):
    """Final, audited grading outcome.

    ``total_score`` is computed deterministically in Python (not by the LLM) as
    the weight-fraction of passed criteria, so it always lies in ``[0, 1]``.
    """

    grades: list[RubricGrade]
    total_score: float
    raw_llm_response: str


def _aggregate_score(
    rubric: Sequence[RubricCriterion],
    grades_by_id: dict[str, RubricGrade],
) -> float:
    """Compute the normalized score from passed criteria weights.

    Returns the sum of weights of passed criteria divided by the total weight of
    all criteria. Criteria the model failed to grade count as not passed. Returns
    ``0.0`` if the rubric has no positive total weight.
    """
    total_weight = sum(c.weight for c in rubric)
    if total_weight <= 0.0:
        return 0.0
    earned = sum(
        c.weight
        for c in rubric
        if (g := grades_by_id.get(c.id)) is not None and g.passed
    )
    return earned / total_weight


async def grade_submission(
    student_reasoning: str,
    rubric: Sequence[RubricCriterion],
    llm: StructuredLLM,
) -> GradingResult:
    """Grade a student's reasoning against a question-specific rubric.

    Parameters
    ----------
    student_reasoning:
        The student's written justification for their market decision.
    rubric:
        The criteria to evaluate against.
    llm:
        Any :class:`~llm.base.StructuredLLM` implementation.

    Returns
    -------
    GradingResult
        Per-criterion grades, a deterministically computed ``total_score`` in
        ``[0, 1]``, and the raw model payload for audit.

    Raises
    ------
    ValueError
        If ``rubric`` is empty.

    Notes
    -----
    The model is only trusted to make binary per-criterion calls; the score is
    aggregated here. Any criterion the model omits is treated as failed, and any
    extra grades it returns for unknown criterion ids are ignored.
    """
    if not rubric:
        raise ValueError("rubric must contain at least one criterion")

    user_prompt = build_grading_user_prompt(student_reasoning, rubric)
    response = await llm.structured_completion(
        GRADING_SYSTEM_PROMPT,
        user_prompt,
        LLMGradingResponse,
        temperature=0.1,
    )

    grades_by_id = {g.criterion_id: g for g in response.grades}
    # Normalize to exactly one grade per rubric criterion, in rubric order.
    normalized: list[RubricGrade] = [
        grades_by_id.get(
            c.id, RubricGrade(criterion_id=c.id, passed=False, evidence="")
        )
        for c in rubric
    ]
    total_score = _aggregate_score(rubric, grades_by_id)

    return GradingResult(
        grades=normalized,
        total_score=total_score,
        raw_llm_response=response.model_dump_json(),
    )
