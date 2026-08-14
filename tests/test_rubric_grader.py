"""Tests for rubric grading, using a fake LLM (no network).

Verifies the key CRE guarantees: the score is computed in Python from weights,
omitted criteria count as failed, unknown criterion ids are ignored, and grades
are returned in rubric order.
"""

from __future__ import annotations

from pydantic import BaseModel

from core.rubric_grader import (
    GradingResult,
    LLMGradingResponse,
    RubricCriterion,
    RubricGrade,
    grade_submission,
)


class FakeLLM:
    """A StructuredLLM that returns a preset response and records its calls."""

    def __init__(self, response: LLMGradingResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> BaseModel:
        self.calls.append((system_prompt, user_prompt))
        return self._response


def _rubric() -> list[RubricCriterion]:
    return [
        RubricCriterion(id="spec", description="States the model spec", weight=0.5),
        RubricCriterion(id="vif", description="Checks multicollinearity", weight=0.3),
        RubricCriterion(id="mech", description="Explains mechanism", weight=0.2),
    ]


async def test_all_passed_scores_one() -> None:
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[
                RubricGrade(criterion_id="spec", passed=True, evidence="..."),
                RubricGrade(criterion_id="vif", passed=True, evidence="..."),
                RubricGrade(criterion_id="mech", passed=True, evidence="..."),
            ]
        )
    )
    result = await grade_submission("my reasoning", _rubric(), llm)
    assert isinstance(result, GradingResult)
    assert result.total_score == 1.0
    assert all(g.passed for g in result.grades)


async def test_partial_pass_uses_weights() -> None:
    # Pass spec (0.5) and mech (0.2), fail vif (0.3) -> 0.7 of total weight 1.0
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[
                RubricGrade(criterion_id="spec", passed=True),
                RubricGrade(criterion_id="vif", passed=False),
                RubricGrade(criterion_id="mech", passed=True),
            ]
        )
    )
    result = await grade_submission("text", _rubric(), llm)
    assert result.total_score == 0.7


async def test_weights_need_not_sum_to_one() -> None:
    rubric = [
        RubricCriterion(id="a", description="x", weight=0.2),
        RubricCriterion(id="b", description="y", weight=0.2),
    ]
    # Pass only "a": earned 0.2 / total 0.4 = 0.5
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[
                RubricGrade(criterion_id="a", passed=True),
                RubricGrade(criterion_id="b", passed=False),
            ]
        )
    )
    result = await grade_submission("text", rubric, llm)
    assert result.total_score == 0.5


async def test_omitted_criterion_counts_as_failed() -> None:
    # LLM only returns a grade for "spec"; vif and mech are missing.
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[RubricGrade(criterion_id="spec", passed=True)]
        )
    )
    result = await grade_submission("text", _rubric(), llm)
    assert result.total_score == 0.5  # only spec's 0.5 of total 1.0
    assert len(result.grades) == 3  # all criteria represented
    assert {g.criterion_id for g in result.grades} == {"spec", "vif", "mech"}
    assert result.grades[1].passed is False  # vif filled in as failed


async def test_unknown_criterion_ids_are_ignored() -> None:
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[
                RubricGrade(criterion_id="spec", passed=True),
                RubricGrade(criterion_id="ghost", passed=True),  # not in rubric
            ]
        )
    )
    result = await grade_submission("text", _rubric(), llm)
    assert result.total_score == 0.5
    assert {g.criterion_id for g in result.grades} == {"spec", "vif", "mech"}


async def test_grades_returned_in_rubric_order() -> None:
    # Provide grades out of order; result should follow rubric order.
    llm = FakeLLM(
        LLMGradingResponse(
            grades=[
                RubricGrade(criterion_id="mech", passed=True),
                RubricGrade(criterion_id="spec", passed=True),
                RubricGrade(criterion_id="vif", passed=True),
            ]
        )
    )
    result = await grade_submission("text", _rubric(), llm)
    assert [g.criterion_id for g in result.grades] == ["spec", "vif", "mech"]


async def test_empty_rubric_raises() -> None:
    llm = FakeLLM(LLMGradingResponse(grades=[]))
    try:
        await grade_submission("text", [], llm)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty rubric")


async def test_raw_response_is_preserved() -> None:
    llm = FakeLLM(
        LLMGradingResponse(grades=[RubricGrade(criterion_id="spec", passed=True)])
    )
    result = await grade_submission("text", _rubric(), llm)
    assert "spec" in result.raw_llm_response
