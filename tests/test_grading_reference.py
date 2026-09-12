"""Справка грейдеру с истиной раунда: попадает в промпт, не к студенту."""

from __future__ import annotations

from core.rubric_grader import RubricCriterion
from dashboard.actions import grading_reference_notes
from db.enums import Method, RoundStatus
from db.models import Round
from llm.prompts.grading import build_grading_user_prompt


def _round(method: Method) -> Round:
    return Round(
        number=1,
        method=method,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        status=RoundStatus.CLOSED,
    )


def test_notes_state_truth_and_nash_for_market_size() -> None:
    notes = grading_reference_notes(_round(Method.OLS_SIMPLE), n_firms=3)
    assert "b = 1" in notes and "c = 10" in notes
    # Нэш при трёх фирмах: q = 90/4 = 22.5, цена = 100 - 67.5 = 32.5.
    assert "22.50" in notes and "32.50" in notes
    assert "trap" not in notes


def test_notes_name_naive_slope_where_trap_exists() -> None:
    notes = grading_reference_notes(_round(Method.OLS_MULTIPLE), n_firms=3)
    assert "0.35" in notes and "trap" in notes


def test_prompt_includes_reference_block_only_when_given() -> None:
    rubric = [RubricCriterion(id="x", description="d", weight=1.0)]
    plain = build_grading_user_prompt("text", rubric)
    assert "Reference facts" not in plain
    with_ref = build_grading_user_prompt("text", rubric, "- b = 1")
    assert "Reference facts" in with_ref and "- b = 1" in with_ref
    # Текст студента идёт после справки — модель читает факты до ответа.
    assert with_ref.index("Reference facts") < with_ref.index("Student reasoning")
