"""Устная микрозащита (core/defence.py): жеребьёвка и банк вопросов."""

from __future__ import annotations

import pytest

from core.cases import supported_methods
from core.defence import QUESTION_BANK, draw_defence
from core.trap import naive_slope_ratio
from db.enums import Method, Role


def test_bank_covers_every_supported_case() -> None:
    assert set(QUESTION_BANK) == set(supported_methods())


@pytest.mark.parametrize("method", list(QUESTION_BANK))
def test_every_question_has_text_and_expected_answer(method: Method) -> None:
    for q in QUESTION_BANK[method]:
        assert q.text.strip() and q.expected.strip()
        assert q.text.rstrip()[-1] in "?."  # вопрос или задание в повелительном


@pytest.mark.parametrize("method", list(QUESTION_BANK))
def test_bank_has_method_specific_questions_beyond_common(method: Method) -> None:
    """Общих вопросов три; у каждого метода должны быть свои — про его ловушку."""
    assert len(QUESTION_BANK[method]) >= 5


def test_biased_methods_ask_about_the_trap_direction() -> None:
    for method, questions in QUESTION_BANK.items():
        if naive_slope_ratio(method) is None:
            continue
        texts = " ".join(q.text.lower() + q.expected.lower() for q in questions)
        assert "занижен" in texts or "в какую сторону" in texts, method


def test_draw_is_deterministic_for_round_and_attempt() -> None:
    teams = ["3", "1", "2"]
    a = draw_defence(7, Method.OLS_MULTIPLE, teams)
    b = draw_defence(7, Method.OLS_MULTIPLE, list(reversed(teams)))
    assert a == b
    assert a.team_id in teams
    assert isinstance(a.role, Role)


def test_second_attempt_is_reproducible_and_may_differ() -> None:
    teams = [str(i) for i in range(20)]
    first = draw_defence(7, Method.OLS_SIMPLE, teams)
    second = draw_defence(7, Method.OLS_SIMPLE, teams, attempt=1)
    assert second == draw_defence(7, Method.OLS_SIMPLE, teams, attempt=1)
    # На двадцати командах перетяжка практически всегда даёт другую — но
    # гарантировать это нельзя, поэтому проверяем только воспроизводимость.
    assert first == draw_defence(7, Method.OLS_SIMPLE, teams)


def test_draw_reaches_every_team_and_role_over_rounds() -> None:
    """Розыгрыш не застревает: за много раундов выпадают все команды и роли."""
    teams = ["a", "b", "c", "d"]
    seen_teams = {draw_defence(r, Method.OLS_SIMPLE, teams).team_id for r in range(60)}
    seen_roles = {draw_defence(r, Method.OLS_SIMPLE, teams).role for r in range(60)}
    assert seen_teams == set(teams)
    assert seen_roles == set(Role)


def test_draw_without_teams_raises() -> None:
    with pytest.raises(ValueError, match="некого"):
        draw_defence(1, Method.OLS_SIMPLE, [])


def test_draw_without_bank_raises() -> None:
    with pytest.raises(ValueError, match="autocorrelation"):
        draw_defence(1, Method.AUTOCORRELATION, ["a"])
