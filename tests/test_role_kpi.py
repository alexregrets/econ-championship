"""Тесты личных KPI ролей (core/role_kpi.py).

Ключевые гарантии:
- КОНФЛИКТ: рост собственного объёма поднимает KPI маркетолога и опускает KPI
  финансиста — ради этого механика и вводилась (GAME_DESIGN.md №1);
- нормировка идёт внутри роли и относительно лучшего в раунде;
- убыточная маржа и «все в минусе» не дают положительных оценок;
- не поданный прогноз цены обнуляет KPI аналитика и помечается has_input=False
  (это не то же самое, что плохой прогноз);
- итог = 70% команда + 30% KPI, веса валидируются.
"""

from __future__ import annotations

import pytest

from core.market_engine import MarketParameters, compute_cournot_round
from core.role_kpi import (
    DEFAULT_WEIGHTS,
    ScoreWeights,
    compute_role_kpis,
    forecast_accuracy,
    implied_quantity_for_price,
    market_share,
    profit_margin,
)
from db.enums import Role


@pytest.fixture
def params() -> MarketParameters:
    return MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


# --------------------------------------------------------------------------- #
# Конфликт интересов — главное свойство механики
# --------------------------------------------------------------------------- #


def test_more_output_helps_marketer_and_hurts_financier(
    params: MarketParameters,
) -> None:
    """Одна команда наращивает Q: доля растёт, маржа падает."""
    modest = compute_cournot_round({"1": 20.0, "2": 25.0, "3": 25.0}, params)
    greedy = compute_cournot_round({"1": 40.0, "2": 25.0, "3": 25.0}, params)

    modest_kpi = compute_role_kpis(modest)["1"]
    greedy_kpi = compute_role_kpis(greedy)["1"]

    assert greedy_kpi[Role.MARKETER].raw > modest_kpi[Role.MARKETER].raw
    assert greedy_kpi[Role.FINANCIER].raw < modest_kpi[Role.FINANCIER].raw


def test_marketer_and_financier_rank_teams_oppositely(
    params: MarketParameters,
) -> None:
    """В одном раунде крупнейшая команда — лучший маркетолог и худший финансист.

    Маржа ``(P − c)/P`` у всех команд симметричного раунда одинакова, поэтому
    сравнение идёт по сырой доле: конфликт виден именно как противоположный
    порядок предпочтений по объёму.
    """
    results = compute_role_kpis(
        compute_cournot_round({"big": 40.0, "small": 10.0}, params)
    )

    assert results["big"][Role.MARKETER].raw > results["small"][Role.MARKETER].raw
    assert results["big"][Role.MARKETER].normalized == pytest.approx(1.0)
    assert results["small"][Role.MARKETER].normalized == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Отдельные показатели
# --------------------------------------------------------------------------- #


def test_market_share_is_own_over_total() -> None:
    assert market_share(25.0, 100.0) == pytest.approx(0.25)


def test_empty_market_gives_zero_share() -> None:
    assert market_share(0.0, 0.0) == 0.0


def test_margin_is_profit_over_revenue(params: MarketParameters) -> None:
    results = compute_cournot_round({"1": 30.0}, params)
    result = results["1"]

    assert profit_margin(result) == pytest.approx(
        (result.price - params.marginal_cost) / result.price
    )


def test_zero_revenue_gives_zero_margin(params: MarketParameters) -> None:
    """Команда ничего не произвела: маржа не определена, а не бесконечна."""
    results = compute_cournot_round({"1": 0.0, "2": 30.0}, params)

    assert profit_margin(results["1"]) == 0.0


def test_exact_forecast_scores_one() -> None:
    assert forecast_accuracy(50.0, 50.0) == pytest.approx(1.0)


def test_forecast_off_by_ten_percent_scores_ninety() -> None:
    assert forecast_accuracy(55.0, 50.0) == pytest.approx(0.9)


def test_wildly_wrong_forecast_floors_at_zero() -> None:
    assert forecast_accuracy(500.0, 50.0) == 0.0


def test_missing_forecast_scores_zero() -> None:
    assert forecast_accuracy(None, 50.0) == 0.0


def test_implied_quantity_reverses_the_demand_curve() -> None:
    """Объём под прогноз цены: подставив его обратно, получаем прогноз."""
    q = implied_quantity_for_price(a=100.0, b=1.0, target_price=40.0, others_total=30.0)

    assert q == pytest.approx(30.0)
    assert 100.0 - 1.0 * (q + 30.0) == pytest.approx(40.0)


def test_unreachable_price_gives_zero_quantity() -> None:
    """Цену выше уже сложившейся не поднять даже нулевым выпуском."""
    q = implied_quantity_for_price(a=100.0, b=1.0, target_price=95.0, others_total=30.0)

    assert q == 0.0


# --------------------------------------------------------------------------- #
# Нормировка и итог
# --------------------------------------------------------------------------- #


def test_normalization_is_within_role(params: MarketParameters) -> None:
    """Лучший в каждой роли получает 1.0 — маркетолога сравнивают с маркетологом."""
    scores = compute_role_kpis(
        compute_cournot_round({"1": 40.0, "2": 10.0}, params),
        {"1": 10.0, "2": 50.0},
    )

    for role in (Role.MARKETER, Role.SALES_ANALYST):
        best = max(s[role].normalized for s in scores.values())
        assert best == pytest.approx(1.0)


def test_total_is_seventy_thirty(params: MarketParameters) -> None:
    scores = compute_role_kpis(
        compute_cournot_round({"1": 40.0, "2": 10.0}, params), {"1": 50.0}
    )
    kpi = scores["1"][Role.MARKETER]

    assert kpi.total == pytest.approx(
        DEFAULT_WEIGHTS.team * kpi.team_component
        + DEFAULT_WEIGHTS.kpi * kpi.normalized
    )


def test_custom_weights_shift_the_total(params: MarketParameters) -> None:
    results = compute_cournot_round({"1": 40.0, "2": 10.0}, params)
    kpi_heavy = compute_role_kpis(results, weights=ScoreWeights(team=0.4, kpi=0.6))
    kpi = kpi_heavy["2"][Role.MARKETER]

    assert kpi.total == pytest.approx(
        0.4 * kpi.team_component + 0.6 * kpi.normalized
    )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="сумме 1.0"):
        ScoreWeights(team=0.5, kpi=0.3)


def test_negative_weights_rejected() -> None:
    with pytest.raises(ValueError, match="неотрицательны"):
        ScoreWeights(team=1.5, kpi=-0.5)


def test_all_teams_losing_money_get_zero_team_component() -> None:
    """Раунд, где все в минусе: «лучший из убыточных» не получает единицу."""
    params = MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)
    results = compute_cournot_round({"1": 95.0, "2": 95.0}, params)
    assert all(r.profit < 0 for r in results.values())

    scores = compute_role_kpis(results)

    assert all(s[Role.MARKETER].team_component == 0.0 for s in scores.values())


def test_missing_forecast_is_flagged(params: MarketParameters) -> None:
    """has_input отличает «не подал прогноз» от «промахнулся»."""
    scores = compute_role_kpis(
        compute_cournot_round({"1": 20.0, "2": 20.0}, params), {"1": 60.0}
    )

    assert scores["1"][Role.SALES_ANALYST].has_input is True
    assert scores["2"][Role.SALES_ANALYST].has_input is False
    assert scores["2"][Role.SALES_ANALYST].raw == 0.0


def test_other_roles_are_always_marked_as_having_input(
    params: MarketParameters,
) -> None:
    """Флаг про прогноз — только у аналитика; остальным он всегда True."""
    scores = compute_role_kpis(compute_cournot_round({"1": 20.0}, params))

    assert scores["1"][Role.MARKETER].has_input is True
    assert scores["1"][Role.FINANCIER].has_input is True


def test_every_team_gets_all_three_roles(params: MarketParameters) -> None:
    scores = compute_role_kpis(
        compute_cournot_round({"1": 20.0, "2": 20.0, "3": 20.0}, params)
    )

    assert set(scores) == {"1", "2", "3"}
    for team_scores in scores.values():
        assert set(team_scores) == {
            Role.MARKETER,
            Role.FINANCIER,
            Role.SALES_ANALYST,
        }


def test_empty_results_rejected() -> None:
    with pytest.raises(ValueError, match="at least one team"):
        compute_role_kpis({})
