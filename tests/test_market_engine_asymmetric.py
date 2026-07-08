"""Тесты асимметричного движка Курно и обратной калибровки издержек.

Ключевые гарантии:
- REGRESSION: при одинаковых c_i у всех фирм результат совпадает с существующим
  симметричным движком (та же a, b, mc — generic-рынок и «Нефть РФ 2013»);
- round-trip: implied_marginal_costs + asymmetric_nash_equilibrium возвращают
  исходную наблюдаемую точку;
- калиброванные издержки «Нефть РФ 2013» воспроизводят реальные доли компаний
  (допуск 1% из постановки задачи, фактическая ошибка — float-шум), средняя
  равна полной себестоимости $50/барр;
- валидация отвергает мусор: отрицательные/бесконечные издержки, издержки
  выше точки насыщения, несовпадение наборов команд, неактивные фирмы.
"""

from __future__ import annotations

import math

import pytest

from core.market_engine import (
    MarketParameters,
    compute_cournot_round,
    monopoly_quantity,
    nash_equilibrium,
)
from core.market_engine_asymmetric import (
    asymmetric_nash_equilibrium,
    compute_asymmetric_cournot_round,
    implied_marginal_costs,
)
from devshell.role_seed import (
    FULL_COST_2013_USD_PER_TON,
    OIL_PRODUCTION_2013_MLN_T,
    TOTAL_PRODUCTION_2013_MLN_T,
    URALS_PRICE_2013_USD_PER_TON,
    implied_oil_2013_costs,
    oil_2013_market_parameters,
)


@pytest.fixture
def params() -> MarketParameters:
    """Тот же generic-рынок, что в tests/test_market_engine.py."""
    return MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


# --------------------------------------------------------------------------- #
# REGRESSION: равные издержки == существующий симметричный движок
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_firms", [3, 7])
def test_equal_costs_reproduce_symmetric_nash(
    params: MarketParameters, n_firms: int
) -> None:
    """При c_i = const результат обязан совпасть с симметричной формулой."""
    symmetric_q = nash_equilibrium(n_firms, params)

    result = asymmetric_nash_equilibrium(
        params.a, params.b, [params.marginal_cost] * n_firms
    )

    for quantity in result.quantities:
        assert quantity == pytest.approx(symmetric_q, rel=1e-12)
    assert result.total_quantity == pytest.approx(n_firms * symmetric_q, rel=1e-12)
    # Цена согласована со спросом: P* = a - b*Q*.
    assert result.price == pytest.approx(
        params.a - params.b * result.total_quantity, rel=1e-12
    )


def test_equal_costs_three_firms_numeric(params: MarketParameters) -> None:
    # q* = (100 - 10) / (4 * 1) = 22.5; P* = 100 - 3*22.5 = 32.5
    result = asymmetric_nash_equilibrium(100.0, 1.0, [10.0, 10.0, 10.0])
    assert result.quantities == pytest.approx((22.5, 22.5, 22.5))
    assert result.price == pytest.approx(32.5)


def test_equal_costs_match_oil_2013_symmetric_calibration() -> None:
    """На калиброванном нефтяном рынке равные издержки дают симметричный Нэш 2013."""
    params = oil_2013_market_parameters()
    result = asymmetric_nash_equilibrium(
        params.a, params.b, [params.marginal_cost] * 3
    )
    # Симметричная калибровка построена так, что Нэш воспроизводит суммарную
    # добычу 2013 и цену Urals — асимметричный движок обязан дать то же самое.
    assert result.total_quantity == pytest.approx(
        TOTAL_PRODUCTION_2013_MLN_T, rel=1e-9
    )
    assert result.price == pytest.approx(URALS_PRICE_2013_USD_PER_TON, rel=1e-9)
    for quantity in result.quantities:
        assert quantity == pytest.approx(nash_equilibrium(3, params), rel=1e-12)


def test_single_firm_equals_monopoly(params: MarketParameters) -> None:
    result = asymmetric_nash_equilibrium(params.a, params.b, [params.marginal_cost])
    assert result.quantities[0] == pytest.approx(monopoly_quantity(params))


# --------------------------------------------------------------------------- #
# Экономика асимметрии
# --------------------------------------------------------------------------- #


def test_lower_cost_firm_produces_more() -> None:
    result = asymmetric_nash_equilibrium(100.0, 1.0, [5.0, 10.0, 20.0])
    q1, q2, q3 = result.quantities
    assert q1 > q2 > q3


def test_round_trip_implied_costs_exact_integers() -> None:
    """Наблюдаемая точка с целыми числами: c_i = P - b*q_i, прямой счёт её возвращает.

    a=100, b=1, q=(30, 20, 10): P = 100 - 60 = 40, c = (10, 20, 30);
    проверка вперёд: Σc=60, Q* = (300-60)/4 = 60, P* = (100+60)/4 = 40,
    q_i* = P* - c_i = (30, 20, 10).
    """
    costs = implied_marginal_costs(100.0, 1.0, [30.0, 20.0, 10.0])
    assert costs == pytest.approx((10.0, 20.0, 30.0))

    result = asymmetric_nash_equilibrium(100.0, 1.0, costs)
    assert result.quantities == pytest.approx((30.0, 20.0, 10.0))
    assert result.price == pytest.approx(40.0)


# --------------------------------------------------------------------------- #
# Калибровка «Нефть РФ 2013»
# --------------------------------------------------------------------------- #


def test_oil_implied_costs_reproduce_real_2013_shares() -> None:
    """Главный тест задачи: равновесие с implied costs = реальные добычи 2013."""
    params = oil_2013_market_parameters()
    costs = implied_oil_2013_costs()
    companies = list(OIL_PRODUCTION_2013_MLN_T)

    result = asymmetric_nash_equilibrium(
        params.a, params.b, [costs[c] for c in companies]
    )

    for company, quantity in zip(companies, result.quantities, strict=True):
        real = OIL_PRODUCTION_2013_MLN_T[company]
        # Допуск 1% — требование постановки; фактическое совпадение точное.
        assert quantity == pytest.approx(real, rel=0.01)
        assert quantity == pytest.approx(real, rel=1e-9)

    assert result.total_quantity == pytest.approx(
        TOTAL_PRODUCTION_2013_MLN_T, rel=1e-9
    )
    assert result.price == pytest.approx(URALS_PRICE_2013_USD_PER_TON, rel=1e-9)


def test_oil_implied_costs_mean_is_full_cost() -> None:
    """Средняя implied-себестоимость = $50/барр (алгебраическое следствие калибровки)."""
    costs = implied_oil_2013_costs()
    mean = sum(costs.values()) / len(costs)
    assert mean == pytest.approx(FULL_COST_2013_USD_PER_TON, rel=1e-9)


def test_oil_implied_costs_magnitudes() -> None:
    """Пофирменные значения задокументированы числом (модельные, не отчётные)."""
    costs = implied_oil_2013_costs()
    assert costs["Роснефть"] == pytest.approx(55.01, abs=0.01)
    assert costs["Лукойл"] == pytest.approx(472.68, abs=0.01)
    assert costs["Сургутнефтегаз"] == pytest.approx(564.30, abs=0.01)
    # Крупнейший производитель — самая низкая implied-себестоимость.
    assert costs["Роснефть"] < costs["Лукойл"] < costs["Сургутнефтегаз"]


# --------------------------------------------------------------------------- #
# compute_asymmetric_cournot_round
# --------------------------------------------------------------------------- #


def test_uniform_costs_round_matches_symmetric_engine(
    params: MarketParameters,
) -> None:
    """REGRESSION: пораундовый счёт с равными издержками == симметричный движок."""
    decisions = {"a": 5.0, "b": 10.0, "c": 15.0}
    symmetric = compute_cournot_round(decisions, params)
    asymmetric = compute_asymmetric_cournot_round(
        decisions,
        a=params.a,
        b=params.b,
        marginal_costs=dict.fromkeys(decisions, params.marginal_cost),
    )
    assert asymmetric == symmetric


def test_asymmetric_round_costs_differ_only_in_cost_and_profit() -> None:
    decisions = {"cheap": 10.0, "pricey": 10.0}
    results = compute_asymmetric_cournot_round(
        decisions, a=100.0, b=1.0, marginal_costs={"cheap": 5.0, "pricey": 15.0}
    )
    # Q=20 -> P=80; выручка одинаковая, различие только в издержках.
    assert results["cheap"].price == results["pricey"].price == pytest.approx(80.0)
    assert results["cheap"].revenue == pytest.approx(results["pricey"].revenue)
    assert results["cheap"].cost == pytest.approx(50.0)
    assert results["pricey"].cost == pytest.approx(150.0)
    assert results["cheap"].profit - results["pricey"].profit == pytest.approx(100.0)


def test_asymmetric_round_price_floored_at_zero() -> None:
    results = compute_asymmetric_cournot_round(
        {"hog": 1000.0}, a=100.0, b=1.0, marginal_costs={"hog": 10.0}
    )
    assert results["hog"].price == 0.0
    assert results["hog"].profit == pytest.approx(-10.0 * 1000.0)


# --------------------------------------------------------------------------- #
# Валидация некорректного ввода
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("a", "b", "costs"),
    [
        (0.0, 1.0, [10.0]),  # a не положительна
        (-5.0, 1.0, [10.0]),  # a отрицательна
        (100.0, 0.0, [10.0]),  # b не положителен
        (100.0, -1.0, [10.0]),  # b отрицателен
        (100.0, 1.0, []),  # пустой список фирм
        (100.0, 1.0, [10.0, -1.0]),  # отрицательные издержки
        (100.0, 1.0, [10.0, math.inf]),  # бесконечные издержки
        (100.0, 1.0, [10.0, math.nan]),  # NaN
        (100.0, 1.0, [10.0, 100.0]),  # издержки == a
        (100.0, 1.0, [10.0, 150.0]),  # издержки > a
    ],
)
def test_equilibrium_invalid_input_raises(
    a: float, b: float, costs: list[float]
) -> None:
    with pytest.raises(ValueError):
        asymmetric_nash_equilibrium(a, b, costs)


def test_equilibrium_inactive_firm_raises() -> None:
    # Σc=99 -> P*=(100+99)/4=49.75 < 99: третья фирма получила бы q*<0.
    with pytest.raises(ValueError, match="inactive"):
        asymmetric_nash_equilibrium(100.0, 1.0, [0.0, 0.0, 99.0])


@pytest.mark.parametrize(
    ("a", "b", "quantities"),
    [
        (100.0, 1.0, []),  # пусто
        (100.0, 1.0, [10.0, 0.0]),  # нулевой объём — FOC не определяет c_i
        (100.0, 1.0, [10.0, -5.0]),  # отрицательный объём
        (100.0, 1.0, [10.0, math.inf]),  # не конечен
        (100.0, 1.0, [60.0, 50.0]),  # P = 100-110 < 0: вне ветви спроса
        (100.0, 1.0, [50.0, 10.0]),  # c_1 = 40 - 50 < 0: точка не равновесие
        (0.0, 1.0, [10.0]),  # спрос некорректен
    ],
)
def test_implied_costs_invalid_input_raises(
    a: float, b: float, quantities: list[float]
) -> None:
    with pytest.raises(ValueError):
        implied_marginal_costs(a, b, quantities)


def test_round_empty_decisions_raise() -> None:
    with pytest.raises(ValueError):
        compute_asymmetric_cournot_round({}, a=100.0, b=1.0, marginal_costs={})


def test_round_mismatched_teams_raise() -> None:
    """Несовпадение наборов команд в решениях и издержках — ошибка, не дефолт."""
    with pytest.raises(ValueError, match="missing costs"):
        compute_asymmetric_cournot_round(
            {"x": 1.0, "y": 2.0}, a=100.0, b=1.0, marginal_costs={"x": 10.0}
        )
    with pytest.raises(ValueError, match="costs without decisions"):
        compute_asymmetric_cournot_round(
            {"x": 1.0}, a=100.0, b=1.0, marginal_costs={"x": 10.0, "y": 10.0}
        )


def test_round_negative_quantity_raises() -> None:
    with pytest.raises(ValueError):
        compute_asymmetric_cournot_round(
            {"x": -1.0}, a=100.0, b=1.0, marginal_costs={"x": 10.0}
        )


def test_round_negative_cost_raises() -> None:
    with pytest.raises(ValueError):
        compute_asymmetric_cournot_round(
            {"x": 1.0}, a=100.0, b=1.0, marginal_costs={"x": -10.0}
        )
