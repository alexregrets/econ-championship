"""Тесты обратного хода — восстановления убеждений команды по сданному объёму.

Инварианты, на которые опирается панель разбора у преподавателя:

- команда, сыгравшая истинное равновесие Нэша, «верит» в истинный наклон,
  и оба разрыва — по цене и по прибыли — равны нулю;
- перепроизводство читается как вера в более пологий спрос и завышенную
  ожидаемую цену, недопроизводство — наоборот;
- разрыв в прибыли против лучшего ответа неотрицателен всегда;
- фактическая цена и прибыль совпадают с тем, что выдаёт движок раунда.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.beliefs import recover_beliefs
from core.market_engine import MarketParameters, compute_cournot_round, nash_equilibrium


@pytest.fixture
def params() -> MarketParameters:
    """Стандартный рынок: P = 100 - 1*Q, издержки 10."""
    return MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


def _nash_round(params: MarketParameters, n: int) -> dict[str, float]:
    """Все ``n`` команд играют симметричное равновесие Нэша."""
    q = nash_equilibrium(n, params)
    return {f"team{i}": q for i in range(n)}


# --------------------------------------------------------------------------- #
# Базовый инвариант: истинный Нэш выдаёт истинные убеждения
# --------------------------------------------------------------------------- #


def test_true_nash_play_reveals_true_slope(params: MarketParameters) -> None:
    beliefs = recover_beliefs(_nash_round(params, 4), params)

    assert len(beliefs) == 4
    for belief in beliefs.values():
        assert belief.implied_slope == pytest.approx(params.b)
        assert belief.price_gap == pytest.approx(0.0, abs=1e-9)
        assert belief.profit_gap == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 5, 6])
def test_true_nash_play_for_any_field_size(params: MarketParameters, n: int) -> None:
    for belief in recover_beliefs(_nash_round(params, n), params).values():
        assert belief.implied_slope == pytest.approx(params.b)
        assert belief.best_response_quantity == pytest.approx(belief.submitted_quantity)


# --------------------------------------------------------------------------- #
# Направление ошибки
# --------------------------------------------------------------------------- #


def test_overproducer_believed_flatter_demand(params: MarketParameters) -> None:
    q = nash_equilibrium(4, params)
    decisions = {"a": 1.3 * q, "b": q, "c": q, "d": q}
    beliefs = recover_beliefs(decisions, params)

    deviator = beliefs["a"]
    assert deviator.implied_slope < params.b  # спрос показался положе
    assert deviator.price_gap > 0  # ждал цену выше рынка
    assert deviator.profit_gap > 0  # недобрал прибыли

    # Соперники остались на старом q*, пока поле сдвинулось: не срезав объём
    # под возросший выпуск «a», они тоже выглядят поверившими в пологий спрос,
    # но слабее — самый заниженный наклон у самого перепроизводителя.
    for other in ("b", "c", "d"):
        assert beliefs[other].implied_slope < params.b
        assert deviator.implied_slope < beliefs[other].implied_slope


def test_underproducer_believed_steeper_demand(params: MarketParameters) -> None:
    q = nash_equilibrium(4, params)
    decisions = {"a": 0.7 * q, "b": q, "c": q, "d": q}

    deviator = recover_beliefs(decisions, params)["a"]
    assert deviator.implied_slope > params.b
    assert deviator.price_gap < 0  # ждал цену ниже той, что дал рынок


@pytest.mark.parametrize("factor", [0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0])
def test_profit_gap_never_negative(params: MarketParameters, factor: float) -> None:
    q = nash_equilibrium(4, params)
    decisions = {"a": factor * q, "b": q, "c": q, "d": q}
    for belief in recover_beliefs(decisions, params).values():
        assert belief.profit_gap >= -1e-9


# --------------------------------------------------------------------------- #
# Внутренняя согласованность величин
# --------------------------------------------------------------------------- #


def test_best_response_satisfies_first_order_condition(params: MarketParameters) -> None:
    q = nash_equilibrium(4, params)
    decisions = {"a": 1.4 * q, "b": q, "c": 0.9 * q, "d": q}
    for belief in recover_beliefs(decisions, params).values():
        if belief.best_response_quantity == 0.0:
            continue
        foc = (
            belief.residual_intercept
            - params.marginal_cost
            - 2 * params.b * belief.best_response_quantity
        )
        assert foc == pytest.approx(0.0, abs=1e-9)


def test_expected_price_uses_true_slope_formula(params: MarketParameters) -> None:
    # CASES.md: P_ожид = c + b * q
    decisions = {"a": 18.0, "b": 22.0, "c": 26.0}
    for belief in recover_beliefs(decisions, params).values():
        assert belief.expected_price == pytest.approx(
            params.marginal_cost + params.b * belief.submitted_quantity
        )


def test_actual_price_and_profit_match_engine(params: MarketParameters) -> None:
    decisions = {"a": 20.0, "b": 25.0, "c": 30.0}
    engine = compute_cournot_round(decisions, params)
    for team_id, belief in recover_beliefs(decisions, params).items():
        assert belief.actual_price == pytest.approx(engine[team_id].price)
        assert belief.profit == pytest.approx(engine[team_id].profit)


def test_residual_intercept_nets_out_only_rivals(params: MarketParameters) -> None:
    decisions = {"a": 20.0, "b": 25.0, "c": 30.0}
    beliefs = recover_beliefs(decisions, params)
    assert beliefs["a"].rivals_quantity == pytest.approx(55.0)
    assert beliefs["a"].residual_intercept == pytest.approx(100.0 - 1.0 * 55.0)


# --------------------------------------------------------------------------- #
# Асимметричные издержки
# --------------------------------------------------------------------------- #


def test_lower_cost_team_has_larger_best_response(params: MarketParameters) -> None:
    decisions = {"lo": 25.0, "hi": 25.0}
    costs = {"lo": 5.0, "hi": 20.0}
    beliefs = recover_beliefs(decisions, params, marginal_costs=costs)

    assert beliefs["lo"].best_response_quantity > beliefs["hi"].best_response_quantity
    assert beliefs["lo"].expected_price < beliefs["hi"].expected_price


def test_profit_uses_each_teams_own_cost(params: MarketParameters) -> None:
    """Прибыль в разборе считается по c_i команды, а не по общей издержке
    рынка: иначе в асимметричном раунде разрыв в прибыли врёт."""
    decisions = {"lo": 25.0, "hi": 25.0}
    costs = {"lo": 5.0, "hi": 20.0}
    beliefs = recover_beliefs(decisions, params, marginal_costs=costs)

    price = params.a - params.b * 50.0
    assert beliefs["lo"].profit == pytest.approx((price - 5.0) * 25.0)
    assert beliefs["hi"].profit == pytest.approx((price - 20.0) * 25.0)
    assert beliefs["lo"].profit > beliefs["hi"].profit
    # Лучший ответ считается на тех же издержках — разрыв не отрицателен.
    assert beliefs["lo"].profit_gap >= -1e-9
    assert beliefs["hi"].profit_gap >= -1e-9


def test_symmetric_default_matches_explicit_equal_costs(params: MarketParameters) -> None:
    decisions = {"a": 20.0, "b": 24.0, "c": 28.0}
    default = recover_beliefs(decisions, params)
    explicit = recover_beliefs(
        decisions, params, marginal_costs=dict.fromkeys(decisions, params.marginal_cost)
    )
    assert default == explicit


# --------------------------------------------------------------------------- #
# Клампы и валидация
# --------------------------------------------------------------------------- #


def test_best_response_clamped_when_rivals_flood_the_market(
    params: MarketParameters,
) -> None:
    # Крупный игрок затопил рынок: остаточный спрос мелкого уже ниже издержек.
    decisions = {"big": 95.0, "small": 2.0}
    small = recover_beliefs(decisions, params)["small"]

    assert small.residual_intercept < params.marginal_cost
    assert small.best_response_quantity == 0.0
    assert small.best_response_profit == 0.0
    assert small.profit_gap >= -1e-9  # не производить лучше, чем производить в убыток


def test_zero_quantity_raises(params: MarketParameters) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        recover_beliefs({"a": 20.0, "b": 0.0}, params)


def test_negative_quantity_raises(params: MarketParameters) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        recover_beliefs({"a": 20.0, "b": -3.0}, params)


def test_non_finite_quantity_raises(params: MarketParameters) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        recover_beliefs({"a": 20.0, "b": float("inf")}, params)


def test_empty_decisions_raise(params: MarketParameters) -> None:
    with pytest.raises(ValueError):
        recover_beliefs({}, params)


def test_incomplete_costs_raise(params: MarketParameters) -> None:
    with pytest.raises(ValueError, match="не для всех"):
        recover_beliefs({"a": 20.0, "b": 21.0}, params, marginal_costs={"a": 10.0})


# --------------------------------------------------------------------------- #
# Чистота функции
# --------------------------------------------------------------------------- #


def test_result_is_frozen(params: MarketParameters) -> None:
    belief = next(iter(recover_beliefs({"a": 20.0, "b": 21.0}, params).values()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        belief.implied_slope = 1.0  # type: ignore[misc]


def test_deterministic(params: MarketParameters) -> None:
    decisions = {"a": 19.0, "b": 23.0, "c": 27.0, "d": 21.0}
    assert recover_beliefs(decisions, params) == recover_beliefs(decisions, params)
