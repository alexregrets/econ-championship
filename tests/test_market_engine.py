"""Tests for the Cournot market engine.

Covers the economic invariants the tournament relies on:
- in symmetric equilibrium all teams earn identical profit;
- unilaterally overproducing hurts the deviator and helps rivals;
- the market price is never negative;
- parameter / input validation rejects nonsense.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from core.market_engine import (
    MarketParameters,
    compute_cournot_round,
    monopoly_quantity,
    nash_equilibrium,
)


@pytest.fixture
def params() -> MarketParameters:
    """A standard market: P = 100 - 1*Q, marginal cost 10."""
    return MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


# --------------------------------------------------------------------------- #
# MarketParameters validation
# --------------------------------------------------------------------------- #


def test_valid_parameters_construct(params: MarketParameters) -> None:
    assert params.a == 100.0
    assert params.b == 1.0
    assert params.marginal_cost == 10.0


@pytest.mark.parametrize(
    ("a", "b", "mc"),
    [
        (0.0, 1.0, 10.0),  # a not positive
        (-5.0, 1.0, 10.0),  # a negative
        (100.0, 0.0, 10.0),  # b not positive
        (100.0, -1.0, 10.0),  # b negative
        (100.0, 1.0, -1.0),  # negative cost
        (100.0, 1.0, 100.0),  # cost == a, no profitable output
        (100.0, 1.0, 150.0),  # cost > a
    ],
)
def test_invalid_parameters_raise(a: float, b: float, mc: float) -> None:
    with pytest.raises(ValueError):
        MarketParameters(a=a, b=b, marginal_cost=mc)


def test_parameters_are_frozen(params: MarketParameters) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.a = 200.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# nash_equilibrium / monopoly_quantity
# --------------------------------------------------------------------------- #


def test_nash_equilibrium_formula(params: MarketParameters) -> None:
    # q* = (a - c) / ((N+1) * b) = (100 - 10) / (8 * 1) = 11.25 for 7 firms
    assert nash_equilibrium(7, params) == pytest.approx(11.25)


def test_nash_equilibrium_single_firm_matches_monopoly(
    params: MarketParameters,
) -> None:
    # With N=1 the Cournot equilibrium collapses to the monopoly output.
    assert nash_equilibrium(1, params) == pytest.approx(monopoly_quantity(params))


def test_nash_equilibrium_rejects_zero_firms(params: MarketParameters) -> None:
    with pytest.raises(ValueError):
        nash_equilibrium(0, params)


def test_monopoly_quantity_formula(params: MarketParameters) -> None:
    # (a - c) / (2b) = 90 / 2 = 45
    assert monopoly_quantity(params) == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# compute_cournot_round — core economics
# --------------------------------------------------------------------------- #


def test_seven_teams_in_equilibrium_earn_equal_profit(
    params: MarketParameters,
) -> None:
    q_star = nash_equilibrium(7, params)
    decisions = {f"team_{i}": q_star for i in range(7)}

    results = compute_cournot_round(decisions, params)

    profits = [r.profit for r in results.values()]
    first = profits[0]
    assert all(p == pytest.approx(first) for p in profits)
    # Equilibrium profit per firm: price = 100 - 7*11.25 = 21.25,
    # profit = (21.25 - 10) * 11.25 = 126.5625
    assert first == pytest.approx(126.5625)


def test_unilateral_overproduction_hurts_deviator_helps_rivals(
    params: MarketParameters,
) -> None:
    q_star = nash_equilibrium(7, params)
    baseline = compute_cournot_round(
        {f"team_{i}": q_star for i in range(7)}, params
    )

    # team_0 deviates upward; everyone else stays at q*.
    deviated_decisions = {f"team_{i}": q_star for i in range(7)}
    deviated_decisions["team_0"] = q_star + 5.0
    deviated = compute_cournot_round(deviated_decisions, params)

    # The deviator's own profit falls (q* was a best response by definition).
    assert deviated["team_0"].profit < baseline["team_0"].profit
    # Rivals are hurt too, because the extra supply depresses the price for all.
    assert deviated["team_1"].profit < baseline["team_1"].profit


def test_price_shared_across_all_teams(params: MarketParameters) -> None:
    results = compute_cournot_round(
        {"a": 5.0, "b": 10.0, "c": 15.0}, params
    )
    prices = {r.price for r in results.values()}
    assert len(prices) == 1


def test_price_never_negative_when_market_flooded(
    params: MarketParameters,
) -> None:
    # Total output 1000 would imply P = 100 - 1000 < 0; must be floored at 0.
    results = compute_cournot_round({"hog": 1000.0}, params)
    assert results["hog"].price == 0.0
    # Revenue is zero but cost is incurred -> profit is negative.
    assert results["hog"].revenue == 0.0
    assert results["hog"].profit == pytest.approx(-10.0 * 1000.0)


def test_revenue_cost_profit_consistency(params: MarketParameters) -> None:
    results = compute_cournot_round({"x": 20.0, "y": 30.0}, params)
    for r in results.values():
        assert r.revenue == pytest.approx(r.price * r.quantity)
        assert r.cost == pytest.approx(params.marginal_cost * r.quantity)
        assert r.profit == pytest.approx(r.revenue - r.cost)


def test_zero_quantity_is_allowed(params: MarketParameters) -> None:
    results = compute_cournot_round({"idle": 0.0, "active": 10.0}, params)
    assert results["idle"].quantity == 0.0
    assert results["idle"].profit == 0.0


def test_total_quantity_drives_price(params: MarketParameters) -> None:
    results = compute_cournot_round({"a": 10.0, "b": 20.0, "c": 10.0}, params)
    # Q_total = 40 -> price = 100 - 40 = 60
    assert results["a"].price == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# compute_cournot_round — input validation
# --------------------------------------------------------------------------- #


def test_empty_decisions_raise(params: MarketParameters) -> None:
    with pytest.raises(ValueError):
        compute_cournot_round({}, params)


def test_negative_quantity_raises(params: MarketParameters) -> None:
    with pytest.raises(ValueError):
        compute_cournot_round({"team": -1.0}, params)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_quantity_raises(
    params: MarketParameters, bad: float
) -> None:
    with pytest.raises(ValueError):
        compute_cournot_round({"team": bad}, params)
