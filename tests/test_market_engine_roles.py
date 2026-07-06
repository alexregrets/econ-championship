"""Тесты обёртки ролевого раунда (core/market_engine_roles.py).

Чистые функции, без сети и БД, конкретные числовые проверки — тем же стилем,
что tests/test_market_engine.py. Главный инвариант: обёртка не добавляет
собственной экономики, результат совпадает с прямым вызовом движка.
"""

from __future__ import annotations

import pytest

from core.market_engine import MarketParameters, compute_cournot_round
from core.market_engine_roles import (
    LeadRoleDecision,
    aggregate_role_proposals,
    compute_roles_round,
)

PARAMS = MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


# --------------------------------------------------------------------------- #
# LeadRoleDecision: валидация multi-attribute решения
# --------------------------------------------------------------------------- #


def test_decision_accepts_valid_attributes() -> None:
    decision = LeadRoleDecision(
        quantity=20.0,
        price_signal=55.0,
        cost_shares={"financier": 0.55, "sales_analyst": 0.25, "marketer": 0.20},
    )
    assert decision.quantity == 20.0
    assert decision.price_signal == 55.0


def test_decision_rejects_negative_quantity() -> None:
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=-1.0)


def test_decision_rejects_non_finite_quantity() -> None:
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=float("nan"))
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=float("inf"))


def test_decision_rejects_negative_price_signal() -> None:
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=1.0, price_signal=-0.01)


def test_decision_rejects_shares_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=1.0, cost_shares={"a": 0.5, "b": 0.6})


def test_decision_rejects_negative_share() -> None:
    with pytest.raises(ValueError):
        LeadRoleDecision(quantity=1.0, cost_shares={"a": -0.2, "b": 1.2})


def test_decision_accepts_shares_within_tolerance() -> None:
    # 0.1*3 в float — не ровно 0.3; допуск обязан это прощать.
    decision = LeadRoleDecision(
        quantity=1.0, cost_shares={"a": 0.1, "b": 0.2, "c": 0.7}
    )
    assert sum(decision.cost_shares.values()) == pytest.approx(1.0)


def test_decision_allows_empty_shares_and_no_signal() -> None:
    decision = LeadRoleDecision(quantity=5.0)
    assert decision.price_signal is None
    assert decision.cost_shares == {}


# --------------------------------------------------------------------------- #
# aggregate_role_proposals: взвешенное среднее с нормировкой
# --------------------------------------------------------------------------- #


def test_aggregate_equal_weights_is_plain_average() -> None:
    q = aggregate_role_proposals({"m": 10.0, "s": 20.0, "f": 30.0})
    assert q == pytest.approx(20.0)


def test_aggregate_weighted_average_exact() -> None:
    q = aggregate_role_proposals(
        {"m": 10.0, "s": 20.0, "f": 40.0},
        weights={"m": 0.5, "s": 0.25, "f": 0.25},
    )
    # 10*0.5 + 20*0.25 + 40*0.25 = 5 + 5 + 10 = 20
    assert q == pytest.approx(20.0)


def test_aggregate_normalizes_unnormalized_weights() -> None:
    # Веса 2:1:1 эквивалентны 0.5:0.25:0.25.
    q = aggregate_role_proposals(
        {"m": 10.0, "s": 20.0, "f": 40.0},
        weights={"m": 2.0, "s": 1.0, "f": 1.0},
    )
    assert q == pytest.approx(20.0)


def test_aggregate_missing_weight_means_zero() -> None:
    # Роль без веса не влияет на агрегат.
    q = aggregate_role_proposals(
        {"m": 10.0, "s": 99.0}, weights={"m": 1.0}
    )
    assert q == pytest.approx(10.0)


def test_aggregate_rejects_empty_proposals() -> None:
    with pytest.raises(ValueError):
        aggregate_role_proposals({})


def test_aggregate_rejects_negative_proposal() -> None:
    with pytest.raises(ValueError):
        aggregate_role_proposals({"m": -5.0})


def test_aggregate_rejects_unknown_weight_key() -> None:
    with pytest.raises(ValueError):
        aggregate_role_proposals({"m": 10.0}, weights={"ghost": 1.0})


def test_aggregate_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError):
        aggregate_role_proposals({"m": 10.0}, weights={"m": 0.0})


def test_aggregate_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        aggregate_role_proposals({"m": 10.0, "s": 20.0}, weights={"m": -1.0, "s": 2.0})


# --------------------------------------------------------------------------- #
# compute_roles_round: движок как чёрный ящик
# --------------------------------------------------------------------------- #


def test_roles_round_matches_plain_engine_exactly() -> None:
    """Обёртка не искажает экономику: результат тождественен прямому вызову."""
    decisions = {
        "1": LeadRoleDecision(quantity=30.0, price_signal=50.0),
        "2": LeadRoleDecision(quantity=20.0),
    }
    roles_results = compute_roles_round(decisions, PARAMS)
    plain_results = compute_cournot_round({"1": 30.0, "2": 20.0}, PARAMS)
    assert roles_results == plain_results


def test_roles_round_duopoly_numbers() -> None:
    """Дуополия: q=(30, 20) → P = 100 - 50 = 50, прибыли 1200 и 800."""
    decisions = {
        "1": LeadRoleDecision(quantity=30.0),
        "2": LeadRoleDecision(quantity=20.0),
    }
    results = compute_roles_round(decisions, PARAMS)
    assert results["1"].price == pytest.approx(50.0)
    assert results["1"].profit == pytest.approx(1200.0)
    assert results["2"].profit == pytest.approx(800.0)


def test_roles_round_aggregated_proposals_end_to_end() -> None:
    """Полный путь: предложения ролей → агрегат → решение → рынок."""
    q = aggregate_role_proposals({"marketer": 18.0, "sales_analyst": 22.0})
    decision = LeadRoleDecision(quantity=q)
    results = compute_roles_round({"1": decision}, PARAMS)
    # Монополист с Q=20: P = 80, прибыль = (80-10)*20 = 1400.
    assert results["1"].price == pytest.approx(80.0)
    assert results["1"].profit == pytest.approx(1400.0)
