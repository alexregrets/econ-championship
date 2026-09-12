"""Детектор по процедуре (core/trap.py, вторая половина) — на настоящих данных
кейсов, а не на игрушечных числах.

Спецификация из живого прогона 12.09: честная наивная команда (сквозной МНК
по данным раунда, лучший ответ по своей линии) обязана получать TRAPPED, а
команда, посчитавшая по истинной линии, — SOUND. Детектор по неявному
наклону это не давал: наивный объём рационализировался наклоном ≈0.58·b.
"""

from __future__ import annotations

import pytest

from core.cases import REGIME_COLUMN, build_case
from core.dataset import HistorySpec
from core.market_engine import MarketParameters
from core.trap import (
    Verdict,
    naive_slope_ratio,
    pooled_ols,
    procedure_quantities,
    quantity_verdict,
)
from db.enums import Method
from services.dataset_export import DEFAULT_PERIODS, _scaled_noise_and_spread

PARAMS = MarketParameters(a=300.0, b=1.0, marginal_cost=30.0)


def _rows(method: Method, n_firms: int, seed: int) -> list[dict[str, float]]:
    """Наблюдения раунда в том виде, в каком их видит выгрузка и разбор."""
    reference = n_firms * (PARAMS.a - PARAMS.marginal_cost) / ((n_firms + 1) * PARAMS.b)
    spec = HistorySpec(
        periods=DEFAULT_PERIODS, seed=seed, **_scaled_noise_and_spread(PARAMS, n_firms)
    )
    case = build_case(method, PARAMS, reference_quantity=reference, spec=spec)
    return [
        {"quantity": o.quantity, "price": o.price, **{k: float(v) for k, v in o.extras.items()}}
        for o in case.observations
    ]


def _procedures(method: Method, n_firms: int, seed: int):  # noqa: ANN202
    rows = _rows(method, n_firms, seed)
    return procedure_quantities(
        rows,
        true_a=PARAMS.a,
        true_b=PARAMS.b,
        marginal_cost=PARAMS.marginal_cost,
        n_firms=n_firms,
        regime_column=REGIME_COLUMN if method is Method.OLS_MULTIPLE else None,
    )


def test_pooled_ols_recovers_exact_line_without_noise() -> None:
    rows = [{"quantity": q, "price": 100.0 - 2.0 * q} for q in (10.0, 20.0, 30.0)]
    a_hat, b_hat = pooled_ols(rows)
    assert a_hat == pytest.approx(100.0)
    assert b_hat == pytest.approx(2.0)


def test_pooled_ols_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        pooled_ols([{"quantity": 1.0, "price": 1.0}])
    with pytest.raises(ValueError):
        pooled_ols([{"quantity": 1.0, "price": 1.0}, {"quantity": 1.0, "price": 2.0}])


@pytest.mark.parametrize("n_firms", [3, 4, 7])
@pytest.mark.parametrize("seed", [1, 2, 3, 11, 42])
def test_regime_shift_naive_overproduces_and_is_trapped(n_firms: int, seed: int) -> None:
    """Наивная процедура даёт наклон ≈0.35·b и объём заметно выше верного;
    сданный наивный объём — TRAPPED, верный — SOUND."""
    proc = _procedures(Method.OLS_MULTIPLE, n_firms, seed)
    assert proc.naive_slope == pytest.approx(0.35 * PARAMS.b, rel=0.05)
    assert proc.naive_quantity > 1.3 * proc.sound_quantity
    ratio = naive_slope_ratio(Method.OLS_MULTIPLE)
    assert quantity_verdict(proc.naive_quantity, proc, ratio) is Verdict.TRAPPED
    assert quantity_verdict(proc.sound_quantity, proc, ratio) is Verdict.SOUND
    # Между процедурами — ни то ни другое.
    mid = (proc.naive_quantity + proc.sound_quantity) / 2
    assert quantity_verdict(mid, proc, ratio) is Verdict.OFF


def test_sound_quantity_equals_nash_when_history_sits_at_nash() -> None:
    """Ожидание соперников из истории ≈ (n−1)/n от равновесного выпуска, и
    верная процедура возвращает объём Нэша — с точностью до шума истории."""
    n = 7
    proc = _procedures(Method.OLS_MULTIPLE, n, seed=5)
    nash = (PARAMS.a - PARAMS.marginal_cost) / (PARAMS.b * (n + 1))
    assert proc.sound_quantity == pytest.approx(nash, rel=0.15)


def test_verdict_does_not_depend_on_other_teams_in_the_round() -> None:
    """Ожидание соперников — из истории, не из фактического раунда: две
    наивные команды не прячут друг друга (ограничение детектора по наклону)."""
    proc = _procedures(Method.OLS_MULTIPLE, 7, seed=9)
    ratio = naive_slope_ratio(Method.OLS_MULTIPLE)
    # Сколько бы команд ни сдали наивный объём — вердикт каждой одинаков.
    for _ in range(3):
        assert quantity_verdict(proc.naive_quantity, proc, ratio) is Verdict.TRAPPED


@pytest.mark.parametrize("method", [Method.OLS_SIMPLE, Method.HETEROSCEDASTICITY])
def test_unbiased_methods_have_indistinguishable_procedures(method: Method) -> None:
    """Без ловушки наивная и верная процедура дают почти один объём — и метод
    честно объявлен NO_TRAP, а не судится по шуму."""
    proc = _procedures(method, 4, seed=3)
    assert proc.naive_quantity == pytest.approx(proc.sound_quantity, rel=0.25)
    assert quantity_verdict(proc.sound_quantity, proc, naive_slope_ratio(method)) is Verdict.NO_TRAP


def test_quantity_verdict_rejects_bad_input() -> None:
    proc = _procedures(Method.OLS_MULTIPLE, 4, seed=1)
    with pytest.raises(ValueError):
        quantity_verdict(-1.0, proc, 0.35)
    with pytest.raises(ValueError):
        quantity_verdict(float("nan"), proc, 0.35)
    with pytest.raises(ValueError):
        quantity_verdict(1.0, proc, 0.35, tolerance=0.0)
