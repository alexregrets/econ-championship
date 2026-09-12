"""Детектор «кто попался»: неявный наклон команды против наивной процедуры.

Спецификация (решение 08.09): в раундах, где метод смещает точечную оценку,
неявный ``b̂`` команды сравнивается с тем, что даст наивная сквозная регрессия
на тех же данных. Совпало — флаг. Там, где наивная оценка не смещена
(парная регрессия, гетероскедастичность), детектору нечего ловить, и он
обязан честно сказать «ловушки нет», а не выдумывать вердикт.
"""

from __future__ import annotations

import pytest

from core.cases import DEFAULT_REGIME_SHIFT
from core.trap import Verdict, naive_slope_ratio, trap_verdict
from db.enums import Method


def test_regime_shift_naive_ratio_comes_from_case_spec() -> None:
    assert naive_slope_ratio(Method.OLS_MULTIPLE) == DEFAULT_REGIME_SHIFT.naive_slope_ratio


@pytest.mark.parametrize("method", [Method.OLS_SIMPLE, Method.HETEROSCEDASTICITY])
def test_unbiased_methods_have_no_trap(method: Method) -> None:
    assert naive_slope_ratio(method) is None


def test_method_without_case_has_no_trap() -> None:
    """Метод без кейса не роняет разбор: ловушки под него просто нет."""
    assert naive_slope_ratio(Method.AUTOCORRELATION) is None


def test_true_slope_is_sound() -> None:
    assert trap_verdict(1.0, 1.0, 0.35) is Verdict.SOUND


def test_naive_slope_is_trapped() -> None:
    assert trap_verdict(0.35, 1.0, 0.35) is Verdict.TRAPPED


def test_verdict_scales_with_true_slope() -> None:
    """Допуск относительный: рынок с b=40 судится теми же долями, что с b=1."""
    assert trap_verdict(14.0, 40.0, 0.35) is Verdict.TRAPPED
    assert trap_verdict(44.0, 40.0, 0.35) is Verdict.SOUND


def test_between_is_off() -> None:
    assert trap_verdict(0.7, 1.0, 0.35) is Verdict.OFF


def test_no_trap_when_method_is_unbiased() -> None:
    assert trap_verdict(0.35, 1.0, None) is Verdict.NO_TRAP


def test_tolerance_bands_must_not_overlap() -> None:
    """Допуск, при котором «верно» и «попался» пересекаются, — ошибка
    настройки, а не тихий выбор одного из двух."""
    with pytest.raises(ValueError, match="tolerance"):
        trap_verdict(0.5, 1.0, 0.35, tolerance=0.4)


@pytest.mark.parametrize("implied", [0.0, -1.0, float("nan"), float("inf")])
def test_degenerate_implied_slope_raises(implied: float) -> None:
    with pytest.raises(ValueError):
        trap_verdict(implied, 1.0, 0.35)


def test_nonpositive_true_slope_raises() -> None:
    with pytest.raises(ValueError):
        trap_verdict(1.0, 0.0, 0.35)
