"""Рубрики по методу (core/rubrics.py): капканы против общего ответа.

Каждая рубрика обязана требовать числа из данных и — там, где ловушка
смещает оценку, — её признание. Здесь проверяется структура: покрытие
методов с кейсом, веса, уникальность id, ключевые слова капканов.
"""

from __future__ import annotations

import pytest

from core.cases import supported_methods
from core.rubrics import DEFAULT_RUBRICS, rubric_for_method
from core.trap import naive_slope_ratio
from db.enums import Method


def test_every_supported_case_has_a_rubric() -> None:
    assert set(DEFAULT_RUBRICS) == set(supported_methods())


@pytest.mark.parametrize("method", list(DEFAULT_RUBRICS))
def test_weights_sum_to_one_and_ids_unique(method: Method) -> None:
    rubric = DEFAULT_RUBRICS[method]
    assert sum(c.weight for c in rubric) == pytest.approx(1.0)
    ids = [c.id for c in rubric]
    assert len(ids) == len(set(ids))
    assert all(0 < c.weight <= 1 for c in rubric)


@pytest.mark.parametrize("method", list(DEFAULT_RUBRICS))
def test_every_rubric_demands_numbers_and_the_decision_link(method: Method) -> None:
    """Общий ответ без чисел не должен закрывать рубрику: хотя бы два критерия
    требуют число, и один связывает оценку с объёмом."""
    rubric = DEFAULT_RUBRICS[method]
    numeric = [c for c in rubric if "числ" in c.description.lower()]
    assert len(numeric) >= 2, method
    assert any(c.id == "quantity_link" for c in rubric), method


def test_biased_methods_require_naming_the_trap() -> None:
    """Там, где наивная оценка смещена (core.trap), рубрика требует назвать
    ловушку и направление ошибки — иначе капкан только в данных."""
    for method in DEFAULT_RUBRICS:
        if naive_slope_ratio(method) is None:
            continue
        descriptions = " ".join(c.description.lower() for c in DEFAULT_RUBRICS[method])
        assert "ловушк" in descriptions, method
        assert "занижа" in descriptions or "завыша" in descriptions, method


def test_rubric_for_method_returns_copy() -> None:
    a = rubric_for_method(Method.OLS_SIMPLE)
    b = rubric_for_method(Method.OLS_SIMPLE)
    assert a == b and a is not b


def test_method_without_rubric_raises() -> None:
    with pytest.raises(NotImplementedError, match="autocorrelation"):
        rubric_for_method(Method.AUTOCORRELATION)
