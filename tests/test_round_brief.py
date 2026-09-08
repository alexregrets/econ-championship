"""Тесты брифинга команды.

Что проверяется:
- на каждый реализованный метод есть легенда, и набор легенд в точности
  совпадает с набором кейсов из :mod:`core.cases`;
- брифинг не разглашает истину (числовые ``a``/``b``/издержки, слова разбора);
- собранный документ самодостаточен: содержит сюжет, задачу и все столбцы
  выданной таблицы;
- метод без брифинга падает громко.
"""

from __future__ import annotations

import pytest

from core.cases import supported_methods
from db.enums import Method
from services.dataset_export import DISCLAIMER, Column, Dataset
from services.round_brief import NARRATIVES, case_narrative, render_team_brief


def _dataset(method: Method) -> Dataset:
    """Небольшая правдоподобная выгрузка для рендера брифинга."""
    extra = (
        (Column("regime_new", "0/1", "Режим рынка: 0 — до, 1 — после"),)
        if method is Method.OLS_MULTIPLE
        else ()
    )
    columns = (
        Column("period", "номер", "Порядковый номер периода"),
        Column("quantity", "млн т", "Суммарный выпуск рынка за период"),
        Column("price", "$/т", "Рыночная цена при этом выпуске"),
        *extra,
    )
    rows = [
        {"period": float(i), "quantity": 100.0 + i, "price": 500.0 - i}
        for i in range(1, 6)
    ]
    return Dataset(
        round_number=1,
        method=method,
        title="Проверочный раунд",
        columns=columns,
        rows=rows,
    )


def test_narratives_match_implemented_cases() -> None:
    # Раунд, для которого можно сгенерировать данные, обязан иметь брифинг —
    # и наоборот. Без этого либо данные без легенды, либо легенда без данных.
    assert set(NARRATIVES) == set(supported_methods())


@pytest.mark.parametrize("method", sorted(NARRATIVES, key=str))
def test_brief_renders_for_every_supported_method(method: Method) -> None:
    brief = render_team_brief(_dataset(method))

    assert brief.strip()
    assert DISCLAIMER in brief
    narrative = case_narrative(method)
    assert narrative.companies[0] in brief
    assert narrative.situation[:40] in brief
    assert "## Задача" in brief


@pytest.mark.parametrize("method", sorted(NARRATIVES, key=str))
def test_brief_lists_every_column_of_the_dataset(method: Method) -> None:
    dataset = _dataset(method)
    brief = render_team_brief(dataset)
    for column in dataset.columns:
        assert f"`{column.name}`" in brief


@pytest.mark.parametrize("method", sorted(NARRATIVES, key=str))
def test_brief_does_not_leak_the_truth(method: Method) -> None:
    brief = render_team_brief(_dataset(method)).lower()
    # Слова разбора и калибровки ловушки не должны попадать в брифинг команде.
    for word in ("ловушк", "наивн", "наказ", "разрыв в прибыл", "b_hat", "b̂"):
        assert word not in brief


def test_unsupported_method_raises() -> None:
    absent = next(m for m in Method if m not in NARRATIVES)
    with pytest.raises(NotImplementedError):
        case_narrative(absent)
    with pytest.raises(NotImplementedError):
        render_team_brief(_dataset(absent))


def test_render_is_deterministic() -> None:
    dataset = _dataset(Method.OLS_SIMPLE)
    assert render_team_brief(dataset) == render_team_brief(dataset)


def test_narrative_is_frozen() -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        case_narrative(Method.OLS_SIMPLE).market = "x"  # type: ignore[misc]
