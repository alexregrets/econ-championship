"""Асимметричный Курно: firm-specific предельные издержки при том же спросе.

Расширение движка для случая, когда фирмы различаются издержками ``c_i`` при
общем линейном спросе ``P = a - b*Q_total``. Существующий симметричный движок
(:mod:`core.market_engine`) не меняется и остаётся поведением по умолчанию;
этот модуль — явно включаемая опция (см. DECISIONS.md №17): асимметрия
работает только там, где вызывающий код сам передал пофирменные издержки.

Формулы внутреннего равновесия (все фирмы активны, q_i* > 0):

    FOC фирмы i:  a - b*Q - b*q_i - c_i = 0
    Сумма FOC:    Q* = (n*a - Σc) / ((n+1)*b)
    Цена:         P* = a - b*Q* = (a + Σc) / (n+1)
    Объём фирмы:  q_i* = (P* - c_i) / b

При одинаковых ``c_i`` всё сводится к существующей симметричной формуле
``q* = (a - c) / ((n+1)*b)`` — закреплено regression-тестами против
:func:`core.market_engine.nash_equilibrium`.

Как и остальной ``core``: никакого I/O, случайности или LLM — одинаковый
вход всегда даёт одинаковый выход.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.market_engine import TeamResult

__all__ = [
    "AsymmetricNashResult",
    "asymmetric_nash_equilibrium",
    "implied_marginal_costs",
    "compute_asymmetric_cournot_round",
]


@dataclass(frozen=True)
class AsymmetricNashResult:
    """Внутреннее равновесие Нэша асимметричного Курно.

    Attributes
    ----------
    quantities:
        Равновесные объёмы ``q_i*`` в том же порядке, в котором переданы
        издержки фирм.
    price:
        Равновесная цена ``P*`` (общая для всех фирм).
    total_quantity:
        Суммарный выпуск ``Q* = Σ q_i*``.
    """

    quantities: tuple[float, ...]
    price: float
    total_quantity: float


def _validate_demand(a: float, b: float) -> None:
    """Проверить параметры спроса — те же требования, что у MarketParameters."""
    if not math.isfinite(a) or a <= 0:
        raise ValueError(f"demand intercept 'a' must be positive, got {a}")
    if not math.isfinite(b) or b <= 0:
        raise ValueError(f"demand slope 'b' must be positive, got {b}")


def _validate_costs(a: float, costs: Sequence[float]) -> None:
    """Проверить список издержек: непустой, конечные значения в [0, a)."""
    if len(costs) == 0:
        raise ValueError("marginal costs must contain at least one firm")
    for index, cost in enumerate(costs):
        if not math.isfinite(cost):
            raise ValueError(f"firm #{index} has non-finite marginal cost {cost}")
        if cost < 0:
            raise ValueError(
                f"firm #{index} has negative marginal cost {cost}; costs must be >= 0"
            )
        if cost >= a:
            raise ValueError(
                f"firm #{index} marginal cost ({cost}) must be below demand "
                f"intercept a ({a}); otherwise it can never produce profitably"
            )


def asymmetric_nash_equilibrium(
    a: float, b: float, marginal_costs: Sequence[float]
) -> AsymmetricNashResult:
    """Посчитать равновесие Нэша для n фирм с разными предельными издержками.

    Parameters
    ----------
    a:
        Точка насыщения спроса (цена, при которой спрос нулевой). Положительна.
    b:
        Наклон обратного спроса. Положителен.
    marginal_costs:
        Издержки ``c_i`` по фирмам; каждая конечна, неотрицательна и меньше
        ``a``. Порядок фирм сохраняется в результате.

    Returns
    -------
    AsymmetricNashResult
        Равновесные объёмы, цена и суммарный выпуск.

    Raises
    ------
    ValueError
        Если параметры спроса или издержки вне допустимых диапазонов, либо
        внутреннее равновесие не существует: у какой-то фирмы ``q_i* < 0``
        (издержки настолько выше конкурентов, что ей выгоднее не производить —
        равновесия с неактивными фирмами вне учебной модели, модуль честно
        отказывается, а не выдаёт отрицательный объём).
    """
    _validate_demand(a, b)
    _validate_costs(a, marginal_costs)

    n = len(marginal_costs)
    total_cost = sum(marginal_costs)
    total_quantity = (n * a - total_cost) / ((n + 1) * b)
    price = (a + total_cost) / (n + 1)

    quantities: list[float] = []
    for index, cost in enumerate(marginal_costs):
        quantity = (price - cost) / b
        if quantity < 0:
            raise ValueError(
                f"firm #{index} (c={cost}) is inactive in equilibrium "
                f"(q*={quantity:.6g} < 0): its cost is too far above rivals'; "
                "equilibria with inactive firms are out of scope"
            )
        quantities.append(quantity)

    return AsymmetricNashResult(
        quantities=tuple(quantities),
        price=price,
        total_quantity=total_quantity,
    )


def implied_marginal_costs(
    a: float, b: float, observed_quantities: Sequence[float]
) -> tuple[float, ...]:
    """Обратная калибровка: издержки, при которых наблюдаемые объёмы — Нэш.

    Из FOC асимметричного Курно ``c_i = P - b*q_i``, где
    ``P = a - b*Σq`` — цена в наблюдаемой точке. По построению
    :func:`asymmetric_nash_equilibrium` с этими издержками воспроизводит
    ``observed_quantities`` точно (с точностью float-арифметики).

    Это «implied costs» в эконометрическом смысле — выявленные из поведения,
    а не взятые из отчётности; см. DECISIONS.md №18.

    Parameters
    ----------
    a:
        Точка насыщения спроса (уже откалиброванная, здесь не пересчитывается).
    b:
        Наклон обратного спроса (аналогично зафиксирован).
    observed_quantities:
        Фактические объёмы фирм в наблюдаемой точке; каждый конечен и строго
        положителен (для неактивной фирмы FOC не определяет издержки).

    Returns
    -------
    tuple[float, ...]
        Издержки ``c_i`` в порядке переданных объёмов.

    Raises
    ------
    ValueError
        Если параметры спроса некорректны, объёмы пусты/неположительны, точка
        лежит вне положительной ветви спроса (``P <= 0``) или какой-то фирме
        приписывается отрицательная себестоимость — наблюдаемая точка
        несовместима с моделью Курно на этом спросе.
    """
    _validate_demand(a, b)
    if len(observed_quantities) == 0:
        raise ValueError("observed_quantities must contain at least one firm")
    for index, quantity in enumerate(observed_quantities):
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(
                f"firm #{index} observed quantity must be finite and > 0, "
                f"got {quantity}"
            )

    price = a - b * sum(observed_quantities)
    if price <= 0:
        raise ValueError(
            f"observed point implies non-positive price P={price:.6g}; "
            "it lies off the positive branch of the demand curve"
        )

    costs: list[float] = []
    for index, quantity in enumerate(observed_quantities):
        cost = price - b * quantity
        if cost < 0:
            raise ValueError(
                f"firm #{index} (q={quantity}) would need negative marginal "
                f"cost {cost:.6g}: the observed point is not a Cournot "
                "equilibrium on this demand curve"
            )
        costs.append(cost)
    return tuple(costs)


def compute_asymmetric_cournot_round(
    decisions: Mapping[str, float],
    *,
    a: float,
    b: float,
    marginal_costs: Mapping[str, float],
) -> dict[str, TeamResult]:
    """Посчитать раунд Курно с пофирменными издержками.

    Асимметричный аналог :func:`core.market_engine.compute_cournot_round`:
    рыночная цена та же — ``max(0, a - b*Q_total)``, — различаются только
    издержки команд. Возвращает тот же :class:`~core.market_engine.TeamResult`,
    поэтому потребители результата (сохранение Result, лидерборд) не отличают
    его от симметричного раунда. При одинаковых издержках у всех команд
    результат совпадает с симметричным движком — закреплено тестом.

    Parameters
    ----------
    decisions:
        ``team_id -> объём``; объёмы конечны и неотрицательны.
    a:
        Точка насыщения спроса. Положительна.
    b:
        Наклон обратного спроса. Положителен.
    marginal_costs:
        ``team_id -> c_i``. Ключи обязаны совпадать с ключами ``decisions``
        один в один: издержки без решения или решение без издержек — ошибка
        конфигурации, а не повод молча подставить дефолт.

    Returns
    -------
    dict[str, TeamResult]
        Результат каждой команды; цена общая для всех.

    Raises
    ------
    ValueError
        Если решений нет, объём отрицателен/не конечен, издержки вне
        диапазона или наборы команд в ``decisions`` и ``marginal_costs``
        не совпадают.
    """
    _validate_demand(a, b)
    if not decisions:
        raise ValueError("decisions must contain at least one team")
    if set(decisions) != set(marginal_costs):
        missing = sorted(set(decisions) - set(marginal_costs))
        extra = sorted(set(marginal_costs) - set(decisions))
        raise ValueError(
            "marginal_costs must cover exactly the deciding teams; "
            f"missing costs for {missing}, costs without decisions for {extra}"
        )

    total_quantity = 0.0
    for team_id, quantity in decisions.items():
        if not math.isfinite(quantity):
            raise ValueError(f"team {team_id!r} has non-finite quantity {quantity}")
        if quantity < 0:
            raise ValueError(
                f"team {team_id!r} has negative quantity {quantity}; "
                "quantities must be >= 0"
            )
        total_quantity += quantity
    for team_id, cost in marginal_costs.items():
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                f"team {team_id!r} has invalid marginal cost {cost}; "
                "costs must be finite and >= 0"
            )

    price = max(0.0, a - b * total_quantity)

    results: dict[str, TeamResult] = {}
    for team_id, quantity in decisions.items():
        revenue = price * quantity
        cost = marginal_costs[team_id] * quantity
        results[team_id] = TeamResult(
            team_id=team_id,
            quantity=quantity,
            price=price,
            revenue=revenue,
            cost=cost,
            profit=revenue - cost,
        )
    return results
