"""Обёртка над движком Курно для ролевого («сложного») раунда.

Идея (см. DECISIONS.md №10): lead-роль команды вносит multi-attribute решение —
финальный объём Q, ценовой сигнал маркетолога и распределение долей издержек.
Этот модуль валидирует и сводит такое решение к единственному числу Q,
совместимому с существующим :func:`core.market_engine.compute_cournot_round`,
и дальше зовёт его как чёрный ящик. Формула рынка ``P = a - b*Q`` не
переписывается и не дублируется.

Как и :mod:`core.market_engine`, модуль полностью чистый: никакого I/O,
случайности или LLM — одинаковый вход всегда даёт одинаковый выход.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from core.market_engine import MarketParameters, TeamResult, compute_cournot_round

__all__ = [
    "SHARE_SUM_TOLERANCE",
    "LeadRoleDecision",
    "aggregate_role_proposals",
    "compute_roles_round",
]

# Допуск на сумму долей издержек: float-арифметика не даст ровно 1.0.
SHARE_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class LeadRoleDecision:
    """Multi-attribute решение lead-роли за команду.

    Parameters
    ----------
    quantity:
        Финальный объём выпуска Q. Слово lead — финальное: именно это число
        идёт в рынок, остальные атрибуты только валидируются и сохраняются
        для аудита.
    price_signal:
        Ожидаемая рыночная цена от маркетолога (``None`` — сигнала не было).
        Не может быть отрицательной.
    cost_shares:
        Распределение долей издержек по ролям, ``role -> доля``. Каждая доля
        неотрицательна, сумма равна 1.0 с допуском ``SHARE_SUM_TOLERANCE``.
        Пустой словарь означает «распределение не задано».

    Raises
    ------
    ValueError
        Если любой атрибут вне допустимого диапазона.
    """

    quantity: float
    price_signal: float | None = None
    cost_shares: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity):
            raise ValueError(f"quantity must be finite, got {self.quantity}")
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0, got {self.quantity}")
        if self.price_signal is not None:
            if not math.isfinite(self.price_signal):
                raise ValueError(
                    f"price_signal must be finite, got {self.price_signal}"
                )
            if self.price_signal < 0:
                raise ValueError(
                    f"price_signal must be >= 0, got {self.price_signal}"
                )
        if self.cost_shares:
            for role, share in self.cost_shares.items():
                if not math.isfinite(share) or share < 0:
                    raise ValueError(
                        f"cost share for {role!r} must be a non-negative "
                        f"finite number, got {share}"
                    )
            total = sum(self.cost_shares.values())
            if abs(total - 1.0) > SHARE_SUM_TOLERANCE:
                raise ValueError(
                    f"cost shares must sum to 1.0 (±{SHARE_SUM_TOLERANCE}), "
                    f"got {total}"
                )


def aggregate_role_proposals(
    proposals: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Свести Q-предложения ролей в одно число — взвешенное среднее.

    Хелпер для lead-роли: если она не задаёт Q напрямую, а строит его из
    предложений ролей, веса нормируются на их сумму (передавать «2, 1, 1»
    так же законно, как «0.5, 0.25, 0.25»).

    Parameters
    ----------
    proposals:
        ``role -> предлагаемый Q``. Все значения конечны и неотрицательны.
    weights:
        ``role -> вес``. ``None`` — равные веса. Ключи весов обязаны быть
        подмножеством ключей предложений; роли без веса получают вес 0.

    Returns
    -------
    float
        Агрегированный объём Q.

    Raises
    ------
    ValueError
        Если предложений нет, среди них есть некорректные числа, вес
        отрицателен, указан для неизвестной роли или сумма весов равна нулю.
    """
    if not proposals:
        raise ValueError("proposals must contain at least one role")
    for role, quantity in proposals.items():
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError(
                f"proposal for {role!r} must be a non-negative finite "
                f"number, got {quantity}"
            )

    if weights is None:
        weights = dict.fromkeys(proposals, 1.0)

    unknown = set(weights) - set(proposals)
    if unknown:
        raise ValueError(f"weights given for unknown roles: {sorted(unknown)}")
    for role, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"weight for {role!r} must be a non-negative finite number, "
                f"got {weight}"
            )
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")

    return sum(
        proposals[role] * weight for role, weight in weights.items()
    ) / total_weight


def compute_roles_round(
    decisions: Mapping[str, LeadRoleDecision],
    params: MarketParameters,
) -> dict[str, TeamResult]:
    """Посчитать ролевой раунд существующим движком Курно.

    Извлекает из каждого multi-attribute решения финальный Q и передаёт его
    в :func:`core.market_engine.compute_cournot_round` без каких-либо
    изменений в самой рыночной механике — результат по построению совпадает
    с обычным симметричным раундом при тех же объёмах.

    Parameters
    ----------
    decisions:
        ``team_id -> LeadRoleDecision`` (решения уже провалидированы своим
        конструктором).
    params:
        Параметры рынка раунда.

    Returns
    -------
    dict[str, TeamResult]
        Ровно тот же формат, что у существующего движка.
    """
    quantities = {team_id: d.quantity for team_id, d in decisions.items()}
    return compute_cournot_round(quantities, params)
