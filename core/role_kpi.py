"""Личные KPI ролей поверх командной прибыли (GAME_DESIGN.md №1).

Зачем. Пока у всех трёх ролей одна цель — прибыль команды, — совещание
вырождается в «сложим кусочки пазла». Личный KPI разводит интересы:

    маркетолог    — доля рынка ``q_i / Q_total``      → тянет объём ВВЕРХ
    финансист     — маржа ``(P − c)/P``               → тянет объём ВНИЗ
    аналитик сбыта— точность прогноза цены            → тянет к своему прогнозу

Конфликт встроен в арифметику: наращивая ``q``, маркетолог сбивает цену, а с
ней маржу финансиста. Lead-роль обязана спор разрулить, а не усреднить.

Итог студента = ``team_weight`` × командная составляющая + ``kpi_weight`` × KPI
(по умолчанию 70/30). Обе составляющие нормируются «относительно лучшего в
раунде» — командная по прибыли, личная внутри своей роли, чтобы маркетолога
сравнивали с маркетологами, а не с финансистами.

Как и остальной ``core``: ни I/O, ни случайности, ни LLM — чистая функция
поверх уже посчитанных результатов движка.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from core.market_engine import TeamResult
from db.enums import Role

__all__ = [
    "ScoreWeights",
    "DEFAULT_WEIGHTS",
    "RoleKpi",
    "KPI_NAMES",
    "market_share",
    "profit_margin",
    "forecast_accuracy",
    "implied_quantity_for_price",
    "compute_role_kpis",
]


@dataclass(frozen=True)
class ScoreWeights:
    """Веса итоговой оценки студента: командная часть + личный KPI.

    Raises
    ------
    ValueError
        Если веса отрицательны или не дают в сумме единицу.
    """

    team: float = 0.7
    kpi: float = 0.3

    def __post_init__(self) -> None:
        if self.team < 0 or self.kpi < 0:
            raise ValueError(
                f"веса должны быть неотрицательны, получено team={self.team}, "
                f"kpi={self.kpi}"
            )
        if not math.isclose(self.team + self.kpi, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"веса должны давать в сумме 1.0, получено {self.team + self.kpi}"
            )


DEFAULT_WEIGHTS = ScoreWeights()

# Человекочитаемое имя показателя каждой роли — для витрины и графиков.
KPI_NAMES: dict[Role, str] = {
    Role.MARKETER: "доля рынка",
    Role.FINANCIER: "маржа",
    Role.SALES_ANALYST: "точность прогноза цены",
}


@dataclass(frozen=True)
class RoleKpi:
    """Оценка одной роли одной команды за раунд.

    Attributes
    ----------
    role:
        Роль студента.
    raw:
        Сырое значение показателя в своих единицах (доля, маржа, точность).
        Показывается на разборе как есть — нормировка скрывает величину.
    raw_name:
        Как этот показатель называется (см. :data:`KPI_NAMES`).
    normalized:
        Сырое значение, отнесённое к лучшему в раунде **внутри своей роли**,
        в диапазоне 0..1. Отрицательные сырые значения (убыточная маржа)
        обрезаются нулём: штраф за минус уже есть в командной части.
    team_component:
        Прибыль команды, отнесённая к лучшей прибыли раунда, 0..1.
    total:
        ``team_weight × team_component + kpi_weight × normalized``.
    has_input:
        Подал ли студент данные, нужные его KPI. Сейчас важно только для
        аналитика сбыта: без прогноза цены точность считать не из чего,
        и KPI обнуляется — это не то же самое, что «прогноз был плохой».
    """

    role: Role
    raw: float
    raw_name: str
    normalized: float
    team_component: float
    total: float
    has_input: bool = True


def market_share(quantity: float, total_quantity: float) -> float:
    """KPI маркетолога: доля команды в суммарном выпуске отрасли.

    Пустой рынок (никто ничего не произвёл) даёт 0 — делить не на что, и
    «доли» в этом раунде не существует ни у кого.
    """
    if total_quantity <= 0:
        return 0.0
    return quantity / total_quantity


def profit_margin(result: TeamResult) -> float:
    """KPI финансиста: маржа ``прибыль / выручка``.

    Равна ``(P − c)/P`` и потому падает, когда команда наращивает объём и
    сбивает цену, — источник конфликта с маркетологом. Нулевая выручка
    (объём 0 или цена обвалилась в ноль) даёт 0: маржа не определена, а
    убыток уже учтён в командной части оценки.
    """
    if result.revenue <= 0:
        return 0.0
    return result.profit / result.revenue


def forecast_accuracy(forecast: float | None, price: float) -> float:
    """KPI аналитика сбыта: ``1 − |прогноз − P| / P``, обрезано снизу нулём.

    Точное попадание — 1.0, промах на величину самой цены и больше — 0.0.
    ``None`` (прогноз не подан) и неположительная фактическая цена дают 0.0:
    в обоих случаях точность не из чего считать.
    """
    if forecast is None or not math.isfinite(forecast) or price <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(forecast - price) / price)


def implied_quantity_for_price(
    *, a: float, b: float, target_price: float, others_total: float
) -> float:
    """Объём, при котором рынок даст ровно ``target_price``.

    Из ``P = a − b·(q + Q_others)`` следует ``q = (a − P)/b − Q_others``.
    Нужен для разбора: «какой объём команды соответствовал бы прогнозу её
    аналитика» — то самое место, где прогноз превращается в спор о решении.
    Отрицательный ответ обрезается нулём: чтобы поднять цену выше, команде
    пришлось бы производить меньше нуля, то есть прогноз недостижим.

    Raises
    ------
    ValueError
        Если наклон спроса неположителен.
    """
    if b <= 0:
        raise ValueError(f"наклон спроса 'b' должен быть положительным, {b}")
    return max(0.0, (a - target_price) / b - others_total)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Отнести значения к лучшему из них; отрицательные считать нулём.

    Если лучшее значение неположительно (все в минусе или все нули), все
    получают 0.0 — «лучший из убыточных» не заслуживает единицы.
    """
    clipped = {key: max(0.0, value) for key, value in values.items()}
    best = max(clipped.values(), default=0.0)
    if best <= 0:
        return dict.fromkeys(clipped, 0.0)
    return {key: value / best for key, value in clipped.items()}


def compute_role_kpis(
    results: Mapping[str, TeamResult],
    price_forecasts: Mapping[str, float | None] | None = None,
    *,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> dict[str, dict[Role, RoleKpi]]:
    """Посчитать KPI всех трёх ролей для каждой команды раунда.

    Parameters
    ----------
    results:
        Результаты движка, ``team_id -> TeamResult`` (симметричного или
        асимметричного — модулю всё равно, он читает готовые цифры).
    price_forecasts:
        ``team_id -> прогноз цены аналитика`` или ``None``, если прогноз не
        подан. Команда, которой нет в словаре, приравнивается к «не подан».
    weights:
        Веса командной и личной составляющих (по умолчанию 70/30).

    Returns
    -------
    dict[str, dict[Role, RoleKpi]]
        По команде — по три оценки (маркетолог, финансист, аналитик).

    Raises
    ------
    ValueError
        Если результатов нет: считать «лучшего в раунде» не из чего.
    """
    if not results:
        raise ValueError("results must contain at least one team")

    forecasts = dict(price_forecasts or {})
    total_quantity = sum(r.quantity for r in results.values())

    raw: dict[Role, dict[str, float]] = {
        Role.MARKETER: {
            team_id: market_share(r.quantity, total_quantity)
            for team_id, r in results.items()
        },
        Role.FINANCIER: {
            team_id: profit_margin(r) for team_id, r in results.items()
        },
        Role.SALES_ANALYST: {
            team_id: forecast_accuracy(forecasts.get(team_id), r.price)
            for team_id, r in results.items()
        },
    }
    normalized = {role: _normalize(values) for role, values in raw.items()}
    team_component = _normalize(
        {team_id: r.profit for team_id, r in results.items()}
    )

    scores: dict[str, dict[Role, RoleKpi]] = {}
    for team_id in results:
        scores[team_id] = {}
        for role in (Role.MARKETER, Role.FINANCIER, Role.SALES_ANALYST):
            kpi_value = normalized[role][team_id]
            scores[team_id][role] = RoleKpi(
                role=role,
                raw=raw[role][team_id],
                raw_name=KPI_NAMES[role],
                normalized=kpi_value,
                team_component=team_component[team_id],
                total=(
                    weights.team * team_component[team_id]
                    + weights.kpi * kpi_value
                ),
                has_input=(
                    forecasts.get(team_id) is not None
                    if role is Role.SALES_ANALYST
                    else True
                ),
            )
    return scores
