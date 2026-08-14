"""Рыночные события: относительные шоки спроса и издержек поверх раунда.

Событие — это внешнее обстоятельство, которое меняет рынок между раундами, не
трогая решения команд: «ОПЕК отказалась сокращать добычу», «налоговый манёвр»,
«девальвация». Механически событие = множитель к одному параметру рынка:

    DEMAND_SHIFT      a *= (1 + magnitude)
    ELASTICITY_SHIFT  b *= (1 + magnitude)
    COST_SHOCK        c *= (1 + magnitude)   (и market_mc, и каждое c_i)

Величина задаётся относительно (``+0.10`` = +10%), поэтому события переносятся
между сценариями с разными единицами. Композиция мультипликативная и потому
**не зависит от порядка** событий — раунд с одним и тем же набором событий
считается одинаково независимо от того, в каком порядке их завёл профессор.

Как и остальной ``core``: ни I/O, ни случайности, ни LLM. Движки Курно
(:mod:`core.market_engine`, :mod:`core.market_engine_asymmetric`) не меняются —
им передаются уже сдвинутые параметры.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from core.market_engine import MarketParameters
from db.enums import EventKind

__all__ = [
    "MarketShock",
    "ShockFactors",
    "EventPreset",
    "EVENT_PRESETS",
    "preset_by_key",
    "combine_shocks",
    "apply_to_parameters",
    "apply_to_demand",
    "apply_to_costs",
    "shock_summary",
]


@dataclass(frozen=True)
class MarketShock:
    """Чистое описание одного шока: что сдвигается и насколько.

    Attributes
    ----------
    kind:
        Какой параметр рынка сдвигается (см. :class:`~db.enums.EventKind`).
    magnitude:
        Относительное изменение: ``0.10`` — плюс 10%, ``-0.25`` — минус 25%.
        Строго больше ``-1``: параметр нельзя обнулить или сделать
        отрицательным одним событием.
    headline:
        Короткий заголовок для витрины и графиков («ОПЕК не сокращает добычу»).

    Raises
    ------
    ValueError
        Если величина не конечна или не больше ``-1``.
    """

    kind: EventKind
    magnitude: float
    headline: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude):
            raise ValueError(
                f"величина шока должна быть конечной, получено {self.magnitude}"
            )
        if self.magnitude <= -1.0:
            raise ValueError(
                f"величина шока должна быть больше -1 (падение менее чем на 100%), "
                f"получено {self.magnitude}"
            )

    @property
    def factor(self) -> float:
        """Множитель ``1 + magnitude``, применяемый к параметру."""
        return 1.0 + self.magnitude


@dataclass(frozen=True)
class ShockFactors:
    """Свёртка набора событий в три множителя — по одному на параметр рынка.

    Нейтральный набор (событий нет) даёт единицы, то есть рынок без событий
    считается ровно так же, как считался до появления этого модуля.
    """

    demand: float = 1.0
    slope: float = 1.0
    cost: float = 1.0


def combine_shocks(shocks: Iterable[MarketShock]) -> ShockFactors:
    """Свернуть события раунда в множители по параметрам.

    Одинаковые по типу события перемножаются: два шока спроса −10% и −20%
    дают ``0.9 * 0.8 = 0.72``, а не −30%. Умножение коммутативно, поэтому
    результат не зависит от порядка событий.
    """
    demand = 1.0
    slope = 1.0
    cost = 1.0
    for shock in shocks:
        if shock.kind is EventKind.DEMAND_SHIFT:
            demand *= shock.factor
        elif shock.kind is EventKind.ELASTICITY_SHIFT:
            slope *= shock.factor
        else:
            cost *= shock.factor
    return ShockFactors(demand=demand, slope=slope, cost=cost)


def apply_to_demand(
    a: float, b: float, shocks: Iterable[MarketShock]
) -> tuple[float, float]:
    """Вернуть параметры спроса после событий раунда.

    Parameters
    ----------
    a, b:
        Базовые параметры раунда до событий.
    shocks:
        События раунда (в любом порядке).

    Returns
    -------
    tuple[float, float]
        Сдвинутые ``(a, b)``.

    Raises
    ------
    ValueError
        Если базовые параметры некорректны — сдвинутые получаются
        неположительными не бывает (множители строго положительны).
    """
    if not math.isfinite(a) or a <= 0:
        raise ValueError(f"точка насыщения спроса 'a' должна быть положительной, {a}")
    if not math.isfinite(b) or b <= 0:
        raise ValueError(f"наклон спроса 'b' должен быть положительным, {b}")
    factors = combine_shocks(shocks)
    return a * factors.demand, b * factors.slope


def apply_to_parameters(
    params: MarketParameters, shocks: Iterable[MarketShock]
) -> MarketParameters:
    """Применить события к параметрам симметричного рынка.

    Возвращает новый :class:`~core.market_engine.MarketParameters` — исходный
    неизменен (он frozen). Проверки жизнеспособности рынка выполняет сам
    ``MarketParameters``; здесь ошибка пересобирается с упоминанием событий,
    чтобы профессор понял, что рынок сломали именно они.

    Raises
    ------
    ValueError
        Если после событий рынок нежизнеспособен (издержки догнали или
        перегнали точку насыщения спроса).
    """
    factors = combine_shocks(shocks)
    try:
        return MarketParameters(
            a=params.a * factors.demand,
            b=params.b * factors.slope,
            marginal_cost=params.marginal_cost * factors.cost,
        )
    except ValueError as exc:
        raise ValueError(
            f"события раунда делают рынок нежизнеспособным: {exc}"
        ) from exc


def apply_to_costs(
    costs: Mapping[str, float],
    shocks: Iterable[MarketShock],
    *,
    demand_intercept: float,
) -> dict[str, float]:
    """Применить события к пофирменным издержкам асимметричного раунда.

    На издержки действует только :attr:`~db.enums.EventKind.COST_SHOCK`;
    шоки спроса меняют ``a``/``b`` и сюда не попадают.

    Parameters
    ----------
    costs:
        ``team_id -> c_i`` до событий.
    shocks:
        События раунда.
    demand_intercept:
        **Уже сдвинутая** точка насыщения спроса: издержки проверяются против
        того же рынка, на котором будет считаться раунд.

    Returns
    -------
    dict[str, float]
        Сдвинутые издержки в том же наборе команд.

    Raises
    ------
    ValueError
        Если после шока чья-то себестоимость достигает точки насыщения спроса —
        такая фирма не может производить прибыльно ни при каком объёме, и
        движок обязан отказаться, а не считать заведомо убыточный рынок.
    """
    factors = combine_shocks(shocks)
    shocked = {team_id: cost * factors.cost for team_id, cost in costs.items()}
    unviable = sorted(
        team_id for team_id, cost in shocked.items() if cost >= demand_intercept
    )
    if unviable:
        raise ValueError(
            "события раунда делают рынок нежизнеспособным: после шока издержек "
            f"себестоимость команд {unviable} не ниже точки насыщения спроса "
            f"({demand_intercept:.6g})"
        )
    return shocked


def shock_summary(shocks: Iterable[MarketShock]) -> str:
    """Однострочная сводка событий для витрины и подписей к графикам.

    Пример: ``«спрос −12.0%, издержки +18.0%»``. Пустой набор даёт пустую
    строку — вызывающему коду проще решать, показывать блок или нет.
    """
    factors = combine_shocks(shocks)
    parts: list[str] = []
    for label, factor in (
        ("спрос", factors.demand),
        ("наклон спроса", factors.slope),
        ("издержки", factors.cost),
    ):
        if factor != 1.0:
            parts.append(f"{label} {(factor - 1.0) * 100:+.1f}%")
    return ", ".join(parts)


@dataclass(frozen=True)
class EventPreset:
    """Готовое событие из библиотеки: шок + текст для витрины.

    Величины — **учебная калибровка порядка величины**, а не измеренный эффект
    конкретного исторического события: точных оценок эластичности спроса на
    российскую нефть в открытых источниках нет (та же оговорка, что у implied
    costs в DECISIONS.md №18). События подобраны так, чтобы направление шока
    было экономически бесспорным, а величина — заметной за один раунд.
    """

    key: str
    kind: EventKind
    magnitude: float
    headline: str
    description: str

    def to_shock(self) -> MarketShock:
        """Превратить пресет в чистый шок для движка."""
        return MarketShock(
            kind=self.kind, magnitude=self.magnitude, headline=self.headline
        )


# Библиотека событий сценария «Нефть РФ 2013» и следующих за ним лет.
EVENT_PRESETS: tuple[EventPreset, ...] = (
    EventPreset(
        key="shale",
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.12,
        headline="Сланцевая революция в США",
        description=(
            "Добыча сланцевой нефти в США растёт третий год подряд. Часть "
            "спроса, который раньше закрывался импортом, теперь покрывается "
            "внутри страны — кривая спроса на нашу нефть сдвигается влево."
        ),
    ),
    EventPreset(
        key="opec_no_cut",
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.25,
        headline="ОПЕК отказалась сокращать добычу",
        description=(
            "Картель принял решение удерживать долю рынка, а не цену. "
            "Дополнительные баррели давят на спрос, доступный остальным "
            "производителям, — самый жёсткий шок сценария."
        ),
    ),
    EventPreset(
        key="china_slowdown",
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.08,
        headline="Замедление спроса в Китае",
        description=(
            "Промышленное производство в Китае растёт медленнее прогноза, "
            "закупки сырья снижаются: умеренный сдвиг спроса вниз."
        ),
    ),
    EventPreset(
        key="tax_maneuver",
        kind=EventKind.COST_SHOCK,
        magnitude=0.10,
        headline="Налоговый манёвр: рост НДПИ",
        description=(
            "Ставка налога на добычу повышается. Полная себестоимость барреля "
            "(в неё НДПИ входит) растёт у всех участников рынка."
        ),
    ),
    EventPreset(
        key="sanctions",
        kind=EventKind.COST_SHOCK,
        magnitude=0.18,
        headline="Санкции: закрыт доступ к технологиям и займам",
        description=(
            "Оборудование для трудноизвлекаемых запасов приходится замещать, "
            "заёмные деньги дорожают — издержки добычи растут."
        ),
    ),
    EventPreset(
        key="devaluation",
        kind=EventKind.COST_SHOCK,
        magnitude=-0.20,
        headline="Девальвация рубля",
        description=(
            "Выручка долларовая, а расходы рублёвые: ослабление рубля "
            "снижает себестоимость барреля в долларах."
        ),
    ),
    EventPreset(
        key="substitutes",
        kind=EventKind.ELASTICITY_SHIFT,
        magnitude=0.15,
        headline="Появились доступные субституты",
        description=(
            "Газ и возобновляемая генерация отъедают часть потребления: "
            "цена теперь резче реагирует на суммарный объём добычи."
        ),
    ),
)

_PRESETS_BY_KEY: dict[str, EventPreset] = {p.key: p for p in EVENT_PRESETS}


def preset_by_key(key: str) -> EventPreset:
    """Найти пресет по ключу.

    Raises
    ------
    KeyError
        Если пресета с таким ключом нет — опечатка в вызывающем коде, а не
        повод молча вернуть нейтральное событие.
    """
    return _PRESETS_BY_KEY[key]
