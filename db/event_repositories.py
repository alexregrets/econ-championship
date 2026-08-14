"""Repository-функции рыночных событий — тем же паттерном, что db/repositories.py.

Каждая функция берёт AsyncSession, делает одну операцию и коммитит. Экономики
здесь нет: как шок влияет на рынок, решает :mod:`core.market_events`.
"""

from __future__ import annotations

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.enums import EventKind
from db.event_models import MarketEvent

__all__ = [
    "create_market_event",
    "get_market_event",
    "list_events_for_round",
    "delete_market_event",
]


async def create_market_event(
    session: AsyncSession,
    *,
    round_id: int,
    kind: EventKind,
    magnitude: float,
    headline: str,
    description: str = "",
    revealed: bool = True,
    preset_key: str = "",
) -> MarketEvent:
    """Завести событие для раунда.

    ``revealed=True`` (по умолчанию) — команды видят событие до подачи решения;
    ``False`` — шок остаётся сюрпризом до закрытия раунда. На расчёт рынка
    флаг не влияет: событие действует в любом случае.
    """
    event = MarketEvent(
        round_id=round_id,
        kind=kind,
        magnitude=magnitude,
        headline=headline,
        description=description,
        revealed=revealed,
        preset_key=preset_key,
    )
    session.add(event)
    await session.commit()
    await session.refresh(event)
    return event


async def get_market_event(
    session: AsyncSession, event_id: int
) -> MarketEvent | None:
    """Найти событие по id."""
    return await session.get(MarketEvent, event_id)


async def list_events_for_round(
    session: AsyncSession, round_id: int
) -> list[MarketEvent]:
    """Все события раунда в порядке заведения.

    Порядок нужен только для отображения: на рынок события действуют
    мультипликативно и результат от порядка не зависит.
    """
    result = await session.exec(
        select(MarketEvent)
        .where(MarketEvent.round_id == round_id)
        .order_by(MarketEvent.id)  # type: ignore[arg-type]
    )
    return list(result.all())


async def delete_market_event(session: AsyncSession, event_id: int) -> None:
    """Удалить событие.

    Профессор заводит события до открытия раунда и может ошибиться; после
    закрытия раунда удаление уже ничего не меняет в сохранённых Result —
    проверку статуса делает слой действий дашборда, не репозиторий.

    Raises
    ------
    ValueError
        Если события с таким id нет.
    """
    event = await session.get(MarketEvent, event_id)
    if event is None:
        raise ValueError(f"market event {event_id} not found")
    await session.delete(event)
    await session.commit()
