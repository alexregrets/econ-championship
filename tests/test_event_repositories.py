"""Тесты репозитория рыночных событий: запись, чтение по раунду, удаление.

Как и tests/test_repositories.py — против настоящей in-memory SQLite, не мока.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from db import event_repositories as event_repo
from db import repositories as repo

# Импорт моделей регистрирует таблицы в SQLModel.metadata.
from db import event_models  # noqa: F401  # isort: skip
from db.enums import EventKind, Method
from db.models import Round


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Сессия на свежей in-memory базе — своя на каждый тест."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def round_(session: AsyncSession) -> Round:
    """Открытый раунд на generic-рынке."""
    return await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )


async def test_create_and_list_events(session: AsyncSession, round_: Round) -> None:
    assert round_.id is not None
    await event_repo.create_market_event(
        session,
        round_id=round_.id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.25,
        headline="ОПЕК не сокращает добычу",
        description="Картель держит долю рынка.",
        preset_key="opec_no_cut",
    )
    await event_repo.create_market_event(
        session,
        round_id=round_.id,
        kind=EventKind.COST_SHOCK,
        magnitude=0.18,
        headline="Санкции",
    )

    events = await event_repo.list_events_for_round(session, round_.id)

    assert [e.kind for e in events] == [EventKind.DEMAND_SHIFT, EventKind.COST_SHOCK]
    assert events[0].preset_key == "opec_no_cut"
    assert events[0].revealed is True  # по умолчанию событие опубликовано
    assert events[1].description == ""


async def test_events_are_scoped_to_their_round(
    session: AsyncSession, round_: Round
) -> None:
    """Событие одного раунда не подмешивается в другой."""
    assert round_.id is not None
    other = await repo.create_round(
        session,
        number=2,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )
    assert other.id is not None
    await event_repo.create_market_event(
        session,
        round_id=round_.id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.1,
        headline="Только первый раунд",
    )

    assert len(await event_repo.list_events_for_round(session, round_.id)) == 1
    assert await event_repo.list_events_for_round(session, other.id) == []


async def test_hidden_event_is_stored_as_hidden(
    session: AsyncSession, round_: Round
) -> None:
    """Сюрприз-событие сохраняет revealed=False."""
    assert round_.id is not None
    event = await event_repo.create_market_event(
        session,
        round_id=round_.id,
        kind=EventKind.COST_SHOCK,
        magnitude=0.1,
        headline="Скрытый налоговый манёвр",
        revealed=False,
    )

    assert event.revealed is False


async def test_delete_removes_the_event(session: AsyncSession, round_: Round) -> None:
    assert round_.id is not None
    event = await event_repo.create_market_event(
        session,
        round_id=round_.id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.1,
        headline="Ошибка ввода",
    )
    assert event.id is not None

    await event_repo.delete_market_event(session, event.id)

    assert await event_repo.list_events_for_round(session, round_.id) == []
    assert await event_repo.get_market_event(session, event.id) is None


async def test_deleting_unknown_event_raises(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="not found"):
        await event_repo.delete_market_event(session, 999)
