"""Тесты aiogram-хендлеров через мок Message — без реального Telegram API.

Message подменяется ``AsyncMock(spec=Message)`` с настоящим ``aiogram.types.User``
внутри; ``get_session_ctx`` в модуле хендлеров патчится на фабрику сессий
in-memory базы (StaticPool — все сессии делят одно соединение, иначе каждая
получила бы собственную пустую :memory:).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.filters import CommandObject
from aiogram.types import Message, User
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from bot import handlers
from db import (
    bot_models,  # noqa: F401  (регистрация таблиц в metadata)
    models,  # noqa: F401
)
from db import bot_repositories as bot_repo
from db import repositories as repo
from db.enums import Method, RoundStatus

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@pytest_asyncio.fixture
async def session_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[SessionFactory]:
    """Дать фабрику сессий одной in-memory базы и подсунуть её хендлерам."""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    @asynccontextmanager
    async def ctx() -> AsyncIterator[AsyncSession]:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(handlers, "get_session_ctx", ctx)
    yield ctx
    await engine.dispose()


def make_message(user_id: int = 555) -> AsyncMock:
    """Собрать мок сообщения от настоящего aiogram-пользователя.

    ``Message.answer`` в aiogram 3 — синхронный метод, возвращающий awaitable,
    поэтому spec делает его MagicMock; подменяем на AsyncMock явно, чтобы
    ``await message.answer(...)`` в хендлерах работал.
    """
    message = AsyncMock(spec=Message)
    message.from_user = User(id=user_id, is_bot=False, first_name="Стас")
    message.answer = AsyncMock()
    return message


def last_reply(message: AsyncMock) -> str:
    """Текст последнего ответа бота."""
    assert message.answer.await_count >= 1, "бот не ответил"
    call = message.answer.await_args
    assert call is not None
    return cast(str, call.args[0])


async def _seed_team_with_code(ctx: SessionFactory) -> str:
    """Создать команду и вернуть её код вступления."""
    async with ctx() as session:
        team = await repo.create_team(
            session, name="Команда 1", company_name="Роснефть"
        )
        assert team.id is not None
        codes = await bot_repo.ensure_join_codes(session)
        return codes[team.id]


async def _seed_open_round(ctx: SessionFactory) -> None:
    async with ctx() as session:
        await repo.create_round(
            session,
            number=1,
            method=Method.OLS_SIMPLE,
            difficulty=1,
            market_a=100.0,
            market_b=1.0,
            market_mc=10.0,
            status=RoundStatus.OPEN,
        )


def _command(name: str, args: str | None) -> CommandObject:
    return CommandObject(command=name, args=args)


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


async def test_start_lists_commands(session_ctx: SessionFactory) -> None:
    message = make_message()
    await handlers.cmd_start(cast(Message, message))
    text = last_reply(message)
    assert "/join" in text and "/submit" in text and "/status" in text


# --------------------------------------------------------------------------- #
# /join
# --------------------------------------------------------------------------- #


async def test_join_without_args_shows_usage(session_ctx: SessionFactory) -> None:
    message = make_message()
    await handlers.cmd_join(cast(Message, message), _command("join", None))
    assert "Использование" in last_reply(message)


async def test_join_happy_path_binds_student(session_ctx: SessionFactory) -> None:
    code = await _seed_team_with_code(session_ctx)
    message = make_message(user_id=777)

    await handlers.cmd_join(cast(Message, message), _command("join", code))

    assert "Команда 1" in last_reply(message)
    async with session_ctx() as session:
        student = await repo.get_student_by_telegram_id(session, 777)
        assert student is not None
        assert student.team_id is not None


async def test_join_with_wrong_code_reports_error(
    session_ctx: SessionFactory,
) -> None:
    await _seed_team_with_code(session_ctx)
    message = make_message()
    await handlers.cmd_join(cast(Message, message), _command("join", "WRONG9"))
    assert "не найден" in last_reply(message)


# --------------------------------------------------------------------------- #
# /submit
# --------------------------------------------------------------------------- #


async def test_submit_without_args_shows_usage(
    session_ctx: SessionFactory,
) -> None:
    message = make_message()
    await handlers.cmd_submit(cast(Message, message), _command("submit", None))
    assert "Использование" in last_reply(message)


async def test_submit_without_reasoning_shows_usage(
    session_ctx: SessionFactory,
) -> None:
    message = make_message()
    await handlers.cmd_submit(cast(Message, message), _command("submit", "120"))
    assert "Использование" in last_reply(message)


async def test_submit_with_non_numeric_quantity_explains(
    session_ctx: SessionFactory,
) -> None:
    message = make_message()
    await handlers.cmd_submit(
        cast(Message, message), _command("submit", "сто двадцать тонн нефти")
    )
    assert "нужно число" in last_reply(message)


async def test_submit_happy_path_stores_decision(
    session_ctx: SessionFactory,
) -> None:
    code = await _seed_team_with_code(session_ctx)
    await _seed_open_round(session_ctx)
    message = make_message(user_id=777)
    await handlers.cmd_join(cast(Message, message), _command("join", code))

    # Запятая как десятичный разделитель — студенты пишут и так.
    await handlers.cmd_submit(
        cast(Message, message),
        _command("submit", "120,5 Оценили спрос по цене Urals."),
    )

    assert "Раунд №1" in last_reply(message)
    async with session_ctx() as session:
        decisions = await repo.list_decisions_for_round(session, 1)
        assert len(decisions) == 1
        assert decisions[0].quantity == 120.5
        assert decisions[0].reasoning == "Оценили спрос по цене Urals."


async def test_submit_without_open_round_reports_error(
    session_ctx: SessionFactory,
) -> None:
    code = await _seed_team_with_code(session_ctx)
    message = make_message(user_id=777)
    await handlers.cmd_join(cast(Message, message), _command("join", code))

    await handlers.cmd_submit(
        cast(Message, message), _command("submit", "120 Обоснование.")
    )

    assert "нет открытого раунда" in last_reply(message)


# --------------------------------------------------------------------------- #
# /status
# --------------------------------------------------------------------------- #


async def test_status_before_join_points_to_join(
    session_ctx: SessionFactory,
) -> None:
    message = make_message()
    await handlers.cmd_status(cast(Message, message))
    assert "/join" in last_reply(message)


async def test_status_after_submit_shows_decision(
    session_ctx: SessionFactory,
) -> None:
    code = await _seed_team_with_code(session_ctx)
    await _seed_open_round(session_ctx)
    message = make_message(user_id=777)
    await handlers.cmd_join(cast(Message, message), _command("join", code))
    await handlers.cmd_submit(
        cast(Message, message), _command("submit", "120 Обоснование.")
    )

    await handlers.cmd_status(cast(Message, message))

    text = last_reply(message)
    assert "Раунд" in text or "раунд" in text
    assert "120" in text
