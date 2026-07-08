"""Тесты сервисного слоя бота и его репозиториев (без aiogram и без сети).

Минимум из постановки задачи:
- /join создаёт связку telegram_user_id -> team_id;
- /submit создаёт Decision через существующий общий слой;
- /submit в закрытый раунд получает явную ошибку и не создаёт Decision.

Плюс контракт кодов команд: генерация идемпотентна, формат читаемый,
поиск регистронезависимый.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from bot.service import join_team, submit_decision, team_status

# Импорт моделей регистрирует таблицы (включая TeamJoinCode) в metadata.
from db import (
    bot_models,  # noqa: F401
    models,  # noqa: F401
)
from db import bot_repositories as bot_repo
from db import repositories as repo
from db.enums import Method, RoundStatus
from db.models import Team


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Свежая in-memory база на каждый тест — как в tests/test_repositories.py."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _seed_team(session: AsyncSession) -> tuple[Team, str]:
    """Создать команду и вернуть её вместе с кодом вступления."""
    team = await repo.create_team(
        session, name="Команда 1", company_name="Роснефть"
    )
    assert team.id is not None
    codes = await bot_repo.ensure_join_codes(session)
    return team, codes[team.id]


async def _seed_round(
    session: AsyncSession, *, status: RoundStatus = RoundStatus.OPEN
) -> int:
    """Создать раунд в заданном статусе и вернуть его id."""
    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        status=status,
    )
    assert round_.id is not None
    return round_.id


# --------------------------------------------------------------------------- #
# Коды команд
# --------------------------------------------------------------------------- #


async def test_ensure_join_codes_creates_one_per_team(
    session: AsyncSession,
) -> None:
    a = await repo.create_team(session, name="A", company_name="X")
    b = await repo.create_team(session, name="B", company_name="Y")
    assert a.id is not None and b.id is not None

    codes = await bot_repo.ensure_join_codes(session)

    assert set(codes) == {a.id, b.id}
    assert len(set(codes.values())) == 2  # коды уникальны
    for code in codes.values():
        assert len(code) == bot_repo.CODE_LENGTH
        assert all(ch in bot_repo.CODE_ALPHABET for ch in code)


async def test_ensure_join_codes_is_idempotent(session: AsyncSession) -> None:
    await repo.create_team(session, name="A", company_name="X")
    first = await bot_repo.ensure_join_codes(session)
    second = await bot_repo.ensure_join_codes(session)
    # Розданные командам коды не перегенерируются.
    assert first == second


async def test_get_team_by_join_code_is_case_insensitive(
    session: AsyncSession,
) -> None:
    team, code = await _seed_team(session)
    found = await bot_repo.get_team_by_join_code(session, code.lower())
    assert found is not None
    assert found.id == team.id


async def test_get_team_by_unknown_code_returns_none(
    session: AsyncSession,
) -> None:
    await _seed_team(session)
    assert await bot_repo.get_team_by_join_code(session, "NOPE99") is None


# --------------------------------------------------------------------------- #
# /join: связка telegram_user_id -> team_id
# --------------------------------------------------------------------------- #


async def test_join_creates_student_bound_to_team(session: AsyncSession) -> None:
    team, code = await _seed_team(session)

    outcome = await join_team(
        session, telegram_id=555, full_name="Стас Студентов", code=code
    )

    assert outcome.created_student is True
    assert outcome.team.id == team.id
    student = await repo.get_student_by_telegram_id(session, 555)
    assert student is not None
    assert student.team_id == team.id
    assert student.role is None  # бот ролей не раздаёт


async def test_join_with_unknown_code_raises(session: AsyncSession) -> None:
    await _seed_team(session)
    with pytest.raises(ValueError, match="не найден"):
        await join_team(
            session, telegram_id=555, full_name="Стас", code="WRONG9"
        )
    # Никого не создали.
    assert await repo.get_student_by_telegram_id(session, 555) is None


async def test_rejoin_switches_team_without_duplicating_student(
    session: AsyncSession,
) -> None:
    team_a, code_a = await _seed_team(session)
    team_b = await repo.create_team(session, name="Команда 2", company_name="Лукойл")
    assert team_b.id is not None
    codes = await bot_repo.ensure_join_codes(session)

    await join_team(session, telegram_id=7, full_name="Петя", code=code_a)
    outcome = await join_team(
        session, telegram_id=7, full_name="Петя", code=codes[team_b.id]
    )

    assert outcome.created_student is False
    student = await repo.get_student_by_telegram_id(session, 7)
    assert student is not None
    assert student.team_id == team_b.id


# --------------------------------------------------------------------------- #
# /submit: решение уходит существующим слоем в открытый раунд
# --------------------------------------------------------------------------- #


async def test_submit_creates_decision_in_open_round(
    session: AsyncSession,
) -> None:
    team, code = await _seed_team(session)
    round_id = await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    outcome = await submit_decision(
        session, telegram_id=555, quantity=120.0, reasoning="Оценили спрос."
    )

    assert outcome.replaced is False
    assert outcome.round.id == round_id
    assert team.id is not None
    saved = await repo.get_decision(session, team_id=team.id, round_id=round_id)
    assert saved is not None
    assert saved.quantity == pytest.approx(120.0)
    assert saved.reasoning == "Оценили спрос."


async def test_resubmit_replaces_decision(session: AsyncSession) -> None:
    _team, code = await _seed_team(session)
    round_id = await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    await submit_decision(
        session, telegram_id=555, quantity=120.0, reasoning="Первая версия."
    )
    outcome = await submit_decision(
        session, telegram_id=555, quantity=100.0, reasoning="Передумали."
    )

    assert outcome.replaced is True
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert len(decisions) == 1  # upsert, не дубль
    assert decisions[0].quantity == pytest.approx(100.0)


async def test_submit_without_join_raises(session: AsyncSession) -> None:
    await _seed_round(session)
    with pytest.raises(ValueError, match="/join"):
        await submit_decision(
            session, telegram_id=999, quantity=10.0, reasoning="Текст."
        )


async def test_submit_into_closed_round_raises_and_creates_nothing(
    session: AsyncSession,
) -> None:
    """Требование постановки: явная ошибка, Decision не появляется."""
    _team, code = await _seed_team(session)
    round_id = await _seed_round(session, status=RoundStatus.CLOSED)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    with pytest.raises(ValueError, match="нет открытого раунда"):
        await submit_decision(
            session, telegram_id=555, quantity=120.0, reasoning="Поздно."
        )

    assert await repo.list_decisions_for_round(session, round_id) == []


async def test_submit_after_round_closes_raises(session: AsyncSession) -> None:
    """Раунд был открыт, но закрылся до /submit — та же явная ошибка."""
    _team, code = await _seed_team(session)
    round_id = await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.CLOSED
    )

    with pytest.raises(ValueError, match="нет открытого раунда"):
        await submit_decision(
            session, telegram_id=555, quantity=120.0, reasoning="Поздно."
        )
    assert await repo.list_decisions_for_round(session, round_id) == []


@pytest.mark.parametrize("bad_quantity", [-1.0, float("inf"), float("nan")])
async def test_submit_invalid_quantity_raises(
    session: AsyncSession, bad_quantity: float
) -> None:
    _team, code = await _seed_team(session)
    await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    with pytest.raises(ValueError, match="Объём"):
        await submit_decision(
            session, telegram_id=555, quantity=bad_quantity, reasoning="Текст."
        )


async def test_submit_empty_reasoning_raises(session: AsyncSession) -> None:
    _team, code = await _seed_team(session)
    await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    with pytest.raises(ValueError, match="обоснование"):
        await submit_decision(
            session, telegram_id=555, quantity=120.0, reasoning="   "
        )


# --------------------------------------------------------------------------- #
# /status
# --------------------------------------------------------------------------- #


async def test_status_requires_join(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="/join"):
        await team_status(session, telegram_id=1)


async def test_status_without_open_round(session: AsyncSession) -> None:
    _team, code = await _seed_team(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)

    status = await team_status(session, telegram_id=555)

    assert status.open_round is None
    assert status.decision is None


async def test_status_shows_submitted_decision(session: AsyncSession) -> None:
    _team, code = await _seed_team(session)
    await _seed_round(session)
    await join_team(session, telegram_id=555, full_name="Стас", code=code)
    await submit_decision(
        session, telegram_id=555, quantity=120.0, reasoning="Обоснование."
    )

    status = await team_status(session, telegram_id=555)

    assert status.open_round is not None
    assert status.decision is not None
    assert status.decision.quantity == pytest.approx(120.0)
