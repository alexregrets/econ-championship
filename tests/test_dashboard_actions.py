"""Тесты действий дашборда (dashboard/actions.py).

Тем же стилем, что tests/test_round_service.py: in-memory база, никакого
Streamlit и никакой сети — проверяем ровно ту логику, которую страница
вызывает по кнопкам, с конкретными числовыми проверками.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from dashboard.actions import (
    close_round_with_results,
    create_and_open_round,
    next_round_number,
    results_table,
    submit_manual_decision,
)
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db.enums import Method, RoundStatus


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _make_round(session: AsyncSession) -> int:
    """Создать и открыть стандартный раунд-«нефть» для тестов, вернуть id."""
    round_ = await create_and_open_round(
        session,
        number=1,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="Нефть РФ 2013",
    )
    assert round_.id is not None
    return round_.id


async def test_next_round_number_starts_at_one(session: AsyncSession) -> None:
    assert await next_round_number(session) == 1


async def test_next_round_number_increments(session: AsyncSession) -> None:
    await _make_round(session)
    assert await next_round_number(session) == 2


async def test_create_and_open_round_is_open_and_ols_simple(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN
    assert round_.method is Method.OLS_SIMPLE
    assert round_.market_a == 100.0


async def test_submit_manual_decision_creates_and_revises(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="Команда А", company_name="Роснефть")
    assert team.id is not None

    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=20.0, reasoning="v1"
    )
    # Повторная отправка перезаписывает, дубликата не появляется.
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=30.0, reasoning="v2"
    )
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert len(decisions) == 1
    assert decisions[0].quantity == 30.0
    assert decisions[0].reasoning == "v2"


async def test_submit_manual_decision_rejects_closed_round(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="Команда А", company_name="Роснефть")
    assert team.id is not None
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.CLOSED
    )
    with pytest.raises(ValueError):
        await submit_manual_decision(
            session, team_id=team.id, round_id=round_id, quantity=5.0, reasoning=""
        )


async def test_close_round_with_results_full_flow(session: AsyncSession) -> None:
    """Дуополия с известным ответом: q1=30, q2=20 → P=50, прибыли 1200 и 800."""
    round_id = await _make_round(session)
    team_a = await repo.create_team(session, name="А", company_name="Роснефть")
    team_b = await repo.create_team(session, name="Б", company_name="Газпром")
    assert team_a.id is not None and team_b.id is not None

    await submit_manual_decision(
        session, team_id=team_a.id, round_id=round_id, quantity=30.0, reasoning="x"
    )
    await submit_manual_decision(
        session, team_id=team_b.id, round_id=round_id, quantity=20.0, reasoning="y"
    )

    rows = await close_round_with_results(session, round_id)

    # Раунд закрыт, решения больше не принимаются.
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.CLOSED

    # P = 100 - 1*(30+20) = 50; прибыль = (P - mc) * q.
    assert len(rows) == 2
    assert rows[0].team_name == "А"  # сортировка по прибыли: 1200 сверху
    assert rows[0].price == pytest.approx(50.0)
    assert rows[0].market_score == pytest.approx((50.0 - 10.0) * 30.0)
    assert rows[1].market_score == pytest.approx((50.0 - 10.0) * 20.0)
    # LLM-грейдинг к сервису не подключён — rubric_score дефолтный.
    assert rows[0].rubric_score == 0.0


async def test_results_table_skips_unscored_decisions(
    session: AsyncSession,
) -> None:
    round_id = await _make_round(session)
    team = await repo.create_team(session, name="А", company_name="Роснефть")
    assert team.id is not None
    await submit_manual_decision(
        session, team_id=team.id, round_id=round_id, quantity=10.0, reasoning=""
    )
    # Раунд не закрывали — Result ещё не посчитан, таблица должна быть пустой.
    assert await results_table(session, round_id) == []
