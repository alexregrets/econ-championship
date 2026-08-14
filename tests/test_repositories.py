"""Tests for the db layer: schema creation, repositories, and constraints.

Each test runs against a fresh in-memory SQLite database so they're isolated and
fast, exercising the real async engine + SQLModel mapping (not mocks).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

# Importing models registers the tables on SQLModel.metadata.
from db import models  # noqa: F401
from db import repositories as repo
from db.enums import Method, Role, RoundStatus


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Yield a session bound to a fresh in-memory database per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with SQLModelAsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def test_create_and_fetch_student(session: AsyncSession) -> None:
    student = await repo.create_student(
        session, telegram_id=42, full_name="Иван Иванов"
    )
    assert student.id is not None

    fetched = await repo.get_student_by_telegram_id(session, 42)
    assert fetched is not None
    assert fetched.full_name == "Иван Иванов"


async def test_unknown_telegram_id_returns_none(session: AsyncSession) -> None:
    assert await repo.get_student_by_telegram_id(session, 999) is None


async def test_assign_student_to_team(session: AsyncSession) -> None:
    team = await repo.create_team(session, name="Команда А", company_name="Газпром")
    student = await repo.create_student(session, telegram_id=7, full_name="Петя")
    assert team.id is not None and student.id is not None

    updated = await repo.assign_student_to_team(
        session, student_id=student.id, team_id=team.id, role=Role.FINANCIER
    )
    assert updated.team_id == team.id
    assert updated.role is Role.FINANCIER


async def test_assign_unknown_student_raises(session: AsyncSession) -> None:
    team = await repo.create_team(session, name="A", company_name="X")
    assert team.id is not None
    with pytest.raises(ValueError):
        await repo.assign_student_to_team(
            session, student_id=12345, team_id=team.id, role=Role.MARKETER
        )


async def test_leaderboard_orders_by_profit(session: AsyncSession) -> None:
    a = await repo.create_team(session, name="A", company_name="X")
    b = await repo.create_team(session, name="B", company_name="Y")
    assert a.id is not None and b.id is not None

    await repo.add_team_profit(session, team_id=a.id, delta=100.0)
    await repo.add_team_profit(session, team_id=b.id, delta=250.0)

    teams = await repo.list_teams(session)
    assert [t.name for t in teams] == ["B", "A"]


async def test_round_lifecycle_and_open_round(session: AsyncSession) -> None:
    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )
    assert round_.status is RoundStatus.DRAFT
    assert await repo.get_open_round(session) is None

    assert round_.id is not None
    await repo.set_round_status(
        session, round_id=round_.id, status=RoundStatus.OPEN
    )
    open_round = await repo.get_open_round(session)
    assert open_round is not None
    assert open_round.id == round_.id


async def test_upsert_decision_replaces_not_duplicates(
    session: AsyncSession,
) -> None:
    team = await repo.create_team(session, name="A", company_name="X")
    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )
    assert team.id is not None and round_.id is not None

    first = await repo.upsert_decision(
        session, team_id=team.id, round_id=round_.id, quantity=10.0, reasoning="v1"
    )
    second = await repo.upsert_decision(
        session, team_id=team.id, round_id=round_.id, quantity=12.0, reasoning="v2"
    )

    assert first.id == second.id  # same row updated, not a new one
    decisions = await repo.list_decisions_for_round(session, round_.id)
    assert len(decisions) == 1
    assert decisions[0].quantity == 12.0
    assert decisions[0].reasoning == "v2"


async def test_one_decision_per_team_round_constraint(
    session: AsyncSession,
) -> None:
    team = await repo.create_team(session, name="A", company_name="X")
    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )
    assert team.id is not None and round_.id is not None

    # Bypass the upsert helper to prove the DB constraint itself holds.
    session.add(
        models.Decision(
            team_id=team.id, round_id=round_.id, quantity=1.0, reasoning="a"
        )
    )
    await session.commit()
    session.add(
        models.Decision(
            team_id=team.id, round_id=round_.id, quantity=2.0, reasoning="b"
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_save_result_is_one_to_one(session: AsyncSession) -> None:
    team = await repo.create_team(session, name="A", company_name="X")
    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
    )
    assert team.id is not None and round_.id is not None
    decision = await repo.upsert_decision(
        session, team_id=team.id, round_id=round_.id, quantity=10.0, reasoning="x"
    )
    assert decision.id is not None

    await repo.save_result(
        session, decision_id=decision.id, price=90.0, profit=800.0
    )
    # Saving again updates the same result row rather than creating a second.
    updated = await repo.save_result(
        session,
        decision_id=decision.id,
        price=85.0,
        profit=750.0,
        rubric_score=0.75,
    )
    fetched = await repo.get_result_for_decision(session, decision.id)
    assert fetched is not None
    assert fetched.id == updated.id
    assert fetched.profit == 750.0
    assert fetched.rubric_score == 0.75


async def test_rubric_template_upsert_per_method(session: AsyncSession) -> None:
    await repo.upsert_rubric_template(
        session, method=Method.HETEROSCEDASTICITY, name="v1", criteria_json="[]"
    )
    await repo.upsert_rubric_template(
        session,
        method=Method.HETEROSCEDASTICITY,
        name="v2",
        criteria_json='[{"id": "x"}]',
    )
    template = await repo.get_rubric_for_method(
        session, Method.HETEROSCEDASTICITY
    )
    assert template is not None
    assert template.name == "v2"
    assert template.criteria_json == '[{"id": "x"}]'
