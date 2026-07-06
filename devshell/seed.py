"""Seed the database with realistic test data for development.

Creates seven teams (each with three students in the three roles) and one open
round, so the rest of the dev-shell has something to act on. Idempotency is not a
goal: call :func:`reset_and_seed` to start from a clean slate.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from db import repositories as repo
from db.enums import Method, Role, RoundStatus
from db.session import drop_all, get_session_ctx, init_db

__all__ = ["TEAM_COMPANIES", "seed", "reset_and_seed", "SeedSummary"]

# Seven teams of three — one company name each.
TEAM_COMPANIES: list[tuple[str, str]] = [
    ("Команда А", "Роснефть"),
    ("Команда Б", "Газпром"),
    ("Команда В", "Лукойл"),
    ("Команда Г", "Сбербанк"),
    ("Команда Д", "Яндекс"),
    ("Команда Е", "Норникель"),
    ("Команда Ж", "Магнит"),
]

_ROLES = (Role.MARKETER, Role.SALES_ANALYST, Role.FINANCIER)


@dataclass(frozen=True)
class SeedSummary:
    """What :func:`seed` created, for display in the dev-shell."""

    team_ids: list[int]
    student_count: int
    round_id: int


async def seed(session: AsyncSession) -> SeedSummary:
    """Create seven teams (3 students each) and one open round.

    Returns
    -------
    SeedSummary
        Ids and counts of the created entities.
    """
    team_ids: list[int] = []
    student_count = 0
    telegram_seq = 1000

    for index, (name, company) in enumerate(TEAM_COMPANIES):
        team = await repo.create_team(session, name=name, company_name=company)
        assert team.id is not None
        team_ids.append(team.id)
        for role in _ROLES:
            telegram_seq += 1
            student = await repo.create_student(
                session,
                telegram_id=telegram_seq,
                full_name=f"{company} {role.value} {index + 1}",
            )
            assert student.id is not None
            await repo.assign_student_to_team(
                session, student_id=student.id, team_id=team.id, role=role
            )
            student_count += 1

    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=100.0,
        market_b=1.0,
        market_mc=10.0,
        case_narrative="Тестовый раунд: парная регрессия спроса.",
        status=RoundStatus.OPEN,
    )
    assert round_.id is not None
    return SeedSummary(
        team_ids=team_ids, student_count=student_count, round_id=round_.id
    )


async def reset_and_seed() -> SeedSummary:
    """Drop all tables, recreate the schema, and seed fresh data.

    Convenience entry point for the dev-shell "clear + seed" action and for
    running the module directly.
    """
    await drop_all()
    await init_db()
    async with get_session_ctx() as session:
        return await seed(session)


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    summary = asyncio.run(reset_and_seed())
    print(
        f"Seeded {len(summary.team_ids)} teams, "
        f"{summary.student_count} students, round id={summary.round_id}"
    )
