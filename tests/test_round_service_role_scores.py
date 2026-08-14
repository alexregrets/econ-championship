"""Wiring личных KPI ролей в закрытие раунда.

`core.role_kpi.compute_role_kpis` покрыт своими тестами как чистая функция —
здесь проверяется не арифметика, а то, что она вообще вызывается, что её
результат доезжает до `RoleScore`, и что закрытие раунда остаётся честным,
когда что-то идёт не так.

Гарантии:
- закрытие раунда пишет по три оценки на команду и берёт прогноз цены из
  ввода аналитика сбыта;
- пересчёт раунда обновляет оценки, а не задваивает их;
- команда без поданного прогноза помечается `has_input=False`, а не получает
  вид, будто прогноз был плохой;
- раунд, который не удалось посчитать, не оставляет за собой оценок и не
  переходит в CLOSED.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from db import event_repositories as event_repo
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import EventKind, Role, RoundStatus
from devshell.seed import seed
from devshell.simulate_team import simulate_all_teams_nash
from services.round_service import close_round, compute_round_results, score_roles


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _seeded_round(session: AsyncSession) -> int:
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    return summary.round_id


async def _submit_forecasts(session: AsyncSession, round_id: int) -> list[int]:
    """Прогноз цены от аналитика каждой команды. Вернуть id команд."""
    teams = await repo.list_teams(session)
    team_ids = []
    for offset, team in enumerate(teams):
        assert team.id is not None
        team_ids.append(team.id)
        await role_repo.upsert_role_input(
            session,
            round_id=round_id,
            team_id=team.id,
            role=Role.SALES_ANALYST,
            quantity_proposal=10.0,
            price_forecast=40.0 + offset,  # у каждой команды свой прогноз
        )
    return team_ids


# --------------------------------------------------------------------------- #
# Оценки появляются
# --------------------------------------------------------------------------- #


async def test_close_round_writes_three_scores_per_team(
    session: AsyncSession,
) -> None:
    round_id = await _seeded_round(session)
    team_ids = await _submit_forecasts(session, round_id)

    await close_round(session, round_id)

    scores = await role_repo.list_role_scores_for_round(session, round_id)
    assert len(scores) == 3 * len(team_ids)
    assert {s.role for s in scores} == {
        Role.MARKETER,
        Role.FINANCIER,
        Role.SALES_ANALYST,
    }


async def test_scores_stay_in_unit_interval(session: AsyncSession) -> None:
    round_id = await _seeded_round(session)
    await _submit_forecasts(session, round_id)
    await close_round(session, round_id)

    for score in await role_repo.list_role_scores_for_round(session, round_id):
        assert 0.0 <= score.kpi_normalized <= 1.0
        assert 0.0 <= score.team_component <= 1.0
        assert 0.0 <= score.total <= 1.0


async def test_best_in_round_normalizes_to_one(session: AsyncSession) -> None:
    """Нормировка идёт внутри роли: в каждой роли кто-то обязан быть лучшим."""
    round_id = await _seeded_round(session)
    await _submit_forecasts(session, round_id)
    await close_round(session, round_id)

    scores = await role_repo.list_role_scores_for_round(session, round_id)
    for role in (Role.MARKETER, Role.FINANCIER, Role.SALES_ANALYST):
        in_role = [s.kpi_normalized for s in scores if s.role is role]
        assert max(in_role) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Прогноз аналитика
# --------------------------------------------------------------------------- #


async def test_missing_forecast_is_marked_not_scored(session: AsyncSession) -> None:
    """Прогноза нет — has_input=False. Это не то же, что плохой прогноз."""
    round_id = await _seeded_round(session)
    await close_round(session, round_id)  # ни одного RoleInput не подано

    analysts = [
        s
        for s in await role_repo.list_role_scores_for_round(session, round_id)
        if s.role is Role.SALES_ANALYST
    ]
    assert analysts
    assert all(not s.has_input for s in analysts)

    # У остальных ролей своего ввода не требуется — они считаются всегда.
    others = [
        s
        for s in await role_repo.list_role_scores_for_round(session, round_id)
        if s.role is not Role.SALES_ANALYST
    ]
    assert all(s.has_input for s in others)


async def test_forecast_of_other_roles_is_ignored(session: AsyncSession) -> None:
    """price_forecast маркетолога не должен подменять прогноз аналитика."""
    round_id = await _seeded_round(session)
    teams = await repo.list_teams(session)
    team = teams[0]
    assert team.id is not None

    await role_repo.upsert_role_input(
        session,
        round_id=round_id,
        team_id=team.id,
        role=Role.MARKETER,
        quantity_proposal=10.0,
        price_forecast=999.0,  # роль, у которой прогноза быть не должно
    )
    await close_round(session, round_id)

    analyst = next(
        s
        for s in await role_repo.list_role_scores_for_round(session, round_id)
        if s.role is Role.SALES_ANALYST and s.team_id == team.id
    )
    assert not analyst.has_input


# --------------------------------------------------------------------------- #
# Пересчёт
# --------------------------------------------------------------------------- #


async def test_rescoring_updates_instead_of_duplicating(
    session: AsyncSession,
) -> None:
    round_id = await _seeded_round(session)
    team_ids = await _submit_forecasts(session, round_id)

    results = await compute_round_results(session, round_id)
    await score_roles(session, round_id, results)
    first = await role_repo.list_role_scores_for_round(session, round_id)

    await score_roles(session, round_id, results)
    second = await role_repo.list_role_scores_for_round(session, round_id)

    assert len(first) == len(second) == 3 * len(team_ids)
    assert {s.id for s in first} == {s.id for s in second}


# --------------------------------------------------------------------------- #
# Неудачное закрытие не оставляет следов
# --------------------------------------------------------------------------- #


async def test_unscorable_round_writes_no_scores_and_stays_open(
    session: AsyncSession,
) -> None:
    """Рынок сломан событием: ни оценок, ни перехода в CLOSED."""
    round_id = await _seeded_round(session)
    await _submit_forecasts(session, round_id)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.95,
        headline="Катастрофа спроса",
    )

    with pytest.raises(ValueError, match="нежизнеспособ"):
        await close_round(session, round_id)

    assert await role_repo.list_role_scores_for_round(session, round_id) == []
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN
