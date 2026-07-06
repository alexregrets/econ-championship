"""End-to-end tests for the round service and team simulation.

Exercises the full flow the dev-shell relies on — seed, simulate, score — against
an in-memory database, with no Telegram bot and no LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from db import models  # noqa: F401  (registers tables)
from db import repositories as repo
from db.enums import RoundStatus
from devshell.seed import seed
from devshell.simulate_team import (
    simulate_all_teams_nash,
    simulate_all_teams_random,
    submit_decision,
)
from services.round_service import close_round, compute_round_results, open_round


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def test_seed_creates_seven_teams_and_open_round(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    assert len(summary.team_ids) == 7
    assert summary.student_count == 21
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN


async def test_nash_simulation_yields_equal_profits(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)

    results = await compute_round_results(session, summary.round_id)
    profits = [r.profit for r in results.values()]
    first = profits[0]
    assert all(p == pytest.approx(first) for p in profits)
    assert first > 0


async def test_compute_persists_results_and_updates_standings(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    results = await compute_round_results(session, summary.round_id)

    # Each decision now has a persisted result, and team profit moved.
    for team_id in summary.team_ids:
        decision = await repo.get_decision(
            session, team_id=team_id, round_id=summary.round_id
        )
        assert decision is not None and decision.id is not None
        stored = await repo.get_result_for_decision(session, decision.id)
        assert stored is not None
        assert stored.profit == pytest.approx(results[str(team_id)].profit)

        team = await repo.get_team(session, team_id)
        assert team is not None
        assert team.cumulative_profit == pytest.approx(stored.profit)


async def test_close_round_locks_it(session: AsyncSession) -> None:
    summary = await seed(session)
    await simulate_all_teams_random(session, summary.round_id, seed=1)
    await close_round(session, summary.round_id)

    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.CLOSED


async def test_scoring_round_without_decisions_raises(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    with pytest.raises(ValueError):
        await compute_round_results(session, summary.round_id)


async def test_open_round_helper_sets_status(session: AsyncSession) -> None:
    summary = await seed(session)
    # Force back to draft, then reopen via the service.
    await repo.set_round_status(
        session, round_id=summary.round_id, status=RoundStatus.DRAFT
    )
    await open_round(session, summary.round_id)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN


async def test_submit_decision_revision(session: AsyncSession) -> None:
    summary = await seed(session)
    team_id = summary.team_ids[0]
    await submit_decision(
        session, team_id=team_id, round_id=summary.round_id, quantity=5.0
    )
    await submit_decision(
        session, team_id=team_id, round_id=summary.round_id, quantity=9.0
    )
    decisions = await repo.list_decisions_for_round(session, summary.round_id)
    assert len(decisions) == 1
    assert decisions[0].quantity == 9.0
