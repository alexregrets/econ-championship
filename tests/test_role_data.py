"""Тесты ролевых данных: модели, репозитории, сидер срезов, ролевая сессия.

In-memory база, без сети (LLM — фейк по протоколу StructuredLLM), числовые
проверки инвариантов согласованности из DECISIONS.md №9 и №12.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, nash_equilibrium
from db import (
    models,  # noqa: F401  (регистрирует базовые таблицы)
    role_models,  # noqa: F401  (регистрирует ролевые таблицы)
)
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import Role, RoundStatus
from devshell.role_seed import ROLE_COST_SHARES, build_role_slices, generate_role_views
from devshell.role_session import (
    commit_lead_decision,
    enter_role,
    lead_overview,
    submit_role_proposal,
    what_if_profit,
)
from devshell.seed import seed

PARAMS = MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


# --------------------------------------------------------------------------- #
# build_role_slices: чистая генерация срезов
# --------------------------------------------------------------------------- #


def test_cost_shares_constant_sums_to_one() -> None:
    assert sum(ROLE_COST_SHARES.values()) == pytest.approx(1.0)


def test_slices_cover_all_three_roles() -> None:
    slices = build_role_slices(PARAMS, n_teams=7, company_name="Роснефть")
    assert set(slices) == {Role.MARKETER, Role.SALES_ANALYST, Role.FINANCIER}


def test_slices_private_signals_derived_from_ground_truth() -> None:
    """7 команд, a=100, b=1, mc=10: q* = 90/8 = 11.25, конкуренты = 67.5."""
    slices = build_role_slices(PARAMS, n_teams=7, company_name="Роснефть")
    assert slices[Role.MARKETER].private_signal == pytest.approx(100.0)
    assert slices[Role.FINANCIER].private_signal == pytest.approx(10.0)
    assert slices[Role.SALES_ANALYST].private_signal == pytest.approx(11.25 * 6)


def test_slices_cost_shares_sum_to_one() -> None:
    slices = build_role_slices(PARAMS, n_teams=3, company_name="Газпром")
    assert sum(s.cost_share for s in slices.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# generate_role_views: сидер срезов против БД
# --------------------------------------------------------------------------- #


async def test_generate_creates_truth_and_three_views_per_team(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    views = await generate_role_views(session, summary.round_id)
    assert len(views) == 7 * 3  # 7 команд × 3 роли

    for team_id in summary.team_ids:
        truth = await role_repo.get_ground_truth(
            session, round_id=summary.round_id, team_id=team_id
        )
        assert truth is not None
        team_views = await role_repo.list_role_views_for_team(
            session, round_id=summary.round_id, team_id=team_id
        )
        assert len(team_views) == 3


async def test_generated_views_hold_shared_buffer_invariants(
    session: AsyncSession,
) -> None:
    """Общие поля одинаковы у всех ролей и равны ground truth; доли дают 1.0."""
    summary = await seed(session)
    await generate_role_views(session, summary.round_id)

    q_star = nash_equilibrium(7, PARAMS)  # сидер создаёт a=100, b=1, mc=10
    expected_total = q_star * 7
    expected_price = 100.0 - 1.0 * expected_total

    for team_id in summary.team_ids:
        truth = await role_repo.get_ground_truth(
            session, round_id=summary.round_id, team_id=team_id
        )
        assert truth is not None
        assert truth.ref_total_quantity == pytest.approx(expected_total)
        assert truth.observed_price == pytest.approx(expected_price)

        views = await role_repo.list_role_views_for_team(
            session, round_id=summary.round_id, team_id=team_id
        )
        for view in views:
            assert view.ref_total_quantity == pytest.approx(truth.ref_total_quantity)
            assert view.observed_price == pytest.approx(truth.observed_price)
        assert sum(v.cost_share for v in views) == pytest.approx(1.0)


async def test_generate_rejects_missing_round(session: AsyncSession) -> None:
    with pytest.raises(ValueError):
        await generate_role_views(session, round_id=999)


class _FakeNarrativeLLM:
    """Фейковый StructuredLLM: отдаёт фиксированные нарративы, сети нет."""

    async def structured_completion[T: BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        return response_model(
            marketer="брифинг маркетолога",
            sales_analyst="брифинг аналитика",
            financier="брифинг финансиста",
        )


async def test_generate_uses_llm_narratives_when_provided(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    views = await generate_role_views(
        session, summary.round_id, llm=_FakeNarrativeLLM()
    )
    narratives = {v.role: v.narrative for v in views[:3]}
    assert narratives[Role.MARKETER] == "брифинг маркетолога"
    assert narratives[Role.FINANCIER] == "брифинг финансиста"


# --------------------------------------------------------------------------- #
# Ролевая сессия: вход в роль, предложения, агрегация, what-if, commit
# --------------------------------------------------------------------------- #


async def test_enter_role_returns_only_own_slice(session: AsyncSession) -> None:
    summary = await seed(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    view = await enter_role(
        session, round_id=summary.round_id, team_id=team_id, role=Role.FINANCIER
    )
    assert view.role is Role.FINANCIER
    assert view.slice_.private_signal == pytest.approx(10.0)  # mc — сигнал финансиста
    assert view.my_proposal is None  # ещё ничего не предлагал


async def test_enter_role_fails_before_seeding_views(session: AsyncSession) -> None:
    summary = await seed(session)
    with pytest.raises(ValueError):
        await enter_role(
            session,
            round_id=summary.round_id,
            team_id=summary.team_ids[0],
            role=Role.MARKETER,
        )


async def test_proposals_flow_into_lead_overview(session: AsyncSession) -> None:
    summary = await seed(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=10.0,
    )
    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.FINANCIER,
        quantity=20.0,
    )

    overview = await lead_overview(
        session, round_id=summary.round_id, team_id=team_id
    )
    assert overview.proposals[Role.MARKETER] == 10.0
    assert overview.proposals[Role.FINANCIER] == 20.0
    # Равные веса → среднее.
    assert overview.aggregated_quantity == pytest.approx(15.0)


async def test_proposal_revision_keeps_single_row(session: AsyncSession) -> None:
    summary = await seed(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=10.0,
    )
    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=12.5,
        note="передумал",
    )
    inputs = await role_repo.list_role_inputs_for_team(
        session, round_id=summary.round_id, team_id=team_id
    )
    assert len(inputs) == 1
    assert inputs[0].quantity_proposal == 12.5
    assert inputs[0].note == "передумал"


async def test_proposal_rejected_when_round_not_open(session: AsyncSession) -> None:
    summary = await seed(session)
    await generate_role_views(session, summary.round_id)
    await repo.set_round_status(
        session, round_id=summary.round_id, status=RoundStatus.CLOSED
    )
    with pytest.raises(ValueError):
        await submit_role_proposal(
            session,
            round_id=summary.round_id,
            team_id=summary.team_ids[0],
            role=Role.MARKETER,
            quantity=5.0,
        )


def test_what_if_profit_exact_numbers() -> None:
    """Мой Q=30 против конкурентов (20): P=50, моя прибыль 1200."""
    price, profit = what_if_profit(PARAMS, my_quantity=30.0, other_quantities=[20.0])
    assert price == pytest.approx(50.0)
    assert profit == pytest.approx(1200.0)


async def test_commit_lead_decision_creates_single_team_decision(
    session: AsyncSession,
) -> None:
    summary = await seed(session)
    team_id = summary.team_ids[0]
    await commit_lead_decision(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        quantity=15.0,
        reasoning="lead: среднее предложений ролей",
    )
    decisions = await repo.list_decisions_for_round(session, summary.round_id)
    assert len(decisions) == 1
    assert decisions[0].team_id == team_id
    assert decisions[0].quantity == 15.0
