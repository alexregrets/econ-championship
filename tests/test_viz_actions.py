"""Тесты read-only действий визуализации (превью, равновесие, история).

Числа проверяются против движков напрямую: действия ничего не считают сами,
только собирают данные для чартов — расхождение с движком было бы багом
сборки, не экономики.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, nash_equilibrium
from dashboard.actions import (
    equilibrium_comparison,
    rounds_history,
    scenario_dataset,
)
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db.enums import EngineMode
from devshell.role_seed import (
    FULL_COST_2013_USD_PER_TON,
    OIL_PRODUCTION_2013_MLN_T,
    URALS_PRICE_2013_USD_PER_TON,
    generate_role_views,
    seed_oil_2013,
)
from devshell.seed import seed
from devshell.simulate_team import simulate_all_teams_nash
from services.round_service import close_round


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


# --------------------------------------------------------------------------- #
# scenario_dataset
# --------------------------------------------------------------------------- #


async def test_scenario_dataset_for_oil_round(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    dataset = await scenario_dataset(session, summary.round_id)
    assert dataset is not None
    assert dataset.productions == OIL_PRODUCTION_2013_MLN_T
    assert dataset.observed_price_per_ton == pytest.approx(
        URALS_PRICE_2013_USD_PER_TON
    )
    assert dataset.industry_cost_per_ton == pytest.approx(FULL_COST_2013_USD_PER_TON)
    assert dataset.revenues["Роснефть"] == pytest.approx(
        OIL_PRODUCTION_2013_MLN_T["Роснефть"] * URALS_PRICE_2013_USD_PER_TON
    )


async def test_scenario_dataset_none_for_generic_round(
    session: AsyncSession,
) -> None:
    """Generic-сидер (Сбербанк, Яндекс...) — датасета нефти нет, превью нет."""
    summary = await seed(session)
    assert await scenario_dataset(session, summary.round_id) is None


# --------------------------------------------------------------------------- #
# equilibrium_comparison
# --------------------------------------------------------------------------- #


async def test_comparison_none_for_open_round(session: AsyncSession) -> None:
    summary = await seed(session)
    assert await equilibrium_comparison(session, summary.round_id) is None


async def test_comparison_symmetric_round(session: AsyncSession) -> None:
    """Симметричный раунд: все команды играли Нэш → факт == равновесие."""
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    await close_round(session, summary.round_id)

    comparison = await equilibrium_comparison(session, summary.round_id)
    assert comparison is not None
    assert comparison.engine_mode is EngineMode.SYMMETRIC

    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    params = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )
    q_star = nash_equilibrium(len(summary.team_ids), params)
    assert comparison.equilibrium_total_quantity == pytest.approx(
        q_star * len(summary.team_ids)
    )
    assert comparison.actual_total_quantity == pytest.approx(
        comparison.equilibrium_total_quantity
    )
    assert comparison.actual_price == pytest.approx(comparison.equilibrium_price)
    for row in comparison.teams:
        assert row.equilibrium_quantity == pytest.approx(q_star)
        assert row.actual_quantity == pytest.approx(q_star)


async def test_comparison_asymmetric_round_uses_implied_costs(
    session: AsyncSession,
) -> None:
    """Асимметричный раунд: равновесный q* команды = её реальная добыча 2013."""
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    await generate_role_views(session, summary.round_id)
    teams = await repo.list_teams(session)
    for team in teams:
        assert team.id is not None
        await repo.upsert_decision(
            session,
            team_id=team.id,
            round_id=summary.round_id,
            quantity=100.0,  # намеренно не равновесие — факт должен отличаться
            reasoning="",
        )
    await close_round(session, summary.round_id)

    comparison = await equilibrium_comparison(session, summary.round_id)
    assert comparison is not None
    assert comparison.engine_mode is EngineMode.ASYMMETRIC
    by_label = {row.team_label: row for row in comparison.teams}
    for team in teams:
        row = by_label[f"{team.name} ({team.company_name})"]
        assert row.actual_quantity == pytest.approx(100.0)
        assert row.equilibrium_quantity == pytest.approx(
            OIL_PRODUCTION_2013_MLN_T[team.company_name], rel=1e-9
        )
    assert comparison.equilibrium_price == pytest.approx(
        URALS_PRICE_2013_USD_PER_TON, rel=1e-9
    )


# --------------------------------------------------------------------------- #
# rounds_history
# --------------------------------------------------------------------------- #


async def test_history_empty_without_closed_rounds(session: AsyncSession) -> None:
    await seed(session)
    assert await rounds_history(session) == []


async def test_history_one_closed_round_is_one_row(session: AsyncSession) -> None:
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    results = await close_round(session, summary.round_id)

    history = await rounds_history(session)
    assert len(history) == 1
    row = history[0]
    assert row.number == 1
    assert row.decisions == len(summary.team_ids)
    expected_avg = sum(r.profit for r in results.values()) / len(results)
    assert row.avg_profit == pytest.approx(expected_avg)
    assert row.price == pytest.approx(next(iter(results.values())).price)
