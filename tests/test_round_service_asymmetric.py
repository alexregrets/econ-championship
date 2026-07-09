"""Wiring асимметричного движка в раунды (OPEN_QUESTIONS.md №7).

Гарантии:
- REGRESSION: раунд, созданный по умолчанию, — симметричный и считается
  байт-в-байт как раньше (существующие тесты round_service не менялись);
- хранение: generate_role_views пишет калиброванные implied costs в
  CompanyGroundTruth — ровно те значения, что закреплены тестами движка;
- асимметричный раунд: команды, играющие асимметричный Нэш по сохранённым
  издержкам, получают Q, равные реальным добычам 2013 (допуск 1% из
  постановки, фактический float-уровень 1e-9), профиты считаются по
  пофирменным c_i;
- отказы: асимметричный раунд без ground truth или без калиброванных
  издержек не закрывается (ValueError, без тихого фолбэка на market_mc).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, compute_cournot_round
from core.market_engine_asymmetric import asymmetric_nash_equilibrium
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import EngineMode, RoundStatus
from devshell.role_seed import (
    OIL_PRODUCTION_2013_MLN_T,
    URALS_PRICE_2013_USD_PER_TON,
    generate_role_views,
    implied_oil_2013_costs,
    seed_oil_2013,
)
from devshell.seed import seed
from devshell.simulate_team import simulate_all_teams_nash
from services.round_service import close_round, compute_round_results


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


# --------------------------------------------------------------------------- #
# REGRESSION: дефолт — симметричный движок, поведение не изменилось
# --------------------------------------------------------------------------- #


async def test_default_round_is_symmetric(session: AsyncSession) -> None:
    """Сидеры без engine_mode создают симметричные раунды — дефолт не сдвинут."""
    summary = await seed(session)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.engine_mode is EngineMode.SYMMETRIC


async def test_symmetric_round_scores_exactly_as_before(
    session: AsyncSession,
) -> None:
    """Закрытие дефолтного раунда == прямой вызов симметричного движка."""
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    decisions = await repo.list_decisions_for_round(session, summary.round_id)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None

    expected = compute_cournot_round(
        {str(d.team_id): d.quantity for d in decisions},
        MarketParameters(
            a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
        ),
    )
    results = await compute_round_results(session, summary.round_id)
    assert results == expected


async def test_oil_seeder_defaults_to_symmetric(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.engine_mode is EngineMode.SYMMETRIC


# --------------------------------------------------------------------------- #
# Хранение калиброванных издержек
# --------------------------------------------------------------------------- #


async def test_generate_role_views_persists_implied_costs(
    session: AsyncSession,
) -> None:
    """Ground truth хранит ровно те implied costs, что закреплены тестами движка."""
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)

    expected = implied_oil_2013_costs()
    truths = await role_repo.list_ground_truths_for_round(session, summary.round_id)
    assert len(truths) == len(OIL_PRODUCTION_2013_MLN_T)
    for truth in truths:
        team = await repo.get_team(session, truth.team_id)
        assert team is not None
        assert truth.implied_marginal_cost == pytest.approx(
            expected[team.company_name], rel=1e-12
        )


# --------------------------------------------------------------------------- #
# Асимметричный раунд: полный цикл
# --------------------------------------------------------------------------- #


async def test_asymmetric_round_reproduces_2013_shares(
    session: AsyncSession,
) -> None:
    """Команды играют асимметричный Нэш по сохранённым c_i → Q = добычи 2013.

    Издержки читаются из БД (а не из калибровочной функции), чтобы проверить
    именно wiring: хранение → равновесие → close_round → профиты по c_i.
    """
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    await generate_role_views(session, summary.round_id)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.engine_mode is EngineMode.ASYMMETRIC

    truths = await role_repo.list_ground_truths_for_round(session, summary.round_id)
    costs_by_team = {t.team_id: t.implied_marginal_cost for t in truths}
    team_ids = sorted(costs_by_team)
    stored_costs = [costs_by_team[tid] for tid in team_ids]
    assert all(c is not None for c in stored_costs)
    costs = [c for c in stored_costs if c is not None]

    equilibrium = asymmetric_nash_equilibrium(
        round_.market_a, round_.market_b, costs
    )
    for team_id, quantity in zip(team_ids, equilibrium.quantities, strict=True):
        await repo.upsert_decision(
            session,
            team_id=team_id,
            round_id=summary.round_id,
            quantity=quantity,
            reasoning="равновесие по своим издержкам",
        )

    results = await close_round(session, summary.round_id)

    closed = await repo.get_round(session, summary.round_id)
    assert closed is not None
    assert closed.status is RoundStatus.CLOSED

    # Равновесные Q команд = реальные добычи 2013 (допуск 1% из постановки,
    # фактическое совпадение — float-шум) и цена = цена Urals.
    productions_by_team: dict[int, float] = {}
    for team_id in team_ids:
        team = await repo.get_team(session, team_id)
        assert team is not None
        productions_by_team[team_id] = OIL_PRODUCTION_2013_MLN_T[team.company_name]

    for team_id, cost in zip(team_ids, costs, strict=True):
        result = results[str(team_id)]
        real_production = productions_by_team[team_id]
        assert result.quantity == pytest.approx(real_production, rel=0.01)
        assert result.quantity == pytest.approx(real_production, rel=1e-9)
        assert result.price == pytest.approx(URALS_PRICE_2013_USD_PER_TON, rel=1e-9)
        # Профит считан по пофирменным издержкам, а не по общему market_mc.
        assert result.profit == pytest.approx(
            (result.price - cost) * result.quantity, rel=1e-9
        )


async def test_asymmetric_and_symmetric_profits_differ(
    session: AsyncSession,
) -> None:
    """Асимметричный счёт реально другой: тем же решениям — другие профиты."""
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    await generate_role_views(session, summary.round_id)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None

    teams = await repo.list_teams(session)
    decisions: dict[str, float] = {}
    for team in teams:
        assert team.id is not None
        quantity = OIL_PRODUCTION_2013_MLN_T[team.company_name]
        decisions[str(team.id)] = quantity
        await repo.upsert_decision(
            session,
            team_id=team.id,
            round_id=summary.round_id,
            quantity=quantity,
            reasoning="фактическая добыча",
        )

    results = await compute_round_results(session, summary.round_id)
    symmetric = compute_cournot_round(
        decisions,
        MarketParameters(
            a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
        ),
    )
    # Цены совпадают (спрос тот же), профиты — нет (издержки пофирменные).
    for team_id, result in results.items():
        assert result.price == pytest.approx(symmetric[team_id].price)
    assert any(
        results[tid].profit != pytest.approx(symmetric[tid].profit)
        for tid in decisions
    )


# --------------------------------------------------------------------------- #
# Отказы конфигурации
# --------------------------------------------------------------------------- #


async def test_asymmetric_round_without_ground_truth_refuses_to_close(
    session: AsyncSession,
) -> None:
    """Без generate_role_views асимметричный раунд закрыть нельзя."""
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    team_id = summary.team_ids[0]
    await repo.upsert_decision(
        session, team_id=team_id, round_id=summary.round_id, quantity=100.0, reasoning=""
    )
    with pytest.raises(ValueError, match="calibrated"):
        await close_round(session, summary.round_id)
    # Раунд не закрыт — профессор может догенерировать срезы и закрыть снова.
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN


async def test_asymmetric_round_with_uncalibrated_truth_refuses_to_close(
    session: AsyncSession,
) -> None:
    """Ground truth без implied_marginal_cost — тоже отказ, не фолбэк на mc."""
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    team_id = summary.team_ids[0]
    await role_repo.create_ground_truth(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        demand_a=round_.market_a,
        demand_b=round_.market_b,
        marginal_cost=round_.market_mc,
        ref_total_quantity=1.0,
        observed_price=1.0,
        # implied_marginal_cost по умолчанию None — калибровки нет.
    )
    await repo.upsert_decision(
        session, team_id=team_id, round_id=summary.round_id, quantity=100.0, reasoning=""
    )
    with pytest.raises(ValueError, match="calibrated"):
        await close_round(session, summary.round_id)
