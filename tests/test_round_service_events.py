"""Wiring рыночных событий в жизненный цикл раунда.

Гарантии:
- REGRESSION: раунд без событий считается байт-в-байт как раньше (обе ветки
  движка) — событие обязано быть строго опциональной надстройкой;
- шок спроса и шок издержек доезжают до движка и меняют результат в
  экономически правильную сторону (цена/прибыль вниз);
- в асимметричном раунде COST_SHOCK применяется к каждому пофирменному c_i,
  а шок спроса — только к a/b;
- скрытое событие (revealed=False) влияет на рынок так же, как публичное;
- события, ломающие рынок, не дают закрыть раунд: ValueError и статус OPEN.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, compute_cournot_round
from core.market_engine_asymmetric import compute_asymmetric_cournot_round
from db import event_repositories as event_repo
from db import models  # noqa: F401  (регистрирует таблицы)
from db import repositories as repo
from db.enums import EngineMode, EventKind, RoundStatus
from devshell.role_seed import (
    OIL_PRODUCTION_2013_MLN_T,
    generate_role_views,
    implied_oil_2013_costs,
    seed_oil_2013,
)
from devshell.seed import seed
from devshell.simulate_team import simulate_all_teams_nash
from services.round_service import (
    close_round,
    compute_round_results,
    effective_parameters,
    round_shocks,
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


async def _seed_symmetric_with_decisions(session: AsyncSession) -> int:
    """Обычный сидер + решения всех команд; вернуть id раунда."""
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    return summary.round_id


async def _seed_asymmetric_with_decisions(session: AsyncSession) -> int:
    """Нефтяной сидер в асимметричном режиме + решения = реальные добычи."""
    summary = await seed_oil_2013(session, engine_mode=EngineMode.ASYMMETRIC)
    await generate_role_views(session, summary.round_id)
    teams = await repo.list_teams(session)
    for team in teams:
        assert team.id is not None
        await repo.upsert_decision(
            session,
            team_id=team.id,
            round_id=summary.round_id,
            quantity=OIL_PRODUCTION_2013_MLN_T[team.company_name],
            reasoning="фактическая добыча 2013",
        )
    return summary.round_id


# --------------------------------------------------------------------------- #
# REGRESSION: без событий ничего не изменилось
# --------------------------------------------------------------------------- #


async def test_round_without_events_scores_exactly_as_before(
    session: AsyncSession,
) -> None:
    round_id = await _seed_symmetric_with_decisions(session)
    decisions = await repo.list_decisions_for_round(session, round_id)
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None

    expected = compute_cournot_round(
        {str(d.team_id): d.quantity for d in decisions},
        MarketParameters(
            a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
        ),
    )

    assert await compute_round_results(session, round_id) == expected


async def test_asymmetric_round_without_events_unchanged(
    session: AsyncSession,
) -> None:
    round_id = await _seed_asymmetric_with_decisions(session)
    decisions = await repo.list_decisions_for_round(session, round_id)
    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    teams = {t.id: t.company_name for t in await repo.list_teams(session)}
    implied = implied_oil_2013_costs()

    expected = compute_asymmetric_cournot_round(
        {str(d.team_id): d.quantity for d in decisions},
        a=round_.market_a,
        b=round_.market_b,
        marginal_costs={
            str(d.team_id): implied[teams[d.team_id]] for d in decisions
        },
    )

    assert await compute_round_results(session, round_id) == expected


async def test_no_events_means_no_shocks(session: AsyncSession) -> None:
    round_id = await _seed_symmetric_with_decisions(session)

    assert await round_shocks(session, round_id) == []


# --------------------------------------------------------------------------- #
# Событие доезжает до движка
# --------------------------------------------------------------------------- #


async def test_demand_shock_lowers_price_and_profit(session: AsyncSession) -> None:
    """Обвал спроса на 25% при тех же решениях: цена и прибыль падают."""
    round_id = await _seed_symmetric_with_decisions(session)
    before = await compute_round_results(session, round_id)

    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.25,
        headline="ОПЕК не сокращает добычу",
        preset_key="opec_no_cut",
    )
    after = await compute_round_results(session, round_id)

    team_id = next(iter(before))
    assert after[team_id].price < before[team_id].price
    assert after[team_id].profit < before[team_id].profit
    assert after[team_id].quantity == before[team_id].quantity  # решения те же


async def test_symmetric_round_matches_shocked_parameters(
    session: AsyncSession,
) -> None:
    """Результат раунда == прямой вызов движка на сдвинутых параметрах."""
    round_id = await _seed_symmetric_with_decisions(session)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.COST_SHOCK,
        magnitude=0.18,
        headline="Санкции",
    )
    round_ = await repo.get_round(session, round_id)
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert round_ is not None

    shocked = effective_parameters(round_, await round_shocks(session, round_id))
    expected = compute_cournot_round(
        {str(d.team_id): d.quantity for d in decisions}, shocked
    )

    assert shocked.marginal_cost == pytest.approx(round_.market_mc * 1.18)
    assert await compute_round_results(session, round_id) == expected


async def test_base_round_parameters_stay_untouched(session: AsyncSession) -> None:
    """Событие не переписывает market_a/market_mc — база нужна для разбора."""
    round_id = await _seed_symmetric_with_decisions(session)
    before = await repo.get_round(session, round_id)
    assert before is not None
    base_a, base_mc = before.market_a, before.market_mc

    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.25,
        headline="Обвал спроса",
    )
    await compute_round_results(session, round_id)

    after = await repo.get_round(session, round_id)
    assert after is not None
    assert (after.market_a, after.market_mc) == (base_a, base_mc)


async def test_hidden_event_still_moves_the_market(session: AsyncSession) -> None:
    """revealed=False управляет показом, а не расчётом."""
    round_id = await _seed_symmetric_with_decisions(session)
    before = await compute_round_results(session, round_id)

    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.2,
        headline="Скрытый шок",
        revealed=False,
    )
    after = await compute_round_results(session, round_id)

    team_id = next(iter(before))
    assert after[team_id].price < before[team_id].price


# --------------------------------------------------------------------------- #
# Асимметричная ветка
# --------------------------------------------------------------------------- #


async def test_cost_shock_hits_every_implied_cost(session: AsyncSession) -> None:
    """В асимметричном раунде COST_SHOCK множит каждое c_i на один множитель."""
    round_id = await _seed_asymmetric_with_decisions(session)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.COST_SHOCK,
        magnitude=0.10,
        headline="Налоговый манёвр",
        preset_key="tax_maneuver",
    )
    round_ = await repo.get_round(session, round_id)
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert round_ is not None
    teams = {t.id: t.company_name for t in await repo.list_teams(session)}
    implied = implied_oil_2013_costs()

    expected = compute_asymmetric_cournot_round(
        {str(d.team_id): d.quantity for d in decisions},
        a=round_.market_a,
        b=round_.market_b,
        marginal_costs={
            str(d.team_id): implied[teams[d.team_id]] * 1.10 for d in decisions
        },
    )

    assert await compute_round_results(session, round_id) == expected


async def test_demand_shock_keeps_asymmetric_costs(session: AsyncSession) -> None:
    """Шок спроса двигает только a — издержки команд остаются прежними."""
    round_id = await _seed_asymmetric_with_decisions(session)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.10,
        headline="Замедление Китая",
    )
    round_ = await repo.get_round(session, round_id)
    decisions = await repo.list_decisions_for_round(session, round_id)
    assert round_ is not None
    teams = {t.id: t.company_name for t in await repo.list_teams(session)}
    implied = implied_oil_2013_costs()

    expected = compute_asymmetric_cournot_round(
        {str(d.team_id): d.quantity for d in decisions},
        a=round_.market_a * 0.90,
        b=round_.market_b,
        marginal_costs={
            str(d.team_id): implied[teams[d.team_id]] for d in decisions
        },
    )

    assert await compute_round_results(session, round_id) == expected


# --------------------------------------------------------------------------- #
# Отказ считать сломанный рынок
# --------------------------------------------------------------------------- #


async def test_market_broken_by_events_keeps_round_open(
    session: AsyncSession,
) -> None:
    """Спрос обвален так, что издержки его догнали — раунд не закрывается."""
    round_id = await _seed_symmetric_with_decisions(session)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.DEMAND_SHIFT,
        magnitude=-0.95,
        headline="Катастрофа спроса",
    )

    with pytest.raises(ValueError, match="нежизнеспособ"):
        await close_round(session, round_id)

    round_ = await repo.get_round(session, round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN


async def test_asymmetric_market_broken_by_cost_shock_raises(
    session: AsyncSession,
) -> None:
    """Издержки, задранные выше точки насыщения, — явная ошибка конфигурации."""
    round_id = await _seed_asymmetric_with_decisions(session)
    await event_repo.create_market_event(
        session,
        round_id=round_id,
        kind=EventKind.COST_SHOCK,
        magnitude=5.0,
        headline="Невозможный налог",
    )

    with pytest.raises(ValueError, match="нежизнеспособ"):
        await compute_round_results(session, round_id)
