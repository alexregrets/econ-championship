"""Тесты ролевых данных: реальный сценарий «Нефть РФ 2013», сидер, сессия.

In-memory база, без сети (LLM — фейк по протоколу StructuredLLM). Числа в
проверках — реальные данные 2013 года и величины, выведенные из них
калибровкой (см. DECISIONS.md №13–15):

- добыча: Роснефть 203.03, Лукойл 86.923, Сургутнефтегаз 61.453 млн т;
- Urals: $107.88/барр × 7.28 барр/т = 785.3664 $/т;
- полная себестоимость: $50/барр × 7.28 = 364.0 $/т;
- спрос: a = 4·785.3664 − 3·364 = 2049.4656, b = (a − P)/351.406 ≈ 3.5973.
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
from devshell.role_seed import (
    FULL_COST_2013_USD_PER_TON,
    OIL_PRODUCTION_2013_MLN_T,
    ROLE_COST_SHARES,
    TOTAL_PRODUCTION_2013_MLN_T,
    URALS_PRICE_2013_USD_PER_TON,
    build_role_slices,
    generate_role_views,
    oil_2013_market_parameters,
    seed_oil_2013,
)
from devshell.role_session import (
    commit_lead_decision,
    enter_role,
    lead_overview,
    submit_role_proposal,
    what_if_profit,
)
from devshell.seed import seed
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
# Реальные константы и калибровка рынка
# --------------------------------------------------------------------------- #


def test_real_2013_constants_exact() -> None:
    """Исходные данные — ровно те, что в источниках (ЦДУ ТЭК, Минфин)."""
    assert OIL_PRODUCTION_2013_MLN_T["Роснефть"] == 203.03
    assert OIL_PRODUCTION_2013_MLN_T["Лукойл"] == 86.923
    assert OIL_PRODUCTION_2013_MLN_T["Сургутнефтегаз"] == 61.453
    assert pytest.approx(351.406) == TOTAL_PRODUCTION_2013_MLN_T
    assert pytest.approx(785.3664) == URALS_PRICE_2013_USD_PER_TON
    assert pytest.approx(364.0) == FULL_COST_2013_USD_PER_TON


def test_cost_shares_constant_sums_to_one() -> None:
    assert sum(ROLE_COST_SHARES.values()) == pytest.approx(1.0)


def test_oil_2013_calibration_reproduces_observed_point() -> None:
    """Главный тест калибровки: Нэш калиброванного рынка = факт 2013 года.

    Симметричное равновесие трёх фирм должно давать суммарный выпуск,
    равный реальной суммарной добыче, и цену, равную реальной цене Urals.
    """
    params = oil_2013_market_parameters()
    assert params.a == pytest.approx(2049.4656)
    assert params.marginal_cost == pytest.approx(364.0)

    q_star = nash_equilibrium(3, params)
    total_nash = 3 * q_star
    assert total_nash == pytest.approx(TOTAL_PRODUCTION_2013_MLN_T)
    price_at_nash = params.a - params.b * total_nash
    assert price_at_nash == pytest.approx(URALS_PRICE_2013_USD_PER_TON)


# --------------------------------------------------------------------------- #
# build_role_slices: чистая генерация срезов из реальных данных
# --------------------------------------------------------------------------- #


def test_slices_cover_all_three_roles() -> None:
    slices = build_role_slices(
        "Роснефть",
        own_production=203.03,
        params=oil_2013_market_parameters(),
        total_production=TOTAL_PRODUCTION_2013_MLN_T,
    )
    assert set(slices) == {Role.MARKETER, Role.SALES_ANALYST, Role.FINANCIER}


def test_slices_private_signals_are_real_2013_numbers() -> None:
    """Маркетолог видит a, аналитик — конкурентов, финансист — себестоимость."""
    slices = build_role_slices(
        "Роснефть",
        own_production=203.03,
        params=oil_2013_market_parameters(),
        total_production=TOTAL_PRODUCTION_2013_MLN_T,
    )
    assert slices[Role.MARKETER].private_signal == pytest.approx(2049.4656)
    assert slices[Role.FINANCIER].private_signal == pytest.approx(364.0)
    # Конкуренты Роснефти: 351.406 − 203.03 = 148.376 млн т.
    assert slices[Role.SALES_ANALYST].private_signal == pytest.approx(148.376)


def test_slices_competitors_differ_per_company() -> None:
    """Асимметрия реальна: у Лукойла и Сургутнефтегаза свои конкуренты."""
    params = oil_2013_market_parameters()
    lukoil = build_role_slices(
        "Лукойл",
        own_production=86.923,
        params=params,
        total_production=TOTAL_PRODUCTION_2013_MLN_T,
    )
    surgut = build_role_slices(
        "Сургутнефтегаз",
        own_production=61.453,
        params=params,
        total_production=TOTAL_PRODUCTION_2013_MLN_T,
    )
    assert lukoil[Role.SALES_ANALYST].private_signal == pytest.approx(264.483)
    assert surgut[Role.SALES_ANALYST].private_signal == pytest.approx(289.953)


def test_slices_cost_shares_sum_to_one() -> None:
    slices = build_role_slices(
        "Лукойл",
        own_production=86.923,
        params=oil_2013_market_parameters(),
        total_production=TOTAL_PRODUCTION_2013_MLN_T,
    )
    assert sum(s.cost_share for s in slices.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# seed_oil_2013 + generate_role_views: сидер против БД
# --------------------------------------------------------------------------- #


async def test_seed_oil_creates_three_real_teams(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    assert len(summary.team_ids) == 3
    assert summary.student_count == 9
    teams = await repo.list_teams(session)
    assert {t.company_name for t in teams} == set(OIL_PRODUCTION_2013_MLN_T)

    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.OPEN
    assert round_.market_a == pytest.approx(2049.4656)
    assert round_.market_mc == pytest.approx(364.0)


async def test_generate_creates_truth_and_three_views_per_team(
    session: AsyncSession,
) -> None:
    summary = await seed_oil_2013(session)
    views = await generate_role_views(session, summary.round_id)
    assert len(views) == 3 * 3  # 3 компании × 3 роли

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
    """Общие поля одинаковы у всех ролей, равны реальным цифрам; доли дают 1."""
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)

    for team_id in summary.team_ids:
        truth = await role_repo.get_ground_truth(
            session, round_id=summary.round_id, team_id=team_id
        )
        assert truth is not None
        assert truth.ref_total_quantity == pytest.approx(351.406)
        assert truth.observed_price == pytest.approx(785.3664)

        views = await role_repo.list_role_views_for_team(
            session, round_id=summary.round_id, team_id=team_id
        )
        for view in views:
            assert view.ref_total_quantity == pytest.approx(truth.ref_total_quantity)
            assert view.observed_price == pytest.approx(truth.observed_price)
        assert sum(v.cost_share for v in views) == pytest.approx(1.0)


async def test_observed_point_lies_on_calibrated_demand_curve(
    session: AsyncSession,
) -> None:
    """Инвариант ground truth: P_набл = a − b·Q_набл (точка на кривой спроса)."""
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)
    truth = await role_repo.get_ground_truth(
        session, round_id=summary.round_id, team_id=summary.team_ids[0]
    )
    assert truth is not None
    implied_price = truth.demand_a - truth.demand_b * truth.ref_total_quantity
    assert implied_price == pytest.approx(truth.observed_price)


async def test_generate_rejects_missing_round(session: AsyncSession) -> None:
    with pytest.raises(ValueError):
        await generate_role_views(session, round_id=999)


async def test_generate_rejects_companies_without_real_data(
    session: AsyncSession,
) -> None:
    """Generic-сидер (Сбербанк, Яндекс...) не годится для ролевого пилота."""
    summary = await seed(session)  # 7 синтетических команд
    with pytest.raises(ValueError, match="Сбербанк"):
        await generate_role_views(session, summary.round_id)


async def test_generate_rejects_round_with_mismatched_params(
    session: AsyncSession,
) -> None:
    """Срезы описывают рынок 2013 — на чужих параметрах раунда они врут."""
    summary = await seed_oil_2013(session)
    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    round_.market_a = 100.0  # профессор «перекрутил» рынок руками
    session.add(round_)
    await session.commit()
    with pytest.raises(ValueError, match="не совпадают со сценарием"):
        await generate_role_views(session, summary.round_id)


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
    summary = await seed_oil_2013(session)
    views = await generate_role_views(
        session, summary.round_id, llm=_FakeNarrativeLLM()
    )
    narratives = {v.role: v.narrative for v in views[:3]}
    assert narratives[Role.MARKETER] == "брифинг маркетолога"
    assert narratives[Role.FINANCIER] == "брифинг финансиста"


async def test_static_narratives_mention_real_numbers(
    session: AsyncSession,
) -> None:
    """Без LLM тексты всё равно содержат реальные цифры, а не заглушки."""
    summary = await seed_oil_2013(session)
    views = await generate_role_views(session, summary.round_id)
    by_role = {(v.ground_truth_id, v.role): v for v in views}
    analyst_texts = [v.narrative for v in views if v.role is Role.SALES_ANALYST]
    assert any("203.0" in text for text in analyst_texts)  # добыча Роснефти
    financier_texts = [v.narrative for v in views if v.role is Role.FINANCIER]
    assert all("$50/барр" in text for text in financier_texts)
    assert by_role  # словарь построился без коллизий (уникальность пар)


# --------------------------------------------------------------------------- #
# Ролевая сессия: вход в роль, предложения, агрегация, what-if, commit
# --------------------------------------------------------------------------- #


async def test_enter_role_returns_only_own_slice(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    view = await enter_role(
        session, round_id=summary.round_id, team_id=team_id, role=Role.FINANCIER
    )
    assert view.role is Role.FINANCIER
    # Сигнал финансиста — реальная полная себестоимость, $/т.
    assert view.slice_.private_signal == pytest.approx(364.0)
    assert view.my_proposal is None  # ещё ничего не предлагал


async def test_enter_role_fails_before_seeding_views(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    with pytest.raises(ValueError):
        await enter_role(
            session,
            round_id=summary.round_id,
            team_id=summary.team_ids[0],
            role=Role.MARKETER,
        )


async def test_proposals_flow_into_lead_overview(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=100.0,
    )
    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.FINANCIER,
        quantity=140.0,
    )

    overview = await lead_overview(
        session, round_id=summary.round_id, team_id=team_id
    )
    assert overview.proposals[Role.MARKETER] == 100.0
    assert overview.proposals[Role.FINANCIER] == 140.0
    # Равные веса → среднее.
    assert overview.aggregated_quantity == pytest.approx(120.0)


async def test_proposal_revision_keeps_single_row(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)
    team_id = summary.team_ids[0]

    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=100.0,
    )
    await submit_role_proposal(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        role=Role.MARKETER,
        quantity=112.5,
        note="передумал",
    )
    inputs = await role_repo.list_role_inputs_for_team(
        session, round_id=summary.round_id, team_id=team_id
    )
    assert len(inputs) == 1
    assert inputs[0].quantity_proposal == 112.5
    assert inputs[0].note == "передумал"


async def test_proposal_rejected_when_round_not_open(session: AsyncSession) -> None:
    summary = await seed_oil_2013(session)
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
            quantity=100.0,
        )


def test_what_if_profit_exact_numbers() -> None:
    """Чистая функция на простых параметрах: q=30 против 20 → P=50, π=1200."""
    params = MarketParameters(a=100.0, b=1.0, marginal_cost=10.0)
    price, profit = what_if_profit(params, my_quantity=30.0, other_quantities=[20.0])
    assert price == pytest.approx(50.0)
    assert profit == pytest.approx(1200.0)


async def test_commit_lead_decision_creates_single_team_decision(
    session: AsyncSession,
) -> None:
    summary = await seed_oil_2013(session)
    team_id = summary.team_ids[0]
    await commit_lead_decision(
        session,
        round_id=summary.round_id,
        team_id=team_id,
        quantity=120.0,
        reasoning="lead: среднее предложений ролей",
    )
    decisions = await repo.list_decisions_for_round(session, summary.round_id)
    assert len(decisions) == 1
    assert decisions[0].team_id == team_id
    assert decisions[0].quantity == 120.0


async def test_full_role_round_integration(session: AsyncSession) -> None:
    """Полный цикл на реальном рынке: срезы → предложения → lead → закрытие.

    Все 3 компании играют агрегат 120 млн т → Q_total = 360,
    P = 2049.4656 − b·360 ≈ 754.45 $/т,
    прибыль каждой = (P − 364)·120 ≈ 46 854 млн $.
    """
    summary = await seed_oil_2013(session)
    await generate_role_views(session, summary.round_id)

    for team_id in summary.team_ids:
        # Две роли предлагают 100 и 140 → lead берёт равновзвешенные 120.
        await submit_role_proposal(
            session,
            round_id=summary.round_id,
            team_id=team_id,
            role=Role.MARKETER,
            quantity=100.0,
        )
        await submit_role_proposal(
            session,
            round_id=summary.round_id,
            team_id=team_id,
            role=Role.FINANCIER,
            quantity=140.0,
        )
        overview = await lead_overview(
            session, round_id=summary.round_id, team_id=team_id
        )
        assert overview.aggregated_quantity is not None
        assert overview.aggregated_quantity == pytest.approx(120.0)
        await commit_lead_decision(
            session,
            round_id=summary.round_id,
            team_id=team_id,
            quantity=overview.aggregated_quantity,
            reasoning="lead: агрегат предложений",
        )

    results = await close_round(session, summary.round_id)
    assert len(results) == 3
    for team_result in results.values():
        assert team_result.price == pytest.approx(754.451, abs=0.01)
        assert team_result.profit == pytest.approx(46854.1, rel=1e-4)

    round_ = await repo.get_round(session, summary.round_id)
    assert round_ is not None
    assert round_.status is RoundStatus.CLOSED


async def test_nash_play_reproduces_2013_profit_scale(
    session: AsyncSession,
) -> None:
    """Если все играют Нэш, цена = реальная цена Urals 2013 (по калибровке).

    Прибыль каждой фирмы: (785.3664 − 364) × 117.135 ≈ 49 360 млн $ —
    масштаб реалистичен (порядок десятков млрд $ выручки-маржи по отрасли).
    """
    summary = await seed_oil_2013(session)
    params = oil_2013_market_parameters()
    q_star = nash_equilibrium(3, params)
    assert q_star == pytest.approx(351.406 / 3, rel=1e-9)

    for team_id in summary.team_ids:
        await commit_lead_decision(
            session,
            round_id=summary.round_id,
            team_id=team_id,
            quantity=q_star,
            reasoning="Нэш",
        )
    results = await close_round(session, summary.round_id)
    for team_result in results.values():
        assert team_result.price == pytest.approx(785.3664)
        assert team_result.profit == pytest.approx(
            (785.3664 - 364.0) * 351.406 / 3, rel=1e-6
        )