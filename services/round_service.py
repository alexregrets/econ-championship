"""Round lifecycle operations: opening, closing, and computing results.

Bridges the Cournot engine in :mod:`core.market_engine` and the persistence layer
in :mod:`db.repositories`. The economics live entirely in the engine; this module
only marshals data in and out and updates cumulative standings.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, TeamResult, compute_cournot_round
from core.market_engine_asymmetric import compute_asymmetric_cournot_round
from core.market_events import (
    MarketShock,
    apply_to_costs,
    apply_to_demand,
    apply_to_parameters,
)
from core.role_kpi import RoleKpi, compute_role_kpis
from db import event_repositories as event_repo
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import EngineMode, Role, RoundStatus
from db.models import Decision, Round

__all__ = [
    "open_round",
    "round_shocks",
    "effective_parameters",
    "compute_round_results",
    "score_roles",
    "close_round",
]


async def open_round(session: AsyncSession, round_id: int) -> None:
    """Mark a round as open for submissions.

    Raises
    ------
    ValueError
        If the round does not exist.
    """
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.OPEN
    )


async def round_shocks(session: AsyncSession, round_id: int) -> list[MarketShock]:
    """Загрузить события раунда как шоки для движка.

    Слой перевода между хранением и расчётом: в БД событие — это строка с
    текстом для витрины и флагом видимости, движку же нужен только вид и
    величина. ``revealed`` сюда сознательно не переезжает — он управляет тем,
    что команды видят до подачи решения, а не тем, как считается рынок.
    Скрытый шок бьёт по рынку ровно так же, как публичный.

    Порядок событий не важен: свёртка мультипликативна и коммутативна.
    """
    events = await event_repo.list_events_for_round(session, round_id)
    return [
        MarketShock(kind=e.kind, magnitude=e.magnitude, headline=e.headline)
        for e in events
    ]


def effective_parameters(
    round_: Round, shocks: Sequence[MarketShock]
) -> MarketParameters:
    """Параметры симметричного рынка после событий раунда.

    ``Round`` не мутируется: базовые ``market_a`` / ``market_b`` / ``market_mc``
    остаются в БД нетронутыми. Это не мелочь — без исходных параметров нельзя
    провести разбор после раунда и показать командам, где именно их подвинуло
    событие, а не собственное решение.

    Raises
    ------
    ValueError
        Если после событий рынок нежизнеспособен (издержки догнали точку
        насыщения спроса).
    """
    base = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )
    return apply_to_parameters(base, shocks)


async def _asymmetric_costs(
    session: AsyncSession, round_: Round, decisions: list[Decision]
) -> dict[str, float]:
    """Collect per-team implied marginal costs for an asymmetric round.

    Costs live on :class:`~db.role_models.CompanyGroundTruth` (written by
    ``generate_role_views`` from the calibrated scenario). Every deciding team
    must have one — a missing ground truth or an empty ``implied_marginal_cost``
    is a configuration error, not a case for a silent fallback to ``market_mc``.

    Raises
    ------
    ValueError
        If any deciding team lacks a ground truth row or a calibrated cost.
    """
    assert round_.id is not None  # loaded from the DB
    truths = await role_repo.list_ground_truths_for_round(session, round_.id)
    costs = {
        str(t.team_id): t.implied_marginal_cost
        for t in truths
        if t.implied_marginal_cost is not None
    }
    missing = sorted(str(d.team_id) for d in decisions if str(d.team_id) not in costs)
    if missing:
        raise ValueError(
            f"asymmetric round {round_.id} has no calibrated per-team costs for "
            f"teams {missing}; generate role views (ground truth with implied "
            "costs) before closing the round"
        )
    return {str(d.team_id): costs[str(d.team_id)] for d in decisions}


async def compute_round_results(
    session: AsyncSession, round_id: int
) -> dict[str, TeamResult]:
    """Run the Cournot engine on a round's decisions and persist the outcomes.

    Loads every decision for the round, evaluates the market at the round's
    *effective* parameters — the stored ones after this round's market events
    are applied — writes a :class:`~db.models.Result` per decision, and adds each
    team's profit to its cumulative standing. A round with no events is scored
    on the stored parameters unchanged: the shock factors are exactly ``1.0``,
    so the arithmetic is bit-for-bit what it was before events existed.

    Parameters
    ----------
    session:
        Active async session.
    round_id:
        Round to score.

    Returns
    -------
    dict[str, TeamResult]
        Engine results keyed by ``str(team_id)``.

    Raises
    ------
    ValueError
        If the round does not exist, has no submitted decisions, is an
        asymmetric round missing per-team implied costs (see
        :func:`_asymmetric_costs`), or if this round's events leave the market
        unviable — costs at or above the demand choke price. Refusing is
        deliberate: a market nobody can profitably produce in is a setup error,
        and scoring it anyway would hand teams meaningless numbers.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")

    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        raise ValueError(f"round {round_id} has no decisions to score")

    quantities = {str(d.team_id): d.quantity for d in decisions}
    shocks = await round_shocks(session, round_id)

    if round_.engine_mode is EngineMode.ASYMMETRIC:
        # Спрос и издержки сдвигаются раздельно: шок спроса трогает только
        # a и b, шок издержек — только c_i. Издержки проверяются против уже
        # сдвинутой точки насыщения, то есть против того рынка, на котором
        # раунд и будет считаться, а не против исходного.
        a, b = apply_to_demand(round_.market_a, round_.market_b, shocks)
        costs = apply_to_costs(
            await _asymmetric_costs(session, round_, decisions),
            shocks,
            demand_intercept=a,
        )
        results = compute_asymmetric_cournot_round(
            quantities, a=a, b=b, marginal_costs=costs
        )
    else:
        results = compute_cournot_round(
            quantities, effective_parameters(round_, shocks)
        )

    for decision in decisions:
        assert decision.id is not None  # persisted rows always have an id
        team_result = results[str(decision.team_id)]
        await repo.save_result(
            session,
            decision_id=decision.id,
            price=team_result.price,
            profit=team_result.profit,
        )
        await repo.add_team_profit(
            session, team_id=decision.team_id, delta=team_result.profit
        )

    return results


async def _price_forecasts(
    session: AsyncSession, round_id: int
) -> dict[str, float | None]:
    """Прогнозы цены аналитиков сбыта, ``team_id -> прогноз или None``.

    Берётся только ввод роли :attr:`~db.enums.Role.SALES_ANALYST`: у остальных
    ролей ``price_forecast`` пуст по смыслу, и тащить их в словарь означало бы
    затирать поданный прогноз пустым значением при совпадении команды.
    """
    inputs = await role_repo.list_role_inputs_for_round(session, round_id)
    return {
        str(role_input.team_id): role_input.price_forecast
        for role_input in inputs
        if role_input.role is Role.SALES_ANALYST
    }


async def score_roles(
    session: AsyncSession, round_id: int, results: dict[str, TeamResult]
) -> dict[str, dict[Role, RoleKpi]]:
    """Посчитать личные KPI ролей за раунд и записать :class:`RoleScore`.

    Личная часть оценки существует ради конфликта интересов: маркетолог
    премируется за долю рынка, финансист — за маржу, аналитик — за точность
    прогноза цены. Маркетолог тянет выпуск вверх, финансист вниз, и команда
    вынуждена разруливать спор, а не усреднять его.

    Запись идёт через upsert: пересчёт раунда (например, после исправления
    события) обновляет оценки, а не задваивает их.

    Parameters
    ----------
    results:
        Уже посчитанные результаты раунда — эта функция ничего не считает
        заново и не трогает движок.

    Returns
    -------
    dict[str, dict[Role, RoleKpi]]
        По команде — по три оценки, в том же виде, что вернул
        :func:`core.role_kpi.compute_role_kpis`.
    """
    scores = compute_role_kpis(results, await _price_forecasts(session, round_id))

    for team_id, per_role in scores.items():
        for role, kpi in per_role.items():
            await role_repo.upsert_role_score(
                session,
                round_id=round_id,
                team_id=int(team_id),
                role=role,
                kpi_raw=kpi.raw,
                kpi_name=kpi.raw_name,
                kpi_normalized=kpi.normalized,
                team_component=kpi.team_component,
                total=kpi.total,
                has_input=kpi.has_input,
            )

    return scores


async def close_round(
    session: AsyncSession, round_id: int
) -> dict[str, TeamResult]:
    """Score the round, write per-role KPI, then lock it against further submissions.

    Order matters. Results and role scores are written *before* the status flips
    to ``CLOSED``, so a round that cannot be scored — an unviable market, a
    missing calibrated cost — stays ``OPEN`` and can be fixed and retried.
    Flipping the status first would leave a round locked and unscored, which is
    the one state nobody can get out of from the dashboard.

    Returns the computed results. Raises :class:`ValueError` for the same reasons
    as :func:`compute_round_results`.
    """
    results = await compute_round_results(session, round_id)
    await score_roles(session, round_id, results)
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.CLOSED
    )
    return results
