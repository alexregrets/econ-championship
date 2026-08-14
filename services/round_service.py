"""Round lifecycle operations: opening, closing, and computing results.

Bridges the Cournot engine in :mod:`core.market_engine` and the persistence layer
in :mod:`db.repositories`. The economics live entirely in the engine; this module
only marshals data in and out and updates cumulative standings.
"""

from __future__ import annotations

from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, TeamResult, compute_cournot_round
from db import repositories as repo
from db.enums import RoundStatus

__all__ = ["open_round", "compute_round_results", "close_round"]


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


async def compute_round_results(
    session: AsyncSession, round_id: int
) -> dict[str, TeamResult]:
    """Run the Cournot engine on a round's decisions and persist the outcomes.

    Loads every decision for the round, evaluates the market with the round's
    stored parameters, writes a :class:`~db.models.Result` per decision, and adds
    each team's profit to its cumulative standing.

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
        If the round does not exist or has no submitted decisions.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")

    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        raise ValueError(f"round {round_id} has no decisions to score")

    params = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )
    quantities = {str(d.team_id): d.quantity for d in decisions}
    results = compute_cournot_round(quantities, params)

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


async def close_round(
    session: AsyncSession, round_id: int
) -> dict[str, TeamResult]:
    """Compute results then mark the round closed (locking further submissions).

    Returns the computed results. Raises :class:`ValueError` for the same reasons
    as :func:`compute_round_results`.
    """
    results = await compute_round_results(session, round_id)
    await repo.set_round_status(
        session, round_id=round_id, status=RoundStatus.CLOSED
    )
    return results
