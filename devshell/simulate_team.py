"""Emulate teams submitting decisions, with no Telegram bot involved.

Useful for exercising a full round from the dev-shell: drop in quantities for
every team (explicitly, at the Nash benchmark, or randomly) and a placeholder
reasoning string, then score the round.
"""

from __future__ import annotations

import random

from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, nash_equilibrium
from db import repositories as repo
from db.models import Decision

__all__ = ["submit_decision", "simulate_all_teams_nash", "simulate_all_teams_random"]

_DEFAULT_REASONING = "[simulated submission — no reasoning provided]"


async def submit_decision(
    session: AsyncSession,
    *,
    team_id: int,
    round_id: int,
    quantity: float,
    reasoning: str = _DEFAULT_REASONING,
) -> Decision:
    """Submit (or revise) a single team's decision for a round."""
    return await repo.upsert_decision(
        session,
        team_id=team_id,
        round_id=round_id,
        quantity=quantity,
        reasoning=reasoning,
    )


async def simulate_all_teams_nash(
    session: AsyncSession, round_id: int
) -> dict[int, float]:
    """Submit the symmetric Nash quantity for every team in the round.

    A handy sanity baseline: after scoring, all teams should earn equal profit.

    Returns
    -------
    dict[int, float]
        Mapping of ``team_id -> submitted quantity``.

    Raises
    ------
    ValueError
        If the round does not exist.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    teams = await repo.list_teams(session)
    params = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )
    q_star = nash_equilibrium(len(teams), params)

    submitted: dict[int, float] = {}
    for team in teams:
        assert team.id is not None
        await submit_decision(
            session,
            team_id=team.id,
            round_id=round_id,
            quantity=q_star,
            reasoning="[simulated: Nash equilibrium quantity]",
        )
        submitted[team.id] = q_star
    return submitted


async def simulate_all_teams_random(
    session: AsyncSession,
    round_id: int,
    *,
    low: float = 0.0,
    high: float = 30.0,
    seed: int | None = None,
) -> dict[int, float]:
    """Submit a random quantity in ``[low, high]`` for every team.

    Parameters
    ----------
    session, round_id:
        Target round.
    low, high:
        Inclusive bounds for the random quantity.
    seed:
        Optional RNG seed for reproducible simulations.

    Returns
    -------
    dict[int, float]
        Mapping of ``team_id -> submitted quantity``.
    """
    rng = random.Random(seed)
    teams = await repo.list_teams(session)
    submitted: dict[int, float] = {}
    for team in teams:
        assert team.id is not None
        quantity = round(rng.uniform(low, high), 2)
        await submit_decision(
            session,
            team_id=team.id,
            round_id=round_id,
            quantity=quantity,
            reasoning="[simulated: random quantity]",
        )
        submitted[team.id] = quantity
    return submitted
