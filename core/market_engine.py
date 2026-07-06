"""Deterministic Cournot competition engine.

The tournament is built on a single Cournot oligopoly market shared by all teams.
Each round, every team commits a production quantity ``q``. The market clears at a
linear inverse-demand price ``P = a - b * Q_total`` (floored at zero), and each
team's profit is ``P * q - marginal_cost * q``.

This module is intentionally free of any I/O, randomness, or LLM calls: given the
same decisions and parameters it always returns the same results, which makes it
the trustworthy "ground truth" that grading and leaderboards build on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "MarketParameters",
    "TeamResult",
    "compute_cournot_round",
    "nash_equilibrium",
    "monopoly_quantity",
]


@dataclass(frozen=True)
class MarketParameters:
    """Parameters of a linear Cournot market: ``P = a - b * Q_total``.

    Set by the professor (Kudryavtsev) per round through the dashboard.

    Parameters
    ----------
    a:
        Demand-curve intercept (the choke price at which demand hits zero).
        Must be positive.
    b:
        Demand-curve slope, i.e. how sharply price falls as total industry
        output rises. Must be positive.
    marginal_cost:
        Constant per-unit production cost shared by all firms. Must be
        non-negative and strictly below ``a`` for a viable market.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """

    a: float
    b: float
    marginal_cost: float = 10.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"demand intercept 'a' must be positive, got {self.a}")
        if self.b <= 0:
            raise ValueError(f"demand slope 'b' must be positive, got {self.b}")
        if self.marginal_cost < 0:
            raise ValueError(
                f"marginal_cost must be non-negative, got {self.marginal_cost}"
            )
        if self.marginal_cost >= self.a:
            raise ValueError(
                f"marginal_cost ({self.marginal_cost}) must be below demand "
                f"intercept a ({self.a}); otherwise no profitable output exists"
            )


@dataclass(frozen=True)
class TeamResult:
    """Outcome for a single team in one Cournot round.

    Attributes
    ----------
    team_id:
        Identifier of the team.
    quantity:
        Quantity the team chose to produce.
    price:
        Market-clearing price for the round (identical for all teams).
    revenue:
        ``price * quantity``.
    cost:
        ``marginal_cost * quantity``.
    profit:
        ``revenue - cost`` (may be negative if a team floods the market).
    """

    team_id: str
    quantity: float
    price: float
    revenue: float
    cost: float
    profit: float


def compute_cournot_round(
    decisions: Mapping[str, float],
    params: MarketParameters,
) -> dict[str, TeamResult]:
    """Compute one round of Cournot competition for N teams.

    Parameters
    ----------
    decisions:
        Mapping of ``team_id -> quantity`` produced by each team. Quantities
        must be finite and non-negative.
    params:
        Market parameters for the round.

    Returns
    -------
    dict[str, TeamResult]
        One :class:`TeamResult` per team, keyed by ``team_id``. The market
        price is shared across all results.

    Raises
    ------
    ValueError
        If ``decisions`` is empty or contains a negative / non-finite quantity.

    Notes
    -----
    The market price is floored at zero: if total output is so large that the
    linear demand curve would imply a negative price, the price is set to ``0``
    (goods are given away). In that regime teams still incur production cost, so
    profits go negative — the intended penalty for overproduction.
    """
    if not decisions:
        raise ValueError("decisions must contain at least one team")

    total_quantity = 0.0
    for team_id, quantity in decisions.items():
        if quantity != quantity or quantity in (float("inf"), float("-inf")):
            raise ValueError(f"team {team_id!r} has non-finite quantity {quantity}")
        if quantity < 0:
            raise ValueError(
                f"team {team_id!r} has negative quantity {quantity}; "
                "quantities must be >= 0"
            )
        total_quantity += quantity

    price = max(0.0, params.a - params.b * total_quantity)

    results: dict[str, TeamResult] = {}
    for team_id, quantity in decisions.items():
        revenue = price * quantity
        cost = params.marginal_cost * quantity
        results[team_id] = TeamResult(
            team_id=team_id,
            quantity=quantity,
            price=price,
            revenue=revenue,
            cost=cost,
            profit=revenue - cost,
        )
    return results


def nash_equilibrium(n_firms: int, params: MarketParameters) -> float:
    """Return the symmetric Cournot–Nash equilibrium quantity per firm.

    For ``N`` identical firms with constant marginal cost the symmetric
    equilibrium output is ``q* = (a - c) / ((N + 1) * b)``. Used as a benchmark
    to judge how close a team's decision is to game-theoretic optimal play.

    Parameters
    ----------
    n_firms:
        Number of competing firms (>= 1).
    params:
        Market parameters.

    Returns
    -------
    float
        Equilibrium quantity for a single firm.

    Raises
    ------
    ValueError
        If ``n_firms`` is less than 1.
    """
    if n_firms < 1:
        raise ValueError(f"n_firms must be >= 1, got {n_firms}")
    return (params.a - params.marginal_cost) / ((n_firms + 1) * params.b)


def monopoly_quantity(params: MarketParameters) -> float:
    """Return the profit-maximising total output for a single monopolist.

    Equal to ``(a - c) / (2 * b)``. This is the total industry output a fully
    colluding cartel of teams would produce; comparing realised total output to
    this and to the competitive benchmark shows how cooperative or aggressive a
    round turned out to be.

    Parameters
    ----------
    params:
        Market parameters.

    Returns
    -------
    float
        Monopoly (collusive) total quantity.
    """
    return (params.a - params.marginal_cost) / (2 * params.b)
