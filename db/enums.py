"""Shared enumerations used across the database and business layers.

Kept in their own module so both :mod:`db.models` and the pure-logic layers
(:mod:`core`) can import them without creating an import cycle through the ORM.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Role", "Method", "RoundStatus", "EngineMode"]


class Role(StrEnum):
    """The three functional roles inside a team of three students."""

    MARKETER = "marketer"
    SALES_ANALYST = "sales_analyst"
    FINANCIER = "financier"


class Method(StrEnum):
    """The six econometric methods from Kudryavtsev's syllabus, in order.

    Each round targets exactly one method, and a :class:`~db.models.RubricTemplate`
    is bound per method. Deliberately stops at panel data — no IV estimation and
    no GARCH, which are out of syllabus.
    """

    OLS_SIMPLE = "ols_simple"  # Simple (pairwise) regression
    OLS_MULTIPLE = "ols_multiple"  # Multiple regression + dummies
    MULTICOLLINEARITY = "multicollinearity"  # VIF diagnostics
    HETEROSCEDASTICITY = "heteroscedasticity"  # White, Breusch-Pagan, GLS
    AUTOCORRELATION = "autocorrelation"  # Durbin-Watson, Breusch-Godfrey
    PANEL_DATA = "panel_data"  # Fixed / random effects


class EngineMode(StrEnum):
    """Which Cournot engine scores a round (see OPEN_QUESTIONS.md №7).

    ``SYMMETRIC``  — one shared marginal cost from ``Round.market_mc``
        (:mod:`core.market_engine`); default, matches all pre-existing rounds.
    ``ASYMMETRIC`` — per-team implied costs stored on
        :class:`~db.role_models.CompanyGroundTruth`
        (:mod:`core.market_engine_asymmetric`); opt-in per round.
    """

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class RoundStatus(StrEnum):
    """Lifecycle of a tournament round.

    ``DRAFT``   — being prepared by the professor, not visible to teams.
    ``OPEN``    — teams may submit decisions.
    ``CLOSED``  — submissions locked; results computed and graded.
    """

    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"
