"""Repository functions: the only place that issues queries against the models.

Each function takes an :class:`AsyncSession` and performs one focused persistence
operation. Services compose these; nothing above this layer writes SQL or touches
the ORM query API directly. Functions that mutate state flush (and commit) so the
caller gets back persisted objects with primary keys populated.
"""

from __future__ import annotations

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.enums import EngineMode, Method, Role, RoundStatus
from db.models import (
    Decision,
    Result,
    Round,
    RubricTemplate,
    Student,
    Team,
)

__all__ = [
    # students
    "create_student",
    "get_student_by_telegram_id",
    "assign_student_to_team",
    "list_students",
    # teams
    "create_team",
    "get_team",
    "list_teams",
    "add_team_profit",
    # rounds
    "create_round",
    "get_round",
    "list_rounds",
    "set_round_status",
    "get_open_round",
    # decisions
    "upsert_decision",
    "get_decision",
    "list_decisions_for_round",
    # results
    "save_result",
    "get_result_for_decision",
    # rubrics
    "upsert_rubric_template",
    "get_rubric_for_method",
]


# --------------------------------------------------------------------------- #
# Students
# --------------------------------------------------------------------------- #


async def create_student(
    session: AsyncSession, *, telegram_id: int, full_name: str
) -> Student:
    """Persist a new student and return it with its generated id."""
    student = Student(telegram_id=telegram_id, full_name=full_name)
    session.add(student)
    await session.commit()
    await session.refresh(student)
    return student


async def get_student_by_telegram_id(
    session: AsyncSession, telegram_id: int
) -> Student | None:
    """Look up a student by their Telegram id, or ``None`` if not registered."""
    result = await session.exec(
        select(Student).where(Student.telegram_id == telegram_id)
    )
    return result.first()


async def assign_student_to_team(
    session: AsyncSession, *, student_id: int, team_id: int, role: Role
) -> Student:
    """Place a student on a team with a role.

    Raises
    ------
    ValueError
        If the student id does not exist.
    """
    student = await session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student {student_id} not found")
    student.team_id = team_id
    student.role = role
    session.add(student)
    await session.commit()
    await session.refresh(student)
    return student


async def list_students(session: AsyncSession) -> list[Student]:
    """Return all registered students (joined and not-yet-joined alike)."""
    result = await session.exec(select(Student))
    return list(result.all())


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #


async def create_team(
    session: AsyncSession, *, name: str, company_name: str
) -> Team:
    """Persist a new team."""
    team = Team(name=name, company_name=company_name)
    session.add(team)
    await session.commit()
    await session.refresh(team)
    return team


async def get_team(session: AsyncSession, team_id: int) -> Team | None:
    """Fetch a team by id."""
    return await session.get(Team, team_id)


async def list_teams(session: AsyncSession) -> list[Team]:
    """Return all teams ordered by descending cumulative profit (leaderboard)."""
    result = await session.exec(
        select(Team).order_by(Team.cumulative_profit.desc())  # type: ignore[attr-defined]
    )
    return list(result.all())


async def add_team_profit(
    session: AsyncSession, *, team_id: int, delta: float
) -> Team:
    """Add ``delta`` to a team's cumulative profit.

    Raises
    ------
    ValueError
        If the team does not exist.
    """
    team = await session.get(Team, team_id)
    if team is None:
        raise ValueError(f"team {team_id} not found")
    team.cumulative_profit += delta
    session.add(team)
    await session.commit()
    await session.refresh(team)
    return team


# --------------------------------------------------------------------------- #
# Rounds
# --------------------------------------------------------------------------- #


async def create_round(
    session: AsyncSession,
    *,
    number: int,
    method: Method,
    difficulty: int,
    market_a: float,
    market_b: float,
    market_mc: float,
    case_narrative: str = "",
    status: RoundStatus = RoundStatus.DRAFT,
    engine_mode: EngineMode = EngineMode.SYMMETRIC,
) -> Round:
    """Persist a new round (defaults to ``DRAFT``, symmetric engine)."""
    round_ = Round(
        number=number,
        method=method,
        difficulty=difficulty,
        market_a=market_a,
        market_b=market_b,
        market_mc=market_mc,
        case_narrative=case_narrative,
        status=status,
        engine_mode=engine_mode,
    )
    session.add(round_)
    await session.commit()
    await session.refresh(round_)
    return round_


async def get_round(session: AsyncSession, round_id: int) -> Round | None:
    """Fetch a round by id."""
    return await session.get(Round, round_id)


async def list_rounds(session: AsyncSession) -> list[Round]:
    """Return all rounds ordered by round number."""
    result = await session.exec(select(Round).order_by(Round.number))  # type: ignore[arg-type]
    return list(result.all())


async def set_round_status(
    session: AsyncSession, *, round_id: int, status: RoundStatus
) -> Round:
    """Transition a round to a new status.

    Raises
    ------
    ValueError
        If the round does not exist.
    """
    round_ = await session.get(Round, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    round_.status = status
    session.add(round_)
    await session.commit()
    await session.refresh(round_)
    return round_


async def get_open_round(session: AsyncSession) -> Round | None:
    """Return the currently open round, if any.

    The tournament runs one round at a time; if multiple are somehow open this
    returns the lowest-numbered one.
    """
    result = await session.exec(
        select(Round)
        .where(Round.status == RoundStatus.OPEN)
        .order_by(Round.number)  # type: ignore[arg-type]
    )
    return result.first()


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


async def upsert_decision(
    session: AsyncSession,
    *,
    team_id: int,
    round_id: int,
    quantity: float,
    reasoning: str,
) -> Decision:
    """Create or replace a team's decision for a round.

    A team may revise its submission while a round is open, so this updates the
    existing row if present (respecting the one-decision-per-team-per-round
    constraint) instead of inserting a duplicate.
    """
    existing = await get_decision(session, team_id=team_id, round_id=round_id)
    if existing is not None:
        existing.quantity = quantity
        existing.reasoning = reasoning
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    decision = Decision(
        team_id=team_id,
        round_id=round_id,
        quantity=quantity,
        reasoning=reasoning,
    )
    session.add(decision)
    await session.commit()
    await session.refresh(decision)
    return decision


async def get_decision(
    session: AsyncSession, *, team_id: int, round_id: int
) -> Decision | None:
    """Fetch a team's decision for a specific round, if submitted."""
    result = await session.exec(
        select(Decision).where(
            Decision.team_id == team_id, Decision.round_id == round_id
        )
    )
    return result.first()


async def list_decisions_for_round(
    session: AsyncSession, round_id: int
) -> list[Decision]:
    """Return all decisions submitted for a round."""
    result = await session.exec(
        select(Decision).where(Decision.round_id == round_id)
    )
    return list(result.all())


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


async def save_result(
    session: AsyncSession,
    *,
    decision_id: int,
    price: float,
    profit: float,
    rubric_score: float = 0.0,
    rubric_breakdown_json: str = "",
) -> Result:
    """Create or replace the result row for a decision (one-to-one)."""
    existing = await get_result_for_decision(session, decision_id)
    if existing is not None:
        existing.price = price
        existing.profit = profit
        existing.rubric_score = rubric_score
        existing.rubric_breakdown_json = rubric_breakdown_json
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    result = Result(
        decision_id=decision_id,
        price=price,
        profit=profit,
        rubric_score=rubric_score,
        rubric_breakdown_json=rubric_breakdown_json,
    )
    session.add(result)
    await session.commit()
    await session.refresh(result)
    return result


async def get_result_for_decision(
    session: AsyncSession, decision_id: int
) -> Result | None:
    """Fetch the result for a decision, if computed."""
    result = await session.exec(
        select(Result).where(Result.decision_id == decision_id)
    )
    return result.first()


# --------------------------------------------------------------------------- #
# Rubric templates
# --------------------------------------------------------------------------- #


async def upsert_rubric_template(
    session: AsyncSession, *, method: Method, name: str, criteria_json: str
) -> RubricTemplate:
    """Create or replace the rubric template for a method.

    Each method has at most one active template; re-authoring it overwrites the
    previous one.
    """
    existing = await get_rubric_for_method(session, method)
    if existing is not None:
        existing.name = name
        existing.criteria_json = criteria_json
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    template = RubricTemplate(method=method, name=name, criteria_json=criteria_json)
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def get_rubric_for_method(
    session: AsyncSession, method: Method
) -> RubricTemplate | None:
    """Fetch the rubric template bound to a method, if any."""
    result = await session.exec(
        select(RubricTemplate).where(RubricTemplate.method == method)
    )
    return result.first()
