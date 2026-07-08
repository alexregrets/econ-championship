"""Операции бота без Telegram: вступление в команду, подача решения, статус.

Функции принимают AsyncSession и не знают ничего про aiogram — тестируются
тем же стилем, что dashboard/actions.py. Валидация «решения принимает только
открытый раунд» НЕ дублируется: подача идёт через существующий
:func:`dashboard.actions.submit_manual_decision` — тот же общий слой, которым
пользуется дашборд (он вынесен из Streamlit именно для переиспользования).

Ошибки для пользователя — ValueError с русским текстом: хендлер показывает
его сообщением, ничего не переформулируя.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from dashboard.actions import submit_manual_decision
from db import bot_repositories as bot_repo
from db import repositories as repo
from db.models import Decision, Round, Team

__all__ = [
    "JoinOutcome",
    "SubmitOutcome",
    "TeamStatus",
    "join_team",
    "submit_decision",
    "team_status",
]


@dataclass(frozen=True)
class JoinOutcome:
    """Результат /join: команда и был ли студент создан впервые."""

    team: Team
    created_student: bool


@dataclass(frozen=True)
class SubmitOutcome:
    """Результат /submit: раунд, команда, решение и было ли оно заменой."""

    round: Round
    team: Team
    decision: Decision
    replaced: bool


@dataclass(frozen=True)
class TeamStatus:
    """Ответ /status: команда, открытый раунд (если есть) и решение в нём."""

    team: Team
    open_round: Round | None
    decision: Decision | None


async def join_team(
    session: AsyncSession, *, telegram_id: int, full_name: str, code: str
) -> JoinOutcome:
    """Привязать Telegram-аккаунт к команде по её коду.

    Новый студент создаётся через существующий ``repo.create_student``;
    уже известный — перепривязывается (опечатка в коде не должна запирать
    студента в чужой команде). Роль не назначается: бот работает только с
    «лёгким» раундом.

    Raises
    ------
    ValueError
        Если код пуст или не соответствует ни одной команде.
    """
    normalized = code.strip()
    if not normalized:
        raise ValueError("Код пустой. Использование: /join <код команды>")
    team = await bot_repo.get_team_by_join_code(session, normalized)
    if team is None:
        raise ValueError(
            "Код команды не найден. Проверьте код у капитана или преподавателя."
        )
    assert team.id is not None  # прочитана из БД

    student = await repo.get_student_by_telegram_id(session, telegram_id)
    created = student is None
    if student is None:
        student = await repo.create_student(
            session, telegram_id=telegram_id, full_name=full_name
        )
    assert student.id is not None
    await bot_repo.bind_student_to_team(
        session, student_id=student.id, team_id=team.id
    )
    return JoinOutcome(team=team, created_student=created)


async def _require_team(session: AsyncSession, telegram_id: int) -> Team:
    """Вернуть команду студента или объяснить, что сначала нужен /join."""
    student = await repo.get_student_by_telegram_id(session, telegram_id)
    if student is None or student.team_id is None:
        raise ValueError(
            "Вы ещё не в команде. Сначала вступите: /join <код команды>."
        )
    team = await repo.get_team(session, student.team_id)
    if team is None:
        raise ValueError(
            "Ваша команда не найдена (база пересоздавалась?). "
            "Вступите заново: /join <код команды>."
        )
    return team


async def submit_decision(
    session: AsyncSession, *, telegram_id: int, quantity: float, reasoning: str
) -> SubmitOutcome:
    """Подать (или заменить) решение своей команды в открытый раунд.

    Раунд не передаётся параметром: турнир идёт по одному раунду за раз,
    решение попадает в текущий открытый (``repo.get_open_round``). Сама
    запись — через ``dashboard.actions.submit_manual_decision``, который
    проверяет статус OPEN и делает upsert существующим репозиторием.

    Raises
    ------
    ValueError
        Если студент не в команде, объём некорректен, обоснование пустое
        или открытого раунда нет.
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None

    if not math.isfinite(quantity) or quantity < 0:
        raise ValueError(
            f"Объём Q должен быть неотрицательным числом, получено: {quantity}."
        )
    if not reasoning.strip():
        raise ValueError(
            "Нужно обоснование: /submit <Q> <текст>. Его оценивает рубрика — "
            "решение без обоснования потеряет половину баллов."
        )

    round_ = await repo.get_open_round(session)
    if round_ is None:
        raise ValueError(
            "Сейчас нет открытого раунда — подождите, когда преподаватель "
            "откроет следующий."
        )
    assert round_.id is not None

    existing = await repo.get_decision(
        session, team_id=team.id, round_id=round_.id
    )
    decision = await submit_manual_decision(
        session,
        team_id=team.id,
        round_id=round_.id,
        quantity=quantity,
        reasoning=reasoning,
    )
    return SubmitOutcome(
        round=round_, team=team, decision=decision, replaced=existing is not None
    )


async def team_status(session: AsyncSession, *, telegram_id: int) -> TeamStatus:
    """Собрать статус для /status: команда, открытый раунд, поданное решение.

    Raises
    ------
    ValueError
        Если студент ещё не вступил в команду.
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None

    round_ = await repo.get_open_round(session)
    decision: Decision | None = None
    if round_ is not None:
        assert round_.id is not None
        decision = await repo.get_decision(
            session, team_id=team.id, round_id=round_.id
        )
    return TeamStatus(team=team, open_round=round_, decision=decision)
