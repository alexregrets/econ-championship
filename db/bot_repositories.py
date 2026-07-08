"""Репозитории Telegram-бота: коды команд и привязка студента без роли.

Дополняют :mod:`db.repositories`, не меняя его: единственное, чего не хватало
боту, — операции с :class:`~db.bot_models.TeamJoinCode` и привязка студента к
команде БЕЗ назначения роли (существующий ``assign_student_to_team`` требует
:class:`~db.enums.Role`, а бот ролей не раздаёт — ролевой трек живёт в
role_tui, см. DECISIONS.md №19).
"""

from __future__ import annotations

import secrets

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.bot_models import TeamJoinCode
from db.models import Student, Team
from db.repositories import list_teams

__all__ = [
    "CODE_ALPHABET",
    "CODE_LENGTH",
    "get_join_code_for_team",
    "get_team_by_join_code",
    "ensure_join_codes",
    "bind_student_to_team",
]

# Без визуально похожих символов (0/O, 1/I/L): коды диктуют голосом на паре.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6


def _generate_code(taken: set[str]) -> str:
    """Сгенерировать код, не входящий в ``taken`` (криптослучайный, читаемый)."""
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code not in taken:
            return code


async def get_join_code_for_team(
    session: AsyncSession, team_id: int
) -> TeamJoinCode | None:
    """Вернуть код команды, если он уже сгенерирован."""
    result = await session.exec(
        select(TeamJoinCode).where(TeamJoinCode.team_id == team_id)
    )
    return result.first()


async def get_team_by_join_code(session: AsyncSession, code: str) -> Team | None:
    """Найти команду по коду (регистронезависимо), либо ``None``."""
    normalized = code.strip().upper()
    result = await session.exec(
        select(TeamJoinCode).where(TeamJoinCode.code == normalized)
    )
    join_code = result.first()
    if join_code is None:
        return None
    return await session.get(Team, join_code.team_id)


async def ensure_join_codes(session: AsyncSession) -> dict[int, str]:
    """Убедиться, что у каждой команды есть код; вернуть ``team_id -> код``.

    Идемпотентна: существующие коды не перегенерируются (розданные командам
    бумажки не должны протухать), коды создаются только командам без кода.
    """
    existing = await session.exec(select(TeamJoinCode))
    by_team: dict[int, str] = {jc.team_id: jc.code for jc in existing.all()}
    taken = set(by_team.values())

    for team in await list_teams(session):
        assert team.id is not None  # прочитана из БД
        if team.id in by_team:
            continue
        code = _generate_code(taken)
        taken.add(code)
        session.add(TeamJoinCode(team_id=team.id, code=code))
        by_team[team.id] = code

    await session.commit()
    return by_team


async def bind_student_to_team(
    session: AsyncSession, *, student_id: int, team_id: int
) -> Student:
    """Привязать студента к команде, не трогая его роль.

    Повторный вызов с другой командой перепривязывает студента — студент,
    опечатавшийся в коде, должен суметь вступить в правильную команду.

    Raises
    ------
    ValueError
        Если студент не существует.
    """
    student = await session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student {student_id} not found")
    student.team_id = team_id
    session.add(student)
    await session.commit()
    await session.refresh(student)
    return student
