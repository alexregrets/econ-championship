"""SQLModel-таблица кодов приглашения для Telegram-бота.

Связка «студент → команда» новых таблиц не требует: она хранится в
существующем :class:`db.models.Student` (``telegram_id`` там уникален с
самого начала, бот только заполняет ``team_id``). Новая сущность одна:

``TeamJoinCode`` — секретный код команды. Профессор раздаёт коды командам
    заранее (``python -m devshell.team_codes``), студент вводит
    ``/join <код>`` один раз, и бот привязывает его Telegram-аккаунт к
    команде. Один код на команду, коды глобально уникальны.

Существующие таблицы :mod:`db.models` не меняются — та же конвенция, что у
ролевого трека (DECISIONS.md №7).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel

__all__ = ["TeamJoinCode"]


def _utcnow() -> datetime:
    """Вернуть текущий момент в UTC (timezone-aware, без deprecated utcnow)."""
    return datetime.now(UTC)


class TeamJoinCode(SQLModel, table=True):
    """Код, по которому студент вступает в команду через бота.

    Хранится в верхнем регистре; сравнение при /join регистронезависимое
    (ввод нормализуется в репозитории).
    """

    id: int | None = Field(default=None, primary_key=True)
    team_id: int = Field(foreign_key="team.id", unique=True, index=True)
    code: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=_utcnow)
