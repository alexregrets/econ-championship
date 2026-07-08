"""Коды вступления команд: сгенерировать недостающие и напечатать таблицу.

Запуск: ``uv run python -m devshell.team_codes``. Профессор раздаёт коды
командам, студенты вводят их боту через ``/join <код>``. Повторный запуск
ничего не перегенерирует (ensure_join_codes идемпотентна) — уже розданные
коды остаются в силе.
"""

from __future__ import annotations

import asyncio

from db import repositories as repo
from db.bot_repositories import ensure_join_codes
from db.session import get_session_ctx, init_db


async def ensure_and_list() -> list[tuple[str, str, str]]:
    """Вернуть таблицу (команда, компания, код), создав недостающие коды."""
    await init_db()
    async with get_session_ctx() as session:
        codes = await ensure_join_codes(session)
        teams = await repo.list_teams(session)
        return [
            (team.name, team.company_name, codes[team.id])
            for team in teams
            if team.id is not None
        ]


if __name__ == "__main__":  # pragma: no cover
    rows = asyncio.run(ensure_and_list())
    if not rows:
        print("Команд нет — сначала засейте базу (devshell.seed или role_tui).")
    else:
        # ASCII-стрелка: консоль Windows (cp1251) не кодирует U+2192.
        width = max(len(f"{name} ({company})") for name, company, _ in rows)
        for name, company, code in rows:
            label = f"{name} ({company})"
            print(f"{label:<{width}}  ->  /join {code}")
