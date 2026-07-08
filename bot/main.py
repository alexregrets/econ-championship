"""Точка входа бота: ``uv run python -m bot.main``.

Long polling, без webhook — для локального запуска и небольшого турнира
этого достаточно (см. DECISIONS.md №19). Токен берётся из ``BOT_TOKEN``
в ``.env``; без него запуск честно падает с подсказкой, а не молчит.
"""

from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher

from bot.handlers import router
from config import settings
from db.session import init_db


async def main() -> None:
    """Создать схему БД (идемпотентно), собрать диспетчер и начать polling."""
    if not settings.bot_token:
        raise SystemExit(
            "BOT_TOKEN не задан — скопируйте .env.example в .env и вставьте "
            "токен от @BotFather (рыночная логика от бота не зависит)."
        )
    await init_db()

    bot = Bot(token=settings.bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    print("Бот запущен (long polling). Ctrl+C — остановить.")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
