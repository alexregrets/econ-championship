"""Синхронный мост между Streamlit и асинхронным слоем БД.

Streamlit выполняет страницу как обычный синхронный скрипт сверху вниз,
а весь слой db/ и services/ у нас асинхронный. Поэтому на каждое действие
создаём свежий движок и сессию и выполняем корутину через asyncio.run:
соединения aiosqlite привязаны к event loop'у, в котором созданы, а
asyncio.run каждый раз создаёт новый loop — переиспользовать глобальный
движок из db.session между кликами было бы нельзя.

Для однопользовательского инструмента профессора цена «новый движок на
каждый клик» пренебрежимо мала.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from config import settings

# Импорт ради побочного эффекта: регистрирует таблицы в SQLModel.metadata,
# чтобы create_all создал схему при первом запуске дашборда.
from db import models as _models  # noqa: F401

__all__ = ["run_db"]


def run_db[T](action: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Выполнить асинхронное действие с БД из синхронного кода Streamlit.

    Принимает функцию «сессия → корутина» (обычно ``functools.partial`` от
    функции из :mod:`dashboard.actions`), открывает свежий движок по
    ``DATABASE_URL`` из настроек, гарантирует существование схемы и
    возвращает результат корутины.
    """

    async def _runner() -> T:
        engine = create_async_engine(settings.database_url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            async with AsyncSession(engine, expire_on_commit=False) as session:
                return await action(session)
        finally:
            await engine.dispose()

    return asyncio.run(_runner())
