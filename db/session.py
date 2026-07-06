"""Async database engine and session management.

Exposes a lazily-created async engine, an :func:`init_db` to create tables, and
a :func:`get_session` async-generator dependency used by FastAPI routers, the
bot middleware, and the dev-shell.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from config import settings

# Import models for their side effect: registering tables on SQLModel.metadata
# so that init_db() can create them. (Kept local-looking to avoid an unused warning.)
from db import models as _models  # noqa: F401
from db import role_models as _role_models  # noqa: F401

__all__ = ["engine", "init_db", "drop_all", "get_session", "get_session_ctx"]

engine: AsyncEngine = create_async_engine(settings.database_url, echo=False)


async def init_db() -> None:
    """Create all tables that don't yet exist.

    Idempotent — safe to call on every startup. For schema *changes* a real
    migration tool would be needed, but create-all is enough for the MVP.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def drop_all() -> None:
    """Drop every table. Used by the dev-shell "clear database" action and tests."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a transactional async session.

    Usable both as a FastAPI dependency and via ``async with`` in scripts::

        async with get_session_ctx() as session:
            ...

    The session is closed automatically when the generator is exhausted.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


# A reusable context-manager form for scripts, the dev-shell, and services that
# manage their own session lifecycle (rather than receiving one via injection).
get_session_ctx = asynccontextmanager(get_session)
