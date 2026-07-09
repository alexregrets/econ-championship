"""Смоук-тест студенческой витрины (dashboard/pages/02_student_view.py).

Как tests/test_rounds_page.py: страница должна отрисоваться без исключений
против временной базы. Логика покрыта в tests/test_student_view_actions.py —
здесь проверяем только «страница рисуется» на пустой и на засеянной базе.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from streamlit.testing.v1 import AppTest

from config import settings
from devshell.role_seed import seed_oil_2013

_PAGE = (
    Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "02_student_view.py"
)


@pytest.fixture
def tmp_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Подменить DATABASE_URL на временный файл, чтобы тест не трогал рабочую БД."""
    db_file = tmp_path / "test_student_view.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    return url


def _seed_oil(url: str) -> None:
    """Засеять пилот «Нефть РФ 2013» во временную базу перед рендером."""

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            async with AsyncSession(engine, expire_on_commit=False) as session:
                await seed_oil_2013(session)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_student_view_renders_on_empty_db(tmp_database: str) -> None:
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception, f"страница упала: {at.exception}"
    assert any("Раундов ещё нет" in str(info.value) for info in at.info)


def test_student_view_renders_seeded_round(tmp_database: str) -> None:
    _seed_oil(tmp_database)
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception, f"страница упала: {at.exception}"
    # Открытый раунд виден, команда по умолчанию выбрана, инструкция на месте.
    assert any("открыт" in str(s.value) for s in at.success)
    assert at.selectbox
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "/join" in markdown_text and "/submit" in markdown_text
