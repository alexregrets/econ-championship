"""Смоук-тест самой Streamlit-страницы раундов через streamlit.testing.

Проверяем, что скрипт dashboard/pages/01_rounds.py исполняется без
исключений против временной пустой базы (а не против рабочего файла
econ_tournament.db). Логика кнопок покрыта отдельно в
tests/test_dashboard_actions.py — здесь только «страница рисуется».
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import settings

_PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "01_rounds.py"


@pytest.fixture
def tmp_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменить DATABASE_URL на временный файл, чтобы тест не трогал рабочую БД."""
    db_file = tmp_path / "test_dashboard.db"
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{db_file.as_posix()}"
    )


def test_rounds_page_renders_without_exception(tmp_database: None) -> None:
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception, f"страница упала: {at.exception}"
    # На пустой базе страница должна предложить создать первый раунд.
    assert any("Раундов ещё нет" in str(info.value) for info in at.info)


def test_rounds_page_renders_review_panel_for_closed_round(
    tmp_database: None,
) -> None:
    """Закрытый раунд по режимному сдвигу: панель разбора рисуется, вердикт есть."""
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel
    from sqlmodel.ext.asyncio.session import AsyncSession

    from dashboard.actions import (
        close_round_with_results,
        create_and_open_round,
        submit_manual_decision,
    )
    from db import repositories as repo
    from db.enums import Method

    async def _seed() -> None:
        engine = create_async_engine(settings.database_url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            async with AsyncSession(engine, expire_on_commit=False) as session:
                ids = []
                for i in range(3):
                    team = await repo.create_team(
                        session, name=f"T{i}", company_name=f"C{i}"
                    )
                    ids.append(team.id)
                round_ = await create_and_open_round(
                    session,
                    number=1,
                    difficulty=2,
                    market_a=100.0,
                    market_b=1.0,
                    market_mc=10.0,
                    case_narrative="",
                    method=Method.OLS_MULTIPLE,
                )
                for team_id, q in zip(ids, (22.5, 22.5, 45.0 / 0.7), strict=True):
                    assert team_id is not None and round_.id is not None
                    await submit_manual_decision(
                        session, team_id=team_id, round_id=round_.id, quantity=q, reasoning=""
                    )
                assert round_.id is not None
                await close_round_with_results(session, round_.id)
        finally:
            await engine.dispose()

    asyncio.run(_seed())
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception, f"страница упала: {at.exception}"
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "Разбор раунда №1" in markdown_text
    assert at.table, "таблицы разбора нет"
