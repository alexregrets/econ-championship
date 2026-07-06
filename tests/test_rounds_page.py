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
