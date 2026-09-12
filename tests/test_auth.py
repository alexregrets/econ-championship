"""Замок на страницах преподавателя (dashboard/auth.py).

Смысл замка: витрина и дашборд — одно приложение, и без пароля студент
видит истинные параметры рынка. Проверяем и чистую функцию сравнения, и
саму страницу через streamlit.testing: с паролем в настройках она не
рендерит управление раундами, пока пароль не введён.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import settings
from dashboard.auth import check_password

_PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "01_rounds.py"


@pytest.fixture
def tmp_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "test_auth.db"
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{db_file.as_posix()}"
    )


def test_check_password_matches_exactly() -> None:
    assert check_password("s3cret", "s3cret")
    assert not check_password("s3cret ", "s3cret")
    assert not check_password("", "s3cret")


def test_empty_expected_password_never_passes() -> None:
    """Пустой ожидаемый пароль — не «любой подходит», а «замок выключен»,
    и это решается выше; сама проверка пустоту не пропускает."""
    assert not check_password("", "")
    assert not check_password("anything", "")


def test_page_is_locked_until_password_entered(
    tmp_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "dashboard_password", "s3cret")
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception, f"страница упала: {at.exception}"
    titles = " ".join(str(t.value) for t in at.title)
    assert "Управление раундами" not in titles
    assert at.text_input, "поля ввода пароля нет"

    at.text_input[0].input("wrong").run()
    assert any("не подошёл" in str(e.value) for e in at.error)
    assert "Управление раундами" not in " ".join(str(t.value) for t in at.title)

    at.text_input[0].input("s3cret").run()
    assert not at.exception, f"страница упала после входа: {at.exception}"
    assert "Управление раундами" in " ".join(str(t.value) for t in at.title)


def test_page_warns_loudly_when_password_unset(
    tmp_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "dashboard_password", "")
    at = AppTest.from_file(str(_PAGE), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("DASHBOARD_PASSWORD" in str(w.value) for w in at.warning)
    assert "Управление раундами" in " ".join(str(t.value) for t in at.title)


def test_student_view_has_no_lock(
    tmp_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Витрина остаётся открытой при заданном пароле — это страница студентов."""
    monkeypatch.setattr(settings, "dashboard_password", "s3cret")
    page = _PAGE.with_name("02_student_view.py")
    at = AppTest.from_file(str(page), default_timeout=30)
    at.run()
    assert not at.exception
    assert not at.text_input
    assert "витрина" in " ".join(str(t.value) for t in at.title).lower()
