"""Пароль на страницы преподавателя.

Витрина и дашборд — одно Streamlit-приложение на одном порту. Без замка
студент, получивший ссылку на витрину, открывает соседнюю страницу и видит
форму раунда с истинными ``a``, ``b``, ``c``, результаты до объявления и
панель разбора с наклоном рынка — то есть ответ на задачу. Это не гипотеза,
а прямое следствие того, как Streamlit строит меню: все страницы из
``pages/`` видны всем.

Пароль — ``DASHBOARD_PASSWORD`` в ``.env``. Пустой пароль оставляет страницы
открытыми и **громко** об этом говорит на каждой из них: удобно на ноутбуке
преподавателя без сети, недопустимо на сервере. Сессия помнит успешный
ввод в ``st.session_state`` — до перезагрузки вкладки.

Сравнение — ``hmac.compare_digest``, чтобы не отдавать длину и префикс
пароля по времени ответа; для учебного сервера это перестраховка, но она
бесплатная.
"""

from __future__ import annotations

import hmac

import streamlit as st

from config import settings

__all__ = ["check_password", "require_teacher"]

_SESSION_KEY = "teacher_authenticated"
_OPEN_WARNING = (
    "Пароль преподавателя не задан (DASHBOARD_PASSWORD в .env пуст) — "
    "эта страница открыта всем, кто знает адрес. На сервере так нельзя."
)


def check_password(candidate: str, expected: str) -> bool:
    """Совпадает ли введённый пароль с ожидаемым.

    Пустой ожидаемый пароль ничего не пропускает: «замок выключен» решается
    выше, в :func:`require_teacher`, а не тем, что пустая строка подходит.
    """
    if not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def require_teacher() -> None:
    """Остановить рендер страницы, пока преподаватель не ввёл пароль.

    Вызывать первым делом в ``main()`` страницы после ``set_page_config``.
    Без пароля в настройках — предупреждение и пропуск; с паролем — форма
    ввода и ``st.stop()`` до успешного ввода.
    """
    expected = settings.dashboard_password
    if not expected:
        st.warning(_OPEN_WARNING)
        return
    if st.session_state.get(_SESSION_KEY):
        return

    st.subheader("Вход для преподавателя")
    candidate = st.text_input("Пароль", type="password", key="teacher_password_input")
    if candidate and check_password(candidate, expected):
        st.session_state[_SESSION_KEY] = True
        st.rerun()
    if candidate:
        st.error("Пароль не подошёл.")
    st.stop()
