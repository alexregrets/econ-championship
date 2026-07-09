"""Студенческая витрина: read-only взгляд на текущий раунд.

Публичная страница того же Streamlit-приложения (без логина — внутренний
учебный инструмент). Показывает статус раунда, нарратив сценария, общую
рыночную картину и прогресс выбранной команды. Подача решений ЗДЕСЬ не
работает — решения принимает только Telegram-бот (/join, /submit); это
осознанное решение, чтобы не дублировать канал подачи.

Числа предложений ролей скрыты до фиксации решения lead'ом (см.
``dashboard.actions.team_role_progress``) — витрина не должна спойлерить
обсуждение внутри команды. Приватные сигналы ролей не показываются вовсе.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import streamlit as st

# Streamlit кладёт в sys.path папку самого скрипта, а не корень проекта,
# поэтому добавляем корень вручную — иначе `import db` не найдётся.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.actions import (  # noqa: E402
    MarketBrief,
    TeamProgress,
    latest_round,
    market_brief,
    team_role_progress,
)
from dashboard.db_runner import run_db  # noqa: E402
from db import repositories as repo  # noqa: E402
from db.enums import RoundStatus  # noqa: E402
from db.models import Round, Team  # noqa: E402

_STATUS_LABELS: dict[RoundStatus, str] = {
    RoundStatus.DRAFT: "черновик (ещё не открыт)",
    RoundStatus.OPEN: "открыт — решения принимаются",
    RoundStatus.CLOSED: "закрыт — решения больше не принимаются",
}

_ROLE_LABELS: dict[str, str] = {
    "marketer": "Маркетолог",
    "sales_analyst": "Аналитик продаж",
    "financier": "Финансист",
}


def render_round_header(round_: Round) -> None:
    """Статус и нарратив раунда — то, что видят все команды одинаково."""
    st.subheader(f"Раунд №{round_.number}")
    if round_.status is RoundStatus.OPEN:
        st.success(f"Статус: {_STATUS_LABELS[round_.status]}")
    else:
        st.info(f"Статус: {_STATUS_LABELS[round_.status]}")
    if round_.case_narrative:
        st.markdown("**Сценарий:**")
        st.markdown(round_.case_narrative)


def render_market_brief(brief: MarketBrief | None) -> None:
    """Общая рыночная картина из ролевых срезов (shared buffer, без приватного)."""
    if brief is None:
        return
    st.markdown("**Рыночная ситуация (общие данные):**")
    col_q, col_p = st.columns(2)
    col_q.metric("Референсный выпуск отрасли, млн т", f"{brief.ref_total_quantity:.1f}")
    col_p.metric("Наблюдаемая цена, $/т", f"{brief.observed_price:.1f}")
    st.caption(
        "Приватные сигналы ролей сюда не попадают — их каждая роль видит "
        "только в своём брифинге."
    )


def render_team_progress(progress: TeamProgress) -> None:
    """Кто из ролей подал предложение; числа — только после фиксации lead'ом."""
    if progress.lead_locked:
        st.markdown("Lead зафиксировал решение команды — предложения ролей открыты.")
    else:
        st.markdown(
            "Решение команды ещё не зафиксировано lead'ом — показываем только "
            "факт подачи, без чисел."
        )
    st.table(
        [
            {
                "роль": _ROLE_LABELS.get(row.role.value, row.role.value),
                "предложение подано": "да" if row.submitted else "нет",
                "Q (после фиксации)": (
                    f"{row.quantity:.2f}" if row.quantity is not None else "—"
                ),
                "заметка": row.note if row.note is not None else "—",
            }
            for row in progress.roles
        ]
    )


def render_how_to_submit() -> None:
    """Инструкция подачи решения — только через Telegram-бота."""
    st.divider()
    st.subheader("Как подать решение")
    st.markdown(
        "Решения принимает **только Telegram-бот** — на этой странице подать "
        "решение нельзя, она только для просмотра.\n\n"
        "1. Получите код команды у преподавателя.\n"
        "2. В боте отправьте `/join <код>` — привяжетесь к своей команде.\n"
        "3. Lead команды отправляет `/submit <объём Q> <обоснование>` — одно "
        "финальное решение на команду за раунд (повторный /submit до закрытия "
        "раунда заменяет предыдущее).\n"
        "4. `/status` — проверить, принято ли решение."
    )


def main() -> None:
    """Собрать витрину: раунд, рынок, прогресс команды, инструкция."""
    st.set_page_config(page_title="Витрина турнира", page_icon="👀")
    st.title("Эконометрический турнир — витрина")
    st.caption("Только просмотр: решения подаются через Telegram-бота.")

    round_ = run_db(latest_round)
    if round_ is None:
        st.info("Раундов ещё нет — турнир не начался.")
        render_how_to_submit()
        return

    render_round_header(round_)
    assert round_.id is not None
    render_market_brief(run_db(partial(market_brief, round_id=round_.id)))

    st.divider()
    st.subheader("Прогресс команды")
    teams: list[Team] = run_db(repo.list_teams)
    if not teams:
        st.info("Команд ещё нет.")
    else:
        labels = {f"{t.name} ({t.company_name})": t for t in teams}
        chosen = st.selectbox("Команда", list(labels))
        team = labels[chosen]
        assert team.id is not None
        progress = run_db(
            partial(team_role_progress, round_id=round_.id, team_id=team.id)
        )
        render_team_progress(progress)

    render_how_to_submit()


main()
