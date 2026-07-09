"""Страница «Раунды»: список, создание, ручной ввод решений, закрытие, дамп.

Вся работа с БД идёт через функции из :mod:`dashboard.actions` (которые, в
свою очередь, зовут только существующие repository-функции и round_service) —
эта страница лишь рисует формы и таблицы.
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
    ResultRow,
    TeacherSummary,
    build_grading_llm,
    close_round_with_results,
    create_and_open_round,
    grade_round_reasoning,
    next_round_number,
    results_table,
    submit_manual_decision,
    teacher_summary,
)
from dashboard.db_runner import run_db  # noqa: E402
from db import repositories as repo  # noqa: E402
from db.enums import EngineMode, RoundStatus  # noqa: E402
from db.models import Round, Team  # noqa: E402
from llm.base import LLMError  # noqa: E402

# Подписи движков в форме и таблице: человекочитаемо, но однозначно мапится
# обратно в EngineMode.
_ENGINE_LABELS: dict[EngineMode, str] = {
    EngineMode.SYMMETRIC: "симметричный (mc раунда для всех)",
    EngineMode.ASYMMETRIC: "асимметричный (пофирменные издержки)",
}


def render_teacher_summary(summary: TeacherSummary) -> None:
    """Короткая сводка явки: кто зашёл через бота и кто уже подал решение."""
    col_teams, col_students, col_decisions = st.columns(3)
    col_teams.metric(
        "Команд в игре", f"{summary.teams_joined} из {summary.teams_total}"
    )
    col_students.metric("Студентов привязано", summary.students_joined)
    if summary.open_round_number is not None:
        col_decisions.metric(
            f"Решений в раунде №{summary.open_round_number}",
            f"{summary.decisions_submitted} из {summary.teams_total}",
        )
    else:
        col_decisions.metric("Решений в открытом раунде", "нет открытого")


def render_rounds_list(rounds: list[Round]) -> None:
    """Показать таблицу всех раундов: номер, метод, статус, создан.

    Поля «дедлайн» в модели Round нет (см. OPEN_QUESTIONS.md), поэтому
    показываем дату создания.
    """
    st.subheader("Раунды")
    if not rounds:
        st.info("Раундов ещё нет — создайте первый в форме ниже.")
        return
    st.table(
        [
            {
                "№": r.number,
                "метод": r.method.value,
                "сложность": r.difficulty,
                "статус": r.status.value,
                "движок": _ENGINE_LABELS[r.engine_mode],
                "создан (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            for r in rounds
        ]
    )


def render_create_form(suggested_number: int) -> None:
    """Форма создания раунда: параметры рынка → create + open одним нажатием."""
    st.subheader("Новый раунд — «Нефть РФ 2013» (парная регрессия)")
    with st.form("create_round"):
        number = st.number_input("Номер раунда", min_value=1, value=suggested_number)
        difficulty = st.number_input("Сложность (1–6)", min_value=1, max_value=6, value=1)
        market_a = st.number_input("a — точка насыщения спроса", min_value=0.1, value=100.0)
        market_b = st.number_input("b — наклон спроса", min_value=0.001, value=1.0)
        market_mc = st.number_input("mc — предельные издержки", min_value=0.0, value=10.0)
        engine_label = st.selectbox(
            "Движок",
            [_ENGINE_LABELS[EngineMode.SYMMETRIC], _ENGINE_LABELS[EngineMode.ASYMMETRIC]],
            help=(
                "Асимметричный движок берёт пофирменные издержки из ground truth "
                "команд (generate_role_views); mc раунда тогда не используется. "
                "Без калиброванных издержек асимметричный раунд не закроется."
            ),
        )
        narrative = st.text_area(
            "Условие кейса",
            value="Нефть РФ 2013: оцените спрос парной регрессией и выберите объём добычи.",
        )
        submitted = st.form_submit_button("Создать и открыть")
    if submitted:
        if market_mc >= market_a:
            st.error("mc должен быть строго меньше a — иначе рынок нежизнеспособен.")
            return
        engine_mode = next(
            mode for mode, label in _ENGINE_LABELS.items() if label == engine_label
        )
        round_ = run_db(
            partial(
                create_and_open_round,
                number=int(number),
                difficulty=int(difficulty),
                market_a=float(market_a),
                market_b=float(market_b),
                market_mc=float(market_mc),
                case_narrative=narrative,
                engine_mode=engine_mode,
            )
        )
        st.success(f"Раунд №{round_.number} создан и открыт (id={round_.id}).")
        st.rerun()


# STUB: заменить на Telegram-бота после MVP, не строить сейчас.
def render_decision_stub(round_: Round, teams: list[Team]) -> None:
    """Форма ручного ввода решения команды за открытый раунд (вместо бота)."""
    st.subheader(f"Внести решение команды — раунд №{round_.number}")
    if not teams:
        st.warning("Команд нет. Засейте их через dev-shell: uv run python -m devshell.seed")
        return
    with st.form(f"decision_{round_.id}"):
        labels = {f"{t.name} ({t.company_name})": t for t in teams}
        chosen = st.selectbox("Команда", list(labels))
        quantity = st.number_input("Объём Q", min_value=0.0, value=0.0)
        reasoning = st.text_area("Обоснование решения (текст команды)")
        submitted = st.form_submit_button("Записать решение")
    if submitted:
        team = labels[chosen]
        assert team.id is not None and round_.id is not None
        run_db(
            partial(
                submit_manual_decision,
                team_id=team.id,
                round_id=round_.id,
                quantity=float(quantity),
                reasoning=reasoning,
            )
        )
        st.success(f"Решение команды «{team.name}» записано (Q={quantity}).")
        st.rerun()


def render_grade_button(round_: Round) -> None:
    """Кнопка «Оценить обоснования»: LLM-грейдинг отдельно от рыночного счёта.

    Не внутри close_round: Groq — сетевой вызов и может упасть; ошибка
    показывается сообщением, рыночные результаты раунда не трогаются.
    Таблица под кнопкой рисуется после обработчика, поэтому после успешной
    оценки она сразу показывает новые rubric score без перезагрузки.
    """
    assert round_.id is not None
    if st.button(
        f"Оценить обоснования раунда №{round_.number} (Groq)",
        key=f"grade_{round_.id}",
    ):
        try:
            llm = build_grading_llm()
            run_db(partial(grade_round_reasoning, round_id=round_.id, llm=llm))
        except (ValueError, LLMError) as exc:
            st.error(f"Оценка не выполнена: {exc}")
        else:
            st.success("Обоснования оценены — колонка rubric score обновлена.")


def render_results(rows: list[ResultRow]) -> None:
    """Сырой дамп результатов: команда | market score | rubric score."""
    if not rows:
        st.info("Результатов нет — раунд закрыт без посчитанных решений.")
        return
    st.table(
        [
            {
                "команда": f"{r.team_name} ({r.company_name})",
                "Q": r.quantity,
                "цена": r.price,
                "market score (прибыль)": r.market_score,
                "rubric score": r.rubric_score,
            }
            for r in rows
        ]
    )


def main() -> None:
    """Собрать страницу: список раундов, форма создания, работа с открытым раундом."""
    st.set_page_config(page_title="Раунды", page_icon="🏁")
    st.title("Управление раундами")

    rounds = run_db(repo.list_rounds)
    teams = run_db(repo.list_teams)

    render_teacher_summary(run_db(teacher_summary))
    render_rounds_list(rounds)
    render_create_form(run_db(next_round_number))

    open_rounds = [r for r in rounds if r.status is RoundStatus.OPEN]
    for round_ in open_rounds:
        st.divider()
        render_decision_stub(round_, teams)
        assert round_.id is not None
        submitted_count = len(run_db(partial(repo.list_decisions_for_round, round_id=round_.id)))
        st.caption(f"Решений подано: {submitted_count} из {len(teams)}")
        if st.button(f"Закрыть раунд №{round_.number} и посчитать", key=f"close_{round_.id}"):
            try:
                rows = run_db(partial(close_round_with_results, round_id=round_.id))
            except ValueError as exc:
                st.error(f"Нельзя закрыть: {exc}")
            else:
                st.success(f"Раунд №{round_.number} закрыт и посчитан.")
                render_results(rows)

    closed_rounds = [r for r in rounds if r.status is RoundStatus.CLOSED]
    if closed_rounds:
        st.divider()
        st.subheader("Результаты закрытых раундов")
        for round_ in closed_rounds:
            assert round_.id is not None
            st.markdown(f"**Раунд №{round_.number}**")
            render_grade_button(round_)
            render_results(run_db(partial(results_table, round_id=round_.id)))


main()
