"""Операции бота без Telegram: вступление в команду, подача решения, статус.

Функции принимают AsyncSession и не знают ничего про aiogram — тестируются
тем же стилем, что dashboard/actions.py. Валидация «решения принимает только
открытый раунд» НЕ дублируется: подача идёт через существующий
:func:`dashboard.actions.submit_manual_decision` — тот же общий слой, которым
пользуется дашборд (он вынесен из Streamlit именно для переиспользования).

Ошибки для пользователя — ValueError с русским текстом: хендлер показывает
его сообщением, ничего не переформулируя.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from dashboard.actions import (
    DataRoom,
    data_room,
    results_table,
    review_panel,
    submit_manual_decision,
)
from db import bot_repositories as bot_repo
from db import repositories as repo
from db.enums import RoundStatus
from db.models import Decision, Round, Team

__all__ = [
    "BriefOutcome",
    "JoinOutcome",
    "ResultOutcome",
    "SubmitOutcome",
    "TeamStatus",
    "brief_as_plain_text",
    "join_team",
    "submit_decision",
    "team_brief",
    "team_result",
    "team_status",
]


@dataclass(frozen=True)
class JoinOutcome:
    """Результат /join: команда и был ли студент создан впервые."""

    team: Team
    created_student: bool


@dataclass(frozen=True)
class SubmitOutcome:
    """Результат /submit: раунд, команда, решение и было ли оно заменой."""

    round: Round
    team: Team
    decision: Decision
    replaced: bool


@dataclass(frozen=True)
class BriefOutcome:
    """Результат /brief и /data: открытый раунд, команда и её комната данных."""

    round: Round
    team: Team
    room: DataRoom


@dataclass(frozen=True)
class ResultOutcome:
    """Разбор последнего закрытого раунда команды — то, что можно показать ей.

    Истинный наклон, наивная оценка и вердикт детектора сюда не входят: это
    страница преподавателя. Команде показывается, что она ждала и что
    получила — этого достаточно, чтобы понять цену ошибки.
    """

    round: Round
    team: Team
    quantity: float
    actual_price: float
    profit: float
    expected_price: float
    best_response_quantity: float
    best_response_profit: float
    profit_gap: float
    rank: int
    teams_scored: int
    cumulative_profit: float
    br_share: float


@dataclass(frozen=True)
class TeamStatus:
    """Ответ /status: команда, открытый раунд (если есть) и решение в нём."""

    team: Team
    open_round: Round | None
    decision: Decision | None


async def join_team(
    session: AsyncSession, *, telegram_id: int, full_name: str, code: str
) -> JoinOutcome:
    """Привязать Telegram-аккаунт к команде по её коду.

    Новый студент создаётся через существующий ``repo.create_student``;
    уже известный — перепривязывается (опечатка в коде не должна запирать
    студента в чужой команде). Роль не назначается: бот работает только с
    «лёгким» раундом.

    Raises
    ------
    ValueError
        Если код пуст или не соответствует ни одной команде.
    """
    normalized = code.strip()
    if not normalized:
        raise ValueError("Код пустой. Использование: /join <код команды>")
    team = await bot_repo.get_team_by_join_code(session, normalized)
    if team is None:
        raise ValueError("Код команды не найден. Проверьте код у капитана или преподавателя.")
    assert team.id is not None  # прочитана из БД

    student = await repo.get_student_by_telegram_id(session, telegram_id)
    created = student is None
    if student is None:
        student = await repo.create_student(session, telegram_id=telegram_id, full_name=full_name)
    assert student.id is not None
    await bot_repo.bind_student_to_team(session, student_id=student.id, team_id=team.id)
    return JoinOutcome(team=team, created_student=created)


async def _require_team(session: AsyncSession, telegram_id: int) -> Team:
    """Вернуть команду студента или объяснить, что сначала нужен /join."""
    student = await repo.get_student_by_telegram_id(session, telegram_id)
    if student is None or student.team_id is None:
        raise ValueError("Вы ещё не в команде. Сначала вступите: /join <код команды>.")
    team = await repo.get_team(session, student.team_id)
    if team is None:
        raise ValueError(
            "Ваша команда не найдена (база пересоздавалась?). Вступите заново: /join <код команды>."
        )
    return team


async def submit_decision(
    session: AsyncSession, *, telegram_id: int, quantity: float, reasoning: str
) -> SubmitOutcome:
    """Подать (или заменить) решение своей команды в открытый раунд.

    Раунд не передаётся параметром: турнир идёт по одному раунду за раз,
    решение попадает в текущий открытый (``repo.get_open_round``). Сама
    запись — через ``dashboard.actions.submit_manual_decision``, который
    проверяет статус OPEN и делает upsert существующим репозиторием.

    Raises
    ------
    ValueError
        Если студент не в команде, объём некорректен, обоснование пустое
        или открытого раунда нет.
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None

    if not math.isfinite(quantity) or quantity < 0:
        raise ValueError(f"Объём Q должен быть неотрицательным числом, получено: {quantity}.")
    if not reasoning.strip():
        raise ValueError(
            "Нужно обоснование: /submit <Q> <текст>. Его оценивает рубрика — "
            "решение без обоснования потеряет половину баллов."
        )

    round_ = await repo.get_open_round(session)
    if round_ is None:
        raise ValueError(
            "Сейчас нет открытого раунда — подождите, когда преподаватель откроет следующий."
        )
    assert round_.id is not None

    existing = await repo.get_decision(session, team_id=team.id, round_id=round_.id)
    decision = await submit_manual_decision(
        session,
        team_id=team.id,
        round_id=round_.id,
        quantity=quantity,
        reasoning=reasoning,
    )
    return SubmitOutcome(round=round_, team=team, decision=decision, replaced=existing is not None)


async def team_brief(session: AsyncSession, *, telegram_id: int) -> BriefOutcome:
    """Брифинг и данные открытого раунда для команды студента.

    Тот же ``data_room``, что рисует витрина, с карточкой фирмы — издержки
    команды и число фирм. Без открытого раунда данных нет: комната данных
    прошлого раунда команде уже не нужна, а будущего — ещё не существует.

    Raises
    ------
    ValueError
        Студент не в команде; открытого раунда нет; данные раунда не
        собираются (текст причины — от сборщика).
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None
    round_ = await repo.get_open_round(session)
    if round_ is None:
        raise ValueError(
            "Сейчас нет открытого раунда — брифинг и данные появятся, когда "
            "преподаватель откроет следующий."
        )
    assert round_.id is not None
    try:
        room = await data_room(session, round_.id, team_id=team.id)
    except NotImplementedError as exc:
        raise ValueError(f"Данные раунда не собираются: {exc}") from exc
    if room is None:  # раунд только что удалили — гонка, не сценарий
        raise ValueError("Раунд не найден, попробуйте ещё раз.")
    return BriefOutcome(round=round_, team=team, room=room)


def brief_as_plain_text(markdown: str) -> str:
    """Перевести Markdown брифинга в текст для Telegram.

    Telegram не рисует таблицы и заголовки Markdown; шлём как есть — студент
    видит решётки и палки. Здесь: заголовки — прописными, жирный — снят,
    цитата — снята, строки таблицы — «имя (единицы): описание».
    """
    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue  # разделитель шапки таблицы
            if cells[0] in ("Столбец",):
                continue  # шапка
            name, unit, desc = (cells + ["", "", ""])[:3]
            lines.append(f"• {name.strip('`')} ({unit}): {desc}")
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip().upper())
            continue
        if line.startswith(">"):
            line = line.lstrip(">").strip()
        lines.append(line.replace("**", ""))
    return "\n".join(lines).strip()


async def team_result(session: AsyncSession, *, telegram_id: int) -> ResultOutcome:
    """Разбор последнего закрытого раунда, в котором команда подавала решение.

    Числа — из той же панели разбора, что видит преподаватель
    (``dashboard.actions.review_panel``), но команде отдаётся только её
    строка и только безопасные поля: ожидаемая цена, лучший ответ, недобор.

    Raises
    ------
    ValueError
        Студент не в команде; закрытых раундов с её решением ещё нет.
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None
    rounds = [r for r in await repo.list_rounds(session) if r.status is RoundStatus.CLOSED]
    rounds.sort(key=lambda r: r.number, reverse=True)
    for round_ in rounds:
        assert round_.id is not None
        rows = await review_panel(session, round_.id)
        if not rows:
            continue
        mine = next((r for r in rows if r.team_name == team.name), None)
        if mine is None:
            continue
        table = await results_table(session, round_.id)
        # Место — по оценке раунда (доля от лучшего ответа), не по сырой
        # прибыли: в Курно перепроизводитель зарабатывает больше тех, кто
        # считал верно, и по деньгам был бы первым.
        ordered = sorted(rows, key=lambda r: r.br_share, reverse=True)
        rank = next((i + 1 for i, r in enumerate(ordered) if r.team_name == team.name), 0)
        return ResultOutcome(
            round=round_,
            team=team,
            quantity=mine.quantity,
            actual_price=mine.actual_price,
            profit=mine.profit,
            expected_price=mine.expected_price,
            best_response_quantity=mine.best_response_quantity,
            best_response_profit=mine.best_response_profit,
            profit_gap=mine.profit_gap,
            rank=rank,
            teams_scored=len(table),
            cumulative_profit=team.cumulative_profit,
            br_share=mine.br_share,
        )
    raise ValueError(
        "Закрытых раундов с вашим решением ещё нет — разбор появится после первого закрытия."
    )


async def team_status(session: AsyncSession, *, telegram_id: int) -> TeamStatus:
    """Собрать статус для /status: команда, открытый раунд, поданное решение.

    Raises
    ------
    ValueError
        Если студент ещё не вступил в команду.
    """
    team = await _require_team(session, telegram_id)
    assert team.id is not None

    round_ = await repo.get_open_round(session)
    decision: Decision | None = None
    if round_ is not None:
        assert round_.id is not None
        decision = await repo.get_decision(session, team_id=team.id, round_id=round_.id)
    return TeamStatus(team=team, open_round=round_, decision=decision)
