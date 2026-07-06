"""Действия страницы раундов: тонкие обёртки над репозиториями и round_service.

Логика вынесена из Streamlit-страницы сюда, чтобы её можно было тестировать
как обычные асинхронные функции (тем же стилем, что tests/test_round_service.py):
функции принимают AsyncSession и не знают ничего про UI.

Экономика здесь не считается: рынок считает services.round_service поверх
core.market_engine, мы только собираем данные для отображения.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from db import repositories as repo
from db.enums import Method, RoundStatus
from db.models import Decision, Round
from services.round_service import close_round, open_round

__all__ = [
    "ResultRow",
    "next_round_number",
    "create_and_open_round",
    "submit_manual_decision",
    "close_round_with_results",
    "results_table",
]


@dataclass(frozen=True)
class ResultRow:
    """Одна строка сырого дампа результатов раунда.

    ``market_score`` — прибыль команды за раунд из движка Курно.
    ``rubric_score`` — оценка по рубрике; пока LLM-грейдинг не подключён к
    round_service, сервис сохраняет её как 0.0 (см. отчёт по MVP).
    """

    team_name: str
    company_name: str
    quantity: float
    price: float
    market_score: float
    rubric_score: float


async def next_round_number(session: AsyncSession) -> int:
    """Вернуть номер для нового раунда: максимум существующих + 1.

    Нужен форме создания раунда, чтобы профессор не подбирал номер вручную.
    """
    rounds = await repo.list_rounds(session)
    if not rounds:
        return 1
    return max(r.number for r in rounds) + 1


async def create_and_open_round(
    session: AsyncSession,
    *,
    number: int,
    difficulty: int,
    market_a: float,
    market_b: float,
    market_mc: float,
    case_narrative: str,
) -> Round:
    """Создать раунд (черновик) и сразу открыть его для приёма решений.

    Метод зафиксирован как OLS_SIMPLE — по скоупу MVP у нас один сценарий
    («Нефть РФ 2013», парная регрессия). Создание идёт через существующий
    repo.create_round, открытие — через round_service.open_round, чтобы вся
    смена статусов проходила одним и тем же путём, что и в остальном коде.
    """
    round_ = await repo.create_round(
        session,
        number=number,
        method=Method.OLS_SIMPLE,
        difficulty=difficulty,
        market_a=market_a,
        market_b=market_b,
        market_mc=market_mc,
        case_narrative=case_narrative,
        status=RoundStatus.DRAFT,
    )
    assert round_.id is not None  # только что сохранён — id уже присвоен
    await open_round(session, round_.id)
    # Перечитываем, чтобы вернуть объект с актуальным статусом OPEN.
    reloaded = await repo.get_round(session, round_.id)
    assert reloaded is not None
    return reloaded


# STUB: заменить на Telegram-бота после MVP, не строить сейчас.
# Пока бота нет, профессор вносит решения команд вручную с этой страницы.
async def submit_manual_decision(
    session: AsyncSession,
    *,
    team_id: int,
    round_id: int,
    quantity: float,
    reasoning: str,
) -> Decision:
    """Внести (или заменить) решение команды за раунд вручную.

    Повторная отправка той же командой перезаписывает прошлое решение —
    это поведение существующего repo.upsert_decision, мы его не меняем.

    Raises
    ------
    ValueError
        Если раунд не существует или уже не открыт: вносить решения можно
        только в открытый раунд.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    if round_.status is not RoundStatus.OPEN:
        raise ValueError(
            f"round {round_id} is {round_.status.value}, decisions are "
            "accepted only while it is open"
        )
    return await repo.upsert_decision(
        session,
        team_id=team_id,
        round_id=round_id,
        quantity=quantity,
        reasoning=reasoning,
    )


async def close_round_with_results(
    session: AsyncSession, round_id: int
) -> list[ResultRow]:
    """Закрыть раунд через существующий сервис и вернуть таблицу результатов.

    Сам расчёт делает round_service.close_round (движок Курно + сохранение
    Result + обновление cumulative_profit); здесь мы только читаем то, что
    сервис записал, и собираем строки для отображения.
    """
    await close_round(session, round_id)
    return await results_table(session, round_id)


async def results_table(session: AsyncSession, round_id: int) -> list[ResultRow]:
    """Собрать сырой дамп результатов раунда: команда | market | rubric.

    Читает только через repository-функции. Решения без посчитанного Result
    пропускаются (такого не должно быть после close_round, но страница не
    должна падать на полусчитанном раунде).
    """
    decisions = await repo.list_decisions_for_round(session, round_id)
    rows: list[ResultRow] = []
    for decision in decisions:
        assert decision.id is not None  # прочитан из БД — id есть всегда
        result = await repo.get_result_for_decision(session, decision.id)
        if result is None:
            continue
        team = await repo.get_team(session, decision.team_id)
        team_name = team.name if team is not None else f"team {decision.team_id}"
        company = team.company_name if team is not None else ""
        rows.append(
            ResultRow(
                team_name=team_name,
                company_name=company,
                quantity=decision.quantity,
                price=result.price,
                market_score=result.profit,
                rubric_score=result.rubric_score,
            )
        )
    # Сортировка по прибыли — победители сверху, как на лидерборде.
    rows.sort(key=lambda r: r.market_score, reverse=True)
    return rows
