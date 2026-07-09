"""Действия страниц дашборда: тонкие обёртки над репозиториями и round_service.

Логика вынесена из Streamlit-страниц сюда, чтобы её можно было тестировать
как обычные асинхронные функции (тем же стилем, что tests/test_round_service.py):
функции принимают AsyncSession и не знают ничего про UI. Этим же слоем
пользуется Telegram-бот (/submit) — валидация живёт в одном месте.

Экономика здесь не считается: рынок считает services.round_service поверх
core.market_engine, мы только собираем данные для отображения.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter
from sqlmodel.ext.asyncio.session import AsyncSession

from config import settings
from core.rubric_grader import RubricCriterion, grade_submission
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import EngineMode, Method, Role, RoundStatus
from db.models import Decision, Round
from llm.base import StructuredLLM
from llm.groq_client import GroqClient
from services.round_service import close_round, open_round

__all__ = [
    "DEFAULT_OLS_SIMPLE_RUBRIC",
    "ResultRow",
    "next_round_number",
    "create_and_open_round",
    "submit_manual_decision",
    "close_round_with_results",
    "results_table",
    "ensure_rubric_for_method",
    "grade_round_reasoning",
    "build_grading_llm",
    # студенческая витрина (read-only) и сводка препода
    "MarketBrief",
    "RoleProgressRow",
    "TeamProgress",
    "TeacherSummary",
    "latest_round",
    "market_brief",
    "team_role_progress",
    "teacher_summary",
]

# Пилотная рубрика для парной регрессии: используется, если профессор ещё не
# завёл свой RubricTemplate для метода. Веса в сумме дают 1.0, но грейдер
# нормирует на сумму весов сам, так что это не жёсткое требование.
DEFAULT_OLS_SIMPLE_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        id="specification",
        description=(
            "Указана спецификация парной регрессии: что зависимая переменная, "
            "что объясняющая (например, спрос от цены)."
        ),
        weight=0.3,
    ),
    RubricCriterion(
        id="interpretation",
        description=(
            "Коэффициент наклона интерпретирован по знаку и смыслу "
            "(как изменение фактора влияет на спрос/цену)."
        ),
        weight=0.3,
    ),
    RubricCriterion(
        id="quantity_link",
        description=(
            "Выбранный объём Q обоснован оценкой спроса или ожидаемой цены, "
            "а не назван произвольно."
        ),
        weight=0.3,
    ),
    RubricCriterion(
        id="fit_check",
        description=(
            "Упомянута проверка качества модели: R², значимость коэффициентов "
            "или анализ остатков."
        ),
        weight=0.1,
    ),
]

_CRITERIA_ADAPTER: TypeAdapter[list[RubricCriterion]] = TypeAdapter(
    list[RubricCriterion]
)


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
    engine_mode: EngineMode = EngineMode.SYMMETRIC,
) -> Round:
    """Создать раунд (черновик) и сразу открыть его для приёма решений.

    Метод зафиксирован как OLS_SIMPLE — по скоупу MVP у нас один сценарий
    («Нефть РФ 2013», парная регрессия). Создание идёт через существующий
    repo.create_round, открытие — через round_service.open_round, чтобы вся
    смена статусов проходила одним и тем же путём, что и в остальном коде.

    ``engine_mode`` по умолчанию симметричный — поведение существующих
    раундов не меняется. Асимметричный раунд считается по пофирменным
    издержкам из CompanyGroundTruth (их пишет generate_role_views); без них
    close_round честно откажется закрывать раунд.
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
        engine_mode=engine_mode,
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


# --------------------------------------------------------------------------- #
# Студенческая витрина (read-only) и сводка препода
# --------------------------------------------------------------------------- #

# Порядок ролей на витрине — тот же, что в сидерах.
_VIEW_ROLES = (Role.MARKETER, Role.SALES_ANALYST, Role.FINANCIER)


@dataclass(frozen=True)
class MarketBrief:
    """Общая (не приватная) рыночная картина раунда из ground truth.

    Оба поля — shared buffer ролевых срезов: одинаковы у всех ролей и команд,
    поэтому их можно показывать публично, не спойлеря приватные сигналы.
    """

    ref_total_quantity: float
    observed_price: float


@dataclass(frozen=True)
class RoleProgressRow:
    """Статус одной роли команды: подала ли предложение и (после фиксации
    lead'ом) само предложение. До фиксации числа скрыты — не спойлерим."""

    role: Role
    submitted: bool
    quantity: float | None
    note: str | None


@dataclass(frozen=True)
class TeamProgress:
    """Прогресс команды в раунде для витрины."""

    lead_locked: bool  # финальное Decision уже подано lead'ом
    roles: list[RoleProgressRow]


@dataclass(frozen=True)
class TeacherSummary:
    """Короткая сводка для страницы препода: явка и активность."""

    teams_total: int
    teams_joined: int  # команд с хотя бы одним привязанным студентом
    students_joined: int  # студентов с заполненным team_id
    open_round_number: int | None  # None — открытого раунда нет
    decisions_submitted: int  # решений в открытом раунде (0, если его нет)


async def latest_round(session: AsyncSession) -> Round | None:
    """Раунд для витрины: открытый, иначе последний по номеру, иначе None."""
    open_round_ = await repo.get_open_round(session)
    if open_round_ is not None:
        return open_round_
    rounds = await repo.list_rounds(session)
    return rounds[-1] if rounds else None


async def market_brief(session: AsyncSession, round_id: int) -> MarketBrief | None:
    """Общая рыночная картина раунда, если ролевые срезы сгенерированы.

    Берётся из первого ground truth раунда: reference-поля одинаковы у всех
    команд по инварианту генератора срезов. ``None`` — срезы не генерировались
    (обычный симметричный раунд без ролевого трека), витрина покажет только
    нарратив раунда.
    """
    truths = await role_repo.list_ground_truths_for_round(session, round_id)
    if not truths:
        return None
    return MarketBrief(
        ref_total_quantity=truths[0].ref_total_quantity,
        observed_price=truths[0].observed_price,
    )


async def team_role_progress(
    session: AsyncSession, *, round_id: int, team_id: int
) -> TeamProgress:
    """Кто из ролей команды уже подал предложение (RoleInput) в раунде.

    Числа предложений раскрываются только после фиксации решения lead'ом
    (существует Decision команды за раунд) — до этого витрина показывает
    лишь факт подачи, чтобы не спойлерить обсуждение внутри команды.
    """
    decision = await repo.get_decision(session, team_id=team_id, round_id=round_id)
    lead_locked = decision is not None
    inputs = await role_repo.list_role_inputs_for_team(
        session, round_id=round_id, team_id=team_id
    )
    by_role = {i.role: i for i in inputs}

    rows: list[RoleProgressRow] = []
    for role in _VIEW_ROLES:
        role_input = by_role.get(role)
        if role_input is not None and lead_locked:
            quantity: float | None = role_input.quantity_proposal
            note: str | None = role_input.note or None
        else:
            quantity = None
            note = None
        rows.append(
            RoleProgressRow(
                role=role,
                submitted=role_input is not None,
                quantity=quantity,
                note=note,
            )
        )
    return TeamProgress(lead_locked=lead_locked, roles=rows)


async def teacher_summary(session: AsyncSession) -> TeacherSummary:
    """Сводка для препода: сколько команд/студентов зашло и подало решения.

    «Зашли» = у студента заполнен team_id (бот делает это в /join; студенты
    из dev-сидеров тоже считаются — различить их по данным нельзя, да и не
    нужно: сводка отвечает на вопрос «кто уже в игре»). Читает только
    существующие repository-функции.
    """
    teams = await repo.list_teams(session)
    students = await repo.list_students(session)
    joined = [s for s in students if s.team_id is not None]
    joined_team_ids = {s.team_id for s in joined}

    open_round_ = await repo.get_open_round(session)
    decisions_submitted = 0
    open_round_number: int | None = None
    if open_round_ is not None:
        assert open_round_.id is not None  # прочитан из БД
        open_round_number = open_round_.number
        decisions_submitted = len(
            await repo.list_decisions_for_round(session, open_round_.id)
        )

    return TeacherSummary(
        teams_total=len(teams),
        teams_joined=len(joined_team_ids),
        students_joined=len(joined),
        open_round_number=open_round_number,
        decisions_submitted=decisions_submitted,
    )


def build_grading_llm() -> StructuredLLM:
    """Собрать Groq-клиент для грейдинга из настроек.

    Вынесено в отдельную фабрику, чтобы страница не знала про Groq напрямую,
    а проверочные сценарии могли подменить её фейковым LLM.

    Raises
    ------
    ValueError
        Если GROQ_API_KEY не задан в окружении/.env (сообщение — своё,
        понятное профессору, а не голое «api_key must not be empty»).
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY не задан — добавьте ключ в .env, чтобы оценивать "
            "обоснования (рыночные результаты от этого не зависят)"
        )
    return GroqClient(settings.groq_api_key)


async def ensure_rubric_for_method(
    session: AsyncSession, method: Method
) -> list[RubricCriterion]:
    """Вернуть рубрику метода; для OLS_SIMPLE без шаблона — завести пилотную.

    Читает RubricTemplate через существующий репозиторий. Если шаблона нет и
    метод — парная регрессия (наш единственный сценарий), пилотная рубрика
    сохраняется в БД через upsert_rubric_template: профессор потом сможет её
    поправить, а повторные оценки будут читать уже сохранённую версию.

    Raises
    ------
    ValueError
        Если шаблона нет и метод не OLS_SIMPLE (рубрики других методов —
        вне скоупа, молча выдумывать их нельзя).
    """
    template = await repo.get_rubric_for_method(session, method)
    if template is not None:
        return _CRITERIA_ADAPTER.validate_json(template.criteria_json)
    if method is not Method.OLS_SIMPLE:
        raise ValueError(
            f"для метода {method.value} не задана рубрика — заведите "
            "RubricTemplate прежде чем оценивать"
        )
    criteria_json = _CRITERIA_ADAPTER.dump_json(DEFAULT_OLS_SIMPLE_RUBRIC).decode(
        "utf-8"
    )
    await repo.upsert_rubric_template(
        session,
        method=method,
        name="Парная регрессия — пилотная рубрика",
        criteria_json=criteria_json,
    )
    return list(DEFAULT_OLS_SIMPLE_RUBRIC)


async def grade_round_reasoning(
    session: AsyncSession, round_id: int, llm: StructuredLLM
) -> list[ResultRow]:
    """Оценить обоснования всех решений закрытого раунда по рубрике.

    Отдельная операция, НЕ часть close_round: вызов LLM сетевой и может
    упасть, а рыночный результат от него зависеть не должен. Для каждого
    Decision раунда вызывается существующий grade_submission; рыночные
    цена/прибыль в Result сохраняются как были, обновляются только
    rubric_score и rubric_breakdown_json (полный GradingResult для аудита).
    Повторный запуск просто перезаписывает оценки — операция идемпотентна.

    Returns
    -------
    list[ResultRow]
        Обновлённый дамп результатов раунда.

    Raises
    ------
    ValueError
        Если раунд не существует, ещё не закрыт, не имеет решений или у
        решения нет посчитанного Result (раунд закрыт в обход сервиса).
    LLMError
        Проброшено из клиента, если Groq недоступен или вернул мусор;
        страница показывает это как ошибку, не роняя раунд.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    if round_.status is not RoundStatus.CLOSED:
        raise ValueError(
            f"round {round_id} is {round_.status.value}; оценивать обоснования "
            "можно только после закрытия раунда"
        )
    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        raise ValueError(f"round {round_id} has no decisions to grade")

    rubric = await ensure_rubric_for_method(session, round_.method)

    for decision in decisions:
        assert decision.id is not None  # прочитан из БД
        existing = await repo.get_result_for_decision(session, decision.id)
        if existing is None:
            raise ValueError(
                f"decision {decision.id} has no market result — закройте "
                "раунд через дашборд, чтобы результаты посчитались"
            )
        grading = await grade_submission(decision.reasoning, rubric, llm)
        await repo.save_result(
            session,
            decision_id=decision.id,
            price=existing.price,
            profit=existing.profit,
            rubric_score=grading.total_score,
            rubric_breakdown_json=grading.model_dump_json(),
        )

    return await results_table(session, round_id)
