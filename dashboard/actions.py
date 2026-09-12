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
from core.beliefs import recover_beliefs
from core.cases import supported_methods
from core.defence import DefenceDraw, DefenceQuestion, draw_defence
from core.market_engine import MarketParameters, nash_equilibrium
from core.market_events import apply_to_costs, apply_to_demand
from core.rubric_grader import RubricCriterion, grade_submission
from core.rubrics import DEFAULT_RUBRICS, rubric_for_method
from core.trap import Verdict, naive_slope_ratio, trap_verdict
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import EngineMode, Method, Role, RoundStatus
from db.models import Decision, Round
from llm.base import StructuredLLM
from llm.groq_client import GroqClient
from services.dataset_export import (
    Dataset,
    build_round_dataset,
    data_dictionary_markdown,
    to_csv,
    to_xlsx,
)
from services.round_brief import FirmCard, render_team_brief
from services.round_service import (
    asymmetric_costs,
    close_round,
    effective_parameters,
    open_round,
    round_shocks,
)

__all__ = [
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
    # визуализация (read-only): превью датасета, равновесие, история
    "ScenarioDataset",
    "TeamQuantityRow",
    "EquilibriumComparison",
    "RoundHistoryRow",
    "scenario_dataset",
    "equilibrium_comparison",
    "rounds_history",
]

# Пилотная рубрика для парной регрессии: используется, если профессор ещё не
# завёл свой RubricTemplate для метода. Веса в сумме дают 1.0, но грейдер
# нормирует на сумму весов сам, так что это не жёсткое требование.
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


# Подписи методов для формы препода. Порядок — порядок курса (db.enums.Method).
# В форму попадают только методы с готовым кейсом: раунд по методу без данных
# не настраивается (core.cases), и предлагать его в выпадашке нельзя.
_METHOD_LABELS: dict[Method, str] = {
    Method.OLS_SIMPLE: "Парная регрессия спроса",
    Method.OLS_MULTIPLE: "Множественная регрессия и фиктивные переменные",
    Method.MULTICOLLINEARITY: "Мультиколлинеарность",
    Method.HETEROSCEDASTICITY: "Гетероскедастичность",
    Method.AUTOCORRELATION: "Автокорреляция",
    Method.PANEL_DATA: "Панельные данные",
}
METHOD_CHOICES: dict[Method, str] = {
    method: _METHOD_LABELS[method] for method in supported_methods()
}


# Полоса «цена Нэша − издержки» уже этой доли цены — рынок тесный: одна
# ошибка команды в 2× роняет цену к нулю, разбор превращается в бойню.
_NARROW_BAND_SHARE = 0.25


@dataclass(frozen=True)
class MarketPreview:
    """Что получится из параметров формы при текущем числе команд."""

    n_firms: int
    nash_quantity: float  # на фирму
    nash_price: float
    price_band: float  # P* − c
    warnings: tuple[str, ...]


async def market_preview(
    session: AsyncSession, *, market_a: float, market_b: float, market_mc: float
) -> MarketPreview:
    """Предпросмотр рынка и предупреждения перед созданием раунда.

    Два предупреждения, оба — из живого прогона 12.09:

    * параметры совпадают с прошлым раундом — после закрытия команды видят
      равновесие, и следующий раунд с теми же `a`, `b`, `c` решается по
      памяти, а не по данным;
    * рынок тесный для этого числа команд — полоса `P* − c` меньше четверти
      цены, одна ошибка в 2× выносит цену в ноль.

    Предупреждения не блокируют: решение за преподавателем.
    """
    teams = await repo.list_teams(session)
    n = max(len(teams), 1)
    params = MarketParameters(a=market_a, b=market_b, marginal_cost=market_mc)
    q = nash_equilibrium(n, params)
    price = market_a - market_b * q * n
    band = price - market_mc
    warnings: list[str] = []

    previous = await latest_round(session)
    if previous is not None and (
        previous.market_a == market_a
        and previous.market_b == market_b
        and previous.market_mc == market_mc
    ):
        warnings.append(
            f"Параметры совпадают с раундом №{previous.number}. После закрытия "
            "команды видят равновесие — этот раунд они решат по памяти, а не по "
            "данным. Измените хотя бы a или b."
        )
    if price > 0 and band < _NARROW_BAND_SHARE * price:
        warnings.append(
            f"Рынок тесный для {n} команд: цена Нэша {price:.1f}, издержки "
            f"{market_mc:g}, полоса {band:.1f}. Одна ошибка в 2× по объёму уронит "
            "цену к нулю. Поднимите a или снизьте c."
        )
    return MarketPreview(
        n_firms=n,
        nash_quantity=q,
        nash_price=price,
        price_band=band,
        warnings=tuple(warnings),
    )


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
    method: Method = Method.OLS_SIMPLE,
) -> Round:
    """Создать раунд (черновик) и сразу открыть его для приёма решений.

    ``method`` определяет, какая закономерность лежит в данных раунда
    (core.cases). Метод без готового кейса отклоняется здесь, при создании,
    а не в момент выгрузки, когда команды уже ждут данные. Создание идёт
    через существующий repo.create_round, открытие — через
    round_service.open_round, чтобы вся смена статусов проходила одним и тем
    же путём, что и в остальном коде.

    ``engine_mode`` по умолчанию симметричный — поведение существующих
    раундов не меняется. Асимметричный раунд считается по пофирменным
    издержкам из CompanyGroundTruth (их пишет generate_role_views); без них
    close_round честно откажется закрывать раунд.
    """
    if method not in METHOD_CHOICES:
        raise ValueError(
            f"под метод {method.value} кейса ещё нет; доступны: "
            f"{', '.join(m.value for m in METHOD_CHOICES)}"
        )
    # Проверка до записи: open_round откажет и сам, но черновик уже лежал бы в
    # базе и всплывал в списке раундов как «ещё один» — путал бы препода.
    current = await repo.get_open_round(session)
    if current is not None:
        raise ValueError(
            f"раунд №{current.number} ещё открыт — закройте его, прежде чем "
            "открывать новый: бот принимает решения только в один раунд"
        )
    round_ = await repo.create_round(
        session,
        number=number,
        method=method,
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
# Панель разбора препода: во что верила команда, кто попался на ловушку
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReviewRow:
    """Одна команда в разборе закрытого раунда.

    Содержит истинный наклон рынка — это страница препода, командам такой
    строки не показывать (см. :func:`review_panel`: до закрытия — ``None``).
    Знаки разрывов как в :mod:`core.beliefs`: положительное = «плохо».
    """

    team_name: str
    company_name: str
    quantity: float
    best_response_quantity: float
    implied_slope: float
    true_slope: float
    naive_slope: float | None
    expected_price: float
    actual_price: float
    price_gap: float
    profit: float
    best_response_profit: float
    profit_gap: float
    verdict: Verdict


async def review_panel(session: AsyncSession, round_id: int) -> list[ReviewRow] | None:
    """Разбор закрытого раунда: убеждения команд и вердикт детектора.

    ``None``, пока раунд не закрыт или его нет: разбор раскрывает истинные
    параметры, и до закрытия ему на экране делать нечего. Параметры берутся
    **эффективные** — после событий раунда, ровно те, на которых раунд
    считался; издержки в асимметричном раунде — пофирменные, как в
    :func:`services.round_service.compute_round_results`.

    Строки отсортированы по разрыву в прибыли: кого ловушка ударила сильнее,
    тот сверху — с него препод и начинает разбор.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None or round_.status is not RoundStatus.CLOSED:
        return None
    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        return []

    quantities = {str(d.team_id): d.quantity for d in decisions}
    shocks = await round_shocks(session, round_id)
    costs: dict[str, float] | None
    if round_.engine_mode is EngineMode.ASYMMETRIC:
        a, b = apply_to_demand(round_.market_a, round_.market_b, shocks)
        costs = apply_to_costs(
            await asymmetric_costs(session, round_, decisions), shocks, demand_intercept=a
        )
        params = MarketParameters(a=a, b=b, marginal_cost=round_.market_mc)
    else:
        costs = None
        params = effective_parameters(round_, shocks)

    beliefs = recover_beliefs(quantities, params, marginal_costs=costs)
    naive_ratio = naive_slope_ratio(round_.method)
    naive_slope = None if naive_ratio is None else naive_ratio * params.b

    rows: list[ReviewRow] = []
    for decision in decisions:
        belief = beliefs[str(decision.team_id)]
        team = await repo.get_team(session, decision.team_id)
        rows.append(
            ReviewRow(
                team_name=team.name if team is not None else f"team {decision.team_id}",
                company_name=team.company_name if team is not None else "",
                quantity=belief.submitted_quantity,
                best_response_quantity=belief.best_response_quantity,
                implied_slope=belief.implied_slope,
                true_slope=params.b,
                naive_slope=naive_slope,
                expected_price=belief.expected_price,
                actual_price=belief.actual_price,
                price_gap=belief.price_gap,
                profit=belief.profit,
                best_response_profit=belief.best_response_profit,
                profit_gap=belief.profit_gap,
                verdict=trap_verdict(belief.implied_slope, params.b, naive_ratio),
            )
        )
    rows.sort(key=lambda r: r.profit_gap, reverse=True)
    return rows


@dataclass(frozen=True)
class DefenceCard:
    """Жеребьёвка микрозащиты для страницы препода: имя команды, роль, вопрос."""

    team_name: str
    company_name: str
    role: Role
    question: DefenceQuestion
    attempt: int


async def defence_draw(
    session: AsyncSession, round_id: int, *, attempt: int = 0
) -> DefenceCard | None:
    """Кого спрашиваем после закрытия раунда. ``None`` — раунд не закрыт.

    Разыгрываются только команды, подавшие решение: спрашивать тех, кто не
    играл, не о чем. Розыгрыш детерминирован (`core.defence`), поэтому
    кнопку можно нажимать сколько угодно — имя не поменяется; перетяжка —
    только явным ``attempt``.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None or round_.status is not RoundStatus.CLOSED:
        return None
    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        return None
    draw: DefenceDraw = draw_defence(
        round_id,
        round_.method,
        [str(d.team_id) for d in decisions],
        attempt=attempt,
    )
    team = await repo.get_team(session, int(draw.team_id))
    return DefenceCard(
        team_name=team.name if team is not None else f"team {draw.team_id}",
        company_name=team.company_name if team is not None else "",
        role=draw.role,
        question=draw.question,
        attempt=attempt,
    )


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


# --------------------------------------------------------------------------- #
# Визуализация (read-only): превью датасета, «где мы против равновесия»,
# история раундов. Новых таблиц нет — всё читается из уже посчитанного.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScenarioDataset:
    """Сырые данные сценария «Нефть РФ 2013» для превью перед решением.

    Только публичная часть кейса (реальная добыча, цена Urals, среднеотраслевая
    себестоимость) — приватные сигналы ролей и implied-издержки сюда не
    попадают, спойлерить асимметричное равновесие нельзя.
    """

    productions: dict[str, float]  # компания -> добыча 2013, млн т
    revenues: dict[str, float]  # компания -> выручка, млн $ (добыча × цена)
    industry_cost_per_ton: float  # среднеотраслевая полная себестоимость, $/т
    observed_price_per_ton: float  # цена Urals 2013, $/т


@dataclass(frozen=True)
class TeamQuantityRow:
    """Q команды в закрытом раунде против её равновесного q*."""

    team_label: str
    actual_quantity: float
    equilibrium_quantity: float


@dataclass(frozen=True)
class EquilibriumComparison:
    """Итог закрытого раунда против расчётного равновесия Нэша.

    Референс зависит от движка раунда: симметричный Нэш по параметрам
    раунда или асимметричное равновесие по implied-издержкам команд.
    """

    engine_mode: EngineMode
    actual_price: float
    equilibrium_price: float
    actual_total_quantity: float
    equilibrium_total_quantity: float
    teams: list[TeamQuantityRow]


@dataclass(frozen=True)
class RoundHistoryRow:
    """Одна строка истории турнира: закрытый раунд и его рыночный итог."""

    number: int
    engine_mode: EngineMode
    price: float
    total_quantity: float
    avg_profit: float
    decisions: int


async def scenario_dataset(
    session: AsyncSession, round_id: int
) -> ScenarioDataset | None:
    """Публичные сырые данные сценария для превью, если раунд — «Нефть РФ 2013».

    Сценарий распознаётся по командам: у каждой должна быть компания из
    реального датасета добычи (иначе это generic-раунд без превью — ``None``).
    Числа берутся из констант сценария (devshell.role_seed) — это и есть
    датасет для регрессии; в БД пофирменной добычи нет.
    """
    from devshell.role_seed import (
        FULL_COST_2013_USD_PER_TON,
        OIL_PRODUCTION_2013_MLN_T,
        URALS_PRICE_2013_USD_PER_TON,
    )

    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        return None
    teams = await repo.list_teams(session)
    if not teams or any(
        t.company_name not in OIL_PRODUCTION_2013_MLN_T for t in teams
    ):
        return None

    return ScenarioDataset(
        productions=dict(OIL_PRODUCTION_2013_MLN_T),
        revenues={
            company: production * URALS_PRICE_2013_USD_PER_TON
            for company, production in OIL_PRODUCTION_2013_MLN_T.items()
        },
        industry_cost_per_ton=FULL_COST_2013_USD_PER_TON,
        observed_price_per_ton=URALS_PRICE_2013_USD_PER_TON,
    )


@dataclass(frozen=True)
class DataRoom:
    """Всё, что команда получает на руки в раунде: брифинг и выгрузка.

    Собирается из одного и того же :class:`~services.dataset_export.Dataset`,
    чтобы брифинг описывал ровно те столбцы, что лежат в файле. Истина
    (``CompanyGroundTruth``, параметры рынка) сюда не попадает — это держит
    тест на утечку в ``tests/test_dataset_export.py``.
    """

    title: str
    brief_markdown: str
    dictionary_markdown: str
    csv_text: str
    xlsx_bytes: bytes
    filename_stem: str
    observations: int


async def firm_card(session: AsyncSession, round_id: int, team_id: int) -> FirmCard:
    """Карточка фирмы для брифинга: издержки команды и число фирм на рынке.

    Симметричный раунд — общие ``market_mc``; асимметричный — пофирменные
    издержки из ground truth, и без них карточка не собирается: выдать
    команде «издержки неизвестны» значит выдать нерешаемую задачу.

    Raises
    ------
    ValueError
        Раунда или команды нет; в асимметричном раунде у команды нет
        калиброванных издержек.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    team = await repo.get_team(session, team_id)
    if team is None:
        raise ValueError(f"team {team_id} not found")
    n_firms = max(len(await repo.list_teams(session)), 1)
    if round_.engine_mode is EngineMode.ASYMMETRIC:
        truths = await role_repo.list_ground_truths_for_round(session, round_id)
        cost = next(
            (t.implied_marginal_cost for t in truths if t.team_id == team_id), None
        )
        if cost is None:
            raise ValueError(
                f"asymmetric round {round_id}: у команды {team.name} нет "
                "калиброванных издержек — сгенерируйте ролевые срезы"
            )
    else:
        cost = round_.market_mc
    return FirmCard(company_name=team.company_name, marginal_cost=cost, n_firms=n_firms)


async def data_room(
    session: AsyncSession, round_id: int, team_id: int | None = None
) -> DataRoom | None:
    """Комната данных раунда для витрины и бота: ``None``, если раунда нет.

    С ``team_id`` брифинг содержит карточку фирмы (издержки, число фирм) —
    рабочий документ команды. Без него — общее превью.

    Раунд по методу без кейса сюда не доходит — его отсекает
    :func:`create_and_open_round`. Если такой всё же лежит в старой базе,
    ``build_round_dataset`` упадёт ``NotImplementedError`` — и пусть: тихо
    показать пустую комнату хуже, чем громко.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        return None
    dataset: Dataset = await build_round_dataset(session, round_id)
    firm = None if team_id is None else await firm_card(session, round_id, team_id)
    return DataRoom(
        title=dataset.title,
        brief_markdown=render_team_brief(dataset, firm=firm),
        dictionary_markdown=data_dictionary_markdown(dataset),
        csv_text=to_csv(dataset),
        xlsx_bytes=to_xlsx(dataset),
        filename_stem=dataset.filename_stem,
        observations=len(dataset.rows),
    )


async def equilibrium_comparison(
    session: AsyncSession, round_id: int
) -> EquilibriumComparison | None:
    """Итог закрытого раунда против равновесия Нэша — «где мы оказались».

    ``None``, если раунд не закрыт, решений нет или (для асимметричного
    раунда) у какой-то решившей команды нет калиброванных издержек —
    страница тогда просто не рисует блок, не падает.

    Движки вызываются как чистые функции только для расчёта референса;
    ничего не пишется.
    """
    from core.market_engine import MarketParameters, nash_equilibrium
    from core.market_engine_asymmetric import asymmetric_nash_equilibrium

    round_ = await repo.get_round(session, round_id)
    if round_ is None or round_.status is not RoundStatus.CLOSED:
        return None
    decisions = await repo.list_decisions_for_round(session, round_id)
    if not decisions:
        return None

    # Фактическая цена — из любого Result (она общая); нет Result — раунд
    # закрыт в обход сервиса, сравнивать нечего.
    assert decisions[0].id is not None
    first_result = await repo.get_result_for_decision(session, decisions[0].id)
    if first_result is None:
        return None
    actual_total = sum(d.quantity for d in decisions)

    team_labels: dict[int, str] = {}
    for decision in decisions:
        team = await repo.get_team(session, decision.team_id)
        team_labels[decision.team_id] = (
            f"{team.name} ({team.company_name})"
            if team is not None
            else f"team {decision.team_id}"
        )

    if round_.engine_mode is EngineMode.ASYMMETRIC:
        truths = await role_repo.list_ground_truths_for_round(session, round_id)
        costs_by_team = {
            t.team_id: t.implied_marginal_cost
            for t in truths
            if t.implied_marginal_cost is not None
        }
        if any(d.team_id not in costs_by_team for d in decisions):
            return None
        result = asymmetric_nash_equilibrium(
            round_.market_a,
            round_.market_b,
            [costs_by_team[d.team_id] for d in decisions],
        )
        eq_quantities = dict(
            zip((d.team_id for d in decisions), result.quantities, strict=True)
        )
        eq_price = result.price
        eq_total = result.total_quantity
    else:
        params = MarketParameters(
            a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
        )
        symmetric_q = nash_equilibrium(len(decisions), params)
        eq_quantities = {d.team_id: symmetric_q for d in decisions}
        eq_total = symmetric_q * len(decisions)
        eq_price = round_.market_a - round_.market_b * eq_total

    return EquilibriumComparison(
        engine_mode=round_.engine_mode,
        actual_price=first_result.price,
        equilibrium_price=eq_price,
        actual_total_quantity=actual_total,
        equilibrium_total_quantity=eq_total,
        teams=[
            TeamQuantityRow(
                team_label=team_labels[d.team_id],
                actual_quantity=d.quantity,
                equilibrium_quantity=eq_quantities[d.team_id],
            )
            for d in decisions
        ],
    )


async def rounds_history(session: AsyncSession) -> list[RoundHistoryRow]:
    """История закрытых раундов турнира по номерам — для графика препода.

    Отдельного поля «сценарий» у Round нет — историей считаются все закрытые
    раунды с посчитанными результатами (текущий турнир и есть один сценарий).
    Один сыгранный раунд = одна точка; это ожидаемо, фейковых точек не рисуем.
    """
    rows: list[RoundHistoryRow] = []
    for round_ in await repo.list_rounds(session):
        if round_.status is not RoundStatus.CLOSED or round_.id is None:
            continue
        decisions = await repo.list_decisions_for_round(session, round_.id)
        profits: list[float] = []
        price: float | None = None
        for decision in decisions:
            assert decision.id is not None
            result = await repo.get_result_for_decision(session, decision.id)
            if result is None:
                continue
            profits.append(result.profit)
            price = result.price
        if price is None:
            continue  # закрыт без результатов — на график не попадает
        rows.append(
            RoundHistoryRow(
                number=round_.number,
                engine_mode=round_.engine_mode,
                price=price,
                total_quantity=sum(d.quantity for d in decisions),
                avg_profit=sum(profits) / len(profits),
                decisions=len(decisions),
            )
        )
    return rows


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
    """Вернуть рубрику метода; без шаблона в базе — завести из `core.rubrics`.

    Читает RubricTemplate через существующий репозиторий. Если шаблона нет,
    рубрика по умолчанию сохраняется через upsert_rubric_template: профессор
    потом сможет её поправить, а повторные оценки будут читать сохранённую
    версию.

    Raises
    ------
    ValueError
        Если шаблона нет и под метод нет рубрики по умолчанию — выдумывать
        её на ходу нельзя.
    """
    template = await repo.get_rubric_for_method(session, method)
    if template is not None:
        return _CRITERIA_ADAPTER.validate_json(template.criteria_json)
    if method not in DEFAULT_RUBRICS:
        raise ValueError(
            f"для метода {method.value} не задана рубрика — заведите "
            "RubricTemplate прежде чем оценивать"
        )
    rubric = rubric_for_method(method)
    criteria_json = _CRITERIA_ADAPTER.dump_json(rubric).decode("utf-8")
    await repo.upsert_rubric_template(
        session,
        method=method,
        name=f"{_METHOD_LABELS[method]} — рубрика по умолчанию",
        criteria_json=criteria_json,
    )
    return rubric


def grading_reference_notes(round_: Round, *, n_firms: int) -> str:
    """Справка грейдеру: истинные числа раунда, чтобы судить названные.

    Идёт в промпт Groq, студенту не показывается. Без неё критерий «наклон
    назван числом» закрывается любым числом; с ней грейдер сверяет
    названное с истиной и наивной оценкой. Наивная оценка — та, что даёт
    ловушка метода (`core.trap`); где ловушки нет, строка не пишется.
    """
    n = max(n_firms, 1)
    nash_q = (round_.market_a - round_.market_mc) / (round_.market_b * (n + 1))
    nash_price = round_.market_a - round_.market_b * nash_q * n
    lines = [
        f"- True demand line: P = {round_.market_a:g} - {round_.market_b:g} * Q "
        "(Q = total market output).",
        f"- True slope b = {round_.market_b:g}; marginal cost c = {round_.market_mc:g}.",
        f"- Firms on the market: {n}. Symmetric Nash: q_i ≈ {nash_q:.2f}, "
        f"price ≈ {nash_price:.2f}.",
    ]
    ratio = naive_slope_ratio(round_.method)
    if ratio is not None:
        lines.append(
            f"- Naive pooled regression on this dataset recovers slope ≈ "
            f"{ratio * round_.market_b:g} ({ratio:.0%} of the truth) — a student "
            "naming a slope near this value fell into the trap."
        )
    return "\n".join(lines)


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
    notes = grading_reference_notes(round_, n_firms=len(await repo.list_teams(session)))

    for decision in decisions:
        assert decision.id is not None  # прочитан из БД
        existing = await repo.get_result_for_decision(session, decision.id)
        if existing is None:
            raise ValueError(
                f"decision {decision.id} has no market result — закройте "
                "раунд через дашборд, чтобы результаты посчитались"
            )
        grading = await grade_submission(
            decision.reasoning, rubric, llm, reference_notes=notes
        )
        await repo.save_result(
            session,
            decision_id=decision.id,
            price=existing.price,
            profit=existing.profit,
            rubric_score=grading.total_score,
            rubric_breakdown_json=grading.model_dump_json(),
        )

    return await results_table(session, round_id)
