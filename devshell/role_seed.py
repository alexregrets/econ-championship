"""Сценарий «Нефть РФ 2013» на реальных данных + генерация ролевых срезов.

Реальные цифры (источники в комментариях у констант): добыча трёх компаний за
2013 год, среднегодовая цена Urals и полная себестоимость барреля. Демо-рынок
калибруется так, чтобы наблюдаемая точка 2013 (суммарная добыча, цена Urals)
лежала на кривой спроса и была симметричным Нэшем трёх фирм — подробности в
DECISIONS.md №13–15.

Единицы движка: объёмы — млн тонн, деньги — $/т (перевод 7.28 барр/т),
поэтому прибыль получается в млн долларов.

Для каждой команды создаётся один :class:`~db.role_models.CompanyGroundTruth`
и три :class:`~db.role_models.RoleDataView`. Согласованность срезов — по
построению: все числа выводятся из одного набора реальных констант.
Groq по умолчанию не вызывается: нарративы статические; передайте
``llm=GroqClient(...)``, чтобы сгенерировать живые тексты.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters
from core.market_engine_asymmetric import implied_marginal_costs
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import Method, Role, RoundStatus
from db.role_models import RoleDataView
from db.session import drop_all, get_session_ctx, init_db
from devshell.seed import SeedSummary
from llm.base import StructuredLLM

__all__ = [
    "OIL_PRODUCTION_2013_MLN_T",
    "URALS_PRICE_2013_USD_PER_BBL",
    "FULL_COST_2013_USD_PER_BBL",
    "BBL_PER_TON",
    "URALS_PRICE_2013_USD_PER_TON",
    "FULL_COST_2013_USD_PER_TON",
    "TOTAL_PRODUCTION_2013_MLN_T",
    "ROLE_COST_SHARES",
    "RoleSlice",
    "oil_2013_market_parameters",
    "implied_oil_2013_costs",
    "build_role_slices",
    "seed_oil_2013",
    "reset_and_seed_oil_2013",
    "generate_role_views",
]

# --- Реальные данные 2013 года --------------------------------------------- #

# Добыча нефти за 2013 год, млн тонн (ЦДУ ТЭК).
OIL_PRODUCTION_2013_MLN_T: dict[str, float] = {
    "Роснефть": 203.03,
    "Лукойл": 86.923,
    "Сургутнефтегаз": 61.453,
}

# Среднегодовая цена Urals за 2013 год, $/баррель (Минфин РФ).
URALS_PRICE_2013_USD_PER_BBL = 107.88

# Полная себестоимость барреля с учётом НДПИ и капзатрат, $/баррель
# (НЕ голый lifting cost $10–15 — обоснование в DECISIONS.md №13).
FULL_COST_2013_USD_PER_BBL = 50.0

# Стандартный коэффициент перевода для Urals: баррелей в тонне.
BBL_PER_TON = 7.28

# Производные в единицах движка ($/т); прибыль = ($/т) × (млн т) = млн $.
URALS_PRICE_2013_USD_PER_TON = URALS_PRICE_2013_USD_PER_BBL * BBL_PER_TON
FULL_COST_2013_USD_PER_TON = FULL_COST_2013_USD_PER_BBL * BBL_PER_TON
TOTAL_PRODUCTION_2013_MLN_T = sum(OIL_PRODUCTION_2013_MLN_T.values())

# Фиксированные доли ответственности за издержки: финансист «держит» добычу,
# аналитик продаж — логистику/сбыт, маркетолог — продвижение.
# Сумма строго 1.0 — это shared-buffer-инвариант ролевых срезов.
ROLE_COST_SHARES: dict[Role, float] = {
    Role.FINANCIER: 0.55,
    Role.SALES_ANALYST: 0.25,
    Role.MARKETER: 0.20,
}

# Допуск сравнения параметров раунда с параметрами сценария (float-шум).
_PARAMS_TOLERANCE = 1e-6

_NARRATIVE_SYSTEM_PROMPT = (
    "Ты пишешь короткие деловые брифинги для учебного турнира по эконометрике "
    "(сценарий: российская нефтяная компания, 2013 год). По одному абзацу на "
    "роль, только по переданным цифрам, без выдуманных фактов."
)


def oil_2013_market_parameters() -> MarketParameters:
    """Откалибровать линейный спрос P = a − b·Q по наблюдаемой точке 2013.

    Калибровка: точка (суммарная добыча трёх компаний; цена Urals в $/т)
    лежит на кривой спроса и является симметричным Нэшем трёх фирм с
    издержками = полной себестоимости. Для n фирм в симметричном Нэше
    P* = (a + n·c)/(n+1), откуда a = (n+1)·P − n·c, а b = (a − P)/Q —
    чтобы точка попала на кривую. Проверяется тестом: Нэш калиброванного
    рынка воспроизводит фактический суммарный выпуск 2013 года.
    """
    n = len(OIL_PRODUCTION_2013_MLN_T)
    price = URALS_PRICE_2013_USD_PER_TON
    cost = FULL_COST_2013_USD_PER_TON
    a = (n + 1) * price - n * cost
    b = (a - price) / TOTAL_PRODUCTION_2013_MLN_T
    return MarketParameters(a=a, b=b, marginal_cost=cost)


def implied_oil_2013_costs() -> dict[str, float]:
    """Неявные (calibrated) предельные издержки компаний из равновесия 2013, $/т.

    Пофирменной полной себестоимости нужной точности в открытых источниках не
    существует, поэтому числа берутся обратной калибровкой (DECISIONS.md №18):
    при уже зафиксированных a и b (:func:`oil_2013_market_parameters`) из FOC
    асимметричного Курно выводятся c_i, при которых равновесные q_i* совпадают
    с фактической добычей 2013 года. Средняя по трём компаниям по построению
    равна полной себестоимости $50/барр (364 $/т); индивидуальные значения —
    модельные (implied), использовать их как факт отчётности нельзя.
    """
    params = oil_2013_market_parameters()
    companies = list(OIL_PRODUCTION_2013_MLN_T)
    costs = implied_marginal_costs(
        params.a,
        params.b,
        [OIL_PRODUCTION_2013_MLN_T[company] for company in companies],
    )
    return dict(zip(companies, costs, strict=True))


class _RoleNarratives(BaseModel):
    """Структурированный ответ LLM: по одному брифинг-абзацу на роль."""

    marketer: str
    sales_analyst: str
    financier: str


@dataclass(frozen=True)
class RoleSlice:
    """Чистое описание среза одной роли до записи в БД (удобно тестировать)."""

    role: Role
    cost_share: float
    private_signal: float
    private_signal_name: str
    narrative: str


def build_role_slices(
    company_name: str,
    *,
    own_production: float,
    params: MarketParameters,
    total_production: float,
) -> dict[Role, RoleSlice]:
    """Построить три согласованных среза из реальных данных — чистая функция.

    Приватные сигналы: маркетолог видит точку насыщения спроса ``a`` ($/т),
    аналитик продаж — реальную суммарную добычу конкурентов (млн т),
    финансист — полную себестоимость ($/т). Все числа выводятся из одного
    набора данных, поэтому срезы не могут противоречить друг другу.

    Parameters
    ----------
    company_name:
        Название компании для текстов срезов.
    own_production:
        Реальная добыча самой компании за 2013 год, млн т.
    params:
        Калиброванные параметры рынка (см. :func:`oil_2013_market_parameters`).
    total_production:
        Реальная суммарная добыча всех компаний-участниц, млн т.
    """
    competitors_production = total_production - own_production

    return {
        Role.MARKETER: RoleSlice(
            role=Role.MARKETER,
            cost_share=ROLE_COST_SHARES[Role.MARKETER],
            private_signal=params.a,
            private_signal_name="оценка точки насыщения спроса (a), $/т",
            narrative=(
                f"{company_name}, 2013: средняя цена Urals — "
                f"${URALS_PRICE_2013_USD_PER_BBL:.2f}/барр. "
                f"(≈{URALS_PRICE_2013_USD_PER_TON:.0f} $/т); по опросам рынка "
                f"спрос иссякает при цене около {params.a:.0f} $/т."
            ),
        ),
        Role.SALES_ANALYST: RoleSlice(
            role=Role.SALES_ANALYST,
            cost_share=ROLE_COST_SHARES[Role.SALES_ANALYST],
            private_signal=competitors_production,
            private_signal_name="суммарная добыча конкурентов 2013, млн т",
            narrative=(
                f"{company_name}, 2013: наша добыча — {own_production:.1f} млн т; "
                f"конкуренты суммарно добывают около "
                f"{competitors_production:.1f} млн т (данные ЦДУ ТЭК)."
            ),
        ),
        Role.FINANCIER: RoleSlice(
            role=Role.FINANCIER,
            cost_share=ROLE_COST_SHARES[Role.FINANCIER],
            private_signal=params.marginal_cost,
            private_signal_name="полная себестоимость (с НДПИ и капзатратами), $/т",
            narrative=(
                f"{company_name}, 2013: полная себестоимость с НДПИ и "
                f"капзатратами — ${FULL_COST_2013_USD_PER_BBL:.0f}/барр. "
                f"({params.marginal_cost:.0f} $/т)."
            ),
        ),
    }


_ROLES = (Role.MARKETER, Role.SALES_ANALYST, Role.FINANCIER)


async def seed_oil_2013(session: AsyncSession) -> SeedSummary:
    """Засеять пилот ролевого трека: 3 реальные компании и открытый раунд.

    Команды — Роснефть, Лукойл и Сургутнефтегаз (по три студента в трёх
    ролях), раунд — «Нефть РФ 2013» с калиброванными параметрами рынка.
    Возвращает тот же :class:`~devshell.seed.SeedSummary`, что и обычный
    сидер, чтобы остальной dev-shell работал без изменений.
    """
    params = oil_2013_market_parameters()

    team_ids: list[int] = []
    student_count = 0
    telegram_seq = 2000  # не пересекается с диапазоном обычного сидера (1000+)

    for index, company in enumerate(OIL_PRODUCTION_2013_MLN_T):
        team = await repo.create_team(
            session, name=f"Команда {index + 1}", company_name=company
        )
        assert team.id is not None
        team_ids.append(team.id)
        for role in _ROLES:
            telegram_seq += 1
            student = await repo.create_student(
                session,
                telegram_id=telegram_seq,
                full_name=f"{company} {role.value}",
            )
            assert student.id is not None
            await repo.assign_student_to_team(
                session, student_id=student.id, team_id=team.id, role=role
            )
            student_count += 1

    round_ = await repo.create_round(
        session,
        number=1,
        method=Method.OLS_SIMPLE,
        difficulty=1,
        market_a=params.a,
        market_b=params.b,
        market_mc=params.marginal_cost,
        case_narrative=(
            "Нефть РФ 2013: по реальным данным о добыче и цене Urals оцените "
            "спрос парной регрессией и выберите объём добычи (млн т)."
        ),
        status=RoundStatus.OPEN,
    )
    assert round_.id is not None
    return SeedSummary(
        team_ids=team_ids, student_count=student_count, round_id=round_.id
    )


async def reset_and_seed_oil_2013() -> SeedSummary:
    """Пересоздать схему и засеять пилот «Нефть РФ 2013» с нуля."""
    await drop_all()
    await init_db()
    async with get_session_ctx() as session:
        return await seed_oil_2013(session)


async def _narratives_via_llm(
    llm: StructuredLLM,
    company_name: str,
    slices: dict[Role, RoleSlice],
) -> dict[Role, str]:
    """Сгенерировать нарративы через Groq — только по цифрам из срезов."""
    facts = "\n".join(
        f"- {s.role.value}: {s.private_signal_name} = {s.private_signal:.2f}, "
        f"доля издержек = {s.cost_share:.2f}"
        for s in slices.values()
    )
    response = await llm.structured_completion(
        _NARRATIVE_SYSTEM_PROMPT,
        f"Компания: {company_name}. Данные ролей:\n{facts}",
        _RoleNarratives,
    )
    return {
        Role.MARKETER: response.marketer,
        Role.SALES_ANALYST: response.sales_analyst,
        Role.FINANCIER: response.financier,
    }


async def generate_role_views(
    session: AsyncSession,
    round_id: int,
    *,
    llm: StructuredLLM | None = None,
) -> list[RoleDataView]:
    """Создать ground truth и три ролевых среза для каждой команды раунда.

    Данные — реальный сценарий «Нефть РФ 2013»: у каждой команды должна быть
    компания из :data:`OIL_PRODUCTION_2013_MLN_T`, а параметры раунда обязаны
    совпадать с калиброванными (:func:`oil_2013_market_parameters`) — иначе
    срезы описывали бы не тот рынок, в котором команды реально играют.
    Раунд пилота создаётся согласованным через :func:`seed_oil_2013`.

    Parameters
    ----------
    session:
        Активная сессия БД.
    round_id:
        Раунд, для которого создаются срезы.
    llm:
        Опциональный Groq-клиент для генерации нарративов. По умолчанию
        ``None`` — тексты статические, сети нет (режим dev-shell).

    Returns
    -------
    list[RoleDataView]
        Все созданные срезы (3 × число команд).

    Raises
    ------
    ValueError
        Если раунд не существует, команд нет, у команды нет реальных данных
        2013 года или параметры раунда не совпадают со сценарием.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    teams = await repo.list_teams(session)
    if not teams:
        raise ValueError("no teams to generate role views for")

    unknown = [
        t.company_name for t in teams
        if t.company_name not in OIL_PRODUCTION_2013_MLN_T
    ]
    if unknown:
        raise ValueError(
            "нет реальных данных 2013 года для команд: "
            + ", ".join(sorted(unknown))
            + " — ролевой пилот сеется через seed_oil_2013"
        )

    params = oil_2013_market_parameters()
    mismatched = (
        abs(round_.market_a - params.a) > _PARAMS_TOLERANCE
        or abs(round_.market_b - params.b) > _PARAMS_TOLERANCE
        or abs(round_.market_mc - params.marginal_cost) > _PARAMS_TOLERANCE
    )
    if mismatched:
        raise ValueError(
            f"параметры раунда {round_id} не совпадают со сценарием «Нефть РФ "
            "2013» — создайте раунд через seed_oil_2013, чтобы срезы описывали "
            "тот же рынок"
        )

    created: list[RoleDataView] = []
    for team in teams:
        assert team.id is not None  # команды прочитаны из БД
        own_production = OIL_PRODUCTION_2013_MLN_T[team.company_name]
        slices = build_role_slices(
            team.company_name,
            own_production=own_production,
            params=params,
            total_production=TOTAL_PRODUCTION_2013_MLN_T,
        )
        if llm is not None:
            narratives = await _narratives_via_llm(llm, team.company_name, slices)
        else:
            narratives = {role: s.narrative for role, s in slices.items()}

        truth = await role_repo.create_ground_truth(
            session,
            round_id=round_id,
            team_id=team.id,
            demand_a=params.a,
            demand_b=params.b,
            marginal_cost=params.marginal_cost,
            ref_total_quantity=TOTAL_PRODUCTION_2013_MLN_T,
            observed_price=URALS_PRICE_2013_USD_PER_TON,
        )
        assert truth.id is not None
        for role, slice_ in slices.items():
            view = await role_repo.create_role_view(
                session,
                ground_truth_id=truth.id,
                role=role,
                ref_total_quantity=TOTAL_PRODUCTION_2013_MLN_T,
                observed_price=URALS_PRICE_2013_USD_PER_TON,
                cost_share=slice_.cost_share,
                private_signal=slice_.private_signal,
                private_signal_name=slice_.private_signal_name,
                narrative=narratives[role],
            )
            created.append(view)
    return created
