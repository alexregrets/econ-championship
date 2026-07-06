"""Генерация согласованных ролевых срезов из единого ground truth.

Пилотный сценарий — «Нефть РФ 2013»: параметры рынка берутся из уже
существующего раунда (сидер devshell/seed.py создаёт a=100, b=1, mc=10).
Для каждой команды создаётся один :class:`~db.role_models.CompanyGroundTruth`
и три :class:`~db.role_models.RoleDataView` — по числу ролей.

Согласованность обеспечивается конструкцией, а не проверкой задним числом:
все числа срезов выводятся из одного ground truth (см. DECISIONS.md №12).
Groq по умолчанию не вызывается: нарративы статические; передайте
``llm=GroqClient(...)``, чтобы сгенерировать живые тексты.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, nash_equilibrium
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import Role
from db.role_models import RoleDataView
from llm.base import StructuredLLM

__all__ = [
    "ROLE_COST_SHARES",
    "RoleSlice",
    "build_role_slices",
    "generate_role_views",
]

# Фиксированные доли издержек пилота: финансист «держит» добычу,
# аналитик продаж — логистику/сбыт, маркетолог — продвижение.
# Сумма строго 1.0 — это shared-buffer-инвариант ролевых срезов.
ROLE_COST_SHARES: dict[Role, float] = {
    Role.FINANCIER: 0.55,
    Role.SALES_ANALYST: 0.25,
    Role.MARKETER: 0.20,
}

_NARRATIVE_SYSTEM_PROMPT = (
    "Ты пишешь короткие деловые брифинги для учебного турнира по эконометрике "
    "(сценарий: нефтяная компания РФ, 2013 год). По одному абзацу на роль, "
    "только по переданным цифрам, без выдуманных фактов."
)


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
    params: MarketParameters, n_teams: int, company_name: str
) -> dict[Role, RoleSlice]:
    """Построить три согласованных среза из параметров рынка — чистая функция.

    Приватные сигналы: маркетолог видит точку насыщения спроса ``a``,
    аналитик продаж — суммарный выпуск конкурентов при симметричном Нэше,
    финансист — предельные издержки ``mc``. Все числа выводятся из одного
    набора параметров, поэтому срезы не могут противоречить друг другу.

    Raises
    ------
    ValueError
        Если ``n_teams`` меньше 1 (проброшено из nash_equilibrium).
    """
    q_star = nash_equilibrium(n_teams, params)
    competitors_quantity = q_star * (n_teams - 1)

    return {
        Role.MARKETER: RoleSlice(
            role=Role.MARKETER,
            cost_share=ROLE_COST_SHARES[Role.MARKETER],
            private_signal=params.a,
            private_signal_name="оценка точки насыщения спроса (a)",
            narrative=(
                f"{company_name}, 2013: опрос рынка показывает, что спрос "
                f"иссякает при цене около {params.a:.0f}."
            ),
        ),
        Role.SALES_ANALYST: RoleSlice(
            role=Role.SALES_ANALYST,
            cost_share=ROLE_COST_SHARES[Role.SALES_ANALYST],
            private_signal=competitors_quantity,
            private_signal_name="суммарный выпуск конкурентов (оценка)",
            narrative=(
                f"{company_name}, 2013: по отгрузкам конкуренты суммарно "
                f"выпускают около {competitors_quantity:.1f} единиц."
            ),
        ),
        Role.FINANCIER: RoleSlice(
            role=Role.FINANCIER,
            cost_share=ROLE_COST_SHARES[Role.FINANCIER],
            private_signal=params.marginal_cost,
            private_signal_name="предельные издержки (mc)",
            narrative=(
                f"{company_name}, 2013: себестоимость дополнительной единицы "
                f"держится на уровне {params.marginal_cost:.1f}."
            ),
        ),
    }


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

    Parameters
    ----------
    session:
        Активная сессия БД.
    round_id:
        Раунд, из параметров которого выводится ground truth.
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
        Если раунд не существует или команд нет.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    teams = await repo.list_teams(session)
    if not teams:
        raise ValueError("no teams to generate role views for")

    params = MarketParameters(
        a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
    )
    n_teams = len(teams)
    q_star = nash_equilibrium(n_teams, params)
    ref_total = q_star * n_teams
    observed_price = max(0.0, params.a - params.b * ref_total)

    created: list[RoleDataView] = []
    for team in teams:
        assert team.id is not None  # команды прочитаны из БД
        slices = build_role_slices(params, n_teams, team.company_name)
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
            ref_total_quantity=ref_total,
            observed_price=observed_price,
        )
        assert truth.id is not None
        for role, slice_ in slices.items():
            view = await role_repo.create_role_view(
                session,
                ground_truth_id=truth.id,
                role=role,
                ref_total_quantity=ref_total,
                observed_price=observed_price,
                cost_share=slice_.cost_share,
                private_signal=slice_.private_signal,
                private_signal_name=slice_.private_signal_name,
                narrative=narratives[role],
            )
            created.append(view)
    return created
