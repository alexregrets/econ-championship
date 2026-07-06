"""Операции ролевой сессии: «войти в роль», предложить Q, агрегировать, what-if.

Слой между ролевым TUI (:mod:`devshell.role_tui`) и данными: всё, что TUI
показывает и делает по кнопкам, реализовано здесь обычными асинхронными и
чистыми функциями — их тестируют без Textual и без сети.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession

from core.market_engine import MarketParameters, compute_cournot_round
from core.market_engine_roles import aggregate_role_proposals
from db import repositories as repo
from db import role_repositories as role_repo
from db.enums import Role, RoundStatus
from db.models import Decision
from db.role_models import RoleDataView, RoleInput

__all__ = [
    "RoleView",
    "LeadOverview",
    "enter_role",
    "submit_role_proposal",
    "lead_overview",
    "what_if_profit",
    "commit_lead_decision",
]


@dataclass(frozen=True)
class RoleView:
    """Что видит участник, вошедший в конкретную роль конкретной команды."""

    team_name: str
    company_name: str
    role: Role
    slice_: RoleDataView
    my_proposal: RoleInput | None


@dataclass(frozen=True)
class LeadOverview:
    """Экран lead-роли: предложения ролей и агрегат по равным весам."""

    proposals: dict[Role, float]
    aggregated_quantity: float | None  # None — предложений ещё нет


async def enter_role(
    session: AsyncSession, *, round_id: int, team_id: int, role: Role
) -> RoleView:
    """«Войти» в роль: вернуть только её срез данных и её же прошлый ввод.

    Ничего чужого функция не отдаёт — приватные сигналы других ролей
    в возвращаемой структуре отсутствуют по построению.

    Raises
    ------
    ValueError
        Если команда не существует или срезы для раунда не сгенерированы.
    """
    team = await repo.get_team(session, team_id)
    if team is None:
        raise ValueError(f"team {team_id} not found")
    truth = await role_repo.get_ground_truth(
        session, round_id=round_id, team_id=team_id
    )
    if truth is None or truth.id is None:
        raise ValueError(
            f"role views for round {round_id} / team {team_id} are not "
            "generated yet — run the role seeder first"
        )
    slice_ = await role_repo.get_role_view(
        session, ground_truth_id=truth.id, role=role
    )
    if slice_ is None:
        raise ValueError(f"no data slice for role {role.value}")
    my_proposal = await role_repo.get_role_input(
        session, round_id=round_id, team_id=team_id, role=role
    )
    return RoleView(
        team_name=team.name,
        company_name=team.company_name,
        role=role,
        slice_=slice_,
        my_proposal=my_proposal,
    )


async def submit_role_proposal(
    session: AsyncSession,
    *,
    round_id: int,
    team_id: int,
    role: Role,
    quantity: float,
    note: str = "",
) -> RoleInput:
    """Сохранить предложение роли (можно перезаписывать, пока раунд открыт).

    Raises
    ------
    ValueError
        Если раунд не существует, не открыт или Q отрицателен.
    """
    if quantity < 0:
        raise ValueError(f"quantity must be >= 0, got {quantity}")
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    if round_.status is not RoundStatus.OPEN:
        raise ValueError(
            f"round {round_id} is {round_.status.value}; proposals are "
            "accepted only while it is open"
        )
    return await role_repo.upsert_role_input(
        session,
        round_id=round_id,
        team_id=team_id,
        role=role,
        quantity_proposal=quantity,
        note=note,
    )


async def lead_overview(
    session: AsyncSession, *, round_id: int, team_id: int
) -> LeadOverview:
    """Собрать экран lead-роли: кто что предложил и равновзвешенный агрегат."""
    inputs = await role_repo.list_role_inputs_for_team(
        session, round_id=round_id, team_id=team_id
    )
    proposals = {i.role: i.quantity_proposal for i in inputs}
    if not proposals:
        return LeadOverview(proposals={}, aggregated_quantity=None)
    aggregated = aggregate_role_proposals(
        {role.value: q for role, q in proposals.items()}
    )
    return LeadOverview(proposals=proposals, aggregated_quantity=aggregated)


def what_if_profit(
    params: MarketParameters, my_quantity: float, other_quantities: list[float]
) -> tuple[float, float]:
    """Чистый what-if: цена и моя прибыль при данных объёмах всех команд.

    Считает через существующий движок (никакой собственной экономики):
    моя команда — ``"me"``, остальные — синтетические id. Возвращает пару
    ``(price, my_profit)``.
    """
    decisions = {"me": my_quantity} | {
        f"other_{i}": q for i, q in enumerate(other_quantities)
    }
    results = compute_cournot_round(decisions, params)
    me = results["me"]
    return me.price, me.profit


async def commit_lead_decision(
    session: AsyncSession,
    *,
    round_id: int,
    team_id: int,
    quantity: float,
    reasoning: str,
) -> Decision:
    """Зафиксировать финальное решение команды от имени lead-роли.

    В движок идёт одно Decision на команду — ровно как в симметричном
    раунде; вводы ролей остаются в RoleInput для аудита.

    Raises
    ------
    ValueError
        Если раунд не существует или уже не открыт.
    """
    round_ = await repo.get_round(session, round_id)
    if round_ is None:
        raise ValueError(f"round {round_id} not found")
    if round_.status is not RoundStatus.OPEN:
        raise ValueError(
            f"round {round_id} is {round_.status.value}; the final decision "
            "can be committed only while it is open"
        )
    return await repo.upsert_decision(
        session,
        team_id=team_id,
        round_id=round_id,
        quantity=quantity,
        reasoning=reasoning,
    )
