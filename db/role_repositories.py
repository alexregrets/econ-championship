"""Repository-функции ролевого раунда — тем же паттерном, что db/repositories.py.

Каждая функция берёт AsyncSession, делает одну операцию и коммитит, возвращая
объект с заполненным id. Ничего выше этого слоя не пишет SQL по ролевым таблицам.
"""

from __future__ import annotations

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.enums import Role
from db.role_models import CompanyGroundTruth, RoleDataView, RoleInput

__all__ = [
    # ground truth
    "create_ground_truth",
    "get_ground_truth",
    # role views
    "create_role_view",
    "get_role_view",
    "list_role_views_for_team",
    # role inputs
    "upsert_role_input",
    "get_role_input",
    "list_role_inputs_for_team",
]


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


async def create_ground_truth(
    session: AsyncSession,
    *,
    round_id: int,
    team_id: int,
    demand_a: float,
    demand_b: float,
    marginal_cost: float,
    ref_total_quantity: float,
    observed_price: float,
) -> CompanyGroundTruth:
    """Сохранить ground-truth агрегат компании за раунд."""
    truth = CompanyGroundTruth(
        round_id=round_id,
        team_id=team_id,
        demand_a=demand_a,
        demand_b=demand_b,
        marginal_cost=marginal_cost,
        ref_total_quantity=ref_total_quantity,
        observed_price=observed_price,
    )
    session.add(truth)
    await session.commit()
    await session.refresh(truth)
    return truth


async def get_ground_truth(
    session: AsyncSession, *, round_id: int, team_id: int
) -> CompanyGroundTruth | None:
    """Найти ground truth команды за раунд, если он сгенерирован."""
    result = await session.exec(
        select(CompanyGroundTruth).where(
            CompanyGroundTruth.round_id == round_id,
            CompanyGroundTruth.team_id == team_id,
        )
    )
    return result.first()


# --------------------------------------------------------------------------- #
# Role views
# --------------------------------------------------------------------------- #


async def create_role_view(
    session: AsyncSession,
    *,
    ground_truth_id: int,
    role: Role,
    ref_total_quantity: float,
    observed_price: float,
    cost_share: float,
    private_signal: float,
    private_signal_name: str,
    narrative: str = "",
) -> RoleDataView:
    """Сохранить срез данных одной роли."""
    view = RoleDataView(
        ground_truth_id=ground_truth_id,
        role=role,
        ref_total_quantity=ref_total_quantity,
        observed_price=observed_price,
        cost_share=cost_share,
        private_signal=private_signal,
        private_signal_name=private_signal_name,
        narrative=narrative,
    )
    session.add(view)
    await session.commit()
    await session.refresh(view)
    return view


async def get_role_view(
    session: AsyncSession, *, ground_truth_id: int, role: Role
) -> RoleDataView | None:
    """Найти срез конкретной роли по ground truth."""
    result = await session.exec(
        select(RoleDataView).where(
            RoleDataView.ground_truth_id == ground_truth_id,
            RoleDataView.role == role,
        )
    )
    return result.first()


async def list_role_views_for_team(
    session: AsyncSession, *, round_id: int, team_id: int
) -> list[RoleDataView]:
    """Вернуть все ролевые срезы команды за раунд (для проверок и экрана lead).

    Пустой список — ground truth для команды ещё не сгенерирован.
    """
    truth = await get_ground_truth(session, round_id=round_id, team_id=team_id)
    if truth is None or truth.id is None:
        return []
    result = await session.exec(
        select(RoleDataView).where(RoleDataView.ground_truth_id == truth.id)
    )
    return list(result.all())


# --------------------------------------------------------------------------- #
# Role inputs
# --------------------------------------------------------------------------- #


async def upsert_role_input(
    session: AsyncSession,
    *,
    round_id: int,
    team_id: int,
    role: Role,
    quantity_proposal: float,
    note: str = "",
) -> RoleInput:
    """Записать (или заменить) предложение роли за раунд.

    Роль может передумать, пока раунд открыт, — одна строка на
    (round, team, role), как у Decision на (team, round).
    """
    existing = await get_role_input(
        session, round_id=round_id, team_id=team_id, role=role
    )
    if existing is not None:
        existing.quantity_proposal = quantity_proposal
        existing.note = note
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    role_input = RoleInput(
        round_id=round_id,
        team_id=team_id,
        role=role,
        quantity_proposal=quantity_proposal,
        note=note,
    )
    session.add(role_input)
    await session.commit()
    await session.refresh(role_input)
    return role_input


async def get_role_input(
    session: AsyncSession, *, round_id: int, team_id: int, role: Role
) -> RoleInput | None:
    """Найти ввод конкретной роли за раунд, если он есть."""
    result = await session.exec(
        select(RoleInput).where(
            RoleInput.round_id == round_id,
            RoleInput.team_id == team_id,
            RoleInput.role == role,
        )
    )
    return result.first()


async def list_role_inputs_for_team(
    session: AsyncSession, *, round_id: int, team_id: int
) -> list[RoleInput]:
    """Все предложения ролей команды за раунд — экран агрегации lead-роли."""
    result = await session.exec(
        select(RoleInput).where(
            RoleInput.round_id == round_id,
            RoleInput.team_id == team_id,
        )
    )
    return list(result.all())
