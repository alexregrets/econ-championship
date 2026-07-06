"""SQLModel-таблицы ролевого («сложного») раунда.

Три сущности поверх существующей схемы (см. DECISIONS.md №9):

``CompanyGroundTruth`` — единый достоверный агрегат на пару (round, team):
    реальные цифры компании в сценарии («Нефть РФ 2013»). Из него генерируются
    все ролевые срезы, поэтому срезы не могут разойтись между собой.
``RoleDataView``       — срез данных, который видит одна роль одной команды:
    общие reference-поля (одинаковы у всех ролей — shared buffer) плюс
    приватный сигнал роли и её доля издержек (сумма долей по команде = 1.0).
``RoleInput``          — ввод конкретной роли (Q-предложение + заметка),
    хранится отдельно для аудита; в движок по-прежнему идёт одно Decision
    на команду от lead-роли.

Существующие таблицы из :mod:`db.models` не меняются.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint

from db.enums import Role

__all__ = ["CompanyGroundTruth", "RoleDataView", "RoleInput"]


def _utcnow() -> datetime:
    """Вернуть текущий момент в UTC (timezone-aware, без deprecated utcnow)."""
    return datetime.now(UTC)


class CompanyGroundTruth(SQLModel, table=True):
    """Единый ground-truth агрегат компании за раунд.

    Одна строка на пару (round, team) — уникальность закреплена констрейнтом.
    Все ролевые срезы (:class:`RoleDataView`) выводятся из этих полей.
    """

    __table_args__ = (
        UniqueConstraint("round_id", "team_id", name="uq_groundtruth_round_team"),
    )

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="round.id", index=True)
    team_id: int = Field(foreign_key="team.id", index=True)
    # Параметры спроса, «известные компании» в сценарии (могут совпадать с
    # параметрами раунда или быть их зашумлённой оценкой).
    demand_a: float
    demand_b: float
    marginal_cost: float
    # Референсный суммарный выпуск отрасли — общий ориентир для всех ролей.
    ref_total_quantity: float
    # Наблюдаемая рыночная цена при референсном выпуске.
    observed_price: float


class RoleDataView(SQLModel, table=True):
    """Срез данных для одной роли одной команды в одном раунде.

    Инварианты согласованности (проверяются тестами и сидером):
    - ``ref_total_quantity`` и ``observed_price`` одинаковы у всех трёх ролей
      команды и равны значениям из :class:`CompanyGroundTruth`;
    - сумма ``cost_share`` по трём ролям команды равна 1.0.
    """

    __table_args__ = (
        UniqueConstraint(
            "ground_truth_id", "role", name="uq_roleview_groundtruth_role"
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    ground_truth_id: int = Field(foreign_key="companygroundtruth.id", index=True)
    role: Role = Field(index=True)
    # Shared buffer: видны всем ролям одинаково.
    ref_total_quantity: float
    observed_price: float
    # Доля издержек, за которую «отвечает» роль; сумма по команде = 1.0.
    cost_share: float
    # Приватный сигнал роли: одно число + человекочитаемое имя показателя.
    private_signal: float
    private_signal_name: str
    # Текст среза для роли (статический или сгенерированный через Groq).
    narrative: str = ""


class RoleInput(SQLModel, table=True):
    """Ввод одной роли: её предложение по выпуску Q и заметка-обоснование.

    Хранится для аудита и для экрана lead-роли («что предложили роли»);
    финальное решение команды остаётся существующим ``Decision``.
    Роль может пересохранить свой ввод — одна строка на (round, team, role).
    """

    __table_args__ = (
        UniqueConstraint(
            "round_id", "team_id", "role", name="uq_roleinput_round_team_role"
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="round.id", index=True)
    team_id: int = Field(foreign_key="team.id", index=True)
    role: Role
    quantity_proposal: float
    note: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
