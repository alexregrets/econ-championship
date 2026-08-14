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
``RoleScore``          — личный KPI роли за раунд и итог студента
    (см. :mod:`core.role_kpi`), считается при закрытии раунда.

Существующие таблицы из :mod:`db.models` не меняются.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint

from db.enums import Role

__all__ = ["CompanyGroundTruth", "RoleDataView", "RoleInput", "RoleScore"]


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
    # Калиброванные пофирменные издержки c_i (implied costs, DECISIONS.md №18)
    # для асимметричного движка. None — у команды нет пофирменной калибровки,
    # асимметричный раунд с ней закрыть нельзя; симметричный движок поле
    # не читает вовсе.
    implied_marginal_cost: float | None = None
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
    # Прогноз рыночной цены — вход KPI аналитика сбыта (core.role_kpi).
    # None у остальных ролей и у аналитика, который прогноз не подал:
    # «прогноза нет» и «прогноз оказался плохим» — разные вещи, и точность
    # в первом случае не считается, а обнуляется явно.
    price_forecast: float | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class RoleScore(SQLModel, table=True):
    """Личный KPI роли за раунд и итоговая оценка студента.

    Пишется при закрытии раунда чистой функцией :func:`core.role_kpi.
    compute_role_kpis` — сохраняются и сырое значение показателя (для разбора),
    и нормированное (для сравнения ролей между собой), и обе составляющие
    итога. Одна строка на (round, team, role); пересчёт раунда её обновляет.
    """

    __table_args__ = (
        UniqueConstraint(
            "round_id", "team_id", "role", name="uq_rolescore_round_team_role"
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    round_id: int = Field(foreign_key="round.id", index=True)
    team_id: int = Field(foreign_key="team.id", index=True)
    role: Role
    # Сырое значение KPI в своих единицах (доля рынка / маржа / точность).
    kpi_raw: float
    kpi_name: str
    # То же значение относительно лучшего в раунде внутри своей роли, 0..1.
    kpi_normalized: float
    # Прибыль команды относительно лучшей в раунде, 0..1.
    team_component: float
    # 0.7 × team_component + 0.3 × kpi_normalized (веса — ScoreWeights).
    total: float
    # Были ли у роли данные для её KPI (для аналитика — подан ли прогноз).
    has_input: bool = True
