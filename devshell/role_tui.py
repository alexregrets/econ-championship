"""Textual-оболочка ролевого раунда: локально «войти в роль» и играть за неё.

Запуск::

    uv run python -m devshell.role_tui

Возможности (та же философия, что devshell/tui.py — без Telegram и без Groq):

- засеять базу и сгенерировать ролевые срезы для открытого раунда;
- войти в конкретную роль конкретной команды и увидеть ТОЛЬКО её срез;
- предложить свой Q с заметкой (сохраняется в RoleInput для аудита);
- экраном lead-роли посмотреть предложения всех ролей, равновзвешенный
  агрегат и what-if (цена/прибыль, если остальные команды сыграют Нэш);
- зафиксировать финальное решение команды (одно Decision — как в
  симметричном раунде).

Вся содержательная логика — в :mod:`devshell.role_session` (покрыта тестами);
здесь только виджеты и вывод.
"""

from __future__ import annotations

from core.market_engine import MarketParameters, nash_equilibrium
from db import repositories as repo
from db.enums import Role
from db.session import get_session_ctx, init_db
from devshell.role_seed import generate_role_views
from devshell.role_session import (
    commit_lead_decision,
    enter_role,
    lead_overview,
    submit_role_proposal,
    what_if_profit,
)
from devshell.seed import reset_and_seed

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Footer, Header, Input, RichLog
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "textual is required for the role dev-shell. Install with: uv add textual"
    ) from exc

# Подписи кнопок ролей — кнопка «входит» в соответствующую роль.
_ROLE_BUTTONS: dict[str, Role] = {
    "role_marketer": Role.MARKETER,
    "role_analyst": Role.SALES_ANALYST,
    "role_financier": Role.FINANCIER,
}


class RoleShellApp(App[None]):
    """Локальная многоролевая оболочка ролевого раунда."""

    TITLE = "Econ Championship — Role Shell"
    CSS = """
    #actions, #role_row, #input_row { height: auto; padding: 0 1; }
    Button { margin: 0 1; }
    Input { width: 24; margin: 0 1; }
    RichLog { border: round $accent; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        # Текущий контекст «кто я»: команда и роль, выбранные пользователем.
        self._team_id: int | None = None
        self._role: Role | None = None

    def compose(self) -> ComposeResult:
        """Собрать три ряда управления и журнал вывода."""
        yield Header()
        with Horizontal(id="actions"):
            yield Button("Seed + срезы", id="seed_roles", variant="warning")
            yield Button("Статус", id="status")
        with Horizontal(id="role_row"):
            yield Input(placeholder="id команды", id="team_input")
            yield Button("Маркетолог", id="role_marketer", variant="primary")
            yield Button("Аналитик", id="role_analyst", variant="primary")
            yield Button("Финансист", id="role_financier", variant="primary")
        with Horizontal(id="input_row"):
            yield Input(placeholder="Q", id="q_input")
            yield Input(placeholder="заметка", id="note_input")
            yield Button("Предложить Q", id="propose", variant="success")
            yield Button("Lead: обзор", id="lead_view")
            yield Button("Lead: зафиксировать Q", id="lead_commit", variant="error")
        with VerticalScroll():
            yield RichLog(highlight=True, markup=True, id="log")
        yield Footer()

    async def on_mount(self) -> None:
        """Создать схему БД (если её нет) и поприветствовать."""
        await init_db()
        self._write(
            "[bold]Role shell готов.[/] «Seed + срезы» → id команды → кнопка роли."
        )

    # -- helpers ---------------------------------------------------------- #

    def _write(self, message: str) -> None:
        """Дописать строку в журнал."""
        self.query_one("#log", RichLog).write(message)

    def _read_team_id(self) -> int | None:
        """Прочитать id команды из поля ввода; None и сообщение — если мусор."""
        raw = self.query_one("#team_input", Input).value.strip()
        if not raw.isdigit():
            self._write("[red]Введите числовой id команды.[/]")
            return None
        return int(raw)

    def _read_quantity(self) -> float | None:
        """Прочитать Q из поля ввода; None и сообщение — если не число."""
        raw = self.query_one("#q_input", Input).value.strip().replace(",", ".")
        try:
            quantity = float(raw)
        except ValueError:
            self._write("[red]Введите числовой Q.[/]")
            return None
        return quantity

    async def _open_round_id(self) -> int | None:
        """Найти открытый раунд; None и сообщение — если его нет."""
        async with get_session_ctx() as session:
            round_ = await repo.get_open_round(session)
        if round_ is None or round_.id is None:
            self._write("[red]Нет открытого раунда.[/] Сначала «Seed + срезы».")
            return None
        return round_.id

    # -- button dispatch -------------------------------------------------- #

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Развести нажатия кнопок по действиям."""
        button_id = event.button.id or ""
        if button_id in _ROLE_BUTTONS:
            await self._action_enter_role(_ROLE_BUTTONS[button_id])
            return
        handlers = {
            "seed_roles": self._action_seed_roles,
            "status": self._action_status,
            "propose": self._action_propose,
            "lead_view": self._action_lead_view,
            "lead_commit": self._action_lead_commit,
        }
        handler = handlers.get(button_id)
        if handler is not None:
            await handler()

    # -- actions ---------------------------------------------------------- #

    async def _action_seed_roles(self) -> None:
        """Пересоздать базу, засеять команды/раунд и сгенерировать срезы."""
        summary = await reset_and_seed()
        async with get_session_ctx() as session:
            views = await generate_role_views(session, summary.round_id)
        self._write(
            f"[green]Засеяно[/] {len(summary.team_ids)} команд, раунд "
            f"id={summary.round_id}, ролевых срезов: {len(views)}."
        )

    async def _action_status(self) -> None:
        """Показать команды с id — чтобы было что вводить в поле команды."""
        async with get_session_ctx() as session:
            teams = await repo.list_teams(session)
        if not teams:
            self._write("[yellow]Команд нет — сначала «Seed + срезы».[/]")
            return
        for team in teams:
            self._write(
                f"  id={team.id}: {team.name} ({team.company_name}) "
                f"cumulative={team.cumulative_profit:.2f}"
            )

    async def _action_enter_role(self, role: Role) -> None:
        """Войти в роль: показать только её срез данных и её прошлый ввод."""
        team_id = self._read_team_id()
        if team_id is None:
            return
        round_id = await self._open_round_id()
        if round_id is None:
            return
        try:
            async with get_session_ctx() as session:
                view = await enter_role(
                    session, round_id=round_id, team_id=team_id, role=role
                )
        except ValueError as exc:
            self._write(f"[red]Не вошли в роль:[/] {exc}")
            return
        self._team_id, self._role = team_id, role
        s = view.slice_
        self._write(
            f"[bold]Вы — {role.value}[/] команды «{view.team_name}» "
            f"({view.company_name})."
        )
        self._write(f"  {s.narrative}")
        self._write(
            f"  общий ориентир: Q отрасли ≈ {s.ref_total_quantity:.2f}, "
            f"цена ≈ {s.observed_price:.2f}; ваша доля издержек: {s.cost_share:.2f}"
        )
        self._write(f"  приватный сигнал — {s.private_signal_name}: {s.private_signal:.2f}")
        if view.my_proposal is not None:
            self._write(
                f"  ваше прошлое предложение: Q={view.my_proposal.quantity_proposal}"
            )

    async def _action_propose(self) -> None:
        """Сохранить предложение текущей роли (Q + заметка) в RoleInput."""
        if self._team_id is None or self._role is None:
            self._write("[red]Сначала войдите в роль (кнопки ролей).[/]")
            return
        quantity = self._read_quantity()
        if quantity is None:
            return
        round_id = await self._open_round_id()
        if round_id is None:
            return
        note = self.query_one("#note_input", Input).value.strip()
        try:
            async with get_session_ctx() as session:
                await submit_role_proposal(
                    session,
                    round_id=round_id,
                    team_id=self._team_id,
                    role=self._role,
                    quantity=quantity,
                    note=note,
                )
        except ValueError as exc:
            self._write(f"[red]Предложение отклонено:[/] {exc}")
            return
        self._write(
            f"[green]Записано:[/] {self._role.value} предлагает Q={quantity}."
        )

    async def _action_lead_view(self) -> None:
        """Экран lead-роли: предложения, агрегат и what-if против Нэша."""
        team_id = self._team_id if self._team_id is not None else self._read_team_id()
        if team_id is None:
            return
        round_id = await self._open_round_id()
        if round_id is None:
            return
        async with get_session_ctx() as session:
            overview = await lead_overview(
                session, round_id=round_id, team_id=team_id
            )
            round_ = await repo.get_round(session, round_id)
            teams = await repo.list_teams(session)
        assert round_ is not None
        if not overview.proposals:
            self._write("[yellow]Роли ещё ничего не предложили.[/]")
            return
        for role, quantity in overview.proposals.items():
            self._write(f"  {role.value}: Q={quantity}")
        assert overview.aggregated_quantity is not None
        aggregated = overview.aggregated_quantity
        self._write(f"[bold]Агрегат (равные веса): Q={aggregated:.2f}[/]")

        # What-if: остальные команды играют симметричный Нэш.
        params = MarketParameters(
            a=round_.market_a, b=round_.market_b, marginal_cost=round_.market_mc
        )
        q_star = nash_equilibrium(len(teams), params)
        others = [q_star] * (len(teams) - 1)
        price, profit = what_if_profit(params, aggregated, others)
        self._write(
            f"  what-if (конкуренты по Нэшу q*={q_star:.2f}): "
            f"цена={price:.2f}, ваша прибыль={profit:.2f}"
        )

    async def _action_lead_commit(self) -> None:
        """Зафиксировать финальный Q команды (из поля Q) как решение lead-роли."""
        team_id = self._team_id if self._team_id is not None else self._read_team_id()
        if team_id is None:
            return
        quantity = self._read_quantity()
        if quantity is None:
            return
        round_id = await self._open_round_id()
        if round_id is None:
            return
        note = self.query_one("#note_input", Input).value.strip()
        reasoning = note or "[lead-role: финальное решение из role shell]"
        try:
            async with get_session_ctx() as session:
                await commit_lead_decision(
                    session,
                    round_id=round_id,
                    team_id=team_id,
                    quantity=quantity,
                    reasoning=reasoning,
                )
        except ValueError as exc:
            self._write(f"[red]Решение отклонено:[/] {exc}")
            return
        self._write(
            f"[green]Финальное решение зафиксировано:[/] команда {team_id}, "
            f"Q={quantity}. Раунд закрывается из основного dev-shell или дашборда."
        )


def main() -> None:
    """Точка входа для ``python -m devshell.role_tui``."""
    RoleShellApp().run()


if __name__ == "__main__":
    main()
