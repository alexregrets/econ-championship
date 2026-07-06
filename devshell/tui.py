"""Textual dev-shell TUI for running and inspecting the tournament.

Run with::

    uv run python -m devshell.tui

Provides one-key actions to seed data, simulate the seven teams (at the Nash
benchmark or randomly), score the open round through the real engine + services,
inspect live submission status, and wipe the database — all without the Telegram
bot or any LLM call.
"""

from __future__ import annotations

from core.market_engine import MarketParameters, monopoly_quantity, nash_equilibrium
from db import repositories as repo
from db.session import drop_all, get_session_ctx, init_db
from devshell.seed import reset_and_seed
from devshell.simulate_team import (
    simulate_all_teams_nash,
    simulate_all_teams_random,
)
from services.round_service import compute_round_results

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Footer, Header, RichLog
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "textual is required for the dev-shell TUI. Install with: uv add textual"
    ) from exc


class DevShellApp(App[None]):
    """Interactive control panel for the tournament's local database."""

    TITLE = "Econ Championship — Dev Shell"
    CSS = """
    #actions { height: auto; padding: 1; }
    Button { margin: 0 1; }
    RichLog { border: round $accent; padding: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="actions"):
            yield Button("Seed (reset)", id="seed", variant="warning")
            yield Button("Simulate Nash", id="sim_nash", variant="primary")
            yield Button("Simulate Random", id="sim_random", variant="primary")
            yield Button("Score round", id="score", variant="success")
            yield Button("Status", id="status")
            yield Button("Clear DB", id="clear", variant="error")
        with VerticalScroll():
            yield RichLog(highlight=True, markup=True, id="log")
        yield Footer()

    async def on_mount(self) -> None:
        """Ensure the schema exists and greet the user."""
        await init_db()
        self._write("[bold]Dev shell ready.[/] Tables ensured. Press a button.")

    # -- helpers ---------------------------------------------------------- #

    def _write(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    async def _open_round_id(self) -> int | None:
        async with get_session_ctx() as session:
            round_ = await repo.get_open_round(session)
            return round_.id if round_ is not None else None

    # -- button dispatch -------------------------------------------------- #

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "seed": self._action_seed,
            "sim_nash": self._action_sim_nash,
            "sim_random": self._action_sim_random,
            "score": self._action_score,
            "status": self._action_status,
            "clear": self._action_clear,
        }
        handler = handlers.get(event.button.id or "")
        if handler is not None:
            await handler()

    # -- actions ---------------------------------------------------------- #

    async def _action_seed(self) -> None:
        summary = await reset_and_seed()
        self._write(
            f"[green]Seeded[/] {len(summary.team_ids)} teams, "
            f"{summary.student_count} students. Open round id="
            f"{summary.round_id}."
        )

    async def _action_sim_nash(self) -> None:
        round_id = await self._open_round_id()
        if round_id is None:
            self._write("[red]No open round.[/] Seed first.")
            return
        async with get_session_ctx() as session:
            submitted = await simulate_all_teams_nash(session, round_id)
        q = next(iter(submitted.values()), 0.0)
        self._write(
            f"[green]Simulated[/] {len(submitted)} teams at Nash q*={q:.4f} "
            f"for round {round_id}."
        )

    async def _action_sim_random(self) -> None:
        round_id = await self._open_round_id()
        if round_id is None:
            self._write("[red]No open round.[/] Seed first.")
            return
        async with get_session_ctx() as session:
            submitted = await simulate_all_teams_random(session, round_id, seed=42)
        self._write(
            f"[green]Simulated[/] {len(submitted)} teams with random "
            f"quantities for round {round_id}: "
            + ", ".join(f"{tid}:{q}" for tid, q in submitted.items())
        )

    async def _action_score(self) -> None:
        round_id = await self._open_round_id()
        if round_id is None:
            self._write("[red]No open round.[/] Seed first.")
            return
        try:
            async with get_session_ctx() as session:
                results = await compute_round_results(session, round_id)
        except ValueError as exc:
            self._write(f"[red]Cannot score:[/] {exc}")
            return
        price = next(iter(results.values())).price
        self._write(f"[bold]Round {round_id} scored.[/] Market price = {price:.2f}")
        for team_id, r in sorted(results.items()):
            self._write(
                f"  team {team_id}: q={r.quantity:.2f} "
                f"revenue={r.revenue:.2f} profit={r.profit:.2f}"
            )

    async def _action_status(self) -> None:
        async with get_session_ctx() as session:
            teams = await repo.list_teams(session)
            round_ = await repo.get_open_round(session)
            if round_ is None:
                self._write("[yellow]No open round.[/]")
            else:
                assert round_.id is not None
                decisions = await repo.list_decisions_for_round(
                    session, round_.id
                )
                submitted_team_ids = {d.team_id for d in decisions}
                params = MarketParameters(
                    a=round_.market_a,
                    b=round_.market_b,
                    marginal_cost=round_.market_mc,
                )
                self._write(
                    f"[bold]Round {round_.number}[/] (id={round_.id}, "
                    f"{round_.method.value}, status={round_.status.value}) "
                    f"P={round_.market_a}-{round_.market_b}*Q, mc={round_.market_mc}"
                )
                self._write(
                    f"  benchmarks: Nash q*="
                    f"{nash_equilibrium(len(teams), params):.4f}, "
                    f"monopoly Q={monopoly_quantity(params):.4f}"
                )
                for team in teams:
                    mark = (
                        "[green]✓[/]"
                        if team.id in submitted_team_ids
                        else "[red]…[/]"
                    )
                    self._write(
                        f"  {mark} {team.name} ({team.company_name}) "
                        f"cumulative={team.cumulative_profit:.2f}"
                    )

    async def _action_clear(self) -> None:
        await drop_all()
        await init_db()
        self._write("[red]Database cleared.[/] Schema recreated empty.")


def main() -> None:
    """Entry point for ``python -m devshell.tui``."""
    DevShellApp().run()


if __name__ == "__main__":
    main()
