# Econometrics Championship

A multi-round Cournot oligopoly played by student teams, built as a data
generator for an econometrics course at the Financial University under the
Government of the Russian Federation.

The point is not the game. The point is what the game leaves behind.

Teams compete for several rounds. Every round writes one row per team into a
panel — output, price, profit, role decisions, which shocks were active. By the
end of the tournament the course has a dataset of teams × rounds that the same
students then have to run regressions on. They are not handed `mroz.csv`; they
are handed the consequences of their own decisions, and they know exactly how
the data-generating process works because they were inside it.

That also makes the tournament a clean setting for causal inference: mechanics
are switched on at specific rounds, for specific teams, so treatment timing is a
design choice rather than something to be recovered from observational noise.

---

## The market

One shared market, linear inverse demand, price floored at zero:

```
P = a - b · Q_total
```

Every team commits a quantity `q`. Profit is `P · q - c · q`.

**Symmetric engine** (`core/market_engine.py`) — all firms share a marginal cost
`c`. Interior Nash equilibrium:

```
q* = (a - c) / (b · (n + 1))
```

**Asymmetric engine** (`core/market_engine_asymmetric.py`) — firm-specific costs
`c_i` under the same demand curve:

```
FOC for firm i:   a - b·Q - b·q_i - c_i = 0
Summing FOCs:     Q* = (n·a - Σc) / (b · (n + 1))
Price:            P* = (a + Σc) / (n + 1)
Firm output:      q_i* = (P* - c_i) / b
```

With equal `c_i` this collapses to the symmetric formula. That collapse is
pinned by regression tests against `nash_equilibrium`, not by inspection.

Everything under `core/` is pure: no I/O, no randomness, no LLM calls. Same
inputs, same outputs, every time. Grading and leaderboards are only trustworthy
if the ground truth underneath them is deterministic, so the boundary is
enforced rather than assumed.

## Calibration

Market parameters are not invented. They are backed out of the Russian oil
market in 2013: given observed prices and quantities, `implied_marginal_costs`
inverts the equilibrium conditions to recover the `c_i` that would have produced
what actually happened. Students play against a market whose numbers have a
source.

## Roles and private information

Each team member holds a role, and each role sees a different slice of the same
ground truth. Nobody sees all of it. A team that does not talk internally is
guessing.

On top of team profit sits a personal KPI, so roles want different things:

| Role | Scored on |
|---|---|
| Marketing | market share, `q_i / Q_total` |
| Finance | margin, `(P - c) · q / revenue` |
| Sales analyst | price forecast accuracy, `\|forecast - P\|` |

Final student score is 70% team, 30% personal. Marketing pulls output up,
finance pulls it down, and the team lead has to actually settle the argument
instead of averaging it away. This is the mechanism the course is really
teaching: conflicting objectives under private information.

## Market events

`core/market_events.py` applies multiplicative shocks to demand intercept,
demand slope, and costs. Composition is commutative — applying a demand shock
then a cost shock gives the same market as the reverse order — which is what
makes a round's outcome independent of the order the professor happened to
click things in. Seven scenario presets ship with it.

## Stack

- **Engines and scoring** — pure Python, no dependencies
- **Persistence** — SQLAlchemy over SQLite
- **Student interface** — Telegram bot (`/join`, `/submit`, `/status`), aiogram
- **Instructor interface** — Streamlit dashboard: round setup, manual entry,
  round closing, results, equilibrium comparison charts
- **Public view** — read-only standings page and attendance summary
- **Qualitative grading** — rubric-based, Groq, run explicitly as a separate
  action rather than automatically

```
core/       market engines, role KPI, market events, rubric grader
db/         SQLAlchemy models and repositories
services/   round lifecycle
bot/        Telegram handlers
dashboard/  Streamlit pages
llm/        Groq client and prompts
devshell/   local multi-role session runner
tests/      engine invariants, repositories, integration
```

## Running it

```bash
uv sync
uv run pytest -q
uv run streamlit run dashboard/app.py
```

Copy `.env.example` to `.env` and fill in the Telegram token and Groq key.
See `DEPLOYMENT.md` for the VPS setup.

## Where this actually stands

Pre-production. Target is a working prototype for the September 2026 semester.

248 tests pass across 18 of 19 test modules.

Working: both engines, calibration on 2013 data, role model with private
slices, market events, role KPI, Telegram bot, dashboard, public view,
rubric grading.

Not done:

- Market events are not wired into round closing. `services.round_service`
  is missing `effective_parameters`, which is why
  `tests/test_round_service_events.py` — the nineteenth module — does not
  import. The test file is the specification; the implementation is what is
  missing, not the design.
- Role KPI has the same gap: `compute_role_kpis` is fully tested but nothing
  calls it outside the tests.
- No migrations. The schema is one SQLite file, so any change means recreating
  the database. Tolerable now, not tolerable once a real round has been played —
  migrations have to land before the first live round, not after.
- Realism mechanics 2 through 6 in `GAME_DESIGN.md` are designed, not built.

`STATE.md` carries the current snapshot and the ordered recovery plan.
`DECISIONS.md` records why things are the way they are — read it before
changing anything in `core/`.

## A note on `core/`

The engines, calibration, and KPI formulas are not written by prompting for
them. Invariant test first, then code. A plausible-but-wrong formula is
indistinguishable from a correct one by eye, and every number the students are
graded on comes out of these files.
