# Dev Shell

Run and inspect the tournament **without Telegram and without any LLM call**, against the local SQLite database.

## Launch the TUI

```bash
uv run python -m devshell.tui
```

Buttons:

| Action | What it does |
|--------|--------------|
| **Seed (reset)** | Drops all tables, recreates the schema, creates 7 teams × 3 students and one **open** round. |
| **Simulate Nash** | Submits the symmetric Nash quantity `q*` for every team. After scoring, all profits should be equal. |
| **Simulate Random** | Submits a reproducible random quantity per team (seed=42). |
| **Score round** | Runs the Cournot engine on the open round's decisions, persists results, updates cumulative profits. |
| **Status** | Live view: market params, Nash/monopoly benchmarks, and which teams have submitted (✓ / …). |
| **Clear DB** | Wipes all data and recreates an empty schema. |

## Scripted use (no TUI)

```bash
# Reset + seed straight from the CLI
uv run python -m devshell.seed
```

Or from your own async script:

```python
from db.session import get_session_ctx
from devshell.seed import seed
from devshell.simulate_team import simulate_all_teams_nash
from services.round_service import compute_round_results

async with get_session_ctx() as session:
    summary = await seed(session)
    await simulate_all_teams_nash(session, summary.round_id)
    results = await compute_round_results(session, summary.round_id)
```

The database file location comes from `DATABASE_URL` in `.env` (defaults to `./econ_tournament.db`).

## Role shell (ролевой раунд)

```bash
uv run python -m devshell.role_tui
```

Локальная многоролевая оболочка: без Telegram и без Groq-вызовов.

| Action | What it does |
|--------|--------------|
| **Seed + срезы** | Пересоздаёт базу, сеет 7 команд и открытый раунд, генерирует ролевые срезы (3 на команду) из единого ground truth. |
| **Статус** | Показывает id команд — их вводят в поле «id команды». |
| **Маркетолог / Аналитик / Финансист** | «Войти» в роль команды: видно только срез этой роли (общий ориентир + приватный сигнал + доля издержек). |
| **Предложить Q** | Сохранить Q-предложение текущей роли (RoleInput, для аудита). |
| **Lead: обзор** | Предложения всех ролей, равновзвешенный агрегат и what-if против Нэша конкурентов. |
| **Lead: зафиксировать Q** | Записать финальное Decision команды (одно на команду, как в симметричном раунде). |

Groq для нарративов срезов — опционально: `generate_role_views(..., llm=GroqClient(api_key))`; по умолчанию тексты статические и сети нет.
