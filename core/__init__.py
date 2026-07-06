"""Pure business logic for the Econometrics Championship.

This package contains deterministic, I/O-free domain logic: the Cournot market
engine, rubric grading primitives, and dataset generation. Nothing here touches
the database, the network, or Telegram — that keeps the core unit-testable and
reusable across the bot, API, dashboard, and dev-shell layers.
"""
