"""Data layer: SQLModel tables, the async engine/session, and repositories.

Everything that touches persistent storage lives here. Higher layers (services,
bot, api, dashboard, dev-shell) talk to the database exclusively through the
repository functions in :mod:`db.repositories`, never with raw SQL.
"""
