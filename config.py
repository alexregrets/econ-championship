"""Application settings loaded from the environment / ``.env``.

Centralizes every secret and connection string so no other module reads
``os.environ`` directly. Import the singleton :data:`settings`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    """Typed application configuration.

    Values come from environment variables or a local ``.env`` file. The bot and
    LLM credentials default to empty so the data layer and dev-shell can run
    without them; consumers that need a credential should validate it themselves.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = ""
    groq_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./econ_tournament.db"


settings = Settings()
