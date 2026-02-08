"""Configuration management using Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Telegram --
    telegram_bot_token: str = Field(
        default="",
        description="Telegram Bot API token from @BotFather",
    )

    # -- OpenRouter --
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for LLM access",
    )
    llm_model: str = Field(
        default="openai/gpt-4o",
        description="Default LLM model identifier on OpenRouter",
    )

    # -- Database --
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/mai_companion.db",
        description="SQLAlchemy async database URL",
    )

    # -- ChromaDB --
    chroma_persist_dir: str = Field(
        default="./data/chroma_data",
        description="Directory for ChromaDB vector store persistence",
    )

    # -- Logging --
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )

    # -- Timezone --
    timezone: str = Field(
        default="UTC",
        description="IANA timezone for proactive messaging and quiet hours",
    )

    # -- Quiet hours --
    quiet_hours_start: int = Field(
        default=23,
        ge=0,
        le=23,
        description="Hour (0-23) when quiet hours begin (no proactive messages)",
    )
    quiet_hours_end: int = Field(
        default=7,
        ge=0,
        le=23,
        description="Hour (0-23) when quiet hours end",
    )

    # -- Debug --
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, SQL echo)",
    )


def get_settings() -> Settings:
    """Create and return a Settings instance.

    This is a factory function rather than a singleton so that tests can
    easily create fresh instances with overridden values.
    """
    return Settings()
