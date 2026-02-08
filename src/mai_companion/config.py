"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Telegram
    telegram_bot_token: str = ""

    # OpenRouter
    openrouter_api_key: str = ""
    llm_model: str = "openai/gpt-4o"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/mai_companion.db"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_data"

    # Logging
    log_level: str = "INFO"

    # Timezone
    timezone: str = "UTC"

    # Quiet hours
    quiet_hours_start: int = 23
    quiet_hours_end: int = 7

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
