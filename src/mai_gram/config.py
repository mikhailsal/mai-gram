"""Configuration management using Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


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
        description="Telegram Bot API token from @BotFather (primary bot)",
    )
    telegram_bot_token_2: str = Field(
        default="",
        description="Telegram Bot API token for second bot (optional)",
    )
    telegram_bot_token_3: str = Field(
        default="",
        description="Telegram Bot API token for third bot (optional)",
    )

    # -- OpenRouter --
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for LLM access",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL (override for local proxy debugging)",
    )
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Default LLM model identifier on OpenRouter",
    )

    # -- Database --
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/mai_gram.db",
        description="SQLAlchemy async database URL",
    )

    # -- Memory subsystem --
    memory_data_dir: str = Field(
        default="./data",
        description="Base directory for wiki and summary memory files",
    )
    wiki_context_limit: int = Field(
        default=20,
        ge=1,
        description="Maximum number of wiki entries to include in prompt context",
    )
    short_term_limit: int = Field(
        default=50,
        ge=1,
        description="Recent message window size for short-term context",
    )
    tool_max_iterations: int = Field(
        default=5,
        ge=1,
        description="Maximum agentic tool-calling iterations per response",
    )

    # -- Model whitelist config --
    models_config_path: str = Field(
        default="config/models.toml",
        description="Path to TOML file defining available models",
    )

    # -- Prompts directory --
    prompts_dir: str = Field(
        default="prompts",
        description="Directory containing system prompt template files",
    )

    # -- Logging --
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )

    # -- Access Control --
    allowed_users: str = Field(
        default="",
        description=(
            "Comma-separated list of Telegram user IDs allowed to use the bot. "
            "Leave empty to allow anyone (not recommended for production)."
        ),
    )

    # -- Debug --
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, SQL echo)",
    )

    def get_allowed_user_ids(self) -> set[str]:
        """Parse the allowed_users string into a set of user IDs."""
        if not self.allowed_users.strip():
            return set()
        return {uid.strip() for uid in self.allowed_users.split(",") if uid.strip()}

    def get_all_bot_tokens(self) -> list[str]:
        """Return all configured Telegram bot tokens."""
        tokens = []
        if self.telegram_bot_token.strip():
            tokens.append(self.telegram_bot_token.strip())
        if self.telegram_bot_token_2.strip():
            tokens.append(self.telegram_bot_token_2.strip())
        if self.telegram_bot_token_3.strip():
            tokens.append(self.telegram_bot_token_3.strip())
        return tokens

    def get_allowed_models(self) -> list[str]:
        """Load the model whitelist from the TOML config file."""
        config_path = Path(self.models_config_path)
        if not config_path.exists():
            return [self.llm_model]
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        models_section = data.get("models", {})
        return models_section.get("allowed", [self.llm_model])

    def get_default_model(self) -> str:
        """Get the default model from the TOML config."""
        config_path = Path(self.models_config_path)
        if not config_path.exists():
            return self.llm_model
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        models_section = data.get("models", {})
        return models_section.get("default", self.llm_model)

    def get_model_params(self, model_id: str) -> dict:
        """Load per-model parameter overrides from the TOML config.

        Returns a dict of extra parameters (provider, reasoning, temperature, etc.)
        that should be merged into the OpenRouter request body for this model.
        Returns an empty dict if no overrides are defined.
        """
        config_path = Path(self.models_config_path)
        if not config_path.exists():
            return {}
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        models_section = data.get("models", {})
        return dict(models_section.get(model_id, {}))

    def get_tool_filter(self) -> tuple[list[str] | None, list[str] | None]:
        """Load tool enable/disable lists from the models config.

        Returns (enabled, disabled) where each is a list of tool names
        or None if not configured. If enabled is set, only those tools
        are allowed (whitelist). If disabled is set, those tools are
        blocked (blacklist). enabled takes precedence over disabled.
        """
        config_path = Path(self.models_config_path)
        if not config_path.exists():
            return None, None
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        tools_section = data.get("tools", {})
        enabled = tools_section.get("enabled")
        disabled = tools_section.get("disabled")
        return enabled, disabled

    def get_available_prompts(self) -> dict[str, str]:
        """Load available system prompt templates from the prompts directory.

        Returns a dict mapping prompt name (filename without extension)
        to the prompt content.
        """
        prompts_path = Path(self.prompts_dir)
        if not prompts_path.exists():
            return {}
        result: dict[str, str] = {}
        for f in sorted(prompts_path.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                result[f.stem] = f.read_text(encoding="utf-8").strip()
        return result


def get_settings() -> Settings:
    """Create and return a Settings instance."""
    return Settings()
