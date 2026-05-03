"""Configuration facade built on environment settings plus file-backed loaders."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mai_gram.config_loaders import (
    BotConfig,
    BotsConfigLoader,
    ModelsConfigLoader,
    PromptConfig,
    PromptConfigLoader,
)

if TYPE_CHECKING:
    from mai_gram.response_templates.base import ResponseTemplate

__all__ = ["BotConfig", "PromptConfig", "Settings", "get_settings"]

logger = logging.getLogger(__name__)


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
        default=500,
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

    # -- Multi-bot config --
    bots_config_path: str = Field(
        default="config/bots.toml",
        description="Path to TOML file defining per-bot configurations",
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

    # -- Timezone --
    default_timezone: str = Field(
        default="UTC",
        description=(
            "Default IANA timezone for new chats (e.g. Europe/Moscow). "
            "Users can override per-chat with /timezone."
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
        """Return all configured Telegram bot tokens.

        If ``config/bots.toml`` exists, tokens are loaded from there.
        Otherwise, falls back to the legacy ``TELEGRAM_BOT_TOKEN*`` env vars.
        """
        bot_configs = self.get_bot_configs()
        if bot_configs:
            return [bc.token for bc in bot_configs]
        tokens = []
        if self.telegram_bot_token.strip():
            tokens.append(self.telegram_bot_token.strip())
        if self.telegram_bot_token_2.strip():
            tokens.append(self.telegram_bot_token_2.strip())
        if self.telegram_bot_token_3.strip():
            tokens.append(self.telegram_bot_token_3.strip())
        return tokens

    _models_loader_instance: ModelsConfigLoader | None = None
    _models_loader_path: str = ""
    _bots_loader_instance: BotsConfigLoader | None = None
    _bots_loader_path: str = ""
    _prompt_loader_instance: PromptConfigLoader | None = None
    _prompt_loader_path: str = ""

    def _models_loader(self) -> ModelsConfigLoader:
        config_path = self.models_config_path
        if self._models_loader_instance is None or self._models_loader_path != config_path:
            self._models_loader_instance = ModelsConfigLoader(config_path)
            self._models_loader_path = config_path
        return self._models_loader_instance

    def _bots_loader(self) -> BotsConfigLoader:
        config_path = self.bots_config_path
        if self._bots_loader_instance is None or self._bots_loader_path != config_path:
            self._bots_loader_instance = BotsConfigLoader(config_path)
            self._bots_loader_path = config_path
        return self._bots_loader_instance

    def _prompt_loader(self) -> PromptConfigLoader:
        prompts_dir = self.prompts_dir
        if self._prompt_loader_instance is None or self._prompt_loader_path != prompts_dir:
            self._prompt_loader_instance = PromptConfigLoader(prompts_dir)
            self._prompt_loader_path = prompts_dir
        return self._prompt_loader_instance

    def get_bot_configs(self) -> list[BotConfig]:
        """Load per-bot configurations from ``config/bots.toml``.

        Returns an empty list if the file doesn't exist (legacy env-var mode).
        The result is mtime-cached, so edits are picked up automatically.
        """
        return self._bots_loader().get_bot_configs()

    def get_bot_config_by_token(self, token: str) -> BotConfig | None:
        """Find the BotConfig for a given token, or None if not found."""
        return self._bots_loader().get_bot_config_by_token(token)

    def refresh_models_config(self) -> None:
        """Refresh the shared models TOML cache if the file changed on disk."""
        self._models_loader().refresh()

    def _load_toml(self) -> dict[str, Any]:
        """Backward-compatible internal alias for the models TOML loader."""
        return self._models_loader().refresh()

    # -- Public TOML accessors ------------------------------------------------

    def get_allowed_models(self) -> list[str]:
        """Load the model whitelist from the TOML config file."""
        return self._models_loader().get_allowed_models(self.llm_model)

    def get_default_model(self) -> str:
        """Get the default model from the TOML config."""
        return self._models_loader().get_default_model(self.llm_model)

    def get_model_params(self, model_id: str) -> dict[str, Any]:
        """Load per-model parameter overrides from the TOML config.

        Returns a dict of extra parameters (provider, reasoning, temperature, etc.)
        that should be merged into the OpenRouter request body for this model.
        Returns an empty dict if no overrides are defined.
        """
        return self._models_loader().get_model_params(model_id)

    def get_tool_filter(self) -> tuple[list[str] | None, list[str] | None]:
        """Load tool enable/disable lists from the models config.

        Returns (enabled, disabled) where each is a list of tool names
        or None if not configured. If enabled is set, only those tools
        are allowed (whitelist). If disabled is set, those tools are
        blocked (blacklist). enabled takes precedence over disabled.
        """
        return self._models_loader().get_tool_filter()

    def get_external_mcp_config(self) -> dict[str, dict[str, Any]]:
        """Load external MCP server configs from the models config.

        Reads the [mcp] section for the config path and whitelist,
        then loads and filters the MCP JSON file.

        Returns a dict mapping server_name -> server_config (command, args, env).
        """
        return self._models_loader().get_external_mcp_config()

    def get_available_prompts(self) -> dict[str, str]:
        """Load available system prompt templates from the prompts directory.

        Returns a dict mapping prompt name (filename without extension)
        to the prompt content.
        """
        return self._prompt_loader().get_available_prompts()

    def get_prompt_config(self, prompt_name: str) -> PromptConfig:
        """Load per-prompt config from a companion TOML file.

        Looks for ``<prompts_dir>/<prompt_name>.toml``. If the file doesn't
        exist or can't be parsed, returns the default PromptConfig (everything
        visible, no tool/MCP overrides).
        """
        return self._prompt_loader().get_prompt_config(prompt_name)

    # -- Response templates ---------------------------------------------------

    def get_available_templates(self) -> list[str]:
        """Return all registered response template names."""
        from mai_gram.response_templates.registry import list_template_names

        return list_template_names()

    def get_response_template(self, name: str | None) -> ResponseTemplate:
        """Look up a response template by name.

        Returns the EmptyTemplate for None or unknown names.
        """
        from mai_gram.response_templates.registry import get_template

        return get_template(name)


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return a shared Settings instance (singleton).

    The instance is created once and reused.  TOML-derived values are
    still refreshed automatically via the delegated file-backed loaders.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
