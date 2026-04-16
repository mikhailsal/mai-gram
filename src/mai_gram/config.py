"""Configuration management using Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
The TOML config (models.toml) is cached and auto-refreshed when the file
is modified on disk -- no restart needed.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class PromptConfig:
    """Per-prompt display settings loaded from a companion TOML file."""

    show_reasoning: bool = True
    show_tool_calls: bool = True


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
        """Return all configured Telegram bot tokens."""
        tokens = []
        if self.telegram_bot_token.strip():
            tokens.append(self.telegram_bot_token.strip())
        if self.telegram_bot_token_2.strip():
            tokens.append(self.telegram_bot_token_2.strip())
        if self.telegram_bot_token_3.strip():
            tokens.append(self.telegram_bot_token_3.strip())
        return tokens

    # -- TOML config with mtime-based caching --------------------------------
    # The parsed dict is cached and only re-read when the file's mtime changes,
    # so editing models.toml takes effect immediately without restarting.

    _toml_cache: dict[str, Any] = {}
    _toml_mtime: float = 0.0

    def _load_toml(self) -> dict[str, Any]:
        """Return parsed TOML data, re-reading only when the file has changed."""
        config_path = Path(self.models_config_path)
        if not config_path.exists():
            return {}
        try:
            mtime = config_path.stat().st_mtime
        except OSError:
            return {}

        if mtime != self._toml_mtime:
            with open(config_path, "rb") as f:
                self._toml_cache = tomllib.load(f)
            self._toml_mtime = mtime
            logger.info("Config reloaded: %s (mtime=%.3f)", config_path, mtime)

        return self._toml_cache

    # -- Public TOML accessors ------------------------------------------------

    def get_allowed_models(self) -> list[str]:
        """Load the model whitelist from the TOML config file."""
        data = self._load_toml()
        result: list[str] = data.get("models", {}).get("allowed", [self.llm_model])
        return result

    def get_default_model(self) -> str:
        """Get the default model from the TOML config."""
        data = self._load_toml()
        result: str = data.get("models", {}).get("default", self.llm_model)
        return result

    def get_model_params(self, model_id: str) -> dict[str, Any]:
        """Load per-model parameter overrides from the TOML config.

        Returns a dict of extra parameters (provider, reasoning, temperature, etc.)
        that should be merged into the OpenRouter request body for this model.
        Returns an empty dict if no overrides are defined.
        """
        data = self._load_toml()
        return dict(data.get("models", {}).get(model_id, {}))

    def get_tool_filter(self) -> tuple[list[str] | None, list[str] | None]:
        """Load tool enable/disable lists from the models config.

        Returns (enabled, disabled) where each is a list of tool names
        or None if not configured. If enabled is set, only those tools
        are allowed (whitelist). If disabled is set, those tools are
        blocked (blacklist). enabled takes precedence over disabled.
        """
        data = self._load_toml()
        tools_section = data.get("tools", {})
        return tools_section.get("enabled"), tools_section.get("disabled")

    def get_external_mcp_config(self) -> dict[str, dict[str, Any]]:
        """Load external MCP server configs from the models config.

        Reads the [mcp] section for the config path and whitelist,
        then loads and filters the MCP JSON file.

        Returns a dict mapping server_name -> server_config (command, args, env).
        """
        import json as _json

        data = self._load_toml()
        mcp_section = data.get("mcp", {})
        mcp_json_path = mcp_section.get("mcp_config_path", "")
        whitelist = set(mcp_section.get("external_servers", []))

        if not mcp_json_path or not whitelist:
            return {}

        mcp_json_path = Path(mcp_json_path).expanduser()
        if not mcp_json_path.exists():
            return {}

        with open(mcp_json_path, encoding="utf-8") as f:
            mcp_data = _json.load(f)

        servers_raw = mcp_data.get("mcpServers", {})
        result: dict[str, dict[str, Any]] = {}
        for name, config in servers_raw.items():
            if name in whitelist:
                result[name] = config

        return result

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

    def get_prompt_config(self, prompt_name: str) -> PromptConfig:
        """Load per-prompt display config from a companion TOML file.

        Looks for ``<prompts_dir>/<prompt_name>.toml``. If the file doesn't
        exist or can't be parsed, returns the default PromptConfig (everything
        visible).
        """
        config_path = Path(self.prompts_dir) / f"{prompt_name}.toml"
        if not config_path.exists():
            return PromptConfig()
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return PromptConfig(
                show_reasoning=data.get("show_reasoning", True),
                show_tool_calls=data.get("show_tool_calls", True),
            )
        except Exception:
            logger.warning("Failed to parse prompt config: %s", config_path)
            return PromptConfig()


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return a shared Settings instance (singleton).

    The instance is created once and reused.  TOML-derived values are
    still refreshed automatically via mtime-based caching in ``_load_toml``.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
