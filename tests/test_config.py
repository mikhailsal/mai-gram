"""Tests for configuration management."""

from __future__ import annotations

import os
from unittest.mock import patch

from mai_gram.config import Settings, get_settings

_SETTINGS_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_MODEL",
    "DATABASE_URL",
    "MEMORY_DATA_DIR",
    "WIKI_CONTEXT_LIMIT",
    "SHORT_TERM_LIMIT",
    "TOOL_MAX_ITERATIONS",
    "LOG_LEVEL",
    "ALLOWED_USERS",
    "DEBUG",
]


class TestSettings:
    def test_default_values(self) -> None:
        clean_env = {k: v for k, v in os.environ.items() if k.upper() not in _SETTINGS_ENV_VARS}
        with patch.dict("os.environ", clean_env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.telegram_bot_token == ""
        assert settings.openrouter_api_key == ""
        assert settings.llm_model == "openai/gpt-4o-mini"
        assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert "sqlite" in settings.database_url
        assert settings.log_level == "INFO"
        assert settings.debug is False
        assert settings.allowed_users == ""
        assert settings.memory_data_dir == "./data"
        assert settings.wiki_context_limit == 20
        assert settings.short_term_limit == 500
        assert settings.tool_max_iterations == 5

    def test_override_via_constructor(self) -> None:
        settings = Settings(
            telegram_bot_token="my-token",
            openrouter_api_key="my-key",
            llm_model="anthropic/claude-3-opus",
            openrouter_base_url="http://localhost:8080/v1",
            log_level="DEBUG",
            debug=True,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.telegram_bot_token == "my-token"
        assert settings.openrouter_api_key == "my-key"
        assert settings.llm_model == "anthropic/claude-3-opus"
        assert settings.openrouter_base_url == "http://localhost:8080/v1"
        assert settings.log_level == "DEBUG"
        assert settings.debug is True

    def test_get_settings_returns_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_all_bot_tokens(self) -> None:
        settings = Settings(
            telegram_bot_token="token1",
            telegram_bot_token_2="token2",
            telegram_bot_token_3="",
            bots_config_path="/nonexistent/bots.toml",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_all_bot_tokens() == ["token1", "token2"]

    def test_get_all_bot_tokens_empty(self) -> None:
        settings = Settings(
            bots_config_path="/nonexistent/bots.toml",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_all_bot_tokens() == []

    def test_refresh_models_config_reloads_changed_file(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text(
            "[models]\nallowed = ['openrouter/free']\ndefault = 'openrouter/free'\n",
            encoding="utf-8",
        )
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )

        assert settings.get_allowed_models() == ["openrouter/free"]

        models_path.write_text(
            "[models]\nallowed = ['openrouter/alt']\ndefault = 'openrouter/alt'\n",
            encoding="utf-8",
        )
        current_mtime = models_path.stat().st_mtime
        os.utime(models_path, (current_mtime + 1, current_mtime + 1))

        settings.refresh_models_config()

        assert settings.get_allowed_models() == ["openrouter/alt"]


class TestAllowedUsers:
    def test_empty_allowed_users_returns_empty_set(self) -> None:
        settings = Settings(allowed_users="", _env_file=None)  # type: ignore[call-arg]
        assert settings.get_allowed_user_ids() == set()

    def test_single_user_id(self) -> None:
        settings = Settings(
            allowed_users="123456789",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {"123456789"}

    def test_multiple_user_ids(self) -> None:
        settings = Settings(
            allowed_users="123456789,987654321,555555555",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {
            "123456789",
            "987654321",
            "555555555",
        }

    def test_user_ids_with_spaces(self) -> None:
        settings = Settings(
            allowed_users=" 123456789 , 987654321 ",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {"123456789", "987654321"}
