"""Tests for configuration management."""

from __future__ import annotations

from unittest.mock import patch

from mai_companion.config import Settings, get_settings

# Environment variable names that correspond to Settings fields.
# Pydantic reads these automatically, so we must clear them in tests
# that assert default values.
_SETTINGS_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "OPENROUTER_API_KEY",
    "LLM_MODEL",
    "DATABASE_URL",
    "CHROMA_PERSIST_DIR",
    "MEMORY_DATA_DIR",
    "SUMMARY_THRESHOLD",
    "WIKI_CONTEXT_LIMIT",
    "SHORT_TERM_LIMIT",
    "TOOL_MAX_ITERATIONS",
    "LOG_LEVEL",
    "TIMEZONE",
    "QUIET_HOURS_START",
    "QUIET_HOURS_END",
    "ALLOWED_USERS",
    "DEBUG",
]


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self) -> None:
        """Settings have sensible defaults when no env vars are set."""
        import os

        # Remove any environment variables that would override defaults.
        clean_env = {
            k: v for k, v in os.environ.items() if k.upper() not in _SETTINGS_ENV_VARS
        }
        with patch.dict("os.environ", clean_env, clear=True):
            settings = Settings(
                _env_file=None,  # type: ignore[call-arg]
            )
        assert settings.telegram_bot_token == ""
        assert settings.openrouter_api_key == ""
        assert settings.llm_model == "openai/gpt-4o"
        assert "sqlite" in settings.database_url
        assert settings.log_level == "INFO"
        assert settings.timezone == "UTC"
        assert settings.quiet_hours_start == 23
        assert settings.quiet_hours_end == 7
        assert settings.debug is False
        assert settings.allowed_users == ""
        assert settings.memory_data_dir == "./data"
        assert settings.summary_threshold == 20
        assert settings.wiki_context_limit == 20
        assert settings.short_term_limit == 30
        assert settings.tool_max_iterations == 5

    def test_override_via_constructor(self) -> None:
        """Settings can be overridden via constructor kwargs."""
        settings = Settings(
            telegram_bot_token="my-token",
            openrouter_api_key="my-key",
            llm_model="anthropic/claude-3-opus",
            log_level="DEBUG",
            debug=True,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.telegram_bot_token == "my-token"
        assert settings.openrouter_api_key == "my-key"
        assert settings.llm_model == "anthropic/claude-3-opus"
        assert settings.log_level == "DEBUG"
        assert settings.debug is True

    def test_memory_settings_override(self) -> None:
        """Memory settings can be overridden."""
        settings = Settings(
            memory_data_dir="/tmp/memory",
            summary_threshold=12,
            wiki_context_limit=10,
            short_term_limit=25,
            tool_max_iterations=7,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.memory_data_dir == "/tmp/memory"
        assert settings.summary_threshold == 12
        assert settings.wiki_context_limit == 10
        assert settings.short_term_limit == 25
        assert settings.tool_max_iterations == 7

    def test_get_settings_returns_instance(self) -> None:
        """get_settings() returns a valid Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)


class TestAllowedUsers:
    """Tests for the allowed_users access control setting."""

    def test_empty_allowed_users_returns_empty_set(self) -> None:
        """Empty allowed_users string returns empty set (allow all)."""
        settings = Settings(
            allowed_users="",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == set()

    def test_whitespace_only_returns_empty_set(self) -> None:
        """Whitespace-only allowed_users returns empty set."""
        settings = Settings(
            allowed_users="   ",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == set()

    def test_single_user_id(self) -> None:
        """Single user ID is parsed correctly."""
        settings = Settings(
            allowed_users="123456789",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {"123456789"}

    def test_multiple_user_ids(self) -> None:
        """Multiple user IDs are parsed correctly."""
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
        """User IDs with surrounding spaces are trimmed."""
        settings = Settings(
            allowed_users=" 123456789 , 987654321 , 555555555 ",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {
            "123456789",
            "987654321",
            "555555555",
        }

    def test_empty_entries_are_ignored(self) -> None:
        """Empty entries in the comma-separated list are ignored."""
        settings = Settings(
            allowed_users="123456789,,987654321,",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_allowed_user_ids() == {"123456789", "987654321"}
