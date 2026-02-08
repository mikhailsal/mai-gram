"""Tests for configuration management."""

from __future__ import annotations

from mai_companion.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self) -> None:
        """Settings have sensible defaults when no env vars are set."""
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

    def test_get_settings_returns_instance(self) -> None:
        """get_settings() returns a valid Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
