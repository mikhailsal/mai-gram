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


def _fake_secret(label: str) -> str:
    return f"test-{label}-value"


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
        telegram_bot_token = _fake_secret("telegram-bot-token")
        openrouter_api_key = _fake_secret("openrouter-api-key")
        settings = Settings(
            telegram_bot_token=telegram_bot_token,
            openrouter_api_key=openrouter_api_key,
            llm_model="anthropic/claude-3-opus",
            openrouter_base_url="http://localhost:8080/v1",
            log_level="DEBUG",
            debug=True,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.telegram_bot_token == telegram_bot_token
        assert settings.openrouter_api_key == openrouter_api_key
        assert settings.llm_model == "anthropic/claude-3-opus"
        assert settings.openrouter_base_url == "http://localhost:8080/v1"
        assert settings.log_level == "DEBUG"
        assert settings.debug is True

    def test_get_settings_returns_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_all_bot_tokens(self) -> None:
        telegram_bot_token = _fake_secret("telegram-bot-token-1")
        telegram_bot_token_2 = _fake_secret("telegram-bot-token-2")
        settings = Settings(
            telegram_bot_token=telegram_bot_token,
            telegram_bot_token_2=telegram_bot_token_2,
            telegram_bot_token_3="",
            bots_config_path="/nonexistent/bots.toml",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_all_bot_tokens() == [telegram_bot_token, telegram_bot_token_2]

    def test_get_all_bot_tokens_empty(self) -> None:
        settings = Settings(
            bots_config_path="/nonexistent/bots.toml",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_all_bot_tokens() == []

    def test_refresh_models_config_reloads_changed_file(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text(
            '[models]\n[models."openrouter/free"]\n',
            encoding="utf-8",
        )
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )

        assert settings.get_allowed_models() == ["openrouter/free"]

        models_path.write_text(
            '[models]\n[models."openrouter/alt"]\n[models."openrouter/new"]\n',
            encoding="utf-8",
        )
        current_mtime = models_path.stat().st_mtime
        os.utime(models_path, (current_mtime + 1, current_mtime + 1))

        settings.refresh_models_config()

        assert settings.get_allowed_models() == ["openrouter/alt", "openrouter/new"]

    def test_get_model_title_and_id(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text(
            "\n".join(
                [
                    "[models]",
                    '[models."flash-creative"]',
                    "id = 'google/gemini-2.5-flash'",
                    "title = 'Flash Creative'",
                    "temperature = 1.5",
                ]
            ),
            encoding="utf-8",
        )
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )

        assert settings.get_model_title("flash-creative") == "Flash Creative"
        assert settings.get_model_id("flash-creative") == "google/gemini-2.5-flash"
        assert settings.get_model_params("flash-creative") == {"temperature": 1.5}
        assert "title" not in settings.get_model_params("flash-creative")
        assert "id" not in settings.get_model_params("flash-creative")


class TestMaxContextTokens:
    def test_get_max_context_tokens_delegates_to_loader(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text(
            "\n".join(
                [
                    "[models]",
                    "max_context_tokens = 100000",
                    "",
                    '[models."google/gemini"]',
                    "max_context_tokens = 250000",
                ]
            ),
            encoding="utf-8",
        )
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )

        assert settings.get_max_context_tokens("google/gemini") == 250_000
        assert settings.get_max_context_tokens("unknown/model") == 100_000

    def test_get_max_context_tokens_defaults_to_zero(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text("[models]\n", encoding="utf-8")
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )

        assert settings.get_max_context_tokens("any-model") == 0


class TestSettingsLoaders:
    def test_get_available_templates_returns_list(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        templates = settings.get_available_templates()
        assert isinstance(templates, list)
        assert "empty" in templates
        assert "xml" in templates

    def test_get_response_template_by_name(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        t = settings.get_response_template("xml")
        assert t.name == "xml"

    def test_get_response_template_none_returns_empty(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        t = settings.get_response_template(None)
        assert t.name == "empty"

    def test_get_response_template_with_params(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        t = settings.get_response_template("xml", {"reasoning_field": "think"})
        fields = t.get_fields()
        assert fields[0].name == "think"

    def test_get_available_prompts(self, tmp_path) -> None:
        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("You are a helpful bot.", encoding="utf-8")
        settings = Settings(
            prompts_dir=str(tmp_path),
            _env_file=None,  # type: ignore[call-arg]
        )
        prompts = settings.get_available_prompts()
        assert "test_prompt" in prompts
        assert prompts["test_prompt"] == "You are a helpful bot."

    def test_get_prompt_config_missing_returns_default(self, tmp_path) -> None:
        settings = Settings(
            prompts_dir=str(tmp_path),
            _env_file=None,  # type: ignore[call-arg]
        )
        config = settings.get_prompt_config("nonexistent")
        assert config.show_reasoning is True
        assert config.tools_enabled is None

    def test_get_tool_filter_default_is_none(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text("[models]\n", encoding="utf-8")
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )
        enabled, disabled = settings.get_tool_filter()
        assert enabled is None
        assert disabled is None

    def test_get_default_model_fallback(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text("[models]\n", encoding="utf-8")
        settings = Settings(
            models_config_path=str(models_path),
            llm_model="fallback/model",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_default_model() == "fallback/model"

    def test_get_external_mcp_config_empty(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text("[models]\n", encoding="utf-8")
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_external_mcp_config() == {}

    def test_get_bot_config_by_token_not_found(self) -> None:
        settings = Settings(
            bots_config_path="/nonexistent/bots.toml",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.get_bot_config_by_token("unknown_token") is None

    def test_load_toml_alias(self, tmp_path) -> None:
        models_path = tmp_path / "models.toml"
        models_path.write_text('[models]\n[models."test/model"]\n', encoding="utf-8")
        settings = Settings(
            models_config_path=str(models_path),
            _env_file=None,  # type: ignore[call-arg]
        )
        data = settings._load_toml()
        assert "models" in data

    def test_reset_settings(self) -> None:
        from mai_gram.config import reset_settings

        reset_settings()
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2


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
