"""Tests for the extracted file-backed config loaders."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mai_gram.config_loaders import BotsConfigLoader, ModelsConfigLoader, PromptConfigLoader


def _fake_secret(label: str) -> str:
    return f"test-{label}-value"


# ── helpers ──────────────────────────────────────────────────────────


def _write_models_toml(path, content: str) -> str:
    models_path = path / "models.toml"
    models_path.write_text(content, encoding="utf-8")
    return str(models_path)


# ── ModelsConfigLoader ───────────────────────────────────────────────


class TestModelsConfigLoaderBasic:
    """Core loader behaviour: enabled flag, defaults, refresh."""

    def test_enabled_models_returned_in_order(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "default = 'b-model'",
                    "",
                    '[models."a-model"]',
                    "",
                    '[models."b-model"]',
                    "",
                    '[models."c-model"]',
                    "enabled = false",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fallback") == ["a-model", "b-model"]

    def test_all_models_enabled_by_default(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."x/alpha"]',
                    "temperature = 0.5",
                    "",
                    '[models."x/beta"]',
                    "temperature = 1.0",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fallback") == ["x/alpha", "x/beta"]

    def test_disabled_model_excluded(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."visible"]',
                    "",
                    '[models."hidden"]',
                    "enabled = false",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fallback") == ["visible"]

    def test_explicitly_enabled_model_included(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."explicit"]',
                    "enabled = true",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fallback") == ["explicit"]

    def test_no_model_sections_falls_back_to_default(self, tmp_path) -> None:
        toml = _write_models_toml(tmp_path, "[models]\n")
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("my-fallback") == ["my-fallback"]

    def test_missing_file_falls_back_to_default(self, tmp_path) -> None:
        loader = ModelsConfigLoader(str(tmp_path / "missing.toml"))

        assert loader.get_enabled_models("fallback") == ["fallback"]

    def test_get_allowed_models_is_alias_for_get_enabled_models(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."m1"]',
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_allowed_models("fb") == loader.get_enabled_models("fb")


class TestModelsConfigLoaderDefault:
    """Default model resolution."""

    def test_returns_configured_default(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "[models]\ndefault = 'preferred'\n",
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_default_model("fallback") == "preferred"

    def test_falls_back_when_no_default_set(self, tmp_path) -> None:
        toml = _write_models_toml(tmp_path, "[models]\n")
        loader = ModelsConfigLoader(toml)

        assert loader.get_default_model("fallback") == "fallback"


class TestModelsConfigLoaderTitle:
    """Display title resolution."""

    def test_returns_title_when_set(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."google/gemini-3.1-flash-lite"]',
                    "title = 'Gemini 3.1 Flash Lite @high'",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_title("google/gemini-3.1-flash-lite") == (
            "Gemini 3.1 Flash Lite @high"
        )

    def test_returns_none_when_no_title(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."no-title"]',
                    "temperature = 0.5",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_title("no-title") is None

    def test_returns_none_for_unknown_model(self, tmp_path) -> None:
        toml = _write_models_toml(tmp_path, "[models]\n")
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_title("unknown") is None


class TestModelsConfigLoaderModelId:
    """Real model ID resolution for aliases."""

    def test_returns_id_when_set(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."flash-creative"]',
                    "id = 'google/gemini-2.5-flash'",
                    "temperature = 1.5",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_id("flash-creative") == "google/gemini-2.5-flash"

    def test_returns_key_when_no_id(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."google/gemini-2.5-flash"]',
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_id("google/gemini-2.5-flash") == "google/gemini-2.5-flash"

    def test_returns_key_for_unknown_model(self, tmp_path) -> None:
        toml = _write_models_toml(tmp_path, "[models]\n")
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_id("unknown") == "unknown"


class TestModelsConfigLoaderParams:
    """Parameter extraction with meta-key stripping."""

    def test_returns_api_params_without_meta_keys(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."my-model"]',
                    "enabled = true",
                    "title = 'My Model'",
                    "id = 'real/model'",
                    "temperature = 0.7",
                    "max_tokens = 2048",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)
        params = loader.get_model_params("my-model")

        assert params == {"temperature": 0.7, "max_tokens": 2048}
        assert "enabled" not in params
        assert "title" not in params
        assert "id" not in params

    def test_nested_params_preserved(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."nested"]',
                    "reasoning.effort = 'high'",
                    "provider.order = ['Google']",
                    "provider.allow_fallbacks = false",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)
        params = loader.get_model_params("nested")

        assert params["reasoning"] == {"effort": "high"}
        assert params["provider"] == {"order": ["Google"], "allow_fallbacks": False}

    def test_empty_params_for_unknown_model(self, tmp_path) -> None:
        toml = _write_models_toml(tmp_path, "[models]\n")
        loader = ModelsConfigLoader(toml)

        assert loader.get_model_params("unknown") == {}


class TestModelsConfigLoaderMaxContextTokens:
    """max_context_tokens resolution: per-model → global → 0 (disabled)."""

    def test_returns_zero_when_not_configured(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    '[models."my-model"]',
                    "temperature = 0.5",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_max_context_tokens("my-model") == 0

    def test_returns_global_default(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "max_context_tokens = 200000",
                    "",
                    '[models."my-model"]',
                    "temperature = 0.5",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_max_context_tokens("my-model") == 200_000

    def test_per_model_overrides_global(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "max_context_tokens = 200000",
                    "",
                    '[models."big-context"]',
                    "max_context_tokens = 500000",
                    "",
                    '[models."small-context"]',
                    "max_context_tokens = 50000",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_max_context_tokens("big-context") == 500_000
        assert loader.get_max_context_tokens("small-context") == 50_000

    def test_per_model_zero_disables_even_with_global(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "max_context_tokens = 120000",
                    "",
                    '[models."no-truncation"]',
                    "max_context_tokens = 0",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_max_context_tokens("no-truncation") == 0

    def test_unknown_model_falls_back_to_global(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "max_context_tokens = 150000",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_max_context_tokens("unknown/model") == 150_000

    def test_not_included_in_model_params(self, tmp_path) -> None:
        """max_context_tokens is a meta-key and must NOT be sent to the API."""
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."my-model"]',
                    "max_context_tokens = 120000",
                    "temperature = 0.7",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)
        params = loader.get_model_params("my-model")

        assert "max_context_tokens" not in params
        assert params == {"temperature": 0.7}


class TestModelsConfigLoaderDuplicateModels:
    """Same base model with different parameter sets."""

    def test_alias_entries_with_shared_base_model(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "default = 'flash-balanced'",
                    "",
                    '[models."flash-balanced"]',
                    "id = 'google/gemini-2.5-flash'",
                    "title = 'Gemini Flash (balanced)'",
                    "reasoning.effort = 'medium'",
                    "temperature = 0.7",
                    "",
                    '[models."flash-creative"]',
                    "id = 'google/gemini-2.5-flash'",
                    "title = 'Gemini Flash (creative)'",
                    "temperature = 1.5",
                    "",
                    '[models."flash-precise"]',
                    "id = 'google/gemini-2.5-flash'",
                    "title = 'Gemini Flash (precise)'",
                    "reasoning.effort = 'high'",
                    "temperature = 0.2",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        enabled = loader.get_enabled_models("fb")
        assert enabled == ["flash-balanced", "flash-creative", "flash-precise"]

        for key in enabled:
            assert loader.get_model_id(key) == "google/gemini-2.5-flash"

        assert loader.get_model_params("flash-balanced") == {
            "reasoning": {"effort": "medium"},
            "temperature": 0.7,
        }
        assert loader.get_model_params("flash-creative") == {"temperature": 1.5}
        assert loader.get_model_params("flash-precise") == {
            "reasoning": {"effort": "high"},
            "temperature": 0.2,
        }

    def test_disabled_alias_excluded(self, tmp_path) -> None:
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."flash-normal"]',
                    "id = 'google/gemini-2.5-flash'",
                    "",
                    '[models."flash-experimental"]',
                    "id = 'google/gemini-2.5-flash'",
                    "enabled = false",
                    "temperature = 2.0",
                ]
            ),
        )
        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fb") == ["flash-normal"]
        assert loader.get_model_id("flash-experimental") == "google/gemini-2.5-flash"
        assert loader.get_model_params("flash-experimental") == {"temperature": 2.0}


class TestModelsConfigLoaderToolsAndMcp:
    def test_reads_tools_and_external_mcp_config(self, tmp_path) -> None:
        mcp_path = tmp_path / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "exa": {"command": "exa"},
                        "ignored": {"command": "ignored"},
                    }
                }
            ),
            encoding="utf-8",
        )
        toml = _write_models_toml(
            tmp_path,
            "\n".join(
                [
                    "[models]",
                    "",
                    '[models."openrouter/free"]',
                    "",
                    "[tools]",
                    "enabled = ['wiki_search']",
                    "",
                    "[mcp]",
                    f"mcp_config_path = '{mcp_path}'",
                    "external_servers = ['exa']",
                ]
            ),
        )

        loader = ModelsConfigLoader(toml)

        assert loader.get_enabled_models("fallback") == ["openrouter/free"]
        assert loader.get_default_model("fallback") == "fallback"
        assert loader.get_tool_filter() == (["wiki_search"], None)
        assert loader.get_external_mcp_config() == {"exa": {"command": "exa"}}


class TestModelsConfigLoaderRefresh:
    def test_mtime_based_refresh(self, tmp_path) -> None:
        import os

        toml_path = tmp_path / "models.toml"
        toml_path.write_text(
            "\n".join(
                [
                    "[models]",
                    '[models."model-a"]',
                ]
            ),
            encoding="utf-8",
        )

        loader = ModelsConfigLoader(str(toml_path))
        assert loader.get_enabled_models("fb") == ["model-a"]

        toml_path.write_text(
            "\n".join(
                [
                    "[models]",
                    '[models."model-b"]',
                    '[models."model-c"]',
                ]
            ),
            encoding="utf-8",
        )
        current_mtime = toml_path.stat().st_mtime
        os.utime(toml_path, (current_mtime + 1, current_mtime + 1))

        assert loader.get_enabled_models("fb") == ["model-b", "model-c"]


# ── BotsConfigLoader ─────────────────────────────────────────────────


class TestBotsConfigLoader:
    def test_skips_empty_tokens_and_finds_token_match(self, tmp_path) -> None:
        bots_path = tmp_path / "bots.toml"
        bots_path.write_text(
            "\n".join(
                [
                    "[[bots]]",
                    "token = 'token-1'",
                    "allowed_models = ['openrouter/free']",
                    "",
                    "[[bots]]",
                    "token = ''",
                ]
            ),
            encoding="utf-8",
        )

        token = _fake_secret("bot-token-1")
        bots_path.write_text(
            bots_path.read_text(encoding="utf-8").replace("token-1", token),
            encoding="utf-8",
        )

        loader = BotsConfigLoader(str(bots_path))
        configs = loader.get_bot_configs()

        assert len(configs) == 1
        assert configs[0].token == token
        assert loader.get_bot_config_by_token(token) == configs[0]
        assert loader.get_bot_config_by_token("missing") is None


# ── PromptConfigLoader ───────────────────────────────────────────────


class TestPromptConfigLoader:
    def test_reads_prompt_text_and_companion_config(self, tmp_path) -> None:
        (tmp_path / "default.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "default.toml").write_text(
            "\n".join(
                [
                    "show_reasoning = false",
                    "show_tool_calls = true",
                    "send_datetime = true",
                    "",
                    "[tools]",
                    "enabled = ['wiki_read']",
                    "",
                    "[mcp_servers]",
                    "disabled = ['messages']",
                ]
            ),
            encoding="utf-8",
        )

        loader = PromptConfigLoader(str(tmp_path))
        prompt_config = loader.get_prompt_config("default")

        assert loader.get_available_prompts() == {"default": "hello"}
        assert prompt_config.show_reasoning is False
        assert prompt_config.tools_enabled == ["wiki_read"]
        assert prompt_config.mcp_servers_disabled == ["messages"]

    def test_invalid_prompt_config_falls_back_to_defaults(self, tmp_path) -> None:
        (tmp_path / "broken.toml").write_text("[tools\ninvalid = true", encoding="utf-8")

        prompt_config = PromptConfigLoader(str(tmp_path)).get_prompt_config("broken")

        assert prompt_config.show_reasoning is True
        assert prompt_config.tools_enabled is None

    def test_unexpected_exception_in_prompt_config_propagates(self, tmp_path) -> None:
        """Unexpected exceptions should propagate instead of being caught."""
        (tmp_path / "valid.toml").write_text(
            "\n".join(
                [
                    "show_reasoning = true",
                ]
            ),
            encoding="utf-8",
        )
        loader = PromptConfigLoader(str(tmp_path))

        with (
            patch("mai_gram.config_loaders.tomllib.load", side_effect=RuntimeError("unexpected")),
            pytest.raises(RuntimeError, match="unexpected"),
        ):
            loader.get_prompt_config("valid")
