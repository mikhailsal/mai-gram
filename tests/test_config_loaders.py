"""Tests for the extracted file-backed config loaders."""

from __future__ import annotations

import json

from mai_gram.config_loaders import BotsConfigLoader, ModelsConfigLoader, PromptConfigLoader


class TestModelsConfigLoader:
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
        models_path = tmp_path / "models.toml"
        models_path.write_text(
            "\n".join(
                [
                    "[models]",
                    "allowed = ['openrouter/free']",
                    "default = 'openrouter/free'",
                    "",
                    "[tools]",
                    "enabled = ['wiki_search']",
                    "",
                    "[mcp]",
                    f"mcp_config_path = '{mcp_path}'",
                    "external_servers = ['exa']",
                ]
            ),
            encoding="utf-8",
        )

        loader = ModelsConfigLoader(str(models_path))

        assert loader.get_allowed_models("fallback") == ["openrouter/free"]
        assert loader.get_default_model("fallback") == "openrouter/free"
        assert loader.get_tool_filter() == (["wiki_search"], None)
        assert loader.get_external_mcp_config() == {"exa": {"command": "exa"}}


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

        loader = BotsConfigLoader(str(bots_path))
        configs = loader.get_bot_configs()

        assert len(configs) == 1
        assert configs[0].token == "token-1"
        assert loader.get_bot_config_by_token("token-1") == configs[0]
        assert loader.get_bot_config_by_token("missing") is None


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
