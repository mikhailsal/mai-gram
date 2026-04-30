"""Tests for shared adapter runtime helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mai_gram.core.adapter_runtime import (
    build_bot_handler,
    build_external_mcp_pool,
    build_openrouter_provider,
)


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.openrouter_api_key = "test-key"
    settings.llm_model = "openrouter/free"
    settings.openrouter_base_url = "https://openrouter.example/v1"
    settings.memory_data_dir = "./data"
    settings.wiki_context_limit = 20
    settings.short_term_limit = 500
    settings.tool_max_iterations = 5
    settings.get_external_mcp_config.return_value = {}
    return settings


def test_build_openrouter_provider_uses_settings() -> None:
    settings = _make_settings()

    with patch("mai_gram.core.adapter_runtime.OpenRouterProvider") as provider_cls:
        build_openrouter_provider(settings)

    provider_cls.assert_called_once_with(
        api_key="test-key",
        default_model="openrouter/free",
        base_url="https://openrouter.example/v1",
    )


def test_build_external_mcp_pool_handles_empty_and_configured() -> None:
    settings = _make_settings()

    assert build_external_mcp_pool(settings) is None

    settings.get_external_mcp_config.return_value = {"wiki": {"url": "http://mcp"}}
    with patch("mai_gram.core.adapter_runtime.ExternalMCPPool") as pool_cls:
        build_external_mcp_pool(settings)

    pool_cls.assert_called_once_with({"wiki": {"url": "http://mcp"}})


def test_build_bot_handler_forwards_shared_runtime_dependencies() -> None:
    settings = _make_settings()
    messenger = MagicMock()
    llm_provider = MagicMock()
    external_mcp_pool = MagicMock()
    bot_config = MagicMock()

    with patch("mai_gram.core.adapter_runtime.BotHandler") as handler_cls:
        build_bot_handler(
            messenger,
            llm_provider,
            settings,
            test_mode=True,
            bot_config=bot_config,
            external_mcp_pool=external_mcp_pool,
        )

    handler_cls.assert_called_once_with(
        messenger,
        llm_provider,
        memory_data_dir="./data",
        wiki_context_limit=20,
        short_term_limit=500,
        tool_max_iterations=5,
        test_mode=True,
        external_mcp_pool=external_mcp_pool,
        bot_config=bot_config,
    )
