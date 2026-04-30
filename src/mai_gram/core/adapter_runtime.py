"""Shared runtime construction helpers for CLI and Telegram adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mai_gram.bot.handler import BotHandler
from mai_gram.llm.openrouter import OpenRouterProvider
from mai_gram.mcp_servers.external import ExternalMCPPool

if TYPE_CHECKING:
    from mai_gram.config import BotConfig, Settings
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.messenger.base import Messenger


def build_openrouter_provider(settings: Settings) -> OpenRouterProvider:
    """Build the standard OpenRouter provider from repository settings."""
    return OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        default_model=settings.llm_model,
        base_url=settings.openrouter_base_url,
    )


def build_external_mcp_pool(settings: Settings) -> ExternalMCPPool | None:
    """Build the shared external MCP pool, if any servers are configured."""
    external_mcp_configs = settings.get_external_mcp_config()
    if not external_mcp_configs:
        return None
    return ExternalMCPPool(external_mcp_configs)


def build_bot_handler(
    messenger: Messenger,
    llm_provider: LLMProvider,
    settings: Settings,
    *,
    test_mode: bool,
    bot_config: BotConfig | None = None,
    external_mcp_pool: ExternalMCPPool | None = None,
) -> BotHandler:
    """Build a BotHandler with the shared adapter/runtime wiring."""
    return BotHandler(
        messenger,
        llm_provider,
        memory_data_dir=settings.memory_data_dir,
        wiki_context_limit=settings.wiki_context_limit,
        short_term_limit=settings.short_term_limit,
        tool_max_iterations=settings.tool_max_iterations,
        test_mode=test_mode,
        external_mcp_pool=external_mcp_pool,
        bot_config=bot_config,
    )
