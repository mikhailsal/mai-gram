"""Entry point for mAI Companion.

Initializes the database, LLM provider, Telegram bot, and starts the application.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any

from mai_companion.bot.handler import BotHandler
from mai_companion.config import get_settings
from mai_companion.db import close_db, init_db, run_migrations
from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.messenger.telegram import TelegramMessenger

logger = logging.getLogger(__name__)

# Global references for graceful shutdown
_messenger: TelegramMessenger | None = None
_llm_provider: OpenRouterProvider | None = None


async def startup() -> None:
    """Initialize all application subsystems."""
    global _messenger, _llm_provider

    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger.info("mAI Companion starting up...")

    # Validate required settings
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is required. Set it in .env or environment.")
        sys.exit(1)

    if not settings.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY is required. Set it in .env or environment.")
        sys.exit(1)

    # Initialize database
    engine = await init_db(
        settings.database_url,
        echo=settings.debug,
    )

    # Run any pending migrations
    await run_migrations(engine)
    logger.info("Database ready")

    # Initialize LLM provider
    _llm_provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        default_model=settings.llm_model,
    )
    logger.info("LLM provider initialized (model: %s)", settings.llm_model)

    # Initialize Telegram messenger
    _messenger = TelegramMessenger(settings.telegram_bot_token)

    # Create and wire up the bot handler with memory/tooling settings.
    _handler = BotHandler(
        _messenger,
        _llm_provider,
        memory_data_dir=settings.memory_data_dir,
        summary_threshold=settings.summary_threshold,
        wiki_context_limit=settings.wiki_context_limit,
        short_term_limit=settings.short_term_limit,
        tool_max_iterations=settings.tool_max_iterations,
    )
    logger.info(
        "Memory subsystem configured: data_dir=%s threshold=%s wiki_limit=%s short_term_limit=%s tool_iterations=%s",
        settings.memory_data_dir,
        settings.summary_threshold,
        settings.wiki_context_limit,
        settings.short_term_limit,
        settings.tool_max_iterations,
    )

    # Start the messenger (begins polling)
    await _messenger.start()

    logger.info("mAI Companion is running! Press Ctrl+C to stop.")


async def shutdown() -> None:
    """Gracefully shut down all application subsystems."""
    global _messenger, _llm_provider

    logger.info("mAI Companion shutting down...")

    # Stop the messenger
    if _messenger:
        await _messenger.stop()
        _messenger = None

    # Close the LLM provider
    if _llm_provider:
        await _llm_provider.close()
        _llm_provider = None

    # Close the database
    await close_db()

    logger.info("Shutdown complete")


async def run() -> None:
    """Run the application with proper signal handling."""
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        asyncio.create_task(shutdown())

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        await startup()

        # Keep running until shutdown
        # The Telegram bot runs in the background via polling
        while _messenger is not None:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Unexpected error")
    finally:
        await shutdown()


def main() -> None:
    """Start the mAI Companion service."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass  # Already handled


if __name__ == "__main__":
    main()
