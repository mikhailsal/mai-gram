"""Entry point for mAI Companion.

Initializes the database, LLM provider, Telegram bot, and starts the application.

Supports ``--reload`` for development: watches ``src/`` for Python file changes
and automatically restarts the bot process when code is modified.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from mai_companion.bot.handler import BotHandler
from mai_companion.config import get_settings
from mai_companion.db import close_db, init_db, run_migrations
from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.messenger.telegram import TelegramMessenger

logger = logging.getLogger(__name__)

# Global references for graceful shutdown
_messengers: list[TelegramMessenger] = []
_llm_provider: OpenRouterProvider | None = None


async def startup() -> None:
    """Initialize all application subsystems."""
    global _llm_provider

    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger.info("mAI Companion starting up...")

    # Validate required settings
    bot_tokens = settings.get_all_bot_tokens()
    if not bot_tokens:
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

    logger.info(
        "Memory subsystem configured: data_dir=%s threshold=%s wiki_limit=%s short_term_limit=%s tool_iterations=%s",
        settings.memory_data_dir,
        settings.summary_threshold,
        settings.wiki_context_limit,
        settings.short_term_limit,
        settings.tool_max_iterations,
    )

    # Initialize and start a Telegram messenger + handler for each bot token
    for i, token in enumerate(bot_tokens, start=1):
        messenger = TelegramMessenger(token)

        # Create and wire up the bot handler with memory/tooling settings.
        _handler = BotHandler(
            messenger,
            _llm_provider,
            memory_data_dir=settings.memory_data_dir,
            summary_threshold=settings.summary_threshold,
            wiki_context_limit=settings.wiki_context_limit,
            short_term_limit=settings.short_term_limit,
            tool_max_iterations=settings.tool_max_iterations,
        )

        # Start the messenger (begins polling).
        # The bot_id (username) is resolved automatically from the Telegram API.
        await messenger.start()
        _messengers.append(messenger)
        logger.info(
            "Bot %d/%d started: @%s",
            i,
            len(bot_tokens),
            messenger.bot_id,
        )

    logger.info(
        "mAI Companion is running with %d bot(s)! Press Ctrl+C to stop.",
        len(_messengers),
    )


async def shutdown() -> None:
    """Gracefully shut down all application subsystems."""
    global _llm_provider

    logger.info("mAI Companion shutting down...")

    # Stop all messengers
    for messenger in _messengers:
        await messenger.stop()
    _messengers.clear()

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
        # The Telegram bots run in the background via polling
        while _messengers:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Unexpected error")
    finally:
        await shutdown()


# ---------------------------------------------------------------------------
# Auto-reload support
# ---------------------------------------------------------------------------

def _run_with_reload() -> None:
    """Run the bot inside a watchfiles-managed process that restarts on code changes.

    Uses ``watchfiles.run_process`` to watch the ``src/`` directory for ``.py``
    file changes.  When a change is detected the bot process is gracefully
    stopped (SIGINT → SIGKILL after timeout) and a fresh process is spawned.
    """
    try:
        from watchfiles import PythonFilter, run_process
    except ImportError:
        print(
            "ERROR: 'watchfiles' is required for --reload mode.\n"
            "Install it with: pip install watchfiles",
            file=sys.stderr,
        )
        sys.exit(1)

    src_dir = Path(__file__).resolve().parent.parent  # src/
    project_root = src_dir.parent  # project root

    print(f"🔄 Auto-reload enabled — watching {src_dir} for changes")
    print("   The bot will restart automatically when you save a .py file.\n")

    def _on_reload(changes: set[tuple[Any, str]]) -> None:
        changed_files = [path for _, path in changes]
        short = [str(Path(p).relative_to(project_root)) for p in changed_files]
        print(f"\n🔄 Detected changes in: {', '.join(short)}")
        print("   Restarting bot process...\n")

    # Build the command string — watchfiles splits it via shlex internally.
    command = f"{sys.executable} -m mai_companion.main"

    run_process(
        src_dir,
        target=command,
        target_type="command",
        watch_filter=PythonFilter(),
        callback=_on_reload,
        grace_period=2,  # Wait 2s after start before watching (let bot initialize)
        sigint_timeout=5,  # Give 5s for graceful shutdown before SIGKILL
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the main entry point."""
    parser = argparse.ArgumentParser(
        prog="mai-companion",
        description="mAI Companion — Telegram bot service.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Enable auto-reload: watch src/ for Python file changes and "
            "restart the bot automatically. Useful during development."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Start the mAI Companion service."""
    args = _parse_args()

    if args.reload:
        _run_with_reload()
    else:
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass  # Already handled


if __name__ == "__main__":
    main()
