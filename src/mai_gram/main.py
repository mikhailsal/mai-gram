"""Entry point for mai-gram.

Initializes the database, LLM provider, Telegram bot, and starts the application.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mai_gram.config import Settings, get_settings
from mai_gram.core.adapter_runtime import (
    build_bot_handler,
    build_openrouter_provider,
)
from mai_gram.core.adapter_runtime import (
    build_external_mcp_pool as shared_build_external_mcp_pool,
)
from mai_gram.db import close_db, init_db, run_migrations
from mai_gram.messenger.telegram import TelegramMessenger

if TYPE_CHECKING:
    from mai_gram.llm.openrouter import OpenRouterProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool

logger = logging.getLogger(__name__)

PID_FILE = Path("data/.mai-gram.pid")


@dataclass(slots=True)
class AppRuntime:
    settings: Settings | None = None
    messengers: list[TelegramMessenger] = field(default_factory=list)
    llm_provider: OpenRouterProvider | None = None
    external_mcp_pool: ExternalMCPPool | None = None
    config_watcher_task: asyncio.Task[None] | None = None
    shutdown_task: asyncio.Task[None] | None = None


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_pid_lock() -> None:
    """Write a PID file and abort if another instance is already running."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            old_pid = None

        if old_pid and old_pid != os.getpid() and _is_process_alive(old_pid):
            print(
                f"ERROR: Another mai-gram instance is already running (PID {old_pid}).\n"
                f"If this is a stale lock, remove {PID_FILE} and try again.",
                file=sys.stderr,
            )
            sys.exit(1)
        elif old_pid and old_pid != os.getpid():
            logger.debug("Removing stale PID file (PID %s no longer running)", old_pid)

    PID_FILE.write_text(str(os.getpid()))
    atexit.register(_release_pid_lock)


def _release_pid_lock() -> None:
    """Remove the PID file on exit."""
    try:
        if PID_FILE.exists():
            stored_pid = int(PID_FILE.read_text().strip())
            if stored_pid == os.getpid():
                PID_FILE.unlink()
    except (ValueError, OSError):
        pass


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def _get_bot_tokens(settings: Settings) -> list[str]:
    bot_tokens = settings.get_all_bot_tokens()
    if not bot_tokens:
        logger.error("TELEGRAM_BOT_TOKEN is required. Set it in .env or environment.")
        sys.exit(1)

    if not settings.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY is required. Set it in .env or environment.")
        sys.exit(1)

    return bot_tokens


def _build_external_mcp_pool(settings: Settings) -> ExternalMCPPool | None:
    external_mcp_configs = settings.get_external_mcp_config()
    pool = shared_build_external_mcp_pool(settings)
    if pool is None:
        return None

    logger.info(
        "External MCP pool: %d server(s) configured: %s",
        len(external_mcp_configs),
        ", ".join(external_mcp_configs.keys()),
    )
    return pool


async def _start_messengers(runtime: AppRuntime, bot_tokens: list[str]) -> None:
    settings = runtime.settings
    llm_provider = runtime.llm_provider
    if settings is None or llm_provider is None:
        raise RuntimeError("Application runtime is not initialized")

    bot_configs = settings.get_bot_configs()

    for i, token in enumerate(bot_tokens, start=1):
        messenger = TelegramMessenger(token)
        bot_config = settings.get_bot_config_by_token(token) if bot_configs else None

        build_bot_handler(
            messenger,
            llm_provider,
            settings,
            test_mode=False,
            external_mcp_pool=runtime.external_mcp_pool,
            bot_config=bot_config,
        )

        await messenger.start()
        runtime.messengers.append(messenger)
        logger.info("Bot %d/%d started: @%s", i, len(bot_tokens), messenger.bot_id)


async def startup(runtime: AppRuntime) -> None:
    """Initialize all application subsystems."""
    settings = get_settings()
    runtime.settings = settings

    _configure_logging(settings)

    _acquire_pid_lock()
    logger.info("mai-gram starting up (PID %d)...", os.getpid())

    bot_tokens = _get_bot_tokens(settings)

    engine = await init_db(settings.database_url, echo=settings.debug)
    await run_migrations(engine)
    logger.info("Database ready")

    runtime.llm_provider = build_openrouter_provider(settings)
    logger.info(
        "LLM provider initialized (model: %s, base_url: %s)",
        settings.llm_model,
        settings.openrouter_base_url,
    )

    runtime.external_mcp_pool = _build_external_mcp_pool(settings)
    await _start_messengers(runtime, bot_tokens)
    runtime.config_watcher_task = asyncio.create_task(_watch_config(settings))

    logger.info(
        "mai-gram is running with %d bot(s)! Press Ctrl+C to stop.",
        len(runtime.messengers),
    )


async def _watch_config(settings: Settings) -> None:
    """Background task that watches models.toml for changes and pre-warms the cache.

    Uses simple mtime polling (2s interval).  The actual cache refresh is
    performed by ``settings.refresh_models_config()`` which compares mtimes.
    """
    config_path = Path(settings.models_config_path)
    last_mtime: float = 0.0

    try:
        if config_path.exists():
            last_mtime = config_path.stat().st_mtime
    except OSError:
        pass

    while True:
        await asyncio.sleep(2)
        try:
            if not config_path.exists():
                continue
            mtime = config_path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                settings.refresh_models_config()
                models = settings.get_allowed_models()
                logger.info(
                    "Config file changed -- reloaded. Models: %s",
                    ", ".join(m.split("/")[-1] for m in models),
                )
        except (OSError, ValueError):
            logger.debug("Config watcher transient error", exc_info=True)


async def shutdown(runtime: AppRuntime) -> None:
    """Gracefully shut down all application subsystems."""
    logger.info("mai-gram shutting down...")

    if runtime.config_watcher_task and not runtime.config_watcher_task.done():
        runtime.config_watcher_task.cancel()
    runtime.config_watcher_task = None

    if runtime.llm_provider and runtime.llm_provider.active_requests > 0:
        logger.info(
            "Waiting for %d in-flight LLM request(s) to complete...",
            runtime.llm_provider.active_requests,
        )

    for messenger in runtime.messengers:
        await messenger.stop()
    runtime.messengers.clear()

    if runtime.external_mcp_pool:
        await runtime.external_mcp_pool.stop_all()
        runtime.external_mcp_pool = None

    if runtime.llm_provider:
        await runtime.llm_provider.close()
        runtime.llm_provider = None

    await close_db()
    runtime.settings = None
    runtime.shutdown_task = None
    _release_pid_lock()
    logger.info("Shutdown complete")


async def run() -> None:
    """Run the application with proper signal handling."""
    loop = asyncio.get_running_loop()
    runtime = AppRuntime()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        if runtime.shutdown_task is None or runtime.shutdown_task.done():
            runtime.shutdown_task = asyncio.create_task(shutdown(runtime))

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        await startup(runtime)
        while runtime.messengers:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:  # intentional: process boundary; log then shut down cleanly
        logger.exception("Unexpected error")
    finally:
        await shutdown(runtime)


def _run_with_reload() -> None:
    """Run the bot with auto-reload on code changes.

    Watches both ``src/`` (Python files) and ``config/`` + ``prompts/``
    (TOML, txt, md) so that config edits also trigger a restart.
    Note: in normal (non-reload) mode, config/models.toml changes are
    picked up automatically without restarting -- see ``_watch_config``.
    """
    try:
        from watchfiles import DefaultFilter, PythonFilter, run_process
    except ImportError:
        print(
            "ERROR: 'watchfiles' is required for --reload mode.\n"
            "Install it with: pip install watchfiles",
            file=sys.stderr,
        )
        sys.exit(1)

    src_dir = Path(__file__).resolve().parent.parent
    project_root = src_dir.parent
    config_dir = project_root / "config"
    prompts_dir = project_root / "prompts"

    watch_dirs = [src_dir]
    if config_dir.exists():
        watch_dirs.append(config_dir)
    if prompts_dir.exists():
        watch_dirs.append(prompts_dir)

    class _CodeAndConfigFilter(DefaultFilter):
        allowed_extensions = (
            *PythonFilter().extensions,
            ".toml",
            ".txt",
            ".md",
        )

    print(f"Auto-reload enabled -- watching {', '.join(str(d) for d in watch_dirs)}")

    def _on_reload(changes: set[tuple[Any, str]]) -> None:
        changed_files = [path for _, path in changes]
        short = [str(Path(p).relative_to(project_root)) for p in changed_files]
        print(f"\nDetected changes in: {', '.join(short)}")
        print("Restarting bot process...\n")

    command = f"{sys.executable} -m mai_gram.main"

    run_process(
        *watch_dirs,
        target=command,
        target_type="command",
        watch_filter=_CodeAndConfigFilter(),
        callback=_on_reload,
        grace_period=2,
        sigint_timeout=5,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mai-gram",
        description="mai-gram -- Telegram-LLM bridge via OpenRouter.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload: watch src/ for changes and restart.",
    )
    return parser.parse_args()


def main() -> None:
    """Start the mai-gram service."""
    args = _parse_args()
    if args.reload:
        _run_with_reload()
    else:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run())


if __name__ == "__main__":
    main()
