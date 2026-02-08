"""Entry point for mAI Companion.

Initializes the database, runs migrations, and starts the application.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mai_companion.config import get_settings
from mai_companion.db import close_db, init_db, run_migrations


async def startup() -> None:
    """Initialize all application subsystems."""
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("mAI Companion starting up...")

    # Initialize database
    engine = await init_db(
        settings.database_url,
        echo=settings.debug,
    )

    # Run any pending migrations
    await run_migrations(engine)

    logger.info("Database ready")
    logger.info(
        "Service initialized. Telegram bot and other subsystems "
        "will be implemented in subsequent phases."
    )


async def shutdown() -> None:
    """Gracefully shut down all application subsystems."""
    logger = logging.getLogger(__name__)
    logger.info("mAI Companion shutting down...")
    await close_db()
    logger.info("Shutdown complete")


def main() -> None:
    """Start the mAI Companion service."""
    try:
        asyncio.run(startup())
    except KeyboardInterrupt:
        asyncio.run(shutdown())


if __name__ == "__main__":
    main()
