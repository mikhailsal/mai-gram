"""Database engine, async session management, and initialization.

Provides:
- AsyncEngine and AsyncSession factory via get_engine() / get_session_factory()
- init_db() to create all tables on first run
- get_session() async context manager for convenient session usage
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mai_companion.db.models import Base

logger = logging.getLogger(__name__)

# Module-level singletons (initialized via init_db)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Create or return the async SQLAlchemy engine.

    Args:
        database_url: SQLAlchemy-style database URL
            (e.g. "sqlite+aiosqlite:///./data/mai_companion.db").
        echo: If True, log all SQL statements (useful for debugging).
    """
    global _engine
    if _engine is None:
        # Ensure the parent directory for a SQLite database file exists
        if "sqlite" in database_url:
            db_path = database_url.split("///")[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(
            database_url,
            echo=echo,
            # SQLite-specific: enable WAL mode for better concurrency
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        )
        logger.info("Database engine created: %s", _sanitize_url(database_url))
    return _engine


def get_session_factory(engine: AsyncEngine | None = None) -> async_sessionmaker[AsyncSession]:
    """Create or return the async session factory.

    Args:
        engine: Optional engine to use. If None, uses the module-level engine.

    Raises:
        RuntimeError: If no engine is available (call get_engine or init_db first).
    """
    global _session_factory
    if _session_factory is None:
        eng = engine or _engine
        if eng is None:
            raise RuntimeError(
                "No database engine available. Call init_db() or get_engine() first."
            )
        _session_factory = async_sessionmaker(eng, expire_on_commit=False)
        logger.debug("Session factory created")
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a database session.

    Usage::

        async with get_session() as session:
            result = await session.execute(select(Companion))
            companions = result.scalars().all()

    Automatically commits on success, rolls back on exception.

    Raises:
        RuntimeError: If the session factory has not been initialized.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Initialize the database: create engine, session factory, and all tables.

    This is the main entry point for database setup. Call this once at
    application startup.

    Args:
        database_url: SQLAlchemy-style database URL.
        echo: If True, log all SQL statements.

    Returns:
        The created AsyncEngine.
    """
    engine = get_engine(database_url, echo=echo)
    get_session_factory(engine)

    # Create all tables that don't exist yet
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized, all tables created/verified")
    return engine


async def close_db() -> None:
    """Close the database engine and reset module-level singletons.

    Call this during application shutdown.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
    _engine = None
    _session_factory = None


def reset_db_state() -> None:
    """Reset module-level singletons without closing connections.

    Primarily useful for testing to ensure a clean state between tests.
    """
    global _engine, _session_factory
    _engine = None
    _session_factory = None


def _sanitize_url(url: str) -> str:
    """Remove sensitive parts from a database URL for logging."""
    # For SQLite there's nothing sensitive, but future-proof for other DBs
    if "@" in url:
        protocol_end = url.index("://") + 3
        at_sign = url.index("@")
        return url[:protocol_end] + "***" + url[at_sign:]
    return url
