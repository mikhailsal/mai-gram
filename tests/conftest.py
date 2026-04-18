"""Shared test fixtures for mai-gram.

Provides an in-memory SQLite database and session for all tests.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mai_gram.config import Settings
from mai_gram.db.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# In-memory SQLite URL for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add project-specific pytest CLI options."""
    parser.addoption(
        "--run-functional",
        action="store_true",
        default=False,
        help="Run tests marked as 'functional' (real LLM/OpenRouter integration).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "functional: integration tests that may call real LLM/provider services",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip functional tests unless explicitly enabled or API key is present."""
    run_functional = config.getoption("--run-functional")
    has_api_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if run_functional or has_api_key:
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "functional tests are skipped; pass --run-functional or set OPENROUTER_API_KEY "
            "to enable them."
        )
    )
    for item in items:
        if "functional" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance configured for testing."""
    return Settings(
        telegram_bot_token="test-token-123",
        openrouter_api_key="test-api-key-456",
        database_url=TEST_DATABASE_URL,
        log_level="DEBUG",
        debug=True,
    )


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an in-memory async engine for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _register_sqlite_functions(dbapi_conn: object, _rec: object) -> None:
        dbapi_conn.create_function("unicode_lower", 1, lambda s: s.lower() if s else s)  # type: ignore[union-attr]

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables and dispose
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional database session for testing.

    Each test gets its own session that is rolled back after the test,
    ensuring test isolation.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
