"""Shared test fixtures for mAI Companion.

Provides an in-memory SQLite database and session for all tests.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from mai_companion.config import Settings
from mai_companion.db.models import Base

# In-memory SQLite URL for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance configured for testing."""
    return Settings(
        telegram_bot_token="test-token-123",
        openrouter_api_key="test-api-key-456",
        database_url=TEST_DATABASE_URL,
        chroma_persist_dir="/tmp/mai_test_chroma",
        log_level="DEBUG",
        debug=True,
    )


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an in-memory async engine for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

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
