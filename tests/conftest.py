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
_DEFAULT_MAX_XDIST_WORKERS = 6


def _fake_secret(label: str) -> str:
    return f"test-{label}-value"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add project-specific pytest CLI options."""
    parser.addoption(
        "--run-functional",
        action="store_true",
        default=False,
        help="Run tests marked as 'functional_live' (real LLM/OpenRouter integration).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "functional_local: CLI subprocess tests that do NOT require an API key",
    )
    config.addinivalue_line(
        "markers",
        "functional_live: integration tests that call real LLM/provider services "
        "(require OPENROUTER_API_KEY)",
    )
    config.addinivalue_line(
        "markers",
        "functional: alias — treated as functional_live for backward compat",
    )
    config.addinivalue_line(
        "markers",
        "integration: in-process integration tests using mock/stub providers (no API key needed)",
    )


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Cap implicit xdist fan-out for stable live-provider test runs."""
    override = os.getenv("PYTEST_XDIST_AUTO_WORKERS", "").strip()
    if override:
        try:
            workers = int(override)
        except ValueError as exc:  # pragma: no cover - defensive config parsing
            raise pytest.UsageError("PYTEST_XDIST_AUTO_WORKERS must be a positive integer") from exc
        if workers < 1:
            raise pytest.UsageError("PYTEST_XDIST_AUTO_WORKERS must be a positive integer")
        return workers

    cpu_count = os.cpu_count() or 1
    return min(cpu_count, _DEFAULT_MAX_XDIST_WORKERS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live functional tests unless explicitly enabled or API key is present.

    Test ordering: local tests run before live tests so fast failures are caught early.
    """
    run_functional = config.getoption("--run-functional")
    has_api_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    skip_marker = pytest.mark.skip(
        reason=(
            "live functional tests are skipped; pass --run-functional or set "
            "OPENROUTER_API_KEY to enable them."
        )
    )

    local_tests: list[pytest.Item] = []
    live_tests: list[pytest.Item] = []
    other_tests: list[pytest.Item] = []

    for item in items:
        is_live = "functional_live" in item.keywords or "functional" in item.keywords
        is_local_functional = "functional_local" in item.keywords

        if is_live and not is_local_functional:
            if not (run_functional or has_api_key):
                item.add_marker(skip_marker)
            live_tests.append(item)
        elif is_local_functional:
            local_tests.append(item)
        else:
            other_tests.append(item)

    items[:] = other_tests + local_tests + live_tests


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance configured for testing."""
    return Settings(
        telegram_bot_token=_fake_secret("telegram-bot-token"),
        openrouter_api_key=_fake_secret("openrouter-api-key"),
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
