"""Conftest for integration tests that use in-process DB and global state.

These tests manipulate global singletons (init_db, _settings_instance). Each test
has its own teardown that resets state, but they MUST run serially (not under xdist)
because their global state changes cannot be isolated across parallel workers.

When run as part of the full test suite, use: pytest --ignore=tests/test_integration
Then run them separately: pytest tests/test_integration -n0
"""

from __future__ import annotations
