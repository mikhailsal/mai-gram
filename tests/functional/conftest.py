"""Fixtures for subprocess-based functional tests."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.functional.helpers.cli import CliHarness

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_DEFAULT_PROMPT = """
You are a concise assistant for CLI integration tests.
Follow direct formatting instructions exactly when the user requests a fixed token.
"""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture
def mai_chat_path() -> str:
    cli_path = shutil.which("mai-chat")
    if cli_path is None:
        pytest.skip("mai-chat is not installed on PATH. Run make install-dev first.")
    return cli_path


@pytest.fixture
def functional_cli_factory(
    tmp_path_factory: pytest.TempPathFactory,
    mai_chat_path: str,
    request: pytest.FixtureRequest,
) -> Callable[[str], CliHarness]:
    roots: list[Path] = []

    def _factory(prefix: str = "functional") -> CliHarness:
        root = tmp_path_factory.mktemp(prefix)
        roots.append(root)
        return _build_cli_harness(root, mai_chat_path)

    def _cleanup() -> None:
        rep_call = getattr(request.node, "rep_call", None)
        should_cleanup = bool(rep_call and rep_call.passed)
        for root in roots:
            if should_cleanup:
                shutil.rmtree(root, ignore_errors=True)
            else:
                print(f"[functional artifacts preserved] {root}")

    request.addfinalizer(_cleanup)
    return _factory


@pytest.fixture
def functional_cli(functional_cli_factory: Callable[[str], CliHarness]) -> CliHarness:
    return functional_cli_factory("functional")


@pytest.fixture
def requires_openrouter_api_key(functional_cli: CliHarness) -> None:
    if not functional_cli.env.get("OPENROUTER_API_KEY", "").strip():
        pytest.skip("OPENROUTER_API_KEY is required for this live functional scenario.")


def _build_cli_harness(root: Path, cli_path: str) -> CliHarness:
    data_dir = root / "data"
    prompts_dir = root / "prompts"
    db_path = root / "mai_gram.db"
    models_config_path = root / "models.toml"
    bots_config_path = root / "missing-bots.toml"

    data_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.joinpath("default.txt").write_text(_DEFAULT_PROMPT.strip() + "\n", encoding="utf-8")
    models_config_path.write_text(
        '[models]\ndefault = "openrouter/free"\n\n[models."openrouter/free"]\n',
        encoding="utf-8",
    )

    env = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
        "MEMORY_DATA_DIR": str(data_dir),
        "MODELS_CONFIG_PATH": str(models_config_path),
        "PROMPTS_DIR": str(prompts_dir),
        "BOTS_CONFIG_PATH": str(bots_config_path),
        "DEFAULT_TIMEZONE": "UTC",
        "LOG_LEVEL": "DEBUG",
        "PYTHONUNBUFFERED": "1",
        "ALLOWED_USERS": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_BOT_TOKEN_2": "",
        "TELEGRAM_BOT_TOKEN_3": "",
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    }
    if os.getenv("OPENROUTER_BASE_URL"):
        env["OPENROUTER_BASE_URL"] = os.environ["OPENROUTER_BASE_URL"]

    return CliHarness(
        cli_path=cli_path,
        root=root,
        env=env,
        db_path=db_path,
        data_dir=data_dir,
        prompts_dir=prompts_dir,
        models_config_path=models_config_path,
        bots_config_path=bots_config_path,
    )
