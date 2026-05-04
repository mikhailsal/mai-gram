"""Unit tests for the functional CLI harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.functional.helpers import cli as cli_helpers
from tests.functional.helpers.cli import CliHarness, CompletedCliRun

if TYPE_CHECKING:
    from pathlib import Path


def _build_harness(root: Path) -> CliHarness:
    return CliHarness(
        cli_path="mai-chat",
        root=root,
        env={},
        db_path=root / "mai_gram.db",
        data_dir=root / "data",
        prompts_dir=root / "prompts",
        models_config_path=root / "models.toml",
        bots_config_path=root / "bots.toml",
    )


def test_run_cli_decodes_timeout_output(monkeypatch, tmp_path) -> None:
    harness = _build_harness(tmp_path)

    async def fake_run_cli_command(*args, **kwargs):
        raise cli_helpers._CliTimeoutError(b"partial stdout", b"partial stderr")

    monkeypatch.setattr(cli_helpers, "_run_cli_command", fake_run_cli_command)
    monkeypatch.setattr(cli_helpers, "_USE_SUBPROCESS", True)

    result = harness.run_cli("hello", timeout=12, allow_retry=False)

    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"


def test_send_message_with_live_retry_uses_extended_timeout(monkeypatch, tmp_path) -> None:
    harness = _build_harness(tmp_path)
    seen_timeouts: list[int] = []

    def fake_send_message(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        debug: bool = False,
        stream_debug: bool = False,
        env_overrides: dict[str, str | None] | None = None,
        timeout: int = 60,
    ) -> CompletedCliRun:
        seen_timeouts.append(timeout)
        return CompletedCliRun(
            command=("mai-chat", "-c", chat_id, text),
            returncode=0,
            stdout="ok",
            stderr="",
            root=self.root,
        )

    monkeypatch.setattr(CliHarness, "send_message", fake_send_message)

    result = harness.send_message_with_live_retry("func-chat", "hello")

    assert result.returncode == 0
    assert seen_timeouts == [120]


def test_send_message_with_live_retry_retries_transient_timeout(monkeypatch, tmp_path) -> None:
    harness = _build_harness(tmp_path)
    attempts = 0

    def fake_send_message(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        debug: bool = False,
        stream_debug: bool = False,
        env_overrides: dict[str, str | None] | None = None,
        timeout: int = 60,
    ) -> CompletedCliRun:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CompletedCliRun(
                command=("mai-chat", "-c", chat_id, text),
                returncode=124,
                stdout="",
                stderr="WARNING: Network error during stream (attempt 1/3): ",
                root=self.root,
            )
        return CompletedCliRun(
            command=("mai-chat", "-c", chat_id, text),
            returncode=0,
            stdout="ok",
            stderr="",
            root=self.root,
        )

    monkeypatch.setattr(CliHarness, "send_message", fake_send_message)

    result = harness.send_message_with_live_retry("func-chat", "hello")

    assert result.returncode == 0
    assert attempts == 2


def test_send_message_with_live_retry_retries_malformed_toolcall_output(
    monkeypatch, tmp_path
) -> None:
    harness = _build_harness(tmp_path)
    attempts = 0

    def fake_send_message(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        debug: bool = False,
        stream_debug: bool = False,
        env_overrides: dict[str, str | None] | None = None,
        timeout: int = 60,
    ) -> CompletedCliRun:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CompletedCliRun(
                command=("mai-chat", "-c", chat_id, text),
                returncode=0,
                stdout='OLCALL&gt;[{"name": "wiki_create"}]\nTools used: none',
                stderr="",
                root=self.root,
            )
        return CompletedCliRun(
            command=("mai-chat", "-c", chat_id, text),
            returncode=0,
            stdout="Tools used: wiki_create",
            stderr="",
            root=self.root,
        )

    monkeypatch.setattr(CliHarness, "send_message", fake_send_message)

    result = harness.send_message_with_live_retry("func-chat", "remember this")

    assert result.returncode == 0
    assert attempts == 2


def test_send_message_with_live_retry_retries_truncated_call_marker(monkeypatch, tmp_path) -> None:
    harness = _build_harness(tmp_path)
    attempts = 0

    def fake_send_message(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        debug: bool = False,
        stream_debug: bool = False,
        env_overrides: dict[str, str | None] | None = None,
        timeout: int = 60,
    ) -> CompletedCliRun:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CompletedCliRun(
                command=("mai-chat", "-c", chat_id, text),
                returncode=0,
                stdout='CALL&gt;[{"name": "wiki_create"}]\nTools used: none',
                stderr="",
                root=self.root,
            )
        return CompletedCliRun(
            command=("mai-chat", "-c", chat_id, text),
            returncode=0,
            stdout="Tools used: wiki_create",
            stderr="",
            root=self.root,
        )

    monkeypatch.setattr(CliHarness, "send_message", fake_send_message)

    result = harness.send_message_with_live_retry("func-chat", "remember this")

    assert result.returncode == 0
    assert attempts == 2
