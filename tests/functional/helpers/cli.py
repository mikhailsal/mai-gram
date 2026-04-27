"""Subprocess helpers for black-box functional CLI tests."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

FREE_MODEL = "openrouter/free"

_RATE_LIMIT_RE = re.compile(r"(?:\b429\b|rate.?limit|too many requests)", re.IGNORECASE)
_TRANSIENT_RE = re.compile(
    r"(?:network error|timed out|temporarily unavailable|connection reset|5\d\d)",
    re.IGNORECASE,
)
_LIVE_OUTPUT_TRANSIENT_RE = re.compile(
    r"(?:the model returned an empty response|stream completed without any data|without any data)",
    re.IGNORECASE,
)
_LIVE_OUTPUT_MALFORMED_TOOLCALL_RE = re.compile(r"toolcall[>\]]", re.IGNORECASE)
_LIVE_OUTPUT_NO_TOOLS_RE = re.compile(r"tools used:\s*none", re.IGNORECASE)
_LIVE_OUTPUT_PROVIDER_ERROR_RE = re.compile(
    r"(?:ai provider error|something went wrong with the ai provider|tap regenerate to retry)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CompletedCliRun:
    """Completed `mai-chat` subprocess invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    root: Path

    @property
    def output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def require_ok(self) -> CompletedCliRun:
        assert self.returncode == 0, self.output
        return self


@dataclass(slots=True)
class CliHarness:
    """High-level interface around the real `mai-chat` binary."""

    cli_path: str
    root: Path
    env: dict[str, str]
    db_path: Path
    data_dir: Path
    prompts_dir: Path
    models_config_path: Path
    bots_config_path: Path

    def run_cli(
        self,
        *args: str,
        timeout: int = 60,
        env_overrides: dict[str, str | None] | None = None,
        allow_retry: bool = True,
    ) -> CompletedCliRun:
        command = (self.cli_path, *args)
        deadline = time.monotonic() + max(float(timeout), 60.0)
        transient_retries = 0

        while True:
            merged_env = os.environ.copy()
            merged_env.update(self.env)
            if env_overrides:
                for key, value in env_overrides.items():
                    if value is None:
                        merged_env.pop(key, None)
                    else:
                        merged_env[key] = value

            try:
                completed = subprocess.run(
                    command,
                    cwd=self.root,
                    env=merged_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return CompletedCliRun(
                    command=command,
                    returncode=124,
                    stdout=_coerce_cli_text(exc.stdout),
                    stderr=_coerce_cli_text(exc.stderr) or f"Timed out after {timeout}s",
                    root=self.root,
                )

            result = CompletedCliRun(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                root=self.root,
            )
            if not allow_retry or result.returncode == 0:
                return result

            retry_delay = _retry_delay(result.output, transient_retries, deadline)
            if retry_delay is None:
                return result

            time.sleep(retry_delay)
            transient_retries += 1

    def write_prompt(self, name: str, text: str, *, toml: str | None = None) -> Path:
        prompt_path = self.prompts_dir / f"{name}.txt"
        prompt_path.write_text(text.strip() + "\n", encoding="utf-8")
        if toml is not None:
            (self.prompts_dir / f"{name}.toml").write_text(toml.strip() + "\n", encoding="utf-8")
        return prompt_path

    def write_json_fixture(self, name: str, payload: str) -> Path:
        fixture_path = self.root / name
        fixture_path.write_text(payload, encoding="utf-8")
        return fixture_path

    def start_chat(
        self,
        chat_id: str,
        *,
        prompt: str = "default",
        model: str = FREE_MODEL,
        user_id: str | None = None,
        env_overrides: dict[str, str | None] | None = None,
    ) -> CompletedCliRun:
        args = ["-c", chat_id, "--start", "--model", model, "--prompt", prompt]
        if user_id is not None:
            args.extend(["--user-id", user_id])
        return self.run_cli(*args, env_overrides=env_overrides)

    def send_message(
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
        args = ["-c", chat_id]
        if user_id is not None:
            args.extend(["--user-id", user_id])
        if debug:
            args.append("--debug")
        if stream_debug:
            args.append("--stream-debug")
        args.append(text)
        return self.run_cli(*args, timeout=timeout, env_overrides=env_overrides)

    def send_message_with_live_retry(
        self,
        chat_id: str,
        text: str,
        *,
        user_id: str | None = None,
        debug: bool = False,
        stream_debug: bool = False,
        env_overrides: dict[str, str | None] | None = None,
        max_attempts: int = 3,
        timeout: int = 120,
    ) -> CompletedCliRun:
        result: CompletedCliRun | None = None
        for attempt in range(1, max_attempts + 1):
            result = self.send_message(
                chat_id,
                text,
                user_id=user_id,
                debug=debug,
                stream_debug=stream_debug,
                env_overrides=env_overrides,
                timeout=timeout,
            )
            if not _should_retry_live_output(result) or attempt == max_attempts:
                return result
            time.sleep(min(2.0 * attempt, 5.0))

        assert result is not None
        return result

    def send_callback_with_live_retry(
        self,
        chat_id: str,
        callback_data: str,
        *,
        user_id: str | None = None,
        env_overrides: dict[str, str | None] | None = None,
        max_attempts: int = 3,
        timeout: int = 120,
    ) -> CompletedCliRun:
        result: CompletedCliRun | None = None
        for attempt in range(1, max_attempts + 1):
            result = self.send_callback(
                chat_id,
                callback_data,
                user_id=user_id,
                env_overrides=env_overrides,
                timeout=timeout,
            )
            if not _should_retry_live_output(result) or attempt == max_attempts:
                return result
            time.sleep(min(2.0 * attempt, 5.0))

        assert result is not None
        return result

    def send_callback(
        self,
        chat_id: str,
        callback_data: str,
        *,
        user_id: str | None = None,
        env_overrides: dict[str, str | None] | None = None,
        timeout: int = 60,
    ) -> CompletedCliRun:
        args = ["-c", chat_id]
        if user_id is not None:
            args.extend(["--user-id", user_id])
        args.extend(["--cb", callback_data])
        return self.run_cli(*args, timeout=timeout, env_overrides=env_overrides)

    def run_command(
        self,
        chat_id: str,
        command: str,
        *,
        args: str | None = None,
        user_id: str | None = None,
        env_overrides: dict[str, str | None] | None = None,
    ) -> CompletedCliRun:
        command_text = command if args is None else f"{command} {args}"
        cli_args = ["-c", chat_id]
        if user_id is not None:
            cli_args.extend(["--user-id", user_id])
        cli_args.extend(["--command", command_text])
        return self.run_cli(*cli_args, env_overrides=env_overrides)

    def read_history(self, chat_id: str) -> CompletedCliRun:
        return self.run_cli("-c", chat_id, "--history")

    def read_wiki(self, chat_id: str) -> CompletedCliRun:
        return self.run_cli("-c", chat_id, "--wiki")

    def repair_wiki(self, chat_id: str) -> CompletedCliRun:
        return self.run_cli("-c", chat_id, "--repair-wiki")

    def show_prompt(self, chat_id: str, *, real: bool = False) -> CompletedCliRun:
        args = ["-c", chat_id, "--show-prompt"]
        if real:
            args.append("--real")
        return self.run_cli(*args)

    def list_chats(self) -> CompletedCliRun:
        return self.run_cli("--list")

    def import_json(self, chat_id: str, json_path: Path) -> CompletedCliRun:
        return self.run_cli("-c", chat_id, "--import-json", str(json_path))

    def chat_data_dir(self, chat_id: str) -> Path:
        return self.data_dir / chat_id

    def chat_wiki_dir(self, chat_id: str) -> Path:
        return self.chat_data_dir(chat_id) / "wiki"


def _retry_delay(output: str, transient_retries: int, deadline: float) -> float | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None

    if _RATE_LIMIT_RE.search(output):
        return min(5.0 * (transient_retries + 1), remaining, 15.0)

    if transient_retries == 0 and _TRANSIENT_RE.search(output):
        return min(2.0, remaining)

    return None


def _coerce_cli_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _should_retry_live_output(result: CompletedCliRun) -> bool:
    if result.returncode == 124:
        return _TRANSIENT_RE.search(result.output) is not None
    if result.returncode != 0:
        return False
    if _LIVE_OUTPUT_TRANSIENT_RE.search(result.output) is not None:
        return True
    if (
        _LIVE_OUTPUT_MALFORMED_TOOLCALL_RE.search(result.output) is not None
        and _LIVE_OUTPUT_NO_TOOLS_RE.search(result.output) is not None
    ):
        return True
    return _LIVE_OUTPUT_PROVIDER_ERROR_RE.search(result.output) is not None
