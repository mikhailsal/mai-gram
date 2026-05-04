"""In-process and subprocess helpers for black-box functional CLI tests.

By default the harness runs ``mai-chat`` **in-process** to avoid the ~1.4s
Python startup + import overhead per invocation.  Set the environment variable
``MAI_FUNCTIONAL_SUBPROCESS=1`` to fall back to the legacy subprocess path.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
_LIVE_OUTPUT_MALFORMED_TOOLCALL_RE = re.compile(
    r"(?:toolcall|olcall|call)(?:[>\]]|&gt;)\s*\[",
    re.IGNORECASE,
)
_LIVE_OUTPUT_NO_TOOLS_RE = re.compile(r"tools used:\s*none", re.IGNORECASE)
_LIVE_OUTPUT_PROVIDER_ERROR_RE = re.compile(
    r"(?:ai provider error|something went wrong with the ai provider|tap regenerate to retry)",
    re.IGNORECASE,
)

_USE_SUBPROCESS = os.environ.get("MAI_FUNCTIONAL_SUBPROCESS", "") == "1"


class _CliTimeoutError(Exception):
    def __init__(self, stdout: bytes | None, stderr: bytes | None) -> None:
        super().__init__("CLI command timed out")
        self.stdout = stdout
        self.stderr = stderr


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


def _run_inprocess(
    argv: tuple[str, ...], *, env: dict[str, str], cwd: Path
) -> tuple[int, str, str]:
    """Run the CLI entry-point in the current process, returning (rc, stdout, stderr)."""
    from mai_gram.config import reset_settings
    from mai_gram.console_cli import build_parser
    from mai_gram.console_runner import _run
    from mai_gram.db import close_db, reset_db_state

    saved_env = {k: os.environ.get(k) for k in env}
    saved_cwd = Path.cwd()
    os.environ.update(env)
    os.chdir(cwd)

    reset_settings()
    reset_db_state()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    returncode = 0
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            parser = build_parser()
            parsed = parser.parse_args(list(argv))
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
            asyncio.run(_run(parsed))
    except SystemExit as exc:
        returncode = _extract_exit_code(exc, stderr_buf)
    except Exception as exc:
        returncode = 1
        stderr_buf.write(f"{exc}\n")
    finally:
        with contextlib.suppress(Exception):
            asyncio.run(close_db())
        reset_db_state()
        reset_settings()

        os.chdir(saved_cwd)
        for key, old_val in saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val

    return returncode, stdout_buf.getvalue(), stderr_buf.getvalue()


def _extract_exit_code(exc: SystemExit, stderr_buf: io.StringIO) -> int:
    """Convert a SystemExit to (returncode), writing messages to stderr_buf."""
    code = exc.code
    if code is None or code == 0:
        return 0
    if isinstance(code, int):
        return code
    stderr_buf.write(str(code) + "\n")
    return 1


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

            if _USE_SUBPROCESS:
                returncode, stdout, stderr = self._run_subprocess(
                    args, merged_env=merged_env, timeout=timeout
                )
            else:
                returncode, stdout, stderr = _run_inprocess(args, env=merged_env, cwd=self.root)

            result = CompletedCliRun(
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                root=self.root,
            )
            if not allow_retry or result.returncode == 0:
                return result

            retry_delay = _retry_delay(result.output, transient_retries, deadline)
            if retry_delay is None:
                return result

            time.sleep(retry_delay)
            transient_retries += 1

    def _run_subprocess(
        self,
        args: tuple[str, ...],
        *,
        merged_env: dict[str, str],
        timeout: int,
    ) -> tuple[int, str, str]:
        command = (self.cli_path, *args)
        try:
            return asyncio.run(
                _run_cli_command(command, cwd=self.root, env=merged_env, timeout=timeout)
            )
        except _CliTimeoutError as exc:
            return (
                124,
                _coerce_cli_text(exc.stdout),
                _coerce_cli_text(exc.stderr) or f"Timed out after {timeout}s",
            )

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
        template: str = "empty",
        template_params: dict[str, str] | None = None,
        user_id: str | None = None,
        env_overrides: dict[str, str | None] | None = None,
    ) -> CompletedCliRun:
        args = [
            "-c",
            chat_id,
            "--start",
            "--model",
            model,
            "--prompt",
            prompt,
            "--template",
            template,
        ]
        if template_params:
            args.append("--template-params")
            args.extend(f"{k}={v}" for k, v in template_params.items())
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

    def import_json(
        self,
        chat_id: str,
        json_path: Path,
        *,
        reasoning_template: str | None = None,
    ) -> CompletedCliRun:
        args = ["-c", chat_id, "--import-json", str(json_path)]
        if reasoning_template:
            args.extend(["--reasoning-template", reasoning_template])
        return self.run_cli(*args)

    def chat_data_dir(self, chat_id: str) -> Path:
        return self.data_dir / chat_id

    def chat_wiki_dir(self, chat_id: str) -> Path:
        return self.chat_data_dir(chat_id) / "wiki"


async def _run_cli_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        process.kill()
        stdout, stderr = await process.communicate()
        raise _CliTimeoutError(stdout, stderr) from exc

    return process.returncode or 0, _coerce_cli_text(stdout), _coerce_cli_text(stderr)


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
