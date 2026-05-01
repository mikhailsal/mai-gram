"""Tests for application startup and shutdown lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import contextlib
import os
import sys
from types import SimpleNamespace

import pytest

from mai_gram import main


class FakeSettings:
    def __init__(
        self,
        bot_tokens: list[str],
        *,
        external_mcp_config: dict[str, object] | None = None,
    ) -> None:
        self.log_level = "INFO"
        self.openrouter_api_key = "test-key"
        self.llm_model = "openrouter/free"
        self.openrouter_base_url = "https://openrouter.example/v1"
        self.database_url = "sqlite+aiosqlite:///tmp/test.db"
        self.debug = False
        self.memory_data_dir = "./data"
        self.wiki_context_limit = 20
        self.short_term_limit = 500
        self.tool_max_iterations = 5
        self.models_config_path = "unused-models.toml"
        self._bot_tokens = bot_tokens
        self._external_mcp_config = external_mcp_config or {}

    def get_all_bot_tokens(self) -> list[str]:
        return self._bot_tokens

    def get_external_mcp_config(self) -> dict[str, object]:
        return self._external_mcp_config

    def get_bot_configs(self) -> list[object]:
        return [object()] if self._bot_tokens else []

    def get_bot_config_by_token(self, token: str) -> str:
        return f"config:{token}"

    def refresh_models_config(self) -> None:
        return None

    def get_allowed_models(self) -> list[str]:
        return ["openrouter/free"]


class FakeProvider:
    def __init__(self, *, api_key: str, default_model: str, base_url: str) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = base_url
        self.active_requests = 0
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class FakePool:
    def __init__(self, configs: dict[str, object]) -> None:
        self.configs = configs
        self.stop_all_calls = 0

    async def stop_all(self) -> None:
        self.stop_all_calls += 1


class FakeMessenger:
    def __init__(self, token: str) -> None:
        self.token = token
        self.bot_id = f"bot-{token[-4:]}"
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


def test_is_process_alive_reflects_os_kill(monkeypatch) -> None:
    monkeypatch.setattr(main.os, "kill", lambda pid, sig: None)
    assert main._is_process_alive(123) is True

    def raise_lookup(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(main.os, "kill", raise_lookup)
    assert main._is_process_alive(123) is False


def test_acquire_pid_lock_writes_pid_and_registers_cleanup(monkeypatch, tmp_path) -> None:
    pid_file = tmp_path / ".mai-gram.pid"
    registered: list[object] = []

    monkeypatch.setattr(main, "PID_FILE", pid_file)
    monkeypatch.setattr(main.os, "getpid", lambda: 321)
    monkeypatch.setattr(main.atexit, "register", registered.append)

    main._acquire_pid_lock()

    assert pid_file.read_text() == "321"
    assert registered == [main._release_pid_lock]


def test_acquire_pid_lock_rejects_running_instance(monkeypatch, tmp_path, capsys) -> None:
    pid_file = tmp_path / ".mai-gram.pid"
    pid_file.write_text("999")

    monkeypatch.setattr(main, "PID_FILE", pid_file)
    monkeypatch.setattr(main.os, "getpid", lambda: 321)
    monkeypatch.setattr(main, "_is_process_alive", lambda pid: True)

    with pytest.raises(SystemExit, match="1"):
        main._acquire_pid_lock()

    assert "already running" in capsys.readouterr().err


def test_acquire_pid_lock_replaces_stale_pid(monkeypatch, tmp_path) -> None:
    pid_file = tmp_path / ".mai-gram.pid"
    pid_file.write_text("999")

    monkeypatch.setattr(main, "PID_FILE", pid_file)
    monkeypatch.setattr(main.os, "getpid", lambda: 321)
    monkeypatch.setattr(main, "_is_process_alive", lambda pid: False)
    monkeypatch.setattr(main.atexit, "register", lambda callback: None)

    main._acquire_pid_lock()

    assert pid_file.read_text() == "321"


def test_release_pid_lock_only_unlinks_current_pid(monkeypatch, tmp_path) -> None:
    pid_file = tmp_path / ".mai-gram.pid"
    monkeypatch.setattr(main, "PID_FILE", pid_file)
    monkeypatch.setattr(main.os, "getpid", lambda: 321)

    pid_file.write_text("321")
    main._release_pid_lock()
    assert not pid_file.exists()

    pid_file.write_text("999")
    main._release_pid_lock()
    assert pid_file.exists()

    pid_file.write_text("not-a-pid")
    main._release_pid_lock()
    assert pid_file.exists()


def test_configure_logging_and_bot_token_validation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main.logging, "basicConfig", fake_basic_config)

    settings = FakeSettings(["token-1"])
    settings.log_level = "not-a-level"
    main._configure_logging(settings)

    assert captured["level"] == main.logging.INFO
    assert captured["stream"] is sys.stdout

    assert main._get_bot_tokens(FakeSettings(["token-1"])) == ["token-1"]

    with pytest.raises(SystemExit, match="1"):
        main._get_bot_tokens(FakeSettings([]))

    missing_key = FakeSettings(["token-1"])
    missing_key.openrouter_api_key = ""
    with pytest.raises(SystemExit, match="1"):
        main._get_bot_tokens(missing_key)


def test_build_external_mcp_pool_handles_empty_and_configured(monkeypatch) -> None:
    created_configs: list[dict[str, object]] = []

    class _CreatedPool(FakePool):
        def __init__(self, configs: dict[str, object]) -> None:
            super().__init__(configs)
            created_configs.append(configs)

    monkeypatch.setattr(main, "shared_build_external_mcp_pool", lambda settings: None)

    assert main._build_external_mcp_pool(FakeSettings(["token-1"])) is None

    settings = FakeSettings(["token-1"], external_mcp_config={"wiki": {"url": "http://mcp"}})
    monkeypatch.setattr(
        main,
        "shared_build_external_mcp_pool",
        lambda current_settings: _CreatedPool(current_settings.get_external_mcp_config()),
    )
    pool = main._build_external_mcp_pool(settings)

    assert isinstance(pool, _CreatedPool)
    assert created_configs == [{"wiki": {"url": "http://mcp"}}]


async def test_start_messengers_requires_initialized_runtime() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        await main._start_messengers(main.AppRuntime(), ["token-1"])


async def test_start_messengers_without_bot_config(monkeypatch) -> None:
    settings = FakeSettings(["token-1"])
    settings.get_bot_configs = lambda: []
    runtime = main.AppRuntime(
        settings=settings,
        llm_provider=FakeProvider(
            api_key="test-key",
            default_model="openrouter/free",
            base_url="https://openrouter.example/v1",
        ),
    )
    created_handlers: list[str | None] = []

    monkeypatch.setattr(main, "TelegramMessenger", FakeMessenger)
    monkeypatch.setattr(
        main,
        "build_bot_handler",
        lambda *args, **kwargs: created_handlers.append(kwargs.get("bot_config")),
    )

    await main._start_messengers(runtime, ["token-1"])

    assert created_handlers == [None]
    assert [messenger.token for messenger in runtime.messengers] == ["token-1"]


async def test_startup_initializes_runtime_and_shutdown_cleans_resources(monkeypatch) -> None:
    settings = FakeSettings(
        ["token-1", "token-2"], external_mcp_config={"wiki": {"url": "http://mcp"}}
    )
    runtime = main.AppRuntime()
    created_handlers: list[str | None] = []
    close_db_calls = 0
    release_pid_calls = 0

    async def fake_init_db(database_url: str, *, echo: bool):
        assert database_url == settings.database_url
        assert echo is False
        return object()

    async def fake_run_migrations(engine: object) -> None:
        assert engine is not None

    async def fake_close_db() -> None:
        nonlocal close_db_calls
        close_db_calls += 1

    async def fake_watch_config(watch_settings: FakeSettings) -> None:
        assert watch_settings is settings
        await asyncio.Future()

    def fake_release_pid_lock() -> None:
        nonlocal release_pid_calls
        release_pid_calls += 1

    def fake_bot_handler(*args, **kwargs):
        created_handlers.append(kwargs.get("bot_config"))
        return object()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "_configure_logging", lambda configured_settings: None)
    monkeypatch.setattr(main, "_acquire_pid_lock", lambda: None)
    monkeypatch.setattr(main, "_release_pid_lock", fake_release_pid_lock)
    monkeypatch.setattr(main, "init_db", fake_init_db)
    monkeypatch.setattr(main, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(main, "close_db", fake_close_db)
    monkeypatch.setattr(
        main,
        "build_openrouter_provider",
        lambda current_settings: FakeProvider(
            api_key=current_settings.openrouter_api_key,
            default_model=current_settings.llm_model,
            base_url=current_settings.openrouter_base_url,
        ),
    )
    monkeypatch.setattr(main, "shared_build_external_mcp_pool", FakePool)
    monkeypatch.setattr(main, "TelegramMessenger", FakeMessenger)
    monkeypatch.setattr(main, "build_bot_handler", fake_bot_handler)
    monkeypatch.setattr(main, "_watch_config", fake_watch_config)

    await main.startup(runtime)

    assert runtime.settings is settings
    assert isinstance(runtime.llm_provider, FakeProvider)
    assert runtime.external_mcp_pool is not None
    assert [messenger.token for messenger in runtime.messengers] == ["token-1", "token-2"]
    assert [messenger.start_calls for messenger in runtime.messengers] == [1, 1]
    assert created_handlers == ["config:token-1", "config:token-2"]
    assert runtime.config_watcher_task is not None

    await main.shutdown(runtime)
    await asyncio.sleep(0)

    assert runtime.settings is None
    assert runtime.llm_provider is None
    assert runtime.external_mcp_pool is None
    assert runtime.messengers == []
    assert runtime.config_watcher_task is None
    assert close_db_calls == 1
    assert release_pid_calls == 1


async def test_shutdown_is_idempotent(monkeypatch) -> None:
    release_pid_calls = 0
    close_db_calls = 0

    async def fake_close_db() -> None:
        nonlocal close_db_calls
        close_db_calls += 1

    def fake_release_pid_lock() -> None:
        nonlocal release_pid_calls
        release_pid_calls += 1

    provider = FakeProvider(api_key="key", default_model="model", base_url="https://example.test")
    provider.active_requests = 1
    pool = FakePool({"wiki": {"url": "http://mcp"}})
    messenger = FakeMessenger("token-1")
    task = asyncio.create_task(asyncio.sleep(3600))
    runtime = main.AppRuntime(
        settings=FakeSettings(["token-1"]),
        messengers=[messenger],
        llm_provider=provider,
        external_mcp_pool=pool,
        config_watcher_task=task,
    )

    monkeypatch.setattr(main, "close_db", fake_close_db)
    monkeypatch.setattr(main, "_release_pid_lock", fake_release_pid_lock)

    await main.shutdown(runtime)
    await main.shutdown(runtime)
    await asyncio.sleep(0)

    assert messenger.stop_calls == 1
    assert provider.close_calls == 1
    assert pool.stop_all_calls == 1
    assert runtime.messengers == []
    assert runtime.llm_provider is None
    assert runtime.external_mcp_pool is None
    assert runtime.config_watcher_task is None
    assert close_db_calls == 2
    assert release_pid_calls == 2


async def test_watch_config_refreshes_models_on_change(monkeypatch, tmp_path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text("[models]\nallowed=['openrouter/free']\n", encoding="utf-8")
    refreshed = asyncio.Event()
    settings = FakeSettings(["token-1"])
    settings.models_config_path = str(models_path)

    def refresh_models_config() -> None:
        refreshed.set()

    settings.refresh_models_config = refresh_models_config
    real_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(main._watch_config(settings))
    await real_sleep(0)

    current_mtime = models_path.stat().st_mtime
    os.utime(models_path, (current_mtime + 1, current_mtime + 1))

    await asyncio.wait_for(refreshed.wait(), timeout=1)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_watch_config_recovers_from_value_error(monkeypatch, tmp_path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text("[models]\nallowed=['openrouter/free']\n", encoding="utf-8")
    settings = FakeSettings(["token-1"])
    settings.models_config_path = str(models_path)
    real_sleep = asyncio.sleep

    def refresh_models_config() -> None:
        raise ValueError("invalid models config")

    settings.refresh_models_config = refresh_models_config

    async def fast_sleep(delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(main._watch_config(settings))
    await real_sleep(0)

    current_mtime = models_path.stat().st_mtime
    os.utime(models_path, (current_mtime + 1, current_mtime + 1))
    await real_sleep(0.01)

    assert not task.done()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_run_registers_signals_and_shuts_down_cleanly(monkeypatch) -> None:
    runtime = main.AppRuntime()
    registered_signals: list[object] = []
    shutdown_calls = 0

    class _Loop:
        def add_signal_handler(self, sig: object, callback: object) -> None:
            del callback
            registered_signals.append(sig)

    async def fake_startup(current_runtime: main.AppRuntime) -> None:
        assert current_runtime is runtime
        runtime.messengers.append(FakeMessenger("token-1"))

    async def fake_shutdown(current_runtime: main.AppRuntime) -> None:
        nonlocal shutdown_calls
        assert current_runtime is runtime
        shutdown_calls += 1
        runtime.messengers.clear()

    async def fake_sleep(delay: float) -> None:
        del delay
        runtime.messengers.clear()

    monkeypatch.setattr(main, "AppRuntime", lambda: runtime)
    monkeypatch.setattr(main.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(main, "startup", fake_startup)
    monkeypatch.setattr(main, "shutdown", fake_shutdown)
    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(main.sys, "platform", "linux")

    await main.run()

    assert registered_signals == [main.signal.SIGINT, main.signal.SIGTERM]
    assert shutdown_calls == 1


async def test_run_handles_keyboard_interrupt(monkeypatch) -> None:
    shutdown_calls = 0

    class _Loop:
        def add_signal_handler(self, sig: object, callback: object) -> None:
            del sig, callback

    async def fake_shutdown(current_runtime: main.AppRuntime) -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1

    async def fake_startup(current_runtime: main.AppRuntime) -> None:
        del current_runtime
        raise KeyboardInterrupt

    monkeypatch.setattr(main.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(main, "startup", fake_startup)
    monkeypatch.setattr(main, "shutdown", fake_shutdown)
    monkeypatch.setattr(main.sys, "platform", "linux")

    await main.run()

    assert shutdown_calls == 1


async def test_run_handles_unexpected_errors_on_windows(monkeypatch) -> None:
    shutdown_calls = 0
    logged: list[str] = []

    class _Loop:
        def add_signal_handler(self, sig: object, callback: object) -> None:
            raise AssertionError(f"unexpected signal registration: {sig}, {callback}")

    async def fake_shutdown(current_runtime: main.AppRuntime) -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1

    async def fake_startup(current_runtime: main.AppRuntime) -> None:
        del current_runtime
        raise RuntimeError("boom")

    monkeypatch.setattr(main.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(main, "startup", fake_startup)
    monkeypatch.setattr(main, "shutdown", fake_shutdown)
    monkeypatch.setattr(main.logger, "exception", logged.append)
    monkeypatch.setattr(main.sys, "platform", "win32")

    await main.run()

    assert shutdown_calls == 1
    assert logged == ["Unexpected error"]


def test_run_with_reload_import_error_prints_help(monkeypatch, capsys) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "watchfiles":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit, match="1"):
        main._run_with_reload()

    assert "watchfiles" in capsys.readouterr().err


def test_run_with_reload_invokes_watchfiles(monkeypatch, tmp_path, capsys) -> None:
    src_dir = tmp_path / "src"
    package_dir = src_dir / "mai_gram"
    package_dir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "prompts").mkdir()
    fake_file = package_dir / "main.py"
    fake_file.write_text("# test\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _DefaultFilter:
        pass

    class _PythonFilter:
        def __init__(self) -> None:
            self.extensions = (".py",)

    def fake_run_process(*watch_dirs: object, **kwargs: object) -> None:
        captured["watch_dirs"] = watch_dirs
        captured.update(kwargs)
        callback = kwargs["callback"]
        callback({("modified", str(tmp_path / "config" / "models.toml"))})

    monkeypatch.setitem(
        sys.modules,
        "watchfiles",
        SimpleNamespace(
            DefaultFilter=_DefaultFilter,
            PythonFilter=_PythonFilter,
            run_process=fake_run_process,
        ),
    )
    monkeypatch.setattr(main, "__file__", str(fake_file))

    main._run_with_reload()

    watch_dirs = captured["watch_dirs"]
    assert watch_dirs == (package_dir.parent, tmp_path / "config", tmp_path / "prompts")
    assert captured["target"] == f"{sys.executable} -m mai_gram.main"
    assert captured["target_type"] == "command"
    assert ".toml" in captured["watch_filter"].allowed_extensions
    output = capsys.readouterr().out
    assert "Auto-reload enabled" in output
    assert "Detected changes in: config/models.toml" in output


def test_reload_filter_uses_real_pythonfilter_api(monkeypatch, tmp_path) -> None:
    """PythonFilter exposes extensions as an *instance* attribute, not a class attribute.

    The production code must not access PythonFilter.allowed_extensions (doesn't exist);
    it should use PythonFilter().extensions instead.  This test mocks PythonFilter to
    match the real watchfiles API so we catch the mismatch.
    """
    src_dir = tmp_path / "src"
    package_dir = src_dir / "mai_gram"
    package_dir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    fake_file = package_dir / "main.py"
    fake_file.write_text("# test\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _DefaultFilter:
        pass

    class _PythonFilter:
        def __init__(self) -> None:
            self.extensions = (".py", ".pyx", ".pyd")

    def fake_run_process(*watch_dirs: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "watchfiles",
        SimpleNamespace(
            DefaultFilter=_DefaultFilter,
            PythonFilter=_PythonFilter,
            run_process=fake_run_process,
        ),
    )
    monkeypatch.setattr(main, "__file__", str(fake_file))

    main._run_with_reload()

    watch_filter = captured["watch_filter"]
    assert ".py" in watch_filter.allowed_extensions
    assert ".pyx" in watch_filter.allowed_extensions
    assert ".toml" in watch_filter.allowed_extensions
    assert ".md" in watch_filter.allowed_extensions


def test_parse_args_and_main_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mai-gram", "--reload"])
    assert main._parse_args() == argparse.Namespace(reload=True)

    branch_calls: list[str] = []

    monkeypatch.setattr(main, "_parse_args", lambda: argparse.Namespace(reload=True))
    monkeypatch.setattr(main, "_run_with_reload", lambda: branch_calls.append("reload"))
    main.main()

    async def fake_run() -> None:
        branch_calls.append("run")

    def fake_asyncio_run(coro: object) -> None:
        branch_calls.append("asyncio.run")
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(main, "_parse_args", lambda: argparse.Namespace(reload=False))
    monkeypatch.setattr(main.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(main, "run", fake_run)
    main.main()

    assert branch_calls == ["reload", "asyncio.run"]
