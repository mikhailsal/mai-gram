"""Tests for application startup and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import os

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
    monkeypatch.setattr(main, "OpenRouterProvider", FakeProvider)
    monkeypatch.setattr(main, "ExternalMCPPool", FakePool)
    monkeypatch.setattr(main, "TelegramMessenger", FakeMessenger)
    monkeypatch.setattr(main, "BotHandler", fake_bot_handler)
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
