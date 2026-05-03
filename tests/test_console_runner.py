"""Tests for the console CLI helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram import console_runner
from mai_gram.console_runner import _incoming_command, _parse_command_text
from mai_gram.core.importer import ImportDataError
from mai_gram.llm.provider import ChatMessage, LLMAuthenticationError, MessageRole, ToolCall


@asynccontextmanager
async def _session_context(session):
    yield session


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: object) -> None:
    monkeypatch.setattr(console_runner, "get_session", lambda: _session_context(session))


def _raise(exc: Exception) -> None:
    raise exc


def test_parse_command_text_without_args() -> None:
    command, command_args = _parse_command_text("help")

    assert command == "help"
    assert command_args is None


def test_parse_command_text_with_slash_and_args() -> None:
    command, command_args = _parse_command_text("/timezone Europe/Moscow")

    assert command == "timezone"
    assert command_args == "Europe/Moscow"


def test_parse_command_text_rejects_empty_input() -> None:
    with pytest.raises(SystemExit, match="--command requires a command name"):
        _parse_command_text("   ")


def test_incoming_command_includes_command_args() -> None:
    incoming = _incoming_command("test-chat", "test-user", "timezone", "Europe/Moscow")

    assert incoming.command == "timezone"
    assert incoming.command_args == "Europe/Moscow"
    assert incoming.text == "/timezone Europe/Moscow"


def test_format_timestamp_handles_none_naive_and_aware_datetimes() -> None:
    naive = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).replace(tzinfo=None)
    aware = datetime(2024, 1, 2, 6, 4, 5, tzinfo=timezone(timedelta(hours=3)))

    assert console_runner._format_timestamp(None) == "---- -- --:--:--"
    assert console_runner._format_timestamp(naive) == "2024-01-02 03:04:05"
    assert console_runner._format_timestamp(aware) == "2024-01-02 03:04:05"


@pytest.mark.asyncio
async def test_offline_cli_provider_rejects_live_calls_and_counts_tokens() -> None:
    provider = console_runner._OfflineCLIProvider()

    with pytest.raises(LLMAuthenticationError):
        await provider.generate([])

    with pytest.raises(LLMAuthenticationError):
        await anext(provider.generate_stream([]))

    token_count = await provider.count_tokens(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="world"),
        ]
    )

    assert token_count == (16 + 5 + 16 + 5) // 4
    assert await provider.close() is None


@pytest.mark.asyncio
async def test_print_chat_list_formats_empty_and_non_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(
                all=MagicMock(
                    return_value=[
                        SimpleNamespace(
                            id="chat-1",
                            llm_model="openrouter/free",
                            message_count=3,
                            last_message=datetime(
                                2024,
                                1,
                                2,
                                3,
                                4,
                                5,
                                tzinfo=timezone.utc,
                            ).replace(tzinfo=None),
                        )
                    ]
                )
            ),
        ]
    )
    _patch_session(monkeypatch, session)

    await console_runner._print_chat_list()
    await console_runner._print_chat_list()

    output = capsys.readouterr().out
    assert "(no chats found)" in output
    assert "chat-1" in output
    assert "openrouter/free" in output


@pytest.mark.asyncio
async def test_print_history_formats_empty_and_non_empty_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = MagicMock()
    _patch_session(monkeypatch, session)
    inspection_service = MagicMock(
        list_history=AsyncMock(
            side_effect=[
                [],
                [
                    SimpleNamespace(
                        timestamp=datetime(
                            2024,
                            1,
                            2,
                            3,
                            4,
                            5,
                            tzinfo=timezone.utc,
                        ).replace(tzinfo=None),
                        role="user",
                        content="hello",
                    )
                ],
            ]
        )
    )
    monkeypatch.setattr(
        console_runner,
        "ChatInspectionService",
        lambda *args, **kwargs: inspection_service,
    )

    await console_runner._print_history("chat-1")
    await console_runner._print_history("chat-1")

    output = capsys.readouterr().out
    assert "=== History: chat-1 ===" in output
    assert "(no messages)" in output
    assert "USER: hello" in output


@pytest.mark.asyncio
async def test_print_wiki_handles_sync_and_empty_entries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = MagicMock(commit=AsyncMock())
    _patch_session(monkeypatch, session)
    inspection_service = MagicMock(
        list_wiki=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    entries=[SimpleNamespace(key="topic", value="value", importance=1)],
                    sync_report=SimpleNamespace(total_changes=1, summary=lambda: "updated 1"),
                ),
                SimpleNamespace(
                    entries=[],
                    sync_report=SimpleNamespace(total_changes=0, summary=lambda: "none"),
                ),
            ]
        )
    )
    monkeypatch.setattr(
        console_runner,
        "ChatInspectionService",
        lambda *args, **kwargs: inspection_service,
    )

    await console_runner._print_wiki("chat-1", "./data")
    await console_runner._print_wiki("chat-1", "./data")

    output = capsys.readouterr().out
    assert "[sync] updated 1" in output
    assert "- (1) topic: value" in output
    assert "(no wiki entries)" in output
    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_repair_wiki_reports_no_changes_and_applied_changes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = MagicMock(commit=AsyncMock())
    _patch_session(monkeypatch, session)
    inspection_service = MagicMock(
        repair_wiki=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    total_changes=0,
                    created=[],
                    updated=[],
                    db_rows_deleted=[],
                    skipped_files=[],
                    summary=lambda: "none",
                ),
                SimpleNamespace(
                    total_changes=4,
                    created=["a"],
                    updated=["b"],
                    db_rows_deleted=["c"],
                    skipped_files=["d.md"],
                    summary=lambda: "created 1, updated 1, deleted 1, skipped 1",
                ),
            ]
        )
    )
    monkeypatch.setattr(
        console_runner,
        "ChatInspectionService",
        lambda *args, **kwargs: inspection_service,
    )

    await console_runner._repair_wiki("chat-1", "./data")
    await console_runner._repair_wiki("chat-1", "./data")

    output = capsys.readouterr().out
    assert "Database is already in sync with disk files." in output
    assert "Result: created 1, updated 1, deleted 1, skipped 1" in output
    assert "  + created: a" in output
    assert "  ~ updated: b" in output
    assert "  - removed orphan DB row: c" in output
    assert "  ? skipped unparseable file: d.md" in output
    assert session.commit.await_count == 2


@pytest.mark.asyncio
async def test_print_prompt_handles_missing_chat_and_formats_preview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = MagicMock()
    _patch_session(monkeypatch, session)
    preview = SimpleNamespace(
        context=[
            ChatMessage(role=MessageRole.SYSTEM, content="system prompt"),
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.TOOL, content="tool output", tool_call_id="call-1"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="assistant text",
                tool_calls=[
                    ToolCall(id="tc-1", name="wiki_search", arguments='{"term": "foo"}'),
                    ToolCall(id="tc-2", name="broken", arguments="not-json"),
                ],
            ),
        ],
        tools=[SimpleNamespace(name="wiki_search", description="Search wiki")],
        token_count=42,
    )
    missing_service = MagicMock(build_preview=AsyncMock(side_effect=LookupError("missing")))
    preview_service = MagicMock(build_preview=AsyncMock(return_value=preview))
    services = iter([missing_service, preview_service])
    service_calls: list[dict[str, object]] = []

    def fake_prompt_preview_service(*args, **kwargs):
        del args
        service_calls.append(kwargs)
        return next(services)

    monkeypatch.setattr(console_runner, "PromptPreviewService", fake_prompt_preview_service)

    with pytest.raises(SystemExit, match="no chat found"):
        await console_runner._print_prompt(
            "chat-1",
            "./data",
            MagicMock(),
            MagicMock(),
            external_mcp_pool="pool",
        )

    await console_runner._print_prompt(
        "chat-1",
        "./data",
        MagicMock(),
        MagicMock(),
        external_mcp_pool="pool",
    )

    output = capsys.readouterr().out
    assert "system prompt" in output
    assert "- wiki_search: Search wiki" in output
    assert "[tool result:call-1] tool output" in output
    assert "[tool call:tc-1] wiki_search(term='foo')" in output
    assert "[tool call:tc-2] broken(not-json)" in output
    assert "Approx tokens: 42" in output
    assert all(call["external_mcp_pool"] == "pool" for call in service_calls)


@pytest.mark.asyncio
async def test_import_json_dialogue_covers_error_and_success_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="file not found"):
        await console_runner._import_json_dialogue("chat-1", str(missing_path))

    unreadable_path = tmp_path / "unreadable.json"
    unreadable_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": _raise(OSError("boom")),
    )
    with pytest.raises(SystemExit, match="cannot read file"):
        await console_runner._import_json_dialogue("chat-1", str(unreadable_path))

    monkeypatch.undo()

    parse_error_path = tmp_path / "parse.json"
    parse_error_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        console_runner,
        "parse_import_payload",
        lambda text: _raise(ImportDataError("bad payload")),
    )
    with pytest.raises(SystemExit, match="bad payload"):
        await console_runner._import_json_dialogue("chat-1", str(parse_error_path))

    session = MagicMock(commit=AsyncMock())
    _patch_session(monkeypatch, session)
    success_path = tmp_path / "ok.json"
    success_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(console_runner, "parse_import_payload", lambda text: {"messages": []})
    monkeypatch.setattr(
        console_runner,
        "import_into_existing_chat",
        AsyncMock(side_effect=LookupError("missing-chat")),
    )
    with pytest.raises(SystemExit, match="Run --start first"):
        await console_runner._import_json_dialogue("chat-1", str(success_path))

    monkeypatch.setattr(
        console_runner,
        "import_into_existing_chat",
        AsyncMock(return_value=SimpleNamespace(imported_count=0)),
    )
    with pytest.raises(SystemExit, match="no messages could be imported"):
        await console_runner._import_json_dialogue("chat-1", str(success_path))

    monkeypatch.setattr(
        console_runner,
        "import_into_existing_chat",
        AsyncMock(return_value=SimpleNamespace(imported_count=3)),
    )
    imported_count = await console_runner._import_json_dialogue("chat-1", str(success_path))

    assert imported_count == 3
    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_handle_console_inspection_dispatches_requested_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = MagicMock(memory_data_dir="./data")
    monkeypatch.setattr(console_runner, "_print_history", AsyncMock())
    monkeypatch.setattr(console_runner, "_repair_wiki", AsyncMock())
    monkeypatch.setattr(console_runner, "_print_wiki", AsyncMock())
    monkeypatch.setattr(console_runner, "_import_json_dialogue", AsyncMock(return_value=5))

    assert await console_runner._handle_console_inspection(
        SimpleNamespace(history=True, repair_wiki=False, wiki=False, import_json=None),
        "chat-1",
        settings,
    )
    assert await console_runner._handle_console_inspection(
        SimpleNamespace(history=False, repair_wiki=True, wiki=False, import_json=None),
        "chat-1",
        settings,
    )
    assert await console_runner._handle_console_inspection(
        SimpleNamespace(history=False, repair_wiki=False, wiki=True, import_json=None),
        "chat-1",
        settings,
    )
    assert await console_runner._handle_console_inspection(
        SimpleNamespace(history=False, repair_wiki=False, wiki=False, import_json="file.json"),
        "chat-1",
        settings,
    )
    assert not await console_runner._handle_console_inspection(
        SimpleNamespace(history=False, repair_wiki=False, wiki=False, import_json=None),
        "chat-1",
        settings,
    )

    assert "Imported 5 messages into chat 'chat-1'." in capsys.readouterr().out


def test_build_cli_llm_covers_missing_key_offline_and_debug_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(console_runner, "needs_live_llm", lambda args: True)

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        console_runner._build_cli_llm(
            SimpleNamespace(debug=False),
            "chat-1",
            MagicMock(openrouter_api_key=""),
        )

    monkeypatch.setattr(console_runner, "needs_live_llm", lambda args: False)
    offline_llm, offline_logger = console_runner._build_cli_llm(
        SimpleNamespace(debug=False),
        "chat-1",
        MagicMock(openrouter_api_key="", memory_data_dir="./data"),
    )

    provider = MagicMock()
    logger_provider = MagicMock()
    monkeypatch.setattr(console_runner, "build_openrouter_provider", lambda settings: provider)
    monkeypatch.setattr(
        console_runner,
        "LLMLoggerProvider",
        lambda *args, **kwargs: logger_provider,
    )
    debug_llm, debug_logger = console_runner._build_cli_llm(
        SimpleNamespace(debug=True),
        "chat-1",
        MagicMock(openrouter_api_key="test-key", memory_data_dir="./data"),
    )

    assert isinstance(offline_llm, console_runner._OfflineCLIProvider)
    assert offline_logger is None
    assert debug_llm is logger_provider
    assert debug_logger is logger_provider


def test_build_cli_llm_uses_shared_provider_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    settings = MagicMock(openrouter_api_key="test-key", memory_data_dir="./data")
    args = SimpleNamespace(debug=False)

    monkeypatch.setattr(console_runner, "needs_live_llm", lambda current_args: True)
    monkeypatch.setattr(console_runner, "build_openrouter_provider", lambda current: provider)

    llm, logger_provider = console_runner._build_cli_llm(args, "chat-1", settings)

    assert llm is provider
    assert logger_provider is None


@pytest.mark.asyncio
async def test_dispatch_console_runtime_uses_shared_handler_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messenger = MagicMock()
    messenger.dispatch_text = AsyncMock()
    messenger.dispatch_message = AsyncMock()
    messenger.dispatch_callback = AsyncMock()
    messenger.flush_edits = MagicMock()
    handler_calls: list[dict[str, object]] = []
    external_mcp_pool = MagicMock()

    monkeypatch.setattr(console_runner, "ConsoleMessenger", lambda **kwargs: messenger)
    monkeypatch.setattr(
        console_runner,
        "build_bot_handler",
        lambda *args, **kwargs: handler_calls.append(kwargs),
    )

    args = SimpleNamespace(
        stream_debug=False,
        real=False,
        start=False,
        model=None,
        prompt=None,
        command=None,
        callbacks=[],
        message="hello",
    )

    await console_runner._dispatch_console_runtime(
        args,
        chat_id="chat-1",
        user_id="user-1",
        llm=MagicMock(),
        settings=MagicMock(),
        external_mcp_pool=external_mcp_pool,
    )

    assert handler_calls == [{"test_mode": True, "external_mcp_pool": external_mcp_pool}]
    messenger.dispatch_text.assert_awaited_once()
    messenger.flush_edits.assert_called_once_with()


@pytest.mark.asyncio
async def test_run_builds_and_stops_external_mcp_pool_for_prompt_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock(
        database_url="sqlite+aiosqlite:///tmp/test.db",
        debug=False,
        memory_data_dir="./data",
    )
    llm = MagicMock(close=AsyncMock())
    pool = MagicMock(stop_all=AsyncMock())
    args = SimpleNamespace(
        list=False,
        show_prompt=True,
        real=False,
        history=False,
        repair_wiki=False,
        wiki=False,
        import_json=None,
        debug=False,
        start=False,
        model=None,
        prompt=None,
        command=None,
        callbacks=[],
        message=None,
        stream_debug=False,
    )

    monkeypatch.setattr(console_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(
        console_runner,
        "resolve_user_id",
        lambda current_args, current_settings: "user-1",
    )
    monkeypatch.setattr(
        console_runner,
        "resolve_chat_id",
        lambda current_args, state_store: "chat-1",
    )
    monkeypatch.setattr(console_runner, "init_db", AsyncMock(return_value=object()))
    monkeypatch.setattr(console_runner, "run_migrations", AsyncMock())
    monkeypatch.setattr(console_runner, "close_db", AsyncMock())
    monkeypatch.setattr(console_runner, "build_external_mcp_pool", lambda current_settings: pool)
    monkeypatch.setattr(console_runner, "_handle_console_inspection", AsyncMock(return_value=False))
    monkeypatch.setattr(
        console_runner,
        "_build_cli_llm",
        lambda current_args, chat_id, current_settings: (llm, None),
    )
    print_prompt = AsyncMock()
    monkeypatch.setattr(console_runner, "_print_prompt", print_prompt)

    await console_runner._run(args)

    print_prompt.assert_awaited_once()
    assert print_prompt.await_args.kwargs["external_mcp_pool"] is pool
    pool.stop_all.assert_awaited_once_with()
    llm.close.assert_awaited_once_with()
    console_runner.close_db.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_handles_list_and_inspection_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock(database_url="sqlite+aiosqlite:///tmp/test.db", debug=False)
    args_list = SimpleNamespace(list=True)
    args_inspect = SimpleNamespace(list=False)

    monkeypatch.setattr(console_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(console_runner, "init_db", AsyncMock(return_value=object()))
    monkeypatch.setattr(console_runner, "run_migrations", AsyncMock())
    monkeypatch.setattr(console_runner, "close_db", AsyncMock())
    print_chat_list = AsyncMock()
    monkeypatch.setattr(console_runner, "_print_chat_list", print_chat_list)

    await console_runner._run(args_list)

    print_chat_list.assert_awaited_once_with()
    assert console_runner.close_db.await_count == 1

    monkeypatch.setattr(console_runner, "resolve_user_id", lambda args, current_settings: "user-1")
    monkeypatch.setattr(console_runner, "resolve_chat_id", lambda args, state_store: "chat-1")
    monkeypatch.setattr(console_runner, "_handle_console_inspection", AsyncMock(return_value=True))
    build_external_mcp_pool = MagicMock()
    monkeypatch.setattr(console_runner, "build_external_mcp_pool", build_external_mcp_pool)

    await console_runner._run(args_inspect)

    build_external_mcp_pool.assert_not_called()
    assert console_runner.close_db.await_count == 2


@pytest.mark.asyncio
async def test_run_dispatches_runtime_and_prints_debug_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock(
        database_url="sqlite+aiosqlite:///tmp/test.db",
        debug=False,
        memory_data_dir="./data",
    )
    llm = MagicMock(close=AsyncMock())
    pool = MagicMock(stop_all=AsyncMock())
    logger_provider = MagicMock(get_session_stats=MagicMock(return_value={"calls": 1}))
    args = SimpleNamespace(
        list=False,
        show_prompt=False,
        real=False,
        history=False,
        repair_wiki=False,
        wiki=False,
        import_json=None,
        debug=True,
        start=False,
        model=None,
        prompt=None,
        command=None,
        callbacks=[],
        message="hello",
        stream_debug=False,
    )

    monkeypatch.setattr(console_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(console_runner, "resolve_user_id", lambda args, current_settings: "user-1")
    monkeypatch.setattr(console_runner, "resolve_chat_id", lambda args, state_store: "chat-1")
    monkeypatch.setattr(console_runner, "init_db", AsyncMock(return_value=object()))
    monkeypatch.setattr(console_runner, "run_migrations", AsyncMock())
    monkeypatch.setattr(console_runner, "close_db", AsyncMock())
    monkeypatch.setattr(console_runner, "_handle_console_inspection", AsyncMock(return_value=False))
    monkeypatch.setattr(console_runner, "build_external_mcp_pool", lambda settings: pool)
    monkeypatch.setattr(
        console_runner,
        "_build_cli_llm",
        lambda args, chat_id, settings: (llm, logger_provider),
    )
    dispatch_runtime = AsyncMock()
    monkeypatch.setattr(console_runner, "_dispatch_console_runtime", dispatch_runtime)
    print_stats = MagicMock()
    monkeypatch.setattr(console_runner, "print_debug_session_stats", print_stats)

    await console_runner._run(args)

    dispatch_runtime.assert_awaited_once()
    print_stats.assert_called_once_with({"calls": 1})
    pool.stop_all.assert_awaited_once_with()
    llm.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_dispatch_console_runtime_covers_start_command_callbacks_and_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messenger = MagicMock()
    messenger.dispatch_text = AsyncMock()
    messenger.dispatch_message = AsyncMock()
    messenger.dispatch_callback = AsyncMock()
    messenger.flush_edits = MagicMock()

    monkeypatch.setattr(console_runner, "ConsoleMessenger", lambda **kwargs: messenger)
    monkeypatch.setattr(console_runner, "build_bot_handler", lambda *args, **kwargs: object())

    args = SimpleNamespace(
        stream_debug=False,
        real=False,
        start=True,
        model="model-a",
        prompt="prompt-a",
        command="/timezone UTC",
        callbacks=["regen:1", "confirm_regen:1"],
        message="hello",
    )

    await console_runner._dispatch_console_runtime(
        args,
        chat_id="chat-1",
        user_id="user-1",
        llm=MagicMock(),
        settings=MagicMock(),
    )

    assert messenger.dispatch_message.await_count == 2
    assert messenger.dispatch_callback.await_count == 5
    messenger.dispatch_text.assert_awaited_once_with(
        chat_id="chat-1",
        user_id="user-1",
        text="hello",
    )
    messenger.flush_edits.assert_called_once_with()

    with pytest.raises(SystemExit, match="nothing to do"):
        await console_runner._dispatch_console_runtime(
            SimpleNamespace(
                stream_debug=False,
                real=False,
                start=False,
                model=None,
                prompt=None,
                command=None,
                callbacks=[],
                message=None,
            ),
            chat_id="chat-1",
            user_id="user-1",
            llm=MagicMock(),
            settings=MagicMock(),
        )


def test_main_parses_args_and_runs_asyncio(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = MagicMock(parse_args=MagicMock(return_value=SimpleNamespace()))
    basic_config = MagicMock()

    def fake_asyncio_run(coro: object) -> None:
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(console_runner, "build_parser", lambda: parser)
    monkeypatch.setattr(console_runner.logging, "basicConfig", basic_config)
    monkeypatch.setattr(console_runner.asyncio, "run", fake_asyncio_run)

    console_runner.main(["hello"])

    parser.parse_args.assert_called_once_with(["hello"])
    basic_config.assert_called_once()
