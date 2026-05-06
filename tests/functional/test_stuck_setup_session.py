"""Regression tests for the stuck setup/import session deadlock.

Previously, if /start or /import was initiated but never completed, the
in-memory session lingered.  /reset said "No chat to reset." without clearing
it, leaving the user permanently stuck.

Fix: handle_reset detects pending setup/import sessions, clears them, and
responds with a context-specific message instead of the generic "no chat" text.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest

from mai_gram.messenger.base import IncomingMessage, MessageType

pytestmark = pytest.mark.functional

CHAT_ID = "test-stuck-setup"
USER_ID = "test-user-stuck"


def _command_message(command: str) -> IncomingMessage:
    return IncomingMessage(
        platform="console",
        chat_id=CHAT_ID,
        user_id=USER_ID,
        message_id=f"cmd-{command}",
        message_type=MessageType.COMMAND,
        text=f"/{command}",
        command=command,
        timestamp=datetime.now(timezone.utc),
    )


def _callback_message(data: str) -> IncomingMessage:
    return IncomingMessage(
        platform="console",
        chat_id=CHAT_ID,
        user_id=USER_ID,
        message_id=f"cb-{data}",
        message_type=MessageType.CALLBACK,
        callback_data=data,
        timestamp=datetime.now(timezone.utc),
    )


async def _build_handler(tmp_path):
    """Build a BotHandler with a live DB and captured console output."""
    import mai_gram.config as _cfg
    from mai_gram.bot.handler import BotHandler
    from mai_gram.config import Settings, reset_settings
    from mai_gram.db import init_db, reset_db_state, run_migrations
    from mai_gram.llm.provider import LLMProvider, LLMResponse, StreamChunk
    from mai_gram.messenger.console import ConsoleMessenger

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    reset_settings()
    reset_db_state()

    settings = Settings(
        telegram_bot_token="fake-token",  # noqa: S106
        openrouter_api_key="fake-key",
        database_url=db_url,
        memory_data_dir=str(tmp_path / "data"),
        allowed_users="",
        log_level="DEBUG",
    )
    _cfg._settings_instance = settings

    engine = await init_db(db_url, echo=False)
    await run_migrations(engine)

    output_buf = io.StringIO()
    messenger = ConsoleMessenger(output=output_buf)

    class _StubLLM(LLMProvider):
        async def generate(self, messages, **kw):  # type: ignore[override]
            return LLMResponse(content="stub", model="stub")

        async def generate_stream(self, messages, **kw):  # type: ignore[override]
            yield StreamChunk(content="stub")

        async def count_tokens(self, messages, **kw):  # type: ignore[override]
            return 0

        async def close(self):  # type: ignore[override]
            pass

    handler = BotHandler(
        messenger,
        _StubLLM(),
        memory_data_dir=settings.memory_data_dir,
        test_mode=True,
    )
    return handler, messenger, output_buf


async def _teardown():
    from mai_gram.config import reset_settings
    from mai_gram.db import close_db, reset_db_state

    await close_db()
    reset_db_state()
    reset_settings()


def _flush_and_read(buf: io.StringIO) -> str:
    buf.truncate(0)
    buf.seek(0)
    return ""


def _read(buf: io.StringIO) -> str:
    return buf.getvalue()


async def test_reset_clears_abandoned_setup_and_unblocks_import(tmp_path) -> None:
    """Abandoned /start -> /reset -> /import should work.

    /reset must say "New chat setup has been reset." and clear the session.
    """
    handler, messenger, output_buf = await _build_handler(tmp_path)

    try:
        await messenger.dispatch_message(_command_message("start"))
        assert "Choose an LLM model" in _read(output_buf)
        assert handler.is_in_setup(USER_ID) is True

        _flush_and_read(output_buf)
        await messenger.dispatch_message(_command_message("reset"))
        reset_output = _read(output_buf)
        assert "New chat setup has been reset" in reset_output
        assert "/start" in reset_output
        assert handler.is_in_setup(USER_ID) is False

        _flush_and_read(output_buf)
        await messenger.dispatch_message(_command_message("import"))
        import_output = _read(output_buf)
        assert "middle of a setup" not in import_output
        assert "Import Mode" in import_output
    finally:
        await _teardown()


async def test_reset_clears_abandoned_import_session(tmp_path) -> None:
    """Abandoned /import -> /reset should say "Import procedure has been reset."."""
    _handler, messenger, output_buf = await _build_handler(tmp_path)

    try:
        await messenger.dispatch_message(_command_message("import"))
        assert "Import Mode" in _read(output_buf)

        _flush_and_read(output_buf)
        await messenger.dispatch_message(_command_message("reset"))
        reset_output = _read(output_buf)
        assert "Import procedure has been reset" in reset_output
        assert "/import" in reset_output

        _flush_and_read(output_buf)
        await messenger.dispatch_message(_command_message("start"))
        assert "Choose an LLM model" in _read(output_buf)
    finally:
        await _teardown()


async def test_reset_with_no_session_says_no_chat(tmp_path) -> None:
    """/reset with no pending session and no chat shows the generic message."""
    _, messenger, output_buf = await _build_handler(tmp_path)

    try:
        await messenger.dispatch_message(_command_message("reset"))
        reset_output = _read(output_buf)
        assert "No chat to reset" in reset_output
        assert "/start" in reset_output
    finally:
        await _teardown()
