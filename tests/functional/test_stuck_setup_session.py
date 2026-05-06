"""Regression test for the stuck setup-session deadlock.

Previously, if /start was called but setup was never completed (user abandons
the model/prompt selection flow), a SetupSession remained in memory.  Subsequent
/reset said "No chat to reset." (correct -- no chat was created) but did NOT
clear the stale setup session.  Then /import was permanently blocked by
"You are in the middle of a setup. Finish it first or use /reset."

Fix: handle_reset now clears any dangling setup session even when no chat exists.
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


async def test_import_after_abandoned_start_and_reset_is_not_stuck(tmp_path) -> None:
    """Verify /reset clears an orphaned setup session so /import can proceed.

    Sequence:
    1. /start (creates setup session, user abandons without completing)
    2. /reset (no chat exists -> clears stale setup session)
    3. /import (should proceed to model selection, not be blocked)
    """
    handler, messenger, output_buf = await _build_handler(tmp_path)

    try:
        # Step 1: User sends /start -- setup session is created, model selection shown.
        # User never picks a model (abandons the flow).
        await messenger.dispatch_message(_command_message("start"))
        start_output = output_buf.getvalue()
        assert "Choose an LLM model" in start_output
        assert handler.is_in_setup(USER_ID) is True

        # Step 2: User sends /reset -- no chat was created, but the stale session is cleared.
        output_buf.truncate(0)
        output_buf.seek(0)
        await messenger.dispatch_message(_command_message("reset"))
        reset_output = output_buf.getvalue()
        assert "No chat to reset" in reset_output
        assert handler.is_in_setup(USER_ID) is False

        # Step 3: User sends /import -- proceeds normally (no stale session blocking it).
        output_buf.truncate(0)
        output_buf.seek(0)
        await messenger.dispatch_message(_command_message("import"))
        import_output = output_buf.getvalue()

        assert "middle of a setup" not in import_output, (
            f"/import should not be blocked after /reset. Got: {import_output!r}"
        )
        assert "Import Mode" in import_output
    finally:
        await _teardown()
