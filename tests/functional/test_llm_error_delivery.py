"""Tests that LLM provider errors are delivered to the user as friendly messages.

Bug reproduced here: ConversationExecutor.execute() only catches
LLMProviderError, but LLMRateLimitError (and other siblings like
LLMAuthenticationError, LLMModelNotFoundError, LLMContextLengthError)
inherit from LLMError -- NOT from LLMProviderError.

When a 429 comes back, _raise_for_status raises LLMRateLimitError which
escapes the except clause in execute(), so the user never sees the
friendly "Rate limit reached" message.
"""

from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.functional

CHAT_ID = "test-error-delivery"
USER_ID = "test-user-errors"


async def _build_handler(tmp_path, llm_class):
    """Build a BotHandler with a live DB, captured console output, and a custom LLM stub."""
    import mai_gram.config as _cfg
    from mai_gram.bot.handler import BotHandler
    from mai_gram.config import Settings, reset_settings
    from mai_gram.db import init_db, reset_db_state, run_migrations
    from mai_gram.db.database import get_session
    from mai_gram.db.models import Chat
    from mai_gram.messenger.console import ConsoleMessenger

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.joinpath("default.txt").write_text("You are a test assistant.\n", encoding="utf-8")

    reset_settings()
    reset_db_state()

    settings = Settings(
        telegram_bot_token="fake-token",  # noqa: S106
        openrouter_api_key="fake-key",
        database_url=db_url,
        memory_data_dir=str(data_dir),
        allowed_users="",
        log_level="DEBUG",
    )
    _cfg._settings_instance = settings

    engine = await init_db(db_url, echo=False)
    await run_migrations(engine)

    async with get_session() as session:
        chat = Chat(
            id=CHAT_ID,
            user_id=USER_ID,
            bot_id="console",
            llm_model="test-model",
            system_prompt="You are a test assistant.",
            prompt_name="default",
        )
        session.add(chat)
        await session.commit()

    output_buf = io.StringIO()
    messenger = ConsoleMessenger(output=output_buf)

    handler = BotHandler(
        messenger,
        llm_class(),
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


def _read(buf: io.StringIO) -> str:
    return buf.getvalue()


async def _send_and_collect(messenger, output_buf, text="Hello"):
    """Send a message and return (output, escaped_exception).

    If the LLM error escapes the handler instead of being caught and
    rendered as a user-friendly message, we capture it here so the test
    can fail with a clear assertion rather than an opaque traceback.
    """
    escaped: BaseException | None = None
    try:
        await messenger.dispatch_text(
            chat_id=CHAT_ID,
            user_id=USER_ID,
            text=text,
        )
    except Exception as exc:
        escaped = exc
    return _read(output_buf), escaped


async def test_rate_limit_error_is_delivered_to_user(tmp_path) -> None:
    """A 429 / LLMRateLimitError must show a user-friendly rate-limit message.

    Currently FAILS because execute() does not catch LLMRateLimitError.
    """
    from mai_gram.llm.provider import LLMProvider, LLMRateLimitError

    class _RateLimitLLM(LLMProvider):
        async def generate(self, messages, **kw):  # type: ignore[override]
            raise LLMRateLimitError("Rate limited: HTTP 429", retry_after=5.0)

        async def generate_stream(self, messages, **kw):  # type: ignore[override]
            raise LLMRateLimitError("Rate limited: HTTP 429", retry_after=5.0)
            yield  # type: ignore[misc]  # make it a generator

        async def count_tokens(self, messages, **kw):  # type: ignore[override]
            return 0

        async def close(self):  # type: ignore[override]
            pass

    _handler, messenger, output_buf = await _build_handler(tmp_path, _RateLimitLLM)

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, (
            f"LLMRateLimitError escaped the handler instead of being delivered "
            f"to the user as a friendly message: {escaped}"
        )
        assert "rate limit" in output.lower(), (
            f"Expected a user-friendly rate-limit message in the output, but got:\n{output}"
        )
    finally:
        await _teardown()


async def test_auth_error_is_delivered_to_user(tmp_path) -> None:
    """A 401 / LLMAuthenticationError must show a user-friendly auth error message.

    Currently FAILS because execute() does not catch LLMAuthenticationError.
    """
    from mai_gram.llm.provider import LLMAuthenticationError, LLMProvider

    class _AuthErrorLLM(LLMProvider):
        async def generate(self, messages, **kw):  # type: ignore[override]
            raise LLMAuthenticationError("Authentication failed: invalid key")

        async def generate_stream(self, messages, **kw):  # type: ignore[override]
            raise LLMAuthenticationError("Authentication failed: invalid key")
            yield  # type: ignore[misc]

        async def count_tokens(self, messages, **kw):  # type: ignore[override]
            return 0

        async def close(self):  # type: ignore[override]
            pass

    _handler, messenger, output_buf = await _build_handler(tmp_path, _AuthErrorLLM)

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, (
            f"LLMAuthenticationError escaped the handler instead of being delivered "
            f"to the user as a friendly message: {escaped}"
        )
        assert "authentication" in output.lower(), (
            f"Expected a user-friendly auth error message in the output, but got:\n{output}"
        )
    finally:
        await _teardown()


async def test_model_not_found_error_is_delivered_to_user(tmp_path) -> None:
    """A 404 / LLMModelNotFoundError must show a user-friendly model error message.

    Currently FAILS because execute() does not catch LLMModelNotFoundError.
    """
    from mai_gram.llm.provider import LLMModelNotFoundError, LLMProvider

    class _ModelNotFoundLLM(LLMProvider):
        async def generate(self, messages, **kw):  # type: ignore[override]
            raise LLMModelNotFoundError("Model not found: test-model")

        async def generate_stream(self, messages, **kw):  # type: ignore[override]
            raise LLMModelNotFoundError("Model not found: test-model")
            yield  # type: ignore[misc]

        async def count_tokens(self, messages, **kw):  # type: ignore[override]
            return 0

        async def close(self):  # type: ignore[override]
            pass

    _handler, messenger, output_buf = await _build_handler(tmp_path, _ModelNotFoundLLM)

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, (
            f"LLMModelNotFoundError escaped the handler instead of being delivered "
            f"to the user as a friendly message: {escaped}"
        )
        assert "no longer available" in output.lower(), (
            f"Expected a user-friendly model-not-found message in the output, but got:\n{output}"
        )
    finally:
        await _teardown()


async def test_context_length_error_is_delivered_to_user(tmp_path) -> None:
    """LLMContextLengthError must show a user-friendly context-length message.

    Currently FAILS because execute() does not catch LLMContextLengthError.
    """
    from mai_gram.llm.provider import LLMContextLengthError, LLMProvider

    class _ContextLengthLLM(LLMProvider):
        async def generate(self, messages, **kw):  # type: ignore[override]
            raise LLMContextLengthError("Context length exceeded: too many tokens")

        async def generate_stream(self, messages, **kw):  # type: ignore[override]
            raise LLMContextLengthError("Context length exceeded: too many tokens")
            yield  # type: ignore[misc]

        async def count_tokens(self, messages, **kw):  # type: ignore[override]
            return 0

        async def close(self):  # type: ignore[override]
            pass

    _handler, messenger, output_buf = await _build_handler(tmp_path, _ContextLengthLLM)

    try:
        output, escaped = await _send_and_collect(messenger, output_buf)
        assert escaped is None, (
            f"LLMContextLengthError escaped the handler instead of being delivered "
            f"to the user as a friendly message: {escaped}"
        )
        assert "too long" in output.lower() or "context" in output.lower(), (
            f"Expected a user-friendly context-length message in the output, but got:\n{output}"
        )
    finally:
        await _teardown()
