"""Telegram history actions with real SQLite, update parsing, and offline transports."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from telegram import Bot, InlineKeyboardMarkup, Update
from telegram.error import BadRequest

from mai_gram.bot.handler import BotHandler
from mai_gram.db.database import close_db, get_session, init_db
from mai_gram.db.migrations import run_migrations
from mai_gram.db.models import Chat, Message
from mai_gram.llm.provider import LLMProviderError, StreamChunk, ToolCall
from mai_gram.messenger.telegram import TelegramMessenger
from mai_gram.messenger.telegram_support import convert_update_to_message

pytestmark = pytest.mark.integration

CHAT_ID = "123@audit"


class TelegramFlow:
    def __init__(self, tmp_path):
        self.db_path = tmp_path / "history.db"
        self.bot = Bot("123:offline")
        self.api = AsyncMock()
        self.ids = count(1000)
        self.api.send_message.side_effect = lambda **_: SimpleNamespace(message_id=next(self.ids))
        messenger = TelegramMessenger("123:offline")
        messenger._app = SimpleNamespace(bot=self.api)
        self.fail_generation = False
        self.prompts = []
        llm = MagicMock(count_tokens=AsyncMock(return_value=0), generate_stream=self.generate)
        self.handler = BotHandler(messenger, llm, test_mode=True, memory_data_dir=str(tmp_path))

    async def generate(self, messages, **kwargs):
        self.prompts.append(messages)
        if self.fail_generation:
            raise LLMProviderError("Offline test failure")
        yield StreamChunk(content="Regenerated answer.")

    def message(self, message_id, text, *, sent_at=None, edited_at=None, buttons=()):
        payload = {
            "message_id": message_id,
            "date": int((sent_at or datetime.now(timezone.utc)).timestamp()),
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 999, "is_bot": True, "first_name": "audit"},
            "text": text,
        }
        if edited_at is not None:
            payload["edit_date"] = int(edited_at.timestamp())
        if buttons:
            payload["reply_markup"] = {
                "inline_keyboard": [[{"text": key, "callback_data": key} for key in buttons]]
            }
        return payload

    async def press(self, data, source):
        event_id = next(self.ids)
        update = Update.de_json(
            {
                "update_id": event_id,
                "callback_query": {
                    "id": str(event_id),
                    "from": {"id": 123, "is_bot": False, "first_name": "QA"},
                    "chat_instance": "1",
                    "data": data,
                    "message": source,
                },
            },
            self.bot,
        )
        incoming = convert_update_to_message(update, bot_id="audit")
        assert incoming is not None
        await self.handler._handle_callback(incoming)

    async def ask(self, text):
        payload = self.message(next(self.ids), text)
        payload["from"] = {"id": 123, "is_bot": False, "first_name": "QA"}
        update = Update.de_json({"update_id": next(self.ids), "message": payload}, self.bot)
        incoming = convert_update_to_message(update, bot_id="audit")
        assert incoming is not None
        await self.handler._handle_message(incoming)

    def last_dialog(self, message_id):
        sent = self.api.send_message.await_args.kwargs
        markup = sent["reply_markup"]
        assert isinstance(markup, InlineKeyboardMarkup), sent["text"]
        buttons = [button.callback_data for button in markup.inline_keyboard[0]]
        assert all(1 <= len(data.encode()) <= 64 for data in buttons)
        return buttons[0], self.message(message_id, sent["text"], buttons=buttons)


@pytest.fixture
async def flow(tmp_path):
    engine = await init_db(f"sqlite+aiosqlite:///{tmp_path / 'history.db'}")
    await run_migrations(engine)
    try:
        async with get_session() as session:
            session.add(
                Chat(
                    id=CHAT_ID,
                    user_id="123",
                    bot_id="audit",
                    llm_model="test/model",
                    system_prompt="Offline test assistant.",
                )
            )
        yield TelegramFlow(tmp_path)
    finally:
        await close_db()


async def add_row(role, content, **kwargs):
    async with get_session() as session:
        row = Message(chat_id=CHAT_ID, role=role, content=content, **kwargs)
        session.add(row)
        await session.flush()
    return row


async def rows():
    async with get_session() as session:
        return list(await session.scalars(select(Message).order_by(Message.id)))


async def boundary():
    async with get_session() as session:
        return await session.scalar(select(Chat.cut_above_message_id).where(Chat.id == CHAT_ID))


@pytest.mark.parametrize("has_edit_date", [True, False])
async def test_legacy_streamed_cut_is_bound_and_duplicate_cleanup_cannot_move_boundary(
    flow, has_edit_date
):
    await add_row("user", "Where should I walk?")
    first = await add_row("assistant", "**Visit** the old harbour.")
    await add_row("user", "What about a cafe?")
    second = await add_row("assistant", "Try the bakery.")
    source_date = first.timestamp.replace(tzinfo=timezone.utc)
    original_id = 2**63 - 1
    old = f"cut:{first.id}"
    await flow.press(
        old,
        flow.message(
            original_id,
            "Visit the old harbour.\n\n12/24 tokens",
            sent_at=source_date - timedelta(seconds=30),
            edited_at=source_date + timedelta(seconds=1) if has_edit_date else None,
            buttons=[old],
        ),
    )
    yes, first_dialog = flow.last_dialog(401)
    assert len(yes.encode()) == 64
    assert yes == f"confirm_cut:{first.history_ref}:{original_id}"
    assert "Where should I walk?" in first_dialog["text"]
    assert "Visit" in first_dialog["text"]
    assert await boundary() is None

    async def assert_committed(**kwargs):
        if kwargs["message_id"] == 401 and flow.api.delete_message.await_count == 1:
            assert await boundary() == first.id
            raise BadRequest("temporary delete failure")

    flow.api.delete_message.side_effect = assert_committed
    await flow.press(yes, first_dialog)
    assert await boundary() == first.id
    assert flow.api.edit_message_text.await_args.kwargs["message_id"] == original_id
    assert "History cut applied" in flow.api.send_message.await_args.kwargs["text"]

    later = f"cut:{second.history_ref}"
    await flow.press(later, flow.message(302, second.content, buttons=[later]))
    later_yes, later_dialog = flow.last_dialog(402)
    await flow.press(later_yes, later_dialog)
    assert await boundary() == second.id
    edits = flow.api.edit_message_text.await_count

    await flow.press(yes, first_dialog)
    assert await boundary() == second.id
    assert flow.api.delete_message.await_args.kwargs["message_id"] == 401
    assert flow.api.edit_message_text.await_count == edits


@pytest.mark.parametrize("mode", ["before_preview", "after_preview", "rendered_excerpt"])
async def test_legacy_numeric_cut_checks_evidence_and_restore_alias(flow, mode):
    await add_row("user", "Which road?")
    answer = await add_row("assistant", "The harbour road.")
    old = f"cut:{answer.id}"

    async def replace_answer():
        async with get_session() as session:
            await session.delete(await session.get(Message, answer.id))
            await session.flush()
            session.add(
                Message(id=answer.id, chat_id=CHAT_ID, role="assistant", content="The forest road.")
            )

    if mode == "before_preview":
        await replace_answer()
    visible = (
        "...excerpt from a rendered response" if mode == "rendered_excerpt" else answer.content
    )
    await flow.press(old, flow.message(301, visible, buttons=[old]))
    if mode == "before_preview":
        assert "no longer available" in flow.api.send_message.await_args.kwargs["text"]
    else:
        yes, dialog = flow.last_dialog(401)
        assert answer.content in dialog["text"]
        assert "Which road?" in dialog["text"]
        if mode == "after_preview":
            await replace_answer()
        await flow.press(yes, dialog)

    assert await boundary() == (answer.id if mode == "rendered_excerpt" else None)
    flow.api.edit_message_text.assert_not_awaited()


@pytest.mark.parametrize("form", ["numeric", "sibling", "old_yes"])
async def test_legacy_regenerate_previews_original_turn_and_rolls_back_once(flow, form):
    question = await add_row("user", "First historical question.")
    answer = await add_row("assistant", "First historical response.")
    await add_row("user", "Unrelated later question.")
    await add_row("assistant", "Unrelated later answer.")
    if form == "sibling":
        action, buttons = "regen", ["regen", f"cut:{answer.id}"]
    elif form == "old_yes":
        action = f"confirm_regen:{answer.id}"
        buttons = [action]
    else:
        action = f"regen:{answer.id}"
        buttons = [action]
    visible = "Regenerate this response?" if form == "old_yes" else answer.content
    await flow.press(action, flow.message(301, visible, buttons=buttons))
    yes, dialog = flow.last_dialog(401)
    assert yes == f"confirm_regen:{answer.history_ref}"
    assert question.content in dialog["text"]
    assert answer.content in dialog["text"]
    assert len(await rows()) == 4
    assert len(flow.prompts) == 0

    await flow.press(yes, dialog)
    assert [row.content for row in await rows()] == [question.content, "Regenerated answer."]
    await flow.press(yes, dialog)
    assert len(flow.prompts) == 1


async def test_old_cut_yes_requires_new_preview(flow):
    await add_row("user", "The preceding question.")
    answer = await add_row("assistant", "The selected answer.")
    old = f"confirm_cut:{answer.id}:301"
    await flow.press(
        old, flow.message(401, f'Cut this message and all above?\n\nMessage: "{answer.content}"')
    )
    yes, dialog = flow.last_dialog(402)
    assert yes == f"confirm_cut:{answer.history_ref}"
    assert "The preceding question." in dialog["text"]
    assert await boundary() is None
    await flow.press(yes, dialog)
    assert await boundary() == answer.id


@pytest.mark.parametrize("action", ["cut", "regen", "retry"])
async def test_old_confirmation_survives_failed_replacement(flow, action):
    await add_row("user", "The pending question.")
    callback = "confirm_regen"
    if action != "retry":
        reply = await add_row("assistant", "The selected answer.")
        callback = f"confirm_{action}:{reply.id}"
    flow.api.send_message.side_effect = BadRequest("Message could not be delivered")
    await flow.press(callback, flow.message(401, "Old confirmation"))
    flow.api.delete_message.assert_not_awaited()
    assert await boundary() is None
    assert not flow.prompts


@pytest.mark.parametrize("form", ["dated", "uncertain", "old_yes"])
async def test_old_error_retry_selects_and_binds_pending_turn(flow, form):
    failed = await add_row("user", "Pending question.")
    failed_at = failed.timestamp.replace(tzinfo=timezone.utc)
    sent_at = failed_at + timedelta(seconds=4 if form == "dated" else -4)
    action = "confirm_regen" if form == "old_yes" else "regen"
    await flow.press(action, flow.message(301, "Error or old Retry dialog.", sent_at=sent_at))
    yes, dialog = flow.last_dialog(401)
    assert yes == f"confirm_retry:{failed.history_ref}"
    assert failed.content in dialog["text"]
    if form == "dated":
        assert "Retry this pending turn?" in dialog["text"]
    else:
        assert "Select the current pending turn" in dialog["text"]
    assert len(flow.prompts) == 0

    if form == "uncertain":
        await add_row("user", "Another turn arrived after the preview.")
    await flow.press(yes, dialog)
    assert len(flow.prompts) == (0 if form == "uncertain" else 1)
    if form != "uncertain":
        await flow.press(yes, dialog)
        assert len(flow.prompts) == 1


async def test_new_error_and_resend_keyboards_use_persisted_tokens(flow):
    flow.fail_generation = True
    await flow.ask("First turn fails.")
    first = (await rows())[-1]
    old_retry, error = flow.last_dialog(301)
    assert old_retry == f"retry:{first.history_ref}"

    await flow.ask("Second turn fails.")
    latest = (await rows())[-1]
    new_retry, latest_error = flow.last_dialog(302)
    assert new_retry == f"retry:{latest.history_ref}"
    await flow.press(old_retry, error)
    assert "no longer available" in flow.api.send_message.await_args.kwargs["text"]

    await flow.press(new_retry, latest_error)
    yes, dialog = flow.last_dialog(401)
    assert yes == f"confirm_retry:{latest.history_ref}"
    flow.fail_generation = False
    await flow.press(yes, dialog)
    response = (await rows())[-1]
    keyboard = [
        call.kwargs["reply_markup"]
        for call in (
            *flow.api.send_message.await_args_list,
            *flow.api.edit_message_text.await_args_list,
        )
        if isinstance(call.kwargs["reply_markup"], InlineKeyboardMarkup)
    ]
    expected = [f"regen:{response.history_ref}", f"cut:{response.history_ref}"]
    assert [button.callback_data for button in keyboard[-1].inline_keyboard[0]] == expected

    payload = flow.message(501, "/resend_last")
    payload["from"] = {"id": 123, "is_bot": False, "first_name": "QA"}
    update = Update.de_json({"update_id": 502, "message": payload}, flow.bot)
    incoming = convert_update_to_message(update, bot_id="audit")
    assert incoming is not None
    await flow.handler._handle_resend_last(incoming)
    resent = flow.api.send_message.await_args_list[-2].kwargs["reply_markup"]
    assert [button.callback_data for button in resent.inline_keyboard[0]] == expected


async def test_new_error_after_tool_call_uses_tool_anchor(flow):
    async def after_tool(*args, on_tool_call_display, on_tool_result_display, **kwargs):
        await on_tool_call_display(
            content="",
            tool_calls=[ToolCall(id="tool-1", name="lookup", arguments="{}")],
        )
        await on_tool_result_display(
            tool_call_id="tool-1",
            tool_name="lookup",
            arguments="{}",
            result=None,
            content="Tool response",
            error=None,
            server_name=None,
        )
        raise LLMProviderError("Offline test failure after tools")

    flow.handler._conversation_executor._stream_response = after_tool
    await flow.ask("Question using a tool.")
    user, assistant_call, tool = await rows()
    assert (user.role, assistant_call.role, tool.role) == ("user", "assistant", "tool")
    action, error = flow.last_dialog(302)
    assert action == f"retry:{tool.history_ref}"
    await flow.press(action, error)
    _, dialog = flow.last_dialog(401)
    assert user.content in dialog["text"]
    assert tool.content in dialog["text"]


@pytest.mark.parametrize("action", ["cut", "regen"])
async def test_bound_confirmations_survive_restart_but_reject_a_restored_id_alias(
    flow, tmp_path, action
):
    await add_row("user", "Question still present in the backup.")
    backup = tmp_path / "snapshot.db"
    with (
        closing(sqlite3.connect(flow.db_path)) as live,
        closing(sqlite3.connect(backup)) as snapshot,
    ):
        live.backup(snapshot)

    answer = await add_row("assistant", "Answer with identical text.")
    old = f"{action}:{answer.id}"
    await flow.press(old, flow.message(301, answer.content, buttons=[old]))
    yes, dialog = flow.last_dialog(401)
    await close_db()
    with (
        closing(sqlite3.connect(flow.db_path)) as live,
        closing(sqlite3.connect(backup)) as snapshot,
    ):
        snapshot.backup(live)
    engine = await init_db(f"sqlite+aiosqlite:///{flow.db_path}")
    await run_migrations(engine)

    replacement = await add_row("assistant", answer.content)
    assert replacement.id == answer.id
    assert replacement.history_ref != answer.history_ref
    restarted = TelegramFlow(tmp_path)
    await restarted.press(yes, dialog)
    assert await boundary() is None
    assert (await rows())[-1].history_ref == replacement.history_ref
    assert len(restarted.prompts) == 0
    assert "no longer available" in restarted.api.send_message.await_args.kwargs["text"]


async def test_targeted_regen_removes_preview_after_stream_failure(flow, monkeypatch):
    await add_row("user", "Original question.")
    original = await add_row("assistant", "Original answer.")
    action = f"regen:{original.history_ref}"
    await flow.press(action, flow.message(301, original.content, buttons=[action]))
    yes, dialog = flow.last_dialog(401)

    async def generate(*args, **kwargs):
        yield StreamChunk(content="Replacement preview.")
        raise LLMProviderError("Offline test failure after preview")

    monkeypatch.setattr("mai_gram.bot.conversation_executor.run_with_tools_stream", generate)
    preview_id = next(flow.ids)

    async def send(**kwargs):
        if kwargs["text"].endswith("▍"):
            return SimpleNamespace(message_id=preview_id)
        return SimpleNamespace(message_id=next(flow.ids))

    flow.api.send_message.side_effect = send
    await flow.press(yes, dialog)

    assert (await rows())[-1].history_ref == original.history_ref
    deleted = [call.kwargs["message_id"] for call in flow.api.delete_message.await_args_list]
    assert [message_id for message_id in deleted if message_id != 401] == [preview_id]
    retry, _ = flow.last_dialog(402)
    assert retry == action


@pytest.mark.parametrize("partial", [False, True], ids=["total_failure", "partial_delivery"])
@pytest.mark.parametrize("retry", [False, True], ids=["regen", "retry"])
async def test_bound_turn_preserves_history_until_full_delivery(flow, monkeypatch, partial, retry):
    question = await add_row("user", "Original question.")
    if retry:
        action = f"retry:{question.history_ref}"
        source = flow.message(301, "Generation failed.", buttons=[action])
    else:
        original = await add_row("assistant", "Original answer.")
        await add_row("user", "Later question.")
        await add_row("assistant", "Later answer.")
        action = f"regen:{original.history_ref}"
        source = flow.message(301, original.content, buttons=[action])
    before = [row.history_ref for row in await rows()]
    await flow.press(action, source)
    yes, dialog = flow.last_dialog(401)

    async def generate(*args, on_assistant_tool_call, on_tool_result, **kwargs):
        await on_assistant_tool_call(
            content="", tool_calls=[ToolCall(id="lookup-1", name="lookup", arguments="{}")]
        )
        await on_tool_result(
            tool_call_id="lookup-1",
            tool_name="lookup",
            arguments="{}",
            result="found",
            content="found",
            error=None,
            server_name=None,
        )
        yield StreamChunk(content="Replacement begins.")
        if partial:
            yield StreamChunk(content=" Middle." * 600 + " Replacement ends.")

    monkeypatch.setattr("mai_gram.bot.conversation_executor.run_with_tools_stream", generate)
    replacement_actions = []
    accepted_body = []
    attempt_ids = []
    reject_first_part = True

    async def send(**kwargs):
        text = kwargs["text"]
        markup = kwargs["reply_markup"]
        if "Replacement" in text and not text.endswith("▍"):
            assert [row.history_ref for row in await rows()] == before
            if isinstance(markup, InlineKeyboardMarkup):
                replacement_actions.append(markup.inline_keyboard[0][0].callback_data)
            if reject_first_part and "Replacement begins." in text:
                raise BadRequest("Final answer send rejected")
            accepted_body.append(text)
        message_id = next(flow.ids)
        if "Could not deliver" not in text:
            attempt_ids.append(message_id)
        return SimpleNamespace(message_id=message_id)

    async def edit(**kwargs):
        if "Replacement" in kwargs["text"]:
            raise BadRequest("Final answer edit rejected")

    flow.api.send_message.side_effect = send
    flow.api.edit_message_text.side_effect = edit
    await flow.press(yes, dialog)

    assert [row.history_ref for row in await rows()] == before
    assert len(replacement_actions) == 1
    assert any("Replacement ends." in text for text in accepted_body) is partial
    assert any("🔧 lookup" in call.kwargs["text"] for call in flow.api.send_message.await_args_list)
    assert "Could not deliver" in flow.api.send_message.await_args.kwargs["text"]
    deleted = [call.kwargs["message_id"] for call in flow.api.delete_message.await_args_list]
    assert attempt_ids
    assert sorted(message_id for message_id in deleted if message_id != 401) == sorted(attempt_ids)

    if retry:
        new_action, notice = flow.last_dialog(502)
        assert new_action == action
        await flow.press(new_action, notice)
        yes, dialog = flow.last_dialog(503)
    reject_first_part = False
    await flow.press(yes, dialog)
    history = await rows()
    assert [row.role for row in history] == ["user", "assistant", "tool", "assistant"]
    assert history[0].id == question.id
    assert history[-1].content.startswith("Replacement begins.")
    assert replacement_actions[1] == f"regen:{history[-1].history_ref}"
    assert replacement_actions[0] != replacement_actions[1]

    rolled_back_action = replacement_actions[0]
    await flow.press(
        rolled_back_action,
        flow.message(501, history[-1].content[-100:], buttons=[rolled_back_action]),
    )
    assert "no longer available" in flow.api.send_message.await_args.kwargs["text"]


async def test_retry_confirmation_replay_cannot_redrive_a_failed_attempt(flow):
    flow.fail_generation = True
    await flow.ask("Retry this question.")
    retry, error = flow.last_dialog(301)
    await flow.press(retry, error)
    yes, dialog = flow.last_dialog(401)

    await flow.press(yes, dialog)
    assert len(flow.prompts) == 2
    await flow.press(yes, dialog)
    assert len(flow.prompts) == 2
    assert "no longer available" in flow.api.send_message.await_args.kwargs["text"]
