"""Callback dispatcher for bot button interactions."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from mai_gram.bot.history_actions import (
    HISTORY_ACTION_UNAVAILABLE,
    legacy_source_matches,
    source_time,
)
from mai_gram.bot.history_buttons import parse_history_target, parse_message_id
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.bot.history_actions import HistoryActions
    from mai_gram.bot.import_workflow import ImportWorkflow
    from mai_gram.bot.regenerate_service import RegenerateService
    from mai_gram.bot.reset_workflow import ResetWorkflow
    from mai_gram.bot.setup_workflow import SetupWorkflow
    from mai_gram.messenger.base import CallbackSourceMessage, Messenger

logger = logging.getLogger(__name__)


def _extract_sibling_msg_id(message: IncomingMessage, *, prefix: str) -> str | None:
    """Extract a DB message ID from a sibling button on the same inline keyboard.

    When a bare ``"regen"`` callback arrives from a pre-fix button, the
    sibling ``"cut:<id>"`` button on the same message carries the target
    DB message ID.  This lets old buttons work without needing the new
    ``regen:<id>`` format.
    """
    raw = getattr(message, "raw", None)
    callback_query = getattr(raw, "callback_query", None)
    source_msg = getattr(callback_query, "message", None)
    markup = getattr(source_msg, "reply_markup", None)
    keyboard = getattr(markup, "inline_keyboard", None)
    if not keyboard:
        return None
    for row in keyboard:
        for button in row:
            cb = getattr(button, "callback_data", None) or ""
            if isinstance(cb, str) and cb.startswith(prefix):
                return cb.split(":", 1)[1]
    return None


class CallbackRouter:
    """Route callback data to the appropriate extracted workflow."""

    def __init__(
        self,
        messenger: Messenger,
        *,
        import_workflow: ImportWorkflow,
        setup_workflow: SetupWorkflow,
        reset_workflow: ResetWorkflow,
        history_actions: HistoryActions,
        regenerate_service: RegenerateService,
        show_confirmation: Callable[..., Awaitable[bool]],
        cut_original_html: dict[str, tuple[str, str | None]],
        response_message_ids: dict[str, list[str]],
    ) -> None:
        self._messenger = messenger
        self._import_workflow = import_workflow
        self._setup_workflow = setup_workflow
        self._reset_workflow = reset_workflow
        self._history_actions = history_actions
        self._regenerate_service = regenerate_service
        self._show_confirmation = show_confirmation
        self._cut_original_html = cut_original_html
        self._response_message_ids = response_message_ids

    async def handle_callback(self, message: IncomingMessage) -> None:
        """Dispatch a callback to the right workflow or action."""
        if self._import_workflow.is_in_import(message.user_id):
            await self._import_workflow.handle_import_callback(message)
            return

        if self._setup_workflow.is_in_setup(message.user_id):
            await self._setup_workflow.handle_setup_callback(message)
            return

        data = message.callback_data or ""

        if data.startswith("setmodel:"):
            await self._setup_workflow.handle_model_change(message, data.split(":", 1)[1])
            return

        if await self._handle_stale_setup_callback(message, data):
            return

        if data == "regen" or data.startswith("regen:"):
            await self._handle_regen_confirmation(message, data)
            return

        if data.startswith("retry:"):
            token = parse_history_target(data.partition(":")[2])
            if isinstance(token, str):
                await self._show_retry_confirmation(message, token=token)
            else:
                await self._history_action_unavailable(message)
            return

        if data.startswith("cut:"):
            await self._handle_cut_confirmation(message, data)
            return

        if await self._handle_confirmation_callback(message, data):
            return

        logger.debug("Unhandled callback: %s", data)

    async def _handle_stale_setup_callback(self, message: IncomingMessage, data: str) -> bool:
        setup_prefixes = ("model:", "prompt:", "template:")
        if not any(data.startswith(p) for p in setup_prefixes):
            return False
        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    f"Setup callback '{data}' ignored — no setup session active.\n"
                    "Hint: use --start --model MODEL --prompt PROMPT in a single command."
                ),
                chat_id=message.chat_id,
            )
        )
        return True

    async def _handle_confirmation_callback(self, message: IncomingMessage, data: str) -> bool:
        if data == "confirm_regen" or data.startswith(("confirm_regen:", "confirm_retry:")):
            await self._confirm_regenerate(message, data)
            return True
        if data.startswith("confirm_cut:"):
            await self._confirm_cut(message, data)
            return True
        if data.startswith("confirm_reset:"):
            await self._confirm_reset(message, data)
            return True
        if data == "cancel_action":
            await self._messenger.delete_callback_source_message(message)
            return True
        return False

    async def _confirm_regenerate(self, message: IncomingMessage, data: str) -> None:
        action, _, suffix = data.partition(":")
        target = parse_history_target(suffix) if action != data else None
        if action == "confirm_retry":
            if not isinstance(target, str):
                await self._history_action_unavailable(message)
                return
        elif action != data and target is None:
            await self._history_action_unavailable(message)
            return
        elif message.platform == "telegram" and not isinstance(target, str):
            if target is None:
                await self._show_retry_confirmation(message, replace_old=True)
            else:
                await self._show_target_confirmation(message, "regen", target, replace_old=True)
            return

        await self._messenger.delete_callback_source_message(message)
        sent_ids = await self._regenerate_service.handle_regenerate(
            message,
            previous_response_ids=self._response_message_ids.get(message.chat_id, []),
            target_assistant_msg_id=target if isinstance(target, int) else None,
            target_ref=target if action == "confirm_regen" and isinstance(target, str) else None,
            retry_ref=target if action == "confirm_retry" and isinstance(target, str) else None,
        )
        if sent_ids is not None:
            self._response_message_ids[message.chat_id] = sent_ids

    async def _confirm_cut(self, message: IncomingMessage, data: str) -> None:
        suffix, _, source_id = data.partition(":")[2].partition(":")
        target = parse_history_target(suffix)
        is_telegram = message.platform == "telegram"
        if target is None or (is_telegram and source_id and parse_message_id(source_id) is None):
            await self._history_action_unavailable(message)
            return
        if is_telegram and isinstance(target, int):
            await self._show_target_confirmation(message, "cut", target, replace_old=True)
            return

        confirmation_id = ""
        if is_telegram:
            source = self._messenger.get_callback_source_message(message)
            if source is None or parse_message_id(source.message_id) is None:
                await self._history_action_unavailable(message)
                return
            confirmation_id = source.message_id

        cache_key = f"{message.chat_id}:{source_id}:{target}"
        cached_original = self._cut_original_html.get(cache_key)
        await self._history_actions.handle_cut_above(
            message,
            target,
            original_tg_msg_id=source_id,
            cached_original=cached_original,
            confirmation_message_id=confirmation_id,
        )
        self._cut_original_html.pop(cache_key, None)

    async def _confirm_reset(self, message: IncomingMessage, data: str) -> None:
        chat_id = data.split(":", 1)[1]
        await self._messenger.delete_callback_source_message(message)
        await self._reset_workflow.execute_reset(message, chat_id)

    async def _history_action_unavailable(self, message: IncomingMessage) -> None:
        await self._messenger.send_message(
            OutgoingMessage(text=HISTORY_ACTION_UNAVAILABLE, chat_id=message.chat_id)
        )

    async def _handle_regen_confirmation(self, message: IncomingMessage, data: str) -> None:
        suffix = (
            _extract_sibling_msg_id(message, prefix="cut:")
            if data == "regen"
            else data.partition(":")[2]
        )
        if suffix is not None:
            await self._show_target_confirmation(message, "regen", parse_history_target(suffix))
        elif message.platform == "telegram":
            await self._show_retry_confirmation(message)
        else:
            await self._show_confirmation(
                message,
                "Regenerate this response?",
                confirm_data="confirm_regen",
                cancel_data="cancel_action",
            )

    async def _handle_cut_confirmation(self, message: IncomingMessage, data: str) -> None:
        await self._show_target_confirmation(
            message, "cut", parse_history_target(data.partition(":")[2])
        )

    async def _show_target_confirmation(
        self,
        message: IncomingMessage,
        action: str,
        target: int | str | None,
        *,
        replace_old: bool = False,
    ) -> None:
        preview = (
            await self._history_actions.get_message_preview(message, target)
            if target is not None
            else None
        )
        if preview is None:
            await self._history_action_unavailable(message)
            return

        source = self._messenger.get_callback_source_message(message)
        evidence = _old_confirmation_preview(source) if replace_old else source
        match = None
        if message.platform == "telegram" and isinstance(target, int) and evidence is not None:
            match = legacy_source_matches(evidence, preview)
            if match is False:
                await self._history_action_unavailable(message)
                return

        if action == "cut":
            question = (
                "Cut this message and all above?\nThey won't be sent to AI but remain searchable."
            )
        else:
            question = "Regenerate this response?"
        if message.platform == "telegram" and isinstance(target, int):
            question += "\nCheck the stored turn below before confirming this older button."
        confirm_data = f"confirm_{action}:{preview.token}"
        if (
            action == "cut"
            and not replace_old
            and source is not None
            and source.text
            and parse_message_id(source.message_id) is not None
            and (isinstance(target, str) or match is True)
        ):
            cache_key = f"{message.chat_id}:{source.message_id}:{preview.token}"
            self._cut_original_html[cache_key] = (source.text, source.parse_mode)
            confirm_data += f":{source.message_id}"
        delivered = await self._show_confirmation(
            message,
            f"{question}\n\n{preview.describe()}",
            confirm_data=confirm_data,
            cancel_data="cancel_action",
        )
        if replace_old and delivered:
            await self._messenger.delete_callback_source_message(message)

    async def _show_retry_confirmation(
        self,
        message: IncomingMessage,
        *,
        token: str | None = None,
        replace_old: bool = False,
    ) -> None:
        preview = await self._history_actions.get_retry_preview(message)
        if preview is None or (token is not None and token != preview.token):
            await self._history_action_unavailable(message)
            return

        sent_at = None
        if token is None and not replace_old:
            source = self._messenger.get_callback_source_message(message)
            sent_at = source_time(source.sent_at) if source is not None else None
        identified = token is not None or (sent_at is not None and sent_at > preview.timestamp_utc)
        question = (
            "Retry this pending turn?"
            if identified
            else "Select the current pending turn shown below to retry."
        )
        delivered = await self._show_confirmation(
            message,
            f"{question}\n\n{preview.describe()}",
            confirm_data=f"confirm_retry:{preview.token}",
            cancel_data="cancel_action",
        )
        if replace_old and delivered:
            await self._messenger.delete_callback_source_message(message)


def _old_confirmation_preview(source: CallbackSourceMessage | None) -> CallbackSourceMessage | None:
    if source is None:
        return None
    dialog = source.plain_text if source.plain_text is not None else source.text
    _, marker, preview = dialog.partition('\n\nMessage: "')
    excerpt = preview.removesuffix('"').removesuffix("...") if marker else ""
    return replace(source, plain_text=excerpt, edited_at=source.edited_at or source.sent_at)
