"""Callback dispatcher for bot button interactions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.bot.history_actions import HistoryActions
    from mai_gram.bot.import_workflow import ImportWorkflow
    from mai_gram.bot.regenerate_service import RegenerateService
    from mai_gram.bot.reset_workflow import ResetWorkflow
    from mai_gram.bot.setup_workflow import SetupWorkflow
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


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
        show_confirmation: Callable[..., Awaitable[None]],
        delete_callback_message: Callable[[IncomingMessage], Awaitable[None]],
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
        self._delete_callback_message = delete_callback_message
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
        if await self._handle_stale_setup_callback(message, data):
            return

        if data == "regen":
            await self._show_confirmation(
                message,
                "Regenerate this response?",
                confirm_data="confirm_regen",
                cancel_data="cancel_action",
            )
            return

        if data.startswith("cut:"):
            await self._handle_cut_confirmation(message, data)
            return

        if await self._handle_confirmation_callback(message, data):
            return

        logger.debug("Unhandled callback: %s", data)

    async def _handle_stale_setup_callback(self, message: IncomingMessage, data: str) -> bool:
        if not (data.startswith("model:") or data.startswith("prompt:")):
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
        if data == "confirm_regen":
            await self._confirm_regenerate(message)
            return True
        if data.startswith("confirm_cut:"):
            await self._confirm_cut(message, data)
            return True
        if data.startswith("confirm_reset:"):
            await self._confirm_reset(message, data)
            return True
        if data == "cancel_action":
            await self._delete_callback_message(message)
            return True
        return False

    async def _confirm_regenerate(self, message: IncomingMessage) -> None:
        await self._delete_callback_message(message)
        sent_ids = await self._regenerate_service.handle_regenerate(
            message,
            previous_response_ids=self._response_message_ids.get(message.chat_id, []),
        )
        self._response_message_ids[message.chat_id] = sent_ids

    async def _confirm_cut(self, message: IncomingMessage, data: str) -> None:
        parts = data.split(":", 2)
        cut_msg_id_str = parts[1]
        original_tg_msg_id = parts[2] if len(parts) > 2 else ""
        await self._delete_callback_message(message)
        cached_original = None
        if original_tg_msg_id:
            cache_key = f"{message.chat_id}:{original_tg_msg_id}"
            cached_original = self._cut_original_html.pop(cache_key, None)
        await self._history_actions.handle_cut_above(
            message,
            int(cut_msg_id_str),
            original_tg_msg_id=original_tg_msg_id,
            cached_original=cached_original,
        )

    async def _confirm_reset(self, message: IncomingMessage, data: str) -> None:
        chat_id = data.split(":", 1)[1]
        await self._delete_callback_message(message)
        await self._reset_workflow.execute_reset(message, chat_id)

    async def _handle_cut_confirmation(self, message: IncomingMessage, data: str) -> None:
        cut_msg_id = data.split(":", 1)[1]
        preview = await self._history_actions.get_message_preview(int(cut_msg_id))
        confirm_text = (
            "Cut this message and all above?\nThey won't be sent to AI but remain searchable."
        )
        if preview:
            confirm_text += f'\n\nMessage: "{preview}"'

        tg_msg_id = ""
        if message.raw and hasattr(message.raw, "callback_query"):
            callback_message = message.raw.callback_query.message
            if callback_message:
                tg_msg_id = str(callback_message.message_id)
                original_html = getattr(callback_message, "text_html", None)
                original_parse = "html" if original_html else None
                if original_html is None:
                    original_html = callback_message.text or ""
                cache_key = f"{message.chat_id}:{tg_msg_id}"
                self._cut_original_html[cache_key] = (original_html, original_parse)

        await self._show_confirmation(
            message,
            confirm_text,
            confirm_data=f"confirm_cut:{cut_msg_id}:{tg_msg_id}",
            cancel_data="cancel_action",
        )
