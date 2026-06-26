"""Chat-record construction and confirmation messaging for setup completion.

Extracted from ``setup_workflow`` to keep that module within size limits and to
give the custom-model flow a single place that knows how a finished chat is
persisted and announced.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mai_gram.bot import custom_model
from mai_gram.db.models import Chat
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.bot.setup_workflow import SetupSession
    from mai_gram.config import Settings
    from mai_gram.config_loaders import PromptConfig
    from mai_gram.messenger.base import IncomingMessage, Messenger

logger = logging.getLogger(__name__)


def build_chat_record(
    chat_id: str,
    message: IncomingMessage,
    session: SetupSession,
    system_prompt: str,
    *,
    settings: Settings,
    prompt_name: str | None,
    template_name: str | None,
    params_json: str | None,
    prompt_cfg: PromptConfig | None,
    custom_model_params_json: str | None,
) -> Chat:
    """Build a :class:`Chat` row from a completed setup session."""
    send_dt = True
    if prompt_cfg is not None and prompt_cfg.send_datetime is not None:
        send_dt = prompt_cfg.send_datetime
    return Chat(
        id=chat_id,
        user_id=message.user_id,
        bot_id=message.bot_id or "",
        llm_model=session.selected_model,
        system_prompt=system_prompt,
        prompt_name=prompt_name,
        response_template=template_name,
        template_params=params_json,
        custom_model_params=custom_model_params_json,
        timezone=settings.default_timezone,
        show_reasoning=prompt_cfg.show_reasoning if prompt_cfg else True,
        show_tool_calls=prompt_cfg.show_tool_calls if prompt_cfg else True,
        send_datetime=send_dt,
    )


async def send_setup_confirmation(
    messenger: Messenger,
    message: IncomingMessage,
    session: SetupSession,
    chat: Chat,
    system_prompt: str,
    template_name: str | None,
) -> None:
    """Send the post-setup confirmation summary to the user."""
    reasoning_status = "ON" if chat.show_reasoning else "OFF"
    toolcalls_status = "ON" if chat.show_tool_calls else "OFF"
    datetime_status = "ON" if chat.send_datetime else "OFF"
    tpl_display = template_name or "empty"
    custom_line = ""
    if session.custom_model_params:
        summary = custom_model.format_params_summary(session.custom_model_params)
        custom_line = f"Custom params: {summary}\n"
    await messenger.send_message(
        OutgoingMessage(
            text=(
                "Chat created!\n"
                f"Model: {session.selected_model}\n"
                f"{custom_line}"
                f"Prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}\n"
                f"Template: {tpl_display}\n"
                f"Reasoning: {reasoning_status} | Tool calls: {toolcalls_status} "
                f"| Datetime: {datetime_status}\n\n"
                "Send a message to start chatting.\n"
                "Toggle display with /reasoning, /toolcalls, /datetime, and /toggle."
            ),
            chat_id=message.chat_id,
        )
    )
    logger.info(
        "Created chat: id=%s model=%s prompt_len=%d template=%s reasoning=%s toolcalls=%s",
        chat.id,
        session.selected_model,
        len(system_prompt),
        tpl_display,
        chat.show_reasoning,
        chat.show_tool_calls,
    )
