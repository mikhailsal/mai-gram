"""Chat-record construction and confirmation messaging for setup completion.

Extracted from ``setup_workflow`` to keep that module within size limits and to
give the custom-model flow a single place that knows how a finished chat is
persisted and announced.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from mai_gram.bot import custom_model
from mai_gram.db.models import Chat
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.bot.setup_workflow import SetupSession
    from mai_gram.config import BotConfig, Settings
    from mai_gram.config_loaders import PromptConfig
    from mai_gram.messenger.base import IncomingMessage, Messenger

DEFAULT_GUEST_WELCOME = "[System] Chat is ready. Send a message to start."

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


def _build_guest_chat_record(
    chat_id: str,
    message: IncomingMessage,
    *,
    bot_config: BotConfig,
    settings: Settings,
    allowed_models: list[str],
    allowed_prompts: dict[str, str],
) -> Chat:
    """Create a :class:`Chat` row using ``guest_mode`` defaults."""
    model = bot_config.default_model or (
        allowed_models[0] if allowed_models else settings.get_default_model()
    )
    all_prompts = settings.get_available_prompts()
    if bot_config.default_prompt and bot_config.default_prompt in all_prompts:
        prompt_name = bot_config.default_prompt
    else:
        prompt_name = next(iter(allowed_prompts)) if allowed_prompts else "default"
    prompt_text: str = (
        all_prompts.get(prompt_name)
        or next((text for text in all_prompts.values() if text), "")
        or ""
    )
    template_name = (
        None
        if not bot_config.default_template or bot_config.default_template == "empty"
        else bot_config.default_template
    )
    prompt_cfg = settings.get_prompt_config(prompt_name)
    send_dt = True
    if prompt_cfg is not None and prompt_cfg.send_datetime is not None:
        send_dt = prompt_cfg.send_datetime
    show_tool_calls = (
        bot_config.tool_calls_override
        if bot_config.tool_calls_override is not None
        else (prompt_cfg.show_tool_calls if prompt_cfg else True)
    )
    return Chat(
        id=chat_id,
        user_id=message.user_id,
        bot_id=message.bot_id or "",
        llm_model=model,
        system_prompt=prompt_text,
        prompt_name=prompt_name,
        response_template=template_name,
        template_params=None,
        custom_model_params=None,
        timezone=settings.default_timezone,
        show_reasoning=prompt_cfg.show_reasoning if prompt_cfg else True,
        show_tool_calls=show_tool_calls,
        send_datetime=send_dt,
    )


async def handle_guest_start(
    messenger: Messenger,
    message: IncomingMessage,
    *,
    chat: Chat | None,
    chat_id: str,
    bot_config: BotConfig,
    settings: Settings,
    allowed_models: list[str],
    allowed_prompts: dict[str, str],
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> None:
    """Handle ``/start`` in ``guest_mode`` without interactive keyboards."""
    if not chat:
        new_chat = _build_guest_chat_record(
            chat_id,
            message,
            bot_config=bot_config,
            settings=settings,
            allowed_models=allowed_models,
            allowed_prompts=allowed_prompts,
        )
        async with session_factory() as db:
            db.add(new_chat)
            await db.commit()
    welcome = bot_config.welcome_message or DEFAULT_GUEST_WELCOME
    await messenger.send_message(OutgoingMessage(text=welcome, chat_id=message.chat_id))


async def complete_setup(
    messenger: Messenger,
    message: IncomingMessage,
    session: SetupSession,
    system_prompt: str,
    *,
    chat_id: str,
    settings: Settings,
    prompt_name: str | None,
    template_name: str | None,
    template_params: dict[str, str] | None,
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> None:
    """Persist a completed setup session and send the confirmation message."""
    prompt_cfg = settings.get_prompt_config(prompt_name) if prompt_name else None
    params_json = json.dumps(template_params, ensure_ascii=False) if template_params else None
    custom_params = session.custom_model_params
    custom_params_json = json.dumps(custom_params, ensure_ascii=False) if custom_params else None

    chat = build_chat_record(
        chat_id,
        message,
        session,
        system_prompt,
        settings=settings,
        prompt_name=prompt_name,
        template_name=template_name,
        params_json=params_json,
        prompt_cfg=prompt_cfg,
        custom_model_params_json=custom_params_json,
    )
    async with session_factory() as db:
        db.add(chat)
        await db.commit()

    await send_setup_confirmation(messenger, message, session, chat, system_prompt, template_name)
