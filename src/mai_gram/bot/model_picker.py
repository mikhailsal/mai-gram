"""In-place model switching for an already-started chat (the ``/model`` command).

Unlike ``/start``, switching here never creates or wipes a chat -- it only
updates ``Chat.llm_model`` so the conversation history is preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from mai_gram.config import BotConfig, Settings
    from mai_gram.messenger.base import IncomingMessage, Messenger

logger = logging.getLogger(__name__)


def allowed_models_for_bot(settings: Settings, bot_config: BotConfig | None) -> list[str]:
    """Return the models a bot may use (per-bot whitelist intersects the global set)."""
    global_models = settings.get_allowed_models()
    if bot_config and bot_config.allowed_models:
        bot_set = set(bot_config.allowed_models)
        return [model for model in global_models if model in bot_set]
    return global_models


def model_display_label(settings: Settings, model_key: str) -> str:
    """Resolve a human-friendly label for a model key (title, else last path segment)."""
    title = settings.get_model_title(model_key)
    if title:
        return title
    return model_key.split("/")[-1] if "/" in model_key else model_key


async def show_model_picker(
    messenger: Messenger,
    message: IncomingMessage,
    *,
    current_model: str,
    allowed_models: list[str],
    label_for: Callable[[str], str],
) -> None:
    """Send an inline keyboard listing models plus a Cancel button.

    The currently active model is marked, and the Cancel button (which maps to
    the shared ``cancel_action`` callback) simply deletes the prompt message.
    """
    keyboard_rows: list[list[tuple[str, str]]] = []
    for model in allowed_models:
        label = label_for(model)
        if model == current_model:
            label = f"✅ {label}"
        keyboard_rows.append([(label, f"setmodel:{model}")])
    keyboard_rows.append([("Cancel", "cancel_action")])
    await messenger.send_message(
        OutgoingMessage(
            text=f"Current model: {current_model}\n\nChoose a new model:",
            chat_id=message.chat_id,
            keyboard=messenger.build_inline_keyboard(keyboard_rows),
        )
    )


async def apply_model_change(
    messenger: Messenger,
    message: IncomingMessage,
    model: str,
    *,
    chat_id: str,
    allowed_models: list[str],
) -> None:
    """Persist a new model for an existing chat without touching its history."""
    if allowed_models and model not in allowed_models:
        await messenger.send_message(
            OutgoingMessage(
                text="This model is not available for this bot. Please choose another.",
                chat_id=message.chat_id,
            )
        )
        return
    async with get_session() as session:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if not chat:
            await messenger.send_message(
                OutgoingMessage(
                    text="No chat exists yet. Use /start to create one.",
                    chat_id=message.chat_id,
                )
            )
            return
        old_model = chat.llm_model
        chat.llm_model = model
        await session.commit()

    await messenger.delete_callback_source_message(message)
    await messenger.send_message(
        OutgoingMessage(
            text=f"Model changed: {old_model} → {model}",
            chat_id=message.chat_id,
        )
    )
    logger.info("Changed model for chat %s: %s -> %s", chat_id, old_model, model)
