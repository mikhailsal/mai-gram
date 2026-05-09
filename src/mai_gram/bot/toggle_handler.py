"""Handler for /toggle command -- show/hide template fields."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from mai_gram.messenger.base import OutgoingMessage
from mai_gram.response_templates.registry import get_template

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.db.models import Chat
    from mai_gram.messenger.base import IncomingMessage, Messenger

logger = logging.getLogger(__name__)


async def handle_toggle(
    message: IncomingMessage,
    chat: Chat,
    session: AsyncSession,
    messenger: Messenger,
) -> None:
    """Handle /toggle command for a resolved chat."""
    field_arg = (message.command_args or "").strip().lower()
    tpl_params = _parse_template_params(chat)
    template = get_template(chat.response_template, tpl_params)
    hideable = [f for f in template.get_fields() if f.user_can_hide]

    if not hideable:
        await messenger.send_message(
            OutgoingMessage(
                text="This template has no toggleable fields.",
                chat_id=message.chat_id,
            )
        )
        return

    if not field_arg:
        names = ", ".join(f.name for f in hideable)
        await messenger.send_message(
            OutgoingMessage(
                text=f"Usage: /toggle <field_name>\nToggleable fields: {names}",
                chat_id=message.chat_id,
            )
        )
        return

    hideable_names = {f.name.lower(): f.name for f in hideable}
    if field_arg not in hideable_names:
        names = ", ".join(f.name for f in hideable)
        await messenger.send_message(
            OutgoingMessage(
                text=f"Unknown or non-toggleable field: {field_arg}\nToggleable: {names}",
                chat_id=message.chat_id,
            )
        )
        return

    canonical_name = hideable_names[field_arg]
    status = _toggle_field(chat, canonical_name)
    await session.commit()

    await messenger.send_message(
        OutgoingMessage(
            text=f"Field '{canonical_name}': {status}",
            chat_id=message.chat_id,
        )
    )


def _parse_template_params(chat: Chat) -> dict[str, object] | None:
    raw = getattr(chat, "template_params", None)
    if not raw:
        return None
    try:
        data: dict[str, object] = json.loads(raw)
        return data
    except (ValueError, TypeError):
        return None


def _toggle_field(chat: Chat, canonical_name: str) -> str:
    """Toggle a field in the hidden_template_fields set and return status string."""
    try:
        hidden = set(json.loads(chat.hidden_template_fields or "[]"))
    except (json.JSONDecodeError, TypeError):
        hidden = set()

    if canonical_name in hidden:
        hidden.discard(canonical_name)
        status = "VISIBLE"
    else:
        hidden.add(canonical_name)
        status = "HIDDEN"

    chat.hidden_template_fields = json.dumps(sorted(hidden)) if hidden else None
    return status
