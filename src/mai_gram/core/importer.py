"""Shared dialogue import logic.

Parses JSON data (OpenAI chat format or AI Proxy v2 request format)
and saves messages to the database via MessageStore.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from mai_gram.llm.provider import MessageRole, ToolCall

if TYPE_CHECKING:
    from mai_gram.memory.messages import MessageStore
    from mai_gram.response_templates.base import ResponseTemplate

logger = logging.getLogger(__name__)


class ImportDataError(Exception):
    """Raised when import data cannot be parsed or validated."""


def _normalize_import_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ""


def _extract_reasoning_text(entry: dict[str, Any]) -> str | None:
    reasoning_raw = entry.get("reasoning") or entry.get("reasoning_content")
    if isinstance(reasoning_raw, str) and reasoning_raw.strip():
        return reasoning_raw.strip()
    return None


def _normalize_tool_calls(entry: dict[str, Any], *, entry_index: int) -> list[ToolCall] | None:
    tool_calls_raw = entry.get("tool_calls")
    if not isinstance(tool_calls_raw, list) or not tool_calls_raw:
        return None

    normalized_tool_calls: list[ToolCall] = []
    for tool_call in tool_calls_raw:
        if not isinstance(tool_call, dict):
            continue

        function_payload = tool_call.get("function", tool_call)
        if not isinstance(function_payload, dict):
            continue

        tool_call_id = tool_call.get("id", f"import_{entry_index}")
        name = function_payload.get("name", "unknown")
        arguments = function_payload.get("arguments", "{}")
        if not isinstance(tool_call_id, str):
            tool_call_id = str(tool_call_id)
        if not isinstance(name, str):
            name = str(name)
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        normalized_tool_calls.append(ToolCall(id=tool_call_id, name=name, arguments=arguments))

    if not normalized_tool_calls:
        return None
    return normalized_tool_calls


def _normalize_tool_call_id(entry: dict[str, Any]) -> str | None:
    tool_call_id = entry.get("tool_call_id")
    if tool_call_id is None:
        return None
    if isinstance(tool_call_id, str):
        return tool_call_id
    return str(tool_call_id)


def _validate_import_entry(entry: Any, *, entry_index: int) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        logger.warning("Skipping entry %d: not a JSON object", entry_index)
        return None

    role = entry.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        logger.warning("Skipping entry %d: invalid role '%s'", entry_index, role)
        return None

    if role == "system":
        logger.debug("Skipping entry %d: system message", entry_index)
        return None

    return entry


def _wrap_reasoning_in_template(
    reasoning: str,
    content: str,
    template: ResponseTemplate,
) -> str:
    """Merge native reasoning and content into a template-structured string.

    Uses the template's field definitions to determine the tag names:
    the first field (by order) becomes the reasoning wrapper, and the
    content field wraps the original response body.
    """
    fields = sorted(template.get_fields(), key=lambda f: f.order)
    reasoning_field = fields[0].name
    content_field = template.content_field_name()
    return (
        f"<{reasoning_field}>\n{reasoning}\n</{reasoning_field}>\n"
        f"<{content_field}>\n{content}\n</{content_field}>"
    )


def _build_import_message_payload(
    entry: dict[str, Any],
    *,
    entry_index: int,
    timestamp: datetime,
    reasoning_template: ResponseTemplate | None = None,
) -> dict[str, Any]:
    content = _normalize_import_content(entry.get("content", ""))
    reasoning = _extract_reasoning_text(entry)

    if reasoning_template is not None and reasoning and entry.get("role") == "assistant":
        content = _wrap_reasoning_in_template(reasoning, content, reasoning_template)
        reasoning = None

    return {
        "role": MessageRole(entry["role"]),
        "content": content,
        "timestamp": timestamp,
        "tool_calls": _normalize_tool_calls(entry, entry_index=entry_index),
        "tool_call_id": _normalize_tool_call_id(entry),
        "reasoning": reasoning,
        "show_datetime": False,
    }


async def _save_import_entry(
    chat_id: str,
    payload: dict[str, Any],
    message_store: MessageStore,
    *,
    entry_index: int,
) -> bool:
    logger.debug(
        "Entry %d: saving role=%s, content_len=%d, ts=%s",
        entry_index,
        payload["role"],
        len(payload["content"]),
        payload["timestamp"].isoformat(),
    )

    try:
        await message_store.save_message(chat_id=chat_id, **payload)
    except ValueError as exc:
        logger.warning("Skipping entry %d due to timestamp conflict: %s", entry_index, exc)
        return False

    logger.debug("Entry %d saved successfully", entry_index)
    return True


def _extract_messages_from_proxy_request(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract messages from an AI Proxy v2 request JSON.

    The proxy format stores the full request/response cycle. Messages come
    from ``request_body.messages`` and the assistant's response is appended
    from ``response_body.choices[0].message``.
    """
    request_body = data.get("request_body") or data.get("client_request_body")
    if not isinstance(request_body, dict):
        raise ImportDataError("Proxy request JSON has no request_body.")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ImportDataError("Proxy request_body has no messages array.")

    result = list(messages)

    response_body = data.get("response_body") or data.get("client_response_body")
    if isinstance(response_body, dict):
        choices = response_body.get("choices", [])
        if choices and isinstance(choices[0], dict):
            resp_message = choices[0].get("message", {})
            if isinstance(resp_message, dict) and resp_message.get("role") == "assistant":
                result.append(resp_message)

    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        for msg in result:
            if isinstance(msg, dict) and "timestamp" not in msg:
                msg["timestamp"] = timestamp

    return result


def parse_import_json(data: str | bytes) -> list[dict[str, Any]]:
    """Parse import data from JSON string/bytes into a list of message dicts.

    Supports two formats:
    1. Array of message objects (OpenAI chat completion format)
    2. AI Proxy v2 request JSON (object with request_body.messages)

    Returns the normalized list of message dicts.
    Raises ImportDataError on parsing failures.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ImportDataError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, dict):
        if "request_body" in parsed or "client_request_body" in parsed:
            return _extract_messages_from_proxy_request(parsed)
        raise ImportDataError(
            "JSON object does not look like a proxy request. "
            "Expected a JSON array or an object with request_body."
        )

    if not isinstance(parsed, list):
        raise ImportDataError("Expected a JSON array of message objects.")

    return parsed


def extract_system_prompt(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the first 'system' role message, if any."""
    for entry in messages:
        if isinstance(entry, dict) and entry.get("role") == "system":
            content = entry.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


async def save_imported_messages(
    chat_id: str,
    messages: list[dict[str, Any]],
    message_store: MessageStore,
    *,
    reasoning_template: ResponseTemplate | None = None,
) -> int:
    """Save a list of parsed message dicts to the database.

    Skips system messages. Each non-system message receives a timestamp
    based on the import moment (now) with 1-second increments. Original
    timestamps from the JSON are ignored. The ``show_datetime=False`` flag
    tells the formatter to display "[imported, real date unknown]" instead
    of the synthetic timestamp.

    When *reasoning_template* is provided, assistant messages that carry
    native ``reasoning_content`` are transformed: the reasoning and content
    are merged into a single structured string using the template's field
    tags (e.g. ``<thought>...<content>``), so the reasoning is preserved
    in conversation history rather than being stored in a provider-strippable
    side column.

    Returns the count of successfully imported messages.
    """
    imported = 0
    import_base = datetime.now(tz=timezone.utc)

    logger.info(
        "save_imported_messages: chat_id=%s, total entries=%d, reasoning_template=%s",
        chat_id,
        len(messages),
        reasoning_template.name if reasoning_template else None,
    )

    for i, entry in enumerate(messages):
        normalized_entry = _validate_import_entry(entry, entry_index=i)
        if normalized_entry is None:
            continue

        timestamp = import_base + timedelta(seconds=imported)
        payload = _build_import_message_payload(
            normalized_entry,
            entry_index=i,
            timestamp=timestamp,
            reasoning_template=reasoning_template,
        )
        saved = await _save_import_entry(
            chat_id,
            payload,
            message_store,
            entry_index=i,
        )
        if not saved:
            continue

        imported += 1
        logger.debug("Entry %d saved successfully (imported=%d)", i, imported)

    logger.info("save_imported_messages: done. imported=%d", imported)
    return imported
