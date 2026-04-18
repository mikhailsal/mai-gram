"""Shared dialogue import logic.

Parses JSON data (OpenAI chat format or AI Proxy v2 request format)
and saves messages to the database via MessageStore.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gram.memory.messages import MessageStore

logger = logging.getLogger(__name__)


class ImportError(Exception):  # noqa: A001
    """Raised when import data cannot be parsed or validated."""


def _extract_messages_from_proxy_request(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract messages from an AI Proxy v2 request JSON.

    The proxy format stores the full request/response cycle. Messages come
    from ``request_body.messages`` and the assistant's response is appended
    from ``response_body.choices[0].message``.
    """
    request_body = data.get("request_body") or data.get("client_request_body")
    if not isinstance(request_body, dict):
        raise ImportError("Proxy request JSON has no request_body.")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ImportError("Proxy request_body has no messages array.")

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
    Raises ImportError on parsing failures.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ImportError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, dict):
        if "request_body" in parsed or "client_request_body" in parsed:
            return _extract_messages_from_proxy_request(parsed)
        raise ImportError(
            "JSON object does not look like a proxy request. "
            "Expected a JSON array or an object with request_body."
        )

    if not isinstance(parsed, list):
        raise ImportError("Expected a JSON array of message objects.")

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
) -> int:
    """Save a list of parsed message dicts to the database.

    Skips system messages. When multiple messages share the same timestamp
    (common in proxy format where all messages inherit a single timestamp),
    each subsequent message gets a 1ms increment to maintain chronological order.

    Returns the count of successfully imported messages.
    """
    imported = 0
    last_saved_ts: datetime | None = None

    logger.info(
        "save_imported_messages: chat_id=%s, total entries=%d",
        chat_id,
        len(messages),
    )

    for i, entry in enumerate(messages):
        if not isinstance(entry, dict):
            logger.warning("Skipping entry %d: not a JSON object", i)
            continue

        role = entry.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            logger.warning("Skipping entry %d: invalid role '%s'", i, role)
            continue

        if role == "system":
            logger.debug("Skipping entry %d: system message", i)
            continue

        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        reasoning_text: str | None = None
        reasoning_raw = entry.get("reasoning") or entry.get("reasoning_content")
        if isinstance(reasoning_raw, str) and reasoning_raw.strip():
            reasoning_text = reasoning_raw.strip()

        tool_calls_raw = entry.get("tool_calls")
        tool_calls_json: str | None = None
        if isinstance(tool_calls_raw, list) and tool_calls_raw:
            tc_list = []
            for tc in tool_calls_raw:
                if isinstance(tc, dict):
                    func = tc.get("function", tc)
                    tc_list.append(
                        {
                            "id": tc.get("id", f"import_{i}"),
                            "name": func.get("name", "unknown"),
                            "arguments": func.get("arguments", "{}"),
                        }
                    )
            if tc_list:
                tool_calls_json = json.dumps(tc_list)

        tool_call_id = entry.get("tool_call_id")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            tool_call_id = str(tool_call_id)

        timestamp = None
        ts_raw = entry.get("timestamp")
        if isinstance(ts_raw, str):
            with contextlib.suppress(ValueError):
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))

        if timestamp is not None and last_saved_ts is not None and timestamp <= last_saved_ts:
            new_ts = last_saved_ts + timedelta(milliseconds=1)
            logger.debug(
                "Entry %d: bumping timestamp %s -> %s",
                i,
                timestamp.isoformat(),
                new_ts.isoformat(),
            )
            timestamp = new_ts

        logger.debug(
            "Entry %d: saving role=%s, content_len=%d, ts=%s",
            i,
            role,
            len(content),
            timestamp.isoformat() if timestamp else "None",
        )

        try:
            await message_store.save_message(
                chat_id=chat_id,
                role=role,
                content=content,
                timestamp=timestamp,
                tool_calls=tool_calls_json,
                tool_call_id=tool_call_id,
                reasoning=reasoning_text,
                show_datetime=False,
            )
        except ValueError as exc:
            logger.warning("Skipping entry %d due to timestamp conflict: %s", i, exc)
            continue

        if timestamp is not None:
            last_saved_ts = timestamp
        imported += 1
        logger.debug("Entry %d saved successfully (imported=%d)", i, imported)

    logger.info("save_imported_messages: done. imported=%d", imported)
    return imported
