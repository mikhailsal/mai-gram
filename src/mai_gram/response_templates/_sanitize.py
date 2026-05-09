"""Shared sanitization utilities for response template auto-correction.

Provides two tiers of correction:
- Tier 1: Deterministic regex-based fixes for common LLM formatting mistakes.
- Tier 2: LLM-based repair using a cheap auxiliary model for complex cases.

All functions are template-agnostic and reusable across XML, JSON, and any
future template formats.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from mai_gram.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared XML regex and extraction (deduplicated from xml_template.py and
# gemma_reasoning_template.py)
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def extract_xml_fields(
    raw_text: str,
    field_names: list[str],
) -> dict[str, str]:
    """Extract content from XML-like tags, preserving order of first occurrence."""
    fields: dict[str, str] = {}
    for match in TAG_RE.finditer(raw_text):
        tag_name = match.group(1)
        if tag_name in field_names and tag_name not in fields:
            fields[tag_name] = match.group(2).strip()
    return fields


# ---------------------------------------------------------------------------
# Tier 1: Regex-based sanitization
# ---------------------------------------------------------------------------


def sanitize_xml_tags(raw_text: str, field_names: list[str]) -> str:
    """Fix common LLM mistakes in XML closing tags.

    Only corrects tags that match declared *field_names* -- never touches
    content between tags or arbitrary HTML/XML the model might embed.
    """
    text = raw_text
    for name in field_names:
        # Fix corrupted closing: <///name>, <//name>, </////name> -> </name>
        text = re.sub(rf"</{{2,}}{name}\s*>", f"</{name}>", text)

        # Fix duplicate closing: </name></name> -> </name>
        text = re.sub(rf"(</\s*{name}\s*>)\s*</\s*{name}\s*>", r"\1", text)

        # Fix unclosed tag: opening exists but no valid closing -> append at end
        if re.search(rf"<{name}>", text) and not re.search(rf"</{name}>", text):
            text += f"\n</{name}>"

    return text


def sanitize_json_structure(raw_text: str) -> str:
    """Fix common LLM mistakes in JSON output.

    Handles trailing commas, missing closing braces, and unescaped control
    characters inside string values.
    """
    import json

    text = raw_text

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    brace_start = text.find("{")
    if brace_start == -1:
        return text

    json_candidate = text[brace_start:]

    # Add missing closing brace if unbalanced
    open_count = json_candidate.count("{")
    close_count = json_candidate.count("}")
    if open_count > close_count:
        json_candidate += "}" * (open_count - close_count)
        text = text[:brace_start] + json_candidate

    # If JSON parses successfully already, return as-is
    try:
        json.loads(text[brace_start:])
        return text
    except json.JSONDecodeError:
        pass

    # Escape unescaped control characters (newlines, tabs, carriage returns)
    # inside JSON string values. We walk through the JSON candidate and only
    # escape raw control chars that appear between unescaped quotes.
    json_part = text[brace_start:]
    fixed = _escape_control_chars_in_json_strings(json_part)
    text = text[:brace_start] + fixed

    return text


def _escape_control_chars_in_json_strings(json_text: str) -> str:
    """Escape raw newlines, tabs, and carriage returns inside JSON strings.

    Walks through the text character-by-character, tracking whether we are
    inside a JSON string value, and replaces unescaped control characters
    with their proper escape sequences.
    """
    result: list[str] = []
    in_string = False
    i = 0
    n = len(json_text)

    while i < n:
        ch = json_text[i]

        if ch == '"' and (i == 0 or json_text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)

        i += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Tier 2: LLM-based repair
# ---------------------------------------------------------------------------

FORMAT_REPAIR_SYSTEM_PROMPT = """\
You are a FORMAT REPAIR tool. Your ONLY job is to fix structural formatting \
errors in the text below. You must:

1. NEVER change, rephrase, translate, shorten, or add any text content
2. NEVER remove or modify any words, sentences, or paragraphs
3. ONLY fix the structural tags/formatting that wraps the content
4. Output the COMPLETE text with ONLY the formatting fixed

The text must conform to this format:
{format_spec}

Common errors to fix:
- Corrupted closing tags (e.g. <///tag> should be </tag>)
- Duplicate closing tags (e.g. </tag></tag> should be </tag>)
- Missing closing tags (add them at the correct position)
- Broken JSON structure (fix braces, commas, escaping)
- Tags in wrong order (reorder without changing content)

CRITICAL: The content between tags must remain BYTE-FOR-BYTE IDENTICAL. \
Do not fix typos, grammar, or anything else in the content itself.\
"""


async def llm_repair(
    llm: LLMProvider,
    raw_text: str,
    format_spec: str,
    *,
    model: str = "openrouter/free",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 2,
    extra_params: dict[str, Any] | None = None,
    validator: Callable[[str], bool] | None = None,
) -> str:
    """Call the auxiliary LLM to repair structural formatting.

    Returns the repaired text on success, or the original *raw_text* on
    any failure (network error, empty response, etc.).  Retries transient
    errors up to *max_retries* times.
    """
    from mai_gram.llm.provider import ChatMessage, MessageRole

    system_prompt = FORMAT_REPAIR_SYSTEM_PROMPT.format(format_spec=format_spec)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=raw_text),
    ]

    for attempt in range(1, max_retries + 2):
        result = await _try_repair_attempt(
            llm,
            messages,
            raw_text,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
            validator=validator,
            attempt=attempt,
            max_retries=max_retries,
        )
        if result is not None:
            return result

    return raw_text


async def _try_repair_attempt(
    llm: LLMProvider,
    messages: list[Any],
    raw_text: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    extra_params: dict[str, Any] | None,
    validator: Callable[[str], bool] | None,
    attempt: int,
    max_retries: int,
) -> str | None:
    """Execute one repair attempt; return repaired text or None to continue."""
    import asyncio as _asyncio

    try:
        response = await llm.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )
    except Exception:
        if attempt <= max_retries:
            await _asyncio.sleep(1.0 * attempt)
            return None
        logger.warning("LLM format repair failed after %d attempts", attempt, exc_info=True)
        return raw_text

    repaired = response.content.strip()

    if repaired and (validator is None or validator(repaired)):
        return repaired

    if attempt <= max_retries:
        if repaired and validator is not None:
            logger.info("LLM repair attempt %d returned invalid output, retrying", attempt)
        await _asyncio.sleep(1.0 * attempt)
        return None

    if not repaired:
        logger.warning("LLM format repair returned empty after %d attempts", attempt)
    else:
        logger.warning("LLM format repair failed validation after %d attempts", attempt)
    return raw_text
