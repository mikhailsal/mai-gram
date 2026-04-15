"""Convert standard markdown to Telegram MarkdownV2 or HTML format.

Telegram MarkdownV2 is very strict about escaping. All special characters
outside of formatting entities must be escaped with a preceding backslash.

Supported conversions:
- **bold** -> *bold* (MDv2) or <b>bold</b> (HTML)
- *italic* or _italic_ -> _italic_ (MDv2) or <i>italic</i> (HTML)
- `code` -> `code` (MDv2) or <code>code</code> (HTML)
- ```lang\\nblock``` -> ```lang\\nblock``` (MDv2) or <pre>block</pre> (HTML)
- [text](url) -> [text](url) (MDv2) or <a href="url">text</a> (HTML)
- ~~strike~~ -> ~strike~ (MDv2) or <s>strike</s> (HTML)
"""

from __future__ import annotations

import html as _html_mod
import re

SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!\\"


def _escape_mdv2(text: str) -> str:
    """Escape all MarkdownV2 special characters in plain text."""
    result = []
    for ch in text:
        if ch in SPECIAL_CHARS:
            result.append("\\")
        result.append(ch)
    return "".join(result)


def _escape_inside_entity(text: str, entity_chars: str = "") -> str:
    """Escape special chars inside a formatting entity, except the entity's own chars."""
    result = []
    for ch in text:
        if ch in SPECIAL_CHARS and ch not in entity_chars:
            result.append("\\")
        result.append(ch)
    return "".join(result)


def markdown_to_mdv2(text: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2.

    Processes the text in order: code blocks -> inline code ->
    links -> bold -> italic -> strikethrough -> escape remaining.
    """
    if not text:
        return text

    placeholders: dict[str, str] = {}
    counter = [0]

    def _placeholder(converted: str) -> str:
        key = f"\x00PH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = converted
        return key

    result = _convert_code_blocks(text, _placeholder)
    result = _convert_inline_code(result, _placeholder)
    result = _protect_list_bullets(result)
    result = _convert_links(result, _placeholder)
    result = _convert_bold(result, _placeholder)
    result = _convert_italic(result, _placeholder)
    result = _convert_strikethrough(result, _placeholder)

    parts = re.split(r"(\x00PH\d+\x00)", result)
    final_parts = []
    for part in parts:
        if part in placeholders:
            final_parts.append(placeholders[part])
        else:
            final_parts.append(_escape_mdv2(part))

    return "".join(final_parts)


_BULLET = "\u2022"


def _protect_list_bullets(text: str) -> str:
    """Replace markdown list markers (* or -) at line start with bullet char.

    This prevents them from being interpreted as bold/italic markers.
    """
    return re.sub(r"^(\s*)[*\-]\s", rf"\1{_BULLET} ", text, flags=re.MULTILINE)


def _convert_code_blocks(
    text: str, placeholder: callable,
) -> str:
    """Convert fenced code blocks: ```lang\\n...``` -> ```lang\\n...```"""
    def _repl(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = m.group(2)
        # Inside code blocks, only ``` and \ need escaping
        escaped_code = code.replace("\\", "\\\\").replace("`", "\\`")
        if lang:
            return placeholder(f"```{lang}\n{escaped_code}```")
        return placeholder(f"```\n{escaped_code}```")

    return re.sub(
        r"```(\w*)\n(.*?)```",
        _repl,
        text,
        flags=re.DOTALL,
    )


def _convert_inline_code(
    text: str, placeholder: callable,
) -> str:
    """Convert inline code: `code` -> `code`"""
    def _repl(m: re.Match) -> str:
        code = m.group(1)
        escaped = code.replace("\\", "\\\\").replace("`", "\\`")
        return placeholder(f"`{escaped}`")

    return re.sub(r"`([^`]+)`", _repl, text)


def _convert_links(
    text: str, placeholder: callable,
) -> str:
    """Convert links: [text](url) -> [text](url)"""
    def _repl(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2)
        escaped_text = _escape_inside_entity(link_text, "")
        escaped_url = url.replace("\\", "\\\\").replace(")", "\\)")
        return placeholder(f"[{escaped_text}]({escaped_url})")

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


def _convert_bold(
    text: str, placeholder: callable,
) -> str:
    """Convert **bold** -> *bold*"""
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "*")
        return placeholder(f"*{escaped}*")

    return re.sub(r"\*\*(.+?)\*\*", _repl, text, flags=re.DOTALL)


def _convert_italic(
    text: str, placeholder: callable,
) -> str:
    r"""Convert *italic* or _italic_ -> _italic_"""
    def _repl_star(m: re.Match) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "_")
        return placeholder(f"_{escaped}_")

    def _repl_under(m: re.Match) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "_")
        return placeholder(f"_{escaped}_")

    result = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", _repl_star, text)
    result = re.sub(r"(?<!\\)_([^_]+?)_", _repl_under, result)
    return result


def _convert_strikethrough(
    text: str, placeholder: callable,
) -> str:
    """Convert ~~strike~~ -> ~strike~"""
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "~")
        return placeholder(f"~{escaped}~")

    return re.sub(r"~~(.+?)~~", _repl, text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML
# ---------------------------------------------------------------------------


def markdown_to_html(text: str) -> str:
    """Convert standard markdown to Telegram-compatible HTML.

    Processes code blocks and inline code first (protecting their contents),
    then converts remaining formatting to HTML tags. Plain text is HTML-escaped.
    """
    if not text:
        return text

    placeholders: dict[str, str] = {}
    counter = [0]

    def _ph(converted: str) -> str:
        key = f"\x00HH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = converted
        return key

    result = _html_code_blocks(text, _ph)
    result = _html_inline_code(result, _ph)
    result = _protect_list_bullets(result)
    result = _html_links(result, _ph)
    result = _html_bold(result, _ph)
    result = _html_italic(result, _ph)
    result = _html_strikethrough(result, _ph)

    parts = re.split(r"(\x00HH\d+\x00)", result)
    final_parts = []
    for part in parts:
        if part in placeholders:
            final_parts.append(placeholders[part])
        else:
            final_parts.append(_html_mod.escape(part))

    return "".join(final_parts)


def _html_code_blocks(text: str, ph: callable) -> str:
    def _repl(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = _html_mod.escape(m.group(2))
        if lang:
            return ph(f'<pre language="{_html_mod.escape(lang)}">{code}</pre>')
        return ph(f"<pre>{code}</pre>")

    return re.sub(r"```(\w*)\n(.*?)```", _repl, text, flags=re.DOTALL)


def _html_inline_code(text: str, ph: callable) -> str:
    def _repl(m: re.Match) -> str:
        return ph(f"<code>{_html_mod.escape(m.group(1))}</code>")

    return re.sub(r"`([^`]+)`", _repl, text)


def _html_links(text: str, ph: callable) -> str:
    def _repl(m: re.Match) -> str:
        link_text = _html_mod.escape(m.group(1))
        url = _html_mod.escape(m.group(2))
        return ph(f'<a href="{url}">{link_text}</a>')

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


def _html_bold(text: str, ph: callable) -> str:
    def _repl(m: re.Match) -> str:
        return ph(f"<b>{m.group(1)}</b>")

    return re.sub(r"\*\*(.+?)\*\*", _repl, text, flags=re.DOTALL)


def _html_italic(text: str, ph: callable) -> str:
    def _repl_star(m: re.Match) -> str:
        return ph(f"<i>{m.group(1)}</i>")

    def _repl_under(m: re.Match) -> str:
        return ph(f"<i>{m.group(1)}</i>")

    result = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", _repl_star, text)
    return re.sub(r"(?<!\\)_([^_]+?)_", _repl_under, result)


def _html_strikethrough(text: str, ph: callable) -> str:
    def _repl(m: re.Match) -> str:
        return ph(f"<s>{m.group(1)}</s>")

    return re.sub(r"~~(.+?)~~", _repl, text, flags=re.DOTALL)
