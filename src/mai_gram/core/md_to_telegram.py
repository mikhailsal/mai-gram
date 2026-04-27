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
- # Header -> bold header with separator
- $\\rightarrow$ and other LaTeX math symbols -> Unicode equivalents
- Numbered lists (1. item) and nested list indentation
"""

from __future__ import annotations

import html as _html_mod
import re
from typing import TYPE_CHECKING

from mai_gram.core.md_to_telegram_html import (
    _html_blockquotes,
    _html_bold,
    _html_code_blocks,
    _html_headers,
    _html_inline_code,
    _html_italic,
    _html_links,
    _html_strikethrough,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!\\"

# LaTeX math symbol -> Unicode mapping
_LATEX_SYMBOLS: dict[str, str] = {
    r"\rightarrow": "\u2192",
    r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2",
    r"\Leftarrow": "\u21d0",
    r"\leftrightarrow": "\u2194",
    r"\Leftrightarrow": "\u21d4",
    r"\uparrow": "\u2191",
    r"\downarrow": "\u2193",
    r"\mapsto": "\u21a6",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    r"\neq": "\u2260",
    r"\approx": "\u2248",
    r"\equiv": "\u2261",
    r"\sim": "\u223c",
    r"\pm": "\u00b1",
    r"\mp": "\u2213",
    r"\times": "\u00d7",
    r"\div": "\u00f7",
    r"\cdot": "\u00b7",
    r"\infty": "\u221e",
    r"\partial": "\u2202",
    r"\nabla": "\u2207",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\sqrt": "\u221a",
    r"\forall": "\u2200",
    r"\exists": "\u2203",
    r"\in": "\u2208",
    r"\notin": "\u2209",
    r"\subset": "\u2282",
    r"\supset": "\u2283",
    r"\cup": "\u222a",
    r"\cap": "\u2229",
    r"\emptyset": "\u2205",
    r"\land": "\u2227",
    r"\lor": "\u2228",
    r"\neg": "\u00ac",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\theta": "\u03b8",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\pi": "\u03c0",
    r"\sigma": "\u03c3",
    r"\omega": "\u03c9",
    r"\Delta": "\u0394",
    r"\Sigma": "\u03a3",
    r"\Omega": "\u03a9",
    r"\Pi": "\u03a0",
    r"\Theta": "\u0398",
    r"\Lambda": "\u039b",
}

_LATEX_INLINE_RE = re.compile(r"\$([^$]+?)\$")


def _replace_latex_symbols(text: str) -> str:
    """Replace LaTeX math expressions with Unicode equivalents.

    Handles both inline ``$...$`` and bare ``\\command`` forms.
    """

    def _replace_inline(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        result = inner
        for cmd, uni in _LATEX_SYMBOLS.items():
            result = result.replace(cmd, uni)
        result = result.strip()
        if result != inner:
            return result
        return m.group(0)

    text = _LATEX_INLINE_RE.sub(_replace_inline, text)

    for cmd, uni in _LATEX_SYMBOLS.items():
        text = text.replace(cmd, uni)

    return text


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
    LaTeX symbols -> headers -> lists -> links -> bold -> italic ->
    strikethrough -> escape remaining.
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
    result = _replace_latex_symbols(result)
    result = _convert_horizontal_rules(result, _placeholder)
    result = _convert_headers_mdv2(result, _placeholder)
    result = _convert_lists(result)
    result = _convert_links(result, _placeholder)
    result = _convert_bold(result, _placeholder)
    result = _convert_italic(result, _placeholder)
    result = _convert_strikethrough(result, _placeholder)

    return _resolve_placeholders(result, placeholders, escape_fn=_escape_mdv2)


_PLACEHOLDER_RE = re.compile(r"(\x00(?:PH|HH)\d+\x00)")
_HR_CHAR = "\u2500"
_HR_LINE = _HR_CHAR * 20

_BULLET = "\u2022"
_INDENT_UNIT = "    "
_NESTED_BULLETS = ["\u2022", "\u25e6", "\u25aa", "\u25ab"]


def _protect_list_bullets(text: str) -> str:
    """Replace markdown list markers (* or -) at line start with bullet char.

    This prevents them from being interpreted as bold/italic markers.
    """
    return re.sub(r"^(\s*)[*\-]\s", rf"\1{_BULLET} ", text, flags=re.MULTILINE)


def _convert_horizontal_rules(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert markdown horizontal rules (``***``, ``---``, ``___``) to a visual separator.

    Must run BEFORE bold/italic conversion to prevent ``***`` from being
    misinterpreted as bold-start (``**`` + leftover ``*``).
    """

    def _repl(m: re.Match[str]) -> str:
        return placeholder(f"\n{_HR_LINE}\n")

    return re.sub(r"^[ \t]*([*\-_])\1{2,}[ \t]*$", _repl, text, flags=re.MULTILINE)


def _resolve_placeholders(
    text: str,
    placeholders: dict[str, str],
    *,
    escape_fn: Callable[[str], str],
) -> str:
    """Resolve placeholder tokens, handling nested placeholders.

    Placeholder values may themselves contain placeholder keys (e.g. when
    bold wraps a region that already had header placeholders). This function
    resolves them recursively until no placeholders remain.
    """
    parts = _PLACEHOLDER_RE.split(text)
    final_parts: list[str] = []
    for part in parts:
        if part in placeholders:
            final_parts.append(placeholders[part])
        else:
            final_parts.append(escape_fn(part))

    result = "".join(final_parts)

    # Resolve any nested placeholders that ended up inside a placeholder value
    safety = 10
    while "\x00" in result and safety > 0:
        safety -= 1
        parts = _PLACEHOLDER_RE.split(result)
        new_parts: list[str] = []
        changed = False
        for part in parts:
            if part in placeholders:
                new_parts.append(placeholders[part])
                changed = True
            else:
                new_parts.append(part)
        if not changed:
            break
        result = "".join(new_parts)

    return result


def _convert_lists(text: str) -> str:
    """Convert markdown lists with proper indentation for nested items.

    Handles:
    - Unordered: ``* item``, ``- item`` with nesting via indentation
    - Numbered: ``1. item``, ``2. item`` with nesting
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        m_bullet = re.match(r"^(\s*)[*\-]\s+(.*)$", line)
        m_number = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)

        if m_bullet:
            indent = m_bullet.group(1)
            content = m_bullet.group(2)
            depth = len(indent) // 4 if indent else 0
            depth = min(depth, len(_NESTED_BULLETS) - 1)
            bullet = _NESTED_BULLETS[depth]
            visual_indent = _INDENT_UNIT * depth
            result.append(f"{visual_indent}{bullet} {content}")
        elif m_number:
            indent = m_number.group(1)
            number = m_number.group(2)
            content = m_number.group(3)
            depth = len(indent) // 4 if indent else 0
            visual_indent = _INDENT_UNIT * depth
            result.append(f"{visual_indent}{number}. {content}")
        else:
            result.append(line)

    return "\n".join(result)


def _convert_headers_mdv2(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    r"""Convert markdown headers to bold text with a separator line.

    ``# Title`` becomes ``*Title*`` (bold in MDv2) preceded by a blank line.
    """

    def _repl(m: re.Match[str]) -> str:
        content = m.group(2).strip()
        escaped = _escape_inside_entity(content, "*")
        return placeholder(f"\n*{escaped}*\n")

    return re.sub(r"^(#{1,6})\s+(.+)$", _repl, text, flags=re.MULTILINE)


def _convert_code_blocks(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert fenced code blocks: ```lang\\n...``` -> ```lang\\n...```"""

    def _repl(m: re.Match[str]) -> str:
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
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert inline code: `code` -> `code`"""

    def _repl(m: re.Match[str]) -> str:
        code = m.group(1)
        escaped = code.replace("\\", "\\\\").replace("`", "\\`")
        return placeholder(f"`{escaped}`")

    return re.sub(r"`([^`]+)`", _repl, text)


def _convert_links(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert links: [text](url) -> [text](url)"""

    def _repl(m: re.Match[str]) -> str:
        link_text = m.group(1)
        url = m.group(2)
        escaped_text = _escape_inside_entity(link_text, "")
        escaped_url = url.replace("\\", "\\\\").replace(")", "\\)")
        return placeholder(f"[{escaped_text}]({escaped_url})")

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


def _convert_bold(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert **bold** -> *bold*"""

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "*")
        return placeholder(f"*{escaped}*")

    return re.sub(r"\*\*(.+?)\*\*", _repl, text, flags=re.DOTALL)


def _convert_italic(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    r"""Convert *italic* or _italic_ -> _italic_"""

    def _repl_star(m: re.Match[str]) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "_")
        return placeholder(f"_{escaped}_")

    def _repl_under(m: re.Match[str]) -> str:
        inner = m.group(1)
        escaped = _escape_inside_entity(inner, "_")
        return placeholder(f"_{escaped}_")

    result = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", _repl_star, text)
    result = re.sub(r"(?<!\\)_([^_]+?)_", _repl_under, result)
    return result


def _convert_strikethrough(
    text: str,
    placeholder: Callable[[str], str],
) -> str:
    """Convert ~~strike~~ -> ~strike~"""

    def _repl(m: re.Match[str]) -> str:
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
    result = _replace_latex_symbols(result)
    result = _convert_horizontal_rules(result, _ph)
    result = _html_headers(result, _ph)
    result = _convert_lists(result)
    result = _html_blockquotes(result, _ph)
    result = _html_links(result, _ph)
    result = _html_bold(result, _ph)
    result = _html_italic(result, _ph)
    result = _html_strikethrough(result, _ph)

    return _resolve_placeholders(result, placeholders, escape_fn=_html_mod.escape)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def format_reasoning_html(reasoning: str, *, expandable: bool = False) -> str:
    """Format reasoning text as a Telegram HTML blockquote with parsed markdown.

    Applies markdown-to-HTML conversion to the reasoning content so that
    list items, bold/italic, nested structures, and LaTeX symbols render
    correctly inside the blockquote.
    """
    if not reasoning or not reasoning.strip():
        return ""

    inner_html = markdown_to_html(reasoning.strip())
    tag = "blockquote expandable" if expandable else "blockquote"
    return f"<{tag}>\U0001f4ad Reasoning\n{inner_html}</{tag.split()[0]}>"
