from __future__ import annotations

import html as _html_mod
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _html_headers(text: str, ph: Callable[[str], str]) -> str:
    """Convert markdown headers to bold HTML with visual separation.

    ``# Title`` becomes ``<b>Title</b>`` preceded by a blank line.
    """

    def _repl(m: re.Match[str]) -> str:
        content = m.group(2).strip()
        escaped = _html_mod.escape(content)
        return ph(f"\n<b>{escaped}</b>\n")

    return re.sub(r"^(#{1,6})\s+(.+)$", _repl, text, flags=re.MULTILINE)


def _render_html_blockquote_body(quote_buf: list[str]) -> str:
    inner = "\n".join(quote_buf)
    inner = _html_mod.escape(inner)
    inner = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: f"<b>{m.group(1)}</b>",
        inner,
        flags=re.DOTALL,
    )
    inner = re.sub(
        r"(?<!\*)\*([^*]+?)\*(?!\*)",
        lambda m: f"<i>{m.group(1)}</i>",
        inner,
    )
    inner = re.sub(
        r"(?<!\\)_([^_]+?)_",
        lambda m: f"<i>{m.group(1)}</i>",
        inner,
    )
    inner = re.sub(r"~~(.+?)~~", lambda m: f"<s>{m.group(1)}</s>", inner, flags=re.DOTALL)
    return re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", inner)


def _html_blockquotes(text: str, ph: Callable[[str], str]) -> str:
    """Convert consecutive `> ` prefixed lines into a <blockquote> block.

    Inline formatting (bold, italic, etc.) inside the quote is converted
    before wrapping in the blockquote tag so it renders correctly.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    quote_buf: list[str] = []

    def _flush_quote() -> None:
        if quote_buf:
            inner = _render_html_blockquote_body(quote_buf)
            result_lines.append(ph(f"<blockquote>{inner}</blockquote>"))
            quote_buf.clear()

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("> "):
            quote_buf.append(stripped[2:])
        elif stripped == ">":
            quote_buf.append("")
        else:
            _flush_quote()
            result_lines.append(line)

    _flush_quote()
    return "\n".join(result_lines)


def _html_code_blocks(text: str, ph: Callable[[str], str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        lang = m.group(1) or ""
        code = _html_mod.escape(m.group(2))
        if lang:
            return ph(f'<pre language="{_html_mod.escape(lang)}">{code}</pre>')
        return ph(f"<pre>{code}</pre>")

    return re.sub(r"```(\w*)\n(.*?)```", _repl, text, flags=re.DOTALL)


def _html_inline_code(text: str, ph: Callable[[str], str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        return ph(f"<code>{_html_mod.escape(m.group(1))}</code>")

    return re.sub(r"`([^`]+)`", _repl, text)


def _html_links(text: str, ph: Callable[[str], str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        link_text = _html_mod.escape(m.group(1))
        url = _html_mod.escape(m.group(2))
        return ph(f'<a href="{url}">{link_text}</a>')

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


def _html_bold(text: str, ph: Callable[[str], str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        return ph(f"<b>{m.group(1)}</b>")

    return re.sub(r"\*\*(.+?)\*\*", _repl, text, flags=re.DOTALL)


def _html_italic(text: str, ph: Callable[[str], str]) -> str:
    def _repl_star(m: re.Match[str]) -> str:
        return ph(f"<i>{m.group(1)}</i>")

    def _repl_under(m: re.Match[str]) -> str:
        return ph(f"<i>{m.group(1)}</i>")

    result = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", _repl_star, text)
    return re.sub(r"(?<!\\)_([^_]+?)_", _repl_under, result)


def _html_strikethrough(text: str, ph: Callable[[str], str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        return ph(f"<s>{m.group(1)}</s>")

    return re.sub(r"~~(.+?)~~", _repl, text, flags=re.DOTALL)
