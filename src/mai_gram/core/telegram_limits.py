"""Telegram message length limits and splitting utilities."""

from __future__ import annotations

TELEGRAM_MAX_LENGTH = 4096
SAFE_MAX_LENGTH = 4000
MAX_CONTENT_LENGTH_FOR_TRUNCATION = 3800


def split_html_safe(text: str, max_len: int = SAFE_MAX_LENGTH) -> list[str]:
    """Split text into chunks that fit within Telegram's message limit.

    Tries to split at paragraph boundaries (double newline), then single
    newlines, then spaces, falling back to hard cut as a last resort.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        cut_at = -1
        for sep in ("\n\n", "\n", " "):
            idx = remaining.rfind(sep, 0, max_len)
            if idx > max_len // 4:
                cut_at = idx + len(sep)
                break

        if cut_at <= 0:
            cut_at = max_len

        chunks.append(remaining[:cut_at])
        remaining = remaining[cut_at:]

    return chunks
