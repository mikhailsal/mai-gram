"""Output parsing helpers for the console CLI."""

from __future__ import annotations

import re

_BUTTON_RE = re.compile(r"^\[\d+\]\s+.+?\s+->\s+(?P<callback>.+)$", re.MULTILINE)
_RESPONSE_RE = re.compile(
    r"^--- AI Response(?: \(final edit of [^)]+\))? ---\n"
    r"(?:\[parse_mode=.*\]\n)?"
    r"(?P<body>.*?)(?=^--- |\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_button_callbacks(output: str) -> list[str]:
    return [match.group("callback").strip() for match in _BUTTON_RE.finditer(output)]


def find_callback(output: str, prefix: str) -> str:
    for callback in extract_button_callbacks(output):
        if callback.startswith(prefix):
            return callback
    raise AssertionError(f"No callback starting with {prefix!r} found in output:\n{output}")


def extract_response_bodies(output: str) -> list[str]:
    return [match.group("body").strip() for match in _RESPONSE_RE.finditer(output)]


def extract_last_response_body(output: str) -> str:
    bodies = extract_response_bodies(output)
    if not bodies:
        raise AssertionError(f"No AI response section found in output:\n{output}")
    return bodies[-1]
