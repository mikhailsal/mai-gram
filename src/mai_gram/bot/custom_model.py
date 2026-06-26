"""Arbitrary ("custom") model selection support.

Privileged users (listed in ``custom_model_allowed_users`` in ``bots.toml``)
may type an arbitrary OpenRouter model id that is not present in
``config/models.toml`` -- optionally followed by request-body parameter
overrides (``reasoning.effort``, ``temperature``, ``provider.order``, ...).

The free-form text input mirrors the existing template-parameter UX: the
first non-empty line is the model id, and each subsequent ``key = value``
line becomes an OpenRouter request parameter. Dotted keys expand into nested
objects (``reasoning.effort = high`` -> ``{"reasoning": {"effort": "high"}}``)
and scalar values are coerced to their natural JSON types.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gram.config_loaders import BotConfig

CUSTOM_MODEL_VALUE = "__custom_model__"
CUSTOM_MODEL_LABEL = "Custom model (type your own)"
INVALID_MODEL_MESSAGE = (
    "That doesn't look like a valid model id. Try again, e.g. openai/gpt-5.4-mini"
)
CUSTOM_MODELS_DISABLED_MESSAGE = "Custom models are not available for this bot."

# A model id looks like ``vendor/model`` or ``vendor/model:tag``; keep the
# validation permissive but reject whitespace and obviously bogus input.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/@\-]{2,150}$")

CUSTOM_MODEL_PROMPT = (
    "Type an arbitrary model id, optionally followed by parameters "
    "(one `key = value` per line).\n\n"
    "Example:\n"
    "openai/gpt-5.4-mini\n"
    'reasoning.effort = "high"\n'
    "temperature = 0.7\n\n"
    "The first line is the model id; dotted keys (e.g. reasoning.effort) "
    "become nested OpenRouter parameters."
)


def is_user_allowed(bot_config: BotConfig | None, user_id: str) -> bool:
    """Return whether *user_id* may use arbitrary models on this bot."""
    if bot_config is None or not bot_config.custom_model_allowed_users:
        return False
    allowed = {str(uid) for uid in bot_config.custom_model_allowed_users}
    return str(user_id) in allowed


def validate_model_name(name: str) -> bool:
    """Return whether *name* is a plausible OpenRouter model id."""
    return bool(_MODEL_ID_RE.match(name.strip()))


def _coerce_value(raw: str) -> Any:
    """Coerce a raw string param value into a natural JSON/Python type."""
    value = raw.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign *value* into *target* following a dotted key path."""
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def parse_model_params(text: str) -> dict[str, Any]:
    """Parse ``key = value`` lines into a nested, type-coerced params dict."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        _assign_nested(result, key, _coerce_value(raw_value))
    return result


def parse_custom_model_input(text: str) -> tuple[str, dict[str, Any]]:
    """Split free-form input into (model_id, params).

    The first non-empty, non-comment line is the model id; remaining lines are
    parsed as parameters.
    """
    lines = text.strip().splitlines()
    model = ""
    params_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        model = stripped
        params_start = index + 1
        break
    params = parse_model_params("\n".join(lines[params_start:]))
    return model, params


def merge_extra_params(base: dict[str, Any], custom: dict[str, Any] | None) -> dict[str, Any]:
    """Merge *custom* params on top of *base* (custom wins at the top level)."""
    if not custom:
        return base
    merged = dict(base)
    merged.update(custom)
    return merged


def load_custom_params(raw_json: str | None) -> dict[str, Any] | None:
    """Decode a stored ``custom_model_params`` JSON string, tolerating bad data."""
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) and data else None


def format_params_summary(params: dict[str, Any] | None) -> str:
    """Render params for confirmation messages (compact JSON, or a dash)."""
    if not params:
        return "(none)"
    return json.dumps(params, ensure_ascii=False, sort_keys=True)
