"""Auto-discovery registry for response template plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mai_gram.response_templates.base import ResponseTemplate

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ResponseTemplate] = {}
_DISCOVERED = False


def _discover() -> None:
    """Import all built-in template modules to trigger registration."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    from mai_gram.response_templates import (  # noqa: F401
        empty,
        json_template,
        markdown_headers_template,
        xml_template,
        xml_with_emotions,
    )

    _DISCOVERED = True


def register_template(template: ResponseTemplate) -> None:
    """Register a template instance in the global registry."""
    if template.name in _REGISTRY:
        logger.warning("Overwriting template registration: %s", template.name)
    _REGISTRY[template.name] = template


def get_template(name: str | None) -> ResponseTemplate:
    """Look up a template by name. Returns EmptyTemplate for None/unknown."""
    _discover()
    if name is None or name not in _REGISTRY:
        return _REGISTRY["empty"]
    return _REGISTRY[name]


def list_template_names() -> list[str]:
    """Return all registered template names in sorted order."""
    _discover()
    return sorted(_REGISTRY.keys())
