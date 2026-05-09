"""Auto-discovery registry for response template plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mai_gram.response_templates.base import ResponseTemplate, TemplateGroup

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ResponseTemplate] = {}
_GROUPS: dict[str, TemplateGroup] = {}
_DISCOVERED = False


def _discover() -> None:
    """Import all built-in template modules to trigger registration."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    from mai_gram.response_templates.base import TemplateGroup

    register_group(
        TemplateGroup(
            id="xml",
            label="XML-based",
            description="Structured XML tags",
            order=1,
        )
    )
    register_group(
        TemplateGroup(
            id="json",
            label="JSON",
            description="JSON object format",
            order=2,
        )
    )
    register_group(
        TemplateGroup(
            id="markdown",
            label="Markdown",
            description="Markdown headers format",
            order=3,
        )
    )

    from mai_gram.response_templates import (  # noqa: F401
        empty,
        gemma_reasoning_prefill,
        gemma_reasoning_template,
        json_prefill,
        json_template,
        markdown_headers_prefill,
        markdown_headers_template,
        xml_emotions_prefill,
        xml_prefill,
        xml_template,
        xml_with_emotions,
    )

    _DISCOVERED = True


def register_group(group: TemplateGroup) -> None:
    """Register a template group."""
    _GROUPS[group.id] = group


def register_template(template: ResponseTemplate) -> None:
    """Register a template instance in the global registry."""
    if template.name in _REGISTRY:
        logger.warning("Overwriting template registration: %s", template.name)
    _REGISTRY[template.name] = template


def get_template(
    name: str | None,
    params: dict[str, Any] | None = None,
) -> ResponseTemplate:
    """Look up a template by name and optionally apply user params.

    Returns EmptyTemplate for None/unknown names.  When *params* is
    provided (and not empty), calls ``template.with_params(params)``
    to produce a configured instance.
    """
    _discover()
    base = _REGISTRY["empty"] if name is None or name not in _REGISTRY else _REGISTRY[name]
    if params:
        return base.with_params(params)
    return base


def list_template_names() -> list[str]:
    """Return all registered template names in sorted order."""
    _discover()
    return sorted(_REGISTRY.keys())


def list_groups() -> list[TemplateGroup]:
    """Return all registered groups sorted by order then id."""
    _discover()
    return sorted(_GROUPS.values(), key=lambda g: (g.order, g.id))


def get_templates_in_group(group_id: str) -> list[ResponseTemplate]:
    """Return templates belonging to a given group, sorted by name."""
    _discover()
    return sorted(
        (t for t in _REGISTRY.values() if t.group == group_id),
        key=lambda t: t.name,
    )
