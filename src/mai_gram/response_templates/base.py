"""Base abstractions for the response template plugin system.

Response templates constrain LLM output into structured formats (XML, JSON,
markdown headers, etc.) with named fields. Each template defines how to
instruct the model, parse raw output, validate compliance, and render
individual fields for Telegram display.

Templates support user-configurable parameters (e.g. field names, counts)
declared via ``get_params()``.  A parameterized copy is obtained through
``with_params()``, which returns a new instance whose instructions, examples,
parsing, and validation all reflect the chosen values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """Describes one named field in a structured response template."""

    name: str
    required: bool = True
    display_label: str = ""
    display_tag: str = "blockquote"
    expandable: bool = False
    user_can_hide: bool = False
    order: int = 0

    @property
    def label(self) -> str:
        return self.display_label or self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """Result of parsing a raw LLM response through a template."""

    fields: dict[str, str]
    raw: str


@dataclass(frozen=True, slots=True)
class TemplateExample:
    """A positive or negative example of the expected response format."""

    text: str
    is_positive: bool


@dataclass(frozen=True, slots=True)
class TemplateParam:
    """Declares a single user-configurable template parameter."""

    key: str
    label: str
    param_type: str  # "int" | "str"
    default: Any
    description: str = ""
    min_value: int | None = None
    max_value: int | None = None
    suggestions: list[str] = field(default_factory=list)


class ResponseTemplate(ABC):
    """Abstract base for response format templates.

    Subclasses define how the LLM should structure its output, how to
    parse that structure, and how to render each field for Telegram.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in config and DB (e.g. 'xml', 'json')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown during template selection."""

    @abstractmethod
    def get_fields(self) -> list[FieldDescriptor]:
        """Return ordered field definitions for this template."""

    @abstractmethod
    def format_instruction(self) -> str:
        """Return text appended to the system prompt describing the format.

        Return empty string for templates that impose no format constraints.
        """

    @abstractmethod
    def parse(self, raw_text: str) -> ParsedResponse:
        """Extract named fields from the raw LLM output.

        Should be lenient with whitespace and minor formatting issues.
        """

    @abstractmethod
    def validate(self, parsed: ParsedResponse) -> list[str]:
        """Return a list of error descriptions (empty list = valid)."""

    def examples(self) -> list[TemplateExample]:
        """Return format examples for the system prompt.

        Override to provide positive/negative examples that help the model
        understand the expected format.
        """
        return []

    def get_params(self) -> list[TemplateParam]:
        """Declare user-configurable parameters for this template.

        Override to expose parameters that users can customize when creating
        a chat. The base implementation returns an empty list.
        """
        return []

    def with_params(self, params: dict[str, Any]) -> ResponseTemplate:
        """Return a new template instance configured with *params*.

        Only keys declared in ``get_params()`` are accepted; unknown keys
        are silently ignored. Values are coerced and clamped to declared
        constraints. The base implementation returns ``self`` unchanged
        when no params are declared.
        """
        declared = {p.key: p for p in self.get_params()}
        if not declared:
            return self
        resolved = self._resolve_params(params, declared)
        return self._build_with_params(resolved)

    def get_effective_params(self) -> dict[str, Any]:
        """Return the current effective parameter values.

        For a default (non-parameterized) instance this returns the declared
        defaults. For a parameterized instance it returns the configured values.
        """
        return {p.key: p.default for p in self.get_params()}

    def _build_with_params(self, params: dict[str, Any]) -> ResponseTemplate:
        """Subclass hook: construct a new instance with resolved *params*.

        Only called when the template has declared parameters.
        """
        return self

    @staticmethod
    def _resolve_params(
        raw: dict[str, Any],
        declared: dict[str, TemplateParam],
    ) -> dict[str, Any]:
        """Coerce, validate, and fill defaults for raw user params."""
        result: dict[str, Any] = {}
        for key, spec in declared.items():
            value = raw.get(key, spec.default)
            if spec.param_type == "int":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    value = spec.default
                if spec.min_value is not None:
                    value = max(value, spec.min_value)
                if spec.max_value is not None:
                    value = min(value, spec.max_value)
            elif spec.param_type == "str":
                value = str(value).strip() if value else spec.default
                if not value:
                    value = spec.default
            result[key] = value
        return result

    def render_field_html(
        self,
        field_name: str,
        content: str,
        *,
        expandable: bool = False,
    ) -> str:
        """Render one field as Telegram HTML for display.

        The default implementation wraps content in a blockquote with a label.
        Subclasses can override for custom rendering per field.
        """
        from mai_gram.core.md_to_telegram import markdown_to_html

        descriptor = self._field_by_name(field_name)
        if descriptor is None:
            return markdown_to_html(content)

        inner_html = markdown_to_html(content.strip())
        if descriptor.display_tag == "none":
            return inner_html

        tag = "blockquote expandable" if expandable else "blockquote"
        close_tag = tag.split()[0]
        return f"<{tag}>{descriptor.label}\n{inner_html}</{close_tag}>"

    def assistant_prefill(self) -> str | None:
        """Return text for an assistant prefill message, or ``None`` to skip.

        When a non-``None`` value is returned, the prompt builder appends an
        assistant message with this content at the very end of the context.
        This "primes" the model to continue from the given prefix, which
        helps enforce structured output from the first token and can disable
        native reasoning on some providers (e.g. Google models).
        """
        return None

    def content_field_name(self) -> str:
        """Return the name of the primary content field (rendered as main body)."""
        return "content"

    def _field_by_name(self, name: str) -> FieldDescriptor | None:
        for f in self.get_fields():
            if f.name == name:
                return f
        return None
