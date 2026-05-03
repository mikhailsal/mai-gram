"""Base abstractions for the response template plugin system.

Response templates constrain LLM output into structured formats (XML, JSON,
markdown headers, etc.) with named fields. Each template defines how to
instruct the model, parse raw output, validate compliance, and render
individual fields for Telegram display.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def content_field_name(self) -> str:
        """Return the name of the primary content field (rendered as main body)."""
        return "content"

    def _field_by_name(self, name: str) -> FieldDescriptor | None:
        for f in self.get_fields():
            if f.name == name:
                return f
        return None
