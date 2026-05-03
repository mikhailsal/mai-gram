"""Response template plugin system for structured LLM output."""

from mai_gram.response_templates.base import FieldDescriptor as FieldDescriptor
from mai_gram.response_templates.base import ParsedResponse as ParsedResponse
from mai_gram.response_templates.base import ResponseTemplate as ResponseTemplate
from mai_gram.response_templates.base import TemplateExample as TemplateExample
from mai_gram.response_templates.base import TemplateParam as TemplateParam
from mai_gram.response_templates.registry import get_template as get_template
from mai_gram.response_templates.registry import list_template_names as list_template_names
from mai_gram.response_templates.registry import register_template as register_template

__all__ = [
    "FieldDescriptor",
    "ParsedResponse",
    "ResponseTemplate",
    "TemplateExample",
    "TemplateParam",
    "get_template",
    "list_template_names",
    "register_template",
]
