"""Unit tests for the response template plugin system.

Covers both default behavior (backward compatibility), user-configurable
template parameters, and assistant prefill support.
"""

from __future__ import annotations

from typing import ClassVar

from mai_gram.response_templates.base import FieldDescriptor, ResponseTemplate, TemplateParam
from mai_gram.response_templates.registry import get_template, list_template_names

PREFILL_TEMPLATES: list[str] = [
    "xml_prefill",
    "json_prefill",
    "markdown_headers_prefill",
    "xml_emotions_prefill",
    "gemma_reasoning_prefill",
]

PARENT_OF: dict[str, str] = {
    "xml_prefill": "xml",
    "json_prefill": "json",
    "markdown_headers_prefill": "markdown_headers",
    "xml_emotions_prefill": "xml_emotions",
    "gemma_reasoning_prefill": "gemma_reasoning",
}

# ──────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_list_names_returns_all_builtin(self) -> None:
        names = list_template_names()
        assert "empty" in names
        assert "xml" in names
        assert "json" in names
        assert "markdown_headers" in names
        assert "xml_emotions" in names
        assert "gemma_reasoning" in names

    def test_list_names_returns_all_prefill(self) -> None:
        names = list_template_names()
        for name in PREFILL_TEMPLATES:
            assert name in names, f"Missing prefill template: {name}"

    def test_get_template_returns_empty_for_none(self) -> None:
        t = get_template(None)
        assert t.name == "empty"

    def test_get_template_returns_empty_for_unknown(self) -> None:
        t = get_template("nonexistent_template")
        assert t.name == "empty"

    def test_get_template_by_name(self) -> None:
        for name in ["empty", "xml", "json", "markdown_headers", "xml_emotions"]:
            t = get_template(name)
            assert t.name == name

    def test_get_prefill_template_by_name(self) -> None:
        for name in PREFILL_TEMPLATES:
            t = get_template(name)
            assert t.name == name

    def test_get_template_with_params_returns_parameterized_instance(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        fields = t.get_fields()
        assert fields[0].name == "think"

    def test_get_template_with_empty_params_returns_default(self) -> None:
        t = get_template("xml", {})
        fields = t.get_fields()
        assert fields[0].name == "thought"

    def test_get_template_with_none_params_returns_default(self) -> None:
        t = get_template("xml", None)
        fields = t.get_fields()
        assert fields[0].name == "thought"


# ──────────────────────────────────────────────────────────────────
# Empty template (no params, backward compat)
# ──────────────────────────────────────────────────────────────────


class TestEmptyTemplate:
    def test_parse_wraps_raw_text(self) -> None:
        t = get_template("empty")
        parsed = t.parse("Hello world")
        assert parsed.fields == {"content": "Hello world"}
        assert parsed.raw == "Hello world"

    def test_validate_never_fails(self) -> None:
        t = get_template("empty")
        parsed = t.parse("")
        assert t.validate(parsed) == []

    def test_format_instruction_is_empty(self) -> None:
        t = get_template("empty")
        assert t.format_instruction() == ""

    def test_examples_is_empty(self) -> None:
        t = get_template("empty")
        assert t.examples() == []

    def test_fields_has_content(self) -> None:
        t = get_template("empty")
        fields = t.get_fields()
        assert len(fields) == 1
        assert fields[0].name == "content"

    def test_render_field_html(self) -> None:
        t = get_template("empty")
        html = t.render_field_html("content", "**bold**")
        assert html

    def test_content_field_name(self) -> None:
        t = get_template("empty")
        assert t.content_field_name() == "content"

    def test_description(self) -> None:
        t = get_template("empty")
        assert t.description

    def test_assistant_prefill_is_none(self) -> None:
        t = get_template("empty")
        assert t.assistant_prefill() is None

    def test_has_no_params(self) -> None:
        t = get_template("empty")
        assert t.get_params() == []

    def test_with_params_returns_self(self) -> None:
        t = get_template("empty")
        t2 = t.with_params({"anything": "ignored"})
        assert t2 is t


# ──────────────────────────────────────────────────────────────────
# XML template -- default behavior
# ──────────────────────────────────────────────────────────────────


class TestXmlTemplate:
    def test_parse_extracts_both_tags(self) -> None:
        t = get_template("xml")
        raw = "<thought>thinking</thought><content>reply</content>"
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "thinking"
        assert parsed.fields["content"] == "reply"

    def test_parse_handles_multiline(self) -> None:
        t = get_template("xml")
        raw = "<thought>\nline1\nline2\n</thought>\n<content>reply</content>"
        parsed = t.parse(raw)
        assert "line1" in parsed.fields["thought"]
        assert "line2" in parsed.fields["thought"]

    def test_validate_passes_complete(self) -> None:
        t = get_template("xml")
        parsed = t.parse("<thought>t</thought><content>c</content>")
        assert t.validate(parsed) == []

    def test_validate_fails_missing_thought(self) -> None:
        t = get_template("xml")
        parsed = t.parse("<content>c</content>")
        errors = t.validate(parsed)
        assert any("thought" in e.lower() for e in errors)

    def test_validate_fails_missing_content(self) -> None:
        t = get_template("xml")
        parsed = t.parse("<thought>t</thought>")
        errors = t.validate(parsed)
        assert any("content" in e.lower() for e in errors)

    def test_validate_fails_empty_tag(self) -> None:
        t = get_template("xml")
        parsed = t.parse("<thought>  </thought><content>c</content>")
        errors = t.validate(parsed)
        assert any("empty" in e.lower() for e in errors)

    def test_format_instruction_mentions_tags(self) -> None:
        t = get_template("xml")
        instruction = t.format_instruction()
        assert "<thought>" in instruction
        assert "<content>" in instruction
        assert "RESPONSE FORMAT" in instruction

    def test_examples_has_positive_and_negative(self) -> None:
        t = get_template("xml")
        examples = t.examples()
        assert any(ex.is_positive for ex in examples)
        assert any(not ex.is_positive for ex in examples)

    def test_fields_order(self) -> None:
        t = get_template("xml")
        fields = t.get_fields()
        assert fields[0].name == "thought"
        assert fields[1].name == "content"

    def test_thought_is_hideable(self) -> None:
        t = get_template("xml")
        fields = t.get_fields()
        thought = next(f for f in fields if f.name == "thought")
        assert thought.user_can_hide is True
        assert thought.expandable is True

    def test_render_thought_html(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("thought", "some reasoning", expandable=True)
        assert "blockquote" in html
        assert "Thought" in html

    def test_render_content_html(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("content", "hello world")
        assert "blockquote" not in html

    def test_assistant_prefill_is_none(self) -> None:
        t = get_template("xml")
        assert t.assistant_prefill() is None

    def test_description(self) -> None:
        t = get_template("xml")
        assert t.description


# ──────────────────────────────────────────────────────────────────
# XML template -- parameterized
# ──────────────────────────────────────────────────────────────────


class TestXmlTemplateParams:
    def test_declares_reasoning_field_param(self) -> None:
        t = get_template("xml")
        param_keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in param_keys
        assert "num_reasoning_paragraphs" in param_keys

    def test_default_effective_params(self) -> None:
        t = get_template("xml")
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "thought"
        assert ep["num_reasoning_paragraphs"] == 2

    def test_custom_reasoning_field_changes_fields(self) -> None:
        t = get_template("xml", {"reasoning_field": "scratchpad"})
        fields = t.get_fields()
        assert fields[0].name == "scratchpad"
        assert fields[1].name == "content"

    def test_custom_reasoning_field_changes_instruction(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        instruction = t.format_instruction()
        assert "<think>" in instruction
        assert "</think>" in instruction
        assert "<thought>" not in instruction

    def test_custom_reasoning_field_changes_examples(self) -> None:
        t = get_template("xml", {"reasoning_field": "scratchpad"})
        examples = t.examples()
        pos = next(ex for ex in examples if ex.is_positive)
        assert "<scratchpad>" in pos.text
        assert "</scratchpad>" in pos.text
        assert "<thought>" not in pos.text

    def test_custom_reasoning_field_changes_parsing(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        raw = "<think>reasoning</think><content>answer</content>"
        parsed = t.parse(raw)
        assert parsed.fields["think"] == "reasoning"
        assert parsed.fields["content"] == "answer"

    def test_custom_reasoning_field_changes_validation(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        parsed = t.parse("<content>c</content>")
        errors = t.validate(parsed)
        assert any("think" in e.lower() for e in errors)

    def test_custom_reasoning_field_changes_description(self) -> None:
        t = get_template("xml", {"reasoning_field": "reflect"})
        assert "reflect" in t.description

    def test_num_reasoning_paragraphs_in_instruction(self) -> None:
        t = get_template("xml", {"num_reasoning_paragraphs": 5})
        instruction = t.format_instruction()
        assert "5 paragraphs" in instruction

    def test_num_reasoning_paragraphs_singular(self) -> None:
        t = get_template("xml", {"num_reasoning_paragraphs": 1})
        instruction = t.format_instruction()
        assert "1 paragraph)" in instruction
        assert "paragraphs" not in instruction.split("1 paragraph")[0]

    def test_example_has_correct_paragraph_count(self) -> None:
        for n in (1, 2, 3, 5, 8):
            t = get_template("xml", {"num_reasoning_paragraphs": n})
            pos = next(ex for ex in t.examples() if ex.is_positive)
            tag_name = "thought"
            start_tag = f"<{tag_name}>\n"
            end_tag = f"\n</{tag_name}>"
            assert start_tag in pos.text
            inner = pos.text.split(start_tag, 1)[1].split(end_tag, 0)[0]
            inner = inner.split(f"\n</{tag_name}>")[0]
            paragraphs = [p.strip() for p in inner.split("\n\n") if p.strip()]
            assert len(paragraphs) == n, f"Expected {n} paragraphs, got {len(paragraphs)}"

    def test_param_clamping_min(self) -> None:
        t = get_template("xml", {"num_reasoning_paragraphs": -5})
        ep = t.get_effective_params()
        assert ep["num_reasoning_paragraphs"] >= 1

    def test_param_clamping_max(self) -> None:
        t = get_template("xml", {"num_reasoning_paragraphs": 100})
        ep = t.get_effective_params()
        assert ep["num_reasoning_paragraphs"] <= 8

    def test_invalid_int_param_uses_default(self) -> None:
        t = get_template("xml", {"num_reasoning_paragraphs": "not_a_number"})
        ep = t.get_effective_params()
        assert ep["num_reasoning_paragraphs"] == 2

    def test_empty_string_param_uses_default(self) -> None:
        t = get_template("xml", {"reasoning_field": ""})
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "thought"

    def test_unknown_params_ignored(self) -> None:
        t = get_template("xml", {"nonexistent_param": "value"})
        assert t.get_fields()[0].name == "thought"

    def test_with_params_returns_new_instance(self) -> None:
        t1 = get_template("xml")
        t2 = t1.with_params({"reasoning_field": "think"})
        assert t1.get_fields()[0].name == "thought"
        assert t2.get_fields()[0].name == "think"

    def test_effective_params_after_with_params(self) -> None:
        t = get_template("xml", {"reasoning_field": "reflect", "num_reasoning_paragraphs": 4})
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "reflect"
        assert ep["num_reasoning_paragraphs"] == 4

    def test_render_html_with_custom_field(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        html = t.render_field_html("think", "some text", expandable=True)
        assert "blockquote" in html
        assert "Think" in html

    def test_param_suggestions_provided(self) -> None:
        t = get_template("xml")
        rf_param = next(p for p in t.get_params() if p.key == "reasoning_field")
        assert len(rf_param.suggestions) > 0
        assert "thought" in rf_param.suggestions
        assert "scratchpad" in rf_param.suggestions


# ──────────────────────────────────────────────────────────────────
# JSON template -- default + parameterized
# ──────────────────────────────────────────────────────────────────


class TestJsonTemplate:
    def test_parse_extracts_json(self) -> None:
        t = get_template("json")
        raw = '{"thought": "thinking", "content": "reply"}'
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "thinking"
        assert parsed.fields["content"] == "reply"

    def test_parse_handles_fenced_block(self) -> None:
        t = get_template("json")
        raw = '```json\n{"thought": "t", "content": "c"}\n```'
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "t"

    def test_parse_returns_empty_on_invalid_json(self) -> None:
        t = get_template("json")
        parsed = t.parse("not json at all")
        assert parsed.fields == {}

    def test_validate_passes_complete(self) -> None:
        t = get_template("json")
        parsed = t.parse('{"thought": "t", "content": "c"}')
        assert t.validate(parsed) == []

    def test_validate_fails_missing_key(self) -> None:
        t = get_template("json")
        parsed = t.parse('{"content": "c"}')
        errors = t.validate(parsed)
        assert any("thought" in e.lower() for e in errors)

    def test_validate_fails_empty_value(self) -> None:
        t = get_template("json")
        parsed = t.parse('{"thought": "  ", "content": "c"}')
        errors = t.validate(parsed)
        assert any("empty" in e.lower() for e in errors)

    def test_validate_fails_on_unparseable(self) -> None:
        t = get_template("json")
        parsed = t.parse("plain text")
        errors = t.validate(parsed)
        assert len(errors) > 0

    def test_format_instruction(self) -> None:
        t = get_template("json")
        instruction = t.format_instruction()
        assert "JSON" in instruction

    def test_examples(self) -> None:
        t = get_template("json")
        assert len(t.examples()) > 0

    def test_description(self) -> None:
        t = get_template("json")
        assert t.description

    def test_fields(self) -> None:
        t = get_template("json")
        fields = t.get_fields()
        assert len(fields) == 2


class TestJsonTemplateParams:
    def test_declares_params(self) -> None:
        t = get_template("json")
        keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in keys
        assert "num_reasoning_paragraphs" in keys

    def test_custom_reasoning_field_changes_fields(self) -> None:
        t = get_template("json", {"reasoning_field": "think"})
        fields = t.get_fields()
        assert fields[0].name == "think"

    def test_custom_reasoning_field_in_instruction(self) -> None:
        t = get_template("json", {"reasoning_field": "scratchpad"})
        instruction = t.format_instruction()
        assert '"scratchpad"' in instruction
        assert '"thought"' not in instruction

    def test_custom_reasoning_field_in_examples(self) -> None:
        t = get_template("json", {"reasoning_field": "reflect"})
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert '"reflect"' in pos.text
        assert '"thought"' not in pos.text

    def test_custom_reasoning_field_in_validation(self) -> None:
        t = get_template("json", {"reasoning_field": "think"})
        parsed = t.parse('{"content": "c"}')
        errors = t.validate(parsed)
        assert any("think" in e.lower() for e in errors)

    def test_custom_reasoning_field_in_parse(self) -> None:
        t = get_template("json", {"reasoning_field": "think"})
        parsed = t.parse('{"think": "r", "content": "c"}')
        assert t.validate(parsed) == []
        assert parsed.fields["think"] == "r"

    def test_num_paragraphs_in_instruction(self) -> None:
        t = get_template("json", {"num_reasoning_paragraphs": 4})
        instruction = t.format_instruction()
        assert "4 paragraphs" in instruction

    def test_example_reasoning_reflects_paragraph_count(self) -> None:
        t = get_template("json", {"num_reasoning_paragraphs": 3})
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "\\n\\n" in pos.text

    def test_description_includes_custom_field(self) -> None:
        t = get_template("json", {"reasoning_field": "reflect"})
        assert "reflect" in t.description


# ──────────────────────────────────────────────────────────────────
# Markdown headers template -- default + parameterized
# ──────────────────────────────────────────────────────────────────


class TestMarkdownHeadersTemplate:
    def test_parse_extracts_sections(self) -> None:
        t = get_template("markdown_headers")
        raw = "## Thought\nthinking here\n\n## Content\nreply here"
        parsed = t.parse(raw)
        assert parsed.fields["Thought"] == "thinking here"
        assert parsed.fields["Content"] == "reply here"

    def test_validate_passes_complete(self) -> None:
        t = get_template("markdown_headers")
        parsed = t.parse("## Thought\nt\n\n## Content\nc")
        assert t.validate(parsed) == []

    def test_validate_fails_missing_section(self) -> None:
        t = get_template("markdown_headers")
        parsed = t.parse("## Content\nc")
        errors = t.validate(parsed)
        assert any("Thought" in e for e in errors)

    def test_validate_fails_empty_section(self) -> None:
        t = get_template("markdown_headers")
        parsed = t.parse("## Thought\n  \n\n## Content\nc")
        errors = t.validate(parsed)
        assert any("empty" in e.lower() for e in errors)

    def test_format_instruction(self) -> None:
        t = get_template("markdown_headers")
        instruction = t.format_instruction()
        assert "## Thought" in instruction

    def test_content_field_name(self) -> None:
        t = get_template("markdown_headers")
        assert t.content_field_name() == "Content"

    def test_examples(self) -> None:
        t = get_template("markdown_headers")
        assert len(t.examples()) > 0

    def test_description(self) -> None:
        t = get_template("markdown_headers")
        assert t.description

    def test_fields(self) -> None:
        t = get_template("markdown_headers")
        fields = t.get_fields()
        assert len(fields) == 2


class TestMarkdownHeadersTemplateParams:
    def test_declares_params(self) -> None:
        t = get_template("markdown_headers")
        keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in keys
        assert "num_reasoning_paragraphs" in keys

    def test_custom_reasoning_field_changes_fields(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Reflection"})
        fields = t.get_fields()
        assert fields[0].name == "Reflection"

    def test_custom_reasoning_field_in_instruction(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Think"})
        instruction = t.format_instruction()
        assert "## Think" in instruction
        assert "## Thought" not in instruction

    def test_custom_reasoning_field_in_examples(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Scratchpad"})
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "## Scratchpad" in pos.text
        assert "## Thought" not in pos.text

    def test_custom_reasoning_field_in_parse(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Think"})
        raw = "## Think\nreasoning here\n\n## Content\nreply"
        parsed = t.parse(raw)
        assert parsed.fields["Think"] == "reasoning here"
        assert parsed.fields["Content"] == "reply"

    def test_custom_reasoning_field_in_validation(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Think"})
        parsed = t.parse("## Content\nc")
        errors = t.validate(parsed)
        assert any("Think" in e for e in errors)

    def test_num_paragraphs_in_instruction(self) -> None:
        t = get_template("markdown_headers", {"num_reasoning_paragraphs": 3})
        instruction = t.format_instruction()
        assert "3 paragraphs" in instruction

    def test_example_paragraphs_match_count(self) -> None:
        for n in (1, 3, 5):
            t = get_template("markdown_headers", {"num_reasoning_paragraphs": n})
            pos = next(ex for ex in t.examples() if ex.is_positive)
            header = "## Thought\n"
            content_header = "\n\n## Content"
            reasoning_section = pos.text.split(header, 1)[1].split(content_header, 1)[0]
            paragraphs = [p.strip() for p in reasoning_section.split("\n\n") if p.strip()]
            assert len(paragraphs) == n

    def test_description_includes_custom_field(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Reflection"})
        assert "Reflection" in t.description


# ──────────────────────────────────────────────────────────────────
# XML with emotions -- default + parameterized
# ──────────────────────────────────────────────────────────────────


class TestXmlWithEmotionsTemplate:
    def test_inherits_parsing(self) -> None:
        t = get_template("xml_emotions")
        raw = "<thought>t</thought><emotions>happy</emotions><content>c</content>"
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "t"
        assert parsed.fields["emotions"] == "happy"
        assert parsed.fields["content"] == "c"

    def test_validates_three_fields(self) -> None:
        t = get_template("xml_emotions")
        parsed = t.parse("<thought>t</thought><emotions>e</emotions><content>c</content>")
        assert t.validate(parsed) == []

    def test_validates_fails_missing_emotions(self) -> None:
        t = get_template("xml_emotions")
        parsed = t.parse("<thought>t</thought><content>c</content>")
        errors = t.validate(parsed)
        assert any("emotions" in e.lower() for e in errors)

    def test_fields_order(self) -> None:
        t = get_template("xml_emotions")
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["thought", "emotions", "content"]

    def test_format_instruction_mentions_emotions(self) -> None:
        t = get_template("xml_emotions")
        instruction = t.format_instruction()
        assert "<emotions>" in instruction

    def test_examples(self) -> None:
        t = get_template("xml_emotions")
        assert len(t.examples()) > 0

    def test_description(self) -> None:
        t = get_template("xml_emotions")
        assert t.description

    def test_emotions_is_hideable(self) -> None:
        t = get_template("xml_emotions")
        fields = t.get_fields()
        emotions = next(f for f in fields if f.name == "emotions")
        assert emotions.user_can_hide is True


class TestXmlWithEmotionsParams:
    def test_declares_all_four_params(self) -> None:
        t = get_template("xml_emotions")
        keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in keys
        assert "num_reasoning_paragraphs" in keys
        assert "emotions_field" in keys
        assert "num_emotions" in keys

    def test_default_effective_params(self) -> None:
        t = get_template("xml_emotions")
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "thought"
        assert ep["num_reasoning_paragraphs"] == 2
        assert ep["emotions_field"] == "emotions"
        assert ep["num_emotions"] == 3

    def test_custom_reasoning_field(self) -> None:
        t = get_template("xml_emotions", {"reasoning_field": "think"})
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["think", "emotions", "content"]

    def test_custom_emotions_field(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "feelings"})
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["thought", "feelings", "content"]

    def test_custom_emotions_field_in_instruction(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "mood"})
        instruction = t.format_instruction()
        assert "<mood>" in instruction
        assert "</mood>" in instruction
        assert "<emotions>" not in instruction

    def test_custom_emotions_field_in_examples(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "mood"})
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "<mood>" in pos.text
        assert "</mood>" in pos.text
        assert "<emotions>" not in pos.text

    def test_custom_emotions_field_in_parse(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "mood"})
        raw = "<thought>t</thought><mood>happy</mood><content>c</content>"
        parsed = t.parse(raw)
        assert parsed.fields["mood"] == "happy"
        assert t.validate(parsed) == []

    def test_custom_emotions_field_in_validation(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "mood"})
        raw = "<thought>t</thought><emotions>happy</emotions><content>c</content>"
        parsed = t.parse(raw)
        errors = t.validate(parsed)
        assert any("mood" in e.lower() for e in errors)

    def test_num_emotions_in_instruction(self) -> None:
        t = get_template("xml_emotions", {"num_emotions": 8})
        instruction = t.format_instruction()
        assert "exactly 8 emotions" in instruction

    def test_num_emotions_singular(self) -> None:
        t = get_template("xml_emotions", {"num_emotions": 1})
        instruction = t.format_instruction()
        assert "exactly 1 emotion " in instruction

    def test_example_has_correct_emotion_count(self) -> None:
        for n in (1, 3, 5, 8, 12):
            t = get_template("xml_emotions", {"num_emotions": n})
            pos = next(ex for ex in t.examples() if ex.is_positive)
            emotions_start = "<emotions>"
            emotions_end = "</emotions>"
            inner = pos.text.split(emotions_start, 1)[1].split(emotions_end, 1)[0]
            items = [e.strip() for e in inner.split(",") if e.strip()]
            assert len(items) == n, f"Expected {n} emotions, got {len(items)}: {items}"

    def test_combined_custom_fields(self) -> None:
        """All four params customized together."""
        t = get_template(
            "xml_emotions",
            {
                "reasoning_field": "scratchpad",
                "num_reasoning_paragraphs": 4,
                "emotions_field": "mood",
                "num_emotions": 5,
            },
        )
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["scratchpad", "mood", "content"]

        instruction = t.format_instruction()
        assert "<scratchpad>" in instruction
        assert "<mood>" in instruction
        assert "4 paragraphs" in instruction
        assert "exactly 5 emotions" in instruction

        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "<scratchpad>" in pos.text
        assert "<mood>" in pos.text

    def test_num_emotions_clamping_min(self) -> None:
        t = get_template("xml_emotions", {"num_emotions": 0})
        ep = t.get_effective_params()
        assert ep["num_emotions"] >= 1

    def test_num_emotions_clamping_max(self) -> None:
        t = get_template("xml_emotions", {"num_emotions": 100})
        ep = t.get_effective_params()
        assert ep["num_emotions"] <= 12

    def test_description_includes_custom_fields(self) -> None:
        t = get_template(
            "xml_emotions",
            {
                "reasoning_field": "think",
                "emotions_field": "mood",
            },
        )
        assert "think" in t.description
        assert "mood" in t.description

    def test_emotions_field_suggestions_provided(self) -> None:
        t = get_template("xml_emotions")
        ef_param = next(p for p in t.get_params() if p.key == "emotions_field")
        assert len(ef_param.suggestions) > 0
        assert "emotions" in ef_param.suggestions

    def test_custom_emotions_hideable(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "vibes"})
        fields = t.get_fields()
        vibes = next(f for f in fields if f.name == "vibes")
        assert vibes.user_can_hide is True

    def test_custom_emotions_label(self) -> None:
        t = get_template("xml_emotions", {"emotions_field": "inner_mood"})
        fields = t.get_fields()
        field = next(f for f in fields if f.name == "inner_mood")
        assert "Inner Mood" in field.label


# ──────────────────────────────────────────────────────────────────
# TemplateParam and base class
# ──────────────────────────────────────────────────────────────────


class TestTemplateParam:
    def test_param_type_int(self) -> None:
        p = TemplateParam(key="k", label="L", param_type="int", default=5)
        assert p.param_type == "int"
        assert p.default == 5

    def test_param_type_str(self) -> None:
        p = TemplateParam(key="k", label="L", param_type="str", default="hello")
        assert p.param_type == "str"
        assert p.default == "hello"

    def test_param_min_max(self) -> None:
        p = TemplateParam(
            key="k",
            label="L",
            param_type="int",
            default=3,
            min_value=1,
            max_value=10,
        )
        assert p.min_value == 1
        assert p.max_value == 10

    def test_param_suggestions(self) -> None:
        p = TemplateParam(
            key="k",
            label="L",
            param_type="str",
            default="a",
            suggestions=["a", "b", "c"],
        )
        assert p.suggestions == ["a", "b", "c"]

    def test_param_description(self) -> None:
        p = TemplateParam(key="k", label="L", param_type="str", default="x", description="A test")
        assert p.description == "A test"


class TestResolveParams:
    """Test the static _resolve_params helper on ResponseTemplate."""

    def test_fills_defaults_for_missing_keys(self) -> None:
        declared = {
            "a": TemplateParam(key="a", label="A", param_type="str", default="default_a"),
            "b": TemplateParam(key="b", label="B", param_type="int", default=10),
        }
        result = ResponseTemplate._resolve_params({}, declared)
        assert result["a"] == "default_a"
        assert result["b"] == 10

    def test_coerces_int(self) -> None:
        declared = {
            "n": TemplateParam(
                key="n",
                label="N",
                param_type="int",
                default=5,
                min_value=1,
                max_value=20,
            ),
        }
        result = ResponseTemplate._resolve_params({"n": "7"}, declared)
        assert result["n"] == 7

    def test_clamps_int_min(self) -> None:
        declared = {
            "n": TemplateParam(
                key="n",
                label="N",
                param_type="int",
                default=5,
                min_value=1,
                max_value=20,
            ),
        }
        result = ResponseTemplate._resolve_params({"n": -10}, declared)
        assert result["n"] == 1

    def test_clamps_int_max(self) -> None:
        declared = {
            "n": TemplateParam(
                key="n",
                label="N",
                param_type="int",
                default=5,
                min_value=1,
                max_value=20,
            ),
        }
        result = ResponseTemplate._resolve_params({"n": 50}, declared)
        assert result["n"] == 20

    def test_invalid_int_falls_back_to_default(self) -> None:
        declared = {
            "n": TemplateParam(key="n", label="N", param_type="int", default=5),
        }
        result = ResponseTemplate._resolve_params({"n": "abc"}, declared)
        assert result["n"] == 5

    def test_empty_str_falls_back_to_default(self) -> None:
        declared = {
            "s": TemplateParam(key="s", label="S", param_type="str", default="fallback"),
        }
        result = ResponseTemplate._resolve_params({"s": "  "}, declared)
        assert result["s"] == "fallback"

    def test_strips_str(self) -> None:
        declared = {
            "s": TemplateParam(key="s", label="S", param_type="str", default="x"),
        }
        result = ResponseTemplate._resolve_params({"s": "  hello  "}, declared)
        assert result["s"] == "hello"


class TestFieldDescriptor:
    def test_label_defaults_to_title_case_name(self) -> None:
        fd = FieldDescriptor(name="my_field")
        assert fd.label == "My Field"

    def test_label_uses_display_label(self) -> None:
        fd = FieldDescriptor(name="my_field", display_label="Custom Label")
        assert fd.label == "Custom Label"


class TestBaseRenderFieldHtml:
    def test_render_unknown_field(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("nonexistent", "content")
        assert html

    def test_render_expandable(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("thought", "text", expandable=True)
        assert "expandable" in html


# ──────────────────────────────────────────────────────────────────
# XML prefill template
# ──────────────────────────────────────────────────────────────────


class TestXmlPrefillTemplate:
    def test_prefill_returns_opening_tag(self) -> None:
        t = get_template("xml_prefill")
        assert t.assistant_prefill() == "<thought>"

    def test_prefill_respects_custom_reasoning_field(self) -> None:
        t = get_template("xml_prefill", {"reasoning_field": "think"})
        assert t.assistant_prefill() == "<think>"

    def test_prefill_respects_custom_scratchpad(self) -> None:
        t = get_template("xml_prefill", {"reasoning_field": "scratchpad"})
        assert t.assistant_prefill() == "<scratchpad>"

    def test_inherits_parse_from_xml(self) -> None:
        t = get_template("xml_prefill")
        raw = "<thought>reasoning</thought><content>answer</content>"
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "reasoning"
        assert parsed.fields["content"] == "answer"

    def test_inherits_validate_from_xml(self) -> None:
        t = get_template("xml_prefill")
        parsed = t.parse("<thought>t</thought><content>c</content>")
        assert t.validate(parsed) == []

    def test_inherits_fields_from_xml(self) -> None:
        t = get_template("xml_prefill")
        fields = t.get_fields()
        assert fields[0].name == "thought"
        assert fields[1].name == "content"

    def test_inherits_format_instruction_from_xml(self) -> None:
        t = get_template("xml_prefill")
        instruction = t.format_instruction()
        assert "<thought>" in instruction
        assert "<content>" in instruction
        assert "RESPONSE FORMAT" in instruction

    def test_inherits_examples_from_xml(self) -> None:
        t = get_template("xml_prefill")
        examples = t.examples()
        assert any(ex.is_positive for ex in examples)
        assert any(not ex.is_positive for ex in examples)

    def test_name_is_xml_prefill(self) -> None:
        t = get_template("xml_prefill")
        assert t.name == "xml_prefill"

    def test_description_mentions_prefill(self) -> None:
        t = get_template("xml_prefill")
        assert "prefill" in t.description.lower()

    def test_with_params_returns_prefill_instance(self) -> None:
        t = get_template("xml_prefill", {"reasoning_field": "reflect"})
        assert t.name == "xml_prefill"
        assert t.assistant_prefill() == "<reflect>"

    def test_custom_field_changes_parse_and_validate(self) -> None:
        t = get_template("xml_prefill", {"reasoning_field": "think"})
        raw = "<think>reasoning</think><content>answer</content>"
        parsed = t.parse(raw)
        assert t.validate(parsed) == []
        assert parsed.fields["think"] == "reasoning"

    def test_description_includes_custom_field(self) -> None:
        t = get_template("xml_prefill", {"reasoning_field": "reflect"})
        assert "reflect" in t.description


# ──────────────────────────────────────────────────────────────────
# JSON prefill template
# ──────────────────────────────────────────────────────────────────


class TestJsonPrefillTemplate:
    def test_prefill_returns_json_start(self) -> None:
        t = get_template("json_prefill")
        assert t.assistant_prefill() == '{"thought": "'

    def test_prefill_respects_custom_reasoning_field(self) -> None:
        t = get_template("json_prefill", {"reasoning_field": "think"})
        assert t.assistant_prefill() == '{"think": "'

    def test_prefill_respects_custom_scratchpad(self) -> None:
        t = get_template("json_prefill", {"reasoning_field": "scratchpad"})
        assert t.assistant_prefill() == '{"scratchpad": "'

    def test_inherits_parse_from_json(self) -> None:
        t = get_template("json_prefill")
        raw = '{"thought": "reasoning", "content": "answer"}'
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "reasoning"
        assert parsed.fields["content"] == "answer"

    def test_inherits_validate_from_json(self) -> None:
        t = get_template("json_prefill")
        parsed = t.parse('{"thought": "t", "content": "c"}')
        assert t.validate(parsed) == []

    def test_inherits_fields_from_json(self) -> None:
        t = get_template("json_prefill")
        fields = t.get_fields()
        assert len(fields) == 2
        assert fields[0].name == "thought"
        assert fields[1].name == "content"

    def test_inherits_format_instruction_from_json(self) -> None:
        t = get_template("json_prefill")
        instruction = t.format_instruction()
        assert "JSON" in instruction

    def test_name_is_json_prefill(self) -> None:
        t = get_template("json_prefill")
        assert t.name == "json_prefill"

    def test_description_mentions_prefill(self) -> None:
        t = get_template("json_prefill")
        assert "prefill" in t.description.lower()

    def test_with_params_returns_prefill_instance(self) -> None:
        t = get_template("json_prefill", {"reasoning_field": "reflect"})
        assert t.name == "json_prefill"
        assert t.assistant_prefill() == '{"reflect": "'

    def test_custom_field_changes_parse_and_validate(self) -> None:
        t = get_template("json_prefill", {"reasoning_field": "think"})
        raw = '{"think": "reasoning", "content": "answer"}'
        parsed = t.parse(raw)
        assert t.validate(parsed) == []
        assert parsed.fields["think"] == "reasoning"


# ──────────────────────────────────────────────────────────────────
# Markdown headers prefill template
# ──────────────────────────────────────────────────────────────────


class TestMarkdownHeadersPrefillTemplate:
    def test_prefill_returns_section_header(self) -> None:
        t = get_template("markdown_headers_prefill")
        assert t.assistant_prefill() == "## Thought\n"

    def test_prefill_respects_custom_reasoning_field(self) -> None:
        t = get_template("markdown_headers_prefill", {"reasoning_field": "Think"})
        assert t.assistant_prefill() == "## Think\n"

    def test_prefill_respects_custom_scratchpad(self) -> None:
        t = get_template("markdown_headers_prefill", {"reasoning_field": "Scratchpad"})
        assert t.assistant_prefill() == "## Scratchpad\n"

    def test_inherits_parse_from_markdown(self) -> None:
        t = get_template("markdown_headers_prefill")
        raw = "## Thought\nreasoning\n\n## Content\nanswer"
        parsed = t.parse(raw)
        assert parsed.fields["Thought"] == "reasoning"
        assert parsed.fields["Content"] == "answer"

    def test_inherits_validate_from_markdown(self) -> None:
        t = get_template("markdown_headers_prefill")
        parsed = t.parse("## Thought\nt\n\n## Content\nc")
        assert t.validate(parsed) == []

    def test_inherits_fields_from_markdown(self) -> None:
        t = get_template("markdown_headers_prefill")
        fields = t.get_fields()
        assert len(fields) == 2
        assert fields[0].name == "Thought"
        assert fields[1].name == "Content"

    def test_inherits_format_instruction_from_markdown(self) -> None:
        t = get_template("markdown_headers_prefill")
        instruction = t.format_instruction()
        assert "## Thought" in instruction

    def test_name_is_markdown_headers_prefill(self) -> None:
        t = get_template("markdown_headers_prefill")
        assert t.name == "markdown_headers_prefill"

    def test_description_mentions_prefill(self) -> None:
        t = get_template("markdown_headers_prefill")
        assert "prefill" in t.description.lower()

    def test_with_params_returns_prefill_instance(self) -> None:
        t = get_template("markdown_headers_prefill", {"reasoning_field": "Reflect"})
        assert t.name == "markdown_headers_prefill"
        assert t.assistant_prefill() == "## Reflect\n"

    def test_custom_field_changes_parse_and_validate(self) -> None:
        t = get_template("markdown_headers_prefill", {"reasoning_field": "Think"})
        raw = "## Think\nreasoning\n\n## Content\nanswer"
        parsed = t.parse(raw)
        assert t.validate(parsed) == []
        assert parsed.fields["Think"] == "reasoning"

    def test_content_field_name(self) -> None:
        t = get_template("markdown_headers_prefill")
        assert t.content_field_name() == "Content"


# ──────────────────────────────────────────────────────────────────
# XML with emotions prefill template
# ──────────────────────────────────────────────────────────────────


class TestXmlWithEmotionsPrefillTemplate:
    def test_prefill_returns_opening_reasoning_tag(self) -> None:
        t = get_template("xml_emotions_prefill")
        assert t.assistant_prefill() == "<thought>"

    def test_prefill_respects_custom_reasoning_field(self) -> None:
        t = get_template("xml_emotions_prefill", {"reasoning_field": "think"})
        assert t.assistant_prefill() == "<think>"

    def test_inherits_parse_from_xml_emotions(self) -> None:
        t = get_template("xml_emotions_prefill")
        raw = "<thought>t</thought><emotions>happy</emotions><content>c</content>"
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "t"
        assert parsed.fields["emotions"] == "happy"
        assert parsed.fields["content"] == "c"

    def test_inherits_validate_from_xml_emotions(self) -> None:
        t = get_template("xml_emotions_prefill")
        parsed = t.parse("<thought>t</thought><emotions>e</emotions><content>c</content>")
        assert t.validate(parsed) == []

    def test_inherits_fields_order_from_xml_emotions(self) -> None:
        t = get_template("xml_emotions_prefill")
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["thought", "emotions", "content"]

    def test_inherits_format_instruction_from_xml_emotions(self) -> None:
        t = get_template("xml_emotions_prefill")
        instruction = t.format_instruction()
        assert "<emotions>" in instruction
        assert "<thought>" in instruction

    def test_name_is_xml_emotions_prefill(self) -> None:
        t = get_template("xml_emotions_prefill")
        assert t.name == "xml_emotions_prefill"

    def test_description_mentions_prefill(self) -> None:
        t = get_template("xml_emotions_prefill")
        assert "prefill" in t.description.lower()

    def test_with_params_returns_prefill_instance(self) -> None:
        t = get_template("xml_emotions_prefill", {"reasoning_field": "reflect"})
        assert t.name == "xml_emotions_prefill"
        assert t.assistant_prefill() == "<reflect>"

    def test_custom_emotions_field(self) -> None:
        t = get_template("xml_emotions_prefill", {"emotions_field": "mood"})
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["thought", "mood", "content"]

    def test_combined_custom_fields(self) -> None:
        t = get_template(
            "xml_emotions_prefill",
            {
                "reasoning_field": "think",
                "emotions_field": "mood",
                "num_emotions": 5,
            },
        )
        assert t.assistant_prefill() == "<think>"
        fields = t.get_fields()
        names = [f.name for f in fields]
        assert names == ["think", "mood", "content"]
        instruction = t.format_instruction()
        assert "<think>" in instruction
        assert "<mood>" in instruction
        assert "exactly 5 emotions" in instruction

    def test_declares_all_four_params(self) -> None:
        t = get_template("xml_emotions_prefill")
        keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in keys
        assert "num_reasoning_paragraphs" in keys
        assert "emotions_field" in keys
        assert "num_emotions" in keys


# ──────────────────────────────────────────────────────────────────
# Gemma-4-style reasoning template -- default + parameterized
# ──────────────────────────────────────────────────────────────────


class TestGemmaReasoningTemplate:
    def test_parse_extracts_both_tags(self) -> None:
        t = get_template("gemma_reasoning")
        raw = (
            "<thought>\n"
            "*   The user asks about X.\n"
            "    *   Tone: curious, practical.\n"
            "</thought>\n"
            "<content>reply</content>"
        )
        parsed = t.parse(raw)
        assert "user asks" in parsed.fields["thought"]
        assert parsed.fields["content"] == "reply"

    def test_parse_handles_multiline_reasoning(self) -> None:
        t = get_template("gemma_reasoning")
        raw = (
            "<thought>\n"
            "*   User wants a practical explanation.\n"
            "    *   Context: second exchange.\n\n"
            "    *   Keep it direct and concrete.\n"
            "</thought>\n"
            "<content>response text</content>"
        )
        parsed = t.parse(raw)
        assert "practical" in parsed.fields["thought"]
        assert "direct" in parsed.fields["thought"]

    def test_validate_passes_complete(self) -> None:
        t = get_template("gemma_reasoning")
        parsed = t.parse("<thought>analysis</thought><content>reply</content>")
        assert t.validate(parsed) == []

    def test_validate_fails_missing_reasoning(self) -> None:
        t = get_template("gemma_reasoning")
        parsed = t.parse("<content>c</content>")
        errors = t.validate(parsed)
        assert any("thought" in e.lower() for e in errors)

    def test_validate_fails_missing_content(self) -> None:
        t = get_template("gemma_reasoning")
        parsed = t.parse("<thought>r</thought>")
        errors = t.validate(parsed)
        assert any("content" in e.lower() for e in errors)

    def test_validate_fails_empty_tag(self) -> None:
        t = get_template("gemma_reasoning")
        parsed = t.parse("<thought>  </thought><content>c</content>")
        errors = t.validate(parsed)
        assert any("empty" in e.lower() for e in errors)

    def test_format_instruction_mentions_tags(self) -> None:
        t = get_template("gemma_reasoning")
        instruction = t.format_instruction()
        assert "<thought>" in instruction
        assert "<content>" in instruction
        assert "RESPONSE FORMAT" in instruction

    def test_format_instruction_describes_block_format(self) -> None:
        t = get_template("gemma_reasoning")
        instruction = t.format_instruction()
        assert "blank-line-separated blocks" in instruction
        assert "guideline" in instruction

    def test_format_instruction_block_count(self) -> None:
        t = get_template("gemma_reasoning")
        instruction = t.format_instruction()
        assert "roughly 4" in instruction

    def test_examples_has_positive_and_negative(self) -> None:
        t = get_template("gemma_reasoning")
        examples = t.examples()
        assert any(ex.is_positive for ex in examples)
        assert any(not ex.is_positive for ex in examples)

    def test_positive_example_uses_unlabelled_bullets(self) -> None:
        t = get_template("gemma_reasoning")
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "*   " in pos.text

    def test_example_with_enough_blocks_includes_options(self) -> None:
        t = get_template("gemma_reasoning", {"num_reasoning_blocks": 6})
        pos = next(ex for ex in t.examples() if ex.is_positive)
        assert "Option 1" in pos.text or "Option 2" in pos.text

    def test_positive_example_has_blank_line_separated_blocks(self) -> None:
        t = get_template("gemma_reasoning")
        pos = next(ex for ex in t.examples() if ex.is_positive)
        tag = f"<{t._reasoning_field}>"
        end_tag = f"</{t._reasoning_field}>"
        reasoning_section = pos.text.split(tag, 1)[1].split(end_tag, 1)[0]
        blocks = [b.strip() for b in reasoning_section.split("\n\n") if b.strip()]
        assert len(blocks) >= 4

    def test_fields_order(self) -> None:
        t = get_template("gemma_reasoning")
        fields = t.get_fields()
        assert fields[0].name == "thought"
        assert fields[1].name == "content"

    def test_reasoning_is_hideable(self) -> None:
        t = get_template("gemma_reasoning")
        fields = t.get_fields()
        reasoning = next(f for f in fields if f.name == "thought")
        assert reasoning.user_can_hide is True
        assert reasoning.expandable is True

    def test_render_reasoning_html(self) -> None:
        t = get_template("gemma_reasoning")
        html = t.render_field_html("thought", "analysis points here", expandable=True)
        assert "blockquote" in html
        assert "Thought" in html

    def test_render_content_html(self) -> None:
        t = get_template("gemma_reasoning")
        html = t.render_field_html("content", "hello world")
        assert "blockquote" not in html

    def test_assistant_prefill_is_none(self) -> None:
        t = get_template("gemma_reasoning")
        assert t.assistant_prefill() is None

    def test_description(self) -> None:
        t = get_template("gemma_reasoning")
        assert "Gemma" in t.description or "gemma" in t.description.lower()
        assert "thought" in t.description.lower()


class TestGemmaReasoningTemplateParams:
    def test_declares_reasoning_field_param(self) -> None:
        t = get_template("gemma_reasoning")
        param_keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in param_keys
        assert "num_reasoning_blocks" in param_keys

    def test_default_effective_params(self) -> None:
        t = get_template("gemma_reasoning")
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "thought"
        assert ep["num_reasoning_blocks"] == 4

    def test_custom_reasoning_field_changes_fields(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "analysis"})
        fields = t.get_fields()
        assert fields[0].name == "analysis"
        assert fields[1].name == "content"

    def test_custom_reasoning_field_changes_instruction(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "think"})
        instruction = t.format_instruction()
        assert "<think>" in instruction
        assert "</think>" in instruction
        assert "<reasoning>" not in instruction

    def test_custom_reasoning_field_changes_examples(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "analysis"})
        examples = t.examples()
        pos = next(ex for ex in examples if ex.is_positive)
        assert "<analysis>" in pos.text
        assert "</analysis>" in pos.text
        assert "<reasoning>" not in pos.text

    def test_custom_reasoning_field_changes_parsing(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "think"})
        raw = "<think>steps here</think><content>answer</content>"
        parsed = t.parse(raw)
        assert parsed.fields["think"] == "steps here"
        assert parsed.fields["content"] == "answer"

    def test_custom_reasoning_field_changes_validation(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "think"})
        parsed = t.parse("<content>c</content>")
        errors = t.validate(parsed)
        assert any("think" in e.lower() for e in errors)

    def test_custom_reasoning_field_changes_description(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "analysis"})
        assert "analysis" in t.description

    def test_num_reasoning_blocks_in_instruction(self) -> None:
        t = get_template("gemma_reasoning", {"num_reasoning_blocks": 6})
        instruction = t.format_instruction()
        assert "roughly 6" in instruction

    def test_example_has_correct_block_count(self) -> None:
        for n in (3, 4, 5, 6):
            t = get_template("gemma_reasoning", {"num_reasoning_blocks": n})
            pos = next(ex for ex in t.examples() if ex.is_positive)
            tag = f"<{t._reasoning_field}>"
            end_tag = f"</{t._reasoning_field}>"
            reasoning_section = pos.text.split(tag, 1)[1].split(end_tag, 1)[0]
            blocks = [b.strip() for b in reasoning_section.split("\n\n") if b.strip()]
            assert len(blocks) >= n, f"Expected at least {n} blocks, got {len(blocks)}"

    def test_param_clamping_min(self) -> None:
        t = get_template("gemma_reasoning", {"num_reasoning_blocks": 0})
        ep = t.get_effective_params()
        assert ep["num_reasoning_blocks"] >= 3

    def test_param_clamping_max(self) -> None:
        t = get_template("gemma_reasoning", {"num_reasoning_blocks": 100})
        ep = t.get_effective_params()
        assert ep["num_reasoning_blocks"] <= 8

    def test_invalid_int_param_uses_default(self) -> None:
        t = get_template("gemma_reasoning", {"num_reasoning_blocks": "not_a_number"})
        ep = t.get_effective_params()
        assert ep["num_reasoning_blocks"] == 4

    def test_empty_string_param_uses_default(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": ""})
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "thought"

    def test_unknown_params_ignored(self) -> None:
        t = get_template("gemma_reasoning", {"nonexistent_param": "value"})
        assert t.get_fields()[0].name == "thought"

    def test_with_params_returns_new_instance(self) -> None:
        t1 = get_template("gemma_reasoning")
        t2 = t1.with_params({"reasoning_field": "think"})
        assert t1.get_fields()[0].name == "thought"
        assert t2.get_fields()[0].name == "think"

    def test_effective_params_after_with_params(self) -> None:
        t = get_template(
            "gemma_reasoning",
            {"reasoning_field": "analysis", "num_reasoning_blocks": 5},
        )
        ep = t.get_effective_params()
        assert ep["reasoning_field"] == "analysis"
        assert ep["num_reasoning_blocks"] == 5

    def test_render_html_with_custom_field(self) -> None:
        t = get_template("gemma_reasoning", {"reasoning_field": "think"})
        html = t.render_field_html("think", "some text", expandable=True)
        assert "blockquote" in html
        assert "Think" in html

    def test_param_suggestions_provided(self) -> None:
        t = get_template("gemma_reasoning")
        rf_param = next(p for p in t.get_params() if p.key == "reasoning_field")
        assert len(rf_param.suggestions) > 0
        assert "thought" in rf_param.suggestions


# ──────────────────────────────────────────────────────────────────
# Gemma reasoning prefill template
# ──────────────────────────────────────────────────────────────────


class TestGemmaReasoningPrefillTemplate:
    def test_prefill_returns_opening_tag_with_bullet(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        prefill = t.assistant_prefill()
        assert "<thought>" in prefill
        assert "*   " in prefill

    def test_prefill_respects_custom_reasoning_field(self) -> None:
        t = get_template("gemma_reasoning_prefill", {"reasoning_field": "think"})
        prefill = t.assistant_prefill()
        assert "<think>" in prefill
        assert "*   " in prefill

    def test_inherits_parse_from_gemma_reasoning(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        raw = "<thought>steps here</thought><content>answer</content>"
        parsed = t.parse(raw)
        assert parsed.fields["thought"] == "steps here"
        assert parsed.fields["content"] == "answer"

    def test_inherits_validate_from_gemma_reasoning(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        parsed = t.parse("<thought>r</thought><content>c</content>")
        assert t.validate(parsed) == []

    def test_inherits_fields_from_gemma_reasoning(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        fields = t.get_fields()
        assert fields[0].name == "thought"
        assert fields[1].name == "content"

    def test_inherits_format_instruction_from_gemma_reasoning(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        instruction = t.format_instruction()
        assert "<thought>" in instruction
        assert "block" in instruction.lower()

    def test_inherits_examples_from_gemma_reasoning(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        examples = t.examples()
        assert any(ex.is_positive for ex in examples)
        assert any(not ex.is_positive for ex in examples)

    def test_name_is_gemma_reasoning_prefill(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        assert t.name == "gemma_reasoning_prefill"

    def test_description_mentions_prefill(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        assert "prefill" in t.description.lower()

    def test_with_params_returns_prefill_instance(self) -> None:
        t = get_template("gemma_reasoning_prefill", {"reasoning_field": "analysis"})
        assert t.name == "gemma_reasoning_prefill"
        assert "<analysis>" in t.assistant_prefill()

    def test_custom_field_changes_parse_and_validate(self) -> None:
        t = get_template("gemma_reasoning_prefill", {"reasoning_field": "think"})
        raw = "<think>steps</think><content>answer</content>"
        parsed = t.parse(raw)
        assert t.validate(parsed) == []
        assert parsed.fields["think"] == "steps"

    def test_description_includes_custom_field(self) -> None:
        t = get_template("gemma_reasoning_prefill", {"reasoning_field": "analysis"})
        assert "analysis" in t.description

    def test_declares_both_params(self) -> None:
        t = get_template("gemma_reasoning_prefill")
        keys = [p.key for p in t.get_params()]
        assert "reasoning_field" in keys
        assert "num_reasoning_blocks" in keys


# ──────────────────────────────────────────────────────────────────
# Non-prefill templates return None
# ──────────────────────────────────────────────────────────────────


class TestNonPrefillTemplatesReturnNone:
    """All non-prefill templates should return None from assistant_prefill()."""

    NON_PREFILL: ClassVar[list[str]] = [
        "empty",
        "xml",
        "json",
        "markdown_headers",
        "xml_emotions",
        "gemma_reasoning",
    ]

    def test_all_non_prefill_return_none(self) -> None:
        for name in self.NON_PREFILL:
            t = get_template(name)
            assert t.assistant_prefill() is None, (
                f"{name} should return None for assistant_prefill()"
            )

    def test_parameterized_non_prefill_still_return_none(self) -> None:
        for name in ["xml", "json", "markdown_headers", "xml_emotions"]:
            t = get_template(name, {"reasoning_field": "custom"})
            assert t.assistant_prefill() is None, (
                f"Parameterized {name} should still return None for assistant_prefill()"
            )


# ──────────────────────────────────────────────────────────────────
# Cross-template consistency checks
# ──────────────────────────────────────────────────────────────────


class TestCrossTemplateConsistency:
    """Verify that all parameterizable templates behave consistently."""

    PARAMETERIZED_TEMPLATES: ClassVar[list[str]] = [
        "xml",
        "json",
        "markdown_headers",
        "xml_emotions",
        "gemma_reasoning",
    ]

    ALL_PARAMETERIZED: ClassVar[list[str]] = [
        *PARAMETERIZED_TEMPLATES,
        *PREFILL_TEMPLATES,
    ]

    def test_all_parameterized_templates_declare_reasoning_field(self) -> None:
        for name in self.ALL_PARAMETERIZED:
            t = get_template(name)
            keys = [p.key for p in t.get_params()]
            assert "reasoning_field" in keys, f"{name} missing reasoning_field param"

    def test_all_parameterized_templates_declare_reasoning_length_param(self) -> None:
        for name in self.ALL_PARAMETERIZED:
            t = get_template(name)
            keys = [p.key for p in t.get_params()]
            has_paragraphs = "num_reasoning_paragraphs" in keys
            has_steps = "num_reasoning_steps" in keys
            has_blocks = "num_reasoning_blocks" in keys
            assert has_paragraphs or has_steps or has_blocks, (
                f"{name} missing reasoning length param "
                f"(num_reasoning_paragraphs, num_reasoning_steps, "
                f"or num_reasoning_blocks)"
            )

    def test_default_params_produce_backward_compatible_fields(self) -> None:
        """Default parameterized templates should match the original field names."""
        xml = get_template("xml")
        assert xml.get_fields()[0].name == "thought"

        json_tpl = get_template("json")
        assert json_tpl.get_fields()[0].name == "thought"

        md = get_template("markdown_headers")
        assert md.get_fields()[0].name == "Thought"

        emo = get_template("xml_emotions")
        names = [f.name for f in emo.get_fields()]
        assert names == ["thought", "emotions", "content"]

    def test_with_params_preserves_template_name(self) -> None:
        for name in self.ALL_PARAMETERIZED:
            t = get_template(name, {"reasoning_field": "custom"})
            assert t.name == name

    def test_examples_always_contain_configured_field_names(self) -> None:
        for name in self.ALL_PARAMETERIZED:
            custom_field = "zz_custom"
            if name in ("markdown_headers", "markdown_headers_prefill"):
                custom_field = "Zzcustom"
            t = get_template(name, {"reasoning_field": custom_field})
            examples = t.examples()
            assert len(examples) > 0
            pos_examples = [ex for ex in examples if ex.is_positive]
            assert len(pos_examples) > 0
            for ex in pos_examples:
                assert custom_field.lower() in ex.text.lower(), (
                    f"{name}: example missing custom field name '{custom_field}'"
                )

    def test_prefill_templates_share_fields_with_parents(self) -> None:
        """Prefill templates must produce identical fields as their parent."""
        for prefill_name, parent_name in PARENT_OF.items():
            parent = get_template(parent_name)
            prefill = get_template(prefill_name)
            parent_fields = [(f.name, f.required, f.order) for f in parent.get_fields()]
            prefill_fields = [(f.name, f.required, f.order) for f in prefill.get_fields()]
            assert parent_fields == prefill_fields, (
                f"{prefill_name} fields differ from {parent_name}"
            )

    def test_prefill_templates_share_instruction_with_parents(self) -> None:
        """Prefill templates produce the same format instruction as their parent."""
        for prefill_name, parent_name in PARENT_OF.items():
            parent = get_template(parent_name)
            prefill = get_template(prefill_name)
            assert parent.format_instruction() == prefill.format_instruction(), (
                f"{prefill_name} instruction differs from {parent_name}"
            )

    def test_prefill_templates_share_examples_with_parents(self) -> None:
        """Prefill templates produce the same examples as their parent."""
        for prefill_name, parent_name in PARENT_OF.items():
            parent = get_template(parent_name)
            prefill = get_template(prefill_name)
            parent_ex = [(e.text, e.is_positive) for e in parent.examples()]
            prefill_ex = [(e.text, e.is_positive) for e in prefill.examples()]
            assert parent_ex == prefill_ex, f"{prefill_name} examples differ from {parent_name}"

    def test_prefill_templates_return_non_none_prefill(self) -> None:
        """All prefill templates must return a non-None assistant_prefill."""
        for name in PREFILL_TEMPLATES:
            t = get_template(name)
            prefill = t.assistant_prefill()
            assert prefill is not None, f"{name} returned None for assistant_prefill()"
            assert len(prefill) > 0, f"{name} returned empty string for assistant_prefill()"

    def test_prefill_changes_with_custom_reasoning_field(self) -> None:
        """Prefill must update when reasoning_field is customized."""
        for name in PREFILL_TEMPLATES:
            field_name = "customfield"
            if name == "markdown_headers_prefill":
                field_name = "Customfield"
            t = get_template(name, {"reasoning_field": field_name})
            prefill = t.assistant_prefill()
            assert prefill is not None
            assert field_name.lower() in prefill.lower(), (
                f"{name}: prefill '{prefill}' doesn't contain custom field '{field_name}'"
            )


# ──────────────────────────────────────────────────────────────────
# render_field_html tests
# ──────────────────────────────────────────────────────────────────


class TestRenderFieldHtml:
    def test_render_known_field_blockquote(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("thought", "deep thinking")
        assert "<blockquote>" in html
        assert "deep thinking" in html

    def test_render_unknown_field_passes_through(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("nonexistent_field", "**bold text**")
        assert "bold text" in html
        assert "<blockquote>" not in html

    def test_render_field_with_expandable(self) -> None:
        t = get_template("xml")
        html = t.render_field_html("thought", "expandable text", expandable=True)
        assert "expandable" in html

    def test_render_field_display_tag_none(self) -> None:
        """Fields with display_tag='none' render without blockquote wrapper."""
        t = get_template("xml")
        html = t.render_field_html("content", "just text")
        assert "<blockquote>" not in html


# ──────────────────────────────────────────────────────────────────
# Additional _resolve_params and base class tests
# ──────────────────────────────────────────────────────────────────


class TestResolveParamsExtended:
    def test_str_param_none_falls_to_default(self) -> None:
        declared = {
            "name": TemplateParam(key="name", label="Name", param_type="str", default="thought")
        }
        result = ResponseTemplate._resolve_params({"name": None}, declared)
        assert result["name"] == "thought"

    def test_get_effective_params_returns_defaults(self) -> None:
        t = get_template("xml")
        params = t.get_effective_params()
        assert "reasoning_field" in params

    def test_build_with_params_default_returns_self(self) -> None:
        """The base _build_with_params returns self."""
        t = get_template("empty")
        result = t._build_with_params({"anything": "value"})
        assert result is t
