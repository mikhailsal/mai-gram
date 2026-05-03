"""Unit tests for the response template plugin system."""

from __future__ import annotations

from mai_gram.response_templates.base import FieldDescriptor
from mai_gram.response_templates.registry import get_template, list_template_names


class TestRegistry:
    def test_list_names_returns_all_builtin(self) -> None:
        names = list_template_names()
        assert "empty" in names
        assert "xml" in names
        assert "json" in names
        assert "markdown_headers" in names
        assert "xml_emotions" in names

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

    def test_description(self) -> None:
        t = get_template("xml")
        assert t.description


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
