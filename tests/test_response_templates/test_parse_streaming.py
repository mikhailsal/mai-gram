"""Unit tests for parse_streaming() on each template type.

Covers progressive parsing of partial input at various stages:
empty, partial tags, one field complete, both fields complete,
malformed/extra text, and custom field names.
"""

from __future__ import annotations

from mai_gram.response_templates.base import StreamingParseResult
from mai_gram.response_templates.registry import get_template

# ──────────────────────────────────────────────────────────────────
# Empty template -- default (no-op) streaming
# ──────────────────────────────────────────────────────────────────


class TestEmptyTemplateStreaming:
    def test_returns_all_text_as_active_content(self) -> None:
        t = get_template("empty")
        result = t.parse_streaming("Hello world")
        assert result.completed_fields == {}
        assert result.active_field is None
        assert result.active_content == "Hello world"

    def test_empty_input(self) -> None:
        t = get_template("empty")
        result = t.parse_streaming("")
        assert result.completed_fields == {}
        assert result.active_content == ""


# ──────────────────────────────────────────────────────────────────
# XML template streaming
# ──────────────────────────────────────────────────────────────────


class TestXmlTemplateStreaming:
    def test_empty_input(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("")
        assert result.completed_fields == {}
        assert result.active_field is None
        assert result.active_content == ""

    def test_preamble_only(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("Some preamble text")
        assert result.completed_fields == {}
        assert result.active_field is None
        assert result.preamble == "Some preamble text"

    def test_partial_opening_tag(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thou")
        assert result.completed_fields == {}
        assert result.active_field is None

    def test_opening_thought_tag_only(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>starting to reason")
        assert result.completed_fields == {}
        assert result.active_field == "thought"
        assert result.active_content == "starting to reason"

    def test_thought_in_progress(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>reasoning about the problem\nstill thinking")
        assert result.completed_fields == {}
        assert result.active_field == "thought"
        assert "still thinking" in result.active_content

    def test_thought_completed_no_content(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>my reasoning</thought>")
        assert result.completed_fields == {"thought": "my reasoning"}
        assert result.active_field is None

    def test_thought_completed_content_started(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>reasoning here</thought>\n<content>reply start")
        assert result.completed_fields == {"thought": "reasoning here"}
        assert result.active_field == "content"
        assert result.active_content == "reply start"

    def test_both_fields_complete(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>reasoning</thought>\n<content>full reply</content>")
        assert result.completed_fields == {"thought": "reasoning", "content": "full reply"}
        assert result.active_field is None

    def test_preamble_before_tags(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("Some intro text\n<thought>reasoning</thought>")
        assert result.preamble == "Some intro text"
        assert result.completed_fields == {"thought": "reasoning"}

    def test_whitespace_stripping_in_completed_fields(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<thought>\n  spaced reasoning  \n</thought>")
        assert result.completed_fields["thought"] == "spaced reasoning"

    def test_multiline_thought_in_progress(self) -> None:
        t = get_template("xml")
        text = "<thought>\nFirst paragraph.\n\nSecond paragraph."
        result = t.parse_streaming(text)
        assert result.active_field == "thought"
        assert "First paragraph" in result.active_content
        assert "Second paragraph" in result.active_content


class TestXmlTemplateStreamingCustomField:
    def test_custom_reasoning_field(self) -> None:
        t = get_template("xml", {"reasoning_field": "think"})
        result = t.parse_streaming("<think>reasoning</think>\n<content>reply start")
        assert result.completed_fields == {"think": "reasoning"}
        assert result.active_field == "content"
        assert result.active_content == "reply start"

    def test_custom_field_partial(self) -> None:
        t = get_template("xml", {"reasoning_field": "scratchpad"})
        result = t.parse_streaming("<scratchpad>working on it")
        assert result.active_field == "scratchpad"
        assert result.active_content == "working on it"


# ──────────────────────────────────────────────────────────────────
# XML with emotions streaming
# ──────────────────────────────────────────────────────────────────


class TestXmlEmotionsStreaming:
    def test_thought_in_progress(self) -> None:
        t = get_template("xml_emotions")
        result = t.parse_streaming("<thought>reasoning")
        assert result.active_field == "thought"
        assert result.active_content == "reasoning"
        assert result.completed_fields == {}

    def test_thought_done_emotions_in_progress(self) -> None:
        t = get_template("xml_emotions")
        result = t.parse_streaming("<thought>reasoning</thought><emotions>curious, excited")
        assert result.completed_fields == {"thought": "reasoning"}
        assert result.active_field == "emotions"
        assert "curious" in result.active_content

    def test_thought_and_emotions_done_content_in_progress(self) -> None:
        t = get_template("xml_emotions")
        result = t.parse_streaming(
            "<thought>reasoning</thought><emotions>happy</emotions><content>reply"
        )
        assert result.completed_fields == {"thought": "reasoning", "emotions": "happy"}
        assert result.active_field == "content"
        assert result.active_content == "reply"

    def test_all_three_complete(self) -> None:
        t = get_template("xml_emotions")
        result = t.parse_streaming("<thought>r</thought><emotions>e</emotions><content>c</content>")
        assert len(result.completed_fields) == 3
        assert result.active_field is None


# ──────────────────────────────────────────────────────────────────
# XML prefill streaming (inherits from XmlTemplate)
# ──────────────────────────────────────────────────────────────────


class TestXmlPrefillStreaming:
    def test_prefill_context_parsing(self) -> None:
        """When prefill is applied, response starts mid-tag -- verify parsing works."""
        t = get_template("xml_prefill")
        prefill = t.assistant_prefill()
        accumulated = prefill + "reasoning here</thought>\n<content>reply"
        result = t.parse_streaming(accumulated)
        assert result.completed_fields == {"thought": "reasoning here"}
        assert result.active_field == "content"
        assert result.active_content == "reply"

    def test_prefill_partial_reasoning(self) -> None:
        t = get_template("xml_prefill")
        prefill = t.assistant_prefill()
        accumulated = prefill + "still thinking"
        result = t.parse_streaming(accumulated)
        assert result.active_field == "thought"
        assert "still thinking" in result.active_content


# ──────────────────────────────────────────────────────────────────
# Gemma reasoning template streaming (inherits from XmlTemplate)
# ──────────────────────────────────────────────────────────────────


class TestGemmaReasoningStreaming:
    def test_bullet_reasoning_in_progress(self) -> None:
        t = get_template("gemma_reasoning")
        text = "<thought>\n*   User asks about X.\n    *   Tone: curious."
        result = t.parse_streaming(text)
        assert result.active_field == "thought"
        assert "*   User asks" in result.active_content

    def test_bullet_reasoning_complete_content_in_progress(self) -> None:
        t = get_template("gemma_reasoning")
        text = "<thought>\n*   Analysis point.\n</thought>\n<content>Here's the answer"
        result = t.parse_streaming(text)
        assert "thought" in result.completed_fields
        assert result.active_field == "content"
        assert "answer" in result.active_content


# ──────────────────────────────────────────────────────────────────
# JSON template streaming
# ──────────────────────────────────────────────────────────────────


class TestJsonTemplateStreaming:
    def test_empty_input(self) -> None:
        t = get_template("json")
        result = t.parse_streaming("")
        assert result.completed_fields == {}
        assert result.active_content == ""

    def test_no_brace_yet(self) -> None:
        t = get_template("json")
        result = t.parse_streaming("Here is my response:")
        assert result.completed_fields == {}
        assert result.preamble == "Here is my response:"

    def test_opening_brace_only(self) -> None:
        t = get_template("json")
        result = t.parse_streaming("{")
        assert result.completed_fields == {}

    def test_thought_key_started_no_value(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "')
        assert result.completed_fields == {}
        assert result.active_field == "thought"
        assert result.active_content == ""

    def test_thought_value_in_progress(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "reasoning about the problem')
        assert result.active_field == "thought"
        assert "reasoning about" in result.active_content

    def test_thought_complete_content_not_started(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "reasoning here", ')
        assert result.completed_fields == {"thought": "reasoning here"}
        assert result.active_field is None

    def test_thought_complete_content_in_progress(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "reasoning", "content": "reply start')
        assert result.completed_fields == {"thought": "reasoning"}
        assert result.active_field == "content"
        assert result.active_content == "reply start"

    def test_both_fields_complete(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "reasoning", "content": "reply"}')
        assert result.completed_fields == {"thought": "reasoning", "content": "reply"}

    def test_escaped_newlines_in_value(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "line1\\nline2", "content": "reply')
        assert result.completed_fields["thought"] == "line1\nline2"
        assert result.active_field == "content"

    def test_escaped_quotes_in_value(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "said \\"hello\\"", "content": "ok')
        assert result.completed_fields["thought"] == 'said "hello"'
        assert result.active_field == "content"

    def test_preamble_before_json(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('Here it is: {"thought": "r')
        assert result.preamble == "Here it is:"
        assert result.active_field == "thought"

    def test_escaped_backslash(self) -> None:
        t = get_template("json")
        result = t.parse_streaming('{"thought": "path\\\\dir", "content": "ok"}')
        assert result.completed_fields["thought"] == "path\\dir"


class TestJsonTemplateStreamingCustomField:
    def test_custom_reasoning_field(self) -> None:
        t = get_template("json", {"reasoning_field": "think"})
        result = t.parse_streaming('{"think": "reasoning", "content": "reply start')
        assert result.completed_fields == {"think": "reasoning"}
        assert result.active_field == "content"

    def test_custom_field_in_progress(self) -> None:
        t = get_template("json", {"reasoning_field": "reflect"})
        result = t.parse_streaming('{"reflect": "working on it')
        assert result.active_field == "reflect"
        assert "working on it" in result.active_content


# ──────────────────────────────────────────────────────────────────
# Markdown headers template streaming
# ──────────────────────────────────────────────────────────────────


class TestMarkdownHeadersStreaming:
    def test_empty_input(self) -> None:
        t = get_template("markdown_headers")
        result = t.parse_streaming("")
        assert result.completed_fields == {}
        assert result.active_content == ""

    def test_preamble_only(self) -> None:
        t = get_template("markdown_headers")
        result = t.parse_streaming("Some intro text before headers")
        assert result.preamble == "Some intro text before headers"
        assert result.active_field is None

    def test_thought_header_in_progress(self) -> None:
        t = get_template("markdown_headers")
        result = t.parse_streaming("## Thought\nReasoning about the problem")
        assert result.active_field == "Thought"
        assert "Reasoning about" in result.active_content

    def test_thought_complete_content_in_progress(self) -> None:
        t = get_template("markdown_headers")
        result = t.parse_streaming("## Thought\nReasoning here\n\n## Content\nReply start")
        assert result.completed_fields == {"Thought": "Reasoning here"}
        assert result.active_field == "Content"
        assert "Reply start" in result.active_content

    def test_both_complete(self) -> None:
        t = get_template("markdown_headers")
        text = "## Thought\nReasoning\n\n## Content\nFull reply"
        result = t.parse_streaming(text)
        assert result.completed_fields == {"Thought": "Reasoning"}
        assert result.active_field == "Content"
        assert "Full reply" in result.active_content

    def test_preamble_before_headers(self) -> None:
        t = get_template("markdown_headers")
        text = "Some preamble\n## Thought\nReasoning"
        result = t.parse_streaming(text)
        assert result.preamble == "Some preamble"
        assert result.active_field == "Thought"


class TestMarkdownHeadersStreamingCustomField:
    def test_custom_reasoning_field(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Think"})
        result = t.parse_streaming("## Think\nReasoning here\n\n## Content\nReply")
        assert result.completed_fields == {"Think": "Reasoning here"}
        assert result.active_field == "Content"

    def test_custom_field_in_progress(self) -> None:
        t = get_template("markdown_headers", {"reasoning_field": "Reflection"})
        result = t.parse_streaming("## Reflection\nReflecting on things")
        assert result.active_field == "Reflection"
        assert "Reflecting" in result.active_content


# ──────────────────────────────────────────────────────────────────
# Edge cases shared across templates
# ──────────────────────────────────────────────────────────────────


class TestStreamingEdgeCases:
    def test_xml_no_crash_on_garbage(self) -> None:
        t = get_template("xml")
        result = t.parse_streaming("<<<>>>totally broken<<//>")
        assert isinstance(result, StreamingParseResult)

    def test_json_no_crash_on_garbage(self) -> None:
        t = get_template("json")
        result = t.parse_streaming("{{{broken json without quotes")
        assert isinstance(result, StreamingParseResult)

    def test_markdown_no_crash_on_garbage(self) -> None:
        t = get_template("markdown_headers")
        result = t.parse_streaming("### Wrong level header\n# Also wrong")
        assert isinstance(result, StreamingParseResult)

    def test_xml_nested_tags_in_content(self) -> None:
        """Content may contain code with XML-like tags -- should not confuse parser."""
        t = get_template("xml")
        text = "<thought>reasoning</thought><content>Use <div> in your HTML"
        result = t.parse_streaming(text)
        assert result.completed_fields["thought"] == "reasoning"
        assert result.active_field == "content"
        assert "<div>" in result.active_content
