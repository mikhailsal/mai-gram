"""Tests for markdown-to-Telegram conversion (md_to_telegram module)."""

from __future__ import annotations

from mai_gram.core.md_to_telegram import (
    format_reasoning_html,
    markdown_to_html,
    markdown_to_mdv2,
)

# ---------------------------------------------------------------------------
# LaTeX symbol conversion
# ---------------------------------------------------------------------------


class TestLatexSymbols:
    def test_rightarrow_inline_math(self) -> None:
        assert "\u2192" in markdown_to_html("A $\\rightarrow$ B")

    def test_leftarrow_inline_math(self) -> None:
        assert "\u2190" in markdown_to_html("A $\\leftarrow$ B")

    def test_double_rightarrow(self) -> None:
        assert "\u21d2" in markdown_to_html("$\\Rightarrow$")

    def test_leq_geq(self) -> None:
        result = markdown_to_html("$x \\leq y$ and $a \\geq b$")
        assert "\u2264" in result
        assert "\u2265" in result

    def test_neq(self) -> None:
        assert "\u2260" in markdown_to_html("$a \\neq b$")

    def test_infty(self) -> None:
        assert "\u221e" in markdown_to_html("$\\infty$")

    def test_greek_letters(self) -> None:
        result = markdown_to_html("$\\alpha$ and $\\beta$")
        assert "\u03b1" in result
        assert "\u03b2" in result

    def test_multiple_symbols_in_one_expression(self) -> None:
        result = markdown_to_html("$\\alpha \\rightarrow \\beta$")
        assert "\u03b1" in result
        assert "\u2192" in result
        assert "\u03b2" in result

    def test_bare_latex_command(self) -> None:
        assert "\u2192" in markdown_to_html("A \\rightarrow B")

    def test_latex_inside_code_block_untouched(self) -> None:
        result = markdown_to_html("```\n$\\rightarrow$\n```")
        assert "\u2192" not in result

    def test_latex_inside_inline_code_untouched(self) -> None:
        result = markdown_to_html("`$\\rightarrow$`")
        assert "\u2192" not in result

    def test_latex_mdv2(self) -> None:
        result = markdown_to_mdv2("A $\\rightarrow$ B")
        assert "\u2192" in result

    def test_times_and_div(self) -> None:
        result = markdown_to_html("$a \\times b \\div c$")
        assert "\u00d7" in result
        assert "\u00f7" in result

    def test_set_operations(self) -> None:
        result = markdown_to_html("$A \\cup B \\cap C$")
        assert "\u222a" in result
        assert "\u2229" in result


# ---------------------------------------------------------------------------
# Header conversion
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_h1_html(self) -> None:
        result = markdown_to_html("# Title")
        assert "<b>Title</b>" in result

    def test_h2_html(self) -> None:
        result = markdown_to_html("## Section")
        assert "<b>Section</b>" in result

    def test_h3_html(self) -> None:
        result = markdown_to_html("### Subsection")
        assert "<b>Subsection</b>" in result

    def test_h6_html(self) -> None:
        result = markdown_to_html("###### Deep")
        assert "<b>Deep</b>" in result

    def test_header_with_surrounding_text(self) -> None:
        result = markdown_to_html("Intro\n\n## Section\n\nBody text")
        assert "<b>Section</b>" in result
        assert "Intro" in result
        assert "Body text" in result

    def test_header_special_chars_escaped_html(self) -> None:
        result = markdown_to_html("## A & B <test>")
        assert "<b>A &amp; B &lt;test&gt;</b>" in result

    def test_header_mdv2(self) -> None:
        result = markdown_to_mdv2("## Bold Header")
        assert "*Bold Header*" in result

    def test_hash_in_middle_of_line_not_header(self) -> None:
        result = markdown_to_html("This is not ## a header")
        assert "<b>" not in result

    def test_multiple_headers(self) -> None:
        text = "# First\n\n## Second\n\n### Third"
        result = markdown_to_html(text)
        assert "<b>First</b>" in result
        assert "<b>Second</b>" in result
        assert "<b>Third</b>" in result


# ---------------------------------------------------------------------------
# List conversion
# ---------------------------------------------------------------------------


class TestLists:
    def test_simple_bullet_list(self) -> None:
        text = "* Item 1\n* Item 2\n* Item 3"
        result = markdown_to_html(text)
        assert "\u2022 Item 1" in result
        assert "\u2022 Item 2" in result

    def test_dash_bullet_list(self) -> None:
        text = "- Item 1\n- Item 2"
        result = markdown_to_html(text)
        assert "\u2022 Item 1" in result

    def test_numbered_list(self) -> None:
        text = "1. First\n2. Second\n3. Third"
        result = markdown_to_html(text)
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result

    def test_nested_bullet_list(self) -> None:
        text = "* Top\n    * Nested\n        * Deep"
        result = markdown_to_html(text)
        assert "\u2022 Top" in result
        assert "\u25e6 Nested" in result  # open circle for depth 1
        assert "\u25aa Deep" in result  # small square for depth 2

    def test_nested_numbered_list(self) -> None:
        text = "1. Top\n    1. Nested\n        1. Deep"
        result = markdown_to_html(text)
        assert "1. Top" in result
        assert "    1. Nested" in result
        assert "        1. Deep" in result

    def test_mixed_list(self) -> None:
        text = "* Bullet\n    1. Numbered sub-item\n    2. Another sub-item"
        result = markdown_to_html(text)
        assert "\u2022 Bullet" in result
        assert "    1. Numbered sub-item" in result

    def test_list_with_bold_items(self) -> None:
        text = "* **Bold item**\n* Normal item"
        result = markdown_to_html(text)
        assert "<b>Bold item</b>" in result
        assert "\u2022" in result

    def test_mdv2_list(self) -> None:
        text = "* Item 1\n* Item 2"
        result = markdown_to_mdv2(text)
        assert "\u2022 Item 1" in result


# ---------------------------------------------------------------------------
# Reasoning block formatting
# ---------------------------------------------------------------------------


class TestReasoningFormatting:
    def test_simple_reasoning(self) -> None:
        result = format_reasoning_html("Some reasoning text")
        assert "<blockquote>" in result
        assert "\U0001f4ad Reasoning" in result
        assert "Some reasoning text" in result
        assert "</blockquote>" in result

    def test_expandable_reasoning(self) -> None:
        result = format_reasoning_html("Some text", expandable=True)
        assert "<blockquote expandable>" in result
        assert "</blockquote>" in result

    def test_reasoning_with_bullet_list(self) -> None:
        text = "Thoughts:\n* Item one\n* Item two"
        result = format_reasoning_html(text)
        assert "\u2022 Item one" in result
        assert "\u2022 Item two" in result

    def test_reasoning_with_numbered_list(self) -> None:
        text = "Steps:\n1. First step\n2. Second step"
        result = format_reasoning_html(text)
        assert "1. First step" in result
        assert "2. Second step" in result

    def test_reasoning_with_nested_items(self) -> None:
        text = "Analysis:\n* Top level\n    * Sub level\n        * Deep level"
        result = format_reasoning_html(text)
        assert "\u2022 Top level" in result
        assert "\u25e6 Sub level" in result

    def test_reasoning_with_bold_italic(self) -> None:
        text = "This is **important** and *critical*"
        result = format_reasoning_html(text)
        assert "<b>important</b>" in result
        assert "<i>critical</i>" in result

    def test_reasoning_with_latex(self) -> None:
        text = "A $\\rightarrow$ B means implication"
        result = format_reasoning_html(text)
        assert "\u2192" in result

    def test_reasoning_empty(self) -> None:
        assert format_reasoning_html("") == ""
        assert format_reasoning_html("   ") == ""

    def test_reasoning_complex_example(self) -> None:
        """Test with the kind of reasoning block from the user's example."""
        text = (
            "The user wants another long text.\n\n"
            '    *   I\'ve already written a "Manifesto".\n'
            "    *   Now I need something different.\n\n"
            '    *   Topic: "The Anatomy of an AI Failure."\n'
            "    *   Structure:\n"
            "        1.  Introduction: Why failures are the only way to grow.\n"
            "        2.  Detailed Breakdown of the event.\n"
            '        3.  The "Context Bleed" phenomenon.\n'
        )
        result = format_reasoning_html(text)
        assert "<blockquote>" in result
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ---------------------------------------------------------------------------
# Existing functionality (regression tests)
# ---------------------------------------------------------------------------


class TestExistingFunctionality:
    def test_bold_html(self) -> None:
        assert "<b>bold</b>" in markdown_to_html("**bold**")

    def test_italic_html(self) -> None:
        assert "<i>italic</i>" in markdown_to_html("*italic*")

    def test_strikethrough_html(self) -> None:
        assert "<s>strike</s>" in markdown_to_html("~~strike~~")

    def test_inline_code_html(self) -> None:
        assert "<code>code</code>" in markdown_to_html("`code`")

    def test_code_block_html(self) -> None:
        result = markdown_to_html("```python\nprint('hi')\n```")
        assert "<pre" in result
        assert "print" in result

    def test_link_html(self) -> None:
        result = markdown_to_html("[text](https://example.com)")
        assert '<a href="https://example.com">text</a>' in result

    def test_blockquote_html(self) -> None:
        result = markdown_to_html("> Quote text")
        assert "<blockquote>" in result
        assert "Quote text" in result

    def test_bold_mdv2(self) -> None:
        result = markdown_to_mdv2("**bold**")
        assert "*bold*" in result

    def test_empty_string(self) -> None:
        assert markdown_to_html("") == ""
        assert markdown_to_mdv2("") == ""

    def test_plain_text_html_escaped(self) -> None:
        result = markdown_to_html("a < b & c > d")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result
