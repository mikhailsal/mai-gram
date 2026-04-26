"""Tests for console output helpers."""

from __future__ import annotations

from mai_gram.console_output import print_debug_session_stats


def test_print_debug_session_stats_formats_summary(capsys: object) -> None:
    print_debug_session_stats(
        {
            "llm_calls": 3,
            "calls_with_tool_calls": 1,
            "tools_used": ["wiki_search"],
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "last_call_total_tokens": 60,
            "last_call_cost_usd": 0.123,
            "session_cost_usd": 0.456,
            "log_path": "/tmp/debug-log.json",
        }
    )

    captured = capsys.readouterr()
    assert "LLM calls: 3 (1 with tool calls)" in captured.out
    assert "Tools used: wiki_search" in captured.out
    assert "This call: 60 tokens ($0.123)" in captured.out
    assert "Full log: /tmp/debug-log.json" in captured.out
