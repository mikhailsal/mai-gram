"""Console-only output helpers."""

from __future__ import annotations

from typing import Any


def print_debug_session_stats(stats: dict[str, Any]) -> None:
    print("")
    print("--- Debug Info ---")
    print(f"LLM calls: {stats['llm_calls']} ({stats['calls_with_tool_calls']} with tool calls)")
    tools_used = ", ".join(stats["tools_used"]) if stats["tools_used"] else "none"
    print(f"Tools used: {tools_used}")
    print(
        f"Tokens: {stats['prompt_tokens']:,} prompt + "
        f"{stats['completion_tokens']:,} completion = "
        f"{stats['total_tokens']:,} total"
    )
    print("")
    print("--- Session Cost ---")
    print(
        f"This call: {stats['last_call_total_tokens']:,} tokens "
        f"(${stats['last_call_cost_usd']:.3f})"
    )
    print(f"Session total: {stats['total_tokens']:,} tokens (${stats['session_cost_usd']:.3f})")
    if stats["log_path"]:
        print(f"Full log: {stats['log_path']}")
