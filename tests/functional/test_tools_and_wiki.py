from __future__ import annotations

import time

import pytest

from tests.functional.helpers.artifacts import fetch_knowledge_entries
from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional_live

_WIKI_PROMPT = """
You are an assistant with wiki memory tools.
When the user says to remember a durable fact, call wiki_create or wiki_edit before answering.
When the user asks you to recall a stored fact, consult wiki_search, wiki_list,
or wiki_read before answering.
Keep the final answer short.
"""

_MAX_WIKI_RETRIES = 5


def _remember_color(cli, chat_id: str) -> bool:
    """Ask the model to remember a color. Returns True if wiki file was created."""
    remember = cli.send_message_with_live_retry(
        chat_id,
        "My favorite color is orange. Remember this exactly.",
        debug=True,
    )
    if remember.returncode != 0:
        return False
    if "Tools used:" not in remember.stdout or "wiki_" not in remember.stdout:
        return False

    wiki_files = list(cli.chat_wiki_dir(chat_id).glob("*.md"))
    return bool(wiki_files)


def test_wiki_creation_and_recall_work_through_real_llm(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.write_prompt("wiki_helper", _WIKI_PROMPT)
    cli.start_chat("func-wiki", prompt="wiki_helper").require_ok()

    remembered = False
    for attempt in range(1, _MAX_WIKI_RETRIES + 1):
        if _remember_color(cli, "func-wiki"):
            remembered = True
            break
        if attempt < _MAX_WIKI_RETRIES:
            time.sleep(2.0 * attempt)

    assert remembered, f"Model failed to call wiki tool after {_MAX_WIKI_RETRIES} attempts"

    wiki_listing = cli.read_wiki("func-wiki")
    assert wiki_listing.returncode == 0
    assert "orange" in wiki_listing.stdout.lower()
    assert fetch_knowledge_entries(cli.db_path, "func-wiki")

    last_body = ""
    for attempt in range(1, _MAX_WIKI_RETRIES + 1):
        follow_up = cli.send_message_with_live_retry(
            "func-wiki",
            "What is my favorite color? Reply with the color only.",
        )
        try:
            last_body = extract_last_response_body(follow_up.stdout)
            if follow_up.returncode == 0 and "orange" in last_body.lower():
                break
        except AssertionError:
            last_body = follow_up.output

        if attempt < _MAX_WIKI_RETRIES:
            time.sleep(2.0 * attempt)

    assert "orange" in last_body.lower(), (
        f"Expected 'orange' after {_MAX_WIKI_RETRIES} recall attempts, got: {last_body!r}"
    )


def test_repair_wiki_updates_imports_and_removes_orphans(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.write_prompt("wiki_helper", _WIKI_PROMPT)
    cli.start_chat("func-repair-wiki", prompt="wiki_helper").require_ok()

    remembered = False
    for attempt in range(1, _MAX_WIKI_RETRIES + 1):
        if _remember_color(cli, "func-repair-wiki"):
            remembered = True
            break
        if attempt < _MAX_WIKI_RETRIES:
            time.sleep(2.0 * attempt)
    assert remembered, "Model failed to call wiki tool for repair test"

    wiki_dir = cli.chat_wiki_dir("func-repair-wiki")
    original_file = next(wiki_dir.glob("*.md"))
    original_file.write_text("My favorite color is teal.", encoding="utf-8")
    malformed_file = wiki_dir / "notes.md"
    malformed_file.write_text("skip me", encoding="utf-8")

    updated = cli.repair_wiki("func-repair-wiki")
    assert updated.returncode == 0
    assert "=== Wiki Repair: func-repair-wiki ===" in updated.stdout
    assert "updated" in updated.stdout
    assert "skipped unparseable file" in updated.stdout

    created_file = wiki_dir / "4321_travel.md"
    created_file.write_text("Favorite destination: Kyoto.", encoding="utf-8")
    created = cli.repair_wiki("func-repair-wiki")
    assert "created: travel" in created.stdout

    created_file.unlink()
    removed = cli.repair_wiki("func-repair-wiki")
    assert "removed orphan DB row: travel" in removed.stdout

    wiki_listing = cli.read_wiki("func-repair-wiki")
    assert "teal" in wiki_listing.stdout.lower()
    assert "travel" not in wiki_listing.stdout.lower()
