from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import fetch_knowledge_entries
from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional

_WIKI_PROMPT = """
You are an assistant with wiki memory tools.
When the user says to remember a durable fact, call wiki_create or wiki_edit before answering.
When the user asks you to recall a stored fact, consult wiki_search, wiki_list,
or wiki_read before answering.
Keep the final answer short.
"""


def _remember_color(functional_cli, chat_id: str) -> None:
    remember = functional_cli.send_message_with_live_retry(
        chat_id,
        "My favorite color is orange. Remember this exactly.",
        debug=True,
    )
    remember.require_ok()
    assert "Tools used:" in remember.stdout
    assert "wiki_" in remember.stdout

    wiki_files = list(functional_cli.chat_wiki_dir(chat_id).glob("*.md"))
    assert wiki_files, remember.output


def test_wiki_creation_and_recall_work_through_real_llm(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.write_prompt("wiki_helper", _WIKI_PROMPT)
    functional_cli.start_chat("func-wiki", prompt="wiki_helper").require_ok()

    _remember_color(functional_cli, "func-wiki")

    wiki_listing = functional_cli.read_wiki("func-wiki")
    assert wiki_listing.returncode == 0
    assert "orange" in wiki_listing.stdout.lower()
    assert fetch_knowledge_entries(functional_cli.db_path, "func-wiki")

    follow_up = functional_cli.send_message_with_live_retry(
        "func-wiki",
        "What is my favorite color? Reply with the color only.",
    )
    assert follow_up.returncode == 0
    assert "orange" in extract_last_response_body(follow_up.stdout).lower()


def test_repair_wiki_updates_imports_and_removes_orphans(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.write_prompt("wiki_helper", _WIKI_PROMPT)
    functional_cli.start_chat("func-repair-wiki", prompt="wiki_helper").require_ok()
    _remember_color(functional_cli, "func-repair-wiki")

    wiki_dir = functional_cli.chat_wiki_dir("func-repair-wiki")
    original_file = next(wiki_dir.glob("*.md"))
    original_file.write_text("My favorite color is teal.", encoding="utf-8")
    malformed_file = wiki_dir / "notes.md"
    malformed_file.write_text("skip me", encoding="utf-8")

    updated = functional_cli.repair_wiki("func-repair-wiki")
    assert updated.returncode == 0
    assert "=== Wiki Repair: func-repair-wiki ===" in updated.stdout
    assert "updated" in updated.stdout
    assert "skipped unparseable file" in updated.stdout

    created_file = wiki_dir / "4321_travel.md"
    created_file.write_text("Favorite destination: Kyoto.", encoding="utf-8")
    created = functional_cli.repair_wiki("func-repair-wiki")
    assert "created: travel" in created.stdout

    created_file.unlink()
    removed = functional_cli.repair_wiki("func-repair-wiki")
    assert "removed orphan DB row: travel" in removed.stdout

    wiki_listing = functional_cli.read_wiki("func-repair-wiki")
    assert "teal" in wiki_listing.stdout.lower()
    assert "travel" not in wiki_listing.stdout.lower()
