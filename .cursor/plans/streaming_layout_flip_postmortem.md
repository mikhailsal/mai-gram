# Post-Mortem: Streaming Layout Flip (Blockquote ↔ Raw Text)

**Date**: 2026-05-09  
**Status**: Open — root cause identified, fix not yet implemented  
**Related commit**: `5791f82` (prior markdown stabilization fix)  
**Affected templates**: All XML-based templates (`xml`, `xml_emotions`) when thought/reasoning field grows large

---

## Symptom

During LLM response streaming, the thought block initially renders correctly
as a Telegram blockquote (with sidebar, label, and formatted inner text).
However, once the thought content reaches a certain length (~800+ characters),
the display begins **alternating every other edit** between:

1. **Correctly parsed blockquote** — with "💭 Thought" label, sidebar bar, and
   markdown-rendered inner text
2. **Raw text dump** — showing literal `<thought>` and `</thought>` XML tags,
   unformatted content, and visible template structure

This oscillation is visible to the end user and is distinct from the
previously-fixed markdown formatting flicker (bold/italic appearing and
disappearing within a correctly structured blockquote).

### Visual Evidence

- **Raw text state** (broken): User sees `<thought>The user wants something...
  </thought><thought>` literally in the message. No blockquote formatting.
  The chat preview shows `<thought>The user wants somethin...`
- **Correct state**: User sees a blockquote with "💭 Thought" heading,
  formatted bullet points, and the sidebar line.

---

## Root Cause Chain

### Step 1: HTML Edit Rejected by Telegram

As the thought grows, `_render_template_live_text` produces increasingly
complex HTML:

```
<blockquote>💭 Thought
...long inner HTML with bold, italic, lists...
</blockquote>
```

At some point, Telegram's strict HTML parser **rejects** the
`edit_message_text` call. Possible triggers:

- **Unclosed/malformed HTML tags** from `markdown_to_html` edge cases
  (despite stabilization, partial streaming text can produce broken HTML)
- **Nested blockquotes** if the LLM's thought text contains `>` markdown
  quotes, which `markdown_to_html` converts to `<blockquote>` inside the
  outer `<blockquote>`, creating invalid nesting
- **Message length close to 4096-char Telegram limit** — the HTML tags add
  overhead that pushes the message over the hard limit even though the raw
  text fits within 4000 chars
- **Entity parsing depth limits** in Telegram's Bot API for complex HTML

### Step 2: Fallback Sends Raw LLM Output

When the HTML edit fails, `_edit_existing` (line 356) falls back:

```python
# stream_display.py, _edit_existing
if not edit_result.success:
    fb = await self._messenger.edit_message(
        request.telegram_chat_id,
        placeholder_msg_id,
        fallback,                  # <-- NO parse_mode="html"
    )
```

The `fallback` text is constructed in `_assemble_live_text` (line 313):

```python
fallback = (remaining or current_content)[:max_len] + " ▍"
```

**The critical flaw**: When the thought is the active (non-content) field,
`remaining` is `""` because `active_text` was set to `""` at line 240
(the thought content was routed to `header_html` instead). So the fallback
degrades to `current_content` — the **raw accumulated LLM output** including
literal `<thought>`, `</thought>`, `<feelings>`, `<content>` tags.

This raw text is sent to Telegram **without** `parse_mode="html"`, so
XML tags render as literal visible text.

### Step 3: Next Edit Succeeds → Oscillation

On the next streaming update (+60 chars), the HTML might be valid again
(e.g., a previously-dangling structure got closed by new content), so
Telegram accepts the edit. The blockquote renders correctly.

Then the next update produces slightly different HTML that Telegram rejects
again → fallback → raw text visible again.

**Result**: Visible oscillation between parsed and raw states.

---

## Why the Previous Fix Didn't Help

Commit `5791f82` introduced `stabilize_markdown_for_streaming()`, which:

- Neutralizes unpaired `**`, `*`, and `~~` markers
- Replaces `*` inside unclosed `**` regions with `∗` (U+2217)
- Strips trailing unpaired markers

This fixed **formatting oscillation within a blockquote** (italic appearing
then disappearing as bold markers arrived incrementally). It operates on the
*content* passed to `markdown_to_html`.

The current bug is a **structural layout flip** — the entire blockquote
structure disappears. It occurs at the Telegram API level (message edit
rejection), not at the markdown conversion level. The stabilization function
cannot prevent Telegram from rejecting complex HTML, and it does not affect
the fallback text construction.

---

## Fix Strategy

### Priority 1: Sanitize the Fallback Text

The fallback must never expose raw XML template structure. When the active
field is a non-content field (thought, feelings), the fallback should be
derived from the *parsed* field content, not the raw LLM output.

```python
# Instead of:
fallback = (remaining or current_content)[:max_len] + " ▍"

# Use a cleaned version:
fallback = self._build_clean_fallback(remaining, current_content, result, template)
```

The clean fallback should:
1. Strip XML tags from `current_content`
2. Or use `result.active_content` (already parsed out of XML) as the
   fallback source
3. Prefix with field label in plain text (e.g., "💭 Thought:\n...")

### Priority 2: Prevent Telegram Rejection

Reduce the chance that Telegram rejects the HTML edit:

- **Cap inner HTML length** before wrapping in blockquote — truncate thought
  display to stay well under 4096 chars including tag overhead
- **Sanitize nested blockquotes** — if `markdown_to_html` produces
  `<blockquote>` inside thought content, flatten or escape them since
  Telegram may reject nested `<blockquote>` tags
- **Validate HTML before sending** — quick check for tag balance before
  attempting the Telegram API call; if validation fails, use the clean
  fallback proactively rather than waiting for Telegram to reject

### Priority 3: Investigate Telegram Rejection Logging

Add more detailed logging when `edit_message` fails:

```python
if not edit_result.success:
    logger.warning(
        "HTML edit rejected (len=%d, first_100=%r): %s",
        len(live_text), live_text[:100], edit_result.error,
    )
```

This will help identify the exact Telegram error messages that trigger
fallback, enabling targeted fixes.

---

## Testing Plan

1. **Integration test with long thought**: Create a streaming test where
   the thought field exceeds ~2000 characters. Verify that every edit in
   `stream_debug` output either contains proper blockquote structure OR
   a clean plain-text fallback (never raw XML tags).

2. **Test fallback content**: When `_edit_existing` falls back, assert the
   fallback text does not contain `<thought>` or `</thought>` as literal
   strings.

3. **Test nested blockquote handling**: Stream thought content containing
   `> quoted text` markdown. Verify the resulting HTML doesn't nest
   `<blockquote>` inside `<blockquote>`.

4. **Manual e2e test**: Use `mai-chat` with a prompt that elicits a long
   thought, observe that no raw XML tags are ever visible during streaming.

---

## Related Files

| File | Role |
|------|------|
| `src/mai_gram/bot/stream_display.py` | Core streaming display logic, fallback construction |
| `src/mai_gram/core/md_to_telegram.py` | `markdown_to_html`, `stabilize_markdown_for_streaming` |
| `src/mai_gram/core/md_to_telegram_html.py` | HTML-specific markdown conversion, stabilization impl |
| `src/mai_gram/response_templates/xml_template.py` | `parse_streaming`, `render_field_html` |
| `src/mai_gram/messenger/telegram.py` | `edit_message` — where Telegram API rejection occurs |
| `tests/test_integration/test_template_streaming.py` | Streaming integration tests |

---

## Key Insight

The fundamental design problem was that **the fallback path was designed for
the non-template case** (where `remaining` contains raw markdown user content)
and was never adapted for template-aware streaming. In template mode, when a
non-content field (like thought) is active, the content goes into
`header_html` and `remaining` becomes empty, causing the fallback to
silently degrade to the raw LLM output — the one string that should *never*
be shown to the user.

## Final Fix (Phase 2)

The initial fix (Phase 1) replaced raw XML in fallback with parsed field
content (`fallback_source`). However, this introduced a new flicker: the
fallback still showed degraded plain-text (or cursor-only blank messages)
alternating with properly formatted blockquotes.

**Phase 2 eliminates the fallback entirely for existing messages.**
When `_edit_existing` receives `success=False` from the HTML edit, it
simply **returns the existing placeholder_msg_id** without sending any
fallback. The last successfully rendered message stays visible. The next
streaming tick (≥60 chars or ≥1s later) retries with fresh HTML that may
succeed. This produces zero flicker — the user only ever sees successful
HTML renders or the most recent successful render frozen in place.

The `_send_new` path still uses fallback for the initial message send
(since there's no previous render to preserve).
