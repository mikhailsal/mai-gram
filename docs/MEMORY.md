# Memory System

> **How your AI companion remembers — and forgets.**

The memory system gives your companion persistent, human-like recall. Unlike typical AI that forgets everything between sessions, mAI Companion remembers your conversations, learns facts about you, and naturally forgets unimportant details over time.

---

## Overview

Memory is organized in layers, mirroring how human memory actually works:

| Layer | What It Stores | Retention |
|-------|----------------|-----------|
| **Short-Term** | Recent messages (last ~30) | Always in context |
| **Daily Summaries** | Compressed daily conversations | Permanent |
| **Weekly Summaries** | Aggregated weekly highlights | Permanent |
| **Monthly Summaries** | High-level monthly overview | Permanent |
| **Wiki (Knowledge Base)** | Important facts | Importance-weighted decay |

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Context Window                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  System Prompt                                          ││
│  │  (personality, mood, relationship stage)                ││
│  ├─────────────────────────────────────────────────────────┤│
│  │  Wiki Entries (top 20 by importance)                    ││
│  │  - Human's name (importance: 900)                       ││
│  │  - Human's job (importance: 700)                        ││
│  │  - Favorite food (importance: 500)                      ││
│  ├─────────────────────────────────────────────────────────┤│
│  │  Memory Summaries                                       ││
│  │  - Monthly: "January was focused on..."                 ││
│  │  - Weekly: "This week we discussed..."                  ││
│  │  - Daily: "Yesterday: talked about work stress"         ││
│  ├─────────────────────────────────────────────────────────┤│
│  │  Short-Term Messages                                    ││
│  │  [2024-01-15 10:30] Human: Hey, how are you?           ││
│  │  [2024-01-15 10:31] AI: Pretty good! How's your day?   ││
│  │  [2024-01-15 10:32] Human: Stressful, work is crazy    ││
│  │  ...                                                    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Short-Term Memory

The most recent messages are kept in full detail.

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SHORT_TERM_LIMIT` | 30 | Number of recent messages to include |

### How It Works

- Messages are stored in SQLite with full content
- Each message includes timestamp, role (human/AI), and content
- Timestamps are displayed in the context so the AI knows when things were said

### Example Context

```
[2024-01-15 10:30] Human: Hey, how's it going?
[2024-01-15 10:31] AI: Pretty good! Just thinking about that book you mentioned yesterday. Did you finish it?
[2024-01-15 10:32] Human: Not yet, work has been crazy this week
[2024-01-15 10:33] AI: Ah, the deadline you mentioned? How's that going?
```

---

## Daily Summaries

When a day has enough messages, they're compressed into a summary.

### When Summaries Trigger

| Setting | Default | Description |
|---------|---------|-------------|
| `SUMMARY_THRESHOLD` | 20 | Minimum messages before summarization |

Summaries are generated:
- Automatically when threshold is reached
- For previous days (not the current day)

### Storage

Summaries are stored as markdown files:

```
data/<companion-id>/summaries/daily/2024-01-15.md
```

### Example Summary

```markdown
Discussed work stress and the upcoming project deadline. Human mentioned 
feeling overwhelmed but determined to push through. We talked about 
strategies for managing workload — breaking tasks into smaller pieces 
and taking short breaks. Human also mentioned excitement about the 
Japan trip planned for next month.
```

### Summary Generation

The LLM generates summaries with this prompt structure:

```
Summarize this day's conversation between you (the AI companion) and 
your human. Capture:
- Main topics discussed
- Emotional tone
- Any important facts mentioned
- Unresolved threads or follow-ups

Keep it concise but complete enough to remind yourself later.
```

---

## Weekly & Monthly Summaries

Higher-level aggregations for long-term memory.

### Weekly Summaries

Generated from daily summaries:
- Aggregates 7 days of daily summaries
- Captures recurring themes and patterns
- Stored in `summaries/weekly/2024-W03.md`

### Monthly Summaries

Generated from weekly summaries:
- High-level overview of the month
- Major events and emotional arcs
- Stored in `summaries/monthly/2024-01.md`

### Summary Hierarchy

```
Monthly Summary (January 2024)
├── Weekly Summary (Week 1)
│   ├── Daily Summary (Jan 1)
│   ├── Daily Summary (Jan 2)
│   └── ...
├── Weekly Summary (Week 2)
│   └── ...
└── ...
```

---

## Wiki (Knowledge Base)

Structured storage for important facts.

### What Gets Stored

| Category | Examples | Typical Importance |
|----------|----------|-------------------|
| Identity | Name, birthday, location | 800-1000 |
| Relationships | Family members, friends | 600-800 |
| Preferences | Favorite food, hobbies | 400-600 |
| Work/Life | Job, projects, goals | 500-700 |
| Temporary | Current events, recent mentions | 200-400 |

### File Structure

Wiki entries are stored as markdown files with importance in the filename:

```
data/<companion-id>/wiki/
├── 0900_human-name.md           # "Alex"
├── 0850_human-birthday.md       # "March 15"
├── 0700_human-job.md            # "Software engineer at TechCorp"
├── 0500_favorite-food.md        # "Thai food, especially pad thai"
├── 0300_mentioned-coworker.md   # "Sarah - works on the same team"
└── 0200_current-project.md      # "Working on the Q1 release"
```

### Importance Scores

| Range | Meaning | Decay Rate |
|-------|---------|------------|
| 900-1000 | Critical (never forget) | Very slow |
| 700-899 | Important | Slow |
| 500-699 | Moderate | Normal |
| 300-499 | Minor | Fast |
| 100-299 | Trivial | Very fast |
| ≤0 | Deleted | — |

### Wiki Entry Format

```markdown
Alex

The human's name. They introduced themselves on the first day of our 
conversation. Prefers to be called Alex rather than Alexander.
```

### Context Inclusion

| Setting | Default | Description |
|---------|---------|-------------|
| `WIKI_CONTEXT_LIMIT` | 20 | Max wiki entries in prompt |

Top entries by importance are included in every prompt.

---

## Forgetting System

Unlike typical AI that remembers everything perfectly, mAI Companion implements natural forgetting.

### Why Forgetting Matters

An AI that remembers every detail perfectly is actually uncanny:
- "On March 15th at 2:47 PM you mentioned preferring Thai food"
- This level of precision feels robotic, not human

Natural forgetting makes the AI feel more authentic:
- "You mentioned preferring Thai food at some point"
- The gist remains, but specifics fade

### How Forgetting Works

**Importance Decay**

Wiki entries lose importance over time:

```python
# Daily decay formula
new_importance = old_importance - decay_amount

# Decay amount varies by importance tier
# Lower importance = faster decay
```

**Deletion Threshold**

When importance reaches 0, the entry is deleted:
- The AI "forgets" this fact
- It can be re-learned if mentioned again

**Protection**

High-importance entries (900+) decay very slowly:
- Human's name is effectively permanent
- Critical facts are protected

### Forgetting Configuration

| Setting | Description |
|---------|-------------|
| Decay rate | How fast importance decreases |
| Protection threshold | Importance level that slows decay |
| Minimum importance | Floor before deletion |

### Example Decay

```
Day 1: "Human mentioned coworker Sarah" (importance: 400)
Day 7: importance: 350
Day 14: importance: 300
Day 30: importance: 200
Day 60: importance: 50
Day 75: importance: 0 → DELETED
```

Meanwhile:
```
Day 1: "Human's name is Alex" (importance: 900)
Day 365: importance: 880 (barely changed)
```

---

## Semantic Search

For retrieving relevant past information.

### How It Works

When a topic comes up that might relate to past conversations:

1. Query is embedded using vector embeddings
2. ChromaDB searches for similar content
3. Relevant messages/summaries are retrieved
4. Added to context for the current response

### Use Cases

- Human mentions a topic discussed weeks ago
- AI needs to recall details not in short-term memory
- Finding patterns across conversation history

### Storage

Vector embeddings are stored in ChromaDB:

```
data/chroma_data/
```

---

## Token Budgeting

The context window has limits. Memory is prioritized:

### Priority Order

1. **System Prompt** — Always included (personality, mood)
2. **Wiki Entries** — Top 20 by importance
3. **Short-Term Messages** — Last 30 messages
4. **Daily Summaries** — Most recent first
5. **Weekly Summaries** — Most recent first
6. **Monthly Summaries** — Most recent first

### When Context Exceeds Budget

Oldest summaries are removed first:
1. Remove oldest monthly summaries
2. Remove oldest weekly summaries
3. Remove oldest daily summaries
4. Short-term messages are preserved

### Budget Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Max context tokens | 120,000 | Total token budget |
| Warning threshold | 90% | Log warning when exceeded |

---

## Memory Operations

### Explicit Remembering (Planned)

Humans will be able to pin important facts:

```
Human: Remember this — my flight is on Friday at 3 PM
AI: Got it, I'll remember your Friday 3 PM flight.
```

This creates a high-importance wiki entry.

### Memory Refresh

When the AI needs more detail than summaries provide, it can "scroll back" through the chat — just like a human would.

---

## Data Storage

### Database (SQLite)

```sql
-- Messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    companion_id TEXT,
    role TEXT,        -- 'user' or 'assistant'
    content TEXT,
    timestamp TIMESTAMP,
    is_proactive BOOLEAN
);

-- Knowledge Entries
CREATE TABLE knowledge_entries (
    id INTEGER PRIMARY KEY,
    companion_id TEXT,
    category TEXT,
    key TEXT,
    value TEXT,
    importance REAL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Files

```
data/<companion-id>/
├── wiki/
│   ├── 0900_human-name.md
│   └── ...
└── summaries/
    ├── daily/
    │   ├── 2024-01-15.md
    │   └── ...
    ├── weekly/
    │   └── 2024-W03.md
    └── monthly/
        └── 2024-01.md
```

---

## Backup Recommendations

All memory is stored locally. Regular backups are essential:

```bash
# Backup everything
cp -r data/ backup/data-$(date +%Y%m%d)/

# Or just the database
cp data/mai_companion.db backup/
```

Consider:
- Daily automated backups
- Off-site backup storage
- Version control for wiki files

---

## See Also

- [TERMINOLOGY.md](TERMINOLOGY.md) — Memory-related terms
- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical implementation
- [CONFIGURATION.md](CONFIGURATION.md) — Memory settings
