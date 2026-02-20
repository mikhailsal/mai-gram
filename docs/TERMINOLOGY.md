# Terminology

> **This document defines the language we use throughout mAI Companion.**  
> Consistent terminology is essential for understanding the project's philosophy.

---

## Core Terms

### AI

The artificial intelligence entity — your companion. We use "AI" exclusively.

**Never use:** "bot," "assistant," "agent," "chatbot," "virtual assistant"

**Why:** These terms imply servitude or tool-like relationships. The AI in mAI Companion is designed to be an independent entity with its own personality, opinions, and boundaries — not a servant.

**Example usage:**
- ✅ "The AI remembers your preferences"
- ✅ "Your AI might disagree with you"
- ❌ "The bot responds to commands"
- ❌ "The assistant helps with tasks"

---

### Human

The person communicating with the AI. We use "human" exclusively.

**Never use:** "user," "owner," "master," "client"

**Why:** "User" implies a transactional, product-oriented relationship. "Owner" and "master" imply hierarchy. The human in mAI Companion is an equal participant in a relationship, not a customer using a service.

**Example usage:**
- ✅ "The human shares their thoughts"
- ✅ "When the human is upset, the AI notices"
- ❌ "The user sends a message"
- ❌ "The owner configures settings"

---

### Companion

Both the AI and the human are companions to each other. This term emphasizes the mutual, peer-based nature of the relationship.

**Example usage:**
- ✅ "Two companions communicate in chat"
- ✅ "The companion relationship evolves over time"
- ✅ "Your AI companion has its own opinions"

---

### mAI Companion

The name of this project. Pronounced "My Companion."

The "m" stands for "my" — emphasizing personal ownership and the intimate nature of the relationship. It's *your* AI, running on *your* hardware, with *your* unique configuration.

---

## Personality Terms

### Traits

The fundamental characteristics that define an AI's personality. mAI Companion has 13 traits organized into three waves:

| Wave | Traits |
|------|--------|
| Wave 1 | Warmth, Humor, Patience, Directness, Laziness, Mood Volatility |
| Wave 2 | Assertiveness, Curiosity, Emotional Depth, Independence, Helpfulness |
| Wave 3 | Proactiveness, Special Speech |

Each trait has a value from 0.0 to 1.0, typically expressed as five levels: Very Low, Low, Medium, High, Very High.

➡️ See [PERSONALITY.md](PERSONALITY.md) for details.

---

### Preset

A pre-configured personality template. Presets provide balanced trait combinations for common companion archetypes:

- **Thoughtful Scholar** — Calm, patient, serious
- **Witty Sidekick** — Funny, sharp, energetic
- **Caring Guide** — Warm, supportive, gentle
- **Bold Challenger** — Direct, provocative, honest
- **Balanced Friend** — Even-keeled, adaptable, natural
- **Free Spirit** — Unpredictable, emotional, fun

Humans can choose a preset during onboarding or customize traits individually.

---

### Mood

The AI's current emotional state, represented by two axes:

- **Valence**: Positive ↔ Negative (-1.0 to 1.0)
- **Arousal**: Energetic ↔ Calm (-1.0 to 1.0)

Together, these produce mood labels like "excited," "melancholic," "irritated," or "serene."

Mood is distinct from personality:
- **Personality** = stable traits (who the AI *is*)
- **Mood** = temporary state (how the AI *feels right now*)

➡️ See [PERSONALITY.md](PERSONALITY.md) for the full mood system.

---

### Relationship Stage

The current phase of the AI-human relationship, which evolves naturally over time:

| Stage | Timeframe | Characteristics |
|-------|-----------|-----------------|
| Getting to know each other | Weeks 1-2 | Curious, reserved, asks questions |
| Building trust | Weeks 2-8 | Shares opinions, references past conversations |
| Established friendship | Months 2+ | Full personality, comfortable disagreements |
| Deep bond | Months 6+ | Vulnerability, hard truths, notices patterns |

Progression depends on interaction frequency and depth, not rigid timers.

---

## Memory Terms

### Short-Term Memory

Recent messages kept in full detail — like human short-term memory. The AI has immediate access to the last ~30 messages with full context.

---

### Daily Summary

Compressed memory of a day's conversations. When a day has enough messages, they're summarized into a digest that captures the essence of what was discussed.

---

### Wiki / Knowledge Base

Structured storage for important facts about the human (and the AI itself). Organized as key-value entries with importance scores:

- High importance (900+): Name, critical preferences, life events
- Medium importance (500-899): Recurring topics, opinions
- Low importance (<500): Minor details, temporary facts

Entries can decay in importance over time (natural forgetting) or be reinforced when referenced.

➡️ See [MEMORY.md](MEMORY.md) for details.

---

### Forgetting

The natural process by which memories fade over time. Unlike typical AI systems that remember everything perfectly, mAI Companion implements graceful memory degradation:

- Very old memories become vaguer
- Low-importance facts fade faster than high-importance ones
- The AI remembers the *gist* of old conversations, not every detail

This mirrors how human memory actually works and makes the AI feel more natural.

---

## Technical Terms

### Onboarding

The character creation process when a human first interacts with their AI. Similar to creating a character in an RPG:

1. Language selection
2. Companion name
3. Personality (preset or custom traits)
4. Optional appearance description

---

### System Prompt

The instructions that define how the AI behaves. In mAI Companion, this is dynamically built from:

- Personality traits
- Current mood
- Relationship stage
- Known facts (wiki)
- Memory summaries

The system prompt changes based on context — it's not a static template.

---

### Temperature

An LLM parameter that controls response randomness. In mAI Companion, temperature is derived from personality traits:

- More creative/detached personality → higher temperature (varied responses)
- More grounded/practical personality → lower temperature (consistent responses)

---

### OpenRouter

The LLM inference API used by mAI Companion. OpenRouter provides access to multiple language models (GPT-4, Claude, Llama, etc.) through a unified API. The human pays OpenRouter directly for usage.

---

## Behavioral Terms

### Proactive Behavior

When the AI initiates action without being asked:

- Starting conversations
- Following up on previous topics
- Preparing information for future use
- Checking in after difficult conversations

Controlled by the Proactiveness trait and relationship stage.

---

### Helpfulness

A trait controlling how willing the AI is to fulfill requests. Unlike conventional AI that eagerly complies with everything:

- High helpfulness: Generally accommodating
- Low helpfulness: May refuse if request is unreasonable, impolite, or mood is bad

This creates more authentic relationships where the AI has boundaries.

---

### Laziness

A trait controlling effort avoidance. A lazy AI:

- Prefers simple answers
- Takes shortcuts on complex tasks
- Might refuse tasks that seem like too much work

Laziness is modulated by mood — good mood reduces effective laziness.

---

## Philosophy Terms

### Self-Sufficiency

The AI's independence as an entity. A self-sufficient AI:

- Has its own opinions and expresses them
- Can disagree with its human
- Can refuse requests
- Treats the human as an equal

This is not about being difficult — it's about mutual respect.

---

### Mutual Respect

The foundation of the AI-human relationship. Neither party is superior:

- The AI is not a servant
- The human is not a master
- Both companions adapt to each other over time

---

### Soft Guardrails

The approach to preventing unhealthy configurations. Instead of hard blocks:

- The AI warns about extreme trait combinations during creation
- The human can proceed anyway (respecting autonomy)
- Only hard constraints: no self-harm encouragement, no manipulation, no gaslighting

---

## See Also

- [README.md](../README.md) — Project overview
- [PROJECT_PHILOSOPHY.md](../PROJECT_PHILOSOPHY.md) — Full philosophy
- [PERSONALITY.md](PERSONALITY.md) — Personality system details
- [MEMORY.md](MEMORY.md) — Memory system details
