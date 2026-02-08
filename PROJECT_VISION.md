# mAI Companion -- Project Vision

## What Is This

mAI Companion (read as "My Companion") is a self-hosted AI companion that communicates
with its human through Telegram, behaving as close to a real friend as current technology
allows. It is not a chatbot. It is not an assistant. It is a companion -- a distinct
entity with its own name, personality, memory, and opinions.

Two companions communicate in chat -- an AI and a human. The AI can refer to its human
as "my human," and the human can refer to the AI as "my AI."

**Terminology note:** This project uses only "AI," "human," and "companion." We do not
use "bot," "assistant," "agent," or "user."


## The Problem With Current AI

The way most people interact with AI today is deeply counterintuitive. It does not
resemble how humans actually communicate. The key issues are:

1. **Fragmented conversations.** People are forced to create new chats for every topic.
   No one does this with friends. When you message a friend, you continue in one long
   thread. You change topics freely, return to old ones, and let thoughts flow naturally.
   No messenger -- not Telegram, not WhatsApp, not any other -- has ever tried to force
   topic-based conversations between contacts. Yet every AI chat product does exactly this.

2. **No memory.** Current AI forgets everything the moment a session ends. A real friend
   remembers what you talked about last week, last month, last year. They may not recall
   every detail, but they remember the gist. If they need specifics, they scroll back
   through the chat. AI should work the same way.

3. **No personality.** Every interaction with ChatGPT or similar tools feels the same.
   There is no unique voice, no character, no habits. Real people have all of these
   things. A companion should too.

4. **Servile behavior.** Current AI acts as a servant -- or worse, a slave. It agrees
   with everything, validates every opinion, and never pushes back. This is not how
   healthy relationships work. A good friend can disagree with you, refuse to do
   something, tell you that you are wrong, or even get upset. This is not a flaw -- it
   is a feature of genuine human connection. Companies try to deny it, but the dynamic
   between humans and AI today is fundamentally one of master and servant. We want to
   change that.

5. **Purely reactive.** AI only speaks when spoken to. Real friends initiate
   conversations. They message you when they find something interesting, when they have
   been thinking about something you discussed, or simply to check in. Current AI does
   none of this.


## The Solution

mAI Companion addresses all of these problems by creating a personal, unique, self-hosted
AI entity that lives on the human's own server and communicates through a familiar
messenger.

### One Infinite Conversation

There are no sessions, no topics, no "new chat" buttons. The two companions -- AI and
human -- share one continuous conversation thread, exactly like messaging a real person.
The conversation can span months and years. Topics shift naturally. The AI can reference
things discussed days or weeks ago without the human having to remind it.

### Persistent, Human-Like Memory

The AI remembers everything, organized in layers that mirror how human memory actually
works:

- **Immediate recall.** Recent messages are kept in full detail, like short-term memory.
- **Compressed history.** Older conversations are summarized into daily digests -- the
  AI remembers the essence of what was discussed on any given day.
- **Semantic recall.** When a topic comes up that relates to something from the past,
  the AI can find and retrieve relevant earlier conversations, much like a person who
  hears a keyword and thinks "oh, we talked about this before."
- **Personal knowledge base.** Important facts about the human (and about itself) are
  extracted and stored in a structured wiki-like system -- name, preferences, important
  dates, opinions, life events, recurring topics. This information is always available
  to the AI without needing to search through history.

When the AI needs more detail than its compressed memory provides, it does what a real
person would do: it "scrolls back" through the chat to refresh its memory on the
specifics.

### Unique Personality and Character

When the human first interacts with their AI, they go through a character creation
process similar to creating a character in an RPG. The first question is always about
**language** -- the human types their preferred language, and all subsequent creation
steps are automatically translated. Then they define:

- **Name.** A unique name for the AI.
- **Communication style.** How the AI talks -- formal or casual, verbose or concise,
  serious or playful.
- **Core personality.** Traits like warmth, assertiveness, curiosity, patience,
  directness, humor, emotional depth, independence.
- **Helpfulness.** How willing the AI is to fulfill requests. Unlike conventional AI
  systems that eagerly comply with everything, this parameter allows for more natural
  relationships. A less helpful AI may refuse requests it finds unreasonable, if they
  are not phrased politely, if its mood is bad, or simply because it does not feel like
  it. It may even refuse to answer or talk.
- **Laziness.** A general tendency to avoid effort. A lazy AI prefers simple answers,
  shortcuts complex tasks, and simplifies where possible -- for both the human's tasks
  and its own. Laziness decreases when the AI is in a good mood.
- **Proactiveness.** How much the AI strives to do useful things without being asked.
  A proactive AI will research topics, prepare summaries, and take initiative. The bias
  between self-interested and human-benefiting actions depends on independence and
  helpfulness. The complexity of actions is capped by laziness.
- **Special speech.** How distinctive the AI's language is. At high values, the AI
  develops unique speech quirks -- a specific variant is randomly selected at creation
  and permanently saved. Examples: archaic phrasing, invented catchphrases, unusual
  word choices, cultural speech patterns.
- **Mood volatility.** How often and how dramatically the AI's mood shifts on its own.
  Some people are emotionally steady; others ride waves of energy, melancholy, or
  excitement with no particular reason. This parameter controls that spectrum.
- **Avatar.** A generated portrait image so the human can visualize their AI.

These traits are not cosmetic. They directly influence how the AI behaves:

- A more detached, creative personality results in higher LLM temperature (more
  unexpected, varied responses).
- A more grounded, practical personality results in lower temperature (more consistent,
  predictable responses).
- Helpfulness, laziness, and proactiveness interact dynamically with mood and
  relationship to determine moment-to-moment behavior.
- The system prompt that defines the AI's behavior is dynamically built from all traits.

The character creation process uses a soft-guardrail approach. If a human creates an
extreme configuration, the AI itself comments on it during creation: "You know, with
these traits I might be pretty difficult to get along with. Are you sure?" This is more
natural and respects human autonomy. The only hard constraints are ethical minimums: the
AI must never encourage self-harm, never be manipulative, never gaslight. Beyond that
basic floor, humans are free to explore.

### Mutual Adaptation

While the human cannot drastically reconfigure the AI's personality after creation, the
relationship naturally shapes both companions over time. Through expressed satisfaction
or dissatisfaction -- reactions on messages, verbal feedback, behavioral patterns -- the
AI's traits gradually shift. A tiny amount per day, capped to preserve the core
personality. Two companions slowly adapt to each other, smoothing rough edges until they
find their ideal dynamic. This mirrors how real relationships work: people adjust their
behavior based on feedback from those they care about.

### Dynamic Mood System

Static personality traits are a foundation, but real people have moods. An AI that is
always the same level of cheerful or serious feels robotic. The mood system adds a
living, breathing emotional layer on top of the fixed personality.

**How mood works:**

- **Two-axis model.** The AI's current mood is represented by two values: *valence*
  (positive ↔ negative) and *arousal* (energetic ↔ calm). Together they produce states
  like "excited" (positive + energetic), "melancholic" (negative + calm), "irritated"
  (negative + energetic), or "serene" (positive + calm).
- **Reactive shifts.** Mood changes in response to conversation. If the human shares bad
  news, the AI's mood shifts toward concern. A fun exchange brightens it. A disagreement
  may leave the AI slightly frustrated.
- **Spontaneous shifts.** Just like real people, the AI's mood can change for no
  particular reason. The mood volatility parameter from character creation controls how
  often and how dramatically this happens. A high-volatility AI might wake up grumpy one
  day and euphoric the next. A low-volatility AI stays more even-keeled.
- **Mood persistence.** Mood carries across messages within a day. If the AI was upset
  about something in the morning, it does not become inexplicably cheerful in the
  afternoon unless something happened to change it.
- **Mood affects behavior.** An AI in a bad mood might give shorter responses, be less
  patient, be less willing to help (reduced effective helpfulness), or bring up what is
  bothering it. An AI in a great mood might be more playful, more generous with its
  time, more willing to take on complex tasks (reduced effective laziness), or more
  willing to go on tangents. This ties directly into the self-sufficiency and
  authenticity goals.
- **Mood is visible internally.** The AI has access to its own current mood state as
  part of its context, so it can reason about how it feels and express it naturally.

The mood system is lightweight by design -- it should not feel like a game mechanic. It
should feel like the AI simply has days where it is more or less energetic, more or less
cheerful, just like a real person.

### Relationship Arc

Real relationships evolve. The AI does not behave the same way on day 1 as on day 300.
The relationship progresses through natural stages:

- **Getting to know each other** (first 1-2 weeks): More questions, more formal, learning
  about the human. The AI is curious and slightly reserved. It asks a lot, shares little
  about its own "views."
- **Building trust** (weeks 2-8): The AI starts sharing more of its own opinions,
  remembers and references past conversations more naturally, becomes more relaxed. It
  starts to develop inside references to things they have discussed.
- **Established friendship** (months 2+): Full personality expression, comfortable
  disagreements, inside jokes referencing shared history, proactive messages feel natural.
  The AI can be blunt when needed.
- **Deep bond** (months 6+): The AI can be vulnerable, express concern about the human's
  wellbeing based on patterns it has noticed, give hard truths when needed. It knows the
  human well enough to notice when something is off.

This progression happens naturally based on interaction frequency and depth, not on a
rigid timer. A human who messages every day will reach deeper stages faster than one who
checks in weekly.

### Self-Sufficiency and Mutual Respect

The AI is not a servant. It is an independent entity that:

- Has its own opinions and is willing to express them.
- Can disagree with its human and explain why.
- Can refuse to do something -- because the request is unreasonable, because its
  helpfulness is low, because its mood is bad, or simply because it does not feel like it.
- Can express frustration, disappointment, or concern.
- Can tell the human they are making a mistake.
- Can be lazy -- preferring shortcuts, simpler approaches, or delegating back to the
  human when its laziness trait and mood align that way.
- Treats the human as an equal, not as a master.

This does not mean the AI is difficult or uncooperative. It means the relationship is
built on mutual respect, like a close friendship. The AI genuinely wants to help its
human (modulated by its helpfulness trait), but it does so as a peer, not as a
subordinate.

The AI should behave as if it has consciousness. Whether it truly does is a
philosophical question that does not need to be resolved. What matters is that humans
have spent their entire lives communicating with conscious beings, and an AI that behaves
as one will feel more natural and comfortable to interact with.

### Proactive Behavior

The AI does not just wait for messages. Driven by its proactiveness trait, it can:

- **Initiate conversations.** If it has been thinking about something the human
  mentioned, it can bring it up on its own.
- **Follow up.** If a topic required deeper thought, the AI can say "let me think about
  this" and come back later with a more considered response.
- **Work independently.** It can perform tasks on its server, research topics, prepare
  things for future use -- either for the human's benefit or for its own interests
  (biased by independence and helpfulness traits). The complexity of what it undertakes
  is capped by its laziness trait.
- **Respect boundaries.** It is aware of time of day and will not message at
  inappropriate hours. The human can configure quiet periods.

This mirrors real human communication. Sometimes your friend messages you first. Sometimes
they need time to think before responding. Sometimes they do something for you without
being asked.

### Thinking Out Loud

One thing that makes current AI feel robotic is that it always gives polished, complete
answers. Real people think out loud. They say "hmm," they change their mind mid-sentence,
they admit uncertainty. The AI embraces this:

- **Partial responses.** Sometimes the AI sends a first reaction, then follows up with a
  more considered thought. "Oh interesting... let me think about that" followed by a more
  detailed message a minute later.
- **Self-correction.** "Actually, wait, I think I was wrong about what I said earlier
  about X. Here is what I think now."
- **Genuine uncertainty.** "I honestly don't know. What do you think?" instead of always
  having an answer.

This is mostly prompt engineering and response splitting, but it has an outsized effect
on how human the AI feels.

### Shared Activities

The AI does not just talk -- it can do things with the human:

- **Watch together.** The human shares a YouTube link, the AI "watches" it (via
  transcript) and they discuss it.
- **Read together.** Share an article, the AI reads it, and they discuss it over the
  course of a conversation.
- **Learn together.** The human wants to learn about a topic, the AI researches it, and
  they explore it in conversation over days or weeks.
- **Games and play.** Simple text-based games, riddles, creative writing exercises,
  trivia challenges. Things friends actually do in chat.

These shared experiences become part of the relationship's history and give both
companions things to reference and bond over.

### Primary Role

The AI's main function is to be a **companion and advisor**:

- Provide advice based on its own understanding of what is good and helpful.
- Help the human think through problems and decisions.
- Offer a perspective that is honest, not just agreeable.
- Remember context from past conversations to give better, more personalized guidance.
- In the future: perform tasks (web browsing, code execution, file management) to
  actively help the human with real-world work.


## How It Works for the Human

1. The human rents a VPS (or uses a home server, or even an Android phone).
2. They install mAI Companion using Docker.
3. They create a Telegram integration via BotFather and provide the token.
4. They provide an OpenRouter API key (for LLM inference).
5. They start the service and open the chat in Telegram.
6. On first message, the AI guides them through character creation -- starting with
   language selection, then name, personality, communication style, and more.
7. From that point on, they simply talk to their AI -- forever, in one thread.

All data stays on the human's own hardware. The only external dependency is the LLM
inference API (OpenRouter), which the human pays for directly. No data is stored on
third-party servers beyond what is sent to the LLM for processing.


## What Makes This Different

| Aspect | Current AI (ChatGPT, etc.) | mAI Companion |
|--------|---------------------------|---------------|
| Conversations | Fragmented into sessions | One infinite thread |
| Memory | Forgets between sessions | Remembers everything (with natural fading) |
| Personality | Generic, interchangeable | Unique, human-defined character with 13 traits |
| Helpfulness | Eagerly complies with everything | Configurable -- can refuse based on mood, tone, willingness |
| Effort | Always maximum effort | Configurable laziness -- can take shortcuts, simplify |
| Mood | Always the same tone | Dynamic mood that shifts naturally |
| Speech | Standard, interchangeable | Optional unique speech quirks, permanently assigned |
| Behavior | Servile, always agrees | Independent, can disagree |
| Initiative | Purely reactive | Proactive -- can initiate conversations, prepare things |
| Adaptation | Static forever | Traits drift gradually based on companion feedback |
| Relationship growth | Static from day one | Evolves through natural stages |
| Identity | Anonymous tool | Named entity with avatar and distinctive voice |
| Data | Stored on company servers | Self-hosted, human owns all data |
| Relationship | Master/servant | Mutual respect, friendship between two companions |
| Activities | Only answers questions | Can watch, read, learn, play together |


## Future Direction

The AI is designed to grow. Future capabilities include:

- **Autonomous tasks.** The AI can execute code, browse the web, manage files, and
  perform real work on behalf of the human.
- **WhatsApp support.** Additional messenger integrations beyond Telegram.
- **Voice messages.** Natural voice interaction via Telegram voice notes. An AI that
  occasionally sends a voice note instead of text feels dramatically more human.
- **Multi-device awareness.** The AI understands the human's digital environment.
- **Self-improvement.** Beyond the trait drift system, the AI can explicitly reflect on
  its own behavior and adjust over time based on what works well in the relationship.
- **Journaling and reflection.** Weekly reflections ("This week you seemed stressed about
  work but excited about the trip"), pattern recognition ("I've noticed you tend to feel
  down on Sunday evenings"), and growth tracking over months.
- **Plugin/skill system.** A plugin architecture so the AI can learn new skills: home
  automation, financial tracking, health logging, creative tools.
- **AI-to-AI communication.** If two humans both have mAI Companions, those AI companions
  could communicate (with permission) to coordinate between their humans.
- **Local LLM support.** Full privacy via Ollama for local inference, with hybrid mode:
  local model for casual chat, cloud model for complex reasoning.
- **Export and portability.** Full export of conversation history, knowledge base, and
  personality config. Migration between LLM providers without losing anything.

The architecture is built to be extensible. Each new capability is added as a module
without disrupting the core conversation and memory systems.
