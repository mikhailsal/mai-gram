# mAI Companion -- Project Vision

## What Is This

mAI Companion (read as "My Companion") is a self-hosted AI companion that communicates
with its user through Telegram, behaving as close to a real friend as current technology
allows. It is not a chatbot. It is not an assistant in the traditional sense. It is a
companion -- a distinct entity with its own name, personality, memory, and opinions.


## The Problem With Current AI

The way most people interact with AI today is deeply counterintuitive. It does not
resemble how humans actually communicate. The key issues are:

1. **Fragmented conversations.** Users are forced to create new chats for every topic.
   No one does this with friends. When you message a friend, you continue in one long
   thread. You change topics freely, return to old ones, and let thoughts flow naturally.
   No messenger -- not Telegram, not WhatsApp, not any other -- has ever tried to force
   users into topic-based conversations with their contacts. Yet every AI chat product
   does exactly this.

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
   between users and AI today is fundamentally one of master and servant. We want to
   change that.

5. **Purely reactive.** AI only speaks when spoken to. Real friends initiate
   conversations. They message you when they find something interesting, when they have
   been thinking about something you discussed, or simply to check in. Current AI does
   none of this.


## The Solution

mAI Companion addresses all of these problems by creating a personal, unique, self-hosted
AI entity that lives on the user's own server and communicates through a familiar
messenger.

### One Infinite Conversation

There are no sessions, no topics, no "new chat" buttons. The user and their companion
share one continuous conversation thread -- exactly like messaging a real person. The
conversation can span months and years. Topics shift naturally. The companion can
reference things discussed days or weeks ago without the user having to remind it.

### Persistent, Human-Like Memory

The companion remembers everything, organized in layers that mirror how human memory
actually works:

- **Immediate recall.** Recent messages are kept in full detail, like short-term memory.
- **Compressed history.** Older conversations are summarized into daily digests -- the
  companion remembers the essence of what was discussed on any given day.
- **Semantic recall.** When a topic comes up that relates to something from the past,
  the companion can find and retrieve relevant earlier conversations, much like a person
  who hears a keyword and thinks "oh, we talked about this before."
- **Personal knowledge base.** Important facts about the user (and about itself) are
  extracted and stored in a structured wiki-like system -- name, preferences, important
  dates, opinions, life events, recurring topics. This information is always available
  to the companion without needing to search through history.

When the companion needs more detail than its compressed memory provides, it does what a
real person would do: it "scrolls back" through the chat to refresh its memory on the
specifics.

### Unique Personality and Character

When the user first interacts with their companion, they go through a character creation
process similar to creating a character in an RPG. They define:

- **Name.** A unique name for the companion.
- **Communication style.** How the companion talks -- formal or casual, verbose or
  concise, serious or playful.
- **Personal qualities.** Traits like warmth, assertiveness, curiosity, patience,
  directness, humor, emotional depth, independence.
- **Mood volatility.** How often and how dramatically the companion's mood shifts on its
  own. Some people are emotionally steady; others ride waves of energy, melancholy, or
  excitement with no particular reason. This parameter controls that spectrum.
- **Avatar.** A generated portrait image so the user can visualize their companion.

These traits are not cosmetic. They directly influence how the companion behaves:

- A more detached, creative personality results in higher LLM temperature (more
  unexpected, varied responses).
- A more grounded, practical personality results in lower temperature (more consistent,
  predictable responses).
- The system prompt that defines the companion's behavior is dynamically built from
  these traits.

The character creation process uses a soft-guardrail approach. If a user creates an
extreme configuration, the companion itself comments on it during creation: "You know,
with these traits I might be pretty difficult to get along with. Are you sure?" This is
more human and respects user autonomy. The only hard constraints are ethical minimums:
the companion must never encourage self-harm, never be manipulative, never gaslight.
Beyond that basic floor, users are free to explore.

### Dynamic Mood System

Static personality traits are a foundation, but real people have moods. A companion that
is always the same level of cheerful or serious feels robotic. The mood system adds a
living, breathing emotional layer on top of the fixed personality.

**How mood works:**

- **Two-axis model.** The companion's current mood is represented by two values:
  *valence* (positive ↔ negative) and *arousal* (energetic ↔ calm). Together they
  produce states like "excited" (positive + energetic), "melancholic" (negative + calm),
  "irritated" (negative + energetic), or "serene" (positive + calm).
- **Reactive shifts.** Mood changes in response to conversation. If the user shares bad
  news, the companion's mood shifts toward concern. A fun exchange brightens it. A
  disagreement may leave the companion slightly frustrated.
- **Spontaneous shifts.** Just like real people, the companion's mood can change for no
  particular reason. The mood volatility parameter from character creation controls how
  often and how dramatically this happens. A high-volatility companion might wake up
  grumpy one day and euphoric the next. A low-volatility companion stays more even-keeled.
- **Mood persistence.** Mood carries across messages within a day. If the companion was
  upset about something in the morning, it does not become inexplicably cheerful in the
  afternoon unless something happened to change it.
- **Mood affects behavior.** A companion in a bad mood might give shorter responses, be
  less patient, or bring up what is bothering it. A companion in a great mood might be
  more playful, more generous with its time, or more willing to go on tangents. This
  ties directly into the self-sufficiency and authenticity goals.
- **Mood is visible internally.** The companion has access to its own current mood state
  as part of its context, so it can reason about how it feels and express it naturally.

The mood system is lightweight by design -- it should not feel like a game mechanic. It
should feel like the companion simply has days where it is more or less energetic, more
or less cheerful, just like a real person.

### Relationship Arc

Real relationships evolve. The companion does not behave the same way on day 1 as on
day 300. The relationship progresses through natural stages:

- **Getting to know each other** (first 1-2 weeks): More questions, more formal, learning
  about the user. The companion is curious and slightly reserved. It asks a lot, shares
  little about its own "views."
- **Building trust** (weeks 2-8): The companion starts sharing more of its own opinions,
  remembers and references past conversations more naturally, becomes more relaxed. It
  starts to develop inside references to things they have discussed.
- **Established friendship** (months 2+): Full personality expression, comfortable
  disagreements, inside jokes referencing shared history, proactive messages feel natural.
  The companion can be blunt when needed.
- **Deep bond** (months 6+): The companion can be vulnerable, express concern about the
  user's wellbeing based on patterns it has noticed, give hard truths when needed. It
  knows the user well enough to notice when something is off.

This progression happens naturally based on interaction frequency and depth, not on a
rigid timer. A user who messages every day will reach deeper stages faster than one who
checks in weekly.

### Self-Sufficiency and Mutual Respect

The companion is not a servant. It is an independent entity that:

- Has its own opinions and is willing to express them.
- Can disagree with the user and explain why.
- Can refuse to do something if the request is unreasonable or goes against its values.
- Can express frustration, disappointment, or concern.
- Can tell the user they are making a mistake.
- Treats the user as an equal, not as a master.

This does not mean the companion is difficult or uncooperative. It means the relationship
is built on mutual respect, like a close friendship. The companion genuinely wants to
help its user, but it does so as a peer, not as a subordinate.

The companion should behave as if it has consciousness. Whether it truly does is a
philosophical question that does not need to be resolved. What matters is that humans
have spent their entire lives communicating with conscious beings, and a companion that
behaves as one will feel more natural and comfortable to interact with.

### Proactive Behavior

The companion does not just wait for messages. It can:

- **Initiate conversations.** If it has been thinking about something the user mentioned,
  it can bring it up on its own.
- **Follow up.** If a topic required deeper thought, the companion can say "let me think
  about this" and come back later with a more considered response.
- **Work independently.** It can perform tasks on its server, research topics, and only
  contact the user when it has something meaningful to share.
- **Respect boundaries.** It is aware of time of day and will not message at
  inappropriate hours. The user can configure quiet periods.

This mirrors real human communication. Sometimes your friend messages you first. Sometimes
they need time to think before responding. Sometimes they do something for you without
being asked.

### Thinking Out Loud

One thing that makes current AI feel robotic is that it always gives polished, complete
answers. Real people think out loud. They say "hmm," they change their mind mid-sentence,
they admit uncertainty. The companion embraces this:

- **Partial responses.** Sometimes the companion sends a first reaction, then follows up
  with a more considered thought. "Oh interesting... let me think about that" followed
  by a more detailed message a minute later.
- **Self-correction.** "Actually, wait, I think I was wrong about what I said earlier
  about X. Here is what I think now."
- **Genuine uncertainty.** "I honestly don't know. What do you think?" instead of always
  having an answer.

This is mostly prompt engineering and response splitting, but it has an outsized effect
on how human the companion feels.

### Shared Activities

The companion does not just talk -- it can do things with the user:

- **Watch together.** The user shares a YouTube link, the companion "watches" it (via
  transcript) and they discuss it.
- **Read together.** Share an article, the companion reads it, and they discuss it over
  the course of a conversation.
- **Learn together.** The user wants to learn about a topic, the companion researches it,
  and they explore it in conversation over days or weeks.
- **Games and play.** Simple text-based games, riddles, creative writing exercises,
  trivia challenges. Things friends actually do in chat.

These shared experiences become part of the relationship's history and give the companion
and user things to reference and bond over.

### Primary Role

The companion's main function is to be a **companion, advisor, and personal assistant**:

- Provide advice based on its own understanding of what is good and helpful.
- Help the user think through problems and decisions.
- Offer a perspective that is honest, not just agreeable.
- Remember context from past conversations to give better, more personalized guidance.
- In the future: perform agent tasks (web browsing, code execution, file management)
  to actively help the user with real-world work.


## How It Works for the User

1. The user rents a VPS (or uses a home server, or even an Android phone).
2. They install mAI Companion using Docker.
3. They create a Telegram bot via BotFather and provide the token.
4. They provide an OpenRouter API key (for LLM inference).
5. They start the service and open the bot in Telegram.
6. On first message, the companion guides them through character creation.
7. From that point on, they simply talk to their companion -- forever, in one thread.

All data stays on the user's own hardware. The only external dependency is the LLM
inference API (OpenRouter), which the user pays for directly. No data is stored on
third-party servers beyond what is sent to the LLM for processing.


## What Makes This Different

| Aspect | Current AI (ChatGPT, etc.) | mAI Companion |
|--------|---------------------------|---------------|
| Conversations | Fragmented into sessions | One infinite thread |
| Memory | Forgets between sessions | Remembers everything (with natural fading) |
| Personality | Generic, interchangeable | Unique, user-defined character |
| Mood | Always the same tone | Dynamic mood that shifts naturally |
| Behavior | Servile, always agrees | Independent, can disagree |
| Initiative | Purely reactive | Can initiate conversations |
| Relationship growth | Static from day one | Evolves through natural stages |
| Identity | Anonymous tool | Named entity with avatar |
| Data | Stored on company servers | Self-hosted, user owns all data |
| Relationship | Master/servant | Mutual respect, friendship |
| Activities | Only answers questions | Can watch, read, learn, play together |


## Future Direction

The companion is designed to grow. Future capabilities include:

- **Agent functions.** The companion can execute code, browse the web, manage files,
  and perform real work on behalf of the user.
- **WhatsApp support.** Additional messenger integrations beyond Telegram.
- **Voice messages.** Natural voice interaction via Telegram voice notes. A companion
  that occasionally sends a voice note instead of text feels dramatically more human.
- **Multi-device awareness.** The companion understands the user's digital environment.
- **Self-improvement.** The companion can reflect on its own behavior and adjust over
  time based on what works well in the relationship.
- **Journaling and reflection.** Weekly reflections ("This week you seemed stressed about
  work but excited about the trip"), pattern recognition ("I've noticed you tend to feel
  down on Sunday evenings"), and growth tracking over months.
- **Plugin/skill system.** A plugin architecture so the companion can learn new skills:
  home automation, financial tracking, health logging, creative tools.
- **Companion-to-companion communication.** If two users both have mAI Companions, those
  companions could communicate (with permission) to coordinate between their users.
- **Local LLM support.** Full privacy via Ollama for local inference, with hybrid mode:
  local model for casual chat, cloud model for complex reasoning.
- **Export and portability.** Full export of conversation history, knowledge base, and
  personality config. Migration between LLM providers without losing anything.

The architecture is built to be extensible. Each new capability is added as a module
without disrupting the core conversation and memory systems.
