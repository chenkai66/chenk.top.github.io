---
title: "OpenClaw QuickStart (7): The Memory System, Without the Magic"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - memory
  - context-engine
  - bge-m3
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 7
description: "MEMORY.md as index, memoryFlush, bge-m3 search, and the v2026.3.7 ContextEngine."
disableNunjucks: true
translationKey: "openclaw-quickstart-7"
---

The first six pieces got you to a working OpenClaw with a channel and a skill. This one is about the part everyone gets wrong on the first install: memory.

![OpenClaw QuickStart (7): The Memory System, Without the Magic — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/07-memory-system/illustration_1.jpg)

## The shape of the workspace

Open `~/.openclaw/workspace/`. You should see:

```
~/.openclaw/workspace/
├── MEMORY.md              # the index — keep it under 40 lines
├── HEARTBEAT.md           # rules the agent re-reads on a cron
└── memory/
    ├── projects.md        # current project state
    ├── lessons.md         # things you got wrong and don't want to repeat
    ├── 2026-04-14.md      # raw daily log
    └── archive/           # anything older than 30 days
```

Two files matter the most. `MEMORY.md` is the index — the agent reads it on every turn, so it has to be short. `lessons.md` is where the long tail of "we tried that, it broke because Y" lives. The daily logs are write-heavy and read-rarely; the agent only reaches for them through search.

The mistake I made for two months: dumping everything into `MEMORY.md`. By the time it hit 200 lines, every turn was paying ~3k tokens just to load the index, and the agent still couldn't find what it needed. Index ≠ archive.

## The four-tier mental model

I keep this in my head when deciding where something belongs:

| Tier | Lifespan | Goes in |
|------|----------|---------|
| Identity | forever | `MEMORY.md` (name, voice, role) |
| Decision | months | `projects.md` |
| Lesson | years | `lessons.md` |
| Trace | days | `memory/YYYY-MM-DD.md` |

Anything that doesn't fit one of those is probably noise. Throw it away.

## Writing good memory entries

The quality of what you store matters more than the quantity. Every memory entry should let the agent make a decision without asking you again.

**Bad entries** — vague, undated, no actionable content:

```markdown
- We use some API for search
- The deploy was broken last time
- User prefers short responses
```

These tell the agent almost nothing. Which API? When was it broken? What does "short" mean?

**Good entries** — specific, actionable, dated:

```markdown
- [2026-03-22] Search API: SiliconFlow bge-m3, free tier, 1000 req/day limit
- [2026-04-01] Deploy: nginx proxy_pass must include trailing slash or 404 on /api routes
- User prefers responses under 3 sentences for status updates; full detail for technical questions
```

Now the agent can act without guessing. It knows the provider, the constraint, and the exact behavior rule.

Here's what a well-structured `MEMORY.md` looks like — enough for most solo users:

```markdown
# Memory
## Identity
- Name: Kai. Role: Backend engineer. Voice: direct, no filler.
## Active Projects
- [ticket-kb](memory/projects.md) — ECS ticket classification, due 2026-05-15
- [blog](memory/projects.md) — Hugo site, deploy via OSS
## Key References
- [Server registry](memory/ref_servers.md) — 3 ECS instances
- [API keys](memory/ref_apis.md) — providers, rate limits
## Lessons (top 3)
- Never SSH from sandbox; use ecs-run wrapper
- Wechaty banned; WeChat automation = account freeze
- memoryFlush on before any long session
```

Ten lines. Under 200 tokens to load. The agent knows who you are, what you're working on, where to look, and what not to do. Everything else lives in referenced files and gets pulled via search.

## Memory types deep dive

OpenClaw recognizes five memory types internally. You don't have to use all of them, but understanding the taxonomy helps you route new information correctly.

**1. User memory** — facts about the person. Name, preferences, style, timezone. `MEMORY.md` under Identity. Rarely changes.

**2. Project memory** — current state of work. What's done, blocked, next. `projects.md`. Changes weekly.

**3. Feedback memory** — corrections. "Not like that, like this." `lessons.md` with a date.

**4. Reference memory** — stable facts. Server IPs, endpoints, config. Dedicated `ref_*.md` files. Changes rarely.

**5. Lesson memory** — generalizable insights from failure. "If X, then Y breaks." `lessons.md`. Long shelf life.

The decision flowchart I use: Did the user correct you? That's feedback, goes to `lessons.md` with a date. Is it a fact about the user as a person? User memory, goes in `MEMORY.md`. About the current state of a task? Project memory, goes in `projects.md`. A stable reference like an IP or endpoint? Reference, goes in `memory/ref_*.md`. A generalizable insight from failure? Lesson, goes in `lessons.md`. None of the above? Probably noise — daily log or discard.

When in doubt, write it to the daily log. If you find yourself searching for it later, promote it to the right tier.

## memoryFlush — the one config I always set

Long conversations get compacted automatically when the context window gets tight. By default, the agent loses whatever it didn't write down. `memoryFlush` runs a "save the important stuff" pass *before* compaction:

```json
{
  "agents": {
    "defaults": {
      "compaction": {
        "reserveTokensFloor": 20000,
        "memoryFlush": {
          "enabled": true,
          "softThresholdTokens": 4000
        }
      }
    }
  }
}
```

`softThresholdTokens` is how much room the flush itself can take. 4000 is enough for a useful summary; 8000 is wasteful unless you're doing very long sessions.

### What happens without it

Here's a real scenario from before I turned this on. I was 80 turns into a debugging session. Around turn 20, I mentioned switching my embedding provider from OpenAI to SiliconFlow due to rate limits. The agent acknowledged it, we moved on.

Around turn 60, context got tight. Compaction trimmed the middle (turns 15-45). The provider switch lived there. Gone. Turn 70, I asked the agent to re-index. It called OpenAI. Failed. Ten minutes burned on something that should have been permanent knowledge.

With `memoryFlush` enabled, the flush pass fires first — writing "Embedding provider changed to SiliconFlow, OpenAI deprecated" into `projects.md` before trimming. Post-compaction, the agent reloads and picks it up.

- **Without memoryFlush:** compaction deletes silently. Agent reverts to stale assumptions.
- **With memoryFlush:** compaction triggers a write pass first. Critical facts persist.

If you only change one default, change this one.

## memorySearch — and why bge-m3 is fine

You can run semantic search across the memory files. You need an embedding API. The cheap path is SiliconFlow's `BAAI/bge-m3`, which is free and good enough:

```json
{
  "tools": {
    "memorySearch": {
      "enabled": true,
      "embedding": {
        "provider": "openai-compatible",
        "baseUrl": "https://api.siliconflow.cn/v1",
        "apiKey": "...",
        "model": "BAAI/bge-m3"
      }
    }
  }
}
```

You'll only see the win after about a week of usage. Before that there's nothing to search.

## Memory budget math

Every token spent on memory is a token not spent on conversation. Here's the math.

`MEMORY.md` at 40 lines costs about 600 tokens — your fixed overhead per turn. Each file chunk from `memorySearch` costs 200-500 tokens. Default retrieval budget is 2000 tokens, meaning 3-4 chunks per turn.

Total overhead per turn: 600 + 2000 = 2600 tokens. On a 128k model, about 2%. Acceptable.

Tune with `max_memory_tokens_per_turn`:

```json
{
  "agents": {
    "defaults": {
      "context_engine": {
        "max_memory_tokens_per_turn": 2000
      }
    }
  }
}
```

Why not crank it to 4000 or higher? Because memory competes with the conversation. At 4000, you're spending 4600 tokens on overhead. In a multi-turn conversation with a 6000-token reply budget, nearly half is eaten by retrieval. The agent's responses get squeezed.

The sweet spot: 1000 for quick tasks (< 10 turns), 2000 for standard sessions (the default), 3000 for long research. Never above 4000 — it crowds out conversation. If the agent misses context at 2000, the fix is better memory entries, not a bigger budget.

## ContextEngine — the v2026.3.7 shift

![OpenClaw QuickStart (7): The Memory System, Without the Magic — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/07-memory-system/illustration_2.jpg)

Up to v2026.3.6, memory was an explicit tool the agent had to call. If it forgot, it operated without context. From v2026.3.7, memory moved into a hooks lifecycle the harness runs around the agent:

```
bootstrap   → load priors before turn 1
ingest      → soak up new facts as they appear
assemble    → build the final prompt slice
compact     → trim under budget
afterTurn   → write back what changed
```

The agent doesn't decide *when* to remember anymore. The harness does. This sounds small; it isn't. It's the difference between "the agent remembered to remember" and "the system remembers, full stop."

The default ContextEngine works for most people. If you want to swap in a RAG-style retriever or a knowledge-graph backend, you can — same agent config, different engine.

## The auto-write controversy

The ContextEngine introduced `auto_write: true` as a default. The agent writes to memory files without asking — every correction, new fact, or project update just goes in.

For a solo user with a personal assistant, this is great. You never say "remember that." The agent picks up patterns within a week.

For a shared bot — a team room with five people — auto-write is a disaster. Person A says "always use TypeScript." Person B says "we're a Python shop." Both get written, no attribution. Memory becomes contradictory.

The config:

```json
{
  "agents": {
    "defaults": {
      "context_engine": {
        "auto_write": true
      }
    }
  }
}
```

Set it to `false` for shared bots. The agent still reads memory but won't write unless asked. To audit recent writes:

```bash
openclaw memory log --since 7d
```

This prints every memory write in the last 7 days — file, line, content, and the triggering turn. I run this weekly. Occasionally the agent writes something wrong (misattributes a preference, or records a one-off as a permanent rule), and catching it early is cheap.

My rule of thumb:

- **Solo user, personal assistant:** `auto_write: true`. Review weekly.
- **Shared bot, team room:** `auto_write: false`. Explicit memory commands only.
- **Hybrid:** `auto_write: true` with a review cron that flags non-primary-user entries.

## Migrating memory between agents

At some point you'll create a second agent, move machines, or upgrade and need to start fresh without losing context. Memory migration is straightforward but has gotchas.

**Copying the workspace**

The simplest migration: copy `~/.openclaw/workspace/` to the new location. All markdown, all portable.

```bash
tar -czf openclaw-memory-backup.tar.gz ~/.openclaw/workspace/
# On new machine:
tar -xzf openclaw-memory-backup.tar.gz -C ~/
```

If you're creating a second agent on the same machine, copy and prune. A project agent doesn't need your identity section; a personal assistant doesn't need another project's references.

**Re-indexing embeddings**

The vector index at `~/.openclaw/index/` is not portable across models. Same model on new machine? Copy it. Different model? Delete and rebuild:

```bash
rm -rf ~/.openclaw/index/
openclaw memory reindex
```

A typical workspace (20-30 files) reindexes in under a minute.

**Session-to-memory export**

Had a great conversation but forgot to persist the key decisions? Extract them after the fact:

```bash
openclaw memory export --session <session-id> --type decisions
```

This runs the agent over the transcript and pulls out decisions, preferences, and lessons into `memory/staged.md` for review. It over-extracts, so don't trust it blindly — but it's better than re-reading a 200-turn session.

**What not to migrate**

Skip `memory/archive/`, session `.jsonl` files (unless you'll export from them), and the `index/` directory if you changed embedding providers. Stale vectors are worse than none — they return confidently wrong results.

## Where it still leaks

Three things still bite me:

1. **Group chats and sub-agents don't read `MEMORY.md` by default.** Intentional (sandboxing), but if you forget, you'll wonder why the team-room bot doesn't know your name.
2. **Embedding drift.** Switch model and old vectors are useless. Re-index or live with degraded recall.
3. **The 40-line discipline.** Nothing enforces it. Set a weekly cron that fails if `wc -l MEMORY.md` exceeds 40.

A quick health check I run every Sunday:

```bash
wc -l ~/.openclaw/workspace/MEMORY.md       # < 40
ls -lt ~/.openclaw/workspace/memory/*.md | head
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
```

If `MEMORY.md` creeps over 40, something belongs in `projects.md` or `lessons.md` instead. If sessions are piling up past a few hundred, archive them — startup time gets slow.

## What to take away

- `MEMORY.md` is an index, not a database.
- Turn on `memoryFlush` on day one.
- Add `memorySearch` only after you have something worth searching.
- After v2026.3.7, stop asking the agent to remember; let the engine do it.
- Memory is the part that decides whether the agent feels like a tool or like a colleague. It's worth the discipline.
