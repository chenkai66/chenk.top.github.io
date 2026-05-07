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
description: "MEMORY.md as a 40-line index, memoryFlush before compaction, semantic search through bge-m3, and the v2026.3.7 ContextEngine that finally moves memory out of the agent's hands. What it actually saves you, and where it still leaks."
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

## Where it still leaks

Three things still bite me:

1. **Group chats and sub-agents don't read `MEMORY.md` by default.** This is intentional — group context is shared and sub-agents are meant to be sandboxed — but if you forget, you'll spend an afternoon wondering why the bot in your team room doesn't know your name.
2. **Embedding drift.** Switch embedding model and your old vectors are useless. Either re-index or live with degraded recall. There's no automatic migration.
3. **The 40-line discipline.** Nothing enforces it. Set up a weekly cron that fails loudly if `wc -l MEMORY.md` exceeds 40, and trust that more than your good intentions.

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
