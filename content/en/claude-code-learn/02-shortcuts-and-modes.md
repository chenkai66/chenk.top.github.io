---
title: "Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes"
date: 2026-04-19 09:00:00
tags:
  - claude-code
  - shortcuts
  - thinking-modes
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 2
description: "Shift+Tab is a four-state cycle, not a binary. Thinking modes have five levels. Escape and double-Escape do different things. Five shortcuts that reshape how Claude Code feels day-to-day."
disableNunjucks: true
translationKey: "claude-code-learn-2"
---

The shortcuts are not in the help screen for a reason — they're discoverable through use, not documentation. Here they are anyway.

![Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/02-shortcuts-and-modes/illustration_1.png)

## `Shift+Tab` — the four-state cycle

Most people think `Shift+Tab` toggles auto-accept on and off. It doesn't. It cycles four states in order:

```
Normal  →  Accept edits  →  Plan mode  →  Bypass permissions  →  Normal
```

Each state changes what Claude does without asking:

| State | What changes |
|-------|--------------|
| **Normal** | Default. Asks before file writes and shell commands. |
| **Accept edits** | Auto-accepts file edits. Still asks for shell commands. |
| **Plan mode** | Builds a plan but does not execute. Useful for reviewing the approach before any side effects. |
| **Bypass permissions** | Skips all confirmation prompts. Use when you trust the current task. |

The status bar tells you which state you're in. `Shift+Tab` cycles. There is no state I use most — they correspond to different tasks. Plan mode is what I use when starting an unknown change. Accept edits is what I use during a long refactor where I'm watching the output anyway. Bypass permissions is what I reach for when I'm running a known, scoped script — never blindly.

## Thinking modes — five levels

![Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/02-shortcuts-and-modes/illustration_2.png)

Type one of these phrases anywhere in your prompt and Claude shifts how much it reasons before responding:

| Phrase | Approximate effort |
|--------|-------------------|
| `think` | Light extra reasoning |
| `think more` | A few seconds more |
| `think a lot` | Noticeably slower, more depth |
| `think longer` | Adequate for architecture decisions |
| `ultrathink` | The maximum |

The cost scales with the depth — each level uses more tokens. The right level is task-dependent:

- "Fix this typo" — no thinking phrase needed
- "Find why this test is flaky" — `think more`
- "Refactor this module" — `think a lot`
- "Should we move this from REST to gRPC" — `ultrathink`

I default to `think a lot` for anything I'd describe as "non-trivial." Below that I just type the prompt. Above, I'm asking a different question and `ultrathink` is honest about that.

The trap is using `ultrathink` on every prompt because it sounds rigorous. It's not — it's expensive. The model does best at the level that matches the actual task complexity.

## `Control+V` — paste images

Paste an image directly into the prompt. UI screenshot, error dialog, design comp, whiteboard photo — Claude reads it like text. The most useful version of this is "the test failed and the output is in this terminal screenshot, what's wrong?". You don't need to transcribe.

## `Escape` and double-`Escape`

A single `Escape` interrupts the current generation. Use it the moment you realize you gave the wrong instruction. The agent stops, you correct.

Double-`Escape` (press it twice quickly) opens a history view of the last few exchanges. You pick one to "rewind" to, and the conversation forks from there. Anything after the picked turn is dropped from context.

This is the feature I miss most when using other coding agents. Forking conversations is the right primitive — it lets you try one path, decide it was wrong, and go back without polluting the agent's memory.

## `/compact` and `/clear`

Two ways to deal with a long conversation:

- `/compact` summarizes the conversation so far into a much shorter message and continues from there. Use when the agent is starting to feel slow because the context is large.
- `/clear` drops the conversation entirely and starts fresh. Use when you've moved on to an unrelated task in the same project.

Both keep `CLAUDE.md` and `.claude/settings.json` in scope. They only affect the per-session message history.

## What this looks like in practice

A real workflow I had this morning:

```
> @src/api/handlers.ts
> think a lot — the team wants to add idempotency keys to the
> three POST handlers in this file. propose an approach that
> reuses the existing middleware pattern.

[Claude proposes — I read it]

[Shift+Tab to plan mode]

> ok, write it out as a plan, don't change anything yet.

[Claude lays out the steps]

[Shift+Tab to accept edits]

> go.

[Claude makes the changes, I watch them scroll]

[Shift+Tab back to normal mode for the test run]

> run the tests.

[Claude pauses to ask — yes — runs them — three pass, one fails]

[Double-Escape to fork back to before the test run]

> the failing test is testing the old non-idempotent path.
> update it to verify the new behavior.
```

That whole flow uses every primitive in this piece. None of them are individually impressive. Together they make Claude Code feel like a real tool instead of an autocomplete on top of a chat.

Next piece: custom slash commands and `$ARGUMENTS`.
