---
title: "Claude Code Hands-On (10): Skills, and When to Reach for Each Extension Mechanism"
date: 2026-04-25 09:00:00
tags:
  - claude-code
  - skills
  - slash-commands
  - mcp
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 10
description: "Skills are the newest extension mechanism in Claude Code: a folder, a SKILL.md, and a body of instructions the model loads on demand. How they differ from slash commands, MCP servers, and hooks — with a decision tree for which one to reach for."
disableNunjucks: true
translationKey: "claude-code-learn-10"
---

Claude Code now has four extension mechanisms: slash commands, MCP servers, hooks, and Skills. They overlap. The first time you have a "Claude should know how to do X" thought, the question is *which* of the four to use.

This is the closing chapter of the series. Let's lay out the decision tree.

![Claude Code Hands-On (10): Skills, and When to Reach for Each Extension Mechanism — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/illustration_1.png)

## What a Skill actually is

A Skill is a folder under `~/.claude/skills/<name>/` (user-level) or `<repo>/.claude/skills/<name>/` (project-level), containing at minimum a `SKILL.md`:

```markdown
---
name: chenk-blog-write
description: Use when writing new content for chenk.top — bilingual EN/ZH posts, series, tutorials. Covers front matter, voice, matplotlib figures, cover generation, deploy.
---

# Voice
- First person, dry, restrained. No "let's", no exclamations.
- One claim, one example. If the claim has no example, cut the claim.

# Front matter
[exact schema goes here]

# Workflow
1. Read source
2. Write EN
3. Adapt to ZH (not translate)
4. Generate covers
5. Build + deploy
```

When you start a session, Claude reads the *descriptions* of all available skills. When something you ask matches, Claude loads the skill body. The body becomes part of the system prompt for that turn.

Two things follow:

- The `description` is load-bearing. If it doesn't say when to use the skill, the skill won't get used.
- The body can be long. It's loaded on demand, so verbosity isn't taxed unless triggered.

## How Skills differ from the other three

| Mechanism | Lives in | Loaded when | Best for |
|---|---|---|---|
| Slash command | `<repo>/.claude/commands/<name>.md` | User types `/<name>` | Repeated workflows that take 1-2 lines to describe |
| MCP server | `mcp.json` config | Always available | Reaching outside the filesystem (browser, DB, third-party API) |
| Hook | `settings.json` referenced script | Around tool calls | Policy enforcement, side-effects on edit/write |
| Skill | `.claude/skills/<name>/SKILL.md` | Description matches the prompt | Domain knowledge, voice, multi-step procedures |

The clearest dividing line: **slash commands are commands, skills are knowledge**. A slash command is "do this exact thing." A skill is "here's how I think about this whole class of problem; use this whenever it applies."

## When to reach for which

![Claude Code Hands-On (10): Skills, and When to Reach for Each Extension Mechanism — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/illustration_2.png)

Walk through these in order:

**1. Does the task need a tool that doesn't exist yet?** (Browser automation, querying a real database, talking to an internal API.)
→ Build an **MCP server**. The other mechanisms can't grant new capabilities.

**2. Should something happen automatically around tool calls — block, validate, log, format?**
→ Write a **hook**. This is the only mechanism that runs without the model deciding to invoke it.

**3. Is this a compact procedure the user will explicitly invoke?** (`/commit`, `/deploy-staging`, `/make-changelog`)
→ Write a **slash command**. Commands are for things you call by name.

**4. Is this a body of domain knowledge — a voice, a workflow, a set of conventions — that should kick in whenever the topic comes up?**
→ Write a **skill**. Skills are for things you want Claude to *recognize and apply* without being told to.

If a thing fits two boxes, prefer the simpler one. A skill that calls a slash command is fine. A slash command that pretends to be a skill is brittle.

## Three skills I've actually written

**1. `chenk-blog-write`** — for writing on this site. Covers front matter, voice, EN/ZH parity, cover generation, deploy. Triggered by anything mentioning chenk.top or "write a post." The body is ~600 lines. Worth every one.

**2. `update-config`** — for changes to `~/.claude/settings.json`. Triggered by "allow X command," "set env Y," "add a hook." Encodes the permission precedence rules above and the typical patterns. Saves me from re-deriving the merge order.

**3. `simplify`** — for code review on my own changes. Triggered by "is there a simpler way to do this." Encodes my taste: prefer composition, kill dead code, name things by what they are not how they're built.

None of those would work as slash commands. They're not invoked by name; they're invoked by *topic*. That's the skill-shaped use case.

## When skills are the wrong answer

A skill that fires too often is worse than no skill — it pollutes context for tasks that don't need it. Three traps:

- **Vague descriptions.** "Use for general programming." Use for everything = use for nothing useful.
- **Skill bodies that overlap.** Two skills that both fire on "write code" → context bloat. Pick one.
- **Skills that should have been hooks.** "Always do X before Y" → that's a hook, not a skill. Skills suggest; hooks enforce.

## The end of the series

Ten chapters in, you have:

- A configured Claude Code with the three-layer settings model in your head (chs 1, 9).
- Fluency with shortcuts, modes, and conversation control (ch 2).
- The four extension mechanisms — slash commands, MCP, hooks, skills — and a decision tree for picking between them (chs 3, 4, 5, 7, 10).
- The concurrency primitives — sub-agents, worktrees, plan mode — for scaling individual sessions to bigger work (ch 8).
- A working SDK + GitHub integration story for putting Claude in CI (ch 6).

That is the surface area. Past it lies the actually interesting work, which is no longer about Claude Code itself — it's about what you build *with* it. Go build.
