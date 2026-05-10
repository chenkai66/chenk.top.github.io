---
title: "Claude Code Hands-On (3): Custom Slash Commands and Conversation Control"
date: 2026-04-20 09:00:00
tags:
  - claude-code
  - slash-commands
  - workflow
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 3
description: "Slash commands turn repeated workflows into one-line invocations. $ARGUMENTS makes them parameterized. The right ones become your team's shared vocabulary."
disableNunjucks: true
translationKey: "claude-code-learn-3"
---

Built-in slash commands like `/clear` and `/init` are the visible part of the iceberg. The whole point of the system is that you write your own, and they live in your repo.

![Claude Code Hands-On (3): Custom Slash Commands and Conversation Control — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/illustration_1.png)

## What a slash command is

A file at `.claude/commands/<name>.md`. Contents are a Markdown prompt. Filename becomes the command. After creation you have to restart Claude Code (one of the few places it's not hot-reloaded).

The simplest possible example. Create `.claude/commands/audit.md`:

```markdown
Run `npm audit` to find vulnerable installed packages.
Run `npm audit fix` to apply non-breaking fixes.
Run `npm test` to confirm nothing broke.
Report which CVEs were patched and which remain.
```

Restart, then in any session:

```
/audit
```

The whole prompt fires. You get a structured audit report instead of having to remember the three commands and their order.

Two things to notice:

1. The command is just a prompt. There's no DSL, no special syntax. That keeps the surface area tiny.
2. You don't have to repeat yourself. The next time anyone on the team needs an audit, they type `/audit`.

## `$ARGUMENTS` — parameterization

![Claude Code Hands-On (3): Custom Slash Commands and Conversation Control — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/illustration_2.png)

Slash commands get a magic `$ARGUMENTS` token that's replaced with whatever you typed after the command name. Example — `.claude/commands/explain.md`:

```markdown
Explain $ARGUMENTS at three levels:

1. One sentence — what is it, in plain language.
2. One paragraph — how it works, key components, why it exists.
3. Code-level — point to where this is implemented in our repo, with line numbers.

If the term is ambiguous, list the meanings and ask which one I want.
```

Then:

```
/explain rate limiter
```

`$ARGUMENTS` becomes `rate limiter`, the prompt fires, you get a three-level explanation grounded in the actual repo.

## The commands I have on every project

After two years of using Claude Code I have settled on a small set:

**`/audit`** — security audit, as above.

**`/test`** — runs the test suite, summarizes failures, suggests fixes.

```markdown
Run the project's test suite (see CLAUDE.md for the test command).
For each failure:
  - quote the failing test name and the assertion that failed
  - propose the smallest plausible fix
  - mark whether you'd patch the test or the code
Do not make changes — this is a report.
```

**`/review`** — code review of a diff.

```markdown
Review the staged diff (`git diff --staged`).
Focus on: correctness, edge cases, naming, and adherence to CLAUDE.md.
Output as numbered findings, each with severity (must-fix / nit / praise).
End with one paragraph summarizing whether you'd approve.
```

**`/explain $ARGUMENTS`** — see above.

**`/onboard`** — produces a one-page brief for someone new to the codebase.

```markdown
Write a one-page onboarding doc for a new engineer joining this repo.
Use CLAUDE.md as the source of truth.
Include: what it does, how to set up, how to run tests, the three things
they're most likely to break, and where to look first.
```

Five commands. Together they cover most of what I'd otherwise type by hand every day.

## Conversation control — the three you should know

Built-in commands worth muscle-memorizing:

**`/compact`** — summarizes the conversation so far. Use when the model starts to feel slow. Keeps the gist, drops the verbose bits.

**`/clear`** — wipes conversation. Keeps memory and settings. Use when switching tasks.

**`/init`** — covered last piece. Run once per repo to bootstrap `CLAUDE.md`.

There are more — `/help` lists everything — but those three are the daily set.

## A note on team adoption

Slash commands are the easiest way to spread a convention across a team. Push three useful ones into `.claude/commands/` on `main`. The next time anyone runs `claude` in that repo, they have them. There's no configuration for the user. That's the point.

I have seen this pattern do more for team-wide AI usage than any internal training. People copy what's already there, and `.claude/commands/` is a directory full of "things that have been useful enough to commit."

## What it stops being good for

Slash commands are bad at:

- Anything that needs runtime arguments more complex than a single string
- Anything that needs to maintain state between invocations
- Anything you'd want to test

For those cases you want the SDK (piece 6). Slash commands are for the workflow shortcuts that don't justify the complexity of code.

Next piece: MCP — the protocol that lets Claude Code talk to anything.
