---
title: "Claude Code Hands-On (1): Install, the Three-Layer Config, and the # @ /init Trio"
date: 2026-04-16 09:00:00
tags:
  - claude-code
  - configuration
  - settings-json
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 1
description: "Install Claude Code, understand the three-layer settings.json system, and learn the three quietly powerful primitives: # to write into context, @ to reference files, /init to generate the project memory file."
disableNunjucks: true
translationKey: "claude-code-learn-1"
---

This is the first in a six-part field guide to Claude Code. The order is deliberate: each piece unlocks the next. By the end you will be using six features 90% of users never touch.

![Claude Code Hands-On (1): Install, the Three-Layer Config, and the # @ /init Trio — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/01-install-and-config/illustration_1.png)

## Install

There is one supported install path right now and it's the right one:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

That sets up the `claude` binary in `~/.local/bin/` (Linux/Mac). Add it to your PATH, then:

```bash
claude --version
# claude-code 0.2.x
claude login
```

The login flow opens a browser, you authorize the CLI, and you're done. Your auth token lives in `~/.claude/auth.json`. There is no API key in `~/.zshrc` to leak — that's worth noting because most "AI CLI" tools get this wrong.

## Run it

`cd` into a real project and run:

```bash
claude
```

You get an interactive prompt. Try one round-trip to confirm the install:

```
> what's in this directory?
```

Claude will use its own tooling to `ls`, summarize, and reply. If that worked, the install is done.

## The three-layer config

![Claude Code Hands-On (1): Install, the Three-Layer Config, and the # @ /init Trio — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/01-install-and-config/illustration_2.png)

This is the part most users never read. Claude Code merges configuration from three locations, in increasing precedence:

| Layer | Path | Tracked in git? | Used for |
|-------|------|-----------------|----------|
| Machine | `~/.claude/settings.json` | No | Your personal global preferences |
| Project | `<repo>/.claude/settings.json` | Yes | Team-shared project conventions |
| Local | `<repo>/.claude/settings.local.json` | No (gitignored) | Your private overrides for this repo |

Why three layers and not two? Because some settings are personal-and-global (your default model, your editor command), some are team-and-shared (the linter the project uses, the test command), and some are personal-but-project-scoped (your own API key for the staging server).

Concrete example. My machine-level `~/.claude/settings.json`:

```json
{
  "model": "claude-opus-4-7",
  "editor": "code"
}
```

A project's `.claude/settings.json` (committed):

```json
{
  "permissions": {
    "allow": ["Bash(npm test)", "Bash(npm run lint)"],
    "deny": ["Bash(rm -rf *)"]
  },
  "tools": {
    "test_command": "npm test"
  }
}
```

A teammate's `.claude/settings.local.json` (gitignored, on their disk only):

```json
{
  "model": "claude-sonnet-4-7",
  "env": {
    "STAGING_API_KEY": "sk-..."
  }
}
```

The merge order means the teammate gets project conventions for permissions and test command, but uses their personal model preference and their own staging key. This sounds obvious; I have seen entire team rollouts trip on it.

## `#` — write into context

Type `#` at the start of a line and the rest of the line is appended to the project's memory file (`CLAUDE.md`) instead of being sent as a prompt. Example:

```
# When working on this repo, never use yarn — npm only.
```

Claude doesn't reply. It opens an editor on `CLAUDE.md`, drops the line in, and saves. The next conversation starts with that line in context.

This is how I avoid re-explaining preferences. "When you write Python, use type hints." "Don't add emoji to commit messages." "The CI is GitHub Actions, not GitLab." Each one is a `#` line and stays for life.

## `@` — reference a file

Type `@` and you get a fuzzy file picker. Pick a file, it gets attached to the next message:

```
@src/router.ts
explain how the route matcher handles trailing slashes
```

The file is sent as a tool result, not pasted into your prompt. That means it counts against tool-budget, not against your typing. Practically: you can attach a 2000-line file without making your prompt unreadable.

## `/init` — bootstrap the project memory

Run `/init` once per repo. Claude reads the codebase, writes a `CLAUDE.md` summarizing:

- What the project does
- Languages and frameworks used
- Build, test, lint commands
- Major directories and their purposes
- Any conventions Claude can detect (commit message style, test naming, etc.)

You then edit it. The point isn't that Claude's draft is perfect — it isn't. The point is that you have a starting structure that takes 30 seconds instead of 30 minutes.

The generated `CLAUDE.md` is committed to the repo. Every teammate's Claude Code session starts with it. This is how a project shares mental model.

## What I do on every new repo

1. `claude` to open
2. `/init` to generate the memory file
3. Edit `CLAUDE.md` to add 3-5 specific conventions
4. Add `.claude/settings.json` with permissions for the test/lint commands
5. Add `.claude/settings.local.json` to `.gitignore`
6. Commit and move on

Five minutes. Pays back the first time anyone — me or a teammate — opens the repo with Claude Code.

Next piece: shortcuts and the four-state mode toggle that everyone misses.
