---
title: "Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env"
date: 2026-04-24 09:00:00
tags:
  - claude-code
  - settings-json
  - permissions
  - configuration
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 9
description: "settings.json is the file that decides what Claude can do, where, and with whose credentials. The three layers (user, project, local), the permission grammar, env vars that change behavior, and the precedence order that catches everyone the first time."
disableNunjucks: true
translationKey: "claude-code-learn-9"
---

If hooks are how you reach into Claude Code, settings.json is where you tell it what it's allowed to touch in the first place. It is also the file that catches everyone with its precedence rules.

This chapter is the missing reference.

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_1.jpg)

## The three layers

There are three settings.json files Claude Code reads, in order:

1. **User settings** — `~/.claude/settings.json`. Applies to every project on your machine.
2. **Project settings** — `<repo>/.claude/settings.json`. Committed to git. Applies to anyone working in this repo.
3. **Local settings** — `<repo>/.claude/settings.local.json`. Gitignored. Your private overrides for this repo.

The merge rule: later layers override earlier ones, key by key. **Permissions are additive** for `allow`, **subtractive** for `deny` — once any layer denies something, no other layer can re-allow it. This asymmetry is what makes the system safe.

Practical consequence: keep org policy in `~/.claude/settings.json`, keep project rules in `.claude/settings.json` (committed), keep your "I trust this exact thing on my machine" overrides in `.claude/settings.local.json`.

## The permissions block — the grammar

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "WebFetch(domain:docs.anthropic.com)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Read(.env)",
      "Read(**/credentials*)"
    ],
    "additionalDirectories": []
  }
}
```

What goes inside the parentheses is a glob-style matcher specific to the tool:

- `Read(path-glob)` — file pattern.
- `Edit(path-glob)` — same.
- `Bash(command-pattern)` — first token must match. Use `*` carefully: `Bash(git *)` allows `git push --force`. Be more specific.
- `WebFetch(domain:host)` — host-only matching, no path.

A bare `Read` or `Bash` allows everything in that tool. That is almost always too broad outside `~/.claude/settings.json` for trusted personal use.

## Why deny wins

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_2.jpg)

Once anything in the merged config denies an action, nothing else can re-allow it. This is the lever you want.

Example. A repo's `.claude/settings.json` says:

```json
{ "permissions": { "deny": ["Bash(git push *)"] } }
```

A teammate adds `.claude/settings.local.json` with `"allow": ["Bash(git push origin main)"]`. It will *not* allow the push. The deny from the project layer wins. This is correct and you should rely on it.

## env — the other half

The `env` block sets environment variables for *every* tool call:

```json
{
  "env": {
    "NODE_ENV": "development",
    "PYTHONPATH": "./src",
    "DEBUG": "false"
  }
}
```

Two things to know:

- These vars apply to Bash and to any hook script that inherits the environment. They do *not* leak into the model's prompt. Safe place to set `*_API_KEY`.
- Local layer overrides project layer overrides user layer. So `DEBUG=true` in `.claude/settings.local.json` will turn on logging just for you, without committing the change.

## hooks — referenced from the same file

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep",
        "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [{ "type": "command", "command": "node ./hooks/format.js" }]
      }
    ]
  }
}
```

Matchers are pipe-separated tool names. Hooks across all three layers run; there's no override for hooks. Adding a hook in a deeper layer adds; it never replaces.

## A real settings.json from a real repo

Here's the project-level config from one of mine, lightly redacted:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(docs/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "WebFetch(domain:docs.anthropic.com)",
      "WebFetch(domain:nodejs.org)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Edit(.github/workflows/**)",
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/credentials*)"
    ]
  },
  "env": {
    "NODE_ENV": "development",
    "CI": "false"
  },
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] },
      { "matcher": "Bash",      "hooks": [{ "type": "command", "command": "node ./hooks/bash-blacklist.js" }] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "node ./hooks/format.js" }] }
    ]
  }
}
```

Three things to notice:

1. The Bash allowlist includes the read-only and reversible Git commands but never `push`, `reset --hard`, or `rebase -i`. Push is a deliberate human action.
2. `Edit(.github/workflows/**)` is denied. CI config changes need review; I do not want them slipping into a normal commit.
3. The hooks belt-and-brace the deny list. If a deny rule has a typo, the hook still blocks the dangerous call.

## The precedence order, as a checklist

When something doesn't behave the way you expect:

1. Is it in any `deny`? → blocked, regardless of allows.
2. Is it in any `allow`? → permitted.
3. Otherwise → Claude will ask before doing it.

If you want to know which rule won, run with `--debug` and read the permission resolution log. It tells you exactly which file and which line provided the verdict.

## Closing

settings.json is the constitution for what Claude can do in a project. Keep deny short and merciless, keep allow specific, keep hooks as the second line of defense. Once you have the layers and precedence in your head, configuring a new repo takes ninety seconds. Until then it will feel like the rules are arbitrary; they aren't.
