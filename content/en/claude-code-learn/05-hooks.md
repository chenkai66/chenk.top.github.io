---
title: "Claude Code Hands-On (5): Hooks, or How to Stop Worrying About Yolo Mode"
date: 2026-04-22 09:00:00
tags:
  - claude-code
  - hooks
  - safety
  - automation
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 5
description: "Hooks are the shell scripts that run before and after every tool call. PreToolUse can block. PostToolUse can format, lint, log. Five hooks I use on every repo, and the one anti-pattern that bites everyone."
disableNunjucks: true
translationKey: "claude-code-learn-5"
---

If MCP is how Claude reaches out, Hooks are how you reach in. They are the way to enforce — not just hope for — the rules you care about.

![Claude Code Hands-On (5): Hooks, or How to Stop Worrying About Yolo Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/05-hooks/illustration_1.png)

## The model

A Hook is a shell command. Claude Code runs it at one of a few well-defined moments. The two you'll use most:

- **`PreToolUse`** — runs before a tool is invoked. Exit code 0 lets the tool proceed; non-zero blocks it.
- **`PostToolUse`** — runs after a tool returns. Exit code is informational; you can use it to format files, run linters, log.

There are others (`UserPromptSubmit`, `Stop`, `Notification`, `SubagentStop`). For day-to-day work, those two are 90% of the value.

## Where hooks live

In `.claude/settings.json` (or its local variant). The minimal example:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "/path/to/check-bash.sh" }
        ]
      }
    ]
  }
}
```

The `matcher` decides which tools the hook applies to. `"Bash"` matches the built-in Bash tool. `"Write"` matches file writes. `"mcp__playwright__.*"` matches all Playwright MCP tools. Regex is supported — wildcards are common.

The hook command receives the tool's input as JSON on stdin and gets the conversation context as environment variables. The simplest possible hook is one that reads stdin, decides, and exits with the appropriate code.

## Hook 1: ban dangerous commands

`.claude/hooks/check-bash.sh`:

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')

# Refuse anything that destroys broadly
if echo "$cmd" | grep -E '(rm -rf /|sudo rm|chmod -R 777)' >/dev/null; then
  echo "Refusing — destructive pattern detected: $cmd" >&2
  exit 1
fi
exit 0
```

`PreToolUse` matcher `Bash`. Block exits with code 1; the message on stderr goes back to the model so it knows why and can adapt. This is the difference between "agent crashes silently" and "agent learns to ask permission."

## Hook 2: auto-format on write

`.claude/hooks/format-on-write.sh`:

```bash
#!/usr/bin/env bash
input=$(cat)
path=$(echo "$input" | jq -r '.tool_input.file_path')

case "$path" in
  *.py) ruff format "$path" 2>/dev/null ;;
  *.ts|*.tsx|*.js|*.jsx) prettier --write "$path" 2>/dev/null ;;
  *.go) gofmt -w "$path" 2>/dev/null ;;
esac
exit 0
```

`PostToolUse` matcher `Write|Edit`. Every time Claude touches a file, it gets formatted before the agent sees the next turn. The agent never has to be told about formatting again — your house style is enforced by code.

## Hook 3: log every tool call

```bash
#!/usr/bin/env bash
input=$(cat)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $(jq -c . <<< "$input")" \
  >> "$CLAUDE_PROJECT_DIR/.claude/tool-log.jsonl"
exit 0
```

`PostToolUse` matcher `.*`. Logs every tool call with timestamp to a JSONL file. One line per call. When something goes wrong — or right — you have a complete audit trail.

I have used this exactly three times. All three were post-mortems where I needed to know "what did Claude actually do during that 40-minute session." Worth its disk space.

## Hook 4: enforce test passing before commit

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')
if echo "$cmd" | grep -qE '^git commit'; then
  if ! npm test --silent >/dev/null 2>&1; then
    echo "Refusing commit: tests fail. Run 'npm test' to see details." >&2
    exit 1
  fi
fi
exit 0
```

`PreToolUse` matcher `Bash`. Intercepts `git commit` calls and runs the test suite first. If tests fail, the commit is blocked and the model is told why. A pre-commit hook for the agent itself.

This sounds aggressive. It is. The point is that you can't *accidentally* commit broken code anymore. The model has to actively work around the hook to do the wrong thing, and it doesn't.

## Hook 5: redact secrets in tool output

```bash
#!/usr/bin/env bash
sed -E '
  s/(Bearer|sk-)[A-Za-z0-9_-]{20,}/\1[REDACTED]/g
  s/(api[_-]?key["\s:=]+)["A-Za-z0-9_-]{16,}/\1[REDACTED]/gI
'
```

`PostToolUse` matcher `Read|Bash`. Filters tool output through a stream redactor before the agent sees it. If you accidentally `cat` a file with secrets in it, the model never reads them — they get replaced inline with `[REDACTED]`.

This is the single most important security hook for working with real codebases.

## The anti-pattern: relative paths

![Claude Code Hands-On (5): Hooks, or How to Stop Worrying About Yolo Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/05-hooks/illustration_2.png)

The most common hook bug is using a relative command path:

```json
{ "command": "./scripts/check.sh" }   // wrong
```

Claude Code runs the hook from a working directory you don't control. Always use absolute paths or use the `$CLAUDE_PROJECT_DIR` environment variable:

```json
{ "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check.sh" }  // right
```

Same for the scripts themselves. Inside a hook, don't `cd` and don't use relative paths.

## Where hooks fit in the team workflow

Hooks committed to `.claude/settings.json` apply to everyone. That's the point. A new teammate clones the repo, runs `claude`, and inherits your safety rails and your formatting policy automatically. No setup, no opt-in.

For personal-only hooks (e.g. ones that depend on your specific tools), put them in `.claude/settings.local.json` instead. They stay on your disk.

## What hooks are not

- **Not a substitute for permissions.** A hook can supplement permissions but should not replace them. Permissions are declarative and explicit; hooks are executable and easy to misconfigure.
- **Not a free check.** Every hook adds latency to every tool call. Five hooks at 100ms each is half a second per tool call. Watch the budget.
- **Not Turing-complete configuration.** When the hook starts looking like a small program, you're better off building an MCP server.

Next piece is the SDK and GitHub integration — programmatic Claude Code, in CI, against PRs. The end of the series and the most powerful piece.
