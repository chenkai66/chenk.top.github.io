---
title: "Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env"
date: 2026-04-26 09:00:00
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

Hooks let you interact with Claude Code, while settings.json specifies what it can access. This file also confuses many with its precedence rules.

This chapter is the missing reference.

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_1.png)

---

## The three layers

There are three settings.json files Claude Code reads, in order:

1. **User settings** — `~/.claude/settings.json`. Applies to every project on your machine.
2. **Project settings** — `<repo>/.claude/settings.json`. Committed to git. Applies to anyone working in this repo.
3. **Local settings** — `<repo>/.claude/settings.local.json`. Gitignored. Your private overrides for this repo.

The merge rule: later layers override earlier ones, key by key. **Permissions are additive** for `allow`, **subtractive** for `deny` — once any layer denies something, no other layer can re-allow it. This asymmetry is what makes the system safe.

### Where each file lives

```text
~/.claude/
  settings.json              # User-level (all projects)
  
my-project/
  .claude/
    settings.json             # Project-level (committed to git)
    settings.local.json       # Local overrides (gitignored)
```

Practical consequence: keep org policy in `~/.claude/settings.json`, keep project rules in `.claude/settings.json` (committed), keep your "I trust this exact thing on my machine" overrides in `.claude/settings.local.json`.
![Figure 3: The three settings.json layers and how each key merges across them.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig3.png)

*Figure 3: The three settings.json layers and how each key merges across them.*


---

## The complete settings.json reference

Here is every top-level key you can set in settings.json, with descriptions:

```json
{
  "permissions": {
    "allow": [],
    "deny": [],
    "additionalDirectories": []
  },
  "env": {},
  "hooks": {
    "PreToolUse": [],
    "PostToolUse": []
  },
  "worktree": {
    "baseRef": "fresh"
  }
}
```

### permissions

Controls what tools Claude can use and on what targets.

### env

Sets environment variables for all tool calls (Bash, hooks, etc.).

### hooks

Defines scripts that run before or after tool calls. See Chapter 7 for the full treatment.

### worktree

Controls worktree behavior. `baseRef` can be `"fresh"` (branch from origin/main) or `"head"` (branch from current HEAD).

---

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

### Tool permission syntax

Every permission entry follows the pattern: `ToolName` or `ToolName(pattern)`.

What goes inside the parentheses is a glob-style matcher specific to the tool:
![Figure 5: Permission rule grammar at a glance — every entry is ToolName(pattern) with tool-specific match semantics.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig5.png)

*Figure 5: Permission rule grammar at a glance — every entry is ToolName(pattern) with tool-specific match semantics.*


| Tool | Pattern type | Example | Matches |
|------|-------------|---------|---------|
| `Read` | File path glob | `Read(src/**)` | Any file under src/ |
| `Read` | File path glob | `Read(.env)` | Only .env in repo root |
| `Read` | File path glob | `Read(**/.env*)` | Any .env file, any depth |
| `Edit` | File path glob | `Edit(src/**)` | Edit files under src/ |
| `Edit` | File path glob | `Edit(*.ts)` | Edit TypeScript files in root |
| `Write` | File path glob | `Write(src/**)` | Write files under src/ |
| `MultiEdit` | File path glob | `MultiEdit(src/**)` | Multi-edit files under src/ |
| `Bash` | Command prefix | `Bash(npm run *)` | Any npm run command |
| `Bash` | Command prefix | `Bash(git status)` | Exactly `git status` |
| `Bash` | Command prefix | `Bash(git *)` | Any git command (careful!) |
| `WebFetch` | Domain | `WebFetch(domain:docs.anthropic.com)` | Only this domain |
| `Grep` | File path glob | `Grep(src/**)` | Grep in src/ only |

A bare tool name with no parentheses (e.g., just `Read`) allows everything for that tool. That is almost always too broad outside `~/.claude/settings.json` for trusted personal use.

### The additionalDirectories field

By default, Claude can only access files within the current project directory. To grant access to files outside the project:

```json
{
  "permissions": {
    "additionalDirectories": [
      "/path/to/shared-libs",
      "/path/to/design-system",
      "~/Documents/specs"
    ]
  }
}
```

Use cases:
- Monorepo where Claude needs to read sibling packages
- Shared design system in a separate directory
- Spec documents stored outside the repo

### Every permission type with examples

Here is the complete list of tool names you can use in permission rules:

| Tool name | What it does | Common allow pattern | Common deny pattern |
|-----------|-------------|---------------------|---------------------|
| `Read` | Read file contents | `Read(src/**)` | `Read(.env)`, `Read(**/credentials*)` |
| `Edit` | Modify existing files | `Edit(src/**)` | `Edit(.github/workflows/**)` |
| `Write` | Create new files | `Write(src/**)` | `Write(.env*)` |
| `MultiEdit` | Multiple edits in one call | `MultiEdit(src/**)` | `MultiEdit(.github/**)` |
| `Bash` | Run shell commands | `Bash(npm run *)` | `Bash(rm -rf *)`, `Bash(git push *)` |
| `Grep` | Search file contents | `Grep` (bare) | rarely denied |
| `Glob` | List files by pattern | `Glob` (bare) | rarely denied |
| `WebFetch` | Fetch web content | `WebFetch(domain:docs.*)` | `WebFetch(domain:internal.corp)` |
| `WebSearch` | Search the web | `WebSearch` (bare) | rarely denied |
| `NotebookEdit` | Edit Jupyter notebooks | `NotebookEdit(notebooks/**)` | project-specific |

---

## Why deny wins

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_2.png)

Once the merged config denies an action, nothing can re-allow it. This is the control you need.
![Figure 6: A deny list, organized by what category of damage it prevents.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig6.png)

*Figure 6: A deny list, organized by what category of damage it prevents.*


### Example: project deny overrides local allow

A repo's `.claude/settings.json` says:

```json
{ "permissions": { "deny": ["Bash(git push *)"] } }
```

A teammate adds `.claude/settings.local.json` with:

```json
{ "permissions": { "allow": ["Bash(git push origin main)"] } }
```

The push is **still blocked**. The deny from the project layer wins. This is correct and you should rely on it.

### Example: user deny overrides everything

Your `~/.claude/settings.json` says:

```json
{ "permissions": { "deny": ["Read(.env*)", "Read(**/secret*)"]} }
```

No project on your machine can read `.env` files or secrets, regardless of what their project settings say. This is your machine-wide policy.

### The deny-wins rule in practice

This asymmetry exists for security. Think of it this way:

- `allow` is a convenience — it removes the "do you want to allow this?" prompt.
- `deny` is a policy — it blocks the action regardless of who says otherwise.

An org can commit a `.claude/settings.json` with deny rules. Individual developers cannot override those denies. This is the mechanism for shared safety policy.

---

## env — the other half

The `env` block sets environment variables for *every* tool call:

```json
{
  "env": {
    "NODE_ENV": "development",
    "PYTHONPATH": "./src",
    "DEBUG": "false",
    "LOG_LEVEL": "info"
  }
}
```

### What env vars affect

- **Bash commands.** Every `Bash` tool call inherits these vars. `NODE_ENV=development` will be set when Claude runs `npm test`.
- **Hook scripts.** Hooks run as child processes and inherit the environment. A hook can read `process.env.LOG_LEVEL`.
- **They do NOT leak into the model's prompt.** The model cannot see env var values. Safe place for configuration.

### Layer precedence for env

Local layer overrides project layer overrides user layer. So `DEBUG=true` in `.claude/settings.local.json` will turn on logging just for you, without committing the change.

```json
// ~/.claude/settings.json (user)
{ "env": { "NODE_ENV": "development", "LOG_LEVEL": "warn" } }

// .claude/settings.json (project)
{ "env": { "NODE_ENV": "test", "CI": "false" } }

// .claude/settings.local.json (local)
{ "env": { "DEBUG": "true" } }

// Effective env:
// NODE_ENV=test        (project overrides user)
// LOG_LEVEL=warn       (only in user, so it persists)
// CI=false             (only in project)
// DEBUG=true           (only in local)
```

### Practical env patterns

**Setting API keys for tools:**

```json
{
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "DATABASE_URL": "postgresql://localhost:5432/dev"
  }
}
```

Put these in `settings.local.json` (gitignored) so they never get committed.

**Controlling test behavior:**

```json
{
  "env": {
    "NODE_ENV": "test",
    "TEST_TIMEOUT": "30000",
    "SKIP_SLOW_TESTS": "true"
  }
}
```

**Python path configuration:**

```json
{
  "env": {
    "PYTHONPATH": "./src:./lib",
    "VIRTUAL_ENV": "./.venv",
    "PATH": "./.venv/bin:${PATH}"
  }
}
```

---

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

### Hook configuration details

Matchers are pipe-separated tool names. The matcher `Read|Grep` fires on both Read and Grep tool calls.

**Special matcher `*`**: matches all tool calls. Use for logging or observability hooks.

**Multiple hooks per matcher**: hooks run in order. If any hook exits 2 (in PreToolUse), the call is blocked and remaining hooks do not run.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "node ./hooks/check-hours.js" },
          { "type": "command", "command": "node ./hooks/bash-blacklist.js" },
          { "type": "command", "command": "node ./hooks/bash-whitelist.js" }
        ]
      }
    ]
  }
}
```

### Hook layer behavior

Hooks across all three layers **accumulate** — they do not override. Adding a hook in a deeper layer adds to the hook chain; it never replaces hooks from higher layers.

```json
// ~/.claude/settings.json (user)
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ~/hooks/global-blacklist.js" }] }] } }

// .claude/settings.json (project)
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ./hooks/project-blacklist.js" }] }] } }

// Both hooks run. The user-level global-blacklist runs first,
// then the project-level project-blacklist.
```

This is different from permissions (where deny overrides allow) and env (where deeper layers override). Hooks always add.

---

## Real settings.json from different project types

### Node.js / TypeScript project

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(docs/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Bash(npm run *)",
      "Bash(npx *)",
      "Bash(node *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git branch *)",
      "Bash(git checkout *)",
      "WebFetch(domain:docs.anthropic.com)",
      "WebFetch(domain:nodejs.org)",
      "WebFetch(domain:typescriptlang.org)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Bash(git rebase *)",
      "Edit(.github/workflows/**)",
      "Edit(.claude/settings.json)",
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
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ./hooks/bash-blacklist.js" }] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "node ./hooks/format.js" }] }
    ]
  }
}
```

Three things to notice:

1. The Bash allowlist includes the read-only and reversible Git commands but never `push`, `reset --hard`, or `rebase`. Push is a deliberate human action.
2. `Edit(.github/workflows/**)` is denied. CI config changes need review; I do not want them slipping into a normal commit.
3. The hooks belt-and-brace the deny list. If a deny rule has a typo, the hook still blocks the dangerous call.

### Python / ML project

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(notebooks/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "NotebookEdit(notebooks/**)",
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(pip install *)",
      "Bash(pip3 install *)",
      "Bash(pytest *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(pip install --user *)",
      "Bash(sudo *)",
      "Read(.env)",
      "Read(**/credentials*)",
      "Read(**/weights/*)",
      "Write(data/**)"
    ]
  },
  "env": {
    "PYTHONPATH": "./src",
    "VIRTUAL_ENV": "./.venv",
    "CUDA_VISIBLE_DEVICES": "0",
    "TOKENIZERS_PARALLELISM": "false"
  }
}
```

Python-specific choices:

- `NotebookEdit` is allowed for the notebooks directory — Claude can modify Jupyter notebooks.
- `Read(**/weights/*)` is denied — model weight files are huge and reading them is pointless.
- `Write(data/**)` is denied — data files should not be modified by Claude.
- `CUDA_VISIBLE_DEVICES` is set to prevent accidental multi-GPU usage during development.

### Rust project

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(benches/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Bash(cargo *)",
      "Bash(rustup *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(cargo publish *)",
      "Edit(Cargo.lock)",
      "Edit(.cargo/**)"
    ]
  },
  "env": {
    "RUST_BACKTRACE": "1",
    "RUST_LOG": "debug"
  }
}
```

Rust-specific choices:

- `cargo publish` is denied — accidental crate publishing is irreversible.
- `Edit(Cargo.lock)` is denied — lockfile changes should come from `cargo update`, not direct edits.
- `RUST_BACKTRACE=1` is set so Claude sees full backtraces when tests fail.

### Monorepo / multi-language project

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(packages/**)",
      "Write(packages/**)",
      "Bash(npm run *)",
      "Bash(npx *)",
      "Bash(pnpm *)",
      "Bash(turbo *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Edit(packages/*/package.json)",
      "Edit(.github/**)",
      "Edit(turbo.json)",
      "Read(.env*)",
      "Read(**/credentials*)"
    ],
    "additionalDirectories": [
      "../shared-configs"
    ]
  },
  "env": {
    "TURBO_TELEMETRY_DISABLED": "1"
  }
}
```

Monorepo-specific choices:

- `Edit(packages/*/package.json)` is denied — dependency changes should be deliberate.
- `additionalDirectories` includes a sibling directory with shared configs.
- Individual packages can have their own `.claude/settings.json` with more permissive rules.

---

## Troubleshooting permission issues

### "Claude keeps asking for permission"

The most common complaint. Claude asks because the action is neither in `allow` nor `deny` — it falls through to the interactive prompt.

**Fix**: Add the action to `allow`:

```json
// Before: Claude asks every time it wants to run tests
// After:
{ "permissions": { "allow": ["Bash(npm test)", "Bash(npm run test:*)"] } }
```

### "Claude is blocked but I don't know why"

Run Claude with `--debug` to see the permission resolution log:

```bash
claude --debug
```

The debug output shows exactly which settings file provided each rule and which rule matched.

### "I allowed something but it's still blocked"

Check for deny rules. Remember: deny wins over allow, always. A common pattern:

```json
// This BLOCKS git push even though it appears in allow:
{
  "permissions": {
    "allow": ["Bash(git push origin main)"],
    "deny": ["Bash(git push *)"]
  }
}
```

The deny on `git push *` matches `git push origin main`, so it wins. To allow a specific push while blocking others, you need to restructure:

```json
// This does NOT work — deny always wins.
// Instead, use a hook that checks the specific target:
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{ "type": "command", "command": "node ./hooks/selective-push-gate.js" }]
    }]
  }
}
```

### "Local settings are not being picked up"

Verify the file name and location:
- Must be `.claude/settings.local.json` (not `settings-local.json` or `local.settings.json`)
- Must be in the repo root's `.claude/` directory
- Must be valid JSON

```bash
# Verify the file exists and is valid JSON
cat .claude/settings.local.json | jq .
```

### "Hooks from user settings don't run"

Check that the hook script path is absolute or relative to the right directory:

```json
// In ~/.claude/settings.json, use absolute paths:
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ~/hooks/global-blacklist.js" }] }] } }

// In .claude/settings.json (project), use relative paths:
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ./hooks/project-blacklist.js" }] }] } }
```

---

## The precedence order, as a checklist

When something does not behave the way you expect:

1. Is it in any `deny`? Blocked, regardless of allows.
2. Is it in any `allow`? Permitted without prompting.
3. Otherwise, Claude will ask before doing it (interactive prompt).
![Figure 4: How a tool call resolves through deny -> allow -> prompt; deny short-circuits everything else.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig4.png)

*Figure 4: How a tool call resolves through deny -> allow -> prompt; deny short-circuits everything else.*


### Precedence for each config type

| Config key | Precedence rule |
|-----------|----------------|
| `permissions.deny` | Union of all layers. **Any deny from any layer blocks.** |
| `permissions.allow` | Union of all layers. Any allow from any layer permits (unless denied). |
| `env` | Later layers override earlier. Local > Project > User. |
| `hooks` | Accumulate across all layers. All hooks run. |
| `worktree` | Later layers override earlier. |

### The merge visualized

```text
User settings        Project settings       Local settings
~/.claude/           .claude/               .claude/
settings.json        settings.json          settings.local.json
     |                    |                       |
     v                    v                       v
  ┌──────┐           ┌──────┐               ┌──────┐
  │allow │  ─────>   │allow │  ─────>       │allow │
  │deny  │  UNION    │deny  │  UNION        │deny  │
  │env   │  ─────>   │env   │  OVERRIDE >   │env   │
  │hooks │  ─────>   │hooks │  ACCUMULATE>   │hooks │
  └──────┘           └──────┘               └──────┘
                                                 |
                                                 v
                                          ┌────────────┐
                                          │  Effective  │
                                          │   Config    │
                                          └────────────┘
```

### Common patterns summarized

| What you want | Where to put it |
|---------------|----------------|
| "Never allow X on any project" | `~/.claude/settings.json` deny |
| "This project forbids X" | `.claude/settings.json` deny (committed) |
| "I personally want to skip the prompt for X" | `.claude/settings.local.json` allow |
| "Set DEBUG=true just for me" | `.claude/settings.local.json` env |
| "Everyone on this project should run Prettier on save" | `.claude/settings.json` hooks |
| "I want an extra logging hook" | `.claude/settings.local.json` hooks |

---

## Building a settings.json from scratch

When you start a new project, here is the process I follow:

### Step 1: Start with deny

What should never happen in this project?

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Read(.env*)",
      "Read(**/credentials*)",
      "Read(**/secret*)"
    ]
  }
}
```

### Step 2: Add allows for common operations

What does Claude do repeatedly that I am tired of approving?

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ]
  }
}
```

### Step 3: Add env vars

What does the development environment need?

```json
{
  "env": {
    "NODE_ENV": "development"
  }
}
```

### Step 4: Add hooks

What policies should be enforced automatically?

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] }
    ]
  }
}
```

### Step 5: Test by using Claude

Run a session and see what prompts you get. If Claude keeps asking for permission on something safe, add it to allow. If Claude does something you do not want, add it to deny. Iterate.

---

## Closing

settings.json is the constitution for what Claude can do in a project. Keep deny short and merciless, keep allow specific, keep hooks as the second line of defense. Once you have the layers and precedence in your head, configuring a new repo takes ninety seconds. Until then it will feel like the rules are arbitrary; they are not.
