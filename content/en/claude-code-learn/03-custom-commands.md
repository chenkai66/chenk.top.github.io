---
title: "Claude Code Hands-On (3): Custom Slash Commands and Conversation Control"
date: 2026-04-20 09:00:00
tags:
  - claude-code
  - slash-commands
  - Workflow
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

### How the command system works internally

When you type `/audit`, Claude Code does the following:

1. Looks for `.claude/commands/audit.md` in the current project
2. Also checks `~/.claude/commands/audit.md` for personal global commands
3. Reads the file contents
4. Sends the contents as a prompt, as if you had typed them

Project commands take precedence over global commands with the same name. This means a team can override your personal `/audit` with a project-specific version.

The command content is treated as a regular prompt. That means:

- It can reference files with `@`
- It can include thinking level triggers ("think a lot")
- It can use `$ARGUMENTS` for parameterization
- It can contain multi-step instructions
- Markdown formatting is preserved

![Slash command lookup order: project commands win over user commands, which win over built-ins](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/fig3.png)
*Figure: how Claude Code resolves a slash command — project scope first, then user, then built-ins.*

## The directory structure

Here's the full layout for a mature project:

```
.claude/
├── commands/               # Project commands (committed to git)
│   ├── audit.md
│   ├── review.md
│   ├── test.md
│   ├── deploy.md
│   ├── explain.md
│   ├── document.md
│   ├── onboard.md
│   └── debug.md
├── settings.json           # Project settings (committed)
├── settings.local.json     # Personal settings (gitignored)
└── CLAUDE.md               # Project memory (committed, or at repo root)
```

And for personal global commands:

```
~/.claude/
├── commands/               # Personal global commands
│   ├── standup.md
│   ├── changelog.md
│   └── quicktest.md
├── settings.json           # Global settings
└── auth.json               # Auth token
```

The split matters. Team commands go in the project's `.claude/commands/`. Personal workflow commands go in `~/.claude/commands/`. Personal commands are available in every project. Team commands are specific to one repo.

### Naming conventions

Names become the slash command, so keep them short and obvious:

| Good | Bad | Why |
|------|-----|-----|
| `/review` | `/code-review-for-pr` | Too long to type |
| `/test` | `/run-all-tests-and-report` | Verbose |
| `/deploy` | `/deploy-to-staging-env` | Include the environment in $ARGUMENTS instead |
| `/explain` | `/e` | Too cryptic for teammates |
| `/debug` | `/dbg` | Abbreviation unclear to new members |

The sweet spot is one word, 4-8 characters, that any team member would guess on their first try. If someone can't guess that `/review` reviews code, the naming is wrong.

Avoid versioning in names (`/review-v2`, `/test-new`). If you need a better version, replace the old file. Git history preserves the old version if you ever need it back.

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

![$ARGUMENTS substitution pipeline: command file plus user input become the final prompt sent to Claude](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/fig4.png)
*Figure: $ARGUMENTS is pure string replacement — everything after the command name lands verbatim inside the prompt.*

### `$ARGUMENTS` patterns

The `$ARGUMENTS` token is just string substitution — everything after the command name becomes the value. This simplicity is a feature. Here are patterns that exploit it:

**Single argument — a name or identifier:**

```markdown
<!-- .claude/commands/find.md -->
Find all usages of $ARGUMENTS in the codebase.
Group by: direct usage, re-export, test usage.
Show the file path and line number for each.
```

```
/find UserService
```

**Multiple arguments as a natural sentence:**

```markdown
<!-- .claude/commands/compare.md -->
Compare $ARGUMENTS.
Show the key differences in a table format.
Include: API surface, performance characteristics, bundle size impact.
Recommend which one to use for our project and explain why.
```

```
/compare axios vs fetch vs ky for HTTP requests
```

Everything after `/compare` becomes `$ARGUMENTS`, so Claude gets "axios vs fetch vs ky for HTTP requests" as a natural language string.

**File path as argument:**

```markdown
<!-- .claude/commands/coverage.md -->
Analyze test coverage for $ARGUMENTS.

1. Find the test file(s) that test this module.
2. List every exported function/class.
3. For each, indicate whether it has test coverage.
4. For untested functions, write a test stub.

Do not modify existing tests. Only create new ones.
```

```
/coverage src/services/payment.ts
```

**No arguments — default behavior:**

If `$ARGUMENTS` is empty (the user just typed `/coverage` with nothing after), the substitution produces an empty string. Design your prompts to handle this gracefully:

```markdown
<!-- .claude/commands/status.md -->
Show the project status.

If a specific area is given: $ARGUMENTS — focus on that area.
Otherwise, give a general overview:
1. Git status (uncommitted changes, branch, ahead/behind)
2. Last 5 commits (one line each)
3. Any failing tests
4. Any lint errors
5. TODO comments added in the last week
```

```
/status                    # General overview
/status authentication     # Focused on auth
```

This pattern — "if arguments exist, scope to them; otherwise, do everything" — makes commands more flexible without needing separate commands for the general and specific cases.

## The commands I have on every project

After two years of using Claude Code I have settled on a core set. Here they are with the full prompt contents.

### `/audit` — security audit

```markdown
Run `npm audit` (or the equivalent for this project's package manager)
to find vulnerable installed packages.

Report:
1. Total vulnerabilities by severity (critical, high, moderate, low)
2. For each critical/high: the package name, CVE, and whether a fix exists
3. Run the fix command for non-breaking updates
4. Run tests to confirm nothing broke
5. List any remaining vulnerabilities that require manual intervention

If the project doesn't use npm, adapt the approach to the actual package
manager (pip, cargo, etc.) — check CLAUDE.md for guidance.
```

### `/test` — run and analyze tests

```markdown
Run the project's test suite (see CLAUDE.md for the test command).

For each failure:
  - Quote the failing test name and the assertion that failed
  - Show the relevant code (both the test and the code under test)
  - Propose the smallest plausible fix
  - Mark whether you'd patch the test or the code
  - Rate confidence: high/medium/low

At the end:
  - Total: X passed, Y failed, Z skipped
  - If all pass, just say "All tests pass" — no need for the detailed format

Do not make changes — this is a report.
```

### `/review` — code review

```markdown
Review the staged diff (`git diff --staged`).

If there's nothing staged, review unstaged changes (`git diff`).
If there are no changes at all, say so and stop.

Focus areas (in priority order):
1. Correctness — will this break anything?
2. Edge cases — what inputs/states aren't handled?
3. Naming — are variable/function names clear and consistent?
4. Adherence to CLAUDE.md conventions
5. Performance — any obvious N+1 queries, unnecessary loops, etc.
6. Security — any injection, auth bypass, data exposure risks?

Output format:
- Numbered findings
- Each has a severity: 🔴 must-fix / 🟡 should-fix / 🟢 nit / 👍 praise
- End with one paragraph: would you approve this PR as-is?

Think a lot before responding.
```

### `/explain` — three-level explanation

```markdown
Explain $ARGUMENTS at three levels:

1. **One sentence** — what is it, in plain language. No jargon.
2. **One paragraph** — how it works, key components, why it exists.
   Use analogies if helpful.
3. **Code-level** — point to where this is implemented in our repo,
   with file paths and line numbers. Walk through the key logic.

If the term is ambiguous in this codebase, list the possible meanings
and ask which one I want before explaining.

If it's not found in the codebase, explain it generally and note that
it doesn't appear to be implemented here.
```

### `/onboard` — new engineer orientation

```markdown
Write a one-page onboarding doc for a new engineer joining this repo.

Use CLAUDE.md as the source of truth. Supplement by reading key files.

Include:
1. **What it does** — one paragraph, plain language
2. **Setup** — step by step, from clone to running locally
3. **How to run tests** — the exact command and what to expect
4. **Architecture** — the 3-5 most important directories and what's in them
5. **Three things you're most likely to break** — common gotchas
6. **Where to look first** — when you need to debug, where do you start?
7. **Key contacts** — who owns what (if mentioned in CLAUDE.md)

Keep it to one page. No filler. A new engineer should be able to read
this in 10 minutes and be productive.
```

### `/deploy` — deployment checklist

```markdown
Prepare for deployment to $ARGUMENTS (default: staging).

Checklist:
1. Run tests — all must pass
2. Run lint — no errors (warnings OK)
3. Check for uncommitted changes — warn if any
4. Check the current branch — warn if not main/master
5. Show the commits that will be deployed (compare to last deploy tag)
6. Check for any TODO or FIXME comments in changed files
7. Verify environment variables are set (check .env.example vs .env)

Output a go/no-go summary at the end.

Do NOT actually deploy. This is a pre-deploy check only.
```

### `/debug` — structured debugging

```markdown
Debug the issue described below:

$ARGUMENTS

Follow this process:
1. **Reproduce** — identify the exact steps or conditions that trigger the issue
2. **Locate** — find the relevant code. Start with error messages, stack traces,
   or the module name mentioned in the description
3. **Hypothesize** — list 2-3 possible causes, ranked by likelihood
4. **Verify** — for the most likely cause, trace the code path and confirm
5. **Fix** — propose the minimal fix. Show the diff.
6. **Test** — suggest how to verify the fix works

Do not apply the fix. Present it for review.
Think a lot before responding.
```

### `/document` — generate documentation

```markdown
Document $ARGUMENTS.

If it's a file path: document that file's exports (functions, classes, types).
If it's a module name: document the module's public API.
If it's a concept: explain how it's implemented in this codebase.

For each function/method:
- Brief description (one line)
- Parameters with types and descriptions
- Return value
- Example usage
- Edge cases or gotchas

Output as JSDoc/docstring comments that can be pasted directly into the code.
Match the existing documentation style in the project.
```

## Building a team command library

Slash commands are the easiest way to spread a convention across a team. The process is dead simple:

1. Write a useful command
2. Put it in `.claude/commands/`
3. Commit it to main
4. Tell nobody

Step 4 is not a joke. The next time anyone runs `claude` in that repo and types `/`, they see the command in the autocomplete list. They try it. It works. They use it again. No training deck, no Confluence page, no onboarding session.

### How command libraries evolve

I've watched command libraries grow across three teams. The pattern is consistent:

**Week 1-2:** One person adds 2-3 commands. Usually `/test`, `/review`, and one project-specific one.

**Week 3-4:** Other team members discover the commands. They start using them. Somebody adds a `/deploy` command.

**Month 2:** The commands start getting refined. The `/review` prompt gets better criteria. Somebody adds `/explain` because they keep asking Claude to explain parts of the codebase.

**Month 3+:** The command library stabilizes at 5-10 commands. New ones get added rarely. The existing ones get tweaked for precision. At this point, the commands are effectively the team's shared vocabulary for interacting with AI.

### Commands as documentation

Here's a non-obvious benefit: your command library documents your team's workflows. A new engineer can read `.claude/commands/` and learn:

- What the deployment process looks like (`/deploy`)
- What the team values in code review (`/review`)
- What the testing strategy is (`/test`)
- What the common debugging approach is (`/debug`)

Each command file is a runnable specification of a workflow. That's more useful than a wiki page because it's always up to date (if it weren't, people would fix it because they use it every day).

### Personal vs. team commands

![Project scope vs user scope: location, git tracking, audience, lifetime](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/fig5.png)
*Figure: project commands ride with the repo; user commands follow you across machines. On a name clash, project wins.*

Keep the distinction clear:

| Type | Location | Git-tracked | Example |
|------|----------|-------------|---------|
| Team | `.claude/commands/` | Yes | `/review`, `/test`, `/deploy` |
| Personal | `~/.claude/commands/` | No | `/standup`, `/journal`, `/quickfix` |

Personal commands are for your individual workflow. Some of mine:

**`~/.claude/commands/standup.md`:**

```markdown
Help me prepare for standup.

1. What did I do yesterday? Check `git log --author="$(git config user.name)" --since="yesterday" --oneline`
2. What am I doing today? Check for any TODO comments I added recently and any open branches.
3. Any blockers? Check for failing tests or lint errors.

Format as three bullet points: yesterday, today, blockers.
```

**`~/.claude/commands/changelog.md`:**

```markdown
Generate a changelog entry for the changes since the last tag.

1. Find the last git tag
2. List all commits since that tag
3. Group by: Features, Fixes, Breaking Changes, Other
4. Write in keepachangelog.com format
5. Suggest the next version number based on the changes (semver)

Output the changelog entry — don't modify any files.
```

These are personal because they're about my workflow, not the team's. A teammate might have a completely different standup format.

## Advanced command patterns

### Commands that chain other commands

A command can reference other commands by name in its prompt:

```markdown
<!-- .claude/commands/morning.md -->
Run the following checks in order:

1. Check git status for any uncommitted changes from yesterday
2. Pull the latest changes from main
3. Run the test suite (same as /test)
4. Show any new TODO or FIXME comments added to main since yesterday
5. Summarize: what's changed, what's broken, what needs attention

This is my morning orientation — give me the full picture in one response.
```

This isn't literally invoking `/test` as a subcommand — it's asking Claude to do what `/test` does. Since Claude has read the `/test` command file (it's in the project), it knows the format. The effect is the same.

### Commands with structured output

```markdown
<!-- .claude/commands/deps.md -->
Analyze the dependency graph for $ARGUMENTS.

Output as a structured report:

## Direct Dependencies
| Package | Version | Used in | Purpose |
|---------|---------|---------|---------|

## Transitive Dependencies (notable)
List any transitive dependency that is:
- Known to have security issues
- Very large (>1MB)
- Duplicated at different versions

## Recommendations
- Any packages that could be removed?
- Any that should be updated?
- Any that have better alternatives?
```

### Commands for specific workflows

**Pre-PR checklist:**

```markdown
<!-- .claude/commands/pre-pr.md -->
I'm about to open a PR. Run through this checklist:

1. [ ] All tests pass
2. [ ] No lint errors
3. [ ] No TypeScript errors (`npx tsc --noEmit`)
4. [ ] No console.log statements in production code
5. [ ] All new functions have docstrings/JSDoc
6. [ ] No hardcoded secrets or API keys
7. [ ] CLAUDE.md is up to date if architecture changed
8. [ ] Changes are committed with clear messages

For each item, check and report pass/fail.
If everything passes, draft a PR title and description based on the commits.
```

**Database migration review:**

```markdown
<!-- .claude/commands/migration-check.md -->
Review the pending database migration(s).

Check for:
1. Reversibility — is there a down migration? Does it work?
2. Data loss — does any step drop columns or tables with data?
3. Locking — will any step lock a table for too long? (ALTER TABLE on large tables)
4. Index impact — are new indexes created concurrently?
5. Default values — do new NOT NULL columns have defaults for existing rows?
6. Foreign keys — are new FK constraints validated?

Think a lot. Database migrations are permanent.
```

## Debugging commands

When a command doesn't work as expected, here's how to diagnose:

**Command not showing up.** Check the file location. It must be exactly `.claude/commands/<name>.md` — not `.claude/command/` (singular), not `.claude/commands/<name>.txt`. Restart Claude Code after creating the file.

**Command runs but produces bad output.** The most common issue is an ambiguous prompt. Test your command prompt by pasting its contents directly into Claude Code as a regular message. If it doesn't work well as a direct prompt, it won't work well as a command.

**`$ARGUMENTS` not substituting.** Make sure you typed something after the command name. `/explain` with nothing after it sends an empty string for `$ARGUMENTS`. If your command requires arguments, add a note at the top:

```markdown
<!-- .claude/commands/explain.md -->
<!-- Usage: /explain <term or concept> -->

Explain $ARGUMENTS at three levels:
...
```

The HTML comment won't affect the prompt but serves as documentation when someone reads the file.

**Command too long.** There's no hard limit on command file length, but very long commands can push out context space for the actual work. Keep commands under 50 lines. If you need more, you're probably trying to do too much in one command — split it into two.

**Command works in one project but not another.** Check whether the command references project-specific details (specific file paths, specific test commands). If it does, either make it generic (use "see CLAUDE.md for the test command" instead of hardcoding `npm test`) or accept that it's a project-specific command.

## Conversation control — the three you should know

Built-in commands worth muscle-memorizing:

**`/compact`** — summarizes the conversation so far. Use when the model starts to feel slow. Keeps the gist, drops the verbose bits. Covered in depth in piece 2.

**`/clear`** — wipes conversation. Keeps memory and settings. Use when switching tasks.

**`/init`** — covered in piece 1. Run once per repo to bootstrap `CLAUDE.md`.

There are more — `/help` lists everything — but those three are the daily set.

### Other built-in commands worth knowing

| Command | What it does | When to use |
|---------|--------------|-------------|
| `/help` | Shows all available commands | When you forget a command name |
| `/compact` | Summarize and shrink context | Long sessions getting slow |
| `/clear` | Fresh conversation | Switching tasks |
| `/init` | Generate CLAUDE.md | New repo setup |
| `/config` | Open settings | Changing preferences |
| `/cost` | Show token usage | Monitoring spend |
| `/doctor` | Diagnose Claude Code issues | When something feels broken |
| `/login` | Re-authenticate | Token expired |
| `/logout` | Clear authentication | Switching accounts |
| `/permissions` | View active permissions | Debugging "why did it ask me?" |

## What stops being good for slash commands

Slash commands are bad at:

- **Anything that needs runtime arguments more complex than a single string.** `$ARGUMENTS` is just one token. You can't have named parameters like `/deploy --env staging --skip-tests`. If you need that, write a shell script and call it from Claude.

- **Anything that needs to maintain state between invocations.** Each `/command` runs in a fresh prompt context. There's no way to have `/step1` pass data to `/step2`. For multi-step stateful workflows, use a single long prompt or the SDK.

- **Anything you'd want to test.** You can't write a unit test for a slash command. If your workflow needs validation, it should be a script that Claude calls, not a command that Claude executes.

- **Anything that needs to be different per environment.** The command file is the same for everyone. If staging and production need different steps, use `$ARGUMENTS` to pass the environment name and handle the branching in the prompt.

For those cases you want the SDK (piece 6). Slash commands are for the workflow shortcuts that don't justify the complexity of code.

### The hierarchy of automation in Claude Code

```
Simple → Complex:

1. CLAUDE.md conventions    (passive — Claude follows them automatically)
2. Slash commands           (one-shot — type /name, get a result)
3. Shell scripts + Claude   (call scripts from Claude, or Claude from scripts)
4. Claude Code SDK          (programmatic — full control, state, testing)
```

![Automation hierarchy: CLAUDE.md, slash commands, shell scripts, SDK — climb only when you must](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/fig6.png)
*Figure: four tiers of automation. Most teams live happily at tiers 1 and 2 forever.*

Start at level 1. Move up only when the lower level can't handle your use case. Most teams never need level 4. Almost every team benefits from levels 1-2.

## A real command library walkthrough

Let me show what a mature `.claude/commands/` directory looks like for a production API project:

```
.claude/commands/
├── audit.md          # Security audit
├── debug.md          # Structured debugging
├── deploy.md         # Pre-deploy checklist
├── deps.md           # Dependency analysis
├── document.md       # Generate docstrings
├── explain.md        # Three-level explanation
├── migration-check.md # Database migration review
├── onboard.md        # New engineer orientation
├── pre-pr.md         # PR preparation checklist
├── review.md         # Code review
└── test.md           # Test run and analysis
```

![A mature .claude/commands directory grouped by purpose: quality gates, knowledge, operations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/03-custom-commands/fig7.png)
*Figure: 11 commands grouped by intent. Each file is one workflow — together they form the team’s shared vocabulary.*

11 commands. Each one is a workflow that used to be either (a) done manually with multiple steps, or (b) not done at all because it was too tedious.

The total investment to create these: maybe 2 hours spread over two months. The time saved per week: hard to measure, but conservatively 30 minutes per developer. With a team of 5, that's 2.5 hours per week. The commands paid for themselves in the first week.

The real value isn't the time savings — it's the consistency. Every code review covers the same criteria. Every debug session follows the same structure. Every deploy goes through the same checklist. The commands encode the team's best practices into repeatable workflows.

Next piece: MCP — the protocol that lets Claude Code talk to anything.
