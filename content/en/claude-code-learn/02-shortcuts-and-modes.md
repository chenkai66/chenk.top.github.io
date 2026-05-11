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

| State | What changes | Status bar shows |
|-------|--------------|------------------|
| **Normal** | Default. Asks before file writes and shell commands. | No indicator |
| **Accept edits** | Auto-accepts file edits. Still asks for shell commands. | `Auto-accept edits` |
| **Plan mode** | Builds a plan but does not execute. Useful for reviewing the approach before any side effects. | `Plan mode` |
| **Bypass permissions** | Skips all confirmation prompts. Use when you trust the current task. | `Yolo mode` |

The status bar tells you which state you're in. `Shift+Tab` cycles. There is no state I use most — they correspond to different tasks.

### When to use each state — practical scenarios

**Normal mode** is where you start every session and where you should stay for:

- First-time interactions with unfamiliar code
- Anything involving production data or deployment
- Working with code you don't fully understand yet
- Teaching someone how Claude Code works (they need to see the permission prompts)

I stay in Normal mode about 40% of the time. The permission prompts aren't just safety — they're a feedback mechanism. When Claude asks "can I run `npm test`?", that's also telling me what it's about to do.

**Accept edits mode** is my default for active development. The typical session looks like:

```
[Shift+Tab → Accept edits]
> refactor the UserService class to separate validation from persistence

[Claude writes files without asking — I watch each one appear]
[Claude reaches a shell command — pauses to ask]
> yes, run the tests

[Tests pass, I keep going]
> now update the controller to use the new service methods
```

This mode hits the sweet spot: Claude can write code without interrupting me, but anything with side effects (shell commands, git operations) still needs my approval. I use this mode for 40% of my sessions — basically any focused coding task where I'm watching the output.

The trap: don't turn on Accept edits and then walk away from the terminal. The point is that you're watching. If you're not watching, stay in Normal.

**Plan mode** is underrated. It turns Claude into an architect instead of a builder:

```
[Shift+Tab twice → Plan mode]
> I need to add WebSocket support to this Express app for real-time
> notifications. The current notification system polls every 30 seconds.

[Claude produces a plan:]
1. Install ws and @types/ws
2. Create src/websocket/server.ts with connection management
3. Create src/websocket/handlers.ts for notification events
4. Modify src/server.ts to attach WebSocket to the HTTP server
5. Update src/services/notifications.ts to push instead of queue
6. Add reconnection logic to the client
7. Update tests
```

Now I can review the plan, ask questions ("why not Socket.io?"), adjust ("skip step 6, the client team handles that"), and then switch to Accept edits mode and say "go." Plan mode is what I use when:

- I'm not sure about the right approach
- The change touches multiple files and I want to see the scope first
- I'm going to hand off the plan to someone else to execute
- The change is risky enough that I want to review before anything happens

**Bypass permissions (Yolo mode)** is exactly what it sounds like. Claude doesn't ask before doing anything — file writes, shell commands, all of it. I use this in very specific situations:

- Running a well-understood script that I've run before
- Doing a bulk rename or format operation across many files
- Quick prototyping where I'll `git stash` or discard everything anyway
- Automated workflows where I'm piping output and don't want interactive prompts

```
[Shift+Tab three times → Bypass permissions]
> rename all .jsx files in src/components/ to .tsx and fix the imports

[Claude does everything without stopping — 30 files renamed, imports updated]
[Check the result, git diff, looks good]
```

The danger is obvious. I have two rules for Yolo mode:

1. Never use it on code I can't easily revert (`git stash` or `git checkout .` should be one command away)
2. Never use it when the task involves external services (APIs, databases, deployments)

If you're new to Claude Code, ignore Yolo mode for the first month. You need to build intuition for what Claude does before you stop watching it do things.

### The muscle memory pattern

After a few weeks, state switching becomes unconscious. My typical session:

```
[Start in Normal — orient myself]
> what changed since yesterday? check git log and the open PRs.

[Switch to Plan mode — think about the day's work]
> I need to implement the invoice PDF generation feature. plan it out.

[Review plan, switch to Accept edits — build]
> go ahead with steps 1-4

[When tests need to run, I approve the shell command]

[If I need to do a mechanical task — switch to Yolo]
> add JSDoc comments to every exported function in src/api/

[Back to Normal when done]
```

## Thinking modes — five levels

![Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/02-shortcuts-and-modes/illustration_2.png)

Type one of these phrases anywhere in your prompt and Claude shifts how much it reasons before responding:

| Phrase | Approximate effort | Token cost | Response time |
|--------|-------------------|------------|---------------|
| *(none)* | Default reasoning | Baseline | Fast |
| `think` | Light extra reasoning | ~1.2x | Slightly slower |
| `think more` | Moderate reasoning | ~1.5x | A few seconds more |
| `think a lot` | Deep reasoning | ~2x | Noticeably slower |
| `think longer` | Extended reasoning | ~2.5x | Several seconds |
| `ultrathink` | Maximum reasoning | ~3x+ | Can take 15-30 seconds |

### How thinking levels affect output quality

The difference isn't just "more time." Each level changes the structure of Claude's internal reasoning:

**No thinking phrase** — Claude responds quickly with its first-pass answer. Good for straightforward tasks: "rename this variable," "add a null check here," "what does this function return?" If the answer is obvious, extra thinking is wasted.

**`think`** — Claude pauses briefly to consider alternatives. I use this for small but non-obvious tasks: "is there a simpler way to write this?" "does this handle the edge case where the list is empty?"

**`think more`** — Claude starts considering trade-offs and second-order effects. Good for debugging: "find why this test is flaky," "trace this null pointer to its source."

**`think a lot`** — This is my default for anything I'd describe as "non-trivial." Architecture questions, refactoring decisions, security reviews. Claude at this level will often identify issues you didn't ask about:

```
> @src/auth/middleware.ts
> think a lot — review this auth middleware for security issues.

[Claude identifies:
 1. Token not validated for expiry
 2. Missing rate limiting on failed attempts
 3. CORS header too permissive
 4. No CSRF protection on state-changing routes
 5. Timing attack possible on password comparison]
```

Without `think a lot`, it might catch items 1 and 3 but miss the timing attack.

**`think longer`** — Reserved for architecture decisions with real consequences. "Should we split this monolith into services?" "What's the right caching strategy for this data access pattern?" Claude will lay out multiple approaches with pros and cons before making a recommendation.

**`ultrathink`** — The maximum. I use this maybe once a week. The right situations:

- "Design the data model for this new feature, considering the existing schema, the anticipated query patterns, and the migration path from the current structure"
- "Review this entire module for correctness, performance, and maintainability. I want a senior engineer-level review."
- "This system is hitting latency spikes under load. Analyze the architecture and propose fixes."

### Practical scenarios — which level for which task

| Task | Thinking level | Why |
|------|---------------|-----|
| Fix a typo | None | Obvious, no reasoning needed |
| Rename a variable across files | None | Mechanical, not analytical |
| Add error handling to a function | `think` | Needs to consider what errors can occur |
| Debug a failing test | `think more` | Needs to trace cause and effect |
| Refactor a module | `think a lot` | Needs to preserve behavior while restructuring |
| Code review for PR | `think a lot` | Needs to find subtle issues |
| Architecture decision | `think longer` | Needs to weigh multiple approaches |
| System design from scratch | `ultrathink` | Needs comprehensive analysis |
| Security audit | `ultrathink` | Needs to think like an attacker |

### The cost trap

The trap is using `ultrathink` on every prompt because it sounds rigorous. It's not — it's expensive. The model does best at the level that matches the actual task complexity. Using `ultrathink` to rename a variable is like hiring a consulting firm to change a lightbulb.

I've measured the practical difference: my daily token spend dropped 40% when I started matching thinking levels to tasks instead of defaulting to `think a lot` for everything.

A good heuristic: if you could explain the task to a junior developer in one sentence, no thinking phrase needed. If you'd need a whiteboard session, `think a lot` or above.

## Every keyboard shortcut, annotated

Here's the complete reference for Claude Code keyboard shortcuts:

### Input line shortcuts

| Shortcut | Action | When I use it |
|----------|--------|---------------|
| `Enter` | Send message | Obviously |
| `Shift+Enter` | New line in input | Multi-line prompts |
| `Shift+Tab` | Cycle permission state | Multiple times per session |
| `Up arrow` | Previous message | Re-edit a prompt I just sent |
| `Ctrl+C` | Cancel current input | When I'm re-typing from scratch |
| `Ctrl+L` | Clear screen (not context) | Visual cleanup, context stays |

### During generation

| Shortcut | Action | When I use it |
|----------|--------|---------------|
| `Escape` | Stop generation | Wrong instruction, stop immediately |
| `Escape Escape` | Conversation fork | Undo the last exchange(s) |

### File and context

| Shortcut | Action | When I use it |
|----------|--------|---------------|
| `@` | File/directory picker | Every session, multiple times |
| `#` | Write to CLAUDE.md | When I discover a new convention |
| `Ctrl+V` | Paste image | Screenshots, error dialogs, designs |

## `Escape` and double-`Escape` — interruption vs. time travel

These deserve their own section because they're the most important shortcuts after `Shift+Tab`.

### Single `Escape` — stop generation

A single `Escape` interrupts the current generation. Use it the moment you realize you gave the wrong instruction. The agent stops, you correct.

```
> refactor all the controllers to—
[Wait, I meant just the payment controller]
[Press Escape]
> refactor only the PaymentController to use the new service pattern
```

The partial output from the interrupted generation stays visible but isn't committed. Claude hasn't written any files yet (assuming you're in Normal or Accept edits mode and it hadn't reached the file-writing stage).

Timing matters. If Claude has already started writing files when you press Escape, the files it already wrote stay changed. You'd need to `git checkout .` or undo manually. Press Escape early.

### Double `Escape` — conversation fork

Double-`Escape` (press it twice quickly) opens a history view of the last few exchanges. You pick one to "rewind" to, and the conversation forks from there. Anything after the picked turn is dropped from context.

```
> add caching to the getUser function
[Claude adds Redis caching]
[The implementation is over-engineered — I wanted in-memory caching]

[Double-Escape]
[Select: rewind to before the caching change]

> add simple in-memory caching to getUser using a Map with TTL.
> no Redis, just a module-level Map.
```

This is the feature I miss most when using other coding agents. Forking conversations is the right primitive — it lets you try one path, decide it was wrong, and go back without polluting the agent's memory.

Important: double-Escape doesn't undo file changes. If Claude wrote files before you fork, those files are still changed on disk. The fork only affects the conversation context. You'll need to `git checkout .` if you also want to undo the file changes.

My common pattern: before asking Claude to do something risky, I make a checkpoint:

```bash
git stash
```

Then if the result is bad, I double-Escape to fork the conversation AND `git stash pop` to restore the files. Clean slate.

## `Control+V` — paste images

Paste an image directly into the prompt. UI screenshot, error dialog, design comp, whiteboard photo — Claude reads it like text. The most useful version of this is "the test failed and the output is in this terminal screenshot, what's wrong?". You don't need to transcribe.

### What works well with image paste

- **Error screenshots:** Stack traces, browser console errors, terminal output
- **UI bugs:** "The button is in the wrong position" with a screenshot
- **Design comps:** "Implement this design" with a Figma export
- **Whiteboard sketches:** Architecture diagrams drawn during meetings
- **Data visualizations:** "What's wrong with this chart?" with a screenshot

### What doesn't work well

- **Screenshots of tiny text:** If you can barely read it, Claude can barely read it
- **Complex UML diagrams:** Claude can read them but often misses details in dense diagrams
- **Video/GIF:** Not supported — take a screenshot of the relevant frame

## `/compact` and `/clear` — context management

Two ways to deal with a long conversation:

- `/compact` summarizes the conversation so far into a much shorter message and continues from there. Use when the agent is starting to feel slow because the context is large.
- `/clear` drops the conversation entirely and starts fresh. Use when you've moved on to an unrelated task in the same project.

Both keep `CLAUDE.md` and `.claude/settings.json` in scope. They only affect the per-session message history.

### `/compact` — when and how

Claude Code has a context window. As your conversation grows, you burn more of it on history and less is available for the current task. Symptoms of a full context:

- Responses get slower
- Claude starts forgetting things you said earlier
- Claude repeats suggestions it already made
- The quality of code generation drops

When you notice these, run `/compact`. Claude summarizes the entire conversation into a few paragraphs and continues from the summary. You lose the exact wording of earlier messages but keep the gist.

```
> /compact

[Claude produces a summary:]
Summary: We've been working on adding WebSocket support. Completed:
server setup, connection handler, authentication middleware. Remaining:
client reconnection logic and integration tests. Current files modified:
src/websocket/server.ts, src/websocket/auth.ts, src/middleware/ws.ts.

[Conversation continues with the summary as context]
```

**When to compact vs. clear:**

| Situation | Action | Why |
|-----------|--------|-----|
| Slow responses, same task | `/compact` | Keep context, reduce tokens |
| Switching to unrelated task | `/clear` | Fresh start, no stale context |
| Lost in a wrong approach | Double-Escape | Fork to before the mistake |
| Starting the work day | `/clear` | Yesterday's context is stale |
| Mid-refactor, many files | `/compact` | Keep file change context |

**The compaction quality trade-off.** Compaction loses detail. If you had a nuanced discussion about why you chose approach A over approach B, the compact summary might just say "chose approach A." If that discussion matters, consider writing the decision to CLAUDE.md with `#` before compacting.

A pattern I use: at the end of a complex decision, before it gets compacted away:

```
# Architecture decision: chose WebSocket over SSE because we need
# bidirectional communication for the collaborative editing feature.
```

Now the reasoning survives compaction because it's in the persistent memory.

### `/compact` with a custom prompt

You can guide what `/compact` preserves by adding a prompt after it:

```
> /compact focus on the database schema changes and migration plan
```

Claude will weight its summary toward those topics. Useful when you know which part of the conversation is most important to carry forward.

## Multi-monitor and multi-terminal workflows

Claude Code is a terminal application, which means it fits into terminal multiplexer workflows naturally.

### Tmux split panes

My typical layout on a wide monitor:

```
┌──────────────────────┬──────────────────────┐
│                      │                      │
│  Claude Code         │  Editor (vim/vscode) │
│  (interactive)       │                      │
│                      │                      │
├──────────────────────┤                      │
│                      │                      │
│  Terminal            │                      │
│  (git, tests, logs)  │                      │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

Claude writes files in the left pane. The editor auto-reloads on the right. I run tests in the bottom-left. This three-pane layout means I never need to switch contexts.

### Multiple Claude sessions

You can run multiple Claude Code instances in the same project. Each has its own conversation context. Both share CLAUDE.md and settings.

Useful when you're working on two unrelated tasks in the same repo:

```
# Terminal 1 (tmux pane or tab)
claude
> work on the authentication refactor

# Terminal 2 (separate pane or tab)
claude
> fix the flaky test in test/integration/orders.test.ts
```

They won't conflict as long as they're working on different files. If they touch the same file, the last one to write wins, which is messy. Use this for genuinely independent tasks.

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

### Another example: debugging with mode switches

```
[Normal mode]
> the /api/orders endpoint is returning 500 in staging.
> here's the error log:
[Ctrl+V — paste screenshot of the error]

[Claude identifies the issue: a null reference in order serialization]

> think more — is this a data issue or a code issue?
> check if there are orders in the database without a customer_id.

[Claude investigates, finds 3 orders with null customer_id]

[Shift+Tab → Accept edits]

> fix the serializer to handle null customer_id gracefully.
> also add a database constraint to prevent this in the future.

[Claude writes the fix and the migration]

[Shift+Tab → Normal]

> run the tests

[All pass]

> now write a one-liner to backfill the 3 broken orders.
> show me the SQL, don't execute it.

[Claude shows the SQL]

> that looks right. I'll run it manually in staging.
```

Notice how the mode switches match the risk level. Investigation in Normal. Code writing in Accept edits. Anything touching the database stays in Normal with explicit approval.

## Quick reference card

Print this, tape it to your monitor for the first week:

```
Shift+Tab          Cycle: Normal → Accept → Plan → Yolo → Normal
Escape             Stop generation
Escape Escape      Fork/rewind conversation
Ctrl+V             Paste image
@                  Reference file
#                  Write to CLAUDE.md
/compact           Summarize and shrink context
/clear             Wipe conversation, keep memory
think              Light reasoning
think more         Moderate reasoning
think a lot        Deep reasoning (my daily default)
think longer       Extended reasoning
ultrathink         Maximum reasoning
```

Next piece: custom slash commands and `$ARGUMENTS`.
