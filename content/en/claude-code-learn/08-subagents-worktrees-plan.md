---
title: "Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode"
date: 2026-04-23 09:00:00
tags:
  - claude-code
  - sub-agents
  - worktrees
  - plan-mode
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 8
description: "Three features that change what Claude Code can take on at once: sub-agents for parallel research, worktrees for isolation, plan mode for the moments before you let it touch anything. The boundaries between them, and when each is the wrong answer."
disableNunjucks: true
translationKey: "claude-code-learn-8"
---

After hooks, the next thing that changes how Claude Code feels is *concurrency control*. Not concurrency in the threading sense — in the "how many things is the model doing for me, in how much isolation, with how much oversight" sense.

Three features, in escalating order of trust required.

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_1.jpg)

## Plan mode — the airlock

Plan mode is the cheapest. Press `Shift+Tab` until the indicator says **plan**. The model now plans without taking any actions. It will read, think, propose, and stop. You read the plan. You either approve, edit, or kill it. Only then does it execute.

When I use it:

- The first 30 seconds of any non-trivial task. "Implement the X feature" → plan first. Almost always, the plan reveals the model misunderstood the codebase.
- Anything touching auth, payments, schema migrations, or production config. Two seconds of reading saves hours of un-fucking.
- When working in a repo I don't know well. The plan doubles as my own onboarding doc.

The mistake: skipping plan mode "because the task is small." Small tasks have the highest density of "wait, that's not what I meant."

## Sub-agents — for things you can run in parallel

A sub-agent is a Claude Code instance the parent agent spawns to handle a scoped task. The classic form lives in `.claude/agents/<name>.md`:

```markdown
---
name: research
description: Reads a topic across the codebase and reports findings. No edits.
tools: Read, Grep, Glob, WebFetch
---

You are a research sub-agent. Your job:
1. Search the codebase for the requested topic.
2. Read enough files to understand it deeply.
3. Return a structured report with file paths and quotes.

Do not edit. Do not run shell commands. Stay focused.
```

Then in conversation: "research how authentication works → use the research agent."

What this buys you:

- **Context isolation.** The sub-agent's context window is its own. The parent's stays clean.
- **Tool restriction.** A research agent literally cannot edit. That's safety as architecture, not as discipline.
- **Parallel work.** You can fan out to three sub-agents at once when the work is independent.

What this costs:

- Tokens. Each sub-agent has its own system prompt, its own context, its own back-and-forth.
- Coordination. The parent has to merge the results. Plan that step explicitly.

When sub-agents are the wrong answer: any task where the parent already has the context it needs. Spawning a sub-agent to "go read this one file and report back" is just expensive recursion.

## Worktrees — for parallel branches without losing your mind

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_2.jpg)

A git worktree is a second working tree of the same repo, on a different branch, in a different directory. Claude Code knows about them: the `EnterWorktree` tool creates a new branch + worktree and switches the session into it.

When this matters:

- You're mid-task on `feat/x` and the user asks for an unrelated quick fix on `main`. Spawn a worktree, do the fix, commit, exit.
- You want to try two different solutions to the same problem without polluting your main branch with abandoned commits.
- You're delegating to a sub-agent and want it physically isolated from your working tree.

The mental model: a worktree is a *physical* version of context isolation. Sub-agents isolate context; worktrees isolate the filesystem.

How to think about exits:

- `keep` — the worktree stays on disk. Use this when the work is partial or might come back.
- `remove` — gone, branch deleted. Use only when you're sure.

If the worktree has uncommitted changes, removal refuses unless you confirm `discard_changes: true`. This is correct. Do not paper over it.

## Composing the three

The pattern I use on hard tasks:

1. **Plan mode.** "Here's what I want; what would you do?" Read the plan. Adjust.
2. **Worktree.** Move into an isolated branch so the experiment can fail without staining the trunk.
3. **Sub-agents** for independent sub-tasks within the worktree. Research first, then implementation, then test-writing — each in its own context.
4. Back in the parent, merge the results, commit, exit the worktree (`keep` if the work is paused, `remove` if it's done).

Three trust gates, three escalations. By the time the model is editing files, you've burned exactly the amount of attention the task deserves.

## When to use none of them

Most tasks. Genuinely. The 80% case is "edit this function, run the test, ship it" — plain mode, no sub-agents, no worktrees. The features above earn their keep on the 20% that are large, irreversible, or branching.

If you find yourself reaching for sub-agents and worktrees on every task, the more interesting question is whether you're making your tasks too big.
