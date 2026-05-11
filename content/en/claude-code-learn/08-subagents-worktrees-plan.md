---
title: "Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode"
date: 2026-04-25 09:00:00
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

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

---

## Plan mode — the airlock

Plan mode is the cheapest of the three. Press `Shift+Tab` until the indicator says **plan**. The model now plans without taking any actions. It will read, think, propose, and stop. You read the plan. You either approve, edit, or kill it. Only then does it execute.

### How to activate it

There are three ways:

1. **Keyboard shortcut**: `Shift+Tab` cycles through modes: `auto` → `plan` → `auto`. The current mode shows in the bottom bar.
2. **CLI flag**: `claude --mode plan` starts a session in plan mode.
3. **Mid-conversation**: Just press `Shift+Tab` at any point. The mode switch is immediate.

### A real plan mode walkthrough

Let me show exactly what happens. Say I am working on an Express API and I want to add rate limiting:

```
You: Add rate limiting to the API. 100 requests per minute per IP.

[Mode: plan]

Claude: Here's my plan for adding rate limiting:

1. Install `express-rate-limit` (it's already in package.json
   as a devDependency but not in dependencies — I'll move it).

2. Create `src/middleware/rate-limit.ts`:
   - Export a configurable rate limiter factory
   - Default: 100 req/min per IP using `req.ip`
   - Use in-memory store for dev, note Redis store for prod

3. Wire it in `src/app.ts`:
   - Import the rate limiter
   - Apply globally with `app.use()` before route handlers
   - Add after CORS middleware, before body parser

4. Add rate limit headers to responses:
   - X-RateLimit-Limit
   - X-RateLimit-Remaining
   - X-RateLimit-Reset

5. Create `tests/middleware/rate-limit.test.ts`:
   - Test: allows 100 requests
   - Test: blocks request 101
   - Test: resets after window expires
   - Test: different IPs have independent limits

6. Update `src/types/env.d.ts` to add RATE_LIMIT_MAX
   and RATE_LIMIT_WINDOW_MS env vars.

Shall I proceed with this plan?
```

Now I can see: step 1 caught that the package is in devDependencies. Step 3 tells me where exactly in the middleware chain it goes. Step 6 reminds me about env vars. Without plan mode, Claude would have just started writing code, and I would have caught these details in review instead of upfront.

### When I use plan mode

- **The first 30 seconds of any non-trivial task.** "Implement the X feature" gets a plan first. Almost always, the plan reveals the model misunderstood the codebase.
- **Anything touching auth, payments, schema migrations, or production config.** Two seconds of reading a plan saves hours of cleanup.
- **When working in a repo I do not know well.** The plan doubles as my own onboarding doc — it shows me the architecture through the model's eyes.
- **Before refactoring.** "Refactor module X" → plan reveals the dependency graph. Sometimes the plan shows me the refactoring is not worth it.

### The common mistake

Skipping plan mode "because the task is small." Small tasks have the highest density of "wait, that is not what I meant." A one-line fix that touches the wrong file is worse than a large feature with a clear plan.

### Plan mode modifiers

You can steer the plan with instructions:

```
You: [plan mode] Add authentication. Focus the plan on the database
     schema changes. I already know the API layer.

Claude: Focusing on schema changes only:

1. Add `sessions` table...
2. Add `users.last_login` column...
3. Migration file...
```

The model respects the scope constraint. This keeps plans from becoming 20-step novels.

### Approving, editing, and rejecting

After Claude shows the plan:

- **Approve**: Switch back to `auto` mode (`Shift+Tab`) and say "go." Claude executes the plan.
- **Edit**: "Looks good, but skip step 4 and do step 3 before step 2." Claude adjusts.
- **Reject**: "Actually, never mind. Let's do Y instead." The plan was cheap — you lost thirty seconds, not thirty minutes.

---

## Sub-agents — for things you can run in parallel

A sub-agent is a Claude Code instance the parent agent spawns to handle a scoped task. It gets its own context window, its own tool set, and its own instructions. The parent orchestrates; the sub-agents do the work.

### Defining a sub-agent

The classic form lives in `.claude/agents/<name>.md`:

```markdown
---
name: research
description: Reads a topic across the codebase and reports findings. No edits.
tools: Read, Grep, Glob, WebFetch
---

You are a research sub-agent. Your job:
1. Search the codebase for the requested topic.
2. Read enough files to understand it deeply.
3. Return a structured report with:
   - File paths and line numbers
   - Direct quotes from the code
   - Dependencies and relationships between modules
   - Any concerns or inconsistencies you notice

Constraints:
- Do not edit any files.
- Do not run shell commands.
- Do not speculate — if you cannot find evidence, say so.
- Stay focused on the assigned topic.
```

### More agent examples

**A test-writing agent:**

```markdown
---
name: test-writer
description: Writes tests for a given module. Reads the source, writes tests. No changes to source code.
tools: Read, Grep, Glob, Write, Bash
---

You are a test-writing sub-agent. Your job:
1. Read the module you are asked to test.
2. Identify all public functions and edge cases.
3. Write comprehensive tests in the project's test framework.
4. Run the tests to verify they pass.

Rules:
- Never modify source files. Only create/modify test files.
- Follow the existing test patterns in the project.
- Each test must have a descriptive name that explains the scenario.
- Aim for edge cases, not just happy paths.
```

**A documentation agent:**

```markdown
---
name: doc-writer
description: Generates documentation for code changes. Reads source and existing docs, writes Markdown.
tools: Read, Grep, Glob, Write
---

You are a documentation sub-agent. Your job:
1. Read the code changes or module you are pointed at.
2. Read existing documentation for context and voice.
3. Write or update documentation that matches the project style.

Rules:
- Match the existing documentation voice and format.
- Include code examples for every public API.
- Do not run any commands.
- Do not modify source code.
```

### How sub-agents are invoked

In conversation, you can direct Claude to use a sub-agent:

```
You: Use the research agent to understand how the payment
     module processes refunds.

Claude: I'll spawn the research agent to investigate the
        payment/refund flow.

[Sub-agent: research]
Searching for refund-related code...
Found 12 files referencing refunds:
  - src/payments/refund.ts (main handler)
  - src/payments/refund-validator.ts (validation)
  ...

[Sub-agent report returned to parent]

Claude: Based on the research agent's findings, here's how
        refunds work in this codebase: ...
```

### What sub-agents buy you

**Context isolation.** The sub-agent's context window is its own. The parent's stays clean. This is critical when you are working on a complex task and need research done without polluting your main context with 30 file reads.

**Tool restriction.** A research agent literally cannot edit files. That is safety as architecture, not as discipline. You do not have to trust the model to avoid editing — it does not have the capability.

**Parallel work.** You can fan out to multiple sub-agents when the work is independent:

```
You: I need three things done in parallel:
     1. Research agent: how does auth work in this project?
     2. Test-writer agent: write tests for src/utils/validator.ts
     3. Doc-writer agent: update the API docs for the /users endpoint

Claude: Spawning three sub-agents in parallel...

[Sub-agent: research] Investigating auth flow...
[Sub-agent: test-writer] Reading validator.ts, writing tests...
[Sub-agent: doc-writer] Reading API docs, updating /users...

[All three complete]

Claude: Here are the results from all three agents: ...
```

### What sub-agents cost

**Tokens.** Each sub-agent has its own system prompt, its own context, its own back-and-forth. Three sub-agents doing research cost three times the tokens of one.

**Coordination.** The parent has to merge the results. If sub-agent A finds something that sub-agent B needs to know, they cannot talk to each other. Plan that step explicitly.

**Startup time.** Spawning a sub-agent is not instant. For a task that takes 10 seconds, the overhead of spawning is not worth it.

### When sub-agents are the wrong answer

- **When the parent already has the context.** Spawning a sub-agent to "go read this one file and report back" is expensive recursion. Just read the file.
- **When tasks are sequential.** If B depends on A's output, you cannot parallelize. Run them in sequence in the parent.
- **When the task is small.** If the whole job is "fix this typo," sub-agents are overhead.

---

## Worktrees — for parallel branches without losing your mind

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_2.png)

A git worktree is a second working tree of the same repo, on a different branch, in a different directory. Claude Code knows about them: the `EnterWorktree` tool creates a new branch + worktree and switches the session into it.

### How worktrees work under the hood

When Claude calls `EnterWorktree`, here is what happens:

1. A new directory is created under `.claude/worktrees/<name>/`.
2. A new git branch is created (based on `origin/main` by default, or current HEAD if configured).
3. `git worktree add` links the new directory to the new branch.
4. Claude's working directory switches to the new worktree.
5. All subsequent file operations happen in the worktree, not the original repo.

The key insight: the worktree shares the same `.git` object store as the original repo. Commits, branches, and refs are shared. But the working tree — the files on disk — is independent.

### Creating and entering a worktree

```
You: Create a worktree called "refactor-auth" and switch to it.

Claude: [EnterWorktree: name="refactor-auth"]
Created worktree at .claude/worktrees/refactor-auth/
Switched to new branch: refactor-auth
Working directory is now: /path/to/repo/.claude/worktrees/refactor-auth/
```

Now any file edits happen in the worktree. The original working tree is untouched.

### Configuring the base ref

By default, worktrees branch from `origin/<default-branch>` (usually `origin/main`). If you want to branch from your current HEAD instead:

The `worktree.baseRef` setting controls this:
- `fresh` (default): branch from `origin/main`
- `head`: branch from current local HEAD

### When worktrees matter

**Mid-task context switch.** You are mid-task on `feat/x` and the user asks for an unrelated quick fix on `main`. Spawn a worktree, do the fix, commit, exit. Your `feat/x` work is untouched.

**Experimental branches.** You want to try two different approaches to the same problem without polluting your main branch with abandoned commits. Create two worktrees, try both, keep the winner, remove the loser.

**Sub-agent isolation.** You are delegating to a sub-agent and want it physically isolated from your working tree. Give it a worktree. If its work is bad, you remove the worktree and nothing in your main tree was touched.

### Exiting a worktree

Two options:

**Keep the worktree:**
```
You: Exit the worktree but keep it around.

Claude: [ExitWorktree: action="keep"]
Returned to original directory: /path/to/repo/
Worktree "refactor-auth" preserved at .claude/worktrees/refactor-auth/
Branch "refactor-auth" preserved with 3 commits.
```

Use `keep` when the work is partial, when you want to come back later, or when you want to review the changes before merging.

**Remove the worktree:**
```
You: Done with the worktree. Remove it.

Claude: [ExitWorktree: action="remove"]
Removed worktree "refactor-auth" and branch.
Returned to original directory: /path/to/repo/
```

Use `remove` for a clean exit when the work is done or abandoned. If the worktree has uncommitted changes, removal refuses unless you confirm with `discard_changes: true`. This is correct. Do not paper over it.

### Merging worktree work back

After exiting a worktree with `keep`, the branch still exists:

```bash
# List worktree branches
git branch | grep -v "^\*"

# Merge the worktree work
git merge refactor-auth

# Or cherry-pick specific commits
git cherry-pick abc1234

# Then clean up
git worktree remove .claude/worktrees/refactor-auth
git branch -d refactor-auth
```

### Worktree gotchas

**You cannot have two worktrees on the same branch.** Git enforces this. If you try to create a worktree on a branch that already has one, it fails. Solution: use different branch names.

**Worktrees share the git lock.** Some git operations (like `gc`) lock the entire repo. If one worktree is running `gc`, the other blocks. This is rarely a problem in practice, but it can cause confusing hangs.

**IDE integration varies.** Some editors detect worktrees and adjust automatically. Others get confused about which directory is the "real" repo. VS Code handles worktrees well. Other editors, check first.

---

## Composing the three

The real power is in composition. Here is a real-world scenario that uses all three features together.

### Scenario: refactoring a monolith's auth module

The task: extract the authentication logic from a monolith Express app into a standalone middleware package. This involves 15+ files, a new package structure, migration of tests, and backward-compatible imports.

#### Step 1: Plan mode

```
You: [plan mode] I want to extract the auth logic into
     its own package. Plan the approach.

Claude: Plan for auth extraction:

1. Research phase:
   - Map all files that import from src/auth/
   - Identify shared types and interfaces
   - Find test files that cover auth logic
   - Check for circular dependencies

2. Extraction phase:
   - Create packages/auth/ with its own package.json
   - Move src/auth/*.ts to packages/auth/src/
   - Create packages/auth/index.ts with public API
   - Update all imports in the monolith

3. Compatibility phase:
   - Add re-export shim at src/auth/index.ts
   - Verify no runtime behavior change
   - Run full test suite

4. Cleanup phase:
   - Remove re-export shim
   - Update documentation
   - Update CI config

Risk: circular dependency between auth and user modules.
Mitigation: extract shared types first.

Shall I proceed?
```

I review the plan. The circular dependency risk is real — I did not think of it. I approve with a modification: "Do the research phase first, then we'll revisit the plan."

#### Step 2: Worktree

```
You: Create a worktree called "extract-auth" for this work.

Claude: [EnterWorktree: name="extract-auth"]
Created worktree on branch extract-auth.
Now in: .claude/worktrees/extract-auth/
```

Now everything happens in isolation. If the extraction fails, I remove the worktree and nothing in my main branch was touched.

#### Step 3: Sub-agents for research

```
You: Use the research agent to map all auth dependencies.

Claude: Spawning research agent...

[Sub-agent: research]
Found 23 files importing from src/auth/:
  - src/routes/api.ts (imports: authenticate, requireRole)
  - src/routes/admin.ts (imports: requireAdmin)
  - src/middleware/session.ts (imports: SessionConfig type)
  ...

Circular dependency detected:
  src/auth/user-lookup.ts imports from src/users/types.ts
  src/users/service.ts imports from src/auth/types.ts

Shared types that need extraction:
  - User (used in 14 files)
  - Session (used in 8 files)
  - AuthConfig (used in 3 files)
```

The research confirms the circular dependency. Now I know to extract shared types first.

#### Step 4: Implementation in the worktree

Claude proceeds with the extraction, working entirely in the worktree. Each step gets its own commit:

```
You: Start the extraction. Shared types first, then the auth
     module, then update imports.

Claude: [Working in worktree extract-auth]

Commit 1: "extract: shared types to packages/shared-types/"
Commit 2: "extract: auth module to packages/auth/"
Commit 3: "update: imports in monolith to use packages/auth"
Commit 4: "add: re-export shim for backward compatibility"
Commit 5: "test: all 47 auth tests passing in new location"
```

#### Step 5: Exit and merge

```
You: Tests pass. Exit the worktree and merge to main.

Claude: [ExitWorktree: action="keep"]
Back in main working directory.

The extract-auth branch has 5 commits.
To merge: git merge extract-auth
```

The whole refactoring happened in isolation. If at any point it went sideways, I could have exited with `remove` and started fresh.

---

## The three features as trust gates

Think of them as escalating levels of trust:

| Feature | Trust level | What you give up | What you gain |
|---------|-------------|-------------------|---------------|
| Plan mode | Low — you review before anything happens | Time (reading the plan) | Confidence the approach is right |
| Worktrees | Medium — edits happen but in isolation | Disk space, branch management | Ability to discard failed experiments |
| Sub-agents | High — parallel work with limited oversight | Tokens, coordination overhead | Speed on independent tasks |

The mistake is jumping to sub-agents for everything. Start with plan mode. Move to a worktree when you are confident in the approach but want isolation. Use sub-agents when you have genuinely independent work.

---

## Limitations and when not to use each

### Plan mode limitations

- **Not persistent.** The plan lives in the conversation. If the session ends, the plan is gone. For critical plans, copy them to a file or CLAUDE.md.
- **Not binding.** Claude can deviate from the plan during execution. It usually follows the plan, but edge cases or errors can cause drift. Review the results against the plan.
- **Adds latency.** For simple tasks, plan mode is overhead. "Fix the typo on line 42" does not need a plan.

### Sub-agent limitations

- **No inter-agent communication.** Sub-agents cannot talk to each other. If agent A discovers something agent B needs, the parent must relay it.
- **No shared state.** Each sub-agent starts with a fresh context. It does not know what other agents have done.
- **Token cost.** Three sub-agents cost three times the tokens. For exploratory work, this is fine. For routine tasks, it is wasteful.
- **No guaranteed consistency.** Two sub-agents editing the same file will create conflicts. Use worktrees or sequential execution for overlapping work.

### Worktree limitations

- **One branch per worktree.** You cannot have two worktrees on the same branch. Plan your branch names.
- **Disk usage.** Each worktree is a full copy of the working tree (though the .git objects are shared). On large repos, this can use significant disk space.
- **Cleanup discipline.** Forgotten worktrees accumulate. Run `git worktree list` periodically to see what is outstanding.
- **Not a substitute for proper branching.** If you need three worktrees for three long-running features, you might want three terminals instead.

---

## When to use none of them

Most tasks. Genuinely. The 80% case is "edit this function, run the test, ship it" — plain mode, no sub-agents, no worktrees. The features above earn their keep on the 20% that are large, irreversible, or branching.

If you find yourself reaching for sub-agents and worktrees on every task, the more interesting question is whether you are making your tasks too big.

### Quick reference

| Situation | Use |
|-----------|-----|
| Simple bug fix | Plain mode |
| Multi-file feature | Plan mode, then auto |
| Risky refactoring | Plan mode + worktree |
| Independent research | Sub-agent |
| Trying two approaches | Two worktrees |
| Large task with independent parts | Worktree + sub-agents |
| "Fix the typo on line 42" | Just let it do it |

Three trust gates, three escalations. By the time the model is editing files in a worktree with sub-agents, you have burned exactly the amount of attention the task deserves.
