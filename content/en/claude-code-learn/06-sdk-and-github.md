---
title: "Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI"
date: 2026-04-23 09:00:00
tags:
  - claude-code
  - sdk
  - github-actions
  - automation
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 6
series_total: 10
description: "The SDK turns Claude Code from a CLI into a library. GitHub Action makes it answer @claude on PRs. Together they let you put Claude inside the same CI pipeline that already runs your tests — without giving up control."
disableNunjucks: true
translationKey: "claude-code-learn-6"
---

The CLI is the obvious surface. The SDK is the interesting one. The GitHub integration is where it pays off.

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_1.png)

---

## The SDK in one paragraph

`@anthropic-ai/claude-code` is the npm package. It exposes the same Claude Code engine the CLI uses, with the same tools and permissions, as a programmatic interface. You give it a prompt; you get an async iterable of conversation events. Plug it into anything — a script, a service, a CI step.

![SDK and CLI share the same agent engine, tools, and permissions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig3_sdk_arch.png)
*Figure: SDK and CLI share the same agent engine, tools, and permissions*

## SDK installation and setup

### Prerequisites

You need Node.js 18+ and an Anthropic API key. The SDK runs the same agent loop as the CLI, so it needs the same credentials.

```bash
# Check Node version
node --version  # needs v18+

# Install the SDK
npm install @anthropic-ai/claude-code

# Set your API key (required)
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
```

### Project setup for TypeScript

```bash
mkdir claude-automation && cd claude-automation
npm init -y
npm install @anthropic-ai/claude-code
npm install -D typescript @types/node tsx

# Create tsconfig
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "node16",
    "esModuleInterop": true,
    "strict": true,
    "outDir": "dist",
    "declaration": true
  },
  "include": ["src/**/*"]
}
EOF

mkdir src
```

### The hello-world

```typescript
// src/hello.ts
import { query } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: 'List the three largest source files in this repo and explain each in one sentence.',
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

Run it:

```bash
npx tsx src/hello.ts
```

You'll see the same agent loop play out in your terminal — tool calls and all. It's the CLI minus the chat UI.

## Understanding conversation events

The async iterable from `query()` yields typed events. Understanding these is critical for building real automations:

```typescript
import { query, type ConversationEvent } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: 'What files are in this directory?',
  options: { cwd: process.cwd(), permissionMode: 'acceptEdits' }
});

for await (const event of conversation) {
  switch (event.type) {
    case 'text':
      // The model's text output, streamed token-by-token
      process.stdout.write(event.text);
      break;

    case 'tool_use':
      // The model is calling a tool
      console.log(`\n[Tool] ${event.name}(${JSON.stringify(event.input)})`);
      break;

    case 'tool_result':
      // A tool returned its result
      console.log(`[Result] ${event.content?.substring(0, 100)}...`);
      break;

    case 'error':
      // Something went wrong
      console.error(`[Error] ${event.error}`);
      break;

    case 'done':
      // Conversation complete
      console.log('\n[Done]');
      break;
  }
}
```

### Collecting the full response

Often you want the complete text response, not the stream:

```typescript
import { query } from '@anthropic-ai/claude-code';

async function getResponse(prompt: string): Promise<string> {
  const conversation = query({
    prompt,
    options: { cwd: process.cwd(), permissionMode: 'acceptEdits' }
  });

  const parts: string[] = [];
  for await (const event of conversation) {
    if (event.type === 'text') {
      parts.push(event.text);
    }
  }
  return parts.join('');
}

const analysis = await getResponse('Analyze the error handling in src/api.ts');
console.log(analysis);
```

## Permissions, programmatically

This is the bit you have to get right. The CLI defaults to "ask the human." A script can't ask. So the SDK exposes permission modes:

| Mode | Behavior | Use when |
|------|----------|----------|
| `default` | Errors on any tool that would normally need confirmation | Testing, safe exploration |
| `acceptEdits` | Auto-accepts file edits, asks for shell commands | Automated refactoring |
| `bypassPermissions` | Auto-accepts everything | CI with full trust (dangerous) |
| Custom callback | You decide per call | Production scripts |

![SDK permission modes form a spectrum from safest to most autonomous](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig6_permission_modes.png)
*Figure: SDK permission modes form a spectrum from safest to most autonomous*

### The custom callback in detail

For real work, use the callback:

```typescript
import { query } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: 'Fix the failing tests and update the changelog',
  options: {
    cwd: process.cwd(),
    permissionMode: 'custom',
    permissionCallback: async (toolName, toolInput) => {
      // Allow all read operations
      if (toolName === 'Read') return 'allow';

      // Allow file edits within src/ and tests/
      if (toolName === 'Write' || toolName === 'Edit') {
        const path = toolInput.file_path || toolInput.path || '';
        if (path.includes('/src/') || path.includes('/tests/') || path.includes('CHANGELOG')) {
          return 'allow';
        }
        console.warn(`Denied write to: ${path}`);
        return 'deny';
      }

      // Allow specific safe bash commands
      if (toolName === 'Bash') {
        const cmd = toolInput.command || '';
        const safePatterns = [
          /^npm test/,
          /^npm run/,
          /^git (status|diff|log|show)/,
          /^ls /,
          /^cat /,
          /^grep /,
          /^find /,
        ];
        if (safePatterns.some(p => p.test(cmd))) return 'allow';

        console.warn(`Denied bash: ${cmd}`);
        return 'deny';
      }

      // Deny everything else by default
      return 'deny';
    }
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

Now you have programmatic policy. The script can run unattended and you know it can't do the things you've explicitly forbidden.

### Permission callback patterns

Here are reusable permission patterns:

```typescript
// Read-only: no writes, no shell
const readOnly = async (tool: string) =>
  tool === 'Read' ? 'allow' : 'deny';

// Edit-only: reads + writes, no shell
const editOnly = async (tool: string) =>
  ['Read', 'Write', 'Edit'].includes(tool) ? 'allow' : 'deny';

// Project-scoped: anything within the project, nothing outside
const projectScoped = async (tool: string, input: any) => {
  if (tool === 'Read') return 'allow';
  if (tool === 'Write' || tool === 'Edit') {
    const path = input.file_path || '';
    return path.startsWith(process.cwd()) ? 'allow' : 'deny';
  }
  if (tool === 'Bash') {
    const cmd = input.command || '';
    // Block commands that could escape the project
    if (/\/(etc|usr|var|root)/.test(cmd)) return 'deny';
    return 'allow';
  }
  return 'deny';
};
```

## Real-world automation scripts

### Script 1: auto-update CHANGELOG

I have this in `scripts/update-changelog.ts` in a few projects:

```typescript
import { query } from '@anthropic-ai/claude-code';
import { execSync } from 'child_process';

const lastTag = execSync('git describe --tags --abbrev=0').toString().trim();
const commits = execSync(`git log ${lastTag}..HEAD --oneline`).toString();

if (!commits.trim()) {
  console.log('No new commits since', lastTag);
  process.exit(0);
}

console.log(`Updating CHANGELOG for commits since ${lastTag}...`);
console.log(`Found ${commits.trim().split('\n').length} commits\n`);

const conversation = query({
  prompt: `
    Update CHANGELOG.md with a new entry for an upcoming release.
    The commits since ${lastTag} are:

    ${commits}

    Group them into Added/Changed/Fixed/Removed following Keep a Changelog format.
    Use semantic versioning to suggest the next version.
    Edit CHANGELOG.md in place. Do not create a new file.
    Today's date is ${new Date().toISOString().split('T')[0]}.
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'acceptEdits'
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}

console.log('\n\nCHANGELOG updated. Review with: git diff CHANGELOG.md');
```

Run before every release. The CHANGELOG writes itself, the commit log gets curated into prose, and I review and commit. Five minutes saved per release.

### Script 2: code review bot

A script that reviews staged changes and writes findings to a file:

```typescript
// scripts/review-staged.ts
import { query } from '@anthropic-ai/claude-code';
import { execSync } from 'child_process';
import { writeFileSync } from 'fs';

const diff = execSync('git diff --cached').toString();

if (!diff.trim()) {
  console.log('No staged changes to review.');
  process.exit(0);
}

const filesChanged = execSync('git diff --cached --name-only').toString().trim();
console.log('Reviewing staged changes in:');
console.log(filesChanged);
console.log('---\n');

const parts: string[] = [];

const conversation = query({
  prompt: `
    Review the following staged git diff. Focus on:
    1. Bugs or logic errors
    2. Security issues (hardcoded secrets, SQL injection, XSS)
    3. Performance concerns
    4. Missing error handling
    5. Breaking API changes

    Be specific. Reference line numbers. Skip style nits.

    Files changed:
    ${filesChanged}

    Diff:
    ${diff}
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'  // read-only, no tool use needed
  }
});

for await (const event of conversation) {
  if (event.type === 'text') {
    process.stdout.write(event.text);
    parts.push(event.text);
  }
}

// Save review to file
writeFileSync('.claude/last-review.md', parts.join(''));
console.log('\n\nReview saved to .claude/last-review.md');
```

Add it to your workflow:

```bash
# Stage your changes
git add -p

# Get a review before committing
npx tsx scripts/review-staged.ts

# If the review looks good, commit
git commit
```

### Script 3: dependency audit

```typescript
// scripts/audit-deps.ts
import { query } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: `
    Audit this project's dependencies:
    1. Read package.json
    2. Run 'npm audit' and analyze the results
    3. Check for outdated packages with 'npm outdated'
    4. Identify any dependencies that are deprecated or unmaintained
    5. Give me a prioritized list of actions (critical security fixes first)
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'custom',
    permissionCallback: async (tool, input) => {
      if (tool === 'Read') return 'allow';
      if (tool === 'Bash') {
        const cmd = input.command || '';
        if (/^npm (audit|outdated|ls|list)/.test(cmd)) return 'allow';
        if (/^(cat|grep|jq)/.test(cmd)) return 'allow';
      }
      return 'deny';
    }
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

### Script 4: multi-file refactoring

For large refactoring tasks that need to touch many files:

```typescript
// scripts/refactor.ts
import { query } from '@anthropic-ai/claude-code';

const task = process.argv[2];
if (!task) {
  console.error('Usage: npx tsx scripts/refactor.ts "description of refactoring"');
  process.exit(1);
}

console.log(`Starting refactoring: ${task}\n`);

const conversation = query({
  prompt: `
    Perform the following refactoring across this codebase:

    ${task}

    Rules:
    - Make changes file by file
    - Run tests after each significant change
    - If tests break, fix them before moving on
    - Do not change public API signatures unless that's the explicit goal
    - Commit each logical change separately with a clear message
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'acceptEdits'
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}

console.log('\n\nRefactoring complete. Review with: git log --oneline -10');
```

Usage:

```bash
npx tsx scripts/refactor.ts "Rename UserService to AccountService everywhere"
npx tsx scripts/refactor.ts "Convert all callback-based functions in src/utils/ to async/await"
npx tsx scripts/refactor.ts "Add TypeScript strict null checks and fix all resulting errors"
```

## SDK vs CLI comparison

![CLI and SDK differ in interface, not capability — same engine underneath](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig5_cli_vs_sdk.png)
*Figure: CLI and SDK differ in interface, not capability — same engine underneath*

When should you use the SDK versus the CLI? Here's a detailed comparison:

| Aspect | CLI (`claude`) | SDK (`@anthropic-ai/claude-code`) |
|--------|---------------|-----------------------------------|
| **Interface** | Interactive chat in terminal | Programmatic async iterable |
| **Permissions** | Prompted interactively | Callback or mode-based |
| **Best for** | Exploratory work, one-off tasks | Automation, CI, scripts |
| **Session state** | Persistent until `/clear` | Fresh per `query()` call |
| **Startup time** | ~2s | ~1s (no TUI overhead) |
| **Output** | Formatted for humans | Typed events for code |
| **Error handling** | Shows errors in chat | Throws/yields error events |
| **Hooks** | From settings.json | From settings.json (same) |
| **MCP servers** | From settings.json | From settings.json (same) |
| **CLAUDE.md** | Auto-loaded | Auto-loaded (same) |
| **Multi-turn** | Natural via chat | Build conversation array manually |
| **Parallel runs** | One at a time | Multiple `query()` calls |

The key insight: the SDK and CLI share the same engine. The difference is the interface, not the capability. Hooks, MCP servers, CLAUDE.md, and settings all work identically.

## The GitHub Action

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_2.png)

Anthropic ships an official Action: `anthropic/claude-code-action@v1`. Add it to a workflow:

![GitHub Action lifecycle: from @claude mention to PR reply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig4_action_flow.png)
*Figure: GitHub Action lifecycle: from @claude mention to PR reply*

### Basic setup

```yaml
# .github/workflows/claude.yml
name: Claude on PR

on:
  pull_request_review_comment:
    types: [created]
  issue_comment:
    types: [created]

jobs:
  claude:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
          fetch-depth: 0

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

Now anyone in your repo can write `@claude please review the failing test` in a PR comment and Claude will:

1. Check out the branch
2. Read the comment thread for context
3. Run whatever it needs to (tests, linters, file reads)
4. Reply on the PR with its analysis
5. Optionally push commits if asked

### Detailed workflow configuration

Here's a more complete setup with multiple triggers:

```yaml
# .github/workflows/claude-full.yml
name: Claude Code Assistant

on:
  # Respond to @claude in PR comments
  pull_request_review_comment:
    types: [created]
  issue_comment:
    types: [created]

  # Auto-review new PRs
  pull_request:
    types: [opened, synchronize]

jobs:
  # Job 1: Respond to @claude mentions
  respond:
    if: >
      (github.event_name == 'issue_comment' || github.event_name == 'pull_request_review_comment')
      && contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
          fetch-depth: 0

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          max-turns: 20
          timeout-minutes: 15

  # Job 2: Auto-review new PRs
  auto-review:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Review this PR. Focus on:
            1. Correctness and logic errors
            2. Security issues
            3. Performance concerns
            4. Missing tests for new functionality
            Leave your review as a PR review with specific file/line comments.
          max-turns: 10
          timeout-minutes: 10
```

### Action parameters reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anthropic-api-key` | Required | Your Anthropic API key |
| `prompt` | From comment | Override the prompt (for automated triggers) |
| `max-turns` | `30` | Maximum agent turns before stopping |
| `timeout-minutes` | `30` | Maximum execution time |
| `model` | Default | Specific model to use |
| `allowed-tools` | All | Restrict which tools Claude can use |

### Secrets management

Store your API key as a GitHub repository secret:

1. Go to Settings > Secrets and variables > Actions
2. Click "New repository secret"
3. Name: `ANTHROPIC_API_KEY`, Value: your key
4. The workflow references it as `${{ secrets.ANTHROPIC_API_KEY }}`

Never hardcode API keys in workflow files. They're visible in the repository.

The Action respects `.claude/settings.json` from the repo. The hooks you wrote in piece 5 still apply. The slash commands you wrote in piece 3 still work — `@claude /review` does what you'd expect.

## PR review workflow — detailed walkthrough

Let me walk through a complete PR review workflow from start to finish.

![Sequence of a PR review loop: developer, GitHub, and Claude Action](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig7_pr_sequence.png)
*Figure: Sequence of a PR review loop: developer, GitHub, and Claude Action*

### Setup once

1. Add the workflow YAML file (above) to `.github/workflows/`
2. Add the `ANTHROPIC_API_KEY` secret
3. Commit and push

### The daily workflow

A developer opens a PR. The auto-review job triggers:

```markdown
Claude Code reviews PR #42: "Add user authentication middleware"

---

## CI integration patterns

Beyond the GitHub Action, there are patterns for integrating Claude into your existing CI.

### Pattern 1: Pre-merge checks

Run Claude as a check before merging:

```yaml
name: Claude Pre-merge Check

on:
  pull_request:
    types: [opened, synchronize]
    branches: [main]

jobs:
  security-check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Review the diff in this PR for security issues only.
            Check for:
            - Hardcoded secrets or credentials
            - SQL injection vulnerabilities
            - XSS vulnerabilities
            - Insecure deserialization
            - Path traversal
            - Command injection

            If you find any security issues, leave a review requesting changes.
            If the code is clean, approve the PR.
          max-turns: 10
```

### Pattern 2: Automated documentation updates

When API files change, update docs automatically:

```yaml
name: Auto-update Docs

on:
  push:
    branches: [main]
    paths:
      - 'src/api/**'
      - 'src/models/**'

jobs:
  update-docs:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            The API or model files have changed. Update the API documentation:
            1. Read the changed files in src/api/ and src/models/
            2. Update docs/api-reference.md to reflect the current state
            3. If any endpoints were added/removed/changed, update the table
            4. Create a PR with the documentation updates
          max-turns: 15
```

### Pattern 3: Release notes generation

```yaml
name: Generate Release Notes

on:
  release:
    types: [created]

jobs:
  notes:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Generate release notes for ${{ github.event.release.tag_name }}.
            Look at commits since the previous tag.
            Group changes by category (Features, Fixes, Breaking Changes).
            Format for GitHub release notes (markdown).
            Update the release body using gh CLI.
          max-turns: 10
```

### Pattern 4: SDK in a custom CI step

If the GitHub Action doesn't fit your workflow, use the SDK directly:

```yaml
name: Custom Claude Check

on: [push]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install SDK
        run: npm install @anthropic-ai/claude-code

      - name: Run analysis
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          node --experimental-vm-modules << 'EOF'
          import { query } from '@anthropic-ai/claude-code';

          const conversation = query({
            prompt: 'Run the test suite and report any failures with root cause analysis.',
            options: {
              cwd: process.cwd(),
              permissionMode: 'bypassPermissions'
            }
          });

          for await (const event of conversation) {
            if (event.type === 'text') process.stdout.write(event.text);
          }
          EOF
```

This gives you full control over the prompting, permissions, and output handling.

## Advanced SDK patterns

### Multi-turn conversations

The basic `query()` is single-turn. For multi-turn conversations, manage the message history yourself:

```typescript
import { query } from '@anthropic-ai/claude-code';

async function multiTurn(turns: string[]) {
  let context = '';

  for (const turn of turns) {
    const fullPrompt = context
      ? `Previous context:\n${context}\n\nNew instruction:\n${turn}`
      : turn;

    const parts: string[] = [];
    const conversation = query({
      prompt: fullPrompt,
      options: {
        cwd: process.cwd(),
        permissionMode: 'acceptEdits'
      }
    });

    for await (const event of conversation) {
      if (event.type === 'text') {
        process.stdout.write(event.text);
        parts.push(event.text);
      }
    }

    context += `\nTurn: ${turn}\nResponse: ${parts.join('')}\n`;
    console.log('\n---\n');
  }
}

await multiTurn([
  'Read src/api/users.ts and identify any potential issues',
  'Fix the issues you identified and add error handling',
  'Write tests for the changes you made'
]);
```

### Parallel execution

Run multiple Claude tasks in parallel for independent analyses:

```typescript
import { query } from '@anthropic-ai/claude-code';

async function runTask(name: string, prompt: string): Promise<string> {
  const parts: string[] = [];
  const conversation = query({
    prompt,
    options: {
      cwd: process.cwd(),
      permissionMode: 'default'
    }
  });

  for await (const event of conversation) {
    if (event.type === 'text') parts.push(event.text);
  }

  return `## ${name}\n${parts.join('')}`;
}

// Run analyses in parallel
const results = await Promise.all([
  runTask('Security', 'Audit this codebase for security vulnerabilities. Read key files.'),
  runTask('Performance', 'Identify performance bottlenecks in the hot paths. Check database queries.'),
  runTask('Debt', 'Find the top 5 areas of technical debt. Look for TODO comments and complex functions.'),
]);

console.log(results.join('\n\n---\n\n'));
```

### Streaming to a file

For long-running tasks, stream output to a file while also displaying it:

```typescript
import { query } from '@anthropic-ai/claude-code';
import { createWriteStream } from 'fs';

const outputFile = createWriteStream('analysis-output.md');

const conversation = query({
  prompt: 'Do a comprehensive architecture review of this project.',
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'
  }
});

for await (const event of conversation) {
  if (event.type === 'text') {
    process.stdout.write(event.text);
    outputFile.write(event.text);
  }
}

outputFile.end();
console.log('\n\nOutput saved to analysis-output.md');
```

## Troubleshooting

### Common SDK issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ANTHROPIC_API_KEY not set` | Missing env var | Export the key in your shell or CI |
| `Permission denied` on tool use | `default` mode blocks everything | Switch to `acceptEdits` or custom callback |
| Timeout on long tasks | Default timeout too short | Set a longer timeout in options |
| `Cannot find module` | Wrong import path | Use `@anthropic-ai/claude-code`, not `claude-code` |
| Empty response | Model didn't produce text | Check for error events in the stream |

### Common GitHub Action issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Action never triggers | `if` condition wrong | Check `contains()` syntax and event type |
| "Resource not accessible" | Missing permissions | Add required permissions block |
| Claude can't push commits | Checkout without ref | Add `ref: ${{ github.head_ref }}` to checkout |
| Timeout after 30 min | Complex task | Increase `timeout-minutes` or reduce scope |
| No response on PR | Secret not configured | Verify `ANTHROPIC_API_KEY` in repo secrets |

### Debugging workflow runs

```bash
gh run list --limit 5

gh run view RUN_ID --log

gh run rerun RUN_ID
```

## What I use the GitHub Action for

Three things have stuck:

**1. PR triage.** New PR opens, a workflow runs `@claude /review` automatically. By the time a human looks, there's already a first-pass review on the conversation. Saves the reviewer 10 minutes per PR and catches the obvious stuff.

**2. Issue summarization.** A workflow runs on new issues to label them, suggest a fix scope, and link related code. The reporter gets a faster ack and the maintainer gets a head start.

**3. Doc updates.** When the schema changes, an Action runs that asks Claude to update the docs to match. Not always perfect, but ~80% of the work is mechanical.

## The boundary between SDK and Action

I use the SDK when:

- I want Claude inside a script I run locally
- The trigger is a cron, a file change, or a manual command
- I need to inspect the events programmatically
- I want fine-grained permission control via callbacks

I use the Action when:

- The trigger is a GitHub event (PR, issue, comment)
- The output should land in a PR or issue comment
- I want a human in the loop and the loop is `@mention`
- I want the simplest possible setup

There's overlap, of course. The Action is built on the SDK underneath.

## Putting the series together

Six pieces, one progression:

1. Install + the three-layer config
2. Shortcuts and modes
3. Slash commands for personal workflows
4. MCP for external integrations
5. Hooks for safety rails
6. SDK + Action for programmatic and CI use

Each piece on its own buys you something concrete. Together they turn Claude Code from a chat client for code into a piece of programmable infrastructure that lives inside your repo.

The single trait that distinguishes power users I've watched: they treat `.claude/` as part of the codebase. Settings, commands, hooks all committed, all reviewed in PRs, all evolving with the project. That's the muscle memory worth building.

Happy shipping.

## Review Summary

### Critical Issues

**1. Missing rate limiting (src/middleware/auth.ts:45)**
The authentication middleware doesn't rate-limit failed attempts.
An attacker could brute-force tokens. Consider adding:

    import rateLimit from 'express-rate-limit';
    const authLimiter = rateLimit({ windowMs: 15*60*1000, max: 100 });

**2. Token stored in localStorage (src/utils/auth.ts:12)**
localStorage is vulnerable to XSS. Consider httpOnly cookies instead.

### Suggestions

- Add tests for the token refresh flow (currently untested)
- The `validateToken` function swallows errors silently — log them
- Consider extracting the JWT config to environment variables

### What looks good
- Clean separation of auth middleware from business logic
- Good use of TypeScript generics in the token types
- Test coverage for the happy path is solid
```

The reviewer reads Claude's analysis, addresses the critical issues, and replies:

```text
@claude good catches. Please fix the rate limiting issue — add express-rate-limit
to the auth middleware with a 100 requests per 15 minute window. Also add a test
for it.
```

Claude reads the comment, checks out the branch, makes the changes, runs tests, and pushes a commit:

```text
I've made the following changes:

1. Added `express-rate-limit` dependency
2. Applied rate limiting to the auth middleware (100 req / 15 min window)
3. Added test in `tests/middleware/auth.test.ts` for rate limiting behavior

Commit: abc1234 "Add rate limiting to auth middleware"

All tests pass (47 passed, 0 failed).
```

This loop — review, request changes, Claude implements — can repeat as many times as needed. The PR conversation becomes the interface.
