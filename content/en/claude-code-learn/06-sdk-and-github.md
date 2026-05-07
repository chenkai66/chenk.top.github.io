---
title: "Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI"
date: 2026-04-21 09:00:00
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
description: "The SDK turns Claude Code from a CLI into a library. GitHub Action makes it answer @claude on PRs. Together they let you put Claude inside the same CI pipeline that already runs your tests — without giving up control."
disableNunjucks: true
translationKey: "claude-code-learn-6"
---

The CLI is the obvious surface. The SDK is the interesting one. The GitHub integration is where it pays off.

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_1.jpg)

## The SDK in one paragraph

`@anthropic-ai/claude-code` is the npm package. It exposes the same Claude Code engine the CLI uses, with the same tools and permissions, as a programmatic interface. You give it a prompt; you get an async iterable of conversation events. Plug it into anything — a script, a service, a CI step.

Install:

```bash
npm install @anthropic-ai/claude-code
```

The hello-world:

```typescript
import { query } from '@anthropic-ai/claude-code';

const result = query({
  prompt: 'List the three largest source files in this repo and explain each in one sentence.',
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'
  }
});

for await (const event of result) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

Run it. You'll see the same agent loop play out in your terminal — tool calls and all. It's the CLI minus the chat UI.

## Permissions, programmatically

This is the bit you have to get right. The CLI defaults to "ask the human." A script can't ask. So the SDK exposes permission modes:

| Mode | Meaning |
|------|---------|
| `default` | Errors on any tool that would normally need a confirmation |
| `acceptEdits` | Auto-accepts file edits, asks for shell |
| `bypassPermissions` | Auto-accepts everything (dangerous) |
| Custom | You provide a callback that decides per call |

For real work, use the callback:

```typescript
const result = query({
  prompt: '...',
  options: {
    permissionMode: 'custom',
    permissionCallback: async (tool, input) => {
      if (tool === 'Bash' && input.command.startsWith('rm ')) return 'deny';
      if (tool === 'Write' && input.path.startsWith('/etc')) return 'deny';
      return 'allow';
    }
  }
});
```

Now you have programmatic policy. The script can run unattended and you know it can't do the things you've explicitly forbidden.

## A real script: auto-update CHANGELOG

I have this in `scripts/update-changelog.ts` in a few projects:

```typescript
import { query } from '@anthropic-ai/claude-code';
import { execSync } from 'child_process';

const lastTag = execSync('git describe --tags --abbrev=0').toString().trim();
const commits = execSync(`git log ${lastTag}..HEAD --oneline`).toString();

const result = query({
  prompt: `
    Update CHANGELOG.md with a new entry for an upcoming release.
    The commits since ${lastTag} are:

    ${commits}

    Group them into Added/Changed/Fixed/Removed.
    Use semantic versioning to suggest the next version.
    Edit CHANGELOG.md in place.
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'acceptEdits'
  }
});

for await (const event of result) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

Run before every release. The CHANGELOG writes itself, the commit log gets curated into prose, and I review and commit. Five minutes saved per release × every release for the last six months.

## The GitHub Action

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_2.jpg)

Anthropic ships an official Action: `anthropic/claude-code-action@v1`. Add it to a workflow:

```yaml
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

The Action respects `.claude/settings.json` from the repo. The hooks you wrote in piece 5 still apply. The slash commands you wrote in piece 3 still work — `@claude /review` does what you'd expect.

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

I use the Action when:

- The trigger is a GitHub event
- The output should land in a PR or issue
- I want a human in the loop and the loop is `@mention`

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
