---
title: "Claude Code 实战（六）：SDK 与 GitHub CI"
date: 2026-04-23 09:00:00
tags:
  - claude-code
  - sdk
  - github-actions
  - automation
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 6
series_total: 10
description: "SDK 把 Claude Code 从 CLI 变成库。GitHub Action 让它在 PR 上响应 @claude。两者一起把 Claude 放进已经在跑你测试的同一条 CI 流水线——又不交出控制。"
disableNunjucks: true
translationKey: "claude-code-learn-6"
---
CLI 是最显而易见的入口，SDK 才是真正有趣的部分，而 GitHub 集成则是它真正体现价值的地方。

![Claude Code 实战（六）：SDK 与 GitHub CI — 章节概览图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_1.png)


---

## 用一段话介绍 SDK

`@anthropic-ai/claude-code` 是 npm 包。它将 CLI 使用的相同 Claude Code 引擎（包括相同的工具和权限）以编程接口的形式暴露出来。你传入一个 prompt，它会返回一个异步可迭代的 conversation events。你可以把它集成到任何地方——脚本、服务、CI 步骤。


![SDK and CLI share the same agent engine, tools, and permissions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig3_sdk_arch.png)
*Figure: SDK 和 CLI 共享相同的 agent 引擎、工具和权限*

## SDK 安装与设置

### 前提条件

你需要 Node.js 18+ 和一个 Anthropic API key。SDK 运行的 agent loop 与 CLI 相同，因此需要相同的凭证。

```bash
# Check Node version
node --version  # needs v18+

# Install the SDK
npm install @anthropic-ai/claude-code

# Set your API key (required)
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
```

### TypeScript 项目设置

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

### Hello World 示例

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

运行它：

```bash
npx tsx src/hello.ts
```

你会在终端中看到相同的 agent loop 执行过程——包括所有工具调用。这本质上就是没有聊天 UI 的 CLI。

## 理解 conversation events

`query()` 返回的异步可迭代对象会生成类型化的事件。理解这些事件对于构建真正的自动化至关重要：

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

### 收集完整响应

很多时候你想要的是完整的文本响应，而不是流式输出：

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

## 以编程方式处理权限

这是你必须正确处理的部分。CLI 默认行为是“询问人类”，但脚本无法询问。因此 SDK 提供了多种权限模式：

| Mode | Behavior | Use when |
|------|----------|----------|
| `default` | 对任何通常需要确认的工具调用抛出错误 | 测试、安全探索 |
| `acceptEdits` | 自动接受文件编辑，对 shell 命令仍需确认 | 自动化重构 |
| `bypassPermissions` | 自动接受所有操作 | 完全可信的 CI（危险） |
| Custom callback | 每次调用由你决定 | 生产环境脚本 |


![SDK permission modes form a spectrum from safest to most autonomous](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig6_permission_modes.png)
*Figure: SDK 权限模式构成一个从最安全到最自主的连续谱*

### 自定义回调详解

对于实际工作，请使用回调函数：

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

现在你拥有了可编程的策略。脚本可以无人值守运行，同时你清楚它无法执行你明确禁止的操作。

### 权限回调模式

以下是一些可复用的权限模式：

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

## 真实世界的自动化脚本

### 脚本 1：自动更新 CHANGELOG

我在几个项目的 `scripts/update-changelog.ts` 中使用了这个脚本：

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

每次发布前运行。CHANGELOG 自动生成，提交日志被整理成连贯的文字，我只需审核并提交。每次发布节省五分钟。

### 脚本 2：代码审查机器人

一个审查暂存区变更并将发现写入文件的脚本：

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

将其添加到你的工作流中：

```bash
# Stage your changes
git add -p

# Get a review before committing
npx tsx scripts/review-staged.ts

# If the review looks good, commit
git commit
```

### 脚本 3：依赖审计

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

### 脚本 4：多文件重构

用于需要修改多个文件的大规模重构任务：

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

用法：

```bash
npx tsx scripts/refactor.ts "Rename UserService to AccountService everywhere"
npx tsx scripts/refactor.ts "Convert all callback-based functions in src/utils/ to async/await"
npx tsx scripts/refactor.ts "Add TypeScript strict null checks and fix all resulting errors"
```

## SDK 与 CLI 对比

![CLI and SDK differ in interface, not capability — same engine underneath](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig5_cli_vs_sdk.png)
*Figure: CLI 和 SDK 在接口上不同，但底层能力相同——共享同一引擎*

何时应使用 SDK 而非 CLI？以下是详细对比：

| Aspect | CLI (`claude`) | SDK (`@anthropic-ai/claude-code`) |
|--------|---------------|-----------------------------------|
| **Interface** | 终端中的交互式聊天 | 编程式的异步可迭代对象 |
| **Permissions** | 交互式提示确认 | 回调函数或模式控制 |
| **Best for** | 探索性工作、一次性任务 | 自动化、CI、脚本 |
| **Session state** | 持久化直到 `/clear` | 每次 `query()` 调用都是全新的 |
| **Startup time** | ~2s | ~1s（无 TUI 开销） |
| **Output** | 为人类格式化 | 为代码提供类型化事件 |
| **Error handling** | 在聊天中显示错误 | 抛出异常或生成错误事件 |
| **Hooks** | 来自 settings.json | 来自 settings.json（相同） |
| **MCP servers** | 来自 settings.json | 来自 settings.json（相同） |
| **CLAUDE.md** | 自动加载 | 自动加载（相同） |
| **Multi-turn** | 通过聊天自然支持 | 需手动构建对话数组 |
| **Parallel runs** | 一次只能运行一个 | 可发起多个 `query()` 调用 |

关键洞察：SDK 和 CLI 共享同一引擎。区别仅在于接口，而非能力。Hooks、MCP servers、CLAUDE.md 和设置都完全一致。

## GitHub Action

![Claude Code 实战（六）：SDK 与 GitHub CI — 章节小结图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_2.png)

Anthropic 提供了一个官方 Action：`anthropic/claude-code-action@v1`。将其添加到工作流中：


![GitHub Action lifecycle: from @claude mention to PR reply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig4_action_flow.png)
*Figure: GitHub Action 生命周期：从 @claude 提及到 PR 回复*

### 基础设置

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

现在，仓库中的任何人都可以在 PR 评论中写下 `@claude please review the failing test`，Claude 将会：

1. 检出该分支
2. 读取评论线程以获取上下文
3. 执行所需操作（运行测试、linter、读取文件等）
4. 在 PR 中回复其分析结果
5. 如果被要求，还可推送提交

### 详细的工作流配置

以下是包含多个触发器的更完整设置：

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

### Action 参数参考

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anthropic-api-key` | Required | 你的 Anthropic API key |
| `prompt` | From comment | 覆盖 prompt（用于自动化触发） |
| `max-turns` | `30` | agent 最大轮次后停止 |
| `timeout-minutes` | `30` | 最大执行时间（分钟） |
| `model` | Default | 指定使用的模型 |
| `allowed-tools` | All | 限制 Claude 可使用的工具 |

### Secrets 管理

将你的 API key 存储为 GitHub 仓库 secret：

1. 进入 Settings > Secrets and variables > Actions
2. 点击 "New repository secret"
3. 名称：`ANTHROPIC_API_KEY`，值：你的 key
4. 工作流中通过 `${{ secrets.ANTHROPIC_API_KEY }}` 引用

切勿在工作流文件中硬编码 API keys，因为它们对仓库可见。

该 Action 会尊重仓库中的 `.claude/settings.json`。你在第 5 篇中编写的 hooks 仍然有效。你在第 3 篇中编写的 slash commands 也依然可用——`@claude /review` 的行为符合预期。

## PR 审查工作流 —— 详细演练

让我从头到尾完整演示一个 PR 审查工作流。


![Sequence of a PR review loop: developer, GitHub, and Claude Action](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig7_pr_sequence.png)
*Figure: PR 审查循环的序列：开发者、GitHub 和 Claude Action*

### 一次性设置

1. 将上述工作流 YAML 文件添加到 `.github/workflows/`
2. 添加 `ANTHROPIC_API_KEY` secret
3. 提交并推送

### 日常工作流

开发者打开一个 PR。自动审查任务被触发：

```markdown
Claude Code reviews PR #42: "Add user authentication middleware"

---

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

审查者阅读 Claude 的分析，解决关键问题，并回复：

```text
@claude good catches. Please fix the rate limiting issue — add express-rate-limit
to the auth middleware with a 100 requests per 15 minute window. Also add a test
for it.
```

Claude 读取评论，检出分支，进行修改，运行测试，并推送一个提交：

```text
I've made the following changes:

1. Added `express-rate-limit` dependency
2. Applied rate limiting to the auth middleware (100 req / 15 min window)
3. Added test in `tests/middleware/auth.test.ts` for rate limiting behavior

Commit: abc1234 "Add rate limiting to auth middleware"

All tests pass (47 passed, 0 failed).
```

这个循环——审查、请求修改、Claude 实现——可以根据需要重复多次。PR 对话本身成为了交互界面。

## CI 集成模式

除了 GitHub Action，还有其他将 Claude 集成到现有 CI 的模式。

### 模式 1：合并前检查

在合并前将 Claude 作为检查步骤运行：

```yaml
# .github/workflows/claude-check.yml
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

### 模式 2：自动化文档更新

当 API 文件变更时，自动更新文档：

```yaml
# .github/workflows/claude-docs.yml
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

### 模式 3：发布说明生成

```yaml
# .github/workflows/release-notes.yml
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

### 模式 4：在自定义 CI 步骤中使用 SDK

如果 GitHub Action 不符合你的工作流，可以直接使用 SDK：

```yaml
# .github/workflows/custom-claude.yml
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

这让你能完全控制 prompt、权限和输出处理。

## 高级 SDK 模式

### 多轮对话

基础的 `query()` 是单轮的。对于多轮对话，你需要自行管理消息历史：

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

### 并行执行

为独立分析任务并行运行多个 Claude 任务：

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

### 流式输出到文件

对于长时间运行的任务，一边将输出流式写入文件，一边显示：

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

## 故障排查

### 常见 SDK 问题

| Problem | Cause | Fix |
|---------|-------|-----|
| `ANTHROPIC_API_KEY not set` | 缺少环境变量 | 在 shell 或 CI 中导出 key |
| `Permission denied` on tool use | `default` 模式阻止所有操作 | 切换到 `acceptEdits` 或自定义回调 |
| Timeout on long tasks | 默认超时太短 | 在选项中设置更长的超时 |
| `Cannot find module` | 导入路径错误 | 使用 `@anthropic-ai/claude-code`，而非 `claude-code` |
| Empty response | 模型未生成文本 | 检查流中的错误事件 |

### 常见 GitHub Action 问题

| Problem | Cause | Fix |
|---------|-------|-----|
| Action never triggers | `if` 条件错误 | 检查 `contains()` 语法和事件类型 |
| "Resource not accessible" | 缺少权限 | 添加所需的 permissions block |
| Claude can't push commits | checkout 时未指定 ref | 在 checkout 中添加 `ref: ${{ github.head_ref }}` \|
| Timeout after 30 min | 任务复杂 | 增加 `timeout-minutes` 或缩小范围 |
| No response on PR | secret 未配置 | 验证 repo secrets 中的 `ANTHROPIC_API_KEY` |

### 调试工作流运行

```bash
# Check recent workflow runs
gh run list --limit 5

# View a specific run's logs
gh run view RUN_ID --log

# Re-run a failed workflow
gh run rerun RUN_ID
```

## 我如何使用 GitHub Action

有三件事我一直坚持使用：

**1. PR 分类（triage）**。新 PR 打开时，工作流自动运行 `@claude /review`。当人类审查者查看时，对话中已经有一份初步审查。每次 PR 节省审查者约 10 分钟，并能捕捉明显问题。

**2. Issue 摘要**。工作流在新 issue 创建时运行，为其打标签、建议修复范围，并关联相关代码。报告者能更快得到响应，维护者也能提前准备。

**3. 文档更新**。当 schema 发生变化时，Action 会运行并让 Claude 更新文档以保持一致。虽然并非完美，但约 80% 的工作是机械性的。

## SDK 与 Action 的边界

我使用 SDK 的场景：

- 我想在本地运行的脚本中集成 Claude
- 触发条件是 cron、文件变更或手动命令
- 我需要以编程方式检查事件
- 我希望通过回调实现细粒度的权限控制

我使用 Action 的场景：

- 触发条件是 GitHub 事件（PR、issue、评论）
- 输出应出现在 PR 或 issue 评论中
- 我希望有人类参与，且交互方式是 `@mention`
- 我希望设置尽可能简单

当然，两者存在重叠。Action 底层正是基于 SDK 构建的。

## 整合整个系列

六篇文章，一条清晰的演进路径：

1. 安装 + 三层配置
2. 快捷键和模式
3. 用于个人工作流的 Slash Commands
4. 用于外部集成的 MCP
5. 用于安全防护的 Hooks
6. 用于编程和 CI 场景的 SDK + Action

每一篇单独看都能带来具体收益。合在一起，它们将 Claude Code 从一个代码聊天客户端转变为可编程的基础设施，深深嵌入你的代码仓库。

我观察到的高级用户有一个共同特征：他们将 `.claude/` 视为代码库的一部分。设置、命令、hooks 都被提交、都在 PR 中被审查、都随项目一起演进。这种肌肉记忆值得培养。

Happy shipping.