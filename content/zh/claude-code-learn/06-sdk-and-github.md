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
description: "SDK 把 Claude Code 从 CLI 变成库。GitHub Action 让它在 PR 上响应 @claude。两者一起把 Claude 放进已经在跑你测试的同一条 CI 流水线——又不交出控制。"
disableNunjucks: true
translationKey: "claude-code-learn-6"
---
CLI 是门面，SDK 才是核心，而 GitHub 集成才是真正释放价值的地方。

![Claude Code 实战 (6)：SDK、GitHub 集成和 CI 中的 Claude — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_1.png)

## 一段话说清 SDK

`@anthropic-ai/claude-code` 就是那个 npm 包。它暴露了 CLI 所用的同一套 Claude Code 引擎——包括相同的工具和权限体系——只不过换成了程序化接口。你传入一个 prompt，它就返回一个异步可迭代对象（async iterable），逐个产出对话事件。你可以把它嵌入任何地方：脚本、服务，甚至 CI 步骤。

![SDK 和 CLI 共享同一个 agent 引擎、工具与权限](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig3_sdk_arch.png)  
*图：SDK 和 CLI 共享同一个 agent 引擎、工具与权限*

## SDK 安装与配置

### 前置条件

你需要 Node.js 18+ 和一个 Anthropic API 密钥。因为 SDK 运行的是和 CLI 完全相同的 agent 循环，所以凭据也必须一致。

```bash
# 检查 Node 版本
node --version  # 要求 v18+

# 安装 SDK
npm install @anthropic-ai/claude-code

# 设置 API key（必需）
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
```

### TypeScript 项目配置

```bash
mkdir claude-automation && cd claude-automation
npm init -y
npm install @anthropic-ai/claude-code
npm install -D typescript @types/node tsx

# 创建 tsconfig
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

### Hello-world 示例

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

运行：

```bash
npx tsx src/hello.ts
```

你会在终端里看到同样的 agent 循环执行过程——包括所有工具调用。本质上，这就是一个去掉了聊天界面的 CLI。

## 理解对话事件

`query()` 返回的异步可迭代对象会产出类型化的事件。要构建真正的自动化流程，理解这些事件至关重要：

```typescript
import { query, type ConversationEvent } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: 'What files are in this directory?',
  options: { cwd: process.cwd(), permissionMode: 'acceptEdits' }
});

for await (const event of conversation) {
  switch (event.type) {
    case 'text':
      // 模型文本输出，按 token 流式返回
      process.stdout.write(event.text);
      break;

    case 'tool_use':
      // 模型调用工具
      console.log(`\n[Tool] ${event.name}(${JSON.stringify(event.input)})`);
      break;

    case 'tool_result':
      // 工具返回结果
      console.log(`[Result] ${event.content?.substring(0, 100)}...`);
      break;

    case 'error':
      // 发生错误
      console.error(`[Error] ${event.error}`);
      break;

    case 'done':
      // 对话结束
      console.log('\n[Done]');
      break;
  }
}
```

### 收集完整响应

很多时候你并不需要流式片段，而是想要完整的文本响应：

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

## 权限，程序化控制

这一点必须处理妥当：CLI 默认策略是“问人”，但脚本没法交互确认。因此，SDK 提供了几种权限模式：

| Mode | 行为 | 适用场景 |
|------|------|----------|
| `default` | 任何通常需要确认的工具都会报错 | 测试、安全探索 |
| `acceptEdits` | 自动接受文件编辑，Shell 操作仍需确认 | 自动化重构 |
| `bypassPermissions` | 自动接受所有操作 | 完全可信的 CI（高风险） |
| Custom callback | 每次调用时由你决定是否放行 | 生产脚本 |

![SDK 权限模式：从最安全到最自主的光谱](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig6_permission_modes.png)  
*图：SDK 权限模式：从最安全到最自主的光谱*

### 自定义回调详解

真正投入生产时，建议直接使用回调函数：

```typescript
import { query } from '@anthropic-ai/claude-code';

const conversation = query({
  prompt: 'Fix the failing tests and update the changelog',
  options: {
    cwd: process.cwd(),
    permissionMode: 'custom',
    permissionCallback: async (toolName, toolInput) => {
      // 所有读操作放行
      if (toolName === 'Read') return 'allow';

      // 仅允许写入 src/ 与 tests/ 范围内的文件
      if (toolName === 'Write' || toolName === 'Edit') {
        const path = toolInput.path || toolInput.file_path;
        if (path?.startsWith('src/') || path?.startsWith('tests/')) {
          return 'allow';
        }
        return 'deny';
      }

      // 危险的 Shell 命令一律拒绝
      if (toolName === 'Bash') {
        const cmd = toolInput.command || '';
        if (cmd.startsWith('rm ') || cmd.includes('sudo')) return 'deny';
        return 'allow';
      }

      return 'deny';
    }
  }
});

for await (const event of conversation) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

这样一来，权限策略就实现了程序化控制。脚本可以无人值守运行，而你也清楚它绝不会做你明确禁止的事情。

## 实战脚本：自动更新 CHANGELOG

我在好几个项目的 `scripts/update-changelog.ts` 里都用了这段代码：

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

每次发布前运行一次：CHANGELOG 自动生成，commit log 被整理成通顺的叙述文字，你只需审查并提交。每次发布能省下约 5 分钟；过去半年累计下来，节省的时间相当可观。

## SDK 与 CLI 对比

什么时候该用 SDK，什么时候该用 CLI？下面是详细对比：

| 方面 | CLI（`claude`） | SDK（`@anthropic-ai/claude-code`） |
|------|----------------|-------------------------------------|
| **接口形式** | 终端中的交互式聊天 | 可编程的异步可迭代对象 |
| **权限控制** | 交互式提示授权 | 基于回调或预设模式 |
| **适用场景** | 探索性工作、一次性任务 | 自动化、CI/CD、脚本集成 |
| **会话状态** | 持久化，直到执行 `/clear` | 每次 `query()` 调用都是全新会话 |
| **启动耗时** | 约 2 秒 | 约 1 秒（无 TUI 开销） |
| **输出格式** | 面向人类阅读的格式化输出 | 面向代码处理的强类型事件 |
| **错误处理** | 在聊天界面中直接显示错误 | 抛出异常或产出 error 事件 |
| **Hooks** | 来自 `settings.json` | 来自 `settings.json`（行为一致） |
| **MCP servers** | 来自 `settings.json` | 来自 `settings.json`（行为一致） |
| **CLAUDE.md** | 自动加载 | 自动加载（行为一致） |
| **多轮对话** | 通过自然聊天流实现 | 需手动构建 conversation 数组 |
| **并行执行** | 单次运行 | 支持多个并发 `query()` 调用 |

关键在于：SDK 和 CLI 共享同一底层引擎。差异仅在接口形态，不在能力边界。Hooks、MCP servers、CLAUDE.md 以及所有配置项的行为完全一致。

## GitHub Action

![Claude Code 实战 (6)：SDK、GitHub 集成和 CI 中的 Claude — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/illustration_2.png)

![GitHub Action 生命周期：从 @claude 评论到 PR 回复](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig4_action_flow.png)  
*图：GitHub Action 生命周期：从 @claude 评论到 PR 回复*

Anthropic 官方提供了 Action：`anthropic/claude-code-action@v1`。

### 基础配置

把它加到你的 workflow 中：

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

现在，仓库里的任何人只要在 PR 评论里写 `@claude please review the failing test`，Claude 就会：

1. 检出对应分支  
2. 读取评论线程获取上下文  
3. 执行所需操作（运行测试、linter、读取文件等）  
4. 在 PR 中回复分析结果  
5. 如果被要求，还能推送新提交  

### 详细工作流配置

下面是一个支持多种触发条件的完整配置示例：

```yaml
# .github/workflows/claude-full.yml
name: Claude Code Assistant

on:
  # 响应 PR 评论中的 @claude 提及
  pull_request_review_comment:
    types: [created]
  issue_comment:
    types: [created]

  # 自动审查新 PR
  pull_request:
    types: [opened, synchronize]

jobs:
  # 任务 1：响应 @claude 提及
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

  # 任务 2：自动审查新 PR
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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `anthropic-api-key` | 必填 | Anthropic API 密钥 |
| `prompt` | 来自评论内容 | 覆盖默认提示词（适用于自动化触发） |
| `max-turns` | `30` | Agent 最大交互轮数，超限即停止 |
| `timeout-minutes` | `30` | 最长执行时间（分钟） |
| `model` | 默认模型 | 指定具体使用的模型 |
| `allowed-tools` | 全部可用工具 | 限制 Claude 可调用的工具列表 |

### 密钥管理

请将 API 密钥以 GitHub 仓库 Secret 的形式存储：

1. 进入 Settings > Secrets and variables > Actions  
2. 点击 “New repository secret”  
3. 名称填 `ANTHROPIC_API_KEY`，值填你的密钥  
4. 工作流中通过 `${{ secrets.ANTHROPIC_API_KEY }}` 引用  

**切勿在 workflow 文件中硬编码 API 密钥**——它们会直接暴露在代码仓库中。

这个 Action 会读取仓库中的 `.claude/settings.json`。你在第 5 篇写的 hooks 依然生效，第 3 篇定义的斜杠命令也能用——比如 `@claude /review` 会按预期工作。

![PR 评审序列：开发者、GitHub、Claude Action 三方交互](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig7_pr_sequence.png)  
*图：PR 评审序列：开发者、GitHub、Claude Action 三方交互*

## PR 审查工作流 — 详细流程

下面完整演示一次从开始到结束的 PR 审查流程。

### 初始配置（仅需一次）

1. 将上述 workflow YAML 文件放入 `.github/workflows/` 目录  
2. 在仓库 Secrets 中添加 `ANTHROPIC_API_KEY`  
3. 提交并推送变更  

### 日常工作流

开发者提交 PR 后，自动审查任务随即触发：

```markdown
Claude Code 正在审查 PR #42："添加用户认证中间件"

---

## 审查摘要

### 致命问题

**1. 缺少速率限制（src/middleware/auth.ts:45）**
认证中间件未对失败尝试做速率限制。攻击者可能暴力枚举令牌。建议添加：

    import rateLimit from 'express-rate-limit';
    const authLimiter = rateLimit({ windowMs: 15*60*1000, max: 100 });

**2. 令牌存于 localStorage（src/utils/auth.ts:12）**
localStorage 易受 XSS 攻击。建议改用 `httpOnly` Cookie。

### 建议项

- 补充令牌刷新流程的测试（当前缺失）
- `validateToken` 函数静默吞掉了错误 —— 请记录日志
- 考虑将 JWT 配置提取至环境变量

### 值得肯定之处
- 认证中间件与业务逻辑职责分离清晰
- TypeScript 泛型在令牌类型定义中使用得当
- 主路径（happy path）测试覆盖率良好
```

审查者阅读 Claude 的分析，修复关键问题，并回复：

```text
@claude 发现得很准。请修复速率限制问题：在 auth 中间件中集成 express-rate-limit，
窗口设为 15 分钟、上限 100 次请求；同时补充对应测试。
```

Claude 解析这条评论，检出对应分支，完成修改、运行测试，并推送提交：

```text
我已执行以下变更：

1. 添加 `express-rate-limit` 依赖
2. 在 auth 中间件中启用速率限制（15 分钟窗口，100 次请求上限）
3. 在 `tests/middleware/auth.test.ts` 中新增速率限制行为测试

提交：abc1234 "为 auth 中间件添加速率限制"

全部测试通过（47 通过，0 失败）。
```

这个循环——审查 → 请求修改 → Claude 实施——可以按需重复多次。PR 的对话本身就成了交互界面。

## CI 集成模式

除了 GitHub Action，你还可以将 Claude 集成到现有 CI 流程中。以下是几种常见模式。

### 模式 1：预合并检查（Pre-merge checks）

在 PR 合并前运行 Claude 执行安全审查：

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

当 API 或模型文件变更时，自动同步文档：

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

### 模式 3：发布说明生成（Release notes generation）

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

如果 GitHub Action 不符合你的需求，可以直接调用 SDK：

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
          node --experimental-vm-modules << 'JSEOF'
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
          JSEOF
```

这种方式让你完全掌控提示词、权限策略和输出处理逻辑。

## 高级 SDK 模式

### 多轮对话

基础的 `query()` 是单轮的。如需多轮对话，请自行维护消息历史：

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

对于彼此独立的分析任务，可以并行运行多个 Claude 调用：

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

// 并行运行分析任务
const results = await Promise.all([
  runTask('Security', 'Audit this codebase for security vulnerabilities. Read key files.'),
  runTask('Performance', 'Identify performance bottlenecks in the hot paths. Check database queries.'),
  runTask('Debt', 'Find the top 5 areas of technical debt. Look for TODO comments and complex functions.'),
]);

console.log(results.join('\n\n---\n\n'));
```

### 流式输出到文件

对于长时间运行的任务，可以一边将输出流式写入文件，一边打印到控制台：

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

## 故障排除

### 常见 SDK 问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `ANTHROPIC_API_KEY not set` | 缺少环境变量 | 在 shell 或 CI 中导出该密钥 |
| 工具调用时报 `Permission denied` | `default` 模式禁止所有操作 | 切换至 `acceptEdits` 模式，或实现自定义回调 |
| 长任务超时 | 默认超时时间太短 | 在 options 中设置更长的 `timeout` |
| `Cannot find module` | 导入路径错误 | 使用 `@anthropic-ai/claude-code`，而非 `claude-code` |
| 返回空响应 | 模型未生成文本 | 检查流中是否包含 error 事件 |

### 常见 GitHub Action 问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| Action 从未触发 | `if` 条件语法错误 | 检查 `contains()` 用法及事件类型 |
| "Resource not accessible" 错误 | 缺少必要权限 | 在 workflow 中添加 `permissions` 配置块 |
| Claude 无法推送提交 | `actions/checkout` 未指定 ref | 在 checkout 步骤中加上 `ref: ${{ github.head_ref }}` |
| 运行 30 分钟后超时 | 任务过于复杂 | 增加 `timeout-minutes`，或缩小任务范围 |
| PR 上无响应 | Secret 未配置 | 确认仓库 Secrets 中已正确设置 `ANTHROPIC_API_KEY` |

### 调试 workflow 运行

```bash
# 查看最近 5 次 workflow 运行
gh run list --limit 5

# 查看某次运行的完整日志
gh run view RUN_ID --log

# 重新运行失败的 workflow
gh run rerun RUN_ID
```

## 我拿 GitHub Action 干什么

有三件事我已经离不开它了：

**1. PR 初审。** 新 PR 一开，workflow 就自动运行 `@claude /review`。等人工审查者点进来时，对话里已经有第一轮分析结果了。每次 PR 能省下 10 分钟，还能抓出那些显而易见的问题。

**2. Issue 总结。** 新 Issue 触发 workflow，自动打标签、建议修复范围、关联相关代码。报告者更快得到回应，维护者也有了起手式。

**3. 文档更新。** Schema 一变，Action 就跑起来让 Claude 同步更新文档。虽然不总是完美，但大约 80% 的工作都是机械性的。

## SDK 和 Action 的边界

![CLI 和 SDK 区别在接口而非能力——底层是同一个引擎](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/06-sdk-and-github/fig5_cli_vs_sdk.png)  
*图：CLI 和 SDK 区别在接口而非能力——底层是同一个引擎*

我用 SDK 的时候：

- 想在本地脚本里嵌入 Claude  
- 触发条件是 cron、文件变更或手动命令  
- 需要程序化检查事件流  
- 想通过回调实现细粒度权限控制  

我用 Action 的时候：

- 触发条件是 GitHub 事件（PR、Issue、评论）  
- 输出结果要落在 PR 或 Issue 里  
- 希望人在环路中，且通过 `@mention` 触发交互  
- 追求最简单的接入方式  

当然，两者有重叠——Action 底层就是基于 SDK 构建的。

## 把系列串起来

六篇文章构成一条清晰的进阶路径：

1. 安装与三层 config  
2. 快捷键与思考模式  
3. 个人工作流中的斜杠命令  
4. MCP 实现外部集成  
5. Hooks 构建安全防线  
6. SDK 与 GitHub Action 打造程序化基础设施  

每一篇单独看都能带来具体收益；合在一起，Claude Code 就从一个代码聊天客户端，变成了内嵌于仓库的可编程基础设施。

我观察过的高手用户，都有一个共同点：他们把 `.claude/` 当作代码库的一部分。配置、命令、hooks 全部提交、全部走 PR 审查、全部随项目演进。这才是值得长期培养的工程习惯。

放心交付。
