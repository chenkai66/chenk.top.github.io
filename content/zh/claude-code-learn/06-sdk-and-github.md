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
CLI 是門面， SDK 提供核心能力， GitHub 集成才是價值真正釋放的關鍵環節。

![Claude Code 实战 (6)：SDK、GitHub 集成和 CI 中的 Claude — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_1.png)

## 一段話說清 SDK

`@anthropic-ai/claude-code` 就是那個 npm 包。它暴露了 CLI 用的同一個 Claude Code 引擎，工具和權限一模一樣，只不過變成了程序化接口。你傳入一個 prompt，它便返回一個可遍歷對話事件的異步迭代器——腳本、服務、 CI 步驟，隨處可嵌入。

![SDK 和 CLI 共享同一个 agent 引擎、工具与权限](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/fig3_sdk_arch.png)
*图： SDK 和 CLI 共享同一个 agent 引擎、工具与权限*

## SDK 安装与配置

### 前置条件

需安裝 Node.js 18+ 並配置 Anthropic API key； SDK 的 agent 循環與 CLI 完全一致，故憑據亦須相同。

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

你會看到相同的 agent 循環在終端中執行——包含工具調用。本質上，這就是一個剝離了聊天界面的 CLI。

## 理解对话事件

`query()` 返回的异步可迭代对象（async iterable）产出类型化的事件。掌握这些事件是构建真实自动化流程的关键：

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

多数场景下你需要的是完整的文本响应，而非流式片段：

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

## 權限，程序化控制

權限配置必須精確： CLI 默認策略是「問人」，而腳本無法觸發交互式確認。所以 SDK 暴露了幾種權限模式：

| Mode | 行为 | 适用场景 |
|------|------|----------|
| `default` | 任何通常需要確認的工具都會報錯 | 测试、安全探索 |
| `acceptEdits` | 自動接受文件編輯， Shell 操作會問 | 自动化重构 |
| `bypassPermissions` | 自動接受所有操作（危險） | 完全可信的 CI （高风险） |
| Custom callback | 你提供一個 callback，每次調用時決定 | 生产脚本 |


![SDK 权限模式：从最安全到最自主的光谱](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/fig6_permission_modes.png)
*图： SDK 权限模式：从最安全到最自主的光谱*

### 自定义回调详解

實際投入生產時，建議直接使用回調函數：

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

這樣，權限策略就實現了程序化控制。腳本可以無人值守運行，你也清楚它絕對幹不了你明確禁止的事。

## 實戰腳本：自動更新 CHANGELOG

我幾個項目的 `scripts/update-changelog.ts` 裡都掛著這個：

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

每次發布前執行一次： CHANGELOG 自動生成， commit log 被梳理為通順文字，你只需審查並提交。每次發布可節省約 5 分鐘；累計過去六個月的所有發布，節省時間效益顯著。

## SDK 与 CLI 对比

何时该选用 SDK，何时该选用 CLI？以下是详细对比：

| 方面 | CLI （`claude`） | SDK （`@anthropic-ai/claude-code`） |
|------|----------------|-------------------------------------|
| **接口形式** | 终端中交互式聊天 | 可编程的异步可迭代对象 |
| **权限控制** | 交互式提示授权 | 基于回调或模式（mode-based） |
| **适用场景** | 探索性工作、一次性任务 | 自动化、 CI/CD、脚本集成 |
| **会话状态** | 持久化，直至执行 `/clear` | 每次 `query()` 调用均为全新会话 |
| **启动耗时** | 约 2 秒 | 约 1 秒（无 TUI 开销） |
| **输出格式** | 面向人类阅读的格式化输出 | 面向代码处理的强类型事件 |
| **错误处理** | 在聊天界面中直接显示错误 | 抛出异常或产出 error 事件 |
| **Hooks** | 来自 `settings.json` | 来自 `settings.json`（行为一致） |
| **MCP servers** | 来自 `settings.json` | 来自 `settings.json`（行为一致） |
| **CLAUDE.md** | 自动加载 | 自动加载（行为一致） |
| **多轮对话** | 通过自然聊天流实现 | 需手动构建 conversation 数组 |
| **并行执行** | 单次运行 | 支持多个并发 `query()` 调用 |

核心要点： SDK 与 CLI 共享同一底层引擎。差异仅在于接口形态，而非能力边界。 Hooks、 MCP servers、 CLAUDE.md 及所有配置项的行为完全一致。

## GitHub Action

![Claude Code 实战 (6)：SDK、GitHub 集成和 CI 中的 Claude — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_2.png)


![GitHub Action 生命周期：从 @claude 评论到 PR 回复](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/fig4_action_flow.png)
*图： GitHub Action 生命周期：从 @claude 评论到 PR 回复*

Anthropic 出了官方 Action：`anthropic/claude-code-action@v1`。

### 基础配置

加到 workflow 裡：

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

現在倉庫裡任何人只要在 PR 評論裡寫 `@claude please review the failing test`， Claude 就會：

1. Checkout 分支
2. 讀取評論線程獲取上下文
3. 運行需要的操作（測試、 lint、讀文件）
4. 在 PR 裡回復分析結果
5. 如果被要求，還可以推送 commit

### 详细工作流配置

以下是一个包含多种触发条件的完整配置示例：

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
| `prompt` | 来自评论内容 | 覆盖默认提示词（适用于自动化触发场景） |
| `max-turns` | `30` | Agent 最大交互轮数，超限即停止 |
| `timeout-minutes` | `30` | 最长执行时间（分钟） |
| `model` | 默认模型 | 指定要使用的具体模型 |
| `allowed-tools` | 全部可用工具 | 限制 Claude 可调用的工具列表 |

### 密钥管理

将你的 API 密钥以 GitHub 仓库 Secret 方式存储：

1. 进入 Settings > Secrets and variables > Actions
2. 点击 "New repository secret"
3. 名称填 `ANTHROPIC_API_KEY`，值填你的密钥
4. 工作流中通过 `${{ secrets.ANTHROPIC_API_KEY }}` 引用

**切勿在 workflow 文件中硬编码 API 密钥**——它们会直接暴露在代码仓库中。

Action 會認倉庫裡的 `.claude/settings.json`。你在第 5 篇寫的 hooks 依然生效。你在第 3 篇寫的 slash commands 也能用——`@claude /review` 會按預期工作。


![PR 评审序列：开发者、GitHub、Claude Action 三方交互](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/fig7_pr_sequence.png)
*图： PR 评审序列：开发者、 GitHub、 Claude Action 三方交互*

## PR 审查工作流 — 详细流程

下面完整演示一次从开始到结束的 PR 审查工作流。

### 初始配置（仅需一次）

1. 将上述 workflow YAML 文件添加至 `.github/workflows/` 目录
2. 在仓库 Secrets 中添加 `ANTHROPIC_API_KEY`
3. 提交并推送变更

### 日常工作流

开发者提交 PR 后，自动审查任务随即触发：

```
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

审查者阅读 Claude 的分析后，修复关键问题，并回复：

```
@claude 发现得很准。请修复速率限制问题：在 auth 中间件中集成 express-rate-limit，
窗口设为 15 分钟、上限 100 次请求；同时补充对应测试。
```

Claude 解析该评论，检出对应分支，完成修改、运行测试并推送提交：

```
我已执行以下变更：

1. 添加 `express-rate-limit` 依赖
2. 在 auth 中间件中启用速率限制（15 分钟窗口，100 次请求上限）
3. 在 `tests/middleware/auth.test.ts` 中新增速率限制行为测试

提交：abc1234 "为 auth 中间件添加速率限制"

全部测试通过（47 通过，0 失败）。
```

此循环 —— 审查 → 请求修改 → Claude 实施 —— 可按需多次重复。 PR 对话本身即为交互界面。

## CI 集成模式

除 GitHub Action 外，还可将 Claude 集成到现有 CI 流程中，常见模式如下。

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

若 GitHub Action 不满足需求，可直接调用 SDK：

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

该方式提供对提示词（prompt）、权限控制与输出处理的完全自主权。

## 高级 SDK 模式

### 多轮对话

基础 `query()` 是单轮的。如需多轮对话，请自行管理消息历史：

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

对独立分析任务，并行运行多个 Claude 调用：

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

对于长时间运行的任务，可将输出同时流式写入文件并打印到控制台：

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
| 工具调用时出现 `Permission denied` | `default` 模式禁止所有操作 | 切换至 `acceptEdits` 模式，或实现自定义回调函数 |
| 长任务超时 | 默认超时时间过短 | 在 options 中设置更长的 `timeout` |
| `Cannot find module` | 导入路径错误 | 使用 `@anthropic-ai/claude-code`，而非 `claude-code` |
| 返回空响应 | 模型未生成文本输出 | 检查流中是否触发了 `error` 事件 |

### 常见 GitHub Action 问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| Action 从未触发 | `if` 条件语法错误 | 核查 `contains()` 用法及事件类型 |
| "Resource not accessible" 错误 | 缺少必要权限 | 在 workflow 中添加 `permissions` 配置块 |
| Claude 无法推送提交 | `actions/checkout` 未指定 ref | 在 checkout 步骤中添加 `ref: ${{ github.head_ref }}` |
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

## 我拿 GitHub Action 幹什麼

有三件事我離不開了：

**1. PR 初審。** 新 PR 一開， workflow 自動跑 `@claude /review`。等人來看的時候，對話裡已經有了第一輪審查結果。每次 PR 省審查者 10 分鐘，還能抓住那些顯而易見的問題。

**2. Issue 總結。** 新 Issue 觸發 workflow，自動打標籤、建議修復範圍、關聯相關代碼。報告者更快得到回應，維護者也有了起手式。

**3. 文檔更新。** Schema 一變， Action 就跑起來讓 Claude 同步更新文檔。不總是完美，但 ~80% 的工作都是機械性的。

## SDK 和 Action 的邊界

![CLI 和 SDK 区别在接口而非能力——底层是同一个引擎](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/fig5_cli_vs_sdk.png)
*图： CLI 和 SDK 区别在接口而非能力——底层是同一个引擎*


我用 SDK 的時候：

- 我想在本地跑的腳本裡嵌入 Claude
- 觸發條件是 cron、文件變更或手動命令
- 我需要程序化檢查事件
- 想通过回调实现细粒度的权限控制

我用 Action 的時候：

- 觸發條件是 GitHub 事件（PR、 Issue、评论）
- 輸出結果要落在 PR 或 Issue 裡
- 我想要人在環路裡，且環路通過 `@mention` 觸發
- 我想要尽可能简单的接入方式

當然有重疊。 Action 底層就是基於 SDK 構建的。

## 把系列串起來

六篇文章構成完整遞進路徑：
1. 安裝與三層 config；
2. 快捷方式與 modes；
3. 個人工作流中的 slash commands；
4. 外部集成： MCP；
5. 安全防線： hooks；
6. 程序化基礎設施： SDK 與 GitHub Action。

單獨閱讀每一篇文章，都能獲得一項具體可用的能力。合在一起， Claude Code 就從代碼聊天客戶端變成了住在倉庫裡的程序化基礎設施。

我觀察過的那些高手，只有一個共同特質：他們把 `.claude/` 當作代碼庫的一部分。配置、命令、 hooks 全部提交，全部在 PR 裡審查，全部隨項目進化。這才是值得長期堅持的工程習慣。

放心交付。
