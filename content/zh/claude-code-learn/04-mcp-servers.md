---
title: "Claude Code 实战（四）：MCP 服务器连接万物"
date: 2026-04-21 09:00:00
tags:
  - claude-code
  - mcp
  - playwright
  - integration
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 4
description: "MCP 是让 Claude Code 伸出文件系统的插件协议。装一个（Playwright），端到端看它跑通，再学会保护它不发疯的权限模型。"
disableNunjucks: true
translationKey: "claude-code-learn-4"
---
如果你在 Claude Code 中只学一种扩展机制，那就学 MCP。这是自动补全和平台之间的区别。

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_1.png)

---

## 秒快速介绍

MCP 是 **Model Context Protocol** —— 一个小型开放规范，允许 Claude Code 调用外部服务器上的工具并读取资源。“服务器”可以是任何通过 stdio 或 HTTP 实现 MCP 的进程。Claude 将 MCP 工具视为内置工具：模型决定何时调用它们，你确认或自动批准，结果以文本形式返回。

这意味着 Claude 可以：

- 驱动浏览器（Playwright MCP）
- 查询 Postgres 数据库（Postgres MCP）
- 读取你的 Notion 或 Obsidian 库
- 与你的内部 API 通信
- 任何你能用 200 行 Node 脚本封装的功能

MCP 服务器大多由社区构建。目录可在 `mcp.so` 找到，Anthropic 的文档中也有一个不断增长的列表。优秀的服务器通常简短、专注，并对其功能诚实透明。

## 理解 MCP 协议

在安装任何东西之前，先了解底层实际发生的事情会很有帮助。MCP 是一个基于 JSON-RPC 2.0 的协议，具有特定的生命周期：

**1. 初始化。** Claude Code 启动 MCP 服务器进程（或连接到远程服务器）。双方在 `initialize` 握手过程中交换能力。服务器声明它支持的内容——工具、资源、提示等——客户端确认它想使用哪些功能。

**2. 发现。** Claude Code 调用 `tools/list` 获取可用工具的完整目录。每个工具都带有名称、描述和输入参数的 JSON Schema。这样模型就知道它可以调用什么以及传递哪些参数。

**3. 调用。** 在对话过程中，当模型决定使用某个 MCP 工具时，Claude Code 会发送一个包含工具名称和参数的 `tools/call` 请求。服务器执行逻辑并返回结果——文本、图像或结构化数据。

**4. 关闭。** 当 Claude Code 退出或你移除服务器时，它会发送关闭通知并终止进程。

整个协议规范出奇地简短。你可以在 `spec.modelcontextprotocol.io` 上大约 30 分钟内读完。关键在于 MCP 在调用之间是无状态的——每个 `tools/call` 都是独立的。服务器不需要跟踪对话上下文；Claude Code 负责处理这一点。

![MCP server lifecycle: spawn, initialize handshake, tools/list discovery, tools/call invocation loop, shutdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_lifecycle_sequence.png)

*MCP 会话的四个阶段。Init 和 Discover 在启动时只发生一次；Invoke 是模型在对话中唯一循环执行的部分；Shutdown 在 Claude Code 退出或你移除服务器时触发。*

### Stdio 与 SSE 传输

MCP 支持两种传输机制，理解它们的区别对于配置和调试服务器至关重要：

**Stdio 传输** 是默认且最常见的。Claude Code 将 MCP 服务器作为子进程启动，并通过 stdin/stdout 进行通信。每个 JSON-RPC 消息作为一行写入 stdout；Claude Code 将请求写入进程的 stdin。

stdio 的优势：
- 无需网络配置
- 无端口冲突
- 进程生命周期自动管理
- 在防火墙和 VPN 后也能工作
- 调试最简单——你可以直接读取管道

**SSE（Server-Sent Events）传输** 用于远程服务器。MCP 服务器作为 HTTP 服务运行，Claude Code 通过 Server-Sent Events 接收服务器到客户端的消息，并通过 HTTP POST 发送客户端到服务器的消息。

SSE 的优势：
- 服务器可以在不同机器上运行
- 多个客户端可以共享一个服务器实例
- 服务器在 Claude Code 会话之间持久存在
- 可以部署在带身份验证的反向代理后面

实际上，你使用的 90% 的 MCP 服务器都是 stdio。SSE 适用于需要共享服务的场景——公司范围的数据库代理、集中式工具服务器等。

```bash
# Stdio: Claude Code spawns the process
Claude Code  ──stdin/stdout──>  MCP Server Process (local)

# SSE: Claude Code connects over HTTP
Claude Code  ──HTTP POST──>  MCP Server (remote)
             <──SSE──────
```

![Side-by-side comparison of stdio and SSE MCP transports, with pros and use cases](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_stdio_vs_sse.png)

*Stdio 是本地默认选项——单客户端、单子进程、零网络。SSE 则反转了拓扑结构：一个长期运行的服务器、多个客户端，但你现在需要负责部署。*

## 安装你的第一个 MCP 服务器

Playwright 是典型的“第一个 MCP”，因为它让价值显而易见。从你的 shell 中操作——*不是* 在 Claude Code 内部：

![MCP Server Communication Flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

MCP 服务器是独立的进程。Claude Code 通过 stdio（或 HTTP，用于远程服务器）与之通信。工具在握手阶段被发现——Claude 问“你能做什么？”，服务器回复工具 schema 列表。此后，模型就可以像调用内置工具一样调用这些工具。

### 分步指南：Playwright MCP

Playwright MCP 为 Claude 提供了一个真实的浏览器。不是无头爬虫——而是完整的 Chromium 实例，可以导航、点击、截图和检查。

```bash
# Add the server to your project
claude mcp add playwright -- npx @anthropic-ai/mcp-playwright

# Verify it's registered
claude mcp list
```

添加后，重启 Claude Code（或启动新会话）。你应该能在工具列表中看到 Playwright 工具。尝试一个简单的提示：

```text
Navigate to https://news.ycombinator.com and tell me the top 3 stories
```

Claude 会调用 `browser_navigate`，然后调用 `browser_snapshot` 读取页面内容，并总结结果。每次工具首次调用时，你会收到权限提示。

以下是会话中实际的工具调用示例：

```text
Claude wants to use mcp tool: mcp__playwright__browser_navigate
  url: "https://news.ycombinator.com"
> Allow? (y/n/always)

Claude wants to use mcp tool: mcp__playwright__browser_snapshot
> Allow? (y/n/always)
```

### 分步指南：Filesystem MCP

filesystem MCP 服务器为 Claude 提供对项目根目录之外目录的受控访问。默认情况下，Claude Code 只能读写当前工作目录内的文件。filesystem MCP 扩展了这一能力——并带有防护措施。

```bash
# Allow Claude to read your SSH config and dotfiles (read-only)
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem \
  /Users/you/.config \
  /Users/you/.ssh/config

# The paths you pass are the allowed directories.
# The server refuses access to anything outside them.
```

现在你可以问 Claude 类似这样的问题：

```text
Read my SSH config and tell me which hosts I have configured
```

```text
Check my git config in ~/.config/git/config and suggest improvements
```

filesystem 服务器强制执行严格的白名单。即使模型尝试读取 `/etc/passwd`，服务器也会返回错误。这是按设计实现的沙箱。

### 分步指南：GitHub MCP

GitHub MCP 服务器为 Claude 提供对 GitHub API 的直接访问——issues、PRs、repos 等。它比直接调用 `gh` CLI 更丰富，因为模型能获得结构化数据。

```bash
# You need a GitHub personal access token
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxx

# Add the server
claude mcp add github -- npx -y @modelcontextprotocol/server-github
```

现在 Claude 可以：

```text
List all open issues in this repo labeled "bug"
```

```text
Show me the review comments on PR #42
```

```text
Create a new issue titled "Fix login timeout" with the label "backend"
```

GitHub MCP 提供了针对仓库、issues、pull requests、分支、文件和搜索的工具。它是工具最丰富的 MCP 服务器之一——大约有 30 多个工具。

### 分步指南：Postgres MCP

Postgres MCP 服务器将 Claude 连接到 PostgreSQL 数据库。它专为只读探索设计——分析 schema、运行查询、解释表结构。

```bash
# Add with your connection string
claude mcp add postgres -- npx -y @modelcontextprotocol/server-postgres \
  "postgresql://user:password@localhost:5432/mydb"
```

现在你可以进行类似这样的对话：

```text
Show me the schema of the users table
```

```text
How many orders were placed in the last 7 days, grouped by status?
```

```text
Find any users who signed up but never placed an order
```

服务器对你的数据库运行查询并将结果以格式化文本返回。几点安全注意事项：

- 使用只读数据库用户。服务器本身不强制只读模式。
- 不要用具有写权限的用户指向生产数据库。使用副本。
- 包含密码的连接字符串最终会出现在你的设置文件中。改用环境变量：

```bash
claude mcp add postgres -- npx -y @modelcontextprotocol/server-postgres \
  "$DATABASE_URL"
```

### 分步指南：Slack MCP

Slack MCP 服务器允许 Claude 搜索你的 Slack 工作区、读取频道并发布消息。

```bash
# Requires a Slack bot token with appropriate scopes
export SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx

claude mcp add slack -- npx -y @anthropic-ai/mcp-slack
```

有用的提示：

```text
Search Slack for messages about the deployment failure yesterday
```

```text
Summarize the #engineering channel from the last 24 hours
```

```text
Post a standup update to #team-backend summarizing what I committed today
```

这个功能很强大，但我一直保持每次调用都需要确认。一个能向整个公司 Slack 发布消息的工具值得仔细监督。

## settings.json 中的配置

当你运行 `claude mcp add` 时，服务器会被注册到 `.claude/settings.json` 中。以下是多服务器设置的示例：

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@anthropic-ai/mcp-playwright"],
      "env": {}
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/.config"
      ],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "${DATABASE_URL}"
      ],
      "env": {}
    }
  }
}
```

需要注意几点：

- **环境变量** 使用 `${VAR}` 语法引用。Claude Code 在启动时解析它们。切勿在设置文件中硬编码密钥。
- **`args` 数组** 直接传递给命令。这里不会发生 shell 插值。
- **服务器名称**（如 `playwright`、`github`）是你在权限和对话中引用它们的方式。

### 配置作用域

MCP 服务器可以在三个级别注册：

| Scope | File | Applies to |
|-------|------|------------|
| Project | `.claude/settings.json` | 所有克隆此 repo 的人 |
| User-project | `.claude/settings.local.json` | 仅你自己，在此 repo 中 |
| Global | `~/.claude/settings.json` | 你所有的项目 |

项目级适用于整个团队应使用的服务器（如用于测试的 Playwright、用于探索的数据库）。用户项目级适用于个人生产力服务器（如你的 Obsidian 库、你的私有 API）。全局级适用于你希望在所有地方使用的服务器（如对 dotfiles 的 filesystem 访问）。

![Three configuration scopes stacked from global to project, showing precedence and typical contents](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_config_scopes.png)

*作用域从最具体到最宽泛解析。对希望提交到 repo 的服务器使用项目作用域，对个人令牌使用用户项目作用域，对跨项目工具使用全局作用域。*

## 权限——你绝不能跳过的部分

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_2.png)

每个 MCP 工具都有一个权限级别。模型首次调用时，你会收到确认对话框。你有三个选项：

- **Allow once** —— 仅本次调用
- **Allow for this session** —— 直到你 `/clear` 或退出
- **Always allow** —— 写入 `.claude/settings.json`

我不建议对任何会改变状态的 MCP 使用“always allow”。只读工具（搜索、获取、查询）——可以。任何写文件、发消息或调用有副作用 API 的工具——每次都确认，或严格限定权限范围。

在 `.claude/settings.json` 中的正确模式：

```json
{
  "permissions": {
    "allow": [
      "mcp__playwright__browser_navigate",
      "mcp__playwright__browser_snapshot",
      "mcp__playwright__browser_click",
      "mcp__github__list_issues",
      "mcp__github__get_pull_request",
      "mcp__postgres__query"
    ],
    "deny": [
      "mcp__playwright__browser_run_code_unsafe",
      "mcp__github__create_issue",
      "mcp__slack__post_message"
    ]
  }
}
```

这会自动允许只读操作，并阻止任何改变外部状态的操作。支持按工具粒度控制。

![Decision tree for MCP tool permission resolution: deny check, allow check, user prompt with three options](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_permission_flow.png)

*每个 MCP 工具调用都会走这个决策树。拒绝列表优先短路；白名单中的工具自动执行；其余情况会弹出三选项提示。*

### 权限命名约定

MCP 工具权限遵循 `mcp__SERVERNAME__TOOLNAME` 模式。双下划线是有意为之。你也可以使用正则表达式模式：

```json
{
  "permissions": {
    "allow": [
      "mcp__playwright__browser_snapshot",
      "mcp__playwright__browser_navigate"
    ],
    "deny": [
      "mcp__playwright__browser_run_code_unsafe"
    ]
  }
}
```

如果你想允许某个服务器的所有工具（针对只读服务器），可以更宽松。但我更喜欢显式白名单。

### 常见权限陷阱

**Gotcha 1: npx 首次运行时下载。** MCP 服务器首次运行时，`npx` 可能会下载包。这需要几秒钟，看起来像服务器卡住了。其实没坏——只需等待。

**Gotcha 2: 权限提示疲劳。** 如果每次 MCP 调用都弹出权限提示，说明你做错了。将安全的只读工具加入白名单。写操作工具保持手动确认。

**Gotcha 3: 环境变量未解析。** 如果你的服务器需要环境变量但 shell 中未设置，服务器会静默失败或报出晦涩错误。在责怪 MCP 服务器前，始终先用 `echo $VAR_NAME` 验证。

**Gotcha 4: 服务器名称冲突。** 如果两个设置文件（如项目和全局）定义了相同的服务器名称，更具体的那个会胜出。如果你期望全局配置生效，这可能会令人困惑。

## 我实际使用的有用 MCP 服务器

在尝试了约 30 个后，以下这些获得了永久位置：

| Server | Package | What it does | Why it's worth it |
|--------|---------|--------------|-------------------|
| Playwright | `@anthropic-ai/mcp-playwright` | Browser automation | Replaces a dozen hand-rolled scrapers |
| Postgres | `@modelcontextprotocol/server-postgres` | Read-only Postgres queries | "Show me the rows" without leaving Claude |
| Filesystem | `@modelcontextprotocol/server-filesystem` | Sandboxed file access outside project root | Read configs in `~/.config/` etc. |
| Slack | `@anthropic-ai/mcp-slack` | Search Slack history, post to channels | Standup summaries write themselves |
| GitHub | `@modelcontextprotocol/server-github` | Issue and PR access via GitHub API | Cleaner than CLI for some queries |
| Fetch | `@anthropic-ai/mcp-fetch` | HTTP GET/POST to any URL | Quick API exploration, web fetching |

模式：每个都专注做好一件事，以单个 npm 或 PyPI 包形式分发，并暴露 3-10 个工具。

### 我评估后弃用的服务器

并非每个 MCP 服务器都值得保留。以下是我尝试过并移除的一些：

| Server | Why I dropped it |
|--------|-----------------|
| Memory server | Interesting concept but the model's built-in context is usually enough |
| Brave Search | Rate limits hit fast; `WebSearch` built-in tool is often sufficient |
| SQLite | Postgres server covers my needs; didn't need a second database tool |
| Docker | Too much power with too little guardrails; prefer scripting Docker commands directly |

## 调试 MCP 连接问题

事情总会出错。以下是系统化调试 MCP 服务器的方法。

### 步骤 1：检查服务器是否已注册

```bash
claude mcp list
```

你应该在列表中看到你的服务器及其命令和参数。如果不在那里，重新添加。

### 步骤 2：独立测试服务器

直接在终端中运行服务器命令：

```bash
# For a stdio server, just run it and see if it starts
npx @anthropic-ai/mcp-playwright

# It should sit there waiting for input on stdin
# If it crashes immediately, you have a dependency issue
```

### 步骤 3：检查错误输出

当服务器在 Claude Code 内部失败时，错误通常会显示在 Claude Code 自己的输出中。查找类似这样的行：

```text
MCP server 'playwright' failed to start: Error: spawn npx ENOENT
```

常见原因：

| Error | Cause | Fix |
|-------|-------|-----|
| `ENOENT` | `npx` not found | Ensure Node.js is in your PATH |
| `ECONNREFUSED` | SSE server not running | Start the remote server first |
| `timeout` | Server took too long to initialize | Check if `npx` is downloading a package |
| `unexpected token` | Version mismatch | Clear npx cache: `npx clear-npx-cache` |

### 步骤 4：启用详细日志

```bash
# Run Claude Code with debug output
CLAUDE_DEBUG=1 claude
```

这会显示 Claude Code 和 MCP 服务器之间的原始 JSON-RPC 消息。你会确切看到通信在哪里中断。

### 步骤 5：检查环境变量

```bash
# Make sure all required env vars are set
env | grep -i github
env | grep -i slack
env | grep -i database
```

当缺少必需的环境变量时，MCP 服务器会静默失败。服务器启动，握手成功，但首次工具调用会因认证错误而失败。

## MCP 不是什么

几点需要澄清：

- **不是沙箱。** MCP 服务器以你的用户权限运行。恶意服务器可以做你能做的任何事。安装前务必审查。
- **不是快速通道。** 每个 MCP 工具调用都是一次进程往返。延迟是真实存在的。不要用 MCP 处理热循环中的任务。
- **不是唯一方式。** 对于一次性集成，有时直接运行 shell 脚本的斜杠命令更合适。MCP 适用于跨会话重复调用的场景。
- **不是流式接口。** MCP 工具返回完整结果，而非流。如果需要实时数据，你需要其他方法。

## 构建自定义 MCP 服务器——完整示例

如果你想自己构建，SDK 有 Node 版的 `@modelcontextprotocol/sdk` 或 Python 版的 `mcp`。让我通过一个真实示例来演示：一个查询 REST API 并将其作为工具暴露的服务器。

### 场景

你有一个内部 API `https://api.internal.company.com` 用于管理部署。你希望 Claude 能列出部署、检查状态并触发回滚。与其教 Claude 使用 `curl`，不如构建一个 MCP 服务器。

### 项目设置

```bash
mkdir mcp-deploy-server
cd mcp-deploy-server
npm init -y
npm install @modelcontextprotocol/sdk zod
```

### 完整服务器

```typescript
// server.ts
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const API_BASE = process.env.DEPLOY_API_URL || 'https://api.internal.company.com';
const API_TOKEN = process.env.DEPLOY_API_TOKEN;

if (!API_TOKEN) {
  console.error('DEPLOY_API_TOKEN environment variable is required');
  process.exit(1);
}

async function apiCall(path: string, method = 'GET', body?: unknown) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

// Create the server
const server = new McpServer({
  name: 'deploy-tools',
  version: '1.0.0',
});

// Tool 1: List deployments
server.tool(
  'list_deployments',
  'List recent deployments with their status',
  {
    environment: z.enum(['staging', 'production']).describe('Target environment'),
    limit: z.number().min(1).max(50).default(10).describe('Number of results'),
  },
  async ({ environment, limit }) => {
    const data = await apiCall(`/deployments?env=${environment}&limit=${limit}`);
    const formatted = data.deployments
      .map((d: any) =>
        `[${d.status}] ${d.id} — ${d.service} @ ${d.version} (${d.deployed_at})`
      )
      .join('\n');

    return {
      content: [{
        type: 'text',
        text: formatted || 'No deployments found.',
      }],
    };
  }
);

// Tool 2: Get deployment details
server.tool(
  'get_deployment',
  'Get detailed information about a specific deployment',
  {
    deployment_id: z.string().describe('The deployment ID'),
  },
  async ({ deployment_id }) => {
    const data = await apiCall(`/deployments/${deployment_id}`);
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(data, null, 2),
      }],
    };
  }
);

// Tool 3: Trigger rollback
server.tool(
  'rollback_deployment',
  'Roll back a deployment to a previous version. This is a destructive action.',
  {
    deployment_id: z.string().describe('The deployment ID to roll back'),
    target_version: z.string().describe('The version to roll back to'),
    reason: z.string().describe('Reason for the rollback'),
  },
  async ({ deployment_id, target_version, reason }) => {
    const data = await apiCall(`/deployments/${deployment_id}/rollback`, 'POST', {
      target_version,
      reason,
    });
    return {
      content: [{
        type: 'text',
        text: `Rollback initiated: ${data.rollback_id}\nStatus: ${data.status}\nETA: ${data.estimated_completion}`,
      }],
    };
  }
);

// Tool 4: Health check
server.tool(
  'check_health',
  'Check the health status of a service in an environment',
  {
    service: z.string().describe('Service name'),
    environment: z.enum(['staging', 'production']).describe('Target environment'),
  },
  async ({ service, environment }) => {
    const data = await apiCall(`/health/${service}?env=${environment}`);
    return {
      content: [{
        type: 'text',
        text: [
          `Service: ${data.service}`,
          `Environment: ${data.environment}`,
          `Status: ${data.status}`,
          `Uptime: ${data.uptime}`,
          `Last check: ${data.last_check}`,
          `Instances: ${data.healthy_instances}/${data.total_instances} healthy`,
        ].join('\n'),
      }],
    };
  }
);

// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);
```

### 在 Claude Code 中注册

```bash
claude mcp add deploy-tools -- npx tsx /path/to/mcp-deploy-server/server.ts
```

或使用环境变量：

```bash
claude mcp add deploy-tools \
  -e DEPLOY_API_TOKEN=$DEPLOY_API_TOKEN \
  -e DEPLOY_API_URL=https://api.internal.company.com \
  -- npx tsx /path/to/mcp-deploy-server/server.ts
```

### 设置权限

在 `.claude/settings.json` 中：

```json
{
  "permissions": {
    "allow": [
      "mcp__deploy-tools__list_deployments",
      "mcp__deploy-tools__get_deployment",
      "mcp__deploy-tools__check_health"
    ],
    "deny": []
  }
}
```

注意我没有将 `rollback_deployment` 加入白名单。这个操作保持手动确认，因为它会改变生产状态。

![Anatomy of the deploy-tools custom MCP server: four tools, three auto-allowed reads and one manual-confirm write, with settings.json fragment](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_custom_server_anatomy.png)

*构建自己的服务器的核心意义：领域操作成为一等工具，权限边界存在于 `settings.json` 中，而非依赖模型的判断。*

### 实际使用

现在在 Claude Code 中，你可以进行类似这样的对话：

```toml
> Show me the last 5 production deployments

Claude calls list_deployments(environment="production", limit=5)

[SUCCESS] dep-4521 — auth-service @ v2.3.1 (2026-04-21 14:30)
[SUCCESS] dep-4520 — api-gateway @ v1.8.0 (2026-04-21 12:15)
[FAILED]  dep-4519 — user-service @ v3.1.0 (2026-04-21 10:00)
[SUCCESS] dep-4518 — api-gateway @ v1.7.9 (2026-04-20 16:45)
[SUCCESS] dep-4517 — auth-service @ v2.3.0 (2026-04-20 14:20)

> What happened with dep-4519?

Claude calls get_deployment(deployment_id="dep-4519")
// Returns full deployment details including error logs

> Roll it back to v3.0.2 because the migration script had a bug

Claude calls rollback_deployment(...)
// You get a permission prompt because this tool isn't auto-allowed
```

这就是 MCP 的威力：领域特定操作感觉像自然对话，但你对允许的操作拥有完全控制权。

### 用 Python 构建

如果 TypeScript 不合你口味，Python SDK 同样强大：

```python
# server.py
from mcp.server.fastmcp import FastMCP
import httpx
import os

mcp = FastMCP("deploy-tools")

API_BASE = os.environ.get("DEPLOY_API_URL", "https://api.internal.company.com")
API_TOKEN = os.environ["DEPLOY_API_TOKEN"]

@mcp.tool()
async def list_deployments(environment: str, limit: int = 10) -> str:
    """List recent deployments with their status.

    Args:
        environment: Target environment (staging or production)
        limit: Number of results to return (1-50)
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{API_BASE}/deployments",
            params={"env": environment, "limit": limit},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        resp.raise_for_status()
        data = resp.json()

    lines = []
    for d in data["deployments"]:
        lines.append(f"[{d['status']}] {d['id']} — {d['service']} @ {d['version']}")
    return "\n".join(lines) or "No deployments found."

@mcp.tool()
async def check_health(service: str, environment: str) -> str:
    """Check the health of a service in an environment.

    Args:
        service: Service name
        environment: Target environment (staging or production)
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{API_BASE}/health/{service}",
            params={"env": environment},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        resp.raise_for_status()
        data = resp.json()

    return (
        f"Service: {data['service']}\n"
        f"Status: {data['status']}\n"
        f"Uptime: {data['uptime']}\n"
        f"Healthy: {data['healthy_instances']}/{data['total_instances']}"
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

注册它：

```bash
claude mcp add deploy-tools -- python3 /path/to/server.py
```

Python SDK 使用类型提示和文档字符串自动生成工具 schema。比 TypeScript 版本更少样板代码。

## 真实工作流示例

### 工作流 1：端到端测试 Web 应用

安装 Playwright MCP 后，你可以让 Claude 像真实用户一样测试你的应用：

```text
I just deployed a change to the login page. Go to http://localhost:3000/login,
try logging in with test@example.com / password123, and tell me if the
redirect to /dashboard works correctly.
```

Claude 会导航、填写表单、点击提交、检查结果 URL 并报告。如果出错，它可以截图错误状态并读取控制台日志。

### 工作流 2：数据库驱动的调试

使用 Postgres MCP：

```text
A user reported they can't see their orders. Their email is jane@example.com.
Check the users table for their account, then the orders table, and tell me
if there's a data issue.
```

Claude 运行查询、关联数据并给出诊断。无需切换到 SQL 客户端。

### 工作流 3：交叉引用 GitHub 和 Slack

同时使用 GitHub 和 Slack MCP：

```text
Find all GitHub issues labeled "regression" that were opened this week,
then search Slack #incidents for any related discussion, and give me a
summary of what broke and whether it's been addressed.
```

这种跨系统关联正是 MCP 大放异彩的地方。每个服务器专注一件事；Claude 负责编排。

## MCP 服务器管理命令

以下是通过 CLI 管理 MCP 服务器的完整参考：

```bash
# Add a stdio server
claude mcp add SERVER_NAME -- COMMAND [ARGS...]

# Add with environment variables
claude mcp add SERVER_NAME -e KEY=VALUE -- COMMAND [ARGS...]

# Add at a specific scope
claude mcp add SERVER_NAME -s user -- COMMAND [ARGS...]    # global
claude mcp add SERVER_NAME -s project -- COMMAND [ARGS...]  # project (default)

# List all registered servers
claude mcp list

# Get details about a specific server
claude mcp get SERVER_NAME

# Remove a server
claude mcp remove SERVER_NAME
```

## 接下来构建什么

如果你已经走到这一步，说明你已经安装并配置了 MCP 服务器，并理解了协议。自然的进阶路径：

1. **从预构建服务器开始。** 从上面列表中安装 2-3 个。
2. **熟悉权限机制。** 在一周使用中逐步构建你的白名单。
3. **为团队内部工具构建自定义服务器。**
4. **分享它。** 将服务器配置提交到 repo 的 `.claude/settings.json` 中，让团队自动获得。

协议足够小，你可以在 30 分钟内读完规范。针对专注用例，构建自己的服务器只需半天。难点在于选择暴露哪些工具；协议本身不会碍事。

下一篇：hooks —— 在每次工具调用前后运行的代码。防御层。