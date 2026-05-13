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
如果在 Claude Code 里只学一种扩展机制，那必须是 MCP——它是从“自动补全”迈向“平台级能力”的关键。

![Claude Code 实战 (4)：MCP 服务器，或 Claude 如何与任何事物通信 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_1.png)

## 60 秒说清楚

MCP（Model Context Protocol）是一个轻量级开放协议，让 Claude Code 通过 stdio 或 HTTP 调用外部进程（即 MCP 服务器）。对 Claude 来说，这些工具和内置功能没有区别：模型自主决定何时调用，你可以选择手动确认、本次会话放行，或永久授权；结果统一以文本形式返回。

这意味着 Claude 能：

- 驱动浏览器（Playwright MCP）
- 查询 Postgres 数据库（Postgres MCP）
- 读取你的 Notion 或 Obsidian 仓库
- 对接内部 API
- 调用任何你能用 200 行 Node 脚本封装的能力

MCP 服务器大多由社区构建。官方目录在 `mcp.so`，Anthropic 文档中也在持续扩充。优秀的 MCP 服务器通常短小精悍，功能边界清晰，对自己的能力毫不含糊。

## 理解 MCP 协议

在安装任何东西前，先搞清楚底层发生了什么，会省去很多麻烦。MCP 本质上是一个 JSON-RPC 2.0 协议，遵循特定的生命周期：

**1. 初始化（Initialization）**。Claude Code 启动 MCP 服务器进程（或连接远程服务），双方通过 `initialize` 握手交换能力声明。服务器列出自己支持的功能——工具、资源、提示模板等，客户端则确认要启用哪些。

**2. 发现（Discovery）**。Claude Code 调用 `tools/list` 获取完整工具目录。每个工具都附带名称、描述和输入参数的 JSON Schema，这样模型才知道能调什么、该怎么传参。

**3. 调用（Invocation）**。对话中，当模型决定使用某个 MCP 工具时，Claude Code 发送 `tools/call` 请求，包含工具名和参数。服务器执行逻辑后，返回文本、图片或结构化数据。

**4. 关闭（Shutdown）**。当 Claude Code 退出或你移除服务器时，会发送关闭通知并终止进程。

整个协议规范出奇地简短，花 30 分钟就能读完（见 `spec.modelcontextprotocol.io`）。关键在于：MCP 在调用之间是无状态的——每个 `tools/call` 彼此独立，服务器无需维护对话上下文，这部分由 Claude Code 负责。

![MCP 服务器生命周期：进程启动、initialize 握手、tools/list 发现、tools/call 调用循环、shutdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_lifecycle_sequence.png)

*一次 MCP 会话的四个阶段：初始化（INIT）和发现（DISCOVER）仅在启动时各执行一次；对话中模型反复调用（INVOKE）；Claude Code 退出或移除服务器时触发 SHUTDOWN。*

### Stdio 与 SSE 传输机制

MCP 支持两种传输方式，理解差异对配置和调试至关重要：

**Stdio 传输**是默认且最常用的方式。Claude Code 将 MCP 服务器作为子进程启动，通过 stdin/stdout 通信。每个 JSON-RPC 消息以单行形式写入 stdout，Claude Code 则将请求写入该进程的 stdin。

Stdio 的优势：
- 零网络配置
- 无端口冲突
- 进程生命周期自动管理
- 防火墙和 VPN 后也能工作
- 调试最简单——直接读管道就行

**SSE（Server-Sent Events）传输**用于远程服务器。MCP 服务器以 HTTP 服务运行，Claude Code 通过 SSE 接收服务端消息，通过 HTTP POST 发送客户端请求。

SSE 的优势：
- 服务器可部署在不同机器上
- 多个客户端共享同一实例
- 服务可在 Claude Code 会话间持久运行
- 可配合反向代理和身份认证部署

实践中，你用的 MCP 服务器约 90% 都是 stdio。SSE 适用于需要共享服务的场景，比如公司级数据库代理或集中式工具服务器。

```bash
# Stdio：Claude Code 启动子进程
Claude Code  ──stdin/stdout──>  MCP Server 进程（本地）

# SSE：Claude Code 通过 HTTP 连接
Claude Code  ──HTTP POST──>  MCP Server（远程）
             <──SSE──────
```

![Stdio 与 SSE 两种 MCP 传输方式的并排对比，含优势与适用场景](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_stdio_vs_sse.png)

*Stdio 是本地默认方案——单客户端、单子进程、零网络依赖。SSE 则反转了拓扑结构：一个长期运行的服务，多个客户端接入，但部署和运维责任落在你身上。*

## 安装你的第一个 MCP 服务器

Playwright 是经典的“入门首选”，因为价值立竿见影。注意：请在 Shell 中操作——*不要*在 Claude Code 内部执行：

![MCP 服务器通信流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

MCP 服务器是独立进程。Claude Code 通过 stdio（远程场景用 HTTP）与其通信。握手阶段，Claude 会问“你能做什么？”，服务器返回工具 Schema 列表。此后，模型就能像调用内置工具一样使用它们。

### 分步指南：Playwright MCP

Playwright MCP 给 Claude 配了一个真正的浏览器——不是无头爬虫，而是完整的 Chromium 实例，能导航、点击、截图、检查页面。

```bash
claude mcp add playwright -- npx @anthropic-ai/mcp-playwright
```

各部分含义：
- `playwright` —— 你为该服务器取的名字，后续用于权限控制和对话中引用（如 `mcp__playwright__browser_navigate`）
- `--` —— 分隔符，后面是实际启动命令
- `npx @anthropic-ai/mcp-playwright` —— 通过 npx 运行 Playwright MCP 服务器

安装成功后验证：

```bash
claude mcp list
# 应显示 playwright 及其命令
```

在 Claude Code 中直接说"打开 https://example.com 看看页面内容"——如果看到浏览器启动并返回结果，就说明一切正常。

### 配置作用域

MCP 服务器可在三个层级注册：

| 作用域 | 文件 | 适用范围 |
|--------|------|----------|
| 项目级 | `.claude/settings.json` | 所有克隆该仓库的人 |
| 用户-项目级 | `.claude/settings.local.json` | 仅你在当前仓库 |
| 全局级 | `~/.claude/settings.json` | 你所有项目 |

项目级用于团队共享的服务器（如测试用的 Playwright、探索用的数据库）；用户-项目级用于个人生产力工具（如你的 Obsidian 仓库、私有 API）；全局级用于跨项目通用工具（如访问 `~/.config/` 的 filesystem）。

*作用域按从具体到宽泛的顺序解析。需随仓库共享的放项目级，个人 token 放用户-项目级，跨项目工具放全局级。*

## 权限设置——这一步绝对不能省

![Claude Code 实战 (4)：MCP 服务器，或 Claude 如何与任何事物通信 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_2.png)

每个 MCP 工具都有权限级别。模型首次调用时，会弹出确认框，提供三个选项：

- **Allow once** — 仅本次有效
- **Allow for this session** — 会话内有效（直到 `/clear` 或退出）
- **Always allow** — 永久授权，写入 `.claude/settings.json`

我建议：对任何会修改外部状态的 MCP 工具，不要选“Always allow”。只读类工具（搜索、获取、查询）可以放宽；涉及写文件、发消息或调用有副作用的 API，则务必每次确认，或精细控制权限范围。

在 `.claude/settings.json` 中的推荐写法：

```json
{
  "permissions": {
    "allow": [
      "mcp__playwright__navigate",
      "mcp__playwright__extract_text"
    ],
    "deny": [
      "mcp__playwright__execute_javascript"
    ]
  }
}
```

这样能自动放行只读操作，同时阻止任何会修改外部状态的工具，实现按工具粒度的控制。

![MCP 工具权限决策树：先查 deny、再查 allow，未命中则弹出三选一确认框](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_permission_flow.png)

*每次 MCP 工具调用都会走这棵决策树：deny 列表优先短路；allow 列表中的工具自动执行；其余情况弹出三选项确认框。*

### 权限命名规范

MCP 工具权限遵循 `mcp__SERVERNAME__TOOLNAME` 格式，双下划线是刻意设计。你也可以用正则表达式：

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

对于只读服务器，可以放宽策略允许全部工具。但我更倾向显式白名单。

### 常见权限陷阱

**陷阱 1：`npx` 首次运行时下载依赖**。第一次启动 MCP 服务器时，`npx` 可能会下载包，耗时几秒，看起来像卡住。别急，它没坏——等等就好。

**陷阱 2：权限提示疲劳**。如果每次调用都弹窗，说明配置不当。把安全的只读工具加入 allow 列表，只对写操作保留手动确认。

**陷阱 3：环境变量未解析**。如果服务器依赖环境变量但 shell 中未设置，会静默失败或报错晦涩。在怪罪 MCP 服务器前，先用 `echo $VAR_NAME` 验证变量。

**陷阱 4：服务器名称冲突**。如果项目级和全局级配置定义了同名服务器，更具体的（项目级）会覆盖全局。若你预期全局生效，这可能会让你困惑。

![三层配置作用域：从 GLOBAL 到 PROJECT，越具体优先级越高](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_config_scopes.png)

*作用域按从最具体到最宽泛的顺序解析。需要随仓库共享的服务器放 project，自己的私人 token 放 user-project，跨项目通用的工具放 global。*

## 我自己真在用的 MCP 服务器

试过约 30 个后，最终常驻的就这几个：

| Server | Package | What it does | Why it's worth it |
|--------|---------|--------------|-------------------|
| Playwright | `@anthropic-ai/mcp-playwright` | 浏览器自动化 | 顶替十几个手写的爬虫脚本 |
| Postgres | `@modelcontextprotocol/server-postgres` | 只读 Postgres 查询 | “把数据查出来给我看”不用切窗口 |
| Filesystem | `@modelcontextprotocol/server-filesystem` | 项目根目录外的沙盒文件访问 | 读取 `~/.config/` 里的配置等 |
| Slack | `@anthropic-ai/mcp-slack` | 搜 Slack 历史、发频道消息 | 站会总结自动生成 |
| GitHub | `@modelcontextprotocol/server-github` | 通过 GitHub API 访问 Issue 和 PR | 某些查询比 CLI 更干净 |
| Fetch | `@anthropic-ai/mcp-fetch` | HTTP GET/POST 到任意 URL | 快速探索 API、抓取网页 |

共同点：专注单一职责，以独立 npm 或 PyPI 包发布，暴露 3–10 个工具。

### 我评估后弃用的服务器

并非所有 MCP 服务器都值得保留。以下是我尝试后移除的：

| Server | Why I dropped it |
|--------|-----------------|
| Memory server | 概念有趣，但模型内置上下文通常已足够 |
| Brave Search | 速率限制太严；内置 `WebSearch` 工具往往够用 |
| SQLite | Postgres 服务器已覆盖需求；不需要第二个数据库工具 |
| Docker | 权限过大且缺乏约束；我更倾向直接写 Docker 脚本 |

## 调试 MCP 连接问题

问题总会发生。以下是系统化调试方法：

### 步骤 1：检查服务器是否已注册

```bash
claude mcp list
```

列表中应显示你的服务器及其命令和参数。若缺失，重新添加。

### 步骤 2：独立测试服务器

直接在终端运行服务器命令：

```bash
# 对于 stdio 服务器，直接运行并观察是否启动成功
npx @anthropic-ai/mcp-playwright

# 它应保持空闲状态，等待 stdin 输入
# 若立即崩溃，则存在依赖项问题
```

### 步骤 3：检查错误输出

服务器在 Claude Code 内失败时，错误通常出现在其输出日志中。查找类似：

```text
MCP server 'playwright' failed to start: Error: spawn npx ENOENT
```

常见原因：

| 错误 | 原因 | 修复方法 |
|------|------|----------|
| `ENOENT` | `npx` 未找到 | 确保 Node.js 在 PATH 中 |
| `ECONNREFUSED` | SSE 服务器未运行 | 先启动远程服务 |
| `timeout` | 初始化超时 | 检查 `npx` 是否在下载包 |
| `unexpected token` | 版本不匹配 | 清除 npx 缓存：`npx clear-npx-cache` |

### 步骤 4：启用详细日志

```bash
# 以调试输出模式运行 Claude Code
CLAUDE_DEBUG=1 claude
```

这会显示 Claude Code 与 MCP 服务器间的原始 JSON-RPC 消息，精准定位通信断点。

### 步骤 5：检查环境变量

```bash
# 确保所有必需的环境变量均已设置
env | grep -i github
env | grep -i slack
env | grep -i database
```

缺失环境变量会导致静默失败：服务器能启动、握手成功，但首次工具调用因认证错误失败。

## MCP 不是什么

几点必须澄清：

- **不是沙盒**。MCP 服务器以你的用户权限运行。恶意服务器能做的，你也能做。安装前务必审查。
- **不是快车道**。每次调用都有进程往返开销，延迟真实存在。别用在热循环里。
- **不是唯一方案**。一次性集成，有时用斜杠命令跑脚本更合适。MCP 适合跨会话反复调用的场景。
- **不是流式接口**。MCP 工具返回完整结果，而非数据流。需要实时数据？得另寻他法。

## 构建自定义 MCP 服务器——完整示例

想自己构建？Node 用 `@modelcontextprotocol/sdk`，Python 用 `mcp`。下面以真实场景为例：封装内部 REST API 为工具。

### 场景说明

你有一个内部 API（`https://api.internal.company.com`）管理部署。希望 Claude 能列出部署、检查状态、触发回滚。与其教它用 `curl`，不如构建 MCP 服务器。

### 项目初始化

```bash
mkdir mcp-deploy-server
cd mcp-deploy-server
npm init -y
npm install @modelcontextprotocol/sdk zod
```

### 完整服务器代码

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

或配合环境变量：

```bash
claude mcp add deploy-tools \
  -e DEPLOY_API_TOKEN=$DEPLOY_API_TOKEN \
  -e DEPLOY_API_URL=https://api.internal.company.com \
  -- npx tsx /path/to/mcp-deploy-server/server.ts
```

### 配置权限

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

注意：我没把 `rollback_deployment` 加入 allow 列表。这个操作会变更生产状态，必须手动确认。

![自定义 deploy-tools MCP 服务器结构：四个工具，三个只读自动放行，一个写操作手动确认，附 settings.json 片段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_custom_server_anatomy.png)

*自建服务器的核心价值：把领域操作变成一等公民工具，权限边界写进 `settings.json`，而非依赖模型自觉。*

### 实际使用

现在，在 Claude Code 中你可以这样对话：

```toml
> 显示最近 5 次生产环境部署

Claude 调用 list_deployments(environment="production", limit=5)

[SUCCESS] dep-4521 — auth-service @ v2.3.1 (2026-04-21 14:30)
[SUCCESS] dep-4520 — api-gateway @ v1.8.0 (2026-04-21 12:15)
[FAILED]  dep-4519 — user-service @ v3.1.0 (2026-04-21 10:00)
[SUCCESS] dep-4518 — api-gateway @ v1.7.9 (2026-04-20 16:45)
[SUCCESS] dep-4517 — auth-service @ v2.3.0 (2026-04-20 14:20)

> dep-4519 发生了什么？

Claude 调用 get_deployment(deployment_id="dep-4519")
// 返回完整的部署详情（含错误日志）

> 将其回滚至 v3.0.2，因为迁移脚本存在缺陷

Claude 调用 rollback_deployment(...)
// 你会收到权限提示，因为该工具未被自动授权
```

这正是 MCP 的威力：领域操作如自然对话般流畅，同时保留对可执行动作的完全控制。

### 用 Python 构建

如果你偏好 Python，SDK 同样强大：

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
    """Check the health of a service in an environment."""
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

注册方式：

```bash
claude mcp add deploy-tools -- python3 /path/to/server.py
```

Python SDK 利用类型提示和文档字符串自动生成工具 Schema，比 TypeScript 版本更少样板代码。

## 真实工作流示例

### 工作流 1：端到端测试 Web 应用

安装 Playwright MCP 后，让 Claude 像真实用户一样测试应用：

```text
我刚刚部署了登录页的变更。请访问 http://localhost:3000/login，
使用 test@example.com / password123 尝试登录，并告诉我重定向至 /dashboard 是否正常。
```

Claude 会导航、填表单、点提交、校验 URL 并反馈结果。若出错，还能截图并读取控制台日志。

### 工作流 2：基于数据库的调试

启用 Postgres MCP：

```text
有用户报告无法查看自己的订单，其邮箱为 jane@example.com。
请查询 users 表确认该账户是否存在，再查询 orders 表，并告知是否存在数据问题。
```

Claude 执行查询、关联数据、给出诊断——无需切换到 SQL 客户端。

### 工作流 3：跨 GitHub 与 Slack 关联分析

同时启用 GitHub 和 Slack MCP：

```text
查找本周内所有标记为 "regression" 的 GitHub Issue，
再在 Slack #incidents 频道中搜索相关讨论，汇总说明发生了什么故障、是否已修复。
```

这类跨系统关联正是 MCP 的闪光点：每个服务专注一事，Claude 负责编排。

## MCP 服务管理命令

CLI 管理命令完整参考：

```bash
# 添加一个 stdio 类型服务
claude mcp add SERVER_NAME -- COMMAND [ARGS...]

# 添加时指定环境变量
claude mcp add SERVER_NAME -e KEY=VALUE -- COMMAND [ARGS...]

# 在特定作用域下添加
claude mcp add SERVER_NAME -s user -- COMMAND [ARGS...]    # 全局作用域
claude mcp add SERVER_NAME -s project -- COMMAND [ARGS...]  # 项目作用域（默认）

# 列出所有已注册的服务
claude mcp list

# 查看指定服务的详细信息
claude mcp get SERVER_NAME

# 移除某个服务
claude mcp remove SERVER_NAME
```

## 下一步构建什么

如果你读到这里，说明已掌握 MCP 的安装、配置和协议原理。自然演进路径如下：

1. **从预构建服务入手**。从上方列表安装 2–3 个。
2. **熟悉权限模型**。一周内逐步构建你的 allow 列表。
3. **构建自定义服务**。为你团队的内部工具开发专属 MCP。
4. **共享配置**。将服务器配置提交到仓库的 `.claude/settings.json`，让团队自动获得。

协议足够精简，30 分钟可读完规范；针对明确场景实现自定义服务，半天专注即可完成。真正的挑战在于决定暴露哪些工具——协议本身会主动退居幕后，不构成障碍。

下一篇讲 hooks——在每个工具调用前后运行的代码。这是你的防御层。
