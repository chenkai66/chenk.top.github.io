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
如果在 Claude Code 里只学一种扩展机制，那必须是 MCP——它是从“自动补全”迈向“平台级能力”的关键分界。

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_1.png)

## 60 秒说清楚

MCP（Model Context Protocol）是一个轻量级开放协议，让 Claude Code 能通过 stdio 或 HTTP 通信调用任意支持该协议的外部进程（即 MCP 服务器）。对 Claude 而言，这些工具与内置能力完全等价：调用时机由模型自主判断，用户可以手动确认、允许本次会话有效，或永久授权；返回结果统一为纯文本。

这意味着 Claude 能驱动浏览器（Playwright MCP）、查询 Postgres 数据库（Postgres MCP）、读取 Notion 或 Obsidian 仓库、对接内部 API，甚至调用任何你能用 200 行 Node 脚本封装的能力。

MCP 服务器大多是社区建的。目录在 `mcp.so`， Anthropic 文档里也在不断增加。优秀的 MCP 服务器通常轻量简洁，功能边界清晰。

## 理解 MCP 协议

在安装任何内容之前，先了解其底层实际发生的情况会很有帮助。MCP 是一种 JSON-RPC 2.0 协议，具有特定的生命周期：

**1. 初始化（Initialization）。** Claude Code 启动 MCP 服务器进程（或连接到远程服务器）。双方通过 `initialize` 握手交换能力声明：服务器声明自身支持的功能（例如工具、资源、提示模板等），客户端则确认希望启用哪些功能。

**2. 发现（Discovery）。** Claude Code 调用 `tools/list` 获取全部可用工具的目录。每个工具包含名称、描述及其输入参数的 JSON Schema。模型借此明确可调用的工具及所需传入的参数结构。

**3. 调用（Invocation）。** 在对话过程中，当模型决定调用某个 MCP 工具时， Claude Code 发送 `tools/call` 请求，附带工具名称与参数。服务器执行相应逻辑，并返回结果——可以是文本、图像或结构化数据。

**4. 关闭（Shutdown）。** 当 Claude Code 退出或用户移除服务器时，发送关闭通知并终止该进程。

整个协议规范出人意料地简洁，约 30 分钟即可通读完毕，详见 `spec.modelcontextprotocol.io`。关键洞见在于： MCP 在各次调用之间是无状态的——每个 `tools/call` 请求彼此独立；服务器无需维护对话上下文，该职责由 Claude Code 承担。

![MCP 服务器生命周期：进程启动、initialize 握手、tools/list 发现、tools/call 调用循环、shutdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_lifecycle_sequence.png)

*一次 MCP 会话的四个阶段：初始化（INIT）和发现（DISCOVER）启动时各执行一次；对话过程中模型反复调用（INVOKE）；Claude Code 退出或你移除服务器时触发 SHUTDOWN。*

### Stdio 与 SSE 传输机制

MCP 支持两种传输机制，理解其差异对服务器的配置与调试至关重要。

**Stdio 传输**是默认且最常用的机制。 Claude Code 将 MCP 服务器作为子进程启动，并通过 stdin/stdout 进行通信。每个 JSON-RPC 消息以单行形式写入 stdout， Claude Code 则将请求写入该进程的 stdin。

Stdio 的优势：
- 无需任何网络配置
- 不会出现端口冲突
- 进程生命周期由系统自动管理
- 可在防火墙和 VPN 后正常工作
- 调试最简单——可直接读取管道数据

**SSE （Server-Sent Events）传输**适用于远程服务器。 MCP 服务器以 HTTP 服务形式运行， Claude Code 通过 Server-Sent Events 接收服务端到客户端的消息，并通过 HTTP POST 发送客户端到服务端的消息。

SSE 的优势：
- 服务器可运行于不同机器上
- 多个客户端可共享同一服务器实例
- 服务器可在 Claude Code 会话间持续运行
- 可部署于带身份认证的反向代理之后

实践中，你所使用的 MCP 服务器中约 90% 采用 stdio；SSE 则适用于需要共享服务的场景，例如企业级数据库代理、集中式工具服务器等。

```
# Stdio：Claude Code 启动子进程
Claude Code  ──stdin/stdout──>  MCP Server 进程（本地）

# SSE：Claude Code 通过 HTTP 连接
Claude Code  ──HTTP POST──>  MCP Server（远程）
             <──SSE──────
```

![Stdio 与 SSE 两种 MCP 传输方式的并排对比，含优势与适用场景](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_stdio_vs_sse.png)

*Stdio 是本地默认方式——单客户端、单子进程、零网络配置。SSE 则相反：一个长期运行的服务，多个客户端共享，但你需要自己负责部署和运维。*

## 装你的第一个 MCP 服务器

Playwright 是标准的“入门首选”，因为效果直观可见。注意，要在 Shell 里跑，*别*在 Claude Code 里面装：

![MCP Server Communication Flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

MCP 服务器是独立进程。 Claude Code 通过 stdio （远程场景下使用 HTTP）与其通信。在初始化握手阶段， Claude 会询问“你能提供哪些工具？”，服务器则返回一组工具 Schema。之后模型就能像调用内置工具一样调用它们。

### 分步指南： Playwright MCP

Playwright MCP 为 Claude 提供了一个真实浏览器——不是无头浏览器，而是一个你可以亲眼目睹其点击与输入操作的真实 Chromium 实例。当你首次看到 Claude 驱动浏览器执行你编写的流程测试时， MCP 的价值便一目了然。

安装只需一条命令：

```bash
claude mcp add playwright -- npx @anthropic-ai/mcp-playwright
```

重启 Claude Code，向其发出打开某个 URL 的指令，并确认权限提示。此后， Claude 即拥有了一组稳定可用的浏览器工具——包括导航（navigate）、点击（click）、输入（type）、截图（snapshot）、执行 JavaScript 表达式（evaluate）——并可将这些工具串联起来，构建多步骤交互。适用于端到端冒烟测试、网页抓取，或“在我最近一次部署后，检查该页面是否确实正常运行”。

## 权限设置——这一步绝对不能省

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_2.png)

每个 MCP 工具都有权限级别。模型第一次调用时，会弹确认框。你有三个选项：

- **Allow once** — 仅这次有效
- **Allow for this session** — 直到你 `/clear` 或退出
- **Always allow** — 写入 `.claude/settings.json`

建议：对任何可能修改系统状态的 MCP 工具，切勿选择“Always allow”。仅执行读操作的工具（如搜索、获取、查询）风险较低，可酌情放宽权限。凡涉及写文件、发送消息或调用有副作用的 API，均需每次确认，或进一步收窄权限范围。

在 `.claude/settings.json` 里的正确写法：

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

这样能自动放行安全的 Playwright 工具，直接封杀那个能在浏览器跑任意 JS 的工具。粒度精确到每个工具。

![MCP 工具权限决策树：先查 deny、再查 allow，未命中则弹出三选一确认框](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_permission_flow.png)

*每次 MCP 工具调用都走这棵树。 deny 列表最先短路； allow 列表内的工具自动执行；介于两者之间的，全部走三选一弹窗。*

### 权限命名规范

MCP 工具权限遵循 `mcp__SERVERNAME__TOOLNAME` 模式。双下划线是刻意为之。你也可以使用正则表达式模式：

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

若你想允许某个服务器（例如只读服务器）的所有工具，可采用更宽松的策略。但我更倾向显式的白名单。

### 常见权限陷阱

**陷阱 1：`npx` 在首次运行时下载依赖。** MCP 服务器首次运行时，`npx` 可能会下载对应包。此过程需数秒，看起来像服务器卡住。这并非故障——请耐心等待。

**陷阱 2：权限提示疲劳。** 若每次调用 MCP 都弹出权限提示，则说明配置有误。请将安全的、只读类工具加入 `allow` 列表；仅对写操作类工具保留手动确认。

**陷阱 3：环境变量未解析。** 若服务器依赖环境变量，而该变量未在你的 shell 中设置，服务器将静默失败或报出晦涩错误。在归咎于 MCP 服务器前，请务必先用 `echo $VAR_NAME` 验证变量值。

**陷阱 4：服务器名称冲突。** 若两个配置文件（例如项目级与全局级）定义了相同的服务器名称，则更具体的配置优先生效。若你预期全局配置生效，此行为可能令人困惑。

![三层配置作用域：从 GLOBAL 到 PROJECT，越具体优先级越高](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_config_scopes.png)

*作用域按从最具体到最宽泛的顺序解析。需要随仓库共享的服务器放 project，自己的私人 token 放 user-project，跨项目通用的工具放 global。*

## 我自己真在用的 MCP 服务器

试过大概 30 个，最后常驻就这几个：

| Server | What it does | Why it's worth it |
|--------|--------------|-------------------|
| `playwright` | 浏览器自动化 | 顶替十几个手写的爬虫脚本 |
| `postgres` | 只读 Postgres 查询 | 不用切窗口就能“把数据查出来给我看” |
| `filesystem` | 项目根目录外的沙盒文件访问 | 读取 `~/.config/` 里的配置等 |
| `slack` | 搜 Slack 历史、发频道消息 | 站会总结自动生成 |
| `github` | 通过 GitHub API 访问 Issue 和 PR | 某些查询比 CLI 更干净 |

设计原则清晰：每个服务器专注单一职责，以独立的 npm 或 PyPI 包形式发布，并提供 3–10 个工具接口。

### 我评估后弃用的服务器

并非所有 MCP 服务器都值得保留。以下是我尝试过并最终移除的服务器及其原因：

| 服务器 | 弃用原因 |
|--------|----------|
| Memory 服务器 | 概念有趣，但模型内置的上下文通常已足够 |
| Brave Search | 频繁触发速率限制；内置 `WebSearch` 工具通常已够用 |
| SQLite | Postgres 服务器已满足我的需求；无需额外的数据库工具 |
| Docker | 权限过大而缺乏必要约束；我更倾向于直接编写 Docker 命令脚本 |

## 调试 MCP 连接问题

问题总会发生。以下是系统化调试 MCP 服务器的方法。

### 步骤 1：检查服务器是否已注册

```bash
claude mcp list
```

你应在列表中看到你的服务器及其命令与参数。若未出现，请重新添加。

### 步骤 2：独立测试服务器

直接在终端中运行服务器命令：

```bash
# 对于 stdio 服务器，直接运行并观察是否启动成功
npx @anthropic-ai/mcp-playwright

# 它应保持空闲状态，等待 stdin 输入
# 若立即崩溃，则存在依赖项问题
```

### 步骤 3：检查错误输出

当服务器在 Claude Code 内部启动失败时，错误通常会出现在 Claude Code 自身的输出中。请查找如下格式的行：

```
MCP server 'playwright' failed to start: Error: spawn npx ENOENT
```

常见原因：

| 错误 | 原因 | 修复方法 |
|------|------|----------|
| `ENOENT` | 未找到 `npx` | 确保 Node.js 已加入 PATH |
| `ECONNREFUSED` | SSE 服务器未运行 | 先启动远程服务器 |
| `timeout` | 服务器初始化耗时过长 | 检查 `npx` 是否正在下载包 |
| `unexpected token` | 版本不匹配 | 清除 npx 缓存：`npx clear-npx-cache` |

### 步骤 4：启用详细日志

```bash
# 以调试输出模式运行 Claude Code
CLAUDE_DEBUG=1 claude
```

这将显示 Claude Code 与 MCP 服务器之间原始的 JSON-RPC 消息，可精准定位通信中断位置。

### 步骤 5：检查环境变量

```bash
# 确保所有必需的环境变量均已设置
env | grep -i github
env | grep -i slack
env | grep -i database
```

当必需的环境变量缺失时， MCP 服务器会静默失败：服务器仍可启动，握手亦能成功，但首次工具调用将因认证错误而失败。

## MCP 不是什么

有几件事得说清楚：

- **不是沙盒。** MCP 服务器用你的用户权限跑。恶意服务器能干嘛你就能干嘛。安装前得 vet。
- **不是快车道。** 每次调用都要进程往返。延迟是实打实的。别把 MCP 用在热循环里。
- **不是唯一方案。** 一次性集成，有时候跑个 Shell 脚本的 slash command 更合适。 MCP 适合跨会话反复调用的场景。
- **不是流式接口。** MCP 工具返回的是完整结果，而不是数据流。如果你需要实时数据，得另寻方案。

## 构建自定义 MCP 服务器——完整示例

如需自行构建， Node 环境下 SDK 为 `@modelcontextprotocol/sdk`， Python 环境下为 `mcp`。下面以一个真实示例展开说明：该服务器通过查询 REST API 并将其封装为工具进行暴露。

### 场景说明

你拥有一个内部 API，地址为 `https://api.internal.company.com`，用于管理部署任务。你希望 Claude 能够列出部署项、检查其状态并触发回滚操作。与其教导 Claude 使用 `curl`，不如构建一个 MCP 服务器。

### 项目初始化

```bash
mkdir mcp-deploy-server
cd mcp-deploy-server
npm init -y
npm install @modelcontextprotocol/sdk zod
```

### 完整的服务器代码

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

四个工具对应了四种典型操作：列查询、详情读取、写入回滚、健康检查。所有副作用由 `apiCall` 这个轻量包装器统一收口，方便后续加重试、熔断或日志。

### 在 Claude Code 中注册该工具

```bash
claude mcp add deploy-tools -- npx tsx /path/to/mcp-deploy-server/server.ts
```

或配合环境变量使用：

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

注意：我未将 `rollback_deployment` 加入 `allow` 列表。该操作会变更生产环境状态，因此始终要求手动确认。

![自定义 deploy-tools MCP 服务器结构：四个工具，三个只读自动放行，一个写操作手动确认，附 settings.json 片段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_custom_server_anatomy.png)

*自己写服务器的核心价值：把领域操作变成一等公民工具，把权限边界写进 `settings.json`，而不是寄希望于模型自觉。*

### 在实践中使用它

现在，在 Claude Code 中，你可以进行如下对话：

```
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

这正是 MCP 的强大之处：领域专属操作如同自然对话般流畅，同时又能对可执行操作实施完整控制。

### 使用 Python 构建

若你偏好 Python 而非 TypeScript， Python SDK 同样功能完备：

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

注册方式如下：

```bash
claude mcp add deploy-tools -- python3 /path/to/server.py
```

Python SDK 利用类型提示（type hints）与文档字符串（docstrings）自动生成工具 schema，相比 TypeScript 版本更少样板代码。

## 真实工作流示例

### 工作流 1：端到端测试 Web 应用

安装 Playwright MCP 后，你可以让 Claude 像真实用户一样测试你的应用：

```
我刚刚部署了登录页的变更。请访问 http://localhost:3000/login，
使用 test@example.com / password123 尝试登录，并告诉我重定向至 /dashboard 是否正常。
```

Claude 将执行页面导航、填写表单、点击提交按钮、校验最终 URL，并反馈结果。若出现异常，它可截取错误状态的屏幕截图并读取控制台日志。

### 工作流 2：基于数据库的调试

启用 Postgres MCP 后：

```
有用户报告无法查看自己的订单，其邮箱为 jane@example.com。
请查询 users 表确认该账户是否存在，再查询 orders 表，并告知是否存在数据问题。
```

Claude 将自动执行 SQL 查询、关联多表数据，并给出诊断结论——无需手动切换至 SQL 客户端。

### 工作流 3：跨 GitHub 与 Slack 关联分析

同时启用 GitHub 和 Slack MCP：

```
查找本周内所有标记为 "regression" 的 GitHub Issue，
再在 Slack #incidents 频道中搜索相关讨论，汇总说明发生了什么故障、是否已修复。
```

此类跨系统关联分析正是 MCP 的核心优势所在：每个 MCP 服务专注单一职责， Claude 充当编排中枢。

## MCP 服务管理命令

以下为通过 CLI 管理 MCP 服务的完整参考：

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

若你已读至此处，说明你已完成 MCP 服务的安装、配置，并理解了协议机制。自然的演进路径如下：

1. **从预构建服务入手。** 从上方列表中安装 2–3 个现成服务。
2. **熟悉权限模型。** 在一周的实际使用中逐步构建你的白名单。
3. **构建自定义服务。** 为你团队的内部工具开发专属 MCP 服务。
4. **共享服务配置。** 将服务配置提交至代码仓库的 `.claude/settings.json`，使团队成员自动获得该服务。

该协议足够精简，你可在 30 分钟内通读规范；针对明确场景实现一个自定义服务，仅需半天专注投入。真正的挑战在于决定开放哪些工具能力；而协议本身会主动退居幕后，不构成障碍。

下一篇讲 hooks——在每个工具调用前后运行的代码。这是防御层。
