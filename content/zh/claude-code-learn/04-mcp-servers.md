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
如果在 Claude Code 里只学一种扩展机制，那必须是 MCP。这是“自动补全”和“平台级能力”的区别。

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_1.png)

## 60 秒说清楚

MCP 是 **Model Context Protocol** —— 一个轻量级的开放协议，让 Claude Code 能调用外部服务器的工具和资源。这里的"Server"可以是任何通过 stdio 或 HTTP 讲 MCP 协议的进程。对 Claude 来说，MCP 工具和内置工具没区别：模型决定什么时候调，你确认或自动批准，结果以文本形式返回。

这意味着 Claude 能：

- 驱动浏览器（Playwright MCP）
- 查询 Postgres 数据库（Postgres MCP）
- 读取你的 Notion 或 Obsidian 仓库
- 对接内部 API
- 任何你能用 200 行 Node 脚本封装的东西

MCP 服务器大多是社区建的。目录在 `mcp.so`，Anthropic 文档里也在不断增加。好的服务器短小精悍，功能诚实。

## 装你的第一个 MCP 服务器

Playwright 是标准的“入门首选”，因为价值立竿见影。注意，要在 Shell 里跑，*别*在 Claude Code 里面装：

![MCP Server Communication Flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

MCP 服务器是独立进程。Claude Code 通过 stdio（远程则是 HTTP）跟它通信。握手时发现工具——Claude 问“你能干嘛？”，服务器回一堆工具 schema。之后模型就能像调用内置工具一样调用它们。

协议很简单，30 分钟能读完 spec。针对特定场景自己写一个，半天就够了。

## 权限设置——这一步绝对不能省

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_2.png)

每个 MCP 工具都有权限级别。模型第一次调用时，会弹确认框。你有三个选项：

- **Allow once** — 仅这次有效
- **Allow for this session** — 直到你 `/clear` 或退出
- **Always allow** — 写入 `.claude/settings.json`

我建议任何会修改状态的 MCP 都别选"Always allow"。只读工具（搜索、获取、查询）没问题。但凡涉及写文件、发消息、调有副作用的 API——每次确认，或者把权限 scope 缩窄。

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

## 我自己真在用的 MCP 服务器

试过大概 30 个，最后常驻就这几个：

| Server | What it does | Why it's worth it |
|--------|--------------|-------------------|
| `playwright` | 浏览器自动化 | 顶替十几个手写的爬虫脚本 |
| `postgres` | 只读 Postgres 查询 | 不用切窗口就能“把数据查出来给我看” |
| `filesystem` | 项目根目录外的沙盒文件访问 | 读取 `~/.config/` 里的配置等 |
| `slack` | 搜 Slack 历史、发频道消息 | 站会总结自动生成 |
| `github` | 通过 GitHub API 访问 Issue 和 PR | 某些查询比 CLI 更干净 |

模式很简单：每个只做好一件事，打包成单个 npm 或 PyPI 包，暴露 3-10 个工具。

## MCP 不是什么

有几件事得说清楚：

- **不是沙盒。** MCP 服务器用你的用户权限跑。恶意服务器能干嘛你就能干嘛。安装前得 vet。
- **不是快车道。** 每次调用都要进程往返。延迟是实打实的。别把 MCP 用在热循环里。
- **不是唯一方案。** 一次性集成，有时候跑个 Shell 脚本的 slash command 更合适。MCP 适合跨会话反复调用的场景。

## 自己写一个（sketch）

想自己写的话，Node 用 `@modelcontextprotocol/sdk`，Python 用 `mcp`。最小服务器大概 50 行：

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({ name: 'my-tools', version: '0.1' }, {
  capabilities: { tools: {} }
});

server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'echo',
    description: 'Echo back the input',
    inputSchema: { type: 'object', properties: { text: { type: 'string' } } }
  }]
}));

server.setRequestHandler('tools/call', async (req) => {
  if (req.params.name === 'echo') {
    return { content: [{ type: 'text', text: req.params.arguments.text }] };
  }
});

await server.connect(new StdioServerTransport());
```

这就是个能跑的 MCP 服务器。存好，注册（`claude mcp add my-echo node my-server.js`），重启，Claude 就多了一个新工具。难点在于选什么工具暴露出来；协议本身不碍事。

下一篇讲 hooks——在每个工具调用前后运行的代码。这是防御层。