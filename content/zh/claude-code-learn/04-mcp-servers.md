---
title: "Claude Code 实战入门（四）：MCP 服务器，让 Claude 跟任何东西对话"
date: 2026-04-19 09:00:00
tags:
  - claude-code
  - mcp
  - playwright
  - 集成
categories: Claude Code
lang: zh-CN
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 4
description: "MCP 是让 Claude Code 伸出文件系统的插件协议。装一个（Playwright），端到端看它跑通，再学会保护它不发疯的权限模型。"
disableNunjucks: true
translationKey: "claude-code-learn-4"
---
如果只掌握一个 Claude Code 的扩展机制，那就学 MCP。它让“自动补全”和“平台”有了本质区别。
![Claude Code 实战入门（四）：MCP 服务器，让 Claude 跟任何东西对话 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_1.jpg)

## 一分钟简介

MCP 是 **Model Context Protocol**，一个简洁的开放规范，允许 Claude Code 调用外部服务器上的工具并读取资源。所谓“服务器”，就是通过 stdio 或 HTTP 支持 MCP 的任何进程。Claude 对待 MCP 工具和内置工具一视同仁：模型决定何时调用，我来确认或设置自动批准，结果以文本形式返回。

有了 MCP，Claude 可以做到：

- 控制浏览器（Playwright MCP）
- 查询 Postgres 数据库（Postgres MCP）
- 读取 Notion 或 Obsidian 笔记库
- 调用内部 API
- 任何能用 200 行 Node.js 脚本封装的功能

MCP 服务器大多由社区开发。`mcp.so` 提供了一个目录，Anthropic 官方文档中也有一个不断扩充的列表。优秀的服务器通常代码简短、功能专注，并且清楚说明自己的能力范围。
## 安装你的第一个 MCP 服务器

Playwright 是最典型的“第一个 MCP”，因为它直接展示了价值。在终端里运行以下命令——*不要*在 Claude Code 内部执行：

```bash
claude mcp add playwright npx @playwright/mcp@latest
```

这条命令完成了三件事：

1. 注册了一个名为 `playwright` 的 MCP 服务器。
2. 运行这个服务器的命令是 `npx @playwright/mcp@latest`，npx 会按需下载并执行。
3. 服务器配置已经写入了你的全局配置文件（`~/.claude/settings.json`）。

验证一下：

```bash
claude mcp list
# playwright    npx @playwright/mcp@latest    enabled
```

打开 Claude Code：

```bash
claude
> 能帮我打开 https://news.ycombinator.com 并告诉我前 3 条新闻标题吗？
```

第一次运行时，你会看到一个权限提示：“Claude 想使用 playwright 工具。” 点击批准。Agent 会启动一个无头浏览器，加载页面，提取标题，并返回结果。整个过程只需要几秒钟。
## 刚才发生了什么（技术上）

```
Claude Code  --(stdio)--> @playwright/mcp 进程
       |                        |
       |                        v
       |                   真实浏览器
       |                        |
       <----- text -------------+
```

MCP 服务器是一个独立的进程。Claude Code 通过 stdio 和它通信，远程服务器则用 HTTP。握手时会发现工具——Claude 问“你能做什么？”，服务器返回一个工具 schema 列表。之后，模型就能像调用内置工具一样使用这些工具。

协议很简单，30 分钟就能读完规范。如果专注于某个具体场景，自己实现一个半天就够了。
## 权限——这一部分绝不能跳过

![Claude Code 实战入门（四）：MCP 服务器，让 Claude 跟任何东西对话 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/04-mcp-servers/illustration_2.jpg)

每个 MCP 工具都有对应的权限级别。第一次调用某个工具时，模型会弹出确认对话框，这时你有三个选择：

- **仅本次允许**——只对当前调用有效
- **允许本次会话**——直到执行 `/clear` 或退出为止
- **始终允许**——写入 `.claude/settings.json`

我不建议对任何会修改状态的 MCP 工具选择“始终允许”。如果是只读工具（比如搜索、抓取、查询），那没问题。但凡是涉及写文件、发消息或者调用带副作用的 API 的工具，每次都确认一下，或者把权限范围限制得非常窄。

`.claude/settings.json` 文件中推荐的配置方式如下：

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

这个配置直接放行了 Playwright 中安全的工具，同时明确禁止了在浏览器中运行任意 JavaScript 的功能。权限控制可以细化到具体工具。
## 我真正在用的几个 MCP 服务器

试了大概 30 个，最后留下这几个：

| 服务器 | 功能 | 留下的理由 |
|--------|------|------------|
| `playwright` | 浏览器自动化 | 替代了一堆手写的爬虫 |
| `postgres` | 只读 Postgres 查询 | 不用离开 Claude 就能查数据行 |
| `filesystem` | 项目根目录外的沙盒文件访问 | 能读取 `~/.config/` 里的配置文件 |
| `slack` | 搜索 Slack 历史记录、发送消息 | 自动生成站会总结 |
| `github` | 通过 GitHub API 访问 Issue 和 PR | 某些查询比命令行更简洁 |

它们的共同点：每个只专注做好一件事，打包成一个 npm 或 PyPI 包，提供 3 到 10 个工具。
## MCP 不是什么

有几点需要明确：

- **不是沙箱。** MCP 服务器用的是我的用户权限。恶意服务器能做的事，跟我自己能做的完全一样。所以安装之前一定要仔细审查。
- **不是加速器。** 每次调用 MCP 工具都会涉及进程的往返通信。延迟确实存在。别把它用在高频循环里。
- **不是唯一选择。** 如果只是一次性集成，有时直接用一个运行 shell 脚本的斜杠命令可能更合适。MCP 更适合跨会话反复调用的场景。
## 自己动手写一个（草稿）

想自己动手的话，Node 用 `@modelcontextprotocol/sdk`，Python 用 `mcp`。最简服务器大概 50 行：

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({ name: 'my-tools', version: '0.1' }, {
  capabilities: { tools: {} }
});

server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'echo',
    description: '回显输入内容',
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

这就是一个能用的 MCP 服务器。保存代码，注册工具（`claude mcp add my-echo node my-server.js`），重启后，Claude 就多了一个新工具。难点在于决定暴露哪些功能；协议本身不会挡路。

下一篇：Hooks——每次调用工具前后执行的代码。这是防御层的关键。
