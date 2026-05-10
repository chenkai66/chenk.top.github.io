---
title: "Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything"
date: 2026-04-21 09:00:00
tags:
  - claude-code
  - mcp
  - playwright
  - integration
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 4
description: "MCP is the plug-in protocol that lets Claude Code reach beyond your filesystem. Install one (Playwright), see it work end-to-end, and learn the permission model that keeps it from going feral."
disableNunjucks: true
translationKey: "claude-code-learn-4"
---

If you only learn one extension mechanism in Claude Code, learn MCP. It is the difference between an autocomplete and a platform.

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_1.png)

## The 60-second pitch

MCP is **Model Context Protocol** — a small open spec that lets Claude Code call tools and read resources from external servers. The "server" is any process that speaks MCP over stdio or HTTP. Claude treats MCP tools the same as its built-in tools: the model decides when to call them, you confirm or auto-approve, results come back as text.

This means Claude can:

- Drive a browser (Playwright MCP)
- Query a Postgres database (Postgres MCP)
- Read your Notion or Obsidian vault
- Talk to your internal API
- Anything you wrap in a 200-line Node script

MCP servers are mostly community-built. There's a catalog at `mcp.so` and a growing one inside Anthropic's docs. The good ones are short, focused, and honest about what they do.

## Install your first MCP server

Playwright is the canonical "first MCP" because it makes the value obvious. From your shell — *not* inside Claude Code:

![MCP Server Communication Flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

The MCP server is its own process. Claude Code talks to it over stdio (or HTTP, for remote servers). Tools are discovered at handshake — Claude asks "what can you do?" and the server replies with a list of tool schemas. From then on, the model can call those tools as if they were built-in.

The protocol is small enough that you can read the spec in 30 minutes. Building your own is a half-day project for a focused use case.

## Permissions — the part you must not skip

![Claude Code Hands-On (4): MCP Servers, or How Claude Talks to Anything — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/illustration_2.png)

Every MCP tool has a permission level. The first time the model calls one, you get a confirmation dialog. You have three options:

- **Allow once** — for this call only
- **Allow for this session** — until you `/clear` or quit
- **Always allow** — written into `.claude/settings.json`

I do not recommend "always allow" for any MCP that mutates state. Read-only tools (search, fetch, query) — sure. Anything that writes files, sends messages, or hits an API with side effects — confirm every time, or scope the permission narrowly.

The right pattern in `.claude/settings.json`:

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

This auto-allows the safe Playwright tools and outright blocks the one that runs arbitrary JS in the browser. Per-tool granularity.

## Useful MCP servers I actually use

After trying ~30, the ones that earned a permanent spot:

| Server | What it does | Why it's worth it |
|--------|--------------|-------------------|
| `playwright` | Browser automation | Replaces a dozen hand-rolled scrapers |
| `postgres` | Read-only Postgres queries | "Show me the rows" without leaving Claude |
| `filesystem` | Sandboxed file access outside the project root | Read configs in `~/.config/` etc. |
| `slack` | Search Slack history, post to channels | Standup summaries write themselves |
| `github` | Issue and PR access via GitHub API | Cleaner than CLI for some queries |

The pattern: each does one thing well, ships as a single npm or PyPI package, and exposes 3-10 tools.

## What MCP isn't

A few things to be clear about:

- **Not a sandbox.** An MCP server runs with your user permissions. A malicious server can do whatever you can do. Vet what you install.
- **Not a fast path.** Each MCP tool call is a process roundtrip. Latency is real. Don't use MCP for things you'd put in a hot loop.
- **Not the only way.** For one-off integrations, a slash command that runs a shell script is sometimes the right answer. MCP is for things you'll call repeatedly across sessions.

## Building one (sketch)

If you want to build your own, the SDK is `@modelcontextprotocol/sdk` for Node or `mcp` for Python. The minimal server is ~50 lines:

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

That's a working MCP server. Save it, register it (`claude mcp add my-echo node my-server.js`), restart, and Claude has a new tool. The hard part is choosing what tools to expose; the protocol gets out of the way.

Next piece: hooks — code that runs before and after every tool call. The defensive layer.
