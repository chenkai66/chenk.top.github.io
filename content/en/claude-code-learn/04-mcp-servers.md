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

## Understanding the MCP protocol

Before installing anything, it helps to understand what's actually happening under the hood. MCP is a JSON-RPC 2.0 protocol with a specific lifecycle:

**1. Initialization.** Claude Code spawns the MCP server process (or connects to a remote one). The two sides exchange capabilities in an `initialize` handshake. The server declares what it supports — tools, resources, prompts — and the client confirms what it wants to use.

**2. Discovery.** Claude Code calls `tools/list` to get the full catalog of available tools. Each tool comes with a name, description, and a JSON Schema for its input. This is how the model knows what it can call and what arguments to pass.

**3. Invocation.** During a conversation, when the model decides to use an MCP tool, Claude Code sends a `tools/call` request with the tool name and arguments. The server executes the logic and returns a result — text, images, or structured data.

**4. Shutdown.** When Claude Code exits or you remove the server, it sends a shutdown notification and terminates the process.

The entire protocol spec is surprisingly short. You can read it at `spec.modelcontextprotocol.io` in about 30 minutes. The key insight is that MCP is stateless between calls — each `tools/call` is independent. The server doesn't need to track conversation context; Claude Code handles that.

### Stdio vs SSE transport

MCP supports two transport mechanisms, and understanding the difference matters for how you configure and debug servers:

**Stdio transport** is the default and most common. Claude Code spawns the MCP server as a child process and communicates over stdin/stdout. Each JSON-RPC message is written as a line to stdout; Claude Code writes requests to the process's stdin.

Advantages of stdio:
- Zero network configuration
- No port conflicts
- Process lifecycle managed automatically
- Works behind firewalls and VPNs
- Simplest to debug — you can just read the pipe

**SSE (Server-Sent Events) transport** is for remote servers. The MCP server runs as an HTTP service, and Claude Code connects to it via Server-Sent Events for server-to-client messages and HTTP POST for client-to-server messages.

Advantages of SSE:
- Server can run on a different machine
- Multiple clients can share one server instance
- Server persists across Claude Code sessions
- Can be deployed behind a reverse proxy with auth

In practice, 90% of the MCP servers you'll use are stdio. SSE is for when you need a shared service — a company-wide database proxy, a centralized tool server, etc.

```
# Stdio: Claude Code spawns the process
Claude Code  ──stdin/stdout──>  MCP Server Process (local)

# SSE: Claude Code connects over HTTP
Claude Code  ──HTTP POST──>  MCP Server (remote)
             <──SSE──────
```

## Install your first MCP server

Playwright is the canonical "first MCP" because it makes the value obvious. From your shell — *not* inside Claude Code:

![MCP Server Communication Flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/04-mcp-servers/fig_mcp_arch_en.png)

The MCP server is its own process. Claude Code talks to it over stdio (or HTTP, for remote servers). Tools are discovered at handshake — Claude asks "what can you do?" and the server replies with a list of tool schemas. From then on, the model can call those tools as if they were built-in.

### Step-by-step: Playwright MCP

Playwright MCP gives Claude a real browser. Not a headless scraper — a full Chromium instance it can navigate, click, screenshot, and inspect.

```bash
# Add the server to your project
claude mcp add playwright -- npx @anthropic-ai/mcp-playwright

# Verify it's registered
claude mcp list
```

After adding, restart Claude Code (or start a new session). You should see the Playwright tools in the tool list. Try a simple prompt:

```
Navigate to https://news.ycombinator.com and tell me the top 3 stories
```

Claude will call `browser_navigate`, then `browser_snapshot` to read the page content, and summarize the results. The first time each tool is called, you'll get a permission prompt.

Here's what the actual tool calls look like in a session:

```
Claude wants to use mcp tool: mcp__playwright__browser_navigate
  url: "https://news.ycombinator.com"
> Allow? (y/n/always)

Claude wants to use mcp tool: mcp__playwright__browser_snapshot
> Allow? (y/n/always)
```

### Step-by-step: Filesystem MCP

The filesystem MCP server gives Claude controlled access to directories outside your project root. By default, Claude Code can only read and write files within the current working directory. The filesystem MCP extends that — with guardrails.

```bash
# Allow Claude to read your SSH config and dotfiles (read-only)
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem \
  /Users/you/.config \
  /Users/you/.ssh/config

# The paths you pass are the allowed directories.
# The server refuses access to anything outside them.
```

Now you can ask Claude things like:

```
Read my SSH config and tell me which hosts I have configured
```

```
Check my git config in ~/.config/git/config and suggest improvements
```

The filesystem server enforces a strict allowlist. Even if the model tries to read `/etc/passwd`, the server returns an error. This is sandbox-by-design.

### Step-by-step: GitHub MCP

The GitHub MCP server gives Claude direct access to the GitHub API — issues, PRs, repos, and more. It's richer than shelling out to `gh` because the model gets structured data back.

```bash
# You need a GitHub personal access token
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxx

# Add the server
claude mcp add github -- npx -y @modelcontextprotocol/server-github
```

Now Claude can:

```
List all open issues in this repo labeled "bug"
```

```
Show me the review comments on PR #42
```

```
Create a new issue titled "Fix login timeout" with the label "backend"
```

The GitHub MCP exposes tools for repositories, issues, pull requests, branches, files, and search. It's one of the most tool-rich MCP servers — around 30+ tools.

### Step-by-step: Postgres MCP

The Postgres MCP server connects Claude to a PostgreSQL database. It's designed for read-only exploration — analyzing schemas, running queries, explaining table structures.

```bash
# Add with your connection string
claude mcp add postgres -- npx -y @modelcontextprotocol/server-postgres \
  "postgresql://user:password@localhost:5432/mydb"
```

Now you can have conversations like:

```
Show me the schema of the users table
```

```
How many orders were placed in the last 7 days, grouped by status?
```

```
Find any users who signed up but never placed an order
```

The server runs queries against your database and returns results as formatted text. A few safety notes:

- Use a read-only database user. The server doesn't enforce read-only mode itself.
- Don't point it at production with a write-capable user. Use a replica.
- Connection strings with passwords end up in your settings file. Use environment variables instead:

```bash
claude mcp add postgres -- npx -y @modelcontextprotocol/server-postgres \
  "$DATABASE_URL"
```

### Step-by-step: Slack MCP

The Slack MCP server lets Claude search your Slack workspace, read channels, and post messages.

```bash
# Requires a Slack bot token with appropriate scopes
export SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx

claude mcp add slack -- npx -y @anthropic-ai/mcp-slack
```

Useful prompts:

```
Search Slack for messages about the deployment failure yesterday
```

```
Summarize the #engineering channel from the last 24 hours
```

```
Post a standup update to #team-backend summarizing what I committed today
```

This one is powerful but I keep it on per-call confirmation. A tool that can post messages to your entire company's Slack deserves careful supervision.

## Configuration in settings.json

When you run `claude mcp add`, the server gets registered in `.claude/settings.json`. Here's what that looks like for a multi-server setup:

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

A few things to notice:

- **Environment variables** are referenced with `${VAR}` syntax. Claude Code resolves them at startup. Never hardcode secrets in settings files.
- **The `args` array** is passed directly to the command. No shell interpolation happens here.
- **Server names** (like `playwright`, `github`) are how you'll reference them in permissions and in conversation.

### Configuration scopes

MCP servers can be registered at three levels:

| Scope | File | Applies to |
|-------|------|------------|
| Project | `.claude/settings.json` | Everyone who clones this repo |
| User-project | `.claude/settings.local.json` | Only you, in this repo |
| Global | `~/.claude/settings.json` | All your projects |

Project-level is for servers the whole team should use (Playwright for testing, database for exploration). User-project is for personal productivity servers (your Obsidian vault, your private APIs). Global is for servers you want everywhere (filesystem access to your dotfiles).

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

This auto-allows read-only operations and blocks anything that mutates external state. Per-tool granularity.

### Permission naming convention

MCP tool permissions follow the pattern `mcp__SERVERNAME__TOOLNAME`. The double underscore is intentional. You can also use regex patterns:

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

If you want to allow all tools from a server (for read-only servers), you can be more permissive. But I prefer explicit allowlists.

### Common permissions gotchas

**Gotcha 1: npx downloads on first run.** The first time an MCP server runs, `npx` may download the package. This takes a few seconds and looks like the server is hanging. It's not broken — just wait.

**Gotcha 2: Permission prompt fatigue.** If you're getting a permission prompt on every single MCP call, you're doing it wrong. Add the safe, read-only tools to your allow list. Keep the write tools on manual confirmation.

**Gotcha 3: Env vars not resolved.** If your server needs environment variables and they're not set in your shell, the server will fail silently or with a cryptic error. Always verify with `echo $VAR_NAME` before blaming the MCP server.

**Gotcha 4: Server name collisions.** If two settings files (e.g., project and global) define the same server name, the more specific one wins. This can be confusing if you expect the global config to apply.

## Useful MCP servers I actually use

After trying ~30, the ones that earned a permanent spot:

| Server | Package | What it does | Why it's worth it |
|--------|---------|--------------|-------------------|
| Playwright | `@anthropic-ai/mcp-playwright` | Browser automation | Replaces a dozen hand-rolled scrapers |
| Postgres | `@modelcontextprotocol/server-postgres` | Read-only Postgres queries | "Show me the rows" without leaving Claude |
| Filesystem | `@modelcontextprotocol/server-filesystem` | Sandboxed file access outside project root | Read configs in `~/.config/` etc. |
| Slack | `@anthropic-ai/mcp-slack` | Search Slack history, post to channels | Standup summaries write themselves |
| GitHub | `@modelcontextprotocol/server-github` | Issue and PR access via GitHub API | Cleaner than CLI for some queries |
| Fetch | `@anthropic-ai/mcp-fetch` | HTTP GET/POST to any URL | Quick API exploration, web fetching |

The pattern: each does one thing well, ships as a single npm or PyPI package, and exposes 3-10 tools.

### Servers I evaluated and dropped

Not every MCP server is worth keeping. Here are some I tried and why I removed them:

| Server | Why I dropped it |
|--------|-----------------|
| Memory server | Interesting concept but the model's built-in context is usually enough |
| Brave Search | Rate limits hit fast; `WebSearch` built-in tool is often sufficient |
| SQLite | Postgres server covers my needs; didn't need a second database tool |
| Docker | Too much power with too little guardrails; prefer scripting Docker commands directly |

## Debugging MCP connection issues

Things will go wrong. Here's a systematic approach to debugging MCP servers.

### Step 1: Check the server is registered

```bash
claude mcp list
```

You should see your server in the list with its command and args. If it's not there, re-add it.

### Step 2: Test the server standalone

Run the server command directly in your terminal:

```bash
# For a stdio server, just run it and see if it starts
npx @anthropic-ai/mcp-playwright

# It should sit there waiting for input on stdin
# If it crashes immediately, you have a dependency issue
```

### Step 3: Check for error output

When a server fails inside Claude Code, the error often shows up in Claude Code's own output. Look for lines like:

```
MCP server 'playwright' failed to start: Error: spawn npx ENOENT
```

Common causes:

| Error | Cause | Fix |
|-------|-------|-----|
| `ENOENT` | `npx` not found | Ensure Node.js is in your PATH |
| `ECONNREFUSED` | SSE server not running | Start the remote server first |
| `timeout` | Server took too long to initialize | Check if `npx` is downloading a package |
| `unexpected token` | Version mismatch | Clear npx cache: `npx clear-npx-cache` |

### Step 4: Enable verbose logging

```bash
# Run Claude Code with debug output
CLAUDE_DEBUG=1 claude
```

This shows the raw JSON-RPC messages between Claude Code and the MCP server. You'll see exactly where the communication breaks down.

### Step 5: Check environment variables

```bash
# Make sure all required env vars are set
env | grep -i github
env | grep -i slack
env | grep -i database
```

MCP servers fail silently when required environment variables are missing. The server starts, the handshake succeeds, but the first tool call fails with an auth error.

## What MCP isn't

A few things to be clear about:

- **Not a sandbox.** An MCP server runs with your user permissions. A malicious server can do whatever you can do. Vet what you install.
- **Not a fast path.** Each MCP tool call is a process roundtrip. Latency is real. Don't use MCP for things you'd put in a hot loop.
- **Not the only way.** For one-off integrations, a slash command that runs a shell script is sometimes the right answer. MCP is for things you'll call repeatedly across sessions.
- **Not a streaming interface.** MCP tools return complete results, not streams. If you need real-time data, you need a different approach.

## Building a custom MCP server — complete example

If you want to build your own, the SDK is `@modelcontextprotocol/sdk` for Node or `mcp` for Python. Let me walk through a real example: a server that queries a REST API and exposes it as tools.

### The scenario

You have an internal API at `https://api.internal.company.com` that manages deployments. You want Claude to be able to list deployments, check their status, and trigger rollbacks. Instead of teaching Claude to use `curl`, you build an MCP server.

### Project setup

```bash
mkdir mcp-deploy-server
cd mcp-deploy-server
npm init -y
npm install @modelcontextprotocol/sdk zod
```

### The full server

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

### Register it with Claude Code

```bash
claude mcp add deploy-tools -- npx tsx /path/to/mcp-deploy-server/server.ts
```

Or with environment variables:

```bash
claude mcp add deploy-tools \
  -e DEPLOY_API_TOKEN=$DEPLOY_API_TOKEN \
  -e DEPLOY_API_URL=https://api.internal.company.com \
  -- npx tsx /path/to/mcp-deploy-server/server.ts
```

### Set up permissions

In `.claude/settings.json`:

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

Notice I left `rollback_deployment` off the allow list. That one stays on manual confirmation because it mutates production state.

### Using it in practice

Now in Claude Code, you can have conversations like:

```
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

This is the power of MCP: domain-specific operations feel like natural conversation, but with full control over what's allowed.

### Building in Python

If TypeScript isn't your thing, the Python SDK is equally capable:

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

Register it:

```bash
claude mcp add deploy-tools -- python3 /path/to/server.py
```

The Python SDK uses type hints and docstrings to auto-generate the tool schemas. Less boilerplate than the TypeScript version.

## Real workflow examples

### Workflow 1: Testing a web app end-to-end

With Playwright MCP installed, you can ask Claude to test your app like a real user:

```
I just deployed a change to the login page. Go to http://localhost:3000/login,
try logging in with test@example.com / password123, and tell me if the
redirect to /dashboard works correctly.
```

Claude will navigate, fill in the form, click submit, check the resulting URL, and report back. If something breaks, it can screenshot the error state and read the console logs.

### Workflow 2: Database-driven debugging

With Postgres MCP:

```
A user reported they can't see their orders. Their email is jane@example.com.
Check the users table for their account, then the orders table, and tell me
if there's a data issue.
```

Claude runs the queries, correlates the data, and gives you a diagnosis. No context-switching to a SQL client.

### Workflow 3: Cross-referencing GitHub and Slack

With both GitHub and Slack MCP:

```
Find all GitHub issues labeled "regression" that were opened this week,
then search Slack #incidents for any related discussion, and give me a
summary of what broke and whether it's been addressed.
```

This kind of cross-system correlation is where MCP really shines. Each server does one thing; Claude orchestrates.

## MCP server management commands

Here's the full reference for managing MCP servers via the CLI:

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

## What to build next

If you've made it this far, you have MCP servers installed, configured, and you understand the protocol. The natural progression:

1. **Start with pre-built servers.** Install 2-3 from the list above.
2. **Get comfortable with permissions.** Build your allowlist over a week of use.
3. **Build a custom server** for your team's internal tools.
4. **Share it.** Commit the server config to your repo's `.claude/settings.json` so the team gets it automatically.

The protocol is small enough that you can read the spec in 30 minutes. Building your own is a half-day project for a focused use case. The hard part is choosing what tools to expose; the protocol gets out of the way.

Next piece: hooks — code that runs before and after every tool call. The defensive layer.
