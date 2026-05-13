---
title: "OpenClaw QuickStart (2): Install and First Chat in 10 Minutes"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - installation
  - tui
  - DashScope
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 2
description: "Install OpenClaw, plug in a model provider, and have a working agent in ten minutes."
disableNunjucks: true
translationKey: "openclaw-quickstart-2"
---

The README claims five minutes, but I'd say ten. The extra time accounts for the common Node version mistake.

![OpenClaw QuickStart (2): Install and First Chat in 10 Minutes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/illustration_1.png)

## Prerequisites

- Node `v22.16` or newer. The project is strict about this. Older Node versions will install, but the gateway will throw errors with optional chaining in some places. I run `v24` because that's the recommended track.
- About 2 GB free RAM at runtime, more if you load big skills.
- An LLM API key from one of: DashScope (free tier works), Anthropic, OpenAI, or the Aliyun Bailian Coding Plan (200元/month for eight models).

Check Node first:

```bash
node -v
# v24.0.x — good
# v20.x.x — too old, see next block
```

If you are stuck on something old, install `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
source ~/.bashrc
nvm install 24
nvm use 24
```

That's the only pitfall. From here, it's smooth sailing.

### Getting your API key

**DashScope (recommended for first-timers, especially in China)**

1. Go to [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com).
2. Sign up with an Aliyun account — phone number verification is required for mainland accounts.
3. Once in the console, click "API-KEY管理" in the left sidebar.
4. Click "创建新的API-KEY". Copy it immediately; they only show it once.
5. The free tier gives you a generous quota for `qwen-plus` and `qwen-turbo`. Enough for days of experimentation before you hit any limit.

DashScope is the easiest option if you're in mainland China. Latency to their endpoints from any domestic network is under 200ms. No proxy, no VPN, no fuss.

**Anthropic** — [console.anthropic.com](https://console.anthropic.com), create an account, add a payment method, generate a key under Settings > API Keys. Minimum top-up is $5. Claude Sonnet is the sweet spot for agent use.

**OpenAI** — [platform.openai.com](https://platform.openai.com), generate a key under API keys. GPT-4o works well as the backing model.

**Network considerations**

If you are in mainland China: DashScope endpoints are domestic and fast. Anthropic and OpenAI endpoints require either a proxy, a Hong Kong VPS forwarding requests, or a SOCKS5 tunnel. OpenClaw respects `HTTPS_PROXY` and `ALL_PROXY` environment variables, so if you already have a proxy running locally you can export those before starting the gateway. Do not waste time debugging "connection timeout" errors before checking whether you can actually reach the provider's endpoint from your network.

If you are outside China: all three providers work without extra configuration. DashScope's international endpoint is available but slightly higher latency from US/Europe.

## Install OpenClaw

Two flavors — `npm` global, or the curl-bash. I prefer npm because I want to know where the binary lives:

```bash
npm install -g @anthropic-ai/openclaw@latest
openclaw --version
# 2026.3.13
```

(Yes, the npm scope is `@anthropic-ai`. The project's relationship with that org is a long story; the short version is "trademark history, harmless now".)

## Troubleshooting the install

Here are the five most common issues, in order of frequency:

**(a) npm permission errors (EACCES on global install)** — Never `sudo npm install -g`. Use `nvm` instead (installs Node into your home directory), or configure npm's prefix to a user-owned path:

```bash
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH="$HOME/.npm-global/bin:$PATH"  # add to shell profile
```

**(b) node-gyp build failures on macOS** — Some optional deps compile native addons. Fix: `xcode-select --install`, wait for the 1.2 GB download, retry.

**(c) Network timeouts in China** — Default npm registry is unreliable from mainland. Use npmmirror:

```bash
npm install -g @anthropic-ai/openclaw@latest --registry=https://registry.npmmirror.com
```

This only affects package downloads, not your LLM API calls.

**(d) `openclaw: command not found` after install** — nvm's bin directory is not in your PATH. Open a new terminal, or run `source ~/.nvm/nvm.sh`. Also check you did not install into a different Node version than the one currently active (`nvm list` shows this).

**(e) Version mismatch — global vs local** — A local `node_modules/@anthropic-ai/openclaw` takes priority over the global one when you run from that directory. Causes confusing "feature not found" errors. Delete the local copy: `rm -rf node_modules/@anthropic-ai/openclaw`. Rule: one global install, no local copies unless you are developing OpenClaw itself.

## Onboard

Run the onboarding wizard. It writes a config file into `~/.openclaw/`:

```bash
openclaw onboard
```

It will ask:

1. What to call the agent — I pick something memorable so I can scold it by name in chat. Mine is `Lobster`.
2. What it should call you — I use my actual handle, not "Boss". Helps when reading logs.
3. Which provider — pick the one whose key you have. I'll use DashScope for this walkthrough since it has a free tier.
4. The API key.
5. The default model — `qwen-plus` is the right default for general use.

The wizard writes to `~/.openclaw/openclaw.json`. You can edit by hand later.

## Start the gateway

```bash
openclaw gateway start
```

You should see something like:

```toml
[gateway] listening on http://127.0.0.1:18789
[agent] loaded skills: 17
[memory] index ready (0 entries)
[channels] none configured (yet)
```

The gateway is a long-running process. In the next pieces we attach channels and skills to it. For now it is sitting there with no input.

## TUI: talk to it from your terminal

![OpenClaw QuickStart (2): Install and First Chat in 10 Minutes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/illustration_2.png)

Open a second terminal and run:

```bash
openclaw tui
```

You get a chat-like terminal UI. Try a few things in order:

```text
Hi — introduce yourself in one sentence.

Read the file ~/.zshrc and tell me what aliases I have.

Make a directory ~/openclaw-test and create a file
notes.md inside with the words "first run" in it.
```

Three things should happen:

1. The first message returns a one-liner — model is talking.
2. The second triggers the `read` tool — the agent asks the gateway to read a file, and you see a tool-call line scroll past in the gateway log.
3. The third actually mutates your filesystem. Verify with `ls ~/openclaw-test/`.

If all three worked, the install is done. If only the first worked, the agent isn't getting access to tools — most likely the model you picked is too small to do tool-calling reliably. Switch to `qwen-plus` or `qwen3-max` and try again.

### More interactions to try

Once the basics work, push it further:

```text
Fetch https://news.ycombinator.com and tell me the top 3 stories right now.
```

This triggers the `web_fetch` tool. In the gateway log you will see a line like:

```text
[tool:web_fetch] url=https://news.ycombinator.com status=200 bytes=48231
```

That confirms the agent made an outbound HTTP request, got a response, and is now summarizing it.

```text
Run `git log --oneline -5` in ~/my-project and explain what the last five commits did.
```

This uses the `exec` tool. The gateway log shows:

```text
[tool:exec] cmd="git log --oneline -5" cwd=/Users/you/my-project exit=0
```

The agent sees the stdout and reasons over it. If the exit code is non-zero, it will tell you what went wrong.

For a multi-step task:

```text
In ~/my-project, find all files that import lodash, then tell me which lodash
functions are used and whether any of them have native ES equivalents I should
switch to.
```

Watch the gateway log — you will see multiple tool calls chained: an `exec` to grep, then several `read` calls to inspect individual files, then the final synthesis. This is the agent loop doing what it does: plan, act, observe, repeat.

### Reading the gateway log

Keep the gateway terminal visible while you experiment. Every tool invocation prints a single line with the tool name, key parameters, and result status. If something fails silently in the TUI, the gateway log is where you find out why. Common things to look for:

- `[tool:*] ... exit=1` — a shell command failed.
- `[tool:web_fetch] ... status=403` — a website blocked the request.
- `[agent] retrying with backoff` — the LLM provider returned a rate-limit or transient error.

## First things to try after install

Five tasks in increasing order of complexity. Each one exercises a different part of the system:

**1. Pure LLM — no tools**

```text
What is the difference between a coroutine and a thread? Two sentences max.
```

This round-trips through the model and back. No tools involved. If this works, your API key and provider config are correct.

**2. Read a local file**

```text
Read /etc/hosts and tell me if there are any custom entries beyond localhost.
```

The agent calls the `read` tool. You are testing that the gateway has filesystem access and the tool registry is loaded.

**3. Search the web**

```text
Search the web for "OpenClaw changelog 2026" and summarize what shipped in March.
```

This exercises `web_search`. If it fails with "tool not found", the web skill may not be enabled — check `openclaw skill list` and enable it with `openclaw skill enable web`.

**4. Create and edit a file**

```text
Create a file ~/openclaw-test/shopping.md with a grocery list: eggs, milk, bread.
Then add "butter" to the list.
```

Two tool calls: `write` then `edit`. The agent should handle both in a single turn. Verify the final file content with `cat ~/openclaw-test/shopping.md`.

**5. Multi-step reasoning — exec + read**

```text
Find all TODO comments in ~/my-project/src and list them grouped by file,
with the line number and the text of each TODO.
```

This typically produces an `exec` call (`grep -rn "TODO" src/`), then the agent formats the output. For large codebases it may paginate with multiple calls. The point is to confirm multi-step execution works end-to-end.

## What the directory structure looks like

After a successful install and first run, `~/.openclaw/` looks like this:

```text
~/.openclaw/
├── openclaw.json          # Main config: provider, model, agent name, preferences
├── workspace/
│   └── default/           # Default workspace for general tasks
│       ├── memory.json    # Conversation memory index
│       └── sessions/      # Individual session transcripts
│           └── 2026-04-04_001.json
├── skills/
│   ├── builtin/           # Ships with the install — read, write, exec, web_fetch, etc.
│   └── community/         # Installed via `openclaw skill install <name>`
├── agents/
│   └── lobster.json       # Agent personality and config (named during onboard)
└── logs/
    └── gateway.log        # Rolling log, last 7 days kept by default
```

Key points: `openclaw.json` is plain JSON with comments — edit it directly any time. `memory.json` starts empty and grows as the agent extracts facts from conversations (this is how it "remembers" across sessions). `skills/builtin/` has the tool definitions the gateway loads at startup — reading them is instructive if you plan to write custom skills later. `logs/gateway.log` persists the same output you see in the gateway terminal; rotation is configurable via `logging.retention_days` in the config.

## Web dashboard (optional, useful)

There's a web UI if you prefer:

```bash
openclaw web start
# open http://127.0.0.1:18790
```

I leave this off most of the time — TUI is faster — but it is useful for inspecting memory and skill state visually. It also shows the cron jobs once you have any.

## What just happened, architecturally

![Architecture topology: terminal -> tui -> gateway -> agent loop / skills index / tool registry -> LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/fig1_topology.png)

`tui` is just a thin client. The gateway is where the agent loop lives. That separation is what lets you later attach Telegram, DingTalk, or the web UI as alternate front-ends, all talking to the same agent.

Next piece, we open up that gateway and look at what's actually happening when you type a message.
