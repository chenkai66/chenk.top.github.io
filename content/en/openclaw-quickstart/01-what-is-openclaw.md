---
title: "OpenClaw QuickStart (1): What This Thing Actually Is"
date: 2026-04-08 09:00:00
tags:
  - openclaw
  - ai-agent
  - self-hosted
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 1
description: "What OpenClaw is, where it sits vs ChatGPT/Claude, and which problems it solves."
disableNunjucks: true
translationKey: "openclaw-quickstart-1"
---

I keep getting asked "is OpenClaw just another wrapper around an LLM?" The short answer is no, and the reason matters enough that I wanted to write it down before walking through the QuickStart.

This is the first piece in a six-part series. By the end you should have a working OpenClaw on your own machine, talking to a model, listening on at least one chat channel, and doing something useful that survives a reboot.

![OpenClaw QuickStart (1): What This Thing Actually Is — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/01-what-is-openclaw/illustration_1.png)

## Prerequisites and assumptions

Before diving in, here is what I expect you to already have:

**Technical baseline.** You should be comfortable with a terminal, editing config files, and reading TypeScript or JavaScript stack traces. You don't need to write code for this series, but when something breaks you will read an error and know whether it is a network timeout or a syntax issue.

**Node.js environment.** OpenClaw runs on Node 18 or later. If you don't have Node installed yet, or you are on an older version, the next article will walk through installation. For now, just know that `node --version` should show at least `v18.0.0`.

**An LLM API key.** You will need at least one working API key for a language model provider. The series shows setup for three options — Alibaba Cloud's Bailian Coding Plan (free for students and verified developers), Claude direct (paid, requires Anthropic API access), and DashScope free tier (limited quota but requires no credit card). Pick whichever fits your situation. If you have none of these yet, I will show you where to sign up in article two.

**A chat platform account.** This series uses Telegram because the setup is trivial — create a bot via BotFather, copy the token, done. You don't need a Telegram account yet, but you will want one by article four. If you prefer DingTalk, WeChat, or Discord, the concepts transfer directly; I just won't hold your hand through their registration flows.

**Why these matter.** OpenClaw is not a SaaS product with a signup form. It is infrastructure you run. If the idea of SSH-ing into a VPS or editing a `.env` file makes you uncomfortable, you might be happier with a hosted solution like ChatGPT Plus or Claude Pro. No judgment — those are excellent products. But if you are here because you want the agent on your infrastructure, reading your local files, and not phoning home, then you are in the right place.

## What it is

OpenClaw is a self-hosted AI agent platform. You install a single Node binary, give it API keys for one or more LLM providers, and it runs a long-lived gateway process that listens for messages on whatever chat platforms you wire up — DingTalk, WeChat (via WorkBuddy), Telegram, Discord, Slack, Feishu, you pick. The gateway routes those messages through an agent loop with tools, skills, memory, and a cron scheduler. The whole thing is MIT-licensed, the data lives on your machine, and you can swap models without rewriting your prompts.

If you have used Claude Code or Cursor, the agent loop will feel familiar. The difference is that those products own the surface — you talk to them in their UI. OpenClaw inverts that. The surface is whatever messenger you already use. The agent comes to you.

**Architecture at 10,000 feet.** When a message arrives from any channel, the gateway deserializes it into a normalized conversation turn. That turn gets passed to the agent core, which decides whether to invoke a skill, call a tool, query memory, or just generate a reply. The agent core is model-agnostic — it speaks to LLM providers through a unified interface, so swapping from Claude to Qwen to GPT-4 is a config change, not a refactor. The response flows back through the channel adapter, which serializes it into whatever format that platform expects — Markdown for Telegram, interactive cards for DingTalk, plain text for SMS.

**The agent loop.** Inside the core, every turn goes through a four-stage cycle: Memory (load context for this user and conversation), Planning (decide what to do next based on the task and available tools), Tool execution (run filesystem ops, web searches, API calls, or MCP servers), and Reflection (evaluate whether the task is done or needs another iteration). This is not a novel architecture — it is the same loop that powers Cursor's agent mode and Claude Code's autonomous workflow. The difference is that OpenClaw exposes every stage as a hook you can customize or observe.

**Channel adapters.** Each chat platform gets its own adapter module. The adapter knows how to authenticate, maintain a connection, parse incoming events, and serialize outgoing messages. Some platforms use webhooks (Slack, Discord), some use long-polling (Telegram), some use persistent WebSocket streams (DingTalk). The adapter abstracts all of this. From the agent core's perspective, it is just sending and receiving structured JSON blobs. This separation is why adding a new channel usually takes a day or two instead of a week — you implement one interface and the rest just works.

**The skill system.** Skills are the agent's vocabulary. A skill is a Markdown file that describes a task, optional helper scripts, and a manifest that declares when the skill should be offered to the agent. At runtime, only the manifest is loaded into memory. The full skill body is paged in lazily when the agent matches the task to the skill. This keeps the system prompt small and lets you accumulate dozens of skills without blowing your context window. I will show you how to write a custom skill in article five.

## Where it sits

I find it useful to put it on a 2x2.

![Where OpenClaw sits in the agent landscape: hosted vs self-hosted, chat-app vs your-surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/01-what-is-openclaw/fig1_quadrant.png)

The bottom-right cell is small but it is the one I want. I want my agent to:

- run on a box I own
- read my files, my notes, my email
- post into the same Telegram thread I already use to nag myself
- not be metered into a subscription

OpenClaw is the most polished way to live in that cell that I have found in 2026.

## How it compares to...

**Dify.** Dify is a visual workflow builder for LLM applications. It has a beautiful web UI, drag-and-drop nodes, and built-in hosting. If you want to build a chatbot for a customer support use case and hand it off to a non-technical team, Dify is a better choice. OpenClaw has no GUI — configuration is YAML and environment variables. The tradeoff is that OpenClaw is channel-native (your agent lives in Telegram or DingTalk, not a web iframe) and it runs on a $5/month VPS instead of requiring a managed platform.

**Coze.** Coze is ByteDance's conversational AI builder. It is tightly integrated with Feishu (Lark) and has excellent support for Chinese-language models. If you are already in the ByteDance ecosystem, Coze is smoother. OpenClaw is model-agnostic and channel-agnostic — you can use Claude with DingTalk, or Qwen with Telegram, or mix three models in one agent. Coze locks you into ByteDance's model offerings and Feishu as the primary surface.

**LangChain agents.** LangChain gives you primitives (chains, agents, retrievers, memory) but you still have to wire up the server, handle chat platform authentication, and build the orchestration loop yourself. If you are a library person, LangChain is the right tool. OpenClaw is a batteries-included runtime — the server, the loop, the channel adapters, and the skill system are all there. You write config, not infrastructure.

**n8n AI nodes.** n8n is a workflow automation tool with visual nodes for LLM calls. It excels at connecting SaaS APIs — "when a Notion page is updated, summarize it and post to Slack." If your use case is stitching together five external services with an LLM in the middle, n8n is cleaner. OpenClaw is built for conversational agents that live in chat platforms and have long-running context. n8n workflows are stateless; OpenClaw agents remember what you talked about last week.

**Honest tradeoffs.** OpenClaw gives up the GUI and the managed hosting. You will edit YAML, read logs, and restart the process when you change config. In return, you get full control over your data, the ability to run on your own hardware, and the flexibility to plug in any model or channel without waiting for a SaaS vendor to add support.

## Three things it earns its keep on

![OpenClaw QuickStart (1): What This Thing Actually Is — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/01-what-is-openclaw/illustration_2.png)

I want to be honest about the things a wrapper would *not* solve and OpenClaw does.

**1. Channel adapters are real engineering.** The DingTalk Stream protocol, WeChat's lockdown, and Discord's rate limits each have their own footguns. OpenClaw ships adapters for 20-plus channels and each one knows the platform's quirks. Writing one yourself is a weekend; writing five is a month.

Here is what you avoid by using the built-in adapters. DingTalk uses a persistent WebSocket connection with a custom heartbeat protocol and expects you to ACK every message within 20 seconds or it will resend. Discord has a global rate limit of 50 requests per second across all endpoints, plus per-route limits. Telegram is the easiest — REST webhooks or long-polling — but even there you have to handle parse modes (Markdown, HTML, MarkdownV2) and the quirks of inline keyboards. WeChat Work (WorkBuddy) requires enterprise verification, callback signature validation, and a registered IP whitelist. The adapters handle all of this.

**2. Skills are not just system prompts.** A Skill is a Markdown file plus optional helper scripts plus a manifest. The agent only loads the manifest at boot — the body of the skill is paged in lazily when the skill matches the current task. This is not glamorous but it is the difference between a 4k context window of system prompt and a 40k library that costs you almost nothing per turn.

Let me make this concrete. Suppose you want the agent to search the web, summarize PDFs, manage a TODO list, send emails, and query a SQL database. If you write all of that into the system prompt, you are burning 8k-10k tokens on every turn, most of which is irrelevant to the current task. If you encode it as skills, the agent loads the 200-token manifest for each skill at boot, sees that "web-search" is relevant to the current query, and only then pages in the 1500-token skill body. The other four skills stay on disk.

**3. Memory survives across sessions.** There is a typed memory store (user profile, project state, references, feedback) plus an indexable `MEMORY.md`. When the same conversation resumes three days later in a different chat, the agent doesn't start cold.

The memory store has five sections: `user_identity` (name, role, preferences), `project_context` (active projects and their state), `references` (API docs, configuration examples), `feedback` (things the user told the agent not to do again), and `learning_profile` (depth preferences per topic). Each section is a Markdown file. The agent reads them at session start and writes back to them when the user says something that changes the context.

If none of those three matter to you, you can stop reading and use Claude Code directly. If at least one of them rings a bell, the rest of this series will pay off.

## What it costs to run

**Infrastructure.** A 2-core VPS with 2GB RAM and 20GB disk is enough for personal use. I run mine on a $6/month Vultr instance in Tokyo. If you already have a home server or NAS that can run Docker, you can host it there for free. Startup time is under five seconds and memory usage sits around 150MB idle, 400MB under load.

**LLM API costs.** Depends on usage. My average monthly bill is around $15 USD on Claude 3.5 Sonnet + Qwen-Plus for batch work. If you are on DashScope's free tier (1 million tokens per month), you can run an agent for personal use at zero cost. For team use (200 messages/day), expect $50-$200/month depending on the model.

**Time investment.** Once stable, 5-10 minutes of maintenance per day. The upfront cost is a weekend or two evenings to work through this series. The payoff is a system that knows your projects, your preferences, and your workflow, and keeps running without monthly fees or vendor lock-in.

## What you'll have at the end

By article six in this series:

- OpenClaw installed locally, gateway running on `:18789`
- A model provider configured (Bailian Coding Plan, Claude direct, or DashScope free tier — your choice)
- TUI working from your terminal
- One real chat channel wired up — I will use Telegram in this series because it has the cleanest setup
- Two custom Skills loaded
- One MCP server attached for browser automation

That is enough surface area to build any of the case studies in the official docs (second-brain, daily briefing, devops automation, content pipeline). I will point at one of those at the end.

## What this series will not cover

- Multi-agent routing — the official docs do this better
- Production hardening beyond the basics
- WeChat WorkBuddy setup beyond a brief demo (it is a Tencent product and the registration flow is non-trivial)

Next piece, we install. Should take ten minutes if your Node is current and twenty if it isn't.
