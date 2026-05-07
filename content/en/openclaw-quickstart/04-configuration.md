---
title: "OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick"
date: 2026-04-06 09:00:00
tags:
  - openclaw
  - configuration
  - bailian
  - coding-plan
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 4
description: "openclaw.json walkthrough, when to use which model provider, and why the Aliyun Bailian Coding Plan is the cheapest sane way to run heavy agent workloads from China — 200元/month gets you 8 models including Claude."
disableNunjucks: true
translationKey: "openclaw-quickstart-4"
---

If you only edit one file in OpenClaw, this is it.

`~/.openclaw/openclaw.json` controls models, tools, channels, memory, cron, and skill loading. The onboarding wizard wrote a default; this piece walks the parts you will actually touch.

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_1.jpg)

## The minimal viable config

Strip away the comments and the default sets, and a working config is about 25 lines:

```json
{
  "agent": {
    "name": "Lobster",
    "default_model": "qwen-plus"
  },
  "providers": {
    "dashscope": {
      "api_key": "sk-...",
      "models": ["qwen-turbo", "qwen-plus", "qwen3-max", "qwen3-coder-plus"]
    }
  },
  "tools": {
    "exec": { "trusted_patterns": ["^ls ", "^cat ", "^echo "] },
    "web_search": { "engine": "bing" }
  },
  "memory": {
    "engine": "semantic",
    "max_tokens_per_turn": 2000
  }
}
```

Five sections. Let's go through them in order.

## `agent` — name and default model

Two fields matter: the name (what the user sees) and the default model. Default model is what the agent uses for the planning loop unless a tool or skill overrides it.

The mistake people make is putting the most expensive model here, "to be safe." Don't. The planning loop runs on every turn. Put `qwen-plus` here, and override to `qwen3-max` only inside specific skills that need real reasoning.

## `providers` — where the models live

You can configure multiple providers and pick per-skill. The four useful options:

| Provider | Use when |
|----------|----------|
| `dashscope` | China-hosted, cheap, Qwen family. Default choice for me. |
| `anthropic` | You want Claude reasoning quality and don't mind paying. |
| `openai` | You need GPT-4 or GPT-5.4 specifically. |
| `bailian-coding-plan` | You want one subscription that bundles Claude + Qwen + DeepSeek + GLM. |

The Coding Plan is genuinely interesting — see below.

## The Coding Plan trick

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_2.jpg)

Aliyun Bailian has a "Coding Plan" subscription: 200元 per month gets you eight models — Claude Sonnet 4.5, Qwen3-Max, Qwen3-Coder-Plus, DeepSeek V3.1, GLM-4.6, and more — at unmetered usage during business hours. This is wildly underpriced for an agent that loops a few hundred times per day.

The provider config is one block:

```json
"bailian-coding-plan": {
  "endpoint": "https://dashscope.aliyuncs.com/coding-plan/v1",
  "api_key": "sk-..."  // your Coding Plan key, not the regular DashScope one
}
```

Then in `agent.default_model`, just point at one of the model IDs the plan exposes (`claude-sonnet-4-5`, `qwen3-max`, etc.) and the requests are billed against your subscription.

I would not recommend the Coding Plan if you are running OpenClaw on a single laptop for personal use — DashScope's free tier is enough. I would absolutely recommend it if you are running anything with cron jobs or multiple channels.

## `tools` — turning sharp things on

Two tool configs are worth tuning:

**`exec.trusted_patterns`** — regexes that bypass the per-call confirmation. I keep mine narrow: read-only commands like `ls`, `cat`, `git status`. Anything that mutates the filesystem stays gated.

**`web_search.engine`** — `bing` (cheap, decent), `serper` (better quality, costs), or `tavily` (best for agentic search, costs more). I default to `bing` and let the agent ask if it needs better.

There are 24 other tools. Most of them have sensible defaults. The ones I have edited:

- `read.max_bytes` — bumped from 50KB to 500KB so the agent can swallow a config file.
- `write.allowed_paths` — restricted to `~/openclaw-workspace/` and `~/Documents/`. This is the single most useful safety setting.

## `memory` — how the agent remembers

```json
"memory": {
  "engine": "semantic",
  "max_tokens_per_turn": 2000,
  "auto_write": true,
  "types": ["user", "project", "reference", "feedback"]
}
```

`max_tokens_per_turn` is the budget for memory snippets in each prompt. Default is 1000. I bump to 2000 because skills already eat tokens; if memory is too small, the agent forgets.

`auto_write` lets the agent decide when to commit something to memory. With this off you have to say "remember that I prefer Python over Node" explicitly. With it on, the agent infers. I leave it on and accept the occasional spurious memory.

## `channels` — wired up later

Empty by default. The next piece in this series adds Telegram. The skeleton looks like:

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "...",
    "allowed_user_ids": [123456789]
  }
}
```

The `allowed_user_ids` field is non-optional in spirit — without it your bot answers anyone who finds it. Don't ship without it.

## `cron` — scheduled jobs

```json
"cron": {
  "jobs": [
    {
      "name": "daily-briefing",
      "schedule": "0 7 * * *",
      "skill": "daily-briefing"
    }
  ]
}
```

This is the feature that turns OpenClaw from "a chatbot" into "a thing that quietly does work for you." A cron entry triggers a skill; the skill produces output; the output is sent to a default channel (or whichever channel the skill specifies). The morning briefing case study is built on this.

## Reload after editing

Most of `openclaw.json` is hot-reloaded on save. The exceptions are channel registrations and provider keys — for those, restart the gateway:

```bash
openclaw gateway restart
```

If you ever see "tool not registered" or "provider not found" after editing, that's the cue.

## What I would change in your config today

If you onboarded with the wizard and haven't touched the file:

1. Set `tools.write.allowed_paths` to something explicit. The default is your home directory and that is too broad.
2. Set `memory.max_tokens_per_turn` to 2000.
3. If you are anywhere near 200元/month of token spend, switch to the Bailian Coding Plan.

Next piece, channels. We wire up Telegram first because it's painless, then take a single screenshot tour of WeChat WorkBuddy because it deserves its own day.
