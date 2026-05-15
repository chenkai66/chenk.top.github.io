---
title: "OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick"
date: 2026-04-11 09:00:00
tags:
  - openclaw
  - configuration
  - Bailian
  - coding-plan
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 4
series_total: 10
description: "openclaw.json walkthrough, model providers, and the Bailian Coding Plan trick."
disableNunjucks: true
translationKey: "openclaw-quickstart-4"
---

If you only edit one file in OpenClaw, this is it.

`~/.openclaw/openclaw.json` controls models, tools, channels, memory, cron, and skill loading. The onboarding wizard wrote a default; this piece walks the parts you will actually touch.

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_1.png)

---

## The minimal viable config

![Configuration hierarchy and provider resolution order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/fig_config.png)

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

## Full config reference

Before diving deep, here are all the top-level keys `openclaw.json` accepts:

| Key | Purpose |
|-----|---------|
| `agent` | Agent identity, default model, planning loop settings |
| `providers` | LLM provider endpoints and credentials (DashScope, Anthropic, OpenAI, Bailian, custom) |
| `tools` | Built-in tool configurations (exec, web_search, read, write, etc.) |
| `memory` | Memory engine, token budget, auto-write behavior, memory types |
| `channels` | External interfaces (Telegram, WeChat, DingTalk, web UI) |
| `cron` | Scheduled jobs that trigger skills at defined intervals |
| `mcp` | Model Context Protocol server registrations for third-party tools |
| `hooks` | Custom scripts to run before/after specific agent actions |
| `security` | Filesystem restrictions, blocked commands, confirmation requirements |

Most new users touch `agent`, `providers`, and `tools` first. The power-user configs are `cron`, `security`, and `mcp`.

## `agent` — name and default model

Two fields matter: the name (what the user sees) and the default model. Default model is what the agent uses for the planning loop unless a tool or skill overrides it.

The mistake people make is putting the most expensive model here, "to be safe." Don't. The planning loop runs on every turn. Put `qwen-plus` here, and override to `qwen3-max` only inside specific skills that need real reasoning.

## `providers` — where the models live

You can configure multiple providers and choose per-skill. Here are the four useful options:

| Provider | Use when |
|----------|----------|
| `dashscope` | China-hosted, cheap, Qwen family. Default choice for me. |
| `anthropic` | You want Claude reasoning quality and don't mind paying. |
| `openai` | You need GPT-4 or GPT-5.4 specifically. |
| `bailian-coding-plan` | You want one subscription that bundles Claude + Qwen + DeepSeek + GLM. |

Here's what the provider blocks look like in practice.

### DashScope (default)

```json
"providers": {
  "dashscope": {
    "api_key": "sk-...",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "models": ["qwen-turbo", "qwen-plus", "qwen3-max", "qwen3-coder-plus"]
  }
}
```

This is the sensible default for China-based workloads. DashScope's free tier gives 1 million tokens per month for `qwen-turbo`; paid tokens are cheap (0.004元/1K input for `qwen-plus`).

### Anthropic

```json
"providers": {
  "anthropic": {
    "api_key": "sk-ant-...",
    "models": ["claude-sonnet-4-5", "claude-opus-4", "claude-haiku-4"]
  }
}
```

Claude Sonnet 4.5 is the strongest reasoning model in OpenClaw today. Use this if your task requires deep multi-step logic (code generation, research planning, legal analysis). Pricing is $3/million input tokens — about 750x more expensive than `qwen-plus` — so route carefully.

### OpenAI

```json
"providers": {
  "openai": {
    "api_key": "sk-proj-...",
    "models": ["gpt-4o", "gpt-5.4", "o1-preview"]
  }
}
```

OpenAI models are useful in two cases: (1) you need GPT-5.4 specifically, or (2) your skill was prototyped with OpenAI and you don't want to rewrite. Otherwise, I avoid them—Claude is better for reasoning, and Qwen is cheaper for bulk work.

### The `compatible` provider type

Any service that speaks the OpenAI-compatible API format (Ollama, vLLM, LiteLLM, LM Studio, Together AI) can be added as a `compatible` provider:

```json
"providers": {
  "ollama-local": {
    "type": "compatible",
    "endpoint": "http://localhost:11434/v1",
    "models": ["llama3.2:8b", "qwen2.5:14b"]
  }
}
```

No API key required if your endpoint is local. This is how you run OpenClaw entirely offline — point `agent.default_model` at `ollama-local/llama3.2:8b` and all planning happens on your machine.

For hosted compatible endpoints (e.g., Together, Groq), add an `api_key` field.

### Model routing

The `agent.default_model` is your baseline, but individual skills can override it via `skill.model_override` in the skill manifest:

```json
{
  "name": "legal-contract-review",
  "model_override": "claude-sonnet-4-5",
  "description": "..."
}
```

This lets you run the planning loop on cheap tokens and switch to expensive reasoning only where needed. A typical setup:

- **Planning loop**: `qwen-plus` (fast, cheap)
- **Code generation skills**: `qwen3-coder-plus` (specialized for code)
- **Deep reasoning skills**: `claude-sonnet-4-5` (strongest logic)
- **Bulk summarization**: `qwen-turbo` (cheapest per token)

You can also override at runtime by passing `--model` to the CLI, but skill-level overrides are how you build cost discipline into your agent.

## The Coding Plan trick

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_2.png)

Aliyun Bailian has a "Coding Plan" subscription: 200元 per month gets you eight models — Claude Sonnet 4.5, Qwen3-Max, Qwen3-Coder-Plus, DeepSeek V3.1, GLM-4.6, and more — at unmetered usage during business hours. This is wildly underpriced for an agent that loops a few hundred times per day.

The provider config is one block:

```json
"bailian-coding-plan": {
  "endpoint": "https://dashscope.aliyuncs.com/coding-plan/v1",
  "api_key": "sk-..."
}
```

Then in `agent.default_model`, just point at one of the model IDs the plan exposes (`claude-sonnet-4-5`, `qwen3-max`, etc.) and the requests are billed against your subscription.

I wouldn't recommend the Coding Plan for personal use on a single laptop—DashScope's free tier is sufficient. However, I highly recommend it for setups with cron jobs or multiple channels.

## `tools` — turning sharp things on

Two tool configs are worth tuning:

**`exec.trusted_patterns`** — regexes that bypass the per-call confirmation. I keep mine narrow: read-only commands like `ls`, `cat`, `git status`. Anything that mutates the filesystem stays gated.

**`web_search.engine`** — `bing` (cheap, decent), `serper` (better quality, costs), or `tavily` (best for agentic search, costs more). I default to `bing` and let the agent ask if it needs better.

There are 24 other tools, most with sensible defaults. The ones I've edited:

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

## `security` — what the agent cannot touch

Security config is optional but highly recommended if your agent runs unattended (cron jobs).bs) or if you expose it to other users (shared Telegram bot).

```json
"security": {
  "allowed_paths": [
    "~/openclaw-workspace/",
    "~/Documents/",
    "~/Desktop/temp/"
  ],
  "blocked_commands": [
    "rm -rf",
    "sudo",
    "curl.*bash",
    "wget.*\\|.*sh"
  ],
  "require_confirmation": [
    "git push",
    "npm publish",
    "docker rm"
  ]
}
```

**`allowed_paths`** restricts filesystem writes. Any `write`, `edit`, or `exec` call that touches a path outside this list is rejected. This is the single biggest safety lever — it prevents an accidental "delete my home directory" from cascading.

**`blocked_commands`** are regexes matched against `exec` tool calls. If the command matches, it's rejected outright — no confirmation prompt. Use this for commands that should never run (`rm -rf /`, piping `curl` into `bash`, privilege escalation).

**`require_confirmation`** forces a user confirmation prompt even if the command matches a `trusted_pattern`. I use this for irreversible actions like `git push --force`, publishing packages, or deleting cloud resources.

Without these, an agent with `exec` enabled can do anything you can do. With them, you have guardrails.

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

### More cron examples

**Weekly project report** — sends a summary of commits, issues closed, and PRs merged every Monday morning:

```json
{
  "name": "weekly-project-report",
  "schedule": "0 9 * * 1",
  "skill": "summarize-github-activity",
  "channel": "telegram",
  "params": { "repo": "org/project", "days": 7 }
}
```

**Competitor monitoring** — checks competitor blogs and product pages every morning, surfaces new launches:

```json
{
  "name": "competitor-monitor",
  "schedule": "30 8 * * *",
  "skill": "web-scrape-and-diff",
  "channel": "dingtalk",
  "params": {
    "urls": ["https://competitor.com/blog", "https://competitor.com/pricing"],
    "notify_on_change": true
  }
}
```

**Dependency security audit** — runs `npm audit` or `pip-audit` every Sunday, sends a report if vulnerabilities are found:

```json
{
  "name": "dependency-audit",
  "schedule": "0 10 * * 0",
  "skill": "run-security-audit",
  "channel": "wechat-workbuddy",
  "params": { "project_path": "~/code/myapp", "severity_threshold": "high" }
}
```

The `schedule` field uses standard cron syntax: `minute hour day-of-month month day-of-week`. All times are server-local unless you set `TZ` in the environment.

## Environment variables

Some config values can be overridden via environment variables. This is useful for Docker deploys where you don't want API keys committed to the config file:

| Config path | Environment variable |
|-------------|---------------------|
| `providers.dashscope.api_key` | `DASHSCOPE_API_KEY` |
| `providers.anthropic.api_key` | `ANTHROPIC_API_KEY` |
| `providers.openai.api_key` | `OPENAI_API_KEY` |
| `channels.telegram.bot_token` | `TELEGRAM_BOT_TOKEN` |
| `memory.max_tokens_per_turn` | `OPENCLAW_MEMORY_TOKENS` |
| `agent.default_model` | `OPENCLAW_DEFAULT_MODEL` |

Precedence order: environment variable > `openclaw.json` > built-in default.

For secrets, prefer environment variables. For everything else, keep it in the config file — it's version-controlled and easier to audit.

## Config validation

Before you restart the gateway after editing, run:

```bash
openclaw config validate
```

This catches:

- **Missing required fields** — e.g., `providers.dashscope.api_key` is empty
- **Invalid model IDs** — e.g., `agent.default_model` set to a model not registered in any provider
- **Malformed JSON** — trailing commas, missing brackets
- **Bad cron expressions** — e.g., `schedule: "0 25 * * *"` (hour 25 doesn't exist)
- **Path conflicts** — e.g., `security.allowed_paths` includes a path that doesn't exist

Common errors I see:

1. **Forgot to quote model names** — `"qwen-plus"`, not `qwen-plus`.
2. **Trailing commas in JSON** — the last item in an object or array must not have a comma after it. JSON is stricter than JavaScript.
3. **Provider mismatch** — `agent.default_model: "claude-sonnet-4-5"` but no `anthropic` provider configured. The validator will tell you which provider is missing.

If validation fails, the error message includes a line number and field path. Fix it, validate again, then restart:

```bash
openclaw gateway restart
```

## Reload after editing

Most of `openclaw.json` is hot-reloaded on save. The exceptions are channel registrations and provider keys — for those, restart the gateway:

```bash
openclaw gateway restart
```

If you ever see "tool not registered" or "provider not found" after editing, that's the cue.

## What I would change in your config today

If you onboarded with the wizard and haven't touched the file:

1. Set `security.allowed_paths` to something explicit. The default is your home directory and that is too broad.
2. Set `memory.max_tokens_per_turn` to 2000.
3. Add `security.blocked_commands` entries for `rm -rf`, `sudo`, and `curl.*bash`.
4. If you are anywhere near 200元/month of token spend, switch to the Bailian Coding Plan.
5. Run `openclaw config validate` after every edit. Catch errors before they break your agent.

Next piece, channels. We wire up Telegram first because it's painless, then take a single screenshot tour of WeChat WorkBuddy because it deserves its own day.
