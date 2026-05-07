---
title: "OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work"
date: 2026-04-05 09:00:00
tags:
  - openclaw
  - architecture
  - agent-loop
  - gateway
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 3
description: "The Gateway, Pi Agent, Tools, Skills, Memory, and Channels — what each does, how they fit, and why the separation matters when you start writing your own skills."
disableNunjucks: true
translationKey: "openclaw-quickstart-3"
---

You can use OpenClaw for months without reading this piece. But the first time you try to write a skill, debug a misrouted message, or figure out why the agent forgot something, you will want to know what each box does.

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_1.jpg)

## The six layers

```
        +-----------------------------------------------+
        |                  Channels                     |   <-- DingTalk, Telegram, ...
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |                  Gateway                      |   <-- :18789, normalizes messages
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Router  +  Sessions   +  Pi Agent (loop)    |   <-- decides which agent runs
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Tools (26)        Skills (53+ built-in)     |   <-- "what it can do" + "how"
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Memory + ContextEngine                      |   <-- persistent context
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   LLM provider                                |   <-- DashScope / Anthropic / ...
        +-----------------------------------------------+
```

I will walk top to bottom.

## Channels — adapters, not transports

A Channel is the code that turns "a DingTalk Stream message" into "a normalized OpenClaw message" and vice versa. Each channel has its own quirks: DingTalk sends Stream events over a WebSocket, Telegram polls or webhooks, Discord uses a Gateway WebSocket of its own. The Channel layer hides all of that.

What you need to remember:

- Channels are configured per-instance. You can run with zero, one, or twenty.
- A message goes Channel → Gateway → Agent → Gateway → Channel. The Channel doesn't talk to the Agent directly.
- Per-channel rate limits and quirks live in the channel adapter — that's why DingTalk replies feel different from Telegram replies.

## Gateway — the central nervous system

The Gateway runs on `:18789`. It accepts messages from any channel, deduplicates them (DingTalk sometimes redelivers), assigns or restores a session, and hands the message to the Router.

The Gateway is also the only thing that talks to the model provider. Every tool result, every memory read, every prompt assembly goes through it. That's why you only need one set of API keys.

## Router and Sessions

The Router decides *which* agent should handle a message — relevant only if you've configured multiple agents (the default install has one, called Pi). Sessions are how OpenClaw keeps a conversation in WeChat distinct from a conversation in Telegram even though they go to the same agent. Session ID is `(channel, conversation_id)`.

If you have ever seen "the agent confused two of my conversations", that is a session-ID collision and almost always a custom-channel bug.

## Pi Agent — the loop

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_2.jpg)

This is the actual agent loop. It is the bit that looks like:

```
while True:
    plan = LLM(messages, tools=enabled_tools, skills=hot_skills)
    if plan.is_terminal:
        return plan.reply
    for tool_call in plan.tool_calls:
        result = run_tool(tool_call)
        messages.append(result)
```

The interesting choices OpenClaw makes here:

- **Skills are loaded lazily.** Only the manifest is in the system prompt. The body is paged in when the model triggers a skill. This keeps token cost low.
- **Tool errors are returned to the model, not raised.** The model gets a chance to recover. This sounds obvious but a lot of agent frameworks just throw.
- **The loop has a hard turn-limit.** Default 30. If the agent is still looping at turn 30 it stops and emits a "I think I'm stuck" message rather than burning your token budget overnight.

## Tools — what the agent *can* do

Tools are the verbs. Read a file, write a file, exec a shell command, fetch a URL, search the web. The default install ships 26 of them. Each tool has:

- A name (`read`, `exec`, `web_search`, ...)
- A schema (typed args)
- A handler (the actual code that runs)
- A permission level

`exec` is the dangerous one. It runs arbitrary shell. By default it requires confirmation per call; you can mark trusted patterns in `openclaw.json`.

## Skills — *how* to do something

Skills are nouns-of-knowledge. A Skill is a Markdown file at `~/.openclaw/skills/<name>/SKILL.md` plus optional helper files. The manifest at the top of the file looks like:

```yaml
---
name: obsidian-notes
description: Manage Obsidian vault notes
trigger: when user asks to take notes, search notes, link notes
tools_required: [read, write, exec]
---
```

The body is the SOP — instructions, templates, and examples. The agent loads the manifest at startup, so the model sees a one-line summary of every skill. When the model decides a skill applies, the gateway expands the body into the prompt for the next turn.

Skills are how you turn an LLM into a reliable worker on your specific tasks. Tools answer "can I read a file?". Skills answer "given that I'm writing a meeting note, what's the right template, where does it go, and what do I link to?".

## Memory + ContextEngine

Memory is per-user, persistent, and typed. Common types:

- `user/profile.md` — preferences
- `project/<name>.md` — project state
- `feedback/*.md` — corrections you gave the agent
- `reference/*.md` — facts the agent should remember

The ContextEngine is the v2026.3.7 addition that decides which memory snippets to include in the next prompt. It scores by recency, relevance to the current message, and explicit tags. You can swap the engine — there is a `noop`, a `recency-only`, and the default semantic one.

This is the layer that makes the agent feel like it remembers you. If yours doesn't, it's almost always because the ContextEngine isn't getting enough write opportunities — the agent has to be told to write memory.

## Why the layering matters

Two practical consequences:

1. **You write skills, not agents.** The agent loop is fixed. Your customization happens at the Skill layer (knowledge) and the Tool layer (verbs). You almost never need to touch the gateway.

2. **The same agent can serve every channel.** Because the loop and the channels are decoupled, the productivity skill you wrote for your terminal works the same way on DingTalk and Telegram. No port, no rewrite.

Next piece is configuration — `openclaw.json`, model providers, and the Bailian Coding Plan that's the cheapest sane way to run this in China.
