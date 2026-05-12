---
title: "OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work"
date: 2026-04-10 09:00:00
tags:
  - openclaw
  - Architecture
  - agent-loop
  - gateway
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 3
description: "Gateway, Pi Agent, Tools, Skills, Memory, Channels — what each layer does."
disableNunjucks: true
translationKey: "openclaw-quickstart-3"
---

You can use OpenClaw for months without reading this. But the first time you need to write a skill, debug a misrouted message, or figure out why the agent forgot something, you'll want to know what each component does.

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_1.png)

## The six layers

![The six layers of an OpenClaw agent: Channels, Gateway, Router+Sessions+Pi Agent, Tools+Skills, Memory+ContextEngine, LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/fig1_six_layers.png)

I will walk top to bottom.

## Channels — adapters, not transports

A Channel is the code that turns "a DingTalk Stream message" into "a normalized OpenClaw message" and vice versa. Each channel has its own quirks: DingTalk sends Stream events over a WebSocket, Telegram polls or webhooks, Discord uses a Gateway WebSocket of its own. The Channel layer hides all of that.

What you need to remember:

- Channels are configured per-instance. You can run with zero, one, or twenty.
- A message goes Channel → Gateway → Agent → Gateway → Channel. The Channel doesn't talk to the Agent directly.
- Per-channel rate limits and quirks live in the channel adapter — that's why DingTalk replies feel different from Telegram replies.

**What breaks here**: The most common failure is WebSocket disconnection that isn't auto-recovered. DingTalk Stream connections drop after 300 seconds of inactivity; if the channel adapter doesn't detect this and reconnect, messages pile up in the broker and you see delivery delays ranging from seconds to never. Check `gateway.log` for "channel reconnect" events and the timestamp gap. If you see gaps over 5 minutes, your keep-alive isn't working.

## Gateway — the central nervous system

The Gateway runs on `:18789`. It accepts messages from any channel, deduplicates them (DingTalk sometimes redelivers), assigns or restores a session, and hands the message to the Router.

The Gateway is also the only thing that talks to the model provider. Every tool result, every memory read, every prompt assembly goes through it. That's why you only need one set of API keys.

**What breaks here**: Rate-limit exhaustion at the provider level. If ten users send messages simultaneously, the Gateway serializes LLM calls but doesn't throttle ingress. You will see HTTP 429 from the provider, which the Gateway retries with exponential backoff (max 3 retries). If all three fail, the user gets "I'm having trouble thinking right now" and the turn is logged to `gateway_errors.jsonl`. The fix is either upgrading your provider tier or configuring `max_concurrent_llm_calls` in `openclaw.json` to match your quota.

## Router and Sessions

The Router decides *which* agent should handle a message — relevant only if you've configured multiple agents (the default install has one, called Pi). Sessions are how OpenClaw keeps a conversation in WeChat distinct from a conversation in Telegram even though they go to the same agent. Session ID is `(channel, conversation_id)`.

If you have ever seen "the agent confused two of my conversations", that is a session-ID collision and almost always a custom-channel bug.

**What breaks here**: Session ID collision when a custom channel doesn't provide a stable `conversation_id`. The symptom is the agent mixing up context from two different chats. DingTalk uses `conversationId` from the webhook payload; Telegram uses `chat.id`. If you are writing a custom channel and hashing multiple fields to create the ID, make sure every field is present in *every* message type (text, image, callback) or you will generate different IDs for the same logical conversation. The diagnostic is `openclaw debug --session <user_id>` which dumps all sessions for that user.

## Pi Agent — the loop

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_2.png)

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

**What breaks here**: Infinite tool-call loops when a tool returns success but doesn't change state. The classic case is `web_search` returning zero results: the model sees "success: []", decides it needs to refine the query, calls `web_search` again with a slightly different query, gets another empty list, repeats until turn 30. The loop protection catches this, but you burn 30 LLM calls for nothing. The fix is making tools return semantic errors — "success: false, reason: no results for query" — so the model can give up earlier. You can also lower `max_turns` in the agent config if you notice this pattern.

## Tools — what the agent *can* do

Tools are the verbs. Read a file, write a file, exec a shell command, fetch a URL, search the web. The default install ships 26 of them. Each tool has:

- A name (`read`, `exec`, `web_search`, ...)
- A schema (typed args)
- A handler (the actual code that runs)
- A permission level

`exec` is the dangerous one. It runs arbitrary shell. By default it requires confirmation per call; you can mark trusted patterns in `openclaw.json`.

### Built-in tools reference

| Tool | What it does | Permission | Common config override |
|------|-------------|-----------|----------------------|
| `read` | Read file contents, supports line range | `safe` | `max_file_size_mb: 10` |
| `write` | Write file (creates parent dirs) | `safe` | `backup_on_overwrite: true` |
| `edit` | Regex-based in-place edit | `safe` | `require_confirmation: false` |
| `exec` | Run shell command, streams output | `dangerous` | `trusted_commands: ["git status", "ls"]` |
| `web_search` | Search via configured provider (DuckDuckGo default) | `safe` | `max_results: 5`, `provider: "bing"` |
| `web_fetch` | Fetch URL, render to markdown | `safe` | `timeout_sec: 10`, `user_agent: "..."` |
| `git_status` | Shortcut for `git status --short` | `safe` | — |
| `git_diff` | Shortcut for `git diff`, respects .gitignore | `safe` | `context_lines: 3` |
| `calendar_list` | List calendar events (requires OAuth) | `safe` | `max_days_ahead: 7` |
| `send_email` | Send via configured SMTP | `requires_confirm` | `from_address: "bot@..."` |

**What breaks here**: Permission denials when a skill tries to use a tool the user hasn't authorized. The model gets a tool error "permission denied: exec requires dangerous approval" and usually surfaces it to the user. But if the skill is tightly scripted and doesn't handle the error, the agent just stops mid-task. You can pre-authorize tools per-skill in `openclaw.json` under `skill_permissions`, e.g., `{"obsidian-notes": {"allow_dangerous": false, "allow_web": true}}`.

### Custom tools

To register a new tool:

1. **Write the handler.** Create `~/.openclaw/tools/my_tool.py`:
   ```python
   from openclaw.tools import Tool, ToolResult
   
   class MyTool(Tool):
       name = "my_tool"
       description = "Does something useful"
       schema = {"arg1": "string", "arg2": "integer"}
       permission = "safe"
       
       def run(self, arg1: str, arg2: int) -> ToolResult:
           # your logic
           return ToolResult(success=True, output="done")
   ```

2. **Declare the schema.** The `schema` dict becomes a JSON Schema fragment. OpenClaw validates args before calling `run()`.

3. **Add to config.** In `openclaw.json`, under `tools.custom`, add `{"module": "my_tool", "enabled": true}`. Restart the gateway.

The tool is now visible to all agents. If you want it only for specific skills, use `tools_required: [my_tool]` in the skill manifest; the gateway won't load the tool into the prompt unless that skill is active.

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

**What breaks here**: Skill trigger ambiguity when multiple skills have overlapping `trigger` clauses. The model picks one (usually the first alphabetically) and you get the wrong skill. The symptom is the agent using the wrong template or searching the wrong directory. The fix is making triggers mutually exclusive — instead of "when user asks about notes" in two skills, use "when user asks about *meeting* notes" and "when user asks about *project* notes". You can also set `priority: high` in one skill's manifest to bias selection.

## Memory + ContextEngine

Memory is per-user, persistent, and typed. Common types:

- `user/profile.md` — preferences
- `project/<name>.md` — project state
- `feedback/*.md` — corrections you gave the agent
- `reference/*.md` — facts the agent should remember

The ContextEngine is the v2026.3.7 addition that decides which memory snippets to include in the next prompt. It scores by recency, relevance to the current message, and explicit tags. You can swap the engine — there is a `noop`, a `recency-only`, and the default semantic one.

This is the layer that makes the agent feel like it remembers you. If yours doesn't, it's almost always because the ContextEngine isn't getting enough write opportunities — the agent has to be told to write memory.

### The five-stage lifecycle in action

Imagine you say "help me deploy this Docker container" in a fresh session. Here is what the ContextEngine does:

1. **Pre-turn (Retrieval)**: Searches `user/*.md` and `project/*.md` for embeddings near "Docker deploy". Finds `project/infra.md` (mentions your ECS instance) and `reference/docker-compose-template.md`. Injects both into system prompt under `## Relevant Context`.

2. **Planning**: The agent sees the context, realizes you have an ECS instance at `120.26.104.90`, and decides to use the `exec` tool to run `docker ps` remotely.

3. **Tool execution**: After the tool returns, the ContextEngine hook `on_tool_result` fires. It checks if the result contains new facts (it does: three containers running, one is PostgreSQL). Appends to `project/infra.md`: "Last checked 2026-05-08: postgres:14 running on :5432".

4. **Response generation**: The model drafts "I see you have PostgreSQL running; does your new container need to connect to it?". Before sending, the ContextEngine hook `on_response_ready` fires. It checks if the response implies a decision (it does: the user will answer yes or no). Writes to `user/conversation_state.json`: `{"pending_decision": "postgres_link", "expires_at": "2026-05-08T18:00"}`.

5. **Post-turn (Cleanup)**: If the user doesn't reply within the TTL, the ContextEngine purges `pending_decision` so the next conversation doesn't start mid-thread.

The result: three turns later, when you say "yes, connect it", the agent doesn't ask "connect what?" because `conversation_state.json` is in context.

**What breaks here**: Memory writes that exceed the file-size budget. The default ContextEngine caps each memory file at 50 KB. If a project file grows past that (common in long projects), old content is evicted FIFO and you lose early decisions. The symptom is the agent asking questions you already answered weeks ago. The fix is splitting large project files by subproject or increasing `max_memory_file_kb` in `openclaw.json`. Check `~/.openclaw/memory/user_<id>/*.md` file sizes with `du -h`.

## Tracing a message end-to-end

Walk through what happens when you send "@Lobster what time is my next meeting" in DingTalk.

**T+0ms**: DingTalk Stream sends a WebSocket frame to your channel adapter. Payload is JSON, includes `conversationId`, `senderId`, `text`. The adapter ACKs immediately (required within 500ms or DingTalk retries).

**T+5ms**: Channel adapter normalizes the message to `OpenClawMessage(channel="dingtalk", conversation_id="...", user_id="...", text="@Lobster what time is my next meeting", timestamp=...)`. Strips the @-mention. POSTs to Gateway `:18789/v1/message`.

**T+8ms**: Gateway deduplicates by computing `hash(conversation_id + timestamp + text)`. Checks the last 100 messages in a ring buffer; if this hash exists, drops it. If not, continues.

**T+10ms**: Gateway looks up or creates a session. Session ID is `("dingtalk", conversationId)`. Loads the last 20 turns from `~/.openclaw/sessions/<session_id>.jsonl`. This gives the agent conversation history.

**T+12ms**: Router picks the agent. Default config has one agent (Pi), so this is instant. In multi-agent setups, the router runs a tiny classifier (100ms).

**T+15ms**: ContextEngine retrieval runs. Embeds "what time is my next meeting", searches `user/*.md` and `reference/*.md`. Finds `user/profile.md` (your calendar is Google) and `reference/calendar-oauth.md`. Injects both. This takes 200ms if the embedding model is remote, <10ms if local.

**T+215ms**: Agent loop starts. First LLM call. Prompt is system + context + history + user message. Model returns a plan: `[{tool: "calendar_list", args: {max_results: 3}}]`. This takes 800ms (Qwen-Max) to 2000ms (GPT-4) depending on provider.

**T+1015ms**: Gateway executes `calendar_list`. This hits Google Calendar API, waits for OAuth token refresh if needed, fetches events. Takes 300ms.

**T+1315ms**: Tool result appended to message history. Second LLM call. Model sees the event list, generates final reply: "Your next meeting is at 2pm: Sprint Planning". Takes another 600ms.

**T+1915ms**: Agent loop terminates (model set `is_terminal=true`). ContextEngine `on_response_ready` hook fires, writes nothing (no new facts). Gateway wraps the reply in a DingTalk card JSON structure.

**T+1920ms**: Gateway POSTs the card to DingTalk webhook API. DingTalk returns 200 in 50ms. Total user-perceived latency: **1970ms** (just under 2 seconds).

The bottlenecks are always LLM calls (1400ms of the total) and external APIs (300ms for calendar). Local tools like `read` add <10ms. If you see >5s latency, it is either provider rate-limiting (check for retries in `gateway.log`) or a tool hanging (check `tool_durations` in the turn JSON).

## Debugging the loop

When things go wrong, you need to see exactly what the agent is thinking. Three tools:

**1. Verbose mode**: `openclaw debug --verbose` tails `gateway.log` and pretty-prints every LLM call. You see the full prompt (including system, tools, and context), the model's reply (parsed tool calls or text), and the token count. Run this in a second terminal while you send test messages.

**2. Gateway log levels**: In `openclaw.json`, set `"log_level": "DEBUG"`. This adds:
   - Every tool input and output (truncated to 500 chars)
   - ContextEngine retrieval scores (which memory files matched and why)
   - Session load/save events (so you can see if history is persisting)

The log file grows to several MB per day at DEBUG. Reset it weekly or configure rotation.

**3. Turn-by-turn JSON dump**: Every turn is appended to `~/.openclaw/sessions/<session_id>.jsonl`. Each line is a JSON object:
   ```json
   {
     "turn": 5,
     "timestamp": "2026-05-08T14:32:10Z",
     "user_message": "what time is my next meeting",
     "context_injected": ["user/profile.md", "reference/calendar-oauth.md"],
     "plan": [{"tool": "calendar_list", "args": {}}],
     "tool_results": [{"success": true, "output": "..."}],
     "final_reply": "Your next meeting is at 2pm: Sprint Planning",
     "tokens_used": 1523
   }
   ```

If the agent did something inexplicable six turns ago, `cat ~/.openclaw/sessions/<session_id>.jsonl | jq 'select(.turn==6)'` shows you exactly what context and tools were in play. This is the fastest way to diagnose "why did it forget X" or "why did it call the wrong tool".

## Why the layering matters

Two practical consequences:

1. **You write skills, not agents.** The agent loop is fixed. Your customization happens at the Skill layer (knowledge) and the Tool layer (verbs). You almost never need to touch the gateway.

2. **The same agent can serve every channel.** Because the loop and the channels are decoupled, the productivity skill you wrote for your terminal works the same way on DingTalk and Telegram. No port, no rewrite.

Next piece is configuration — `openclaw.json`, model providers, and the Bailian Coding Plan that's the cheapest sane way to run this in China.
