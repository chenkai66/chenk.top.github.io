---
title: "OpenClaw 指南（三）：Agent Loop 六层架构"
date: 2026-04-10 09:00:00
tags:
  - openclaw
  - Architecture
  - agent-loop
  - gateway
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 3
description: "Gateway、Pi Agent、工具、技能、记忆和渠道——每一层做什么、怎么配合、为什么在你写自定义技能时这个分层设计很重要。"
disableNunjucks: true
translationKey: "openclaw-quickstart-3"
---
你可能用好几个月 OpenClaw 都不需要看这篇，但只要第一次编写 Skill、调试消息路由异常或疑惑 Agent 为何突然“失忆”，就必须厘清各模块的职责。

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_1.png)

## 六层架构

![The six layers of an OpenClaw agent: Channels, Gateway, Router+Sessions+Pi Agent, Tools+Skills, Memory+ContextEngine, LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/fig1_six_layers.png)

我们将按自上而下的顺序梳理。

## Channels —— 是适配器，不是传输层

Channel 的代码负责双向转换：把钉钉 Stream 消息转为标准化 OpenClaw 消息，反之亦然。不同平台实现各异——钉钉通过 WebSocket 接收 Stream 事件，Telegram 用轮询或 Webhook，Discord 则接入其专属 Gateway WebSocket。Channel 层将这些差异完全屏蔽。

需注意三点：Channel 按实例配置，可部署零个、一个或多个；消息严格遵循 Channel → Gateway → Agent → Gateway → Channel 流向，Channel 与 Agent 从不直连；速率限制和平台特有行为均由各 Channel 适配器自行实现，因此钉钉与 Telegram 的响应表现天然不同。

**这里容易出什么问题**：最常见的问题是 WebSocket 连接断开后未触发自动重连。钉钉 Stream 连接空闲 300 秒后会断开；若 Channel 适配器未能及时检测并重连，消息将在 Broker 中积压，最终导致交付延迟从数秒演变为永久失败。去查 `gateway.log` 里的 "channel reconnect" 事件和时间戳间隙。如果看到超过 5 分钟的间隙，说明你的 keep-alive 没起作用。

## Gateway —— 中枢神经系统

Gateway 跑在 `:18789`。它接收来自任何 Channel 的消息，做去重（钉钉有时会重发），分配或恢复 Session，然后把消息交给 Router。

Gateway 是唯一对接模型服务商的组件，所有工具调用结果、内存读取、Prompt 组装都由它统一处理——因此只需配置一套 API Key。

**这里容易出什么问题**：模型服务商的速率限制被耗尽。如果十个用户同时发消息， Gateway 会对 LLM 调用进行串行化处理，但不对接入流量做限流。你会看到提供商返回 HTTP 429， Gateway 会用指数退避重试（最多 3 次）。如果三次都失败，用户会收到“我现在有点思考困难”，并且这次_turn_会被记录到 `gateway_errors.jsonl`。解决办法要么升级提供商套餐，要么在 `openclaw.json` 里配置 `max_concurrent_llm_calls` 来匹配你的配额。

## Router 与 Sessions

Router 负责路由消息到目标 Agent——仅当配置多个 Agent 时才生效（默认仅 Pi 一个）； Session 则用于区分不同渠道的对话（如微信 vs Telegram），即使它们共用同一 Agent。 Session ID 是 `(channel, conversation_id)`。

若出现‘Agent 混淆了两个对话’的情况，通常是 Session ID 冲突所致，根源往往是自定义 Channel 的实现缺陷。

**这里容易出什么问题**：自定义 Channel 没提供稳定的 `conversation_id` 导致 Session ID 冲突。典型表现为 Agent 将两个独立对话的上下文混淆。钉钉用 Webhook payload 里的 `conversationId`； Telegram 用 `chat.id`。如果你在写自定义 Channel 并且哈希多个字段来创建 ID，确保*每种*消息类型（文本、图片、回调）都包含这些字段，否则你会为同一个逻辑对话生成不同的 ID。诊断命令是 `openclaw debug --session <user_id>`，这会 dump 出该用户的所有 Session。

## Pi Agent —— 核心执行循环

![OpenClaw QuickStart (3): The Six Layers That Make the Agent Loop Work — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_2.png)

这才是 Agent 的核心执行循环。代码如下：

```
while True:
    plan = LLM(messages, tools=enabled_tools, skills=hot_skills)
    if plan.is_terminal:
        return plan.reply
    for tool_call in plan.tool_calls:
        result = run_tool(tool_call)
        messages.append(result)
```

OpenClaw 在这里做了几个有意思的选择：

- **Skills 懒加载。** 系统 Prompt 仅包含 manifest。只有当模型触发某个 Skill 时，主体内容才会被分页加载，从而降低 Token 成本。
- **工具执行错误会返回给模型，而非抛出异常。** 模型有机会恢复。这看似理所当然，但多数 Agent 框架会直接抛出异常。
- **循环设有硬性_turn_限制。** 默认 30。若 Agent 在第 30 轮仍处于循环中，将主动终止并返回‘我觉得我卡住了’提示，避免持续消耗 Token 预算。

**这里容易出什么问题**：工具返回成功，但实际状态未改变，从而引发无限工具调用循环。经典案例是 `web_search` 返回零结果：模型看到 "success: []"，决定细化查询，再次调用 `web_search` 换个查询词，又拿到空列表，重复直到第 30 轮。循环保护机制会捕获这个，但你白白浪费了 30 次 LLM 调用。解决办法是让工具返回语义化错误——"success: false, reason: no results for query"——这样模型可以早点放弃。如果你注意到这种模式，也可以在 Agent 配置里调低 `max_turns`。

## Tools —— Agent 能*做什么*

Tools 表示可执行的操作，例如读取文件、写入文件、执行 Shell 命令、抓取 URL、网页搜索等。默认安装包含 26 个内置 Tool。每个 Tool 都有：

- 名字（`read`, `exec`, `web_search`, ...）
- Schema （类型化的参数）
- Handler （实际运行的代码）
- 权限级别

`exec` 很危险。它运行任意 Shell。默认每次调用都需要确认；你可以在 `openclaw.json` 里标记受信任的模式。

### 内置 Tools 参考

| Tool | 功能 | 权限 | 常见配置覆盖 |
|------|-------------|-----------|----------------------|
| `read` | 读文件内容，支持行范围 | `safe` | `max_file_size_mb: 10` |
| `write` | 写文件（创建父目录） | `safe` | `backup_on_overwrite: true` |
| `edit` | 基于正则的原地编辑 | `safe` | `require_confirmation: false` |
| `exec` | 运行 Shell 命令，流式输出 | `dangerous` | `trusted_commands: ["git status", "ls"]` |
| `web_search` | 通过配置的提供商搜索（默认 DuckDuckGo） | `safe` | `max_results: 5`, `provider: "bing"` |
| `web_fetch` |  Fetch URL，渲染为 Markdown | `safe` | `timeout_sec: 10`, `user_agent: "..."` |
| `git_status` | `git status --short` 的快捷方式 | `safe` | — |
| `git_diff` | `git diff` 的快捷方式，尊重 .gitignore | `safe` | `context_lines: 3` |
| `calendar_list` | 列出日历事件（需要 OAuth） | `safe` | `max_days_ahead: 7` |
| `send_email` | 通过配置的 SMTP 发送 | `requires_confirm` | `from_address: "bot@..."` |

**这里容易出什么问题**： Skill 试图使用用户未授权的 Tool 导致权限拒绝。模型会收到工具错误 "permission denied: exec requires dangerous approval" 并通常展示给用户。但如果 Skill 是 tightly scripted 且不处理错误， Agent 就会任务中途停止。你可以在 `openclaw.json` 的 `skill_permissions` 下按 Skill 预授权 Tools，例如 `{"obsidian-notes": {"allow_dangerous": false, "allow_web": true}}`。

### 自定义 Tools

注册新 Tool 的步骤：

1. **写 Handler。** 创建 `~/.openclaw/tools/my_tool.py`：
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

2. **声明 Schema。** `schema` 字典会变成 JSON Schema 片段。 OpenClaw 在调用 `run()` 之前验证参数。

3. **加入配置。** 在 `openclaw.json` 的 `tools.custom` 下，添加 `{"module": "my_tool", "enabled": true}`。重启 Gateway。

现在所有 Agent 都能看到这个 Tool。如果你只想让特定 Skills 使用它，在 Skill manifest 里用 `tools_required: [my_tool]`；除非该 Skill 激活，否则 Gateway 不会把这个 Tool 加载到 Prompt 里。

## Skills —— *怎么做*

Skills 是知识名词。 Skill 是 `~/.openclaw/skills/<name>/SKILL.md` 文件加上可选的辅助文件。文件顶部的 manifest 长这样：

```yaml
---
name: obsidian-notes
description: Manage Obsidian vault notes
trigger: when user asks to take notes, search notes, link notes
tools_required: [read, write, exec]
---
```

主体内容是 SOP——指令、模板和示例。 Agent 在启动时加载 manifest，所以模型能看到每个 Skill 的一行摘要。当模型决定适用某个 Skill 时， Gateway 会在下一轮把主体内容展开进 Prompt。

Skills 是将 LLM 转化为特定任务可靠执行者的核心机制。 Tools 解决‘能否读取文件’这类能力问题； Skills 则解决‘撰写会议记录时，应使用何种模板、保存至何处、需关联哪些内容’等流程性问题。

**这里容易出什么问题**：多个 Skill 的 `trigger` 子句重叠导致触发歧义。模型将选择其一（通常为字典序首个），导致调用错误的 Skill。症状是 Agent 用了错的模板或者搜了错的目录。解决办法是让触发条件互斥——不要在两个 Skill 里都用“当用户询问笔记时”，而是用“当用户询问*会议*笔记时”和“当用户询问*项目*笔记时”。你也可以在一个 Skill 的 manifest 里设置 `priority: high` 来 bias 选择。
## Memory + ContextEngine

内存是按用户隔离的，持久化且带类型。常见的类型包括：

- `user/profile.md` — 偏好设置
- `project/<name>.md` — 项目状态
- `feedback/*.md` — 你给 Agent 的修正反馈
- `reference/*.md` — Agent 应该记住的事实

ContextEngine 是 v2026.3.7 版本新加的成分，负责决定哪些内存片段会进入下一个 Prompt。它根据近期性、与当前消息的相关性以及显式标签来打分。你可以替换这个引擎——目前有一个 `noop` 模式，一个 `recency-only` 模式，以及默认的语义搜索模式。

正是这一层让 Agent 显得像是真的记得你。若 Agent 缺乏上下文连贯性，通常是因为 ContextEngine 写入机会不足——需显式调用内存写入指令。

### 五阶段生命周期实战

假设你在新会话里说了一句 "help me deploy this Docker container"。 ContextEngine 会这么处理：

1. **Pre-turn (Retrieval)**：在 `user/*.md` 和 `project/*.md` 中搜索与 "Docker deploy" 接近的嵌入向量。找到 `project/infra.md`（提到了你的 ECS 实例）和 `reference/docker-compose-template.md`。把这两者注入到系统 Prompt 的 `## Relevant Context` 下。

2. **Planning**： Agent 看到上下文，意识到你有一台 IP 为 `120.26.104.90` 的 ECS 实例，决定使用 `exec` 工具远程运行 `docker ps`。

3. **Tool execution**：工具返回结果后， ContextEngine 的钩子 `on_tool_result` 触发。它检查结果是否包含新事实（确实有：三个容器运行中，一个是 PostgreSQL）。追加内容到 `project/infra.md`："Last checked 2026-05-08: postgres:14 running on :5432"。

4. **Response generation**：模型起草回复 "I see you have PostgreSQL running; does your new container need to connect to it?"。发送前， ContextEngine 钩子 `on_response_ready` 触发。它检查回复是否暗示了一个决策（确实：用户会回答 yes 或 no）。写入 `user/conversation_state.json`：`{"pending_decision": "postgres_link", "expires_at": "2026-05-08T18:00"}`。

5. **Post-turn (Cleanup)**：如果用户在 TTL 内未回复， ContextEngine 会清除 `pending_decision`，避免下次对话接着半个话题聊。

结果就是：三轮之后，当你说 "yes, connect it" 时， Agent 不会问 "connect what?"，因为 `conversation_state.json` 就在上下文里。

**这里容易出问题的地方**：内存写入超出了文件大小预算。默认 ContextEngine 限制每个内存文件最大 50 KB。如果项目文件超过这个限制（长周期项目很常见），旧内容会按 FIFO 被淘汰，你就会丢失早期的决策。症状就是 Agent 会问那些你几周前就已经回答过的问题。解决办法是按子项目拆分大文件，或者在 `openclaw.json` 里调大 `max_memory_file_kb`。用 `du -h` 检查 `~/.openclaw/memory/user_<id>/*.md` 的文件大小。

## 端到端消息追踪

咱们来追踪一下当你在 DingTalk 里发送 "@Lobster what time is my next meeting" 时会发生什么。

**T+0ms**： DingTalk Stream 发送 WebSocket 帧到你的频道适配器。 Payload 是 JSON，包含 `conversationId`、`senderId`、`text`。适配器立即 ACK （必须在 500ms 内，否则 DingTalk 会重试）。

**T+5ms**：频道适配器将消息标准化为 `OpenClawMessage(channel="dingtalk", conversation_id="...", user_id="...", text="@Lobster what time is my next meeting", timestamp=...)`。去掉 @-mention。 POST 到 Gateway `:18789/v1/message`。

**T+8ms**： Gateway 通过计算 `hash(conversation_id + timestamp + text)` 进行去重。检查环形缓冲区中最近 100 条消息；如果 hash 存在则丢弃。否则继续。

**T+10ms**： Gateway 查找或创建会话。 Session ID 是 `("dingtalk", conversationId)`。从 `~/.openclaw/sessions/<session_id>.jsonl` 加载最近 20 轮对话。这给了 Agent 对话历史。

**T+12ms**： Router 选择 Agent。默认配置只有一个 Agent (Pi)，所以这是瞬间完成的。在多 Agent 设置中， Router 会运行一个小型分类器 (100ms)。

**T+15ms**： ContextEngine 检索运行。嵌入 "what time is my next meeting"，搜索 `user/*.md` 和 `reference/*.md`。找到 `user/profile.md`（你的日历是 Google）和 `reference/calendar-oauth.md`。注入两者。如果嵌入模型是远程的，这需要 200ms，本地则 <10ms。

**T+215ms**： Agent 循环启动。第一次 LLM 调用。 Prompt 是 system + context + history + user message。模型返回计划：`[{tool: "calendar_list", args: {max_results: 3}}]`。根据提供商不同，这需要 800ms (Qwen-Max) 到 2000ms (GPT-4)。

**T+1015ms**： Gateway 执行 `calendar_list`。这会请求 Google Calendar API，如果需要则等待 OAuth token 刷新，获取事件。耗时 300ms。

**T+1315ms**：工具结果追加到消息历史。第二次 LLM 调用。模型看到事件列表，生成最终回复："Your next meeting is at 2pm: Sprint Planning"。再耗时 600ms。

**T+1915ms**： Agent 循环终止（模型设置 `is_terminal=true`）。 ContextEngine `on_response_ready` 钩子触发，不写入内容（无新事实）。 Gateway 将回复包装在 DingTalk card JSON 结构中。

**T+1920ms**： Gateway POST 卡片到 DingTalk webhook API。 DingTalk 在 50ms 内返回 200。用户感知的总延迟：**1970ms**（不到 2 秒）。

瓶颈永远卡在 LLM 调用（占总时间的 1400ms）和外部 API （日历 300ms）。像 `read` 这样的本地工具增加 <10ms。如果你看到 >5s 的延迟，要么是提供商限流（检查 `gateway.log` 中的重试），要么是工具卡住（检查 turn JSON 中的 `tool_durations`）。

## 调试循环

出问题时，你得看清 Agent 到底在想什么。三个工具：

**1. 详细模式**：`openclaw debug --verbose` 尾随 `gateway.log` 并美化打印每次 LLM 调用。你能看到完整的 Prompt （包括 system、 tools 和 context）、模型的回复（解析后的工具调用或文本）以及 Token 计数。当你发送测试消息时，在第二个终端运行这个。

**2. Gateway 日志级别**：在 `openclaw.json` 中设置 `"log_level": "DEBUG"`。这会添加：
   - 每个工具的输入和输出（截断到 500 字符）
   - ContextEngine 检索分数（哪些内存文件匹配以及原因）
   - 会话加载/保存事件（这样你可以看到历史是否持久化）

DEBUG 级别下日志文件每天会长到几 MB。每周重置或配置轮转。

**3. 逐轮 JSON 转储**：每一轮都追加到 `~/.openclaw/sessions/<session_id>.jsonl`。每一行是一个 JSON 对象：
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

如果 Agent 在六轮前做了些莫名其妙的事，`cat ~/.openclaw/sessions/<session_id>.jsonl | jq 'select(.turn==6)'` 能向你展示当时到底用了什么上下文和工具。这是排查 "为什么它忘了 X" 或 "为什么它调用了错误的工具" 最快的路子。

## 为什么分层很重要

两个实际好处：

1. **你写的是技能，不是 Agent。** Agent 循环是固定的。你的定制发生在 Skill 层（知识）和 Tool 层（动词）。你几乎不需要碰 Gateway。

2. **同一个 Agent 可以服务所有渠道。** 因为循环和渠道是解耦的，你为终端写的生产力技能，在 DingTalk 和 Telegram 上工作方式完全一样。不用移植，不用重写。

下一篇讲讲配置——`openclaw.json`、模型提供商，以及在国内运行这套方案最划算的 Bailian Coding Plan。