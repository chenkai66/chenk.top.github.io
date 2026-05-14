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
尽管你可能在使用 OpenClaw 的前几个月都不需要阅读这篇文章，但当你第一次编写 Skill、调试消息路由异常，或疑惑 Agent 为何突然“失忆”时，就必须厘清各模块的职责。

![OpenClaw 快速入门（3）：使代理循环工作的六层结构 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_1.png)


---

## 六层架构

![OpenClaw 代理的六层结构：通道、网关、路由+会话+Pi 代理、工具+技能、内存+上下文引擎、LLM 提供商](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/fig1_six_layers.png)

接下来按自上而下的顺序梳理。

## Channels —— 是适配器，不是传输层

Channel 的代码负责双向转换：将钉钉 Stream 消息转为标准化的 OpenClaw 消息，反之亦然。不同平台实现方式各异——钉钉通过 WebSocket 接收 Stream 事件，Telegram 使用轮询或 Webhook，Discord 则接入其专属 Gateway WebSocket。Channel 层将这些差异完全屏蔽。

你需要记住以下几点：

- Channel 按实例配置，可以部署零个、一个或多个。
- 消息严格遵循 Channel → Gateway → Agent → Gateway → Channel 的流向，Channel 与 Agent 从不直连。
- 各平台的速率限制和特有行为均由对应的 Channel 适配器处理，因此钉钉与 Telegram 的回复体验天然不同。

**这里容易出什么问题**：最常见的问题是 WebSocket 连接断开后未自动重连。钉钉 Stream 连接在空闲 300 秒后会断开；如果 Channel 适配器未能检测并重连，消息将在 Broker 中积压，导致交付延迟从几秒延长到永久失败。此时应检查 `gateway.log` 中的 "channel reconnect" 事件及其时间戳间隔。若间隔超过 5 分钟，说明你的 keep-alive 机制未生效。

## Gateway —— 中枢神经系统

Gateway 运行在 `:18789` 端口，负责接收来自任意 Channel 的消息，进行去重（钉钉偶尔会重复投递）、分配或恢复 Session，并将消息交给 Router。

Gateway 也是唯一与模型提供商通信的组件——所有工具调用结果、内存读取、Prompt 组装都经由它处理，因此你只需配置一套 API Key。

**这里容易出什么问题**：模型提供商的速率限制被耗尽。当十个用户同时发送消息时，Gateway 会对 LLM 调用串行化，但不会限制入口流量。你会看到提供商返回 HTTP 429 错误，Gateway 会以指数退避策略重试（最多 3 次）。若三次均失败，用户将收到“我现在有点思考困难”的提示，且该轮对话会被记录到 `gateway_errors.jsonl`。解决方法要么升级提供商套餐，要么在 `openclaw.json` 中设置 `max_concurrent_llm_calls`，使其匹配你的配额。

## Router 与 Sessions

Router 决定由哪个 Agent 处理消息——仅在配置了多个 Agent 时才起作用（默认安装只有一个名为 Pi 的 Agent）。Session 则用于区分不同渠道的对话（例如微信 vs Telegram），即使它们共用同一个 Agent。Session ID 由 `(channel, conversation_id)` 唯一确定。

如果你曾遇到“Agent 混淆了两个对话”的情况，那几乎肯定是 Session ID 冲突所致，通常源于自定义 Channel 的实现缺陷。

**这里容易出什么问题**：自定义 Channel 未提供稳定的 `conversation_id`，导致 Session ID 冲突。典型症状是 Agent 将两个独立对话的上下文混在一起。钉钉使用 Webhook payload 中的 `conversationId`，Telegram 使用 `chat.id`。如果你在编写自定义 Channel 并通过哈希多个字段生成 ID，请确保*每种*消息类型（文本、图片、回调）都包含这些字段，否则同一个逻辑对话可能生成不同的 ID。诊断方法是运行 `openclaw debug --session <user_id>`，它会 dump 出该用户的所有 Session。

## Pi Agent —— 核心执行循环

![OpenClaw 快速入门 (3)：使代理循环工作的六个层级 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/03-architecture/illustration_2.png)

这才是 Agent 的核心执行循环。代码如下：

```text
while True:
    plan = LLM(messages, tools=enabled_tools, skills=hot_skills)
    if plan.is_terminal:
        return plan.reply
    for tool_call in plan.tool_calls:
        result = run_tool(tool_call)
        messages.append(result)
```

OpenClaw 在此处做了几个关键设计选择：

- **Skills 懒加载**：系统 Prompt 仅包含 manifest，只有当模型触发某个 Skill 时，其主体内容才会被动态加载，从而显著降低 Token 成本。
- **工具错误返回给模型，而非抛出异常**：模型有机会自行恢复。这看似理所当然，但许多 Agent 框架直接抛出异常导致流程中断。
- **循环设有硬性 turn 限制**：默认为 30 轮。若 Agent 在第 30 轮仍未结束，将主动终止并返回“我觉得我卡住了”的提示，避免整夜消耗 Token 预算。

**这里容易出什么问题**：工具返回“成功”但未实际改变状态，从而引发无限调用循环。经典案例是 `web_search` 返回空结果：模型看到 `"success: []"`，决定优化查询词后再次调用，又得到空列表，如此反复直至第 30 轮。虽然循环保护机制会终止流程，但你已白白浪费 30 次 LLM 调用。解决方法是让工具返回语义化错误，例如 `"success: false, reason: no results for query"`，使模型能尽早放弃。若频繁出现此模式，也可在 Agent 配置中调低 `max_turns`。

## Tools —— Agent 能*做什么*

Tools 定义了 Agent 可执行的操作，例如读取文件、写入文件、执行 Shell 命令、抓取 URL、网页搜索等。默认安装包含 26 个内置 Tool。每个 Tool 都包含：

- 名称（如 `read`、`exec`、`web_search`）
- Schema（带类型的参数定义）
- Handler（实际执行的代码）
- 权限级别

其中 `exec` 最危险——它可执行任意 Shell 命令。默认每次调用都需要用户确认；你可以在 `openclaw.json` 中标记受信任的命令模式。

### 内置 Tools 参考

| Tool | 功能 | 权限 | 常见配置覆盖 |
|------|-------------|-----------|----------------------|
| `read` | 读取文件内容，支持指定行范围 | `safe` | `max_file_size_mb: 10` |
| `write` | 写入文件（自动创建父目录） | `safe` | `backup_on_overwrite: true` |
| `edit` | 基于正则表达式的原地编辑 | `safe` | `require_confirmation: false` |
| `exec` | 执行 Shell 命令，流式输出结果 | `dangerous` | `trusted_commands: ["git status", "ls"]` |
| `web_search` | 通过配置的搜索引擎搜索（默认 DuckDuckGo） | `safe` | `max_results: 5`, `provider: "bing"` |
| `web_fetch` | 抓取 URL 并渲染为 Markdown | `safe` | `timeout_sec: 10`, `user_agent: "..."` |
| `git_status` | `git status --short` 的快捷方式 | `safe` | — |
| `git_diff` | `git diff` 的快捷方式，尊重 .gitignore | `safe` | `context_lines: 3` |
| `calendar_list` | 列出日历事件（需 OAuth 授权） | `safe` | `max_days_ahead: 7` |
| `send_email` | 通过配置的 SMTP 发送邮件 | `requires_confirm` | `from_address: "bot@..."` |

**这里容易出什么问题**：Skill 尝试使用用户未授权的 Tool，导致权限拒绝。模型会收到类似 "permission denied: exec requires dangerous approval" 的错误，并通常将其展示给用户。但如果 Skill 是 tightly scripted（高度脚本化）且未处理该错误，Agent 就会在任务中途静默停止。你可以在 `openclaw.json` 的 `skill_permissions` 字段中为特定 Skill 预授权工具，例如 `{"obsidian-notes": {"allow_dangerous": false, "allow_web": true}}`。

### 自定义 Tools

注册新 Tool 的步骤如下：

1. **编写 Handler**：创建 `~/.openclaw/tools/my_tool.py`：
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

2. **声明 Schema**：`schema` 字典会被转换为 JSON Schema 片段。OpenClaw 会在调用 `run()` 前验证参数。

3. **加入配置**：在 `openclaw.json` 的 `tools.custom` 下添加 `{"module": "my_tool", "enabled": true}`，然后重启 Gateway。

此后，该 Tool 对所有 Agent 可见。若希望仅特定 Skill 使用它，可在 Skill manifest 中指定 `tools_required: [my_tool]`；Gateway 仅在该 Skill 激活时才将其加载到 Prompt 中。

## Skills —— *怎么做*

Skills 是知识的载体。每个 Skill 由 `~/.openclaw/skills/<name>/SKILL.md` 文件及可选辅助文件组成。文件顶部的 manifest 如下所示：

```yaml
---
name: obsidian-notes
description: Manage Obsidian vault notes
trigger: when user asks to take notes, search notes, link notes
tools_required: [read, write, exec]
---
```

主体内容是标准操作流程（SOP）——包含指令、模板和示例。Agent 启动时加载所有 manifest，因此模型能看到每个 Skill 的一行摘要。当模型判定某 Skill 适用时，Gateway 会在下一轮对话中将其完整内容注入 Prompt。

Skills 是将 LLM 转化为特定任务可靠执行者的关键。Tools 回答“能否读取文件”这类能力问题，而 Skills 则回答“撰写会议纪要时，应使用何种模板、保存到哪里、需关联哪些内容”等流程性问题。

**这里容易出什么问题**：多个 Skill 的 `trigger` 条件重叠，导致触发歧义。模型通常会选择字典序靠前的那个，从而调用错误的 Skill。症状表现为 Agent 使用了错误的模板或搜索了错误的目录。解决方法是让触发条件互斥——例如将两个 Skill 的触发条件分别设为“当用户询问*会议*笔记时”和“当用户询问*项目*笔记时”，而非笼统的“当用户询问笔记时”。你也可以在某个 Skill 的 manifest 中设置 `priority: high` 来提高其被选中的概率。

## 内存 + 上下文引擎

内存按用户隔离、持久化且带类型。常见类型包括：

- `user/profile.md` — 用户偏好
- `project/<name>.md` — 项目状态
- `feedback/*.md` — 用户对 Agent 的修正反馈
- `reference/*.md` — Agent 应记住的事实

ContextEngine 是 v2026.3.7 新增的组件，负责决定哪些内存片段应注入下一轮 Prompt。它根据近期性、与当前消息的相关性以及显式标签进行打分。你可以切换引擎——目前提供 `noop`（无操作）、`recency-only`（仅按时间）和默认的语义搜索模式。

正是这一层让 Agent 显得“记得你”。若 Agent 缺乏上下文连贯性，通常是因为 ContextEngine 缺少写入机会——你必须明确指示 Agent 记录信息。

### 五阶段生命周期实战

假设你在新会话中说：“help me deploy this Docker container”。ContextEngine 会按以下流程处理：

1. **Pre-turn (Retrieval)**：在 `user/*.md` 和 `project/*.md` 中搜索与 “Docker deploy” 语义相近的片段，找到 `project/infra.md`（提及你的 ECS 实例）和 `reference/docker-compose-template.md`，并将二者注入系统 Prompt 的 `## Relevant Context` 区块。

2. **Planning**：Agent 结合上下文，识别出你有一台 IP 为 `120.26.104.90` 的 ECS 实例，决定使用 `exec` 工具远程执行 `docker ps`。

3. **Tool execution**：工具返回结果后，ContextEngine 的 `on_tool_result` 钩子触发，检测到新事实（三个容器运行中，含 PostgreSQL），于是向 `project/infra.md` 追加记录：“Last checked 2026-05-08: postgres:14 running on :5432”。

4. **Response generation**：模型草拟回复：“I see you have PostgreSQL running; does your new container need to connect to it?”。发送前，`on_response_ready` 钩子触发，识别出回复隐含一个待决事项（用户将回答 yes 或 no），于是写入 `user/conversation_state.json`：`{"pending_decision": "postgres_link", "expires_at": "2026-05-08T18:00"}`。

5. **Post-turn (Cleanup)**：若用户在 TTL 内未回复，ContextEngine 会清除 `pending_decision`，避免下次对话从中断处继续。

结果是：三轮之后，当你回复 “yes, connect it” 时，Agent 不会问 “connect what?”，因为 `conversation_state.json` 已在上下文中。

**这里容易出问题的地方**：内存写入超出文件大小限制。默认 ContextEngine 将每个内存文件上限设为 50 KB。长期项目中，文件极易超限，旧内容按 FIFO 被淘汰，导致早期决策丢失。症状是 Agent 重复询问你几周前已回答过的问题。解决方法是按子项目拆分大文件，或在 `openclaw.json` 中调高 `max_memory_file_kb`。可用 `du -h` 检查 `~/.openclaw/memory/user_<id>/*.md` 的实际大小。

## 端到端消息追踪

我们来追踪在钉钉中发送 “@Lobster what time is my next meeting” 时的完整流程：

**T+0ms**：钉钉 Stream 通过 WebSocket 发送消息帧到 Channel 适配器，Payload 为 JSON，包含 `conversationId`、`senderId`、`text`。适配器立即 ACK（必须在 500ms 内完成，否则钉钉会重试）。

**T+5ms**：Channel 适配器将消息标准化为 `OpenClawMessage(channel="dingtalk", conversation_id="...", user_id="...", text="@Lobster what time is my next meeting", timestamp=...)`，移除 @ 提及，并 POST 到 Gateway 的 `:18789/v1/message`。

**T+8ms**：Gateway 通过计算 `hash(conversation_id + timestamp + text)` 去重，检查最近 100 条消息的环形缓冲区；若哈希已存在则丢弃，否则继续。

**T+10ms**：Gateway 查找或创建 Session，ID 为 `("dingtalk", conversationId)`，并从 `~/.openclaw/sessions/<session_id>.jsonl` 加载最近 20 轮对话历史。

**T+12ms**：Router 选择 Agent。默认仅 Pi 一个 Agent，因此瞬间完成；多 Agent 场景下会运行一个小型分类器（约 100ms）。

**T+15ms**：ContextEngine 执行检索，对 “what time is my next meeting” 生成嵌入向量，在 `user/*.md` 和 `reference/*.md` 中搜索，找到 `user/profile.md`（日历为 Google）和 `reference/calendar-oauth.md` 并注入。若嵌入模型为远程服务，耗时约 200ms；本地则低于 10ms。

**T+215ms**：Agent 循环启动，首次 LLM 调用。Prompt = system + context + history + user message。模型返回计划：`[{tool: "calendar_list", args: {max_results: 3}}]`。耗时取决于模型——Qwen-Max 约 800ms，GPT-4 约 2000ms。

**T+1015ms**：Gateway 执行 `calendar_list`，调用 Google Calendar API（必要时刷新 OAuth Token），获取事件，耗时 300ms。

**T+1315ms**：工具结果追加至历史，第二次 LLM 调用。模型生成最终回复：“Your next meeting is at 2pm: Sprint Planning”，耗时 600ms。

**T+1915ms**：Agent 循环终止（模型设置 `is_terminal=true`）。ContextEngine 的 `on_response_ready` 钩子触发，因无新事实，未写入内存。Gateway 将回复封装为钉钉卡片 JSON。

**T+1920ms**：Gateway POST 卡片至钉钉 Webhook API，钉钉在 50ms 内返回 200。用户感知总延迟：**1970ms**（不到 2 秒）。

瓶颈始终在 LLM 调用（占 1400ms）和外部 API（日历 300ms）。本地工具如 `read` 仅增加 <10ms 开销。若延迟超过 5 秒，要么是提供商限流（检查 `gateway.log` 中的重试记录），要么是工具卡住（检查 turn JSON 中的 `tool_durations`）。

## 调试循环

当出现问题时，你需要看清 Agent 的完整思考过程。以下是三个实用工具：

**1. 详细模式**：运行 `openclaw debug --verbose`，实时跟踪 `gateway.log` 并美化打印每次 LLM 调用。你能看到完整 Prompt（含 system、tools、context）、模型回复（解析后的工具调用或文本）及 Token 消耗。建议在另一终端运行此命令，同时发送测试消息。

**2. Gateway 日志级别**：在 `openclaw.json` 中设置 `"log_level": "DEBUG"`，将额外记录：
   - 每个工具的输入输出（截断至 500 字符）
   - ContextEngine 检索评分（哪些内存文件匹配及原因）
   - Session 加载/保存事件（便于确认历史是否持久化）

DEBUG 级别下日志每日可达数 MB，建议每周清理或配置轮转策略。

**3. 逐轮 JSON 转储**：每轮对话均追加至 `~/.openclaw/sessions/<session_id>.jsonl`，每行为一个 JSON 对象：
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

若 Agent 在六轮前行为异常，运行 `cat ~/.openclaw/sessions/<session_id>.jsonl | jq 'select(.turn==6)'` 即可精确还原当时的上下文与工具调用。这是诊断“为何遗忘 X”或“为何调错工具”最快的方法。

## 为什么分层很重要

这种架构带来两个实际优势：

1. **你编写的是技能，而非 Agent**：Agent 循环是固定的，你的定制工作集中在 Skill 层（知识）和 Tool 层（能力）。几乎无需修改 Gateway。

2. **同一 Agent 可服务所有渠道**：由于循环与 Channel 解耦，你在终端编写的生产力 Skill，在钉钉和 Telegram 上表现完全一致——无需移植，也无需重写。

下一篇将介绍配置——`openclaw.json`、模型提供商，以及在国内运行这套方案最划算的 Bailian Coding Plan。
