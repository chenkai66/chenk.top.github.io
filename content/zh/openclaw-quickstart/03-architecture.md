---
title: "OpenClaw 快速上手（三）：让 Agent 循环跑起来的六个层"
date: 2026-04-05 09:00:00
tags:
  - openclaw
  - 架构
  - agent-loop
  - gateway
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 3
description: "Gateway、Pi Agent、工具、技能、记忆和渠道——每一层做什么、怎么配合、为什么在你写自定义技能时这个分层设计很重要。"
disableNunjucks: true
translationKey: "openclaw-quickstart-3"
---

你可以用 OpenClaw 好几个月都不需要读这篇文章。但是当你第一次要写一个技能、排查一条消息为什么被路由到了错误的 agent、或者搞清楚 agent 为什么突然"失忆"了——你就需要知道每个盒子里装的是什么。

![OpenClaw 快速上手（三）：让 Agent 循环跑起来的六个层 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_1.jpg)

## 六层架构总览

```
        +-----------------------------------------------+
        |                  渠道 Channels                 |
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |                  网关 Gateway                  |
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   路由 Router  +  会话 Sessions  +  Pi Agent   |
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   工具 Tools (26)      技能 Skills (53+)       |
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   记忆 Memory + 上下文引擎 ContextEngine        |
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   LLM 供应商 (通义千问 / DeepSeek / GPT-4)      |
        +-----------------------------------------------+
```

从上到下，一层一层讲。

## 渠道层 (Channels) — 适配器，不是传输层

渠道就是那段把"一条钉钉 Stream 消息"变成"一条标准化 OpenClaw 消息"的代码，反向也一样。每个渠道有自己的怪癖：钉钉走 Stream 协议（底层是 WebSocket），Telegram 用 long-poll 或 webhook，Discord 有自己的 Gateway WebSocket。渠道层的职责就是把这些差异全藏起来。

需要记住的几点：

- 渠道是按实例配置的。你可以跑零个、一个、二十个。
- 消息流向：渠道 -> 网关 -> Agent -> 网关 -> 渠道。渠道不直接和 Agent 通信。
- 每个渠道的速率限制和特殊处理都封装在适配器内部。这就是为什么钉钉回复和 Telegram 回复"手感"不一样。

对于国内用户，钉钉渠道是你最常用的入口。OpenClaw 用的是钉钉的 Stream 模式（不是旧的 HTTP 回调），配置起来其实更简单——不需要公网 IP，不需要配 Nginx 反代，只要有 Client ID 和 Client Secret 就能跑。

**这一层容易出什么问题**：最常见的故障是 WebSocket 断连后没有自动恢复。钉钉 Stream 连接在 300 秒无活动后会断开；如果渠道适配器没有检测到断连并自动重连，消息就会堆积在 broker 那边，你看到的表现是回复延迟从几秒到永远不到。排查方法：去 `gateway.log` 搜 "channel reconnect" 事件，看时间戳间隔。如果间隔超过 5 分钟，说明 keep-alive 机制失效了。

## 网关 (Gateway) — 中枢神经

网关跑在 `:18789`。它接收来自所有渠道的消息，做去重（钉钉有时候会重复投递），分配或恢复会话，然后把消息交给路由。

网关也是唯一和模型供应商通信的组件。所有的工具调用结果、所有的记忆读取、所有的 prompt 拼装都经过它。所以你只需要配一套 API key。

在国内环境跑的话，网关和模型供应商之间走的是内网请求（通义千问 / 百炼），延迟比调海外的 GPT-4 低很多。这也是为什么我在后面的配置篇会推荐百炼编码套餐。

**这一层容易出什么问题**：模型供应商端的限流。如果十个用户同时发消息，网关会序列化 LLM 调用但不会限制入口流量。你会看到供应商返回 HTTP 429，网关用指数退避重试（最多 3 次）。三次都失败的话，用户收到一句"我现在思考有点困难"，这个 turn 会记录到 `gateway_errors.jsonl`。解决办法要么升级供应商额度，要么在 `openclaw.json` 里配 `max_concurrent_llm_calls` 让它和你的配额匹配。

## 路由和会话 (Router & Sessions)

路由决定哪个 agent 来处理这条消息——只有配了多 agent 时才有意义（默认安装只有一个，叫 Pi）。会话是 OpenClaw 用来区分"同一个 agent 上不同对话"的机制。会话 ID 是 `(channel, conversation_id)` 这个二元组。

如果你遇到过"agent 把我两个对话的上下文搞混了"，那就是会话 ID 冲突，几乎肯定是自定义渠道的 bug。

**这一层容易出什么问题**：自定义渠道没有提供稳定的 `conversation_id` 导致会话 ID 碰撞。表现就是 agent 把两个不同聊天的上下文混在一起。钉钉用 webhook 载荷中的 `conversationId`；Telegram 用 `chat.id`。如果你写自定义渠道时用多个字段 hash 来生成 ID，必须确保每种消息类型（文本、图片、回调）里这些字段都存在，否则同一个逻辑会话会生成不同的 ID。诊断方法：`openclaw debug --session <user_id>` 可以 dump 这个用户的所有会话。

## Pi Agent — 智能体循环

![OpenClaw 快速上手（三）：让 Agent 循环跑起来的六个层 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_2.jpg)

这是真正的智能体循环。核心逻辑长这样：

```python
while True:
    plan = LLM(messages, tools=enabled_tools, skills=hot_skills)
    if plan.is_terminal:
        return plan.reply
    for tool_call in plan.tool_calls:
        result = run_tool(tool_call)
        messages.append(result)
```

OpenClaw 在这里做了几个关键设计决策：

- **技能懒加载。** system prompt 里只放 manifest（一行描述）。技能正文只在模型触发该技能时才注入。这样 token 成本保持低位。
- **工具报错返回给模型，不抛异常。** 模型有机会自行恢复。这听起来理所当然，但很多 agent 框架遇到工具报错直接 raise。
- **循环有硬性 turn 上限。** 默认 30。如果 agent 到第 30 轮还在循环，它会停下来说一句"我可能卡住了"，而不是一整晚烧你的 token。

**这一层容易出什么问题**：无限工具调用循环——工具返回"成功"但没有改变状态。最经典的场景是 `web_search` 返回零结果：模型看到 `"success: []"`，觉得需要换个关键词搜，再调一次 `web_search`，又是空列表，反复循环到第 30 轮。循环保护机制会兜底，但你白白烧了 30 次 LLM 调用。根本解法是让工具返回语义化错误——`"success: false, reason: 没有找到相关结果"`——这样模型可以更早放弃。你也可以在 agent 配置里降低 `max_turns`。

## 工具 (Tools) — agent 能做什么

工具是动词。读文件、写文件、跑 shell 命令、抓网页、搜索。默认安装带 26 个工具。每个工具有：

- 名称（`read`、`exec`、`web_search` ...）
- 参数模式（类型化 schema）
- 处理函数（实际执行的代码）
- 权限级别

`exec` 是最危险的——它跑任意 shell 命令。默认每次调用都要确认；你可以在 `openclaw.json` 中把常用的安全命令加入信任列表。

### 内置工具速查表

| 工具 | 功能 | 权限 | 常用配置 |
|------|------|------|----------|
| `read` | 读取文件内容，支持行范围 | `safe` | `max_file_size_mb: 10` |
| `write` | 写入文件（自动创建父目录） | `safe` | `backup_on_overwrite: true` |
| `edit` | 基于正则的原地编辑 | `safe` | `require_confirmation: false` |
| `exec` | 运行 shell 命令，流式输出 | `dangerous` | `trusted_commands: ["git status", "ls"]` |
| `web_search` | 通过配置的搜索引擎搜索（默认 DuckDuckGo） | `safe` | `max_results: 5`, `provider: "bing"` |
| `web_fetch` | 抓取 URL，渲染为 markdown | `safe` | `timeout_sec: 10` |
| `git_status` | `git status --short` 的快捷方式 | `safe` | -- |
| `git_diff` | `git diff` 快捷方式，尊重 .gitignore | `safe` | `context_lines: 3` |
| `calendar_list` | 列出日历事件（需要 OAuth） | `safe` | `max_days_ahead: 7` |
| `send_email` | 通过配置的 SMTP 发邮件 | `requires_confirm` | `from_address: "bot@..."` |

**这一层容易出什么问题**：权限拒绝。某个技能尝试调用用户没有授权的工具。模型收到 "permission denied: exec requires dangerous approval" 这样的错误，通常会把情况告诉用户。但如果技能逻辑写得太死没有容错，agent 会直接卡在半截。你可以在 `openclaw.json` 的 `skill_permissions` 下预授权，比如 `{"obsidian-notes": {"allow_dangerous": false, "allow_web": true}}`。

### 注册自定义工具

三步走：

**第一步：写处理函数。** 创建 `~/.openclaw/tools/my_tool.py`：

```python
from openclaw.tools import Tool, ToolResult

class MyTool(Tool):
    name = "my_tool"
    description = "做一些有用的事情"
    schema = {"arg1": "string", "arg2": "integer"}
    permission = "safe"

    def run(self, arg1: str, arg2: int) -> ToolResult:
        # 你的逻辑
        return ToolResult(success=True, output="完成")
```

**第二步：声明参数模式。** `schema` 字典会变成一个 JSON Schema 片段。OpenClaw 在调用 `run()` 之前会先验参。

**第三步：注册到配置。** 在 `openclaw.json` 的 `tools.custom` 下添加 `{"module": "my_tool", "enabled": true}`。重启网关。

工具注册后对所有 agent 可见。如果你想让它只在特定技能里生效，在技能 manifest 里写 `tools_required: [my_tool]`；网关只有在该技能激活时才会把这个工具加到 prompt 里。

## 技能 (Skills) — 怎么做某件事

技能是知识的封装。一个技能是放在 `~/.openclaw/skills/<name>/SKILL.md` 的 Markdown 文件，加上可选的辅助文件。文件顶部的 manifest 长这样：

```yaml
---
name: obsidian-notes
description: 管理 Obsidian 笔记库
trigger: 当用户要求记笔记、搜索笔记、关联笔记时
tools_required: [read, write, exec]
---
```

正文就是 SOP——指令、模板、示例。agent 启动时加载所有 manifest，模型看到的是每个技能的一行摘要。当模型判断某个技能适用时，网关在下一个 turn 把完整正文展开到 prompt 里。

工具回答的是"我能不能读文件"。技能回答的是"既然我在写会议纪要，正确的模板是什么、放到哪个目录、需要关联哪些文档"。

**这一层容易出什么问题**：触发歧义。多个技能的 `trigger` 子句有重叠时，模型会选一个（通常是字母序靠前的），于是你得到了错误的技能。表现是 agent 用了错误的模板或搜了错误的目录。解决方法是让 trigger 互斥——不要在两个技能里都写"当用户问到笔记时"，而要写"当用户问到*会议*笔记时"和"当用户问到*项目*笔记时"。你也可以在某个技能的 manifest 里加 `priority: high` 来影响选择权重。

## 记忆 + 上下文引擎 (Memory + ContextEngine)

记忆是按用户隔离的、持久化的、有类型的。常见类型：

- `user/profile.md` — 用户偏好
- `project/<name>.md` — 项目状态
- `feedback/*.md` — 你给 agent 的纠正
- `reference/*.md` — agent 需要记住的事实

上下文引擎（ContextEngine，v2026.3.7 新增）决定在下一个 prompt 里注入哪些记忆片段。它按时效性、与当前消息的相关性、以及显式标签来打分。你可以替换引擎——有 `noop`（不注入）、`recency-only`（只看时间）和默认的语义引擎三种。

这一层是让 agent "记得你"的关键。如果你的 agent 总忘事，几乎肯定是上下文引擎没有足够的写入机会——agent 需要被告知要写记忆。

### 五阶段生命周期实例

假设你在一个新会话里说："帮我部署这个 Docker 容器"。上下文引擎做的事情：

**1. 回合前（检索）**：搜索 `user/*.md` 和 `project/*.md`，用"Docker 部署"做向量相似度匹配。找到 `project/infra.md`（里面提到你的 ECS 实例）和 `reference/docker-compose-template.md`。把两者注入到 system prompt 的 `## 相关上下文` 部分。

**2. 规划**：agent 看到上下文，意识到你有一台 ECS 实例在 `120.26.104.90`，决定用 `exec` 工具远程执行 `docker ps`。

**3. 工具执行**：工具返回后，上下文引擎的 `on_tool_result` 钩子触发。它检查结果里有没有新事实（有：三个容器在跑，其中一个是 PostgreSQL）。追加到 `project/infra.md`："最后检查 2026-05-08: postgres:14 运行在 :5432"。

**4. 生成回复**：模型拟出 "我看到你有 PostgreSQL 在跑；你的新容器需要连接它吗？"。发送前，上下文引擎的 `on_response_ready` 钩子触发。它检测到回复暗含一个决策点（用户会回答是或否）。写入 `user/conversation_state.json`：`{"pending_decision": "postgres_link", "expires_at": "2026-05-08T18:00"}`。

**5. 回合后（清理）**：如果用户在 TTL 内没有回复，上下文引擎清除 `pending_decision`，这样下次对话不会从中间状态开始。

效果：三轮之后你说"好，连上它"，agent 不会反问"连什么？"——因为 `conversation_state.json` 还在上下文里。

**这一层容易出什么问题**：记忆文件超过大小预算。默认上下文引擎对每个记忆文件限制 50 KB。如果一个项目文件长期累积超过这个值（长期项目很常见），旧内容按 FIFO 被淘汰，你早期的决策就丢了。表现是 agent 又问你几周前已经回答过的问题。解决办法是按子项目拆分大文件，或者在 `openclaw.json` 里调大 `max_memory_file_kb`。排查时看一眼 `~/.openclaw/memory/user_<id>/*.md` 的文件大小：`du -h`。

## 端到端追踪一条消息

我们完整走一遍：你在钉钉群里发 "@海星 我下个会议几点"。

**T+0ms**：钉钉 Stream 发出一个 WebSocket 帧给你的渠道适配器。载荷是 JSON，包含 `conversationId`、`senderId`、`text`。适配器立即 ACK（钉钉要求 500ms 内 ACK，否则重发）。

**T+5ms**：渠道适配器把消息标准化为 `OpenClawMessage(channel="dingtalk", conversation_id="...", user_id="...", text="我下个会议几点", timestamp=...)`。去掉 @mention。POST 到网关 `:18789/v1/message`。

**T+8ms**：网关去重。计算 `hash(conversation_id + timestamp + text)`，在一个 100 条的环形缓冲区里查重。存在则丢弃，不存在则继续。

**T+10ms**：网关查找或创建会话。会话 ID 是 `("dingtalk", conversationId)`。从 `~/.openclaw/sessions/<session_id>.jsonl` 加载最近 20 轮对话。这给 agent 提供了会话历史。

**T+12ms**：路由选 agent。默认只有一个 agent（Pi），所以这一步瞬间完成。多 agent 配置下路由会跑一个小分类器（约 100ms）。

**T+15ms**：上下文引擎检索。对"我下个会议几点"做 embedding，搜索 `user/*.md` 和 `reference/*.md`。找到 `user/profile.md`（你的日历是 Google Calendar）和 `reference/calendar-oauth.md`。注入两者。远程 embedding 模型约 200ms，本地模型 <10ms。

**T+215ms**：Agent 循环启动。第一次 LLM 调用。Prompt = system + 上下文 + 历史 + 用户消息。模型返回计划：`[{tool: "calendar_list", args: {max_results: 3}}]`。通义千问约 800ms，GPT-4 约 2000ms。

**T+1015ms**：网关执行 `calendar_list`。命中 Google Calendar API，必要时刷新 OAuth token，拉取事件。约 300ms。

**T+1315ms**：工具结果追加到消息历史。第二次 LLM 调用。模型看到事件列表，生成最终回复："你的下个会议是下午 2 点：Sprint Planning"。约 600ms。

**T+1915ms**：Agent 循环终止（模型设置了 `is_terminal=true`）。上下文引擎 `on_response_ready` 钩子触发，没有新事实要写。网关把回复包装成钉钉卡片 JSON 结构。

**T+1920ms**：网关 POST 卡片到钉钉 webhook API。钉钉 50ms 内返回 200。用户感知总延迟：**1970ms**（不到 2 秒）。

瓶颈永远是 LLM 调用（总计 1400ms）和外部 API（日历 300ms）。本地工具如 `read` 只加 <10ms。如果你看到 >5s 的延迟，要么是供应商在限流（检查 `gateway.log` 里的重试记录），要么是某个工具 hang 住了（检查 turn JSON 里的 `tool_durations`）。

## 调试循环

出问题时你需要看到 agent 到底在想什么。三个手段：

**1. Verbose 模式**：`openclaw debug --verbose` 实时 tail `gateway.log`，并把每次 LLM 调用格式化输出。你能看到完整 prompt（包括 system、tools、context）、模型回复（解析后的工具调用或文本）、以及 token 数。开一个终端窗口跑这个，在另一个窗口发测试消息。

**2. 网关日志级别**：在 `openclaw.json` 里设 `"log_level": "DEBUG"`。这会额外记录：
   - 每个工具的输入和输出（截断到 500 字符）
   - 上下文引擎的检索分数（哪些记忆文件命中了、为什么）
   - 会话加载/保存事件（确认历史是否在持久化）

DEBUG 级别每天几 MB 日志。建议每周清一次或配置 rotation。

**3. 逐轮 JSON dump**：每个 turn 追加到 `~/.openclaw/sessions/<session_id>.jsonl`。每行是一个 JSON 对象：

```json
{
  "turn": 5,
  "timestamp": "2026-05-08T14:32:10Z",
  "user_message": "我下个会议几点",
  "context_injected": ["user/profile.md", "reference/calendar-oauth.md"],
  "plan": [{"tool": "calendar_list", "args": {}}],
  "tool_results": [{"success": true, "output": "..."}],
  "final_reply": "你的下个会议是下午 2 点：Sprint Planning",
  "tokens_used": 1523
}
```

如果 agent 六轮前做了一件莫名其妙的事，`cat ~/.openclaw/sessions/<session_id>.jsonl | jq 'select(.turn==6)'` 能告诉你当时的上下文和工具调用情况。这是排查"它为什么忘了 X"和"它为什么调了错误的工具"最快的方法。

## 这个分层设计为什么重要

两个实际意义：

**1. 你写的是技能，不是 agent。** 智能体循环是固定的。你的定制发生在技能层（知识）和工具层（动词）。你几乎永远不需要碰网关代码。

**2. 同一个 agent 服务所有渠道。** 因为循环和渠道解耦了，你为命令行写的生产力技能在钉钉、Telegram、Discord 上跑起来效果一模一样。不用移植，不用改写。

这意味着如果你在公司里主要用钉钉，偶尔在手机上用 Telegram，晚上回家习惯在终端里直接敲——三个渠道共享同一个 agent、同一套技能、同一份记忆。你对 agent 做的任何训练（写 feedback、调技能）在所有渠道立即生效。

---

下一篇讲配置——`openclaw.json` 的详细字段、模型供应商选择、以及百炼编码套餐为什么是国内跑 OpenClaw 最划算的方案。
