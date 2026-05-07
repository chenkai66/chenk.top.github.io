---
title: "OpenClaw 快速上手（三）：让 Agent 循环跑起来的六层架构"
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
description: "Gateway、Pi Agent、Tools、Skills、Memory、Channels——每一层在做什么、它们怎么拼在一起、以及为什么这种分层在你开始写 Skill 时会变得很重要。"
disableNunjucks: true
translationKey: "openclaw-quickstart-3"
---
用 OpenClaw 几个月，完全可以不看这篇文章。但第一次写 Skill、调试消息路由错误、或者搞不明白 Agent 为什么忘记某些内容时，我肯定会想弄清楚每个盒子的具体作用。
![OpenClaw 快速上手（三）：让 Agent 循环跑起来的六层架构 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_1.jpg)

## 六层结构

```
        +-----------------------------------------------+
        |                  Channels                     |   ← 钉钉、Telegram 等
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |                  Gateway                      |   ← :18789，统一消息格式
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Router  +  Sessions   +  Pi Agent (loop)    |   ← 决定由哪个 agent 处理
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Tools (26)        Skills (53+ 内置)         |   ← 能力范围 + 实现方式
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   Memory + ContextEngine                      |   ← 维护持久化上下文
        +-----------------------------------------------+
                          |
        +-----------------------------------------------+
        |   LLM Provider                                |   ← DashScope、Anthropic 等
        +-----------------------------------------------+
```

我从上到下依次介绍。
## Channels——适配器，不是传输层

Channel 的作用是把“一条钉钉 Stream 消息”转换成“标准化的 OpenClaw 消息”，再反向转换回去。每个 Channel 都有自己的特点：钉钉通过 WebSocket 推送 Stream 事件，Telegram 使用轮询或 webhook，Discord 则依赖自己的 Gateway WebSocket。Channel 层把这些细节全都屏蔽掉了。

需要记住的重点：

- Channel 按实例配置。可以运行 0 个、1 个，也可以运行 20 个。
- 消息的流转路径是 Channel → Gateway → Agent → Gateway → Channel。Channel 不会直接和 Agent 通信。
- 每个 Channel 的限流规则和特殊行为都封装在适配器里。这就是为什么钉钉的回复体验和 Telegram 不一样。
## Gateway——中枢神经系统

Gateway 运行在 `:18789` 端口。它接收来自所有渠道的消息，进行去重处理（钉钉有时会重复投递），分配或恢复 Session，然后将消息交给 Router。

Gateway 是唯一与模型 Provider 通信的组件。无论是工具结果、记忆读写，还是 Prompt 的组装，都必须经过它。这就是为什么我只需要配置一套 API Key 就够了。
## Router 与 Sessions

Router 的作用是决定*哪个* Agent 负责处理消息。这在配置了多个 Agent 时才有意义（默认安装只有一个，叫 Pi）。Sessions 的作用是让 OpenClaw 能区分不同渠道的对话，比如微信和 Telegram 的对话，即使它们都由同一个 Agent 处理。Session ID 的格式是 `(channel, conversation_id)`。

如果遇到过“Agent 把两个对话搞混了”的情况，那通常是 Session ID 冲突导致的，基本可以确定是自定义渠道的 Bug。
## Pi Agent——循环主体

![OpenClaw 快速上手（三）：让 Agent 循环跑起来的六层架构 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/03-architecture/illustration_2.jpg)

这才是真正的 Agent 循环，代码逻辑如下：

```
while True:
    plan = LLM(messages, tools=enabled_tools, skills=hot_skills)
    if plan.is_terminal:
        return plan.reply
    for tool_call in plan.tool_calls:
        result = run_tool(tool_call)
        messages.append(result)
```

OpenClaw 在这里做了几个很有趣的设计：

- **Skills 按需加载。** 系统 Prompt 里只放 Manifest，正文部分等到模型触发某个 Skill 时才加载。这样能有效降低 Token 消耗。
- **工具错误直接返回给模型，而不是中断流程。** 模型有机会自行修复问题。这听起来很简单，但很多 Agent 框架遇到错误就直接抛异常了。
- **循环有严格的轮数限制。** 默认上限是 30 轮。如果到了第 30 轮还在运行，循环会自动停止，并返回一句“我好像卡住了”，避免浪费你的 Token 预算。
## Tools——Agent 能做什么

Tools 就是动词。读文件、写文件、执行 shell 命令、抓取 URL、搜索网页，这些都是工具。默认安装自带 26 个工具。每个工具包含以下内容：

- 名称（`read`, `exec`, `web_search` 等）
- 参数模式（带类型的参数）
- 处理函数（实际运行的代码）
- 权限级别

`exec` 是最危险的一个。它可以执行任意 shell 命令。默认情况下，每次调用都需要确认。如果某些模式可信，我可以在 `openclaw.json` 里标记出来。
## Skills——*如何*完成任务

Skills 是一种知识名词。每个 Skill 对应一个 Markdown 文件，路径是 `~/.openclaw/skills/<name>/SKILL.md`，还可以带一些辅助文件。文件开头的 Manifest 部分如下：

```yaml
---
name: obsidian-notes
description: 管理 Obsidian 笔记
trigger: 当用户要求记笔记、搜笔记、关联笔记时
tools_required: [read, write, exec]
---
```

正文部分是 SOP，包含操作步骤、模板和示例。Agent 在启动时会加载 Manifest，这样模型就能看到每个 Skill 的一句话简介。当模型判断某个 Skill 适用时，Gateway 会把正文展开，插入到下一轮 Prompt 中。

Skills 的作用是让 LLM 成为你具体任务上的可靠帮手。Tools 解决的是“我能不能读取文件”的问题，而 Skills 解决的是“我要写会议笔记时，该用什么模板、存到哪里、需要链接哪些内容”的问题。
## Memory + ContextEngine

Memory 是按用户区分的，持久存储，并且有明确类型。常见的类型包括：

- `user/profile.md` —— 用户偏好
- `project/<name>.md` —— 项目状态
- `feedback/*.md` —— 我给 Agent 的纠正反馈
- `reference/*.md` —— 希望 Agent 记住的事实

ContextEngine 是 2026.3.7 版新增的功能，负责决定哪些记忆片段应该包含在下一次 Prompt 中。它根据时效性、与当前消息的相关性以及显式标签来打分。引擎可以更换，目前有三种选择：`noop`、`recency-only` 和默认的语义引擎。

这一层让 Agent 显得像是认识我。如果我的 Agent 没有这种感觉，几乎都是因为 ContextEngine 没有足够的写入机会——我需要主动引导 Agent 去写入 Memory。
## 为什么分层设计很重要

分层设计带来了两个实际好处：

1. **我只用写技能，不用管智能体。** 智能体的主循环是固定的，我的工作集中在 Skill 层（知识）和 Tool 层（动作）。Gateway 基本碰不到。

2. **一个智能体可以适配所有渠道。** 主循环和渠道是解耦的，我为终端开发的生产力技能，直接搬到钉钉和 Telegram 上也能用，完全不用改。

接下来聊聊配置——`openclaw.json`、模型提供商，以及在国内运行成本最低的百炼 Coding Plan。
