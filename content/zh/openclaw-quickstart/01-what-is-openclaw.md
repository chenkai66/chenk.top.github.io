---
title: "OpenClaw 快速上手（一）：这玩意到底是什么"
date: 2026-04-03 09:00:00
tags:
  - openclaw
  - ai-agent
  - 自托管
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 1
description: "OpenClaw 是一个把 20 多个聊天平台桥接到统一 Agent 运行时的自托管 AI Agent 网关。这一篇先把它讲清楚——它是什么、跟 ChatGPT/Claude 的差别在哪、它真正解决了哪些值得占用你硬盘的问题。"
disableNunjucks: true
translationKey: "openclaw-quickstart-1"
---
总有人问我：“OpenClaw 不就是给 LLM 套了个壳吗？”答案是否定的，而且原因很重要，我觉得有必要在开始 QuickStart 之前先写清楚。

这是六篇系列文章的第一篇。等你全部看完，你的机器上应该已经跑起了一个 OpenClaw，连上了模型，接入了至少一个聊天频道，并且能完成一些重启后依然有效的实际任务。
![OpenClaw 快速上手（一）：这玩意到底是什么 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_1.jpg)

## 一句话定义

OpenClaw 是一个自托管的 AI 智能体平台。只需安装一个 Node 二进制文件，再配上一个或多个 LLM 服务商的 API Key，它就能启动一个长期运行的 Gateway 进程。这个进程会监听你接入的聊天平台消息——钉钉、微信（通过 WorkBuddy）、Telegram、Discord、Slack、飞书，随便选。Gateway 会将消息传递到一个包含工具、技能、记忆和定时任务调度的智能体循环中。整个项目采用 MIT 开源协议，数据完全存储在你的设备上，而且可以随时更换模型，无需重写提示词。

如果你用过 Claude Code 或 Cursor，这个智能体循环会让你觉得熟悉。不同的是，那些产品掌控了交互界面——你得在它们的 UI 里与它们对话。而 OpenClaw 则反过来：界面是你已经在用的任何聊天工具，智能体主动来找你。
## 它在生态中的位置

```
                托管                自托管
            +-----------------+-----------------+
   厂商     | ChatGPT, Claude | （少见）        |
   界面     | （Anthropic UI）|                 |
            +-----------------+-----------------+
   我的     | 各类 SaaS 机器人| OpenClaw        |
   界面     | （如 Glean、Slack AI）|           |
            +-----------------+-----------------+
```

右下角这一格虽然小，但正是我想要的。我希望我的工具能做到：

- 运行在我自己的设备上
- 读取我的文件、笔记和邮件
- 把消息发到我平时用来提醒自己的那个 Telegram 聊天里
- 不需要按月订阅付费

2026 年，我发现 OpenClaw 是目前把这一格做得最完善的方案。
## 它真正解决的三件事

![OpenClaw 快速上手（一）：这玩意到底是什么 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_2.jpg)

我得说清楚，有些问题靠“套壳”是解决不了的，但 OpenClaw 可以。

**1. 渠道适配器是真刀真枪的工程活。** 钉钉的 Stream 协议、微信的各种限制、Discord 的速率限制，每个平台都有自己的坑。OpenClaw 提供了 20 多个渠道适配器，每个都针对平台特性做了专门处理。自己写一个适配器可能只需要一个周末，但写五个就得花一个月。

**2. Skills 不仅仅是系统 Prompt。** 一个 Skill 包括一份 Markdown 文件、可选的辅助脚本，以及一个 Manifest。Agent 启动时只会加载 Manifest，技能的具体内容只有在匹配到任务时才会按需加载。这听起来不酷，但它决定了你是用 4k 的上下文窗口塞满系统 Prompt，还是用一个几乎不增加单轮成本的 40k 知识库。

**3. 记忆能跨会话保存。** OpenClaw 提供了一套带类型的记忆存储（用户画像、项目状态、参考资料、反馈），还有一个可索引的 `MEMORY.md`。三天后换到另一个聊天继续同一个话题，Agent 不会从头开始。

如果你对这三件事完全无感，那可以直接用 Claude Code，不用再往下看了。但如果有任何一点戳中你，接下来的内容绝对值得一看。
## 最终你会得到什么

读完第六篇时，你将拥有以下内容：

- OpenClaw 已安装到本地，Gateway 运行在 `:18789`
- 配置好的模型提供商（百炼 Coding Plan、Claude 直连或 DashScope 免费版——任你选择）
- 终端中可以正常使用的 TUI
- 一个真正接通的聊天渠道——本系列选用 Telegram，因为它的配置最简洁
- 加载完成的两个自定义 Skill
- 一个用于浏览器自动化的 MCP 服务器

这些已经足够支持你实现官方文档中的任意案例（第二大脑、每日简报、运维自动化、内容流水线）。最后，我会为你指明具体方向。
## 本系列不会涉及的内容

- 多 Agent 路由——官方文档写得更清楚  
- 基础之外的生产环境加固  
- WeChat WorkBuddy 的详细设置（这是腾讯的产品，注册流程比较复杂）  

下一篇开始安装。如果你的 Node 是最新版本，大概十分钟就能搞定；如果不是，可能需要二十分钟。
