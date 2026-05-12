---
title: "OpenClaw 指南（一）：这到底是什么东西"
date: 2026-04-08 09:00:00
tags:
  - openclaw
  - ai-agent
  - self-hosted
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 1
description: "OpenClaw 是什么、跟 ChatGPT/Claude 有什么区别、解决哪些问题。"
disableNunjucks: true
translationKey: "openclaw-quickstart-1"
---
总有人问我："OpenClaw 不就是套了个 LLM 的壳吗？" 短回答是：不是。这件事值得专门写一篇文章讲清楚，之后再进入 QuickStart。

这是六篇系列文章的第一篇。读完这一系列，你应该能在自己的机器上跑起一个能用的 OpenClaw，它能对接模型，监听至少一个聊天频道，并且干点正事——哪怕重启服务器也能活下来。

![OpenClaw QuickStart (1): What This Thing Actually Is — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_1.png)

## 前置准备和预期

动手之前，我先说明一下前提条件：

**技术底子。** 你需要熟悉终端操作、能修改配置文件，并能看懂 TypeScript 或 JavaScript 的错误堆栈。本系列无需编写代码，但当服务无法启动时，你需要能通过错误日志判断问题根源——是网络超时，还是语法错误。

**Node.js 环境。** OpenClaw 跑在 Node 18 或更高版本上。如果你尚未安装 Node，或版本低于 18，下一篇将手把手指导安装。现在你只需要知道，`node --version` 至少得显示 `v18.0.0`。

**LLM API Key。** 你手头得有一个能用的大模型 API 密钥。系列文章里我会演示三种方案：阿里云百炼 Coding Plan （学生和认证开发者免费）、 Claude 直连（付费，需要 Anthropic API 权限）、以及 DashScope 免费层（额度有限但不需要信用卡）。选适合你的就行。如果现在还没有，第二篇文章我会告诉你去哪注册。

**聊天平台账号。** 本系列以 Telegram 为例，因其接入最简单：通过 BotFather 创建机器人、获取 Token 即可完成配置。你现在还不需要有 Telegram 账号，但到第四篇的时候最好有一个。如果你更想用钉钉、微信或 Discord，概念是通用的，只是我不会手把手带你走它们的注册流程。

**为什么这些很重要。** OpenClaw 不是一个只需填写表单即可使用的 SaaS 服务。它是你自己跑的基础设施。如果想到要 SSH 连 VPS 或者编辑 `.env` 文件你就头大，那 hosted 方案比如 ChatGPT Plus 或 Claude Pro 可能更适合你。这没高低之分——这些同样是优秀的产品。但如果你来这里是因为想把 agent 跑在自己的基础设施上，读本地文件，且不向外部打电话汇报，那你来对地方了。

## 它到底是什么

OpenClaw 是一个自托管的 AI agent 平台。你装一个 Node 二进制文件，给它配上一个或多个 LLM 提供商的 API Key，它就会跑起一个长驻的网关进程，监听你接入的任何聊天平台——钉钉、微信（通过 WorkBuddy）、 Telegram、 Discord、 Slack、飞书，随便选。网关把这些消息路由进一个包含工具、技能、记忆和 cron 调度器的 agent 循环。整个项目是 MIT 协议开源，数据存在你机器上，而且你可以随时换模型，不用重写 prompt。

如果你用过 Claude Code 或 Cursor，这个 agent 循环你会觉得很熟悉。区别在于，那些产品将交互界面内置于自身 UI 中——你需在它们的界面上与 agent 对话。 OpenClaw 把这个反转了。交互界面是你日常使用的即时通讯工具，而 agent 主动与你交互。

**架构概览。** 当消息从任何渠道进来，网关把它反序列化成标准化的对话轮次。这个轮次传给 agent core，由它决定是调用技能、调用工具、查记忆，还是直接生成回复。 agent core 不依赖特定模型——它通过统一接口跟 LLM 提供商对话，所以从 Claude 换到 Qwen 再到 GPT-4 只是改配置，不是重构代码。响应流回渠道适配器，适配器把它序列化成该平台需要的格式——Telegram 用 Markdown，钉钉用交互卡片，短信用纯文本。

**Agent 循环。** 在 core 内部，每一轮对话都走一个四阶段循环： Memory （加载该用户和对话的上下文）、 Planning （根据任务和可用工具决定下一步做什么）、 Tool execution （执行文件系统操作、 web 搜索、 API 调用或 MCP 服务器）、 Reflection （评估任务是完成了还是需要再来一轮）。这不是什么新架构—— powering Cursor 的 agent mode 和 Claude Code 的自主工作流也是这套循环。区别在于 OpenClaw 把每个阶段都暴露成了 hook，你可以自定义或观察。

**渠道适配器。** 每个聊天平台都有自己的适配器模块。适配器知道怎么认证、维持连接、解析 传入事件，以及序列化 传出消息。不同平台采用不同的通信机制： Slack 和 Discord 使用 Webhook， Telegram 使用长轮询，钉钉则依赖持久化的 WebSocket 连接。适配器将这些细节全部封装。对 agent core 而言，它只需接收和发送结构化的 JSON 数据。这种分离为什么加一个新渠道通常只要一两天而不是几周——你实现一个接口，剩下的自动就好使。

**技能系统。** 技能是 agent 的词汇表。一个技能是一个 Markdown 文件，描述一个任务，可选的辅助脚本，还有一个 manifest 声明什么时候该把这个技能提供给 agent。运行时，只有 manifest 被加载到内存。当 agent 把任务匹配到技能时，完整的技能 body 才会被懒加载进来。这使得系统提示词保持精简，让你能在不超出上下文窗口限制的前提下，积累数十个技能。第五篇文章我会展示怎么写自定义技能。

## 它处在什么位置

我觉得把它放在一个 2x2 矩阵里看最清楚。

![Where OpenClaw sits in the agent landscape: hosted vs self-hosted, chat-app vs your-surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/fig1_quadrant.png)

右下角那个格子很小，但那是我要待的地方。我希望我的 agent：

- 跑在我自己的机器上
- 读我的文件、我的笔记、我的邮件
- 发帖到我用来提醒自己的那个 Telegram 线程里
- 不被订阅制计量收费

截至 2026 年， OpenClaw 是我在该象限中发现的最成熟的方案。

## 跟其他方案比怎么样

**Dify。** Dify 是个可视化的 LLM 应用工作流构建器。它有漂亮的 web UI，拖拽节点，还带内置托管。如果你想做个客服场景的 chatbot 然后交给非技术团队用， Dify 是更好的选择。 OpenClaw 没有 GUI——配置全靠 YAML 和环境变量。权衡在于 OpenClaw 是渠道原生的（你的 agent 活在 Telegram 或钉钉里，而不是 web iframe），而且它跑在 $5/月的 VPS 上，不需要托管平台。

**Coze。** Coze 是字节跳动的对话式 AI 构建器。它跟飞书（Lark）集成得很紧，对中文模型支持也很好。如果你已经在字节生态里， Coze 更顺滑。 OpenClaw 模型无关、渠道无关——你可以用 Claude 配钉钉，或者 Qwen 配 Telegram，甚至在一个 agent 里混用三个模型。 Coze 把你锁定在字节的模型 offerings 和飞书这个主要 surface 上。

**LangChain agents。** LangChain 提供基础组件（如 Chain、 Agent、 Retriever、 Memory），但你需要自行搭建服务器、处理聊天平台认证，并实现任务编排循环。如果你是库派的人， LangChain 是对的 tool。 OpenClaw 是一个开箱即用的运行时环境——服务器、编排循环、渠道适配器和技能系统均已内置。你写配置，不写基础设施。

**n8n AI nodes。** n8n 是个带 LLM 调用可视化节点的工作流自动化工具。它擅长连接 SaaS API——"当 Notion 页面更新时，总结它并发到 Slack"。如果你的用例是用 LLM 把五个外部服务串起来， n8n 更干净。 OpenClaw 是为活在聊天平台里且有长运行上下文的对话 agent 建的。 n8n 工作流是无状态的； OpenClaw agent 记得你上周聊过什么。

**需要说明的是：** OpenClaw 放弃了 GUI 和托管 hosting。你需要编辑 YAML 配置文件、查看日志，并在修改配置后重启服务进程。作为回报，你拿到数据的完全控制权，能在自己的硬件上跑，而且能灵活接入任何模型或渠道，不用等 SaaS 厂商加支持。

## 它靠哪三件事立足

![OpenClaw QuickStart (1): What This Thing Actually Is — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_2.png)

我想老实交代哪些事一个 wrapper 解决不了，但 OpenClaw 能解决。

**1. 渠道适配器是真正的工程活。** 钉钉 Stream 协议、微信的封锁策略、 Discord 的速率限制，各有各的坑。 OpenClaw 自带 20 多个渠道的适配器，每个都知道平台的 quirks。自己写一个要花个周末；写五个就是一个月。

用了内置适配器，你能避开这些坑。钉钉采用持久化 WebSocket 连接，并使用自定义心跳协议：要求客户端在 20 秒内对每条消息返回 ACK，否则服务端将重发。 Discord 全局速率限制是所有端点每秒 50 请求，再加上每路由限制。 Telegram 最简单——REST webhook 或长轮询——但你也得处理 parse modes （Markdown, HTML, MarkdownV2）和 inline keyboards 的 quirks。企业微信（WorkBuddy）需要企业认证、回调签名验证和注册 IP 白名单。适配器把这些全处理了。

**2. 技能不只是 system prompts。** 一个 Skill 是一个 Markdown 文件加可选的辅助脚本加一个 manifest。 Agent 启动时只加载 manifest——技能的 body 只有在技能匹配当前任务时才会懒加载进来。这看似平淡，却直接决定了：你的系统提示词是每次对话都消耗约 4k token，还是能构建一个 40k token 的技能库，仅在调用时按需加载、几乎不额外占用上下文。

举个具体的例子。假设你想让 agent 搜索 web、总结 PDF、管理 TODO 列表、发邮件、查 SQL 数据库。如果你把这些全写进 system prompt，每轮对话都要烧掉 8k-10k tokens，其中大部分跟当前任务无关。如果你把它编码成技能， agent 启动时为每个技能加载 200 token 的 manifest，发现 "web-search" 跟当前查询相关，然后才把那 1500 token 的技能 body 分页加载进来。其他四个技能留在磁盘上。

**3. 记忆跨会话存活。** 这里有一个类型化的记忆存储（用户画像、项目状态、参考资料、反馈）加上一个可索引的 `MEMORY.md`。当同一段对话在三天后于不同聊天会话中恢复时， agent 无需冷启动。

记忆存储有五个部分：`user_identity`（姓名、角色、偏好）、`project_context`（活跃项目及其状态）、`references`（API 文档、配置示例）、`feedback`（用户告诉 agent 别再做的事）、`learning_profile`（每个主题的深度偏好）。每个部分均对应一个 Markdown 文件。 Agent 在会话开始时读取这些文件，并在用户输入导致上下文变更时将其更新写回。

如果这三点对你都不重要，你可以停止阅读，直接用 Claude Code。如果至少有一点让你觉得有共鸣，这个系列的剩余部分会物有所值。
## 运行成本

**基础设施。** 个人用， 2-core VPS 配 2GB RAM 和 20GB disk 就够了。我自己跑在 Tokyo 的 $6/month Vultr instance 上。要是家里已经有能跑 Docker 的 home server 或 NAS，直接部署上去，完全免费。启动时间不到 5 秒，内存占用空闲时大概 150MB，负载起来也就 400MB 左右。

**LLM API 成本。** 看使用情况。我平时批处理任务混用 Claude 3.5 Sonnet + Qwen-Plus，平均每月账单 $15 USD 左右。如果你用 DashScope 的 free tier (1 million tokens per month)，个人跑个 Agent 基本等于零成本。团队用 (200 messages/day)，看模型选型，预算大概在 $50-$200/month 之间。

**时间投入。** 系统稳定后，每天 5-10 minutes 维护就够了。前期投入主要是花时间跟完这个系列，大概一个周末或者两个晚上。回报值得：你会得到一个懂你项目、偏好和工作流的系统，而且没有月费，也不怕 vendor lock-in。

## 系列结束后你会得到什么

等到第 6 篇结束时：

- 本地装好 OpenClaw，网关运行在 `:18789`
- 配置好模型提供商 (Bailian Coding Plan, Claude direct, 或 DashScope free tier — 任选)
- 终端里 TUI 能正常工作
- 接通一个真实的聊天渠道 — 系列里我用 Telegram，因为 setup 最干净
- 加载两个自定义 Skills
- 挂载一个 MCP server 用于浏览器自动化

有了这些基础，官方文档里的案例 (second-brain, daily briefing, devops automation, content pipeline) 随便你搭。最后我会挑一个演示。

## 本系列不涉及的内容

- Multi-agent routing — 官方文档讲得更好
- 超出基础范围的 Production hardening
- WeChat WorkBuddy 的深度配置 (只做个 brief demo，毕竟是腾讯产品，注册流程挺麻烦)

下一篇我们直接安装。 Node 环境要是最新的， 10 分钟搞定；要不是， 20 分钟也够了。