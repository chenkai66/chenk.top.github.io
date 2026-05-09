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

## 又一个 LLM Wrapper？

每次有人跟我说"我在搞一个 AI Agent 框架"，我的第一反应都是：又一个把 system prompt 套在 API 外面的东西？

说实话，90% 的所谓"Agent 平台"确实就是这样——一个 prompt 模板、一个 API 转发层、一个花里胡哨的 dashboard。你用它和直接调 API 的区别，仅限于多了一层抽象和一笔额外账单。

OpenClaw 不是这个东西。它解决的问题是：**你在哪跟 AI 对话**，以及**对话过程中 AI 记住了什么、能做什么**。

这篇文章的目标很简单：把 OpenClaw 是什么讲清楚，让你能判断它值不值得占用你硬盘上那几百 MB 空间。后续文章负责安装和配置，这篇只负责"认知对齐"。

![OpenClaw 架构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_1.jpg)

---

## 在你继续之前

这个系列假设你具备以下条件。不是门槛——只是确保你不会在第二篇就卡住。

### 你需要有的东西

| 条件 | 为什么需要 | 最低要求 |
|------|-----------|---------|
| 终端操作能力 | OpenClaw 是命令行工具，没有 GUI | 能 `cd`、`ls`、编辑文件 |
| Node.js | 运行时依赖 | v18+ |
| 一个 LLM API Key | Agent 的"大脑" | 任选一个即可 |
| 一个聊天平台账号 | Agent 的"嘴" | Telegram 最容易上手 |

### 关于 LLM API Key

你有几个选择，按上手难度排序：

1. **DashScope（通义千问）**—— 注册送免费额度，国内网络直连，延迟低。如果你在中国大陆，这是阻力最小的选择。
2. **百炼 Coding Plan**—— 阿里云百炼平台的开发者计划，有更高的免费配额，适合长期使用。
3. **Anthropic（Claude）**—— 效果最好，但需要海外支付方式，延迟偏高。

**重要的一点**：OpenClaw 是你自己跑在自己机器上的基础设施，不是一个注册就能用的 SaaS。你需要自己管 API Key、自己选模型、自己承担调用成本。这既是它的门槛，也是它的优势——后面会展开。

### 关于聊天平台

Telegram 是本系列的默认示例平台，原因很实际：

- 创建 Bot 只需要跟 @BotFather 说几句话
- 不需要企业认证
- 国内可以用（需要代理，但技术上完全可行）
- API 干净、文档清晰

当然，你也可以用钉钉、飞书、Discord、微信公众号——OpenClaw 都支持。但每个平台的接入复杂度差异很大，本系列只拿 Telegram 举例。

---

## 那它到底是什么

一句话版本：

> OpenClaw 是一个自托管的 AI Agent 网关。它把 20 多个聊天平台桥接到一个统一的 Agent 运行时，让你在任何你习惯的聊天界面跟 AI 对话，同时保留完整的记忆、技能和工具调用能力。

拆开来说：

### 自托管

跑在你自己的机器上。可以是一台阿里云 ECS、一台腾讯云轻量应用服务器、你家里吃灰的 NUC、甚至一台树莓派。数据不过第三方，对话记录不离开你的硬盘。

### 单一 Node 进程

不是微服务架构，不需要 Docker Compose 编排七八个容器。一个 `node` 进程跑起来就完事了。启动快、资源占用低、调试方便。

### 网关模式

OpenClaw 本身不是模型，它是你和模型之间的一层"智能路由"。消息从聊天平台进来，经过 Agent 运行时处理（记忆检索、技能匹配、工具调用），然后把结果返回给用户。

### MIT 协议

完全开源，MIT 许可。你可以改、可以商用、可以二次分发。没有"社区版/企业版"的把戏。

### 架构鸟瞰图

整体分三层：

![OpenClaw 架构鸟瞰：聊天平台 - Channel Adapter - Agent Runtime - LLM](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/fig1_birds_eye.png)

**Agent Runtime** 是核心，它实现了一个经典的 Agent Loop：

1. **Memory（记忆）**—— 从持久化存储中检索与当前对话相关的上下文
2. **Planning（规划）**—— 决定是直接回答、调用工具、还是加载某个技能
3. **Tool（工具）**—— 执行外部操作（搜索、代码执行、API 调用等）
4. **Reflection（反思）**—— 评估结果质量，决定是否需要重试或补充

这不是什么新发明——ReAct、AutoGPT 都是这个模式。OpenClaw 的价值不在于 Agent Loop 本身，而在于把这个 Loop 稳定地跑在 20 多个聊天平台上，并且提供了一套成熟的技能和记忆管理方案。

---

## 它的定位在哪

理解一个工具最快的方式是看它和同类的差别。我用两个维度来定位：

- **横轴**：你通过什么界面跟 AI 交互？是厂商提供的网页/App（vendor UI），还是你自己的聊天平台（your surface）？
- **纵轴**：服务跑在哪？是厂商的云（hosted），还是你自己的机器（self-hosted）？

|  | Vendor UI | Your Surface |
|--|-----------|-------------|
| **Hosted** | ChatGPT, Claude.ai, 通义千问 | Coze, Dify Cloud |
| **Self-hosted** | Ollama Web UI, text-generation-webui | **OpenClaw** |

OpenClaw 占据右下角：**自托管 + 你自己的聊天界面**。

这意味着什么？意味着你打开钉钉/飞书/Telegram，直接跟一个具有完整记忆和工具调用能力的 AI Agent 对话——而这个 Agent 跑在你自己的服务器上，用你自己选的模型，没有中间商。

---

## 跟竞品比

说到这里，你肯定想问：市面上这么多 Agent 平台，OpenClaw 有什么不一样？

### vs Dify

Dify 是一个很好的产品。它的可视化 Workflow 编辑器对非技术团队非常友好，拖拽连线就能搭出复杂的 Agent 流程。

但 Dify 的核心设计是"工作流构建器"——你的 Agent 是一组预定义的节点和边。OpenClaw 的设计是"对话运行时"——Agent 是一个持续运行的对话进程，能动态加载技能、维护长期记忆。

另外，Dify 的自托管版本资源占用不小（PostgreSQL + Redis + 向量数据库 + 多个 Worker 进程）。OpenClaw 是单进程 + SQLite，一台 1C1G 的 ECS 就能跑。

**选 Dify**：你的团队非技术居多，需要可视化编排，预算允许更大的实例。
**选 OpenClaw**：你是开发者，想要极致轻量，想在现有聊天平台里直接对话。

### vs Coze（扣子）

Coze 是字节做的，跟飞书深度绑定。如果你全公司都在用飞书，Coze 几乎是零摩擦接入。

问题在于：它锁死了平台和模型。你只能用它支持的模型列表里的东西，只能在它支持的渠道里发布。而且数据过字节的服务器。

OpenClaw 是完全 agnostic 的——模型随便换（DashScope 换 Anthropic 一行配置的事），渠道随便加，数据不出你的机器。

**选 Coze**：你在飞书生态里，不想折腾，能接受数据过第三方。
**选 OpenClaw**：你需要模型自由度，需要多平台同时接入，需要数据本地化。

### vs LangChain

LangChain 是一个库（library），不是一个运行时（runtime）。它给你提供了 Agent 的积木块——prompt template、chain、tool、memory 的抽象。但从"写好代码"到"跑成一个能接收消息的服务"，中间的路你得自己走。

OpenClaw 是开箱即用的 runtime。你不需要写 Python/TypeScript 代码，不需要自己处理 WebSocket 连接、不需要自己管消息队列。改 YAML 就行。

**选 LangChain**：你在做深度定制的 AI 应用，需要精细控制每一步。
**选 OpenClaw**：你想快速让一个 Agent 在聊天平台里跑起来，不想从零搭服务。

### vs n8n

n8n 是工作流自动化工具，类似 Zapier 的自托管版。它擅长"事件 A 触发动作 B"这种模式。

但 n8n 本质是无状态的——每次触发都是一个独立执行，不存在"跨会话记忆"。OpenClaw 的核心能力之一就是有状态的持续对话。AI 记得你昨天说过什么、你的项目上下文是什么。

**选 n8n**：你需要的是自动化工作流（收到邮件 → 解析 → 入库 → 通知）。
**选 OpenClaw**：你需要的是一个能对话的 Agent，一个有记忆的助手。

### 诚实的代价

OpenClaw 没有 GUI。所有配置都是 YAML 文件。技能定义是 Markdown。你需要用终端操作一切。

对很多人来说这是劝退项，我完全理解。如果你觉得编辑 YAML 文件是一种折磨，那 Dify 或 Coze 可能更适合你。

但如果你是那种"给我一个配置文件比给我一个 dashboard 更踏实"的人——OpenClaw 就是为你设计的。配置即代码，Git 版本控制，所有变更可追溯。

---

## 三个真正值回票价的能力

抛开框架的抽象概念，OpenClaw 有三个能力是我用了几个月后觉得真正有价值的。

### 1. Channel Adapter——20 多个平台的脏活它帮你干了

每个聊天平台的接入方式都不一样，而且每一个都有自己的坑：

- **钉钉**：用 WebSocket Stream 模式，需要处理心跳、断线重连、企业内应用鉴权
- **Discord**：有严格的 Rate Limit，消息超过 2000 字符就得分片，Slash Commands 注册有延迟
- **微信公众号**：需要通过域名验证，消息加解密，5 秒超时限制
- **Telegram**：相对最简单，但 Webhook 和 Long Polling 两种模式各有利弊
- **飞书**：事件订阅 + 回调验证，消息卡片格式独特

如果你自己做，光把这些平台的消息收发稳定跑通，至少得花两三周。OpenClaw 把这些全部封装成了 Channel Adapter——你在配置里写好 Token，启动进程，消息就开始流入 Agent Runtime 了。

而且关键的是：你同时接入多个平台时，Agent 的记忆和技能是共享的。你在 Telegram 告诉它你的项目名称，切到钉钉它照样记得。

### 2. Skill System——不是 system prompt，是按需加载的知识包

大部分"Agent 框架"对技能的理解就是：把一大段指令塞进 system prompt。结果就是每次对话都把 40K token 的"全部技能"打包发给模型，不管用户问的是什么。

OpenClaw 的 Skill 是有 manifest 的：

```yaml
name: "write-blog-post"
trigger: "当用户提到写博客、写文章、发帖时"
description: "Hugo 博客文章创作工作流"
context_cost: "~4000 tokens"
```

当用户消息进来时，Agent 先判断需不需要加载技能、加载哪个。不需要的技能根本不会出现在 context 里。

这意味着什么？同样的对话：

- **传统方式**：system prompt 40K tokens + user message → 每次都花 40K 的输入成本
- **OpenClaw 方式**：base prompt 2K + 匹配到的 skill 4K + user message → 只花 6K

在 Claude Sonnet 的价格下，这是 $3/MTok vs $0.45/MTok 输入成本的区别。长期使用下来差距巨大。

而且 Skill 是 Markdown 文件，你可以用 Git 管理、版本控制、跨实例共享。写一个技能就像写一篇文档——定义触发条件、描述工作流步骤、列出注意事项。

### 3. Memory——不是 Chat History，是有类型的知识库

ChatGPT 的"记忆"是什么？一坨混在一起的 bullet points，你无法控制它记什么、忘什么。

OpenClaw 的 Memory Store 是分区的，每个区域有明确的语义：

| Section | 用途 | 示例 |
|---------|------|------|
| `user_identity` | 你是谁 | 名字、角色、偏好 |
| `project_context` | 你在做什么 | 项目信息、进度、架构 |
| `references` | 关键参考 | API 地址、配置、文档链接 |
| `feedback` | 行为约束 | "不要用 XX"、"遇到 YY 时这样做" |
| `learning_profile` | 沟通偏好 | 技术深度、解释风格 |

这些记忆跨会话持久化。不管你是关掉 Telegram 重新打开、还是换到钉钉继续聊、还是过了一周才回来——Agent 都知道你是谁、你在做什么项目、你的偏好是什么。

而且你可以直接编辑 Memory 文件（它就是 Markdown），这比对着 ChatGPT 说"请记住我叫张三"然后祈祷它真的记住了要可靠得多。

![Skill 和 Memory 的协作流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/01-what-is-openclaw/illustration_2.jpg)

---

## 跑起来要花多少钱

这是大家最关心的问题。我拆成三块来算。

### 基础设施成本

| 方案 | 月成本 | 适合谁 |
|------|--------|--------|
| 阿里云 ECS 1C1G（抢占式） | ~30 元/月 | 大多数人 |
| 腾讯云轻量 2C2G | ~50 元/月 | 想稳定一点的 |
| 家里的 NUC/旧笔记本 | 电费 | 极客 |
| 树莓派 4B | 一次性 300 元 | 折腾爱好者 |

OpenClaw 本身内存占用约 100-200MB，CPU 基本空闲（瓶颈在 LLM API 的网络延迟）。你已有的任何能跑 Node.js 18+ 的机器都行。

### LLM 调用成本

这取决于你用什么模型、聊多少。我自己的使用情况（每天大概 30-50 轮对话）：

| 模型 | 月均成本 | 备注 |
|------|---------|------|
| 通义千问（qwen-max） | ~50 元/月 | DashScope 价格，有免费额度 |
| Claude 3.5 Sonnet | ~100 元/月 | 效果最好，价格最贵 |
| DeepSeek V3 | ~30 元/月 | 性价比极高 |
| qwen-plus（免费额度期） | 0 元/月 | 新用户前 3 个月 |

如果你只是轻度使用（每天几轮对话），DashScope 的免费额度完全够用，实际成本可以是零。

### 时间成本

- **初始安装配置**：一个周末（本系列会帮你压缩到几小时）
- **日常维护**：5-10 分钟/天（看看日志、偶尔更新 Skill）
- **学习曲线**：如果你会用 Docker 和 YAML，基本没有门槛

---

## 这个系列结束后你会有什么

跟完整个系列，你会得到：

1. **一个跑起来的 OpenClaw 实例**——在你的服务器或本地机器上
2. **至少一个聊天平台接入**——能在 Telegram/钉钉里直接跟 Agent 对话
3. **配置好的 LLM 后端**——连接到 DashScope 或其他模型提供商
4. **几个实用的 Skill**——让 Agent 能做一些具体的事，而不只是闲聊
5. **MCP（Model Context Protocol）工具接入**——让 Agent 能调用外部工具

从"一个空壳"到"一个真正有用的助手"，这个过程大概是 5 篇文章的事。

---

## 这个系列不会覆盖的内容

为了控制篇幅和设定预期，以下话题不在本系列范围内：

- **多 Agent 路由**——一个 OpenClaw 实例根据消息内容分发给不同 Agent 处理。这是高级玩法，之后单独写。
- **生产环境加固**——systemd 管理、日志轮转、监控告警、备份策略。这些是运维话题，不是入门话题。
- **WorkBuddy 深度解析**——OpenClaw 的上层应用之一，有自己的复杂度，值得单独一个系列。

---

## 下一篇

下一篇我们动手：安装 OpenClaw、配置第一个 LLM 后端、把 Agent 跑起来。

如果你已经决定要试，现在可以做的准备工作：

1. 确认你有一台能跑 Node.js 18+ 的机器（本地或云上都行）
2. 注册一个 DashScope 账号（[dashscope.aliyun.com](https://dashscope.aliyun.com)），获取 API Key
3. 如果打算用 Telegram，确认你能访问 Telegram 并注册一个 Bot

准备好了就接着看第二篇。
