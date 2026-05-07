---
title: "OpenClaw 快速上手（二）：10 分钟装好并完成第一次对话"
date: 2026-04-04 09:00:00
tags:
  - openclaw
  - 安装
  - tui
  - dashscope
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 2
description: "在 macOS 或 Ubuntu 上装好 OpenClaw、接上模型、跑起 TUI——10 分钟内拿到一个能用的 Agent。再加上一个让大多数新人栽跟头的 Node 版本陷阱。"
disableNunjucks: true
translationKey: "openclaw-quickstart-2"
---
README 上写的是 5 分钟。我估计得 10 分钟——多出来的 5 分钟是留给几乎每个人第一次都会踩的 Node 版本坑。

这篇文章的目标，是让你在 10 分钟内搞定一个**真正能干活**的 Agent。不是那种只会聊天的——现在能聊天的工具遍地都是。我要的是它能读你的文件、能在你桌面上写文件、还能指挥 Gateway 调用一个 tool。这三件事都跑通了，安装才算完成。
![OpenClaw 快速上手（二）：10 分钟装好并完成第一次对话 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_1.jpg)

## 前置条件

- Node `v22.16` 或更高版本。这个要求很严格，低版本虽然能装上，但 Gateway 在某些地方用到可选链时会直接报错。我用的是 `v24`，因为这是官方推荐的版本。
- 运行时需要大约 2 GB 的空闲内存，如果加载大型 Skills，可能需要更多。
- 一个 LLM API Key，可以从以下平台获取：DashScope（免费档够用）、Anthropic、OpenAI 或阿里云百炼 Coding Plan（200 元/月，支持 8 个模型）。

先检查 Node 版本：

```bash
node -v
# v24.0.x —— 没问题
# v20.x.x —— 太老了，看下面
```

如果版本太旧，安装 `nvm`：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
source ~/.bashrc
nvm install 24
nvm use 24
```

这就是唯一的坑，过了这关后面就很顺了。

补充一句：macOS 上用 Homebrew 装的 Node 我踩过坑——它默认装的是 `node@20`，和 OpenClaw 的依赖链不兼容。我的建议是，**别用包管理器装 Node，永远用 nvm**。这一点我重复三遍：nvm、nvm、nvm。
## 安装 OpenClaw

两种方法：`npm` 全局安装，或者用 `curl-bash`。我更喜欢 npm，因为我想清楚二进制文件具体装在哪里。

```bash
npm install -g @anthropic-ai/openclaw@latest
openclaw --version
# 2026.3.13
```

（对了，npm 的 scope 是 `@anthropic-ai`。这背后有一段商标历史，不过现在完全无害，不用在意。）

装完之后，最好确认一下二进制文件的路径，方便后面配置 pm2 或者写 systemd 服务时用：

```bash
which openclaw
# /Users/you/.nvm/versions/node/v24.0.0/bin/openclaw
```

记下这个路径。后面无论是接 supervisor，还是排查“为什么 cron 调不起来”，都用得上。
## 初始化

运行初始化向导，它会生成一个配置文件到 `~/.openclaw/`：

```bash
openclaw onboard
```

向导会问几个问题：

1. 给助手起个名字——我选了个好记的，这样聊天时可以直接喊名字骂它。我的叫 `Lobster`。
2. 助手该怎么称呼你——我用的是我的真实用户名，而不是 "Boss"，方便查看日志时区分。
3. 选择服务提供商——看你手头有哪个平台的 API Key。这里我选 DashScope，因为它有免费套餐。
4. 输入 API Key。
5. 设置默认模型——通用场景下，`qwen-plus` 是最合适的选择。

向导会把配置写入 `~/.openclaw/openclaw.json`，后面可以随时手动修改。建议完成这一步后立刻打开文件，检查一下内容，顺便在 `model.fallback` 里填一个轻量级模型。主模型出问题时，至少 TUI 还能正常响应。
## 启动 Gateway

```bash
openclaw gateway start
```

你应该会看到类似这样的输出：

```
[gateway] listening on http://127.0.0.1:18789
[agent] loaded skills: 17
[memory] index ready (0 entries)
[channels] none configured (yet)
```

Gateway 是一个长时间运行的进程。后面我会介绍如何给它添加渠道和技能。目前它只是空闲在那里，没有接收任何输入。

如果 18789 端口已经被占用——这是 OpenClaw 的默认端口，很多人可能会撞上——可以在 `openclaw.json` 文件里修改 `gateway.port` 的值。改完后记得在 TUI 这边用 `OPENCLAW_GATEWAY=http://127.0.0.1:新端口` 指定新的地址。
## TUI：直接在终端里和它对话

![OpenClaw 快速上手（二）：10 分钟装好并完成第一次对话 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_2.jpg)

打开第二个终端，运行以下命令：

```bash
openclaw tui
```

你会看到一个类似聊天的终端界面。按顺序试试这几件事：

```
你好，用一句话介绍一下你自己。

读取文件 ~/.zshrc，告诉我里面定义了哪些 alias。

创建一个目录 ~/openclaw-test，并在里面新建一个文件 notes.md，
内容写上 "first run"。
```

接下来应该发生三件事：

1. 第一条消息返回一句话——模型正在回应你。
2. 第二条触发 `read` 工具——Agent 会请求 Gateway 读取文件，你能在 Gateway 日志里看到一行 tool-call 滚动过去。
3. 第三条实际修改了你的文件系统。用 `ls ~/openclaw-test/` 验证一下。

如果这三件事都成功，说明安装完成了。如果只有第一条成功，说明 Agent 没有正确调用工具——大概率是因为你选的模型太小，无法稳定支持 tool-calling。换成 `qwen-plus` 或 `qwen3-max` 再试一次。

我自己第一次就卡在第二条上：当时用了 `qwen-turbo`，模型直接回我“我现在还不能读文件”。换成 `qwen-plus` 后立刻搞定。我的经验是，入门阶段别省 token，先把主流程跑通，再回头优化成本。
## Web 控制面板（可选，但实用）

如果喜欢图形界面，可以用这个：

```bash
openclaw web start
# 打开 http://127.0.0.1:18790
```

我平时基本不开，因为 TUI 更快。不过，查看内存和技能状态时，图形化展示确实直观。配置了 cron 任务后，也能在这里查看任务列表。第一次连接 MCP 服务器或者编写复杂 Skill 时，开着 Web 界面对照调试，能省不少事。
## 架构上到底发生了什么

```
你的终端 --(stdin)--> openclaw tui
                                |
                                v
                         openclaw gateway   :18789
                                |
                +---------------+-------------------+
                |               |                   |
                v               v                   v
            agent loop     skills index      tool registry
                |
                v
         LLM provider (DashScope, Anthropic, ...)
```

`tui` 只是个轻量客户端，真正的核心是网关。agent 的主循环就在网关里运行。这种分离设计让我可以随时接入 Telegram、钉钉或者 Web UI 作为替代前端，它们最终都和同一个 agent 交互。

接下来，我打开这个网关，看看输入消息时具体发生了什么。


## 工作目录长什么样

向导跑完后，`~/.openclaw/` 目录结构大概像这样：

```
~/.openclaw/
├── openclaw.json       # 主配置文件
├── agents/             # 每个 Agent 的会话和日志
│   └── main/
│       └── sessions/
├── skills/             # 自定义 Skill 放这里
├── memory/             # 跨会话记忆存储
└── workspace/
    └── MEMORY.md       # 全局上下文，每次运行都会加载
```

后面五篇文章都会反复提到这套目录结构——建议记住它，排查问题时就不用每次都翻文档了。
## 刚才在架构上发生了什么

```
你的终端 --(stdin)--> openclaw tui
                            |
                            v
                     openclaw gateway   :18789
                            |
            +---------------+-------------------+
            |               |                   |
            v               v                   v
        Agent 循环      Skills 索引       Tool 注册表
            |
            v
     LLM Provider（DashScope, Anthropic, ...）
```

`tui` 只是一个轻客户端。Agent 循环住在 Gateway 里。这层分离正是为什么后面你可以再接 Telegram、钉钉或 Web UI 当替代前端，全都通向同一个 Agent。

换句话说，**Gateway 是大脑，TUI 只是一张嘴**。后面接钉钉，就是给它再加一张嘴；接 cron，就是让它自己长出一只手。这一篇你只是把大脑装上电。

下一篇打开 Gateway，看你按下回车的时候里面到底在发生什么。
