---
title: "OpenClaw 指南（二）：十分钟安装与首次对话"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - installation
  - tui
  - DashScope
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 2
description: "在 macOS 或 Ubuntu 上装好 OpenClaw，接入模型服务商，启动 TUI，十分钟之内得到一个能用的 Agent。顺便解决那个浪费最多时间的 Node 版本坑。"
disableNunjucks: true
translationKey: "openclaw-quickstart-2"
---
README 声称只需五分钟，但实际上常常需要十分钟，多出的时间几乎全耗在 Node 版本兼容性上，新手极易卡在这里。

![OpenClaw 快速入门（2）：10 分钟内安装并进行首次聊天 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_1.png)

## 前置条件

- Node `v22.16` 或更新。项目文档没开玩笑。老版本能装上，但网关会在可选链（optional chaining）的地方报错。我自己跑的是 `v24`，这是推荐轨道。
- 运行时大约需要 2 GB 空闲内存，如果加载大技能则需要更多。
- 来自以下任一服务的 LLM API Key： DashScope （免费 tier 够用）、 Anthropic、 OpenAI 或者阿里云百炼编码计划（200 元/月八个模型）。

先查 Node 版本：

```bash
node -v
# v24.0.x — good
# v20.x.x — too old, see next block
```

如果卡在老版本，装 `nvm`：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
source ~/.bashrc
nvm install 24
nvm use 24
```

这是唯一的真正‘陷阱’（foot-gun）：Node 版本不兼容会导致后续运行时错误，修复后流程即可恢复正常。

### 获取 API Key

**DashScope （推荐新手，尤其是国内用户）**

1. 访问 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)。
2. 用阿里云账号注册——大陆账号需要手机号验证。
3. 进入控制台后，点击左侧边栏的 “API-KEY 管理”。
4. 点击 “创建新的 API-KEY”。立刻复制，他们只显示一次。
5. 免费 tier 给 `qwen-plus` 和 `qwen-turbo` 的额度很大。够你折腾好几天才会碰到限制。

国内用户首选 DashScope，其国内端点延迟通常低于 200ms，无需代理或 VPN。

**Anthropic** — [console.anthropic.com](https://console.anthropic.com)，创建账号，添加支付方式，在 Settings > API Keys 下生成 key。最低充值 $5。 Claude Sonnet 是 Agent 场景下的理想选择。

**OpenAI** — [platform.openai.com](https://platform.openai.com)，在 API keys 下生成 key。 GPT-4o 作为底层模型表现不错。

**网络注意事项**

如果你在国内， DashScope 端点在国内，速度很快；而 Anthropic 和 OpenAI 端点则需要代理、香港 VPS 转发请求或 SOCKS5 隧道。 OpenClaw 尊重 `HTTPS_PROXY` 和 `ALL_PROXY` 环境变量，所以如果你本地已经跑了代理，启动网关前导出这些变量就行。不要急于调试‘连接超时’错误，先确认你的网络能否实际访问对应提供商的 API 端点。

如果你不在国内，三家提供商都不需要额外配置即可使用。DashScope 的国际端点可用，但从美欧连接时延迟稍高。

## 安装 OpenClaw

两种方式 — `npm` 全局安装，或者 curl-bash。我偏好 npm，因为我想知道二进制文件在哪：

```bash
npm install -g @anthropic-ai/openclaw@latest
openclaw --version
# 2026.3.13
```

（没错， npm scope 是 `@anthropic-ai`。项目和这个组织的关系说来话长，短版本是“商标历史遗留问题，现在无害”。）

## 安装排错

五大高频踩坑点（按发生频率排序）：

**(a) npm 权限错误（全局安装时的 EACCES）** — 永远别 `sudo npm install -g`。改用 `nvm`（把 Node 装到家目录），或者配置 npm 的 prefix 到用户拥有的路径：

```bash
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH="$HOME/.npm-global/bin:$PATH"  # add to shell profile
```

**(b) macOS 上的 node-gyp 构建失败** — 一些可选依赖要编译原生 addons。解决：`xcode-select --install`，等它下载 1.2 GB，重试。

**(c) 国内网络超时** — 默认 npm registry 在国内不稳定。用 npmmirror：

```bash
npm install -g @anthropic-ai/openclaw@latest --registry=https://registry.npmmirror.com
```

这仅影响包下载，不影响 LLM API 调用。

**(d) 安装后 `openclaw: command not found`** — nvm 的 bin 目录不在你的 PATH 里。开个新终端，或者跑 `source ~/.nvm/nvm.sh`。另外检查你没把包装到了和当前激活版本不同的 Node 版本里（`nvm list` 能看到这个）。

**(e) 版本不匹配 — 全局 vs 本地** — 当你从某个目录运行时，本地的 `node_modules/@anthropic-ai/openclaw` 优先级高于全局的。这会导致令人困惑的 "feature not found" 错误。删掉本地副本：`rm -rf node_modules/@anthropic-ai/openclaw`。原则是全局安装一次，除非你在开发 OpenClaw 本身，否则不要创建本地副本。

## 初始化向导

运行 onboarding 向导。它会把配置文件写到 `~/.openclaw/`：

```bash
openclaw onboard
```

它会问：

1. 给 Agent 起什么名 — 我选个好记的，这样聊天时能指名道姓地训它。 mine is `Lobster`。
2. 它该怎么称呼你 — 我用真实的 handle，不用 "Boss"。看日志时有帮助。
3. 选哪个提供商 — 选你有 key 的那个。这次演示我用 DashScope，因为有免费 tier。
4. API key。
5. 默认模型 — `qwen-plus` 是通用场景的正确默认值。

向导会写入 `~/.openclaw/openclaw.json`。后面你可以手改。

## 启动网关

```bash
openclaw gateway start
```

你应该看到类似这样的输出：

```toml
[gateway] listening on http://127.0.0.1:18789
[agent] loaded skills: 17
[memory] index ready (0 entries)
[channels] none configured (yet)
```

网关是个长运行进程。接下来的部分我们会给它挂载 channels 和 skills。现在它只是待命，没有输入。

## TUI：在终端里跟它聊

![OpenClaw 快速入门 (2)：10 分钟内安装和首次聊天 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_2.png)

开第二个终端，运行：

```bash
openclaw tui
```

你会得到一个聊天式的终端 UI。按顺序试几个东西：

```text
Hi — introduce yourself in one sentence.

Read the file ~/.zshrc and tell me what aliases I have.

Make a directory ~/openclaw-test and create a file
notes.md inside with the words "first run" in it.
```

应该发生三件事：

1. 第一条消息返回一句话 — 模型在说话。
2. 第二条触发 `read` 工具 — Agent 让网关读文件，你会在网关日志里看到一行 tool-call 闪过。
3. 第三条实际改动了你的文件系统。用 `ls ~/openclaw-test/` 验证。

如果三个都成了，安装完毕。如果只有第一个成了， Agent 没拿到工具访问权限 — 最可能是你选的模型太小，没法可靠地做 tool-calling。换到 `qwen-plus` 或 `qwen3-max` 再试。

### 更多交互尝试

 basics 跑通后，推进一步：

```text
Fetch https://news.ycombinator.com and tell me the top 3 stories right now.
```

这会触发 `web_fetch` 工具。在网关日志里你会看到类似这样的一行：

```text
[tool:web_fetch] url=https://news.ycombinator.com status=200 bytes=48231
```

这表明 Agent 已发出出站 HTTP 请求、成功获取响应，并正在对内容进行总结。

```text
Run `git log --oneline -5` in ~/my-project and explain what the last five commits did.
```

这用到了 `exec` 工具。网关日志显示：

```text
[tool:exec] cmd="git log --oneline -5" cwd=/Users/you/my-project exit=0
```

Agent 会读取命令的标准输出（stdout）进行推理；若退出码（exit code）非零，则直接返回具体错误信息。

来个多步任务：

```text
In ~/my-project, find all files that import lodash, then tell me which lodash
functions are used and whether any of them have native ES equivalents I should
switch to.
```

盯着网关日志 — 你会看到多个工具调用链式执行：一个 `exec` 去 grep，然后几个 `read` 调用来检查单个文件，最后是做综合总结。这就是 Agent 循环在做的事：计划、行动、观察、循环。

### 阅读网关日志

实验时让网关终端保持可见。每次工具调用都会打印一行，包含工具名、关键参数和结果状态。如果 TUI 中有静默失败的情况，网关日志可以帮助你找出原因。常见的问题包括：

- `[tool:*] ... exit=1` — shell 命令失败了。
- `[tool:web_fetch] ... status=403` — 网站 blocked 了请求。
- `[agent] retrying with backoff` — LLM 提供商返回了速率限制或暂时性错误。

## 安装后首先尝试的事

五个任务，复杂度递增。每个测试系统的不同部分：

**1. 纯 LLM — 不用工具**

```text
What is the difference between a coroutine and a thread? Two sentences max.
```

这在模型里往返了一次。没涉及工具。如果这个能成，你的 API key 和提供商配置是对的。

**2. 读本地文件**

```text
Read /etc/hosts and tell me if there are any custom entries beyond localhost.
```

Agent 调用 `read` 工具。你在测试网关有没有文件系统访问权，工具 registry 是否加载了。

**3. 搜网页**

```text
Search the web for "OpenClaw changelog 2026" and summarize what shipped in March.
```

这测试 `web_search`。如果报 "tool not found"， web skill 可能没启用 — 查 `openclaw skill list` 然后用 `openclaw skill enable web` 启用它。

**4. 创建并编辑文件**

```text
Create a file ~/openclaw-test/shopping.md with a grocery list: eggs, milk, bread.
Then add "butter" to the list.
```

两个工具调用：`write` 然后 `edit`。 Agent 应该在一轮里处理完这两个。用 `cat ~/openclaw-test/shopping.md` 验证最终文件内容。

**5. 多步推理 — exec + read**

```text
Find all TODO comments in ~/my-project/src and list them grouped by file,
with the line number and the text of each TODO.
```

这通常会产生一个 `exec` 调用（`grep -rn "TODO" src/`），然后 Agent 格式化输出。对于大代码库，它可能会分页进行多次调用。重点是确认多步执行端到端能跑通。

## 目录结构长什么样

成功安装并首次运行后，`~/.openclaw/` 长这样：

```text
~/.openclaw/
├── openclaw.json          # Main config: provider, model, agent name, preferences
├── workspace/
│   └── default/           # Default workspace for general tasks
│       ├── memory.json    # Conversation memory index
│   └── sessions/      # Individual session transcripts
│           └── 2026-04-04_001.json
├── skills/
│   ├── builtin/           # Ships with the install — read, write, exec, web_fetch, etc.
│   └── community/         # Installed via `openclaw skill install <name>`
├── agents/
│   └── lobster.json       # Agent personality and config (named during onboard)
└── logs/
    └── gateway.log        # Rolling log, last 7 days kept by default
```

关键点：`openclaw.json` 是带注释的纯 JSON — 随时直接编辑它。`memory.json` 开始是空的，随着 Agent 从对话中提取事实而增长（这就是它如何在会话间“记忆”的方式）。`skills/builtin/` 有网关启动时加载的工具定义 — 如果你计划后面写自定义技能，读读它们很有启发。`logs/gateway.log` 持久化你在网关终端看到的相同输出；轮转可通过配置中的 `logging.retention_days` 配置。
## Web 面板（可选，但挺有用）

如果你更习惯用网页界面，也可以启动一个：

```bash
openclaw web start
# open http://127.0.0.1:18790
```

我平时多半不开它——毕竟 TUI 响应更快——但如果想直观地检查内存状态或 Skill 状态，这个界面非常实用。配置定时任务后，这里还能看到 cron jobs 的运行情况。

## 刚才到底发生了什么（架构视角）

![架构拓扑：终端 -> tui -> gateway -> agent loop / skills index / tool registry -> LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/fig1_topology.png)

`tui` 其实只是个薄客户端。核心逻辑 agent loop 其实是在 gateway 里跑的。正是这种分离设计，让你以后能轻松挂上 Telegram、钉钉或者 Web UI 作为不同的前端，它们全都连着同一个 agent 核心。

下一篇，咱们拆开这个 gateway 看看，当你敲下消息时，底层到底在跑什么逻辑。