---
title: "OpenClaw 指南（二）：十分钟安装与首次对话"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - installation
  - tui
  - dashscope
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
README 上写的是五分钟，但我得说十分钟——多出来的五分钟，几乎全是栽在 Node 版本上，第一次装的人难免踩这个坑。

![OpenClaw QuickStart (2): Install and First Chat in 10 Minutes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_1.png)

## 前置条件

- Node `v22.16` 或更新。项目文档没开玩笑。老版本能装上，但网关会在可选链（optional chaining）的地方报错。我自己跑的是 `v24`，这是推荐轨道。
- 运行时大概需要 2 GB 空闲内存，加载大技能的话还得更多。
- 来自以下任一服务的 LLM API Key：DashScope（免费 tier 够用）、Anthropic、OpenAI，或者阿里云百炼编码计划（200 元/月八个模型）。

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

这是唯一的“脚枪”（foot-gun），过了这关后面就顺了。

### 获取 API Key

**DashScope（推荐新手，尤其是国内用户）**

1. 访问 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)。
2. 用阿里云账号注册——大陆账号需要手机号验证。
3. 进入控制台后，点击左侧边栏的 "API-KEY 管理"。
4. 点击 "创建新的 API-KEY"。立刻复制，他们只显示一次。
5. 免费 tier 给 `qwen-plus` 和 `qwen-turbo` 的额度很大。够你折腾好几天才会碰到限制。

如果你在国内，DashScope 是阻力最小的路径。国内网络连他们的端点延迟低于 200ms。不用代理，不用 VPN，没麻烦。

**Anthropic** — [console.anthropic.com](https://console.anthropic.com)，创建账号，添加支付方式，在 Settings > API Keys 下生成 key。最低充值 $5。Claude Sonnet 是 Agent 使用的 sweet spot。

**OpenAI** — [platform.openai.com](https://platform.openai.com)，在 API keys 下生成 key。GPT-4o 作为底层模型表现不错。

**网络注意事项**

如果你在国内：DashScope 端点在国内，很快。Anthropic 和 OpenAI 端点需要代理、香港 VPS 转发请求，或者 SOCKS5 隧道。OpenClaw 尊重 `HTTPS_PROXY` 和 `ALL_PROXY` 环境变量，所以如果你本地已经跑了代理，启动网关前导出这些变量就行。别先在“连接超时”错误上浪费时间调试，先确认你的网络能不能 actually reach 提供商的端点。

如果你在国内以外：三家提供商都不用额外配置就能用。DashScope 的国际端点可用，但从美欧连延迟稍高。

## 安装 OpenClaw

两种方式 — `npm` 全局安装，或者 curl-bash。我偏好 npm，因为我想知道二进制文件在哪：

```bash
npm install -g @anthropic-ai/openclaw@latest
openclaw --version
# 2026.3.13
```

（没错，npm scope 是 `@anthropic-ai`。项目和这个组织的关系说来话长，短版本是“商标历史遗留问题，现在无害”。）

## 安装排错

五个让人栽跟头的地方，按频率排序：

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

这只影响包下载，不影响你的 LLM API 调用。

**(d) 安装后 `openclaw: command not found`** — nvm 的 bin 目录不在你的 PATH 里。开个新终端，或者跑 `source ~/.nvm/nvm.sh`。另外检查你没把包装到了和当前激活版本不同的 Node 版本里（`nvm list` 能看到这个）。

**(e) 版本不匹配 — 全局 vs 本地** — 当你从某个目录运行时，本地的 `node_modules/@anthropic-ai/openclaw` 优先级高于全局的。这会导致令人困惑的 "feature not found" 错误。删掉本地副本：`rm -rf node_modules/@anthropic-ai/openclaw`。原则：一个全局安装，除非你在开发 OpenClaw 本身，否则不要本地副本。

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

```
[gateway] listening on http://127.0.0.1:18789
[agent] loaded skills: 17
[memory] index ready (0 entries)
[channels] none configured (yet)
```

网关是个长运行进程。接下来的部分我们会给它挂载 channels 和 skills。现在它只是待命，没有输入。

## TUI：在终端里跟它聊

![OpenClaw QuickStart (2): Install and First Chat in 10 Minutes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/illustration_2.png)

开第二个终端，运行：

```bash
openclaw tui
```

你会得到一个聊天式的终端 UI。按顺序试几个东西：

```
Hi — introduce yourself in one sentence.

Read the file ~/.zshrc and tell me what aliases I have.

Make a directory ~/openclaw-test and create a file
notes.md inside with the words "first run" in it.
```

应该发生三件事：

1. 第一条消息返回一句话 — 模型在说话。
2. 第二条触发 `read` 工具 — Agent 让网关读文件，你会在网关日志里看到一行 tool-call 闪过。
3. 第三条实际改动了你的文件系统。用 `ls ~/openclaw-test/` 验证。

如果三个都成了，安装完毕。如果只有第一个成了，Agent 没拿到工具访问权限 — 最可能是你选的模型太小，没法可靠地做 tool-calling。换到 `qwen-plus` 或 `qwen3-max` 再试。

### 更多交互尝试

 basics 跑通后，推进一步：

```
Fetch https://news.ycombinator.com and tell me the top 3 stories right now.
```

这会触发 `web_fetch` 工具。在网关日志里你会看到类似这样的一行：

```
[tool:web_fetch] url=https://news.ycombinator.com status=200 bytes=48231
```

这确认 Agent 发出了出站 HTTP 请求，拿到了响应，现在正在总结它。

```
Run `git log --oneline -5` in ~/my-project and explain what the last five commits did.
```

这用到了 `exec` 工具。网关日志显示：

```
[tool:exec] cmd="git log --oneline -5" cwd=/Users/you/my-project exit=0
```

Agent 看到 stdout 并在此基础上推理。如果 exit code 非零，它会告诉你哪里出了问题。

来个多步任务：

```
In ~/my-project, find all files that import lodash, then tell me which lodash
functions are used and whether any of them have native ES equivalents I should
switch to.
```

盯着网关日志 — 你会看到多个工具调用链式执行：一个 `exec` 去 grep，然后几个 `read` 调用来检查单个文件，最后是做综合总结。这就是 Agent 循环在做的事：计划、行动、观察、循环。

### 阅读网关日志

实验时让网关终端保持可见。每次工具调用都会打印一行，包含工具名、关键参数和结果状态。如果 TUI 里有什么静默失败了，网关日志是你找出原因的地方。常见要看的：

- `[tool:*] ... exit=1` — shell 命令失败了。
- `[tool:web_fetch] ... status=403` — 网站 blocked 了请求。
- `[agent] retrying with backoff` — LLM 提供商返回了速率限制或暂时性错误。

## 安装后首先尝试的事

五个任务，复杂度递增。每个测试系统的不同部分：

**1. 纯 LLM — 不用工具**

```
What is the difference between a coroutine and a thread? Two sentences max.
```

这在模型里往返了一次。没涉及工具。如果这个能成，你的 API key 和提供商配置是对的。

**2. 读本地文件**

```
Read /etc/hosts and tell me if there are any custom entries beyond localhost.
```

Agent 调用 `read` 工具。你在测试网关有没有文件系统访问权，工具 registry 是否加载了。

**3. 搜网页**

```
Search the web for "OpenClaw changelog 2026" and summarize what shipped in March.
```

这测试 `web_search`。如果报 "tool not found"，web skill 可能没启用 — 查 `openclaw skill list` 然后用 `openclaw skill enable web` 启用它。

**4. 创建并编辑文件**

```
Create a file ~/openclaw-test/shopping.md with a grocery list: eggs, milk, bread.
Then add "butter" to the list.
```

两个工具调用：`write` 然后 `edit`。Agent 应该在一轮里处理完这两个。用 `cat ~/openclaw-test/shopping.md` 验证最终文件内容。

**5. 多步推理 — exec + read**

```
Find all TODO comments in ~/my-project/src and list them grouped by file,
with the line number and the text of each TODO.
```

这通常会产生一个 `exec` 调用（`grep -rn "TODO" src/`），然后 Agent 格式化输出。对于大代码库，它可能会分页进行多次调用。重点是确认多步执行端到端能跑通。

## 目录结构长什么样

成功安装并首次运行后，`~/.openclaw/` 长这样：

```
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

我平时多半不开它——毕竟 TUI 响应更快——但如果想直观地检查内存状态或者 Skill 状态，这玩意儿挺好使。等你配置了定时任务后，这里也能看到 cron jobs 的运行情况。

## 刚才到底发生了什么（架构视角）

![架构拓扑：终端 -> tui -> gateway -> agent loop / skills index / tool registry -> LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/02-install-and-first-chat/fig1_topology.png)

`tui` 其实只是个薄客户端。核心逻辑 agent loop 其实是在 gateway 里跑的。正是这种分离设计，让你以后能轻松挂上 Telegram、钉钉或者 Web UI 作为不同的前端，它们全都连着同一个 agent 核心。

下一篇，咱们拆开这个 gateway 看看，当你敲下消息时，底层到底在跑什么逻辑。