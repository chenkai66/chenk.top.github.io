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
README 声称只需五分钟，但我觉得十分钟更现实——多出来的几分钟基本都花在常见的 Node 版本问题上了。

![OpenClaw 快速入门（2）：10 分钟内安装并进行首次聊天 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/illustration_1.png)

## 前置知识
- Node `v22.16` 或更新版本。项目对此要求严格：老版本虽然能安装成功，但在某些使用可选链（optional chaining）的地方，网关会报错。我用的是 `v24`，因为这是官方推荐的版本。
- 运行时需约 2 GB 空闲内存；若加载大型技能，需求会更高。
- 需要以下任一平台的 LLM API Key：DashScope（免费额度足够）、Anthropic、OpenAI，或阿里云百炼编码计划（200 元/月，支持八个模型）。

先检查你的 Node 版本：

```bash
node -v
# v24.0.x — good
# v20.x.x — too old, see next block
```

如果卡在旧版本，建议安装 `nvm`：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
source ~/.bashrc
nvm install 24
nvm use 24
```

这是唯一真正的坑。只要 Node 版本没问题，后续流程就会非常顺畅。

### 获取 API Key

**DashScope（新手首选，尤其适合国内用户）**

1. 访问 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)。
2. 使用阿里云账号注册——大陆用户需完成手机号验证。
3. 进入控制台后，点击左侧边栏的 “API-KEY 管理”。
4. 点击 “创建新的 API-KEY”，并立即复制保存（密钥仅显示一次）。
5. 免费额度对 `qwen-plus` 和 `qwen-turbo` 非常慷慨，足够你连续实验好几天才可能触及上限。

如果你身处中国大陆，DashScope 是最省心的选择：其国内节点延迟通常低于 200ms，无需代理、VPN 或其他绕行手段。

**Anthropic** — 访问 [console.anthropic.com](https://console.anthropic.com)，注册账号并绑定支付方式，然后在 Settings > API Keys 中生成密钥。最低充值为 5 美元。Claude Sonnet 是 Agent 场景下的性价比之选。

**OpenAI** — 访问 [platform.openai.com](https://platform.openai.com)，在 API keys 页面生成密钥。GPT-4o 作为底层模型表现优异。

**网络注意事项**

如果你在中国大陆：DashScope 的端点位于境内，访问迅速；而 Anthropic 和 OpenAI 的端点则需要通过代理、香港 VPS 转发请求，或 SOCKS5 隧道才能连通。OpenClaw 支持 `HTTPS_PROXY` 和 `ALL_PROXY` 环境变量，因此如果你本地已有代理服务，只需在启动网关前导出这些变量即可。切勿在未确认网络连通性的情况下盲目排查“连接超时”错误——先确保你能从当前网络实际访问目标提供商的 API 端点。

如果你在境外：三家服务均可直接使用，无需额外配置。DashScope 也提供国际端点，但从美国或欧洲访问时延迟略高。

## 安装 OpenClaw

有两种方式：`npm` 全局安装，或使用 curl-bash 脚本。我偏好 `npm`，因为这样能清楚知道二进制文件的位置：

```bash
npm install -g @anthropic-ai/openclaw@latest
openclaw --version
# 2026.3.13
```

（是的，npm 包的作用域是 `@anthropic-ai`。这背后有一段复杂的历史，简单说就是“商标遗留问题，现已无害”。）

## 安装排错

以下是五大高频问题，按发生频率排序：

**(a) npm 权限错误（全局安装时报 EACCES）** — 永远不要用 `sudo npm install -g`。推荐改用 `nvm`（将 Node 安装到用户目录），或手动配置 npm 的 prefix 到用户拥有的路径：

```bash
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH="$HOME/.npm-global/bin:$PATH"  # add to shell profile
```

**(b) macOS 上的 node-gyp 构建失败** — 某些可选依赖需要编译原生插件。解决方法：运行 `xcode-select --install`，等待约 1.2 GB 的组件下载完成后重试。

**(c) 国内网络超时** — 默认 npm registry 在中国大陆访问不稳定。建议切换为 npmmirror：

```bash
npm install -g @anthropic-ai/openclaw@latest --registry=https://registry.npmmirror.com
```

注意：这只影响包下载过程，不影响后续的 LLM API 调用。

**(d) 安装后提示 `openclaw: command not found`** — 很可能是 nvm 的 bin 目录未加入 PATH。解决方法：打开新终端，或运行 `source ~/.nvm/nvm.sh`。同时确认你安装时使用的 Node 版本与当前激活的版本一致（可通过 `nvm list` 查看）。

**(e) 全局与本地版本冲突** — 如果你在某个项目目录下运行命令，且该目录的 `node_modules/@anthropic-ai/openclaw` 存在，它会优先于全局安装的版本，导致出现“feature not found”等令人困惑的错误。解决方法：删除本地副本：`rm -rf node_modules/@anthropic-ai/openclaw`。原则是：只保留一个全局安装，除非你正在开发 OpenClaw 本身，否则不要保留本地副本。

## 初始化向导

运行初始化向导，它会将配置文件写入 `~/.openclaw/`：

```bash
openclaw onboard
```

向导会依次询问：

1. **Agent 的名字** — 我喜欢选个好记的名字，方便在聊天中“点名批评”。我的叫 `Lobster`。
2. **它该如何称呼你** — 我用真实用户名而非“Boss”，这样看日志时更清晰。
3. **选择模型提供商** — 选你已有 API Key 的那一家。本教程使用 DashScope，因其提供免费额度。
4. **输入 API Key**。
5. **默认模型** — 推荐使用 `qwen-plus`，适合大多数通用场景。

向导最终会生成 `~/.openclaw/openclaw.json`。后续可随时手动编辑该文件。

## 启动网关

```bash
openclaw gateway start
```

你应该会看到类似如下输出：

```toml
[gateway] listening on http://127.0.0.1:18789
[agent] loaded skills: 17
[memory] index ready (0 entries)
[channels] none configured (yet)
```

网关是一个长期运行的进程。后续文章会为其接入各种通信渠道（channels）和技能（skills）。目前它处于待命状态，尚未接收任何输入。

## TUI：在终端中与它对话

![OpenClaw 快速入门 (2)：10 分钟内安装和首次聊天 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/illustration_2.png)

另开一个终端，运行：

```bash
openclaw tui
```

你会进入一个类聊天的终端界面。按顺序尝试以下几条指令：

```text
Hi — introduce yourself in one sentence.

Read the file ~/.zshrc and tell me what aliases I have.

Make a directory ~/openclaw-test and create a file
notes.md inside with the words "first run" in it.
```

此时应发生三件事：

1. 第一条消息返回一句简短回复——说明模型已正常响应。
2. 第二条触发 `read` 工具——Agent 请求网关读取文件，你会在网关日志中看到一行 tool-call 记录闪过。
3. 第三条实际修改了你的文件系统。可通过 `ls ~/openclaw-test/` 验证结果。

如果三项均成功，说明安装已完成。如果仅第一项成功，很可能是 Agent 无法调用工具——最常见的原因是所选模型太小，无法可靠执行工具调用。建议切换至 `qwen-plus` 或 `qwen3-max` 后重试。

### 更多交互尝试

基础功能验证通过后，可以进一步测试：

```text
Fetch https://news.ycombinator.com and tell me the top 3 stories right now.
```

这会触发 `web_fetch` 工具。在网关日志中，你会看到类似这样的记录：

```text
[tool:web_fetch] url=https://news.ycombinator.com status=200 bytes=48231
```

这表明 Agent 已成功发起 HTTP 请求、获取响应，并正在对内容进行总结。

```text
Run `git log --oneline -5` in ~/my-project and explain what the last five commits did.
```

这使用了 `exec` 工具。网关日志会显示：

```text
[tool:exec] cmd="git log --oneline -5" cwd=/Users/you/my-project exit=0
```

Agent 会读取命令的标准输出（stdout）并据此推理；若退出码非零，它会明确告诉你哪里出错了。

再试一个多步任务：

```text
In ~/my-project, find all files that import lodash, then tell me which lodash
functions are used and whether any of them have native ES equivalents I should
switch to.
```

观察网关日志——你会看到多个工具调用被串联执行：先是一个 `exec` 执行 grep，接着若干次 `read` 读取具体文件，最后进行综合归纳。这正是 Agent Loop 的核心机制：计划 → 行动 → 观察 → 重复。

### 阅读网关日志

实验过程中，请保持网关终端可见。每次工具调用都会输出一行日志，包含工具名、关键参数和执行状态。如果 TUI 中出现静默失败，网关日志就是排查问题的关键。常见日志模式包括：

- `[tool:*] ... exit=1` — shell 命令执行失败。
- `[tool:web_fetch] ... status=403` — 目标网站拒绝了请求。
- `[agent] retrying with backoff` — LLM 提供商返回了速率限制或临时错误。

## 安装后首先尝试的五件事

以下五个任务按复杂度递增排列，分别测试系统不同模块：

**1. 纯 LLM 对话（不使用工具）**

```text
What is the difference between a coroutine and a thread? Two sentences max.
```

此操作仅在模型中往返一次，不涉及任何工具。若能成功，说明你的 API Key 和提供商配置正确。

**2. 读取本地文件**

```text
Read /etc/hosts and tell me if there are any custom entries beyond localhost.
```

Agent 调用 `read` 工具。此步骤用于验证网关是否具备文件系统访问权限，以及工具注册表是否已正确加载。

**3. 网页搜索**

```text
Search the web for "OpenClaw changelog 2026" and summarize what shipped in March.
```

此操作测试 `web_search` 功能。若提示 “tool not found”，可能是 web 技能未启用——运行 `openclaw skill list` 查看，并用 `openclaw skill enable web` 启用。

**4. 创建并编辑文件**

```text
Create a file ~/openclaw-test/shopping.md with a grocery list: eggs, milk, bread.
Then add "butter" to the list.
```

涉及两次工具调用：先 `write`，再 `edit`。Agent 应能在单轮对话中完成这两步。最终可通过 `cat ~/openclaw-test/shopping.md` 验证文件内容。

**5. 多步推理（exec + read）**

```text
Find all TODO comments in ~/my-project/src and list them grouped by file,
with the line number and the text of each TODO.
```

通常会先触发一个 `exec` 调用（如 `grep -rn "TODO" src/`），随后 Agent 对输出进行格式化。对于大型代码库，它可能会分多次调用以分页处理。重点在于验证端到端的多步执行能力是否正常。

## 目录结构概览

成功安装并首次运行后，`~/.openclaw/` 目录结构如下：

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

关键说明：
- `openclaw.json` 是带注释的纯 JSON 文件，可随时手动编辑。
- `memory.json` 初始为空，随着 Agent 从对话中提取事实而逐渐增长——这就是它实现跨会话“记忆”的机制。
- `skills/builtin/` 包含网关启动时加载的内置工具定义。如果你计划后续开发自定义技能，阅读这些文件会很有启发。
- `logs/gateway.log` 持久化存储你在网关终端看到的所有输出；日志轮转策略可通过配置项 `logging.retention_days` 调整。

## Web 面板（可选但实用）

如果你更习惯图形界面，也可以启动 Web 控制台：

```bash
openclaw web start
# open http://127.0.0.1:18790
```

我平时很少开启它——TUI 响应更快——但在需要直观查看记忆状态或技能状态时，这个面板非常有用。一旦你配置了定时任务，这里还会显示 cron job 的运行情况。

## 刚才发生了什么？（架构视角）

![架构拓扑：终端 -> tui -> gateway -> agent loop / skills index / tool registry -> LLM provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/02-install-and-first-chat/fig1_topology.png)

`tui` 本质上只是一个轻量级客户端，真正的 Agent Loop 逻辑运行在网关（gateway）中。正是这种前后端分离的设计，使得后续可以轻松接入 Telegram、钉钉或 Web UI 等多种前端，它们都共享同一个 Agent 核心。

下一篇，我们将深入网关内部，看看当你输入一条消息时，底层究竟发生了什么。
