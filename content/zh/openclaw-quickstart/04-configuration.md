---
title: "OpenClaw 指南（四）：配置与模型选型"
date: 2026-04-11 09:00:00
tags:
  - openclaw
  - configuration
  - bailian
  - coding-plan
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 4
description: "openclaw.json 完全解析：什么时候该用哪个模型服务商，以及为什么阿里云百炼 Coding Plan 是在国内跑重度 Agent 工作负载最划算的方式——每月 200 元，8 个模型任用包括 Claude。"
disableNunjucks: true
translationKey: "openclaw-quickstart-4"
---
要是整个 OpenClaw 你只打算动一个文件，那必须是它。

`~/.openclaw/openclaw.json` 管着模型、工具、渠道、记忆、定时任务和技能加载。初始化向导会给你生成个默认值，咱们这篇直接过一遍你真正需要上手改的部分。

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_1.png)

## 最小可用配置

![Configuration hierarchy and provider resolution order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/fig_config.png)

去掉注释和默认集，一个能跑起来的配置大概就 25 行：

```json
{
  "agent": {
    "name": "Lobster",
    "default_model": "qwen-plus"
  },
  "providers": {
    "dashscope": {
      "api_key": "sk-...",
      "models": ["qwen-turbo", "qwen-plus", "qwen3-max", "qwen3-coder-plus"]
    }
  },
  "tools": {
    "exec": { "trusted_patterns": ["^ls ", "^cat ", "^echo "] },
    "web_search": { "engine": "bing" }
  },
  "memory": {
    "engine": "semantic",
    "max_tokens_per_turn": 2000
  }
}
```

一共五个板块。咱们按顺序过一遍。

## 完整配置参考

深入细节之前，先看一眼 `openclaw.json` 支持的所有顶层 key：

| Key | 用途 |
|-----|---------|
| `agent` | Agent 身份、默认模型、规划循环设置 |
| `providers` | LLM 提供商端点和凭证（DashScope、Anthropic、OpenAI、百炼、自定义） |
| `tools` | 内置工具配置（exec、web_search、read、write 等） |
| `memory` | 记忆引擎、token 预算、自动写入行为、记忆类型 |
| `channels` | 外部接口（Telegram、微信、钉钉、Web UI） |
| `cron` | 按计划间隔触发技能的定时任务 |
| `mcp` | 第三方工具的 Model Context Protocol 服务器注册 |
| `hooks` | 在特定 Agent 动作前后运行的自定义脚本 |
| `security` | 文件系统限制、 blocked 命令、确认要求 |

大多数新用户最先动的是 `agent`、`providers` 和 `tools`。高阶玩家才会去配 `cron`、`security` 和 `mcp`。

## `agent` — 名字和默认模型

两个字段最关键：名字（用户看到的）和默认模型。默认模型是 Agent 跑规划循环时用的，除非工具或技能显式覆盖。

很多人容易犯的错是把最贵的模型填在这儿，觉得“稳妥”。别这么做。规划循环每轮都要跑。在这儿填 `qwen-plus`，只在那些真正需要推理的特定技能里覆盖成 `qwen3-max`。

## `providers` — 模型在哪

你可以配多个提供商，然后按技能挑选。四个常用选项：

| Provider | 适用场景 |
|----------|----------|
| `dashscope` | 国内托管、便宜、Qwen 系列。我的默认选择。 |
| `anthropic` | 想要 Claude 的推理质量，且不在乎成本。 |
| `openai` |  就是需要 GPT-4 或 GPT-5.4。 |
| `bailian-coding-plan` | 想要一个订阅包揽 Claude + Qwen + DeepSeek + GLM。 |

实际配置长这样。

### DashScope（默认）

```json
"providers": {
  "dashscope": {
    "api_key": "sk-...",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "models": ["qwen-turbo", "qwen-plus", "qwen3-max", "qwen3-coder-plus"]
  }
}
```

这是国内负载的合理默认值。DashScope 免费版每月给 100 万 token 用于 `qwen-turbo`；付费 token 很便宜（`qwen-plus` 输入 0.004 元/1K）。

### Anthropic

```json
"providers": {
  "anthropic": {
    "api_key": "sk-ant-...",
    "models": ["claude-sonnet-4-5", "claude-opus-4", "claude-haiku-4"]
  }
}
```

Claude Sonnet 4.5 是 OpenClaw 目前推理最强的模型。如果你的任务需要深度的多步逻辑（代码生成、研究规划、法律分析），用这个。价格是 3 美元/百万输入 token —— 比 `qwen-plus` 贵大概 750 倍 —— 所以路由要小心。

### OpenAI

```json
"providers": {
  "openai": {
    "api_key": "sk-proj-...",
    "models": ["gpt-4o", "gpt-5.4", "o1-preview"]
  }
}
```

OpenAI 模型主要在两种情况下有用：(1) 你就是需要 GPT-5.4，或者 (2) 你的技能是基于 OpenAI 原型的，不想重写。否则我避开它们 —— 推理 Claude 更强， bulk 工作 Qwen 更便宜。

### `compatible` 提供商类型

任何讲 OpenAI 兼容 API 格式的服务（Ollama、vLLM、LiteLLM、LM Studio、Together AI）都可以作为 `compatible` 提供商添加：

```json
"providers": {
  "ollama-local": {
    "type": "compatible",
    "endpoint": "http://localhost:11434/v1",
    "models": ["llama3.2:8b", "qwen2.5:14b"]
  }
}
```

如果端点在本地，不需要 API key。这就是你完全离线运行 OpenClaw 的方法 —— 把 `agent.default_model` 指向 `ollama-local/llama3.2:8b`，所有规划都在你机器上跑。

对于托管的兼容端点（比如 Together、Groq），加一个 `api_key` 字段就行。

### 模型路由

`agent.default_model` 是你的基线，但单个技能可以通过技能 manifest 里的 `skill.model_override` 覆盖它：

```json
{
  "name": "legal-contract-review",
  "model_override": "claude-sonnet-4-5",
  "description": "..."
}
```

这让你能在便宜 token 上跑规划循环，只在需要的地方切换到昂贵的推理模型。一个典型模式：

- **规划循环**：`qwen-plus`（快、便宜）
- **代码生成技能**：`qwen3-coder-plus`（专为代码优化）
- **深度推理技能**：`claude-sonnet-4-5`（逻辑最强）
- **批量总结**：`qwen-turbo`（每 token 最便宜）

你也可以在运行时通过给 CLI 传 `--model` 来覆盖，但技能级别的覆盖才是把成本纪律 built into 你的 Agent 的方法。

## Coding Plan 技巧

![OpenClaw QuickStart (4): Configuration, Model Providers, and the Coding Plan Trick — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_2.png)

阿里云百炼有个 "Coding Plan" 订阅：每月 200 元拿到八个模型 —— Claude Sonnet 4.5、Qwen3-Max、Qwen3-Coder-Plus、DeepSeek V3.1、GLM-4.6 等等 —— 工作时间用量不限。对于一个每天循环几百次的 Agent 来说，这价格低得离谱。

提供商配置就是一个块：

```json
"bailian-coding-plan": {
  "endpoint": "https://dashscope.aliyuncs.com/coding-plan/v1",
  "api_key": "sk-..."
}
```

然后在 `agent.default_model` 里，直接指向该计划暴露的某个模型 ID（`claude-sonnet-4-5`、`qwen3-max` 等），请求就会算在你的订阅里。

如果你只是在单台笔记本上个人使用 OpenClaw，我不推荐 Coding Plan —— DashScope 的免费额度够了。但如果你要跑 cron 任务或多渠道，我绝对推荐。

## `tools` —— 启用危险工具

有两个工具配置值得调优：

**`exec.trusted_patterns`** —— 绕过每次调用确认的正则。我保持得很窄：只读命令如 `ls`、`cat`、`git status`。任何修改文件系统的操作都保持 gated。

**`web_search.engine`** —— `bing`（便宜、还行）、`serper`（质量更好、收费）、或 `tavily`（最适合 Agent 搜索、更贵）。我默认用 `bing`，让 Agent 需要更好的时候再问。

还有 24 个其他工具。大多数默认值都合理。我改过的几个：

- `read.max_bytes` —— 从 50KB 调到 500KB，这样 Agent 能吞下配置文件。
- `write.allowed_paths` —— 限制在 `~/openclaw-workspace/` 和 `~/Documents/`。这是最有用的安全设置。

## `memory` —— Agent 怎么记事儿

```json
"memory": {
  "engine": "semantic",
  "max_tokens_per_turn": 2000,
  "auto_write": true,
  "types": ["user", "project", "reference", "feedback"]
}
```

`max_tokens_per_turn` 是每个 prompt 里记忆片段的预算。默认是 1000。我调到 2000，因为技能已经吃掉不少 token；如果记忆太小，Agent 会忘事。

`auto_write` 让 Agent 自己决定什么时候提交内容到记忆。关掉这个，你得显式说“记住我更喜欢 Python 而不是 Node"。开着的话，Agent 会推断。我留着它开，接受偶尔的误记。

## `security` —— Agent 不能碰什么

安全配置是可选的，但如果你的 Agent 无人值守运行（cron 任务）或暴露给其他用户（共享 Telegram 机器人），强烈推荐。

```json
"security": {
  "allowed_paths": [
    "~/openclaw-workspace/",
    "~/Documents/",
    "~/Desktop/temp/"
  ],
  "blocked_commands": [
    "rm -rf",
    "sudo",
    "curl.*bash",
    "wget.*\\|.*sh"
  ],
  "require_confirmation": [
    "git push",
    "npm publish",
    "docker rm"
  ]
}
```

**`allowed_paths`** 限制文件系统写入。任何 `write`、`edit` 或 `exec` 调用如果触碰列表外的路径，会被拒绝。这是最大的安全杠杆 —— 它防止意外的“删除我家目录”级联发生。

**`blocked_commands`** 是针对 `exec` 工具调用的正则匹配。如果命令匹配，直接拒绝 —— 没有确认 prompt。用来禁止那些永远不该跑的命令（`rm -rf /`、把 `curl` 管道进 `bash`、提权）。

**`require_confirmation`** 即使命令匹配了 `trusted_pattern`，也强制用户确认。我用它在不可逆操作上，比如 `git push --force`、发布包、或删除云资源。

没有这些，开了 `exec` 的 Agent 能做任何你能做的事。有了它们，你才有护栏。

## `channels` —— 稍后连接

默认是空的。本系列下一篇会加 Telegram。骨架长这样：

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "...",
    "allowed_user_ids": [123456789]
  }
}
```

`allowed_user_ids` 字段精神上是非可选的 —— 没有它，你的机器人会回答任何找到它的人。别不带它就上线。

## `cron` —— 定时任务

```json
"cron": {
  "jobs": [
    {
      "name": "daily-briefing",
      "schedule": "0 7 * * *",
      "skill": "daily-briefing"
    }
  ]
}
```

这个功能把 OpenClaw 从“聊天机器人”变成“默默帮你干活的东西”。cron 条目触发一个技能；技能产生输出；输出发送到默认渠道（或技能指定的渠道）。晨间简报案例就是基于这个建的。

### 更多 cron 示例

**每周项目报告** —— 每周一早上发送提交、关闭 issue 和合并 PR 的总结：

```json
{
  "name": "weekly-project-report",
  "schedule": "0 9 * * 1",
  "skill": "summarize-github-activity",
  "channel": "telegram",
  "params": { "repo": "org/project", "days": 7 }
}
```

**竞争对手监控** —— 每天早上检查竞争对手博客和产品页面，展示新发布：

```json
{
  "name": "competitor-monitor",
  "schedule": "30 8 * * *",
  "skill": "web-scrape-and-diff",
  "channel": "dingtalk",
  "params": {
    "urls": ["https://competitor.com/blog", "https://competitor.com/pricing"],
    "notify_on_change": true
  }
}
```

**依赖安全审计** —— 每周日运行 `npm audit` 或 `pip-audit`，如果发现漏洞发送报告：

```json
{
  "name": "dependency-audit",
  "schedule": "0 10 * * 0",
  "skill": "run-security-audit",
  "channel": "wechat-workbuddy",
  "params": { "project_path": "~/code/myapp", "severity_threshold": "high" }
}
```

`schedule` 字段使用标准 cron 语法：`minute hour day-of-month month day-of-week`。所有时间都是服务器本地时间，除非你在环境变量里设置 `TZ`。
## 环境变量

有些配置项可以直接用环境变量覆盖。特别是跑 Docker 的时候，你肯定不想把 API Key 明文写进配置文件里提交到仓库：

| 配置路径 | 环境变量 |
|-------------|---------------------|
| `providers.dashscope.api_key` | `DASHSCOPE_API_KEY` |
| `providers.anthropic.api_key` | `ANTHROPIC_API_KEY` |
| `providers.openai.api_key` | `OPENAI_API_KEY` |
| `channels.telegram.bot_token` | `TELEGRAM_BOT_TOKEN` |
| `memory.max_tokens_per_turn` | `OPENCLAW_MEMORY_TOKENS` |
| `agent.default_model` | `OPENCLAW_DEFAULT_MODEL` |

优先级顺序很简单：环境变量 > `openclaw.json` > 内置默认值。

敏感信息一律走环境变量，其他配置留在文件里就好，毕竟文件能版本控制，审计起来也方便。

## 配置校验

改完配置重启网关前，先跑一遍这个：

```bash
openclaw config validate
```

它能帮你揪出这些问题：

- **必填字段缺失** — 比如 `providers.dashscope.api_key` 是空的
- **模型 ID 无效** — 比如 `agent.default_model` 设了一个任何 provider 都没注册的模型
- **JSON 格式错误** — 多余的逗号、缺了括号
- **Cron 表达式写错了** — 比如 `schedule: "0 25 * * *"`（哪有 25 点）
- **路径冲突** — 比如 `security.allowed_paths` 里填了个不存在的路径

我经常见到的几个坑：

1. **模型名忘了加引号** — 得是 `"qwen-plus"`，不能是 `qwen-plus`。
2. **JSON 末尾多逗号** — 对象或数组的最后一项后面不能带逗号，JSON 比 JavaScript 严得多。
3. **Provider 不匹配** — `agent.default_model: "claude-sonnet-4-5"` 但根本没配 `anthropic`。校验器会直接告诉你缺了哪个 provider。

校验失败的话，错误信息会带上行号和字段路径。改好再验一遍，没问题了再重启：

```bash
openclaw gateway restart
```

## 编辑后重载

`openclaw.json` 大部分配置改完保存就自动热重载了。例外的是渠道注册和 provider 密钥——这两类改完得重启网关：

```bash
openclaw gateway restart
```

要是改完配置看到 "tool not registered" 或者 "provider not found" 这类报错，就知道该重启了。

## 如果今天我来改你的配置

如果你是用向导初始化的，之后一直没动过配置文件：

1. 把 `security.allowed_paths` 设得明确点。默认是你家目录，范围太大了。
2. `memory.max_tokens_per_turn` 设成 2000。
3. 在 `security.blocked_commands` 里加上 `rm -rf`、`sudo` 和 `curl.*bash`。
4. 要是 token 花费接近 200 元/月，直接换百炼 Coding 计划。
5. 每次编辑完都跑一遍 `openclaw config validate`。别让错误把 agent 搞挂。

下一篇讲渠道。先接 Telegram，因为这事儿不折腾；然后单独截个图过一遍企业微信助手，这服务值得单独花篇博客讲。