---
title: "OpenClaw 指南（四）：配置与模型选型"
date: 2026-04-11 09:00:00
tags:
  - openclaw
  - configuration
  - Bailian
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
如果你只打算修改 OpenClaw 项目中的一个文件，那一定是这个文件。

`~/.openclaw/openclaw.json` 负责管理模型、工具、通信渠道、记忆模块、定时任务和技能加载。初始化向导会生成一套默认配置，本文聚焦于你通常需要手动调整的关键配置项。

![OpenClaw 快速入门（4）：配置、模型提供者和编码计划技巧 — 视图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_1.png)

---

## 最小可用配置

![配置层次结构和提供者解析顺序](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/fig_config.png)

剔除注释和未使用的默认配置后，一个最小可用配置约 25 行：

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

共分为五个配置板块，将按顺序逐一说明。

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
| `security` | 文件系统限制、blocked 命令、确认要求 |

大多数新用户最先调整的是 `agent`、`providers` 和 `tools`；高阶用户才会深入配置 `cron`、`security` 和 `mcp`。

## `agent` — 名字和默认模型

两个最关键的字段是名字（用户看到的）和默认模型。默认模型用于 Agent 的规划循环，除非工具或技能显式覆盖。

常见误区是将最昂贵的模型设为默认模型，以为这样更“稳妥”。千万别这么做——规划循环每轮都会执行，成本会迅速累积。建议在这里使用 `qwen-plus`，仅在真正需要强推理能力的特定技能中覆盖为 `qwen3-max`。

## `providers` — 模型在哪

你可以配置多个提供商，并按技能灵活选择。以下是四个常用选项：

| Provider | 适用场景 |
|----------|----------|
| `dashscope` | 国内托管、便宜、Qwen 系列。我的默认选择。 |
| `anthropic` | 想要 Claude 的推理质量，且不在乎成本。 |
| `openai` | 就是需要 GPT-4 或 GPT-5.4。 |
| `bailian-coding-plan` | 想用一个订阅包揽 Claude + Qwen + DeepSeek + GLM。 |

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

这是国内负载的合理默认值。DashScope 免费版每月提供 100 万 token 用于 `qwen-turbo`；付费 token 很便宜（`qwen-plus` 输入仅 0.004 元/1K）。

### Anthropic

```json
"providers": {
  "anthropic": {
    "api_key": "sk-ant-...",
    "models": ["claude-sonnet-4-5", "claude-opus-4", "claude-haiku-4"]
  }
}
```

Claude Sonnet 4.5 是 OpenClaw 目前推理能力最强的模型。如果你的任务涉及深度多步逻辑（如代码生成、研究规划、法律分析），就选它。价格为 3 美元/百万输入 token——比 `qwen-plus` 贵约 750 倍，因此务必谨慎路由。

### OpenAI

```json
"providers": {
  "openai": {
    "api_key": "sk-proj-...",
    "models": ["gpt-4o", "gpt-5.4", "o1-preview"]
  }
}
```

OpenAI 模型主要适用于两种情况：(1) 你明确需要 GPT-5.4，或 (2) 技能原型基于 OpenAI 开发且不想重写。其他情况下，我通常避免使用——Claude 在推理任务上更强，而 Qwen 在批量处理中更便宜。

### `compatible` 提供商类型

任何兼容 OpenAI API 格式的服务（如 Ollama、vLLM、LiteLLM、LM Studio、Together AI）均可作为 `compatible` 提供商添加：

```json
"providers": {
  "ollama-local": {
    "type": "compatible",
    "endpoint": "http://localhost:11434/v1",
    "models": ["llama3.2:8b", "qwen2.5:14b"]
  }
}
```

如果端点部署在本地，则无需 API Key。这正是实现 OpenClaw 完全离线运行的方法：将 `agent.default_model` 指向 `ollama-local/llama3.2:8b`，所有规划逻辑都在本地执行。

对于托管型兼容端点（如 Together、Groq），只需额外添加 `api_key` 字段即可。

### 模型路由

`agent.default_model` 是你的基线配置，但单个技能可通过技能 manifest 中的 `skill.model_override` 覆盖它：

```json
{
  "name": "legal-contract-review",
  "model_override": "claude-sonnet-4-5",
  "description": "..."
}
```

这种机制让你能在廉价 token 上运行规划循环，仅在必要时切换至高成本推理模型。典型配置如下：

- **规划循环**：`qwen-plus`（快速、便宜）
- **代码生成技能**：`qwen3-coder-plus`（专为代码优化）
- **深度推理技能**：`claude-sonnet-4-5`（逻辑最强）
- **批量摘要任务**：`qwen-turbo`（每 token 成本最低）

你也可以在运行时通过 CLI 的 `--model` 参数临时覆盖，但技能级别的覆盖才是将成本控制内建到 Agent 中的最佳实践。

## Coding Plan 技巧

![OpenClaw 快速入门 (4)：配置、模型提供者和编码计划技巧 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/04-configuration/illustration_2.png)

阿里云百炼提供一种“Coding Plan”订阅：每月 200 元即可获得八个模型——包括 Claude Sonnet 4.5、Qwen3-Max、Qwen3-Coder-Plus、DeepSeek V3.1、GLM-4.6 等——工作时间用量不限。对于每天循环数百次的 Agent 来说，这个定价堪称超值。

提供商配置只需一个区块：

```json
"bailian-coding-plan": {
  "endpoint": "https://dashscope.aliyuncs.com/coding-plan/v1",
  "api_key": "sk-..."
}
```

随后在 `agent.default_model` 中直接指向该计划暴露的任一模型 ID（如 `claude-sonnet-4-5` 或 `qwen3-max`），请求便会自动计入你的订阅额度。

如果你只是在单台笔记本上个人使用，不推荐启用 Coding Plan——DashScope 的免费额度已足够。但若涉及定时任务或多渠道接入，强烈建议切换至此方案。

## `tools` — 启用危险工具

有两个工具配置值得调优：

**`exec.trusted_patterns`** —— 匹配该正则的命令可绕过每次调用的确认提示。我将其限制得很窄，仅包含只读命令如 `ls`、`cat`、`git status`；任何可能修改文件系统的操作都保持受控。

**`web_search.engine`** —— 可选 `bing`（便宜、效果尚可）、`serper`（质量更好、收费）或 `tavily`（最适合 Agent 搜索、价格更高）。我默认使用 `bing`，让 Agent 在需要更高精度时主动请求升级。

其余 24 个工具大多已有合理默认值。我调整过的几个包括：

- `read.max_bytes` —— 从 50KB 提升至 500KB，使 Agent 能完整读取配置文件。
- `write.allowed_paths` —— 限制为 `~/openclaw-workspace/` 和 `~/Documents/`。这是最有价值的安全设置。

## `memory` — Agent 怎么记事儿

```json
"memory": {
  "engine": "semantic",
  "max_tokens_per_turn": 2000,
  "auto_write": true,
  "types": ["user", "project", "reference", "feedback"]
}
```

`max_tokens_per_turn` 控制每轮对话中记忆片段的 token 预算，默认为 1000。我将其提升至 2000，因为技能本身已消耗大量 token；若记忆预算过小，Agent 很容易遗忘上下文。

`auto_write` 允许 Agent 自主决定何时将信息存入记忆。关闭时，你必须显式说出“记住我偏好 Python 而非 Node.js”；开启后，Agent 会自动推断并存档。我选择开启，并接受偶尔产生的冗余记忆。

## `security` — Agent 不能碰什么

安全配置虽为可选，但若你的 Agent 以无人值守方式运行（如 cron 任务）或面向多人开放（如共享 Telegram 机器人），强烈建议启用。

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

**`allowed_paths`** 严格限制文件系统写入范围。任何 `write`、`edit` 或 `exec` 操作若触及列表外路径，将被直接拒绝。这是最重要的安全杠杆——能有效防止“误删家目录”这类灾难性级联故障。

**`blocked_commands`** 通过正则表达式匹配 `exec` 调用的命令。一旦匹配，立即拒绝执行（不提供确认提示）。适用于绝对禁止的命令，如 `rm -rf /`、将 `curl` 输出管道至 `bash`、或任何提权操作。

**`require_confirmation`** 即使命令符合 `trusted_patterns`，也强制要求用户确认。我将其用于不可逆操作，例如 `git push --force`、发布软件包或删除云资源。

没有这些防护，启用了 `exec` 的 Agent 将拥有与你同等的系统权限；有了它们，你才真正拥有了安全护栏。

## `channels` — 稍后连接

默认为空。本系列下一篇将介绍如何接入 Telegram 渠道，其基础配置结构如下：

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "...",
    "allowed_user_ids": [123456789]
  }
}
```

`allowed_user_ids` 字段在实践中不可或缺——若未设置，你的机器人会响应任何找到它的人。切勿在未配置此字段的情况下上线。

## `cron` — 定时任务

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

这一功能将 OpenClaw 从“聊天机器人”转变为“默默为你工作的自动化助手”：cron 条目触发技能，技能生成输出，并发送至默认或指定渠道。晨间简报案例正是基于此构建。

### 更多 cron 示例

**每周项目报告** —— 每周一早晨汇总提交记录、已关闭 issue 和合并的 PR：

```json
{
  "name": "weekly-project-report",
  "schedule": "0 9 * * 1",
  "skill": "summarize-github-activity",
  "channel": "telegram",
  "params": { "repo": "org/project", "days": 7 }
}
```

**竞争对手监控** —— 每日清晨检查竞品博客与产品页面，及时发现新品发布：

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

**依赖安全审计** —— 每周日运行 `npm audit` 或 `pip-audit`，发现漏洞即发送报告：

```json
{
  "name": "dependency-audit",
  "schedule": "0 10 * * 0",
  "skill": "run-security-audit",
  "channel": "wechat-workbuddy",
  "params": { "project_path": "~/code/myapp", "severity_threshold": "high" }
}
```

`schedule` 字段采用标准 cron 语法：`minute hour day-of-month month day-of-week`。所有时间均以服务器本地时区为准，除非通过环境变量 `TZ` 显式指定。

## 环境变量

部分配置项可通过环境变量覆盖，特别适用于 Docker 部署场景——避免将 API Key 明文写入配置文件并提交至代码仓库：

| 配置路径 | 环境变量 |
|-------------|---------------------|
| `providers.dashscope.api_key` | `DASHSCOPE_API_KEY` |
| `providers.anthropic.api_key` | `ANTHROPIC_API_KEY` |
| `providers.openai.api_key` | `OPENAI_API_KEY` |
| `channels.telegram.bot_token` | `TELEGRAM_BOT_TOKEN` |
| `memory.max_tokens_per_turn` | `OPENCLAW_MEMORY_TOKENS` |
| `agent.default_model` | `OPENCLAW_DEFAULT_MODEL` |

优先级顺序为：环境变量 > `openclaw.json` > 内置默认值。

敏感信息（如密钥）应始终通过环境变量传递；其余配置保留在文件中即可，便于版本控制与审计。

## 配置校验

编辑配置后、重启网关前，请务必运行：

```bash
openclaw config validate
```

该命令可检测以下问题：

- **必填字段缺失** —— 例如 `providers.dashscope.api_key` 为空
- **无效模型 ID** —— 例如 `agent.default_model` 指向了未在任何 provider 中注册的模型
- **JSON 格式错误** —— 如多余逗号、缺失括号
- **Cron 表达式错误** —— 如 `schedule: "0 25 * * *"`（不存在第 25 小时）
- **路径冲突** —— 如 `security.allowed_paths` 包含不存在的目录

实践中常见的错误包括：

1. **模型名未加引号** —— 必须写作 `"qwen-plus"`，而非 `qwen-plus`。
2. **JSON 末尾多逗号** —— 对象或数组的最后一项后不可带逗号，JSON 比 JavaScript 更严格。
3. **Provider 不匹配** —— 例如设定了 `agent.default_model: "claude-sonnet-4-5"`，却未配置 `anthropic` provider。校验器会明确提示缺失的 provider。

校验失败时，错误信息会包含行号和字段路径。修正后重新验证，确认无误再重启：

```bash
openclaw gateway restart
```

## 编辑后重载

`openclaw.json` 的大部分配置支持热重载——保存即生效。例外是渠道注册和 provider 密钥，这两类修改需重启网关：

```bash
openclaw gateway restart
```

若修改后出现 “tool not registered” 或 “provider not found” 等错误，即表明需要重启服务。

## 如果今天我来改你的配置

如果你使用向导初始化后从未修改过配置文件，建议立即做以下五件事：

1. 明确设置 `security.allowed_paths`。默认值为整个家目录，范围过大。
2. 将 `memory.max_tokens_per_turn` 设为 2000。
3. 在 `security.blocked_commands` 中加入 `rm -rf`、`sudo` 和 `curl.*bash`。
4. 若月度 token 花费接近 200 元，果断切换至百炼 Coding Plan。
5. 每次编辑后运行 `openclaw config validate`，提前拦截错误，避免 Agent 崩溃。

下一篇将讲解渠道对接。我们先接入 Telegram——因为它简单可靠；随后用一张截图快速过一遍企业微信助手，这项服务值得单独成文详解。
