---
title: "OpenClaw 快速上手（四）：配置文件、模型 Provider，以及 Coding Plan 的小窍门"
date: 2026-04-06 09:00:00
tags:
  - openclaw
  - 配置
  - 百炼
  - coding-plan
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 4
description: "openclaw.json 全解读、什么时候该用哪个模型 Provider，以及为什么阿里云百炼 Coding Plan 是国内跑重型 Agent 工作负载最划算的选择——200 元/月含 Claude 在内的 8 个模型。"
disableNunjucks: true
translationKey: "openclaw-quickstart-4"
---
如果只让我在 OpenClaw 里改一个文件，那一定是这个。

`~/.openclaw/openclaw.json` 负责管理模型、工具、渠道、内存、定时任务和技能加载。安装向导已经生成了一份默认配置，我来详细说明一下实际会用到的部分。
![OpenClaw 快速上手（四）：配置文件、模型 Provider，以及 Coding Plan 的小窍门 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_1.jpg)

## 最简可用配置

去掉注释和默认设置，一个能用的配置文件大概 25 行：

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

一共 5 个部分，我挨个讲一下。
## `agent`——名字与默认模型

有两个关键字段：名字（用户能看到的）和默认模型。默认模型是 Agent 在规划循环中使用的模型，除非某些工具或技能替换了它。

很多人会犯一个错误：为了“保险”，在这里填上最贵的模型。其实没必要。规划循环每轮都会运行，所以这里直接写 `qwen-plus` 就行。如果某些技能确实需要更强的推理能力，可以在那些技能里单独覆盖成 `qwen3-max`。
## `providers`——模型放在哪

我可以配置多个 providers，按技能需求选择。常用的有四个选项：

| Provider | 什么时候用 |
|----------|-----------|
| `dashscope` | 国内托管，价格便宜，Qwen 系列。我一般默认选它。 |
| `anthropic` | 如果想要 Claude 的推理质量，又不在乎花钱，就选它。 |
| `openai` | 需要特定用 GPT-4 或 GPT-5.4 的时候。 |
| `bailian-coding-plan` | 想一次性订阅，打包 Claude、Qwen、DeepSeek 和 GLM。 |

Coding Plan 确实很有意思，下面详细说说。
## Coding Plan 的妙用

![OpenClaw 快速上手（四）：配置文件、模型 Provider，以及 Coding Plan 的小窍门 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_2.jpg)

阿里云百炼提供了一个叫“Coding Plan”的订阅服务：每月 200 元，包含 8 个模型——Claude Sonnet 4.5、Qwen3-Max、Qwen3-Coder-Plus、DeepSeek V3.1、GLM-4.6 等——工作时间随便用。如果一天需要跑几百次任务，这个价格简直低得离谱。

配置很简单，只需要一个块：

```json
"bailian-coding-plan": {
  "endpoint": "https://dashscope.aliyuncs.com/coding-plan/v1",
  "api_key": "sk-..."  // 用 Coding Plan 的 Key，不是 DashScope 普通的 Key
}
```

接着在 `agent.default_model` 里指定订阅中提供的某个模型 ID（比如 `claude-sonnet-4-5` 或 `qwen3-max`），请求就会自动计入你的订阅。

如果只是单台电脑自用，跑得也不多，没必要买——DashScope 的免费额度完全够用。但如果你有定时任务（cron jobs）或者多渠道运行的需求，Coding Plan 绝对值得入手。
## `tools`——启用利器

有两个工具配置值得调整：

**`exec.trusted_patterns`**——设置正则表达式，跳过每次调用时的确认。我设得比较严格：只允许像 `ls`、`cat`、`git status` 这样的只读命令。凡是会修改文件系统的操作，一律保持拦截。

**`web_search.engine`**——可以选择 `bing`（便宜，够用）、`serper`（质量更好，收费）或 `tavily`（最适合智能体搜索，价格更高）。我默认用 `bing`，如果需要更好的结果，让智能体自己提需求。

其他 24 个工具大多默认值就够用了。我改过的有以下两个：

- `read.max_bytes`——从 50KB 提高到 500KB，这样智能体可以一次性读取一个配置文件。
- `write.allowed_paths`——限制在 `~/openclaw-workspace/` 和 `~/Documents/` 范围内。这是最有用的一条安全设置。
## `memory`——Agent 的记忆方式

```json
"memory": {
  "engine": "semantic",
  "max_tokens_per_turn": 2000,
  "auto_write": true,
  "types": ["user", "project", "reference", "feedback"]
}
```

`max_tokens_per_turn` 是每轮 Prompt 中分配给记忆片段的 Token 预算。默认值是 1000，但我调到了 2000。因为技能本身就会消耗不少 Token，如果记忆空间太小，Agent 很容易忘事。

`auto_write` 决定了 Agent 是否能自动写入记忆。关掉这个选项的话，我得明确告诉它“记住我更喜欢 Python 而不是 Node”。开着的时候，它会自己推断。我选择开着，虽然偶尔会出现一些无意义的记忆，但可以接受。
## `channels`——后面再接入

默认是空的。后面会接入 Telegram。结构大致如下：

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "...",
    "allowed_user_ids": [123456789]
  }
}
```

`allowed_user_ids` 这个字段实际上不能省略——没有它，任何找到你机器人的人都能用。别忘了加上。
## `cron`——定时任务

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

这个功能让 OpenClaw 从一个“聊天机器人”变成了“默默帮你干活的助手”。每条 cron 配置会触发一个技能，技能运行后生成输出，输出会被发送到默认频道，或者技能指定的其他频道。早间简报的功能就是基于这个实现的。
## 编辑后重新加载

编辑 `openclaw.json` 文件时，大部分内容会在保存后自动热重载。但渠道注册和 Provider Key 是例外，需要重启 Gateway：

```bash
openclaw gateway restart
```

如果修改后遇到“tool not registered”或“provider not found”的提示，那就是在提醒我需要重启了。
## 今天我会调整的几项配置

如果你用向导初始化后就没动过这个文件：

1. 把 `tools.write.allowed_paths` 改成明确的路径。默认是整个 home 目录，范围太大了。
2. 将 `memory.max_tokens_per_turn` 设置为 2000。
3. 如果每月的 token 开支接近 200 元，换成 Bailian Coding Plan。

接下来是渠道部分。先搞定 Telegram，因为它很简单。然后用一张截图快速过一遍 WeChat WorkBuddy——它值得单独花一天时间研究。
