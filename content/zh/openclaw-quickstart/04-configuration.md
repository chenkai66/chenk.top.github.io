---
title: "OpenClaw 快速上手（四）：配置文件、模型选择与百炼 Coding Plan 的窍门"
date: 2026-04-06 09:00:00
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

上一篇我们把 OpenClaw 跑起来了。但真正让 Agent 好用的关键不在代码，而在配置。今天把 `openclaw.json` 从头到尾拆一遍——模型怎么选、钱怎么省、安全边界怎么画。


![OpenClaw 配置文件结构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_1.jpg)

## 最小可用配置

![配置层级与 Provider 解析顺序](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/fig_config.png)

只是试试效果的话，这 25 行就够了：

```json
{
  "agent": {
    "name": "my-agent",
    "default_model": "qwen-plus"
  },
  "providers": {
    "dashscope": {
      "api_key": "${DASHSCOPE_API_KEY}",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
  },
  "tools": {
    "exec": { "enabled": true },
    "web_search": { "enabled": true, "engine": "quark" }
  },
  "memory": { "max_tokens_per_turn": 2000 },
  "security": { "allowed_paths": ["./"] }
}
```

走阿里云 DashScope 接口，国内零延迟、无需代理，新账号有免费额度。


## 完整配置参考

`openclaw.json` 有 9 个顶层字段（`agent` 和 `providers` 必填）：

| 字段 | 用途 | 必填 |
|------|------|------|
| `agent` | Agent 名称与默认模型 | 是 |
| `providers` | 模型服务商配置 | 是 |
| `tools` | 工具权限与参数 | 推荐 |
| `memory` | 记忆管理策略 | 推荐 |
| `channels` | 接入渠道（钉钉等） | 否 |
| `cron` | 定时任务 | 否 |
| `mcp` | MCP 服务器连接 | 否 |
| `hooks` | 生命周期钩子 | 否 |
| `security` | 安全约束 | 强烈推荐 |

## agent 部分

```json
{
  "agent": {
    "name": "code-helper",
    "default_model": "qwen-plus",
    "system_prompt_file": "./prompts/system.md",
    "max_turns": 50
  }
}
```

关键原则：**不要把最贵的模型放在 default_model 上**。Agent 绝大多数 turn 是常规对话和工具调用确认，qwen-plus（0.004 元/千 token）足够。强推理场景通过 model_override 在特定 skill 指定。我见过有人把 Claude 放 default_model 跑了一晚上定时任务，第二天账单 80 美元。

## providers 部分

OpenClaw 支持四种 provider，按国内优先级：

### DashScope（国内首选）

```json
"dashscope": {
  "type": "dashscope",
  "api_key": "${DASHSCOPE_API_KEY}",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "default_model": "qwen-plus",
  "timeout": 120
}
```

qwen-plus 0.004 元/千 token，qwen-turbo 0.002 元/千，qwen3-coder-plus 0.008 元/千（代码专用）。新开通有 100 万 token 免费额度。国内直连延迟低（北京区域 50ms 首包），推荐做 default。

### Anthropic（需代理或香港服务器）

```json
"anthropic": {
  "type": "anthropic",
  "api_key": "${ANTHROPIC_API_KEY}",
  "base_url": "https://api.anthropic.com",
  "default_model": "claude-sonnet-4-5",
  "proxy": "${HTTPS_PROXY}"
}
```

Claude Sonnet 4.5 复杂编程仍是第一梯队。$3/百万 token（约 0.022 元/千），加上汇率和代理实际约 qwen-plus 的 7-8 倍。适合跨文件复杂 bug、架构重构；简单 CRUD、文档润色不值得用。

### OpenAI

```json
"openai": {
  "type": "openai",
  "api_key": "${OPENAI_API_KEY}",
  "base_url": "https://api.openai.com/v1",
  "default_model": "gpt-5.4"
}
```

GPT-5.4 多语言翻译、结构化数据抽取有优势，纯代码不如 Claude。价格相当，网络问题也一样。有现成 key 和代理就配上，从零开始不推荐首选。

### compatible（本地模型）

```json
"local": {
  "type": "compatible",
  "base_url": "http://localhost:11434/v1",
  "api_key": "not-needed",
  "default_model": "qwen3:32b"
}
```

兼容所有 OpenAI API 格式的服务（Ollama、vLLM、内部模型服务）。零成本、数据不出内网，但能力天花板低，需要 GPU。

## 模型路由：让对的模型做对的事

skill 级别 `model_override` 让不同任务选最合适的模型：

```json
"skills": {
  "planning": { "model_override": "dashscope/qwen-plus" },
  "code_generation": { "model_override": "dashscope/qwen3-coder-plus" },
  "reasoning": { "model_override": "anthropic/claude-sonnet-4-5" },
  "bulk_processing": { "model_override": "dashscope/qwen-turbo" }
}
```

| 任务类型 | 模型 | 单价 | 理由 |
|----------|------|------|------|
| 规划/分解 | qwen-plus | 0.004 元/千 | 够用就行 |
| 代码生成 | qwen3-coder-plus | 0.008 元/千 | 代码专用，性价比最高 |
| 复杂推理 | claude-sonnet-4-5 | 0.022 元/千 | 只在真正需要时调用 |
| 批量处理 | qwen-turbo | 0.002 元/千 | 量大、质量要求不高 |

这个路由策略让我月均 token 费用从 300+ 元降到 60-80 元。

## 百炼 Coding Plan：重度用户的最优解

每天都用 OpenClaw 的话按量付费不划算。百炼 Coding Plan：**200 元/月，8 个模型无限调用**——qwen-plus、turbo、qwen3-coder-plus、qwen-max、qwen-long、qwen-vl-plus，以及通过百炼网关转发的 Claude Sonnet 4.5。没看错，200 元/月包含 Claude。阿里云和 Anthropic 合作，不需要自己的 Anthropic key，不需要代理，国内直连。

```json
"dashscope": {
  "type": "dashscope",
  "api_key": "${DASHSCOPE_API_KEY}",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "plan": "coding",
  "models": {
    "claude-sonnet-4-5": {
      "model_id": "claude-sonnet-4-5-20260301",
      "via_gateway": true
    }
  }
}
```

开通：百炼控制台 -> 资源包管理 -> 购买 Coding Plan，即时生效。**该开**：月费超 150 元、需频繁用 Claude、团队共用实例。**不该开**：月费低于 50 元、只跑 turbo 批量任务、需要 GPT-5.4（不包含 OpenAI）。

## tools 部分

```json
"tools": {
  "exec": { "enabled": true, "trusted_patterns": ["npm *", "git *", "python *", "node *"], "timeout": 30 },
  "web_search": { "enabled": true, "engine": "quark", "max_results": 5 },
  "read": { "enabled": true, "max_bytes": 102400 },
  "write": { "enabled": true, "allowed_paths": ["./src", "./docs", "./tests"] }
}
```

- `exec.trusted_patterns`：匹配的命令自动执行。加 npm/git/python/node，绝不加 rm/curl/sudo。
- `web_search.engine`：国内推荐 `quark`，`bing`/`google` 需代理。
- `read.max_bytes`：单次读取上限，100KB 够用。
- `write.allowed_paths`：Agent 只能写入这些路径。

## memory 部分

```json
"memory": {
  "max_tokens_per_turn": 2000,
  "auto_write": true,
  "storage": "local",
  "summarize_after": 20
}
```

`max_tokens_per_turn` 设太大浪费 token，太小 Agent "失忆"，2000 是平衡点。`auto_write` 建议开，每轮自动写入关键信息。`summarize_after` 超 20 轮自动压缩早期对话防溢出。

## security 部分

```json
"security": {
  "allowed_paths": ["/home/user/projects/my-app", "/tmp/openclaw"],
  "blocked_commands": ["rm -rf", "sudo", "curl|bash", "wget|sh", "chmod 777"],
  "require_confirmation": ["git push", "npm publish", "docker run"],
  "network_allowlist": ["*.aliyuncs.com", "registry.npmmirror.com", "github.com"]
}
```

- `allowed_paths`：**永远不要设成 `/`**。
- `blocked_commands`：`rm -rf`、`sudo`、`curl|bash` 必须 block。
- `require_confirmation`：有副作用的操作放这里，执行前等人工确认。
- `network_allowlist`：国内 `*.aliyuncs.com` + `registry.npmmirror.com` 够日常开发。

![安全配置与模型路由的关系](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/04-configuration/illustration_2.png)

## channels 部分（预告）

```json
"channels": {
  "dingtalk": {
    "enabled": true,
    "app_key": "${DINGTALK_APP_KEY}",
    "app_secret": "${DINGTALK_APP_SECRET}",
    "mode": "stream"
  }
}
```

让 Agent 通过钉钉、飞书、企业微信对外服务。下一篇专门讲。

## cron 部分

定时任务让 Agent 无人值守自动执行：

```json
"cron": {
  "weekly_report": {
    "schedule": "0 9 * * 1",
    "skill": "generate_report",
    "params": { "type": "weekly", "output": "./reports/" },
    "model_override": "dashscope/qwen-plus"
  },
  "competitor_monitor": {
    "schedule": "0 8 * * *",
    "skill": "monitor_competitors",
    "params": { "targets": ["competitor-a.com"], "notify": "dingtalk" },
    "model_override": "dashscope/qwen-turbo"
  },
  "dependency_audit": {
    "schedule": "0 3 * * 0",
    "skill": "audit_dependencies",
    "model_override": "dashscope/qwen-plus"
  }
}
```

周一早 9 点生成周报、每天早 8 点竞品监控推钉钉、周日凌晨依赖安全审计。cron 任务用 qwen-plus/turbo 就够。

## 环境变量

敏感信息用 `${VAR_NAME}` 引用，不要写死在配置里：

| 环境变量 | 用途 | 在哪里设置 |
|----------|------|------------|
| `DASHSCOPE_API_KEY` | 百炼 API Key | 百炼控制台 |
| `ANTHROPIC_API_KEY` | Anthropic API Key | console.anthropic.com |
| `OPENAI_API_KEY` | OpenAI API Key | platform.openai.com |
| `HTTPS_PROXY` | 代理地址 | 你的代理服务 |
| `OPENCLAW_HOME` | 数据目录（默认 `~/.openclaw`） | 自行设置 |
| `OPENCLAW_LOG_LEVEL` | 日志级别 | debug/info/warn/error |

写在 `.env` 文件并加入 `.gitignore`。

## 配置校验

用 `openclaw config validate` 检查配置。三个常见错误：

**1. 模型 ID 不存在** — DashScope 不带 `-latest` 后缀，直接 `qwen-plus`。**2. model_override 格式错** — 必须 `provider名/模型ID`（如 `anthropic/claude-sonnet-4-5`），不能只写模型名。**3. 相对路径缺基准目录** — `allowed_paths` 用 `./` 需设 `agent.project_root`，否则用绝对路径。

## 热重载与重启

**热重载（立即生效）：** `memory`、`tools` 参数调整、`cron` 时间表、`security` 列表。

**需重启：** `agent.default_model`、`providers` 增删、`channels`、`mcp` 配置。

启动时输出配置摘要：

```
$ openclaw start
[INFO] Agent: code-helper | Model: dashscope/qwen-plus
[INFO] Providers: dashscope, anthropic | Tools: exec, web_search, read, write
[INFO] Cron: 3 jobs | Security: 2 paths, 6 blocked cmds
```

## 配置策略总结

1. **default_model 用 qwen-plus** — 便宜、快、直连
2. **Coding Plan 开着** — 200 元/月包 Claude
3. **model_override 精细控制** — 只有需要 Claude 的 skill 才走 Claude
4. **security 从严** — 先锁死，需要时再放开
5. **npmmirror 加到 allowlist** — 国内装包必须走镜像

下一篇讲 Channels——让 Agent 在钉钉群里随叫随到。

---

**本系列：** [1. 什么是 OpenClaw](/posts/zh/openclaw-quickstart-1) | [2. 安装](/posts/zh/openclaw-quickstart-2) | [3. 第一个 Skill](/posts/zh/openclaw-quickstart-3) | **4. 配置（本文）** | [5. 渠道接入](/posts/zh/openclaw-quickstart-5)
