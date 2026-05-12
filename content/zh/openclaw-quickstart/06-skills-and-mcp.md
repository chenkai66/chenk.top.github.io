---
title: "OpenClaw 指南（六）：技能系统与 MCP 落地"
date: 2026-04-13 09:00:00
tags:
  - openclaw
  - skills
  - mcp
  - playwright
  - cron
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 6
description: "Skill 不是 Prompt 模板——它是一套完整的 SOP，包括触发条件、工具权限和执行流程。再加上 MCP 把外部能力接入 Agent，从浏览器自动化到数据库查询，一个配置文件搞定。"
disableNunjucks: true
translationKey: "openclaw-quickstart-6"
---
学到第五篇，你的 OpenClaw 已经可以正常运行并支持对话了。从这一步起，它就不再只是一个演示原型（Demo）了。

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_1.png)

## 我们要做什么

![Skill composition pipeline — from trigger to tool execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/fig_skills.png)

做一个晨间简报 Agent：

1. 每个工作日早 7 点自动运行
2. 抓取 Hacker News 头条（通过 Playwright MCP server）
3. 读取当天的日历安排（通过封装 `gcalcli` 的 Skill）
4. 把两者总结成一段话，推送到我的 Telegram

这才是一个真实的端到端流程。完成这一步后，你就获得了一个可复用的系统骨架，后续只需替换数据源即可。但在动手构建之前，我们需要先厘清所要整合的两个系统。

## Skills、Tools 与 MCP —— 心智模型

这三个术语常被混用，但它们本质不同：

| 概念 | 是什么 | 谁编写 | 何时加载 |
|------|--------|--------|----------|
| **Tool（工具）** | 一个动词：读取文件、执行命令、网页搜索等。具有类型化 Schema 和处理函数（handler function）。 | 框架作者 或 你（自定义工具） | 始终加载；模型每轮都可见完整工具列表 |
| **Skill（技能）** | 一种知识性名词：一份 Markdown 格式的标准操作流程（SOP），告诉 Agent *如何* 完成特定任务。 | 你 | 懒加载 —— 仅在触发时才实例化，正文按需载入 |
| **MCP Server（MCP 服务端）** | 一个外部进程，通过 Model Context Protocol（MCP）暴露 *额外* 的工具。 | 第三方 或 你 | 网关启动时加载；其工具与内置工具并列呈现 |

三者关系：**Skills 使用 Tools；MCP Servers 提供 Tools**。例如某 Skill 可能写道："使用 Playwright 工具抓取该网页"——其中 Playwright 工具来自某个 MCP Server，而该 Skill 则指导 Agent 如何组合运用这些工具。

类比理解：Tools 是 Agent 的「双手」，Skills 是「操作手册」，MCP 则是为 Agent「增配更多双手」的机制。

## 第一步：写一个 Skill

Skill 存放在 `~/.openclaw/skills/<name>/SKILL.md`。我们先写一个负责"总结头条"的：

```bash
mkdir -p ~/.openclaw/skills/summarize-headlines
```

创建 `~/.openclaw/skills/summarize-headlines/SKILL.md`：

```markdown
---
name: summarize-headlines
description: Summarize a list of headlines into a one-paragraph briefing
trigger: when user asks for a news briefing, headline summary, or daily news digest
tools_required: [web_search]
---

# Summarize Headlines

You have been given a list of headlines and source URLs.
Produce a single paragraph summary.

## Rules
- Maximum 4 sentences.
- Group related headlines into a single sentence.
- If a headline is paywalled or the title is unclear, skip it.
- Lead with the highest-signal item, not the chronological first.
- Tone: dry, analytical. No "exciting!" or "breaking!".

## Output template
> [4 sentences max]
>
> _Sources: [domain1], [domain2], [domain3]_
```

### SKILL.md 文件结构解析

该文件包含两个部分：YAML 前置元数据（即 **manifest**）和 Markdown 正文（即 **SOP**）。二者均不可或缺，且在不同阶段承担不同职责。

**Manifest** 在 gateway 启动时加载。每个 skill 的 manifest 均被注入系统 prompt，供模型判断应调用哪个 skill。各字段说明如下：

| 字段 | 是否必需 | 用途 | 示例 |
|------|----------|------|------|
| `name` | 是 | 唯一标识符，用于日志记录与跨引用。 | `summarize-headlines` |
| `description` | 是 | 单行摘要，模型据此判断技能相关性。 | `Summarize a list of headlines...` |
| `trigger` | 是 | 自然语言触发条件。请从用户视角撰写，而非实现视角。 | `when user asks for a news briefing` |
| `tools_required` | 否 | 本 skill 所需的 tool 列表。若声明，gateway 将预先授权。 | `[web_search, exec]` |
| `skills_required` | 否 | 本 skill 所依赖的其他 skill。触发时，其 body 将被热加载。 | `[today-calendar]` |
| `priority` | 否 | 取值为 `high`、`normal` 或 `low`，用于多 skill 匹配时的优先级裁决；默认为 `normal`。 | `high` |
| `version` | 否 | 语义化版本号（Semver），仅作信息参考，便于 skill 分享与协作。 | `1.0.0` |

**Body** 仅在模型触发该 skill 时加载，即标准操作流程（SOP）——含指令、模板、边界情况处理及输出格式规范。可将其视作新员工入职文档：越具体，效果越好。模糊指令（如 "summarize well"）将导致不可控输出；而明确指令（如 "最多 4 句，首句必须为信号最强项，跳过付费墙内容"）则保障结果一致性。

### 如何编写高效的 trigger

`trigger` 是整个 skill 中最关键的一行。若过于宽泛，skill 将误触发；若过于狭窄，则可能完全不触发。常见模式如下：

**低效 trigger 示例：**
- `when the user asks about news` —— 过于宽泛，会在 "what's new in the codebase" 等非新闻场景下误触发
- `when summarize-headlines should run` —— 循环定义，对模型无意义

**高效 trigger 示例：**
- `when user asks for a news briefing, headline summary, or daily news digest` —— 明确列举具体名词
- `when user asks to take meeting notes or document a meeting` —— 动作 + 领域组合

**调试 trigger 问题：** 若 skill 未按预期触发，请设置环境变量 `OPENCLAW_LOG=debug` 并发送测试消息。检查 `gateway.log` 中的 `skill_selection` 日志条目——它会清晰列出模型评估了哪些 skill，以及最终选择（或未选择）某 skill 的原因。

重启网关，确认 Skill 加载成功：

```bash
openclaw skills list | grep summarize
# summarize-headlines  (loaded)
```

你还可以查看模型所见内容：

```bash
openclaw skills inspect summarize-headlines
# Name:        summarize-headlines
# Description: Summarize a list of headlines into a one-paragraph briefing
# Trigger:     when user asks for a news briefing, headline summary, or daily news digest
# Tools:       web_search
# Body:        287 chars (loaded on trigger)
```

## 第二步：挂载 MCP server

MCP（Model Context Protocol）是一种将大语言模型连接至外部工具服务器的标准协议。OpenClaw 本身不原生支持 MCP，而是通过 `MCPorter` 作为适配层（shim），在 OpenClaw 内部的 tool 格式与 MCP 协议之间进行双向转换。

### 安装 MCPorter

```bash
npm i -g mcporter
curl -LsSf https://astral.sh/uv/install.sh | sh   # 用于部分 MCP 服务器的 uvx 运行时
```

验证安装：

```bash
mcporter --version
# mcporter v0.4.2
```

### 添加 Playwright 作为 MCP server

```bash
mcporter add playwright npx @playwright/mcp@latest
```

该命令告知 MCPorter："存在一个名为 `playwright` 的 MCP 服务器，启动方式为执行 `npx @playwright/mcp@latest`。" MCPorter 将自动拉起该进程并管理其生命周期。

接着，在 `openclaw.json` 中配置 OpenClaw 使用 MCPorter：

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright"]
}
```

重启网关。此时 Playwright MCP server 所暴露的浏览器自动化 tool 即可被 Agent 调用。可用 tool 如下：

| MCP Tool | 功能说明 | 典型用途 |
|----------|----------|----------|
| `browser_navigate` | 导航至指定 URL | 打开页面以进行爬取 |
| `browser_snapshot` | 获取当前页面的可访问性树 | 读取结构化页面内容 |
| `browser_click` | 点击指定元素 | 驱动多页流程（分步表单、分页导航） |
| `browser_type` | 向输入框中键入文本 | 表单填写、搜索框输入 |
| `browser_evaluate` | 在页面上下文中执行任意 JavaScript | 提取 DOM 树未覆盖的数据 |
| `browser_take_screenshot` | 截取当前视口图像 | 可视化验证、调试排查 |

### 测试 MCP 连接

在 TUI 里测试一下：

```
Use Playwright to fetch the top 5 stories from
https://news.ycombinator.com and just give me the titles and URLs.
```

如果 Agent 返回了一个列表，链路就通了。若失败，常见原因如下：

1. **MCPorter 未运行**：运行 `mcporter status`，确认 `playwright` 显示为 `running`。若为 `stopped`，手动执行 `mcporter start playwright`，并检查日志 `~/.mcporter/logs/playwright.log` 排查错误。
2. **端口冲突**：MCPorter 默认监听 `:7890`。若该端口被占用，可设置环境变量 `MCPORTER_PORT=7891`，并同步更新 `openclaw.json` 中的 `porter_endpoint`。
3. **Playwright 浏览器未安装**：首次运行 `npx @playwright/mcp@latest` 会自动下载 Chromium 等浏览器，耗时约 2–3 分钟、占用约 400MB 磁盘空间。若中途中断，请手动运行 `npx playwright install chromium` 补全安装。

### 添加其他 MCP server

该模式适用于任意符合 MCP 规范的服务器。以下为几个常用示例：

```bash
# 文件系统 MCP — 允许 agent 在沙箱外读写文件
mcporter add filesystem npx @anthropic/mcp-filesystem@latest /path/to/allowed/dir

# GitHub MCP — 支持 PR 审查、Issue 分类等
mcporter add github npx @anthropic/mcp-github@latest

# SQLite MCP — 直接查询本地 SQLite 数据库
mcporter add sqlite npx @anthropic/mcp-sqlite@latest /path/to/database.db
```

添加后，更新 `openclaw.json`：

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright", "filesystem", "github"]
}
```

每个 MCP server 所注册的 tool 均会与 OpenClaw 内置 tool 一同出现在 Agent 的 tool 列表中，模型对其调用方式完全一致，无需区分来源。

## 第三步：封装 CLI 工具的 Skill

并非所有集成都需要 MCP server。对简单的 CLI 工具来说，写一个用 `exec` tool 的 Skill 通常更简洁。我用 `gcalcli` 管理日历：

```bash
mkdir -p ~/.openclaw/skills/today-calendar
```

`~/.openclaw/skills/today-calendar/SKILL.md`：

```markdown
---
name: today-calendar
description: Fetch today's Google Calendar events for the user
trigger: when user asks for today's calendar, today's meetings, today's schedule
tools_required: [exec]
---

# Today's Calendar

Run the command `gcalcli agenda --tsv "$(date +%F) 00:00" "$(date +%F) 23:59"`.

Parse the TSV output (columns: start, end, title, location).

Format as a markdown bullet list:
- HH:MM–HH:MM **Title** (Location, if any)

If there are no events, return "Nothing on the calendar today".
```

注意，这个 Skill 本质上是一份食谱（recipe），不是函数。模型是运行时，`exec` 工具是动词，Skill 则是把两者绑在一起的知识名词。

### 何时使用 `exec` vs. MCP

决策非常明确：

| 场景 | 使用 `exec` | 使用 MCP |
|------|-------------|----------|
| 一次性 CLI 命令 | 是 | 过度设计 |
| 有状态交互（浏览器、数据库） | 否 | 是 |
| 输出结构复杂、类型明确的工具 | 可能 | 首选 |
| 需跨项目复用的工具 | 否 | 是 |
| 快速原型开发 | 是 | 否 |

经验法则：若交互模式是「执行命令 → 解析 stdout」，用 `exec`；若涉及多轮往返或需维持持久状态，则应使用 MCP server。

### `exec` 类 Skill 的安全注意事项

`exec` tool 被标记为 `dangerous` 权限级别，原因即在于此。当 Skill 使用 `exec` 时，请务必考虑以下几点：

1. **固定命令字面量。** 在 Skill 正文中硬编码具体命令，而非写成"运行任意所需 shell 命令"。若留出自由发挥空间，模型可能擅自构造危险命令。
2. **在配置中声明 `trusted_commands`。** 将该 Skill 实际使用的命令显式加入可信列表，避免用户每次调用都收到确认提示：
   ```json
   "tools": {
     "exec": {
       "trusted_commands": ["gcalcli agenda", "gcalcli list", "git status"]
     }
   }
   ```
3. **校验输出内容。** 若 Skill 会解析工具输出并将其传回模型，需预判异常输出的影响——例如，恶意构造的日历事件标题理论上可能注入指令。

## 第四步：组合 Skill

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_2.png)

Skill 真正强大的能力在这里浮现：组合型 Skill 把多个 Skill 与 tool 编排成一条完整的多步工作流。

```bash
mkdir -p ~/.openclaw/skills/morning-briefing
```

`~/.openclaw/skills/morning-briefing/SKILL.md`：

```markdown
---
name: morning-briefing
description: Generate the daily morning briefing
trigger: when user asks for the morning briefing, daily briefing, or 7am report
tools_required: [exec]
skills_required: [today-calendar, summarize-headlines]
---

# Morning Briefing

1. Use the `today-calendar` skill to get today's schedule.
2. Use the Playwright MCP tools to fetch the top 5 stories from
   https://news.ycombinator.com.
3. Use the `summarize-headlines` skill on those stories.
4. Compose the final message in this format:

```
## Morning Briefing — YYYY-MM-DD

### Today
[output of today-calendar]

### News
[output of summarize-headlines]
```

Send the result to the default channel.
```

`skills_required` 字段告诉 OpenClaw，当这个 Skill 触发时，要把那些 Skill 的内容预加载进来。不用重新 fetch，也没有额外延迟。这是一个关键优化——若没有它，Agent 必须每次单独触发各个子 Skill，每次都要付出 manifest 查询的成本。

### Skill 组合的内部机制

当 `morning-briefing` 触发时，gateway 执行以下操作：

1. 将 `morning-briefing` 的 body 载入 prompt；
2. 发现 `skills_required: [today-calendar, summarize-headlines]`；
3. 把这两个子 Skill 的 body 一并载入 prompt，与父 Skill 并列；
4. 此时模型同时拥有全部三个 SOP，可按序执行各步骤。

Agent 循环照常运行：模型规划 tool 调用 → gateway 执行 → 结果返回。关键区别在于：因三个 Skill body 均在 prompt 中，模型对**每个 tool 的用途和上下文**理解更充分。

这与传统代码中的函数组合不同——没有调用栈，没有显式返回值。模型通读全部三个 SOP，内化其逻辑，再自主生成并执行融合后的完整计划。该机制可行的根本原因，是大语言模型擅长遵循多步指令。

### 调试组合型 Skill

最常见的失败现象：模型跳过某一步骤。例如成功获取了新闻，却遗漏了日历查询（或反之）。这通常是 prompt 过长、模型上下文丢失导致的。

诊断方法：检查逐轮 JSON 日志：

```bash
cat ~/.openclaw/sessions/<session_id>.jsonl | jq '.tool_calls[].name'
```

若输出含 `browser_navigate` 但无 `exec`（对应 `gcalcli` 调用），即表明日历步骤被跳过。修复方案：

1. **显式编号所有步骤**（如上例所示）；
2. **在 Skill body 末尾添加核对清单**："发送前请确认：日历内容已包含？新闻摘要已包含？日期是否正确？"
3. **为该 Skill 降低 `max_turns`**，防止循环失控。在 `openclaw.json` 中配置：
   ```json
   "skill_config": {
     "morning-briefing": {
       "max_turns": 15
     }
   }
   ```

## 第五步：Cron

Skill 真正变得有用，是它能在你不在场时自动运行。在 `openclaw.json` 里配置：

```json
"cron": {
  "jobs": [
    {
      "name": "morning-briefing",
      "schedule": "0 7 * * 1-5",
      "skill": "morning-briefing",
      "channel": "telegram"
    }
  ]
}
```

`0 7 * * 1-5` 代表周一到周五早 7 点。重启网关。用以下命令验证：

```bash
openclaw cron list
# morning-briefing | 0 7 * * 1-5 | next: tomorrow 07:00 | channel: telegram
```

首次运行时，请观察网关日志：你会看到 Agent 循环触发、Skill 加载、Playwright 工具执行滚动操作，最终消息推送至你的 Telegram。

### Cron 配置参考

| 字段 | 是否必需 | 类型 | 说明 |
|------|----------|------|------|
| `name` | 是 | string | 唯一任务名称，用于日志及 `openclaw cron list`。 |
| `schedule` | 是 | cron 表达式 | 标准 5 字段 cron 表达式。 |
| `skill` | 是 | string | 触发的 Skill 名称，必须存在于 `~/.openclaw/skills/`。 |
| `channel` | 是 | string | 输出发送的目标 channel，必须为已配置的 channel。 |
| `user_id` | 否 | string | 使用哪位用户的 memory / context。默认为 admin 用户。 |
| `timeout_sec` | 否 | integer | 最大执行时长（秒），默认值：120。 |
| `retry` | 否 | integer | 失败时重试次数，默认值：0。 |
| `env` | 否 | object | 该任务运行时传递给 tool 的额外环境变量。 |

### Cron 下的限制与注意事项

Cron 任务在模拟会话（synthetic session）中运行 —— 无真实用户参与。这带来两点关键影响：

1. **无确认提示**：若 Skill 使用 `exec` 且所执行命令未列入 `trusted_commands`，cron 任务将因等待永不出现的用户确认而挂起。请务必把所有 cron 触发的命令加入可信命令列表。
2. **无后续交互**：若模型响应中包含提问（例如："是否包含加密货币新闻？"），无人可作答；消息发布至 channel 后即结束会话。因此 cron 触发的 Skill 必须自包含 —— 所有决策应由 SOP 明确定义，不可依赖用户实时交互。

调试 cron 任务的实用技巧：

```bash
# 手动执行 Cron 任务的干运行（dry-run）
openclaw cron run morning-briefing --dry-run

# 查看最近 5 次该 Cron 任务的执行历史
openclaw cron history morning-briefing --limit 5
```

## 真实场景下的 Skill 示例

晨间简报是一个入门级项目。以下是我在实践中常用的另外三种模式，附关键设计考量。

### 模式 1：Git 周变更日志

一个从 git 提交记录生成周度变更日志的 Skill：

```markdown
---
name: weekly-changelog
trigger: when user asks for changelog, release notes, or weekly summary
tools_required: [exec, write]
---

在项目目录中运行 `git log --oneline --since="7 days ago"`。

按前缀分组提交（feat:, fix:, refactor:, docs:），生成 Markdown：
- "## 本周变更概览"
- 每组一个二级标题，含计数：`### Features (3)`
- 每条提交作为列表项

写入项目根目录下的 `CHANGELOG-weekly.md`。
```

设计理由：输入明确（git log 输出）、转换清晰（按 prefix 分组）、输出确定（固定路径的 Markdown 文件），无歧义。

### 模式 2：数据库健康检查

一个基于 SQLite MCP server 的周期性健康检查 Skill：

```markdown
---
name: db-health
trigger: when user asks to check database health, db stats, or table sizes
tools_required: [sqlite_query]
---

依次执行以下查询：
1. `SELECT name, SUM(pgsize) as size_bytes FROM dbstat GROUP BY name ORDER BY size_bytes DESC LIMIT 10;`
2. `SELECT COUNT(*) as total_rows FROM main_table;`
3. `SELECT created_at FROM main_table ORDER BY created_at DESC LIMIT 1;`

格式化为健康报告：
- 按大小排序的 Top 10 表（单位转为 KB/MB，人类可读）
- 总行数
- 最近插入时间戳
- 标记任何超过 100MB 的表为 "large"
```

### 模式 3：PR 审查助手

一个融合 GitHub MCP 与代码分析能力的 Skill：

```markdown
---
name: pr-review
trigger: when user asks to review a PR, check a pull request, or code review
tools_required: [exec, web_fetch]
skills_required: []
---

1. 使用 `gh pr diff <number>` 获取 PR 差异。
2. 针对每个变更文件：
   - 读取完整文件以获取上下文
   - 提示：新增行数 > 200 行时建议拆分该文件
3. 检查以下问题：
   - 缺失错误处理（IO 操作未包裹 try/catch）
   - 硬编码敏感信息（API keys、密码等）
   - 存在 TODO/FIXME 但无对应 issue 链接
4. 使用 `gh pr review <number> --comment --body "..."` 发送内联评论。
```

## 故障排查

这是我上线首月遇到的主要问题及解决方式。

### Skill 触发成功，但输出错误

**现象**：Skill 正常触发，但 Agent 忽略了 SOP 中约一半的指令。

**原因**：Skill 正文过长或表述模糊。模型会像人类一样跳读长文本。

**解决**：正文控制在 500 字以内；用编号步骤替代段落叙述；将输出模板置于末尾（利用 recency bias）；若逻辑必须复杂，请拆分为多个子 Skill，并通过 `skills_required` 显式编排。

### MCP server 在对话中途崩溃

**现象**：Agent 执行工作流中途报错 `connection refused`。

**原因**：MCP server 进程意外退出。Playwright 尤其容易因页面触发 OOM、弹出无法处理的下载对话框而崩溃。

**解决**：MCPorter 支持 auto-restart，请确认已启用：

```bash
mcporter config playwright
# auto_restart: true (default)
# max_restarts: 3
# restart_delay_ms: 1000
```

若 60 秒内崩溃次数超过 `max_restarts`，MCPorter 将放弃重试并记录错误。请检查 `~/.mcporter/logs/playwright.log` 定位原因。常见诱因：页面触发无限 JS 循环，或加载超 100MB 的资源。

### Cron 任务显示执行成功，但未发送消息

**现象**：`openclaw cron history` 显示任务成功运行，但目标 channel 无任何消息。

**原因**：通常为 channel 认证失效 —— OAuth token 过期、钉钉 Webhook URL 轮转、Telegram Bot Token 被撤销等。

**解决**：查看 `gateway.log` 中对应 cron 时间戳，搜索 `channel send failed` 错误。重新认证 channel：`openclaw channel test telegram` 会发送测试消息并报告所有 auth 异常。

### 多个 Skill 抢占同一 trigger

**现象**：同一用户消息下行为不一致 —— 有时 Skill A 触发，有时 Skill B 触发。

**原因**：trigger 描述存在重叠。模型会依据细微措辞差异随机选择其一。

**解决**：确保 trigger 互斥。例如，若同时存在 "meeting notes" 和 "project notes" Skill，**切勿**对二者都写 `when user asks about notes`。应改为：
- `when user asks about meeting notes or documenting a meeting`
- `when user asks about project notes or project status`

你也可强制指定 Skill：

```
/skill morning-briefing
```

该命令完全绕过 trigger 匹配机制。

## 你现在拥有了什么

- 一个长驻 Agent 连着真实的聊天平台
- 把领域知识和 Agent 循环分离的 Skills
- 一个提供 OpenClaw 原生没有的能力的 MCP server
- 一个把"我得去问"变成"它自己来"的 Cron 任务
- 一套出问题时帮你定位原因的调试工具

这就是完整闭环。官方文档里的其他案例——第二大脑、内容 pipeline、DevOps 自动化——都是这五步的变体。换些 Skill，换些 MCPs，改几行 Cron 配置而已。

## 接下来我会做什么

按投入精力从小到大，三件事：

1. **加个反馈循环。** 回复晨间简报进行纠正（比如"跳过 crypto 头条"）。写个 Skill 把这些纠正记入 `~/.openclaw/memory/feedback/morning-briefing.md`。第二天早上的简报会读取这些内容。一周之后，简报会悄无声息地适配你的偏好——你完全不用动 SOP。
2. **让新闻源可配置。** 写个 Skill 从 `~/openclaw-workspace/sources.yaml` 读取并遍历。这几乎等于免费得了个"Agent 版 RSS 阅读器"。这份 YAML 文件就是简单的 UI——加一行 URL，第二天的简报就有了。
3. **接第二个渠道。** 同一个 Agent，工作时间也推送到钉钉。Skill 不用改——channel 层是解耦的。你可以同时在两个地方收到同样的简报，也可以根据上下文把不同的 Skill 路由到不同 channel。

QuickStart 到此结束。官方文档其余章节将深入讲解各层实现细节。现在你已掌握整体架构，并清楚如何在各模块之间切换与协作。

如果只记住一件事，请牢记：真正具有长期价值的是那些看似平凡的基础层——Skills、记忆机制（memory）和通信渠道（channels）。Agent 的核心循环逻辑大同小异；让整个系统真正落地可用的，是你构建的 Skill 库和接入的实际渠道。祝好运。
