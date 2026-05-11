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

这才是一个真实的端到端流程。完成这一步后，你就获得了一个可复用的系统骨架，后续只需替换数据源即可。

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

这里有两个设计点：

- **trigger** 字段是模型决定要不要调用这个 Skill 的依据。站在用户角度写，别管实现细节。
- **body** 部分是 SOP。请按新员工入职文档的标准编写：包含调用示例、典型边界情况，以及严格定义的输出格式。细节越充分越好。

重启网关，确认 Skill 加载成功：

```bash
openclaw skills list | grep summarize
# summarize-headlines  (loaded)
```

## 第二步：挂载 MCP server

OpenClaw 不原生支持 MCP，我们用 `MCPorter` 做个 shim。先安装：

```bash
npm i -g mcporter
curl -LsSf https://astral.sh/uv/install.sh | sh   # for uvx, used by some MCP servers
```

然后通过 MCPorter 添加 Playwright：

```bash
mcporter add playwright npx @playwright/mcp@latest
```

在 `openclaw.json` 里告诉 OpenClaw 关于 MCPorter 的事：

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright"]
}
```

重启网关。Playwright MCP 会暴露浏览器自动化工具（`navigate`, `screenshot`, `click`, `extract_text`），Agent 现在可以调用它们了。

在 TUI 里测试一下：

```
Use Playwright to fetch the top 5 stories from
https://news.ycombinator.com and just give me the titles and URLs.
```

如果 Agent 返回了一个列表，链路就通了。

## 第三步：封装 CLI 工具的 Skill

我用 `gcalcli` 管理日历。写个 Skill：

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

## 第四步：组合 Skill

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_2.png)

现在写一个把上面两者用起来的 Skill：

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

`skills_required` 字段告诉 OpenClaw，当这个 Skill 触发时，要把那些 Skill 的内容预加载进来。不用重新 fetch，也没有额外延迟。

## 第五步：Cron

在 `openclaw.json` 里配置：

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

## 你现在拥有了什么

- 一个长驻 Agent 连着真实的聊天平台
- 把领域知识和 Agent 循环分离的 Skills
- 一个提供 OpenClaw 原生没有的能力的 MCP server
- 一个把"我得去问"变成"它自己来"的 Cron 任务

这就是完整闭环。官方文档里的其他案例——第二大脑、内容 pipeline、DevOps 自动化——都是这五步的变体。换些 Skill，换些 MCPs，改几行 Cron 配置而已。

## 接下来我会做什么

按投入精力从小到大，三件事：

1. **加个反馈循环。** 回复晨间简报进行纠正（比如"跳过 crypto 头条"）。写个 Skill 把这些纠正记入 `~/.openclaw/memory/feedback/morning-briefing.md`。第二天早上的简报会读取这些内容。
2. **让新闻源可配置。** 写个 Skill 从 `~/openclaw-workspace/sources.yaml` 读取并遍历。这几乎等于免费得了个"Agent 版 RSS 阅读器"。
3. **接第二个渠道。** 同一个 Agent，工作时间也推送到钉钉。Skill 不用改。

QuickStart 到此结束。官方文档其余章节将深入讲解各层实现细节。现在你已掌握整体架构，并清楚如何在各模块之间切换与协作。

如果只记住一件事，请牢记：真正具有长期价值的是那些看似平凡的基础层——Skills、记忆机制（memory）和通信渠道（channels）。Agent 的核心循环逻辑大同小异；让整个系统真正落地可用的，是你构建的 Skill 库和接入的实际渠道。祝好运。