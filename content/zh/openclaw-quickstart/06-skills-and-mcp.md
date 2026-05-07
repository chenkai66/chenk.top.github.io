---
title: "OpenClaw 快速上手（六）：Skills、MCP、以及交付一个真正有用的东西"
date: 2026-04-08 09:00:00
tags:
  - openclaw
  - skills
  - mcp
  - playwright
  - cron
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 6
description: "写第一个 Skill，接一个 MCP 浏览器自动化服务器，配 cron 跑起来，落地一个真正的早间简报 Agent——也就是这套安装从 Demo 变成“少了会想念”的那一刻。"
disableNunjucks: true
translationKey: "openclaw-quickstart-6"
---
写到第五篇，你已经搭好了一个能用的 OpenClaw，还配上了聊天频道。到这里，它不再是个简单的 Demo 了。

我判断一个 Agent 是否真正“立得住”，标准很简单：**关掉它，我会不会想念它**。只会聊天的不算，能回答问题的也不算。关键在于，它能不能在我还没想到的时候主动出现，替我完成每天必做的那件事。这才是分水岭。这一章的目标，就是带你迈过这个坎。
![OpenClaw 快速上手（六）：Skills、MCP、以及交付一个真正有用的东西 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_1.jpg)

## 要实现的目标

一个早间简报助手：

1. 每个工作日早上 7 点运行
2. 抓取 Hacker News 的头条新闻（通过 Playwright MCP 服务器）
3. 查看当天的日程安排（通过一个封装了 `gcalcli` 的 Skill）
4. 将两部分内容汇总成一段文字，发送到我的 Telegram

这是一套真实的工作流程。完成之后，你可以轻松替换数据源，比如把 Hacker News 换成 InfoQ，或者把 gcalcli 换成 Outlook。

动手之前，先搞清楚 Skills 的本质：**Skill 不是函数、不是插件、也不是脚本，而是给模型看的标准操作流程（SOP）**。它解决的问题是："模型已经会用工具了，但它不知道在什么场景下调用哪个工具、按什么顺序调用、输出应该是什么样子"。Skill 把你脑子里的隐性流程写下来，明确告诉模型："这件事就这么办。"
## 第一步：编写一个 Skill

每个 Skill 都存放在 `~/.openclaw/skills/<name>/SKILL.md`。现在来写一个叫“总结头条”的 Skill：

```bash
mkdir -p ~/.openclaw/skills/summarize-headlines
```

创建文件 `~/.openclaw/skills/summarize-headlines/SKILL.md`，内容如下：

```markdown
---
name: summarize-headlines
description: 把一组头条新闻总结成一段简报
trigger: 用户请求新闻简报、头条总结或每日新闻摘要时触发
tools_required: [web_search]
---

# 总结头条

你会收到一组头条新闻及其来源 URL。
生成一段总结性的文字。
```
## 规则
- 每次最多写 4 句话。
- 相关的头条内容合并成一句。
- 标题模糊或者有付费墙的直接跳过。
- 最重要的信息放前面，别管时间顺序。
- 语气要平实、分析为主，别用"激动人心"或"重大突破"这种词。
## 输出模板
> [最多 4 句]
>
> _来源：[domain1], [domain2], [domain3]_
```

两个设计要点：

- **trigger** 是模型判断是否使用该技能时看到的内容。要站在用户角度写，而不是实现角度。比如，不要写“调用新闻汇总函数时触发”，而是写“当用户需要新闻简报时触发”。
- **body** 是标准操作流程（SOP）。把它当作给新人写的入职文档——包含示例、边界情况和输出格式。越具体越好。我的经验是，给出一个反面例子往往比十个正面例子更有用。模型容易陷入平庸，明确告诉它“不要这样”反而能让它表现更好。

重启 Gateway 并验证 Skill 是否加载成功：

```bash
openclaw skills list | grep summarize
## summarize-headlines  (loaded)
```

如果没加载出来，大概率是 front-matter 里的 `name` 字段和目录名不一致，或者 YAML 缩进有问题。OpenClaw 对这块的容错性较差，少一个引号就会静默跳过。建议检查一遍 `openclaw doctor` 的 skills 部分。
## 第二步：连接 MCP 服务器

OpenClaw 本身不支持 MCP 协议，需要借助 `MCPorter` 作为中间层。先安装它：

```bash
npm i -g mcporter
curl -LsSf https://astral.sh/uv/install.sh | sh   # 安装 uvx，某些 MCP 服务器会用到
```

接着通过 MCPorter 添加 Playwright：

```bash
mcporter add playwright npx @playwright/mcp@latest
```

然后在 `openclaw.json` 中配置 MCPorter 的信息：

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright"]
}
```

重启 Gateway。Playwright MCP 提供了浏览器自动化功能（`navigate`、`screenshot`、`click`、`extract_text`），现在 Agent 可以直接调用这些工具。

在 TUI 中测试一下：

```
用 Playwright 打开 https://news.ycombinator.com，
提取前 5 条故事的标题和链接。
```

如果 Agent 返回了一个列表，说明链路已经通了。

为什么需要绕道 MCPorter？因为 MCP 协议的传输层有多种实现，Gateway 直连容易碰到各种握手细节问题。MCPorter 屏蔽了这些差异，统一暴露一个 HTTP 端点给 OpenClaw。虽然多了一个进程，但好处也很明显：添加新的 MCP 服务器（比如 GitHub MCP 或文件系统 MCP）时，只需运行一句 `mcporter add` 就能搞定。
## 第三步：封装一个 CLI 工具的 Skill

我用 `gcalcli` 来管理日历。下面是创建 Skill 的步骤：

```bash
mkdir -p ~/.openclaw/skills/today-calendar
```

在 `~/.openclaw/skills/today-calendar/SKILL.md` 文件中写入以下内容：

```markdown
---
name: today-calendar
description: 获取用户当天的 Google Calendar 事件
trigger: 用户询问今天的日历、今天的会议或今天的日程安排时触发
tools_required: [exec]
---

## 今天的日历

运行命令 `gcalcli agenda --tsv "$(date +%F) 00:00" "$(date +%F) 23:59"`。

解析 TSV 格式的输出，列包括：开始时间、结束时间、标题、地点。

按照以下格式生成 Markdown 列表：
- HH:MM–HH:MM **标题**（如果有地点，则加上地点）

如果当天没有任何事件，返回“今天没有安排”。
```

需要注意的是，这个 Skill 更像是一份操作指南，而不是一个函数。模型充当了运行时的角色，`exec` 是执行动作的核心工具，而 Skill 则是将它们串联起来的知识载体。

我在第三篇架构文章里提到过这一点，现在实际操作一遍就能更直观地感受到——同一个 Skill，换一个模型也能正常运行。原因很简单，流程逻辑写在 Markdown 文件里，而不是硬编码在程序中。
## 第四步：一个组合技能

![OpenClaw 快速上手（六）：Skills、MCP、以及交付一个真正有用的东西 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_2.jpg)

接下来写一个结合前两个技能的例子：

```bash
mkdir -p ~/.openclaw/skills/morning-briefing
```

`~/.openclaw/skills/morning-briefing/SKILL.md`：

```markdown
---
name: morning-briefing
description: 生成每日早间简报
trigger: 用户请求早间简报、每日简报或 7 点报告时触发
tools_required: [exec]
skills_required: [today-calendar, summarize-headlines]
---

## 早间简报

1. 调用 `today-calendar` 技能获取今天的日程安排。
2. 使用 Playwright MCP 工具从 https://news.ycombinator.com 抓取前 5 条新闻。
3. 对这 5 条新闻使用 `summarize-headlines` 技能生成摘要。
4. 按以下格式整理最终消息：
```
## 早间简报 — YYYY-MM-DD

### 今天
[today-calendar 的输出]

### 新闻
[summarize-headlines 的输出]
```

将结果发送到默认频道。
```

`skills_required` 字段的作用是告诉 OpenClaw，当这个技能触发时，直接热加载相关技能的内容，避免重新获取数据，也省去了额外的延迟。
## 第五步：Cron

在 `openclaw.json` 文件中：

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

`0 7 * * 1-5` 表示周一到周五早上 7 点。重启 Gateway 后，用以下命令验证：

```bash
openclaw cron list
## morning-briefing | 0 7 * * 1-5 | next: 明天 07:00 | channel: telegram
```

第一次运行时，盯着 Gateway 的日志看。你会看到 Agent 循环启动、Skill 加载完成、Playwright 工具调用快速滚动，最后消息会出现在你的 Telegram 中。

不过第一次跑的时候，大概率会遇到一些小问题。比如 Playwright 首次启动需要下载浏览器内核，可能会超时；或者 gcalcli 需要在服务器端完成 OAuth 授权。这时可以把 Cron 表达式改成 `*/5 * * * *`，让它每 5 分钟运行一次，方便快速调试。等流程跑通了，再改回 `0 7 * * 1-5`。
## 你现在有了什么

- 一个能长期运行并连接真实聊天平台的 Agent
- 把领域知识从 Agent 循环中独立出来的 Skills
- 一台提供 OpenClaw 原生不具备能力的 MCP 服务器
- 一条 cron 任务，把“我得主动问”变成“它自动出现”

这就是整个流程。官方文档里的其他案例——第二大脑、内容流水线、运维自动化——全都是这五个步骤的不同变体。换个 Skill、换个 MCP、改条 cron 就行了。
## 接下来我会做什么

按工作量从小到大排序：

1. **增加反馈回路**  
   早上看简报时，直接回复需要调整的内容，比如“跳过加密货币头条”。写一个 Skill，把这些调整保存到 `~/.openclaw/memory/feedback/morning-briefing.md`。第二天的简报会自动应用这些反馈。

2. **让新闻源可配置**  
   写一个 Skill，从 `~/openclaw-workspace/sources.yaml` 读取数据并迭代处理。这样基本就实现了一个“作为 Agent 的 RSS 阅读器”，几乎不用额外花力气。

3. **接入第二个渠道**  
   同一个 Agent，工作时间也放到钉钉上用。Skills 不用改。

QuickStart 就写到这里。官方文档的其他部分会对每一层深入讲解，现在你已经有一张清晰的地图，可以按需探索。

如果只能记住一点，那就是：真正有价值的地方在那些看似枯燥的部分——Skills、Memory 和 Channels。Agent 循环大家都差不多，让你的安装脱颖而出的是你构建的技能库，以及你选择接入的渠道。祝好运！
