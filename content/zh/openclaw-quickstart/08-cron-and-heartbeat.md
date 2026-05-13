---
title: "OpenClaw 指南（八）：心跳巡逻与定时任务"
date: 2026-04-15 09:00:00
tags:
  - openclaw
  - cron
  - heartbeat
  - automation
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 8
description: "OpenClaw 的两个调度原语——Heartbeat 是带脉搏的巡逻，Cron 是厨房计时器。每日简报、周报、竞品监控的完整配置，以及那个会让你凌晨三点被吵醒的反模式。"
disableNunjucks: true
translationKey: "openclaw-quickstart-8"
---
第一次部署 OpenClaw 时，我一直在主动发消息；两天后才意识到——它当时只是一个聊天机器人，远非真正的 Agent。真正的转折点，是它开始在没有被询问的情况下主动给我发消息。

本文将介绍实现这种主动行为的两种机制。

![OpenClaw 快速入门（8）：心跳、定时任务和早上7点的提醒 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.png)

## Heartbeat 还是 Cron —— 心里得有数

![心跳与定时任务调度模型比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/fig_hb_vs_cron.png)

两者都能调度任务，但本质不同。

| | Heartbeat | Cron |
|---|---|---|
| 触发条件 | 固定间隔（每 30 分钟） | 时钟时间（每天 8:30） |
| 默认行为 | “定期巡检，仅在发现问题时上报” | “严格按计划执行，确保结果产出” |
| 适用场景 | 捕获异常、发现遗漏任务 | 日常例行、定时报告 |
| 会话上下文 | 最新活跃渠道 | 默认隔离 |

> Heartbeat 像巡逻警察，Cron 像厨房定时器。

在生产环境中，我两者并用：Cron 负责每天早上 7 点的晨间简报和下午 5 点的收工总结，而 Heartbeat 则每 45 分钟巡逻一次，专门检查“有没有漏掉的 PR”。

## 开启功能

```json
{
  "tools": {
    "cron":    { "enabled": true },
    "message": { "enabled": true }
  }
}
```

注意 `message` 这个工具——没有它，Agent 就无法主动向外发送任何消息。

## Heartbeat —— 巡逻机制

```json5
{
  agents: {
    defaults: {
      heartbeat: {
        every: "45m",
        target: "last",
        prompt: "Read HEARTBEAT.md if it exists. Follow it strictly. If nothing needs attention, reply HEARTBEAT_OK.",
        activeHours: {
          start: "09:00",
          end:   "22:00",
          timezone: "Asia/Shanghai"
        }
      }
    }
  }
}
```

关键在于 `HEARTBEAT_OK` 这个约定。你想要的是*安静*的巡逻。默认提示词明确告诉 Agent：除非有事需要你关注，否则保持沉默。如果没有这一行，你每 45 分钟就会收到一条“一切正常”，听起来就烦，实际更烦。

具体的巡逻规则写在 `HEARTBEAT.md` 中：

```markdown
# Heartbeat rules

## Every patrol
- Check `gh pr list --search "review-requested:@me"`. If non-empty, summarize.
- Check inbox count via `mail-count` skill. If > 20, ping.
- Otherwise: reply HEARTBEAT_OK.
```

## 写好 HEARTBEAT.md

上面的例子只是最小配置。在生产环境中，你需要分类、优先级，以及周末模式：

```markdown
# Heartbeat Rules — Production

## Priority system
- P0 (always interrupt): CI failure on main, security advisory, service down
- P1 (interrupt during work hours): PR review requested, blocked thread > 4h
- P2 (batch into next cron summary): unread count high, disk above 80%
- If only P2 items found: reply HEARTBEAT_OK

## Code checks
- `gh pr list --search "review-requested:@me"` — P1
- `gh run list --branch main --status failure --limit 3` — P0
- Draft PR idle > 48h — P2

## Communication checks
- `mail-count` skill, unread > 30 — P2
- Tagged thread waiting on me > 4h — P1

## Infrastructure checks
- `disk-usage` any mount > 85% — P1
- `service-health` any non-200 — P0

## Weekend mode (Saturday & Sunday)
- Skip all P2 checks. Only run: CI failures, service health, security.
```

优先级系统是核心洞察。没有它，每次检查都会触发通知；有了它，低优先级事项会静默累积，最终由 Cron 在每日总结中统一汇总。周末模式则让基础设施监控持续运行，同时过滤掉那些可以等到周一再处理的噪音。

## Cron —— 闹钟

有两种方式。CLI 方式更可靠，即使 Gateway 重启也不会丢失：

```bash
openclaw cron add \
  --name "morning-brief" \
  --cron "30 7 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Generate today's brief: weather, calendar, top 3 PR/issue items" \
  --wake now
```

一次性延迟触发（适合“20 分钟后提醒我”）：

```bash
openclaw cron add \
  --name "standup-prep" \
  --at "20m" \
  --session main \
  --system-event "Standup in 20. Summarize yesterday's work."
```

列出和删除：

```bash
openclaw cron list
openclaw cron delete --name morning-brief
```

Cron 任务会持久化到磁盘，Gateway 重启后依然存在。

## 我实际在用的四种模式

**模式 1：每日简报 —— Cron，早上 6:47。**

天气、当天日程、最活跃的三个 GitHub 事项。之所以选 6:47 而不是 7:00，是为了让你拿起手机时，消息已经静静躺在那里了。

```bash
openclaw cron add \
  --name "daily-brief" \
  --cron "47 6 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Morning brief. Include: (1) weather for Shanghai today, (2) calendar events from gcal skill, (3) top 3 most-commented GitHub issues/PRs in the last 24h. Under 15 lines."
```

6:47 收到的内容如下：

```yaml
Morning brief — Thursday Apr 10
Weather: Shanghai, 18C, overcast, rain after 3pm.
Calendar: Sprint planning 10:00 | 1:1 with L 14:30 | Design review 16:00
GitHub:
1. #412 "Fix race condition in queue worker" — waiting on your review
2. #409 "Add rate limiting to /api/generate" — merged overnight
3. #415 "Deps: bump vite to 6.2" — security patch, auto-merged
```

**模式 2：仓库监控 —— Cron，每小时一次。**

```bash
openclaw cron add \
  --name "repo-watcher" \
  --cron "0 * * * *" \
  --tz "Asia/Shanghai" \
  --session dev \
  --system-event "Check repos [myorg/backend, myorg/frontend]: new PRs, new issues, CI failures on main, dependabot advisories. If nothing new, reply empty."
```

只有存在新内容时才发送。“reply empty”这条指令确保不会产生“一切正常”的垃圾消息。

**模式 3：竞品监控 —— Cron，每晚 9 点。**

```bash
openclaw cron add \
  --name "competitor-watch" \
  --cron "0 21 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Run web-diff skill against competitor-urls.yaml. Report only pages that changed: URL, what changed, one-sentence interpretation. If nothing changed, reply empty."
```

输出示例：

```text
Competitor watch — Apr 10, 9:00pm

Changes detected (2 of 8 pages):
- competitor.com/pricing — New "Enterprise" tier ($299/mo), previously only Free and Pro
- competitor.com/blog — New post: "Why We're Betting on Local-First AI"
```

这个功能上线不到两周就证明了自己的价值。

**模式 4：下班收工 —— Cron，下午 5:30。**

归档当天工作，生成总结并写入记忆，闭环处理晨间简报中提到的事项。

```bash
openclaw cron add \
  --name "eod-shutdown" \
  --cron "30 17 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "End of day. (1) List accomplishments from git commits, closed issues, merged PRs. (2) List what's still open from the morning brief. (3) Write 3-line summary to memory under today's date. (4) Flag anything blocked for tomorrow."
```

`1-5` 表示仅限工作日。写入记忆后，第二天的简报就能引用前一天未完成的任务——这两个任务由此形成一个闭环。

## 凌晨 3 点的反模式

![OpenClaw 快速入门（8）：心跳、定时任务和早上7点的 ping 检测 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.png)

那个把你吵醒的典型错误是：给 Heartbeat 设置了 `target: "all"`，却没配 `activeHours`，同时 `HEARTBEAT.md` 在无事发生时也不返回 `HEARTBEAT_OK`。

结果就是：每 45 分钟，你加入的所有频道都会弹出“Patrolled, no anomalies.”，包括凌晨 3 点，甚至那些根本没要求此功能的群聊。

两个解决办法：一是尊重 `activeHours` 设置，二是在规则文件中让 `HEARTBEAT_OK` 成为默认且明确的静默响应。

## Cron 任务调试

以下是我亲身踩过的四个坑。

**故障 1：时区不匹配。** 你写了 `"30 7 * * *"`，以为是上海早上 7:30，但 Gateway 实际运行在 UTC 时区，结果简报下午 3:30 才到。诊断方法：`openclaw cron list --verbose`（会显示本地时区的下次触发时间）。修复方式：始终显式指定 `--tz` 参数。

**故障 2：输出无处可去。** 任务确实执行了（日志可见），但没有任何消息出现。这是因为会话未绑定频道。诊断命令：`openclaw cron logs --name "morning-brief" --last 3`，查找 `delivery: no_channel_bound`。修复命令：`openclaw session bind --name main --channel general`。

**故障 3：重启期间错过触发。** Gateway 恰好在 Cron 应该触发的那一分钟重启，导致任务被跳过。诊断命令：`openclaw cron history --name "morning-brief" --days 7`（你会看到一个时间缺口）。修复方法：

```bash
openclaw cron edit --name "morning-brief" --missed-fire retry --missed-window "15m"
```

但窗口别设太宽——如果中午才收到晨间简报，那就毫无意义了。

**故障 4：任务耗时过长。** 一个复杂的技能链跑了 8 分钟，触发超时，最终只收到部分输出或什么都没有。诊断命令：`openclaw cron logs --name "competitor-watch" --last 1 --full`，查找 `timeout` 或 `context_cancelled`。修复方式：`openclaw cron edit --name "competitor-watch" --timeout "5m"`。如果任务确实需要更长时间，就简化提示词，并把重负载工作卸载到后台技能中。

## 组合使用 Heartbeat 和 Cron

真正的威力在于它们协同构成一个闭环系统：

1. **晨间 Cron（6:47）** —— 生成简报，并将当天的行动项写入记忆；
2. **Heartbeat（每 45 分钟，9:00–18:00）** —— 检查这些行动项是否正在推进。比如 PR 提交 3 小时后仍未被 review？立即升级提醒；
3. **晚间 Cron（17:30）** —— 汇报当日完成情况，标记未完成事项，并将总结写入记忆。

Heartbeat 的配置需与晨间简报的输出联动：

```json5
{
  agents: {
    defaults: {
      heartbeat: {
        every: "45m",
        target: "last",
        prompt: "Read HEARTBEAT.md. Also check if action items from today's morning brief (memory key 'today_actions') remain unaddressed >3h. If so, escalate as P1.",
        activeHours: { start: "09:00", end: "18:00", timezone: "Asia/Shanghai" }
      }
    }
  }
}
```

```bash
openclaw cron add \
  --name "morning-brief" \
  --cron "47 6 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Morning brief. List action items (PRs to review, assigned issues, calendar prep). Write list to memory key 'today_actions'. Deliver brief."

openclaw cron add \
  --name "eod-summary" \
  --cron "30 17 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "EOD. Read 'today_actions' from memory. Check completion status of each. Report done/open/new items. Write to 'yesterday_summary'. Clear 'today_actions'."
```

这样一来，三个原本独立的定时任务就组成了一套连贯、可追溯的日常工作流。如果我的晨间简报写着“Review PR #412”，而到了下午 2 点还没处理，Heartbeat 就会提醒我；随后的 EOD Cron 则会记录我是否真正跟进。

## 定时任务的 Token 开销

定时任务确实消耗 Token，下面是具体计算。

**Heartbeat：** 每天活跃 13 小时，按 45 分钟间隔计算，约 17 次调用/天。每次安静巡逻（系统提示词 + 记忆 + HEARTBEAT_OK）约 800 tokens，合计 13,600 tokens/天。以 qwen-plus（$0.003/1K tokens）计价，约为 $0.04/天。若巡逻触发技能，每次额外消耗 3,000–8,000 tokens。假设有 3 次触发，总量升至约 26,200 tokens，即 $0.08/天。

**Cron：** 开销通常更大，因为它们总是产生输出。晨间简报：3,500 tokens；仓库监控（8 次/天）：12,000 tokens；竞品监控：10,000 tokens；EOD 总结：4,000 tokens。总计约 29,500 tokens，即 $0.09/天。

**Combined：** 合计 $0.15–0.20/天，使用 qwen-plus 时每月约 $4.50–6.00。若换成 gpt-4o 或 claude-sonnet，成本会高出 5–8 倍。

**建议：** Heartbeat 的提示词务必精简——系统提示词和 `HEARTBEAT.md` 每次巡逻都会加载，多一行就等于每天多付 17 次费用。把重活交给 Cron（如复杂技能、长篇总结），因为它们只运行一次。如果你的 Heartbeat 开销超过了 Cron，说明任务分工出了问题。

## 结语

OpenClaw 的核心价值在于：Agent 会主动来找你。Heartbeat 和 Cron 就是实现这一点的两扇门。用 Cron 处理那些无论发生什么都必须按时执行的任务；用 Heartbeat 处理那些只在异常发生时才需要你注意的事情。别把它们搞混了。
