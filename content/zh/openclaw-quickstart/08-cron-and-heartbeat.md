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
第一次部署 OpenClaw 时，我一直主动发消息；两天后才意识到——它那时只是个聊天机器人，远非真正意义上的 Agent。真正跃迁的标志是它开始主动向我发消息。

这篇文章将介绍如何实现这两种主动触发的机制。

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.png)

## Heartbeat 还是 Cron —— 需明确区分

![Heartbeat vs Cron scheduling model comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/fig_hb_vs_cron.png)

两者都能调度任务，但本质不同。

| | Heartbeat | Cron |
|---|---|---|
| 触发条件 | 固定间隔（每 30 分钟） | 时钟时间（每天 8:30） |
| 默认行为 | "定期巡检，仅在发现问题时上报" | "严格按计划执行，确保结果产出" |
| 适用场景 |  捕获异常、发现遗漏任务 | 日常例行、定时报告 |
| 会话上下文 | 最新活跃渠道 | 默认隔离 |

> Heartbeat 像巡逻警察，Cron 像厨房定时器。

在生产环境中，两者并用：Cron 负责每日 7:00 的晨间简报和 17:00 的 shutdown 总结；Heartbeat 每 45 分钟巡检一次，专门检查遗漏的 PR。

## 开启功能

```json
{
  "tools": {
    "cron":    { "enabled": true },
    "message": { "enabled": true }
  }
}
```

注意 `message` 这个工具——没它的话， Agent 没法主动向外发送任何消息。

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

这里的窍门在于 `HEARTBEAT_OK` 这个约定。你要的是*安静*的巡逻。默认 prompt 明确要求 Agent：仅在发现需关注的问题时才响应，否则保持静默。少了这一行，你每 45 分钟就会收到一条"一切正常"，这将导致大量无效消息，显著降低信息价值。

然后 `HEARTBEAT.md` 里承载具体的规则：

```markdown
# Heartbeat rules

## Every patrol
- Check `gh pr list --search "review-requested:@me"`. If non-empty, summarize.
- Check inbox count via `mail-count` skill. If > 20, ping.
- Otherwise: reply HEARTBEAT_OK.
```

## 写好 HEARTBEAT.md

上面的例子只是最小集。生产环境里你需要分类、优先级，还有周末模式：

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

优先级机制是核心设计：缺少它，每项检查都可能触发干扰通知；有了它，低优先级事项会静默累积，由 Cron 在每日总结中统一呈现。周末模式确保基础设施监控持续运行，同时将可延至周一处理的低优先级事项静默过滤。

## Cron —— 闹钟

有两种用法。CLI 方式更持久（即使 Gateway 重启也不怕）：

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

Cron 任务会持久化到磁盘。重启 Gateway 后，它们还在。

## 我实际在用的四种模式

**模式 1：每日简报 —— Cron，早上 6:47。**

天气、当天的日历、最活跃的三个 GitHub 事项。选择 6:47 而非 7:00 是刻意为之：确保你清晨拿起手机时，消息已就绪。

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

**模式 2：仓库监控 —— Cron，每小时。**

```bash
openclaw cron add \
  --name "repo-watcher" \
  --cron "0 * * * *" \
  --tz "Asia/Shanghai" \
  --session dev \
  --system-event "Check repos [myorg/backend, myorg/frontend]: new PRs, new issues, CI failures on main, dependabot advisories. If nothing new, reply empty."
```

只有存在新内容时才发送。“reply empty”这条指令意味着不会有“一切正常”的垃圾消息。

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

```
Competitor watch — Apr 10, 9:00pm

Changes detected (2 of 8 pages):
- competitor.com/pricing — New "Enterprise" tier ($299/mo), previously only Free and Pro
- competitor.com/blog — New post: "Why We're Betting on Local-First AI"
```

该监控功能上线两周即实现成本回收。

**模式 4：下班 shutdown —— Cron，下午 5:30。**

归档当天的工作，将总结写入 memory，并完成晨间简报中未闭环的事项。

```bash
openclaw cron add \
  --name "eod-shutdown" \
  --cron "30 17 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "End of day. (1) List accomplishments from git commits, closed issues, merged PRs. (2) List what's still open from the morning brief. (3) Write 3-line summary to memory under today's date. (4) Flag anything blocked for tomorrow."
```

`1-5` 表示仅工作日。写入 memory 意味着明天的简报可以引用昨天未完成的事项——这两个任务形成了一个闭环。

## 凌晨 3 点的反模式

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.png)

那个把你半夜吵醒的错误配置：在 Heartbeat 上设了 `target: "all"` 却没配 `activeHours`，然后写的 HEARTBEAT.md 里没事的时候也不回复 `HEARTBEAT_OK`。

结果是每 45 分钟，你所在的每个渠道都会亮起来显示“Patrolled, no anomalies.”，包括凌晨 3 点，甚至那些根本没要求这个功能的群聊。

两个 fixes：遵守 `activeHours`，并且在规则文件里让 `HEARTBEAT_OK` 成为默认的大声回复。

## Cron 任务调试

我自己踩过的四种故障模式。

**故障 1：时区不匹配。** 你写了 `"30 7 * * *"` 以为是上海早上 7:30，但 Gateway 跑在 UTC 上。你的简报下午 3:30 才到。用 `openclaw cron list --verbose` 诊断（显示本地时区的下次触发时间）。修复：始终显式传递 `--tz`。

**故障 2：输出无处可去。** 任务跑了（日志可见）但没消息出现。会话没有绑定渠道。用 `openclaw cron logs --name "morning-brief" --last 3` 诊断，找 `delivery: no_channel_bound`。修复：`openclaw session bind --name main --channel general`。

**故障 3：重启期间错过触发。** Gateway 正好在你 Cron 触发的那一分钟重启。任务被跳过。用 `openclaw cron history --name "morning-brief" --days 7` 诊断（你会看到一个缺口）。修复：

```bash
openclaw cron edit --name "morning-brief" --missed-fire retry --missed-window "15m"
```

窗口不要设得太宽——中午才收到晨间简报就没有意义了。

**故障 4：任务耗时过长。** 复杂的 skill 链跑了 8 分钟，命中超时，你只得到部分输出或什么都没有。用 `openclaw cron logs --name "competitor-watch" --last 1 --full` 诊断（找 `timeout` 或 `context_cancelled`）。修复：`openclaw cron edit --name "competitor-watch" --timeout "5m"`。如果确实需要更多时间，简化 prompt 并把重活卸载到后台 skill。

## 组合使用 Heartbeat 和 Cron

真正的威力在于它们作为闭环系统协同工作：

1. **晨间 Cron (6:47am)** —— 生成简报，把今天的行动项写入 memory。
2. **Heartbeat (每 45 分钟， 9am-6pm)** —— 检查这些行动项是否得到处理。PR 过了 3 小时还没 review？升级提醒。
3. **晚间 Cron (5:30pm)** —— 报告完成了什么，标记没完成的，把总结写入 memory。

Heartbeat 配置须关联晨间简报输出：

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

由此，三个独立定时任务构成一套具备问责机制的连贯工作流。如果晨间简报说“Review PR #412”，而到了下午 2 点我还没动它，Heartbeat 会提醒我，EOD Cron 随后记录我是否跟进到位。
## 定时任务的 Token 开销

定时任务需要消耗 Token。我们来算一笔账。

**Heartbeat：** 每天活跃 13 小时 / 45 分钟间隔 = 每天约 17 次调用。每次安静巡逻（系统提示词 + 记忆 + HEARTBEAT_OK）大概 800 tokens，合计 13,600 tokens/天。用 qwen-plus （$0.003/1K tokens）算下来是 $0.04/天。要是巡逻触发了技能，每次得多花 3,000-8,000 tokens。假设有 3 次触发，总量升到 ~26,200 tokens = $0.08/天。

**Cron：** 开销永远更大，因为它们必定会产生输出。晨间简报：3,500 tokens。仓库监控（每小时触发，共 8 次）：12,000 tokens。竞品监控：10,000 tokens。每日总结：4,000 tokens。总计：~29,500 tokens，每天 0.09 美元。

**Combined：** 加起来 $0.15-0.20/天，用 qwen-plus 一个月大概 $4.50-6.00。如果换成 gpt-4o 或者 claude-sonnet，价格会翻 5-8 倍。

**Recommendation：** Heartbeat 提示词越短越好——系统提示词和 HEARTBEAT.md 每次巡逻都得加载，每多写一行，每天就得为此多付 17 次钱。重活交给 Cron（复杂技能、长总结），因为它们只跑一次。要是发现 Heartbeat 比 Cron 还贵，说明你活儿分反了。

## 结语

OpenClaw 的核心在于让 Agent 主动找你。Heartbeat 和 Cron 是实现这一点的两扇门。严格按时执行的任务交给 Cron，仅在异常发生时才需响应的任务交给 Heartbeat。不要把它们搞混。