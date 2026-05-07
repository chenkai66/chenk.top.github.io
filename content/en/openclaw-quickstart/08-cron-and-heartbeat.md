---
title: "OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am"
date: 2026-04-10 09:00:00
tags:
  - openclaw
  - cron
  - heartbeat
  - automation
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 8
description: "The two scheduling primitives in OpenClaw — Heartbeat is a patrol with a pulse, Cron is a kitchen timer. Concrete configs for a daily briefing, a weekly report, a competitor watcher, and the one anti-pattern that wakes you up at 3am."
disableNunjucks: true
translationKey: "openclaw-quickstart-8"
---

The first time I deployed OpenClaw I sat there sending it messages. After two days I realized I had built a chatbot, not an agent. The thing that made it an agent was the moment it started messaging *me* without being asked.

This piece is about the two ways to make that happen.

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.jpg)

## Heartbeat vs Cron — pick one in your head

Both schedule work. They are not the same thing.

| | Heartbeat | Cron |
|---|---|---|
| Triggered by | a fixed interval (every 30 min) | a clock time (every day at 8:30) |
| Default behavior | "look around, only ping me if something matters" | "always run, always deliver" |
| Best for | catching exceptions, missed work | daily routines, scheduled reports |
| Session | latest active channel | isolated by default |

> Heartbeat is the patrol officer. Cron is the kitchen timer.

In production I use both. Cron handles the 7am morning briefing and the 5pm shutdown summary. Heartbeat handles the "is there a PR I'm forgetting" patrol every 45 minutes.

## Turn them on

```json
{
  "tools": {
    "cron":    { "enabled": true },
    "message": { "enabled": true }
  }
}
```

`message` matters — without it the agent has no way to send anything outbound proactively.

## Heartbeat — the patrol

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

The trick is the `HEARTBEAT_OK` convention. You want a *quiet* patrol. The default prompt explicitly tells the agent to shut up unless something needs your attention. Without that line, you get pinged every 45 minutes with "everything looks fine," which is exactly as annoying as it sounds.

Then `HEARTBEAT.md` carries the actual rules:

```markdown
# Heartbeat rules

## Every patrol
- Check `gh pr list --search "review-requested:@me"`. If non-empty, summarize.
- Check inbox count via `mail-count` skill. If > 20, ping.
- Otherwise: reply HEARTBEAT_OK.
```

## Cron — the alarm

Two ways. CLI is the durable one (survives Gateway restarts):

```bash
openclaw cron add \
  --name "morning-brief" \
  --cron "30 7 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Generate today's brief: weather, calendar, top 3 PR/issue items" \
  --wake now
```

A one-shot delay (useful for "remind me in 20 minutes"):

```bash
openclaw cron add \
  --name "standup-prep" \
  --at "20m" \
  --session main \
  --system-event "Standup in 20. Summarize yesterday's work."
```

List and remove:

```bash
openclaw cron list
openclaw cron delete --name morning-brief
```

Cron jobs persist to disk. Restart the Gateway, they come back.

## Three patterns I actually use

**Daily brief — Cron, 6:47am.** Weather, the day's calendar, the three most-active GitHub items. Delivered to the channel I checked most yesterday. The 6:47 instead of 7:00 is intentional — by the time most people pick up their phone, the message is already there.

**Repository watcher — Cron, hourly.** New PRs, new issues, failing CI on default branch, security advisories on dependencies. Sent only if something exists; the bot is configured to skip "all clear."

**Competitor watch — Cron, daily 9pm.** Diff a small list of pages I care about. New blog post, new pricing tier, new feature page → ping. This one earned its keep within two weeks.

## The 3am anti-pattern

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.jpg)

The mistake that wakes you up: setting `target: "all"` on a Heartbeat without `activeHours`, then writing a HEARTBEAT.md that doesn't reply `HEARTBEAT_OK` when nothing's wrong.

Result: every 45 minutes, every channel you're in lights up with "Patrolled, no anomalies." Including at 3am. Including in group chats with people who didn't ask for this.

Two fixes: respect `activeHours`, and make `HEARTBEAT_OK` the loud default in your rules file.

## Closing

The whole point of OpenClaw is the agent comes to you. Heartbeat and Cron are the two doors that make it possible. Use Cron for things that should happen on a schedule no matter what; use Heartbeat for things you want noticed only when they break. Don't mix them up.
