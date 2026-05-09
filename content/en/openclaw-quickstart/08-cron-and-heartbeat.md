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
description: "Heartbeat patrols and Cron timers — configs for daily briefings and weekly reports."
disableNunjucks: true
translationKey: "openclaw-quickstart-8"
---

The first time I deployed OpenClaw I sat there sending it messages. After two days I realized I had built a chatbot, not an agent. The thing that made it an agent was the moment it started messaging *me* without being asked.

This piece is about the two ways to make that happen.

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.png)

## Heartbeat vs Cron — pick one in your head

![Heartbeat vs Cron scheduling model comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/fig_hb_vs_cron.png)

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

## Writing a good HEARTBEAT.md

The example above is minimal. In production you need categories, priorities, and a weekend mode:

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

The priority system is the key insight. Without it, every check becomes a ping. With it, low-priority items accumulate silently and get swept into your EOD summary via Cron. The weekend mode keeps infrastructure monitoring alive but drops the noise that can wait until Monday.

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

## Four patterns I actually use

**Pattern 1: Daily brief — Cron, 6:47am.**

Weather, the day's calendar, the three most-active GitHub items. The 6:47 instead of 7:00 is intentional — by the time you pick up your phone, the message is already there.

```bash
openclaw cron add \
  --name "daily-brief" \
  --cron "47 6 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Morning brief. Include: (1) weather for Shanghai today, (2) calendar events from gcal skill, (3) top 3 most-commented GitHub issues/PRs in the last 24h. Under 15 lines."
```

What arrives at 6:47:

```
Morning brief — Thursday Apr 10
Weather: Shanghai, 18C, overcast, rain after 3pm.
Calendar: Sprint planning 10:00 | 1:1 with L 14:30 | Design review 16:00
GitHub:
1. #412 "Fix race condition in queue worker" — waiting on your review
2. #409 "Add rate limiting to /api/generate" — merged overnight
3. #415 "Deps: bump vite to 6.2" — security patch, auto-merged
```

**Pattern 2: Repository watcher — Cron, hourly.**

```bash
openclaw cron add \
  --name "repo-watcher" \
  --cron "0 * * * *" \
  --tz "Asia/Shanghai" \
  --session dev \
  --system-event "Check repos [myorg/backend, myorg/frontend]: new PRs, new issues, CI failures on main, dependabot advisories. If nothing new, reply empty."
```

Sent only if something exists. The "reply empty" instruction means no "all clear" spam.

**Pattern 3: Competitor watch — Cron, daily 9pm.**

```bash
openclaw cron add \
  --name "competitor-watch" \
  --cron "0 21 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "Run web-diff skill against competitor-urls.yaml. Report only pages that changed: URL, what changed, one-sentence interpretation. If nothing changed, reply empty."
```

Example output:

```
Competitor watch — Apr 10, 9:00pm

Changes detected (2 of 8 pages):
- competitor.com/pricing — New "Enterprise" tier ($299/mo), previously only Free and Pro
- competitor.com/blog — New post: "Why We're Betting on Local-First AI"
```

This one earned its keep within two weeks.

**Pattern 4: End-of-day shutdown — Cron, 5:30pm.**

Archives the day's work and writes a summary to memory, closing the loop on anything the morning brief opened.

```bash
openclaw cron add \
  --name "eod-shutdown" \
  --cron "30 17 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "End of day. (1) List accomplishments from git commits, closed issues, merged PRs. (2) List what's still open from the morning brief. (3) Write 3-line summary to memory under today's date. (4) Flag anything blocked for tomorrow."
```

The `1-5` means weekdays only. The memory write means tomorrow's brief can reference yesterday's unfinished items — the two jobs form a cycle.

## The 3am anti-pattern

![OpenClaw QuickStart (8): Heartbeat, Cron, and Getting Pinged at 7am — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.png)

The mistake that wakes you up: setting `target: "all"` on a Heartbeat without `activeHours`, then writing a HEARTBEAT.md that doesn't reply `HEARTBEAT_OK` when nothing's wrong.

Result: every 45 minutes, every channel you're in lights up with "Patrolled, no anomalies." Including at 3am. Including in group chats with people who didn't ask for this.

Two fixes: respect `activeHours`, and make `HEARTBEAT_OK` the loud default in your rules file.

## Cron job debugging

Four failure modes I have hit personally.

**Failure 1: Timezone mismatch.** You write `"30 7 * * *"` expecting 7:30am Shanghai, but the Gateway runs in UTC. Your brief arrives at 3:30pm. Diagnose with `openclaw cron list --verbose` (shows next fire time in local tz). Fix: always pass `--tz` explicitly.

**Failure 2: Output goes nowhere.** The job runs (visible in logs) but no message appears. The session has no channel binding. Diagnose with `openclaw cron logs --name "morning-brief" --last 3` and look for `delivery: no_channel_bound`. Fix: `openclaw session bind --name main --channel general`.

**Failure 3: Missed fire during restart.** Gateway restarts at the exact minute your cron fires. The job is skipped. Diagnose with `openclaw cron history --name "morning-brief" --days 7` (you'll see a gap). Fix:

```bash
openclaw cron edit --name "morning-brief" --missed-fire retry --missed-window "15m"
```

Don't set the window too wide — a morning brief at noon is useless.

**Failure 4: Job takes too long.** A complex skill chain runs 8 minutes, hits a timeout, and you get partial output or nothing. Diagnose with `openclaw cron logs --name "competitor-watch" --last 1 --full` (look for `timeout` or `context_cancelled`). Fix: `openclaw cron edit --name "competitor-watch" --timeout "5m"`. If it legitimately needs more time, simplify the prompt and offload heavy work to a background skill.

## Combining Heartbeat and Cron

The real power is when they work together as a closed-loop system:

1. **Morning Cron (6:47am)** — generates the brief, writes today's action items to memory.
2. **Heartbeat (every 45min, 9am-6pm)** — checks if those action items are getting addressed. PR still un-reviewed after 3 hours? Escalate.
3. **Evening Cron (5:30pm)** — reports what got done, flags what didn't, writes summary to memory.

The heartbeat config ties into the morning brief's output:

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

This turns three independent scheduled jobs into a coherent daily workflow with accountability. If my morning brief said "Review PR #412" and by 2pm I haven't touched it, the heartbeat nudges me. The EOD cron then records whether I followed through.

## Token cost of scheduled jobs

Scheduled jobs cost tokens. Here is the math.

**Heartbeat:** 13 active hours / 45min interval = ~17 calls/day. At ~800 tokens per quiet patrol (system prompt + memory + HEARTBEAT_OK): 13,600 tokens/day. On qwen-plus ($0.003/1K tokens) that is $0.04/day. When patrols trigger skills, they cost 3,000-8,000 tokens each. With 3 triggered patrols: total rises to ~26,200 tokens = $0.08/day.

**Cron:** Always heavier because they always produce output. Morning brief: 3,500 tokens. Repo watcher (8 hourly fires): 12,000. Competitor watch: 10,000. EOD summary: 4,000. Total: ~29,500 tokens = $0.09/day.

**Combined:** $0.15-0.20/day = $4.50-6.00/month on qwen-plus. On gpt-4o or claude-sonnet, multiply by 5-8x.

**Recommendation:** Keep heartbeat prompts short — the system prompt and HEARTBEAT.md load on every single patrol, so every extra line costs 17x per day. Use cron for heavy lifting (complex skills, long summaries) because those run once. If your heartbeat costs more than your cron, you have the division of labor backwards.

## Closing

The whole point of OpenClaw is the agent comes to you. Heartbeat and Cron are the two doors that make it possible. Use Cron for things that should happen on a schedule no matter what; use Heartbeat for things you want noticed only when they break. Don't mix them up.
