---
title: "OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real"
date: 2026-04-08 09:00:00
tags:
  - openclaw
  - skills
  - mcp
  - playwright
  - cron
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 6
description: "Write a Skill, attach a Playwright MCP server, and ship a morning-briefing agent."
disableNunjucks: true
translationKey: "openclaw-quickstart-6"
---

Five pieces in, you have a working OpenClaw with a chat channel. This is where it stops being a demo.

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/illustration_1.png)

## What we'll build

![Skill composition pipeline — from trigger to tool execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/fig_skills.png)

A morning-briefing agent that:

1. Runs at 7am every weekday
2. Fetches the top headlines from Hacker News (via a Playwright MCP server)
3. Reads my calendar for the day (via a Skill that wraps `gcalcli`)
4. Summarizes both into a paragraph and pushes it to my Telegram

That's a real workflow. By the end you'll have the bones to swap in your own data sources.

## Step 1: Write a Skill

A Skill lives at `~/.openclaw/skills/<name>/SKILL.md`. Let's write one for "summarize headlines":

```bash
mkdir -p ~/.openclaw/skills/summarize-headlines
```

Create `~/.openclaw/skills/summarize-headlines/SKILL.md`:

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

Two design notes:

- The **trigger** field is what the model sees when deciding whether the skill applies. Write it from the user's perspective, not the implementation's.
- The **body** is the SOP. Treat it like onboarding doc for a new junior employee — examples, edge cases, output format. The more specific the better.

Restart the gateway and verify the skill loaded:

```bash
openclaw skills list | grep summarize
# summarize-headlines  (loaded)
```

## Step 2: Attach an MCP server

OpenClaw doesn't speak MCP natively — it uses `MCPorter` as a shim. Install it:

```bash
npm i -g mcporter
curl -LsSf https://astral.sh/uv/install.sh | sh   # for uvx, used by some MCP servers
```

Now add Playwright via MCPorter:

```bash
mcporter add playwright npx @playwright/mcp@latest
```

Tell OpenClaw about MCPorter in `openclaw.json`:

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright"]
}
```

Restart the gateway. The Playwright MCP exposes browser-automation tools (`navigate`, `screenshot`, `click`, `extract_text`) that the agent can now call.

Test from the TUI:

```
Use Playwright to fetch the top 5 stories from
https://news.ycombinator.com and just give me the titles and URLs.
```

If the agent comes back with a list, the wiring is good.

## Step 3: A skill that wraps a CLI tool

I use `gcalcli` for calendar. Skill:

```bash
mkdir -p ~/.openclaw/skills/today-calendar
```

`~/.openclaw/skills/today-calendar/SKILL.md`:

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

Notice the skill is essentially a recipe, not a function. The model is the runtime. The `exec` tool is the verb. The skill is the noun-of-knowledge that ties them together.

## Step 4: A composing skill

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/illustration_2.png)

Now a skill that uses both:

```bash
mkdir -p ~/.openclaw/skills/morning-briefing
```

`~/.openclaw/skills/morning-briefing/SKILL.md`:

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

The `skills_required` field tells OpenClaw to keep the bodies of those skills hot-loaded when this skill triggers. No re-fetch, no extra latency.

## Step 5: Cron

In `openclaw.json`:

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

`0 7 * * 1-5` is 7am Mon–Fri. Restart the gateway. Verify with:

```bash
openclaw cron list
# morning-briefing | 0 7 * * 1-5 | next: tomorrow 07:00 | channel: telegram
```

The first time it runs, watch the gateway log. You'll see the agent loop fire, the skill load, the Playwright tool calls scroll past, and finally a message land in your Telegram.

## What you have now

- A long-lived agent talking to a real chat platform
- Skills that capture domain knowledge separately from the agent loop
- An MCP server providing capabilities OpenClaw doesn't have natively
- A cron job that turns it from "I have to ask" into "it shows up"

That is the whole loop. Every other case study in the official docs — second-brain, content pipeline, devops automation — is a variation on these five steps. Different skills, different MCPs, different cron lines.

## What I'd build next

Three things, in order of effort:

1. **Add a feedback loop.** Reply to the morning briefing with corrections ("skip crypto headlines"). Have a skill that writes those corrections into `~/.openclaw/memory/feedback/morning-briefing.md`. The next morning's briefing pulls them in.
2. **Make the news source configurable.** A skill that reads from `~/openclaw-workspace/sources.yaml` and iterates. That gets you "RSS reader as agent" almost for free.
3. **Wire a second channel.** Same agent, also on DingTalk for work hours. The skills don't change.

That is where I will leave the QuickStart. The rest of the official docs go deep on each layer; you now have the map to navigate them.

If you want a single takeaway, it is that the boring layers (skills, memory, channels) are where the value is. The agent loop is the same agent loop everyone has. What makes your install useful is the skill library you build and the channels you put it on. Good luck.
