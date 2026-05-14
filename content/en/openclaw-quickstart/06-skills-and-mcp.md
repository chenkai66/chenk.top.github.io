---
title: "OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real"
date: 2026-04-13 09:00:00
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

After five pieces, you have a working OpenClaw with a chat channel. This is where it stops being a demo.

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/illustration_1.png)


---

## What we'll build

![Skill composition pipeline — from trigger to tool execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/fig_skills.png)

A morning-briefing agent that:

1. Runs at 7am every weekday
2. Fetches the top headlines from Hacker News (via a Playwright MCP server)
3. Reads my calendar for the day (via a Skill that wraps `gcalcli`)
4. Summarizes both into a paragraph and pushes it to my Telegram

This is a real workflow. By the end, you'll have the structure to swap in your own data sources. But first, let's understand the two systems we're combining.

## Skills vs. Tools vs. MCP — the mental model

These three words often get mixed up, but they shouldn't.

| Concept | What it is | Who writes it | When it loads |
|---------|-----------|--------------|--------------|
| **Tool** | A verb. Read file, exec command, web search. Has a typed schema and a handler function. | Framework author or you (custom tools) | Always loaded; the model sees the full tool list every turn |
| **Skill** | A noun-of-knowledge. A Markdown SOP that tells the agent *how* to accomplish a specific task. | You | Lazily — manifest only until triggered, then body is paged in |
| **MCP Server** | An external process that exposes *additional* tools via the Model Context Protocol. | Third-party or you | Loaded at gateway startup; tools appear alongside built-in tools |

The relationship: Skills *use* tools. MCP servers *provide* tools. A skill might say "use the Playwright tools to scrape this page" — the Playwright tools come from an MCP server, and the skill tells the agent what to do with them.

Think of it like this: Tools are the hands. Skills are the training manual. MCP is a way to give the agent more hands.

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

### Anatomy of a SKILL.md

The file has two sections: the YAML front matter (the manifest) and the Markdown body (the SOP). Both matter, and they do different things at different times.

**The manifest** is loaded at gateway startup. Every skill's manifest is included in the system prompt so the model can decide which skill to invoke. The fields:

| Field | Required | Purpose | Example |
|-------|----------|---------|---------|
| `name` | Yes | Unique identifier. Used in logs and cross-references. | `summarize-headlines` |
| `description` | Yes | One-line summary. The model reads this to decide relevance. | `Summarize a list of headlines...` |
| `trigger` | Yes | Natural-language clause. Write it from the user's perspective, not the implementation's. | `when user asks for a news briefing` |
| `tools_required` | No | Which tools the skill needs. If listed, the gateway pre-authorizes them. | `[web_search, exec]` |
| `skills_required` | No | Other skills this skill depends on. Their bodies are hot-loaded when this skill triggers. | `[today-calendar]` |
| `priority` | No | `high`, `normal`, or `low`. Breaks ties when multiple skills match. Default: `normal`. | `high` |
| `version` | No | Semver string. Informational only; helps when you share skills. | `1.0.0` |

**The body** is only loaded when the model triggers the skill. It is the SOP — the instructions, templates, edge cases, and output format. Treat it like an onboarding doc for a new employee. The more specific the better. Vague instructions ("summarize well") produce vague results. Concrete instructions ("maximum 4 sentences, lead with highest-signal item, skip paywalled sources") produce consistent output.

### Writing effective triggers

The trigger clause is the single most important line in a skill. If it is too broad, the skill fires when it should not. If it is too narrow, it never fires when it should. Some patterns:

**Bad triggers:**
- `when the user asks about news` — too broad, will fire on "what's new in the codebase"
- `when summarize-headlines should run` — circular, means nothing to the model

**Good triggers:**
- `when user asks for a news briefing, headline summary, or daily news digest` — specific nouns
- `when user asks to take meeting notes or document a meeting` — action + domain

**Debugging trigger issues:** If a skill is not firing, add `OPENCLAW_LOG=debug` to your environment and send a test message. Look for `skill_selection` in `gateway.log` — it shows which skills the model considered and why it picked (or didn't pick) each one.

Restart the gateway and verify the skill loaded:

```bash
openclaw skills list | grep summarize
# summarize-headlines  (loaded)
```

You can also inspect what the model sees:

```bash
openclaw skills inspect summarize-headlines
# Name:        summarize-headlines
# Description: Summarize a list of headlines into a one-paragraph briefing
# Trigger:     when user asks for a news briefing, headline summary, or daily news digest
# Tools:       web_search
# Body:        287 chars (loaded on trigger)
```

## Step 2: Attach an MCP server

MCP — Model Context Protocol — is a standard for connecting LLMs to external tool servers. OpenClaw doesn't speak MCP natively; it uses `MCPorter` as a shim that translates between OpenClaw's internal tool format and the MCP protocol.

### Installing MCPorter

```bash
npm i -g mcporter
curl -LsSf https://astral.sh/uv/install.sh | sh   # for uvx, used by some MCP servers
```

Verify:

```bash
mcporter --version
# mcporter v0.4.2
```

### Adding Playwright as an MCP server

```bash
mcporter add playwright npx @playwright/mcp@latest
```

This tells MCPorter: "there is an MCP server called `playwright`, and you start it by running `npx @playwright/mcp@latest`." MCPorter will launch the process and manage its lifecycle.

Now tell OpenClaw about MCPorter in `openclaw.json`:

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright"]
}
```

Restart the gateway. The Playwright MCP server exposes browser-automation tools that the agent can now call. Here is what you get:

| MCP Tool | What it does | Typical use |
|----------|-------------|-------------|
| `browser_navigate` | Go to a URL | Opening a page to scrape |
| `browser_snapshot` | Accessibility tree of current page | Reading structured content |
| `browser_click` | Click an element | Navigating multi-page flows |
| `browser_type` | Type into a field | Form filling, search boxes |
| `browser_evaluate` | Run arbitrary JS on page | Extracting data the DOM tree misses |
| `browser_take_screenshot` | Capture the viewport | Visual verification, debugging |

### Testing the MCP connection

From the TUI:

```text
Use Playwright to fetch the top 5 stories from
https://news.ycombinator.com and just give me the titles and URLs.
```

If the agent comes back with a list, the wiring is good. If it fails, the most common issues are:

1. **MCPorter not running.** Check `mcporter status` — the `playwright` server should show `running`. If it shows `stopped`, run `mcporter start playwright` manually and check `~/.mcporter/logs/playwright.log` for errors.
2. **Port conflict.** MCPorter defaults to `:7890`. If something else uses that port, set `MCPORTER_PORT=7891` and update `porter_endpoint` in `openclaw.json`.
3. **Playwright not installed.** The first `npx @playwright/mcp@latest` run downloads browsers. This can take 2-3 minutes and ~400MB. If it was interrupted, run `npx playwright install chromium` manually.

### Adding other MCP servers

The same pattern works for any MCP-compliant server. Some useful ones:

```bash
# Filesystem MCP — lets the agent read/write files outside the sandbox
mcporter add filesystem npx @anthropic/mcp-filesystem@latest /path/to/allowed/dir

# GitHub MCP — PR reviews, issue triage
mcporter add github npx @anthropic/mcp-github@latest

# SQLite MCP — query a database directly
mcporter add sqlite npx @anthropic/mcp-sqlite@latest /path/to/database.db
```

After adding, update `openclaw.json`:

```json
"mcp": {
  "porter_endpoint": "http://127.0.0.1:7890",
  "servers": ["playwright", "filesystem", "github"]
}
```

Each server's tools appear in the agent's tool list alongside built-in tools. The model treats them identically.

## Step 3: A skill that wraps a CLI tool

Not every integration needs an MCP server. For simple CLI tools, a skill that uses the `exec` tool is often easier. I use `gcalcli` for Google Calendar:

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
- HH:MM-HH:MM **Title** (Location, if any)

If there are no events, return "Nothing on the calendar today".
```

Notice the skill is essentially a recipe, not a function. The model is the runtime. The `exec` tool is the verb. The skill is the noun-of-knowledge that ties them together.

### When to use exec vs. MCP

The decision is straightforward:

| Scenario | Use exec | Use MCP |
|----------|----------|---------|
| One-shot CLI command | Yes | Overkill |
| Stateful interaction (browser, database) | No | Yes |
| Tool with complex typed output | Maybe | Preferred |
| Tool you want to share across projects | No | Yes |
| Quick prototype | Yes | No |

The rule of thumb: if the interaction is "run a command, parse stdout," use `exec`. If the interaction involves multiple back-and-forth steps or persistent state, use an MCP server.

### Security considerations for exec-based skills

The `exec` tool has `dangerous` permission level for a reason. When a skill uses `exec`, consider:

1. **Pin the command.** Write the exact command in the skill body, not "run whatever shell command is needed." The model will sometimes improvise if given room.
2. **Use `trusted_commands` in config.** Add the specific commands your skill uses to the trusted list so the user does not get a confirmation prompt every time:
   ```json
   "tools": {
     "exec": {
       "trusted_commands": ["gcalcli agenda", "gcalcli list", "git status"]
     }
   }
   ```
3. **Validate output.** If the skill processes tool output and feeds it back to the model, consider what happens if the command returns unexpected content. A malicious calendar event title could theoretically inject instructions.

## Step 4: A composing skill

![OpenClaw QuickStart (6): Skills, MCP, and Shipping Something Real — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/06-skills-and-mcp/illustration_2.png)

This is where skills become powerful. A composing skill orchestrates other skills and tools into a multi-step workflow:

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
```text

Send the result to the default channel.
```

The `skills_required` field tells OpenClaw to keep the bodies of those skills hot-loaded when this skill triggers. No re-fetch, no extra latency. This is an important optimization — without it, the agent would need to trigger each sub-skill individually, paying the manifest-lookup cost each time.

### How skill composition works internally

When the morning-briefing skill triggers, the gateway does the following:

1. Loads the morning-briefing body into the prompt.
2. Sees `skills_required: [today-calendar, summarize-headlines]`.
3. Loads both sub-skill bodies into the prompt alongside the parent.
4. The model now has all three SOPs and can execute steps sequentially.

The agent loop runs normally — the model plans tool calls, the gateway executes them, results come back. The difference is that the model has more context about *how* to use each tool because all three skill bodies are present.

This is different from function composition in code. There is no call stack, no return value. The model reads all three SOPs, internalizes them, and executes the combined plan. It works because language models are good at following multi-step instructions.

### Debugging composing skills

The most common failure: the model skips a step. It fetches the news but forgets the calendar, or vice versa. This happens when the prompt is too long and the model loses track.

To diagnose, check the turn-by-turn JSON dump:

```bash
cat ~/.openclaw/sessions/<session_id>.jsonl | jq '.tool_calls[].name'
```

If you see `browser_navigate` but no `exec` (for gcalcli), the model skipped the calendar step. Fixes:

1. **Number your steps explicitly** (the example above does this).
2. **Add a checklist at the end** of the skill body: "Before sending, verify: calendar section present? news section present? Date correct?"
3. **Lower `max_turns`** for this skill if it tends to spiral. In `openclaw.json`:
   ```json
   "skill_config": {
     "morning-briefing": {
       "max_turns": 15
     }
   }
   ```

## Step 5: Cron

Skills become truly useful when they run without you. In `openclaw.json`:

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

`0 7 * * 1-5` is 7am Mon-Fri. Restart the gateway. Verify with:

```bash
openclaw cron list
# morning-briefing | 0 7 * * 1-5 | next: tomorrow 07:00 | channel: telegram
```

The first time it runs, watch the gateway log. You will see the agent loop fire, the skill load, the Playwright tool calls scroll past, and finally a message land in your Telegram.

### Cron configuration reference

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Unique job name. Used in logs and `openclaw cron list`. |
| `schedule` | Yes | cron expr | Standard 5-field cron expression. |
| `skill` | Yes | string | Skill name to trigger. Must exist in `~/.openclaw/skills/`. |
| `channel` | Yes | string | Where to send output. Must be a configured channel. |
| `user_id` | No | string | Which user's memory/context to use. Default: the admin user. |
| `timeout_sec` | No | integer | Maximum execution time. Default: 120. |
| `retry` | No | integer | Number of retries on failure. Default: 0. |
| `env` | No | object | Extra environment variables passed to tools during this job. |

### What breaks with cron

The agent loop runs in a synthetic session — there is no real user on the other end. This has two consequences:

1. **No confirmation prompts.** If the skill uses `exec` and the command is not in `trusted_commands`, the cron job hangs waiting for confirmation that never comes. Always add cron-triggered commands to the trusted list.
2. **No follow-up.** If the model's response includes a question ("Should I include crypto news?"), nobody answers. The message posts to the channel and the session ends. Design cron-triggered skills to be self-contained — all decisions should be in the SOP, not left to user interaction.

A useful pattern for cron debugging:

```bash
# Dry-run a cron job manually
openclaw cron run morning-briefing --dry-run

# See the last 5 cron executions
openclaw cron history morning-briefing --limit 5
```

## Real-world skill examples

The morning-briefing is a starter project. Here are three more patterns I use, with their key design choices.

### Pattern 1: Git changelog

A skill that generates a weekly changelog from git commits:

```markdown
---
name: weekly-changelog
trigger: when user asks for changelog, release notes, or weekly summary
tools_required: [exec, write]
---

Run `git log --oneline --since="7 days ago"` in the project directory.

Group commits by prefix (feat:, fix:, refactor:, docs:).
Generate markdown:
- "## What changed this week"
- One bullet per group with count: "### Features (3)"
- Sub-bullets for each commit

Write to `CHANGELOG-weekly.md` in the project root.
```

Why this works: the skill has a clear input (git log), a clear transformation (grouping by prefix), and a clear output (markdown file). No ambiguity.

### Pattern 2: Database health check

A skill using an SQLite MCP server for periodic checks:

```markdown
---
name: db-health
trigger: when user asks to check database health, db stats, or table sizes
tools_required: [sqlite_query]
---

Run these queries in sequence:
1. `SELECT name, SUM(pgsize) as size_bytes FROM dbstat GROUP BY name ORDER BY size_bytes DESC LIMIT 10;`
2. `SELECT COUNT(*) as total_rows FROM main_table;`
3. `SELECT created_at FROM main_table ORDER BY created_at DESC LIMIT 1;`

Format as a health report:
- Top 10 tables by size (human-readable: KB/MB)
- Total row count
- Last insert timestamp
- Flag any table over 100MB as "large"
```

### Pattern 3: PR review assistant

A skill combining GitHub MCP with code analysis:

```markdown
---
name: pr-review
trigger: when user asks to review a PR, check a pull request, or code review
tools_required: [exec, web_fetch]
skills_required: []
---

1. Fetch the PR diff using `gh pr diff <number>`.
2. For each changed file:
   - Read the full file for context
   - Note: additions > 200 lines suggest the file should be split
3. Check for:
   - Missing error handling (try/catch around IO)
   - Hardcoded secrets (API keys, passwords)
   - TODO/FIXME without issue links
4. Write review as inline comments using `gh pr review <number> --comment --body "..."`.
```

## Troubleshooting

Here are the problems I hit in my first month, and how I fixed them.

### Skill fires but produces wrong output

**Symptom**: The skill triggers correctly, but the agent ignores half the SOP instructions.

**Cause**: The skill body is too long or too vague. Models skim long instructions just like humans do.

**Fix**: Keep the body under 500 words. Use numbered steps, not prose. Put the output template last (recency bias helps). If the skill must be complex, split it into sub-skills with `skills_required`.

### MCP server crashes mid-conversation

**Symptom**: The agent gets a tool error "connection refused" partway through a workflow.

**Cause**: The MCP server process exited. Playwright in particular crashes if the browser encounters an out-of-memory page or a download dialog it cannot handle.

**Fix**: MCPorter has auto-restart. Verify it is enabled:

```bash
mcporter config playwright
# auto_restart: true (default)
# max_restarts: 3
# restart_delay_ms: 1000
```

If the server crashes more than `max_restarts` times in 60 seconds, MCPorter gives up and logs an error. Check `~/.mcporter/logs/playwright.log` for the crash reason. Common culprits: pages that trigger infinite JS loops, or sites that serve 100MB+ resources.

### Cron job runs but sends nothing

**Symptom**: `openclaw cron history` shows the job ran successfully, but no message appeared in the channel.

**Cause**: Usually a channel authentication issue. OAuth tokens expire, DingTalk webhook URLs rotate, Telegram bot tokens get revoked.

**Fix**: Check `gateway.log` for the cron run timestamp. Look for "channel send failed" errors. Re-authenticate the channel: `openclaw channel test telegram` sends a test message and reports any auth errors.

### Two skills fight over the same trigger

**Symptom**: Inconsistent behavior — sometimes skill A fires, sometimes skill B, for the same user message.

**Cause**: Overlapping trigger clauses. The model picks one based on subtle phrasing differences.

**Fix**: Make triggers mutually exclusive. If you have both a "meeting notes" and a "project notes" skill, do not write `when user asks about notes` for either. Use `when user asks about meeting notes or documenting a meeting` and `when user asks about project notes or project status`.

You can also force a specific skill:

```text
/skill morning-briefing
```

This bypasses trigger matching entirely.

## What you have now

- A long-lived agent talking to a real chat platform
- Skills that capture domain knowledge separately from the agent loop
- An MCP server providing capabilities OpenClaw doesn't have natively
- A cron job that turns it from "I have to ask" into "it shows up"
- The debugging tools to figure out what went wrong when it does

That is the whole loop. Every other case study in the official docs — second-brain, content pipeline, devops automation — is a variation on these five steps. Different skills, different MCPs, different cron lines.

## What I'd build next

Three things, in order of effort:

1. **Add a feedback loop.** Reply to the morning briefing with corrections ("skip crypto headlines"). Have a skill that writes those corrections into `~/.openclaw/memory/feedback/morning-briefing.md`. The next morning's briefing pulls them in via ContextEngine retrieval. After a week, the briefing silently adapts to your preferences without you touching the SOP.

2. **Make the news source configurable.** A skill that reads from `~/openclaw-workspace/sources.yaml` and iterates over a list of URLs. That gets you "RSS reader as agent" almost for free. The YAML file becomes a simple UI — add a URL, get it in tomorrow's briefing.

3. **Wire a second channel.** Same agent, also on DingTalk for work hours. The skills don't change — the channel layer is decoupled. You get the same morning briefing in both places, or route different skills to different channels based on context.

That is where I will leave the QuickStart. The rest of the official docs go deep on each layer; you now have the map to navigate them.

If you want a single takeaway, it is that the boring layers — skills, memory, channels — are where the value is. The agent loop is the same agent loop everyone has. What makes your install useful is the skill library you build and the channels you put it on. Good luck.
