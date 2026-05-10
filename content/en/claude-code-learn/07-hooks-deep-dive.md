---
title: "Claude Code Hands-On (7): Ten Hooks I Actually Use, with the Code"
date: 2026-04-24 09:00:00
tags:
  - claude-code
  - hooks
  - security
  - workflow
categories: Claude Code
lang: en
mathjax: false
series: claude-code-learn
series_title: "Claude Code Hands-On"
series_order: 7
description: "Picking ten hooks out of the 100 in the reference repo and walking through each: what it does, the actual JS, the settings.json wire-up, and where it bites. PreToolUse for safety, PostToolUse for hygiene, the boring ones that earn their keep."
disableNunjucks: true
translationKey: "claude-code-learn-7"
---

Chapter 5 was the conceptual tour of hooks. This one is the field guide. Out of the 100-script reference repo, ten earn their place in every serious project I run. Those are the ten I will walk through, with code.

All examples assume Node 18+, save scripts to `./hooks/`, mark them `chmod +x`, and wire them in `.claude/settings.json` like:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] }
    ]
  }
}
```

![Claude Code Hands-On (7): Ten Hooks I Actually Use, with the Code — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/illustration_1.png)

## 1. block-env-read — protect secrets

The single highest-ROI hook. Stops `Read` and `Grep` from touching `.env`, `id_rsa`, `credentials.json`:

```javascript
#!/usr/bin/env node
const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const t = JSON.parse(Buffer.concat(chunks).toString());
const p = t.tool_input?.file_path || t.tool_input?.path || "";
const sensitive = ['.env', 'credentials.json', 'secrets.yaml', 'id_rsa', '.aws/credentials'];
if (sensitive.some(s => p.includes(s))) {
  console.error(`Blocked: ${p} matches a sensitive file pattern.`);
  process.exit(2);   // 2 = block, only valid in PreToolUse
}
process.exit(0);
```

Wire on `Read|Grep|MultiEdit|Edit|Write`. Exit code 2 in PreToolUse blocks the call; the stderr text is fed back to the model so it sees *why*.

## 2. bash-blacklist — stop `rm -rf /`

![Claude Code Hands-On (7): Ten Hooks I Actually Use, with the Code — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/illustration_2.png)

The most common foot-gun. PreToolUse on `Bash`:

```javascript
const cmd = JSON.parse(/* stdin */).tool_input?.command || "";
const banned = [/rm\s+-rf\s+\/(\s|$)/, /:\(\)\s*\{.*:\|:.*\}\s*;:/, /mkfs\./, /dd\s+if=.*of=\/dev\//];
if (banned.some(re => re.test(cmd))) {
  console.error("Blocked: dangerous command pattern.");
  process.exit(2);
}
```

The regex list is short on purpose. Long blocklists get ignored when they cause false positives.

## 3. bash-whitelist — for production-adjacent boxes

The inverse, for repos that touch production. Allow only an explicit set of binaries:

```javascript
const allow = new Set(['ls','cat','grep','rg','git','npm','node','python','python3','curl','jq','head','tail']);
const first = (cmd.trim().split(/\s+/)[0] || "").split('/').pop();
if (!allow.has(first)) { console.error(`Not on allowlist: ${first}`); process.exit(2); }
```

Whitelists win where blocklists lose: you cannot accidentally allow something new.

## 4. block-git-push — no surprise pushes

I have never wanted Claude to push without asking. PreToolUse on `Bash`:

```javascript
if (/^\s*git\s+push\b/.test(cmd) || /git.*push.*--force/.test(cmd)) {
  console.error("Blocked: git push must be human-initiated.");
  process.exit(2);
}
```

The cost of being wrong is so much worse than the friction of typing `git push` myself.

## 5. format-on-write — Prettier as a PostToolUse

PostToolUse on `Write|Edit|MultiEdit`:

```javascript
const path = t.tool_input?.file_path;
if (path && /\.(ts|tsx|js|jsx|json|md|css)$/.test(path)) {
  require('child_process').execSync(`npx prettier --write ${path}`, { stdio: 'inherit' });
}
```

PostToolUse runs *after* the edit, so exit code 2 doesn't roll anything back. The point is hygiene, not policy.

## 6. test-on-edit — fail fast

PostToolUse on `Edit|MultiEdit` for source files:

```javascript
if (/\/(src|lib)\/.*\.(ts|js)$/.test(path)) {
  try { require('child_process').execSync('npm run -s test:related -- ' + path, { stdio: 'inherit' }); }
  catch { console.error("Tests failed after edit"); process.exit(1); }
}
```

Exit code 1 surfaces the failure to the model, which then sees the test output and tries again. This is the single hook that taught Claude to write better code over time on my repos.

## 7. backup-before-edit — the safety net

PreToolUse on `Edit|Write|MultiEdit`:

```javascript
const fs = require('fs');
if (fs.existsSync(path)) {
  fs.copyFileSync(path, `/tmp/cc-backup-${Date.now()}-${path.replace(/\//g,'_')}`);
}
```

Cheap insurance. I have recovered files from `/tmp` exactly twice, both times worth a year of cron job pay.

## 8. log-tool-calls — observability

PostToolUse on `*`:

```javascript
const fs = require('fs');
const line = JSON.stringify({ ts: Date.now(), tool: t.tool_name, input: t.tool_input }) + "\n";
fs.appendFileSync('.claude/tool-calls.jsonl', line);
```

You will not look at this file every day. The day you do, you will be glad it exists.

## 9. read-before-write — no blind edits

PreToolUse on `Edit|MultiEdit`:

```javascript
const fs = require('fs');
const seen = JSON.parse(fs.existsSync('.claude/seen.json') ? fs.readFileSync('.claude/seen.json') : "{}");
if (t.tool_name === 'Read') { seen[path] = Date.now(); fs.writeFileSync('.claude/seen.json', JSON.stringify(seen)); process.exit(0); }
if (!seen[path]) { console.error(`Blocked: ${path} was not Read in this session.`); process.exit(2); }
```

Forces the model to read a file before editing it. Catches the subtle bug where the model edits based on its prior, not the file's current state.

## 10. work-hours-only — humane boundaries

PreToolUse on `Bash`:

```javascript
const h = new Date().getHours();
if (h < 9 || h >= 22) {
  console.error("Outside work hours. Refuse.");
  process.exit(2);
}
```

I run this on the box that handles after-hours pings. If the bot tries to do something destructive at 2am, that is almost certainly a misfire.

## What ties them together

Three rules I picked up the hard way:

1. **PreToolUse for policy, PostToolUse for hygiene.** Don't try to undo things in PostToolUse — the side-effect already happened.
2. **Stderr is feedback, exit code is verdict.** Exit 2 blocks (PreToolUse only). Anything in stderr gets fed back to Claude verbatim. Use both.
3. **Hooks fail closed.** A misbehaving hook will block all your tool calls. Test the script with `echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/x"}}' | node hook.js` before wiring it in.

Ten hooks does not sound like much. It is enough to make a Yolo-mode session feel responsible.
