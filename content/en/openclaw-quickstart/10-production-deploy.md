---
title: "OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About"
date: 2026-04-12 09:00:00
tags:
  - openclaw
  - production
  - ecs
  - nginx
  - troubleshooting
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 10
description: "Putting OpenClaw on a real server: an ECS box, pm2 as the supervisor, nginx in front, certs renewed by acme.sh. Then the long tail — the seven failures I have seen at least twice in production, and what each one actually was."
disableNunjucks: true
translationKey: "openclaw-quickstart-10"
---

The local install gets you to "it works on my machine." The server install is what makes it survive a kernel update.

This chapter walks through the deploy I actually use on a 2-core 4G ECS box, then the failures I have seen often enough to put in writing.

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/10-production-deploy/illustration_1.jpg)

## The deploy

OS: Ubuntu 22.04. The 4G of RAM matters — 2G works for one agent and chokes the moment a sub-agent spawns.

```bash
# 1. Node 22 via nvm — system Node is too old on most distros
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22 && nvm use 22

# 2. OpenClaw + pm2 globally
npm i -g openclaw@latest pm2

# 3. Workspace
openclaw init
# Edit ~/.openclaw/openclaw.json — at minimum set the model provider
```

pm2 supervises the Gateway so that crashes don't take you down:

```bash
pm2 start "openclaw gateway" --name openclaw-gateway --time
pm2 save
pm2 startup           # follow the printed sudo line
```

For the Web Dashboard (port 18789), nginx in front with certs from acme.sh:

```nginx
server {
  listen 443 ssl http2;
  server_name agent.example.com;

  ssl_certificate     /etc/nginx/ssl/agent.example.com.cer;
  ssl_certificate_key /etc/nginx/ssl/agent.example.com.key;

  location / {
    proxy_pass http://127.0.0.1:18789;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 600s;
  }
}
```

The 600s read timeout is not optional — long-running agent turns will exceed the nginx default of 60s and fail mid-stream.

## The seven failures

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/10-production-deploy/illustration_2.jpg)

### 1. `command not found: openclaw` after reboot

nvm doesn't load in non-interactive shells. pm2 startup uses one. Either source nvm in `/etc/profile.d/`, or symlink the binary:

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

I prefer the symlink. Less magic.

### 2. `Node.js version too old`

OpenClaw needs ≥ 22.16. The error is clear; the real bug is using whatever Node your distro ships. Pin Node 22 and check `node -v` in the same shell pm2 will use.

### 3. `401 Unauthorized` from DashScope

Two real causes:

- The Coding Plan key (`sk-sp-...`) was used against the standard DashScope base URL. They're different endpoints. Coding Plan goes to `https://coding.dashscope.aliyuncs.com/v1`, regular DashScope to `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- The key was leaked, rotated, and not replaced. Check the dashboard.

### 4. `Connection refused` on Gateway start

Port 18789 is taken. Find the squatter and kill it:

```bash
lsof -i :18789
kill $(lsof -t -i :18789)
pm2 restart openclaw-gateway
```

If 18789 is taken by *another* OpenClaw, you forgot you ran `openclaw gateway` outside pm2 earlier today. Stop it, let pm2 own it.

### 5. DingTalk goes silent after 30 minutes

The long-poll connection is being torn down by an upstream NAT. Two fixes that compound:

```json
"dingtalk": {
  "reconnectMs": 60000,
  "heartbeatMs":  30000
}
```

If you control the network, also pin the egress to a single IP. Rotating egresses are the actual root cause more often than not.

### 6. The agent forgets things mid-conversation

Compaction ran without `memoryFlush` enabled. See [chapter 7](../07-memory-system/) — set `memoryFlush.enabled: true` and a sensible `softThresholdTokens`. This single config line is what turns "it forgot" into "it remembered."

### 7. `Token consumption is way too high`

Three reasons, in descending order of likelihood:

1. Every turn is using your most expensive model. Use a tiered config — `qwen3.5-flash` for routing, `qwen3-max` only when the task needs it.
2. `MEMORY.md` ballooned past 40 lines and is being loaded every turn. Audit it.
3. Sub-agents are being spawned for trivial tasks. Inline them.

If none of those: turn on the per-turn token log and look at the actual breakdown. It's almost never what you guessed.

## The "is it healthy" five-liner

I run this every morning the bot exists:

```bash
pm2 status openclaw-gateway
openclaw doctor
wc -l ~/.openclaw/workspace/MEMORY.md
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
df -h /
```

pm2 status confirms the supervisor is happy. `openclaw doctor` runs the built-in checks. `wc -l` on MEMORY.md catches creep. The session count tells me whether to archive. `df -h` catches the disk filling up from logs — which it will, eventually.

## Closing

A production OpenClaw is not a clever piece of software; it is a boring piece of software that has been left running for thirty days. Do the deploy by the book, set up the supervisor, put a stable proxy in front, fix the seven failures above before they hit you, and you'll get there.

That's the end of the QuickStart. From here the path forks — into custom skills, custom channels, custom MCP servers, multi-agent topologies. All of it builds on the foundations these ten pieces laid down.
