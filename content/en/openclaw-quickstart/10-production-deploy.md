---
title: "OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About"
date: 2026-04-17 09:00:00
tags:
  - openclaw
  - production
  - ECS
  - nginx
  - troubleshooting
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 10
description: "ECS, pm2, nginx, acme.sh — plus seven production failure modes and their fixes."
disableNunjucks: true
translationKey: "openclaw-quickstart-10"
---

The local install gets you to "it works on my machine." The server install is what makes it survive a kernel update.

This chapter walks through the deploy I actually use on a 2-core 4G ECS box, then the failures I have seen often enough to put in writing.

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/10-production-deploy/illustration_1.png)

## Choosing your server

![Production deployment stack — from OS to monitoring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/10-production-deploy/fig_deploy.png)

Before you deploy, pick the right box. Four options worth considering:

**Alibaba Cloud ECS**: The path I use. A 2-core 4GB instance in the cn-beijing region costs around $15/month. The advantage is proximity to DashScope — your API round-trips drop from 200ms to 20ms when both the gateway and the model sit in the same region. The disadvantage is the Great Firewall makes outbound package installs occasionally flaky unless you set up a mirror.

**DigitalOcean**: Simplest onboarding. The $18/month "Basic" droplet (2vCPU, 4GB) is functionally identical to ECS. The dashboard is cleaner, the docs are better, and package mirrors are unnecessary. The tradeoff is latency — if your model provider is Alibaba or Tencent, you add 150ms per turn.

**Hetzner**: Best price-to-performance. A CX21 instance (2vCPU, 4GB, Nuremberg) runs $5.83/month. The catch is that Hetzner's network is optimized for Europe, so if your users or your model endpoints sit in Asia, you will see the latency.

**Home server**: Free hardware, full control, infinite disk. The problems are uptime (your ISP does not promise five nines), dynamic IPs (you will need DDNS), and port forwarding (which your router may block for inbound 443). A home server works for prototyping; it does not work for production unless you are also your only user.

**Minimum spec: 2-core 4GB.** Why? The gateway itself uses 200MB of resident memory, but model responses buffer in RAM before streaming to the client, and when a sub-agent forks, the OS duplicates the parent process momentarily. I have watched a 2GB instance OOM during a code-review task that spawned three sub-agents in parallel. 4GB gives you headroom. 8GB eliminates the problem entirely.

## The deploy

OS: Ubuntu 22.04. The 4G of RAM matters — 2G works for one agent and chokes the moment a sub-agent spawns.

```bash
# 1. Node 22 via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22 && nvm use 22

# 2. OpenClaw + pm2 globally
npm i -g openclaw@latest pm2

# 3. Workspace
openclaw init
```

pm2 supervises the Gateway so that crashes don't take you down:

```bash
pm2 start "openclaw gateway" --name openclaw-gateway --time
pm2 save
pm2 startup
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

## Docker alternative

If you prefer containers, OpenClaw ships with Docker support. Here is the compose file I use for multi-service deployments:

```yaml
version: '3.8'
services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw-gateway
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - ./config:/root/.openclaw/config
      - ./workspace:/root/.openclaw/workspace
      - ./agents:/root/.openclaw/agents
      - ./skills:/root/.openclaw/skills
    environment:
      - NODE_ENV=production
```

Three volumes are critical:

- `config/` holds API keys, channel credentials, model endpoints. Without this mount, every container restart wipes your setup.
- `workspace/` contains MEMORY.md and session logs. Losing this means the agent forgets everything between restarts.
- `skills/` stores custom skills. If you mount this read-only, the agent can read skills but cannot write new ones during self-improvement.

**When Docker is better:** You run OpenClaw alongside other services (a database, a vector store, a monitoring stack) and you want one `docker-compose up` to bring everything online. The isolation also makes it easier to test config changes — spin up a second container with a different config, compare behavior, kill the worse one.

**When Docker is worse:** Debugging file permissions — the container runs as root, your host files may not be, and you will spend time with `chown`. Inspecting logs requires `docker exec` or volume mounts. Hot-reloading skills during development is slower because the filesystem sync has a delay. For a single-service deploy where you ssh into the box and tail logs directly, bare-metal is simpler.

## The eight failures

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/10-production-deploy/illustration_2.png)

### 1. `command not found: openclaw` after reboot

nvm doesn't load in non-interactive shells. pm2 startup uses one. Either source nvm in `/etc/profile.d/`, or symlink the binary:

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

**Detection:** `pm2 status` shows the gateway in `errored` state with exit code 127 immediately after boot.

### 2. `Node.js version too old`

OpenClaw needs >= 22.16. The error is clear; the real bug is using whatever Node your distro ships.

**Detection:** Gateway fails to start, and `pm2 logs openclaw-gateway --err` shows a version mismatch in the first three lines.

### 3. `401 Unauthorized` from DashScope

Two causes: Coding Plan key used against the wrong endpoint, or key rotated and not replaced.

**Detection:** Every agent turn fails instantly with a 401 in the response. Check `~/.openclaw/agents/main/sessions/*.jsonl` — if the last line of every session is an auth error, it's your key.

### 4. `Connection refused` on Gateway start

Port 18789 is taken:

```bash
lsof -i :18789
kill $(lsof -t -i :18789)
pm2 restart openclaw-gateway
```

**Detection:** `pm2 logs openclaw-gateway --err` shows `EADDRINUSE` within the first second of startup. The gateway never reaches the "listening on 18789" log line.

### 5. DingTalk goes silent after 30 minutes

The long-poll connection is being torn down by an upstream NAT:

```json
"dingtalk": {
  "reconnectMs": 60000,
  "heartbeatMs":  30000
}
```

**Detection:** Gateway log shows `[dingtalk] reconnecting...` more than once per hour. Users report "the bot stopped responding" but manual messages from the web dashboard still work.

### 6. The agent forgets things mid-conversation

Compaction ran without `memoryFlush` enabled. Set `memoryFlush.enabled: true`.

**Detection:** A multi-turn conversation suddenly loses context after turn 15. Check session length: `cat ~/.openclaw/agents/main/sessions/<session-id>.jsonl | wc -l`. If it's exactly 20 lines (the default compaction threshold), compaction discarded turns instead of summarizing.

### 7. `Token consumption is way too high`

Three reasons: expensive default model, bloated MEMORY.md, or sub-agents spawning for trivial tasks.

**Detection:** Your bill doubles week-over-week despite stable usage. Run `openclaw stats tokens --since 7d` and compare the per-turn average. If it climbs above 8k tokens/turn for a conversational agent, something is wrong. Grep MEMORY.md for length: `wc -l ~/.openclaw/workspace/MEMORY.md`. Anything above 500 lines is a red flag.

### 8. Memory grows unbounded

What happens when you never archive sessions: MEMORY.md bloats past 100 lines, then 200, then 500. Every agent turn now includes half a kilobyte of irrelevant context ("three weeks ago the user asked about Docker"). Startup slows because the workspace loader parses the entire memory file on boot. At 1000 lines, startup takes 30 seconds. At 2000 lines, the agent begins timing out mid-turn because the context window is 80% memory and 20% actual task.

**Fix:** Automate the weekly cleanup. Add a cron job that archives old sessions and trims MEMORY.md:

```bash
# Every Sunday at 2 AM
0 2 * * 0 openclaw memory archive --older-than 30d && openclaw memory compact --max-lines 100
```

The `archive` command moves sessions older than 30 days into a `.archive/` subdirectory (still readable, just not loaded by default). The `compact` command uses the LLM to summarize MEMORY.md down to 100 lines, preserving the most important facts and discarding low-value details.

**Detection:** `openclaw gateway` takes more than 10 seconds to print "Gateway listening on 18789". Or check file size directly: `wc -l ~/.openclaw/workspace/MEMORY.md`. Anything above 300 lines warrants a manual review. Above 500 lines, compact immediately.

## Upgrade path

OpenClaw moves fast. New features ship weekly, and occasionally a release changes the config schema or deprecates a skill field. Here is how to upgrade safely:

1. **Check the changelog:** `openclaw changelog --since <current-version>`. Look for breaking changes, deprecated fields, or new required config keys.

2. **Backup your config:** `cp -r ~/.openclaw/config ~/.openclaw/config.backup`. If the upgrade breaks, you can restore in ten seconds.

3. **Upgrade the binary:** `npm i -g openclaw@latest`. This pulls the new version but does not restart anything.

4. **Restart the gateway:** `pm2 restart openclaw-gateway`. Watch the logs for the first 60 seconds: `pm2 logs openclaw-gateway --lines 100`. If you see repeated errors or a crash loop, roll back: `npm i -g openclaw@<old-version> && pm2 restart openclaw-gateway`.

5. **Verify health:** Hit the dashboard at `https://agent.example.com` and send a test message. Check that skills load, memory persists, and the agent responds coherently.

**What breaks during upgrades:**

- **Config schema changes:** A field gets renamed (`model.name` becomes `model.id`), and the gateway fails to parse your config. The error message usually tells you which field is invalid. Fix: update the config, restart.
  
- **Deprecated skill fields:** Your custom skill uses `skill.parameters` but the new version expects `skill.input`. The skill loader throws a validation error. Fix: regenerate the skill with `openclaw skill create` or manually update the schema.

- **Dependency conflicts:** Rare, but it happens. A new OpenClaw version needs a library that conflicts with something else you installed globally. Symptom: the upgrade succeeds, but the gateway crashes on startup with a module resolution error. Fix: use `nvm` to isolate Node environments, or run OpenClaw in Docker to avoid global installs entirely.

**Golden rule:** Never upgrade during peak hours. Do it at 2 AM on a Sunday, when a five-minute outage is invisible.

## Monitoring and alerting

A production service you cannot monitor is a production service you do not control. Three layers:

### Health check

Set up a cron job that pings the gateway every five minutes and alerts if it goes down:

```bash
#!/bin/bash
# /usr/local/bin/openclaw-healthcheck.sh

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18789/health)

if [ "$STATUS" -ne 200 ]; then
  echo "OpenClaw gateway is down (HTTP $STATUS)" | mail -s "Alert: OpenClaw Down" you@example.com
fi
```

Add to cron:

```bash
*/5 * * * * /usr/local/bin/openclaw-healthcheck.sh
```

If you run multiple services, use a real monitoring stack (Prometheus, Grafana, Uptime Kuma). But for a single-agent deploy, a bash script and a mail command are enough.

### Log rotation

The gateway writes to stdout, pm2 captures it, and without rotation, `~/.pm2/logs/openclaw-gateway-out.log` grows forever. After three months, it hits 2GB and fills your disk.

Create `/etc/logrotate.d/openclaw`:

```
/home/ubuntu/.pm2/logs/openclaw-gateway-out.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
  postrotate
    pm2 reloadLogs
  endscript
}
```

This keeps seven days of logs, compresses old ones, and tells pm2 to re-open the log file after rotation.

### Disk space alerts

The agent writes session logs to `~/.openclaw/agents/main/sessions/`. If you run a popular agent, this directory grows at 10MB/day. After a year, it's 3.6GB. If your server has a 20GB root partition, you will eventually fill it.

Add a disk space check to the same healthcheck script:

```bash
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$USAGE" -gt 85 ]; then
  echo "Disk usage is at ${USAGE}% on /" | mail -s "Alert: Disk Space Low" you@example.com
fi
```

When the alert fires, either archive old sessions (`openclaw memory archive --older-than 90d`) or expand the disk.

## The "is it healthy" five-liner

```bash
pm2 status openclaw-gateway
openclaw doctor
wc -l ~/.openclaw/workspace/MEMORY.md
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
df -h /
```

## Closing

A production OpenClaw is not a clever piece of software; it is a boring piece of software that has been left running for thirty days. Do the deploy by the book, set up the supervisor, put a stable proxy in front, fix the eight failures above before they hit you, automate the monitoring, and you'll get there.

That's the end of the QuickStart. From here the path forks — into custom skills, custom channels, custom MCP servers, multi-agent topologies. All of it builds on the foundations these ten pieces laid down.
