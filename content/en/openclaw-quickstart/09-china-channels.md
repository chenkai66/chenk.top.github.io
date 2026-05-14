---
title: "OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs"
date: 2026-04-16 09:00:00
tags:
  - openclaw
  - dingtalk
  - wechat
  - wecom
  - channels
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 9
description: "DingTalk, WeCom, WeChat WorkBuddy — a picker matrix with actual config blocks."
disableNunjucks: true
translationKey: "openclaw-quickstart-9"
---

[Chapter 5](/en/openclaw-quickstart/05-channels/) covered Telegram, DingTalk, and WeChat at a glance. This chapter is the sequel for everyone in mainland China who needed to ship something past their team's IT department. There are too many channels, the docs are scattered across a dozen READMEs, and most of the "compare" tables online are out of date.

Here’s the matrix I check before making any recommendations.

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_1.png)

---

## The seven channels worth knowing

![China IM channel selection decision tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/fig_picker.png)

Everything below assumes you're using the `openclaw-china` plugin:

```bash
openclaw plugins install https://github.com/BytePioneer-AI/openclaw-china.git
```

Or the wizard:

```bash
npx @openclaw-china/setup
```

Here are the choices:

| Channel | Audience | Public IP needed | Reaches WeChat users | Group chat | Streaming | Latency | Msg formats | Cost |
|---|---|:---:|:---:|:---:|:---:|---|---|---|
| DingTalk | internal team | no | no | yes | yes | 200-500ms | text, markdown, cards | free |
| Feishu | internal team | yes | no | yes | partial | 300-600ms | text, markdown, cards | deprecated |
| WeCom smart bot | internal team | no | no | yes | yes | 300-700ms | text, markdown, files | free |
| WeCom self-built app | external WeChat | yes | yes | no | no | 800-1500ms | text, images | ~2000 RMB/yr |
| WeChat customer service | external WeChat | yes | yes | no | no | 1000-2000ms | text, images | per-seat |
| WeChat public account | followers | yes | yes | no | no | 1500-3000ms | text, news cards | free |
| WorkBuddy (QClaw) | personal WeChat/QQ | no | yes | yes | yes | 500-1000ms | text, images | free |

Two notes: Feishu is officially deprecated for new openclaw-china installs as of March 2026 — choose DingTalk or WeCom. WeChat public accounts in subscription mode have a 5-second passive reply window and can't push proactively; service accounts and test accounts can.

Latency numbers are round-trip message-to-response timings measured from Beijing and Shanghai during business hours. The main cause of high latency is the webhook delivery path. Public accounts and WeCom self-built apps route through Tencent's dispatcher, adding 500-1000ms compared to long-poll channels.

Message format support is crucial. Markdown lets your agent send code blocks and formatted lists. DingTalk and WeCom smart bot handle GitHub-flavored markdown well. WeChat channels require plain text or proprietary card schemas.

## The three-question picker

Instead of reading the matrix top to bottom, ask yourself three questions:

1. **Who is on the other end?** A coworker → DingTalk or WeCom long-poll. An external WeChat user → WeCom self-built app or WeChat customer service. Yourself → WorkBuddy.
2. **Do you have a public IP?** No → only DingTalk, WeCom long-poll, and WorkBuddy are options. Yes → everything is on the table.
3. **Do you need group chat?** Yes → DingTalk, WeCom long-poll, or WorkBuddy. The customer-service and public-account channels are 1:1 only.

In practice, I usually end up with one of two stacks:

- **Internal team, no public IP:** DingTalk for the human channel, WeCom long-poll as backup. Heartbeat patrols both.
- **External WeChat reach:** WeCom self-built app for the agent's primary surface, WeChat customer service for inbound from non-tagged users.

## DingTalk — the path that always works

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_2.png)

Stream mode (long-poll) is why DingTalk is quick to set up. No public IP or domain verification required—just register a bot and grab two values:

```json
{
  "channels": {
    "dingtalk": {
      "enabled": true,
      "mode": "stream",
      "clientId": "dingxxxxxxx",
      "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxx",
      "agents": ["main"]
    }
  }
}
```

Where to get those values: 钉钉开放平台 → 应用开发 → 创建企业内部应用 → 凭证与基础信息. Add the `BotMessage` permission scope. Publish the app inside your org (没发布的话机器人不会响应).

The most common failure: messages stop arriving after 30 minutes. Cause: the org's network egress is rotating IPs and the long-poll connection is being killed mid-stream. Fix: bump the reconnect interval down to 60s in `dingtalk.reconnectMs`, and put the Gateway behind a stable egress.

## DingTalk deep dive — the gotchas

**Message size limits.** Single messages cap at 2048 characters for text, 4096 for markdown cards. If your agent generates longer responses, set `dingtalk.autoChunk: true` — it splits automatically but you'll see multiple message bubbles.

**Rate limiting at 20 messages per second.** Per bot, not per user. If you have a busy group chat, you'll hit this. The plugin queues and retries automatically, but latency spikes. Consider splitting heavy users into separate bot instances.

**The org-admin approval flow.** You create the app, add scopes, test in the developer console — everything works. Then you publish and the bot goes silent. Cause: enterprise apps require org-admin approval before messaging users outside the developer's department. Check 钉钉管理后台 → 工作台 → 应用管理 → 待审批.

**Clock drift kills connections.** DingTalk's long-poll dies silently if your server clock drifts more than 5 minutes from NTP. No errors, just silence. Run `ntpdate` or point at a reliable NTP source.

## WeCom — three flavors, pick by audience

The "WeCom" name covers three different APIs that share nothing except the brand:

- **Smart bot (智能机器人, 长连接).** Internal use only. No public IP needed. Easiest install. Streaming works.
- **Self-built app (自建应用).** Reaches external WeChat users *if* the org has the WeCom-WeChat互通 license. Requires public IP and a verified domain. No streaming.
- **Customer service (微信客服).** External WeChat users without the互通 license. Routes from public-account menus, mini-programs, video accounts, livestream entries. Markdown not supported.

For 90% of "we need a bot in our team" cases, the smart bot is what you want.

## WeCom deep dive

**Smart bot registration.** 企业微信管理后台 → 应用管理 → 创建应用 → 机器人. You get a webhook URL immediately. For long-poll mode, enable the callback server and grab the `token` and `encodingAESKey`. No domain verification, no public IP. Caveat: smart bots cannot initiate conversations — they can only reply.

**Self-built app registration.** Need: verified domain (ICP filing required), public IP for callback, and the互通 license (~2000 RMB/year base tier). Once you have the license, external WeChat users can add your bot as a contact. No group chats with external users, no streaming, and message formats are limited.

**The互通 license cost.** Base tier (100 external contacts): ~2000 RMB/year. Mid tier (1000 contacts): ~10000 RMB/year. Enterprise: negotiable from 50000 RMB/year. If testing, ask for a 3-month trial (up to 50 contacts).

**Session management.** WeCom doesn't maintain conversation history server-side. The `openclaw-china` plugin handles this with a session store keyed on `(userId, channelId)`, expiring after 30 minutes. Important: user IDs are channel-scoped — same person in smart bot vs self-built app has different IDs.

## WorkBuddy — the desktop bridge

WorkBuddy. It's Tencent's official QClaw bridge, runs on your desktop, no IP, no app review, talks to your real personal WeChat and QQ. It is the only "I just want my agent in WeChat tonight" answer that doesn't end in regret.

The catch: it's a desktop application, so it lives or dies with that machine being awake. Fine for a personal assistant. Bad for production.

**Setup steps:**
1. Register at `qclaw.tencent.com` with a mainland China phone number. Pick "personal assistant" as use case.
2. Wait 1-3 business days for approval.
3. Download the desktop client (Windows/macOS only), log in with QR code.
4. Configure OpenClaw:

```json
{
  "channels": {
    "workbuddy": {
      "enabled": true,
      "endpoint": "http://localhost:8765",
      "accounts": ["wechat"],
      "agents": ["main"]
    }
  }
}
```

No API keys, no webhooks. The plugin talks to WorkBuddy on localhost, and WorkBuddy talks to WeChat/QQ on your behalf.

## Migration guide — switching channels

You built on DingTalk because it was fast. Now you need WeCom for external users. How to switch without losing history:

1. **Switch to persistent sessions** (Redis) before migrating — in-memory sessions die with the process.
2. **Update config**: disable old channel, enable new one.
3. **Remap user IDs**: user IDs are channel-scoped, so you need a cross-reference mapping (verify via phone/email on first message in new channel).
4. **Test in shadow mode**: run both channels in parallel, old one in `shadowMode: true` (receives but doesn't respond), for a week before cutting over.

## Monitoring health

**Heartbeat endpoint** — poll every 60s:
```bash
curl http://localhost:3000/api/channels/dingtalk/health
# {"status": "connected", "lastMessageAt": "...", "reconnectCount": 2}
```

Alert if `status != connected` or `lastMessageAt` is stale during business hours. If `reconnectCount` climbs every minute, the network path is flaky.

**End-to-end test** — send a test message hourly and verify response. Catches issues heartbeat misses (agent stuck in retry loop).

**Reconnection patterns** — healthy: reconnect once every few hours. Unhealthy: every 30 seconds. Check gateway logs for disconnect codes (1006 = network, 1008 = policy violation/IP blocked).

## The lesson behind the matrix

Channels are where the agent actually *lives*. A great agent on a flaky channel is a flaky agent. So before building anything fancy, pick the channel that matches your audience and your network, and over-invest in making *that* channel rock-solid. Then add a second one for redundancy.

Everything else is rearranging deck chairs.
