---
title: "OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality"
date: 2026-04-12 09:00:00
tags:
  - openclaw
  - telegram
  - dingtalk
  - wechat
  - channels
categories: OpenClaw
lang: en
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw QuickStart"
series_order: 5
description: "Telegram in five minutes, DingTalk in fifteen, WeChat via WorkBuddy."
disableNunjucks: true
translationKey: "openclaw-quickstart-5"
---

The point of OpenClaw is that the agent comes to you. So far ours hasn't — it's only on the TUI. Time to wire a channel.

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_1.png)

## Telegram — five minutes

![Channel routing architecture — message flow from IM platforms through the gateway](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/fig_channels.png)

I always tell people to start here even if they don't plan to use Telegram in production. The setup is the cleanest of any channel and you can verify the agent end-to-end without other moving parts.

**Step 1.** Talk to `@BotFather` in Telegram. Send `/newbot`, give it a name, get a token that looks like `7891234567:AAH...`.

**Step 2.** Find your own user ID. Easiest way: message `@userinfobot` and it tells you. Mine is a 9-digit number.

**Step 3.** Add a `telegram` block to `openclaw.json`:

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "7891234567:AAH...",
    "allowed_user_ids": [123456789],
    "polling": true
  }
}
```

Two things to call out:

- **`allowed_user_ids`** is non-optional in spirit. Without it your bot answers any random person who finds it. Don't deploy without this.
- **`polling: true`** is the right default. The webhook flavor needs a public HTTPS endpoint. Polling works behind NAT, behind a corporate VPN, anywhere.

**Step 4.** Restart the gateway:

```bash
openclaw gateway restart
```

You should see in the log:

```toml
[telegram] polling started, listening as @your_bot_name
```

**Step 5.** Open Telegram, find your bot, send "hello". You should get the same one-liner the TUI gave you back in piece 2. If the bot stays silent, 90% of the time it's that you forgot your own user ID in `allowed_user_ids`.

### Troubleshooting Telegram

Even with the simple setup, you'll hit friction. Here are the five issues I see repeatedly:

**1. Bot not responding — check `allowed_user_ids`.** You send a message, the bot ignores you. Double-check the user ID in your config matches what `@userinfobot` told you. Common mistakes include copying the username instead of the numeric ID, or forgetting the square brackets in the JSON array. The gateway log will show `[telegram] message from unauthorized user 987654321, ignoring` if this is the problem.

**2. Messages delayed — check polling interval.** Default polling interval is 1 second, which feels instant. If you changed `polling_interval` to something like 30 seconds to reduce API calls, that's your lag. For interactive chat, keep it at 1.

**3. Bot responds but cuts off mid-sentence — check `max_tokens` in agent config.** If the agent starts to reply but stops abruptly, the issue is usually in `agents.json`, not the Telegram channel config. Look for `max_tokens` in the Pi Agent block. Bump it to 2048 or 4096 for conversational use.

**4. Media not supported — which file types work.** Telegram bots can receive text, images (JPEG/PNG), documents (PDF/TXT/JSON), and voice messages. They cannot natively handle video or stickers in a useful way without extra processing. If you send a PDF, the gateway downloads it and passes the file path to the agent — but the agent needs a skill that knows how to read PDFs.

**5. Group chat — how to enable group mode.** By default, your bot only works in direct messages. To add it to a group, tell BotFather to allow group access: send `/setjoingroups`, select your bot, choose "Enable". In a group, the bot only sees messages that mention it (`@your_bot_name what's the weather`) unless you disable Privacy Mode with `/setprivacy` in BotFather.

## DingTalk — fifteen minutes

DingTalk is more complex because it expects a real registered "robot application" with permissions. Here's the short version:

1. **Register a DingTalk Open Platform app.** Go to `https://open-dev.dingtalk.com`, create an app of type "Stream Mode". Note the `Client ID` and `Client Secret`.
2. **Grant the app the messaging scopes.** It needs at minimum `Contact.User.Read` and the bot-message scopes.
3. **Add the bot to a group.** In DingTalk, make a group, go to settings, add your bot.

Then the channel config:

```json
"channels": {
  "dingtalk": {
    "enabled": true,
    "client_id": "dingxxxxxxxxxxxx",
    "client_secret": "...",
    "robot_name": "Lobster",
    "stream_mode": true
  }
}
```

Stream Mode is the modern way — it uses a long-lived WebSocket from the gateway out to DingTalk's servers, so you don't need a public webhook. This is the single biggest improvement DingTalk made for self-hosted bots.

### How Stream Mode WebSocket actually works

Worth understanding the architecture because it's the reason you can run this on a laptop behind NAT.

Traditional webhook mode: DingTalk's servers make an HTTPS POST to your endpoint every time someone sends a message. That means you need a public IP, a domain, a valid SSL cert, and your firewall must allow inbound 443.

Stream Mode flips it: your gateway opens an **outbound** WebSocket connection to `wss://stream.dingtalk.com` and keeps it alive. When someone sends a message, DingTalk pushes an event frame down that WebSocket. Your gateway reads the frame, decodes the JSON, and routes it to the agent. The reply goes back up the same WebSocket. No inbound ports, no DNS, no cert management.

### Testing in sandbox

DingTalk's developer console gives you a sandbox environment. In sandbox mode, you get a test `Client ID` and a simulated group where you can message the bot without deploying to real users. Once it works in sandbox, promoting to production is just swapping the Client ID and Secret in your config.

After restart, in your DingTalk group, mention your bot: `@Lobster how are you`. If the agent replies, you're done. If you get nothing, the most common cause is the app permissions never got approved by your enterprise admin. Check the DingTalk admin console.

## WeChat — the honest version

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_2.png)

There are three paths and only one of them is sane.

**Path 1: openclaw-china (community plugin).** This wraps an unofficial WeChat protocol. Don't. Tencent will ban accounts that use it, and a friend of mine almost lost his personal account this way.

### Why Path 1 is actually dangerous

Tencent's detection is sophisticated. They don't just look for a flag in your client version string. The ban patterns include:

- **IP correlation:** If your account suddenly starts sending messages from a datacenter IP block, that's a red flag. Personal WeChat is mobile-first; server IPs are not normal.
- **Protocol fingerprinting:** The unofficial protocols emulate the WeChat client, but they get small details wrong — packet timing, keepalive intervals, the order of certain handshake fields.
- **Message rate and pattern analysis:** A human doesn't reply to 50 messages in a minute with perfectly formatted markdown. The anomaly detection flags it.

The bans are not always immediate. Tencent sometimes lets the account run for days, collects behavioral data, then bans in a wave. When they ban, it's often permanent.

**Path 2: Work WeChat.** Has an official OpenClaw integration. Works well, but you need a registered enterprise WeChat account and the bot only operates inside enterprise channels. Good for teams; awkward for personal use.

**Path 3: WorkBuddy.** Tencent's own desktop client that lets a registered AI agent talk through your personal WeChat. This is the official supported path for personal-WeChat agents.

```json
"channels": {
  "workbuddy": {
    "enabled": true,
    "workbuddy_id": "wb_...",
    "api_key": "..."
  }
}
```

### WorkBuddy registration timeline

Expect this to take a few days:

1. **Apply:** Go to `https://workbuddy.weixin.qq.com`, log in with your WeChat, submit a developer application.
2. **1-3 day review:** Tencent's team manually reviews. "Personal assistant" usually passes; "marketing bot" might get rejected.
3. **Configure:** Once approved, get your WorkBuddy ID and API credentials. Download the desktop client, log in, bind the agent.
4. **Test:** Send a message to your bot's WeChat contact. It should route through WorkBuddy to OpenClaw and back.
5. **Go live:** Once testing works, you can chat with the bot from your phone or any WeChat client.

If you want WeChat, use WorkBuddy. Budget an evening for the setup.

## Multi-channel setup

You don't have to pick one. The gateway can run multiple channels simultaneously.

### How the gateway multiplexes channels

When you enable both `telegram` and `dingtalk`, the gateway spawns a listener thread for each. Telegram polls its API every second. DingTalk holds a WebSocket open. Both threads feed into the same message queue, which the dispatcher routes to agents.

### Config example: Telegram + DingTalk together

```json
"channels": {
  "telegram": {
    "enabled": true,
    "bot_token": "7891234567:AAH...",
    "allowed_user_ids": [123456789],
    "polling": true,
    "agent": "pi"
  },
  "dingtalk": {
    "enabled": true,
    "client_id": "dingxxxxxxxxxxxx",
    "client_secret": "...",
    "robot_name": "Lobster",
    "stream_mode": true,
    "agent": "pi"
  }
}
```

Both channels route to the same Pi Agent. You can also route different channels to different agents by changing the `agent` field.

## Channel health monitoring

### Gateway logs

Each channel writes startup and periodic heartbeat logs:

```toml
[telegram] polling started, listening as @your_bot_name
[telegram] heartbeat: 142 messages processed, 0 errors
[dingtalk] stream connected, session_id=abc123
```

If you stop seeing heartbeat logs, the channel is dead.

### `openclaw status` command

```bash
openclaw status
# Gateway: running (PID 12345)
# Channels:
#   telegram: active, last message 12s ago
#   dingtalk: active, last message 3m ago
```

## Security considerations

### Rate limiting

The gateway has a built-in rate limiter. Default is 10 messages per user per minute. Configure in `openclaw.json`:

```json
"rate_limit": {
  "messages_per_minute": 10,
  "burst": 20
}
```

### User allowlisting

`allowed_user_ids` for Telegram, `allowed_staff_ids` for DingTalk. If your bot has access to sensitive tools (file system, email, internal APIs), you want a hard list of who can invoke it.

### Token rotation

The `bot_token` for Telegram and `client_secret` for DingTalk are long-lived credentials. Rotate them every 90 days. For Telegram, use `/token` in BotFather to regenerate. For DingTalk, issue a new secret in the developer console.

## Picking your starting channel

If you live outside China and just want it to work: **Telegram**. Five minutes, no review process.

If your team already uses DingTalk and you want the agent in a group: **DingTalk Stream Mode**. The fifteen minutes is a one-time cost.

If you want WeChat: budget an evening for WorkBuddy.

## What "wired up" actually means

After this piece you should have at least one channel that:

1. Receives messages from a real chat platform
2. Routes them to the same Pi Agent the TUI talks to
3. Returns replies in the same chat

That last point is what makes it feel like an assistant rather than a CLI. Everything you ask the agent to do — read a file, run a search, summarize a document — happens through the gateway, and you can see the gateway log scroll while the chat is still in mid-reply. That visibility is rare in commercial products.

Next piece, the part that turns the agent from a chatbot into something that does specific work for you: Skills, plus an MCP server for browser automation.
