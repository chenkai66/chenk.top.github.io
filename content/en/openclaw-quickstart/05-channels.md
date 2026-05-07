---
title: "OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality"
date: 2026-04-07 09:00:00
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
description: "Telegram in five minutes, DingTalk in fifteen, WeChat in 'it depends'. Concrete config blocks for each channel and an honest take on which to start with depending on where you actually live."
disableNunjucks: true
translationKey: "openclaw-quickstart-5"
---

The point of OpenClaw is that the agent comes to you. So far ours hasn't — it's only on the TUI. Time to wire a channel.

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_1.jpg)

## Telegram — five minutes

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

```
[telegram] polling started, listening as @your_bot_name
```

**Step 5.** Open Telegram, find your bot, send "hello". You should get the same one-liner the TUI gave you back in piece 2. If the bot stays silent, 90% of the time it's that you forgot your own user ID in `allowed_user_ids`.

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

After restart, in your DingTalk group, mention your bot: `@Lobster how are you`. If the agent replies, you're done. If you get nothing, the most common cause is the app permissions never got approved by your enterprise admin. Check the DingTalk admin console.

## WeChat — the honest version

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_2.jpg)

There are three paths and only one of them is sane.

**Path 1: openclaw-china (community plugin).** This wraps an unofficial WeChat protocol. Don't. Tencent will ban accounts that use it, and a friend of mine almost lost his personal account this way.

**Path 2: 企业微信 (Work WeChat).** Has an official OpenClaw integration. Works well, but you need a registered enterprise WeChat account and the bot only operates inside enterprise channels. Good for teams; awkward for personal use.

**Path 3: WorkBuddy.** Tencent's own desktop client that lets a registered AI agent talk through your personal WeChat. This is the official supported path for personal-WeChat agents. Setup involves registering as a WorkBuddy developer, getting a `workbuddy_id`, and pointing OpenClaw at it.

```json
"channels": {
  "workbuddy": {
    "enabled": true,
    "workbuddy_id": "wb_...",
    "api_key": "..."
  }
}
```

If you want WeChat, use WorkBuddy. The official OpenClaw docs have the full registration walkthrough — it's about an hour, mostly waiting on Tencent's review.

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
