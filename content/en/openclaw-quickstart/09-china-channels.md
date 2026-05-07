---
title: "OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs"
date: 2026-04-11 09:00:00
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
description: "DingTalk, Feishu (deprecated), WeCom in three flavors, WeChat customer service, public accounts, and the WorkBuddy desktop bridge. A picker matrix and the actual config blocks for the two paths that work in 2026."
disableNunjucks: true
translationKey: "openclaw-quickstart-9"
---

Chapter 5 covered Telegram, DingTalk, and WeChat at a glance. This chapter is the sequel for everyone in mainland China who needed to ship something past their team's IT department. There are too many channels, the docs are scattered across a dozen READMEs, and most of the "compare" tables online are out of date.

What follows is the matrix I check before recommending anything to anyone.

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_1.jpg)

## The seven channels worth knowing

Everything below assumes you're using the `openclaw-china` plugin:

```bash
openclaw plugins install https://github.com/BytePioneer-AI/openclaw-china.git
```

Or the wizard:

```bash
npx @openclaw-china/setup
```

Then the choices:

| Channel | Audience | Public IP needed | Reaches WeChat users | Group chat | Streaming |
|---|---|:---:|:---:|:---:|:---:|
| DingTalk | internal team | no | no | yes | yes |
| Feishu | internal team | yes | no | yes | partial |
| WeCom smart bot (long-poll) | internal team | no | no | yes | yes |
| WeCom self-built app | external WeChat users | yes | yes | no | no |
| WeChat customer service | external WeChat users | yes | yes | no | no |
| WeChat public account | followers | yes | yes | no | no |
| WorkBuddy (Tencent QClaw) | personal WeChat/QQ | no | yes | yes | yes |

Two notes: Feishu is officially listed as deprecated for new openclaw-china installs as of March 2026 — pick DingTalk or WeCom. WeChat public accounts in subscription mode have a 5-second passive reply window and cannot push proactively; service accounts and test accounts don't.

## The three-question picker

Don't read the matrix top to bottom. Ask yourself three questions:

1. **Who is on the other end?** A coworker → DingTalk or WeCom long-poll. An external WeChat user → WeCom self-built app or WeChat customer service. Yourself → WorkBuddy.
2. **Do you have a public IP?** No → only DingTalk, WeCom long-poll, and WorkBuddy are options. Yes → everything is on the table.
3. **Do you need group chat?** Yes → DingTalk, WeCom long-poll, or WorkBuddy. The customer-service and public-account channels are 1:1 only.

In practice I land on one of two stacks:

- **Internal team, no public IP:** DingTalk for the human channel, WeCom long-poll as backup. Heartbeat patrols both.
- **External WeChat reach:** WeCom self-built app for the agent's primary surface, WeChat customer service for inbound from non-tagged users.

## DingTalk — the path that always works

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_2.jpg)

Stream mode (long-poll) is the only reason DingTalk is fast to ship. No public IP, no domain verification, just register a bot and grab two values:

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

## WeCom — three flavors, pick by audience

The "WeCom" name covers three different APIs that share nothing except the brand:

- **Smart bot (智能机器人, 长连接).** Internal use only. No public IP needed. Easiest install. Streaming works.
- **Self-built app (自建应用).** Reaches external WeChat users *if* the org has the WeCom-WeChat互通 license. Requires public IP and a verified domain. No streaming.
- **Customer service (微信客服).** External WeChat users without the互通 license. Routes from public-account menus, mini-programs, video accounts, livestream entries. Markdown not supported.

For 90% of "we need a bot in our team" cases, the smart bot is what you want. The self-built app comes up when the goal is "talk to our customers on WeChat from inside our agent system" — and at that point you're also going to want to read [the WeChat专题](../04-channels/06-wechat-workbuddy.md).

## What I tell people who just want to play

WorkBuddy. It's Tencent's official QClaw bridge, runs on your desktop, no IP, no app review, talks to your real personal WeChat and QQ. It is the only "I just want my agent in WeChat tonight" answer that doesn't end in regret.

The catch: it's a desktop application, so it lives or dies with that machine being awake. Fine for a personal assistant. Bad for production.

## The lesson behind the matrix

Channels are where the agent actually *lives*. A great agent on a flaky channel is a flaky agent. So before building anything fancy, pick the channel that matches your audience and your network, and over-invest in making *that* channel rock-solid. Then add a second one for redundancy.

Everything else is rearranging deck chairs.
