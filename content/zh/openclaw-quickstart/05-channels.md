---
title: "OpenClaw 快速上手（五）：接 Telegram、钉钉，以及微信的真实情况"
date: 2026-04-07 09:00:00
tags:
  - openclaw
  - telegram
  - dingtalk
  - wechat
  - 渠道
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 5
description: "Telegram 五分钟搞定，钉钉十五分钟，微信“看情况”。每个渠道的具体配置块，以及“按你住在哪儿”挑起步渠道的诚实建议。"
disableNunjucks: true
translationKey: "openclaw-quickstart-5"
---
OpenClaw 的核心思想是让 Agent 主动来找你。但目前为止，我的 Agent 还没做到这一点——它只停留在 TUI 界面。现在是时候接入一个渠道了。
![OpenClaw 快速上手（五）：接 Telegram、钉钉，以及微信的真实情况 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_1.jpg)

## Telegram——5 分钟

我总是建议从这里开始，即使你并不打算在生产环境中使用 Telegram。它的配置是最简单的，没有其他干扰因素，可以轻松完成端到端的验证。

**第一步** 打开 Telegram，找 `@BotFather` 对话。发送 `/newbot`，给它起个名字，然后你会得到一个类似 `7891234567:AAH...` 的 token。

**第二步** 找到自己的用户 ID。最简单的方法是给 `@userinfobot` 发条消息，它会直接告诉你。我的 ID 是一个 9 位数。

**第三步** 在 `openclaw.json` 文件中添加 `telegram` 配置块：

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

这里有两点需要注意：

- **`allowed_user_ids`** 虽然不是强制字段，但非常重要。如果不设置，任何找到你 Bot 的人都能和它交互。别忘了加上。
- **`polling: true`** 是推荐的默认值。Webhook 模式需要一个公网 HTTPS 地址，而 Polling 模式可以在 NAT 后、公司 VPN 内，或者任何地方正常工作。

**第四步** 重启网关服务：

```bash
openclaw gateway restart
```

日志里应该会出现这样的内容：

```
[telegram] polling started, listening as @your_bot_name
```

**第五步** 打开 Telegram，找到你的 Bot，发送 "hello"。你应该会收到和第二篇教程里 TUI 返回的相同的一行回复。如果 Bot 没有回应，90% 的可能是因为你忘记把自己的用户 ID 加到 `allowed_user_ids` 里了。
## 钉钉——15 分钟

钉钉的配置稍微复杂一些，因为它需要一个真正注册过的“机器人应用”，并且要分配好权限。简单来说，分三步走：

1. **注册钉钉开放平台应用**  
   打开 `https://open-dev.dingtalk.com`，创建一个类型为“Stream Mode”的应用。记下 `Client ID` 和 `Client Secret`。

2. **分配消息相关权限**  
   至少需要 `Contact.User.Read` 和机器人消息相关的权限范围。

3. **把机器人加入群聊**  
   在钉钉里建一个群，进入群设置，添加你的机器人。

接下来是渠道配置：

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

Stream Mode 是一种现代化的方式。它通过网关与钉钉服务器建立一条长连接的 WebSocket，因此不需要公网 webhook。这是钉钉在自托管机器人方面最大的改进。

重启服务后，在群里试试 @ 一下机器人：`@Lobster 你好啊`。如果机器人回复了，说明配置成功。如果没有反应，最常见的原因是企业管理员还没有批准应用权限。这时可以去钉钉管理后台检查一下。
## 微信——说实话

![OpenClaw 快速上手（五）：接 Telegram、钉钉，以及微信的真实情况 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_2.jpg)

三条路，只有一条靠谱。

**第一种：openclaw-china（社区插件）。** 这个插件封装了一个非官方微信协议。千万别用。腾讯会封禁使用它的账号，我有个朋友差点因此丢了自己的个人号。

**第二种：企业微信。** 它有官方的 OpenClaw 集成，功能不错，但需要注册企业微信账号，而且机器人只能在企业内部频道里运行。适合团队协作，但用来处理个人事务就很别扭。

**第三种：WorkBuddy。** 腾讯自家的桌面客户端，允许注册的 AI Agent 通过你的个人微信进行对话。这是目前官方唯一支持的个人微信解决方案。具体操作是先注册成为 WorkBuddy 开发者，获取 `workbuddy_id`，然后把 OpenClaw 配置指向它。

```json
"channels": {
  "workbuddy": {
    "enabled": true,
    "workbuddy_id": "wb_...",
    "api_key": "..."
  }
}
```

如果想用微信，就选 WorkBuddy。OpenClaw 官方文档里详细写了注册步骤——大概花一个小时，大部分时间是在等腾讯审核。
## 如何选择起步渠道

如果不在国内，想快速上手：**Telegram**。5 分钟搞定，无需审核。

团队已经在用钉钉，希望机器人进群：**钉钉 Stream Mode**。一次性花 15 分钟配置即可。

想要微信：留出一个晚上给 WorkBuddy。
## “接好了”到底是什么意思

看完这篇文章，你应该至少配置好了一个通道：

1. 能接收来自真实聊天平台的消息  
2. 把消息路由到 TUI 所连接的同一个 Pi Agent  
3. 在同一个聊天中返回回复  

最后一点是让它更像一个助手而不是命令行工具的关键。我让 Agent 做的所有事情——读取文件、运行搜索、总结文档——都通过 Gateway 完成。聊天还没回复完时，我就能看到 Gateway 的日志在滚动。这种透明性在商业产品中很少见。

下一篇会讲如何把 Agent 从一个聊天机器人变成能为你完成特定任务的工具：Skills，再加上一个用于浏览器自动化的 MCP 服务器。
