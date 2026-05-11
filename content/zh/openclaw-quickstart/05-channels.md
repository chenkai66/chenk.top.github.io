---
title: "OpenClaw 指南（五）：对接 Telegram、钉钉与微信"
date: 2026-04-12 09:00:00
tags:
  - openclaw
  - telegram
  - dingtalk
  - wechat
  - channels
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 5
description: "把 Agent 从终端搬到聊天工具里：Telegram 十分钟配通、钉钉 Stream 模式免公网 IP、微信 WorkBuddy 桌面桥接。三条路径的完整配置和踩坑记录。"
disableNunjucks: true
translationKey: "openclaw-quickstart-5"
---
OpenClaw 的设计初衷是让 Agent 能主动与你交互。目前它仅支持 TUI（文本界面）。现在，是时候接入一个聊天通道了。

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_1.png)

## Telegram — 五分钟搞定

![Channel routing architecture — message flow from IM platforms through the gateway](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/fig_channels.png)

即使你无意在生产环境使用 Telegram，也建议优先选用它作为接入起点。其配置最简洁，无需额外依赖，可完整验证 Agent 的端到端流程。

**第一步。** 在 Telegram 里找 `@BotFather`。发送 `/newbot`，起个名字，拿到一个形如 `7891234567:AAH...` 的 token。

**第二步。** 找到你自己的用户 ID。最简单的办法：给 `@userinfobot` 发条消息，它会告诉你。我的是一串 9 位数字。

**第三步。** 在 `openclaw.json` 里加一个 `telegram` 块：

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

有两点必须强调：

- **`allowed_user_ids`** 本质上不能省。不加这个，任何找到你 bot 的人都能让它响应。别省这一步。
- **`polling: true`** 是默认首选。Webhook 模式需要公网 HTTPS  endpoint。Polling 模式在 NAT 后面、公司 VPN 里、任何地方都能跑。

**第四步。** 重启 gateway：

```bash
openclaw gateway restart
```

日志里应该能看到：

```
[telegram] polling started, listening as @your_bot_name
```

**第五步。** 打开 Telegram，找到你的 bot，发个 "hello"。你应该能收到和第二篇里 TUI 返回一样的单行回复。如果 bot 没反应，90% 是因为你忘了把自己的用户 ID 填进 `allowed_user_ids`。

### 排查 Telegram 问题

哪怕配置这么简单，也难免踩坑。这是我反复遇到的五个问题：

**1. Bot 不回复 — 检查 `allowed_user_ids`。** 你发了消息，bot 不理你。再三确认 config 里的用户 ID 和 `@userinfobot` 告诉你的完全一致。常见错误包括抄成了用户名而不是数字 ID，或者忘了 JSON 数组的方括号。如果是这个问题，gateway 日志会显示 `[telegram] message from unauthorized user 987654321, ignoring`。

**2. 消息延迟 — 检查 polling 间隔。** 默认 polling 间隔是 1 秒，感觉上是实时的。如果你把 `polling_interval` 改成了 30 秒想减少 API 调用，那就是延迟的来源。交互式对话就保持在 1 秒。

**3. Bot 回复但说到一半断了 — 检查 agent config 里的 `max_tokens`。** 如果 Agent 开始回复却突然停止，问题通常在 `agents.json` 而不是 Telegram 通道配置。找找 Pi Agent 块里的 `max_tokens`。对话场景把它调到 2048 或 4096。

**4. 不支持媒体 — 哪些文件类型可用。** Telegram bot 能接收文本、图片（JPEG/PNG）、文档（PDF/TXT/JSON）和语音消息。如果没有额外处理，它们无法原生处理视频或贴纸。如果你发一个 PDF，gateway 会下载它并把文件路径传给 agent — 但 agent 需要一个知道怎么读 PDF 的 skill。

**5. 群聊 — 如何开启群模式。** 默认情况下，bot 仅响应私聊消息。要把它加进群，告诉 BotFather 允许群访问：发送 `/setjoingroups`，选中你的 bot，选 "Enable"。在群里，bot 只能看到提及它的消息（`@your_bot_name what's the weather`），除非你在 BotFather 里用 `/setprivacy` 关闭 Privacy Mode。

## 钉钉 — 十五分钟

钉钉接入更复杂，需注册一个具备相应权限的官方“机器人应用”。简单说：

1. **注册钉钉开放平台应用。** 去 `https://open-dev.dingtalk.com`，创建一个类型为 "Stream Mode" 的应用。记下 `Client ID` 和 `Client Secret`。
2. **授予应用消息 scope。** 至少需要 `Contact.User.Read` 和 bot 消息相关的 scope。
3. **把 bot 加进群。** 在钉钉里建个群，进设置，添加你的 bot。

然后是通道配置：

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

Stream Mode 是现代做法 — 它利用从 gateway 到钉钉服务器的长连 WebSocket，所以你不需要公网 webhook。这是钉钉为自托管 bot 做的最大改进。

### Stream Mode WebSocket 到底怎么工作

理解这一架构很重要，因为它使 gateway 能在 NAT 后的笔记本上正常运行。

传统 webhook 模式：钉钉服务器每次有人发消息都向你的 endpoint 发起 HTTPS POST。这意味着你需要公网 IP、域名、有效的 SSL 证书，防火墙还得放行入站 443 端口。

Stream Mode 反过来了：你的 gateway 向 `wss://stream.dingtalk.com` 发起一个 **出站** WebSocket 连接并保持存活。有人发消息时，钉钉通过这个 WebSocket 推送事件帧。你的 gateway 读取帧，解码 JSON，路由给 agent。回复也通过同一个 WebSocket 传回去。不需要入站端口，不需要 DNS，不需要证书管理。

### 沙箱测试

钉钉开发者控制台提供沙箱环境。沙箱模式下，你拿到一个测试 `Client ID` 和一个模拟群，可以在不部署给真实用户的情况下给 bot 发消息。沙箱跑通了，生产环境只是替换 config 里的 Client ID 和 Secret。

重启后，在钉钉群里提及你的 bot：`@Lobster how are you`。如果 agent 回复，就完成了。如果没反应，最常见的原因是应用权限没过企业管理员审批。检查钉钉管理后台。

## 微信 — 大实话版本

![OpenClaw QuickStart (5): Wiring Telegram, DingTalk, and the WeChat Reality — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_2.png)

有三条接入路径，其中仅 WorkBuddy 是官方支持且可靠的选择。

**路径 1：openclaw-china（社区插件）。** 这封装了一个非官方微信协议。别用。腾讯会封禁使用它的账号，我有个朋友差点因此丢了个人号。

### 为什么路径 1 其实很危险

腾讯的检测机制非常复杂。他们不只是看客户端版本字符串里的标记。封禁模式包括：

- **IP 关联：** 如果你的账号突然开始从数据中心 IP 段发消息，这就是危险信号。个人微信是移动优先的，服务器 IP 不正常。
- **协议指纹：** 非官方协议模拟微信客户端，但会在小细节上出错 — 包 timing、keepalive 间隔、某些握手字段的顺序。
- **消息速率和模式分析：** 正常人不会在一分钟内回复 50 条格式完美的 markdown 消息。异常检测会标记它。

封禁未必即时触发：腾讯可能先放行账号数日，持续采集行为数据，再集中执行封禁。一旦封禁，通常是永久的。

**路径 2：企业微信。** 有官方 OpenClaw 集成。 works well，但你需要注册企业微信账号，且 bot 只能在企业渠道内操作。适合团队；个人用很别扭。

**路径 3：WorkBuddy。** 腾讯自家的桌面客户端，让注册的 AI agent 通过你的个人微信对话。这是个人微信 agent 的官方支持路径。

```json
"channels": {
  "workbuddy": {
    "enabled": true,
    "workbuddy_id": "wb_...",
    "api_key": "..."
  }
}
```

### WorkBuddy 注册时间线

预计耗时几天：

1. **申请：** 去 `https://workbuddy.weixin.qq.com`，用微信登录，提交开发者申请。
2. **1-3 天审核：** 腾讯团队人工审核。“个人助理”通常能过；“营销 bot"可能会被拒。
3. **配置：** 批准后，拿到 WorkBuddy ID 和 API 凭证。下载桌面客户端，登录，绑定 agent。
4. **测试：** 给你的 bot 微信联系人发消息。它应该通过 WorkBuddy 路由到 OpenClaw 再回来。
5. **上线：** 测试通过后，你可以从手机或任何微信客户端跟 bot 聊天。

如需接入微信，请使用 WorkBuddy，并预留一晚时间完成注册与配置。

## 多通道 setup

你不必只选一个。gateway 可以同时跑多个通道。

### gateway 如何复用通道

当你同时启用 `telegram` 和 `dingtalk`，gateway 会为每个 spawn 一个 listener 线程。Telegram 每秒 poll 它的 API。钉钉保持 WebSocket 长连。两个线程都喂给同一个消息队列，dispatcher 再路由给 agents。

### 配置示例：Telegram + 钉钉一起用

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

两个通道都路由到同一个 Pi Agent。你也可以通过改 `agent` 字段把不同通道路由给不同 agent。

## 通道健康监控

### gateway 日志

每个通道都会写启动和定期心跳日志：

```
[telegram] polling started, listening as @your_bot_name
[telegram] heartbeat: 142 messages processed, 0 errors
[dingtalk] stream connected, session_id=abc123
```

如果看不到心跳日志，通道就挂了。

### `openclaw status` 命令

```bash
openclaw status
# Gateway: running (PID 12345)
# Channels:
#   telegram: active, last message 12s ago
#   dingtalk: active, last message 3m ago
```

## 安全考虑

### 速率限制

gateway 内置了 rate limiter。默认是每个用户每分钟 10 条消息。在 `openclaw.json` 配置：

```json
"rate_limit": {
  "messages_per_minute": 10,
  "burst": 20
}
```

### 用户白名单

Telegram 用 `allowed_user_ids`，钉钉用 `allowed_staff_ids`。如果你的 bot 能访问敏感工具（文件系统、邮件、内部 API），你需要一个硬性的名单来控制谁能调用它。

### Token 轮换

Telegram 的 `bot_token` 和钉钉的 `client_secret` 都是长活凭证。每 90 天轮换一次。Telegram 在 BotFather 里用 `/token` 重新生成。钉钉在开发者控制台 issuing 新 secret。

## 选择你的起始通道

如果你住在中国境外只想让它跑起来：**Telegram**。五分钟，无需审核。

如果你的团队已经在用钉钉且想要 agent 在群里：**钉钉 Stream Mode**。这十五分钟是一次性成本。

如果你想要微信：预留一个晚上给 WorkBuddy。

## “接好线”到底意味着什么

读完这篇，你应该至少有一个通道能：

1. 从真实聊天平台接收消息
2. 把它们路由到 TUI 对话的那个 Pi Agent
3. 在同一个聊天里返回回复

最后一点让它感觉像个助手而不是 CLI。你让 agent 做的每件事 — 读文件、跑搜索、总结文档 — 都通过 gateway 发生，你能在聊天还在回复中途时看到 gateway 日志滚动。这种可见性在商业产品里很少见。

下一篇，把 agent 从聊天机器人变成能为你做具体工作的东西：Skills，加上用于浏览器自动化的 MCP server。