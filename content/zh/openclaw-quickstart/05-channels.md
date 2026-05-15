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
series_total: 10
description: "把 Agent 从终端搬到聊天工具里：Telegram 十分钟配通、钉钉 Stream 模式免公网 IP、微信 WorkBuddy 桌面桥接。三条路径的完整配置和踩坑记录。"
disableNunjucks: true
translationKey: "openclaw-quickstart-5"
---
OpenClaw 的核心理念是让 Agent 主动来找你——而目前它还只停留在 TUI（文本用户界面）里。现在，是时候为它接上真正的聊天通道了。

![OpenClaw 快速入门（5）：连接 Telegram、钉钉和微信 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_1.png)

---

## Telegram — 五分钟搞定

![通道路由架构 —— 消息从即时通讯平台通过网关的流动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/fig_channels.png)

即使你不打算在生产环境中使用 Telegram，我也强烈建议从这里开始。它的配置最为清爽，没有额外依赖，能让你快速验证 Agent 的端到端流程是否通畅。

**第一步。** 在 Telegram 中找到 `@BotFather`，发送 `/newbot`，给你的机器人起个名字，然后你会拿到一个形如 `7891234567:AAH...` 的 token。

**第二步。** 获取你自己的用户 ID。最简单的方法是给 `@userinfobot` 发一条消息，它会直接告诉你。我的 ID 是一串 9 位数字。

**第三步。** 在 `openclaw.json` 中添加一个 `telegram` 配置块：

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

这里有两点必须强调：

- **`allowed_user_ids`** 虽然在语法上可选，但实际使用中绝不能省略。否则，任何偶然发现你机器人的人都能与之交互。上线前务必配置此项。
- **`polling: true`** 是推荐的默认选项。Webhook 模式要求你拥有公网 HTTPS endpoint，而轮询（polling）模式则能在 NAT 后、公司 VPN 内，甚至任何网络环境下正常工作。

**第四步。** 重启 gateway：

```bash
openclaw gateway restart
```

重启后，你应该在日志中看到类似这样的输出：

```toml
[telegram] polling started, listening as @your_bot_name
```

**第五步。** 打开 Telegram，找到你的机器人，发送 “hello”。你应该会收到和《指南（二）》中 TUI 返回完全相同的单行回复。如果机器人毫无反应，90% 的可能性是你忘记把自己的用户 ID 加入 `allowed_user_ids` 列表。

### Telegram 常见问题排查

即便配置简单，也难免遇到障碍。以下是五个高频问题：

**1. 机器人不回复 —— 检查 `allowed_user_ids`。** 你发了消息，但机器人无动于衷。请仔细核对配置文件中的用户 ID 是否与 `@userinfobot` 返回的完全一致。常见错误包括复制了用户名而非数字 ID，或遗漏了 JSON 数组的方括号。若确为此问题，gateway 日志会明确提示：`[telegram] message from unauthorized user 987654321, ignoring`。

**2. 消息延迟 —— 检查轮询间隔。** 默认轮询间隔为 1 秒，几乎感觉不到延迟。如果你为了减少 API 调用将 `polling_interval` 改为 30 秒之类，那这就是延迟的根源。对于交互式聊天，请保持为 1 秒。

**3. 回复说到一半就中断 —— 检查 agent 配置中的 `max_tokens`。** 如果 Agent 开始回复却突然截断，问题通常出在 `agents.json`，而非 Telegram 通道配置。请检查 Pi Agent 配置块中的 `max_tokens` 字段。对于对话场景，建议将其提升至 2048 或 4096。

**4. 媒体支持有限 —— 哪些文件类型可用？** Telegram 机器人可接收文本、图片（JPEG/PNG）、文档（PDF/TXT/JSON）和语音消息。但视频和贴纸无法被原生有效处理，除非你额外实现解析逻辑。例如，当你发送 PDF 时，gateway 会下载文件并把路径传给 Agent，但 Agent 必须具备读取 PDF 的技能才能继续处理。

**5. 群聊支持 —— 如何启用群模式？** 默认情况下，机器人仅响应私聊消息。若要加入群聊，需通过 BotFather 启用群组权限：发送 `/setjoingroups`，选择你的机器人，点击 “Enable”。在群聊中，机器人默认只响应被 @ 提及的消息（如 `@your_bot_name what's the weather`），除非你在 BotFather 中通过 `/setprivacy` 关闭隐私模式。

## 钉钉 — 十五分钟

钉钉的接入更复杂，因为它要求你注册一个正式的“机器人应用”并申请相应权限。简要步骤如下：

1. **注册钉钉开放平台应用。** 访问 `https://open-dev.dingtalk.com`，创建一个类型为 “Stream Mode” 的应用，并记下 `Client ID` 和 `Client Secret`。
2. **授予必要的消息权限。** 至少需要 `Contact.User.Read` 和机器人消息相关的 scopes。
3. **将机器人添加到群聊。** 在钉钉中新建或选择一个群，进入设置，添加你的机器人。

随后，在通道配置中填入：

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

Stream Mode 是钉钉为自托管机器人提供的现代方案：gateway 会主动向钉钉服务器建立一个长连接 WebSocket，无需暴露公网 webhook。这是钉钉近年来对开发者体验的最大改进。

### Stream Mode WebSocket 的工作原理

理解这一机制很重要，因为它正是你能在 NAT 后的笔记本上运行机器人的关键。

传统 Webhook 模式下，钉钉服务器会在每次收到消息时，向你的公网 HTTPS endpoint 发起 POST 请求。这意味着你需要公网 IP、域名、有效的 SSL 证书，还要开放防火墙的 443 入站端口。

而 Stream Mode 完全反转了这一流程：你的 gateway 主动发起一个 **出站** WebSocket 连接到 `wss://stream.dingtalk.com` 并保持连接。当有人发送消息时，钉钉会通过该连接推送一个事件帧；gateway 接收后解码 JSON 并路由给 Agent，回复也沿同一通道返回。整个过程无需公网 IP、DNS 解析或证书管理。

### 沙箱测试

钉钉开发者控制台提供沙箱环境。在沙箱中，你会获得一个测试用的 `Client ID` 和一个模拟群聊，可在不影响真实用户的情况下调试机器人。一旦沙箱测试通过，只需将配置中的 Client ID 和 Secret 替换为生产凭证即可上线。

重启 gateway 后，在钉钉群中 @ 你的机器人，例如：`@Lobster how are you`。如果收到回复，说明配置成功。若无响应，最常见的原因是企业管理员尚未审批应用权限，请前往钉钉管理后台确认。

## 微信 — 大实话版本

![OpenClaw 快速入门 (5)：连接 Telegram、DingTalk 和微信 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/05-channels/illustration_2.png)

微信有三条接入路径，但只有一条真正靠谱。

**路径 1：openclaw-china（社区插件）。** 它封装了一个非官方微信协议。千万别用。腾讯会封禁此类账号，我一位朋友就差点因此永久失去个人微信。

### 为什么路径 1 实际上很危险

腾讯的风控机制相当精密，远不止检查客户端版本字符串那么简单。其封号策略包括：

- **IP 关联分析：** 如果你的账号突然从数据中心 IP 段发送消息，系统会立即警觉——个人微信以移动端为主，服务器 IP 极不寻常。
- **协议指纹识别：** 非官方协议虽试图模拟官方客户端，但在数据包时序、心跳间隔、握手字段顺序等细节上往往存在偏差。
- **消息行为模式检测：** 正常人不会在一分钟内精准回复 50 条格式工整的 Markdown 消息。这种异常行为极易触发风控。

封号未必即时发生。腾讯有时会让账号运行数日，持续收集行为数据，再集中执行永久封禁。一旦被封，基本无法解封。

**路径 2：企业微信。** 它有官方 OpenClaw 集成，运行稳定，但前提是你要有注册的企业微信账号，且机器人只能在企业内部渠道中使用。适合团队协作，但对个人用户来说非常别扭。

**路径 3：WorkBuddy。** 这是腾讯自家的桌面客户端，允许已注册的 AI Agent 通过你的个人微信进行对话。这是目前唯一官方支持个人微信机器人的方案。

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

整个流程预计需要几天时间：

1. **申请：** 访问 `https://workbuddy.weixin.qq.com`，用微信扫码登录，提交开发者申请。
2. **1–3 天审核：** 腾讯团队会人工审核。“个人助理”类用途通常能通过；“营销机器人”则可能被拒。
3. **配置：** 审核通过后，你会获得 WorkBuddy ID 和 API 凭证。下载桌面客户端，登录并绑定你的 Agent。
4. **测试：** 给机器人的微信联系人发一条消息，应能通过 WorkBuddy 路由到 OpenClaw 并返回回复。
5. **上线：** 测试成功后，你就可以从手机或其他微信客户端与机器人正常聊天了。

如果你确实需要微信支持，请选择 WorkBuddy，并预留一个晚上完成整个注册与配置流程。

## 多通道配置

你不必只选一个通道。gateway 可同时运行多个通道。

### gateway 如何复用多个通道

当你同时启用 `telegram` 和 `dingtalk`，gateway 会为每个通道启动一个监听线程：Telegram 线程每秒轮询一次 API，钉钉线程则维持一个 WebSocket 长连接。两个线程都将消息送入同一个内部队列，由调度器统一转发给对应的 Agent。

### 配置示例：Telegram + 钉钉同时启用

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

上述配置让两个通道都路由到同一个 Pi Agent。你也可以通过修改 `agent` 字段，将不同通道指向不同的 Agent。

## 通道健康监控

### gateway 日志

每个通道启动时及运行期间都会定期输出心跳日志，例如：

```toml
[telegram] polling started, listening as @your_bot_name
[telegram] heartbeat: 142 messages processed, 0 errors
[dingtalk] stream connected, session_id=abc123
```

如果某通道的心跳日志停止更新，说明该通道已断开。

### `openclaw status` 命令

你还可以使用内置命令查看通道状态：

```bash
openclaw status
# Gateway: running (PID 12345)
# Channels:
#   telegram: active, last message 12s ago
#   dingtalk: active, last message 3m ago
```

## 安全注意事项

### 速率限制

gateway 内置了速率限制器，默认限制为每位用户每分钟最多 10 条消息。你可以在 `openclaw.json` 中调整：

```json
"rate_limit": {
  "messages_per_minute": 10,
  "burst": 20
}
```

### 用户白名单

Telegram 使用 `allowed_user_ids`，钉钉使用 `allowed_staff_ids`。如果你的机器人具备访问敏感资源的能力（如文件系统、邮件、内部 API），务必通过硬编码白名单严格控制可调用用户。

### Token 轮换

Telegram 的 `bot_token` 和钉钉的 `client_secret` 均为长期有效的凭证，建议每 90 天轮换一次。Telegram 可在 BotFather 中使用 `/token` 命令重新生成；钉钉则需在开发者控制台重新签发新的 secret。

## 如何选择起始通道

- 如果你身处中国境外，只想快速跑通：选 **Telegram** —— 五分钟搞定，无需审核。
- 如果你的团队已在使用钉钉，希望机器人加入工作群：选 **钉钉 Stream Mode** —— 十五分钟是一次性投入。
- 如果你坚持要用微信：请为 **WorkBuddy** 预留一整个晚上。

## “接好线”到底意味着什么

完成本文操作后，你至少会有一个通道能够：

1. 从真实的聊天平台接收消息；
2. 将消息路由给与 TUI 相同的 Pi Agent；
3. 在同一聊天窗口中返回回复。

正是这第三点，让 Agent 真正有了“助手”的感觉，而不再只是一个命令行工具。无论你是让它读文件、做搜索，还是总结文档，所有操作都经由 gateway 执行——你甚至能在机器人回复的过程中，实时看到 gateway 日志滚动输出。这种透明度，在大多数商业产品中极为罕见。

下一篇，我们将让这个聊天机器人真正为你干活：引入 Skills 系统，并部署一个用于浏览器自动化的 MCP 服务器。
