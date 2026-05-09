---
title: "OpenClaw 快速上手（五）：接入聊天渠道——Telegram、钉钉和微信的第一步"
date: 2026-04-07 09:00:00
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

## 为什么要接入聊天渠道

前四篇我们搭建了一个能跑、会思考、有记忆的 Agent。但它一直活在终端里——你需要 SSH 到服务器，敲命令，盯着输出。这不是一个"助手"该有的样子。

一个真正有用的 Agent，应该是你打开日常用的聊天工具，@它一下，它就来了。这就是 Channel（渠道）的意义：**把 Agent 从"你去找它"变成"它在你身边"。**

OpenClaw 目前支持三条渠道路径：

| 渠道 | 适用场景 | 需要公网 IP | 配置难度 |
|------|---------|------------|---------|
| Telegram | 海外用户、个人使用 | 否（长轮询） | 低 |
| 钉钉 | 国内企业、团队协作 | 否（Stream） | 中 |
| 微信（WorkBuddy） | 个人微信触达 | 否（本地桥接） | 中高 |

![渠道架构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_1.jpg)

---

## Telegram：最简单的起步方式

![渠道路由架构——消息从 IM 平台经网关到 Agent 的完整链路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/fig_channels.png)

如果你只想最快看到效果——十分钟内让 Agent 在手机上回复你——Telegram 是首选。

### 创建 Bot

1. 在 Telegram 搜索 `@BotFather`，发送 `/newbot`
2. 输入 bot 名称和用户名（必须以 `bot` 结尾），拿到 `bot_token`
3. 发送消息给 `@userinfobot` 获取你的数字 user_id

### 配置文件

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
    allowed_user_ids:
      - 987654321
      - 112233445
    polling_timeout: 30
    parse_mode: "Markdown"
```

### 安全：allowed_user_ids 是强制的

Telegram bot 默认对所有人开放。如果 bot 背后是一个能执行 shell 命令的 Agent，任何人都能操控你的服务器。所以 OpenClaw 不允许省略这个字段——**这是必须的访问控制，不是可选的安全加固。**

### 常见问题排查

**1. Bot 在群组里不响应** -- Telegram bot 默认开启 Privacy Mode，群组中只收到 `/command` 或 @提及。找 BotFather 发 `/setprivacy` 选 Disable 即可。

**2. Webhook 冲突** -- 之前用过 webhook 模式的 bot 换 polling 会冲突，先执行：
```bash
curl "https://api.telegram.org/bot<TOKEN>/deleteWebhook"
```

**3. 消息格式乱码** -- `parse_mode: "Markdown"` 时，Agent 回复中的 `_`、`*` 等未转义字符会导致发送失败。建议改用 `parse_mode: "HTML"`。

**4. 触发速率限制** -- 同一聊天每秒约 1 条，全局每秒约 30 条。Agent 回复过长被拆分时容易触发 429。OpenClaw 内置退避重试。

**5. 国内服务器连不上** -- Polling 模式下日志持续 timeout，说明服务器到 Telegram API 不通。国内服务器需要配置代理。

---

## 钉钉：国内开发者的首选

对于国内团队，钉钉几乎是绕不开的。而钉钉的 Stream 模式（长连接）让接入变得出奇简单——**不需要公网 IP，不需要域名，不需要 nginx 反代。**

### 为什么 Stream 模式是杀手级特性

传统 bot 接入需要暴露 HTTPS 回调地址——公网 IP、域名、SSL 证书、防火墙，开发阶段还得 ngrok 穿透。Stream 模式基于 WebSocket 长连接：你的程序主动连钉钉服务器，消息推下来，不需要别人知道你的 IP。开发机在公司内网、没有公网 IP（国内绝大多数开发者的日常），这几乎是唯一合理的选择。

### 配置步骤

**第一步：创建应用**

1. 访问 [钉钉开放平台](https://open-dev.dingtalk.com/)
2. 进入「应用开发」->「企业内部开发」->「创建应用」
3. 记录 `ClientID` 和 `ClientSecret`

**第二步：配置机器人**

1. 在应用页面左侧找到「机器人」，启用
2. 消息接收模式选择「Stream 模式」
3. 配置单聊/群聊场景

**第三步：添加权限并发布**

在「权限管理」中添加 `qyapi_robot_sendmsg` 等权限。然后——**关键一步**——在「版本管理与发布」中创建版本并发布。应用必须发布后才能生效，这是最多人踩的坑。

### OpenClaw 配置

```yaml
channels:
  dingtalk:
    enabled: true
    client_id: "dingXXXXXXXXXX"
    client_secret: "your-client-secret-here"
    mode: "stream"
    allowed_user_ids:
      - "user123"
    allowed_group_ids:
      - "chatXXXXXXXX"
```

### WebSocket 架构

核心在于方向：你的程序主动连钉钉（WebSocket），消息推送到这条连接上，OpenClaw 处理后回调 API 回复。完美规避"没有公网 IP"的问题。

### 踩坑记录

**应用没发布** -- 我第一次配了半小时所有参数都对，就是收不到消息，最后发现没点「发布」。开发中的应用对其他用户不可见，和 Telegram（创建即可用）完全不同。

**限流判断** -- 钉钉机器人有频率限制（单聊/群聊各 20 条/秒）。日志中 429 或 "request too frequent" 时等待即可恢复，OpenClaw 会标记 `[RATE_LIMITED]`。

---

## 微信：WorkBuddy 桌面桥接

微信是国内用户最高频的通讯工具，但也是接入最复杂的——微信没有公开的 bot API。

### 为什么不用 Wechaty / itchat

这类工具通过逆向微信协议实现自动化。问题是微信官方持续封杀，账号被封风险极高（我个人就差点翻车）。**OpenClaw 选择了 WorkBuddy——一个合规的桌面桥接方案。**

### WorkBuddy 是什么

WorkBuddy 是 OpenClaw 配套的桌面客户端：

1. 你在电脑上运行 WorkBuddy 程序
2. 它用你的微信账号登录（和电脑版微信一样）
3. 收到的消息通过本地 HTTP 接口转发给 OpenClaw
4. OpenClaw 处理后通过 WorkBuddy 回复

本质是一个"受控的微信桌面客户端"，不是协议破解。

### 配置

1. 在 [WorkBuddy 官网](https://workbuddy.openclaw.dev) 注册，等待审核（1-2 个工作日）
2. 下载桌面客户端，扫码登录微信
3. 客户端在本地开放 HTTP 端口

```yaml
channels:
  wechat:
    enabled: true
    provider: "workbuddy"
    endpoint: "http://localhost:9100"
    allowed_contacts:
      - "friend_wxid_xxx"
    allowed_groups:
      - "group_wxid_xxx"
```

### 限制

- **桌面程序必须保持运行** —— 关掉就断
- **同一微信号不能两台电脑同时登录** —— 微信本身的限制
- **消息延迟 1-3 秒** —— 比 Telegram/钉钉的毫秒级慢
- **仅支持文本和图片** —— 小程序、视频号等不支持

如果需要 7x24 运行，建议在一台始终开机的工作站上部署 WorkBuddy。

![多渠道架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/05-channels/illustration_2.jpg)

---

## 多渠道并行运行

OpenClaw 支持同时启用多个渠道，把各 channel 的 `enabled` 设为 `true` 即可。

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "..."
    allowed_user_ids: [...]
  dingtalk:
    enabled: true
    client_id: "..."
    client_secret: "..."
    mode: "stream"
  wechat:
    enabled: true
    provider: "workbuddy"
    endpoint: "http://localhost:9100"
```

### 消息路由与 Session 隔离

Gateway 层负责路由：从哪个渠道来的消息，回复走哪条路回去。Agent 是同一个实例（工具和知识库共享），但会话按渠道 + 用户隔离——Telegram 用户 A 的历史不会出现在钉钉用户 B 的上下文里。

如果需要跨渠道打通同一用户的会话：

```yaml
session:
  cross_channel: true
  user_mapping:
    - telegram_id: 987654321
      dingtalk_id: "user123"
      label: "我"
```

---

## 渠道安全注意事项

### 访问控制

所有渠道强制配置白名单（Telegram: `allowed_user_ids`; 钉钉: `allowed_user_ids` + `allowed_group_ids`; 微信: `allowed_contacts` + `allowed_groups`）。没有白名单的渠道拒绝启动。

### 速率限制

每个渠道可独立配置：

```yaml
channels:
  telegram:
    rate_limit:
      per_user: 10/min
      global: 60/min
  dingtalk:
    rate_limit:
      per_user: 20/min
      per_group: 30/min
```

超限时用户会收到提示消息，而不是沉默。

### Token 泄露应急

**Telegram:** 找 BotFather 发 `/revoke` 重新生成 token，更新配置重启。**钉钉:** 进开放平台重置 ClientSecret，更新配置重启，检查调用日志。

**通用建议:** 不把 token 提交到 Git；用环境变量引用 `bot_token: "${TELEGRAM_BOT_TOKEN}"`；定期轮换凭证。

---

## 我的实际部署选择

- **日常**：钉钉 Stream 模式。团队都在钉钉上，@机器人即用。
- **海外/个人**：Telegram。配置最简，全球可达。
- **微信**：暂未启用。需要始终在线的桌面端，维护成本不值得。

如果你在国内团队，建议**先从钉钉开始**——Stream 模式零运维成本，投入产出比最高。

---

## 下一篇

渠道配好了，Agent 能在聊天工具里响应了。下一篇进入 **Prompt Engineering 和 System Prompt 设计**，让 Agent 成为某个领域的专家。

---

## 本篇速查

| 操作 | 命令 |
|------|-----|
| 启动带渠道的 Agent | `openclaw start --config config.yaml` |
| 查看渠道连接状态 | `openclaw status channels` |
| 测试钉钉连通性 | `openclaw test channel dingtalk` |
| 查看消息日志 | `openclaw logs --channel telegram --tail 50` |
| 清除 Telegram webhook | `curl https://api.telegram.org/bot<TOKEN>/deleteWebhook` |
