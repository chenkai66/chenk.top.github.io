---
title: "OpenClaw 快速上手（九）：国内 IM 渠道选型——说点掏心窝的"
date: 2026-04-11 09:00:00
tags:
  - openclaw
  - dingtalk
  - wechat
  - wecom
  - channels
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 9
description: "钉钉、飞书（已弃用）、企业微信三种模式、微信客服、公众号，以及 WorkBuddy 桌面端桥接。一张选型矩阵和两条在 2026 年真正能跑通的路径的完整配置。"
disableNunjucks: true
translationKey: "openclaw-quickstart-9"
---

做国内 IM 渠道接入，跟做海外是两个完全不同的世界。海外你有 Slack、Discord、Telegram，协议开放，文档清晰，webhook 一配就完事。国内呢？钉钉的 Stream 模式文档散落在三个不同的页面，企业微信的"互通"能力需要你先搞懂它到底有几种机器人类型，微信生态更是一个深不见底的坑——个人号自动化早就被封杀得干干净净（别问我怎么知道的）。

这篇文章不讲空话。我会给你一张选型矩阵，三个筛选问题，两条实测跑通的路径，以及每个渠道的详细配置。所有配置都在 OpenClaw v0.9+ 上验证过。

![国内 IM 渠道选型矩阵总览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_1.png)

## 七渠道选型矩阵

![国内 IM 渠道选型决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/fig_picker.png)

先上结论。下面这张表是我在实际项目中反复验证后的总结：

| 渠道 | 适用场景 | 需要公网 IP | 触达微信用户 | 群聊支持 | 流式输出 | 延迟 | 消息格式 | 成本 |
|------|----------|-------------|--------------|----------|----------|------|----------|------|
| **钉钉 Stream** | 企业内部 | 否 | 否 | 是 | 伪流式(分段) | 200-400ms | Markdown/卡片/文件 | 免费 |
| **飞书** (已弃用) | - | - | - | - | - | - | - | 已停止维护 |
| **企微智能机器人** | 企业内部群聊 | 否 (长轮询) | 否 | 仅群聊 | 否 | 300-600ms | Markdown/图文 | 免费 |
| **企微自建应用** | 企业+外部客户 | 是 (回调) | 是 (需互通) | 是 | 否 | 200-500ms | 文本/图文/小程序卡片 | 互通许可 ~2000元/年 |
| **微信客服** | 外部客户触达 | 是 (回调) | 是 (原生) | 否 | 否 | 300-800ms | 文本/图片/菜单 | 免费 (需企微认证) |
| **微信公众号** | 品牌运营+客服 | 是 (回调) | 是 (原生) | 否 | 否 | 800-1500ms | 文本/图文/模板消息 | 认证费 300元/年 |
| **WorkBuddy** | 本地开发/演示 | 否 | 否 | 否 | 是 (原生) | 50-100ms | 富文本/文件 | 免费 |

几点说明：

- **飞书已弃用**：OpenClaw 在 v0.8 之后正式移除了飞书适配器。原因很简单——飞书的开放平台在 2025Q4 做了一次大改版，事件订阅机制完全重构，维护成本太高。如果你一定要用飞书，社区有个非官方 fork 可以参考，但我不推荐。
- **伪流式**：钉钉不支持真正的 SSE 流式推送，但支持「更新卡片」的方式模拟流式效果。用户体验接近流式，但实现上是分段替换。
- **互通许可**：企微自建应用要触达微信用户，需要购买「互通许可」，按人头计费，最低档大约 2000 元/年/100 人。

## 三个问题帮你做决策

不想看表？回答三个问题就够了：

**问题一：对面坐的是谁？**

- 公司内部同事 → 钉钉 Stream 或 企微智能机器人
- 外部客户（且客户用微信） → 企微自建应用 + 微信客服
- 本地开发调试 → WorkBuddy

**问题二：你有公网 IP 吗？**

- 没有（比如在公司内网、家里、或者不想折腾 ngrok） → 钉钉 Stream（天然长连接，不需要公网 IP）或 企微智能机器人（长轮询模式）
- 有（云服务器、已备案域名） → 企微自建应用、微信客服、公众号都可以

**问题三：需要群聊吗？**

- 是 → 钉钉 Stream（群内 @机器人）或 企微智能机器人（群聊专属）
- 否 → 根据前两个问题选择即可

## 两条推荐路径

根据上面的分析，2026 年真正能跑通且维护成本低的路径就两条：

### 路径 A：内部场景，无公网 IP

**钉钉 Stream + 企微智能机器人（长轮询）**

适合：内部工具、团队助手、知识库问答。不需要任何公网暴露，开发者本地就能跑。

### 路径 B：外部场景，触达微信用户

**企微自建应用 + 微信客服**

适合：客服机器人、销售助手、用户运营。需要公网 IP、ICP 备案、企业微信认证。前期成本高一些，但这是目前唯一合规触达微信用户的路径。

下面分渠道详细展开。

## 钉钉深度配置

### Stream 模式原理

钉钉的 Stream 模式是 2024 年推出的新协议，本质上是一个从客户端发起的 WebSocket 长连接。机器人启动时主动连接钉钉服务器，消息通过这条长连接推送下来。不需要公网 IP，不需要配置回调地址，对开发者极其友好。

### 获取凭证

路径：**钉钉开放平台 (open.dingtalk.com) → 应用开发 → 企业内部应用 → 创建应用**

1. 创建一个「企业内部应用」（不是第三方应用）
2. 在「凭证与基础信息」页面获取 `ClientId` 和 `ClientSecret`
3. 在「消息推送」页面启用 Stream 模式
4. 在「机器人」配置页面开启机器人能力
5. 在「权限管理」中添加 `qyapi_chat_manage` 权限（如果需要群聊）

### OpenClaw 配置

```yaml
# openclaw.channels.yaml
channels:
  dingtalk:
    adapter: "@openclaw/adapter-dingtalk"
    mode: stream
    credentials:
      clientId: "dingxxxxxxxxxxxxxxxx"
      clientSecret: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    options:
      reconnectMs: 25000          # 关键：必须小于 30000
      maxMessageLength: 2048      # 单条消息上限（Markdown 模式）
      cardMaxLength: 4096         # 卡片消息上限
      rateLimit: 20               # 每秒最大发送数
      enableGroupChat: true
      groupChatIds:               # 限定响应的群
        - "chatxxxxxxxxxxxxxxxx"
```

### 常见坑与解决方案

**坑一：30 分钟断连**

这是最常见的问题。钉钉 Stream 的 WebSocket 连接有一个 30 分钟的心跳超时。如果你的 `reconnectMs` 设置不当（比如设成 60000），连接会在 30 分钟后静默断开，机器人变成"死人"状态——看起来在线，但收不到任何消息。

解决方案：`reconnectMs` 设为 25000（25 秒），确保在超时前发送心跳包。OpenClaw 的钉钉适配器会自动处理心跳，你只需要确保这个值正确。

**坑二：消息长度截断**

钉钉 Markdown 消息上限 2048 字符，卡片消息上限 4096 字符。LLM 的回复经常超过这个长度。OpenClaw 默认会自动分段，但如果你的回复中有代码块，分段可能会破坏 Markdown 格式。

建议：在 prompt 层面限制输出长度，或者用卡片模式（支持折叠展开）。

**坑三：组织管理员审批**

企业内部应用需要组织管理员（通常是你们公司的 IT 部门或行政部门）审批后才能上线。开发阶段可以先在「开发环境」中测试，但正式上线需要走审批流程。

提前沟通，别等做完了才发现审批要一周。

**坑四：时钟漂移**

钉钉的签名验证对时钟敏感。如果你的服务器时间与 NTP 偏差超过 5 分钟，Stream 连接会被拒绝，错误信息是一个不太直观的 `invalid timestamp`。

解决方案：确保服务器开启 NTP 同步。Docker 环境下注意宿主机时间。

**频率限制**

钉钉的发送频率限制是 20 条/秒/应用。对于大多数场景足够了，但如果你在群里做批量通知，需要注意节流。OpenClaw 的适配器内置了令牌桶限流，超出部分会自动排队。

## 企业微信深度配置

企业微信是最复杂的渠道，因为它有三种完全不同的机器人形态。搞不清楚这三种的区别，你就会掉进坑里。

### 三种形态详解

**1. 智能机器人（群聊 Webhook）**

- 本质：一个群聊里的 Webhook 地址
- 能力：只能主动发消息到群里，不能接收用户消息（除非用长轮询模式）
- 适用：通知类场景（告警、日报、CI/CD 通知）
- 限制：不能 1v1 对话，不能触达微信用户

配置最简单，但能力也最弱。如果你只是要在群里发通知，用这个就够了。

**2. 自建应用**

- 本质：一个完整的企业微信应用，有独立的应用 ID 和密钥
- 能力：1v1 对话、群聊、触达微信用户（需互通许可）、接收消息回调
- 适用：客服机器人、业务助手
- 限制：需要公网 IP（接收回调）、需要 ICP 备案、互通许可有成本

这是功能最完整的形态，也是唯一能触达微信用户的方式。

**3. 微信客服**

- 本质：企业微信提供的客服功能，用户通过微信小程序/公众号/网页发起咨询
- 能力：接收微信用户消息、自动回复、人工转接
- 适用：售前咨询、售后客服
- 限制：只能被动接收消息，不能主动触达用户

### 智能机器人注册步骤

1. 登录 **企业微信管理后台 (work.weixin.qq.com)**
2. 进入 **应用管理 → 应用 → 创建应用**
3. 选择「机器人」类型
4. 获取 Webhook 地址和 Token
5. 配置消息接收（选择长轮询模式，不需要公网 IP）

```yaml
# openclaw.channels.yaml
channels:
  wecom-bot:
    adapter: "@openclaw/adapter-wecom"
    mode: smart-bot
    credentials:
      corpId: "wxxxxxxxxxxxxxxxxx"
      botKey: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
      token: "xxxxxxxxxxxxxxxxxxxxxxxxx"
      encodingAesKey: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    options:
      pollIntervalMs: 3000        # 长轮询间隔
      enableMention: true         # 需要 @机器人 才响应
```

### 自建应用配置

自建应用的配置复杂一些，因为涉及回调地址验证：

```yaml
channels:
  wecom-app:
    adapter: "@openclaw/adapter-wecom"
    mode: self-built
    credentials:
      corpId: "wxxxxxxxxxxxxxxxxx"
      agentId: 1000002
      secret: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      token: "xxxxxxxxxxxxxxxxxxxxxxxxx"
      encodingAesKey: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    options:
      callbackUrl: "https://your-domain.com/wecom/callback"
      enableWechatInterop: true   # 开启互通
    interop:
      licenseType: "base"         # base/互通基础版, premium/互通高级版
      maxUsers: 100
```

### 互通许可费用参考（2026 年）

| 许可类型 | 价格 | 能力 |
|----------|------|------|
| 基础互通 | ~2000 元/年/100 人 | 单聊、群聊、微信用户发消息 |
| 高级互通 | ~5000 元/年/100 人 | 基础 + 朋友圈、客户标签、客户群 |

注意：这里的"100 人"是指你的企业微信中使用互通能力的员工数，不是外部客户数。

### 会话管理注意事项

企业微信的 `userId` 是渠道内唯一的，但跨渠道不通用。也就是说，同一个人在钉钉和企业微信上的 ID 是不同的。OpenClaw 通过 `channelId:userId` 的复合键来管理会话，你不需要手动处理这个问题，但在做数据分析时需要注意。

另外，企业微信的外部联系人（微信用户）的 ID 格式是 `wmxxxxxxxx`（以 `wm` 开头），内部员工的 ID 格式是企业自定义的。

## WorkBuddy 桌面端配置

WorkBuddy 是腾讯出的一个桌面端 IM 桥接工具，定位是给开发者在本地调试用的。它的好处是：不需要公网 IP，不需要企业认证，支持真正的流式输出（因为通信走的是本地 WebSocket）。

### 注册与审批

1. 访问 **qclaw.tencent.com**，用微信扫码注册
2. 填写开发者信息（个人开发者即可）
3. 等待审批：**通常 1-3 个工作日**
4. 审批通过后下载桌面客户端（支持 macOS 和 Windows）

### 配置

```yaml
channels:
  workbuddy:
    adapter: "@openclaw/adapter-workbuddy"
    mode: desktop
    credentials:
      appId: "wbxxxxxxxxxx"
      appSecret: "xxxxxxxxxxxxxxxxxxxxxxxxxx"
    options:
      localPort: 8765             # 桌面客户端监听端口
      endpoint: "ws://localhost:8765/openclaw"
      streaming: true             # 支持真流式
      reconnectOnClose: true
```

### 注意事项

**WorkBuddy 的桌面客户端必须保持运行。** 这是它最大的局限——它本质上是一个本地代理，如果桌面应用关了，OpenClaw 就收不到消息。所以它只适合开发调试和演示，不适合生产环境。

另外，WorkBuddy 目前还在内测阶段，功能和稳定性都不如钉钉和企微的适配器成熟。如果你只是需要一个本地调试工具，考虑直接用 OpenClaw 的 CLI 模式（`openclaw dev --interactive`）可能更省事。

![渠道切换与会话迁移示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_2.png)

## 渠道迁移：不丢历史记录

实际项目中，渠道切换是常见需求。比如你先用钉钉做内部验证，验证通过后要切到企微面向客户。问题是：怎么保留对话历史？

### Redis 会话持久化

OpenClaw 的会话数据默认存在内存里，重启就丢。要做渠道迁移，首先要开启 Redis 持久化：

```yaml
# openclaw.store.yaml
store:
  driver: redis
  redis:
    url: "redis://localhost:6379"
    prefix: "openclaw:sessions:"
    ttl: 2592000                  # 30 天过期
```

### 用户 ID 映射

渠道切换时，用户的 ID 会变（钉钉 ID ≠ 企微 ID）。你需要建立一个映射表：

```yaml
# openclaw.migration.yaml
migration:
  userMapping:
    source: dingtalk
    target: wecom-app
    mappingFile: "./user-mapping.csv"   # 格式：dingtalk_id,wecom_id
    strategy: merge                      # merge=合并历史, fresh=只保留最近 N 条
    mergeOptions:
      maxHistory: 50                     # 合并时最多保留 50 条历史
```

### Shadow 模式

如果你不确定新渠道是否稳定，可以用 Shadow 模式——新渠道接收消息，但回复同时发到新旧两个渠道。用户在旧渠道看到回复确认后，再正式切换。

```yaml
channels:
  wecom-app:
    # ... 正常配置 ...
    shadow:
      enabled: true
      mirrorTo: "dingtalk"          # 回复同时发到钉钉
      duration: "7d"                 # 7 天后自动关闭 shadow
```

这个功能在大规模迁移时特别有用，能确保切换过程中不丢消息。

## 健康监控

生产环境跑 IM 渠道，最怕的就是"静默故障"——机器人看起来在线，但实际上已经断连了。下面是一套完整的健康监控方案。

### 心跳端点

OpenClaw 内置了一个 `/health` 端点，暴露每个渠道的连接状态：

```bash
curl http://localhost:3000/health

# 返回示例
{
  "status": "degraded",
  "channels": {
    "dingtalk": {
      "connected": true,
      "lastHeartbeat": "2026-04-11T08:59:45Z",
      "uptime": "3d 12h 45m"
    },
    "wecom-bot": {
      "connected": false,
      "lastHeartbeat": "2026-04-11T08:45:00Z",
      "error": "poll timeout after 30000ms",
      "reconnectAttempts": 3
    }
  }
}
```

### 端到端测试

光看心跳不够，你需要定期发一条测试消息，确认完整链路是通的：

```yaml
# openclaw.monitoring.yaml
monitoring:
  e2e:
    enabled: true
    intervalMs: 300000            # 每 5 分钟一次
    testMessage: "ping"
    expectedResponse: "pong"
    timeout: 10000
    alertOn:
      - consecutiveFailures: 3
        action: webhook
        webhookUrl: "https://your-alert-endpoint.com/openclaw"
```

### 断连错误码与重连策略

WebSocket 断连时会返回关闭码，不同的码意味着不同的处理策略：

| 关闭码 | 含义 | 处理策略 |
|--------|------|----------|
| 1000 | 正常关闭 | 不重连（主动断开） |
| 1001 | 服务端关闭 | 立即重连 |
| 1006 | 异常关闭（无 close frame） | 指数退避重连（1s, 2s, 4s, 8s, max 60s） |
| 1008 | 策略违规 | 检查凭证是否过期，修复后重连 |
| 1011 | 服务端内部错误 | 延迟 5s 后重连 |
| 4001 | 认证失败（钉钉自定义） | 刷新 token 后重连 |
| 4002 | 频率超限（钉钉自定义） | 延迟 60s 后重连 |

OpenClaw 的默认重连策略：

```yaml
channels:
  dingtalk:
    # ... 其他配置 ...
    reconnect:
      strategy: exponential-backoff
      initialDelayMs: 1000
      maxDelayMs: 60000
      maxAttempts: -1              # -1 = 无限重试
      jitter: true                 # 加随机抖动，避免惊群
```

### 告警集成

建议把健康检查接入你现有的监控体系。OpenClaw 支持以下告警渠道（是的，套娃了）：

- Webhook（通用，对接任何系统）
- 钉钉群机器人（用另一个群通知运维人员）
- 企微应用消息
- 邮件（SMTP）

## 实战建议

最后说几条经验：

1. **先跑通钉钉 Stream，再考虑其他渠道。** 钉钉 Stream 是目前国内 IM 渠道里接入成本最低的方案——不需要公网 IP，不需要 ICP 备案，不需要企业认证（只要你有钉钉组织就行）。先用它验证你的 bot 逻辑，再考虑扩展到其他渠道。

2. **企微自建应用的审批流程比你想象的长。** 从申请到上线，算上企业认证、ICP 备案验证、互通许可购买，整个流程可能需要 2-4 周。提前启动。

3. **不要试图接入个人微信。** 微信对个人号自动化的打击力度极大，任何基于网页版协议、iPad 协议的方案都有封号风险。企业微信的微信客服是目前唯一合规触达微信用户的方式。

4. **消息格式要做降级处理。** 不同渠道支持的消息格式不同。你的 bot 回复如果包含 Markdown 表格，在钉钉上显示正常，在企微智能机器人上可能就变成乱码。OpenClaw 的格式适配器会自动降级，但复杂格式（如 Mermaid 图表）建议渲染成图片再发送。

5. **测试要覆盖断网场景。** 国内网络环境复杂（办公室 WiFi 不稳定、4G/5G 切换、VPN 断连），你的机器人必须能优雅地处理断连和重连。用 `tc netem` 模拟网络抖动来测试。

---

下一篇我们将进入 **OpenClaw 快速上手（十）：生产部署与可观测性**，覆盖 Docker Compose 编排、日志聚合、分布式追踪、以及灰度发布策略。
