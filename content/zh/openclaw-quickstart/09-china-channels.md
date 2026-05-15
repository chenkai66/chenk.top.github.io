---
title: "OpenClaw 指南（九）：国内 IM 选型与权衡"
date: 2026-04-16 09:00:00
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
series_total: 10
description: "钉钉、飞书（已弃用）、企业微信三种模式、微信客服、公众号，以及 WorkBuddy 桌面端桥接。一张选型矩阵和两条在 2026 年真正能跑通的路径的完整配置。"
disableNunjucks: true
translationKey: "openclaw-quickstart-9"
---
第五章快速对比了 Telegram、DingTalk 和 WeChat，而本章则聚焦国内企业场景——所有渠道落地前均需通过公司 IT 部门审批。渠道众多，官方文档分散在数十个 README 中，网上那些所谓的“对比表”也大多早已过时。

下方矩阵表是我每次向他人推荐方案前必查的决策清单。

![OpenClaw 快速入门（9）：中国 IM 选择器，权衡利弊 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_1.png)

---

## 值得了解的七个渠道

![中国 IM 频道选择决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/fig_picker.png)

下文默认你都用了 `openclaw-china` 插件：

```bash
openclaw plugins install https://github.com/BytePioneer-AI/openclaw-china.git
```

或者用向导：

```bash
npx @openclaw-china/setup
```

然后就是选择：

| 渠道 | 受众 | 需公网 IP | 触达微信用户 | 群聊 | 流式 | 延迟 | 消息格式 | 成本 |
|---|---|:---:|:---:|:---:|:---:|---|---|---|
| DingTalk | 内部团队 | no | no | yes | yes | 200-500ms | text, markdown, cards | free |
| Feishu | 内部团队 | yes | no | yes | partial | 300-600ms | text, markdown, cards | deprecated |
| WeCom smart bot | 内部团队 | no | no | yes | yes | 300-700ms | text, markdown, files | free |
| WeCom self-built app | 外部微信 | yes | yes | no | no | 800-1500ms | text, images | ~2000 RMB/yr |
| WeChat customer service | 外部微信 | yes | yes | no | no | 1000-2000ms | text, images | per-seat |
| WeChat public account | 粉丝 | yes | yes | no | no | 1500-3000ms | text, news cards | free |
| WorkBuddy (QClaw) | 个人微信/QQ | no | yes | yes | yes | 500-1000ms | text, images | free |

两点注意：飞书自 2026 年 3 月起停止支持新部署的 openclaw-china 实例（状态标记为 deprecated），推荐选用 DingTalk 或 WeCom；微信订阅号仅支持用户发起消息后 5 秒内的被动回复，不支持主动推送，但服务号与测试号则无此限制。

延迟数据取自业务高峰期北京和上海节点的往返消息耗时，主要瓶颈在于 webhook 投递链路——公众号与企微自建应用需经腾讯 dispatcher 中转，相比长轮询渠道额外增加 500–1000ms。

消息格式支持比看起来更重要。Markdown 支持意味着 Agent 可发送代码块与格式化列表；DingTalk 和 WeCom 智能机器人对 GitHub-flavored markdown 兼容良好，微信渠道则仅支持纯文本或其私有 card schema。

## 三道题选型法

与其逐行对照表格，不如先问自己三个核心问题：

1. **目标用户是谁？** 同事 → DingTalk 或 WeCom 长轮询；外部微信用户 → 企微自建应用或微信客服；个人使用者 → WorkBuddy。
2. **有公网 IP 吗？** 没有 → 只能选 DingTalk、WeCom 长轮询和 WorkBuddy；有 → 所有选项都开放。
3. **需要群聊吗？** 要 → DingTalk、WeCom 长轮询或 WorkBuddy；客服和公众号渠道只能 1:1。

实践中，我们通常只用以下两类技术栈：

- **内部团队，无公网 IP：** 主通道用 DingTalk，备用通道用 WeCom 长轮询，并对两条通道同时进行心跳监控。
- **触达外部微信：** 以企微自建应用作为 Agent 的主界面，用微信客服接收未打标用户的 inbound 消息。

## 钉钉 —— 永远能走通的路

![OpenClaw 快速入门（9）：中国 IM 选择器，权衡利弊 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/09-china-channels/illustration_2.png)

流式模式（long-poll）是钉钉能快速上线的关键——无需公网 IP 和域名验证，只需注册一个机器人并获取两个凭证即可。

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

这些值在哪拿？路径是：钉钉开放平台 → 应用开发 → 创建企业内部应用 → 凭证与基础信息。记得加上 `BotMessage` 权限 scope，并在组织内发布该应用（否则机器人不会响应）。

最常见的故障是：消息传了 30 分钟后突然中断。原因通常是公司网络出口 IP 轮换，导致长连接被中途切断。解决方法是将 `dingtalk.reconnectMs` 的重连间隔调至 60 秒，并确保 Gateway 走的是稳定的出口线路。

## 钉钉深潜 —— 那些坑

**消息大小限制。** 单条文本消息上限为 2048 字符，markdown card 上限为 4096。如果 Agent 生成的回复更长，可开启 `dingtalk.autoChunk: true`，插件会自动拆分消息，但你会看到多个消息气泡。

**频率限制为每秒 20 条消息。** 这个限制是按机器人计算的，而非按用户。在活跃的群聊中很容易触发上限。插件虽会自动排队重试，但会导致延迟飙升。建议将重度用户分流到不同的机器人实例。

**组织管理员审批流。** 你创建了应用、添加了权限 scope、在开发者控制台测试一切正常，但一发布机器人就哑火。这是因为企业应用若要向开发者所在部门以外的用户发消息，必须经过组织管理员审批。请检查：钉钉管理后台 → 工作台 → 应用管理 → 待审批。

**时钟漂移会杀死连接。** 如果服务器时钟与 NTP 时间偏差超过 5 分钟，钉钉的长连接会静默断开——没有错误提示，就是彻底没动静。解决方案是运行 `ntpdate` 或配置可靠的 NTP 源。

## 企业微信 —— 三种形态，看人下菜

“企业微信”这个名字其实涵盖了三种除了品牌外毫无关联的 API：

- **智能机器人（智能机器人，长连接）：** 仅限内部使用，无需公网 IP，安装最简单，且支持流式通信。
- **自建应用（自建应用）：** 可触达外部微信用户，但前提是组织已购买企微-微信互通 license；需要公网 IP 和已备案的验证域名，且不支持流式。
- **微信客服（微信客服）：** 面向未开通互通 license 的外部微信用户，消息可从公众号菜单、小程序、视频号或直播入口进入，但不支持 Markdown。

90% 的“我们需要一个团队机器人”场景中，智能机器人就是你的最佳选择。

## 企微深潜

**智能机器人注册。** 路径：企业微信管理后台 → 应用管理 → 创建应用 → 机器人。注册后立即获得 webhook URL。若使用长轮询模式，需启用回调服务器并获取 `token` 和 `encodingAESKey`。无需域名验证，也无需公网 IP。但请注意：智能机器人无法主动发起对话，只能被动回复。

**自建应用注册。** 需满足三个条件：已备案的验证域名、用于回调的公网 IP，以及互通 license（基础版约 2000 RMB/年）。获得 license 后，外部微信用户可将你的机器人添加为联系人，但无法与外部用户建立群聊，不支持流式通信，且消息格式受限。

**互通 license 成本。** 基础版（100 个外部联系人）约 2000 RMB/年；中级版（1000 联系人）约 10000 RMB/年；企业版起价约 50000 RMB/年，可议价。如需测试，可申请 3 个月试用（最多支持 50 个联系人）。

**会话管理。** 企业微信服务端不保存对话历史。`openclaw-china` 插件通过以 `(userId, channelId)` 为键的 session store 来管理会话，默认 30 分钟后过期。重要提示：用户 ID 是渠道隔离的——同一个人在智能机器人和自建应用中的 ID 完全不同。

## WorkBuddy —— 桌面桥接

WorkBuddy 是腾讯官方的 QClaw 桥接工具，运行在你的桌面端，无需公网 IP，也无需应用审核，可直接连接你真实的个人微信和 QQ。它是目前唯一能实现“今晚就在微信里用上 Agent”且无需妥协的方案。

但它的局限也很明显：作为桌面应用，一旦宿主机休眠或关机，连接就会中断。这在个人助理场景中尚可接受，但在生产环境中显然不可靠。

**Setup 步骤：**
1. 使用中国大陆手机号在 `qclaw.tencent.com` 注册，用例选择“个人助理”。
2. 等待 1–3 个工作日完成审批。
3. 下载桌面客户端（仅支持 Windows/macOS），扫码登录。
4. 配置 OpenClaw：

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

无需 API keys，也无需 webhooks。插件通过 localhost 与 WorkBuddy 通信，而 WorkBuddy 则代表你与微信/QQ 对话。

## 迁移指南 —— 切换渠道

你最初基于 DingTalk 快速上线，现在却需要通过 WeCom 触达外部用户。如何切换渠道又不丢失历史记录？

1. **先切换到持久化会话（如 Redis）** —— 内存会话会随进程终止而消失。
2. **更新配置：** 禁用旧渠道，启用新渠道。
3. **重映射用户 ID：** 由于用户 ID 是渠道隔离的，你需要建立跨渠道的映射关系（可在新渠道首条消息时通过手机号或邮箱验证身份）。
4. **影子模式测试：** 让新旧渠道并行运行一周，旧渠道设为 `shadowMode: true`（仅接收消息但不回复），确认稳定后再完全切换。

## 监控健康度

**心跳接口** —— 每 60 秒轮询一次：

```bash
curl http://localhost:3000/api/channels/dingtalk/health
# {"status": "connected", "lastMessageAt": "...", "reconnectCount": 2}
```

若 `status != connected`，或在业务高峰期 `lastMessageAt` 长时间未更新，应立即告警。如果 `reconnectCount` 每分钟都在增长，说明网络路径不稳定。

**端到端测试** —— 每小时发送一条测试消息并验证回复，可捕获心跳监控遗漏的问题（例如 Agent 卡在重试循环中）。

**重连模式** —— 健康状态：平均每几小时重连一次；异常状态：频繁重连（如每 30 秒一次）。此时应检查 gateway 日志中的 disconnect codes（1006 表示网络问题，1008 表示策略违规或 IP 被封）。

## 矩阵背后的教训

渠道才是 Agent 真正“活着”的地方。再强大的 Agent，若跑在不稳定的渠道上，最终也只是个不稳定的 Agent。因此，在折腾任何花哨功能之前，请先选一个匹配你受众和网络环境的渠道，并不惜代价把它打磨得坚如磐石。之后，再加一个备用渠道做冗余。

其余的一切，都不过是在甲板上重新摆放椅子罢了。
