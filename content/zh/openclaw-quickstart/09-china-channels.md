---
title: "OpenClaw 指南（九）：国内 IM 选型与权衡"
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
第五章咱们快速扫了一眼 Telegram、DingTalk 和 WeChat。这一章是专门给国内需要把东西推过公司 IT 部门的朋友准备的续篇。渠道太多，文档散落在十几个 README 里，网上那些“对比表”大多也过时了。

下面这个矩阵，是我给别人推荐方案前的必查项。

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_1.png)

## 值得了解的七个渠道

![China IM channel selection decision tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/fig_picker.png)

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

两点注意：Feishu 从 2026 年 3 月起 officially listed as deprecated for new openclaw-china installs —— 选 DingTalk 或 WeCom。微信订阅号有 5 秒被动回复窗口，无法主动推送；服务号和测试号没有这个限制。

延迟数据是业务高峰期从北京/上海测得的往返消息耗时。最大的延迟杀手是 webhook 交付路径 —— 公众号和企微自建应用要走腾讯的 dispatcher，比长轮询渠道多出 500-1000ms。

消息格式支持比看起来更重要。Markdown 意味着你的 Agent 能发代码块和格式化列表。DingTalk 和 WeCom 智能机器人对 GitHub-flavored markdown 支持很好。微信渠道要求纯文本或私有 card schema。

## 三道题选型法

别从上往下读矩阵。问自己三个问题：

1. **对面是谁？** 同事 → DingTalk 或 WeCom 长轮询。外部微信用户 → 企微自建应用或微信客服。你自己 → WorkBuddy。
2. **有公网 IP 吗？** 没有 → 只能选 DingTalk、WeCom 长轮询和 WorkBuddy。有 → 随便选。
3. **需要群聊吗？** 要 → DingTalk、WeCom 长轮询或 WorkBuddy。客服和公众号渠道只能 1:1。

实操中我通常选这两种栈：

- **内部团队，无公网 IP：** 主通道 DingTalk，备用 WeCom 长轮询。心跳监控两边。
- **触达外部微信：** 企微自建应用作为 Agent 主界面，微信客服接收未打标用户的 inbound。

## 钉钉 —— 永远能走通的路

![OpenClaw QuickStart (9): The China IM Picker, with Honest Tradeoffs — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_2.png)

流式模式（long-poll）是钉钉上线快的唯一原因。不需要公网 IP，不需要域名验证，注册个机器人拿两个值就行：

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

去哪拿这些值：钉钉开放平台 → 应用开发 → 创建企业内部应用 → 凭证与基础信息。加上 `BotMessage` 权限 scope。要在组织内发布应用（没发布的话机器人不会响应）。

最常见的故障：消息传了 30 分钟停了。原因：公司网络出口 IP 轮换，把长连接掐断了。解决：把 `dingtalk.reconnectMs` 的重连间隔调到 60s，让 Gateway 走稳定的出口。

## 钉钉深潜 —— 那些坑

**消息大小限制。** 单条消息文本上限 2048 字符，markdown card 上限 4096。如果 Agent 生成的回复更长，设 `dingtalk.autoChunk: true` —— 它会自动拆分，但你会看到多个消息气泡。

**频率限制每秒 20 条消息。** 按机器人算，不是按用户。群聊活跃的话很容易触顶。插件会自动排队重试，但延迟会飙升。建议把重度用户拆分到不同的机器人实例。

**组织管理员审批流。** 你创建了应用，加了 scope，在开发者控制台测试都正常。一发布机器人就哑火。原因：企业应用发消息给开发者部门以外的用户需要组织管理员审批。查 钉钉管理后台 → 工作台 → 应用管理 → 待审批。

**时钟漂移会杀死连接。** 服务器时钟跟 NTP 漂移超过 5 分钟，钉钉的长连接会静默死亡。没有报错，就是没动静。跑 `ntpdate` 或者指向可靠的 NTP 源。

## 企业微信 —— 三种形态，看人下菜

“企业微信”这个名字覆盖了三种除了品牌外毫无关系的 API：

- **智能机器人（智能机器人，长连接）。** 仅限内部使用。不需要公网 IP。安装最简单。支持流式。
- **自建应用（自建应用）。** 能触达外部微信用户 *如果* 组织有企微 - 微信互通 license。需要公网 IP 和已验证域名。不支持流式。
- **微信客服（微信客服）。** 没有互通 license 的外部微信用户。从公众号菜单、小程序、视频号、直播入口路由。不支持 Markdown。

90% 的“我们团队需要个机器人”的情况，智能机器人就是你要的。

## 企微深潜

**智能机器人注册。** 企业微信管理后台 → 应用管理 → 创建应用 → 机器人。马上拿到 webhook URL。长轮询模式的话，启用回调服务器，拿 `token` 和 `encodingAESKey`。不需要域名验证，不需要公网 IP。注意：智能机器人不能发起对话 —— 只能回复。

**自建应用注册。** 需要：已验证域名（要 ICP 备案）、回调用的公网 IP、互通 license（~2000 RMB/年起）。拿到 license 后，外部微信用户能把你的机器人加为联系人。不能跟外部用户群聊，不支持流式，消息格式也受限。

**互通 license 成本。** 基础版（100 外部联系人）：~2000 RMB/年。中级版（1000 联系人）：~10000 RMB/年。企业版：50000 RMB/年起谈。如果测试，申请 3 个月试用（最多 50 联系人）。

**会话管理。** 企微不在服务端维护对话历史。`openclaw-china` 插件用 `(userId, channelId)` 键值的 session store 处理这个，30 分钟后过期。注意：用户 ID 是渠道隔离的 —— 同一个人在智能机器人和自建应用里 ID 不同。

## WorkBuddy —— 桌面桥接

WorkBuddy。这是腾讯官方的 QClaw 桥接，跑在你的桌面上，不要 IP，不要应用审核，直连你真实的个人微信和 QQ。这是唯一能让我“今晚就想在微信里用上 Agent"且不后悔的方案。

麻烦在于：这是个桌面应用，所以机器休眠它就挂了。个人助理没问题。生产环境不行。

**Setup 步骤：**
1. 用大陆手机号在 `qclaw.tencent.com` 注册。用例选“个人助理”。
2. 等 1-3 个工作日审批。
3. 下载桌面客户端（仅 Windows/macOS），扫码登录。
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

不要 API keys，不要 webhooks。插件在 localhost 跟 WorkBuddy 对话，WorkBuddy 替你跟微信/QQ 对话。

## 迁移指南 —— 切换渠道

你因为快选了钉钉。现在需要企微触达外部用户。怎么切换且不丢历史：

1. **切到持久化会话**（Redis）再迁移 —— 内存会话随进程死亡。
2. **更新配置**：禁用旧渠道，启用新渠道。
3. **重映射用户 ID**：用户 ID 是渠道隔离的，所以需要交叉引用映射（在新渠道第一条消息时通过手机/邮箱验证）。
4. **影子模式测试**：双渠道并行跑一周，旧渠道设 `shadowMode: true`（收消息但不回复），然后再切换。

## 监控健康度

**心跳接口** —— 每 60s 轮询：
```bash
curl http://localhost:3000/api/channels/dingtalk/health
# {"status": "connected", "lastMessageAt": "...", "reconnectCount": 2}
```

如果 `status != connected` 或者业务高峰期 `lastMessageAt`  stale 就报警。如果 `reconnectCount` 每分钟都在涨，网络路径不稳定。

**端到端测试** —— 每小时发条测试消息并验证回复。能抓到心跳漏掉的问题（比如 Agent 卡在重试循环里）。

**重连模式** —— 健康：几小时重连一次。不健康：每 30 秒一次。查 gateway 日志里的 disconnect codes（1006 = 网络，1008 = 策略违规/IP 被封）。

## 矩阵背后的教训

渠道是 Agent 真正 *活着* 的地方。渠道不稳定，Agent 再好也不稳定。所以在搞任何花哨功能之前，先选个匹配受众和网络的渠道，然后过度投入把 *那个* 渠道搞稳。再加第二个做冗余。

其他都是瞎折腾。