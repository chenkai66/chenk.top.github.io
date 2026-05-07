---
title: "OpenClaw 快速上手（九）：国内 IM 怎么挑，把权衡说透"
date: 2026-04-11 09:00:00
tags:
  - openclaw
  - dingtalk
  - wechat
  - wecom
  - channels
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 9
description: "钉钉、飞书（已弃坑）、企微三种、微信客服、公众号，加上 WorkBuddy 桌面桥。给一份对照矩阵，外加 2026 年还能跑通的两条路的实际配置。"
disableNunjucks: true
translationKey: "openclaw-quickstart-9"
---
第 5 章简单介绍了 Telegram、钉钉和微信。这一章专门写给国内的读者，特别是那些需要绕过公司 IT 部门限制的人。渠道太多，文档分散在十几个 README 文件里，网上能找到的“对比表”大多已经过时。

接下来是我每次给别人推荐工具前都会参考的一张矩阵表。
![OpenClaw 快速上手（九）：国内 IM 怎么挑，把权衡说透 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_1.jpg)

## 七个值得关注的渠道

以下内容默认你已经安装了 `openclaw-china` 插件：

```bash
openclaw plugins install https://github.com/BytePioneer-AI/openclaw-china.git
```

或者用安装向导：

```bash
npx @openclaw-china/setup
```

接下来是具体选项：

| 渠道 | 适用对象 | 是否需要公网 IP | 能否触达微信用户 | 支持群聊 | 支持流式 |
|---|---|:---:|:---:|:---:|:---:|
| 钉钉 | 内部团队 | 否 | 否 | 是 | 是 |
| 飞书 | 内部团队 | 是 | 否 | 是 | 部分支持 |
| 企微智能机器人（长轮询） | 内部团队 | 否 | 否 | 是 | 是 |
| 企微自建应用 | 外部微信用户 | 是 | 是 | 否 | 否 |
| 微信客服 | 外部微信用户 | 是 | 是 | 否 | 否 |
| 微信公众号 | 粉丝 | 是 | 是 | 否 | 否 |
| WorkBuddy（腾讯 QClaw） | 个人微信/QQ | 否 | 是 | 是 | 是 |

两点需要注意：飞书从 2026 年 3 月起在 openclaw-china 中被标记为弃用，建议直接选择钉钉或企微。另外，微信公众号订阅号有 5 秒被动回复窗口，无法主动推送；服务号和测试号则没有这个限制。
## 三个问题帮你选对方案

别从头到尾看矩阵，直接问自己三个问题：

1. **对方是谁？**  
   同事 → 钉钉或企微长连接  
   外部微信用户 → 企微自建应用或微信客服  
   自己 → WorkBuddy  

2. **有没有公网 IP？**  
   没有 → 只能用钉钉、企微长连接或 WorkBuddy  
   有 → 所有选项都可以考虑  

3. **需不需要群聊？**  
   需要 → 钉钉、企微长连接或 WorkBuddy  
   客服和公众号只支持一对一聊天  

实际操作中，我一般会选以下两种方案之一：

- **内部团队，没有公网 IP：**  
  主渠道用钉钉，备份用企微长连接，Heartbeat 同时监控两边。  

- **需要触达外部微信用户：**  
  主力用企微自建应用，微信客服负责接收未打标签的零散消息。
## 钉钉——总能跑通的那条路

![OpenClaw 快速上手（九）：国内 IM 怎么挑，把权衡说透 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/09-china-channels/illustration_2.jpg)

Stream 模式（长轮询）是钉钉快速上线的关键。不需要公网 IP，也不用域名验证，注册一个机器人，拿到两个值就够了：

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

这两个值从哪儿拿？打开钉钉开放平台 → 应用开发 → 创建企业内部应用 → 凭证与基础信息。记得加上 `BotMessage` 权限范围。最后，在组织内发布这个应用——不发布的话，机器人不会响应。

最常见的问题：用了 30 分钟后消息突然断了。原因很简单：组织出口 IP 在轮换，导致长轮询连接被中断。解决方法：把 `dingtalk.reconnectMs` 的重连间隔调到 60 秒，同时把 Gateway 放在稳定的出口后面。
## 企微——三种类型，按需求选择

"企微"这个名字涵盖了三套完全不同的 API，除了品牌名之外几乎没有共通点：

- **智能机器人（长连接）。** 只能用于企业内部，不需要公网 IP，安装最简单，支持流式传输。
- **自建应用。** 如果企业开通了 WeCom-WeChat 互通许可，可以用来触达外部微信用户。需要公网 IP 和已备案域名，不支持流式传输。
- **微信客服。** 没有互通许可时，用来触达外部微信用户。可以通过公众号菜单、小程序、视频号、直播间入口接入。不支持 Markdown。

90% 的“团队群里需要一个 bot”场景，选智能机器人就够了。如果目标是“从 Agent 系统里通过微信跟客户沟通”，那就需要用到自建应用——这时候我也建议你去读一下[微信专题](../04-channels/06-wechat-workbuddy.md)。
## 给那些只想简单玩玩的人

WorkBuddy。这是腾讯官方推出的 QClaw 桥，直接运行在桌面端，不需要 IP，也不用审核，能连接你的真实个人微信和 QQ。如果你今晚就想让智能助手出现在微信里，这个工具是唯一不会让你后悔的选择。

不过有个前提：它是个桌面应用，必须依赖运行它的那台机器保持开机状态。用来当个人助理完全没问题，但要投入生产环境就不合适了。
## 矩阵背后的教训

渠道是 Agent 的生存之地。如果渠道不稳定，再优秀的 Agent 也会变得不稳定。所以，在搞复杂功能之前，先选一个适合目标用户和网络条件的渠道，全力以赴把它做到坚如磐石。然后再添加第二个渠道作为备份。

其他都是无谓的折腾。
