---
title: "OpenClaw 快速上手（八）：Heartbeat、Cron，以及怎么让它早上 7 点叫你"
date: 2026-04-10 09:00:00
tags:
  - openclaw
  - cron
  - heartbeat
  - automation
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 8
description: "OpenClaw 的两个调度原语——Heartbeat 是带脉搏的巡逻员，Cron 是厨房计时器。给一份能跑的早报、周报、竞品监控配置，外加那个会半夜把你叫醒的反模式。"
disableNunjucks: true
translationKey: "openclaw-quickstart-8"
---
第一次部署 OpenClaw 时，我坐在那儿不停地给它发消息。折腾了两天，我才突然明白：自己搞出来的是个聊天机器人，根本不是什么 Agent。直到它**主动给我发消息**的那一刻，才算是真正迈进了 Agent 的门槛。

这篇文章就来聊聊实现这个目标的两种方法。
![OpenClaw 快速上手（八）：Heartbeat、Cron，以及怎么让它早上 7 点叫你 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.jpg)

## Heartbeat 和 Cron——先在脑子里分清楚

这两个都能用来调度任务，但完全是两回事。

| | Heartbeat | Cron |
|---|---|---|
| 触发方式 | 固定间隔（比如每 30 分钟一次） | 固定时间点（比如每天 8:30） |
| 默认逻辑 | "转一圈，有事才叫我" | "到点就执行，到点就推送" |
| 适用场景 | 捕捉异常、查漏补缺 | 日常例行、定时报告 |
| 会话模式 | 使用最近活跃的通道 | 默认隔离运行 |

> **Heartbeat 是巡逻员，Cron 是闹钟。**

我自己在生产环境里两个都用：Cron 负责早上 7 点的晨会简报和下午 5 点的收工总结；Heartbeat 每 45 分钟跑一次，检查有没有忘记处理的 PR。
## 先启用

```json
{
  "tools": {
    "cron":    { "enabled": true },
    "message": { "enabled": true }
  }
}
```

`message` 很关键。不开的话，Agent 根本没法主动发送任何消息。
## Heartbeat——巡逻机制

```json5
{
  agents: {
    defaults: {
      heartbeat: {
        every: "45m",
        target: "last",
        prompt: "Read HEARTBEAT.md if it exists. Follow it strictly. If nothing needs attention, reply HEARTBEAT_OK.",
        activeHours: {
          start: "09:00",
          end:   "22:00",
          timezone: "Asia/Shanghai"
        }
      }
    }
  }
}
```

关键是 `HEARTBEAT_OK` 这个约定。我希望巡逻是安静的，默认提示明确告诉 Agent：没事就别出声。少了这一句，每隔 45 分钟就会收到一条“一切正常”的消息，烦得让人抓狂。

真正的规则写在 `HEARTBEAT.md` 里：

```markdown
# Heartbeat 规则
```
## 每次巡逻
- 执行 `gh pr list --search "review-requested:@me"`，如果有结果就总结一下。
- 用 `mail-count` skill 查看收件箱，超过 20 封邮件就发提醒。
- 如果都没问题，回复 HEARTBEAT_OK。
```
## Cron——定时任务

有两种方式。CLI 最靠谱，Gateway 重启也不会丢：

```bash
openclaw cron add \
  --name "morning-brief" \
  --cron "30 7 * * *" \
  --tz "Asia/Shanghai" \
  --session main \
  --system-event "生成今日简报：天气、日程、PR/issue 前三条" \
  --wake now
```

如果只需要一次性延时（比如"20 分钟后提醒我"）：

```bash
openclaw cron add \
  --name "standup-prep" \
  --at "20m" \
  --session main \
  --system-event "20 分钟后开站会，总结昨天的工作"
```

查看和删除任务也很简单：

```bash
openclaw cron list
openclaw cron delete --name morning-brief
```

Cron 任务会保存到磁盘。即使重启 Gateway，任务也会自动恢复。
## 我常用的三种模式

**早报 —— Cron，6:47。** 天气、当天日程、GitHub 上最活跃的三条动态。消息会发到我昨天看得最多的那个频道。时间定在 6:47 而不是 7:00 是有讲究的——大多数人拿起手机时，消息已经在那里等着了。

**仓库监控 —— Cron，每小时一次。** 新 PR、新 issue、主分支 CI 构建失败、依赖库的安全告警。只有发现问题时才会发送；如果一切正常，机器人会自动跳过。

**竞品追踪 —— Cron，每天 21:00。** 对比一份我关注的小页面列表：新博客文章、新定价方案、新功能页面 → 触发提醒。这个工具上线不到两周就值回票价了。
## 凌晨三点的反模式

![OpenClaw 快速上手（八）：Heartbeat、Cron，以及怎么让它早上 7 点叫你 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.jpg)

这种错误绝对能把你吵醒：给 Heartbeat 设置了 `target: "all"`，但忘了加 `activeHours`，而且 HEARTBEAT.md 还没在正常情况下返回 `HEARTBEAT_OK`。

结果就是：每隔 45 分钟，你加入的所有频道都会弹出一条“已巡查，无异常”。凌晨三点也不例外。更惨的是，连那些根本没人要求你发消息的群聊也逃不掉。

解决办法有两个：一是老老实实加上 `activeHours`，二是把规则文件里的默认值改成显眼的 `HEARTBEAT_OK`。
## 收尾

OpenClaw 的核心思想就是让 Agent 主动来找你。Heartbeat 和 Cron 是实现这个目标的两个关键工具。需要用固定周期执行的任务交给 Cron；需要在出问题时才通知你的任务交给 Heartbeat。千万别弄混了。
