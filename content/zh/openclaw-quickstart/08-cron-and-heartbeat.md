---
title: "OpenClaw 快速上手（八）：心跳巡逻、定时任务，以及早上七点被戳醒"
date: 2026-04-10 09:00:00
tags:
  - openclaw
  - cron
  - heartbeat
  - automation
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 8
description: "OpenClaw 的两个调度原语——Heartbeat 是带脉搏的巡逻，Cron 是厨房计时器。每日简报、周报、竞品监控的完整配置，以及那个会让你凌晨三点被吵醒的反模式。"
disableNunjucks: true
translationKey: "openclaw-quickstart-8"
---

## 从聊天机器人到主动 Agent

部署完 OpenClaw 的头两天，我一直在给它发消息。问它问题、让它跑脚本、叫它总结文档。用了两天后突然意识到——这跟 ChatGPT 有什么区别？本质上还是一个被动应答的聊天机器人。

转折点发生在第三天早上。我还没拿起手机，钉钉就弹了一条：

> 早上好。昨晚 CI 跑挂了 2 个 job，是 lint 报错，看起来是昨天那个 PR 引入的。已经贴了报错截图在下面。要我开个修复 PR 吗？

这一刻它不再是聊天机器人了。它是一个会主动巡逻、主动汇报的 Agent。

让 OpenClaw 从"被动回答"变成"主动行动"，只需要两个调度原语：**Heartbeat** 和 **Cron**。

![Heartbeat 和 Cron 的调度模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_1.jpg)

## Heartbeat vs Cron：巡逻员与厨房计时器

![Heartbeat 与 Cron 调度模型对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/fig_hb_vs_cron.png)

先搞清楚这两个东西的本质区别：

| 维度 | Heartbeat（心跳） | Cron（定时任务） |
|------|-------------------|-----------------|
| 触发方式 | 固定间隔（如每 45 分钟） | 固定时刻（如每天 6:47） |
| 语义 | "有事才报告" | "到点就执行" |
| 输出 | 无异常时沉默（HEARTBEAT_OK） | 每次都产出结果 |
| 适用场景 | 监控、异常捕获 | 日报、周报、定时清理 |
| 类比 | 巡逻的保安——没事就不打扰你 | 厨房计时器——滴滴滴，时间到了 |

一句话总结：**Heartbeat 是巡逻员，Cron 是厨房计时器。**

Heartbeat 的逻辑是："我每隔一段时间检查一下，如果一切正常就闭嘴，只有发现问题才喊你。"

Cron 的逻辑是："不管天塌不塌，到了这个时间点就执行这个任务，然后把结果发给你。"

搞混这两个，你就会在凌晨三点收到"一切正常"的通知。

## 开启调度功能

在 `openclaw.config.json5` 中开启调度：

```json5
{
  // 基础配置省略...
  "scheduling": {
    "enabled": true,
    "heartbeat": { /* 下面详细讲 */ },
    "cron": { /* 下面详细讲 */ }
  },
  "tools": {
    "message": true,    // 允许主动发消息
    "cron": true,       // 允许管理定时任务
    "system": true      // 允许读取系统信息
  }
}
```

关键：`tools.message` 必须开启。不然 Agent 有话说但嘴被封住了。

## Heartbeat 配置

完整的 Heartbeat 配置块：

```json5
{
  "scheduling": {
    "heartbeat": {
      // 每 45 分钟执行一次心跳检查
      "every": "45m",

      // 结果发到哪个对话：last = 最近活跃对话
      "target": "last",

      // 活跃时段——只在这些时间内执行心跳
      "activeHours": {
        "start": "08:00",
        "end": "21:00",
        "timezone": "Asia/Shanghai"
      },

      // 静默约定：返回这个字符串时不发送任何消息
      "silentToken": "HEARTBEAT_OK",

      // 心跳指令文件
      "promptFile": "./HEARTBEAT.md",

      // 单次心跳最大执行时间
      "timeout": "30s"
    }
  }
}
```

核心设计：`HEARTBEAT_OK` 约定。当 Agent 检查完一切发现没有异常时，返回 `HEARTBEAT_OK` 这个字符串，调度器看到后就不发送任何消息。只有返回其他内容时，才会推送到目标对话。

这就是"有事才报告"的实现方式。

## 写好 HEARTBEAT.md

`HEARTBEAT.md` 是心跳的灵魂。它告诉 Agent 每次心跳时应该检查什么、怎么判断是否有异常、什么情况该通知我。

一份生产级的 HEARTBEAT.md：

```markdown
# Heartbeat 巡逻清单

你是我的基础设施巡逻员。每次心跳时按以下顺序检查，发现异常立即报告。

## 代码与 CI（优先级：高）
- 检查最近 1 小时的 CI/CD 状态，有失败的 pipeline 吗？
- 有新的 PR 等待我 review 吗？超过 4 小时未处理的标记为紧急。
- main 分支最近一次部署是否成功？

## 通讯与消息（优先级：中）
- 钉钉是否有 @我 的未读消息？
- 是否有超过 2 小时未回复的紧急工单？
- 团队群里有没有需要我关注的讨论？

## 基础设施（优先级：中）
- 服务器磁盘使用率是否超过 85%？
- 核心服务健康检查是否全部通过？
- 最近 1 小时是否有异常告警？

## 判断规则
- 如果以上全部正常：返回 HEARTBEAT_OK
- 如果有异常但不紧急（如 PR 等待 review）：简短汇报，一句话说清楚
- 如果有紧急问题（CI 挂了、服务宕机）：立即详细报告，附上关键信息

## 周末模式
周六周日只检查"基础设施"部分，其他跳过。
```

写 HEARTBEAT.md 的几个原则：

1. **分类清晰**：让 Agent 知道检查什么、按什么顺序
2. **设定阈值**：不是"有 PR 就报告"，而是"超过 4 小时未处理才报告"
3. **优先级分级**：什么值得打断你，什么可以等
4. **周末降级**：别让 Agent 在周末告诉你有 PR 等你 review

## Cron 命令

Cron 任务通过 CLI 管理：

```bash
# 添加持久化定时任务（重启后依然存在）
openclaw cron add --schedule "47 6 * * *" \
  --name "daily-brief" \
  --prompt "生成今日简报" \
  --target "#my-channel" \
  --timezone "Asia/Shanghai" \
  --durable

# 添加一次性延迟任务（执行完自动删除）
openclaw cron delay --after "2h" \
  --prompt "提醒我回复那封邮件" \
  --target "last"

# 查看所有定时任务
openclaw cron list

# 删除指定任务
openclaw cron delete --name "daily-brief"
```

`--durable` 标记很重要。没有它的任务在 Gateway 重启后就消失了。生产环境永远加 `--durable`。

## 四种实战模式

### 模式一：每日简报（6:47 AM）

为什么是 6:47 而不是 7:00？因为所有人都在整点设闹钟，网关在整点的负载最高。错开几分钟是最佳实践。

```json5
{
  "cron": {
    "jobs": [
      {
        "name": "daily-brief",
        "schedule": "47 6 * * *",
        "timezone": "Asia/Shanghai",
        "durable": true,
        "target": "#daily-channel",
        "event": "system.cron.daily_brief",
        "prompt": "生成今日简报。包含：1) 昨天的 Git 提交摘要；2) 今天的日历事项；3) 待处理 PR 列表；4) 未完成的 TODO 项。格式简洁，用 bullet points。",
        "timeout": "60s"
      }
    ]
  }
}
```

示例输出：

```
早安。今日简报：

- Git: 昨天 3 个 commit 合入 main，涉及用户鉴权模块重构
- 日历: 10:00 周会，14:30 和前端对齐接口
- PR: #247 等待 review（已 6 小时），#251 CI 通过待合入
- TODO: 竞品分析文档还差结论部分

需要我先处理哪一项？
```

### 模式二：仓库监控（每小时）

```json5
{
  "name": "repo-watcher",
  "schedule": "15 * * * *",
  "timezone": "Asia/Shanghai",
  "durable": true,
  "target": "#dev-alerts",
  "prompt": "检查 GitLab 仓库最近 1 小时的变动。关注：1) 新开的 PR；2) CI 失败；3) main 分支直接 push（应该禁止）。如果什么都没发生，不要输出任何内容。",
  "skipEmpty": true,
  "timeout": "45s"
}
```

`skipEmpty: true` —— 如果 Agent 判断没有值得报告的内容，就不发送消息。这是 Cron 版的"静默模式"。

### 模式三：竞品监控（每晚 9 点）

```json5
{
  "name": "competitor-watch",
  "schedule": "0 21 * * 1-5",
  "timezone": "Asia/Shanghai",
  "durable": true,
  "target": "#product-intel",
  "prompt": "访问以下竞品页面，与昨天的快照对比差异。有变动就报告具体改了什么；没变动就说'无更新'。\n\n- https://competitor-a.com/pricing\n- https://competitor-b.com/changelog\n- https://competitor-c.com/features",
  "timeout": "90s",
  "tools": ["web_fetch", "diff", "memory"]
}
```

这个模式的威力在于 `memory` 工具——Agent 会把每天的页面快照存入记忆，第二天再取出来做 diff。不需要外部数据库。

### 模式四：收工总结（每天 5 点）

```json5
{
  "name": "eod-shutdown",
  "schedule": "0 17 * * 1-5",
  "timezone": "Asia/Shanghai",
  "durable": true,
  "target": "last",
  "prompt": "一天结束了。请：1) 总结今天处理的所有任务和对话；2) 列出未完成的事项；3) 将今日摘要写入 memory，key 为 daily_summary_YYYY-MM-DD；4) 如果有明天需要跟进的事，列出来。",
  "tools": ["memory", "list_conversations"],
  "timeout": "60s"
}
```

这是我最喜欢的一个模式。每天下班时 Agent 自动归档当天的工作，写入记忆。第二天早上的 daily-brief 可以引用昨天的总结，形成完整的工作流闭环。

## 组合使用：全天候工作流

最强大的用法是把 Heartbeat 和 Cron 组合起来，形成完整的一天：

```json5
{
  "scheduling": {
    "heartbeat": {
      "every": "45m",
      "target": "last",
      "activeHours": {
        "start": "08:00",
        "end": "21:00",
        "timezone": "Asia/Shanghai"
      },
      "silentToken": "HEARTBEAT_OK",
      "promptFile": "./HEARTBEAT.md",
      "timeout": "30s"
    },
    "cron": {
      "jobs": [
        {
          "name": "morning-brief",
          "schedule": "47 6 * * 1-5",
          "timezone": "Asia/Shanghai",
          "durable": true,
          "target": "#daily-channel",
          "prompt": "生成今日简报，引用 memory 中昨天的 daily_summary。"
        },
        {
          "name": "eod-summary",
          "schedule": "0 17 * * 1-5",
          "timezone": "Asia/Shanghai",
          "durable": true,
          "target": "last",
          "prompt": "归档今天的工作，写入 memory。列出明天待办。"
        }
      ]
    }
  }
}
```

工作流是这样的：

1. **6:47** — Cron 触发早报，Agent 读取昨天的记忆，生成今日待办
2. **8:00-21:00** — Heartbeat 每 45 分钟巡逻一次，CI 挂了立刻通知，PR 超时提醒
3. **17:00** — Cron 触发收工总结，归档当天工作，写入记忆供明天引用

三个组件各司其职，不重叠、不遗漏。

![全天工作流时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/08-cron-and-heartbeat/illustration_2.jpg)

## Cron 调试指南

定时任务不触发是最烦人的 bug，因为你只能等。以下是最常见的四个坑：

### 时区不匹配

```bash
# 检查 Gateway 的系统时区
openclaw debug timezone

# 输出示例：
# System: UTC
# Config: Asia/Shanghai (UTC+8)
# Next fire: 2026-04-10 06:47 CST = 2026-04-09 22:47 UTC
```

如果你的 `schedule` 写的是 `47 6 * * *`，但没设 `timezone`，默认按 UTC 算。你期望早上 6:47 收到简报，实际要等到下午 14:47。永远显式指定 `timezone: "Asia/Shanghai"`。

### 输出没有送达

定时任务执行了，但你没收到消息。检查：

```bash
# 查看任务执行历史
openclaw cron history --name "daily-brief" --last 5

# 常见问题：
# - target 指向了不存在的频道
# - Agent 输出了空字符串（skipEmpty 生效）
# - 消息工具没有开启（tools.message = false）
```

### Gateway 重启导致漏执行

Gateway 重启时，非 durable 的任务会丢失。即使是 durable 任务，如果重启时间恰好跨过了触发时刻，也会 miss fire。

```json5
{
  "cron": {
    "missedFirePolicy": "run_immediately",  // 启动后补执行
    // 或者
    "missedFirePolicy": "skip"              // 跳过，等下次
  }
}
```

生产环境建议 `run_immediately`。启动后先把欠的账还了。

### 任务超时

复杂任务（如竞品监控需要抓取多个页面）可能超过默认 timeout：

```bash
# 查看哪些任务超时了
openclaw cron history --status timeout

# 解决：增大 timeout 或拆分任务
```

## Token 成本估算

调度功能意味着 Agent 会在你不知情的情况下持续消耗 token。提前算清楚：

**Heartbeat 成本：**

- 间隔：每 45 分钟
- 活跃时段：08:00-21:00 = 13 小时
- 每天调用次数：13 * 60 / 45 = **约 17 次**
- 每次心跳约 800 token（读取 HEARTBEAT.md + 检查 + 返回判断）
- 日消耗：17 * 800 = **13,600 token/天**
- 按 qwen-plus 计费（输入 0.8元/百万 token）：约 **0.01 元/天**
- 按 Claude Sonnet 计费（输入 $3/百万 token）：约 **0.04 元/天**

**加上 Cron 任务：**

- daily-brief：约 2000 token
- repo-watcher：24 次 * 1000 token = 24,000 token
- eod-summary：约 3000 token
- 日合计：约 **42,600 token/天**

总计不到 **0.15 元/天**。一杯咖啡都不到的成本换一个 24 小时在线的助手，我觉得很值。

**但要注意一个陷阱：** 当 Heartbeat 检测到异常并触发复杂 Skill 时（比如自动修复 CI），单次消耗可能飙到 10,000+ token。如果你的 HEARTBEAT.md 写得太激进（每次都让 Agent 深入分析），成本会翻 10 倍。

建议：

- Heartbeat prompt 保持精简，只做快速检查
- 复杂操作交给 Cron 或手动触发
- 设置月度 token 预算告警

## 凌晨三点的反模式

分享一个真实教训。

我最初的配置长这样：

```json5
{
  "heartbeat": {
    "every": "30m",
    "target": "all",          // 发到所有频道
    // 注意：没有 activeHours
    // 注意：没有 silentToken
    "promptFile": "./HEARTBEAT.md"
  }
}
```

三个致命错误：

1. **`target: "all"`** —— 检查结果发到所有关联的对话和群组
2. **没有 `activeHours`** —— 7x24 小时不间断执行
3. **没有 `silentToken`** —— 即使一切正常也会发消息（"已检查，一切正常"）

后果：凌晨 3:15，我的手机、钉钉群、同事的群全部炸了。Agent 在所有频道发了一条"已巡检完毕，未发现异常"。

三个同事被吵醒。第二天站会变成了批斗会。

正确配置：

```json5
{
  "heartbeat": {
    "every": "45m",
    "target": "last",                    // 只发到最近对话
    "activeHours": {
      "start": "08:00",
      "end": "21:00",
      "timezone": "Asia/Shanghai"
    },
    "silentToken": "HEARTBEAT_OK",       // 没事别说话
    "promptFile": "./HEARTBEAT.md"
  }
}
```

记住这三条规则：

1. **永远设置 `activeHours`**，除非你真的需要 7x24 监控（那也应该只发到告警频道）
2. **永远设置 `silentToken`**，没有异常就闭嘴
3. **`target` 永远用 `last` 或指定频道**，别用 `all`

## 总结

|  | Heartbeat | Cron |
|--|-----------|------|
| 用途 | 发现异常 | 按时执行 |
| 触发 | 间隔 | 时刻 |
| 无事时 | 沉默 | 仍然输出 |
| 典型场景 | CI 监控、服务告警 | 日报、归档、竞品监控 |

两条原则：

- **Heartbeat 用于"坏了告诉我"** —— 被动发现问题
- **Cron 用于"到点就做"** —— 主动执行计划

不要混用。不要让 Heartbeat 做日报（因为它可能因为 HEARTBEAT_OK 而跳过），不要让 Cron 做监控（因为 1 小时的间隔可能漏掉关键告警）。

配好这两个之后，OpenClaw 就真正从一个"你问它答"的聊天机器人，变成了一个"替你盯着、到点干活"的数字同事。

下一篇我们讲 Skill 编排——怎么让 Agent 在检测到 CI 失败后，自动定位问题、生成修复 PR、并通知你 review。
