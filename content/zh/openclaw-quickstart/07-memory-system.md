---
title: "OpenClaw 快速上手（七）：把记忆系统讲清楚，不靠玄学"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - memory
  - context-engine
  - bge-m3
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 7
description: "MEMORY.md 是 40 行以内的索引，不是数据库；memoryFlush 在压缩前抢救信息；用 bge-m3 做语义搜索；v2026.3.7 之后 ContextEngine 把记忆从 Agent 手里拿回到 harness。这一篇讲它能省什么，又在哪里漏。"
disableNunjucks: true
translationKey: "openclaw-quickstart-7"
---
前六篇内容带你搭建了一个可用的 OpenClaw，配置了频道，还写了一个技能。这篇重点聊聊大家第一次安装时最容易出错的地方：记忆功能。
![OpenClaw 快速上手（七）：把记忆系统讲清楚，不靠玄学 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/illustration_1.jpg)

## workspace 的结构

打开 `~/.openclaw/workspace/`，你会看到以下内容：

```
~/.openclaw/workspace/
├── MEMORY.md              # 索引文件，控制在 40 行以内
├── HEARTBEAT.md           # 定时重读的规则文件
└── memory/
    ├── projects.md        # 当前项目的状态记录
    ├── lessons.md         # 踩过的坑，避免再犯
    ├── 2026-04-14.md      # 每日原始日志
    └── archive/           # 超过 30 天的内容归档到这里
```

有两个文件最重要。`MEMORY.md` 是索引文件，Agent 每次运行都会读取它，所以必须保持简短。`lessons.md` 用来记录那些“试了发现不行，原因是因为 Y”的经验教训。日常日志写得多、读得少，Agent 通常只通过搜索来查找这些内容。

我曾经犯过一个错误，持续了两个月：把所有东西都塞进 `MEMORY.md`。等它膨胀到 200 行时，每次运行光加载索引就要消耗大约 3000 个 token，而 Agent 依然找不到需要的信息。记住，**索引不是档案**。
## 四层心智模型

判断信息该放在哪里时，我脑子里会对照这张表：

| 层级 | 存留时间 | 放在哪里 |
|------|----------|----------|
| 身份 | 永久 | `MEMORY.md`（姓名、语气、角色） |
| 决策 | 几个月 | `projects.md` |
| 经验 | 几年 | `lessons.md` |
| 痕迹 | 几天 | `memory/YYYY-MM-DD.md` |

如果一条信息无法归入其中任何一类，那大概率是噪音。直接扔掉就行。
## memoryFlush——我一定会加的一行配置

对话太长时，系统会自动压缩上下文。默认情况下，没写下来的内容就直接丢了。`memoryFlush` 的作用是在压缩之前，先跑一轮“抢救重要信息”的操作：

```json
{
  "agents": {
    "defaults": {
      "compaction": {
        "reserveTokensFloor": 20000,
        "memoryFlush": {
          "enabled": true,
          "softThresholdTokens": 4000
        }
      }
    }
  }
}
```

`softThresholdTokens` 是抢救阶段能用的 token 数量上限。4000 够用了，足够生成一段有用的总结；8000 就有点浪费，除非你处理的是超长对话。
## memorySearch——bge-m3 完全够用

想对记忆文件做语义搜索，得先接入一个 Embedding API。最划算的选择是 SiliconFlow 提供的 `BAAI/bge-m3`，免费又好用：

```json
{
  "tools": {
    "memorySearch": {
      "enabled": true,
      "embedding": {
        "provider": "openai-compatible",
        "baseUrl": "https://api.siliconflow.cn/v1",
        "apiKey": "...",
        "model": "BAAI/bge-m3"
      }
    }
  }
}
```

用上一周左右，才能看到效果。之前没什么数据，搜也搜不到什么。
## ContextEngine——v2026.3.7 的转变

![OpenClaw 快速上手（七）：把记忆系统讲清楚，不靠玄学 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/illustration_2.jpg)

在 v2026.3.6 之前，记忆是一个显式的工具，Agent 必须主动调用 `memory_search` 或 `memory_get` 才能获取历史信息。一旦忘记调用，就等于没有记忆。

从 v2026.3.7 开始，记忆被整合到了 hooks 生命周期中，由 harness 自动围绕 Agent 运行：

```
bootstrap   → 在第一轮之前加载先验知识
ingest      → 实时吸收新出现的事实
assemble    → 构建最终的 Prompt 切片
compact     → 按预算限制进行裁剪
afterTurn   → 每轮结束后写回变化
```

现在，Agent 不再需要决定什么时候记住什么，harness 全权接管了这个任务。这听起来是个小改动，但影响深远。它把“Agent 记得去记”变成了“系统帮你记住，没得商量”。

默认的 ContextEngine 已经能满足大多数人的需求。如果想换成 RAG 检索器或者知识图谱后端，也可以做到——Agent 的配置完全不用动，只需换一个引擎插件即可。
## 它仍然会漏的地方

有三件事到现在还会让我踩坑：

1. **群聊和子 Agent 默认不读 `MEMORY.md`。** 这是设计如此——群聊上下文是共享的，子 Agent 本来就是沙箱隔离的——但如果你忘了这一点，可能会花一下午琢磨：为什么团队群里的 bot 连我的名字都不知道？
2. **Embedding 漂移问题。** 换了 embedding 模型后，旧的向量就没用了。要么重新建索引，要么接受召回率下降。没有自动迁移这回事。
3. **40 行纪律没人帮你守。** 我设了一个每周运行的 cron 任务，如果 `wc -l MEMORY.md` 超过 40 行就强制报错。比起靠自觉，这样更靠谱。

每到周日，我都会跑这几行命令做健康检查：

```bash
wc -l ~/.openclaw/workspace/MEMORY.md      # < 40
ls -lt ~/.openclaw/workspace/memory/*.md | head
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
```

如果 `MEMORY.md` 超过 40 行，说明有些内容该挪到 `projects.md` 或 `lessons.md` 里了。如果 session 文件堆积超过几百个，就得归档了——不然启动速度会越来越慢。
## 要点总结

- `MEMORY.md` 是索引，不是数据库。
- 一开始就启用 `memoryFlush`。
- 等有内容值得搜索时，再添加 `memorySearch`。
- 从 v2026.3.7 开始，别再让 Agent 记东西，交给引擎处理。
- 记忆功能决定了 Agent 是像工具还是像同事。这点规矩值得遵守。
