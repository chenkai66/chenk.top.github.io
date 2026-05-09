---
title: "OpenClaw 快速上手（七）：记忆系统——别指望魔法"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - memory
  - context-engine
  - bge-m3
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 7
description: "MEMORY.md 索引、memoryFlush、bge-m3 语义搜索，以及 ContextEngine 的工作原理。"
disableNunjucks: true
translationKey: "openclaw-quickstart-7"
---

## 记忆是第一个翻车的地方

装完 OpenClaw 的人，十个有八个第一天就在记忆系统上栽跟头。不是配置难——配置反而简单——而是对"记忆"这个词的预期完全错了。

你以为装上就是 Jarvis，什么都记得住。实际情况是：Agent 昨天还知道你的 API Key 放在哪，今天又问你一遍。你告诉它三次"别用 gpt-4o，用 qwen-plus"，第四次它又犯。

这不是 bug，这是设计。Agent 的上下文窗口是有限的，记忆系统的本质是一套**决定什么该出现在上下文里**的工程。本篇把这套工程拆开讲。

![记忆系统架构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/illustration_1.jpg)

## Workspace 目录结构

![记忆系统四层架构——热、温、冷、归档层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/fig_memory.png)

OpenClaw 的记忆全部落盘在 `~/.openclaw/workspace/` 下，没有数据库，纯文件：

![记忆类型分类决策](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/fig_memory_decision_zh.png)

## ContextEngine v2026.3.7：从工具变成生命周期

在 v2026.3.7 之前，记忆是一个"工具"——Agent 决定什么时候调用 `memoryWrite`、什么时候调用 `memorySearch`。问题很明显：Agent 经常忘记写、忘记搜，或者在不该写的时候乱写。

v2026.3.7 引入了 ContextEngine，把记忆从 Agent 可调用的工具变成了 **harness 生命周期的一部分**。Agent 不再决定"要不要记"，系统在每个阶段自动处理：

| 阶段 | 名称 | 动作 |
|------|------|------|
| 1 | bootstrap | 加载 MEMORY.md，注入身份和索引 |
| 2 | ingest | 解析用户消息，识别潜在记忆条目 |
| 3 | assemble | 根据当前话题执行 memorySearch，拼装上下文 |
| 4 | compact | 对话过长时 flush + 压缩 |
| 5 | afterTurn | 每轮结束后，写入新产生的记忆 |

这个转变的意义在于：**记忆的可靠性不再依赖 Agent 的"自觉性"。** 你不需要在 prompt 里写"记得保存重要信息"这种无效指令了。

![ContextEngine 生命周期](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/illustration_2.jpg)

升级到 v2026.3.7 后，如果你之前有自定义的 `memoryWrite` 工具配置，需要迁移到 ContextEngine 格式：

```json
{
  "contextEngine": {
    "enabled": true,
    "version": "v2",
    "memoryBudgetTokens": 2000,
    "autoWrite": true,
    "autoSearch": true
  }
}
```

## 记忆预算的数学

这是很多人不算的一笔账：

- MEMORY.md 40 行，约 600 tokens
- 每条 memorySearch 结果，200-500 tokens
- 默认 `memoryBudgetTokens: 2000`
- 2000 tokens 的预算，减去 600（MEMORY.md），剩 1400
- 1400 / 350（平均每条结果）= 4 条搜索结果

也就是说，每轮对话中，系统最多能注入 MEMORY.md + 3~4 条相关记忆文件的片段。

**为什么不把 budget 调到 4000 甚至更高？**

- 每增加 1000 tokens 的记忆注入，Agent 的有效推理空间就少 1000 tokens
- 超过 4000 tokens 的记忆注入，实测会导致 Agent 对当前任务的注意力下降
- 信息越多，Agent 越容易"选择性忽略"其中一部分
- 多数场景下，2000 tokens 的预算（索引 + 3 条精准搜索结果）完全够用

经验法则：**除非你的项目有超过 50 个活跃记忆条目且高度互相关联，否则不要碰默认值。**

## auto_write 的取舍

ContextEngine 的 `autoWrite: true` 意味着系统自动判断哪些对话内容值得记录，不需要用户手动触发。这在个人使用时很方便，但有场景需要关掉：

**适合开 autoWrite 的场景：**
- 单人使用，你是唯一与 Agent 交互的人
- 日常开发，信息变化频繁需要实时记录
- 懒得手动管理记忆（大多数人）

**需要关 autoWrite 的场景：**
- 共享 Bot（多人使用同一个 Agent 实例）——不同人的偏好会互相覆盖
- 演示/教学场景——不希望临时对话污染长期记忆
- 安全敏感环境——不想 API Key 之类的内容被自动记录

关掉 autoWrite 后，你仍然可以手动触发记忆写入：

```bash
openclaw memory write "DashScope endpoint 已迁移到 /compatible-mode/v1"
```

**审查 autoWrite 的写入记录：**

```bash
# 查看最近 7 天的自动写入
openclaw memory log --since 7d

# 输出示例：
# 2026-04-08 14:32  [auto] project: 新增 Aidge 文档项目配置
# 2026-04-08 15:01  [auto] reference: DashScope qwen-plus endpoint 变更
# 2026-04-07 09:15  [auto] feedback: 不用 emoji
# 2026-04-06 16:44  [auto] lesson: SSH 在沙箱中被阻断
```

建议每周花 2 分钟扫一遍，删掉不准确的条目。

## 跨机器迁移记忆

换电脑、换 Agent 框架、或者想在两台机器上同步记忆，操作如下：

**同框架迁移（OpenClaw -> OpenClaw）：**

```bash
# 源机器：打包 workspace
tar -czf openclaw-memory-backup.tar.gz ~/.openclaw/workspace/

# 目标机器：解压
tar -xzf openclaw-memory-backup.tar.gz -C ~/

# 重建向量索引（因为 index.bin 是机器相关的）
openclaw memory reindex
```

**跨框架迁移（导出为通用格式）：**

```bash
# 导出所有记忆为 JSON
openclaw memory export --format json > memory-export.json

# 在新框架中导入
# 具体命令取决于目标框架，但 JSON 格式是通用的
```

**关于 embeddings/index.bin：**

- 这个文件**不需要迁移**，在目标机器上 `reindex` 即可重建
- 它是 bge-m3 对所有记忆文件的向量缓存
- 重建时间取决于记忆量，通常几百条记忆 < 30 秒

**session 导出（保留完整对话历史）：**

```bash
# 导出最近 30 天的对话 session
openclaw session export --since 30d --output sessions-backup/
```

session 不属于记忆系统，但迁移时一起带上可以保留完整的上下文恢复能力。

## 还在漏的地方

记忆系统不是万能的，以下是当前版本（v2026.3.7）已知的短板：

**1. 群聊场景不读 MEMORY.md**

如果你在群聊中 @Agent，当前实现不会加载 MEMORY.md。Agent 在群里就是一个"失忆"状态。Workaround：在群聊的 system prompt 里手动注入关键信息。

**2. Embedding 漂移**

bge-m3 的向量表示是静态的——同一句话永远得到同一个向量。但你的用词习惯会变化。半年前你说"部署"，现在你说"发布"，语义搜索可能匹配不上旧记忆。

Workaround：每季度跑一次 `openclaw memory audit`，把过时的表述更新一下。

**3. 40 行纪律没有强制执行**

系统不会阻止你把 MEMORY.md 写到 200 行。它只会在超过 60 行时输出一条 warning，但不会拒绝加载。自律是唯一的防线。

**4. 跨 Agent 实例不同步**

如果你同时开着两个 OpenClaw session，它们的 memory 写入可能冲突。目前没有锁机制，后写入的会覆盖先写入的。最佳实践：同一时间只开一个 session。

**5. 无法从记忆中"遗忘"**

删除记忆文件后，对应的 embedding 不会自动清理。需要手动 `openclaw memory reindex` 才能彻底移除。

## 每周健康检查

建议每周日花 5 分钟做一次记忆系统体检：

```bash
# 1. 检查 MEMORY.md 行数（应该 < 40）
wc -l ~/.openclaw/workspace/MEMORY.md

# 2. 检查记忆总量
openclaw memory stats

# 输出示例：
# Total entries: 147
# MEMORY.md: 38 lines (950 tokens)
# Active projects: 3
# Lessons: 12
# Daily logs (14d): 14 files
# Archive: 89 files
# Index health: OK (last rebuilt 2d ago)

# 3. 检查过时条目
openclaw memory audit --stale 30d

# 4. 查看自动写入质量
openclaw memory log --since 7d --type auto

# 5. 清理归档（可选，超过 90 天的日志）
openclaw memory archive --older-than 90d
```

整个流程 5 分钟。一周一次，记忆系统就能保持健康状态。

## 总结

记忆系统的核心认知：

1. **MEMORY.md 是索引，不是数据库。** 40 行以内，只放指针和身份信息。
2. **第一天就开 memoryFlush。** 不开等于没有持久记忆，Agent 的"记忆力"和金鱼一样。
3. **v2026.3.7 之后让 ContextEngine 接管。** 不要再手动管 memoryWrite 调用时机，系统比 Agent 更可靠。
4. **语义搜索需要数据积累。** 先写两周再开 memorySearch，否则搜出来全是噪音。
5. **记忆预算是有限的。** 2000 tokens 默认值别乱改，信息过载比信息不足更致命。
6. **每周 5 分钟维护。** 不维护的记忆系统会在一个月后变成垃圾场。

记忆系统不是魔法。它是一套文件管理 + 向量检索 + 生命周期钩子的工程方案。期望越准确，失望越少。
