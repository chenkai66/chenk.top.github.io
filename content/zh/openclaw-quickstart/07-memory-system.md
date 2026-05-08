---
title: "OpenClaw 快速上手（七）：记忆系统——别指望魔法"
date: 2026-04-09 09:00:00
tags:
  - openclaw
  - 记忆
  - context-engine
  - bge-m3
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 7
description: "MEMORY.md 作为 40 行索引、压缩前的 memoryFlush、通过 bge-m3 的语义搜索，以及 v2026.3.7 的 ContextEngine——终于把记忆从 Agent 的手里接管过来了。它真正帮你省了什么，以及哪里还在漏。"
disableNunjucks: true
translationKey: "openclaw-quickstart-7"
---

## 记忆是第一个翻车的地方

装完 OpenClaw 的人，十个有八个第一天就在记忆系统上栽跟头。不是配置难——配置反而简单——而是对"记忆"这个词的预期完全错了。

你以为装上就是 Jarvis，什么都记得住。实际情况是：Agent 昨天还知道你的 API Key 放在哪，今天又问你一遍。你告诉它三次"别用 gpt-4o，用 qwen-plus"，第四次它又犯。

这不是 bug，这是设计。Agent 的上下文窗口是有限的，记忆系统的本质是一套**决定什么该出现在上下文里**的工程。本篇把这套工程拆开讲。

![记忆系统架构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/07-memory-system/illustration_1.jpg)

## Workspace 目录结构

OpenClaw 的记忆全部落盘在 `~/.openclaw/workspace/` 下，没有数据库，纯文件：

```
~/.openclaw/workspace/
├── MEMORY.md            # 索引文件，每次对话开头注入
├── HEARTBEAT.md         # 最近一次活跃状态
├── memory/
│   ├── projects.md      # 活跃项目上下文
│   ├── lessons.md       # 经验教训（长期）
│   ├── 2026-04-08.md    # 当日记录
│   ├── 2026-04-07.md    # 前日记录
│   └── archive/         # 超过 14 天的自动归档
└── embeddings/
    └── index.bin        # bge-m3 向量索引
```

核心逻辑：**MEMORY.md 是索引，不是数据库。** 它告诉 Agent "去哪里找"，而不是把所有信息塞进去。

## 第一个错误：把 MEMORY.md 当垃圾场

我见过太多人（包括早期的我自己）把 MEMORY.md 写成 200 行的知识库。结果如何？

- 200 行 Markdown 约等于 3000 tokens
- 每轮对话开头就吃掉 3000 tokens
- Agent 反而因为信息过载找不到关键内容
- 更讽刺的是：你写了 200 行，Agent 真正需要的可能只有 3 行

正确做法：MEMORY.md 控制在 **40 行以内**，只放指针和最高优先级的身份信息。

## 四层记忆模型

记忆不是一坨东西，它有层次。我用一个表来说明：

| 层级 | 名称 | 生命周期 | 存储位置 | 举例 |
|------|------|----------|----------|------|
| L0 | 身份 | 永久 | MEMORY.md | 用户名、角色、核心偏好 |
| L1 | 决策 | 数月 | projects.md | 当前项目架构、技术选型 |
| L2 | 经验 | 数年 | lessons.md | "别用 SSH，用 ecs-run" |
| L3 | 痕迹 | 数天 | YYYY-MM-DD.md | 今天改了什么、遇到了什么 |

四层各有各的写法、各有各的生命周期。把 L3 的内容放到 L0，就是在用索引当日记写。

## 写好记忆条目：三组对比

记忆条目的质量直接决定系统有没有用。看三组对比：

**对比一：API 变更**

```
# 差：模糊、无日期、无法定位
API 改了

# 好：具体、可执行、带日期
2026-04-08: DashScope qwen-plus endpoint 从 /v1 迁移到 /compatible-mode/v1，旧端点 5 月废弃
```

**对比二：用户偏好**

```
# 差：冗余、没有优先级
我喜欢简洁的代码，不要太复杂，要有注释，变量名要清楚

# 好：精简、可判断
代码风格：类型标注必须、docstring 必须、单文件 <300 行、无 emoji
```

**对比三：项目状态**

```
# 差：过时信息混着当前信息
项目用的是 Next.js，之前用 Vue 但迁过来了，数据库从 MySQL 换到 PostgreSQL 了

# 好：只保留当前态
Tech stack: Next.js 14 + PostgreSQL 16 + Prisma 5。迁移历史见 memory/projects.md
```

**示范 MEMORY.md（10 行索引格式）：**

```markdown
# Memory

## Identity
- 陈锴（嘿呀），云计算文档工程师
- 偏好：无 emoji、中文直接、要深度不要泛

## Active Projects
- [Aidge 文档](memory/projects.md#aidge) — 跨境电商 AI 产品，5 月底上线
- [Ticket KB](memory/projects.md#ticket-kb) — ECS 工单知识库，18513 条

## Critical Rules
- [绝对不用 Wechaty](memory/lessons.md#wechaty) — 封号风险
- [不用 SSH](memory/lessons.md#ssh) — 沙箱阻断，用 ecs-run
```

这就够了。40 行以内，Agent 每轮能瞬间定位到需要的信息。

## memoryFlush：压缩前的保命机制

OpenClaw 的上下文管理有一个"压缩"步骤：当对话长度接近窗口上限时，系统会丢弃早期内容。问题来了——如果用户在对话前半段告诉 Agent 一个关键信息，压缩之后这个信息就没了。

`memoryFlush` 解决的就是这个问题：在压缩发生之前，把对话中新产生的记忆写入磁盘。

配置方法（`~/.openclaw/config.json`）：

```json
{
  "memory": {
    "memoryFlush": {
      "enabled": true,
      "softThresholdTokens": 12000,
      "hardThresholdTokens": 16000,
      "flushTarget": "memory/daily"
    }
  }
}
```

参数解释：

- `softThresholdTokens`：软阈值。达到这个量时，系统开始标记哪些内容该持久化
- `hardThresholdTokens`：硬阈值。达到这个量时，强制执行 flush + compact
- `flushTarget`：写入的目标目录，默认按日期落盘

## 没有 memoryFlush 会怎样

真实场景：

1. 你在对话第 5 轮告诉 Agent："我们的模型提供商从 OpenAI 换成了 DashScope"
2. 对话继续到第 20 轮，触发压缩
3. 压缩算法判断第 5 轮的内容"不够近期"，丢弃
4. 第 21 轮，Agent 调用 API 时继续用 OpenAI 的 endpoint
5. 报错。你再告诉它一次。循环往复。

开了 memoryFlush 之后，第 5 轮的信息在软阈值时就已经写入 `memory/2026-04-08.md`，压缩丢不掉了。

**第一天就开 memoryFlush。不开等于没装记忆系统。**

## memorySearch + bge-m3：语义搜索

当记忆文件积累到一定量，光靠文件名和索引已经不够用了。这时候需要语义搜索——用向量匹配找到相关记忆。

OpenClaw 使用 bge-m3 作为默认 embedding 模型。对国内用户来说，SiliconFlow 提供免费的 bge-m3 API，不需要翻墙：

```json
{
  "memory": {
    "memorySearch": {
      "enabled": true,
      "provider": "siliconflow",
      "model": "BAAI/bge-m3",
      "endpoint": "https://api.siliconflow.cn/v1/embeddings",
      "apiKey": "sk-your-siliconflow-key",
      "topK": 3,
      "minScore": 0.65
    }
  }
}
```

注意事项：

- SiliconFlow 的免费额度足够个人使用（每天 10 万 tokens embedding）
- bge-m3 对中英文混合场景表现很好，不需要单独配中文模型
- `topK: 3` 意味着每次搜索最多返回 3 条结果，够用且不浪费 budget
- `minScore: 0.65` 是经验值，低于这个分数的结果噪音太多

**重要提醒：memorySearch 至少要积累一周的数据才有意义。** 刚装完第一天就开语义搜索，索引里什么都没有，搜出来全是噪音。先老老实实手写两周记忆，再开搜索。

## 记忆类型详解

OpenClaw 内部把记忆分成五种类型，每种有不同的写入逻辑和生命周期：

**user（用户偏好）**
- 内容：名字、角色、代码风格、沟通偏好
- 来源：用户明确说 "我叫..."、"我喜欢..."
- 存储：MEMORY.md
- 生命周期：永久，除非用户主动修改

**project（项目状态）**
- 内容：技术栈、当前阶段、关键文件路径
- 来源：Agent 在工作过程中自动提取
- 存储：memory/projects.md
- 生命周期：项目存续期间，结束后归档

**feedback（用户纠正）**
- 内容：用户对 Agent 行为的纠正
- 来源："不要这样做"、"以后用 X 代替 Y"
- 存储：memory/lessons.md
- 生命周期：长期有效

**reference（事实参考）**
- 内容：API endpoint、配置参数、账号信息
- 来源：用户提供或 Agent 在工作中发现
- 存储：memory/ 目录下对应的 ref 文件
- 生命周期：直到信息过期

**lesson（血泪教训）**
- 内容：犯过的严重错误、绝对不能踩的坑
- 来源：用户强调 "绝对不要"、Agent 踩坑后记录
- 存储：memory/lessons.md
- 生命周期：永久

**判断流程（文字版）：**

```
用户说了一句话 →
  ├─ 是关于用户自己的？ → user
  ├─ 是纠正 Agent 行为的？ → feedback/lesson
  ├─ 是项目当前状态的？ → project
  ├─ 是事实性信息（URL、配置）？ → reference
  └─ 都不是 → 不记录
```

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
