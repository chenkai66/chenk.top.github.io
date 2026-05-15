---
title: "OpenClaw 指南（七）：记忆系统去魔法化"
date: 2026-04-14 09:00:00
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
series_total: 10
description: "MEMORY.md 索引、memoryFlush、bge-m3 语义搜索，以及 ContextEngine 的工作原理。"
disableNunjucks: true
translationKey: "openclaw-quickstart-7"
---
前六篇已经带你跑通了一个可用的 OpenClaw，渠道和技能都已具备；这一篇将专门解决初次安装时最常见的问题——记忆系统。

![OpenClaw 快速入门 (7)：记忆系统，去掉魔法 — 示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/07-memory-system/illustration_1.png)

---

## 工作区的样子

![记忆系统四层架构 — 热、温、冷和归档层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/07-memory-system/fig_memory.png)

打开 `~/.openclaw/workspace/`。你应该能看到这样的结构：

```text
~/.openclaw/workspace/
├── MEMORY.md              # 索引 — 控制在 40 行以内
├── HEARTBEAT.md           # 代理按 cron 周期重读的规则
└── memory/
    ├── projects.md        # 当前项目状态
    ├── lessons.md         # 踩过的坑，不想再犯的错
    ├── 2026-04-14.md      # 原始日报日志
    └── archive/           # 超过 30 天的内容
```

最关键的是两个文件。`MEMORY.md` 是索引——代理每轮对话都会读它，所以必须短。`lessons.md` 存放长尾知识，比如“我们试过那个，因为 Y 原因挂了”。日报日志写多读少，代理通常只通过搜索去调取。

我头两个月踩的坑：恨不得把所有东西都塞进 `MEMORY.md`。等它涨到 200 行时，每轮对话光加载索引就要消耗约 3k tokens，代理还是找不到需要的信息。索引不是归档。

## 四层思维模型

决定某件事该放哪儿时，我脑子里始终有这张表：

| Tier | Lifespan | Goes in |
|------|----------|---------|
| Identity | forever | `MEMORY.md` (name, voice, role) |
| Decision | months | `projects.md` |
| Lesson | years | `lessons.md` |
| Trace | days | `memory/YYYY-MM-DD.md` |

无法归入这四类的信息，大概率是噪音，可以直接丢弃。

## 写好记忆条目

存什么远比存多少重要。每条记忆都必须让代理能够直接决策，无需再询问。

**糟糕的条目** — 模糊、没日期、不可执行：

```markdown
- We use some API for search
- The deploy was broken last time
- User prefers short responses
```

这对代理几乎没用。哪个 API？什么时候挂的？“短”是多短？

**好的条目** — 具体、可执行、带日期：

```markdown
- [2026-03-22] Search API: SiliconFlow bge-m3, free tier, 1000 req/day limit
- [2026-04-01] Deploy: nginx proxy_pass must include trailing slash or 404 on /api routes
- User prefers responses under 3 sentences for status updates; full detail for technical questions
```

现在代理不用猜了，它知道提供商、约束条件和具体的行为规则。

这是一个结构良好的 `MEMORY.md` 示例——对大多数 solo 用户来说足够了：

```markdown
# Memory
## Identity
- Name: Kai. Role: Backend engineer. Voice: direct, no filler.
## Active Projects
- [ticket-kb](memory/projects.md) — ECS ticket classification, due 2026-05-15
- [blog](memory/projects.md) — Hugo site, deploy via OSS
## Key References
- [Server registry](memory/ref_servers.md) — 3 ECS instances
- [API keys](memory/ref_apis.md) — providers, rate limits
## Lessons (top 3)
- Never SSH from sandbox; use ecs-run wrapper
- Wechaty banned; WeChat automation = account freeze
- memoryFlush on before any long session
```

十行，加载消耗不到 200 tokens。代理知道你是谁、在做什么、去哪里找资料以及什么不能做。其他内容都在引用文件中，通过搜索拉取。

## 记忆类型深挖

OpenClaw 内部识别五种记忆类型。你不必全用，但理解分类有助于正确路由新信息。

**1. 用户记忆** — 关于人的事实。名字、偏好、风格、时区。放在 `MEMORY.md` 的 Identity 下。很少变。

**2. 项目记忆** — 当前工作状态。已完成、被阻塞、下一步。放在 `projects.md`。每周变。

**3. 反馈记忆** — 纠正。“不是那样，是这样。”放在 `lessons.md` 并带日期。

**4. 参考记忆** — 稳定事实。服务器 IP、端点、配置。放在独立的 `ref_*.md` 文件。很少变。

**5. 教训记忆** — 从失败中泛化的洞察。“如果 X，则 Y 会挂。”放在 `lessons.md`。保质期长。

我用的决策流程图：用户纠正你了？那是反馈，带日期进 `lessons.md`。是关于用户本人的事实？用户记忆，进 `MEMORY.md`。关于当前任务状态？项目记忆，进 `projects.md`。像 IP 或端点这样的稳定参考？参考记忆，进 `memory/ref_*.md`。从失败中得出的泛化洞察？教训，进 `lessons.md`。都不沾边？大概率是噪音——写进日报或丢弃。

拿不准的话，先写进日报；若后续频繁搜索，再升至对应层级。

## memoryFlush — 我唯一必改的配置

长对话在上下文窗口紧张时会自动压缩。默认情况下，代理会丢掉那些没写下来的内容。`memoryFlush` 会在压缩*之前*运行一次“保存重要内容”的操作：

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

`softThresholdTokens` 是 flush 操作本身能占用的空间。4000 足够做个有用的摘要；除非你做超长会话，否则 8000 就是浪费。

### 不开此功能会怎样？

这是开启前的真实场景：调试会话进行到第 80 轮，第 20 轮时我已说明因速率限制将 embedding 提供商从 OpenAI 切换至 SiliconFlow，代理确认后继续；但到第 60 轮上下文紧张，压缩操作裁剪了第 15–45 轮——关键切换信息恰在此区间，彻底丢失；第 70 轮我令其重新索引，代理却仍调用 OpenAI 导致失败，十分钟白耗在本该持久化的知识上。

开启 `memoryFlush` 后，flush pass 会先触发——在修剪前把“Embedding 提供商已改为 SiliconFlow，OpenAI 已弃用”写进 `projects.md`。压缩后，代理重载并读取到这条信息。

- **Without memoryFlush:** 压缩静默删除，代理会回退到过时的假设。
- **With memoryFlush:** 压缩先触发写入 pass，关键事实持久化。

如果只改一项默认配置，就选它。

## memorySearch — 以及为什么 bge-m3 够用了

你可以对记忆文件运行语义搜索。你需要一个 embedding API。便宜的路子是 SiliconFlow 的 `BAAI/bge-m3`，免费且够用：

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

大概用了一周后你才会看到效果，在此之前没什么可搜的。

## 记忆预算算笔账

花在记忆上的每个 token 都不能用于对话，让我们来算笔账。

`MEMORY.md` 40 行大约消耗 600 tokens — 这是你每轮的固定开销。`memorySearch` 返回的每个文件块消耗 200–500 tokens。默认检索预算是 2000 tokens，意味着每轮 3–4 个块。

每轮总开销：600 + 2000 = 2600 tokens。在 128k 模型上，大约 2%。可以接受。

用 `max_memory_tokens_per_turn` 调整：

```json
{
  "agents": {
    "defaults": {
      "context_engine": {
        "max_memory_tokens_per_turn": 2000
      }
    }
  }
}
```

为什么不拉到 4000 或更高？因为记忆会与对话竞争资源。到了 4000，开销上的 token 就达到 4600 个。在多轮对话中，如果回复预算是 6000 tokens，近一半被检索占用，代理的回复会被挤压。

最佳甜蜜点：快速任务（< 10 轮）用 1000，标准会话（默认）用 2000，长研究用 3000。绝不超过 4000，否则会挤占对话空间。如果代理在 2000 时错过上下文，解决办法是写好记忆条目，而不是增加预算。

## ContextEngine — v2026.3.7 的变动

![OpenClaw 快速入门 (7)：记忆系统，去掉魔法 — 示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/openclaw-quickstart/07-memory-system/illustration_2.png)

在 v2026.3.6 之前，记忆是代理必须显式调用的工具，如果忘记，就会在没有上下文的情况下运行。从 v2026.3.7 开始，记忆移到了 harness 围绕代理运行的 hooks 生命周期中：

```text
bootstrap   → load priors before turn 1
ingest      → soak up new facts as they appear
assemble    → build the final prompt slice
compact     → trim under budget
afterTurn   → write back what changed
```

代理不再决定何时记忆，由 harness 决定。这听起来很小，其实不然，这是“代理主动记忆”和“系统自动记忆，无法跳过”之间的区别。

默认 ContextEngine 对大多数人够用。如果你想换进 RAG 风格的检索器或知识图谱后端，也可以——代理配置不变，只换引擎。

## 自动写入的争议

ContextEngine 默认引入了 `auto_write: true`。代理不经询问直接写入记忆文件——每个纠正、新事实或项目更新都会进去。

对于带个人助理的 solo 用户，这很棒。你从不说“记住那个”，代理一周内就能捕捉到模式。

对于共享机器人，比如五个人的团队房间，自动写入就是灾难。A 说“永远用 TypeScript”，B 说“我们是 Python 店”，两者都被写入，没有归属，记忆变得矛盾。

配置如下：

```json
{
  "agents": {
    "defaults": {
      "context_engine": {
        "auto_write": true
      }
    }
  }
}
```

共享机器人设为 `false`。代理仍然读记忆，但除非被要求否则不写。审计最近的写入：

```bash
openclaw memory log --since 7d
```

这会打印过去 7 天内的每次记忆写入——文件、行、内容以及触发的轮次。我每周跑一次，偶尔代理会写错东西（归因错了偏好，或者把一次性事件记成了永久规则），尽早发现并修正，成本很低。

我的经验法则：

- **Solo 用户，个人助理：** `auto_write: true`。每周审查。
- **共享机器人，团队房间：** `auto_write: false`。仅使用显式记忆命令。
- **混合模式：** `auto_write: true` 配合审查 cron，标记非主用户条目。

## 在 Agent 之间迁移记忆

迟早你会创建第二个 Agent，或者更换机器、升级环境，想要重新开始但又不想丢掉上下文。记忆迁移其实不难，但有些坑需要注意。

**复制工作区**

最简单的迁移方案：直接把 `~/.openclaw/workspace/` 拷贝到新位置。全是 Markdown，便携性没问题。

```bash
tar -czf openclaw-memory-backup.tar.gz ~/.openclaw/workspace/
# On new machine:
tar -xzf openclaw-memory-backup.tar.gz -C ~/
```

如果在同一台机器上创建第二个 Agent，拷贝后记得修剪。项目 Agent 不需要你的身份板块，个人助理也不需要其他项目的参考资料。

**重新索引 Embedding**

`~/.openclaw/index/` 里的向量索引跨模型不通用。新机器上是同一个模型？直接拷过去。换了模型？删掉重建：

```bash
rm -rf ~/.openclaw/index/
openclaw memory reindex
```

一般工作区（20–30 个文件）重新索引只需不到一分钟。

**会话导出到记忆**

如果会话中产生了关键决策但未及时持久化，事后仍可补救：

```bash
openclaw memory export --session <session-id> --type decisions
```

这会让 Agent 跑一遍会话记录，把决策、偏好和经验教训抽出来放到 `memory/staged.md` 里供你 review。它可能过度提取，需人工审核；但相比重读 200 轮会话，效率仍显著更高。

**别迁移这些**

`memory/archive/`、会话 `.jsonl` 文件（除非你要从中导出），还有换了 Embedding 提供商时的 `index/` 目录，这些都跳过。过期的向量比没有更糟——它们会自信地返回错误结果。

## 哪里还会漏

有三件事还是会坑到我：

1. **群聊和子 Agent 默认不读 `MEMORY.md`。** 这是故意的（沙箱隔离），但如果你忘了，就会奇怪为什么团队机器人不知道你的名字。
2. **Embedding 漂移。** 换了模型，旧向量就失效了，要么重新索引，要么忍受召回率下降。
3. **40 行纪律。** 没人强制约束。设个每周 cron，如果 `wc -l MEMORY.md` 超过 40 行就报错。

我每周日都会跑个快速健康检查：

```bash
wc -l ~/.openclaw/workspace/MEMORY.md       # < 40
ls -lt ~/.openclaw/workspace/memory/*.md | head
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
```

如果 `MEMORY.md` 悄悄超过 40 行，说明有些内容该移到 `projects.md` 或 `lessons.md` 里了。如果会话文件堆了几百个，赶紧归档——启动速度会变慢。

## 总结
- `MEMORY.md` 是索引，不是数据库。
- 第一天就打开 `memoryFlush`。
- 等有值得搜索的内容后再加 `memorySearch`。
- v2026.3.7 之后，别再让 Agent 去记东西；让引擎来做。
- 记忆决定了 Agent 像个工具还是像个同事。这份纪律值得遵守。
