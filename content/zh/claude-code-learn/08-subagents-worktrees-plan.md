---
title: "Claude Code 实战（八）：Sub-Agent 与计划模式"
date: 2026-04-25 09:00:00
tags:
  - claude-code
  - sub-agents
  - worktrees
  - plan-mode
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 8
description: "三个改变 Claude Code 一次能扛多少事的特性：子 Agent 用来并行调研、worktree 用来物理隔离、计划模式用在它真要动手之前。三者的边界，以及各自不该用的场景。"
disableNunjucks: true
translationKey: "claude-code-learn-8"
---
说完 hooks，接下来让 Claude Code 手感大变的就是*并发控制*了。这里说的并发不是线程层面的，而是指模型到底同时在帮你做多少事、隔离程度如何、你需要盯着多紧。

三个功能，按所需信任度从低到高排列。

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

## Plan mode —— 气闸舱

Plan mode 成本最低。按 `Shift+Tab` 直到指示器显示 **plan**。这时候模型只规划不行动。它会阅读、思考、提出方案，然后停住。你来看计划。要么 approve，要么 edit，要么直接 kill。只有这一步过了，它才会执行。

我一般在这些时候用：

- 任何非 trivial 任务的前 30 秒。比如“实现 X 功能”→ 先 plan。几乎每次，计划都会暴露模型误解了代码库。
- 任何涉及 auth、支付、schema 迁移或生产配置的操作。花两秒扫一眼，能省下几小时收拾烂摊子的时间。
- 在不熟悉的 repo 里干活。这份计划顺便就成了我的 onboarding 文档。

容易踩的坑：觉得“任务小”就跳过 plan mode。恰恰是小任务最容易冒出“等等，这不是我想要的”这种状况。

## Sub-agents —— 适合并行跑的任务

子代理是父 agent spawn 出来的一个 Claude Code 实例，用来处理 scoped task。经典写法放在 `.claude/agents/<name>.md`：

```markdown
---
name: research
description: Reads a topic across the codebase and reports findings. No edits.
tools: Read, Grep, Glob, WebFetch
---

You are a research sub-agent. Your job:
1. Search the codebase for the requested topic.
2. Read enough files to understand it deeply.
3. Return a structured report with file paths and quotes.

Do not edit. Do not run shell commands. Stay focused.
```

然后在对话里说：“research 一下认证怎么工作的 → 用 research agent。”

这能换来什么：

- **上下文隔离。** 子代理有自己的 context window。父 agent 的保持干净。
- **工具限制。** research agent  literally 无法编辑。这是架构级别的安全，不是靠自律。
- **并行工作。** 任务独立时，你可以同时 fan out 给三个子代理。

代价是什么：

- Tokens。每个子代理都有自己的 system prompt、自己的上下文、自己的来回对话。
- 协调。父 agent 得合并结果。这一步要显式规划好。

什么时候不该用子代理：父 agent 已经拥有所需上下文的任何任务。spawn 一个子代理去“读这一个文件然后汇报”，纯粹的昂贵递归。

## Worktrees —— 并行分支且不至于搞疯自己

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_2.png)

git worktree 是同一个 repo 的第二个 working tree，在不同分支，不同目录。Claude Code 认识这东西：`EnterWorktree` 工具会创建新分支 + worktree 并把 session 切换进去。

这些场景很关键：

- 你正在 `feat/x` 上干活，用户突然要求在 `main` 上修个无关的 quick fix。Spawn 一个 worktree，修完，commit，退出。
- 你想尝试同一问题的两种不同解法，又不想用废弃的 commit 弄脏主干。
- 你委托给子代理，希望它在物理上跟你的 working tree 隔离。

心智模型：worktree 是上下文隔离的*物理*版本。子代理隔离上下文；worktree 隔离文件系统。

关于退出怎么想：

- `keep` — worktree 留在磁盘上。工作未完成或可能回来继续时用。
- `remove` — 彻底删除，分支也删掉。只有确定不再需要时才用。

如果 worktree 里有未 commit 的变更，除非你确认 `discard_changes: true`，否则拒绝删除。这是对的。别想着绕过它。

## 组合这三者

处理硬任务时我的模式：

1. **Plan mode。** “我要做这个；你打算怎么做？”看计划。调整。
2. **Worktree。** 进一个隔离分支，这样实验失败了也不至于弄脏主干。
3. **Sub-agents** 处理 worktree 内的独立子任务。先 research，再 implementation，最后写 test —— 每个都在自己的上下文里。
4. 回到父 agent，合并结果，commit，退出 worktree（工作暂停用 `keep`，完成了用 `remove`）。

三层信任门，三级 escalation。等到模型开始改文件时，你投入的注意力刚好匹配任务所需的难度。

## 什么时候都不用

大多数任务。说真的。80% 的情况就是“改这个函数，跑测试，发布”—— 普通模式，不要子代理，不要 worktrees。上面那些功能是为那 20% 庞大、不可逆或分支复杂的任务准备的。

如果你发现自己每个任务都想掏子代理和 worktree，更有趣的问题是：你是不是把任务拆得太大了。