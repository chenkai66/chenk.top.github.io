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
说完 hooks，Claude Code 使用体验发生显著变化的下一个关键点是*并发控制*。此处的‘并发’并非指线程级并发，而是指模型在同一时间为你并行处理多少任务、各任务间的上下文隔离程度如何，以及你需要投入多少监督注意力。

三个功能，按所需信任度从低到高排列。

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

## Plan mode —— 气闸舱

Plan mode 成本最低。按 `Shift+Tab` 直到指示器显示 **plan**。这时候模型只规划不行动。它会阅读、思考、提出方案，然后停住。由你审阅该计划：可批准（approve）、修改（edit），或中止（kill）。只有这一步过了，它才会执行。



![Plan 模式生命周期：阅读、计划、批准、执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig5.png)
*图 1. Plan mode 生命周期。意图与动作之间的气闸舱。*

我一般在这些时候用：

- 任何非 trivial 任务的前 30 秒。比如“实现 X 功能”→ 先 plan。几乎每次，计划都会暴露出模型对代码库的理解偏差。
- 任何涉及 auth、支付、schema 迁移或生产配置的操作。只需花两秒快速浏览，就可能避免数小时的故障修复。
- 在不熟悉的 repo 里干活。这份计划也自然成为我熟悉该仓库的入门参考。

常见误区：认为任务简单就跳过 plan mode。实际上，小任务反而更容易出现‘等等，这不是我想要的’这类偏差。

## Sub-agents —— 适合并行跑的任务



![Sub-Agent 拓扑：上下文隔离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig3.png)
*图 2. 父 Agent 启动多个子代理；每个子代理拥有独立的 context window 和工具子集。*

子代理是由父 agent 启动的一个独立 Claude Code 实例，用于处理边界明确的子任务。经典写法放在 `.claude/agents/<name>.md`：

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
- **工具限制。** research agent 确实不具备编辑能力——这是由系统架构保障的安全机制，而非依赖使用者自觉。
- **并行工作。** 任务独立时，你可以同时 fan out 给三个子代理。

代价是什么：

- Tokens。每个子代理都有自己的 system prompt、自己的上下文、自己的来回对话。
- 协调。父 agent 得合并结果。这一步要显式规划好。

什么时候不该用子代理：父 agent 已经拥有所需上下文的任何任务。为‘读取单个文件并汇报’而 spawn 子代理，属于不必要的高开销递归调用。

## Worktrees —— 并行分支且不至于搞疯自己

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_2.png)

git worktree 是同一个 repo 的第二个 working tree，在不同分支，不同目录。Claude Code 认识这东西：`EnterWorktree` 工具会创建新分支 + worktree 并把 session 切换进去。



![Worktree 文件系统布局：共享 .git，独立 working tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig4.png)
*图 3. 多个 worktree 共享同一个 `.git/` 对象库，但各自拥有独立的工作目录，分别绑定到不同分支。*

这些场景很关键：

- 你正在 `feat/x` 上干活，用户突然要求在 `main` 上修个无关的 quick fix。新建一个 worktree，完成修复后提交（commit），再退出。
- 你想尝试同一问题的两种不同解法，又不想用废弃的 commit 弄脏主干。
- 你委托给子代理，希望它在物理上跟你的 working tree 隔离。

心智模型：worktree 是上下文隔离的一种*物理层面实现*：子代理通过会话隔离实现逻辑上下文隔离，worktree 则通过独立工作目录实现文件系统层面的隔离。

关于关于清理方式：

- `keep` — worktree 留在磁盘上。工作未完成或可能回来继续时用。
- `remove` — 彻底删除，分支也删掉。只有确定不再需要时才用。

如果 worktree 里有未 commit 的变更，除非你确认 `discard_changes: true`，否则拒绝删除。这是对的。别想着绕过它。



![三道信任门：自主性与成本逐级递增](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig6.png)
*图 4. 任务值得时再向上爬阶梯。*

## 组合这三者

处理硬任务时我的模式：

1. **Plan mode。** “我要做这个；你打算怎么做？”看计划。调整。
2. **Worktree。** 进一个隔离分支，这样实验失败了也不至于弄脏主干。
3. **Sub-agents** 处理 worktree 内的独立子任务。先 research，再 implementation，最后写 test —— 每个都在自己的上下文里。
4. 回到父 agent，合并结果，commit，退出 worktree（工作暂停用 `keep`，完成了用 `remove`）。

三层信任门，三级 escalation。当模型进入文件修改阶段时，你所投入的注意力程度，恰好与任务复杂度相匹配。



![决策树：plain / plan / worktree / sub-agent 如何选](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig7.png)
*图 5. 决策树：先问最简单的问题，被迫时才升级。*

## 什么时候都不用

大多数任务。说真的。80% 的情况就是“改这个函数，跑测试，发布”—— 普通模式，不要子代理，不要 worktrees。上面那些功能是为那 20% 庞大、不可逆或分支复杂的任务准备的。

如果你发现自己每个任务都想掏子代理和 worktree，更有趣的问题是：你是不是把任务拆得太大了。