---
title: "Claude Code 实战入门（八）：子 Agent、worktree、计划模式"
date: 2026-04-23 09:00:00
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
学完 hooks 之后，接下来影响 Claude Code 使用体验的就是**并发控制**。这里说的不是线程层面的并发，而是指“模型同时在帮我做几件事、隔离程度如何、我能掌控多少”这种意义上的并发。

三个功能特性，按信任需求从低到高排列。
![Claude Code 实战入门（八）：子 Agent、worktree、计划模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_1.jpg)

## 计划模式——气闸

计划模式成本最低。按 `Shift+Tab`，直到指示器显示 **plan**。这时模型只会规划，不会执行任何操作：读取信息、思考、提出方案，然后停下来。我看完计划后，可以选择批准、修改或者直接放弃。只有在我确认之后，它才会开始执行。

我一般在这些情况下使用计划模式：

- 开始任何非简单任务的前 30 秒。比如“实现 X 功能”，先让模型生成计划。几乎每次，计划都会暴露出模型对代码库的理解有偏差。
- 涉及到鉴权、支付、schema 迁移或者生产环境配置的任务。花两秒钟看计划，能省下数小时的善后工作。
- 在我不熟悉的代码仓库中工作时。计划本身就成了我的入门文档。

常见错误：因为“任务很小”就跳过计划模式。实际上，小任务最容易出现“等等，这不是我想要的”这种情况。
## 子 Agent——适合并行处理的任务

子 Agent 是父 Agent 创建的一个 Claude Code 实例，专门用来完成某个特定范围的任务。它的经典定义文件存放在 `.claude/agents/<name>.md` 中：

```markdown
---
name: research
description: 跨代码库阅读某个主题并生成报告。不进行任何编辑操作。
tools: Read, Grep, Glob, WebFetch
---

你是一个调研子 Agent。你的任务如下：
1. 在代码库中搜索目标主题。
2. 阅读足够多的文件，深入理解这个主题。
3. 返回一份结构化的报告，包含文件路径和关键引用。

不要修改文件。不要运行 shell 命令。保持专注。
```

在对话中可以这样用："研究一下鉴权是如何实现的，调用 research agent。"

使用子 Agent 的好处：

- **上下文隔离**：子 Agent 有自己的上下文窗口，不会干扰父 Agent 的上下文。
- **工具限制**：research agent 根本无法修改文件。这种安全性是架构层面的设计，而不是靠人为约束。
- **并行处理**：当任务相互独立时，可以同时启动三个子 Agent 来分头工作。

使用子 Agent 的代价：

- Token 消耗：每个子 Agent 都有自己的系统提示、上下文以及交互过程。
- 协调成本：父 Agent 需要合并各个子 Agent 的结果，这一步必须提前规划好。

什么时候不该用子 Agent：如果父 Agent 已经掌握了所需的上下文，就没必要再创建子 Agent。比如让子 Agent "去读一个文件然后汇报"，这就是一种昂贵的递归操作。
## Worktrees — 并行分支不混乱

![Claude Code 实战入门（八）：子 Agent、worktree、计划模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_2.jpg)

git worktree 是同一个仓库的第二个工作树，位于不同的分支和目录。Claude Code 支持这个功能：`EnterWorktree` 工具可以创建一个新分支和对应的工作树，并将当前会话切换过去。

什么时候需要它：

- 我正在 `feat/x` 上开发功能，用户突然要求在 `main` 分支上修复一个无关的小问题。这时可以启动一个 worktree，完成修复后提交并退出。
- 我想尝试两种不同的解决方案，但又不想让废弃的提交污染主分支。
- 我需要把任务交给子代理（sub-agent），并且希望它与我的工作树完全隔离。

如何理解它：worktree 是上下文隔离的**物理实现**。子代理负责隔离上下文，而 worktree 负责隔离文件系统。

退出时的选择：

- `keep` — worktree 保留在磁盘上。适合任务只完成了一部分、可能需要回头继续的情况。
- `remove` — 彻底删除，连分支一起删掉。只有确定任务完全结束时才用。

如果 worktree 中有未提交的改动，删除操作会被拒绝，除非我明确设置 `discard_changes: true`。这是合理的。不要试图绕过这个限制。
## 三者组合

我处理复杂任务时的套路：

1. **计划模式**。我会问自己：“目标是这个，你会怎么下手？” 看一遍计划，调整一下。
2. **worktree**。切到一个隔离分支，实验失败了也不会影响主干。
3. 在 worktree 里分配子任务，交给不同的子 Agent。先调研，再实现，最后写测试，每个任务独立运行。
4. 回到主分支，合并结果，提交代码，退出 worktree（如果任务暂停用 `keep`，完成就用 `remove`）。

三道信任关卡，三次逐步深入。等到模型真正开始修改文件时，我已经精准投入了任务所需的注意力。
## 什么时候都不用

大部分任务真的不用。80% 的场景就是“改函数、跑测试、提交”——普通模式，不开子 Agent，也不开 worktree。那些高级功能是留给那 20% 的大活儿的，比如改动大、不可逆或者需要分支的情况。

如果每个任务都想着用子 Agent 和 worktree，我建议先想想：**你的任务是不是拆得太粗了？**
