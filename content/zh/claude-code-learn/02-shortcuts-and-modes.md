---
title: "Claude Code 实战（二）：快捷键与 Thinking Mode"
date: 2026-04-19 09:00:00
tags:
  - claude-code
  - shortcuts
  - thinking-modes
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 2
description: "Shift+Tab 不是开关而是四态循环。Thinking Mode 有 5 档。Escape 和双击 Escape 做的事不同。五个会重塑日常手感的快捷键。"
disableNunjucks: true
translationKey: "claude-code-learn-2"
---
快捷键没放在帮助屏幕里是有原因的——它们是靠用出来的，不是靠查文档出来的。不过既然你们问了，我还是列出来。

![Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/02-shortcuts-and-modes/illustration_1.png)

## `Shift+Tab` — 四态循环

大多数人以为 `Shift+Tab` 只是开关自动接受。其实不是。它是四个状态按顺序循环：

```
Normal  →  Accept edits  →  Plan mode  →  Bypass permissions  →  Normal
```

每个状态决定了 Claude 在不询问的情况下能做什么：

| State | What changes |
|-------|--------------|
| **Normal** | 默认。写文件和跑 shell 命令前会询问。 |
| **Accept edits** | 自动接受文件修改。shell 命令还是会问。 |
| **Plan mode** | 只生成计划不执行。适合在看副作用前先审视方案。 |
| **Bypass permissions** | 跳过所有确认提示。只在信任当前任务时用。 |

状态栏会显示当前处在哪个状态。`Shift+Tab` 负责循环切换。没有哪个状态是我用得最多的——它们对应不同的任务。刚开始改未知代码时我用 Plan mode。长时间重构且我会盯着输出时我用 Accept edits。跑已知范围的脚本时我会用 Bypass permissions——但绝不盲用。

## 思考模式 — 五个层级

![Claude Code Hands-On (2): Shortcuts, the Four-State Toggle, and Thinking Modes — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/02-shortcuts-and-modes/illustration_2.png)

在 prompt 里输入这些短语，Claude 就会调整回应前的推理力度：

| Phrase | Approximate effort |
|--------|-------------------|
| `think` | 轻度额外推理 |
| `think more` | 多花几秒 |
| `think a lot` | 明显变慢，深度更深 |
| `think longer` | 适合架构决策 |
| `ultrathink` | 最大值 |

成本随深度增加——每级消耗的 token 更多。选哪级取决于任务：

- "Fix this typo" — 不用加思考短语
- "Find why this test is flaky" — `think more`
- "Refactor this module" — `think a lot`
- "Should we move this from REST to gRPC" — `ultrathink`

只要是“非 trivial"的任务，我默认都用 `think a lot`。低于这个级别直接输 prompt。高于这个级别，意味着我在问一个不同的问题，`ultrathink` 对此很诚实。

陷阱是因为 `ultrathink` 听起来严谨就在每个 prompt 都用。并不是——它很贵。模型在匹配实际任务复杂度的级别上表现最好。

## `Control+V` — 粘贴图片

直接把图片粘贴到 prompt 里。UI 截图、错误弹窗、设计稿、白板照片——Claude 像读文本一样读它。最有用的场景是“测试挂了，输出在这张终端截图里，哪错了？”。不用手动转录。

## `Escape` 和双按 `Escape`

单按 `Escape` 中断当前生成。一旦发现指令给错了，马上用。Agent 停下，你修正。

双按 `Escape`（快速按两次）打开最近几次交互的历史视图。选一个节点“回滚”，对话就从那里分叉。选中节点之后的内容会从 context 中丢弃。

这是我用其他 coding agent 时最想念的功能。对话分叉才是正确的原语——它让你试一条路，发现错了，回去重来，而不污染 Agent 的记忆。

## `/compact` 和 `/clear`

处理长对话的两种方法：

- `/compact` 把目前的对话总结成更短的消息继续。当 context 太大导致 Agent 变慢时用。
- `/clear` 直接丢弃对话重新开始。当你在同一项目里切换到无关任务时用。

两者都保留 `CLAUDE.md` 和 `.claude/settings.json` 在 scope 内。只影响单次会话的消息历史。

## 实际工作流长什么样

今天早上我的实际工作流：

```
> @src/api/handlers.ts
> think a lot — the team wants to add idempotency keys to the
> three POST handlers in this file. propose an approach that
> reuses the existing middleware pattern.

[Claude proposes — I read it]

[Shift+Tab to plan mode]

> ok, write it out as a plan, don't change anything yet.

[Claude lays out the steps]

[Shift+Tab to accept edits]

> go.

[Claude makes the changes, I watch them scroll]

[Shift+Tab back to normal mode for the test run]

> run the tests.

[Claude pauses to ask — yes — runs them — three pass, one fails]

[Double-Escape to fork back to before the test run]

> the failing test is testing the old non-idempotent path.
> update it to verify the new behavior.
```

整个流程用到了本文提到的所有原语。单看每一个都不算惊艳。合在一起，Claude Code 才像个真正的工具，而不是挂在聊天窗口上的自动补全。

下一篇：自定义 slash 命令和 `$ARGUMENTS`。