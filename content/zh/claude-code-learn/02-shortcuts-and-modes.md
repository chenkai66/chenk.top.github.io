---
title: "Claude Code 实战入门（二）：快捷键、四态切换、Thinking Mode"
date: 2026-04-17 09:00:00
tags:
  - claude-code
  - 快捷键
  - thinking-mode
categories: Claude Code
lang: zh-CN
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 2
description: "Shift+Tab 不是开关而是四态循环。Thinking Mode 有 5 档。Escape 和双击 Escape 做的事不同。五个会重塑日常手感的快捷键。"
disableNunjucks: true
translationKey: "claude-code-learn-2"
---
这些快捷键没出现在帮助界面是有原因的——要用的时候自然会发现，而不是靠文档说明。不过我还是写一下吧。
![Claude Code 实战入门（二）：快捷键、四态切换、Thinking Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/02-shortcuts-and-modes/illustration_1.jpg)

## `Shift+Tab`——四态循环

很多人以为 `Shift+Tab` 只是用来开关 auto-accept 的。其实不是，它会按顺序在四个状态之间循环：

```
Normal → Accept edits → Plan mode → Bypass permissions → Normal
```

每个状态都会改变 Claude 的行为，而且不需要额外确认：

| 状态 | 改变 |
|------|------|
| **Normal** | 默认状态。写文件和执行 shell 命令前都会先询问。 |
| **Accept edits** | 自动接受文件修改，但执行 shell 命令时仍会询问。 |
| **Plan mode** | 只生成计划，不实际执行。适合在产生副作用之前先检查方案是否合理。 |
| **Bypass permissions** | 跳过所有确认提示。只有在完全信任当前任务时才用这个状态。 |

状态栏会显示当前处于哪个状态。按 `Shift+Tab` 就能切换。我不会固定使用某个状态，因为每个状态对应不同的场景。开始一个陌生改动时，我会用 Plan mode；进行长时间重构时，我会用 Accept edits，毕竟我本来就在盯着输出看；运行已知且范围明确的脚本时，我会选择 Bypass permissions，但绝不会盲目使用。
## 思考模式——五个级别

![Claude Code 实战入门（二）：快捷键、四态切换、Thinking Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/02-shortcuts-and-modes/illustration_2.jpg)

在 Prompt 的任意位置输入以下短语之一，Claude 会调整回答前的推理深度：

| 短语 | 大致耗时 |
|------|----------|
| `think` | 轻微额外推理 |
| `think more` | 再多花几秒 |
| `think a lot` | 明显变慢，但更深入 |
| `think longer` | 适合架构决策 |
| `ultrathink` | 最高推理级别 |

推理深度越高，成本越高——每级消耗更多 Token。选择哪一级取决于具体任务：

- "修复这个拼写错误"——直接写就行，不用加思考指令  
- "找出这个测试不稳定的原因"——用 `think more`  
- "重构这个模块"——用 `think a lot`  
- "是否应该从 REST 切换到 gRPC"——用 `ultrathink`  

我默认对所有“非简单”任务都用 `think a lot`。比这简单的任务，我直接写 Prompt；更复杂的任务，其实是在问一个完全不同的问题，而 `ultrathink` 对此很坦诚。

常见误区是觉得 `ultrathink` 听起来严谨，就每个 Prompt 都用它。其实不然——它只是更贵。模型在与任务复杂度匹配的级别上表现最佳。
## `Control+V`——粘贴图片

直接把图片粘贴到 Prompt 里。无论是 UI 截图、错误弹窗、设计稿，还是白板照片，Claude 都能像读文字一样读懂。最实用的场景是："测试失败了，输出结果在这张终端截图里，问题出在哪？" 这样一来，我完全不用手动转录内容。
## `Escape` 和双击 `Escape`

按一下 `Escape`，可以中断当前的生成。只要发现指令给错了，马上按就行。Agent 会停下，然后我来修正。

快速按两下 `Escape`，也就是双击，会打开最近几轮对话的历史视图。选中某一轮，对话就会从那里开始分叉。选中轮次之后的内容会被移出上下文。

用其他编码助手时，我最想念的就是这个功能。对话分叉才是正确的基础操作——试了一条路，发现不对，就可以回头，完全不会影响助手的记忆。
## `/compact` 和 `/clear`

处理长对话有两种方法：

- `/compact` 会把当前对话总结成一段简短的消息，然后接着继续。如果上下文太大导致 Agent 变慢，就用这个命令。
- `/clear` 会直接清空当前对话，从头开始。如果你在同一个项目里切换到不相关的任务，可以用它。

这两个命令都会保留 `CLAUDE.md` 和 `.claude/settings.json` 文件的内容，只会影响当前会话的消息记录。
## 实际工作流的样子

今天早上我经历了一个真实的工作流程：

```
> @src/api/handlers.ts
> 仔细思考——团队想给这个文件里的 3 个 POST 处理函数加上幂等键。
> 提出一个方案，复用现有的 middleware 模式。

[Claude 提出方案——我看了下]

[按 Shift+Tab 切到计划模式]

> 好的，把方案写成计划，先别改动代码。

[Claude 列出步骤]

[按 Shift+Tab 切到接受编辑模式]

> 开始。

[Claude 修改代码，我看着代码滚动]

[按 Shift+Tab 切回普通模式，准备运行测试]

> 运行测试。

[Claude 停下来确认——是的——运行测试——三个通过，一个失败]

[按两次 Escape 回滚到测试运行之前]

> 那个失败的测试针对的是旧的非幂等路径。
> 修改它，让它验证新的行为。
```

整个流程用到了这篇文章提到的所有基础操作。单独看每个操作都很普通，但合在一起，Claude Code 就像一个真正的工具，而不是在聊天界面上加了个自动补全功能。

下一篇：自定义斜杠命令与 `$ARGUMENTS`。
