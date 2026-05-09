---
title: "Claude Code 实战入门（三）：自定义斜杠命令与对话控制"
date: 2026-04-18 09:00:00
tags:
  - claude-code
  - slash-commands
  - workflow
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 3
description: "斜杠命令把重复工作流变成一行调用。$ARGUMENTS 让它们参数化。挑得对，它们会变成你团队的共享词汇。"
disableNunjucks: true
translationKey: "claude-code-learn-3"
---
内置的斜杠命令，比如 `/clear` 和 `/init`，只是冰山一角。这套系统的核心思想是让你自己编写命令，并且这些命令会存放在你的代码仓库中。
![Claude Code 实战入门（三）：自定义斜杠命令与对话控制 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_1.jpg)

## 什么是斜杠命令

斜杠命令其实就是一个 Markdown 文件，存放在 `.claude/commands/<name>.md` 路径下。文件内容是一段 Markdown 格式的提示词（Prompt），而文件名会直接成为命令的名称。创建文件后需要重启 Claude Code 才能生效——这是少数不支持热加载的地方。

举个最简单的例子。创建一个文件 `.claude/commands/audit.md`，内容如下：

```markdown
运行 `npm audit` 检查存在漏洞的依赖包。
运行 `npm audit fix` 自动修复不会破坏代码的漏洞。
运行 `npm test` 确保修复没有引入新的问题。
最后，报告哪些 CVE 漏洞已被修复，哪些仍然存在。
```

保存文件并重启后，在任意会话中输入以下命令：

```
/audit
```

整个提示词就会被触发，系统会自动生成一份结构化的审计报告。你不用再费心记住那三条命令以及它们的执行顺序。

这里有两点值得关注：

1. 斜杠命令本质上就是一段 Prompt，没有任何复杂的 DSL 或特殊语法。这种设计刻意保持了功能的简洁性。
2. 不用重复劳动。下次团队里任何人需要做审计时，只需输入 `/audit` 就能完成任务。
## `$ARGUMENTS`——参数化

![Claude Code 实战入门（三）：自定义斜杠命令与对话控制 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_2.jpg)

斜杠命令自带一个特殊的 `$ARGUMENTS`，它会自动替换为你在命令后面输入的内容。举个例子——`.claude/commands/explain.md`：

```markdown
解释 $ARGUMENTS，分三个层次：

1. 一句话概括——这是什么？用大白话说明。
2. 一段话展开——它是如何工作的？核心组件有哪些？为什么需要它？
3. 代码层面——指出它在我们代码库中的具体实现位置，并附上行号。

如果这个词有歧义，列出可能的含义，然后问我具体指的是哪一个。
```

接着，运行以下命令：

```
/explain rate limiter
```

这时，`$ARGUMENTS` 被替换为 `rate limiter`，提示触发后，你会得到基于实际代码库的三层详细解释。
## 每个项目我都离不开的命令

用了两年 Claude Code，我最终固定了几个命令：

**`/audit`**——执行安全审计。

**`/test`**——运行测试套件，汇总失败情况，并给出修复建议。

```markdown
运行项目的测试套件（具体命令参考 CLAUDE.md）。
针对每个失败：
  - 引用失败的测试名称和断言内容
  - 提出最小可行的修复方案
  - 标明是修改测试还是代码
不要做任何实际改动——这是报告。
```

**`/review`**——对代码差异进行审查。

```markdown
审查暂存区的差异（`git diff --staged`）。
重点关注：正确性、边界条件、命名规范以及是否符合 CLAUDE.md。
输出编号形式的审查结果，每条标注严重程度（must-fix / nit / praise）。
最后写一段总结，说明是否会批准。
```

**`/explain $ARGUMENTS`**——解释功能。

**`/onboard`**——为新人生成一份简短的项目入门指南。

```markdown
为新加入此仓库的工程师编写一页入门文档。
以 CLAUDE.md 为准。
内容包括：项目功能、如何配置环境、如何运行测试、最容易踩坑的三件事，以及应该先看哪里。
```

总共五个命令。它们几乎涵盖了我每天需要手动输入的所有操作。
## 对话控制——必须掌握的三个命令

有几个内置命令非常实用，建议直接形成肌肉记忆：

**`/compact`** —— 压缩当前对话。模型因为上下文太长变慢时用它。保留重点，去掉冗余内容。

**`/clear`** —— 清空对话记录。Memory 和 settings 会保留。切换任务时很有用。

**`/init`** —— 上一篇文章提到过。每个代码库运行一次，用来初始化 `CLAUDE.md`。

更多命令可以用 `/help` 查看，但日常工作中最常用的还是这三个。
## 关于团队推广

斜杠命令是让整个团队快速接受某种规范的最简单方法。在 `main` 分支上，把三条实用的命令放进 `.claude/commands/` 目录。下次有人在这个代码库运行 `claude` 时，这些命令就自动生效了。用户完全不用做任何配置，这正是它的优势。

我发现，这种模式对推动团队整体使用 AI 的效果，比任何内部培训都好。大家会直接复用已经存在的内容，而 `.claude/commands/` 就是一个专门存放“被认为足够有用才被提交”的内容的目录。
## 它不适合做什么

斜杠命令有明显的局限性：

- 处理运行时参数如果比单个字符串复杂，就搞不定了
- 无法在多次调用之间保持状态
- 想要测试的功能也很难实现

遇到这些情况，就得用 SDK（第 6 部分）。斜杠命令适合的是那些不值得花精力写代码的简单工作流快捷操作。

接下来讲讲 MCP——这个协议能让 Claude Code 和任何系统进行交互。
