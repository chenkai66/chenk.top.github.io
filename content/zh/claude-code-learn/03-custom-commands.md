---
title: "Claude Code 实战（三）：自定义 Slash 命令"
date: 2026-04-20 09:00:00
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
像 `/clear` 和 `/init` 这种内置斜杠命令，不过是冰山露出水面的一角。这系统的核心在于你自己写命令，而且直接放在仓库里。

![Claude Code Hands-On (3): Custom Slash Commands and Conversation Control — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_1.png)

## 斜杠命令到底是什么

文件放在 `.claude/commands/<name>.md`。内容就是 Markdown 提示词。文件名即命令名。建好后得重启 Claude Code（这是少数不支持热重载的地方）。

举个最简单的例子。创建 `.claude/commands/audit.md`：

```markdown
Run `npm audit` to find vulnerable installed packages.
Run `npm audit fix` to apply non-breaking fixes.
Run `npm test` to confirm nothing broke.
Report which CVEs were patched and which remain.
```

重启后，在任何会话里：

```
/audit
```

整个提示词瞬间触发。你拿到一份结构化的审计报告，再也不用死记那三个命令及其顺序。

两点要注意：

1. 命令本质上就是提示词。没有 DSL，没有特殊语法。表面积极小。
2. 不用重复造轮子。下次团队里谁需要审计，敲个 `/audit` 就行。

## `$ARGUMENTS` —— 参数化

![Claude Code Hands-On (3): Custom Slash Commands and Conversation Control — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_2.png)

斜杠命令有个魔法 token `$ARGUMENTS`，会被替换成你命令后面敲的内容。举个例子 —— `.claude/commands/explain.md`：

```markdown
Explain $ARGUMENTS at three levels:

1. One sentence — what is it, in plain language.
2. One paragraph — how it works, key components, why it exists.
3. Code-level — point to where this is implemented in our repo, with line numbers.

If the term is ambiguous, list the meanings and ask which one I want.
```

然后：

```
/explain rate limiter
```

`$ARGUMENTS` 变成了 `rate limiter`，提示词触发，你拿到一份基于仓库实际代码的三层解释。

## 我每个项目必存的命令

用了两年 Claude Code，我沉淀下来一套固定的命令组合：

**`/audit`** —— 安全审计，同上。

**`/test`** —— 跑测试套件，总结失败项，建议修复方案。

```markdown
Run the project's test suite (see CLAUDE.md for the test command).
For each failure:
  - quote the failing test name and the assertion that failed
  - propose the smallest plausible fix
  - mark whether you'd patch the test or the code
Do not make changes — this is a report.
```

**`/review`** —— 代码 Diff 审查。

```markdown
Review the staged diff (`git diff --staged`).
Focus on: correctness, edge cases, naming, and adherence to CLAUDE.md.
Output as numbered findings, each with severity (must-fix / nit / praise).
End with one paragraph summarizing whether you'd approve.
```

**`/explain $ARGUMENTS`** —— 见上文。

**`/onboard`** —— 为新加入代码库的人生成一页简报。

```markdown
Write a one-page onboarding doc for a new engineer joining this repo.
Use CLAUDE.md as the source of truth.
Include: what it does, how to set up, how to run tests, the three things
they're most likely to break, and where to look first.
```

五个命令。加起来覆盖了我平时大部分得手工敲的东西。

## 对话控制 —— 这三个你得肌肉记忆

内置命令里值得形成肌肉记忆的有三个：

**`/compact`** —— 总结当前对话。模型变慢时用。保留主旨，去掉冗余。

**`/clear`** —— 清空对话。保留记忆和设置。切换任务时用。

**`/init`** —— 上一篇讲过。每个仓库跑一次，引导生成 `CLAUDE.md`。

还有更多 —— `/help` 全列出来了 —— 但这三个是日常必备。

## 关于团队落地

斜杠命令是团队内推广规范成本最低的方式。往 `main` 分支的 `.claude/commands/` 丢三个好用的命令。下次谁在这个 repo 跑 `claude`，直接就有了。用户无需配置。这才是重点。

我见过这种模式对团队 AI 使用的推动作用，比任何内部培训都大。人们会复制现成的东西，而 `.claude/commands/` 就是个装满“有用到值得提交”的文件的目录。

## 它不适合干什么

斜杠命令搞不定这些：

- 任何需要比单字符串更复杂的运行时参数的事
- 任何需要在调用间维持状态的事
- 任何你需要测试的事

这些情况得用 SDK（第 6 篇）。斜杠命令只适合那些不值得写代码复杂度的工作流捷径。

下一篇：MCP —— 让 Claude Code 能跟任何东西对话的协议。