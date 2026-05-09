---
title: "Claude Code 实战入门（十）：Skills，以及四种扩展机制各自该用在哪儿"
date: 2026-04-25 09:00:00
tags:
  - claude-code
  - skills
  - slash-commands
  - mcp
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 10
description: "Claude Code 现在有四种扩展机制：斜杠命令、MCP server、Hooks、Skills。Skill 是最新的——一个文件夹、一份 SKILL.md、一段按需加载的指令。它和其他三个怎么分工，给一棵决策树。"
disableNunjucks: true
translationKey: "claude-code-learn-10"
---
Claude Code 目前支持四种扩展机制：斜杠命令、MCP 服务器、Hooks 和 Skills。这些机制之间存在功能重叠。当我第一次想到“Claude 应该学会做 X”时，第一个问题就是：这四种机制中，我该选哪一个？

这是本系列的最后一篇文章。接下来，我会详细展开决策树。
![Claude Code 实战入门（十）：Skills，以及四种扩展机制各自该用在哪儿 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_1.png)

## Skill 到底是什么

Skill 是存放在 `~/.claude/skills/<name>/`（用户级）或者 `<repo>/.claude/skills/<name>/`（项目级）下的一个文件夹，至少要包含一个 `SKILL.md` 文件：

```markdown
---
name: chenk-blog-write
description: 为 chenk.top 撰写新内容时使用——双语 EN/ZH 文章、系列、教程。包括 front matter、语气风格、matplotlib 图表、封面生成、部署。
---

# 语气
- 第一人称，简洁克制。不要用 "let's"，也不要加感叹号。
- 一个观点配一个例子。如果没有例子，就删掉这个观点。

# Front matter
[具体 schema 放在这里]

# 流程
1. 阅读源材料
2. 写英文内容
3. 转换为中文（不是直译）
4. 生成封面
5. 构建并部署
```

每次会话开始时，Claude 会读取所有可用 Skill 的 **description**。如果你的提问匹配到某个 Skill，Claude 就会加载它的主体内容。这部分内容会直接加入当轮的 system prompt。

这里有两点需要注意：

- **description 很关键**。如果它没写清楚什么时候用这个 Skill，那这个 Skill 就不会被触发。
- **主体内容可以很长**。它是按需加载的，只有在触发时才会占用 token，平时不用担心长度问题。
## 技能与其他三种机制的区别

| 机制 | 存储位置 | 加载时机 | 最适合的场景 |
|---|---|---|---|
| 斜杠命令 | `<repo>/.claude/commands/<name>.md` | 用户输入 `/<name>` | 只需 1-2 行描述的重复工作流 |
| MCP server | `mcp.json` 配置文件 | 始终加载 | 跨越文件系统边界（如浏览器、数据库、第三方 API） |
| Hook | `settings.json` 引用的脚本 | 工具调用前后触发 | 策略执行、编辑或写入时的附加操作 |
| Skill | `.claude/skills/<name>/SKILL.md` | 描述匹配用户提示时 | 领域知识、语气风格、多步骤流程 |

最明显的区别在于：**斜杠命令是操作指令，而技能是知识库。** 斜杠命令的意思是“执行这个具体的操作”，而技能则是“这是我解决这类问题的思路，只要相关就尽管用”。
## 什么时候该用哪个

![Claude Code 实战入门（十）：Skills，以及四种扩展机制各自该用在哪儿 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_2.png)

按顺序过一遍：

**1. 这个任务需要一个还没出现的工具吗？** 比如浏览器自动化、查询真实数据库、调用内部 API。  
→ 写一个 **MCP server**。其他方法没法提供新能力。

**2. 需不需要在工具调用时自动做点什么？** 比如拦截、校验、记录日志或者格式化。  
→ 写一个 **hook**。这是唯一一种模型不主动调用也会执行的机制。

**3. 是不是一个用户会明确调用的小功能？** 比如 `/commit`、`/deploy-staging`、`/make-changelog`。  
→ 写一个 **斜杠命令**。命令就是那种你喊名字才会跑的东西。

**4. 是不是一组领域知识？** 比如某种语气、一套流程或者一些约定，希望话题一提到就自动生效。  
→ 写一个 **skill**。Skill 就是那些我希望 Claude 能自己 **认出来并应用** 的东西，不用我明说。

如果一件事能归到两类，选简单的那个。Skill 调用斜杠命令没问题，但斜杠命令假装自己是 Skill 就容易出问题。
## 我写过的三个 Skill

**1. `chenk-blog-write`** — 用来给本站写文章。包括 front matter、语气风格、中英文一致性、封面生成和部署。只要提到 chenk.top 或者说“写一篇”，就会触发。代码主体大约 600 行，每一行都值得。

**2. `update-config`** — 用来修改 `~/.claude/settings.json`。像“允许 X 命令”、“设置环境变量 Y”、“加个 hook”这样的请求会触发它。里面编码了权限优先级规则和常见模式，省得我每次都重新推导合并顺序。

**3. `simplify`** — 用来审查我自己写的代码改动。如果问“有没有更简单的实现方式”，它就会触发。我的代码风格是这样的：优先用组合，删掉无用代码，命名要看它**是什么**，而不是看它**怎么实现的**。

这三个 Skill 都没法用斜杠命令调用。它们不是靠名字触发，而是靠**话题**触发。这就是典型的 skill 形状用例。
## 技能何时不是正确答案

一个触发过于频繁的技能，比没有技能更糟糕——它会干扰那些不需要它的任务。以下是三个常见问题：

- **描述太模糊。** “适用于通用编程。” 什么都适用 = 什么都不适用。
- **技能主体重复。** 两个技能都在“写代码”时触发 → 导致上下文混乱。选一个就行。
- **本该用 Hook 却用了 Skill。** “做 Y 前必须完成 X” → 这是 Hook，不是 Skill。Skill 提供建议，Hook 强制执行。
## 系列的尾声

十篇文章读完，你已经掌握了以下内容：

- 配置好的 Claude Code，脑子里有了三层 settings 模型（第 1、9 篇）。
- 熟练使用快捷键、模式和对话控制（第 2 篇）。
- 四种扩展机制——斜杠命令、MCP、Hooks、Skills——以及如何选择它们的决策树（第 3、4、5、7、10 篇）。
- 并发原语——子 Agent、worktree、计划模式——能够将单个会话扩展到更大规模任务的能力（第 8 篇）。
- 可用的 SDK 和 GitHub 集成方案，把 Claude 嵌入 CI 流程（第 6 篇）。

这些只是基础。真正有趣的部分不在 Claude Code 本身，而在于你能用它**创造什么**。去动手实践吧。
