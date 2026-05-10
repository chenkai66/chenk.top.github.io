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
Claude Code 现在有四种扩展机制：slash commands、MCP servers、hooks 和 Skills。功能上有重叠。当你冒出"Claude 应该知道怎么做 X"的念头时，关键问题是*选哪一个*。

这是系列的最后一章。直接上决策树。

![Claude Code Hands-On (10): Skills, and When to Reach for Each Extension Mechanism — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_1.png)

## Skill 到底是什么

Skill 就是一个文件夹，位置在 `~/.claude/skills/<name>/`（用户级）或 `<repo>/.claude/skills/<name>/`（项目级）。里面至少得有个 `SKILL.md`：

```markdown
---
name: chenk-blog-write
description: Use when writing new content for chenk.top — bilingual EN/ZH posts, series, tutorials. Covers front matter, voice, matplotlib figures, cover generation, deploy.
---

# Voice
- First person, dry, restrained. No "let's", no exclamations.
- One claim, one example. If the claim has no example, cut the claim.

# Front matter
[exact schema goes here]

# Workflow
1. Read source
2. Write EN
3. Adapt to ZH (not translate)
4. Generate covers
5. Build + deploy
```

会话开始时，Claude 会读取所有可用 skill 的 *descriptions*。当你问的东西匹配上了，Claude 才加载 skill 正文。正文会成为那一轮 system prompt 的一部分。

两点要注意：

- `description` 是承重墙。如果没写清楚什么时候用，skill 就不会被触发。
- 正文可以很长。因为是按需加载，除非触发，否则啰嗦点也不消耗 context。

## Skill 和其他三者的区别

| 机制 | 位置 | 加载时机 | 适合场景 |
|---|---|---|---|
| Slash command | `<repo>/.claude/commands/<name>.md` | 用户输入 `/<name>` | 1-2 行就能说清的重复工作流 |
| MCP server | `mcp.json` 配置 | 一直可用 | 伸手到文件系统之外（浏览器、DB、第三方 API） |
| Hook | `settings.json` 引用的脚本 | 工具调用前后 | 策略 enforcement，edit/write 的副作用 |
| Skill | `.claude/skills/<name>/SKILL.md` | Description 匹配 prompt | 领域知识、风格、多步骤流程 |

最核心的界限：**slash commands 是指令，skills 是知识**。Slash command 是"做这件具体的事"。Skill 是"这类问题我是这么想的，碰到了就用"。

## 什么时候用哪个

![Claude Code Hands-On (10): Skills, and When to Reach for Each Extension Mechanism — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_2.png)

按顺序过一遍：

**1. 任务需要尚不存在的工具吗？**（浏览器自动化、查真实数据库、调内部 API。）
→ 建一个 **MCP server**。其他机制给不了新能力。

**2. 工具调用前后需要自动发生点什么吗？**（拦截、验证、日志、格式化？）
→ 写一个 **hook**。这是唯一不需要模型决定调用就能运行的机制。

**3. 这是个紧凑的流程，用户会显式调用吗？**（`/commit`, `/deploy-staging`, `/make-changelog`）
→ 写一个 **slash command**。命令是用来点名调用的。

**4. 这是一堆领域知识吗？**（风格、工作流、一套规范）
→ 写一个 **skill**。Skill 是用来让 Claude 自己*识别并应用*的，不用你特意喊它。

如果一个东西 fit 两个框，选简单的那个。Skill 调 slash command 没问题。Slash command 装成 skill 会很脆。

## 我实际写过的三个 skills

**1. `chenk-blog-write`** — 给这个站点写文章用的。覆盖 front matter、风格、EN/ZH parity、封面生成、部署。只要提到 chenk.top 或 "write a post" 就触发。正文大概 600 行。每一行都值。

**2. `update-config`** — 用来改 `~/.claude/settings.json` 的。触发词是 "allow X command," "set env Y," "add a hook." 编码了上面的权限优先级规则和典型模式。省得我去推 merge order。

**3. `simplify`** — 用来 review 我自己代码的。触发词是 "is there a simpler way to do this." 编码了我的品味：偏好组合，删死代码，命名看本质不看实现。

这几个都没法做成 slash command。不是靠名字调用，是靠*话题*调用。这才是 Skill 该用的地方。

## 什么时候 Skill 不是正确答案

触发太频繁的 skill 比没有更糟——它会污染那些不需要它的任务的 context。三个坑：

- **描述模糊。** "用于一般编程。" 用于一切 = 用于 nothing useful。
- **正文重叠。** 两个 skill 都响应 "write code" → context 膨胀。选一个。
- **本该是 Hook。** "总是在 Y 之前做 X" → 这是 hook，不是 skill。Skill 是建议，Hook 是强制。

## 系列结束

十章下来，你有了：

- 一个配置好的 Claude Code，脑子里有三层设置模型 (第 1、9 章)。
- 熟练使用 shortcuts、modes 和对话控制 (第 2 章)。
- 四种扩展机制 — slash commands, MCP, hooks, skills — 以及如何在它们之间选择的决策树 (第 3、4、5、7、10 章)。
- 并发原语 — sub-agents, worktrees, plan mode — 把单个 session 扩展到更大的工作 (第 8 章)。
- 一套能用的 SDK + GitHub 集成方案，把 Claude 放进 CI (第 6 章)。

表面功夫就这些。后面真正有趣的工作，不再关于 Claude Code 本身——而是你用它*构建*什么。

去构建吧。