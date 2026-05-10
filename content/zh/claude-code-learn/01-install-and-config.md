---
title: "Claude Code 实战入门（一）：安装、三层配置、以及 # @ /init 这一组"
date: 2026-04-16 09:00:00
tags:
  - claude-code
  - configuration
  - settings-json
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 1
description: "把 Claude Code 装好、理解 settings.json 的三层覆盖、学会三个低调但极强的原语：# 写入上下文、@ 引用文件、/init 生成项目记忆文件。"
disableNunjucks: true
translationKey: "claude-code-learn-1"
---
这是 Claude Code 实战指南六部曲的第一篇。顺序是我精心安排的，每一篇都是下一篇的基础。读完这一系列，你能用上 90% 的用户从未碰过的六个功能。

![Claude Code 实战 (1)：安装、三层配置与 # @ /init  trio — 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/01-install-and-config/illustration_1.png)

## 安装

目前官方支持的安装路径只有一条，而且就是这条最稳妥：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

这会把 `claude` 二进制文件装到 `~/.local/bin/`（Linux/Mac）。把它加进 PATH，然后：

```bash
claude --version
# claude-code 0.2.x
claude login
```

登录流程会弹浏览器，授权 CLI 后就搞定了。Auth token 存在 `~/.claude/auth.json` 里。`~/.zshrc` 里不会留 API key 导致泄露——这点值得提，因为大多数"AI CLI"工具都在这栽跟头。

## 跑起来

进一个真实项目目录，跑：

```bash
claude
```

你会看到交互式提示符。试一次往返确认安装没问题：

```
> what's in this directory?
```

Claude 会用自己的工具 `ls`，总结然后回复。成了的话，安装就算完毕。

## 三层配置

![Claude Code 实战 (1)：安装、三层配置与 # @ /init  trio — 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/01-install-and-config/illustration_2.png)

这部分大多数用户从来不看。Claude Code 会合并三个位置的配置，优先级依次升高：

| 层级 | 路径 | 进 Git 吗？ | 用途 |
|-------|------|-----------------|----------|
| 机器层 (Machine) | `~/.claude/settings.json` | 否 | 个人全局偏好 |
| 项目层 (Project) | `<repo>/.claude/settings.json` | 是 | 团队共享的项目规范 |
| 本地层 (Local) | `<repo>/.claude/settings.local.json` | 否 (gitignored) | 个人对该 repo 的私有覆盖 |

为什么是三层而不是两层？因为有些设置是个人全局的（默认模型、编辑器命令），有些是团队共享的（项目用的 linter、测试命令），还有些是个人但限定项目的（比如你在 staging 服务器上的 API key）。

具体例子。我机器层面的 `~/.claude/settings.json`：

```json
{
  "model": "claude-opus-4-7",
  "editor": "code"
}
```

项目的 `.claude/settings.json`（已提交）：

```json
{
  "permissions": {
    "allow": ["Bash(npm test)", "Bash(npm run lint)"],
    "deny": ["Bash(rm -rf *)"]
  },
  "tools": {
    "test_command": "npm test"
  }
}
```

队友的 `.claude/settings.local.json`（被 gitignore，只在他本地磁盘）：

```json
{
  "model": "claude-sonnet-4-7",
  "env": {
    "STAGING_API_KEY": "sk-..."
  }
}
```

合并顺序意味着队友继承了项目的权限规范和测试命令，但用的是他偏好的模型和他自己的 staging key。这听起来理所当然，但我见过整个团队部署时在这上面翻车。

## `#` — 写入上下文

行首打 `#`，这行内容会追加到项目的记忆文件（`CLAUDE.md`）里，而不是作为 prompt 发送。例子：

```
# When working on this repo, never use yarn — npm only.
```

Claude 不会回复。它会打开编辑器编辑 `CLAUDE.md`，把这行丢进去，保存。下次对话开始时，这行就在上下文里了。

这就是我避免反复解释偏好的方法。“写 Python 时用类型提示。”“提交信息别加 emoji。”"CI 是 GitHub Actions 不是 GitLab。”每一条都是一行 `#`，永久生效。

## `@` — 引用文件

打 `@` 会弹出模糊文件选择器。选个文件，它会附着在下一条消息上：

```
@src/router.ts
explain how the route matcher handles trailing slashes
```

文件是作为 tool result 发送的，不是粘贴进你的 prompt。这意味着它扣的是工具配额，不是你的输入字数。实际上：你可以附着个 2000 行的文件，而不会让 prompt 变得不可读。

## `/init` — 引导项目记忆

每个 repo 跑一次 `/init`。Claude 会读取代码库，写一个 `CLAUDE.md` 总结：

- 项目是做什么的
- 使用的语言和框架
- 构建、测试、lint 命令
- 主要目录及其用途
- Claude 能检测到的任何规范（提交信息风格、测试命名等）

然后你编辑它。重点不是 Claude 的草稿有多完美——并不完美。重点是你有了一个起步结构，耗时 30 秒而不是 30 分钟。

生成的 `CLAUDE.md` 提交到 repo。每个队友的 Claude Code 会话都从这里开始。这就是项目共享心智模型的方式。

## 我在每个新 repo 上都会做的事

1. `claude` 打开
2. `/init` 生成记忆文件
3. 编辑 `CLAUDE.md` 加上 3-5 条具体规范
4. 添加 `.claude/settings.json` 配置测试/lint 命令的权限
5. 把 `.claude/settings.local.json` 加进 `.gitignore`
6. 提交然后继续工作

五分钟。第一次任何人——我或队友——用 Claude Code 打开 repo 时就回本了。

下一篇：快捷键和那个所有人都忽略的四状态模式切换。