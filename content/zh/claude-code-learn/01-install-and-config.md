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
这是六篇实战指南的第一篇。顺序是精心安排的：每一篇都为下一篇铺路。等你读完，就能掌握 90% 用户从未用过的六个功能。
![Claude Code 实战入门（一）：安装、三层配置、以及 # @ /init 这一组 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/01-install-and-config/illustration_1.jpg)

## 安装

目前只有一种官方支持的安装方法，而且这种方法是正确的：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

这个命令会把 `claude` 二进制文件安装到 `~/.local/bin/`（Linux/Mac）。把它加到 PATH 里，然后运行：

```bash
claude --version
# claude-code 0.2.x
claude login
```

登录时会弹出浏览器窗口，授权 CLI 后就完成了。认证信息保存在 `~/.claude/auth.json`。我特别想提一句，`~/.zshrc` 里不会存任何 API Key，所以不用担心泄露问题——很多所谓的“AI CLI”工具在这点上都做错了。
## 跑起来

进入一个真实项目目录，执行以下命令：

```bash
claude
```

你会看到一个交互式提示符。试着输入一句命令，确认安装是否成功：

```
> 这个目录里有什么？
```

Claude 会调用自己的工具执行 `ls`，然后总结并回复内容。如果这一步没问题，说明安装已经完成。
## 三层配置

![Claude Code 实战入门（一）：安装、三层配置、以及 # @ /init 这一组 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/01-install-and-config/illustration_2.jpg)

很多用户可能从来没看过这部分内容。Claude Code 的配置来自三个地方，优先级从低到高依次是：

| 层 | 路径 | 是否进 git？ | 用途 |
|----|------|--------------|------|
| Machine | `~/.claude/settings.json` | 否 | 我的个人全局偏好 |
| Project | `<repo>/.claude/settings.json` | 是 | 团队共享的项目约定 |
| Local | `<repo>/.claude/settings.local.json` | 否（gitignore） | 针对当前仓库的私有覆盖 |

为什么需要三层而不是两层？因为有些设置是“个人 + 全局”的，比如默认模型和编辑器命令；有些是“团队 + 共享”的，比如项目的 linter 和测试命令；还有些是“个人 + 项目内”的，比如你自己的 staging 环境 API Key。

举个具体例子。我的机器级配置文件 `~/.claude/settings.json`：

```json
{
  "model": "claude-opus-4-7",
  "editor": "code"
}
```

某项目的 `.claude/settings.json`（已提交到 git）：

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

队友的 `.claude/settings.local.json`（仅在本地，未提交）：

```json
{
  "model": "claude-sonnet-4-7",
  "env": {
    "STAGING_API_KEY": "sk-..."
  }
}
```

合并规则决定了：队友会使用项目的权限和测试命令配置，但模型选择和 staging Key 则是自己的偏好。这听起来很简单，但我见过整支团队在推广时都卡在这个问题上。
## `#`——写入上下文

在行首输入 `#`，后面的内容会直接追加到项目的记忆文件 `CLAUDE.md`，而不会作为 Prompt 发送。比如：

```
# 在这个仓库里永远不要用 yarn——只用 npm。
```

Claude 不会回复。它会打开 `CLAUDE.md` 文件，把这行内容加进去，然后保存。下一次对话时，这行内容已经包含在上下文中。

我用这种方式避免重复说明自己的偏好。比如："写 Python 时加上类型注解"、"提交信息不要加表情符号"、"CI 用的是 GitHub Actions，不是 GitLab"。每一条都是一行 `#`，写进去就一直有效。
## `@`——引用文件

输入 `@`，就会弹出一个模糊文件选择器。选好文件后，它会自动附加到下一条消息中：

```
@src/router.ts
解释一下路由匹配器如何处理结尾的斜杠
```

文件是以工具结果的形式发送的，不会直接粘贴到你的 Prompt 里。也就是说，即使你附加一个 2000 行的大文件，也不会让 Prompt 变得难以阅读。
## `/init`——初始化项目记忆

每个代码库运行一次 `/init`。Claude 会读取代码库内容，生成一份 `CLAUDE.md` 文件，包含以下内容：

- 项目功能
- 使用的语言和框架
- 构建、测试、Lint 命令
- 主要目录及其作用
- Claude 能识别的规范（如提交信息风格、测试命名规则等）

接着你来修改这份文件。关键不在于 Claude 生成的内容是否完美——它肯定不完美。关键是，你用 30 秒就能得到一个起点，而不是花 30 分钟从头开始。

生成的 `CLAUDE.md` 文件需要提交到代码库中。每个团队成员的 Claude Code 会话都会以它为基础启动。这就是项目共享心智模型的方式。
## 每次新建仓库时我的操作

1. 输入 `claude` 打开
2. 运行 `/init` 生成记忆文件
3. 修改 `CLAUDE.md`，补充 3 到 5 条具体规范
4. 添加 `.claude/settings.json`，配置测试和 lint 命令的权限
5. 把 `.claude/settings.local.json` 加入 `.gitignore`
6. 提交代码，继续下一步

整个过程五分钟。无论是我还是队友，第一次用 Claude Code 打开这个仓库时，这些准备就值回时间了。

下一篇聊聊快捷键，还有那个容易被忽略的“四态切换”功能。
