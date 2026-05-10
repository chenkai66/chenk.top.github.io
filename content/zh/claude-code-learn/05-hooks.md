---
title: "Claude Code 实战（五）：Hooks——Yolo 模式的安全网"
date: 2026-04-20 09:00:00
tags:
  - claude-code
  - hooks
  - safety
  - automation
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 5
description: "Hooks 是每次工具调用前后跑的 shell 脚本。PreToolUse 可以阻止。PostToolUse 可以格式化、Lint、记日志。我每个 Repo 都用的 5 个 Hook，加上一个把所有人都坑过的反模式。"
disableNunjucks: true
translationKey: "claude-code-learn-5"
---
如果说 MCP 是 Claude 向外延伸的方式，那 Hooks 就是你向内管控的手段。它是用来强制执行——而不是单纯指望——那些你在乎的规则。

![Claude Code Hands-On (5): Hooks, or How to Stop Worrying About Yolo Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_1.png)

## 模型

Hook 本质上就是一个 shell 命令。Claude Code 会在几个定义明确的时刻运行它。你最常用到的两个时刻：

- **`PreToolUse`** — 在工具调用前运行。退出码 0 表示放行；非零则阻断。
- **`PostToolUse`** — 在工具返回后运行。退出码仅作信息参考；你可以用它来格式化文件、运行 linter 或记录日志。

还有其他的（`UserPromptSubmit`, `Stop`, `Notification`, `SubagentStop`）。但对于日常工作，上面那两个提供了 90% 的价值。

## Hooks 存放在哪

在 `.claude/settings.json`（或者它的本地变体）里。最小化示例：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "/path/to/check-bash.sh" }
        ]
      }
    ]
  }
}
```

`matcher` 决定 hook 作用于哪些工具。`"Bash"` 匹配内置的 Bash 工具。`"Write"` 匹配文件写入。`"mcp__playwright__.*"` 匹配所有 Playwright MCP 工具。支持正则表达式——通配符很常见。

Hook 命令通过 stdin 接收工具的输入（JSON 格式），并通过环境变量获取对话上下文。最简单的 hook 就是读取 stdin，做出判断，然后返回合适的退出码。

## Hook 1：禁止危险命令

`.claude/hooks/check-bash.sh`:

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')

# Refuse anything that destroys broadly
if echo "$cmd" | grep -E '(rm -rf /|sudo rm|chmod -R 777)' >/dev/null; then
  echo "Refusing — destructive pattern detected: $cmd" >&2
  exit 1
fi
exit 0
```

`PreToolUse` 匹配器 `Bash`。阻断时退出码为 1；stderr 上的消息会回传给模型，让它知道原因并调整行为。这就是“代理静默崩溃”和“代理学会请求许可”之间的区别。

## Hook 2：写入时自动格式化

`.claude/hooks/format-on-write.sh`:

```bash
#!/usr/bin/env bash
input=$(cat)
path=$(echo "$input" | jq -r '.tool_input.file_path')

case "$path" in
  *.py) ruff format "$path" 2>/dev/null ;;
  *.ts|*.tsx|*.js|*.jsx) prettier --write "$path" 2>/dev/null ;;
  *.go) gofmt -w "$path" 2>/dev/null ;;
esac
exit 0
```

`PostToolUse` 匹配器 `Write|Edit`。每次 Claude 触碰文件，在代理进入下一轮对话前都会先格式化。你再也不需要跟代理啰嗦格式化问题——你的代码风格由代码强制执行。

## Hook 3：记录每次工具调用

```bash
#!/usr/bin/env bash
input=$(cat)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $(jq -c . <<< "$input")" \
  >> "$CLAUDE_PROJECT_DIR/.claude/tool-log.jsonl"
exit 0
```

`PostToolUse` 匹配器 `.*`。将每次工具调用连同时间戳记录到 JSONL 文件中。每次调用一行。当出问题——或者一切顺利时——你拥有完整的审计轨迹。

这个功能我正好用过三次。三次都是事后复盘，我需要知道“在那 40 分钟的会话里 Claude 到底做了什么”。这点磁盘空间花得值。

## Hook 4：提交前强制测试通过

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')
if echo "$cmd" | grep -qE '^git commit'; then
  if ! npm test --silent >/dev/null 2>&1; then
    echo "Refusing commit: tests fail. Run 'npm test' to see details." >&2
    exit 1
  fi
fi
exit 0
```

`PreToolUse` 匹配器 `Bash`。拦截 `git commit` 调用并先运行测试套件。如果测试失败，提交被阻断，模型会被告知原因。这是给代理本身准备的 pre-commit hook。

这听起来很激进。确实如此。重点在于你再也无法*意外*提交坏代码了。模型必须主动绕过 hook 才能做错事，而它不会这么做。

## Hook 5：脱敏工具输出中的 secrets

```bash
#!/usr/bin/env bash
sed -E '
  s/(Bearer|sk-)[A-Za-z0-9_-]{20,}/\1[REDACTED]/g
  s/(api[_-]?key["\s:=]+)["A-Za-z0-9_-]{16,}/\1[REDACTED]/gI
'
```

`PostToolUse` 匹配器 `Read|Bash`。在代理看到之前，通过流脱敏器过滤工具输出。如果你不小心 `cat` 了包含 secrets 的文件，模型永远读不到它们——它们会被内联替换为 `[REDACTED]`。

这是处理真实代码库时最重要的安全 hook。

## 反模式：相对路径

![Claude Code Hands-On (5): Hooks, or How to Stop Worrying About Yolo Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_2.png)

最常见的 hook bug 是使用相对命令路径：

```json
{ "command": "./scripts/check.sh" }   // wrong
```

Claude Code 会在一个你无法控制的工作目录下运行 hook。始终使用绝对路径，或者使用 `$CLAUDE_PROJECT_DIR` 环境变量：

```json
{ "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check.sh" }  // right
```

脚本内部也一样。在 hook 里不要 `cd`，也不要用相对路径。

## Hooks 如何融入团队工作流

提交到 `.claude/settings.json` 的 Hooks 对所有人都生效。这就是重点。新同事 clone 仓库，运行 `claude`，自动继承你的安全护栏和格式化策略。无需设置，无需 opting-in。

对于仅限个人的 hooks（例如依赖你特定工具的），请放在 `.claude/settings.local.json` 中。它们只留在你的磁盘上。

## Hooks 不是什么

- **不是权限的替代品。** Hook 可以补充权限，但不应该取代它们。权限是声明式且明确的；Hook 是可执行的，容易配置错误。
- **不是免费的检查。** 每个 hook 都会给每次工具调用增加延迟。五个 hook 每个 100ms，每次工具调用就是半秒。注意预算。
- **不是图灵完备的配置。** 当 hook 看起来像是一个小程序时，你最好去构建一个 MCP server。

下一篇是 SDK 和 GitHub 集成——程序化的 Claude Code，在 CI 中，针对 PR。系列的最后一篇，也是最强大的一篇。