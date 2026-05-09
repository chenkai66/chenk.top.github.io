---
title: "Claude Code 实战入门（五）：Hooks，让你不再担心 Yolo 模式"
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
如果 MCP 是 Claude 主动伸出的手，那么 Hooks 就是我用来介入的工具。它让我能够强制执行那些我关心的规则，而不是仅仅寄希望于它们被遵守。
![Claude Code 实战入门（五）：Hooks，让你不再担心 Yolo 模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_1.jpg)

## 模型

Hook 是一个 shell 命令，Claude Code 会在几个明确的时机运行它。最常用的有两个：

- **`PreToolUse`** —— 在工具调用前运行。退出码为 0 时允许工具继续执行；非 0 则阻止。
- **`PostToolUse`** —— 在工具返回后运行。退出码只提供信息，可以用它来格式化文件、运行 Lint 或记录日志。

其他还有几个（`UserPromptSubmit`、`Stop`、`Notification`、`SubagentStop`），但日常工作中这两个就占了 90% 的使用场景。
## Hook 的位置

在 `.claude/settings.json` 文件里（或者它的本地变体 `.local`）。最简单的例子如下：

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

`matcher` 决定了这个 Hook 适用于哪些工具。`"Bash"` 对应内置的 Bash 工具，`"Write"` 对应文件写入操作，`"mcp__playwright__.*"` 匹配所有 Playwright MCP 工具。支持正则表达式，通配符也很常用。

Hook 命令会通过标准输入（stdin）接收工具的输入数据（JSON 格式），同时对话上下文会通过环境变量传递。一个最简单的 Hook 只需要读取 stdin，做出判断，然后用合适的退出码退出即可。
## Hook 1：拦截危险命令

`.claude/hooks/check-bash.sh`：

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')

# 拦截所有可能造成破坏的命令
if echo "$cmd" | grep -E '(rm -rf /|sudo rm|chmod -R 777)' >/dev/null; then
  echo "拒绝执行 —— 检测到危险命令模式：$cmd" >&2
  exit 1
fi
exit 0
```

`PreToolUse` 匹配器 `Bash`。脚本以退出码 1 阻止操作，错误信息会通过 stderr 返回给模型，让模型知道原因并调整行为。这样，Agent 不会默默崩溃，而是学会请求权限后再行动。
## Hook 2：写入时自动格式化

`.claude/hooks/format-on-write.sh`：

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

`PostToolUse` 匹配器 `Write|Edit`。每次 Claude 修改文件后，文件会在下一轮操作前自动完成格式化。我再也不用操心代码风格问题——团队的代码规范完全由工具强制执行。
## Hook 3：记录每次工具调用

```bash
#!/usr/bin/env bash
input=$(cat)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $(jq -c . <<< "$input")" \
  >> "$CLAUDE_PROJECT_DIR/.claude/tool-log.jsonl"
exit 0
```

`PostToolUse` 匹配 `.*`。每次工具调用都会带上时间戳，写入 JSONL 文件，一行一条记录。出了问题或者一切正常时，你都能查到完整的操作记录。

我用过三次，次次都是事后分析。当时我需要搞清楚："那 40 分钟的会话里，Claude 到底做了什么？" 占这点硬盘空间，值。
## Hook 4：提交前强制通过测试

```bash
#!/usr/bin/env bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')
if echo "$cmd" | grep -qE '^git commit'; then
  if ! npm test --silent >/dev/null 2>&1; then
    echo "提交被拒绝：测试未通过。运行 'npm test' 查看详细信息。" >&2
    exit 1
  fi
fi
exit 0
```

`PreToolUse` matcher `Bash`。拦截 `git commit` 命令，先运行测试套件。如果测试失败，提交会被阻止，并告诉模型原因。这是给 Agent 自己用的 pre-commit hook。

听起来确实有点强硬，但目的很明确：你不可能*无意中*提交有问题的代码。模型如果想绕过这个限制，必须主动想办法，而它不会这么做。
## Hook 5：隐藏工具输出中的密钥

```bash
#!/usr/bin/env bash
sed -E '
  s/(Bearer|sk-)[A-Za-z0-9_-]{20,}/\1[REDACTED]/g
  s/(api[_-]?key["\s:=]+)["A-Za-z0-9_-]{16,}/\1[REDACTED]/gI
'
```

`PostToolUse` matcher `Read|Bash`。这个 Hook 会在 Agent 看到工具输出之前，用流式过滤器把敏感信息替换掉。如果我不小心用 `cat` 命令查看了包含密钥的文件，模型也永远不会读到这些内容——它们会被直接替换成 `[REDACTED]`。

这是在处理真实代码库时最重要的一个安全 Hook。
## 反模式：相对路径

![Claude Code 实战入门（五）：Hooks，让你不再担心 Yolo 模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_2.jpg)

最常见的 Hook 问题就是用了相对路径来指定命令：

```json
{ "command": "./scripts/check.sh" }   // 错
```

Claude Code 会在一个我不控制的工作目录下运行 Hook。所以，永远使用绝对路径，或者用 `$CLAUDE_PROJECT_DIR` 环境变量：

```json
{ "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check.sh" }  // 对
```

脚本内部也是一样，别用 `cd`，也别用相对路径。
## Hook 在团队工作流中的作用

把 Hook 提交到 `.claude/settings.json`，它们就会对所有人生效。这是它的核心价值。新成员克隆仓库后，运行 `claude`，就能自动继承你的安全规则和格式化策略。不用配置，也不用额外操作。

如果是只给自己用的 Hook（比如依赖特定工具的），就放到 `.claude/settings.local.json` 里。这些 Hook 只会留在你本地，不会影响别人。
## Hook 不是什么

- **不能替代权限。** Hook 可以补充权限，但绝对不能取代。权限是声明式的，明确清晰；而 Hook 是可执行的，容易配置出错。
- **不是零成本检查。** 每个 Hook 都会增加工具调用的延迟。5 个 Hook 每个耗时 100 毫秒，加起来就是每次调用多出半秒。注意控制开销。
- **不是图灵完备的配置。** 如果你的 Hook 开始变得像个小脚本，那就应该考虑搭建一个 MCP 服务器了。

接下来要讲的是 SDK 和 GitHub 集成——通过程序化的方式运行 Claude Code，在 CI 中针对 PR 进行操作。这是系列文章的最后一部分，也是功能最强大的一部分。
