---
title: "Claude Code 实战（五）：Hooks 与 Yolo 安全网"
date: 2026-04-22 09:00:00
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
如果说 MCP 是 Claude 向外扩展能力的机制，那么 Hooks 就是你向内施加约束的手段——它强制执行你真正关心的规则，而不只是寄托于模型的自觉。

![Claude Code 实战 (5)：Hooks，或如何不再担心 Yolo 模式 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_1.png)

## 模型

Hook 本质上是一条 shell 命令，Claude Code 会在特定时机触发。最常用的两个是：

- **`PreToolUse`** —— 在工具调用前运行。退出码为 0 表示放行；非零则直接拦截。
- **`PostToolUse`** —— 在工具返回后运行。此时退出码仅作信息用途，不影响流程；可用于格式化文件、运行 linter 或记录日志。

此外还有 `UserPromptSubmit`、`Stop`、`Notification` 和 `SubagentStop` 等事件。但在日常开发中，上述两个 Hook 已覆盖 90% 的实用场景。

![Claude Code 的 7 个 Hook 事件按触发时机分类如下：](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig2_taxonomy.png)
*Claude Code 的 7 个 Hook 事件按触发时机分类。*

## 完整的 Hook 生命周期

让我们走一遍 Claude 调用工具且已配置 Hook 时的完整流程。理解这一生命周期，是写出高效可靠 Hook 的关键。

![在一次对话轮次中，每个 Hook 的触发时机。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig1_lifecycle.png)
*在一次对话轮次中，每个 Hook 的触发时机。*

### PreToolUse —— 守门人

当 Claude 决定调用某个工具时，会按以下顺序执行：

1. Claude Code 解析出工具名称和输入参数；
2. 检查 `settings.json` 中所有 `PreToolUse` 类型的 Hook，并根据 `matcher` 筛选出匹配项；
3. 所有匹配的 Hook **按配置中的顺序依次串行执行**；
4. 每个 Hook 通过 **stdin 接收 JSON 格式的工具调用详情**；
5. **只要有一个 Hook 以非零退出码退出，整个工具调用就会被阻断**，其写入 `stderr` 的内容会作为错误消息返回给模型；
6. 如果所有 Hook 都以退出码 0 正常结束，工具调用才会继续。

传入 stdin 的 JSON 结构如下：

```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /tmp/test"
  }
}
```

Hook 可访问的环境变量包括：

| 变量 | 含义 |
|------|------|
| `CLAUDE_PROJECT_DIR` | 项目根目录的绝对路径 |
| `CLAUDE_SESSION_ID` | 当前会话的唯一标识符 |
| `CLAUDE_TOOL_NAME` | 正在被调用的工具名称 |

### PostToolUse —— 检查员

工具执行完成后：

1. Claude Code 收集工具的原始输出；
2. 查找匹配 `matcher` 的 `PostToolUse` Hook；
3. 匹配的 Hook 按序串行执行；
4. 每个 Hook 通过 stdin 接收包含调用与结果的完整 JSON；
5. 如果 Hook 向 stdout 写入内容，该输出将**完全替代原始工具输出**，供模型后续使用——这是实现过滤、脱敏或标注的核心机制；
6. 退出码在此仅为信息用途：**非零退出不会撤销已执行的操作**，但 stderr 会被记录，便于调试。

PostToolUse 的 stdin 输入结构如下（同时包含调用与结果）：

```json
{
  "tool_name": "Read",
  "tool_input": {
    "file_path": "/home/user/project/config.yml"
  },
  "tool_output": {
    "content": "database:\n  host: localhost\n  password: s3cr3t\n"
  }
}
```

### UserPromptSubmit —— 输入校验器

在用户 prompt 发送给模型前触发，适用于：

- 检测并警告包含敏感数据的输入；
- 自动为每个 prompt 注入标准上下文（如团队规范）；
- 审计日志，记录所有用户输入。

其 stdin 输入结构为：

```json
{
  "prompt": "The user's raw prompt text",
  "session_id": "abc123"
}
```

### Stop —— 会话终结器

当 Claude 完成响应、即将交还控制权给用户时触发，可用于：

- 生成任务摘要报告；
- 清理临时文件；
- 发送完成通知（例如：“Claude 已完成该任务”）。

### Notification —— 通知处理器

当 Claude Code 触发桌面通知（如长任务结束）时运行，适合将通知转发至 Slack、邮件或 Webhook 等外部通道。

### SubagentStop —— 子代理监视器

当主 Agent 启动的子 Agent（用于并行任务）完成时触发，可用于聚合结果或记录子任务状态。

## Hook 的执行模型

以下是 Hook 实际运行时的关键细节：

![PreToolUse 与 PostToolUse 的退出码语义对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig3_exitcodes.png)
*PreToolUse 与 PostToolUse 的退出码语义对比。*

**Hook 是 shell 命令，不是脚本**。`command` 字段会被直接传给系统 shell（`/bin/sh -c "..."`），因此你可以直接使用管道、重定向等特性：

```json
{
  "command": "cat | jq -r '.tool_input.command' | grep -q 'rm -rf' && exit 1 || exit 0"
}
```

但逻辑稍复杂时，请改用外部脚本文件。

**Hook 有超时限制**。默认几秒内必须完成，超时会被 kill。对 `PreToolUse` 来说，超时等效于阻断；对 `PostToolUse` 则跳过。**切勿在 Hook 中发起无超时保护的 HTTP 请求**。

**Hook 是同步执行的**。五个各耗 200ms 的 Hook 会让单次工具调用延迟整整一秒。务必保持轻量。

**Hook 继承 Claude Code 的进程环境**，包括 PATH 和环境变量，但**不会加载你的交互式 shell 配置**（如 `.bashrc` 或别名）。

**stdin 会被完整复制给每个 Hook**。多个 Hook 不会互相干扰输入流。

![单次 Hook 调用的输入、输出与约束。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig5_anatomy.png)
*单次 Hook 调用的输入、输出与约束。*

## Hook 存放在哪里

配置位于 `.claude/settings.json`（或其本地变体 `.claude/settings.local.json`）。最小示例：

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

`matcher` 决定 Hook 作用于哪些工具。`"Bash"` 匹配内置 Bash 工具，`"Write"` 匹配文件写入，`"mcp__playwright__.*"` 匹配所有 Playwright MCP 工具。支持正则表达式，通配符很常见。

Hook 通过 stdin 接收工具输入（JSON），并通过环境变量获取上下文。最简单的 Hook 就是读取 stdin、做判断、返回合适退出码。

### Matcher 模式参考

| Matcher | 匹配范围 |
|---------|----------|
| `Bash` | 仅内置 Bash 工具 |
| `Write` | 文件写入工具 |
| `Edit` | 文件编辑工具 |
| `Read` | 文件读取工具 |
| `Write\|Edit` | Write 或 Edit |
| `mcp__playwright__.*` | 所有 Playwright MCP 工具 |
| `mcp__.*` | 所有 MCP 工具 |
| `.*` | 所有工具调用 |

请尽量使用最具体的 matcher。`.*` 会匹配每次 Read、Bash、Edit——单次会话可能触发上百次，开销巨大。

## Hook 1：禁止危险命令

`.claude/hooks/check-bash.sh`：

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

注册方式：

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

`PreToolUse` + `matcher: "Bash"`。阻断时返回退出码 1，stderr 内容会回传给模型，使其知道原因并调整策略。这正是“代理静默失败”和“代理主动请求许可”之间的关键区别。

### 模型如何响应拦截

当 `PreToolUse` 阻断调用时，模型会收到 stderr 中的错误信息。一条清晰的提示能帮助它快速找到替代方案：

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

通常，模型会立即承认限制，并尝试其他方法。如果提示足够明确，往往一次就能成功。

## Hook 2：写入时自动格式化

![format-on-write：按扩展名路由的 PostToolUse 模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig4_formatflow.png)
*format-on-write：按扩展名路由的 PostToolUse 模式。*

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

注册为 `PostToolUse`，matcher 为 `Write|Edit`。每次 Claude 修改文件，都会在进入下一轮对话前自动格式化。从此你再也不用提醒它“记得格式化”——代码风格由 Hook 自动保障。

该 Hook 的几个细节：

- 使用 `command -v` 检查格式化工具是否存在，缺失时静默降级；
- 用 `2>/dev/null` 抑制 formatter 的警告，避免污染模型上下文；
- 对 Python 同时运行 `ruff format`（风格）和 `ruff check --fix`（自动修复）。

注册方式：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/format-on-write.sh"
          }
        ]
      }
    ]
  }
}
```

## Hook 3：记录每次工具调用

```bash
#!/usr/bin/env bash
input=$(cat)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $(jq -c . <<< "$input")" \
  >> "$CLAUDE_PROJECT_DIR/.claude/tool-log.jsonl"
exit 0
```

`PostToolUse` + `matcher: ".*"`。将每次调用连同时间戳写入 JSONL 文件，一行一记录。无论出错还是成功，你都有完整的审计轨迹。

我实际用过三次，全是事后复盘——比如搞清楚“那 40 分钟里 Claude 到底干了什么”。这点磁盘开销完全值得。

### 日志分析

JSONL 格式天然适合命令行分析：

```bash
# 统计今日工具调用次数
wc -l .claude/logs/tools-2026-04-22.jsonl

# 统计各工具调用频次（从高到低）
jq -r '.tool' .claude/logs/tools-2026-04-22.jsonl | sort | uniq -c | sort -rn

# 提取所有执行过的 Bash 命令
jq -r 'select(.tool == "Bash") | .data.tool_input.command' \
  .claude/logs/tools-2026-04-22.jsonl

# 查找失败的工具调用（含 error 字段）
jq 'select(.data.tool_output.error != null)' \
  .claude/logs/tools-2026-04-22.jsonl
```

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

`PreToolUse` + `matcher: "Bash"`。拦截 `git commit`，先跑测试套件。测试失败则阻断提交，并告知模型原因。这是给 Agent 自己设的 pre-commit hook。

听起来激进？确实如此。但好处是：你再也不会**意外**提交坏代码。模型必须主动绕过 Hook 才能犯错，而它通常不会这么做。

改进版能自动识别测试框架，原生支持 npm、Make、pytest 和 Cargo：

```bash
#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')

# 仅拦截 git commit 命令
if ! echo "$cmd" | grep -qE '^git commit'; then
  exit 0
fi

# 根据项目文件推断测试命令
project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
test_cmd=""

if [ -f "$project_dir/package.json" ]; then
  if jq -e '.scripts.test' "$project_dir/package.json" &>/dev/null; then
    test_cmd="npm test --silent"
  fi
elif [ -f "$project_dir/Makefile" ]; then
  if grep -q '^test:' "$project_dir/Makefile"; then
    test_cmd="make test"
  fi
elif [ -f "$project_dir/pyproject.toml" ]; then
  test_cmd="python -m pytest --quiet"
elif [ -f "$project_dir/Cargo.toml" ]; then
  test_cmd="cargo test --quiet"
fi

if [ -z "$test_cmd" ]; then
  exit 0
fi

cd "$project_dir"
if ! eval "$test_cmd" >/dev/null 2>&1; then
  echo "BLOCKED: Refusing commit because tests fail." >&2
  echo "Run '$test_cmd' to see the failures." >&2
  exit 1
fi
exit 0
```

## Hook 5：脱敏工具输出中的 secrets

```bash
#!/usr/bin/env bash
sed -E '
  s/(Bearer|sk-)[A-Za-z0-9_-]{20,}/\1[REDACTED]/g
  s/(api[_-]?key["\s:=]+)["A-Za-z0-9_-]{16,}/\1[REDACTED]/gI
'
```

`PostToolUse` + `matcher: "Read|Bash"`。在模型看到前，通过流式过滤器脱敏输出。如果你不小心 `cat` 了含密钥的文件，模型永远看不到原文——只会看到 `[REDACTED]`。

这是在真实代码库中最关键的安全 Hook。

扩展版覆盖更多模式：GitHub tokens（`ghp_`, `gho_` 等）、Slack tokens（`xoxb-`）、AWS keys（`AKIA...`）、私钥块、含密码的连接字符串、JWT、配置文件中的 `password=` 字段等。

```bash
#!/usr/bin/env bash
sed -E '
  s/(Bearer |sk-|ghp_|gho_|ghs_|ghr_|xoxb-|xoxp-|xoxs-)[A-Za-z0-9_-]{16,}/\1[REDACTED]/g
  s/(api[_-]?key["\s:=]+)["\x27]?[A-Za-z0-9_-]{16,}["\x27]?/\1[REDACTED]/gI
  s/(AKIA)[A-Z0-9]{16}/\1[REDACTED]/g
  s/(-----BEGIN [A-Z ]*PRIVATE KEY-----).*/\1 [REDACTED]/g
  s|(://[^:]+:)[^@]+(@)|\1[REDACTED]\2|g
  s/(password["\s:=]+)["\x27]?[^\s"]+["\x27]?/\1[REDACTED]/gI
  s/eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/[REDACTED_JWT]/g
'
```

可独立测试：

```bash
echo '{"tool_output": "Bearer sk-abc123456789012345678901234567890 and password=\"mysecret\""}' \
  | bash .claude/hooks/redact-secrets.sh
```

预期输出：

```bash
echo '{"tool_output": "Bearer sk-abc123456789012345678901234567890 and password=\"mysecret\""}' \
  | bash .claude/hooks/redact-secrets.sh
```

## Hook 6：禁止写入受保护文件

团队项目标配：

```bash
#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
path=$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.path // empty')

if [ -z "$path" ]; then
  exit 0
fi

PROTECTED=(
  '.env' '.env.local' '.env.production'
  'credentials.json' 'secrets.yaml'
  'id_rsa' 'id_ed25519' '*.pem' '*.key'
)

filename=$(basename "$path")
for pattern in "${PROTECTED[@]}"; do
  if [[ "$filename" == $pattern ]]; then
    echo "BLOCKED: Refusing to modify protected file: $path" >&2
    exit 1
  fi
done

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
case "$path" in
  "$project_dir"/*) ;;
  /tmp/*|/var/tmp/*) ;;
  *)
    echo "BLOCKED: Refusing to write outside project directory: $path" >&2
    exit 1
    ;;
esac
exit 0
```

`PreToolUse` + `matcher: "Write|Edit"`。防止修改 `.env`、`credentials.json` 等敏感文件，也禁止写入项目目录之外。这是在内置权限之上再加一层“裤带+背带”式防护。

## Hook 7：长操作完成通知

```bash
#!/usr/bin/env bash
set -euo pipefail
input=$(cat)

if command -v osascript &>/dev/null; then
  osascript -e 'display notification "Claude has finished the task" with title "Claude Code"'
elif command -v notify-send &>/dev/null; then
  notify-send "Claude Code" "Claude has finished the task"
fi

if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  curl -s -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-Type: application/json' \
    -d '{"text":"Claude Code has finished a task in '"$CLAUDE_PROJECT_DIR"'"}' \
    >/dev/null 2>&1
fi
exit 0
```

注册到 `Notification` 事件：

```bash
#!/usr/bin/env bash
set -euo pipefail
input=$(cat)

if command -v osascript &>/dev/null; then
  osascript -e 'display notification "Claude has finished the task" with title "Claude Code"'
elif command -v notify-send &>/dev/null; then
  notify-send "Claude Code" "Claude has finished the task"
fi

if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  curl -s -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-Type: application/json' \
    -d '{"text":"Claude Code has finished a task in '"$CLAUDE_PROJECT_DIR"'"}' \
    >/dev/null 2>&1
fi
exit 0
```

## Hook 中的错误处理

崩溃的 Hook 比没有更糟。以下是构建健壮 Hook 的方法。

### 始终使用 `set -euo pipefail`

```bash
#!/usr/bin/env bash
set -euo pipefail
```

它能捕获：
- `set -e`：任一命令失败即退出；
- `set -u`：引用未定义变量时报错；
- `set -o pipefail`：管道中任意命令失败都会触发退出。

### 优雅处理缺失的 `jq`

并非所有机器都有 `jq`。若依赖它，请先检查：

```bash
if ! command -v jq &>/dev/null; then
  exit 0
fi
```

### 避免静默失败

Hook 崩溃时，Claude Code 会记录错误，但行为因类型而异：
- `PostToolUse`：继续执行；
- `PreToolUse`：**直接阻断**工具调用。

这意味着一个崩溃的 `PreToolUse` Hook 会让功能完全失效。务必测试！

### 使用 `trap` 清理临时文件

若 Hook 创建了临时文件：

```bash
tmp_file=$(mktemp)
trap "rm -f $tmp_file" EXIT
```

## 本地测试 Hook

无需启动 Claude Code。Hook 本质就是读 stdin、返回退出码的脚本。

### 手动测试

```bash
# 测试 PreToolUse 钩子
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' \
  | bash .claude/hooks/check-bash.sh
echo "Exit code: $?"

# 测试安全命令
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' \
  | bash .claude/hooks/check-bash.sh
echo "Exit code: $?"
```

### 自动化测试脚本

我在 Hook 目录旁维护一个测试文件：

```bash
#!/usr/bin/env bash
set -euo pipefail
PASS=0; FAIL=0

assert_blocked() {
  if echo "$2" | bash "$1" >/dev/null 2>&1; then
    echo "FAIL: Expected block — $3"; ((FAIL++))
  else
    echo "PASS: Correctly blocked — $3"; ((PASS++))
  fi
}

assert_allowed() {
  if echo "$2" | bash "$1" >/dev/null 2>&1; then
    echo "PASS: Correctly allowed — $3"; ((PASS++))
  else
    echo "FAIL: Unexpected block — $3"; ((FAIL++))
  fi
}

HOOK=".claude/hooks/check-bash.sh"
assert_blocked "$HOOK" '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' "rm -rf /"
assert_allowed "$HOOK" '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' "ls -la"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
```

运行：`bash .claude/hooks/test-hooks.sh`。建议加入 CI，确保 Hook 持续有效。

## Hook 性能考量

每个 Hook 都增加延迟。以下是优化建议。

### 测量耗时

```bash
time echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' \
  | bash .claude/hooks/check-bash.sh
```

目标：**单个 Hook < 50ms**；超过 200ms 用户就能感知卡顿。

### 常见性能陷阱

| 陷阱 | 开销 | 修复 |
|------|------|------|
| `PreToolUse` 中跑 `npm test` | 秒级 | 缓存结果，仅文件变更时重跑 |
| 调用外部 API | 100ms–5s | 加超时，或移至异步 PostToolUse |
| 读大文件 | 不定 | 用 `head`/`tail` 替代全读 |
| 多次调用 `jq` | 每次 ~10ms | 合并为单条 jq 命令 |
| 启动 Python/Node | 100–300ms | 简单逻辑用 bash 实现 |

### 性能预算

```text
PreToolUse 钩子总耗时： < 200ms  
PostToolUse 钩子总耗时： < 500ms（非关键路径，工具调用后执行）
```

例如：5 个 Hook × 100ms = 单次调用额外 500ms；若会话含 50 次调用，纯 Hook 开销达 **25 秒**。务必精简。

## 组合多个 Hook

同一事件和 matcher 可配置多个 Hook，按声明顺序执行：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-dangerous.sh" },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-paths.sh" },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/log-commands.sh" }
        ]
      }
    ]
  }
}
```

对 `PreToolUse`，顺序至关重要：**快检查放前，慢检查放后**。若前置 Hook 已拦截，后续不会执行。

也可组合不同 matcher：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-dangerous.sh" },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-paths.sh" },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/log-commands.sh" }
        ]
      }
    ]
  }
}
```

## 反模式：相对路径

最常见的 Bug 是使用相对路径：

```json
{ "command": "./scripts/check.sh" }   // wrong
```

Claude Code 执行 Hook 时的工作目录不可控。**始终用绝对路径，或借助 `$CLAUDE_PROJECT_DIR`**：

```json
{ "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check.sh" }  // right
```

脚本内部也一样：不要 `cd`，不要用相对路径。

### 其他反模式

**反模式：Hook 修改工具输入**  
`PreToolUse` 只能拦截，不能改参数。如需改写行为，请用 slash 命令包装器。

**反模式：Hook 依赖网络**  
远程 API 调用会拖慢每次工具调用，且易失败。如需远程日志，请本地缓冲 + 异步刷新。

**反模式：Hook 启动重量级进程**  
在热路径 Hook 中启动 Python、Node 或 Docker 开销巨大。优先用 bash、jq、grep、sed。

**反模式：matcher 过于宽泛**  
`.*` 会匹配每次 Read/Bash/Edit——单会话可能执行数百次。请精准匹配。

## 完整的生产级 Hook 配置

以下是我为新项目初始化的 `.claude/settings.json` 配置：

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-bash.sh" }] },
      { "matcher": "Write|Edit", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-protected-files.sh" }] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/format-on-write.sh" }] },
      { "matcher": "Read|Bash", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/redact-secrets.sh" }] }
    ],
    "Notification": [
      { "matcher": ".*", "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/notify.sh" }] }
    ]
  }
}
```

它提供：
1. 危险命令拦截（PreToolUse on Bash）
2. 受保护文件防护（PreToolUse on Write/Edit）
3. 自动格式化（PostToolUse on Write/Edit）
4. 敏感信息脱敏（PostToolUse on Read/Bash）
5. 完成通知（Notification）

五个 Hook，每个不超过 50 行，总开销 < 200ms/调用，覆盖 90% 的安全与自动化需求。

## Hook 如何融入团队工作流

提交到 `.claude/settings.json` 的 Hook 对所有人都生效。新成员 clone 仓库后运行 `claude`，自动继承你的安全护栏和格式策略——无需额外设置，无需手动启用。

个人专用 Hook（如依赖本地工具的）请放在 `.claude/settings.local.json`，它们不会被提交。

## Hook 不是什么

- **不是权限的替代品**。Hook 可补充权限，但不应取代。权限是声明式的、明确的；Hook 是可执行的、易误配的。
- **不是免费的检查**。每个 Hook 都增加延迟：五个各 100ms 的 Hook = 单次调用 +500ms。请严控性能预算。
- **不是图灵完备的配置**。当 Hook 逻辑复杂到像小程序时，该考虑构建 MCP Server 了。
- **不是安全边界**。Hook 与 Claude Code 共享进程上下文，足够聪明的模型**可能绕过**它们。它们是护栏（guardrails），不是防火墙（firewalls）。

下一篇是 SDK 与 GitHub 集成——程序化调用 Claude Code，在 CI 中处理 PR。这是系列的终章，也是最强大的一篇。

![Claude Code 实操 (5)：Hooks，或如何不再担心 Yolo 模式 —— 视觉呈现](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_2.png)
