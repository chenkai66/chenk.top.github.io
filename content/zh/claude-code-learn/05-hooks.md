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
如果说 MCP 是 Claude 向外扩展能力的机制，那么 Hooks 则是向内施加约束的手段——强制执行你关心的规则，而不依赖模型自觉遵守。

![Claude Code 实战 (5)：Hooks，或如何不再担心 Yolo 模式 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_1.png)

## 模型

Hook 本质上是一条 shell 命令，Claude Code 会在若干预定义时机触发。最常用的两个时刻是：

- **`PreToolUse`** — 在工具调用前运行。退出码为 0 表示允许执行；非零则中止操作。
- **`PostToolUse`** — 在工具返回后运行。此时退出码仅用于信息提示，不影响流程；可用于格式化文件、运行 linter 或记录日志。

还有其他的（`UserPromptSubmit`、`Stop`、`Notification`、`SubagentStop`）。但在日常工作中，这两个 Hook 已能满足绝大多数需求。

![Claude Code 的 7 个 Hook 事件按触发时机分类如下：](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig2_taxonomy.png)
*Claude Code 的 7 个 Hook 事件按触发时机分类。*

## Hook 生命周期

下图展示了在一次完整对话轮次中，每个 Hook 的触发时机。

![在一次对话轮次中，每个 Hook 的触发时机。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig1_lifecycle.png)
*在一次对话轮次中，每个 Hook 的触发时机。*

下面详细描述 Claude 调用工具且已配置 hook 时的实际完整流程——理解这一生命周期是编写正确可靠的 hook 的关键。

---

### `PreToolUse` — 工具调用前的守门人

当 Claude 决定调用某个工具时，将按以下顺序执行：

1. Claude Code 解析工具名称与输入参数；
2. 检查设置中的 `PreToolUse` 钩子，**逐个匹配 `matcher`**，选出所有匹配项；
3. 选中的钩子**按配置中出现的顺序串行执行**；
4. 每个钩子通过 **stdin 接收 JSON 格式的工具调用详情**；
5. 若**任一钩子以非零退出码退出，则整个工具调用被阻断**；该钩子写入 `stderr` 的内容将作为错误消息返回给模型；
6. 若所有钩子均以 `exit code 0` 退出，则工具调用继续执行。

传入 stdin 的 JSON 示例：

```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /tmp/test"
  }
}
```

钩子可访问的环境变量：

| 变量 | 含义 |
|------|------|
| `CLAUDE_PROJECT_DIR` | 项目根目录的绝对路径 |
| `CLAUDE_SESSION_ID` | 当前会话唯一标识符 |
| `CLAUDE_TOOL_NAME` | 正在被调用的工具名称 |

---

### `PostToolUse` — 工具调用后的检查者

工具执行完成后，将进行以下步骤：

1. Claude Code 收集工具原始输出；
2. 查找匹配 `matcher` 的 `PostToolUse` 钩子；
3. 匹配钩子**串行执行**；
4. 每个钩子通过 stdin 接收包含调用与结果的 JSON；
5. 若钩子向 stdout 写入内容，则该输出将完全替代原始工具输出，供模型后续使用——这是实现过滤、脱敏或标注的核心机制；
6. 退出码仅作信息用途：**非零退出码不会撤销已执行的工具调用**；但 `stderr` 输出会被记录，可用于调试。

`PostToolUse` 的 stdin 输入示例（含调用上下文与执行结果）：

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

---

### `UserPromptSubmit` — 用户输入的校验器

在用户 prompt 发送给模型之前触发。适用于：

- 检测并警告含敏感数据的 prompt；
- 自动为每个 prompt 注入标准上下文（如团队规范、格式要求）；
- 审计日志记录所有用户输入。

stdin 输入结构：

```json
{
  "prompt": "The user's raw prompt text",
  "session_id": "abc123"
}
```

---

### `Stop` — 会话终结通知器

当 Claude 完成响应并即将交还控制权给用户时触发。典型用途包括：

- 生成任务摘要报告；
- 清理临时文件或资源；
- 发送完成通知（例如：“Claude 已完成该任务”）。

---

### `Notification` — 桌面通知处理器

当 Claude Code 发出桌面通知（如长耗时任务结束）时触发。可以用于：

- 将通知转发至 Slack、邮件、Webhook 等外部通道；
- 实现自定义提醒逻辑（如仅对高优先级任务发声）。

---

### `SubagentStop` — 子代理终止监视器

当主 agent 启动的子 agent（用于并行任务）完成时触发。适用于：

- 聚合多个子任务结果；
- 记录子 agent 执行耗时与状态；
- 触发下游协调逻辑。

---

## 钩子执行模型：关键细节

以下是钩子实际运行时必须注意的核心行为：

✅ **钩子本质是 shell 命令，而非独立脚本**  
`command` 字段直接交由系统 shell 执行：`/bin/sh -c "..."`。因此可原生使用管道、重定向等 shell 特性：

```json
{
  "command": "cat | jq -r '.tool_input.command' | grep -q 'rm -rf' && exit 1 || exit 0"
}
```

> ⚠️ 但超过单行逻辑时，请务必改用外部脚本文件，以保障可读性与可维护性。

⏱️ **钩子有默认超时限制**  
通常为数秒。超时后进程将被强制终止：  
- 对 `PreToolUse`：超时 ≈ 阻断调用（行为等效于非零退出）；  
- 对其他类型：超时后跳过，继续流程。  
**切勿在钩子中发起无超时保护的 HTTP 请求。**

↔️ **钩子严格同步执行**  
前一个钩子未退出，下一个绝不会启动。例如：5 个各耗时 200ms 的钩子 → 单次工具调用增加 1 秒延迟。请确保钩子轻量、高效。

🌍 **钩子继承 Claude Code 进程的完整环境**  
可直接使用当前 shell 的 `PATH`、自定义环境变量、工作目录等，无需额外配置。

---

掌握以上机制，你就能精准控制工具调用的每一环节——从拦截危险命令，到净化输出、审计输入、协同子任务，全部尽在掌握。

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

Hook 命令通过 stdin 接收工具调用的输入（JSON 格式），并通过环境变量获取当前对话上下文。最简单的 hook 就是读取 stdin，做出判断，然后返回合适的退出码。

![单次 Hook 调用的输入、输出与约束。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig5_anatomy.png)
*单次 Hook 调用的输入、输出与约束。*

### Matcher 模式参考

`matcher` 字段是一个针对工具名的正则表达式。常用模式如下：

| Matcher | 匹配范围 |
|---------|----------|
| `Bash` | 仅内置 Bash 工具 |
| `Write` | 文件写入工具 |
| `Edit` | 文件编辑工具 |
| `Read` | 文件读取工具 |
| `Write\|Edit` | Write 或 Edit 工具 |
| `mcp__playwright__.*` | 所有 Playwright MCP 工具 |
| `mcp__.*` | 所有 MCP 工具 |
| `.*` | 所有工具调用 |

请尽量使用最具体的 matcher。`.*` 会匹配每次工具调用——包括频繁发生的读取操作，可能在单次会话中触发数十次，累积开销不容忽视。

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

注册方式：

![PreToolUse 与 PostToolUse 的退出码语义对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig3_exitcodes.png)
*PreToolUse 与 PostToolUse 的退出码语义对比。*

`PreToolUse` 匹配器设为 `Bash`。阻断时返回退出码 1；stderr 输出的消息会回传给模型，使其了解原因并调整行为。这正是“代理静默失败”和“代理主动请求许可”之间的关键区别。

### 模型如何响应阻断

当 `PreToolUse` 钩子阻止工具调用时，模型会收到 stderr 中的错误消息。一条清晰的错误提示能帮助模型快速恢复：

§§35§§

模型通常会承认限制，并尝试替代方案。如果提示足够明确，它往往能在第一次尝试就找到正确做法。

## Hook 2：写入时自动格式化

![format-on-write：按扩展名路由的 PostToolUse 模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/fig4_formatflow.png)
*format-on-write：按扩展名路由的 PostToolUse 模式。*

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

`PostToolUse` 匹配器设为 `Write|Edit`。每次 Claude 修改文件，都会在代理进入下一轮对话前自动格式化。你无需再反复提醒代理格式化代码——代码风格由 Hook 自动保障。

关于该钩子的几点说明：

- 它在调用格式化工具前，先通过 `command -v` 检查其是否存在；若目标工具未安装，钩子会静默降级，不报错也不中断流程。
- 使用 `2>/dev/null` 抑制格式化器输出的警告信息——避免这些无关日志污染模型上下文。
- 对 Python 文件同时运行 `ruff format`（代码风格）和 `ruff check --fix`（静态检查自动修复），实现双重保障。

## Hook 3：记录每次工具调用

```bash
#!/usr/bin/env bash
input=$(cat)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $(jq -c . <<< "$input")" \
  >> "$CLAUDE_PROJECT_DIR/.claude/tool-log.jsonl"
exit 0
```

`PostToolUse` 匹配器设为 `.*`。将每次工具调用连同时间戳记录到 JSONL 文件中，每行一条记录。无论出问题还是运行顺利，你都能获得完整的审计轨迹。

这个功能我已实际使用过三次，全部用于事后复盘——例如，需要厘清“在那 40 分钟的会话中，Claude 到底执行了哪些操作”。这点磁盘开销完全值得。

### 日志分析（JSONL 格式）

JSONL 结构天然适合命令行分析：

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

`PreToolUse` 匹配器设为 `Bash`。拦截 `git commit` 调用并先运行测试套件。如果测试失败，提交被阻断，模型会被告知原因。这是给代理本身准备的 pre-commit hook。

这听起来可能有些激进，实际上确实如此。关键是，你再也不会*意外*提交坏代码。模型必须主动绕过 Hook 才可能出错，而默认情况下它不会这么做。

改进版能自动检测项目所用测试框架，并执行对应命令，原生支持 npm、Make、pytest 和 Cargo：

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

`PostToolUse` 匹配器设为 `Read|Bash`。在代理看到之前，通过流脱敏器过滤工具输出。如果你不小心 `cat` 了包含 secrets 的文件，模型永远读不到它们——它们会被内联替换为 `[REDACTED]`。

这是在真实代码库中落地时最关键的安全类 Hook。

扩展版覆盖更全面的密钥模式：
- GitHub tokens（`ghp_`、`gho_`、`ghs_`、`ghr_`）
- Slack tokens（`xoxb-`、`xoxp-`、`xoxs-`）
- AWS Access Key ID（`AKIA...`）
- 私钥块（`-----BEGIN ... PRIVATE KEY-----`）
- 含密码的连接字符串
- JWT tokens
- 配置文件中常见的 `password=` 字段

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

§§36§§

## Hook 6：禁止写入受保护文件

团队项目标配钩子，用于阻断对常见敏感文件的修改（如 `.env`、`credentials.json`、`*.pem`），并拒绝写入项目目录之外的路径：

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

注册为 `PreToolUse` 钩子，匹配 `Write|Edit` —— 在内置权限控制之上再加一层防护（belt-and-suspenders）。

## Hook 7：长时操作完成通知

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

注册于 `Notification` 事件：

§§37§§

## 钩子中的错误处理

崩溃的钩子比不存在的钩子更危险。以下是构建健壮钩子的方法。

### 始终使用 `set -euo pipefail`

```bash
#!/usr/bin/env bash
set -euo pipefail
```

该设置可捕获以下问题：
- `set -e`：任一命令失败时立即退出  
- `set -u`：引用未定义变量时报错  
- `set -o pipefail`：捕获管道中任意命令的失败（而非仅最后一个）

### 优雅处理缺失的 `jq`

并非所有机器都预装 `jq`。若钩子依赖它，请先检测：

```bash
if ! command -v jq &>/dev/null; then
  exit 0
fi
```

### 避免钩子静默失败

钩子崩溃时，Claude Code 会记录错误，但行为因类型而异：
- `PostToolUse`：继续执行工具调用  
- `PreToolUse`：**阻塞**工具调用（即完全不执行）  
因此，崩溃的 `PreToolUse` 钩子会导致功能中断。务必测试你的钩子。

### 使用 `trap` 清理临时资源

若钩子创建临时文件：

```bash
tmp_file=$(mktemp)
trap "rm -f $tmp_file" EXIT
```

## 本地测试钩子

无需运行 Claude Code 即可测试钩子——它们只是读取 `stdin` 并返回 `exit code` 的普通脚本。

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

在钩子目录旁维护一个测试文件：

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

运行方式：`bash .claude/hooks/test-hooks.sh`。建议加入 CI 流程以保障钩子持续有效。

## 钩子性能考量

每个钩子都会增加延迟。以下方法可确保延迟可控。

### 测量钩子耗时

```bash
time echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' \
  | bash .claude/hooks/check-bash.sh
```

目标：单个钩子 **< 50ms**；超过 **200ms** 即可感知明显卡顿。

### 常见性能陷阱

| 陷阱 | 开销 | 修复方案 |
|------|------|----------|
| `PreToolUse` 中执行 `npm test` | 秒级 | 缓存测试结果，仅当文件变更时重跑 |
| 调用外部 HTTP API | 100ms–5s | 设置超时，或移至异步 `PostToolUse` |
| 读取大文件 | 不定 | 用 `head`/`tail` 替代全量读取 |
| 多次调用 `jq` | 每次约 10ms | 合并在单条 `jq` 命令中链式过滤 |
| 启动 Python/Node 解释器 | 100–300ms | 简单逻辑优先用 bash 实现 |

### 性能预算建议

```text
PreToolUse 钩子总耗时： < 200ms  
PostToolUse 钩子总耗时： < 500ms（非关键路径，工具调用后执行）
```

例如：5 个钩子 × 100ms = 每次工具调用额外 500ms；若单会话含 50 次调用，则纯钩子开销达 **25 秒**。务必保持精简。

## 组合多个钩子

同一事件与 matcher 可配置多个钩子，按声明顺序依次执行：

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

`PreToolUse` 中顺序至关重要：  
✅ 将最快检查（如字符串匹配）前置  
✅ 将最慢检查（如运行测试）后置  
✅ 若前置钩子已阻塞，后续钩子将**不再执行**

你也可以组合不同 matcher：

§§38§§

## 反模式：相对路径

![Claude Code 实操 (5)：Hooks，或如何不再担心 Yolo 模式 —— 视觉呈现](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/05-hooks/illustration_2.png)

最常见的 hook bug 是使用相对命令路径：

```json
{ "command": "./scripts/check.sh" }   // wrong
```

Claude Code 执行 Hook 时的工作目录是不可预知的。始终使用绝对路径，或者使用 `$CLAUDE_PROJECT_DIR` 环境变量：

```json
{ "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check.sh" }  // right
```

脚本内部也一样。在 hook 里不要 `cd`，也不要用相对路径。

### 其他常见反模式

**反模式：钩子修改工具输入**  
`PreToolUse` 钩子仅能阻塞，**不可修改**工具参数。如需改写行为，请改用 slash 命令包装器等替代方案。

**反模式：钩子依赖网络**  
调用远程 API 的钩子会拖慢每次工具调用，且易因网络抖动失败。如需远程日志，请本地缓冲 + 异步刷新。

**反模式：钩子启动重量级进程**  
在钩子中启动 Python、Node 或 Docker 进程开销巨大。热路径（hot-path）钩子应坚持使用 `bash`、`jq`、`grep`、`sed`。

**反模式：过度宽泛的 matcher**  
例如 `.*` 匹配所有 `PostToolUse` 事件，会对每个 `Read`、`Bash`、`Edit` 都触发——单会话可能执行数百次。请精准匹配。

## 完整生产级钩子配置

以下是我为新项目初始化的 `.claude/settings.json` 钩子配置（精简版）。共 5 个钩子，每个 ≤ 50 行，单次工具调用总开销 < 200ms，覆盖 90% 安全与自动化需求：

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

效果包括：  
1. 危险命令拦截（如 `rm -rf /`）  
2. 受保护文件写入防护（如 `.env`、`package-lock.json`）  
3. 写入后自动格式化（如 Prettier）  
4. 敏感信息脱敏（如密码、token）  
5. 工具调用完成通知  

## Hooks 如何融入团队工作流

提交到 `.claude/settings.json` 的 Hooks 对所有人都生效。这就是重点。新同事 clone 仓库，运行 `claude`，自动继承你的安全护栏和格式化策略。无需设置，无需 opting-in。

对于仅限个人的 hooks（例如依赖你特定工具的），请放在 `.claude/settings.local.json` 中。它们只留在你的磁盘上。

## Hooks 不是什么

- **不是权限的替代品。** Hook 可以作为权限的补充，但不应取而代之。权限是声明式的，明确且静态；Hook 是可执行的，配置不当的风险更高。
- **不是免费的检查。** 每个 hook 都会为每次工具调用增加延迟：五个 hook 各耗时 100ms，单次工具调用就累计增加 500ms 延迟。请合理规划 Hook 的性能开销。
- **不是图灵完备的配置。** 当 hook 的逻辑复杂到接近一个小程序时，建议直接构建 MCP Server。
- **不是安全边界。** Hook 与 Claude Code 运行于同一进程上下文。一个足够有创造力的模型可能会绕过它们。它们是护栏，不是防火墙。

下一篇是 SDK 和 GitHub 集成——程序化的 Claude Code，在 CI 中，针对 PR。系列的最后一篇，也是最强大的一篇。
