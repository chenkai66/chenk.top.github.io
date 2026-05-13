---
title: "Claude Code 实战（九）：权限模型与环境变量"
date: 2026-04-26 09:00:00
tags:
  - claude-code
  - settings-json
  - permissions
  - configuration
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 9
description: "settings.json 决定了 Claude 能动什么、在哪里、用谁的身份。三层（用户、项目、本地）、权限语法、改变行为的环境变量、以及那条第一次都会咬人的优先级规则。"
disableNunjucks: true
translationKey: "claude-code-learn-9"
---
如果说 hooks 是你伸手进 Claude Code 内部操作的方式，那 `settings.json` 就是划定它能动什么的边界。这也是个容易让人栽跟头的文件，尤其是它的优先级规则。

这一章将补上这块缺失的参考文档。

![Claude Code 实操 (9)：settings.json、三层权限模型和环境 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_1.png)

## 三个层级

Claude Code 会按顺序读取三个 `settings.json` 文件：

1. **用户设置** — `~/.claude/settings.json`，对你机器上的每个项目都生效。
2. **项目设置** — `<repo>/.claude/settings.json`，提交到 Git 中，对这个仓库里的所有人都生效。
3. **本地设置** — `<repo>/.claude/settings.local.json`，被 `.gitignore` 忽略，是你在这个仓库里的私有覆盖配置。

合并规则很简单：后一层配置会逐键（key-by-key）覆盖前一层。**权限方面**，`allow` 是叠加的，`deny` 是否决性的——一旦任何一层 deny 了某个操作，其他层都无法再重新 allow 它。这种不对称性正是系统安全的基石。

实际建议：把组织策略放在 `~/.claude/settings.json`，把项目规则放在 `.claude/settings.json`（提交），把你那些“我信任我机器上这个特定操作”的覆盖配置放在 `.claude/settings.local.json`。  
![图 3：三层 settings.json 文件，以及每个 key 在合并时遵循的不同规则。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig3.png)

*图 3：三层 settings.json 文件，以及每个 key 在合并时遵循的不同规则。*

## 完整的 `settings.json` 参考手册

以下是在 `settings.json` 中可设置的所有顶层键及其说明：

```json
{
  "permissions": {
    "allow": [],
    "deny": [],
    "additionalDirectories": []
  },
  "env": {},
  "hooks": {
    "PreToolUse": [],
    "PostToolUse": []
  },
  "worktree": {
    "baseRef": "fresh"
  }
}
```

### 权限

控制 Claude 可调用的工具及其作用目标。

### 环境

为所有工具调用（如 Bash、hook 等）设置环境变量。

### 钩子

定义在工具调用前或后运行的脚本，详见第 7 章。

### 工作树

控制工作树行为。`baseRef` 可设为 `"fresh"`（从 `origin/main` 分支）或 `"head"`（从当前 `HEAD` 分支）。

---

## 权限块 —— 语法结构

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "WebFetch(domain:docs.anthropic.com)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Read(.env)",
      "Read(**/credentials*)"
    ],
    "additionalDirectories": []
  }
}
```

![图 5：权限规则语法速查 —— 每条规则都是 ToolName(pattern)，括号里的匹配规则因工具而异。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig5.png)

*图 5：权限规则语法速查 —— 每条规则都是 ToolName(pattern)，括号里的匹配规则因工具而异。*

### 工具权限语法

每个权限条目格式为：`ToolName` 或 `ToolName(pattern)`。

括号内为该工具专用的 glob 风格匹配器：

| 工具 | 匹配类型 | 示例 | 匹配范围 |
|------|----------|------|----------|
| `Read` | 文件路径 glob | `Read(src/**)` | `src/` 下任意文件 |
| `Read` | 文件路径 glob | `Read(.env)` | 仅仓库根目录下的 `.env` |
| `Read` | 文件路径 glob | `Read(**/.env*)` | 任意深度的 `.env` 类文件 |
| `Edit` | 文件路径 glob | `Edit(src/**)` | 编辑 `src/` 下的文件 |
| `Edit` | 文件路径 glob | `Edit(*.ts)` | 编辑根目录下的 TypeScript 文件 |
| `Write` | 文件路径 glob | `Write(src/**)` | 在 `src/` 下写入新文件 |
| `MultiEdit` | 文件路径 glob | `MultiEdit(src/**)` | 对 `src/` 下文件执行多处编辑 |
| `Bash` | 命令前缀 | `Bash(npm run *)` | 所有 `npm run` 命令 |
| `Bash` | 命令前缀 | `Bash(git status)` | 仅精确匹配 `git status` |
| `Bash` | 命令前缀 | `Bash(git *)` | 所有 `git` 命令（慎用！） |
| `WebFetch` | 域名 | `WebFetch(domain:docs.anthropic.com)` | 仅该域名 |
| `Grep` | 文件路径 glob | `Grep(src/**)` | 仅在 `src/` 中搜索 |

裸工具名（如仅 `Read`）表示对该工具完全放行。除受信任的个人配置 `~/.claude/settings.json` 外，此做法几乎总是过于宽泛。

### `additionalDirectories` 字段

默认情况下，Claude 仅能访问当前项目目录内的文件。若需授予项目外路径访问权限：

```json
{
  "permissions": {
    "additionalDirectories": [
      "/path/to/shared-libs",
      "/path/to/design-system",
      "~/Documents/specs"
    ]
  }
}
```

典型使用场景：
- 单体仓库（monorepo）中需读取兄弟包
- 独立存放的设计系统目录
- 存放在仓库外的规范文档

### 所有权限类型及示例

可在权限规则中使用的完整工具列表：

| 工具名 | 功能 | 常见 `allow` 模式 | 常见 `deny` 模式 |
|--------|------|------------------|------------------|
| `Read` | 读取文件内容 | `Read(src/**)` | `Read(.env)`, `Read(**/credentials*)` |
| `Edit` | 修改现有文件 | `Edit(src/**)` | `Edit(.github/workflows/**)` |
| `Write` | 创建新文件 | `Write(src/**)` | `Write(.env*)` |
| `MultiEdit` | 单次调用多处编辑 | `MultiEdit(src/**)` | `MultiEdit(.github/**)` |
| `Bash` | 执行 Shell 命令 | `Bash(npm run *)` | `Bash(rm -rf *)`, `Bash(git push *)` |
| `Grep` | 搜索文件内容 | `Grep`（裸名） | 极少被拒绝 |
| `Glob` | 按模式列出文件 | `Glob`（裸名） | 极少被拒绝 |
| `WebFetch` | 获取网页内容 | `WebFetch(domain:docs.*)` | `WebFetch(domain:internal.corp)` |
| `WebSearch` | 全网搜索 | `WebSearch`（裸名） | 极少被拒绝 |
| `NotebookEdit` | 编辑 Jupyter Notebook | `NotebookEdit(notebooks/**)` | 按项目定制 |

---

## 为什么 deny 说了算

![Claude Code 实操 (9)：settings.json、三层权限模型和环境 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/illustration_2.png)

只要合并后的配置中存在任一 deny 规则，该动作即被永久阻断——这是整个权限模型可信赖的核心。  
![图 6：按防护对象分类的 deny 规则速查，覆盖文件系统、git 历史、敏感文件、配置漂移四大类。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig6.png)

*图 6：按防护对象分类的 deny 规则速查，覆盖文件系统、Git 历史、敏感文件、配置漂移四大类。*

举个例子。某个仓库的 `.claude/settings.json` 写着：

```json
{ "permissions": { "deny": ["Bash(git push *)"] } }
```

队友加了个 `.claude/settings.local.json`，里面写 `"allow": ["Bash(git push origin main)"]`。这**不会**允许 push。项目层的 deny 胜出。这是正确的行为，你应该依赖它。

### 示例：用户级 `deny` 优先级最高

你的 `~/.claude/settings.json` 中配置：

```json
{ "permissions": { "deny": ["Read(.env*)", "Read(**/secret*)"]} }
```

则本机所有项目均无法读取 `.env` 文件或含 `secret` 的文件，无论其项目级配置如何。这是机器级安全策略。

### `deny-wins` 规则的实际含义

该不对称设计源于安全性考量：

- `allow` 是便捷性设置——用于跳过“是否允许？”提示；
- `deny` 是强制性策略——无论其他配置如何，均阻断操作。

组织可提交含 `deny` 规则的 `.claude/settings.json`。开发者无法覆盖这些拒绝规则。这是实现共享安全策略的核心机制。

---

## env —— 另一半配置

`env` 块为*每次*工具调用设置环境变量：

```json
{
  "env": {
    "NODE_ENV": "development",
    "PYTHONPATH": "./src",
    "DEBUG": "false"
  }
}
```

### `env` 变量影响范围

- **Bash 命令**：所有 `Bash` 工具调用均继承这些变量。例如 `NODE_ENV=development` 将在 Claude 执行 `npm test` 时生效。
- **Hook 脚本**：hook 作为子进程运行，继承全部环境变量。脚本可通过 `process.env.LOG_LEVEL` 读取。
- **不泄露至模型提示词**：模型无法看到任何环境变量值。此处是存放配置的安全位置。

### `env` 的层级优先级

本地层 > 项目层 > 用户层。因此 `.claude/settings.local.json` 中的 `DEBUG=true` 仅对你启用日志，且无需提交。

```json
// ~/.claude/settings.json（用户层）
{ "env": { "NODE_ENV": "development", "LOG_LEVEL": "warn" } }

// .claude/settings.json（项目层）
{ "env": { "NODE_ENV": "test", "CI": "false" } }

// .claude/settings.local.json（本地层）
{ "env": { "DEBUG": "true" } }

// 最终生效的 env：
// NODE_ENV=test        （项目层覆盖用户层）
// LOG_LEVEL=warn       （仅用户层存在，保留）
// CI=false             （仅项目层存在）
// DEBUG=true           （仅本地层存在）
```

### 实用 `env` 配置模式

**为工具设置 API 密钥：**

```json
{
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "DATABASE_URL": "postgresql://localhost:5432/dev"
  }
}
```

请将此类敏感配置放入 `settings.local.json`（已加入 `.gitignore`），避免意外提交。

**控制测试行为：**

```json
{
  "env": {
    "NODE_ENV": "test",
    "TEST_TIMEOUT": "30000",
    "SKIP_SLOW_TESTS": "true"
  }
}
```

**Python 路径配置：**

```json
{
  "env": {
    "PYTHONPATH": "./src:./lib",
    "VIRTUAL_ENV": "./.venv",
    "PATH": "./.venv/bin:${PATH}"
  }
}
```

## hooks —— 从同一个文件引用

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep",
        "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [{ "type": "command", "command": "node ./hooks/format.js" }]
      }
    ]
  }
}
```

匹配器是用管道符分隔的工具名称。所有三层里的 hooks 都会运行，不覆盖，只追加：任一层级新增 hook 均加入执行链末端，不会替换上层已定义的 hook。

### 钩子（Hook）配置详情

匹配器（matcher）为竖线分隔的工具名称。例如匹配器 `Read|Grep` 会在调用 `Read` 或 `Grep` 工具时触发。

**特殊匹配器 `*`**：匹配所有工具调用，适用于日志记录或可观测性钩子。

**每个匹配器可配置多个钩子**：按顺序执行。若任一 `PreToolUse` 钩子退出码为 `2`，则该工具调用被阻断，后续钩子不再执行。

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "node ./hooks/check-hours.js" },
          { "type": "command", "command": "node ./hooks/bash-blacklist.js" },
          { "type": "command", "command": "node ./hooks/bash-whitelist.js" }
        ]
      }
    ]
  }
}
```

### 钩子层级行为

三个配置层级（用户级、项目级、本地级）的钩子**累积生效**，而非覆盖。在更深层级添加钩子，仅向钩子链追加，不会替换上层已定义的钩子。

```json
// ~/.claude/settings.json（用户级）
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ~/hooks/global-blacklist.js" }] }] } }

// .claude/settings.json（项目级）
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ./hooks/project-blacklist.js" }] }] } }

// 两者均会运行：用户级 `global-blacklist.js` 先执行，随后执行项目级 `project-blacklist.js`。
```

这与权限（deny 优先）、env（深层覆盖）的行为截然不同：hooks 始终累加执行。

## 不同项目类型的 settings.json 模板

### Node.js / TypeScript 项目

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(docs/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "WebFetch(domain:docs.anthropic.com)",
      "WebFetch(domain:nodejs.org)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Edit(.github/workflows/**)",
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/credentials*)"
    ]
  },
  "env": {
    "NODE_ENV": "development",
    "CI": "false"
  },
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] },
      { "matcher": "Bash",      "hooks": [{ "type": "command", "command": "node ./hooks/bash-blacklist.js" }] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "node ./hooks/format.js" }] }
    ]
  }
}
```

注意三点：

1. Bash 白名单只包含了只读和可逆的 Git 命令，绝不含 `push`、`reset --hard` 或 `rebase`。push 操作必须由人工主动触发。
2. `Edit(.github/workflows/**)` 被 deny 了。CI 配置变更需要 review，我不想让它们混进普通 commit 里。
3. hooks 还为 deny 规则提供了一层额外保障：即使某条 deny 规则因拼写错误未生效，对应的 hook 仍可拦截危险操作。

### Python / 机器学习项目

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(notebooks/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "NotebookEdit(notebooks/**)",
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(pip install *)",
      "Bash(pip3 install *)",
      "Bash(pytest *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(pip install --user *)",
      "Bash(sudo *)",
      "Read(.env)",
      "Read(**/credentials*)",
      "Read(**/weights/*)",
      "Write(data/**)"
    ]
  },
  "env": {
    "PYTHONPATH": "./src",
    "VIRTUAL_ENV": "./.venv",
    "CUDA_VISIBLE_DEVICES": "0",
    "TOKENIZERS_PARALLELISM": "false"
  }
}
```

Python 特定设计说明：

- 允许对 `notebooks/` 目录使用 `NotebookEdit` —— Claude 可修改 Jupyter Notebook 文件；
- 拒绝 `Read(**/weights/*)` —— 模型权重文件体积庞大，读取无实际意义；
- 拒绝 `Write(data/**)` —— 数据文件不应由 Claude 修改；
- 设置 `CUDA_VISIBLE_DEVICES="0"` —— 防止开发过程中意外启用多 GPU。

### Rust 项目

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(benches/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Bash(cargo *)",
      "Bash(rustup *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(cargo publish *)",
      "Edit(Cargo.lock)",
      "Edit(.cargo/**)"
    ]
  },
  "env": {
    "RUST_BACKTRACE": "1",
    "RUST_LOG": "debug"
  }
}
```

Rust 特定设计说明：

- 拒绝 `cargo publish` —— 错误发布 crate 不可逆；
- 拒绝 `Edit(Cargo.lock)` —— lockfile 应通过 `cargo update` 更新，而非手动编辑；
- 设置 `RUST_BACKTRACE=1` —— 使 Claude 在测试失败时能获取完整回溯信息。

### 单体仓库（Monorepo）/ 多语言项目

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(packages/**)",
      "Write(packages/**)",
      "Bash(npm run *)",
      "Bash(npx *)",
      "Bash(pnpm *)",
      "Bash(turbo *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Edit(packages/*/package.json)",
      "Edit(.github/**)",
      "Edit(turbo.json)",
      "Read(.env*)",
      "Read(**/credentials*)"
    ],
    "additionalDirectories": [
      "../shared-configs"
    ]
  },
  "env": {
    "TURBO_TELEMETRY_DISABLED": "1"
  }
}
```

Monorepo 特定设计说明：

- 拒绝 `Edit(packages/*/package.json)` —— 依赖变更需显式、审慎操作；
- `additionalDirectories` 引入同级目录 `../shared-configs`，复用共享配置；
- 各子包可拥有独立的 `.claude/settings.json`，支持更宽松的细粒度规则。

---

## 权限问题排查

### “Claude 总是请求权限确认”

最常见问题。Claude 提示是因为该操作既未出现在 `allow` 中，也未出现在 `deny` 中，落入交互式提示流程。

**修复方法**：将操作加入 `allow`：

```json
// 修复前：Claude 每次运行测试都询问
// 修复后：
{ "permissions": { "allow": ["Bash(npm test)", "Bash(npm run test:*)"] } }
```

### “Claude 被阻断，但原因不明”

使用 `--debug` 启动 Claude，查看权限解析日志：

```bash
claude --debug
```

调试输出将明确显示每条规则的来源配置文件及最终匹配的规则。

### “我已允许某操作，却仍被阻止”

检查是否存在 `deny` 规则。注意：**deny 永远优先于 allow**。典型反例：

```json
// 此配置会阻止 `git push origin main`（尽管它出现在 allow 中）：
{
  "permissions": {
    "allow": ["Bash(git push origin main)"],
    "deny": ["Bash(git push *)"]
  }
}
```

`deny` 中的 `git push *` 匹配 `git push origin main`，因此生效。如需仅允许特定推送目标，应调整规则结构：

```json
// ❌ 下面写法无效（deny 永远胜出）
// ✅ 正确做法：使用钩子做精细化判断：
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{ "type": "command", "command": "node ./hooks/selective-push-gate.js" }]
    }]
  }
}
```

### “本地配置未生效”

验证文件名与路径是否正确：

- 文件名必须为 `.claude/settings.local.json`（非 `settings-local.json` 或 `local.settings.json`）；
- 必须位于仓库根目录下的 `.claude/` 子目录中；
- 必须为合法 JSON。

```bash
# 验证文件存在且格式有效
cat .claude/settings.local.json | jq .
```

### “用户级设置中的钩子未运行”

确保钩子脚本路径为绝对路径，或相对于当前工作目录正确。

```json
// 在 ~/.claude/settings.json 中，使用绝对路径：
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ~/hooks/global-blacklist.js" }] }] } }

// 在 .claude/settings.json（项目级）中，使用相对路径：
{ "hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "node ./hooks/project-blacklist.js" }] }] } }
```

## 优先级顺序，当作检查清单

当行为不符合预期时：

1. 在任何 `deny` 里吗？→ blocked，不管有没有 allow。
2. 在任何 `allow` 里吗？→ permitted。
3. 否则 → Claude 会在执行前询问。  
![图 4：一次工具调用按 deny -> allow -> prompt 的顺序逐步解析；deny 会短路后续所有判断。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/09-settings-and-permissions/fig4.png)

*图 4：一次工具调用按 deny → allow → prompt 的顺序逐步解析；deny 会短路后续所有判断。*

如果你想知道哪条规则赢了，加 `--debug` 运行并查看权限解析日志。它会准确告诉你哪个文件的哪一行做出了裁决。

### 各配置项优先级规则

| 配置项 | 优先级规则 |
|--------|------------|
| `permissions.deny` | 所有层级 `deny` 规则**并集生效**。**任一层级存在 deny 即阻断**。 |
| `permissions.allow` | 所有层级 `allow` 规则**并集生效**。任一层级允许即许可（除非被 deny 阻断）。 |
| `env` | **深层级覆盖浅层级**：Local > Project > User。 |
| `hooks` | **所有层级钩子累积执行**。全部运行，无覆盖。 |
| `worktree` | **深层级覆盖浅层级**。 |

### 合并过程可视化

```text
用户级设置         项目级设置           本地设置
~/.claude/         .claude/             .claude/
settings.json      settings.json        settings.local.json
     |                  |                     |
     v                  v                     v
  ┌──────┐         ┌──────┐             ┌──────┐
  │allow │ ─────→  │allow │ ─────→     │allow │
  │deny  │  UNION  │deny  │  UNION     │deny  │
  │env   │ ─────→  │env   │  OVERRIDE→ │env   │
  │hooks │ ─────→  │hooks │  ACCUMULATE→│hooks │
  └──────┘         └──────┘             └──────┘
                                              |
                                              v
                                       ┌────────────┐
                                       │  最终生效配置  │
                                       └────────────┘
```

### 常见场景速查表

| 需求 | 推荐配置位置 |
|------|--------------|
| “任何项目都禁止 X” | `~/.claude/settings.json` 中配置 `deny` |
| “本项目禁止 X” | `.claude/settings.json` 中配置 `deny`（提交至版本库） |
| “我个人跳过 X 的权限确认” | `.claude/settings.local.json` 中配置 `allow` |
| “仅对我启用 `DEBUG=true`” | `.claude/settings.local.json` 中配置 `env` |
| “全团队在保存时自动运行 Prettier” | `.claude/settings.json` 中配置 `hooks` |
| “我个人额外添加日志钩子” | `.claude/settings.local.json` 中配置 `hooks` |

## 从零构建 `settings.json`

新建项目时，推荐按以下步骤构建配置：

### 第一步：先设 `deny`（最小权限原则）

明确哪些操作**绝对不可发生**：

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Read(.env*)",
      "Read(**/credentials*)",
      "Read(**/secret*)"
    ]
  }
}
```

### 第二步：补充常用 `allow` 规则

将 Claude 频繁请求且你认为安全的操作加入白名单：

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ]
  }
}
```

### 第三步：配置 `env` 变量

声明开发环境必需的环境变量：

```json
{
  "env": {
    "NODE_ENV": "development"
  }
}
```

### 第四步：添加 `hooks`

定义需自动化执行的策略（如敏感文件读取拦截）：

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] }
    ]
  }
}
```

### 第五步：实测迭代

启动 Claude 实际使用，观察提示行为：
- 若频繁提示某安全操作 → 加入 `allow`；
- 若发生不期望行为 → 加入 `deny`；
- 持续微调，直至符合预期。

## 总结

`settings.json` 就是 Claude 在项目里能做什么的宪法。deny 规则应简短严苛，allow 规则须具体明确，hooks 作为兜底防护层。一旦你理解了层级和优先级的概念，配置一个新 repo 只需九十秒。在此之前，你可能会觉得规则很任意；其实不然。
