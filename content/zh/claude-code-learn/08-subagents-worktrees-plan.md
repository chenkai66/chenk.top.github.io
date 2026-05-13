---
title: "Claude Code 实战（八）：Sub-Agent 与计划模式"
date: 2026-04-25 09:00:00
tags:
  - claude-code
  - sub-agents
  - worktrees
  - plan-mode
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 8
description: "三个改变 Claude Code 一次能扛多少事的特性：子 Agent 用来并行调研、worktree 用来物理隔离、计划模式用在它真要动手之前。三者的边界，以及各自不该用的场景。"
disableNunjucks: true
translationKey: "claude-code-learn-8"
---
说完 Hooks，Claude Code 使用体验的下一个关键点是**并发控制**——这里的“并发”并非指线程层面的并发，而是指模型同时处理多少任务、任务之间如何隔离，以及你需要投入多少监督注意力。

这三个功能按所需信任度从低到高依次为：Plan mode、Worktrees 和 Sub-agents。

![Claude Code 实战（8）：子代理、工作树和计划模式 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

## Plan mode —— 气闸舱

Plan mode 是三者中成本最低的。按下 `Shift+Tab`，直到状态栏显示 **plan**。此时模型只做规划，不执行任何操作。它会读代码、思考、提出方案，然后停下来等你确认。你可以批准、修改，或者直接放弃。只有你点头后，它才会真正动手。

![Plan 模式生命周期：阅读、计划、批准、执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig5.png)  
*图 1. Plan mode 生命周期。意图与动作之间的气闸舱。*

### 如何启用 Plan mode

有三种方式：

1. **键盘快捷键**：`Shift+Tab` 在模式间循环切换：`auto` → `plan` → `auto`。当前模式会显示在底部状态栏。
2. **命令行参数**：运行 `claude --mode plan` 可直接启动一个 Plan mode 会话。
3. **对话中途切换**：任何时候按下 `Shift+Tab`，模式会立即生效。

### 一次真实的 Plan mode 实操演示

假设我正在开发一个 Express API，想加上限流功能：

```text
你：为 API 添加限流功能，要求每 IP 每分钟最多 100 次请求。

[Mode: plan]

Claude：以下是添加限流功能的完整计划：

1. 安装 `express-rate-limit`（它已在 `package.json` 中作为 devDependency 存在，但未列入 dependencies —— 我将把它移至 dependencies）。

2. 创建 `src/middleware/rate-limit.ts`：
   - 导出一个可配置的限流器工厂函数
   - 默认配置：基于 `req.ip`，每 IP 每分钟 100 次请求
   - 开发环境使用内存存储；备注生产环境应改用 Redis 存储

3. 在 `src/app.ts` 中接入限流中间件：
   - 导入限流器
   - 使用 `app.use()` 全局注册，置于路由处理器之前
   - 位置要求：CORS 中间件之后、body parser 之前

4. 在响应中添加限流相关 Header：
   - `X-RateLimit-Limit`
   - `X-RateLimit-Remaining`
   - `X-RateLimit-Reset`

5. 创建 `tests/middleware/rate-limit.test.ts`：
   - 测试：前 100 次请求被允许
   - 测试：第 101 次请求被拒绝
   - 测试：时间窗口过期后计数重置
   - 测试：不同 IP 地址拥有独立限流计数

6. 更新 `src/types/env.d.ts`，新增环境变量类型声明：
   - `RATE_LIMIT_MAX`
   - `RATE_LIMIT_WINDOW_MS`

是否确认执行此计划？
```

这时我能清楚看到：  
- 第 1 步就发现依赖包被错误地放在了 `devDependencies`；  
- 第 3 步明确指出中间件应该插在调用链的哪个位置；  
- 第 6 步还提醒我要补充环境变量的类型定义。  

如果没有 Plan mode，Claude 会直接开始写代码，这些细节只能等到 Code Review 阶段才被发现。而 Plan mode 强制我们在执行前对齐理解，避免返工。

### 我通常在这些场景使用 Plan mode：

- **任何非简单任务的前 30 秒**。比如“实现 X 功能”，先让它出个计划。几乎每次都能暴露模型对代码库的误解。
- **涉及认证、支付、数据库迁移或生产配置的操作**。花两秒看一眼计划，可能省下几小时的故障排查。
- **在不熟悉的仓库里工作**。这份计划本身就成了我的快速入门文档——它展示了模型眼中的系统架构。
- **重构前**。“重构模块 X” → 先看计划，它会画出依赖关系图。有时计划甚至告诉我：这重构根本不值得做。

### 常见误区

很多人觉得“任务很小，不用计划”，结果反而更容易出错。在一个错误的文件里改一行代码，比带着清晰计划做大功能更危险。

### Plan mode 修饰符

你可以通过指令约束计划的范围：

```text
你：[plan mode] 添加认证功能。聚焦于数据库 schema 变更。
     我已熟悉 API 层。

Claude：仅聚焦 schema 变更：

1. 新增 `sessions` 表……
2. 为 `users` 表添加 `last_login` 字段……
3. 生成迁移文件……
```

模型会严格遵守你的限制，避免生成动辄二十步的“长篇小说”。

### 批准、编辑与拒绝

Claude 输出计划后，你可以：

- **批准**：切回 `auto` 模式（再按一次 `Shift+Tab`），然后说 “go”，它就会执行计划。
- **编辑**：比如“整体不错，但跳过第 4 步，把第 3 步提到第 2 步前面。” 它会据此调整。
- **拒绝**：“算了，我们改做 Y 吧。” Plan mode 成本极低——你最多损失三十秒，而不是三十分钟。

## Sub-agents —— 适合并行执行的任务

![Sub-Agent 拓扑：上下文隔离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig3.png)  
*图 2. 父 Agent 启动多个子代理；每个子代理拥有独立的 context window 和工具子集。*

Sub-agent 是父 Agent 启动的一个独立 Claude Code 实例，专门处理边界清晰的子任务。它有自己的上下文窗口、工具集和指令。父 Agent 负责协调，子 Agent 负责干活。

### 定义一个 sub-agent

标准写法放在 `.claude/agents/<name>.md`：

```markdown
---
name: research
description: Reads a topic across the codebase and reports findings. No edits.
tools: Read, Grep, Glob, WebFetch
---

You are a research sub-agent. Your job:
1. Search the codebase for the requested topic.
2. Read enough files to understand it deeply.
3. Return a structured report with file paths and quotes.

Do not edit. Do not run shell commands. Stay focused.
```

### 更多 Agent 示例

**测试编写 Agent：**

```markdown
---
name: test-writer
description: 为指定模块编写测试。读取源码，生成测试，不修改源码。
tools: Read, Grep, Glob, Write, Bash
---

你是一个测试编写 sub-agent。你的任务：
1. 读取被测模块的源码；
2. 识别所有公开函数及其边界场景（edge cases）；
3. 按项目所用测试框架，编写全面的测试；
4. 运行测试，验证其通过。

规则：
- 禁止修改任何源文件，仅可创建或修改测试文件；
- 严格遵循项目中已有的测试模式与结构；
- 每个测试用例名称须具备描述性，清晰表达所覆盖的场景；
- 侧重覆盖边界场景，而非仅“快乐路径”（happy paths）。
```

**文档编写 Agent：**

```markdown
---
name: doc-writer
description: 为代码变更生成文档。读取源码与现有文档，输出 Markdown。
tools: Read, Grep, Glob, Write
---

你是一个文档编写 sub-agent。你的任务：
1. 读取被指定的代码变更或模块；
2. 读取现有文档，理解上下文、语气与风格；
3. 编写或更新文档，确保与项目整体风格一致。

规则：
- 严格匹配现有文档的语气（voice）与格式（format）；
- 每个公开 API 均须附带可运行的代码示例；
- 不执行任何命令（如 Bash）；
- 不修改源代码。
```

### 如何调用 sub-agent

在对话中，你可以直接指示 Claude 使用某个子代理：

```text
你：请使用 research agent，调研 payment 模块如何处理退款。

Claude：我将启动 research sub-agent，分析 payment/refund 流程。

[Sub-agent: research]
正在搜索与 refund 相关的代码……
找到 12 个含 refund 的文件：
  - src/payments/refund.ts（主处理逻辑）
  - src/payments/refund-validator.ts（校验逻辑）
  ...

[Sub-agent 报告返回至父 Agent]

Claude：根据 research sub-agent 的分析，该代码库中退款流程如下：……
```

### Sub-agent 带来的好处

**上下文隔离**：子代理拥有独立的上下文窗口，不会污染父 Agent 的主上下文。这对需要大量文件阅读的复杂任务尤其重要。

**工具限制**：比如 research agent 根本没有编辑文件的能力。这不是靠“自觉”，而是架构层面的安全设计——它压根不具备这个权限。

**并行执行**：当多个任务彼此独立时，你可以同时分发给多个子代理：

```text
你：我需要并行完成三件事：
     1. Research agent：该项目的认证（auth）机制是如何工作的？
     2. Test-writer agent：为 src/utils/validator.ts 编写测试
     3. Doc-writer agent：更新 `/users` 端点的 API 文档

Claude：正在并行启动三个 sub-agent...

[Sub-agent: research] 正在调研认证流程...
[Sub-agent: test-writer] 正在读取 validator.ts 并编写测试...
[Sub-agent: doc-writer] 正在读取 API 文档，并更新 `/users` 部分...

[三个 sub-agent 均已完成]

Claude：以下是三个 sub-agent 的结果：...
```

### Sub-agent 的代价

**Token 开销**：每个子代理都有自己的系统提示、上下文和对话历史。三个子代理做研究，消耗的 Token 大约是单个的三倍。

**协调成本**：父 Agent 必须手动合并结果。如果子代理 A 发现了 B 需要的信息，它们无法直接通信——你得显式安排这一步。

**启动延迟**：创建子代理需要时间。对于只需十秒就能完成的小任务，这点开销并不划算。

### 什么时候不该用 sub-agent

- **父 Agent 已经拥有全部上下文**。比如“去读一下这个文件然后汇报”，这种场景下启动子代理属于高开销的递归调用，不如直接读。
- **任务之间存在依赖**。如果 B 必须等 A 的输出，那就无法并行，应该在父 Agent 中顺序执行。
- **任务本身很小**。比如“修个拼写错误”，用 sub-agent 纯属画蛇添足。

## Worktrees —— 并行分支还不至于搞疯自己

![Claude Code 实战 (8)：子代理、工作树和计划模式 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_2.png)

Git worktree 是同一个仓库的另一个工作目录，对应不同分支，存放在不同路径。Claude Code 原生支持它：`EnterWorktree` 工具会自动创建新分支 + worktree，并将当前会话切换进去。

### Worktree 的底层机制

当 Claude 调用 `EnterWorktree` 时，会发生以下几步：

1. 在 `.claude/worktrees/<name>/` 下创建新目录；
2. 基于 `origin/main`（默认）或当前 HEAD（可配置）创建新分支；
3. 通过 `git worktree add` 将新目录关联到新分支；
4. Claude 的工作目录切换到该 worktree；
5. 此后的所有文件操作都在 worktree 中进行，**完全不影响原始仓库**。

关键在于：worktree 和原仓库**共享同一个 `.git` 对象存储**（commits、branches、refs 都是共享的），但**工作目录（磁盘上的文件）是完全独立的**。

![Worktree 文件系统布局：共享 .git，独立 working tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig4.png)  
*图 3. Worktrees 共享一个 `.git/` 对象库，但各自拥有独立的工作目录，分别绑定到不同分支。*

### 创建并进入 worktree

```text
你：创建一个名为 “refactor-auth” 的 worktree，并切换到它。

Claude：[EnterWorktree: name="refactor-auth"]
已创建 worktree，路径为 .claude/worktrees/refactor-auth/
已切换至新分支：refactor-auth
当前工作目录为：/path/to/repo/.claude/worktrees/refactor-auth/
```

此后所有编辑都发生在 worktree 中，原始工作目录毫发无损。

### 配置 base ref

默认情况下，worktree 从 `origin/<default-branch>`（通常是 `origin/main`）派生。如果你想从当前本地 HEAD 派生，可以通过 `worktree.baseRef` 设置：

- `fresh`（默认）：从 `origin/main` 派生  
- `head`：从当前本地 HEAD 派生

### 这些场景特别有用

- **中途切换上下文**：你正在 `feat/x` 上开发，突然需要在 `main` 上修一个无关的紧急 bug。新建一个 worktree，修完提交，退出即可，`feat/x` 的工作完全不受干扰。
- **尝试多种方案**：你想用两种不同方式解决同一个问题，又不想让废弃的提交污染主干。创建两个 worktree，分别尝试，保留胜出者，删掉失败者。
- **子代理物理隔离**：当你把任务委托给 sub-agent，又希望它完全隔离于你的主工作区，就给它分配一个 worktree。万一它搞砸了，直接删除 worktree，主目录毫无影响。

### 退出 worktree

有两种方式：

**保留 worktree：**  
```bash
# 列出所有 worktree 分支（排除当前检出分支）
git branch | grep -v "^\*"

# 合并该 worktree 的全部变更
git merge refactor-auth

# 或仅选取特定提交
git cherry-pick abc1234

# 清理：移除 worktree 并删除分支
git worktree remove .claude/worktrees/refactor-auth
git branch -d refactor-auth
```  
适用于工作未完成、需要稍后继续，或想先 review 再合并的情况。

**删除 worktree：**  
```text
你：[plan mode] 我想把 auth 逻辑抽离为独立包，请规划整体方案。

Claude：auth 抽离计划如下：

1. 调研阶段：
   - 扫描所有导入 `src/auth/` 的文件
   - 识别共享类型与接口（如 `User`、`Session`）
   - 定位覆盖 auth 逻辑的测试文件
   - 检查是否存在循环依赖

2. 抽离阶段：
   - 创建 `packages/auth/` 目录并初始化 `package.json`
   - 将 `src/auth/*.ts` 迁移至 `packages/auth/src/`
   - 在 `packages/auth/index.ts` 中定义公共 API 表面
   - 全量更新单体中对 `src/auth/` 的导入路径

3. 兼容阶段：
   - 在 `src/auth/index.ts` 添加 re-export shim（透传导出）
   - 验证运行时行为无变更
   - 运行完整测试套件

4. 清理阶段：
   - 移除 re-export shim
   - 更新文档
   - 同步 CI 配置

风险：auth 模块与 user 模块存在循环依赖。  
缓解措施：优先抽离共享类型。

是否继续？
```  
适用于任务已完成或已放弃。如果 worktree 中有未提交的更改，除非你明确设置 `discard_changes: true`，否则删除操作会被拒绝——这是正确的保护机制，不要试图绕过。

### 将 worktree 的改动合并回来

用 `keep` 退出后，对应的分支仍然存在，你可以像普通分支一样处理它：

```text
你：为此任务创建名为 `extract-auth` 的 git worktree。

Claude：[EnterWorktree: name="extract-auth"]
已基于分支 `extract-auth` 创建 worktree。
当前路径：`.claude/worktrees/extract-auth/`
```

### Worktree 注意事项（Gotchas）

**同一分支不能有多个 worktree**。Git 明确禁止这一点。如果尝试为已有 worktree 的分支再创建一个，操作会失败。解决办法：使用不同的分支名。

**所有 worktree 共享 Git 锁**。某些 Git 操作（如 `gc`）会锁定整个仓库。如果一个 worktree 正在执行 `gc`，其他 worktree 的同类操作会被阻塞。虽然实践中很少遇到，但可能导致难以排查的卡顿。

**IDE 支持参差不齐**。有些编辑器能自动识别 worktree 并正确调整上下文，有些则会混淆“哪个目录才是真正的 repo 根”。VS Code 对 worktree 支持良好；其他编辑器建议先验证兼容性。

![三道信任门：自主性与成本逐级递增](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig6.png)  
*图 4. 任务值得时再向上爬阶梯。*

## 组合这三者

真正的威力在于组合使用。下面是一个真实场景，同时用到了这三项功能。

### 场景：重构单体应用的 auth 模块

任务：将 Express 单体应用中的认证逻辑抽离成一个独立的中间件包。涉及 15+ 个文件、新包结构、测试迁移，以及保持向后兼容的导入路径。

#### 第一步：Plan mode

```text
你：调用 research sub-agent，绘制 auth 的全部依赖关系图。

Claude：启动 research sub-agent...

[Sub-agent: research]
共发现 23 个文件导入 `src/auth/`：
  - `src/routes/api.ts`（导入：`authenticate`, `requireRole`）
  - `src/routes/admin.ts`（导入：`requireAdmin`）
  - `src/middleware/session.ts`（导入：`SessionConfig` 类型）
  ...

检测到循环依赖：
  `src/auth/user-lookup.ts` → 导入 `src/users/types.ts`  
  `src/users/service.ts` → 导入 `src/auth/types.ts`

需优先抽离的共享类型：
  - `User`（被 14 个文件使用）
  - `Session`（被 8 个文件使用）
  - `AuthConfig`（被 3 个文件使用）
```

我审阅计划，发现循环依赖的风险确实存在——这是我之前没考虑到的。于是我批准执行，但加了个条件：“先做调研阶段，之后我们再一起复核计划。”

#### 第二步：Worktree

```text
你：开始抽离：先共享类型，再 auth 模块，最后更新导入路径。

Claude：[Working in worktree extract-auth]

Commit 1：`extract: shared types to packages/shared-types/`  
Commit 2：`extract: auth module to packages/auth/`  
Commit 3：`update: imports in monolith to use packages/auth`  
Commit 4：`add: re-export shim for backward compatibility`  
Commit 5：`test: all 47 auth tests passing in new location`
```

现在所有操作都在隔离环境中进行。万一抽离失败，只需删除 worktree，主分支完全不受影响。

#### 第三步：Sub-agent 执行调研

```text
你：测试全部通过。退出 worktree，并合并至 main。

Claude：[ExitWorktree: action="keep"]
已返回主工作目录。

`extract-auth` 分支含 5 个 commit。  
合并命令：`git merge extract-auth`
```

调研结果证实了循环依赖的存在。现在我知道：必须先把共享类型抽出来。

#### 第四步：在 worktree 中实施

Claude 在 `extract-auth` worktree 中完整执行抽离，每一步都生成独立的 commit：

![Claude Code 实战（8）：子代理、工作树和计划模式 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

#### 第五步：退出并合并

![Plan 模式生命周期：阅读、计划、批准、执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig5.png)

整个重构过程全程隔离。如果中途出错，我随时可以用 `remove` 彻底回退，主分支零污染。

---

## 三重信任门控

可以把它们看作逐级提升的信任阶梯：

| 功能 | 信任等级 | 你放弃的 | 你获得的 |
|------|----------|------------|------------|
| Plan mode | 低 —— 你在任何操作前先审阅计划 | 时间（阅读计划） | 对方案正确性的信心 |
| Worktrees | 中 —— 编辑确实发生，但被隔离 | 磁盘空间、分支管理开销 | 可随时丢弃失败实验的能力 |
| Sub-agents | 高 —— 并行工作，但监督有限 | Token 消耗、协调开销 | 独立任务的执行速度 |

常见错误是一上来就对所有任务启用 sub-agents。正确的做法是：  
✅ 先用 Plan mode；  
✅ 当你确认方案可行、只需要隔离时，再用 worktree；  
✅ 只有任务真正独立（无依赖、可并行）时，才启用 sub-agents。

![决策树：plain / plan / worktree / sub-agent 如何选](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/08-subagents-worktrees-plan/fig7.png)  
*图 5. 决策树：先问最简单的问题，被迫时才升级。*

## 局限性及适用场景说明

### Plan mode 局限性

- **非持久化**：计划只存在于当前对话中。会话结束后计划就没了。对关键计划，记得手动复制到文件（比如 `CLAUDE.md`）保存。
- **无强制约束力**：Claude 在执行时可能偏离计划。虽然通常能遵守，但在边缘情况或出错时可能发生 drift。务必对照原始计划复核最终结果。
- **引入延迟**：对简单任务来说，Plan mode 是冗余开销。比如“修复第 42 行的拼写错误”这种指令，根本不需要计划。

### Sub-agent 局限性

- **无法跨代理通信**：Sub-agent 之间不能直接对话。如果 A 发现了 B 需要的信息，必须由父 Agent 显式中转。
- **无共享状态**：每个 sub-agent 都从干净上下文启动，不知道其他代理做过什么。
- **Token 开销大**：三个 sub-agent 消耗大约三倍的 Token。探索性任务可以接受，但常规任务就是浪费。
- **不保证一致性**：如果多个 sub-agent 同时修改同一个文件，必然产生冲突。重叠操作应改用 worktree 隔离，或改为串行执行。

### Worktree 局限性

- **单分支单 worktree**：同一分支不能绑定多个 worktree。请提前规划好分支命名。
- **占用磁盘空间**：每个 worktree 都是完整的工作目录副本（虽然 `.git/objects` 可共享）。大型仓库中可能消耗大量磁盘。
- **需要主动清理**：遗忘的 worktree 会不断累积。建议定期运行 `git worktree list` 查看现存 worktree。
- **不能替代规范分支流程**：如果你需要长期并行开发三个特性，用三个终端分别 checkout 不同分支，通常比依赖三个 worktree 更合理。

## 什么时候都不用

大多数任务。真的。80% 的场景就是“改这个函数，跑测试，发布”——用 plain mode 就够了，不需要 sub-agent，也不需要 worktree。上面这些功能，是专门为那 20% 的大型、不可逆或高度分支化的任务准备的。

如果你发现自己每个任务都想用 sub-agent 和 worktree，或许该反思的是：是不是把任务拆得太大了？

### 快速参考

| 场景 | 推荐方式 |
|------|----------|
| 简单 Bug 修复 | Plain mode |
| 跨多文件的功能开发 | Plan mode → auto |
| 高风险重构 | Plan mode + worktree |
| 独立调研任务 | Sub-agent |
| 尝试两种不同方案 | 两个 worktree |
| 大型任务且含多个独立子任务 | Worktree + sub-agents |
| “修复第 42 行的拼写错误” | 直接执行即可 |
