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
说完 hooks，Claude Code 使用体验发生显著变化的下一个关键点是*并发控制*。此处的‘并发’并非指线程级并发，而是指模型在同一时间为你并行处理多少任务、各任务间的上下文隔离程度如何，以及你需要投入多少监督注意力。

三个功能，按所需信任度从低到高排列。

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_1.png)

## Plan mode —— 气闸舱

Plan mode 成本最低。按 `Shift+Tab` 直到指示器显示 **plan**。这时候模型只规划不行动。它会阅读、思考、提出方案，然后停住。由你审阅该计划：可批准（approve）、修改（edit），或中止（kill）。只有这一步过了，它才会执行。



![Plan 模式生命周期：阅读、计划、批准、执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig5.png)
*图 1. Plan mode 生命周期。意图与动作之间的气闸舱。*

### 如何启用 plan 模式

共有三种方式：

1. **键盘快捷键**：按 `Shift+Tab` 循环切换模式：`auto` → `plan` → `auto`。当前模式显示在底部状态栏。
2. **CLI 参数**：运行 `claude --mode plan` 可直接以 plan 模式启动会话。
3. **对话中途切换**：任意时刻按下 `Shift+Tab` 即可立即切换模式。

### 一次真实的 plan 模式实操演示

下面以在 Express API 中添加限流功能为例，完整展示 plan 模式下的交互流程：

```
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

此时你能清晰看到：  
- 第 1 步已识别出依赖项位置异常（devDependency vs dependencies）；  
- 第 3 步明确定义了中间件在调用链中的精确插入位置；  
- 第 6 步提前提醒了环境变量类型定义的补充需求。  

若未启用 plan 模式，Claude 会直接开始编码，而这些关键上下文细节只能留待后续 Code Review 阶段才发现——plan 模式则确保它们在执行前就被显式对齐。

### 我一般在这些时候用


- 任何非 trivial 任务的前 30 秒。比如“实现 X 功能”→ 先 plan。几乎每次，计划都会暴露出模型对代码库的理解偏差。
- 任何涉及 auth、支付、schema 迁移或生产配置的操作。只需花两秒快速浏览，就可能避免数小时的故障修复。
- 在不熟悉的 repo 里干活。这份计划也自然成为我熟悉该仓库的入门参考。

常见误区：认为任务简单就跳过 plan mode。实际上，小任务反而更容易出现‘等等，这不是我想要的’这类偏差。

### Plan mode 修饰符

你可以通过指令引导 plan mode 的规划方向：

```
你：[plan mode] 添加认证功能。聚焦于数据库 schema 变更。
     我已熟悉 API 层。

Claude：仅聚焦 schema 变更：

1. 新增 `sessions` 表……
2. 为 `users` 表添加 `last_login` 字段……
3. 生成迁移文件……
```

模型严格遵守你设定的作用域约束，避免计划膨胀为长达 20 步的“小说式”方案。

### 批准、编辑与拒绝

Claude 输出计划后，你可选择：

- **批准（Approve）**：切回 `auto` 模式（`Shift+Tab`），并说 “go”。Claude 将执行该计划。  
- **编辑（Edit）**：例如，“整体不错，但跳过第 4 步，并将第 3 步提前到第 2 步之前。” Claude 会据此调整计划。  
- **拒绝（Reject）**：“算了，我们改做 Y 吧。” Plan mode 成本极低——你只损失约三十秒，而非三十分钟。

## Sub-agents —— 适合并行跑的任务



![Sub-Agent 拓扑：上下文隔离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig3.png)
*图 2. 父 Agent 启动多个子代理；每个子代理拥有独立的 context window 和工具子集。*

子代理是由父 agent 启动的一个独立 Claude Code 实例，用于处理边界明确的子任务。经典写法放在 `.claude/agents/<name>.md`：

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

然后在对话里说：“research 一下认证怎么工作的 → 用 research agent。”

### 更多 Agent 示例

**测试编写 sub-agent：**

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

**文档编写 sub-agent：**

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

### sub-agent 的调用方式

在对话中，你可以直接指示 Claude 启用某个 sub-agent：

```
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

这能换来什么：

- **上下文隔离。** 子代理有自己的 context window。父 agent 的保持干净。
- **工具限制。** research agent 确实不具备编辑能力——这是由系统架构保障的安全机制，而非依赖使用者自觉。
- **并行工作。** 任务独立时，你可以同时 fan out 给三个子代理。

当任务相互独立时，可并行分发至多个 sub-agent：

```
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

代价是什么：

- Tokens。每个子代理都有自己的 system prompt、自己的上下文、自己的来回对话。
- 协调。父 agent 得合并结果。这一步要显式规划好。

什么时候不该用子代理：父 agent 已经拥有所需上下文的任何任务。为‘读取单个文件并汇报’而 spawn 子代理，属于不必要的高开销递归调用。

## Worktrees —— 并行分支且不至于搞疯自己

![Claude Code Hands-On (8): Sub-Agents, Worktrees, and Plan Mode — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/illustration_2.png)

git worktree 是同一个 repo 的第二个 working tree，在不同分支，不同目录。Claude Code 认识这东西：`EnterWorktree` 工具会创建新分支 + worktree 并把 session 切换进去。

### worktree 的底层工作机制

当 Claude 调用 `EnterWorktree` 时，实际发生以下步骤：

1. 在 `.claude/worktrees/<name>/` 下创建一个新目录；  
2. 基于 `origin/main`（默认）或当前 HEAD（若已配置）新建一个 git 分支；  
3. 通过 `git worktree add` 将该新目录关联至该新分支；  
4. Claude 的工作目录切换至该 worktree；  
5. 此后所有文件操作均在该 worktree 中进行，**不再影响原始仓库**。

关键要点：该 worktree 与原始仓库**共享同一 `.git` 对象存储**（即 commits、branches、refs 完全共享），但**工作树（磁盘上的文件）是完全独立的**。



![Worktree 文件系统布局：共享 .git，独立 working tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig4.png)
*图 3. 多个 worktree 共享同一个 `.git/` 对象库，但各自拥有独立的工作目录，分别绑定到不同分支。*

### 创建并进入 worktree

```
你：创建一个名为 “refactor-auth” 的 worktree，并切换到它。

Claude：[EnterWorktree: name="refactor-auth"]
已创建 worktree，路径为 .claude/worktrees/refactor-auth/
已切换至新分支：refactor-auth
当前工作目录为：/path/to/repo/.claude/worktrees/refactor-auth/
```

此后所有文件编辑均发生在该 worktree 中，原始工作树保持不变。

### 配置 base ref

默认情况下，worktree 从 `origin/<default-branch>`（通常是 `origin/main`）派生。若希望改为从当前本地 HEAD 派生：

可通过 `worktree.baseRef` 设置控制：
- `fresh`（默认）：从 `origin/main` 派生  
- `head`：从当前本地 HEAD 派生

### 这些场景很关键


- 你正在 `feat/x` 上干活，用户突然要求在 `main` 上修个无关的 quick fix。新建一个 worktree，完成修复后提交（commit），再退出。
- 你想尝试同一问题的两种不同解法，又不想用废弃的 commit 弄脏主干。
- 你委托给子代理，希望它在物理上跟你的 working tree 隔离。

心智模型：worktree 是上下文隔离的一种*物理层面实现*：子代理通过会话隔离实现逻辑上下文隔离，worktree 则通过独立工作目录实现文件系统层面的隔离。

关于关于清理方式：

- `keep` — worktree 留在磁盘上。工作未完成或可能回来继续时用。
- `remove` — 彻底删除，分支也删掉。只有确定不再需要时才用。

如果 worktree 里有未 commit 的变更，除非你确认 `discard_changes: true`，否则拒绝删除。这是对的。别想着绕过它。

### 将 worktree 的工作合并回主分支

退出 `keep` 模式的 worktree 后，对应分支依然存在：

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

### 使用 worktree 的注意事项（Gotchas）

**同一分支不能拥有多个 worktree。** Git 显式禁止此操作。若尝试为已存在 worktree 的分支新建 worktree，命令将失败。解决方案：使用不同分支名。

**所有 worktree 共享 Git 仓库级锁（git lock）。** 某些 Git 操作（如 `git gc`）会锁定整个 repo。若一个 worktree 正在执行 `gc`，其他 worktree 的同类操作将被阻塞。实践中极少发生，但可能引发难以排查的挂起（hang）。

**IDE 集成支持程度不一。** 部分编辑器能自动识别 worktree 并正确调整上下文；另一些则可能混淆“哪个目录才是真正的 repo 根”。VS Code 对 worktree 支持良好；其他编辑器建议先验证兼容性。



![三道信任门：自主性与成本逐级递增](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig6.png)
*图 4. 任务值得时再向上爬阶梯。*

## 组合这三者

处理硬任务时我的模式：

1. **Plan mode。** “我要做这个；你打算怎么做？”看计划。调整。
2. **Worktree。** 进一个隔离分支，这样实验失败了也不至于弄脏主干。
3. **Sub-agents** 处理 worktree 内的独立子任务。先 research，再 implementation，最后写 test —— 每个都在自己的上下文里。
4. 回到父 agent，合并结果，commit，退出 worktree（工作暂停用 `keep`，完成了用 `remove`）。

### 场景：重构单体应用的 auth 模块

任务：将 Express 单体应用中的认证逻辑抽离为独立的中间件包。涉及 15+ 个文件、新建包结构、测试迁移，以及保持向后兼容的导入路径。

#### 第一步：Plan mode（计划模式）

```
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

我审阅该 plan。循环依赖风险真实存在——此前我并未意识到。我批准执行，但附加一项修改：“先完成调研阶段，之后我们再共同复核并优化 plan。”

#### 第二步：Worktree（工作树）

```
你：为此任务创建名为 `extract-auth` 的 git worktree。

Claude：[EnterWorktree: name="extract-auth"]
已基于分支 `extract-auth` 创建 worktree。
当前路径：`.claude/worktrees/extract-auth/`
```

此后所有操作均在隔离环境中进行。若抽离失败，只需删除该 worktree，主分支完全不受影响。

#### 第三步：Sub-agent（子 Agent）执行调研

```
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

调研结果证实了循环依赖的存在。现在我明确：必须先抽离 `shared-types`。

#### 第四步：在 worktree 中实施

Claude 在 `extract-auth` worktree 内完整执行抽离，每步生成独立 commit：

```
你：开始抽离：先共享类型，再 auth 模块，最后更新导入路径。

Claude：[Working in worktree extract-auth]

Commit 1：`extract: shared types to packages/shared-types/`  
Commit 2：`extract: auth module to packages/auth/`  
Commit 3：`update: imports in monolith to use packages/auth`  
Commit 4：`add: re-export shim for backward compatibility`  
Commit 5：`test: all 47 auth tests passing in new location`
```

#### 第五步：退出 worktree 并合并

```
你：测试全部通过。退出 worktree，并合并至 main。

Claude：[ExitWorktree: action="keep"]
已返回主工作目录。

`extract-auth` 分支含 5 个 commit。  
合并命令：`git merge extract-auth`
```

整场重构全程在隔离 worktree 中完成。任一环节出错，均可通过 `ExitWorktree: action="remove"` 彻底回退，零污染主分支。

三层信任门，三级 escalation。当模型进入文件修改阶段时，你所投入的注意力程度，恰好与任务复杂度相匹配。

| 功能 | 信任等级 | 你放弃的 | 你获得的 |
|------|----------|------------|------------|
| Plan mode | 低 — 你在任何操作执行前先审阅计划 | 时间（阅读计划） | 对方案正确性的信心 |
| Worktrees | 中 — 编辑确实发生，但被隔离在独立环境 | 磁盘空间、分支管理开销 | 可随时丢弃失败实验的能力 |
| Sub-agents | 高 — 并行工作，但监督有限 | Token 消耗、协调开销 | 独立任务的执行速度 |

常见错误是：一上来就对所有任务启用 sub-agents。正确路径是：  
✅ 先用 plan mode；  
✅ 当你已确认方案可行、仅需隔离时，再切到 worktree；  
✅ 仅当任务真正独立（无依赖、可并行）时，才启用 sub-agents。



![决策树：plain / plan / worktree / sub-agent 如何选](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/08-subagents-worktrees-plan/fig7.png)
*图 5. 决策树：先问最简单的问题，被迫时才升级。*

## 局限性及适用场景说明

### Plan 模式局限性

- **非持久化。** 计划仅存在于当前对话中；会话结束后计划即丢失。对关键计划，请手动复制到文件（如 `CLAUDE.md`）中保存。
- **无强制约束力。** Claude 在执行过程中可能偏离计划——通常能遵循，但在边界情况或出错时可能发生 drift。请务必对照原始计划复核最终结果。
- **引入额外延迟。** 对简单任务而言，Plan 模式属于冗余开销。例如「修复第 42 行的拼写错误」这类指令无需生成计划。

### Subagent 局限性

- **无跨 agent 通信能力。** Subagent 之间无法直接交互。若 Agent A 发现了 Agent B 所需的信息，必须由 parent agent 显式中转。
- **无共享状态。** 每个 subagent 均以全新 context 启动，彼此 unaware 其他 agent 已执行的操作。
- **Token 开销显著。** 启动 3 个 subagent 将消耗约 3 倍 tokens。适用于探索性任务，但对常规、确定性任务属资源浪费。
- **不保证一致性。** 若多个 subagent 并发修改同一文件，将引发冲突。重叠操作应改用 worktree 隔离或改为串行执行。

### Worktree 局限性

- **单分支单 worktree 约束。** 同一分支不可绑定多个 worktree。请提前规划好分支命名策略。
- **磁盘空间占用。** 每个 worktree 是完整工作树副本（`.git/objects` 可共享，其余文件独立）。大型仓库中易造成显著磁盘消耗。
- **需主动清理。** 遗忘的 worktree 会持续累积。建议定期运行 `git worktree list` 检查现存 worktree。
- **不能替代规范分支流程。** 若需长期并行开发三个特性，使用三个终端 + 独立分支通常比依赖三个 worktree 更合理。

## 什么时候都不用

大多数任务。说真的。80% 的情况就是“改这个函数，跑测试，发布”—— 普通模式，不要子代理，不要 worktrees。上面那些功能是为那 20% 庞大、不可逆或分支复杂的任务准备的。

如果你发现自己每个任务都想掏子代理和 worktree，更有趣的问题是：你是不是把任务拆得太大了。

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
