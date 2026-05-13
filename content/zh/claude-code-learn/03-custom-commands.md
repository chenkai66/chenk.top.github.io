---
title: "Claude Code 实战指南（三）：自定义斜杠命令与对话控制"
date: 2026-04-20 09:00:00
tags:
  - claude-code
  - slash-commands
  - Workflow
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战指南"
series_order: 3
description: "斜杠命令将重复性工作流压缩为一行调用；$ARGUMENTS 让它支持参数化；而真正优秀的命令，会自然演变为团队共享的协作语言。"
disableNunjucks: true
translationKey: "claude-code-learn-3"
---
`/clear` 和 `/init` 这类内置斜杠命令，只是冰山一角。整个系统的核心设计，正是让你**亲手编写属于自己的命令**——它们就存放在你的代码仓库里。

![Claude Code 实战指南（三）：自定义斜杠命令与对话控制 — 示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_1.png)

## 什么是斜杠命令？

一个位于 `.claude/commands/<name>.md` 的 Markdown 文件。文件内容即为向 Claude 发送的提示词（prompt），文件名就是命令名。创建后需重启 Claude Code（这是少数几个不支持热重载的功能）。

最简示例：新建 `.claude/commands/audit.md`：

```markdown
运行 `npm audit` 查找已安装包中的安全漏洞。
运行 `npm audit fix` 应用非破坏性修复。
运行 `npm test` 确认修复未引入新问题。
报告哪些 CVE 已被修复、哪些仍待处理。
```

重启后，在任意会话中输入：

```text
/audit
```

整段提示词会立即执行，你将直接获得一份结构清晰的安全审计报告，无需手动回忆和逐条输入命令。

注意两点：

1. **命令本质就是 prompt**：没有 DSL，没有特殊语法，极简表面积，极大灵活性。
2. **一次编写，全员复用**：下次团队中任何人需要做安全审计，只需输入 `/audit`。

### 斜杠命令的内部工作机制

当你输入 `/audit` 时，Claude Code 会执行以下步骤：

1. 在当前项目中查找 `.claude/commands/audit.md`  
2. 同时检查 `~/.claude/commands/audit.md`（个人全局命令目录）  
3. 读取文件内容  
4. 将其作为 prompt 发送给 Claude，效果等同于你亲手输入了这段文字  

**项目级命令的优先级高于全局命令**。这意味着团队可直接在项目中覆盖你个人的 `/audit`，提供更贴合当前项目的版本。

命令内容就是一段普通 prompt，因此支持以下能力：

- 使用 `@` 引用项目内文件  
- 插入思考层级指令（如 “think a lot”）  
- 通过 `$ARGUMENTS` 实现参数化  
- 编写多步骤复杂指令  
- 完整保留 Markdown 格式（加粗、列表、表格等）

![斜杠命令查找顺序：项目命令优先、其次用户命令、最后内置命令](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/fig3.png)
*图：Claude Code 解析斜杠命令时的查找顺序——先项目级，再用户级，最后内置命令。*

## 目录结构规范

一个成熟项目的 `.claude/` 目录结构如下：

```text
.claude/
├── commands/               # 项目级命令（提交至 Git）
│   ├── audit.md
│   ├── review.md
│   ├── test.md
│   ├── deploy.md
│   ├── explain.md
│   ├── document.md
│   ├── onboard.md
│   └── debug.md
├── settings.json           # 项目配置（提交至 Git）
├── settings.local.json     # 个人配置（.gitignore 排除）
└── CLAUDE.md               # 项目记忆文件（建议提交至 Git，或置于仓库根目录）
```

个人全局命令则存放于：

```text
~/.claude/
├── commands/               # 个人全局命令
│   ├── standup.md
│   ├── changelog.md
│   └── quicktest.md
├── settings.json           # 全局配置
└── auth.json               # 认证 token
```

**两者的核心区别在于作用域**：  
✅ 团队命令 → 放在项目 `.claude/commands/` 下 → 仅对该仓库生效  
✅ 个人命令 → 放在 `~/.claude/commands/` 下 → 在所有项目中均可使用  

### 命名规范：短、直、准

命令名即斜杠后的字符串，应简洁明确：

| 推荐 | 不推荐 | 原因 |
|------|--------|------|
| `/review` | `/code-review-for-pr` | 过长，影响输入效率 |
| `/test` | `/run-all-tests-and-report` | 冗余，违背“一词一意”原则 |
| `/deploy` | `/deploy-to-staging-env` | 环境应通过 `$ARGUMENTS` 指定，而非硬编码进命令名 |
| `/explain` | `/e` | 对队友不友好，缺乏可发现性 |
| `/debug` | `/dbg` | 新成员无法直观理解含义 |

**黄金标准**：单个英文单词，4–8 字符，团队新人第一次见就能猜中用途。如果有人看到 `/review` 却想不到是“代码评审”，那命名就失败了。

🚫 避免在名称中加入版本号（如 `/review-v2`、`/test-new`）。需要升级时，直接替换原文件即可。Git 历史会自动保留旧版本，随时可回溯。

## `$ARGUMENTS`：让命令支持参数化

![Claude Code 实战指南（三）：自定义斜杠命令与对话控制 — 参数化示意](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/illustration_2.png)

斜杠命令内置一个魔法变量 `$ARGUMENTS` —— 它会被自动替换成你在命令名后输入的全部内容。例如，`.claude/commands/explain.md`：

```markdown
请从三个层面解释 $ARGUMENTS：

1. **一句话定义**：用大白话说明它是什么，禁用术语。
2. **一段原理**：它如何工作？核心组件有哪些？为何存在？可辅以类比。
3. **代码级定位**：指出它在本仓库中的具体实现位置（含文件路径与行号）。

若该术语在本代码库中存在歧义，请先列出可能含义，并询问我具体想了解哪一种。
```

然后输入：

```text
/explain rate limiter
```

`$ARGUMENTS` 即被替换为 `rate limiter`，Claude 将基于真实代码库生成一份三层嵌套的精准解释。

![$ARGUMENTS 替换流程：命令文件 + 用户输入 -> 最终发送给 Claude 的 prompt](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/fig4.png)
*图：`$ARGUMENTS` 仅做纯字符串替换——命令名后的全部内容会原样拼入 prompt。*

### `$ARGUMENTS` 的典型用法模式

`$ARGUMENTS` 本质是纯字符串替换 —— 命令名后所有内容原样填入。这种简单性正是其优势所在。以下是几种经实战验证的高效模式：

**单参数：标识符或名称**

```markdown
<!-- .claude/commands/find.md -->
在代码库中查找所有对 $ARGUMENTS 的引用。
按以下三类分组呈现：
- 直接调用
- 重新导出（re-export）
- 测试中使用

每处引用均标注文件路径与行号。
```

```text
/find UserService
```

**多参数：自然语言句子**

```markdown
<!-- .claude/commands/compare.md -->
对比 $ARGUMENTS。
以表格形式呈现关键差异，包含：
- API 表面（API surface）
- 性能影响（perf impact）

最后给出建议：针对本项目，应选用哪一个？为什么？
```

```text
/compare axios vs fetch vs ky for HTTP requests
```

`/compare` 后的所有内容（`axios vs fetch vs ky for HTTP requests`）整体成为 `$ARGUMENTS`，Claude 将其视为一条自然语言指令处理。

**参数为文件路径**

```markdown
<!-- .claude/commands/coverage.md -->
分析 $ARGUMENTS 模块的测试覆盖率。

1. 找出测试该模块的测试文件；
2. 列出该模块导出的所有函数/类；
3. 对每个导出项，标明是否已有测试覆盖；
4. 对未覆盖的函数，生成测试桩（stub）。

⚠️ 不得修改现有测试，仅允许新增。
```

```text
/coverage src/services/payment.ts
```

**无参数：默认行为兜底**

若用户只输入 `/coverage` 而未带任何后续内容，`$ARGUMENTS` 将为空字符串。请在 prompt 中优雅处理：

```markdown
<!-- .claude/commands/status.md -->
展示项目当前状态。

若指定了具体领域：$ARGUMENTS → 聚焦该领域分析；
否则 → 提供全局概览：
1. Git 状态（未提交变更、当前分支、落后/超前提交数）
2. 最近 5 条提交（每条一行）
3. 是否存在失败的测试
4. 是否存在 lint 错误
5. 近一周新增的 TODO 注释
```

```text
/status                    # 全局概览
/status authentication     # 聚焦鉴权模块
```

这种“有参则聚焦，无参则全量”的设计，让一条命令同时满足通用与专用场景，无需拆分成多个命令。

## 我在每个项目中必配的核心命令集

经过两年的实践沉淀，我提炼出一套高复用率的命令模板，以下是完整的 prompt 内容：

### `/audit` —— 安全审计

```markdown
运行 `npm audit`（或本项目实际包管理器的等效命令），
查找已安装依赖中的安全漏洞。

输出报告需包含：
1. 按严重等级（critical / high / moderate / low）统计漏洞总数；
2. 对每个 critical/high 级别漏洞：列出包名、CVE 编号、是否存在可用修复；
3. 对非破坏性更新，自动执行修复命令；
4. 运行测试确认修复未引发回归；
5. 列出仍需人工干预的剩余漏洞。

若项目不使用 npm，请根据 `CLAUDE.md` 中的指引，适配 pip / cargo 等对应工具。
```

### `/test` —— 测试运行与分析

```markdown
运行项目测试套件（具体命令见 `CLAUDE.md`）。

对每个失败用例：
  - 引用失败的测试名及断言；
  - 展示相关代码（含测试代码与被测代码）；
  - 提出最小可行修复方案；
  - 明确标注应修改测试还是业务代码；
  - 给出置信度评级：high / medium / low。

最终汇总：
  - 总计：X 通过，Y 失败，Z 跳过；
  - 若全部通过，仅输出 “All tests pass” —— 无需冗余格式。

⚠️ 本命令仅作诊断报告，禁止实际修改代码。
```

### `/review` —— 代码评审

```markdown
评审暂存区变更（`git diff --staged`）。

若暂存区为空，则评审工作区变更（`git diff`）；
若无任何变更，则直接说明并终止。

评审重点（按优先级排序）：
1. **正确性**：是否会引发崩溃或逻辑错误？
2. **边界情况**：哪些输入/状态未被处理？
3. **命名规范**：变量/函数名是否清晰、一致？
4. **`CLAUDE.md` 约定**：是否遵循项目约定？
5. **性能**：是否存在明显 N+1 查询、冗余循环等？
6. **安全**：是否有注入、权限绕过、数据泄露风险？

输出格式要求：
- 编号条目式呈现；
- 每条标注严重等级：🔴 必须修复 / 🟡 建议修复 / 🟢 微小建议 / 👍 值得表扬；
- 结尾附一段总结：此 PR 是否可直接合并？

Think a lot before responding.
```

### `/explain` —— 三层解释法

```markdown
请从三个层面解释 $ARGUMENTS：

1. **一句话定义**：用大白话说明它是什么，禁用术语。
2. **一段原理**：它如何工作？核心组件有哪些？为何存在？可辅以类比。
3. **代码级定位**：指出它在本仓库中的具体实现位置（含文件路径与行号），并梳理关键逻辑。

若该术语在本代码库中存在歧义，请先列出可能含义，并询问我具体想了解哪一种。
若代码库中未实现该概念，请作通用解释，并注明 “未在本项目中找到实现”。
```

### `/onboard` —— 新工程师入职指南

```markdown
为即将加入本仓库的新工程师撰写一份一页纸的入职文档。

以 `CLAUDE.md` 为唯一权威来源，辅以关键文件阅读。

必须包含：
1. **功能定位**：一句话讲清本项目是做什么的（禁用黑话）；
2. **本地搭建**：从 clone 到成功运行的完整步骤；
3. **运行测试**：精确命令与预期输出；
4. **架构概览**：最重要的 3–5 个目录及其职责；
5. **高频雷区**：新人最容易踩坑的三件事；
6. **调试起点**：遇到问题时，第一步该看哪里？
7. **关键联系人**：各模块负责人（若 `CLAUDE.md` 中有记录）。

严格控制在一页内，拒绝注水。目标：新人 10 分钟内通读，即可上手开发。
```

### `/deploy` —— 部署前检查清单

```markdown
准备部署至 $ARGUMENTS（默认为 staging）。

检查清单：
1. 测试全部通过；
2. Lint 无错误（警告可接受）；
3. 无未提交变更（如有则预警）；
4. 当前分支为 main/master（否则预警）；
5. 展示本次将部署的提交（对比上次部署 tag）；
6. 检查变更文件中是否存在 TODO / FIXME 注释；
7. 核对环境变量是否完备（对比 `.env.example` 与 `.env`）。

结尾输出明确的 “Go / No-Go” 决策摘要。

⚠️ 本命令仅为预检，禁止实际执行部署。
```

### `/debug` —— 结构化调试

```markdown
请结构化调试以下问题：

$ARGUMENTS

执行流程：
1. **复现**：明确触发该问题的精确步骤或条件；
2. **定位**：找出相关代码。从报错信息、堆栈追踪或描述中提到的模块名入手；
3. **假设**：列出 2–3 种可能原因，按可能性排序；
4. **验证**：对最可能原因，追踪代码路径并确认；
5. **修复**：提出最小改动方案，以 diff 形式呈现；
6. **验证**：说明如何测试该修复是否生效。

⚠️ 禁止实际应用修复，仅提供待审阅方案。
Think a lot before responding.
```

### `/document` —— 自动生成文档

```markdown
为 $ARGUMENTS 生成文档。

若为文件路径 → 文档化该文件的导出项（函数/类/类型）；  
若为模块名 → 文档化该模块的公共 API；  
若为概念 → 解释其在本代码库中的实现方式。

对每个函数/方法，需包含：
- 一句话简介（单行）；
- 参数（含类型与说明）；
- 返回值；
- 示例用法；
- 边界情况或注意事项。

输出格式为可直接粘贴到代码中的 JSDoc/docstring 注释，风格需与项目现有文档保持一致。
```

## 构建团队级命令库

斜杠命令是**传播团队协作规范最轻量、最有效的方式**，落地流程极其简单：

1. 编写一个实用命令；  
2. 放入 `.claude/commands/`；  
3. 提交至主干分支；  
4. **无需通知任何人**。

第 4 步并非夸张。下一位同事在该仓库中启动 Claude Code，输入 `/` 时，命令会自动出现在补全列表中。他尝试输入 `/audit`，结果完美运行 —— 无需培训文档、无需 Confluence 页面、无需入职会议。

### 命令库的自然演化路径

我观察过三个团队的命令库成长过程，规律高度一致：

**第 1–2 周**：一人添加 2–3 个基础命令，通常是 `/test`、`/review` 及一个项目专属命令（如 `/api-docs`）。  
**第 3–4 周**：其他成员陆续发现并开始使用这些命令；有人主动补充 `/deploy`。  
**第 2 个月**：命令开始精细化迭代 —— `/review` 的评审维度更全面；有人因频繁提问而新增 `/explain`。  
**第 3 个月起**：命令库稳定在 5–10 个，新增极少，优化为主。此时，这些命令已成为团队与 AI 协作的**事实标准语言**。

### 命令即文档：隐性的团队知识库

一个常被忽视的价值：**命令库本身就是团队工作流的活文档**。新人只需浏览 `.claude/commands/` 目录，即可快速掌握：

- 部署流程长什么样（`/deploy`）  
- 团队代码评审关注什么（`/review`）  
- 测试策略如何执行（`/test`）  
- 调试问题的标准路径（`/debug`）  

每个 `.md` 文件都是一份**可执行的流程说明书**。它比 Wiki 页面更有价值，因为永远与实践同步 —— 一旦过时，使用者当天就会更新它。

### 个人命令 vs 团队命令：边界必须清晰

![项目作用域 vs 用户作用域：位置、Git 跟踪、可见性、生命周期对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/fig5.png)
*图：项目命令随仓库流转，个人命令随机器流转。同名冲突时，项目命令胜出。*

| 类型 | 存放位置 | 是否提交 Git | 示例 |
|------|----------|--------------|------|
| 团队命令 | `.claude/commands/` | ✅ 是 | `/review`, `/test`, `/deploy` |
| 个人命令 | `~/.claude/commands/` | ❌ 否 | `/standup`, `/journal`, `/quickfix` |

个人命令服务于你的工作习惯。以下是我的两个常用命令：

**`~/.claude/commands/standup.md`:**

```markdown
帮我准备每日站会发言。

1. 昨日工作：执行 `git log --author="$(git config user.name)" --since="yesterday" --oneline`  
2. 今日计划：检查近期添加的 TODO 注释及当前打开的分支  
3. 阻塞问题：检查失败的测试或 lint 错误  

输出格式：三个项目符号，分别对应「昨日」「今日」「阻塞」。
```

**`~/.claude/commands/changelog.md`:**

```markdown
为上次 tag 至今的变更生成符合规范的更新日志。

1. 查找最近 git tag  
2. 列出该 tag 后的所有提交  
3. 按类别分组：Features / Fixes / Breaking Changes / Other  
4. 采用 keepachangelog.com 格式书写  
5. 基于变更内容建议下一个版本号（遵循 SemVer）

⚠️ 仅输出日志内容，禁止修改任何文件。
```

这些命令是“个人化”的：描述个人工作节奏，而非团队共识；另一位同事的站会格式可能完全不同。

## 高级命令模式

### 命令链式调用（组合其他命令）

一个命令可在 prompt 中直接引用其他命令名，实现逻辑复用：

```markdown
<!-- .claude/commands/morning.md -->
按顺序执行以下晨间检查：

1. 检查 Git 状态，确认昨日无未提交变更  
2. 从 main 分支拉取最新代码  
3. 运行测试套件（同 `/test` 行为）  
4. 列出昨日至今 main 分支新增的 TODO / FIXME 注释  
5. 总结：有何变更？有何问题？需关注什么？

这是我每日启动工作的完整快照，请一次性返回全部信息。
```

这并非技术上“调用子命令”，而是让 Claude 基于已读取的 `/test` 命令内容，复现其行为。效果等同于真实调用。

### 结构化输出命令

```markdown
<!-- .claude/commands/deps.md -->
分析 $ARGUMENTS 的依赖图谱。

输出结构化报告：

## 直接依赖
| 包名 | 版本 | 使用位置 | 用途 |
|------|------|----------|------|

## 重要传递依赖（需关注）
列出满足任一条件的传递依赖：
- 已知存在安全漏洞  
- 体积过大（>1MB）  
- 多版本共存（duplicated at different versions）

## 优化建议
- 哪些包可移除？  
- 哪些包应升级？  
- 哪些包有更优替代方案？
```

### 面向特定工作流的命令

**PR 提交前检查清单：**

```markdown
<!-- .claude/commands/pre-pr.md -->
我即将提交 PR，请执行以下检查：

1. [ ] 所有测试通过  
2. [ ] 无 lint 错误  
3. [ ] 无 TypeScript 错误（`npx tsc --noEmit`）  
4. [ ] 生产代码中无 `console.log`  
5. [ ] 所有新函数均有 docstring/JSDoc  
6. [ ] 无硬编码密钥或 API Key  
7. [ ] 若架构变更，`CLAUDE.md` 已同步更新  
8. [ ] 提交信息清晰明确  

对每项检查，明确标注通过/失败。  
若全部通过，基于提交信息草拟 PR 标题与描述。
```

**数据库迁移审查：**

```markdown
<!-- .claude/commands/migration-check.md -->
审查待执行的数据库迁移脚本。

检查要点：
1. **可逆性**：是否存在 down 迁移？能否正常执行？  
2. **数据丢失**：是否有步骤会删除含数据的列或表？  
3. **锁表风险**：是否有步骤会在大表上执行耗时操作（如 `ALTER TABLE`）？  
4. **索引影响**：新索引是否并发创建？  
5. **默认值**：新增的 `NOT NULL` 列是否为历史数据提供了默认值？  
6. **外键约束**：新增外键是否启用校验（validated）？

Think a lot. 数据库迁移不可逆。
```

## 命令调试指南

当命令未按预期工作时，按此流程排查：

**命令未出现在补全列表中？**  
→ 检查文件路径是否严格为 `.claude/commands/<name>.md`（注意复数 `commands`，且扩展名为 `.md`）；  
→ 创建后务必重启 Claude Code。

**命令能运行但输出异常？**  
→ 最常见原因是 prompt 表述模糊。将命令内容**直接粘贴为普通消息发送给 Claude**，测试其独立表现。若 prompt 本身效果差，作为命令必然更差。

**`$ARGUMENTS` 未被替换？**  
→ 确保你在命令后输入了内容（如 `/explain foo`）。仅输入 `/explain` 会导致 `$ARGUMENTS` 为空字符串。  
→ 若命令强制要求参数，可在文件顶部添加说明性 HTML 注释（不影响 prompt 执行）：

```markdown
<!-- .claude/commands/explain.md -->
<!-- 用法：/explain <术语或概念> -->

Explain $ARGUMENTS at three levels:
...
```

**命令过长？**  
→ 虽无硬性长度限制，但过长的 prompt 会挤占 Claude 的上下文空间。建议控制在 50 行以内。若需更多逻辑，说明你试图在一个命令中塞入过多职责——拆分为两个命令更合理。

**命令在 A 项目有效，在 B 项目失效？**  
→ 检查是否硬编码了项目特有细节（如 `npm test`、`src/api/`）。解决方案：  
 ✓ 用通用表述替代（如 “见 `CLAUDE.md` 中的测试命令”）；  
 ✓ 或坦然接受：这就是一个项目专属命令。

## 对话控制：三个必背内置命令

以下内置命令值得肌肉记忆：

**`/compact`** —— 汇总当前对话。当模型响应变慢时使用，保留核心信息，剔除冗余细节。（详见第二篇详解）  
**`/clear`** —— 清空当前对话。保留记忆与设置，适用于切换任务场景。  
**`/init`** —— （详见第一篇）首次在新仓库中运行，用于自动生成 `CLAUDE.md`。

其余命令可通过 `/help` 查看全量列表，但这三个是日常高频组合。

### 其他值得掌握的内置命令

| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `/help` | 列出所有可用命令 | 忘记命令名时 |
| `/compact` | 汇总并压缩上下文 | 长对话导致响应变慢 |
| `/clear` | 开启全新对话 | 切换任务主题 |
| `/init` | 生成 `CLAUDE.md` | 新仓库初始化 |
| `/config` | 打开设置面板 | 修改偏好设置 |
| `/cost` | 查看 token 消耗 | 监控使用成本 |
| `/doctor` | 诊断 Claude Code 异常 | 感觉功能异常时 |
| `/login` | 重新认证 | token 过期 |
| `/logout` | 清除认证信息 | 切换账号 |
| `/permissions` | 查看当前权限 | 调试 “为何它向我索要权限？” |

## 斜杠命令的适用边界

斜杠命令**不适合**以下场景：

- **需要复杂运行时参数**：`$ARGUMENTS` 仅支持单字符串参数。无法实现 `/deploy --env staging --skip-tests` 这类命名参数。此时应写 Shell 脚本，由 Claude 调用。  
- **需跨多次调用维持状态**：每次 `/command` 都在全新 prompt 上下文中执行，无法实现 `/step1` → `/step2` 的数据传递。多步有状态工作流，请用 SDK 或单个长 prompt。  
- **需要单元测试验证**：无法为斜杠命令编写测试用例。若工作流需强保障，应封装为脚本，由 Claude 调用而非执行。  
- **需按环境差异化执行**：命令文件对所有人相同。若 staging 与 production 步骤不同，请用 `$ARGUMENTS` 传入环境名，并在 prompt 中分支处理。

这些场景，请转向 **Claude Code SDK**（第六篇）。斜杠命令的本质，是为那些**不值得写代码的高频快捷操作**而生。

### Claude Code 自动化层级金字塔

```text
由简至繁：

1. CLAUDE.md 约定    （被动）→ Claude 自动遵循  
2. 斜杠命令          （单次）→ 输入 `/name`，立即获得结果  
3. Shell 脚本 + Claude（混合）→ 脚本调用 Claude，或 Claude 调用脚本  
4. Claude Code SDK   （编程）→ 全面可控、支持状态、可测试  
```

![自动化层级金字塔：CLAUDE.md、斜杠命令、Shell 脚本、SDK，由简至繁](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/fig6.png)
*图：四级自动化体系。绝大多数团队终其一生只需停留在第 1、2 层。*

**从第 1 层起步，仅当低层无法满足需求时，才向上跃迁。**  
绝大多数团队终身无需第 4 层；几乎所有团队都能从第 1–2 层显著受益。

## 真实命令库案例：一个生产级 API 项目

来看一个成熟 `.claude/commands/` 目录的实际结构：

```text
.claude/commands/
├── audit.md          # 安全审计
├── debug.md          # 结构化调试
├── deploy.md         # 部署前检查
├── deps.md           # 依赖分析
├── document.md       # 生成文档注释
├── explain.md        # 三层解释
├── migration-check.md # 数据库迁移审查
├── onboard.md        # 新人入职指南
├── pre-pr.md         # PR 提交前检查
├── review.md         # 代码评审
└── test.md           # 测试运行与分析
```

![成熟的 .claude/commands 目录按职责分组：质量门禁、知识沉淀、运维操作](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/03-custom-commands/fig7.png)
*图：11 条命令按意图分组。每个文件即一条工作流，共同构成团队的“AI 协作语言”。*

共 11 个命令。每一个，都曾是：  
✅ 手动执行的多步骤流程，或  
❌ 因过于繁琐而被长期忽略的环节。

创建总投入：分散在两个月内的约 2 小时。  
每周节省时间：保守估计每位开发者 30 分钟。  
5 人团队 = 每周 2.5 小时 → **首周即回本**。

真正的价值远不止时间节省 —— **是稳定性**。  
每一次代码评审，覆盖相同的维度；  
每一次调试，遵循相同的路径；  
每一次部署，走过相同的检查点。  
这些命令，已将团队的最佳实践，固化为可重复、可传承的工作流。

下一篇：MCP —— 让 Claude Code 与万物对话的协议。
