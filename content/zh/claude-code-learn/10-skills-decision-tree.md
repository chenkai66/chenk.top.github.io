---
title: "Claude Code 实战（十）：Skills 与四种扩展机制"
date: 2026-04-27 09:00:00
tags:
  - claude-code
  - skills
  - slash-commands
  - mcp
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 10
description: "Claude Code 现在有四种扩展机制：斜杠命令、MCP server、Hooks、Skills。Skill 是最新的——一个文件夹、一份 SKILL.md、一段按需加载的指令。它和其他三个怎么分工，给一棵决策树。"
disableNunjucks: true
translationKey: "claude-code-learn-10"
---
Claude Code 提供四种扩展机制：slash commands、MCP servers、hooks 和 skills，它们的功能存在交叉。当你想到‘Claude 应该知道怎么做 X’时，首要问题是：该选哪一种？

作为系列的终篇，本章将直接进入决策树。

![Claude Code 实操 (10)：技能及其适用的扩展机制 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_1.png)

## Skill 到底是什么

Skill 就是一个文件夹，位置在 `~/.claude/skills/<name>/`（用户级）或 `<repo>/.claude/skills/<name>/`（项目级）。里面至少得有个 `SKILL.md`。每个 SKILL.md 由两部分构成： frontmatter 与 body。

```markdown
---
name: chenk-blog-write
description: Use when writing new content for chenk.top — bilingual EN/ZH posts, series, tutorials. Covers front matter, voice, matplotlib figures, cover generation, deploy.
---

# Voice
- First person, dry, restrained. No "let's", no exclamations.
- One claim, one example. If the claim has no example, cut the claim.

# Front matter
[exact schema goes here]

# Workflow
1. Read source
2. Write EN
3. Adapt to ZH (not translate)
4. Generate covers
5. Build + deploy
```

会话启动时，Claude 会预先读取所有 skill 的 description，只有当用户的提问语义匹配某条 description 时，才会按需加载其 body 并注入当前系统提示。

需注意两点：

- `description` 是承重墙。如果没写清楚什么时候用， skill 就不会被触发。
- 正文可以较长——由于按需加载，只要未被触发，就不会占用任何上下文（context）资源。

![Skill 加载生命周期：description 始终在上下文中，正文仅在 description 与 prompt 匹配时才加载](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig3.png)

### Frontmatter 字段

| 字段 | 是否必需 | 说明 |
|------|----------|------|
| `name` | 是 | skill 的唯一标识符，用于日志与调试。请使用小写短横线格式（`lowercase-with-dashes`）。 |
| `description` | 是 | **核心字段**。 Claude 依据该描述决定是否加载此 skill。若描述中未明确说明“何时使用”，该 skill 将永远不会被触发。 |

`description` 是整个文件中最重要的字段。 Claude 在每次会话启动时，会扫描所有可用 skill 的 `description`；仅当你的 prompt 语义匹配某条描述时，才会加载对应 skill 的完整 body。若描述模糊， skill 要么泛化触发（对所有输入都加载），要么完全沉默（永不触发）——二者皆不可取。

### Body 内容

body 长度无硬性限制，仅在 skill 触发时注入 system prompt，因此冗长零开销：未触发即不计成本。我有多个 body 超过 600 行的 skill，每一行都物有所值。

Body 支持自由格式的 Markdown，但以下几类结构已被验证为高效：**语气/风格规范**（定义 Claude 在该 skill 激活时的表达方式）、**Schema/格式规范**（明确必须严格遵守的输出格式）、**工作流**（分步执行逻辑）、**规则/约束**（明确禁止行为）。

例如，工作流和规则可以这样组织：

```markdown
# Workflow
1. 读取 `[path]` 下的现有内容
2. 生成含 N 个章节的提纲
3. 逐章撰写正文
4. 运行 `build` 命令验证
5. 部署

# Rules
- 除非用户明确要求，否则不得创建文档类文件
- 禁止添加 emoji
- 必须使用绝对路径
- 保持文件整洁：禁止使用 `v1`/`v2` 等版本后缀
```

## Skill 目录结构

### 用户级 skill

```
~/.claude/
  skills/
    chenk-blog-write/
      SKILL.md
    update-config/
      SKILL.md
    simplify/
      SKILL.md
```

用户级 skill 在本机所有项目中均可用。

### 项目级 skill

```
my-project/
  .claude/
    skills/
      deploy-staging/
        SKILL.md
      code-review/
        SKILL.md
```

项目级 skill 仅在当前项目中可用，可提交至 Git 供团队共享。

### 选择层级的依据

| 场景 | 层级 | 原因 |
|------|------|------|
| 个人写作风格 | 用户级 | 在所有写作场景中通用 |
| 项目部署流程 | 项目级 | 与该代码库强耦合 |
| 代码审查偏好 | 用户级 | 反映个人风格，跨项目一致 |
| API 集成工作流 | 项目级 | 依赖特定 API 和约定 |
| 通用“简化代码”能力 | 用户级 | 具有普适性 |

## 从零编写一个真实 skill

以一个具体用例为例：为团队构建一个将 Node.js 应用部署到预发布（staging）环境的 skill。

### 步骤 1：识别“skill 形态”的使用场景

关键在于这是应在相关话题出现时自动触发的领域知识，还是需要显式调用的命令？

向 staging 部署是团队在 PR 合并后、功能测试时、排查生产问题等多种上下文中反复执行的操作——这不是孤立命令，而是关于“我们如何部署”的结构化知识，属于典型的 skill-shaped 场景。

### 步骤 2：撰写 description

先写描述，因为它决定了 skill 何时被触发：

```markdown
---
name: deploy-staging
description: Use when deploying to the staging environment, preparing a staging build, or troubleshooting staging deployment issues. Covers the deploy pipeline, environment config, health checks, and rollback procedures for the acme-api project.
---
```

务必具体。“Use for deployment”过于宽泛，会导致 skill 在讨论部署理论时误触发；而“Use when deploying to the staging environment”则精准锚定到实际操作。

### 步骤 3：撰写 body

一个真实的 `deploy-staging` skill 正文应包含以下要素：环境信息（staging URL、 region、 ECS cluster、默认部署分支）；预部署检查清单（测试通过、 TypeScript 编译成功、无未提交变更、当前分支正确）；部署命令流（构建 Docker 镜像 → 为 ECR 打标签 → 推送 → 更新 ECS service）；健康检查（等待 60 秒后调用 `/health` 端点）；回滚流程（定位上一版 task definition ARN，执行 `aws ecs update-service --task-definition ...`）；以及常见故障排查——部署超时查 CloudWatch、健康检查失败查依赖连通性、 ECR 推送失败用 `aws ecr get-login-password` 刷新凭证。

最后一定要有几条硬性规则：禁止从此 skill 直接部署至 production，每次部署后必须执行健康检查，若健康检查失败，立即回滚，不得在 staging 上调试。

### 步骤 4：测试

创建文件并启动 Claude 会话：

```bash
mkdir -p .claude/skills/deploy-staging
# (在 .claude/skills/deploy-staging/SKILL.md 中写入上述内容)

# 启动 Claude 并测试
claude
> Deploy the latest changes to staging.
```

Claude 应自动加载该 skill 并严格遵循其中步骤。若未触发，请检查 `description` 是否与你的自然语言表述匹配（例如你说了 "push to staging"，但描述中写的是 "deploy to staging"）。

### 步骤 5：持续迭代

经过数次使用后，你会发现遗漏点，逐条补入正文：

- “得注明必须先连接公司 VPN”；
- “ECR 登录步骤应前置并加粗强调”；
- “补充 Slack 通知步骤”。

每次迭代都让 skill 更完整。一周后，它将成为该部署流程唯一可信、可执行、可审计的权威文档。

## Skill vs slash command vs hook vs MCP — 完整对比

这是我初学时最希望拥有的对照表：

| 维度 | Slash command | MCP server | Hook | Skill |
|------|----------------|-------------|------|--------|
| **存放位置** | `.claude/commands/<name>.md` | `mcp.json` 配置文件 | `settings.json` | `.claude/skills/<name>/SKILL.md` |
| **加载时机** | 用户输入 `/<name>` 时 | 始终可用 | 工具调用前后自动触发 | prompt 匹配 description 时 |
| **触发方式** | 显式（用户主动输入） | 显式（Claude 主动调用工具） | 自动（工具调用时隐式触发） | 隐式（主题匹配即触发） |
| **适用场景** | 可复用的工作流 | 外部能力扩展 | 策略强制执行（如安全、审计） | 领域知识建模 |
| **能否编辑文件** | 是（通过 Claude） | 是（通过工具） | 是（可产生副作用） | 是（通过 Claude） |
| **能否阻断操作** | 否 | 否 | 是（返回退出码 `2` 即可中断） | 否 |
| **上下文开销** | 仅在调用时加载一次 | 工具描述始终驻留内存 | 极低（脚本在外部运行） | 触发时按需加载 |
| **是否可共享** | 是（提交至代码仓库） | 是（提交 `mcp.json`） | 是（提交 `settings.json` + 脚本） | 是（提交至代码仓库） |
| **编写复杂度** | 低（纯 Markdown） | 高（需编写服务端代码） | 中（Node.js 脚本） | 低（纯 Markdown） |
| **用户级可用性** | 否（仅限当前项目） | 是 | 是 | 是 |

![能力矩阵：四种扩展机制各自的强项、部分支持与不适用场景](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig2.png)

### 最清晰的边界划分

**Slash command 是指令， skill 是知识。** Slash command 表达“请执行这个确定动作”； skill 表达“这是我对某类问题的系统性理解，请在适用时自主调用”。

**Hook 是自动的，其余皆为显式启用。** Hook 无需任何授权即可在工具调用时强制运行——这使其成为安全策略、合规审计等强制场景的理想选择，但完全不适合可选工作流。

**MCP 提供能力，其余提供指导。** 只有 MCP 能赋予 Claude 原生不具备的新能力（如浏览器自动化、数据库访问、第三方 API 集成）。其余机制（slash command / hook / skill）均不扩展能力边界，而仅影响 Claude 如何使用其已有能力。

## Skill 的发现与加载

会话启动时， Claude 扫描 `~/.claude/skills/*/SKILL.md`（用户级）和 `.claude/skills/*/SKILL.md`（项目级），读取每个文件 frontmatter 中的 `description` 字段。这些描述始终在上下文中——它们是 system prompt 的一部分。

当你发送消息时， Claude 将 prompt 与所有 skill 的 description 进行语义匹配。若存在匹配项，则将对应 skill 的**完整正文**加载进当前 turn 的上下文：

- **描述始终加载**（开销小——仅 description 行本身）；
- **正文按需加载**（可能较大，但仅在相关时才加载）；
- **多个 skill 可同时触发**（若多个 description 匹配），可能导致上下文膨胀。

若某 skill 未按预期触发，按以下顺序排查：

1. **描述内容**：是否包含你 prompt 中的关键语义？ Claude 基于语义相似度匹配，但具体描述比模糊描述更易命中。
2. **文件路径**：`SKILL.md` 是否在正确目录？
3. **frontmatter 格式**： YAML 是否有效？`---` 分隔符是否存在？
4. 若必须靠名字调用，该功能更适合作为 slash command 实现——skill 的核心价值在于隐式加载。

## 条件化 skill

某些 skill 仅应在特定上下文触发。可通过 description 中的限定语句（如 "Only relevant in Python projects"）和 body 中的条件规则共同表达：

```markdown
---
name: strict-review
description: Use when reviewing code changes before commit. Applies strict standards.
---

# When to apply strict standards

Apply these standards ONLY when:
- The file is in `src/core/` or `src/security/`
- The change touches authentication, authorization, or payment logic
- The user explicitly asks for a strict review
```

这不是硬性强制（Claude 仍可能偏离），而是为模型提供清晰、可操作的触发边界。

## 什么时候用哪个

![Claude Code 实操 (10)：技能及其适用的扩展机制 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/10-skills-decision-tree/illustration_2.png)

按顺序过一遍：

**1. 任务需要尚不存在的工具吗？**（浏览器自动化、查真实数据库、调内部 API。）
→ 建一个 **MCP server**。其他机制给不了新能力。

**2. 工具调用前后需要自动发生点什么吗？**（拦截、验证、日志、格式化？）
→ 写一个 **hook**。这是唯一不需要模型决定调用就能运行的机制。

**3. 这是个紧凑的流程，用户会显式调用吗？**（`/commit`, `/deploy-staging`, `/make-changelog`）
→ 写一个 **slash command**。命令是用来点名调用的。

**4. 这是一堆领域知识吗？**（风格、工作流、一套规范）
→ 写一个 **skill**。 Skill 是用来让 Claude 自己*识别并应用*的，不用你特意喊它。

若某需求适配多种机制，优先选择实现最简单的一种。 Skill 内部调用 slash command 是合理设计，但将本应是 slash command 的功能强行包装为 skill 会导致触发不可靠、维护困难。

![决策树：按顺序走一遍，遇到第一个 YES 就停 —— MCP 加新能力、Hook 做自动策略、Slash command 显式调用、Skill 按话题自动加载](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig1.png)

## 我实际写过的三个 skill

**1. `chenk-blog-write`** — 给这个站点写文章用的。覆盖 front matter、风格、 EN/ZH parity、封面生成、部署。只要提到 chenk.top 或 "write a post" 就触发。正文大概 600 行。每一行都值。

**2. `update-config`** — 用来改 `~/.claude/settings.json` 的。触发词是 "allow X command," "set env Y," "add a hook." 编码了上面的权限优先级规则和典型模式。省得我去推 merge order。

**3. `simplify`** — 用来 review 我自己代码的。触发词是 "is there a simpler way to do this." 编码了我的品味：偏好组合，删死代码，命名看本质不看实现。

这几个都无法实现为 slash command——因为它们依赖话题触发，而非显式命名调用。这正是 skill 的适用场景。

## 什么时候 skill 不是正确答案

触发过于频繁的 skill 反而有害：它会将无关知识注入本无需该 skill 的任务上下文，造成干扰。三个坑：

![三种 Skill 反模式：描述模糊导致触发过频、正文重叠导致上下文膨胀、本该用 Hook 强制的规则被写成 Skill](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig4.png)

- **描述模糊。** “用于一般编程。” 用于一切 = 用于 nothing useful。
- **正文重叠。** 两个 skill 都响应 "write code" → context 膨胀。选一个。
- **本该是 Hook。** “总是在 Y 之前做 X” → 这是 hook，不是 skill。 Skill 是建议， hook 是强制。

判别标准：若指令中含 "always" 或 "never" 且应用于某类工具调用的每一次，那就是 hook；若含 “讨论 X 时” 或 “处理 Y 类任务时”，那就是 skill。

## 构建你自己的扩展库

随着时间推移，你会通过全部四种扩展机制（slash command、 hook、 MCP、 skill）积累大量扩展。以下是我组织它们的方式：

```
~/.claude/
  settings.json                  # 全局拒绝规则、全局 hook
  skills/
    chenk-blog-write/SKILL.md    # 博客写作
    simplify/SKILL.md            # 代码审查
    update-config/SKILL.md       # 设置管理
    claude-api/SKILL.md          # Anthropic SDK 使用

my-project/
  .claude/
    settings.json                # 项目级权限、环境变量、hook
    settings.local.json          # 本地覆盖配置
    commands/
      deploy.md                  # /deploy 命令
      release.md                 # /release 命令
    skills/
      project-conventions/SKILL.md  # 项目专属编码规范
    agents/
      research.md                # 研究型 subagent
      test-writer.md             # 测试生成型 subagent
  hooks/
    block-env-read.js            # 敏感环境变量读取拦截
    bash-blacklist.js            # 危险命令阻断
    format-on-write.js           # 保存时自动格式化
    test-on-edit.js              # 编辑时自动运行测试
  mcp.json                       # MCP server 配置（浏览器、数据库等）
```

组织原则：

- **用户级（`~/.claude/`）**：存放个人偏好与通用工具（如 SDK 封装、通用 skill）；
- **项目级（`my-project/.claude/`）**：存放团队共享知识与项目特有流程（如部署命令、测试 subagent、项目规范 skill）。

![扩展库布局示意图：用户级承载个人偏好与通用工具；项目级承载团队共享配置与项目特有逻辑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig5.png)

## 系列结束

十章下来，你有了：

- 一个配置好的 Claude Code，脑子里有三层设置模型 (第 1、 9 章)。
- 熟练使用 shortcuts、 modes 和对话控制 (第 2 章)。
- 四种扩展机制 — slash commands, MCP, hooks, skills — 以及如何在它们之间选择的决策树 (第 3、 4、 5、 7、 10 章)。
- 并发原语 — sub-agents, worktrees, plan mode — 把单个 session 扩展到更大的工作 (第 8 章)。
- 一套能用的 SDK + GitHub 集成方案，把 Claude 放进 CI (第 6 章)。

表面功夫就这些。后面真正有趣的工作，不再关于 Claude Code 本身——而是你用它*构建*什么。

去构建吧。
