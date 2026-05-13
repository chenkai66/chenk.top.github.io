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
Claude Code 提供四种扩展机制：slash commands、MCP servers、hooks 和 skills，它们的功能存在交叉。当你冒出“Claude 应该知道怎么做 X”这个念头时，首要问题其实是：该选哪一种？

作为本系列的终章，我将直接为你梳理出清晰的决策路径。

![Claude Code 实操 (10)：技能及其适用的扩展机制 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/illustration_1.png)

## Skill 到底是什么

Skill 本质上是一个文件夹，位于 `~/.claude/skills/<name>/`（用户级）或 `<repo>/.claude/skills/<name>/`（项目级），其中至少包含一个 `SKILL.md` 文件。

每个 `SKILL.md` 由两部分组成：frontmatter（前置元数据）和 body（正文内容）。

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

会话启动时，Claude 会预先读取所有 skill 的 `description` 字段；只有当你的提问在语义上匹配某条描述时，才会按需加载对应的完整正文，并将其注入当前轮次的系统提示中。

这里有两个关键点：

- `description` 是承重墙。如果没写清楚“何时使用”，这个 skill 就永远不会被触发。
- 正文可以很长——因为它只在触发时才加载，只要没被激活，就不会占用任何上下文资源。

![Skill 加载生命周期：description 始终在上下文中，正文仅在 description 与 prompt 匹配时才加载](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig3.png)

### Frontmatter 字段

| 字段 | 是否必需 | 说明 |
|------|----------|------|
| `name` | 是 | skill 的唯一标识符，用于日志和调试。请使用小写短横线格式（`lowercase-with-dashes`）。 |
| `description` | 是 | **核心字段**。Claude 正是依据这条描述决定是否加载该 skill。如果描述中没有明确说明适用场景，这个 skill 将永远沉睡。 |

`description` 是整个文件中最重要的部分。Claude 在每次会话开始时都会扫描所有可用 skill 的描述；只有当你的 prompt 与某条描述在语义上匹配时，才会加载其完整正文。如果描述太模糊，skill 要么对所有输入都触发（泛化过度），要么完全不触发（形同虚设）——这两种情况同样糟糕。

### Body 内容

正文长度没有硬性限制，因为它只在 skill 被触发时才注入系统提示，因此冗长并不会带来额外开销：未触发即零成本。我有不少 skill 的正文超过 600 行，每一行都物有所值。

Body 支持自由格式的 Markdown，但以下几种结构已被实践验证为高效：

- **语气/风格规范**：定义 Claude 在该 skill 激活时应采用的表达方式；
- **Schema/格式规范**：明确要求必须严格遵守的输出格式；
- **工作流**：分步骤的操作流程；
- **规则/约束**：明确禁止的行为。

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

```text
~/.claude/
  skills/
    chenk-blog-write/
      SKILL.md
    update-config/
      SKILL.md
    simplify/
      SKILL.md
```

用户级 skill 在你本地的所有项目中均可使用。

### 项目级 skill

```text
my-project/
  .claude/
    skills/
      deploy-staging/
        SKILL.md
      code-review/
        SKILL.md
```

项目级 skill 仅在当前项目中生效，可以提交到 Git 仓库，便于团队共享。

### 如何选择层级

| 场景 | 层级 | 原因 |
|------|------|------|
| 个人写作风格 | 用户级 | 适用于所有写作场景 |
| 项目部署流程 | 项目级 | 与特定代码库紧密绑定 |
| 代码审查偏好 | 用户级 | 反映个人风格，跨项目一致 |
| API 集成工作流 | 项目级 | 依赖特定 API 和项目约定 |
| 通用“简化代码”能力 | 用户级 | 具有普适性，适用于各类项目 |

## 从零编写一个真实 skill

我们以一个具体用例为例：为团队构建一个将 Node.js 应用部署到预发布（staging）环境的 skill。

### 步骤 1：识别“skill 形态”的使用场景

关键在于判断：这是应在相关话题出现时自动激活的领域知识，还是需要用户显式调用的命令？

向 staging 环境部署是团队在 PR 合并后、功能测试中、排查生产问题等多种上下文中反复执行的操作——它不是一个孤立命令，而是一整套关于“我们如何部署”的结构化知识，这正是典型的 skill-shaped 场景。

### 步骤 2：撰写 description

先写描述，因为它直接决定了 skill 的触发条件：

```markdown
---
name: deploy-staging
description: Use when deploying to the staging environment, preparing a staging build, or troubleshooting staging deployment issues. Covers the deploy pipeline, environment config, health checks, and rollback procedures for the acme-api project.
---
```

务必具体。“Use for deployment” 过于宽泛，会导致 skill 在讨论部署理论时也被误触发；而 “Use when deploying to the staging environment” 则精准锚定到实际操作行为。

### 步骤 3：撰写 body

一个真实的 `deploy-staging` skill 正文应包含以下要素：环境信息（staging URL、region、ECS cluster、默认部署分支）；预部署检查清单（测试通过、TypeScript 编译成功、无未提交变更、当前分支正确）；部署命令流（构建 Docker 镜像 → 为 ECR 打标签 → 推送 → 更新 ECS service）；健康检查（等待 60 秒后调用 `/health` 端点）；回滚流程（定位上一版 task definition ARN，执行 `aws ecs update-service --task-definition ...`）；以及常见故障排查——部署超时查 CloudWatch、健康检查失败查依赖连通性、ECR 推送失败用 `aws ecr get-login-password` 刷新凭证。

最后一定要加入几条硬性规则：禁止从此 skill 直接部署至 production，每次部署后必须执行健康检查，若健康检查失败，立即回滚，不得在 staging 上调试。

### 步骤 4：测试

创建文件并启动 Claude 会话：

```bash
mkdir -p .claude/skills/deploy-staging
# (在 .claude/skills/deploy-staging/SKILL.md 中写入上述内容)

# 启动 Claude 并测试
claude
> Deploy the latest changes to staging.
```

Claude 应自动加载该 skill 并严格遵循其中步骤。如果未触发，请检查 `description` 是否与你的自然语言表述匹配（例如你说的是 “push to staging”，但描述中写的是 “deploy to staging”）。

### 步骤 5：持续迭代

经过几次使用后，你会逐渐发现遗漏点，逐条补充进正文：

- “得注明必须先连接公司 VPN”；
- “ECR 登录步骤应前置并加粗强调”；
- “补充 Slack 通知步骤”。

每次迭代都让 skill 更加完整。一周之后，它就会成为该部署流程唯一可信、可执行、可审计的权威文档。

## Skill vs slash command vs hook vs MCP — 完整对比

这是我刚开始学习时最希望拥有的对照表：

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

**Slash command 是指令，skill 是知识。**  
Slash command 表达的是“请执行这个确定动作”；skill 表达的是“这是我对某类问题的系统性理解，请在适用时自主调用”。

**Hook 是自动的，其余皆为显式启用。**  
Hook 无需任何授权即可在工具调用时强制运行——这使其成为安全策略、合规审计等强制场景的理想选择，但完全不适合可选工作流。

**MCP 提供能力，其余提供指导。**  
只有 MCP 能赋予 Claude 原生不具备的新能力（如浏览器自动化、数据库访问、第三方 API 集成）。其余机制（slash command / hook / skill）均不扩展能力边界，而仅影响 Claude 如何使用其已有能力。

## Skill 的发现与加载

会话启动时，Claude 会扫描 `~/.claude/skills/*/SKILL.md`（用户级）和 `.claude/skills/*/SKILL.md`（项目级），读取每个文件 frontmatter 中的 `description` 字段。这些描述始终处于上下文中——它们是系统提示的一部分。

当你发送消息时，Claude 会将 prompt 与所有 skill 的 description 进行语义匹配。若存在匹配项，则将对应 skill 的**完整正文**加载进当前轮次的上下文：

- **描述始终加载**（开销小——仅 description 行本身）；
- **正文按需加载**（可能较大，但仅在相关时才加载）；
- **多个 skill 可同时触发**（若多个 description 匹配），可能导致上下文膨胀。

如果某个 skill 未按预期触发，请按以下顺序排查：

1. **描述内容**：是否包含你 prompt 中的关键语义？Claude 基于语义相似度匹配，但具体描述比模糊描述更易命中。
2. **文件路径**：`SKILL.md` 是否在正确目录？
3. **frontmatter 格式**：YAML 是否有效？`---` 分隔符是否存在？
4. **尝试显式调用**：某些 skill 如果也注册为 slash command，可通过名称调用。但 skill 的核心价值在于隐式加载——如果你必须靠名字才能触发它，那它很可能更适合做成 slash command。

## 条件化 skill

有时，一个 skill 只应在特定上下文中触发。你可以通过 description 中的限定语句（如 “Only relevant in Python projects”）和 body 中的条件规则共同表达：

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

![Claude Code 实操 (10)：技能及其适用的扩展机制 —— 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/illustration_2.png)

遇到“Claude 应该知道怎么做 X”时，按以下流程判断：

**1. 任务需要尚不存在的工具吗？**（比如浏览器自动化、查询真实数据库、调用内部 API。）  
→ 构建一个 **MCP server**。其他机制无法赋予新能力。

**2. 工具调用前后需要自动发生点什么吗？**（比如拦截、验证、记录日志、自动格式化？）  
→ 编写一个 **hook**。这是唯一不需要模型决定就能自动运行的机制。

**3. 这是个紧凑的流程，用户会显式调用吗？**（比如 `/commit`、`/deploy-staging`、`/make-changelog`）  
→ 创建一个 **slash command**。命令就是用来点名调用的。

**4. 这是一堆领域知识吗？**（比如写作风格、标准工作流、一套规范）  
→ 编写一个 **skill**。Skill 的价值在于让 Claude 自己**识别并应用**，而不需要你特意喊它。

如果某个需求适配多种机制，优先选择实现最简单的一种。Skill 内部调用 slash command 是合理设计，但将本应是 slash command 的功能强行包装为 skill，会导致触发不可靠、维护困难。

![决策树：按顺序走一遍，遇到第一个 YES 就停 —— MCP 加新能力、Hook 做自动策略、Slash command 显式调用、Skill 按话题自动加载](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig1.png)

## 我实际写过的三个 skill

**1. `chenk-blog-write`**  
用于在这个站点写文章。涵盖 front matter 结构、写作风格、中英文一致性要求、封面生成（Wanxiang）、部署流程和常见陷阱。只要提到 chenk.top、“write a post” 或 “new article”，就会触发。正文约 600 行，每一行都值得保留。

为什么是 skill 而不是 command？因为我并不总是显式调用它。有时我说“写一篇关于 hooks 的文章”，Claude 就需要知道这意味着“在 chenk.top 上用指定风格和格式写一篇博客”。skill 因描述匹配而自动激活。

**2. `update-config`**  
用于修改 `~/.claude/settings.json` 和 `.claude/settings.json`。触发词包括 “allow X command”、“set env Y”、“add a hook”、“move permission to project settings”。它编码了第 9 章中的权限优先级规则和典型配置模式，省去了我每次都要重新推导合并顺序的麻烦。

为什么是 skill 而不是 hook？因为它提供的是“如何编辑配置”的知识，而非强制策略。hook 会阻止或允许操作，而这个 skill 是在“教学”。

**3. `simplify`**  
用于审查我自己的代码。触发词如 “is there a simpler way to do this”、“review for quality”、“check for reuse”。它编码了我的代码品味：偏好组合而非继承，立即删除死代码，命名应反映本质而非实现方式，函数超过 30 行大概率该拆成两个。

为什么是 skill 而不是 command？因为它不是一套固定流程，而是一组价值观和启发式规则，用于指导 Claude 如何阅读和评价代码。品味无法通过命令传递，但可以被教会。

## 什么时候 skill 不是正确答案

触发过于频繁的 skill 反而有害：它会将无关知识注入本无需该 skill 的任务上下文，造成干扰甚至误导。以下是三个常见陷阱：

![三种 Skill 反模式：描述模糊导致触发过频、正文重叠导致上下文膨胀、本该用 Hook 强制的规则被写成 Skill](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig4.png)

- **描述模糊**  
  “用于一般编程。”——这种描述匹配一切，结果等于什么都用不上。
  
- **技能重叠**  
  两个 skill 都响应 “write code”，导致上下文膨胀。模型收到两套可能矛盾的指令，难以判断该遵循哪一套。解决方法是合并技能，或让描述更具体，确保它们针对不同子话题触发。
  
- **本该是 Hook**  
  “总是在编辑后格式化代码。”——这是 hook 的职责，不是 skill。Skill 提供建议，hook 实施强制。

判别标准很简单：如果指令中包含 “always” 或 “never”，且适用于某类工具调用的每一次，那就是 hook；如果包含 “讨论 X 时” 或 “处理 Y 类任务时”，那就是 skill。

## 构建你自己的扩展库

随着时间推移，你会通过全部四种扩展机制（slash command、hook、MCP、skill）积累大量扩展。以下是我组织它们的方式：

```text
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

组织原则如下：

- **用户级（`~/.claude/`）**：存放个人偏好与通用工具（如 SDK 封装、通用 skill）；
- **项目级（`my-project/.claude/`）**：存放团队共享知识与项目特有流程（如部署命令、测试 subagent、项目规范 skill）。

![扩展库布局示意图：用户级承载个人偏好与通用工具；项目级承载团队共享配置与项目特有逻辑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/10-skills-decision-tree/fig5.png)

## 系列结束

十章走下来，你已经掌握了：

- 一个配置完善的 Claude Code，脑中已建立三层设置模型（第 1、9 章）；
- 熟练运用快捷键、思考模式和对话控制（第 2 章）；
- 四种扩展机制——slash commands、MCP、hooks、skills——以及在它们之间选择的决策树（第 3、4、5、7、10 章）；
- 并发原语——sub-agents、worktrees、plan mode——用于将单个会话扩展到更大规模的工作（第 8 章）；
- 一套可用的 SDK 与 GitHub 集成方案，能把 Claude 放进 CI 流程（第 6 章）。

以上构成了 Claude Code 的完整能力表面。但真正有趣的部分才刚刚开始——接下来的工作不再关乎 Claude Code 本身，而是你用它**构建什么**。

去构建吧。
