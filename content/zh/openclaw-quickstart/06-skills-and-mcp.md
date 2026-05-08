---
title: "OpenClaw 快速上手（六）：技能系统与 MCP——让 Agent 学会做事"
date: 2026-04-08 09:00:00
tags:
  - openclaw
  - skills
  - mcp
  - 技能
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 6
description: "Skill 不是 Prompt 模板——它是一套完整的 SOP，包括触发条件、工具权限和执行流程。再加上 MCP 把外部能力接入 Agent，从浏览器自动化到数据库查询，一个配置文件搞定。"
disableNunjucks: true
translationKey: "openclaw-quickstart-6"
---

## 从"能回答"到"能做事"

大语言模型很聪明，但聪明不等于靠谱。你问它怎么写 SQL，它答得头头是道；让它连上数据库执行查询、格式化结果、发到钉钉群——这就是另一回事了。差距在哪？**缺少标准操作流程**。

OpenClaw 的技能系统填补这个缺口。一个 Skill 定义了做什么、什么时候做、用什么工具做、按什么步骤做。再配合 MCP 把外部能力统一接入，Agent 就从问答机器人变成了能干活的助手。

![技能系统架构示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_1.jpg)

## Skill 的组成

**Skill 不是 Prompt 模板**。模板只是带占位符的文本；Skill 是完整的执行规范——触发条件、工具权限、执行流程、输出格式，全部声明在一个 Markdown 文件里。

文件位置：`~/.openclaw/skills/<name>/SKILL.md`，由 YAML Frontmatter（清单）和 Body（SOP 正文）两部分组成：

```markdown
---
name: daily-standup-prep
description: "准备每日站会的汇报内容"
trigger: "准备站会|standup prep|今天要汇报什么"
tools_required: [git, calendar, file_read]
model_override: null
priority: 5
---

# 每日站会准备
## 步骤
1. 读取昨天的 git log，总结完成的工作
2. 检查日历，列出今天的会议和截止日期
3. 生成站会汇报模板
```

**懒加载机制**：启动时只读 Frontmatter（几十字节）建立触发索引；匹配成功后才加载 Body 到上下文；执行完毕释放。你可以放心写几十个 Skill，不用担心 token 预算。

## 动手写第一个 Skill

完整的"每日站会准备"技能：

```markdown
---
name: daily-standup-prep
description: "自动准备每日站会汇报材料：汇总昨日工作、今日计划、阻塞项"
trigger: "准备站会|站会准备|standup|今天汇报什么|daily standup"
tools_required: [git, calendar, file_read, file_write]
priority: 5
tags: [productivity, daily]
---

# 每日站会准备

你是团队协作助手。按以下步骤执行。

## 第一步：收集昨日工作
执行 git 命令获取最近 24 小时的提交记录：
git log --since="24 hours ago" --oneline --author="$(git config user.name)"
将结果整理为要点列表。

## 第二步：检查今日日程
调用 calendar 工具获取今天的事件。重点标注截止任务和协作会议。

## 第三步：生成汇报
按模板输出：
### 昨日完成
- [从 git log 整理]
### 今日计划
- [从日历整理]
### 阻塞项
- [有则列出，无则写"无"]
```

注册只需要把文件放对位置：

```bash
mkdir -p ~/.openclaw/skills/daily-standup-prep
# 写入 SKILL.md 后重启 Gateway
openclaw gateway restart
```

在 TUI 中输入"帮我准备站会"即可触发。

## 四种设计模式

**模板型**：定义输出格式，Agent 填充内容。适合周报、邮件、commit message。

**工作流型**：多步骤、有前后依赖。每步定义输入/操作/检查点：

```markdown
## 步骤 1：拉取变更
操作：git pull origin {branch}
检查点：确认无冲突

## 步骤 2：运行测试
操作：npm test
检查点：全部通过
```

**守卫型**：在危险操作前做验证，拦截并确认。比如生产部署前检查分支、CI 状态、未合并的热修复。设 `priority: 10` 确保它优先于实际执行的 Skill 被触发。

**组合型**：一个 Skill 通过子 Agent 调用其他 Skill——先 `code-review`，再 `changelog-gen`，最后 `deploy-guard`，全部通过才执行部署。

## 触发条件最佳实践

触发条件写得好不好，决定 Skill 能不能被正确激活。三个原则：

**具体且不重叠**。`trigger: "帮我做一下"` 太模糊；`trigger: "准备站会|standup prep|daily standup"` 才是正确粒度。

**用 priority 处理歧义**。多个 Skill 可能同时匹配时，数值高的优先。安全检查类设 10，实际执行类设 5。

**主动测试**：

```bash
openclaw skill test-trigger "帮我准备明天的站会"
# Matched: daily-standup-prep (score: 0.92)
# Partial: meeting-prep (score: 0.45)
```

## 内置技能

OpenClaw 预装四个通用技能：

| 技能名 | 功能 | 典型触发 |
|--------|------|----------|
| `web-research` | 搜索 + 摘要总结 | "帮我调研一下..." |
| `code-review` | 分析 diff，指出问题 | "review 这个 PR" |
| `file-organizer` | 按规则批量重命名/移动 | "整理这个目录" |
| `email-draft` | 根据模板撰写邮件 | "帮我写一封..." |

直接用，或者 fork 后改成你自己的版本。

## MCP：统一的外部能力接入

Skill 定义了"怎么做"，工具层回答"用什么做"。OpenClaw 的工具分两类：原生工具（git、file_read 等）和 MCP 工具（外部服务）。

**MCP（Model Context Protocol）** 是一个开放标准，定义 LLM Agent 与外部工具的通信协议——"Agent 世界的 USB 接口"。工具声明用统一 JSON Schema，调用用标准请求/响应格式，错误处理有规范化的错误码。一个协议，成百上千的集成。

![MCP 协议架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/06-skills-and-mcp/illustration_2.jpg)

在 `~/.openclaw/config.yaml` 添加 MCP Server：

```yaml
mcp_servers:
  - name: playwright
    command: npx
    args: ["@anthropic/mcp-playwright"]
    env:
      DISPLAY: ":0"
  - name: postgres
    command: npx
    args: ["@anthropic/mcp-postgres"]
    env:
      DATABASE_URL: "postgresql://user:pass@localhost:5432/mydb"
```

重启 Gateway 后自动可用。Skill 的 `tools_required` 声明对应工具名即可调用。

## MCP 实战

**浏览器自动化**：配置 Playwright MCP 后，Skill 可以调用 `browser_navigate`、`browser_click`、`browser_snapshot` 等工具。实际场景：自动登录内网系统抓取报表、监控页面状态变化。

**数据库查询**：配置 PostgreSQL MCP，注意用只读账号——Agent 不应该有写库权限。让它查数据、生成报表，但不碰写操作。

**自定义 MCP Server**——三步搞定：

```typescript
// 第一步：定义工具 Schema
const tools = [{
  name: "query_dingtalk_messages",
  description: "查询钉钉群最近的消息",
  inputSchema: {
    type: "object",
    properties: {
      group_id: { type: "string", description: "钉钉群 ID" },
      limit: { type: "number", default: 20 }
    },
    required: ["group_id"]
  }
}];

// 第二步：实现工具逻辑
async function handleToolCall(name: string, args: any) {
  if (name === "query_dingtalk_messages") {
    const messages = await dingtalkAPI.getMessages(args.group_id, args.limit);
    return { content: [{ type: "text", text: JSON.stringify(messages) }] };
  }
}
```

```yaml
# 第三步：注册到 OpenClaw
mcp_servers:
  - name: dingtalk
    command: node
    args: ["/path/to/dingtalk-mcp/index.js"]
    env:
      DINGTALK_APP_KEY: "your-key"
      DINGTALK_APP_SECRET: "your-secret"
```

同样的模式适用于飞书文档、企业微信、任何有 HTTP 接口的内部服务。

## MCP vs 原生工具

| 维度 | 原生工具 | MCP 工具 |
|------|----------|----------|
| 集成深度 | 深，框架耦合 | 浅，协议解耦 |
| 性能 | 快，进程内调用 | 略慢，跨进程通信 |
| 类型安全 | 编译时检查 | 运行时 Schema 验证 |
| 生态广度 | 有限 | 社区维护大量 Server |
| 维护成本 | 自己维护 | 社区/第三方维护 |

我的建议：核心操作（文件、git、shell）用原生工具；外部服务（数据库、浏览器、消息平台）用 MCP。如果某个 MCP Server 性能不够，再考虑封装为原生工具。

## 调试技能

```bash
# 列出所有已注册技能
openclaw skill list

# 查看某个技能详情
openclaw skill describe daily-standup-prep

# 测试触发匹配
openclaw skill test-trigger "你的输入"
```

常见问题：

- **没触发**：`test-trigger` 检查匹配分数，低于 0.7 说明 trigger 需要调整
- **触发错了**：多个 Skill trigger 重叠，调整 priority 或收窄关键词
- **触发了但报错**：`openclaw tool list` 检查工具是否可用，`openclaw mcp status` 检查 MCP Server 进程状态

## 小结

- **Skill** 把"我想让 AI 帮我做 X"变成精确的、可复用的、可测试的执行规范
- **MCP** 把外部能力通过标准协议接入，不管是浏览器、数据库还是钉钉消息

两者结合，Agent 获得的不只是更多知识，而是更大的**行动空间**——从"告诉你怎么做"到"直接帮你做"。

下一篇讲 Context Management：对话越来越长时，怎么让 Agent 记住该记的、忘掉该忘的。
