---
title: "Claude Code 实战入门（六）：SDK、GitHub 集成、把 Claude 放进 CI"
date: 2026-04-21 09:00:00
tags:
  - claude-code
  - sdk
  - github-actions
  - automation
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 6
description: "SDK 把 Claude Code 从 CLI 变成库。GitHub Action 让它在 PR 上响应 @claude。两者一起把 Claude 放进已经在跑你测试的同一条 CI 流水线——又不交出控制。"
disableNunjucks: true
translationKey: "claude-code-learn-6"
---
CLI 是最直观的界面，SDK 才是真正有趣的部分，而 GitHub 集成则是价值体现的关键所在。
![Claude Code 实战入门（六）：SDK、GitHub 集成、把 Claude 放进 CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_1.png)

## 用一段话介绍 SDK

`@anthropic-ai/claude-code` 是一个 npm 包。它把 CLI 使用的 Claude Code 引擎直接封装成编程接口，工具和权限完全一致。你给它一个 Prompt，它返回一个异步可迭代的对话事件流。无论是脚本、服务还是 CI 步骤，都可以轻松集成。

安装命令：

```bash
npm install @anthropic-ai/claude-code
```

Hello world 示例：

```typescript
import { query } from '@anthropic-ai/claude-code';

const result = query({
  prompt: '列出本仓库最大的 3 个源文件，并用一句话说明每个文件的作用。',
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'
  }
});

for await (const event of result) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

运行代码后，你会在终端看到完整的 Agent 执行过程——包括工具调用等细节。这就是 CLI 的核心功能，只是去掉了聊天界面。
## 程序化的权限

这一部分必须处理好。CLI 默认会“询问用户”，但脚本没法做到这一点。因此，SDK 提供了权限模式：

| 模式             | 含义                                   |
|------------------|--------------------------------------|
| `default`        | 对任何需要确认的工具直接报错               |
| `acceptEdits`    | 自动接受文件修改，但对 shell 操作仍需确认     |
| `bypassPermissions` | 全部自动接受（危险）                   |
| 自定义           | 提供一个回调函数，按需决定每次调用的权限       |

实际开发中，推荐使用回调函数的方式：

```typescript
const result = query({
  prompt: '...',
  options: {
    permissionMode: 'custom',
    permissionCallback: async (tool, input) => {
      if (tool === 'Bash' && input.command.startsWith('rm ')) return 'deny';
      if (tool === 'Write' && input.path.startsWith('/etc')) return 'deny';
      return 'allow';
    }
  }
});
```

这样一来，你就有了程序化的权限控制策略。脚本可以无人值守运行，同时你也能确保它不会执行那些你明确禁止的操作。
## 一个实用脚本：自动更新 CHANGELOG

我在几个项目里都写了这个脚本，放在 `scripts/update-changelog.ts`：

```typescript
import { query } from '@anthropic-ai/claude-code';
import { execSync } from 'child_process';

const lastTag = execSync('git describe --tags --abbrev=0').toString().trim();
const commits = execSync(`git log ${lastTag}..HEAD --oneline`).toString();

const result = query({
  prompt: `
    给 CHANGELOG.md 添加一个新版本的条目。
    自 ${lastTag} 以来的提交记录如下：

    ${commits}

    按 Added/Changed/Fixed/Removed 分类整理。
    使用语义化版本号建议下一个版本。
    直接修改 CHANGELOG.md 文件。
  `,
  options: {
    cwd: process.cwd(),
    permissionMode: 'acceptEdits'
  }
});

for await (const event of result) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

每次发版前运行一次。CHANGELOG 自动生成，提交记录被整理成清晰的文字，我只需要检查一遍然后提交。每次发版能省 5 分钟，半年下来已经省了不少时间。
## GitHub Action

![Claude Code 实战入门（六）：SDK、GitHub 集成、把 Claude 放进 CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_2.png)

Anthropic 提供了一个官方的 Action：`anthropic/claude-code-action@v1`。把它加到你的 workflow 里：

```yaml
name: Claude on PR

on:
  pull_request_review_comment:
    types: [created]
  issue_comment:
    types: [created]

jobs:
  claude:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: anthropic/claude-code-action@v1
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

现在，仓库里的任何人都可以在 PR 评论中写 `@claude please review the failing test`，Claude 就会自动执行以下操作：

1. 检出对应的分支  
2. 阅读评论线程获取上下文  
3. 运行需要的内容（测试、Lint 工具、文件读取等）  
4. 在 PR 中回复分析结果  
5. 如果被要求，还可以推送提交  

这个 Action 会遵循仓库中的 `.claude/settings.json` 文件。第 5 篇文章里写的 hooks 依然有效。第 3 篇文章里提到的斜杠命令也照常工作——比如 `@claude /review` 会按你预期的方式运行。
## 我用 GitHub Action 做什么

主要用来处理三件事：

**1. PR 初步审查。** 每当有新 PR 提交时，workflow 会自动运行 `@claude /review`。等我去看的时候，系统已经生成了初步的评审意见。这能帮我节省大约 10 分钟时间，还能提前发现一些显而易见的问题。

**2. Issue 自动分类和建议。** 新 issue 提交后，workflow 会自动给它打标签、评估修复范围，并关联相关代码。这样一来，问题提交者能更快得到反馈，我也能迅速掌握背景信息。

**3. 文档同步更新。** 当 schema 发生变化时，一个 Action 会调用 Claude 来更新文档，确保内容与最新改动一致。虽然结果不一定每次都完美，但大约 80% 的工作是机械性的，效果还不错。
## SDK 与 Action 的区别

我会用 SDK 的情况：

- 我希望在本地运行的脚本中调用 Claude
- 触发条件是定时任务、文件变更或者手动命令
- 需要通过代码检查事件

我会选择 Action 的场景：

- 触发条件是 GitHub 事件
- 输出结果需要出现在 PR 或 Issue 中
- 希望有人参与其中，而参与方式是 `@mention`

两者当然有重叠部分。Action 的底层实际上就是基于 SDK 构建的。
## 整合整个系列

6 篇文章，层层递进：

1. 安装 + 三层配置
2. 快捷键与模式
3. 个人工作流的斜杠命令
4. 外部集成的 MCP
5. 提供安全保障的 Hooks
6. 用于程序化和 CI 的 SDK + Action

每篇文章单独看都有实际价值。合在一起，Claude Code 就从一个代码聊天工具变成了嵌入你代码库的可编程基础设施。

我观察到的高级用户有一个共同点：他们把 `.claude/` 当成代码库的一部分。设置、命令、Hooks 都会提交，都会经过 PR 审查，都会随着项目一起演进。这种习惯值得养成。

祝你发布顺利。
