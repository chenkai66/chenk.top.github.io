---
title: "Claude Code 实战（六）：SDK 与 GitHub CI"
date: 2026-04-23 09:00:00
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
CLI 只是門面，SDK 提供核心功能，而 GitHub 集成才是真正釋放價值的關鍵環節。

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_1.png)

## 一段話說清 SDK

`@anthropic-ai/claude-code` 就是那個 npm 包。它暴露了 CLI 用的同一個 Claude Code 引擎，工具和權限一模一樣，只不過變成了程序化接口。你傳入一個 prompt，它會返回一個用於遍歷對話事件的異步迭代器。隨便插到哪兒都行——腳本、服務、CI 步驟。

安裝：

```bash
npm install @anthropic-ai/claude-code
```

Hello-world 示例：

```typescript
import { query } from '@anthropic-ai/claude-code';

const result = query({
  prompt: 'List the three largest source files in this repo and explain each in one sentence.',
  options: {
    cwd: process.cwd(),
    permissionMode: 'default'
  }
});

for await (const event of result) {
  if (event.type === 'text') process.stdout.write(event.text);
}
```

運行一下，你會看到相同的 agent 循環在終端中執行——包含工具調用。這本質上就是一個去除了聊天界面的 CLI。

## 權限，程序化控制

這塊必須搞對。CLI 默認策略是「問人」。腳本無法與用戶進行交互式確認。所以 SDK 暴露了幾種權限模式：

| Mode | Meaning |
|------|---------|
| `default` | 任何通常需要確認的工具都會報錯 |
| `acceptEdits` | 自動接受文件編輯，Shell 操作會問 |
| `bypassPermissions` | 自動接受所有操作（危險） |
| Custom | 你提供一個 callback，每次調用時決定 |

實際投入生產時，建議直接使用回調函數：

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

這樣，權限策略就實現了程序化控制。腳本可以無人值守運行，你也清楚它絕對幹不了你明確禁止的事。

## 實戰腳本：自動更新 CHANGELOG

我幾個項目的 `scripts/update-changelog.ts` 裡都掛著這個：

```typescript
import { query } from '@anthropic-ai/claude-code';
import { execSync } from 'child_process';

const lastTag = execSync('git describe --tags --abbrev=0').toString().trim();
const commits = execSync(`git log ${lastTag}..HEAD --oneline`).toString();

const result = query({
  prompt: `
    Update CHANGELOG.md with a new entry for an upcoming release.
    The commits since ${lastTag} are:

    ${commits}

    Group them into Added/Changed/Fixed/Removed.
    Use semantic versioning to suggest the next version.
    Edit CHANGELOG.md in place.
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

每次發布前跑一次。CHANGELOG 自己寫自己，commit log 被整理成通順的文字，我只負責審查和提交。每次發布可節省約 5 分鐘；累計過去六個月的所有發布，節省時間效益顯著。

## GitHub Action

![Claude Code Hands-On (6): The SDK, GitHub Integration, and Claude in CI — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/06-sdk-and-github/illustration_2.png)

Anthropic 出了官方 Action：`anthropic/claude-code-action@v1`。加到 workflow 裡：

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

現在倉庫裡任何人只要在 PR 評論裡寫 `@claude please review the failing test`，Claude 就會：

1. Checkout 分支
2. 讀取評論線程獲取上下文
3. 運行需要的操作（測試、lint、讀文件）
4. 在 PR 裡回復分析結果
5. 如果被要求，還可以推送 commit

Action 會認倉庫裡的 `.claude/settings.json`。你在第 5 篇寫的 hooks 依然生效。你在第 3 篇寫的 slash commands 也能用——`@claude /review` 會按預期工作。

## 我拿 GitHub Action 幹什麼

有三件事我離不開了：

**1. PR 初審。** 新 PR 一開，workflow 自動跑 `@claude /review`。等人來看的時候，對話裡已經有了第一輪審查結果。每次 PR 省審查者 10 分鐘，還能抓住那些顯而易見的問題。

**2. Issue 總結。** 新 Issue 觸發 workflow，自動打標籤、建議修復範圍、關聯相關代碼。報告者更快得到回應，維護者也有了起手式。

**3. 文檔更新。** Schema 一變，Action 就跑起來讓 Claude 同步更新文檔。不總是完美，但 ~80% 的工作都是機械性的。

## SDK 和 Action 的邊界

我用 SDK 的時候：

- 我想在本地跑的腳本裡嵌入 Claude
- 觸發條件是 cron、文件變更或手動命令
- 我需要程序化檢查事件

我用 Action 的時候：

- 觸發條件是 GitHub 事件
- 輸出結果要落在 PR 或 Issue 裡
- 我想要人在環路裡，且環路通過 `@mention` 觸發

當然有重疊。Action 底層就是基於 SDK 構建的。

## 把系列串起來

六篇文章，一個遞進過程：

1. 安裝 + 三層 config
2. 快捷方式和 modes
3. 個人工作流的 Slash commands
4. 外部集成的 MCP
5. 安全護欄 Hooks
6. 程序化和 CI 用的 SDK + Action

單獨閱讀每一篇文章，都能獲得一項具體可用的能力。合在一起，Claude Code 就從代碼聊天客戶端變成了住在倉庫裡的程序化基礎設施。

我觀察過的那些高手，只有一個共同特質：他們把 `.claude/` 當作代碼庫的一部分。配置、命令、hooks 全部提交，全部在 PR 裡審查，全部隨項目進化。這才是值得長期堅持的工程習慣。

放心交付。