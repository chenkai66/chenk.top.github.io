---
title: "Claude Code 实战入门（九）：settings.json、三层权限模型、env"
date: 2026-04-24 09:00:00
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

这一章就是来补上这块缺失的参考文档。

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/09-settings-and-permissions/illustration_1.png)

## 三个层级

Claude Code 会按顺序读取三个 `settings.json` 文件：

1. **用户设置** — `~/.claude/settings.json`。对你机器上的每个项目都生效。
2. **项目设置** — `<repo>/.claude/settings.json`。提交到 git 里。对这个 repo 里的所有人都生效。
3. **本地设置** — `<repo>/.claude/settings.local.json`。被 gitignore 忽略。你在这个 repo 里的私有覆盖配置。

合并规则很简单：后一层覆盖前一层，键对键（key by key）。**权限方面**，`allow` 是叠加的，`deny` 是否决性的——一旦任何一层 deny 了某个操作，其他层都无法再重新 allow 它。这种不对称性正是系统安全的基石。

实际建议：把组织策略放在 `~/.claude/settings.json`，把项目规则放在 `.claude/settings.json`（提交），把你那些“我信任我机器上这个特定操作”的覆盖配置放在 `.claude/settings.local.json`。

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

括号里的内容是针对特定工具的 glob 风格匹配器：

- `Read(path-glob)` — 文件模式。
- `Edit(path-glob)` — 同上。
- `Bash(command-pattern)` — 必须匹配第一个 token。用 `*` 要小心：`Bash(git *)` 会允许 `git push --force`。写具体点。
- `WebFetch(domain:host)` — 只匹配 host，不包含路径。

裸写的 `Read` 或 `Bash` 会允许该工具下的所有操作。除了 `~/.claude/settings.json` 这种受信任的个人用途外，这几乎总是太宽泛了。

## 为什么 deny 说了算

![Claude Code Hands-On (9): settings.json, the Three-Layer Permission Model, and Env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/09-settings-and-permissions/illustration_2.png)

一旦合并后的配置里有任何东西 deny 了一个动作，别的都救不回来。这是你要利用的杠杆。

举个例子。某个 repo 的 `.claude/settings.json` 写着：

```json
{ "permissions": { "deny": ["Bash(git push *)"] } }
```

队友加了个 `.claude/settings.local.json`，里面写 `"allow": ["Bash(git push origin main)"]`。这**不会**允许 push。项目层的 deny 胜出。这是正确的行为，你应该依赖它。

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

有两点要知道：

- 这些变量适用于 Bash 和任何继承环境的 hook 脚本。它们*不会*泄露到模型的 prompt 里。这是设置 `*_API_KEY` 的安全位置。
- 本地层覆盖项目层覆盖用户层。所以 `.claude/settings.local.json` 里的 `DEBUG=true` 只会为你开启日志，而不需要提交这个改动。

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

匹配器是用管道符分隔的工具名称。所有三层里的 hooks 都会运行；hooks 没有覆盖机制。在更深层添加 hook 是追加，永远不会替换。

## 一个真实 repo 里的真实 settings.json

这是我某个项目的配置，稍微脱敏了一下：

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

1. Bash 白名单只包含了只读和可逆的 Git 命令，绝不含 `push`、`reset --hard` 或 `rebase -i`。Push 必须是人为的 deliberate 动作。
2. `Edit(.github/workflows/**)` 被 deny 了。CI 配置变更需要 review，我不想让它们混进普通 commit 里。
3. Hooks 给 deny 列表上了双保险。如果 deny 规则有拼写错误，hook 依然能拦截危险调用。

## 优先级顺序，当作检查清单

当行为不符合预期时：

1. 在任何 `deny` 里吗？→  blocked，不管有没有 allow。
2. 在任何 `allow` 里吗？→  permitted。
3. 否则 → Claude 会在执行前询问。

如果你想知道哪条规则赢了，加 `--debug` 运行并查看权限解析日志。它会准确告诉你哪个文件的哪一行做出了裁决。

## 结语

settings.json 就是 Claude 在项目里能做什么的宪法。Keep deny short and merciless，keep allow specific，keep hooks 作为第二道防线。一旦你脑子里有了层级和优先级的概念，配置一个新 repo 只要九十秒。在这之前，你会觉得规则很 arbitrary；其实不是。