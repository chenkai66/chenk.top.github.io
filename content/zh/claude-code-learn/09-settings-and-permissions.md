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
Hooks 是你接入 Claude Code 的入口，而 settings.json 则是你提前告诉它能访问哪些内容的地方。这个文件还有一个特点，就是它的优先级规则总能让人大吃一惊。

这一章补上了缺失的参考指南。
![Claude Code 实战入门（九）：settings.json、三层权限模型、env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/09-settings-and-permissions/illustration_1.png)

## 三层设置

Claude Code 按顺序读取三个 `settings.json` 文件：

1. **用户设置** — `~/.claude/settings.json`。作用于我这台机器上的所有项目。
2. **项目设置** — `<repo>/.claude/settings.json`。提交到 Git，适用于任何在这个仓库里工作的人。
3. **本地设置** — `<repo>/.claude/settings.local.json`。被 Git 忽略，是我对这个仓库的私有覆盖。

合并规则很简单：后面的层会逐个键值覆盖前面的层。权限处理上，`allow` 是叠加的，而 `deny` 是减法——一旦某一层拒绝了某个权限，其他层无法再允许它。这种不对称设计正是系统安全的核心。

实际用法也很清晰：把组织策略放在 `~/.claude/settings.json`，项目规则放在 `.claude/settings.json`（提交版本），而“我只在这台机器上信任这件事”的特殊覆盖则放在 `.claude/settings.local.json`。
## permissions 块——语法

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

括号里的内容是工具特定的 glob 匹配规则：

- `Read(path-glob)` — 文件路径模式。
- `Edit(path-glob)` — 和上面一样。
- `Bash(command-pattern)` — 第一个命令必须匹配。`*` 要谨慎使用，比如 `Bash(git *)` 会允许 `git push --force`。尽量写得更具体。
- `WebFetch(domain:host)` — 只匹配主机名，不包括路径。

直接写 `Read` 或 `Bash` 表示允许该工具的所有操作。这种权限范围通常太宽泛，除非是在个人完全信任的 `~/.claude/settings.json` 文件里，否则我不建议这样用。
## 为什么 deny 总是优先

![Claude Code 实战入门（九）：settings.json、三层权限模型、env — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/09-settings-and-permissions/illustration_2.png)

只要合并后的配置中有任何地方拒绝了某个操作，其他地方就无法再允许它。这就是你想要的控制方式。

举个例子。仓库的 `.claude/settings.json` 文件内容如下：

```json
{ "permissions": { "deny": ["Bash(git push *)"] } }
```

这时，队友在 `.claude/settings.local.json` 中添加了 `"allow": ["Bash(git push origin main)"]`。但这个设置不会生效，push 操作仍然会被拒绝。项目层级的 deny 优先级更高。这是正确的逻辑，我建议你完全依赖它。
## env——另一半

`env` 块为每次工具调用设置环境变量：

```json
{
  "env": {
    "NODE_ENV": "development",
    "PYTHONPATH": "./src",
    "DEBUG": "false"
  }
}
```

需要记住两点：

- 这些变量对 Bash 和所有继承环境的 hook 脚本生效，但不会泄露到模型的 prompt 中。这里是存放 `*_API_KEY` 的安全位置。
- 本地配置优先于项目配置，项目配置又优先于用户配置。所以如果在 `.claude/settings.local.json` 中设置 `DEBUG=true`，只会为你开启日志功能，改动也不会被提交到代码库。
## hooks——同文件引用

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

matcher 是用竖线分隔的工具名称。三层的所有 hooks 都会执行，不存在覆盖的情况。在深层添加 hook 是追加，不会替换原有的配置。
## 真实项目中的 settings.json 文件

这是我从自己的一个项目里提取的配置文件，稍微做了点脱敏处理：

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

有三点需要特别留意：

1. Bash 的允许列表只包含只读和可逆的 Git 命令，像 `push`、`reset --hard` 和 `rebase -i` 这种操作一律禁止。Push 是一个需要人为确认的动作，不能随意放行。
2. 禁止编辑 `.github/workflows/**` 文件。CI 配置的改动必须经过严格审查，不能混进普通的提交里。
3. Hooks 给禁止列表加了一层保险。如果禁止规则写错了，Hooks 依然能拦截危险操作，避免意外发生。
## 优先级顺序——检查清单

遇到行为不符合预期时，按以下顺序排查：

1. 是否在任何 `deny` 中？→ 直接拦截，不管 `allow` 怎么设置。
2. 是否在任何 `allow` 中？→ 允许通过。
3. 如果都不符合 → Claude 会在执行前询问我。

如果想确认是哪条规则生效，可以加上 `--debug` 参数运行，查看权限解析日志。日志会明确指出是哪个文件的哪一行做出了最终判断。
## 收尾

settings.json 就是 Claude 在项目里的行为准则，相当于一部宪法。deny 要写得简短、严格，allow 必须具体明确，hooks 则作为第二道防线。只要把分层和优先级的逻辑搞清楚，配置一个新仓库只需要 90 秒。在此之前，你可能会觉得这些规则很随意，但其实它们一点都不含糊。
