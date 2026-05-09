---
title: "Claude Code 实战入门（七）：我每天真用的十个 Hooks，附完整代码"
date: 2026-04-22 09:00:00
tags:
  - claude-code
  - hooks
  - security
  - workflow
categories: Claude Code
lang: zh
mathjax: false
series: claude-code-learn
series_title: "Claude Code 实战入门"
series_order: 7
description: "从参考仓库的 100 个脚本里挑出 10 个，每一个都给：它做什么、JS 全文、settings.json 的接线方式、它会咬人的地方。PreToolUse 守安全，PostToolUse 做卫生，那些朴素但救命的脚本。"
disableNunjucks: true
translationKey: "claude-code-learn-7"
---
第 5 篇讲了 Hooks 的概念，这篇是实战指南。在我的参考仓库里有 100 个脚本，其中 10 个是每个重要项目都会用到的。我会把这 10 个挨个讲一遍，并附上代码。

所有示例默认使用 Node 18 及以上版本，脚本存放在 `./hooks/` 目录下，设置为可执行权限（`chmod +x`），并在 `.claude/settings.json` 中这样配置：

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] }
    ]
  }
}
```
![Claude Code 实战入门（七）：我每天真用的十个 Hooks，附完整代码 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/07-hooks-deep-dive/illustration_1.png)

## 1. block-env-read——保护敏感信息

这是投资回报率最高的一个 hook。它可以阻止 `Read` 和 `Grep` 操作访问 `.env`、`id_rsa`、`credentials.json` 等敏感文件：

```javascript
#!/usr/bin/env node
const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const t = JSON.parse(Buffer.concat(chunks).toString());
const p = t.tool_input?.file_path || t.tool_input?.path || "";
const sensitive = ['.env', 'credentials.json', 'secrets.yaml', 'id_rsa', '.aws/credentials'];
if (sensitive.some(s => p.includes(s))) {
  console.error(`Blocked: ${p} 匹配到敏感文件模式。`);
  process.exit(2);   // 2 表示拦截，仅在 PreToolUse 中有效
}
process.exit(0);
```

将这个脚本绑定到 `Read|Grep|MultiEdit|Edit|Write` 上。在 PreToolUse 阶段，返回退出码 2 会直接拦截调用；stderr 输出的内容会反馈给模型，让它明白拦截的原因。
## 2. bash-blacklist——拦截 `rm -rf /`

![Claude Code 实战入门（七）：我每天真用的十个 Hooks，附完整代码 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/07-hooks-deep-dive/illustration_2.png)

这是最容易踩坑的地方。PreToolUse 对接 `Bash`：

```javascript
const cmd = JSON.parse(/* stdin */).tool_input?.command || "";
const banned = [/rm\s+-rf\s+\/(\s|$)/, /:\(\)\s*\{.*:\|:.*\}\s*;:/, /mkfs\./, /dd\s+if=.*of=\/dev\//];
if (banned.some(re => re.test(cmd))) {
  console.error("Blocked: dangerous command pattern.");
  process.exit(2);
}
```

正则列表故意保持简短。太长的黑名单一旦误报，我肯定会直接关掉它。
## 3. bash-whitelist——用于靠近生产环境的机器

反过来，针对那些涉及生产环境的代码仓库。只允许明确列出的二进制文件：

```javascript
const allow = new Set(['ls','cat','grep','rg','git','npm','node','python','python3','curl','jq','head','tail']);
const first = (cmd.trim().split(/\s+/)[0] || "").split('/').pop();
if (!allow.has(first)) { console.error(`Not on allowlist: ${first}`); process.exit(2); }
```

白名单的优势就在于黑名单的不足：我不会无意中放行任何新东西。
## 4. block-git-push——禁止意外推送

我从来都不想让 Claude 悄悄执行 git push。PreToolUse 对接 `Bash`：

```javascript
if (/^\s*git\s+push\b/.test(cmd) || /git.*push.*--force/.test(cmd)) {
  console.error("Blocked: git push must be human-initiated.");
  process.exit(2);
}
```

出错的代价远远高于我自己敲 `git push` 的那点麻烦。
## 5. format-on-write——Prettier 作为 PostToolUse

PostToolUse 挂在 `Write|Edit|MultiEdit` 上：

```javascript
const path = t.tool_input?.file_path;
if (path && /\.(ts|tsx|js|jsx|json|md|css)$/.test(path)) {
  require('child_process').execSync(`npx prettier --write ${path}`, { stdio: 'inherit' });
}
```

PostToolUse 在编辑完成后运行，exit code 2 不会触发任何回滚。它的目标是保持代码整洁，而不是强制某种规则。
## 6. test-on-edit——快速失败

PostToolUse 挂载到 `Edit|MultiEdit`，专门针对源码文件生效：

```javascript
if (/\/(src|lib)\/.*\.(ts|js)$/.test(path)) {
  try { require('child_process').execSync('npm run -s test:related -- ' + path, { stdio: 'inherit' }); }
  catch { console.error("Tests failed after edit"); process.exit(1); }
}
```

返回退出码 1，把失败信息反馈给模型。模型读取测试输出后会重新尝试。这个 hook 是唯一一个让 Claude 在我的代码库中逐步写出更高质量代码的关键。
## 7. backup-before-edit——安全网

PreToolUse 接 `Edit|Write|MultiEdit`：

```javascript
const fs = require('fs');
if (fs.existsSync(path)) {
  fs.copyFileSync(path, `/tmp/cc-backup-${Date.now()}-${path.replace(/\//g,'_')}`);
}
```

这是一道简单的保险。我从 `/tmp` 恢复过两次文件，每次挽回的损失都抵得上一年的 cron 任务工资。
## 8. log-tool-calls——可观测性

PostToolUse 接 `*`：

```javascript
const fs = require('fs');
const line = JSON.stringify({ ts: Date.now(), tool: t.tool_name, input: t.tool_input }) + "\n";
fs.appendFileSync('.claude/tool-calls.jsonl', line);
```

平时不会去翻这个文件。但真用到的那天，你会庆幸它就在那儿。
## 9. read-before-write——杜绝盲改

PreToolUse 接 `Edit|MultiEdit`：

```javascript
const fs = require('fs');
const seen = JSON.parse(fs.existsSync('.claude/seen.json') ? fs.readFileSync('.claude/seen.json') : "{}");
if (t.tool_name === 'Read') { seen[path] = Date.now(); fs.writeFileSync('.claude/seen.json', JSON.stringify(seen)); process.exit(0); }
if (!seen[path]) { console.error(`Blocked: ${path} was not Read in this session.`); process.exit(2); }
```

我让模型改文件前必须先读。这样能抓住一个隐蔽的 bug：模型有时会按自己的先验知识改文件，而不是基于文件当前的实际状态。
## 10. work-hours-only——合理的工作时间限制

PreToolUse 接 `Bash`：

```javascript
const h = new Date().getHours();
if (h < 9 || h >= 22) {
  console.error("Outside work hours. Refuse.");
  process.exit(2);
}
```

我把它部署在处理下班后消息的服务器上。如果凌晨 2 点 bot 想执行破坏性操作，那几乎可以肯定是误触发。
## 把它们串起来的三条规则

我自己踩坑总结出来的三条：

1. **PreToolUse 负责策略，PostToolUse 负责清理。** 别指望在 PostToolUse 里撤销操作——副作用早就发生了。
2. **stderr 是反馈，exit code 是结果。** Exit 2 会拦截调用（只在 PreToolUse 中有效）。stderr 的内容会原样传给 Claude。两者都要用好。
3. **Hooks 失败时默认关闭。** 一个出问题的 hook 会阻断所有工具调用。在正式接入之前，先用 `echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/x"}}' | node hook.js` 测试一下脚本。

10 个 hooks 听起来不多，但已经足够让一个 Yolo 模式的会话变得可控了。
