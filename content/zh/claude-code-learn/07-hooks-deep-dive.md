---
title: "Claude Code 实战（七）：十个实用 Hooks 配方"
date: 2026-04-24 09:00:00
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
第五章咱们过了遍 Hook 的概念巡礼，这一章就是实地指南。那个 100 脚本的参考库里，只有十个能进我每个严肃项目的法眼。我就挑这十个，配上代码 walkthrough 一遍。

所有示例默认 Node 18+ 环境，脚本存到 `./hooks/`，加上 `chmod +x` 权限，然后在 `.claude/settings.json` 里这样配置：

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read|Grep", "hooks": [{ "type": "command", "command": "node ./hooks/block-env-read.js" }] }
    ]
  }
}
```

![Claude Code Hands-On (7): Ten Hooks I Actually Use, with the Code — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/07-hooks-deep-dive/illustration_1.png)

## 1. block-env-read — 保护 secrets

这是 ROI 最高的一个 Hook。防止 `Read` 和 `Grep` 触碰 `.env`、`id_rsa`、`credentials.json`：

```javascript
#!/usr/bin/env node
const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const t = JSON.parse(Buffer.concat(chunks).toString());
const p = t.tool_input?.file_path || t.tool_input?.path || "";
const sensitive = ['.env', 'credentials.json', 'secrets.yaml', 'id_rsa', '.aws/credentials'];
if (sensitive.some(s => p.includes(s))) {
  console.error(`Blocked: ${p} matches a sensitive file pattern.`);
  process.exit(2);   // 2 = block, only valid in PreToolUse
}
process.exit(0);
```

把它挂在 `Read|Grep|MultiEdit|Edit|Write` 上。PreToolUse 里退出码 2 会阻断调用；stderr 的内容会原样喂给模型，让它知道*为什么*被拦。

## 2. bash-blacklist — 阻止 `rm -rf /`

![Claude Code Hands-On (7): Ten Hooks I Actually Use, with the Code — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/claude-code-learn/07-hooks-deep-dive/illustration_2.png)

最常见的自伤操作。挂在 `Bash` 的 PreToolUse 上：

```javascript
const cmd = JSON.parse(/* stdin */).tool_input?.command || "";
const banned = [/rm\s+-rf\s+\/(\s|$)/, /:\(\)\s*\{.*:\|:.*\}\s*;:/, /mkfs\./, /dd\s+if=.*of=\/dev\//];
if (banned.some(re => re.test(cmd))) {
  console.error("Blocked: dangerous command pattern.");
  process.exit(2);
}
```

正则列表故意写得很短。太长的黑名单一旦误报多了，反而会被无视。

## 3. bash-whitelist — 适合生产环境周边的机器

反过来，适合那些要碰生产环境的 repo。只允许明确列出的二进制文件：

```javascript
const allow = new Set(['ls','cat','grep','rg','git','npm','node','python','python3','curl','jq','head','tail']);
const first = (cmd.trim().split(/\s+/)[0] || "").split('/').pop();
if (!allow.has(first)) { console.error(`Not on allowlist: ${first}`); process.exit(2); }
```

白名单在黑名单失手的地方赢：你不可能意外放行新东西。

## 4. block-git-push — 禁止意外 push

我从来没想过让 Claude 不打招呼就直接 push。挂在 `Bash` 的 PreToolUse 上：

```javascript
if (/^\s*git\s+push\b/.test(cmd) || /git.*push.*--force/.test(cmd)) {
  console.error("Blocked: git push must be human-initiated.");
  process.exit(2);
}
```

写错一次的代价，远比我自己敲一遍 `git push` 麻烦得多。

## 5. format-on-write — 把 Prettier 做成 PostToolUse

挂在 `Write|Edit|MultiEdit` 的 PostToolUse 上：

```javascript
const path = t.tool_input?.file_path;
if (path && /\.(ts|tsx|js|jsx|json|md|css)$/.test(path)) {
  require('child_process').execSync(`npx prettier --write ${path}`, { stdio: 'inherit' });
}
```

PostToolUse 在编辑*之后*运行，所以退出码 2 也回滚不了什么。重点是卫生，不是策略。

## 6. test-on-edit — 快速失败

针对源文件，挂在 `Edit|MultiEdit` 的 PostToolUse 上：

```javascript
if (/\/(src|lib)\/.*\.(ts|js)$/.test(path)) {
  try { require('child_process').execSync('npm run -s test:related -- ' + path, { stdio: 'inherit' }); }
  catch { console.error("Tests failed after edit"); process.exit(1); }
}
```

退出码 1 会把失败暴露给模型，它看到测试输出后会重试。就是这一个 Hook，随着时间推移教会了 Claude 在我的 repo 里写出更好的代码。

## 7. backup-before-edit — 安全网

挂在 `Edit|Write|MultiEdit` 的 PreToolUse 上：

```javascript
const fs = require('fs');
if (fs.existsSync(path)) {
  fs.copyFileSync(path, `/tmp/cc-backup-${Date.now()}-${path.replace(/\//g,'_')}`);
}
```

便宜的保险。我从 `/tmp` 恢复过两次文件，这两次都值回一年的 cron 任务工资。

## 8. log-tool-calls — 可观测性

挂在 `*` 的 PostToolUse 上：

```javascript
const fs = require('fs');
const line = JSON.stringify({ ts: Date.now(), tool: t.tool_name, input: t.tool_input }) + "\n";
fs.appendFileSync('.claude/tool-calls.jsonl', line);
```

你不会天天看这个文件。但用到它的那天，你会庆幸它在。

## 9. read-before-write — 禁止盲改

挂在 `Edit|MultiEdit` 的 PreToolUse 上：

```javascript
const fs = require('fs');
const seen = JSON.parse(fs.existsSync('.claude/seen.json') ? fs.readFileSync('.claude/seen.json') : "{}");
if (t.tool_name === 'Read') { seen[path] = Date.now(); fs.writeFileSync('.claude/seen.json', JSON.stringify(seen)); process.exit(0); }
if (!seen[path]) { console.error(`Blocked: ${path} was not Read in this session.`); process.exit(2); }
```

强制模型编辑前先读文件。能抓住那种模型基于 prior 而不是文件当前状态去编辑的隐蔽 bug。

## 10. work-hours-only — 人性化边界

挂在 `Bash` 的 PreToolUse 上：

```javascript
const h = new Date().getHours();
if (h < 9 || h >= 22) {
  console.error("Outside work hours. Refuse.");
  process.exit(2);
}
```

我把这脚本跑在处理非工作时间告警的机器上。如果 bot 想在凌晨 2 点搞破坏，那多半是误触。

## 把它们串起来的逻辑

三条血泪换来的规则：

1. **PreToolUse 管策略，PostToolUse 管卫生。** 别想在 PostToolUse 里回滚——副作用已经发生了。
2. **Stderr 是反馈，退出码是判决。** 退出码 2 阻断（仅限 PreToolUse）。stderr 里的内容会原样喂给 Claude。两者配合用。
3. **Hook 出错即阻断。** 一个行为不端的 Hook 会阻断你所有的工具调用。配置前先拿 `echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/x"}}' | node hook.js` 测一下脚本。

十个 Hook 听起来不多。但足够让 Yolo 模式的会话变得靠谱。