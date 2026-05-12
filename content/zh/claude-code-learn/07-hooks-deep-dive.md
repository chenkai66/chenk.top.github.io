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
第五章我们梳理了 Hook 的基本概念，本章则聚焦实战应用。在收录百余个脚本的参考库中，仅有这十个被我纳入所有严肃项目的标准配置。我们逐一介绍这十个 Hook，并附上代码级操作说明。

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

![Hook 的 I/O 契约：stdin 输入 JSON，退出码 + stderr 输出](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/fig3.png)
*每个 Hook 都是一个从 stdin 读取 JSON 载荷的脚本，通过退出码（判决）和 stderr（说明）回传结果。settings.json 里的 matcher 决定哪些 Hook 能看到本次工具调用。*

## 1. block-env-read — 保护 secrets

这是性价比最高（ROI 最高）的一个 Hook。防止 `Read` 和 `Grep` 触碰 `.env`、`id_rsa`、`credentials.json`：

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

把它挂在 `Read|Grep|MultiEdit|Edit|Write` 上。在 PreToolUse 阶段，退出码 2 将阻断工具调用；stderr 输出的内容会原样传递给模型，用于说明拦截原因。

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

正则表达式列表刻意保持简短。过长的黑名单容易因误报率升高而被实际忽略。

## 3. bash-whitelist — 适合生产环境周边的机器

反过来，适合那些要碰生产环境的 repo。只允许明确列出的二进制文件：

```javascript
const allow = new Set(['ls','cat','grep','rg','git','npm','node','python','python3','curl','jq','head','tail']);
const first = (cmd.trim().split(/\s+/)[0] || "").split('/').pop();
if (!allow.has(first)) { console.error(`Not on allowlist: ${first}`); process.exit(2); }
```

白名单的优势在于弥补黑名单的不足：它天然杜绝意外放行未经显式允许的新命令。

## 4. block-git-push — 禁止意外 push

我从来没想过让 Claude 不打招呼就直接 push。挂在 `Bash` 的 PreToolUse 上：

```javascript
if (/^\s*git\s+push\b/.test(cmd) || /git.*push.*--force/.test(cmd)) {
  console.error("Blocked: git push must be human-initiated.");
  process.exit(2);
}
```

写错一次的代价，远比我自己敲一遍 `git push` 麻烦得多。

![退出码在 PreToolUse 与 PostToolUse 中的语义差异](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/fig4.png)
*同样的退出码在不同生命周期阶段含义完全不同。exit 2 只在 PreToolUse 阶段阻断调用；在 PostToolUse 阶段副作用已经发生，无法回滚。*

## 5. format-on-write — 把 Prettier 做成 PostToolUse

挂在 `Write|Edit|MultiEdit` 的 PostToolUse 上：

```javascript
const path = t.tool_input?.file_path;
if (path && /\.(ts|tsx|js|jsx|json|md|css)$/.test(path)) {
  require('child_process').execSync(`npx prettier --write ${path}`, { stdio: 'inherit' });
}
```

PostToolUse 在编辑操作完成后触发，此时退出码 2 已无法回滚变更——它的作用是保障代码质量（‘卫生’），而非执行访问控制（‘策略’）。

## 6. test-on-edit — 快速失败

针对源文件，挂在 `Edit|MultiEdit` 的 PostToolUse 上：

```javascript
if (/\/(src|lib)\/.*\.(ts|js)$/.test(path)) {
  try { require('child_process').execSync('npm run -s test:related -- ' + path, { stdio: 'inherit' }); }
  catch { console.error("Tests failed after edit"); process.exit(1); }
}
```

退出码 1 会将测试失败信息返回给模型，模型据此输出可触发自动重试。长期使用该 Hook，显著提升了 Claude 在本仓库中生成代码的质量。

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

该日志文件日常无需关注，但在排查问题时至关重要。

![read-before-write 状态机：UNSEEN / FRESH / STALE](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/fig6.png)
*Hook 在 `seen.json` 中维护每个文件最近一次 Read 的时间戳。对 UNSEEN 或 STALE 文件的编辑会被 exit 2 阻断，stderr 直接告诉 Claude 该如何补救。*

## 9. read-before-write — 禁止盲改

挂在 `Edit|MultiEdit` 的 PreToolUse 上：

```javascript
const fs = require('fs');
const seen = JSON.parse(fs.existsSync('.claude/seen.json') ? fs.readFileSync('.claude/seen.json') : "{}");
if (t.tool_name === 'Read') { seen[path] = Date.now(); fs.writeFileSync('.claude/seen.json', JSON.stringify(seen)); process.exit(0); }
if (!seen[path]) { console.error(`Blocked: ${path} was not Read in this session.`); process.exit(2); }
```

强制模型编辑前先读文件。可捕获一类隐蔽缺陷：模型依赖先验知识（prior）而非文件当前实际内容进行编辑。

## 10. work-hours-only — 人性化边界

挂在 `Bash` 的 PreToolUse 上：

```javascript
const h = new Date().getHours();
if (h < 9 || h >= 22) {
  console.error("Outside work hours. Refuse.");
  process.exit(2);
}
```

该脚本部署在专门处理非工作时间告警的机器上。若 bot 在凌晨 2 点尝试执行操作，极大概率是误触发。

![Edit 调用的 Hook 执行顺序：PreToolUse → 工具执行 → PostToolUse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/fig5.png)
*一次 Edit 会依次穿过六个 Hook：三个 PreToolUse 守门员、工具本体、三个 PostToolUse 卫生作业。任何 PreToolUse 阶段的 exit 2 都会让整条链路中断。*

## 把它们串起来的逻辑

以下三条经验规则源于实际踩坑：

![Hook 调试四步法以及最常见的五类错误](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/claude-code-learn/07-hooks-deep-dive/fig7.png)
*在隔离环境里把失败的调用回放给 Hook，捕获 stderr，再用 `claude --debug` 和 JSONL 日志逐层排查。下方表格覆盖了 90% 的 Hook 故障原因。*


1. **PreToolUse 管策略，PostToolUse 管卫生。** 别想在 PostToolUse 里回滚——副作用已经发生了。
2. **Stderr 是反馈，退出码是判决。** 退出码 2 阻断（仅限 PreToolUse）。stderr 里的内容会原样喂给 Claude。两者配合用。
3. **Hook 出错即阻断。** 一个行为不端的 Hook 会阻断你所有的工具调用。配置前先拿 `echo '{"tool_name":"Read","tool_input":{"file_path":"/tmp/x"}}' | node hook.js` 测一下脚本。

十个 Hook 数量看似有限，却足以将‘YOLO’（You Only Live Once）式的随意会话转化为可控、可靠的工程实践。