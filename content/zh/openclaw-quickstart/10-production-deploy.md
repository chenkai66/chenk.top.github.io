---
title: "OpenClaw 快速上手（十）：上生产部署，以及没人提醒你的那些故障模式"
date: 2026-04-12 09:00:00
tags:
  - openclaw
  - production
  - ecs
  - nginx
  - troubleshooting
categories: OpenClaw
lang: zh-CN
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 10
description: "把 OpenClaw 放到一台真正的服务器上：ECS、pm2 当 supervisor、nginx 反代、acme.sh 续证书。然后是长尾——我在生产至少见过两次的七种故障，每一种到底是什么。"
disableNunjucks: true
translationKey: "openclaw-quickstart-10"
---
本地安装只能保证“在我机器上能跑”。服务器安装才是确保它在内核更新后依然能用的关键。

这一章我会详细讲一下我自己在一台 2 核 4G ECS 上的部署流程，接着列出一些我遇到过足够多次、觉得有必要写下来的故障场景。最后我会分享一个每天早上我都快速检查一遍的健康脚本，以及备份策略——这两点如果不写清楚，过几个月你大概率会后悔。
![OpenClaw 快速上手（十）：上生产部署，以及没人提醒你的那些故障模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_1.jpg)

## 部署

操作系统：Ubuntu 22.04。内存至少要 4G，2G 跑单个 Agent 还凑合，但子 Agent 一启动就卡死了。我用的是阿里云 ECS t6 规格，按量计费，香港区，因为 Telegram 和 GitHub 都能通。

```bash
# 1. 用 nvm 安装 Node 22——大多数发行版自带的 Node 版本太老
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22 && nvm use 22

# 2. 全局安装 OpenClaw 和 pm2
npm i -g openclaw@latest pm2

# 3. 初始化工作区
openclaw init
# 编辑 ~/.openclaw/openclaw.json，至少设置好模型提供方
```

pm2 用来监控 Gateway，崩溃了也不会拖垮整个系统：

```bash
pm2 start "openclaw gateway" --name openclaw-gateway --time
pm2 save
pm2 startup           # 按照打印的 sudo 命令执行
```

加上 `--time` 参数后，pm2 的日志每行都会带时间戳。排查问题时你会庆幸自己加了这个参数。

Web Dashboard 默认跑在 18789 端口。前面用 nginx 反向代理，证书用 acme.sh 生成：

```nginx
server {
  listen 443 ssl http2;
  server_name agent.example.com;

  ssl_certificate     /etc/nginx/ssl/agent.example.com.cer;
  ssl_certificate_key /etc/nginx/ssl/agent.example.com.key;

  location / {
    proxy_pass http://127.0.0.1:18789;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 600s;
  }
}
```

600 秒的读取超时不是可选项。长时间运行的 Agent 对话会超过 nginx 默认的 60 秒限制，连接会被掐断。我就踩过坑：第一次部署时没改这个值，复杂任务跑到一半就返回 504，前端看起来像是 Agent 挂了，其实是 nginx 把它干掉了。

防火墙只开放 443 端口，18789 和 18790 全部本地化。安全组别抱侥幸心理——OpenClaw Web Dashboard 默认没有认证，暴露到公网等于把家门钥匙扔街上。
## 七种故障

![OpenClaw 快速上手（十）：上生产部署，以及没人提醒你的那些故障模式 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_2.jpg)

### 1. 重启后报 `command not found: openclaw`

nvm 在非交互式 shell 中不会自动加载，而 pm2 startup 用的就是这种 shell。解决办法有两个：要么把 nvm 配置到 `/etc/profile.d/`，要么直接创建软链接：

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

我更喜欢软链接的方式，简单直接。

### 2. `Node.js version too old`

OpenClaw 要求 Node.js 版本 ≥ 22.16。错误信息很清楚，但真正的问题是用了系统自带的 Node.js。建议固定使用 Node 22，并在 pm2 使用的同一个 shell 中检查 `node -v`。

### 3. DashScope 返回 `401 Unauthorized`

主要有两个原因：

- Coding Plan 的密钥（`sk-sp-...`）被错误地用到了标准 DashScope 的基础 URL 上。两者的端点不同：Coding Plan 对应 `https://coding.dashscope.aliyuncs.com/v1`，普通 DashScope 对应 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- 密钥泄露后被轮换，但未及时更新。去控制台检查一下。

我自己踩过第二个坑——不小心把密钥提交到公开的 GitHub 仓库，阿里云 30 秒内就检测到并禁用了它。虽然 Gateway 报错是对的，但我第一反应以为是配置问题。所以记住：**遇到 401 先去后台查密钥状态**。

### 4. Gateway 启动时报 `Connection refused`

18789 端口被占用了。找到占用进程并杀掉：

```bash
lsof -i :18789
kill $(lsof -t -i :18789)
pm2 restart openclaw-gateway
```

如果占用的是另一个 OpenClaw 实例，那可能是你今天早些时候在 pm2 外手动运行了 `openclaw gateway`。停掉它，让 pm2 接管。

### 5. 钉钉 30 分钟后没声音了

长连接被上游 NAT 断开了。可以通过以下两种方式修复：

```json
"dingtalk": {
  "reconnectMs": 60000,
  "heartbeatMs":  30000
}
```

如果你能控制网络，建议将出口 IP 固定为单一地址。出口 IP 轮换才是真正的罪魁祸首，比“网络抖动”靠谱得多。

### 6. Agent 中途忘记内容

压缩运行时没有启用 `memoryFlush`。参考[第 7 篇](../07-memory-system/)，设置 `memoryFlush.enabled: true` 并配置一个合理的 `softThresholdTokens`。这一行配置决定了 Agent 是“忘了”还是“记住了”。

### 7. `Token 消耗过高`

按可能性从高到低，主要有三个原因：

1. 每一轮都用了最贵的模型。改成分级配置——用 `qwen3.5-flash` 做路由，仅在任务需要时调用 `qwen3-max`。
2. `MEMORY.md` 文件超过 40 行，且每轮都被加载。检查一下这个文件。
3. 子 Agent 被滥用于处理琐碎任务。把这些任务内联化。

如果以上都不是问题，打开每轮的 token 日志，查看实际分布。结果几乎从来不是你以为的那个。我曾经遇到过一个案例，怎么也想不通为什么消耗这么快——最后发现是一个 Skill 把整个 README 当上下文塞进去，每轮多出 8k tokens，跑了三天烧掉 200 块。
## 体检五行

每天早上我都会跑这五条命令，检查一下系统状态：

```bash
pm2 status openclaw-gateway
openclaw doctor
wc -l ~/.openclaw/workspace/MEMORY.md
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
df -h /
```

`pm2 status` 查看 supervisor 是否正常。`openclaw doctor` 执行内置的健康检查。`wc -l` 统计 MEMORY.md 的行数，防止它悄悄膨胀。session 文件的数量告诉我是否需要归档。`df -h` 检查磁盘使用情况，日志迟早会把磁盘撑满。

我把这五条命令写进了 `~/bin/oc-check`，每天 SSH 登录后直接敲 `oc-check` 就行。两秒钟的事，省下的麻烦可不是一点两点。
## 备份与轮转

OpenClaw 的所有“不可再生”数据都存放在 `~/.openclaw/` 目录下：包括配置、记忆、Skills 和会话。其他内容都可以重新安装，所以备份时只需要关注这一个目录：

```bash
# 每天凌晨备份一次到 OSS
0 3 * * * tar czf /tmp/oc-$(date +\%F).tgz ~/.openclaw && \
  ossutil cp /tmp/oc-$(date +\%F).tgz oss://ck-planet/openclaw-backup/ && \
  rm /tmp/oc-$(date +\%F).tgz
```

日志需要单独处理一下。pm2 默认会把所有的 stdout/stderr 输出写入 `~/.pm2/logs/`，而且不会自动轮转。我建议安装 `pm2-logrotate` 插件：

```bash
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

如果不做这个设置，两个月后你会发现根分区被 pm2 日志占满，Gateway 因为无法写入数据直接崩溃。
## 收尾

生产用的 OpenClaw 不是聪明的软件，是**已经稳跑了三十天的无聊软件**。按本子做部署、装好 supervisor、前面套个稳定代理、把上面七种故障在被咬之前先治好，你就到了。

QuickStart 系列到这儿结束。再往后路就分叉了——自定义 Skill、自定义渠道、自定义 MCP server、多 Agent 拓扑。所有这些都建立在这十篇打下的地基上。

最后一句话送给你：**Agent 不是越聪明越好用，是越无声越好用**。它在那里、它在跑、它需要你的时候才出现——这才是这套东西做出来的全部意义。
