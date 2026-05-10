---
title: "OpenClaw 快速上手（十）：生产部署与没人提前告诉你的故障模式"
date: 2026-04-12 09:00:00
tags:
  - openclaw
  - production
  - ecs
  - nginx
  - troubleshooting
categories: OpenClaw
lang: zh
mathjax: false
series: openclaw-quickstart
series_title: "OpenClaw 快速上手"
series_order: 10
description: "ECS + pm2 + nginx + acme.sh 上线部署，加七个生产环境常见故障及其排查方法。"
disableNunjucks: true
translationKey: "openclaw-quickstart-10"
---
本地安装只能保证“在我机器上能跑”，服务器上能扛住内核升级才是真本事。

这一章我就聊聊自己在 2 核 4G ECS 上实际用的部署方案，还有那些踩得多了不得不记下来的故障模式。

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_1.png)

## 选服务器

![Production deployment stack — from OS to monitoring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/fig_deploy.png)

部署前先选对机器。下面这四个选项值得考虑：

**Alibaba Cloud ECS**：这是我用的路子。北京区 2 核 4G 实例大概 15 刀/月。优势是离 DashScope 近——网关和模型都在同一区域，API 往返延迟能从 200ms 降到 20ms。劣势是大防火墙偶尔会让 outbound 包安装变得不稳定，除非你配置了镜像源。

**DigitalOcean**：上手最简单。18 刀/月的 "Basic" droplet（2vCPU, 4GB）功能上和 ECS 没区别。控制台更干净，文档更好，也不需要配包镜像。代价是延迟——如果你的模型提供商是阿里或腾讯，每轮对话得多花 150ms。

**Hetzner**：性价比之王。CX21 实例（2vCPU, 4GB, 纽伦堡）只要 5.83 刀/月。坑在于 Hetzner 的网络优化针对欧洲，如果你的用户或模型端点在亚洲，延迟会很明显。

**Home server**：硬件免费，控制权全，磁盘无限。问题是 uptime（ISP 不承诺五个九），动态 IP（你得配 DDNS），还有端口转发（路由器可能会 blocking  inbound 443）。家用服务器适合原型验证；除非你是唯一用户，否则别拿来跑生产。

**最低配置：2 核 4G。** 为什么？网关本身只占 200MB 常驻内存，但模型响应在流式传输给客户端前会缓冲在 RAM 里，而且当子 agent fork 时，操作系统会瞬间复制父进程。我见过 2G 实例在做代码 review 任务时，因为并行 spawned 三个子 agent 直接 OOM。4G 给你留了余量，8G 则彻底解决这个问题。

## 部署步骤

系统：Ubuntu 22.04。4G 内存很关键——2G 跑一个 agent 还行，一旦 spawn 子 agent 就卡死。

```bash
# 1. Node 22 via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22 && nvm use 22

# 2. OpenClaw + pm2 globally
npm i -g openclaw@latest pm2

# 3. Workspace
openclaw init
```

pm2 负责 supervise Gateway，确保崩溃不会拖垮整个服务：

```bash
pm2 start "openclaw gateway" --name openclaw-gateway --time
pm2 save
pm2 startup
```

Web Dashboard（端口 18789）前面挂 nginx，证书用 acme.sh 生成：

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

600s 读取超时不是可选项——长运行的 agent 轮次会超过 nginx 默认的 60s，导致流式传输中途失败。

## Docker 方案

如果你更喜欢容器，OpenClaw 自带 Docker 支持。这是我多服务部署用的 compose 文件：

```yaml
version: '3.8'
services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw-gateway
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - ./config:/root/.openclaw/config
      - ./workspace:/root/.openclaw/workspace
      - ./agents:/root/.openclaw/agents
      - ./skills:/root/.openclaw/skills
    environment:
      - NODE_ENV=production
```

三个 volume 至关重要：

- `config/` 存 API 密钥、渠道凭证、模型端点。不挂载这个，每次容器重启都会清空配置。
- `workspace/` 包含 MEMORY.md 和会话日志。丢了这个，agent 重启后就会失忆。
- `skills/` 存自定义技能。如果只读挂载，agent 能读技能，但在自我改进时无法写入新技能。

**什么时候 Docker 更好：** 你让 OpenClaw 和其他服务（数据库、向量库、监控栈）一起跑，想要一条 `docker-compose up` 全部上线。隔离性也让测试配置变更更容易——起第二个容器配不同 config，对比行为，干掉差的那个。

**什么时候 Docker 更糟：** 调试文件权限——容器以 root 运行，宿主机文件可能不是，你得花时间 `chown`。查看日志需要 `docker exec` 或挂载 volume。开发时热重载技能更慢，因为文件系统同步有延迟。对于单服务部署，直接 ssh 进机器 tail 日志，裸机更简单。

## 八大故障

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_2.png)

### 1. 重启后 `command not found: openclaw`

nvm 不会在非交互式 shell 中加载。pm2 startup 用的就是非交互式 shell。要么在 `/etc/profile.d/` 里 source nvm，要么给二进制文件建软链：

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

**检测：** `pm2 status` 显示网关在启动后立即处于 `errored` 状态，退出码 127。

### 2. `Node.js version too old`

OpenClaw 需要 >= 22.16。报错很清楚；真正的 bug 是用了 distro 自带的 Node。

**检测：** 网关启动失败，`pm2 logs openclaw-gateway --err` 前三行显示版本不匹配。

### 3. DashScope 返回 `401 Unauthorized`

两个原因：Coding Plan 密钥用错了端点，或者密钥轮换后没更新。

**检测：** 每轮 agent 对话瞬间失败，响应带 401。检查 `~/.openclaw/agents/main/sessions/*.jsonl` —— 如果每个会话最后一行都是 auth 错误，那就是密钥问题。

### 4. 网关启动时 `Connection refused`

端口 18789 被占用了：

```bash
lsof -i :18789
kill $(lsof -t -i :18789)
pm2 restart openclaw-gateway
```

**检测：** `pm2 logs openclaw-gateway --err` 在启动第一秒内显示 `EADDRINUSE`。网关永远达不到 "listening on 18789" 那行日志。

### 5. 钉钉 30 分钟后沉默

上游 NAT 正在 tearing down 长轮询连接：

```json
"dingtalk": {
  "reconnectMs": 60000,
  "heartbeatMs":  30000
}
```

**检测：** 网关日志显示 `[dingtalk] reconnecting...` 每小时超过一次。用户报告“机器人没反应了”，但 Web  dashboard 手动发消息依然有效。

### 6. 对话中途 agent 失忆

Compaction 运行了但没开 `memoryFlush`。设置 `memoryFlush.enabled: true`。

**检测：** 多轮对话在第 15 轮后突然丢失上下文。检查会话长度：`cat ~/.openclaw/agents/main/sessions/<session-id>.jsonl | wc -l`。如果正好 20 行（默认 compaction 阈值），说明 compaction 丢弃了轮次而不是总结。

### 7. `Token consumption is way too high`

三个原因：默认模型太贵，MEMORY.md 臃肿，或者琐碎任务也 spawn 子 agent。

**检测：** 用量稳定但账单周环比翻倍。运行 `openclaw stats tokens --since 7d` 对比每轮平均值。如果对话型 agent 每轮超过 8k tokens，肯定有问题。grep MEMORY.md 看长度：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 500 行就是红旗。

### 8. 内存无限增长

从不归档会话会发生什么：MEMORY.md 膨胀超过 100 行，然后 200，然后 500。每轮 agent 对话现在都包含半 KB 无关上下文（“三周前用户问过 Docker"）。启动变慢，因为 workspace loader 启动时要解析整个 memory 文件。1000 行时启动要 30 秒。2000 行时，agent 开始在对话中途超时，因为 context window 80% 是 memory，20% 才是实际任务。

**修复：** 自动化每周清理。加一个 cron  job 归档旧会话并修剪 MEMORY.md：

```bash
# Every Sunday at 2 AM
0 2 * * 0 openclaw memory archive --older-than 30d && openclaw memory compact --max-lines 100
```

`archive` 命令把 30 天前的会话移进 `.archive/` 子目录（仍可读，只是默认不加载）。`compact` 命令用 LLM 把 MEMORY.md 总结到 100 行，保留最重要事实，丢弃低价值细节。

**检测：** `openclaw gateway` 打印 "Gateway listening on 18789" 耗时超过 10 秒。或者直接查文件大小：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 300 行值得人工审查。超过 500 行，立即 compact。

## 升级路径

OpenClaw 迭代很快。每周都有新功能，偶尔发布版会改 config schema 或废弃 skill 字段。安全升级流程如下：

1.  **查 changelog：** `openclaw changelog --since <current-version>`。找 breaking changes、废弃字段或新必填 config 键。

2.  **备份配置：** `cp -r ~/.openclaw/config ~/.openclaw/config.backup`。升级搞砸了，十秒就能恢复。

3.  **升级二进制：** `npm i -g openclaw@latest`。这拉取新版本但不会重启任何服务。

4.  **重启网关：** `pm2 restart openclaw-gateway`。盯着前 60 秒日志：`pm2 logs openclaw-gateway --lines 100`。如果看到重复错误或 crash loop，回滚：`npm i -g openclaw@<old-version> && pm2 restart openclaw-gateway`。

5.  **验证健康：** 访问 `https://agent.example.com` 发测试消息。确认技能加载、记忆持久化、agent 响应连贯。

**升级时会坏什么：**

-   **Config schema 变更：** 字段重命名（`model.name` 变 `model.id`），网关解析配置失败。报错通常会告诉你是哪个字段无效。修复：更新配置，重启。
  
-   **废弃 skill 字段：** 自定义技能用了 `skill.parameters` 但新版本期望 `skill.input`。技能 loader 抛验证错误。修复：用 `openclaw skill create` 重新生成技能或手动更新 schema。

-   **依赖冲突：** 少见，但会发生。新 OpenClaw 版本需要的库和你全局装的其他东西冲突。症状：升级成功，但网关启动时因模块解析错误崩溃。修复：用 `nvm` 隔离 Node 环境，或直接在 Docker 里跑 OpenClaw 避免全局安装。

**黄金法则：** 别在高峰期升级。周日凌晨 2 点做，这时候五分钟停机没人看得见。
## 监控与告警

没法监控的线上服务，就等于没法掌控。分三层来做：

### 健康检查

弄个 cron 任务，每五分钟 ping 一次 gateway，挂了就直接报警：

```bash
#!/bin/bash
# /usr/local/bin/openclaw-healthcheck.sh

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18789/health)

if [ "$STATUS" -ne 200 ]; then
  echo "OpenClaw gateway is down (HTTP $STATUS)" | mail -s "Alert: OpenClaw Down" you@example.com
fi
```

加到 cron 里：

```bash
*/5 * * * * /usr/local/bin/openclaw-healthcheck.sh
```

要是服务多，得上正经监控栈（Prometheus, Grafana, Uptime Kuma）。但单代理部署的话，bash 脚本加个 mail 命令就够了。

### 日志轮转

gateway 往 stdout 写日志，pm2 负责捕获。要是不做轮转，`~/.pm2/logs/openclaw-gateway-out.log` 会无限膨胀。撑不过三个月，这文件就能涨到 2GB，直接把你的磁盘塞满。

创建 `/etc/logrotate.d/openclaw`：

```
/home/ubuntu/.pm2/logs/openclaw-gateway-out.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
  postrotate
    pm2 reloadLogs
  endscript
}
```

这样保留七天日志，旧的压缩，轮转后通知 pm2 重新打开日志文件。

### 磁盘空间告警

代理会把会话日志写到 `~/.openclaw/agents/main/sessions/`。要是跑个热门代理，这目录每天涨 10MB。一年就是 3.6GB。要是你服务器根分区只有 20GB，迟早爆盘。

把磁盘检查加到刚才的健康检查脚本里：

```bash
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$USAGE" -gt 85 ]; then
  echo "Disk usage is at ${USAGE}% on /" | mail -s "Alert: Disk Space Low" you@example.com
fi
```

告警触发时，要么归档旧会话（`openclaw memory archive --older-than 90d`），要么扩容磁盘。

## “健康吗？”五连查

```bash
pm2 status openclaw-gateway
openclaw doctor
wc -l ~/.openclaw/workspace/MEMORY.md
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
df -h /
```

## 结语

生产环境的 OpenClaw 不是什么 clever 的软件，它就是那种能稳稳当当跑三十天的 boring 软件。按部就班部署，配好 supervisor，前面挂个稳定的 proxy，把上面那八个故障隐患提前解决，监控自动化，你就能搞定。

QuickStart 就到这儿。从这里开始，路就分叉了——定制技能、定制渠道、定制 MCP 服务器、多代理拓扑。所有这些，都建立在刚才这十块基石之上。