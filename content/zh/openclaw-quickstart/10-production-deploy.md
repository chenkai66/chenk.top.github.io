---
title: "OpenClaw 指南（十）：生产部署与故障模式"
date: 2026-04-17 09:00:00
tags:
  - openclaw
  - production
  - ECS
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
本地安装只能保证‘在我机器上能跑’，但服务器在内核升级等系统变更后能否持续稳定运行，才是真正的考验。

本章介绍我在 2 核 4G ECS 实例上的实际部署方案，以及实践中高频出现且必须记录和防范的八类典型故障模式。

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_1.png)

## 选服务器

![Production deployment stack — from OS to monitoring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/fig_deploy.png)

部署前先选对机器。下面这四个选项值得考虑：

**Alibaba Cloud ECS**：这是我实际采用的方案。北京区 2 核 4G 实例大概 15 刀/月。优势在于地理位置靠近 DashScope——网关与模型服务部署在同一地域，API 往返延迟可从约 200ms 降至约 20ms。劣势是受网络策略限制，出向连接有时不稳定，可能导致依赖安装失败；配置国内镜像源可显著缓解该问题。

**DigitalOcean**：上手最简单。 18 刀/月的 "Basic" droplet （2vCPU, 4GB）功能上和 ECS 没区别。控制台更干净，文档更好，也不需要配包镜像。代价是延迟——如果你的模型提供商是阿里或腾讯，每轮对话会多花 150ms。

**Hetzner**：性价比之王。 CX21 实例（2vCPU, 4GB, 纽伦堡）只要 5.83 刀/月。问题在于 Hetzner 的网络优化主要面向欧洲地区，如果你的用户或模型端点在亚洲，延迟会非常明显。

**Home server**：硬件零成本，完全自主可控，磁盘空间理论上无上限。但可用性受限于家庭宽带服务商（通常不承诺 99.999% SLA），IP 地址为动态分配（需配置 DDNS 解析），且家用路由器常默认屏蔽入向 443 端口。家用服务器适合原型验证；除非服务对象仅为单个用户，否则不建议用于生产环境。

**最低配置： 2 核 4G。** 为什么？网关自身常驻内存约 200MB，但模型响应在流式传输至客户端前需全部缓存在 RAM 中；此外，当子 agent fork 时，操作系统会通过写时复制（COW）机制瞬时复制父进程内存页。我们曾观察到： 2GB 内存实例在执行代码审查任务时，因并行启动三个子 agent 而触发 OOM。 4G 给你留了余量， 8G 内存则可完全避免此类 OOM 风险。

## 部署步骤

系统：Ubuntu 22.04。4G 内存很关键——2G 跑一个 agent 还行，一旦 spawn 子 agent 就会卡死。

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

使用 pm2 管理网关进程，确保单点崩溃不会影响整体服务可用性。

```bash
pm2 start "openclaw gateway" --name openclaw-gateway --time
pm2 save
pm2 startup
```

Web Dashboard（端口 18789）前面挂 nginx，证书用 acme.sh 生成。

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

必须将读取超时设为 600 秒，因为长周期 agent 轮次耗时可能超过 Nginx 默认的 60 秒超时阈值，否则流式响应将被中断。

## Docker 方案

若倾向容器化部署，OpenClaw 原生支持 Docker。这是我多服务部署用的 compose 文件。

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

三个 volume 至关重要。

- `config/` 存 API 密钥、渠道凭证、模型端点。不挂载这个，每次容器重启都会清空配置。
- `workspace/` 包含 MEMORY.md 和会话日志。丢了这个， agent 重启后就会失忆。
- `skills/` 存自定义技能。如果只读挂载， agent 能读技能，但在自我改进时无法写入新技能。

**什么时候 Docker 更好：** 你让 OpenClaw 和其他服务（数据库、向量库、监控栈）一起跑，想要一条 `docker-compose up` 全部上线。隔离性也让测试配置变更更容易——起第二个容器配不同 config，对比行为，干掉差的那个。

**什么时候 Docker 更糟：** 调试文件权限——容器以 root 运行，宿主机文件可能不是，你得花时间 `chown`。查看日志需要 `docker exec` 或挂载 volume。开发时热重载技能更慢，因为文件系统同步有延迟。对于单服务部署场景，直接 SSH 登录服务器并用 tail 查看日志，相比容器方案更轻量，调试也更直观。

## 八大故障

![OpenClaw QuickStart (10): Production Deploy and the Failure Modes Nobody Warns You About — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_2.png)

### 1. 重启后 `command not found: openclaw`

nvm 默认不会在非交互式 Shell 中初始化，而 pm2 的开机自启正是通过非交互式 Shell 触发的。要么在 `/etc/profile.d/` 里 source nvm，要么给二进制文件建软链：

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

**检测：** `pm2 status` 显示网关在启动后立即处于 `errored` 状态，退出码 127。

### 2. `Node.js version too old`

OpenClaw 需要 >= 22.16。错误信息明确指出版本不兼容；根本原因是使用了发行版预装的 Node.js，其版本通常过低。

**检测：** 网关启动失败，`pm2 logs openclaw-gateway --err` 前三行显示版本不匹配。

### 3. DashScope 返回 `401 Unauthorized`

两个原因： Coding Plan 密钥用错了端点，或者密钥轮换后没更新。

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

上游 NAT 设备主动终止了长轮询连接：

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

三个原因：默认模型太贵， MEMORY.md 臃肿，或者琐碎任务也 spawn 子 agent。

**检测：** 用量稳定但账单周环比翻倍。运行 `openclaw stats tokens --since 7d` 对比每轮平均值。如果对话型 agent 每轮超过 8k tokens，肯定有问题。 grep MEMORY.md 看长度：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 500 行就是红旗。

### 8. 内存无限增长

若长期不归档会话， MEMORY.md 文件将持续膨胀：突破 100 行后，迅速增至 200 行、 500 行乃至更多。每轮 agent 对话现在都包含半 KB 无关上下文（“三周前用户问过 Docker"）。启动变慢，因为 workspace loader 启动时要解析整个 memory 文件。 1000 行时启动要 30 秒。 2000 行时， agent 开始在对话中途超时，因为 context window 80% 是 memory， 20% 才是实际任务。

**修复：** 自动化每周清理。加一个 cron  job 归档旧会话并修剪 MEMORY.md：

```bash
# Every Sunday at 2 AM
0 2 * * 0 openclaw memory archive --older-than 30d && openclaw memory compact --max-lines 100
```

`archive` 命令把 30 天前的会话移进 `.archive/` 子目录（仍可读，只是默认不加载）。`compact` 命令用 LLM 把 MEMORY.md 总结到 100 行，保留最重要事实，丢弃低价值细节。

**检测：** `openclaw gateway` 打印 "Gateway listening on 18789" 耗时超过 10 秒。或者直接查文件大小：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 300 行值得人工审查。超过 500 行，立即 compact。

## 升级路径

OpenClaw 迭代频繁，每周均发布新功能；部分版本会变更配置结构（config schema）或废弃特定技能（skill）字段。推荐的安全升级流程如下：

1.  **查 changelog：** `openclaw changelog --since <current-version>`。找 breaking changes、废弃字段或新必填 config 键。

2.  **备份配置：** `cp -r ~/.openclaw/config ~/.openclaw/config.backup`。升级搞砸了，十秒就能恢复。

3.  **升级二进制：** `npm i -g openclaw@latest`。这拉取新版本但不会重启任何服务。

4.  **重启网关：** `pm2 restart openclaw-gateway`。盯着前 60 秒日志：`pm2 logs openclaw-gateway --lines 100`。如果看到重复错误或 crash loop，回滚：`npm i -g openclaw@<old-version> && pm2 restart openclaw-gateway`。

5.  **验证健康：** 访问 `https://agent.example.com` 发测试消息。确认技能加载、记忆持久化、 agent 响应连贯。

**升级时会坏什么：**

-   **Config schema 变更：** 字段重命名（`model.name` 变 `model.id`），网关解析配置失败。报错通常会告诉你是哪个字段无效。修复：更新配置，重启。
  
-   **废弃 skill 字段：** 自定义技能用了 `skill.parameters` 但新版本期望 `skill.input`。技能 loader 抛验证错误。修复：用 `openclaw skill create` 重新生成技能或手动更新 schema。

-   **依赖冲突：** 少见，但会发生。新 OpenClaw 版本需要的库和你全局装的其他东西冲突。症状：升级成功，但网关启动时因模块解析错误崩溃。修复：用 `nvm` 隔离 Node 环境，或直接在 Docker 里跑 OpenClaw 避免全局安装。

**黄金法则：** 别在高峰期升级。周日凌晨 2 点做，这时候五分钟停机没人看得见。
## 监控与告警

缺乏监控的线上服务，意味着无法有效掌控其运行状态。分三层来做：

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

要是服务多，得上正经监控栈（Prometheus, Grafana, Uptime Kuma）。但单代理部署的话， bash 脚本加个 mail 命令就够了。

### 日志轮转

gateway 往 stdout 写日志， pm2 负责捕获。要是不做轮转，`~/.pm2/logs/openclaw-gateway-out.log` 会无限膨胀。撑不过三个月，这文件就能涨到 2GB，直接把你的磁盘塞满。

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

生产环境中的 OpenClaw 并非追求炫技的软件，而是一款能够连续稳定运行三十天的可靠系统。按部就班部署，配好 supervisor，前面挂个稳定的 proxy，把上面那八个故障隐患提前解决，监控自动化，你就能搞定。

QuickStart 就到这儿。从这里开始，路就分叉了——定制技能、定制渠道、定制 MCP 服务器、多代理拓扑。所有这些，都建立在刚才这十块基石之上。