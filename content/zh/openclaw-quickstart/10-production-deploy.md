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
本地安装只能保证“在我机器上能跑”，而真正的考验在于：服务器在内核升级等系统变更后，是否还能持续稳定运行。

本章介绍我在 2 核 4GB ECS 实例上的实际部署方案，以及实践中高频出现、却很少有人提前预警的八类典型故障模式。

![OpenClaw 快速入门 (10)：生产部署及那些没人告诉你的故障模式 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_1.png)

## 选服务器

![生产部署堆栈 —— 从操作系统到监控](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/fig_deploy.png)

部署前先选对机器。下面这四个选项值得考虑：

**Alibaba Cloud ECS**：这是我实际采用的方案。北京区 2 核 4GB 实例月费约 15 美元。优势在于地理位置靠近 DashScope——当网关与模型服务部署在同一地域时，API 往返延迟可从约 200ms 降至 20ms。劣势是受网络策略影响，出向连接有时不稳定，可能导致依赖包安装失败；配置国内镜像源可显著缓解该问题。

**DigitalOcean**：上手最简单。18 美元/月的 “Basic” Droplet（2vCPU, 4GB）在功能上与 ECS 几乎一致。控制台更简洁，文档更清晰，也无需配置包镜像。代价是延迟——如果你的模型提供商是阿里云或腾讯云，每轮对话会额外增加约 150ms 延迟。

**Hetzner**：性价比之王。CX21 实例（2vCPU, 4GB，位于纽伦堡）月费仅 5.83 美元。但 Hetzner 的网络优化主要面向欧洲，若你的用户或模型端点位于亚洲，延迟会明显升高。

**Home server**：硬件零成本，完全自主可控，磁盘空间理论上无上限。但可用性受限于家庭宽带服务商（通常不承诺 99.999% SLA），IP 地址为动态分配（需配置 DDNS），且家用路由器常默认屏蔽入站 443 端口。家用服务器适合原型验证；除非服务对象仅为单个用户，否则不建议用于生产环境。

**最低配置：2 核 4GB。** 为什么？网关自身常驻内存约 200MB，但模型响应在流式传输至客户端前需全部缓存在 RAM 中；此外，当子 Agent fork 时，操作系统会通过写时复制（COW）机制瞬时复制父进程内存页。我们曾观察到：2GB 内存实例在执行代码审查任务时，因并行启动三个子 Agent 而触发 OOM。4GB 可提供安全余量，8GB 则能彻底规避此类问题。

## 部署步骤

系统：Ubuntu 22.04。4GB 内存很关键——2GB 跑一个 Agent 还行，一旦 spawn 子 Agent 就可能卡死。

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

使用 pm2 管理网关进程，确保单点崩溃不会导致服务整体中断。

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

必须将读取超时设为 600 秒，因为长周期 Agent 轮次耗时可能超过 Nginx 默认的 60 秒超时阈值，否则流式响应会被中途切断。

## Docker 方案

若倾向容器化部署，OpenClaw 原生支持 Docker。以下是我用于多服务部署的 compose 文件。

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

- `config/` 存放 API 密钥、渠道凭证和模型端点。若不挂载此目录，每次容器重启都会清空配置。
- `workspace/` 包含 MEMORY.md 和会话日志。若丢失此目录，Agent 重启后将彻底“失忆”。
- `skills/` 存储自定义技能。若以只读方式挂载，Agent 能读取技能，但在自我改进过程中无法写入新技能。

**什么时候 Docker 更好？** 当你让 OpenClaw 与其他服务（如数据库、向量库、监控栈）协同运行，并希望一条 `docker-compose up` 命令即可启动整套系统。容器隔离也让配置变更测试更轻松——启动第二个容器加载不同配置，对比行为后干掉表现较差的那个。

**什么时候 Docker 更糟？** 调试文件权限时——容器以 root 身份运行，而宿主机文件可能属于普通用户，你不得不花时间处理 `chown`。查看日志需借助 `docker exec` 或挂载 volume。开发阶段热重载技能也更慢，因为文件系统同步存在延迟。对于单服务部署场景，直接 SSH 登录服务器并用 `tail` 查看日志反而更轻量、调试更直观。

## 八大故障

![OpenClaw 快速入门 (10)：生产部署及那些没人提醒你的故障模式 — 图解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/illustration_2.png)

### 1. 重启后 `command not found: openclaw`

nvm 默认不会在非交互式 Shell 中初始化，而 pm2 的开机自启正是通过非交互式 Shell 触发的。要么在 `/etc/profile.d/` 中 source nvm，要么为二进制文件创建软链接：

```bash
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which node) /usr/local/bin/node
```

**检测：** `pm2 status` 显示网关在启动后立即处于 `errored` 状态，退出码为 127。

### 2. `Node.js version too old`

OpenClaw 要求 Node.js 版本不低于 22.16。错误信息通常很明确；真正的问题在于使用了发行版自带的旧版 Node.js。

**检测：** 网关启动失败，`pm2 logs openclaw-gateway --err` 的前三行显示版本不匹配。

### 3. DashScope 返回 `401 Unauthorized`

两个原因：Coding Plan 密钥被用于错误的端点，或密钥已轮换但未更新。

**检测：** 每轮 Agent 对话瞬间失败，响应中包含 401 错误。检查 `~/.openclaw/agents/main/sessions/*.jsonl` —— 如果每个会话的最后一行都是认证错误，那就是密钥问题。

### 4. 网关启动时 `Connection refused`

端口 18789 已被占用：

```bash
lsof -i :18789
kill $(lsof -t -i :18789)
pm2 restart openclaw-gateway
```

**检测：** `pm2 logs openclaw-gateway --err` 在启动第一秒内就显示 `EADDRINUSE`。网关始终无法打印出 “listening on 18789” 的日志。

### 5. 钉钉 30 分钟后沉默

上游 NAT 设备主动终止了长轮询连接：

```json
"dingtalk": {
  "reconnectMs": 60000,
  "heartbeatMs":  30000
}
```

**检测：** 网关日志中 `[dingtalk] reconnecting...` 出现频率超过每小时一次。用户反馈“机器人没反应了”，但从 Web Dashboard 手动发送消息仍能正常工作。

### 6. 对话中途 Agent 失忆

Compaction 已运行，但未启用 `memoryFlush`。请设置 `memoryFlush.enabled: true`。

**检测：** 多轮对话在第 15 轮左右突然丢失上下文。检查会话长度：`cat ~/.openclaw/agents/main/sessions/<session-id>.jsonl | wc -l`。如果恰好为 20 行（默认 compaction 阈值），说明 compaction 直接丢弃了历史轮次，而非进行摘要总结。

### 7. `Token consumption is way too high`

三个原因：使用了昂贵的默认模型、MEMORY.md 文件过于臃肿，或为琐碎任务也频繁 spawn 子 Agent。

**检测：** 使用量稳定，但账单周环比翻倍。运行 `openclaw stats tokens --since 7d` 并对比每轮平均 token 消耗。若对话型 Agent 每轮超过 8k tokens，肯定存在问题。检查 MEMORY.md 长度：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 500 行就是危险信号。

### 8. 内存无限增长

若长期不归档会话，MEMORY.md 文件将持续膨胀：突破 100 行后，迅速增至 200 行、500 行乃至更多。每轮 Agent 对话现在都夹带半 KB 的无关上下文（例如“三周前用户问过 Docker”）。启动变慢，因为 workspace loader 在启动时需解析整个 memory 文件。达到 1000 行时，启动耗时约 30 秒；到 2000 行时，Agent 开始在对话中途超时——因为上下文窗口中 80% 是记忆内容，仅 20% 是当前任务。

**修复：** 自动化每周清理。添加一个 cron job，定期归档旧会话并修剪 MEMORY.md：

```bash
# Every Sunday at 2 AM
0 2 * * 0 openclaw memory archive --older-than 30d && openclaw memory compact --max-lines 100
```

`archive` 命令将 30 天前的会话移入 `.archive/` 子目录（仍可读取，但默认不加载）。`compact` 命令则利用 LLM 将 MEMORY.md 摘要压缩至 100 行，保留关键事实，剔除低价值细节。

**检测：** `openclaw gateway` 打印 “Gateway listening on 18789” 耗时超过 10 秒。或直接检查文件行数：`wc -l ~/.openclaw/workspace/MEMORY.md`。超过 300 行就值得人工审查；超过 500 行，请立即执行 compact。

## 升级路径

OpenClaw 迭代迅速，每周都有新功能发布；部分版本会变更配置结构（config schema）或废弃特定技能（skill）字段。以下是推荐的安全升级流程：

1. **查看 changelog：** `openclaw changelog --since <current-version>`。重点关注破坏性变更、废弃字段或新增的必填配置项。
2. **备份配置：** `cp -r ~/.openclaw/config ~/.openclaw/config.backup`。万一升级出错，十秒内即可恢复。
3. **升级二进制：** `npm i -g openclaw@latest`。此命令仅拉取新版本，不会重启任何服务。
4. **重启网关：** `pm2 restart openclaw-gateway`。密切观察前 60 秒日志：`pm2 logs openclaw-gateway --lines 100`。若出现重复错误或崩溃循环，请立即回滚：`npm i -g openclaw@<old-version> && pm2 restart openclaw-gateway`。
5. **验证健康状态：** 访问 `https://agent.example.com` 发送测试消息，确认技能正常加载、记忆持久化有效、Agent 响应连贯。

**升级时可能出什么问题？**

- **配置结构变更：** 字段重命名（如 `model.name` 变为 `model.id`），导致网关无法解析配置。错误信息通常会指明无效字段。修复方法：更新配置后重启。
- **废弃技能字段：** 自定义技能使用了 `skill.parameters`，但新版本要求 `skill.input`。技能加载器会抛出验证错误。修复方法：使用 `openclaw skill create` 重新生成技能，或手动调整 schema。
- **依赖冲突：** 虽少见但确实存在。新版本 OpenClaw 所需的库与你全局安装的其他软件冲突。症状：升级成功，但网关启动时因模块解析失败而崩溃。修复方法：使用 `nvm` 隔离 Node.js 环境，或直接在 Docker 中运行 OpenClaw，彻底避免全局安装。

**黄金法则：** 切勿在业务高峰期升级。选择周日凌晨 2 点操作——此时五分钟的停机几乎无人察觉。

## 监控与告警

缺乏监控的线上服务，等于失控的服务。建议分三层构建监控体系：

### 健康检查

设置一个 cron 任务，每五分钟 ping 一次网关，若服务不可用则立即告警：

```bash
#!/bin/bash
# /usr/local/bin/openclaw-healthcheck.sh

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18789/health)

if [ "$STATUS" -ne 200 ]; then
  echo "OpenClaw gateway is down (HTTP $STATUS)" | mail -s "Alert: OpenClaw Down" you@example.com
fi
```

将其加入 crontab：

```bash
*/5 * * * * /usr/local/bin/openclaw-healthcheck.sh
```

若你运行多个服务，建议使用专业监控栈（如 Prometheus、Grafana、Uptime Kuma）。但对于单 Agent 部署，一个 bash 脚本配合邮件通知已足够。

### 日志轮转

网关将日志输出到 stdout，由 pm2 捕获。若不做轮转，`~/.pm2/logs/openclaw-gateway-out.log` 会无限增长。不到三个月，该文件就可能膨胀至 2GB，最终塞满磁盘。

创建 `/etc/logrotate.d/openclaw`：

```text
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

该配置保留最近七天的日志，自动压缩旧日志，并在轮转后通知 pm2 重新打开日志文件。

### 磁盘空间告警

Agent 会将会话日志写入 `~/.openclaw/agents/main/sessions/`。若运行的是热门 Agent，该目录每天增长约 10MB，一年可达 3.6GB。若服务器根分区仅有 20GB，迟早会耗尽空间。

将磁盘检查逻辑加入前述健康检查脚本：

```bash
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$USAGE" -gt 85 ]; then
  echo "Disk usage is at ${USAGE}% on /" | mail -s "Alert: Disk Space Low" you@example.com
fi
```

告警触发时，可选择归档旧会话（`openclaw memory archive --older-than 90d`）或直接扩容磁盘。

## “健康吗？”五连查

```bash
pm2 status openclaw-gateway
openclaw doctor
wc -l ~/.openclaw/workspace/MEMORY.md
ls ~/.openclaw/agents/main/sessions/*.jsonl | wc -l
df -h /
```

## 结语

生产环境中的 OpenClaw 并非追求炫技的软件，而是一款能够连续稳定运行三十天的可靠系统。只要按部就班完成部署，配好 supervisor，前端挂上稳定的代理，提前解决上述八大故障隐患，并实现监控自动化，你就已经成功了一大半。

至此，《OpenClaw 指南》系列告一段落。从此刻起，前路分叉——你可以深入定制技能、对接专属渠道、搭建 MCP 服务器，或构建多 Agent 拓扑。所有这些进阶探索，都建立在本系列十篇文章所奠定的基础之上。
