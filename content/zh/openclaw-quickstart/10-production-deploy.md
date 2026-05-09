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

前九篇我们都在本地跑 OpenClaw。本地开发体验很好，但你迟早要面对一个现实：笔记本合上了，agent 就断了。钉钉群里凌晨三点有人 @bot 问问题，没人应答。

这篇讲的是把 OpenClaw 放到一台真正的服务器上，让它 7x24 跑着。然后——更重要的——是我在生产环境里踩过的八个坑，每一个都至少见过两次。

![生产部署架构概览](/posts/zh/openclaw-quickstart/10-production-deploy/illustration_1.jpg)

## 选服务器

![生产部署架构——从操作系统到监控的完整栈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/openclaw-quickstart/10-production-deploy/fig_deploy.png)

四个选项，按我的推荐排序：

### 阿里云 ECS（推荐）

2核4G 的轻量应用服务器，按量付费大概 100 元/月出头。如果你用 DashScope 作为 LLM 后端（大概率是的），ECS 到 DashScope 的内网延迟在 2-5ms，比任何海外机器都快一个数量级。

选地域时优先上海或杭州——DashScope 的主力节点在这两个区域。北京也行，多 3-5ms 感知不到。

### DigitalOcean

$18/月能拿到 2核4G。优势是面板简单、文档好、社区资源多。劣势是到国内有墙的问题——如果你的钉钉 webhook 走的是公网回调，可能需要额外处理。适合面向海外用户的场景。

### Hetzner

$5.83/月，性价比极高。但节点在欧洲，到 DashScope 延迟 200ms+，到钉钉服务器也要绕一圈。除非你的用户和 LLM 后端都在海外，否则不推荐。

### 家用服务器

免费，但不可靠。断电、断网、路由器重启、动态 IP 变化——任何一个都会让你的 agent 挂掉。我试过用树莓派跑，撑了 11 天后因为小区停电 2 小时彻底放弃。

### 最低配置

**2核4GB 内存是硬性下限。** 原因：OpenClaw 在执行复杂任务时会 fork 子 agent 进程。主进程本身吃 400-600MB，每个子 agent 再加 200-300MB。1GB 内存的机器在第一次多步推理时就会触发 OOM killer，然后你看到的症状是 agent 突然沉默——没有报错，没有日志，进程直接被内核干掉了。

如果你用 Docker 部署，再加 512MB 的余量，即 2核4.5G 以上。

## 部署步骤

以下基于 Ubuntu 22.04 + 阿里云 ECS，其他发行版大同小异。

### 基础环境

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc

# 安装 Node.js 22（必须 >=22.16）
nvm install 22
nvm use 22
nvm alias default 22

# 验证
node -v  # 应该输出 v22.x.x
```

### 安装 OpenClaw

```bash
npm install -g openclaw
openclaw init --config /opt/openclaw/config.yaml
```

把你本地调好的 `config.yaml` 和 `MEMORY.md` 传上来。我一般用 `scp` 或者直接 git clone 配置仓库。

### pm2 进程守护

```bash
npm install -g pm2

# 启动 OpenClaw
pm2 start openclaw -- serve --config /opt/openclaw/config.yaml

# 保存进程列表（重启后自动恢复）
pm2 save

# 设置开机自启
pm2 startup
# 按照输出的命令执行一次（通常是一行 sudo 命令）
```

pm2 会在 OpenClaw 崩溃时自动重启它。默认重启间隔是 1 秒，最多连续重启 16 次后停止。这个默认值很合理，不用改。

### nginx 反代 + SSL

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:18789;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # 关键：LLM 调用可能耗时很长
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

为什么是 600 秒？因为当 agent 调用一个复杂的多步工具链时，从发出请求到拿到最终结果可能需要 3-5 分钟。nginx 默认的 60 秒超时会在中间把连接切断，客户端看到的是一个莫名其妙的 502。600 秒是我测试下来的安全值——如果一个请求真的跑了 10 分钟，那大概率是死循环，让它超时反而是正确行为。

用 acme.sh 做自动续证：

```bash
curl https://get.acme.sh | sh
acme.sh --issue -d your-domain.com --nginx
acme.sh --install-cert -d your-domain.com \
  --key-file /etc/letsencrypt/live/your-domain.com/privkey.pem \
  --fullchain-file /etc/letsencrypt/live/your-domain.com/fullchain.pem \
  --reloadcmd "systemctl reload nginx"
```

acme.sh 会自动注册 cron 任务，每 60 天续一次。不用管它。

## Docker 方案

如果你更喜欢容器化部署：

```yaml
version: "3.8"
services:
  openclaw:
    image: openclaw/openclaw:latest
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - ./config:/app/config        # 配置文件
      - ./workspace:/app/workspace  # 工作目录
      - ./skills:/app/skills        # 自定义 skill
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
      - OPENCLAW_CONFIG=/app/config/config.yaml
```

三个 volume 缺一不可：

- `config` — 配置文件和 MEMORY.md。不挂载的话每次重建容器都要重新配置。
- `workspace` — agent 的工作区。对话历史、生成的文件都在这里。丢了就是丢了。
- `skills` — 自定义 skill 脚本。如果你在第七篇写了自己的 skill，它们住在这里。

### Docker 什么时候值得用

**值得：** 你有多个服务跑在同一台机器上，需要隔离；或者你需要快速回滚到上一个版本（`docker-compose down && docker-compose up -d` 配合 image tag）。

**不值得：** 你的 ECS 只跑 OpenClaw 一个东西。Docker 额外吃 200-400MB 内存，对于 2核4G 的机器这不是小数目。而且 debug 时多了一层容器抽象，排查 permission 问题和网络问题都更麻烦。

我个人在 2核4G 的机器上选裸跑 + pm2。8G 以上再考虑 Docker。

## 八种故障模式

以下每一种我都在生产环境至少遇到过两次。不是假设，是血泪。

### 1. 重启后 `command not found`

**症状：** 服务器重启后 pm2 把 OpenClaw 拉起来了，但日志里全是 `openclaw: command not found`。

**原因：** nvm 的环境变量只在交互式 shell 里加载。pm2 的 startup 脚本跑在非交互式 shell 里，`$PATH` 里没有 nvm 管理的 node 路径。

**修复：**

```bash
# 创建系统级符号链接
sudo ln -sf $(which node) /usr/local/bin/node
sudo ln -sf $(which openclaw) /usr/local/bin/openclaw
sudo ln -sf $(which pm2) /usr/local/bin/pm2
```

做完之后 `pm2 unstartup && pm2 startup && pm2 save` 重新注册一次。

### 2. Node.js 版本过旧

**症状：** `SyntaxError: Unexpected token` 或者 `ERR_MODULE_NOT_FOUND`。

**原因：** OpenClaw 从 v0.9 开始要求 Node.js >= 22.16。22.16 引入了稳定的 `--experimental-strip-types` 支持，OpenClaw 的部分内部模块依赖这个特性。如果你的机器上装的是 Node 20 或者 22.0-22.15，就会炸。

**修复：**

```bash
nvm install 22
# 确认版本号 >= 22.16
node -v
```

### 3. DashScope 返回 401

**症状：** 日志里出现 `401 Unauthorized`，agent 所有请求都失败。

**原因：** 两种可能：
1. API Key 被轮换了。阿里云的 RAM 策略或者组织管理员定期轮换 key，你的 config 里还是旧的。
2. endpoint 写错了。DashScope 有多个 endpoint（`dashscope.aliyuncs.com` vs `dashscope-intl.aliyuncs.com`），用错了 key 对不上 endpoint 就是 401。

**修复：** 去 DashScope 控制台重新拿一个 key，确认 endpoint 匹配。国内账号用 `dashscope.aliyuncs.com`。

### 4. Connection refused（端口冲突）

**症状：** nginx 报 `connect() failed (111: Connection refused)`，但 pm2 显示 OpenClaw 状态是 online。

**原因：** 18789 端口被另一个进程占了。OpenClaw 启动时发现端口被占，静默退出后被 pm2 重启，无限循环。pm2 显示 online 是因为进程确实存在——只是每隔 1 秒就死一次又活一次。

**修复：**

```bash
# 找出谁占了端口
sudo lsof -i :18789

# 干掉它，或者改 OpenClaw 配置用别的端口
kill -9 <PID>
pm2 restart openclaw
```

看 pm2 的 restart count——如果数字在几分钟内飙到两位数，说明就是这个问题。

### 5. 钉钉 30 分钟后沉默

**症状：** 部署后前 30 分钟一切正常，之后 bot 不再响应任何消息。重启 OpenClaw 又好了，30 分钟后又挂。

**原因：** 阿里云的 NAT 网关默认 30 分钟无流量就断开 TCP 连接。钉钉的 Stream 模式走的是长连接，如果 30 分钟内没有消息进来，NAT 把连接拆了，OpenClaw 这边还不知道。

**修复：** 在 config.yaml 里加心跳配置：

```yaml
dingtalk:
  reconnectMs: 25000    # 25秒检测一次连接状态
  heartbeatMs: 20000    # 20秒发一次心跳包
```

20 秒的心跳间隔远小于 NAT 的 30 分钟超时，连接不会被拆。

### 6. Agent 对话到一半失忆

**症状：** 长对话中 agent 突然忘记前面说过的内容，回答变得前言不搭后语。

**原因：** OpenClaw 的上下文压缩机制（compaction）在 token 快要超限时会自动裁剪历史。但如果你没有配置 `memoryFlush`，被裁剪掉的内容就真的消失了——不会写入 MEMORY.md，不会有任何持久化。

**修复：**

```yaml
agent:
  compaction:
    enabled: true
    threshold: 80000
    memoryFlush: true   # 压缩前先把关键信息写入 MEMORY.md
```

`memoryFlush: true` 会在每次压缩前触发一次 memory write，把当前对话的关键上下文持久化。代价是每次压缩多花 2-3 秒和几百 token，但值得。

### 7. Token 消耗异常高

**症状：** DashScope 账单突然飙升，一天消耗的 token 是预期的 3-5 倍。

**原因：** 通常是三个因素叠加：
1. **默认模型太贵。** 如果你没有显式指定 model，OpenClaw 用的是 `qwen-max`。对于 80% 的日常问答，`qwen-plus` 就够了。
2. **MEMORY.md 膨胀。** 每次对话都会把 MEMORY.md 的全文塞进 system prompt。如果这个文件长到了 200+ 行，每次请求都要多花几千 token。
3. **不必要的子 agent 调用。** 有些 skill 配置了 `useSubAgent: true`，每次调用都会启动一个完整的 agent 循环。如果那个 skill 其实只需要跑个脚本，把它改成 `useSubAgent: false`。

**修复：**

```yaml
agent:
  defaultModel: qwen-plus        # 日常用便宜模型
  complexModel: qwen-max         # 只有复杂任务才用贵模型
  complexThreshold: 3            # 超过3步推理才切换
```

同时定期清理 MEMORY.md，删掉过时的内容。

### 8. MEMORY.md 无限增长

**症状：** 跑了几周后 MEMORY.md 涨到 500+ 行，agent 响应变慢，token 消耗持续攀升。

**原因：** OpenClaw 的 memory write 是追加式的。它只会往 MEMORY.md 里加东西，不会主动删东西。时间长了，里面堆满了过期信息、重复条目、早已不再需要的上下文。

**修复：** 设置定期清理的 cron：

```bash
# 每周日凌晨3点执行 memory 压缩
0 3 * * 0 openclaw memory compact --config /opt/openclaw/config.yaml
```

`memory compact` 会用 LLM 重新审视整个 MEMORY.md，删除过时条目，合并重复内容，保留核心信息。一次 compact 通常能把 500 行缩到 80-120 行。

成本大概 5000 token，一周跑一次完全可以接受。

![故障排查流程图](/posts/zh/openclaw-quickstart/10-production-deploy/illustration_2.png)

## 升级路径

OpenClaw 大概每 2-3 周发一个小版本。升级步骤：

```bash
# 1. 看 changelog，确认没有 breaking change
openclaw changelog --from $(openclaw --version)

# 2. 备份当前配置
cp /opt/openclaw/config.yaml /opt/openclaw/config.yaml.bak
cp /opt/openclaw/MEMORY.md /opt/openclaw/MEMORY.md.bak

# 3. 升级
npm install -g openclaw@latest

# 4. 重启
pm2 restart openclaw

# 5. 验证
openclaw health --config /opt/openclaw/config.yaml
```

### 什么会炸

- **配置 schema 变更。** 新版本可能 rename 或者 deprecate 某些字段。`openclaw validate` 会告诉你哪些字段需要改。
- **废弃字段。** 通常给两个版本的缓冲期，之后硬删。不看 changelog 就升级的话，启动直接报错。
- **依赖冲突。** 如果你全局装了其他 npm 包，版本可能打架。建议 OpenClaw 用独立的 nvm 环境。

### 黄金法则

**周日凌晨 2 点升级。** 原因：

1. 这是你的用户最不活跃的时间。
2. 如果炸了，你有整个周日白天来修。
3. 周一早上用户上班时，你要么已经修好了，要么已经回滚了。

回滚方法：

```bash
npm install -g openclaw@<上一个版本号>
cp /opt/openclaw/config.yaml.bak /opt/openclaw/config.yaml
pm2 restart openclaw
```

## 监控

### 健康检查脚本

```bash
#!/bin/bash
# /opt/openclaw/healthcheck.sh

HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18789/health)

if [ "$HEALTH" != "200" ]; then
    echo "OpenClaw health check failed at $(date)" | \
        mail -s "ALERT: OpenClaw Down" your-email@example.com
    # 尝试自动重启
    pm2 restart openclaw
fi
```

cron 每 5 分钟跑一次：

```bash
*/5 * * * * /opt/openclaw/healthcheck.sh >> /var/log/openclaw-health.log 2>&1
```

### 日志轮转

OpenClaw 的日志默认写到 `~/.pm2/logs/`，不设 logrotate 的话磁盘迟早满。

```bash
# /etc/logrotate.d/openclaw
/root/.pm2/logs/openclaw-*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

`copytruncate` 很重要——pm2 不支持 SIGHUP 重新打开日志文件，只能用截断方式轮转。

### 磁盘空间告警

```bash
# 加到 cron，每小时检查一次
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$USAGE" -gt 85 ]; then
    echo "Disk usage at ${USAGE}%" | \
        mail -s "ALERT: Disk Space Low" your-email@example.com
fi
```

## 五行诊断

服务器上出了问题，先跑这五个命令：

```bash
# 1. OpenClaw 进程还在不在
pm2 list

# 2. 最近的错误日志
pm2 logs openclaw --lines 50 --err

# 3. 端口在不在监听
ss -tlnp | grep 18789

# 4. 内存和 CPU
free -h && top -bn1 | head -20

# 5. 磁盘
df -h /
```

90% 的问题在这五个命令里能看出端倪。如果看不出来，那大概率是网络层的问题——去安全组/防火墙规则里找。

## 结语

生产部署不难。难的是让它稳定跑 30 天。

本篇的核心观点只有一个：**无聊的软件才是好软件。** 你不需要 Kubernetes，不需要服务网格，不需要蓝绿部署。一台 ECS、一个 pm2、一个 nginx、一个 cron——四个组件，每一个都已经被检验了十几年，每一个出了问题都能在 5 分钟内定位。

把花哨的架构留给需要花哨架构的场景。对于一个 AI agent 来说，最重要的可靠性来自于你能在半夜被叫醒后 5 分钟内理解整个系统的状态。如果你的部署方案需要你完全清醒才能 debug，那它就不是一个好的部署方案。

下一篇我们讲多 agent 协作——当一个 agent 不够用的时候怎么办。

---

*本文是 OpenClaw 快速上手系列的第十篇。完整系列目录见[第一篇](/posts/zh/openclaw-quickstart-01)。*
