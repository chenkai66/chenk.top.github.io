---
title: "阿里云全栈实战（二）：ECS——让计算回归本质"
date: 2026-05-10 09:00:00
tags:
  - Alibaba Cloud
  - ECS
  - Compute
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 2
description: "Everything you need to know about ECS: instance families (g8, c8, r8, GPU), pricing models, cloud-init automation, security groups, and key pairs. We deploy a production-ready app server from scratch."
disableNunjucks: true
translationKey: "aliyun-fullstack-2"
---
记得我第一次启动 ECS 实例时，配置选得离谱地高。挑了个能找到的最大规格——`ecs.r6.8xlarge`，32 vCPU 配 256 GiB 内存，就为了跑个每分钟大概 20 个请求的 Flask 应用。结果一周内把额度烧光了，慌得不行，赶紧学会怎么在线降配，最后发现换个 2 vCPU 的机器跑得一样好，成本还低了 94%。选对规格比单纯堆算力重要得多，搞懂计算层也是你掌握任何云平台最有价值的一步。

这篇文章就是 ECS 的完全指南。我们从 ECS 到底是什么讲起，穿过实例家族和定价模型，最后用 CLI 从零搭建一个生产级的应用服务器。读完这篇，你就有足够的实战知识去为真实业务配置、加固和运维 ECS 实例了。

![ECS Compute](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/02-ecs-compute/cover.png)

## ECS 到底是什么

弹性计算服务（ECS）就是阿里云的虚拟机产品。如果你用过 AWS EC2、Azure VMs 或者 GCP Compute Engine，ECS 就是它们的直接对标物。你拿到的是一个跑着 Linux 或 Windows 的虚拟服务器，连着虚拟网络，挂着块存储。通过 SSH 或 RDP 控制，按小时或按月付费。

但 ECS 不仅仅是“一台 VM"。它由六个构建块组成，分开理解每一块能省去很多麻烦：

| 组件 | 说明 | AWS 对标 |
|---|---|---|
| **Instance** | 虚拟机本身 — vCPU、RAM、本地 NVMe | EC2 instance |
| **Image** | 用于启动实例的 OS 模板 | AMI |
| **Block Storage (disk)** | 网络附加持久存储 — 系统盘 + 数据盘 | EBS |
| **Security Group** | 绑定在实例网卡上的有状态防火墙规则 | Security Group |
| **VPC / VSwitch** | 实例所在的虚拟网络和子网 | VPC / Subnet |
| **ENI** | 弹性网卡 — 虚拟 NIC | ENI |

在控制台“创建 ECS 实例”时，其实是在一次性配置这六个组件。控制台把它们打包是为了减少摩擦，但它们是拥有独立生命周期的独立资源。你可以把磁盘从一台实例卸下来挂到另一台。ENI 可以在实例间迁移。安全组是跨实例共享的。搞懂这种分解，才是区分“使用 ECS 的人”和“运维 ECS 的人”的关键。

### ECS 生命周期

每个实例都会经过一个定义明确的状态机：

```
                  ┌──────────┐
         create   │          │  start
    ────────────► │ Stopped  │ ──────────┐
                  │          │           │
                  └────┬─────┘           ▼
                       │          ┌──────────┐
                 delete│          │          │
                       │          │ Starting │
                       ▼          │          │
                  ┌──────────┐    └────┬─────┘
                  │          │         │
                  │ Released │         ▼
                  │          │    ┌──────────┐
                  └──────────┘    │          │  stop
                                  │ Running  │ ──────┐
                                  │          │       │
                                  └────┬─────┘       ▼
                                       │       ┌──────────┐
                                       │       │          │
                                       └───────│ Stopping │
                                               │          │
                                               └──────────┘
```

实践中几个关键状态：

- **Stopped**: 按量付费实例不收计算费（磁盘和 IP 继续计费）。这是开发实例过夜该有的状态。
- **Running**: 实例运行中，正在计费。CPU/内存/网络计量生效。
- **Stopping**: 短暂过渡。`ForceStop` 最多耗时 60 秒。优雅停止会向 OS 发送 ACPI shutdown 信号并等待。
- **Released**: 没了。实例、系统盘和本地盘被永久删除。数据盘只有配置了 `DeleteWithInstance = false` 才能保留。

有个地方容易踩坑：Stopped 状态的实例依然占用私网 IP 和弹性 IP 关联。停止实例并不会释放网络资源。如果想彻底释放一切，得释放实例。

### ECS vs EC2：差异在哪

如果你从 AWS 过来，以下是值得注意的差异：

1. **No instance store volumes**：没有 EC2 意义上的实例存储卷。ECS 本地 NVMe 磁盘只存在于特定实例家族（i-series），大多数工作负载 exclusively 使用云盘。
2. **Security groups are VPC-scoped**：安全组是 VPC 范围的，不是 Region 范围。不能跨 VPC 共享安全组。
3. **Metadata endpoint**：是 `http://100.100.100.200/latest/meta-data/` 而不是 `169.254.169.254`。Cloud-init 工作方式一样，但如果你要移植脚本，得更新 URL。
4. **Chinese regions**：中国区域有独立的基础设施。`cn-hangzhou` 和 `us-east-1` 不只是可用区不同——它们是完全独立的控制平面，账号和账单都分开。

## 实例家族深挖

实例家族是硬件专业化的核心抽象。命名 convention 是：

```
ecs.{family}{generation}.{size}

Examples:
  ecs.g8i.large      →  General Purpose, gen 8 (Intel), large
  ecs.c8y.xlarge     →  Compute Optimized, gen 8 (Yitian ARM), xlarge
  ecs.r7.2xlarge     →  Memory Optimized, gen 7, 2xlarge
  ecs.gn7i.xlarge    →  GPU (NVIDIA), gen 7 (Intel), xlarge
```

代际后面的字母后缀表示处理器：

- 无后缀或 `i` = Intel Xeon
- `a` = AMD EPYC
- `y` = Alibaba Yitian 710 (ARM)

以下是你实际会用到的实例家族参考：

| 家族 | 类型 | vCPU : 内存 | 处理器 | 网络 (Gbps) | 最佳场景 |
|---|---|---|---|---|---|
| **g7** | 通用型 | 1:4 | Intel Xeon (Ice Lake) | 最高 25 | Web 服务器，中层 API |
| **g8i** | 通用型 | 1:4 | Intel Xeon (Sapphire Rapids) | 最高 40 | 通用工作负载，最新代际 |
| **g8y** | 通用型 | 1:4 | Yitian 710 (ARM) | 最高 40 | 成本敏感的 ARM 工作负载 |
| **c7** | 计算优化型 | 1:2 | Intel Xeon (Ice Lake) | 最高 25 | 高 CPU：编码，CI/CD |
| **c8i** | 计算优化型 | 1:2 | Intel Xeon (Sapphire Rapids) | 最高 40 | 批处理，游戏服务器 |
| **c8y** | 计算优化型 | 1:2 | Yitian 710 (ARM) | 最高 40 | ARM CI/CD，构建农场 |
| **r7** | 内存优化型 | 1:8 | Intel Xeon (Ice Lake) | 最高 25 | 数据库，内存缓存 |
| **r8i** | 内存优化型 | 1:8 | Intel Xeon (Sapphire Rapids) | 最高 40 | Redis，Elasticsearch |
| **r8y** | 内存优化型 | 1:8 | Yitian 710 (ARM) | 最高 40 | 成本敏感的内存工作负载 |
| **gn7i** | GPU | 可变 | A10 (24 GB) | 最高 32 | ML 推理，微调 |
| **gn7** | GPU | 可变 | V100 (16/32 GB) | 最高 25 | ML 训练，HPC |
| **gn6v** | GPU | 可变 | V100 (16 GB) | 最高 5 | 预算 ML，开发推理 |
| **t6** | 突发型 | 1:1/1:2/1:4 | Intel Xeon | 最高 1.2 | 开发/测试，微工作负载 |
| **ebmg7** | 裸金属 | 1:4 | Intel Xeon (Ice Lake) | 最高 65 | 高性能，无 Hypervisor |
| **ebmc7** | 裸金属 | 1:2 | Intel Xeon (Ice Lake) | 最高 65 | 专用硬件，合规 |

### 家族内的规格

每个家族提供的规格每一步资源翻倍：

| 规格 | vCPU | 内存 (g-family, 1:4 比例) |
|---|---|---|
| small | 1 | 2 GiB |
| large | 2 | 8 GiB |
| xlarge | 4 | 16 GiB |
| 2xlarge | 8 | 32 GiB |
| 4xlarge | 16 | 64 GiB |
| 8xlarge | 32 | 128 GiB |
| 16xlarge | 64 | 256 GiB |

> **关于 Yitian 710 (ARM) 的实战建议：** `*y` 家族同规格下比 Intel equivalents 便宜 20-30%。如果你的工作负载跑在 Linux 上且不依赖 x86 特定二进制文件，优先尝试 ARM variant。大多数 Python/Node/Go/Java 工作负载直接能跑。Docker 镜像需要是多架构的（`linux/arm64`），这在 Dockerfile build 里只是一行改动。

### 突发实例：陷阱与修复

`t6` 家族值得单独提一下，因为它几乎坑遍了所有人。突发性能实例在空闲时积累 CPU 积分，忙碌时消耗积分。一旦积分耗尽，CPU 就会被限制到基线——通常是 vCPU 的 10-20%。

这对大部分时间空闲、偶尔跑构建的开发/测试机器很完美。但对于任何有持续 CPU 使用率的工作负载来说都很糟糕。我亲眼见过生产数据库在 `t6.large` 实例上跑了几周没问题，然后在流量高峰期间突然崩盘，因为积分余额归零了。

规则很简单：如果你的平均 CPU 超过基线（具体规格去产品页查），你就需要非突发实例。没得商量。

## 选择合适的规格

这里的决策比看起来简单。从工作负载出发，而不是从实例出发：

| 工作负载 | 推荐起步 | 原因 |
|---|---|---|
| 静态站点 / 反向代理 | `ecs.c7.large` (2 vCPU, 4 GiB) | CPU 密集，几乎不需要内存 |
| REST API 后端 (Node/Python) | `ecs.g7.xlarge` (4 vCPU, 16 GiB) | 平衡 — 一些 CPU 用于 JSON 序列化，内存用于连接池 |
| PostgreSQL / MySQL | `ecs.r7.2xlarge` (8 vCPU, 64 GiB) | 内存密集用于 buffer pool。考虑直接用 RDS。 |
| Redis / Memcached | `ecs.r7.xlarge` (4 vCPU, 32 GiB) | 全是内存。同样，考虑托管 Redis。 |
| CI/CD  runner | `ecs.c8y.xlarge` (4 vCPU, 8 GiB) | CPU 密集编译。ARM 对大多数构建没问题。 |
| ML 推理 (LLM) | `ecs.gn7i.xlarge` (4 vCPU, 1x A10) | GPU 用于矩阵运算，适度 CPU 用于前/后处理 |
| ML 训练 | `ecs.gn7.8xlarge` (32 vCPU, 8x V100) | 多 GPU 用于分布式训练 |
| 开发/测试一次性 | `ecs.t6.large` (2 vCPU, 4 GiB) | 便宜，突发，晚上停止它 |

黄金法则：**从小开始，监控一周，然后调整规格。** ECS 支持大多数家族的在线实例类型变更——停止实例，更改类型，再启动。整个过程不到两分钟。从一开始就过度配置就是在烧钱赌运气。
## 计费模式详解

ECS 计费方式主要有四种，选对方案账单能省 80%。按价格从高到低排个序：

### 按量付费 (PAYG)

按秒计费，最小粒度一分钟。停止实例后，计算费用停止（磁盘和 IP 继续计费）。这是默认选项，单位小时价格最贵，但零承诺、零浪费——用多少付多少。

适用场景：开发测试、波动型负载、每天只跑几小时的实例。

### 包年包月 (Subscription)

承诺使用 1 个月、3 个月、6 个月、1 年、2 年或 3 年。相比 PAYG，折扣范围从 ~15%（1 个月）到 ~50%（3 年）不等。需要预付。不管你用不用，实例都在跑。

适用场景：利用率稳定可预测的生产环境负载。

### 抢占式实例 (Spot)

硬件和 PAYG 一样，但价格随供需浮动——通常便宜 70-90%。缺点是：需求高峰时，阿里云会提前 5 分钟通知回收实例。你会被打断。

适用场景：无状态批处理、CI/CD、分布式 ML 训练（带 checkpoint 机制）、任何能容忍中断的负载。

### 节省计划与预留实例

节省计划允许你承诺一个消费金额（比如 100 CNY/小时），适用于区域内任意实例族，折扣 30-60%。预留实例类似，但锁定特定实例类型和可用区。

适用场景：规模较大、基线用量明确的实例集群。

### 价格对比

下面是 `ecs.c7.large` (2 vCPU, 4 GiB) 在 `cn-beijing` 区域，24/7 运行一个月的真实成本对比（价格近似，请以当前费率为准）：

| Model | Hourly (CNY) | Monthly (CNY) | Savings vs PAYG |
|---|---|---|---|
| Pay-as-you-go | 0.68 | ~490 | — |
| Subscription (1 month) | — | ~415 | 15% |
| Subscription (1 year) | — | ~310 | 37% |
| Subscription (3 years) | — | ~245 | 50% |
| Preemptible (avg) | 0.10 | ~72 | 85% |
| Savings Plan (1 year) | — | ~295 | 40% |

抢占式实例的价格没写错。如果你的负载能容忍中断，Spot 实例几乎等于免费。我把所有 CI/CD 都跑在 Spot 上，一年大概只被中断过三次——都是在中国 major 购物节需求峰值期间。

> **我自己实际采用的混合策略：** 生产环境跑包年包月（1 年期）。开发测试跑 PAYG，配合午夜自动停止脚本。批处理任务跑抢占式实例，如果 Spot 容量不足则 fallback 到 PAYG。相比全用 PAYG，这套组合拳能把总账单砍掉约 45%。

## 手把手创建 ECS 实例

### 前置准备

创建实例前，你需要准备好：

1. **VPC 和 VSwitch**，位于目标区域。我们在 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 详细讲过 VPC  setup，但为了本教程，假设你已经在 `cn-beijing-h` 有了一个带 VSwitch 的 VPC。
2. **安全组**，位于该 VPC 内（下面会创建）。
3. **密钥对**，用于 SSH 访问（下面会创建）。
4. **Alibaba Cloud CLI (`aliyun`)**，已安装并配置。如果还没配，运行 `aliyun configure`。

### 控制台操作（快速版）

控制台路径：**ECS 控制台** > **实例** > **创建实例**。

1. **计费方式**：暂时选按量付费。
2. **地域**：`China (Beijing)`，可用区 H。
3. **实例规格**：搜索 `ecs.c7.large`。选中它。
4. **镜像**：Alibaba Cloud Linux 3.2104 LTS 64-bit（默认）。兼容 CentOS 且免费。
5. **存储**：系统盘 = 40 GiB ESSD PL0。暂时不加数据盘。
6. **网络**：选中你的 VPC，可用区 H 的 VSwitch。分配公网 IP（SSH 用 1 Mbps 够了；生产流量走 SLB/ALB）。
7. **安全组**：选中或创建一个允许你的 IP 访问 TCP 22 端口的组。
8. **登录方式**：密钥对（别用密码）。
9. **高级选项**：在 User Data 里粘贴 cloud-init 脚本（下面会写）。
10. **创建**。

大概七次点击加三个下拉菜单。但如果你要创建多个实例，或者想要可复现性，请用 CLI。

### CLI 操作（正经玩法）

先创建支撑资源。如果你已经有了 VPC 和安全组，直接跳到实例创建步骤。

**创建安全组：**

```bash
# Create security group
aliyun ecs CreateSecurityGroup \
  --RegionId cn-beijing \
  --VpcId vpc-2ze1234567890abcdef \
  --SecurityGroupName "app-server-sg" \
  --SecurityGroupType normal \
  --Description "Security group for app servers"
```

把返回结果里的 `SecurityGroupId` 存好——后面要用。

**创建密钥对：**

```bash
# Create key pair — the private key is returned ONLY ONCE
aliyun ecs CreateKeyPair \
  --RegionId cn-beijing \
  --KeyPairName "app-server-key" | tee keypair-response.json

# Extract and save the private key
cat keypair-response.json | jq -r '.PrivateKeyBody' > ~/.ssh/app-server-key.pem
chmod 600 ~/.ssh/app-server-key.pem
rm keypair-response.json
```

**创建 ECS 实例：**

```bash
aliyun ecs CreateInstance \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --InstanceType ecs.c7.large \
  --ImageId aliyun_3_x64_20G_alibase_20240528.vhd \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --VSwitchId vsw-2ze1234567890abcdef \
  --InstanceName "app-server-01" \
  --HostName "app-server-01" \
  --KeyPairName "app-server-key" \
  --SystemDisk.Category cloud_essd \
  --SystemDisk.Size 40 \
  --SystemDisk.PerformanceLevel PL0 \
  --InternetMaxBandwidthOut 5 \
  --InternetChargeType PayByTraffic \
  --InstanceChargeType PostPaid \
  --SpotStrategy NoSpot \
  --UserData "$(base64 -w0 cloud-init.yaml)" \
  --Description "Production app server" \
  --Tag.1.Key Environment \
  --Tag.1.Value production \
  --Tag.2.Key App \
  --Tag.2.Value web-server
```

注意：`--ImageId` 会随着新镜像发布而变化。查找最新的 Alibaba Cloud Linux 3 镜像：

```bash
aliyun ecs DescribeImages \
  --RegionId cn-beijing \
  --OSType linux \
  --ImageOwnerAlias system \
  --ImageName "Alibaba Cloud Linux 3*" \
  --PageSize 1 \
  --SortKey CreatedOn \
  | jq '.Images.Image[0] | {ImageId, ImageName, CreationTime}'
```

创建完成后，实例处于 `Stopped` 状态。启动它：

```bash
aliyun ecs StartInstance --InstanceId i-2ze1234567890abcdef
```

等大概 30 秒，然后 SSH 登录：

```bash
ssh -i ~/.ssh/app-server-key.pem root@<public-ip>
```

## Cloud-init：开机即自动化

千万别 SSH 进新实例手动装包。Cloud-init 在首次启动时运行，自动配置实例。每个 ECS 镜像都预装了 cloud-init。

创建实例时，你把 cloud-init 配置作为 `UserData` 传进去。必须 base64 编码。下面是一个全面的 `cloud-init.yaml`，能 setup 一个生产就绪的应用服务器：

```yaml
#cloud-config

# --- System ---
timezone: Asia/Shanghai
locale: en_US.UTF-8

# --- Package management ---
package_update: true
package_upgrade: true
packages:
  - nginx
  - python3
  - python3-pip
  - python3-venv
  - git
  - curl
  - htop
  - jq
  - supervisor
  - certbot
  - python3-certbot-nginx
  - fail2ban

# --- Users ---
users:
  - name: deploy
    groups: [sudo, www-data]
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ... deploy@workstation

# --- SSH hardening ---
ssh_pwauth: false
disable_root: false

write_files:
  - path: /etc/ssh/sshd_config.d/hardened.conf
    content: |
      PermitRootLogin prohibit-password
      PasswordAuthentication no
      ChallengeResponseAuthentication no
      MaxAuthTries 3
      ClientAliveInterval 300
      ClientAliveCountMax 2

  - path: /opt/app/deploy.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      set -euxo pipefail
      cd /opt/app
      python3 -m venv venv
      source venv/bin/activate
      pip install --upgrade pip
      pip install flask gunicorn
      
      cat > app.py << 'PYEOF'
      from flask import Flask, jsonify
      import os, socket
      
      app = Flask(__name__)
      
      @app.route("/health")
      def health():
          return jsonify(status="ok", host=socket.gethostname())
      
      @app.route("/")
      def index():
          return jsonify(message="Hello from ECS", instance=os.uname().nodename)
      PYEOF
      
      cat > /etc/supervisor/conf.d/app.conf << 'SUPEOF'
      [program:app]
      command=/opt/app/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
      directory=/opt/app
      user=deploy
      autostart=true
      autorestart=true
      stderr_logfile=/var/log/app/error.log
      stdout_logfile=/var/log/app/access.log
      SUPEOF
      
      mkdir -p /var/log/app
      chown deploy:deploy /var/log/app
      supervisorctl reread
      supervisorctl update

  - path: /etc/nginx/sites-available/app
    content: |
      server {
          listen 80;
          server_name _;
          
          location / {
              proxy_pass http://127.0.0.1:8000;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header X-Forwarded-Proto $scheme;
          }
          
          location /health {
              proxy_pass http://127.0.0.1:8000/health;
              access_log off;
          }
      }

# --- Run commands on first boot ---
runcmd:
  # Enable and start fail2ban
  - systemctl enable fail2ban
  - systemctl start fail2ban

  # Configure nginx
  - ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app
  - rm -f /etc/nginx/sites-enabled/default
  - nginx -t && systemctl reload nginx

  # Deploy the application
  - /opt/app/deploy.sh

  # Restart SSH with hardened config
  - systemctl restart sshd

  # Log completion
  - echo "cloud-init complete at $(date)" >> /var/log/cloud-init-complete.log
```

实例启动后，cloud-init 分阶段处理这个文件：设时区、装包、建用户、写文件、跑命令。在 `ecs.c7.large` 上全过程大概 2-3 分钟。

验证 cloud-init 是否成功完成：

```bash
# Check cloud-init status
cloud-init status --long

# If something failed, check the log
tail -100 /var/log/cloud-init-output.log

# Verify the app is running
curl http://localhost/health
```

> **调试技巧：** 最常见的 cloud-init 失败原因是 YAML 缩进错误。Base64 编码前，先用 `cloud-init schema --config-file cloud-init.yaml` 在本地验证配置。另外，`UserData` 有 16 KiB 限制——如果脚本复杂，让 cloud-init 从 OSS 拉取脚本 instead。
## 安全组：你的第一道防火墙

安全组就是挂在 ENI 层面的状态防火墙。进出 ECS 实例的每个数据包都要过一遍安全组规则。要是没匹配上任何规则，包就直接**丢弃**——默认拒绝。

"状态化"是什么意思？比如你放行了入站 TCP 80，响应包会自动允许出站。你不需要为返回流量专门配一条出站规则。

### 常见安全组模式

**Web 服务器（对外）：**

```bash
# Allow HTTP from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow HTTP"

# Allow HTTPS from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow HTTPS"

# Allow SSH from your office IP only
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 22/22 \
  --SourceCidrIp 203.0.113.50/32 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow SSH from office"
```

**数据库服务器（仅内网）：**

```bash
# Allow PostgreSQL from app server security group
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2zeDBSERVERSG \
  --IpProtocol tcp \
  --PortRange 5432/5432 \
  --SourceGroupId sg-2zeAPPSERVERSG \
  --Policy accept \
  --Priority 1 \
  --Description "Allow PostgreSQL from app servers"
```

注意最后一个例子用的是 `--SourceGroupId` 而不是 `--SourceCidrIp`。这意味着“允许来自该安全组内任何实例的流量”。这才是内部服务通信的正确姿势——别硬编码 IP。

### 我给每个安全组定的规矩

1. **SSH (22) 别对 `0.0.0.0/0` 开。** 用办公室 CIDR，或者更好，用堡垒机。
2. **数据库端口别暴露公网。** PostgreSQL 的 5432，MySQL 的 3306，Redis 的 6379——这些只能让应用安全组访问。
3. **每条规则写描述。** 半年后你肯定忘了 8443 是干嘛的。描述会告诉你。
4. **每季度审查。** 安全组规则会像藤壶一样堆积。设个日历提醒。

## 密钥对与 SSH 访问

密码登录 SSH 是大忌。容易被爆破，诱导密码复用，轮换还得登录实例。密钥对才是标准做法。

### 创建和使用密钥对

前面我们创建过密钥对。下面是完整的 SSH 工作流：

```bash
# Your SSH config (~/.ssh/config)
Host app-server
    HostName 47.94.xxx.xxx
    User root
    IdentityFile ~/.ssh/app-server-key.pem
    StrictHostKeyChecking accept-new

Host bastion
    HostName 47.94.yyy.yyy
    User deploy
    IdentityFile ~/.ssh/bastion-key.pem
    StrictHostKeyChecking accept-new

Host app-internal
    HostName 172.16.1.10
    User deploy
    IdentityFile ~/.ssh/app-server-key.pem
    ProxyJump bastion
```

有了这个配置：

```bash
# Direct access to public instance
ssh app-server

# Access internal instance via bastion (two-hop)
ssh app-internal

# Copy files through bastion
scp -o ProxyJump=bastion localfile.tar.gz app-internal:/opt/app/
```

`ProxyJump` 指令是穿越堡垒机 SSH 隧道的现代替代方案。它通过 `bastion` 透明建立 SSH 连接——你不需要先 SSH 到堡垒机，然后再 SSH 到内部主机。一条命令，两跳，内部 IP 不暴露。

### 密钥轮换

要想不停机轮换密钥：

1. 在 ECS 控制台或 CLI 生成新密钥对。
2. 把新公钥加到目标实例的 `~/.ssh/authorized_keys`（或者用 cloud-init 通过 API 管理密钥）。
3. 测试新密钥登录。
4. 从 `authorized_keys` 移除旧公钥。
5. 从 ECS 删除旧密钥对。

对于大批量机器，通过 cloud-init 或 Ansible 实战手册 管理 SSH 密钥——别手动去改 20 台实例的 `authorized_keys`。

## 磁盘与存储

每个 ECS 实例至少有一块磁盘：系统盘，装着 OS。你可以挂载最多 16 块额外数据盘。所有磁盘都是网络附加块存储——配置得当的话，它们独立于实例生命周期持久存在。

### 磁盘类型

| Type | IOPS (max) | Throughput (max) | Latency | Best for |
|---|---|---|---|---|
| **ESSD PL0** | 10,000 | 180 MB/s | 0.2-0.5 ms | System disks, light workloads |
| **ESSD PL1** | 50,000 | 350 MB/s | 0.1-0.3 ms | General production, databases |
| **ESSD PL2** | 100,000 | 750 MB/s | 0.1-0.3 ms | High-IOPS databases |
| **ESSD PL3** | 1,000,000 | 4,000 MB/s | 0.1-0.3 ms | Extreme performance, OLTP |
| **Standard SSD** | 25,000 | 300 MB/s | 0.5-2 ms | Legacy, non-critical |
| **Ultra Disk** | 5,000 | 140 MB/s | 1-3 ms | Cold storage, archives |

性能等级（PL0 到 PL3）是你做的最重要的存储决策。数据库跑在 PL0 上会撞 10,000 IOPS 天花板，操作排队；同样的数据库在 PL1 上就有 5 倍余量。PL0 和 PL1 的差价大概 2 倍——比起计算成本还是很便宜的。

> **我的默认选择：** 系统盘用 PL0（OS 不需要高 IOPS），跑数据库或任何有 fsync 的数据盘用 PL1。拿不准就先上 PL1——100 GiB 磁盘差价也就 40 元/月。

### 在线扩容磁盘

ECS 支持在线扩容磁盘——不用停机也不用卸载文件系统就能扩容：

```bash
# Step 1: Expand the disk via API (does not touch the filesystem)
aliyun ecs ResizeDisk \
  --DiskId d-2ze1234567890abcdef \
  --NewSize 100 \
  --Type online

# Step 2: SSH into the instance and grow the filesystem
# For ext4:
sudo growpart /dev/vda 1
sudo resize2fs /dev/vda1

# For xfs:
sudo growpart /dev/vda 1
sudo xfs_growfs /
```

只能扩，不能缩。初始大小保守点，按需增长——这是云，不是物理服务器，换盘不用申请维护窗口。

### 快照

快照是磁盘的时间点拷贝。它们是增量的（只存变化的块）且崩溃一致性。用途如下：

1. **高风险操作前备份。** 跑数据库迁移前执行 `aliyun ecs CreateSnapshot --DiskId d-xxx`。
2. **制作镜像。** 快照配置好的实例系统盘，然后基于它创建自定义镜像。新实例启动就是全配置状态，30 秒搞定，不用等 cloud-init 跑 3 分钟。
3. **灾难恢复。** 设自动快照策略——每天快照保留 7 天是个合理的起点。

```bash
# Create an on-demand snapshot
aliyun ecs CreateSnapshot \
  --DiskId d-2ze1234567890abcdef \
  --SnapshotName "pre-migration-backup" \
  --Description "Before v2.0 database migration" \
  --RetentionDays 30

# Create an automatic snapshot policy (daily at 02:00, keep 7 days)
aliyun ecs CreateAutoSnapshotPolicy \
  --RegionId cn-beijing \
  --autoSnapshotPolicyName "daily-7d" \
  --timePoints '["2"]' \
  --repeatWeekdays '["1","2","3","4","5","6","7"]' \
  --retentionDays 7

# Apply the policy to a disk
aliyun ecs ApplyAutoSnapshotPolicy \
  --autoSnapshotPolicyId sp-2ze1234567890abcdef \
  --diskIds '["d-2ze1234567890abcdef"]'
```

## 监控与维护

### 云监控指标

每个 ECS 实例自动上报指标给云监控。你要盯着这几个：

- **CPUUtilization**: 持续 >80% 意味着你需要升配或扩容。
- **MemoryUsedUtilization**: 持续 >85% 意味着快接近 OOM 了。注意：这个指标需要云监控插件（Alibaba Cloud Linux 默认安装）。
- **DiskReadIOPS / DiskWriteIOPS**: 对照磁盘的 PL 上限。如果持续跑到上限的 80%，升级 PL 或加盘。
- **IntranetInRate / IntranetOutRate**: 网络吞吐。如果撞到实例族上限，得换 bigger instance。
- **vm.TcpConnectionCount**: 连接数。 useful for detecting connection leaks.

### 设置告警

```bash
# Alert when CPU > 90% for 5 minutes
aliyun cms PutResourceMetricRule \
  --RuleId "ecs-cpu-high" \
  --RuleName "ECS CPU High" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "CPUUtilization" \
  --Resources '[{"instanceId":"i-2ze1234567890abcdef"}]' \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 90 \
  --Escalations.Critical.Times 3 \
  --Period 60
```

### 计划维护与热迁移

阿里云定期维护物理基础设施。当你的实例宿主机需要维护时，你会收到通知（邮件、短信或控制台告警），会给一个计划窗口——通常提前 2-4 周。

对于大多数实例族，阿里云执行**热迁移**：实例被移到另一台物理主机，几乎零停机（最后内存拷贝期间短暂暂停 10-100ms）。你不需要做任何事。

对于不能热迁移的实例（裸金属、带 VGPU 的 GPU 实例），你会得到一个维护窗口，需要自己重启实例。设个流程检查 pending 维护事件：

```bash
# Check for pending system events
aliyun ecs DescribeInstancesFullStatus \
  --RegionId cn-beijing \
  --InstanceId.1 i-2ze1234567890abcdef \
  | jq '.InstanceFullStatusSet.InstanceFullStatusType[] | .ScheduledSystemEventSet'
```
## 解决方案：生产级应用服务器

咱们把之前的内容串起来。下面这套流程，能让你从零开始，只用 CLI 就在 ECS 上跑起一个安全、可监控的 Flask 应用。

### Step 1: Create the VPC and VSwitch

```bash
# Create VPC
VPC_ID=$(aliyun vpc CreateVpc \
  --RegionId cn-beijing \
  --CidrBlock 172.16.0.0/12 \
  --VpcName "prod-vpc" \
  | jq -r '.VpcId')

echo "VPC: $VPC_ID"

# Wait for VPC to be available
sleep 5

# Create VSwitch in zone H
VSWITCH_ID=$(aliyun vpc CreateVSwitch \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --VpcId "$VPC_ID" \
  --CidrBlock 172.16.1.0/24 \
  --VSwitchName "prod-app-subnet" \
  | jq -r '.VSwitchId')

echo "VSwitch: $VSWITCH_ID"
```

### Step 2: Create security group with rules

```bash
# Create security group
SG_ID=$(aliyun ecs CreateSecurityGroup \
  --RegionId cn-beijing \
  --VpcId "$VPC_ID" \
  --SecurityGroupName "prod-app-sg" \
  | jq -r '.SecurityGroupId')

echo "Security Group: $SG_ID"

# Allow HTTP
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0 --Policy accept --Priority 1

# Allow HTTPS
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0 --Policy accept --Priority 1

# Allow SSH from specific IP (replace with yours)
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 22/22 \
  --SourceCidrIp "$(curl -s ifconfig.me)/32" --Policy accept --Priority 1
```

### Step 3: Create key pair

```bash
aliyun ecs CreateKeyPair \
  --RegionId cn-beijing \
  --KeyPairName "prod-app-key" \
  | jq -r '.PrivateKeyBody' > ~/.ssh/prod-app-key.pem

chmod 600 ~/.ssh/prod-app-key.pem
```

### Step 4: Find the latest image

```bash
IMAGE_ID=$(aliyun ecs DescribeImages \
  --RegionId cn-beijing \
  --OSType linux \
  --ImageOwnerAlias system \
  --ImageName "Alibaba Cloud Linux 3*" \
  --PageSize 1 \
  | jq -r '.Images.Image[0].ImageId')

echo "Image: $IMAGE_ID"
```

### Step 5: Create and start the instance

```bash
# Create instance with cloud-init user data
INSTANCE_ID=$(aliyun ecs CreateInstance \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --InstanceType ecs.c7.large \
  --ImageId "$IMAGE_ID" \
  --SecurityGroupId "$SG_ID" \
  --VSwitchId "$VSWITCH_ID" \
  --InstanceName "prod-app-01" \
  --HostName "prod-app-01" \
  --KeyPairName "prod-app-key" \
  --SystemDisk.Category cloud_essd \
  --SystemDisk.Size 40 \
  --SystemDisk.PerformanceLevel PL0 \
  --InternetMaxBandwidthOut 10 \
  --InternetChargeType PayByTraffic \
  --InstanceChargeType PostPaid \
  --UserData "$(base64 -w0 cloud-init.yaml)" \
  | jq -r '.InstanceId')

echo "Instance: $INSTANCE_ID"

# Start the instance
aliyun ecs StartInstance --InstanceId "$INSTANCE_ID"

# Wait for it to be running
echo "Waiting for instance to start..."
sleep 30

# Get the public IP
PUBLIC_IP=$(aliyun ecs DescribeInstances \
  --RegionId cn-beijing \
  --InstanceIds "[\"$INSTANCE_ID\"]" \
  | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

echo "Public IP: $PUBLIC_IP"
```

### Step 6: Verify the deployment

```bash
# Wait for cloud-init to finish (usually 2-3 min)
sleep 120

# Test the health endpoint
curl -s "http://$PUBLIC_IP/health" | jq .
# Expected: {"host":"prod-app-01","status":"ok"}

# SSH in and check logs
ssh -i ~/.ssh/prod-app-key.pem root@"$PUBLIC_IP" \
  "cloud-init status --long && supervisorctl status"
```

### Step 7: Set up HTTPS with certbot

After pointing your domain's DNS A record to the public IP:

```bash
ssh -i ~/.ssh/prod-app-key.pem root@"$PUBLIC_IP" << 'EOF'
  certbot --nginx -d app.example.com \
    --non-interactive --agree-tos \
    --email admin@example.com

  # Verify auto-renewal
  certbot renew --dry-run

  # Set up cron for renewal
  echo "0 3 * * * certbot renew --quiet" | crontab -
EOF
```

### Step 8: Set up monitoring

```bash
# Create alert for CPU
aliyun cms PutResourceMetricRule \
  --RuleId "prod-app-01-cpu" \
  --RuleName "Prod App CPU High" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "CPUUtilization" \
  --Resources "[{\"instanceId\":\"$INSTANCE_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 90 \
  --Escalations.Critical.Times 3 \
  --Period 60

# Create alert for disk usage
aliyun cms PutResourceMetricRule \
  --RuleId "prod-app-01-disk" \
  --RuleName "Prod App Disk Full" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "diskusage_utilization" \
  --Resources "[{\"instanceId\":\"$INSTANCE_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 85 \
  --Escalations.Critical.Times 1 \
  --Period 300
```

这就搞定了一套完整的生产部署：VPC、安全组、密钥对、带 cloud-init 自动化的 ECS 实例、nginx 反向代理、supervisor 托管的 Flask 应用、自动续期的 HTTPS，还有监控告警。从零到生产环境，API 调用大概 5 分钟，cloud-init 执行 3 分钟。

## Key takeaways

**先小后大，随时扩容。** ECS 支持在线变配实例规格。别靠猜来决定配置，先跑起来，看监控数据再调整。

**选对实例族。** 通用型（g-series）是默认选项。CPU 密集型换计算型（c-series），内存密集型换内存型（r-series），ML 任务上 GPU（gn-series）。想省 20-30% 成本，试试 ARM 架构（y-suffix）。

**混合计费模式。** 长期生产用包年包月，开发测试用按量付费（PAYG），批处理任务用抢占式实例。混合策略通常比全按量付费省 40-50%。

**开机即自动化。** cloud-init 应该让实例启动后就处于生产就绪状态，无需手动 SSH 安装软件。如果你还得 SSH 进去装东西，说明 cloud-init 脚本写得不够完善。

**安全组必不可少。** 默认拒绝，只开必要的端口。内部流量用安全组引用而不是 IP 地址，每季度审查一次规则。

**快照是救命稻草。** 自动每日快照几乎不花钱，但已经救过我好几次命了。

如果想用 Infrastructure-as-Code 的方式管理 ECS，可以参考我们的 [Terraform 系列第 4 部分：计算资源](/zh/terraform-agents/04-compute-for-agent-runtime/)，里面用 Terraform 资源实现了同样的概念。本系列下一篇 [第 3 部分](/zh/aliyun-fullstack/03-vpc-networking/) 会深入讲解 VPC、交换机、路由表和 NAT 网关——这是连接所有 ECS 实例的网络层基础。