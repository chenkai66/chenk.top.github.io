---
title: "用 Terraform 给 AI Agent 上云（四）：计算层选 ECS、ACK 还是函数计算？"
date: 2026-03-18 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - ECS
  - ACK
  - Function Compute
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 4
description: "Agent 主循环在阿里云上有三个合理落点：长跑 ECS + pm2、ACK 上的 Kubernetes Pod、函数计算触发式调用。我用来挑选的成本拐点模型，再加一段真实的 cloud-init 脚本，从裸 Ubuntu 到 Agent 跑起来 90 秒搞定。"
disableNunjucks: true
translationKey: "terraform-agents-4"
---

Agent 系统最重要的架构决策就是 *Agent 主循环进程到底跑在哪里*。阿里云上有三个合理答案，再加一个几乎所有人都会忘的第四个。选错不会致命——后面可以迁——但会让你浪费几周搭无谓的脚手架，外加每月几千块的空闲算力账单。

本篇用真实 Terraform、成本拐点和我反复栽进去的运维坑，走完四种方案。

## 四种运行模式

![Agent 的三种主要部署位置：ECS、ACK、FC](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig1_three_compute_patterns.png)

每种模式各有甜区：

- **ECS** 是一台 Linux 虚拟机。长跑、有状态、调试时能 SSH 进去。原型期、单租户 Agent，以及任何需要保留一台"热机"缓存模型或本地状态的场景，都该选它。
- **ACK**（容器服务 Kubernetes 版）是规模化生产的答案。多种 Agent 并存、自动扩缩、滚动发布、GPU 调度。只有当你至少有三四种 Agent 服务，并且团队里有熟 K8s 的 SRE 时，这套运维负担才划得来。
- **函数计算（FC）** 按调用计费、scale-to-zero。冷启动 200-800ms，单次调用上限 24 小时。适合 Webhook 触发的 Agent、定时爬虫，以及任何"突发跑、其余空闲"的工作负载。
- **弹性容器实例（ECI）** 是大家忘的那一个——一个没有 K8s 节点的容器。冷启动 ~5 秒，按秒计费，没有节点池要管。甜区是 2-30 分钟一次、每小时跑几次的突发批任务。

前三种覆盖大多数场景。ECI 填的是"FC 跑不完、ECS 又太浪费"的那段空白。

## 成本拐点

按持续 QPS 粗估的月度成本曲线：

![计算成本拐点——粗略模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig2_compute_cost_curve.png)

持续 QPS 低于 ~1 时 FC 完胜——空闲几乎不花钱。1 到 30 QPS 之间单台 ECS 最划算。超过 30，ACK 较高的固定成本被流量摊薄，比往 ECS 上叠机器便宜。ECI 是另一条正交曲线：利用率低于 50% 比 ECS 便宜，高于则反过来。

模型粗略——实际数字取决于实例族、网络、Agent 通信频率——但*趋势*稳。我用的决策规则：

> 突发 + 平均负载低 → 函数计算
>
> 稳定 + 中低负载 → ECS + pm2
>
> 多 Agent + 持续中高负载 → ACK
>
> 突发批任务、单次 2-30 分钟 → ECI

框架立完了，逐个看每种模式和落地的 Terraform。

## 模式 1：ECS + pm2

80% 的 Agent 项目要的就是这个。一两台 ECS 挂在 ALB 后面，每台用 `pm2` 做 Python 或 Node Agent 进程的 supervisor。

```hcl
data "alicloud_images" "ubuntu" {
  owners      = "system"
  name_regex  = "^ubuntu_22_04_x64.*"
  most_recent = true
}

data "alicloud_instance_types" "agent" {
  cpu_core_count       = 4
  memory_size          = 16
  availability_zone    = "cn-shanghai-l"
  instance_type_family = "ecs.c7"
}

resource "alicloud_instance" "agent" {
  count = var.agent_count

  instance_name        = "agent-${terraform.workspace}-${count.index + 1}"
  image_id             = data.alicloud_images.ubuntu.images[0].id
  instance_type        = data.alicloud_instance_types.agent.instance_types[0].id
  availability_zone    = "cn-shanghai-l"

  vswitch_id      = module.vpc.private_vswitch_ids[count.index % 3]
  security_groups = [module.vpc.agent_runtime_sg_id]

  system_disk_category   = "cloud_essd"
  system_disk_size       = 80
  system_disk_encrypted  = true
  system_disk_kms_key_id = module.vpc.kms_keys["memory"]

  user_data = base64encode(templatefile("${path.module}/cloud-init.sh", {
    repo_url     = var.agent_repo_url
    branch       = var.agent_branch
    gateway_url  = "http://${alicloud_alb_listener.gateway.id}.alb.aliyuncs.com"
    sls_project  = alicloud_log_project.agents.name
    sls_logstore = alicloud_log_store.agent_runs.name
    role_name    = alicloud_ram_role.agent.name
    region       = "cn-shanghai"
  }))

  tags = {
    Role = "agent-runtime"
    App  = "research-agent"
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [user_data]   # 别在每次 cloud-init 改动时重建
  }
}
```

三个值得强调的点：

1. **`data` 块挑镜像和实例类型**，而不是硬写。`ubuntu_22_04_x64` 解析到最新打补丁的镜像；`data.alicloud_instance_types.agent` 找一个 `ecs.c7` 4 vCPU 16 GiB 的实例。阿里云某天弃用某个 SKU，下次 plan 会自动拿到新的。硬编码 `ecs.c7.xlarge` 在你可用区缺货那天就会 plan 失败——让数据源动态选才能优雅降级。
2. **`system_disk_kms_key_id` 把磁盘和第三篇里的 `memory` CMK 绑起来。** 静态加密不额外花钱，省掉一整堆合规头痛。
3. **`lifecycle { create_before_destroy = true }`** 让计划内的替换先建新实例、挂上 ALB、排空老的、再销毁——零停机轮换。代价是短时需要 2 倍容量，两台 fleet 没问题，到 50 台就开始重要。

### 真能扛住生产 bootstrap 的 cloud-init

大多数博客给的 cloud-init 是 happy path：`apt-get install`、`git clone`、`pm2 start`，完事。Demo 够用。生产 bootstrap 要处理 happy path 没讲的六件事：DNS 竞争、慢的公网镜像、RAM 凭据时序、boot 盘快照泄密、root vs 服务账号、以及 ALB 健康检查能 gate 的完成标记。

我实际跑的版本：

```bash
#!/bin/bash
set -euxo pipefail

# 1. 等网络——cloud-init 启动时 DNS 不一定就绪
for i in {1..30}; do
  curl -sf https://mirrors.aliyun.com/ -o /dev/null && break
  sleep 2
done

# 2. 用阿里云 apt 镜像——cn-shanghai 上公网 Ubuntu 镜像慢得离谱
sed -i 's|http://.*\.ubuntu\.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl ca-certificates jq nodejs npm
npm install -g pm2 uv

# 3. 在任何工具需要前先从元数据拿 instance role 凭据
TOKEN_URL="http://100.100.100.200/latest/meta-data/ram/security-credentials/${role_name}"
TOKEN_JSON=$(curl -s --max-time 5 "$TOKEN_URL")
export ALICLOUD_ACCESS_KEY_ID=$(echo $TOKEN_JSON | jq -r .AccessKeyId)
export ALICLOUD_ACCESS_KEY_SECRET=$(echo $TOKEN_JSON | jq -r .AccessKeySecret)
export ALICLOUD_SECURITY_TOKEN=$(echo $TOKEN_JSON | jq -r .SecurityToken)

# 4. 从 KMS 把密钥拉到 tmpfs——绝不落盘、绝不出现在 boot 盘快照里
mkdir -p /run/agent
mount -t tmpfs -o size=64M tmpfs /run/agent
python3 - <<'PY'
import os
from alibabacloud_kms20160120.client import Client as KmsClient
from alibabacloud_tea_openapi.models import Config
client = KmsClient(Config(
    region_id='${region}',
    access_key_id=os.environ['ALICLOUD_ACCESS_KEY_ID'],
    access_key_secret=os.environ['ALICLOUD_ACCESS_KEY_SECRET'],
    security_token=os.environ['ALICLOUD_SECURITY_TOKEN'],
))
for name in ['rds-password', 'litellm-master-key']:
    resp = client.get_secret_value(...)
    open(f'/run/agent/{name}', 'w').write(resp.body.secret_data)
PY
chmod 600 /run/agent/*

# 5. 拉 Agent runtime，用 uv 装依赖（比 pip 快 10 倍）
mkdir -p /opt/agent && cd /opt/agent
git clone --depth 1 -b ${branch} ${repo_url} src
cd src && uv venv .venv && uv pip sync requirements.txt

cat > /opt/agent/src/.env <<EOF
LLM_GATEWAY_URL=${gateway_url}
SLS_PROJECT=${sls_project}
SLS_LOGSTORE=${sls_logstore}
EOF

# 6. 用非 root 服务账号跑应用
useradd -r -s /bin/false agent || true
chown -R agent:agent /opt/agent /run/agent

# 7. 完成标记，让 ALB 健康检查能 gate
sudo -u agent pm2 start ecosystem.config.js
sudo -u agent pm2 save
echo "$(date -Iseconds)" > /run/agent/bootstrap-done
```

整个流程：

![Cloud-init bootstrap 流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig3_cloud_init_flow.png)

从 `apply` 到 `pm2 status` 显示 `online` 大约 90 秒。第一次 `apt-get install` 是慢的那一步（~60s）。等你有了稳定镜像，**用 Packer 烤一份**，后续 ECS 跳过 apt，25 秒就能启动。

> **实操提示：** `user_data` 在实例上写到 `/var/log/cloud-init-output.log`。Agent 起不来时先看这里。脚本顶部的 `set -euxo pipefail` 让失败响亮可追——没有它我曾花两小时排一个静默的 `pip install` 失败，最后发现是缺 `gcc`。

## 模式 2：ACK 跑生产 fleet

当三种以上 Agent 并存时，每台 VM 的运维成本会主导，ECS 不再 scale。ACK 给你一个集群、一个调度器、一条升级路径，外加一个统一接自动扩缩和可观测的入口。

阿里云上跑托管 K8s 集群的最小 Terraform：

```hcl
resource "alicloud_cs_managed_kubernetes" "agents" {
  name_prefix                  = "agents-${terraform.workspace}"
  version                      = "1.30.1-aliyun.1"
  cluster_spec                 = "ack.pro.small"
  vpc_id                       = module.vpc.vpc_id
  worker_vswitch_ids           = module.vpc.private_vswitch_ids
  pod_vswitch_ids              = module.vpc.private_vswitch_ids
  service_cidr                 = "172.21.0.0/20"
  proxy_mode                   = "ipvs"
  load_balancer_spec           = "slb.s2.small"
  enable_ssh                   = false
  delete_protection            = true
  control_plane_log_components = ["apiserver", "audit"]

  addons { name = "managed-arms-prometheus" }
  addons { name = "logtail-ds" }
}

resource "alicloud_cs_kubernetes_node_pool" "agents" {
  cluster_id            = alicloud_cs_managed_kubernetes.agents.id
  node_pool_name        = "agent-workers"
  vswitch_ids           = module.vpc.private_vswitch_ids
  instance_types        = ["ecs.c7.xlarge"]
  desired_size          = 3
  system_disk_category  = "cloud_essd"
  system_disk_size      = 80
  install_cloud_monitor = true
  scaling_config {
    enable   = true
    min_size = 2
    max_size = 10
  }
}
```

几个要点：

- **`ack.pro.small`** 是托管控制平面规格。阿里云负责 master，你只为 worker ECS 付费——控制平面本身约 ¥350/月。除非有特别理由，别选非托管。
- **`pod_vswitch_ids`** 是给 Terway 用的，阿里云原生 CNI。每个 Pod 拿到一个真实的 VPC IP——没有 overlay 网络，安全组直接生效。这是该选的默认；Flannel 会让网络排障痛不欲生。
- **`delete_protection = true`** 顾名思义——`terraform destroy` 杀不掉集群。每个生产集群都该开。
- `addons` 块启用 ARMS Prometheus（第七篇详解）和 SLS 日志采集。用 Terraform 配进去，新集群自带可观测性。

实际的 Agent Pod 通过 Kubernetes Deployment 清单部署——通常用单独的 `kubectl` 步骤，或借 `kubernetes` Terraform provider 管理。我把集群放在这个 `terraform` 项目里，把 workload 放在独立的 Helm Chart 里，因为两者发布节奏完全不同。集群一个季度改一次；Agent 镜像一天改十次。

## 模式 3：函数计算跑事件触发 Agent

有些 Agent 只在被触发时跑——webhook、cron、OSS 对象落地。这类场景 FC 无敌：零空闲成本、自动 scale-out、runtime 完全云端托管。

```hcl
resource "alicloud_fc_service" "agent" {
  name        = "agent-${terraform.workspace}"
  description = "事件触发的 Agent 函数"

  log_config {
    project  = alicloud_log_project.agents.name
    logstore = alicloud_log_store.agent_runs.name
  }

  vpc_config {
    vswitch_ids       = module.vpc.private_vswitch_ids
    security_group_id = module.vpc.agent_runtime_sg_id
  }

  role            = alicloud_ram_role.fc_agent.arn
  internet_access = false
}

resource "alicloud_fc_function" "scheduled_research" {
  service     = alicloud_fc_service.agent.name
  name        = "scheduled-research"
  description = "每日 research agent 跑批"
  filename    = "${path.module}/dist/scheduled-research.zip"
  handler     = "index.handler"
  runtime     = "python3.11"
  memory_size = 1024
  timeout     = 600
  ca_port     = 8080

  environment_variables = {
    LLM_GATEWAY_URL = "http://${alicloud_alb_listener.gateway.id}.alb.aliyuncs.com"
  }
}

resource "alicloud_fc_trigger" "daily" {
  service  = alicloud_fc_service.agent.name
  function = alicloud_fc_function.scheduled_research.name
  name     = "daily-9am"
  type     = "timer"
  config = jsonencode({
    cronExpression = "0 0 9 * * *"
    enable         = true
    payload        = "{}"
  })
}
```

得到的结果：Python 3.11 函数，1 GiB 内存，10 分钟超时，挂在和其他资源同一 VPC 同一安全组上，每天上午 9 点触发。零服务器维护。成本：这个规模约 ¥0.10/次调用，加上 ¥0.0001/GB-second 执行。每天跑 5 分钟的定时任务，月成本约 ¥3。同样的 Agent 跑在空闲 ECS 上要 ~¥250/月。

我反复栽过的三个坑：

1. **冷启动。** 空闲后第一次调用比后续多 200-800ms。亚秒 SLA 的 webhook 在意；cron 不在意。可以买 provisioned concurrency，但那就违背 FC 的初衷——既然 24/7 付暖实例的钱，不如直接 ECS。
2. **VPC 挂载** 给冷启动再加 200-400ms，因为 FC 要给你的 VPC 加一个 ENI。函数要走内网访问 RDS/OpenSearch 时值得；只调公网 API 时跳过 `vpc_config`。
3. **24 小时最长运行时间。** 长 Agent 循环不适合 FC。要么把循环切成短步骤（中间状态放 OSS 或 Redis），要么改用 ECS/ECI。

第三个坑正是 ECI 要填的空白。

## 模式 4：ECI 跑突发批量 Agent

ECS、ACK、FC 覆盖了多数场景，但还有第四个模式专治*突发批量* Agent：**弹性容器实例**。ECI 是"不带 K8s 节点的容器"——阿里云直接在他们管的裸金属池上跑你的容器，你只为运行的秒数付费。

场景很尖。Agent 每次跑 2-30 分钟，每小时几次，常常并行。函数计算上限 24 小时但批任务的冷启动很糙；ECS 浪费（突发之间空着也付钱）；ACK 多了一层 K8s 运维负担你不想要。ECI 在中间命中：冷启动 ~5 秒、零空闲成本、不用管 node pool。

```hcl
resource "alicloud_eci_container_group" "research_batch" {
  container_group_name = "research-batch-${formatdate("YYYYMMDDhhmm", timestamp())}"
  security_group_id    = module.vpc.agent_runtime_sg_id
  vswitch_id           = module.vpc.private_vswitch_ids[0]
  cpu                  = 4
  memory               = 16
  restart_policy       = "OnFailure"
  ram_role_name        = alicloud_ram_role.agent_eci.name

  containers {
    name              = "research"
    image             = "registry-vpc.cn-shanghai.aliyuncs.com/agents/research:${var.image_tag}"
    image_pull_policy = "IfNotPresent"
    cpu               = 4
    memory            = 16
    commands          = ["python", "-m", "research_agent"]
    args              = ["--session-id", var.session_id]

    environment_vars {
      key   = "LLM_GATEWAY_URL"
      value = module.gateway.url
    }
    volume_mounts {
      name       = "scratch"
      mount_path = "/scratch"
    }
  }

  volumes {
    name = "scratch"
    type = "EmptyDirVolume"
  }

  image_registry_credential {
    server    = "registry-vpc.cn-shanghai.aliyuncs.com"
    user_name = "agent-puller"
    password  = data.alicloud_kms_secret.acr_pull.secret_data
  }

  lifecycle {
    create_before_destroy = false                       # ECI 单次执行
    ignore_changes        = [container_group_name]      # 时间戳后缀会漂
  }
}
```

我踩过的生产坑：

- **镜像仓库走 `registry-vpc` endpoint**，不是公网。公网走 NAT 还要付出网费；VPC endpoint 免费而且更快。我曾经一周烧了 ¥200 出网费才发现配错了。
- **`restart_policy = "OnFailure"` 不是 `"Always"`。** ECI 设计上单次执行；`Always` 让它跑成功也循环——和批任务想要的相反。
- **`EmptyDirVolume` 是容器组级 ephemeral**，和 K8s emptyDir 一样。别存任何不能重生的东西——输出写 OSS 或 RDS。
- **`ram_role_name`** 让容器从实例元数据拿凭据，和 ECS 一样的姿势。环境变量里没有 AK/SK。

`alicloud_eci_container_group` 资源有个烦人的怪癖：`containers[*]` 的变更不一定干净触发 replace，因为 API 部分是 merge 语义。生产批任务我用 Terraform 一次性 provision 容器组（运行时 spec 上 `ignore_changes`），新一轮跑通过 orchestrator 直接调 ECI API 触发，用 Terraform 准备好的 role 和 SG。**Terraform 这边当*模板*用，不是*执行器*。**

成本算术：4 vCPU / 16 GB ECI 大约 ¥0.96/小时，按秒计费、最少 1 分钟。一次 8 分钟的研究跑 ~¥0.13。每天 100 次 = ¥13/天 = ¥390/月。同样 workload 常驻 ECS 即使空闲也要 ¥250-400/月。**利用率 50% 以下 ECI 赢；以上 ECS 或 ACK 赢。**

## 一个真实案例：混合架构

我在生产环境部署过的多数 Agent 栈最后都是混合的：

- ECS 跑常驻对话 Agent，会话状态在内存里
- ACK 跑后台任务的 worker 集群，三四种 Agent 一起调度
- FC 跑 webhook 接收和每天的 cron 任务
- ECI 跑每天几十次、单次 5-20 分钟的批任务

Terraform 让这件事很简单——同一个项目里四个模块，共享第三篇里的 VPC 和安全组。真正的技巧是知道哪种工作负载配哪种模式，而不是死记每种资源的语法。

## 实例规格选型

经常有人问：跑 Agent runtime 该选哪个 `ecs.*` 系列？我的默认：

| 工作负载                       | 推荐系列              | 原因 |
|--------------------------------|----------------------|------|
| 对话型 Agent，无 GPU            | `ecs.c7`             | 瓶颈在 tokenize 的 CPU 和 LLM 调用的 I/O |
| 内存密集（大上下文）            | `ecs.r7`             | 每 vCPU 配更多内存 |
| 批处理/定时（有突发）           | `ecs.c7a`（AMD）      | 便宜 ~15%，单核略慢 |
| 小模型 GPU 推理                 | `ecs.gn7i`           | T4 级别，阿里云上最便宜的 GPU |
| 预训练或大规模微调              | 用 PAI-DLC，不是 ECS | 别重复造编排轮子 |

避免突发性能型 `ecs.t6` 跑 Agent runtime——CPU 积分在持续负载下会耗尽，延迟立刻断崖。这类实例适合跑 `terraform apply` 的堡垒机这种轻量场景。

## 当 `alicloud_pai_*` 资源缺失时（以及应对方案）

如果你部署的 Agent 自己训练模型或托管自定义 LLM，第一反应是用 Terraform 起 PAI-EAS。现实是：截至 `alicloud` provider 1.230，PAI 资源覆盖很薄。`alicloud_pai_workspace` 和几个相关资源有，但完整的 EAS 服务部署不是一等公民——没有 `alicloud_pai_eas_service` 让你在 HCL 里声明式定义模型服务端点。

我用过的几个替代方案，按优先级排：

**方案 1：Terraform 管 workspace，EAS 走 API。** Terraform 拥有 workspace 和 IAM；命令行工具（`eascmd` 或 EAS Python SDK）拥有运行时 spec。

```hcl
resource "alicloud_pai_workspace" "agents" {
  workspace_name = "agents-${terraform.workspace}"
  description    = "PAI workspace for agent inference"
  env_types      = ["prod"]
  display_name   = "Agents"
}

output "pai_workspace_id" {
  value = alicloud_pai_workspace.agents.id
}
```

然后用 `Makefile` 里的 `eascmd create eas-service.json`，JSON 描述镜像、processor、实例规格、自动扩缩。这种分工诚实：稳定底座声明式，运行时命令式。

**方案 2：用 ROS 部署 EAS。** ROS（阿里云原生 IaC）经常比 Terraform provider 早支持新资源。可以在 Terraform 里通过 `alicloud_ros_stack` 调用 ROS 模板：

```hcl
resource "alicloud_ros_stack" "eas_serving" {
  stack_name    = "eas-serving-${terraform.workspace}"
  template_body = file("${path.module}/eas-service.ros.json")

  parameters {
    parameter_key   = "WorkspaceId"
    parameter_value = alicloud_pai_workspace.agents.id
  }
  parameters {
    parameter_key   = "ImageUrl"
    parameter_value = "registry-vpc.cn-shanghai.aliyuncs.com/agents/qwen-server:${var.model_version}"
  }
  timeout_in_minutes = 30
}
```

`eas-service.ros.json` 是 ROS 模板（阿里云风味的 CloudFormation），暴露了 alicloud Terraform provider 还没暴露的 EAS 属性。Terraform 拥有 stack，ROS 拥有深层云资源，你在两者之上拿到统一的 `terraform plan`。

**方案 3：自己写 provider。** 别。我干过一次——两周 Go 代码，一个月后官方 provider 把我要的资源加上了，我多了一份要永远维护的 fork。阿里云团队每月都有新 alicloud provider 资源；等他们追上比维护 fork 划算。

通用原则，对所有"Terraform 还没支持"的场景：**Terraform 拥有稳定底座，轻量工具拥有变化部分。** RAM 角色、VPC、KMS、OSS——Terraform。全新资源、Beta API——`null_resource` + `local-exec`、嵌入 ROS、或独立的编排器。等 provider 追上，再用 `terraform import` 把 `null_resource` 块迁成正式资源。

> **实战小贴士：** `alicloud_ros_stack` 是被严重低估的资源。alicloud Terraform provider 滞后时，ROS 经常领先一个季度。在 Terraform 里嵌入 ROS 模板，两边的好处都拿到。

## 接下来的内容

第五篇补存储层——向量库、关系型、对象存储、备份——我们刚 provision 出来的所有计算资源都需要的"记忆存放地"。ECS 实例、ACK Pod、FC 函数、ECI 容器，没有存储都只是空壳。

接下来第六篇在所有计算资源前面搭 LLM 网关，第七篇接可观测性和成本告警，第八篇用一次 `terraform apply` 把整套技术栈缝在一起。
