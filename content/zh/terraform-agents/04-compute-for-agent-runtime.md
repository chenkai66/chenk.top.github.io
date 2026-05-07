---
title: "用 Terraform 给 AI Agent 上云（四）：计算层选 ECS、ACK 还是函数计算？"
date: 2026-03-18 09:00:00
tags:
  - Terraform
  - 阿里云
  - ECS
  - ACK
  - 函数计算
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 4
description: "Agent 主循环在阿里云上有三个合理落点：长跑 ECS + pm2、ACK 上的 Kubernetes Pod、Function Compute 触发式调用。我用来挑选的成本拐点模型，再加一段真实的 cloud-init 脚本，从裸 Ubuntu 到 Agent 跑起来 90 秒搞定。"
disableNunjucks: true
translationKey: "terraform-agents-4"
---

Agent 系统最重要的架构决策就是 *Agent 主循环进程到底跑在哪里*。阿里云上正好有三个好答案。选错不会致命——后面可以迁——但会让你浪费几周搭无谓的脚手架。

本篇用真实 Terraform、成本拐点和运维坑走完三种方案。

![用 Terraform 给 AI Agent 上云（四）：计算层选 ECS、ACK 还是函数计算？ — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/illustration_1.jpg)

## 三种运行模式

![Agent 的三种部署位置：ECS、ACK、FC](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig1_three_compute_patterns.png)

每种模式都有其适用场景：

- **ECS** 是一台 Linux 虚拟机，适合长期运行、有状态的服务，调试时可以通过 SSH 快速进入。无论是开发原型、单租户 Agent，还是需要保留一台“热机”以缓存模型或维护本地状态的场景，ECS 都是理想选择。
- **ACK**（容器服务 Kubernetes 版）是大规模生产环境的最佳实践。它支持多种 Agent 类型、自动扩缩容、滚动更新以及 GPU 调度。不过，只有当你的服务包含至少三到四个 Agent，并且团队中有熟悉 K8s 的 SRE 时，ACK 的复杂性才显得物有所值。
- **Function Compute** 按调用计费，支持 scale-to-zero，冷启动时间为 200-800 毫秒。非常适合由 Webhook 触发的 Agent、定时任务爬虫，或者那些需要短时间爆发运行、其余时间保持空闲的任务。
## 成本临界点

以下是根据持续 QPS 粗略估算的月度成本分布情况：

![计算成本临界点——粗略模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig2_compute_cost_curve.png)

当持续 QPS 低于约 1 时，Function Compute（FC）占据绝对优势——空闲时段几乎不产生费用。在持续 QPS 介于 1 到 30 之间时，单台 ECS 实例是最经济的选择。而当 QPS 超过 30 后，ACK 的较高固定成本会被足够的流量摊薄，相比在 ECS 上堆叠更多实例，ACK 反而更划算。

这个模型比较粗略——实际数字会因实例规格族、网络环境以及 Agent 的通信频率而有所不同——但整体趋势是可靠的。我通常采用以下决策规则：

> 突发流量 + 平均负载低  →  Function Compute
>
> 稳定流量 + 中低负载  →  ECS 搭配 pm2
>
> 多 Agent 场景 + 持续中高负载  →  ACK
## 模式 1：ECS + pm2

80% 的 Agent 项目你想要的就是这个。一两台 ECS 挂在 ALB 后面，每台用 `pm2` 当 Python 或 Node Agent 进程的 supervisor。

官方"Create an ECS instance"实践文档给了能跑的基线。我们在 Agent 场景下改写：

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

  system_disk_category = "cloud_essd"
  system_disk_size     = 80
  system_disk_encrypted = true
  system_disk_kms_key_id = module.vpc.kms_keys["memory"]

  user_data = base64encode(templatefile("${path.module}/cloud-init.sh", {
    repo_url       = var.agent_repo_url
    branch         = var.agent_branch
    gateway_url    = "http://${alicloud_alb_listener.gateway.id}.alb.aliyuncs.com"
    sls_project    = alicloud_log_project.agents.name
    sls_logstore   = alicloud_log_store.agent_runs.name
  }))

  tags = {
    Role = "agent-runtime"
    App  = "research-agent"
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes = [user_data]    # 别在每次 cloud-init 改动时重建
  }
}
```

三个值得强调的点：

1. **`data` 块挑镜像和实例类型**而不是硬写。`ubuntu_22_04_x64` 解析到最新打补丁的镜像；`data.alicloud_instance_types.agent` 找一个 `ecs.c7` 4 vCPU 16 GiB 的。阿里云某天弃用某个 SKU 时，下次 plan 会自动拿到新的。
2. **`system_disk_kms_key_id` 把磁盘和第三篇里的 `memory` CMK 绑起来。** 静态加密不额外花钱，去掉一整堆合规头痛。
3. **`lifecycle { create_before_destroy = true }`** 意味着计划内的替换会先建新实例、挂上 ALB、排空老的、再销毁——零停机轮换。代价是你短时需要 2 倍容量，两台 fleet 没问题，到 50 台就开始重要。

### Cloud-init 脚本

`cloud-init.sh` 是 templatefile，把机器从裸 Ubuntu 启动到 Agent 跑起来：

```bash
#!/bin/bash
set -euxo pipefail

# 更新 apt 装基础依赖
apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl ca-certificates

# Node 20 给 Agent 偶尔 shell 出去用的 JS 工具
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# pm2 当进程 supervisor
npm install -g pm2 uv

# 拉 Agent runtime
mkdir -p /opt/agent
cd /opt/agent
git clone --depth 1 -b ${branch} ${repo_url} src
cd src
uv venv .venv
uv pip sync requirements.txt

# 写环境变量（来自 Terraform 的 templatefile）
cat > /opt/agent/src/.env <<EOF
LLM_GATEWAY_URL=${gateway_url}
SLS_PROJECT=${sls_project}
SLS_LOGSTORE=${sls_logstore}
EOF

# pm2 拉起来 + 持久化
pm2 start ecosystem.config.js
pm2 save
pm2 startup systemd -u root --hp /root
```

整个流程：

![Cloud-init bootstrap 流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/fig3_cloud_init_flow.png)

从 `apply` 到 `pm2 status` 显示 `online` 大约 90 秒。第一次 `apt-get install` 是慢的那一步（~60s）。等你有了稳定镜像，**用 Packer 烤一份**，后续 ECS 跳过 apt，25 秒就能启动。

> **实操提示：** `user_data` 在实例上写到 `/var/log/cloud-init-output.log`。Agent 起不来时先看这里。脚本顶部加 `set -euxo pipefail`，让失败响亮可追。

## 模式 2：使用 ACK 部署生产环境的集群

当你需要同时运行三种或更多类型的 Agent 时，每台虚拟机（VM）的运维成本会成为主要开销。而阿里云容器服务 Kubernetes 版（ACK）为你提供了一个统一的解决方案：一个集群、一个调度器、一条升级路径，大大简化了管理复杂度。

以下是最简化的 Terraform 配置，用于在阿里云上创建托管的 Kubernetes 集群：

```hcl
resource "alicloud_cs_managed_kubernetes" "agents" {
  name_prefix          = "agents-${terraform.workspace}"
  version              = "1.30.1-aliyun.1"
  cluster_spec         = "ack.pro.small"
  vpc_id               = module.vpc.vpc_id
  worker_vswitch_ids   = module.vpc.private_vswitch_ids
  pod_vswitch_ids      = module.vpc.private_vswitch_ids
  service_cidr         = "172.21.0.0/20"
  proxy_mode           = "ipvs"
  load_balancer_spec   = "slb.s2.small"
  enable_ssh           = false
  delete_protection    = true
  control_plane_log_components = ["apiserver", "audit"]

  addons {
    name = "managed-arms-prometheus"
  }
  addons {
    name = "logtail-ds"
  }
}

resource "alicloud_cs_kubernetes_node_pool" "agents" {
  cluster_id           = alicloud_cs_managed_kubernetes.agents.id
  node_pool_name       = "agent-workers"
  vswitch_ids          = module.vpc.private_vswitch_ids
  instance_types       = ["ecs.c7.xlarge"]
  desired_size         = 3
  system_disk_category = "cloud_essd"
  system_disk_size     = 80
  install_cloud_monitor = true
  scaling_config {
    enable    = true
    min_size  = 2
    max_size  = 10
  }
}
```

### 关键点说明

- **`ack.pro.small`** 是阿里云托管控制平面的规格。阿里云负责管理 master 节点，你只需为 worker 节点（ECS 实例）付费。除非有特殊需求，否则不建议选择非托管规格。
- **`pod_vswitch_ids`** 是为 Terway（阿里云原生 CNI 插件）配置的。每个 Pod 都会分配一个真实的 VPC IP 地址，无需额外的 overlay 网络，安全组规则可以直接应用。这是推荐的默认配置；相比之下，Flannel 会让网络调试变得异常痛苦。
- **`delete_protection = true`** 的作用正如其名——即使执行 `terraform destroy`，也不会误删集群。建议在所有生产环境中启用此选项。
- **`addons`** 配置块启用了 ARMS Prometheus（详见第七篇文章）和 SLS 日志采集功能。通过 Terraform 配置这些插件，可以让新创建的集群自带监控能力，省去后续手动配置的麻烦。

### Agent Pod 的部署方式

实际的 Agent Pod 通常通过 Kubernetes 的 Deployment 清单文件来定义。这类资源可以通过单独的 `kubectl apply` 命令应用，也可以借助 `kubernetes` Terraform 提供商进行管理。为了更好地分离关注点，我将集群基础设施的管理放在当前的 `terraform` 项目中，而将工作负载（workload）的管理交给独立的 Helm Chart。这样做的原因是两者发布节奏不同，分开管理更加灵活高效。
## 模式 3：Function Compute 跑事件触发 Agent

有些 Agent 只在被触发时才跑——webhook、cron、OSS 对象落地。这种场景 FC 无敌：零空闲成本、自动 scale-out、runtime 完全云端托管。

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

  role          = alicloud_ram_role.fc_agent.arn
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

它给你的：Python 3.11 函数，1 GiB 内存，10 分钟超时，挂在和其他资源同一 VPC 同一安全组上，每天上午 9 点触发。零服务器维护。成本：这个规模约 ¥0.10/次调用，加上 ¥0.0001/GB-second 执行。

我反复栽过的三个坑：

1. **冷启动。** 空闲后第一次调用比后续多 200-800ms。亚秒 SLA 的 webhook 在意；cron 不在意。可以买 provisioned concurrency，但那就违背 FC 的初衷了。
2. **VPC 挂载** 给冷启动再加 200-400ms，因为 FC 要给你 VPC 加一个 ENI。函数要访问 RDS/OpenSearch 时值得；只调公网 API 时跳过 `vpc_config`。
3. **24 小时最长运行时间。** 长 Agent 循环不适合 FC。要么把循环切成短步骤，要么用 ECS。

## 一个实际案例：混合架构

在实际生产环境中，我部署过的大多数 Agent 栈最终都采用了混合架构：

- 使用 ECS 来运行需要保持会话状态的常驻对话 Agent
- 使用 ACK 来管理处理后台任务的 worker 节点集群
- 使用 FC 来处理 webhook 请求以及每天或每小时的定时任务

借助 Terraform，这种架构的实现变得非常简单——只需在同一个项目中创建三个模块，共享第三篇文章中提到的 VPC 和安全组配置即可。真正的技巧在于根据工作负载选择合适的模式，而不是死记硬背各种资源的语法。
## 实例规格选型

![用 Terraform 给 AI Agent 上云（四）：计算层选 ECS、ACK 还是函数计算？ — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/04-compute-for-agent-runtime/illustration_2.jpg)

在实际工作中，经常会有人问：运行 Agent 时应该选择哪个 `ecs.*` 系列？以下是我的推荐配置：

| 工作负载类型                   | 推荐系列      | 推荐理由 |
|--------------------------------|---------------|----------|
| 对话型 Agent，无需 GPU         | `ecs.c7`      | 主要瓶颈在于 Tokenize 的 CPU 消耗和 LLM 调用的 I/O 开销 |
| 内存密集型（大上下文场景）     | `ecs.r7`      | 每个 vCPU 配备更多内存，适合高内存需求的任务 |
| 批处理/定时任务（有突发需求）  | `ecs.c7a`（AMD） | 性价比高，比标准实例便宜约 15%，但单核性能略低 |
| 小模型 GPU 推理               | `ecs.gn7i`    | 基于 T4 级别的 GPU，阿里云上最经济的 GPU 实例 |
| 预训练或大规模微调             | 使用 PAI-DLC，而非 ECS | 不必重复造轮子，PAI-DLC 提供更专业的编排能力 |

对于持续运行的 Agent runtime，尽量避免使用突发性能型实例 `ecs.t6`——因为在这种实例下，CPU 积分会在高负载时迅速耗尽，导致延迟飙升。这类实例更适合用来运行像 `terraform apply` 这样的轻量级任务，比如堡垒机。

> **实战经验分享：** 使用 `data.alicloud_instance_types` 数据源，动态查询当前可用区中满足条件的实例类型，例如“4 vCPU、16 GiB 内存的实例”。如果直接硬编码指定 `ecs.c7.xlarge`，一旦该规格在你的可用区缺货，Terraform 就会报错失败。而通过数据源动态选择，可以实现更灵活的降级方案，确保部署流程更加稳健。
## 接下来的内容

第五篇文章将补充存储层的内容——包括向量存储、关系型数据库、对象存储以及备份系统。这些是我们刚刚创建的所有计算资源正常运行所必需的“记忆存放地”。如果没有这些存储设施，ECS 实例就像是没有内存的空壳，无法发挥作用。

接下来，第六篇文章会在所有计算资源的前面搭建 LLM 网关，第七篇文章会接入可观测性和成本告警功能，而第八篇文章则会通过一次 `terraform apply` 操作，将整个技术栈无缝整合在一起。
## ECI——大家忘了的第四种 Agent 算力

ECS、ACK、FC 是文章主线讲的三个，覆盖大多数场景。但还有一个**突发型**批量 Agent 任务的天然解：**弹性容器实例（ECI）**。ECI 是"不带 K8s 节点的容器"——阿里云直接在他们管的裸金属池上跑你的容器，你只为运行的秒数付费。

场景很尖。你的 Agent 每次跑 2-30 分钟，每小时几次，常常并行。Function Compute 上限 24 小时但批任务的冷启动很糙；ECS 浪费（突发之间空着也付钱）；ACK 多了 K8s 的运维负担。ECI 在中间命中：冷启动 ~5 秒、零空闲成本、不用管 node pool。

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
    working_dir       = "/app"

    commands = ["python", "-m", "research_agent"]
    args     = ["--session-id", var.session_id]

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
    server   = "registry-vpc.cn-shanghai.aliyuncs.com"
    user_name = "agent-puller"
    password  = data.alicloud_kms_secret.acr_pull.secret_data
  }

  lifecycle {
    create_before_destroy = false
    ignore_changes = [container_group_name]
  }
}
```

几个生产坑：

- **镜像仓库走 `registry-vpc` endpoint**，不是公网。公网走 NAT 还要付出网费；VPC endpoint 免费而且更快。
- **`restart_policy = "OnFailure"` 不是 `"Always"`。** ECI 设计上单次执行；`Always` 让它跑成功也循环——和批任务想要的相反。
- **`EmptyDirVolume` 是容器组级 ephemeral**，和 K8s emptyDir 一样。别存任何不能重生的东西——输出写 OSS 或 RDS。
- **`ram_role_name`** 让容器从实例元数据拿凭据，和 ECS 一样的姿势。环境变量里没有 AK/SK。

`alicloud_eci_container_group` 资源有个烦人的怪癖：`containers[*]` 的变更不一定干净触发 replace，因为 API 部分是 merge 语义。生产批任务我用 Terraform 一次性 provision 容器组（运行时 spec 上 `ignore_changes`），新一轮跑通过 orchestrator 直接调 ECI API 触发，用 Terraform 准备好的 role 和 SG。Terraform 这边当**模板**用，不是**执行器**。

成本算术：4 vCPU / 16 GB ECI 大约 ¥0.96/小时，按秒计费、最少 1 分钟。一次 8 分钟的研究跑 ~¥0.13。每天 100 次 = ¥13/天 = ¥390/月。同样 workload 常驻 ECS 即使空闲也要 ¥250-400/月。利用率 50% 以下 ECI 赢；以上 ECS 或 ACK 赢。

## 当 `alicloud_pai_*` 资源缺失时（以及应对方案）

如果你正在部署一个能够自主训练模型或托管自定义大语言模型（LLM）的 Agent，第一反应很可能是通过 Terraform 来启动 PAI-EAS 服务。然而，现实情况是：截至 `alicloud` 提供者版本 1.230，PAI 相关资源的支持仍然非常有限。虽然有 `alicloud_pai_workspace` 和一些相关资源，但完整的 EAS 服务部署并不属于一等公民——目前还没有类似 `alicloud_pai_eas_service` 的资源可以让你在 HCL 中声明式地定义模型服务端点。

以下是我在实际工作中使用的几种替代方案，按照优先级排序：

### 方案 1：用 Terraform 管理 workspace，通过 API 部署 EAS

```hcl
resource "alicloud_pai_workspace" "agents" {
  workspace_name        = "agents-${terraform.workspace}"
  description           = "PAI 工作空间，用于 Agent 推理"
  env_types             = ["prod"]
  display_name          = "Agents"
}

output "pai_workspace_id" {
  value = alicloud_pai_workspace.agents.id
}
```

接着，通过 `Makefile` 使用 `eascmd` 工具来声明 EAS 服务：

```bash
eascmd create eas-service.json
# eas-service.json 包含镜像、处理器类型、实例规格和自动扩展配置
```

这种分工方式清晰明了：Terraform 负责管理基础资源（如 workspace 和 IAM），而运行时的具体配置则交给命令行工具（如 `eascmd` 或 EAS Python SDK）来完成。

### 方案 2：借助 ROS 部署 EAS

ROS（阿里云原生的基础设施即代码工具）通常会比 Terraform 提供者更早支持新资源。你可以通过 `alicloud_ros_stack` 在 Terraform 中调用 ROS 模板来实现这一目标：

```hcl
resource "alicloud_ros_stack" "eas_serving" {
  stack_name        = "eas-serving-${terraform.workspace}"
  template_body     = file("${path.module}/eas-service.ros.json")
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

这里的 `eas-service.ros.json` 是一个 ROS 模板（阿里云风格的 CloudFormation），它提供了目前 alicloud Terraform 提供者尚未暴露的 EAS 资源属性。通过这种方式，Terraform 负责管理栈（stack），而 ROS 则负责处理更深层次的云资源，最终你可以在两者之间获得统一的 `terraform plan` 输出。

### 方案 3：自己编写 provider（不推荐）

理论上，你可以编写一个自定义的 Terraform provider 来封装 EAS API。我曾经尝试过一次，大约花费了两周的时间用 Go 实现。除非你有持续的需求且其他方法都无法满足，否则不建议这样做——阿里云团队每月都会发布新的 alicloud provider 资源，等待官方支持比维护一个自定义 fork 更加划算。

### 通用原则

在面对“Terraform 尚未支持”的场景时，遵循以下原则：**Terraform 负责稳定的基础资源，轻量级工具负责动态部分**。例如，RAM 角色、VPC、KMS、OSS 等稳定的基础设施可以通过 Terraform 管理；而对于全新的资源或 Beta 阶段的 API，则可以通过 `null_resource` + `local-exec`、嵌入 ROS 模板或使用独立的编排工具来实现。当官方 provider 追上后，再将这些临时方案迁移到正式资源，并通过 `terraform import` 将其纳入管理。

> **实战小贴士：** `alicloud_ros_stack` 是一个被严重低估的资源。当 alicloud Terraform 提供者滞后时，ROS 往往能提前一个季度支持相关 API。在 Terraform 中嵌入 ROS 模板，可以让你同时享受两者的优点。
## 真能扛住生产 bootstrap 的 cloud-init

文章里的 `cloud-init.sh` 是 happy path。生产里 bootstrap 脚本要处理原版没讲的六件事。这是我实际跑的版本：

```bash
#!/bin/bash
set -euxo pipefail

# 1. 等网络——cloud-init 启动时 DNS 不一定就绪
for i in {1..30}; do
  curl -sf https://mirrors.aliyun.com/ -o /dev/null && break
  sleep 2
done

# 2. 用阿里云 apt 镜像，丢掉 Ubuntu 默认源
sed -i 's|http://.*\.ubuntu\.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl ca-certificates jq

# 3. 在任何需要凭据的工具之前先拿到 instance role 凭据
TOKEN_URL="http://100.100.100.200/latest/meta-data/ram/security-credentials/${role_name}"
TOKEN_JSON=$(curl -s --max-time 5 "$TOKEN_URL")
export ALICLOUD_ACCESS_KEY_ID=$(echo $TOKEN_JSON | jq -r .AccessKeyId)
export ALICLOUD_ACCESS_KEY_SECRET=$(echo $TOKEN_JSON | jq -r .AccessKeySecret)
export ALICLOUD_SECURITY_TOKEN=$(echo $TOKEN_JSON | jq -r .SecurityToken)

# 4. 从 KMS 拉密钥，写到 tmpfs 路径——绝不落盘
mkdir -p /run/agent
mount -t tmpfs -o size=64M tmpfs /run/agent
python3 -c "
import os, json
from alibabacloud_kms20160120.client import Client as KmsClient
# ... 拉每个 secret，写到 /run/agent/<name>
"
chmod 600 /run/agent/*

# 5. 用服务账号跑应用，不要用 root
useradd -r -s /bin/false agent || true
chown -R agent:agent /opt/agent /run/agent

# 6. 让失败可见——写一个完成标记，让健康检查能 gate
echo "$(date -Iseconds)" > /run/agent/bootstrap-done
```

把 happy-path 脚本变成生产脚本的六个修复：

1. 等网络（DNS 在前 5 秒和 cloud-init 抢）
2. 阿里云 apt 镜像（公网 Ubuntu 镜像在 cn-shanghai 慢）
3. 在任何工具需要前先取 RAM 凭据
4. 密钥写 tmpfs 不落盘（避免 boot 盘快照泄漏）
5. 用非 root 服务账号跑
6. Bootstrap 完成标记，让 ALB 健康检查能 gate

填完 Python 全部 ~80 行。每次写 cloud-init 都值得贴。
