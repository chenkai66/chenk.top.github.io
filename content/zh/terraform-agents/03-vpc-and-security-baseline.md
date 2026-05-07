---
title: "用 Terraform 给 AI Agent 上云（三）：可复用的 VPC 与安全基线"
date: 2026-03-16 09:00:00
tags:
  - Terraform
  - 阿里云
  - VPC
  - 安全
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 3
description: "第一个可复用 module——三可用区 VPC，公私网交换机分层，NAT 出网，按 tier 分层的安全组，再加上按数据域分的 KMS 主密钥。同样的代码出现在我交付过的每一个 Agent stack 里，参数化但本体不变。"
disableNunjucks: true
translationKey: "terraform-agents-3"
---

这一篇造的是我所有 Agent 项目里被复制粘贴最多的一段 Terraform：一个 `vpc-baseline` module，给后续每一个组件（ECS、RDS、OpenSearch、ACK）一个合理的落点。

读完之后你会拥有：

- 一个跨三个可用区的 VPC
- 六个交换机（每个区一个公网 + 一个私网），CIDR 不重叠
- 一个 NAT 网关 + EIP，让私网子网能出网调 LLM API
- 三个按 tier 叠的安全组（ALB → agent runtime → memory）
- 三把 KMS 主密钥，每个数据域一把（memory、secrets、logs）
- 一个干净的 module 接口：进 name + CIDR + zones，出一堆 ID

总共大概 200 行 HCL。一次写完，永久参考。

![用 Terraform 给 AI Agent 上云（三）：可复用的 VPC 与安全基线 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/illustration_1.jpg)

## 心智模型

在看代码之前，先来了解一下整体架构图：

![VPC 拓扑——三可用区、公私网分离、NAT 出网](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/fig1_vpc_topology.png)

为什么选择三个可用区？这是因为阿里云保留了在任意周日对某个可用区进行维护的权利。如果只部署在一个可用区内，那么在维护窗口期间，你的 Agent 将完全离线。而在 VPC 内部，跨可用区的流量是免费的；采用三可用区的唯一代价就是子网规划和管理上的复杂度稍高一些。

为什么要划分公网和私网？Agent 的运行环境应该部署在私网中，这样即使安全组配置错误，也不会意外将服务暴露给 `0.0.0.0/0`。而公网子网则用来放置 ALB（应用负载均衡）和 NAT 网关——这些必须与互联网通信的组件。Agent 通过 NAT 网关访问外部网络，而不是直接连接互联网，从而进一步提升安全性。

以下是推荐的 CIDR 划分方案：

| 子网名称            | 可用区 | CIDR 范围       | 主机数量 |
|---------------------|--------|-----------------|----------|
| `public-a`          | l      | `10.20.0.0/28`  |    11    |
| `public-b`          | m      | `10.20.0.16/28` |    11    |
| `public-c`          | n      | `10.20.0.32/28` |    11    |
| `private-a`         | l      | `10.20.1.0/24`  |   251    |
| `private-b`         | m      | `10.20.2.0/24`  |   251    |
| `private-c`         | n      | `10.20.3.0/24`  |   251    |

公网子网使用 `/28` 的掩码，因为它们只需要容纳一个 NAT 网关和一个 ALB 的 IP 地址。而私网子网则采用 `/24` 的掩码，因为这里需要承载 Agent 的 ECS 实例、RDS 数据库以及 OpenSearch 节点等资源。
## Module 骨架

建目录：

```
modules/vpc-baseline/
├── main.tf
├── variables.tf
├── outputs.tf
└── versions.tf
```

输入（`variables.tf`）：

```hcl
variable "name" {
  description = "所有资源的命名前缀，例如 agents-prod"
  type        = string
}

variable "cidr_block" {
  description = "顶层 VPC CIDR，子网从这里推导"
  type        = string
  default     = "10.20.0.0/16"
}

variable "zones" {
  description = "目标 region 的三个可用区 ID"
  type        = list(string)
  validation {
    condition     = length(var.zones) == 3
    error_message = "vpc-baseline 必须正好三个 zone。"
  }
}

variable "tags" {
  description = "module 创建的所有资源都打这些 tag"
  type        = map(string)
  default     = {}
}
```

强制三个 zone 是有立场的，但和图对得上。如果你需要两区或四区，fork 出来一份——别加条件分支。带条件的 module 会变得没法读。

## VPC 和子网的配置

`main.tf` 文件的第一部分如下：

```hcl
resource "alicloud_vpc" "this" {
  vpc_name   = var.name
  cidr_block = var.cidr_block
  tags       = var.tags
}

resource "alicloud_vswitch" "public" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id     = alicloud_vpc.this.id
  zone_id    = each.value
  cidr_block = cidrsubnet(var.cidr_block, 12, each.key)        # /28，从 .0 开始
  vswitch_name = "${var.name}-public-${substr(each.value, -1, 1)}"
  tags       = var.tags
}

resource "alicloud_vswitch" "private" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id     = alicloud_vpc.this.id
  zone_id    = each.value
  cidr_block = cidrsubnet(var.cidr_block, 8, each.key + 1)     # /24，从 .1.0 开始
  vswitch_name = "${var.name}-private-${substr(each.value, -1, 1)}"
  tags       = var.tags
}
```

这里有三个关键点需要注意：

- `cidrsubnet(prefix, newbits, netnum)` 是 Terraform 提供的一个用于 CIDR 计算的函数。例如，`cidrsubnet("10.20.0.0/16", 8, 1)` 会返回 `"10.20.1.0/24"`。这个函数在日常使用中非常常见，建议熟记。
- 使用 `for_each` 结合索引和值的映射可以确保资源地址的稳定性。比如，`alicloud_vswitch.private["0"]` 始终指向第一个可用区，即使你调整了列表顺序。相比之下，如果使用 `count`，重新排序会导致资源被大规模重建。
- `substr(each.value, -1, 1)` 的作用是从可用区 ID 中提取最后一个字符（通常是 `l`、`m` 或 `n`），这样可以让生成的资源名称更加整齐有序，便于后续管理和查看。
## NAT 网关 + EIP

```hcl
resource "alicloud_nat_gateway" "this" {
  vpc_id           = alicloud_vpc.this.id
  vswitch_id       = alicloud_vswitch.public["0"].id
  nat_gateway_name = "${var.name}-nat"
  nat_type         = "Enhanced"
  payment_type     = "PayAsYouGo"
  tags             = var.tags
}

resource "alicloud_eip_address" "nat" {
  address_name         = "${var.name}-nat-eip"
  bandwidth            = "100"
  internet_charge_type = "PayByTraffic"
  isp                  = "BGP"
  tags                 = var.tags
}

resource "alicloud_eip_association" "nat" {
  allocation_id = alicloud_eip_address.nat.id
  instance_id   = alicloud_nat_gateway.this.id
}

resource "alicloud_snat_entry" "private" {
  for_each = alicloud_vswitch.private

  snat_table_id     = alicloud_nat_gateway.this.snat_table_ids
  source_vswitch_id = each.value.id
  snat_ip           = alicloud_eip_address.nat.ip_address
}
```

Enhanced NAT 是现代版本——Tablestore、PrivateLink 和大多数新服务都需要它。PayByTraffic 适合 Agent 这种突发出网（LLM 流式）而不是稳态出网的场景。

SNAT 条目才是真正让私网实例出得了网的东西。没有它，`private-a` 里的 Agent 连 `dashscope.aliyuncs.com` 都解析不到。

## 安全组按 tier 分层

![用 Terraform 给 AI Agent 上云（三）：可复用的 VPC 与安全基线 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/illustration_2.jpg)

阿里云上正确的安全组用法是 **每个 tier 一个 SG**，规则引用 SG ID 而不是 CIDR：

![安全组策略——紧入站、宽出站、按 tier 分层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/fig2_sg_layers.png)

```hcl
resource "alicloud_security_group" "alb_public" {
  name   = "${var.name}-alb-public"
  vpc_id = alicloud_vpc.this.id
  tags   = var.tags
}

resource "alicloud_security_group_rule" "alb_https_in" {
  security_group_id = alicloud_security_group.alb_public.id
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "443/443"
  cidr_ip           = "0.0.0.0/0"
  policy            = "accept"
  priority          = 1
}

resource "alicloud_security_group" "agent_runtime" {
  name   = "${var.name}-agent-runtime"
  vpc_id = alicloud_vpc.this.id
  tags   = var.tags
}

resource "alicloud_security_group_rule" "agent_from_alb" {
  security_group_id        = alicloud_security_group.agent_runtime.id
  type                     = "ingress"
  ip_protocol              = "tcp"
  port_range               = "8080/8080"
  source_security_group_id = alicloud_security_group.alb_public.id
  policy                   = "accept"
  priority                 = 1
}
```

关键是 `source_security_group_id = alicloud_security_group.alb_public.id` 这一行。它说"只接受来自 ALB SG 里任何实例的入站 8080"——而不是某个 CIDR。后面给 ALB 换 IP 也不会破任何东西。

> **实操提示：** 阿里云安全组的默认行为是入站全拒、出站全允。这个默认是对的——别加"出站全拒"规则，你只会把 SDK 调用搞挂。除非有明确合规要求，否则不要限制出站；Agent 系统全开出站是常态。

每个下游 tier 我都重复这个模式：

```hcl
resource "alicloud_security_group" "memory_rds" {
  name   = "${var.name}-memory-rds"
  vpc_id = alicloud_vpc.this.id
  tags   = var.tags
}

resource "alicloud_security_group_rule" "rds_from_agent" {
  security_group_id        = alicloud_security_group.memory_rds.id
  type                     = "ingress"
  ip_protocol              = "tcp"
  port_range               = "5432/5432"
  source_security_group_id = alicloud_security_group.agent_runtime.id
  policy                   = "accept"
  priority                 = 1
}

resource "alicloud_security_group" "vector_store" {
  name   = "${var.name}-vector-store"
  vpc_id = alicloud_vpc.this.id
  tags   = var.tags
}

resource "alicloud_security_group_rule" "vector_from_agent" {
  security_group_id        = alicloud_security_group.vector_store.id
  type                     = "ingress"
  ip_protocol              = "tcp"
  port_range               = "9200/9200"
  source_security_group_id = alicloud_security_group.agent_runtime.id
  policy                   = "accept"
  priority                 = 1
}
```

写完之后给一台 ECS 挂正确的 SG 就只要一行 `security_groups = [module.vpc.agent_runtime_sg_id]`，网络层结构性正确。

## 每个数据域一把 KMS

任何说得过去的合规框架都强制要求静态加密。阿里云的方式是 **每个数据域一把客户主密钥（CMK）**，这样轮转一把不影响另一把，访问也能按 key 审计。

![KMS 加密——每个数据域一把 CMK](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/fig3_kms_encrypt.png)

```hcl
locals {
  cmks = {
    memory  = "RDS 数据和 OSS 对象的加密"
    secrets = "KMS Secrets Manager 条目加密"
    logs    = "SLS 日志数据加密"
  }
}

resource "alicloud_kms_key" "this" {
  for_each = local.cmks

  description            = each.value
  key_usage              = "ENCRYPT/DECRYPT"
  key_spec               = "Aliyun_AES_256"
  pending_window_in_days = 7
  status                 = "Enabled"
  automatic_rotation     = "Enabled"
  rotation_interval      = "365d"
  protection_level       = "SOFTWARE"
  tags                   = merge(var.tags, { Domain = each.key })
}

resource "alicloud_kms_alias" "this" {
  for_each = local.cmks

  alias_name = "alias/${var.name}-${each.key}"
  key_id     = alicloud_kms_key.this[each.key].id
}
```

为什么用别名？因为 CMK ID 是个没人记得住的 UUID；别名 `alias/agents-prod-memory` 人类可读，且在 key 轮转时保持稳定。从 RDS、OSS 等地方引用别名，你可以换底层 key 而下游配置不变。

`pending_window_in_days = 7` 意味着删除的 key 有 7 天恢复窗口。别缩短——误删 key 是那种能终结职业生涯的错误。

## Module 输出

`outputs.tf`：

```hcl
output "vpc_id" {
  value = alicloud_vpc.this.id
}

output "private_vswitch_ids" {
  value = [for s in alicloud_vswitch.private : s.id]
}

output "public_vswitch_ids" {
  value = [for s in alicloud_vswitch.public : s.id]
}

output "nat_gateway_id" {
  value = alicloud_nat_gateway.this.id
}

output "nat_eip_address" {
  value = alicloud_eip_address.nat.ip_address
}

output "alb_public_sg_id" {
  value = alicloud_security_group.alb_public.id
}

output "agent_runtime_sg_id" {
  value = alicloud_security_group.agent_runtime.id
}

output "memory_rds_sg_id" {
  value = alicloud_security_group.memory_rds.id
}

output "vector_store_sg_id" {
  value = alicloud_security_group.vector_store.id
}

output "kms_keys" {
  value = { for k, v in alicloud_kms_key.this : k => v.id }
}

output "kms_aliases" {
  value = { for k, v in alicloud_kms_alias.this : k => v.alias_name }
}
```

这些刚好就是后面五篇会用到的 ID。把输出有意识地命名和成形之后，调用方可以这样写：

```hcl
module "vpc" {
  source = "./modules/vpc-baseline"
  name   = "agents-prod"
  zones  = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]
}

resource "alicloud_instance" "agent" {
  vswitch_id      = module.vpc.private_vswitch_ids[0]
  security_groups = [module.vpc.agent_runtime_sg_id]
  # ...
}
```

## 调用 module

顶层 `main.tf`：

```hcl
module "vpc" {
  source = "./modules/vpc-baseline"

  name       = "agents-${terraform.workspace}"
  cidr_block = "10.20.0.0/16"
  zones      = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]

  tags = {
    Project     = "research-agent-stack"
    Environment = terraform.workspace
    ManagedBy   = "terraform"
  }
}
```

从项目根目录跑 `terraform plan` 会输出大致：

```
Plan: 27 to add, 0 to change, 0 to destroy.

Changes to Outputs:
  + agent_runtime_sg_id = (known after apply)
  + nat_eip_address     = (known after apply)
  + private_vswitch_ids = [
      + (known after apply),
      + (known after apply),
      + (known after apply),
    ]
  + vpc_id              = (known after apply)
```

27 个资源差不多对（1 VPC + 6 vSwitch + 1 NAT + 1 EIP + 1 EIP-assoc + 3 SNAT + 4 SG + 4 SG-rule + 3 KMS key + 3 KMS alias = 27）。Apply，约 90 秒，你拿到一副生产级别的网络。

## 成本估算

在 `cn-shanghai` 区域，按月粗略计算：

- VPC、vSwitch、安全组以及 KMS 密钥：完全免费
- NAT 网关：Enhanced 类型每月约 ¥120，外加按 GB 计费的出站流量费用
- EIP（弹性公网 IP）：每月约 ¥20 的 IP 保留费用，再加上按流量计费的实际使用成本
- KMS（密钥管理服务）：每个密钥每天前 100 次调用免费，超出部分每次约 ¥0.005

综合来看，在中低流量的情况下，网络基础架构的成本大约在 ¥150 到 ¥300 每月。考虑到所获得的功能和灵活性，这个价格非常划算——后续的所有内容都会基于这一基础框架展开。
## 接下来的内容

第四篇文章将计算资源部署到这张网络上。我们会探讨三种模式：使用 `pm2` 的 ECS、用于生产环境的 ACK 集群，以及基于事件驱动的 Function Compute Agent。同时，我还会介绍一个成本交叉模型，帮助你在这些选项之间做出选择。最后，我们会编写一个真实的 `alicloud_instance` 配置块，通过 cloud-init 自动化初始化 Python、Node.js 以及 Agent 运行时环境。

> **实战建议：** 如果未来需要增加第四个可用区（阿里云会不定期扩展可用区），只需执行一次 `terraform apply` 即可——`for_each` 模式能够优雅地处理更长的列表。不过需要注意的是，`variables.tf` 文件中的 `validation` 校验块会拒绝这一变更，因此你需要先放宽校验规则。这种有意为之的“阻力”正是关键所在——增加可用区是一项值得深思的网络变更，不应轻率行事。
## 配置漂移检测：当线上 VPC 与 HCL 不再匹配时

网络配置会随着时间发生漂移。比如，有人半夜十一点在控制台开了个端口调试问题；有人加了条 SNAT 规则测试临时方案；路由表里多了个没人清理的临时条目。半年后，生产环境的 VPC 和 HCL 文件可能已经悄然分道扬镳——直到下一次运行 `terraform apply` 时，Terraform 要么把手工改动覆盖回去（搞坏依赖这些改动的系统），要么更糟，因为差异让 provider 的更新逻辑混乱，直接重建资源。

解决办法是**尽早发现漂移**，并把它当作一个重要信号来处理。我在每个 VPC 栈中都会用到以下三种模式：

### 模式 1：CI 中每晚运行 `terraform plan`

通过 GitHub Actions 工作流，每天凌晨三点对每个 workspace 执行 `terraform plan -lock=false -detailed-exitcode`，如果退出码为 `2`（表示“计划中有变更”），就发送钉钉通知：

```yaml
# .github/workflows/drift-check.yml
name: drift-check
on:
  schedule:
    - cron: '0 19 * * *'   # 北京时间凌晨 3 点
  workflow_dispatch:

jobs:
  plan:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        workspace: [dev, staging, prod]
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.9.7
      - name: terraform init
        run: terraform init -input=false
        env:
          ALICLOUD_ACCESS_KEY: ${{ secrets.ALICLOUD_AK }}
          ALICLOUD_SECRET_KEY: ${{ secrets.ALICLOUD_SK }}
      - name: terraform plan
        id: plan
        continue-on-error: true
        run: |
          terraform workspace select ${{ matrix.workspace }}
          terraform plan -lock=false -detailed-exitcode -no-color > plan.txt
        env:
          ALICLOUD_REGION: cn-shanghai
      - name: notify drift
        if: steps.plan.outcome == 'failure' && steps.plan.outputs.exitcode == '2'
        run: |
          curl -X POST ${{ secrets.DINGTALK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"DRIFT: ${{ matrix.workspace }} - $(head -50 plan.txt)\"}}"
```

这里的 `-detailed-exitcode` 参数是关键。如果不加它，即使有变更，`plan` 也会返回 0；加上后，返回值会明确区分三种情况：0（无变更）、1（错误）、2（有待应用的变更）。CI 关注的是 2，因为它意味着发生了漂移。

我每晚都会对所有生产环境的 workspace 运行这套流程。大概每两周会抓到一次问题——通常是某个同事的“快速修复”，本该写进 HCL，但忘了。

### 模式 2：怀疑某资源漂移时单独刷新

如果你怀疑某个资源被手动修改过，`terraform refresh` 是一个精准的工具。它会从 API 重新读取资源状态并更新 Terraform 的状态文件，但不会应用 HCL 中的变更：

```bash
terraform apply -refresh-only
# Terraform has detected the following changes made outside of Terraform since
# the last "terraform apply":
#   ~ resource "alicloud_security_group_rule" "agent_from_alb" {
#       port_range = "8080/8080" -> "8080/8090"   # 有人放宽了端口范围
#     }
```

这个命令让你看到当前的实际状态，而不会做任何改动。根据显示的差异，你可以决定：是还原（普通 `apply`）还是将改动编码到 HCL 中以匹配现实。

### 模式 3：使用 `lifecycle { ignore_changes }` 忽略合法漂移

有时候，漂移是**合理的**。例如，阿里云可能会给资源添加元数据标签（如 `created_by_console`），或者 Auto Scaling 在 Terraform 的管理范围外调整了 `desired_capacity`。这种情况下，正确的做法是告诉 Terraform：“这些属性会发生漂移，没关系，不用管它们”：

```hcl
resource "alicloud_security_group" "agent_runtime" {
  name   = "${var.name}-agent-runtime"
  vpc_id = alicloud_vpc.this.id
  tags   = var.tags

  lifecycle {
    ignore_changes = [
      tags["created_by_console"],
      tags["last_audited_by"],
    ]
  }
}
```

`ignore_changes` 是一个精密的工具。不要直接忽略整个 `tags` 映射表——这样会掩盖真正的漂移。只忽略那些你明确知道由外部管理的特定键即可。
## 模块版本管理：当 VPC 模块逐步演进

本文中的 `vpc-baseline` 当前是 v1 版本。十八个月后，它可能会发展到 v4。届时，新的可用区会出现，NAT 网关的默认类型可能发生变化，你也可能会发现 `/28` 的公网子网在添加面向互联网的 NLB 后显得太小了。

错误的做法是“直接修改模块代码，然后在所有环境中运行 `terraform apply`”。某天下午，你将公网子网从 `/28` 改为 `/27`，结果现有子网需要重新创建，进而导致三个环境中的 NAT 网关和 EIP 被级联销毁。

正确的做法是采用**版本化的模块**，并制定明确的升级路径：

```hcl
# 像引用库一样固定到具体版本
module "vpc" {
  source  = "git::ssh://git@github.com/your-org/terraform-modules.git//vpc-baseline?ref=v1.4.0"

  name       = "agents-${terraform.workspace}"
  cidr_block = "10.20.0.0/16"
  zones      = local.zones
}
```

每当模块发生破坏性变更时，主版本号（遵循语义化版本 semver）都会递增。使用者需要有计划地进行升级，每次只针对一个 workspace，并仔细审查 `terraform plan` 的输出。如果在 `dev` 环境中发现问题，不要继续推送到 `prod`——而是回滚 `?ref=` 标签，提交问题单，修复模块。

对于小型团队，我通常会将所有模块集中存放在一个代码仓库中，使用 git tag 来管理版本号。而对于大型组织，则可以利用 Terraform Registry 的私有模块支持（或者阿里云的等效功能），通过 UI 发布模块。无论哪种方式，核心原则都是一样的：**模块是库，而不是代码片段**。它们应该像 Python 包一样，遵循严格的发布流程。

实际操作步骤如下：

```bash
# 升级某个 workspace
cd envs/dev
sed -i 's|?ref=v1.3.0|?ref=v1.4.0|' main.tf
terraform init -upgrade
terraform plan
# 仔细检查，尤其是涉及资源销毁的部分
terraform apply
```

如果 `dev` 环境在一周内运行正常，接下来可以在 `staging` 环境重复相同的操作，最后再到 `prod`。整个升级过程通过 PR 进行管理，每个 PR 都很小，且可以通过回滚提交轻松撤销。

> **实战建议：** 当破坏性模块变更需要重建资源时（例如 `/28` → `/27` 子网扩容），可以结合 `moved` 块和手动数据迁移来处理。`moved` 块的作用是告诉 Terraform：“这个旧子网的身份现在对应到这个新子网”；而数据迁移则负责将状态信息搬运过去。对于 VPC 子网这种特殊情况，更推荐的做法是**在现有子网旁边新增子网**，然后按可用区逐步迁移工作负载——切记，永远不要销毁包含活跃 ECS 实例的生产子网。
## 网络基线成本的计算逻辑

文章提到“¥150-300/月”，这个数字是准确的，但为了更好地根据实际流量估算成本，我们需要拆解一下。

### 固定成本（即使没有流量也需要支付）

| 项目                       | 每月费用（cn-shanghai） | 备注 |
|----------------------------|-----------------------:|------|
| VPC + vSwitch + 路由表     | ¥0                     | 免费使用，无规模限制 |
| 安全组                     | ¥0                     | 免费，单账号最多 100 个 |
| KMS 密钥（3 把，软件类型） | ¥9                     | 每把 CMK ¥3/月 |
| EIP 预留费用               | ¥18                    | 未绑定时 ¥0.6/天；绑定后免费持有 |
| NAT（增强型）预留费用      | ¥120                   | 增强型 NAT ¥4/天 |
| **固定成本总计**           | **~¥147/月**           |      |

### 可变成本

| 项目              | 单价                          | 示例 |
|-------------------|------------------------------:|------|
| EIP 出站流量      | BGP ¥0.8/GB，错峰 ¥0.3/GB   | Agent 每月出站 100 GB = ¥80 |
| KMS API 调用      | 免费额度后 ¥0.005/次          | 每月 10 万次调用 = ¥500 |
| NAT 跨区流量      | 同一 VPC 内免费               | 无费用 |

#### 示例场景
- **低流量开发环境**（10 GB 出站流量，1k KMS 调用）：¥147 + ¥8 + ¥0 ≈ **¥155/月**。
- **中等流量生产环境**（1 TB 出站流量，10 万 KMS 调用——重度 LLM 流式传输）：¥147 + ¥800 + ¥500 ≈ **¥1450/月**。

### 成本优化的关键：出站流量
如果您的应用需要从公网 LLM 端点流式获取长文本结果，建议优先使用 PrivateLink 或 VPC 对等连接到托管模型。PrivateLink 的流量费用约为 ¥0.1/GB，远低于公网出站的 ¥0.8/GB。以 DashScope 为例，其 PrivateLink 终端节点为 `com.aliyun.dashscope`。将其接入您的 VPC 后，出站流量账单可降低约 80%。

> **实战小贴士：**  
为每个资源添加 `Cost-Center` 和 `Owner` 标签。阿里云的账单仪表盘支持按标签进行费用透视，这样在季度末您无需联系财务部门，就能直接回答“某团队的网络成本为 ¥X”。本模块中的 `tags = var.tags` 配置正是为此设计的。
