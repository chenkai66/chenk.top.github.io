---
title: "Terraform 实战（三）：复用 VPC 与安全基线"
date: 2026-03-16 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - VPC
  - Security
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 3
series_total: 8
description: "第一个可复用 module——三可用区 VPC，公私网交换机分层，NAT 出网，按 tier 分层的安全组，再加上按数据域分的 KMS 主密钥。同样的代码出现在我交付过的每一个 Agent stack 里，参数化但本体不变。"
disableNunjucks: true
translationKey: "terraform-agents-3"
---
今天要写的是我 Agent 项目里被复制次数最多的 Terraform 代码：一个 `vpc-baseline` 模块。它为后续所有组件（ECS、RDS、OpenSearch、ACK）提供了统一、可复用的网络基础。总共约 200 行 HCL，建议亲手编写一遍，方便后续复用和定制。

读完这篇，你将获得：

- 一个跨三个可用区的 VPC（单 Region）
- 六个 vSwitch（每区一个公网 + 一个私网），CIDR 互不重叠
- 带 EIP 的 NAT Gateway，供私网子网 outbound 访问 LLM API
- 三层安全组堆叠（ALB → agent runtime → memory）
- 三个 KMS customer master keys，每个数据域一个（memory, secrets, logs）
- 干净的模块接口：输入 `name + CIDR + zones`，输出 IDs
- CI 里的 drift detection、semver 锁定的模块引用，以及按行计算的成本模型

---

## 心智模型

先别急着写代码，先看图：

![VPC 拓扑 — 3 个可用区，公有 + 私有，NAT 出口](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig1_vpc_topology.png)

为什么选三个可用区？因为阿里云可能在任意周日发起可用区级维护，单可用区部署会导致整个维护窗口期内 Agent 全面不可用。而 VPC 内跨可用区流量免费，唯一的额外成本是子网规划的运维复杂度，这部分已由 Terraform 自动处理。

为什么要分公网和私网？Agent runtime 应该待在私网 vSwitch 里，这样就算安全组配错，也不会意外把服务暴露在 `0.0.0.0/0` 上。公网 vSwitch 留给 ALB 和 NAT Gateway——这些是*必须*能通互联网的设备。Agent 通过 NAT 上网，绝不直连。

我的 CIDR 布局如下：

| 子网 | 可用区 | CIDR | 主机数 |
|-------------|------|------------------|------:|
| `public-a` | l | `10.20.0.0/28` | 11 |
| `public-b` | m | `10.20.0.16/28` | 11 |
| `public-c` | n | `10.20.0.32/28` | 11 |
| `private-a` | l | `10.20.1.0/24` | 251 |
| `private-b` | m | `10.20.2.0/24` | 251 |
| `private-c` | n | `10.20.3.0/24` | 251 |

公网用 `/28` 是因为里面只放 NAT 和 ALB IP。私网用 `/24` 是因为 Agent ECS、RDS、OpenSearch 节点都住这儿。如果你觉得 `/24` 对 Agent 舰队来说太紧，乘以三个可用区——753 个可用 IP 比我交付过的任何单个 Agent 应用都多。

## 模块骨架

创建目录结构：

```text
modules/vpc-baseline/
├── main.tf
├── variables.tf
├── outputs.tf
└── versions.tf
```

输入参数（`variables.tf`）：

```hcl
variable "name" {
  description = "Name prefix for all resources, e.g. agents-prod"
  type        = string
}

variable "cidr_block" {
  description = "Top-level VPC CIDR; subnets are derived from this"
  type        = string
  default     = "10.20.0.0/16"
}

variable "zones" {
  description = "Three availability zone IDs in the target region"
  type        = list(string)
  validation {
    condition     = length(var.zones) == 3
    error_message = "vpc-baseline requires exactly 3 zones."
  }
}

variable "tags" {
  description = "Tags applied to every resource the module creates"
  type        = map(string)
  default     = {}
}
```

强制三可用区虽然属于强约定，但与架构图严格对齐。如需双可用区或四可用区，直接 fork 模块——切勿引入条件逻辑。含条件分支的模块会损害可读性，半年内必被推倒重写。

## VPC 和 vSwitches

`main.tf` 第一部分：

```hcl
resource "alicloud_vpc" "this" {
  vpc_name   = var.name
  cidr_block = var.cidr_block
  tags       = var.tags
}

resource "alicloud_vswitch" "public" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id       = alicloud_vpc.this.id
  zone_id      = each.value
  cidr_block   = cidrsubnet(var.cidr_block, 12, each.key)        # /28 starting at .0
  vswitch_name = "${var.name}-public-${substr(each.value, -1, 1)}"
  tags         = var.tags
}

resource "alicloud_vswitch" "private" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id       = alicloud_vpc.this.id
  zone_id      = each.value
  cidr_block   = cidrsubnet(var.cidr_block, 8, each.key + 1)     # /24 starting at .1.0
  vswitch_name = "${var.name}-private-${substr(each.value, -1, 1)}"
  tags         = var.tags
}
```

这里有三个细节：

- `cidrsubnet(prefix, newbits, netnum)` 是 Terraform 的 CIDR 数学工具。`cidrsubnet("10.20.0.0/16", 8, 1)` 返回 `"10.20.1.0/24"`。记牢它，你会经常用到。
- 配合 index/value map 使用 `for_each` 能获得稳定的资源地址——`alicloud_vswitch.private["0"]` 永远指向第一个可用区，哪怕你调整了列表顺序。对比一下 `count`，重排顺序会导致大规模重建。
- `substr(each.value, -1, 1)` 提取可用区 ID 的最后一个字符（`l`/`m`/`n`），这样资源名在控制台里排序更清晰。

## NAT Gateway 和 EIP

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

Enhanced NAT 是当前标准，Tablestore、PrivateLink 及绝大多数新服务均强制要求；老式 Standard NAT 已进入 deprecation 倒计时，新项目严禁使用。按流量计费（PayByTraffic）更适合 Agent 负载，因为其出站带宽具有突发性（如 LLM 流式响应），而不是持续稳定。

SNAT 条目才是让私网子网实例能通互联网的关键。少了它们，`private-a` 里的 Agent 解析不了 `dashscope.aliyuncs.com`——第一次遇到这问题你会花一个小时调试。这个问题我亲身遇到过，调试花了一小时。

## 安全组，分层设计

在阿里云上，正确的安全组做法是每层一个 SG，规则引用 SG ID 而不是 CIDR：

![多层网络安全架构与防火墙屏障](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/wanxiang_network_layers.png)

![安全组策略 — 严格的入站，宽松的出站，分层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig2_sg_layers.png)

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

关键就在这行 `source_security_group_id = alicloud_security_group.alb_public.id`。它的意思是“只接受来自 ALB SG 内任何实例的 8080 入站”——而不是来自某个 CIDR。以后 ALB 换 IP 也不会挂。

> **实战建议：** 阿里云的默认行为是*拒绝*所有入站，*允许*所有出站。该默认策略合理——无需额外添加‘拒绝所有出站’规则，否则可能导致 SDK 调用失败。除非你有特定的合规要求，否则限制出站流量没必要；对 Agent 系统来说，出站全开是常态。

我将这一模式扩展到每一个下游层：

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

做完这些，把 ECS 挂到正确的 SG 上只需要 `security_groups = [module.vpc.agent_runtime_sg_id]`，网络层级天生就是对的。审计变得很简单：grep 一下安全组名字，就能找到所有用过它的资源。

## 每个数据域配一把 KMS 密钥

静态加密（Encryption-at-rest）是任何合规制度的底线，阿里云的做法是每个数据域使用一把 Customer Master Key (CMK)，这样可以单独轮换某一把而不影响其他，并且可以按密钥审计访问记录。

![数据在存储和传输过程中加密，并进行密钥管理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/wanxiang_encryption.png)


![KMS 加密 — 每个数据域一个 CMK](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig3_kms_encrypt.png)

```hcl
locals {
  cmks = {
    memory  = "Encryption for RDS data and OSS objects"
    secrets = "Encryption for KMS Secrets Manager entries"
    logs    = "Encryption for SLS log data"
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

为什么要用别名？因为 CMK ID 是一串没人记得住的 UUID；别名 `alias/agents-prod-memory` 是人类可读的，而且在密钥轮换期间保持稳定。从 RDS、OSS 和 SLS 引用别名，你就可以 swapping 底层密钥而不必动下游配置。

`pending_window_in_days = 7` 意味着删除的密钥有 7 天窗口期可以恢复。不建议缩短该时间——误删密钥可能引发严重生产事故，7 天恢复窗口已在实践中多次规避此类风险。

## 模块的输出

`outputs.tf`:

```hcl
output "vpc_id"              { value = alicloud_vpc.this.id }
output "private_vswitch_ids" { value = [for s in alicloud_vswitch.private : s.id] }
output "public_vswitch_ids"  { value = [for s in alicloud_vswitch.public  : s.id] }
output "nat_gateway_id"      { value = alicloud_nat_gateway.this.id }
output "nat_eip_address"     { value = alicloud_eip_address.nat.ip_address }
output "alb_public_sg_id"    { value = alicloud_security_group.alb_public.id }
output "agent_runtime_sg_id" { value = alicloud_security_group.agent_runtime.id }
output "memory_rds_sg_id"    { value = alicloud_security_group.memory_rds.id }
output "vector_store_sg_id"  { value = alicloud_security_group.vector_store.id }
output "kms_keys"    { value = { for k, v in alicloud_kms_key.this   : k => v.id } }
output "kms_aliases" { value = { for k, v in alicloud_kms_alias.this : k => v.alias_name } }
```

接下来五篇文章需要的 ID 都在这儿了。故意把输出的命名和结构设计好，调用方就能这样用：

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

## 调用模块

在你的顶层 `main.tf` 里：

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

在项目根目录跑 `terraform plan`，输出大概是这样：

```text
Plan: 27 to add, 0 to change, 0 to destroy.
```

27 个资源差不多正好（1 个 VPC + 6 个 vSwitch + 1 个 NAT + 1 个 EIP + 1 个 EIP 关联 + 3 个 SNAT + 4 个 SG + 4 个 SG 规则 + 3 个 KMS key + 3 个 KMS alias = 27）。执行 apply，大约 90 秒你就能得到一个生产级网络。

## 漂移检测：当线上 VPC 跟 HCL 对不上时

网络总会漂移的。半夜 11 点有人为了调试在控制台开了个端口。有人为了测试临时方案加了条 SNAT 规则。路由表里多了条临时条目，后来没人删。半年后生产环境 VPC 和 HCL 静默分叉——直到下次 `terraform apply` 要么回滚了手动变更（害了依赖它的人），要么更糟，把 provider 的更新逻辑搞混，导致资源重建。

解决办法是尽早发现漂移，把它当成真实信号来处理。每个 VPC 栈我都跑这三种模式：

### 模式 1：CI 里 nightly 跑 `terraform plan`

GitHub Actions workflow 在北京时间凌晨 3 点跑每个 workspace 的 `terraform plan -lock=false -detailed-exitcode`，如果退出码是 `2`（“plan 会有变更”）就发 DingTalk 通知：

```yaml
# .github/workflows/drift-check.yml
name: drift-check
on:
  schedule:
    - cron: '0 19 * * *'   # 3am Beijing
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

`-detailed-exitcode` 参数是关键。没它的话，哪怕有变更 `plan` 也总是返回 0。有了它，你才能拿到 0（无变更）、1（错误）或 2（待变更）。CI 只关心 2——这意味着漂移。

我对每个 prod workspace 都夜间运行这个。每两周总能抓到点什么——通常是队友忘了写进 HCL 的“快速修复”。

### 模式 2：怀疑有问题时跑 refresh-only

怀疑单个资源漂移时，`terraform apply -refresh-only` 是手术刀式的工具。它从 API 重读资源并更新 state，但不应用 HCL 变更：

```bash
terraform apply -refresh-only
# Terraform has detected the following changes made outside of Terraform since
# the last "terraform apply":
#   ~ resource "alicloud_security_group_rule" "agent_from_alb" {
#       port_range = "8080/8080" -> "8080/8090"   # someone widened it
#     }
```

看到 diff 后你再来决定：回滚（常规 `apply`）还是固化（编辑 HCL 去匹配现状）。

### 模式 3：`lifecycle { ignore_changes }` 逃生通道

有时候漂移是*合法*的。阿里云会给资源自动打元数据标签（比如 `created_by_console`）。Auto Scaling 会在 Terraform 管辖范围外调整 `desired_capacity`。正确的做法是告诉 Terraform“这个属性会漂移，没关系”：

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

`ignore_changes` 是精密仪器。别把 `tags`（整个 map）放进去——那样会掩盖真正的漂移。只放你知道由外部管理的具体 key。

## 模块版本管理：把它当库来对待

这篇文章的 `vpc-baseline` 是 v1 版。十八个月后它会变成 v4 版。会有新的可用区出现。默认 NAT 类型可能会变。你会发现加了面向公网的 NLB 后，`/28` 的公网子网太小了。

错误的做法是“原地编辑模块然后到处 `terraform apply`”。某个周五下午你把公网子网从 `/28` 改成 `/27`，现有子网需要重建，结果你在三个环境里级联销毁了 NAT 和 EIP。（没错，这也是我的个人血泪史。）

正确的做法是用 **版本化模块** 配合明确的升级路径：

```hcl
module "vpc" {
  source = "git::ssh://git@github.com/your-org/terraform-modules.git//vpc-baseline?ref=v1.4.0"

  name       = "agents-${terraform.workspace}"
  cidr_block = "10.20.0.0/16"
  zones      = local.zones
}
```

模块每次破坏性变更都升大版本（semver）。消费者 deliberate 升级，一次一个 workspace，每个都 review `plan`。当 `dev` 里出问题，别推到 `prod`——回滚 `?ref=` 标签，提 issue，修模块。

小团队我用单个 repo 存模块，用 git tag 做版本。大组织的话，Terraform Registry 的私有模块支持（或者阿里云托管的等价物）能提供带 UI 的发布产物。不管怎样原则一样：模块是库，不是代码片段。得像对待 Python 包一样对待它们的发布纪律。

升级的实际操作手感：

```bash
cd envs/dev
sed -i 's|?ref=v1.3.0|?ref=v1.4.0|' main.tf
terraform init -upgrade
terraform plan          # review carefully, especially the destroys
terraform apply
```

如果 `dev` 稳一周，再在 `staging` 重复。最后才是 `prod`。整个 rollout 都在 PR 里，每个都小，每个都能通过回滚 commit 撤销。

> **Real-world tip:** 当破坏性的模块变更需要重建资源时（比如我们的 `/28` → `/27` 子网扩容），配合手动数据迁移使用 `moved` 块。`moved` 块告诉 Terraform“这个旧子网的身份现在是这个新的”；迁移则复制状态。具体到 VPC 子网，更简单的路径是*旁边*加新子网，然后逐可用区迁移工作负载——千万别销毁跑着线上 ECS 的生产子网。

## 网络基线的成本算账

大致在 `cn-shanghai` 区域，低到中等流量下基线成本每月 ¥150–300。这个数字是真实的，但值得拆开来看，方便你根据自己的流量 sizing。

固定成本（哪怕零流量也得付）：

| Item                       | Monthly (cn-shanghai) | Notes |
|----------------------------|----------------------:|-------|
| VPC + vSwitch + RT         | ¥0                    | free at any scale |
| 安全组 | ¥0 | 免费，每个账号最多 100 个安全组 |
| KMS keys (3, software)     | ¥9                    | ¥3/mo per CMK |
| EIP 预留 | ¥18 | 未绑定时¥0.6/天；已绑定的 EIP 免费持有 |
| NAT（增强型）预留 | ¥120 | 增强型 NAT ¥4/天 |
| **Fixed total**            | **~¥147/mo**          |       |

变动成本：

| Item                 | Unit price                   | Example |
|----------------------|-----------------------------:|---------|
| EIP 出站流量 | BGP ¥0.8/GB，高峰时段¥0.3/GB | 每月 100 GB 代理流量 = ¥80 |
| KMS API 调用 | 超出免费额度后¥0.005/次 | 每月 10 万次调用 = ¥500 |
| 同 VPC 内跨可用区 NAT | 免费 | 无费用 |

低流量 dev workspace（10 GB 出站，1k 次 KMS 调用）：¥147 + ¥8 + ¥0 ≈ **¥155/mo**。

中流量 prod workspace（1 TB 出站，100k 次 KMS 调用——重度 LLM streaming）：¥147 + ¥800 + ¥500 ≈ **¥1,450/mo**。

杠杆在于出站流量。如果你的 agent 从公网 LLM 端点 stream 长 completion，有条件的话要用 PrivateLink 或 VPC peering 连托管模型——PrivateLink 流量大概 ¥0.1/GB 而不是 ¥0.8/GB。对于 DashScope，PrivateLink 端点是 `com.aliyun.dashscope`；把它接进你的 VPC，出站账单能降 ~80%。

> **Real-world tip:** 给每个资源打上 `Cost-Center` 和 `Owner` 标签。阿里云的计费 dashboard 能按标签透视，季度末你能直接回答“这个团队的网络成本是 ¥X”，不用去求财务。这个模块里的 `tags = var.tags` plumbing 就是为此准备的。

## 接下来做什么

第四篇文章要把计算资源落在这个网络上。三种模式——带 `pm2` 的 ECS、生产集群用的 ACK、事件驱动 agent 用的 Function Compute——以及我用来做选择的成本交叉模型。然后是一个真实的 `alicloud_instance` 块，通过 cloud-init 引导 Python + Node + agent 运行时。

> **Real-world tip:** 如果需要加第四个可用区（阿里云会 periodically 加），一次 `terraform apply` 就能搞定——`for_each` 模式能干净地处理更长的列表。但 `variables.tf` 里的 `validation` 块会拒绝它，所以你得先放宽 validation。这种故意制造的摩擦正是重点——加可用区是值得思考的网络变更，不是随手滑进来的 typo。
