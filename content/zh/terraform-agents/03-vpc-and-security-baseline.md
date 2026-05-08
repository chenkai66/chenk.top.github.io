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

这一篇造的是我所有 Agent 项目里被复制粘贴最多的一段 Terraform：一个 `vpc-baseline` module，给后续每一个组件（ECS、RDS、OpenSearch、ACK）一个合理的落点。总共大概 200 行 HCL，一次写完，永久参考。

读完之后你会拥有：

- 一个跨三可用区的 VPC
- 六个交换机（每区一个公网 + 一个私网），CIDR 不重叠
- 一个 NAT 网关 + EIP，让私网交换机能出网调 LLM API
- 三个按 tier 叠的安全组（ALB → agent runtime → memory）
- 三把 KMS 主密钥，每个数据域一把（memory、secrets、logs）
- 一个干净的 module 接口：`name + CIDR + zones` 进，一堆 ID 出
- CI 里的漂移检测、按 semver 锁版本的 module 引用、按行拆解的成本模型

## 心智模型

先看图，再看代码：

![VPC 拓扑——三可用区、公私网分离、NAT 出网](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/03-vpc-and-security-baseline/fig1_vpc_topology.png)

为什么三个可用区？因为阿里云保留任意周日对某个可用区做维护的权利，单可用区部署意味着维护窗口期间 Agent 全离线。VPC 内部跨可用区流量免费，三可用区唯一的代价是子网规划稍复杂——而这部分 Terraform 帮你抹平了。

为什么要分公网/私网？Agent 的运行环境必须放在私网交换机里，这样安全组配错也不会意外暴露到 `0.0.0.0/0`。公网交换机只放 ALB 和 NAT 网关——这些必须能上公网。Agent 走 NAT 出网，绝不直连。

我用的 CIDR 划分：

| 子网        | 可用区 | CIDR            | 主机数 |
|-------------|--------|-----------------|-------:|
| `public-a`  | l      | `10.20.0.0/28`  |     11 |
| `public-b`  | m      | `10.20.0.16/28` |     11 |
| `public-c`  | n      | `10.20.0.32/28` |     11 |
| `private-a` | l      | `10.20.1.0/24`  |    251 |
| `private-b` | m      | `10.20.2.0/24`  |    251 |
| `private-c` | n      | `10.20.3.0/24`  |    251 |

公网用 `/28`，因为只放 NAT 和 ALB 的 IP。私网用 `/24`，承载 Agent 的 ECS、RDS、OpenSearch。觉得 `/24` 紧？乘以三可用区是 753 个可用 IP——比我交付过的任何单 Agent 应用都富裕。

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

强制三个 zone 是有立场的，但和图对得上。如果你需要两区或四区，fork 一份——别加条件分支。带条件的 module 没法读，没法读的 module 半年之后会被人推倒重写。

## VPC 与交换机

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
  cidr_block   = cidrsubnet(var.cidr_block, 12, each.key)        # /28，从 .0 开始
  vswitch_name = "${var.name}-public-${substr(each.value, -1, 1)}"
  tags         = var.tags
}

resource "alicloud_vswitch" "private" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id       = alicloud_vpc.this.id
  zone_id      = each.value
  cidr_block   = cidrsubnet(var.cidr_block, 8, each.key + 1)     # /24，从 .1.0 开始
  vswitch_name = "${var.name}-private-${substr(each.value, -1, 1)}"
  tags         = var.tags
}
```

三个细节：

- `cidrsubnet(prefix, newbits, netnum)` 是 Terraform 的 CIDR 计算函数。`cidrsubnet("10.20.0.0/16", 8, 1)` 返回 `"10.20.1.0/24"`，背下来，每天都用。
- `for_each` 配 index/value 映射保证资源地址稳定——`alicloud_vswitch.private["0"]` 永远指向第一个可用区，即使你调整列表顺序。换成 `count`，调顺序会触发资源整体重建。
- `substr(each.value, -1, 1)` 截取可用区 ID 的最后一位（`l`/`m`/`n`），让控制台里的资源名按顺序排得整齐。

## NAT 网关与 EIP

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

Enhanced NAT 是现代版本，Tablestore、PrivateLink 以及大多数新服务都需要它。老的 Standard NAT 已在弃用路径上，新项目别用。PayByTraffic 适合 Agent 这种突发出网（LLM 流式）而非稳态出网的场景。

SNAT 条目才是真正让私网实例出得了网的东西。没有它，`private-a` 里的 Agent 连 `dashscope.aliyuncs.com` 都解析不到——第一次踩这个坑能让你 debug 一小时，我有伤疤。

## 安全组按 tier 分层

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

关键是 `source_security_group_id = alicloud_security_group.alb_public.id` 这一行：「只接受来自 ALB SG 里任何实例的入站 8080」，而不是某个 CIDR。后面给 ALB 换 IP 也不会破任何东西。

> 实操提示：阿里云安全组的默认行为是入站全拒、出站全允。这个默认是对的——别加「出站全拒」规则，你只会把 SDK 调用搞挂。除非有明确合规要求，否则不要限制出站；Agent 系统全开出站是常态。

下游每个 tier 我都重复这个模式：

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

写完之后给一台 ECS 挂正确的 SG 就只要一行 `security_groups = [module.vpc.agent_runtime_sg_id]`，网络层结构性正确。审计也变得很简单：grep SG 名字就能找到所有挂过它的资源。

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

为什么用别名？因为 CMK ID 是个没人记得住的 UUID；别名 `alias/agents-prod-memory` 人类可读，且在 key 轮转时保持稳定。从 RDS、OSS、SLS 引用别名，换底层 key 时下游配置不用动。

`pending_window_in_days = 7` 意味着删除的 key 有 7 天恢复窗口。别缩短——误删 key 是那种能终结职业生涯的错误，这个窗口救过我不止一次。

## Module 输出

`outputs.tf`：

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

这些刚好是后面五篇会用到的 ID。把输出有意识地命名和成形之后，调用方可以这样写：

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

从项目根目录跑 `terraform plan`，输出大致：

```
Plan: 27 to add, 0 to change, 0 to destroy.
```

27 个资源差不多对（1 VPC + 6 vSwitch + 1 NAT + 1 EIP + 1 EIP-assoc + 3 SNAT + 4 SG + 4 SG-rule + 3 KMS key + 3 KMS alias = 27）。Apply 一下，约 90 秒后你拿到一副生产级别的网络。

## 漂移检测：当线上 VPC 与 HCL 不再匹配

网络是会漂移的。有人半夜十一点在控制台开了个端口调试问题；有人加了条 SNAT 规则测临时方案；路由表里多了条没人清理的临时条目。半年后，生产 VPC 和 HCL 已经悄然分道扬镳——直到下一次 `terraform apply` 把手工改动覆盖回去（搞坏依赖这些改动的系统），或者更糟，差异让 provider 的更新逻辑混乱直接重建资源。

解决办法是尽早发现漂移，并把它当成真信号处理。我在每个 VPC stack 上都跑这三种模式：

### 模式 1：CI 里每晚跑 `terraform plan`

GitHub Actions workflow，每天北京时间凌晨三点对每个 workspace 跑 `terraform plan -lock=false -detailed-exitcode`，退出码为 `2`（「计划中有变更」）就发钉钉：

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

`-detailed-exitcode` 是关键。不加它，即使有变更 `plan` 也只会返回 0；加上之后返回 0（无变更）、1（错误）或 2（有待 apply 的变更）。CI 只关心 2——那是漂移。

每晚跑，每个 prod workspace 都跑。两周左右就会抓到一次——通常是同事的「快速修复」本该写进 HCL 但忘了。

### 模式 2：怀疑某资源漂移时单独刷新

怀疑某个资源被手动改过，`terraform apply -refresh-only` 是手术刀。它从 API 重新读取资源状态并更新 state，但不应用 HCL 中的变更：

```bash
terraform apply -refresh-only
# Terraform has detected the following changes made outside of Terraform since
# the last "terraform apply":
#   ~ resource "alicloud_security_group_rule" "agent_from_alb" {
#       port_range = "8080/8080" -> "8080/8090"   # 有人放宽了端口
#     }
```

看到 diff 之后再决定：还原（普通 `apply`）还是把现实 codify 到 HCL（编辑代码匹配现状）。

### 模式 3：用 `lifecycle { ignore_changes }` 给合法漂移开口子

有时候漂移是合理的。阿里云会自动给资源加元数据标签（如 `created_by_console`）；Auto Scaling 会在 Terraform 管辖外调整 `desired_capacity`。这种情况下正确的做法是告诉 Terraform：「这个属性会漂移，没事」：

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

`ignore_changes` 是精密工具。别把整个 `tags` map 塞进去——那样会把真漂移也屏蔽掉。只列你明确知道由外部管理的具体 key。

## Module 版本管理：把它当库来对待

本文这版 `vpc-baseline` 是 v1。十八个月后它会到 v4。新可用区会出现，NAT 默认类型可能变，加了面向公网的 NLB 之后你会发现 `/28` 公网子网太小。

错误做法是「直接改 module 然后到处 `terraform apply`」。某个周五下午你把公网子网从 `/28` 改成 `/27`，现有子网必须重建，三个环境的 NAT 和 EIP 被级联销毁。（亲身踩的坑。）

正确做法是版本化的 module + 明确的升级路径：

```hcl
module "vpc" {
  source = "git::ssh://git@github.com/your-org/terraform-modules.git//vpc-baseline?ref=v1.4.0"

  name       = "agents-${terraform.workspace}"
  cidr_block = "10.20.0.0/16"
  zones      = local.zones
}
```

每次破坏性变更都按 semver 升主版本号。使用方有计划地升级，一次只动一个 workspace，每次都仔细看 `plan`。`dev` 炸了不要往 `prod` 推——回滚 `?ref=` 标签，开 issue，修 module。

小团队我用单 repo 存 module，git tag 当版本号。大团队可以用 Terraform Registry 的私有 module 支持（或阿里云的对应物），通过 UI 发布。无论哪种方式，原则一样：**module 是库，不是代码片段**，要按 Python 包的发版纪律来管。

实操步骤：

```bash
cd envs/dev
sed -i 's|?ref=v1.3.0|?ref=v1.4.0|' main.tf
terraform init -upgrade
terraform plan          # 仔细看，特别是 destroy 的部分
terraform apply
```

`dev` 撑过一周就推 `staging`，再推 `prod`。整个 rollout 用 PR 推进，每个 PR 都小，回滚一个 commit 就能撤回。

> 实操提示：破坏性变更需要资源重建时（比如 `/28` → `/27` 子网扩容），用 `moved` 块 + 手动数据迁移。`moved` 告诉 Terraform：「旧子网的身份等同于这个新子网」；数据迁移搬运 state。VPC 子网这种情况，更稳的做法是在旁边新增子网，按可用区逐步迁移工作负载——绝不销毁还有活跃 ECS 的生产子网。

## 网络基线的成本拆解

`cn-shanghai`，按月粗算，中低流量下网络基线大约 ¥150-300/月。这个数字真实，但拆开来看才能按你自己的流量套尺。

固定成本（零流量也得付）：

| 项目                       | 月费用（cn-shanghai） | 备注 |
|----------------------------|----------------------:|------|
| VPC + vSwitch + 路由表     | ¥0                    | 任何规模都免费 |
| 安全组                     | ¥0                    | 免费，单账号最多 100 个 |
| KMS 密钥（3 把，软件类型） | ¥9                    | 每把 CMK ¥3/月 |
| EIP 预留费                 | ¥18                   | 未绑定时 ¥0.6/天；绑定后免费持有 |
| NAT（增强型）预留费        | ¥120                  | 增强型 NAT ¥4/天 |
| 固定总计                   | **~¥147/月**          |      |

可变成本：

| 项目         | 单价                          | 示例 |
|--------------|------------------------------:|------|
| EIP 出站流量 | BGP ¥0.8/GB，错峰 ¥0.3/GB     | 每月出 100 GB = ¥80 |
| KMS API 调用 | 免费额度后 ¥0.005/次          | 每月 10 万次 = ¥500 |
| NAT 跨可用区 | 同 VPC 内免费                 | 0 |

低流量 dev workspace（10 GB 出站，1k KMS 调用）：¥147 + ¥8 + ¥0 ≈ **¥155/月**。

中等流量 prod workspace（1 TB 出站，10 万 KMS 调用——重度 LLM 流式）：¥147 + ¥800 + ¥500 ≈ **¥1,450/月**。

杠杆在出站流量。如果你的 Agent 从公网 LLM 端点流式拉长结果，能用 PrivateLink 或 VPC 对等就用——PrivateLink 流量约 ¥0.1/GB，是公网 ¥0.8/GB 的八分之一。DashScope 的 PrivateLink 终端节点是 `com.aliyun.dashscope`，接进 VPC 之后出站账单大约能降 80%。

> 实操提示：每个资源都打 `Cost-Center` 和 `Owner` 标签。阿里云账单仪表盘按标签透视，季度末你不用问财务就能回答「这个团队的网络成本是 ¥X」。本 module 里的 `tags = var.tags` 管线就是为这个准备的。

## 接下来

第四篇会把计算资源落到这张网络上。三种模式——`pm2` + ECS、生产用 ACK、事件驱动用 Function Compute——以及我用来在它们之间做选择的成本交叉模型。然后给一段真实的 `alicloud_instance` 配置块，通过 cloud-init 装 Python + Node + Agent runtime。

> 实操提示：将来要加第四个可用区（阿里云会不定期加），跑一次 `terraform apply` 就行——`for_each` 模式能优雅吃下更长的列表。但 `variables.tf` 里的 `validation` 会先拒绝它，你得先放宽校验。这种刻意的摩擦正是重点——加可用区是值得想清楚的网络变更，不该是手抖捎带进去的。
