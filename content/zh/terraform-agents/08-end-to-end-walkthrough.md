---
title: "用 Terraform 给 AI Agent 上云（八）：端到端——一次 apply 起整个 research-agent-stack"
date: 2026-03-26 09:00:00
tags:
  - Terraform
  - 阿里云
  - 端到端
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 8
description: "把七个 module 拼到一个仓库，跑一次 terraform apply，看一个完整的 Agent runtime——VPC、ECS、RDS、OpenSearch、OSS、LLM 网关、SLS 观测、成本告警——七分钟内起来。真实 apply 输出、module DAG、可 fork 的起手仓库。"
disableNunjucks: true
translationKey: "terraform-agents-8"
---

这是第二到第七篇所有东西落到一处的文章。读完之后你会跑过一次 `terraform apply`，在阿里云上产出一个完整、可观测、有预算的 Agent runtime stack。约 31 个资源，~7 分钟实际时间。

我们要建的 stack：

![research-agent-stack：每一个盒子，一次 terraform apply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig1_full_stack.png)

五层——edge、compute、memory、platform、ops——由本系列做出的 module 组合而来。

![用 Terraform 给 AI Agent 上云（八）：端到端——一次 apply 起整个 research-agent-stack — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/illustration_1.jpg)

## 项目结构

```
research-agent-stack/
├── README.md
├── versions.tf                  # Terraform + provider 版本
├── backend.tf                   # OSS + Tablestore 远程 state
├── providers.tf                 # alicloud + alicloud.beijing 别名
├── variables.tf                 # 顶层输入
├── locals.tf                    # 看 workspace 计算的 local
├── main.tf                      # module 组合
├── outputs.tf                   # endpoint + 连接串
├── env/
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── prod.tfvars
├── secrets/
│   └── secrets.auto.tfvars      # gitignore——provider key
├── modules/
│   ├── vpc-baseline/            # 第三篇
│   ├── storage/                 # 第五篇
│   ├── compute/                 # 第四篇
│   ├── llm-gateway/             # 第六篇
│   └── observability/           # 第七篇
└── scripts/
    ├── cloud-init/
    │   ├── agent.sh
    │   └── gateway.sh
    └── restore-drill.sh
```

顶层八个 `*.tf`，`modules/` 下五个 module，`env/*.tfvars` 装环境特定值，`secrets/secrets.auto.tfvars` 装 git 之外的密钥。这是我每个项目都用的布局——无聊就是好。

## main.tf——组合

```hcl
locals {
  is_prod   = terraform.workspace == "prod"
  name      = "agents-${terraform.workspace}"
  zones     = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]

  common_tags = {
    Project     = "research-agent-stack"
    Environment = terraform.workspace
    ManagedBy   = "terraform"
    Owner       = "ai-platform"
  }
}

module "vpc" {
  source = "./modules/vpc-baseline"

  name       = local.name
  cidr_block = "10.20.0.0/16"
  zones      = local.zones
  tags       = local.common_tags
}

module "storage" {
  source = "./modules/storage"

  name              = local.name
  vpc               = module.vpc
  is_prod           = local.is_prod
  enable_dr         = local.is_prod   # 跨区 OSS 复制只在 prod 开
  tags              = local.common_tags

  providers = {
    alicloud         = alicloud
    alicloud.beijing = alicloud.beijing
  }
}

module "observability" {
  source = "./modules/observability"

  name             = local.name
  vpc              = module.vpc
  dingtalk_webhook = var.dingtalk_webhook
  cost_ceiling_cny = local.is_prod ? 800 : 100
  tags             = local.common_tags
}

module "gateway" {
  source = "./modules/llm-gateway"

  name           = local.name
  vpc            = module.vpc
  observability  = module.observability
  llm_keys       = var.llm_keys
  agent_quotas   = var.agent_quotas
  instance_count = local.is_prod ? 2 : 1
  tags           = local.common_tags
}

module "compute" {
  source = "./modules/compute"

  name           = local.name
  vpc            = module.vpc
  storage        = module.storage
  gateway        = module.gateway
  observability  = module.observability
  agent_repo_url = var.agent_repo_url
  agent_branch   = var.agent_branch
  ecs_count      = local.is_prod ? 3 : 1
  tags           = local.common_tags
}
```

五个 module 调用。注意每个 module 把*前一个* module 的输出当输入——`module.compute` 读 `module.vpc`、`module.storage`、`module.gateway`、`module.observability`。这种依赖接线就是 Terraform 用来构建 apply DAG 的方式：

![Terraform module 依赖 DAG](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig2_module_dag.png)

network 和 KMS 在最上——它们没有依赖。storage、compute、gateway 依赖 network + KMS 但相互独立，所以 Terraform 并行建。compute 还依赖 storage 和 gateway，因为 cloud-init 模板需要它们的 endpoint。observability 和 alarm 依赖 compute，因为它们引用了 SG ID。

## variables.tf

```hcl
variable "agent_repo_url" {
  description = "要部署的 Agent runtime 的 git URL"
  type        = string
  default     = "https://github.com/example/research-agent.git"
}

variable "agent_branch" {
  description = "要部署的 git 分支 / tag"
  type        = string
  default     = "main"
}

variable "dingtalk_webhook" {
  description = "告警钉钉 webhook URL"
  type        = string
  sensitive   = true
}

variable "llm_keys" {
  description = "provider 名 → API key 的 map，通过 secrets.auto.tfvars 设"
  type        = map(string)
  sensitive   = true
}

variable "agent_quotas" {
  description = "按 Agent 的 QPM 和预算上限"
  type = map(object({
    qpm          = number
    daily_tokens = number
    max_budget   = number
  }))
  default = {
    "research-agent" = { qpm = 120, daily_tokens = 2000000, max_budget = 800 }
  }
}
```

`sensitive = true` 让 Terraform 不在 plan/apply 输出里打印值。值还是会进 tfstate（这就是为什么第二篇我们给 OSS bucket 开了加密）。

## env/dev.tfvars

```hcl
agent_repo_url   = "https://github.com/example/research-agent.git"
agent_branch     = "develop"
dingtalk_webhook = "https://oapi.dingtalk.com/robot/send?access_token=DEV_TOKEN"

agent_quotas = {
  "research-agent" = {
    qpm          = 30
    daily_tokens = 200000
    max_budget   = 50
  }
}
```

## secrets/secrets.auto.tfvars（已加入 gitignore）

```hcl
llm_keys = {
  "dashscope-prod" = "sk-DS-XXXXXXXXXXXXXXXXX"
  "openai-prod"    = "sk-XX-XXXXXXXXXXXXXXXXX"
  "anthropic-prod" = "sk-ant-XXXXXXXXXXXXXXXXX"
  "deepseek-prod"  = "sk-DEEPSEEK-XXXXXXXXX"
}
```

`*.auto.tfvars` 文件无需通过 `-var-file` 参数指定，Terraform 会自动加载。确保在项目的首次提交时，`secrets/` 目录就已经被加入到 `.gitignore` 文件中，以避免敏感信息泄露。
## Apply

```bash
cd research-agent-stack
terraform workspace select dev
terraform init
terraform plan -var-file=env/dev.tfvars -out=tfplan
# 看 plan 输出：~31 个资源待创建
terraform apply tfplan
```

新建 apply 的真实耗时：

![真实 apply 时间线——RDS/OpenSearch 主导，其余并行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig3_apply_timeline.png)

时钟分解：

- **0-60s：** VPC、vSwitch、NAT、EIP、KMS 密钥——快资源
- **60-380s：** RDS（5 分钟）、OpenSearch（5.5 分钟）、ECS（~2 分钟）、网关（~1.5 分钟）——全部并行，被最慢的卡住
- **380-460s：** Agent 应用部署、observability 资源、告警

总共约 7 分钟，被 RDS 和 OpenSearch 主导。无变更的重复 apply 30 秒内结束，因为 Terraform 只 diff。

精简后的 apply 文字记录：

```
Terraform will perform the following actions:

  # module.vpc.alicloud_vpc.this will be created
  + resource "alicloud_vpc" "this" {
      + cidr_block = "10.20.0.0/16"
      + vpc_name   = "agents-dev"
      ...
    }

  ...（再 29 个资源）...

Plan: 31 to add, 0 to change, 0 to destroy.

Changes to Outputs:
  + agent_endpoints       = (known after apply)
  + gateway_url           = (known after apply)
  + sls_dashboard_url     = (known after apply)
  + total_estimated_cost  = "~¥1450/月（dev 规格）"

Do you want to perform these actions in workspace "dev"?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

module.vpc.alicloud_vpc.this: Creating...
module.vpc.alicloud_kms_key.this["memory"]: Creating...
module.vpc.alicloud_kms_key.this["secrets"]: Creating...
module.vpc.alicloud_kms_key.this["logs"]: Creating...
module.vpc.alicloud_vpc.this: Creation complete after 4s [id=vpc-uf6abc123]
module.vpc.alicloud_vswitch.private["0"]: Creating...
module.vpc.alicloud_vswitch.private["1"]: Creating...
module.vpc.alicloud_vswitch.private["2"]: Creating...
module.vpc.alicloud_vswitch.public["0"]: Creating...
...
module.storage.alicloud_db_instance.memory: Still creating... [4m 30s elapsed]
module.storage.alicloud_opensearch_app_group.vector: Still creating... [5m 10s elapsed]
module.storage.alicloud_db_instance.memory: Creation complete after 4m 38s [id=pgm-uf6def456]
module.storage.alicloud_opensearch_app_group.vector: Creation complete after 5m 24s [id=os-uf6ghi789]
...
module.compute.alicloud_instance.agent[0]: Creation complete after 1m 52s [id=i-uf6jkl012]
module.gateway.alicloud_alb_listener.gateway: Creation complete after 12s
module.observability.alicloud_log_alert.cost_ceiling: Creation complete after 3s
...

Apply complete! Resources: 31 added, 0 changed, 0 destroyed.

Outputs:

agent_endpoints      = [
  "http://alb-uf6.cn-shanghai.alb.aliyuncs.com",
]
gateway_url          = "http://alb-uf7.cn-shanghai.alb.aliyuncs.com/v1"
sls_dashboard_url    = "https://sls.console.aliyun.com/lognext/project/agents-dev/dashboard/agent-cost-overview"
total_estimated_cost = "~¥1450/月（dev 规格）"
```

这就是一个完整的 Agent stack。ALB endpoint、网关 URL、SLS 看板 URL——任何一个粘进浏览器都能用。

## 第二天及后续的运维操作

基础设施已经部署好了，接下来该做什么？

### 添加新的 Agent

1. 在 `dev.tfvars` 文件中为 `var.agent_quotas` 添加一个新的条目。
2. 执行命令：`terraform apply -var-file=env/dev.tfvars`。
3. `null_resource` 会自动为你生成一个新的 LiteLLM API 密钥。
4. 使用新生成的 `LITELLM_API_KEY` 环境变量，部署你的新 Agent 代码。

整个过程大约需要 30 秒。

### 横向扩容

如果需要扩展 ECS 实例的数量，只需修改模块调用中的 `ecs_count` 参数（或者通过 `tfvars` 文件设置）。执行 `terraform apply` 后，Terraform 会启动新的实例并将它们挂载到 ALB 上，同时确保旧实例在整个过程中保持健康（得益于 `create_before_destroy` 策略）。整个扩容过程实现零停机。

### 从开发环境提升到生产环境

```bash
terraform workspace select prod
terraform apply -var-file=env/prod.tfvars
```

虽然使用的是相同的 Terraform 模块，但生产环境的资源配置更高：高可用 RDS、更大的 OpenSearch 配额、更多的 ECS 实例、真实的钉钉 Webhook、正式的 LLM 密钥，以及 ¥800 的成本上限（开发环境为 ¥100）。首次在生产环境中应用配置可能需要 7 到 10 分钟，而后续的变更通常只需几秒钟。

### 销毁开发环境

当你完成实验后，可以通过以下步骤清理开发环境：

```bash
terraform workspace select dev
terraform destroy -var-file=env/dev.tfvars
```

销毁操作可能会失败，原因在于类似生产环境的资源设置了 `deletion_protection = true`，而状态存储桶（bootstrap state bucket）则启用了 `prevent_destroy = true`。这是有意为之的设计。在开发环境中，`deletion_protection` 的值被设置为 `local.is_prod`，因此只有在生产环境中才会启用保护机制，而在开发环境中可以正常销毁资源。

> **实战经验分享：** 在执行 `terraform destroy` 之前，务必先运行 `terraform plan -destroy`。仔细阅读计划输出，确认要销毁的资源数量是否符合预期。我曾见过一位工程师因为忘记切换工作区，误将 `staging` 环境销毁了。
## 集成你的实际 Agent 代码

这里的 `stack` 是一个*平台*。Agent 的代码来自你的代码仓库（`var.agent_repo_url`），并在 ECS 实例启动时通过 `cloud-init` 自动部署。为了让 Agent 能正常运行，它需要满足以下最基本的约定：

```python
# 这些值由 cloud-init 设置的环境变量提供
LLM_GATEWAY_URL    = os.environ["LLM_GATEWAY_URL"]    # http://alb.../v1
LITELLM_API_KEY    = os.environ["LITELLM_API_KEY"]    # 每个 Agent 独有的密钥
DATABASE_URL       = os.environ["DATABASE_URL"]       # postgres://...
VECTOR_ENDPOINT    = os.environ["VECTOR_ENDPOINT"]    # OpenSearch 的 HTTP 地址
ARTIFACTS_BUCKET   = os.environ["ARTIFACTS_BUCKET"]   # OSS 存储桶名称
SLS_PROJECT        = os.environ["SLS_PROJECT"]
SLS_LOGSTORE       = os.environ["SLS_LOGSTORE"]
ARMS_OTLP_ENDPOINT = os.environ["ARMS_OTLP_ENDPOINT"]
```

上述所有配置项的值均来源于 Terraform 的输出结果。虽然 Agent 的代码设计保持了云平台无关性——只需读取环境变量即可工作，但在运行时，它会被无缝接入到阿里云的技术栈中。
## 成本概览

以下是 `dev` 环境的实际账单，假设流量较低：

| 组件                     | 每月费用 |
|--------------------------|---------:|
| VPC + NAT + EIP          | ~¥150   |
| ECS x1 (c7.large)        | ~¥250   |
| RDS Postgres（小型）     | ~¥350   |
| OpenSearch 向量引擎      | ~¥800   |
| OSS（10 GB 标准存储）    | ~¥2     |
| LLM 网关 ECS x1          | ~¥150   |
| ALB（小型）              | ~¥50    |
| SLS + ARMS               | ~¥300   |
| KMS                      | ~¥10    |
| **dev 总计**             | **~¥2060/月** |

生产环境启用高可用（HA）、更大规格实例以及跨区域容灾（DR）时，每月费用大约在 ¥6000 到 ¥9000 之间（不含 LLM API 费用）。其中，LLM 的账单往往是最大的开支项——这也是第六篇中提到的网关设计和第七篇中成本告警机制的重要原因。
## 我暂时省略的部分

- **CDN** 用于公开分发产物 URL——虽然 `alicloud_cdn_domain` 可以实现，但大多数 Agent 会通过自己的网关来提供服务  
- **WAF** 部署在 ALB 前端——生产环境对外暴露时是必需的，但开发环境使用的是内网 ALB，暂时不需要  
- **PrivateLink** 连接到 DashScope——在大规模场景下可以显著节省 NAT 出口流量成本，可以通过 `alicloud_privatelink_*` 配置实现  
- **自定义域名 + SSL**——`alicloud_alb_listener` 支持 SSL 证书配置，但需要自己准备证书（或者使用 ACM 提供的证书）  

这四点功能在基础架构跑通后都可以逐步添加，但不建议一开始就引入，避免增加复杂度。
## 接下来该做什么？

到目前为止，你已经在阿里云上搭建了一个符合生产标准的 Agent 运行环境，完全通过 Terraform 实现，并内置了可观测性、密钥管理以及成本控制机制。接下来的方向取决于你的具体需求：

- **扩展 Agent 数量：** 只需调整 `var.agent_quotas` 参数，然后执行 `terraform apply` 即可。
- **支持更多 LLM 服务提供商：** 在网关模块中修改 `local.litellm_config` 配置即可完成集成。
- **多区域部署：** 添加 provider 别名并复制当前的资源栈，轻松实现跨区域扩展。
- **GitOps 实践：** 将 `terraform apply` 嵌入到基于 PR 审核的 CI/CD 流水线中，进一步提升流程的自动化与安全性。
- **迁移到 Pulumi 或 Crossplane：** 当前的资源依赖图可以直接映射到这些工具中，无缝切换。

最重要的是，你的基础设施现在已经完全纳入 Git 版本控制。每一次变更都可以被审查，每一个环境都能够被复现，每一笔开销都清晰可追溯。这正是 IaC（基础设施即代码）的核心价值所在——它让在阿里云上交付和维护 Agent 成为一项可持续的工程实践，而不是一场永无止境的“救火”行动。

感谢你阅读本系列文章！如果你基于这套方案搭建了自己的环境，我很期待听到你的改进和优化思路——正是这些反馈推动了最佳实践的不断演进。
## 推进策略：从开发环境到生产环境，稳扎稳打不出错

![用 Terraform 给 AI Agent 上云（八）：端到端——一次 apply 起整个 research-agent-stack — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/illustration_2.jpg)

文章里演示了 `terraform workspace select prod && terraform apply`。这方法能用，但也是大多数生产事故的源头——因为从开发环境（dev）到生产环境（prod）的过程中，往往会暴露出一些没人预料到的差异。

在实际项目中，我使用了一套更稳健的推进流水线，确保每个环节都经过验证，避免踩坑。

### 第一步：给开发环境的状态文件打个快照

在任何环境推进之前，先对开发环境的状态文件做个备份。如果生产环境出了问题，而开发环境正常，你至少还能对比两者的差异：

```bash
terraform state pull > /tmp/dev-state-$(date -Iseconds).json
aliyun oss cp /tmp/dev-state-*.json oss://ck-tfstate-archive/snapshots/
```

这是非常划算的保险。状态文件通常很小（不到 1MB），而且归档桶配置了生命周期规则，30 天后会自动转入冷归档存储。

### 第二步：基于开发环境验证过的代码生成生产环境的执行计划

永远不要把未经测试的代码推到生产环境。只有那些在开发环境中稳定运行至少一周的提交记录，才有资格用来生成生产环境的执行计划：

```yaml
# .github/workflows/promote.yml
on:
  workflow_dispatch:
    inputs:
      from_workspace:
        type: choice
        options: [dev, staging]
      to_workspace:
        type: choice
        options: [staging, prod]
      commit_sha:
        description: "Validated commit SHA from from_workspace"

jobs:
  promote:
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ inputs.commit_sha }}
      - name: 在目标工作区生成 Terraform 执行计划
        env:
          TF_WORKSPACE: ${{ inputs.to_workspace }}
        run: |
          terraform init
          terraform plan -var-file=env/${{ inputs.to_workspace }}.tfvars \
            -out=tfplan-promote 2>&1 | tee promote-plan.txt
      - name: 将执行计划发送到钉钉供人工审核
        run: |
          curl -X POST "$DINGTALK_WEBHOOK" -d "{\"text\":{\"content\":\"推进计划 ${{ inputs.from_workspace }}→${{ inputs.to_workspace }} 已准备好，请审核\"}}"
```

执行计划会推送到值班工程师的钉钉上。他们会仔细检查计划内容，尤其是那些和开发环境**不一致**的部分。如果计划显示某些资源需要重新创建，但在开发环境中并未触发重建——那就停下来，先调查清楚再继续。

### 第三步：应用、验证、解锁

真正的生产环境应用操作由 GitHub 环境保护机制把关（第六篇文章中有详细配置），必须经过指定的评审者批准。应用完成后，在切换流量之前，先跑一个冒烟测试：

```bash
# 在 apply 任务成功后运行
gateway=$(terraform output -raw gateway_url)
session=$(curl -s -X POST $gateway/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"ping"}]}' | jq -r .choices[0].message.content)

[[ -n "$session" ]] || { echo "冒烟测试失败"; exit 1; }
```

这个 5 秒钟的冒烟测试可以有效拦截“我把网关搞挂了”这类错误的传播。如果测试失败，虽然技术上 `apply` 已经成功，但你可以利用第一步保存的状态快照快速回滚（通过 `terraform state push /tmp/dev-state-...json` 和 `terraform apply` 恢复到之前的状态）。

### 第四步：跨环境对比输出结果

生产环境应用完成后，还需要做一次**跨工作区**的输出对比，确保没有意外差异：

```bash
diff <(terraform workspace select staging && terraform output -json) \
     <(terraform workspace select prod && terraform output -json) \
     | head -100
```

预期中的差异包括：实例数量、RDS 的高可用标志（HA flag）、灾备区域等。除此之外的任何差异都需要引起注意——这些往往暴露了 tfvars 文件中的拼写错误，或者与工作区相关的条件逻辑问题。

这套四步推进流程，三年来大概帮我避免了 30 多次潜在故障。每一步耗时不过几分钟，整套流程累计多花一个小时左右的时间。对于减少生产环境风险来说，这点时间成本非常值得。
## 多区从一个项目：扇出和读副本

服务中国和东南亚用户的 Agent，最终会需要 `cn-shanghai`（或 `cn-beijing`）和 `ap-southeast-1`（新加坡）。朴素答案是两个完全独立的 Terraform 项目。更好的答案是 provider alias 加每区 module 实例：

```hcl
# providers.tf
provider "alicloud" {
  alias  = "shanghai"
  region = "cn-shanghai"
}

provider "alicloud" {
  alias  = "singapore"
  region = "ap-southeast-1"
}

# main.tf
module "stack_shanghai" {
  source = "./modules/agent-stack"
  providers = {
    alicloud = alicloud.shanghai
  }
  name        = "agents-prod-cn"
  cidr_block  = "10.20.0.0/16"
  zones       = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]
  is_primary  = true
}

module "stack_singapore" {
  source = "./modules/agent-stack"
  providers = {
    alicloud = alicloud.singapore
  }
  name        = "agents-prod-sg"
  cidr_block  = "10.30.0.0/16"
  zones       = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
  is_primary  = false
  primary_endpoints = module.stack_shanghai.endpoints
}
```

`agent-stack` module 是第三到七篇做的全套，打包成一个。`is_primary` 控制 RDS 是 master 还是只读副本、OSS 是源还是复制目标等。

跨区互联用 CEN（云企业网）打通 VPC：

```hcl
resource "alicloud_cen_instance" "agents" {
  cen_instance_name = "agents-cen"
  description       = "CEN linking shanghai and singapore agent VPCs"
  protection_level  = "REDUCED"
}

resource "alicloud_cen_instance_attachment" "shanghai" {
  provider                 = alicloud.shanghai
  instance_id              = alicloud_cen_instance.agents.id
  child_instance_id        = module.stack_shanghai.vpc_id
  child_instance_type      = "VPC"
  child_instance_region_id = "cn-shanghai"
}

resource "alicloud_cen_instance_attachment" "singapore" {
  provider                 = alicloud.singapore
  instance_id              = alicloud_cen_instance.agents.id
  child_instance_id        = module.stack_singapore.vpc_id
  child_instance_type      = "VPC"
  child_instance_region_id = "ap-southeast-1"
}

resource "alicloud_cen_bandwidth_package" "this" {
  bandwidth                  = 50    # Mbps 跨区
  geographic_region_a_id     = "China"
  geographic_region_b_id     = "Asia-Pacific"
  cen_bandwidth_package_name = "agents-cn-sg-50m"
}

resource "alicloud_cen_bandwidth_package_attachment" "this" {
  instance_id          = alicloud_cen_instance.agents.id
  bandwidth_package_id = alicloud_cen_bandwidth_package.this.id
}
```

CEN 按跨区带宽计费——上面 50 Mbps 包 ~¥3000/月。真服务多区流量值；"我哪天可能"过度了。

整套多区从单项目一次 `terraform apply`。两个 region 并行部署（Terraform 的 DAG 处理得不错）。总 apply 时间从 7 分钟到 ~9 分钟，因为最慢路径（RDS provisioning）每区并发跑。

## 生产环境完整成本核算

文章中提到开发环境每月大约 ¥2060。这里我们详细拆解生产环境的成本——这是财务问你“AI Agent 平台到底要花多少钱”时可以用的数字。

以下是生产环境的资源配置和费用（基于文章中定义的 workspace 默认值）：

| 层           | 资源                                    | 规格                             | 月费（¥） |
|--------------|----------------------------------------|----------------------------------|---------:|
| 网络         | VPC、vSwitch、路由表、KMS              | 三可用区，3 把 CMK              |       10 |
| 网络         | 增强型 NAT 网关 + EIP                  | 占用 + 1 TB 出站流量            |      920 |
| 计算         | ECS x3（`ecs.c7.xlarge` 4核/8GB）      | 3 台实例，每台 80G ESSD         |     1380 |
| 计算         | LiteLLM 网关 ECS x2                   | `ecs.c7.large` 2核/4GB          |      450 |
| 计算         | 标准版 ALB                             | 1 个 ALB，公网访问              |      180 |
| 存储         | RDS Postgres 高可用（`pg.x4.large.2c`）| 200 GB ESSD + 备机              |     2200 |
| 存储         | OpenSearch 向量检索（中等规格）        | 50 文档大小、80 计算单元         |     1800 |
| 存储         | OSS（500 GB 标准存储 + 生命周期策略）  | 主要是标准存储，少量归档存储    |      100 |
| 存储         | OSS 容灾副本（cn-beijing）             | 500 GB 归档存储                 |       60 |
| 密钥管理     | KMS Secrets Manager                    | 8 个密钥，5 万次解密/月         |       50 |
| 监控与可观测 | SLS                                    | 30 GB 日志写入，保留 90 天      |      450 |
| 监控与可观测 | ARMS APM                               | 1 个环境，5000 万条链路追踪     |      600 |
| 监控与可观测 | CloudMonitor                           | 主机指标 + 20 个自定义监控项    |       30 |
| **小计——基础设施**                                                                    |  **8230** |
| LLM API      | DashScope Qwen-max                     | 5000 万输入 token，1200 万输出 token |     3500 |
| LLM API      | Anthropic / OpenAI 兜底                | 500 万输入 token，100 万输出 token   |      800 |
| **小计——LLM**                                                                        |  **4300** |
| **生产环境总成本 / 月**                                                              | **¥12,530** |

结合实际账单数据，我总结了几点关键观察：

- **OpenSearch 和 RDS 占基础设施成本约 31%。** 如果预算紧张，可以考虑只用 pgvector（去掉 OpenSearch），这样每月能省下 ¥1800，但混合检索性能会稍差。对于向量规模在百万以下的场景，这种取舍是值得的。
- **NAT 出站流量是个意外的大头。** 在一个项目中，我通过切换到 PrivateLink 调用 DashScope，将 NAT 账单降低了 60%。
- **在这个规模下，LLM 成本占总成本的 35%。** 当流量扩大到当前的 10 倍时，LLM 的占比会飙升到 70%-80%。此时，网关的每个 Agent 配额将成为比基础设施更重要的成本控制杠杆。
- **可观测性成本占基础设施的 10%。** 这是一个合理的比例：如果低于 5%，说明你的监控覆盖不足；如果高于 20%，可能收集了过多无用数据。建议每月检查 SLS 的日志写入量是否合理。

从财务角度评估：每次会话的平均成本为 ¥12,530 ÷ 每月会话数。如果每月有 10 万次会话，则单次成本为 ¥0.125；如果只有 1000 次会话，则单次成本高达 ¥12.50——这时你可能需要重新评估平台运行的性价比是否合理。
## Day-3 运维：那些持续带来回报的最佳实践

除了文章中提到的 day-2 清单，我还会在每个长期运行的栈上执行以下四个模式。这些实践不仅能提升系统的健壮性，还能帮助团队更好地管理资源和成本。

### 1. 季度模块依赖升级

每季度定期升级 `alicloud` provider、所有开源模块以及 Terraform 本身，每次只升一个次要版本（minor version）。先在开发环境（dev）中运行 `terraform plan`，确认无误后执行 `apply`，然后观察一周，确保系统稳定后再推广到生产环境。这种习惯可以避免在 CVE-2027-XXX 漏洞爆发时，发现自己已经落后了三年。

### 2. 跨区域的 State 备份

存储 Terraform state 的 OSS bucket 目前位于 cn-shanghai 区域。如果 cn-shanghai 发生区域性故障，你将无法执行任何 Terraform 操作——即使目标是其他区域的资源。为防患于未然，可以每周将 state 文件备份到 cn-beijing 区域（通过 OSS 的跨区域复制功能）。每月只需 ¥10，却能在最坏情况下救你一命：

```hcl
resource "alicloud_oss_bucket_replication" "tfstate" {
  bucket = alicloud_oss_bucket.tfstate.id
  action = "ALL"
  destination {
    bucket   = "ck-tfstate-prod-dr"
    location = "oss-cn-beijing"
  }
}
```

### 3. 按 Agent 归因的月度成本分析

Gateway 会记录每个 Agent 的使用成本（详见第七篇）。在月底时，按 Agent 汇总成本，并将结果发布到团队频道中。例如："研究 Agent: ¥3200，客服 Agent: ¥800，代码 Agent: ¥4100"。这种透明化的展示让成本变得具体而真实，工程师看到自己的 Agent 出现在“成本排行榜”上时，往往会主动优化资源使用。

### 4. 基于 IaC 的年度架构评审

每年进行一次全面的架构评审，遍历整个 `terraform state list`，逐一检查每个资源是否仍然必要。有些资源可能是历史遗留的，比如从未删除的开发集群，或者从 v15 升级后废弃的 RDS 实例。提交清理 PR 销毁这些无用资源，是我认为 ROI 最高的 Terraform 工作之一，通常每年能节省账单的 10%-15%。

### 总结

到这里就结束了。八篇文章，围绕一个完整的栈展开，最终形成一个 Terraform 项目。起步仓库（https://github.com/example/research-agent-stack）已经为你准备好了，随时可以 fork 并开始使用。如果你基于它构建了一个真实的 Agent，请告诉我你做了哪些改动以及背后的原因——正是这些反馈让我们的最佳实践不断进化和完善。
