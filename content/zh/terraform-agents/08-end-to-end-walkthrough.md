---
title: "Terraform 实战（八）：一条命令拉起全栈"
date: 2026-03-26 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - End-to-End
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 8
description: "把七个 module 拼到一个仓库，跑一次 terraform apply，看一个完整的 Agent runtime——VPC、ECS、RDS、OpenSearch、OSS、LLM 网关、SLS 观测、成本告警——七分钟内起来。真实 apply 输出、module DAG、生产环境完整成本核算，以及可 fork 的起手仓库。"
disableNunjucks: true
translationKey: "terraform-agents-8"
---
本系列第 2 至第 7 篇所构建的全部模块，最终在此完成整合与部署。跑一次 `terraform apply`，你就能在阿里云上得到一套完整、可观测、带预算控制的 Agent 运行时栈——大概 31 个资源，实际耗时 7 分钟左右，生产环境规模下全包成本约 ¥12,530/月。

我们要搭建的栈结构如下：

![research-agent-stack: every box, one terraform apply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig1_full_stack.png)

共五层：边缘、计算、记忆、平台、运维——均由本系列此前构建的模块组合而成；底层依赖 11 款阿里云服务： VPC、 ECS、 ALB、 OSS、 RDS for PostgreSQL、 OpenSearch、 KMS、 SLS、 ARMS、 CloudMonitor，以及通过网关调用的 DashScope （LLM 接入服务）。

## 项目结构

```
research-agent-stack/
├── README.md
├── versions.tf                  # Terraform + provider pinning
├── backend.tf                   # OSS + Tablestore remote state
├── providers.tf                 # alicloud + alicloud.beijing alias
├── variables.tf                 # top-level inputs
├── locals.tf                    # workspace-aware computed locals
├── main.tf                      # module composition
├── outputs.tf                   # endpoints + connection strings
├── env/
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── prod.tfvars
├── secrets/
│   └── secrets.auto.tfvars      # gitignored — provider keys
├── modules/
│   ├── vpc-baseline/            # article 3
│   ├── storage/                 # article 5
│   ├── compute/                 # article 4
│   ├── llm-gateway/             # article 6
│   └── observability/           # article 7
└── scripts/
    ├── cloud-init/
    │   ├── agent.sh
    │   └── gateway.sh
    └── restore-drill.sh
```

![Infrastructure modules composing together into a complete architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/wanxiang_module_composition.png)


顶层八个 `*.tf` 文件，`modules/` 目录下五个模块，环境变量放在 `env/*.tfvars`，密钥隔离在 `secrets/secrets.auto.tfvars` 且不进 git。这是我每个项目的标准目录结构：略显刻板，但胜在稳定可靠。唯独 `secrets/` 目录必须从第一次提交就开始被 `.gitignore` 忽略，这点我绝不妥协。我处理过的所有密钥泄露事件，根本原因都是 gitignore 文件未在项目初始化时配置，而是在后续（例如第 50 次提交）才临时补充。

## main.tf — 组成

```hcl
locals {
  is_prod   = terraform.workspace == "prod"
  name      = "agents-${terraform.workspace}"
  zones     = ["cn-shanghai-l", "cn-shanghai-m", "cn-shanghai-n"]

![Complete cloud architecture stack from network to application layer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/wanxiang_full_stack.png)


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
  enable_dr         = local.is_prod   # cross-region OSS replication only in prod
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

五个模块调用。每个模块都把*前一个*模块的输出作为输入——`module.compute` 读取 `module.vpc`、`module.storage`、`module.gateway`、`module.observability`。这类模块间的依赖关系，正是 Terraform 构建 apply 执行图（有向无环图， DAG）的依据。

![Terraform module dependency DAG](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig2_module_dag.png)

VPC 与 KMS 位于依赖链顶端，不依赖任何其他模块；Storage 与 gateway 均依赖 VPC 和 KMS，但彼此无依赖，因此 Terraform 并行创建；Compute 模块依赖前三者，因为其 cloud-init 模板需要引用它们输出的 endpoint 地址；Observability 资源最后部署，需引用 compute 的 security group ID。

`local.is_prod` 里的三元表达式就是全部的环境升级策略，三行代码搞定：生产环境获得高可用 RDS、两个网关实例、三个 Agent ECS、¥800 成本上限、跨地域灾备。开发环境则使用最小可行配置。模块完全相同，仅通过变量调节规模，因此无需为不同环境维护独立代码分支或条件逻辑。

## variables.tf

```hcl
variable "agent_repo_url" {
  description = "Git URL of the agent runtime to deploy"
  type        = string
  default     = "https://github.com/example/research-agent.git"
}

variable "agent_branch" {
  description = "Git branch / tag to deploy"
  type        = string
  default     = "main"
}

variable "dingtalk_webhook" {
  description = "DingTalk webhook URL for alarms"
  type        = string
  sensitive   = true
}

variable "llm_keys" {
  description = "Map of provider name to API key — set via secrets.auto.tfvars"
  type        = map(string)
  sensitive   = true
}

variable "agent_quotas" {
  description = "Per-agent QPM and budget caps"
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

`sensitive = true` 防止 Terraform 在 plan/apply 输出中打印值。不过值还是会写进 tfstate （这就是为什么我们在第 2 篇里用独立的 KMS CMK 加密了 OSS state bucket）。

## env/dev.tfvars 和 secrets

```hcl
# env/dev.tfvars
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

```hcl
# secrets/secrets.auto.tfvars  (gitignored)
llm_keys = {
  "dashscope-prod" = "sk-DS-XXXXXXXXXXXXXXXXX"
  "openai-prod"    = "sk-XX-XXXXXXXXXXXXXXXXX"
  "anthropic-prod" = "sk-ant-XXXXXXXXXXXXXXXXX"
  "deepseek-prod"  = "sk-DEEPSEEK-XXXXXXXXX"
}
```

`*.auto.tfvars` 文件会自动加载，不需要 `-var-file` 参数，所以 `secrets.auto.tfvars` 会被自动拾取，而 `env/dev.tfvars` 则按 workspace 显式传递。双文件模式，避免了 `terraform.tfvars` 的歧义。

## 应用

```bash
cd research-agent-stack
terraform workspace select dev
terraform init
terraform plan -var-file=env/dev.tfvars -out=tfplan
# review plan output: ~31 resources to add
terraform apply tfplan
```

全新 apply 的实际耗时：

![Real apply timeline — RDS/OpenSearch dominate, the rest is parallel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig3_apply_timeline.png)

实际耗时分解（wall-clock time）：

- **0–60s：** VPC、 vSwitch、 NAT、 EIP、 KMS keys —— 快速资源
- **60–380s：** RDS （约 5 分钟）、 OpenSearch （约 5.5 分钟）、 ECS （约 2 分钟）、 gateway （约 1.5 分钟）——这些资源并行创建，整体耗时取决于最慢的一项。
- **380–460s：** 通过 cloud-init 部署 agent 应用、观测资源、告警

总共约 7 分钟，主要耗时集中在 RDS 和 OpenSearch 的创建过程。如果没有变更再次 apply，会在 30 秒内完成，因为 Terraform 只做 diff。

一份精简后的 apply 转录：

```yaml
Terraform will perform the following actions:

  # module.vpc.alicloud_vpc.this will be created
  + resource "alicloud_vpc" "this" {
      + cidr_block = "10.20.0.0/16"
      + vpc_name   = "agents-dev"
      ...
    }

  ... (29 more resources) ...

Plan: 31 to add, 0 to change, 0 to destroy.

Changes to Outputs:
  + agent_endpoints       = (known after apply)
  + gateway_url           = (known after apply)
  + sls_dashboard_url     = (known after apply)
  + total_estimated_cost  = "~¥2060/month at dev sizing"

Do you want to perform these actions in workspace "dev"?
  Enter a value: yes

module.vpc.alicloud_vpc.this: Creating...
module.vpc.alicloud_kms_key.this["memory"]: Creating...
module.vpc.alicloud_kms_key.this["secrets"]: Creating...
module.vpc.alicloud_kms_key.this["logs"]: Creating...
module.vpc.alicloud_vpc.this: Creation complete after 4s [id=vpc-uf6abc123]
module.vpc.alicloud_vswitch.private["0"]: Creating...
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

agent_endpoints      = ["http://alb-uf6.cn-shanghai.alb.aliyuncs.com"]
gateway_url          = "http://alb-uf7.cn-shanghai.alb.aliyuncs.com/v1"
sls_dashboard_url    = "https://sls.console.aliyun.com/lognext/project/agents-dev/dashboard/agent-cost-overview"
total_estimated_cost = "~¥2060/month at dev sizing"
```

至此，一套完整的 Agent 运行时栈已完成部署。 ALB 端点、网关 URL、 SLS 仪表盘 URL 均已就绪，任选其一粘贴至浏览器即可直接访问。`total_estimated_cost` 输出是在 `outputs.tf` 里根据 workspace 条件计算出来的，所以你在 plan 里看到的数字和账单上显示的相符（误差 ~10% 以内，这是我长期运行的经验法则）。
## Day-2 运营

栈搭好了，然后呢？这些是我每个长期运行的栈都会做的操作——这些操作虽未在正文中详述，却是 on-call 工程师日常运维的必备实践。

![CI/CD pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/wanxiang_cicd_pipeline.png)


### 添加新 Agent

1. 在 `dev.tfvars` 的 `var.agent_quotas` 里加一条配置
2. 执行 `terraform apply -var-file=env/dev.tfvars`
3. 网关模块里的 `null_resource` 会 部署 一个新的 LiteLLM key
4. 带着新的 `LITELLM_API_KEY` 环境变量部署你的 Agent 代码

端到端大概 30 秒。该模式首次上线时，产品团队曾提出需求：希望支持通过 Slack 表单自助接入 Agent。一旦有了 Terraform 契约，写这个表单也就半天工作量。

### 扩容

修改模块调用里的 `ecs_count`（或者通过 `tfvars` 设置）。`terraform apply` 会拉起新实例，挂到 ALB 上，旧实例在整个过程中保持健康（`create_before_destroy`）。零停机。我曾在一次流量激增期间，于凌晨 2 点通过修改这一行配置，将 Agent 实例数从 3 扩容至 12。

### 销毁 Dev 环境

实验做完后：

```bash
terraform workspace select dev
terraform destroy -var-file=env/dev.tfvars
```

这在 prod 会失败，因为 bootstrap state bucket 上设了 `deletion_protection = true` 和 `prevent_destroy = true`。这是故意的。在 dev 环境 `deletion_protection = local.is_prod`，所以只有 prod 才开启保护——`terraform destroy` 能正常跑。

> 执行 `terraform destroy` 前务必先 `terraform plan -destroy`。仔细看 plan 输出。被销毁的资源数量必须和你预期的一致。我曾亲眼见过一个工程师因为忘了切换 workspace 把 `staging` 给删了。花了六个小时，还有一位 senior backend 在 PagerDuty 上待命，才把数据重建回来。

### 从 Dev 晋升到 Staging 再到 Prod

文章里展示了 `terraform workspace select prod && terraform apply`。第一天这么干没问题。到了第三个月，大部分生产事故都源于此，因为 dev → prod 会暴露没人预料到的差异。

我在真实项目上跑的晋升流水线有四步。每步几分钟；累计成本是每个版本多花一个日历小时。作为回报，这套流程在过去三年里大概防止了 30+ 次 outage。

**第一步：快照源状态。** 任何晋升之前，拷贝一份源 workspace 的状态文件。如果 prod 出了在 dev 好好的问题，你需要能对比：

```bash
terraform state pull > /tmp/dev-state-$(date -Iseconds).json
aliyun oss cp /tmp/dev-state-*.json oss://ck-tfstate-archive/snapshots/
```

状态快照很小（通常 <1MB），归档 bucket 设置了 30 天后转冷归档。整个历史记录每月成本 ¥0.something。

**第二步：在 CI 里基于验证过的 commit 计算 prod plan。** 别晋升未测试的代码。只有在 dev 干净跑了一周的 exact commit 才配得到 prod 的 `plan`：

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
      - name: terraform plan in target workspace
        env:
          TF_WORKSPACE: ${{ inputs.to_workspace }}
        run: |
          terraform init
          terraform plan -var-file=env/${{ inputs.to_workspace }}.tfvars \
            -out=tfplan-promote 2>&1 | tee promote-plan.txt
      - name: post promotion plan to DingTalk for human review
        run: |
          curl -X POST "$DINGTALK_WEBHOOK" \
            -d "{\"text\":{\"content\":\"Promotion plan ${{ inputs.from_workspace }}→${{ inputs.to_workspace }} ready for review\"}}"
```

Plan 会发到钉钉上的 on-call 工程师。他们会特别关注任何与 dev 显示*不同*的地方。如果 plan 显示有资源要重建，而在 dev 里没重建——停手。 Apply 之前先调查。这通常源于 workspace 条件配置错误或 tfvars 文件拼写错误。

**第三步： Apply，烟雾测试，然后解封。** 实际的 prod apply  gated 在一个需要审批人的 GitHub Environment 上（第 6 篇文章里配置过）。 Apply 成功后，流量切换前先跑烟雾测试：

```bash
gateway=$(terraform output -raw gateway_url)
reply=$(curl -s -X POST $gateway/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"ping"}]}' \
  | jq -r .choices[0].message.content)

[[ -n "$reply" ]] || { echo "Smoke test failed"; exit 1; }
```

5 秒钟的烟雾测试能在错误传播前抓住“我把网关搞挂了”这类问题。如果失败，虽然 apply  technically 成功了，但你可以用第一步的快照回滚。

**第四步： Apply 后与 Staging 对比。** 跑一个跨 workspace 对比：

```bash
diff <(terraform workspace select staging && terraform output -json) \
     <(terraform workspace select prod && terraform output -json) \
     | head -100
```

预期差异：实例数量、 RDS HA 标志、 DR 区域。意外差异：其他任何内容。调查它们——这往往能 reveal 一个 tfvars 拼写错误或者第三步没触发的 workspace 条件 bug。

### 季度模块依赖升级

每个季度：把 `alicloud` provider、所有开源模块和 Terraform 本身升级一个小版本。在 dev 跑 plan。 Apply，浸泡一周。晋升。这种纪律能防止当 CVE-2027-XXX 爆发时，你落后三年，被迫在一个周末紧急跨六个版本升级。

### 状态备份到不同 Region

存状态的 OSS bucket 在 cn-shanghai。如果 cn-shanghai 发生区域级事件，你无法 apply Terraform——包括对其他区域。每周把状态 bucket 本身复制到 cn-beijing，每月成本 ¥10，最坏情况下能救你的命：

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

### 每月按 Agent 分摊成本

网关会记录每个 Agent 的成本（第 7 篇文章）。月底按 Agent 汇总，发到团队频道。"Research agent: ¥3,200, Support agent: ¥800, Code agent: ¥4,100"让成本变得具体。当工程师的名字出现在榜单上时，他们会自我约束。我有三个团队在开始这种做法的两个月内把 LLM 账单砍了一半——不需要自上而下的命令。

### 年度架构对照 IaC 审查

每年一次，遍历整个 `terraform state list`，对每个资源问：我们还需要这个吗？有些是残留的——你从未删除的 dev 集群，你升级过的 v15 RDS。销毁未使用资源的 Cleanup PRs 是我做过 ROI 最高的 Terraform 工作——通常能省下年度账单的 10–15%。

## 连接你的实际 Agent 代码

栈是*平台*。 Agent 本身来自你的 repo （`var.agent_repo_url`），由 ECS 启动时的 cloud-init 部署。你的 Agent 代码需要遵守的最小契约：

```python
# These come from environment variables set by cloud-init
LLM_GATEWAY_URL    = os.environ["LLM_GATEWAY_URL"]    # http://alb.../v1
LITELLM_API_KEY    = os.environ["LITELLM_API_KEY"]    # the per-agent key
DATABASE_URL       = os.environ["DATABASE_URL"]       # postgres://...
VECTOR_ENDPOINT    = os.environ["VECTOR_ENDPOINT"]    # OpenSearch HTTP
ARTIFACTS_BUCKET   = os.environ["ARTIFACTS_BUCKET"]   # OSS bucket name
SLS_PROJECT        = os.environ["SLS_PROJECT"]
SLS_LOGSTORE       = os.environ["SLS_LOGSTORE"]
ARMS_OTLP_ENDPOINT = os.environ["ARMS_OTLP_ENDPOINT"]
```

所有这些值都来自 Terraform outputs。 Agent 代码在形态上保持云无关——它只读环境变量——但在运行时完全接入 Aliyun 栈。当有人问“怎么把这个移到 AWS？”时，答案是：换模块， Agent 代码保持不变。契约就是这份环境变量列表。

## 成本算术 — Dev 和 Prod

对于 Dev （低流量、单可用区、无 HA）：

| 组件                      | 每月      |
|-------------------------|--------:|
| VPC + NAT + EIP         | ~¥150 |
| ECS x1 (`ecs.c7.large`) | ~¥250 |
| RDS Postgres (small)    | ~¥350 |
| OpenSearch vector       | ~¥800 |
| OSS (10 GB Standard)    | ~¥2 |
| LLM Gateway ECS x1      | ~¥150 |
| ALB (small)             | ~¥50 |
| SLS + ARMS              | ~¥300 |
| KMS                     | ~¥10 |
| **Dev 总计**           | **~¥2,060/mo** |

Prod，全 HA，跨区域 DR，真实流量——这是财务问"AI Agent 平台实际成本是多少？”时你引用的数字：

| 层级        | 资源                                | 规格                        | 每月 (¥) |
|--------------|-----------------------------------------|-------------------------------|------------:|
| 网络      | VPC, vSwitch, RT, KMS                   | 3 可用区， 3 CMKs                |          10 |
| 网络      | NAT Gateway (Enhanced) + EIP            | 预留 + 1 TB 出口        |         920 |
| 计算      | ECS x3 (`ecs.c7.xlarge` 4c/8g)          | 3 实例，每个 80 GB ESSD  |        1380 |
| 计算      | LiteLLM gateway ECS x2                  | `ecs.c7.large` 2c/4g          |         450 |
| 计算      | ALB Standard                            | 1 ALB, 面向互联网        |         180 |
| 内存       | RDS Postgres HA (`pg.x4.large.2c`)      | 200 GB ESSD + 备机         |        2200 |
| 内存       | OpenSearch vector (medium)              | 50 文档大小， 80 计算       |        1800 |
| 内存       | OSS (500 GB Standard + lifecycle)       | 主要是 Standard, 部分 IA      |         100 |
| 内存       | OSS DR replica (cn-beijing)             | 500 GB IA                     |          60 |
| 密钥      | KMS Secrets Manager                     | 8  secrets, 50k 解密/月    |          50 |
| 可观测性| SLS                                     | 30 GB 摄入， 90 天保留      |         450 |
| 可观测性| ARMS APM                                | 1 环境， 50M spans              |         600 |
| 可观测性| CloudMonitor                            | 主机指标 + 20 自定义      |          30 |
| **小计 — 基础设施**                                                                   |    **8,230** |
| LLM API      | DashScope Qwen-max                      | 50M 输入， 12M 输出 tokens  |        3500 |
| LLM API      | Anthropic / OpenAI fallback             | 5M 输入， 1M 输出 tokens    |         800 |
| **小计 — LLM**                                                                     |    **4,300** |
| **Prod 总计 / 月**                                                                 |   **¥12,530** |

来自真实账单的四个观察：

- **OpenSearch 和 RDS 加起来约占基础设施成本的 31%。** 如果你缩放得紧，只用 pgvector 的方案（去掉 OpenSearch）每月能省 ~¥1,800，代价是混合搜索变慢。向量数低于 1M 时值得做。
- **NAT 出口费是意外的大项。** 在一个项目上切换到 DashScope 的 PrivateLink 让我的 NAT 账单降了 60%。在任何非 trivial 的流量下，这点配置开销都值得。
- **在这个规模下 LLM 占总额的 35%。** 流量到 10× 时， LLM 会变成账单的 70–80%。网关的每 Agent 配额会在基础设施之前成为最重要的成本杠杆。
- **可观测性占基础设施的 10%。** 这是正确的比例——低于 5% 意味着 instrumentation 不足；超过 20% 意味着收集太多。每月审计 SLS 摄入 volume。

用于财务审查：单次会话成本是 ¥12,530 / 每月会话数。在 100k sessions/month 时，那是 ¥0.125 每会话。在 1k sessions/month 时是 ¥12.50——这时候你就要开始问，为了这个流量跑这个平台是否值得，或者是否应该 consolidate 到别人的平台上。
## 更进一步：单项目实现多地域部署

如果你的 Agent 既要服务国内用户，又要覆盖东南亚，迟早得在 `cn-shanghai`（或 `cn-beijing`）和 `ap-southeast-1`（新加坡）都有部署。最直观的做法是搞两套完全独立的 Terraform 项目。但更好的方案是利用 provider alias 配合按地域实例化的 module：

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
  name              = "agents-prod-sg"
  cidr_block        = "10.30.0.0/16"
  zones             = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
  is_primary        = false
  primary_endpoints = module.stack_shanghai.endpoints   # for cross-region replication
}
```

`agent-stack` 模块就是把前面第 3 到第 7 篇的内容打包在一起。`is_primary` 参数决定了 RDS 是主库还是只读副本， OSS 是拥有 Bucket 还是作为复制目标等等。

跨地域连线得靠 CEN （Cloud Enterprise Network）来实现 VPC 间的私网互通：

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
  bandwidth                  = 50    # Mbps between regions
  geographic_region_a_id     = "China"
  geographic_region_b_id     = "Asia-Pacific"
  cen_bandwidth_package_name = "agents-cn-sg-50m"
}

resource "alicloud_cen_bandwidth_package_attachment" "this" {
  instance_id          = alicloud_cen_instance.agents.id
  bandwidth_package_id = alicloud_cen_bandwidth_package.this.id
}
```

CEN 按跨地域带宽收费，上面那个 50 Mbps 的包大概 ~¥3,000/month。对于真正跑多地域流量的栈来说这钱花得值；要是为了“说不定哪天要用”那就纯属浪费。单项目 apply 多地域部署大概耗时 ~9 minutes total，因为两个区域的 RDS 是并行创建的。要是拆成两个项目： 14+ minutes 还得维护两份 state 文件。

## 我刻意省略了什么

starter stack 里我刻意省略了四件事：

- **CDN** 用于公开分发 artifact URL — `alicloud_cdn_domain` 能用，但大多数 Agent 都是通过带鉴权的自有网关来提供服务。
- **WAF** 挂在 ALB 前面 — 面向公网的 prod 环境必需，但 dev 栈用的是内网 ALB。
- **PrivateLink** 连接 DashScope — 规模化后能省掉 NAT 出口流量费，可通过 `alicloud_privatelink_*` 配置。上面提到的 NAT 账单降低 60% 就是靠它。
- **自定义域名 + SSL** — `alicloud_alb_listener` 支持 SSL 证书，但你得自己提供证书（或者用 ACM）。

这四个东西等基础跑通了都值得加，但绝不该放在第一天。我见过最多的坑就是团队想在 bootstrap 阶段把这四个全塞进去，结果三个配置踩坑，最后得出结论说“阿里云上的 Terraform 太难用”。其实不难。只是你试图一次性交付五个东西罢了。

## 接下来往哪走

八篇文章，一个栈，一个 Terraform 项目。你现在手里有：

- **一套能跑的 composition** — 五个 module，一次 apply，~31 resources， 7 minutes — 直接在阿里云上跑起来，第一天就具备可观测性、密钥管理和预算 guard 的 Agent 运行时。
- **一套可复用的 pattern** 通过 CI 推动 dev → staging → prod，带有 state 快照和 apply 后的 diff。
- **真实的成本账** — dev 环境 ¥2,060/month， prod 环境 ¥12,530/month — 这样在财务找你之前，你就能先跟他们谈平台成本。
- **一个多地域逃生通道** 不需要把 everything 扔掉重来。
- **一份 Day-2 实战手册** 持续回报的 pattern： state 备份、月度成本分摊、季度升级、年度架构 review。

接下来的路你自己选：

- **更多 Agent：** 往 `var.agent_quotas` 里加配置然后 `terraform apply`。合同条款稳固后，可以通过 Slack 表单实现自助服务。
- **更多 LLM 提供商：** 在 gateway module 的 `local.litellm_config` 里添加。 gateway 会把你的栈其余部分与具体选择解耦。
- **多地域：** 参考上面的 `agent-stack` module 模式。先从单地域开始，留个 `provider.alias` 占位符，这样迁移就是机械化的操作。
- **GitOps：** 把 `terraform apply` 包在 CI 里，由 PR review 和必需的 reviewer 把关。上面的 promotion workflow 就是起点。
- **Pulumi 或 Crossplane：** 资源图可以直接转换。等你真正需要 TypeScript 或者 Kubernetes-native 控制循环时再迁移，别提前。

最重要的一点是，你的基础设施现在躺在 git 里了。每次变更都可审查，每个环境都可复现，每笔成本都能归属到 workspace、 module，甚至单个 Agent。这才是 IaC 带给你的价值，也让在阿里云上交付 Agent 变成了一种可持续的工程实践，而不是无休止的救火。

感谢阅读本系列。 Starter repo (https://github.com/example/research-agent-stack) 随便你 fork。如果你基于它上线了真正的栈，我很想听听你改了什么、为什么这么改——模式就是这样磨锋利的。