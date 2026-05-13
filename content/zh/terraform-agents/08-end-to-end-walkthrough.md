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
本系列第 2 至第 7 篇所构建的全部模块，最终在此完成整合。只需运行一次 `terraform apply`，你就能在阿里云上部署一套完整、可观测、带预算控制的 Agent 运行时栈——包含约 31 个资源，实际耗时约 7 分钟，生产环境规模下全包成本约为 ¥12,530/月。

我们要搭建的栈结构如下：

![research-agent-stack：每个框，一个 terraform apply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig1_full_stack.png)

共五层：边缘、计算、记忆、平台、运维——均由本系列此前构建的模块组合而成；底层依赖 11 款阿里云服务：VPC、ECS、ALB、OSS、RDS for PostgreSQL、OpenSearch、KMS、SLS、ARMS、CloudMonitor，以及通过网关调用的 DashScope（LLM 接入服务）。

## 项目结构

```text
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

![基础设施模块组合成完整的架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/wanxiang_module_composition.png)

顶层包含八个 `*.tf` 文件，`modules/` 目录下有五个模块，环境变量存放在 `env/*.tfvars`，密钥则隔离在 `secrets/secrets.auto.tfvars` 中且不纳入 Git。这是我每个项目的标准目录结构：略显刻板，但胜在稳定可靠。唯独 `secrets/` 目录必须从第一次提交起就被 `.gitignore` 忽略，这点我绝不妥协。我处理过的所有密钥泄露事件，根本原因都是团队未在项目初始化时配置 `.gitignore`，而是在后续（例如第 50 次提交）才临时补充。

## main.tf — 组合逻辑

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

这里调用了五个模块。每个模块都以*前一个*模块的输出作为输入——例如 `module.compute` 会读取 `module.vpc`、`module.storage`、`module.gateway` 和 `module.observability` 的输出。正是这些依赖关系，让 Terraform 能构建出 apply 执行的有向无环图（DAG）。

![Terraform 模块依赖 DAG](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig2_module_dag.png)

VPC 与 KMS 位于依赖链顶端，不依赖任何其他模块；Storage 与 Gateway 均依赖 VPC 和 KMS，但彼此独立，因此 Terraform 会并行创建；Compute 模块依赖前三者，因为其 cloud-init 模板需要引用它们输出的 endpoint 地址；Observability 资源最后部署，需引用 Compute 模块的安全组 ID。

`local.is_prod` 中的三元表达式就是整套环境升级策略，仅用三行代码：生产环境启用高可用 RDS、两个网关实例、三个 Agent ECS、¥800 成本上限以及跨地域灾备；开发环境则使用最小可行配置。模块完全相同，仅通过变量调节规模，无需为不同环境维护独立代码分支或条件逻辑。

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

`sensitive = true` 可防止 Terraform 在 plan 或 apply 输出中打印敏感值。不过这些值仍会写入 tfstate（这也是为什么我们在第 2 篇中使用独立的 KMS CMK 对 OSS state bucket 进行了加密）。

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

`*.auto.tfvars` 文件会被自动加载，无需 `-var-file` 参数，因此 `secrets.auto.tfvars` 会被自动拾取，而 `env/dev.tfvars` 则根据 workspace 显式传入。这种双文件模式避免了 `terraform.tfvars` 可能带来的歧义。

## 应用过程

```bash
cd research-agent-stack
terraform workspace select dev
terraform init
terraform plan -var-file=env/dev.tfvars -out=tfplan
# review plan output: ~31 resources to add
terraform apply tfplan
```

全新 apply 的实际耗时如下：

![实际应用时间线 — RDS/OpenSearch 占主导，其余并行执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig3_apply_timeline.png)

- **0–60 秒**：VPC、vSwitch、NAT、EIP、KMS 密钥等快速资源
- **60–380 秒**：RDS（约 5 分钟）、OpenSearch（约 5.5 分钟）、ECS（约 2 分钟）、Gateway（约 1.5 分钟）——这些资源并行创建，整体耗时由最慢的一项决定
- **380–460 秒**：通过 cloud-init 部署 Agent 应用、创建可观测性资源及告警

总耗时约 7 分钟，主要瓶颈在 RDS 和 OpenSearch 的创建。若无变更再次执行 apply，通常在 30 秒内完成，因为 Terraform 仅做差异比对。

一份精简后的 apply 输出如下：

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

至此，一套完整的 Agent 运行时栈已部署完毕。ALB 端点、网关 URL、SLS 仪表盘 URL 均可直接访问。`total_estimated_cost` 输出在 `outputs.tf` 中根据 workspace 条件动态计算，因此你在 plan 中看到的数字与账单基本一致（误差通常在 10% 以内，这是我长期实践的经验法则）。

## Day-2 运维操作

栈已就绪，接下来呢？以下是我对每个长期运行栈都会执行的操作——虽未在正文中详述，却是 on-call 工程师日常必备的实践。

![CI/CD 流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/wanxiang_cicd_pipeline.png)

### 添加新 Agent

1. 在 `dev.tfvars` 的 `var.agent_quotas` 中新增一条配置
2. 执行 `terraform apply -var-file=env/dev.tfvars`
3. 网关模块中的 `null_resource` 会自动创建一个新的 LiteLLM 密钥
4. 使用新的 `LITELLM_API_KEY` 环境变量部署你的 Agent 代码

整个过程约 30 秒。首次上线该模式时，产品团队曾提出能否通过 Slack 表单自助接入 Agent。一旦有了 Terraform 契约，实现这样的表单通常只需半天。

### 扩容

修改模块调用中的 `ecs_count`（或通过 `tfvars` 设置）。`terraform apply` 会先创建新实例并挂载到 ALB，旧实例在整个过程中保持健康（得益于 `create_before_destroy` 策略），实现零停机扩容。我曾在凌晨 2 点流量激增时，仅靠修改这一行配置，将 Agent 实例数从 3 扩容至 12。

### 销毁 Dev 环境

实验结束后：

```bash
terraform workspace select dev
terraform destroy -var-file=env/dev.tfvars
```

该命令在 prod 环境会失败，因为 bootstrap state bucket 上设置了 `deletion_protection = true` 和 `prevent_destroy = true`——这是有意为之。在 dev 环境中，`deletion_protection = local.is_prod`，因此仅在 prod 启用保护，`terraform destroy` 可正常执行。

> 执行 `terraform destroy` 前务必先运行 `terraform plan -destroy`，并仔细检查输出。被销毁的资源数量必须与预期一致。我曾目睹一位工程师因忘记切换 workspace 而误删 `staging` 环境，花了六小时并动用 PagerDuty 上的资深后端才重建数据。

### 从 Dev 晋升到 Staging 再到 Prod

文章中展示了 `terraform workspace select prod && terraform apply`。第一天这么操作没问题，但到了第三个月，大多数生产事故恰恰源于此——因为 dev → prod 会暴露许多未预料到的差异。

我在真实项目中采用的晋升流水线包含四步。每步仅需几分钟，累计增加约一小时的日历时间，却在过去三年中避免了至少 30 次线上故障。

**第一步：快照源状态**。晋升前，先复制一份源 workspace 的状态文件。若 prod 出现 dev 中未复现的问题，可快速对比：

```bash
terraform state pull > /tmp/dev-state-$(date -Iseconds).json
aliyun oss cp /tmp/dev-state-*.json oss://ck-tfstate-archive/snapshots/
```

状态快照通常小于 1 MB，归档桶配置了 30 天后转冷归档策略，整套历史记录每月成本仅 ¥0.x。

**第二步：在 CI 中基于已验证的 commit 计算 prod plan**。切勿晋升未经测试的代码。只有在 dev 环境稳定运行一周的 commit 才有资格生成 prod 的 plan：

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

该 plan 会发送给钉钉上的 on-call 工程师。他们会特别关注与 dev 环境 *不同* 的部分。若 plan 显示某些资源将被重建，而 dev 中并未重建——立即停止，调查原因。这通常是 workspace 条件逻辑错误或 tfvars 拼写问题。

**第三步：Apply、烟雾测试、再解封**。实际的 prod apply 通过 GitHub Environment 控制，并要求审批人（如第 6 篇所述）。Apply 成功后、流量切换前，先运行烟雾测试：

```bash
gateway=$(terraform output -raw gateway_url)
reply=$(curl -s -X POST $gateway/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"ping"}]}' \
  | jq -r .choices[0].message.content)

[[ -n "$reply" ]] || { echo "Smoke test failed"; exit 1; }
```

5 秒钟的烟雾测试能在错误扩散前捕获“网关不可用”类问题。若测试失败，尽管 apply 技术上成功，仍可借助第一步的快照回滚。

**第四步：Apply 后与 Staging 对比**。执行跨 workspace 差异分析：

```bash
diff <(terraform workspace select staging && terraform output -json) \
     <(terraform workspace select prod && terraform output -json) \
     | head -100
```

预期差异包括：实例数量、RDS HA 标志、DR 区域。其余任何差异均为意外，需深入调查——往往能发现 tfvars 拼写错误或未触发的 workspace 条件 bug。

### 季度模块依赖升级

每季度将 `alicloud` provider、所有开源模块及 Terraform 本身升级一个小版本。先在 dev 环境运行 plan，Apply 后观察一周，再晋升至更高环境。这种纪律性能避免在 CVE-2027-XXX 爆发时，因落后三年而被迫在一个周末紧急跨六个版本升级。

### 状态备份至不同 Region

当前状态存储在 cn-shanghai 的 OSS bucket 中。若该区域发生全域故障，你将无法执行任何 Terraform 操作（包括对其他区域）。每周将状态复制到 cn-beijing，每月成本仅 ¥10，却能在最坏情况下救命：

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

网关会记录每个 Agent 的成本（见第 7 篇）。月底按 Agent 汇总并发布至团队频道：“Research agent: ¥3,200，Support agent: ¥800，Code agent: ¥4,100”——让成本变得具体可见。当工程师的名字出现在榜单上时，他们会主动优化。我曾有三个团队在实施该做法两个月内将 LLM 账单削减一半，无需任何自上而下的指令。

### 年度架构对照 IaC 审查

每年遍历 `terraform state list`，逐项确认：我们是否仍需要此资源？有些是历史残留——从未删除的 dev 集群、已升级的 v15 RDS。提交销毁未用资源的 Cleanup PR 是我做过 ROI 最高的 Terraform 工作，通常可节省年度账单的 10–15%。

## 连接你的实际 Agent 代码

栈是*平台*，Agent 本身来自你的代码仓库（`var.agent_repo_url`），由 ECS 启动时的 cloud-init 自动部署。你的 Agent 代码需遵守的最小契约如下：

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

所有值均来自 Terraform outputs。Agent 代码在形态上保持云无关——仅读取环境变量——但在运行时完全接入阿里云栈。当有人问“如何迁移到 AWS？”时，答案很简单：替换模块，保留 Agent 代码不变。契约就是这份环境变量列表。

## 成本分析 — Dev 与 Prod

Dev 环境（低流量、单可用区、无 HA）成本估算：

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

Prod 环境（全 HA、跨区域 DR、真实流量）——这是财务询问“AI Agent 平台实际成本是多少？”时的标准答案：

| 层级        | 资源                                | 规格                        | 每月 (¥) |
|--------------|-----------------------------------------|-------------------------------|------------:|
| 网络      | VPC, vSwitch, RT, KMS                   | 3 可用区，3 CMKs                |          10 |
| 网络      | NAT Gateway (Enhanced) + EIP            | 预留 + 1 TB 出口        |         920 |
| 计算      | ECS x3 (`ecs.c7.xlarge` 4c/8g)          | 3 实例，每个 80 GB ESSD  |        1380 |
| 计算      | LiteLLM gateway ECS x2                  | `ecs.c7.large` 2c/4g          |         450 |
| 计算      | ALB Standard                            | 1 ALB，面向互联网        |         180 |
| 内存       | RDS Postgres HA (`pg.x4.large.2c`)      | 200 GB ESSD + 备机         |        2200 |
| 内存       | OpenSearch vector (medium)              | 50 文档大小，80 计算       |        1800 |
| 内存       | OSS (500 GB Standard + lifecycle)       | 主要是 Standard，部分 IA      |         100 |
| 内存       | OSS DR replica (cn-beijing)             | 500 GB IA                     |          60 |
| 密钥      | KMS Secrets Manager                     | 8 secrets，50k 解密/月    |          50 |
| 可观测性| SLS                                     | 30 GB 摄入，90 天保留      |         450 |
| 可观测性| ARMS APM                                | 1 环境，50M spans              |         600 |
| 可观测性| CloudMonitor                            | 主机指标 + 20 自定义      |          30 |
| **小计 — 基础设施**                                                                   |    **8,230** |
| LLM API      | DashScope Qwen-max                      | 50M 输入，12M 输出 tokens  |        3500 |
| LLM API      | Anthropic / OpenAI fallback             | 5M 输入，1M 输出 tokens    |         800 |
| **小计 — LLM**                                                                     |    **4,300** |
| **Prod 总计 / 月**                                                                 |   **¥12,530** |

基于真实账单的四点观察：

- **OpenSearch 和 RDS 合计占基础设施成本的 31%**。若严格控制成本，采用纯 pgvector 方案（移除 OpenSearch）每月可节省约 ¥1,800，代价是混合搜索变慢。向量规模低于 100 万时值得考虑。
- **NAT 出口费用常成意外大项**。某项目切换至 DashScope 的 PrivateLink 后，NAT 账单下降 60%。只要流量非 trivial，这点配置开销完全值得。
- **当前规模下 LLM 占总成本的 35%**。当流量增至 10 倍时，LLM 将占账单的 70–80%。此时，网关的 per-agent 配额会比基础设施更早成为关键成本杠杆。
- **可观测性占基础设施成本的 10%**。这是合理比例：低于 5% 说明监控不足，超过 20% 则可能过度采集。建议每月审计 SLS 摄入量。

用于财务评审时，单次会话成本为 ¥12,530 ÷ 月会话数。若月会话量为 10 万，则单次成本为 ¥0.125；若仅为 1 千，则高达 ¥12.50——此时应评估：是否值得为如此低频流量独立运行平台，还是应合并至他人服务。

## 更进一步：单项目实现多地域部署

若 Agent 需同时服务中国与东南亚用户，迟早要在 `cn-shanghai`（或 `cn-beijing`）和 `ap-southeast-1`（新加坡）部署。直观做法是维护两套独立 Terraform 项目，但更优方案是使用 provider alias 配合按地域实例化的模块：

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

`agent-stack` 模块即本系列第 3 至第 7 篇内容的打包整合。`is_primary` 参数控制 RDS 是主库还是只读副本、OSS 是源桶还是复制目标等。

跨地域通信需借助 CEN（Cloud Enterprise Network）实现 VPC 间私网互通：

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

CEN 按跨地域带宽计费，上述 50 Mbps 套餐月费约 ¥3,000。对真正承载多地域流量的栈而言物有所值；若仅为“未来可能用到”，则纯属浪费。单项目 apply 多地域部署总耗时约 9 分钟（因两地 RDS 并行创建）；若拆为两个项目，则需 14+ 分钟且需维护两份状态文件。

## 我刻意省略的内容

Starter stack 中我明确排除了四项功能：

- **CDN**：用于公开分发 artifact URL。`alicloud_cdn_domain` 可用，但多数 Agent 通过自带鉴权的网关提供服务。
- **WAF**：置于 ALB 前。公网 prod 环境必需，但 dev 栈使用内网 ALB。
- **PrivateLink**：连接 DashScope。规模化后可显著降低 NAT 出口费用，通过 `alicloud_privatelink_*` 配置。前述 NAT 账单下降 60% 正得益于此。
- **自定义域名 + SSL**：`alicloud_alb_listener` 支持 SSL 证书，但需自行提供（或使用 ACM）。

这四项功能在基础稳定后都值得添加，但绝不应放在第一天。我见过太多团队试图在 bootstrap 阶段一次性集成全部四项，结果在其中三项上踩坑，最终得出“阿里云 Terraform 太难用”的错误结论。其实并非难用，只是试图一次性交付太多罢了。

## 下一步方向

八篇文章，一个栈，一个 Terraform 项目。你现在拥有：

- **一套可运行的组合**：五个模块、一次 apply、约 31 个资源、7 分钟部署时间——在阿里云上直接跑起具备可观测性、密钥管理与预算防护的 Agent 运行时。
- **一套可复用的晋升模式**：通过 CI 实现 dev → staging → prod 演进，含状态快照与 apply 后差异比对。
- **真实的成本数据**：dev 环境 ¥2,060/月，prod 环境 ¥12,530/月——让你能在财务找上门前主动沟通平台成本。
- **一个多地域逃生通道**：无需推倒重来。
- **一份 Day-2 实战手册**：持续产生回报的操作模式，包括状态备份、月度成本分摊、季度升级与年度架构审查。

接下来的路径由你选择：

- **更多 Agent**：向 `var.agent_quotas` 添加配置并 `terraform apply`。契约稳固后，可通过 Slack 表单实现自助服务。
- **更多 LLM 提供商**：在网关模块的 `local.litellm_config` 中添加。网关会将栈其余部分与具体提供商解耦。
- **多地域部署**：参考上述 `agent-stack` 模块模式。先从单地域起步，预留 `provider.alias` 占位符，使迁移变为机械操作。
- **GitOps**：将 `terraform apply` 纳入 CI，由 PR 审核与指定审批人把关。前述晋升流程即是起点。
- **Pulumi 或 Crossplane**：资源图可直接转换。仅在真正需要 TypeScript 或 Kubernetes 原生控制循环时迁移，切勿提前。

最重要的是，你的基础设施现已纳入 Git。每次变更均可审查，每个环境均可复现，每笔成本均可追溯至 workspace、模块甚至单个 Agent。这正是 IaC 的核心价值，也让在阿里云上交付 Agent 成为可持续的工程实践，而非永无止境的救火行动。

感谢阅读本系列。Starter repo (https://github.com/example/research-agent-stack) 欢迎随意 fork。若你基于它上线了真实栈，我很期待听到你做了哪些改动及原因——模式正是这样不断打磨锋利的。
