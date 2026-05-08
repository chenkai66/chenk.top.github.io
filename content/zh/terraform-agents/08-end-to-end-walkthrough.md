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
description: "把七个 module 拼到一个仓库，跑一次 terraform apply，看一个完整的 Agent runtime——VPC、ECS、RDS、OpenSearch、OSS、LLM 网关、SLS 观测、成本告警——七分钟内起来。真实 apply 输出、module DAG、生产环境完整成本核算，以及可 fork 的起手仓库。"
disableNunjucks: true
translationKey: "terraform-agents-8"
---

这是第二到第七篇所有内容落到一处的文章。读完之后你会跑过一次 `terraform apply`，在阿里云上产出一个完整、可观测、有预算的 Agent runtime stack——约 31 个资源、~7 分钟实际时间、生产规格 ¥12,530/月全包。

我们要建的 stack：

![research-agent-stack：每一个盒子，一次 terraform apply](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig1_full_stack.png)

五层——edge、compute、memory、platform、ops——由本系列做出的 module 组合而来。底层用到 11 个阿里云产品：VPC、ECS、ALB、OSS、RDS for PostgreSQL、OpenSearch、KMS、SLS、ARMS、CloudMonitor，以及 DashScope（LLM 提供方，通过网关访问）。

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

顶层 8 个 `*.tf`，`modules/` 下 5 个 module，`env/*.tfvars` 装环境特定值，`secrets/secrets.auto.tfvars` 装不入 git 的密钥。这是我每个项目都用的布局——无聊就是好。一条不让步的规矩：`secrets/` 必须从第一次 commit 就在 `.gitignore` 里。我清理过的所有密钥泄漏事故，都能追到"第 50 次 commit 才补上 gitignore"这种事上。

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

五个 module 调用。每个 module 把*前一个* module 的输出当输入——`module.compute` 读 `module.vpc`、`module.storage`、`module.gateway`、`module.observability`。这种依赖接线就是 Terraform 用来构建 apply DAG 的方式：

![Terraform module 依赖 DAG](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/08-end-to-end-walkthrough/fig2_module_dag.png)

VPC 和 KMS 在最上——它们没有依赖。Storage 和 gateway 依赖 VPC + KMS，但相互独立，所以 Terraform 并行建。Compute 依赖前面三个，因为 cloud-init 模板需要它们的 endpoint。Observability 资源在最后扇出，因为它们引用了 compute 的 SG ID。

`local.is_prod` 三元式就是整个 promotion 策略的全部实现：prod 拿 HA RDS、两台网关、三台 Agent ECS、¥800 成本上限、跨区 DR；dev 拿最小可用规格。同一套 module，不同尺寸，没有"环境特例"代码要维护。

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

`sensitive = true` 让 Terraform 不在 plan/apply 输出里打印值。值还是会进 tfstate（这就是为什么第二篇我们给 OSS state bucket 单独建了一把 KMS CMK 做加密）。

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
# secrets/secrets.auto.tfvars （已加入 gitignore）
llm_keys = {
  "dashscope-prod" = "sk-DS-XXXXXXXXXXXXXXXXX"
  "openai-prod"    = "sk-XX-XXXXXXXXXXXXXXXXX"
  "anthropic-prod" = "sk-ant-XXXXXXXXXXXXXXXXX"
  "deepseek-prod"  = "sk-DEEPSEEK-XXXXXXXXX"
}
```

`*.auto.tfvars` 文件不需要 `-var-file` 就会自动加载，所以 `secrets.auto.tfvars` 自动生效，而 `env/dev.tfvars` 按 workspace 显式传入。两文件分工，避免 `terraform.tfvars` 这种隐式默认带来的歧义。

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

- **0–60s：** VPC、vSwitch、NAT、EIP、KMS 密钥——快资源
- **60–380s：** RDS（5 分钟）、OpenSearch（5.5 分钟）、ECS（~2 分钟）、网关（~1.5 分钟）——全部并行，被最慢的卡住
- **380–460s：** cloud-init 部署 Agent 应用、observability 资源、告警

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
  + total_estimated_cost  = "~¥2060/月（dev 规格）"

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
total_estimated_cost = "~¥2060/月（dev 规格）"
```

这就是一个完整的 Agent stack。ALB endpoint、网关 URL、SLS 看板 URL——任何一个粘进浏览器都能用。`total_estimated_cost` 这个 output 是 `outputs.tf` 里基于 workspace 三元式算出来的，所以 plan 看到的数字和最后的账单基本对得上（误差约 10%，我多年的经验值）。

## Day-2 运维

栈起来了，然后呢？这些是我在每个长期运行的栈上都会跑的操作——文章不会告诉你的，但 on-call 会。

### 加新 Agent

1. 在 `dev.tfvars` 里给 `var.agent_quotas` 加一项
2. `terraform apply -var-file=env/dev.tfvars`
3. 网关 module 里的 `null_resource` 自动生成新的 LiteLLM key
4. 用新的 `LITELLM_API_KEY` 环境变量部署 Agent 代码

端到端约 30 秒。我第一次把这套交付出去之后，产品同学问能不能用钉钉表单自助接入新 Agent。Terraform 这层契约稳了之后，那个表单半天就能搭。

### 横向扩容

改 module 调用里的 `ecs_count`（或 `tfvars`）。`terraform apply` 拉新实例、挂到 ALB，旧实例全程健康（`create_before_destroy`）。零停机。一次半夜两点突发流量，我用这一行从 3 台扩到 12 台，没掉一个请求。

### 销毁 dev

```bash
terraform workspace select dev
terraform destroy -var-file=env/dev.tfvars
```

prod 上会失败，因为类生产资源设了 `deletion_protection = true`，bootstrap state bucket 设了 `prevent_destroy = true`。这是有意的。dev 里 `deletion_protection = local.is_prod`，只在 prod 启用——`terraform destroy` 能跑。

> 永远在 `terraform destroy` 之前先 `terraform plan -destroy`。看 plan 输出，确认要销毁的资源数和你想的对得上。我亲眼见过工程师忘了切 workspace，把 `staging` 销了。一位资深后端被 PagerDuty 叫起来花了六小时重建数据。

### 推进 dev → staging → prod

文章里写的是 `terraform workspace select prod && terraform apply`。第一天能用。第三个月就成了大多数生产事故的源头——dev → prod 会暴露没人预料到的差异。

我在真实项目里跑的推进流水线分四步。每步几分钟，整套累计多花一小时。回报：三年下来这套流程大概帮我避免了 30 多次故障。

**第一步：源 workspace state 打快照。** 任何推进之前，先备份源 workspace 的 state。如果 prod 出了问题而 dev 正常，你至少能比对：

```bash
terraform state pull > /tmp/dev-state-$(date -Iseconds).json
aliyun oss cp /tmp/dev-state-*.json oss://ck-tfstate-archive/snapshots/
```

State 快照很小（通常 <1MB），归档桶配了 30 天后转冷归档的生命周期。整个历史每月几毛钱。

**第二步：CI 里基于验证过的 commit 算 prod plan。** 永远不要把没测过的代码推到 prod。在 dev 稳定跑了一周的那个 commit，才有资格生成 prod 的 `plan`：

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
      - name: 在目标 workspace 生成 plan
        env:
          TF_WORKSPACE: ${{ inputs.to_workspace }}
        run: |
          terraform init
          terraform plan -var-file=env/${{ inputs.to_workspace }}.tfvars \
            -out=tfplan-promote 2>&1 | tee promote-plan.txt
      - name: 推送到钉钉供人工审核
        run: |
          curl -X POST "$DINGTALK_WEBHOOK" \
            -d "{\"text\":{\"content\":\"推进计划 ${{ inputs.from_workspace }}→${{ inputs.to_workspace }} 待审核\"}}"
```

Plan 推到值班的钉钉。重点看任何与 dev 不一致的部分。如果计划显示某些资源要重建、但 dev 里没有重建——停下来，先排查。这种情况几乎都是 workspace 条件 bug 或者 tfvars 拼写错误。

**第三步：apply、冒烟、解锁。** 真正的 prod apply 由 GitHub Environment 把关，需要指定评审者批准（第六篇配过）。apply 成功后，切流量之前先跑一个冒烟测试：

```bash
gateway=$(terraform output -raw gateway_url)
reply=$(curl -s -X POST $gateway/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"ping"}]}' \
  | jq -r .choices[0].message.content)

[[ -n "$reply" ]] || { echo "冒烟测试失败"; exit 1; }
```

5 秒钟的冒烟能拦住"我把网关搞挂了"这一类错误的扩散。如果失败了，apply 技术上算成功，但你可以用第一步的快照回滚。

**第四步：跨 workspace 对比 output。**

```bash
diff <(terraform workspace select staging && terraform output -json) \
     <(terraform workspace select prod && terraform output -json) \
     | head -100
```

预期差异：实例数、RDS HA 标志、DR 区域。预期之外的差异：都得查——往往暴露出第三步没捕获的 tfvars 拼写错或 workspace 条件 bug。

### 季度依赖升级

每季度一次：把 `alicloud` provider、所有开源 module、Terraform 本身各升一个 minor 版本。dev 里跑 plan，apply，泡一周，推到 prod。这种纪律让你在 CVE-2027-XXX 爆出来时不至于落后三年，被迫一个周末跨六个版本紧急升级。

### State 跨区备份

State OSS bucket 在 cn-shanghai。如果 cn-shanghai 出区域级故障，你连其他区的 Terraform 都跑不了。给 state bucket 本身加一个跨区复制到 cn-beijing，每月 ¥10，最坏情况救命：

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

### 月度按 Agent 成本归因

网关记录每个 Agent 的成本（第七篇）。月底按 Agent 汇总，发到团队群："研究 Agent: ¥3,200，客服 Agent: ¥800，代码 Agent: ¥4,100"。把成本变具体，工程师看到自己 Agent 上"成本榜"会主动优化。我带过三个团队，从开始这个实践到 LLM 账单减半，平均两个月——不需要任何自上而下的指标。

### 年度架构评审

每年一次，遍历整个 `terraform state list`，逐个问：这个还需要吗？有些是历史遗留——从未删除的 dev 集群、升级前的 v15 RDS。提交清理 PR 销毁这些无用资源，是我做过 ROI 最高的 Terraform 工作——通常每年省下账单的 10%–15%。

## 接 Agent 代码

栈是*平台*。Agent 自己来自你的代码仓库（`var.agent_repo_url`），ECS 启动时 cloud-init 拉下来部署。Agent 代码要遵守的最小契约：

```python
# 这些值由 cloud-init 设置的环境变量提供
LLM_GATEWAY_URL    = os.environ["LLM_GATEWAY_URL"]    # http://alb.../v1
LITELLM_API_KEY    = os.environ["LITELLM_API_KEY"]    # 每个 Agent 独有的 key
DATABASE_URL       = os.environ["DATABASE_URL"]       # postgres://...
VECTOR_ENDPOINT    = os.environ["VECTOR_ENDPOINT"]    # OpenSearch HTTP
ARTIFACTS_BUCKET   = os.environ["ARTIFACTS_BUCKET"]   # OSS bucket 名
SLS_PROJECT        = os.environ["SLS_PROJECT"]
SLS_LOGSTORE       = os.environ["SLS_LOGSTORE"]
ARMS_OTLP_ENDPOINT = os.environ["ARMS_OTLP_ENDPOINT"]
```

所有值都来自 Terraform 的 output。Agent 代码形态上保持云无关——只读环境变量——但运行时被完整接入阿里云栈。有人问"怎么迁到 AWS"，答案是：换 module，Agent 代码不动。契约就是这串环境变量。

## 成本核算——dev 与 prod

Dev（低流量、单可用区、无 HA）：

| 组件                    | 月费 |
|-------------------------|-----:|
| VPC + NAT + EIP         | ~¥150 |
| ECS x1（`ecs.c7.large`）| ~¥250 |
| RDS Postgres（小型）    | ~¥350 |
| OpenSearch 向量         | ~¥800 |
| OSS（10 GB 标准）       | ~¥2 |
| LLM 网关 ECS x1         | ~¥150 |
| ALB（小型）             | ~¥50 |
| SLS + ARMS              | ~¥300 |
| KMS                     | ~¥10 |
| **dev 总计**            | **~¥2,060/月** |

Prod，全 HA、跨区 DR、真实流量——这是财务问"AI Agent 平台到底花多少钱"时你给出的数字：

| 层           | 资源                                    | 规格                             | 月费（¥） |
|--------------|----------------------------------------|----------------------------------|---------:|
| 网络         | VPC、vSwitch、路由表、KMS              | 三可用区，3 把 CMK              |       10 |
| 网络         | 增强型 NAT 网关 + EIP                  | 包年 + 1 TB 出站                |      920 |
| 计算         | ECS x3（`ecs.c7.xlarge` 4核/8GB）      | 3 台，每台 80 GB ESSD           |     1380 |
| 计算         | LiteLLM 网关 ECS x2                   | `ecs.c7.large` 2核/4GB          |      450 |
| 计算         | 标准版 ALB                             | 1 个，公网                      |      180 |
| 存储         | RDS Postgres HA（`pg.x4.large.2c`）    | 200 GB ESSD + 备机              |     2200 |
| 存储         | OpenSearch 向量（中等规格）            | 50 文档大小、80 计算单元        |     1800 |
| 存储         | OSS（500 GB 标准 + 生命周期）          | 主标准、少量低频                |      100 |
| 存储         | OSS DR 副本（cn-beijing）              | 500 GB 低频                     |       60 |
| 密钥         | KMS Secrets Manager                    | 8 个密钥，5 万次解密/月         |       50 |
| 可观测       | SLS                                    | 30 GB 写入、保留 90 天          |      450 |
| 可观测       | ARMS APM                               | 1 个环境，5000 万 span          |      600 |
| 可观测       | CloudMonitor                           | 主机指标 + 20 自定义            |       30 |
| **小计——基础设施**                                                                  | **8,230** |
| LLM API      | DashScope Qwen-max                     | 5000 万输入、1200 万输出 token  |     3500 |
| LLM API      | Anthropic / OpenAI 兜底                | 500 万输入、100 万输出 token    |      800 |
| **小计——LLM**                                                                       | **4,300** |
| **prod 总计 / 月**                                                                   | **¥12,530** |

从真实账单里得出的四点观察：

- **OpenSearch + RDS 占基础设施约 31%。** 预算紧的话，纯 pgvector（去掉 OpenSearch）每月省 ~¥1,800，代价是混合检索变慢。向量规模 100 万以下值得这么做。
- **NAT 出站是个意外大头。** 一个项目里我把 DashScope 调用切到 PrivateLink，NAT 账单降了 60%。任何规模值得做的配置改动。
- **这个体量下 LLM 占 35%。** 流量 ×10 之后 LLM 会到 70%–80%。比基础设施更早成为最重要的成本杠杆，而抓手就是网关里的每 Agent 配额。
- **可观测性占基础设施 10%。** 这是合理比例——低于 5% 是监控不够，高于 20% 是采得太多。每月审一次 SLS 写入量。

财务视角：单次会话成本 = ¥12,530 / 月会话数。10 万次/月时，每次 ¥0.125；1000 次/月时，每次 ¥12.50——这就到了"是否还值得自己跑"的临界点，要么提量、要么并到别人的平台。

## 进阶：从一个项目跑多区

服务中国和东南亚用户的 Agent，最终会需要 `cn-shanghai`（或 `cn-beijing`）和 `ap-southeast-1`（新加坡）。朴素答案是两个独立的 Terraform 项目。更好的答案是 provider 别名加每区一个 module 实例：

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
  primary_endpoints = module.stack_shanghai.endpoints   # 跨区复制用
}
```

`agent-stack` module 是第三到七篇做的全套打包。`is_primary` 控制 RDS 是 master 还是只读副本、OSS 是源还是复制目标等。

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

CEN 按跨区带宽计费——上面 50 Mbps 包 ~¥3,000/月。真服务多区流量值；"我哪天可能"过度了。单项目多区一次 apply：~9 分钟，因为两区 RDS 并行 provision。两个独立项目：14+ 分钟，还要带两套 state。

## 我暂时省略的部分

四样我特意没放进起手栈：

- **CDN** 用于公开分发产物 URL——`alicloud_cdn_domain` 能用，但大多数 Agent 通过自己带鉴权的网关分发产物。
- **WAF** 部署在 ALB 前——公网生产必备，但 dev 用的是内网 ALB。
- **PrivateLink** 连 DashScope——大规模时能显著省 NAT 出站，配置在 `alicloud_privatelink_*`。前面说的 NAT 降 60% 就是这条。
- **自定义域名 + SSL**——`alicloud_alb_listener` 支持 SSL 证书，但你得自己带证书（或用 ACM）。

四样基础跑通后都值得加。第一天都不要加。我见得最多的错误是团队把这四样一股脑塞进 bootstrap 栈，三样卡在配置上，然后得出"Terraform 在阿里云上很难"的结论。不难。是你一次性想干五件事。

## 接下来该做什么

八篇文章，一个栈，一个 Terraform 项目。你现在拥有：

- **一套能跑的组合**——五个 module、一次 apply、~31 资源、7 分钟——在阿里云上交付一个第一天就带可观测、密钥管理、预算护栏的 Agent runtime。
- **一套可复用的推进模式**：dev → staging → prod 通过 CI 推进，带 state 快照和 apply 后对比。
- **真实的成本数字**——dev ¥2,060/月，prod ¥12,530/月——可以在财务找你之前先把这个对话讲清楚。
- **多区扩展的逃生通道**，不需要推倒重来。
- **一份 Day-2 Playbook**：state 备份、月度成本归因、季度升级、年度架构评审，长期持续回报的几个动作。

接下来由你选：

- **加 Agent：** 改 `var.agent_quotas` + `terraform apply`。契约稳了之后用钉钉表单做自助。
- **加 LLM 提供方：** 网关 module 里改 `local.litellm_config`。网关把这层选择从其他模块抽离了。
- **多区域：** 上面 `agent-stack` module 那套。先在单区写好 `provider.alias` 占位，迁移就是机械动作。
- **GitOps：** 把 `terraform apply` 包进 CI，加 PR 评审和必选 reviewer。上面的 promotion workflow 就是起点。
- **Pulumi 或 Crossplane：** 资源图直接平移。等到你真需要 TypeScript 或者 K8s 原生控制环时再迁，不要更早。

最重要的一点：你的基础设施已经在 git 里了。每次变更可审，每个环境可复现，每一笔成本都能归因到 workspace、module、有时候到单个 Agent。这就是 IaC 的价值，也是在阿里云上交付 Agent 能成为一项可持续工程实践、而不是一场永无止境救火的根本原因。

感谢读完整个系列。起手仓库（https://github.com/example/research-agent-stack）随时可 fork。如果你基于它落地了一个真实栈，告诉我你改了什么、为什么——这就是模式变得更锋利的方式。
