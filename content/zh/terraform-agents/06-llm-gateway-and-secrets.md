---
title: "Terraform 实战（六）：LLM Gateway 与密钥管理"
date: 2026-03-22 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - KMS
  - API Gateway
  - LLM
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 6
description: "把所有 LLM 访问收敛到一个网关：按 Agent 限流、请求落 SLS 日志、KMS 之外不留密钥。Terraform 配 API Gateway + ECS 上自托管 LiteLLM，DashScope/OpenAI/Anthropic 的 key 通过 KMS Secrets Manager 自动轮转。"
disableNunjucks: true
translationKey: "terraform-agents-6"
---
我在很多还没成熟的 Agent 架构里总看到一个通病：每个 Agent 都在自己的 `.env` 文件里存一份 `OPENAI_API_KEY`。有时候是同一个 key，有时候不一样，甚至还有同事原型阶段留下的个人密钥。等到账单来了，没人说得清是哪个 Agent 烧了多少 token；一旦密钥泄露（迟早的事），你就得在十几个 `.env` 文件里打地鼠。

真正让我警醒的是两年前的一件事。有个外包周五结束三个月的合同， laptop 带走了，结果周二 DashScope 账单报警，显示有 1200 万 `qwen-max` token 的流量来自一个陌生 IP。他个人 API key——当初复制粘贴到侧边项目里的——还留在我们 Agent 的 `.env` 里。轮换该密钥耗时六小时，涉及三名工程师、四个代码仓库、两条 CI 流水线和一条迅速失控的 Slack 讨论线程。这类事故，绝不能再发生。

这篇文章旨在解决这个问题。我们构建了一个 **LLM 网关**，实现了以下目标：

- 把所有厂商的密钥统一收进 KMS Secrets Manager（一个 bucket、一套 ACL、统一的轮换节奏）
- Agent 通过 RAM 颁发的短期 Token 进行认证，机器上没有静态 AK
- 限制每个 Agent 的每分钟请求数（QPM）和每日 token 上限，防止死循环导致单日费用飙升（最高达 800 元）或季度预算耗尽。
- 所有请求记入 SLS，方便取证、成本分摊和 SOC-2 审计。
- 轮换密钥无需重启或重新部署任何 Agent，一个 PR 和一次 apply 就能搞定。

两天内即可完成部署，长期收益显著。

![Terraform for AI Agents (6): LLM Gateway and Secrets Management — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_1.png)

## 架构形态

![Centralised LLM gateway: one egress, one quota, one audit log](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig1_gateway_topology.png)

架构上， Agent 在左，模型厂商在右，网关居中充当代理层。每个 Agent 发往 "LLM" 的 HTTP 请求实际上都会先到达网关，再由网关统一分发请求至对应厂商、注入正确密钥、执行配额控制并记录调用结果。

实现方案主要有两种选择：

1. **Aliyun API Gateway 加自定义后端** —— 最托管，配额策略配置最简单，原生集成 RAM。适合路由逻辑简单的场景：“一个模型，一个厂商，只管限流”。
2. **ALB 后面自托管 LiteLLM on ECS** —— 最灵活，支持长尾厂商（DeepSeek、 Moonshot、 Zhipu、你自己微调的 PAI 端点），更容易扩展成本追踪和跨厂商 fallback。

我们建议根据路由复杂度来选择。如果是纯代理加配额， API Gateway 就足够了。但如果是多厂商路由，还需要带预算 guard 和熔断机制——这通常是大多数团队在半年内会进入的阶段——此时 LiteLLM on ECS 更具优势。本文剩余部分将基于 LiteLLM 展开，因为 80% 的团队都需要这种方案。

## 第一步：把所有密钥存进 KMS Secrets Manager

第一条铁律：厂商密钥绝不出现在 `.env` 文件里，不在 `provider {}` 块里，不在 Agent 代码里，也不以明文出现在 tfstate 里。它们只活在 KMS Secrets Manager 里，网关启动时通过 STS 拉取。

![Secure vault for managing API keys and credentials](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/wanxiang_secret_vault.png)


```hcl
locals {
  llm_secrets = {
    "dashscope-prod"  = "DashScope (Bailian) API key"
    "openai-prod"     = "OpenAI API key"
    "anthropic-prod"  = "Anthropic API key"
    "deepseek-prod"   = "DeepSeek API key"
  }
}

resource "alicloud_kms_secret" "llm" {
  for_each = local.llm_secrets

  secret_name              = each.key
  secret_data              = var.llm_keys[each.key]   # passed via -var or env
  version_id               = "v1"
  description              = each.value
  encryption_key_id        = module.vpc.kms_keys["secrets"]
  rotation_interval        = "30d"
  enable_automatic_rotation = false   # we rotate by updating secret_data
  recovery_window_in_days  = 7
}
```

密钥本身通过 `var.llm_keys` 传入——用 `-var-file=secrets.auto.tfvars`（已 gitignore）或者 CI secret 里的 `TF_VAR_llm_keys='{...}'` 设置。它们绝不会出现在你的代码仓库里。

我们来算一笔账： KMS Secrets Manager 在上海地域的费用约为每个密钥每月 0.4 元，外加每万次 API 调用 0.03 元。四个厂商密钥，两台网关机器每小时拉取两次，每月账单几乎是忽略不计——不到 10 块钱。阿里云默认提供的服务密钥免费，仅当使用客户自主创建和管理的 CMK （客户主密钥）时，才按每个每月 1 元计费。别让"KMS 听起来很贵”成为你继续用 `.env` 文件的理由。

> **实战建议：** 轮换厂商密钥时，修改 `secret_data` 并 bump `version_id`。 KMS 会在恢复窗口期内保持旧版本有效，确保进行中的请求不受影响；新启动的网关实例则会拉取并使用新版本。把这个流程写成 PR 形式，方便审计。

## 第二步：网关 assumable 的 RAM Role

网关的 ECS 或函数需要权限读取这些密钥——而且**只能**读这些：

```hcl
resource "alicloud_ram_role" "gateway" {
  name = "agent-gateway-${terraform.workspace}"

  assume_role_policy_document = jsonencode({
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = ["ecs.aliyuncs.com"]
      }
    }]
    Version = "1"
  })
}

resource "alicloud_ram_policy" "gateway_kms" {
  policy_name = "agent-gateway-kms-${terraform.workspace}"

  policy_document = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = [for s in alicloud_kms_secret.llm : s.arn]
      }
    ]
  })
}

resource "alicloud_ram_role_policy_attachment" "gateway_kms" {
  policy_name = alicloud_ram_policy.gateway_kms.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.gateway.name
}
```

这里的设计有三个关键点：

- **资源级权限控制。** 只针对这些特定密钥，而不是对 `*` 开放 `kms:GetSecretValue`。即使网关机器被攻破，攻击者也无法横向访问其他 KMS 密钥——例如账单密钥、 RDS 密码、 OSS 存储桶等，均保持隔离状态。
- **无长期 AK。** 角色由 ECS 实例通过 metadata service 假设。磁盘上、环境变量里、 cloud-init 中零静态凭证。
- **`kms:Decrypt` 是必须的。** 即使只需读取密钥，也必须显式声明该权限——因为 KMS 在静态存储时已对密钥加密。忽略此配置，是网关启动后每次 fetch 均返回 401 的最常见原因。

## 第三步：在 ECS 上部署 LiteLLM

LiteLLM 是目前我使用体验最好的开源 LLM 网关：前端兼容 OpenAI API 格式，后端可对接多家厂商的协议。在 ECS 上自托管 keeps things flexible：

```hcl
resource "alicloud_instance" "gateway" {
  count = 2  # two for HA, fronted by ALB

  instance_name        = "llm-gateway-${terraform.workspace}-${count.index + 1}"
  image_id             = data.alicloud_images.ubuntu.images[0].id
  instance_type        = "ecs.c7.large"
  availability_zone    = "cn-shanghai-${count.index == 0 ? "l" : "m"}"

  vswitch_id      = module.vpc.private_vswitch_ids[count.index]
  security_groups = [module.vpc.agent_runtime_sg_id]   # same SG; gateway is part of the runtime tier

  role_name = alicloud_ram_role.gateway.name           # gateway assumes this role

  system_disk_category = "cloud_essd"
  system_disk_size     = 40

  user_data = base64encode(templatefile("${path.module}/gateway-init.sh", {
    config_b64 = base64encode(local.litellm_config)
    sls_project = alicloud_log_project.agents.name
    sls_logstore = alicloud_log_store.gateway_requests.name
  }))

  tags = { Role = "llm-gateway" }
}

locals {
  litellm_config = yamlencode({
    model_list = [
      {
        model_name = "qwen-max"
        litellm_params = {
          model     = "dashscope/qwen-max-2026-01-15"
          api_key   = "os.environ/DASHSCOPE_API_KEY"
        }
      },
      {
        model_name = "claude-opus"
        litellm_params = {
          model     = "anthropic/claude-opus-4.7"
          api_key   = "os.environ/ANTHROPIC_API_KEY"
        }
      },
      {
        model_name = "gpt-4o"
        litellm_params = {
          model     = "openai/gpt-4o-2026-01-15"
          api_key   = "os.environ/OPENAI_API_KEY"
        }
      }
    ]
    general_settings = {
      master_key = "os.environ/LITELLM_MASTER_KEY"
      database_url = "os.environ/DATABASE_URL"
    }
  })
}
```

两台 `ecs.c7.large` —— 2 vCPU, 4 GB 内存 —— 轻松扛住 200+ QPS 的纯代理流量。 LiteLLM 属于异步 I/O 密集型服务， CPU 使用率通常不超过 30%。别配大了。如果流量有突发，放进弹性伸缩组，让云监控在 CPU 持续超过 60% 时加节点。

`gateway-init.sh` 负责启动流程：

```bash
#!/bin/bash
set -euxo pipefail

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl jq

# Pull provider keys from KMS via instance role (no AK needed)
TOKEN=$(curl -s http://100.100.100.200/latest/meta-data/ram/security-credentials/agent-gateway-${ENV})
ACCESS_KEY_ID=$(echo $TOKEN | jq -r .AccessKeyId)
ACCESS_KEY_SECRET=$(echo $TOKEN | jq -r .AccessKeySecret)
SECURITY_TOKEN=$(echo $TOKEN | jq -r .SecurityToken)

# Use the Aliyun KMS CLI (or Python SDK) to fetch each key
pip install alibabacloud-kms20160120
export DASHSCOPE_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('dashscope-prod'))")
export OPENAI_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('openai-prod'))")
export ANTHROPIC_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('anthropic-prod'))")

# Write LiteLLM config
mkdir -p /etc/litellm
echo "${config_b64}" | base64 -d > /etc/litellm/config.yaml

# Install and run LiteLLM under pm2
pip install 'litellm[proxy]'
npm install -g pm2
pm2 start --name llm-gateway -- litellm --config /etc/litellm/config.yaml --port 4000
pm2 save
pm2 startup systemd -u root --hp /root
```

每个实例现在都在 4000 端口运行网关，所有厂商密钥加载进进程环境变量——绝不在磁盘上落地。前面的 ALB 负责分发：

```hcl
resource "alicloud_alb_load_balancer" "gateway" {
  vpc_id              = module.vpc.vpc_id
  address_type        = "Intranet"
  load_balancer_name  = "llm-gateway-${terraform.workspace}"
  load_balancer_edition = "Standard"

  zone_mappings {
    vswitch_id = module.vpc.private_vswitch_ids[0]
    zone_id    = "cn-shanghai-l"
  }
  zone_mappings {
    vswitch_id = module.vpc.private_vswitch_ids[1]
    zone_id    = "cn-shanghai-m"
  }
}

resource "alicloud_alb_server_group" "gateway" {
  vpc_id            = module.vpc.vpc_id
  server_group_name = "llm-gateway"
  protocol          = "HTTP"
  health_check_config {
    health_check_enabled = true
    health_check_path    = "/health"
    health_check_protocol = "HTTP"
  }
  servers = [
    for inst in alicloud_instance.gateway : {
      server_id = inst.id
      port      = 4000
      weight    = 100
    }
  ]
}

resource "alicloud_alb_listener" "gateway" {
  load_balancer_id     = alicloud_alb_load_balancer.gateway.id
  listener_port        = 80
  listener_protocol    = "HTTP"
  default_actions {
    type = "ForwardGroup"
    forward_group_config {
      server_group_tuples {
        server_group_id = alicloud_alb_server_group.gateway.id
      }
    }
  }
}
```

现在 Agent 只要访问 `http://<alb-id>.cn-shanghai.alb.aliyuncs.com/v1/chat/completions` 就能到达网关，完全看不到厂商密钥。 ALB 仅允许内网访问：不分配公网 IP，也不监听任何面向公网的端口（如 443）。如果 Agent 需要从 VPC 外部调用，走 bastion 或 CEN，绝不直连。
## 步骤 4：单 Agent 配额

LiteLLM 原生支持按 Key 配额。最干净的做法是用 Terraform 给每个 Agent 配一个 LiteLLM "virtual key"，独立设置 QPM 和 Token 预算。因为 LiteLLM 把这些存在自己数据库里，我们得在 apply 阶段调它的 API 来创建，用个 `null_resource` 就行：

![Per-agent quota policy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig3_quota_table.png)

```hcl
locals {
  agent_quotas = {
    "research-agent" = { qpm = 120, daily_tokens = 2000000, max_budget = 800 }
    "code-agent"     = { qpm = 60,  daily_tokens = 1000000, max_budget = 500 }
    "support-agent"  = { qpm = 300, daily_tokens = 3000000, max_budget = 600 }
    "schedule-agent" = { qpm = 10,  daily_tokens = 100000,  max_budget = 40  }
  }
}

resource "null_resource" "agent_keys" {
  for_each = local.agent_quotas

  triggers = {
    config_hash = sha256(jsonencode(each.value))
  }

  provisioner "local-exec" {
    command = <<-EOT
      curl -X POST http://${alicloud_alb_load_balancer.gateway.dns_name}/key/generate \
        -H "Authorization: Bearer ${var.litellm_master_key}" \
        -H "Content-Type: application/json" \
        -d '{
          "key_alias": "${each.key}",
          "rpm_limit": ${each.value.qpm},
          "max_budget": ${each.value.max_budget},
          "tpm_limit": ${each.value.daily_tokens / 1440}
        }'
    EOT
  }
}
```

我不太待见 `null_resource` + `local-exec` 这套组合，这通常是"Provider 里还没这个资源”时的逃生舱口。但它管用，而且为了这一个团队去写个自定义 Terraform Provider 代码量太大，不划算。要是 LiteLLM 哪天出了官方 Provider，换起来也就一天的事。

输出结果是每个 Agent 拿到独立的 `LITELLM_API_KEY` 环境变量，第 4 篇文章里的 cloud-init 脚本会读这个。配额超限会返回 `429 Too Many Requests`， Agent 端必须用指数退避处理——把这逻辑写进共享的 HTTP 客户端里，别指望每个 Agent 作者都能记住。

说说数字。`schedule-agent` 的上限设在每天 10 万 Token 和 40 块钱，看着挺低。确实低，但我是故意的。一个调度 Agent 要是突然飙到 200 万 Token，大概率是卡在规划循环里了，这时候硬截断比月底账单多出 3000 块惊喜要好得多。可将上限设为“Agent 过去 30 天日用量的 P99 值的 10 倍”，并每季度复查一次。

## 步骤 5：密钥轮换流程

![Terraform for AI Agents (6): LLM Gateway and Secrets Management — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_2.png)

把密钥放进 KMS Secrets Manager 的核心目的就是轮换：

![Secret rotation flow — KMS as single source of truth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig2_secret_rotation.png)

生命周期如下：

1. 你在 Terraform 里改 `secret_data`（或者调 KMS API），把 `version_id` 升到 `v2`
2. KMS 会让 `v1` 在轮换窗口期内保持活跃（默认 30 天）
3. Gateway 实例冷启动时重新拉取；现有实例继续用缓存值，直到下次刷新（每 15 分钟，在 `gateway-init.sh` 里配置）
4. 30 天后，`v1` 被禁用——谁还在用就会拿到 `InvalidSecretVersion`
5. 你通过 SLS 确认 `v1`  usage 为零，然后提升 `v2` 并退役 `v1`

对团队来说，把这写成 Runbook，哪怕没泄露也要每季度执行一次。存活超过一个季度的密钥默认视为过期（stale），应按低优先级安全事件处理。开篇那个外包商的故事就是因为没人轮换 DashScope 密钥，拖了 14 个月。看完这篇文章，那种情况就不可能发生了——就算你忘了， 30 天的窗口期也会逼着你处理。

## 步骤 6：明文到底还会怎么漏（以及怎么堵）

Step 1 保护的是静态存储（at rest）的密钥。如果你不小心，至少还有三个地方会漏明文，这三个坑我都见过项目栽过。

### Leak 1: terraform.tfstate

`alicloud_kms_secret.secret_data` 每次 apply 都会明文出现在 tfstate 里。哪怕变量设了 `sensitive = true`，*值* 还是活在 state JSON 里。缓解措施得分层：

1. **OSS Bucket KMS 加密**（第 2 篇）——已经做了。保护静态状态文件。
2. **OSS Bucket 访问策略**——`oss:GetObject` 只限 CI runner 角色，开发人员永远不行。
3. **用 `data` source 模式，别把明文写进 HCL。** 当密钥是带外创建的（比如由 HSM 轮换任务或 KMS 控制台创建）， Terraform 只读不写：

```hcl
data "alicloud_kms_secret" "openai" {
  secret_name = "openai-prod"
  version_id  = "ACSCurrent"   # always the current version
}

# Use it without the value ever entering tfstate's resource section
resource "alicloud_instance" "gateway" {
  user_data = base64encode(templatefile("${path.module}/init.sh", {
    # Don't pass the secret here — pass the secret name and have the box fetch it
    openai_secret_name = "openai-prod"
  }))
}
```

核心原则是： Terraform 只需知晓密钥的名称，无需掌握其具体值。运行时通过实例元数据从 KMS 拉取值。这是最重要的一条习惯，跟大多数教程教的正好相反。

### Leak 2: CI logs

如果变量设了 `sensitive = true`，`terraform plan` 输出会把敏感值标记为 `(sensitive value)`。*但是* 只针对变量本身——衍生出的资源属性不会自动标记。常见的失误：

```hcl
variable "openai_key" {
  type      = string
  sensitive = true
}

# This still leaks in plan output:
resource "alicloud_kms_secret" "openai" {
  secret_data = var.openai_key
  # plan shows: secret_data = (sensitive value)  ✓
}

# But this can leak:
output "gateway_config_url" {
  value = "https://gateway.example.com?key=${var.openai_key}"
  # plan shows the full URL with key ✗
}
```

把所有衍生自敏感值的 output 都标上 `sensitive = true`：

```hcl
output "gateway_config_url" {
  value     = "https://gateway.example.com?key=${var.openai_key}"
  sensitive = true
}
```

对于 tfvars 文件，加进 `.gitignore` *并且* 配置 CI 一旦提交就报错：

```yaml
# .github/workflows/no-secrets.yml
- name: check no secrets in repo
  run: |
    if git ls-files | grep -E '\.auto\.tfvars$|secrets/'; then
      echo "ERROR: secret files committed"; exit 1
    fi
```

### Leak 3: provider debug logs

`TF_LOG=DEBUG terraform apply` 是调试 Provider 问题最快的方法。也是把每个 API 请求响应——包括含密钥的请求体—— dump 到终端历史记录的最快方法。我见过两次 Slack 截图泄露，发生在两家不同的公司。

必须用 `TF_LOG` 时，重定向到权限受限的文件，别直接粘贴：

```bash
TF_LOG=DEBUG terraform apply 2> /tmp/tf.log
chmod 600 /tmp/tf.log
# review locally; never paste verbatim
shred -u /tmp/tf.log    # delete when done
```

更好的是用 `TF_LOG_CORE=DEBUG`（仅限 Terraform 核心），通常能隔离问题且不包含 Provider 请求体。
## 步骤 7： CI 中的 plan-review-apply 门禁

单个工程师玩票，用 `null_resource` 生成 LiteLLM 密钥没问题。但要是团队里有多个 Agent、多套环境，还得轮流 On-call，那就得上结构化的 CI 流水线，让人类在 Diff 层面做评审。这是我跑的 GitHub Actions 工作流：

```yaml
# .github/workflows/terraform-plan.yml
name: terraform-plan
on:
  pull_request:
    paths:
      - '**/*.tf'
      - '**/*.tfvars'
      - 'modules/**'

jobs:
  plan:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write   # to post the plan as a PR comment
    strategy:
      fail-fast: false
      matrix:
        workspace: [dev, staging, prod]
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.9.7
      - name: terraform fmt
        run: terraform fmt -check -recursive
      - name: terraform init
        env:
          ALICLOUD_ACCESS_KEY: ${{ secrets.ALICLOUD_AK_PLAN }}
          ALICLOUD_SECRET_KEY: ${{ secrets.ALICLOUD_SK_PLAN }}
        run: terraform init -input=false
      - name: terraform validate
        run: terraform validate -no-color
      - name: tflint
        uses: terraform-linters/setup-tflint@v4
      - run: tflint --init && tflint -f compact
      - name: tfsec security scan
        uses: aquasecurity/tfsec-action@v1.0.3
      - name: terraform plan
        id: plan
        env:
          TF_WORKSPACE: ${{ matrix.workspace }}
          ALICLOUD_REGION: cn-shanghai
          TF_VAR_dingtalk_webhook: ${{ secrets[format('DINGTALK_WEBHOOK_{0}', matrix.workspace)] }}
        run: |
          terraform plan -input=false -no-color -out=tfplan-${{ matrix.workspace }} \
            -var-file=env/${{ matrix.workspace }}.tfvars 2>&1 | tee plan.txt
      - name: post plan to PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('plan.txt', 'utf8').slice(0, 60000);
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `### Plan for ${{ matrix.workspace }}\n\`\`\`\n${plan}\n\`\`\``
            });
      - uses: actions/upload-artifact@v4
        with:
          name: tfplan-${{ matrix.workspace }}
          path: tfplan-${{ matrix.workspace }}
```

这套流程强制落实了三件事，平时人工评审容易扫一眼就放过：

- **`terraform fmt -check`** 拒绝未格式化的 HCL。再也不用在评审里扯“你跑 fmt 了吗”这种废话。
- **`tfsec`** 跑 Checkov 风格的扫描——标记公开桶、未加密卷、范围过大的 SG 规则。
- **Plan 作为 PR 评论发布** 意味着评审读的是*实际的 Plan 输出*，而不是“信我兄弟，这看起来没问题”。

对应的 Apply 工作流：

```yaml
# .github/workflows/terraform-apply.yml
on:
  workflow_dispatch:
    inputs:
      workspace:
        type: choice
        options: [dev, staging, prod]

jobs:
  apply:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: ${{ inputs.workspace }}   # GitHub Environment with required reviewers for prod
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: tfplan-${{ inputs.workspace }}
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
      - name: terraform apply
        env:
          ALICLOUD_ACCESS_KEY: ${{ secrets.ALICLOUD_AK_APPLY }}
          ALICLOUD_SECRET_KEY: ${{ secrets.ALICLOUD_SK_APPLY }}
        run: terraform apply -input=false -auto-approve tfplan-${{ inputs.workspace }}
```

GitHub 的 `environment` 机制把生产环境的 Apply 权限锁死，必须经过评审人批准——通常是另一个真人（比如 On-call 同事）点一下"approve"才能跑。 Apply 用的 RAM 角色跟 Plan 也不一样： Plan 只读， Apply 可写。典型的双钥启动流程。

这套流水线在我跑的一年裡抓到了五次真实事故：一个不小心把 `prevent_destroy = false` 翻了个面，一个会破坏 VPC 对等连接的 CIDR 重叠，一个忘了锁定的模块版本，一个掩盖了真实配置漂移的 `ignore_changes`，还有一个混进 PR 的安全组规则 `0.0.0.0/0`。每个都是在 PR 上留评论，合并前就修好了。

### 什么时候该升级到 Atlantis

GitHub Actions 是个好的起点。一旦跑 Terraform 的工程师超过五个，每个 PR 挂 Plan 评论、锁竞争、人工审批这些运维开销就开始让人头疼了。 Atlantis 是下一步：一个自托管的 Webhook 服务器，监听 PR，自动跑 `terraform plan`，评论 Plan，并且允许授权用户通过 `atlantis apply` 执行应用。

跟 Actions 比：

- Plan 跑*在你的 VPC 内部*——没必要给外部 Runner 访问 OSS State 桶的权限。
- 一个持久服务器持有锁，顺序执行 Apply——没有跨并发 PR 的竞争条件。
- 每项目配置（`atlantis.yaml`）让你能 scope 哪些目录被管理，带每目录的审批策略。

部署 Atlantis 本身就是个 `vpc-baseline` + `compute` 的活儿——一台 ECS，一个 ALB，一个 RAM 角色。在 10 人团队跑了两个月，吞吐量提升是实打实的： Plan 到 Apply 的周期从 25 分钟（Actions 队列 + 人工审批）降到了 8 分钟。

工程师少于 5 个我不推荐 Atlantis——运维成本划不来。超过 5 个，一个季度内就能回本。

> **实战建议：** 不管选哪种流水线，**单仓库，多环境**。别把生产环境 repo 从开发环境 fork 出来。单个 repo 配合 `env/dev.tfvars`、`env/staging.tfvars`、`env/prod.tfvars` 能保持跨环境的代码路径一致——这才是 IaC 的核心意义。按环境 fork 仓库是一种反模式，会侵蚀你当初追求的特性。

## 针对百炼 / DashScope 的特殊说明

在 LiteLLM 眼里， DashScope 就是个兼容 OpenAI 协议的端点。模型名是 `dashscope/qwen-max`、`dashscope/qwen-plus` 等。 API Key 就是你从 DashScope 控制台生成的那个。

如果你想用阿里云原生的待遇（比如用 STS 代替 API Key）， DashScope 在某些端点上支持基于 STS 的认证——但在 2026 年， API Key 路径仍然是标准做法，上面提到的通过 KMS 轮换 Key 是正确的运维模式。等 STS 成为默认（路线图建议是 2027 年），这篇文章的配置只需要翻个面就行；轮换纪律保持不变。

> **实战建议：** 在 LiteLLM 上设一个 `master_key`（`LITELLM_MASTER_KEY` 环境变量）。没有它，任何能触达网关的人都能给自己发 API Key。有了它，只有 master 能铸造 subordinate keys——而且 master 永远不出 Terraform 的变量空间。

## 这套方案带来了什么

读完这篇文章，你拥有了：

- 所有 Agent 调用"LLM"的统一 URL
- 添加新模型提供商的统一入口（编辑 `litellm_config`，`terraform apply`）
- 轮换任何提供商 Key 的统一入口（编辑 `var.llm_keys`，`terraform apply`）
- 统一日志流（下篇文章）展示每个请求、延迟、 Token 数、模型和 Agent
- 每个 Agent 的硬 QPM 和预算上限——跑飞了的循环每天最多花¥800，而不是整个月的预算
- 能抓住人类容易忽略的回归的 CI 流水线，配合生产环境的双钥启动
- 堵死了三个具体的明文泄露向量，不只是最明显的那个

网关是战略资产。我交付过的每个团队都在一个月内感谢了我——通常是第一次有人不小心把 API Key 提交到 git，结果发现轮换它只需要一个一行的 PR，而不是我经历过的那种六小时救火。

## 接下来是什么

第 7 篇是可观测性和成本控制： SLS 做日志， ARMS 做链路追踪， CloudMonitor 做指标，每日 LLM 花费超过阈值时 ping DingTalk 的预算报警，以及让你能看到“哪个 Agent 在烧我预算”的 SLS 驱动成本仪表盘。

第 8 篇是端到端 walkthrough，第 2 到 7 篇的所有内容作为一个 `terraform apply` 落地。