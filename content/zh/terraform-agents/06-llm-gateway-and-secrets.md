---
title: "用 Terraform 给 AI Agent 上云（六）：LLM 网关与密钥管理"
date: 2026-03-22 09:00:00
tags:
  - Terraform
  - 阿里云
  - KMS
  - API 网关
  - LLM
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 6
description: "把所有 LLM 访问收敛到一个网关：按 Agent 限流、请求落 SLS 日志、KMS 之外不留密钥。Terraform 配 API Gateway + ECS 上自托管 LiteLLM，DashScope/OpenAI/Anthropic 的 key 通过 KMS Secrets Manager 自动轮转。"
disableNunjucks: true
translationKey: "terraform-agents-6"
---

不成熟的 Agent stack 有个常见模式：每个 Agent 自己 `.env` 文件里有一份 `OPENAI_API_KEY`。有时是同一份，有时不是，有时是某个同事原型阶段留下的个人 key。账单到了没人能说清是哪个 Agent 烧的 token，key 泄露的时候（一定会泄露）你在十几个 `.env` 文件之间打地鼠。

本篇结束这个状态。我们建一个 **LLM 网关**：

- 所有 provider key 在 KMS Secrets Manager 里
- Agent 通过短期 RAM token 认证
- 按 Agent 强制 QPM 和每日 token 上限
- 每次请求落 SLS 用于审计和成本归因
- 轮转 key 不重启任何 Agent

两天搭，永久赢。

![用 Terraform 给 AI Agent 上云（六）：LLM 网关与密钥管理 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_1.jpg)

## 架构设计

![集中式 LLM 网关：统一出口、统一配额、统一审计日志](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig1_gateway_topology.png)

左边是 Agent，右边是服务提供商（provider），中间是网关。每个 Agent 发起的 HTTP 请求看似调用某个“LLM”，实际上都会先经过网关。网关负责决定请求分发给哪个 provider，附加正确的 API 密钥，执行配额限制，并记录操作日志。

在实现方案上，你有两种较为合理的选择：

1. **阿里云 API Gateway 搭配自定义后端**  
   这种方式托管程度最高，配置配额计划最为便捷，同时能够与 RAM 无缝集成，适合对灵活性要求不高的场景。

2. **基于 ECS 自托管 LiteLLM（或自研方案）并配合 ALB**  
   这种方案灵活性最强，能够支持长尾 provider 的接入，同时也更容易扩展成本追踪功能，适合需要复杂路由逻辑的场景。

具体选择哪种方案，取决于路由逻辑的定制化程度。如果只是实现一个简单的代理功能并附带配额管理，单独使用 API Gateway 就足够了；如果需要支持多 provider 路由、具备 fallback 机制以及预算控制能力，那么部署在 ECS 上的 LiteLLM 显然是更好的选择。
## 第一步：将所有密钥存储到 KMS Secrets Manager

首要原则是：provider 的密钥绝不能出现在 `.env` 文件、`provider {}` 配置块、Agent 代码或 tfstate 的明文内容中。这些密钥统一存储在 KMS Secrets Manager 中，网关在启动时通过 STS 动态拉取。

```hcl
locals {
  llm_secrets = {
    "dashscope-prod"  = "DashScope (百炼) API 密钥"
    "openai-prod"     = "OpenAI API 密钥"
    "anthropic-prod"  = "Anthropic API 密钥"
    "deepseek-prod"   = "DeepSeek API 密钥"
  }
}

resource "alicloud_kms_secret" "llm" {
  for_each = local.llm_secrets

  secret_name              = each.key
  secret_data              = var.llm_keys[each.key]   # 通过 -var 或环境变量注入
  version_id               = "v1"
  description              = each.value
  encryption_key_id        = module.vpc.kms_keys["secrets"]
  rotation_interval        = "30d"
  enable_automatic_rotation = false   # 我们通过更新 secret_data 来手动轮换
  recovery_window_in_days  = 7
}
```

密钥的实际值通过 `var.llm_keys` 传入，可以通过 `-var-file=secrets.auto.tfvars`（已加入 gitignore）或者 CI 环境中的 `TF_VAR_llm_keys='{...}'` 提供。这些敏感信息永远不应直接存储在代码仓库中。

> **实战建议：** 当需要轮换 provider 密钥时，修改 `secret_data` 并递增 `version_id`。KMS 会在 recovery window 内保留旧版本的可用性，确保正在进行的请求不会中断；而新启动的网关会自动获取最新版本。建议将这一操作以 PR 的形式提交，以便后续审计和追踪。
## 第 2 步：网关能 assume 的 RAM role

网关 ECS 或函数需要权限读这些 secret——而且只读这些：

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

三处刻意：

- **Resource 级 policy。** 只这些 secret，不是 `kms:GetSecretValue` on `*`。网关被攻陷时攻击者无法横向扩展到其他 KMS secret。
- **没有长期 AK。** Role 由 ECS 实例通过 metadata service assume。零静态凭证。
- **`kms:Decrypt` 也要给**——仅仅读 secret 也需要它，因为 secret 是静态加密的。

## 第 3 步：在 ECS 上部署 LiteLLM

LiteLLM 是我所知最简单的开源 LLM 代理。前端讲 OpenAI API 格式，后端翻译成各 provider 的方言。ECS 自托管保留灵活性：

```hcl
resource "alicloud_instance" "gateway" {
  count = 2  # 双机 HA，前面挂 ALB

  instance_name        = "llm-gateway-${terraform.workspace}-${count.index + 1}"
  image_id             = data.alicloud_images.ubuntu.images[0].id
  instance_type        = "ecs.c7.large"
  availability_zone    = "cn-shanghai-${count.index == 0 ? "l" : "m"}"

  vswitch_id      = module.vpc.private_vswitch_ids[count.index]
  security_groups = [module.vpc.agent_runtime_sg_id]   # 同 SG；网关也是 runtime tier

  role_name = alicloud_ram_role.gateway.name           # 网关 assume 这个 role

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

`gateway-init.sh` 启动：

```bash
#!/bin/bash
set -euxo pipefail

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl jq

# 通过实例 role 从 KMS 拉 provider key（不需要 AK）
TOKEN=$(curl -s http://100.100.100.200/latest/meta-data/ram/security-credentials/agent-gateway-${ENV})
ACCESS_KEY_ID=$(echo $TOKEN | jq -r .AccessKeyId)
ACCESS_KEY_SECRET=$(echo $TOKEN | jq -r .AccessKeySecret)
SECURITY_TOKEN=$(echo $TOKEN | jq -r .SecurityToken)

# 用阿里云 KMS CLI（或 Python SDK）拿每个 key
pip install alibabacloud-kms20160120
export DASHSCOPE_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('dashscope-prod'))")
export OPENAI_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('openai-prod'))")
export ANTHROPIC_API_KEY=$(python3 -c "import kms_helper; print(kms_helper.get('anthropic-prod'))")

# 写 LiteLLM 配置
mkdir -p /etc/litellm
echo "${config_b64}" | base64 -d > /etc/litellm/config.yaml

# 装 + 用 pm2 跑 LiteLLM
pip install 'litellm[proxy]'
npm install -g pm2
pm2 start --name llm-gateway -- litellm --config /etc/litellm/config.yaml --port 4000
pm2 save
pm2 startup systemd -u root --hp /root
```

每台实例上网关现在跑起来了，监听 4000，所有 provider key 已加载。前面挂 ALB 分流：

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

Agent 现在通过 `http://<alb-id>.cn-shanghai.alb.aliyuncs.com/v1/chat/completions` 访问网关，永远见不到 provider key。

## 第 4 步：按 Agent 配额

LiteLLM 原生支持按 key 配额。最干净的 Terraform 接法是给每个 Agent 建一个 LiteLLM "virtual key"，各自带 QPM 和 token 预算。LiteLLM 把这些存在自己数据库里，所以你 apply 时通过它的 API 用 `null_resource` 配：

![按 Agent 配额策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig3_quota_table.png)

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

我不太喜欢 `null_resource` + `local-exec`——它是"provider 里还没这个资源"的逃生口。但能用，替代方案（写一个 LiteLLM 的 Terraform provider）一个团队的代价大于回报。

输出是每个 Agent 拿到一个独立的 `LITELLM_API_KEY` 环境变量，第四篇的 cloud-init 脚本去读。配额超限会返回 `429 Too Many Requests`，Agent 应该用指数退避处理。

## 第 5 步：密钥轮转流程

![用 Terraform 部署 AI Agent（六）：LLM 网关与密钥管理 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_2.jpg)

将密钥存储在 KMS Secrets Manager 中的核心目的，就是为了实现安全的密钥轮转：

![密钥轮转流程——KMS 作为唯一可信来源](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig2_secret_rotation.png)

整个轮转生命周期可以分为以下几个阶段：

1. 在 Terraform 配置中更新 `secret_data` 的值（也可以通过 KMS API 修改），同时将 `version_id` 更新为 `v2`。
2. KMS 会在轮转窗口期内（默认 30 天）继续保持 `v1` 的可用性，以确保平滑过渡。
3. 网关实例在冷启动时会重新拉取最新的密钥；而已经在运行的实例则会继续使用缓存中的密钥值，直到下一次刷新（每 15 分钟刷新一次，具体频率可在 `gateway-init.sh` 中配置）。
4. 30 天后，`v1` 将被禁用。如果此时仍有服务尝试使用 `v1`，将会收到 `InvalidSecretVersion` 错误提示。
5. 通过 SLS（日志服务）确认 `v1` 已无任何调用记录后，正式将 `v2` 提升为主版本，并彻底退役 `v1`。

对于团队而言，建议将上述流程整理成一份标准化的操作手册（Runbook），并每季度定期执行一次，即便没有发生任何泄露事件。通常来说，超过一个季度未轮换的密钥可被视为“陈旧密钥”，应将其视作潜在风险或低级别事故来处理。
## 百炼和 DashScope 具体怎么处理？

在 LiteLLM 的视角中，DashScope 只是一个兼容 OpenAI 的接口。它的模型名称类似于 `dashscope/qwen-max`、`dashscope/qwen-plus` 等等。而 API Key 则需要通过 DashScope 控制台生成。

如果你希望获得更“阿里云原生”的体验（比如使用 STS 而不是直接依赖 API Key），DashScope 的部分接口确实支持基于 STS 的认证方式。不过，即便到了 2026 年，API Key 的使用仍然是主流，而通过 KMS 定期轮换密钥依然是推荐的最佳实践。

> **实用建议：** 为 LiteLLM 设置一个 `master_key`（即环境变量 `LITELLM_MASTER_KEY`）。如果不设置这个值，任何能够访问网关的人都可以自行生成 API Key。而设置了之后，只有持有主密钥的人才能签发子密钥——并且主密钥始终只存在于 Terraform 的变量空间中，不会外泄。
## 你将获得什么

读完这篇文章后，你将拥有以下能力：

- 一个统一的 URL，所有 Agent 都通过它调用“LLM”
- 一个添加新模型提供商的入口（修改 `litellm_config` 后执行 `terraform apply`）
- 一个轮换任意提供商密钥的地方（修改 `var.llm_keys` 后执行 `terraform apply`）
- 一个日志流（将在下一篇中介绍），记录每次请求的延迟、Token 数量、使用的模型以及对应的 Agent
- 每个 Agent 的硬性 QPM 和预算上限——即使出现失控循环，每天最多花费 ¥800，而不会耗尽整个月的预算

这个网关是一项战略性资产。我为每个团队部署类似的方案后，通常不到一个月他们都会向我表示感谢。尤其是当有人不小心把 API 密钥提交到 Git 仓库时，他们会意识到，轮换密钥只需要提交一个简单的单行 PR，而不是手忙脚乱地救火。
## 接下来的内容

第七篇聚焦于可观测性与成本管理：使用 SLS 处理日志、ARMS 跟踪分布式链路、CloudMonitor 监控指标，设置预算告警以便在每日 LLM 花费超出阈值时通知钉钉，同时通过 SLS 驱动的成本仪表盘，您可以直观地了解“究竟是哪个 Agent 在消耗我的预算”。

第八篇则是一个完整的端到端实践，将第二篇到第七篇的所有内容整合起来，最终通过一次 `terraform apply` 操作落地实现。
## 使用 KMS 托管敏感变量：明文泄露的风险与防范

本文介绍了如何通过 `random_password` 和 `alicloud_kms_secret` 来保护静态存储中的敏感信息。然而，即使这样做了，仍然存在至少三个可能导致明文泄露的隐患。这些隐患在实际项目中屡见不鲜，值得警惕。

### 隐患 1：terraform.tfstate 文件

每次执行 `terraform apply` 时，`alicloud_kms_secret.secret_data` 的明文值都会被写入 tfstate 文件。即使你在变量上设置了 `sensitive = true`，也只能隐藏变量本身的输出，而无法阻止其值被记录到 state JSON 中。以下是多层防护措施：

1. **OSS 存储桶的 KMS 加密**（参考第二篇文章）—— 这一点通常已经完成，确保了状态文件的静态加密。
2. **OSS 存储桶的访问策略** —— 严格限制 `oss:GetObject` 权限，仅允许 CI 跑步者角色访问，禁止开发者直接操作。
3. **使用 `data` 数据源代替 HCL 中的明文**。如果密钥是通过外部流程（例如 HSM 轮换任务）生成的，Terraform 只需读取而不应直接写入密钥值：

```hcl
data "alicloud_kms_secret" "openai" {
  secret_name = "openai-prod"
  version_id  = "ACSCurrent"   # 始终指向最新版本
}

resource "alicloud_instance" "gateway" {
  user_data = base64encode(templatefile("${path.module}/init.sh", {
    # 不要直接传递密钥值，而是传递密钥名称，让实例自行获取
    openai_secret_name = "openai-prod"
  }))
}
```

核心原则是：**Terraform 应该只知道密钥的名称，而不是其具体值**。运行时，实例可以通过元数据服务从 KMS 动态拉取密钥值。

### 隐患 2：CI 日志

当变量设置了 `sensitive = true` 时，`terraform plan` 的输出会将敏感值显示为 `(sensitive value)`。但这种保护仅限于变量本身，派生的资源属性或输出仍可能泄露敏感信息。以下是一个常见问题示例：

```hcl
variable "openai_key" {
  type      = string
  sensitive = true
}

# 这里的输出是安全的：
resource "alicloud_kms_secret" "openai" {
  secret_data = var.openai_key
  # plan 输出：secret_data = (sensitive value)  ✓
}

# 但这里的输出可能会泄露：
output "gateway_config_url" {
  value = "https://gateway.example.com?key=${var.openai_key}"
  # plan 输出会显示完整的 URL，包括密钥 ✗
}
```

解决方法是，对所有依赖敏感值的输出也设置 `sensitive = true`：

```hcl
output "gateway_config_url" {
  value     = "https://gateway.example.com?key=${var.openai_key}"
  sensitive = true
}
```

此外，对于 tfvars 文件，务必将其添加到 `.gitignore` 中，并配置 CI 管道以防止意外提交：

```yaml
# .github/workflows/no-secrets.yml
- name: 检查仓库中是否有敏感文件
  run: |
    if git ls-files | grep -E '\.auto\.tfvars$|secrets/'; then
      echo "错误：敏感文件被提交"; exit 1
    fi
```

### 隐患 3：Provider 调试日志

使用 `TF_LOG=DEBUG terraform apply` 是排查 Provider 问题的最快方法，但它也会将每个 API 请求和响应（包括含密钥的请求体）打印到终端滚动日志中。我曾见过有人因截图 Slack 而导致密钥泄露。

如果必须启用调试日志，请务必将输出重定向到权限受限的文件，切勿直接粘贴到聊天工具中：

```bash
TF_LOG=DEBUG terraform apply 2> /tmp/tf.log
chmod 600 /tmp/tf.log
# 本地查看日志，避免直接复制内容
shred -u /tmp/tf.log    # 查看完毕后销毁文件
```

更好的做法是使用 `TF_LOG_CORE=DEBUG`，它仅记录 Terraform 核心的日志，通常足以定位问题，同时避免包含 Provider 的请求体内容。
## Plan-review-apply gate：CI 流水线挡住人会漏的东西

文章里 LiteLLM key 生成用 `null_resource`，单工程师可以。多 agent、多环境、共享值班的团队，你想要一个结构化 CI 流水线，由人在 diff 层评审。我跑的 GitHub Actions：

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
      pull-requests: write
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

三件事人难做的：

- **`terraform fmt -check`** 拒绝未格式化的 HCL。PR 评审里再没"你跑 fmt 了吗"的评论。
- **`tfsec`** 跑 Checkov 风格的安全扫描——标记公开 OSS bucket、未加密磁盘、过宽 SG 规则。抓人会眼瞎放过的回归。
- **plan 作为 PR 评论发出来** 意味着评审看的是**真实 plan 输出**，不是"你信我，看着没事"。

配套 apply workflow：

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
    environment: ${{ inputs.workspace }}   # GitHub Environment，prod 配 required reviewers
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

GitHub 的 `environment` 机制把 prod apply gate 在必需评审者批准之后——另一个人（常常是我，常常是值班）点"approve"才能跑 apply。Apply 用和 plan 不同的 RAM role（plan role 只读、apply 才有写）。两把钥匙启动。

这套流水线一年抓到了五个真实故障：误翻 `prevent_destroy = false`、会破 VPC peering 的 CIDR 重叠、忘记钉 module 版本、把真 drift 遮住的 `ignore_changes`、混进 PR 的 `0.0.0.0/0` 安全组规则。每个都是 PR 上的评论，merge 前修掉。

## Atlantis 和 GitHub Actions：什么时候该升级？

GitHub Actions 是一个不错的起点。但当团队中有超过几位工程师（比如 5 人或更多）同时使用 Terraform 时，管理每个 PR 的 plan 评论、锁冲突以及 Actions 上的审批流程，就会变得越来越吃力。这时，Atlantis 就成了下一步的选择。

Atlantis 是一个自托管的 webhook 服务器，它能够监听 PR，自动运行 `terraform plan`，并将计划结果直接评论到 PR 中。授权用户只需在 PR 中输入 `atlantis apply`，即可触发应用操作。与 GitHub Actions 相比，Atlantis 有以下优势：

- **Plan 在你的 VPC 内执行** —— 不需要让 GitHub Actions 的 runner 访问存储 Terraform 状态的 OSS bucket。
- **单服务器持锁，顺序执行 apply** —— 避免了多个 PR 并发操作时可能出现的竞争问题。
- **支持项目级配置（`atlantis.yaml`）** —— 可以灵活定义哪些仓库或目录需要被管理，并针对不同目录设置独立的审批策略。

部署 Atlantis 本身是一个典型的 `vpc-baseline` + `compute` 场景：一台 ECS 实例、一个 ALB 和一个 RAM 角色即可搞定。在一个 10 人工程师团队中使用两个月后，效率提升非常明显：从 plan 到 apply 的完整周期从原来的 25 分钟（Actions 排队 + 手动审批）缩短到了 8 分钟。

如果团队规模小于 5 人，我不建议使用 Atlantis，因为运维它的成本可能得不偿失。但如果团队规模超过 5 人，Atlantis 往往能在一个季度内通过效率提升收回成本。

> **实战建议：** 无论选择哪种流水线工具，记住一个原则：**一个主仓库，多个环境**。不要为生产环境单独 fork 开发环境的代码库。使用单个仓库并通过 `env/dev.tfvars`、`env/staging.tfvars` 和 `env/prod.tfvars` 文件来区分不同环境的配置，这样可以确保所有环境的代码路径完全一致——这正是基础设施即代码（IaC）的核心价值所在。按环境 fork 仓库的做法是一种反模式，会逐渐侵蚀 IaC 带来的收益。
