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
我在许多尚未成熟的 Agent 架构中反复看到一个通病：每个 Agent 都在自己的 `.env` 文件里存一份 `OPENAI_API_KEY`。有时是同一个 key，有时各不相同，甚至还有同事在原型阶段留下的个人密钥。等到账单来了，没人说得清哪个 Agent 消耗了多少 token；而一旦密钥泄露（这几乎是必然的），你就得像打地鼠一样，在十几个 `.env` 文件里来回折腾。

真正让我警醒的是两年前的一次事故。一位外包工程师周五结束了为期三个月的合作，把笔记本带回家，结果到了下周二，DashScope 账单突然报警——有 1200 万 `qwen-max` token 的流量来自一个我们从未见过的 IP。原来他个人的 API key——当初复制粘贴到某个侧边项目里的——还留在我们 Agent 的 `.env` 文件中。轮换这个密钥花了整整六小时：三名工程师、四个代码仓库、两条 CI 流水线，外加一条迅速失控的 Slack 讨论线程。这种事，绝不能再发生。

本文旨在彻底解决这一问题，通过构建一个 **LLM 网关**，实现以下目标：

- 所有模型提供商的密钥统一托管在 KMS Secrets Manager 中（一个存储桶、一套 ACL、统一的轮换节奏）；
- Agent 通过 RAM 颁发的短期 Token 进行认证，机器上绝不存放静态 AK；
- 对每个 Agent 设置每分钟请求数（QPM）和每日 token 上限，确保失控循环最多每天烧掉 ¥800，而不是吃掉你整个季度的预算；
- 所有请求日志写入 SLS，便于事后追溯、成本分摊和 SOC-2 审计；
- 轮换密钥无需重新部署任何 Agent——只需提交一个 PR 并执行一次 `terraform apply` 即可完成。

两天的初始投入，换来长期的运维红利。

![Terraform for AI Agents (6)：LLM Gateway 和密钥管理 — 视图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/illustration_1.png)

## 架构形态

![集中式 LLM 网关：一个出口，一个配额，一个审计日志](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig1_gateway_topology.png)

整体架构上，Agent 在左侧，模型提供商在右侧，网关居中作为代理层。每个 Agent 发往“LLM”的 HTTP 请求实际上都会先到达网关，由网关决定路由到哪家提供商、注入正确的密钥、执行配额限制，并记录完整的调用日志。

目前有两种主流实现方案：

1. **阿里云 API Gateway + 自定义后端** —— 托管程度最高，配额策略配置最简单，原生集成 RAM。适合路由逻辑简单的场景：“一个模型对应一个提供商，只需限流”。
2. **ALB 后挂载自托管的 LiteLLM on ECS** —— 灵活性最强，支持长尾提供商（如 DeepSeek、Moonshot、Zhipu，甚至你自己微调的 PAI 端点），也更容易扩展成本追踪和跨提供商的故障转移（fallback）逻辑。

我会根据路由复杂度灵活选择。如果只是做简单代理并加上限流，API Gateway 已足够。但若涉及多提供商路由、预算熔断器（budget guard）和电路保护（circuit breaker）——这通常是大多数团队在半年内就会遇到的需求——那么 ECS 上自托管 LiteLLM 是更优解。本文后续将聚焦于 LiteLLM 方案，因为它适用于约 80% 的团队。

## 第一步：将所有密钥存入 KMS Secrets Manager

首要原则：模型提供商的密钥绝不能出现在 `.env` 文件、`provider {}` 配置块、Agent 代码，或以明文形式存在于 tfstate 中。它们只应存在于 KMS Secrets Manager 中，并由网关在启动时通过 STS 动态拉取。

![用于管理 API 密钥和凭证的安全保险库](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/wanxiang_secret_vault.png)

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

密钥本身通过 `var.llm_keys` 变量传入——可通过 `-var-file=secrets.auto.tfvars`（已加入 `.gitignore`）或从 CI 的 secret 中设置 `TF_VAR_llm_keys='{...}'`。它们绝不会出现在你的代码仓库中。

成本方面值得一提：在上海区域，KMS Secrets Manager 的费用约为每个密钥每月 ¥0.4，外加每 1 万次 API 调用 ¥0.03。假设你有四个提供商密钥，两台网关实例每小时拉取两次，月度账单几乎可以忽略——不到 ¥10。阿里云默认的服务密钥（SMK）免费使用；只有当你创建客户托管的 CMK（Customer Master Key）时，才需额外支付 ¥1/个/月。别让“KMS 听起来很贵”成为你继续依赖 `.env` 文件的借口。

> **实战建议**：轮换提供商密钥时，修改 `secret_data` 并递增 `version_id`。KMS 会在恢复窗口期（默认 30 天）内保持旧版本有效，确保进行中的请求不受影响；新启动的网关实例则会自动拉取新版本。将此操作以 PR 形式提交，便于审计追踪。

## 第二步：为网关配置可被 Assume 的 RAM 角色

网关所在的 ECS 实例或函数计算需要权限读取这些密钥——而且**仅限**这些密钥：

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

该权限设计包含三个关键点：

- **资源级策略限制**：仅允许访问指定密钥，而非对 `kms:GetSecretValue` 开放 `*` 权限。即使网关实例被攻破，攻击者也无法横向移动至其他 KMS 密钥（如账单密钥、RDS 密码、OSS 存储桶等），确保敏感信息隔离。
- **无长期 AK**：角色由 ECS 实例通过元数据服务（metadata service）自动 Assume，磁盘、环境变量或 cloud-init 中均无静态凭证。
- **必须显式授权 `kms:Decrypt`**：即使只是读取密钥，也需要该权限，因为 KMS 在静态存储时已对密钥加密。这是网关启动后频繁返回 401 错误的最常见原因。

## 第三步：在 ECS 上部署 LiteLLM

LiteLLM 是我所知最易用的开源 LLM 代理网关：前端兼容 OpenAI API 格式，后端可自动适配各家提供商的协议。将其自托管在 ECS 上，能保持高度灵活性：

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

两台 `ecs.c7.large` 实例（2 vCPU、4 GB 内存）足以轻松处理 200+ QPS 的纯代理流量。LiteLLM 属于异步 I/O 密集型应用，CPU 使用率通常不超过 30%，无需过度配置。若流量存在突发性，可将其纳入弹性伸缩组，当 CPU 持续超过 60% 时由 CloudMonitor 自动扩容。

`gateway-init.sh` 脚本负责启动流程：

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

每个实例现在都在 4000 端口运行网关，所有提供商密钥以环境变量形式加载进进程内存——绝不会落盘。前端的 ALB 负责流量分发：

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

Agent 现在只需访问 `http://<alb-id>.cn-shanghai.alb.aliyuncs.com/v1/chat/completions` 即可调用 LLM，完全无需接触任何提供商密钥。该 ALB 为内网专用：不分配公网 IP，也不监听任何面向公网的端口（如 443）。若 Agent 需从 VPC 外部发起调用，应通过堡垒机或 CEN 接入，绝不允许直连。

## 步骤 4：为每个 Agent 设置独立配额

LiteLLM 原生支持按 API Key 设置配额。最清晰的做法是通过 Terraform 为每个 Agent 创建一个 LiteLLM “虚拟密钥”，并为其单独配置 QPM 和每日 token 预算。由于这些配额信息存储在 LiteLLM 自身的数据库中，我们需要在 `terraform apply` 阶段通过其 API 动态创建，可借助 `null_resource` 实现：

![每个代理的配额策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig3_quota_table.png)

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

坦白说，我对 `null_resource` + `local-exec` 的组合并不热衷——它本质上是“Provider 尚未支持该资源”时的临时逃生舱。但它确实有效，而为单一团队专门开发一个自定义 Terraform Provider 的成本显然过高。一旦 LiteLLM 官方推出 Terraform Provider，切换起来不过一天之事。

最终效果是：每个 Agent 获得一个独立的 `LITELLM_API_KEY` 环境变量，第 4 篇文章中的 cloud-init 脚本会自动读取它。当配额超限时，网关返回 `429 Too Many Requests`，Agent 必须通过指数退避重试——这一逻辑应固化在共享的 HTTP 客户端中，切勿依赖每位 Agent 开发者自行实现。

关于配额数值：`schedule-agent` 的上限设为每日 10 万 token 和 ¥40，看似偏低，但这是有意为之。一个调度 Agent 若突然飙升至 200 万 token，极大概率陷入了规划死循环。此时硬性截断远比月底收到 ¥3000 的意外账单要好得多。建议将上限设为“该 Agent 过去 30 天日用量 P99 值的 10 倍”，并每季度复审一次。

## 步骤 5：密钥轮换流程

![Terraform for AI Agents (6)：LLM Gateway 和密钥管理 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/illustration_2.png)

将密钥存入 KMS Secrets Manager 的核心价值就在于安全轮换：

![密钥轮换流程 — KMS 作为单一可信源](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig2_secret_rotation.png)

其生命周期如下：

1. 你在 Terraform 中更新 `secret_data`（或通过 KMS API），并将 `version_id` 提升至 `v2`；
2. KMS 会在轮换窗口期（默认 30 天）内保持 `v1` 版本有效，确保进行中的请求不受影响；
3. 新启动的网关实例会立即拉取 `v2`；现有实例则继续使用缓存值，直到下一次刷新（每 15 分钟一次，由 `gateway-init.sh` 配置）；
4. 30 天后，`v1` 被自动禁用——任何仍在使用它的请求将收到 `InvalidSecretVersion` 错误；
5. 你通过 SLS 日志确认 `v1` 已无调用，随后正式启用 `v2` 并退役 `v1`。

对团队而言，应将此流程固化为 Runbook，并**即使未发生泄露也每季度执行一次**。任何存活超过一个季度的密钥都应被视为“过期”（stale），按低优先级安全事件处理。开篇提到的外包事故，正是因为 DashScope 密钥长达 14 个月未轮换。而采用本文方案后，此类情况将不可能重现——即便你忘记轮换，30 天的窗口期也会强制触发处理。

## 步骤 6：明文仍可能泄露的三个地方（及应对措施）

第一步仅保护了静态存储（at rest）的密钥。若不加注意，至少还有三个地方可能导致明文泄露——这三个坑我都亲眼见过项目踩过。

### 泄露 1：terraform.tfstate

`alicloud_kms_secret.secret_data` 的明文值会在每次 `apply` 后写入 tfstate。即使变量标记了 `sensitive = true`，其**实际值**仍以明文形式存在于 state JSON 中。缓解措施需分层实施：

1. **OSS Bucket 启用 KMS 加密**（见第 2 篇）——已配置，保护状态文件静态安全；
2. **严格限制 OSS Bucket 访问策略**——仅允许 CI Runner 角色执行 `oss:GetObject`，开发人员一律禁止；
3. **采用 `data` source 模式，避免在 HCL 中硬编码明文**。当密钥由外部系统创建（如 HSM 轮换任务或 KMS 控制台），Terraform 应只读取而不生成：

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

核心原则是：**Terraform 只需知道密钥的名称，无需知晓其具体值**。运行时通过实例元数据从 KMS 动态获取。这是最重要的安全习惯，与多数教程的示范恰恰相反。

### 泄露 2：CI 日志

若变量设置了 `sensitive = true`，`terraform plan` 输出会将其标记为 `(sensitive value)`。**但注意**：此标记仅作用于变量本身，**不自动传递给衍生出的资源属性**。常见疏漏如下：

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

务必为所有源自敏感值的 output 显式添加 `sensitive = true`：

```hcl
output "gateway_config_url" {
  value     = "https://gateway.example.com?key=${var.openai_key}"
  sensitive = true
}
```

对于 tfvars 文件，不仅要加入 `.gitignore`，还应在 CI 中配置：一旦检测到提交即失败：

```yaml
# .github/workflows/no-secrets.yml
- name: check no secrets in repo
  run: |
    if git ls-files | grep -E '\.auto\.tfvars$|secrets/'; then
      echo "ERROR: secret files committed"; exit 1
    fi
```

### 泄露 3：Provider 调试日志

`TF_LOG=DEBUG terraform apply` 是调试 Provider 问题的最快方式，但同时也是将所有 API 请求与响应（包括含密钥的请求体）dump 到终端历史记录的最快途径。我曾两次在不同公司的 Slack 截图中目睹此类泄露。

若必须使用 `TF_LOG`，请将输出重定向至权限受限的文件，切勿直接粘贴：

```bash
TF_LOG=DEBUG terraform apply 2> /tmp/tf.log
chmod 600 /tmp/tf.log
# review locally; never paste verbatim
shred -u /tmp/tf.log    # delete when done
```

更佳做法是使用 `TF_LOG_CORE=DEBUG`（仅限 Terraform 核心日志），通常足以定位问题，且不会包含 Provider 的请求体。

## 步骤 7：CI 中的 plan-review-apply 门禁机制

对单人开发而言，用 `null_resource` 生成 LiteLLM 密钥尚可接受。但若团队拥有多个 Agent、多套环境，还需轮值 On-call，则必须建立结构化的 CI 流水线，确保人类在 Diff 层面进行评审。以下是我使用的 GitHub Actions 工作流：

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

该流程强制落实三项人工易忽略的关键检查：

- **`terraform fmt -check`**：拒绝未格式化的 HCL，避免评审中出现“你跑 fmt 了吗？”这类低效讨论；
- **`tfsec`**：执行类似 Checkov 的安全扫描，自动标记公开存储桶、未加密卷、过于宽松的安全组规则；
- **Plan 结果以 PR 评论形式发布**：确保评审基于**实际的 Plan 输出**，而非一句“信我，没问题”。

对应的 Apply 工作流如下：

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

GitHub 的 `environment` 机制为生产环境 Apply 设置审批门禁——必须由另一位成员（通常是 On-call 工程师）手动点击“approve”才能执行。Apply 使用的 RAM 角色与 Plan 不同：Plan 仅有读权限，Apply 才具备写权限，形成典型的“双钥启动”机制。

这套流水线在我运行的一年内成功拦截了五起真实事故：一次误将 `prevent_destroy = false` 翻转、一次会导致 VPC 对等连接中断的 CIDR 冲突、一个未锁定的模块版本、一个掩盖配置漂移的 `ignore_changes`，以及一条混入 PR 的 `0.0.0.0/0` 安全组规则。所有问题均在 PR 阶段被发现并修复，未流入生产环境。

### 何时该升级到 Atlantis？

GitHub Actions 是理想的起点。但当团队中使用 Terraform 的工程师超过五人时，PR 级 Plan 评论、锁竞争、人工审批等运维开销将显著增加。此时，Atlantis 成为更优选择：它是一个自托管的 Webhook 服务器，可监听 PR、自动执行 `terraform plan`、评论结果，并允许授权用户通过 `atlantis apply` 触发部署。

相比 Actions，Atlantis 具备以下优势：

- Plan 在**你的 VPC 内部执行**——无需向外部 Runner 开放 OSS State 桶的访问权限；
- 单一持久化服务器持有锁，确保 Apply 顺序执行——彻底避免并发 PR 间的竞争条件；
- 通过项目级配置文件（`atlantis.yaml`）可精细控制哪些目录受管理，并设置目录级审批策略。

部署 Atlantis 本身就是一个标准的 `vpc-baseline` + `compute` 场景：一台 ECS、一个 ALB、一个 RAM 角色。在 10 人团队运行两个月后，吞吐效率显著提升：Plan 到 Apply 的周期从 25 分钟（Actions 排队 + 人工审批）缩短至 8 分钟。

若团队少于 5 名工程师，我不推荐 Atlantis——其运维成本难以回本。但一旦超过 5 人，通常一个季度内即可收回投入。

> **实战建议**：无论选择哪种流水线，务必坚持 **“单仓库，多环境”** 原则。切勿通过 fork 方式为不同环境创建独立仓库。正确的做法是在单一仓库中通过 `env/dev.tfvars`、`env/staging.tfvars`、`env/prod.tfvars` 等文件区分环境——这才是 IaC 的核心价值所在。按环境 fork 仓库是一种反模式，会逐渐侵蚀你最初追求的一致性保障。

## 针对百炼 / DashScope 的特殊说明

在 LiteLLM 看来，DashScope 仅是一个兼容 OpenAI 协议的端点。模型名称为 `dashscope/qwen-max`、`dashscope/qwen-plus` 等，API Key 即从 DashScope 控制台生成的密钥。

若希望使用阿里云原生认证（如 STS 代替 API Key），DashScope 在部分端点已支持 STS 认证。但在 2026 年，API Key 仍是主流方案，本文所述的 KMS 轮换机制依然是最佳实践。待 STS 成为默认（路线图预计在 2027 年），本文配置仅需一次调整即可适配，而轮换纪律始终不变。

> **实战建议**：务必为 LiteLLM 设置 `master_key`（通过 `LITELLM_MASTER_KEY` 环境变量）。若未设置，任何能访问网关的人都可自行签发 API Key；设置后，仅 master 可生成子密钥——且该 master 密钥永远不会离开 Terraform 的变量空间。

## 这套方案带来了什么

完成本文后，你将拥有：

- 所有 Agent 调用“LLM”的统一入口 URL；
- 添加新模型提供商的唯一位置（编辑 `litellm_config`，执行 `terraform apply`）；
- 轮换任意提供商密钥的标准化流程（修改 `var.llm_keys`，执行 `terraform apply`）；
- 统一日志流（见下篇）记录每个请求的延迟、token 数、模型及调用 Agent；
- 每个 Agent 的硬性 QPM 与预算上限——失控循环每日最多花费 ¥800，而非整月预算；
- 能捕获人类易忽略问题的 CI 流水线，配合生产环境的双钥启动机制；
- 三个关键明文泄露路径已被堵死，远不止最显眼的那个。

这个网关是一项战略资产。我交付过的每个团队都在一个月内表达了感谢——通常发生在首次有人不慎将 API Key 提交到 Git 时，他们惊喜地发现：轮换它只需一行 PR，而非我曾经历过的六小时救火。

## 下一步

第 7 篇将聚焦可观测性与成本控制：使用 SLS 收集日志、ARMS 追踪链路、CloudMonitor 监控指标，设置每日 LLM 花费超阈值时自动钉钉告警，并构建基于 SLS 的成本仪表盘，让你一眼看清“哪个 Agent 在烧我的预算”。

第 8 篇则是端到端实战：将第 2 至 7 篇的所有内容整合为一个 `terraform apply`，一键拉起完整 Agent 栈。
