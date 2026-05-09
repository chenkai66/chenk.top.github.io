---
title: "用 Terraform 给 AI Agent 上云（六）：LLM 网关与密钥管理"
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

不成熟的 Agent stack 有个常见模式：每个 Agent 自己 `.env` 文件里有一份 `OPENAI_API_KEY`。有时是同一份，有时不是，有时是某个同事原型阶段留下的个人 key。账单到了没人能说清是哪个 Agent 烧的 token；key 泄露的时候（一定会泄露），你要在十几个 `.env` 文件之间打地鼠。

真正让我警醒的是两年前一次事故。一位外包同学周五结束三个月驻场，笔记本带回家。下周二 DashScope 计费侧告警：12M 个 `qwen-max` token 来自我们不认识的 IP。他个人 API key——不知什么时候顺手粘到了一个副业项目里——还躺在我们 Agent 的 `.env` 里。轮换花了六个小时：三个工程师、四个仓库、两条 CI 流水线、一个慌乱的群。再也不要。

这篇文章终结这个模式。我们建一个 **LLM 网关**：

- 所有 provider key 收在 KMS Secrets Manager 里（一个桶、一份 ACL、一套轮换节奏）
- Agent 通过 RAM 短期凭证认证，业务机器上零静态 AK
- 按 Agent 强制 QPM 与每日 token 上限，失控循环最多烧 ¥800/天，不是一个季度的预算
- 每次请求落 SLS，做事后取证、成本归因和 SOC-2 证据
- 轮换 key 不重启任何 Agent，一个 PR、一次 apply 搞定

两天搭，永久收益。

![用 Terraform 给 AI Agent 上云（六）：LLM 网关与密钥管理 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_1.png)

## 架构设计

![集中式 LLM 网关：统一出口、统一配额、统一审计日志](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig1_gateway_topology.png)

左边是 Agent，右边是 provider，中间是网关。每个 Agent 看似在调用某个"LLM"，实际上请求都先走网关。网关决定分发到哪个 provider、附上正确的 key、执行配额、把结果落日志。

合理的实现方案有两种：

1. **阿里云 API Gateway + 自定义后端**——托管程度最高，配额计划最容易接，原生集成 RAM。适合"一个模型一个 provider，限个流就行"的场景。
2. **ECS 自托管 LiteLLM + ALB**——灵活性最强，长尾 provider（DeepSeek、Moonshot、智谱、你自己 PAI 上的微调端点）都接得了，更容易扩成本追踪和跨 provider fallback。

具体怎么选，看路由复杂度。纯代理加配额，单独用 API Gateway 够了；多 provider 路由 + 预算护栏 + 熔断（半年内每个团队都会走到这一步），LiteLLM 在 ECS 上更划算。本文走 LiteLLM 路线，因为 80% 的团队最终都在这个形态。

## 第一步：所有密钥收进 KMS Secrets Manager

第一条铁律：provider 密钥不能出现在 `.env`、`provider {}` 块、Agent 代码或 tfstate 明文里。它们统一存在 KMS Secrets Manager，网关启动时通过 STS 拉取。

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
  enable_automatic_rotation = false   # 手动通过更新 secret_data 来轮换
  recovery_window_in_days  = 7
}
```

密钥实际值通过 `var.llm_keys` 传入：本地用 `-var-file=secrets.auto.tfvars`（已 gitignore），CI 里用 `TF_VAR_llm_keys='{...}'` 从 secret 注入。永远不进仓库。

成本上算笔账。KMS Secrets Manager 在上海地域大约 ¥0.4/秘密/月，外加每万次 API 调用 ¥0.03。四个 provider key、两台网关每小时拉两次，月账单是零头——不到 ¥10。默认服务密钥免费，只有用户托管 CMK 每月 ¥1。别让"KMS 听起来贵"成为你继续用 `.env` 的借口。

> **实战建议：** 轮换 provider 密钥时，改 `secret_data` 并把 `version_id` 升一档。KMS 会在 recovery window 内保留旧版本，在途请求不会断；新拉取拿新版本。把这一步固化为 PR，便于审计追溯。

## 第二步：网关可以 assume 的 RAM 角色

网关 ECS 或函数需要权限读这些 secret——而且**只**读这些：

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
- **Resource 级 policy。** 只这些 secret，不是 `kms:GetSecretValue` on `*`。网关被打穿，攻击者无法横向到计费 key、RDS 密码、其他 OSS 桶。
- **零长期 AK。** 角色由 ECS 实例通过 metadata 服务 assume，磁盘、env、cloud-init 里都没静态凭证。
- **`kms:Decrypt` 必给。** 仅仅读 secret 也需要它，因为 secret 静态加密。漏掉这一行，是网关启动后每次拉取都 401 的头号原因。

## 第三步：在 ECS 上部署 LiteLLM

LiteLLM 是我用过最简单的开源 LLM 代理。前端讲 OpenAI API 格式，后端翻成各 provider 的方言。ECS 自托管保留灵活性：

```hcl
resource "alicloud_instance" "gateway" {
  count = 2  # 双机 HA，前面挂 ALB

  instance_name        = "llm-gateway-${terraform.workspace}-${count.index + 1}"
  image_id             = data.alicloud_images.ubuntu.images[0].id
  instance_type        = "ecs.c7.large"
  availability_zone    = "cn-shanghai-${count.index == 0 ? "l" : "m"}"

  vswitch_id      = module.vpc.private_vswitch_ids[count.index]
  security_groups = [module.vpc.agent_runtime_sg_id]   # 同 SG；网关属于 runtime tier

  role_name = alicloud_ram_role.gateway.name           # 网关 assume 这个角色

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

两台 `ecs.c7.large`（2 vCPU、4 GB）足以扛 200+ QPS 的纯代理流量。LiteLLM 是异步 I/O bound，CPU 很少超 30%，不要超配。如果流量有突发，扔进伸缩组，CloudMonitor 看到 CPU 持续超 60% 自动扩。

`gateway-init.sh` 启动：

```bash
#!/bin/bash
set -euxo pipefail

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl jq

# 通过实例角色从 KMS 拉 provider key（不需要 AK）
TOKEN=$(curl -s http://100.100.100.200/latest/meta-data/ram/security-credentials/agent-gateway-${ENV})
ACCESS_KEY_ID=$(echo $TOKEN | jq -r .AccessKeyId)
ACCESS_KEY_SECRET=$(echo $TOKEN | jq -r .AccessKeySecret)
SECURITY_TOKEN=$(echo $TOKEN | jq -r .SecurityToken)

# 用阿里云 KMS CLI（或 Python SDK）拉每个 key
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

每台实例上网关跑在 4000 端口，所有 provider key 加载到进程 env，从不落盘。前面挂 ALB 分流：

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

Agent 通过 `http://<alb-id>.cn-shanghai.alb.aliyuncs.com/v1/chat/completions` 访问网关，永远见不到 provider key。ALB 仅内网，不挂公网 IP、不开 443 对外。VPC 外要调走堡垒机或 CEN，不直连。

## 第四步：按 Agent 配额

LiteLLM 原生支持按 key 配额。最干净的 Terraform 接法是给每个 Agent 建一个 LiteLLM "virtual key"，各自带 QPM 和 token 预算。LiteLLM 把这些存在自己的数据库里，所以 apply 时通过 API 用 `null_resource` 配：

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

我不太喜欢 `null_resource` + `local-exec`——它是"provider 里还没这个资源"的逃生口。但它能用，替代方案（写一个 LiteLLM 的 Terraform provider）对一个团队来说成本远大于回报。LiteLLM 哪天发了官方 provider，一天就能换。

输出是每个 Agent 拿到一个独立的 `LITELLM_API_KEY` 环境变量，第四篇 cloud-init 脚本去读。配额超限返回 `429 Too Many Requests`，Agent 必须用指数退避处理——这要写进共享 HTTP 客户端，别指望每个 Agent 作者都记得。

数字上多说一句。`schedule-agent` 上限 100k token/天、¥40/天看起来很小，是故意的。一个调度 Agent 突然飙到 2M token，几乎肯定是规划循环卡住了。一个硬上限报错，远好过月底一笔 ¥3000 的"惊喜"。上限按"该 Agent 过去 30 天 p99 日用量的 10x"设，每季度复盘。

## 第五步：密钥轮转流程

![用 Terraform 部署 AI Agent（六）：LLM 网关与密钥管理 — 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/illustration_2.png)

把 key 放进 KMS Secrets Manager 的全部意义就在轮转：

![密钥轮转流程——KMS 作为唯一可信来源](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/06-llm-gateway-and-secrets/fig2_secret_rotation.png)

生命周期：
1. 在 Terraform 里改 `secret_data`（或调 KMS API），把 `version_id` 升到 `v2`
2. KMS 在 rotation 窗口内（默认 30 天）保留 `v1` 可用
3. 网关实例冷启动重新拉取；运行中实例继续用缓存值，直到下次刷新（每 15 分钟，在 `gateway-init.sh` 里配）
4. 30 天后 `v1` 禁用，仍在用 `v1` 的会拿到 `InvalidSecretVersion`
5. 用 SLS 确认 `v1` 零调用，把 `v2` 提为正式，退役 `v1`

对团队，把这套流程固化成 runbook，每季度执行一次，哪怕没有泄露。超过一个季度没换的 key 按定义就是"陈旧"，把陈旧当低级别事故对待。开篇那个外包同学的事故之所以会发生，就是 DashScope key 14 个月没换。读完这篇，那个剧本不再可能——就算你忘了，30 天窗口也会逼你做。

## 第六步：明文还会从哪些地方漏出去

第一步保护了 secret 在**静态存储**里的安全。但还有至少三个地方会漏明文，每一个都坑过我审过的项目。

### 漏点 1：terraform.tfstate

`alicloud_kms_secret.secret_data` 的明文每次 apply 都写进 tfstate。即使变量上加了 `sensitive = true`，那只挡住变量本身的输出，**值**还是会落进 state JSON。多层缓解：

1. **OSS 桶 KMS 加密**（第二篇已经做了）——保护 state 静态层。
2. **OSS 桶访问策略**——`oss:GetObject` 仅授给 CI runner 角色，开发者不能直接拉。
3. **用 `data` 数据源代替在 HCL 里写明文。** 当 secret 是带外创建的（比如 HSM 轮换 job 或 KMS 控制台），Terraform 只读不写：

```hcl
data "alicloud_kms_secret" "openai" {
  secret_name = "openai-prod"
  version_id  = "ACSCurrent"   # 始终指当前版本
}

# 用它而不让值进入 tfstate 资源段
resource "alicloud_instance" "gateway" {
  user_data = base64encode(templatefile("${path.module}/init.sh", {
    # 不要把 secret 传进来，传 secret 名字让机器自己拉
    openai_secret_name = "openai-prod"
  }))
}
```

核心原则：**Terraform 只该知道 secret 的名字，不该知道它的值**。运行时由实例通过 metadata 从 KMS 拉。这是要建立的最重要的习惯，也颠覆了大部分教程对 secret 的写法。

### 漏点 2：CI 日志

变量上 `sensitive = true` 时，`terraform plan` 输出会显示 `(sensitive value)`。**但**只对变量本身有效，从它派生的资源属性不会自动屏蔽。一个常见踩点：

```hcl
variable "openai_key" {
  type      = string
  sensitive = true
}

# 这里 OK：
resource "alicloud_kms_secret" "openai" {
  secret_data = var.openai_key
  # plan 输出：secret_data = (sensitive value)  ✓
}

# 这里漏：
output "gateway_config_url" {
  value = "https://gateway.example.com?key=${var.openai_key}"
  # plan 输出会带完整 URL，含密钥 ✗
}
```

每一个引用了敏感值的 output 都要标 `sensitive = true`：

```hcl
output "gateway_config_url" {
  value     = "https://gateway.example.com?key=${var.openai_key}"
  sensitive = true
}
```

tfvars 文件除了 `.gitignore`，还要在 CI 里加防御：

```yaml
# .github/workflows/no-secrets.yml
- name: 检查仓库中是否有敏感文件
  run: |
    if git ls-files | grep -E '\.auto\.tfvars$|secrets/'; then
      echo "ERROR: 敏感文件被提交"; exit 1
    fi
```

### 漏点 3：provider 调试日志

`TF_LOG=DEBUG terraform apply` 是排查 provider 问题最快的方法，也是最快把每一次 API 请求/响应（含密钥的请求体）打到终端滚动条里的方法。我两次在两家公司都见过这种事故的 Slack 截图。

必须用 `TF_LOG` 时，输出重定向到权限受限的文件，绝不复制粘贴：

```bash
TF_LOG=DEBUG terraform apply 2> /tmp/tf.log
chmod 600 /tmp/tf.log
# 本地查看，永远不要原文粘贴
shred -u /tmp/tf.log    # 看完销毁
```

更好的做法是用 `TF_LOG_CORE=DEBUG`（仅 Terraform 核心），通常足以定位问题，又不会带 provider 请求体。

## 第七步：CI 里的 plan-review-apply 闸门

文章里 LiteLLM key 生成用 `null_resource`，单工程师可以。多 Agent、多环境、共享值班的团队，你想要一条结构化 CI 流水线，由人在 diff 层评审。我跑的 GitHub Actions：

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

三件人会眼瞎放过的事，它能挡住：
- **`terraform fmt -check`** 拒绝未格式化的 HCL，PR 评审里再没"你跑 fmt 了吗"的废话。
- **`tfsec`** Checkov 风格的安全扫描，标记公开桶、未加密磁盘、过宽的 SG 规则。
- **plan 作为 PR 评论贴出来**，评审看的是**真实 plan 输出**，不是"信我的，看着没事"。

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

GitHub `environment` 把 prod apply 锁在必需评审者批准之后——另一个人（常常是值班）点 approve 才能跑。Apply 用和 plan 不同的 RAM 角色：plan 只读、apply 才有写。两把钥匙启动。

这套流水线一年抓到了五个真实故障：误翻 `prevent_destroy = false`、会打掉 VPC peering 的 CIDR 重叠、忘记钉 module 版本、把真 drift 遮住的 `ignore_changes`、混进 PR 的 `0.0.0.0/0` 安全组规则。每一个都是 PR 上的评论，merge 前修掉。

### 什么时候该升级到 Atlantis

GitHub Actions 是个不错的起点。当用 Terraform 的工程师超过 5 人，每 PR plan 评论、锁冲突、Actions 上的审批流，就开始吃力。Atlantis 是下一步：自托管的 webhook 服务器，监听 PR、自动跑 `terraform plan`、把计划评论到 PR、授权用户在 PR 里 `atlantis apply` 触发应用。

相比 Actions：
- Plan 在**你的 VPC 内**跑，不需要把 OSS state bucket 暴露给外部 runner
- 单服务器持锁、顺序 apply，没有并发 PR 的竞态
- 项目级配置（`atlantis.yaml`）按目录划归属，配独立审批策略

部署 Atlantis 本身就是一个 `vpc-baseline` + `compute` 练习——一台 ECS、一个 ALB、一个 RAM 角色。在一个 10 人团队跑两个月，效率提升很实在：plan 到 apply 的全周期从 25 分钟（Actions 排队 + 手动审批）降到 8 分钟。

5 人以下不建议上 Atlantis，运维它的成本不值。5 人以上一个季度回本。

> **实战建议：** 不管选哪条流水线，记住**一个主仓库、多个环境**。不要为生产 fork 一份开发的代码库。单仓库 + `env/dev.tfvars`、`env/staging.tfvars`、`env/prod.tfvars` 保持所有环境代码路径一致——这就是 IaC 的核心价值。按环境 fork 仓库是反模式，会一点点蚕食你来 IaC 想拿到的好处。

## 百炼/DashScope 怎么处理

在 LiteLLM 的视角里，DashScope 只是另一个 OpenAI 兼容端点。模型名是 `dashscope/qwen-max`、`dashscope/qwen-plus` 这类。API key 从百炼控制台拿。

如果想要更"阿里云原生"（用 STS 替代 API key），DashScope 部分接口确实支持 STS 鉴权。但 2026 年 API key 仍是主流，按上文用 KMS 轮换 key 是正确的运维姿势。等 STS 成为默认（路线图说大概 2027），这篇文章只需要改一个开关；轮换纪律不变。

> **实战建议：** 给 LiteLLM 设一个 `master_key`（环境变量 `LITELLM_MASTER_KEY`）。不设它，谁能访问网关谁就能给自己签 API key。设了之后，只有持主密钥的人才能签子密钥——而主密钥永远只在 Terraform 变量空间里。

## 你将获得什么

读完这篇，你拥有：
- 一个 URL，所有 Agent 都通过它调"LLM"
- 一个加新 provider 的入口（改 `litellm_config`、`terraform apply`）
- 一个轮换任意 provider key 的入口（改 `var.llm_keys`、`terraform apply`）
- 一条日志流（下篇细讲），每次请求的延迟、token 数、模型、Agent 都有
- 每个 Agent 的硬性 QPM 与预算上限——失控循环最多烧 ¥800/天，不是整月预算
- 一条 CI 流水线，挡住人会眼瞎放过的回归，进 prod 走两把钥匙
- 三个具体的明文泄露通道全部封死，不只是最显眼那个

这个网关是战略性资产。我帮每个团队部署完后，通常一个月内就会收到感谢——往往是有人不小心把 API key 提交进了 git，他们意识到轮换只是一个单行 PR，而不是开篇我经历过那场六小时火警。

## 接下来的内容

第七篇聚焦可观测性与成本控制：SLS 处理日志、ARMS 跟踪分布式链路、CloudMonitor 监控指标、超日预算阈值的钉钉告警，以及 SLS 驱动的成本看板，让你直观看到"哪个 Agent 在烧我的预算"。

第八篇是端到端实践，把第二篇到第七篇的内容收束成一次 `terraform apply`。
