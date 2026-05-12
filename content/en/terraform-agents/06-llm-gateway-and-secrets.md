---
title: "Terraform for AI Agents (6): LLM Gateway and Secrets Management"
date: 2026-03-22 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - KMS
  - API Gateway
  - LLM
  - AI Agents
categories: Terraform
lang: en
mathjax: false
series: terraform-agents
series_title: "Terraform for AI Agents on Alibaba Cloud"
series_order: 6
description: "Centralise LLM API access through one gateway: per-agent quotas, request logging, and zero secrets outside KMS. Terraform-provisioned API Gateway plus self-hosted LiteLLM on ECS, with DashScope/OpenAI/Anthropic keys rotating automatically through KMS Secrets Manager."
disableNunjucks: true
translationKey: "terraform-agents-6"
---

A pattern I see repeatedly in immature agent stacks: each agent has its own copy of `OPENAI_API_KEY` in its own `.env` file. Sometimes the same key, sometimes different ones, sometimes a colleague's personal key from when they prototyped. When the bill arrives nobody can tell which agent caused which token spend, and when a key leaks (it always does) you're playing whack-a-mole across a dozen `.env` files.

The real wake-up call hit me two years ago. A contractor finished his three-month engagement on a Friday, his laptop went home, and on the following Tuesday DashScope billing flagged 12 million tokens of `qwen-max` traffic from an IP we didn't recognise. His personal API key — copy-pasted into a side project — was still sitting in our agent's `.env`. Rotating it took six hours: three engineers, four repos, two CI pipelines, one panicked Slack thread. Never again.

This article addresses that pattern by building one **LLM gateway** that:

- Holds every provider key in KMS Secrets Manager (one bucket, one ACL, one rotation cadence)
- Authenticates agents via short-lived RAM-issued tokens — zero static AK on agent boxes
- Enforces per-agent QPM and daily token caps so a runaway loop costs ¥800/day, not your quarter
- Logs every request to SLS for forensics, cost attribution, and SOC-2 evidence
- Rotates keys without redeploying any agent — one PR, one apply, done

Two days of setup. Permanent operational dividend.

![Terraform for AI Agents (6): LLM Gateway and Secrets Management — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/illustration_1.png)

## The shape

![Centralised LLM gateway: one egress, one quota, one audit log](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig1_gateway_topology.png)

Agents on the left, providers on the right, gateway in the middle. Every agent's HTTP call to "an LLM" actually goes to the gateway, which decides which provider to dispatch to, attaches the right key, enforces quotas, and logs the result.

You have two options:

1. **Aliyun API Gateway in front of a custom backend** — most managed, easiest to wire quota plans, native RAM integration. Best when routing is "one model, one provider, just rate-limit me."
2. **Self-hosted LiteLLM on ECS behind an ALB** — most flexible, supports the long tail of providers (DeepSeek, Moonshot, Zhipu, your fine-tuned PAI endpoint), easier to extend with cost tracking and cross-provider fallback.

I use both, depending on the routing complexity. For a simple proxy with quotas, API Gateway is sufficient. For multi-provider routing with budget guards and circuit breakers — which most teams need within six months — LiteLLM on ECS is better. The rest of this article focuses on LiteLLM because it suits 80% of teams.

## Step 1: store every key in KMS Secrets Manager

The first rule: provider keys never appear in `.env` files, in `provider {}` blocks, in agent code, or in tfstate plaintext. They live in KMS Secrets Manager and the gateway pulls them at startup via STS.

![Secure vault for managing API keys and credentials](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/wanxiang_secret_vault.png)


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

The keys themselves come in via `var.llm_keys` — set with `-var-file=secrets.auto.tfvars` (gitignored) or `TF_VAR_llm_keys='{...}'` from a CI secret. They never live in your repository.

Cost note: KMS Secrets Manager bills around ¥0.4 per secret per month plus ¥0.03 per 10k API calls in Shanghai region. For four provider keys pulled twice an hour by two gateway boxes, the monthly bill is rounding error — under ¥10. The default service KMS key is free; only customer-managed CMKs cost ¥1/month each. Don't let "KMS sounds expensive" be the reason you stay on `.env` files.

> **Real-world tip:** When you rotate a provider key, change `secret_data` and bump `version_id`. KMS keeps the old version active for the recovery window so in-flight requests don't fail; new gateway pulls get the new version. Plan this in PR form so it's auditable.

## Step 2: a RAM role the gateway can assume

The gateway ECS or function needs permission to read these secrets — and **only** these secrets:

![Centralized API gateway routing requests to multiple LLM endpoints](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/wanxiang_api_gateway.png)


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

Three key points:

- **Resource-scoped policy.** Only these secrets, not `kms:GetSecretValue` on `*`. If the gateway box is compromised, the attacker cannot pivot to other KMS secrets — billing keys, RDS passwords, OSS buckets all stay sealed.
- **No long-lived AK.** The role is assumed by the ECS instance via metadata service. Zero static credentials on disk, in env, or in cloud-init.
- **`kms:Decrypt` is needed** even just to read the secret because secrets are KMS-encrypted at rest. Forgetting this is the #1 reason your gateway boots and then 401s on every fetch.

## Step 3: deploy LiteLLM on ECS

LiteLLM is the easiest open-source LLM proxy I know. It uses the OpenAI API format on the frontend and translates to each provider's format on the backend. Self-hosting it on ECS keeps things flexible.

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

Two `ecs.c7.large` boxes — 2 vCPU, 4 GB each — comfortably handle 200+ requests per second of pure proxy traffic. LiteLLM is async I/O-bound; CPU rarely tops 30%. Don't oversize. If traffic is bursty, put it in a scaling group and let CloudMonitor add nodes when CPU crosses 60% sustained.

`gateway-init.sh` does the boot:

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

Each instance now runs the gateway on port 4000 with all provider keys loaded into the process env — never on disk. The ALB in front fans out:

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

Agents now reach the gateway at `http://<alb-id>.cn-shanghai.alb.aliyuncs.com/v1/chat/completions` and never see a provider key. Internal-only ALB — no public IP, no listener on 443 facing the world. If an agent needs to call from outside the VPC, it goes through the bastion or CEN, never directly.

## Step 4: per-agent quotas

LiteLLM supports per-key quotas natively. The cleanest way to wire this through Terraform is to provision one LiteLLM "virtual key" per agent, each with its own QPM and token budget. Since LiteLLM stores these in its own database, you provision them via its API at apply time using a `null_resource`:

![Per-agent quota policy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig3_quota_table.png)

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

I'm not in love with `null_resource` + `local-exec` — it's the exit hatch for "the resource doesn't exist in the provider yet." But it works, and the alternative (writing a custom Terraform provider for LiteLLM) is more code than it's worth for one team. If LiteLLM ever ships an official Terraform provider, swap in a day.

The output: each agent gets a distinct `LITELLM_API_KEY` env var that the cloud-init script in article 4 reads. Quota violations return `429 Too Many Requests`, which agents must handle with exponential backoff — bake this into your shared HTTP client, don't trust each agent author to remember.

A note on numbers. The `schedule-agent` ceiling at 100k tokens/day and ¥40/day might look low. It is — and intentionally so. A scheduling agent that suddenly spikes to 2M tokens is almost certainly stuck in a planning loop, and a hard cap is a far better error than a ¥3000 surprise on the monthly bill. Tune ceilings to "10x the agent's last 30-day p99 daily usage" and revisit quarterly.

## Step 5: secret rotation flow

![Terraform for AI Agents (6): LLM Gateway and Secrets Management — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/illustration_2.png)

The whole point of putting keys in KMS Secrets Manager is rotation:

![Secret rotation flow — KMS as single source of truth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/06-llm-gateway-and-secrets/fig2_secret_rotation.png)

The lifecycle:

1. You change the `secret_data` in Terraform (or via the KMS API), bump `version_id` to `v2`
2. KMS keeps `v1` active for the rotation window (default 30 days)
3. Gateway instances re-pull on cold start; existing instances keep using the cached value until their next refresh (every 15min, configured in `gateway-init.sh`)
4. After 30 days, `v1` is disabled — anyone still using it gets `InvalidSecretVersion`
5. You confirm zero usage of `v1` via SLS, then promote `v2` and retire `v1`

For a team, codify this as a runbook and re-execute it quarterly even if nothing leaks. Keys that have lived longer than a quarter are by definition stale; treat staleness as a low-grade incident. The contractor story I opened with happened because nobody had rotated DashScope keys in 14 months. After this article, that scenario is impossible — even if you forget, the 30-day window forces the issue.

## Step 6: how plaintext still leaks (and how to plug it)

Step 1 protects the secret *at rest*. There are at least three other places plaintext leaks if you're not careful, and all three have bitten projects I've reviewed.

### Leak 1: terraform.tfstate

`alicloud_kms_secret.secret_data` is in your tfstate, in plaintext, every time you apply. Even with `sensitive = true` on the variable, the *value* lives in state JSON. The mitigation is layered:

1. **OSS bucket KMS-encryption** (article 2) — already done. Protects the state at rest.
2. **OSS bucket access policy** — restrict `oss:GetObject` to the CI runner role only, never developers.
3. **Use the `data` source pattern instead of putting plaintext in HCL.** When the secret is created out of band (e.g. by an HSM-rotation job or the KMS console), Terraform reads it but never authors it:

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

The principle: **Terraform should know the *name* of the secret, not the *value***. The runtime fetches the value from KMS via instance metadata. This is the single most important habit to build, and it inverts how most tutorials present secrets.

### Leak 2: CI logs

`terraform plan` output marks sensitive values as `(sensitive value)` if `sensitive = true` is set on the variable. *But* only on the variable — not on resource attributes that derive from it. A common slip:

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

Mark every output that derives from a sensitive value as `sensitive = true`:

```hcl
output "gateway_config_url" {
  value     = "https://gateway.example.com?key=${var.openai_key}"
  sensitive = true
}
```

For tfvars files, add them to `.gitignore` *and* configure CI to fail if they're committed:

```yaml
# .github/workflows/no-secrets.yml
- name: check no secrets in repo
  run: |
    if git ls-files | grep -E '\.auto\.tfvars$|secrets/'; then
      echo "ERROR: secret files committed"; exit 1
    fi
```

### Leak 3: provider debug logs

`TF_LOG=DEBUG terraform apply` is the quickest way to debug a provider issue. It is also the quickest way to dump every API request and response — including request bodies that contain secrets — to your terminal scrollback. I have seen Slack screenshots of this happen, twice in two different companies.

When you must use `TF_LOG`, redirect to a file with restrictive permissions, never paste:

```bash
TF_LOG=DEBUG terraform apply 2> /tmp/tf.log
chmod 600 /tmp/tf.log
# review locally; never paste verbatim
shred -u /tmp/tf.log    # delete when done
```

Better, use `TF_LOG_CORE=DEBUG` (Terraform-only) which usually isolates the issue without including provider request bodies.

## Step 7: plan-review-apply gates in CI

The `null_resource` for LiteLLM key generation is fine for one engineer. For a team with multiple agents, multiple environments, and shared on-call rotation, you want a structured CI pipeline that humans review at the diff level. Here's the GitHub Actions workflow I run:

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

Three things this enforces that humans skim past:

- **`terraform fmt -check`** rejects un-formatted HCL. No more "did you run fmt?" comments cluttering review.
- **`tfsec`** runs Checkov-style scans — flags public buckets, unencrypted volumes, overly broad SG rules.
- **The plan posted as a PR comment** means review reads *the actual plan output*, not "trust me bro, this looks fine."

The matching apply workflow:

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

GitHub's `environment` mechanism gates prod applies behind required reviewer approval — a separate human (often the on-call) clicks "approve" before apply runs. Apply uses a different RAM role than plan: plan only has read; apply has write. Two-key launch.

This pipeline has caught five real incidents in the year I've run it: an accidental `prevent_destroy = false` flip, a CIDR overlap that would have broken VPC peering, a forgotten module version pin, an `ignore_changes` that masked real config drift, and a security group rule with `0.0.0.0/0` that snuck into a PR. Each was a comment on the PR, fixed before merge.

### When to graduate to Atlantis

GitHub Actions is the right starting point. Once you have more than five engineers running Terraform, the operational overhead of per-PR plan comments, lock contention, and manual approvals starts to creak. Atlantis is the next step: a self-hosted webhook server that listens to PRs, runs `terraform plan` automatically, comments the plan, and applies on `atlantis apply` from authorised users.

Compared to Actions:

- Plans run *inside your VPC* — no need to give external runners access to the OSS state bucket.
- One persistent server holds the lock for sequential applies — no race conditions across concurrent PRs.
- Per-project config (`atlantis.yaml`) lets you scope which dirs are managed, with per-dir approval policies.

Provisioning Atlantis itself is a `vpc-baseline` + `compute` exercise — one ECS, one ALB, one RAM role. After two months on a 10-engineer team, the throughput improvement was concrete: plan-to-apply cycle dropped from 25 minutes (Actions queue + manual approval) to 8 minutes.

I don't recommend Atlantis below 5 engineers — the ops cost of running it isn't worth the throughput. Above 5, it pays for itself within a quarter.

> **Real-world tip:** Whichever pipeline you pick, **one master repo, many envs**. Don't fork your prod repo from dev. The single repo with `env/dev.tfvars`, `env/staging.tfvars`, `env/prod.tfvars` keeps the codepath identical across environments — which is the whole point of IaC. A forked-per-env layout is an anti-pattern that erodes the property you came for.

## Bailian / DashScope specifically

DashScope is just another OpenAI-compatible endpoint in LiteLLM's eyes. The model names are `dashscope/qwen-max`, `dashscope/qwen-plus`, etc. The API key is what you generate from the DashScope console.

If you want first-class Aliyun-native treatment (so you can use STS instead of an API key), DashScope supports STS-based auth on some endpoints — but in 2026 the API-key path is still the standard, and rotating the key via KMS as above is the right operational pattern. When STS becomes the default (the roadmap suggests 2027), this article becomes one config flip easier; the rotation discipline remains.

> **Real-world tip:** Set a `master_key` on LiteLLM (the `LITELLM_MASTER_KEY` env var). Without it, anyone who can reach the gateway can issue themselves an API key. With it, only the master can mint subordinate keys — and the master never leaves Terraform's variable space.

## What this gives you

After this article you have:

- One URL where every agent calls "the LLM"
- One place to add a new model provider (edit `litellm_config`, `terraform apply`)
- One place to rotate any provider key (edit `var.llm_keys`, `terraform apply`)
- One log stream (next article) showing every request, latency, token count, model, and agent
- Hard QPM and budget caps per agent — a runaway loop costs at most ¥800/day, not your entire month
- A CI pipeline that catches the regressions humans skim past, with two-key launch into prod
- Three concrete plaintext leak vectors closed off, not just the obvious one

The gateway is a strategic asset. Every team I've shipped one for has thanked me within a month — usually the first time someone's API key gets accidentally checked into git and they realise rotating it is a one-line PR instead of the six-hour fire drill I lived through.

## What's next

Article 7 is observability and cost control: SLS for logs, ARMS for traces, CloudMonitor for metrics, the budget alarm that pings DingTalk when daily LLM spend crosses a threshold, and the SLS-driven cost dashboard that lets you see "which agent is burning my budget."

Article 8 is the end-to-end walkthrough where everything in articles 2-7 lands as one `terraform apply`.
