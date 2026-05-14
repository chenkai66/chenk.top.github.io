---
title: "Terraform for AI Agents (3): A Reusable VPC and Security Baseline"
date: 2026-03-16 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - VPC
  - Security
  - AI Agents
categories: Terraform
lang: en
mathjax: false
series: terraform-agents
series_title: "Terraform for AI Agents on Alibaba Cloud"
series_order: 3
description: "The first reusable module — a three-zone VPC with public/private subnets, NAT egress, security groups layered by tier, and KMS keys per data domain. The same code shows up in every agent stack I've shipped, parameterised but otherwise unchanged."
disableNunjucks: true
translationKey: "terraform-agents-3"
---

This article builds the single most copied piece of Terraform in my agent projects: a `vpc-baseline` module that gives every later component (ECS, RDS, OpenSearch, ACK) a sane place to land. It's about 200 lines of HCL all-in. Worth typing once, refer to it forever.

By the end you'll have:

- A VPC across three availability zones in one region
- Six vSwitches (one public + one private per zone) with non-overlapping CIDRs
- A NAT Gateway with EIP for private-subnet outbound to LLM APIs
- Three security groups stacked by tier (ALB → agent runtime → memory)
- Three KMS customer master keys, one per data domain (memory, secrets, logs)
- A clean module interface: `name + CIDR + zones` in, IDs out
- Drift detection in CI, semver-pinned module references, and a per-line cost model

---

## The mental model

Before code, the picture:

![VPC topology — 3 zones, public + private, NAT egress](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig1_vpc_topology.png)

Why three zones? Because Aliyun reserves the right to do a zone-level maintenance on any given Sunday, and a single-zone deployment means your agents are offline for the whole window. Cross-zone traffic inside a VPC is free; the only cost of three zones is the operational complexity of subnet math, which Terraform absorbs for you.

Why public + private? The agent runtime should live in private vSwitches so a misconfigured security group can't accidentally expose it on `0.0.0.0/0`. Public vSwitches hold the ALB and the NAT Gateway — things that *must* reach the internet. The agent reaches the internet via NAT, never directly.

The CIDR layout I use:

| Subnet      | Zone | CIDR             | Hosts |
|-------------|------|------------------|------:|
| `public-a`  | l    | `10.20.0.0/28`   |    11 |
| `public-b`  | m    | `10.20.0.16/28`  |    11 |
| `public-c`  | n    | `10.20.0.32/28`  |    11 |
| `private-a` | l    | `10.20.1.0/24`   |   251 |
| `private-b` | m    | `10.20.2.0/24`   |   251 |
| `private-c` | n    | `10.20.3.0/24`   |   251 |

Public is `/28` because it only holds a NAT and an ALB IP. Private is `/24` because that's where the agent ECS, RDS, OpenSearch nodes live. If you think `/24` is tight for agent fleets, multiply by three zones — 753 usable IPs is more than any single agent app I've shipped.

## The module skeleton

Create the directory layout:

```text
modules/vpc-baseline/
├── main.tf
├── variables.tf
├── outputs.tf
└── versions.tf
```

Inputs (`variables.tf`):

```hcl
variable "name" {
  description = "Name prefix for all resources, e.g. agents-prod"
  type        = string
}

variable "cidr_block" {
  description = "Top-level VPC CIDR; subnets are derived from this"
  type        = string
  default     = "10.20.0.0/16"
}

variable "zones" {
  description = "Three availability zone IDs in the target region"
  type        = list(string)
  validation {
    condition     = length(var.zones) == 3
    error_message = "vpc-baseline requires exactly 3 zones."
  }
}

variable "tags" {
  description = "Tags applied to every resource the module creates"
  type        = map(string)
  default     = {}
}
```

Requiring exactly three zones is opinionated but aligns with the diagram. If you need two or four zones, fork the module—don't make it conditional. Conditional modules become unreadable and often need to be rewritten from scratch six months later.

## VPC and vSwitches

`main.tf`, part one:

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
  cidr_block   = cidrsubnet(var.cidr_block, 12, each.key)        # /28 starting at .0
  vswitch_name = "${var.name}-public-${substr(each.value, -1, 1)}"
  tags         = var.tags
}

resource "alicloud_vswitch" "private" {
  for_each = { for i, z in var.zones : i => z }

  vpc_id       = alicloud_vpc.this.id
  zone_id      = each.value
  cidr_block   = cidrsubnet(var.cidr_block, 8, each.key + 1)     # /24 starting at .1.0
  vswitch_name = "${var.name}-private-${substr(each.value, -1, 1)}"
  tags         = var.tags
}
```

Three things worth noting:

- `cidrsubnet(prefix, newbits, netnum)` is Terraform's CIDR math. `cidrsubnet("10.20.0.0/16", 8, 1)` returns `"10.20.1.0/24"`. Memorise it — you'll use it constantly.
- `for_each` with the index/value map gives stable resource addresses — `alicloud_vswitch.private["0"]` always points to the first zone, even if you rearrange the list. Compare to `count`, where reordering causes wholesale recreation.
- `substr(each.value, -1, 1)` extracts the last char of the zone ID (the `l`/`m`/`n`) so resource names sort nicely in the console.

## NAT Gateway and EIP

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

Enhanced NAT is the modern type — required for Tablestore, PrivateLink, and most newer services. The legacy "Standard" NAT will be deprecated; don't start a new project on it. PayByTraffic is right for agent workloads where outbound bandwidth is bursty (LLM streaming) rather than steady.

The SNAT entries are what actually let private-subnet instances reach the internet. Without them, an agent in `private-a` cannot resolve `dashscope.aliyuncs.com` — and you will spend an hour debugging that the first time it bites you. I have the scar.

## Security groups, layered

The right way to do security groups on Aliyun is **one SG per tier**, with rules that reference SG IDs not CIDRs:

![Multi-layer network security architecture with firewall barriers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/wanxiang_network_layers.png)


![Security group strategy — tight ingress, loose egress, layered](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig2_sg_layers.png)

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

The key line is `source_security_group_id = alicloud_security_group.alb_public.id`. It says "accept inbound 8080 only from any instance in the ALB SG" — not from a CIDR. Re-IPing the ALB later doesn't break anything.

> **Real-world tip:** Aliyun's default behaviour is to *deny* all ingress and *allow* all egress. The default is correct — don't add a "deny all egress" rule, you'll just break SDK calls. Limit egress only when you have a specific compliance requirement; for an agent system, all-egress-open is normal.

I extend this pattern for every downstream tier:

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

By the time you're done, attaching an ECS to the right SG is just `security_groups = [module.vpc.agent_runtime_sg_id]` and the network tier is correct by construction. Audit becomes trivial: you grep for the SG name and find every resource that ever held it.

## KMS keys per data domain

Encryption-at-rest is mandatory for any compliance regime worth its salt. The Aliyun way is **one Customer Master Key (CMK) per data domain**, so you can rotate one without touching another and audit access per-key.

![Data encryption at rest and in transit with key management](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/wanxiang_encryption.png)


![KMS encryption — one CMK per data domain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/03-vpc-and-security-baseline/fig3_kms_encrypt.png)

```hcl
locals {
  cmks = {
    memory  = "Encryption for RDS data and OSS objects"
    secrets = "Encryption for KMS Secrets Manager entries"
    logs    = "Encryption for SLS log data"
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

Why the alias? Because the CMK ID is a UUID nobody remembers; the alias `alias/agents-prod-memory` is human-readable and stable across key rotations. Reference the alias from RDS, OSS, and SLS, and you can swap the underlying key without touching downstream config.

`pending_window_in_days = 7` means a deleted key has a 7-day window where you can recover it. Don't shorten this — accidental key deletion is the kind of mistake that ends careers, and the recovery window has saved me more than once.

## The module outputs

`outputs.tf`:

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

These are exactly the IDs the next five articles need. By naming and shaping outputs deliberately, callers can do:

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

## Calling the module

In your top-level `main.tf`:

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

`terraform plan` from the project root will produce something like:

```text
Plan: 27 to add, 0 to change, 0 to destroy.
```

27 resources is about right (1 VPC + 6 vSwitch + 1 NAT + 1 EIP + 1 EIP-assoc + 3 SNAT + 4 SG + 4 SG-rule + 3 KMS key + 3 KMS alias = 27). Apply, and you have a production-grade network in about 90 seconds.

## Drift detection: when the live VPC stops matching the HCL

Networks drift. Someone opens a port in the console at 11pm to debug a thing. Someone adds an SNAT rule to test a workaround. The route table gets a temporary entry that nobody removes. Six months later the prod VPC and the HCL diverge silently — until the next `terraform apply` either reverts the manual change (breaking whoever depended on it) or, worse, confuses the provider's update logic into recreating a resource.

The fix is to *find* drift early and treat it as a real signal. Three patterns I run on every VPC stack:

### Pattern 1: nightly `terraform plan` in CI

A GitHub Actions workflow that runs `terraform plan -lock=false -detailed-exitcode` on every workspace at 3am Beijing time, and posts to DingTalk if the exit code is `2` ("plan would make changes"):

```yaml
# .github/workflows/drift-check.yml
name: drift-check
on:
  schedule:
    - cron: '0 19 * * *'   # 3am Beijing
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

The `-detailed-exitcode` flag is the trick. Without it, `plan` always returns 0 even when it would make changes. With it, you get 0 (no changes), 1 (error), or 2 (changes pending). The CI cares about 2 — that means drift.

I run this against every prod workspace nightly. Once a fortnight it catches something — usually a teammate's "quick fix" they meant to put in the HCL but forgot.

### Pattern 2: refresh-only on suspicion

When you suspect drift on a single resource, `terraform apply -refresh-only` is the surgical tool. It re-reads the resource from the API and updates state, but doesn't apply HCL changes:

```bash
terraform apply -refresh-only
# Terraform has detected the following changes made outside of Terraform since
# the last "terraform apply":
#   ~ resource "alicloud_security_group_rule" "agent_from_alb" {
#       port_range = "8080/8080" -> "8080/8090"   # someone widened it
#     }
```

Once you see the diff, you decide: revert (regular `apply`) or codify (edit HCL to match the world).

### Pattern 3: the `lifecycle { ignore_changes }` escape valve

Sometimes drift is *legitimate*. Aliyun adds metadata tags to resources (e.g. `created_by_console`). Auto Scaling adjusts `desired_capacity` outside Terraform's purview. The right fix is to tell Terraform "this attribute will drift, and that's fine":

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

`ignore_changes` is a precision instrument. Don't put `tags` (the whole map) in it — you'll mask real drift. Put the specific keys you know are externally managed.

## Module versioning: treat it like a library

This article's `vpc-baseline` is version 1. Eighteen months from now it will be version 4. New zones will exist. The default NAT type might change. You'll discover that the `/28` public subnets are too small once you add an internet-facing NLB.

The wrong way to handle this is "edit the module in place and `terraform apply` everywhere". One Friday afternoon you'll change the public subnet from `/28` to `/27`, the existing subnets will need recreation, and you'll cascade-destroy your NAT and EIPs across three environments. (Yes, also a personal scar.)

The right way is **versioned modules** with an explicit upgrade path:

```hcl
module "vpc" {
  source = "git::ssh://git@github.com/your-org/terraform-modules.git//vpc-baseline?ref=v1.4.0"

  name       = "agents-${terraform.workspace}"
  cidr_block = "10.20.0.0/16"
  zones      = local.zones
}
```

Every breaking change to the module bumps a major version (semver). Consumers upgrade deliberately, one workspace at a time, with a `plan` review on each. When something blows up in `dev`, you don't roll out to `prod` — you roll back the `?ref=` tag, file an issue, fix the module.

For a small team I use a single repo for modules with git tags as versions. For a larger org, the Terraform Registry's private module support (or the Aliyun-hosted equivalent) gives you a published artefact with a UI. Either way the principle is the same: **modules are libraries, not snippets**. Treat them with the same release discipline you'd give a Python package.

The practical ergonomics of an upgrade:

```bash
cd envs/dev
sed -i 's|?ref=v1.3.0|?ref=v1.4.0|' main.tf
terraform init -upgrade
terraform plan          # review carefully, especially the destroys
terraform apply
```

If `dev` survives a week, repeat in `staging`. Then `prod`. The whole rollout is in PRs, each one small, each one reversible by reverting the commit.

> **Real-world tip:** When a breaking module change requires resource recreation (e.g. our `/28` → `/27` subnet expansion), use a `moved` block plus a manual data migration. The `moved` block tells Terraform "this old subnet's identity is now this new one"; the migration copies any state across. For VPC subnets specifically, the easier path is to add new subnets *alongside* and migrate workloads zone-by-zone — never destroy a production subnet that has live ECS in it.

## Cost arithmetic for the network baseline

Roughly, in `cn-shanghai`, the baseline costs ¥150–300/month at low-to-moderate traffic. That number is real but worth breaking apart so you can size for your own traffic.

Fixed costs (you pay these even with zero traffic):

| Item                       | Monthly (cn-shanghai) | Notes |
|----------------------------|----------------------:|-------|
| VPC + vSwitch + RT         | ¥0                    | free at any scale |
| Security groups            | ¥0                    | free, 100 SGs/account hard limit |
| KMS keys (3, software)     | ¥9                    | ¥3/mo per CMK |
| EIP reservation            | ¥18                   | ¥0.6/day if unattached; attached EIPs are free to hold |
| NAT (Enhanced) reservation | ¥120                  | ¥4/day for Enhanced NAT |
| **Fixed total**            | **~¥147/mo**          |       |

Variable costs:

| Item                 | Unit price                   | Example |
|----------------------|-----------------------------:|---------|
| EIP outbound traffic | ¥0.8/GB BGP, ¥0.3/GB on-peak | 100 GB/mo agent traffic = ¥80 |
| KMS API calls        | ¥0.005/call after free tier  | 100k calls/mo = ¥500 |
| NAT inter-zone       | free inside same VPC         | nothing |

For a low-traffic dev workspace (10 GB egress, 1k KMS calls): ¥147 + ¥8 + ¥0 ≈ **¥155/mo**.

For a medium-traffic prod workspace (1 TB egress, 100k KMS calls — heavy LLM streaming): ¥147 + ¥800 + ¥500 ≈ **¥1,450/mo**.

The lever is egress. If your agents stream long completions from public LLM endpoints, you want PrivateLink or VPC peering to hosted models when available — PrivateLink traffic is ~¥0.1/GB instead of ¥0.8/GB. For DashScope, the PrivateLink endpoint is `com.aliyun.dashscope`; wire it into your VPC and your egress bill drops by ~80%.

> **Real-world tip:** Tag every resource with `Cost-Center` and `Owner`. Aliyun's billing dashboard pivots by tag, and at end of quarter you can answer "this team's network cost was ¥X" without asking finance. The `tags = var.tags` plumbing in this module is what makes it work.

## What's next

Article 4 lands compute on this network. Three patterns — ECS with `pm2`, ACK for production fleets, Function Compute for event-driven agents — and the cost-crossover model I use to pick between them. Then a real `alicloud_instance` block that bootstraps Python + Node + the agent runtime via cloud-init.

> **Real-world tip:** If you ever need to add a fourth zone (Aliyun adds them periodically), it's a `terraform apply` away — the `for_each` pattern handles a longer list cleanly. But the `validation` block in `variables.tf` will reject it, so you'll first relax the validation. That deliberate friction is the point — adding a zone is a network change worth thinking about, not a typo to slip in.
