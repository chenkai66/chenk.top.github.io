---
title: "Terraform for AI Agents (2): Provider, Auth, and Remote State on OSS"
date: 2026-03-14 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - Infrastructure as Code
  - AI Agents
categories: Terraform
lang: en
mathjax: false
series: terraform-agents
series_title: "Terraform for AI Agents on Alibaba Cloud"
series_order: 2
description: "Pinning the alicloud provider, picking between AK/SK, AssumeRole, and ECS RAM role auth, putting tfstate on OSS with Tablestore locking, and the workspace pattern that keeps dev/staging/prod from stomping each other. Plus the dozen failure modes that bite first-timers."
disableNunjucks: true
translationKey: "terraform-agents-2"
---

This is the article where you stop reading and start typing. By the end you will have:

1. The `alicloud` Terraform provider installed and version-pinned
2. Authentication wired up — through the right method, not the convenient one
3. Remote state on an OSS bucket with Tablestore-based locking
4. Three workspaces (`dev`, `staging`, `prod`) that share a backend but isolate state
5. A working `terraform plan` against an empty config

Nothing here provisions an agent yet. We're laying the foundation that every later article assumes.

![Terraform for AI Agents (2): Provider, Auth, and Remote State on OSS — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/illustration_1.jpg)

## Step 0: install Terraform

I won't dwell — the official `Install Terraform` doc covers all OSes. On macOS:

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version
# Terraform v1.9.x
# on darwin_arm64
```

Pin to a recent stable. The Aliyun docs are tested against `>= 0.12`, but on a fresh project you should use `>= 1.9`. There are real ergonomic improvements in newer versions (`for_each`, `optional()`, refined `moved` blocks).

## Step 1: pin the provider

Create a project directory and a `versions.tf`:

```hcl
# versions.tf
terraform {
  required_version = ">= 1.9"

  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230"   # any 1.230.x — minor patches OK, no major bumps
    }
  }
}
```

The `~> 1.230` constraint allows `1.230.0` through `1.230.x` but blocks `1.231.0`. This is the right default. Once you commit `.terraform.lock.hcl` to git (Terraform creates it on `terraform init`), you also lock the *exact* provider version and its checksum. If a teammate runs `terraform init` later, they get the same provider — bit-identical.

Pinning early is cheap insurance. The alicloud provider has shipped breaking changes between minor versions (last big one was the OSS bucket schema rework around 1.220). You will eventually need to upgrade — do it deliberately, in a PR, with the diff in plan output, not by accident on a teammate's laptop.

## Step 2: authenticate — three options, ranked

The provider needs Aliyun credentials. There are three real choices, in increasing order of professional acceptability:

![Three ways to authenticate the alicloud provider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig2_auth_methods.png)

### Option A: static AK/SK (only on a personal laptop)

```bash
export ALICLOUD_ACCESS_KEY="LTAI5tXXXXXXXXXXXXXX"
export ALICLOUD_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export ALICLOUD_REGION="cn-shanghai"
```

The provider auto-discovers these env vars. Do not — under any circumstances — write the keys into your `.tf` files. The state file does not store the secret; the `provider {}` block does, and that block is committed to git.

If the AK/SK is for a sub-account scoped to only the resources Terraform manages, this is acceptable for a solo project. For anything shared, skip to option B.

### Option B: AssumeRole (CI runners)

CI runners shouldn't carry long-lived AKs. Instead, give the CI runner an AK with one permission only — `sts:AssumeRole` on a target role — and have Terraform assume that role at apply time:

```hcl
provider "alicloud" {
  region = var.region

  assume_role {
    role_arn           = "acs:ram::${var.account_id}:role/TerraformDeployRole"
    session_name       = "ci-${var.commit_sha}"
    session_expiration = 3600
  }
}
```

The role has the actual write permissions; the AK only has the right to assume it. STS sessions are short-lived (one hour by default), audit-logged in ActionTrail, and can be revoked instantly by detaching the trust policy. This is the model GitLab CI, GitHub Actions, and Jenkins runners should use.

### Option C: ECS RAM role (the bastion / IaC service runner)

If `terraform apply` runs on an Aliyun ECS instance — say, your team's ops bastion or the Aliyun-hosted IaC Service runner — attach a RAM role to the instance and the provider picks credentials up automatically from instance metadata:

```hcl
provider "alicloud" {
  region = var.region
  # No assume_role block, no env vars — provider auto-detects from
  # http://100.100.100.200/latest/meta-data/ram/security-credentials/
}
```

Zero secrets in any config, in any env var, in any file. Rotation is automatic. This is the gold standard.

> **Real-world tip:** Whatever you pick, set `ALICLOUD_REGION` (or `provider { region = ... }`) explicitly. If unset, the provider does not pick a default — you get a confusing "Region must be specified" error on `terraform plan` that has tripped me up more than once.

## Step 3: state — why local tfstate is a footgun

When you run `terraform apply`, by default Terraform writes `terraform.tfstate` in the current directory. That file is the source of truth for what infrastructure exists. Three things will go wrong:

1. **Loss.** Delete the directory and Terraform thinks nothing exists. Next `apply` tries to recreate everything (or fails on duplicates).
2. **Conflict.** Two engineers running `apply` simultaneously can corrupt the state file.
3. **Secrets in plaintext.** Some resource attributes (database passwords, key material) end up in tfstate. Leaving it on a laptop is bad. Committing it to git is worse — and people do.

The fix is **remote state** with **state locking**. On Aliyun, the canonical setup is OSS + Tablestore:

![Remote state on OSS with Tablestore locking](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig1_state_backend.png)

OSS holds the actual `terraform.tfstate` file (with versioning enabled — recovery is one CLI command if something corrupts). Tablestore holds a tiny "lock" row that Terraform writes before any apply and deletes after. If a second `apply` starts while the first holds the lock, the second one waits or fails — never both running at once.

## Step 4: bootstrap the backend (chicken-and-egg)

The OSS bucket and Tablestore that hold our backend... need to exist before the backend can use them. The honest workflow is to provision them in a tiny one-off `bootstrap/` directory using a *local* state file, then never touch it again.

```hcl
# bootstrap/main.tf — run once, store local tfstate in this directory only
provider "alicloud" {
  region = "cn-shanghai"
}

resource "alicloud_oss_bucket" "tfstate" {
  bucket = "ck-tfstate-prod"
  acl    = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm = "KMS"
  }

  lifecycle {
    prevent_destroy = true   # never let terraform destroy this bucket
  }
}

resource "alicloud_ots_instance" "tflock" {
  name        = "tf-state-lock"
  description = "Terraform state lock"
  instance_type = "Capacity"
}

resource "alicloud_ots_table" "tflock" {
  instance_name = alicloud_ots_instance.tflock.name
  table_name    = "TerraformLock"
  primary_key {
    name = "LockID"
    type = "String"
  }
  time_to_live  = -1
  max_version   = 1
}
```

`terraform init && terraform apply` from inside `bootstrap/`. About 30 seconds. Then archive the local tfstate somewhere (I keep it in 1Password as a sanity backup) and never run from this directory again.

## Step 5: configure the backend

Back in your real project, add:

```hcl
# backend.tf
terraform {
  backend "oss" {
    bucket              = "ck-tfstate-prod"
    prefix              = "agents/"
    key                 = "terraform.tfstate"
    region              = "cn-shanghai"
    tablestore_endpoint = "https://tf-state-lock.cn-shanghai.ots.aliyuncs.com"
    tablestore_table    = "TerraformLock"
    encrypt             = true
  }
}
```

The `prefix` lets you stash multiple state files in one bucket — handy when you split your infra into multiple Terraform projects later. `encrypt = true` enables OSS-side encryption (we already turned on the bucket-level KMS rule, but defense-in-depth never hurts).

Run:

```bash
terraform init
# Initializing the backend...
# Successfully configured the backend "oss"!
# Initializing provider plugins...
# - Installing aliyun/alicloud v1.230.x...
```

If this fails with "AccessDenied", your auth role doesn't have `oss:GetObject`/`PutObject` on the bucket. The minimum role policy is:

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["oss:GetObject", "oss:PutObject", "oss:DeleteObject"],
      "Resource": ["acs:oss:*:*:ck-tfstate-prod/agents/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["ots:GetRow", "ots:PutRow", "ots:DeleteRow"],
      "Resource": ["acs:ots:*:*:instance/tf-state-lock/table/TerraformLock"]
    }
  ]
}
```

Apply this to the role you authenticate with. Don't grant `oss:*` — least privilege matters even for backend roles, because that role is in your CI runner.

## Step 6: workspaces for env isolation

![Terraform for AI Agents (2): Provider, Auth, and Remote State on OSS — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/illustration_2.jpg)

A workspace is a separate state file inside the same backend. The default workspace is — usefully — called `default`. Create the others you need:

```bash
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

terraform workspace list
#   default
# * prod
#   dev
#   staging

terraform workspace select dev
```

Inside HCL, `terraform.workspace` resolves to the current workspace name, which lets you parameterise resource sizes:

```hcl
locals {
  is_prod = terraform.workspace == "prod"

  ecs_count        = local.is_prod ? 3 : 1
  ecs_instance_type = local.is_prod ? "ecs.c7.xlarge" : "ecs.c7.large"
  rds_class         = local.is_prod ? "pg.x4.large.2c" : "pg.n2.medium.1c"
}
```

A clean alternative is one `*.tfvars` file per env:

```bash
terraform plan -var-file=env/dev.tfvars
terraform plan -var-file=env/prod.tfvars
```

I use `tfvars` files for "configuration that obviously differs" (CIDR blocks, region, instance counts) and `terraform.workspace` only for the conditional `is_prod` toggle. Mixing both is fine — pick one as the primary mechanism per project.

## Step 7: the five-command loop

Day-to-day Terraform is just five commands:

![The five-command loop you'll run hundreds of times](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png)

```bash
terraform fmt        # normalise indentation; pre-commit hook
terraform validate   # static schema check; runs in <1s
terraform plan       # diff desired vs real; READ THIS CAREFULLY
terraform apply      # send the API calls
terraform show       # inspect current state
```

Three rules:

1. **Always read the plan output before applying.** It tells you exactly what's about to happen — which resources will create (`+`), update in-place (`~`), force replace (`-/+`), or destroy (`-`). The replace-in-place arrows in particular hide downtime.
2. **Make `plan` and `apply` two steps in CI.** Run `terraform plan -out=tfplan`, post the plan output to the PR, get human approval, then `terraform apply tfplan` on merge. Never auto-apply on push.
3. **Don't rush past `state`.** `terraform state list` shows everything you currently manage; `terraform state show <addr>` shows one resource's full attributes. When you're debugging weird drift, this is where you start.

## The eight failure modes you will hit on day one

In the order they happened to me:

1. **`Error: Failed to query available provider packages` on `terraform init`.** GFW. Set `HTTPS_PROXY` or use the official `Configure an acceleration solution for Terraform initialization` doc — the registry mirror is `https://mirrors.aliyun.com/terraform/`.
2. **`Error: state lock`.** You hit Ctrl-C during a previous apply and the lock is stale. Run `terraform force-unlock <LOCK_ID>` (the ID is in the error). Verify nothing's running first.
3. **`Error: Region must be specified`.** Set `ALICLOUD_REGION` env var or `region` in the `provider` block.
4. **`AccessDenied` on backend init.** RAM permissions on the OSS bucket prefix. Re-check step 5's policy.
5. **`InvalidParameter.NotFound` on Tablestore.** You bootstrapped the wrong region. Tablestore endpoint and OSS bucket region must match.
6. **`Provider produced inconsistent result after apply`.** Almost always a stale `.terraform/` cache after a provider version bump. `rm -rf .terraform .terraform.lock.hcl && terraform init`.
7. **`Resource already exists`.** You created the resource by hand in the console. Either delete it or import: `terraform import alicloud_vpc.main vpc-uf6xxxxxx`.
8. **A `terraform plan` diff you didn't expect on a freshly-applied resource.** "Drift". Either someone touched the resource in the console, or the provider's read logic differs from create. Look at the specific attributes in the diff; usually the fix is to set the attribute explicitly so Terraform stops "noticing" the difference.

> **Real-world tip:** Run `terraform plan` immediately after every `apply`, even on no changes. The plan should be empty. If it isn't, you have drift, and the longer you let drift live the harder it is to reconcile.

## What's next

Article 3 builds the first real piece of infrastructure: a reusable `vpc-baseline` module. VPC, three vSwitches across three zones, NAT gateway, EIP, security group baseline, KMS key. We will use it in every subsequent article and it is the single most copy-pasted module in my agent stacks.

If this article worked end-to-end for you, you should now be able to run `terraform init`, `terraform workspace select dev`, `terraform plan` and see "No changes." That's the foundation. Everything else stacks on top of it.

## State surgery: when `apply` won't unstick a problem

The five-command loop covers 95% of days. The other 5% are when something is wrong with the state file itself, and the right fix is direct surgery — not a panicked `terraform destroy && apply`. Here are the four manoeuvres I reach for, in increasing order of nervousness.

### `terraform import` — adopting a manually-created resource

You created a VPC in the console six months ago. Now you want Terraform to manage it. The wrong move is to write the HCL and `apply` — Terraform will try to *create a second VPC*. The right move is `import`:

```bash
# Write the HCL stub first, with the right resource address
cat > vpc.tf <<EOF
resource "alicloud_vpc" "legacy" {
  vpc_name   = "legacy-prod"
  cidr_block = "172.16.0.0/16"
}
EOF

# Import the existing resource into the state at that address
terraform import alicloud_vpc.legacy vpc-uf6abc123def456

# Now diff. The HCL almost certainly doesn't match reality yet.
terraform plan
# Plan: 0 to add, 1 to change, 0 to destroy.
#   ~ tags = { ... }   # the console-set tags Terraform doesn't know about
```

Then iterate the HCL until `terraform plan` shows no changes. This is the only safe way to bring console-built infra under management. Newer Terraform supports the `import` block, which makes this declarative — set `import { to = alicloud_vpc.legacy; id = "vpc-..." }` and import happens on the next `plan`.

### `terraform state rm` — telling Terraform to forget a resource

The opposite of import. You decided to manage a resource in a different stack, or hand it back to the ops team. You don't want to *destroy* it, you want Terraform to stop tracking it:

```bash
terraform state rm alicloud_oss_bucket.legacy_archive
# Removed alicloud_oss_bucket.legacy_archive
```

The bucket still exists in OSS. Terraform's state no longer references it. Subsequent `plan` won't try to destroy it (because it's not in state) and won't try to create it (because the HCL is gone too — remove the HCL block in the same PR).

I use this for the chicken-and-egg bootstrap directory from earlier in this article. After we move the OSS+Tablestore management into the main stack, I `terraform state rm` them from the bootstrap state and delete the bootstrap directory entirely.

### `terraform state mv` — renaming or restructuring

You want to move resources from one module path to another — say, `alicloud_vpc.this` → `module.vpc.alicloud_vpc.this`. Without `mv`, Terraform sees this as "destroy old, create new" and would actually delete and recreate your production VPC. With `mv`:

```bash
terraform state mv alicloud_vpc.this module.vpc.alicloud_vpc.this
# Move "alicloud_vpc.this" to "module.vpc.alicloud_vpc.this"
# Successfully moved 1 object(s).
```

Now the state thinks the resource was always at the new address. The next `plan` shows no changes. This is how every refactor of a Terraform repo without downtime is done.

Modern Terraform (`>= 1.1`) gives you the `moved` block, which is the declarative version:

```hcl
moved {
  from = alicloud_vpc.this
  to   = module.vpc.alicloud_vpc.this
}
```

Always prefer `moved` over `state mv` — it's checked into git, it's reviewable in PR, and it works for everyone on the team without anyone having to run a manual command.

### `terraform taint` (legacy) and `-replace` (current) — force-recreating one resource

An ECS instance is in a weird state — disk full, kernel panic, who knows. You want Terraform to destroy and recreate just that one instance without touching anything else:

```bash
terraform apply -replace=alicloud_instance.agent[1]
# Plan: 1 to add, 0 to change, 1 to destroy.
#   ~ alicloud_instance.agent[1] is tainted, so must be replaced
```

The old `terraform taint` command does the same thing but is deprecated. `-replace` is the current spelling. I use this maybe once a quarter when an ECS instance gets into an unrecoverable state and a stop/start doesn't help.

> **Real-world tip:** Before any state surgery, `terraform state pull > backup.tfstate` so you have a known-good copy. State surgery is one of the few places where a typo can lose you hours; the backup turns it into a 10-second restore (`terraform state push backup.tfstate`).

## What goes in tfstate (and why you must encrypt the bucket)

A common misconception: tfstate just stores resource IDs. False. The state file contains every attribute Terraform knows about every resource — including computed ones like RDS connection strings, generated passwords, KMS key material, and `sensitive` variables.

Run this on any non-trivial state and look at the output:

```bash
terraform show -json | jq '.values.root_module.resources[] | select(.values | tostring | test("password|secret|key"; "i"))'
```

You will see passwords. You will see API keys. You may see entire JSON blobs of credentials. This is by design — Terraform needs the values for diff and apply.

Three implications:

1. **The OSS bucket holding state must be encrypted.** We did this in the bootstrap step with `server_side_encryption_rule { sse_algorithm = "KMS" }`. If you skipped it, anyone who can `oss:GetObject` can read your secrets.
2. **Do not commit `terraform.tfstate` or `*.tfstate.backup` to git.** Add them to `.gitignore` immediately. I have seen production keys leak this way more than once.
3. **Mark sensitive variables with `sensitive = true`.** This stops Terraform from printing them in plan/apply output, which prevents them from leaking into CI logs:

```hcl
variable "rds_admin_password" {
  type      = string
  sensitive = true
}
```

The value still lives in tfstate (encrypted), but at least your GitHub Actions log doesn't paste the password into the world.

For the truly paranoid, Terraform supports a `-target` filter for `state pull` (per-resource), and you can write a wrapper that pipes state through `sops` or `age` for at-rest encryption beyond what OSS provides. I haven't found that necessary as long as the OSS bucket itself is KMS-encrypted with access strictly controlled.

## Workspace vs separate state files: the real call

The article above showed `terraform workspace new dev/staging/prod`. This is the simple answer and works for most teams. But there's a real architectural choice hiding inside it, and you should make it deliberately rather than by default.

**Workspaces (one project, N states):**
- One backend, prefix splits by workspace name (`agents/env:dev/terraform.tfstate`)
- One set of HCL files, parameterised by `terraform.workspace`
- Pro: the diff between dev and prod is visible — same code, different vars
- Con: a typo in a `count` expression can affect every workspace; one PR can't isolate "only changes prod"

**Separate state files (one project per env):**
- Separate directories: `envs/dev/`, `envs/staging/`, `envs/prod/`
- Each env has its own `backend.tf`, `main.tf` calling shared modules
- Pro: prod has its own PR review, its own apply, its own permissions
- Con: code duplication; bumping a module version needs N edits; hard to enforce "all envs run the same code"

My rule of thumb: **workspaces below 5 engineers, separate states above.** Below 5, the simplicity wins. Above 5, the blast radius matters — you want a prod PR that only touches `envs/prod/` and is approved by the on-call rotation, not a `dev.tfvars` change that accidentally cascades.

For this series I use workspaces because the focus is one engineer or a small team. Be aware of the upgrade path though: `terraform state pull` in the workspace + `terraform state push` in the new project is how you migrate, and it's not bad if you only have to do it once.

> **Real-world tip:** The workspace name is *just a string*. Set it via `TF_WORKSPACE=prod terraform plan` in CI to avoid the "I forgot to switch workspace" class of incident. Combine with branch protection so that the prod workspace can only be applied from the `main` branch.
