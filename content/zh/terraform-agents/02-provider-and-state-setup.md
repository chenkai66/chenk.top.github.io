---
title: "Terraform 实战（二）：Provider 认证与 State"
date: 2026-03-14 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - Infrastructure as Code
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 2
description: "钉死 alicloud provider 版本，在 AK/SK、AssumeRole、ECS RAM role 三种认证方式之间正确选择，把 tfstate 放到 OSS 并用 Tablestore 加锁，再加上让 dev/staging/prod 不互相踩脚的 workspace 模式。还有初学者第一天必踩的十几个坑。"
disableNunjucks: true
translationKey: "terraform-agents-2"
---
读到这里，关掉页面，打开终端吧。等你回来时，应该已经准备好以下内容：

1. 安装好且版本锁定的 `alicloud` Terraform Provider。
2. 配置妥当的认证方式——用的是正确的方法，而非图省事的做法。
3. 基于 OSS Bucket 和 Tablestore 锁定的远程状态存储。
4. 三个工作空间（`dev`、`staging`、`prod`），共用后端但状态相互隔离。
5. 能跑通的 `terraform plan`，哪怕配置文件还是空的。

至此，Agent 尚未部署——本阶段仅搭建基础设施底座，后续所有文章都以此为基础。如果跳过此步骤，等到第三篇文章再临时补救，一周内遭遇 tfstate 损坏的概率极高。

---

## Step 0: 安装 Terraform

安装过程我不赘述，官方《Install Terraform》文档已覆盖所有操作系统。macOS 用户可直接运行：

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version
# Terraform v1.9.x
# on darwin_arm64
```

建议锁定一个近期的稳定版本。虽然阿里云文档测试过 `>= 0.12`，但新项目请直接使用 `>= 1.9`。新版在体验上有实实在在的改进——比如 `for_each`、`optional()`、更精细的 `moved` 块，以及声明式的 `import` 块——本系列后续文章都会用到这些特性。

## Step 1: 锁定 Provider

创建项目目录，并编写 `versions.tf`：

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

`~> 1.230` 这个约束允许 `1.230.0` 到 `1.230.x`，但会阻止升级到 `1.231.0`。这是合理的默认做法。一旦将 `.terraform.lock.hcl` 提交到 Git（Terraform 在 `terraform init` 时自动生成），你就锁定了 *确切* 的 Provider 版本及其校验和。此后队友运行 `terraform init` 时，获取的 Provider 与你完全一致，比特级相同。

尽早锁定版本是一项低成本高回报的风险控制措施。alicloud Provider 曾在小版本间引入破坏性变更（例如 1.220 附近对 OSS Bucket schema 的重构，让我耗费了整整三个下午）。升级不可避免，但应主动推进：通过 PR 明确升级，审查 `plan` 输出的差异，而非在深夜 11 点因队友机器上的意外操作触发升级。

## Step 2: 认证——三种方案，按专业程度排序

Provider 需要阿里云凭证。以下是三种方案，按专业性和安全性递增排序：

![认证流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/wanxiang_auth_flow.png)


![三种认证阿里云提供商的方法](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig2_auth_methods.png)

### 方案 A：静态 AK/SK（仅限个人笔记本）

```bash
export ALICLOUD_ACCESS_KEY="LTAI5tXXXXXXXXXXXXXX"
export ALICLOUD_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export ALICLOUD_REGION="cn-shanghai"
```

Provider 会自动发现这些环境变量。**切勿**——在任何情况下——将密钥硬编码进 `.tf` 文件。状态文件本身不存储密钥，但 `provider {}` 块会，而该块通常会被提交到 Git。

如果该 AK/SK 属于一个 RAM 子账号，且权限严格限定于 Terraform 所管理的资源，那么单人项目可以接受。但只要是协作项目，请直接跳到方案 B。

### 方案 B：AssumeRole（CI Runner）

CI Runner 不应持有长期有效的 AK。应为其分配一个仅具备 `sts:AssumeRole` 权限的 AK（作用于目标角色），让 Terraform 在 `apply` 时动态扮演该角色：

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

角色本身拥有实际的写权限；AK 仅用于扮演角色。STS 会话默认有效期为一小时，会在 ActionTrail 中留下审计日志，并可通过移除信任策略立即撤销。这是 GitLab CI、GitHub Actions 和 Jenkins 等 CI/CD 环境的推荐做法。

### 方案 C：ECS RAM 角色（堡垒机 / IaC 服务 Runner）

如果 `terraform apply` 运行在阿里云 ECS 实例上——无论是团队的运维堡垒机，还是阿里云托管的 IaC Service Runner——只需为实例绑定 RAM 角色，Provider 便会自动从实例元数据中获取凭证：

```hcl
provider "alicloud" {
  region = var.region
  # No assume_role block, no env vars — provider auto-detects from
  # http://100.100.100.200/latest/meta-data/ram/security-credentials/
}
```

配置、环境变量或文件中均无需存放任何密钥，轮换也由系统自动完成。这是业界公认的黄金标准，我建议每个团队在落地 Terraform 的首月内就向此方案靠拢。

> **实战建议：** 无论选择哪种方案，务必显式设置 `ALICLOUD_REGION`（或在 `provider` 块中指定 `region = ...`）。若未设置，Provider 不会自动选用默认值，而是在 `terraform plan` 时抛出令人困惑的 “Region must be specified” 错误——这个问题我曾多次踩坑。

## Step 3: 状态文件——为什么本地 tfstate 是隐患

运行 `terraform apply` 时，默认会在当前目录生成 `terraform.tfstate` 文件。该文件是基础设施当前状态的唯一权威来源，但存在三大风险：

![分布式状态锁定](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/wanxiang_state_lock.png)

1. **丢失**：删除项目目录后，Terraform 会认为基础设施不存在，下次 `apply` 将尝试重建所有资源（或因资源重复而失败）。
2. **冲突**：两名工程师同时运行 `apply` 可能导致状态文件损坏。
3. **明文泄露敏感信息**：某些资源属性（如 RDS 密码、KMS 密钥材料、生成的 Token）会写入 tfstate。将其留在笔记本上已属高危，若误提交到 Git 则后果更严重——而现实中真有人这么干。

### tfstate 里到底存了什么？

一个常见误解是：tfstate 只存储资源 ID。这是错误的。状态文件包含 Terraform 所知的每个资源的所有属性——包括计算得出的值，例如 RDS 连接字符串、自动生成的密码、KMS 密钥材料，以及标记为 `sensitive` 的变量。

在任意非平凡的 tfstate 上运行以下命令，查看其内容：

```bash
terraform show -json | jq '.values.root_module.resources[] | select(.values | tostring | test("password|secret|key"; "i"))'
```

你会看到密码，看到 API Key，甚至可能看到完整的凭证 JSON 对象。这是 Terraform 的设计使然——它需要这些值来计算差异（diff）并执行变更（apply）。

正因如此，本地 tfstate 根本撑不过第一天。解决方案是 **远程状态** 加 **状态锁定**。在阿里云上，标准组合是 OSS + Tablestore：

![使用 Tablestore 锁定的 OSS 远程状态](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig1_state_backend.png)

OSS 存储实际的 `terraform.tfstate` 文件（建议开启版本控制——一旦损坏，一条 CLI 命令即可恢复）；Tablestore 存储一个微小的“锁”记录，Terraform 在 `apply` 前写入，结束后删除。若第一个 `apply` 持有锁时第二个启动，后者会等待或失败——绝不会同时运行。

基于 tfstate 的内容，有两条不可妥协的原则：

- **存储状态的 OSS Bucket 必须启用 KMS 加密**。我们将在 Step 4 中开启此功能。若跳过，任何拥有 `oss:GetObject` 权限的人都能读取你的敏感信息。
- **将敏感变量标记为 `sensitive = true`**，防止 `plan`/`apply` 输出将其泄露到 CI 日志中：

```hcl
variable "rds_admin_password" {
  type      = string
  sensitive = true
}
```

值仍会存在于 tfstate 中（但已加密），至少 GitHub Actions 日志不会再把密码公之于众。

## Step 4: 引导后端（先有鸡还是先有蛋）

用于承载后端的 OSS Bucket 和 Tablestore 实例，必须在 Terraform 后端配置生效前预先创建。合理的做法是：使用 *本地状态文件*，在一个一次性 `bootstrap/` 目录中完成初始化，之后永不触碰。

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
  name          = "tf-state-lock"
  description   = "Terraform state lock"
  instance_type = "Capacity"
}

resource "alicloud_ots_table" "tflock" {
  instance_name = alicloud_ots_instance.tflock.name
  table_name    = "TerraformLock"
  primary_key {
    name = "LockID"
    type = "String"
  }
  time_to_live = -1
  max_version  = 1
}
```

在 `bootstrap/` 目录内运行 `terraform init && terraform apply`，耗时约 30 秒。随后将本地 tfstate 归档（我习惯存入 1Password 作为 sanity backup），并彻底弃用该目录。**务必在忘记前**，将 `bootstrap/terraform.tfstate*` 加入 `.gitignore`。

## Step 5: 配置后端

回到主项目目录，添加以下配置：

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

`prefix` 允许你在同一 Bucket 中存放多个状态文件——未来若将基础设施拆分为多个 Terraform 项目，这将非常有用。`encrypt = true` 启用 OSS 侧加密（我们已在 Bucket 级别启用 KMS 规则，但纵深防御永不过时）。

运行以下命令初始化后端：

```bash
terraform init
# Initializing the backend...
# Successfully configured the backend "oss"!
# Initializing provider plugins...
# - Installing aliyun/alicloud v1.230.x...
```

若报 `AccessDenied`，说明认证角色缺少对 Bucket 的 `oss:GetObject`/`PutObject` 权限。最小权限策略如下：

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

将此策略绑定到你的认证角色。**切勿授予 `oss:*`**——即使对后端角色，最小权限原则也至关重要，因为该角色运行在 CI Runner 中，一旦 AK 泄露，攻击者将能读取你所有的状态文件。

## Step 6: 用 Workspace 隔离环境

Workspace 本质上是同一后端中的独立状态文件。默认 Workspace 名为 `default`，非常实用。其他环境可手动创建：

![开发、测试和生产环境的多环境工作区隔离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/wanxiang_workspace.png)


```bash
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

terraform workspace list
#   default
#   dev
#   staging
# * prod

terraform workspace select dev
```

在 HCL 中，`terraform.workspace` 会解析为当前 Workspace 名称，可用于参数化资源配置：

```hcl
locals {
  is_prod = terraform.workspace == "prod"

  ecs_count         = local.is_prod ? 3 : 1
  ecs_instance_type = local.is_prod ? "ecs.c7.xlarge" : "ecs.c7.large"
  rds_class         = local.is_prod ? "pg.x4.large.2c" : "pg.n2.medium.1c"
}
```

另一种清晰的做法是为每个环境准备独立的 `*.tfvars` 文件：

```bash
terraform plan -var-file=env/dev.tfvars
terraform plan -var-file=env/prod.tfvars
```

我通常用 `tfvars` 文件处理“明显因环境而异”的配置（如 CIDR 块、地域、实例数量），而仅用 `terraform.workspace` 作为 `is_prod` 的条件开关。两者混用无妨，但每个项目应明确以其中一种为主要机制。

### Workspace 还是独立状态文件：真正的抉择

本文目前展示的是 `terraform workspace new dev/staging/prod`，这是简单方案，适用于大多数团队。但背后隐藏着一个关键架构决策，应主动选择，而非默认接受。

**Workspaces（单项目，N 个状态）**：共用一个后端，通过前缀区分状态（如 `agents/env:dev/terraform.tfstate`）；一套 HCL 文件，通过 `terraform.workspace` 参数化。优点：dev 与 prod 的差异清晰可见——代码相同，变量不同。缺点：`count` 表达式中的笔误可能波及所有环境；一次 PR 无法做到“仅修改 prod”。

**独立状态文件（每环境单项目）**：分别创建 `envs/dev/`、`envs/staging/`、`envs/prod/` 目录；每个环境拥有独立的 `backend.tf` 和 `main.tf`（调用共享模块）。优点：prod 拥有独立的 PR 审查、独立的 `apply` 流程和独立的 RAM 权限。缺点：代码重复；模块版本升级需多处修改；难以强制“所有环境运行相同代码”。

我的经验法则是：**5 人以下团队用 Workspace，5 人以上拆分为独立状态文件**。5 人以下，简洁性胜出；5 人以上，爆炸半径成为关键考量——你希望 prod 的变更仅影响 `envs/prod/`，并由 on-call 轮值人员审批，而非因 `dev.tfvars` 的修改意外 cascades 到生产环境。

本系列采用 Workspace，因目标读者为单人或小团队。若未来规模扩大，迁移路径依然可行：从 Workspace 执行 `terraform state pull`，再 `terraform state push` 到新项目，最后退役旧 Workspace。这种痛苦最多经历一次。

> **实战建议：** Workspace 名称 *只是一个字符串*。在 CI 中通过 `TF_WORKSPACE=prod terraform plan` 显式设置，避免“忘记切换 Workspace”类事故。配合分支保护策略，确保 prod Workspace 仅能从 `main` 分支触发 `apply`。

## Step 7: 五命令循环

日常 Terraform 工作流其实只有五个核心命令：

![你将运行数百次的五命令循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png)

```bash
terraform fmt        # 标准化缩进；预提交 hook
terraform validate   # 静态 schema 检查；1 秒内跑完
terraform plan       # 对比期望 vs 现实；这步得仔细看
terraform apply      # 发送 API 调用
terraform show       # 检查当前状态
```

三条铁律：

1. **Apply 前务必阅读 plan 输出**。它会明确告诉你即将发生什么——哪些资源将被创建（`+`）、原地更新（`~`）、强制替换（`-/+`）或销毁（`-`）。特别是原地替换操作，往往意味着服务中断。
2. **在 CI 中将 `plan` 与 `apply` 拆分为两步**。先运行 `terraform plan -out=tfplan`，将 plan 输出附在 PR 中供人工审批，合并后再执行 `terraform apply tfplan`。切勿在 push 后自动 apply。
3. **不要忽视 `state` 命令**。`terraform state list` 列出当前管理的所有资源；`terraform state show <addr>` 显示单个资源的完整属性。调试诡异的 drift 时，这是你的第一站。

## Step 8: 状态手术，解决那 5% `apply` 搞不定的日子

五命令循环覆盖 95% 的日常场景。剩下的 5%，问题出在状态文件本身，此时正确的做法是精准“手术”，而非慌乱地 `terraform destroy && apply`。以下是我常用的四种操作，按风险程度递增排列。

### `terraform import` —— 纳管手动创建的资源

假设你六个月前在控制台创建了一个 VPC，现在想交由 Terraform 管理。错误做法是直接编写 HCL 并 `apply`——Terraform 会尝试 *创建第二个 VPC*。正确做法是 `import`：

```bash
# 先写 HCL  stub，地址要对
cat > vpc.tf <<EOF
resource "alicloud_vpc" "legacy" {
  vpc_name   = "legacy-prod"
  cidr_block = "172.16.0.0/16"
}
EOF

# 把现有资源 import 到该地址的状态里
terraform import alicloud_vpc.legacy vpc-uf6abc123def456

# 现在 diff。HCL 几乎肯定跟现实还不匹配。
terraform plan
# Plan: 0 to add, 1 to change, 0 to destroy.
#   ~ tags = { ... }   # 控制台设置的标签 Terraform 还不知道
```

随后调整 HCL，直到 `terraform plan` 显示无变更。这是将控制台资源纳入 Terraform 管理的唯一安全方式。Terraform `>= 1.5` 支持声明式的 `import` 块，这是我现在的首选：

```hcl
import {
  to = alicloud_vpc.legacy
  id = "vpc-uf6abc123def456"
}
```

### `terraform state rm` —— 让 Terraform 忘记某个资源

这是 `import` 的反向操作。当你决定将某资源移交至其他 Stack 或交还给运维团队时，你不想 *销毁* 它，只想让 Terraform 停止追踪：

```bash
terraform state rm alicloud_oss_bucket.legacy_archive
# Removed alicloud_oss_bucket.legacy_archive
```

Bucket 仍存在于 OSS 中，但 Terraform 状态不再引用它。后续 `plan` 既不会尝试销毁（因不在状态中），也不会尝试创建（因 HCL 已移除——记得在同一 PR 中删除对应代码）。

我常用此操作处理“鸡生蛋”问题的 `bootstrap/` 目录。一旦 OSS + Tablestore 的管理逻辑被整合进主 Stack，我就会从 bootstrap 状态中 `terraform state rm` 这些资源，并彻底删除该目录。

### `terraform state mv` —— 重命名或重构

你想将资源从一个模块路径迁移至另一个——例如 `alicloud_vpc.this` → `module.vpc.alicloud_vpc.this`。若不使用 `mv`，Terraform 会视为“销毁旧资源，创建新资源”，真可能删除并重建你的生产 VPC。使用 `mv`：

```bash
terraform state mv alicloud_vpc.this module.vpc.alicloud_vpc.this
# Move "alicloud_vpc.this" to "module.vpc.alicloud_vpc.this"
# Successfully moved 1 object(s).
```

此时状态认为资源始终位于新地址，下次 `plan` 将显示无变更。所有无停机的 Terraform 重构都依赖此操作。

现代 Terraform（`>= 1.1`）提供了声明式的 `moved` 块：

```hcl
moved {
  from = alicloud_vpc.this
  to   = module.vpc.alicloud_vpc.this
}
```

**永远优先使用 `moved` 而非 `state mv`**——它可提交至 Git，在 PR 中被审查，且团队成员无需手动执行命令即可生效。

### `terraform apply -replace` —— 强制重建单个资源

某个 ECS 实例处于异常状态——磁盘满、内核崩溃，原因不明。你希望 Terraform 仅销毁并重建该实例，不影响其他资源：

```bash
terraform apply -replace=alicloud_instance.agent[1]
# Plan: 1 to add, 0 to change, 1 to destroy.
```

旧版 `terraform taint` 功能相同但已被弃用。`-replace` 是当前标准写法。我大约每季度使用一次，用于处理 stop/start 无法恢复的 ECS 实例。

> **实战建议：** 任何状态手术前，先执行 `terraform state pull > backup.tfstate` 保存已知良好的副本。状态操作是少数几个因笔误即可导致数小时损失的场景；有了备份，恢复只需 10 秒（`terraform state push backup.tfstate`）。

## 第一天就会遇到的八种报错

按我踩坑的顺序排列：

1. **`Error: Failed to query available provider packages`（`terraform init` 时）**：GFW 干扰。设置 `HTTPS_PROXY` 或使用阿里云镜像：`https://mirrors.aliyun.com/terraform/`。
2. **`Error: state lock`**：上次 `apply` 被 Ctrl-C 中断，锁未释放。运行 `terraform force-unlock <LOCK_ID>`（ID 在错误信息中）。**务必先确认无其他进程正在运行**。
3. **`Error: Region must be specified`**：未设置 `ALICLOUD_REGION` 环境变量或 `provider` 块中的 `region`。
4. **`AccessDenied`（后端初始化时）**：OSS Bucket 前缀的 RAM 权限不足。复查 Step 5 中的策略。
5. **`InvalidParameter.NotFound`（Tablestore 相关）**：引导时选错地域。Tablestore Endpoint 与 OSS Bucket 地域必须一致。
6. **`Provider produced inconsistent result after apply`**：几乎总是因 Provider 升级后 `.terraform/` 缓存过期。解决方法：`rm -rf .terraform .terraform.lock.hcl && terraform init`。
7. **`Resource already exists`**：你在控制台手动创建了资源。要么删除它，要么导入（见 Step 8）。
8. **刚 `apply` 完的资源在 `terraform plan` 中出现意外差异**：存在 drift。可能是有人在控制台修改了资源，或 Provider 的读取逻辑与创建逻辑不一致。查看 diff 中的具体属性；通常解法是显式设置该属性，让 Terraform 不再“察觉”差异。

> **实战建议：** 每次 `apply` 后立即运行 `terraform plan`，即使预期无变更。理想情况下输出应为空。若非空，说明存在 drift——drift 存留越久，修复难度越大。

## 下一步
如果本文操作顺利，你现在应能运行 `terraform init`、`terraform workspace select dev`、`terraform plan`，并看到 “No changes.”。这就是地基，后续一切皆构建于此。

第三篇文章将搭建首个真实基础设施组件：一个可复用的 `vpc-baseline` 模块，包含 VPC、跨三个可用区的 vSwitch、NAT 网关、EIP、安全组基线和 KMS 密钥。该模块将在后续每篇文章中复用——它是我 Agent 栈中被复制粘贴最多的模块。
