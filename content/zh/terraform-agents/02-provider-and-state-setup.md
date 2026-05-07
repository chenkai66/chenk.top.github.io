---
title: "用 Terraform 给 AI Agent 上云（二）：Provider、认证与 OSS 上的远程 State"
date: 2026-03-14 09:00:00
tags:
  - Terraform
  - 阿里云
  - 基础设施即代码
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 2
description: "钉死 alicloud provider 版本，在 AK/SK、AssumeRole、ECS RAM role 三种认证方式之间正确选择，把 tfstate 放到 OSS 并用 Tablestore 加锁，再加上让 dev/staging/prod 不互相踩脚的 workspace 模式。还有初学者第一天必踩的十几个坑。"
disableNunjucks: true
translationKey: "terraform-agents-2"
---

这一篇你不再是读，是开始动手。读完之后你会有：

1. `alicloud` Terraform provider 装好且版本钉死
2. 认证接好——用对的方式，不是方便的方式
3. 远程 state 放在 OSS，用 Tablestore 加锁
4. 三个 workspace（`dev`、`staging`、`prod`），共用 backend、隔离 state
5. 一个能跑通的 `terraform plan`（即使配置是空的）

本篇还不会建出任何 Agent 资源。我们在打的是后续每一篇都会假设的地基。

![用 Terraform 给 AI Agent 上云（二）：Provider、认证与 OSS 上的远程 State — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/illustration_1.jpg)

## 第 0 步：安装 Terraform

这里就不赘述了——官方的《Install Terraform》文档已经涵盖了所有操作系统。如果你用的是 macOS，可以参考以下步骤：

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version
# Terraform v1.9.x
# on darwin_arm64
```

建议固定到一个近期的稳定版本。阿里云的文档基于 `>= 0.12` 进行测试，但如果是一个全新的项目，推荐使用 `>= 1.9`。新版本在易用性上做了不少改进，比如新增了 `for_each` 和 `optional()` 功能，还优化了 `moved` 块的使用体验。
## 第一步：锁定 Provider 版本

首先，创建一个项目目录，并在其中添加一个名为 `versions.tf` 的文件，内容如下：

```hcl
# versions.tf
terraform {
  required_version = ">= 1.9"

  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230"   # 允许 1.230.x，但拒绝 1.231.0
    }
  }
}
```

这里的 `~> 1.230` 表示允许使用 `1.230.x` 系列的版本（例如 `1.230.0` 到 `1.230.x`），但会阻止升级到 `1.231.0` 或更高版本。这是一个非常合理的默认设置。当你将 `.terraform.lock.hcl` 文件提交到 Git 仓库后（该文件会在运行 `terraform init` 时自动生成），不仅 Provider 的版本会被锁定，其校验和也会被固定下来。这样一来，团队中的其他成员在运行 `terraform init` 时，都会使用完全相同的 Provider 版本——字节级别的一致性。

尽早锁定版本是一种低成本的“保险”。过去，alicloud Provider 曾在次版本之间引入过破坏性变更（比如 1.220 版本附近对 OSS Bucket Schema 的重构）。虽然升级是不可避免的，但应该有计划地进行：通过 PR 提交，查看 `terraform plan` 的差异输出，而不是让某个同事的本地环境无意中触发升级。
## 第二步：认证方式——三种方案，按推荐程度排序

Terraform 的 `alicloud` 提供器需要阿里云的认证信息。以下是三种可行的认证方式，按照专业性和安全性从低到高排列：

![三种 alicloud 提供器认证方式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig2_auth_methods.png)

### 方案一：静态 AK/SK（仅适用于个人开发环境）

如果你是在自己的笔记本上运行 Terraform，可以通过环境变量设置 AccessKey 和 SecretKey：

```bash
export ALICLOUD_ACCESS_KEY="LTAI5tXXXXXXXXXXXXXX"
export ALICLOUD_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export ALICLOUD_REGION="cn-shanghai"
```

提供器会自动读取这些环境变量。**切记**，不要将密钥直接写入 `.tf` 文件中。虽然状态文件（state file）不会存储密钥，但 `provider {}` 配置块会包含这些信息，并且该配置块通常会被提交到版本控制系统（如 Git）。

如果 AccessKey 和 SecretKey 是为一个子账号生成的，并且该子账号的权限仅限于 Terraform 管理的资源范围，那么这种方式在个人项目中是可以接受的。但在团队协作或共享环境中，请直接跳过此方案，选择更安全的方式。

---

### 方案二：AssumeRole（适合 CI/CD 环境）

在 CI/CD 环境中，不应该使用长期有效的 AccessKey。更好的做法是为 CI 跑步机（runner）分配一个仅具备 `sts:AssumeRole` 权限的 AccessKey，然后让 Terraform 在执行时动态地扮演指定的角色（Role）。配置如下：

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

在这个方案中，实际的资源操作权限由 RAM 角色（Role）持有，而 AccessKey 只有扮演该角色的权利。STS 会话的有效期较短（默认 1 小时），并且所有操作都会被记录在 ActionTrail 中。如果需要撤销权限，只需移除角色的信任策略即可。这种模式非常适合 GitLab CI、GitHub Actions 或 Jenkins 等 CI/CD 平台。

---

### 方案三：ECS RAM 角色（最佳实践，适合运维堡垒机或托管服务）

如果 `terraform apply` 运行在阿里云的 ECS 实例上（例如团队的运维堡垒机，或者阿里云托管的 IaC 服务 runner），可以直接为该实例绑定一个 RAM 角色。Terraform 提供器会自动从实例元数据中获取认证信息，无需额外配置：

```hcl
provider "alicloud" {
  region = var.region
  # 不需要 assume_role 块，也不需要环境变量，提供器会自动从以下路径读取：
  # http://100.100.100.200/latest/meta-data/ram/security-credentials/
}
```

这种方式完全避免了在配置文件、环境变量或代码中暴露任何敏感信息。RAM 角色的密钥轮转由阿里云自动完成，因此是最安全、最推荐的方案。

---

> **实用建议：**  
无论选择哪种认证方式，都务必显式设置 `ALICLOUD_REGION`（或在 `provider { region = ... }` 中定义）。如果不设置，提供器不会自动选择默认区域，这会导致在执行 `terraform plan` 时出现令人困惑的错误提示：“Region must be specified”。我在这方面踩过多次坑，提醒大家注意。
## 第 3 步：State——为什么本地 tfstate 是个隐患

当你执行 `terraform apply` 时，Terraform 默认会在当前目录生成一个名为 `terraform.tfstate` 的文件。这个文件是记录基础设施状态的唯一真相源。然而，使用本地 tfstate 文件会带来三个主要问题：

1. **数据丢失风险**  
   如果不小心删除了工作目录，Terraform 会认为所有资源都不存在了。下次运行 `apply` 时，它可能会尝试重新创建所有资源（或者因为资源已存在而报错）。

2. **并发冲突**  
   如果两位工程师同时运行 `apply`，很可能会导致 state 文件被破坏，进而引发不可预知的问题。

3. **敏感信息明文存储**  
   某些资源属性（例如数据库密码、密钥材料等）会被直接写入 tfstate 文件。如果这个文件留在个人电脑上，安全隐患已经不小；更糟糕的是，有些人甚至会把它提交到 git 仓库中，后果不堪设想。

解决这些问题的最佳实践是启用 **远程 state** 并配合 **state 锁机制**。在阿里云上，推荐的方案是使用 OSS 和 Tablestore 的组合：

![OSS + Tablestore 实现远程 state 与锁机制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig1_state_backend.png)

具体来说，OSS 用于存储实际的 `terraform.tfstate` 文件，并且可以开启版本控制功能。如果 state 文件意外损坏，只需一条 CLI 命令就能快速恢复。而 Tablestore 则用来管理一个轻量级的“锁”记录：Terraform 在每次执行 `apply` 前会写入锁，完成后自动删除。如果第二个 `apply` 操作在第一个操作持有锁时启动，它会进入等待状态或直接失败，从而避免了两个操作同时运行的风险。
## 第 4 步：bootstrap backend（先有鸡还是先有蛋）

承载 backend 的 OSS bucket 和 Tablestore……得先于 backend 存在。诚实的做法是用一个单独的 `bootstrap/` 目录、用 *本地* state file 把它们建出来，然后再也不动它。

```hcl
# bootstrap/main.tf —— 跑一次，只在这个目录里留 local tfstate
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
    prevent_destroy = true   # 永远别让 terraform destroy 掉这个 bucket
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

进 `bootstrap/`，`terraform init && terraform apply`。约 30 秒。然后把 local tfstate 归档到某处（我放 1Password 里做最后一道备份），从此不再从这个目录跑 Terraform。

## 第 5 步：配置后端存储

回到你的实际项目中，添加以下内容：

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

`prefix` 参数允许你在一个 OSS 存储桶中存放多个状态文件——这在后续将基础设施拆分为多个 Terraform 项目时非常实用。`encrypt = true` 则启用了 OSS 端的加密功能（虽然我们已经在存储桶级别启用了 KMS 加密规则，但多一层防护总是好的）。

执行以下命令初始化后端：

```bash
terraform init
# Initializing the backend...
# Successfully configured the backend "oss"!
# Initializing provider plugins...
# - Installing aliyun/alicloud v1.230.x...
```

如果遇到 `AccessDenied` 错误，说明当前使用的认证角色缺少对存储桶的 `oss:GetObject` 和 `oss:PutObject` 权限。以下是所需的最小权限策略：

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

将上述策略绑定到用于认证的角色上。注意不要直接授予 `oss:*` 权限——即使是后端存储相关的角色，也应遵循最小权限原则，因为该角色通常运行在 CI 环境中，安全性尤为重要。
## 第 6 步：通过 Workspace 实现环境隔离

![用 Terraform 给 AI Agent 上云（二）：Provider、认证与 OSS 上的远程 State — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/illustration_2.jpg)

Workspace 是存储在同一个后端中的独立状态文件。默认的 Workspace 很贴心地被命名为 `default`。根据需要创建其他 Workspace：

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

在 HCL 中，`terraform.workspace` 会解析为当前的 Workspace 名称，利用这一点可以动态调整资源配置规模：

```hcl
locals {
  is_prod = terraform.workspace == "prod"

  ecs_count        = local.is_prod ? 3 : 1
  ecs_instance_type = local.is_prod ? "ecs.c7.xlarge" : "ecs.c7.large"
  rds_class         = local.is_prod ? "pg.x4.large.2c" : "pg.n2.medium.1c"
}
```

另一种简洁的方式是为每个环境单独准备一份 `*.tfvars` 文件：

```bash
terraform plan -var-file=env/dev.tfvars
terraform plan -var-file=env/prod.tfvars
```

我个人习惯用 `tfvars` 文件来管理那些“显而易见会因环境而异的配置”（比如 CIDR 范围、区域、实例数量），而只用 `terraform.workspace` 来实现类似 `is_prod` 这样的条件开关逻辑。两种方式混合使用也没问题——只需在项目中明确一种作为主要机制即可。
## 第 7 步：五条核心命令的循环

在日常使用 Terraform 的过程中，其实只需要记住五条核心命令：

![你会反复执行数百次的五条命令循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png)

```bash
terraform fmt        # 格式化代码缩进，适合作为 pre-commit hook
terraform validate   # 静态校验配置文件结构，耗时不到 1 秒
terraform plan       # 对比期望状态与实际状态，务必仔细阅读输出
terraform apply      # 发送 API 请求以应用变更
terraform show       # 查看当前的状态详情
```

三条黄金法则：

1. **apply 前必须认真阅读 plan 输出。** 它会明确告诉你即将发生哪些变更：哪些资源会被创建（`+`）、更新（`~`）、强制替换（`-/+`）或删除（`-`）。尤其是那些标记为“原地更新但实际重建”的变更，可能会引发服务中断，需要特别留意。
2. **在 CI 流程中将 plan 和 apply 分成两个独立步骤。** 先运行 `terraform plan -out=tfplan`，将生成的计划输出贴到 PR 中，供团队成员审核并批准；等合并代码后，再执行 `terraform apply tfplan`。切记不要在代码推送时自动触发 apply。
3. **重视 `state` 相关命令的作用。** 使用 `terraform state list` 可以查看当前管理的所有资源列表；通过 `terraform state show <addr>` 可以查看某个资源的完整属性信息。当你遇到难以解释的状态漂移问题时，这些命令往往是排查问题的第一步。
## 第一天必踩的八个坑

按照它们坑到我的顺序整理如下：

1. **执行 `terraform init` 时遇到 `Error: Failed to query available provider packages`。** 原因是 GFW 的网络限制。可以通过设置 `HTTPS_PROXY` 环境变量，或者参考官方文档《Configure an acceleration solution for Terraform initialization》，使用阿里云提供的镜像地址：`https://mirrors.aliyun.com/terraform/`。

2. **报错 `Error: state lock`。** 通常是之前运行 `terraform apply` 时按了 Ctrl-C，导致状态锁残留。解决方法是运行 `terraform force-unlock <LOCK_ID>`（错误信息中会提供 LOCK_ID）。不过在操作前，请务必确认当前没有其他 Terraform 操作正在运行。

3. **提示 `Error: Region must be specified`。** 需要设置 `ALICLOUD_REGION` 环境变量，或者在 `provider` 配置块中显式指定 `region` 参数。

4. **初始化后端时出现 `AccessDenied` 错误。** 这是因为 OSS 存储桶前缀的 RAM 权限配置有问题。请重新检查第 5 步中的权限策略（policy），确保权限正确。

5. **Tablestore 报错 `InvalidParameter.NotFound`。** 可能是因为初始化时选择了错误的区域（region）。请注意，Tablestore 的 endpoint 和 OSS 存储桶必须位于同一个 region。

6. **执行 `apply` 后提示 `Provider produced inconsistent result after apply`。** 这种情况几乎总是因为升级了 provider 版本后，`.terraform/` 缓存未清理干净。建议直接删除缓存目录并重新初始化：`rm -rf .terraform .terraform.lock.hcl && terraform init`。

7. **报错 `Resource already exists`。** 这通常是因为你手动在控制台创建了该资源。解决方法有两种：要么在控制台删除该资源，要么通过 `terraform import` 将其导入到状态文件中，例如：`terraform import alicloud_vpc.main vpc-uf6xxxxxx`。

8. **刚执行完 `apply`，立刻运行 `terraform plan` 却出现了 diff。** 这种现象被称为“漂移”（Drift）。可能的原因有两种：一是有人在控制台手动修改了资源；二是 provider 的读取逻辑与创建逻辑不一致。仔细查看 diff 中的具体属性，通常的解决方法是将相关属性显式写入配置文件，避免 Terraform 再次检测到差异。

> **实战小贴士：** 每次执行完 `terraform apply` 后，即使没有任何改动，也建议立即运行一次 `terraform plan`。正常情况下，plan 应该为空。如果出现了非预期的 diff，说明已经发生了漂移（Drift）。漂移存在的时间越长，后续修复的难度就越大，因此尽早处理非常重要。
## 接下来的内容

在第三篇文章中，我们将构建第一个真正意义上的基础设施模块：一个可复用的 `vpc-baseline` 模块。这个模块包含了 VPC、跨三个可用区的三个 vSwitch、NAT 网关、EIP（弹性公网 IP）、安全组基线配置以及 KMS 密钥。在后续的所有文章中，我们都会用到这个模块，它也是我所有 Agent 栈中最常被复制粘贴的一个模块。

如果你已经顺利完成了本文的所有步骤，那么现在应该可以依次执行 `terraform init`、`terraform workspace select dev` 和 `terraform plan`，并看到“No changes.”的输出结果。这说明基础环境已经准备就绪。后续的所有内容都将基于这个基础逐步展开。
## State 手术：当 `apply` 无能为力时的解决方案

在日常使用 Terraform 的过程中，五个核心命令（`plan`、`apply`、`destroy`、`import`、`state`）已经能够覆盖 95% 的场景。然而，剩下的 5% 往往是由于 state 文件本身存在问题，而解决这些问题的关键在于直接对 state 进行手术，而不是慌乱地执行 `terraform destroy && apply`。以下是我在实际工作中常用的四种操作方法，按照风险和复杂度从低到高排序。

### `terraform import`——将手动创建的资源纳入管理

假设半年前你通过控制台手动创建了一个 VPC，现在希望将其交给 Terraform 管理。错误的做法是直接编写 HCL 并运行 `apply`，因为 Terraform 会尝试**再创建一个新的 VPC**。正确的做法是使用 `import` 命令：

```bash
# 先编写 HCL 占位代码，确保资源地址正确
cat > vpc.tf <<EOF
resource "alicloud_vpc" "legacy" {
  vpc_name   = "legacy-prod"
  cidr_block = "172.16.0.0/16"
}
EOF

# 将现有资源导入到 state 中的指定地址
terraform import alicloud_vpc.legacy vpc-uf6abc123def456

# 检查差异。此时 HCL 很可能与实际情况不完全匹配。
terraform plan
# Plan: 0 to add, 1 to change, 0 to destroy.
#   ~ tags = { ... }   # 控制台设置的标签，Terraform 尚未感知
```

接下来需要反复调整 HCL，直到 `terraform plan` 显示没有变更为止。这是将手动创建的基础设施安全纳入 Terraform 管理的唯一方法。较新的 Terraform 版本支持 `import` 块，可以将这一过程声明化。只需添加如下配置，下次运行 `plan` 时会自动完成导入：

```hcl
import {
  to = alicloud_vpc.legacy
  id = "vpc-uf6abc123def456"
}
```

### `terraform state rm`——让 Terraform 停止追踪某个资源

这是 `import` 的反向操作。如果你决定将某个资源转移到其他栈中管理，或者交还给运维团队，通常并不希望销毁它，而是让 Terraform 停止对其的追踪：

```bash
terraform state rm alicloud_oss_bucket.legacy_archive
# Removed alicloud_oss_bucket.legacy_archive
```

执行后，OSS 中的 bucket 仍然存在，但 Terraform 的 state 文件不再引用它。后续的 `plan` 不会尝试销毁它（因为它已不在 state 中），也不会尝试重新创建（因为对应的 HCL 配置也一并删除了）。

我经常用这招处理项目初期的 bootstrap 目录。例如，当我们将 OSS 和 Tablestore 的管理迁移到主栈后，就可以从 bootstrap 的 state 中移除这些资源，并彻底删除 bootstrap 目录。

### `terraform state mv`——重命名或重构资源路径

当你需要将资源从一个模块路径移动到另一个路径时（例如从 `alicloud_vpc.this` 移动到 `module.vpc.alicloud_vpc.this`），如果不使用 `mv`，Terraform 会将其视为“销毁旧资源并创建新资源”，从而可能导致生产环境中的 VPC 被删除并重建。为了避免这种风险，可以使用 `mv` 命令：

```bash
terraform state mv alicloud_vpc.this module.vpc.alicloud_vpc.this
# Move "alicloud_vpc.this" to "module.vpc.alicloud_vpc.this"
# Successfully moved 1 object(s).
```

执行后，state 文件会认为该资源一直位于新的路径下，后续的 `plan` 不会显示任何变更。这是实现零停机重构 Terraform 仓库的标准方法。

在较新的 Terraform 版本（`>= 1.1`）中，新增了 `moved` 块，提供了一种声明式的替代方案：

```hcl
moved {
  from = alicloud_vpc.this
  to   = module.vpc.alicloud_vpc.this
}
```

相比 `state mv`，`moved` 更加推荐，因为它会被提交到 git 中，可以在 PR 中进行评审，并且无需团队成员手动执行命令即可生效。

### `terraform taint`（旧版）和 `-replace`（新版）——强制重建单个资源

有时某台 ECS 实例可能会进入异常状态，例如磁盘空间耗尽或内核崩溃。此时你希望 Terraform 仅销毁并重建这台实例，而不影响其他资源。可以使用以下命令：

```bash
terraform apply -replace=alicloud_instance.agent[1]
# Plan: 1 to add, 0 to change, 1 to destroy.
```

旧版 Terraform 提供了 `terraform taint` 命令来实现类似功能，但该命令已被废弃。目前推荐使用 `-replace` 参数。我个人大约每季度会用到一次，通常是在某台 ECS 实例进入不可恢复状态且重启无效的情况下。

> **实战建议：** 在进行任何 state 手术之前，务必先备份当前的 state 文件：
```bash
terraform state pull > backup.tfstate
```
这样即使出现失误，也可以快速恢复：
```bash
terraform state push backup.tfstate
```
state 手术是少数几个因小错误可能导致数小时损失的操作之一，备份可以将恢复时间缩短至 10 秒以内。
## tfstate 文件里到底存了什么？为什么要加密存储？

很多人以为 tfstate 文件只保存资源的 ID，其实不然。实际上，state 文件记录了 Terraform 所管理的每个资源的所有属性——包括那些动态生成的内容，比如 RDS 的连接字符串、自动生成的密码、KMS 密钥材料，以及标记为 `sensitive` 的变量。

你可以试着在任何一个稍微复杂一点的项目中运行以下命令，看看输出结果：

```bash
terraform show -json | jq '.values.root_module.resources[] | select(.values | tostring | test("password|secret|key"; "i"))'
```

你会发现里面可能包含明文密码、API 密钥，甚至完整的 JSON 格式凭据。这是 Terraform 的设计决定——它需要这些信息来计算资源变更（diff）并执行应用（apply）。

基于此，我们可以得出三个重要结论：

1. **存储 tfstate 文件的 OSS bucket 必须启用加密。**  
   在初始化环境时，我们通过 `server_side_encryption_rule { sse_algorithm = "KMS" }` 配置了 KMS 加密。如果你跳过了这一步，任何有 `oss:GetObject` 权限的人都能直接读取你的敏感信息。
   
2. **千万不要把 `terraform.tfstate` 或 `*.tfstate.backup` 提交到 git 仓库。**  
   立刻将这些文件加入 `.gitignore`。我见过不止一次因为疏忽导致生产环境的密钥泄露到代码仓库中的事故。

3. **对敏感变量设置 `sensitive = true`。**  
   这样可以防止 Terraform 在执行 `plan` 或 `apply` 时将这些变量的值打印到日志中，从而避免它们被意外记录到 CI/CD 的日志系统中。例如：

```hcl
variable "rds_admin_password" {
  type      = string
  sensitive = true
}
```

虽然这些值仍然会以加密形式存储在 tfstate 文件中，但至少你的 GitHub Actions 日志不会把密码暴露给全世界。

对于安全要求特别高的场景，Terraform 提供了 `-target` 参数，可以针对单个资源提取状态，你还可以通过工具如 `sops` 或 `age` 对状态文件进行额外的应用层加密。不过，只要 OSS bucket 本身启用了 KMS 加密，并且访问权限严格控制，我认为额外的加密措施通常是不必要的。
## Workspace 和独立 State 文件：如何选择？

前面提到的 `terraform workspace new dev/staging/prod` 是一个简单直接的解决方案，适合大多数团队使用。但其实这背后隐藏着一个重要的架构决策，需要你深思熟虑，而不是盲目采用默认选项。

**Workspace（单项目，多状态）：**
- 使用一个统一的 backend，通过 workspace 名称区分前缀（例如 `agents/env:dev/terraform.tfstate`）。
- 一套 HCL 文件，通过 `terraform.workspace` 参数化来适配不同环境。
- 优点：开发环境和生产环境的差异清晰可见——代码相同，变量不同。
- 缺点：如果在 `count` 表达式中犯了一个拼写错误，可能会影响到所有 workspace；此外，一个 PR 很难做到“只修改生产环境”。

**独立 State 文件（每个环境一个项目）：**
- 按环境划分目录，例如 `envs/dev/`、`envs/staging/`、`envs/prod/`。
- 每个环境都有自己独立的 `backend.tf` 和 `main.tf`，调用共享模块。
- 优点：生产环境有独立的 PR 审核流程、独立的 `apply` 操作以及独立的权限控制。
- 缺点：容易出现代码重复；升级模块版本时需要修改多个地方；难以强制保证“所有环境运行相同的代码”。

我的经验法则是：**5 人以下的小团队用 Workspace，5 人以上的团队建议分 State 文件。** 小团队追求的是简单高效；而大团队则需要关注“爆炸半径”——你希望生产环境的改动只影响 `envs/prod/` 目录，并且由值班人员审核，而不是因为一次 `dev.tfvars` 的修改意外波及生产环境。

在这个系列中，我选择了 Workspace，主要面向单人工程师或小型团队的场景。不过需要注意的是，未来如果需要切换到独立 State 文件，可以通过 `terraform state pull` 导出当前 Workspace 的状态，再用 `terraform state push` 导入新项目——只要操作得当，迁移并不复杂。

> **实战小贴士：** Workspace 名称本质上只是一个**字符串**。在 CI 中通过 `TF_WORKSPACE=prod terraform plan` 显式指定，可以有效避免“忘记切换 Workspace”这类低级错误。同时，结合分支保护策略，确保生产环境的变更只能从 `main` 分支发起并应用。
