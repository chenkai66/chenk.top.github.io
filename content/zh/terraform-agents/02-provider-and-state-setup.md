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
读到这里，关闭页面，打开终端吧。等会儿回来时，你应该已经准备好以下内容：

1. 安装好且版本锁定的 `alicloud` Terraform provider。
2. 配置好的认证流程——用的是正确的方法，不是图省事的方法。
3. 基于 OSS Bucket 和 Tablestore 锁定的远程状态存储。
4. 三个工作空间（`dev`, `staging`, `prod`），共用后端但状态隔离。
5. 能跑通的 `terraform plan`，哪怕配置还是空的。

至此，Agent 尚未部署——本阶段仅涉及基础设施底座搭建，后续所有文章均以此为前提；如果跳过此步骤，推迟到第三篇文章再补，一周内遇到 tfstate 损坏的概率极高。

## Step 0: 安装 Terraform

我不多废话安装过程，官方 `Install Terraform` 文档覆盖了所有系统。 macOS 用户直接：

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version
# Terraform v1.9.x
# on darwin_arm64
```

锁定一个近期的稳定版。 Aliyun 文档虽然测试过 `>= 0.12`，但新项目直接上 `>= 1.9`。新版本在体验上有实打实的改进——`for_each`、`optional()`、更精细的 `moved` 块、声明式的 `import` 块——本系列后续文章都会用到这些特性。

## Step 1: 锁定 Provider

建个项目目录，写好 `versions.tf`：

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

`~> 1.230` 这个约束允许 `1.230.0` 到 `1.230.x`，但会拦住 `1.231.0`。这是默认的最佳实践。一旦你把 `.terraform.lock.hcl` 提交到 git （`terraform init` 时 Terraform 会自动生成），你就锁死了 *确切* 的 provider 版本和校验和。队友 later 跑 `terraform init` 时，拿到的 provider 跟你比特级一致。

尽早锁定 provider 版本是一项低成本但高回报的风险控制措施。例如，alicloud provider 在 1.220 版本附近对 OSS Bucket 的 schema 进行了重构，导致我花了三个下午排查问题。你最终仍需升级，但必须主动推进：先提交 PR，仔细审查 plan 输出的变更差异，确认无误后再执行 apply，切勿在未经评审的情况下，在深夜 11 点在他人机器上意外触发升级。

## Step 2: 认证——三种方案，按靠谱程度排个序

Provider 需要阿里云 credentials，真正可选的有三种，按专业程度递增：

![认证流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/wanxiang_auth_flow.png)


![三种认证阿里云提供商的方法](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig2_auth_methods.png)

### 方案 A: 静态 AK/SK （仅限个人笔记本）

```bash
export ALICLOUD_ACCESS_KEY="LTAI5tXXXXXXXXXXXXXX"
export ALICLOUD_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export ALICLOUD_REGION="cn-shanghai"
```

Provider 会自动发现这些环境变量。千万别——在任何情况下都别——把密钥写进 `.tf` 文件。状态文件不存这玩意儿，但 `provider {}` 块会，而那块代码是进 git 的。

如果这个 AK/SK 属于一个 RAM 子账号，且权限仅限于 Terraform 管理的资源，单人项目可以接受。如果是协作项目，请直接查看方案 B。

### 方案 B: AssumeRole （CI  runner）

CI runner 不该持长期有效的 AK。给 runner 一个只有一种权限的 AK——针对目标角色的 `sts:AssumeRole`——让 Terraform 在 apply 时 assume 这个角色：

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

角色才有实际的写权限；AK 只有 assume 它的权限。STS 会话是短期的（默认一小时），在 ActionTrail 里有审计日志，一旦剥离信任策略就能立即撤销。这也是 GitLab CI、GitHub Actions 和 Jenkins 等 CI/CD 环境中推荐的模型。

### 方案 C: ECS RAM 角色（堡垒机 / IaC 服务 runner）

如果 `terraform apply` 跑在 Aliyun ECS 实例上——团队的 ops 堡垒机或者 Aliyun 托管的 IaC Service runner——给实例绑定 RAM 角色， Provider 会自动从实例元数据抓取凭证：

```hcl
provider "alicloud" {
  region = var.region
  # No assume_role block, no env vars — provider auto-detects from
  # http://100.100.100.200/latest/meta-data/ram/security-credentials/
}
```

配置、环境变量和文件中都没有密钥。凭证轮换由系统自动完成。这是业界公认的黄金标准，建议每个团队在落地 Terraform 的首月内采用该方案。

> **实战建议：** 不管选哪个，显式设置 `ALICLOUD_REGION`（或者 `provider { region = ... }`）。如果不设， Provider 不会选默认值——你会在 `terraform plan` 时得到一个让人困惑的 "Region must be specified" 错误，该错误我在实践中多次遇到。

## Step 3: 状态文件——为什么本地 tfstate 是隐患

跑 `terraform apply` 时，默认 Terraform 会把 `terraform.tfstate` 写在当前目录。该文件是基础设施当前状态的唯一权威来源。以下三种情况极易发生：

![分布式状态锁定](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/wanxiang_state_lock.png)


1. **丢失。** 删了目录， Terraform 就觉得什么都不存在了。下次 `apply` 试图重建一切（或者因为重复资源失败）。
2. **冲突。** 两个工程师同时跑 `apply` 会搞坏状态文件。
3. **明文 secrets。** 有些资源属性（RDS 密码、 KMS 材料、生成的 token）会落进 tfstate。把它扔在笔记本上已经很糟了——提交到 git 更糟，但真有人这么干。

### tfstate 里到底存了什么

有个常见误区： tfstate 只存资源 ID。错。状态文件包含 Terraform 知道的每个资源的每个属性——包括计算出来的值，比如 RDS 连接串、生成的密码、 KMS 密钥材料，还有 `sensitive` 变量。

任选一个结构较复杂的 tfstate 文件，运行以下命令查看其内容：

```bash
terraform show -json | jq '.values.root_module.resources[] | select(.values | tostring | test("password|secret|key"; "i"))'
```

你会看到密码。你会看到 API key。你可能看到整块整块的 credential JSON blob。这是 Terraform 的设计决定：它需要这些值来计算差异（diff）和执行变更（apply）。

这正是本地 tfstate 过不了第一天就必须换掉的原因。解法是 **远程状态** 加 **状态锁定**。在 Aliyun 上，标准做法是 OSS + Tablestore：

![使用 Tablestore 锁定的 OSS 远程状态](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig1_state_backend.png)

OSS 存实际的 `terraform.tfstate` 文件（开启版本控制——万一损坏了，一条 CLI 命令就能恢复）。 Tablestore 存一行小小的 "lock" 记录， Terraform 在 apply 前写入，结束后删除。如果第一个 apply 还持着锁时第二个启动了，第二个会等待或者失败——绝不可能两个同时跑。

基于 tfstate 的内容，有两条不可妥协的原则：

- **存状态的 OSS bucket 必须开启 KMS 加密。** 我们在 step 4 会打开这个开关。要是跳过这步，任何有 bucket `oss:GetObject` 权限的人都能读你的 secrets。
- **把敏感变量标记为 `sensitive = true`**，防止 plan/apply 输出把它们泄露到 CI 日志里：

```hcl
variable "rds_admin_password" {
  type      = string
  sensitive = true
}
```

值依然会在 tfstate 里（加密的），但至少你的 GitHub Actions 日志不会把密码 paste 给全世界看。

## Step 4: 引导后端（先有鸡还是先有蛋）

 用于承载后端的 OSS Bucket 和 Tablestore 实例，必须在 Terraform 后端配置启用前预先创建完成。诚实的工作流是：用一个 *本地* 状态文件，在小小的一次性 `bootstrap/` 目录里 provision 它们，然后再也不碰它。

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

在 `bootstrap/` 里面跑 `terraform init && terraform apply`。大概 30 秒。然后把本地 tfstate 归档到某个地方（我把它存在 1Password 里做个 sanity backup），再也别从这个目录跑命令。在你忘记之前，先把 `bootstrap/terraform.tfstate*` 加进 `.gitignore`。

## Step 5: 配置后端

回到你的真实项目，加上：

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

`prefix` 让你能在一个 bucket 里塞多个状态文件——以后把 infra 拆成多个 Terraform 项目时会很方便。`encrypt = true` 开启 OSS 侧加密（我们已经在 bucket 级别开了 KMS 规则，但纵深防御总没坏处）。

跑一下：

```bash
terraform init
# Initializing the backend...
# Successfully configured the backend "oss"!
# Initializing provider plugins...
# - Installing aliyun/alicloud v1.230.x...
```

如果报 `AccessDenied`，说明你的 auth role 在 bucket 上没有 `oss:GetObject`/`PutObject` 权限。最小角色策略是：

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

把这个策略绑到你用来认证的角色上。别给 `oss:*`——哪怕对后端角色，最小权限原则也至关重要，因为这角色就在你的 CI runner 里，一旦那里泄露了 AK，你所有的状态文件就都被人读了。
## Step 6: 用 workspace 隔离环境

workspace 其实就是同一个 backend 里的独立状态文件。默认那个很有用，就叫 `default`。你需要其他环境就自己建：

![开发、测试和生产环境的多环境工作区隔离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/wanxiang_workspace.png)


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

在 HCL 里，`terraform.workspace` 会解析成当前的 workspace 名字，这样你就能根据环境参数化资源大小：

```hcl
locals {
  is_prod = terraform.workspace == "prod"

  ecs_count         = local.is_prod ? 3 : 1
  ecs_instance_type = local.is_prod ? "ecs.c7.xlarge" : "ecs.c7.large"
  rds_class         = local.is_prod ? "pg.x4.large.2c" : "pg.n2.medium.1c"
}
```

还有个干净的做法是每个环境配一个 `*.tfvars` 文件：

```bash
terraform plan -var-file=env/dev.tfvars
terraform plan -var-file=env/prod.tfvars
```

我一般用 `tfvars` 文件处理那些“显而易见的差异”（比如 CIDR 块、 region、实例数量），而 `terraform.workspace` 只用来做 `is_prod` 这种条件开关。混着用没问题——每个项目选一个作为主要机制就行。

### Workspace 还是独立状态文件：真正的抉择

文章到现在展示的都是 `terraform workspace new dev/staging/prod`。这是简单答案，对大多数团队都管用。但这背后藏着一个真正的架构选择，你应该 deliberate 地做决定，而不是无脑默认。

**Workspaces （单项目， N 个状态）：** 一个 backend，前缀按 workspace 名字拆分（`agents/env:dev/terraform.tfstate`）；一套 HCL 文件，靠 `terraform.workspace` 参数化。优点： dev 和 prod 的差异肉眼可见——代码一样，变量不同。缺点：`count` 表达式写错一个字母可能影响所有 workspace；一次 PR 没法隔离“只改 prod"。

**独立状态文件（每环境单项目）：**  separate 目录 `envs/dev/`、`envs/staging/`、`envs/prod/`；每个环境有自己的 `backend.tf`、`main.tf` 调用共享模块。优点： prod 有独立的 PR  review、独立的 apply、独立的 RAM 权限。缺点：代码重复；升级模块版本需要改 N 处；更难强制“所有环境跑同一份代码”。

我的经验法则：**5 个工程师以下用 workspace， 5 个以上拆状态文件。** 5 人以下，简单至上。 5 人以上，爆炸半径 matters——你想要一个只动 `envs/prod/` 且由 on-call 轮值批准的 prod PR，而不是一个 `dev.tfvars` 变更意外 cascades 到生产。

这个系列用 workspace 是因为重点在单兵或小团队。万一以后规模大了，升级路径也是通的：从 workspace `terraform state pull`，`terraform state push` 到新项目，退役 workspace。也就疼这一次。

> **实战建议：** workspace 名字 *就是个字符串*。在 CI 里用 `TF_WORKSPACE=prod terraform plan` 设置，避免“忘了切换 workspace"这类事故。配合分支保护，确保 prod workspace 只能从 `main` 分支 apply。

## Step 7: 五命令循环

日常玩 Terraform 其实就这五个命令：

![你将运行数百次的五命令循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png)

```bash
terraform fmt        # 标准化缩进；预提交 hook
terraform validate   # 静态 schema 检查；1 秒内跑完
terraform plan       # 对比期望 vs 现实；这步得仔细看
terraform apply      # 发送 API 调用
terraform show       # 检查当前状态
```

三条规矩：

1. **Apply 之前必须读 plan 输出。** 它会告诉你到底要发生什么——哪些资源要创建（`+`）、原地更新（`~`）、强制替换（`-/+`）或者销毁（`-`）。特别是原地替换箭头，背后藏着停机时间。
2. **在 CI 里把 `plan` 和 `apply` 分成两步。** 跑 `terraform plan -out=tfplan`，把 plan 输出贴到 PR 里，拿到人工批准，合并后再 `terraform apply tfplan`。千万别 push 了就自动 apply。
3. **别跳过 `state`。** `terraform state list` 显示你现在管理的所有东西；`terraform state show <addr>` 显示单个资源的全部属性。调试奇怪的 drift 时，从这儿入手。

## Step 8: 状态手术，解决那 5% `apply` 搞不定的日子

五命令循环覆盖 95% 的日子。剩下 5% 是状态文件本身出了问题，这时候正确的 fix 是直接手术——而不是 panicked 地 `terraform destroy && apply`。下面是我常用的四个操作，按紧张程度递增。

### `terraform import` —— 纳管手动创建的资源

你六个月前在控制台创建了一个 VPC。现在想让 Terraform 管理它。错误的做法是写好 HCL 然后 `apply` —— Terraform 会试图 *创建第二个 VPC*。正确的做法是 `import`：

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

然后迭代 HCL 直到 `terraform plan` 显示无变更。这是把控制台建的基建纳入管理的唯一安全方式。 Terraform `>= 1.5` 支持声明式的 `import` block，这是我现在的首选：

```hcl
import {
  to = alicloud_vpc.legacy
  id = "vpc-uf6abc123def456"
}
```

### `terraform state rm` —— 让 Terraform 忘记某个资源

import 的反面。你决定在另一个 stack 里管理某个资源，或者把它交还给 ops 团队。你不想 *销毁* 它，只想让 Terraform 停止追踪：

```bash
terraform state rm alicloud_oss_bucket.legacy_archive
# Removed alicloud_oss_bucket.legacy_archive
```

Bucket 还在 OSS 里。 Terraform 的状态不再引用它。后续的 `plan` 不会试图销毁它（因为不在状态里），也不会试图创建它（因为 HCL 也没了——同一个 PR 里把 HCL 块删掉）。

我用这个处理鸡生蛋蛋生鸡的 `bootstrap/` 目录。一旦 OSS+Tablestore 管理折叠进主 stack，我就从 bootstrap 状态里 `terraform state rm` 它们，然后彻底删除 bootstrap 目录。

### `terraform state mv` —— 重命名或重构

你想把资源从一个 module 路径移到另一个——比如 `alicloud_vpc.this` → `module.vpc.alicloud_vpc.this`。不用 `mv` 的话， Terraform 会看成“销毁旧的，创建新的”，真会把你的生产 VPC 删了重建。用了 `mv`：

```bash
terraform state mv alicloud_vpc.this module.vpc.alicloud_vpc.this
# Move "alicloud_vpc.this" to "module.vpc.alicloud_vpc.this"
# Successfully moved 1 object(s).
```

现在状态认为资源一直都在新地址。下次 `plan` 显示无变更。这就是所有 Terraform  repo 重构而不停机的做法。

现代 Terraform （`>= 1.1`）给了你 `moved` block，这是声明式版本：

```hcl
moved {
  from = alicloud_vpc.this
  to   = module.vpc.alicloud_vpc.this
}
```

永远优先用 `moved` 而不是 `state mv` —— 它能进 git，能在 PR 里 review，团队里每个人都能用，不用谁再去跑手动命令。

### `terraform apply -replace` —— 强制重建单个资源

某个 ECS 实例状态诡异——磁盘满、 kernel panic，谁知道呢。你想让 Terraform 只销毁重建这一个实例，不动别的：

```bash
terraform apply -replace=alicloud_instance.agent[1]
# Plan: 1 to add, 0 to change, 1 to destroy.
```

老的 `terraform taint` 做同样的事但已经 deprecated 了。`-replace` 是现在的写法。我大概每季度用一次，当某个 ECS 实例进入不可恢复状态且 stop/start 没用时。

> **实战建议：** 任何状态手术前，先 `terraform state pull > backup.tfstate` 留个已知好的副本。状态手术是少数几个敲错字就能让你损失几小时的地方；有了备份，恢复只要 10 秒（`terraform state push backup.tfstate`）。

## 第一天就会遇到的八种报错

按我遇到的顺序排列：

1. **`Error: Failed to query available provider packages` on `terraform init`.** GFW 问题。设 `HTTPS_PROXY` 或者用阿里镜像：`https://mirrors.aliyun.com/terraform/`。
2. **`Error: state lock`.** 上次 apply 时按了 Ctrl-C，锁僵死了。跑 `terraform force-unlock <LOCK_ID>`（ID 在报错里）。先确认真没人在跑。
3. **`Error: Region must be specified`.** 设 `ALICLOUD_REGION` 环境变量或者在 `provider` 块里设 `region`。
4. **`AccessDenied` on backend init.** OSS bucket 前缀的 RAM 权限。复查 Step 5 的 policy。
5. **`InvalidParameter.NotFound` on Tablestore.** bootstrap 错了 region。 Tablestore endpoint 和 OSS bucket region 必须一致。
6. **`Provider produced inconsistent result after apply`.** 几乎是 provider 版本升级后 `.terraform/` 缓存僵了。`rm -rf .terraform .terraform.lock.hcl && terraform init`。
7. **`Resource already exists`.** 你在控制台手建了资源。要么删了它，要么 import 它（见 Step 8）。
8. **刚 apply 完的资源 `terraform plan` 又有意外 diff。** Drift。要么有人在控制台动了资源，要么 provider 的 read 逻辑跟 create 不一样。看 diff 里的具体属性；通常 fix 是显式设置该属性，让 Terraform 别再“注意到”差异。

> **实战建议：** 每次 `apply` 后立即跑 `terraform plan`，哪怕没变更。 plan 应该是空的。如果不是，你就有 drift 了， drift 留得越久越难 reconcile。

## 接下来

如果这篇文章从头到尾跑通了，你现在应该能跑 `terraform init`，`terraform workspace select dev`，`terraform plan` 然后看到 "No changes."。这是地基。其他东西都堆在这上面。

文章 3 会构建第一个真正的基建组件：一个可复用的 `vpc-baseline` 模块。 VPC、三个可用区的三个 vSwitch、 NAT 网关、 EIP、安全组基线、 KMS 密钥。后续每篇文章都会用它——这是我 agent 栈里被复制粘贴最多的模块。