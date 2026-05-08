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

这一篇你不再是读，是开始动手敲代码。读完之后你会有：

1. `alicloud` Terraform provider 装好且版本钉死
2. 认证接好——用对的方式，不是方便的方式
3. 远程状态文件放在 OSS，用 Tablestore 加锁
4. 三个工作空间（`dev`、`staging`、`prod`），共用 backend、隔离状态
5. 一个能跑通的 `terraform plan`（即使配置是空的）

本篇还不会建出任何 Agent 资源。我们打的是后续每一篇都会假设的地基。如果你跳过这一步，直接奔第三篇硬上，一周之内必出一次状态文件被踩坏的事故。

## 第 0 步：安装 Terraform

不赘述——官方《Install Terraform》文档已经覆盖了所有操作系统。macOS 上：

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version
# Terraform v1.9.x
# on darwin_arm64
```

钉到一个近期的稳定版。阿里云的文档基于 `>= 0.12` 测试，但新项目直接上 `>= 1.9`。新版本里 `for_each`、`optional()`、`moved` 块、声明式 `import` 块这些改进都很实用，本系列后面会全用到。

## 第 1 步：钉死 Provider 版本

建一个项目目录，放一个 `versions.tf`：

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

`~> 1.230` 表示允许 `1.230.0` 到 `1.230.x` 的所有补丁版本，但拒绝 `1.231.0`。这是一个非常合理的默认值。你把 `.terraform.lock.hcl` 提交到 git 之后（这个文件在第一次 `terraform init` 时自动生成），不止 provider 版本被钉死，连校验和也一起钉死。同事再跑 `terraform init`，拿到的 provider 是字节级一致的。

尽早钉版本是最便宜的保险。alicloud provider 在小版本之间出过破坏性变更（1.220 附近 OSS Bucket schema 那次重构吃了我三个下午）。升级当然要做，但应该是有意识地做：发 PR、看 plan diff、合代码——而不是在某个同事晚上 11 点的笔记本上意外触发。

## 第 2 步：认证——三种方式，按推荐度排序

provider 需要阿里云凭证。三种真正可选的方式，按专业可接受度从低到高：

![三种 alicloud provider 认证方式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig2_auth_methods.png)

### 方案 A：静态 AK/SK（仅个人笔记本）

```bash
export ALICLOUD_ACCESS_KEY="LTAI5tXXXXXXXXXXXXXX"
export ALICLOUD_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export ALICLOUD_REGION="cn-shanghai"
```

provider 会自动读取这些环境变量。任何情况下都不要把密钥写进 `.tf` 文件——状态文件不会存它，但 `provider {}` 块会，而这个块是要进 git 的。

如果 AK/SK 是给一个 RAM 子账号用的，且权限只覆盖 Terraform 管理范围内的资源，那么个人项目这样用可以接受。一旦项目共享，直接跳到方案 B。

### 方案 B：AssumeRole（CI runner）

CI runner 不该持有长期 AK。给 runner 一个 AK，权限只有一个——对目标角色的 `sts:AssumeRole`，apply 的时候让 Terraform 去扮演那个角色：

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

真正的写权限挂在角色上，AK 只有扮演权。STS 会话寿命短（默认 1 小时）、ActionTrail 全程审计、解开信任策略就立刻吊销。GitLab CI、GitHub Actions、Jenkins runner 都该用这个模式。

### 方案 C：ECS RAM 角色（堡垒机 / IaC 服务 runner）

如果 `terraform apply` 跑在阿里云 ECS 上——你团队的运维堡垒机，或者阿里云托管的 IaC 服务 runner——给实例绑一个 RAM 角色，provider 会自动从实例元数据拿凭证：

```hcl
provider "alicloud" {
  region = var.region
  # 不需要 assume_role 块，也不需要环境变量，provider 会从下面这个地址读取：
  # http://100.100.100.200/latest/meta-data/ram/security-credentials/
}
```

任何配置、任何环境变量、任何文件里都没有密钥。轮转自动完成。这是黄金标准，我每带一个新团队，第一个月必把他们推到这条路上。

> **实战提醒：** 不论选哪种方式，都要显式设 `ALICLOUD_REGION`（或 `provider { region = ... }`）。不设的话，provider 不会挑默认值，`terraform plan` 会甩出一个让人迷惑的 “Region must be specified”。这个坑我踩过不止一次。

## 第 3 步：状态文件——为什么本地 tfstate 是个雷

跑 `terraform apply` 时，Terraform 默认在当前目录写一个 `terraform.tfstate`。这个文件是“到底有哪些基础设施”的唯一真相源。本地存放有三个必出的问题：

1. **丢失。** 删掉目录，Terraform 就以为什么都不存在了，下次 `apply` 要么把所有资源重建一遍，要么因重名失败。
2. **冲突。** 两位工程师同时 `apply`，状态文件分分钟被踩坏。
3. **明文密钥。** 部分资源属性（RDS 密码、KMS 材料、生成的 token）会落到 tfstate。留在笔记本上已经够糟，提交到 git 更糟——而且真的有人这么干。

### tfstate 里到底存了什么

一个常见误解：tfstate 只存资源 ID。错。状态文件存的是 Terraform 知道的每一个资源的每一个属性——包括计算属性，比如 RDS 的连接串、自动生成的密码、KMS 密钥材料、被标 `sensitive` 的变量。

在任何一个稍有规模的项目里跑这条命令看看：

```bash
terraform show -json | jq '.values.root_module.resources[] | select(.values | tostring | test("password|secret|key"; "i"))'
```

你会看到密码、API key，可能还有整段 JSON 凭据。这是设计使然——Terraform 算 diff 和 apply 都要这些值。

正因为本地 tfstate 撑不过第一天，正确做法是远程状态加上状态锁。阿里云上的标准方案是 OSS + Tablestore：

![OSS 上的远程状态 + Tablestore 加锁](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig1_state_backend.png)

OSS 存真正的 `terraform.tfstate`（开版本控制，万一被踩坏一条 CLI 就能恢复）。Tablestore 存一行小小的“锁”记录，Terraform 在 apply 之前写进去，结束后删掉。第二个 `apply` 想在第一个还持锁的时候启动，要么排队，要么直接失败——绝不会两个一起跑。

由 tfstate 的内容直接推出两条不可妥协的要求：

- 存放状态文件的 OSS bucket 必须开 KMS 加密。第 4 步里我们会打开。跳过的话，任何拿到 `oss:GetObject` 权限的人都能直接读到你的密钥。
- 敏感变量必须标 `sensitive = true`，避免 plan/apply 输出泄漏到 CI 日志：

```hcl
variable "rds_admin_password" {
  type      = string
  sensitive = true
}
```

值还是会在 tfstate 里（加密存的），但至少你的 GitHub Actions 日志不会把密码贴给全世界看。

## 第 4 步：bootstrap backend（先有鸡还是先有蛋）

承载 backend 的 OSS bucket 和 Tablestore……得先于 backend 存在。诚实的做法是用一个单独的 `bootstrap/` 目录，用本地状态文件把它们建出来，然后再也不动它。

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

进 `bootstrap/`，`terraform init && terraform apply`。约 30 秒。然后把本地 tfstate 归档（我塞 1Password 里做最后一道兜底），从此不再从这个目录跑 Terraform。在你忘记之前，把 `bootstrap/terraform.tfstate*` 加进 `.gitignore`。

## 第 5 步：配置 backend

回到正式项目里，加上：

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

`prefix` 让你在一个 bucket 里塞多个状态文件——后面把基础设施拆成多个 Terraform 项目时很有用。`encrypt = true` 启用 OSS 端的加密（虽然 bucket 级 KMS 已经开了，但纵深防御没坏处）。

跑：

```bash
terraform init
# Initializing the backend...
# Successfully configured the backend "oss"!
# Initializing provider plugins...
# - Installing aliyun/alicloud v1.230.x...
```

如果报 `AccessDenied`，说明你认证用的角色对 bucket 没有 `oss:GetObject`/`PutObject` 权限。最小策略如下：

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

把这条策略挂到你认证用的角色上。别图省事给 `oss:*`——最小权限对 backend 角色尤其重要，因为这个角色就在 CI runner 里，CI 那边一旦泄一把 AK，你所有状态文件全裸了。

## 第 6 步：用工作空间隔离环境

工作空间就是同一个 backend 下的多个独立状态文件。默认工作空间名字很贴心，叫 `default`。按需创建其他几个：

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

HCL 里 `terraform.workspace` 解析为当前工作空间名，可以用来参数化资源规格：

```hcl
locals {
  is_prod = terraform.workspace == "prod"

  ecs_count         = local.is_prod ? 3 : 1
  ecs_instance_type = local.is_prod ? "ecs.c7.xlarge" : "ecs.c7.large"
  rds_class         = local.is_prod ? "pg.x4.large.2c" : "pg.n2.medium.1c"
}
```

另一种干净的玩法是每个环境一份 `*.tfvars`：

```bash
terraform plan -var-file=env/dev.tfvars
terraform plan -var-file=env/prod.tfvars
```

我个人的分工是：`tfvars` 管那些“一眼就知道环境之间会变”的配置（CIDR、region、实例数），`terraform.workspace` 只用来跑 `is_prod` 这种条件开关。两种混用没问题——只要每个项目里挑一种当主轴就行。

### 工作空间还是独立状态文件：真正要做的判断

上面用的是 `terraform workspace new dev/staging/prod`，简单方案，对大多数团队够用。但这背后藏着一个真正的架构选择，应该有意识地做，而不是默认按下不表。

**工作空间（一个项目，N 个状态）：** 一个 backend，按工作空间名分前缀（`agents/env:dev/terraform.tfstate`）；一套 HCL，靠 `terraform.workspace` 参数化。优点：dev 和 prod 的差异一眼可见——同样的代码、不同的变量。缺点：`count` 表达式里一个错字能影响所有工作空间；一个 PR 没法做到“只改 prod”。

**独立状态文件（每个环境一个项目）：** 按目录分 `envs/dev/`、`envs/staging/`、`envs/prod/`；每个环境有自己的 `backend.tf`、`main.tf`，调用共享模块。优点：prod 有自己的 PR 评审、自己的 apply、自己的 RAM 权限。缺点：代码重复；升模块版本要改 N 处；难以强制“所有环境跑同一份代码”。

我的经验法则：5 个工程师以下用工作空间，5 个工程师以上分独立状态。人少的时候简洁性占优。人多以后爆炸半径很重要——你想要的是“只动 `envs/prod/`、由值班审核”的 prod PR，而不是一个 `dev.tfvars` 改动顺带波及生产。

本系列用工作空间，因为我们瞄准的是单人或小团队场景。万一以后规模上来，迁移路径并不痛：从工作空间 `terraform state pull`，往新项目 `terraform state push`，把工作空间退役。痛苦只有一次。

> **实战提醒：** 工作空间名本质上就是一个字符串。CI 里用 `TF_WORKSPACE=prod terraform plan` 显式指定，可以彻底消灭“忘了切工作空间”这一类事故。再配合分支保护，prod 工作空间只允许从 `main` 分支 apply。

## 第 7 步：五条命令的循环

日常 Terraform 就是五条命令：

![你会反复执行数百次的五条命令循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/02-provider-and-state-setup/fig3_init_apply_loop.png)

```bash
terraform fmt        # 格式化缩进，挂 pre-commit hook
terraform validate   # 静态校验配置结构，1 秒内
terraform plan       # diff 期望与实际，认真看
terraform apply      # 真正发 API 调用
terraform show       # 看当前状态
```

三条铁律：

1. apply 前必须读 plan 输出。它会明确告诉你即将发生什么——哪些资源新建（`+`）、原地更新（`~`）、强制替换（`-/+`）、销毁（`-`）。尤其是“原地更新但实际重建”的箭头，背后藏的就是停机。
2. CI 里 plan 和 apply 必须分两步。先 `terraform plan -out=tfplan`，把 plan 输出贴到 PR 让人审，合并后再跑 `terraform apply tfplan`。永远不要 push 触发自动 apply。
3. 别跳过 `state` 命令。`terraform state list` 列出当前管理的所有资源；`terraform state show <addr>` 看某个资源的完整属性。排查诡异漂移问题，第一步就是这个。

## 第 8 步：状态手术——`apply` 救不了的那 5%

五条命令的循环覆盖 95% 的日子。剩下 5% 是状态文件本身出了问题，正确解法是直接对状态做手术，而不是慌不择路地 `terraform destroy && apply`。下面四种操作我经常用，按胆战心惊程度从低到高排：

### `terraform import`——把控制台手建的资源纳入管理

半年前你在控制台手建了一个 VPC，现在想交给 Terraform 管。错误做法是写好 HCL 然后 `apply`——Terraform 会再建一个新 VPC。正确做法是 `import`：

```bash
# 先写 HCL 占位，确保资源地址写对
cat > vpc.tf <<EOF
resource "alicloud_vpc" "legacy" {
  vpc_name   = "legacy-prod"
  cidr_block = "172.16.0.0/16"
}
EOF

# 把现有资源导入到 state 中的对应地址
terraform import alicloud_vpc.legacy vpc-uf6abc123def456

# 看 diff，HCL 大概率跟实际不完全一致
terraform plan
# Plan: 0 to add, 1 to change, 0 to destroy.
#   ~ tags = { ... }   # 控制台设的 tag，Terraform 还不知道
```

然后反复调 HCL，直到 `terraform plan` 显示无变更。这是把控制台手建的基础设施安全纳管的唯一办法。Terraform `>= 1.5` 支持声明式 `import` 块，这是我现在的首选：

```hcl
import {
  to = alicloud_vpc.legacy
  id = "vpc-uf6abc123def456"
}
```

### `terraform state rm`——让 Terraform 忘掉某个资源

import 的反向操作。你决定把某个资源转到另一个栈管理，或者交还给运维。你不想销毁它，只想让 Terraform 不再追踪：

```bash
terraform state rm alicloud_oss_bucket.legacy_archive
# Removed alicloud_oss_bucket.legacy_archive
```

OSS 里的 bucket ��在，Terraform 的状态不再引用它。后续 `plan` 不会尝试销毁（因为已经不在 state 里），也不会尝试创建（因为同一个 PR 里把 HCL 也删了）。

我经常用这招处理前面那个 `bootstrap/` 目录。把 OSS+Tablestore 收回主栈管理之后，`terraform state rm` 把它们从 bootstrap 状态里移掉，整个 bootstrap 目录也一起删掉。

### `terraform state mv`——重命名或重构

要把资源从一个模块路径搬到另一个路径——比如 `alicloud_vpc.this` 搬到 `module.vpc.alicloud_vpc.this`。不用 `mv` 的话，Terraform 会理解成“销毁旧的、创建新的”，真的会把生产 VPC 删了重建。用 `mv`：

```bash
terraform state mv alicloud_vpc.this module.vpc.alicloud_vpc.this
# Move "alicloud_vpc.this" to "module.vpc.alicloud_vpc.this"
# Successfully moved 1 object(s).
```

状态文件会认为这个资源一直就在新地址。下一次 `plan` 显示无变更。所有“零停机重构 Terraform 仓库”靠的都是这一步。

Terraform `>= 1.1` 提供了 `moved` 块，是声明式版本：

```hcl
moved {
  from = alicloud_vpc.this
  to   = module.vpc.alicloud_vpc.this
}
```

`moved` 永远优于 `state mv`——它进 git，能在 PR 里评审，团队所有人都不需要手动跑命令就能生效。

### `terraform apply -replace`——强制重建单个资源

某台 ECS 进了奇怪状态——磁盘满、内核 panic、谁知道。你想让 Terraform 只销毁并重建这一台，不动其他资源：

```bash
terraform apply -replace=alicloud_instance.agent[1]
# Plan: 1 to add, 0 to change, 1 to destroy.
```

老的 `terraform taint` 命令做的是同一件事，但已废弃。`-replace` 是当前的拼法。我一年用三四次，基本都是 ECS 进了不可恢复状态、stop/start 也救不回来的时候。

> **实战提醒：** 任何状态手术之前，先 `terraform state pull > backup.tfstate` 留底。状态手术是少数几个一个错字能让你赔进去几小时的操作，留底能把恢复时间压到 10 秒（`terraform state push backup.tfstate`）。

## 第一天必踩的八个坑

按它们坑到我的顺序：

1. **`terraform init` 报 `Error: Failed to query available provider packages`。** GFW。设 `HTTPS_PROXY`，或用阿里云镜像 `https://mirrors.aliyun.com/terraform/`。
2. **`Error: state lock`。** 上次 apply 时按了 Ctrl-C，锁残留。跑 `terraform force-unlock <LOCK_ID>`（错误信息里有 ID）。先确认确实没在跑别的。
3. **`Error: Region must be specified`。** 设 `ALICLOUD_REGION` 环境变量，或者在 `provider` 块里写 `region`。
4. **backend init 报 `AccessDenied`。** OSS bucket 前缀的 RAM 权限。回去重看第 5 步的策略。
5. **Tablestore 报 `InvalidParameter.NotFound`。** bootstrap 时选错了 region。Tablestore endpoint 必须和 OSS bucket 同 region。
6. **`Provider produced inconsistent result after apply`。** 几乎一定是升级 provider 后 `.terraform/` 缓存没清。`rm -rf .terraform .terraform.lock.hcl && terraform init`。
7. **`Resource already exists`。** 你在控制台手建了一份。要么删掉，要么 import（见第 8 步）。
8. **刚 apply 完立刻 `terraform plan` 出现非预期 diff。** 漂移。要么有人在控制台动了，要么 provider 的 read 逻辑跟 create 不一致。看 diff 里具体是哪些属性，通常解法是把那个属性显式写进 HCL，让 Terraform 不再“看到”差异。

> **实战提醒：** 每次 `apply` 之后立刻再跑一次 `terraform plan`，哪怕没改东西。plan 应该是空的。如果不是，就有漂移。漂移留得越久，后面想对回去越难。

## 接下来

如果这一篇你跑通了，现在你应该可以依次执行 `terraform init`、`terraform workspace select dev`、`terraform plan`，看到 “No changes.”——这就是地基，后面所有内容都堆在它上面。

第三篇要建第一块真正的基础设施：可复用的 `vpc-baseline` 模块。VPC、跨三可用区的三个 vSwitch、NAT 网关、EIP、安全组基线、KMS 密钥。后续每一篇都会用它，是我所有 Agent 栈里被复制粘贴最多的模块。
