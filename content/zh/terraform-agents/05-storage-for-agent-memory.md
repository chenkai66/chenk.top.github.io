---
title: "用 Terraform 给 AI Agent 上云（五）：存储层——向量、关系、对象记忆"
date: 2026-03-20 09:00:00
tags:
  - Terraform
  - 阿里云
  - OSS
  - RDS
  - OpenSearch
  - AI Agent
categories: Terraform
lang: zh-CN
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 5
description: "Agent 有三种记忆，分别落到三个阿里云服务上：会话用 PolarDB/RDS，embedding 用 OpenSearch 向量版或 pgvector，产物用 OSS。每一层的真实 Terraform，再加上让账单不暴涨的 lifecycle 和备份规则。"
disableNunjucks: true
translationKey: "terraform-agents-5"
---

Agent 的"记忆"是多数教程一笔带过的部分。"embedding 丢 Pinecone，会话进 Postgres，截图传 S3。"在阿里云上，三种都有托管服务，而正确地用 Terraform 把它们建出来，就是"记忆好用"和"凌晨四点磁盘满了我们丢了三周对话历史"的差。

本篇覆盖三层、每层的 Terraform，再加上无聊但关键的 lifecycle 和备份规则。

![用 Terraform 给 AI Agent 上云（五）：存储层——向量、关系、对象记忆 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/illustration_1.jpg)

## 三层记忆模型

![Agent 的三种记忆对应到三个阿里云服务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig1_memory_three_layers.png)

理解这个模型可以从以下三个方面入手：

- **短期 / 会话记忆**——记录 Agent 在当前运行以及最近几次运行中的操作内容，比如对话的轮次、工具调用记录以及中间状态。这类数据的特点是 schema 稳定、需要低延迟访问，并且通常涉及事务性操作，因此适合存储在关系型数据库中。
- **长期 / 语义记忆**——主要用于存储文档、历史输出以及召回库的 embedding 数据。这种场景通常需要结合词汇检索和向量检索的能力，因此更适合存放在向量存储系统中。
- **产物 / 大文件存储**——包括生成的图片、PDF 文件、截图以及运行快照等内容。这些数据通常体积较大，写入后很少会被频繁读取，因此对象存储（如 OSS）是最合适的选择。

千万不要把这三类存储混为一谈。我曾经见过一个团队试图将 50 GB 的生成 PDF 文件直接塞进 Postgres，理由是“它有个 `bytea` 列可以用”。结果呢？成本比使用 OSS 高出十倍，查询延迟高得离谱，备份时间更是长达数小时，整个系统几乎被拖垮。
## 第 1 层：关系型，RDS for PostgreSQL

会话状态——对话轮次、工具调用 trace、用户身份——你需要一个真正的 RDBMS。PostgreSQL 是我的默认；MySQL 也行，如果团队偏好。需要横向 scale 时升 PolarDB。

```hcl
resource "random_password" "rds_admin" {
  length  = 32
  special = true
}

resource "alicloud_kms_secret" "rds_admin" {
  secret_name              = "agents-${terraform.workspace}-rds-admin"
  secret_data              = random_password.rds_admin.result
  version_id               = "v1"
  description              = "agents-${terraform.workspace} 的 RDS admin 密码"
  encryption_key_id        = module.vpc.kms_keys["secrets"]
  recovery_window_in_days  = 7
  force_delete_without_recovery = false
}

resource "alicloud_db_instance" "memory" {
  engine           = "PostgreSQL"
  engine_version   = "16.0"
  instance_type    = terraform.workspace == "prod" ? "pg.x4.large.2c" : "pg.n2.medium.1c"
  instance_storage = 100
  instance_name    = "agents-memory-${terraform.workspace}"

  vswitch_id          = module.vpc.private_vswitch_ids[0]
  security_ips        = [module.vpc.vpc_cidr_block]
  db_instance_storage_type = "cloud_essd"

  encryption_key = module.vpc.kms_keys["memory"]

  backup_period   = ["Monday", "Wednesday", "Friday"]
  backup_time     = "02:00Z-03:00Z"
  retention_period = terraform.workspace == "prod" ? 30 : 7
  log_backup_retention_period = 30
  preferred_backup_period = ["Monday", "Wednesday", "Friday"]

  deletion_protection = terraform.workspace == "prod"

  zone_id = "cn-shanghai-l"
  zone_id_slave_a = terraform.workspace == "prod" ? "cn-shanghai-m" : null
}

resource "alicloud_db_account" "agent" {
  db_instance_id   = alicloud_db_instance.memory.id
  account_name     = "agent"
  account_password = random_password.rds_admin.result
  account_type     = "Super"
}

resource "alicloud_db_database" "session" {
  instance_id   = alicloud_db_instance.memory.id
  name          = "agent_sessions"
  character_set = "UTF8"
}
```

要点：

- **密码出生即在 KMS Secrets Manager 里。** `random_password` 生成，写进 `alicloud_kms_secret`，Agent 启动时通过 STS 取。明文密码只在 Terraform 内存里短暂存在，不进 tfstate（用 `secret_id` 引用）。
- **`encryption_key`** 把磁盘和 `memory` CMK 绑定。静态加密，零额外成本。
- **`backup_period` + `retention_period`** 一周三次自动备份，prod 留 30 天、dev 留 7 天。RDS 备份存在 OSS 上，bucket 不用你管。
- **`zone_id_slave_a`** 在 prod 里在第二个 zone 建热备。30 秒内 failover。代价是 2 倍——prod 值得，dev 过头。
- **`deletion_protection`** 在 prod 里挡住 `terraform destroy` 杀数据库。永远要开。

> **实操提示：** 当你的 sessions 表过 ~10M 行或者需要无停机读副本时，PolarDB 是对的选择。RDS → PolarDB 迁移文档完善，Terraform 两边都支持。但别从一开始就用——小规模时 RDS 更简单更便宜。

## 第 2 层：向量存储

![用 Terraform 构建 AI Agent 的存储层（五）：向量、关系与对象记忆 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/illustration_2.jpg)

在阿里云上，针对向量存储这一层，我们有两个不错的选择：

1. **OpenSearch 向量检索版**——托管服务，基于 Lucene，支持 HNSW 和 IVF 算法，按 QPS 配额计费。
2. **PolarDB 或 RDS PostgreSQL 搭配 `pgvector` 插件**——与关系型数据共存，无需额外基础设施成本，但在数据规模超过约 100 万条向量后性能会下降。

如果是生产环境而非简单原型开发，我个人更推荐 OpenSearch。虽然它的成本不低（最小实例大约 ¥800/月），但它提供了开箱即用的混合检索能力（词项检索 + 向量检索），这种特性非常适合召回场景。

```hcl
resource "alicloud_opensearch_app_group" "vector" {
  app_group_name  = "agent-vec-${terraform.workspace}"
  payment_type    = "PayAsYouGo"
  type            = "vector"
  quota {
    doc_size   = 100
    compute_resource = 20
    spec       = "opensearch.share.junior"
  }
  description = "Agent 的长期语义记忆存储"
}
```

在 OpenSearch 中，`app group` 是一个逻辑概念，用于管理索引。你可以通过 OpenSearch 控制台或 SDK 创建索引模式（schema）。虽然 Terraform 提供了 `alicloud_opensearch_app` 资源，但 schema 的定义属于运维范畴，而不是基础设施即代码的一部分。

如果你选择使用 `pgvector`，可以在创建 RDS 数据库时添加以下配置：

```hcl
resource "alicloud_db_database" "vectors" {
  instance_id   = alicloud_db_instance.memory.id
  name          = "agent_vectors"
  character_set = "UTF8"
}

# pgvector 扩展需要通过迁移工具创建，而不是直接用 Terraform：
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE embeddings (id bigserial primary key, vec vector(1536), meta jsonb);
# CREATE INDEX embeddings_vec_idx ON embeddings USING hnsw (vec vector_cosine_ops);
```

Terraform 的职责仅限于数据库的创建，而表结构（schema）的设计和管理应该交给应用层的迁移工具（如 Alembic、Flyway 或 sqlx-migrate）。千万不要尝试用 Terraform 来管理表结构，这条路只会让你陷入无尽的麻烦。
## 第三层：对象存储

OSS 是存放各类产物的地方，比如生成的图片、PDF 文件、截图、运行追踪的 tar 包，以及微调模型时生成的 checkpoint 文件。

官方文档《使用 Terraform 创建存储桶》已经涵盖了基础内容。而对于 Agent 栈，可以参考以下配置：

```hcl
resource "alicloud_oss_bucket" "artifacts" {
  bucket = "agents-artifacts-${terraform.workspace}-${random_id.suffix.hex}"
  acl    = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm   = "KMS"
    kms_master_key_id = module.vpc.kms_keys["memory"]
  }

  lifecycle_rule {
    id      = "agent-artifacts-tiering"
    enabled = true

    transitions {
      days          = 30
      storage_class = "IA"
    }
    transitions {
      days          = 90
      storage_class = "Archive"
    }
    transitions {
      days          = 365
      storage_class = "ColdArchive"
    }
    expiration {
      days = 730
    }
  }

  logging {
    target_bucket = alicloud_oss_bucket.access_logs.id
    target_prefix = "artifacts-access/"
  }

  tags = {
    Domain = "agent-artifacts"
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}
```

这里有三点值得深入探讨：

### 存储桶名称的全局唯一性

OSS 存储桶的名称在所有阿里云用户中是全局唯一的。`random_id` 后缀的设计正是为了避免新手常遇到的“名称已被占用”问题，导致 `terraform plan` 失败。一旦存储桶创建完成，名称就会固定下来，不再变化。

### 生命周期分层策略

`lifecycle_rule` 配置块是 OSS 成本优化的关键所在：

![OSS 生命周期管理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig2_oss_lifecycle.png)

- **标准存储**（0-30 天，约 ¥0.12/GB/月）——默认写入的存储类型。
- **低频访问存储**（30-90 天，约 ¥0.08/GB/月）——存储成本更低，但每 GB 数据取回需额外支付 ¥0.0125。
- **归档存储**（90-365 天，约 ¥0.033/GB/月）——数据取回时间从分钟到小时不等。
- **冷归档存储**（365 天以上，约 ¥0.015/GB/月）——取回时间以小时计，成本最低。

对于 Agent 的产物，这条规则的意思是：前 30 天保持热存储，随后转入低频访问存储，3 个月后转入归档存储，1 年后转入冷归档存储，2 年后自动删除。假设有一个 1 TB 的产物集合，这种策略可以让每月的存储成本从 ¥1500（全部使用标准存储）降低到 ¥250 左右。将这一规则用 HCL 编码一次，一年下来能节省五位数的成本。

### 版本控制

`versioning { status = "Enabled" }` 的作用是保留每个对象的所有历史版本。例如，当某个 Agent 覆盖了 `artifacts/run-123/output.pdf` 文件时，之前的版本并不会被真正删除，而是以不同的版本 ID 保存下来。这一点之所以重要，主要有两个原因：

1. **数据恢复。** 如果因为 Bug 导致 50,000 个对象被错误覆盖为无效数据，可以通过脚本快速恢复到之前的版本。
2. **防篡改能力。** 结合 WORM（Write-Once-Read-Many）策略，可以轻松满足合规性要求。

当然，启用版本控制也会带来额外的成本——历史版本会不断累积。因此，建议在生命周期规则中添加 `noncurrent_version_expiration`，例如设置 180 天后自动清理非当前版本的数据，从而有效控制存储开销。
## 备份的那些事儿

用 Terraform 管理的备份架构大致是这样的：

![备份不是可选项，而是预算内的必要投入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig3_backup_topology.png)

- **RDS**：自带自动化备份功能（已经在前面的 HCL 配置中体现）
- **OSS**：通过版本控制和跨区域复制实现灾备
- **OpenSearch**：利用 `alicloud_opensearch_*` 快照资源将数据备份到 OSS

OSS 的跨区域复制配置其实只需要定义一个资源：

```hcl
resource "alicloud_oss_bucket" "artifacts_dr" {
  provider = alicloud.beijing       # 第二 region 的 provider 别名
  bucket   = "${alicloud_oss_bucket.artifacts.bucket}-dr"
  acl      = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm = "AES256"        # KMS 密钥是区域级别的，灾备场景下直接用 AES256 更简单
  }
}

resource "alicloud_oss_bucket_replication" "artifacts" {
  bucket = alicloud_oss_bucket.artifacts.id

  action = "ALL"
  destination {
    bucket   = alicloud_oss_bucket.artifacts_dr.bucket
    location = "oss-cn-beijing"
  }
  enable_historical_object_replication = "enabled"

  encryption_configuration {
    replica_kms_key_id = "alias/agents-prod-memory-dr"
  }
}
```

通过给 provider 设置别名，可以让一次 Terraform 执行同时操作两个区域：

```hcl
provider "alicloud" {
  alias  = "beijing"
  region = "cn-beijing"
}
```

对于以无状态为主的研发型 Agent，你可能会觉得灾备（DR）带来的存储成本翻倍不划算。但如果是面向客户的 Agent，尤其是需要长期保存对话历史以满足合规要求的场景，灾备就是必不可少的。

> **实战建议：** 每季度做一次恢复演练。从未验证过的备份不过是一场昂贵的幻想。我每个月都会运行一个叫 `restore-drill.sh` 的脚本，随机抽取一个 RDS 备份，恢复到 `cn-shanghai-dr` 实例上，并执行 schema 和 checksum 校验。这短短 30 分钟，是我每个月最有价值的时间投入。
## 计算与存储的连接

在第四篇文章中提到的 ECS 实例需要实际访问这些存储资源。这里分为三个关键部分：

1. **网络**——这部分已经完成。`agent_runtime_sg_id` 是从 VPC 模块输出的，它被用作 `memory_rds_sg` 和 `vector_store_sg` 入站规则的来源。
2. **凭证**——Agent 在启动时通过 STS 从 KMS Secrets Manager 获取数据库密码：
   ```python
   from alibabacloud_kms20160120.client import Client as KmsClient
   resp = kms_client.get_secret_value(GetSecretValueRequest(secret_name="agents-prod-rds-admin"))
   db_password = resp.body.secret_data
   ```
3. **访问地址**——Terraform 输出了这些地址：
   ```hcl
   output "rds_endpoint" {
     value = alicloud_db_instance.memory.connection_string
   }
   output "vector_endpoint" {
     value = alicloud_opensearch_app_group.vector.api_domain
   }
   output "artifacts_bucket" {
     value = alicloud_oss_bucket.artifacts.bucket
   }
   ```

Agent 会从 cloud-init 设置的环境变量中读取这些信息，而这些环境变量则是由 Terraform 的输出自动生成的。整个过程无需硬编码任何地址，也无需手动维护配置文件。
## 每月成本估算（开发环境、低流量场景）

- RDS PostgreSQL (`pg.n2.medium.1c`，100 GB ESSD）：约 ¥350/月  
- OpenSearch 向量引擎（最小规格）：约 ¥800/月  
- OSS（10 GB 标准存储，开启生命周期管理）：约 ¥1.5/月 + 流量费用  
- KMS（详见第三篇文章）：约 ¥10/月  

在开发环境中，存储层的月成本大约为 ¥1200。而在生产环境中，如果使用高可用 RDS、更大规格的 OpenSearch 以及更多的 OSS 存储，成本会达到 ¥3000 至 ¥5000/月。这也是成本压力开始显现的地方——第七篇文章将详细介绍如何对成本进行追踪和设置告警。
## 接下来的内容

第六篇文章将在第四篇中配置的计算资源和刚刚创建的存储资源前搭建 LLM 网关。这个网关是 API 密钥管理、配额限制执行以及按 Agent 归因成本的核心所在。完成第六篇后，你将拥有一个完整的、可运行 Agent 的技术栈——最后两篇文章将进一步在其上集成可观测性和成本控制能力。
## 那个 apply 错 workspace 的夜晚

这件事让我赔上了一个周六。值得分享的原因在于，解决方法是结构性的改进，而不仅仅是“下次小心点”。

背景是这样的：我有三个 Terraform 工作区——`dev`、`staging` 和 `prod`。开发环境的 RDS 用的是一个小型实例 `pg.n2.medium.1c`，而生产环境则是带高可用（HA）的 `pg.x4.large.2c`。那天我正在笔记本上干活，切换到一个功能分支后，运行了 `terraform plan` 检查改动。输出显示“计划新增 2 项，修改 1 项，删除 0 项”，看起来很干净，于是我直接运行了 `terraform apply`，然后起身去泡咖啡。

等我回来时，发现生产环境的数据库已经被销毁了。

问题的根源在于：之前的一次会话中，我选择了 `prod` 工作区，但后来忘了切回 `dev`。这次提交的改动（一个标签微调）在开发环境是无害的，但在生产环境，看似“修改 1 项”的操作实际上触发了强制重建。原因是主分支中引入了一次与本次改动无关的 provider 升级，导致 RDS 实例需要重新创建。然而，provider 的计划输出并没有清晰地提示这一点。

以下是引发问题的 HCL 代码片段：

```hcl
# PR 中看似无害的改动
resource "alicloud_db_instance" "memory" {
  # ... 其他配置未变 ...
  parameter {
    name  = "log_min_duration_statement"
    value = "100"   # 原值为 "1000"，一次无害的调优
  }
}
```

问题出在 provider 的 1.231 版本中，这个参数对某些数据库引擎版本被标记为 `force_new`。计划输出显示的是 `~ parameter`，看起来像是原地修改，但实际执行时却触发了资源重建。

从自动备份恢复花了整整 90 分钟（幸好备份是最新的）。事后复盘总结出了四个结构性改进措施，这些措施在两年内成功避免了类似问题的再次发生。

### 改进 1：为生产环境有状态资源添加 `lifecycle { prevent_destroy = true }`

```hcl
resource "alicloud_db_instance" "memory" {
  # ... 配置 ...

  lifecycle {
    prevent_destroy = local.is_prod
  }
}
```

现在，任何试图销毁生产环境 RDS 的 `terraform apply` 操作都会报错，提示 `Resource has lifecycle.prevent_destroy set`。如果确实需要销毁，必须手动移除 HCL 中的相关配置，提交 PR 并通过审批后才能继续。仅仅这一行配置，就能避免那次周六事故的发生。

### 改进 2：Shell 提示当前工作区

为了避免误操作，我在 Shell 中加入了一个函数，每次运行 `terraform` 命令时都会提醒当前的工作区：

```bash
function terraform() {
  local ws=$(/usr/local/bin/terraform workspace show 2>/dev/null)
  if [[ "$ws" == "prod" ]]; then
    echo -e "\033[1;31m================================\033[0m"
    echo -e "\033[1;31m WARNING: 当前工作区为 prod\033[0m"
    echo -e "\033[1;31m================================\033[0m"
    read -p "确定继续吗？[y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && return 1
  fi
  /usr/local/bin/terraform "$@"
}
```

这短短 1 秒的暂停足以打破“自动驾驶”模式。我把这段代码加到了 `.zshrc` 文件中，之后再也没有误操作过生产环境。

### 改进 3：生产环境的 apply 只能通过 CI 执行

更彻底的做法是：完全禁止从本地笔记本对生产环境的状态文件执行 `apply` 操作。只有 GitHub Actions 的运行器拥有针对生产环境状态文件前缀的 `oss:PutObject` 权限。本地运行 `terraform plan` 仍然可以正常读取状态文件，但尝试执行 `apply` 时会因权限不足而失败，提示 `AccessDenied`。

```hcl
# 开发者角色绑定的策略
{
  "Effect": "Deny",
  "Action": "oss:PutObject",
  "Resource": "acs:oss:*:*:ck-tfstate-prod/agents/env:prod/*"
}
```

开发者角色上的 `Deny` 策略优先级高于任何 `Allow`。开发者可以执行计划操作，但只有 CI 系统能够执行应用操作。CI 流程还受到 PR 审核的严格控制。

### 改进 4：增加 `pre-apply` 钩子，汇总销毁操作

为了进一步降低风险，我还编写了一个 `pre-apply` 钩子，用于检查计划中是否包含销毁操作，并要求显式确认：

```bash
# .git/hooks/pre-commit（或 tflint 自定义规则）
plan_file=$1
n_destroy=$(terraform show -json "$plan_file" | jq '[.resource_changes[] | select(.change.actions[] == "delete")] | length')
if [[ "$n_destroy" -gt 0 ]]; then
  echo "计划将销毁 $n_destroy 个资源："
  terraform show -json "$plan_file" | jq -r '.resource_changes[] | select(.change.actions[] == "delete") | .address'
  echo "请使用 DESTROY=yes terraform apply 确认"
  [[ "$DESTROY" != "yes" ]] && exit 1
fi
```

任何包含销毁操作的计划都需要在环境变量中显式设置 `DESTROY=yes`。这种显式的确认方式几乎不可能因为手滑而误触发。这是对 `prevent_destroy` 的补充，能够捕获那些发生在未锁定子模块中的销毁操作。

### 总结

单独采用上述任何一个改进措施都不足以完全避免类似问题，但四者结合在一起，就使得这种故障在结构上变得不可能发生。
## 使用 Terraform 进行 RDS 大版本升级

Postgres 每两年发布一个大版本，旧版本随之进入 EOL（生命周期终止）。RDS for PostgreSQL 的大版本升级是一个实实在在的运维操作——会有停机时间，可能失败，而 Terraform 提供的 `engine_version` 参数看似简单无害，实则暗藏玄机。

以下是我在 v15 升级到 v16 时验证过的一套流程：

### 第一步：动手前先备份

```bash
aliyun rds CreateBackup \
  --DBInstanceId pgm-uf6abc123 \
  --BackupMethod Physical \
  --BackupType FullBackup
```

等待备份完成（可以通过 `aliyun rds DescribeBackups ...` 查看状态）。这是你的“救命稻草”，关键时刻能用来回滚。

### 第二步：创建一个兄弟实例用于测试

```hcl
resource "alicloud_db_instance" "memory_v16_trial" {
  engine           = "PostgreSQL"
  engine_version   = "16.0"
  instance_type    = alicloud_db_instance.memory.instance_type
  instance_storage = alicloud_db_instance.memory.instance_storage
  vswitch_id       = module.vpc.private_vswitch_ids[0]

  source_db_instance_name = alicloud_db_instance.memory.id
  backup_id               = data.alicloud_db_backups.latest.ids[0]

  instance_name = "memory-v16-trial"
}
```

执行 `terraform apply` 后，会基于 v15 的备份数据拉起一个 v16 实例。QA 团队可以将测试流量指向这个实例，观察一周左右，确保一切正常。这一步对生产环境没有任何影响。

### 第三步：确认无误后原地升级

真正的生产环境升级只需修改 `engine_version` 参数：

```hcl
resource "alicloud_db_instance" "memory" {
  engine_version = "16.0"   # 原本是 "15.0"
}
```

运行 `terraform plan` 时，你会看到类似 `~ engine_version: "15.0" -> "16.0"` 的变更提示，执行 `terraform apply` 后触发原地升级。停机时间取决于数据库大小——小型数据库通常在 1 分钟内完成，而多 TB 的大型数据库可能需要 30 分钟左右。如果升级失败，只能通过第一步的备份进行回滚。

### 第四步：清理测试实例

```bash
terraform state rm alicloud_db_instance.memory_v16_trial
# 删除对应的 HCL 配置块
# 下次执行 plan 时显示：0 changes
```

然后通过控制台或 API 删除测试实例。注意，不要在同一个项目中使用 `terraform destroy` 删除它，因为这可能会触发依赖关系，影响其他资源。

整个流程大约需要两周的时间（包括测试和观察），但实际投入的工作时间只有 3 小时左右，并且不会产生计划外的停机。通过 Terraform 管理的好处在于，测试实例的 HCL 配置会被保存在 Git 中——半年后升级到 v17 时，这套流程可以直接复用。

> **实战建议：** 在正式升级之前，务必在 agent 代码中测试连接字符串的变更。Postgres v16 引入了一些新特性（例如移除了 `password_encryption = md5`），这些改动可能导致旧版客户端库无法正常工作。建议在测试实例上运行 agent 至少一整天，确保兼容性后再进行切换。
## 跨区 DR 演练：证明 OSS 副本真能用

前文配过 `alicloud_oss_bucket_replication` 到北京副本。从未恢复过的副本就是不存在的备份。每月演练，脚本化：

```bash
#!/bin/bash
# scripts/dr-drill.sh —— 每月 1 号 CI cron
set -euo pipefail

PRIMARY_BUCKET="agents-artifacts-prod-abc12345"
REPLICA_BUCKET="${PRIMARY_BUCKET}-dr"
DRILL_KEY="dr-drill/$(date -Iseconds).probe"

# 1. 主桶写探针
echo "drill $(uuidgen)" | aliyun oss put oss://$PRIMARY_BUCKET/$DRILL_KEY -

# 2. 等复制（China 内通常 <60s）
for i in {1..30}; do
  if aliyun oss --region cn-beijing stat oss://$REPLICA_BUCKET/$DRILL_KEY > /dev/null 2>&1; then
    echo "Replicated in $((i*5))s"
    break
  fi
  sleep 5
done

# 3. 校验内容一致
PRIMARY_HASH=$(aliyun oss cat oss://$PRIMARY_BUCKET/$DRILL_KEY | sha256sum | awk '{print $1}')
REPLICA_HASH=$(aliyun oss --region cn-beijing cat oss://$REPLICA_BUCKET/$DRILL_KEY | sha256sum | awk '{print $1}')
[[ "$PRIMARY_HASH" == "$REPLICA_HASH" ]] || { echo "HASH MISMATCH"; exit 1; }

# 4. 清理探针
aliyun oss rm oss://$PRIMARY_BUCKET/$DRILL_KEY
aliyun oss --region cn-beijing rm oss://$REPLICA_BUCKET/$DRILL_KEY

# 5. 成功通知钉钉
curl -X POST "$DINGTALK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d '{"msgtype":"text","text":{"content":"DR drill OK at '$(date -Iseconds)'"}}'
```

副本不健康，30 天内你就知道，不是真灾难发生时才知道。演练本身 2 分钟算力、零人工注意力。和 drift check 挂同一个 GitHub Actions cron 上。

这种模式——周期性自动演练那些你否则会在最坏时间发现是坏的东西——是我建立过杠杆率最高的运维习惯。OSS 复制、RDS 恢复、KMS 密钥轮换、每条 failover 路径都套上。
