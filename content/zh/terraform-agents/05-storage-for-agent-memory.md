---
title: "Terraform 实战（五）：向量库、RDS 与对象存储"
date: 2026-03-20 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - OSS
  - RDS
  - OpenSearch
  - AI Agents
categories: Terraform
lang: zh
mathjax: false
series: terraform-agents
series_title: "用 Terraform 在阿里云上部署 AI Agent"
series_order: 5
description: "Agent 有三种记忆，分别落到三个阿里云服务上：会话用 PolarDB/RDS，embedding 用 OpenSearch 向量版或 pgvector，产物用 OSS。每一层的真实 Terraform，再加上让账单不暴涨的 lifecycle 和备份规则。"
disableNunjucks: true
translationKey: "terraform-agents-5"
---
大多数教程讲到 Agent 记忆这块都在糊弄。“把 embeddings 扔 Pinecone，sessions 扔 Postgres，截图扔 S3 就完事了。”在阿里云上，这三类服务均有托管方案，但 Terraform 配置是否合理，直接决定了记忆功能的稳定性：配置不当可能导致凌晨 4 点磁盘告警、三周对话历史永久丢失。

本文将系统梳理这三层架构：各层对应的 Terraform 实现、看似枯燥却至关重要的备份与容灾机制、PostgreSQL 大版本升级的实操经验，以及一次真实的周六故障——正是这次事故，直接塑造了我后续所有关键架构决策。

## 三层记忆模型

![An agent's three kinds of memory map onto three Aliyun services](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig1_memory_three_layers.png)

我们采用如下三层记忆模型：

- **短期 / 会话（Short-term / session）** —— Agent 在当前 run 和最近几次 run 里做了什么。对话轮次、工具调用、中间状态。Schema 稳定、读写低延迟、支持事务——适合存入关系型数据库。
- **长期 / 语义（Long-term / semantic）** —— 文档 embeddings、 prior outputs、recall corpus。混合 lexical + vector 搜索。扔进向量存储。
- **Artifact / blob** —— 生成的图片、PDF、截图、run 快照。通常很大，写一次读很少。扔进对象存储。

别把它们混为一谈。我见过有团队想把 50GB 生成的 PDF 塞进 Postgres，理由是“它有 `bytea` 列”。其存储成本可达 OSS 的十倍，查询延迟明显上升，全量备份耗时甚至长达数小时。每层都有专门擅长干这个活的服务——选对服务，账单才能控制在合理范围内。

## 第一层：关系型数据库，RDS for PostgreSQL

会话状态——逐轮对话、工具调用 trace、用户身份——你得用正经 RDBMS。PostgreSQL 是我的默认选项；如果团队偏好 MySQL 也没问题。需要水平扩展时，下一步是 PolarDB。

```hcl
resource "random_password" "rds_admin" {
  length  = 32
  special = true
}

resource "alicloud_kms_secret" "rds_admin" {
  secret_name              = "agents-${terraform.workspace}-rds-admin"
  secret_data              = random_password.rds_admin.result
  version_id               = "v1"
  description              = "RDS admin password for agents-${terraform.workspace}"
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

  deletion_protection = terraform.workspace == "prod"

  zone_id         = "cn-shanghai-l"
  zone_id_slave_a = terraform.workspace == "prod" ? "cn-shanghai-m" : null

  lifecycle {
    prevent_destroy = terraform.workspace == "prod"
  }
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

该模块配置中有几个关键点需特别说明：

- **密码自创建起即存储于 KMS Secrets Manager。** 由 `random_password` 生成，写入 `alicloud_kms_secret`，Agent 启动时通过 STS 获取。明文永远不会离开 Terraform 的内存，下游通过 `secret_id` 引用而不是值，所以不会留在 tfstate 里。
- **`encryption_key`** 把磁盘绑定到 `memory` CMK 上。落盘加密，没有额外成本。
- **`backup_period` + `retention_period`** 创建每周三次的自动备份，prod 保留 30 天，dev 保留 7 天。RDS 自动备份默认存储在 OSS 中，无需额外管理备份 Bucket。
- **`zone_id_slave_a`** 在 prod 环境会在第二个可用区创建热备。故障切换时间低于 30 秒；但成本增加一倍——生产环境值得投入，开发环境则无需启用。
- **`deletion_protection`** 加上 **`lifecycle.prevent_destroy`** 在 prod 环境能挡住 `terraform destroy` 和 provider 驱动的强制替换。下文故障复盘部分我会解释为什么两者都需要——简言之，其中一项防护措施成功避免了那次周六故障。

> **提示。** 当 sessions 表数据量超过约 1000 万行，或需要无停机读副本能力时，PolarDB 才是更合适的选择。RDS 迁移至 PolarDB 的官方文档完备，且 Terraform 对两者均提供良好支持。建议初期优先选用 RDS——其架构更简单，成本也更低。

关系型数据库层是整个记忆系统的基石；而向量存储层，才是真正赋予 Agent 语义检索能力的核心。

## 第二层：向量存储

阿里云上向量层有两个合理选择：

![Vector embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/wanxiang_vector_search.png)


1. **OpenSearch Vector Search Edition** —— 托管版，基于 Lucene，支持 HNSW + IVF，按 QPS 配额计费。
2. **带 `pgvector` 的 PolarDB 或 RDS PostgreSQL** —— 和关系数据在一起，不用新 infra，超过 ~100 万向量后变慢。

过了原型阶段我首选 OpenSearch。成本明确（最小规格实例约 ¥800/月），但开箱即支持 lexical + vector 混合搜索——这才是生产级检索应有的形态：纯向量相似度在大量真实查询中表现不如 BM25，这一点不可忽视。

```hcl
resource "alicloud_opensearch_app_group" "vector" {
  app_group_name  = "agent-vec-${terraform.workspace}"
  payment_type    = "PayAsYouGo"
  type            = "vector"
  quota {
    doc_size         = 100
    compute_resource = 20
    spec             = "opensearch.share.junior"
  }
  description = "Long-term semantic memory for agents"
}
```

App Group 是 OpenSearch 中用于组织索引的逻辑单元。从这里开始你要通过 OpenSearch 控制台或 SDK 创建索引 schema——`alicloud_opensearch_app` 资源确实存在，但 索引 schema 需通过运维操作配置，而非 Terraform 预配。在索引设置里定死 embedding 维度（`text-embedding-3-small` 是 1536，阿里云 bge-m3 是 1024），请勿随意修改；重建 1000 万向量的索引可能耗时数天。

如果走 pgvector 路线，在 RDS 数据库创建时加上这个：

```hcl
resource "alicloud_db_database" "vectors" {
  instance_id   = alicloud_db_instance.memory.id
  name          = "agent_vectors"
  character_set = "UTF8"
}

# pgvector extension is created via your migration tool, not Terraform:
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE embeddings (id bigserial primary key, vec vector(1536), meta jsonb);
# CREATE INDEX embeddings_vec_idx ON embeddings USING hnsw (vec vector_cosine_ops);
```

Terraform 只管数据库；schema 是应用代码的事（Alembic、Flyway、sqlx-migrate——选一个）。切勿尝试在 Terraform 中管理表结构变更——这极易引发状态混乱，我们团队已在实践中多次踩坑。

## 第三层：对象存储

OSS 是 artifacts 的去处：生成的图片、PDF、截图、run-trace tarballs，如果你做 fine-tune 还有模型 checkpoints。对于 Agent 栈：

![Data lifecycle management](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/wanxiang_data_lifecycle.png)


```hcl
resource "alicloud_oss_bucket" "artifacts" {
  bucket = "agents-artifacts-${terraform.workspace}-${random_id.suffix.hex}"
  acl    = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm     = "KMS"
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
    noncurrent_version_expiration {
      days = 180
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

三点值得细看。

### Bucket 名称唯一性

OSS Bucket 名称在阿里云全网范围内全局唯一，规则与 AWS S3 一致。`random_id` 后缀能避开每个新手都会踩的“名称已被占用”plan 失败。Bucket 创建后，名称即永久固定，不可更改。

### 生命周期分层

`lifecycle_rule` 块是 OSS 里最大的成本杠杆：

![OSS lifecycle for agent artifacts](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig2_oss_lifecycle.png)

- **Standard**（0–30 天，~¥0.12/GB/月）—— 默认写入层。
- **Infrequent Access**（30–90 天，~¥0.08/GB/月）—— 存储更便宜，检索 ~¥0.0125/GB。
- **Archive**（90–365 天，~¥0.033/GB/月）—— 分钟到小时级检索。
- **Cold Archive**（365 天+，~¥0.015/GB/月）—— 小时级检索，最便宜层。

针对 Agent 的 artifacts，我们采用如下生命周期策略：标准存储（Standard）保留 30 天，低频访问（IA）保留 2 个月，归档（Archive）保留 9 个月，冷归档（Cold Archive）保留 1 年，期满后自动删除。对于 1TB 的 artifact corpus，这决定了账单是 ~¥1500/月（全 Standard）还是 ~¥250/月。HCL 里写一次，一年省五位数。IA 层设有最低存储时长（30 天），Archive 层存在分钟级检索延迟——只要避免将热数据误存至冷层，即可规避主要风险。

### 版本控制

`versioning { status = "Enabled" }` 保留每个对象版本。Agent 覆盖 `artifacts/run-123/output.pdf` 实际上不会销毁旧版本——它还在那，只是 version ID 不同。这有两点重要性：

1. **恢复。** 一个 bug 用垃圾数据覆盖了 50,000 个对象？写个脚本恢复旧版本。
2. **防篡改。** 配合 WORM (Write-Once-Read-Many) 策略，免费搞定合规。

版本对象会累积，所以上面生命周期规则里的 `noncurrent_version_expiration` 会在 180 天后修剪旧版本。没有它，存储账单每六个月就会悄悄翻倍。
## 备份、容灾，以及如何证明它们真的管用

用 Terraform 管备份，架构大概长这样：

![Backups: not optional, just budgeted](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig3_backup_topology.png)

- **RDS** — 内置自动备份（上面的 HCL 里已经配了）。
- **OSS** — 开启版本控制 + 跨地域复制，用来做容灾。
- **OpenSearch** — 通过 `alicloud_opensearch_*` 快照资源把快照打到 OSS 上。

OSS 的跨地域复制只需要一个资源：

```hcl
resource "alicloud_oss_bucket" "artifacts_dr" {
  provider = alicloud.beijing       # second-region provider alias
  bucket   = "${alicloud_oss_bucket.artifacts.bucket}-dr"
  acl      = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm = "AES256"        # KMS keys are region-scoped; AES256 keeps the DR bucket simple
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

通过 provider alias，一次 Terraform 运行就能操作两个地域：

```hcl
provider "alicloud" {
  alias  = "beijing"
  region = "cn-beijing"
}
```

如果是那种 mainly stateless 的研究型 Agent，你可能觉得容灾不值得让存储成本翻倍。但要是面向客户、涉及对话历史且法律要求必须持久化的场景，这就是 mandatory。

### 证明副本真的能用——每月演练

未经恢复验证的副本，本质上等同于无效备份。下面这个脚本能在灾难发生前 30 天 catches a broken replica，而不是等到真出事的时候：

```bash
#!/bin/bash
# scripts/dr-drill.sh — run on the 1st of every month from a CI cron
set -euo pipefail

PRIMARY_BUCKET="agents-artifacts-prod-abc12345"
REPLICA_BUCKET="${PRIMARY_BUCKET}-dr"
DRILL_KEY="dr-drill/$(date -Iseconds).probe"

# 1. Write a probe to primary
echo "drill $(uuidgen)" | aliyun oss put oss://$PRIMARY_BUCKET/$DRILL_KEY -

# 2. Wait for replication (typically <60s within China)
for i in {1..30}; do
  if aliyun oss --region cn-beijing stat oss://$REPLICA_BUCKET/$DRILL_KEY > /dev/null 2>&1; then
    echo "Replicated in $((i*5))s"
    break
  fi
  sleep 5
done

# 3. Verify content match
PRIMARY_HASH=$(aliyun oss cat oss://$PRIMARY_BUCKET/$DRILL_KEY | sha256sum | awk '{print $1}')
REPLICA_HASH=$(aliyun oss --region cn-beijing cat oss://$REPLICA_BUCKET/$DRILL_KEY | sha256sum | awk '{print $1}')
[[ "$PRIMARY_HASH" == "$REPLICA_HASH" ]] || { echo "HASH MISMATCH"; exit 1; }

# 4. Clean up the probe
aliyun oss rm oss://$PRIMARY_BUCKET/$DRILL_KEY
aliyun oss --region cn-beijing rm oss://$REPLICA_BUCKET/$DRILL_KEY

# 5. Notify DingTalk on success
curl -X POST "$DINGTALK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d '{"msgtype":"text","text":{"content":"DR drill OK at '$(date -Iseconds)'"}}'
```

该脚本仅消耗约两分钟计算资源，全程无需人工干预，副本健康状态从此由实证而非假设保障。把它跟第二篇文章里的 drift check 一样挂到 GitHub Actions cron 上。同样的模式——定期自动化演练那些否则只能在最糟糕时刻才发现坏了的东西——也适用于 RDS 恢复、KMS 密钥轮转，以及每一个 failover 路径。

> **Tip.** 我每月还会跑一个单独的 `restore-drill.sh`，把随机的 RDS 备份拉到 `cn-shanghai-dr` 实例里，跑 schema/checksum 验证。这是我每月投入产出比最高的一次运维操作，耗时约 30 分钟。

## 通过 Terraform 进行 RDS 大版本升级

Postgres 每两年一个大版本，旧版本就会 EOL。RDS for PostgreSQL 升级是个真正的 operational event——会有 downtime，可能失败，而且 Terraform provider 通过 `engine_version` 变更暴露出的接口看着无害，其实不然。

我在 v15 → v16 升级上跑通的流程是这样：

**Step 1 — 动任何东西之前先快照。**

```bash
aliyun rds CreateBackup \
  --DBInstanceId pgm-uf6abc123 \
  --BackupMethod Physical \
  --BackupType FullBackup
```

通过 `aliyun rds DescribeBackups` 等待完成。这是回滚至升级前状态的最终保障。

**Step 2 — 克隆一个兄弟实例做试验。**

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

`terraform apply` 拉起一个 v16 实例，恢复的是 v15 的数据。QA 把 staging 流量切过去跑一周。全程不影响生产环境。

**Step 3 — 有信心了，原地升级。**

```hcl
resource "alicloud_db_instance" "memory" {
  engine_version = "16.0"   # was "15.0"
  # everything else unchanged
}
```

`terraform plan` 显示 `~ engine_version: "15.0" -> "16.0"`，apply 触发原地升级。停机时长取决于数据库规模：小型实例通常在 1 分钟内完成，TB 级以上实例可能长达 30 分钟。升级只有通过 Step 1 的快照恢复才能回滚，所以别跳过 Step 1。

**Step 4 — 拆掉试验实例。**

```bash
terraform state rm alicloud_db_instance.memory_v16_trial
# remove the HCL block
# next plan: 0 changes
```

然后在控制台删除试验实例。别在同一个项目里 `terraform destroy` 它——那样会走依赖链，可能误伤兄弟资源。

整个过程日历时间约 2 周，专注工作时间约 3 小时，零计划外 downtime。采用 Terraform 的优势在于：试验实例的全部 HCL 配置均纳入 Git 版本管理。六个月后执行 v17 升级时，这份经过团队 Code Review 的实战手册已就绪于同一仓库。

> **Tip.** 升级前先在 agent 代码里测试 connection-string 变更。有些 Postgres v16 的改动（比如移除了 `password_encryption = md5`）会搞挂旧客户端库。在提升为主库前，让 agent 对着试验实例跑满一整天。

## 真实事故：那个我 `terraform apply` 错 workspace 的夜晚

本次事件导致我不得不在周六全天进行紧急修复。详述此事，是为了说明：最终落地的是一套系统性防护机制，而非依赖人工谨慎的临时性补救措施。

环境 setup：三个 workspace — `dev`、`staging`、`prod`。`dev` 的 RDS 是个小规格的 `pg.n2.medium.1c`。Prod 是 `pg.x4.large.2c` 带 HA。我当时用笔记本干活，切了个 feature 分支，跑 `terraform plan` 检查。看到 "Plan: 2 to add, 1 to change, 0 to destroy" — 看着挺干净。于是跑了 `terraform apply`。起身去冲了杯咖啡。

回来发现 prod 数据库没了。

根因：我在前一次会话里选了 `prod` workspace，之后忘了切回来。我要应用的变更（改个 tag）在 dev 里无害。但在 prod 里，看着像 "1 to change" 的操作实际上是 force-replace，因为 `main` 分支里合入了一个无关的 provider  bump，要求重建 RDS。Provider 的 plan 输出本该更清楚些——可惜并没有。

引发该问题的具体 HCL 片段如下：

```hcl
# innocent-looking change in a PR
resource "alicloud_db_instance" "memory" {
  # ... unchanged ...
  parameter {
    name  = "log_min_duration_statement"
    value = "100"   # was "1000", harmless tuning
  }
}
```

Provider 在新 bump 的 1.231 版本里，决定这个参数对某些 engine 版本现在是 `force_new` 了。Plan 显示 `~ parameter` — 看着像原地变更 — apply 却触发了重建。

从自动备份恢复花了 90 分钟（幸好备份够新）。事后复盘产出了四个结构性修复，过去两年防止了所有复发。

### 修复 1 — 给 prod 有状态资源加上 `lifecycle { prevent_destroy = true }`

本文前面 RDS 块里已经加上了。任何试图销毁 prod RDS 的 `terraform apply` 现在都会报错 `Resource has lifecycle.prevent_destroy set`。真要销毁的话，得先在 HCL 里删掉这行，提 PR，获得批准，*然后* 才允许销毁。单单这一行就能把我的周六 outage 直接拦下。

### 修复 2 — Shell 里的 workspace 提示

一个函数，每次调用 `terraform` 都会吼一嗓子：

```bash
function terraform() {
  local ws=$(/usr/local/bin/terraform workspace show 2>/dev/null)
  if [[ "$ws" == "prod" ]]; then
    echo -e "\033[1;31m================================\033[0m"
    echo -e "\033[1;31m WARNING: workspace = prod\033[0m"
    echo -e "\033[1;31m================================\033[0m"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && return 1
  fi
  /usr/local/bin/terraform "$@"
}
```

这 1 秒钟的停顿足以打断 autopilot。写在我的 `.zshrc` 里。从那以后我再没误操作过 prod。

### 修复 3 — prod apply 只走 CI，绝不用笔记本

更干净的版本：撤销笔记本对 prod state 文件的 apply 权限。只有 GitHub Actions runner 拥有的 RAM role 才有 prod state prefix 上的 `oss:PutObject` 权限。本地 `terraform plan` 照常工作（只读）；本地 `apply` 会报 `AccessDenied`。

```hcl
# Attached to the developer role
{
  "Effect": "Deny",
  "Action": "oss:PutObject",
  "Resource": "acs:oss:*:*:ck-tfstate-prod/agents/env:prod/*"
}
```

Developer role 上的 Deny 优先级高于任何 Allow。开发能 plan；只有 CI 能 apply。CI 运行还 gated by PR review。

### 修复 4 — 一个总结销毁操作的 `pre-apply` hook

```bash
# .git/hooks/pre-commit (or a tflint custom rule)
plan_file=$1
n_destroy=$(terraform show -json "$plan_file" | jq '[.resource_changes[] | select(.change.actions[] == "delete")] | length')
if [[ "$n_destroy" -gt 0 ]]; then
  echo "Plan would destroy $n_destroy resources:"
  terraform show -json "$plan_file" | jq -r '.resource_changes[] | select(.change.actions[] == "delete") | .address'
  echo "Confirm with DESTROY=yes terraform apply"
  [[ "$DESTROY" != "yes" ]] && exit 1
fi
```

任何涉及删除的 plan 都需要环境变量里设 `DESTROY=yes`。你不可能手滑敲出这个。这是 `prevent_destroy` 双重保险里的第二道保险——它能 catch 住那些你忘了锁定的子模块里的销毁操作。

把这四条全用上。单条救不了我；四条合在一起让这种 failure mode 在结构上变得不可能。

## 把计算节点连到存储

第四篇文章里的 ECS 实例得真正连上这些存储。三块内容：

1. **网络** — 已经搞定。VPC 模块里的 `agent_runtime_sg_id` 是 `memory_rds_sg` 和 `vector_store_sg` 入站规则的 source。
2. **凭证** — agent 通过 STS 从 KMS Secrets Manager 读 DB 密码：
   ```python
   from alibabacloud_kms20160120.client import Client as KmsClient
   resp = kms_client.get_secret_value(GetSecretValueRequest(secret_name="agents-prod-rds-admin"))
   db_password = resp.body.secret_data
   ```
3. **Endpoints** — Terraform 输出它们：
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

Agent 从环境变量里读这些值，环境变量由 cloud-init 根据 Terraform outputs 设置。没有硬编码的 endpoint，没有手动配置文件，轮转密钥也不需要人工介入。
## 到底要花多少钱

按月算，开发环境，低流量场景：

- RDS PostgreSQL (`pg.n2.medium.1c`, 100 GB ESSD)：约 ¥350/月。
- OpenSearch vector（最小规格）：约 ¥800/月。
- OSS（10 GB Standard，开启 lifecycle）：约 ¥1.5/月 + 流量费。
- KMS（第 3 篇讲过）：约 ¥10/月。

开发环境下，存储层加起来大概 ¥1200/月。要是上生产，加上 RDS 高可用、OpenSearch 扩容、OSS 用量增加，还有跨地域副本，一个月得 ¥3000–5000。这时候成本压力才算真正来了——别等到月底对账单才傻眼，第 7 篇我会手把手教你怎么提前监控和设报警。

## 下一步计划

第 6 篇的核心任务是搭建 LLM gateway，它挡在我们第 4 篇准备的计算资源和刚搞定的存储层前面。API key 管理、配额限制、每个 Agent 的成本分摊，全在这儿解决。走完第 6 篇，你就拥有一个完整的可运行 Agent 栈了——最后两篇我们再往上叠加可观测性和成本控制。