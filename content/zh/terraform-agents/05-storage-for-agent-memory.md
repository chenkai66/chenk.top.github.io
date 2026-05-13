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
大多数教程在讲解 Agent 记忆时都一笔带过，简单说一句“把 embeddings 放 Pinecone，会话存 Postgres，截图扔 S3”就完事。但在阿里云上，这三类存储其实都有对应的托管服务。能否用 Terraform 正确配置它们，直接决定了你的记忆系统是稳定运行，还是某天凌晨 4 点因磁盘爆满而丢失整整三周的对话历史。

本文将系统梳理这三层架构：各自的 Terraform 配置、看似枯燥却至关重要的备份与容灾（DR）机制、PostgreSQL 大版本升级的实际操作流程，以及一次真实的周六故障——正是那次事故彻底改变了我的运维方式。

## 三层记忆模型

![代理的三种内存映射到三种阿里云服务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig1_memory_three_layers.png)

我们采用如下三层记忆模型：

- **短期 / 会话（Short-term / session）** —— Agent 在当前 run 及最近几次 run 中的行为，包括对话轮次、工具调用记录和中间状态。这类数据 schema 稳定、要求低延迟、需事务支持，适合存入关系型数据库。
- **长期 / 语义（Long-term / semantic）** —— 文档的 embeddings、历史输出、召回语料库等，通常需要混合词法（lexical）与向量（vector）搜索能力，应存入向量数据库。
- **Artifact / blob** —— 生成的图片、PDF、截图、运行快照等，通常体积大、写一次后极少读取，适合存入对象存储。

千万别混用。我曾亲眼见过一个团队试图把 50 GB 的生成 PDF 塞进 Postgres，理由竟是“它有 `bytea` 列”。结果成本飙升至 OSS 的十倍，查询延迟变得难以忍受，全量备份甚至要花数小时。每一层都有专精其职的服务——选对工具，账单才能保持理性。

## 第一层：关系型数据库，RDS for PostgreSQL

会话状态——包括逐轮对话、工具调用 trace 和用户身份信息——必须依赖一个可靠的 RDBMS。我个人首选 PostgreSQL；如果你的团队更熟悉 MySQL，也完全可行。当需要水平扩展时，再考虑 PolarDB。

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

这段配置中有几个关键设计：

- **密码从创建起就存入 KMS Secrets Manager。** 它由 `random_password` 生成，通过 `alicloud_kms_secret` 写入，Agent 启动时通过 STS 获取。明文从未离开 Terraform 内存，下游仅通过 `secret_id` 引用（而非明文值），因此不会泄露到 tfstate 中。
- **`encryption_key`** 将磁盘绑定到名为 `memory` 的 CMK，实现落盘加密，且无额外费用。
- **`backup_period` + `retention_period`** 配置了每周三次的自动备份：生产环境保留 30 天，开发环境保留 7 天。RDS 备份默认存储在 OSS 上，你无需管理备份桶。
- **`zone_id_slave_a`** 在生产环境中会在第二可用区创建热备实例，故障切换时间低于 30 秒。虽然成本翻倍，但对生产环境值得投入，开发环境则不必启用。
- **`deletion_protection`** 加上 **`lifecycle.prevent_destroy`** 能同时阻止 `terraform destroy` 和 Provider 触发的强制替换。下文的事故复盘会解释为何两者缺一不可——简而言之，其中一项防护措施曾成功保住了我的周六。

> **提示。** 当 sessions 表超过约 1000 万行，或你需要无停机读副本时，PolarDB 才是更优选择。从 RDS 迁移到 PolarDB 的官方文档很完善，Terraform 对两者也都支持良好。初期建议优先使用 RDS——架构更简单，成本也更低。

这个轻量级的关系型层是整个记忆系统的骨架；而下一层，才是真正让 Agent 显得“聪明”的关键。

## 第二层：向量存储

在阿里云上，向量层有两个合理选项：

![向量嵌入空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/wanxiang_vector_search.png)

1. **OpenSearch Vector Search Edition** —— 托管服务，基于 Lucene，支持 HNSW + IVF，按 QPS 配额计费。
2. **带 `pgvector` 插件的 PolarDB 或 RDS PostgreSQL** —— 与关系数据共存，无需新增基础设施，但向量数量超过约 100 万后性能明显下降。

只要过了原型阶段，我都会优先选择 OpenSearch。虽然成本明确（最小规格约 ¥800/月），但它原生支持词法+向量混合搜索——这在真实检索场景中至关重要：纯向量相似度在大量实际查询中往往不如 BM25 表现好。

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

App Group 是 OpenSearch 中承载索引的逻辑单元。索引 schema 需通过控制台或 SDK 创建——尽管存在 `alicloud_opensearch_app` 资源，但 schema 配置属于运维操作，不适合用 Terraform 预配。务必在索引设置中固定 embedding 维度（`text-embedding-3-small` 为 1536，阿里云 bge-m3 为 1024），切勿随意更改；重建 1000 万条向量的索引可能耗时数天。

如果选择 pgvector 路线，在 RDS 数据库创建时加上以下配置：

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

Terraform 只负责数据库实例本身；表结构应由应用代码管理（Alembic、Flyway、sqlx-migrate——任选其一）。千万别尝试在 Terraform 中管理表 schema——这条路通往混乱，我们团队已为此付出过惨痛代价。

## 第三层：对象存储

OSS 用于存储各类 artifacts：生成的图片、PDF、截图、运行轨迹压缩包，以及如果你做微调，还有模型 checkpoints。对于 Agent 栈：

![数据生命周期管理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/wanxiang_data_lifecycle.png)

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

有三点值得特别关注。

### Bucket 名称唯一性

OSS Bucket 名称在阿里云全局唯一，规则与 AWS S3 相同。通过 `random_id` 后缀可避免新手常踩的“名称已被占用”错误。Bucket 一旦创建，名称即永久固定。

### 生命周期分层

`lifecycle_rule` 是 OSS 中最有效的成本控制杠杆：

![OSS 存储桶中代理工件的生命周期](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig2_oss_lifecycle.png)

- **Standard**（0–30 天，~¥0.12/GB/月）—— 默认写入层。
- **Infrequent Access**（30–90 天，~¥0.08/GB/月）—— 存储更便宜，但检索费用约 ¥0.0125/GB。
- **Archive**（90–365 天，~¥0.033/GB/月）—— 检索需几分钟到几小时。
- **Cold Archive**（365 天以上，~¥0.015/GB/月）—— 检索需数小时，价格最低。

针对 Agent artifacts，建议策略：Standard 保留 30 天，IA 保留 2 个月，Archive 保留 9 个月，Cold Archive 保留 1 年，之后自动删除。对于 1 TB 的 artifact corpus，这能让月成本从 ~¥1500（全 Standard）降至 ~¥250。在 HCL 中固化此策略，一年可省下五位数。主要风险在于 IA 层有 30 天最低存储期，Archive 层检索延迟高——只要避免将热数据误存入冷层即可。

### 版本控制

`versioning { status = "Enabled" }` 会保留每个对象的所有版本。即使 Agent 覆盖了 `artifacts/run-123/output.pdf`，旧版本依然存在，只是 version ID 不同。这带来两大价值：

1. **恢复能力。** 若 bug 覆盖了 5 万个对象，可通过脚本批量回滚到旧版本。
2. **防篡改证据。** 结合 WORM（Write-Once-Read-Many）策略，可免费满足合规要求。

但版本对象会持续累积，因此上述生命周期规则中的 `noncurrent_version_expiration` 会在 180 天后自动清理旧版本。若不设置，存储费用每六个月就会悄然翻倍。

## 备份、容灾，以及如何证明它们真的管用

用 Terraform 管理备份的典型架构如下：

![备份：不是可选项，而是预算内事项](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig3_backup_topology.png)

- **RDS** — 内置自动备份（前文 HCL 已配置）。
- **OSS** — 启用版本控制 + 跨地域复制，用于容灾。
- **OpenSearch** — 通过 `alicloud_opensearch_*` 快照资源将数据备份至 OSS。

OSS 跨地域复制只需一个资源：

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

借助 provider alias，单次 Terraform 运行即可操作两个地域：

```hcl
provider "alicloud" {
  alias  = "beijing"
  region = "cn-beijing"
}
```

对于主要无状态的研究型 Agent，你或许认为容灾不值得让存储成本翻倍。但若涉及客户对话历史且法律要求持久化，则跨地域复制必不可少。

### 证明副本真的能用——每月演练

从未验证过的副本，本质上等于没有备份。下面这个脚本能提前 30 天发现损坏的副本，而不是等到灾难真正发生：

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

该脚本仅消耗约两分钟计算资源，全程无需人工干预，从此副本健康状态不再靠猜测。建议将其与第二篇文章中的 drift check 一样，接入 GitHub Actions 定时任务。这种模式——定期自动化演练那些否则只能在最糟时刻才发现失效的关键路径——同样适用于 RDS 恢复、KMS 密钥轮转及所有故障转移流程。

> **Tip.** 我每月还会单独运行 `restore-drill.sh`，将随机 RDS 备份恢复到 `cn-shanghai-dr` 实例，并执行 schema 与 checksum 验证。这是我每月最有价值的 30 分钟运维投入。

## 通过 Terraform 进行 RDS 大版本升级

PostgreSQL 每两年发布一个大版本，旧版本随即停止支持。RDS 升级是一次真正的运维事件——伴随停机、可能失败，而 Terraform Provider 通过 `engine_version` 变更暴露的接口看似无害，实则暗藏风险。

我在 v15 → v16 升级中验证有效的流程如下：

**Step 1 — 动手前先打快照。**

```bash
aliyun rds CreateBackup \
  --DBInstanceId pgm-uf6abc123 \
  --BackupMethod Physical \
  --BackupType FullBackup
```

通过 `aliyun rds DescribeBackups` 等待完成。这是你的“救命稻草”。

**Step 2 — 克隆一个试验实例。**

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

`terraform apply` 会创建一个 v16 实例，并恢复 v15 的数据。QA 将 staging 流量切过去测试一周，全程零生产风险。

**Step 3 — 确认无误后，原地升级。**

```hcl
resource "alicloud_db_instance" "memory" {
  engine_version = "16.0"   # was "15.0"
  # everything else unchanged
}
```

`terraform plan` 显示 `~ engine_version: "15.0" -> "16.0"`，apply 触发原地升级。停机时间取决于数据库大小：小型实例通常不到 1 分钟，TB 级实例可能长达 30 分钟。升级仅能通过 Step 1 的快照回滚，因此绝不能跳过第一步。

**Step 4 — 清理试验实例。**

```bash
terraform state rm alicloud_db_instance.memory_v16_trial
# remove the HCL block
# next plan: 0 changes
```

随后在控制台手动删除试验实例。切勿在同一项目中执行 `terraform destroy`——这会触发依赖链，可能误删其他资源。

整个过程历时约两周，专注工作时间约 3 小时，且无计划外停机。使用 Terraform 的最大优势在于：试验实例的完整 HCL 配置已纳入 Git。六个月后升级 v17 时，这份经过团队评审的 playbook 已就绪于同一仓库。

> **Tip.** 升级前务必在 Agent 代码中测试连接字符串变更。某些 Postgres v16 的改动（如移除 `password_encryption = md5`）会破坏旧版客户端库。在正式切换前，让 Agent 对着试验实例完整运行一天。

## 真实事故：那个我 `terraform apply` 错 workspace 的夜晚

这次事故让我赔上了整个周六。之所以详述，是因为最终落地的是一套结构性防护机制，而非依赖“下次小心”的临时补救。

当时环境有三个 workspace：`dev`、`staging`、`prod`。`dev` RDS 是小规格 `pg.n2.medium.1c`，而 `prod` 是带 HA 的 `pg.x4.large.2c`。我在笔记本上切了个 feature 分支，运行 `terraform plan`，看到 “Plan: 2 to add, 1 to change, 0 to destroy” —— 看似干净，便执行了 `terraform apply`，然后去冲咖啡。

回来发现生产数据库没了。

根因：我此前会话中选了 `prod` workspace，之后忘记切换回来。我要应用的变更（仅修改一个 tag）在 dev 中无害，但在 prod 中，由于 `main` 分支刚合入一个无关的 Provider 升级，导致该参数被标记为 `force_new`，实际触发了实例重建。Provider 的 plan 输出本应更清晰——可惜没有。

出问题的 HCL 如下：

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

Provider 在 1.231 版本中判定该参数对某些引擎版本需强制新建。Plan 显示 `~ parameter`（看似原地变更），apply 却执行了重建。

恢复耗时 90 分钟（幸好自动备份较新）。事后复盘产出了四项结构性修复，过去两年再未复发。

### 修复 1 — 为生产环境有状态资源添加 `lifecycle { prevent_destroy = true }`

前文 RDS 配置中已包含此项。任何试图销毁 prod RDS 的 `terraform apply` 都会报错 `Resource has lifecycle.prevent_destroy set`。真要销毁，必须先在 HCL 中移除该行、提交 PR、获得审批——仅这一条就能彻底避免我的周六事故。

### 修复 2 — Shell 中的 workspace 提示

一个每次调用 `terraform` 都会高亮提醒的函数：

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

这 1 秒停顿足以打断“自动驾驶”模式。它存在于我的 `.zshrc` 中，此后我再未误操作生产环境。

### 修复 3 — 生产环境 apply 仅限 CI，禁止本地执行

更彻底的做法：撤销本地对生产 state 文件的写权限。只有 GitHub Actions Runner 的 RAM 角色拥有 `oss:PutObject` 权限（针对 prod state 前缀）。本地 `terraform plan` 仍可运行（只读），但 `apply` 会因 `AccessDenied` 失败。

```hcl
# Attached to the developer role
{
  "Effect": "Deny",
  "Action": "oss:PutObject",
  "Resource": "acs:oss:*:*:ck-tfstate-prod/agents/env:prod/*"
}
```

Developer 角色上的 Deny 策略优先级高于任何 Allow。开发者可 plan，仅 CI 可 apply，且 CI 执行需经 PR 审核。

### 修复 4 — 添加 `pre-apply` hook，显式确认销毁操作

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

任何涉及删除的 plan 都需显式设置 `DESTROY=yes` 环境变量。你不可能手误输入这个值。这是对 `prevent_destroy` 的双重保险——能捕获那些你忘记锁定的子模块中的销毁操作。

这四条必须全部启用。单条无法救我；四条合力，才让此类故障在结构上变得不可能。

## 把计算节点连到存储

第四篇文章中的 ECS 实例需要真正连通这些存储服务，涉及三部分：

1. **网络** — 已完成。VPC 模块输出的 `agent_runtime_sg_id` 已作为 `memory_rds_sg` 和 `vector_store_sg` 的入站规则源。
2. **凭证** — Agent 通过 STS 从 KMS Secrets Manager 读取 DB 密码：
   ```python
   from alibabacloud_kms20160120.client import Client as KmsClient
   resp = kms_client.get_secret_value(GetSecretValueRequest(secret_name="agents-prod-rds-admin"))
   db_password = resp.body.secret_data
   ```
3. **Endpoints** — 由 Terraform 输出：
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

Agent 通过环境变量读取这些值，而环境变量由 cloud-init 根据 Terraform outputs 自动注入。全程无硬编码 endpoint、无手动配置文件，密钥轮转也无需人工介入。

## 到底要花多少钱

开发环境、低流量场景下的月成本估算：

- RDS PostgreSQL (`pg.n2.medium.1c`, 100 GB ESSD)：约 ¥350/月。
- OpenSearch vector（最小规格）：约 ¥800/月。
- OSS（10 GB Standard，启用生命周期）：约 ¥1.5/月 + 流量费。
- KMS（第三篇已涵盖）：约 ¥10/月。

开发环境存储层总计约 ¥1200/月。生产环境若启用 RDS 高可用、更大规格 OpenSearch、更多 OSS 数据及跨地域副本，月成本将在 ¥3000–5000 区间。此时成本压力才真正显现——别等到月底对账单才惊觉，第七篇将教你如何提前监控并设置告警。

## 下一步计划

第六篇将构建 LLM Gateway，它位于第四篇配置的计算层与本篇搭建的存储层之前，负责 API 密钥管理、配额控制及各 Agent 的成本分摊。完成第六篇后，你将拥有一个完整的可运行 Agent 栈——最后两篇则在其上叠加可观测性与成本控制能力。
