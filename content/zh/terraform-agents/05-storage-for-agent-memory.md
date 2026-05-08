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

Agent 的"记忆"是大多数教程一笔带过的部分。"embedding 丢 Pinecone，会话进 Postgres，截图传 S3"——讲得轻巧。在阿里云上这三种存储都有托管服务，而能不能用 Terraform 把它们正确地建出来，就是"记忆好用"和"凌晨四点磁盘满了，丢了三周对话历史"之间的差距。

本文覆盖三层存储、对应的 Terraform、那些枯燥但救命的备份与灾备配置、大版本升级的完整流程，以及那个让我赔上整个周六、之后改变了我所有运维习惯的事故。

## 三层记忆模型

![Agent 的三种记忆对应到三个阿里云服务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig1_memory_three_layers.png)

心智模型：

- **短期 / 会话记忆**——Agent 当前运行和最近几次运行的痕迹：对话轮次、工具调用、中间状态。schema 稳定、低延迟、需要事务，归关系型数据库。
- **长期 / 语义记忆**——文档、历史输出、召回库的 embedding。需要词项+向量混合检索，归向量存储。
- **产物 / 大文件**——生成的图片、PDF、截图、运行快照。体积大、写多读少，归对象存储。

千万别把三层混在一起。我见过一个团队把 50 GB 生成 PDF 直接塞进 Postgres，理由是"它有 `bytea` 列"。结果成本是 OSS 的十倍，查询延迟惨不忍睹，备份动辄几小时。每一层都有专门擅长它的服务，选对了账单才不会失控。

## 第 1 层：关系型，RDS for PostgreSQL

会话状态——对话轮次、工具调用 trace、用户身份——必须用真正的 RDBMS。我默认 PostgreSQL；MySQL 也行，看团队偏好。需要横向扩展时再上 PolarDB。

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

值得逐行说明的几个点：

- 密码出生即在 KMS Secrets Manager 里。`random_password` 生成，写进 `alicloud_kms_secret`，Agent 启动时通过 STS 取。明文只在 Terraform 内存里短暂存在，下游通过 `secret_id` 引用而非引用值，因此不会落进 tfstate。
- `encryption_key` 把磁盘绑到 `memory` CMK。静态加密，零额外成本。
- `backup_period` + `retention_period` 一周三次自动备份，prod 留 30 天、dev 留 7 天。RDS 备份存在 OSS 上，bucket 不用你管。
- `zone_id_slave_a` 在 prod 里建第二可用区热备。30 秒内 failover。代价是 2 倍——prod 值得，dev 过头。
- `deletion_protection` 加上 `lifecycle.prevent_destroy` 在 prod 里同时挡住 `terraform destroy` 和 provider 触发的强制重建。为什么两个都要？后面那次事故复盘里会讲，简短版本是：其中一个曾经救了我的周六。

> **提示。** 当 sessions 表过 ~10M 行、或者需要无停机读副本时，再上 PolarDB。RDS → PolarDB 迁移文档完善，Terraform 两边都支持。但别一开始就上——小规模时 RDS 更简单更便宜。

关系层是骨架。下一层才是让 Agent 显得"聪明"的部分。

## 第 2 层：向量存储

阿里云上向量层有两个像样的选择：

1. OpenSearch 向量检索版——托管、Lucene 底层、支持 HNSW 和 IVF，按 QPS 配额计费。
2. PolarDB 或 RDS PostgreSQL 加 `pgvector`——和关系数据共驻，无新增基础设施，过 ~100 万向量后变慢。

原型之后我都选 OpenSearch。成本是真的（最小规格 ~¥800/月），但开箱即用的词项+向量混合检索是召回场景该有的形状——纯向量相似度在太多真实查询里会输给 BM25，没法忽视。

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
  description = "Agent 的长期语义记忆"
}
```

App group 是 OpenSearch 里挂索引的逻辑单元。索引 schema 通过控制台或 SDK 创建——`alicloud_opensearch_app` 资源是有的，但 schema 这层属于运维操作，不是 IaC。embedding 维度（OpenAI `text-embedding-3-small` 是 1536，阿里 bge-m3 是 1024）在索引设置里钉死，永远别改；重建 1000 万向量的索引是好几天的活。

如果走 pgvector 路线，在 RDS 的 database 创建里加：

```hcl
resource "alicloud_db_database" "vectors" {
  instance_id   = alicloud_db_instance.memory.id
  name          = "agent_vectors"
  character_set = "UTF8"
}

# pgvector 扩展通过迁移工具创建，不是 Terraform：
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE embeddings (id bigserial primary key, vec vector(1536), meta jsonb);
# CREATE INDEX embeddings_vec_idx ON embeddings USING hnsw (vec vector_cosine_ops);
```

Terraform 只管数据库；表结构是应用代码（Alembic、Flyway、sqlx-migrate 任挑一个）。别用 Terraform 管表结构，那条路只有疯掉一种结局，我有亲历的伤痕作证。

## 第 3 层：对象存储

OSS 装产物：生成的图片、PDF、截图、运行 trace 的 tar 包、微调出来的 checkpoint。Agent 栈的 bucket 配置：

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

三个值得展开的点。

### Bucket 名称全局唯一

OSS bucket 名跨所有阿里云用户唯一——和 S3 一样。`random_id` 后缀是为了避开新手必踩的"名字被占用"plan 报错。Bucket 一旦创建，名字就稳定了。

### 生命周期分层

`lifecycle_rule` 是 OSS 成本的最大杠杆：

![OSS 生命周期管理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig2_oss_lifecycle.png)

- 标准存储（0–30 天，~¥0.12/GB/月）——默认写入。
- 低频访问（30–90 天，~¥0.08/GB/月）——更便宜，每 GB 取回 ~¥0.0125。
- 归档（90–365 天，~¥0.033/GB/月）——分钟到小时级取回。
- 冷归档（365 天以上，~¥0.015/GB/月）——小时级取回，最便宜。

对 Agent 产物，这条规则的意思：30 天热存储，2 个月低频，9 个月归档，1 年冷归档，2 年删除。1 TB 的产物语料，全标准是 ¥1500/月，分层后是 ~¥250/月。HCL 里写一次，一年省五位数。唯一的坑是 IA 的最小存储期（30 天）和 Archive 的取回延迟——别把热数据塞进冷层就行。

### 版本控制

`versioning { status = "Enabled" }` 保留每个对象的所有版本。Agent 覆盖 `artifacts/run-123/output.pdf` 时，老版本不会真删，只是换了 version ID。两个理由：

1. 数据恢复。Bug 把 5 万个对象覆盖成垃圾？写个脚本回滚。
2. 防篡改。配合 WORM（Write-Once-Read-Many）策略，合规白送。

历史版本会持续累积，所以上面 lifecycle 里的 `noncurrent_version_expiration` 设了 180 天清理。不加这条，存储账单半年悄悄翻一倍。

## 备份、灾备，以及证明它们真能用

Terraform 管的备份拓扑大致是这样：

![备份不是可选项，而是预算内的必要投入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/terraform-agents/05-storage-for-agent-memory/fig3_backup_topology.png)

- RDS：自带自动备份（前面 HCL 已写）。
- OSS：版本控制 + 跨区复制做灾备。
- OpenSearch：通过 `alicloud_opensearch_*` 快照资源备到 OSS。

OSS 跨区复制就一个资源：

```hcl
resource "alicloud_oss_bucket" "artifacts_dr" {
  provider = alicloud.beijing       # 第二 region 的 provider 别名
  bucket   = "${alicloud_oss_bucket.artifacts.bucket}-dr"
  acl      = "private"

  versioning {
    status = "Enabled"
  }

  server_side_encryption_rule {
    sse_algorithm = "AES256"        # KMS 密钥按 region 隔离，灾备桶用 AES256 更省事
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

provider 别名让一次 Terraform 执行同时操作两个 region：

```hcl
provider "alicloud" {
  alias  = "beijing"
  region = "cn-beijing"
}
```

无状态的研发型 Agent，灾备让存储成本翻倍可能不划算。但面向客户、对话历史合规要求长期保存的场景，灾备是硬约束。

### 证明副本能用——每月演练

从未恢复过的副本就是不存在的备份。下面这个脚本能让"副本坏了 30 天才发现"变成"演练当天就发现"：

```bash
#!/bin/bash
# scripts/dr-drill.sh —— 每月 1 号 CI cron 跑一次
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

2 分钟算力、零人工，副本健康从此不再是信念问题。挂到第二篇里那个 drift check 的同一个 GitHub Actions cron 上。这种模式——把"否则会在最坏时间发现是坏的"东西周期性自动演练一遍——对 RDS 恢复、KMS 密钥轮换、每条 failover 路径都适用。

> **提示。** 我还有个 `restore-drill.sh` 每月跑一次，随机抽一个 RDS 备份恢复到 `cn-shanghai-dr` 实例做 schema/checksum 校验。这是我每月最有价值的 30 分钟。

## 用 Terraform 做 RDS 大版本升级

Postgres 每两年发一个大版本，旧版本随之 EOL。RDS for PostgreSQL 的大版本升级是实打实的运维事件——会有停机、可能失败，而 Terraform provider 把它暴露成"看起来无害实则危险"的 `engine_version` 改动。

我在 v15 → v16 升级里跑通的流程：

**第 1 步：动手前先备份。**

```bash
aliyun rds CreateBackup \
  --DBInstanceId pgm-uf6abc123 \
  --BackupMethod Physical \
  --BackupType FullBackup
```

通过 `aliyun rds DescribeBackups` 等完成。这是你的"救命按钮"。

**第 2 步：克隆一个兄弟实例做试运行。**

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

`terraform apply` 会基于 v15 备份拉起一个 v16 实例。QA 把 staging 流量指过去跑一周。生产零风险。

**第 3 步：确认无误后原地升级。**

```hcl
resource "alicloud_db_instance" "memory" {
  engine_version = "16.0"   # 原本是 "15.0"
  # 其余不变
}
```

`terraform plan` 显示 `~ engine_version: "15.0" -> "16.0"`，apply 触发原地升级。停机时间看库大小——小库 1 分钟内，多 TB 的库可能 30 分钟。升级回滚只能靠第 1 步的快照，别跳过第 1 步。

**第 4 步：拆掉试运行实例。**

```bash
terraform state rm alicloud_db_instance.memory_v16_trial
# 删 HCL 配置块
# 下次 plan：0 changes
```

然后通过控制台或 API 删试运行实例。别在同一个项目里 `terraform destroy`，那会顺着依赖走，可能误伤兄弟资源。

整个流程 2 周日历时间、3 小时聚焦工作、零计划外停机。用 Terraform 做的好处是试运行实例的 HCL 留在 git 里——半年后升 v17，playbook 已经在仓库里、被同一拨人 review 过了。

> **提示。** 升级前在 Agent 代码里测一遍连接字符串变更。Postgres v16 有些改动（比如移除 `password_encryption = md5`）会打挂老客户端库。先让 Agent 在试运行实例上跑一整天再切。

## 那个 apply 错 workspace 的夜晚

这件事赔上了我一个周六。值得讲是因为修复方法是结构性的，不是"下次小心点"。

背景：三个 workspace——`dev`、`staging`、`prod`。dev 的 RDS 是小规格 `pg.n2.medium.1c`，prod 是带 HA 的 `pg.x4.large.2c`。我在笔记本上干活，切了个 feature 分支，跑 `terraform plan` 检查改动。看到"Plan: 2 to add, 1 to change, 0 to destroy"——挺干净，跑了 `terraform apply`，起身去泡咖啡。

回来发现 prod 数据库被销毁了。

根因：之前一次会话我选了 prod workspace，后来没切回去。这次的改动（一个 tag 微调）在 dev 是无害的。在 prod，看起来"1 to change"实际是强制重建，因为 main 上落了一次和本次无关的 provider 升级，要求 RDS 重建。Provider 的 plan 输出本该说清楚——它没有。

肇事的 HCL：

```hcl
# PR 里看似无害的改动
resource "alicloud_db_instance" "memory" {
  # ... 其余不变 ...
  parameter {
    name  = "log_min_duration_statement"
    value = "100"   # 原值 "1000"，无害的调优
  }
}
```

provider 在新升的 1.231 版本里，把这个参数对某些引擎版本标成了 `force_new`。Plan 显示 `~ parameter`——看起来原地修改——apply 干的是重建。

从自动备份恢复花了 90 分钟（幸好备份够新）。事后复盘出了四条结构性改进，两年内零复发。

### 修复 1：prod 有状态资源加 `lifecycle { prevent_destroy = true }`

前面 RDS 块里已经加了。现在任何会销毁 prod RDS 的 `terraform apply` 都会报错：`Resource has lifecycle.prevent_destroy set`。要真销毁得先去 HCL 里删掉这行、提 PR、过审、然后 destroy 才被允许。仅这一行，就能挡掉那次周六的事故。

### 修复 2：Shell 里提示当前 workspace

每次 `terraform` 调用都吼一嗓子的函数：

```bash
function terraform() {
  local ws=$(/usr/local/bin/terraform workspace show 2>/dev/null)
  if [[ "$ws" == "prod" ]]; then
    echo -e "\033[1;31m================================\033[0m"
    echo -e "\033[1;31m WARNING: 当前 workspace = prod\033[0m"
    echo -e "\033[1;31m================================\033[0m"
    read -p "确定继续？[y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && return 1
  fi
  /usr/local/bin/terraform "$@"
}
```

这 1 秒的暂停足够打破"自动驾驶"。塞进 `.zshrc`，从此再没误触过 prod。

### 修复 3：prod apply 只走 CI，绝不在笔记本上

更彻底的版本：直接吊销笔记本对 prod state 文件的写权限。只有 GitHub Actions runner 持有的 RAM 角色对 prod state 前缀有 `oss:PutObject`。本地 `terraform plan` 能跑（只读），本地 `apply` 直接 `AccessDenied`。

```hcl
# 开发者角色绑定的策略
{
  "Effect": "Deny",
  "Action": "oss:PutObject",
  "Resource": "acs:oss:*:*:ck-tfstate-prod/agents/env:prod/*"
}
```

开发者角色上的 Deny 优先级高于任何 Allow。开发者能 plan，只有 CI 能 apply，CI 又被 PR review 卡住。

### 修复 4：`pre-apply` 钩子汇总销毁动作

```bash
# .git/hooks/pre-commit（或者 tflint 自定义规则）
plan_file=$1
n_destroy=$(terraform show -json "$plan_file" | jq '[.resource_changes[] | select(.change.actions[] == "delete")] | length')
if [[ "$n_destroy" -gt 0 ]]; then
  echo "本次 plan 将销毁 $n_destroy 个资源："
  terraform show -json "$plan_file" | jq -r '.resource_changes[] | select(.change.actions[] == "delete") | .address'
  echo "请用 DESTROY=yes terraform apply 显式确认"
  [[ "$DESTROY" != "yes" ]] && exit 1
fi
```

任何带销毁的 plan 都需要环境变量里显式给 `DESTROY=yes`。这个不可能手滑。它是 `prevent_destroy` 的腰带——专门兜底子模块里漏锁的情况。

四条全要。任何一条单独都救不了那次事故；四条合在一起让这种失败模式在结构上不可能发生。

## 把计算和存储连起来

第四篇里的 ECS 实例需要真的能访问到这些存储。三件事：

1. 网络——已经搞定。VPC 模块里的 `agent_runtime_sg_id` 是 `memory_rds_sg` 和 `vector_store_sg` 入站规则的 source。
2. 凭证——Agent 启动时通过 STS 从 KMS Secrets Manager 取 DB 密码：
   ```python
   from alibabacloud_kms20160120.client import Client as KmsClient
   resp = kms_client.get_secret_value(GetSecretValueRequest(secret_name="agents-prod-rds-admin"))
   db_password = resp.body.secret_data
   ```
3. 端点——Terraform 输出：
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

Agent 从 cloud-init 注入的环境变量里读这些值，环境变量来自 Terraform output。无硬编码端点、无手动配置文件、轮换时无人工介入。

## 成本

每月，dev workspace，低流量：

- RDS PostgreSQL（`pg.n2.medium.1c`，100 GB ESSD）：~¥350/月。
- OpenSearch 向量版（最小规格）：~¥800/月。
- OSS（10 GB 标准存储，开生命周期）：~¥1.5/月 + 流量。
- KMS（第三篇覆盖）：~¥10/月。

dev 存储层大约 ¥1200/月。prod 上 HA RDS、更大的 OpenSearch、更多 OSS、再加跨区副本，¥3000–5000/月。这是成本压力开始变真的地方——第七篇讲怎么追踪和告警，避免它在月度账单评审上突袭你。

## 下一篇

第六篇在第四篇的计算和本篇的存储之上搭 LLM 网关：API key 放哪、配额怎么拦、按 Agent 怎么归因成本。第六篇之后你就有一个完整的、能跑 Agent 的栈了——最后两篇把可观测性和成本控制盖上去。
