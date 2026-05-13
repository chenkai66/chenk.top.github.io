---
title: "阿里云全栈实战（五）：RDS 与 PolarDB 数据基石"
date: 2026-05-02 09:00:00
tags:
  - Alibaba Cloud
  - RDS
  - PolarDB
  - MySQL
  - Database
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 5
description: "RDS MySQL 与 PolarDB 对比：何时使用哪个。实例规格、读副本、代理端点、备份/恢复、监控、慢查询分析。构建高可用性和读扩展的生产数据库。"
disableNunjucks: true
translationKey: "aliyun-fullstack-5"
---
我在 ECS 上自建的 MySQL 只撑了四个月。流量高峰期的一次磁盘 I/O 飙升直接让整个服务宕机：InnoDB buffer pool 和 OS page cache 争抢内存，binary log 写满系统盘的速度远超 cron 任务的清理能力，而那台所谓“备份”实例的单线程复制竟已落后九小时。凌晨三点，我只能靠扩容磁盘勉强救火；结果两周后，同样的故障再次上演。那一刻我才真正明白托管数据库存在的意义——不是因为我不会部署或运维 MySQL，而是我不想在凌晨三点被报警叫醒，只因 MySQL 判定 relay log 损坏，而唯一的修复方式竟是依赖一份一致性无法保证的冷备份来重建副本。

本文将深入阿里云的数据库层：RDS 用于托管关系型数据库，PolarDB 则是 RDS 触及瓶颈时的升级方案，同时涵盖规格选型、复制、备份、监控与安全等关键运维实践。该数据库所在的 VPC 已在 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 中搭建完成。若想了解如何通过 Terraform 自动化部署数据库，请参考 [Terraform Part 5](/zh/terraform-agents/05-storage-for-agent-memory/)。

## 为什么选择托管数据库？

在裸 ECS 实例上运行数据库确实赋予你完全的控制权：你可以自由选择 MySQL 版本、精细调整每一个 `my.cnf` 参数、安装任意插件，甚至以 root 身份直接对 mysqld 进程执行 `perf` 或 `strace` 调试。这种控制力真实存在，偶尔也确实有用。

但这也意味着你签下了一份沉重的“运维契约”：

- **操作系统补丁**：内核安全更新需要重启——你是否已配置好故障转移？
- **备份管理**：使用 Percona XtraBackup 执行物理备份、每月测试恢复流程、管理备份存储空间、通过归档 binary log 实现时间点恢复（PITR）。
- **高可用架构**：配置半同步复制与 GTID、借助 MHA 或 ProxySQL 等 orchestrator 实现自动故障切换、管理 VIP、防止脑裂。
- **性能监控**：分析慢查询、监控 InnoDB buffer pool 命中率、追踪复制延迟、预防连接池耗尽、排查锁等待超时。
- **横向扩展**：添加只读副本、配置 ProxySQL 实现读写分离、管理连接路由策略。
- **磁盘管理**：选择文件系统（XFS vs ext4）、调优 I/O 调度器、规划 IOPS 配额、执行在线磁盘扩容。
- **安全加固**：配置 SSL/TLS、启用审计日志、实施静态数据加密、定期轮换密钥。

而托管数据库正是为解决这些问题而生。你虽失去了 OS root 权限和随意安装 MySQL 插件的能力（例如自定义 UDF 或实验 Group Replication 拓扑），但换来的是自动备份、一键高可用故障切换、内置监控，以及点击即可创建的只读副本。更重要的是，你再也不用因磁盘爆满而在深夜被警报惊醒。

对于 95% 的生产工作负载而言，这一权衡显然利大于弊。

### 阿里云数据库家族全景

在深入 RDS 之前，先一览阿里云数据库服务的完整版图：

| 服务 | 类型 | 引擎 | AWS 对应服务 | 适用场景 |
|---|---|---|---|---|
| **RDS** | 托管关系型 | MySQL, PostgreSQL, SQL Server, MariaDB | Amazon RDS | 标准 OLTP，绝大多数 Web 应用 |
| **PolarDB** | 云原生关系型 | MySQL 兼容, PostgreSQL 兼容 | Amazon Aurora | 高读吞吐、弹性伸缩、大型数据库 |
| **Lindorm** | 多模型 | 宽列, 时间序列, 搜索 | DynamoDB + Timestream | IoT 数据、时序指标、大规模 KV 存储 |
| **Tair** | 托管 Redis | Redis 兼容 | Amazon ElastiCache | 缓存、会话存储、限流 |
| **AnalyticDB (ADB)** | 分析型 | MySQL 兼容, PostgreSQL 兼容 | Amazon Redshift | OLAP、实时分析、数据仓库 |
| **PolarDB-X** | 分布式关系型 | MySQL 兼容 | Aurora Limitless / CockroachDB | 水平分片、分布式事务 |
| **OceanBase** | 分布式关系型 | MySQL/Oracle 兼容 | Spanner-like | 金融级应用、跨可用区强一致性 |
| **MongoDB** | 文档存储 | MongoDB 兼容 | Amazon DocumentDB | 文档型负载、灵活 Schema |

本文聚焦于 RDS（主力军）与 PolarDB（RDS 的升级路径），这两者已覆盖阿里云上 90% 的生产数据库需求。

![阿里云数据库服务选型决策图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_db_family_map.png)

*上图清晰展示了各服务的定位——本文内容主要集中在 OLTP 分支。*

## RDS MySQL 深度解析

RDS MySQL 是阿里云最广泛使用的数据库服务。它提供完全托管的 MySQL 实例，内置自动备份、补丁更新、监控告警与高可用能力。你只需通过标准的 MySQL 客户端连接其兼容端点：`mysql -h <endpoint> -u <user> -p`。

### 架构设计

RDS MySQL 高可用（HA）实例在同一地域内以主备对（primary-standby pair）形式运行。主节点处理全部读写请求，备节点通过半同步复制实时同步变更。一旦主节点故障，备节点将自动晋升为主，整个故障切换过程通常在 30 秒内完成，且对应用透明——DNS 端点保持不变，仅连接会短暂中断。

值得注意的是，该备节点不可读，其唯一作用就是保障高可用。若需读扩展能力，必须额外添加只读副本（read replicas），这部分将在后文详述。

![RDS 高可用故障切换时间线（约 30 秒）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_ha_failover_sequence.png)

*上述故障切换流程完全由 RDS 托管——你的应用仅感知到一次短暂连接中断，随后即可通过原 DNS 地址重连。*

### 版本选择

RDS MySQL 提供三种版本：

| 版本 | 节点数量 | 高可用故障转移 | 适用场景 | 成本 |
|---|---|---|---|---|
| **基础版** | 1（仅主节点） | 无 | 开发/测试、非关键业务 | 最低 |
| **高可用版 (HA)** | 2（主 + 备） | 是（约 30 秒） | 生产环境 | 标准 |
| **企业版 (集群)** | 3（主 + 2 备） | 是（<30 秒，零数据丢失） | 金融级、关键任务系统 | 最高 |

基础版无备用节点。一旦主节点故障，阿里云需从最新备份恢复，恢复时间从几分钟到数小时不等，具体取决于数据库规模。**切勿在生产环境中使用基础版。**

高可用版是大多数生产场景的首选，其同地域备节点通过半同步复制实现接近零的 RPO（恢复点目标）和约 30 秒的 RTO（恢复时间目标）。

企业版则引入第三个节点，基于 Paxos 协议达成共识，确保故障切换过程中零数据丢失。这对金融系统至关重要——哪怕丢失一笔交易都是不可接受的。

### 存储类型对比

| 存储类型 | IOPS 范围 | 延迟 | 最大容量 | 最佳适用场景 |
|---|---|---|---|---|
| **本地 SSD** | 最高 240,000 | 最低 (<0.1ms) | 3 TB | 极致性能、中小型数据库 |
| **ESSD PL1** | 最高 50,000 | 0.1–0.3ms | 32 TB | 通用生产环境 |
| **ESSD PL2** | 最高 100,000 | 0.1–0.3ms | 32 TB | 高 IOPS 负载 |
| **ESSD PL3** | 最高 1,000,000 | 0.1–0.3ms | 32 TB | 极端 IOPS、大型 OLTP |

本地 SSD 因物理直连宿主机，延迟最低，但容量上限仅为 3 TB。若数据库规模可能超过此限制，建议直接选用 ESSD。

ESSD（增强型 SSD）是网络附加块存储，支持在线无感扩容，最大容量达 32 TB，足以满足绝大多数场景。ESSD PL1 是默认选项，适用于大多数工作负载。仅当 CloudMonitor 显示 IOPS 持续逼近 PL1 上限时，才需考虑升级至 PL2 或 PL3。

> **实用建议**：起步阶段选择 ESSD PL1，并在 CloudMonitor 中持续观察 `IOPSUsage` 指标一周。若持续使用率超过 PL1 限制的 70%，再考虑升级；若始终低于 20%，说明你为冗余容量付费了——不过 PL0 与 PL1 价差微小，这点“保险费”通常值得投入。

### 连接方式

RDS MySQL 提供三种连接端点：

1. **内网端点**：仅限同一 VPC 内访问，供应用服务器使用，格式如 `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`。
2. **公网端点**：默认禁用，仅建议在远程管理时临时开启，并务必通过 IP 白名单严格限制来源。**切勿让生产应用通过公网端点连接数据库。**
3. **数据库代理端点**：提供读写分离、连接池及短连接优化功能，是使用只读副本时的推荐接入点。

## PolarDB：当 RDS 不够用时

PolarDB 是阿里云的云原生数据库。如果说 RDS 是“托管的 MySQL 实例”，那么 PolarDB 则是一种底层架构迥异、仅在协议层面兼容 MySQL 的全新系统。

![RDS 与 PolarDB 功能对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_rds_vs_polardb.png)

### PolarDB 与 RDS 的核心差异

关键区别在于 **计算与存储分离**。RDS 中，每个实例（主、备、只读）都拥有独立的数据副本；而 PolarDB 的所有计算节点共享同一分布式存储层。

这一架构变革带来诸多优势：

- **只读副本分钟级创建**：RDS 需全量拷贝数据，耗时数小时；PolarDB 仅需启动新计算节点挂载共享存储，无需数据复制。
- **存储自动伸缩**：PolarDB 存储随数据增长自动扩容，上限达 128 TB，彻底告别手动容量规划，也无需再担心凌晨三点因磁盘满载被叫醒。
- **最多支持 15 个只读副本**：RDS 限制为 5 个，而 PolarDB 支持 15 个，且因共享存储，复制延迟通常低于 10ms。
- **更快故障切换**：备节点已共享存储，无需数据同步，故障切换可在 10 秒内完成。

### RDS 与 PolarDB 功能对比

| 特性 | RDS MySQL HA | PolarDB MySQL |
|---|---|---|
| 架构 | 主备独立存储 | 共享存储，存算分离 |
| 最大存储 | 32 TB (ESSD) | 128 TB（自动扩容） |
| 只读副本数量 | 最多 5 个 | 最多 15 个 |
| 副本创建时间 | 数小时（需数据拷贝） | 数分钟（共享存储） |
| 复制延迟 | 秒级（异步/半同步） | <10ms（共享存储） |
| 故障切换时间 | ~30 秒 | <10 秒 |
| 存储扩容 | 手动 | 自动 |
| Serverless 模式 | 不支持 | 支持 |
| 协议兼容性 | MySQL 5.6/5.7/8.0 | 兼容 MySQL 5.6/5.7/8.0 |
| 价格（同等配置） | 1x | 1.2–1.5x |
| 最佳适用场景 | 标准 OLTP | 高读吞吐、大库、弹性负载 |

### PolarDB Serverless 模式

PolarDB 提供 Serverless 模式，计算资源可根据负载自动伸缩。你只需设定 PCU（PolarDB 计算单元）的最小与最大值：

- **最小 PCU: 1**：空闲时段自动缩容至 1 PCU，成本极低。
- **最大 PCU: 32**：流量高峰时自动扩容。
- **Scale-to-zero**：长期无活动时可完全暂停计算，仅收取存储费用。

该模式特别适合开发/测试环境，或流量波动剧烈的生产场景（如大促期间流量激增百倍的电商平台）。

### 何时选择 PolarDB 而非 RDS？

**选择 PolarDB 当且仅当**：
- 数据库规模将超过 10 TB 且持续增长
- 需要超过 5 个只读副本
- 要求复制延迟低于 10ms 以保障读一致性
- 负载具有显著突发性，可受益于 Serverless 自动伸缩
- 故障切换时间必须控制在 10 秒以内
- 希望彻底摆脱手动存储容量管理

**否则，优先选择 RDS**：
- 数据库小于 5 TB 且增长可预测
- 只需 3 个或更少只读副本
- 追求同等配置下的最低成本
- 依赖 PolarDB 尚未支持的特定 MySQL 功能
- 偏好更简单的运维模型

对大多数项目，建议从 RDS 起步，待遇到具体瓶颈后再迁移至 PolarDB。得益于 MySQL 协议兼容性，迁移通常只需执行数据导出与导入。

## 实例规格选型指南

选择合适的 RDS 实例规格是影响成本最关键的决策：规格过小会导致高负载下查询变慢，过大则造成资源闲置浪费。

![RDS 实例规格选择指南](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_sizing_guide.png)

### 核心选型指标

- **vCPU 数量**：决定并发查询处理能力，复杂 JOIN 或聚合操作密集型负载需更多 CPU。
- **内存大小**：直接决定 InnoDB buffer pool 容量。理想情况下，热点数据集应完全容纳于 buffer pool 中。
- **最大 IOPS**：磁盘每秒操作次数上限，事务密集型负载易触及此瓶颈。
- **最大连接数**：每种实例类型均有硬性限制，高并发场景必须在应用层实施连接池。

### 常见工作负载规格参考表

| 工作负载类型 | 实例类型 | vCPU | 内存 | 最大 IOPS | 最大连接数 | 月成本（约） |
|---|---|---|---|---|---|---|
| 小型博客 / CMS | rds.mysql.s2.large | 2 | 4 GiB | 2,000 | 300 | ~200 CNY |
| 中型 Web 应用 | rds.mysql.s3.large | 4 | 8 GiB | 5,000 | 600 | ~400 CNY |
| API 后端服务 | rds.mysql.m1.medium | 4 | 16 GiB | 7,000 | 4,000 | ~800 CNY |
| 重度 OLTP | rds.mysql.c1.xlarge | 8 | 32 GiB | 12,000 | 8,000 | ~1,600 CNY |
| 大型 SaaS 平台 | rds.mysql.c2.xlarge | 16 | 64 GiB | 14,000 | 16,000 | ~3,200 CNY |
| 数据密集型应用 | rds.mysql.st.h43 | 60 | 470 GiB | 120,000 | 48,000 | ~20,000 CNY |

### Buffer Pool 黄金法则

InnoDB 性能几乎完全取决于数据能否命中 buffer pool。buffer pool 是内存中的数据与索引页缓存：命中时查询耗时微秒级，未命中需读磁盘则达毫秒级——性能相差千倍。

计算公式如下：

```text
Buffer Pool Size ≈ 75% of instance memory
Working Set = frequently accessed data + all indexes
If Working Set ≤ Buffer Pool Size → fast
If Working Set > Buffer Pool Size → disk thrashing
```

检查当前 buffer pool 使用情况：

```sql
-- Buffer pool hit ratio (should be > 99%)
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests';
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_reads';

-- Calculate: hit_ratio = 1 - (reads / read_requests)
-- If hit_ratio < 0.99, your buffer pool is too small

-- Actual buffer pool size
SHOW GLOBAL VARIABLES LIKE 'innodb_buffer_pool_size';

-- Data + index size for each database
SELECT 
  table_schema AS db,
  ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
FROM information_schema.tables
GROUP BY table_schema
ORDER BY size_mb DESC;
```

若你的热点数据集为 12 GiB，16 GiB 内存的实例仅能分配约 12 GiB（75%）给 buffer pool，空间过于紧张。建议直接选用 32 GiB 内存实例，为未来增长预留缓冲。

### 连接数限制应对策略

每种实例均有最大连接数上限，超出即返回 `Too many connections` 错误——这是生产环境最常见的故障之一。

**正确解法并非盲目升级实例**，而是在应用层实施连接池：

```python
# SQLAlchemy connection pool example
from sqlalchemy import create_engine

engine = create_engine(
    "mysql+pymysql://appuser:password@rm-bp1xxx.mysql.rds.aliyuncs.com:3306/mydb",
    pool_size=20,          # Maintain 20 connections
    max_overflow=10,       # Allow 10 extra connections under load
    pool_timeout=30,       # Wait 30s for a connection before erroring
    pool_recycle=3600,     # Recycle connections every hour
    pool_pre_ping=True     # Verify connection is alive before using
)
```

一个配置合理的连接池（20–30 个连接）即可支撑数千并发请求。若未使用连接池，每个请求新建数据库连接，几百并发用户便足以耗尽连接配额。

## 创建 RDS 实例

### CLI 创建全流程

以下为创建生产级 RDS MySQL 实例的完整 CLI 流程，采用高可用版与 ESSD 存储，并部署于应用服务器所在 VPC：

```bash
# Create RDS MySQL 8.0 HA instance
aliyun rds CreateDBInstance \
  --RegionId cn-hangzhou \
  --Engine MySQL \
  --EngineVersion "8.0" \
  --DBInstanceClass rds.mysql.s3.large \
  --DBInstanceStorage 100 \
  --DBInstanceStorageType cloud_essd \
  --Category HighAvailability \
  --DBInstanceNetType Intranet \
  --VPCId vpc-bp1xxxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-h \
  --ZoneIdSlave1 cn-hangzhou-i \
  --SecurityIPList "10.0.10.0/24,10.0.11.0/24" \
  --PayType Postpaid \
  --DBInstanceDescription "Production MySQL" \
  --ConnectionMode Standard \
  --InstanceNetworkType VPC
```

关键参数解析：

- **Engine/EngineVersion**：选用 MySQL 8.0。新项目应避免使用已进入维护期的 5.7。
- **DBInstanceClass**：`rds.mysql.s3.large` 提供 4 vCPU 与 8 GiB 内存，适合中型 Web 应用。
- **DBInstanceStorage**：初始 100 GiB，后续可通过 ESSD 在线扩容。
- **Category**：`HighAvailability` 启用主备架构。
- **ZoneId/ZoneIdSlave1**：主节点置于可用区 H，备节点置于 I，实现跨可用区高可用。
- **SecurityIPList**：仅允许应用子网（10.0.10.0/24 与 10.0.11.0/24）访问。注意：此为 RDS IP 白名单，独立于安全组。
- **ConnectionMode**：`Standard` 表示直连模式；`Safe`（已废弃）应改用 Database Proxy。

等待实例就绪：

```bash
# Check instance status (wait for "Running")
aliyun rds DescribeDBInstanceAttribute \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=DBInstanceStatus,DBInstanceClass,Engine,EngineVersion
```

### 创建数据库与账号

```bash
# Create the application database
aliyun rds CreateDatabase \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --DBName myapp \
  --CharacterSetName utf8mb4 \
  --DBDescription "Main application database"

# Create a normal account (not superuser) for the application
aliyun rds CreateAccount \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --AccountName appuser \
  --AccountPassword "YourStr0ngP@ssw0rd" \
  --AccountType Normal \
  --AccountDescription "Application service account"

# Grant permissions
aliyun rds GrantAccountPrivilege \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --AccountName appuser \
  --DBName myapp \
  --AccountPrivilege ReadWrite
```

**切勿授予应用账号 `Super` 权限**。遵循最小权限原则，为应用创建 `Normal` 账号并仅授权特定数据库的 `ReadWrite` 权限。另设独立管理员账号用于 DBA 操作。

### 获取连接端点

```bash
# Get the internal connection string
aliyun rds DescribeDBInstanceNetInfo \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=ConnectionString,IPAddress,Port,DBInstanceNetType
```

内网端点格式如 `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`，将其配置至应用：

```bash
# Test connection from an ECS instance in the same VPC
mysql -h rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com \
  -u appuser -p \
  -e "SELECT VERSION(); SHOW DATABASES;"
```

### 配置实例参数

RDS 通过参数模板（parameter groups）管理 MySQL 配置，可单独修改参数或应用模板：

```bash
# View current parameters
aliyun rds DescribeParameters \
  --DBInstanceId rm-bp1xxxxxxxxx

# Modify key parameters for production
aliyun rds ModifyParameter \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --Parameters "innodb_buffer_pool_size:6442450944;slow_query_log:ON;long_query_time:1;max_connections:500;innodb_flush_log_at_trx_commit:1;sync_binlog:1"
```

生产环境关键参数建议：

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `innodb_buffer_pool_size` | 实例内存的 75% | 最大化内存数据访问效率 |
| `slow_query_log` | ON | 必须开启，用于性能问题排查 |
| `long_query_time` | 1（秒） | 记录执行超 1 秒的查询 |
| `innodb_flush_log_at_trx_commit` | 1 | 保证完整 ACID 持久性；仅当可接受数据丢失风险时设为 2 |
| `sync_binlog` | 1 | 每次提交同步 binlog，确保崩溃安全复制 |
| `max_connections` | 实际需求 + 20% 余量 | 避免盲目设为实例上限 |

## 只读副本与数据库代理

### 添加只读副本

![数据库代理读写分离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_proxy_architecture.png)

![只读副本 binlog 复制架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_replica_flow.png)

![RDS 高可用架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_ha_architecture.png)

只读副本是主实例的异步拷贝，专用于处理 `SELECT` 查询，从而卸载主库读压力。对读多写少的 Web 应用而言，这是最主要的扩展手段。

```bash
# Create a read replica
aliyun rds CreateReadOnlyDBInstance \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --RegionId cn-hangzhou \
  --ZoneId cn-hangzhou-i \
  --EngineVersion "8.0" \
  --DBInstanceClass rds.mysql.s2.large \
  --DBInstanceStorage 100 \
  --DBInstanceStorageType cloud_essd \
  --PayType Postpaid \
  --DBInstanceDescription "Read replica 1"
```

副本创建流程如下：
1. RDS 对主实例创建内部快照
2. 基于快照启动新实例
3. 自动配置 binary log 复制
4. 副本就绪（小库需数分钟，大库可能数小时）

检查复制状态：

```sql
-- On the read replica, check replication lag
SHOW SLAVE STATUS\G

-- Key fields to watch:
-- Seconds_Behind_Master: should be 0 or very low
-- Slave_IO_Running: Yes
-- Slave_SQL_Running: Yes
```

### 数据库代理（Database Proxy）

手动管理只读副本十分繁琐：应用需自行区分主库（写）与副本（读）端点并实现路由逻辑。数据库代理正是为此而生。

作为内置代理层，Database Proxy 提供：
- **自动读写分离**：`INSERT/UPDATE/DELETE` 路由至主库，`SELECT` 路由至副本
- **连接池**：降低数据库连接开销
- **单一接入点**：应用仅需连接一个地址，路由由代理透明处理

启用数据库代理：

```bash
# Enable Database Proxy (dedicated proxy for HA edition)
aliyun rds ModifyDBProxy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --ConfigDBProxyService true \
  --DBProxyInstanceType 2 \
  --DBProxyInstanceNum 2

# Get the proxy endpoint
aliyun rds DescribeDBProxy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=DBProxyConnectString,DBProxyConnectStringPort
```

配置读写分离规则：

```bash
# Enable read/write splitting with weight distribution
aliyun rds ModifyReadWriteSplittingConnection \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --ConnectionStringPrefix "myapp-proxy" \
  --DistributionType Standard \
  --MaxDelayTime 10 \
  --Weight '{"rm-bp1xxxxxxxxx":"0","rr-bp1yyyyyyyyy":"100"}'
```

权重配置含义：
- 主库 (`rm-bp1xxxxxxxxx`)：权重 0 —— 读请求不发往主库（写请求仍自动路由至此）
- 副本 (`rr-bp1yyyyyyyyy`)：权重 100 —— 所有读请求均发往副本

若副本复制延迟超过 `MaxDelayTime`（默认 10 秒），代理将自动将读请求切回主库，直至副本追平进度。

应用连接字符串更新为代理端点：

```python
# Before: direct connection to primary
engine = create_engine(
    "mysql+pymysql://appuser:pass@rm-bp1xxx.mysql.rds.aliyuncs.com:3306/mydb"
)

# After: connection through Database Proxy
engine = create_engine(
    "mysql+pymysql://appuser:pass@myapp-proxy.rwlb.rds.aliyuncs.com:3306/mydb"
)
```

**无需修改应用代码**，所有路由逻辑由代理透明处理。

## 备份与恢复

备份是唯一不可省略的防线。当遭遇部署失误、数据损坏或误执行 `DROP TABLE` 时，备份是你最后的救命稻草。

![备份时间线与时间点恢复](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_backup_timeline.png)

### 自动备份

RDS 按可配置计划执行自动备份，默认策略：
- **全量备份**：每日低峰期执行
- **Binlog 备份**：持续进行，支持时间点恢复（PITR）
- **保留周期**：7 天（可延长至 730 天）

配置备份策略：

```bash
# Set backup to run between 02:00-03:00 UTC, retain 14 days
aliyun rds ModifyBackupPolicy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --PreferredBackupTime "02:00Z-03:00Z" \
  --PreferredBackupPeriod "Monday,Wednesday,Friday,Sunday" \
  --BackupRetentionPeriod 14 \
  --EnableBackupLog true \
  --LogBackupRetentionPeriod 14
```

自动备份对性能影响极小，RDS 采用快照技术，即使大型数据库也能快速完成。

### 手动备份

在执行高风险操作前（如 Schema 变更、批量数据导入、版本升级），务必创建手动备份：

```bash
# Create a manual backup
aliyun rds CreateBackup \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --BackupMethod Physical \
  --BackupType FullBackup \
  --DBName myapp

# List available backups
aliyun rds DescribeBackups \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --StartTime "2026-05-01T00:00Z" \
  --EndTime "2026-05-16T23:59Z" \
  --output cols=BackupId,BackupStartTime,BackupSize,BackupStatus,BackupType
```

### 时间点恢复（PITR）

PITR 是最重要的备份功能，允许将数据库恢复至保留期内任意一秒。例如，若 `DELETE FROM users WHERE 1=1` 于 14:32:07 执行，可恢复至 14:32:06。

PITR 通过组合最近全量备份与 binlog 重放实现：

![PITR 按时间点恢复流程：完整快照 + 持续 binlog 回放](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_pitr_flow.png)

*PITR 始终新建实例进行恢复——源数据库绝不会被修改。*

```bash
# Restore to a specific point in time (creates a new instance)
aliyun rds CreateTempDBInstance \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --RestoreTime "2026-05-15T14:32:06Z" \
  --BackupId 12345

# Or clone to a new instance at a specific point in time
aliyun rds CloneDBInstance \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --RestoreTime "2026-05-15T14:32:06Z" \
  --PayType Postpaid \
  --DBInstanceClass rds.mysql.s3.large \
  --DBInstanceStorage 100 \
  --DBInstanceStorageType cloud_essd \
  --VPCId vpc-bp1xxxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxxx
```

PITR 恢复操作**总是创建新实例**，永不覆盖原实例。此设计允许你在新实例验证数据正确性后再决定是否切换。

### 跨地域备份

为应对区域性灾难，务必启用跨地域备份。该功能将备份复制至另一地域，确保即使主地域完全失效，数据依然安全：

```bash
# Enable cross-region backup
aliyun rds ModifyInstanceCrossBackupPolicy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --CrossBackupType 1 \
  --CrossBackupRegion cn-shanghai \
  --RetentType 1 \
  --Retention 30
```

跨地域备份虽增加成本（数据传输费 + 异地存储费），但这是防范区域性故障的唯一手段。**任何承载关键业务数据的生产库都应启用此功能。**

## 监控与性能优化

### 云监控核心指标

![RDS 监控仪表盘指标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_monitoring_metrics.png)

RDS 自动上报指标至 CloudMonitor，以下关键指标需重点监控：

![四个关键 RDS 指标仪表盘及对应告警阈值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_four_metrics.png)

*建议在 80% 阈值设置告警——若等到 100% 才响应，用户早已遭遇故障。*

| 指标 | 警告阈值 | 严重阈值 | 应对措施 |
|---|---|---|---|
| CPU 利用率 | >70% 持续 | >90% 持续 | 升级实例或优化查询 |
| IOPS 使用率 | >70% 限额 | >90% 限额 | 升级存储 PL 或优化 I/O |
| 连接使用率 | >70% 上限 | >90% 上限 | 引入连接池，调高 max_connections |
| 磁盘使用率 | >70% | >85% | 立即扩容存储 |
| 复制延迟 | >5 秒 | >30 秒 | 检查副本性能，降低写负载 |

设置告警规则：

```bash
# Alert when CPU > 80% for 5 minutes
aliyun cms PutResourceMetricRule \
  --RuleId "rds-cpu-high" \
  --RuleName "RDS CPU High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "CpuUsage" \
  --Resources '[{"instanceId":"rm-bp1xxxxxxxxx"}]' \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 3 \
  --Period 60

# Alert when disk usage > 80%
aliyun cms PutResourceMetricRule \
  --RuleId "rds-disk-high" \
  --RuleName "RDS Disk Usage High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "DiskUsage" \
  --Resources '[{"instanceId":"rm-bp1xxxxxxxxx"}]' \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 1 \
  --Period 300

# Alert when active connections > 80% of limit
aliyun cms PutResourceMetricRule \
  --RuleId "rds-conn-high" \
  --RuleName "RDS Connections High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "ConnectionUsage" \
  --Resources '[{"instanceId":"rm-bp1xxxxxxxxx"}]' \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 2 \
  --Period 60
```

### DAS：数据库自治服务

DAS 是阿里云为 RDS 提供的智能诊断工具，超越基础指标监控，提供：
- **自动 SQL 诊断**：识别慢查询并推荐索引优化
- **实时会话分析**：展示活跃会话、锁等待及阻塞查询
- **性能趋势分析**：追踪指标变化，预警性能退化
- **自动优化**：可自动应用索引建议（需手动开启）

DAS 默认启用，通过 RDS 控制台“自治服务”入口访问。

### 慢查询分析

慢查询是数据库性能问题的头号元凶。RDS 会记录所有超过 `long_query_time`（前文设为 1 秒）的查询。

![从 EXPLAIN 出发的慢查询排查决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_slow_query_flow.png)

*优化起点永远是 `EXPLAIN`——`type` 列通常能揭示 80% 的执行计划健康度。*

```sql
-- Find the slowest queries in the slow log
-- (Use DAS console for a visual interface, or query directly)

-- Check current slow queries from process list
SELECT * FROM information_schema.processlist
WHERE command != 'Sleep'
AND time > 5
ORDER BY time DESC;

-- Find tables without indexes being used in queries
SELECT 
  t.table_schema,
  t.table_name,
  t.table_rows,
  ROUND(t.data_length / 1024 / 1024, 2) AS data_mb
FROM information_schema.tables t
LEFT JOIN information_schema.statistics s 
  ON t.table_schema = s.table_schema 
  AND t.table_name = s.table_name
WHERE s.table_name IS NULL
AND t.table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
AND t.table_rows > 1000
ORDER BY t.table_rows DESC;

-- Check index usage statistics
SELECT 
  object_schema,
  object_name,
  index_name,
  count_star AS total_ops,
  count_read,
  count_write,
  count_fetch
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE object_schema = 'myapp'
ORDER BY count_star DESC;
```

下载慢查询日志进行离线分析：

```bash
# List slow query log files
aliyun rds DescribeSlowLogRecords \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --StartTime "2026-05-15T00:00Z" \
  --EndTime "2026-05-16T00:00Z" \
  --DBName myapp \
  --PageSize 30 \
  --PageNumber 1
```

### 索引优化建议

最高效的性能优化往往源于正确的索引。使用 `EXPLAIN` 分析查询执行计划：

```sql
-- Analyze a slow query
EXPLAIN SELECT u.name, o.order_id, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.email = 'user@example.com'
AND o.created_at > '2026-01-01';

-- If type shows "ALL" (full table scan), add an index:
ALTER TABLE users ADD INDEX idx_email (email);
ALTER TABLE orders ADD INDEX idx_user_created (user_id, created_at);

-- Re-run EXPLAIN to verify the index is used
-- type should show "ref" or "range" instead of "ALL"
```

## 安全加固

### IP 白名单

RDS IP 白名单是第一道防线，在认证前即在网络层限制可访问 IP。

```bash
# View current whitelist
aliyun rds DescribeDBInstanceIPArrayList \
  --DBInstanceId rm-bp1xxxxxxxxx

# Set whitelist to app tier subnets only
aliyun rds ModifySecurityIps \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --SecurityIps "10.0.10.0/24,10.0.11.0/24" \
  --DBInstanceIPArrayName "app-tier"

# Add a DBA jump host IP for administrative access
aliyun rds ModifySecurityIps \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --SecurityIps "10.0.1.50/32" \
  --DBInstanceIPArrayName "dba-access" \
  --ModifyMode Append
```

**严禁添加 `0.0.0.0/0`**！这相当于向全网开放数据库——若同时启用公网端点，等于将数据库暴露于互联网。即便密码强度足够，也会遭受暴力破解攻击。

### SSL 加密

启用 SSL 可加密应用与数据库间的传输数据：

```bash
# Enable SSL
aliyun rds ModifyDBInstanceSSL \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --ConnectionString "rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com" \
  --SSLEnabled 1

# Download the CA certificate
aliyun rds DescribeDBInstanceSSL \
  --DBInstanceId rm-bp1xxxxxxxxx
```

并在应用侧配置 SSL 连接：

```python
# SQLAlchemy with SSL
engine = create_engine(
    "mysql+pymysql://appuser:pass@rm-bp1xxx.mysql.rds.aliyuncs.com:3306/mydb",
    connect_args={
        "ssl": {
            "ca": "/path/to/ca-cert.pem"
        }
    }
)
```

### TDE（透明数据加密）

TDE 对磁盘上的静态数据进行加密，即使攻击者获取底层存储权限，无密钥也无法读取数据。

```bash
# Enable TDE
aliyun rds ModifyDBInstanceTDE \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --TDEStatus Enabled \
  --EncryptionKey "kms-key-id-from-kms"
```

TDE 采用 AES-256 加密，并集成 KMS（密钥管理服务）进行密钥管理。**注意：TDE 一旦启用无法关闭**，但性能开销通常低于 5%。

### SQL 审计

SQL Audit 记录所有执行的 SQL 语句，对合规审计、安全调查及故障排查至关重要。

```bash
# Enable SQL audit
aliyun rds ModifySQLCollectorPolicy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --SQLCollectorStatus Enable

# Query audit logs
aliyun rds DescribeSQLLogRecords \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --StartTime "2026-05-15T00:00Z" \
  --EndTime "2026-05-16T00:00Z" \
  --DBName myapp \
  --QueryKeywords "DROP,DELETE,TRUNCATE"
```

审计日志默认保留 30 天。如需长期留存，应导出至 SLS（日志服务）进行归档分析。

### ECS 到 RDS 的安全连接链路

从应用 ECS 安全连接 RDS 的完整防护体系：

![RDS 安全防御的五层同心结构：从 VPC 隔离到 TDE 加密](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_security_layers.png)

*纵深防御理念——攻击者突破一层防线后，仍需攻破后续四层才能接触真实数据。*

1. **VPC 隔离**：RDS 与 ECS 同属一 VPC，但分属不同交换机（数据层 vs 应用层）
2. **安全组策略**：ECS 安全组放行出站 3306；RDS 白名单仅允入应用层 CIDR
3. **SSL 加密**：传输数据全程加密
4. **最小权限账号**：应用使用仅具特定库 ReadWrite 权限的 Normal 账号
5. **禁用公网端点**：数据库完全不可从互联网访问

```bash
# Verify: from app-tier ECS, connection works
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p --ssl-ca=/path/to/ca.pem -e "SELECT 1"

# Verify: from web-tier ECS (different security group), connection fails
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p -e "SELECT 1"
# ERROR 2003 (HY000): Can't connect to MySQL server (connection refused / timeout)
```

## 解决方案：生产级高可用数据库架构

现在，我们从零构建一套完整的生产级数据库方案：RDS MySQL 高可用实例 + 1 个只读副本 + Database Proxy 读写分离 + 自动备份 + 监控告警。此为中大型 Web 应用的标准参考架构。

### 步骤 1：创建 RDS MySQL 高可用实例

```bash
# Create primary HA instance (cross-AZ)
RDS_ID=$(aliyun rds CreateDBInstance \
  --RegionId cn-hangzhou \
  --Engine MySQL \
  --EngineVersion "8.0" \
  --DBInstanceClass rds.mysql.s3.large \
  --DBInstanceStorage 200 \
  --DBInstanceStorageType cloud_essd \
  --Category HighAvailability \
  --DBInstanceNetType Intranet \
  --VPCId vpc-bp1xxxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-h \
  --ZoneIdSlave1 cn-hangzhou-i \
  --SecurityIPList "10.0.10.0/24,10.0.11.0/24" \
  --PayType Postpaid \
  --DBInstanceDescription "Production MySQL HA" \
  --InstanceNetworkType VPC \
  | jq -r '.DBInstanceId')

echo "RDS Instance: $RDS_ID"

# Wait for Running status
echo "Waiting for instance to be ready..."
while true; do
  STATUS=$(aliyun rds DescribeDBInstanceAttribute \
    --DBInstanceId $RDS_ID \
    | jq -r '.Items.DBInstanceAttribute[0].DBInstanceStatus')
  echo "Status: $STATUS"
  [ "$STATUS" = "Running" ] && break
  sleep 30
done
```

### 步骤 2：创建数据库与账号

```bash
# Create the application database
aliyun rds CreateDatabase \
  --DBInstanceId $RDS_ID \
  --DBName myapp \
  --CharacterSetName utf8mb4 \
  --DBDescription "Main application database"

# Create application account
aliyun rds CreateAccount \
  --DBInstanceId $RDS_ID \
  --AccountName appuser \
  --AccountPassword "Pr0d_Str0ng!P@ss2026" \
  --AccountType Normal \
  --AccountDescription "Application service account"

# Grant ReadWrite on the application database
aliyun rds GrantAccountPrivilege \
  --DBInstanceId $RDS_ID \
  --AccountName appuser \
  --DBName myapp \
  --AccountPrivilege ReadWrite

# Create a DBA admin account
aliyun rds CreateAccount \
  --DBInstanceId $RDS_ID \
  --AccountName dbadmin \
  --AccountPassword "DB@dm1n_Str0ng!2026" \
  --AccountType Super \
  --AccountDescription "DBA administrative account"
```

### 步骤 3：配置生产级参数

```bash
# Apply production-tuned parameters
aliyun rds ModifyParameter \
  --DBInstanceId $RDS_ID \
  --Parameters "slow_query_log:ON;long_query_time:1;innodb_flush_log_at_trx_commit:1;sync_binlog:1;max_connections:500;innodb_print_all_deadlocks:ON;log_queries_not_using_indexes:ON"
```

### 步骤 4：创建只读副本

```bash
# Create read replica in the alternate AZ
REPLICA_ID=$(aliyun rds CreateReadOnlyDBInstance \
  --DBInstanceId $RDS_ID \
  --RegionId cn-hangzhou \
  --ZoneId cn-hangzhou-i \
  --EngineVersion "8.0" \
  --DBInstanceClass rds.mysql.s2.large \
  --DBInstanceStorage 200 \
  --DBInstanceStorageType cloud_essd \
  --PayType Postpaid \
  --DBInstanceDescription "Read Replica 1" \
  | jq -r '.DBInstanceId')

echo "Read Replica: $REPLICA_ID"
```

### 步骤 5：启用 Database Proxy 读写分离

```bash
# Enable Database Proxy
aliyun rds ModifyDBProxy \
  --DBInstanceId $RDS_ID \
  --ConfigDBProxyService true \
  --DBProxyInstanceType 2 \
  --DBProxyInstanceNum 2

# Get the proxy endpoint
PROXY_ENDPOINT=$(aliyun rds DescribeDBProxy \
  --DBInstanceId $RDS_ID \
  | jq -r '.DBProxyConnectStringItems.DBProxyConnectStringItems[0].DBProxyConnectString')

echo "Proxy Endpoint: $PROXY_ENDPOINT"

# Configure read/write splitting weights
# Primary: weight 0 (writes only), Replica: weight 100 (all reads)
aliyun rds ModifyReadWriteSplittingConnection \
  --DBInstanceId $RDS_ID \
  --ConnectionStringPrefix "myapp-proxy" \
  --DistributionType Standard \
  --MaxDelayTime 10 \
  --Weight "{\"$RDS_ID\":\"0\",\"$REPLICA_ID\":\"100\"}"
```

### 步骤 6：配置备份策略

```bash
# Set backup policy: daily at 02:00 UTC, retain 14 days
aliyun rds ModifyBackupPolicy \
  --DBInstanceId $RDS_ID \
  --PreferredBackupTime "02:00Z-03:00Z" \
  --PreferredBackupPeriod "Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday" \
  --BackupRetentionPeriod 14 \
  --EnableBackupLog true \
  --LogBackupRetentionPeriod 14

# Enable cross-region backup to cn-shanghai
aliyun rds ModifyInstanceCrossBackupPolicy \
  --DBInstanceId $RDS_ID \
  --CrossBackupType 1 \
  --CrossBackupRegion cn-shanghai \
  --RetentType 1 \
  --Retention 30

# Enable SSL encryption
aliyun rds ModifyDBInstanceSSL \
  --DBInstanceId $RDS_ID \
  --ConnectionString "$PROXY_ENDPOINT" \
  --SSLEnabled 1
```

### 步骤 7：设置监控告警

```bash
# CPU alert
aliyun cms PutResourceMetricRule \
  --RuleId "prod-rds-cpu" \
  --RuleName "Prod RDS CPU High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "CpuUsage" \
  --Resources "[{\"instanceId\":\"$RDS_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 3 \
  --Period 60

# Disk usage alert
aliyun cms PutResourceMetricRule \
  --RuleId "prod-rds-disk" \
  --RuleName "Prod RDS Disk High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "DiskUsage" \
  --Resources "[{\"instanceId\":\"$RDS_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 1 \
  --Period 300

# Connection usage alert
aliyun cms PutResourceMetricRule \
  --RuleId "prod-rds-conn" \
  --RuleName "Prod RDS Connections High" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "ConnectionUsage" \
  --Resources "[{\"instanceId\":\"$RDS_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 80 \
  --Escalations.Critical.Times 2 \
  --Period 60

# Replication lag alert (on the replica)
aliyun cms PutResourceMetricRule \
  --RuleId "prod-rds-replag" \
  --RuleName "Prod RDS Replication Lag" \
  --Namespace "acs_rds_dashboard" \
  --MetricName "ReplicationLag" \
  --Resources "[{\"instanceId\":\"$REPLICA_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 30 \
  --Escalations.Critical.Times 2 \
  --Period 60
```

### 步骤 8：验证完整架构

```bash
# Get all endpoints
echo "=== Connection Info ==="
echo "Primary (direct): $(aliyun rds DescribeDBInstanceNetInfo \
  --DBInstanceId $RDS_ID \
  | jq -r '.DBInstanceNetInfos.DBInstanceNetInfo[] | select(.DBInstanceNetType=="Intranet") | .ConnectionString')"

echo "Replica (direct): $(aliyun rds DescribeDBInstanceNetInfo \
  --DBInstanceId $REPLICA_ID \
  | jq -r '.DBInstanceNetInfos.DBInstanceNetInfo[] | select(.DBInstanceNetType=="Intranet") | .ConnectionString')"

echo "Proxy (r/w split): $PROXY_ENDPOINT"
```

从应用层 ECS（同 VPC）测试端到端链路：

```bash
# Connect through the proxy endpoint
mysql -h myapp-proxy.rwlb.rds.aliyuncs.com -u appuser -p --ssl-ca=/path/to/ca.pem << 'SQL'

-- Verify we can write
CREATE TABLE IF NOT EXISTS healthcheck (
  id INT AUTO_INCREMENT PRIMARY KEY,
  checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  hostname VARCHAR(255)
);

INSERT INTO healthcheck (hostname) VALUES (@@hostname);

-- Verify read/write splitting is working
-- This SELECT should go to the replica via the proxy
SELECT * FROM healthcheck ORDER BY checked_at DESC LIMIT 5;

-- Check the current server (shows which node handled the query)
SELECT @@hostname AS current_server, @@read_only AS is_readonly;

-- Verify backup status
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit';
SHOW VARIABLES LIKE 'sync_binlog';

-- Check replication health
SHOW SLAVE HOSTS;

SQL
```

### 架构总结

最终构建的系统包含：

```text
                    ┌────────────────────────┐
                    │   Application (ECS)    │
                    │   app tier VPC subnet  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Database Proxy       │
                    │   (read/write split)   │
                    │   myapp-proxy.rwlb...  │
                    └──────┬──────────┬──────┘
                           │          │
                    Write  │          │  Read
                           │          │
              ┌────────────▼──┐  ┌───▼─────────────┐
              │   Primary     │  │   Read Replica   │
              │   (AZ-H)      │  │   (AZ-I)         │
              │ 4vCPU / 8GiB  │  │ 2vCPU / 4GiB     │
              │ 200GB ESSD    │──│ async repl        │
              └───────┬───────┘  └───────────────────┘
                      │
              ┌───────▼───────┐
              │   Standby     │
              │   (AZ-I)      │
              │   semi-sync   │
              └───────────────┘
```

- 主实例处理全部写请求
- 备实例（RDS 托管，不可见）在主实例故障时自动接管
- 只读副本承担所有读查询，减轻主实例压力
- Database Proxy 自动路由读写流量
- 每日自动备份，保留 14 天
- 跨地域备份至 cn-shanghai，实现灾难恢复
- 全链路 SSL 加密
- 监控告警在故障发生前触发

该方案月度成本估算：
- 主实例 (rds.mysql.s3.large, 200GB ESSD): ~600 CNY
- 备实例：高可用版已包含
- 只读副本 (rds.mysql.s2.large, 200GB ESSD): ~350 CNY
- Database Proxy: ~100 CNY
- 跨地域备份存储：~50 CNY
- **总计：~1,100 CNY/月**（约 150 美元）

对于一套具备高可用、读扩展、自动备份及跨地域灾备的生产数据库而言，此成本极具性价比。

## 核心要点总结

**优先使用托管数据库**。除非有特殊需求（如定制插件、特殊复制拓扑或合规强制要求 OS 层控制），否则 RDS 能大幅降低运维负担。

**从 RDS 起步，按需升级 PolarDB**。RDS 可满足 10 TB 以下绝大多数场景；当需要超过 5 个只读副本、自动存储扩容或亚 10ms 复制延迟时，再迁移至 PolarDB。

**按 Buffer Pool 选型，而非仅看 CPU**。性能关键在于热点数据能否完全载入 InnoDB buffer pool，内存规格应据此确定。

**读写分离交由 Database Proxy 处理**。应用层手动路由既易出错又缺乏弹性，Database Proxy 通过单一端点透明实现。

**备份绝非可选项**。务必启用每日自动备份（保留 ≥14 天）、binlog 备份（支持 PITR）及跨地域备份（防灾难），并至少每季度测试恢复流程。

**紧盯四大核心指标**：CPU 利用率、磁盘使用率、连接数使用率、复制延迟。在 80% 阈值设置告警——等你手动发现问题时，用户早已受害多时。

**严格锁定访问权限**：IP 白名单仅限应用层 CIDR、禁用公网端点、强制 SSL 加密、实施最小权限账号。数据库存储着你最宝贵的资产，安全投入绝不能省。

## 下一篇预告

数据库已就绪，高可用、备份、监控均已到位。但生产级应用的需求远不止于此。

下一篇将探讨缓存与存储层：使用 Tair（托管 Redis）处理会话与缓存，OSS 承载对象存储，NAS 提供共享文件系统。这些服务与数据库协同工作，专门应对关系型数据库不擅长的负载场景。
