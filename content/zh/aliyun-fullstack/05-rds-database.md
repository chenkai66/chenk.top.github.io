---
title: "阿里云全栈实战（五）：RDS 与 PolarDB 数据基石"
date: 2026-05-16 09:00:00
tags:
  - Alibaba Cloud
  - RDS
  - PolarDB
  - MySQL
  - Database
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 5
description: "RDS MySQL vs PolarDB: when to use which. Instance sizing, read replicas, proxy endpoints, backup/restore, monitoring, slow query analysis. Build a production database with HA and read scaling."
disableNunjucks: true
translationKey: "aliyun-fullstack-5"
---
我在 ECS 上自建 MySQL 只撑了四个月。流量高峰期的磁盘 I/O 飙升直接把服务打挂。InnoDB buffer pool 跟 OS page cache 抢内存，binary log 填盘速度比 cron 任务清理得还快，单线程复制到“备份”实例已经 lag 了九个小时。凌晨三点，我只能靠扩容磁盘救火。两周后，同样的问题又来了。那天我才真正明白托管数据库存在的意义——不是因为我不会跑 MySQL，而是我不想成为那个半夜三点被叫醒的人，只因 MySQL 判定 relay log 损坏，唯一的修复方案是用可能一致也可能不一致的冷备份重建副本。

![RDS Database Layer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/05-rds-database/cover.png)

这篇文章咱们聊聊阿里云的数据库层：RDS 用于托管关系型数据库，PolarDB 用于应对 RDS 的瓶颈，还有那些让数据保持存活的运维实践——规格 sizing、复制、备份、监控、安全。这个数据库所在的 VPC 是在 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 搭建的。如果想看用 Terraform 部署 数据库的方法，参考 [Terraform Part 5](/zh/terraform-agents/05-storage-for-agent-memory/)。

## 为什么选托管数据库？

在裸 ECS 实例上跑数据库，控制权确实全在你手里。MySQL 版本随便选，`my.cnf` 参数随便调，自定义插件随便装，root 权限在手，`perf` 和 `strace` 直接挂在 mysqld 进程上 debug。这种控制权是真实的，偶尔也确实有用。

但你也签下了这份“卖身契”：

- **OS 补丁。** 内核安全更新需要重启——你配置好 failover 了吗？
- **备份管理。** 用 Percona XtraBackup 做物理备份，每月测试恢复，管理备份文件存储，靠归档 binary log 实现 point-in-time recovery。
- **高可用。** 半同步复制，GTID 配置，用 MHA 或 ProxySQL 这类 orchestrator 做自动 failover，VIP 管理，防止 split-brain。
- **监控。** 慢查询分析，InnoDB buffer pool 命中率，复制延迟，连接池耗尽，锁等待超时。
- **扩容。** 加 read replicas，配置 ProxySQL 做读写分离，管理连接路由。
- **磁盘管理。** 文件系统选型（XFS vs ext4），I/O scheduler 调优，IOPS provisioning，在线磁盘扩容。
- **安全。** SSL/TLS 配置，审计日志，静态加密，密钥轮转。

托管数据库把这些活儿全包了。你失去了 root 权限，也没法随便装 MySQL 插件（别想自定义 UDF，也别想拿 Group Replication 拓扑做实验）。但作为交换，你拿到了自动备份、一键 HA failover、内置监控、一键创建 read replicas，而且再也不会因为磁盘满了半夜被叫醒。

对于 95% 的生产负载，选托管绝对划算。

### 阿里云数据库家族

在深入 RDS 之前，先看看阿里云数据库服务的全貌：

| Service | Type | Engine | AWS Equivalent | When to use |
|---|---|---|---|---|
| **RDS** | Managed relational | MySQL, PostgreSQL, SQL Server, MariaDB | Amazon RDS | Standard OLTP, most web applications |
| **PolarDB** | Cloud-native relational | MySQL-compatible, PostgreSQL-compatible | Amazon Aurora | High read throughput, elastic scaling, large databases |
| **Lindorm** | Multi-model | Wide-column, time-series, search | DynamoDB + Timestream | IoT data, time-series metrics, large-scale KV |
| **Tair** | Managed Redis | Redis-compatible | Amazon ElastiCache | Caching, session storage, rate limiting |
| **AnalyticDB (ADB)** | Analytical | MySQL-compatible, PostgreSQL-compatible | Amazon Redshift | OLAP, real-time analytics, data warehousing |
| **PolarDB-X** | Distributed relational | MySQL-compatible | Aurora Limitless / CockroachDB | Horizontal sharding, distributed transactions |
| **OceanBase** | Distributed relational | MySQL/Oracle-compatible | Spanner-like | Financial-grade, strong consistency across zones |
| **MongoDB** | Document store | MongoDB-compatible | Amazon DocumentDB | Document workloads, flexible schema |

这篇文章咱们重点聊 RDS（主力军）和 PolarDB（RDS 扛不住时的升级路径）。这两个覆盖了阿里云上 90% 的生产数据库需求。

## RDS MySQL 深度解析

RDS MySQL 是阿里云上用得最多的数据库服务。它是 fully managed MySQL 实例，自带自动备份、补丁、监控和高可用。你拿到一个 MySQL-compatible endpoint，连接方式和普通 MySQL 服务器一样——`mysql -h <endpoint> -u <user> -p`。

### 架构

RDS MySQL HA 实例在同一地域内以主备对（primary-standby pair）运行。Primary 处理所有读写。Standby 通过半同步复制接收变更，如果 primary 挂掉，它会自动 promoted。Failover 通常在 30 秒内完成，对应用透明（DNS endpoint 不变，连接只是瞬间断开）。

Standby 不可读——它只为 failover 存在。如果想做读扩容，得加 read replicas（后面会讲）。

### 版本

RDS MySQL 有三个版本：

| Edition | Nodes | HA Failover | Use Case | Cost |
|---|---|---|---|---|
| **Basic** | 1 (primary only) | No | Dev/test, non-critical workloads | Lowest |
| **High-Availability (HA)** | 2 (primary + standby) | Yes (30s) | Production | Standard |
| **Enterprise (Cluster)** | 3 (primary + 2 standby) | Yes (< 30s, zero data loss) | Mission-critical, finance | Highest |

Basic 版没有 standby。如果 primary 挂了，阿里云会从最新备份恢复，这意味着几分钟到几小时的 downtime，取决于数据库大小。**生产环境千万别用 Basic。**

HA 版是大多数生产负载的正确选择。同地域的 standby 提供接近零的 RPO（半同步复制）和约 30 秒的 RTO。

Enterprise 版加了第三个节点做基于 Paxos 的 consensus，保证即使 failover 也不丢数据。这对金融系统很重要，那里丢一个交易都是不可接受的。

### 存储类型

| Storage Type | IOPS Range | Latency | Max Size | Best For |
|---|---|---|---|---|
| **Local SSD** | Up to 240,000 | Lowest (< 0.1ms) | 3 TB | Maximum performance, small-to-medium databases |
| **ESSD PL1** | Up to 50,000 | 0.1-0.3ms | 32 TB | General production |
| **ESSD PL2** | Up to 100,000 | 0.1-0.3ms | 32 TB | High-IOPS workloads |
| **ESSD PL3** | Up to 1,000,000 | 0.1-0.3ms | 32 TB | Extreme IOPS, large OLTP |

Local SSD 延迟最低，因为 SSD 物理附着在宿主机上，不走网络。代价是容量：Local SSD 上限 3 TB。数据库要是会超过这个大小，直接用 ESSD。

ESSD（Enhanced SSD）是网络附加块存储，容量弹性。你可以在线扩容 ESSD 不停机，32 TB 上限能容纳大多数数据库。ESSD PL1 是默认选项，也是大多数负载的正确选择。只有当 CloudMonitor 显示你的 IOPS 持续 hitting PL1 天花板时，再升级到 PL2 或 PL3。

> **Practical tip:** 从 ESSD PL1 起步。在 CloudMonitor 里盯一周 `IOPSUsage` 指标。如果看到持续 usage 超过 PL1 限制的 70%，再升级。如果从来没超过 20%，那你就是在为不需要的余量付费——不过 PL0 和 PL1 的价差很小，这点保险钱值得花。

### 连接方式

RDS MySQL 提供三种连接方式：

1. **内网 endpoint** —— 只能从同一 VPC 内访问。这是你的应用服务器用的。长得像 `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`。

2. **公网 endpoint** —— 互联网可访问。默认禁用。只为了远程管理才开启（并且要用 IP whitelist 锁死）。千万别让生产应用通过公网 endpoint 连数据库。

3. **数据库代理 endpoint** —— 代理层，提供读写分离、连接池和短连接优化。这是使用 read replicas 的应用推荐的 endpoint。
## PolarDB: When RDS Isn't Enough

PolarDB 是阿里云的云原生数据库。如果说 RDS 是“托管的 MySQL 实例”，那 PolarDB 就是骨子里完全不同的架构，只不过恰好兼容 MySQL 协议而已。

### How PolarDB differs from RDS

核心架构差异在于 **存算分离**。RDS 里，每个实例（主、备、只读）都有自己的磁盘和数据副本。PolarDB 则是所有计算节点共享一个分布式存储层。

这个改动影响深远：

- **加只读实例只要几分钟，不是几小时。** 新的 RDS 只读实例得把全量数据拷到自己的磁盘上。PolarDB 只读实例只是启动一个新的计算节点挂载到共享存储——根本不用拷数据。

- **存储自动扩容。** PolarDB 存储随数据增长，上限 128 TB。你不用手动预配存储容量。再也不用半夜 3 点因为磁盘满了被报警叫醒。

- **最多 15 个只读实例。** RDS 最多支持 5 个只读实例。PolarDB 支持 15 个，全都读同一份共享存储，复制延迟几乎为零（通常低于 10ms）。

- **故障切换更快。** 因为备节点本来就能访问同一份存储，切换不需要数据同步。PolarDB 故障切换能在 10 秒内完成。

### RDS vs PolarDB comparison

| 特性 | RDS MySQL HA | PolarDB MySQL |
|---|---|---|
| 架构 | 主 + 备，独立存储 | 共享存储，存算分离 |
| 最大存储 | 32 TB (ESSD) | 128 TB (自动扩容) |
| 只读实例 | 最多 5 个 | 最多 15 个 |
| 实例创建时间 | 几小时 (数据拷贝) | 几分钟 (共享存储) |
| 复制延迟 | 秒级 (异步/半同步) | < 10ms (共享存储) |
| 故障切换时间 | ~30 秒 | < 10 秒 |
| 存储扩容 | 手动 (扩容磁盘) | 自动 |
| Serverless 模式 | 无 | 有 |
| 兼容性 | MySQL 5.6/5.7/8.0 | 兼容 MySQL 5.6/5.7/8.0 |
| 价格 (同等配置) | 1x | 1.2-1.5x |
| 适用场景 | 标准 OLTP | 高读吞吐，大库，弹性负载 |

### PolarDB Serverless

PolarDB 提供 Serverless 模式，计算资源随负载自动伸缩。你设定 PCU（PolarDB Compute Units）的最小最大值，系统会在其间自动调整：

- **Minimum PCU: 1** -- 空闲时数据库缩容到 1 PCU，成本极低。
- **Maximum PCU: 32** -- 流量高峰时，数据库自动扩容。
- **Scale-to-zero** -- 长期无活动时，PolarDB Serverless 可以完全暂停计算，只收存储费。

这对开发库、staging 环境，或者流量波动极大的生产负载（比如大促期间流量翻 100 倍的电商站点）简直完美。

### When to choose PolarDB over RDS

满足以下情况选 PolarDB：

- 数据库将超过 10 TB 且还在增长
- 需要超过 5 个只读实例
- 需要低于 10ms 的复制延迟以保证读一致性
- 负载波动大，适合 Serverless 自动伸缩
- 要求故障切换时间小于 10 秒
- 不想手动管理存储容量

满足以下情况选 RDS：

- 数据库小于 5 TB 且增长可预测
- 只需要 3 个或更少的只读实例
- 追求同等配置下的最低成本
- 需要 PolarDB 不支持的特定 MySQL 特性
- 偏好更简单的运维模型

大部分项目，先上 RDS。碰到具体瓶颈再迁 PolarDB。MySQL 线协议兼容，迁移通常就是 dump 加 restore。

## Instance Sizing Guide

选对 RDS 实例规格是成本控制最关键的一步。选小了，负载一高查询就慢；选大了，闲置资源全是钱。

### Key metrics for sizing

- **vCPUs** -- 决定并发查询处理能力。CPU 密集型负载（复杂 JOIN、聚合）需要更多。
- **Memory** -- 直接对应 InnoDB buffer pool 大小。理想情况是，你的整个热点数据集都能放进 buffer pool。
- **Max IOPS** -- 每秒磁盘操作上限。事务密集型负载容易撞到这个天花板。
- **Max connections** -- 每个实例类型都有硬限制。高并发负载必须在应用层做连接池。

### Sizing table for common workloads

| 负载类型 | 实例类型 | vCPU | Memory | Max IOPS | Max Connections | 月成本 (约) |
|---|---|---|---|---|---|---|
| 小型博客 / CMS | rds.mysql.s2.large | 2 | 4 GiB | 2,000 | 300 | ~200 CNY |
| 中型 Web 应用 | rds.mysql.s3.large | 4 | 8 GiB | 5,000 | 600 | ~400 CNY |
| API 后端 | rds.mysql.m1.medium | 4 | 16 GiB | 7,000 | 4,000 | ~800 CNY |
| 重度 OLTP | rds.mysql.c1.xlarge | 8 | 32 GiB | 12,000 | 8,000 | ~1,600 CNY |
| 大型 SaaS | rds.mysql.c2.xlarge | 16 | 64 GiB | 14,000 | 16,000 | ~3,200 CNY |
| 数据密集型 | rds.mysql.st.h43 | 60 | 470 GiB | 120,000 | 48,000 | ~20,000 CNY |

### The buffer pool rule

InnoDB 性能就看一点：数据能不能放进 buffer pool。buffer pool 是内存里的数据和索引页缓存。走 buffer pool 的查询是微秒级。走磁盘的查询是毫秒级——慢 1000 倍。

公式如下：

```
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

如果你的 working set 是 12 GiB，16 GiB 内存的实例给你分配的 buffer pool 大概 12 GiB（16 的 75%）。这就太紧了。直接上 32 GiB 留点增长余量。

### Connection limits

每个实例类型都有最大连接数限制。超了就直接 `Too many connections` 拒绝。这是生产环境最常见的问题之一。

解决办法不是买大实例。解决办法是在应用层用连接池：

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

配好连接池，20-30 个连接能扛几千并发应用请求。不用连接池的话，每个请求新建数据库连接，几百并发用户就把连接数耗光了。

## Creating an RDS Instance

### Step-by-step via CLI

下面是一套完整的 CLI 创建生产环境 RDS MySQL 实例的流程。我们用高可用版，ESSD 存储，跟应用服务器放在同一个 VPC。

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

拆解一下关键参数：

- **Engine/EngineVersion**: MySQL 8.0。新项目直接用 8.0 -- 5.7 已经进入维护期了。
- **DBInstanceClass**: `rds.mysql.s3.large` 给的是 4 vCPU, 8 GiB 内存。适合中型 Web 应用。
- **DBInstanceStorage**: 100 GiB 起步。ESSD 后面可以在线扩容。
- **Category**: `HighAvailability` 给的是主 + 备。
- **ZoneId/ZoneIdSlave1**: 主节点在可用区 H，备节点在可用区 I。跨可用区高可用意味着单个可用区故障不会搞挂数据库。
- **SecurityIPList**: 只允许应用层子网连接（10.0.10.0/24 和 10.0.11.0/24）。这是 RDS IP 白名单，跟安全组是分开的。
- **ConnectionMode**: `Standard` 走直连。`Safe` 走代理（已废弃，改用 Database Proxy）。

等待实例就绪：

```bash
# Check instance status (wait for "Running")
aliyun rds DescribeDBInstanceAttribute \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=DBInstanceStatus,DBInstanceClass,Engine,EngineVersion
```

### Create a database and accounts

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

别给应用账号 `Super` 权限。`Normal` 账号配特定数据库的 `ReadWrite` 符合最小权限原则。单独建个带 `Super` 的管理员账号给 DBA 操作用。

### Get the connection endpoint

```bash
# Get the internal connection string
aliyun rds DescribeDBInstanceNetInfo \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=ConnectionString,IPAddress,Port,DBInstanceNetType
```

内网地址格式类似 `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`。应用配置里用这个：

```bash
# Test connection from an ECS instance in the same VPC
mysql -h rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com \
  -u appuser -p \
  -e "SELECT VERSION(); SHOW DATABASES;"
```

### Configure instance parameters

RDS 用参数模板（parameter groups）管 MySQL 配置。可以改单个参数或者套用模板：

```bash
# View current parameters
aliyun rds DescribeParameters \
  --DBInstanceId rm-bp1xxxxxxxxx

# Modify key parameters for production
aliyun rds ModifyParameter \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --Parameters "innodb_buffer_pool_size:6442450944;slow_query_log:ON;long_query_time:1;max_connections:500;innodb_flush_log_at_trx_commit:1;sync_binlog:1"
```

生产环境关键参数：

| 参数 | 推荐值 | 原因 |
|---|---|---|
| `innodb_buffer_pool_size` | 实例内存的 75% | 最大化内存数据访问 |
| `slow_query_log` | ON | 必须开。性能调试全靠它。 |
| `long_query_time` | 1 (秒) | 记录超过 1 秒的查询 |
| `innodb_flush_log_at_trx_commit` | 1 | 完整的 ACID 耐久性。只有能接受潜在数据丢失时才设 2。 |
| `sync_binlog` | 1 | 每次提交同步 binlog。崩溃安全复制必需。 |
| `max_connections` | 实际需求 + 20% 余量 | 除非必要，别设到实例上限 |
## 只读副本与数据库代理

### 添加只读副本

只读副本其实是主实例的一个异步拷贝。它专门处理 SELECT 查询，把读流量从主库卸下来。对于读多写少的场景（大多数 Web 应用），这是最主要的扩容手段。

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

创建副本的流程大概是这样：

1. RDS 给主库打个内部快照
2. 用快照数据开通一个新实例
3. 自动配置 Binlog 复制
4. 副本就绪（小库几分钟，大库几小时）

检查一下复制状态：

```sql
-- On the read replica, check replication lag
SHOW SLAVE STATUS\G

-- Key fields to watch:
-- Seconds_Behind_Master: should be 0 or very low
-- Slave_IO_Running: Yes
-- Slave_SQL_Running: Yes
```

### 数据库代理 (Database Proxy)

手动管理只读副本挺痛苦的。应用得知道哪个端点是主库（写），哪个是副本（读），还得自己路由。数据库代理就是来解决这个问题的。

数据库代理是位于应用和 RDS 实例之间的内置代理层。它能提供：

- **自动读写分离** -- `INSERT/UPDATE/DELETE` 走主库，`SELECT` 走副本
- **连接池** -- 减少数据库的连接开销
- **单一端点** -- 应用只连一个地址，路由交给代理处理

开启数据库代理：

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

配置读写分离：

```bash
# Enable read/write splitting with weight distribution
aliyun rds ModifyReadWriteSplittingConnection \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --ConnectionStringPrefix "myapp-proxy" \
  --DistributionType Standard \
  --MaxDelayTime 10 \
  --Weight '{"rm-bp1xxxxxxxxx":"0","rr-bp1yyyyyyyyy":"100"}'
```

权重配置的意思是：

- 主库 (`rm-bp1xxxxxxxxx`)：权重 0 -- 读请求不发往主库（写请求依然自动发到这里）
- 副本 (`rr-bp1yyyyyyyyy`)：权重 100 -- 所有读请求都发往副本

如果副本的复制延迟超过 `MaxDelayTime`（10 秒），代理会自动把读请求切回主库，直到副本追上进度。

应用的连接字符串改成代理端点：

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

应用代码不用改。代理会透明地处理所有路由。

## 备份与恢复

备份是唯一不能省的东西。当其他都挂了——部署搞砸、数据损坏、手滑 `DROP TABLE`——备份是你最后的防线。

### 自动备份

RDS 会按可配置的计划自动备份。默认情况下：

- **全量备份** 每天在低峰期窗口执行
- **Binlog 备份** 持续进行（用于按时间点恢复）
- **保留周期**：7 天（可配置最长 730 天）

配置备份计划：

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

自动备份对实例性能影响不大。RDS 用的是基于快照的备份技术，哪怕库很大也能快速完成。

### 手动备份

在任何 risky 操作之前（schema 变更、大批量导入、版本升级），先做个手动备份：

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

### 按时间点恢复 (PITR)

这是最重要的备份功能。PITR 让你能把数据库恢复到保留期内的任意一秒。如果有人手滑在 14:32:07 执行了 `DELETE FROM users WHERE 1=1`，你可以恢复到 14:32:06。

PITR 的原理是结合最近的全量备份和 Binlog，重放变更到指定时间点。

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

PITR 恢复总是会创建一个新实例。它绝不会覆盖现有实例。这是设计使然——你先在新实例上验证恢复的数据，确认没问题再切换。

### 跨地域备份

为了灾难恢复，开启跨地域备份。这会把备份复制到一个不同的地域，这样即使主地域整个挂了，数据也是安全的：

```bash
# Enable cross-region backup
aliyun rds ModifyInstanceCrossBackupPolicy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --CrossBackupType 1 \
  --CrossBackupRegion cn-shanghai \
  --RetentType 1 \
  --Retention 30
```

跨地域备份会增加成本（secondary 地域的数据传输 + 存储），但这是唯一能防止整个地域故障的手段。对于生产库，只要数据涉及业务核心，必须开。

## 监控与性能

### 云监控指标

RDS 会自动把指标上报给云监控。这几个关键指标要盯紧：

| Metric | Warning Threshold | Critical Threshold | What to do |
|---|---|---|---|
| CPU Utilization | > 70% sustained | > 90% sustained | Upgrade instance type or optimize queries |
| IOPS Usage | > 70% of limit | > 90% of limit | Upgrade storage PL or optimize I/O patterns |
| Connection Usage | > 70% of max | > 90% of max | Add connection pooling, increase max_connections |
| Disk Usage | > 70% | > 85% | Expand storage immediately |
| Replication Lag | > 5 seconds | > 30 seconds | Check replica performance, reduce write load |

设置报警：

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

DAS 是阿里云为 RDS 提供的智能诊断工具。它不止看原始指标，还能提供：

- **自动 SQL 诊断** -- 识别慢查询并建议索引优化
- **实时会话分析** -- 展示活跃会话、锁等待和阻塞查询
- **性能趋势分析** -- 跟踪性能指标随时间的变化，检测退化
- **自动优化** -- 可以自动应用推荐的索引变更（需确认开启）

RDS 实例默认开启 DAS。在 RDS 控制台的“自治服务”下访问。

### 慢查询分析

慢查询是数据库性能问题的头号杀手。RDS 会记录所有超过 `long_query_time` 的查询（我们之前设的是 1 秒）。

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

下载慢查询日志供离线分析：

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

### 索引建议

最简单的性能优化就是加对的索引。用 `EXPLAIN` 看看 MySQL 怎么执行查询：

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
## 安全性

### IP 白名单

RDS 的 IP 白名单是咱们的第一道防线。它在任何认证发生之前，就在网络层控制了哪些 IP 能连数据库。

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

千万别往白名单里加 `0.0.0.0/0`。这等于把数据库对所有 IP 敞开——要是开了公网 endpoint，那就是向整个互联网敞开。密码再强，也扛不住暴力破解的轮番轰炸。

### SSL 加密

启用 SSL 能给应用和数据库之间的传输数据加密：

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

接着配置你的应用启用 SSL：

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

### TDE (Transparent Data Encryption)

TDE 负责加密磁盘上的静态数据。就算有人拿到了底层存储权限，没有密钥也休想读懂数据。

```bash
# Enable TDE
aliyun rds ModifyDBInstanceTDE \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --TDEStatus Enabled \
  --EncryptionKey "kms-key-id-from-kms"
```

TDE 用的是 AES-256 加密，密钥管理直接对接 KMS (Key Management Service)。注意，TDE 一旦开启就关不掉。性能开销很小，大多数负载下不到 5%。

### SQL 审计

SQL Audit 会记录数据库执行的每一条 SQL 语句。不管是合规检查、安全调查还是排查问题，这都是必需品。

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

审计日志默认存 30 天。要想长期保存，得导出到 SLS (Simple Log Service) 去做归档和分析。

### 从 ECS 建立安全连接

下面是从应用侧 ECS 连到 RDS 的完整安全链条：

1. **VPC 隔离** -- RDS 和 ECS 放在同一个 VPC 里，但要用不同的 VSwitch（数据层 vs 应用层）
2. **安全组** -- ECS 安全组放开 outbound 3306；RDS 白名单只允许应用层 CIDR 入站
3. **SSL 加密** -- 传输数据加密
4. **最小权限账号** -- 应用只用 Normal 账号，且只给特定库的 ReadWrite 权限
5. **不开公网 endpoint** -- 数据库不让互联网访问

```bash
# Verify: from app-tier ECS, connection works
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p --ssl-ca=/path/to/ca.pem -e "SELECT 1"

# Verify: from web-tier ECS (different security group), connection fails
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p -e "SELECT 1"
# ERROR 2003 (HY000): Can't connect to MySQL server (connection refused / timeout)
```
## 解决方案：具备高可用和读扩展的生产级数据库

咱们直接动手，从零搭建一套完整的生产级数据库环境：RDS MySQL 高可用版，带一个只读实例，配上 Database Proxy 做读写分离，再加上自动备份和监控。这是中大型 Web 应用的标准参考架构。

### 第一步：创建 RDS MySQL 高可用实例

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

### 第二步：创建数据库和账号

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

### 第三步：配置生产级参数

```bash
# Apply production-tuned parameters
aliyun rds ModifyParameter \
  --DBInstanceId $RDS_ID \
  --Parameters "slow_query_log:ON;long_query_time:1;innodb_flush_log_at_trx_commit:1;sync_binlog:1;max_connections:500;innodb_print_all_deadlocks:ON;log_queries_not_using_indexes:ON"
```

### 第四步：创建只读实例

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

### 第五步：启用 Database Proxy 实现读写分离

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

### 第六步：配置备份策略

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

### 第七步：设置监控告警

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

### 第八步：验证完整 setup

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

从应用层的 ECS 实例（同 VPC 内）测试整个链路：

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

咱们到底搭建了个啥：

```
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

主实例扛所有写流量。备实例（用户不可见，RDS 托管）在主实例挂掉时自动接管。只读实例处理所有读查询，把主实例的压力卸掉。Database Proxy 自动路由流量。备份每天跑，保留 14 天。跨地域备份到 cn-shanghai 做灾难恢复。SSL 加密所有连接。监控告警会在问题变成故障前触发。

这套方案每月的总成本（估算）：

- 主实例 (rds.mysql.s3.large, 200GB ESSD): ~600 CNY
- 备实例：高可用版自带
- 只读实例 (rds.mysql.s2.large, 200GB ESSD): ~350 CNY
- Database Proxy: ~100 CNY
- 跨地域备份存储：~50 CNY
- **总计：~1,100 CNY/月** (~$150 USD)

对于一套具备高可用、读扩展、自动备份和跨地域灾备的生产数据库来说，这价格相当划算。

## 核心要点

**能用托管就别自己建。** 除非你有非常特殊的理由（比如需要定制插件、特殊的复制拓扑，或者合规要求必须控制操作系统层面），否则 RDS 能帮你省去巨大的运维负担。

**先从 RDS 起步，需要时再升级 PolarDB。** 10 TB 以下的负载 RDS 都能扛。当你需要超过 5 个只读实例、存储自动扩容，或者复制延迟要压在 10ms 以内时，再考虑迁到 PolarDB。

**按 Buffer Pool sizing，别只看 CPU。** 性能最关键的因素是你的工作数据集能不能塞进 InnoDB buffer pool。实例内存要按这个配，而不是盯着 CPU 核数。

**读写分离交给 Database Proxy。** 在应用代码里手动路由读写既容易出错又不够灵活。Database Proxy 用一个连接地址就能透明搞定。

**备份没得商量。** 开启自动每日备份，保留期至少 14 天，启用 binlog 备份支持 PITR，再配上跨地域备份防灾难。每季度至少测试一次恢复流程。

**盯死四个核心指标。** CPU 利用率、磁盘使用率、连接数使用率、复制延迟。告警阈值设在 80%。等你手动发现问题时，用户早就受害好几个小时了。

**把访问权限锁死。** IP 白名单只放应用层 CIDR，别开公网地址，强制 SSL 加密，账号权限最小化。数据库里存的是你最宝贵的资产，安全上别省钱。
## 接下来

数据库跑起来了，复制、备份、监控也都到位了。但要想撑起生产环境，光靠关系型数据库可不够。

下一篇咱们聊聊缓存和存储层——用 Tair（托管 Redis）做会话存储和缓存，OSS 负责对象存储，NAS 搞定共享文件系统。这些服务就跟数据库并排部署，专门处理那些关系型数据库不擅长扛的负载。