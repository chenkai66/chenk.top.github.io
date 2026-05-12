---
title: "Alibaba Cloud Full Stack (5): RDS and PolarDB — The Database Layer"
date: 2026-05-02 09:00:00
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

My self-managed MySQL on ECS lasted exactly four months before a disk I/O spike during peak traffic brought the whole thing down. The InnoDB buffer pool was fighting the OS page cache for memory, the binary log was filling the system disk faster than my cron job could rotate it, and the single-threaded replication to my "backup" instance was nine hours behind. I fixed it at 3 AM by throwing more disk at it. Then it happened again two weeks later. That is the day I learned why managed databases exist — not because I cannot run MySQL, but because I do not want to be the person paged at 3 AM when MySQL decides the relay log is corrupted and the only fix is to rebuild the replica from a cold backup that may or may not be consistent.


This article covers the database layer on Alibaba Cloud: RDS for managed relational databases, PolarDB for when RDS hits its limits, and the operational practices — sizing, replication, backup, monitoring, security — that keep your data alive. The VPC where this database lives was set up in [Part 3](/en/aliyun-fullstack/03-vpc-networking/). For the Terraform approach to database provisioning, see [Terraform Part 5](/en/terraform-agents/05-storage-for-agent-memory/).

## Why Managed Databases?

Running a database on a raw ECS instance gives you complete control. You choose the exact MySQL version, you tune every `my.cnf` parameter, you install custom plugins, you have root access to the OS and can run `perf` and `strace` directly on the mysqld process. That control is real and occasionally valuable.

But here is what you are also signing up for:

- **OS patching.** The kernel security update that requires a reboot — do you have failover configured?
- **Backup management.** Physical backups with Percona XtraBackup, testing restores monthly, managing storage for backup files, implementing point-in-time recovery with binary log archival.
- **High availability.** Semi-synchronous replication, GTID configuration, automated failover with orchestrators like MHA or ProxySQL, VIP management, split-brain prevention.
- **Monitoring.** Slow query analysis, InnoDB buffer pool hit ratio, replication lag, connection pool exhaustion, lock wait timeouts.
- **Scaling.** Adding read replicas, configuring ProxySQL for read/write splitting, managing connection routing.
- **Disk management.** Filesystem choice (XFS vs ext4), I/O scheduler tuning, IOPS provisioning, online disk expansion.
- **Security.** SSL/TLS configuration, audit logging, encryption at rest, key rotation.

A managed database handles all of this. You lose root OS access and the ability to install arbitrary MySQL plugins (no custom UDFs, no Group Replication topology experiments). In return, you get automated backups, one-click HA failover, built-in monitoring, and read replicas with a click. You also avoid being paged at 3 AM because a disk filled up.

For 95% of production workloads, the trade-off is overwhelmingly in favor of managed.

### The Alibaba Cloud database family

Before diving into RDS, here is the complete picture of database services on Alibaba Cloud:

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

For this article, we focus on RDS (the workhorse) and PolarDB (the upgrade path when RDS is not enough). These two cover 90% of production database needs on Alibaba Cloud.

![Decision map for choosing the right Alibaba Cloud database service](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_db_family_map.png)

*The decision map above shows where each service fits — most of this article concentrates on the OLTP branch.*

## RDS MySQL Deep Dive

RDS MySQL is the most commonly used database service on Alibaba Cloud. It is a fully managed MySQL instance with automated backups, patching, monitoring, and high availability. You get a MySQL-compatible endpoint and connect to it the same way you connect to any MySQL server — `mysql -h <endpoint> -u <user> -p`.

### Architecture

An RDS MySQL HA instance runs as a primary-standby pair within the same region. The primary handles all reads and writes. The standby receives changes via semi-synchronous replication and is promoted automatically if the primary fails. Failover typically completes in 30 seconds or less and is transparent to the application (the DNS endpoint stays the same, the connection just drops momentarily).

The standby is not readable — it exists solely for failover. If you want read scaling, you add read replicas (covered later in this article).

![HA failover timeline showing the ~30-second sequence from heartbeat loss to writes resuming](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_ha_failover_sequence.png)

*The failover sequence above is fully managed by RDS — your application sees nothing but a brief connection drop and a reconnect against the unchanged DNS hostname.*

### Editions

RDS MySQL comes in three editions:

| Edition | Nodes | HA Failover | Use Case | Cost |
|---|---|---|---|---|
| **Basic** | 1 (primary only) | No | Dev/test, non-critical workloads | Lowest |
| **High-Availability (HA)** | 2 (primary + standby) | Yes (30s) | Production | Standard |
| **Enterprise (Cluster)** | 3 (primary + 2 standby) | Yes (< 30s, zero data loss) | Mission-critical, finance | Highest |

The Basic edition has no standby. If the primary fails, Alibaba Cloud restores from the latest backup, which means minutes to hours of downtime depending on database size. Never use Basic for production.

The HA edition is the right choice for most production workloads. The standby in the same region provides RPO near zero (semi-sync replication) and RTO of about 30 seconds.

The Enterprise edition adds a third node for Paxos-based consensus, guaranteeing zero data loss even during failover. This matters for financial systems where losing even one transaction is unacceptable.

### Storage types

| Storage Type | IOPS Range | Latency | Max Size | Best For |
|---|---|---|---|---|
| **Local SSD** | Up to 240,000 | Lowest (< 0.1ms) | 3 TB | Maximum performance, small-to-medium databases |
| **ESSD PL1** | Up to 50,000 | 0.1-0.3ms | 32 TB | General production |
| **ESSD PL2** | Up to 100,000 | 0.1-0.3ms | 32 TB | High-IOPS workloads |
| **ESSD PL3** | Up to 1,000,000 | 0.1-0.3ms | 32 TB | Extreme IOPS, large OLTP |

Local SSD provides the lowest latency because the SSD is physically attached to the host machine, not accessed over the network. The trade-off is capacity: Local SSD maxes out at 3 TB. For databases that will grow beyond that, start with ESSD.

ESSD (Enhanced SSD) is network-attached block storage with elastic capacity. You can expand ESSD online without downtime, and the maximum size of 32 TB accommodates most databases. ESSD PL1 is the default and the right choice for most workloads. Move to PL2 or PL3 only if CloudMonitor shows your IOPS consistently hitting the PL1 ceiling.

> **Practical tip:** Start with ESSD PL1. Watch the `IOPSUsage` metric in CloudMonitor for a week. If you see sustained usage above 70% of the PL1 limit, upgrade. If you never break 20%, you are paying for headroom you do not need — but the cost difference between PL0 and PL1 is small enough that the insurance is worth it.

### Connection methods

RDS MySQL provides three connection methods:

1. **Internal endpoint** — accessible only from within the same VPC. This is what your application servers use. Looks like `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`.

2. **Public endpoint** — accessible from the internet. Disabled by default. Only enable this for remote administration (and lock it down with the IP whitelist). You should never have your production application connecting to the database over the public endpoint.

3. **Database Proxy endpoint** — a proxy layer that provides read/write splitting, connection pooling, and short-lived connection optimization. This is the recommended endpoint for applications that use read replicas.

## PolarDB: When RDS Isn't Enough

PolarDB is Alibaba Cloud's cloud-native database. While RDS is "a managed MySQL instance," PolarDB is a fundamentally different architecture that happens to speak the MySQL protocol.

![RDS vs PolarDB feature comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_rds_vs_polardb.png)

### How PolarDB differs from RDS

The key architectural difference is **compute-storage separation**. In RDS, each instance (primary, standby, replica) has its own copy of the data on its own disk. In PolarDB, all compute nodes share a single distributed storage layer.

This changes everything:

- **Adding a read replica takes minutes, not hours.** A new RDS read replica must copy the entire dataset to its own disk. A PolarDB read replica just boots a new compute node that attaches to the shared storage — no data copy required.

- **Storage scales automatically.** PolarDB storage grows as your data grows, up to 128 TB. You never provision storage capacity manually. You never run out of disk at 3 AM.

- **Up to 15 read replicas.** RDS supports up to 5 read replicas. PolarDB supports 15, all reading from the same shared storage with near-zero replication lag (typically under 10ms).

- **Faster failover.** Because the standby already has access to the same storage, failover does not require data sync. PolarDB failover completes in under 10 seconds.

### RDS vs PolarDB comparison

| Feature | RDS MySQL HA | PolarDB MySQL |
|---|---|---|
| Architecture | Primary + standby, separate storage | Shared storage, compute-storage separation |
| Max storage | 32 TB (ESSD) | 128 TB (auto-scaling) |
| Read replicas | Up to 5 | Up to 15 |
| Replica creation time | Hours (data copy) | Minutes (shared storage) |
| Replication lag | Seconds (async/semi-sync) | < 10ms (shared storage) |
| Failover time | ~30 seconds | < 10 seconds |
| Storage scaling | Manual (expand disk) | Automatic |
| Serverless mode | No | Yes |
| Compatibility | MySQL 5.6/5.7/8.0 | MySQL 5.6/5.7/8.0 compatible |
| Price (comparable spec) | 1x | 1.2-1.5x |
| Best for | Standard OLTP | High read throughput, large databases, elastic workloads |

### PolarDB Serverless

PolarDB offers a Serverless mode where compute scales automatically based on load. You set minimum and maximum PCU (PolarDB Compute Units), and the system scales between them:

- **Minimum PCU: 1** — the database scales down to 1 PCU during idle periods, costing very little.
- **Maximum PCU: 32** — during traffic spikes, the database scales up automatically.
- **Scale-to-zero** — PolarDB Serverless can pause compute entirely during sustained inactivity, charging only for storage.

This is perfect for development databases, staging environments, or production workloads with extreme traffic variation (e.g., an e-commerce site that gets 100x traffic during promotions).

### When to choose PolarDB over RDS

Choose PolarDB when:

- Your database will exceed 10 TB and is still growing
- You need more than 5 read replicas
- You need sub-10ms replication lag for read consistency
- Your workload is bursty and would benefit from serverless auto-scaling
- Failover time under 10 seconds is a requirement
- You want to avoid manual storage capacity management

Choose RDS when:

- Your database is under 5 TB and growth is predictable
- You need 3 or fewer read replicas
- You want the lowest cost for a given spec
- You need specific MySQL features that PolarDB does not support
- You prefer the simpler operational model

For most projects, start with RDS. Migrate to PolarDB when you hit a specific limitation. The MySQL wire protocol compatibility means migration is typically a data dump and restore.

## Instance Sizing Guide

Picking the right RDS instance type is the most impactful cost decision you will make. Too small and your queries slow down under load. Too large and you burn money on idle capacity.

![RDS instance sizing decision guide](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_sizing_guide.png)

### Key metrics for sizing

- **vCPUs** — determines concurrent query processing capacity. CPU-bound workloads (complex JOINs, aggregations) need more.
- **Memory** — directly maps to InnoDB buffer pool size. Ideally, your entire working dataset fits in the buffer pool.
- **Max IOPS** — the ceiling for disk operations per second. Transaction-heavy workloads hit this.
- **Max connections** — each instance type has a hard limit. Connection pooling on the application side is mandatory for high-concurrency workloads.

### Sizing table for common workloads

| Workload | Instance Type | vCPU | Memory | Max IOPS | Max Connections | Monthly Cost (approx) |
|---|---|---|---|---|---|---|
| Small blog / CMS | rds.mysql.s2.large | 2 | 4 GiB | 2,000 | 300 | ~200 CNY |
| Medium web app | rds.mysql.s3.large | 4 | 8 GiB | 5,000 | 600 | ~400 CNY |
| API backend | rds.mysql.m1.medium | 4 | 16 GiB | 7,000 | 4,000 | ~800 CNY |
| Heavy OLTP | rds.mysql.c1.xlarge | 8 | 32 GiB | 12,000 | 8,000 | ~1,600 CNY |
| Large SaaS | rds.mysql.c2.xlarge | 16 | 64 GiB | 14,000 | 16,000 | ~3,200 CNY |
| Data-intensive | rds.mysql.st.h43 | 60 | 470 GiB | 120,000 | 48,000 | ~20,000 CNY |

### The buffer pool rule

InnoDB performance is dominated by one thing: whether your data fits in the buffer pool. The buffer pool is an in-memory cache of data and index pages. A query that reads from the buffer pool takes microseconds. The same query reading from disk takes milliseconds — 1,000x slower.

The formula:

```
Buffer Pool Size ≈ 75% of instance memory
Working Set = frequently accessed data + all indexes
If Working Set ≤ Buffer Pool Size → fast
If Working Set > Buffer Pool Size → disk thrashing
```

To check your current buffer pool usage:

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

If your working set is 12 GiB, an instance with 16 GiB memory gives you a buffer pool of about 12 GiB (75% of 16). That is a tight fit. Go with 32 GiB to have headroom for growth.

### Connection limits

Each instance type has a maximum connection limit. Exceeding it means new connections are refused with `Too many connections`. This is one of the most common production issues.

The fix is not to buy a bigger instance. The fix is to use connection pooling in your application:

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

A well-configured connection pool with 20-30 connections can handle thousands of concurrent application requests. Without pooling, each request opens a new database connection, and you exhaust the limit at a few hundred concurrent users.

## Creating an RDS Instance

### Step-by-step via CLI

Here is the complete sequence to create a production RDS MySQL instance. We use the HA edition with ESSD storage, placed in the same VPC as our application servers.

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

Breaking down the important parameters:

- **Engine/EngineVersion**: MySQL 8.0. Use 8.0 for new projects — 5.7 is in maintenance mode.
- **DBInstanceClass**: `rds.mysql.s3.large` gives us 4 vCPU, 8 GiB memory. Right for a medium web app.
- **DBInstanceStorage**: 100 GiB initial. ESSD can be expanded online later.
- **Category**: `HighAvailability` gives us primary + standby.
- **ZoneId/ZoneIdSlave1**: Primary in zone H, standby in zone I. Cross-AZ HA means a zone failure does not take out the database.
- **SecurityIPList**: Only allow connections from the app tier subnets (10.0.10.0/24 and 10.0.11.0/24). This is the RDS IP whitelist, separate from security groups.
- **ConnectionMode**: `Standard` uses direct connections. `Safe` routes through a proxy (deprecated, use Database Proxy instead).

Wait for the instance to become available:

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

Never give the application account `Super` privilege. The `Normal` account with `ReadWrite` on specific databases follows the principle of least privilege. Create a separate admin account with `Super` for DBA operations.

### Get the connection endpoint

```bash
# Get the internal connection string
aliyun rds DescribeDBInstanceNetInfo \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --output cols=ConnectionString,IPAddress,Port,DBInstanceNetType
```

The internal endpoint looks like `rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com:3306`. Use this in your application configuration:

```bash
# Test connection from an ECS instance in the same VPC
mysql -h rm-bp1xxxxxxxxx.mysql.rds.aliyuncs.com \
  -u appuser -p \
  -e "SELECT VERSION(); SHOW DATABASES;"
```

### Configure instance parameters

RDS uses parameter templates (parameter groups) to manage MySQL configuration. You can modify individual parameters or apply a template:

```bash
# View current parameters
aliyun rds DescribeParameters \
  --DBInstanceId rm-bp1xxxxxxxxx

# Modify key parameters for production
aliyun rds ModifyParameter \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --Parameters "innodb_buffer_pool_size:6442450944;slow_query_log:ON;long_query_time:1;max_connections:500;innodb_flush_log_at_trx_commit:1;sync_binlog:1"
```

The critical parameters for production:

| Parameter | Recommended Value | Why |
|---|---|---|
| `innodb_buffer_pool_size` | 75% of instance memory | Maximize in-memory data access |
| `slow_query_log` | ON | Always. You need this for performance debugging. |
| `long_query_time` | 1 (second) | Log queries taking more than 1 second |
| `innodb_flush_log_at_trx_commit` | 1 | Full ACID durability. Set to 2 only if you accept potential data loss. |
| `sync_binlog` | 1 | Sync binlog on every commit. Required for crash-safe replication. |
| `max_connections` | Actual need + 20% headroom | Do not set to instance max unless needed |

## Read Replicas and Database Proxy

### Adding read replicas

![Database proxy read/write split](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_proxy_architecture.png)

![Read replica binlog replication architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_replica_flow.png)

![RDS high-availability architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_ha_architecture.png)

A read replica is an asynchronous copy of the primary instance. It handles SELECT queries, offloading read traffic from the primary. For read-heavy workloads (most web applications), this is the primary scaling mechanism.

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

The replica creation process:

1. RDS takes an internal snapshot of the primary
2. A new instance is provisioned with the snapshot data
3. Binary log replication is configured automatically
4. The replica becomes available (minutes for small databases, hours for large ones)

Check replication status:

```sql
-- On the read replica, check replication lag
SHOW SLAVE STATUS\G

-- Key fields to watch:
-- Seconds_Behind_Master: should be 0 or very low
-- Slave_IO_Running: Yes
-- Slave_SQL_Running: Yes
```

### Database Proxy

Managing read replicas manually is painful. Your application needs to know which endpoint is the primary (for writes) and which are replicas (for reads), and it needs to route queries accordingly. Database Proxy solves this.

Database Proxy is a built-in proxy layer that sits between your application and the RDS instances. It provides:

- **Automatic read/write splitting** — `INSERT/UPDATE/DELETE` goes to the primary, `SELECT` goes to replicas
- **Connection pooling** — reduces connection overhead on the database
- **Single endpoint** — your application connects to one address, the proxy handles routing

Enable Database Proxy:

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

Configure read/write splitting:

```bash
# Enable read/write splitting with weight distribution
aliyun rds ModifyReadWriteSplittingConnection \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --ConnectionStringPrefix "myapp-proxy" \
  --DistributionType Standard \
  --MaxDelayTime 10 \
  --Weight '{"rm-bp1xxxxxxxxx":"0","rr-bp1yyyyyyyyy":"100"}'
```

The weight configuration means:

- Primary (`rm-bp1xxxxxxxxx`): weight 0 — no reads go to primary (writes still go here automatically)
- Replica (`rr-bp1yyyyyyyyy`): weight 100 — all reads go to replica

If the replica's replication lag exceeds `MaxDelayTime` (10 seconds), the proxy automatically routes reads back to the primary until the replica catches up.

Your application connection string changes to use the proxy endpoint:

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

No application code changes needed. The proxy handles all routing transparently.

## Backup and Recovery

Backups are the one thing you cannot skip. When everything else fails — bad deploy, data corruption, accidental `DROP TABLE` — backups are your last line of defense.

![Backup timeline with point-in-time recovery](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_backup_timeline.png)

### Automatic backups

RDS performs automatic backups on a configurable schedule. By default:

- **Full backup** every day during a low-traffic window
- **Binary log backup** continuously (for point-in-time recovery)
- **Retention period**: 7 days (configurable up to 730 days)

Configure the backup schedule:

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

Automatic backups do not impact instance performance significantly. RDS uses snapshot-based backup technology that completes quickly even for large databases.

### Manual backups

Before any risky operation (schema migration, bulk data import, version upgrade), take a manual backup:

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

### Point-in-time recovery (PITR)

This is the most important backup feature. PITR lets you restore your database to any second within the backup retention period. If someone runs `DELETE FROM users WHERE 1=1` at 14:32:07, you can restore to 14:32:06.

PITR works by combining the most recent full backup with binary logs to replay changes up to the specified point in time.

![Point-in-time recovery flow combining daily snapshots with continuous binlog replay](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_pitr_flow.png)

*PITR always provisions a new instance from the snapshot, then replays binlog up to the chosen second — the source database is never touched.*

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

PITR restores always create a new instance. They never overwrite the existing instance. This is by design — you verify the restored data on the new instance, then swap it in if everything looks correct.

### Cross-region backup

For disaster recovery, enable cross-region backup. This replicates backups to a different region, so even if the entire primary region goes down, your data is safe:

```bash
# Enable cross-region backup
aliyun rds ModifyInstanceCrossBackupPolicy \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --CrossBackupType 1 \
  --CrossBackupRegion cn-shanghai \
  --RetentType 1 \
  --Retention 30
```

Cross-region backup adds cost (data transfer + storage in the secondary region), but it is the only thing that protects against a full regional failure. For production databases with any business-critical data, enable it.

## Monitoring and Performance

### CloudMonitor metrics

![RDS monitoring dashboard metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_monitoring_metrics.png)

RDS automatically reports metrics to CloudMonitor. The critical ones to watch:

![The four critical RDS metrics shown as gauge charts with action thresholds](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_four_metrics.png)

*Set alerts at 80% on each gauge — if you wait until they hit 100% you are already shipping errors to users.*

| Metric | Warning Threshold | Critical Threshold | What to do |
|---|---|---|---|
| CPU Utilization | > 70% sustained | > 90% sustained | Upgrade instance type or optimize queries |
| IOPS Usage | > 70% of limit | > 90% of limit | Upgrade storage PL or optimize I/O patterns |
| Connection Usage | > 70% of max | > 90% of max | Add connection pooling, increase max_connections |
| Disk Usage | > 70% | > 85% | Expand storage immediately |
| Replication Lag | > 5 seconds | > 30 seconds | Check replica performance, reduce write load |

Set up alerts:

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

### DAS: Database Autonomy Service

DAS is Alibaba Cloud's intelligent diagnostics tool for RDS. It goes beyond raw metrics and provides:

- **Automatic SQL diagnostics** — identifies slow queries and suggests index optimizations
- **Real-time session analysis** — shows active sessions, lock waits, and blocking queries
- **Performance trend analysis** — tracks performance metrics over time to detect degradation
- **Automatic optimization** — can apply recommended index changes automatically (opt-in)

DAS is enabled by default for RDS instances. Access it through the RDS console under "Autonomy Service."

### Slow query analysis

Slow queries are the number one cause of database performance issues. RDS logs all queries exceeding `long_query_time` (we set it to 1 second earlier).

![Decision tree for investigating a slow query starting from EXPLAIN output](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_slow_query_flow.png)

*Always start with EXPLAIN — the `type` column tells you 80% of what you need to know about whether the query plan is healthy.*

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

Download slow query logs for offline analysis:

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

### Index recommendations

The simplest performance optimization is adding the right indexes. Use `EXPLAIN` to understand how MySQL executes a query:

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

## Security

### IP whitelist

The RDS IP whitelist is the first line of defense. It controls which IP addresses can connect to the database at the network level, before any authentication happens.

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

Never add `0.0.0.0/0` to the whitelist. This opens the database to all IPs — including the entire internet if the public endpoint is enabled. Even with strong passwords, brute force attacks will hammer your database.

### SSL encryption

Enable SSL to encrypt data in transit between your application and the database:

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

Configure your application to use SSL:

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

TDE encrypts data at rest on disk. Even if someone gains access to the underlying storage, the data is unreadable without the encryption key.

```bash
# Enable TDE
aliyun rds ModifyDBInstanceTDE \
  --DBInstanceId rm-bp1xxxxxxxxx \
  --TDEStatus Enabled \
  --EncryptionKey "kms-key-id-from-kms"
```

TDE uses AES-256 encryption and integrates with KMS (Key Management Service) for key management. Once enabled, TDE cannot be disabled. The performance overhead is minimal (< 5% for most workloads).

### SQL Audit

SQL Audit logs every SQL statement executed on the database. This is essential for compliance, security investigation, and debugging.

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

SQL Audit stores logs for up to 30 days. For longer retention, export to SLS (Simple Log Service) for long-term storage and analysis.

### Secure connection from ECS

The complete security chain for connecting to RDS from your application:

![Five concentric layers of RDS security defense from VPC isolation to TDE encryption](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/05-rds-database/05_security_layers.png)

*Defense in depth — an attacker who breaches one layer still has to defeat the next four before touching real data.*

1. **VPC isolation** — RDS and ECS are in the same VPC, different VSwitches (data tier vs app tier)
2. **Security groups** — ECS security group allows outbound 3306; RDS whitelist allows inbound from app tier CIDR
3. **SSL encryption** — data in transit is encrypted
4. **Least-privilege accounts** — application uses a Normal account with ReadWrite on specific databases only
5. **No public endpoint** — the database is not accessible from the internet

```bash
# Verify: from app-tier ECS, connection works
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p --ssl-ca=/path/to/ca.pem -e "SELECT 1"

# Verify: from web-tier ECS (different security group), connection fails
mysql -h rm-bp1xxx.mysql.rds.aliyuncs.com -u appuser -p -e "SELECT 1"
# ERROR 2003 (HY000): Can't connect to MySQL server (connection refused / timeout)
```

## Solution: Production Database with HA and Read Scaling

Let us build a complete production database setup from scratch: RDS MySQL HA with one read replica, Database Proxy for read/write splitting, automated backups, and monitoring. This is the reference architecture for a medium-to-large web application.

### Step 1: Create the RDS MySQL HA instance

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

### Step 2: Create database and accounts

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

### Step 3: Configure production parameters

```bash
# Apply production-tuned parameters
aliyun rds ModifyParameter \
  --DBInstanceId $RDS_ID \
  --Parameters "slow_query_log:ON;long_query_time:1;innodb_flush_log_at_trx_commit:1;sync_binlog:1;max_connections:500;innodb_print_all_deadlocks:ON;log_queries_not_using_indexes:ON"
```

### Step 4: Create a read replica

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

### Step 5: Enable Database Proxy for read/write splitting

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

### Step 6: Configure backups

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

### Step 7: Set up monitoring alerts

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

### Step 8: Verify the complete setup

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

From an ECS instance in the app tier VPC, test the complete chain:

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

### Architecture summary

What we built:

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
              └───────┬───────┘  └──────────────────┘
                      │
              ┌───────▼───────┐
              │   Standby     │
              │   (AZ-I)      │
              │   semi-sync   │
              └───────────────┘
```

The primary handles all writes. The standby (invisible, managed by RDS) takes over if the primary fails. The read replica handles all read queries, offloading the primary. The Database Proxy routes traffic automatically. Backups run daily with 14-day retention. Cross-region backup to cn-shanghai provides disaster recovery. SSL encrypts all connections. Monitoring alerts fire before problems become outages.

Total monthly cost for this setup (approximate):

- Primary (rds.mysql.s3.large, 200GB ESSD): ~600 CNY
- Standby: included with HA edition
- Read replica (rds.mysql.s2.large, 200GB ESSD): ~350 CNY
- Database Proxy: ~100 CNY
- Cross-region backup storage: ~50 CNY
- **Total: ~1,100 CNY/month** (~$150 USD)

For a production database with HA, read scaling, automated backup, and cross-region DR, that is remarkably inexpensive.

## Key Takeaways

**Use managed databases.** Unless you have a very specific reason to self-manage (custom plugins, exotic replication topologies, regulatory requirements that mandate OS-level control), RDS saves you from an enormous operational burden.

**Start with RDS, upgrade to PolarDB when needed.** RDS handles most workloads under 10 TB. PolarDB is the upgrade path when you need more than 5 read replicas, auto-scaling storage, or sub-10ms replication lag.

**Size by buffer pool, not by CPU.** The single most important performance factor is whether your working dataset fits in the InnoDB buffer pool. Size your instance's memory for that, not for CPU.

**Use Database Proxy for read/write splitting.** Manually routing reads and writes in application code is error-prone and inflexible. Database Proxy handles it transparently with a single connection endpoint.

**Backups are not optional.** Enable automatic daily backups, set retention to at least 14 days, enable binary log backup for PITR, and set up cross-region backup for disaster recovery. Test your restore process at least quarterly.

**Monitor the four critical metrics.** CPU utilization, disk usage, connection usage, and replication lag. Set alerts at 80% thresholds. By the time you notice problems manually, your users have been suffering for hours.

**Lock down access.** IP whitelist to app tier CIDRs only, no public endpoint, SSL encryption, least-privilege accounts. The database contains your most valuable asset — treat its security accordingly.

## What's Next

The database is running, replicated, backed up, and monitored. But a production application needs more than just a relational database. In the next article, we cover the caching and storage layer — Tair (managed Redis) for session storage and caching, OSS for object storage, and NAS for shared file systems. These are the services that sit alongside your database to handle the workloads that relational databases are not built for.
