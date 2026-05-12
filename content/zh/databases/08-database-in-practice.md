---
title: "数据库（八）：实战中的数据库——迁移、监控与故障案例"
date: 2024-04-30 09:00:00
tags:
  - Databases
  - Operations
  - Monitoring
  - Migration
  - Cloud
categories: Databases
series: databases
lang: zh
description: "数据库的运维实践——模式迁移、连接池、监控、备份策略、托管数据库选型，以及来自生产环境的血泪教训。"
disableNunjucks: true
series_order: 8
translationKey: "databases-8"
---

理解数据库内部原理只是成功的一半；另一半，是在生产环境中持续稳定运行它——不丢数据、不掉可用性、更别在凌晨三点被告警叫醒。本文聚焦于那些只能靠实战积累的运维知识：没人会在出事前告诉你，但一旦出事，你立刻就需要它们。

## 模式迁移：边飞行边换引擎

你的数据库模式一定会变：新功能需要新字段、新表、新索引。真正的挑战在于：如何在零停机的前提下完成演进？

![Schema evolution strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-schema-evolution.png)


### 迁移工具对比

| 工具 | 语言 | 支持数据库 | 核心特性 |
|------|----------|-----------------|-------------|
| Flyway | Java （提供 CLI） | PostgreSQL, MySQL, Oracle, SQL Server | SQL + Java 迁移脚本，版本追踪 |
| Liquibase | Java （提供 CLI） | PostgreSQL, MySQL, Oracle, SQL Server | XML/YAML/JSON 变更日志，支持回滚 |
| golang-migrate | Go | PostgreSQL, MySQL, SQLite, 更多 | CLI + 库集成，支持 up/down 迁移 |
| Alembic | Python （SQLAlchemy） | 所有 SQLAlchemy 支持的数据库 | 基于模型自动生成迁移 |
| Prisma Migrate | TypeScript | PostgreSQL, MySQL, SQLite, MongoDB | Schema-first，自动生成 SQL |
| dbmate | Go | PostgreSQL, MySQL, SQLite, ClickHouse | 简洁、无框架依赖 |

### 迁移文件结构示例

```
migrations/
├── 001_create_users.up.sql
├── 001_create_users.down.sql
├── 002_create_orders.up.sql
├── 002_create_orders.down.sql
├── 003_add_email_index.up.sql
├── 003_add_email_index.down.sql
├── 004_add_phone_to_users.up.sql
└── 004_add_phone_to_users.down.sql
```

```sql
-- 004_add_phone_to_users.up.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
CREATE INDEX idx_users_phone ON users (phone);

-- 004_add_phone_to_users.down.sql
DROP INDEX idx_users_phone;
ALTER TABLE users DROP COLUMN phone;
```

```bash
# golang-migrate 示例
# 应用所有待执行迁移
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" up

# 回滚上一次迁移
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" down 1

# 查看当前版本
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" version
```

```bash
# Alembic（Python）示例
# 基于模型变更自动生成迁移
alembic revision --autogenerate -m "add phone to users"

# 应用所有迁移至最新版
alembic upgrade head

# 回滚一步
alembic downgrade -1

# 查看当前版本
alembic current
```

### 迁移最佳实践

#### 1. 向后兼容性（Backward Compatibility）

**永远不要破坏正在运行的应用程序。** 迁移必须同时兼容旧代码和新代码：

```
部署时间线：
  1. 执行迁移（新增字段，设为 NULLABLE）
  2. 部署新代码（开始写入新字段）
  3. 补全历史数据（backfill）
  4. 执行第二步迁移（添加 NOT NULL 约束）
```

❌ 错误做法（立即破坏旧代码）：

```sql
-- 此操作会立刻导致旧代码报错（因不认识该字段）
ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL;
```

✅ 正确做法（兼容新旧代码）：

```sql
-- 第一步：添加可空字段（旧代码忽略该字段）
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 第二步：部署新代码，开始写入 phone 字段

-- 第三步：补全已有行
UPDATE users SET phone = 'unknown' WHERE phone IS NULL;

-- 第四步：添加非空约束
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
```

#### 2. 在线 DDL （Online DDL）

某些 DDL 操作会锁住整张表，阻塞读写 —— 对于一亿行的表，这可能意味着数分钟不可用。

✅ 安全操作（无锁或极短锁， PostgreSQL）：

```sql
-- 添加可空字段：瞬时完成（无需重写表）
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 添加带常量默认值的字段（PostgreSQL 11+）：瞬时完成
ALTER TABLE users ADD COLUMN active BOOLEAN DEFAULT TRUE;

-- 并发创建索引：不锁表
CREATE INDEX CONCURRENTLY idx_users_email ON users (email);
-- 注意：耗时更长，但不阻塞写入
```

❌ 危险操作（全表重写或长时间锁）：

```sql
-- 危险：修改列类型将重写整张表
ALTER TABLE users ALTER COLUMN phone TYPE TEXT;

-- MySQL 中危险：未指定 ALGORITHM=INPLACE 的建索引
ALTER TABLE users ADD INDEX idx_email (email);
-- 应改用：
ALTER TABLE users ADD INDEX idx_email (email), ALGORITHM=INPLACE, LOCK=NONE;

-- 危险：对已存在列添加 NOT NULL 默认值（旧版 PostgreSQL）
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
-- 安全替代方案：先加 CHECK 约束再验证
ALTER TABLE users ADD CONSTRAINT users_phone_not_null
    CHECK (phone IS NOT NULL) NOT VALID;
ALTER TABLE users VALIDATE CONSTRAINT users_phone_not_null;
```

MySQL 生态中，`pt-online-schema-change`（Percona）和 `gh-ost`（GitHub）通过影子表（shadow table）实现在线变更：复制数据、捕获 binlog 变更、原子切换：

```bash
# gh-ost：MySQL 在线模式变更
gh-ost \
  --host=localhost \
  --database=mydb \
  --table=users \
  --alter="ADD COLUMN phone VARCHAR(20)" \
  --execute

# gh-ost 执行流程：
# 1. 创建 _users_gho（ghost 表），含新 schema
# 2. 创建 _users_ghc（changelog 表）
# 3. 分批拷贝现有数据
# 4. 通过 binlog 捕获实时变更
# 5. 原子重命名：users → _users_old，_users_gho → users
```

#### 3. 零停机重命名策略

重命名字段极具挑战：旧代码引用旧名，新代码引用新名：

```sql
-- 第一步：添加新字段
ALTER TABLE users ADD COLUMN full_name VARCHAR(200);

-- 第二步：双写（应用同时写入 name 和 full_name）

-- 第三步：补全历史数据
UPDATE users SET full_name = name WHERE full_name IS NULL;

-- 第四步：读取切到新字段（部署只读 full_name 的代码）

-- 第五步：停止写入旧字段（部署仅写 full_name 的代码）

-- 第六步：删除旧字段
ALTER TABLE users DROP COLUMN name;
```

## 连接池（Connection Pooling）

每个数据库连接都消耗资源：内存（PostgreSQL 中约 5–10 MB/连接）、文件描述符、 CPU （用于进程/线程管理）。若无连接池，应用实例突发增长极易耗尽数据库连接上限。

![Connection pooling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-connection-pooling.png)


### 问题场景

```
无连接池：
  100 台应用服务器 × 每台 20 线程 = 2,000 数据库连接
  PostgreSQL 默认 max_connections = 100
  结果：connection refused，应用崩溃

启用连接池（如 PgBouncer）：
  100 台应用服务器 × 每台 20 线程 = 2,000 应用连接
  PgBouncer 维持 50 个真实数据库连接
  复用比（multiplexing ratio）：40:1
```

### PgBouncer （PostgreSQL）

```ini
; pgbouncer.ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

; 连接池配置
default_pool_size = 20        ; 每用户/数据库对的连接数
max_client_conn = 1000        ; PgBouncer 最大客户端连接数
min_pool_size = 5             ; 至少保持 5 个空闲连接
reserve_pool_size = 5         ; 预留连接应对突发流量

; 池模式
pool_mode = transaction
; session    = 连接绑定整个客户端会话（最安全，效率最低）
; transaction = 每事务结束后归还连接（平衡之选）
; statement   = 每语句结束后归还连接（最高效，功能受限）
```

```bash
# 连接 PgBouncer 管理控制台
psql -h localhost -p 6432 -U pgbouncer pgbouncer

# 查看连接池状态
SHOW POOLS;
#  database |   user    | cl_active | cl_waiting | sv_active | sv_idle | sv_used
# ----------+-----------+-----------+------------+-----------+---------+--------
#  mydb     | app_user  |        45 |          0 |        18 |       2 |       0

# 查看客户端连接
SHOW CLIENTS;

# 查看服务端（数据库）连接
SHOW SERVERS;

# 查看统计信息
SHOW STATS;
```

### ProxySQL （MySQL）

```sql
-- ProxySQL 管理接口（端口 6032）
-- 配置后端服务器
INSERT INTO mysql_servers (hostgroup_id, hostname, port, max_connections)
VALUES
  (10, 'mysql-leader', 3306, 100),    -- 写节点主机组
  (20, 'mysql-replica1', 3306, 100),  -- 读节点主机组
  (20, 'mysql-replica2', 3306, 100);  -- 读节点主机组

-- 配置查询路由规则
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup)
VALUES
  (1, 1, '^SELECT .* FOR UPDATE', 10),  -- SELECT FOR UPDATE → 主节点
  (2, 1, '^SELECT', 20),                -- 其他所有 SELECT → 从节点
  (3, 1, '.*', 10);                     -- 其余全部 → 主节点

-- 加载并持久化配置
LOAD MYSQL SERVERS TO RUNTIME;
LOAD MYSQL QUERY RULES TO RUNTIME;
SAVE MYSQL SERVERS TO DISK;
SAVE MYSQL QUERY RULES TO DISK;
```

### 应用层连接池

```python
# Python：SQLAlchemy 连接池
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost:5432/mydb",
    pool_size=20,           # 持久连接数
    max_overflow=10,        # 突发流量时额外连接数
    pool_timeout=30,        # 获取连接超时（秒）
    pool_recycle=3600,      # 连接复用 1 小时后回收
    pool_pre_ping=True,     # 使用前检测连接有效性
)
```

```java
// Java：HikariCP（最快的 JDBC 连接池）
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
config.setUsername("user");
config.setPassword("pass");
config.setMaximumPoolSize(20);
config.setMinimumIdle(5);
config.setConnectionTimeout(30000);  // 30 秒
config.setIdleTimeout(600000);       // 10 分钟
config.setMaxLifetime(1800000);      // 30 分钟
config.setLeakDetectionThreshold(60000); // 连接持有超 60 秒则告警

HikariDataSource ds = new HikariDataSource(config);
```

## 数据库监控


![Database monitoring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-monitoring-dashboard.png)


![Database monitoring dashboard control room with holographic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/08-database-monitoring-dashboard-control-room-with-holographic-.jpg)

### 关键指标

| 指标 | 健康范围 | 异常时行动建议 |
|--------|--------------|-------------------|
| **QPS**（每秒查询数） | 基线 ± 20% | 排查突增或骤降原因 |
| **查询延迟 p99** | OLTP 场景 < 100 ms | 定位并优化慢查询 |
| **活跃连接数** | < 80% max_connections | 增加连接池大小或 max_connections |
| **复制延迟** | < 1 秒 | 检查副本负载、网络状况 |
| **缓冲池命中率** | > 99% | 增大缓冲池或缩小热数据集 |
| **磁盘 I/O 等待** | < 10% | 升级存储或增加 RAM 缓存 |
| **锁等待比例** | < 5% 事务数 | 缩小事务作用域 |
| **死锁/秒** | < 1 | 修复应用中锁获取顺序 |
| **WAL 生成速率** | 基线 ± 30% | 检查写风暴 |
| **表膨胀率**（PostgreSQL） | < 20% 死元组 | 调优 autovacuum |

### PostgreSQL 监控查询

```sql
-- 当前活动会话：此刻正在运行什么？
SELECT
    pid,
    usename,
    state,
    wait_event_type,
    wait_event,
    age(now(), query_start) AS query_duration,
    left(query, 80) AS query_preview
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- 表级统计：哪些表最“热”？
SELECT
    schemaname,
    relname AS table_name,
    seq_scan,
    idx_scan,
    n_tup_ins AS inserts,
    n_tup_upd AS updates,
    n_tup_del AS deletes,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows,
    ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 1) AS dead_pct,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;

-- 索引使用率：索引是否真被用了？
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC
LIMIT 20;
-- 扫描次数为 0 的索引可考虑删除

-- 缓存命中率
SELECT
    'index hit rate' AS name,
    ROUND(sum(idx_blks_hit)::numeric / NULLIF(sum(idx_blks_hit + idx_blks_read), 0), 4) AS ratio
FROM pg_statio_user_indexes
UNION ALL
SELECT
    'table hit rate',
    ROUND(sum(heap_blks_hit)::numeric / NULLIF(sum(heap_blks_hit + heap_blks_read), 0), 4)
FROM pg_statio_user_tables;
-- 两者均应 > 0.99

-- 数据库大小
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

### MySQL 监控查询

```sql
-- 当前进程列表
SHOW PROCESSLIST;

-- InnoDB 状态（全面诊断）
SHOW ENGINE INNODB STATUS\G

-- 关键 InnoDB 指标
SELECT
    variable_name,
    variable_value
FROM performance_schema.global_status
WHERE variable_name IN (
    'Innodb_buffer_pool_read_requests',
    'Innodb_buffer_pool_reads',
    'Innodb_rows_read',
    'Innodb_rows_inserted',
    'Innodb_rows_updated',
    'Innodb_rows_deleted',
    'Innodb_deadlocks',
    'Innodb_row_lock_waits',
    'Innodb_row_lock_time_avg',
    'Threads_connected',
    'Threads_running',
    'Slow_queries',
    'Questions'
);
```

## 慢查询分析


![Migration workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-migration-workflow.png)


![Database migration journey old schema transforming into new](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/08-database-migration-journey-old-schema-transforming-into-new-.jpg)

### 启用慢查询日志

```bash
# PostgreSQL：postgresql.conf
log_min_duration_statement = 100  # 记录 > 100ms 的查询
log_statement = 'none'            # 不记录所有语句（太嘈杂）
auto_explain.log_min_duration = '200ms'  # 对慢查询自动 EXPLAIN
auto_explain.log_analyze = on
```

```bash
# MySQL：my.cnf
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 0.1  # 记录 > 100ms 的查询
log_queries_not_using_indexes = 1
```

### 查询优化工作流

```
步骤 1：定位慢查询  
  → 慢日志、pg_stat_statements 或监控工具  

步骤 2：执行 EXPLAIN ANALYZE  
  → 理解执行计划  

步骤 3：识别瓶颈  
  → 全表扫描？缺索引？连接顺序错误？大排序？  

步骤 4：尝试修复  
  → 加索引、重写查询、调参数  

步骤 5：再次 EXPLAIN ANALYZE 验证  
  → 确认性能提升  

步骤 6：生产环境持续观测  
  → 防止回归
```

```sql
-- PostgreSQL：pg_stat_statements（Top 慢查询）
-- 启用：shared_preload_libraries = 'pg_stat_statements'

SELECT
    calls,
    ROUND(total_exec_time::numeric, 2) AS total_ms,
    ROUND(mean_exec_time::numeric, 2) AS avg_ms,
    ROUND((stddev_exec_time)::numeric, 2) AS stddev_ms,
    rows,
    left(query, 100) AS query_preview
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;
```

## 备份策略

### 逻辑备份（Logical Backups）

逻辑备份导出为 SQL 语句或结构化数据文件。

![Backup strategy comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-backup-strategy.png)


```bash
# PostgreSQL：pg_dump
# 单库压缩备份
pg_dump -Fc -h localhost -U postgres mydb > mydb_$(date +%Y%m%d).dump

# 仅备份特定表
pg_dump -Fc -t users -t orders mydb > tables_$(date +%Y%m%d).dump

# 仅备份 schema（不含数据）
pg_dump -s mydb > schema.sql

# 恢复
pg_restore -h localhost -U postgres -d mydb mydb_20231228.dump

# 全部数据库
pg_dumpall -h localhost -U postgres > all_databases.sql
```

```bash
# MySQL：mysqldump
# 单库一致性快照备份
mysqldump --single-transaction --routines --triggers \
  -h localhost -u root -p mydb > mydb_$(date +%Y%m%d).sql

# 压缩备份
mysqldump --single-transaction mydb | gzip > mydb_$(date +%Y%m%d).sql.gz

# 恢复
mysql -h localhost -u root -p mydb < mydb_20231228.sql
# 或解压后恢复：
gunzip < mydb_20231228.sql.gz | mysql -h localhost -u root -p mydb
```

### 物理备份（Physical Backups）

物理备份直接拷贝数据文件，大型数据库下速度远超逻辑备份。

```bash
# PostgreSQL：pg_basebackup
pg_basebackup -h localhost -U repl_user -D /backup/base \
  --checkpoint=fast --wal-method=stream -P

# 创建完整数据目录副本（含 WAL 文件，保障一致性恢复）

# 恢复：停库 → 替换数据目录 → 启动
```

```bash
# MySQL：Percona XtraBackup
# 全量备份
xtrabackup --backup --target-dir=/backup/full \
  --user=root --password=xxx

# 准备（应用 redo 日志）
xtrabackup --prepare --target-dir=/backup/full

# 恢复：停 MySQL → 拷贝文件 → 修权限 → 启动
xtrabackup --copy-back --target-dir=/backup/full
chown -R mysql:mysql /var/lib/mysql
```

### 时间点恢复（Point-in-Time Recovery, PITR）

PITR 允许恢复到任意时刻，不止是备份快照时间点。

```bash
# PostgreSQL PITR
# 1. 创建基础备份
pg_basebackup -D /backup/base --checkpoint=fast

# 2. 持续归档 WAL 段
# postgresql.conf：
# archive_mode = on
# archive_command = 'cp %p /backup/wal/%f'

# 3. 恢复到指定时间点：
# recovery.conf（PG 12+ 为 postgresql.conf）：
# restore_command = 'cp /backup/wal/%f %p'
# recovery_target_time = '2023-12-28 14:30:00 UTC'
# recovery_target_action = 'promote'
```

### 备份方式对比

| 维度 | 逻辑备份（pg_dump） | 物理备份（pg_basebackup） | PITR |
|--------|-------------------|-------------------------|------|
| 备份速度 | 慢（SQL 层读取全量数据） | 快（文件拷贝） | 持续（WAL 流式归档） |
| 恢复速度 | 慢（重放 SQL） | 快（文件拷贝） | 中等（基础备份 + WAL 重放） |
| 存储大小 | 较小（压缩 SQL） | 较大（完整数据目录） | 基础备份 + WAL 文件 |
| 粒度 | 支持按表备份 | 仅支持集群级 | 任意时间点 |
| 版本兼容性 | 跨版本兼容 | 需同主版本 | 需同主版本 |
| 部分恢复 | 支持（指定表） | 不支持 | 不支持 |

### 备份验证（Backup Testing）

**从未验证过的备份，不是备份，只是希望。** 务必定期执行恢复测试：

```bash
#!/bin/bash
# 月度备份恢复测试
set -e

echo "Starting backup restore test: $(date)"

# 创建测试库
createdb restore_test

# 恢复最新备份
pg_restore -d restore_test /backup/latest/mydb.dump

# 运行校验查询
psql -d restore_test -c "SELECT COUNT(*) FROM users;" | grep -q "[0-9]"
psql -d restore_test -c "SELECT COUNT(*) FROM orders;" | grep -q "[0-9]"

# 对比生产环境行数
PROD_USERS=$(psql -d mydb -t -c "SELECT COUNT(*) FROM users;")
TEST_USERS=$(psql -d restore_test -t -c "SELECT COUNT(*) FROM users;")

if [ "$PROD_USERS" != "$TEST_USERS" ]; then
    echo "ALERT: Row count mismatch! Prod=$PROD_USERS, Restored=$TEST_USERS"
    exit 1
fi

# 清理
dropdb restore_test

echo "Backup restore test passed: $(date)"
```

## 托管数据库选型（Managed Database）

自建数据库虽有助学习，但运维成本高昂。托管服务自动处理补丁、备份、复制与监控：

| 服务 | 提供商 | 支持引擎 | 核心特性 |
|---------|---------|---------------|-------------|
| RDS | 阿里云 / AWS | MySQL, PostgreSQL, SQL Server, MariaDB | 自动备份、多可用区 |
| PolarDB | 阿里云 | MySQL, PostgreSQL, Oracle 兼容 | 共享存储、最大 100 TB、读扩展 |
| AnalyticDB | 阿里云 | MySQL 兼容（OLAP） | PB 级分析、列式存储 |
| Aurora | AWS | MySQL, PostgreSQL 兼容 | 6 节点复制、自动扩缩容存储 |
| Cloud SQL | Google Cloud | MySQL, PostgreSQL, SQL Server | 自动故障转移、 IAM 集成 |
| AlloyDB | Google Cloud | PostgreSQL 兼容 | 列式引擎加速分析 |

### 自建 vs 托管决策指南

| 自建适用场景 | 托管适用场景 |
|-----------------|-----------------|
| 需要特定扩展或定制补丁 | 标准配置即可满足需求 |
| 极致性能调优要求 | 运维简洁性优先 |
| 超大规模下成本敏感 | 团队精简，无专职 DBA |
| 合规要求完全掌控基础设施 | 云厂商满足合规认证 |
| 教学/学习目的 | 生产级工作负载 |

## 容量规划（Capacity Planning）

### 垂直扩展（Scale Up） vs 水平扩展（Scale Out）

```
垂直扩展（Scale UP）：
  同一服务器升级 CPU、RAM、更快存储
  + 简单（无需应用改造）
  + 全功能支持（事务、JOIN 等）
  - 有天花板（最大实例规格限制）
  - 单点故障（无复制时）

水平扩展（Scale OUT）：
  新增服务器（复制 + 分片）
  + 无上限（按需加节点）
  + 内置冗余
  - 应用复杂（查询路由、分布式事务）
  - 非所有操作跨分片有效（如跨分片 JOIN）
```

### 容量估算示例

```
示例：SaaS 应用数据库容量估算

用户数：100 万  
平均行大小：200 字节  
表清单：users（100 万行）、orders（1000 万）、order_items（3000 万）、products（10 万）、sessions（500 万）

数据体积：
  users:       1,000,000 × 200 B = 200 MB  
  orders:     10,000,000 × 150 B = 1.5 GB  
  order_items: 30,000,000 × 100 B = 3.0 GB  
  products:       100,000 × 500 B = 50 MB  
  sessions:    5,000,000 × 1 KB  = 5.0 GB  
  总计数据：≈ 10 GB  

索引开销：≈ 数据量 30–50% = 3–5 GB  
WAL/binlog：≈ 2–5 GB（滚动）  
热数据集（working set）：约 20% = 2 GB  

推荐配置：
  内存：16 GB（容纳全部数据于 buffer pool）  
  存储：50 GB SSD（预留增长 + 备份空间）  
  CPU：4 核（支撑数百 QPS）
```

## 故障案例（War Stories）：常见生产事故

| 事故 | 根因 | 表象 | 修复 | 预防 |
|----------|-----------|---------|-----|------------|
| **死亡查询** | 新功能缺失索引 | CPU 100%，所有查询变慢 | `CREATE INDEX CONCURRENTLY` | 代码审查强制要求新查询附带 EXPLAIN |
| **连接耗尽** | 无连接池，应用扩至 50 个 Pod | “too many connections” 报错 | 加 PgBouncer，降低每 Pod 连接池大小 | 始终启用连接池 |
| **复制中断** | 主库 DDL 未设 `SET STATEMENT_FORMAT=ROW` | 从库数据漂移、读取陈旧 | 重新拉取从库快照 | 统一 binlog 格式为 ROW |
| **磁盘满** | 闲置复制槽（replication slot）导致 WAL 无限增长 | 数据库拒绝写入 | `SELECT pg_drop_replication_slot('dead_slot')` | 监控 `pg_replication_slots`，对非活跃槽告警 |
| **表膨胀** | Autovacuum 无法跟上高频 UPDATE 表 | 查询逐次变慢 | 手动 `VACUUM FULL`（锁表） | 按表调优 `autovacuum_vacuum_scale_factor` |
| **锁队列** | 长迁移持有排他锁 | 所有查询排队等待 | 杀掉迁移，重试并设 `lock_timeout` | 迁移设置 `lock_timeout = '5s'` |
| **OOM Killer** | 内存中巨型排序（未限制 work_mem） | PostgreSQL 进程被杀，连接断开 | 设 `work_mem = '256MB'`，加索引 | 配置 `work_mem` 与 `temp_file_limit` |
| **级联故障** | DB 变慢 → 应用重试 ×3 → 负载 ×3 → DB 更慢 | 全站宕机 | 熔断器介入，终止重试 | 实现熔断器，设置查询超时 |
| **数据损坏** | 为“性能”关闭 `fsync=off` | 断电后静默丢数据 | 从备份恢复 | 生产环境永不关闭 fsync |
| **迁移失败** | 5 亿行表上 `ALTER TABLE` 超时 | 锁卡住，写入阻塞 20 分钟 | 改用 `gh-ost` / `pt-online-schema-change` | 大表迁移必须用在线 DDL 工具 |

### 黄金守则（The Golden Rules）

1. **永远有备份，并验证恢复流程。** 从未恢复过的备份，只是幻想，不是策略。

2. **监控先行。** 上线首个生产用户前，就应建好仪表盘与告警。关键指标：连接数、查询延迟 p99、复制延迟、磁盘使用率。

3. **始终启用连接池。** 即使你的框架声称“内置连接池”，也请独立部署专业连接池（如 PgBouncer / ProxySQL）。

4. **处处设超时。** 语句超时、锁超时、连接超时、空闲事务超时。没有超时的查询，终将永久持有锁。

```sql
-- PostgreSQL：设置安全超时
ALTER DATABASE mydb SET statement_timeout = '30s';
ALTER DATABASE mydb SET lock_timeout = '10s';
ALTER DATABASE mydb SET idle_in_transaction_session_timeout = '60s';
```

5. **绝不未经测试就在生产执行迁移。** 每次迁移必须在生产数据副本上实测：耗时多少？是否加锁？影响范围？

6. **只读副本不能解决写瓶颈。** 它们只加速读请求。若写是瓶颈，你需要分片（sharding）。

7. **事务务必短小。** 一个持有锁 30 秒的事务，会阻塞所有后续需要这些行的事务。

8. **数据库不是消息队列。** 如果你在轮询 `SELECT ... WHERE status = 'pending' FOR UPDATE SKIP LOCKED`，你应该用真正的消息队列（如 Kafka、 RabbitMQ）。

## 系列结语

八篇文章，我们从关系模型与 SQL 基础出发，一路抵达分布式事务与生产运维实践。这条路径并非偶然：你不理解隔离级别，就无法明白为何复制延迟如此关键；你不理解两阶段提交（2PC）的阻塞本质，就无法真正欣赏 Saga 模式的精妙。

数据库正是这样一个领域：浅层认知极其危险。不了解索引的开发者，会构建出在小数据集上完美运行、却在生产中彻底崩塌的系统；不了解隔离级别的团队，会发布只有在高并发下才暴露的竞态 Bug；不验证备份的组织，会在最需要它时发现备份毫无价值。

底层原理变化甚微： B-tree、 WAL、 MVCC、共识算法，数十年来始终是核心基石。掌握它们一次，你面对 PostgreSQL、 CockroachDB、 DynamoDB，乃至未来任何新数据库，看到的都只是你早已理解思想的不同变体。