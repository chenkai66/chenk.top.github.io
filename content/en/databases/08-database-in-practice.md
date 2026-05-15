---
title: "Databases (8): Databases in Practice — Migration, Monitoring, and War Stories"
date: 2024-04-30 09:00:00
tags:
  - Databases
  - Operations
  - Monitoring
  - Migration
  - Cloud
categories: Databases
series: databases
lang: en
description: "The operational side of databases — schema migrations, connection pooling, monitoring, backup strategies, managed database options, and hard-won lessons from production incidents."
disableNunjucks: true
series_order: 8
series_total: 8
translationKey: "databases-8"
---

Knowing how databases work internally is half the battle. The other half is keeping them running in production without losing data, dropping availability, or waking up at 3 AM. This article covers the operational knowledge that comes from experience — the things nobody teaches you until something breaks.

---

## Schema Migrations: Changing the Engine While Flying

Your schema will change. New features require new columns, new tables, new indexes. The question is how to evolve the schema without downtime.

![Schema evolution strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-schema-evolution.png)


### Migration Tools

| Tool | Language | Database Support | Key Feature |
|------|----------|-----------------|-------------|
| Flyway | Java (CLI available) | PostgreSQL, MySQL, Oracle, SQL Server | SQL + Java migrations, version tracking |
| Liquibase | Java (CLI available) | PostgreSQL, MySQL, Oracle, SQL Server | XML/YAML/JSON changelogs, rollback support |
| golang-migrate | Go | PostgreSQL, MySQL, SQLite, more | CLI + library, up/down migrations |
| Alembic | Python (SQLAlchemy) | Any SQLAlchemy-supported DB | Auto-generation from models |
| Prisma Migrate | TypeScript | PostgreSQL, MySQL, SQLite, MongoDB | Schema-first, auto-generated SQL |
| dbmate | Go | PostgreSQL, MySQL, SQLite, ClickHouse | Simple, framework-agnostic |

![Migration workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-migration-workflow.png)


### Migration File Structure

```text
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
# golang-migrate example
# Apply all pending migrations
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" up

# Roll back the last migration
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" down 1

# Check current version
migrate -path ./migrations -database "postgresql://user:pass@localhost/mydb?sslmode=disable" version
```

```bash
# Alembic (Python) example
# Generate migration from model changes
alembic revision --autogenerate -m "add phone to users"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# Show current revision
alembic current
```

### Migration Best Practices

#### Backward Compatibility

Never break the current running application. This means migrations must be compatible with both the old and new code:

```text
Deployment timeline:
  1. Run migration (add column, make nullable)
  2. Deploy new code (writes to new column)
  3. Backfill old rows
  4. Run second migration (add NOT NULL constraint)
```

Bad (breaks old code):

```sql
-- This immediately breaks the old code that doesn't know about the column
ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL;
```

Good (compatible with old and new code):

```sql
-- Step 1: Add nullable column (old code ignores it)
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Step 2: Deploy new code that writes phone

-- Step 3: Backfill existing rows
UPDATE users SET phone = 'unknown' WHERE phone IS NULL;

-- Step 4: Add NOT NULL constraint
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
```

#### Online DDL

Some DDL operations lock the entire table, blocking reads and writes for the duration. On a 100 million row table, this can mean minutes of downtime.

Operations that are safe (no lock or very brief lock in PostgreSQL):

```sql
-- Adding a nullable column: instant (no table rewrite)
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Adding a column with a constant default (PostgreSQL 11+): instant
ALTER TABLE users ADD COLUMN active BOOLEAN DEFAULT TRUE;

-- Creating an index concurrently: no table lock
CREATE INDEX CONCURRENTLY idx_users_email ON users (email);
-- Note: takes longer but doesn't block writes
```

Operations that are dangerous (full table rewrite or extended lock):

```sql
-- DANGEROUS: Changing column type rewrites the entire table
ALTER TABLE users ALTER COLUMN phone TYPE TEXT;

-- DANGEROUS in MySQL: Adding index without ALGORITHM=INPLACE
ALTER TABLE users ADD INDEX idx_email (email);
-- Use instead:
ALTER TABLE users ADD INDEX idx_email (email), ALGORITHM=INPLACE, LOCK=NONE;

-- DANGEROUS: Adding NOT NULL with default to existing column (older PG versions)
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
-- Safe approach: add a CHECK constraint first
ALTER TABLE users ADD CONSTRAINT users_phone_not_null
    CHECK (phone IS NOT NULL) NOT VALID;
ALTER TABLE users VALIDATE CONSTRAINT users_phone_not_null;
```

For MySQL, tools like `pt-online-schema-change` (Percona) and `gh-ost` (GitHub) perform online schema changes by creating a shadow table, copying data, and swapping:

```bash
# gh-ost: online schema change for MySQL
gh-ost \
  --host=localhost \
  --database=mydb \
  --table=users \
  --alter="ADD COLUMN phone VARCHAR(20)" \
  --execute

# What gh-ost does:
# 1. Creates _users_gho (ghost table) with new schema
# 2. Creates _users_ghc (changelog table)
# 3. Copies existing rows in batches
# 4. Captures ongoing changes via binlog
# 5. Atomically renames: users → _users_old, _users_gho → users
```

#### Zero-Downtime Rename Strategy

Renaming a column is tricky — old code references the old name, new code references the new name:

```sql
-- Step 1: Add new column
ALTER TABLE users ADD COLUMN full_name VARCHAR(200);

-- Step 2: Dual-write (application writes to both columns)
-- Deploy code that writes to both 'name' and 'full_name'

-- Step 3: Backfill
UPDATE users SET full_name = name WHERE full_name IS NULL;

-- Step 4: Switch reads to new column
-- Deploy code that reads from 'full_name'

-- Step 5: Stop writing to old column
-- Deploy code that only writes to 'full_name'

-- Step 6: Drop old column
ALTER TABLE users DROP COLUMN name;
```

## Connection Pooling

Every database connection consumes resources: memory (5-10 MB per connection in PostgreSQL), file descriptors, and CPU for process/thread management. Without connection pooling, a spike in application instances can exhaust the database's connection limit.

![Connection pooling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-connection-pooling.png)


### The Problem

```text
Without pooling:
  100 app servers × 20 threads each = 2,000 database connections
  PostgreSQL default max_connections = 100
  Result: connection refused errors, application crashes

With pooling:
  100 app servers × 20 threads each = 2,000 application connections
  PgBouncer maintains 50 actual database connections
  Multiplexing ratio: 40:1
```

### PgBouncer (PostgreSQL)

```ini
; pgbouncer.ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

; Pool sizing
default_pool_size = 20        ; connections per user/db pair
max_client_conn = 1000        ; max client connections to pgbouncer
min_pool_size = 5             ; keep at least 5 connections ready
reserve_pool_size = 5         ; extra connections for burst

; Pool mode
pool_mode = transaction
; session    = connection held for entire client session (safest, least efficient)
; transaction = connection returned after each transaction (best balance)
; statement   = connection returned after each statement (most efficient, limited features)
```

```bash
# Connect to PgBouncer admin console
psql -h localhost -p 6432 -U pgbouncer pgbouncer

# Show pool status
SHOW POOLS;
#  database |   user    | cl_active | cl_waiting | sv_active | sv_idle | sv_used
# ----------+-----------+-----------+------------+-----------+---------+--------
#  mydb     | app_user  |        45 |          0 |        18 |       2 |       0

# Show client connections
SHOW CLIENTS;

# Show server (database) connections
SHOW SERVERS;

# Show stats
SHOW STATS;
```

### ProxySQL (MySQL)

```sql
-- ProxySQL admin interface (port 6032)
-- Configure backend servers
INSERT INTO mysql_servers (hostgroup_id, hostname, port, max_connections)
VALUES
  (10, 'mysql-leader', 3306, 100),    -- writer hostgroup
  (20, 'mysql-replica1', 3306, 100),  -- reader hostgroup
  (20, 'mysql-replica2', 3306, 100);  -- reader hostgroup

-- Configure query routing rules
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup)
VALUES
  (1, 1, '^SELECT .* FOR UPDATE', 10),  -- SELECT FOR UPDATE → leader
  (2, 1, '^SELECT', 20),                -- all other SELECTs → replicas
  (3, 1, '.*', 10);                     -- everything else → leader

-- Apply configuration
LOAD MYSQL SERVERS TO RUNTIME;
LOAD MYSQL QUERY RULES TO RUNTIME;
SAVE MYSQL SERVERS TO DISK;
SAVE MYSQL QUERY RULES TO DISK;
```

### Application-Level Pooling

```python
# Python: SQLAlchemy connection pool
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost:5432/mydb",
    pool_size=20,           # number of persistent connections
    max_overflow=10,        # additional connections during burst
    pool_timeout=30,        # seconds to wait for a connection
    pool_recycle=3600,      # recycle connections after 1 hour
    pool_pre_ping=True,     # test connections before use
)
```

```java
// Java: HikariCP (fastest JDBC pool)
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
config.setUsername("user");
config.setPassword("pass");
config.setMaximumPoolSize(20);
config.setMinimumIdle(5);
config.setConnectionTimeout(30000);  // 30 seconds
config.setIdleTimeout(600000);       // 10 minutes
config.setMaxLifetime(1800000);      // 30 minutes
config.setLeakDetectionThreshold(60000); // warn if connection held > 60s

HikariDataSource ds = new HikariDataSource(config);
```

## Database Monitoring


![Database monitoring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-monitoring-dashboard.png)


![Database monitoring dashboard control room with holographic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/08-database-monitoring-dashboard-control-room-with-holographic-.jpg)

### Key Metrics

| Metric | Healthy Range | Action if Unhealthy |
|--------|--------------|-------------------|
| **QPS** (queries/second) | Baseline ± 20% | Investigate if spike or drop |
| **Query latency p99** | < 100 ms for OLTP | Find and optimize slow queries |
| **Active connections** | < 80% of max | Increase pool or max_connections |
| **Replication lag** | < 1 second | Check replica load, network |
| **Buffer pool hit ratio** | > 99% | Increase buffer pool or reduce dataset |
| **Disk I/O wait** | < 10% | Faster storage or more RAM for caching |
| **Lock waits** | < 5% of transactions | Optimize transaction scope |
| **Deadlocks/sec** | < 1 | Fix lock ordering in application |
| **WAL generation rate** | Baseline ± 30% | Check for write storms |
| **Table bloat** (PostgreSQL) | < 20% dead tuples | Tune autovacuum |

### PostgreSQL Monitoring Queries

```sql
-- Current activity: what is running right now?
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

-- Table statistics: which tables are hot?
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

-- Index usage: are your indexes being used?
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
-- Indexes with 0 scans are candidates for removal

-- Cache hit ratio
SELECT
    'index hit rate' AS name,
    ROUND(sum(idx_blks_hit)::numeric / NULLIF(sum(idx_blks_hit + idx_blks_read), 0), 4) AS ratio
FROM pg_statio_user_indexes
UNION ALL
SELECT
    'table hit rate',
    ROUND(sum(heap_blks_hit)::numeric / NULLIF(sum(heap_blks_hit + heap_blks_read), 0), 4)
FROM pg_statio_user_tables;
-- Both should be > 0.99

-- Database size
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

### MySQL Monitoring Queries

```sql
-- Current processlist
SHOW PROCESSLIST;

-- InnoDB status (comprehensive)
SHOW ENGINE INNODB STATUS\G

-- Key InnoDB metrics
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

## Slow Query Analysis


![Database migration journey old schema transforming into new](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/08-database-migration-journey-old-schema-transforming-into-new-.jpg)

### Enable Slow Query Logging

```bash
# PostgreSQL: postgresql.conf
log_min_duration_statement = 100  # log queries taking > 100ms
log_statement = 'none'            # don't log all statements (too noisy)
auto_explain.log_min_duration = '200ms'  # auto-explain for slow queries
auto_explain.log_analyze = on
```

```bash
# MySQL: my.cnf
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 0.1  # log queries taking > 100ms
log_queries_not_using_indexes = 1
```

### Query Optimization Workflow

```text
Step 1: Find the slow query
  → slow query log, pg_stat_statements, or monitoring tool

Step 2: Run EXPLAIN ANALYZE
  → Understand the execution plan

Step 3: Identify bottleneck
  → Seq Scan? Missing index? Bad join order? Large sort?

Step 4: Test fix
  → Add index, rewrite query, or adjust configuration

Step 5: Verify with EXPLAIN ANALYZE again
  → Confirm improvement

Step 6: Monitor in production
  → Watch for regression
```

```sql
-- PostgreSQL: pg_stat_statements (top slow queries)
-- Enable: shared_preload_libraries = 'pg_stat_statements'

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

## Backup Strategies


![Backup strategy comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/08-backup-strategy.png)

### Logical Backups

Logical backups export data as SQL statements or structured data files.

```bash
# PostgreSQL: pg_dump
# Single database, compressed
pg_dump -Fc -h localhost -U postgres mydb > mydb_$(date +%Y%m%d).dump

# Specific tables only
pg_dump -Fc -t users -t orders mydb > tables_$(date +%Y%m%d).dump

# Schema only (no data)
pg_dump -s mydb > schema.sql

# Restore
pg_restore -h localhost -U postgres -d mydb mydb_20231228.dump

# All databases
pg_dumpall -h localhost -U postgres > all_databases.sql
```

```bash
# MySQL: mysqldump
# Single database, with consistent snapshot
mysqldump --single-transaction --routines --triggers \
  -h localhost -u root -p mydb > mydb_$(date +%Y%m%d).sql

# Compressed
mysqldump --single-transaction mydb | gzip > mydb_$(date +%Y%m%d).sql.gz

# Restore
mysql -h localhost -u root -p mydb < mydb_20231228.sql
# Or compressed:
gunzip < mydb_20231228.sql.gz | mysql -h localhost -u root -p mydb
```

### Physical Backups

Physical backups copy the actual data files. Much faster for large databases.

```bash
# PostgreSQL: pg_basebackup
pg_basebackup -h localhost -U repl_user -D /backup/base \
  --checkpoint=fast --wal-method=stream -P

# This creates a complete copy of the data directory
# including WAL files needed for consistent recovery

# Restore: stop PostgreSQL, replace data directory, start
```

```bash
# MySQL: Percona XtraBackup
# Full backup
xtrabackup --backup --target-dir=/backup/full \
  --user=root --password=xxx

# Prepare (apply redo logs)
xtrabackup --prepare --target-dir=/backup/full

# Restore: stop MySQL, copy files, fix permissions, start
xtrabackup --copy-back --target-dir=/backup/full
chown -R mysql:mysql /var/lib/mysql
```

### Point-in-Time Recovery (PITR)

PITR lets you restore to any moment in time, not just when the backup was taken.

```bash
# PostgreSQL PITR
# 1. Take a base backup
pg_basebackup -D /backup/base --checkpoint=fast

# 2. Archive WAL segments continuously
# postgresql.conf:
# archive_mode = on
# archive_command = 'cp %p /backup/wal/%f'

# 3. To recover to a specific time:
# recovery.conf (or postgresql.conf in PG 12+):
# restore_command = 'cp /backup/wal/%f %p'
# recovery_target_time = '2023-12-28 14:30:00 UTC'
# recovery_target_action = 'promote'
```

### Backup Comparison

| Aspect | Logical (pg_dump) | Physical (pg_basebackup) | PITR |
|--------|-------------------|-------------------------|------|
| Speed (backup) | Slow (reads all data via SQL) | Fast (copies files) | Continuous (WAL streaming) |
| Speed (restore) | Slow (replays SQL) | Fast (copies files) | Medium (base + replay WAL) |
| Size | Smaller (compressed SQL) | Larger (full data dir) | Base + WAL segments |
| Granularity | Per-table possible | Full cluster only | Any point in time |
| Version compatibility | Cross-version OK | Same major version | Same major version |
| Partial restore | Yes (specific tables) | No | No |

### Backup Testing

**A backup that has never been tested is not a backup.** Schedule regular restore tests:

```bash
#!/bin/bash
# Monthly backup restore test
set -e

echo "Starting backup restore test: $(date)"

# Create a test database
createdb restore_test

# Restore the latest backup
pg_restore -d restore_test /backup/latest/mydb.dump

# Run validation queries
psql -d restore_test -c "SELECT COUNT(*) FROM users;" | grep -q "[0-9]"
psql -d restore_test -c "SELECT COUNT(*) FROM orders;" | grep -q "[0-9]"

# Compare row counts with production
PROD_USERS=$(psql -d mydb -t -c "SELECT COUNT(*) FROM users;")
TEST_USERS=$(psql -d restore_test -t -c "SELECT COUNT(*) FROM users;")

if [ "$PROD_USERS" != "$TEST_USERS" ]; then
    echo "ALERT: Row count mismatch! Prod=$PROD_USERS, Restored=$TEST_USERS"
    exit 1
fi

# Clean up
dropdb restore_test

echo "Backup restore test passed: $(date)"
```

## Choosing a Managed Database

Running your own database is educational but operationally expensive. Managed services handle patching, backups, replication, and monitoring:

| Service | Provider | Engine Support | Key Feature |
|---------|---------|---------------|-------------|
| RDS | Alibaba Cloud / AWS | MySQL, PostgreSQL, SQL Server, MariaDB | Automated backups, multi-AZ |
| PolarDB | Alibaba Cloud | MySQL, PostgreSQL, Oracle-compatible | Shared storage, up to 100 TB, read scaling |
| AnalyticDB | Alibaba Cloud | MySQL-compatible (OLAP) | Petabyte-scale analytics, columnar |
| Aurora | AWS | MySQL, PostgreSQL compatible | 6-way replication, auto-scaling storage |
| Cloud SQL | Google Cloud | MySQL, PostgreSQL, SQL Server | Automated failover, IAM integration |
| AlloyDB | Google Cloud | PostgreSQL compatible | Columnar engine for analytics |

### When to Self-Manage vs Use Managed

| Self-manage when | Use managed when |
|-----------------|-----------------|
| You need specific extensions or patches | Standard configurations suffice |
| Extreme performance tuning required | Operational simplicity is priority |
| Cost-sensitive at very large scale | Team is small, no dedicated DBA |
| Compliance requires full control | Provider meets compliance needs |
| Educational/learning purposes | Production workloads |

## Capacity Planning

### When to Scale Up vs Scale Out

```text
Scale UP (vertical):
  More CPU, RAM, faster storage on the same server
  + Simpler (no application changes)
  + All features work (transactions, joins)
  - Has a ceiling (largest available instance)
  - Single point of failure without replication

Scale OUT (horizontal):
  Add more servers (replication + sharding)
  + No ceiling (add nodes as needed)
  + Built-in redundancy
  - Application complexity (query routing, distributed transactions)
  - Not all operations work across shards (cross-shard joins)
```

### Capacity Estimation

```yaml
Example: estimate database size for a SaaS application

Users: 1 million
Average row size: 200 bytes
Tables: users (1M rows), orders (10M), order_items (30M), products (100K), sessions (5M)

Data size:
  users:       1,000,000 × 200 B = 200 MB
  orders:     10,000,000 × 150 B = 1.5 GB
  order_items: 30,000,000 × 100 B = 3.0 GB
  products:       100,000 × 500 B = 50 MB
  sessions:    5,000,000 × 1 KB  = 5.0 GB
  Total data: ~10 GB

Index overhead: ~30-50% of data = 3-5 GB
WAL/binlog: ~2-5 GB rotating
Working set (hot data): maybe 20% = 2 GB

Recommendation:
  RAM: 16 GB (fits entire dataset in buffer pool)
  Storage: 50 GB SSD (room for growth + backups)
  CPU: 4 cores (for a few hundred QPS)
```

## War Stories: Common Production Incidents

| Incident | Root Cause | Symptom | Fix | Prevention |
|----------|-----------|---------|-----|------------|
| **Query of death** | Missing index on new feature | CPU 100%, all queries slow | `CREATE INDEX CONCURRENTLY` | Require EXPLAIN for new queries in code review |
| **Connection exhaustion** | No connection pooling, app scaled to 50 pods | "too many connections" errors | Add PgBouncer, reduce pool size per pod | Always use connection pooling |
| **Replication broke** | DDL on leader without `SET STATEMENT_FORMAT=ROW` | Replica diverged, stale reads | Re-snapshot replica | Standardize binlog format to ROW |
| **Disk full** | Unmonitored WAL growth from idle replication slot | Database refuses writes | `SELECT pg_drop_replication_slot('dead_slot')` | Monitor `pg_replication_slots`, alert on inactive |
| **Table bloat** | Autovacuum not keeping up with UPDATE-heavy table | Queries progressively slower | Manual `VACUUM FULL` (locks table) | Tune `autovacuum_vacuum_scale_factor` per table |
| **Lock convoy** | Long-running migration held exclusive lock | All queries queued behind lock | Kill migration, retry with `lock_timeout` | Set `lock_timeout = '5s'` for migrations |
| **OOM killer** | Huge sort in memory (no work_mem limit) | PostgreSQL process killed, connections dropped | Set `work_mem = '256MB'`, add index | Configure `work_mem` and `temp_file_limit` |
| **Cascading failure** | Database slow → app retries × 3 → 3x load → database slower | Complete outage | Circuit breaker, stop retries | Implement circuit breakers, set query timeouts |
| **Data corruption** | `fsync=off` for "performance" | Silent data loss after power outage | Restore from backup | Never disable fsync in production |
| **Schema migration failure** | ALTER TABLE timeout on 500M row table | Stuck lock, blocked writes for 20 min | `gh-ost` / `pt-online-schema-change` | Use online DDL tools for large tables |

### The Golden Rules

1. **Always have backups. Test your restores.** A backup you have never restored is a hope, not a strategy.

2. **Monitor before you need to.** Set up dashboards and alerts before the first production user. Key metrics: connection count, query latency p99, replication lag, disk usage.

3. **Use connection pooling.** Always. Even if your application framework claims to handle it.

4. **Set timeouts everywhere.** Statement timeout, lock timeout, connection timeout, idle-in-transaction timeout. A query without a timeout is a query that will eventually hold a lock forever.

```sql
-- PostgreSQL: set safety timeouts
ALTER DATABASE mydb SET statement_timeout = '30s';
ALTER DATABASE mydb SET lock_timeout = '10s';
ALTER DATABASE mydb SET idle_in_transaction_session_timeout = '60s';
```

5. **Never run untested migrations in production.** Test every migration against a copy of production data. Time it. Check for locks.

6. **Read replicas are not a scaling strategy for writes.** They only help with reads. If writes are your bottleneck, you need sharding.

7. **Keep transactions short.** A transaction that holds locks for 30 seconds blocks every other transaction that needs those rows.

8. **The database is not a queue.** If you are polling a table with `SELECT ... WHERE status = 'pending' FOR UPDATE SKIP LOCKED`, you should probably be using an actual message queue.

## Summary
Over these eight articles, we have gone from the relational model and SQL basics all the way to distributed transactions and production operations. The path was intentional: you cannot understand why replication lag matters until you understand isolation levels, and you cannot appreciate the Saga pattern until you understand why 2PC blocks.

Databases are one of those areas where superficial knowledge is dangerous. A developer who does not understand indexing will build systems that work perfectly on small datasets and fail catastrophically in production. A team that does not understand isolation levels will ship concurrency bugs that only appear under load. An organization that does not test its backups will discover they are useless precisely when they are needed most.

The fundamentals do not change much. B-trees, WALs, MVCC, and consensus have been the core building blocks for decades. Master them once, and every new database you encounter — whether it is PostgreSQL, CockroachDB, DynamoDB, or whatever comes next — is a variation on ideas you already understand.
