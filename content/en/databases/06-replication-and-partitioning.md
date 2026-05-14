---
title: "Databases (6): Replication and Partitioning — Scaling Beyond One Machine"
date: 2024-04-26 09:00:00
tags:
  - Databases
  - Replication
  - Sharding
  - Distributed Systems
categories: Databases
series: databases
lang: en
description: "How databases replicate data for availability and partition data for scale — covering leader-follower, multi-leader, leaderless, sharding strategies, and consistent hashing."
disableNunjucks: true
series_order: 6
translationKey: "databases-6"
---

A single database server can handle a remarkable amount of load — a well-tuned PostgreSQL instance can serve tens of thousands of queries per second. But eventually you hit a wall. Maybe you need more read throughput than one CPU can provide. Maybe you need your data to survive a data center fire. Maybe your dataset exceeds what fits on a single disk. That is when you need replication and partitioning.

These are two orthogonal strategies:
- **Replication**: copy the same data to multiple machines (for availability and read scaling)
- **Partitioning** (sharding): split the data into pieces, each stored on a different machine (for write scaling and data size)

Most production databases use both.

---

## Replication: Keeping Copies of Your Data

### Why Replicate?

| Goal | How replication helps |
|------|----------------------|
| **High availability** | If one server dies, another takes over |
| **Read scaling** | Spread read queries across multiple replicas |
| **Geographic distribution** | Put data closer to users in different regions |
| **Disaster recovery** | Keep a copy in a different data center |

### Leader-Follower (Master-Slave) Replication

The most common replication topology. One node (the leader/master/primary) handles all writes. One or more followers (slaves/replicas/standbys) receive a copy of every write and serve read queries.

![Leader-follower replication](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-leader-follower.png)


```text
                    Writes
Client ────────────► Leader (Primary)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          Follower  Follower  Follower
          (Replica) (Replica) (Replica)
              ▲        ▲        ▲
              └────────┼────────┘
                    Reads
```

#### Synchronous vs Asynchronous Replication

| Aspect | Synchronous | Asynchronous |
|--------|------------|--------------|
| Write acknowledged when | Leader AND follower(s) have written | Leader has written |
| Data loss risk | None (if sync replicas are alive) | Up to seconds of data on leader crash |
| Write latency | Higher (wait for follower) | Lower (return immediately) |
| Availability impact | Follower failure blocks writes | Follower failure does not affect writes |
| Common default | PostgreSQL: one sync replica | MySQL: async by default |

```sql
-- PostgreSQL: configure synchronous replication
-- postgresql.conf on primary
synchronous_standby_names = 'FIRST 1 (replica1, replica2)'
-- FIRST 1 = wait for at least 1 of the listed replicas
-- This means writes block until replica1 OR replica2 confirms

-- Check replication status
SELECT
    client_addr,
    state,
    sync_state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    (sent_lsn - replay_lsn) AS replication_lag_bytes
FROM pg_stat_replication;
```

In practice, most setups use **semi-synchronous** replication: one follower is synchronous (guarantees no data loss), the rest are asynchronous (for read scaling).

### Replication Lag

With asynchronous replication, followers may be slightly behind the leader. This causes consistency anomalies:

#### Read-After-Write Consistency

A user writes data, then immediately reads it — but the read goes to a follower that has not received the write yet.

```yaml
Timeline:
1. User posts a comment (goes to leader)
2. Leader writes: OK
3. User reloads page (goes to follower)
4. Follower hasn't received step 2 yet
5. User sees: "No comments" — their comment disappeared!
```

Solutions:

```python
# Solution 1: Read from leader for recently-written data
def get_user_profile(user_id, requesting_user_id):
    if user_id == requesting_user_id:
        # User is viewing their own profile — read from leader
        return db_leader.query("SELECT * FROM users WHERE id = %s", user_id)
    else:
        # Viewing someone else's profile — replica is fine
        return db_replica.query("SELECT * FROM users WHERE id = %s", user_id)

# Solution 2: Track the write timestamp
def get_comments(post_id, last_write_ts=None):
    if last_write_ts and (time.time() - last_write_ts) < 5:
        # Written within last 5 seconds — read from leader
        return db_leader.query("SELECT * FROM comments WHERE post_id = %s", post_id)
    return db_replica.query("SELECT * FROM comments WHERE post_id = %s", post_id)
```

#### Monotonic Reads

A user makes two reads. The first happens to hit an up-to-date replica. The second hits a lagging replica. The user sees data go *backwards in time*.

Solution: route each user consistently to the same replica (e.g., hash the user ID to pick a replica).

### Multi-Leader Replication

Each of several leaders accepts writes independently. Changes are replicated between leaders. Common in multi-datacenter setups.

![Multi-leader replication](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-multi-leader.png)


```text
Data Center A              Data Center B
┌──────────────┐          ┌──────────────┐
│   Leader A   │◄────────►│   Leader B   │
│  (read/write)│          │  (read/write)│
│      │       │          │      │       │
│  Follower    │          │  Follower    │
│  Follower    │          │  Follower    │
└──────────────┘          └──────────────┘
```

The hard part: **conflict resolution**. If user A updates a row in DC-A and user B updates the same row in DC-B at the same time, which write wins?

| Strategy | How it works | Trade-off |
|----------|-------------|-----------|
| **Last-writer-wins (LWW)** | Highest timestamp wins | Simple but loses data silently |
| **Merge values** | Application-specific merge logic | Complex but preserves both changes |
| **Conflict-free (CRDT)** | Data structures that merge automatically | Limited to specific operations (counters, sets) |
| **Manual resolution** | Flag conflicts for human review | Slow but accurate |

```sql
-- Example: last-writer-wins with a version timestamp
-- Both leaders accept updates independently
-- On sync, the row with the latest updated_at wins

-- Leader A: user updates their name
UPDATE users SET name = 'Alice Chen', updated_at = '2023-12-15T10:00:01Z'
WHERE user_id = 1;

-- Leader B: same user updates their name (at almost the same time)
UPDATE users SET name = 'Alice C.', updated_at = '2023-12-15T10:00:02Z'
WHERE user_id = 1;

-- After replication sync: Leader B's update wins (later timestamp)
-- But Leader A's change is silently lost
```

### Leaderless Replication (Dynamo-Style)

No leader at all. Any node can accept reads and writes. Used by Amazon DynamoDB, Apache Cassandra, and Riak.

#### Quorum Reads and Writes

With N replicas, you configure:
- **W** = number of nodes that must acknowledge a write
- **R** = number of nodes that must respond to a read

The rule: **W + R > N** guarantees you will read at least one node that has the latest write.

```text
N = 3 (three replicas of each piece of data)
W = 2 (write must be acknowledged by 2 nodes)
R = 2 (read must be answered by 2 nodes)

Write "balance = 500" to key "account:1":
  Node 1: ✓ acknowledged (balance = 500)
  Node 2: ✓ acknowledged (balance = 500)
  Node 3: ✗ unreachable (still has balance = 1000)
  --> Write succeeds (W=2 met)

Read key "account:1":
  Node 1: balance = 500 (version 2)
  Node 2: balance = 500 (version 2)
  --> Returns 500 (latest version)
  OR
  Node 2: balance = 500 (version 2)
  Node 3: balance = 1000 (version 1)
  --> Returns 500 (client picks highest version)
```

Common configurations:

| Config | Properties |
|--------|-----------|
| W=N, R=1 | Strong writes, fast reads (write-heavy penalty) |
| W=1, R=N | Fast writes, strong reads (read-heavy penalty) |
| W=2, R=2 (N=3) | Balanced — tolerates 1 node failure |
| W=1, R=1 | Fast but no consistency guarantee |

#### Read Repair and Anti-Entropy

When a read detects stale data on a node, the client can write the latest value back to the stale node. This is called **read repair**:

```text
Read key "account:1":
  Node 1: balance = 500 (version 2) ✓ latest
  Node 3: balance = 1000 (version 1) ✗ stale
  --> Return 500 to client
  --> Background: write balance=500 to Node 3 (read repair)
```

For keys that are rarely read, an **anti-entropy process** runs in the background, comparing data between replicas and fixing discrepancies.

## Partitioning (Sharding)

Replication puts the same data on multiple machines. Partitioning puts *different* data on different machines. This lets you:
- Store more data than fits on one machine
- Spread write load across multiple machines
- Keep hot data closer to specific users (geographic partitioning)

![Partitioning strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-partition-strategies.png)


### Range-Based Partitioning

Assign contiguous ranges of the partition key to each shard:

```text
Shard 1: user_id    1 - 1,000,000
Shard 2: user_id    1,000,001 - 2,000,000
Shard 3: user_id    2,000,001 - 3,000,000
```

```sql
-- PostgreSQL declarative partitioning (range)
CREATE TABLE orders (
    order_id    BIGSERIAL,
    user_id     INT NOT NULL,
    created_at  TIMESTAMP NOT NULL,
    total       DECIMAL(10,2)
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2023_q4 PARTITION OF orders
    FOR VALUES FROM ('2023-10-01') TO ('2024-01-01');

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Queries automatically route to the correct partition
SELECT * FROM orders WHERE created_at = '2023-11-15';
-- Only scans orders_2023_q4, skips other partitions
```

**Advantage**: Range scans are efficient (adjacent keys are on the same shard).
**Disadvantage**: Hot spots — if most writes have recent timestamps, the latest partition gets all the write traffic.

### Hash-Based Partitioning

Apply a hash function to the partition key and assign hash ranges to shards:

```text
partition = hash(user_id) % num_shards

hash("user:1")  = 0x3A2B... → shard 2
hash("user:2")  = 0x8F1C... → shard 0
hash("user:3")  = 0x12D4... → shard 1
```

```sql
-- PostgreSQL hash partitioning
CREATE TABLE sessions (
    session_id  UUID PRIMARY KEY,
    user_id     INT NOT NULL,
    data        JSONB,
    expires_at  TIMESTAMP
) PARTITION BY HASH (session_id);

CREATE TABLE sessions_p0 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE sessions_p1 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE sessions_p2 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE sessions_p3 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

**Advantage**: Even distribution of data and load.
**Disadvantage**: Range queries must hit all shards (the hash destroys ordering).

### Consistent Hashing

The problem with `hash(key) % N`: when you add or remove a shard, almost every key maps to a different shard, requiring massive data migration.

![Consistent hashing ring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-consistent-hashing.png)


Consistent hashing solves this by mapping both keys and nodes onto a ring (0 to 2^32):

```text
                    0 / 2^32
                      │
              Node C  ●
                     ╱  ╲
                    ╱    ╲
           ●──────╱──────●
         Node A  ╱      Node B
                ╱
               ╱
Key "user:42" hashes to position X
→ Walk clockwise → first node encountered = owner

Adding Node D:
  Only keys between Node C and Node D need to move
  (not all keys in the cluster)
```

With **virtual nodes** (vnodes), each physical node appears at multiple positions on the ring, improving balance:

```text
Physical Node A → Virtual nodes: A1, A2, A3, A4, A5 (5 positions on ring)
Physical Node B → Virtual nodes: B1, B2, B3, B4, B5

More virtual nodes = more even distribution
Cassandra default: 256 vnodes per physical node
```

### Rebalancing Strategies

When you add or remove nodes, data must move. There are two approaches:

**Fixed number of partitions**: Create many more partitions than nodes (e.g., 1,000 partitions for 10 nodes). When adding a node, move entire partitions to it.

```text
Before (10 nodes, 1000 partitions):
  Node 1: partitions 0-99
  Node 2: partitions 100-199
  ...

After adding Node 11:
  Node 1:  partitions 0-89     (gave away 10)
  Node 2:  partitions 100-189  (gave away 10)
  ...
  Node 11: partitions 90-99, 190-199, ...  (received ~91 partitions)
```

**Dynamic partitioning**: Start with a few partitions. Split them when they get too large, merge them when they get too small. Used by HBase and MongoDB.

### Secondary Indexes in Partitioned Databases

Primary key lookups are straightforward — hash or range the key, go to the right shard. But what about secondary indexes?

```sql
-- Table partitioned by user_id
-- But we also query by email
SELECT * FROM users WHERE email = 'alice@example.com';
-- Which shard has this user? We don't know without checking all of them.
```

Two approaches:

**Local (document-partitioned) index**: Each shard maintains its own secondary index covering only its data.

```text
Shard 1: local index on email → {alice@...: row 1, bob@...: row 2}
Shard 2: local index on email → {carol@...: row 3, dave@...: row 4}

Query by email → scatter to ALL shards, gather results
(called "scatter-gather" — expensive for fan-out)
```

**Global (term-partitioned) index**: The secondary index itself is partitioned across shards.

```text
Email index shard A (emails a-m): alice@... → shard 1, carol@... → shard 2
Email index shard B (emails n-z): zara@... → shard 3

Query by email → go to the right index shard → then to the data shard
(2 hops but no scatter-gather)
```

| Approach | Read cost | Write cost | Consistency |
|----------|----------|------------|-------------|
| Local index | Scatter to all shards | Update only local index | Always consistent |
| Global index | Single shard lookup | Must update remote index shard | Eventually consistent |

## MySQL Replication Setup Walkthrough


![Consistent hashing ring as a futuristic carousel with data d](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/06-consistent-hashing-ring-as-a-futuristic-carousel-with-data-d.jpg)

Let us set up a basic leader-follower replication in MySQL.

### Leader Configuration

```bash
# /etc/mysql/mysql.conf.d/mysqld.cnf on the leader
[mysqld]
server-id           = 1
log_bin              = /var/log/mysql/mysql-bin
binlog_format        = ROW          # safest format
binlog_expire_logs_seconds = 604800 # 7 days retention
sync_binlog          = 1            # sync binlog on every commit
innodb_flush_log_at_trx_commit = 1  # full durability
```

```sql
-- Create a replication user on the leader
CREATE USER 'repl_user'@'%' IDENTIFIED BY 'strong_password_here';
GRANT REPLICATION SLAVE ON *.* TO 'repl_user'@'%';
FLUSH PRIVILEGES;

-- Get the current binlog position
SHOW MASTER STATUS;
-- +------------------+----------+
-- | File             | Position |
-- +------------------+----------+
-- | mysql-bin.000003 |      785 |
-- +------------------+----------+
```

### Take a Consistent Backup

```bash
# Option 1: mysqldump with consistent snapshot
mysqldump --all-databases --single-transaction \
  --source-data=2 --routines --triggers \
  -u root -p > leader_backup.sql

# Option 2: For large databases, use xtrabackup
xtrabackup --backup --target-dir=/backup/full \
  --user=root --password=xxx
```

### Follower Configuration

```bash
# /etc/mysql/mysql.conf.d/mysqld.cnf on the follower
[mysqld]
server-id            = 2       # must be unique
relay_log            = /var/log/mysql/mysql-relay
read_only            = ON      # prevent accidental writes
super_read_only      = ON      # even root can't write
```

```sql
-- Restore the backup on the follower
-- Then configure replication
CHANGE REPLICATION SOURCE TO
    SOURCE_HOST='leader-hostname',
    SOURCE_USER='repl_user',
    SOURCE_PASSWORD='strong_password_here',
    SOURCE_LOG_FILE='mysql-bin.000003',
    SOURCE_LOG_POS=785;

-- Start replication
START REPLICA;

-- Check replication status
SHOW REPLICA STATUS\G
-- Key fields to check:
--   Replica_IO_Running: Yes
--   Replica_SQL_Running: Yes
--   Seconds_Behind_Source: 0
--   Last_Error: (should be empty)
```

### Monitoring Replication Health

```sql
-- On the follower: check lag
SHOW REPLICA STATUS\G
-- Seconds_Behind_Source: 0  <-- healthy
-- Seconds_Behind_Source: 45 <-- concerning
-- Seconds_Behind_Source: NULL <-- replication broken!

-- On the leader: check connected replicas
SHOW REPLICAS;
-- or older syntax: SHOW SLAVE HOSTS;
```

```bash
# Quick health check script
#!/bin/bash
LAG=$(mysql -e "SHOW REPLICA STATUS\G" | grep "Seconds_Behind_Source" | awk '{print $2}')
IO_RUNNING=$(mysql -e "SHOW REPLICA STATUS\G" | grep "Replica_IO_Running" | awk '{print $2}')
SQL_RUNNING=$(mysql -e "SHOW REPLICA STATUS\G" | grep "Replica_SQL_Running" | awk '{print $2}')

echo "Replication Lag: ${LAG}s"
echo "IO Thread: $IO_RUNNING"
echo "SQL Thread: $SQL_RUNNING"

if [ "$LAG" -gt 60 ] || [ "$IO_RUNNING" != "Yes" ] || [ "$SQL_RUNNING" != "Yes" ]; then
    echo "ALERT: Replication unhealthy!"
    exit 1
fi
```

### Failover: Promoting a Follower

When the leader fails, promote a follower:

```sql
-- On the follower to be promoted:
STOP REPLICA;
RESET REPLICA ALL;

-- The follower is now a standalone server
-- Reconfigure other followers to point to the new leader

-- On remaining followers:
STOP REPLICA;
CHANGE REPLICATION SOURCE TO
    SOURCE_HOST='new-leader-hostname',
    SOURCE_LOG_FILE='...',
    SOURCE_LOG_POS=...;
START REPLICA;

-- Application configuration must also be updated
-- (or use a proxy like ProxySQL / HAProxy)
```

In production, use orchestration tools for automated failover:
- **Orchestrator** (MySQL): Detects leader failure, promotes a follower, reconfigures replicas
- **Patroni** (PostgreSQL): Manages leader election via etcd/ZooKeeper/Consul
- **pg_auto_failover**: Simpler alternative for PostgreSQL

## What's Next


![Distributed database replication data streams flowing betwee](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/06-distributed-database-replication-data-streams-flowing-betwee.jpg)

Replication and partitioning get data onto multiple machines. But what happens when a single transaction needs to update data on multiple machines? That is the problem of **distributed transactions** — 2PC, Saga, consensus, and why most people avoid them when they can. We will cover that next.
