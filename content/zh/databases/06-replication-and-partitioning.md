---
title: "数据库（六）：复制与分片——突破单机限制的扩展之道"
date: 2024-04-26 09:00:00
tags:
  - Databases
  - Replication
  - Sharding
  - Distributed Systems
categories: Databases
series: databases
lang: zh
description: "数据库如何通过复制保障高可用性，又如何通过分片实现水平扩展——涵盖主从复制、多主复制、无主复制、分片策略及一致性哈希。"
disableNunjucks: true
series_order: 6
translationKey: "databases-6"
---
一台数据库服务器能承载惊人的负载——一个调优良好的 PostgreSQL 实例每秒可处理数万次查询。但终究会遇到瓶颈：可能是读吞吐量超出了单颗 CPU 的能力，需要数据在数据中心火灾中幸存，又或者数据集已经超出单块磁盘的容量。此时，你就需要**复制（Replication）**与**分片（Partitioning / Sharding）**。

这是两种正交的扩展策略：
- **复制**：将**相同的数据**拷贝到多台机器上（提升可用性与读扩展能力）
- **分片**（Sharding）：将**不同的数据**切分为多个片段，每个片段存储在不同机器上（提升写扩展能力与总数据容量）

绝大多数生产环境数据库同时采用这两种策略。

## 复制：维护数据的多份副本

### 为何要复制？

| 目标 | 复制如何帮助实现 |
|------|------------------|
| **高可用性（High availability）** | 若某台服务器宕机，另一台可立即接管 |
| **读扩展（Read scaling）** | 将读请求分散至多个副本节点 |
| **地理分布（Geographic distribution）** | 将数据部署在离不同地区用户更近的位置 |
| **灾难恢复（Disaster recovery）** | 在另一个数据中心保留一份副本 |

### 主从复制（Leader-Follower / Master-Slave）

最常用的复制拓扑结构是主节点（Leader/Master/Primary）负责全部写操作，一个或多个从节点（Follower/Slave/Replica/Standby）接收所有写入的副本并可服务读请求。

![主从复制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-leader-follower.png)


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

#### 同步复制 vs 异步复制

| 维度 | 同步复制 | 异步复制 |
|------|----------|----------|
| 写入确认时机 | 主节点 **和** 从节点均完成写入后才返回成功 | 仅主节点完成写入即返回成功 |
| 数据丢失风险 | 零风险（只要同步副本存活） | 主节点崩溃时最多丢失数秒数据 |
| 写入延迟 | 更高（需等待从节点响应） | 更低（立即返回） |
| 对可用性影响 | 任一同步从节点故障将阻塞写入 | 从节点故障不影响写入 |
| 常见默认配置 | PostgreSQL：支持配置至少一个同步副本 | MySQL：默认异步 |

```sql
-- PostgreSQL：配置同步复制
-- postgresql.conf on primary
synchronous_standby_names = 'FIRST 1 (replica1, replica2)'
-- FIRST 1 = 等待列表中至少 1 个副本确认
-- 即写入将阻塞，直到 replica1 或 replica2 中任意一个确认

-- 检查复制状态
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

实践中，多数部署采用**半同步复制（semi-synchronous）**：指定一个从节点为同步以确保零数据丢失，其余为异步以用于读扩展。

### 复制延迟（Replication Lag）

在异步复制下，从节点可能略微落后于主节点，从而引发一致性异常：

#### 读己所写一致性（Read-After-Write Consistency）

用户刚写入数据，随即发起读取，但该读请求被路由到了尚未收到该写入的从节点。

```text
时间线：
1. 用户发布一条评论（请求发往主节点）
2. 主节点写入成功：OK
3. 用户刷新页面（请求发往从节点）
4. 该从节点尚未收到第 2 步的写入
5. 用户看到：“暂无评论”——自己的评论“消失”了！
```

解决方案：

```python
# 方案 1：对近期写入的数据，强制从主节点读取
def get_user_profile(user_id, requesting_user_id):
    if user_id == requesting_user_id:
        # 用户查看自己资料 → 从主节点读
        return db_leader.query("SELECT * FROM users WHERE id = %s", user_id)
    else:
        # 查看他人资料 → 从副本读即可
        return db_replica.query("SELECT * FROM users WHERE id = %s", user_id)

# 方案 2：记录写入时间戳
def get_comments(post_id, last_write_ts=None):
    if last_write_ts and (time.time() - last_write_ts) < 5:
        # 5 秒内写入 → 从主节点读
        return db_leader.query("SELECT * FROM comments WHERE post_id = %s", post_id)
    return db_replica.query("SELECT * FROM comments WHERE post_id = %s", post_id)
```

#### 单调读（Monotonic Reads）

用户连续发起两次读请求：第一次命中了最新副本，第二次却命中了一个滞后的副本，导致用户看到数据“倒退”。

解决方案：为每个用户固定路由到同一副本（例如，对用户 ID 做哈希后选择副本）。

### 多主复制（Multi-Leader Replication）

多个主节点均可独立接受写入，彼此之间双向同步变更。常见于多数据中心架构。

![多主复制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-multi-leader.png)


```text
数据中心 A              数据中心 B
┌──────────────┐          ┌──────────────┐
│   主节点 A   │◄────────►│   主节点 B   │
│  （读/写）   │          │  （读/写）   │
│      │       │          │      │       │
│  从节点      │          │  从节点      │
└──────────────┘          └──────────────┘
```

难点在于：**冲突解决（Conflict resolution）**。若用户 A 在 DC-A 更新某行，用户 B 在 DC-B 几乎同时更新同一行，谁的写入胜出？

| 策略 | 工作原理 | 权衡 |
|------|----------|------|
| **最后写入者胜出（LWW）** | 时间戳最大的写入胜出 | 简单，但静默丢弃旧数据 |
| **值合并（Merge values）** | 应用层定义合并逻辑（如合并 JSON 字段） | 复杂，但可保留双方变更 |
| **无冲突复制数据类型（CRDT）** | 使用天然支持自动合并的数据结构（计数器、集合等） | 仅适用于特定操作类型 |
| **人工干预（Manual resolution）** | 标记冲突，交由人工审核 | 缓慢但准确 |

```sql
-- 示例：基于时间戳的 LWW 策略
-- 两个主节点均可独立接受更新
-- 同步时，`updated_at` 时间戳最新的行获胜

-- 主节点 A：用户更新姓名
UPDATE users SET name = 'Alice Chen', updated_at = '2023-12-15T10:00:01Z'
WHERE user_id = 1;

-- 主节点 B：同一用户几乎同时更新姓名
UPDATE users SET name = 'Alice C.', updated_at = '2023-12-15T10:00:02Z'
WHERE user_id = 1;

-- 同步完成后：主节点 B 的更新胜出（时间戳更新）
-- 但主节点 A 的变更被静默丢弃
```

### 无主复制（Leaderless Replication， Dynamo 风格）

完全不设主节点，任何节点均可接受读写请求。Amazon DynamoDB、Apache Cassandra 和 Riak 均采用此模型。

#### 法定人数读写（Quorum Reads and Writes）

设总副本数为 `N`，配置：
- **W** = 写入必须获得确认的最小节点数
- **R** = 读取必须响应的最小节点数

核心规则：**W + R > N** 可保证至少一次读取能命中包含最新写入的节点。

```text
N = 3（每份数据有 3 个副本）
W = 2（写入需获 2 个节点确认）
R = 2（读取需从 2 个节点获取响应）

向 key "account:1" 写入 "balance = 500"：
  节点 1：✓ 已确认（balance = 500）
  节点 2：✓ 已确认（balance = 500）
  节点 3：✗ 不可达（仍为 balance = 1000）
  --> 写入成功（满足 W=2）

读取 key "account:1"：
  节点 1：balance = 500（版本 2）
  节点 2：balance = 500（版本 2）
  --> 返回 500（最新版本）
  或
  节点 2：balance = 500（版本 2）
  节点 3：balance = 1000（版本 1）
  --> 返回 500（客户端选取最高版本）
```

常见配置：

| 配置 | 特性 |
|------|------|
| W=N, R=1 | 强一致性写入，快速读取（写入代价高） |
| W=1, R=N | 快速写入，强一致性读取（读取代价高） |
| W=2, R=2 （N=3） | 平衡型 —— 可容忍 1 个节点故障 |
| W=1, R=1 | 快速但无一致性保证 |

#### 读修复（Read Repair）与反熵（Anti-Entropy）

当读取发现某节点数据陈旧时，客户端可将最新值回写至该节点，称为**读修复**：

```text
读取 key "account:1"：
  节点 1：balance = 500（版本 2）✓ 最新
  节点 3：balance = 1000（版本 1）✗ 陈旧
  --> 向客户端返回 500
  --> 后台：将 balance=500 写回节点 3（读修复）
```

对于极少被读取的键，系统后台运行**反熵进程（anti-entropy process）**，定期比对副本间数据并修复差异。

## 分片（Partitioning / Sharding）

复制将**相同数据**放在多台机器上；分片则将**不同数据**放在不同机器上。这使你能够：
- 存储远超单机容量的数据
- 将写负载分散至多台机器
- 将热点数据就近部署给特定用户（地理分片）

![分区策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-partition-strategies.png)


### 基于范围的分片（Range-Based Partitioning）

按分区键的连续区间分配数据至各分片：

```text
分片 1：user_id    1 - 1,000,000
分片 2：user_id    1,000,001 - 2,000,000
分片 3：user_id    2,000,001 - 3,000,000
```

```sql
-- PostgreSQL 声明式分区（按范围）
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

-- 查询自动路由至正确分区
SELECT * FROM orders WHERE created_at = '2023-11-15';
-- 仅扫描 orders_2023_q4，跳过其他分区
```

**优势**：范围扫描高效（相邻键位于同一分片）。  
**劣势**：热点问题——若大量写入集中在最近时间戳，则最新分区将承受全部写压力。

### 基于哈希的分片（Hash-Based Partitioning）

对分区键应用哈希函数，再按哈希值取模分配至分片：

```text
分片编号 = hash(user_id) % 分片总数

hash("user:1")  = 0x3A2B... → 分片 2
hash("user:2")  = 0x8F1C... → 分片 0
hash("user:3")  = 0x12D4... → 分片 1
```

```sql
-- PostgreSQL 哈希分区
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

**优势**：数据与负载分布均匀。  
**劣势**：范围查询需访问所有分片（哈希破坏了顺序性）。

### 一致性哈希（Consistent Hashing）

传统 `hash(key) % N` 的问题在于：增减分片时，几乎所有键都会映射到新分片，引发海量数据迁移。

![一致性哈希环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/06-consistent-hashing.png)


一致性哈希通过将键与节点共同映射到一个环（0 到 2^32）来解决：

```text
                    0 / 2^32
                      │
              节点 C  ●
                     ╱  ╲
                    ╱    ╲
           ●──────╱──────●
         节点 A  ╱      节点 B
                ╱
               ╱
键 "user:42" 哈希至位置 X
→ 顺时针遍历 → 首个遇到的节点即为归属者

新增节点 D：
  仅节点 C 与 D 之间的键需迁移
  （而非全集群所有键）
```

引入**虚拟节点（vnodes）**：每个物理节点在环上占据多个位置，显著提升负载均衡：

```text
物理节点 A → 虚拟节点：A1, A2, A3, A4, A5（环上 5 个位置）
物理节点 B → 虚拟节点：B1, B2, B3, B4, B5

虚拟节点越多 → 分布越均衡
Cassandra 默认：每物理节点 256 个 vnode
```

### 重新平衡策略（Rebalancing Strategies）

增删节点时需迁移数据。主流方案有两种：

**固定分区数（Fixed number of partitions）**：预先创建远超当前节点数的分区（如 1000 个分区配 10 个节点）。扩容时，将部分完整分区迁移至新节点。

```text
扩容前（10 节点，1000 分区）：
  节点 1：分区 0–99
  节点 2：分区 100–199
  ...

新增节点 11 后：
  节点 1：分区 0–89（让出 10 个）
  节点 2：分区 100–189（让出 10 个）
  ...
  节点 11：分区 90–99, 190–199, ...（共接收约 91 个分区）
```

**动态分区（Dynamic partitioning）**：初始仅设少量分区；当分区过大时分裂，过小时合并。HBase 和 MongoDB 采用此方式。

### 分片数据库中的二级索引（Secondary Indexes）

主键查询简单直接——哈希或范围定位后直达目标分片。但二级索引（如按 email 查询）呢？

```sql
-- 表按 user_id 分片
-- 但我们还需按 email 查询
SELECT * FROM users WHERE email = 'alice@example.com';
-- 这个用户在哪一分片？无法得知，只能遍历全部分片。
```

两种方案：

**本地索引（Local / document-partitioned index）**：每个分片仅维护自身数据的二级索引。

```text
分片 1：email 本地索引 → {alice@...: row 1, bob@...: row 2}
分片 2：email 本地索引 → {carol@...: row 3, dave@...: row 4}

按 email 查询 → 向**所有分片**广播请求，聚合结果
（即 “scatter-gather” —— 扇出成本高昂）
```

**全局索引（Global / term-partitioned index）**：二级索引本身也按某种规则（如 email 字母范围）分片。

```text
Email 索引分片 A（a–m）：alice@... → 分片 1，carol@... → 分片 2
Email 索引分片 B（n–z）：zara@... → 分片 3

按 email 查询 → 先查索引分片 → 再查数据分片
（2 次跳转，但无需 scatter-gather）
```

| 方案 | 读开销 | 写开销 | 一致性 |
|------|--------|--------|--------|
| 本地索引 | 向所有分片广播 | 仅更新本地索引 | 始终强一致 |
| 全局索引 | 单分片查询 | 需跨网络更新远程索引分片 | 最终一致 |

## MySQL 主从复制实操指南


![一致性哈希环，形如带有数据的未来旋转木马](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/06-consistent-hashing-ring-as-a-futuristic-carousel-with-data-d.jpg)

下面我们搭建一个基础的 MySQL 主从复制环境。

### 主节点（Leader）配置

```bash
# /etc/mysql/mysql.conf.d/mysqld.cnf on the leader
[mysqld]
server-id           = 1
log_bin              = /var/log/mysql/mysql-bin
binlog_format        = ROW          # 最安全格式
binlog_expire_logs_seconds = 604800 # 7 天日志保留
sync_binlog          = 1            # 每次提交均刷盘 binlog
innodb_flush_log_at_trx_commit = 1  # 完整持久化
```

```sql
-- 在主节点创建复制用户
CREATE USER 'repl_user'@'%' IDENTIFIED BY 'strong_password_here';
GRANT REPLICATION SLAVE ON *.* TO 'repl_user'@'%';
FLUSH PRIVILEGES;

-- 获取当前 binlog 位置
SHOW MASTER STATUS;
-- +------------------+----------+
-- | File             | Position |
-- +------------------+----------+
-- | mysql-bin.000003 |      785 |
-- +------------------+----------+
```

### 获取一致性备份

```bash
# 方案 1：mysqldump + 一致性快照
mysqldump --all-databases --single-transaction \
  --source-data=2 --routines --triggers \
  -u root -p > leader_backup.sql

# 方案 2：大数据量场景使用 xtrabackup
xtrabackup --backup --target-dir=/backup/full \
  --user=root --password=xxx
```

### 从节点（Follower）配置

```bash
# /etc/mysql/mysql.conf.d/mysqld.cnf on the follower
[mysqld]
server-id            = 2       # 必须唯一
relay_log            = /var/log/mysql/mysql-relay
read_only            = ON      # 防止误写
super_read_only      = ON      # 连 root 也无法写
```

```sql
-- 在从节点恢复备份
-- 然后配置复制源
CHANGE REPLICATION SOURCE TO
    SOURCE_HOST='leader-hostname',
    SOURCE_USER='repl_user',
    SOURCE_PASSWORD='strong_password_here',
    SOURCE_LOG_FILE='mysql-bin.000003',
    SOURCE_LOG_POS=785;

-- 启动复制
START REPLICA;

-- 检查复制状态
SHOW REPLICA STATUS\G
-- 关键字段检查：
--   Replica_IO_Running: Yes
--   Replica_SQL_Running: Yes
--   Seconds_Behind_Source: 0
--   Last_Error: （应为空）
```

### 复制健康监控

```sql
-- 在从节点：检查延迟
SHOW REPLICA STATUS\G
-- Seconds_Behind_Source: 0  <-- 健康
-- Seconds_Behind_Source: 45 <-- 需关注
-- Seconds_Behind_Source: NULL <-- 复制已中断！

-- 在主节点：查看已连接的从节点
SHOW REPLICAS;
-- 或旧语法：SHOW SLAVE HOSTS;
```

```bash
# 快速健康检查脚本
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

### 故障转移（Failover）：提升从节点为主节点

当主节点宕机时，将某个从节点提升为主节点：

```sql
-- 在待提升的从节点上执行：
STOP REPLICA;
RESET REPLICA ALL;

-- 该节点现已成为独立服务器
-- 重新配置其余从节点，使其指向新主节点

-- 在剩余从节点上执行：
STOP REPLICA;
CHANGE REPLICATION SOURCE TO
    SOURCE_HOST='new-leader-hostname',
    SOURCE_LOG_FILE='...',
    SOURCE_LOG_POS=...;
START REPLICA;

-- 应用程序配置也需更新
-- （或使用 ProxySQL / HAProxy 等代理层）
```

生产环境中，应使用编排工具实现自动化故障转移：
- **Orchestrator**（MySQL）：自动检测主节点故障、提升从节点、重配复制拓扑  
- **Patroni**（PostgreSQL）：基于 etcd/ZooKeeper/Consul 实现主节点选举与高可用管理  
- **pg_auto_failover**：PostgreSQL 的轻量级高可用替代方案  

## 下一步


![分布式数据库复制数据流在节点间流动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/06-distributed-database-replication-data-streams-flowing-betwee.jpg)

复制与分片解决了数据跨多机部署的问题。但当一个事务需要同时更新多个机器上的数据时，又该如何保证原子性与一致性？这就是**分布式事务（Distributed Transactions）** 的挑战——两阶段提交（2PC）、Saga 模式、共识算法（Consensus），以及为何大多数工程师在可行时都尽量规避它。我们将在下一篇文章中深入探讨。
