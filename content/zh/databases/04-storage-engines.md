---
title: "数据库（四）：存储引擎——数据如何落盘"
date: 2024-04-22 09:00:00
tags:
  - Databases
  - Storage Engines
  - InnoDB
  - LSM Tree
categories: Databases
series: databases
lang: zh
description: "数据库存储引擎在底层如何工作——B 树 vs LSM 树、WAL、缓冲池、合并压缩（compaction），以及为何引擎选型从根本上塑造了你的数据库行为。"
disableNunjucks: true
series_order: 4
translationKey: "databases-4"
---
你写的每一条 SQL 语句，最终都会变成写入磁盘的字节流。负责完成这一转换的组件——**存储引擎（Storage Engine）**——对数据库性能的影响，几乎超过其他任何因素。两张结构完全相同、数据也完全一致的表，仅因底层存储引擎不同，性能表现可能天差地别。只有深入理解这一层，才能真正明白**数据库为何如此行为**。

---

## 基础概念：页（Page）、区（Extent）与表空间（Tablespace）

数据库不会按单行读写磁盘。磁盘 I/O 的最小单位是 **页（Page）**（也称块，Block），通常为 4 KB、8 KB 或 16 KB。

![数据库页面结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-page-structure.png)

```text
表空间（逻辑容器）
  └── 数据文件（磁盘上的物理文件）
       └── 区（Extent，一组连续的页，例如 64 页 = 1 MB）
            └── 页（Page，最小 I/O 单位，例如 PostgreSQL 为 8 KB，InnoDB 为 16 KB）
                 └── 行（Row，实际数据，被紧凑打包进页中）
```

| 概念 | PostgreSQL | MySQL InnoDB |
|---------|-----------|--------------|
| 页大小 | 8 KB（编译时固定） | 16 KB（可配置：4/8/16/32/64 KB） |
| 区（Extent） | 1 MB（128 页） | 1 MB（64 页），仅适用于 > 32 MB 的表 |
| 表空间 | 文件系统目录 | 每张表一个 `.ibd` 文件 |
| 行存储 | HEAP（无序） | 按主键聚簇（Clustered） |

当你执行 `SELECT * FROM users WHERE id = 42` 时，数据库并不会在磁盘上“寻道”到第 42 行。它会将**包含该行的整个页**加载进内存，再从中提取出目标行并返回。如果该页已在缓冲池（Buffer Pool，即内存缓存）中，则完全不需要磁盘 I/O。

## B-Tree 存储引擎

B-Tree 引擎（被 PostgreSQL、MySQL InnoDB、Oracle 和 SQL Server 采用）将数据以 B+ 树结构组织在磁盘上。这种树既是表数据本身，也是所有索引的基础。

### InnoDB：MySQL 的默认引擎

InnoDB 是一种 B-Tree 引擎，具有若干鲜明特性。

#### 聚簇索引（Clustered Index）

在 InnoDB 中，**表数据本身就是一个 B+ 树，并按主键排序组织**。这被称为 **聚簇索引（Clustered Index）**（或主索引）。

```text
聚簇索引（主键 = id）
根节点: [id=500 | id=1000 | id=1500]
       |          |           |
    叶子页（按 id 排序的实际数据行）：
    页 1: id=1, name="Alice", email="alice@..."
         id=2, name="Bob",   email="bob@..."
         ...
    页 2: id=501, name="Carol", email="carol@..."
         ...
```

InnoDB 中的每一行都物理存在于聚簇索引内，**不存在独立的“堆（Heap）”文件**。

影响如下：

- **顺序主键访问极快**：读取 `id = 1, 2, 3, ...` 会顺序读取连续的页。
- **随机主键插入代价高昂**：使用 UUID 作为主键插入时，会在 B+ 树中间位置频繁触发页分裂（Page Split）。
- **二级索引体积更大**：二级索引存储的是主键值（而非物理地址）作为行指针。通过二级索引查找需要两次树遍历——一次查二级索引，一次查聚簇索引。

```sql
-- 此二级索引存储 (email -> primary_key) 键值对
CREATE INDEX idx_email ON users (email);

-- 查询流程：
-- 1. 遍历 idx_email B+ 树，找到 email='alice@...' → id=1
-- 2. 遍历聚簇索引 B+ 树，找到 id=1 → 完整行数据
-- 这称为“书签查找（Bookmark Lookup）”或“索引查找（Index Lookup）”
```

#### 为何 InnoDB 中自增主键至关重要

```sql
-- 优秀：顺序插入追加至 B+ 树末尾
CREATE TABLE orders (
    order_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ...
);

-- 问题：随机 UUID 导致全树范围内的随机插入
CREATE TABLE orders (
    order_id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    ...
);
```

使用随机主键会引发以下问题：

1. **页分裂（Page Splits）**：当页已满而新行需插入中间位置时，该页必须一分为二。
2. **碎片化（Fragmentation）**：分裂后页的填充率下降，空间利用率降低。
3. **缓冲池抖动（Buffer Pool Thrashing）**：随机访问不同页会导致大量热数据被挤出缓存。

如果业务确实需要 UUID，建议使用时间有序的 UUIDv7，或者采用 `BIGINT AUTO_INCREMENT` 作为主键，并额外添加一个 UUID 字段。

#### 缓冲池（Buffer Pool）

InnoDB 的缓冲池是一块内存区域，用于缓存数据页和索引页，也是最重要的性能相关配置项。

```sql
-- 查看缓冲池大小（MySQL）
SHOW VARIABLES LIKE 'innodb_buffer_pool_size';

-- 将缓冲池设为可用内存的 80%（典型建议）
-- 在 my.cnf 中：
-- innodb_buffer_pool_size = 12G  （针对 16 GB 内存服务器）

-- 监控缓冲池命中率
SHOW STATUS LIKE 'Innodb_buffer_pool_read%';
-- Innodb_buffer_pool_read_requests: 总逻辑读次数
-- Innodb_buffer_pool_reads: 实际触发磁盘读的次数
-- 命中率 = 1 - (reads / read_requests)
-- 目标：> 99%
```

缓冲池采用一种改进的 **LRU（Least Recently Used）算法**，内部划分为两个子链表：

```text
缓冲池 LRU：
┌─────────────────────────────────────────────────────────────┐
│  新子链表（热页，占 5/8）  │  旧子链表（占 3/8）      │
│  [频繁访问的页]           │  [新加载的页]           │
│                            │  [老化淘汰中]           │
└─────────────────────────────────────────────────────────────┘
```

新页首先插入旧子链表头部。若在可配置的时间窗口内再次被访问，则晋升至新子链表。这种设计能防止一次性全表扫描将所有热页全部挤出缓存——纯 LRU 算法就会出现这种情况。

### PostgreSQL 存储

PostgreSQL 采用了不同的策略。表以 **堆文件（Heap File）** 形式存储，即页的无序集合，默认情况下**没有聚簇索引**。

```text
表 "users"（堆）：
页 0: [行: id=7, ...] [行: id=3, ...] [行: id=12, ...]
页 1: [行: id=1, ...] [行: id=9, ...] [死元组] [行: id=5, ...]
页 2: [行: id=15, ...] [行: id=2, ...] [行: id=11, ...]
```

行之间没有任何特定顺序。所有索引（包括主键索引）都存储一个物理元组 ID `(page_number, offset)`，指向堆中的具体位置。

```text
主键索引（B+ 树）：
  id=1 → (页 1, 偏移 0)
  id=2 → (页 2, 偏移 1)
  id=3 → (页 0, 偏移 1)
  ...
```

这意味着：

- **主键查找需一次索引遍历 + 一次堆获取**（开销与 InnoDB 的二级索引查找相当）
- **随机主键无惩罚**——因为堆本身就是无序的
- **VACUUM 至关重要**——删除或更新操作留下的死元组（Dead Tuples）必须被回收

PostgreSQL 的缓冲池称为 **共享缓冲池（Shared Buffer Pool）**，通过 `shared_buffers` 参数配置：

```bash
# postgresql.conf
shared_buffers = 4GB      # 典型起始点为内存的 25%
effective_cache_size = 12GB  # 操作系统磁盘缓存估算值（供查询规划器使用）
```

## LSM-Tree 存储引擎

![存储引擎内部的LSM树压缩过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/04-storage-engine-internals-lsm-tree-compaction-process-like-ge.jpg)

日志结构合并树（Log-Structured Merge-tree, LSM-tree）引擎采取了截然不同的设计思路。它不就地更新数据（如 B-Tree 所做），而是将写操作批量暂存于内存，再以**顺序方式**刷入磁盘。这种方式大幅提升了写入速度，但使读取路径变得更复杂。

典型的 LSM-Tree 引擎包括：RocksDB、LevelDB、Cassandra 的存储引擎、HBase、CockroachDB（基于 RocksDB 构建）以及 TiKV（TiDB 的存储层）。

### LSM-Tree 工作原理

```text
写入路径（Write Path）：
1. 写入 WAL（仅追加，顺序写入）
2. 插入 MemTable（内存中有序结构，通常为跳表或红黑树）
3. 当 MemTable 满（约 64 MB）时，冻结它并新建一个 MemTable
4. 将冻结的 MemTable 刷入磁盘，生成 SSTable（Sorted String Table）
5. 后台进行合并压缩（Compaction）以合并 SSTables

读取路径（Read Path）：
1. 检查 MemTable（当前 + 已冻结）
2. 检查各层级 SSTable（利用布隆过滤器 Bloom Filter 跳过不含目标 key 的 SSTable）
3. 合并结果（最新版本胜出）
```

```text
                    ┌─────────────┐
  写入 ──────────► │  MemTable   │ （内存中，有序）
                    └──────┬──────┘
                           │ 刷盘（满时）
                    ┌──────▼──────┐
                    │  Level 0    │  SSTable 文件（文件间无序）
                    └──────┬──────┘
                           │ 合并压缩
                    ┌──────▼──────┐
                    │  Level 1    │  SSTable 文件（有序，key 范围不重叠）
                    └──────┬──────┘
                           │ 合并压缩
                    ┌──────▼──────┐
                    │  Level 2    │  SSTable 文件（有序，不重叠，容量为 Level 1 的 10 倍）
                    └──────┬──────┘
                           │ 合并压缩
                    ┌──────▼──────┐
                    │  Level 3    │  （容量再翻 10 倍）
                    └─────────────┘
```

### SSTable（排序字符串表）

SSTable 是磁盘上一个**不可变、有序**的文件。一旦写入，就永不修改——只能在合并压缩过程中被替换。

```text
SSTable 格式：
┌───────────────────────────────────────────────────┐
│  数据块 1: [key1=val1, key2=val2, ...]        │
│  数据块 2: [key5=val5, key6=val6, ...]        │
│  ...                                              │
│  索引块: [key1 -> 块1, key5 -> 块2...] │
│  布隆过滤器: [用于快速 key 查找的位图]      │
│  页脚: [元数据, 魔数]                 │
└───────────────────────────────────────────────────┘
```

### 合并压缩（Compaction）

随着时间推移，SSTable 数量不断累积。多个 SSTable 可能包含同一 key 的不同版本。合并压缩（Compaction）会将这些 SSTable 合并，以实现：

- 清除过时版本（仅保留最新版）
- 清除墓碑（Tombstones，即删除标记）
- 减少读取时需检查的 SSTable 数量

![LSM树压缩](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-compaction-process.png)

两种主流合并压缩策略：

**大小分层合并（Size-Tiered Compaction）**（写优化）：
- 将大小相近的 SSTable 合并
- 优点：更高的写吞吐量
- 缺点：更高的空间放大（Space Amplification，最高达 2x），合并期间读延迟可能飙升

**层级合并（Leveled Compaction）**（读优化）：
- 每一层有大小上限（Level N+1 是 Level N 的 10 倍）
- 每一层的 SSTable 具有互不重叠的 key 范围
- 优点：更可预测的读性能，更低的空间放大
- 缺点：更高的写放大（Write Amplification）

### 布隆过滤器（Bloom Filters）：避免不必要的磁盘读取

布隆过滤器是一种概率性数据结构，可以告诉你：

- **绝对不在集合中** → 跳过此 SSTable（节省一次磁盘读！）
- **可能在集合中** → 必须检查此 SSTable

```text
查找 key "user:42"：
  SSTable-1 布隆过滤器: "user:42" → NO  → 跳过（省下一次磁盘读！）
  SSTable-2 布隆过滤器: "user:42" → MAYBE → 读取 SSTable-2 → 找到了！
  SSTable-3 布隆过滤器: → 不再检查（已找到）
```

误报率（False Positive Rate）可配置（通常为 1%）。1% 的误报率大约需要为每个 key 分配 10 位（bits）。

## B-Tree vs LSM-Tree 对比

![硬盘内部结构，数据页像书页一样排列](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/04-hard-drive-internals-with-data-pages-arranged-like-book-page.jpg)

| 维度 | B-Tree（InnoDB, PostgreSQL） | LSM-Tree（RocksDB, LevelDB） |
|--------|---------------------------|----------------------------|
| **写入模式** | 随机（就地更新） | 顺序（仅追加） |
| **写入吞吐量** | 较低（随机 I/O） | 更高（顺序 I/O） |
| **读取延迟** | 可预测（单次树遍历） | 可变（可能需检查多层） |
| **写放大（Write Amplification）** | ~10x（页重写） | ~10–30x（合并压缩重写） |
| **读放大（Read Amplification）** | 1（单次索引遍历） | ~1–5（需检查多层） |
| **空间放大（Space Amplification）** | ~1.5x（页碎片） | ~1.1–2x（取决于合并策略） |
| **适用场景** | 读密集型 OLTP、随机读 | 写密集型负载、时序数据 |
| **并发控制** | MVCC + 行锁 | 无锁读（SSTable 不可变） |

![B树与LSM树的权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-btree-vs-lsm.png)

**写放大（Write Amplification）**：磁盘总写入字节数 / 应用逻辑写入字节数。例如，你写入 1 KB 数据，但引擎共写入 10 KB（含索引更新、页重写、合并压缩），则写放大为 10x。

**读放大（Read Amplification）**：回答一次点查询（Point Query）所需的磁盘读次数。B-Tree 通常只需 1 次（页已缓存，或单次树遍历）；LSM-Tree 则可能每层都要读一次。

**空间放大（Space Amplification）**：磁盘总占用空间 / 实际数据大小。B-Tree 因页未填满而浪费空间；LSM-Tree 则因合并前多版本数据跨层存在而浪费空间。

## 预写日志（Write-Ahead Log, WAL）

WAL（在 MySQL 中也称重做日志 Redo Log）是持久性的基石。**任何数据页被修改前，其变更必须先写入 WAL**。

![预写日志流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-wal-flow.png)

```text
带 WAL 的写入流程：
1. 应用程序: INSERT INTO users (name) VALUES ('Alice')
2. 引擎写入 WAL: "页 42，偏移 3：插入行 {name='Alice'}"
3. WAL 调用 fsync 刷盘（此时持久性已保证）
4. 引擎在缓冲池中修改该页（仅内存）
5. 最终，脏页被刷入数据文件（Checkpoint 检查点）
```

为什么不直接写数据文件？

- 数据文件需要随机 I/O（该行可能在任意位置）
- WAL 是仅追加（Sequential I/O），速度比随机 I/O 快 10–100 倍
- 如果系统在步骤 4 和 5 之间崩溃，WAL 中的信息足以重放（Replay）所有变更

```sql
-- PostgreSQL WAL 配置
SHOW wal_level;          -- minimal, replica, or logical
SHOW max_wal_size;       -- 超过此值触发检查点（默认：1GB）
SHOW checkpoint_timeout; -- 检查点最大间隔（默认：5分钟）

-- MySQL 重做日志配置
SHOW VARIABLES LIKE 'innodb_log_file_size';    -- 每个日志文件大小
SHOW VARIABLES LIKE 'innodb_log_files_in_group'; -- 日志文件数量
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit';
-- 1 = 每次提交都刷盘（最安全，默认）
-- 2 = 每次提交刷到 OS 缓存（更快，风险：OS 崩溃丢失）
-- 0 = 每秒刷一次（最快，风险：最多丢失 1 秒数据）
```

### 检查点（Checkpoint）

**检查点（Checkpoint）** 是指缓冲池中所有脏页被刷入磁盘数据文件的时刻。在此之后，该点之前的 WAL 条目就不再需要用于崩溃恢复。

![检查点流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-checkpoint-flow.png)

```text
时间线：
  WAL: [条目1] [条目2] [条目3] [CHECKPOINT] [条目4] [条目5]
                                       ↑
                              数据文件在此点前已一致
                              崩溃后只需重放条目4 和 条目5
```

如果没有检查点，崩溃恢复就需要重放整个 WAL 历史。检查点有效限定了恢复所需的时间。

## InnoDB 架构全景图

```text
客户端连接
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                     InnoDB 引擎                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              缓冲池（Buffer Pool，内存中）                │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │     │
│  │  │数据页    │ │数据页    │ │索引页    │  ...        │     │
│  │  │（干净）  │ │（脏页）  │ │（干净）  │            │     │
│  │  └──────────┘ └──────────┘ └──────────┘            │     │
│  └─────────────────────────┬───────────────────────────┘     │
│                            │                                  │
│  ┌────────────┐    ┌───────▼──────┐    ┌─────────────┐      │
│  │ 变更       │    │ 重做日志     │    │ 回滚日志    │      │
│  │ 缓冲       │    │ （WAL）      │    │ （用于 MVCC │      │
│  │ （二级索引 │    │              │    │  和回滚）   │      │
│  │  变更）    │    │ 仅顺序写入   │    │             │      │
│  └────────────┘    └───────┬──────┘    └─────────────┘      │
│                            │                                  │
│  ┌─────────────────────────▼──────────────────────────┐      │
│  │              数据文件（.ibd）                      │      │
│  │  磁盘上的表空间文件                               │      │
│  │  （聚簇索引 + 二级索引）                          │      │
│  └─────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

一次 UPDATE 的完整流程如下：

1. 在缓冲池中定位目标页（若不在，则从磁盘加载）
2. 将变更写入重做日志（WAL）
3. 在缓冲池中修改该页（标记为脏页）
4. 将旧行版本写入回滚日志（Undo Log，用于 MVCC 和回滚）
5. 提交（COMMIT）时：将重做日志刷盘
6. 最终：检查点将脏页刷入数据文件

## 压缩（Compression）

### 页级压缩（Page Compression）

InnoDB 支持透明页级压缩：

```sql
-- 创建启用压缩的表（MySQL 5.7+）
CREATE TABLE logs (
    log_id BIGINT PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP
) ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;

-- 查看压缩率
SELECT
    table_name,
    ROUND(data_length / 1024 / 1024, 2) AS data_mb,
    ROUND(data_free / 1024 / 1024, 2) AS free_mb
FROM information_schema.tables
WHERE table_schema = 'mydb' AND table_name = 'logs';
```

PostgreSQL 使用 TOAST（The Oversized-Attribute Storage Technique）处理大值：

```sql
-- 大于 ~2 KB 的值会自动压缩和/或移出主行存储（out-of-line）
-- 基础压缩无需显式配置

-- 查看 TOAST 表使用情况
SELECT
    relname,
    pg_size_pretty(pg_total_relation_size(oid)) AS total_size,
    pg_size_pretty(pg_relation_size(oid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(oid) - pg_relation_size(oid)) AS toast_size
FROM pg_class
WHERE relname = 'logs';
```

### 透明数据加密（Transparent Data Encryption, TDE）

在存储引擎层面进行加密——数据在磁盘上加密，但加载进缓冲池时自动解密：

```sql
-- MySQL 8.0：为每个表空间启用加密
ALTER TABLE users ENCRYPTION = 'Y';

-- PostgreSQL：使用 pgcrypto 进行列级加密
-- 或在操作系统层面使用全盘加密（dm-crypt, LUKS）
```

## 列式存储（Column-Oriented Storage）

截至目前我们讨论的所有内容均为 **行式存储（Row-Oriented）**：每一页包含完整的行。这对 OLTP（事务型）工作负载极为高效，因为通常只需访问少数几行的全部列。

而对于 **分析型（OLAP）** 工作负载，列式存储则具有压倒性优势：

```json
行式存储（适合：SELECT * FROM users WHERE id = 42）：
页 1: [id=1, name="Alice", email="a@...", age=30]
         [id=2, name="Bob",   email="b@...", age=25]

列式存储（适合：SELECT AVG(age) FROM users）：
列 "id":    [1, 2, 3, 4, 5, ...]     -- 集中存储
列 "name":  ["Alice", "Bob", ...]     -- 集中存储
列 "email": ["a@..", "b@..", ...]     -- 集中存储
列 "age":   [30, 25, 35, 28, ...]    -- 集中存储
```

为何列式存储更适合分析：

| 优势 | 解释 |
|---------|-------------|
| **I/O 更少** | `SELECT AVG(age)` 仅需读取 age 列，无需 name/email 等无关列 |
| **压缩率更高** | 同一列内值高度相似，压缩率可达 5–10 倍 |
| **向量化处理** | CPU 可使用 SIMD 指令批量处理同类型值 |
| **缓存效率高** | 所有被处理的值能紧密装入 CPU 缓存行 |

列式数据库包括：ClickHouse、Apache Parquet（文件格式）、DuckDB、Amazon Redshift、Google BigQuery。

```sql
-- ClickHouse 示例：分析查询
SELECT
    toStartOfMonth(event_time) AS month,
    count() AS events,
    uniqExact(user_id) AS unique_users,
    avg(duration_ms) AS avg_duration
FROM events
WHERE event_time >= '2023-01-01'
GROUP BY month
ORDER BY month;
-- 因为只读取所需的 4 列（而非表中全部 50 列），可在秒级扫描数十亿行。
```

大多数 OLTP 数据库并不采用列式存储（InnoDB、PostgreSQL 均为行式），但混合方案正在兴起：

- PostgreSQL 的 **cstore_fdw**（列式外部数据包装器）
- MySQL HeatWave（内存中列式加速器）
- Oracle 的内存列存储（In-Memory Column Store）
- SQL Server 的列存储索引（Columnstore Indexes）

## 存储引擎性能度量

评估存储引擎时，三个“放大系数（Amplification）”指标最为关键。让我们用具体数字使其具象化。

### 实践中的写放大（Write Amplification）

写放大 = 存储设备总写入字节数 / 应用程序逻辑写入字节数。

```text
示例：插入一条 1 KB 的行

B-Tree 引擎（InnoDB）：
  1. 写入重做日志：           1 KB  （WAL 条目）
  2. 修改缓冲池中的页：      0 KB  （内存中，暂无 I/O）
  3. 最终刷出 16 KB 页：      16 KB （为 1 KB 变更重写整页）
  4. 更新二级索引：          16 KB （另一次页写入）
  总计：约 33 KB 写入对应 1 KB 数据 → 写放大 = 33x

LSM-Tree 引擎（RocksDB）：
  1. 写入 WAL：                1 KB
  2. 刷入 MemTable 到 L0：     1 KB  （写入一次）
  3. L0 → L1 合并：            1 KB  （重写一次）
  4. L1 → L2 合并：            1 KB  （重写一次）
  5. L2 → L3 合并：            1 KB  （重写一次）
  总计：约 5 KB 顺序写入 → 写放大 = 5x
  但若有 10 层：可能达 10–30x
```

### 使用真实工具进行基准测试

```bash
# sysbench：标准数据库基准测试工具
# 安装
apt-get install sysbench

# 准备测试数据（100 万行）
sysbench oltp_read_write \
  --db-driver=mysql \
  --mysql-host=localhost \
  --mysql-user=root \
  --mysql-password=secret \
  --mysql-db=sbtest \
  --tables=4 \
  --table-size=1000000 \
  prepare

# 运行 OLTP 读写基准测试（60 秒，16 线程）
sysbench oltp_read_write \
  --db-driver=mysql \
  --mysql-host=localhost \
  --mysql-user=root \
  --mysql-password=secret \
  --mysql-db=sbtest \
  --tables=4 \
  --table-size=1000000 \
  --threads=16 \
  --time=60 \
  --report-interval=5 \
  run
```

样例输出：

```text
SQL statistics:
    queries performed:
        read:                            560420
        write:                           160120
        other:                           80060
        total:                           800600
    transactions:                        40030  (667.11 per sec.)
    queries:                             800600 (13342.20 per sec.)
    ignored errors:                      0      (0.00 per sec.)
    reconnects:                          0      (0.00 per sec.)

Latency (ms):
         min:                                    3.42
         avg:                                   23.97
         max:                                  245.12
         95th percentile:                       41.85
```

```bash
# fio：测试原始磁盘 I/O（对理解引擎行为至关重要）
# 随机读测试（模拟 B-Tree 索引查找）
fio --name=randread --ioengine=libaio --iodepth=32 \
  --rw=randread --bs=8k --size=1G --numjobs=4 \
  --runtime=30 --group_reporting

# 顺序写测试（模拟 WAL/LSM 写入）
fio --name=seqwrite --ioengine=libaio --iodepth=32 \
  --rw=write --bs=64k --size=1G --numjobs=1 \
  --runtime=30 --group_reporting
```

硬件上**随机读 IOPS** 与**顺序写吞吐量**的比值，直接解释了为何 B-Tree 引擎和 LSM-Tree 引擎在同一台机器上性能迥异。

## 如何选择存储引擎

| 工作负载 | 推荐引擎 | 原因 |
|----------|-------------------|-----|
| 通用 OLTP | InnoDB / PostgreSQL | 成熟稳定、ACID、MVCC |
| 写密集型（日志、指标） | 基于 LSM（RocksDB、TiKV） | 顺序写入、高吞吐 |
| 分析型 / OLAP | 列式（ClickHouse、DuckDB） | 扫描效率高、压缩率优 |
| 嵌入式 / 边缘端 | SQLite / DuckDB | 零配置、单文件 |
| 键值型（Key-Value） | RocksDB / LevelDB | 针对简单 get/put 优化 |
| 时序型（Time-Series） | TimescaleDB / InfluxDB | 时间分区、数据保留策略 |

## 下一步

至此，我们已了解了单机环境下各类存储引擎如何组织数据。但并非所有数据都适合放入关系表中，也并非所有工作负载都由 SQL 最优服务。在下一篇文章中，我们将探索 **NoSQL 生态**——文档数据库、键值存储、宽列数据库与图数据库——并阐明每种技术的适用场景。
