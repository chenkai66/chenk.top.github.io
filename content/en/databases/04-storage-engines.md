---
title: "Databases (4): Storage Engines — How Data Hits Disk"
date: 2024-04-22 09:00:00
tags:
  - Databases
  - Storage Engines
  - InnoDB
  - LSM Tree
categories: Databases
series: databases
lang: en
description: "How database storage engines work under the hood — B-tree vs LSM-tree, WAL, buffer pools, compaction, and why your choice of engine shapes everything."
disableNunjucks: true
series_order: 4
translationKey: "databases-4"
---

Every SQL statement you write eventually becomes bytes written to a disk. The component responsible for this translation — the storage engine — determines your database's performance characteristics more than almost any other factor. Two tables with identical schemas and identical data can perform wildly differently depending on the storage engine underneath. Understanding this layer explains *why* databases behave the way they do.

## The Basics: Pages, Extents, and Tablespaces

Databases do not read or write individual rows from disk. Disk I/O operates on **pages** (also called blocks), typically 4 KB, 8 KB, or 16 KB.

![Database page structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-page-structure.png)


```
Tablespace (logical container)
  └── Data File (physical file on disk)
       └── Extent (group of contiguous pages, e.g., 64 pages = 1 MB)
            └── Page (smallest I/O unit, e.g., 8 KB in PostgreSQL, 16 KB in InnoDB)
                 └── Row (actual data, packed into pages)
```

| Concept | PostgreSQL | MySQL InnoDB |
|---------|-----------|--------------|
| Page size | 8 KB (compile-time) | 16 KB (configurable: 4/8/16/32/64 KB) |
| Extent | 1 MB (128 pages) | 1 MB (64 pages) for tables > 32 MB |
| Tablespace | Directories on filesystem | `.ibd` files per table |
| Row storage | HEAP (unordered) | Clustered by primary key |

When you `SELECT * FROM users WHERE id = 42`, the database does not seek to row 42 on disk. It loads the page containing row 42 into memory, extracts the row, and returns it. If the page is already in the buffer pool (in-memory cache), no disk I/O is needed at all.

## B-Tree Storage Engines

B-tree engines (used by PostgreSQL, MySQL InnoDB, Oracle, SQL Server) organize data in B+tree structures on disk. The tree is the primary data structure for both tables and indexes.

### InnoDB: MySQL's Default Engine

InnoDB is a B-tree engine with some distinctive characteristics.

#### Clustered Index

In InnoDB, the table data itself *is* a B+tree, organized by the primary key. This is called the **clustered index** (or primary index).

```
Clustered Index (primary key = id)
Root: [id=500 | id=1000 | id=1500]
       |          |           |
    Leaf pages (actual data rows sorted by id):
    Page 1: id=1, name="Alice", email="alice@..."
             id=2, name="Bob",   email="bob@..."
             ...
    Page 2: id=501, name="Carol", email="carol@..."
             ...
```

Every row in InnoDB lives inside the clustered index. There is no separate "heap" file.

Implications:

- **Sequential primary key access is fast**: Reading `id = 1, 2, 3, ...` reads consecutive pages.
- **Random primary key inserts are expensive**: Inserting with a UUID primary key causes random page splits across the tree.
- **Secondary indexes are larger**: A secondary index stores the primary key as the row pointer, not a physical address. Looking up by secondary index requires two tree traversals — one for the secondary index, one for the clustered index.

```sql
-- This secondary index stores (email -> primary_key) pairs
CREATE INDEX idx_email ON users (email);

-- Query flow:
-- 1. Traverse idx_email B+tree to find email='alice@...' -> id=1
-- 2. Traverse clustered index B+tree to find id=1 -> full row
-- This is called a "bookmark lookup" or "index lookup"
```

#### Why Auto-Increment Primary Keys Matter in InnoDB

```sql
-- Good: sequential inserts append to the end of the B+tree
CREATE TABLE orders (
    order_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ...
);

-- Problematic: random UUIDs cause random inserts across the tree
CREATE TABLE orders (
    order_id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    ...
);
```

Random primary keys cause:
1. **Page splits**: When a page is full and a new row belongs in the middle, the page must be split in two.
2. **Fragmentation**: Pages are only partially filled after splits.
3. **Buffer pool thrashing**: Random pages are accessed, evicting useful cached pages.

If you need UUIDs, consider UUIDv7 (time-ordered) or use a `BIGINT AUTO_INCREMENT` primary key with a separate UUID column.

#### Buffer Pool

InnoDB's buffer pool is a region of memory that caches data pages and index pages. It is the most important performance-related configuration.

```sql
-- Check buffer pool size (MySQL)
SHOW VARIABLES LIKE 'innodb_buffer_pool_size';

-- Set buffer pool to 80% of available RAM (typical recommendation)
-- In my.cnf:
-- innodb_buffer_pool_size = 12G  (for a 16 GB server)

-- Monitor buffer pool hit ratio
SHOW STATUS LIKE 'Innodb_buffer_pool_read%';
-- Innodb_buffer_pool_read_requests: total logical reads
-- Innodb_buffer_pool_reads: reads that went to disk
-- Hit ratio = 1 - (reads / read_requests)
-- Target: > 99%
```

The buffer pool uses a modified **LRU (Least Recently Used)** algorithm with two sublists:

```
Buffer Pool LRU:
┌─────────────────────────────────────────────────────────────┐
│  Young sublist (hot pages, 5/8)  │  Old sublist (3/8)      │
│  [Frequently accessed pages]     │  [Newly loaded pages]   │
│                                  │  [Aging out]            │
└─────────────────────────────────────────────────────────────┘
```

New pages enter at the head of the old sublist. If accessed again (within a configurable window), they move to the young sublist. This prevents a one-time full table scan from evicting all hot pages — a pure LRU would push every hot page out.

### PostgreSQL Storage

PostgreSQL uses a different approach. Tables are stored as **heap files** — unordered collections of pages. There is no clustered index by default.

```
Table "users" (heap):
Page 0: [row: id=7, ...] [row: id=3, ...] [row: id=12, ...]
Page 1: [row: id=1, ...] [row: id=9, ...] [dead tuple] [row: id=5, ...]
Page 2: [row: id=15, ...] [row: id=2, ...] [row: id=11, ...]
```

Rows are not in any particular order. Every index (including the primary key index) stores a physical tuple ID `(page_number, offset)` pointing to the heap.

```
Primary Key Index (B+tree):
  id=1 -> (page 1, offset 0)
  id=2 -> (page 2, offset 1)
  id=3 -> (page 0, offset 1)
  ...
```

This means:
- **Primary key lookups require one index traversal + one heap fetch** (same as InnoDB secondary indexes)
- **No penalty for random primary keys** — the heap is already unordered
- **VACUUM is critical** — deleted/updated rows leave dead tuples in the heap that must be reclaimed

PostgreSQL's buffer pool is called the **shared buffer pool**, configured via `shared_buffers`:

```bash
# postgresql.conf
shared_buffers = 4GB      # 25% of RAM is typical starting point
effective_cache_size = 12GB  # estimate of OS disk cache (for planner)
```

## LSM-Tree Storage Engines


![Storage engine internals lsm tree compaction process like ge](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/04-storage-engine-internals-lsm-tree-compaction-process-like-ge.jpg)

Log-Structured Merge-tree (LSM-tree) engines take a fundamentally different approach. Instead of updating data in place (like B-trees), they batch writes in memory and flush them sequentially. This makes writes much faster but reads more complex.

LSM-tree engines include: RocksDB, LevelDB, Cassandra's storage engine, HBase, CockroachDB (built on RocksDB), and TiKV (TiDB's storage layer).

### How LSM-Trees Work

```
Write Path:
1. Write to WAL (append-only, sequential)
2. Insert into MemTable (in-memory sorted structure, usually a skip list or red-black tree)
3. When MemTable is full (~64 MB), freeze it and start a new MemTable
4. Flush frozen MemTable to disk as an SSTable (Sorted String Table)
5. Background compaction merges SSTables

Read Path:
1. Check MemTable (current + frozen)
2. Check each SSTable level (using Bloom filters to skip non-matching SSTables)
3. Merge results (most recent version wins)
```

```
                    ┌─────────────┐
  Write ──────────► │  MemTable   │ (in-memory, sorted)
                    └──────┬──────┘
                           │ flush (when full)
                    ┌──────▼──────┐
                    │  Level 0    │  SSTable files (unsorted between files)
                    └──────┬──────┘
                           │ compaction
                    ┌──────▼──────┐
                    │  Level 1    │  SSTable files (sorted, non-overlapping)
                    └──────┬──────┘
                           │ compaction
                    ┌──────▼──────┐
                    │  Level 2    │  SSTable files (sorted, non-overlapping, 10x larger)
                    └──────┬──────┘
                           │ compaction
                    ┌──────▼──────┐
                    │  Level 3    │  (10x larger again)
                    └─────────────┘
```

### SSTables (Sorted String Tables)

An SSTable is an immutable, sorted file on disk. Once written, it is never modified — only replaced during compaction.

```
SSTable format:
┌───────────────────────────────────────────────────┐
│  Data Block 1: [key1=val1, key2=val2, ...]        │
│  Data Block 2: [key5=val5, key6=val6, ...]        │
│  ...                                              │
│  Index Block: [key1 -> block1, key5 -> block2...] │
│  Bloom Filter: [bitmap for quick key lookup]      │
│  Footer: [metadata, magic number]                 │
└───────────────────────────────────────────────────┘
```

### Compaction

Over time, SSTables accumulate. Multiple SSTables may contain different versions of the same key. Compaction merges SSTables to:
- Remove obsolete versions (keep only the latest)
- Remove tombstones (delete markers)
- Reduce the number of SSTables to check during reads

![LSM-tree compaction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-compaction-process.png)


Two main compaction strategies:

**Size-Tiered Compaction** (write-optimized):
- Merge SSTables of similar size together
- Pros: Higher write throughput
- Cons: More space amplification (up to 2x), read latency spikes during compaction

**Leveled Compaction** (read-optimized):
- Each level has a size limit (level N+1 is 10x level N)
- SSTables at each level have non-overlapping key ranges
- Pros: More predictable read performance, lower space amplification
- Cons: Higher write amplification

### Bloom Filters: Avoiding Unnecessary Disk Reads

A Bloom filter is a probabilistic data structure that tells you:
- **Definitely not in the set** — skip this SSTable (no disk read needed)
- **Possibly in the set** — must check this SSTable

```
Looking for key "user:42":
  SSTable-1 Bloom Filter: "user:42" → NO  → skip (saved a disk read!)
  SSTable-2 Bloom Filter: "user:42" → MAYBE → read SSTable-2 → found it!
  SSTable-3 Bloom Filter: → not checked (already found)
```

False positive rate is configurable (typically 1%). A 1% false positive rate requires about 10 bits per key.

## B-Tree vs LSM-Tree Comparison


![Hard drive internals with data pages arranged like book page](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/04-hard-drive-internals-with-data-pages-arranged-like-book-page.jpg)

| Aspect | B-Tree (InnoDB, PostgreSQL) | LSM-Tree (RocksDB, LevelDB) |
|--------|---------------------------|----------------------------|
| **Write pattern** | Random (update in place) | Sequential (append-only) |
| **Write throughput** | Lower (random I/O) | Higher (sequential I/O) |
| **Read latency** | Predictable (single tree traversal) | Variable (may check multiple levels) |
| **Write amplification** | ~10x (page rewrites) | ~10-30x (compaction rewrites) |
| **Read amplification** | 1 (one index traversal) | ~1-5 (multiple levels to check) |
| **Space amplification** | ~1.5x (page fragmentation) | ~1.1-2x (depends on compaction) |
| **Best for** | Read-heavy OLTP, random reads | Write-heavy workloads, time-series |
| **Concurrency** | MVCC + row locks | Lock-free reads (immutable SSTables) |

![B-tree vs LSM-tree tradeoffs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-btree-vs-lsm.png)


**Write amplification**: total bytes written to disk / bytes of actual data written. If you write 1 KB of data but the engine writes 10 KB total (including index updates, page rewrites, compaction), the write amplification is 10x.

**Read amplification**: number of disk reads needed to answer a point query. B-trees: typically 1 (the page is cached or one tree traversal). LSM-trees: potentially one read per level.

**Space amplification**: total disk space used / actual data size. B-trees waste space due to partially-filled pages. LSM-trees waste space due to multiple versions existing across levels before compaction.

## Write-Ahead Log (WAL)

The WAL (also called redo log in MySQL) is the foundation of durability. Before any data page is modified, the change is written to the WAL.

![Write-ahead log flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-wal-flow.png)


```
Write flow with WAL:
1. Application: INSERT INTO users (name) VALUES ('Alice')
2. Engine writes to WAL: "Page 42, offset 3: insert row {name='Alice'}"
3. WAL is fsynced to disk (durability guaranteed)
4. Engine modifies the page in the buffer pool (memory only)
5. Eventually, dirty pages are flushed to data files (checkpoint)
```

Why not write directly to data files?
- Data files require random I/O (the row could be anywhere)
- WAL is append-only (sequential I/O), which is 10-100x faster
- If the system crashes between step 4 and 5, the WAL contains enough information to replay the changes

```sql
-- PostgreSQL WAL configuration
SHOW wal_level;          -- minimal, replica, or logical
SHOW max_wal_size;       -- triggers checkpoint when exceeded (default: 1GB)
SHOW checkpoint_timeout; -- max time between checkpoints (default: 5min)

-- MySQL redo log configuration
SHOW VARIABLES LIKE 'innodb_log_file_size';    -- size per log file
SHOW VARIABLES LIKE 'innodb_log_files_in_group'; -- number of log files
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit';
-- 1 = flush to disk on every commit (safest, default)
-- 2 = flush to OS cache on every commit (faster, risk: OS crash)
-- 0 = flush every second (fastest, risk: 1 second of data loss)
```

### Checkpoint

A **checkpoint** is when dirty pages in the buffer pool are flushed to the data files on disk. After a checkpoint, the WAL entries before that point are no longer needed for crash recovery.

![Checkpoint flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/04-checkpoint-flow.png)


```
Timeline:
  WAL: [entry1] [entry2] [entry3] [CHECKPOINT] [entry4] [entry5]
                                       ↑
                              data files are consistent up to here
                              on crash, only replay entry4 and entry5
```

Without checkpoints, crash recovery would need to replay the entire WAL from the beginning of time. Checkpoints bound recovery time.

## InnoDB Architecture: The Big Picture

```
Client Connection
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                     InnoDB Engine                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Buffer Pool (in memory)                │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │     │
│  │  │Data Page │ │Data Page │ │Index Page│  ...        │     │
│  │  │(clean)   │ │(dirty)   │ │(clean)   │            │     │
│  │  └──────────┘ └──────────┘ └──────────┘            │     │
│  └─────────────────────────┬───────────────────────────┘     │
│                            │                                  │
│  ┌────────────┐    ┌───────▼──────┐    ┌─────────────┐      │
│  │ Change     │    │ Redo Log     │    │ Undo Log    │      │
│  │ Buffer     │    │ (WAL)        │    │ (for MVCC   │      │
│  │ (secondary │    │              │    │  & rollback)│      │
│  │  index     │    │ Sequential   │    │             │      │
│  │  changes)  │    │ writes only  │    │             │      │
│  └────────────┘    └───────┬──────┘    └─────────────┘      │
│                            │                                  │
│  ┌─────────────────────────▼──────────────────────────┐      │
│  │              Data Files (.ibd)                      │      │
│  │  Tablespace files on disk                           │      │
│  │  (clustered index + secondary indexes)              │      │
│  └─────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

The flow for an UPDATE:
1. Find the page in the buffer pool (or load from disk)
2. Write the change to the redo log (WAL)
3. Modify the page in the buffer pool (mark as dirty)
4. Write old row version to the undo log (for MVCC and rollback)
5. On COMMIT: flush redo log to disk
6. Eventually: checkpoint flushes dirty pages to data files

## Compression

### Page Compression

InnoDB supports transparent page compression:

```sql
-- Create table with compression (MySQL 5.7+)
CREATE TABLE logs (
    log_id BIGINT PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP
) ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;

-- Check compression ratio
SELECT
    table_name,
    ROUND(data_length / 1024 / 1024, 2) AS data_mb,
    ROUND(data_free / 1024 / 1024, 2) AS free_mb
FROM information_schema.tables
WHERE table_schema = 'mydb' AND table_name = 'logs';
```

PostgreSQL uses TOAST (The Oversized-Attribute Storage Technique) for large values:

```sql
-- Values larger than ~2 KB are automatically compressed and/or stored out-of-line
-- No explicit configuration needed for basic compression

-- Check TOAST table usage
SELECT
    relname,
    pg_size_pretty(pg_total_relation_size(oid)) AS total_size,
    pg_size_pretty(pg_relation_size(oid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(oid) - pg_relation_size(oid)) AS toast_size
FROM pg_class
WHERE relname = 'logs';
```

### Transparent Data Encryption (TDE)

Encryption at the storage engine level — data is encrypted on disk but decrypted transparently when loaded into the buffer pool:

```sql
-- MySQL 8.0: enable encryption per tablespace
ALTER TABLE users ENCRYPTION = 'Y';

-- PostgreSQL: use pgcrypto for column-level encryption
-- or full-disk encryption at the OS level (dm-crypt, LUKS)
```

## Column-Oriented Storage

Everything we have discussed so far is **row-oriented** storage: each page contains complete rows. This is optimal for OLTP (transactional) workloads where you typically access all columns of a few rows.

For **analytical** (OLAP) workloads, column-oriented storage is dramatically better:

```
Row-oriented (good for: SELECT * FROM users WHERE id = 42):
Page 1: [id=1, name="Alice", email="a@...", age=30]
         [id=2, name="Bob",   email="b@...", age=25]

Column-oriented (good for: SELECT AVG(age) FROM users):
Column "id":    [1, 2, 3, 4, 5, ...]     -- stored together
Column "name":  ["Alice", "Bob", ...]     -- stored together
Column "email": ["a@..", "b@..", ...]     -- stored together
Column "age":   [30, 25, 35, 28, ...]    -- stored together
```

Why column-oriented is better for analytics:

| Benefit | Explanation |
|---------|-------------|
| Less I/O | `SELECT AVG(age)` reads only the age column, not name/email/etc. |
| Better compression | Similar values in the same column compress 5-10x better |
| Vectorized processing | CPU can process batches of same-type values using SIMD |
| Cache efficiency | All values being processed fit in CPU cache lines |

Column-oriented databases include: ClickHouse, Apache Parquet (file format), DuckDB, Amazon Redshift, Google BigQuery.

```sql
-- ClickHouse example: analytics query
SELECT
    toStartOfMonth(event_time) AS month,
    count() AS events,
    uniqExact(user_id) AS unique_users,
    avg(duration_ms) AS avg_duration
FROM events
WHERE event_time >= '2023-01-01'
GROUP BY month
ORDER BY month;
-- Scans billions of rows in seconds because it only reads
-- the 4 columns needed, not all 50 columns in the table.
```

Most OLTP databases do not use column-oriented storage (InnoDB, PostgreSQL are row-oriented). But hybrid approaches are emerging:

- PostgreSQL's **cstore_fdw** (columnar foreign data wrapper)
- MySQL HeatWave (in-memory columnar accelerator)
- Oracle's in-memory column store
- SQL Server's columnstore indexes

## Measuring Storage Engine Performance

When evaluating storage engines, three amplification metrics matter most. Let us make them concrete with numbers.

### Write Amplification in Practice

Write amplification is the ratio of total bytes written to storage versus the logical bytes written by the application.

```
Example: Insert a 1 KB row

B-tree engine (InnoDB):
  1. Write to redo log:           1 KB  (WAL entry)
  2. Modify page in buffer pool:  0 KB  (in memory, no I/O yet)
  3. Eventually flush 16 KB page: 16 KB (entire page rewritten for 1 KB change)
  4. Update secondary index:      16 KB (another page write)
  Total: ~33 KB written for 1 KB of data → write amplification = 33x

LSM-tree engine (RocksDB):
  1. Write to WAL:                1 KB
  2. Flush memtable to L0:        1 KB  (written once)
  3. Compact L0 → L1:             1 KB  (rewritten)
  4. Compact L1 → L2:             1 KB  (rewritten)
  5. Compact L2 → L3:             1 KB  (rewritten)
  Total: ~5 KB sequential writes → write amplification = 5x
  But with 10 levels: could be 10-30x
```

### Benchmarking With Real Tools

```bash
# sysbench: standard database benchmark
# Install
apt-get install sysbench

# Prepare test data (1 million rows)
sysbench oltp_read_write \
  --db-driver=mysql \
  --mysql-host=localhost \
  --mysql-user=root \
  --mysql-password=secret \
  --mysql-db=sbtest \
  --tables=4 \
  --table-size=1000000 \
  prepare

# Run OLTP read-write benchmark (60 seconds, 16 threads)
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

Sample output:

```
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
# fio: test raw disk I/O (important for understanding engine behavior)
# Random read test (simulates B-tree index lookups)
fio --name=randread --ioengine=libaio --iodepth=32 \
  --rw=randread --bs=8k --size=1G --numjobs=4 \
  --runtime=30 --group_reporting

# Sequential write test (simulates WAL/LSM writes)
fio --name=seqwrite --ioengine=libaio --iodepth=32 \
  --rw=write --bs=64k --size=1G --numjobs=1 \
  --runtime=30 --group_reporting
```

Understanding the ratio between random read IOPS and sequential write throughput on your hardware directly explains why B-tree engines and LSM-tree engines perform differently on the same machine.

## Choosing a Storage Engine

| Workload | Recommended Engine | Why |
|----------|-------------------|-----|
| General OLTP | InnoDB / PostgreSQL | Proven, ACID, MVCC |
| Write-heavy (logs, metrics) | LSM-based (RocksDB, TiKV) | Sequential writes, high throughput |
| Analytics / OLAP | Column-oriented (ClickHouse, DuckDB) | Scan efficiency, compression |
| Embedded / edge | SQLite / DuckDB | Zero configuration, single-file |
| Key-value workloads | RocksDB / LevelDB | Optimized for simple get/put |
| Time-series | TimescaleDB / InfluxDB | Time-partitioned, retention policies |

## What's Next

We have now seen how individual storage engines organize data on a single machine. But not all data fits in tables, and not all workloads are best served by SQL. In the next article, we will explore the **NoSQL landscape** — document stores, key-value stores, wide-column databases, and graph databases — and when each one makes sense.
