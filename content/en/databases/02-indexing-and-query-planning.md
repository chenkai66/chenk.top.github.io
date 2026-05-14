---
title: "Databases (2): Indexing and Query Planning — How Databases Find Your Data"
date: 2024-04-19 09:00:00
tags:
  - Databases
  - SQL
  - Indexing
  - Performance
categories: Databases
series: databases
lang: en
description: "Deep dive into B-tree and B+tree indexes, hash indexes, composite indexes, covering indexes, and how to read EXPLAIN output to diagnose slow queries."
disableNunjucks: true
series_order: 2
translationKey: "databases-2"
---

A query that returns in 2 milliseconds on your laptop with 1,000 rows will take 45 seconds on a production database with 50 million rows — unless you have the right indexes. Indexes are the single most impactful performance tool in your database toolkit, and understanding how they work changes the way you think about every schema and every query you write.

---

## The Fundamental Problem: Finding a Row

Imagine a table with 10 million rows, stored on disk as a heap file. Each row sits somewhere in a sequence of 8 KB pages. When you run:

![Index selectivity impact](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/02-index-selectivity.png)


```sql
SELECT * FROM users WHERE email = 'alice@example.com';
```

Without an index, the database must perform a **sequential scan** (also called a full table scan): read every single page, examine every single row, check if `email` matches. If the table occupies 2 GB on disk, the database reads 2 GB. For one row.

An index is a separate data structure that maps column values to row locations. With a B-tree index on `email`, the same lookup touches perhaps 3-4 pages instead of 250,000. That is the difference between milliseconds and minutes.

## Sequential Scan vs Index Scan

| Aspect | Sequential Scan | Index Scan |
|--------|----------------|------------|
| How it works | Reads every page in table order | Traverses index tree, then fetches matching rows |
| Best for | Small tables, queries returning >10-15% of rows | Selective queries returning few rows |
| I/O pattern | Sequential (fast on HDDs) | Random (each row may be on a different page) |
| CPU cost | Low per-row (just filter) | Higher per-row (tree traversal + heap fetch) |
| When chosen | No suitable index, or optimizer estimates scan is cheaper | Suitable index exists and query is selective |

![Index scan vs sequential scan](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/02-scan-comparison.png)


The database's query optimizer makes this decision automatically. Sometimes a sequential scan *is* faster — for example, when your `WHERE` clause matches 80% of the table, random I/O from an index would be slower than just reading everything sequentially.

## B-Tree Index: The Workhorse

The B-tree (balanced tree) is the default index type in virtually every relational database. Here is how it works.

![B-tree index structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/02-btree-index.png)


### Structure

A B-tree is a self-balancing tree where:

- Each node contains multiple keys in sorted order
- Each internal node has pointers to child nodes between and around keys
- All leaf nodes are at the same depth (balanced)
- The **branching factor** (number of children per node) is typically hundreds or thousands

For a table with 10 million rows and a branching factor of 500:
- Level 0 (root): 1 node
- Level 1: up to 500 nodes
- Level 2: up to 250,000 nodes
- Level 3 (leaves): up to 125 million entries

Three levels of tree traversal — three page reads — to find any row among 10 million. That is the O(log N) guarantee, but with a very large base logarithm.

### How a Lookup Works

To find `email = 'alice@example.com'`:

1. Start at the root node. Binary search through keys to find which child pointer to follow.
2. Load the child node. Binary search again.
3. Repeat until you reach a leaf node.
4. The leaf contains a pointer to the actual row on disk (a tuple ID or row ID).
5. Fetch the row from the heap (the main table data).

```text
Root Node: [charlie@... | mike@... | zara@...]
                |              |           |
          Child < charlie  charlie-mike  mike-zara   > zara
                |
    [alice@... | bob@...]  <-- leaf node
         |
    Pointer to heap page 4721, offset 23
```

### Creating B-Tree Indexes

```sql
-- Single-column index
CREATE INDEX idx_users_email ON users (email);

-- Unique index (also enforces uniqueness constraint)
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- Check existing indexes (PostgreSQL)
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'users';

-- Check existing indexes (MySQL)
SHOW INDEX FROM users;
```

## B+Tree: Why Databases Prefer It


![Btree index structure as a futuristic city skyline branching](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/02-btree-index-structure-as-a-futuristic-city-skyline-branching.jpg)

Most database implementations actually use a **B+tree**, a variation of the B-tree:

| Feature | B-tree | B+tree |
|---------|--------|--------|
| Data pointers | In both internal and leaf nodes | **Only in leaf nodes** |
| Leaf nodes linked | No | **Yes, via sibling pointers** |
| Internal node size | Larger (stores data pointers) | Smaller (keys + child pointers only) |
| Branching factor | Lower | **Higher** (more keys fit per node) |
| Range queries | Requires traversing back up the tree | **Follow leaf pointers** |
| Point lookups | Can terminate early at internal nodes | Always goes to leaf level |

The key advantage: because internal nodes only store keys (not data pointers), more keys fit per page, increasing the branching factor. A higher branching factor means a shallower tree, which means fewer disk reads.

The linked leaf nodes are critical for range queries:

```sql
-- Range query: find orders from the last 7 days
SELECT * FROM orders
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at;
```

With a B+tree index on `created_at`, the database traverses to the first matching leaf, then follows sibling pointers to read all matching entries sequentially — no need to revisit internal nodes.

## Hash Indexes

Hash indexes use a hash function to map keys directly to row locations.

![Hash index structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/02-hash-index.png)


```sql
-- PostgreSQL: explicitly create a hash index
CREATE INDEX idx_users_email_hash ON users USING hash (email);
```

| Aspect | Hash Index | B-Tree Index |
|--------|-----------|--------------|
| Equality lookups (`=`) | O(1) average | O(log N) |
| Range queries (`>`, `<`, `BETWEEN`) | Not supported | Supported |
| Sorting / `ORDER BY` | Not supported | Supported |
| Prefix matching (`LIKE 'abc%'`) | Not supported | Supported |
| WAL-logged (crash-safe) | PostgreSQL 10+ | Always |
| Common usage | Rarely used in practice | Default, almost always preferred |

Hash indexes win on pure equality lookups but lose everywhere else. In practice, B-tree indexes are fast enough for equality lookups that the limited functionality of hash indexes is rarely worth it. PostgreSQL did not even make hash indexes crash-safe until version 10.

## Composite Indexes: Column Order Matters

A composite (multi-column) index indexes multiple columns together:

```sql
CREATE INDEX idx_orders_user_status ON orders (user_id, status);
```

This creates a B+tree sorted first by `user_id`, then by `status` within each `user_id`. Think of it like a phone book sorted by last name, then first name.

### The Leftmost Prefix Rule

A composite index on `(a, b, c)` can satisfy queries that filter on:

| Query filters on | Uses index? | Why |
|-----------------|-------------|-----|
| `a` | Yes | Leftmost prefix |
| `a, b` | Yes | Leftmost prefix |
| `a, b, c` | Yes | Full index |
| `b` | **No** | Skips leftmost column |
| `b, c` | **No** | Skips leftmost column |
| `a, c` | **Partially** | Uses `a`, then scans for `c` |

```sql
-- This query uses the composite index efficiently
SELECT * FROM orders
WHERE user_id = 42 AND status = 'completed';

-- This query can use the index (leftmost prefix)
SELECT * FROM orders
WHERE user_id = 42;

-- This query CANNOT use the composite index
-- It needs a separate index on (status)
SELECT * FROM orders
WHERE status = 'pending';
```

**Column order strategy**: put the most selective column first (the one that filters out the most rows), followed by columns commonly used together.

## Covering Indexes and Index-Only Scans

Normally, an index scan involves two steps:
1. Traverse the index to find matching entries
2. Fetch the actual rows from the heap (table data) to get the remaining columns

Step 2 is called a "heap fetch" and involves random I/O. A **covering index** includes all columns needed by the query, eliminating the heap fetch entirely:

```sql
-- Create a covering index
CREATE INDEX idx_orders_covering ON orders (user_id, status)
INCLUDE (created_at, order_id);

-- This query can be satisfied entirely from the index
-- No heap fetch needed = "index-only scan"
SELECT order_id, status, created_at
FROM orders
WHERE user_id = 42 AND status = 'completed';
```

In PostgreSQL, you use the `INCLUDE` clause for non-searchable but covered columns. In MySQL (InnoDB), covering indexes work because InnoDB's secondary indexes can cover queries if all needed columns are in the index.

```sql
-- MySQL covering index
CREATE INDEX idx_orders_covering ON orders (user_id, status, created_at, order_id);

-- Check if index-only scan is used (MySQL)
EXPLAIN SELECT order_id, status, created_at
FROM orders WHERE user_id = 42 AND status = 'completed';
-- Look for "Using index" in Extra column
```

## EXPLAIN: Reading the Query Plan

`EXPLAIN` shows you the execution plan the optimizer chose. `EXPLAIN ANALYZE` actually runs the query and shows real timing.

![Query cost model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/02-query-cost-model.png)


### PostgreSQL EXPLAIN ANALYZE

```sql
EXPLAIN ANALYZE
SELECT u.full_name, COUNT(*) AS order_count
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE o.status = 'completed'
GROUP BY u.full_name
ORDER BY order_count DESC
LIMIT 10;
```

Output:

```text
 Limit  (cost=1845.23..1845.26 rows=10 width=40) (actual time=12.456..12.461 rows=10 loops=1)
   ->  Sort  (cost=1845.23..1857.45 rows=4889 width=40) (actual time=12.454..12.458 rows=10 loops=1)
         Sort Key: (count(*)) DESC
         Sort Method: top-N heapsort  Memory: 25kB
         ->  HashAggregate  (cost=1723.56..1772.45 rows=4889 width=40) (actual time=11.234..11.890 rows=4889 loops=1)
               Group Key: u.full_name
               Batches: 1  Memory Usage: 913kB
               ->  Hash Join  (cost=234.67..1601.23 rows=24450 width=32) (actual time=2.345..8.901 rows=24450 loops=1)
                     Hash Cond: (o.user_id = u.user_id)
                     ->  Seq Scan on orders o  (cost=0.00..1156.00 rows=24450 width=4) (actual time=0.012..4.567 rows=24450 loops=1)
                           Filter: ((status)::text = 'completed'::text)
                           Rows Removed by Filter: 25550
                     ->  Hash  (cost=159.00..159.00 rows=10000 width=36) (actual time=2.123..2.123 rows=10000 loops=1)
                           Buckets: 16384  Batches: 1  Memory Usage: 641kB
                           ->  Seq Scan on users u  (cost=0.00..159.00 rows=10000 width=36) (actual time=0.008..0.987 rows=10000 loops=1)
 Planning Time: 0.234 ms
 Execution Time: 12.567 ms
```

Key things to look for:

| Field | What it tells you |
|-------|-------------------|
| `Seq Scan` | Full table scan — possibly needs an index |
| `Index Scan` | Using an index — good for selective queries |
| `Index Only Scan` | Covering index — best case |
| `Bitmap Index Scan` | Combines multiple index results |
| `Hash Join` / `Nested Loop` / `Merge Join` | Join strategy |
| `actual time` | Real execution time (first row..last row) in ms |
| `rows` | Actual number of rows processed |
| `Rows Removed by Filter` | Rows read but discarded — high numbers suggest missing index |
| `loops` | How many times this step ran |

### MySQL EXPLAIN

```sql
EXPLAIN
SELECT u.full_name, COUNT(*) AS order_count
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE o.status = 'completed'
GROUP BY u.full_name
ORDER BY order_count DESC
LIMIT 10;
```

```text
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
| id | select_type | table | type | possible_keys    | key     | key_len | ref              | rows  | Extra                                        |
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
|  1 | SIMPLE      | o     | ref  | idx_order_status | idx_... | 82      | const            | 24450 | Using where; Using temporary; Using filesort |
|  1 | SIMPLE      | u     | ref  | PRIMARY          | PRIMARY | 4       | mydb.o.user_id   |     1 | NULL                                         |
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
```

The `type` column is the most important in MySQL EXPLAIN:

| type | Meaning | Performance |
|------|---------|-------------|
| `system` / `const` | At most one matching row | Best |
| `eq_ref` | One row per join (primary key / unique) | Excellent |
| `ref` | Multiple rows via non-unique index | Good |
| `range` | Index range scan | Good |
| `index` | Full index scan (reads all index entries) | Moderate |
| `ALL` | Full table scan | Worst — usually needs an index |

### Spotting Problems

Here is a bad query plan (PostgreSQL):

```sql
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE status = 'pending'
  AND created_at > '2023-01-01';
```

```text
 Seq Scan on orders  (cost=0.00..2456.00 rows=245 width=48) (actual time=0.034..18.567 rows=234 loops=1)
   Filter: (((status)::text = 'pending'::text) AND (created_at > '2023-01-01'))
   Rows Removed by Filter: 49766
 Planning Time: 0.089 ms
 Execution Time: 18.623 ms
```

Red flags:
- `Seq Scan` on a table with 50,000 rows
- `Rows Removed by Filter: 49766` — read 50,000 rows to find 234

Fix:

```sql
CREATE INDEX idx_orders_status_created ON orders (status, created_at);
```

After creating the index:

```text
 Index Scan using idx_orders_status_created on orders  (cost=0.29..12.45 rows=245 width=48) (actual time=0.023..0.189 rows=234 loops=1)
   Index Cond: (((status)::text = 'pending'::text) AND (created_at > '2023-01-01'))
 Planning Time: 0.102 ms
 Execution Time: 0.234 ms
```

From 18.6 ms to 0.23 ms — an 80x improvement — from one index.

## Index Selection Strategies

### What to Index

1. **Primary keys**: automatically indexed in all databases
2. **Foreign keys**: always index these — JOINs use them constantly
3. **Columns in WHERE clauses**: especially in high-frequency queries
4. **Columns in ORDER BY**: avoids expensive sorts
5. **Columns in GROUP BY**: helps with aggregation
6. **Columns used in JOINs**: besides foreign keys, any join condition

### Cardinality Matters

**Cardinality** = the number of distinct values in a column.

| Column | Cardinality | Good index candidate? |
|--------|-------------|----------------------|
| `email` (unique) | 10,000,000 | Yes — highly selective |
| `country` | 195 | Maybe — depends on query patterns |
| `status` (active/inactive) | 2 | **Rarely** — not selective enough |
| `is_deleted` (true/false) | 2 | **No** — use partial index instead |

Low-cardinality columns return too many rows per value. The optimizer will often choose a sequential scan over an index scan for low-cardinality lookups.

Exception: if a low-cardinality value is rare (e.g., `status = 'fraud'` matches 0.01% of rows), it *is* selective and an index helps. A partial index is even better.

## Over-Indexing: The Hidden Cost


![Magnifying glass over database index revealing optimized que](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/02-magnifying-glass-over-database-index-revealing-optimized-que.jpg)

Every index you create has costs:

| Cost | Impact |
|------|--------|
| **Write amplification** | Every INSERT/UPDATE/DELETE must update all affected indexes |
| **Storage** | Each index can be 10-30% the size of the table |
| **Memory pressure** | Indexes compete for buffer pool space |
| **Planning time** | More indexes = more options for the optimizer to evaluate |
| **Maintenance** | VACUUM, REINDEX, statistics updates |

A table with 10 indexes means every INSERT writes to 11 data structures (table + 10 indexes). For write-heavy workloads, this is devastating.

**Rule of thumb**: most OLTP tables should have 3-5 indexes. If you have more than 8, audit them.

Check for unused indexes in PostgreSQL:

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Partial Indexes

A partial index only indexes rows that match a condition:

```sql
-- Only index non-deleted users (if 99% of users are not deleted)
CREATE INDEX idx_users_active ON users (email)
WHERE is_deleted = FALSE;

-- Only index pending orders (if most orders are completed)
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
```

Benefits:
- Much smaller than a full index
- Faster to maintain (fewer entries to update)
- Higher hit rate in the buffer pool

The query must include the partial index's `WHERE` condition for the optimizer to use it:

```sql
-- This uses idx_orders_pending
SELECT * FROM orders
WHERE status = 'pending' AND created_at > '2023-12-01';

-- This does NOT use idx_orders_pending
SELECT * FROM orders
WHERE status = 'completed' AND created_at > '2023-12-01';
```

## Expression Indexes

You can index the result of an expression or function:

```sql
-- Index on lowercase email for case-insensitive lookups
CREATE INDEX idx_users_email_lower ON users (LOWER(email));

-- Query must use the same expression
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- Index on extracted year from timestamp
CREATE INDEX idx_orders_year ON orders (EXTRACT(YEAR FROM created_at));

-- Index on JSONB field (PostgreSQL)
CREATE INDEX idx_users_metadata_country
ON users ((metadata->>'country'));
```

Without an expression index, using a function in `WHERE` prevents the optimizer from using a regular index on that column:

```sql
-- This CANNOT use a regular index on email
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';
-- The database sees LOWER(email), not email — different thing

-- This CAN use a regular index on email
SELECT * FROM users WHERE email = 'alice@example.com';
```

## GIN and GiST Indexes (PostgreSQL)

Beyond B-tree, PostgreSQL offers specialized index types:

```sql
-- GIN index for full-text search
CREATE INDEX idx_products_search ON products
USING gin (to_tsvector('english', name || ' ' || description));

-- Query using full-text search
SELECT name, ts_rank(to_tsvector('english', name || ' ' || description),
                     plainto_tsquery('english', 'wireless keyboard')) AS rank
FROM products
WHERE to_tsvector('english', name || ' ' || description)
      @@ plainto_tsquery('english', 'wireless keyboard')
ORDER BY rank DESC;

-- GIN index for JSONB containment queries
CREATE INDEX idx_users_metadata ON users USING gin (metadata jsonb_path_ops);

SELECT * FROM users WHERE metadata @> '{"country": "US"}';

-- GiST index for geometric/range data
CREATE INDEX idx_events_timerange ON events USING gist (time_range);
```

| Index Type | Best For | Supported Operations |
|-----------|---------|---------------------|
| B-tree | Equality, range, sorting | `=`, `<`, `>`, `BETWEEN`, `ORDER BY`, `LIKE 'prefix%'` |
| Hash | Equality only | `=` |
| GIN | Arrays, JSONB, full-text | `@>`, `&&`, `@@`, `?`, `?&` |
| GiST | Geometric, range, nearest-neighbor | `<<`, `>>`, `&&`, `@>`, `<->` |
| BRIN | Large, naturally ordered tables | `<`, `>`, `=` (with reduced precision) |

## Practical Index Design Workflow

When you have a slow query, follow this process:

```text
1. Run EXPLAIN ANALYZE on the slow query
2. Look for Seq Scans with high "Rows Removed by Filter"
3. Identify which WHERE/JOIN/ORDER BY columns lack indexes
4. Check cardinality of those columns
5. Create the most selective composite index
6. Re-run EXPLAIN ANALYZE to verify improvement
7. Monitor pg_stat_user_indexes for actual usage
8. Drop unused indexes after 30 days
```

A complete example:

```sql
-- Step 1: The slow query
EXPLAIN ANALYZE
SELECT p.name, SUM(oi.quantity) AS total_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.created_at >= '2023-11-01'
  AND o.status = 'completed'
GROUP BY p.name
ORDER BY total_sold DESC
LIMIT 10;
-- Execution Time: 234.567 ms

-- Step 2-4: Seq Scan on orders, filtering 80% of rows
-- Need index on (status, created_at)

-- Step 5: Create index
CREATE INDEX idx_orders_status_date ON orders (status, created_at);

-- Step 6: Verify
EXPLAIN ANALYZE
-- ... same query ...
-- Execution Time: 3.456 ms  (68x faster)
```

## What's next

Indexes tell the database *where* to find data. But what happens when two transactions try to modify the same data at the same time? In the next article, we will explore **transactions and concurrency** — ACID guarantees, isolation levels, locking, and the dark art of preventing deadlocks.
