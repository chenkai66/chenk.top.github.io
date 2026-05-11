---
title: "数据库（二）：索引与查询规划——数据库如何找到你的数据"
date: 2024-04-19 09:00:00
tags:
  - Databases
  - SQL
  - Indexing
  - Performance
categories:
  - Databases
series: databases
lang: zh
description: "深入剖析 B 树与 B+ 树索引、哈希索引、复合索引、覆盖索引，以及如何阅读 EXPLAIN 输出以诊断慢查询。"
disableNunjucks: true
series_order: 2
translationKey: "databases-2"
---

在你的笔记本电脑上仅含 1,000 行数据时耗时 2 毫秒的查询，在生产环境中面对 5,000 万行数据时可能需要 45 秒——除非你拥有正确的索引。索引是你数据库工具箱中**性能影响最大**的单一手段；理解其工作原理，将彻底改变你对每一个数据库 Schema 和每一条 SQL 查询的思考方式。

## 根本问题：如何定位一行数据

想象一张包含 1,000 万行的表，以堆文件（heap file）形式存储在磁盘上。每一行都位于一系列 8 KB 数据页中的某个位置。当你执行：

```sql
SELECT * FROM users WHERE email = 'alice@example.com';
```

若无索引，数据库必须执行一次 **顺序扫描（sequential scan）**（也称全表扫描）：逐页读取整张表，逐行检查 `email` 字段是否匹配。若该表在磁盘上占 2 GB，则数据库需读取整整 2 GB —— 仅仅为了查找一行数据。

而索引是一种独立的数据结构，它将列值映射到对应行的物理位置。若在 `email` 列上建立了 B 树索引，同样的查找操作只需访问约 3–4 个数据页，而非 250,000 个。这正是毫秒级响应与分钟级延迟之间的本质区别。

## 顺序扫描 vs 索引扫描

| 维度 | 顺序扫描 | 索引扫描 |
|------|----------|----------|
| 工作原理 | 按表的物理顺序逐页读取 | 遍历索引树，再获取匹配的行 |
| 最适用场景 | 小型表；或查询返回 >10–15% 的行 | 高选择性查询（返回少量行） |
| I/O 模式 | 顺序读取（HDD 上较快） | 随机读取（每行可能位于不同页） |
| CPU 开销 | 每行开销低（仅过滤） | 每行开销较高（树遍历 + 堆获取） |
| 触发条件 | 无合适索引；或优化器估算扫描更便宜 | 存在合适索引且查询具备高选择性 |

数据库的查询优化器会自动做出这一决策。有时顺序扫描确实更快——例如当 `WHERE` 条件匹配了表中 80% 的行时，索引带来的随机 I/O 反而比一次性顺序读取全部数据更慢。

## B 树索引：主力索引类型

B 树（平衡树）几乎是所有关系型数据库的默认索引类型。其工作原理如下。

### 结构

B 树是一种自平衡树，具有以下特征：

- 每个节点包含多个按升序排列的键（key）
- 每个内部节点包含指向子节点的指针，这些指针分布在键之间及两端
- 所有叶子节点处于同一深度（即“平衡”）
- **分支因子（branching factor）**（每个节点的子节点数）通常为数百至数千

假设一张表有 1,000 万行，分支因子为 500：
- 第 0 层（根节点）：1 个节点  
- 第 1 层：最多 500 个节点  
- 第 2 层：最多 250,000 个节点  
- 第 3 层（叶子层）：最多 1.25 亿个条目  

仅需三次树遍历（即三次页读取），即可在 1,000 万行中定位任意一行。这就是 O(log N) 时间复杂度的保证——只不过其对数底数非常大。

### 查找过程

要查找 `email = 'alice@example.com'`：

1. 从根节点开始，通过二分查找确定应跟随哪个子节点指针；
2. 加载该子节点，再次进行二分查找；
3. 重复此过程，直至抵达叶子节点；
4. 叶子节点中包含指向磁盘上实际行（元组 ID 或行 ID）的指针；
5. 从堆（主表数据区）中获取该行。

```
Root Node: [charlie@... | mike@... | zara@...]
                |              |           |
          Child < charlie  charlie-mike  mike-zara   > zara
                |
    [alice@... | bob@...]  <-- leaf node
         |
    Pointer to heap page 4721, offset 23
```

### 创建 B 树索引

```sql
-- 单列索引
CREATE INDEX idx_users_email ON users (email);

-- 唯一索引（同时强制唯一性约束）
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- 查看现有索引（PostgreSQL）
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'users';

-- 查看现有索引（MySQL）
SHOW INDEX FROM users;
```

## B+ 树：为何数据库更偏爱它？

大多数数据库实现实际上使用的是 **B+ 树**——B 树的一种变体：

| 特性 | B 树 | B+ 树 |
|------|------|--------|
| 数据指针位置 | 内部节点和叶子节点均包含 | **仅叶子节点包含** |
| 叶子节点是否链式连接 | 否 | **是，通过兄弟指针（sibling pointers）** |
| 内部节点大小 | 更大（存储数据指针） | 更小（仅存储键 + 子节点指针） |
| 分支因子 | 较低 | **更高**（每页可容纳更多键） |
| 范围查询效率 | 需回溯父节点 | **直接沿叶子链顺序遍历** |
| 点查（point lookup） | 可在内部节点提前终止 | 总是需到达叶子层 |

关键优势在于：由于内部节点不存储数据指针，仅存键和子节点指针，因此每页可容纳更多键，从而提升分支因子。更高的分支因子意味着更浅的树结构，进而减少磁盘读取次数。

叶子节点间的链式连接对范围查询至关重要：

```sql
-- 范围查询：查找最近 7 天的订单
SELECT * FROM orders
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at;
```

若 `created_at` 上建有 B+ 树索引，数据库只需定位第一个匹配的叶子节点，然后沿兄弟指针顺序读取所有匹配项——无需反复回溯内部节点。

## 哈希索引

哈希索引利用哈希函数将键直接映射到行的物理位置：

```sql
-- PostgreSQL：显式创建哈希索引
CREATE INDEX idx_users_email_hash ON users USING hash (email);
```

| 维度 | 哈希索引 | B 树索引 |
|------|-----------|--------------|
| 等值查找（`=`） | 平均 O(1) | O(log N) |
| 范围查询（`>`, `<`, `BETWEEN`） | 不支持 | 支持 |
| 排序 / `ORDER BY` | 不支持 | 支持 |
| 前缀匹配（`LIKE 'abc%'`） | 不支持 | 支持 |
| WAL 日志记录（崩溃安全） | PostgreSQL 10+ 支持 | 始终支持 |
| 实际使用频率 | 实践中极少使用 | 默认索引类型，几乎总是首选 |

哈希索引在纯等值查找场景下胜出，但在其他所有场景下均失效。实践中，B 树索引的等值查找已足够快，哈希索引功能受限的代价通常得不偿失。PostgreSQL 直到版本 10 才使哈希索引具备崩溃安全性。

## 复合索引：列顺序至关重要

复合索引（多列索引）将多个列作为一个整体进行索引：

```sql
CREATE INDEX idx_orders_user_status ON orders (user_id, status);
```

该索引构建的 B+ 树首先按 `user_id` 排序，再在每个 `user_id` 下按 `status` 排序。类比电话簿：先按姓氏排序，再在同姓下按名字排序。

### 最左前缀规则（Leftmost Prefix Rule）

一个 `(a, b, c)` 复合索引可满足以下查询的过滤条件：

| 查询过滤条件 | 是否使用索引？ | 原因 |
|-----------------|-------------|-----|
| `a` | 是 | 最左前缀 |
| `a, b` | 是 | 最左前缀 |
| `a, b, c` | 是 | 完整索引 |
| `b` | **否** | 跳过了最左列 |
| `b, c` | **否** | 跳过了最左列 |
| `a, c` | **部分使用** | 使用 `a`，但对 `c` 需额外扫描 |

```sql
-- 此查询高效利用复合索引
SELECT * FROM orders
WHERE user_id = 42 AND status = 'completed';

-- 此查询可利用索引（最左前缀）
SELECT * FROM orders
WHERE user_id = 42;

-- 此查询**无法**利用该复合索引
-- 需单独为 (status) 创建索引
SELECT * FROM orders
WHERE status = 'pending';
```

**列顺序策略**：将选择性最高的列（即能过滤掉最多行的列）放在最前面，随后是常被一起使用的列。

## 覆盖索引与仅索引扫描（Index-Only Scan）

通常，索引扫描包含两个步骤：
1. 遍历索引以定位匹配项；
2. 从堆（主表数据）中获取剩余所需列（即“堆获取”，heap fetch）。

第 2 步涉及随机 I/O。而 **覆盖索引（covering index）** 包含查询所需的所有列，从而完全避免堆获取：

```sql
-- 创建覆盖索引
CREATE INDEX idx_orders_covering ON orders (user_id, status)
INCLUDE (created_at, order_id);

-- 此查询可完全由索引满足
-- 无需堆获取 → “仅索引扫描”
SELECT order_id, status, created_at
FROM orders
WHERE user_id = 42 AND status = 'completed';
```

在 PostgreSQL 中，使用 `INCLUDE` 子句添加非搜索用但需覆盖的列。而在 MySQL（InnoDB）中，覆盖索引天然有效——因为 InnoDB 的二级索引若已包含查询所需全部列，即可直接服务查询。

```sql
-- MySQL 覆盖索引
CREATE INDEX idx_orders_covering ON orders (user_id, status, created_at, order_id);

-- 检查是否启用仅索引扫描（MySQL）
EXPLAIN SELECT order_id, status, created_at
FROM orders WHERE user_id = 42 AND status = 'completed';
-- 查看 Extra 列是否显示 "Using index"
```

## EXPLAIN：解读查询执行计划

`EXPLAIN` 显示优化器选定的执行计划；`EXPLAIN ANALYZE` 则真正执行查询并报告真实耗时。

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

输出示例：

```
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

关键字段解读：

| 字段 | 含义 |
|------|------|
| `Seq Scan` | 全表扫描 —— 可能需要添加索引 |
| `Index Scan` | 使用索引 —— 对高选择性查询有利 |
| `Index Only Scan` | 覆盖索引 —— 最优情况 |
| `Bitmap Index Scan` | 合并多个索引结果 |
| `Hash Join` / `Nested Loop` / `Merge Join` | 连接策略 |
| `actual time` | 真实执行耗时（首行..末行），单位毫秒 |
| `rows` | 实际处理的行数 |
| `Rows Removed by Filter` | 已读取但被过滤丢弃的行数 —— 数值高表明缺少索引 |
| `loops` | 当前步骤执行次数 |

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

```
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
| id | select_type | table | type | possible_keys    | key     | key_len | ref              | rows  | Extra                                        |
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
|  1 | SIMPLE      | o     | ref  | idx_order_status | idx_... | 82      | const            | 24450 | Using where; Using temporary; Using filesort |
|  1 | SIMPLE      | u     | ref  | PRIMARY          | PRIMARY | 4       | mydb.o.user_id   |     1 | NULL                                         |
+----+-------------+-------+------+------------------+---------+---------+------------------+-------+----------------------------------------------+
```

MySQL EXPLAIN 中最重要的字段是 `type`：

| type | 含义 | 性能 |
|------|------|------|
| `system` / `const` | 至多匹配一行 | 最优 |
| `eq_ref` | 每次连接返回一行（主键/唯一键） | 极佳 |
| `ref` | 通过非唯一索引匹配多行 | 良好 |
| `range` | 索引范围扫描 | 良好 |
| `index` | 全索引扫描（读取所有索引条目） | 中等 |
| `ALL` | 全表扫描 | 最差 —— 通常需加索引 |

### 识别性能问题

以下是一个糟糕的查询计划（PostgreSQL）：

```sql
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE status = 'pending'
  AND created_at > '2023-01-01';
```

```
 Seq Scan on orders  (cost=0.00..2456.00 rows=245 width=48) (actual time=0.034..18.567 rows=234 loops=1)
   Filter: (((status)::text = 'pending'::text) AND (created_at > '2023-01-01'))
   Rows Removed by Filter: 49766
 Planning Time: 0.089 ms
 Execution Time: 18.623 ms
```

警示信号：
- 在 50,000 行的表上执行 `Seq Scan`；
- `Rows Removed by Filter: 49766` —— 读取了 50,000 行才找到 234 行。

修复方案：

```sql
CREATE INDEX idx_orders_status_created ON orders (status, created_at);
```

建索引后：

```
 Index Scan using idx_orders_status_created on orders  (cost=0.29..12.45 rows=245 width=48) (actual time=0.023..0.189 rows=234 loops=1)
   Index Cond: (((status)::text = 'pending'::text) AND (created_at > '2023-01-01'))
 Planning Time: 0.102 ms
 Execution Time: 0.234 ms
```

从 18.6 毫秒降至 0.23 毫秒 —— **80 倍性能提升**，仅靠一个索引。

## 索引选型策略

### 应该建立索引的列

1. **主键（Primary keys）**：所有数据库均自动为其建立索引；
2. **外键（Foreign keys）**：务必索引 —— JOIN 操作频繁依赖；
3. **WHERE 子句中的列**：尤其是高频查询中的过滤条件；
4. **ORDER BY 子句中的列**：避免昂贵的排序操作；
5. **GROUP BY 子句中的列**：加速聚合计算；
6. **JOIN 条件中的列**：除外键外，任何连接表达式中的列。

### 基数（Cardinality）至关重要

**基数** = 列中不同值的数量。

| 列 | 基数 | 是否适合建索引？ |
|--------|-------------|----------------------|
| `email`（唯一） | 10,000,000 | 是 —— 高度选择性 |
| `country` | 195 | 视查询模式而定 |
| `status`（active/inactive） | 2 | **极少** —— 选择性不足 |
| `is_deleted`（true/false） | 2 | **否** —— 应改用部分索引（partial index） |

低基数列的每个值对应大量行，优化器往往倾向于顺序扫描而非索引扫描。

例外：若某低基数值极为稀有（如 `status = 'fraud'` 仅匹配 0.01% 的行），则它具备高选择性，索引依然有效；此时部分索引效果更佳。

## 过度索引：隐藏的成本

每个索引都会带来成本：

| 成本 | 影响 |
|------|--------|
| **写放大（Write amplification）** | 每次 INSERT/UPDATE/DELETE 都需更新所有相关索引 |
| **存储空间** | 每个索引体积可达表的 10–30% |
| **内存压力** | 索引与表竞争缓冲池（buffer pool）空间 |
| **规划耗时** | 索引越多，优化器评估选项越多，规划时间越长 |
| **维护开销** | VACUUM、REINDEX、统计信息更新等 |

一张表若有 10 个索引，则每次 INSERT 实际需写入 11 个数据结构（表 + 10 个索引）。对写密集型负载而言，这是灾难性的。

**经验法则**：多数 OLTP 表应维持 3–5 个索引。若超过 8 个，请全面审计。

在 PostgreSQL 中检查未使用的索引：

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

## 部分索引（Partial Indexes）

部分索引仅对满足特定条件的行建立索引：

```sql
-- 仅索引未删除的用户（若 99% 用户未被删除）
CREATE INDEX idx_users_active ON users (email)
WHERE is_deleted = FALSE;

-- 仅索引待处理订单（若大部分订单已完成）
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
```

优势：
- 体积远小于全量索引；
- 维护更快（需更新的条目更少）；
- 缓冲池命中率更高。

但查询中**必须包含**部分索引的 `WHERE` 条件，优化器才会选用它：

```sql
-- 此查询可使用 idx_orders_pending
SELECT * FROM orders
WHERE status = 'pending' AND created_at > '2023-12-01';

-- 此查询**不会**使用 idx_orders_pending
SELECT * FROM orders
WHERE status = 'completed' AND created_at > '2023-12-01';
```

## 表达式索引（Expression Indexes）

你可以对表达式或函数的结果建立索引：

```sql
-- 为大小写不敏感查找建立小写 email 索引
CREATE INDEX idx_users_email_lower ON users (LOWER(email));

-- 查询必须使用相同表达式
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- 为时间戳提取年份建立索引
CREATE INDEX idx_orders_year ON orders (EXTRACT(YEAR FROM created_at));

-- 为 JSONB 字段建立索引（PostgreSQL）
CREATE INDEX idx_users_metadata_country
ON users ((metadata->>'country'));
```

若未使用表达式索引，`WHERE` 子句中调用函数将导致优化器无法使用该列上的普通索引：

```sql
-- 此查询**无法**使用 email 列上的普通索引
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';
-- 数据库看到的是 LOWER(email)，而非 email —— 二者语义不同

-- 此查询**可以**使用 email 列上的普通索引
SELECT * FROM users WHERE email = 'alice@example.com';
```

## GIN 与 GiST 索引（PostgreSQL）

除 B 树外，PostgreSQL 提供多种专用索引类型：

```sql
-- GIN 索引用于全文检索
CREATE INDEX idx_products_search ON products
USING gin (to_tsvector('english', name || ' ' || description));

-- 全文检索查询
SELECT name, ts_rank(to_tsvector('english', name || ' ' || description),
                     plainto_tsquery('english', 'wireless keyboard')) AS rank
FROM products
WHERE to_tsvector('english', name || ' ' || description)
      @@ plainto_tsquery('english', 'wireless keyboard')
ORDER BY rank DESC;

-- GIN 索引用于 JSONB 包含查询
CREATE INDEX idx_users_metadata ON users USING gin (metadata jsonb_path_ops);

SELECT * FROM users WHERE metadata @> '{"country": "US"}';

-- GiST 索引用于几何/区间数据
CREATE INDEX idx_events_timerange ON events USING gist (time_range);
```

| 索引类型 | 最适用场景 | 支持的操作符 |
|-----------|---------|---------------------|
| B-tree | 等值、范围、排序 | `=`, `<`, `>`, `BETWEEN`, `ORDER BY`, `LIKE 'prefix%'` |
| Hash | 仅等值 | `=` |
| GIN | 数组、JSONB、全文检索 | `@>`, `&&`, `@@`, `?`, `?&` |
| GiST | 几何、区间、最近邻 | `<<`, `>>`, `&&`, `@>`, `<->` |
| BRIN | 大型、天然有序的表 | `<`, `>`, `=`（精度略降） |

## 实用索引设计流程

遇到慢查询时，请遵循以下流程：

```
1. 对慢查询运行 EXPLAIN ANALYZE
2. 寻找带有高 "Rows Removed by Filter" 的 Seq Scan
3. 识别 WHERE/JOIN/ORDER BY 中缺失索引的列
4. 检查这些列的基数
5. 创建最具选择性的复合索引
6. 再次运行 EXPLAIN ANALYZE 验证改进效果
7. 监控 pg_stat_user_indexes 以确认实际使用率
8. 30 天后删除未使用的索引
```

完整示例：

```sql
-- 步骤 1：慢查询
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

-- 步骤 2–4：orders 表出现 Seq Scan，过滤掉 80% 的行
-- 需要在 (status, created_at) 上建索引

-- 步骤 5：创建索引
CREATE INDEX idx_orders_status_date ON orders (status, created_at);

-- 步骤 6：验证
EXPLAIN ANALYZE
-- ... 同一查询 ...
-- Execution Time: 3.456 ms  （68 倍提速）
```

## 下一步

索引告诉数据库**去哪里**查找数据。但当两个事务同时尝试修改同一行数据时，会发生什么？在下一篇文章中，我们将深入探讨 **事务与并发控制** —— ACID 保证、隔离级别、锁机制，以及预防死锁的“黑魔法”。