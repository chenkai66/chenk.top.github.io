---
title: "数据库（五）：NoSQL——文档型、键值型、列式与图数据库"
date: 2024-04-24 09:00:00
tags:
  - Databases
  - NoSQL
  - MongoDB
  - Redis
  - Cassandra
categories: Databases
series: databases
lang: zh
description: "一次面向实践的四大 NoSQL 家族巡礼——文档型、键值型、宽列式与图数据库，涵盖 CAP 定理及各类模型的适用场景。"
disableNunjucks: true
series_order: 5
series_total: 8
translationKey: "databases-5"
---
并非所有数据都能被整齐地塞进行与列中。社交网络中的好友关系图、属性千差万别的商品目录、实时排行榜，以及推荐引擎背后的关系网络——这些工作负载都会让关系型数据库显得力不从心。NoSQL 数据库之所以存在，正是因为不同的数据模型能更高效地解决不同类型的问题。关键在于，你要知道该选用哪一种。

---

## 为何需要 NoSQL？

“NoSQL”这个术语其实颇具误导性。它并不意味着“不用 SQL”——事实上，有些 NoSQL 数据库支持类 SQL 的查询语言。它真正的含义是“不仅仅是 SQL”（Not Only SQL），或者更准确地说，“非关系型”（non-relational）。采用 NoSQL 的动因主要可以归为三类：

![文档模型与关系模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/05-document-model.png)

1. **模式灵活性**：你的数据没有固定 schema，或者 schema 经常变化
2. **横向扩展架构**：你需要水平扩展的能力，而单机关系型数据库已无法满足
3. **数据模型契合度**：你的数据天然就是文档、图、键值对或时间序列的形式，而不是表格

接下来，我们逐一探索这四大家族。

## 文档型数据库：MongoDB

文档数据库以半结构化文档的形式存储数据，通常使用 JSON（MongoDB 中则使用其二进制变体 BSON）。

![列族存储布局](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/05-column-store.png)

每个文档都可以拥有不同的结构，无需预定义固定的 schema。

### 数据模型

```json
// MongoDB 中的一个用户文档
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "email": "alice@example.com",
  "name": "Alice Chen",
  "addresses": [
    {
      "type": "home",
      "street": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "zip": "94105"
    },
    {
      "type": "work",
      "street": "456 Market St",
      "city": "San Francisco",
      "state": "CA",
      "zip": "94103"
    }
  ],
  "preferences": {
    "newsletter": true,
    "theme": "dark",
    "language": "en"
  },
  "created_at": ISODate("2023-11-15T10:30:00Z")
}
```

在关系型数据库中，这类数据至少需要拆分成 `users`、`addresses` 和 `user_preferences` 三张表；而在 MongoDB 中，它就是一个完整的文档——完全不需要 JOIN。

### CRUD 操作

```javascript
// 连接 MongoDB
const db = client.db("ecommerce");
const users = db.collection("users");

// 创建（Create）
await users.insertOne({
  email: "alice@example.com",
  name: "Alice Chen",
  addresses: [{ type: "home", city: "San Francisco" }],
  created_at: new Date()
});

// 读取（Read）
const user = await users.findOne({ email: "alice@example.com" });

// 投影读取（仅返回指定字段）
const userBasic = await users.findOne(
  { email: "alice@example.com" },
  { projection: { name: 1, email: 1, _id: 0 } }
);

// 更新（Update）：添加新地址
await users.updateOne(
  { email: "alice@example.com" },
  { $push: { addresses: { type: "work", city: "Oakland" } } }
);

// 更新（Update）：递增计数器
await users.updateOne(
  { _id: userId },
  { $inc: { login_count: 1 }, $set: { last_login: new Date() } }
);

// 删除（Delete）
await users.deleteOne({ email: "alice@example.com" });
```

### 查询与过滤

```javascript
// 查找位于旧金山且偏好深色主题的用户
const result = await users.find({
  "addresses.city": "San Francisco",
  "preferences.theme": "dark"
}).sort({ created_at: -1 }).limit(10).toArray();

// 查找最近 30 天内创建的用户
const recent = await users.find({
  created_at: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
}).toArray();

// 全文搜索（需先建立 text 索引）
await users.createIndex({ name: "text", email: "text" });
const searchResults = await users.find({
  $text: { $search: "alice chen" }
}).toArray();
```

### 聚合管道

MongoDB 的聚合框架出人意料地强大，能够完成许多原本需要 SQL 中的 `GROUP BY`、`JOIN` 甚至窗口函数才能实现的功能：

```javascript
// 计算上季度各品类商品营收
const pipeline = [
  // 阶段 1：筛选上季度已完成订单
  { $match: {
    status: "completed",
    created_at: { $gte: new Date("2023-10-01") }
  }},
  // 阶段 2：展开 order items 数组（每项生成一个文档）
  { $unwind: "$items" },
  // 阶段 3：关联产品详情
  { $lookup: {
    from: "products",
    localField: "items.product_id",
    foreignField: "_id",
    as: "product"
  }},
  // 阶段 4：展开 product 数组
  { $unwind: "$product" },
  // 阶段 5：按品类分组
  { $group: {
    _id: "$product.category",
    total_revenue: { $sum: { $multiply: ["$items.quantity", "$items.price"] } },
    order_count: { $sum: 1 },
    avg_order_value: { $avg: { $multiply: ["$items.quantity", "$items.price"] } }
  }},
  // 阶段 6：按营收降序排序
  { $sort: { total_revenue: -1 } },
  // 阶段 7：重命名输出字段
  { $project: {
    category: "$_id",
    total_revenue: { $round: ["$total_revenue", 2] },
    order_count: 1,
    avg_order_value: { $round: ["$avg_order_value", 2] },
    _id: 0
  }}
];

const results = await orders.aggregate(pipeline).toArray();
// [
//   { category: "Electronics", total_revenue: 45230.50, order_count: 312, avg_order_value: 145.00 },
//   { category: "Books",       total_revenue: 12890.00, order_count: 567, avg_order_value: 22.73 },
//   ...
// ]
```

### 文档型数据库适用场景

| 场景 | 为何契合文档模型 |
|------|------------------|
| 商品目录 | 不同品类的商品属性差异巨大（比如鞋子和笔记本电脑） |
| 内容管理系统 | 文章、博客及其嵌套评论的结构天然适合文档模型 |
| 用户档案 | 用户偏好和元数据高度可变 |
| 事件日志 | 半结构化的事件数据灵活多变 |
| 移动端后端 | 输入输出均为 JSON，且 schema 演进迅速 |

### 文档型数据库不适用场景

- **多对多关系**：容易导致数据冗余，或需要手动管理引用
- **跨文档的复杂事务**：多文档事务支持有限
- **重度聚合或分析任务**：SQL 数据库和列式存储通常更高效
- **强一致性要求**：在分布式部署下，默认采用最终一致性

## 键值型数据库：Redis

![在空间中浮动的CAP定理三角形：一致性、可用性](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/05-cap-theorem-triangle-floating-in-space-consistency-availabil.jpg)

键值存储是最简单的 NoSQL 模型：你提供一个 key，它就返回对应的 value。Redis 在此基础上更进一步，允许 value 是丰富的数据结构。

![图数据库遍历](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/05-graph-traversal.png)

### 数据结构

```bash
# 字符串（Strings）：最基础的键值对
SET user:1:name "Alice Chen"
GET user:1:name                    # "Alice Chen"

SET page:home:views 0
INCR page:home:views               # 1（原子递增）
INCR page:home:views               # 2
INCRBY page:home:views 100         # 102

# 带过期时间（TTL）的字符串
SET session:abc123 '{"user_id":1,"role":"admin"}' EX 3600  # 1 小时后过期
TTL session:abc123                 # 3597（剩余秒数）

# 哈希（Hashes）：类似微型文档
HSET user:1 name "Alice" email "alice@example.com" login_count 42
HGET user:1 name                   # "Alice"
HGETALL user:1                     # name "Alice" email "alice@example.com" login_count "42"
HINCRBY user:1 login_count 1       # 43

# 列表（Lists）：有序集合（底层为链表）
LPUSH notifications:user:1 "New order #1234"
LPUSH notifications:user:1 "Payment received"
LRANGE notifications:user:1 0 9    # 最近 10 条通知
LLEN notifications:user:1          # 通知总数

# 集合（Sets）：无序唯一集合
SADD product:42:tags "electronics" "wireless" "bluetooth"
SMEMBERS product:42:tags           # {"electronics", "wireless", "bluetooth"}
SISMEMBER product:42:tags "wireless"  # 1（true）
# 集合运算
SINTER product:42:tags product:99:tags  # 两产品共有的标签

# 有序集合（Sorted Sets）：按 score 排序（极适合排行榜、排名）
ZADD leaderboard 1500 "player:alice"
ZADD leaderboard 2300 "player:bob"
ZADD leaderboard 1800 "player:carol"
ZREVRANGE leaderboard 0 2 WITHSCORES
# 1) "player:bob"    2) "2300"
# 3) "player:carol"  4) "1800"
# 5) "player:alice"  6) "1500"
ZRANK leaderboard "player:carol"   # 1（0-indexed，升序）
ZREVRANK leaderboard "player:carol" # 1（0-indexed，降序）
```

### 持久化：RDB vs AOF

Redis 主要是一个内存数据库，但它提供了两种持久化机制：

| 特性 | RDB（快照） | AOF（追加日志） |
|------|-------------|-----------------|
| 工作原理 | 定期将全量数据快照写入磁盘 | 记录每一条写命令 |
| 数据丢失风险 | 最多丢失上次快照间隔内的数据 | 可配置为每秒同步或每条命令同步 |
| 恢复速度 | 快（直接加载二进制文件） | 较慢（需重放所有命令） |
| 文件大小 | 紧凑（二进制格式） | 更大（文本命令，但可通过重写压缩） |
| CPU 开销 | 快照时 fork 会造成瞬时峰值 | 开销平稳（持续追加到文件） |

```bash
# redis.conf：启用双持久化以获得最高安全性
save 900 1        # 900 秒内 ≥1 个 key 变更则快照
save 300 10       # 300 秒内 ≥10 个 key 变更则快照
save 60 10000     # 60 秒内 ≥10000 个 key 变更则快照

appendonly yes
appendfsync everysec   # 每秒 fsync（良好平衡点）
```

### Redis 常见模式

```bash
# 限流（滑动窗口）
# 每用户每分钟最多 100 次请求
MULTI
ZADD ratelimit:user:1 1702345678.123 "req-uuid-1"
ZREMRANGEBYSCORE ratelimit:user:1 0 1702345618.123  # 清除 >60s 的旧记录
ZCARD ratelimit:user:1  # 统计剩余请求数
EXPIRE ratelimit:user:1 60
EXEC

# 分布式锁（简化版）
SET lock:process-payments "" NX EX 30  # 获取锁，30 秒超时
# NX = 仅当 key 不存在时才设置
# 返回 OK 表示成功获取，nil 表示已被锁定

# 发布/订阅（Pub/Sub）
SUBSCRIBE channel:orders
PUBLISH channel:orders '{"order_id": 1234, "action": "created"}'

# 缓存旁路（Cache-aside）模式
# 1. 检查缓存
GET product:42
# 2. 若未命中，则查库并写入缓存
SET product:42 '{"name":"Widget","price":9.99}' EX 300  # 5 分钟 TTL
```

## 宽列式数据库：Cassandra

![不同NoSQL数据库类型作为不同的架构风格](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/05-different-nosql-database-types-as-distinct-architectural-sty.jpg)

宽列式数据库（有时也称为列族存储）专为海量规模和可预测的性能而设计。Apache Cassandra 是其中最具代表性的实现。

![NoSQL数据库类型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/05-nosql-types.png)

### 数据模型

Cassandra 使用表结构，其主键由两部分组成：
- **分区键（Partition key）**：决定数据分布到哪个节点（用于数据分片）
- **聚类键（Clustering key）**：决定同一分区内数据的排序顺序

```sql
-- CQL（Cassandra Query Language）
CREATE TABLE user_activity (
    user_id     UUID,
    activity_date DATE,
    activity_time TIMESTAMP,
    activity_type TEXT,
    details       MAP<TEXT, TEXT>,
    PRIMARY KEY ((user_id), activity_date, activity_time)
) WITH CLUSTERING ORDER BY (activity_date DESC, activity_time DESC);
```

这里，`user_id` 是分区键，`(activity_date, activity_time)` 是聚类键。同一个用户的所有活动都存储在同一节点上，并按日期和时间降序排列。

```sql
-- 插入数据
INSERT INTO user_activity (user_id, activity_date, activity_time, activity_type, details)
VALUES (
    550e8400-e29b-41d4-a716-446655440000,
    '2023-12-15',
    '2023-12-15T14:30:00Z',
    'purchase',
    {'product_id': '42', 'amount': '149.99'}
);

-- 查询：获取某用户近期活动（高效——单分区查询）
SELECT * FROM user_activity
WHERE user_id = 550e8400-e29b-41d4-a716-446655440000
  AND activity_date >= '2023-12-01'
LIMIT 20;

-- 跨分区查询：应避免！（全集群扫描）
-- SELECT * FROM user_activity WHERE activity_type = 'purchase';
-- 此操作需 `ALLOW FILTERING`，将扫描整个集群
```

### 何时选用 Cassandra

| 场景 | 为何契合 Cassandra |
|------|-------------------|
| 时间序列数据 | 按实体分区，按时间聚类 |
| IoT 传感器数据 | 支持极高的写入吞吐量，延迟可预测 |
| 用户行为日志 | 按用户分区，便于查询近期活动 |
| 即时通讯 / 聊天 | 按会话分区，按时间戳聚类 |
| 地理分布数据 | 支持跨数据中心复制 |

### Cassandra 反模式

- **跨分区的随机读取**：每个分区可能位于不同节点，导致多次网络跳转
- **JOIN 操作**：不支持——需通过反范式化或物化视图解决
- **轻量级事务（LWT）**：虽然支持，但基于 Paxos，开销较大
- **高基数列上的二级索引**：性能表现很差

## 图数据库：Neo4j

当“关系”本身就是核心数据时——比如社交网络、欺诈检测、推荐引擎或知识图谱——图数据库就是最自然的选择。

### 数据模型

图由两个基本元素构成：
- **节点（Nodes / vertices）**：带标签和属性的实体
- **关系（Relationships / edges）**：有类型的节点间连接，也可以携带属性

```text
(Alice:Person {name: "Alice", age: 30})
    -[:FRIENDS_WITH {since: 2020}]->
(Bob:Person {name: "Bob", age: 28})
    -[:WORKS_AT {role: "Engineer"}]->
(Acme:Company {name: "Acme Corp", industry: "Tech"})
```

### Cypher 查询语言

```cypher
// 创建节点与关系
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 28})
CREATE (carol:Person {name: "Carol", age: 32})
CREATE (acme:Company {name: "Acme Corp"})
CREATE (alice)-[:FRIENDS_WITH {since: 2020}]->(bob)
CREATE (alice)-[:FRIENDS_WITH {since: 2019}]->(carol)
CREATE (bob)-[:WORKS_AT {role: "Engineer", since: 2021}]->(acme)
CREATE (carol)-[:WORKS_AT {role: "Designer", since: 2020}]->(acme)

// 查找 Alice 的好友
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN friend.name, friend.age

// 查找好友的好友（2 跳）
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH*2]->(fof)
WHERE fof <> alice
RETURN DISTINCT fof.name

// 查找两人间最短路径
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(bob:Person {name: "Bob"})
)
RETURN path, length(path)

// 推荐：与 Alice 的好友就职于同一家公司的其他人
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)-[:WORKS_AT]->(company)<-[:WORKS_AT]-(colleague)
WHERE NOT (alice)-[:FRIENDS_WITH]->(colleague)
  AND colleague <> alice
RETURN colleague.name, company.name, count(*) AS mutual_connections
ORDER BY mutual_connections DESC
```

### 图数据库 vs 关系型数据库：JOIN 困境

在 SQL 中查找“朋友的朋友的朋友”：

```sql
-- SQL 中的 3 跳好友查询（痛苦）
SELECT DISTINCT p4.name
FROM friendships f1
JOIN friendships f2 ON f1.friend_id = f2.person_id
JOIN friendships f3 ON f2.friend_id = f3.person_id
JOIN people p4 ON f3.friend_id = p4.person_id
WHERE f1.person_id = 1
  AND p4.person_id != 1;
-- 每增加一跳，性能呈指数级下降
-- 在百万级用户的社交图中，此查询几乎不可行
```

同样的查询用 Cypher 表达：

```cypher
MATCH (alice:Person {id: 1})-[:FRIENDS_WITH*3]->(fofof)
WHERE fofof <> alice
RETURN DISTINCT fofof.name
// 图数据库采用“无索引邻接”（index-free adjacency）——每个节点直接引用其邻居。
// 无需关联表，无需索引查找。
// 性能取决于结果数量，而非图的总规模。
```

## 时序数据库

时序数据——监控指标、IoT 传感器读数、金融行情、应用日志——具有独特的访问模式：写多读少、只追加、查询几乎总是按时间范围过滤、旧数据可降采样或过期。

### 为什么不直接用 PostgreSQL？

PostgreSQL *可以*存储时序数据，但在规模化时遇到困难：

| 挑战 | RDBMS 方式 | 时序数据库方式 |
|------|-----------|--------------|
| 数十亿行 | 表膨胀，vacuum 缓慢 | 按时间分区，自动保留策略 |
| 写入吞吐 | WAL 瓶颈（>10万行/秒） | 批量追加，LSM/列式 |
| 范围查询 | B-tree 逐行寻址 | 时间有序块顺序扫描 |
| 数据保留 | 手动 DELETE + vacuum | 自动 TTL 策略 |
| 聚合 | 全表扫描或物化视图 | 预计算汇总 |

### TimescaleDB（PostgreSQL 扩展）

TimescaleDB 为 PostgreSQL 添加时序超能力——保持完整 SQL 兼容性：

```sql
-- 创建超表（按时间自动分区）
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    host VARCHAR(50),
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_io FLOAT
);

SELECT create_hypertable('metrics', 'time');

-- 时间桶聚合（TimescaleDB 特有）
SELECT
    time_bucket('5 minutes', time) AS bucket,
    host,
    AVG(cpu_usage) AS avg_cpu,
    MAX(cpu_usage) AS peak_cpu
FROM metrics
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY bucket, host
ORDER BY bucket DESC;

-- 连续聚合（物化，自动刷新）
CREATE MATERIALIZED VIEW hourly_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    host,
    AVG(cpu_usage) AS avg_cpu,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY cpu_usage) AS p95_cpu
FROM metrics
GROUP BY hour, host;

-- 数据保留策略（自动删除 90 天前的数据）
SELECT add_retention_policy('metrics', INTERVAL '90 days');
```

### InfluxDB

InfluxDB 使用为时序优化的查询语言 Flux：

```flux
from(bucket: "monitoring")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu" and r.host == "web-01")
  |> aggregateWindow(every: 5m, fn: mean)
  |> yield()
```

### 时序数据库选型

| 数据库 | 最佳场景 | 查询语言 | 部署方式 |
|--------|----------|----------|----------|
| TimescaleDB | 已在用 PostgreSQL 的团队 | SQL（完整） | 扩展（自建或云） |
| InfluxDB | 监控指标管道 | Flux / InfluxQL | 独立 / 云 |
| QuestDB | 超低延迟摄入 | SQL 子集 | 独立部署 |
| ClickHouse | 事件数据分析 | SQL（扩展） | 独立 / 云 |
| Prometheus | 拉取式指标采集 | PromQL | 独立（配远端存储） |

## 向量数据库

向量数据库存储高维嵌入向量并支持相似度搜索。它们驱动语义搜索、推荐系统、RAG（检索增强生成）和图像检索。

### 向量搜索工作原理

传统数据库查找精确匹配。向量数据库查找嵌入空间中的*最近邻*：

```text
查询: "如何部署 Python 应用？"
                                          余弦相似度
嵌入 → [0.23, -0.14, 0.87, ...]  ──────────────→  Top-K 结果
                                        0.94: "Flask 生产部署指南"
                                        0.91: "Python 应用 Docker 教程"
                                        0.87: "CI/CD 流水线配置"
```

### 索引类型

| 算法 | 速度 | 精度 | 内存 | 适用 |
|------|------|------|------|------|
| Flat（暴力搜索） | 慢 | 100% | 低 | <10万向量 |
| IVF（倒排索引） | 快 | ~95% | 中 | 10万-1000万 |
| HNSW（分层可导航小世界） | 很快 | ~98% | 高 | 通用场景 |
| PQ（乘积量化） | 快 | ~90% | 很低 | 十亿级向量 |

### Milvus / Zilliz

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect(host="localhost", port="19530")

fields = [
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("text", DataType.VARCHAR, max_length=1000),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)

# 创建 HNSW 索引
collection.create_index("embedding", {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
})

# 搜索
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    output_fields=["text"],
)
```

### pgvector（PostgreSQL 扩展）

已在用 PostgreSQL 的团队想要向量搜索但不想引入新数据库时：

```sql
CREATE EXTENSION vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- HNSW 索引
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 语义搜索
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM documents
ORDER BY embedding <=> $1
LIMIT 10;
```

### 专用向量数据库 vs pgvector

| 因素 | pgvector | 专用（Milvus, Pinecone, Qdrant） |
|------|----------|----------------------------------|
| 向量数量 | <500万 | 百万至数十亿 |
| 已有架构 | 已在用 PostgreSQL | 全新项目或专用场景 |
| 混合查询 | SQL + 向量一条查询 | 分离系统 |
| 吞吐量 | 中等（~1K QPS） | 高（~100K QPS） |
| 过滤 | 完整 SQL WHERE | 元数据过滤（有限） |
| 运维成本 | 零（扩展） | 新增基础设施 |

## 多模型数据库

部分数据库在单一系统中支持多种数据模型，免去多个数据库之间的同步。

### 示例

| 数据库 | 支持的模型 | 适用场景 |
|--------|-----------|----------|
| PostgreSQL + 扩展 | 关系、文档(jsonb)、向量(pgvector)、时序(TimescaleDB)、图(AGE) | 中等规模的"万能数据库" |
| ArangoDB | 文档、图、键值 | 需要图+文档的应用 |
| SurrealDB | 文档、图、关系 | 追求灵活性的新项目 |
| CosmosDB | 文档、图、键值、列族、表 | Azure 生态 |

### PostgreSQL 生态方式

不必学习新数据库，扩展 PostgreSQL 即可：

```sql
-- 文档模型
CREATE TABLE products (id SERIAL PRIMARY KEY, data JSONB);

-- 时序（TimescaleDB）
CREATE TABLE metrics (time TIMESTAMPTZ, value FLOAT);
SELECT create_hypertable('metrics', 'time');

-- 向量搜索（pgvector）
CREATE TABLE embeddings (id INT, vec vector(768));

-- 全文搜索（内置）
CREATE INDEX idx_fts ON articles USING GIN (to_tsvector('english', content));
```

一套部署、一套备份策略、一个连接池、所有模型完整 ACID。代价：每个扩展称职但在极端规模下非同类最强。

## CAP 定理

CAP 定理指出：一个分布式系统最多只能同时满足以下三个保证中的两项：

![CAP定理权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/05-cap-theorem.png)

- **一致性（Consistency）**：每次读操作都能获得最新的写入结果
- **可用性（Availability）**：每个请求都能得到响应（即使不是最新数据）
- **分区容忍性（Partition Tolerance）**：即使发生网络分区，系统仍能继续运行

由于网络分区在分布式系统中不可避免，实际的选择只能在 **CP** 和 **AP** 之间权衡：

| 选项 | 分区期间行为 | 示例 |
|------|--------------|------|
| **CP**（一致性 + 分区容忍） | 拒绝无法保证一致性的请求 | HBase、MongoDB（启用 majority write concern）、etcd、ZooKeeper |
| **AP**（可用性 + 分区容忍） | 仍响应请求，但可能返回陈旧数据 | Cassandra、DynamoDB、CouchDB、Riak |
| **CA**（一致性 + 可用性） | 在分布式系统中无法实现 | 单机 PostgreSQL / MySQL（非分布式） |

实践中，大多数数据库允许你针对每个操作单独调整一致性与可用性之间的权衡：

```javascript
// MongoDB：可调写的写关注（write concern）
await collection.insertOne(doc, {
  writeConcern: { w: "majority", j: true }  // CP 行为
});

await collection.insertOne(doc, {
  writeConcern: { w: 1 }  // AP 行为（仅主节点确认）
});
```

```sql
-- Cassandra：可调每条查询的一致性级别
-- Quorum 读 + Quorum 写 = 强一致性
SELECT * FROM users WHERE user_id = ? CONSISTENCY QUORUM;
INSERT INTO users (...) VALUES (...) USING CONSISTENCY QUORUM;

-- ONE = 快速但可能返回陈旧数据
SELECT * FROM users WHERE user_id = ? CONSISTENCY ONE;
```

## NewSQL：鱼与熊掌兼得？

NewSQL 数据库试图融合 SQL、ACID 事务和水平扩展能力：

| 数据库 | 架构 | 核心特性 |
|--------|------|----------|
| CockroachDB | Raft 共识 + 基于 Range 的分片 | 兼容 PostgreSQL 协议，可容忍区域故障 |
| TiDB | TiKV 存储层（基于 RocksDB）+ TiDB SQL 层 | 兼容 MySQL 协议，支持 HTAP（混合事务/分析处理） |
| YugabyteDB | DocDB 存储 + Raft 共识 | 同时提供 PostgreSQL 和 Cassandra 兼容的 API |
| Google Spanner | TrueTime（原子钟）+ Paxos | 提供全局强一致性，具备外部一致性（external consistency） |

```sql
-- CockroachDB：语法像 PostgreSQL，扩展性如 Cassandra
CREATE TABLE orders (
    order_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    total DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 事务行为与 PostgreSQL 完全一致
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- 但数据通过 Raft 共识分布在多个节点上
```

## 决策表：如何选择合适的数据库

| 需求 | 最佳匹配 | 示例 |
|------|----------|------|
| ACID 事务、复杂查询 | 关系型数据库 | PostgreSQL、MySQL |
| 灵活 schema、嵌套文档 | 文档型数据库 | MongoDB、Firestore |
| 超低延迟缓存 | 键值型数据库 | Redis、Memcached |
| 海量写入吞吐、时间序列 | 宽列式数据库 | Cassandra、HBase |
| 关系密集型查询 | 图数据库 | Neo4j、Amazon Neptune |
| SQL + 水平扩展 | NewSQL | CockroachDB、TiDB |
| 实时分析 | 列式数据库 | ClickHouse、DuckDB |
| 全文搜索 | 搜索引擎 | Elasticsearch、Meilisearch |
| 全球部署 + 强一致性 | 托管 NewSQL | Google Spanner、CockroachDB |

现实中，最佳答案往往是：以 PostgreSQL 作为主数据存储，再为特定工作负载搭配专用数据库。大多数成功的系统都会组合使用 2–3 种数据库，而不是孤注一掷地依赖单一方案。

## 下一步

无论你选择关系型还是 NoSQL，单台机器终将成为瓶颈。在下一篇文章中，我们将深入探讨 **复制（replication）与分片（partitioning）**——这些技术让数据库得以突破单机限制，在保持（某种程度）一致性的同时实现规模化扩展。
