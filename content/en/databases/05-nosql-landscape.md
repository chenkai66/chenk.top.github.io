---
title: "Databases (5): NoSQL — Document, Key-Value, Column, and Graph"
date: 2024-04-24 09:00:00
tags:
  - Databases
  - NoSQL
  - MongoDB
  - Redis
  - Cassandra
categories:
  - Databases
series: databases
lang: en
description: "A practical tour of the four major NoSQL families — document, key-value, wide-column, and graph — including the CAP theorem and when to use each model."
disableNunjucks: true
series_order: 5
translationKey: "databases-5"
---

Not everything fits neatly into rows and columns. A social network's friend graph, a product catalog with wildly varying attributes, a real-time leaderboard, a recommendation engine's relationship web — these workloads push relational databases into awkward territory. NoSQL databases exist because different data models solve different problems better. The trick is knowing which one to reach for.

## Why NoSQL?

The term "NoSQL" is misleading. It does not mean "no SQL" — some NoSQL databases support SQL-like query languages. It means "not only SQL" or, more accurately, "non-relational." The motivations for NoSQL fall into three categories:

1. **Schema flexibility**: Your data does not have a fixed schema, or the schema changes frequently
2. **Scale-out architecture**: You need horizontal scaling beyond what a single relational database can handle
3. **Data model fit**: Your data is naturally a document, graph, key-value pair, or time series — not a table

Let us explore each family.

## Document Stores: MongoDB

Document databases store data as semi-structured documents, typically JSON (or BSON in MongoDB's case). Each document can have a different structure — no fixed schema.

### Data Model

```json
// A user document in MongoDB
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

In a relational database, this would require at least 3 tables: `users`, `addresses`, `user_preferences`. In MongoDB, it is one document. No JOINs needed.

### CRUD Operations

```javascript
// Connect to MongoDB
const db = client.db("ecommerce");
const users = db.collection("users");

// Create
await users.insertOne({
  email: "alice@example.com",
  name: "Alice Chen",
  addresses: [{ type: "home", city: "San Francisco" }],
  created_at: new Date()
});

// Read
const user = await users.findOne({ email: "alice@example.com" });

// Read with projection (select specific fields)
const userBasic = await users.findOne(
  { email: "alice@example.com" },
  { projection: { name: 1, email: 1, _id: 0 } }
);

// Update: add a new address
await users.updateOne(
  { email: "alice@example.com" },
  { $push: { addresses: { type: "work", city: "Oakland" } } }
);

// Update: increment a counter
await users.updateOne(
  { _id: userId },
  { $inc: { login_count: 1 }, $set: { last_login: new Date() } }
);

// Delete
await users.deleteOne({ email: "alice@example.com" });
```

### Querying and Filtering

```javascript
// Find users in San Francisco with dark theme
const result = await users.find({
  "addresses.city": "San Francisco",
  "preferences.theme": "dark"
}).sort({ created_at: -1 }).limit(10).toArray();

// Find users created in the last 30 days
const recent = await users.find({
  created_at: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
}).toArray();

// Text search (requires text index)
await users.createIndex({ name: "text", email: "text" });
const searchResults = await users.find({
  $text: { $search: "alice chen" }
}).toArray();
```

### Aggregation Pipeline

MongoDB's aggregation framework is surprisingly powerful — it can do many things that SQL does with GROUP BY, JOINs, and window functions:

```javascript
// Revenue by product category for the last quarter
const pipeline = [
  // Stage 1: Filter orders from last quarter
  { $match: {
    status: "completed",
    created_at: { $gte: new Date("2023-10-01") }
  }},
  // Stage 2: Unwind order items array (one doc per item)
  { $unwind: "$items" },
  // Stage 3: Lookup product details
  { $lookup: {
    from: "products",
    localField: "items.product_id",
    foreignField: "_id",
    as: "product"
  }},
  // Stage 4: Flatten product array
  { $unwind: "$product" },
  // Stage 5: Group by category
  { $group: {
    _id: "$product.category",
    total_revenue: { $sum: { $multiply: ["$items.quantity", "$items.price"] } },
    order_count: { $sum: 1 },
    avg_order_value: { $avg: { $multiply: ["$items.quantity", "$items.price"] } }
  }},
  // Stage 6: Sort by revenue
  { $sort: { total_revenue: -1 } },
  // Stage 7: Rename fields for output
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

### When Documents Work Well

| Scenario | Why documents fit |
|----------|------------------|
| Product catalogs | Products have different attributes (shoes vs laptops) |
| Content management | Articles, blog posts with nested comments |
| User profiles | Varying preferences and metadata |
| Event logging | Semi-structured event data |
| Mobile app backends | JSON in, JSON out, schema evolves fast |

### When Documents Do Not Work Well

- **Many-to-many relationships**: Duplication or manual reference management
- **Complex transactions across documents**: Limited multi-document transaction support
- **Heavy aggregation/analytics**: SQL databases and column stores are more efficient
- **Strong consistency requirements**: Eventual consistency is the default in distributed setups

## Key-Value Stores: Redis

A key-value store is the simplest NoSQL model: you give it a key, it gives you back a value. Redis takes this further by supporting rich data structures as values.

### Data Structures

```bash
# Strings: the basic key-value pair
SET user:1:name "Alice Chen"
GET user:1:name                    # "Alice Chen"

SET page:home:views 0
INCR page:home:views               # 1 (atomic increment)
INCR page:home:views               # 2
INCRBY page:home:views 100         # 102

# Strings with expiration (TTL)
SET session:abc123 '{"user_id":1,"role":"admin"}' EX 3600  # expires in 1 hour
TTL session:abc123                 # 3597 (seconds remaining)

# Hashes: like a mini-document
HSET user:1 name "Alice" email "alice@example.com" login_count 42
HGET user:1 name                   # "Alice"
HGETALL user:1                     # name "Alice" email "alice@example.com" login_count "42"
HINCRBY user:1 login_count 1       # 43

# Lists: ordered collection (linked list under the hood)
LPUSH notifications:user:1 "New order #1234"
LPUSH notifications:user:1 "Payment received"
LRANGE notifications:user:1 0 9    # latest 10 notifications
LLEN notifications:user:1          # number of notifications

# Sets: unordered unique collection
SADD product:42:tags "electronics" "wireless" "bluetooth"
SMEMBERS product:42:tags           # {"electronics", "wireless", "bluetooth"}
SISMEMBER product:42:tags "wireless"  # 1 (true)
# Set operations
SINTER product:42:tags product:99:tags  # tags common to both products

# Sorted Sets: ordered by score (great for leaderboards, rankings)
ZADD leaderboard 1500 "player:alice"
ZADD leaderboard 2300 "player:bob"
ZADD leaderboard 1800 "player:carol"
ZREVRANGE leaderboard 0 2 WITHSCORES
# 1) "player:bob"    2) "2300"
# 3) "player:carol"  4) "1800"
# 5) "player:alice"  6) "1500"
ZRANK leaderboard "player:carol"   # 1 (0-indexed, ascending)
ZREVRANK leaderboard "player:carol" # 1 (0-indexed, descending)
```

### Persistence: RDB vs AOF

Redis is primarily an in-memory database, but it offers two persistence mechanisms:

| Feature | RDB (Snapshotting) | AOF (Append-Only File) |
|---------|-------------------|----------------------|
| How it works | Periodic full snapshot to disk | Logs every write command |
| Data loss risk | Up to last snapshot interval | Configurable: every second or every command |
| Recovery speed | Fast (load binary file) | Slower (replay all commands) |
| File size | Compact (binary format) | Larger (text commands, but can be compacted) |
| CPU impact | Spike during snapshot (fork) | Steady (append to file) |

```bash
# redis.conf: enable both for maximum safety
save 900 1        # snapshot if >= 1 key changed in 900 seconds
save 300 10       # snapshot if >= 10 keys changed in 300 seconds
save 60 10000     # snapshot if >= 10000 keys changed in 60 seconds

appendonly yes
appendfsync everysec   # fsync once per second (good balance)
```

### Common Redis Patterns

```bash
# Rate limiting (sliding window)
# Allow max 100 requests per minute per user
MULTI
ZADD ratelimit:user:1 1702345678.123 "req-uuid-1"
ZREMRANGEBYSCORE ratelimit:user:1 0 1702345618.123  # remove entries > 60s old
ZCARD ratelimit:user:1  # count remaining entries
EXPIRE ratelimit:user:1 60
EXEC

# Distributed lock (simplified)
SET lock:process-payments "" NX EX 30  # acquire lock, 30s timeout
# NX = only set if key does not exist
# Returns OK if acquired, nil if already locked

# Pub/Sub
SUBSCRIBE channel:orders
PUBLISH channel:orders '{"order_id": 1234, "action": "created"}'

# Cache-aside pattern
# 1. Check cache
GET product:42
# 2. If miss, query database and set cache
SET product:42 '{"name":"Widget","price":9.99}' EX 300  # 5-min TTL
```

## Wide-Column Stores: Cassandra

Wide-column databases (sometimes called column-family stores) are designed for massive scale with predictable performance. Apache Cassandra is the most prominent example.

### Data Model

Cassandra uses tables with a primary key composed of:
- **Partition key**: determines which node stores the data (distribution)
- **Clustering key**: determines sort order within a partition

```sql
-- CQL (Cassandra Query Language)
CREATE TABLE user_activity (
    user_id     UUID,
    activity_date DATE,
    activity_time TIMESTAMP,
    activity_type TEXT,
    details       MAP<TEXT, TEXT>,
    PRIMARY KEY ((user_id), activity_date, activity_time)
) WITH CLUSTERING ORDER BY (activity_date DESC, activity_time DESC);
```

Here, `user_id` is the partition key and `(activity_date, activity_time)` is the clustering key. All activities for a single user are stored together on the same node, sorted by date and time descending.

```sql
-- Insert data
INSERT INTO user_activity (user_id, activity_date, activity_time, activity_type, details)
VALUES (
    550e8400-e29b-41d4-a716-446655440000,
    '2023-12-15',
    '2023-12-15T14:30:00Z',
    'purchase',
    {'product_id': '42', 'amount': '149.99'}
);

-- Query: get recent activity for a user (efficient - single partition)
SELECT * FROM user_activity
WHERE user_id = 550e8400-e29b-41d4-a716-446655440000
  AND activity_date >= '2023-12-01'
LIMIT 20;

-- Query across partitions: AVOID THIS (full cluster scan)
-- SELECT * FROM user_activity WHERE activity_type = 'purchase';
-- This requires ALLOW FILTERING and scans the entire cluster
```

### When to Use Cassandra

| Use case | Why Cassandra fits |
|----------|-------------------|
| Time-series data | Partition by entity, cluster by time |
| IoT sensor data | Massive write throughput, predictable latency |
| User activity logs | Partition by user, query recent activity |
| Messaging / chat | Partition by conversation, cluster by timestamp |
| Geographic data | Replicate across data centers |

### Cassandra Anti-Patterns

- **Random reads across partitions**: Each partition may be on a different node
- **Joins**: Not supported — denormalize or use materialized views
- **Lightweight transactions**: Cassandra supports them but they are expensive (Paxos-based)
- **Secondary indexes on high-cardinality columns**: Poor performance

## Graph Databases: Neo4j

When relationships *are* the data — social networks, fraud detection, recommendation engines, knowledge graphs — a graph database is the natural fit.

### Data Model

Graphs have two primitives:
- **Nodes** (vertices): entities with labels and properties
- **Relationships** (edges): typed connections between nodes, also with properties

```
(Alice:Person {name: "Alice", age: 30})
    -[:FRIENDS_WITH {since: 2020}]->
(Bob:Person {name: "Bob", age: 28})
    -[:WORKS_AT {role: "Engineer"}]->
(Acme:Company {name: "Acme Corp", industry: "Tech"})
```

### Cypher Query Language

```cypher
// Create nodes and relationships
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 28})
CREATE (carol:Person {name: "Carol", age: 32})
CREATE (acme:Company {name: "Acme Corp"})
CREATE (alice)-[:FRIENDS_WITH {since: 2020}]->(bob)
CREATE (alice)-[:FRIENDS_WITH {since: 2019}]->(carol)
CREATE (bob)-[:WORKS_AT {role: "Engineer", since: 2021}]->(acme)
CREATE (carol)-[:WORKS_AT {role: "Designer", since: 2020}]->(acme)

// Find Alice's friends
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN friend.name, friend.age

// Find friends of friends (2 hops)
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH*2]->(fof)
WHERE fof <> alice
RETURN DISTINCT fof.name

// Shortest path between two people
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(bob:Person {name: "Bob"})
)
RETURN path, length(path)

// Recommendation: people who work at the same company as Alice's friends
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)-[:WORKS_AT]->(company)<-[:WORKS_AT]-(colleague)
WHERE NOT (alice)-[:FRIENDS_WITH]->(colleague)
  AND colleague <> alice
RETURN colleague.name, company.name, count(*) AS mutual_connections
ORDER BY mutual_connections DESC
```

### Graph vs Relational: The JOIN Problem

Finding friends-of-friends-of-friends in SQL:

```sql
-- 3-hop friend query in SQL (painful)
SELECT DISTINCT p4.name
FROM friendships f1
JOIN friendships f2 ON f1.friend_id = f2.person_id
JOIN friendships f3 ON f2.friend_id = f3.person_id
JOIN people p4 ON f3.friend_id = p4.person_id
WHERE f1.person_id = 1
  AND p4.person_id != 1;
-- Performance degrades exponentially with each hop
-- On a social graph with millions of users, this is impractical
```

The same in Cypher:

```cypher
MATCH (alice:Person {id: 1})-[:FRIENDS_WITH*3]->(fofof)
WHERE fofof <> alice
RETURN DISTINCT fofof.name
// Graph databases use index-free adjacency — each node directly
// references its neighbors. No join tables, no index lookups.
// Performance depends on the number of results, not the total graph size.
```

## The CAP Theorem

The CAP theorem states that a distributed system can provide at most two of these three guarantees:

- **Consistency**: Every read receives the most recent write
- **Availability**: Every request receives a response (even if not the most recent data)
- **Partition Tolerance**: The system continues operating despite network partitions

Since network partitions are unavoidable in distributed systems, the real choice is between **CP** and **AP**:

| Choice | Behavior during partition | Examples |
|--------|--------------------------|---------|
| **CP** (Consistency + Partition Tolerance) | Refuses requests it cannot guarantee are consistent | HBase, MongoDB (with majority write concern), etcd, ZooKeeper |
| **AP** (Availability + Partition Tolerance) | Serves requests but may return stale data | Cassandra, DynamoDB, CouchDB, Riak |
| **CA** (Consistency + Availability) | Not possible in a distributed system | Single-node PostgreSQL / MySQL (not distributed) |

In practice, most databases let you tune the consistency/availability trade-off per operation:

```javascript
// MongoDB: tunable write concern
await collection.insertOne(doc, {
  writeConcern: { w: "majority", j: true }  // CP behavior
});

await collection.insertOne(doc, {
  writeConcern: { w: 1 }  // AP behavior (acknowledged by primary only)
});
```

```sql
-- Cassandra: tunable consistency per query
-- Quorum reads + quorum writes = strong consistency
SELECT * FROM users WHERE user_id = ? CONSISTENCY QUORUM;
INSERT INTO users (...) VALUES (...) USING CONSISTENCY QUORUM;

-- ONE = fast but possibly stale
SELECT * FROM users WHERE user_id = ? CONSISTENCY ONE;
```

## NewSQL: The Best of Both Worlds?

NewSQL databases attempt to provide SQL + ACID + horizontal scaling:

| Database | Architecture | Key feature |
|----------|-------------|-------------|
| CockroachDB | Raft consensus, range-based sharding | PostgreSQL wire protocol, survives zone failures |
| TiDB | TiKV storage (RocksDB) + TiDB SQL layer | MySQL protocol compatible, HTAP (hybrid) |
| YugabyteDB | DocDB storage, Raft consensus | PostgreSQL and Cassandra compatible APIs |
| Google Spanner | TrueTime (atomic clocks), Paxos | Global consistency with external consistency |

```sql
-- CockroachDB: looks like PostgreSQL, scales like Cassandra
CREATE TABLE orders (
    order_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    total DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Transactions work exactly like PostgreSQL
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- But the data is distributed across multiple nodes with Raft consensus
```

## Decision Table: Choosing the Right Database

| Requirement | Best fit | Examples |
|-------------|---------|---------|
| ACID transactions, complex queries | Relational | PostgreSQL, MySQL |
| Flexible schema, nested documents | Document store | MongoDB, Firestore |
| Ultra-low latency caching | Key-value | Redis, Memcached |
| Massive write throughput, time-series | Wide-column | Cassandra, HBase |
| Relationship-heavy queries | Graph | Neo4j, Amazon Neptune |
| SQL + horizontal scaling | NewSQL | CockroachDB, TiDB |
| Real-time analytics | Column-oriented | ClickHouse, DuckDB |
| Full-text search | Search engine | Elasticsearch, Meilisearch |
| Global distribution with strong consistency | Managed NewSQL | Google Spanner, CockroachDB |

The right answer is often "PostgreSQL" for your primary data store, with a specialized database for specific workloads. Most successful systems use 2-3 databases, not one.

## What's Next

Whether you choose relational or NoSQL, a single machine eventually becomes a bottleneck. In the next article, we will explore **replication and partitioning** — the techniques that let databases scale beyond one server while maintaining (some level of) consistency.
