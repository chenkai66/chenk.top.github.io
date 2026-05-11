---
title: "System Design (4): Caching — Where to Cache, What to Evict, and When Caching Hurts"
date: 2025-07-17 09:00:00
tags:
  - System Design
  - Caching
  - Redis
  - Performance
categories:
  - System Design
series: system-design
lang: en
description: "A deep dive into caching strategies across every layer of the stack — from CDN to database buffer pools — covering cache-aside, write-through, write-behind patterns, eviction policies, thundering herd mitigation, and practical Redis configurations."
disableNunjucks: true
series_order: 4
translationKey: "system-design-4"
---

There is an old joke in computer science that the two hardest problems are cache invalidation, naming things, and off-by-one errors. The joke works because cache invalidation really is that hard. But caching is also the single most effective technique for improving system performance. A well-placed cache can reduce latency by 100x, cut database load by 90%, and save thousands of dollars in infrastructure costs per month.

The trick is knowing where to cache, what patterns to use, and — critically — when caching will make your system worse instead of better.

## Why Caching Works

Caching exploits a fundamental property of most systems: access patterns are not uniform. A small fraction of data is accessed far more frequently than the rest.

Consider a social media platform. At any given moment, a tiny percentage of posts are trending and being viewed by millions of users. The remaining 99% of posts are viewed rarely. If you cache that top 1% in memory, you handle 80% of your read traffic without touching the database.

The benefits cascade:

**Latency reduction**: Redis serves a cached value in 0.1-0.5ms. A database query takes 5-50ms. A cross-service API call takes 10-100ms. Caching eliminates the most expensive operations.

**Throughput amplification**: A single Redis instance handles 100,000+ operations per second. A PostgreSQL instance handles 5,000-20,000 queries per second. Caching multiplies your system's effective throughput.

**Cost savings**: One Redis instance replaces 10-20 database read replicas. At cloud pricing, this can save $10,000-$50,000 per month.

## Cache Layers

Modern systems have caches at every layer. Understanding each layer prevents you from solving the wrong problem.

![Cache layers in a web application](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-cache-layers.png)


### Client-Side Cache

The browser caches HTTP responses based on `Cache-Control` headers.

```
Cache-Control: public, max-age=31536000    # CDN + browser cache for 1 year
Cache-Control: private, max-age=3600       # Browser only, 1 hour
Cache-Control: no-cache                    # Always revalidate with server
Cache-Control: no-store                    # Never cache
```

`ETag` and `If-None-Match` headers enable conditional requests:

```
# First request
GET /api/user/123
Response:
  200 OK
  ETag: "abc123"
  { "name": "Alice", "email": "alice@example.com" }

# Subsequent request
GET /api/user/123
If-None-Match: "abc123"
Response:
  304 Not Modified    # No body, use cached version
```

Client-side caching is free performance. Configure it first.

### CDN Cache

As covered in the previous article, CDNs cache static assets at edge locations worldwide. For API responses, CDN caching is possible but requires careful `Cache-Control` headers and `Vary` headers to avoid serving one user's data to another.

```
# Cacheable at CDN (public data)
Cache-Control: public, max-age=300, s-maxage=600
Vary: Accept-Encoding

# NOT cacheable at CDN (user-specific data)
Cache-Control: private, max-age=60
```

`s-maxage` overrides `max-age` for shared caches (CDN), letting you cache longer at the edge than in the browser.

### Application Cache

This is where Redis, Memcached, and in-process caches like Caffeine or Guava live. The application explicitly manages what is cached, when it is invalidated, and how it is refreshed.

### Database Cache

Databases have their own internal caches:

**PostgreSQL shared buffers**: Caches frequently accessed table and index pages in memory. Default is 128 MB; production systems typically set this to 25% of available RAM.

```
# postgresql.conf
shared_buffers = 8GB          # 25% of 32GB RAM
effective_cache_size = 24GB   # 75% of RAM (OS + PG cache combined)
```

**MySQL InnoDB Buffer Pool**: Caches table data and indexes. Should be 70-80% of available memory on a dedicated database server.

```
# my.cnf
innodb_buffer_pool_size = 24G   # 75% of 32GB RAM
innodb_buffer_pool_instances = 8  # Reduce contention
```

**Query Cache** (MySQL, deprecated in 8.0): Cached the result set of SELECT queries. Invalidated on any write to any table referenced in the query. Caused more problems than it solved for write-heavy workloads — every write invalidated all cached queries on that table, creating lock contention.

## Caching Patterns

There are four fundamental patterns for integrating a cache with a database. Each has different consistency guarantees and failure modes.

![Caching patterns comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-caching-patterns.png)


### Cache-Aside (Lazy Loading)

The application manages the cache explicitly. On read, check the cache first. On cache miss, read from the database and populate the cache.

```python
import redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_user(user_id: str) -> dict:
    # Step 1: Check cache
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # Step 2: Cache miss — read from database
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user is None:
        return None

    # Step 3: Populate cache with TTL
    r.setex(f"user:{user_id}", 3600, json.dumps(user))

    return user

def update_user(user_id: str, data: dict):
    # Update database
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # Invalidate cache (NOT update — delete is safer)
    r.delete(f"user:{user_id}")
```

**Why delete instead of update the cache?** Consider two concurrent requests that update the same user. If both try to update the cache, a race condition can leave the cache with stale data. Deleting the cache forces the next read to fetch from the database, which is always authoritative.

**Advantages**:
- Simple to implement and reason about
- Cache only contains data that is actually requested (no wasted space)
- Cache failure is not catastrophic (falls through to database)

**Disadvantages**:
- First request for each key always hits the database (cold start)
- Potential for stale data between database update and cache invalidation
- Application code is cluttered with caching logic

### Write-Through

Every write goes to the cache and the database simultaneously. The cache is always up-to-date.

```python
def update_user_write_through(user_id: str, data: dict):
    # Write to database
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # Write to cache (same transaction conceptually)
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))

def get_user_write_through(user_id: str) -> dict:
    # Always read from cache
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # Only on cold start or eviction
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user:
        r.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Advantages**:
- Cache is always consistent with database (no stale reads)
- Read path is simple (always hit cache)

**Disadvantages**:
- Write latency increases (must write to both cache and DB)
- Caches data that may never be read (waste of cache space)
- Cache and DB writes are not truly atomic (failure between them causes inconsistency)

### Write-Behind (Write-Back)

Writes go to the cache immediately. The cache asynchronously flushes to the database in the background.

```python
import threading
import time
from collections import defaultdict

class WriteBehindCache:
    def __init__(self, flush_interval=5):
        self.dirty = {}
        self.lock = threading.Lock()
        self.flush_interval = flush_interval
        self._start_flusher()

    def write(self, key: str, value: dict):
        with self.lock:
            r.set(f"user:{key}", json.dumps(value))
            self.dirty[key] = value

    def read(self, key: str) -> dict:
        cached = r.get(f"user:{key}")
        if cached:
            return json.loads(cached)
        return None

    def _flush(self):
        while True:
            time.sleep(self.flush_interval)
            with self.lock:
                batch = dict(self.dirty)
                self.dirty.clear()

            for key, value in batch.items():
                try:
                    db.execute(
                        "INSERT INTO users (id, name) VALUES (%s, %s) "
                        "ON CONFLICT (id) DO UPDATE SET name = %s",
                        key, value["name"], value["name"]
                    )
                except Exception as e:
                    # Re-add to dirty set for retry
                    with self.lock:
                        self.dirty[key] = value
                    logger.error(f"Flush failed for {key}: {e}")

    def _start_flusher(self):
        t = threading.Thread(target=self._flush, daemon=True)
        t.start()
```

**Advantages**:
- Extremely low write latency (only writes to cache)
- Batches database writes, reducing DB load
- Absorbs write spikes

**Disadvantages**:
- Data loss risk: if cache crashes before flushing, unflushed writes are lost
- Complexity: async flushing, retry logic, ordering guarantees
- Debugging is harder (data in cache may not be in database yet)

Use write-behind for high-write-throughput scenarios where some data loss is acceptable (analytics counters, view counts, activity logs).

### Read-Through

The cache sits in front of the database and handles reads transparently. On cache miss, the cache itself loads from the database (not the application).

This pattern is typically implemented by cache libraries or frameworks rather than application code. It is functionally similar to cache-aside but encapsulates the loading logic within the cache layer.

```python
from cachetools import TTLCache

# Python example using cachetools with a loader function
cache = TTLCache(maxsize=10000, ttl=3600)

def get_user_read_through(user_id: str) -> dict:
    if user_id in cache:
        return cache[user_id]

    # Cache loads from DB on miss
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user:
        cache[user_id] = user
    return user
```

## Cache Eviction Policies

When a cache reaches capacity, it must decide which entries to remove. The choice of eviction policy significantly affects cache hit rate.

![Cache eviction policies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-eviction-policies.png)


### Policy Comparison

| Policy | How It Works | Hit Rate | Overhead | Best For |
|--------|-------------|----------|----------|----------|
| LRU (Least Recently Used) | Evict the entry that was accessed longest ago | Good | O(1) with doubly-linked list + hash map | General purpose, most common |
| LFU (Least Frequently Used) | Evict the entry with the lowest access count | Better for skewed distributions | Higher (maintain counters) | Hot/cold data with stable popularity |
| FIFO (First In, First Out) | Evict the oldest entry | Poor | O(1) | Simple, when access recency does not matter |
| Random | Evict a random entry | Surprisingly decent | O(1) | When access patterns are uniform |
| TTL-based | Evict entries after a time limit | N/A (not capacity-based) | O(1) per entry | Time-sensitive data |

### LRU Implementation

LRU is the most widely used eviction policy. Here is a clean implementation:

![LRU cache eviction animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/sysdesign-04-lru-cache.gif)


```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove from front (least recently used)
            self.cache.popitem(last=False)
```

Redis uses an approximated LRU algorithm. Instead of tracking the exact LRU order (which would require significant memory), it samples a configurable number of keys and evicts the least recently used among the sample. With the default sample size of 5, the approximation is remarkably close to true LRU.

## Redis as a Cache

Redis is the de facto standard for application-level caching. Here is a practical configuration.

![Redis architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-redis-architecture.png)


### Redis Configuration for Caching

```bash
# redis.conf for caching use case

# Memory limit
maxmemory 8gb

# Eviction policy when memory limit is reached
# allkeys-lru: Evict any key using approximated LRU
# volatile-lru: Only evict keys with TTL set
# allkeys-lfu: Evict any key using approximated LFU
# noeviction: Return errors when memory is full
maxmemory-policy allkeys-lru

# LRU approximation sample size (higher = more accurate, more CPU)
maxmemory-samples 10

# Persistence: disable for pure caching (faster, no disk I/O)
save ""
appendonly no

# Connection limits
maxclients 10000

# TCP keepalive
tcp-keepalive 300

# Timeout for idle connections (0 = no timeout)
timeout 300
```

### Common Redis Caching Patterns

**Simple key-value caching**:

```python
import redis
import json

r = redis.Redis(host="cache.internal", port=6379, decode_responses=True)

# Cache a user profile for 1 hour
def cache_user(user_id: str, user_data: dict):
    r.setex(f"user:{user_id}", 3600, json.dumps(user_data))

# Cache with conditional set (only if not exists — prevent overwrite)
def cache_user_if_missing(user_id: str, user_data: dict):
    r.set(f"user:{user_id}", json.dumps(user_data), ex=3600, nx=True)
```

**Hash-based caching** (more memory-efficient for objects):

```python
# Store user as a Redis hash
def cache_user_hash(user_id: str, user_data: dict):
    key = f"user:{user_id}"
    r.hset(key, mapping=user_data)
    r.expire(key, 3600)

# Read specific fields without deserializing entire object
def get_user_name(user_id: str) -> str:
    return r.hget(f"user:{user_id}", "name")
```

**Sorted set for leaderboards/rankings**:

```python
# Add score for a user
r.zadd("leaderboard:daily", {"user:123": 1500, "user:456": 2300})

# Get top 10
top_10 = r.zrevrange("leaderboard:daily", 0, 9, withscores=True)

# Get a user's rank
rank = r.zrevrank("leaderboard:daily", "user:123")
```

## Cache Invalidation Strategies

Cache invalidation is the hard part. Here are the practical strategies.

### TTL-Based Invalidation

The simplest approach: every cache entry expires after a fixed time. After expiration, the next read triggers a fresh database lookup.

```python
# User profile: changes infrequently, can tolerate 5 minutes of staleness
r.setex(f"user:{user_id}", 300, json.dumps(user_data))

# Product price: changes rarely, can tolerate 1 hour of staleness
r.setex(f"product:{product_id}:price", 3600, json.dumps(price_data))

# Session data: should expire for security
r.setex(f"session:{session_id}", 86400, json.dumps(session_data))
```

TTL is easy to implement but provides no consistency guarantee. Data can be stale for up to the TTL duration.

### Event-Driven Invalidation

When data changes, publish an invalidation event. Cache subscribers process the event and delete or update the cached entry.

```python
# On user update — publish invalidation event
def update_user(user_id: str, data: dict):
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # Publish to Redis Pub/Sub
    r.publish("cache:invalidate", json.dumps({
        "type": "user",
        "id": user_id,
    }))

# Cache invalidation subscriber (runs as a separate process)
def invalidation_listener():
    pubsub = r.pubsub()
    pubsub.subscribe("cache:invalidate")

    for message in pubsub.listen():
        if message["type"] == "message":
            event = json.loads(message["data"])
            if event["type"] == "user":
                r.delete(f"user:{event['id']}")
                logger.info(f"Invalidated cache for user {event['id']}")
```

Event-driven invalidation provides near-real-time consistency but adds complexity (message delivery guarantees, subscriber management).

### Versioned Keys

Append a version number or hash to cache keys. When data changes, increment the version. Old cached data becomes unreachable (and eventually evicted by LRU).

```python
# Write with version
version = db.query("SELECT version FROM users WHERE id=%s", user_id)
r.setex(f"user:{user_id}:v{version}", 3600, json.dumps(user_data))

# Read with current version
version = db.query("SELECT version FROM users WHERE id=%s", user_id)
cached = r.get(f"user:{user_id}:v{version}")
```

This requires a version lookup but guarantees you never read stale data. The trade-off is that the version lookup itself may need to be cached (and now you have a meta-caching problem).

## The Thundering Herd Problem

When a popular cache entry expires, hundreds of concurrent requests simultaneously experience a cache miss and all query the database for the same data. This spike can overwhelm the database.

![Thundering herd problem](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-thundering-herd.png)


### Visualization

```
Normal operation:
  1000 req/sec → Cache (hit) → Response   [DB load: 0]

Cache entry expires at T=0:
  T=0.001: Request 1 → Cache MISS → DB query
  T=0.002: Request 2 → Cache MISS → DB query
  T=0.003: Request 3 → Cache MISS → DB query
  ...
  T=0.050: Request 50 → Cache MISS → DB query   [DB load: 50 concurrent queries]
  T=0.100: Request 1 populates cache
  T=0.101: Request 101 → Cache HIT → Response   [DB load drops]
```

### Mitigation: Cache Stampede Protection (Locking)

Only one request fetches from the database on cache miss. Other requests wait for the cache to be repopulated.

```python
import time

def get_user_with_lock(user_id: str) -> dict:
    cache_key = f"user:{user_id}"
    lock_key = f"lock:{cache_key}"

    # Check cache
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    # Try to acquire lock
    lock_acquired = r.set(lock_key, "1", ex=10, nx=True)

    if lock_acquired:
        try:
            # We have the lock — fetch from DB and populate cache
            user = db.query("SELECT * FROM users WHERE id = %s", user_id)
            if user:
                r.setex(cache_key, 3600, json.dumps(user))
            return user
        finally:
            r.delete(lock_key)
    else:
        # Another request is fetching — wait and retry
        for _ in range(50):  # Wait up to 5 seconds
            time.sleep(0.1)
            cached = r.get(cache_key)
            if cached:
                return json.loads(cached)

        # Timeout — fall through to DB (safety valve)
        return db.query("SELECT * FROM users WHERE id = %s", user_id)
```

### Mitigation: Probabilistic Early Expiration

Instead of all entries expiring at exactly the same time, add random jitter to the TTL. Entries expire at slightly different times, spreading the database load.

```python
import random

def cache_with_jitter(key: str, value: str, base_ttl: int):
    # Add +/- 10% jitter to TTL
    jitter = int(base_ttl * 0.1)
    ttl = base_ttl + random.randint(-jitter, jitter)
    r.setex(key, ttl, value)

# Base TTL: 3600 seconds
# Actual TTL: 3240-3960 seconds (spread over 12 minutes)
```

A more sophisticated approach: proactively refresh the cache before it expires, using a probabilistic trigger.

```python
def get_with_early_refresh(key: str, base_ttl: int) -> dict:
    cached = r.get(key)
    remaining_ttl = r.ttl(key)

    if cached and remaining_ttl > 0:
        # Probabilistically refresh when TTL is getting low
        # Probability increases as TTL approaches 0
        refresh_probability = max(0, 1 - (remaining_ttl / base_ttl))
        if random.random() < refresh_probability * 0.1:  # Scale factor
            # Refresh in background (non-blocking)
            threading.Thread(
                target=refresh_cache, args=(key, base_ttl)
            ).start()
        return json.loads(cached)

    # Cache miss — fetch and populate
    return fetch_and_cache(key, base_ttl)
```

## Cache Warming

After a deploy, restart, or failover, the cache is empty. All requests hit the database until the cache is populated organically. For high-traffic systems, this cold start can overwhelm the database.

### Warming Strategies

**Preload on startup**: Before marking the server as healthy, preload the cache with frequently accessed data.

```python
def warm_cache():
    """Preload top 1000 users and hot content on startup."""
    # Top users by request frequency (from analytics)
    top_users = db.query(
        "SELECT id FROM users ORDER BY last_active DESC LIMIT 1000"
    )
    for user in top_users:
        user_data = db.query("SELECT * FROM users WHERE id = %s", user.id)
        r.setex(f"user:{user.id}", 3600, json.dumps(user_data))

    logger.info(f"Warmed cache with {len(top_users)} users")

# Call before registering with load balancer
warm_cache()
register_with_load_balancer()
```

**Shadow traffic**: Route a copy of production traffic to the new cache to warm it before it serves real traffic.

**Staggered rollout**: Deploy to one server at a time, letting each server warm its cache before moving to the next.

## When NOT to Cache

Caching is not always beneficial. Here are cases where it hurts.

**Write-heavy workloads**: If data changes more often than it is read, cache invalidation overhead exceeds the benefit. A cache that is invalidated on every write and read once provides zero benefit and adds latency (the invalidation step).

```
Read:Write ratio 100:1 → Cache helps (100 reads served from cache per invalidation)
Read:Write ratio   1:1 → Cache breaks even at best
Read:Write ratio   1:5 → Cache hurts (5 invalidations per read)
```

**Low hit rate**: If the data access pattern is uniform (no hot set), caching does not help. A cache with a 10% hit rate saves only 10% of database load while adding the complexity of a cache layer.

**Consistency-critical paths**: Payment processing, inventory management, and ledger updates must read the authoritative data source. Caching introduces staleness that is unacceptable for these use cases. You can still cache read-only views of this data (account balance display), but the write path must bypass the cache.

**Large, rarely-accessed objects**: Caching a 10 MB report that is accessed once per day wastes 10 MB of cache memory that could store 10,000 frequently-accessed 1 KB objects.

## Real Example: Caching User Profiles with Redis

Here is a complete, production-ready caching layer for user profile data.

```python
import redis
import json
import logging
import time
import random
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    id: str
    name: str
    email: str
    avatar_url: str
    bio: str
    follower_count: int
    following_count: int

class UserProfileCache:
    """Production caching layer for user profiles.

    Uses cache-aside pattern with:
    - TTL + jitter to prevent thundering herd
    - Distributed lock for cache stampede protection
    - Negative caching (cache misses to prevent DB hammering for deleted users)
    - Metrics for monitoring hit rate
    """

    BASE_TTL = 1800          # 30 minutes
    NEGATIVE_TTL = 300       # 5 minutes for "user not found" cache
    LOCK_TTL = 10            # 10 seconds lock timeout
    LOCK_WAIT_ATTEMPTS = 50  # 5 seconds total wait (50 * 0.1s)

    def __init__(self, redis_client: redis.Redis, db):
        self.r = redis_client
        self.db = db
        self.hits = 0
        self.misses = 0

    def get(self, user_id: str) -> Optional[UserProfile]:
        cache_key = f"user_profile:{user_id}"

        # Check cache
        cached = self.r.get(cache_key)
        if cached is not None:
            self.hits += 1
            data = json.loads(cached)
            if data is None:
                return None  # Negative cache entry
            return UserProfile(**data)

        self.misses += 1

        # Cache miss — acquire lock to prevent stampede
        return self._fetch_with_lock(user_id, cache_key)

    def invalidate(self, user_id: str):
        cache_key = f"user_profile:{user_id}"
        self.r.delete(cache_key)
        logger.debug(f"Invalidated cache for user {user_id}")

    def update(self, user_id: str, data: dict):
        # Update database first (source of truth)
        self.db.update_user(user_id, data)
        # Invalidate cache (not update — avoids race conditions)
        self.invalidate(user_id)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _fetch_with_lock(
        self, user_id: str, cache_key: str
    ) -> Optional[UserProfile]:
        lock_key = f"lock:{cache_key}"
        lock_acquired = self.r.set(lock_key, "1", ex=self.LOCK_TTL, nx=True)

        if lock_acquired:
            try:
                return self._fetch_and_cache(user_id, cache_key)
            finally:
                self.r.delete(lock_key)
        else:
            return self._wait_for_cache(user_id, cache_key)

    def _fetch_and_cache(
        self, user_id: str, cache_key: str
    ) -> Optional[UserProfile]:
        user = self.db.get_user(user_id)

        if user is None:
            # Negative caching — prevent repeated DB lookups for deleted users
            self.r.setex(cache_key, self.NEGATIVE_TTL, json.dumps(None))
            return None

        profile = UserProfile(
            id=user.id,
            name=user.name,
            email=user.email,
            avatar_url=user.avatar_url,
            bio=user.bio,
            follower_count=user.follower_count,
            following_count=user.following_count,
        )

        # Cache with jitter to prevent synchronized expiration
        ttl = self.BASE_TTL + random.randint(-180, 180)
        self.r.setex(cache_key, ttl, json.dumps(asdict(profile)))

        return profile

    def _wait_for_cache(
        self, user_id: str, cache_key: str
    ) -> Optional[UserProfile]:
        for _ in range(self.LOCK_WAIT_ATTEMPTS):
            time.sleep(0.1)
            cached = self.r.get(cache_key)
            if cached is not None:
                data = json.loads(cached)
                if data is None:
                    return None
                return UserProfile(**data)

        # Timeout — bypass lock and fetch directly (safety valve)
        logger.warning(f"Lock wait timeout for user {user_id}, fetching directly")
        return self._fetch_and_cache(user_id, cache_key)
```

This implementation handles the common edge cases:
- **Thundering herd**: Distributed lock ensures only one request fetches from DB
- **Negative caching**: Prevents hammering the DB for non-existent users
- **TTL jitter**: Prevents synchronized cache expiration
- **Monitoring**: Hit rate metric for operational visibility
- **Safety valve**: Direct DB fetch on lock timeout prevents deadlock

## What's Next

Caching handles the read path. But what about the write path when you need to decouple producers from consumers, smooth out traffic spikes, and build event-driven architectures? The next article covers message queues — Kafka, RabbitMQ, delivery guarantees, and the patterns that make asynchronous systems reliable.
