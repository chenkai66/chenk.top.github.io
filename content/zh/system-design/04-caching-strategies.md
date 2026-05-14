---
title: "系统设计（四）：缓存——在哪里缓存、淘汰什么，以及缓存何时反而有害"
date: 2025-07-17 09:00:00
tags:
  - System Design
  - Caching
  - Redis
  - Performance
categories: System Design
series: system-design
lang: zh
description: "深入剖析全栈各层的缓存策略——从 CDN 到数据库缓冲池——涵盖 cache-aside、write-through、write-behind 模式，淘汰策略、惊群效应（thundering herd）缓解方案，以及实用的 Redis 配置。"
disableNunjucks: true
series_order: 4
translationKey: "system-design-4"
---
计算机科学中有个老笑话：最难的两个问题，是缓存失效（cache invalidation）、命名（naming things），以及 off-by-one 错误。这个笑话之所以成立，正是因为缓存失效确实极难处理。但与此同时，缓存又是提升系统性能最有效的单一技术——一个部署得当的缓存，可将延迟降低 100 倍，减少 90% 的数据库负载，并每月节省数千美元的基础设施成本。

关键在于：知道在哪里缓存、采用何种模式，以及——至关重要的是——何时缓存反而会让系统变得更糟，而非更好。

---

## 缓存为何有效

缓存利用了大多数系统的根本特性：访问模式并非均匀分布。一小部分数据被访问的频率远高于其余数据。

![Web 应用中的缓存层](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-cache-layers.png)

以社交平台为例：在任意时刻，仅有极小比例的帖子正在“热榜”上，被数百万用户浏览；而其余 99% 的帖子极少被查看。若你将这前 1% 的帖子缓存在内存中，就能用零数据库查询承载 80% 的读请求流量。

这些收益会逐层放大：

**延迟降低**：Redis 返回缓存值仅需 0.1–0.5 ms；数据库查询需 5–50 ms；跨服务 API 调用则需 10–100 ms。缓存直接消除了最昂贵的操作。

**吞吐量放大**：单个 Redis 实例每秒可处理 10 万+ 操作；PostgreSQL 实例通常为每秒 5,000–20,000 查询。缓存显著提升了系统的有效吞吐能力。

**成本节约**：一台 Redis 实例可替代 10–20 台数据库只读副本。按云厂商定价，这每月可节省 $10,000–$50,000。

## 缓存层级

现代系统在每一层都部署了缓存。理解各层职责，可避免你解决错误的问题。

### 客户端缓存（Client-Side Cache）

浏览器依据 `Cache-Control` HTTP 响应头缓存响应内容。

```text
Cache-Control: public, max-age=31536000    # CDN + 浏览器缓存 1 年
Cache-Control: private, max-age=3600       # 仅浏览器缓存，1 小时
Cache-Control: no-cache                    # 每次均向服务器重新验证
Cache-Control: no-store                    # 禁止任何缓存
```

`ETag` 和 `If-None-Match` 头支持条件请求：

```json
# 首次请求
GET /api/user/123
Response:
  200 OK
  ETag: "abc123"
  { "name": "Alice", "email": "alice@example.com" }

# 后续请求
GET /api/user/123
If-None-Match: "abc123"
Response:
  304 Not Modified    # 无响应体，复用本地缓存
```

客户端缓存是零成本的性能提升，应优先配置。

### CDN 缓存

如前文所述，CDN 在全球边缘节点缓存静态资源。对于 API 响应，CDN 缓存虽可行，但需谨慎设置 `Cache-Control` 和 `Vary` 头，避免将某个用户的响应错误返回给其他用户。

```bash
# CDN 可缓存（公开数据）
Cache-Control: public, max-age=300, s-maxage=600
Vary: Accept-Encoding

# CDN 不可缓存（用户私有数据）
Cache-Control: private, max-age=60
```

`s-maxage` 会覆盖 `max-age` 对共享缓存（如 CDN）的限制，允许你在边缘缓存比浏览器更久。

### 应用层缓存（Application Cache）

此处运行着 Redis、Memcached，以及进程内缓存（如 Caffeine 或 Guava）。应用层显式管理缓存内容、失效时机与刷新逻辑。

### 数据库缓存（Database Cache）

数据库自身也内置缓存机制：

**PostgreSQL 共享缓冲区（shared buffers）**：将频繁访问的表和索引页缓存在内存中。默认大小为 128 MB；生产环境通常设为可用内存的 25%。

```bash
# postgresql.conf
shared_buffers = 8GB          # 32GB 内存的 25%
effective_cache_size = 24GB   # 内存的 75%（OS 缓存 + PG 缓存总和）
```

**MySQL InnoDB 缓冲池（Buffer Pool）**：缓存表数据与索引。在专用数据库服务器上，建议设为可用内存的 70–80%。

```bash
# my.cnf
innodb_buffer_pool_size = 24G   # 32GB 内存的 75%
innodb_buffer_pool_instances = 8  # 减少锁竞争
```

**查询缓存（Query Cache）**（MySQL，8.0 中已弃用）：缓存 `SELECT` 查询的结果集。只要查询所涉任一表发生写入，该缓存即失效。对写密集型负载而言，弊大于利——每次写入都会使该表所有缓存查询失效，引发严重锁争用。

## 缓存模式（Caching Patterns）

![缓存失效问题：陈旧数据和幽灵读取](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/04-cache-invalidation-problem-stale-data-ghost-haunting-fresh-d.jpg)

将缓存与数据库集成，有四种基础模式。每种模式具有一致性保证与失败行为上的差异。

![缓存模式对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-caching-patterns.png)

### Cache-Aside（懒加载）

应用层显式管理缓存。读操作先查缓存；缓存未命中时，再从数据库读取并填充缓存。

```python
import redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_user(user_id: str) -> dict:
    # 步骤 1：检查缓存
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # 步骤 2：缓存未命中 —— 从数据库读取
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user is None:
        return None

    # 步骤 3：带 TTL 填充缓存
    r.setex(f"user:{user_id}", 3600, json.dumps(user))

    return user

def update_user(user_id: str, data: dict):
    # 更新数据库
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # 失效缓存（不更新——删除更安全）
    r.delete(f"user:{user_id}")
```

**为何删除而非更新缓存？**  
考虑两个并发请求同时更新同一用户。若两者均尝试更新缓存，则可能因竞态条件导致缓存中残留过期数据。删除缓存可强制下次读取时回源数据库——而数据库始终是权威数据源。

**优势**：
- 实现与理解简单
- 缓存仅包含实际被请求的数据（无空间浪费）
- 缓存故障不会导致服务中断（可自动降级至数据库）

**劣势**：
- 每个 key 的首次请求必击数据库（冷启动）
- 数据库更新与缓存失效之间存在短暂的时间差，可能导致读取到过期数据
- 应用代码充斥缓存逻辑，侵入性强

### Write-Through（直写）

每次写入均同步写入缓存与数据库。缓存始终与数据库保持一致。

```python
def update_user_write_through(user_id: str, data: dict):
    # 写入数据库
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # 写入缓存（概念上同属一个事务）
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))

def get_user_write_through(user_id: str) -> dict:
    # 总是从缓存读取
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # 仅冷启动或缓存淘汰时触发
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user:
        r.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**优势**：
- 缓存与数据库强一致（无脏读）
- 读路径极简（恒命中缓存）

**劣势**：
- 写延迟升高（必须双写）
- 缓存了可能永不被读取的数据（空间浪费）
- 缓存与 DB 写入并非真正原子（中间失败将导致不一致）

### Write-Behind（后写 / 写回）

写入操作先写入缓存，再由缓存异步批量同步到数据库。

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
                    # 失败后重加回 dirty 集合以重试
                    with self.lock:
                        self.dirty[key] = value
                    logger.error(f"Flush failed for {key}: {e}")

    def _start_flusher(self):
        t = threading.Thread(target=self._flush, daemon=True)
        t.start()
```

**优势**：
- 写延迟极低（仅写缓存）
- 批量写入数据库，降低 DB 负载
- 可吸收写入峰值

**劣势**：
- 数据丢失风险：若缓存进程崩溃且尚未将数据刷写到磁盘，这部分数据将永久丢失
- 复杂度高：需异步刷盘、重试逻辑、顺序保障
- 调试困难（缓存中数据尚未落库）

适用于高吞吐写入场景，且可容忍少量数据丢失（如分析计数器、浏览量、活动日志）。

### Read-Through（读透）

缓存位于数据库前方，透明处理读请求。缓存未命中时，由缓存自身（而非应用）从数据库加载数据。

此模式通常由缓存库或框架实现，而非手写应用代码。功能上类似 cache-aside，但将加载逻辑封装于缓存层内部。

```python
from cachetools import TTLCache

# Python 示例：使用 cachetools + 加载函数
cache = TTLCache(maxsize=10000, ttl=3600)

def get_user_read_through(user_id: str) -> dict:
    if user_id in cache:
        return cache[user_id]

    # 缓存自身在未命中时从 DB 加载
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user:
        cache[user_id] = user
    return user
```

## 缓存淘汰策略（Cache Eviction Policies）

当缓存达到容量上限时，必须决定淘汰哪些条目。淘汰策略的选择对缓存命中率影响巨大。

![缓存淘汰策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-eviction-policies.png)

### 策略对比

| 策略 | 工作原理 | 命中率 | 开销 | 最适用场景 |
|------|----------|--------|------|------------|
| LRU（最近最少使用） | 淘汰最久未被访问的条目 | 良好 | O(1)，需双向链表 + 哈希表 | 通用场景，最常用 |
| LFU（最不常使用） | 淘汰访问频次最低的条目 | 对偏斜分布更优 | 较高（需维护计数器） | 热/冷数据稳定、热度分布不均 |
| FIFO（先进先出） | 淘汰最老的条目 | 较差 | O(1) | 简单场景，访问时间无关紧要 |
| Random（随机） | 随机淘汰一条 | 意外良好 | O(1) | 访问模式均匀时 |
| TTL-based（基于过期时间） | 条目超时后自动淘汰 | N/A（非容量驱动） | O(1)/条目 | 时间敏感型数据 |

### LRU 实现

LRU 是最广泛使用的淘汰策略。以下是一个简洁实现：

![LRU 缓存淘汰动画](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/sysdesign-04-lru-cache.gif)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        # 移至末尾（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 从头部移除（最久未用）
            self.cache.popitem(last=False)
```

Redis 使用一种近似 LRU 算法。为避免精确追踪 LRU 顺序带来的巨大内存开销，它采样一组可配置数量的 key，并从中淘汰最近最少使用的那个。默认采样数为 5，其效果已非常接近真实 LRU。

## Redis 作为缓存

Redis 是应用层缓存的事实标准。以下是面向缓存场景的实用配置。

![Redis 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-redis-architecture.png)

### Redis 缓存场景配置

```bash
# redis.conf for caching use case

# 内存上限
maxmemory 8gb

# 内存满时的淘汰策略
# allkeys-lru: 对任意 key 使用近似 LRU 淘汰
# volatile-lru: 仅淘汰设置了 TTL 的 key
# allkeys-lfu: 对任意 key 使用近似 LFU 淘汰
# noeviction: 内存满时返回错误
maxmemory-policy allkeys-lru

# LRU 近似采样数（越高越准，CPU 开销越大）
maxmemory-samples 10

# 持久化：纯缓存场景下禁用（更快，无磁盘 I/O）
save ""
appendonly no

# 连接数限制
maxclients 10000

# TCP keepalive
tcp-keepalive 300

# 空闲连接超时（0 表示永不超时）
timeout 300
```

### 常见 Redis 缓存模式

**简单键值缓存**：

```python
import redis
import json

r = redis.Redis(host="cache.internal", port=6379, decode_responses=True)

# 缓存用户资料 1 小时
def cache_user(user_id: str, user_data: dict):
    r.setex(f"user:{user_id}", 3600, json.dumps(user_data))

# 条件式缓存（仅当 key 不存在时设置，防止覆盖）
def cache_user_if_missing(user_id: str, user_data: dict):
    r.set(f"user:{user_id}", json.dumps(user_data), ex=3600, nx=True)
```

**哈希结构缓存（对象更省内存）**：

```python
# 将用户存为 Redis Hash
def cache_user_hash(user_id: str, user_data: dict):
    key = f"user:{user_id}"
    r.hset(key, mapping=user_data)
    r.expire(key, 3600)

# 无需反序列化整个对象即可读取特定字段
def get_user_name(user_id: str) -> str:
    return r.hget(f"user:{user_id}", "name")
```

**有序集合用于排行榜/排名**：

```python
# 为用户添加分数
r.zadd("leaderboard:daily", {"user:123": 1500, "user:456": 2300})

# 获取 Top 10
top_10 = r.zrevrange("leaderboard:daily", 0, 9, withscores=True)

# 获取某用户排名
rank = r.zrevrank("leaderboard:daily", "user:123")
```

## 缓存失效策略（Cache Invalidation Strategies）

缓存失效才是真正的难点。以下是几种实用策略。

### 基于 TTL 的失效

最简单方式：每个缓存条目固定过期时间。过期后，下次读取将触发全新数据库查询。

```python
# 用户资料：变更不频繁，可容忍 5 分钟陈旧
r.setex(f"user:{user_id}", 300, json.dumps(user_data))

# 商品价格：极少变动，可容忍 1 小时陈旧
r.setex(f"product:{product_id}:price", 3600, json.dumps(price_data))

# 会话数据：出于安全必须过期
r.setex(f"session:{session_id}", 86400, json.dumps(session_data))
```

TTL 实现简单，但无法保证数据一致性。数据最多可陈旧达 TTL 时长。

### 事件驱动失效（Event-Driven Invalidation）

数据变更时发布失效事件，缓存订阅者接收并执行删除或更新。

```python
# 用户更新时 —— 发布失效事件
def update_user(user_id: str, data: dict):
    db.execute("UPDATE users SET name=%s WHERE id=%s", data["name"], user_id)

    # 发布到 Redis Pub/Sub
    r.publish("cache:invalidate", json.dumps({
        "type": "user",
        "id": user_id,
    }))

# 缓存失效监听器（独立进程运行）
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

事件驱动失效提供近实时一致性，但增加了复杂度（消息投递保障、订阅者管理）。

### 版本化 Key（Versioned Keys）

在缓存 key 中附加版本号或哈希值。数据变更时升级版本号，旧缓存数据自然不可达（最终被 LRU 淘汰）。

```python
# 写入时带版本
version = db.query("SELECT version FROM users WHERE id=%s", user_id)
r.setex(f"user:{user_id}:v{version}", 3600, json.dumps(user_data))

# 读取时使用当前版本
version = db.query("SELECT version FROM users WHERE id=%s", user_id)
cached = r.get(f"user:{user_id}:v{version}")
```

这种方式需要额外查询一次版本号，但能确保永远不会读取到过期数据。代价是版本查询本身也可能需要缓存（于是又引入了元缓存问题）。

## 惊群效应（Thundering Herd Problem）

![缓存层像盾牌一样保护数据库免受流量冲击](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/04-caching-layers-stacked-shields-protecting-database-from-traf.jpg)

当热门缓存条目过期时，数百个并发请求同时遭遇缓存未命中，并全部涌向数据库查询相同数据。这一突发流量可能压垮数据库。

![惊群问题](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/04-thundering-herd.png)

### 可视化示意

```text
正常情况：
  1000 req/sec → 缓存（命中）→ 响应   [DB 负载：0]

缓存条目在 T=0 过期：
  T=0.001: 请求 1 → 缓存未命中 → DB 查询
  T=0.002: 请求 2 → 缓存未命中 → DB 查询
  T=0.003: 请求 3 → 缓存未命中 → DB 查询
  ...
  T=0.050: 请求 50 → 缓存未命中 → DB 查询   [DB 负载：50 并发查询]
  T=0.100: 请求 1 完成并填充缓存
  T=0.101: 请求 101 → 缓存命中 → 响应   [DB 负载骤降]
```

### 缓解方案：缓存雪崩防护（Cache Stampede Protection，加锁）

仅一个请求在缓存未命中时访问数据库；其余请求等待缓存被重新填充。

```python
import time

def get_user_with_lock(user_id: str) -> dict:
    cache_key = f"user:{user_id}"
    lock_key = f"lock:{cache_key}"

    # 检查缓存
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    # 尝试获取锁
    lock_acquired = r.set(lock_key, "1", ex=10, nx=True)

    if lock_acquired:
        try:
            # 持有锁：从 DB 查询并填充缓存
            user = db.query("SELECT * FROM users WHERE id = %s", user_id)
            if user:
                r.setex(cache_key, 3600, json.dumps(user))
            return user
        finally:
            r.delete(lock_key)
    else:
        # 其他请求正在获取 —— 等待并重试
        for _ in range(50):  # 最多等待 5 秒
            time.sleep(0.1)
            cached = r.get(cache_key)
            if cached:
                return json.loads(cached)

        # 超时 —— 直接降级至 DB（安全阀）
        return db.query("SELECT * FROM users WHERE id = %s", user_id)
```

### 缓解方案：概率性提前过期（Probabilistic Early Expiration）

不令所有条目在同一刻过期，而是为 TTL 添加随机抖动（jitter），使过期时间分散，从而平滑数据库负载。

```python
import random

def cache_with_jitter(key: str, value: str, base_ttl: int):
    # 为 TTL 添加 ±10% 抖动
    jitter = int(base_ttl * 0.1)
    ttl = base_ttl + random.randint(-jitter, jitter)
    r.setex(key, ttl, value)

# 基础 TTL：3600 秒
# 实际 TTL：3240–3960 秒（跨度 12 分钟）
```

更高级做法：在缓存过期前主动刷新，使用概率性触发器。

```python
def get_with_early_refresh(key: str, base_ttl: int) -> dict:
    cached = r.get(key)
    remaining_ttl = r.ttl(key)

    if cached and remaining_ttl > 0:
        # 当 TTL 即将耗尽时，按概率刷新
        # 概率随剩余 TTL 减小而增大
        refresh_probability = max(0, 1 - (remaining_ttl / base_ttl))
        if random.random() < refresh_probability * 0.1:  # 缩放因子
            # 后台非阻塞刷新
            threading.Thread(
                target=refresh_cache, args=(key, base_ttl)
            ).start()
        return json.loads(cached)

    # 缓存未命中 —— 获取并填充
    return fetch_and_cache(key, base_ttl)
```

## 缓存预热（Cache Warming）

部署、重启或故障转移后，缓存为空。所有请求将击穿缓存直达数据库，直至缓存被自然填充。对高流量系统，这种冷启动可能瞬间压垮数据库。

### 预热策略

**启动时预加载**：在标记服务健康前，预先加载高频访问数据。

```python
def warm_cache():
    """启动时预加载 Top 1000 用户及热门内容。"""
    # 根据分析数据获取访问最频繁的用户
    top_users = db.query(
        "SELECT id FROM users ORDER BY last_active DESC LIMIT 1000"
    )
    for user in top_users:
        user_data = db.query("SELECT * FROM users WHERE id = %s", user.id)
        r.setex(f"user:{user.id}", 3600, json.dumps(user_data))

    logger.info(f"Warmed cache with {len(top_users)} users")

# 在注册至负载均衡器前调用
warm_cache()
register_with_load_balancer()
```

**影子流量（Shadow traffic）**：将一份生产流量复制到新缓存，使其在正式服务前完成预热。

**分阶段灰度发布（Staggered rollout）**：逐台部署服务器，让每台服务器在继续下一台前完成自身缓存预热。

## 何时不应使用缓存？

缓存并非总是有益。以下情形中，它反而有害。

**写密集型工作负载（Write-heavy workloads）**：若数据变更频率高于读取频率，缓存失效开销将超过收益。一个每次写入即失效、每次读取仅命中一次的缓存，毫无价值，还额外增加延迟（失效步骤）。

```text
读:写比例 100:1 → 缓存有益（每次失效服务 100 次读）
读:写比例   1:1 → 缓存最多打平
读:写比例   1:5 → 缓存有害（每次读伴随 5 次失效）
```

**低命中率（Low hit rate）**：若数据访问模式均匀（无热点），缓存无效。命中率仅 10% 的缓存，仅节省 10% 的数据库负载，却引入了整套缓存层的复杂性。

**强一致性关键路径（Consistency-critical paths）**：支付处理、库存管理、账本更新等必须读取权威数据源。缓存引入的陈旧性对此类场景不可接受。你仍可缓存这些数据的只读视图（如账户余额展示），但写入路径必须绕过缓存。

**大体积、低频访问对象（Large, rarely-accessed objects）**：缓存一份每日仅访问一次的 10 MB 报告，将浪费 10 MB 缓存空间——这些空间本可存储 10,000 个高频访问的 1 KB 对象。

## 实战案例：使用 Redis 缓存用户资料

以下是一个完整、可用于生产的用户资料缓存层。

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
    """用户资料的生产级缓存层。

    基于 cache-aside 模式，具备：
    - TTL + 抖动（防惊群）
    - 分布式锁（防缓存雪崩）
    - 负向缓存（negative caching，防对已删用户反复查 DB）
    - 监控指标（命中率）
    """

    BASE_TTL = 1800          # 30 分钟
    NEGATIVE_TTL = 300       # “用户不存在”缓存 5 分钟
    LOCK_TTL = 10            # 锁超时 10 秒
    LOCK_WAIT_ATTEMPTS = 50  # 总等待时间 5 秒（50 × 0.1s）

    def __init__(self, redis_client: redis.Redis, db):
        self.r = redis_client
        self.db = db
        self.hits = 0
        self.misses = 0

    def get(self, user_id: str) -> Optional[UserProfile]:
        cache_key = f"user_profile:{user_id}"

        # 检查缓存
        cached = self.r.get(cache_key)
        if cached is not None:
            self.hits += 1
            data = json.loads(cached)
            if data is None:
                return None  # 负向缓存条目
            return UserProfile(**data)

        self.misses += 1

        # 缓存未命中 —— 加锁防雪崩
        return self._fetch_with_lock(user_id, cache_key)

    def invalidate(self, user_id: str):
        cache_key = f"user_profile:{user_id}"
        self.r.delete(cache_key)
        logger.debug(f"Invalidated cache for user {user_id}")

    def update(self, user_id: str, data: dict):
        # 先更新数据库（唯一真相源）
        self.db.update_user(user_id, data)
        # 失效缓存（不更新——避免竞态）
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
            # 负向缓存：防止对已删用户反复查 DB
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

        # 带抖动缓存，防同步过期
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

        # 超时 —— 绕过锁直接获取（安全阀）
        logger.warning(f"Lock wait timeout for user {user_id}, fetching directly")
        return self._fetch_and_cache(user_id, cache_key)
```

该实现处理了常见边界情况：
- **惊群效应**：分布式锁确保仅一个请求访问数据库  
- **负向缓存**：防止对不存在用户反复查询数据库  
- **TTL 抖动**：防止缓存同步过期  
- **监控**：命中率指标用于运维可观测性  
- **安全阀**：锁等待超时后直接查库，避免死锁  

## 接下来

缓存解决了读路径优化。但当你需要解耦生产者与消费者、削峰填谷、构建事件驱动架构时，写路径该如何设计？下一篇文章将详解消息队列——Kafka、RabbitMQ、投递保障机制，以及构建可靠异步系统的核心模式。
