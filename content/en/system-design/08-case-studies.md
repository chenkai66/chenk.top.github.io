---
title: "System Design (8): Case Studies — URL Shortener, Chat System, News Feed"
date: 2025-07-27 09:00:00
tags:
  - System Design
  - Case Studies
  - Distributed Systems
  - Architecture
categories: System Design
series: system-design
lang: en
description: "Three complete system design walkthroughs — a URL shortener, a real-time chat system, and a news feed — each following the full process from requirements and estimation through high-level design, deep dives, and scaling strategies."
disableNunjucks: true
series_order: 8
translationKey: "system-design-8"
---

The best way to learn system design is to practice it. Reading about individual components — caching, queues, load balancers — builds your vocabulary, but designing a complete system is where you learn to compose those components into something that actually works.

This article walks through three classic system design problems end to end. Each follows the framework from the first article in this series: clarify requirements, estimate scale, design the architecture, deep dive into critical components, and identify bottlenecks.

---

## Case Study 1: URL Shortener

A URL shortener takes a long URL and produces a short alias (e.g., `https://short.ly/abc123`) that redirects to the original. It sounds trivially simple, but at scale it touches hashing, distributed storage, caching, and analytics.

![URL shortener design](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/08-url-shortener.png)


### Requirements

**Functional**:
- Given a long URL, generate a short URL
- Given a short URL, redirect to the original long URL
- Users can optionally specify a custom alias
- Short URLs expire after a configurable period (default: 5 years)
- Track click analytics (count, referrer, geography)

**Non-functional**:
- 100 million new URLs per day
- Read:write ratio of 100:1 (10 billion redirects per day)
- Redirect latency under 50ms (p99)
- Availability: 99.99% (this is critical infrastructure for anyone using the short URLs)
- Short URLs should be as short as possible

### Estimation

**Write QPS (URL creation)**:
```text
100M URLs/day ÷ 86,400 sec/day ≈ 1,160 writes/sec
Peak (3x): ~3,500 writes/sec
```

**Read QPS (redirects)**:
```text
10B redirects/day ÷ 86,400 sec/day ≈ 115,000 reads/sec
Peak (3x): ~350,000 reads/sec
```

This is an extremely read-heavy system. Caching will be essential.

**Storage**:
```text
Per URL record:
  Short code: 7 bytes
  Long URL: 500 bytes (average)
  User ID: 8 bytes
  Created timestamp: 8 bytes
  Expiration timestamp: 8 bytes
  Total: ~530 bytes

Daily: 100M × 530 bytes = 53 GB/day
Yearly: 53 GB × 365 = 19.3 TB/year
5-year retention: ~100 TB
```

**Cache memory** (using the 80/20 rule — 20% of URLs handle 80% of traffic):
```text
Daily unique URLs accessed: ~1B (estimate)
Hot set (20%): 200M URLs
Cache per entry: 530 bytes
Cache memory: 200M × 530 bytes ≈ 106 GB
```

106 GB is manageable across a Redis cluster (e.g., 6 nodes with 32 GB each).

### Short URL Generation

The core design challenge: how to generate a unique, short code for each URL.

**Approach 1: Base62 encoding of an auto-increment ID**

Use a distributed ID generator (like Twitter's Snowflake) to generate a unique 64-bit integer, then encode it in base62 (a-z, A-Z, 0-9).

```python
import string

ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits  # 62 chars

def encode_base62(num: int) -> str:
    if num == 0:
        return ALPHABET[0]
    result = []
    while num > 0:
        result.append(ALPHABET[num % 62])
        num //= 62
    return "".join(reversed(result))

def decode_base62(s: str) -> int:
    num = 0
    for char in s:
        num = num * 62 + ALPHABET.index(char)
    return num

# Examples:
# encode_base62(1000000) → "4C92"     (4 characters)
# encode_base62(1000000000) → "15FTGg" (6 characters)
# encode_base62(3500000000000) → "zzzzzz" (6 characters, max 6-char value)
```

With 7 characters of base62, we can represent 62^7 = 3.5 trillion unique URLs, which is more than enough for decades.

**Approach 2: Hash and truncate**

Hash the long URL with MD5 or SHA-256, then take the first 7 characters of the base62-encoded hash.

```python
import hashlib

def generate_short_code(long_url: str) -> str:
    hash_bytes = hashlib.md5(long_url.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    code = encode_base62(hash_int)[:7]
    return code
```

**Problem**: Collisions. Two different URLs can produce the same 7-character code. You must check for collisions and append a counter or use a different hash seed if one occurs.

**Approach 3: Pre-generated key pool**

Pre-generate a large pool of unique short codes in a separate service. When a new URL is created, grab the next available code from the pool.

```python
class KeyGenerationService:
    """Pre-generates unique short codes for the URL shortener."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.pool_key = "available_codes"

    def generate_batch(self, batch_size: int = 100000):
        """Generate a batch of unique codes and add to the pool."""
        codes = set()
        while len(codes) < batch_size:
            code = encode_base62(random.randint(0, 62**7 - 1))
            code = code.ljust(7, "0")  # Pad to 7 chars
            codes.add(code)

        # Add to Redis set (automatically deduplicates)
        pipeline = self.redis.pipeline()
        for code in codes:
            pipeline.sadd(self.pool_key, code)
        pipeline.execute()

    def get_code(self) -> str:
        """Pop a code from the pool. Thread-safe and atomic."""
        code = self.redis.spop(self.pool_key)
        if code is None:
            raise RuntimeError("Code pool exhausted — generate more codes")
        return code.decode()
```

For this design, I will use Approach 1 (base62 encoding of a distributed ID) because it is simple, collision-free, and produces predictably short codes.

### High-Level Architecture

Components:
1. **API servers** (stateless) — handle create and redirect requests
2. **Distributed ID generator** — produces unique IDs for new URLs
3. **Database** — stores URL mappings (short code → long URL)
4. **Redis cache** — caches hot URL mappings for fast redirects
5. **Analytics pipeline** — records click events for analytics

Data flow for URL creation:
```text
Client → Load Balancer → API Server
  → Generate unique ID (Snowflake)
  → Encode as base62 short code
  → Store mapping in database
  → Return short URL to client
```

Data flow for redirect:
```text
Client → Load Balancer → API Server
  → Look up short code in Redis cache
  → Cache hit: redirect immediately
  → Cache miss: look up in database, populate cache, redirect
  → Async: record click event to Kafka for analytics
```

### Deep Dive: Redirect Logic

```python
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse
import redis
import json

app = FastAPI()
cache = redis.Redis(host="cache.internal", port=6379, decode_responses=True)

@app.get("/{short_code}")
async def redirect(short_code: str):
    # Step 1: Check cache
    long_url = cache.get(f"url:{short_code}")

    if long_url is None:
        # Step 2: Cache miss — check database
        record = await db.fetch_one(
            "SELECT long_url, expires_at FROM urls WHERE short_code = $1",
            short_code,
        )

        if record is None:
            raise HTTPException(status_code=404, detail="Short URL not found")

        if record["expires_at"] and record["expires_at"] < datetime.utcnow():
            raise HTTPException(status_code=410, detail="Short URL has expired")

        long_url = record["long_url"]

        # Populate cache (TTL: 24 hours)
        cache.setex(f"url:{short_code}", 86400, long_url)

    # Step 3: Record analytics event (async, non-blocking)
    await kafka_producer.send("click-events", {
        "short_code": short_code,
        "timestamp": datetime.utcnow().isoformat(),
        "referrer": request.headers.get("referer"),
        "user_agent": request.headers.get("user-agent"),
        "ip": request.client.host,
    })

    # Step 4: Redirect
    # 301 (permanent) is more cache-friendly but hides analytics
    # 302 (temporary) forces the browser to always hit our server (better for analytics)
    return RedirectResponse(url=long_url, status_code=302)
```

### 301 vs 302 Redirect

This is a meaningful trade-off:

| Redirect | Behavior | Analytics | CDN Caching |
|----------|----------|-----------|-------------|
| 301 (Permanent) | Browser caches, never hits server again | Undercounts (misses cached redirects) | CDN caches aggressively |
| 302 (Temporary) | Browser hits server every time | Accurate (every click recorded) | CDN may or may not cache |

Most URL shorteners use 302 because analytics is a core feature. Some offer both: 302 by default, 301 for performance-critical use cases.

### Scaling Strategy

**Database partitioning**: Hash the short code to determine the partition. This distributes writes evenly and allows lookups without scanning.

```text
Partition 0: short codes starting with [0-9]
Partition 1: short codes starting with [a-m]
Partition 2: short codes starting with [n-z]
Partition 3: short codes starting with [A-M]
Partition 4: short codes starting with [N-Z]
```

**Cache hot URLs**: The top 1% of URLs receive 90%+ of traffic. A Redis cluster caching these hot URLs handles the vast majority of redirects without touching the database.

**Analytics pipeline**: Click events go to Kafka, not directly to the database. A Flink job aggregates clicks per minute/hour/day and writes to a time-series database. This decouples the real-time redirect path from the analytics path.

---

## Case Study 2: Real-Time Chat System

A chat application requires real-time bidirectional communication, persistent message storage, presence awareness, and efficient fan-out for group messages.

![Real-time chat system](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/08-chat-system.png)


### Requirements

**Functional**:
- 1:1 messaging between users
- Group chat (up to 500 members)
- Online/offline status (presence)
- Message history (persistent, searchable)
- Sent/delivered/read receipts
- Push notifications for offline users
- Support text, images, and file attachments

**Non-functional**:
- 50 million daily active users
- Each user sends 40 messages per day
- Average group size: 10 members
- 30% of messages are group messages
- Message delivery latency under 200ms (p95)
- Availability: 99.9%
- Message ordering guaranteed within a conversation

### Estimation

**Message volume**:
```text
50M DAU × 40 messages/user/day = 2B messages/day
2B ÷ 86,400 = ~23,000 messages/sec
Peak (3x): ~70,000 messages/sec
```

**Connection count**:
```text
50M DAU, assume 30% are connected simultaneously = 15M concurrent WebSocket connections
Each connection: ~10 KB memory overhead
Total memory for connections: 15M × 10 KB = 150 GB
```

150 GB of connection state requires multiple servers. If each server handles 500K connections, you need ~30 connection servers.

**Storage**:
```text
Average message size: 200 bytes (text) + 100 bytes (metadata) = 300 bytes
Daily: 2B × 300 bytes = 600 GB/day
Yearly: 600 GB × 365 = 219 TB/year
With 3x replication: ~660 TB/year
```

For media attachments, object storage (S3) with CDN is appropriate. Text messages go to a database.

### High-Level Architecture

Components:
1. **WebSocket Gateway** — maintains persistent connections with clients
2. **Chat Service** — handles message routing and business logic
3. **Message Store** — persistent storage for message history
4. **Presence Service** — tracks online/offline status
5. **Push Notification Service** — sends push notifications to offline users
6. **Media Service** — handles file uploads and serving (backed by object storage + CDN)
7. **Kafka** — decouples message ingestion from storage and delivery

### WebSocket Connection Management

The WebSocket gateway maintains a mapping from user ID to connection.

```python
# Connection manager (runs on each WebSocket server)
import asyncio
import websockets
from collections import defaultdict

class ConnectionManager:
    def __init__(self, server_id: str, redis_client):
        self.server_id = server_id
        self.connections = {}  # user_id → websocket
        self.redis = redis_client

    async def connect(self, user_id: str, websocket):
        self.connections[user_id] = websocket
        # Register in Redis: user_id → server_id
        self.redis.hset("user_connections", user_id, self.server_id)
        # Publish presence event
        self.redis.publish("presence", json.dumps({
            "user_id": user_id, "status": "online"
        }))

    async def disconnect(self, user_id: str):
        self.connections.pop(user_id, None)
        self.redis.hdel("user_connections", user_id)
        self.redis.publish("presence", json.dumps({
            "user_id": user_id, "status": "offline"
        }))

    async def send_to_user(self, user_id: str, message: dict):
        ws = self.connections.get(user_id)
        if ws:
            await ws.send(json.dumps(message))
            return True
        return False

    def find_server(self, user_id: str) -> str:
        """Find which server a user is connected to."""
        return self.redis.hget("user_connections", user_id)
```

### Message Routing

When User A sends a message to User B:

1. User A's WebSocket server receives the message
2. Message is published to Kafka (for persistence and ordering)
3. Chat Service consumes from Kafka, looks up User B's connection server in Redis
4. If User B is on the same server: deliver directly via WebSocket
5. If User B is on a different server: route via inter-server communication (Redis Pub/Sub or internal gRPC)
6. If User B is offline: store message and send push notification

```python
class ChatService:
    async def handle_message(self, message: dict):
        sender_id = message["sender_id"]
        recipient_id = message["recipient_id"]
        conversation_id = message["conversation_id"]

        # Generate message ID and timestamp (server-side for consistency)
        message["message_id"] = str(uuid.uuid4())
        message["server_timestamp"] = datetime.utcnow().isoformat()

        # Persist to Kafka (for ordering and durability)
        await kafka_producer.send(
            topic=f"chat-messages",
            key=conversation_id,  # Same conversation → same partition → ordered
            value=message,
        )

        # Send acknowledgment to sender
        await self.connection_manager.send_to_user(sender_id, {
            "type": "ack",
            "message_id": message["message_id"],
            "status": "sent",
        })

        # Deliver to recipient
        recipient_server = self.connection_manager.find_server(recipient_id)
        if recipient_server:
            if recipient_server == self.server_id:
                # Same server — deliver directly
                await self.connection_manager.send_to_user(
                    recipient_id, message
                )
            else:
                # Different server — route via Redis Pub/Sub
                self.redis.publish(
                    f"deliver:{recipient_server}",
                    json.dumps(message),
                )
        else:
            # User is offline — send push notification
            await push_service.notify(recipient_id, message)
```

### Group Message Fan-Out

Group messages require delivering to all group members. For a group with N members, this is a fan-out of N.

```python
async def handle_group_message(self, message: dict):
    group_id = message["group_id"]
    sender_id = message["sender_id"]

    # Get group members
    members = await db.fetch_all(
        "SELECT user_id FROM group_members WHERE group_id = $1",
        group_id,
    )

    # Persist message
    message["message_id"] = str(uuid.uuid4())
    message["server_timestamp"] = datetime.utcnow().isoformat()
    await kafka_producer.send(
        topic="chat-messages",
        key=group_id,
        value=message,
    )

    # Fan-out to all members (except sender)
    delivery_tasks = []
    for member in members:
        if member["user_id"] != sender_id:
            delivery_tasks.append(
                self.deliver_to_user(member["user_id"], message)
            )

    # Deliver in parallel
    await asyncio.gather(*delivery_tasks, return_exceptions=True)
```

For large groups (100+ members), fan-out should be asynchronous. The chat service publishes the message to Kafka, and a separate delivery worker handles the fan-out.

### Message Storage

Messages need to be stored for history and searchability. The access pattern is: "Get the last N messages in conversation X, ordered by timestamp."

```sql
-- Cassandra-style schema (wide column store, optimized for time-series access)
CREATE TABLE messages (
    conversation_id TEXT,
    message_id TIMEUUID,
    sender_id TEXT,
    content TEXT,
    message_type TEXT,  -- 'text', 'image', 'file'
    media_url TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Query: last 50 messages in a conversation
SELECT * FROM messages
WHERE conversation_id = 'conv_123'
ORDER BY message_id DESC
LIMIT 50;
```

Cassandra is a common choice for chat message storage because:
- Write-optimized (append-only)
- Partition by conversation_id distributes data evenly
- Time-ordered clustering allows efficient range queries
- Linear scalability (add nodes to increase capacity)

### Presence Service

Presence tracking (online/offline status) uses a heartbeat mechanism:

```python
class PresenceService:
    HEARTBEAT_INTERVAL = 30  # seconds
    OFFLINE_THRESHOLD = 90    # seconds without heartbeat = offline

    def __init__(self, redis_client):
        self.redis = redis_client

    async def heartbeat(self, user_id: str):
        """Called every 30 seconds by connected clients."""
        self.redis.setex(
            f"presence:{user_id}",
            self.OFFLINE_THRESHOLD,
            "online",
        )

    def is_online(self, user_id: str) -> bool:
        return self.redis.exists(f"presence:{user_id}")

    def get_online_friends(self, user_id: str) -> list:
        friend_ids = self.get_friends(user_id)
        pipeline = self.redis.pipeline()
        for fid in friend_ids:
            pipeline.exists(f"presence:{fid}")
        results = pipeline.execute()
        return [fid for fid, online in zip(friend_ids, results) if online]
```

### Scaling Considerations

**Partition WebSocket connections by user ID hash**: Consistent hashing maps each user to a specific gateway server. If a server fails, only its users need to reconnect.

**Message ordering**: Kafka partitions by conversation_id, guaranteeing ordering within a conversation. Different conversations can be processed in parallel across partitions.

**Hot groups**: A group with 500 active members generates 500x fan-out per message. Isolate hot groups on dedicated delivery workers to prevent them from affecting 1:1 chat latency.

---

## Case Study 3: News Feed System


![System design case study architect blueprint of large scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/08-system-design-case-study-architect-blueprint-of-large-scale-.jpg)

A news feed system displays a personalized, ranked stream of content from users and pages that you follow. This is the core product feature of platforms like Facebook, Twitter, and Instagram.

![News feed design](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/08-news-feed.png)


### Requirements

**Functional**:
- Users can create posts (text, images, links)
- Users can follow/unfollow other users
- Users see a feed of posts from people they follow
- Feed is ranked (not purely chronological)
- Support pagination (infinite scroll)
- Like and comment on posts

**Non-functional**:
- 200 million DAU
- Average user follows 200 accounts
- Average user publishes 1 post per day and reads their feed 10 times per day
- Feed generation latency under 500ms
- Availability: 99.9%
- Eventual consistency is acceptable (a post can appear in followers' feeds with a few seconds of delay)

### Estimation

**Post creation QPS**:
```text
200M DAU × 1 post/day ÷ 86,400 = ~2,300 posts/sec
Peak (3x): ~7,000 posts/sec
```

**Feed read QPS**:
```text
200M DAU × 10 reads/day ÷ 86,400 = ~23,000 reads/sec
Peak (3x): ~70,000 reads/sec
```

**Fan-out volume**:
```text
Each post fans out to the poster's followers.
Average followers per user: 200
2,300 posts/sec × 200 followers = 460,000 fan-out writes/sec
```

460,000 writes per second to feed caches/stores is significant but manageable with a distributed system.

### The Fan-Out Problem

The central design challenge: how to build each user's feed from the posts of the people they follow.

**Fan-Out on Write (Push Model)**: When a user publishes a post, immediately write it to each follower's feed cache.

```text
User A publishes a post:
  → For each of A's 200 followers:
    → Add post to follower's pre-computed feed cache

When a user opens their feed:
  → Read directly from their pre-computed feed cache (fast!)
```

Advantages:
- Feed reads are extremely fast (pre-computed)
- No complex query at read time

Disadvantages:
- High write amplification (1 post → 200+ writes)
- Celebrity problem: a user with 10M followers triggers 10M writes per post
- Wasted work for inactive users who never read their feed

**Fan-Out on Read (Pull Model)**: When a user opens their feed, query the posts from everyone they follow in real-time.

```text
User opens their feed:
  → Get list of followed users (200 users)
  → Query recent posts from each followed user
  → Merge and rank all posts
  → Return top N posts
```

Advantages:
- No write amplification
- No wasted work (only compute when someone reads)
- Handles celebrities naturally (no special case)

Disadvantages:
- Slow feed reads (must query 200+ users' posts and merge)
- High database load at read time
- Latency spikes during traffic peaks

**Hybrid Model (the practical choice)**: Use fan-out on write for regular users and fan-out on read for celebrities.

```text
User with < 10,000 followers: fan-out on write (push to followers' feeds)
User with >= 10,000 followers: fan-out on read (followers pull at read time)
```

This is the approach used by Twitter and most large social platforms.

### High-Level Architecture

Components:
1. **Post Service** — handles post creation and storage
2. **Feed Generation Service** — builds and caches feeds
3. **Fan-Out Service** — distributes posts to followers' feeds
4. **Ranking Service** — ranks feed items by relevance
5. **Social Graph Service** — manages follow relationships
6. **Feed Cache** — pre-computed feeds stored in Redis
7. **Post Cache** — frequently accessed posts cached in Redis
8. **Kafka** — decouples post creation from fan-out

### Post Creation Flow

```python
class PostService:
    async def create_post(self, user_id: str, content: dict) -> dict:
        # Create post in database
        post_id = str(uuid.uuid4())
        post = {
            "post_id": post_id,
            "user_id": user_id,
            "content": content["text"],
            "media_urls": content.get("media_urls", []),
            "created_at": datetime.utcnow().isoformat(),
        }

        await db.execute(
            "INSERT INTO posts (id, user_id, content, media_urls, created_at) "
            "VALUES ($1, $2, $3, $4, $5)",
            post_id, user_id, post["content"],
            json.dumps(post["media_urls"]), post["created_at"],
        )

        # Cache the post
        await redis.setex(
            f"post:{post_id}", 86400, json.dumps(post)
        )

        # Publish event for fan-out
        await kafka_producer.send(
            topic="new-posts",
            key=user_id,
            value=post,
        )

        return post
```

### Fan-Out Service

```python
class FanOutService:
    CELEBRITY_THRESHOLD = 10000

    async def process_new_post(self, post: dict):
        user_id = post["user_id"]

        # Get follower count
        follower_count = await social_graph.get_follower_count(user_id)

        if follower_count >= self.CELEBRITY_THRESHOLD:
            # Celebrity: skip fan-out, fans will pull at read time
            await redis.sadd("celebrity_users", user_id)
            return

        # Regular user: fan-out on write
        followers = await social_graph.get_followers(user_id)

        # Batch fan-out for efficiency
        pipeline = redis.pipeline()
        for follower_id in followers:
            feed_key = f"feed:{follower_id}"
            # Add post ID to follower's feed (sorted set, scored by timestamp)
            pipeline.zadd(
                feed_key,
                {post["post_id"]: float(post["created_at_epoch"])},
            )
            # Trim feed to last 1000 posts (prevent unbounded growth)
            pipeline.zremrangebyrank(feed_key, 0, -1001)

        await pipeline.execute()
```

### Feed Read Flow

```python
class FeedService:
    FEED_SIZE = 50  # Posts per page

    async def get_feed(self, user_id: str, cursor: str = None) -> dict:
        feed_key = f"feed:{user_id}"

        # Step 1: Get pre-computed feed (fan-out on write posts)
        if cursor:
            max_score = float(cursor)
        else:
            max_score = float("inf")

        post_ids = await redis.zrevrangebyscore(
            feed_key,
            max_score, "-inf",
            start=0, num=self.FEED_SIZE,
            withscores=True,
        )

        # Step 2: Merge with celebrity posts (fan-out on read)
        celebrity_ids = await self.get_followed_celebrities(user_id)
        if celebrity_ids:
            celebrity_posts = await self.fetch_celebrity_posts(
                celebrity_ids, max_score, self.FEED_SIZE
            )
            # Merge celebrity posts with pre-computed feed
            all_posts = self.merge_sorted(post_ids, celebrity_posts)
        else:
            all_posts = post_ids

        # Step 3: Fetch full post data (batch from cache/DB)
        enriched_posts = await self.enrich_posts(
            [pid for pid, _ in all_posts[:self.FEED_SIZE]]
        )

        # Step 4: Rank posts
        ranked_posts = await self.ranking_service.rank(
            user_id, enriched_posts
        )

        # Step 5: Build response with cursor for pagination
        next_cursor = None
        if len(ranked_posts) == self.FEED_SIZE:
            next_cursor = str(all_posts[self.FEED_SIZE - 1][1])

        return {
            "posts": ranked_posts,
            "next_cursor": next_cursor,
        }

    async def fetch_celebrity_posts(
        self, celebrity_ids: list, max_timestamp: float, limit: int
    ) -> list:
        """Pull recent posts from celebrity users (fan-out on read)."""
        tasks = [
            db.fetch_all(
                "SELECT post_id, created_at_epoch FROM posts "
                "WHERE user_id = $1 AND created_at_epoch < $2 "
                "ORDER BY created_at_epoch DESC LIMIT $3",
                celeb_id, max_timestamp, limit,
            )
            for celeb_id in celebrity_ids
        ]
        results = await asyncio.gather(*tasks)
        # Merge all celebrity posts, sort by timestamp
        merged = []
        for result in results:
            merged.extend([(r["post_id"], r["created_at_epoch"]) for r in result])
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:limit]
```

### Ranking

A purely chronological feed is straightforward but does not maximize engagement. A ranked feed uses signals to surface the most relevant posts:

```python
class RankingService:
    async def rank(self, user_id: str, posts: list) -> list:
        """Score and rank posts based on relevance signals."""
        scored_posts = []
        for post in posts:
            score = self.compute_score(user_id, post)
            scored_posts.append((score, post))

        scored_posts.sort(key=lambda x: x[0], reverse=True)
        return [post for _, post in scored_posts]

    def compute_score(self, user_id: str, post: dict) -> float:
        """Simple scoring function combining multiple signals."""
        score = 0.0

        # Recency: exponential decay over time
        age_hours = (time.time() - post["created_at_epoch"]) / 3600
        recency_score = math.exp(-0.1 * age_hours)
        score += recency_score * 10

        # Engagement: posts with more likes/comments rank higher
        engagement = post.get("like_count", 0) + post.get("comment_count", 0) * 2
        score += math.log1p(engagement) * 3

        # Affinity: how often the user interacts with the post author
        interaction_count = self.get_interaction_count(user_id, post["user_id"])
        affinity_score = math.log1p(interaction_count)
        score += affinity_score * 5

        # Content type boost: images and videos rank higher than text
        if post.get("media_urls"):
            score += 2

        return score
```

In production, this simple scoring function is replaced by an ML model trained on user behavior (click-through rate, dwell time, likes, shares). But the simple version illustrates the concept.

### The Celebrity Problem Deep Dive

When a user with 50 million followers publishes a post, fan-out on write would require 50 million cache writes. At 1 microsecond per write, that takes 50 seconds. Meanwhile, the next post from another celebrity starts its own fan-out. The system falls behind.

The hybrid model solves this: celebrities are handled via fan-out on read. But there is a spectrum between "regular user" and "celebrity." Some practical thresholds:

```text
Followers < 10,000:    Fan-out on write (pre-compute feed)
Followers 10K-1M:      Fan-out on write with lower priority (async, may be delayed)
Followers > 1M:        Fan-out on read only (pull at query time)
```

The threshold is not fixed. It depends on your infrastructure capacity, acceptable latency, and the percentage of followers who are actually active.

### Scaling Strategy

**Feed cache partitioning**: Partition by user ID hash across Redis cluster nodes. Each user's feed lives on a deterministic shard.

**Post storage**: Partition by user ID for write-optimized access (all of a user's posts on the same shard). Use a secondary index or search service for cross-user queries.

**Fan-out workers**: The fan-out service is a pool of Kafka consumers. Scale workers horizontally to handle post volume. Each worker processes fan-out for a subset of posts.

**Read path optimization**: Pre-compute and cache the top 200 posts per user's feed. Most users only scroll through the first 20-50 posts, so cache hit rates are high.

---

## Cross-Cutting Themes


![Url shortener architecture long url compressed into short co](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/08-url-shortener-architecture-long-url-compressed-into-short-co.jpg)

Looking across all three case studies, several patterns recur:

![Cross-cutting concerns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/08-cross-cutting.png)


**Read-heavy systems benefit from caching**: The URL shortener, chat history, and news feed all have read:write ratios of 10:1 to 100:1. Caching transforms an unscalable system into a scalable one.

**Async processing via message queues**: All three systems use Kafka to decouple the write path from downstream processing. The URL shortener decouples analytics. The chat system decouples message storage from delivery. The news feed decouples post creation from fan-out.

**The right data store for the right access pattern**: The URL shortener uses a key-value store (hash lookup). The chat system uses a wide-column store (time-ordered messages per conversation). The news feed uses a sorted set cache (ranked posts per user). No single database fits all three.

**Estimation drives architecture**: The numbers calculated in the estimation phase determine which components are needed. 350,000 reads/sec demands a cache. 460,000 fan-out writes/sec demands a message queue. 15 million concurrent connections demands a distributed WebSocket gateway. Without the math, these decisions are guesswork.

## What's Next

This article concludes the System Design series. The eight articles together cover the full spectrum from estimation fundamentals to complete system designs. The next step is practice: pick a system you use daily, define its requirements, estimate its scale, and design its architecture. The more systems you design, the more patterns you recognize, and the faster you converge on good solutions.

![System design template](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/08-design-template.png)

