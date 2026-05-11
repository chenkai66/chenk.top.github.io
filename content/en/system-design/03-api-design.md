---
title: "System Design (3): API Design — REST, gRPC, GraphQL, and Choosing Wisely"
date: 2025-07-15 09:00:00
tags:
  - System Design
  - API Design
  - REST
  - gRPC
categories:
  - System Design
series: system-design
lang: en
description: "A practical comparison of REST, gRPC, and GraphQL — covering protocol design, real-world trade-offs, rate limiting algorithms, idempotency, and a decision framework for choosing the right API style."
disableNunjucks: true
series_order: 3
translationKey: "system-design-3"
---

In 2015, Facebook published a blog post introducing GraphQL, describing how their mobile app was drowning in REST API calls. A single news feed screen required data from posts, users, comments, likes, and media — each a separate endpoint, each returning far more data than the client needed. The over-fetching was killing mobile performance on slow networks. GraphQL was their solution, but it was not a universal solution.

Every API style exists because it solves a specific set of problems well, and every API style creates new problems. The skill is matching the right protocol to the right context.

## REST: The Lingua Franca of Web APIs

REST (Representational State Transfer) is an architectural style, not a protocol. It was defined by Roy Fielding in his 2000 doctoral dissertation, but what most people call "REST" is really "HTTP-based APIs that use JSON."

![REST vs gRPC vs GraphQL](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-rest-grpc-graphql.png)


### Core Concepts

REST models everything as a **resource**, identified by a URL. Operations on resources map to HTTP methods.

| HTTP Method | CRUD Operation | Example |
|------------|---------------|---------|
| GET | Read | `GET /users/123` |
| POST | Create | `POST /users` |
| PUT | Replace | `PUT /users/123` |
| PATCH | Partial update | `PATCH /users/123` |
| DELETE | Delete | `DELETE /users/123` |

### Status Codes

Status codes communicate the result of an operation. Using them correctly is the difference between a good API and a frustrating one.

```
2xx Success
  200 OK           — Request succeeded, response body contains result
  201 Created      — Resource created, Location header points to it
  204 No Content   — Success, no response body (common for DELETE)

3xx Redirection
  301 Moved Permanently — Resource has a new URL
  304 Not Modified      — Cached version is still valid

4xx Client Error
  400 Bad Request    — Malformed request (validation errors)
  401 Unauthorized   — Authentication required
  403 Forbidden      — Authenticated but not authorized
  404 Not Found      — Resource does not exist
  409 Conflict       — Request conflicts with current state
  429 Too Many Requests — Rate limited

5xx Server Error
  500 Internal Server Error — Unhandled exception
  502 Bad Gateway          — Upstream service failure
  503 Service Unavailable  — Server temporarily overloaded
  504 Gateway Timeout      — Upstream service timeout
```

### REST Best Practices

**URL Design**: Use nouns for resources, not verbs. The HTTP method is the verb.

```
Good:
  GET    /photos              # List photos
  POST   /photos              # Create a photo
  GET    /photos/456          # Get a specific photo
  DELETE /photos/456          # Delete a photo
  GET    /users/123/photos    # List photos by user 123

Bad:
  GET    /getPhotos
  POST   /createPhoto
  POST   /deletePhoto?id=456
```

**Versioning**: Two common approaches, each with trade-offs.

URL versioning (`/v1/photos`):
- Easy to understand and implement
- Clear in logs and documentation
- Clutters URL namespace
- Forces clients to update URLs on version change

Header versioning (`Accept: application/vnd.example.v1+json`):
- Cleaner URLs
- More RESTful (same resource, different representation)
- Harder to test in browser
- Easy to forget, leading to implicit versioning bugs

In practice, URL versioning wins for public APIs because of its simplicity. Header versioning works for internal APIs where you control all clients.

**Pagination**: Never return unbounded lists.

```
# Offset-based pagination
GET /photos?offset=20&limit=10

# Cursor-based pagination (better for large datasets)
GET /photos?cursor=eyJpZCI6MTAwfQ&limit=10

Response:
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6MTEwfQ",
    "has_more": true
  }
}
```

Cursor-based pagination is superior for large, actively-changing datasets. Offset-based pagination breaks when items are inserted or deleted between page requests (you skip or duplicate items).

**Filtering and Sorting**:

```
GET /photos?user_id=123&created_after=2025-01-01&sort=-created_at&fields=id,url,caption
```

The `fields` parameter reduces payload size — a lightweight alternative to GraphQL's field selection.

### REST Anti-Patterns

**RPC-style URLs**: Using POST for everything and encoding the operation in the URL.

```
POST /api/executeAction
Body: { "action": "getUser", "userId": 123 }
```

This loses all the benefits of REST: cacheability (GET is cacheable, POST is not), discoverability, standard tooling.

**Ignoring HTTP semantics**: Returning 200 for errors with an error flag in the body.

```json
{
  "success": false,
  "error": "User not found"
}
```

HTTP status codes exist for a reason. Middleware, proxies, and client libraries all depend on them.

**Deeply nested resources**: URLs like `/companies/1/departments/2/teams/3/members/4/roles` suggest your API models your database schema rather than your use cases. Flatten when depth exceeds 2 levels.

## gRPC: The Performance Protocol

gRPC is a high-performance, open-source RPC framework developed by Google. It uses Protocol Buffers (protobuf) for serialization and HTTP/2 for transport.

![API versioning strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-api-versioning.png)


### Protocol Buffers

Protobuf is a binary serialization format. You define your data structures and service interfaces in `.proto` files, and code generators produce client and server stubs in your target language.

```protobuf
syntax = "proto3";

package photos;

// Data structures
message Photo {
  string id = 1;
  string url = 2;
  string caption = 3;
  string user_id = 4;
  int64 created_at = 5;
  repeated string tags = 6;
}

message GetPhotoRequest {
  string id = 1;
}

message ListPhotosRequest {
  string user_id = 1;
  int32 page_size = 2;
  string page_token = 3;
}

message ListPhotosResponse {
  repeated Photo photos = 1;
  string next_page_token = 2;
}

message UploadPhotoRequest {
  bytes image_data = 1;
  string caption = 2;
  repeated string tags = 3;
}

// Service definition
service PhotoService {
  rpc GetPhoto(GetPhotoRequest) returns (Photo);
  rpc ListPhotos(ListPhotosRequest) returns (ListPhotosResponse);
  rpc UploadPhoto(UploadPhotoRequest) returns (Photo);
  rpc StreamPhotos(ListPhotosRequest) returns (stream Photo);  // Server streaming
}
```

### gRPC Streaming Modes

gRPC supports four communication patterns:

**Unary**: Standard request-response. Client sends one message, server returns one message.

**Server streaming**: Client sends one request, server returns a stream of messages. Use case: subscribing to real-time updates, downloading large datasets.

**Client streaming**: Client sends a stream of messages, server returns one response. Use case: file upload, batch ingestion.

**Bidirectional streaming**: Both client and server send streams of messages independently. Use case: chat applications, collaborative editing.

### Using gRPC in Python

```python
# server.py
import grpc
from concurrent import futures
import photos_pb2
import photos_pb2_grpc

class PhotoServicer(photos_pb2_grpc.PhotoServiceServicer):
    def GetPhoto(self, request, context):
        # Look up photo by ID
        photo = db.get_photo(request.id)
        if photo is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Photo {request.id} not found")
            return photos_pb2.Photo()

        return photos_pb2.Photo(
            id=photo.id,
            url=photo.url,
            caption=photo.caption,
            user_id=photo.user_id,
            created_at=photo.created_at,
        )

    def StreamPhotos(self, request, context):
        """Server streaming - yields photos one at a time"""
        photos = db.list_photos(user_id=request.user_id)
        for photo in photos:
            yield photos_pb2.Photo(
                id=photo.id,
                url=photo.url,
                caption=photo.caption,
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    photos_pb2_grpc.add_PhotoServiceServicer_to_server(PhotoServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
```

```python
# client.py
import grpc
import photos_pb2
import photos_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = photos_pb2_grpc.PhotoServiceStub(channel)

# Unary call
photo = stub.GetPhoto(photos_pb2.GetPhotoRequest(id="abc123"))
print(f"Photo: {photo.caption}")

# Server streaming
request = photos_pb2.ListPhotosRequest(user_id="user456", page_size=10)
for photo in stub.StreamPhotos(request):
    print(f"Streamed photo: {photo.id}")
```

### Why Choose gRPC

gRPC excels in service-to-service communication within a backend:
- **Binary serialization**: 5-10x smaller payloads than JSON
- **HTTP/2**: Multiplexed streams over a single connection, header compression
- **Code generation**: Type-safe client/server stubs in 10+ languages
- **Streaming**: Native support for all four streaming patterns
- **Deadlines**: Built-in timeout propagation across service chains

### Why Not gRPC

gRPC is a poor fit for browser-to-server communication:
- Browsers do not support HTTP/2 trailers (required by gRPC)
- Binary format is not human-readable (harder to debug)
- gRPC-Web exists but adds complexity and limitations
- No native browser support means you need a proxy

## GraphQL: Client-Driven Queries

GraphQL is a query language for APIs. Instead of the server defining fixed endpoint shapes, the client specifies exactly what data it needs.

### Schema Definition

```graphql
type Photo {
  id: ID!
  url: String!
  caption: String
  user: User!
  tags: [String!]!
  likes: Int!
  comments(first: Int = 10): [Comment!]!
  createdAt: DateTime!
}

type User {
  id: ID!
  name: String!
  avatarUrl: String
  photos(first: Int = 20, after: String): PhotoConnection!
  followerCount: Int!
}

type Comment {
  id: ID!
  text: String!
  author: User!
  createdAt: DateTime!
}

type PhotoConnection {
  edges: [PhotoEdge!]!
  pageInfo: PageInfo!
}

type PhotoEdge {
  node: Photo!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  endCursor: String
}

type Query {
  photo(id: ID!): Photo
  feed(first: Int = 20, after: String): PhotoConnection!
  user(id: ID!): User
}

type Mutation {
  uploadPhoto(input: UploadPhotoInput!): Photo!
  likePhoto(photoId: ID!): Photo!
  addComment(photoId: ID!, text: String!): Comment!
}

type Subscription {
  newComment(photoId: ID!): Comment!
}
```

### Client Queries

The client requests exactly the fields it needs:

```graphql
# Mobile app: needs minimal data
query FeedMinimal {
  feed(first: 10) {
    edges {
      node {
        id
        url
        likes
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}

# Web app: needs full data including user info
query FeedFull {
  feed(first: 20) {
    edges {
      node {
        id
        url
        caption
        likes
        user {
          name
          avatarUrl
        }
        comments(first: 3) {
          text
          author {
            name
          }
        }
      }
    }
  }
}
```

Both queries hit the same endpoint (`POST /graphql`) but return different shapes. The mobile app fetches only 3 fields per photo; the web app fetches 8+ fields including nested user and comment data. This eliminates the over-fetching problem that motivated GraphQL's creation.

### GraphQL Drawbacks

**N+1 Query Problem**: A naive resolver implementation fetches related data one at a time.

```python
# Bad: N+1 queries
class PhotoResolver:
    def resolve_user(self, photo, info):
        return db.get_user(photo.user_id)  # Called once per photo!

# If you fetch 20 photos, this makes 1 query for photos + 20 queries for users = 21 queries
```

The solution is a DataLoader that batches and caches lookups:

```python
from promise import Promise
from promise.dataloader import DataLoader

class UserLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        users = db.get_users_by_ids(user_ids)  # Single batch query
        user_map = {u.id: u for u in users}
        return Promise.resolve([user_map.get(uid) for uid in user_ids])

# Now 20 photos result in 1 query for photos + 1 batch query for users = 2 queries
```

**Caching Complexity**: REST APIs can use HTTP caching (CDN, browser cache) because each URL is a unique cacheable resource. GraphQL uses a single endpoint with POST requests, which are not cacheable by default. You need application-level caching (persisted queries, response caching by query hash).

**Authorization Complexity**: In REST, you authorize at the endpoint level. In GraphQL, a single query can traverse multiple resource types, each with different authorization rules. You need field-level authorization.

**Query Complexity Attacks**: A malicious client can craft deeply nested queries that consume enormous server resources.

```graphql
# Malicious query: exponential expansion
query Evil {
  user(id: "1") {
    photos(first: 100) {
      edges { node {
        comments(first: 100) {
          author {
            photos(first: 100) {
              edges { node {
                comments(first: 100) {
                  author { name }
                }
              }}
            }
          }
        }
      }}
    }
  }
}
```

Mitigation: query depth limiting, query cost analysis, persisted queries (only allow pre-registered queries).

## Rate Limiting

Rate limiting protects your API from abuse and ensures fair resource allocation. Three common algorithms:

![Token bucket rate limiting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-rate-limiting.png)


### Token Bucket

A bucket holds tokens. Each request consumes one token. Tokens are added at a fixed rate. When the bucket is empty, requests are rejected.

```python
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # Tokens per second
        self.capacity = capacity  # Maximum burst size
        self.tokens = capacity    # Current tokens
        self.last_refill = time.monotonic()
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# 100 requests/second with burst of 200
limiter = TokenBucket(rate=100, capacity=200)
```

Token bucket allows bursts up to the bucket capacity, then throttles to the refill rate. This matches real-world traffic patterns well.

### Sliding Window Log

Track the timestamp of every request. Count requests within the window. Reject if count exceeds limit.

```python
import time
from collections import deque
from threading import Lock

class SlidingWindowLog:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.requests = deque()
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            # Remove expired entries
            while self.requests and self.requests[0] <= now - self.window:
                self.requests.popleft()

            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False
```

Precise but memory-intensive (stores every timestamp). Not practical for high-traffic APIs.

### Fixed Window Counter

Divide time into fixed windows (e.g., 1-minute intervals). Count requests per window. Reject if count exceeds limit.

```python
import time
from threading import Lock

class FixedWindowCounter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.current_window = 0
        self.count = 0
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = int(time.monotonic() / self.window)
            if now != self.current_window:
                self.current_window = now
                self.count = 0

            if self.count < self.limit:
                self.count += 1
                return True
            return False
```

Simple and memory-efficient, but has a boundary problem: a burst at the end of window N and start of window N+1 can allow 2x the limit. The sliding window counter variant fixes this by weighting the previous window's count.

### Rate Limit Headers

Always communicate rate limit status in response headers:

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1689436800

HTTP/1.1 429 Too Many Requests
Retry-After: 30
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1689436800
```

## Idempotency

An operation is idempotent if performing it multiple times has the same effect as performing it once. This is critical for reliability because network failures cause retries, and retries must not create duplicate side effects.

![Idempotency key pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-idempotency.png)


HTTP methods and idempotency:
- **GET**: Naturally idempotent (reading data does not change state)
- **PUT**: Naturally idempotent (replacing a resource with the same data is a no-op)
- **DELETE**: Naturally idempotent (deleting an already-deleted resource is a no-op)
- **POST**: NOT naturally idempotent (creating a resource twice creates duplicates)
- **PATCH**: May or may not be idempotent (depends on the operation)

### Idempotency Keys

For non-idempotent operations, the client generates a unique key and includes it with the request. The server uses this key to detect duplicates.

```python
# Client sends:
POST /payments
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
Content-Type: application/json

{
  "amount": 100.00,
  "currency": "USD",
  "recipient": "user_456"
}
```

```python
# Server implementation
import hashlib
import json

def process_payment(request):
    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key:
        return error_response(400, "Idempotency-Key header required")

    # Check if we already processed this key
    existing = redis.get(f"idempotency:{idempotency_key}")
    if existing:
        return json.loads(existing)  # Return cached response

    # Process the payment
    result = payment_service.charge(
        amount=request.json["amount"],
        currency=request.json["currency"],
        recipient=request.json["recipient"],
    )

    # Cache the response for 24 hours
    redis.setex(
        f"idempotency:{idempotency_key}",
        86400,
        json.dumps(result),
    )

    return result
```

If the client retries (network timeout, 5xx error), the server recognizes the duplicate key and returns the cached response without processing the payment again.

## API Authentication

A brief overview of common authentication mechanisms:

![API authentication methods](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-api-auth.png)


**API Keys**: Simple, suitable for server-to-server communication. Include in a header (`X-API-Key: abc123`) or query parameter. Easy to implement but hard to scope (all-or-nothing access).

**OAuth 2.0**: Delegated authorization. The user grants a third-party application limited access to their resources. Four grant types (authorization code, implicit, client credentials, device code). Complex but industry standard for user-facing APIs.

**JWT (JSON Web Tokens)**: Self-contained tokens that encode claims (user ID, roles, expiration). The server validates the token's signature without a database lookup. Useful for stateless authentication but cannot be revoked individually (use short expiration + refresh tokens).

```python
# JWT validation example
import jwt

def authenticate(request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")
```

## Comparison: REST vs gRPC vs GraphQL

| Feature | REST | gRPC | GraphQL |
|---------|------|------|---------|
| Protocol | HTTP/1.1 or HTTP/2 | HTTP/2 | HTTP/1.1 or HTTP/2 |
| Data format | JSON (text) | Protobuf (binary) | JSON (text) |
| Schema/contract | OpenAPI (optional) | .proto (required) | GraphQL schema (required) |
| Code generation | Optional | Built-in | Optional |
| Streaming | SSE, WebSocket | Native (4 modes) | Subscriptions (WebSocket) |
| Browser support | Native | Via gRPC-Web proxy | Native |
| Caching | HTTP caching (native) | No standard caching | Complex (single endpoint) |
| Learning curve | Low | Medium | Medium-High |
| Over-fetching | Common problem | Minimal (typed messages) | Solved (client selects fields) |
| Under-fetching | Common (multiple calls) | Minimal (design per RPC) | Solved (nested queries) |
| Tooling maturity | Excellent | Good | Good |
| Best for | Public APIs, web apps | Internal microservices | Mobile apps, complex UIs |
| Worst for | Complex nested data | Browser clients | Simple CRUD APIs |

### Decision Framework

**Use REST when**:
- Building a public API consumed by third-party developers
- Clients are primarily web browsers
- The data model is simple and resource-oriented
- HTTP caching is important
- You want maximum ecosystem compatibility

**Use gRPC when**:
- Building internal service-to-service communication
- Low latency and high throughput are critical
- You need streaming (real-time data, file transfers)
- You want strong typing and code generation
- All clients are backend services you control

**Use GraphQL when**:
- Multiple clients need different data shapes (mobile vs web vs TV)
- The data model has many relationships (graph-like)
- Reducing network requests is critical (mobile on slow networks)
- Frontend teams need to iterate independently of backend
- You are willing to invest in the tooling (DataLoader, caching, authorization)

**Hybrid approaches** are common. Many systems use REST for public APIs, gRPC for internal services, and GraphQL for their frontend-facing gateway.

## What's Next

Once your API design is solid, the next performance lever is caching. The next article covers caching strategies — where to cache, what to evict, and the surprisingly common ways that caching makes things worse instead of better.
