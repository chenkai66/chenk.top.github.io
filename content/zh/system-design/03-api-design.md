---
title: "系统设计（三）：API 设计——REST、gRPC、GraphQL 及如何明智选型"
date: 2025-07-15 09:00:00
tags:
  - System Design
  - API Design
  - REST
  - gRPC
categories: System Design
series: system-design
lang: zh
description: "REST、gRPC 和 GraphQL 的实用对比 — 涵盖协议设计、真实场景下的权衡取舍、限流算法、幂等性保障，以及一套用于选择合适 API 风格的决策框架。"
disableNunjucks: true
series_order: 3
translationKey: "system-design-3"
---
2015 年，Facebook 发布了一篇博客文章，正式介绍 GraphQL，并描述了其移动应用正被海量 REST API 调用所“淹没”。单个新闻信息流页面就需要从帖子、用户、评论、点赞和媒体等多个资源获取数据——每个资源对应一个独立端点，且每个端点返回的数据远超客户端实际所需。这种过度获取（over-fetching）在弱网环境下严重拖垮了移动端性能。GraphQL 是他们的解决方案，但它绝非万能解药。

每一种 API 风格之所以存在，是因为它能出色地解决某类特定问题；而与此同时，每一种风格也必然引入新的挑战。真正的技术能力，在于将正确的协议匹配到正确的上下文中。

## REST：Web API 的通用语言

REST（Representational State Transfer）是一种架构风格，而非具体协议。它由 Roy Fielding 在其 2000 年博士论文中提出，但现实中人们常说的“REST”，往往指的是“基于 HTTP、使用 JSON 的 API”。

![REST、gRPC 和 GraphQL 的比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-rest-grpc-graphql.png)

### 核心概念

REST 将一切建模为 **资源（resource）**，每个资源通过 URL 唯一标识。对资源的操作则映射为标准 HTTP 方法。

| HTTP 方法 | CRUD 操作 | 示例 |
|------------|---------------|---------|
| GET | 读取 | `GET /users/123` |
| POST | 创建 | `POST /users` |
| PUT | 全量替换 | `PUT /users/123` |
| PATCH | 局部更新 | `PATCH /users/123` |
| DELETE | 删除 | `DELETE /users/123` |

### 状态码（Status Codes）

状态码用于传达操作结果。正确使用状态码，是区分优秀 API 与令人沮丧 API 的关键。

```text
2xx 成功
  200 OK           — 请求成功，响应体包含结果
  201 Created      — 资源已创建，Location 头指向新资源
  204 No Content   — 成功，无响应体（常用于 DELETE）

3xx 重定向
  301 Moved Permanently — 资源已永久迁移至新 URL
  304 Not Modified      — 缓存版本仍有效

4xx 客户端错误
  400 Bad Request    — 请求格式错误（校验失败）
  401 Unauthorized   — 需要身份认证
  403 Forbidden      — 已认证但无权限
  404 Not Found      — 资源不存在
  409 Conflict       — 请求与当前状态冲突
  429 Too Many Requests — 已触发限流

5xx 服务端错误
  500 Internal Server Error — 未捕获异常
  502 Bad Gateway          — 上游服务故障
  503 Service Unavailable  — 服务暂时过载
  504 Gateway Timeout      — 上游服务超时
```

### REST 最佳实践

**URL 设计**：使用名词表示资源，而非动词。HTTP 方法本身即为动词。

```sql
良好实践：
  GET    /photos              # 列出照片
  POST   /photos              # 创建照片
  GET    /photos/456          # 获取指定照片
  DELETE /photos/456          # 删除照片
  GET    /users/123/photos    # 列出用户 123 的所有照片

反模式：
  GET    /getPhotos
  POST   /createPhoto
  POST   /deletePhoto?id=456
```

**版本控制**：两种主流方式，各有取舍。

URL 版本化（`/v1/photos`）：
- 易理解、易实现
- 日志与文档中清晰可见
- 污染 URL 命名空间
- 版本变更时强制客户端更新 URL

Header 版本化（`Accept: application/vnd.example.v1+json`）：
- URL 更简洁
- 更符合 REST 原则（同一资源，不同表现形式）
- 浏览器中难以直接测试
- 容易被遗忘，导致隐式版本化 bug

实践中，公开 API 多采用 URL 版本化，因其简单可靠；内部 API 若可完全掌控所有客户端，则 Header 版本化更合适。

**分页（Pagination）**：切勿返回无界列表。

```json
# 偏移量分页（Offset-based pagination）
GET /photos?offset=20&limit=10

# 游标分页（Cursor-based pagination，更适合大数据集）
GET /photos?cursor=eyJpZCI6MTAwfQ&limit=10

响应体：
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6MTEwfQ",
    "has_more": true
  }
}
```

对于大规模、高频变更的数据集，游标分页明显优于偏移量分页。后者在两次分页请求之间若发生插入或删除操作，会导致跳过或重复条目。

**过滤与排序**：

```text
GET /photos?user_id=123&created_after=2025-01-01&sort=-created_at&fields=id,url,caption
```

`fields` 参数可显著减小响应负载——这是 GraphQL 字段选择机制的一种轻量级替代方案。

### REST 反模式（Anti-Patterns）

**RPC 风格 URL**：全部使用 POST，并将操作语义编码进 URL。

```json
POST /api/executeAction
Body: { "action": "getUser", "userId": 123 }
```

这彻底丧失了 REST 的核心优势：缓存能力（GET 可缓存，POST 不可）、可发现性、标准化工具链支持。

**忽略 HTTP 语义**：无论成功或失败均返回 `200 OK`，仅靠响应体中的标志位区分。

```json
{
  "success": false,
  "error": "User not found"
}
```

HTTP 状态码的存在自有其意义。中间件、代理服务器及各类客户端库均依赖其进行自动化处理。

**深层嵌套资源**：如 `/companies/1/departments/2/teams/3/members/4/roles` 这类 URL，往往暗示你的 API 直接暴露了数据库表结构，而非围绕业务用例建模。当嵌套深度超过 2 层时，应考虑扁平化设计。

## gRPC：高性能通信协议

gRPC 是 Google 开发的高性能开源 RPC 框架，采用 Protocol Buffers（protobuf）进行序列化，以 HTTP/2 作为传输层。

![API 版本控制策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-api-versioning.png)

### Protocol Buffers

Protobuf 是一种二进制序列化格式。你通过 `.proto` 文件定义数据结构与服务接口，再由代码生成器为多种目标语言自动生成客户端与服务端桩（stub）。

```protobuf
syntax = "proto3";

package photos;

// 数据结构
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

// 服务定义
service PhotoService {
  rpc GetPhoto(GetPhotoRequest) returns (Photo);
  rpc ListPhotos(ListPhotosRequest) returns (ListPhotosResponse);
  rpc UploadPhoto(UploadPhotoRequest) returns (Photo);
  rpc StreamPhotos(ListPhotosRequest) returns (stream Photo);  // 服务端流式响应
}
```

### gRPC 流式通信模式

gRPC 支持四种通信模式：

**Unary（一元调用）**：标准请求-响应。客户端发送一条消息，服务端返回一条消息。

**Server streaming（服务端流式）**：客户端发送一个请求，服务端返回一个消息流。适用场景：实时更新订阅、大批量数据下载。

**Client streaming（客户端流式）**：客户端发送一个消息流，服务端返回一条响应。适用场景：文件上传、批量数据写入。

**Bidirectional streaming（双向流式）**：客户端和服务端各自独立发送消息流。适用场景：聊天应用、协同编辑。

### Python 中使用 gRPC

```python
# server.py
import grpc
from concurrent import futures
import photos_pb2
import photos_pb2_grpc

class PhotoServicer(photos_pb2_grpc.PhotoServiceServicer):
    def GetPhoto(self, request, context):
        # 根据 ID 查询照片
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
        """服务端流式响应 —— 逐个 yield 照片"""
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

# 一元调用
photo = stub.GetPhoto(photos_pb2.GetPhotoRequest(id="abc123"))
print(f"Photo: {photo.caption}")

# 服务端流式
request = photos_pb2.ListPhotosRequest(user_id="user456", page_size=10)
for photo in stub.StreamPhotos(request):
    print(f"Streamed photo: {photo.id}")
```

### 为何选择 gRPC

gRPC 在后端服务间通信中表现出色：
- **二进制序列化**：负载比 JSON 小 5–10 倍
- **HTTP/2**：单连接多路复用、头部压缩
- **代码生成**：支持 10+ 种语言的类型安全客户端/服务端桩
- **原生流式支持**：完整支持全部四种流式模式
- **截止时间（Deadlines）**：内置超时传播机制，贯穿整个服务调用链

### 为何不选 gRPC

gRPC 不适合浏览器直连服务端的场景：
- 浏览器不支持 HTTP/2 trailers（gRPC 所必需）
- 二进制格式不可读（调试困难）
- gRPC-Web 方案虽存在，但增加了复杂性和功能限制
- 无原生浏览器支持，必须经由代理中转

## GraphQL：由客户端驱动的查询

GraphQL 是一种专为 API 设计的查询语言。不同于服务端预定义固定端点结构，GraphQL 允许客户端精确声明所需数据。

### Schema 定义

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

### 客户端查询示例

客户端按需请求字段：

```graphql
# 移动端 App：只需最小化数据
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

# Web 端 App：需要完整数据（含用户信息）
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

两个查询均命中同一端点（`POST /graphql`），但返回结构截然不同。移动端每张照片仅拉取 3 个字段；Web 端则拉取 8+ 字段，包括嵌套的用户与评论数据。这从根本上解决了催生 GraphQL 的过度获取问题。

### GraphQL 的缺陷

**N+1 查询问题**：低效的解析器（resolver）实现会逐个获取关联数据。

```python
# 反模式：N+1 查询
class PhotoResolver:
    def resolve_user(self, photo, info):
        return db.get_user(photo.user_id)  # 每张照片调用一次！

# 若拉取 20 张照片，则执行 1 次照片查询 + 20 次用户查询 = 共 21 次查询
```

解决方案是使用 DataLoader 批量合并并缓存查询：

```python
from promise import Promise
from promise.dataloader import DataLoader

class UserLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        users = db.get_users_by_ids(user_ids)  # 单次批量查询
        user_map = {u.id: u for u in users}
        return Promise.resolve([user_map.get(uid) for uid in user_ids])

# 现在 20 张照片仅需 1 次照片查询 + 1 次批量用户查询 = 共 2 次查询
```

**缓存复杂性**：REST API 可天然利用 HTTP 缓存（CDN、浏览器缓存），因每个 URL 对应唯一可缓存资源。而 GraphQL 仅有一个 `POST /graphql` 端点，POST 请求默认不可缓存。你必须构建应用层缓存策略（如持久化查询、按查询哈希缓存响应）。

**鉴权复杂性**：REST 中鉴权发生在端点粒度；GraphQL 中单个查询可横跨多个资源类型，而每种资源可能拥有不同的访问规则。因此需实现字段级（field-level）鉴权。

**查询复杂度攻击**：恶意客户端可构造深度嵌套查询，耗尽服务端计算资源。

```graphql
# 恶意查询：指数级膨胀
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

缓解措施：限制查询深度、分析查询成本（cost analysis）、启用持久化查询（只允许预注册的查询）。

## 限流（Rate Limiting）

![API 设计的三条路径：REST、gRPC 和 GraphQL 分歧之路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/03-api-design-three-paths-rest-grpc-graphql-diverging-roads.jpg)

限流用于保护 API 免受滥用，并确保资源公平分配。三种常用算法如下：

![令牌桶限流动画](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/sysdesign-03-token-bucket.gif)

![令牌桶限流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-rate-limiting.png)

### 令牌桶（Token Bucket）

一个“桶”中存放令牌。每次请求消耗一个令牌；令牌以固定速率持续注入。桶空时，新请求被拒绝。

```python
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # 每秒令牌数
        self.capacity = capacity  # 最大突发容量
        self.tokens = capacity    # 当前令牌数
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

# 每秒 100 次请求，最大突发 200 次
limiter = TokenBucket(rate=100, capacity=200)
```

令牌桶允许突发流量达桶容量上限，之后严格按注入速率限流，非常贴合真实世界流量特征。

### 滑动窗口日志（Sliding Window Log）

记录每次请求的时间戳，统计窗口内请求数。超限时拒绝。

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
            # 清理过期时间戳
            while self.requests and self.requests[0] <= now - self.window:
                self.requests.popleft()

            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False
```

精度高，但内存开销大（需存储每个时间戳），高流量场景下不实用。

### 固定窗口计数器（Fixed Window Counter）

将时间划分为固定长度窗口（如 1 分钟），统计各窗口内请求数。超限时拒绝。

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

简单高效、内存占用极小，但存在边界问题：窗口 N 末尾与窗口 N+1 开头的突发流量可能导致总请求数达限值的 2 倍。滑动窗口计数器变体可通过加权前一窗口计数来修复此问题。

### 限流响应头（Rate Limit Headers）

务必在响应头中明示限流状态：

```text
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

## 幂等性（Idempotency）

若某操作重复执行多次与执行一次效果相同，则称其为幂等操作。这是系统可靠性基石——网络故障必然引发重试，而重试绝不能产生重复副作用。

![幂等键模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-idempotency.png)

HTTP 方法与幂等性关系：
- **GET**：天然幂等（读取不改变状态）
- **PUT**：天然幂等（用相同数据替换资源即为无操作）
- **DELETE**：天然幂等（删除已不存在的资源即为无操作）
- **POST**：**非**天然幂等（重复创建将产生重复资源）
- **PATCH**：视具体操作而定（可能幂等，也可能不幂等）

### 幂等性密钥（Idempotency Keys）

针对非幂等操作，客户端生成唯一密钥并随请求携带；服务端据此识别并拒绝重复请求。

```python
# 客户端发送：
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
# 服务端实现
import hashlib
import json

def process_payment(request):
    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key:
        return error_response(400, "Idempotency-Key header required")

    # 检查是否已处理过该密钥
    existing = redis.get(f"idempotency:{idempotency_key}")
    if existing:
        return json.loads(existing)  # 返回缓存响应

    # 执行支付
    result = payment_service.charge(
        amount=request.json["amount"],
        currency=request.json["currency"],
        recipient=request.json["recipient"],
    )

    # 缓存响应 24 小时
    redis.setex(
        f"idempotency:{idempotency_key}",
        86400,
        json.dumps(result),
    )

    return result
```

若客户端因网络超时或服务端 5xx 错误而重试，服务端将识别重复密钥，直接返回缓存响应，避免重复扣款。

## API 认证（Authentication）

常见认证机制简述：

![API 认证方法](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/03-api-auth.png)

**API Keys**：简单，适用于服务间通信。置于请求头（`X-API-Key: abc123`）或查询参数中。易于实现，但难以精细化授权（通常为全有或全无）。

**OAuth 2.0**：委托式授权。用户授予第三方应用对其资源的有限访问权。含四种授权模式（授权码、隐式、客户端凭证、设备码）。复杂但已成为面向用户的行业标准。

**JWT（JSON Web Tokens）**：自包含令牌，内嵌声明（如用户 ID、角色、过期时间）。服务端仅需验证签名，无需查库。适用于无状态认证，但单个 JWT 无法主动吊销（需配合短有效期 + Refresh Token）。

```python
# JWT 校验示例
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

## 对比：REST vs gRPC vs GraphQL

![将令牌桶限流比作可控的喷泉](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/03-rate-limiting-token-bucket-as-a-water-fountain-with-controll.jpg)

| 特性 | REST | gRPC | GraphQL |
|---------|------|------|---------|
| 协议 | HTTP/1.1 或 HTTP/2 | HTTP/2 | HTTP/1.1 或 HTTP/2 |
| 数据格式 | JSON（文本） | Protobuf（二进制） | JSON（文本） |
| Schema/契约 | OpenAPI（可选） | `.proto`（必需） | GraphQL Schema（必需） |
| 代码生成 | 可选 | 内置 | 可选 |
| 流式支持 | SSE、WebSocket | 原生（4 种模式） | 订阅（Subscription，基于 WebSocket） |
| 浏览器支持 | 原生 | 需 gRPC-Web 代理 | 原生 |
| 缓存 | HTTP 缓存（原生） | 无标准缓存机制 | 复杂（单端点） |
| 学习曲线 | 低 | 中等 | 中高 |
| 过度获取（Over-fetching） | 常见问题 | 极少（强类型消息） | 已解决（客户端自主选字段） |
| 获取不足（Under-fetching） | 常见（需多次调用） | 极少（按 RPC 设计） | 已解决（嵌套查询） |
| 工具链成熟度 | 极佳 | 良好 | 良好 |
| 最适用场景 | 公开 API、Web 应用 | 内部微服务通信 | 移动端、复杂 UI |
| 最不适用场景 | 复杂嵌套数据 | 浏览器直连 | 简单 CRUD 场景 |

### 决策框架（Decision Framework）

**选用 REST 当**：
- 构建供第三方开发者使用的公开 API
- 主要客户端为 Web 浏览器
- 数据模型简单、以资源为中心
- HTTP 缓存至关重要
- 追求最大生态兼容性

**选用 gRPC 当**：
- 构建内部服务间通信
- 对低延迟与高吞吐有严苛要求
- 需要流式能力（实时数据、文件传输）
- 需强类型保障与自动代码生成
- 所有客户端均为可控的后端服务

**选用 GraphQL 当**：
- 多类客户端（移动端、Web、TV）需差异化数据结构
- 数据模型高度关联（图状结构）
- 减少网络请求至关重要（如弱网移动环境）
- 前端团队需独立于后端快速迭代
- 愿投入工具链建设（DataLoader、缓存、鉴权）

**混合架构（Hybrid approaches）十分常见**。许多系统采用 REST 对外提供公开 API，gRPC 用于内部服务通信，GraphQL 作为面向前端的统一网关。

## 下一步

当 API 设计趋于稳定后，下一个关键性能杠杆便是缓存。下一篇文章将深入探讨缓存策略——缓存应放在何处？哪些数据应被淘汰？以及那些令人惊讶却极为常见的、反而让系统变得更慢的缓存误用方式。
