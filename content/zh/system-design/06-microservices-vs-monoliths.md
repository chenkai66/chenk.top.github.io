---
title: "系统设计（六）：微服务 vs 单体架构——坦诚的权衡分析"
date: 2025-07-22 09:00:00
tags:
  - System Design
  - Microservices
  - Architecture
  - Distributed Systems
categories: System Design
series: system-design
lang: zh
description: "对单体架构与微服务的坦诚评估——涵盖分布式系统开销、基于领域驱动设计（DDD）的服务边界划分、服务间通信模式、熔断器、服务网格、API 网关，以及一套实用的决策框架。"
disableNunjucks: true
series_order: 6
translationKey: "system-design-6"
---
2020 年，客户数据平台 Segment 的工程团队发布了一篇题为《告别微服务》（Goodbye Microservices）的博客文章。当时，他们已将原有单体应用拆分为 **140 多个微服务**，但结果并非预期中的工程乌托邦。相反，团队大部分时间都在对抗分布式系统自身带来的复杂性：服务发现失败、级联超时、不一致的部署流水线，以及爆炸式增长的服务间通信缺陷。最终，他们选择回归单体架构，并报告称开发者生产力与系统可靠性均获得显著提升。

这个故事并非孤例。微服务模式虽已成为业界默认的架构选择，但坦率地说，它并不适合大多数团队。准确判断微服务何时带来收益、何时反而造成伤害，是系统设计中最重要的判断之一。

---

## 单体架构（Monolith）

单体是一个**单一可部署单元**，包含应用全部功能。整个代码库被统一编译、统一部署；所有模块共享同一进程、同一内存空间、同一数据库。

![单体架构与微服务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-monolith-vs-micro.png)

### 为什么单体架构行之有效

**简洁性（Simplicity）**：一个代码库、一条部署流水线、一个数据库、一套日志。新开发者只需克隆仓库，运行 `docker-compose up`，几分钟内即可在本地启动完整系统。

**性能（Performance）**：进程内函数调用耗时仅纳秒级。无需网络序列化、无 HTTP 开销、无服务发现延迟。单体可在极简基础设施上实现极高吞吐量。

**事务一致性（Transactional consistency）**：因所有模块共享数据库，整个业务域天然支持 ACID 事务。例如“扣减库存”与“记录销售”可置于同一数据库事务中完成，无需分布式协调。

**重构便利性（Refactoring）**：模块间移动代码仅需简单重构。而在微服务世界中，功能迁移意味着修改 API、更新消费者、管理数据迁移、协调多服务部署——成本陡增。

### 单体架构的瓶颈场景

**团队规模扩展（Team scaling）**：当 50+ 工程师共用同一代码库时，合并冲突频发、部署队列拉长，且某团队的 Bug 可能直接破坏其他团队的功能。

**伸缩性瓶颈（Scaling bottlenecks）**：若某一模块需比其余模块高 10 倍的计算资源，则必须将整个单体扩容 10 倍——无法对热点路径进行独立伸缩。

**技术栈锁定（Technology lock-in）**：整个应用被绑定在同一语言、框架与数据库上。即使某个模块明显更适合另一种技术（如 Rust 或 Elasticsearch），也无法局部采用而不影响全局。

**故障爆炸半径（Blast radius）**：某模块的内存泄漏会崩溃整个进程；一次糟糕的部署将同时影响全部功能。

### 模块化单体（Modular Monolith）

在跃向微服务前，请先考虑**模块化单体**——即一个可独立部署的单元，但内部具备严格模块边界。

```text
project/
├── modules/
│   ├── orders/
│   │   ├── api.py          # 公共接口（其他模块仅通过此调用）
│   │   ├── service.py      # 业务逻辑
│   │   ├── repository.py   # 数据访问
│   │   └── models.py       # 领域模型
│   ├── payments/
│   │   ├── api.py
│   │   ├── service.py
│   │   ├── repository.py
│   │   └── models.py
│   ├── inventory/
│   │   ├── api.py
│   │   ├── service.py
│   │   ├── repository.py
│   │   └── models.py
│   └── notifications/
│       ├── api.py
│       ├── service.py
│       └── repository.py
├── shared/
│   ├── database.py
│   └── events.py
└── main.py
```

约束规则：
1. 模块间**仅可通过其公共 API 通信**（禁止直连其他模块内部实现）  
2. 每个模块**独占其数据库表**（禁止跨模块直接访问表）  
3. 跨模块数据访问**必须经由该模块的 API**  
4. 模块边界可通过 Lint 规则或架构测试强制校验  

```python
# 架构测试：强制校验模块边界
import ast
import os

def check_module_boundaries():
    """验证无模块直接导入另一模块内部实现"""
    violations = []

    for module_dir in os.listdir("modules"):
        for py_file in glob(f"modules/{module_dir}/**/*.py"):
            tree = ast.parse(open(py_file).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("modules."):
                        imported_module = node.module.split(".")[1]
                        if imported_module != module_dir:
                            # 跨模块导入 —— 仅允许从 api.py 导入
                            imported_file = node.module.split(".")[-1]
                            if imported_file != "api":
                                violations.append(
                                    f"{py_file} imports {node.module} "
                                    f"(should only import from "
                                    f"modules.{imported_module}.api)"
                                )

    return violations
```

模块化单体能提供微服务的大部分组织优势（如清晰的模块归属、独立开发节奏），同时避免了分布式系统固有的额外开销。

## 微服务架构（Microservices Architecture）

微服务将应用拆解为多个**独立可部署的服务**，每个服务运行于独立进程，通过网络通信。

### 你所获得的优势

**独立部署能力（Independent deployability）**：订单团队可每日部署 10 次，无需与支付团队协调。各服务拥有专属 CI/CD 流水线、发布周期与回滚策略。

**技术多样性（Technology diversity）**：订单服务可用 Python，推荐引擎可用 Go 提升性能，搜索服务可集成 Elasticsearch。各团队按领域需求自由选型。

**团队自治（Team autonomy）**：每支团队端到端负责其服务——编码、测试、部署、监控与值班。Conway 定律在此成为助力：架构自然映射组织结构。

**独立伸缩（Independent scaling）**：黑五期间搜索服务流量激增 100 倍？只需将其扩至 50 实例，而订单服务维持 5 实例即可。

**故障隔离（Fault isolation）**：通知服务的内存泄漏不会导致支付服务崩溃。若实施得当，故障将被严格限制在服务边界内。

### 分布式系统开销（The Distributed Systems Tax）

每一项优势都伴随代价。微服务引入了单体中根本不存在的分布式系统问题。

**网络延迟（Network latency）**：函数调用耗时纳秒级；网络调用则达毫秒级。若用户请求需串行调用 5 个服务，仅网络延迟就增加 5–50ms（尚未计入处理时间）。

**部分失败（Partial failures）**：单体中进程非“全活”即“全死”；分布式系统中，服务 A 正常而服务 B 已宕机是常态。每次服务调用均需超时控制、重试逻辑与降级策略。

**数据一致性（Data consistency）**：失去共享数据库后，跨服务一致性需依赖分布式协调（Saga、最终一致性、补偿事务）——其难度远高于单数据库事务。

**运维复杂度（Operational complexity）**：监控对象从 1 个应用变为 20+ 个服务，每个服务均有独立日志、指标、告警、部署流水线与故障模式。你必须构建集中式日志、分布式追踪与服务网格基础设施。

**测试复杂度（Testing complexity）**：集成测试需启动多个服务；端到端测试缓慢且易失败；契约测试（Contract testing）成为刚需。

**调试难度（Debugging difficulty）**：单体中一个堆栈跟踪即可定位的 Bug，在微服务中演变为横跨多服务、多日志、多网络轨迹的侦探游戏。

## 服务边界：领域驱动设计（Domain-Driven Design）

![断路器模式：电气断路器保护](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/06-circuit-breaker-pattern-electrical-circuit-breaker-protectin.jpg)

微服务最难的部分，是**如何划定服务边界**。划错边界，轻则导致服务间高频“聊天”，重则催生大量无意义的细粒度服务。

领域驱动设计（DDD）为此提供了原则性方法。

### 限界上下文（Bounded Contexts）

限界上下文是**特定领域模型被定义并适用的边界**。在此边界内，术语具有精确、无歧义的含义；跨越边界时，同一术语可能指向不同概念。

![领域驱动设计的限界上下文](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-ddd-bounded-contexts.png)

示例：在电商系统中，“Order”（订单）在不同上下文中含义迥异：

- **销售上下文（Sales context）**：订单是客户的购买意向，含定价、折扣与支付条款  
- **履约上下文（Fulfillment context）**：订单是一组待拣货、打包、从仓库发货的商品  
- **财务上下文（Accounting context）**：订单是一笔财务交易，涉及收入确认、税额计算与账本记账  

每个限界上下文天然适合作为服务边界。关键洞见：**服务应围绕业务能力组织，而非技术分层**。

```text
错误方式（技术分层 —— 每次请求穿越全部服务）：
  API 网关 → 认证服务 → 业务逻辑服务 → 数据服务

正确方式（业务能力 —— 大多数请求停留于单一服务内）：
  订单服务（自有 API、逻辑、数据）
  支付服务（自有 API、逻辑、数据）
  库存服务（自有 API、逻辑、数据）
  配送服务（自有 API、逻辑、数据）
```

### 上下文映射（Context Mapping）

服务需相互通信，而边界之间必须建立显式契约。DDD 定义了若干关系模式：

**公开语言（Published Language）**：服务间约定共享的数据格式（如 Protobuf Schema、JSON Schema、OpenAPI 规范），用于事件或 API 合约。

**防腐层（Anti-Corruption Layer）**：对接遗留系统或外部服务时，构建翻译层，将外部模型转换为内部模型。此举可防止外部系统的设计决策污染你的代码库。

```python
# 针对遗留支付网关的防腐层
class PaymentGatewayAdapter:
    """在领域模型与遗留网关 API 之间进行转换。"""

    def __init__(self, legacy_client):
        self.client = legacy_client

    def charge(self, amount: Decimal, currency: str, customer_id: str) -> PaymentResult:
        # 遗留 API 使用“美分”而非“美元”
        # 遗留 API 使用 “cust_num” 而非 “customer_id”
        # 遗留 API 返回 “rc”（返回码）而非结构化错误
        response = self.client.process_transaction(
            amt_cents=int(amount * 100),
            cust_num=customer_id,
            curr_code=currency.upper(),
        )

        if response["rc"] == "00":
            return PaymentResult(
                success=True,
                transaction_id=response["txn_id"],
                amount=amount,
            )
        else:
            return PaymentResult(
                success=False,
                error_code=self._translate_error(response["rc"]),
                error_message=response.get("msg", "未知错误"),
            )
```

## 服务间通信（Inter-Service Communication）

![从单体架构到微服务的演变：从单一建筑到城市](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/06-monolith-to-microservices-evolution-single-building-to-city-.jpg)

### 同步通信：REST 与 gRPC

当调用方需要**即时响应**时使用同步通信。

```python
# 具备超时、重试与熔断能力的 REST 客户端
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_client() -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,           # 重试间隔：0.5s, 1s, 2s
        status_forcelist=[502, 503, 504],  # 对这些状态码重试
        allowed_methods=["GET", "PUT", "DELETE"],  # 仅对幂等方法重试
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

client = create_resilient_client()

def get_user(user_id: str) -> dict:
    response = client.get(
        f"http://user-service.internal/api/users/{user_id}",
        timeout=(3, 10),  # （连接超时，读取超时）
    )
    response.raise_for_status()
    return response.json()
```

### 异步通信：事件与消息

当调用方**无需即时响应**，或需广播至多个消费者时，采用异步通信。

上一篇关于消息队列的文章已详述此主题。核心设计原则：用 **事件（Events）表达事实**（如 `"OrderCreated"`），用 **命令（Commands）表达指令**（如 `"ProcessPayment"`）。

## 熔断器模式（Circuit Breaker Pattern）

当下游服务持续失败时，继续发送请求不仅浪费资源，更会引发级联故障。熔断器可及时阻断故障蔓延。

![断路器状态转换动画](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/sysdesign-06-circuit-breaker.gif)

![断路器状态机](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-circuit-breaker.png)

### 三种状态

熔断器拥有三个状态：

**关闭态（Closed）**（正常）：请求正常通过；失败次数被统计；若失败数超阈值，则熔断器开启。

**开启态（Open）**（故障）：请求被立即拒绝，不调用下游服务；超时后进入半开启态。

**半开启态（Half-Open）**（试探）：允许有限数量请求通过以探测下游健康度；若成功则恢复关闭态；若失败则重新开启。

```python
import time
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.lock = Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitOpenError(
                        "Circuit is open, request rejected"
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        "Circuit is half-open, max test calls reached"
                    )
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

# 使用示例
payment_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

def process_payment(order):
    try:
        result = payment_breaker.call(payment_service.charge, order)
        return result
    except CircuitOpenError:
        # 降级：加入队列延后处理
        queue.send("payment_retry", order)
        return {"status": "queued", "message": "Payment will be processed shortly"}
```

## 分布式追踪（Distributed Tracing）

在微服务架构中，单个用户请求可能穿越 5–10 个服务。出问题时，你必须能跨所有服务追踪该请求。

![分布式追踪时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-distributed-tracing.png)

### OpenTelemetry

OpenTelemetry 是分布式追踪的行业标准，它通过传播 trace context（trace ID + span ID）贯穿所有服务调用。

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# 初始化追踪
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="jaeger:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 自动注入 HTTP 库追踪
RequestsInstrumentor().instrument()
FlaskInstrumentor().instrument_app(app)

tracer = trace.get_tracer(__name__)

@app.route("/api/orders", methods=["POST"])
def create_order():
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("user_id", request.json["user_id"])

        # 校验订单
        with tracer.start_as_current_span("validate_order"):
            validate(request.json)

        # 检查库存（跨服务调用 —— trace context 自动透传）
        with tracer.start_as_current_span("check_inventory"):
            inventory_response = requests.get(
                f"http://inventory-service/api/check",
                json={"items": request.json["items"]},
            )

        # 处理支付（跨服务调用）
        with tracer.start_as_current_span("process_payment"):
            payment_response = requests.post(
                f"http://payment-service/api/charge",
                json={"amount": request.json["total"]},
            )

        return jsonify({"order_id": order_id}), 201
```

该 trace 将在 Jaeger/Zipkin 中呈现为时间轴，清晰展示每个 Span 的耗时及执行服务。当支付服务变慢时，你可精准定位哪个 Span 耗时异常。

## API 网关（API Gateway）

API 网关位于外部客户端与内部服务之间，提供统一入口点。

![API 网关模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-api-gateway.png)

### 网关职责

```text
客户端 → API 网关 → 内部服务

网关承担以下职责：
  1. 路由（Routing）：/api/orders/* → 订单服务  
  2. 认证（Authentication）：校验 JWT，拒绝未授权请求  
  3. 限流（Rate limiting）：按客户端施加速率限制  
  4. TLS 终止（TLS termination）：处理 HTTPS，内部流量可走 HTTP  
  5. 请求聚合（Request aggregation）：将多次服务调用合并为单次响应  
  6. 协议转换（Protocol translation）：对外接受 REST，对内使用 gRPC  
  7. 缓存（Caching）：缓存高频、低频变更数据  
  8. 日志与指标（Logging/metrics）：集中式请求日志与监控  
```

### 使用 Nginx 作为 API 网关

```nginx
upstream orders_service {
    server orders-1:8080;
    server orders-2:8080;
}

upstream users_service {
    server users-1:8080;
    server users-2:8080;
}

# 限流区域
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # 限流
    limit_req zone=api burst=200 nodelay;
    limit_req_status 429;

    # JWT 校验（使用 nginx-jwt 模块或 Lua）
    location /api/ {
        # 认证
        access_by_lua_block {
            local jwt = require("resty.jwt")
            local token = ngx.req.get_headers()["Authorization"]
            if not token then
                ngx.status = 401
                ngx.say('{"error": "Missing authorization"}')
                return ngx.exit(401)
            end
            -- 校验 JWT ...
        }

        # 路由至服务
        location /api/orders/ {
            proxy_pass http://orders_service;
        }

        location /api/users/ {
            proxy_pass http://users_service;
        }
    }
}
```

## 微服务中的数据管理（Data Management in Microservices）

### 每服务一库（Database Per Service）

每个服务**独占其数据**，其他服务不得直接读写其数据库。

```text
订单服务  → orders_db  （PostgreSQL）  
用户服务  → users_db   （PostgreSQL）  
搜索服务  → search_idx （Elasticsearch）  
缓存服务  → cache_db   （Redis）  
```

**后果**：无法跨服务 JOIN。若订单服务需用户信息，必须调用用户服务 API——这增加了延迟与复杂度，但**强制实现了松耦合**。

### 共享数据库反模式（The Shared Database Anti-Pattern）

多个服务共享单一数据库看似便捷，实则极具破坏性：

- 任一服务均可读写任意表，形成隐式依赖  
- 表结构变更需协调所有关联服务  
- 某服务的重型查询会拖垮其他服务性能  
- 若服务共享数据库迁移脚本，则无法真正独立部署  

### 数据一致性：Saga 模式（The Saga Pattern）

缺乏分布式事务时，跨服务一致性需依赖 Saga——即由事件或编排驱动的一系列本地事务。

**基于编排的 Saga（Choreography-based saga）**（事件驱动）：

```text
1. 订单服务：创建订单（状态：PENDING）  
   → 发布 OrderCreated 事件  

2. 支付服务：为客户扣款  
   → 成功：发布 PaymentCompleted  
   → 失败：发布 PaymentFailed  
     → 订单服务：取消订单（补偿动作）  

3. 库存服务：预留库存  
   → 成功：发布 InventoryReserved  
     → 订单服务：确认订单（状态：CONFIRMED）  
   → 失败：发布 OutOfStock  
     → 支付服务：为客户退款（补偿动作）  
     → 订单服务：取消订单（补偿动作）  
```

各服务发布事件并响应事件，**无中心协调者**。短流程简单，但复杂工作流难以追踪。

**基于编排的 Saga（Orchestration-based saga）**（中心协调者）：

```python
class OrderSaga:
    """订单处理的中心协调器。"""

    def execute(self, order):
        try:
            # 步骤 1：预留支付
            payment = payment_service.reserve(order.total)

            # 步骤 2：预留库存
            try:
                inventory = inventory_service.reserve(order.items)
            except OutOfStockError:
                payment_service.release(payment.id)  # 补偿
                raise

            # 步骤 3：确认订单
            order_service.confirm(order.id)
            payment_service.capture(payment.id)

        except Exception as e:
            # 补偿所有已完成步骤
            self._compensate(order)
            raise
```

## 决策框架（Decision Framework）

### 何时坚持单体架构

- 团队规模小于 20 名工程师  
- 应用领域已被充分理解且稳定  
- 需强一致性保障（金融系统、库存管理）  
- 你是初创公司，迭代速度比可扩展性更重要  
- 尚不具备分布式系统的运维成熟度（监控、追踪、值班体系）  

### 何时考虑微服务

- 团队规模超 30–50 人，且存在清晰的团队所有权边界  
- 系统不同部分的伸缩需求差异巨大  
- 需技术多样性（如 ML 模型服务 vs CRUD API）  
- 多支团队需独立部署，避免协调成本  
- 已具备支撑分布式系统的运维基础设施  

### 抽取模式（The Extraction Pattern）

最稳妥的路径是：**从单体起步，待痛点明确后再抽取服务**：

1. **构建结构良好的单体**，明确模块边界  
2. **识别抽取候选模块**：伸缩需求最独立、部署最频繁、团队归属最清晰的模块  
3. **定义 API 边界**：该模块向系统其余部分暴露什么接口？  
4. **执行抽取**：将模块部署为独立服务，将内部函数调用替换为 API 调用  
5. **验证效果**：确认抽取确实解决了预设问题  
6. **重复迭代**：待下一个痛点浮现，再抽取下一个服务  

此方法避免过早拆分。你只为系统中**真正需要微服务特性的部分**支付“分布式开销”。

## 下一步

无论你的系统是单体还是微服务，数据总以特定模式流经其中。下一篇将探讨**数据管道（Data Pipelines）**——批处理、流处理，以及将原始数据转化为可操作洞察的架构模式。
