---
title: "System Design (6): Microservices vs Monoliths — The Honest Tradeoff"
date: 2025-07-22 09:00:00
tags:
  - System Design
  - Microservices
  - Architecture
  - Distributed Systems
categories: System Design
series: system-design
lang: en
description: "An honest assessment of monoliths vs microservices — covering the distributed systems tax, service boundary design with DDD, inter-service communication patterns, circuit breakers, service mesh, API gateways, and a practical decision framework."
disableNunjucks: true
series_order: 6
translationKey: "system-design-6"
---

In 2020, the team behind Segment — a customer data platform processing billions of events per month — published a blog post titled "Goodbye Microservices." They had decomposed their monolith into over 140 microservices, and the result was not the engineering utopia they expected. Instead, they spent most of their time fighting the complexity of the distributed system itself: service discovery failures, cascading timeouts, inconsistent deployment pipelines, and an explosion of inter-service communication bugs. They consolidated back to a monolith and reported dramatic improvements in developer productivity and system reliability.

This story is not unique. The microservices pattern has become the default architectural assumption in the industry, but the honest truth is that it is the wrong choice for most teams. Understanding when microservices help and when they hurt is one of the most important judgment calls in system design.

---

## The Monolith

A monolith is a single deployable unit that contains all the application's functionality. The entire codebase is compiled and deployed together. All modules share the same process, the same memory space, and the same database.

![Monolith vs microservices](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-monolith-vs-micro.png)


### Why Monoliths Work

**Simplicity**: One codebase, one deployment pipeline, one database, one set of logs. A new developer can clone the repo, run `docker-compose up`, and have the entire system running locally in minutes.

**Performance**: Function calls within a process take nanoseconds. There is no network serialization, no HTTP overhead, no service discovery. A monolith can handle remarkably high throughput with minimal infrastructure.

**Transactional consistency**: Because all modules share a database, you get ACID transactions across the entire domain. Deducting inventory and recording a sale happen in a single database transaction — no distributed coordination required.

**Refactoring**: Moving code between modules is a simple refactor. In a microservices world, moving functionality between services requires changing APIs, updating consumers, managing data migration, and coordinating deployments.

### When Monoliths Struggle

**Team scaling**: When 50+ engineers work on the same codebase, merge conflicts become constant, deploy queues get long, and one team's bug breaks another team's feature.

**Scaling bottlenecks**: If one module needs 10x more compute than the rest, you must scale the entire monolith 10x. You cannot independently scale the hot path.

**Technology lock-in**: The entire application uses the same language, framework, and database. If one module would benefit from a different technology, you cannot adopt it without affecting everything else.

**Blast radius**: A memory leak in one module crashes the entire process. A bad deploy affects all functionality simultaneously.

### The Modular Monolith

Before jumping to microservices, consider the modular monolith — a single deployable unit with strict internal module boundaries.

```text
project/
├── modules/
│   ├── orders/
│   │   ├── api.py          # Public API (other modules call this)
│   │   ├── service.py      # Business logic
│   │   ├── repository.py   # Data access
│   │   └── models.py       # Domain models
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

Rules:
1. Modules communicate only through their public APIs (no reaching into another module's internals)
2. Each module owns its own database tables (no cross-module table access)
3. Cross-module data access goes through the module's API
4. Module boundaries can be enforced by linting rules or architecture tests

```python
# Architecture test: enforce module boundaries
import ast
import os

def check_module_boundaries():
    """Verify no module directly imports another module's internals."""
    violations = []

    for module_dir in os.listdir("modules"):
        for py_file in glob(f"modules/{module_dir}/**/*.py"):
            tree = ast.parse(open(py_file).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("modules."):
                        imported_module = node.module.split(".")[1]
                        if imported_module != module_dir:
                            # Cross-module import — only allowed from api.py
                            imported_file = node.module.split(".")[-1]
                            if imported_file != "api":
                                violations.append(
                                    f"{py_file} imports {node.module} "
                                    f"(should only import from "
                                    f"modules.{imported_module}.api)"
                                )

    return violations
```

The modular monolith gives you most of the organizational benefits of microservices (clear ownership, independent development) without the distributed systems tax.

## The Microservices Architecture

Microservices decompose the application into independently deployable services, each running in its own process and communicating over the network.

### What You Gain

**Independent deployability**: The orders team can deploy 10 times a day without coordinating with the payments team. Each service has its own CI/CD pipeline, its own release cycle, and its own rollback strategy.

**Technology diversity**: The orders service can use Python, the recommendation engine can use Go for performance, and the search service can use Elasticsearch. Each team picks the best tool for their domain.

**Team autonomy**: Each team owns a service end-to-end: code, tests, deployment, monitoring, and on-call. They make decisions independently. Conway's Law works in your favor — the architecture mirrors the organization.

**Independent scaling**: The search service gets 100x more traffic during Black Friday. You scale it to 50 instances while keeping the orders service at 5 instances.

**Fault isolation**: A memory leak in the notification service does not crash the payment service. Failures are contained within service boundaries (if you implement proper isolation patterns).

### The Distributed Systems Tax

Every benefit comes with a cost. Microservices introduce distributed systems problems that do not exist in a monolith.

**Network latency**: A function call takes nanoseconds. A network call takes milliseconds. If a user request touches 5 services in sequence, you add 5-50ms of network latency even before processing.

**Partial failures**: In a monolith, either the process is up or it is down. In a distributed system, Service A might be up while Service B is down. Every service call needs timeout handling, retry logic, and fallback strategies.

**Data consistency**: Without a shared database, maintaining consistency across services requires distributed coordination (sagas, eventual consistency, compensating transactions). This is fundamentally harder than a database transaction.

**Operational complexity**: Instead of monitoring one application, you monitor 20+. Each has its own logs, metrics, alerts, deployment pipeline, and failure modes. You need centralized logging, distributed tracing, and service mesh infrastructure.

**Testing complexity**: Integration tests must spin up multiple services. End-to-end tests are slow and flaky. Contract testing becomes essential.

**Debugging difficulty**: A bug that would be a single stack trace in a monolith becomes a detective story across multiple services, logs, and network traces.

## Service Boundaries: Domain-Driven Design


![Circuit breaker pattern electrical circuit breaker protectin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/06-circuit-breaker-pattern-electrical-circuit-breaker-protectin.jpg)

The hardest part of microservices is deciding where to draw the boundaries. Get them wrong, and you end up with chatty services that need to call each other constantly, or services that are too granular to be useful.

Domain-Driven Design (DDD) provides a principled approach.

### Bounded Contexts

A bounded context is a boundary within which a particular domain model is defined and applicable. Within a bounded context, terms have precise, unambiguous meanings. Across bounded contexts, the same term may mean different things.

![DDD bounded contexts](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-ddd-bounded-contexts.png)


Example: In an e-commerce system, "Order" means different things in different contexts:

- **Sales context**: An Order is a customer's intent to purchase items, with pricing, discounts, and payment terms
- **Fulfillment context**: An Order is a set of items that need to be picked, packed, and shipped from a warehouse
- **Accounting context**: An Order is a financial transaction with revenue recognition, tax calculations, and ledger entries

Each bounded context is a natural candidate for a service boundary. The key insight: services should be organized around business capabilities, not technical layers.

```text
Bad (technical layers — every request crosses all services):
  API Gateway → Authentication Service → Business Logic Service → Data Service

Good (business capabilities — most requests stay within one service):
  Orders Service (own API, logic, and data)
  Payments Service (own API, logic, and data)
  Inventory Service (own API, logic, and data)
  Shipping Service (own API, logic, and data)
```

### Context Mapping

Services need to communicate, and the boundaries between them need explicit contracts. DDD defines several relationship patterns:

**Published Language**: Services agree on a shared data format for events or API contracts (protobuf schemas, JSON Schema, OpenAPI specs).

**Anti-Corruption Layer**: When integrating with a legacy or external system, build a translation layer that converts the external model to your internal model. This prevents the external system's design decisions from leaking into your codebase.

```python
# Anti-corruption layer for a legacy payment gateway
class PaymentGatewayAdapter:
    """Translates between our domain model and the legacy gateway's API."""

    def __init__(self, legacy_client):
        self.client = legacy_client

    def charge(self, amount: Decimal, currency: str, customer_id: str) -> PaymentResult:
        # Legacy API uses cents, not dollars
        # Legacy API uses "cust_num" not "customer_id"
        # Legacy API returns "rc" (return code) not structured errors
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
                error_message=response.get("msg", "Unknown error"),
            )
```

## Inter-Service Communication


![Monolith to microservices evolution single building to city](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/06-monolith-to-microservices-evolution-single-building-to-city-.jpg)

### Synchronous: REST and gRPC

Use synchronous communication when the caller needs an immediate response.

```python
# REST client with timeout, retry, and circuit breaker
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_client() -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,           # 0.5s, 1s, 2s between retries
        status_forcelist=[502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET", "PUT", "DELETE"],  # Only retry idempotent methods
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
        timeout=(3, 10),  # (connect timeout, read timeout)
    )
    response.raise_for_status()
    return response.json()
```

### Asynchronous: Events and Messages

Use asynchronous communication when the caller does not need an immediate response, or when you need to fan out to multiple consumers.

The previous article on message queues covers this in detail. The key design choice: use events for facts ("OrderCreated") and commands for instructions ("ProcessPayment").

## The Circuit Breaker Pattern

When a downstream service is failing, continuing to send requests is wasteful and can cascade the failure. A circuit breaker stops the bleeding.

![Circuit breaker state transitions animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/sysdesign-06-circuit-breaker.gif)


![Circuit breaker state machine](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-circuit-breaker.png)


### States

The circuit breaker has three states:

**Closed** (normal): Requests pass through. Failures are counted. If failures exceed a threshold, the circuit opens.

**Open** (failing): Requests are immediately rejected without calling the downstream service. After a timeout, the circuit moves to half-open.

**Half-Open** (testing): A limited number of requests are sent to the downstream service. If they succeed, the circuit closes. If they fail, the circuit opens again.

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

# Usage
payment_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

def process_payment(order):
    try:
        result = payment_breaker.call(payment_service.charge, order)
        return result
    except CircuitOpenError:
        # Fallback: queue for later processing
        queue.send("payment_retry", order)
        return {"status": "queued", "message": "Payment will be processed shortly"}
```

## Distributed Tracing

In a microservices architecture, a single user request may traverse 5-10 services. When something goes wrong, you need to trace the request across all of them.

![Distributed tracing timeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-distributed-tracing.png)


### OpenTelemetry

OpenTelemetry is the industry standard for distributed tracing. It propagates a trace context (trace ID + span ID) through all service calls.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Initialize tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="jaeger:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Auto-instrument HTTP libraries
RequestsInstrumentor().instrument()
FlaskInstrumentor().instrument_app(app)

tracer = trace.get_tracer(__name__)

@app.route("/api/orders", methods=["POST"])
def create_order():
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("user_id", request.json["user_id"])

        # Validate order
        with tracer.start_as_current_span("validate_order"):
            validate(request.json)

        # Check inventory (cross-service call — trace context propagated automatically)
        with tracer.start_as_current_span("check_inventory"):
            inventory_response = requests.get(
                f"http://inventory-service/api/check",
                json={"items": request.json["items"]},
            )

        # Process payment (cross-service call)
        with tracer.start_as_current_span("process_payment"):
            payment_response = requests.post(
                f"http://payment-service/api/charge",
                json={"amount": request.json["total"]},
            )

        return jsonify({"order_id": order_id}), 201
```

The trace appears in Jaeger/Zipkin as a timeline showing each span, its duration, and which service executed it. When the payment service is slow, you see exactly which span took too long.

## API Gateway

An API gateway sits between external clients and internal services, providing a single entry point.

![API gateway pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/06-api-gateway.png)


### Gateway Responsibilities

```text
Client → API Gateway → Internal Services

The gateway handles:
  1. Routing: /api/orders/* → Orders Service
  2. Authentication: Validate JWT, reject unauthorized requests
  3. Rate limiting: Enforce per-client rate limits
  4. TLS termination: Handle HTTPS, internal traffic can be HTTP
  5. Request aggregation: Combine multiple service calls into one response
  6. Protocol translation: Accept REST from clients, use gRPC internally
  7. Caching: Cache frequently-requested, slowly-changing data
  8. Logging/metrics: Centralized request logging and monitoring
```

### Nginx as an API Gateway

```nginx
upstream orders_service {
    server orders-1:8080;
    server orders-2:8080;
}

upstream users_service {
    server users-1:8080;
    server users-2:8080;
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # Rate limiting
    limit_req zone=api burst=200 nodelay;
    limit_req_status 429;

    # JWT validation (using nginx-jwt module or lua)
    location /api/ {
        # Authenticate
        access_by_lua_block {
            local jwt = require("resty.jwt")
            local token = ngx.req.get_headers()["Authorization"]
            if not token then
                ngx.status = 401
                ngx.say('{"error": "Missing authorization"}')
                return ngx.exit(401)
            end
            -- Validate JWT ...
        }

        # Route to services
        location /api/orders/ {
            proxy_pass http://orders_service;
        }

        location /api/users/ {
            proxy_pass http://users_service;
        }
    }
}
```

## Data Management in Microservices

### Database Per Service

Each service owns its data exclusively. No other service can read from or write to its database directly.

```text
Orders Service  → orders_db  (PostgreSQL)
Users Service   → users_db   (PostgreSQL)
Search Service  → search_idx (Elasticsearch)
Cache Service   → cache_db   (Redis)
```

**Consequence**: You cannot JOIN across service boundaries. If the Orders Service needs user information, it must call the Users Service API. This adds latency and complexity but enforces loose coupling.

### The Shared Database Anti-Pattern

Multiple services sharing a single database is tempting but destructive:

- Any service can read/write any table, creating hidden dependencies
- Schema changes require coordinating all services that touch the affected tables
- One service's heavy query can degrade another service's performance
- You cannot deploy services independently if they share schema migration

### Data Consistency: The Saga Pattern

Without distributed transactions, maintaining consistency across services requires sagas — a sequence of local transactions coordinated by events or orchestration.

**Choreography-based saga** (event-driven):

```text
1. Order Service: Create order (status: PENDING)
   → Publish OrderCreated event

2. Payment Service: Charge customer
   → Success: Publish PaymentCompleted
   → Failure: Publish PaymentFailed
     → Order Service: Cancel order (compensating action)

3. Inventory Service: Reserve stock
   → Success: Publish InventoryReserved
     → Order Service: Confirm order (status: CONFIRMED)
   → Failure: Publish OutOfStock
     → Payment Service: Refund customer (compensating action)
     → Order Service: Cancel order (compensating action)
```

Each service publishes events and reacts to events. No central coordinator. Simple for short sagas but becomes hard to follow for complex workflows.

**Orchestration-based saga** (central coordinator):

```python
class OrderSaga:
    """Central coordinator for order processing."""

    def execute(self, order):
        try:
            # Step 1: Reserve payment
            payment = payment_service.reserve(order.total)

            # Step 2: Reserve inventory
            try:
                inventory = inventory_service.reserve(order.items)
            except OutOfStockError:
                payment_service.release(payment.id)  # Compensate
                raise

            # Step 3: Confirm order
            order_service.confirm(order.id)
            payment_service.capture(payment.id)

        except Exception as e:
            # Compensate all completed steps
            self._compensate(order)
            raise
```

## Decision Framework

### When to Stay Monolithic

- Team size is under 20 engineers
- The application domain is well-understood and stable
- You need strong consistency (financial systems, inventory management)
- You are a startup and speed of iteration matters more than scalability
- You do not have operational maturity for distributed systems (monitoring, tracing, on-call)

### When to Consider Microservices

- Team size exceeds 30-50 engineers, with clear team ownership boundaries
- Different parts of the system have vastly different scaling requirements
- You need technology diversity (ML model serving vs. CRUD APIs)
- Multiple teams need to deploy independently without coordination
- You have the operational infrastructure to support distributed systems

### The Extraction Pattern

The safest approach is to start with a monolith and extract services when pain is clear:

1. **Build a well-structured monolith** with clear module boundaries
2. **Identify the extraction candidate**: the module that has the most independent scaling needs, the most frequent deployments, or the most distinct team ownership
3. **Define the API boundary**: what does this module expose to the rest of the system?
4. **Extract**: deploy the module as a separate service, replace internal function calls with API calls
5. **Verify**: confirm the extraction solves the problem it was intended to solve
6. **Repeat**: extract the next service when the next pain point emerges

This approach avoids premature decomposition. You only pay the microservices tax for parts of the system that genuinely need it.

## What's next

Whether your system is a monolith or microservices, data flows through it in patterns. The next article covers data pipelines — batch processing, stream processing, and the architectural patterns that connect raw data to actionable insights.
