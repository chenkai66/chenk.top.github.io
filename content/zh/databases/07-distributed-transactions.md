---
title: "数据库（七）：分布式事务——两阶段提交、Saga 模式，以及为何共识如此困难"
date: 2024-04-28 09:00:00
tags:
  - Databases
  - Distributed Systems
  - Consensus
  - Transactions
categories: Databases
series: databases
lang: zh
description: "分布式数据库如何跨机器协调事务——两阶段提交（2PC）、Raft 共识、Saga 模式，以及 outbox 和 CDC 等实用模式。"
disableNunjucks: true
series_order: 7
translationKey: "databases-7"
---

第 3 篇文章中关于事务的所有内容，都基于单数据库服务器的假设：一台机器、一份事务日志、一个锁管理器。一旦数据分布到多台机器上——例如实施分片（sharding）、采用微服务架构并为每个服务配置独立数据库，或启用强一致性复制——你就直接面对分布式系统中最棘手的问题：**如何让多台机器就某个值达成一致？**

## 分布式事务问题

考虑一个电商系统，订单服务和库存服务彼此分离，各自拥有独立数据库：

```
Order Service (DB-1)              Inventory Service (DB-2)
┌─────────────────────┐          ┌─────────────────────────┐
│ INSERT INTO orders   │          │ UPDATE products          │
│ (user_id, total)     │          │ SET stock = stock - 1    │
│ VALUES (1, 99.99)    │          │ WHERE product_id = 42    │
└─────────────────────┘          └─────────────────────────┘
```

如果订单插入成功，但库存更新失败（网络故障、约束冲突、进程崩溃），问题就出现了：一笔订单对应的商品从未被预留。若缺乏协调机制，数据不一致将不可避免。

在单数据库中，用 `BEGIN ... COMMIT` 将二者包裹即可解决。但在两个数据库之间，这是不可能的——它们拥有各自独立的事务日志、独立的崩溃恢复机制、独立的时钟。

## 两阶段提交（2PC）

分布式事务的经典教科书解法。由一个协调者节点（coordinator）主导协议，多个参与者节点（participants）配合执行。

![Two-phase commit protocol](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-two-phase-commit.png)


### 协议流程

```
Phase 1: PREPARE (投票阶段)
  Coordinator → Participant A: "Can you commit transaction T?"
  Coordinator → Participant B: "Can you commit transaction T?"
  Participant A → Coordinator: "Yes, I'm prepared" (locks held, WAL flushed)
  Participant B → Coordinator: "Yes, I'm prepared"

Phase 2: COMMIT (决策阶段)
  Coordinator → Participant A: "COMMIT transaction T"
  Coordinator → Participant B: "COMMIT transaction T"
  Participant A → Coordinator: "Done"
  Participant B → Coordinator: "Done"
```

若任一参与者在 Phase 1 投“否”票，则协调者向所有参与者发送 ROLLBACK。

```
Coordinator                   Participant A      Participant B
    │                              │                   │
    ├──── PREPARE ────────────────►│                   │
    ├──── PREPARE ───────────────────────────────────►│
    │                              │                   │
    │◄─── YES (prepared) ─────────┤                   │
    │◄─── YES (prepared) ──────────────────────────────┤
    │                              │                   │
    │  (writes COMMIT to own log)  │                   │
    │                              │                   │
    ├──── COMMIT ─────────────────►│                   │
    ├──── COMMIT ──────────────────────────────────►│
    │                              │                   │
    │◄─── ACK ────────────────────┤                   │
    │◄─── ACK ─────────────────────────────────────────┤
    │                              │                   │
```

### 协调者失效问题（The Coordinator Failure Problem）

2PC 的关键缺陷在于：若协调者在发送 PREPARE 消息后、发送 COMMIT 或 ROLLBACK 消息前崩溃，参与者将陷入僵持状态。它们已投“是”，并持有锁，却无法得知最终决策。

```
Coordinator                   Participant A      Participant B
    │                              │                   │
    ├──── PREPARE ────────────────►│                   │
    ├──── PREPARE ───────────────────────────────────►│
    │                              │                   │
    │◄─── YES ────────────────────┤                   │
    │◄─── YES ─────────────────────────────────────────┤
    │                              │                   │
    ╳ CRASH                        │                   │
                                   │                   │
                               "I voted YES          "I voted YES
                                but don't know        but don't know
                                the decision.         the decision.
                                Can't commit.         Can't commit.
                                Can't rollback.       Can't rollback.
                                Locks held..."        Locks held..."
```

这被称为 **阻塞问题（blocking problem）**。参与者必须等待（可能无限期）协调者恢复并揭示其决策。实践中这意味着：
- 锁被无限期持有，阻塞其他事务；
- 可能需要人工干预；
- 该协议不具备容错能力。

### 2PC 在实践中的应用

尽管存在局限性，2PC 仍在实际系统中被使用：

```sql
-- PostgreSQL: 预处理事务（2PC 参与者）
-- 应用程序/协调者调用以下命令：

-- Phase 1: Prepare
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
PREPARE TRANSACTION 'transfer_001_debit';

-- Phase 2: Commit（若所有参与者均成功 prepare）
COMMIT PREPARED 'transfer_001_debit';

-- 或回滚
ROLLBACK PREPARED 'transfer_001_debit';

-- 检查孤立的预处理事务（卡住的 2PC）
SELECT gid, prepared, owner, database
FROM pg_prepared_xacts;
```

```java
// Java XA 事务（JTA）—— 2PC 的标准 API
UserTransaction ut = (UserTransaction) ctx.lookup("java:comp/UserTransaction");
ut.begin();

// 将两个不同数据库纳入同一事务
Connection conn1 = ds1.getConnection();  // 订单数据库
Connection conn2 = ds2.getConnection();  // 库存数据库

conn1.prepareStatement("INSERT INTO orders ...").execute();
conn2.prepareStatement("UPDATE inventory SET stock = stock - 1 ...").execute();

ut.commit();  // 事务管理器运行 2PC 协议
```

## 三阶段提交（3PC）


![Distributed consensus protocol servers voting in a digital p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-distributed-consensus-protocol-servers-voting-in-a-digital-p.jpg)

3PC 在 PREPARE 和 COMMIT 之间增加了一个 PRE-COMMIT 阶段，使参与者可在协调者失效时自主恢复：

```
Phase 1: CAN-COMMIT?  → 参与者检查自身是否可提交
Phase 2: PRE-COMMIT   → 协调者通知参与者准备（但暂不提交）
Phase 3: DO-COMMIT    → 最终提交

若协调者在 PRE-COMMIT 后崩溃：
  参与者可超时后直接提交（因已知所有节点均已就绪）
```

理论上，3PC 是非阻塞的。但实践中极少使用，原因如下：
- 网络分区仍可能导致不一致（某参与者可能未收到 PRE-COMMIT）；
- 额外的一次往返增加了延迟；
- Raft/Paxos 等共识协议更稳健地解决了该问题。

## 共识算法（Consensus Algorithms）

共识问题是：即使部分节点发生故障，多个节点仍需就某个值达成一致。它是强一致性分布式数据库的基石。

### Paxos（概念性）

Paxos（Leslie Lamport 于 1989 年提出）是首个被严格证明正确的共识算法。它定义了三个角色：
- **提议者（Proposers）**：提出值；
- **接受者（Acceptors）**：对提议进行投票；
- **学习者（Learners）**：获知最终被选定的值。

Single-Decree Paxos 的简化视图如下：

```
Phase 1: Prepare
  Proposer → Acceptors: "Prepare(proposal_number=5)"
  Acceptors → Proposer: "Promise: I won't accept proposals < 5"
                          + "Last accepted: (proposal=3, value='X')" if any

Phase 2: Accept
  Proposer → Acceptors: "Accept(proposal=5, value='Y')"
  Acceptors → Proposer: "Accepted" (if no higher proposal seen)
  
When a majority of acceptors accept → value is decided
```

Paxos 在理论上正确，但以难以实现而闻名。正如 Lamport 所言，社区花了数年才真正理解他的论文。这种复杂性催生了 Raft。

### Raft：可理解的共识协议

Raft（2014 年，Diego Ongaro 和 John Ousterhout 提出）旨在提供与 Paxos 等价但更易理解的共识方案。它将共识分解为三个子问题：

![Raft leader election](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-raft-election.png)


1. **领导者选举（Leader election）**
2. **日志复制（Log replication）**
3. **安全性（Safety）**

#### 领导者选举

每个节点初始为 **追随者（Follower）**。若追随者在随机超时（如 150–300 ms）内未收到来自领导者的心跳，则转变为 **候选人（Candidate）** 并发起选举。

```
Node States:
  Follower  → times out, no heartbeat  → Candidate
  Candidate → receives majority votes  → Leader
  Candidate → discovers higher-term leader → Follower
  Leader    → discovers higher-term leader → Follower

Election Process:
  1. Candidate increments its term number
  2. Votes for itself
  3. Sends RequestVote RPCs to all other nodes
  4. If majority responds with vote: becomes Leader
  5. Starts sending periodic heartbeats to prevent new elections
```

```
Term 1: Node A is leader
         Node A ──heartbeat──► Node B
         Node A ──heartbeat──► Node C

Term 2: Node A crashes. Node B times out.
         Node B: "I'm a candidate for term 2. Vote for me."
         Node C: "OK, you have my vote for term 2."
         Node B: Now leader. Starts sending heartbeats.
```

#### 日志复制

领导者当选后，接收客户端请求并追加至本地日志，再将日志条目复制给追随者：

```
Leader Log:   [term1:SET x=1] [term1:SET y=2] [term2:SET x=3]
                    │                │               │
                    ▼                ▼               ▼
Follower B:   [term1:SET x=1] [term1:SET y=2] [term2:SET x=3]  ✓ up to date
Follower C:   [term1:SET x=1] [term1:SET y=2]                  ✗ catching up

A log entry is "committed" when replicated to a majority of nodes.
The leader then applies the entry to its state machine and responds to the client.
```

```
Client ─── "SET x=3" ──► Leader
                          │
                 1. Append to log
                 2. Send AppendEntries RPC to followers
                          │
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                 Foll.B  Foll.C  Foll.D
                    │      │      │
                    └──────┼──────┘
                 3. Wait for majority acknowledgment
                          │
                 4. Commit entry
                 5. Apply to state machine
                 6. Respond to client: "OK"
```

#### Raft 的实际应用

| 系统 | Raft 使用场景 |
|--------|-------------|
| etcd | 键值存储（Kubernetes 后端存储） |
| CockroachDB | 每个 range（分区）使用独立的 Raft 组 |
| TiKV | TiDB 的存储层 |
| Consul | 服务发现与配置管理 |
| RethinkDB | 集群协调 |

## Saga 模式

当 2PC 成本过高或不切实际（这在微服务架构中几乎总是如此）时，Saga 模式提供了替代方案。它不依赖单一的、跨服务的分布式事务，而是将业务流程拆解为一系列本地事务；每个本地事务都对应一个 **补偿事务（compensating transaction）**，以便在后续步骤失败时回退其影响。

![Saga pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-saga-pattern.png)


### 编排（Orchestration） vs. 协作（Choreography）

**协作式（Choreography）**：各服务发布事件，下游服务监听并响应。

```
Order Service              Inventory Service          Payment Service
     │                          │                          │
     │ OrderCreated event ──────►                          │
     │                          │                          │
     │                    InventoryReserved event ─────────►
     │                          │                          │
     │                          │              PaymentProcessed event
     │◄─────────────────────────────────────────────────────┤
     │                          │                          │
     │ OrderConfirmed           │                          │
```

**编排式（Orchestration）**：由中心化协调器（orchestrator）依次指挥各服务。

```
Saga Orchestrator
     │
     ├──── "Create order" ──────► Order Service
     │◄─── "Order created" ──────┤
     │
     ├──── "Reserve inventory" ──► Inventory Service
     │◄─── "Inventory reserved" ──┤
     │
     ├──── "Process payment" ────► Payment Service
     │◄─── "Payment failed!" ─────┤
     │
     │  COMPENSATION:
     ├──── "Release inventory" ──► Inventory Service
     │◄─── "Inventory released" ──┤
     │
     ├──── "Cancel order" ──────► Order Service
     │◄─── "Order cancelled" ─────┤
```

### 补偿事务（Compensating Transactions）

每个正向操作都需配套一个补偿操作：

| 步骤 | 正向操作 | 补偿操作 |
|------|---------------|-------------------|
| 1 | 创建订单（状态：pending） | 取消订单（状态：cancelled） |
| 2 | 预留库存（stock - 1） | 释放库存（stock + 1） |
| 3 | 扣款支付 | 退款 |
| 4 | 发货 | 取消发货 |

```python
# Saga 编排器伪代码
class OrderSaga:
    steps = [
        SagaStep(
            action=lambda ctx: order_service.create_order(ctx.user_id, ctx.items),
            compensation=lambda ctx: order_service.cancel_order(ctx.order_id)
        ),
        SagaStep(
            action=lambda ctx: inventory_service.reserve(ctx.items),
            compensation=lambda ctx: inventory_service.release(ctx.items)
        ),
        SagaStep(
            action=lambda ctx: payment_service.charge(ctx.user_id, ctx.total),
            compensation=lambda ctx: payment_service.refund(ctx.payment_id)
        ),
    ]

    def execute(self, context):
        completed = []
        for step in self.steps:
            try:
                result = step.action(context)
                completed.append(step)
            except Exception as e:
                # 按逆序执行补偿
                for completed_step in reversed(completed):
                    completed_step.compensation(context)
                raise SagaFailedError(f"Step failed: {e}")
```

## 线性一致性（Linearizability） vs. 可串行化（Serializability）

这两个术语常被混淆，但描述的是完全不同的保证：

**可串行化（Serializability）**（来自事务）：并发执行多个事务的结果，等价于这些事务以某种**串行顺序**执行的结果。它关注的是事务与数据库层面的正确性。

**线性一致性（Linearizability）**（来自分布式系统）：每个操作看起来都在其调用与完成之间的某个瞬时点原子生效；一旦写入被确认，所有后续读取都必须看到该值。它关注的是**单个操作**与实时顺序。

```
线性一致系统（寄存器初始值为 0）：

Client A: write(1)  ─────────────────► OK
                                        │ （从此刻起，所有读取必须返回 1）
Client B:              read() ──────────► 1  ✓
Client C:                     read() ──► 1   ✓

非线性一致：
Client A: write(1)  ─────────────────► OK
Client B:              read() ──────────► 0  ✗ （陈旧！）
Client C:                     read() ──► 1   ✓
```

| 属性 | 可串行化（Serializability） | 线性一致性（Linearizability） |
|----------|----------------|-----------------|
| 范围 | 多操作事务 | 单个操作 |
| 排序要求 | 存在某种串行顺序（任意顺序均可） | 实时顺序（real-time order） |
| 关键场景 | 数据库 | 分布式键值存储、分布式锁 |
| 示例系统 | 任何 SERIALIZABLE 隔离级别的数据库 | ZooKeeper、etcd、Spanner |

**严格可串行化（Strict serializability）** = 可串行化 + 线性一致性。这是最强的一致性保证，也是 Google Spanner 提供的保证。

## 最终一致性（Eventual Consistency）


![Saga pattern as a chain of compensating transactions domino](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-saga-pattern-as-a-chain-of-compensating-transactions-domino-.jpg)

在线性一致性的另一端是最终一致性：若不再有新的写入，所有副本将**最终**收敛到相同值。

![Consistency spectrum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-consistency-spectrum.png)


“最终”没有明确定义——收敛时间可能短至毫秒，也可能长达数分钟。实践中：

```
向 Node A 写入 "x = 5"
  t=0ms:  Node A: x=5,  Node B: x=3,  Node C: x=3
  t=50ms: Node A: x=5,  Node B: x=5,  Node C: x=3
  t=100ms: Node A: x=5, Node B: x=5,  Node C: x=5  ← 收敛完成
```

默认采用最终一致性的系统包括：
- DynamoDB（除非显式请求强一致性读取）；
- Cassandra（一致性级别设为 ONE）；
- DNS；
- CDN 缓存。

它适用于以下场景：
- 陈旧数据可容忍（社交媒体动态流、商品推荐）；
- 系统能检测并解决冲突（购物车、CRDT）；
- 性能比一致性更重要（分析、日志）。

## 真实世界常用模式

### Outbox 模式

如何**原子性地**更新数据库 *并* 向消息中间件（如 Kafka）发布消息？你无法在数据库与 Kafka 之间使用分布式事务。

Outbox 模式：将消息写入数据库内的一个 “outbox” 表（与业务更新在同一事务中）。再由独立进程读取该表，并向消息中间件发布。

```sql
-- 单数据库事务（原子性！）
BEGIN;

-- 业务逻辑
INSERT INTO orders (user_id, total, status)
VALUES (1, 99.99, 'created');

-- Outbox 条目（同事务、同数据库）
INSERT INTO outbox (
    aggregate_type,
    aggregate_id,
    event_type,
    payload,
    created_at
) VALUES (
    'Order',
    currval('orders_order_id_seq'),
    'OrderCreated',
    '{"user_id": 1, "total": 99.99}',
    NOW()
);

COMMIT;
```

一个独立的发布进程（或 Debezium + CDC）持续读取 outbox 表并向 Kafka 发布事件：

```python
# Outbox 发布器（持续运行）
while True:
    rows = db.query("""
        SELECT id, event_type, payload
        FROM outbox
        WHERE published_at IS NULL
        ORDER BY created_at
        LIMIT 100
    """)

    for row in rows:
        kafka.produce(
            topic=f"events.{row.event_type}",
            value=row.payload
        )
        db.execute(
            "UPDATE outbox SET published_at = NOW() WHERE id = %s",
            row.id
        )
```

### 变更数据捕获（CDC）

不主动写 outbox 表，而是直接从数据库的事务日志（WAL / Binlog）中捕获变更：

```
Database WAL/Binlog ──► CDC Tool (Debezium) ──► Kafka ──► Consumers

PostgreSQL WAL → Debezium → Kafka topic "db.public.orders"
MySQL Binlog   → Debezium → Kafka topic "db.inventory.products"
```

```json
// Debezium CDC 事件（来自 PostgreSQL）
{
  "before": null,
  "after": {
    "order_id": 1234,
    "user_id": 1,
    "total": 99.99,
    "status": "created"
  },
  "source": {
    "version": "2.4.0",
    "connector": "postgresql",
    "name": "orders-db",
    "ts_ms": 1702656000000,
    "db": "ecommerce",
    "schema": "public",
    "table": "orders"
  },
  "op": "c",  // c=create, u=update, d=delete, r=read (snapshot)
  "ts_ms": 1702656000123
}
```

CDC 相较于 Outbox 模式的优点：
- 无需修改应用代码或数据库 schema；
- 捕获所有变更，而非仅靠人工“记得”写 outbox 的那些；
- 更低延迟（直接读 WAL）；
- 规避双写风险（dual-write risk）。

## 何时应避免分布式事务？

**最好的分布式事务，就是你根本不需要的那个。** 替代策略包括：

1. **将需事务协同的数据保留在同一节点**：设计分片键（partition key），使关联数据共置（co-locate）。

2. **接受最终一致性**：许多业务流程天然异步（邮件、通知、分析）。

3. **使用幂等操作（Idempotent operations）**：设计操作使其重试安全。

```sql
-- 幂等插入（PostgreSQL）
INSERT INTO processed_events (event_id, processed_at)
VALUES ('evt-123', NOW())
ON CONFLICT (event_id) DO NOTHING;
-- 安全重试——重复插入会被静默忽略
```

4. **面向补偿设计（Design for compensation）**：不追求预防不一致，而是在事后检测并修复。银行正是这样做的——每日运行对账（reconciliation）流程。

5. **使用单数据库**：若你的微服务共享同一数据库（虽属“异端”，但务实），则可直接使用常规事务。

## 下一篇预告

我们已覆盖理论：数据如何存储、查询、复制、分片与事务处理。但仅有理论远远不够。在最后一篇中，我们将走向实战：**生产环境中的数据库**——迁移、监控、连接池、备份、容量规划，以及来自真实线上事故的“战争故事”。