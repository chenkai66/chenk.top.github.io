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
series_total: 8
translationKey: "databases-7"
---

第 3 篇文章中关于事务的所有内容，均基于单数据库服务器假设——一台机器、一份事务日志、一个锁管理器。一旦数据分布到多台机器上——例如实施分片（sharding）、采用微服务架构（各服务独享数据库）或启用强一致性复制——便直面分布式系统最棘手的问题：**多台机器如何就某个值达成一致？**

---

## 分布式事务问题

考虑一个电商系统，订单服务和库存服务彼此分离，各自拥有独立数据库：

```sql
Order Service (DB-1)              Inventory Service (DB-2)
┌─────────────────────┐          ┌─────────────────────────┐
│ INSERT INTO orders   │          │ UPDATE products          │
│ (user_id, total)     │          │ SET stock = stock - 1    │
│ VALUES (1, 99.99)    │          │ WHERE product_id = 42    │
└─────────────────────┘          └─────────────────────────┘
```

若订单插入成功而库存更新失败（网络故障、约束冲突或进程崩溃），商品便从未被预留；若缺乏协调机制，数据不一致将不可避免。

在单数据库中，用 `BEGIN ... COMMIT` 将二者包裹即可解决。但在两个数据库之间，这是不可能的——它们拥有各自独立的事务日志、独立的崩溃恢复机制、独立的时钟。

## 两阶段提交（2PC）

分布式事务的经典教科书解法。由一个协调者节点（coordinator）主导协议，多个参与者节点（participants）配合执行。

![两阶段提交协议](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-two-phase-commit.png)


### 协议流程

```text
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

```text
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

2PC 的关键缺陷是：协调者若在发送 PREPARE 后、 COMMIT 或 ROLLBACK 前崩溃，参与者即陷入僵持——它们已投“是”并持有锁，却无法得知最终决策。

```text
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

这称为 **阻塞问题（blocking problem）**：参与者须等待协调者恢复并揭晓最终决策——可能无限期。实践中这意味着：
- 锁被无限期持有，阻塞其他事务；
- 可能需要人工干预；
- 该协议不具备容错能力。

### 2PC 在实践中的应用

尽管存在局限性， 2PC 仍在实际系统中被使用。

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


![分布式共识协议服务器在数字环境中投票](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-distributed-consensus-protocol-servers-voting-in-a-digital-p.jpg)

3PC 在 PREPARE 和 COMMIT 之间增加了一个 PRE-COMMIT 阶段，使参与者可在协调者失效时自主恢复：

```text
Phase 1: CAN-COMMIT?  → 参与者检查自身是否可提交
Phase 2: PRE-COMMIT   → 协调者通知参与者准备（但暂不提交）
Phase 3: DO-COMMIT    → 最终提交

若协调者在 PRE-COMMIT 后崩溃：
  参与者可超时后直接提交（因已知所有节点均已就绪）
```

理论上， 3PC 是非阻塞的；但实践中却极少使用，原因在于：
- 网络分区仍可能导致不一致（某参与者可能未收到 PRE-COMMIT）；
- 额外的一次往返增加了延迟；
- Raft/Paxos 等共识协议更稳健地解决了该问题。

## 共识算法（Consensus Algorithms）

共识问题是：即使部分节点发生故障，多个节点仍需就某个值达成一致，这是强一致性分布式数据库的基石。

### Paxos （概念性）

Paxos （Leslie Lamport， 1989）是首个经严格证明正确的共识算法。它定义三类角色：
- **提议者（Proposers）**：提出值；
- **接受者（Acceptors）**：投票批准提议；
- **学习者（Learners）**：获知最终选定的值。

Single-Decree Paxos 的简化视图如下：

```text
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

Raft （Diego Ongaro & John Ousterhout， 2014）旨在提供与 Paxos 等价、但更易理解的共识方案。它将共识分解为三个子问题：

![Raft 领导者选举](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-raft-election.png)


1. **领导者选举（Leader election）**
2. **日志复制（Log replication）**
3. **安全性（Safety）**

#### 领导者选举

各节点初始均为 **追随者（Follower）**，若在随机超时（如 150–300 ms）内未收到领导者心跳，即转为 **候选人（Candidate）** 并发起选举。

```text
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

```text
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

```text
Leader Log:   [term1:SET x=1] [term1:SET y=2] [term2:SET x=3]
                    │                │               │
                    ▼                ▼               ▼
Follower B:   [term1:SET x=1] [term1:SET y=2] [term2:SET x=3]  ✓ up to date
Follower C:   [term1:SET x=1] [term1:SET y=2]                  ✗ catching up

A log entry is "committed" when replicated to a majority of nodes.
The leader then applies the entry to its state machine and responds to the client.
```

```text
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
| CockroachDB | 每个 range （分区）使用独立的 Raft 组 |
| TiKV | TiDB 的存储层 |
| Consul | 服务发现与配置管理 |
| RethinkDB | 集群协调 |

## Saga 模式

当 2PC 成本过高或不切实际——这在微服务架构中几乎总是如此——Saga 模式便成为替代方案。它不依赖跨服务的分布式事务，而是将业务流程拆解为一系列本地事务，每项均配有一个 **补偿事务（compensating transaction）**，用于在后续步骤失败时回退影响。

![Saga 模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-saga-pattern.png)


### 编排（Orchestration） vs. 协作（Choreography）

**协作式（Choreography）**：各服务发布事件，下游服务监听并响应。

```text
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

```text
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

每个正向操作都需配套一个补偿操作。

| 步骤 | 正向操作 | 补偿操作 |
|------|---------------|-------------------|
| 1 | 创建订单（状态： pending） | 取消订单（状态： cancelled） |
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

## 时钟同步与全局排序

分布式事务需要对操作顺序达成一致。但分布式系统没有共享时钟——每个节点各自维护时钟，而且会漂移。本节解释生产系统如何解决排序问题。

### 时钟问题

```text
节点 A 的时钟: 14:00:00.000
节点 B 的时钟: 14:00:00.003  (快 3ms)
节点 C 的时钟: 13:59:59.997  (慢 3ms)

事务 T1 在节点 A 提交，时间戳 "14:00:00.001"
事务 T2 在节点 B 提交，时间戳 "14:00:00.002"

T1 真的发生在 T2 之前吗？仅凭时间戳无法判断。
如果 A 的时钟偏慢，T2 可能实际上先提交的。
```

物理时钟不足以确定顺序。三种方案解决这个问题：

### Lamport 时钟与向量时钟

```text
Lamport 时钟：单一计数器，每个事件递增
  - 若事件 A 导致事件 B → L(A) < L(B)
  - 但：L(A) < L(B) 不意味着 A 导致了 B（并发事件获得任意顺序）

向量时钟：每个节点一个计数器，追踪因果历史
  节点 A: [3, 1, 2]  → "看过自己 3 个事件，B 的 1 个，C 的 2 个"
  节点 B: [2, 4, 2]  → "看过 A 的 2 个，自己 4 个事件，C 的 2 个"

  比较：若 VA 的所有分量 ≤ VB → A 因果先于 B
       若某些 VA[i] > VB[i] 且某些 VA[j] < VB[j] → 并发
```

向量时钟能实现因果一致性，但不提供全序。对于可串行化分布式事务，需要更强的机制。

### 混合逻辑时钟（HLC）

CockroachDB 和 YugabyteDB 使用 HLC——物理时间与逻辑计数器的组合：

```text
HLC = (physical_time, logical_counter)

规则：
1. 本地事件：hlc.physical = max(hlc.physical, wall_clock); hlc.logical = 0
2. 发送消息：在消息中包含 hlc
3. 收到消息：hlc.physical = max(local.physical, msg.physical, wall_clock)
   若物理时间相等：hlc.logical = max(local.logical, msg.logical) + 1

结果：HLC ≈ 物理时钟时间，但具有因果排序保证
约束：HLC 始终在 max_clock_offset 内接近真实时间
```

```go
// CockroachDB 的不确定性区间
// 读取时，事务必须考虑其他节点在时钟不确定窗口内写入的值

type ReadTimestamp struct {
    ReadTS     hlc.Timestamp  // "我在此时间开始读"
    MaxOffset  time.Duration  // "时钟可能偏差这么多"
    // 不确定性区间：[ReadTS, ReadTS + MaxOffset]
    // 此区间内的值可能在我们之前写入
}

// 若在不确定性区间内发现值：
// 方案 1：推进读时间戳（以更高 ts 重启事务）
// 方案 2：若写事务仍在进行中，等待它完成
```

CockroachDB 的时钟偏移默认值为 500ms。保持 NTP 紧凑（< 250ms）可减少事务重启。

### Google Spanner 与 TrueTime

Spanner 用硬件解决时钟问题：每个数据中心部署 GPS 接收器和原子钟，提供有界不确定性的时间 API。

```text
TrueTime API：
  TT.now() → 返回区间 [earliest, latest]
  TT.after(t) → 若 t 确定在过去则返回 true
  TT.before(t) → 若 t 确定在未来则返回 true

典型不确定性：ε ≈ 1-7ms（平均 ~4ms）
  GPS + 原子钟同步将漂移保持在极小范围
```

Spanner 的提交协议使用 TrueTime 分配全局有意义的时间戳：

```text
Commit-wait 协议：
  1. 事务 T 获取所有锁，执行写入
  2. 协调者选择提交时间戳 s = TT.now().latest
  3. 协调者等待直到 TT.after(s) 为 true
     （最多等待 2ε ≈ 7ms，等不确定性过去）
  4. 释放锁，响应客户端

保证：若 T1 在 T2 开始之前提交（物理时间），
     则 T1 的时间戳 < T2 的时间戳
     → 外部一致性（线性一致）
```

```text
为什么 commit-wait 有效：

T1 在时间 t_commit 以 s1 = TT.now().latest 提交
  → 真实提交时间 ≤ s1（因为 s1 是最晚可能时间）
  → 等待直到 TT.after(s1)：真实时间现在确定 > s1

T2 在时间 t_start > t_commit 开始（物理时间）
  → T2 在某个 ≥ t_start 的时刻选取 s2 = TT.now().latest
  → s2 ≥ t_start 时的真实时间 > s1（因为我们等待了）
  → s1 < s2 有保证！
```

| 系统 | 时钟机制 | 不确定性 | 排序保证 |
|------|---------|---------|---------|
| Spanner | GPS + 原子钟（TrueTime） | 1-7ms | 外部一致性（线性一致） |
| CockroachDB | NTP + HLC | ~250-500ms | 可串行化（非严格线性一致） |
| YugabyteDB | NTP + HLC | 可配置 | 可串行化 |
| TiDB | TSO（集中式时间戳预言机） | 0（单点） | 线性一致（但 TSO 是瓶颈） |

### 实际影响

```sql
-- CockroachDB：检查时钟偏移健康状态
SHOW CLUSTER SETTING server.clock.max_offset;  -- 默认 500ms

-- 若时钟漂移超过 max_offset，节点会自行终止以保护正确性
-- 监控所有节点的 NTP 偏移：
-- $ chronyc tracking | grep "Last offset"

-- Spanner：用户无需关心时钟问题，但要为此付费
-- 只读事务可通过快照读避免 commit-wait：
SELECT * FROM orders
WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
-- Spanner 自动选择安全的读取时间戳
```

## 生产级 Saga 实现

前面展示的 Saga 伪代码捕获了概念，但生产环境的 Saga 需要：持久化状态、幂等步骤、带退避的重试、超时处理和可观测性。以下是生产级模式。

### 状态机设计

```text
Saga 状态：
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌───────────┐
│ STARTED │────►│ RUNNING  │────►│ COMPLETING │────►│ COMPLETED │
└─────────┘     └──────────┘     └────────────┘     └───────────┘
                     │
                     ▼
               ┌──────────────┐     ┌────────────┐
               │ COMPENSATING │────►│  FAILED    │
               └──────────────┘     └────────────┘
                     │
                     ▼
               ┌──────────────┐
               │ COMP_FAILED  │  ← 需要人工介入
               └──────────────┘
```

### Saga 状态数据库 Schema

```sql
CREATE TABLE sagas (
    saga_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type VARCHAR(100) NOT NULL,      -- 'order_creation', 'payment_refund'
    state VARCHAR(20) NOT NULL DEFAULT 'STARTED',
    context JSONB NOT NULL DEFAULT '{}',  -- 步骤间共享数据
    current_step INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deadline_at TIMESTAMPTZ,              -- 全局超时
    error_message TEXT
);

CREATE TABLE saga_steps (
    saga_id UUID REFERENCES sagas(saga_id),
    step_index INT NOT NULL,
    step_name VARCHAR(100) NOT NULL,
    state VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    -- PENDING → EXECUTING → SUCCEEDED / FAILED
    -- COMPENSATING → COMPENSATED / COMP_FAILED
    request_payload JSONB,
    response_payload JSONB,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    last_attempted_at TIMESTAMPTZ,
    idempotency_key UUID NOT NULL DEFAULT gen_random_uuid(),
    PRIMARY KEY (saga_id, step_index)
);

CREATE INDEX idx_sagas_stuck ON sagas (state, updated_at)
    WHERE state IN ('RUNNING', 'COMPENSATING');
```

### 编排器实现

```python
import asyncio
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class StepState(Enum):
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"

@dataclass
class SagaStepDef:
    name: str
    action: Callable  # async (context, idempotency_key) -> result
    compensation: Callable  # async (context, idempotency_key) -> None
    max_attempts: int = 3
    timeout_seconds: int = 30

class SagaOrchestrator:
    def __init__(self, db_pool, steps: list[SagaStepDef]):
        self.db = db_pool
        self.steps = steps

    async def execute(self, saga_type: str, initial_context: dict) -> str:
        saga_id = str(uuid.uuid4())
        async with self.db.acquire() as conn:
            await conn.execute("""
                INSERT INTO sagas (saga_id, saga_type, state, context, deadline_at)
                VALUES ($1, $2, 'RUNNING', $3, $4)
            """, saga_id, saga_type, initial_context,
                datetime.utcnow() + timedelta(minutes=5))

            for i, step in enumerate(self.steps):
                await conn.execute("""
                    INSERT INTO saga_steps (saga_id, step_index, step_name, max_attempts)
                    VALUES ($1, $2, $3, $4)
                """, saga_id, i, step.name, step.max_attempts)

        await self._run_forward(saga_id)
        return saga_id

    async def _run_forward(self, saga_id: str):
        saga = await self._load_saga(saga_id)

        for i in range(saga["current_step"], len(self.steps)):
            step_def = self.steps[i]
            step_row = await self._load_step(saga_id, i)

            success = await self._execute_step_with_retry(
                saga_id, i, step_def, saga["context"], step_row["idempotency_key"]
            )

            if not success:
                await self._compensate(saga_id, i - 1)
                return

            await self._advance_step(saga_id, i + 1)

        await self._mark_completed(saga_id)

    async def _execute_step_with_retry(
        self, saga_id, step_index, step_def, context, idempotency_key
    ) -> bool:
        for attempt in range(step_def.max_attempts):
            try:
                await self._update_step_state(saga_id, step_index, "EXECUTING")
                result = await asyncio.wait_for(
                    step_def.action(context, idempotency_key),
                    timeout=step_def.timeout_seconds
                )
                context.update(result or {})
                await self._update_step_state(
                    saga_id, step_index, "SUCCEEDED", response=result
                )
                return True
            except asyncio.TimeoutError:
                await self._record_attempt(saga_id, step_index, "timeout")
            except Exception as e:
                await self._record_attempt(saga_id, step_index, str(e))
                if attempt < step_def.max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避

        await self._update_step_state(saga_id, step_index, "FAILED")
        return False

    async def _compensate(self, saga_id: str, from_step: int):
        await self._update_saga_state(saga_id, "COMPENSATING")

        for i in range(from_step, -1, -1):
            step_def = self.steps[i]
            step_row = await self._load_step(saga_id, i)
            if step_row["state"] != "SUCCEEDED":
                continue

            try:
                await self._update_step_state(saga_id, i, "COMPENSATING")
                await asyncio.wait_for(
                    step_def.compensation(
                        await self._load_context(saga_id),
                        step_row["idempotency_key"]
                    ),
                    timeout=step_def.timeout_seconds
                )
                await self._update_step_state(saga_id, i, "COMPENSATED")
            except Exception as e:
                await self._update_step_state(saga_id, i, "COMP_FAILED")
                await self._update_saga_state(saga_id, "COMP_FAILED")
                await self._alert_manual_intervention(saga_id, i, e)
                return

        await self._update_saga_state(saga_id, "FAILED")
```

### 僵死 Saga 恢复

如果编排器在执行中途崩溃，Saga 会卡住。后台清扫器负责恢复：

```python
async def sweep_stuck_sagas(db_pool, orchestrator, interval_seconds=60):
    """后台任务：恢复卡住的 Saga。"""
    while True:
        async with db_pool.acquire() as conn:
            stuck = await conn.fetch("""
                SELECT saga_id, state, current_step, saga_type
                FROM sagas
                WHERE state IN ('RUNNING', 'COMPENSATING')
                AND updated_at < NOW() - INTERVAL '2 minutes'
                FOR UPDATE SKIP LOCKED
                LIMIT 10
            """)
            for saga in stuck:
                if saga["state"] == "RUNNING":
                    await orchestrator._run_forward(saga["saga_id"])
                elif saga["state"] == "COMPENSATING":
                    await orchestrator._compensate(
                        saga["saga_id"], saga["current_step"]
                    )

            # 处理超时的 Saga
            expired = await conn.fetch("""
                UPDATE sagas SET state = 'COMPENSATING'
                WHERE state = 'RUNNING' AND deadline_at < NOW()
                RETURNING saga_id, current_step
            """)
            for saga in expired:
                await orchestrator._compensate(
                    saga["saga_id"], saga["current_step"]
                )

        await asyncio.sleep(interval_seconds)
```

### Saga 步骤的幂等性

每个 Saga 步骤必须幂等——安全重试而无副作用：

```python
async def charge_payment(context: dict, idempotency_key: str) -> dict:
    """Saga 步骤：向客户收费。通过 idempotency_key 实现幂等。"""
    response = await payment_api.create_charge(
        amount=context["total"],
        customer_id=context["user_id"],
        idempotency_key=str(idempotency_key),  # 支付服务端去重
    )
    return {"payment_id": response["id"], "charge_status": response["status"]}


async def refund_payment(context: dict, idempotency_key: str):
    """补偿：退款。同样幂等。"""
    if "payment_id" not in context:
        return  # 收费从未发生
    await payment_api.refund(
        payment_id=context["payment_id"],
        idempotency_key=f"refund-{idempotency_key}",
    )
```

### Saga 可观测性

```sql
-- 仪表板查询：Saga 健康概览
SELECT
    saga_type,
    state,
    count(*) AS count,
    avg(EXTRACT(EPOCH FROM (updated_at - created_at))) AS avg_duration_sec
FROM sagas
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY saga_type, state
ORDER BY saga_type, state;

-- 查找需要人工介入的卡住 Saga
SELECT s.saga_id, s.saga_type, s.state, s.error_message,
       ss.step_name, ss.step_index, ss.state AS step_state
FROM sagas s
JOIN saga_steps ss ON s.saga_id = ss.saga_id
WHERE s.state = 'COMP_FAILED'
ORDER BY s.created_at;
```

## 线性一致性（Linearizability） vs. 可串行化（Serializability）

这两个术语常被混淆，但描述的是完全不同的保证。

**可串行化（Serializability）**（来自事务）：并发执行多个事务的结果，等价于这些事务以某种**串行顺序**执行的结果。它关注的是事务与数据库层面的正确性。

**线性一致性（Linearizability）**（来自分布式系统）：每个操作看起来都在其调用与完成之间的某个瞬时点原子生效；一旦写入被确认，所有后续读取都必须看到该值。它关注的是**单个操作**与实时顺序。

```text
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
| 示例系统 | 任何 SERIALIZABLE 隔离级别的数据库 | ZooKeeper、 etcd、 Spanner |

**严格可串行化（Strict serializability）** = 可串行化 + 线性一致性。这是最强的一致性保证，也是 Google Spanner 提供的保证。

## 最终一致性（Eventual Consistency）


![Saga 模式作为补偿事务链的多米诺效应](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-saga-pattern-as-a-chain-of-compensating-transactions-domino-.jpg)

在线性一致性的另一端是最终一致性：若不再有新的写入，所有副本将**最终**收敛到相同值。

![一致性谱](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-consistency-spectrum.png)


“最终”没有明确定义——收敛时间可能短至毫秒，也可能长达数分钟。实践中：

```text
向 Node A 写入 "x = 5"
  t=0ms:  Node A: x=5,  Node B: x=3,  Node C: x=3
  t=50ms: Node A: x=5,  Node B: x=5,  Node C: x=3
  t=100ms: Node A: x=5, Node B: x=5,  Node C: x=5  ← 收敛完成
```

默认采用最终一致性的系统包括：
- DynamoDB （除非显式请求强一致性读取）；
- Cassandra （一致性级别设为 ONE）；
- DNS；
- CDN 缓存。

它适用于以下场景：
- 陈旧数据可容忍（社交媒体动态流、商品推荐）；
- 系统能检测并解决冲突（购物车、 CRDT）；
- 性能比一致性更重要（分析、日志）。

## 真实世界常用模式

### Outbox 模式

如何**原子性地**更新数据库 *并* 向消息中间件（如 Kafka）发布消息？你无法在数据库与 Kafka 之间使用分布式事务。

Outbox 模式：将消息写入数据库内的一个 “outbox” 表（与业务更新在同一事务中），再由独立进程读取该表并向消息中间件发布。

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

```text
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

## 下一步

我们已覆盖理论：数据如何存储、查询、复制、分片与事务处理。但仅有理论远远不够。在最后一篇中，我们将走向实战：**生产环境中的数据库**——迁移、监控、连接池、备份、容量规划，以及来自真实线上事故的“战争故事”。