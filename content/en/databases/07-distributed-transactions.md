---
title: "Databases (7): Distributed Transactions — 2PC, Saga, and Why Consensus Is Hard"
date: 2024-04-28 09:00:00
tags:
  - Databases
  - Distributed Systems
  - Consensus
  - Transactions
categories: Databases
series: databases
lang: en
description: "How distributed databases coordinate transactions across machines — two-phase commit, Raft consensus, the Saga pattern, and practical patterns like outbox and CDC."
disableNunjucks: true
series_order: 7
series_total: 8
translationKey: "databases-7"
---

Everything we covered about transactions in Article 3 assumed a single database server: one machine, one transaction log, one lock manager. When your data spans multiple machines—through sharding, using microservices with separate databases, or replicating with strong consistency—you face the hardest problem in distributed systems: how do you get multiple machines to agree?

---

## The Distributed Transaction Problem

Consider an e-commerce system with separate services for orders and inventory, each with its own database:

```sql
Order Service (DB-1)              Inventory Service (DB-2)
┌─────────────────────┐          ┌─────────────────────────┐
│ INSERT INTO orders   │          │ UPDATE products          │
│ (user_id, total)     │          │ SET stock = stock - 1    │
│ VALUES (1, 99.99)    │          │ WHERE product_id = 42    │
└─────────────────────┘          └─────────────────────────┘
```

If the order insert succeeds but the inventory update fails (because of a network issue, constraint violation, or crash), you have a problem: an order exists for a product that was never reserved. Without coordination, this leads to inconsistency.

On a single database, wrapping both in a `BEGIN ... COMMIT` solves this. Across two databases, that is not possible — they have separate transaction logs, separate crash recovery, separate clocks.

## Two-Phase Commit (2PC)

The textbook solution to distributed transactions. A coordinator node orchestrates the protocol with participating nodes.

![Two-phase commit protocol](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-two-phase-commit.png)


### The Protocol

```text
Phase 1: PREPARE (Voting Phase)
  Coordinator → Participant A: "Can you commit transaction T?"
  Coordinator → Participant B: "Can you commit transaction T?"
  Participant A → Coordinator: "Yes, I'm prepared" (locks held, WAL flushed)
  Participant B → Coordinator: "Yes, I'm prepared"

Phase 2: COMMIT (Decision Phase)
  Coordinator → Participant A: "COMMIT transaction T"
  Coordinator → Participant B: "COMMIT transaction T"
  Participant A → Coordinator: "Done"
  Participant B → Coordinator: "Done"
```

If any participant votes "No" in Phase 1, the coordinator sends ROLLBACK to everyone.

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

### The Coordinator Failure Problem

The critical weakness of 2PC is that if the coordinator crashes after sending PREPARE but before sending COMMIT/ROLLBACK, the participants are stuck. They have voted "Yes" and hold locks, but they don't know the final decision.

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

This is called the **blocking problem**. Participants must wait (potentially forever) until the coordinator recovers and reveals its decision. In practice, this means:
- Locks held indefinitely, blocking other transactions
- Manual intervention may be required
- The protocol is not fault-tolerant

### 2PC in Practice

Despite its limitations, 2PC is still used in practice:

```sql
-- PostgreSQL: prepared transactions (2PC participant)
-- The application/coordinator calls these:

-- Phase 1: Prepare
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
PREPARE TRANSACTION 'transfer_001_debit';

-- Phase 2: Commit (if all participants prepared successfully)
COMMIT PREPARED 'transfer_001_debit';

-- Or rollback
ROLLBACK PREPARED 'transfer_001_debit';

-- Check for orphaned prepared transactions (stuck 2PC)
SELECT gid, prepared, owner, database
FROM pg_prepared_xacts;
```

```java
// Java XA transactions (JTA) — standard API for 2PC
UserTransaction ut = (UserTransaction) ctx.lookup("java:comp/UserTransaction");
ut.begin();

// Enlist two different databases in the same transaction
Connection conn1 = ds1.getConnection();  // Order database
Connection conn2 = ds2.getConnection();  // Inventory database

conn1.prepareStatement("INSERT INTO orders ...").execute();
conn2.prepareStatement("UPDATE inventory SET stock = stock - 1 ...").execute();

ut.commit();  // Transaction manager runs 2PC protocol
```

## Three-Phase Commit (3PC)


![Distributed consensus protocol servers voting in a digital p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-distributed-consensus-protocol-servers-voting-in-a-digital-p.jpg)

3PC adds a PRE-COMMIT phase between PREPARE and COMMIT, which allows participants to recover without the coordinator:

```text
Phase 1: CAN-COMMIT?  → participants check if they can commit
Phase 2: PRE-COMMIT   → coordinator tells participants to prepare (but not commit yet)
Phase 3: DO-COMMIT    → final commit

If coordinator crashes after PRE-COMMIT:
  Participants can time out and commit (they know everyone was ready)
```

In theory, 3PC is non-blocking, but in practice, it is rarely used because:
- Network partitions can still cause inconsistency (a participant might not receive PRE-COMMIT)
- The additional round trip adds latency
- Raft/Paxos consensus protocols solve the problem more robustly

## Consensus Algorithms

Consensus is the problem of getting multiple nodes to agree on a value, even when some nodes fail. It is the foundation of strongly consistent distributed databases.

### Paxos (Conceptual)

Paxos (invented by Leslie Lamport in 1989) was the first proven consensus algorithm. It uses three roles:
- **Proposers**: Propose values
- **Acceptors**: Vote on proposals
- **Learners**: Learn the decided value

A simplified view of Single-Decree Paxos:

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

Paxos is correct but notoriously difficult to implement. As Lamport noted, it took years for the community to understand his paper. This difficulty led to Raft.

### Raft: Understandable Consensus

Raft (2014, by Diego Ongaro and John Ousterhout) was designed to be equivalent to Paxos but easier to understand. It decomposes consensus into three sub-problems:

![Raft leader election](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-raft-election.png)


1. **Leader election**
2. **Log replication**
3. **Safety**

#### Leader Election

Every node starts as a **follower**. If a follower does not hear from a leader within a random timeout (150-300 ms), it becomes a **candidate** and initiates an election.

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

#### Log Replication

Once elected, the leader accepts client requests, appends them to its log, and then replicates the entries to followers:

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

#### Where Raft Is Used

| System | Use of Raft |
|--------|-------------|
| etcd | Key-value store (Kubernetes backing store) |
| CockroachDB | Each range (partition) uses a separate Raft group |
| TiKV | Storage layer for TiDB |
| Consul | Service discovery and configuration |
| RethinkDB | Cluster coordination |

## Saga Pattern

When 2PC is too expensive or impractical (often the case in microservices), the Saga pattern provides an alternative. Instead of one large distributed transaction, it breaks the process into a sequence of local transactions, each with a **compensating transaction** that undoes its work if a later step fails.

![Saga pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-saga-pattern.png)


### Choreography vs Orchestration

**Choreography**: Each service publishes events. The next service listens and acts.

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

**Orchestration**: A central orchestrator tells each service what to do.

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

### Compensating Transactions

Each forward action requires a compensating action:

| Step | Forward Action | Compensating Action |
|------|---------------|-------------------|
| 1 | Create order (status: pending) | Cancel order (status: cancelled) |
| 2 | Reserve inventory (stock - 1) | Release inventory (stock + 1) |
| 3 | Charge payment | Refund payment |
| 4 | Ship order | Cancel shipment |

```python
# Saga orchestrator pseudocode
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
                # Compensate in reverse order
                for completed_step in reversed(completed):
                    completed_step.compensation(context)
                raise SagaFailedError(f"Step failed: {e}")
```

## Clock Synchronization and Global Ordering

Distributed transactions require agreement on ordering. But distributed systems have no shared clock — each node has its own, and they drift. This section explains how production systems solve the ordering problem.

### The Clock Problem

```text
Node A's clock: 14:00:00.000
Node B's clock: 14:00:00.003  (3ms ahead)
Node C's clock: 13:59:59.997  (3ms behind)

Transaction T1 commits on Node A at "14:00:00.001"
Transaction T2 commits on Node B at "14:00:00.002"

Did T1 happen before T2? We cannot know from timestamps alone.
If A's clock is slow, T2 might have actually committed first.
```

Physical clocks are insufficient for ordering. Three approaches solve this:

### Lamport Clocks and Vector Clocks

```text
Lamport clock: single counter, incremented on every event
  - If event A causes event B → L(A) < L(B)
  - BUT: L(A) < L(B) does NOT mean A caused B (concurrent events get arbitrary order)

Vector clock: one counter per node, tracks causal history
  Node A: [3, 1, 2]  → "I've seen 3 of my events, 1 from B, 2 from C"
  Node B: [2, 4, 2]  → "I've seen 2 from A, 4 of my events, 2 from C"

  Comparing: if all elements of VA ≤ VB → A causally precedes B
             if some VA[i] > VB[i] and some VA[j] < VB[j] → concurrent
```

Vector clocks enable causal consistency but don't provide a total order. For serializable distributed transactions, you need something stronger.

### Hybrid Logical Clocks (HLC)

CockroachDB and YugabyteDB use HLCs — a combination of physical time and a logical counter:

```text
HLC = (physical_time, logical_counter)

Rules:
1. On local event: hlc.physical = max(hlc.physical, wall_clock); hlc.logical = 0
2. On send: include hlc in message
3. On receive: hlc.physical = max(local.physical, msg.physical, wall_clock)
   if physical times are equal: hlc.logical = max(local.logical, msg.logical) + 1

Result: HLC ≈ wall clock time, but with causal ordering guarantees
Bound: HLC is always within max_clock_offset of real time
```

```go
// CockroachDB's uncertainty interval
// When reading, a transaction must consider values written by other nodes
// within the clock uncertainty window

type ReadTimestamp struct {
    ReadTS     hlc.Timestamp  // "I started reading at this time"
    MaxOffset  time.Duration  // "Clocks might be off by this much"
    // Uncertainty interval: [ReadTS, ReadTS + MaxOffset]
    // Values in this interval might have been written before us
}

// If a value is found with timestamp in the uncertainty interval:
// Option 1: Push read timestamp forward (restart transaction at higher ts)
// Option 2: If the writing transaction is still in-flight, wait for it
```

CockroachDB's clock offset default is 500ms. Keeping NTP tight (< 250ms) reduces transaction restarts.

### Google Spanner and TrueTime

Spanner solves the clock problem with hardware: GPS receivers and atomic clocks in every data center, providing a bounded-uncertainty time API.

```text
TrueTime API:
  TT.now() → returns an interval [earliest, latest]
  TT.after(t) → true if t is definitely in the past
  TT.before(t) → true if t is definitely in the future

Typical uncertainty: ε ≈ 1-7ms (average ~4ms)
  GPS + atomic clock synchronization keeps drift minimal
```

Spanner's commit protocol uses TrueTime to assign globally meaningful timestamps:

```text
Commit-wait protocol:
  1. Transaction T acquires all locks, performs writes
  2. Coordinator picks commit timestamp s = TT.now().latest
  3. Coordinator WAITS until TT.after(s) is true
     (waits at most 2ε ≈ 7ms for uncertainty to pass)
  4. Release locks, respond to client

Guarantee: if T1 commits before T2 starts (real time),
           then T1's timestamp < T2's timestamp
           → externally consistent (linearizable)
```

```text
Why commit-wait works:

T1 commits at s1 = TT.now().latest at time t_commit
  → real commit time ≤ s1 (because s1 is the latest possible time)
  → waits until TT.after(s1): real time is now definitely > s1

T2 starts at time t_start > t_commit (real time)
  → T2 picks s2 = TT.now().latest at some point ≥ t_start
  → s2 ≥ real_time at t_start > s1 (because we waited)
  → s1 < s2 guaranteed!
```

| System | Clock mechanism | Uncertainty | Ordering guarantee |
|--------|----------------|-------------|-------------------|
| Spanner | GPS + atomic clocks (TrueTime) | 1-7ms | External consistency (linearizable) |
| CockroachDB | NTP + HLC | ~250-500ms | Serializable (not linearizable without caution) |
| YugabyteDB | NTP + HLC | configurable | Serializable |
| TiDB | TSO (centralized timestamp oracle) | 0 (single point) | Linearizable (but TSO is a bottleneck) |

### Practical Impact

```sql
-- CockroachDB: check clock offset health
SHOW CLUSTER SETTING server.clock.max_offset;  -- default 500ms

-- If clocks drift beyond max_offset, nodes self-terminate to protect correctness
-- Monitor NTP offset on all nodes:
-- $ chronyc tracking | grep "Last offset"

-- Spanner: no clock concerns for the user, but you pay for it
-- Read-only transactions can avoid commit-wait by reading at a snapshot:
SELECT * FROM orders
WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
-- Spanner picks a safe read timestamp automatically
```

## Production Saga Implementation

The pseudocode Saga shown earlier captures the concept, but production sagas need: persistent state, idempotent steps, retries with backoff, timeout handling, and observability. Here is a production-grade pattern.

### State Machine Design

```text
Saga States:
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
               │ COMP_FAILED  │  ← requires manual intervention
               └──────────────┘
```

### Database Schema for Saga State

```sql
CREATE TABLE sagas (
    saga_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type VARCHAR(100) NOT NULL,      -- 'order_creation', 'payment_refund'
    state VARCHAR(20) NOT NULL DEFAULT 'STARTED',
    context JSONB NOT NULL DEFAULT '{}',  -- shared data between steps
    current_step INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deadline_at TIMESTAMPTZ,              -- global timeout
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

### Orchestrator Implementation

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
                    await asyncio.sleep(2 ** attempt)  # exponential backoff

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

### Dead Saga Recovery

Sagas can get stuck if the orchestrator crashes mid-execution. A background sweeper recovers them:

```python
async def sweep_stuck_sagas(db_pool, orchestrator, interval_seconds=60):
    """Background task that recovers stuck sagas."""
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

            # Also handle deadline-exceeded sagas
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

### Idempotency in Saga Steps

Every saga step must be idempotent — safe to retry without side effects:

```python
async def charge_payment(context: dict, idempotency_key: str) -> dict:
    """Saga step: charge the customer. Idempotent via idempotency_key."""
    response = await payment_api.create_charge(
        amount=context["total"],
        customer_id=context["user_id"],
        idempotency_key=str(idempotency_key),  # payment provider deduplicates
    )
    return {"payment_id": response["id"], "charge_status": response["status"]}


async def refund_payment(context: dict, idempotency_key: str):
    """Compensation: refund the charge. Also idempotent."""
    if "payment_id" not in context:
        return  # charge never happened
    await payment_api.refund(
        payment_id=context["payment_id"],
        idempotency_key=f"refund-{idempotency_key}",
    )
```

### Saga Observability

```sql
-- Dashboard query: saga health overview
SELECT
    saga_type,
    state,
    count(*) AS count,
    avg(EXTRACT(EPOCH FROM (updated_at - created_at))) AS avg_duration_sec
FROM sagas
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY saga_type, state
ORDER BY saga_type, state;

-- Find stuck sagas requiring manual intervention
SELECT s.saga_id, s.saga_type, s.state, s.error_message,
       ss.step_name, ss.step_index, ss.state AS step_state
FROM sagas s
JOIN saga_steps ss ON s.saga_id = ss.saga_id
WHERE s.state = 'COMP_FAILED'
ORDER BY s.created_at;
```

## Linearizability vs Serializability

These two terms are frequently confused but describe different things:

**Serializability** (from transactions): The result of executing concurrent transactions is equivalent to *some* serial execution of those transactions. This is about transactions and databases.

**Linearizability** (from distributed systems): Every operation appears to take effect instantaneously at some point between its invocation and completion. Once a write is acknowledged, all subsequent reads see it. This is about individual operations and real-time ordering.

```text
Linearizable system (register with value initially 0):

Client A: write(1)  ─────────────────► OK
                                        │ (from this point, all reads must return 1)
Client B:              read() ──────────► 1  ✓
Client C:                     read() ──► 1   ✓

Non-linearizable:
Client A: write(1)  ─────────────────► OK
Client B:              read() ──────────► 0  ✗ (stale!)
Client C:                     read() ──► 1   ✓
```

| Property | Serializability | Linearizability |
|----------|----------------|-----------------|
| Scope | Multi-operation transactions | Single operations |
| Ordering | Some serial order (any order is fine) | Real-time order |
| Where it matters | Databases | Distributed key-value stores, locks |
| Example systems | Any SERIALIZABLE database | ZooKeeper, etcd, Spanner |

**Strict serializability** = serializability + linearizability. This is the strongest guarantee and what Google Spanner provides.

## Eventual Consistency


![Saga pattern as a chain of compensating transactions domino](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/07-saga-pattern-as-a-chain-of-compensating-transactions-domino-.jpg)

At the opposite end of the spectrum from linearizability is eventual consistency: if no new writes are madede, all replicas will *eventually* converge to the same value.

![Consistency spectrum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-consistency-spectrum.png)


"Eventually" is vague — it could be milliseconds or minutes. In practice:

```text
Write "x = 5" to Node A
  t=0ms:  Node A: x=5,  Node B: x=3,  Node C: x=3
  t=50ms: Node A: x=5,  Node B: x=5,  Node C: x=3
  t=100ms: Node A: x=5, Node B: x=5,  Node C: x=5  ← converged
```

Eventual consistency is the default in:
- DynamoDB (unless you request strongly consistent reads)
- Cassandra (with consistency level ONE)
- DNS
- CDN caches

It is acceptable when:
- Stale data is tolerable (social media feeds, product recommendations)
- The system can detect and resolve conflicts (shopping carts, CRDTs)
- Performance is more important than consistency (analytics, logging)

## Real-World Patterns

### The Outbox Pattern

How do you atomically update a database AND publish a message to a message broker? You cannot use a distributed transaction between your database and Kafka.

The outbox pattern: write the message to an "outbox" table in the same database transaction. A separate process reads the outbox and publishes to the message broker.

```sql
-- Single database transaction (atomic!)
BEGIN;

-- Business logic
INSERT INTO orders (user_id, total, status)
VALUES (1, 99.99, 'created');

-- Outbox entry (same transaction, same database)
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

A separate publisher process (or Debezium with CDC) reads the outbox table and publishes events to Kafka:

```python
# Outbox publisher (runs continuously)
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

### Change Data Capture (CDC)

Instead of writing to an outbox table, capture changes directly from the database's transaction log:

```text
Database WAL/Binlog ──► CDC Tool (Debezium) ──► Kafka ──► Consumers

PostgreSQL WAL → Debezium → Kafka topic "db.public.orders"
MySQL Binlog   → Debezium → Kafka topic "db.inventory.products"
```

```json
// Debezium CDC event (from PostgreSQL)
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

CDC advantages over the outbox pattern:
- No need to modify application code or schema
- Captures ALL changes, not just ones you remembered to outbox
- Lower latency (reads directly from the WAL)
- No dual-write risk

## When to Avoid Distributed Transactions

The best distributed transaction is the one you do not need. Strategies:

1. **Keep data that transacts together on the same node**: Design partitioning keys so related data co-locates.

2. **Accept eventual consistency**: Many business processes are naturally asynchronous (email, notifications, analytics).

3. **Use idempotent operations**: Design operations so retrying is safe.

```sql
-- Idempotent insert (PostgreSQL)
INSERT INTO processed_events (event_id, processed_at)
VALUES ('evt-123', NOW())
ON CONFLICT (event_id) DO NOTHING;
-- Safe to retry — duplicate inserts are silently ignored
```

4. **Design for compensation**: Instead of preventing inconsistency, detect and fix it. This is what banks actually do — reconciliation processes run nightly.

5. **Use a single database**: If your microservices share the same database (heresy, but practical), use regular transactions.

## What's Next

We have covered the theory: how data is stored, queried, replicated, partitioned, and transacted. But theory is not enough. In the final article, we will get practical: **databases in production** — migrations, monitoring, connection pooling, backups, capacity planning, and real war stories from production incidents.
