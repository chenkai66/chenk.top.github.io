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
translationKey: "databases-7"
---

Everything we covered about transactions in Article 3 assumed a single database server. One machine, one transaction log, one lock manager. The moment your data lives on more than one machine — which happens the moment you shard, use microservices with separate databases, or replicate with strong consistency — you face the hardest problem in distributed systems: how do you make multiple machines agree?

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

If the order insert succeeds but the inventory update fails (network issue, constraint violation, crash), you have a problem: an order exists for a product that was never reserved. Without coordination, you get inconsistency.

On a single database, wrapping both in a `BEGIN ... COMMIT` solves this. Across two databases, that is not possible — they have separate transaction logs, separate crash recovery, separate clocks.

## Two-Phase Commit (2PC)

The textbook solution to distributed transactions. A coordinator node orchestrates the protocol with participating nodes.

![Two-phase commit protocol](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-two-phase-commit.png)


### The Protocol

```
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

### The Coordinator Failure Problem

Here is the critical weakness of 2PC: if the coordinator crashes after sending PREPARE but before sending COMMIT/ROLLBACK, the participants are stuck. They have voted "Yes" and hold locks, but they do not know the final decision.

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

This is called the **blocking problem**. Participants must wait (potentially forever) until the coordinator recovers and reveals its decision. In practice, this means:
- Locks held indefinitely, blocking other transactions
- Manual intervention may be required
- The protocol is not fault-tolerant

### 2PC in Practice

Despite its limitations, 2PC is used in practice:

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

```
Phase 1: CAN-COMMIT?  → participants check if they can commit
Phase 2: PRE-COMMIT   → coordinator tells participants to prepare (but not commit yet)
Phase 3: DO-COMMIT    → final commit

If coordinator crashes after PRE-COMMIT:
  Participants can time out and commit (they know everyone was ready)
```

In theory, 3PC is non-blocking. In practice, it is rarely used because:
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

Paxos is correct but notoriously difficult to implement. In Lamport's own words, it took years for the community to understand his paper. This difficulty led to Raft.

### Raft: Understandable Consensus

Raft (2014, Diego Ongaro and John Ousterhout) was designed to be equivalent to Paxos but easier to understand. It decomposes consensus into three sub-problems:

![Raft leader election](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-raft-election.png)


1. **Leader election**
2. **Log replication**
3. **Safety**

#### Leader Election

Every node starts as a **follower**. If a follower does not hear from a leader within a random timeout (150-300 ms), it becomes a **candidate** and starts an election.

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

#### Log Replication

Once elected, the leader accepts client requests and appends them to its log. It then replicates entries to followers:

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

#### Where Raft Is Used

| System | Use of Raft |
|--------|-------------|
| etcd | Key-value store (Kubernetes backing store) |
| CockroachDB | Each range (partition) uses a separate Raft group |
| TiKV | Storage layer for TiDB |
| Consul | Service discovery and configuration |
| RethinkDB | Cluster coordination |

## Saga Pattern

When 2PC is too expensive or impractical (which is most of the time in microservices), the Saga pattern provides an alternative. Instead of one big distributed transaction, break it into a sequence of local transactions, each with a **compensating transaction** that undoes its work if a later step fails.

![Saga pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-saga-pattern.png)


### Choreography vs Orchestration

**Choreography**: Each service publishes events. The next service listens and acts.

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

**Orchestration**: A central orchestrator tells each service what to do.

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

### Compensating Transactions

Each forward action needs a compensating action:

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

## Linearizability vs Serializability

These two terms are frequently confused but describe different things:

**Serializability** (from transactions): The result of executing concurrent transactions is equivalent to *some* serial execution of those transactions. This is about transactions and databases.

**Linearizability** (from distributed systems): Every operation appears to take effect instantaneously at some point between its invocation and completion. Once a write is acknowledged, all subsequent reads see it. This is about individual operations and real-time ordering.

```
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

At the opposite end of the spectrum from linearizability is eventual consistency: if no new writes are made, all replicas will *eventually* converge to the same value.

![Consistency spectrum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/07-consistency-spectrum.png)


"Eventually" is vague — it could be milliseconds or minutes. In practice:

```
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

```
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
