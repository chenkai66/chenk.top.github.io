---
title: "Databases (3): Transactions and Concurrency — ACID, Isolation Levels, and Locking"
date: 2024-04-21 09:00:00
tags:
  - Databases
  - SQL
  - Transactions
  - Concurrency
categories: Databases
series: databases
lang: en
description: "A thorough guide to ACID properties, isolation levels, MVCC, locking strategies, and deadlock prevention — with concrete SQL examples for every concept."
disableNunjucks: true
series_order: 3
series_total: 8
translationKey: "databases-3"
---

Every application that handles money, inventory, or any state that matters eventually hits a concurrency bug. Two users buy the last item in stock. A bank transfer debits one account but crashes before crediting the other. A report reads half-updated data and produces nonsense numbers. Transactions exist to prevent these failures, and understanding how they work is non-negotiable for anyone building production systems.

---

## What Is a Transaction?

A transaction is a group of operations that the database treats as a single unit. Either **all** operations succeed, or **none** of them do.

```sql
BEGIN;
    UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
    UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;
COMMIT;
```

If the server crashes between the two `UPDATE` statements, the transaction is rolled back. Account 1 does not lose $500 without Account 2 gaining it. This is the fundamental guarantee.

## ACID: The Four Guarantees

ACID is not just an acronym you memorize for interviews. Each letter represents a specific guarantee, and understanding what breaks *without* each one is more important than the definition itself.

### Atomicity — All or Nothing

**Definition**: A transaction either completes fully or has no effect at all.

**What breaks without it**:

```sql
-- Without atomicity: server crashes between these two statements
UPDATE inventory SET stock = stock - 1 WHERE product_id = 42;
-- CRASH HERE
INSERT INTO order_items (order_id, product_id, quantity) VALUES (101, 42, 1);
-- Stock decreased but order item never created. Inventory leak.
```

Atomicity means the database uses a **write-ahead log (WAL)** to record changes before applying them. On crash recovery, incomplete transactions are rolled back.

### Consistency — Valid State to Valid State

**Definition**: A transaction moves the database from one valid state to another. All constraints (foreign keys, CHECK, UNIQUE, NOT NULL) are enforced.

**What breaks without it**:

```sql
-- Without consistency enforcement:
INSERT INTO orders (order_id, user_id) VALUES (999, 12345);
-- user_id 12345 does not exist in the users table
-- Now we have an orphaned order with no associated user
```

The foreign key constraint `REFERENCES users(user_id)` prevents this. Consistency also includes application-level invariants — for example, "the sum of all account balances must remain constant during a transfer."

### Isolation — Concurrent Transactions Don't Interfere

**Definition**: Concurrently executing transactions produce the same result as if they ran sequentially.

![Isolation levels comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-isolation-levels.png)


**What breaks without it**:

```sql
-- Transaction A: reads account balance
SELECT balance FROM accounts WHERE account_id = 1;  -- Returns 1000

-- Transaction B: withdraws 500 and commits
UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
COMMIT;

-- Transaction A: reads again (same transaction)
SELECT balance FROM accounts WHERE account_id = 1;  -- Returns 500!
-- The balance changed mid-transaction. A's view of the world is inconsistent.
```

This is a **non-repeatable read**. Isolation levels control which anomalies are allowed.

### Durability — Committed Means Permanent

**Definition**: Once a transaction commits, its changes survive any subsequent crash (power failure, OS crash, hardware failure).

**What breaks without it**: You commit a bank transfer, see "Success" on screen, and then the server reboots. The transfer is gone. The database reverted to a state before the commit because changes were only in memory.

Durability is enforced by flushing the WAL to persistent storage before reporting a commit as successful. The actual data pages may still be in memory (dirty pages), but the WAL contains enough information to reconstruct them after a crash.

## Transaction Lifecycle


![Mvcc timeline visualization parallel universes of database s](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/03-mvcc-timeline-visualization-parallel-universes-of-database-s.jpg)

```sql
-- Start a transaction
BEGIN;
-- or: START TRANSACTION;

-- Do work
INSERT INTO orders (user_id, status) VALUES (1, 'pending');
UPDATE inventory SET stock = stock - 1 WHERE product_id = 42;

-- Create a savepoint (partial rollback target)
SAVEPOINT after_inventory;

-- More work
INSERT INTO shipping (order_id, address) VALUES (currval('orders_order_id_seq'), '123 Main St');

-- Oops, wrong address. Roll back to savepoint.
ROLLBACK TO SAVEPOINT after_inventory;

-- Fix and retry
INSERT INTO shipping (order_id, address) VALUES (currval('orders_order_id_seq'), '456 Oak Ave');

-- Commit everything
COMMIT;
```

| Command | Effect |
|---------|--------|
| `BEGIN` | Start a new transaction |
| `COMMIT` | Make all changes permanent |
| `ROLLBACK` | Undo all changes since `BEGIN` |
| `SAVEPOINT name` | Create a named checkpoint within the transaction |
| `ROLLBACK TO SAVEPOINT name` | Undo changes back to the savepoint (transaction continues) |
| `RELEASE SAVEPOINT name` | Remove the savepoint (changes kept) |

In autocommit mode (the default in most databases), every statement is its own transaction. An explicit `BEGIN` starts a multi-statement transaction.

## Isolation Levels


![Database transaction locks as golden padlocks on digital vau](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/03-database-transaction-locks-as-golden-padlocks-on-digital-vau.jpg)

The SQL standard defines four isolation levels, each allowing different concurrency anomalies. Stronger isolation = fewer anomalies but lower throughput.

### The Three Anomalies

Before we look at isolation levels, let us define the anomalies precisely.

![Isolation level anomalies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-isolation-anomalies.png)


**Dirty Read**: Transaction A reads data written by Transaction B before B commits. If B rolls back, A has read data that never officially existed.

```sql
-- Session 1                          -- Session 2
BEGIN;
UPDATE products SET price = 0.01
WHERE product_id = 1;
                                      BEGIN;
                                      -- Dirty read: sees price = 0.01
                                      SELECT price FROM products
                                      WHERE product_id = 1;
ROLLBACK;  -- price reverts to 49.99
                                      -- Session 2 used price 0.01 for
                                      -- a decision, but that price
                                      -- never existed.
                                      COMMIT;
```

**Non-Repeatable Read**: Transaction A reads a row, Transaction B modifies and commits that row, then A reads the same row again and gets different data.

```sql
-- Session 1                          -- Session 2
BEGIN;
SELECT balance FROM accounts
WHERE id = 1;  -- Returns 1000
                                      BEGIN;
                                      UPDATE accounts SET balance = 500
                                      WHERE id = 1;
                                      COMMIT;
SELECT balance FROM accounts
WHERE id = 1;  -- Returns 500!
-- Same query, different result within
-- the same transaction.
COMMIT;
```

**Phantom Read**: Transaction A runs a query with a range condition, Transaction B inserts a new row that matches, then A re-runs the query and sees a new "phantom" row.

```sql
-- Session 1                          -- Session 2
BEGIN;
SELECT COUNT(*) FROM orders
WHERE status = 'pending';  -- Returns 5
                                      BEGIN;
                                      INSERT INTO orders (user_id, status)
                                      VALUES (99, 'pending');
                                      COMMIT;
SELECT COUNT(*) FROM orders
WHERE status = 'pending';  -- Returns 6!
-- A new row appeared (phantom).
COMMIT;
```

### Isolation Level Matrix

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|----------------|------------|--------------------:|-------------:|-------------|
| READ UNCOMMITTED | Possible | Possible | Possible | Fastest |
| READ COMMITTED | **Prevented** | Possible | Possible | Fast |
| REPEATABLE READ | **Prevented** | **Prevented** | Possible* | Moderate |
| SERIALIZABLE | **Prevented** | **Prevented** | **Prevented** | Slowest |

*In PostgreSQL, REPEATABLE READ also prevents phantom reads (it uses snapshot isolation, which is stronger than the SQL standard requires).

```sql
-- Set isolation level for a transaction
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- ... your queries ...
COMMIT;

-- Set default isolation level for the session (PostgreSQL)
SET default_transaction_isolation = 'read committed';

-- Check current isolation level (PostgreSQL)
SHOW default_transaction_isolation;

-- MySQL
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT @@transaction_isolation;
```

### What Should You Use?

| Use Case | Recommended Level |
|----------|------------------|
| Most web applications | READ COMMITTED (PostgreSQL default) |
| Financial transactions | SERIALIZABLE or REPEATABLE READ |
| Reporting / analytics | REPEATABLE READ (consistent snapshot) |
| Best effort / monitoring | READ UNCOMMITTED (only if you really need it) |

PostgreSQL defaults to READ COMMITTED. MySQL (InnoDB) defaults to REPEATABLE READ. Both are reasonable defaults for most applications.

## MVCC: How Databases Implement Isolation Efficiently

Multi-Version Concurrency Control (MVCC) is the mechanism that makes isolation levels practical. Instead of blocking readers when writers are active (which kills performance), MVCC keeps multiple versions of each row.

![Multi-version concurrency control](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-mvcc-timeline.png)


### PostgreSQL MVCC

In PostgreSQL, each row has hidden system columns:

- `xmin` — the transaction ID that created (inserted) this row version
- `xmax` — the transaction ID that deleted/updated this row version (0 if still live)

When you `UPDATE` a row, PostgreSQL does not modify it in place. Instead:
1. Marks the old row version as expired (`xmax = current transaction ID`)
2. Creates a new row version (`xmin = current transaction ID`)

Both versions coexist. Each transaction sees only the version that was "alive" at its snapshot time.

```sql
-- Transaction 100 inserts a row
INSERT INTO accounts (id, balance) VALUES (1, 1000);
-- Row: xmin=100, xmax=0, balance=1000

-- Transaction 200 updates the row
UPDATE accounts SET balance = 500 WHERE id = 1;
-- Old row: xmin=100, xmax=200, balance=1000  (still visible to old snapshots)
-- New row: xmin=200, xmax=0,   balance=500   (visible to new snapshots)
```

This is why PostgreSQL needs `VACUUM` — dead row versions accumulate and must be cleaned up.

### MySQL InnoDB MVCC

InnoDB uses a different approach:
- Each row has a hidden 6-byte transaction ID and a 7-byte roll pointer
- The roll pointer points to an **undo log** entry containing the previous version
- Multiple undo log entries form a chain for each row

To reconstruct an old version, InnoDB walks the undo log chain backwards. This means old versions do not consume extra space in the main table, but long-running transactions force InnoDB to keep long undo log chains.

### MVCC Implications

| Behavior | PostgreSQL | MySQL InnoDB |
|----------|-----------|--------------|
| Readers block writers | No | No |
| Writers block readers | No | No |
| Writers block writers | Yes (same row) | Yes (same row) |
| Dead version cleanup | VACUUM (manual/auto) | Purge thread (automatic) |
| Long transaction cost | Table bloat | Long undo log chains |

The key insight: **reads never block writes, and writes never block reads.** This is why modern databases can handle thousands of concurrent connections without everything grinding to a halt.

## Locking

Despite MVCC, databases still need locks when multiple transactions write to the same data.

![Lock types hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-lock-types.png)


### Row-Level Locks

```sql
-- SELECT FOR UPDATE acquires a row lock
-- Other transactions trying to update the same row will wait
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Row is now locked. Other transactions wait if they try to modify it.
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;  -- Lock released
```

```sql
-- SELECT FOR SHARE (read lock)
-- Other transactions can read but not modify
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
-- Row is read-locked. Others can read it but not UPDATE/DELETE.
COMMIT;
```

```sql
-- SKIP LOCKED: non-blocking queue pattern
-- Great for job queues — workers grab unlocked tasks
BEGIN;
SELECT * FROM tasks
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
-- If a row is locked by another worker, skip it and get the next one

UPDATE tasks SET status = 'processing', worker_id = 'worker-3'
WHERE task_id = ...;  -- the ID from the SELECT above
COMMIT;
```

```sql
-- NOWAIT: fail immediately instead of waiting
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;
-- If locked, immediately raises: ERROR: could not obtain lock on row
```

### Table-Level Locks

Table locks are rare in OLTP but appear in DDL operations:

```sql
-- Explicit table lock (PostgreSQL)
LOCK TABLE accounts IN EXCLUSIVE MODE;

-- ACCESS EXCLUSIVE: blocks everything, required for DROP TABLE, ALTER TABLE
-- ACCESS SHARE: compatible with everything except ACCESS EXCLUSIVE
```

PostgreSQL lock modes (from weakest to strongest):

| Lock Mode | Conflicts With |
|-----------|---------------|
| ACCESS SHARE | ACCESS EXCLUSIVE |
| ROW SHARE | EXCLUSIVE, ACCESS EXCLUSIVE |
| ROW EXCLUSIVE | SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE UPDATE EXCLUSIVE | SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE | ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE ROW EXCLUSIVE | ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| EXCLUSIVE | ROW SHARE, ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| ACCESS EXCLUSIVE | All lock modes |

### Advisory Locks

Application-level locks using the database as a coordination point:

```sql
-- Acquire an advisory lock (PostgreSQL)
-- The lock number is arbitrary — your application defines the meaning
SELECT pg_advisory_lock(12345);
-- ... do work that needs exclusive access ...
SELECT pg_advisory_unlock(12345);

-- Try to acquire without blocking
SELECT pg_try_advisory_lock(12345);  -- Returns true/false

-- Session-level advisory locks (released when session ends)
SELECT pg_advisory_lock(hashtext('process_daily_report'));
```

Use cases: cron job coordination (only one instance runs), rate limiting, distributed locking without Redis.

## Deadlocks

A deadlock occurs when two transactions each hold a lock that the other needs.

![Deadlock detection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-deadlock-detection.png)


```sql
-- Transaction A                      -- Transaction B
BEGIN;                                BEGIN;
UPDATE accounts SET balance = 900
WHERE id = 1;  -- Locks row 1
                                      UPDATE accounts SET balance = 1100
                                      WHERE id = 2;  -- Locks row 2

UPDATE accounts SET balance = 1100
WHERE id = 2;  -- WAITS for B's lock
                                      UPDATE accounts SET balance = 900
                                      WHERE id = 1;  -- WAITS for A's lock

-- DEADLOCK! Both transactions are waiting for each other.
```

### How Databases Handle Deadlocks

Databases detect deadlocks using a **wait-for graph**. When a cycle is detected, one transaction is chosen as the victim and rolled back:

```text
ERROR:  deadlock detected
DETAIL: Process 12345 waits for ShareLock on transaction 67890;
        blocked by process 67891.
        Process 67891 waits for ShareLock on transaction 12345;
        blocked by process 12345.
HINT:   See server log for query details.
```

### Deadlock Prevention Strategies

1. **Always lock resources in the same order**: If every transaction that needs accounts 1 and 2 always locks account 1 first, deadlocks cannot occur.

```sql
-- Always lock the lower account_id first
BEGIN;
SELECT * FROM accounts WHERE id = LEAST(1, 2) FOR UPDATE;
SELECT * FROM accounts WHERE id = GREATEST(1, 2) FOR UPDATE;
UPDATE accounts SET balance = balance - 500 WHERE id = 1;
UPDATE accounts SET balance = balance + 500 WHERE id = 2;
COMMIT;
```

2. **Keep transactions short**: The longer a transaction holds locks, the higher the chance of deadlock.

3. **Use NOWAIT or lock timeouts**:

```sql
-- PostgreSQL: fail fast on lock contention
SET lock_timeout = '5s';
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- If locked for > 5 seconds, abort instead of waiting forever
```

4. **Reduce lock scope**: Lock only what you need, when you need it.

## Optimistic vs Pessimistic Concurrency

Two fundamentally different approaches to handling concurrent modifications:

### Pessimistic Locking

Lock the data before modifying it. This is what `SELECT FOR UPDATE` does.

```sql
-- Pessimistic: lock the row first
BEGIN;
SELECT * FROM products WHERE product_id = 42 FOR UPDATE;
-- Check stock, calculate, etc.
UPDATE products SET stock = stock - 1 WHERE product_id = 42;
COMMIT;
```

**Pros**: Simple, guaranteed correctness.
**Cons**: Reduces throughput, risk of deadlocks, does not work across distributed systems.

### Optimistic Locking

Do not lock anything. Instead, detect conflicts at commit time using a version number or timestamp.

```sql
-- Add a version column
ALTER TABLE products ADD COLUMN version INT NOT NULL DEFAULT 1;

-- Read the current state (no lock)
SELECT product_id, stock, version FROM products WHERE product_id = 42;
-- Returns: stock = 10, version = 5

-- Application does its computation...

-- Update with version check
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE product_id = 42 AND version = 5;
-- If affected rows = 0, someone else modified the row. Retry.
```

```python
# Application-side optimistic locking pattern
def purchase_product(product_id: int, quantity: int):
    max_retries = 3
    for attempt in range(max_retries):
        # Read current state
        product = db.query(
            "SELECT stock, version FROM products WHERE product_id = %s",
            [product_id]
        )

        if product.stock < quantity:
            raise InsufficientStockError()

        # Try to update with version check
        rows_affected = db.execute(
            """UPDATE products
               SET stock = stock - %s, version = version + 1
               WHERE product_id = %s AND version = %s""",
            [quantity, product_id, product.version]
        )

        if rows_affected == 1:
            return  # Success
        # Version mismatch — retry
    raise ConflictError("Too many retries")
```

**Pros**: Higher throughput when conflicts are rare, works across distributed systems.
**Cons**: Must handle retries, more complex application logic.

| Criterion | Pessimistic | Optimistic |
|-----------|------------|------------|
| Conflict frequency | High — locks prevent conflicts | Low — detects conflicts after the fact |
| Throughput | Lower (waiting for locks) | Higher (no waiting) |
| Complexity | Simple SQL | Application must handle retries |
| Deadlock risk | Yes | No |
| Best for | High-contention data (inventory, balances) | Low-contention data (user profiles, settings) |

## Real Example: Concurrent Bank Transfer

Let us put it all together with a realistic bank transfer scenario.

```sql
-- The safe transfer function (PostgreSQL)
CREATE OR REPLACE FUNCTION transfer(
    from_account INT,
    to_account INT,
    amount DECIMAL(12, 2)
) RETURNS VOID AS $$
DECLARE
    from_balance DECIMAL(12, 2);
BEGIN
    -- Lock both accounts in consistent order (lowest ID first)
    -- This prevents deadlocks
    IF from_account < to_account THEN
        PERFORM 1 FROM accounts WHERE account_id = from_account FOR UPDATE;
        PERFORM 1 FROM accounts WHERE account_id = to_account FOR UPDATE;
    ELSE
        PERFORM 1 FROM accounts WHERE account_id = to_account FOR UPDATE;
        PERFORM 1 FROM accounts WHERE account_id = from_account FOR UPDATE;
    END IF;

    -- Check sufficient funds
    SELECT balance INTO from_balance
    FROM accounts WHERE account_id = from_account;

    IF from_balance < amount THEN
        RAISE EXCEPTION 'Insufficient funds: balance=%, amount=%',
            from_balance, amount;
    END IF;

    -- Perform transfer
    UPDATE accounts SET balance = balance - amount
    WHERE account_id = from_account;

    UPDATE accounts SET balance = balance + amount
    WHERE account_id = to_account;

    -- Log the transfer
    INSERT INTO transfer_log (from_account, to_account, amount, transferred_at)
    VALUES (from_account, to_account, amount, NOW());
END;
$$ LANGUAGE plpgsql;
```

```sql
-- Usage
BEGIN;
SELECT transfer(1, 2, 500.00);
COMMIT;
```

This function:
1. Locks accounts in consistent order (prevents deadlocks)
2. Checks sufficient funds after locking (prevents race condition)
3. Performs both updates in a single transaction (atomicity)
4. Logs the transfer for audit trail
5. Runs at the default isolation level (READ COMMITTED is sufficient here because we hold explicit row locks)

## Monitoring Lock Contention

```sql
-- PostgreSQL: view current locks
SELECT
    l.pid,
    a.usename,
    l.locktype,
    l.relation::regclass AS table_name,
    l.mode,
    l.granted,
    a.query,
    age(now(), a.query_start) AS query_age
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY a.query_start;

-- Find blocking queries
SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query,
    age(now(), blocked.query_start) AS waiting_time
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocked_locks.locktype = blocking_locks.locktype
    AND blocked_locks.relation = blocking_locks.relation
    AND blocked_locks.pid != blocking_locks.pid
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;
```

```sql
-- MySQL: view InnoDB lock waits
SELECT
    r.trx_id AS waiting_trx,
    r.trx_mysql_thread_id AS waiting_thread,
    r.trx_query AS waiting_query,
    b.trx_id AS blocking_trx,
    b.trx_mysql_thread_id AS blocking_thread,
    b.trx_query AS blocking_query
FROM information_schema.innodb_lock_waits w
JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;
```

## What's Next

Transactions guarantee correctness at the logical level. But how does the database actually store data on disk? How does a `COMMIT` survive a power failure? In the next article, we will go a level deeper and explore **storage engines** — the machinery that turns SQL into bytes on disk.
