---
title: "数据库（三）：事务与并发控制——ACID、隔离级别与锁机制"
date: 2024-04-21 09:00:00
tags:
  - Databases
  - SQL
  - Transactions
  - Concurrency
categories: Databases
series: databases
lang: zh
description: "一份详尽指南：涵盖 ACID 特性、隔离级别、MVCC、锁策略及死锁预防——每个概念均附带可运行的 SQL 示例。"
disableNunjucks: true
series_order: 3
translationKey: "databases-3"
---
任何处理资金、库存或任何关键状态的应用，最终都会遭遇并发 Bug：两名用户同时抢购最后一件商品；一笔银行转账从一个账户扣款成功，却在向另一账户入账前崩溃；一份报表读取到半更新的数据，输出荒谬的统计结果。事务（Transaction）正是为防止此类故障而生——理解其工作原理，对构建生产级系统而言绝非可选项，而是必修课。

---

## 什么是事务？

事务是一组被数据库视为单一逻辑单元的操作。这些操作要么**全部成功**，要么**全部失败**。

![隔离级别异常](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-isolation-anomalies.png)

```sql
BEGIN;
    UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
    UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;
COMMIT;
```

若服务器在两条 `UPDATE` 语句之间崩溃，整个事务将被回滚（rollback）。账户 1 不会凭空损失 $500，而账户 2 却未获得相应入账。这便是事务最根本的保障。

## ACID：四大核心保证

ACID 不仅是面试中需要背诵的缩写词。每个字母都代表一项具体保障；真正重要的是理解——**缺少其中任一保障时，系统究竟会出什么问题**，而非仅仅记住定义本身。

### 原子性（Atomicity）——全有或全无

**定义**：事务要么完全执行成功，要么完全不产生任何效果。

**缺失原子性时会发生什么**：

```sql
-- 若无原子性：服务器在两条语句之间崩溃
UPDATE inventory SET stock = stock - 1 WHERE product_id = 42;
-- 此处发生崩溃
INSERT INTO order_items (order_id, product_id, quantity) VALUES (101, 42, 1);
-- 库存已扣减，但订单明细未创建 → 库存泄漏（inventory leak）
```

原子性依赖数据库的**预写日志（Write-Ahead Log, WAL）**：所有变更先写入 WAL，再应用到数据页。崩溃恢复时，未完成的事务将被自动回滚。

### 一致性（Consistency）——从有效状态到有效状态

**定义**：事务将数据库从一个满足所有约束的有效状态，迁移至另一个同样有效的状态。所有约束（外键、CHECK、UNIQUE、NOT NULL）均被强制执行。

**缺失一致性时会发生什么**：

```sql
-- 若无一致性检查：
INSERT INTO orders (order_id, user_id) VALUES (999, 12345);
-- user_id 12345 在 users 表中并不存在
-- 现在我们拥有一条“孤儿”订单（orphaned order），无关联用户
```

外键约束 `REFERENCES users(user_id)` 可阻止该错误。一致性还涵盖应用层不变量（invariant），例如：“转账过程中，所有账户余额之和必须保持恒定”。

### 隔离性（Isolation）——并发事务互不干扰

**定义**：多个并发执行的事务，其效果等价于以某种顺序串行执行。

![隔离级别比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-isolation-levels.png)

**缺失隔离性时会发生什么**：

```sql
-- 事务 A：读取账户余额
SELECT balance FROM accounts WHERE account_id = 1;  -- 返回 1000

-- 事务 B：取款 500 并提交
UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
COMMIT;

-- 事务 A：再次读取（同一事务内）
SELECT balance FROM accounts WHERE account_id = 1;  -- 返回 500！
-- 余额在事务中途被修改 → 事务 A 对世界的视图已不一致。
```

这就是**不可重复读（Non-Repeatable Read）**。隔离级别（isolation level）即用于控制哪些异常现象被允许发生。

### 持久性（Durability）——已提交即永久

**定义**：一旦事务成功提交，其变更将能抵御后续任何崩溃（断电、操作系统崩溃、硬件故障）。

**缺失持久性时会发生什么**：你提交了一笔银行转账，屏幕上显示“成功”，随后服务器重启——转账记录消失。数据库回退到了提交前的状态，因为变更仅驻留在内存中。

持久性通过**在报告提交成功前，将 WAL 刷写（flush）至持久化存储**来保障。实际的数据页可能仍处于内存中（称为 dirty pages），但 WAL 已包含足够信息，可在崩溃后重建这些页面。

## 事务生命周期


![MVCC时间线可视化：数据库的平行宇宙](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/03-mvcc-timeline-visualization-parallel-universes-of-database-s.jpg)

```sql
-- 启动事务
BEGIN;
-- 或：START TRANSACTION;

-- 执行操作
INSERT INTO orders (user_id, status) VALUES (1, 'pending');
UPDATE inventory SET stock = stock - 1 WHERE product_id = 42;

-- 创建保存点（savepoint）——支持部分回滚
SAVEPOINT after_inventory;

-- 继续操作
INSERT INTO shipping (order_id, address) VALUES (currval('orders_order_id_seq'), '123 Main St');

-- 哎呀，地址错了。回滚至保存点。
ROLLBACK TO SAVEPOINT after_inventory;

-- 修正后重试
INSERT INTO shipping (order_id, address) VALUES (currval('orders_order_id_seq'), '456 Oak Ave');

-- 提交全部变更
COMMIT;
```

| 命令 | 效果 |
|---------|--------|
| `BEGIN` | 启动一个新事务 |
| `COMMIT` | 将所有变更永久化 |
| `ROLLBACK` | 撤销自 `BEGIN` 以来的所有变更 |
| `SAVEPOINT name` | 在事务内创建一个具名检查点 |
| `ROLLBACK TO SAVEPOINT name` | 回滚至该保存点（事务继续执行） |
| `RELEASE SAVEPOINT name` | 删除该保存点（其变更保留） |

在自动提交模式（autocommit mode，大多数数据库默认启用）下，每条语句自身即为一个事务。显式 `BEGIN` 则开启多语句事务。

## 隔离级别


![数据库事务锁如同数字金库上的金色挂锁](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/03-database-transaction-locks-as-golden-padlocks-on-digital-vau.jpg)

SQL 标准定义了四种隔离级别，各自允许不同类型的并发异常。更强的隔离性意味着更少的异常，但吞吐量也会随之降低。

### 三大并发异常

在介绍隔离级别前，我们先精确定义这三种异常。

**脏读（Dirty Read）**：事务 A 读取了事务 B 写入但尚未提交的数据。若 B 回滚，则 A 读取了从未正式存在的数据。

```sql
-- 会话 1                          -- 会话 2
BEGIN;
UPDATE products SET price = 0.01
WHERE product_id = 1;
                                      BEGIN;
                                      -- 脏读：看到 price = 0.01
                                      SELECT price FROM products
                                      WHERE product_id = 1;
ROLLBACK;  -- price 恢复为 49.99
                                      -- 会话 2 基于 price 0.01 做出了决策，
                                      -- 但该价格从未真实存在过。
                                      COMMIT;
```

**不可重复读（Non-Repeatable Read）**：事务 A 读取某行，事务 B 修改并提交该行，然后 A 再次读取同一行，得到不同结果。

```sql
-- 会话 1                          -- 会话 2
BEGIN;
SELECT balance FROM accounts
WHERE id = 1;  -- 返回 1000
                                      BEGIN;
                                      UPDATE accounts SET balance = 500
                                      WHERE id = 1;
                                      COMMIT;
SELECT balance FROM accounts
WHERE id = 1;  -- 返回 500！
-- 同一查询，在同一事务内返回不同结果。
COMMIT;
```

**幻读（Phantom Read）**：事务 A 执行一条带范围条件的查询，事务 B 插入一条新行且满足该条件，然后 A 再次执行相同查询，发现了一条新的“幻影”行（phantom row）。

```sql
-- 会话 1                          -- 会话 2
BEGIN;
SELECT COUNT(*) FROM orders
WHERE status = 'pending';  -- 返回 5
                                      BEGIN;
                                      INSERT INTO orders (user_id, status)
                                      VALUES (99, 'pending');
                                      COMMIT;
SELECT COUNT(*) FROM orders
WHERE status = 'pending';  -- 返回 6！
-- 新增了一行（幻影行）。
COMMIT;
```

### 隔离级别对照表

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 性能 |
|----------------|------------|--------------------:|-------------:|-------------|
| READ UNCOMMITTED | 允许 | 允许 | 允许 | 最快 |
| READ COMMITTED | **禁止** | 允许 | 允许 | 快 |
| REPEATABLE READ | **禁止** | **禁止** | 允许* | 中等 |
| SERIALIZABLE | **禁止** | **禁止** | **禁止** | 最慢 |

*PostgreSQL 中，REPEATABLE READ 也禁止幻读（它采用快照隔离，比 SQL 标准要求更强）。

```sql
-- 为单个事务设置隔离级别
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- ... your queries ...
COMMIT;

-- 为会话设置默认隔离级别（PostgreSQL）
SET default_transaction_isolation = 'read committed';

-- 查看当前会话默认隔离级别（PostgreSQL）
SHOW default_transaction_isolation;

-- MySQL
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT @@transaction_isolation;
```

### 该如何选择？

| 使用场景 | 推荐级别 |
|----------|------------------|
| 大多数 Web 应用 | READ COMMITTED（PostgreSQL 默认） |
| 金融交易 | SERIALIZABLE 或 REPEATABLE READ |
| 报表/分析类查询 | REPEATABLE READ（提供一致快照） |
| 尽力而为型/监控类任务 | READ UNCOMMITTED（仅当真有此需求） |

PostgreSQL 默认使用 READ COMMITTED；MySQL（InnoDB）默认使用 REPEATABLE READ。两者对大多数应用而言都是合理的选择。

## MVCC：数据库如何高效实现隔离性

多版本并发控制（Multi-Version Concurrency Control, MVCC）是让隔离级别具备实用性的核心机制。它不采用“写时阻塞读”的低效方式（这会严重损害性能），而是为每一行维护多个版本。

![多版本并发控制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-mvcc-timeline.png)

### PostgreSQL 的 MVCC 实现

PostgreSQL 中，每行数据包含隐藏的系统列：

- `xmin` —— 创建（插入）该行版本的事务 ID
- `xmax` —— 删除/更新该行版本的事务 ID（若值为 0，表示该版本仍有效）

当你执行 `UPDATE` 时，PostgreSQL **不会就地修改**原行，而是：
1. 将旧行版本标记为过期（`xmax = 当前事务 ID`）
2. 创建一个新行版本（`xmin = 当前事务 ID`）

新旧两个版本共存。每个事务仅能看到在其快照时间点“存活”的那个版本。

```sql
-- 事务 100 插入一行
INSERT INTO accounts (id, balance) VALUES (1, 1000);
-- 行：xmin=100, xmax=0, balance=1000

-- 事务 200 更新该行
UPDATE accounts SET balance = 500 WHERE id = 1;
-- 旧行：xmin=100, xmax=200, balance=1000  （对旧快照仍可见）
-- 新行：xmin=200, xmax=0,   balance=500   （对新快照可见）
```

这正是 PostgreSQL 需要 `VACUUM` 的原因——已失效的行版本不断累积，必须被清理。

### MySQL InnoDB 的 MVCC 实现

InnoDB 采用不同策略：
- 每行包含一个隐藏的 6 字节事务 ID 和一个 7 字节回滚指针（roll pointer）
- 回滚指针指向**undo log**中的条目，其中保存了该行的前一版本
- 多个 undo log 条目为每行构成一条链

为重构旧版本，InnoDB 沿 undo log 链反向遍历。这意味着旧版本不占用主表空间，但长事务会迫使 InnoDB 保留冗长的 undo log 链。

### MVCC 的影响对比

| 行为 | PostgreSQL | MySQL InnoDB |
|----------|-----------|--------------|
| 读操作阻塞写操作 | 否 | 否 |
| 写操作阻塞读操作 | 否 | 否 |
| 写操作阻塞写操作 | 是（同一行） | 是（同一行） |
| 过期版本清理 | VACUUM（手动/自动） | Purge 线程（自动） |
| 长事务开销 | 表膨胀（bloat） | 长 undo log 链 |

关键洞见：**读操作从不阻塞写操作，写操作也从不阻塞读操作。** 这正是现代数据库能支撑数千并发连接而不致全面卡顿的根本原因。

## 锁机制（Locking）

尽管有 MVCC，当多个事务尝试写入相同数据时，数据库仍需加锁。

![锁类型层次结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-lock-types.png)

### 行级锁（Row-Level Locks）

```sql
-- SELECT FOR UPDATE 获取行锁
-- 其他事务若尝试更新同一行，将被阻塞等待
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- 此行已被锁定。其他事务若尝试修改它，将等待。
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;  -- 锁释放
```

```sql
-- SELECT FOR SHARE（共享读锁）
-- 其他事务可读，但不可修改
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
-- 此行被读锁保护。其他事务可读，但不能 UPDATE/DELETE。
COMMIT;
```

```sql
-- SKIP LOCKED：非阻塞队列模式
-- 适用于任务队列——工作者（worker）获取未被锁定的任务
BEGIN;
SELECT * FROM tasks
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
-- 若某行已被其他 worker 锁定，则跳过，获取下一条

UPDATE tasks SET status = 'processing', worker_id = 'worker-3'
WHERE task_id = ...;  -- 上述 SELECT 返回的 ID
COMMIT;
```

```sql
-- NOWAIT：立即失败，而非等待
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;
-- 若已被锁定，立即报错：ERROR: could not obtain lock on row
```

### 表级锁（Table-Level Locks）

表锁在 OLTP 场景中罕见，但常见于 DDL 操作：

```sql
-- 显式表锁（PostgreSQL）
LOCK TABLE accounts IN EXCLUSIVE MODE;

-- ACCESS EXCLUSIVE：阻塞一切操作，DROP TABLE、ALTER TABLE 所需
-- ACCESS SHARE：与除 ACCESS EXCLUSIVE 外的所有锁兼容
```

PostgreSQL 锁模式（由弱到强）：

| 锁模式 | 冲突对象 |
|-----------|---------------|
| ACCESS SHARE | ACCESS EXCLUSIVE |
| ROW SHARE | EXCLUSIVE, ACCESS EXCLUSIVE |
| ROW EXCLUSIVE | SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE UPDATE EXCLUSIVE | SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE | ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE ROW EXCLUSIVE | ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| EXCLUSIVE | ROW SHARE, ROW EXCLUSIVE, SHARE UPDATE EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| ACCESS EXCLUSIVE | 所有锁模式 |

### 顾问锁（Advisory Locks）

一种应用层锁，利用数据库作为协调中心：

```sql
-- 获取顾问锁（PostgreSQL）
-- 锁编号任意，由你的应用定义其含义
SELECT pg_advisory_lock(12345);
-- ... 执行需要独占访问的工作 ...
SELECT pg_advisory_unlock(12345);

-- 尝试获取（不阻塞）
SELECT pg_try_advisory_lock(12345);  -- 返回 true/false

-- 会话级顾问锁（会话结束时自动释放）
SELECT pg_advisory_lock(hashtext('process_daily_report'));
```

典型用例：定时任务（cron job）协调（确保仅一个实例运行）、速率限制（rate limiting）、无需 Redis 的分布式锁。

## 死锁（Deadlocks）

当两个事务各自持有对方所需锁时，即发生死锁。

![死锁检测](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/03-deadlock-detection.png)

```sql
-- 事务 A                      -- 事务 B
BEGIN;                                BEGIN;
UPDATE accounts SET balance = 900
WHERE id = 1;  -- 锁定第 1 行
                                      UPDATE accounts SET balance = 1100
                                      WHERE id = 2;  -- 锁定第 2 行

UPDATE accounts SET balance = 1100
WHERE id = 2;  -- 等待事务 B 的锁
                                      UPDATE accounts SET balance = 900
                                      WHERE id = 1;  -- 等待事务 A 的锁

-- 死锁！双方都在等待对方释放锁。
```

### 数据库如何处理死锁

数据库通过**等待图（wait-for graph）** 检测死锁。一旦发现环路，便选择一个事务作为牺牲者（victim）并将其回滚：

```text
ERROR:  deadlock detected
DETAIL: Process 12345 waits for ShareLock on transaction 67890;
        blocked by process 67891.
        Process 67891 waits for ShareLock on transaction 12345;
        blocked by process 12345.
HINT:   See server log for query details.
```

### 死锁预防策略

1. **始终按相同顺序锁定资源**：若所有需要账户 1 和 2 的事务，都先锁定 `account_id` 较小的那个，则死锁不可能发生。

```sql
-- 始终先锁定较小的 account_id
BEGIN;
SELECT * FROM accounts WHERE id = LEAST(1, 2) FOR UPDATE;
SELECT * FROM accounts WHERE id = GREATEST(1, 2) FOR UPDATE;
UPDATE accounts SET balance = balance - 500 WHERE id = 1;
UPDATE accounts SET balance = balance + 500 WHERE id = 2;
COMMIT;
```

2. **保持事务简短**：事务持有锁的时间越长，发生死锁的概率越高。

3. **使用 NOWAIT 或锁超时**：

```sql
-- PostgreSQL：在锁竞争时快速失败
SET lock_timeout = '5s';
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- 若被锁超过 5 秒，则中止，而非无限等待
```

4. **缩小锁作用域**：只在真正需要时，锁定真正需要的数据。

## 乐观并发 vs 悲观并发

处理并发修改的两种根本不同思路：

### 悲观锁（Pessimistic Locking）

在修改前先锁定数据。`SELECT FOR UPDATE` 即属此类。

```sql
-- 悲观锁：先锁定行
BEGIN;
SELECT * FROM products WHERE product_id = 42 FOR UPDATE;
-- 检查库存、计算等
UPDATE products SET stock = stock - 1 WHERE product_id = 42;
COMMIT;
```

**优点**：简单，正确性有绝对保障。  
**缺点**：降低吞吐量，有死锁风险，不适用于分布式系统。

### 乐观锁（Optimistic Locking）

不预先加锁，而是在提交时通过版本号或时间戳检测冲突。

```sql
-- 添加版本列
ALTER TABLE products ADD COLUMN version INT NOT NULL DEFAULT 1;

-- 读取当前状态（无锁）
SELECT product_id, stock, version FROM products WHERE product_id = 42;
-- 返回：stock = 10, version = 5

-- 应用层进行计算...

-- 带版本检查的更新
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE product_id = 42 AND version = 5;
-- 若受影响行数为 0，说明他人已修改该行 → 需重试。
```

```python
# 应用层乐观锁模式
def purchase_product(product_id: int, quantity: int):
    max_retries = 3
    for attempt in range(max_retries):
        # 读取当前状态
        product = db.query(
            "SELECT stock, version FROM products WHERE product_id = %s",
            [product_id]
        )

        if product.stock < quantity:
            raise InsufficientStockError()

        # 尝试带版本检查的更新
        rows_affected = db.execute(
            """UPDATE products
               SET stock = stock - %s, version = version + 1
               WHERE product_id = %s AND version = %s""",
            [quantity, product_id, product.version]
        )

        if rows_affected == 1:
            return  # 成功
        # 版本不匹配 → 重试
    raise ConflictError("Too many retries")
```

**优点**：冲突率低时吞吐量更高，天然适用于分布式系统。  
**缺点**：需处理重试逻辑，应用层代码更复杂。

| 维度 | 悲观锁 | 乐观锁 |
|-----------|------------|------------|
| 冲突频率 | 高——锁主动预防冲突 | 低——事后检测冲突 |
| 吞吐量 | 较低（等待锁） | 较高（无等待） |
| 复杂度 | 简单 SQL | 应用需处理重试 |
| 死锁风险 | 是 | 否 |
| 最佳适用场景 | 高争用数据（库存、余额） | 低争用数据（用户资料、配置项） |

## 真实案例：并发银行转账

让我们整合所有知识，构建一个真实的银行转账场景。

```sql
-- 安全转账函数（PostgreSQL）
CREATE OR REPLACE FUNCTION transfer(
    from_account INT,
    to_account INT,
    amount DECIMAL(12, 2)
) RETURNS VOID AS $$
DECLARE
    from_balance DECIMAL(12, 2);
BEGIN
    -- 按一致顺序锁定两个账户（ID 小者优先）
    -- 此举可预防死锁
    IF from_account < to_account THEN
        PERFORM 1 FROM accounts WHERE account_id = from_account FOR UPDATE;
        PERFORM 1 FROM accounts WHERE account_id = to_account FOR UPDATE;
    ELSE
        PERFORM 1 FROM accounts WHERE account_id = to_account FOR UPDATE;
        PERFORM 1 FROM accounts WHERE account_id = from_account FOR UPDATE;
    END IF;

    -- 锁定后检查余额（防止竞态条件）
    SELECT balance INTO from_balance
    FROM accounts WHERE account_id = from_account;

    IF from_balance < amount THEN
        RAISE EXCEPTION 'Insufficient funds: balance=%, amount=%',
            from_balance, amount;
    END IF;

    -- 执行转账
    UPDATE accounts SET balance = balance - amount
    WHERE account_id = from_account;

    UPDATE accounts SET balance = balance + amount
    WHERE account_id = to_account;

    -- 记录转账日志
    INSERT INTO transfer_log (from_account, to_account, amount, transferred_at)
    VALUES (from_account, to_account, amount, NOW());
END;
$$ LANGUAGE plpgsql;
```

```sql
-- 调用方式
BEGIN;
SELECT transfer(1, 2, 500.00);
COMMIT;
```

该函数：
1. 按一致顺序（ID 小者优先）锁定账户（防死锁）
2. 锁定后检查余额（防竞态条件）
3. 在单个事务内执行两次更新（保障原子性）
4. 记录转账日志（满足审计追踪）
5. 运行于默认隔离级别（READ COMMITTED 在此处已足够，因为我们显式持有了行锁）

## 监控锁争用

```sql
-- PostgreSQL：查看当前锁状态
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

-- 查找阻塞查询
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
-- MySQL：查看 InnoDB 锁等待
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

## 下一步

事务保障了逻辑层面的正确性。但数据库究竟如何将数据持久化到磁盘？一次 `COMMIT` 如何扛住断电冲击？在下一篇文章中，我们将深入底层，探索**存储引擎（Storage Engines）**——那套将 SQL 语句转化为磁盘字节的精密机械。
