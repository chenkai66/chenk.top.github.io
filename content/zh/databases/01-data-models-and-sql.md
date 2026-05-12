---
title: "数据库（一）：数据模型与 SQL —— 为何关系型表结构至今仍占主导地位"
date: 2024-04-17 09:00:00
tags:
  - Databases
  - SQL
  - Relational Model
  - Normalization
categories: Databases
series: databases
lang: zh
description: "从零开始深入关系模型、SQL 基础、范式化设计及高级查询模式——掌握流利使用 SQL 所需的一切知识。"
disableNunjucks: true
series_order: 1
translationKey: "databases-1"
---

你用过的每一个应用程序，其底层都依赖某种数据模型。若选错模型，接下来三年你将疲于与自己的数据库搏斗，而非交付新功能。

过去四十年间，一种模型始终占据主导地位：**关系模型（Relational Model）**——扁平的表结构、外键、SQL。它不炫酷，也不时髦；但几乎每一家银行、航空公司、医院和电商平台仍在运行它，自有其深刻原因。理解 *为何如此*，是你真正理解数据库的第一步。

## 关系模型：科德（Codd）的伟大构想

1970 年，埃德加·F·科德（Edgar F. Codd）发表了论文《大型共享数据库的关系数据模型》（"A Relational Model of Data for Large Shared Data Banks"）。其核心洞见在当时极为激进：将数据的 **逻辑表示** 与 **物理存储** 彻底分离。应用程序无需关心数据究竟存于磁盘、内存，还是跨十台机器分布；它们只需看到 **表（tables）**——仅此而已。

![Normalization forms comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/01-normalization-forms.png)


关系型数据库将数据组织为 **关系（relations）**（即表）。每张表包含：

- **列（columns）**（属性）——带类型的字段，例如 `name VARCHAR(100)` 或 `price DECIMAL(10,2)`
- **行（rows）**（元组）——单条记录
- **主键（primary key）**——一个（或一组）能唯一标识每一行的列
- **外键（foreign key）**——一个引用另一张表主键的列，用于建立表间关联

这四个概念，足以建模出令人惊讶的复杂业务领域。

## 实用 Schema 示例：电商系统

理论结合具体表结构更易理解。以下是一个极简的电商 Schema，本文后续将反复使用：

![Entity-Relationship diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/01-er-diagram.png)


```sql
CREATE TABLE users (
    user_id     SERIAL PRIMARY KEY,
    email       VARCHAR(255) NOT NULL UNIQUE,
    full_name   VARCHAR(200) NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    product_id  SERIAL PRIMARY KEY,
    name        VARCHAR(300) NOT NULL,
    category    VARCHAR(100),
    price       DECIMAL(10, 2) NOT NULL,
    stock       INT NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    order_id    SERIAL PRIMARY KEY,
    user_id     INT NOT NULL REFERENCES users(user_id),
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    item_id     SERIAL PRIMARY KEY,
    order_id    INT NOT NULL REFERENCES orders(order_id),
    product_id  INT NOT NULL REFERENCES products(product_id),
    quantity    INT NOT NULL CHECK (quantity > 0),
    unit_price  DECIMAL(10, 2) NOT NULL
);
```

四张表，三个外键。这已足够表达“用户下单购买多种商品”这一核心场景——正是支撑着年营收达数十亿美元公司的模型。

## SQL 基础要点


![Abstract visualization of relational database tables connect](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/01-abstract-visualization-of-relational-database-tables-connect.jpg)

SQL（Structured Query Language）是与关系型数据库对话的语言。它是**声明式（declarative）**的：你只需描述 *想要什么数据*，而非 *如何获取它*；执行计划由数据库引擎自行推导。

![SQL query execution pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/01-sql-query-flow.png)


### SELECT、FROM、WHERE

最基础的查询：

```sql
-- 查找价格低于 $500 的 "Electronics" 类别商品
SELECT product_id, name, price
FROM products
WHERE category = 'Electronics'
  AND price < 500.00;
```

输出：

```
 product_id |         name          | price  
------------+-----------------------+--------
          3 | Wireless Mouse        |  29.99
          7 | USB-C Hub             |  45.00
         12 | Mechanical Keyboard   | 149.99
         18 | Noise-Cancelling Buds | 199.00
```

### ORDER BY 和 LIMIT

```sql
-- 最贵的 5 款商品
SELECT name, price
FROM products
ORDER BY price DESC
LIMIT 5;
```

### JOIN：关系型查询的核心

JOIN 将多张表的行组合起来。你需要掌握四种类型：

![SQL join types visualized](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/01-join-types.png)


| JOIN 类型 | 返回结果 |
|-----------|---------|
| `INNER JOIN` | 仅返回在**两张表中均有匹配**的行 |
| `LEFT JOIN` | 返回左表所有行；右表无匹配时对应列为 `NULL` |
| `RIGHT JOIN` | 返回右表所有行；左表无匹配时对应列为 `NULL` |
| `FULL OUTER JOIN` | 返回两张表所有行；任一侧无匹配时对应列为 `NULL` |

```sql
-- 获取所有已完成订单及其用户信息与商品明细
SELECT
    u.full_name,
    o.order_id,
    p.name AS product_name,
    oi.quantity,
    oi.unit_price,
    (oi.quantity * oi.unit_price) AS line_total
FROM orders o
INNER JOIN users u ON o.user_id = u.user_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
WHERE o.status = 'completed'
ORDER BY o.order_id;
```

输出：

```
  full_name   | order_id |    product_name     | quantity | unit_price | line_total
--------------+----------+---------------------+----------+------------+------------
 Alice Chen   |        1 | Mechanical Keyboard |        1 |     149.99 |     149.99
 Alice Chen   |        1 | USB-C Hub           |        2 |      45.00 |      90.00
 Bob Martinez |        3 | Wireless Mouse      |        3 |      29.99 |      89.97
 Bob Martinez |        3 | Standing Desk       |        1 |     599.00 |     599.00
```

当需要保留左表所有行（即使右表无匹配）时，`LEFT JOIN` 至关重要：

```sql
-- 查找从未下过单的用户
SELECT u.user_id, u.email, u.full_name
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_id IS NULL;
```

### GROUP BY 和 HAVING

聚合操作将多行压缩为汇总行：

```sql
-- 各品类营收（仅显示营收 > $1000 的品类）
SELECT
    p.category,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(oi.quantity) AS total_units,
    SUM(oi.quantity * oi.unit_price) AS total_revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p.category
HAVING SUM(oi.quantity * oi.unit_price) > 1000.00
ORDER BY total_revenue DESC;
```

输出：

```
  category    | order_count | total_units | total_revenue
--------------+-------------+-------------+--------------
 Electronics  |          47 |         132 |      18432.50
 Furniture    |          23 |          31 |      12890.00
 Books        |          89 |         234 |       4567.80
```

`WHERE` 在分组前过滤行；`HAVING` 在聚合后过滤分组。这个区别常令初学者困惑。

## 数据类型：了解你的选项

选择合适的数据类型直接影响存储开销、性能与数据正确性。以下是 PostgreSQL 与 MySQL 中常见类型的对比：

| 类型 | PostgreSQL | MySQL | 字节数 | 取值范围 / 说明 |
|------|-----------|-------|--------|-----------------|
| 小整数 | `SMALLINT` | `SMALLINT` | 2 | -32,768 到 32,767 |
| 整数 | `INT` / `INTEGER` | `INT` | 4 | -21 亿 到 21 亿 |
| 大整数 | `BIGINT` | `BIGINT` | 8 | -9.2 × 10¹⁸ 到 9.2 × 10¹⁸ |
| 可变长文本 | `VARCHAR(n)` | `VARCHAR(n)` | 1-4 + 长度 | 最大 1GB（PG），65,535 字节（MySQL） |
| 无限长文本 | `TEXT` | `TEXT` / `LONGTEXT` | 1-4 + 长度 | 无长度限制（PG），4GB（MySQL LONGTEXT） |
| 精确小数 | `DECIMAL(p,s)` / `NUMERIC` | `DECIMAL(p,s)` | 可变 | 货币场景必用。**切勿用 FLOAT 存货币！** |
| 时间戳 | `TIMESTAMP` / `TIMESTAMPTZ` | `TIMESTAMP` / `DATETIME` | 8 | PG 中务必用 `TIMESTAMPTZ` |
| 布尔值 | `BOOLEAN` | `BOOLEAN` / `TINYINT(1)` | 1 | MySQL 的 `BOOLEAN` 实为 `TINYINT` |
| JSON | `JSON` / `JSONB` | `JSON` | 可变 | `JSONB`（PG）为二进制格式，可索引且更快 |
| UUID | `UUID` | `CHAR(36)` 或 `BINARY(16)` | 16 | PG 原生支持，MySQL 需模拟 |

**经验法则：**

- 对任何可能大规模增长的表，主键优先选用 `BIGINT`。`INT` 最多仅支持约 21 亿行。
- 货币值一律使用 `DECIMAL(10,2)`。`FLOAT`/`DOUBLE` 会引入舍入误差。
- PostgreSQL 使用 `TIMESTAMPTZ`；或统一以 UTC 存储所有时间。
- 仅当确实需要 schema-free 字段时才用 `JSONB`（PostgreSQL）；**切勿用它逃避合理 schema 设计。**

## ALTER TABLE：Schema 演进

Schema 必然随业务演进。新功能常需新增列：

![The relational model in action](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/databases/01-relational-model.png)


```sql
-- 添加电话号码列
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 为现有列设置默认值
ALTER TABLE products ALTER COLUMN stock SET DEFAULT 0;

-- 重命名列（PostgreSQL）
ALTER TABLE products RENAME COLUMN name TO product_name;

-- 添加复合唯一约束
ALTER TABLE order_items ADD CONSTRAINT uq_order_product
    UNIQUE (order_id, product_id);

-- 删除列（生产环境务必谨慎！）
ALTER TABLE users DROP COLUMN phone;
```

在生产环境中，对大表执行 `ALTER TABLE` 可能导致表被锁住数分钟甚至数小时。第 8 篇文章将详解在线 DDL（Online DDL）策略。

## 范式化（Normalization）：消除冗余

范式化是通过组织列与表来减少数据冗余、防止更新异常的过程。

### 范式化前（反范式化混乱）

设想所有数据挤在一张表里：

```
| order_id | customer_name | customer_email      | product_name | product_price | quantity |
|----------|---------------|---------------------|--------------|---------------|----------|
| 1        | Alice Chen    | alice@example.com   | Keyboard     | 149.99        | 1        |
| 1        | Alice Chen    | alice@example.com   | USB Hub      | 45.00         | 2        |
| 2        | Alice Chen    | alice@example.com   | Mouse        | 29.99         | 1        |
| 3        | Bob Martinez  | bob@example.com     | Keyboard     | 149.99        | 1        |
```

问题：
- **更新异常（Update anomaly）**：Alice 修改邮箱？你必须更新她出现过的每一行。
- **插入异常（Insert anomaly）**：未产生订单，就无法添加新产品。
- **删除异常（Delete anomaly）**：删掉 Bob 唯一的一笔订单，他的客户信息也彻底丢失。

### 第一范式（1NF）

**规则**：每列只含原子（不可再分）值，禁止重复组。

违规示例：

```
| order_id | products                |
|----------|-------------------------|
| 1        | Keyboard, USB Hub       |  -- 逗号分隔 = 违反 1NF
```

修复：每个订单-商品组合单独一行（这正是我们的 `order_items` 表所做）。

### 第二范式（2NF）

**规则**：满足 1NF，且每个非主键列完全依赖于**整个**主键，而非主键的一部分。

这对含复合主键的表适用。若 `(order_id, product_id)` 是主键，则 `product_name` 仅依赖 `product_id` —— 这违反 2NF。应将产品数据移至独立的 `products` 表。

### 第三范式（3NF）

**规则**：满足 2NF，且无非主键列依赖于其他非主键列（即无传递依赖）。

违规示例：

```
| user_id | zip_code | city       | state |
|---------|----------|------------|-------|
| 1       | 94105    | San Francisco | CA |
```

`city` 和 `state` 依赖 `zip_code`，而非 `user_id`。修复：新建 `addresses` 或 `zip_codes` 查找表。

### 范式化后

我们当前的四表 Schema 已符合 3NF：
- `users` —— 仅用户数据
- `products` —— 仅产品数据
- `orders` —— 订单元数据，引用 `users`
- `order_items` —— 订单明细，同时引用 `orders` 与 `products`

零冗余，每个事实仅存储一次。

## 何时反范式化（Denormalize）


![Sql query processing engine mechanical gears and data pipes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/databases/01-sql-query-processing-engine-mechanical-gears-and-data-pipes-.jpg)

范式化消除了冗余，但读取常需 JOIN 多张表——而 JOIN 有开销。有时你**主动反范式化以换取性能**：

```sql
-- 为避免每次查询都 JOIN orders + order_items + products，
-- 直接在 orders 表中缓存预计算的总金额
ALTER TABLE orders ADD COLUMN total_amount DECIMAL(12, 2);

-- 当订单项变更时更新该字段
UPDATE orders o
SET total_amount = (
    SELECT SUM(oi.quantity * oi.unit_price)
    FROM order_items oi
    WHERE oi.order_id = o.order_id
)
WHERE o.order_id = 42;
```

常见反范式化模式：

| 模式 | 适用场景 | 权衡点 |
|------|----------|--------|
| 缓存聚合值（Cached aggregates） | 频繁运行的仪表板查询 | 每次写入均需同步更新 |
| 物化视图（Materialized views） | 复杂报表查询 | 刷新间隔内数据陈旧 |
| 冗余列（Redundant columns） | 热路径（hot paths）中规避昂贵 JOIN | 更新异常风险重现 |
| 汇总表（Summary tables） | 时序数据滚动聚合（按小时/天） | 额外存储开销，ETL 复杂性 |

准则：**先范式化，仅在实测存在性能瓶颈时才反范式化。**

## 高级 SQL：子查询、CTE 与窗口函数

### 子查询（Subqueries）

子查询是嵌套在另一查询内部的查询：

```sql
-- 查找总消费额超过平均用户消费额的用户
SELECT u.full_name, user_totals.total_spent
FROM users u
JOIN (
    SELECT o.user_id, SUM(oi.quantity * oi.unit_price) AS total_spent
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY o.user_id
) user_totals ON u.user_id = user_totals.user_id
WHERE user_totals.total_spent > (
    SELECT AVG(sub.total_spent)
    FROM (
        SELECT SUM(oi.quantity * oi.unit_price) AS total_spent
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'completed'
        GROUP BY o.user_id
    ) sub
)
ORDER BY user_totals.total_spent DESC;
```

此写法可行，但可读性差。CTE（Common Table Expressions）可解决此问题。

### 公共表表达式（CTEs）

`WITH` 子句允许你为中间结果集命名：

```sql
WITH user_spending AS (
    SELECT
        o.user_id,
        SUM(oi.quantity * oi.unit_price) AS total_spent
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY o.user_id
),
avg_spending AS (
    SELECT AVG(total_spent) AS avg_spent
    FROM user_spending
)
SELECT
    u.full_name,
    us.total_spent,
    a.avg_spent,
    ROUND(us.total_spent / a.avg_spent, 2) AS spending_ratio
FROM user_spending us
JOIN users u ON us.user_id = u.user_id
CROSS JOIN avg_spending a
WHERE us.total_spent > a.avg_spent
ORDER BY us.total_spent DESC;
```

输出：

```
  full_name    | total_spent | avg_spent | spending_ratio
---------------+-------------+-----------+----------------
 David Park    |     4523.90 |    892.34 |           5.07
 Alice Chen    |     2891.45 |    892.34 |           3.24
 Carol White   |     1567.20 |    892.34 |           1.76
```

CTE 更易读、可在同一查询中复用；部分数据库（如 PostgreSQL 12+）还能将其内联（inline）以提升性能。

### 递归 CTE（Recursive CTEs）

CTE 可自我引用，适用于层级数据：

```sql
-- 员工组织架构：查找某经理下属的所有员工
WITH RECURSIVE reports AS (
    -- 基础情况：经理本人
    SELECT employee_id, name, manager_id, 0 AS depth
    FROM employees
    WHERE employee_id = 1

    UNION ALL

    -- 递归情况：查找当前层级员工的直接下属
    SELECT e.employee_id, e.name, e.manager_id, r.depth + 1
    FROM employees e
    JOIN reports r ON e.manager_id = r.employee_id
)
SELECT depth, employee_id, name
FROM reports
ORDER BY depth, name;
```

### 窗口函数（Window Functions）

窗口函数在与当前行相关的行集上计算值，但**不压缩行**（与 `GROUP BY` 不同）：

```sql
-- 按月份统计用户消费，并在每月内排名
SELECT
    u.full_name,
    DATE_TRUNC('month', o.created_at) AS month,
    SUM(oi.quantity * oi.unit_price) AS monthly_spend,
    RANK() OVER (
        PARTITION BY DATE_TRUNC('month', o.created_at)
        ORDER BY SUM(oi.quantity * oi.unit_price) DESC
    ) AS spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY DATE_TRUNC('month', o.created_at)
        ORDER BY SUM(oi.quantity * oi.unit_price) DESC
    ) AS row_num
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY u.full_name, DATE_TRUNC('month', o.created_at)
ORDER BY month, spend_rank;
```

输出：

```
  full_name    |   month    | monthly_spend | spend_rank | row_num
---------------+------------+---------------+------------+---------
 David Park    | 2023-10-01 |       1245.00 |          1 |       1
 Alice Chen    | 2023-10-01 |        892.50 |          2 |       2
 Carol White   | 2023-10-01 |        892.50 |          2 |       3
 Bob Martinez  | 2023-10-01 |        334.99 |          4 |       4
 Alice Chen    | 2023-11-01 |       1567.20 |          1 |       1
 David Park    | 2023-11-01 |        998.00 |          2 |       2
```

注意：`RANK` 对并列值赋予相同排名（Alice 与 Carol 同为第 2 名），而 `ROW_NUMBER` 总是分配唯一序号。

**LAG 和 LEAD** 可访问前一行或后一行：

```sql
-- 对比本月营收与上月营收
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', o.created_at) AS month,
        SUM(oi.quantity * oi.unit_price) AS revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY DATE_TRUNC('month', o.created_at)
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / LAG(revenue) OVER (ORDER BY month) * 100,
        1
    ) AS growth_pct
FROM monthly_revenue
ORDER BY month;
```

输出：

```
   month    |  revenue  | prev_month_revenue | growth_pct
------------+-----------+--------------------+------------
 2023-08-01 |  12450.00 |                    |
 2023-09-01 |  15670.30 |          12450.00  |       25.9
 2023-10-01 |  14230.50 |          15670.30  |       -9.2
 2023-11-01 |  18900.00 |          14230.50  |       32.8
 2023-12-01 |  24560.80 |          18900.00  |       30.0
```

你应该掌握的常用窗口函数：

| 函数 | 用途 |
|----------|---------|
| `ROW_NUMBER()` | 顺序编号，无并列 |
| `RANK()` | 并列排名（有空缺，如 1, 2, 2, 4） |
| `DENSE_RANK()` | 并列排名（无空缺，如 1, 2, 2, 3） |
| `LAG(col, n)` | 当前行之前第 n 行的值 |
| `LEAD(col, n)` | 当前行之后第 n 行的值 |
| `SUM() OVER(...)` | 累计和（running total） |
| `AVG() OVER(...)` | 移动平均（moving average） |
| `NTILE(n)` | 将行划分为 n 个桶 |
| `FIRST_VALUE()` | 窗口帧内第一个值 |
| `LAST_VALUE()` | 窗口帧内最后一个值 |

## 综合实战

以下是一个融合 CTE、JOIN 与窗口函数的真实查询：基于电商 Schema，找出每个品类营收最高的商品，并计算其占该品类总营收的百分比：

```sql
WITH product_revenue AS (
    SELECT
        p.product_id,
        p.name AS product_name,
        p.category,
        SUM(oi.quantity * oi.unit_price) AS revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.status = 'completed'
    GROUP BY p.product_id, p.name, p.category
),
ranked AS (
    SELECT
        *,
        RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rank_in_category,
        SUM(revenue) OVER (PARTITION BY category) AS category_total
    FROM product_revenue
)
SELECT
    category,
    product_name,
    revenue,
    category_total,
    ROUND(revenue / category_total * 100, 1) AS pct_of_category
FROM ranked
WHERE rank_in_category = 1
ORDER BY revenue DESC;
```

输出：

```
  category    |    product_name     | revenue  | category_total | pct_of_category
--------------+---------------------+----------+----------------+-----------------
 Electronics  | Mechanical Keyboard |  8934.50 |      18432.50  |            48.5
 Furniture    | Standing Desk       |  7190.00 |      12890.00  |            55.8
 Books        | DDIA                |  1890.30 |       4567.80  |            41.4
```

若不用 SQL，此单一查询需数十行应用代码，或多次数据库往返才能实现。

## 为何表结构至今仍占主导地位（暂且）

关系模型之所以胜出，在于它提供了其他模型当时所不具备的关键能力：**数据独立性（Data Independence）**。你可以随意更改物理存储方式、添加索引、分表、复制数据——所有这些操作，均无需修改一行应用代码。SQL 接口保持不变。

它并非完美。某些数据（社交图谱、时序数据、深度嵌套的文档）难以优雅地映射到表结构中。第 5 篇文章将探讨这些替代方案。但对于大多数应用——尤其是数据一致性至关重要的场景——关系模型仍是默认且合理的选择。

## 下一步

掌握 SQL 是必要条件，但远非充分条件。“写出正确的查询”与“写出高效的查询”是两种不同技能。下一篇我们将聚焦 **索引与查询规划（Indexing and Query Planning）**——数据库如何实际定位你的数据，以及如何让它更快地找到。