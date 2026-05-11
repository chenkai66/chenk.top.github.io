---
title: "系统设计（7）：数据管道——批处理、流处理与Lambda架构"
date: 2025-07-24 09:00:00
tags:
  - System Design
  - Data Engineering
  - Stream Processing
  - Apache Kafka
categories:
  - System Design
series: system-design
lang: zh
description: "一份实用的数据管道架构指南——涵盖ETL vs ELT、Spark批处理、Flink流处理、Lambda vs Kappa架构、变更数据捕获（CDC），以及端到端实时分析管道设计。"
disableNunjucks: true
series_order: 7
translationKey: "system-design-7"
---

每秒，一家大型电商平台都会生成数千个数据点：页面浏览、搜索查询、加入购物车事件、下单、库存变更、价格更新、配送状态变化。这些原始数据在未经处理时毫无价值——它们散落在数十个服务中，以不同格式存储，且到达速率不可预测。而将这类原始数据转化为可操作洞察的系统——实时仪表盘、个性化推荐、欺诈检测告警、业务报表——正是**数据管道（Data Pipeline）**。

数据管道并不光鲜。它不直接面向用户。但它却是每个数据驱动型组织的“神经系统”，其设计优劣，直接决定了企业是基于“昨天的数据”做决策，还是基于“30秒前的数据”做决策。

## ETL vs ELT

数据管道设计的两种基础范式，核心差异在于**转换（Transformation）发生的时机**。

### ETL：Extract, Transform, Load（抽取、转换、加载）

传统范式。数据从源系统抽取，在暂存区完成转换，再加载至目标系统（通常是数据仓库）。

```
源系统 → 抽取 → 转换（暂存区） → 加载 → 数据仓库

示例：
  MySQL（订单） ─┐
  PostgreSQL（用户） ─┤→ 转换（清洗、关联、聚合） → 加载 → Snowflake
  MongoDB（日志） ─┘
```

转换发生在加载之前。这意味着：
- 只有经过清洗和校验的数据才能进入数据仓库；
- 数据仓库的Schema受控且可预测；
- 支持复杂转换（如多表JOIN、聚合、去重）；
- 修改转换逻辑需重新运行整个管道。

### ELT：Extract, Load, Transform（抽取、加载、转换）

现代范式。数据从源系统抽取后，以原始格式直接加载至数据湖或云数据仓库；转换则在目标系统内部，利用其计算能力完成。

```
源系统 → 抽取 → 加载（原始） → 转换（在仓库内） → 数仓分层/视图

示例：
  MySQL（订单） ─┐
  PostgreSQL（用户） ─┤→ 原始加载 → BigQuery → dbt模型 → 清洗后的表
  MongoDB（日志） ─┘
```

转换发生在加载之后。这意味着：
- 原始数据被完整保留（可随时用新逻辑重转换，无需重新抽取）；
- 云数仓提供廉价、可扩展的计算资源用于转换；
- Schema-on-read：原始数据无需预定义Schema；
- 数据入库更快；但产出干净数据更慢。

### 如何选择？

| 因素 | ETL | ELT |
|--------|-----|-----|
| 数据量 | 中等 | 大到海量 |
| 转换复杂度 | 复杂、多步骤 | 可用SQL表达 |
| 数据质量要求 | 高（前置校验） | 灵活（原始层 + 校验层） |
| 基础设施 | 本地部署或自建 | 云数据仓库 |
| Schema稳定性 | 稳定、预先定义 | 演进式、Schema-on-read |
| 延迟要求 | 批处理（小时级/天级） | 批处理或近实时 |
| 成本模型 | 计算密集型暂存区 | 存储廉价、按需计算 |

## 批处理（Batch Processing）

批处理以固定周期（如每小时、每天或每周）处理大量数据。数据被收集、存储后，作为完整数据集进行处理。

### MapReduce（概念模型）

MapReduce由Google于2004年提出，是分布式批处理的基础模型。尽管已被更高层框架广泛取代，但理解其思想仍至关重要。

该模型分为两个阶段：

**Map（映射）**：独立处理每个输入记录，输出键值对（key-value pairs）。

**Reduce（归约）**：按Key对所有Value分组并聚合。

```python
# 概念化MapReduce：统计各URL的页面浏览次数

# Map阶段（在多台机器上并行执行）
def map_function(log_line):
    """解析日志行，为每次页面浏览输出 (url, 1)"""
    url = parse_url(log_line)
    emit(url, 1)

# Shuffle阶段（框架自动按键分组）
# "/products/123" → [1, 1, 1, 1, 1]
# "/products/456" → [1, 1, 1]

# Reduce阶段（对每个Key的Values求和）
def reduce_function(url, counts):
    """对每个URL的所有计数求和"""
    emit(url, sum(counts))

# 输出：
# "/products/123" → 5
# "/products/456" → 3
```

MapReduce的局限在于：多步转换需串联多个MapReduce作业，每个作业都需从磁盘读取输入、写入中间结果。这种阶段间频繁的磁盘I/O使复杂管道性能低下。

### Apache Spark

Spark已取代MapReduce成为主流批处理框架。其关键创新在于**内存计算**：不再将中间结果写入磁盘，而是跨转换步骤保留在内存中，使迭代类算法提速10–100倍。

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("DailyOrderAnalytics") \
    .config("spark.sql.shuffle.partitions", 200) \
    .getOrCreate()

# 从数据湖读取原始订单数据
orders = spark.read.parquet("s3://data-lake/raw/orders/date=2025-07-24/")
users = spark.read.parquet("s3://data-lake/raw/users/")
products = spark.read.parquet("s3://data-lake/raw/products/")

# 转换：关联订单、用户与商品信息
enriched_orders = orders \
    .join(users, orders.user_id == users.id, "left") \
    .join(products, orders.product_id == products.id, "left") \
    .select(
        orders.order_id,
        orders.created_at,
        users.country,
        users.age_group,
        products.category,
        products.name.alias("product_name"),
        orders.quantity,
        orders.total_price,
    )

# 聚合：按国家与商品类目统计日销售额
daily_sales = enriched_orders \
    .groupBy("country", "category") \
    .agg(
        F.count("order_id").alias("order_count"),
        F.sum("total_price").alias("total_revenue"),
        F.avg("total_price").alias("avg_order_value"),
        F.countDistinct("user_id").alias("unique_buyers"),
    ) \
    .orderBy(F.desc("total_revenue"))

# 写入数据湖（分国家分区）
daily_sales.write \
    .mode("overwrite") \
    .partitionBy("country") \
    .parquet("s3://data-lake/curated/daily_sales/date=2025-07-24/")

# 同时写入报表数据库
daily_sales.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://reporting-db:5432/analytics") \
    .option("dbtable", "daily_sales") \
    .option("user", "etl_user") \
    .mode("append") \
    .save()
```

### 批处理特性

- **高吞吐量**：面向全量数据集处理；优化目标是容量，而非延迟；
- **完整数据**：可重处理历史数据；迟到数据可在下一批次中补入；
- **简单容错**：若任务失败，直接重跑即可；
- **高延迟**：结果仅在批次完成后才可用（数小时或数天）；
- **资源可预测**：在预定时间运行，资源可提前规划。

## 流处理（Stream Processing）

流处理持续处理实时到达的数据，产出结果的延迟通常在亚秒级至分钟级。

### 核心概念

**事件（Event）**：带时间戳的单个数据点。例如：一次页面浏览、一笔交易、一个传感器读数。

**流（Stream）**：无界、持续到达的事件序列。

**窗口（Windowing）**：将事件分组为有限集合以便聚合。没有窗口机制，无法在无限流上计算“每分钟计数”等指标。

### 窗口类型

**滚动窗口（Tumbling Window）**：固定大小、互不重叠的时间区间。每个事件恰好属于一个窗口。

```
时间：    |----1分钟----|----1分钟----|----1分钟----|
事件：    [e1, e2, e3] [e4, e5]    [e6, e7, e8, e9]
窗口：    [  窗口 1   ] [ 窗口 2  ] [   窗口 3    ]
```

**滑动窗口（Sliding Window）**：固定大小的窗口，以固定步长推进。窗口之间存在重叠。

```
时间：    |----1分钟-----------|
         |     |----1分钟----|------|
事件：    [e1, e2, e3, e4, e5, e6, e7]
窗口：    [  窗口 1（0–60秒）      ]
              [  窗口 2（30–90秒）     ]
                   [  窗口 3（60–120秒）    ]
```

**会话窗口（Session Window）**：基于活动间隙动态生成的窗口。当指定间隔内无新事件到达时，当前窗口关闭。

```
事件：    e1..e2..e3........e4..e5..e6............e7..e8
         |--- 会话 1 ---|--- 会话 2 ------|--会话 3--|
         （间隙 < 阈值）   （间隙 < 阈值）     （间隙 > 阈值）
```

### 水位线（Watermarks）

在分布式系统中，事件可能乱序到达：时间戳为 T=100 的事件，可能晚于 T=105 的事件抵达。水位线用于追踪事件时间（event time）的进展，并告知系统何时可安全关闭某个窗口。

```
水位线 W=100 表示：“所有时间戳 ≤ 100 的事件均已到达。”

若当前水位线为 W=100：
  - 窗口 [0, 60] 已完成（所有事件均已到达）
  - 窗口 [60, 120] 仍可能收到新事件
  
迟到事件（时间戳 < 水位线）可被：
  - 丢弃（最简单）
  - 发送到旁路输出（side output）供后续修正
  - 用于更新先前结果（retractions）
```

## Apache Flink

Flink是目前领先的开源流处理框架，提供**精确一次（exactly-once）** 处理保证、事件时间（event time）处理及高级窗口支持。

### 关键概念

**DataStream API**：流处理的核心抽象。DataStream代表一个事件流，可通过各类算子（map、filter、keyBy、window、aggregate）进行变换。

**事件时间 vs 处理时间**：事件时间指事件实际发生的时间（嵌入在数据中）；处理时间指系统处理该事件的时间。为保障结果正确性，应优先使用事件时间——因为处理延迟不会影响最终结果。

**检查点（Checkpointing）**：Flink定期将所有算子的状态快照持久化至可靠存储。发生故障时，系统从最新检查点恢复，并从源头（如Kafka offset）重放事件。这实现了精确一次语义。

### Flink管道示例（概念化Python代码）

```python
# 概念化Flink电商实时分析管道
# 使用PyFlink Table API

from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble

# 初始化环境
env_settings = EnvironmentSettings.in_streaming_mode()
t_env = TableEnvironment.create(env_settings)

# 定义Kafka源表
t_env.execute_sql("""
    CREATE TABLE page_views (
        user_id STRING,
        page_url STRING,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'page-views',
        'properties.bootstrap.servers' = 'kafka:9092',
        'properties.group.id' = 'analytics-pipeline',
        'format' = 'json',
        'scan.startup.mode' = 'latest-offset'
    )
""")

# 定义输出目标表
t_env.execute_sql("""
    CREATE TABLE page_view_counts (
        window_start TIMESTAMP(3),
        window_end TIMESTAMP(3),
        page_url STRING,
        view_count BIGINT,
        unique_users BIGINT
    ) WITH (
        'connector' = 'jdbc',
        'url' = 'jdbc:postgresql://analytics-db:5432/metrics',
        'table-name' = 'page_view_counts',
        'driver' = 'org.postgresql.Driver',
        'username' = 'flink_user',
        'password' = '...'
    )
""")

# 滚动窗口聚合：统计每分钟各URL的浏览量与独立用户数
t_env.execute_sql("""
    INSERT INTO page_view_counts
    SELECT
        window_start,
        window_end,
        page_url,
        COUNT(*) AS view_count,
        COUNT(DISTINCT user_id) AS unique_users
    FROM TABLE(
        TUMBLE(TABLE page_views, DESCRIPTOR(event_time), INTERVAL '1' MINUTE)
    )
    GROUP BY window_start, window_end, page_url
""")
```

## Lambda架构

Lambda架构由Nathan Marz提出，融合批处理与流处理，兼顾**历史数据的准确性**与**新数据的低延迟性**。

### 三层结构

**批处理层（Batch Layer）**：周期性（如每小时）处理全量数据，产出准确、全面的结果，存入批处理视图（Batch View）。

**速度层（Speed Layer）**：实时处理自上次批处理以来的新数据，产出近似、低延迟的结果，存入实时视图（Real-time View）。

**服务层（Serving Layer）**：合并批处理视图与实时视图以响应查询。查询时，历史数据查批处理视图，近期数据查实时视图。

```
                    ┌──────────────┐
全部数据 ──────────→│  批处理层    │ → 批处理视图 ─────┐
     │              │  （Spark）   │                     │
     │              └──────────────┘                     ▼
     │                                              ┌──────────┐
     │                                              │ 服务层    │ → 查询请求
     │                                              │           │
     │              ┌──────────────┐                └──────────┘
     └─────────────→│  速度层      │ → 实时视图 ─┘
                    │  （Flink）   │
                    └──────────────┘
```

### Lambda示例：页面浏览计数器

```
批处理层（每小时运行）：
  - 从数据湖读取全部页面浏览事件
  - 统计各URL的历史总浏览量
  - 写入 batch_view 表：{url, total_views, last_updated}

速度层（持续运行）：
  - 从Kafka读取自上次批处理以来的浏览事件
  - 实时统计各URL的浏览量（最近一小时）
  - 写入 realtime_view 表：{url, recent_views, last_updated}

服务层（查询时）：
  SELECT
    batch.url,
    batch.total_views + COALESCE(realtime.recent_views, 0) AS total_views
  FROM batch_view batch
  LEFT JOIN realtime_view realtime ON batch.url = realtime.url
```

### Lambda的缺陷

主要问题：需维护两套代码（批处理 + 流处理），且二者必须产出一致结果。任一端出现Bug都将导致数据偏差。每次业务逻辑变更都需双份实现。

## Kappa架构

Kappa架构由Kafka联合创始人Jay Kreps提出，通过**仅使用流处理**来简化Lambda。其核心洞见是：若流处理器能回溯重放历史数据（例如从Kafka起始offset重读），则无需单独的批处理层。

### 工作原理

```
全部数据 → Kafka（保留数月/数年） → 流处理器 → 服务层

历史数据重处理流程：
  1. 部署新版流处理器
  2. 从Kafka Topic起始位置（offset 0）开始消费
  3. 将全部历史事件经由新逻辑处理
  4. 切换服务层至新输出
  5. 下线旧版处理器
```

### Lambda vs Kappa对比

| 因素 | Lambda | Kappa |
|--------|--------|-------|
| 代码库数量 | 两个（批处理 + 流处理） | 一个（仅流处理） |
| 复杂度 | 更高（需维护两套系统） | 更低（一套系统） |
| 准确性 | 批处理层始终准确 | 取决于流处理器逻辑正确性 |
| 重处理 | 自然（重跑批处理作业） | 从Kafka起点重读 |
| 迟到数据处理 | 批处理在下次运行中自动修正 | 取决于水位线/窗口策略 |
| 存储成本 | 数据湖 + Kafka | Kafka（需长期保留） |
| 延迟 | 速度层提供实时性 | 全链路实时 |
| 成熟度 | 已验证的成熟模式 | 较新，采用率持续上升 |
| 最佳适用场景 | 对历史数据准确性要求极高，且批处理逻辑过于复杂（如复杂ML训练、图算法） | 事件驱动系统，逻辑可表达为流操作，Kafka保留期满足重处理需求，且希望避免双代码库 |

**选用Lambda当**：你需要对历史数据提供强准确性保证，且批处理逻辑过于复杂，难以在流式环境中高效实现（例如复杂机器学习模型训练、图算法）。

**选用Kappa当**：你的处理逻辑可完全用流操作表达，Kafka保留期足以覆盖重处理所需历史范围，且你希望规避维护两套代码的成本。

## 数据湖 vs 数据仓库

| 特性 | 数据湖（Data Lake） | 数据仓库（Data Warehouse） |
|---------|---------------------|----------------------------|
| Schema | Schema-on-read（原始数据） | Schema-on-write（结构化） |
| 数据格式 | 任意（JSON、Parquet、Avro、CSV、图像等） | 结构化表格 |
| 处理引擎 | Spark、Flink、Presto | SQL（BigQuery、Snowflake、Redshift） |
| 主要用户 | 数据工程师、数据科学家 | 业务分析师、BI工具 |
| 成本 | 存储廉价（S3、GCS） | 计算昂贵（按查询计费） |
| 治理难度 | 较难（非结构化数据） | 较易（Schema明确） |
| 典型用例 | 机器学习训练、原始数据探索 | 业务报表、仪表盘 |
| 示例 | S3 + Spark、Delta Lake、Apache Iceberg | Snowflake、BigQuery、Redshift |

现代趋势是**湖仓一体（Lakehouse）**——融合数据湖的低成本与Schema灵活性，以及数据仓库的ACID事务、SQL查询与Schema强制能力。Delta Lake、Apache Iceberg、Apache Hudi等技术正推动这一范式落地。

## 数据质量（Data Quality）

### Schema校验

在处理前，对输入数据进行Schema校验：

```python
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

class OrderEvent(BaseModel):
    order_id: str
    user_id: str
    product_id: str
    quantity: int
    total_price: float
    currency: str
    timestamp: datetime

    @validator("quantity")
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @validator("currency")
    def currency_must_be_valid(cls, v):
        valid = {"USD", "EUR", "GBP", "JPY", "CNY"}
        if v not in valid:
            raise ValueError(f"currency must be one of {valid}")
        return v

def validate_event(raw_event: dict) -> Optional[OrderEvent]:
    try:
        return OrderEvent(**raw_event)
    except Exception as e:
        # 发送至死信主题（DLQ）供人工排查
        send_to_dlq(raw_event, str(e))
        return None
```

### 数据血缘（Data Lineage）

追踪数据来源及转换过程。这对调试、合规审计与影响分析至关重要。

```yaml
# 示例血缘元数据（随处理后数据一同存储）
lineage:
  source:
    system: "orders-service"
    topic: "orders"
    partition: 3
    offset: 145892
    timestamp: "2025-07-24T10:30:00Z"
  transformations:
    - step: "schema_validation"
      version: "1.2.0"
      timestamp: "2025-07-24T10:30:01Z"
    - step: "currency_conversion"
      version: "2.0.1"
      rates_source: "ecb_daily_2025-07-24"
      timestamp: "2025-07-24T10:30:01Z"
    - step: "user_enrichment"
      version: "1.0.0"
      source: "users-service-api"
      timestamp: "2025-07-24T10:30:02Z"
  destination:
    table: "enriched_orders"
    partition: "date=2025-07-24"
```

## 变更数据捕获（Change Data Capture, CDC）

CDC从数据库事务日志中捕获行级变更（INSERT/UPDATE/DELETE），并以事件形式流式输出。这使得实时数据同步无需轮询。

### Debezium

Debezium是最广泛使用的开源CDC平台。它读取数据库的WAL（Write-Ahead Log）或binlog，并将变更事件发布至Kafka。

```yaml
# PostgreSQL的Debezium连接器配置
{
  "name": "orders-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "orders-db",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "...",
    "database.dbname": "orders",
    "database.server.name": "orders",
    "table.include.list": "public.orders,public.order_items",
    "plugin.name": "pgoutput",
    "publication.name": "dbz_publication",
    "slot.name": "debezium_slot",
    "topic.prefix": "cdc",
    "transforms": "route",
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "cdc\\.public\\.(.*)",
    "transforms.route.replacement": "cdc.$1"
  }
}
```

一条CDC变更事件如下所示：

```json
{
  "before": null,
  "after": {
    "order_id": "ord_789",
    "user_id": "user_123",
    "total": 59.98,
    "status": "created",
    "created_at": 1721822400000
  },
  "source": {
    "version": "2.5.0",
    "connector": "postgresql",
    "name": "orders",
    "ts_ms": 1721822400123,
    "db": "orders",
    "schema": "public",
    "table": "orders",
    "txId": 12345,
    "lsn": 98765432
  },
  "op": "c",
  "ts_ms": 1721822400200
}
```

`op` 字段表示操作类型：`c`（创建/插入）、`u`（更新）、`d`（删除）、`r`（读取/快照）。

### CDC典型用例

- **实时分析**：将数据库变更直接流式接入分析管道，避免查询数据库；
- **搜索索引同步**：保持Elasticsearch与源数据库实时一致；
- **缓存失效**：数据库行变更时，自动失效Redis对应缓存条目；
- **跨服务数据同步**：在服务间复制数据，无需调用API；
- **审计日志**：记录每一次数据变更，满足合规要求。

## 管道中的幂等处理（Idempotent Processing）

在分布式管道中，重复事件不可避免（至少一次投递、重试、重处理）。每个处理环节都必须能优雅地应对重复。

```python
class IdempotentProcessor:
    """通过去重存储确保每个事件仅被处理一次"""

    def __init__(self, redis_client, ttl=86400):
        self.redis = redis_client
        self.ttl = ttl  # 记录已处理事件的过期时间（秒）

    def process(self, event: dict) -> bool:
        event_id = event["event_id"]
        dedup_key = f"processed:{event_id}"

        # 原子性“检查并设置”
        if self.redis.set(dedup_key, "1", ex=self.ttl, nx=True):
            # 首次见到该事件 —— 执行处理
            return True
        else:
            # 重复事件 —— 跳过
            return False

# 在管道中使用
processor = IdempotentProcessor(redis_client)

for event in stream:
    if processor.process(event):
        transform_and_store(event)
    else:
        metrics.increment("duplicate_events_skipped")
```

## 实战案例：电商实时分析系统

### 需求

- 实时跟踪页面浏览、加入购物车、下单事件；
- 展示实时仪表盘，按分钟粒度展示浏览量、转化率、营收；
- 支持按商品类目、国家、设备类型下钻分析；
- 数据从事件发生到可查，延迟 ≤ 30 秒；
- 峰值吞吐量：50,000 事件/秒。

### 架构

```
Web/Mobile App → 事件采集器（HTTP API） → Kafka → Flink → PostgreSQL → Grafana仪表盘
                                           ↓
                                      S3数据湖（原始事件，用于批处理重处理）
```

组件说明：

1. **事件采集器**：轻量级HTTP API，负责事件校验并发布至Kafka。作为无状态服务，部署在负载均衡器后。

2. **Kafka**：三个Topic——`page-views`、`add-to-cart`、`purchases`。均按`user_id`分区以保证顺序。保留期：30天（支持重处理）。

3. **Flink**：三个流处理作业：
   - 按URL、国家、设备类型，每分钟聚合页面浏览量；
   - 每分钟计算转化漏斗（浏览 → 加购 → 下单）；
   - 每分钟按商品类目与国家统计营收。

4. **PostgreSQL**：存储聚合指标（非原始事件）。启用TimescaleDB扩展以优化时序数据。

5. **S3数据湖**：原始事件以Parquet格式存储，按日期与事件类型分区。用于即席分析与批处理重处理。

6. **Grafana**：仪表盘直接查询PostgreSQL获取实时指标。

### 容量估算

```
峰值：50,000 事件/秒  
平均事件大小：500 字节  

Kafka吞吐量：50,000 × 500 = 25 MB/秒  
日数据量：25 MB/秒 × 86,400 = 2.16 TB/天  
月存储量（Kafka，30天保留）：64.8 TB  
S3存储量（原始，压缩后）：约15 TB/月（Parquet压缩率≈70%）

Flink输出：约1,000条聚合记录/分钟  
PostgreSQL存储：可忽略（聚合数据量极小）
```

该架构以合理基础设施即可满足全部需求：3节点Kafka集群、2–4个Flink TaskManager、单节点PostgreSQL（仅存聚合指标）。

## 下一篇预告

至此，我们已覆盖系统设计全部核心模块：容量估算、网络、API、缓存、消息队列、微服务与数据管道。最后一篇文章将整合所有要素，呈现三个完整案例研究：短链接服务、实时聊天系统、新闻信息流。每个案例均从需求出发，逐步推演至可扩展架构设计。