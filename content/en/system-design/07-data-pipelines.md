---
title: "System Design (7): Data Pipelines — Batch, Stream, and the Lambda Architecture"
date: 2025-07-24 09:00:00
tags:
  - System Design
  - Data Engineering
  - Stream Processing
  - Apache Kafka
categories: System Design
series: system-design
lang: en
description: "A practical guide to data pipeline architectures — covering ETL vs ELT, batch processing with Spark, stream processing with Flink, Lambda vs Kappa architectures, Change Data Capture, and a complete real-time analytics pipeline design."
disableNunjucks: true
series_order: 7
translationKey: "system-design-7"
---

Every second, a large e-commerce platform generates thousands of data points: page views, search queries, add-to-cart events, purchases, inventory changes, price updates, and delivery status changes. This raw data is useless in its original form — scattered across dozens of services, stored in different formats, and arriving at unpredictable rates. The system that transforms this raw data into actionable insights — real-time dashboards, personalized recommendations, fraud detection alerts, business reports — is the data pipeline.

Data pipelines are not glamorous. They do not face users directly. But they are the nervous system of every data-driven organization, and designing them well is the difference between decisions made on yesterday's data and decisions made on data from 30 seconds ago.

---

## ETL vs ELT

The two fundamental approaches to data pipeline design differ in when transformation happens.

![ETL vs ELT comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/07-etl-vs-elt.png)


### ETL: Extract, Transform, Load

The traditional approach. Data is extracted from source systems, transformed in a staging area, and loaded into the destination (typically a data warehouse).

```yaml
Source Systems → Extract → Transform (staging) → Load → Data Warehouse

Example:
  MySQL (orders) ─┐
  PostgreSQL (users) ─┤→ Transform (clean, join, aggregate) → Load → Snowflake
  MongoDB (logs) ─┘
```

The transformation happens before loading. This means:
- Only clean, validated data enters the warehouse
- The warehouse schema is controlled and predictable
- Transformations can be complex (joins, aggregations, deduplication)
- Changes to transformation logic require re-running the pipeline

### ELT: Extract, Load, Transform

The modern approach. Data is extracted from sources and loaded raw into a data lake or cloud warehouse. Transformation happens inside the destination using its compute power.

```yaml
Source Systems → Extract → Load (raw) → Transform (in warehouse) → Mart/Views

Example:
  MySQL (orders) ─┐
  PostgreSQL (users) ─┤→ Load raw → BigQuery → dbt models → Clean tables
  MongoDB (logs) ─┘
```

Transformation happens after loading. This means:
- Raw data is preserved (you can re-transform without re-extracting)
- Cloud warehouses provide cheap, scalable compute for transformations
- Schema-on-read: the raw data does not need a predefined schema
- Faster to get data in; slower to get clean data out

### When to Use Each

| Factor | ETL | ELT |
|--------|-----|-----|
| Data volume | Moderate | Large to massive |
| Transformation complexity | Complex, multi-step | SQL-expressible |
| Data quality requirements | High (pre-validated) | Flexible (raw + validated layers) |
| Infrastructure | On-premise or custom | Cloud data warehouse |
| Schema stability | Stable, pre-defined | Evolving, schema-on-read |
| Latency requirements | Batch (hourly/daily) | Batch or near-real-time |
| Cost model | Compute-heavy staging | Storage-cheap, compute-on-demand |

## Batch Processing

Batch processing handles large volumes of data in scheduled intervals — hourly, daily, or weekly. The data is collected, stored, and processed as a complete set.

![Batch vs stream processing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/07-batch-vs-stream.png)


### MapReduce (Conceptual)

MapReduce, introduced by Google in 2004, is the foundational model for distributed batch processing. While it has been largely superseded by higher-level frameworks, understanding the concept is essential.

The model has two phases:

**Map**: Process each input record independently, emitting key-value pairs.

**Reduce**: Group all values by key, aggregate them.

```python
# Conceptual MapReduce: count page views per URL

# Map phase (runs on many machines in parallel)
def map_function(log_line):
    """Parse log line, emit (url, 1) for each page view."""
    url = parse_url(log_line)
    emit(url, 1)

# Shuffle phase (framework groups by key)
# "/products/123" → [1, 1, 1, 1, 1]
# "/products/456" → [1, 1, 1]

# Reduce phase (aggregates values per key)
def reduce_function(url, counts):
    """Sum all counts for each URL."""
    emit(url, sum(counts))

# Output:
# "/products/123" → 5
# "/products/456" → 3
```

MapReduce's limitation is that multi-step transformations require chaining multiple MapReduce jobs, each reading from and writing to disk. This disk I/O between stages makes complex pipelines slow.

### Apache Spark

Spark replaced MapReduce for most batch processing workloads. Its key innovation: in-memory computation. Instead of writing intermediate results to disk, Spark keeps data in memory across transformation steps, making iterative algorithms 10-100x faster.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("DailyOrderAnalytics") \
    .config("spark.sql.shuffle.partitions", 200) \
    .getOrCreate()

# Read raw order data from data lake
orders = spark.read.parquet("s3://data-lake/raw/orders/date=2025-07-24/")
users = spark.read.parquet("s3://data-lake/raw/users/")
products = spark.read.parquet("s3://data-lake/raw/products/")

# Transform: join orders with users and products
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

# Aggregate: daily sales by country and category
daily_sales = enriched_orders \
    .groupBy("country", "category") \
    .agg(
        F.count("order_id").alias("order_count"),
        F.sum("total_price").alias("total_revenue"),
        F.avg("total_price").alias("avg_order_value"),
        F.countDistinct("user_id").alias("unique_buyers"),
    ) \
    .orderBy(F.desc("total_revenue"))

# Write to warehouse
daily_sales.write \
    .mode("overwrite") \
    .partitionBy("country") \
    .parquet("s3://data-lake/curated/daily_sales/date=2025-07-24/")

# Also write to a reporting database
daily_sales.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://reporting-db:5432/analytics") \
    .option("dbtable", "daily_sales") \
    .option("user", "etl_user") \
    .mode("append") \
    .save()
```

### Batch Processing Characteristics

- **High throughput**: Processes entire datasets; optimized for volume, not latency
- **Complete data**: Can re-process historical data, handle late arrivals in the next batch
- **Simple error handling**: If a job fails, re-run it
- **High latency**: Results are available only after the batch completes (hours or days)
- **Predictable resource usage**: Runs at scheduled times, resources can be provisioned accordingly

## Stream Processing


![Data pipeline river system streams flowing through processin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/07-data-pipeline-river-system-streams-flowing-through-processin.jpg)

Stream processing handles data continuously as it arrives, producing results with sub-second to minute-level latency.

### Core Concepts

**Event**: A single data point with a timestamp. Examples: a page view, a transaction, a sensor reading.

**Stream**: An unbounded, continuously-arriving sequence of events.

**Windowing**: Grouping events into finite sets for aggregation. Without windowing, you cannot compute aggregates like "count per minute" on an infinite stream.

### Window Types

**Tumbling window**: Fixed-size, non-overlapping time intervals. Every event belongs to exactly one window.

```text
Time:    |----1min----|----1min----|----1min----|
Events:  [e1, e2, e3] [e4, e5]    [e6, e7, e8, e9]
Windows: [  Window 1 ] [ Window 2 ] [  Window 3   ]
```

**Sliding window**: Fixed-size windows that advance by a fixed step. Windows overlap.

```text
Time:    |----1min-----------|
         |     |----1min----|------|
Events:  [e1, e2, e3, e4, e5, e6, e7]
Windows: [  Window 1 (0-60s)      ]
              [  Window 2 (30-90s)     ]
                   [  Window 3 (60-120s)    ]
```

**Session window**: Dynamic windows based on activity gaps. A window closes when no events arrive for a specified gap duration.

```text
Events:  e1..e2..e3........e4..e5..e6............e7..e8
         |--- session 1 ---|--- session 2 ------|--session 3--|
         (gap < threshold)  (gap < threshold)    (gap > threshold)
```

### Watermarks

In a distributed system, events can arrive out of order. An event with timestamp T=100 might arrive after an event with timestamp T=105. Watermarks track the progress of event time and tell the system when it is safe to close a window.

```text
A watermark of W=100 means: "All events with timestamp <= 100 have arrived."

If the current watermark is W=100:
  - Window [0, 60] is complete (all events arrived)
  - Window [60, 120] might still receive events
  
Late events (timestamp < watermark) can be:
  - Dropped (simplest)
  - Processed in a side output (for correction)
  - Used to update previous results (retractions)
```

## Apache Flink

Flink is the leading open-source stream processing framework. It provides exactly-once processing guarantees, event time processing, and sophisticated windowing.

![Apache Flink architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/07-flink-architecture.png)


### Key Concepts

**DataStream API**: The core abstraction for stream processing. A DataStream represents a stream of events that can be transformed through operators (map, filter, keyBy, window, aggregate).

**Event Time vs Processing Time**: Event time is when the event actually occurred (embedded in the data). Processing time is when the system processes the event. Event time is preferred for correctness because processing delays do not affect results.

**Checkpointing**: Flink periodically snapshots the state of all operators to durable storage. On failure, the system restores from the latest checkpoint and replays events from the source (e.g., Kafka offsets). This provides exactly-once processing semantics.

### Flink Pipeline Example (Conceptual Python)

```python
# Conceptual Flink pipeline for real-time e-commerce analytics
# Using PyFlink's Table API

from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble

# Initialize environment
env_settings = EnvironmentSettings.in_streaming_mode()
t_env = TableEnvironment.create(env_settings)

# Define Kafka source
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

# Define output sink
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

# Tumbling window aggregation: count page views per URL per minute
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

## Lambda Architecture

The Lambda architecture, proposed by Nathan Marz, combines batch and stream processing to provide both accurate historical results and low-latency approximate results.

### Three Layers

**Batch layer**: Processes the complete dataset periodically (e.g., every hour). Produces accurate, comprehensive results. Stored in a batch view.

**Speed layer**: Processes new data in real-time since the last batch run. Produces approximate, low-latency results. Stored in a real-time view.

**Serving layer**: Merges batch and real-time views to answer queries. Queries check the batch view for historical data and the speed view for recent data.

```text
                    ┌──────────────┐
All Data ──────────→│  Batch Layer │ → Batch Views ─────┐
     │              │  (Spark)     │                     │
     │              └──────────────┘                     ▼
     │                                              ┌──────────┐
     │                                              │ Serving   │ → Queries
     │                                              │ Layer     │
     │              ┌──────────────┐                └──────────┘
     └─────────────→│  Speed Layer │ → Real-time Views ─┘
                    │  (Flink)     │
                    └──────────────┘
```

### Lambda Example: Page View Counter

```text
Batch layer (runs every hour):
  - Read all page view events from data lake
  - Count page views per URL for all time
  - Write to batch_view table: {url, total_views, last_updated}

Speed layer (runs continuously):
  - Read page view events from Kafka (since last batch run)
  - Count page views per URL in real-time
  - Write to realtime_view table: {url, recent_views, last_updated}

Serving layer (query time):
  SELECT
    batch.url,
    batch.total_views + COALESCE(realtime.recent_views, 0) AS total_views
  FROM batch_view batch
  LEFT JOIN realtime_view realtime ON batch.url = realtime.url
```

### Lambda Drawbacks

The main problem: you maintain two codebases (batch and stream) that must produce consistent results. Bugs in one but not the other cause discrepancies. Every business logic change must be implemented twice.

## Kappa Architecture

The Kappa architecture, proposed by Jay Kreps (co-creator of Kafka), simplifies Lambda by using only stream processing. The key insight: if your stream processor can replay historical data (by re-reading from Kafka), you do not need a separate batch layer.

### How It Works

```text
All Data → Kafka (retained for months/years) → Stream Processor → Serving Layer

For historical reprocessing:
  1. Deploy new version of stream processor
  2. Start from beginning of Kafka topic (offset 0)
  3. Process all historical events through the new logic
  4. Switch serving layer to the new output
  5. Shut down old processor version
```

### Lambda vs Kappa Comparison

| Factor | Lambda | Kappa |
|--------|--------|-------|
| Codebases | Two (batch + stream) | One (stream only) |
| Complexity | Higher (two systems to maintain) | Lower (one system) |
| Accuracy | Batch is always accurate | Depends on stream processor correctness |
| Reprocessing | Natural (just re-run batch job) | Re-read Kafka from beginning |
| Late data handling | Batch corrects in next run | Depends on watermark/window strategy |
| Storage cost | Data lake + Kafka | Kafka (with long retention) |
| Latency | Real-time from speed layer | Real-time |
| Maturity | Well-established pattern | Newer, growing adoption |
| Best for | Complex analytics with strict accuracy | Event-driven systems with simpler logic |

**Use Lambda when**: You need guaranteed accuracy for historical data and your batch logic is too complex for stream processing (e.g., complex ML model training, graph algorithms).

**Use Kappa when**: Your processing logic can be expressed as stream operations, Kafka retention covers your reprocessing needs, and you want to avoid maintaining two codebases.

## Data Lake vs Data Warehouse

| Feature | Data Lake | Data Warehouse |
|---------|-----------|----------------|
| Schema | Schema-on-read (raw data) | Schema-on-write (structured) |
| Data format | Any (JSON, Parquet, Avro, CSV, images) | Structured tables |
| Processing | Spark, Flink, Presto | SQL (BigQuery, Snowflake, Redshift) |
| Users | Data engineers, data scientists | Business analysts, BI tools |
| Cost | Cheap storage (S3, GCS) | Expensive compute (per-query pricing) |
| Governance | Harder (unstructured data) | Easier (defined schemas) |
| Use cases | ML training, raw data exploration | Business reporting, dashboards |
| Examples | S3 + Spark, Delta Lake, Apache Iceberg | Snowflake, BigQuery, Redshift |

The modern trend is the **Lakehouse** — combining data lake storage (cheap, schema-flexible) with data warehouse features (ACID transactions, SQL queries, schema enforcement). Technologies like Delta Lake, Apache Iceberg, and Apache Hudi enable this pattern.

## Data Quality


![Data quality dimensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/07-data-quality.png)


![Etl vs elt two factory layouts transform first vs load first](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/07-etl-vs-elt-two-factory-layouts-transform-first-vs-load-first.jpg)

### Schema Validation

Validate incoming data against an expected schema before processing:

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
        # Route to dead letter topic for investigation
        send_to_dlq(raw_event, str(e))
        return None
```

### Data Lineage

Track where data came from and how it was transformed. This is critical for debugging, compliance, and impact analysis.

```yaml
# Example lineage metadata (stored alongside processed data)
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

## Change Data Capture (CDC)

CDC captures row-level changes (INSERT, UPDATE, DELETE) from a database's transaction log and streams them as events. This enables real-time data synchronization without polling.

![Change data capture pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/07-cdc.png)


### Debezium

Debezium is the most widely-used open-source CDC platform. It reads the database's write-ahead log (WAL/binlog) and publishes change events to Kafka.

```yaml
# Debezium connector configuration for PostgreSQL
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

A CDC change event looks like:

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

The `op` field indicates the operation type: `c` (create/insert), `u` (update), `d` (delete), `r` (read/snapshot).

### CDC Use Cases

- **Real-time analytics**: Stream database changes to an analytics pipeline without querying the database
- **Search index sync**: Keep Elasticsearch in sync with the source database
- **Cache invalidation**: Invalidate Redis cache entries when the underlying database row changes
- **Cross-service data sync**: Replicate data between services without API calls
- **Audit logging**: Capture every data change for compliance

## Idempotent Processing in Pipelines

Duplicate events are inevitable in distributed pipelines (at-least-once delivery, retries, reprocessing). Every pipeline stage must handle duplicates gracefully.

```python
class IdempotentProcessor:
    """Ensures each event is processed exactly once using a deduplication store."""

    def __init__(self, redis_client, ttl=86400):
        self.redis = redis_client
        self.ttl = ttl  # How long to remember processed events

    def process(self, event: dict) -> bool:
        event_id = event["event_id"]
        dedup_key = f"processed:{event_id}"

        # Atomic check-and-set
        if self.redis.set(dedup_key, "1", ex=self.ttl, nx=True):
            # First time seeing this event — process it
            return True
        else:
            # Duplicate — skip
            return False

# Usage in pipeline
processor = IdempotentProcessor(redis_client)

for event in stream:
    if processor.process(event):
        transform_and_store(event)
    else:
        metrics.increment("duplicate_events_skipped")
```

## Real Example: Real-Time Analytics for E-Commerce

### Requirements

- Track page views, add-to-cart, and purchase events in real-time
- Display a live dashboard showing metrics per minute (views, conversions, revenue)
- Support drilling down by product category, country, and device type
- Data must be available within 30 seconds of the event
- Handle 50,000 events per second at peak

### Architecture

```text
Web/Mobile App → Event Collector (API) → Kafka → Flink → PostgreSQL → Grafana Dashboard
                                           ↓
                                      S3 Data Lake (raw events for batch reprocessing)
```

Components:

1. **Event Collector**: A lightweight HTTP API that validates events and publishes to Kafka. Runs as a stateless service behind a load balancer.

2. **Kafka**: Three topics — `page-views`, `add-to-cart`, `purchases`. Each partitioned by user ID for ordering. Retention: 30 days for reprocessing.

3. **Flink**: Three stream processing jobs:
   - Per-minute aggregation of page views by URL, country, and device
   - Per-minute conversion funnel (view → cart → purchase)
   - Per-minute revenue by product category and country

4. **PostgreSQL**: Stores aggregated metrics (not raw events). TimescaleDB extension for time-series optimization.

5. **S3 Data Lake**: Raw events stored in Parquet format, partitioned by date and event type. Used for ad-hoc analysis and batch reprocessing.

6. **Grafana**: Dashboard querying PostgreSQL for real-time metrics.

### Estimation

```text
Peak: 50,000 events/sec
Average event size: 500 bytes

Kafka throughput: 50,000 × 500 = 25 MB/sec
Daily volume: 25 MB/sec × 86,400 = 2.16 TB/day
Monthly storage (Kafka, 30-day retention): 64.8 TB
S3 storage (raw, compressed): ~15 TB/month (Parquet ≈ 70% compression)

Flink output: ~1,000 aggregated rows per minute
PostgreSQL storage: negligible (aggregated data is small)
```

This architecture handles the requirements with reasonable infrastructure: a 3-broker Kafka cluster, 2-4 Flink task managers, and a single PostgreSQL instance for aggregated metrics.

## What's Next

With all the building blocks in place — estimation, networking, APIs, caching, message queues, microservices, and data pipelines — the final article puts them together. Three complete case studies: a URL shortener, a real-time chat system, and a news feed. Each walks through the full design process from requirements to scaling strategies.
