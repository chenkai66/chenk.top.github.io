---
title: "云存储系统与分布式架构"
date: 2023-03-30 09:00:00
tags:
  - 云计算
  - 分布式存储
  - 对象存储
  - HDFS
  - Ceph
categories: 云计算
series:
  name: "云计算"
  part: 3
  total: 8
lang: zh-CN
mathjax: false
description: "从 CAP 定理到 S3、HDFS、Ceph —— 深入剖析分布式存储原语、一致性、副本、纠删码与成本优化。"
disableNunjucks: true
series_order: 4
---

Netflix 存储 PB 级视频，Instagram 提供数十亿张照片，量化基金几分钟内回放一年的行情数据 —— 它们背后都是 *分布式存储系统*。从开发者的视角看，存储简单得近乎透明（`PUT key`、`GET key`），但只要跨过单机的边界，你就接管了一整摞折磨了学术界几十年的难题：如何在磁盘失效时不丢数据、如何线性扩展、如何提供一个不会让上层应用踩坑的一致性模型，还要把每 GB 的成本压到几分钱。

本文打通整条堆栈：*理论地基*（CAP、一致性、一致性哈希），*三种存储形态*（对象、块、文件），*生产系统*（S3、HDFS、Ceph），以及把裸容量变成 SLA 的 *工程杠杆*（副本、纠删码、生命周期、分片上传）。

## 你将学到

1. **取舍空间** —— CAP、PACELC，以及为什么分区容错是底线
2. **云存储的三种形态** —— 块、文件、对象：什么场景选什么原语
3. **对象存储的内部** —— 分区、放置、持久性、S3 的请求路径
4. **分布式文件系统** —— HDFS（主从）vs Ceph（CRUSH 对等）
5. **副本 vs 纠删码** —— 11 个 9 的持久性背后的数学
6. **运营层设计** —— 一致性模型、仲裁、分片上传、生命周期策略
7. **成本工程** —— 存储等级、压缩、去重、分层策略

## 前置知识

- 熟悉 Python 与 Unix shell
- 对 TCP、HTTP 与文件系统有基本心智模型
- 建议先读本系列前 2 篇（基础与虚拟化）

---

## 1. 问题的形状

### 1.1 分布式存储为什么难

一块 SSD 能跑出几十万 IOPS、用上好几年。问题在于，生产环境里 *单个* 任何东西都是负债：

- 一块磁盘几年坏一次 —— 一万块的机群里，每天都要换。
- 单机有容量上限，受限于机柜槽位和电源预算。
- 单条网络路径是单点，一台 ToR 重启就把整柜打黑。

所以我们做副本。一旦做副本就引出两个更深的问题：让副本保持一致（一致性），以及在网络把副本切成孤岛时仍能服务（分区）。这正是 CAP 定理刻画的疆域。

### 1.2 对象 vs 块 vs 文件：选对原语

在谈一致性理论之前，先把词汇定下来。云存储一共三种 *形态*，几乎所有云产品都是它们的某种伪装：

![对象 vs 块 vs 文件 多维对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/storage-systems/fig1_storage_comparison.png)

| 形态 | 访问单元 | API | 延迟 | 规模 | 典型产品 | 最佳场景 |
|------|---------|-----|------|------|---------|---------|
| **块** | 512 B / 4 KB 的块 | iSCSI、NVMe、virtio | µs - 低 ms | TB / 卷 | EBS、Azure Disk、ESSD | 数据库、虚机系统盘、需要 `O_DIRECT` 的负载 |
| **文件** | 目录树中的字节流 | NFS / SMB / POSIX | 低 ms | PB / 文件系统 | EFS、Azure Files、NAS | 共享暂存、传统应用搬迁 |
| **对象** | 带元数据的不可变 Blob | HTTP/REST | 几十 ms | EB | S3、GCS、OSS | Web 资源、备份、数据湖、ML 数据集 |

一句心法：**块是扇区，文件是路径，对象是 URL**。每往上一层，都是用延迟换规模。块给你一个虚拟磁盘（内核再在上面盖文件系统）；文件直接给你共享文件系统；对象彻底放弃 POSIX 语义，换来近乎无限的规模和扁平 Key 空间。

不知道选哪个时：负载是「打开文件、seek、写入」就用块或文件；是「按 Key 上传、下载、删除」就用对象。

---

## 2. CAP、PACELC 与一致性菜单

### 2.1 CAP 定理

Brewer 2000 年提出的 CAP 猜想，2002 年被 Gilbert 和 Lynch 证明：当出现网络分区时，分布式系统必须在 **C**（一致性）与 **A**（可用性）之间二选一。两者不可兼得。

![CAP 定理 韦恩图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/storage-systems/fig3_cap_theorem.png)

- **一致性 C** —— 任何读总是返回最新的提交（线性一致）。
- **可用性 A** —— 每个未失效节点都在有限时间内返回响应。
- **分区容错 P** —— 节点之间任意消息丢失时，系统仍能运行。

真实网络里分区不可选 —— 交换机重启、光纤被挖、内核长 GC —— 所以 P 不是选项。真正要选的是 **CP vs AP**。"CA" 描述的是不承认分区会发生的单机系统。

| 系统 | 类别 | 分区时的行为 |
|------|------|-------------|
| ZooKeeper、etcd、HBase、MongoDB（多数写） | CP | 少数派一侧拒绝写；集群停摆但永不分歧 |
| Cassandra、DynamoDB、Riak、S3（旧版覆盖写） | AP | 两侧都接受写；事后再合并 |
| 单机 Postgres | 类 CA | 一旦分区直接失能（"分区" = "网络挂了"） |

### 2.2 PACELC：CAP 漏掉的那一半

CAP 只描述 *分区时* 的行为。PACELC（Abadi, 2010）补全了它：**若分区，选 A 或 C；否则，选延迟 L 或一致性 C**。

这就解释了为什么 Cassandra 是 "AP/EL"（正常情况下最终一致的读更快），而 Spanner 是 "CP/EC"（每次提交都要等 TrueTime 不确定窗口以保证严格一致）。生产中绝大多数决策其实是 PACELC 决策而不是 CAP 决策；分区罕见，但延迟 vs 一致性的旋钮每次请求都在拨。

### 2.3 实用一致性模型

| 模型 | 保证 | 出现位置 |
|------|------|---------|
| **线性一致** | 所有操作存在一个与真实时间一致的全局顺序 | etcd、Spanner、ZooKeeper |
| **顺序一致** | 存在全局顺序，但不必匹配真实时间 | 经典共享内存 |
| **读己之写** | 客户端总能读到自己写的最新值 | S3（2020-12 起）、大多数会话型系统 |
| **单调读** | 一旦读到 v，就不会再读到更旧的值 | 带粘性会话的缓存 |
| **最终一致** | 停止写后读最终收敛 | Cassandra 默认、DynamoDB 最终一致读 |

S3 在 2020 年 12 月把所有操作切到 **强读己之写一致** 是个非常震撼的例子：通过重建元数据层（内部叫 *witness* 服务），AWS 把一个看似根本性的取舍硬给压平了 —— 既保留 AP 的数据路径，又给到线性一致的语义。

---

## 3. 一致性哈希：路由原语

按 `hash(key) % N` 分片在加减节点之前都好用，一旦 `(N -> N+1)` 几乎所有 Key 都要重排。**一致性哈希**（Karger 等, 1997）把扰动降到约 `1/N`。它是 DynamoDB、Cassandra、Riak、memcached 客户端的放置算法，Ceph 的 CRUSH 也是它的变体。

技巧：把 Key *和* 节点都映射到同一个圆环上；每个 Key 归顺时针下一个节点所有。新加节点只从它顺时针的邻居那里偷一份 Key。引入 **虚拟节点**（同一物理节点哈希多次）就能把负载摊得很均匀。

```python
import bisect
import hashlib


class ConsistentHash:
    """一个最小可用的、带虚拟节点的一致性哈希环。"""

    def __init__(self, nodes=None, vnodes=128):
        self.vnodes = vnodes
        self.ring = {}              # hash -> 节点
        self.sorted_keys = []       # 排序后的 hash 列表
        for node in nodes or []:
            self.add_node(node)

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        for i in range(self.vnodes):
            h = self._hash(f"{node}#{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node: str) -> None:
        for i in range(self.vnodes):
            h = self._hash(f"{node}#{i}")
            self.sorted_keys.remove(h)
            del self.ring[h]

    def get_node(self, key: str) -> str | None:
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


ring = ConsistentHash(["s3-az-a", "s3-az-b", "s3-az-c"])
print(ring.get_node("user:12345/profile.jpg"))
```

**为什么用 128 个虚拟节点？** 每个物理节点 1 个虚拟节点时，负载标准差大约 30%；换成 128 个，方差降到几个百分点 —— 接近均匀分布，可以按平均值规划容量了。

---

## 4. 对象存储：S3 究竟是怎么工作的

S3 是云对象存储的事实标准，OSS、GCS、R2、Backblaze B2 等几十个兼容系统都以它为蓝本。从一万米高空看，S3 是「一个把 `(bucket, key)` 映射到字节的 HTTP 服务器」。从内部看，它是有史以来最大的分布式系统之一。

![S3 类对象存储请求路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/storage-systems/fig2_s3_architecture.png)

### 4.1 请求路径

一次 `PUT s3://my-bucket/users/42.jpg` 的全程：

1. **DNS + 边缘** —— 客户端解析到地域端点，落到边缘前端。
2. **鉴权** —— 校验 SigV4 签名；按 Bucket ACL / IAM 策略 / SCP / VPC 端点策略 / Object ACL 逐层授权。
3. **索引查找** —— Bucket 名称映射到内部一个键范围索引的 *分区*。热 Bucket 会自动分裂（一旦 QPS 或大小超过阈值；这是历史上「Key 前缀打散」建议的来源，现在已基本过时）。
4. **放置** —— Placement 服务按目标持久性方案选一组 OSD（例如跨 3 个 AZ 的 6+3 EC）。
5. **写入** —— 字节流式写到足以满足写仲裁的节点。
6. **应答** —— 一旦持久化，S3 返回 `200 OK` 与 ETag。

### 4.2 存储等级 —— 成本杠杆

| 等级 | $/GB-月（US-East-1） | 取回延迟 | 最短存储 | 取回费 |
|------|---------------------|---------|---------|--------|
| Standard | ~$0.023 | ms | 无 | 无 |
| Intelligent-Tiering | 高频 $0.023，IA $0.0125 | ms | 30 d（自动） | 无 |
| Standard-IA | $0.0125 | ms | 30 d | $0.01 / GB |
| One Zone-IA | $0.01 | ms | 30 d | $0.01 / GB |
| Glacier Instant | $0.004 | ms | 90 d | $0.03 / GB |
| Glacier Flexible | $0.0036 | 1 分钟 - 12 小时 | 90 d | $0.01 / GB（标准） |
| Glacier Deep Archive | $0.00099 | 12 - 48 小时 | 180 d | $0.02 / GB |

两个实战要点：

- **冷层有最短保留期。** Deep Archive 存 30 天就删，仍按 180 天计费。只把 *真不动* 的数据沉到冷层。
- **取回费可能远超存储费。** 一个标 IA 的数据被意外频繁访问，比放 Standard 还贵。Intelligent-Tiering 的存在就是为了挡住这种坑。

### 4.3 生产级 Python SDK

```python
import boto3
from botocore.config import Config

# 调优过的客户端：积极超时、签名 v4、POST 不重试
s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "adaptive"},
        connect_timeout=3,
        read_timeout=60,
        max_pool_connections=64,
    ),
)

# 服务端 KMS 加密、请求完整性校验、Intelligent-Tiering
s3.put_object(
    Bucket="my-app-data",
    Key="users/42/profile.jpg",
    Body=open("/local/profile.jpg", "rb"),
    ContentType="image/jpeg",
    StorageClass="INTELLIGENT_TIERING",
    ServerSideEncryption="aws:kms",
    SSEKMSKeyId="alias/app-data",
    Metadata={"user-id": "42", "uploader": "ios-7.2"},
    ChecksumAlgorithm="SHA256",     # 端到端完整性
)
```

三个团队常踩的坑：

- **`ChecksumAlgorithm`** 让客户端算校验、S3 验校验；不设的话只能依赖 TCP/TLS，不是端到端。
- **`StorageClass`** 在 PUT 时设几乎免费；事后改类要 `CopyObject`，可能产生取回费。
- **自适应重试** 在收到 503 SlowDown 时退避；没有它，热 Prefix 能把整个机群打趴。

### 4.4 分片上传

几百 MB 以上的对象都应分片上传：可并行、可断点续传、可分片校验。`boto3.s3.transfer` 已经封装好；下面给出显式版本，看清里面发生了什么：

```python
from concurrent.futures import ThreadPoolExecutor
import os

def multipart_upload(path: str, bucket: str, key: str,
                     part_mb: int = 16, workers: int = 16) -> None:
    size = os.path.getsize(path)
    part_size = part_mb * 1024 * 1024
    num_parts = (size + part_size - 1) // part_size

    init = s3.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id = init["UploadId"]

    def _upload_part(part_no: int) -> dict:
        with open(path, "rb") as f:
            f.seek((part_no - 1) * part_size)
            data = f.read(part_size)
        resp = s3.upload_part(
            Bucket=bucket, Key=key, PartNumber=part_no,
            UploadId=upload_id, Body=data,
            ChecksumAlgorithm="SHA256",
        )
        return {"PartNumber": part_no, "ETag": resp["ETag"],
                "ChecksumSHA256": resp["ChecksumSHA256"]}

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_upload_part, range(1, num_parts + 1)))
        s3.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise
```

这里的 `try/except` 不是装饰：失败但没清理的分片上传会悄悄把孤片计入存储账单。给每个 Bucket 都加上 `AbortIncompleteMultipartUpload after 7 days` 的生命周期规则。

---

## 5. 副本 vs 纠删码

3 副本是简单答案，纠删码是便宜答案。现代云对象存储基本两个都用：热数据走副本，冷数据走 EC。

![副本 vs 纠删码 开销与持久性](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/storage-systems/fig4_erasure_vs_replication.png)

### 5.1 数学

**k+m** 的 Reed-Solomon EC 把对象切成 `k` 个数据片 + `m` 个校验片；任意 `m` 片丢失仍可恢复，存储开销 `m/k`。

| 方案 | 开销 | 容忍故障 | 重建时读放大 |
|------|------|---------|-------------|
| 3 副本 | 200% | 2 块磁盘 | 1（再读一块就行） |
| EC 6+3 | 50% | 3 片 | 6（重建要读 6 片） |
| EC 10+4 | 40% | 4 片 | 10 |
| EC 12+4 | 33% | 4 片 | 12 |

取舍直白：

- 副本：CPU 便宜，磁盘贵，重建快（拷一块就好）。
- EC：磁盘便宜，重建时 CPU + 网络贵，重建慢。

热数据要副本（一次读一次 HTTP 拉取）。冷数据要 EC（很少摸到，重建成本能在年的尺度上摊掉）。

### 5.2 11 个 9 是个预算，不是奇迹

S3 宣传 99.999999999%（11 个 9）的对象持久性 —— 也就是每 1000 亿对象 · 年期望丢一个。这个数字不是魔法，是工程出来的：

- 磁盘年故障率约 1%。
- `k+m` 个分片至少跨 3 个 AZ。
- 后台持续扫描（scrubbing）在位翻转累积成下一次磁盘故障之前发现它。
- 修复带宽节流但永不停。

实战教训：**持久性的本质不是「我有 3 份」，而是「修复速度比下一次故障到来更快」**。如果你要自建存储，先把 repair pipeline 的指标盘做出来。

---

## 6. 分布式文件系统：HDFS 与 Ceph

当你确实需要跨多机的类 POSIX 语义时，两个参考设计占绝对主导：HDFS（主协调、追加为主）与 Ceph（对等、全功能）。

![HDFS vs Ceph 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/storage-systems/fig5_distributed_fs.png)

### 6.1 HDFS —— 主协调、批处理友好

HDFS 建立在三个判断之上：

1. **文件大、写一次、读多次**（日志、parquet、视频）。默认块大小 **128 MB** 正是为此。
2. **失败是常态**，副本下沉到块级，永不到文件级。
3. **把计算搬到数据旁边**（MapReduce 立身之本的数据本地性原则）。

架构：

- **NameNode** —— 命名空间和块位置的唯一真理来源。Active NN 与 **Standby NN** 通过 **JournalNode** 仲裁（QJM）共享 edits 实现 HA。
- **DataNode** —— 存块。默认副本因子 3。
- **块放置** —— 第一份在写入者所在节点（写入者在集群外则随机），第二份在不同 *机架*，第三份在第二份的同机架不同节点。能扛全机架故障，又保留机架内读速度。

```xml
<!-- hdfs-site.xml：能上生产的最小配置 -->
<configuration>
  <property><name>dfs.replication</name>             <value>3</value></property>
  <property><name>dfs.blocksize</name>               <value>134217728</value></property> <!-- 128 MB -->
  <property><name>dfs.namenode.handler.count</name>  <value>100</value></property>
  <property><name>dfs.datanode.handler.count</name>  <value>40</value></property>
  <property><name>dfs.namenode.name.dir</name>       <value>/data/nn1,/data/nn2</value></property>
  <property><name>dfs.datanode.data.dir</name>       <value>/data/dn1,/data/dn2,/data/dn3</value></property>
  <property><name>dfs.namenode.shared.edits.dir</name>
            <value>qjournal://jn1:8485;jn2:8485;jn3:8485/cluster1</value></property>
</configuration>
```

```python
# 现代 HDFS 客户端（PyArrow 的 HDFS 接口；hdfs3 已无人维护）
import pyarrow.fs as pafs

hdfs = pafs.HadoopFileSystem(host="namenode", port=9000, user="hadoop")
hdfs.create_dir("/user/data", recursive=True)
with hdfs.open_output_stream("/user/data/events.parquet") as f:
    f.write(open("/local/events.parquet", "rb").read())

for info in hdfs.get_file_info(pafs.FileSelector("/user/data", recursive=False)):
    print(f"{info.path:50s}  {info.size:>12} bytes")
```

**HDFS 不适合**：海量小文件（每个吃 NameNode 约 150 字节内存）、随机写（它只能追加）、低延迟点查。除此之外，在 Hadoop / Spark 生态里它仍然非常好用。

### 6.2 Ceph —— 一个集群、三种接口

Ceph 的贡献是 *统一存储*：单一 RADOS 集群暴露块（RBD）、对象（RGW）、文件（CephFS）三种接口，全部由同一组 OSD 承载。

让 Ceph 跟 HDFS 不一样的两个想法：

- **没有放置元数据的中心节点**。放置由 **CRUSH**（Controlled Replication Under Scalable Hashing）算出 —— 一个伪随机算法，输入对象名 + 集群图，输出确定的 OSD 集。从根上消除 NameNode 瓶颈。
- **OSD 自管理**。每个 OSD 知道自己的对端，副本、扫描、恢复都自己跑，没有中心调度器。

```bash
# 现代 Ceph 部署用 cephadm + 容器
cephadm bootstrap --mon-ip 10.0.1.10
ceph orch host add node2 10.0.1.11
ceph orch host add node3 10.0.1.12
ceph orch apply osd --all-available-devices

# 用 EC 4+2 建对象池（50% 开销，可容忍 2 个 OSD 故障）
ceph osd erasure-code-profile set ec42 k=4 m=2 \
    crush-failure-domain=host
ceph osd pool create cold_objects erasure ec42
```

```bash
# RBD：精简配置、可快照的虚拟磁盘
rbd create --size 100G --pool rbd vm-disk-01
rbd map rbd/vm-disk-01            # 出现 /dev/rbd0
mkfs.xfs /dev/rbd0
rbd snap create rbd/vm-disk-01@before-upgrade
```

```python
# RGW：基于同一集群的 S3 兼容端点
import boto3
from botocore.client import Config

rgw = boto3.client(
    "s3",
    endpoint_url="http://rgw.internal:7480",
    aws_access_key_id="<access>",
    aws_secret_access_key="<secret>",
    config=Config(signature_version="s3v4"),
)
rgw.create_bucket(Bucket="ml-datasets")
rgw.put_object(Bucket="ml-datasets", Key="cifar10.tar", Body=b"...")
```

Ceph 的灵活性是有运维代价的：调 CRUSH 规则、平衡 PG、规划 OSD 容量、监控重建带宽，规模上来后都是全职工作。多数团队应该先用云上托管的对象/块存储，只有出现硬约束（数据主权、本地化部署、定制硬件）时再考虑 Ceph。

---

## 7. 副本策略与运维

### 7.1 同步 vs 异步副本

| 维度 | 同步 | 异步 |
|------|------|------|
| 一致性 | 强（所有副本 ack 后才返回） | 最终一致 |
| 延迟 | 受最慢副本的 RTT 限制 | 仅本地磁盘 |
| RPO | 0 | 秒 - 分钟 |
| 吞吐 | 受最慢链路限 | 受主节点磁盘限 |
| 适用 | 金融账本、单地域数据库 HA | 跨地域 DR、分析副本 |

常见组合：**地域内同步**（跨 AZ，亚毫秒）+ **跨地域异步**（DR 副本，几秒 RPO）。

### 7.2 仲裁

`N` 副本，读/写仲裁分别为 `R` 与 `W`：

- `W + R > N` 保证读必与最新写有交集。
- `W = N` 写最强（任一节点失败即不可写）；`W = 1` 写最弱。
- Dynamo 系把 `R / W` 让 *应用* 按请求选；ZooKeeper / etcd 直接写死多数派。

### 7.3 生命周期策略 —— 把成本工程自动化

成本优化大多是几行策略，而不是工程改造。把日志 30 天降到 IA、90 天降到 Glacier、365 天删；过期未完成的分片上传；过期对象旧版本：

```python
s3.put_bucket_lifecycle_configuration(
    Bucket="prod-data",
    LifecycleConfiguration={
        "Rules": [
            {
                "ID": "AbortStuckUploads",
                "Status": "Enabled",
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
            },
            {
                "ID": "TierDataset",
                "Status": "Enabled",
                "Filter": {"Prefix": "datasets/"},
                "Transitions": [
                    {"Days": 30,  "StorageClass": "STANDARD_IA"},
                    {"Days": 90,  "StorageClass": "GLACIER_IR"},
                    {"Days": 365, "StorageClass": "DEEP_ARCHIVE"},
                ],
            },
            {
                "ID": "ExpireOldLogs",
                "Status": "Enabled",
                "Filter": {"Prefix": "logs/"},
                "Expiration": {"Days": 365},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
            },
        ]
    },
)
```

### 7.4 备份、RTO 与 RPO

| 策略 | RTO | RPO | 成本 | 适用 |
|------|-----|-----|------|------|
| 热备（多地域双活） | 秒 | 接近 0 | 2-3x | 支付、社交 Feed |
| 温备（副本闲置，故障时拉起） | 分钟 | 秒 | 1.2-1.5x | 多数生产 Web |
| 火种（数据有副本，基础设施缺位） | 小时 | 分钟 | 1.05-1.1x | 内部应用 |
| 仅备份 | 小时 - 天 | 小时 | 1.01x | 开发/分析 |

正确答案很少是最严格的那个，而是 *成本与每分钟宕机损失匹配* 的那个。

---

## 8. 性能优化

### 8.1 长尾延迟才是主导

要看的是 **P99 延迟**，不是平均值。一个扇出 10 后端、等所有返回的请求，整体 P99 大约就是单后端 P99（Dean & Barroso，*The Tail at Scale*, 2013）。每一个依赖都是一个倍率。

### 8.2 四个杠杆

- **并行** —— 分片上传、Range GET、并发 Prefix。
- **本地性** —— 计算与存储就近（HDFS 数据本地性、S3 跨地域里的 AZ 亲和）。
- **缓存** —— 应用层 LRU、CloudFront 挡在公开 S3 前面、OSS CDN 挡在热 Prefix 前面。
- **压缩** —— zstd 是现代默认（level 3 ≈ gzip-6 速度 + gzip-9 压缩比）；CPU 紧张时用 Snappy。

### 8.3 一条吞吐拇指法则

长距链路上的单条 TCP 流被 `窗口 / RTT` 限制。100 ms RTT、64 KB 默认窗口的天花板大约 5 Mbit/s。要把一根 10 Gbit/s 的管子打满，要么把窗口放大（内核自动调优能帮你），要么开多条流。分片上传就是「多条流」的显式形态 —— 这才是它快的 *原因*，而不仅仅是 *现象*。

---

## 9. 成本工程

### 9.1 三条账单线

| 账单项 | 驱动因素 | 杠杆 |
|--------|---------|------|
| 存储 | GB-月 | 分层、压缩、去重、过期 |
| 请求 | PUT / GET / LIST 计数 | 批量、分片、Prefix 设计 |
| 出网 | 跨地域 / 出云的 GB | CDN、地域内计算、专线 |

对多数团队来说，**出网是账单上的"惊喜"**。1 PB 跨地域复制按 $0.02/GB 是 $20,000。能在地域内复制就在地域内复制；非要出去，就走更便宜的传输通道（Direct Connect、专属对接、阿里云的 CEN）。

### 9.2 一个具体的成本计算

1 TB 数据集，每天访问 2 次，存 1 年：

- **Standard：** 1 TB × $0.023 × 12 = **$276/年**，请求成本可忽略。
- **Standard + 30 天后转 IA：** 存储约 $160/年 + IA 取回约 $10 ≈ **$170/年**。
- **90 天后转 Deep Archive，每年 2 次再训练取回：** 存储约 $15/年 + 取回费约 $40 ≈ **$55/年**。

精确数字会随官方定价漂移，但 *量级* 不会变 —— **生命周期是 5-10 倍的成本杠杆，且开通它本身免费**。

---

## 10. 常见问题

**问：新数据湖选 HDFS 还是 S3？**

绿地几乎一律选 S3（或对应物）。现代引擎（Trino、Spark、Flink、DuckDB）原生读对象存储；存算分离让两者独立扩缩；省去维护 NameNode HA 集群的麻烦。只有当本地化部署或极端的数据本地性是硬约束时，才回头选 HDFS。

**问：Ceph 在什么情况下值得它的运维代价？**

三种情况：监管要求数据留在自有硬件上；块 + 对象量大到云上托管价更高；已有 OpenStack 环境受益于 Ceph 的紧密集成。除此之外，把 SRE 工时算上之后，云上托管基本更便宜。

**问：3 副本还是纠删码？**

热的、低延迟敏感的数据用副本；冷数据和大对象用 EC。许多系统会自动两个都做（S3 内部、Ceph 通过分层池）。这个决定基本不需要应用层操心。

**问：对象存储前面的缓存该怎么定容？**

把对象访问频次按对象数量画出来，找"拐点"。典型 Web 负载缓存前 1-5% 的对象就能吸收 80-95% 的 GET。再往后是边际收益递减。CloudFront / OSS CDN 把它从容量决策变成了配置决策。

---

## 系列导航

| 篇 | 主题 |
|----|------|
| 1 | [基础与架构体系](/zh/cloud-computing-fundamentals/) |
| 2 | [虚拟化技术深度解析](/zh/cloud-computing-virtualization/) |
| **3** | **存储系统与分布式架构（当前）** |
| 4 | [网络架构与 SDN](/zh/cloud-computing-networking-sdn/) |
| 5 | [安全与隐私保护](/zh/cloud-computing-security-privacy/) |
| 6 | [运维与 DevOps 实践](/zh/cloud-computing-operations-devops/) |
| 7 | [云原生与容器技术](/zh/cloud-computing-cloud-native-containers/) |
| 8 | [多云与混合架构](/zh/cloud-computing-multi-cloud-hybrid/) |
