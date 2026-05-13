---
title: "云计算（四）：云存储系统与分布式架构"
date: 2023-03-30 09:00:00
tags:
  - Cloud Computing
  - Distributed Storage
  - Object Storage
  - HDFS
  - Ceph
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "从 CAP 定理到 S3、HDFS、Ceph —— 深入剖析分布式存储原语、一致性、副本、纠删码与成本优化。"
disableNunjucks: true
series_order: 4
translationKey: "cloud-computing-4"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/illustration_1.png)

Netflix 存储 PB 级视频，Instagram 提供数十亿张照片，量化基金几分钟内回放一年的行情数据——这些都依赖于 *分布式存储系统*。从开发者的视角看，存储简单得近乎透明（`PUT key`、`GET key`），但只要跨过单机的边界，你就接管了一整摞折磨了学术界几十年的难题：如何在磁盘故障时保障数据不丢失、如何实现线性扩展、如何提供上层应用不易误用的一致性模型，同时将每 GB 存储成本压缩至几分钱。

本文贯通整条技术栈：*理论基础*（CAP、一致性模型、一致性哈希）、*三类存储形态*（对象、块、文件）、*典型生产系统*（S3、HDFS、Ceph），以及将原始存储容量转化为可靠 SLA 的 *核心工程手段*（副本、纠删码、生命周期管理、分片上传）。

## 你将学到的内容

1. **权衡空间** —— CAP 和 PACELC 理论，以及为什么分区容错性是不可避免的  
2. **云存储的三种形式** —— 块存储、文件存储和对象存储：如何选择合适的底层原语  
3. **对象存储的核心机制** —— 数据分片、存储位置选择、持久性保障及 S3 请求路径的工作原理  
4. **分布式文件系统对比** —— HDFS（基于主节点）与 Ceph（基于 CRUSH 的对等架构）的设计差异  
5. **副本与纠删码的取舍** —— 如何通过数学实现 11 个 9 的持久性，同时只需 1.5 倍存储开销  
6. **运维设计要点** —— 一致性模型、仲裁机制、分段上传及生命周期管理策略的实际应用  
7. **成本优化实践** —— 存储分级、数据压缩、去重技术及冷热分层策略的综合运用
## 前置知识
- 熟悉 Python 与 Unix shell
- 对 TCP、HTTP 与文件系统有基本了解
- 建议先读本系列前 2 篇（基础与虚拟化）

---

## 1. 问题的本质

### 1.1 分布式存储的难点在哪里？
一块 SSD 通常能提供数十万 IOPS，使用寿命长达数年。但在生产环境中，依赖单点设备始终存在显著风险：单块硬盘的年故障率（AFR）通常在 0.5%–2% 之间——在一个拥有上万台服务器、数万块硬盘的集群中，平均每天都会发生硬盘故障；单台机器有容量瓶颈，无法突破机柜槽位或电源预算的限制；单条网络路径是单点故障（SPOF），一台 ToR 交换机重启，整台机器就失联了。

因此，必须引入数据副本机制。但副本一经启用，会立即引发两个根本性挑战：如何保障多副本间的数据一致性（Consistency），以及在网络分区导致副本集分裂时，如何在保障服务可用性（Partition Tolerance）的同时维持系统正确性——这正是 CAP 定理所描述的根本矛盾。

### 1.2 对象存储 vs 块存储 vs 文件存储：选择合适的底层模型

在深入讨论一致性理论之前，先明确几个关键概念。云存储本质上只有三种基本形态，市面上几乎所有存储产品都可以归为这三类之一：

![对象 vs 块 vs 文件 多维对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/fig1_storage_comparison.png)

| 形态 | 访问粒度 | 接口 | 延迟 | 扩展性 | 典型产品 | 最佳适用场景 |
|------|---------|-----|------|--------|----------|--------------|
| **块存储** | 512 B / 4 KB 的块 | iSCSI、 NVMe、 virtio | 微秒级 - 毫秒级 | TB 级别 / 卷 | EBS、 Azure Disk、 ESSD | 数据库、虚拟机系统盘、需要 `O_DIRECT` 的工作负载 |
| **文件存储** | 目录树中的字节流 | NFS / SMB / POSIX | 毫秒级 | PB 级别 / 文件系统 | EFS、 Azure Files、 NAS | 共享临时存储、传统应用迁移 |
| **对象存储** | 带元数据的不可变 Blob | HTTP/REST | 数十毫秒 | EB 级别 | S3、 GCS、 OSS | Web 资源托管、备份、数据湖、机器学习数据集 |

一个简单的记忆法则：**块存储对应扇区，文件存储对应路径，对象存储对应 URL**。每上升一层，都是用更高的延迟换取更大的扩展性。块存储提供了一个虚拟磁盘（内核可以在其上构建文件系统）；文件存储直接提供共享文件系统；对象存储则彻底抛弃了 POSIX 语义，换来近乎无限的扩展能力和扁平化的键值空间。

如果不确定该选哪种存储：如果是「打开文件、定位、写入」这类操作，适合用块存储或文件存储；如果是「按 Key 上传、下载、删除」这类操作，则更适合用对象存储。
## 2. CAP、 PACELC 与一致性选择

### 2.1 CAP 定理

2000 年， Eric Brewer 提出了著名的 CAP 猜想，随后在 2002 年被 Gilbert 和 Lynch 证明。该定理指出：当网络分区发生时，分布式系统必须在 **一致性（Consistency）** 和 **可用性（Availability）** 之间做出权衡，无法同时满足两者。

![CAP 定理 韦恩图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/fig3_cap_theorem.png)

- **一致性（C）** —— 每次读操作都能返回最新的写入结果（线性一致性）。
- **可用性（A）** —— 即使部分节点失效，系统仍能在有限时间内响应请求。
- **分区容错性（P）** —— 即使节点之间的通信中断或消息丢失，系统依然能够正常运行。

在实际网络中，分区是不可避免的——比如交换机重启、光纤被挖断或者内核暂停等场景。因此，分区容错性（P）并不是可选项，而是必须面对的现实。真正需要决策的是 **CP 还是 AP**。而所谓的 "CA" 系统，实际上只是单机系统的一种理想化描述，它们假设分区不会发生。

| 系统 | 类型 | 分区时的行为 |
|------|------|-------------|
| ZooKeeper、 etcd、 HBase、 MongoDB （多数派写入） | CP | 少数派节点拒绝写入；集群暂停服务，但保证数据一致 |
| Cassandra、 DynamoDB、 Riak、 S3 （旧版覆盖写） | AP | 双方节点均接受写入；后续通过合并解决冲突 |
| 单机 Postgres | 类 CA | 分区发生时完全停止服务（分区被视为“网络不可用”） |

### 2.2 PACELC： CAP 的补充视角

CAP 定理仅描述了系统在分区情况下的行为，而 PACELC （Abadi, 2010）则进一步扩展了这一理论：**如果发生分区，选择 A （可用性）或 C （一致性）；否则，在无分区时，选择 L （低延迟）或 C （一致性）。**

这解释了为什么 Cassandra 被归类为 "AP/EL"（在正常情况下，最终一致的读取速度更快），而 Spanner 则是 "CP/EC"（每次提交都会付出 TrueTime 不确定性的代价以确保强一致性）。在实际生产环境中，大多数设计决策其实是基于 PACELC 而非 CAP，因为分区虽然罕见，但延迟与一致性之间的权衡却贯穿于每个请求之中。

### 2.3 常见的一致性模型

| 模型 | 保障 | 典型应用 |
|------|------|---------|
| **线性一致性** | 所有操作都按照真实时间顺序排列成一个全局序列 | etcd、 Spanner、 ZooKeeper |
| **顺序一致性** | 存在一个全局顺序，但不严格匹配真实时间 | 经典共享内存系统 |
| **读己之写** | 客户端总能读到自己最近写入的数据 | S3 （自 2020 年 12 月起）、大多数会话级系统 |
| **单调读** | 一旦客户端读取到某个值，就不会再看到更旧的值 | 带粘性会话的缓存系统 |
| **最终一致性** | 如果写入停止，所有读取最终会收敛到相同的结果 | Cassandra 默认配置、 DynamoDB 最终一致读 |

S3 在 2020 年 12 月全面支持 **强读己之写一致性** 是一个极具启发性的案例。 AWS 通过重构其元数据层（内部称为 *witness* 服务），成功地将看似不可调和的权衡点压平——既保留了 AP 数据路径的高可用性，又实现了线性一致性的语义。这种工程上的突破展示了如何在理论限制下找到创新的解决方案。
## 3. 一致性哈希：路由的核心机制

使用 `hash(key) % N` 进行分片的方法在节点数量固定时表现良好，但一旦增加或移除节点（例如从 `N` 变为 `N+1`），几乎所有的 Key 都需要重新分配。这种大规模的数据迁移显然不现实。**一致性哈希**（Karger 等， 1997）通过将数据迁移的比例降低到大约 `1/N`，有效解决了这一问题。它是 DynamoDB、 Cassandra、 Riak 和 memcached 客户端等系统的核心放置算法，而 Ceph 的 CRUSH 算法也可以看作是它的一种改进版本。

一致性哈希的巧妙之处在于：将 Key 和节点都映射到一个虚拟的环形空间中，每个 Key 归属于它顺时针方向的第一个节点。当新增一个节点时，它只会从其顺时针方向的下一个节点“接管”一部分 Key，而不会影响其他节点上的数据分布。为了进一步均衡负载，引入了 **虚拟节点** 的概念——每个物理节点会被映射多次到环上，从而显著平滑负载分布。

```python
import bisect
import hashlib

class ConsistentHash:
    """一个简单的一致性哈希实现，支持虚拟节点。"""

    def __init__(self, nodes=None, vnodes=128):
        self.vnodes = vnodes
        self.ring = {}              # hash -> 节点
        self.sorted_keys = []       # 排序后的哈希值列表
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

**为什么选择 128 个虚拟节点？** 如果每个物理节点只对应一个虚拟节点，负载的标准差可能高达 30%；而当虚拟节点数增加到 128 时，负载的标准差会降到几个百分点以内。此时的分布已经足够均匀，可以基于平均值进行容量规划，极大地简化了系统设计和运维工作。
## 4. 对象存储： S3 的工作原理

S3 是云对象存储领域的事实标准， OSS、 GCS、 R2、 Backblaze B2 等数十个兼容系统都借鉴了它的设计。从宏观上看， S3 就是一个「将 `(bucket, key)` 映射到字节流的 HTTP 服务器」。而从内部来看，它是有史以来规模最大的分布式系统之一。

![S3 类对象存储请求路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/fig2_s3_architecture.png)

### 4.1 请求处理流程

以 `PUT s3://my-bucket/users/42.jpg` 为例，整个请求的处理过程如下：

1. **DNS + 边缘节点** —— 客户端解析地域端点后，流量会落到边缘节点。
2. **鉴权** —— 验证 SigV4 签名，并依次评估 Bucket ACL、 IAM 策略、 SCP、 VPC 端点策略以及 Object ACL。
3. **索引查询** —— Bucket 名称映射到内部键范围索引的一个 *分区*。如果某个分区的 QPS 或大小超过阈值，热 Bucket 会被自动拆分（这也是早期「Key 前缀随机化」建议的由来，不过现在基本已经过时）。
4. **数据放置** —— Placement 服务根据目标持久性策略选择一组 OSD （例如跨 3 个可用区的 6+3 纠删码方案）。
5. **写入数据** —— 数据以流式传输到足够多的节点，以满足写仲裁要求。
6. **返回响应** —— 数据持久化完成后， S3 返回 `200 OK` 和 ETag。

### 4.2 存储类别 —— 成本控制的关键

| 类别 | $/GB-月（US-East-1） \vert  取回延迟 \vert  最短存储期限 \vert  取回费用 \vert 
|------|---------------------|---------|-------------|----------|
| 标准存储 (Standard) | ~$0.023 \vert  毫秒级 \vert  无 \vert  无 \vert 
| 智能分层 (Intelligent-Tiering) | 高频 $0.023，低频 $0.0125 | 毫秒级 | 30 天（自动） | 无 |
| 低频访问 (Standard-IA) | $0.0125 \vert  毫秒级 \vert  30 天 \vert  $0.01 / GB |
| 单区低频访问 (One Zone-IA) | $0.01 \vert  毫秒级 \vert  30 天 \vert  $0.01 / GB |
| Glacier 即时取回 | $0.004 \vert  毫秒级 \vert  90 天 \vert  $0.03 / GB |
| Glacier 灵活取回 | $0.0036 \vert  1 分钟 - 12 小时 \vert  90 天 \vert  $0.01 / GB （标准） |
| Glacier 深度归档 | $0.00099 \vert  12 - 48 小时 \vert  180 天 \vert  $0.02 / GB |

两个需要注意的实战细节：

- **冷存储有最低存储期限**。比如，一个 Deep Archive 文件存了 30 天就删除，仍然会按 180 天计费。只有那些 *真正不会再用* 的数据才适合放到冷存储。
- **取回费用可能远高于存储成本**。如果一个标记为 IA 的文件被频繁访问，其总成本可能比放在标准存储还高。智能分层（Intelligent-Tiering）就是为了避免这种问题而设计的。

### 4.3 生产环境中的 Python SDK

```python
import boto3
from botocore.config import Config

# 调优后的客户端：激进超时设置、签名 v4、POST 不重试
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

# 服务端 KMS 加密、请求完整性校验、智能分层
s3.put_object(
    Bucket="my-app-data",
    Key="users/42/profile.jpg",
    Body=open("/local/profile.jpg", "rb"),
    ContentType="image/jpeg",
    StorageClass="INTELLIGENT_TIERING",
    ServerSideEncryption="aws:kms",
    SSEKMSKeyId="alias/app-data",
    Metadata={"user-id": "42", "uploader": "ios-7.2"},
    ChecksumAlgorithm="SHA256",     # 端到端完整性校验
)
```

三个容易踩坑的地方：

- **`ChecksumAlgorithm`** 让客户端计算校验和， S3 会验证它；如果不设置，只能依赖 TCP/TLS 的可靠性，无法保证端到端的数据一致性。
- **`StorageClass`** 在 PUT 时指定几乎是免费的；但事后修改需要调用 `CopyObject`，可能会产生取回费用。
- **自适应重试** 在收到 503 SlowDown 响应时会自动退避；如果没有这个机制，热点 Prefix 很可能拖垮整个集群。

### 4.4 分片上传

对于几百 MB 以上的文件，推荐使用分片上传：支持并行、断点续传以及分片校验。`boto3.s3.transfer` 已经封装好了这部分逻辑；下面是显式实现版本，方便理解背后的原理：

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

这里的 `try/except` 至关重要：如果分片上传失败但未清理，残留的分片会继续计入存储账单。务必为每个 Bucket 设置生命周期规则（如 `AbortIncompleteMultipartUpload after 7 days`），以避免不必要的费用。
## 5. 副本与纠删码：如何选择？

三副本方案简单直接，纠删码（Erasure Coding, EC）则更加经济。现代云对象存储系统通常会结合两者：热数据采用三副本策略，冷数据则切换到纠删码。

![副本与纠删码的开销和持久性对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/fig4_erasure_vs_replication.png)

### 5.1 背后的数学逻辑

Reed-Solomon 纠删码采用 **k+m** 模型，将一个对象分割为 `k` 个数据分片和 `m` 个校验分片。只要任意 `m` 个分片丢失，数据依然可以恢复。存储开销为 `m/k`。

| 方案         | 存储开销 | 容忍故障数       | 重建时读放大倍数       |
|--------------|----------|------------------|------------------------|
| 三副本       | 200%     | 2 块磁盘         | 1 （只需额外读取一块） |
| EC 6+3       | 50%      | 3 个分片         | 6 （需读取 6 个分片）  |
| EC 10+4      | 40%      | 4 个分片         | 10                    |
| EC 12+4      | 33%      | 4 个分片         | 12                    |

两种方案的权衡非常明显：

- **三副本**：对 CPU 消耗低，但存储成本高；重建速度快（只需复制一个数据块）。
- **纠删码**：存储成本低，但在重建过程中对 CPU 和网络带宽的需求较高，且速度较慢。

热数据通常需要快速访问，因此更适合三副本模式（一次读取只需一次 HTTP 请求）。而冷数据由于访问频率低，使用纠删码更为划算（重建成本可以分摊到多年的时间跨度上）。

### 5.2 “11 个 9”的持久性是如何实现的？

Amazon S3 承诺的对象持久性高达 99.999999999% （即“11 个 9”），这意味着每 1000 亿对象年（object-years）才可能发生一次数据丢失。这个数字并非凭空而来，而是通过精密的工程设计实现的：

- 单块硬盘的年故障率（AFR）约为 1%。
- 将 `k+m` 分片分布到至少 3 个可用区（AZ）中。
- 后台持续运行的数据扫描（scrubbing）机制能够在位翻转累积成磁盘故障之前发现问题。
- 数据修复过程虽然受到带宽限制，但始终保持运行。

实际经验告诉我们：**持久性的关键不在于“我存了三份”，而在于“我能比下一次故障更快完成修复”**。如果你打算自建存储系统，第一步应该是构建并监控修复流水线（repair pipeline）的性能指标。
## 6. 分布式文件系统： HDFS 与 Ceph

在需要跨多台机器实现类似 POSIX 的语义时，有两个设计堪称经典： HDFS （主节点协调、追加写为主）和 Ceph （去中心化、功能全面）。

![HDFS vs Ceph 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/storage-systems/fig5_distributed_fs.png)

### 6.1 HDFS —— 主节点协调，面向批处理优化

HDFS 的设计理念可以归结为三点：

1. **大文件为主，一次写入、多次读取**（如日志、 Parquet 文件、视频）。默认块大小设为 **128 MB**，正是为了适应这种场景。
2. **故障是常态**，数据副本存储在块级别，而不是文件级别。
3. **计算靠近数据**（MapReduce 的核心理念——数据本地性）。

架构概览：

- **NameNode** —— 负责命名空间和块位置的单一可信源。 Active NameNode 和 Standby NameNode 通过 JournalNode 集群（QJM）共享编辑日志，实现高可用（HA）。
- **DataNode** —— 存储实际的数据块，默认副本数为 3。
- **块放置策略** —— 第一个副本放在写入者所在节点（如果写入者不在集群内，则随机选择），第二个副本放在不同机架上，第三个副本放在与第二个副本同机架的不同节点上。这种策略既能应对整个机架的故障，又能保证机架内的读取性能。

```xml
<!-- hdfs-site.xml：生产环境的核心配置 -->
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
# 现代 HDFS 客户端（PyArrow 的 HDFS 接口；hdfs3 已停止维护）
import pyarrow.fs as pafs

hdfs = pafs.HadoopFileSystem(host="namenode", port=9000, user="hadoop")
hdfs.create_dir("/user/data", recursive=True)
with hdfs.open_output_stream("/user/data/events.parquet") as f:
    f.write(open("/local/events.parquet", "rb").read())

for info in hdfs.get_file_info(pafs.FileSelector("/user/data", recursive=False)):
    print(f"{info.path:50s}  {info.size:>12} bytes")
```

**HDFS 不适合的场景**：海量小文件（每个文件会占用约 150 字节的 NameNode 内存）、随机写（仅支持追加写）、低延迟查询。除此之外，在 Hadoop 和 Spark 生态中，它仍然是非常强大的工具。

### 6.2 Ceph —— 一套集群，三种接口

Ceph 的亮点在于 *统一存储*：一个 RADOS 集群同时提供块存储（RBD）、对象存储（RGW）和文件系统（CephFS）三种接口，所有数据都由同一组 OSD 承载。

Ceph 与 HDFS 的两个主要区别：

- **无需中心元数据服务器**。数据放置由 **CRUSH**（Controlled Replication Under Scalable Hashing）算法决定，这是一种伪随机算法，根据对象名和集群拓扑生成确定性的 OSD 列表，彻底消除了 NameNode 的瓶颈。
- **OSD 自管理**。每个 OSD 都知道自己的对等节点，并独立完成副本同步、数据校验和故障恢复，无需中央调度器。

```bash
# 现代 Ceph 部署推荐使用 cephadm + 容器
cephadm bootstrap --mon-ip 10.0.1.10
ceph orch host add node2 10.0.1.11
ceph orch host add node3 10.0.1.12
ceph orch apply osd --all-available-devices

# 创建 EC 4+2 对象池（50% 开销，可容忍 2 个 OSD 故障）
ceph osd erasure-code-profile set ec42 k=4 m=2 \
    crush-failure-domain=host
ceph osd pool create cold_objects erasure ec42
```

```bash
# RBD：精简配置、支持快照的虚拟磁盘
rbd create --size 100G --pool rbd vm-disk-01
rbd map rbd/vm-disk-01            # 映射为 /dev/rbd0
mkfs.xfs /dev/rbd0
rbd snap create rbd/vm-disk-01@before-upgrade
```

```python
# RGW：基于同一集群的 S3 兼容接口
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

Ceph 的灵活性是有代价的：调整 CRUSH 规则、平衡 PG 分布、规划 OSD 容量、监控数据重建流量，这些任务在大规模部署中都需要专人负责。大多数团队应该优先考虑云厂商提供的托管对象存储或块存储服务，只有在遇到硬性需求（如数据主权、本地化部署、定制硬件）时才转向 Ceph。

---
## 7. 数据复制策略与运维实践

### 7.1 同步复制 vs 异步复制

| 维度       | 同步复制                     | 异步复制           |
|------------|-----------------------------|--------------------|
| 数据一致性 | 强一致性（所有副本确认后才返回） | 最终一致性          |
| 延迟       | 受最慢副本的往返时间（RTT）限制  | 仅受限于本地磁盘    |
| RPO        | 0                          | 秒级到分钟级       |
| 吞吐量     | 受最慢链路限制               | 受主节点磁盘性能限制 |
| 使用场景   | 金融账本、单区域数据库高可用     | 跨区域容灾、分析型副本 |

常见模式是：**区域内同步复制**（跨可用区，亚毫秒延迟）+ **跨区域异步复制**（容灾副本， RPO 在几秒内）。

### 7.2 仲裁机制

对于 `N` 个副本，读写仲裁分别记为 `R` 和 `W`：

- `W + R > N` 确保每次读取都能覆盖最新的写入。
- `W = N` 提供最强的写入一致性（任何节点故障都会阻塞写入）；`W = 1` 则是最弱的一致性。
- 类似 Dynamo 的存储系统允许 *应用层* 按需选择 `R` 和 `W`；而 ZooKeeper 和 etcd 则固定使用多数派规则（`W = R = majority`）。

### 7.3 生命周期管理 —— 自动化成本优化

成本优化通常不需要复杂的工程改造，而是通过简单的策略配置实现。例如：日志数据在 30 天后转为低频访问存储（IA）， 90 天后归档至 Glacier， 365 天后删除；清理未完成的分片上传任务；自动过期旧版本对象：

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

### 7.4 备份策略与 RTO/RPO 权衡

| 策略             | RTO         | RPO         | 成本   | 适用场景           |
|------------------|-------------|-------------|--------|--------------------|
| 热备（多区域双活） | 秒级        | 几乎为零     | 2-3 倍 | 支付系统、社交动态流 |
| 温备（备用副本待命） | 分钟级      | 秒级        | 1.2-1.5 倍 | 大部分生产环境 Web 应用 |
| 火种（仅数据备份） | 小时级      | 分钟级      | 1.05-1.1 倍 | 内部工具或非核心系统 |
| 仅备份恢复       | 小时到天级   | 小时级      | 1.01 倍 | 开发测试、数据分析 |

最优解往往不是最严格的方案，而是那个 *成本与业务中断损失相匹配* 的方案。
## 8. 性能优化

### 8.1 长尾延迟才是关键

在性能优化中，真正值得关注的是 **P99 延迟**，而不是平均延迟。假设一个系统需要向 10 个后端发起请求并等待所有响应，那么整体请求的 P99 延迟大致等同于单个后端的 P99 延迟（Dean & Barroso，*The Tail at Scale*, 2013）。换句话说，每个依赖都会成为延迟的放大器。

### 8.2 四大优化方向

- **并行化** —— 使用分片上传、 Range GET 请求、并发处理多个前缀（Prefix）。
- **数据本地性** —— 尽量让计算靠近存储（例如 HDFS 的数据本地性策略，或者 S3 跨区域设置中的可用区亲和性）。
- **缓存策略** —— 在应用层实现 LRU 缓存；为公开的 S3 存储启用 CloudFront 加速；利用 OSS CDN 为热点前缀提供更快的访问。
- **压缩算法** —— zstd 是目前的主流选择（level 3 的速度接近 gzip-6，但压缩比能达到 gzip-9）；如果 CPU 资源紧张，可以考虑使用 Snappy。

### 8.3 吞吐量优化的经验法则

在长距离网络链路上，单条 TCP 流的吞吐量受限于公式 `窗口大小 / RTT`。例如，在 RTT 为 100 毫秒、默认窗口大小为 64 KB 的情况下，吞吐量的上限大约是 5 Mbit/s。要充分利用一条 10 Gbit/s 的带宽，要么增大窗口大小（现代内核的自动调优功能可以帮忙），要么增加并发流的数量。分片上传的本质就是通过多流并发来提升吞吐量 —— 这正是它高效的原因，而不仅仅是表现出来的结果。
## 9. 成本优化的艺术

### 9.1 账单的三大组成部分

| 账单项       | 影响因素          | 优化手段                     |
|--------------|-------------------|------------------------------|
| 存储         | 按 GB-月计费      | 数据分层、压缩、去重、生命周期过期 |
| 请求         | PUT / GET / LIST 次数 | 批量操作、分片上传、 Prefix 规划   |
| 出网流量     | 跨地域或出云的 GB 数 | 使用 CDN、区域内部计算、专线互联    |

对于大多数团队来说，**出网流量往往是账单中的“隐藏炸弹”**。举个例子： 1 PB 的跨地域数据复制，按 $0.02/GB 计算，直接产生 $20,000 的费用。因此，尽量在同一个区域内完成数据复制；如果必须跨地域传输，建议通过更经济的方式（如 Direct Connect、专用对等连接或阿里云的 Cloud Enterprise Network）来降低成本。

### 9.2 实际案例：成本估算

假设有一个 1 TB 的数据集，每天访问两次，并存储一年：

- **标准存储：** 1 TB × $0.023 × 12 = **$276/年**，请求费用几乎可以忽略不计。
- **标准存储 + 30 天后转为低频访问存储（IA）：** 存储费用约 $160/年，加上 IA 数据取回费用约 $10，总计 ≈ **$170/年**。
- **90 天后转为深度归档存储（Deep Archive），每年取回两次用于重新训练：** 存储费用约 $15/年，加上取回费用约 $40，总计 ≈ **$55/年**。

这里的关键不是具体的数字——因为官方定价会随时间调整——而是数量级的变化。**合理利用生命周期策略，可以将存储成本降低 5 到 10 倍，而且启用生命周期规则本身是免费的**。
## 10. 常见问题

**问：新数据湖应该选 HDFS 还是 S3？**

对于大多数新项目， S3 （或类似服务）是更好的选择。现代计算引擎如 Trino、 Spark、 Flink 和 DuckDB 都原生支持对象存储，使用 S3 可以实现存储与计算的分离（独立扩展），还能避免维护复杂的 NameNode 高可用集群。只有在必须本地部署或者对数据本地性有极高要求的情况下，才需要考虑 HDFS。

**问： Ceph 在什么情况下值得投入运维成本？**

主要有三种场景：一是监管要求数据必须存放在自有硬件上；二是块存储和对象存储的工作负载非常大，放到云上的托管服务成本过高；三是已经运行了 OpenStack 环境，并且能从 Ceph 的深度集成中获益。除此之外，考虑到 SRE 的人力成本，云服务商提供的托管方案通常更划算。

**问：数据存储该用 3 副本还是纠删码（Erasure Coding）？**

热数据和对延迟敏感的数据适合用副本机制，而冷数据以及大文件则更适合用纠删码。实际上，很多系统会自动处理这两种策略，比如 S3 内部的机制，或者 Ceph 通过分层存储池实现的方式。应用层通常不需要手动干预这个选择。

**问：如何为对象存储设计缓存容量？**

可以通过绘制访问频率与对象数量的关系图，找到“拐点”。典型的 Web 工作负载中，缓存最热门的 1%-5% 的对象，就能满足 80%-95% 的 GET 请求。再增加缓存容量的话，收益会迅速递减，因为更多的 RAM 投入并不会显著提升命中率。 CloudFront 或 OSS CDN 则进一步简化了这一问题，将其从容量规划转变为配置选项。