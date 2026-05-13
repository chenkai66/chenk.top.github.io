---
title: "阿里云全栈实战（四）：OSS——对象存储最佳实践"
date: 2026-05-01 09:00:00
tags:
  - Alibaba Cloud
  - OSS
  - Storage
  - CDN
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 4
description: "掌握阿里云 OSS：存储桶类型、存储类别、访问控制（ACL、RAM、STS、签名 URL）、生命周期规则、跨区域复制、CDN 集成、自定义域名。构建完整的媒体存储后端。"
disableNunjucks: true
translationKey: "aliyun-fullstack-4"
---
以前我把用户上传的文件直接塞进 ECS 磁盘里。头像、PDF 发票、CSV 导出文件——统统丢在一台跑着 Flask 应用的 `ecs.g7.large` 实例的 `/var/data/uploads/` 目录下。我还写了个 cron 任务，每六小时把整个目录 rsync 到另一台 ECS 上，美其名曰“备份”。直到某个周五凌晨三点，一个批处理任务生成了 40GB 没人下载的报表，系统盘瞬间飙到 100%，实例进入只读状态，应用彻底宕机——而上一次 rsync 还是前一天晚上跑的。我丢了整整六小时的用户上传数据，整个周末都在向客户道歉。正是这次事故让我明白：对象存储不是可有可无的附加项，而是云上架构的基石。你的应用服务器是临时的，但数据必须持久。

本文将从第一性原理出发，带你完整掌握阿里云对象存储 OSS（Object Storage Service），直至生产级部署。读完后，你将拥有一个功能完备的媒体存储后端，支持生命周期管理、CDN 加速，并能通过 Python API 生成预签名上传链接。我们在 [Part 2](/zh/aliyun-fullstack/02-ecs-compute/) 和 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 中已经搭建好了 VPC 与 ECS 基础设施——现在，我们为其加上这个能抵御实例故障、轻松扩展至 PB 级、成本却仅为块存储零头的存储层。

## 什么是 OSS？

OSS 是阿里云对标 AWS S3 的对象存储服务。你将文件（称为“对象”）存入名为“桶”（Bucket）的容器中，每个对象由唯一的 key（即路径字符串）、数据本身和元数据组成。这就是全部的数据模型——没有目录结构，没有文件层级，也没有 POSIX 文件系统语义。当你在 OSS 中看到 `images/2026/05/avatar.png` 时，斜杠只是 key 字符串的一部分，并非真实目录。控制台为了便于浏览会将其渲染为文件夹，但底层存储本质上是扁平的。

这种简洁正是其强大之处。由于无需维护复杂的文件系统树，OSS 能透明地将对象分布到成千上万个存储节点上。你完全不用操心容量规划、磁盘 IOPS 或 RAID 配置。只需 PUT 一个对象，OSS 就会自动决定存储位置、跨可用区复制以保障持久性，并在你 GET 时高效返回。标准存储的持久性高达 99.9999999999%（十二个 9），这意味着即使你存储一百亿个对象，设计上也最多丢失一个。

### 三种云存储类型

阿里云提供三种本质不同的存储产品，选错是新手最常见的失误：

![OSS 存储类型对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_storage_classes.png)

| 存储类型 | 产品 | 访问模式 | 类比 |
|---|---|---|---|
| **块存储** | EBS (云盘) | 挂载到单台 ECS，支持随机读写 | 插在电脑上的硬盘 |
| **文件存储** | NAS / CPFS | 多台 ECS 通过 NFS/SMB 共享访问 | 办公室里的网络共享盘 |
| **对象存储** | OSS | 通过 HTTP API 访问，无需挂载，容量近乎无限 | 带 API 的 Dropbox |

**块存储**（即挂载到 ECS 的云盘）提供原始块设备，由操作系统格式化为 ext4 或 xfs。它延迟低、性能高，支持随机 I/O，非常适合数据库、系统盘或任何依赖 POSIX 语义的应用。但缺点也很明显：只能挂载到一台 ECS，且无论是否使用，都要为预分配的容量付费。

**文件存储**（如 NAS）提供共享文件系统，多台 ECS 可同时通过 NFS v3/v4 或 SMB 挂载。适用于需要共享 `/data` 目录的遗留系统、CMS 或开发环境。但它每 GB 成本较高，且性能取决于所购的容量层级。

**对象存储**（OSS）则适用于其余绝大多数场景——而这部分通常占你总数据量的 90%。静态资源、用户上传、备份、日志、数据湖文件、机器学习训练集、音视频、文档……只要通过 HTTP 访问，且不需要随机修改文件中间的字节，OSS 就是最优解。

### OSS 与 AWS S3 对照

如果你熟悉 AWS，映射关系非常直观：

| AWS S3 概念 | OSS 对应项 | 说明 |
|---|---|---|
| Bucket | Bucket | 命名规则相同（3–63 字符） |
| Object | Object | key/value/metadata 模型一致 |
| Region | Region | Bucket 按区域隔离 |
| S3 Standard | Standard | 热数据，频繁访问 |
| S3 Standard-IA | Infrequent Access (IA) | 最少存储 30 天 |
| S3 Glacier | Archive | 最少 90 天，1 分钟快速恢复 |
| S3 Glacier Deep Archive | Deep Cold Archive | 最少 180 天，需数小时恢复 |
| Presigned URL | Signed URL | 概念相同，SDK 方法名略有差异 |
| Bucket Policy | Bucket Policy | JSON 格式，语法相似 |
| S3 Lifecycle | Lifecycle Rules | 支持相同的转换与过期逻辑 |
| Cross-Region Replication | Cross-Region Replication | 异步复制模型一致 |
| CloudFront + S3 | CDN + OSS | 原生集成，回源逻辑相同 |

主要区别在于：OSS 使用 AccessKey ID/Secret 而非 AWS Signature V4（不过 SDK 已封装）；端点格式为 `oss-{region}.aliyuncs.com`，而非 `s3.{region}.amazonaws.com`；更重要的是，OSS 为每个区域提供了“内网端点”（如 `oss-cn-beijing-internal.aliyuncs.com`），同地域 ECS 访问时流量免费——而 AWS 对同类流量收费。

### 核心概念

动手编码前，务必理解以下四点：

**Bucket** —— 对象的全局唯一容器。名称需为 3–63 位小写字母、数字或连字符，且必须以字母或数字开头结尾。Bucket 按区域划分（如 `cn-beijing` 的数据就存北京），创建后无法重命名或迁移。

**Object** —— 存于 Bucket 中的文件，由 key（路径字符串）唯一标识。单个对象最大可达 48.8 TB。对象不可变：更新时需整体替换，无法原地修改字节。

**Region 与 Endpoint** —— 每个 Bucket 仅属于一个区域。可通过公网端点（`oss-cn-beijing.aliyuncs.com`）、内网端点（同地域 ECS 免费访问）或绑定的自定义域名访问。

**AccessKey** —— API 访问凭证。生产环境中切勿使用主账号 AccessKey，应使用 RAM 用户或 STS 临时凭证（详见下文访问控制部分）。

## 存储类型

OSS 提供五种存储类型，选对可节省 80% 成本，选错则可能让账单暴涨十倍。核心原则是：存储越便宜，取回越贵、越慢。

| 存储类型 | $/GB/月 \|最低存储时长 \|取回费用 \|恢复时间 \|适用场景 \|
|---|---|---|---|---|---|
| **Standard** | ~0.020 | 无 | 免费 | 即时 | 热数据，高频访问 |
| **Infrequent Access (IA)** | ~0.012 | 30 天 | ~0.010/GB | 即时 | 每月访问 <1–2 次 |
| **Archive** | ~0.005 | 90 天 | ~0.020/GB | 1 分钟（加急） | 季度报表、旧备份 |
| **Cold Archive** | ~0.002 | 180 天 | ~0.030/GB | 1–10 小时 | 合规归档、法律保留 |
| **Deep Cold Archive** | ~0.001 | 180 天 | ~0.050/GB | 12–48 小时 | 几乎永不读取的数据 |

*以上价格为 cn-beijing 区域估算值，具体请参考 [OSS 定价页](https://www.alibabacloud.com/product/object-storage-service/pricing)。*

几个常见误区：

**最低存储时长按账单计费，而非实际存储时间。** 例如，将文件存入 Archive 后 10 天删除，仍需支付 90 天费用。此规则适用于除 Standard 外的所有类型。

**取回费用按 GB 计算。** 从 Cold Archive 恢复 1TB 数据，仅取回费就约 30 美元，还不含传输费。归档前务必三思。

**IA 有最小计费单元。** 小于 64KB 的对象按 64KB 收费。若存储数百万个微小 JSON 文件，IA 成本反而高于 Standard。

**Archive 与 Cold Archive 需手动恢复。** 无法直接读取，必须先发起恢复请求，等待完成后才能在指定时段（1–7 天）内访问，之后自动回归归档状态。

黄金法则：所有数据初始均设为 Standard，通过 OSS 访问日志观察 30 天访问模式，再配置生命周期规则自动降冷。切勿凭感觉猜测。

## 创建与管理 Bucket

### 控制台操作指南

![Bucket CRUD 操作](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_bucket_operations.png)

最快创建 Bucket 的方式：

1. 打开 [OSS 控制台](https://oss.console.aliyun.com/)
2. 点击 **创建 Bucket**
3. 输入全局唯一名称（如 `myapp-prod-media-cn`）
4. 选择区域（如 `cn-beijing`）
5. 存储类型：Standard（后续可通过生命周期规则调整）
6. 访问控制：**私有**（始终以此起步）
7. 版本控制：启用（后续可暂停，但启用后不会追溯已有对象）
8. 服务端加密：AES-256 或 KMS（推荐 AES-256，免费且透明）
9. 点击 **确定**

### 使用 ossutil 命令行工具

`ossutil` 是 OSS 官方 CLI 工具。先安装：

```bash
# macOS
brew install ossutil

# Linux (amd64)
curl -o ossutil https://gosspublic.alicdn.com/ossutil/v2/2.0.3/ossutil-linux-amd64
chmod +x ossutil
sudo mv ossutil /usr/local/bin/

# Configure credentials
ossutil config set --access-key-id $ALIBABA_CLOUD_ACCESS_KEY_ID \
                   --access-key-secret $ALIBABA_CLOUD_ACCESS_KEY_SECRET \
                   --region cn-beijing
```

然后创建 Bucket 并操作对象：

```bash
# Create a bucket in cn-beijing with Standard storage
ossutil mb oss://myapp-prod-media --region cn-beijing

# Upload a single file
ossutil cp ./avatar.png oss://myapp-prod-media/images/users/avatar.png

# Upload a directory recursively
ossutil cp ./static/ oss://myapp-prod-media/static/ --recursive

# List objects in a bucket
ossutil ls oss://myapp-prod-media/

# List with details (size, last modified, storage class)
ossutil ls oss://myapp-prod-media/ --all-versions

# Download a file
ossutil cp oss://myapp-prod-media/images/users/avatar.png ./downloaded-avatar.png

# Copy between buckets
ossutil cp oss://myapp-prod-media/images/ oss://myapp-backup-media/images/ \
  --recursive

# Delete a file
ossutil rm oss://myapp-prod-media/old-file.txt

# Delete all objects with a prefix
ossutil rm oss://myapp-prod-media/temp/ --recursive

# Get bucket info
ossutil bucket-info oss://myapp-prod-media
```

### Bucket 命名规范

- 长度 3–63 字符
- 仅限小写字母、数字、连字符
- 必须以字母或数字开头结尾
- 全阿里云全局唯一（不仅限于你的账号）
- 创建后不可重命名

我习惯采用 `{app}-{env}-{purpose}-{region-short}` 格式，如 `myapp-prod-media-cn` 或 `myapp-staging-logs-cn`。这样既能避免命名冲突，也能在凌晨两点面对三十个 Bucket 时一眼认出用途。

### 版本控制

启用版本控制后，每次覆盖或删除对象都会保留历史版本。例如，覆盖 `report.pdf` 时，旧版变为“非当前版本”；删除时则添加删除标记，但数据仍保留。

```bash
# Enable versioning
ossutil bucket-versioning --method put oss://myapp-prod-media enabled

# Check versioning status
ossutil bucket-versioning --method get oss://myapp-prod-media

# List all versions of objects
ossutil ls oss://myapp-prod-media/ --all-versions
```

任何涉及用户数据的 Bucket 都应开启版本控制。虽然存储成本可能翻倍，但相比因误覆盖导致数据永久丢失的风险，这点代价微不足道。配合生命周期规则，可设置 30 天后自动清理非当前版本，有效控制成本。

## 访问控制深度解析

OSS 的访问控制分为四层，理解其优先级是避免数据泄露的关键。权限判定顺序为：STS/RAM 策略 > Bucket 策略 > Bucket ACL。

![OSS 访问控制模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_access_control.png)

### 第一层：Bucket ACL

最简单但也最粗糙的控制方式，仅三种选项：

| ACL | 匿名可读 | 匿名可写 | 适用场景 |
|---|---|---|---|
| **private** | 否 | 否 | 默认选项，适用于绝大多数场景 |
| **public-read** | 是 | 否 | 静态网站、公共 CDN 源站 |
| **public-read-write** | 是 | 是 | **绝对禁止使用** |

```bash
# Set bucket ACL
ossutil bucket-acl --method put oss://myapp-prod-media private

# Check bucket ACL
ossutil bucket-acl --method get oss://myapp-prod-media
```

关于 `public-read-write`，我绝非危言耸听。一旦设置，任何互联网用户都能向你的 Bucket 上传任意文件，不仅会导致账单暴增，还可能被用作恶意软件分发点。我在生产环境中亲眼见过此类事故，切勿重蹈覆辙。

`public-read` 仅适用于无需 CDN、直接通过 OSS 提供静态资源的极简场景。即便如此，我也更推荐保持 `private`，并通过 CDN 的源站访问身份（Origin Access Identity）授权——后文会详述。

### 第二层：Bucket Policy

Bucket Policy 是附加在 Bucket 上的 JSON 策略文档，用于定义细粒度权限。它属于资源策略，类似 S3 Bucket Policy，适合跨账号授权或复杂条件控制，且无需改动 RAM 配置。

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": ["203917385849****"],
      "Action": [
        "oss:GetObject",
        "oss:GetObjectAcl"
      ],
      "Resource": [
        "acs:oss:*:*:myapp-prod-media/shared/*"
      ],
      "Condition": {
        "IpAddress": {
          "acs:SourceIp": ["203.0.113.0/24"]
        }
      }
    }
  ]
}
```

上述策略表示：“允许阿里云账号 `203917385849****` 读取 `shared/` 前缀下的对象，但仅限来自 IP 段 `203.0.113.0/24` 的请求。”你还能按 VPC、时间段、Referer 头或 HTTPS 强制要求进行限制。

通过 CLI 应用策略：

```bash
ossutil bucket-policy --method put oss://myapp-prod-media ./bucket-policy.json
```

### 第三层：RAM Policy

RAM 策略基于身份，可绑定到 RAM 用户、用户组或角色，是应用服务器最常用的授权方式。

为应用创建最小权限的 RAM 用户：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "oss:PutObject",
        "oss:GetObject",
        "oss:DeleteObject",
        "oss:ListObjects"
      ],
      "Resource": [
        "acs:oss:*:*:myapp-prod-media",
        "acs:oss:*:*:myapp-prod-media/*"
      ]
    }
  ]
}
```

注意需同时授权两个资源：Bucket 本身（用于 `ListObjects`）和 `bucket/*`（用于对象操作）。遗漏前者是“ListBuckets 权限拒绝”错误的常见原因。

```bash
# Create RAM user
aliyun ram CreateUser --UserName app-oss-user

# Create and attach the policy
aliyun ram CreatePolicy --PolicyName AppOSSReadWrite \
  --PolicyDocument "$(cat oss-policy.json)"

aliyun ram AttachPolicyToUser --PolicyName AppOSSReadWrite \
  --PolicyType Custom --UserName app-oss-user

# Create AccessKey for the user
aliyun ram CreateAccessKey --UserName app-oss-user
```

### 第四层：STS 临时凭证

安全令牌服务（STS）可颁发有效期可控（15 分钟至 1 小时）的临时凭证，适用于浏览器上传或移动 App——切勿将长期有效的 AccessKey 硬编码到客户端。

典型流程如下：

1. 客户端向后端请求上传凭证
2. 后端调用 STS `AssumeRole`，传入范围受限的策略
3. STS 返回临时 AccessKeyId、AccessKeySecret 和 SecurityToken
4. 客户端使用这些凭证直传 OSS
5. 凭证到期后自动失效

```python
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401.models import AssumeRoleRequest
from alibabacloud_tea_openapi.models import Config

config = Config(
    access_key_id='<RAM_USER_AK>',
    access_key_secret='<RAM_USER_SK>',
    endpoint='sts.cn-beijing.aliyuncs.com'
)
sts_client = StsClient(config)

# Scope the temporary credentials to a specific upload path
policy = '''{
    "Version": "1",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["oss:PutObject"],
        "Resource": ["acs:oss:*:*:myapp-prod-media/uploads/user-12345/*"]
    }]
}'''

request = AssumeRoleRequest(
    role_arn='acs:ram::123456789:role/oss-upload-role',
    role_session_name='user-12345-upload',
    duration_seconds=900,  # 15 minutes
    policy=policy
)

response = sts_client.assume_role(request)
credentials = response.body.credentials
print(f"AccessKeyId: {credentials.access_key_id}")
print(f"AccessKeySecret: {credentials.access_key_secret}")
print(f"SecurityToken: {credentials.security_token}")
print(f"Expiration: {credentials.expiration}")
```

关键细节在于：`AssumeRole` 中的 `policy` 参数会进一步收窄权限。即使角色本身拥有 OSS 全权限，临时凭证也仅能在指定路径执行 `PutObject`。这正是纵深防御的体现。

### 签名 URL

对于临时分享或限时下载，可生成带过期时间的签名 URL：

```bash
# Generate a signed URL valid for 1 hour (3600 seconds)
ossutil sign oss://myapp-prod-media/reports/q1-2026.pdf --timeout 3600
```

生成的 URL 内嵌签名参数，持有者可在过期前直接下载，无需额外认证。

Python 示例：

```python
import oss2

auth = oss2.Auth('<ACCESS_KEY_ID>', '<ACCESS_KEY_SECRET>')
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'myapp-prod-media')

# Generate a signed URL for download, valid for 1 hour
url = bucket.sign_url('GET', 'reports/q1-2026.pdf', 3600)
print(url)

# Generate a signed URL for upload
upload_url = bucket.sign_url('PUT', 'uploads/new-file.pdf', 600,
                              headers={'Content-Type': 'application/pdf'})
print(upload_url)
```

## 上传与下载

### 简单上传

小于 5GB 的文件，直接 PUT 即可：

```python
import oss2

auth = oss2.Auth('<ACCESS_KEY_ID>', '<ACCESS_KEY_SECRET>')
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'myapp-prod-media')

# Upload from a local file
bucket.put_object_from_file('images/photo.jpg', '/tmp/photo.jpg')

# Upload from memory (bytes or string)
bucket.put_object('config/settings.json', '{"debug": false, "version": 3}')

# Upload with metadata
headers = {
    'Content-Type': 'image/jpeg',
    'x-oss-meta-uploaded-by': 'user-12345',
    'x-oss-meta-original-filename': 'vacation.jpg'
}
bucket.put_object_from_file('images/photo.jpg', '/tmp/photo.jpg', headers=headers)
```

### 分片上传

大于 100MB 的文件建议使用分片上传。文件被切分为多个片段（除最后一片外，每片至少 100KB），并行上传后由服务端组装。若某片段失败，仅需重传该片段。

```python
import oss2

auth = oss2.Auth('<ACCESS_KEY_ID>', '<ACCESS_KEY_SECRET>')
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'myapp-prod-media')

# The SDK handles multipart automatically for large files
# Default part size is 10 MB, parallelism is 1
oss2.resumable_upload(
    bucket,
    'videos/presentation.mp4',
    '/tmp/presentation.mp4',
    part_size=10 * 1024 * 1024,    # 10 MB per part
    num_threads=4                   # Upload 4 parts in parallel
)
```

底层 `resumable_upload` 自动完成以下步骤：
1. 调用 `InitiateMultipartUpload` 获取上传 ID
2. 切分文件
3. 并行执行 `UploadPart`
4. 调用 `CompleteMultipartUpload` 组装对象
5. 本地保存 checkpoint 文件，支持断点续传

### 断点续传下载

在网络不稳定的环境下下载大文件：

```python
oss2.resumable_download(
    bucket,
    'videos/presentation.mp4',
    '/tmp/downloaded-presentation.mp4',
    part_size=10 * 1024 * 1024,
    num_threads=4
)
```

### 使用 ossutil 批量操作

```bash
# Upload a directory with 8 parallel threads
ossutil cp ./media/ oss://myapp-prod-media/media/ \
  --recursive --jobs 8 --parallel 4

# Download with filters
ossutil cp oss://myapp-prod-media/logs/2026-05/ ./local-logs/ \
  --recursive --include "*.gz"

# Sync (like rsync -- only uploads changed files)
ossutil sync ./static/ oss://myapp-prod-media/static/ --delete

# The --delete flag removes objects in OSS that don't exist locally.
# Be very careful with this -- test without --delete first.
```

`--delete` 选项会删除 OSS 中本地不存在的文件，风险较高，建议先不带该参数试运行。

### 浏览器直传（Presigned URL）

面向用户的应用最常用模式：服务端生成预签名 PUT URL，浏览器直接上传至 OSS，全程不经过应用服务器。

```python
# Server-side: generate presigned upload URL
url = bucket.sign_url(
    'PUT',
    f'uploads/{user_id}/{filename}',
    600,  # 10 minutes
    headers={
        'Content-Type': content_type,
        'x-oss-forbid-overwrite': 'true'  # Prevent overwrites
    }
)
# Return this URL to the frontend
```

```javascript
// Client-side: upload directly to OSS
async function uploadToOSS(file, presignedUrl) {
  const response = await fetch(presignedUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': file.type,
      'x-oss-forbid-overwrite': 'true'
    },
    body: file
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }
  return response;
}
```

这种方式避免了应用服务器代理上传流量，否则带宽和内存消耗将与文件大小成正比。借助预签名 URL，浏览器直连 OSS，服务器仅负责协调。

## 生命周期规则

生命周期规则可自动转换存储类型或清理过期对象，是降本增效的核心手段。配置一次，长期受益。

![存储生命周期转换时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_lifecycle_rules.png)

### 常见模式

**模式一：渐进式归档**

```json
{
  "Rules": [
    {
      "ID": "progressive-archive",
      "Prefix": "logs/",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "IA"
        },
        {
          "Days": 90,
          "StorageClass": "Archive"
        },
        {
          "Days": 365,
          "StorageClass": "ColdArchive"
        }
      ],
      "Expiration": {
        "Days": 730
      }
    }
  ]
}
```

该规则作用于 `logs/` 前缀，含义如下：
- 30 天后转为低频访问（节省约 40%）
- 90 天后转为归档（节省约 75%）
- 365 天后转为冷归档（节省约 90%）
- 730 天（两年）后彻底删除

**模式二：清理未完成的分片上传**

未完成的分片上传会占用存储空间，但在常规列表中不可见，容易悄然累积。务必添加此规则：

```json
{
  "Rules": [
    {
      "ID": "abort-incomplete-uploads",
      "Prefix": "",
      "Status": "Enabled",
      "AbortMultipartUpload": {
        "Days": 3
      }
    }
  ]
}
```

**模式三：清理旧版本**

启用版本控制后，非当前版本会持续堆积。可通过以下规则定期清理：

```json
{
  "Rules": [
    {
      "ID": "cleanup-old-versions",
      "Prefix": "",
      "Status": "Enabled",
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      }
    }
  ]
}
```

### 通过 CLI 应用规则

```bash
# 从 JSON 文件应用生命周期配置
ossutil lifecycle --method put oss://myapp-prod-media ./lifecycle.json

# 查看当前生命周期规则
ossutil lifecycle --method get oss://myapp-prod-media
```

### 成本影响

以下是我管理的一个生产 Bucket 的真实数据：2TB 日志，每月新增约 50GB。

| 策略 | 月成本 | 年成本 |
|---|---|---|
| 全部 Standard，无生命周期 | ~$40 \|~$480 |
| 30 天转 IA，90 天归档 | ~$18 \|~$216 |
| 30 天转 IA，90 天归档，365 天删除 | ~$14 \|~$168 |

仅靠一个 JSON 配置文件，成本降低 65%。若组织内有 20 个 Bucket，十分钟工作量每年可省数千美元。

## 跨区域复制（CRR）

CRR 能将对象异步复制到另一区域的目标 Bucket，主要用于两类场景：

![跨区域复制拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_crr_topology.png)

1. **灾难恢复** —— 若 `cn-beijing` 区域故障，数据在 `cn-shanghai` 仍有副本
2. **合规要求** —— 法规强制数据需在特定地理区域留存副本

### 配置 CRR

```bash
# 步骤 1：在另一个 Region 创建目标 bucket
ossutil mb oss://myapp-dr-media --region cn-shanghai

# 步骤 2：通过控制台或 API 启用 CRR
# (ossutil 不支持直接配置 CRR -- 用控制台或 SDK)
```

通过 SDK：

```python
import oss2
from oss2.models import ReplicationRule

auth = oss2.Auth('<ACCESS_KEY_ID>', '<ACCESS_KEY_SECRET>')
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'myapp-prod-media')

rule = ReplicationRule(
    rule_id='replicate-to-shanghai',
    target_bucket_name='myapp-dr-media',
    target_bucket_location='oss-cn-shanghai',
    target_transfer_type='oss_acc',  # 使用传输加速
    prefix_list=['images/', 'documents/'],  # 只复制这些前缀
    action_list=['ALL'],  # 复制 PUT, DELETE, 和 AbortMultipartUpload
    is_enable_historical_object_replication=True  # 复制现有对象
)

bucket.put_bucket_replication(rule)
```

### CRR 细节

| 项目 | 说明 |
|---|---|
| **复制延迟** | 通常 <10 分钟，大对象可能更长 |
| **复制内容** | 对象数据、元数据、ACL（可选） |
| **不复制内容** | 生命周期规则、Bucket 策略、服务端加密设置 |
| **成本** | 目标区域存储费 + 跨区域传输费 |
| **方向** | 默认单向，双向需配置两条规则 |
| **删除同步** | 可选，可控制删除操作是否传播 |

重要提醒：CRR 为最终一致性，无复制时效 SLA。切勿将其视为实时同步方案。如需跨区域强一致性访问，应考虑 CEN 与多区域部署。

## CDN 集成

阿里云 CDN 与 OSS 的组合是生产环境的黄金搭档。CDN 边缘节点将对象缓存至用户附近，访问延迟从数百毫秒降至个位数，仅缓存未命中时才回源至 OSS。

![OSS CDN 集成数据流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_cdn_flow.png)

### 为何要用 CDN + OSS 而非直接访问 OSS？

| 指标 | 直连 OSS | CDN + OSS |
|---|---|---|
| **延迟** | 50–200ms（随用户位置波动） | 5–30ms（就近访问） |
| **每 GB 传输成本** | ~0.12 美元（公网） | ~0.04 美元（CDN 高流量更优） |
| **DDoS 防护** | 基础防护 | CDN 边缘内置高级防护 |
| **HTTPS** | 支持 | CDN 提供免费证书 |
| **缓存控制** | 无 | 可配置 TTL，支持缓存刷新 API |
| **自定义域名** | 支持但无免费 HTTPS | 完整支持自定义域名 + 免费 HTTPS |

只要是面向用户的内容（图片、CSS、JS、视频、下载文件），CDN + OSS 绝对优于直连 OSS。唯一例外是纯后端 API 访问（如服务间程序化读取）。

### 完整 CDN + OSS 配置流程

#### 步骤一：添加 CDN 域名

```bash
# 使用 aliyun CLI
aliyun cdn AddCdnDomain \
  --CdnType web \
  --DomainName cdn.example.com \
  --Sources '[{
    "content": "myapp-prod-media.oss-cn-beijing.aliyuncs.com",
    "type": "oss",
    "priority": "20",
    "port": 443
  }]'
```

#### 步骤二：配置 CNAME DNS

添加 CDN 域名后，阿里云会提供 CNAME 地址（如 `cdn.example.com.w.kunlunsl.com`），在你的 DNS 中添加记录：

```text
cdn.example.com  CNAME  cdn.example.com.w.kunlunsl.com
```

#### 步骤三：启用免费 HTTPS 证书

```bash
aliyun cdn SetDomainServerCertificate \
  --DomainName cdn.example.com \
  --ServerCertificateStatus on \
  --CertType free
```

阿里云 CDN 提供免费 DV（域名验证）证书，自动续期。生产环境也可上传自有证书或使用证书管理服务。

#### 步骤四：设置缓存规则

```bash
# 图片缓存 30 天
aliyun cdn BatchSetCdnDomainConfig \
  --DomainNames cdn.example.com \
  --Functions '[{
    "functionName": "filetype_based_ttl_set",
    "functionArgs": [{
      "argName": "ttl", "argValue": "2592000"
    }, {
      "argName": "file_type", "argValue": "jpg,jpeg,png,gif,webp,svg,ico"
    }, {
      "argName": "weight", "argValue": "99"
    }]
  }]'
```

#### 步骤五：优化回源配置

OSS 作为 CDN 源站虽可直接使用，但建议启用以下优化：

```bash
# 启用 CDN 访问 OSS 私有 bucket
aliyun cdn BatchSetCdnDomainConfig \
  --DomainNames cdn.example.com \
  --Functions '[{
    "functionName": "l2_oss_key",
    "functionArgs": [{
      "argName": "private_oss_auth", "argValue": "on"
    }]
  }]'
```

此配置允许 CDN 访问私有 Bucket，无需公开 Bucket 权限。CDN 通过内部鉴权机制获取对象，既保障安全，又确保缓存未命中时能正常回源。

#### 步骤六：验证配置

```bash
# 测试 CDN 解析
dig cdn.example.com

# 测试内容分发
curl -I https://cdn.example.com/images/test.jpg

# 检查响应头 -- 找这些字段：
# X-Cache: HIT or MISS (CDN 缓存状态)
# Via: S.mix... (CDN 边缘节点标识)
# Age: 3600 (缓存后的秒数)
```

### 缓存刷新

当 OSS 文件更新但 CDN 仍返回旧版本时：

```bash
# 刷新特定 URL
aliyun cdn RefreshObjectCaches \
  --ObjectPath "https://cdn.example.com/images/logo.png" \
  --ObjectType File

# 刷新整个目录
aliyun cdn RefreshObjectCaches \
  --ObjectPath "https://cdn.example.com/static/" \
  --ObjectType Directory
```

## 图片处理（IMM）

OSS 内置图片处理能力，通过 URL 参数即可实时转换图像，无需额外服务或预处理流水线。

![通过 IMM 的图片处理流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_image_processing.png)

### 基础变换

```bash
# 原图
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg

# 缩放至 800px 宽，保持 aspect ratio
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,w_800

# 缩放至 200x200 中心裁剪
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,m_fill,w_200,h_200

# 转 WebP 格式（省 ~30% 带宽）
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/format,webp

# 降低质量 (80%)
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/quality,q_80

# 链式操作：resize + webp + quality
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,w_800/format,webp/quality,q_80
```

### 添加水印

```bash
# 文字水印
?x-oss-process=image/watermark,text_Q2hlbmsgQmxvZw==,type_d3F5LXplbmhlaQ,size_30,color_FFFFFF,t_80,g_se,x_10,y_10

# 文字是 base64 编码。"Chenk Blog" = Q2hlbmsgQmxvZw==
# g_se = southeast (右下角)
# t_80 = 80% 透明度
```

### 获取图片信息

```bash
# 获取图片尺寸、格式、文件大小
curl "https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/info"

# Response:
# {
#   "FileSize": {"value": "2458632"},
#   "Format": {"value": "jpg"},
#   "ImageHeight": {"value": "2048"},
#   "ImageWidth": {"value": "3072"}
# }
```

### 与 CDN 协同工作

访问 `https://cdn.example.com/images/photo.jpg?x-oss-process=image/resize,w_800/format,webp` 时，CDN 会缓存处理后的版本。后续相同请求直接命中 CDN 缓存，无需回源 OSS，兼顾实时处理与高速分发。

处理后的图像与原图独立缓存，完整 URL（含查询参数）作为缓存键。因此 `photo.jpg`、`photo.jpg?x-oss-process=image/resize,w_800` 和 `photo.jpg?x-oss-process=image/resize,w_400` 被视为三个独立缓存项。

## 解决方案：媒体存储后端

现在，我们将前述内容整合为一个完整的媒体存储后端：包含生命周期规则的 OSS Bucket、绑定自定义域名的 CDN，以及一个 Python Flask API，用于生成预签名上传 URL 并通过 CDN 提供带处理能力的图片服务。

### 第一步：创建并配置 Bucket

```bash
# Create bucket
ossutil mb oss://myapp-prod-media --region cn-beijing

# Enable versioning
ossutil bucket-versioning --method put oss://myapp-prod-media enabled

# Set CORS for browser uploads
cat > /tmp/cors.json << 'CORS'
{
  "CORSRules": [
    {
      "AllowedOrigin": ["https://myapp.example.com"],
      "AllowedMethod": ["PUT", "GET", "HEAD"],
      "AllowedHeader": ["*"],
      "ExposeHeader": ["ETag", "x-oss-request-id"],
      "MaxAgeSeconds": 3600
    }
  ]
}
CORS

ossutil cors --method put oss://myapp-prod-media /tmp/cors.json
```

### 第二步：应用生命周期规则

```bash
cat > /tmp/lifecycle.json << 'LIFECYCLE'
{
  "Rules": [
    {
      "ID": "user-uploads-archive",
      "Prefix": "uploads/",
      "Status": "Enabled",
      "Transitions": [
        {"Days": 30, "StorageClass": "IA"},
        {"Days": 90, "StorageClass": "Archive"}
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    },
    {
      "ID": "temp-cleanup",
      "Prefix": "temp/",
      "Status": "Enabled",
      "Expiration": {"Days": 1}
    },
    {
      "ID": "abort-incomplete-uploads",
      "Prefix": "",
      "Status": "Enabled",
      "AbortMultipartUpload": {"Days": 3}
    }
  ]
}
LIFECYCLE

ossutil lifecycle --method put oss://myapp-prod-media /tmp/lifecycle.json
```

### 第三步：接入 CDN

```bash
# Add CDN domain pointing to OSS
aliyun cdn AddCdnDomain \
  --CdnType web \
  --DomainName media.myapp.com \
  --Sources '[{
    "content": "myapp-prod-media.oss-cn-beijing.aliyuncs.com",
    "type": "oss",
    "priority": "20",
    "port": 443
  }]'

# Enable HTTPS
aliyun cdn SetDomainServerCertificate \
  --DomainName media.myapp.com \
  --ServerCertificateStatus on \
  --CertType free

# Enable private bucket origin access
aliyun cdn BatchSetCdnDomainConfig \
  --DomainNames media.myapp.com \
  --Functions '[{
    "functionName": "l2_oss_key",
    "functionArgs": [{"argName": "private_oss_auth", "argValue": "on"}]
  }]'

# Add CNAME record in your DNS provider:
# media.myapp.com  CNAME  media.myapp.com.w.kunlunsl.com
```

### 第四步：Flask API 实现

```python
"""
Media storage API with presigned OSS uploads and CDN delivery.

Requirements:
    pip install flask oss2 alibabacloud-sts20150401
"""

import os
import uuid
import time
from flask import Flask, request, jsonify
import oss2
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401.models import AssumeRoleRequest
from alibabacloud_tea_openapi.models import Config

app = Flask(__name__)

# Configuration
OSS_REGION = 'cn-beijing'
OSS_BUCKET_NAME = 'myapp-prod-media'
OSS_ENDPOINT = f'https://oss-{OSS_REGION}.aliyuncs.com'
OSS_INTERNAL_ENDPOINT = f'https://oss-{OSS_REGION}-internal.aliyuncs.com'
CDN_DOMAIN = 'https://media.myapp.com'
STS_ROLE_ARN = os.environ['STS_ROLE_ARN']
AK_ID = os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
AK_SECRET = os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']

# Use internal endpoint when running on ECS in same region (free transfer)
endpoint = OSS_INTERNAL_ENDPOINT if os.environ.get('ON_ECS') else OSS_ENDPOINT

auth = oss2.Auth(AK_ID, AK_SECRET)
bucket = oss2.Bucket(auth, endpoint, OSS_BUCKET_NAME)


ALLOWED_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'application/pdf': '.pdf',
    'video/mp4': '.mp4',
}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@app.route('/api/upload/presign', methods=['POST'])
def get_upload_url():
    """Generate a presigned URL for direct browser-to-OSS upload."""
    data = request.json
    content_type = data.get('content_type')
    filename = data.get('filename', 'unnamed')
    user_id = data.get('user_id')

    if content_type not in ALLOWED_TYPES:
        return jsonify({'error': f'Unsupported file type: {content_type}'}), 400

    # Generate a unique object key
    ext = ALLOWED_TYPES[content_type]
    date_prefix = time.strftime('%Y/%m/%d')
    unique_id = uuid.uuid4().hex[:12]
    object_key = f'uploads/{user_id}/{date_prefix}/{unique_id}{ext}'

    # Generate presigned PUT URL (valid for 10 minutes)
    upload_url = bucket.sign_url(
        'PUT',
        object_key,
        600,
        headers={
            'Content-Type': content_type,
            'x-oss-forbid-overwrite': 'true'
        }
    )

    # CDN URL for accessing the file after upload
    cdn_url = f'{CDN_DOMAIN}/{object_key}'

    return jsonify({
        'upload_url': upload_url,
        'object_key': object_key,
        'cdn_url': cdn_url,
        'expires_in': 600
    })


@app.route('/api/image/<path:object_key>')
def get_image_url(object_key):
    """Return CDN URL with optional image processing parameters."""
    width = request.args.get('w', type=int)
    height = request.args.get('h', type=int)
    fmt = request.args.get('format', 'webp')
    quality = request.args.get('q', 80, type=int)

    url = f'{CDN_DOMAIN}/{object_key}'

    # Build image processing parameters
    processes = []
    if width and height:
        processes.append(f'image/resize,m_fill,w_{width},h_{height}')
    elif width:
        processes.append(f'image/resize,w_{width}')
    elif height:
        processes.append(f'image/resize,h_{height}')

    if fmt:
        processes.append(f'format,{fmt}')
    if quality and quality < 100:
        processes.append(f'quality,q_{quality}')

    if processes:
        process_string = '/'.join(processes)
        # Ensure "image/" prefix is only on the first operation
        if not process_string.startswith('image/'):
            process_string = 'image/' + process_string
        url += f'?x-oss-process={process_string}'

    return jsonify({'url': url})


@app.route('/api/upload/sts-token', methods=['POST'])
def get_sts_token():
    """Issue STS temporary credentials for mobile/SPA uploads."""
    user_id = request.json.get('user_id')

    sts_config = Config(
        access_key_id=AK_ID,
        access_key_secret=AK_SECRET,
        endpoint=f'sts.{OSS_REGION}.aliyuncs.com'
    )
    sts_client = StsClient(sts_config)

    # Scope credentials to this user's upload directory only
    policy = f'''{{
        "Version": "1",
        "Statement": [{{
            "Effect": "Allow",
            "Action": ["oss:PutObject"],
            "Resource": [
                "acs:oss:*:*:{OSS_BUCKET_NAME}/uploads/{user_id}/*"
            ]
        }}]
    }}'''

    resp = sts_client.assume_role(AssumeRoleRequest(
        role_arn=STS_ROLE_ARN,
        role_session_name=f'upload-{user_id}',
        duration_seconds=900,
        policy=policy
    ))

    creds = resp.body.credentials
    return jsonify({
        'access_key_id': creds.access_key_id,
        'access_key_secret': creds.access_key_secret,
        'security_token': creds.security_token,
        'expiration': creds.expiration,
        'bucket': OSS_BUCKET_NAME,
        'region': OSS_REGION,
        'endpoint': OSS_ENDPOINT
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 第五步：全流程测试

```bash
# 1. Request a presigned upload URL
curl -X POST http://localhost:8080/api/upload/presign \
  -H "Content-Type: application/json" \
  -d '{"content_type": "image/jpeg", "filename": "photo.jpg", "user_id": "u123"}'

# Response:
# {
#   "upload_url": "https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/uploads/u123/2026/05/14/a1b2c3d4e5f6.jpg?OSSAccessKeyId=...&Signature=...&Expires=...",
#   "object_key": "uploads/u123/2026/05/14/a1b2c3d4e5f6.jpg",
#   "cdn_url": "https://media.myapp.com/uploads/u123/2026/05/14/a1b2c3d4e5f6.jpg",
#   "expires_in": 600
# }

# 2. Upload the file directly to OSS using the presigned URL
curl -X PUT "<presigned_upload_url>" \
  -H "Content-Type: image/jpeg" \
  -H "x-oss-forbid-overwrite: true" \
  --data-binary @photo.jpg

# 3. Access via CDN with image processing
curl -I "https://media.myapp.com/uploads/u123/2026/05/14/a1b2c3d4e5f6.jpg?x-oss-process=image/resize,w_400/format,webp/quality,q_80"

# 4. Get a processed image URL from the API
curl "http://localhost:8080/api/image/uploads/u123/2026/05/14/a1b2c3d4e5f6.jpg?w=400&format=webp&q=80"
```

### 架构总结

```text
Browser                          Your Flask API                OSS Bucket
  │                                   │                           │
  │  1. POST /api/upload/presign      │                           │
  │──────────────────────────────────►│                           │
  │                                   │  (generates signed URL)   │
  │  2. {upload_url, cdn_url}         │                           │
  │◄──────────────────────────────────│                           │
  │                                                               │
  │  3. PUT (file bytes) ────────────────────────────────────────►│
  │                                                               │
  │  4. GET cdn_url ──────► CDN Edge ──── (cache miss) ─────────►│
  │  ◄── cached response ◄─── CDN ◄──── object data ◄───────────│
  │                                                               │
  │  5. GET cdn_url ──────► CDN Edge                              │
  │  ◄── cached response ◄─── CDN (cache HIT, no OSS request)    │
```

该架构的最大优势在于：应用服务器完全不处理文件 I/O。上传流量由浏览器直连 OSS，下载流量经 CDN 边缘节点分发。Flask API 仅作为协调者，负责生成签名 URL 和构造 CDN 路径，从而保持轻量、易扩展且成本低廉。

## 核心要点

**OSS 不是文件系统。** 它是基于 HTTP 访问的扁平键值存储。切勿将其当作挂载盘使用，也别存储海量小文件（此时 NAS 或数据库更合适）。应专注于其优势场景：以极高持久性存储任意大小的二进制对象，并通过 HTTP 高效分发。

**默认私有，谨慎开放。** 所有 Bucket 初始都应设为私有。临时访问用签名 URL，客户端上传用 STS 令牌，公开内容通过 CDN 回源授权。基础设施中绝不应出现 `public-read-write` ACL。

**生命周期规则等于白捡的钱。** 每个 Bucket 都应配置。哪怕仅设置“30 天后转 IA”，也能为低频数据节省 40% 成本。配置零成本，收益自动化。

**务必使用内网 Endpoint。** 当 ECS 与 OSS 同处一个区域时，使用 `oss-{region}-internal.aliyuncs.com`。内网流量免费，而公网访问每 GB 约 0.12 美元，积少成多不容忽视。

**面向用户的内容，CDN 不是可选项。** 更低延迟、更低成本、内置 DDoS 防护——CDN + OSS 在各方面都优于单独使用 OSS。配置仅需 15 分钟。

**预签名 URL 让服务器保持轻量。** 切勿让应用服务器代理文件上传下载。生成预签名 URL，让客户端直连 OSS（或 CDN）。服务器只处理元数据与授权，不碰字节流。

关于通过基础设施即代码管理 OSS，请参阅 [Terraform Part 5: Storage](/zh/terraform-agents/05-storage-for-agent-memory/)。我们将在 [Part 11: PAI](/zh/aliyun-fullstack/11-pai-ml-platform/) 中使用 OSS 作为机器学习模型的后端存储。

## 接下来聊什么

存储是数据安身立命之所。随着 OSS 配置到位——Bucket、生命周期规则、访问控制、CDN 与图片处理均已就绪——我们的持久化层已稳固可靠。下一篇文章，我们将转向托管数据库：用 RDS 处理关系型数据，用 Redis 提供缓存，并探讨复制、备份与故障转移策略，确保在硬件不可避免地失效时，数据依然安然无恙。
