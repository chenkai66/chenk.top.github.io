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
description: "掌握阿里云OSS：存储桶类型、存储类别、访问控制（ACL、RAM、STS、签名URL）、生命周期规则、跨区域复制、CDN集成、自定义域名。构建完整的媒体存储后端。"
disableNunjucks: true
translationKey: "aliyun-fullstack-4"
---
以前我把用户上传的文件直接塞进 ECS 磁盘里。头像、 PDF 发票、 CSV 导出文件——统统丢在一台跑着 Flask 应用的 `ecs.g7.large` 实例的 `/var/data/uploads/` 目录下。我写了个 cron 任务，每六小时将数据 rsync 到第二台 ECS 上，这被称为“备份”。直到某个周五凌晨 3 点，批处理任务生成了 40GB 无人下载的报表，导致系统盘使用率达到 100%，ECS 实例进入只读状态，应用彻底宕机——而 rsync 从前一天晚上起就再未成功执行。我丢失了过去六小时内所有用户上传的数据，整个周末都在向客户致歉。正是那次故障让我意识到，对象存储不仅是‘锦上添花’的可选项，更是云上架构的基石。应用服务器具有临时性，而数据具有持久性。


这篇文章将从第一性原理讲到生产部署，带你彻底搞定阿里云对象存储 OSS。读完后，你将拥有一个具备生命周期管理、CDN 加速和 Python API 预签名上传功能的媒体存储后端。我们在 [Part 2](/zh/aliyun-fullstack/02-ecs-compute/) 和 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 已经搭好了 VPC 和 ECS 基础——现在加上这个能扛住实例故障、扩展到 PB 级、成本却只有块存储零头的存储层。

## 什么是 OSS？

对象存储服务（Object Storage Service）就是阿里云版的 AWS S3。你将文件（称为“对象”）存入名为“桶”（Bucket）的容器中，每个对象都有唯一的 key（即路径）、数据本身和元数据。这就是全部的数据模型，没有目录、文件层级或 POSIX 语义。当你在 OSS 里看到 `images/2026/05/avatar.png` 时，斜杠只是 key 字符串的一部分，不是目录结构。控制台为便于理解，会将这些 key 渲染为文件夹形式，但底层存储本身是扁平的。

这种简单性正是关键所在。OSS 不需要维护文件系统树，可以透明地将对象分发到成千上万个存储节点上，因此你无需考虑容量规划、磁盘 IOPS 或 RAID 配置。当你通过 PUT 操作上传对象时，OSS 会自动选择存储位置并应用跨可用区复制策略以保障持久性，并在你发起 GET 请求时高效返回数据。标准存储的耐久性保证是 99.9999999999%（12 个 9），这意味着在设计层面，即使存储一百亿个对象，预期丢失数量也不超过一个。

### 三种云存储类型

阿里云提供三种本质不同的存储产品，选错是我见过的最常见错误：

![OSS 存储类型对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_storage_classes.png)

| 存储类型 | 产品 | 访问模式 | 类比 |
|---|---|---|---|
| **块存储** | EBS (云盘) | 挂载到单台 ECS，随机读写 | 插在你电脑上的硬盘 |
| **文件存储** | NAS / CPFS | 通过 NFS/SMB 被多台 ECS 共享 | 办公室里的网络文件共享 |
| **对象存储** | OSS | HTTP API，无需挂载，容量无限 | 带 API 的 Dropbox |

**块存储**（挂载到 ECS 的云盘）给你的是一个原始块设备，操作系统用 ext4 或 xfs 格式化。它快、延迟低、支持随机 I/O——完美适合数据库、系统启动卷以及任何需要 POSIX 文件系统语义的场景。但它一次只能挂载到一台 ECS 实例，且无论是否使用，均需为预分配容量付费。

**文件存储**（NAS）提供一个共享文件系统，多台 ECS 实例可以通过 NFS v3/v4 或 SMB 同时挂载。适合那些需要共享 `/data` 目录的遗留应用、 CMS 系统或开发环境。但它每 GB 成本高，性能取决于你购买的容量层级。

**对象存储**（OSS）适合其他所有场景——而“其他所有”通常占你数据的 90%。静态资源、用户上传、备份、日志、数据湖文件、 ML 训练数据集、视频、音频、文档。只要通过 HTTP 访问，且无需随机修改文件内容（即不依赖字节级修改）， OSS 就是合适的选择。

### OSS 对比 AWS S3

如果你是从 AWS 过来的，映射关系很直接：

| AWS S3 概念 | OSS 对应物 | 备注 |
|---|---|---|
| Bucket | Bucket | 同样的 3-63 字符命名规则 |
| Object | Object | 同样的 key/value/metadata 模型 |
| Region | Region | 同样的区域 scoped bucket 概念 |
| S3 Standard | Standard | 热数据，频繁访问 |
| S3 Standard-IA | Infrequent Access (IA) | 最低存储 30 天 |
| S3 Glacier | Archive | 最低 90 天， 1 分钟恢复 |
| S3 Glacier Deep Archive | Deep Cold Archive | 最低 180 天，数小时恢复 |
| Presigned URL | Signed URL | 概念相同， SDK 方法名不同 |
| Bucket Policy | Bucket Policy | 基于 JSON，语法相似 |
| S3 Lifecycle | Lifecycle Rules | 同样的转换/过期模型 |
| Cross-Region Replication | Cross-Region Replication | 同样的异步复制模型 |
| CloudFront + S3 | CDN + OSS | 原生集成，同样的回源模式 |

主要区别： OSS 用 AccessKey ID/Secret 而不是 AWS Signature V4 （不过 SDK 会处理）。 OSS 端点遵循 `oss-{region}.aliyuncs.com` 模式，而不是 `s3.{region}.amazonaws.com`。而且 OSS 每个区域都有独特的“内网端点”（例如 `oss-cn-beijing-internal.aliyuncs.com`），当同一区域的 ECS 实例访问时流量免费——AWS 对同样流量收费。

### 核心概念

写代码前得搞清楚这四个概念：

**Bucket** -- 对象的全局唯一容器。 Bucket 名必须 3-63 字符，仅限小写字母、数字和连字符。它是区域 scoped 的——`cn-beijing` 的 bucket 数据就存北京。创建后不能重命名或移动。

**Object** -- 存在 bucket 里的文件，由 key （路径字符串）标识。最大对象大小 48.8 TB。对象是不可变的——更新时你替换整个对象，不能原地修改字节。

**Region 和 Endpoint** -- 每个 bucket  lives 在一个区域。通过公共端点（`oss-cn-beijing.aliyuncs.com`）、内网端点（同区域 ECS 免费）或你绑定的自定义域名访问。

**AccessKey** -- API 访问的凭证。生产环境永远别用根账号的 AccessKey。用 RAM 用户或 STS 临时凭证，我们在下面的访问控制部分会讲。

## 存储类型

OSS 有五种存储类型，选对能省 80% 账单，选错能 inflate 10 倍。核心权衡原则：存储成本越低，取回成本越高、延迟越大。

| 存储类型 | $/GB/月 | 最低存储时长 | 取回成本 | 恢复时间 | 最佳场景 |
|---|---|---|---|---|---|
| **Standard** | ~0.020 | 无 | 免费 | 即时 | 热数据，频繁访问文件 |
| **Infrequent Access (IA)** | ~0.012 | 30 天 | ~0.010/GB | 即时 | 每月访问 < 1-2 次的数据 |
| **Archive** | ~0.005 | 90 天 | ~0.020/GB | 1 分钟 (Expedited) | 季度报表，旧备份 |
| **Cold Archive** | ~0.002 | 180 天 | ~0.030/GB | 1-10 小时 | 合规归档，法律保留 |
| **Deep Cold Archive** | ~0.001 | 180 天 | ~0.050/GB | 12-48 小时 | 再也不想读的数据 |

*价格为 cn-beijing 近似值。查看 [OSS 定价页](https://www.alibabacloud.com/product/object-storage-service/pricing) 获取当前费率和区域差异。*

有几个坑容易踩：

**最低存储时长是按账单算的，不是按实际存储算。** 如果你上传文件到 Archive 存储， 10 天后删除，你依然要被收 90 天的钱。除了 Standard，所有类型都这样。

**取回成本是按 GB 算的。** 从 Cold Archive 恢复 1TB 数据，光取回费用就要约 30 美元，这还不算传输费。归档前要想清楚。

**IA 有最小对象大小限制。** 小于 64KB 的对象按 64KB 计费。如果你存几百万个 tiny JSON 文件， IA 会比 Standard 更贵。

**Archive 和 Cold Archive 需要恢复步骤。** 你不能直接读对象。你得发起恢复请求，等恢复完成，然后对象在可配置周期内（1-7 天）可读。过后它又回到归档状态。

黄金法则：所有数据先从 Standard 开始，用 OSS 访问日志测 30 天访问模式，然后设生命周期规则自动转换冷数据。别猜。

## 创建和管理 Bucket

### 控制台 walkthrough

![Bucket CRUD 操作](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_bucket_operations.png)

创建第一个 bucket 最快的方法：

1. 打开 [OSS 控制台](https://oss.console.aliyun.com/)
2. 点击 **创建 Bucket**
3. 设置 bucket 名称（全局唯一，例如 `myapp-prod-media-cn`）
4. 选择区域（例如 `cn-beijing`）
5. 存储类型： Standard （later 通过生命周期规则改）
6. 访问控制：**私有**（永远从私有开始）
7. 版本控制：启用（以后可以暂停，但 retroactively 启用不会给现有对象版本化）
8. 服务器端加密： AES-256 或 KMS （我建议大多数 workload 用 AES-256——免费且透明）
9. 点击 **确定**

###  CLI 使用 ossutil

`ossutil` 是 OSS 命令行工具。先安装：

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

现在创建 bucket 并开始操作对象：

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

### Bucket 命名规则

- 3-63 字符
- 仅限小写字母、数字、连字符
- 必须以字母或数字开头和结尾
- 在整个阿里云全局唯一（不只是你的账号内）
- 创建后不能重命名

我用 `{app}-{env}-{purpose}-{region-short}` 模式——例如 `myapp-prod-media-cn`、`myapp-staging-logs-cn`。这能防止命名冲突，当你凌晨 2 点盯着 30 个 bucket 列表时，一眼就能看出每个是干嘛的。

### 版本控制

版本控制保留每个对象的每个版本。当你覆盖 `report.pdf` 时，旧版本不会被删——它变成非当前版本。当你删除 `report.pdf` 时，它会打个删除标记，但数据还在。

```bash
# Enable versioning
ossutil bucket-versioning --method put oss://myapp-prod-media enabled

# Check versioning status
ossutil bucket-versioning --method get oss://myapp-prod-media

# List all versions of objects
ossutil ls oss://myapp-prod-media/ --all-versions
```

任何包含用户数据的 bucket 都必须开版本控制。存储成本会翻倍（因为留着旧版本），但替代方案—— accidental overwrite 导致数据永久丢失——更糟。配合生命周期规则， 30 天后自动删除非当前版本，这样能控制成本。
## 访问控制深度解析

OSS 的访问控制分四层，搞清楚它们怎么交互，是安全系统和数据裸奔的区别。优先级从高到低： STS/RAM 策略 > Bucket 策略 > Bucket ACL。

![OSS 访问控制模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_access_control.png)

### 第一层： Bucket ACL

最简单也是最粗粒度的控制。就三个选项：

| ACL | 匿名读取 | 匿名写入 | 使用场景 |
|---|---|---|---|
| **private** | 否 | 否 | 默认。适用于大多数情况。 |
| **public-read** | 是 | 否 | 静态网站，公共 CDN 源 |
| **public-read-write** | 是 | 是 | **不要使用此设置。** |

```bash
# Set bucket ACL
ossutil bucket-acl --method put oss://myapp-prod-media private

# Check bucket ACL
ossutil bucket-acl --method get oss://myapp-prod-media
```

我没吓唬你，`public-read-write` 真的不能用。把 Bucket 设成这个，意味着互联网上任何人都能往你 Bucket 里随便传文件，你的存储账单会爆炸，甚至变成 malware 分发点。生产环境我真见过这么干的，别学。

`public-read` 只适合直接从 OSS 托管静态资源（不用 CDN）且想要最简单配置的场景。即便如此，我更推荐 `private` 配合 CDN 的 Origin Access Identity——后面会细说。

### 第二层： Bucket Policy

Bucket Policy 是挂在 Bucket 上的 JSON 文档，定义谁能干什么。这是基于资源的策略，跟 S3 Bucket Policy 类似。如果要搞跨账号访问或者细粒度权限，又不想动 RAM，这是推荐做法。

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

这策略的意思是：“允许阿里云账号 `203917385849****` 读取 `shared/` 前缀下的对象，但限制来源 IP 必须是 `203.0.113.0/24`。”你可以限制 IP、 VPC、时间段、 Referer 头，或者强制要求 HTTPS。

用 CLI 应用策略：

```bash
ossutil bucket-policy --method put oss://myapp-prod-media ./bucket-policy.json
```

### 第三层： RAM Policy

RAM 策略是基于身份的——挂在 RAM 用户、用户组或角色上。你的应用服务器主要用这个。

给你的应用创建一个 RAM 用户，只给最小必要权限：

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

这里需要配两个 Resource： Bucket 本身（用于 `ListObjects`）和 `bucket/*`（用于对象操作）。很多人配漏了第一个，结果报 ListBuckets 权限错误。

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

### 第四层： STS 临时凭证

STS 发的是临时凭证，过期时间可配（15 分钟到 1 小时）。浏览器上传和移动端 App 都用这个——千万别把长期 AccessKey 写死在客户端代码里。

流程如下：

1. 客户端向后端请求一个上传 token
2. 后端调用 STS `AssumeRole`，带上缩小范围的 policy
3. STS 返回临时 AccessKeyId、 AccessKeySecret 和 SecurityToken
4. 客户端用这些凭证直传 OSS
5. 凭证自动过期

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

关键点在于：`AssumeRole` 里的 `policy` 参数会进一步限制角色的权限。哪怕角色本身有 OSS 全权，临时凭证也只能在特定路径下 `PutObject`。这就是纵深防御。

### 签名 URL

偶尔分享个文件或者搞个限时下载，生成个带过期的签名 URL 就行：

```bash
# Generate a signed URL valid for 1 hour (3600 seconds)
ossutil sign oss://myapp-prod-media/reports/q1-2026.pdf --timeout 3600
```

生成的 URL 里嵌了签名参数。拿到 URL 的人能在过期前下载文件，客户端无需额外认证。

Python 里这么写：

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

5GB 以下的文件，直接 PUT 就行：

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

超过 100MB 的文件，走分片上传。文件被切分成片（每片最小 100KB，最后一片除外），并行上传，服务端组装。如果某片失败，只重传那一片，不用重传整个文件。

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

底层逻辑是 `resumable_upload` 自动处理：

1. 调用 `InitiateMultipartUpload` 获取 upload ID
2. 切分文件
3. 并行调用 `UploadPart` 上传每片
4. 调用 `CompleteMultipartUpload` 组装对象
5. 本地存 checkpoint 文件，中断后可续传

### 断点续传下载

网络不稳下载大文件时用：

```python
oss2.resumable_download(
    bucket,
    'videos/presentation.mp4',
    '/tmp/downloaded-presentation.mp4',
    part_size=10 * 1024 * 1024,
    num_threads=4
)
```

### 用 ossutil 搞批量操作

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

`--delete` 标志会删掉 OSS 上本地没有的文件。这玩意儿危险，先不带参数跑一遍试试。

### 浏览器端直传 Presigned URL

面向用户的应用最常用的模式：服务端生成 presigned PUT URL，发给浏览器，浏览器直传 OSS。服务器根本不碰文件流。

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

这样就不用让应用服务器代理上传流量了，否则带宽和内存消耗跟文件大小成正比。用了 presigned URL，浏览器直连 OSS，服务器只负责协调。
## 生命周期规则

生命周期规则能自动切换存储类型和处理对象过期。真正的省钱大招都在这儿。配好一次，后面基本可以忘掉了。

![存储生命周期转换时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_lifecycle_rules.png)

### 常见模式

**模式 1：渐进式归档**

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

这条规则挂在 `logs/` 前缀下，逻辑很简单：
- 30 天后，转低频访问（省 ~40%）
- 90 天后，转归档存储（省 ~75%）
- 365 天后，转冷归档（省 ~90%）
- 730 天（2 年）后，直接删除

**模式 2：清理未完成的多部分上传**

未完成的多部分上传会占空间但在 `ls` 里看不见。它们会静默积累。这条规则必须加：

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

**模式 3：删除旧版本**

开了版本控制后，非当前版本会堆叠。定期清理：

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

### 通过 CLI 应用生命周期规则

```bash
# 从 JSON 文件应用生命周期配置
ossutil lifecycle --method put oss://myapp-prod-media ./lifecycle.json

# 查看当前生命周期规则
ossutil lifecycle --method get oss://myapp-prod-media
```

### 成本影响

这是我管理的一个生产 bucket 的真实数据。 2 TB 日志数据，每月增长 ~50 GB：

| Strategy | Monthly cost | Annual cost |
|---|---|---|
| All Standard, no lifecycle | ~$40 | ~$480 |
| Lifecycle: IA at 30d, Archive at 90d | ~$18 | ~$216 |
| 生命周期：30天后转为IA，90天后归档，365天后删除 | ~$14 | ~$168 |

单个 JSON 文件就能让成本降低 65%。如果组织内有 20 个 bucket，十分钟的工作量一年能省几千刀。

## 跨区域复制 (CRR)

跨区域复制会把对象异步拷贝到另一个 Region 的目标 bucket。主要两个场景：

![跨区域复制拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_crr_topology.png)

1. **灾难恢复** -- 如果 cn-beijing 发生 Region 级故障，你的数据在 cn-shanghai 还有份
2. **合规** -- 监管要求必须在特定地理位置存储副本

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

| Aspect | Details |
|---|---|
| **复制延迟** | 大多数对象通常小于10分钟，大对象可能更长 |
| **复制内容** | 对象数据、元数据、ACL（可选） |
| **不复制内容** | 生命周期转换、存储桶策略、服务器端加密设置 |
| **成本** | 你需支付目标存储费用及跨区域数据传输费用 |
| **方向** | 默认单向。双向需要设置两条规则。 |
| **删除复制** | 可选。可以选择是否传播删除操作。 |

得提醒一句： CRR 是最终一致性，复制时间没有 SLA 保障。别把它当成实时同步机制。如果需要跨 Region 实时访问，去看 CEN + 多 Region 部署。

## CDN 集成

阿里云 CDN + OSS 是最常见的生产模式。 CDN 边缘节点把对象缓存到离用户最近的地方，延迟从几百毫秒降到个位数。只有缓存 miss 才会回源到你的 OSS bucket。

![OSS CDN 集成数据流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_cdn_flow.png)

### 为什么用 CDN + OSS 而不是直接用 OSS？

| Factor | OSS direct | CDN + OSS |
|---|---|---|
| **延迟** | 50-200ms（根据用户位置而异） | 5-30ms（从最近的边缘节点） |
| **每GB传输成本** | ~0.12/GB（互联网） | ~0.04/GB（CDN对高流量更便宜） |
| **DDoS防护** | 基本 | CDN边缘内置 |
| **HTTPS** | 支持 | 通过CDN提供免费证书 |
| **缓存控制** | 无 | 可配置TTL，缓存清除API |
| **自定义域名** | 支持但无免费HTTPS | 完全自定义域名+免费HTTPS |

只要是面向用户的内容（图片、 CSS、 JS、视频、下载）， CDN 绝对更优。唯一不用 CDN 的情况是私有 API 访问（比如后端服务程序化读取文件）。

### 完整的 CDN + OSS setup

#### 步骤 1：添加 CDN 域名

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

#### 步骤 2：配置 CNAME DNS

添加 CDN 域名后，阿里云会给你一个 CNAME 值，比如 `cdn.example.com.w.kunlunsl.com`。在你的 DNS 里加一条 CNAME 记录：

```
cdn.example.com  CNAME  cdn.example.com.w.kunlunsl.com
```

#### 步骤 3：启用 HTTPS 免费证书

```bash
aliyun cdn SetDomainServerCertificate \
  --DomainName cdn.example.com \
  --ServerCertificateStatus on \
  --CertType free
```

阿里云 CDN 提供免费的 DV (Domain Validated) 证书。自动续期。生产环境你可以上传自己的证书或用证书管理服务。

#### 步骤 4：设置缓存规则

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

#### 步骤 5：配置回源

OSS 做 CDN 源站默认就能用，但建议配这些优化：

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

这样 CDN 就能访问私有 bucket 而不需要把 bucket 公开。 CDN 通过内部授权机制认证 OSS。你的 bucket 保持私有，但 CDN 可以在缓存 miss 时获取对象。

#### 步骤 6：验证 setup

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

当你更新了 OSS 里的文件但 CDN 还在服旧版本时：

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

## 图片处理 (IMM)

OSS 内置了图片处理功能，通过 URL 参数实时转换图片。不用单独起服务，也不用预处理流水线，直接在对象 URL 后追加查询参数就行。

![通过 IMM 的图片处理流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/04-oss-storage/04_image_processing.png)

### 基础转换

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

### 水印

```bash
# 文字水印
?x-oss-process=image/watermark,text_Q2hlbmsgQmxvZw==,type_d3F5LXplbmhlaQ,size_30,color_FFFFFF,t_80,g_se,x_10,y_10

# 文字是 base64 编码。"Chenk Blog" = Q2hlbmsgQmxvZw==
# g_se = southeast (右下角)
# t_80 = 80% 透明度
```

### 图片信息

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

### 结合 CDN 使用图片处理

当你访问 `https://cdn.example.com/images/photo.jpg?x-oss-process=image/resize,w_800/format,webp`， CDN 会缓存处理后的版本。后续请求同样的转换会命中 CDN 缓存，不再回源 OSS。这意味着你既能实时处理，又能享受 CDN 速度的分发。

处理后的图片和原图分开缓存，完整 URL 包括查询参数才是缓存 key。所以 `photo.jpg`、`photo.jpg?x-oss-process=image/resize,w_800` 和 `photo.jpg?x-oss-process=image/resize,w_400` 是三个独立的缓存条目。
## 解决方案：媒体存储后端

咱们把前面聊的都串起来。这次我们要落地一个完整的媒体存储后端：带生命周期规则的 OSS Bucket、配了自定义域名的 CDN，还有一个 Python Flask API。它负责生成预签名上传 URL，并通过 CDN 提供带处理能力的图片服务。

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

### 第二步：配置生命周期规则

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

### 第四步： Flask API 服务

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

```
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

这套架构最爽的地方在于：应用服务器完全不碰文件 I/O。上传流量直接从浏览器打到 OSS。下载流量走 CDN 边缘节点。 Flask API 只是个协调员，生成签名 URL、拼拼 CDN 路径。它保持轻量，容易扩展，跑起来也省钱。

## 核心要点

**OSS 不是文件系统。** 它是基于 HTTP 访问的扁平键值存储。别把它当挂载盘用。别存几百万个小文件，那种场景 NAS 或数据库更合适。用它擅长的场景：存任意大小的 Blob，极致耐用， HTTP 分发。

**默认私有，谨慎开放。** 每个 Bucket 默认都该是私有的。临时访问用签名 URL，客户端上传用 STS Token，公开内容走 CDN 回源。基础设施里永远不该出现 `public-read-write` 这种 ACL。

**生命周期规则就是白捡的钱。** 每个 Bucket 都配上。哪怕只配一条“30 天转 IA”，你不常读的数据能省 40%。配置不要钱，自动运行。

**务必用内网 Endpoint。** ECS 和 OSS 在同地域时，用 `oss-{region}-internal.aliyuncs.com`。内网流量免费。走公网 Endpoint 要付 ~$0.12/GB。这钱积少成多。

**面向用户的内容， CDN 不是可选项。** 更低延迟、更低成本、自带 DDoS 防护， CDN + OSS 绝对比单用 OSS 强。配置也就 15 分钟。

**预签名 URL 让服务器保持轻量。** 别让应用服务器代理文件上传下载。生成预签名 URL，让客户端直连 OSS （或 CDN）。服务器只管元数据和权限，别碰字节流。

关于用基础设施即代码管理 OSS，可以参考 [Terraform Part 5: Storage](/zh/terraform-agents/05-storage-for-agent-memory/)。我们会在 [Part 11: PAI](/zh/aliyun-fullstack/11-pai-ml-platform/) 中把 OSS 用作 ML 模型的后端存储。
## 接下来聊什么

存储就是数据安家的地方。 OSS 配置妥当——Bucket、生命周期规则、访问控制、 CDN 和图片处理都就位——持久化层这块算是稳了。下一篇文章，咱们转向托管数据库：关系型数据交给 RDS，缓存交给 Redis，顺便聊聊复制、备份和故障转移策略。毕竟硬件总会出问题，得确保数据能活下来。