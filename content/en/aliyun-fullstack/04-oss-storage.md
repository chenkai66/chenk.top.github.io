---
title: "Alibaba Cloud Full Stack (4): OSS — Object Storage Done Right"
date: 2026-05-01 09:00:00
tags:
  - Alibaba Cloud
  - OSS
  - Storage
  - CDN
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 4
description: "Master Alibaba Cloud OSS: bucket types, storage classes, access control (ACL, RAM, STS, signed URLs), lifecycle rules, cross-region replication, CDN integration, and custom domains. Build a complete media storage backend."
disableNunjucks: true
translationKey: "aliyun-fullstack-4"
---

I used to store user uploads on the ECS disk. Profile pictures, PDF invoices, CSV exports — all dumped into `/var/data/uploads/` on a single `ecs.g7.large` running my Flask app. I had a cron job that rsynced the directory to a second ECS instance every six hours as a "backup." Then one Friday at 3am, the system disk hit 100% because a batch job generated 40GB of reports nobody ever downloaded, the instance went read-only, the app crashed, and the rsync hadn't run since the previous evening. I lost six hours of user uploads and spent the weekend apologizing to customers. That was the week I learned that object storage is not a nice-to-have — it is the foundation of everything you build in the cloud. Your application server is ephemeral. Your data is not.


This article covers Alibaba Cloud's Object Storage Service from first principles through production deployment. By the end, you will have a working media storage backend with lifecycle management, CDN acceleration, and presigned uploads from a Python API. We set up the VPC and ECS foundation in [Part 2](/en/aliyun-fullstack/02-ecs-compute/) and [Part 3](/en/aliyun-fullstack/03-vpc-networking/) — now we add the storage layer that survives instance failures, scales to petabytes, and costs a fraction of block storage.

## What Is OSS?

Object Storage Service is Alibaba Cloud's equivalent of AWS S3. You store files — called "objects" — in containers called "buckets." Each object has a unique key (its path), the data itself, and metadata. That is the entire data model. There are no directories, no file hierarchies, no POSIX semantics. When you see `images/2026/05/avatar.png` in OSS, the slashes are part of the key string, not a directory structure. The console renders them as folders for convenience, but the storage layer is flat.

This simplicity is the point. Because OSS does not need to maintain a filesystem tree, it can distribute objects across thousands of storage nodes transparently. You never think about capacity planning, disk IOPS, or RAID configurations. You PUT an object, and OSS figures out where to store it, how to replicate it across zones for durability, and how to serve it back when you GET it. The durability guarantee is 99.9999999999% (twelve nines) for Standard storage. That is "designed to lose at most one object if you store ten billion."

### Three types of cloud storage

Alibaba Cloud offers three fundamentally different storage products, and using the wrong one is a common mistake:

| Storage type | Product | Access pattern | Analogy |
|---|---|---|---|
| **Block storage** | EBS (Cloud Disks) | Attach to one ECS, random read/write | A hard drive plugged into your computer |
| **File storage** | NAS / CPFS | Shared across multiple ECS via NFS/SMB | A network file share in your office |
| **Object storage** | OSS | HTTP API, no mount, unlimited capacity | Dropbox with an API |

**Block storage** (cloud disks attached to ECS) gives you a raw block device that the OS formats with ext4 or xfs. It is fast, low-latency, and supports random I/O — perfect for databases, OS boot volumes, and anything that needs POSIX filesystem semantics. But it can only be attached to one instance at a time, and you pay for provisioned capacity whether you use it or not.

**File storage** (NAS) provides a shared filesystem that multiple ECS instances can mount simultaneously via NFS v3/v4 or SMB. Great for legacy applications that need a shared `/data` directory, CMS systems, or development environments. But it is expensive per GB and performance depends on the capacity tier you purchase.

**Object storage** (OSS) is for everything else — and "everything else" is usually 90% of your data. Static assets, user uploads, backups, logs, data lake files, ML training datasets, video, audio, documents. If you access it via HTTP and do not need to edit bytes in the middle of the file, OSS is the right answer.

### OSS vs AWS S3

If you are coming from AWS, the mapping is straightforward:

| AWS S3 concept | OSS equivalent | Notes |
|---|---|---|
| Bucket | Bucket | Same 3-63 character naming rules |
| Object | Object | Same key/value/metadata model |
| Region | Region | Same region-scoped bucket concept |
| S3 Standard | Standard | Hot data, frequent access |
| S3 Standard-IA | Infrequent Access (IA) | 30-day minimum storage |
| S3 Glacier | Archive | 90-day minimum, 1-minute restore |
| S3 Glacier Deep Archive | Deep Cold Archive | 180-day minimum, hours to restore |
| Presigned URL | Signed URL | Same concept, different SDK method names |
| Bucket Policy | Bucket Policy | JSON-based, similar syntax |
| S3 Lifecycle | Lifecycle Rules | Same transition/expiration model |
| Cross-Region Replication | Cross-Region Replication | Same async replication model |
| CloudFront + S3 | CDN + OSS | Native integration, same back-to-origin pattern |

The main differences: OSS uses AccessKey ID/Secret instead of AWS Signature V4 (though the SDK handles this). OSS endpoints follow the pattern `oss-{region}.aliyuncs.com` rather than `s3.{region}.amazonaws.com`. And OSS has a unique "internal endpoint" for each region (e.g., `oss-cn-beijing-internal.aliyuncs.com`) that provides free data transfer when accessed from ECS instances in the same region — AWS charges for the same traffic.

### Key concepts

Four things you need to understand before writing any code:

**Bucket** — A globally unique container for objects. Bucket names must be 3-63 characters, lowercase letters, numbers, and hyphens only. They are region-scoped — a bucket in `cn-beijing` stores data in Beijing. You cannot rename or move a bucket after creation.

**Object** — A file stored in a bucket, identified by a key (path string). Maximum object size is 48.8 TB. Objects are immutable — you replace the entire object on update, you cannot modify bytes in place.

**Region and Endpoint** — Each bucket lives in one region. Access it via the public endpoint (`oss-cn-beijing.aliyuncs.com`), the internal endpoint (free from ECS in the same region), or a custom domain you bind.

**AccessKey** — Your credentials for API access. In production, never use your root account AccessKey. Use RAM users or STS temporary credentials, which we cover in the Access Control section below.

## Storage Classes

OSS has five storage classes, and choosing the right one can cut your bill by 80% or inflate it by 10x. The mental model: the cheaper the storage, the more expensive and slower the retrieval.

![OSS storage class comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_storage_classes.png)

| Storage class | $/GB/month | Minimum duration | Retrieval cost | Restore time | Best for |
|---|---|---|---|---|---|
| **Standard** | ~0.020 | None | Free | Instant | Hot data, frequently accessed files |
| **Infrequent Access (IA)** | ~0.012 | 30 days | ~0.010/GB | Instant | Data accessed < 1-2x per month |
| **Archive** | ~0.005 | 90 days | ~0.020/GB | 1 minute (Expedited) | Quarterly reports, old backups |
| **Cold Archive** | ~0.002 | 180 days | ~0.030/GB | 1-10 hours | Compliance archives, legal hold |
| **Deep Cold Archive** | ~0.001 | 180 days | ~0.050/GB | 12-48 hours | Data you never want to read again |

*Prices are approximate for cn-beijing. Check the [OSS Pricing Page](https://www.alibabacloud.com/product/object-storage-service/pricing) for current rates and regional variations.*

A few things that trip people up:

**Minimum storage duration is billed, not stored.** If you upload a file to Archive storage and delete it after 10 days, you are still charged for 90 days. This is true for all classes except Standard.

**Retrieval costs are per-GB.** Restoring 1TB from Cold Archive costs about $30 just for the retrieval, on top of the transfer costs. Think before you archive.

**IA has a minimum object size.** Objects smaller than 64KB are charged as 64KB. If you are storing millions of tiny JSON files, IA will cost more than Standard.

**Archive and Cold Archive require a restore step.** You cannot read the object directly. You issue a restore request, wait for the restore to complete, then the object is readable for a configurable period (1-7 days). After that, it goes back to archived state.

The golden rule: start everything in Standard, measure your access patterns for 30 days using OSS access logging, then set lifecycle rules to auto-transition cold data. Do not guess.

## Creating and Managing Buckets

### Console walkthrough

![Bucket CRUD operations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_bucket_operations.png)

The fastest way to create your first bucket:

1. Open the [OSS Console](https://oss.console.aliyun.com/)
2. Click **Create Bucket**
3. Set the bucket name (globally unique, e.g., `myapp-prod-media-cn`)
4. Select region (e.g., `cn-beijing`)
5. Storage class: Standard (change later via lifecycle rules)
6. Access Control: **Private** (always start private)
7. Versioning: Enable (you can always suspend it later, but enabling retroactively does not version existing objects)
8. Server-Side Encryption: AES-256 or KMS (I recommend AES-256 for most workloads — it is free and transparent)
9. Click **OK**

### CLI with ossutil

`ossutil` is the OSS command-line tool. Install it first:

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

Now create a bucket and start working with objects:

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

### Bucket naming rules

- 3-63 characters
- Lowercase letters, numbers, hyphens only
- Must start and end with a letter or number
- Globally unique across all of Alibaba Cloud (not just your account)
- Cannot be renamed after creation

I use the pattern `{app}-{env}-{purpose}-{region-short}` — e.g., `myapp-prod-media-cn`, `myapp-staging-logs-cn`. This prevents naming collisions and makes it obvious what each bucket is for when you are staring at a list of 30 buckets at 2am.

### Versioning

Versioning keeps every version of every object. When you overwrite `report.pdf`, the old version is not deleted — it becomes a non-current version. When you delete `report.pdf`, it gets a delete marker but the data remains.

```bash
# Enable versioning
ossutil bucket-versioning --method put oss://myapp-prod-media enabled

# Check versioning status
ossutil bucket-versioning --method get oss://myapp-prod-media

# List all versions of objects
ossutil ls oss://myapp-prod-media/ --all-versions
```

Versioning is essential for any bucket containing user data. The storage cost doubles (because you keep old versions), but the alternative — losing data permanently on accidental overwrite — is worse. Combine versioning with lifecycle rules to auto-delete non-current versions after 30 days, which keeps costs controlled.

## Access Control Deep Dive

OSS access control has four layers, and understanding how they interact is the difference between a secure system and a public data breach. They are evaluated from most specific to least specific: STS/RAM policies override bucket policies, which override bucket ACLs.

![OSS access control model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_access_control.png)

### Layer 1: Bucket ACL

The simplest and coarsest control. Three options:

| ACL | Anonymous read | Anonymous write | Use case |
|---|---|---|---|
| **private** | No | No | Default. Almost everything. |
| **public-read** | Yes | No | Static websites, public CDN origin |
| **public-read-write** | Yes | Yes | **Never use this.** |

```bash
# Set bucket ACL
ossutil bucket-acl --method put oss://myapp-prod-media private

# Check bucket ACL
ossutil bucket-acl --method get oss://myapp-prod-media
```

I am not exaggerating about `public-read-write`. Setting a bucket to public-read-write means anyone on the internet can upload arbitrary files to your bucket, run up your storage bill, and use your bucket as a malware distribution point. I have seen this in production. Do not do it.

`public-read` is appropriate only for static assets served directly from OSS (without CDN) where you want the simplest possible setup. Even then, I prefer `private` plus CDN with origin access identity — but we will get to that.

### Layer 2: Bucket Policy

Bucket policies are JSON documents attached to the bucket that define who can do what. They are resource-based policies, similar to S3 bucket policies. This is the recommended way to grant cross-account access or fine-grained permissions without touching RAM.

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

This policy says: "Allow Alibaba Cloud account `203917385849****` to read objects under the `shared/` prefix, but only from the IP range `203.0.113.0/24`." You can restrict by IP, by VPC, by time of day, by referer header, or by whether the request uses HTTPS.

Apply a bucket policy via the CLI:

```bash
ossutil bucket-policy --method put oss://myapp-prod-media ./bucket-policy.json
```

### Layer 3: RAM Policy

RAM (Resource Access Management) policies are identity-based — attached to RAM users, groups, or roles. This is what your application server uses.

Create a RAM user for your application with minimum necessary permissions:

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

Two resources are needed: the bucket itself (for `ListObjects`) and `bucket/*` (for object operations). Missing the first one is a common cause of "Access Denied on ListBuckets."

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

### Layer 4: STS Temporary Credentials

Security Token Service issues temporary credentials that expire after a configurable period (15 minutes to 1 hour). This is what you use for browser-based uploads and mobile apps — never embed long-lived AccessKeys in client code.

The flow:

1. Client requests an upload token from your backend
2. Your backend calls STS `AssumeRole` with a scoped-down policy
3. STS returns temporary AccessKeyId, AccessKeySecret, and SecurityToken
4. Client uses those credentials to upload directly to OSS
5. Credentials expire automatically

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

The critical detail: the `policy` parameter in `AssumeRole` further restricts the role's permissions. Even if the role has full OSS access, the temporary credentials only get `PutObject` on one specific path. This is defense in depth.

### Signed URLs

For one-off sharing or time-limited downloads, generate a signed URL that expires:

```bash
# Generate a signed URL valid for 1 hour (3600 seconds)
ossutil sign oss://myapp-prod-media/reports/q1-2026.pdf --timeout 3600
```

This outputs a URL with the signature embedded as query parameters. Anyone with the URL can download the file until it expires. No authentication needed on the client side.

In Python:

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

## Uploading and Downloading

### Simple upload

For files under 5 GB, a simple PUT request does the job:

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

### Multipart upload

For files larger than 100 MB, use multipart upload. The file is split into parts (minimum 100 KB each, except the last), uploaded in parallel, then assembled server-side. If a part fails, you retry just that part — not the entire file.

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

Under the hood, `resumable_upload` does:

1. Calls `InitiateMultipartUpload` to get an upload ID
2. Splits the file into parts
3. Uploads each part with `UploadPart` (parallelized)
4. Calls `CompleteMultipartUpload` to assemble the object
5. Saves a checkpoint file locally so it can resume if interrupted

### Resumable download

For large downloads on unreliable networks:

```python
oss2.resumable_download(
    bucket,
    'videos/presentation.mp4',
    '/tmp/downloaded-presentation.mp4',
    part_size=10 * 1024 * 1024,
    num_threads=4
)
```

### Using ossutil for bulk operations

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

### Presigned URL upload from the browser

The most common pattern for user-facing applications: generate a presigned PUT URL on the server, send it to the browser, let the browser upload directly to OSS. Your server never touches the file bytes.

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

This saves you from proxying file uploads through your application server, which would consume bandwidth and memory proportional to file size. With presigned URLs, the browser talks directly to OSS, and your server just coordinates.

## Lifecycle Rules

Lifecycle rules automate storage class transitions and object expiration. This is where the real cost savings happen. Set them up once and forget about them.

![Storage lifecycle transition timeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_lifecycle_rules.png)

### Common patterns

**Pattern 1: Progressive archival**

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

This rule, applied to the `logs/` prefix, says:
- After 30 days, move to Infrequent Access (saves ~40%)
- After 90 days, move to Archive (saves ~75%)
- After 365 days, move to Cold Archive (saves ~90%)
- After 730 days (2 years), delete entirely

**Pattern 2: Clean up incomplete multipart uploads**

Incomplete multipart uploads consume storage but are invisible to `ls`. They accumulate silently. Always add this rule:

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

**Pattern 3: Delete old versions**

If versioning is enabled, non-current versions pile up. Prune them:

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

### Apply lifecycle rules via CLI

```bash
# Apply lifecycle configuration from a JSON file
ossutil lifecycle --method put oss://myapp-prod-media ./lifecycle.json

# Check current lifecycle rules
ossutil lifecycle --method get oss://myapp-prod-media
```

### Cost impact

Here is a real example from a production bucket I manage. 2 TB of log data, growing ~50 GB/month:

| Strategy | Monthly cost | Annual cost |
|---|---|---|
| All Standard, no lifecycle | ~$40 | ~$480 |
| Lifecycle: IA at 30d, Archive at 90d | ~$18 | ~$216 |
| Lifecycle: IA at 30d, Archive at 90d, delete at 365d | ~$14 | ~$168 |

That is a 65% reduction by adding a single JSON file. Multiply by 20 buckets across an organization and you are saving thousands of dollars a year for ten minutes of work.

## Cross-Region Replication (CRR)

Cross-Region Replication asynchronously copies objects from a source bucket to a destination bucket in a different region. Two use cases:

![Cross-region replication topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_crr_topology.png)

1. **Disaster recovery** — If cn-beijing has a regional outage, your data exists in cn-shanghai
2. **Compliance** — Regulatory requirements to store copies in specific geographic locations

### Setting up CRR

```bash
# Step 1: Create the destination bucket in another region
ossutil mb oss://myapp-dr-media --region cn-shanghai

# Step 2: Enable CRR via the console or API
# (ossutil does not support CRR configuration directly -- use the console or SDK)
```

Via the SDK:

```python
import oss2
from oss2.models import ReplicationRule

auth = oss2.Auth('<ACCESS_KEY_ID>', '<ACCESS_KEY_SECRET>')
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'myapp-prod-media')

rule = ReplicationRule(
    rule_id='replicate-to-shanghai',
    target_bucket_name='myapp-dr-media',
    target_bucket_location='oss-cn-shanghai',
    target_transfer_type='oss_acc',  # Use transfer acceleration
    prefix_list=['images/', 'documents/'],  # Only replicate these prefixes
    action_list=['ALL'],  # Replicate PUT, DELETE, and AbortMultipartUpload
    is_enable_historical_object_replication=True  # Replicate existing objects
)

bucket.put_bucket_replication(rule)
```

### CRR details

| Aspect | Details |
|---|---|
| **Replication lag** | Usually < 10 minutes for most objects, can be longer for large objects |
| **What is replicated** | Object data, metadata, ACL (optionally) |
| **What is NOT replicated** | Lifecycle transitions, bucket policies, server-side encryption settings |
| **Cost** | You pay for storage in the destination + data transfer between regions |
| **Direction** | One-way by default. For bidirectional, set up two rules. |
| **Delete replication** | Optional. You can choose whether deletes propagate. |

A warning: CRR is eventual consistency with no SLA on replication time. Do not use it as a real-time sync mechanism. If you need synchronous cross-region access, look at CEN + multi-region deployment instead.

## CDN Integration

Alibaba Cloud CDN + OSS is one of the most common production patterns. CDN edge nodes cache your OSS objects close to users, reducing latency from hundreds of milliseconds to single digits. The origin (your OSS bucket) only gets hit on cache misses.

![OSS CDN integration data flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_cdn_flow.png)

### Why CDN + OSS instead of just OSS?

| Factor | OSS direct | CDN + OSS |
|---|---|---|
| **Latency** | 50-200ms (varies by user location) | 5-30ms (from nearest edge) |
| **Cost per GB transfer** | ~0.12/GB (internet) | ~0.04/GB (CDN is cheaper for high volume) |
| **DDoS protection** | Basic | Built-in at the CDN edge |
| **HTTPS** | Supported | Free certificate via CDN |
| **Cache control** | None | Configurable TTL, cache purge API |
| **Custom domain** | Supported but no free HTTPS | Full custom domain + free HTTPS |

For any bucket serving user-facing content (images, CSS, JS, videos, downloads), CDN is strictly better. The only case where you would not use CDN is for private, API-only access (e.g., backend services reading files programmatically).

### Complete CDN + OSS setup

#### Step 1: Add a CDN domain

```bash
# Using aliyun CLI
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

#### Step 2: Configure CNAME DNS

After adding the CDN domain, Alibaba Cloud gives you a CNAME value like `cdn.example.com.w.kunlunsl.com`. Add a CNAME record in your DNS:

```
cdn.example.com  CNAME  cdn.example.com.w.kunlunsl.com
```

#### Step 3: Enable HTTPS with a free certificate

```bash
aliyun cdn SetDomainServerCertificate \
  --DomainName cdn.example.com \
  --ServerCertificateStatus on \
  --CertType free
```

Alibaba Cloud CDN provides free DV (Domain Validated) certificates. They auto-renew. For production, you can upload your own certificate or use Certificate Management Service.

#### Step 4: Set cache rules

```bash
# Cache images for 30 days
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

#### Step 5: Configure back-to-origin

OSS as CDN origin works automatically, but configure these optimizations:

```bash
# Enable OSS private bucket access from CDN
aliyun cdn BatchSetCdnDomainConfig \
  --DomainNames cdn.example.com \
  --Functions '[{
    "functionName": "l2_oss_key",
    "functionArgs": [{
      "argName": "private_oss_auth", "argValue": "on"
    }]
  }]'
```

This lets CDN access a private bucket without making the bucket public. CDN authenticates to OSS using an internal authorization mechanism. Your bucket stays private, but CDN can fetch objects on cache misses.

#### Step 6: Verify the setup

```bash
# Test CDN resolution
dig cdn.example.com

# Test content delivery
curl -I https://cdn.example.com/images/test.jpg

# Check response headers -- look for these:
# X-Cache: HIT or MISS (CDN cache status)
# Via: S.mix... (CDN edge node identifier)
# Age: 3600 (seconds since cached)
```

### Cache purge

When you update a file in OSS but CDN still serves the old version:

```bash
# Purge a specific URL
aliyun cdn RefreshObjectCaches \
  --ObjectPath "https://cdn.example.com/images/logo.png" \
  --ObjectType File

# Purge an entire directory
aliyun cdn RefreshObjectCaches \
  --ObjectPath "https://cdn.example.com/static/" \
  --ObjectType Directory
```

## Image Processing (IMM)

OSS has built-in image processing that transforms images on the fly via URL parameters. No separate service, no pre-processing pipeline — just append query parameters to the object URL.

![Image processing pipeline via IMM](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/04-oss-storage/04_image_processing.png)

### Basic transformations

```bash
# Original image
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg

# Resize to 800px width, maintain aspect ratio
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,w_800

# Resize to 200x200 with center crop
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,m_fill,w_200,h_200

# Convert to WebP format (saves ~30% bandwidth)
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/format,webp

# Quality reduction (80%)
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/quality,q_80

# Chain operations: resize + webp + quality
https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/resize,w_800/format,webp/quality,q_80
```

### Watermarking

```bash
# Text watermark
?x-oss-process=image/watermark,text_Q2hlbmsgQmxvZw==,type_d3F5LXplbmhlaQ,size_30,color_FFFFFF,t_80,g_se,x_10,y_10

# The text is base64 encoded. "Chenk Blog" = Q2hlbmsgQmxvZw==
# g_se = southeast (bottom-right corner)
# t_80 = 80% transparency
```

### Image info

```bash
# Get image dimensions, format, file size
curl "https://myapp-prod-media.oss-cn-beijing.aliyuncs.com/images/photo.jpg?x-oss-process=image/info"

# Response:
# {
#   "FileSize": {"value": "2458632"},
#   "Format": {"value": "jpg"},
#   "ImageHeight": {"value": "2048"},
#   "ImageWidth": {"value": "3072"}
# }
```

### Using image processing with CDN

When you access `https://cdn.example.com/images/photo.jpg?x-oss-process=image/resize,w_800/format,webp`, CDN caches the processed version. Subsequent requests for the same transformation hit the CDN cache, not OSS. This means you get on-the-fly processing with CDN-speed delivery.

The processed images are cached separately from the originals — the full URL including query parameters is the cache key. So `photo.jpg`, `photo.jpg?x-oss-process=image/resize,w_800`, and `photo.jpg?x-oss-process=image/resize,w_400` are three separate cache entries.

## Solution: Media Storage Backend

Let us put everything together. We will build a complete media storage backend: OSS bucket with lifecycle rules, CDN with a custom domain, and a Python Flask API that generates presigned upload URLs and serves images through CDN with processing.

### Step 1: Create and configure the bucket

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

### Step 2: Apply lifecycle rules

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

### Step 3: Set up CDN

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

### Step 4: The Flask API

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

### Step 5: Test the complete flow

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

### Architecture summary

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

The beauty of this architecture: your application server handles zero file I/O. Upload bytes flow directly from the browser to OSS. Download bytes flow from CDN edge nodes. Your Flask API is just a coordinator that generates signed URLs and constructs CDN paths. It stays lightweight, easy to scale, and cheap to run.

## Key Takeaways

**OSS is not a filesystem.** It is a flat key-value store accessed over HTTP. Do not try to use it like a mounted disk. Do not store millions of tiny files where NAS or a database would be better. Use it for what it excels at: storing blobs of any size with extreme durability, served over HTTP.

**Start private, loosen carefully.** Every bucket should be private by default. Use signed URLs for temporary access, STS tokens for client uploads, and CDN with origin access for public content. The `public-read-write` ACL should never appear in your infrastructure.

**Lifecycle rules are free money.** Set them on every bucket. Even a simple "transition to IA after 30 days" rule saves 40% on data you are not actively reading. The rule costs nothing to configure and runs automatically.

**Use the internal endpoint.** When your ECS instances and OSS bucket are in the same region, use `oss-{region}-internal.aliyuncs.com`. Data transfer over the internal network is free. Over the public endpoint, you pay ~$0.12/GB. This adds up fast.

**CDN is not optional for user-facing content.** The combination of lower latency, lower cost, and built-in DDoS protection makes CDN + OSS strictly better than OSS alone for any public content. The setup takes 15 minutes.

**Presigned URLs keep your server thin.** Never proxy file uploads or downloads through your application server. Generate presigned URLs and let the client talk directly to OSS (or CDN). Your server handles metadata and authorization, not bytes.

For using OSS with infrastructure-as-code, see [Terraform Part 5: Storage](/en/terraform-agents/05-storage-for-agent-memory/). We will use OSS as the backing store for our ML models in [Part 11: PAI](/en/aliyun-fullstack/11-pai-ml-platform/).

## What's Next

Storage is where your data lives. With OSS configured — buckets, lifecycle rules, access control, CDN, and image processing in place — we have the persistence layer sorted. In the next article, we move to managed databases: RDS for relational data, Redis for caching, and the replication, backup, and failover strategies that keep your data alive when hardware inevitably fails.
