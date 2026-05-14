---
title: "Alibaba Cloud Full Stack (1): The Ecosystem Map — What Alibaba Cloud Actually Is"
date: 2026-04-28 09:00:00
tags:
  - Alibaba Cloud
  - Cloud Computing
  - ECS
  - Getting Started
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 1
description: "A no-BS guide to Alibaba Cloud's product ecosystem. We map every major service to its AWS/Azure/GCP equivalent, set up an account from scratch, and deploy our first ECS instance — all in under an hour."
disableNunjucks: true
translationKey: "aliyun-fullstack-1"
---

I spent my first week on Alibaba Cloud completely lost in a sea of product names. ECS, SLB, SLS, RDS, OSS, NAS, PAI, ARMS, ACK, FC, CDN, WAF, RAM, KMS, ROS, CloudMonitor, EventBridge, PolarDB, Lindorm, AnalyticDB, MaxCompute, DataWorks, Flink, DashScope, Bailian, OpenSearch... Every console page links to three more products I haven't heard of. The documentation assumes you already know what everything is. The English translations are sometimes literal, sometimes creative, and occasionally missing. This is the guide I wish someone had handed me before I burned my first weekend clicking through consoles and reading translated docs that explained feature flags without ever explaining what the product does.

This article is the lay of the land. We are going to map the entire Alibaba Cloud ecosystem to the AWS/Azure/GCP services you probably already know, set up an account from scratch, understand the billing model well enough to not get surprised, and deploy a working ECS instance by the end. No theory for the sake of theory. Everything here is something I use in production or consciously chose not to use after evaluating it.


---

## Why Alibaba Cloud?

The first question anyone coming from AWS asks is: why would I use Alibaba Cloud at all?

![Alibaba Cloud product family tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_product_families.png)

Three reasons, in descending order of how often they actually matter:

**1. China.** If your users are in China, you need Chinese infrastructure. AWS China (operated by Sinnet/NWCD) exists but is a separate entity with a separate account, a subset of services, and a slower release cycle. Azure China (operated by 21Vianet) has the same limitations. GCP has no mainland China regions at all. Alibaba Cloud is the native cloud — it runs Taobao, Alipay, Tmall, and Ele.me. It has seven mainland regions, the largest data center footprint in Asia, and every service is available on day one without an ICP-licensed partner arrangement.

**2. Market position.** Alibaba Cloud is the largest cloud provider in Asia-Pacific and the third largest globally behind AWS and Azure. According to Gartner's 2025 numbers, it holds roughly 35% of the Chinese cloud market and about 5% globally. This matters for the same reason that being on AWS matters in the West: the ecosystem (third-party integrations, community support, hiring pool, vendor partnerships) follows the market leader.

**3. Unique services.** Some Alibaba Cloud products have no direct equivalent elsewhere, or are meaningfully ahead:

- **DashScope / Bailian** — The Qwen model family (LLM, multimodal, TTS, video generation) is only available natively here. You can self-host Qwen on any cloud, but the managed API with per-token billing and Chinese-language optimization is Alibaba-only.
- **PolarDB** — A cloud-native distributed database that is genuinely different from Aurora. It separates compute from storage at the page level and supports PostgreSQL, MySQL, and a distributed (sharded) mode in the same product.
- **MaxCompute / DataWorks** — The data warehousing and ETL stack that handles Alibaba's own internal analytics. Nothing on AWS matches the integration depth between these two.
- **Alipay / Taobao ecosystem integration** — If you are building anything for Chinese e-commerce (mini-programs, payment flows, merchant tools), the native integrations save months of work.

If none of these apply to you — if your users are in the US, you don't need Chinese language models, and you don't care about the Asian market — then stick with AWS. I am not going to pretend otherwise. But if even one of them applies, Alibaba Cloud is worth learning properly, which is why we are here.

## The Service Map: Alibaba Cloud vs AWS vs Azure vs GCP

The single most useful thing I can do in this article is give you the Rosetta Stone. Every Alibaba Cloud product mapped to the service you already know. I am keeping the descriptions tight — one sentence each — because you don't need a paragraph to understand that "OSS is S3."

![Alibaba Cloud service map compared with AWS, Azure, and GCP](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_service_map.png)

### Compute

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **ECS** (Elastic Compute Service) | EC2 | Virtual Machines | Compute Engine | Virtual machines. The bread and butter. |
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | Serverless functions, event-driven. |
| **ACK** (Container Service for Kubernetes) | EKS | AKS | GKE | Managed Kubernetes. |
| **SAE** (Serverless App Engine) | App Runner | Container Apps | Cloud Run | Serverless containers without managing K8s. |
| **ECI** (Elastic Container Instance) | Fargate | Container Instances | Cloud Run Jobs | Serverless container instances, no cluster. |
| **Batch Compute** | AWS Batch | Azure Batch | Cloud Batch | Managed batch job scheduling. |

### Storage

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **OSS** (Object Storage Service) | S3 | Blob Storage | Cloud Storage | Object storage. Put files in, get files out. |
| **NAS** (Network Attached Storage) | EFS | Azure Files | Filestore | POSIX-compatible shared filesystem. |
| **CPFS** (Cloud Parallel File System) | FSx for Lustre | — | — | High-performance parallel filesystem for HPC/AI. |
| **Block Storage** (EBS equivalent) | EBS | Managed Disks | Persistent Disk | Block storage attached to ECS instances. |
| **Tablestore** | DynamoDB | Table Storage | Bigtable | Wide-column NoSQL, time-series friendly. |

### Database

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **RDS** | RDS | Azure SQL / Azure DB | Cloud SQL | Managed MySQL, PostgreSQL, SQL Server, MariaDB. |
| **PolarDB** | Aurora | — | AlloyDB | Cloud-native distributed relational database. |
| **Lindorm** | DynamoDB + Timestream | Cosmos DB | Bigtable + BigTable | Multi-model database (wide-column, time-series, search). |
| **AnalyticDB (ADB)** | Redshift | Synapse | BigQuery | Cloud data warehouse, columnar, MPP. |
| **Tair** (formerly Redis Enterprise) | ElastiCache | Azure Cache for Redis | Memorystore | Managed Redis-compatible in-memory store. |
| **MongoDB** (ApsaraDB for MongoDB) | DocumentDB | Cosmos DB (Mongo API) | — | Managed MongoDB. |

### Networking

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **VPC** | VPC | Virtual Network | VPC | Virtual private network, the foundation. |
| **SLB** (Server Load Balancer) | ALB / NLB | Load Balancer | Cloud Load Balancing | Layer 4 + Layer 7 load balancing. |
| **ALB** (Application Load Balancer) | ALB | Application Gateway | — | Layer 7 load balancing (newer product). |
| **CDN** | CloudFront | Azure CDN | Cloud CDN | Content delivery network. |
| **DCDN** (Dynamic CDN) | CloudFront + Global Accelerator | Front Door | — | Dynamic content acceleration + edge compute. |
| **CEN** (Cloud Enterprise Network) | Transit Gateway | Virtual WAN | Cloud Interconnect | Multi-VPC, multi-region network backbone. |
| **NAT Gateway** | NAT Gateway | NAT Gateway | Cloud NAT | Outbound internet for private instances. |
| **EIP** (Elastic IP) | Elastic IP | Public IP | Static IP | Static public IP address. |
| **PrivateLink** | PrivateLink | Private Link | Private Service Connect | Private endpoint to services without traversing internet. |

### AI and Machine Learning

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **PAI** (Platform for AI) | SageMaker | Azure ML | Vertex AI | Full ML platform: notebooks, training, serving. |
| **DashScope** | Bedrock | Azure OpenAI Service | Vertex AI (Gemini) | Model API gateway (Qwen, Wanxiang, embeddings). |
| **Bailian (百炼)** | Bedrock console | Azure AI Studio | — | Model management console, RAG builder, fine-tuning UI. |
| **OpenSearch** | OpenSearch Service | Azure AI Search | Vertex AI Search | Managed search with AI ranking and retrieval. |

### Security and Identity

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **RAM** (Resource Access Management) | IAM | Entra ID (Azure AD) | Cloud IAM | Users, roles, policies. |
| **KMS** (Key Management Service) | KMS | Key Vault | Cloud KMS | Encryption key management. |
| **WAF** (Web Application Firewall) | WAF | Azure WAF | Cloud Armor | Web application firewall. |
| **Security Center** | GuardDuty + Inspector | Defender for Cloud | Security Command Center | Threat detection, vulnerability scanning. |
| **SSL Certificates** | ACM | App Service Certificates | Certificate Manager | Managed TLS/SSL certificates. |
| **ActionTrail** | CloudTrail | Activity Log | Cloud Audit Logs | API audit logging. |

### Observability

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **SLS** (Simple Log Service) | CloudWatch Logs | Azure Monitor Logs | Cloud Logging | Log collection, indexing, analysis. |
| **CloudMonitor** | CloudWatch Metrics | Azure Monitor | Cloud Monitoring | Infrastructure metrics and alerting. |
| **ARMS** | X-Ray + CloudWatch APM | Application Insights | Cloud Trace | Application performance monitoring, distributed tracing. |
| **Grafana Service** | Amazon Managed Grafana | Azure Managed Grafana | — | Managed Grafana dashboards. |

### Serverless and Event-Driven

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | Serverless functions. |
| **EventBridge** | EventBridge | Event Grid | Eventarc | Event bus, routing, filtering. |
| **Message Queue (RocketMQ)** | SQS + SNS | Service Bus | Pub/Sub | Distributed message queue (Alibaba's own). |
| **Kafka (ApsaraDB for Kafka)** | MSK | Event Hubs (Kafka) | Managed Kafka | Managed Apache Kafka. |

### Infrastructure as Code

| Alibaba Cloud | AWS | Azure | GCP | What it does |
|---|---|---|---|---|
| **Terraform** (alicloud provider) | Terraform (aws provider) | Terraform (azurerm) | Terraform (google) | HashiCorp Terraform with official provider. |
| **ROS** (Resource Orchestration Service) | CloudFormation | ARM/Bicep | Deployment Manager | Native template-based IaC. |
| **Pulumi** (alicloud provider) | Pulumi (aws) | Pulumi (azure) | Pulumi (gcp) | Programmatic IaC in Python/TS/Go. |

That is a lot of products. You do not need to memorize this table. Bookmark it. Come back when you see an acronym in a doc page and need to know what it maps to. The mental model that helps most is: **Alibaba Cloud has a near-1:1 equivalent for every AWS service**, with a few extras (PolarDB, DashScope, MaxCompute) and a few gaps (no equivalent to AWS Organizations' multi-account strategy is as mature, and the English SDK docs lag behind Chinese ones by weeks to months).

## Account Setup: From Zero to Console

Setting up an Alibaba Cloud account is not like signing up for AWS. There is an extra step that trips up every Western developer the first time: **real-name verification** (实名认证).

### Step 1: Register

Go to [aliyun.com](https://www.alibabacloud.com/) and register. You can use an international phone number and email. For the international site, credit card is accepted. For the Chinese site (aliyun.com), you will need an Alipay account or Chinese bank card.

> **Which site?** If your workloads target Chinese users, use the Chinese site (aliyun.com). If your workloads target international users, use the international site (alibabacloud.com). They are separate accounts, separate billing, and (mostly) separate regions. You can use both, but pick one as your primary. This series assumes the Chinese site because that is where the full product catalog lives.

### Step 2: Real-name verification

China requires real-name verification for all cloud services. This is a regulatory requirement, not an Alibaba policy. AWS China requires it too. The process:

- **Individual account**: Upload a photo of your ID (Chinese ID card for residents, passport for foreigners). Verification takes 1-3 business days.
- **Enterprise account**: Upload business license, legal representative ID, and (sometimes) a bank verification deposit. Takes 3-5 business days.

If you are just learning, an individual account is fine. You can upgrade to enterprise later. The key difference is invoice type (个人 vs 企业发票) and spending limits.

### Step 3: Enable MFA

Do this immediately. The console at `home.console.aliyun.com` holds the keys to your billing, your data, and your infrastructure. Enable MFA on the root account:

1. Go to **Account Management** > **Security Settings**
2. Enable **Virtual MFA Device**
3. Scan the QR code with your authenticator app (Google Authenticator, 1Password, etc.)
4. Enter two consecutive codes to confirm

### Step 4: Create a RAM admin user

Never use the root account for daily work. This is the same advice as AWS — the root account is for billing and emergency recovery. Everything else goes through RAM.

```bash
# Install the Alibaba Cloud CLI first
# macOS
brew install aliyun-cli

# Or download from https://github.com/aliyun/aliyun-cli/releases

# Configure with your root account (temporarily, to create the admin user)
aliyun configure set \
  --profile root \
  --mode AK \
  --access-key-id <YOUR_ROOT_AK_ID> \
  --access-key-secret <YOUR_ROOT_AK_SECRET> \
  --region cn-hangzhou
```

Now create a RAM admin user:

```bash
# Create the RAM user
aliyun ram CreateUser --UserName admin

# Create an access key for the new user
aliyun ram CreateAccessKey --UserName admin

# Attach AdministratorAccess policy
aliyun ram AttachPolicyToUser \
  --PolicyType System \
  --PolicyName AdministratorAccess \
  --UserName admin

# Enable console login for the RAM user
aliyun ram CreateLoginProfile \
  --UserName admin \
  --Password 'YourStrongPassword123!' \
  --MFABindRequired true
```

Save the access key output somewhere safe (a password manager, not a sticky note). Then reconfigure the CLI to use the RAM user:

```bash
aliyun configure set \
  --profile admin \
  --mode AK \
  --access-key-id <RAM_USER_AK_ID> \
  --access-key-secret <RAM_USER_AK_SECRET> \
  --region cn-hangzhou

# Set as default profile
aliyun configure set --profile admin
```

From this point on, never touch the root credentials again unless you are managing billing or recovering from a lockout.

### Step 5: Set up billing alerts

Go to **Billing Management** > **Spending Alerts** and set a budget. I recommend starting with 100 CNY/month for learning purposes. The free tier is generous enough for most experiments, but a runaway ECS instance or a forgotten GPU VM will burn through your budget overnight.

```bash
# Quick check: how much have I spent this month?
aliyun bssopenapi QueryAccountBalance
```

### Step 6: Understand the free tier

Alibaba Cloud's free tier for new accounts (valid for 12 months) includes:

- **ECS**: 1 instance (1 vCPU, 1 GB RAM) for 3 months — enough for this series
- **OSS**: 5 GB storage, 5 GB outbound traffic / month
- **RDS**: 1 micro instance for 1 month
- **Function Compute**: 1 million requests + 400,000 GB-seconds / month
- **SLS**: 500 MB / day ingestion
- **Various AI services**: Free quota per model (DashScope, PAI, etc.)

The exact offerings change quarterly. Check [the free trial page](https://free.aliyun.com/) for current details.

## Regions and Availability Zones

Alibaba Cloud has more China regions than any other provider. This matters because Chinese internet topology means that a user in Shenzhen hitting a server in Beijing will experience noticeably higher latency than a user in Shenzhen hitting a server in Shenzhen — the Great Firewall and the physical distance both contribute.

![Alibaba Cloud global regions and availability zones](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_region_map.png)

### China Regions

| Region ID | Location | Good for |
|---|---|---|
| `cn-hangzhou` | Hangzhou, Zhejiang | Default for most services. Alibaba's HQ. Best service coverage. |
| `cn-shanghai` | Shanghai | Financial services, AI/ML (best GPU stock). |
| `cn-beijing` | Beijing | Government, media, northern China users. |
| `cn-shenzhen` | Shenzhen, Guangdong | Southern China, Hong Kong adjacent. |
| `cn-zhangjiakou` | Zhangjiakou, Hebei | Cold storage, batch processing (cheapest). |
| `cn-hohhot` | Hohhot, Inner Mongolia | Big data, cost-sensitive workloads. |
| `cn-wulanchabu` | Ulanqab, Inner Mongolia | AI training, large GPU clusters. |
| `cn-chengdu` | Chengdu, Sichuan | Western China coverage. |
| `cn-guangzhou` | Guangzhou, Guangdong | Alternative to Shenzhen for southern China. |
| `cn-nanjing` | Nanjing, Jiangsu | Eastern China redundancy. |
| `cn-fuzhou` | Fuzhou, Fujian | Cross-strait (Taiwan adjacent). |
| `cn-heyuan` | Heyuan, Guangdong | Cost-optimized southern China. |

### International Regions

| Region ID | Location | Good for |
|---|---|---|
| `ap-southeast-1` | Singapore | Southeast Asia, lowest latency to China without being in China. |
| `ap-northeast-1` | Tokyo | Japan, Korean users. |
| `ap-south-1` | Mumbai | India, South Asia. |
| `ap-southeast-5` | Jakarta | Indonesia. |
| `ap-southeast-3` | Kuala Lumpur | Malaysia. |
| `eu-central-1` | Frankfurt | Europe. |
| `eu-west-1` | London | UK. |
| `us-east-1` | Virginia | US East Coast. |
| `us-west-1` | Silicon Valley | US West Coast. |
| `ap-southeast-2` | Sydney | Australia, Oceania. |
| `me-east-1` | Dubai | Middle East. |

### How to choose

My decision tree, distilled from real production experience:

1. **Where are your users?** Pick the closest region to them. Latency trumps everything.
2. **Do you need to comply with Chinese data residency laws?** If yes, pick a mainland China region. Data in `cn-*` regions cannot leave China without explicit cross-border data transfer approval.
3. **Do you need GPUs?** Shanghai and Hangzhou have the best GPU stock. Ulanqab has the cheapest GPU pricing but the worst network to end users.
4. **Cost-sensitive batch workloads?** Zhangjiakou and Hohhot are 15-30% cheaper for compute.
5. **Default choice?** `cn-hangzhou` for China, `ap-southeast-1` for international.

> **Real-world tip:** Every region has multiple Availability Zones (AZs), labeled `a`, `b`, `c`, etc. Always deploy across at least two AZs for anything that matters. An AZ outage is rare but it happens — I have seen `cn-hangzhou-h` go degraded twice in 18 months.

## Understanding the Billing Model

Alibaba Cloud billing is more flexible than AWS but also more confusing if you are not used to it. There are four purchasing modes and you need to understand all of them.

![Alibaba Cloud billing model comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_billing_comparison.png)

### Pay-As-You-Go (PAYG, 按量付费)

Exactly what it sounds like. You use it, you pay for it, by the second (for compute) or by the GB (for storage/transfer). This is the default and what you should use while learning. No commitment, no upfront cost, stop anytime.

The downside: it is the most expensive mode per unit. An ECS `ecs.c7.large` (2 vCPU, 4 GB) in `cn-hangzhou` costs roughly:
- PAYG: ~0.35 CNY/hour (~252 CNY/month if running 24/7)
- Subscription: ~125 CNY/month (1-year commitment)

That is a 50% premium for flexibility.

### Subscription (包年包月)

You commit to 1 month, 3 months, 6 months, 1 year, 2 years, or 3 years. The longer the commitment, the deeper the discount. A 3-year ECS commitment can be 40-60% cheaper than PAYG. But you pay upfront (or monthly with auto-renewal), and if you stop using the instance, you still pay.

Use this for: production servers that you know will run for at least a year.

### Preemptible / Spot Instances (抢占式实例)

Same concept as AWS Spot Instances. You bid for unused capacity at 10-90% discount. The instance can be reclaimed with 5 minutes of warning. Alibaba Cloud's spot market is less competitive than AWS's, which means discounts are often deeper and interruptions less frequent — at least in my experience in `cn-shanghai`.

Use this for: batch processing, training jobs that can checkpoint, CI runners.

```bash
# Launch a spot instance
aliyun ecs RunInstances \
  --RegionId cn-hangzhou \
  --InstanceType ecs.c7.large \
  --SpotStrategy SpotAsPriceGo \
  --SpotPriceLimit 0.2 \
  --ImageId aliyun_3_x64_20G_alibase_20240819.vhd \
  --SecurityGroupId sg-bp1xxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxx
```

### Savings Plans and Resource Packs

**Savings Plans** work like AWS Savings Plans — you commit to a certain hourly spend and get a discount on all compute (ECS, ECI, ACK) regardless of instance type or region. More flexible than Subscription but requires you to understand your baseline spend.

**Resource Packs** (资源包) are prepaid bundles for specific services. Buy 500 GB of OSS storage for a year at a 30% discount. Buy 10 million Function Compute invocations at a 40% discount. These are worth it once you know your usage patterns, but do not buy them while learning.

### The billing console

The billing console at `usercenter2.aliyun.com` is where you go to understand your spend. The key pages:

- **Bills** > **Bill Overview**: Monthly summary by product
- **Bills** > **Bill Details**: Line-item detail (instance-level)
- **Cost Management** > **Cost Analysis**: Breakdown by product, region, tag
- **Budget** > **Budget Alert**: Set spending thresholds

I check this weekly. The single biggest cost surprise in my production Alibaba Cloud account was a forgotten NAT Gateway in a test VPC — 15 CNY/day for a resource I wasn't even using. Tags and budget alerts would have caught it on day one.

## Your First ECS Instance

Enough background. Let us deploy something.

![Alibaba Cloud free tier offerings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_free_tier.png)

We are going to launch a single ECS instance running Alibaba Cloud Linux 3, SSH into it, install nginx, and serve a test page. This takes about 15 minutes.

### Prerequisites

- Alibaba Cloud account with real-name verification (done above)
- RAM admin user with access key configured in `aliyun` CLI (done above)
- A region chosen (I will use `cn-hangzhou`)

### Step 1: Create a VPC and VSwitch

Every ECS instance lives in a VPC. You need a VPC and at least one VSwitch (subnet).

```bash
# Create a VPC
VPC_ID=$(aliyun vpc CreateVpc \
  --RegionId cn-hangzhou \
  --CidrBlock 172.16.0.0/16 \
  --VpcName "fullstack-vpc" \
  --Description "VPC for aliyun-fullstack series" \
  | jq -r '.VpcId')

echo "VPC ID: $VPC_ID"

# Wait for VPC to become available
sleep 5

# Create a VSwitch (subnet) in zone cn-hangzhou-h
VSWITCH_ID=$(aliyun vpc CreateVSwitch \
  --RegionId cn-hangzhou \
  --ZoneId cn-hangzhou-h \
  --VpcId "$VPC_ID" \
  --CidrBlock 172.16.1.0/24 \
  --VSwitchName "fullstack-vsw" \
  | jq -r '.VSwitchId')

echo "VSwitch ID: $VSWITCH_ID"
```

### Step 2: Create a Security Group

A security group is a virtual firewall. We want to allow SSH (port 22) from our IP only and HTTP (port 80) from anywhere.

```bash
# Create security group
SG_ID=$(aliyun ecs CreateSecurityGroup \
  --RegionId cn-hangzhou \
  --VpcId "$VPC_ID" \
  --SecurityGroupName "fullstack-sg" \
  --SecurityGroupType normal \
  | jq -r '.SecurityGroupId')

echo "Security Group ID: $SG_ID"

# Allow SSH from your IP only
MY_IP=$(curl -s ifconfig.me)
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-hangzhou \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp \
  --PortRange 22/22 \
  --SourceCidrIp "${MY_IP}/32" \
  --Description "SSH from my IP"

# Allow HTTP from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-hangzhou \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp \
  --PortRange 80/80 \
  --SourceCidrIp "0.0.0.0/0" \
  --Description "HTTP from anywhere"

# Allow HTTPS from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-hangzhou \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp \
  --PortRange 443/443 \
  --SourceCidrIp "0.0.0.0/0" \
  --Description "HTTPS from anywhere"
```

### Step 3: Create a Key Pair

Never use password auth for ECS. Always use key pairs.

```bash
# Create key pair
aliyun ecs CreateKeyPair \
  --RegionId cn-hangzhou \
  --KeyPairName "fullstack-key" \
  | jq -r '.PrivateKeyBody' > ~/.ssh/fullstack-key.pem

chmod 600 ~/.ssh/fullstack-key.pem
```

### Step 4: Launch the Instance

```bash
# Find the latest Alibaba Cloud Linux 3 image
IMAGE_ID=$(aliyun ecs DescribeImages \
  --RegionId cn-hangzhou \
  --OSType linux \
  --ImageOwnerAlias system \
  --ImageName "aliyun_3_x64*" \
  --PageSize 1 \
  --SortKey CreationDate \
  | jq -r '.Images.Image[0].ImageId')

echo "Image: $IMAGE_ID"

# Launch the instance
INSTANCE_ID=$(aliyun ecs RunInstances \
  --RegionId cn-hangzhou \
  --InstanceType ecs.c7.large \
  --ImageId "$IMAGE_ID" \
  --SecurityGroupId "$SG_ID" \
  --VSwitchId "$VSWITCH_ID" \
  --InstanceName "fullstack-01" \
  --HostName "fullstack-01" \
  --KeyPairName "fullstack-key" \
  --SystemDiskCategory cloud_essd \
  --SystemDiskSize 40 \
  --InternetMaxBandwidthOut 5 \
  --InstanceChargeType PostPaid \
  --SpotStrategy NoSpot \
  --Amount 1 \
  | jq -r '.InstanceIdSets.InstanceIdSet[0]')

echo "Instance ID: $INSTANCE_ID"
```

A few notes on the parameters:

- **`ecs.c7.large`**: 2 vCPU, 4 GB RAM. Enough for learning, cheap enough to not worry about. The `c7` family is Intel Ice Lake — a good default for general workloads.
- **`cloud_essd`**: Enhanced SSD. ESSD is the default for new instances and the right choice for anything that touches disk. The old `cloud_efficiency` (ultra disk) is cheaper but noticeably slower.
- **`InternetMaxBandwidthOut 5`**: 5 Mbps public bandwidth. Setting this to a non-zero value also assigns a public IP automatically — no need for a separate EIP. Fine for learning; in production you would use an EIP or SLB.
- **`PostPaid`**: Pay-as-you-go. Stop the instance when you are done and you stop paying for compute (you still pay a small amount for the disk).

### Step 5: Get the public IP and SSH in

```bash
# Wait for the instance to start (usually 30-60 seconds)
aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0] | "\(.Status) \(.PublicIpAddress.IpAddress[0])"'
```

When the status shows `Running` and you see an IP:

```bash
# SSH in
ECS_IP=$(aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

ssh -i ~/.ssh/fullstack-key.pem root@$ECS_IP
```

You should see the Alibaba Cloud Linux 3 welcome banner. Congratulations — you have a running cloud server.

### The Terraform Alternative

If you have been following along and thinking "this is a lot of imperative CLI commands that I will never be able to reproduce reliably" — you are right. Here is the same thing in Terraform:

```hcl
# main.tf
terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230"
    }
  }
}

provider "alicloud" {
  region = "cn-hangzhou"
}

resource "alicloud_vpc" "main" {
  vpc_name   = "fullstack-vpc"
  cidr_block = "172.16.0.0/16"
}

resource "alicloud_vswitch" "main" {
  vswitch_name = "fullstack-vsw"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "172.16.1.0/24"
  zone_id      = "cn-hangzhou-h"
}

resource "alicloud_security_group" "main" {
  name   = "fullstack-sg"
  vpc_id = alicloud_vpc.main.id
}

resource "alicloud_security_group_rule" "ssh" {
  security_group_id = alicloud_security_group.main.id
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "22/22"
  cidr_ip           = "YOUR_IP/32"
  description       = "SSH from my IP"
}

resource "alicloud_security_group_rule" "http" {
  security_group_id = alicloud_security_group.main.id
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "80/80"
  cidr_ip           = "0.0.0.0/0"
  description       = "HTTP from anywhere"
}

resource "alicloud_ecs_key_pair" "main" {
  key_pair_name = "fullstack-key"
}

data "alicloud_images" "alinux3" {
  owners      = "system"
  name_regex  = "^aliyun_3_x64"
  most_recent = true
}

resource "alicloud_instance" "main" {
  instance_name        = "fullstack-01"
  host_name            = "fullstack-01"
  instance_type        = "ecs.c7.large"
  image_id             = data.alicloud_images.alinux3.images[0].id
  security_groups      = [alicloud_security_group.main.id]
  vswitch_id           = alicloud_vswitch.main.id
  key_name             = alicloud_ecs_key_pair.main.key_pair_name
  system_disk_category = "cloud_essd"
  system_disk_size     = 40

  internet_max_bandwidth_out = 5
  instance_charge_type       = "PostPaid"

  tags = {
    Project = "aliyun-fullstack"
    Series  = "01-ecosystem-map"
  }
}

output "public_ip" {
  value = alicloud_instance.main.public_ip
}

output "instance_id" {
  value = alicloud_instance.main.id
}
```

```bash
terraform init
terraform plan
terraform apply
```

Three commands instead of twelve. The state is tracked. The config is version-controlled. Tear it down with `terraform destroy` when you are done. We cover Terraform in depth in [Article 7](/en/aliyun-fullstack/07-observability) and there is an entire [Terraform for AI Agents series](/en/terraform-agents/01-why-terraform-for-agents/) if you want to go deep.

## First Deployment Checklist

Your ECS instance is running. Now let us make it useful.

![First deployment flow from signup to access](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_deployment_flow.png)

### Install essential packages

```bash
# On the ECS instance (Alibaba Cloud Linux 3 uses dnf, same as RHEL 8+)
sudo dnf update -y

# Web server
sudo dnf install -y nginx
sudo systemctl enable --now nginx

# Development tools
sudo dnf install -y python3 python3-pip nodejs npm git curl wget jq

# Verify
nginx -v
python3 --version
node --version
```

### Serve a test page

```bash
# Create a simple test page
cat > /usr/share/nginx/html/index.html << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head><title>Alibaba Cloud Full Stack</title></head>
<body>
  <h1>It works.</h1>
  <p>ECS instance is live. Series: aliyun-fullstack, Article 01.</p>
  <p>Server time: <script>document.write(new Date().toISOString())</script></p>
</body>
</html>
HTMLEOF

# Restart nginx
sudo systemctl restart nginx
```

### Verify public access

From your local machine:

```bash
curl http://$ECS_IP
```

You should see the HTML. If you get a timeout, check:
1. Security group rules — is port 80 open?
2. Is nginx running? (`systemctl status nginx` on the instance)
3. Is the public IP correct? (`aliyun ecs DescribeInstances` to verify)

### Set up basic monitoring

CloudMonitor is enabled by default for ECS instances. But the default agent might not be installed:

```bash
# On the ECS instance — install CloudMonitor agent
ARGUS_VERSION=3.5.8
curl -sL "https://cms-agent-cn-hangzhou.oss-cn-hangzhou.aliyuncs.com/cms-go-agent/${ARGUS_VERSION}/CmsGoAgent.linux-amd64.tar.gz" \
  | tar xzf - -C /usr/local/

/usr/local/CmsGoAgent/CmsGoAgent start
/usr/local/CmsGoAgent/CmsGoAgent status
```

Now go to the CloudMonitor console and you should see CPU, memory, disk, and network metrics for your instance within a few minutes.

### Clean up when done

Do not forget to stop or release the instance when you are done experimenting. A running `ecs.c7.large` costs about 0.35 CNY/hour. That is 252 CNY/month if you leave it running.

```bash
# Stop the instance (you still pay for the disk, ~0.5 CNY/day for 40 GB ESSD)
aliyun ecs StopInstance --InstanceId "$INSTANCE_ID"

# Or release it entirely (all data is lost)
aliyun ecs DeleteInstance --InstanceId "$INSTANCE_ID" --Force true

# If using Terraform, just:
terraform destroy
```

## The Architecture We Are Building

This article is the first of twelve in the Alibaba Cloud Full Stack series. Here is the full roadmap:

![12-article series roadmap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_architecture_roadmap.png)

```text
Article 01: The Ecosystem Map (you are here)
    |
    v
Article 02: ECS Deep Dive — Instance Types, Images, Disks
    |
    v
Article 03: Networking — VPC, SLB, CDN, DNS
    |
    v
Article 04: Storage — OSS, NAS, Block Storage Patterns
    |
    v
Article 05: Databases — RDS, PolarDB, Redis, MongoDB
    |
    v
Article 06: Security — RAM, KMS, Security Groups, WAF
    |
    v
Article 07: Infrastructure as Code — Terraform on Alibaba Cloud
    |
    v
Article 08: Containers — ACK, SAE, and When to Use What
    |
    v
Article 09: Serverless — Function Compute, EventBridge
    |
    v
Article 10: Observability — SLS, CloudMonitor, ARMS
    |
    v
Article 11: AI/ML — PAI, DashScope, Bailian
    |
    v
Article 12: Putting It All Together — A Production Architecture
```

Each article builds on the previous one. By article 12, you will have a complete production-ready stack running on Alibaba Cloud with proper networking, security, monitoring, and CI/CD.

Some of these topics have dedicated deep-dive series on this blog already:

- **PAI (articles 1-5)**: If you want the full treatment on GPU notebooks, distributed training, and model serving, see the [PAI series](/en/aliyun-pai/01-platform-overview/).
- **Bailian and DashScope (articles 1-5)**: For LLM APIs, multimodal models, TTS, and video generation, see the [Bailian series](/en/aliyun-bailian/01-platform-overview/).
- **Terraform (articles 1-8)**: For infrastructure-as-code patterns specific to AI agent systems, see the [Terraform for AI Agents series](/en/terraform-agents/01-why-terraform-for-agents/).

This Full Stack series is the breadth-first tour. Those series are the depth-first dives. They complement each other — I will point you to the right deep-dive article when we reach each topic.

## Summary

1. **Alibaba Cloud has a near-1:1 mapping to AWS.** If you know AWS, you already know 80% of Alibaba Cloud. The remaining 20% is naming, console layout, and China-specific features. Use the service map table in this article as your Rosetta Stone.

2. **Account setup has an extra step.** Real-name verification is mandatory and takes 1-3 days. Start this process before you need to deploy anything. Create a RAM admin user immediately and never use the root account for daily work.

3. **Region choice matters more than on AWS.** Chinese internet topology means region-to-user latency varies significantly. Default to `cn-hangzhou` for China workloads and `ap-southeast-1` for international. Data residency laws mean China data must stay in China.

4. **Use Pay-As-You-Go while learning, Subscription for production.** The price difference is 40-60%. Do not commit to Subscription until you know your workload is stable. And always set billing alerts — a forgotten NAT Gateway or idle GPU instance will quietly drain your account.

5. **Terraform from day one.** The CLI walkthrough above is instructive, but in practice you should never create infrastructure imperatively. Every resource in this series will have a Terraform equivalent. Your future self will thank you.

Next up: [Article 02 — ECS Deep Dive](/en/aliyun-fullstack/02-ecs-compute), where we go deep on instance type selection, image management, disk performance characteristics, and the placement strategies that actually matter for production workloads.
