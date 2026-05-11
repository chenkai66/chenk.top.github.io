---
title: "阿里云全栈实战（一）：生态全景图——阿里云到底是什么"
date: 2026-04-28 09:00:00
tags:
  - Alibaba Cloud
  - Cloud Computing
  - ECS
  - Getting Started
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 1
description: "A 实战指南 to Alibaba Cloud's product ecosystem. We map every major service to its AWS/Azure/GCP equivalent, set up an account from scratch, and deploy our first ECS instance — all in under an hour."
disableNunjucks: true
translationKey: "aliyun-fullstack-1"
---
刚上阿里云的第一周，我彻底迷失在产品命名的海洋里。ECS、SLB、SLS、RDS、OSS、NAS、PAI、ARMS、ACK、FC、CDN、WAF、RAM、KMS、ROS、CloudMonitor、EventBridge、PolarDB、Lindorm、AnalyticDB、MaxCompute、DataWorks、Flink、DashScope、Bailian、OpenSearch……每个控制台页面都连着三个我没听过的产品。文档假定你已经知道所有东西是什么。英文翻译有时直译，有时意译，偶尔还会缺失。这正是我初上阿里云时最需要的一份指南——不用把第一个周末全花在点控台和硬啃文档上；那些翻译文档常常只告诉你怎么打开某个开关，却从不解释这个产品到底用来做什么。

这篇文章就是全景地图。我们要把整个阿里云生态映射到你可能已经熟悉的 AWS/Azure/GCP 服务，从零 设置账号，搞清楚计费模型以免被账单吓一跳，最后部署一个能跑的 ECS 实例。不讲空洞的理论。文中提到的每一项服务，我都已在生产环境实际使用过，或经审慎评估后决定弃用。


## 为什么选阿里云？

从 AWS 过来的人第一个问题通常是：凭什么选阿里云？

![阿里云产品家族树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_product_families.png)

三个理由，按实际重要性降序排列：

**1. 中国。** 如果你的用户在中国，你就需要中国的基础设施。AWS 中国（由光环新网/西云数据运营）确实存在，但它是独立实体，账号独立，服务子集，发布周期更慢。Azure 中国（由世纪互联运营）也有同样的限制。GCP 根本没有中国大陆区域。阿里云是原生云——它运行着淘宝、支付宝、天猫和饿了么。它有七个大陆地域，亚洲最大的数据中心足迹，每个服务第一天就能用，不需要 ICP 牌照合作伙伴安排。

**2. 市场地位。** 阿里云是亚太地区最大的云提供商，全球第三，仅次于 AWS 和 Azure。根据 Gartner 2025 年的数据，它占据了中国云市场约 35% 的份额，全球约 5%。这和你在西方选 AWS 的理由一样：生态系统（第三方集成、社区支持、招聘池、供应商合作伙伴）跟随市场领导者。

**3. 独特服务。** 有些阿里云产品在其他地方没有直接对应，或者明显领先：

- **DashScope / Bailian** —— Qwen 模型家族（LLM、多模态、TTS、视频生成）只在这里原生可用。你可以在任何云上自托管 Qwen，但带有按 Token 计费和中文语言优化的托管 API 是阿里云独有的。
- **PolarDB** —— 云原生分布式数据库，确实和 Aurora 不同。它在页级别分离计算和存储，并在同一产品中支持 PostgreSQL、MySQL 和分布式（分片）模式。
- **MaxCompute / DataWorks** —— 处理阿里内部分析的数据仓库和 ETL 栈。AWS 上没有东西能匹配这两者之间的集成深度。
- **支付宝 / 淘宝生态集成** —— 如果你为中国电商构建任何东西（小程序、支付流程、商家工具），原生集成能节省数月工作。

如果以上几点都不适用——比如用户主要在美国、无需中文大模型支持、也没有亚洲市场拓展需求——那么继续使用 AWS 完全合理。我完全认同这一点。但如果其中任何一条适用，阿里云就值得系统学习——这正是我们写这篇文章的原因。

## 服务地图：阿里云 vs AWS vs Azure vs GCP

这篇文章最有价值的部分就是给你这份对照表。每个阿里云产品映射到你已经知道的服务。描述保持紧凑——每句一句话——因为你不需要一段话来理解"OSS 就是 S3"。

![阿里云服务地图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_service_map.png)

### 计算

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **ECS** (Elastic Compute Service) | EC2 | Virtual Machines | Compute Engine | 虚拟机。基础中的基础。 |
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | 无服务器函数，事件驱动。 |
| **ACK** (Container Service for Kubernetes) | EKS | AKS | GKE | 托管 Kubernetes。 |
| **SAE** (Serverless App Engine) | App Runner | Container Apps | Cloud Run | 无服务器容器，无需管理 K8s。 |
| **ECI** (Elastic Container Instance) | Fargate | Container Instances | Cloud Run Jobs | 无服务器容器实例，无需集群。 |
| **Batch Compute** | AWS Batch | Azure Batch | Cloud Batch | 托管批作业调度。 |

### 存储

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **OSS** (Object Storage Service) | S3 | Blob Storage | Cloud Storage | 对象存储。存文件，取文件。 |
| **NAS** (Network Attached Storage) | EFS | Azure Files | Filestore | 兼容 POSIX 的共享文件系统。 |
| **CPFS** (Cloud Parallel File System) | FSx for Lustre | — | — | 高性能并行文件系统，用于 HPC/AI。 |
| **Block Storage** (EBS equivalent) | EBS | Managed Disks | Persistent Disk | 挂载到 ECS 实例的块存储。 |
| **Tablestore** | DynamoDB | Table Storage | Bigtable | 宽列 NoSQL，适合时间序列。 |

### 数据库

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **RDS** | RDS | Azure SQL / Azure DB | Cloud SQL | 托管 MySQL、PostgreSQL、SQL Server、MariaDB。 |
| **PolarDB** | Aurora | — | AlloyDB | 云原生分布式关系数据库。 |
| **Lindorm** | DynamoDB + Timestream | Cosmos DB | Bigtable + BigTable | 多模型数据库（宽列、时间序列、搜索）。 |
| **AnalyticDB (ADB)** | Redshift | Synapse | BigQuery | 云数据仓库，列式，MPP。 |
| **Tair** (formerly Redis Enterprise) | ElastiCache | Azure Cache for Redis | Memorystore | 托管 Redis 兼容内存存储。 |
| **MongoDB** (ApsaraDB for MongoDB) | DocumentDB | Cosmos DB (Mongo API) | — | 托管 MongoDB。 |

### 网络

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **VPC** | VPC | Virtual Network | VPC | 虚拟私有网络，基础架构。 |
| **SLB** (Server Load Balancer) | ALB / NLB | Load Balancer | Cloud Load Balancing | 4 层 + 7 层负载均衡。 |
| **ALB** (Application Load Balancer) | ALB | Application Gateway | — | 7 层负载均衡（较新产品）。 |
| **CDN** | CloudFront | Azure CDN | Cloud CDN | 内容分发网络。 |
| **DCDN** (Dynamic CDN) | CloudFront + Global Accelerator | Front Door | — | 动态内容加速 + 边缘计算。 |
| **CEN** (Cloud Enterprise Network) | Transit Gateway | Virtual WAN | Cloud Interconnect | 多 VPC、多地域网络骨干。 |
| **NAT Gateway** | NAT Gateway | NAT Gateway | Cloud NAT | 私有实例 outbound 互联网访问。 |
| **EIP** (Elastic IP) | Elastic IP | Public IP | Static IP | 静态公网 IP 地址。 |
| **PrivateLink** | PrivateLink | Private Link | Private Service Connect | 私有端点访问服务，不经过互联网。 |

### AI 与机器学习

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **PAI** (Platform for AI) | SageMaker | Azure ML | Vertex AI | 完整 ML 平台：Notebooks、训练、服务。 |
| **DashScope** | Bedrock | Azure OpenAI Service | Vertex AI (Gemini) | 模型 API 网关（Qwen、万相、Embeddings）。 |
| **Bailian (百炼)** | Bedrock console | Azure AI Studio | — | 模型管理控制台，RAG 构建，微调 UI。 |
| **OpenSearch** | OpenSearch Service | Azure AI Search | Vertex AI Search | 托管搜索，带 AI 排序和检索。 |

### 安全与身份

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **RAM** (Resource Access Management) | IAM | Entra ID (Azure AD) | Cloud IAM | 用户、角色、策略。 |
| **KMS** (Key Management Service) | KMS | Key Vault | Cloud KMS | 加密密钥管理。 |
| **WAF** (Web Application Firewall) | WAF | Azure WAF | Cloud Armor | Web 应用防火墙。 |
| **Security Center** | GuardDuty + Inspector | Defender for Cloud | Security Command Center | 威胁检测，漏洞扫描。 |
| **SSL Certificates** | ACM | App Service Certificates | Certificate Manager | 托管 TLS/SSL 证书。 |
| **ActionTrail** | CloudTrail | Activity Log | Cloud Audit Logs | API 审计日志。 |

### 可观测性

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **SLS** (Simple Log Service) | CloudWatch Logs | Azure Monitor Logs | Cloud Logging | 日志收集、索引、分析。 |
| **CloudMonitor** | CloudWatch Metrics | Azure Monitor | Cloud Monitoring | 基础设施指标和告警。 |
| **ARMS** | X-Ray + CloudWatch APM | Application Insights | Cloud Trace | 应用性能监控，分布式追踪。 |
| **Grafana Service** | Amazon Managed Grafana | Azure Managed Grafana | — | 托管 Grafana 仪表盘。 |

### 无服务器与事件驱动

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | 无服务器函数。 |
| **EventBridge** | EventBridge | Event Grid | Eventarc | 事件总线，路由，过滤。 |
| **Message Queue (RocketMQ)** | SQS + SNS | Service Bus | Pub/Sub | 分布式消息队列（阿里自研）。 |
| **Kafka (ApsaraDB for Kafka)** | MSK | Event Hubs (Kafka) | Managed Kafka | 托管 Apache Kafka。 |

### 基础设施即代码

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **Terraform** (alicloud provider) | Terraform (aws provider) | Terraform (azurerm) | Terraform (google) | 带官方 Provider 的 HashiCorp Terraform。 |
| **ROS** (Resource Orchestration Service) | CloudFormation | ARM/Bicep | Deployment Manager | 原生基于模板的 IaC。 |
| **Pulumi** (alicloud provider) | Pulumi (aws) | Pulumi (azure) | Pulumi (gcp) | Python/TS/Go 编程式 IaC。 |

产品确实很多。你不需要背下这张表。把它加入书签。当你在文档页面看到缩写需要知道映射关系时再回来查。最有帮助的心智模型是：**阿里云几乎为每个 AWS 服务都有接近 1:1 的对应产品**，外加一些 extras（PolarDB、DashScope、MaxCompute）和一些 gaps（没有像 AWS Organizations 多账号策略那样成熟的对应方案，英文 SDK 文档比中文落后几周到几个月）。
## 账号搭建：从零到控制台

搭阿里云账号跟 AWS 不太一样。有个额外步骤，每次都能把西方开发者第一次就卡住：**实名认证**（实名认证）。

### 第一步：注册

去 [aliyun.com](https://www.alibabacloud.com/) 注册。国际手机号和邮箱都能用。国际站支持信用卡，中国站（aliyun.com）则需要支付宝或国内银行卡。

> **选哪个站？** 如果你的业务面向国内用户，选中国站（aliyun.com）。如果面向国际用户，选国际站（alibabacloud.com）。账号独立，账单独立，区域也（大部分）独立。两个都能用，但得选一个作为主账号。本系列默认用中国站，毕竟完整的产品目录都在这儿。

### 第二步：实名认证

国内所有云服务都必须实名认证。这是监管硬性要求，不是阿里自己的规矩。AWS 中国也一样。流程如下：

- **个人账号**：上传身份证照片（国内居民用身份证，外国人用护照）。审核通常需要 1-3 个工作日。
- **企业账号**：上传营业执照、法人身份证，有时还需要银行打款验证。耗时 3-5 个工作日。

如果你只是学习折腾，个人账号就够了。以后可以随时升级为企业账号。主要区别在于发票类型（个人 vs 企业发票）和消费限额。

### 第三步：开启 MFA

这事得马上做。`home.console.aliyun.com` 这个控制台拿着你账单、数据和基础设施的钥匙。给根账号开启 MFA：

1. 进入 **账号管理** > **安全设置**
2. 开启 **虚拟 MFA 设备**
3. 用认证器 App（Google Authenticator、1Password 等）扫码
4. 输入两个连续验证码确认

### 第四步：创建 RAM 管理员用户

日常干活别用根账号。这点跟 AWS 建议一致——根账号只管账单和紧急恢复。其他所有操作都走 RAM。

```bash
# 先安装 Alibaba Cloud CLI
# macOS
brew install aliyun-cli

# 或者从 https://github.com/aliyun/aliyun-cli/releases 下载

# 用根账号配置 CLI（临时，用来创建管理员用户）
aliyun configure set \
  --profile root \
  --mode AK \
  --access-key-id <YOUR_ROOT_AK_ID> \
  --access-key-secret <YOUR_ROOT_AK_SECRET> \
  --region cn-hangzhou
```

现在创建一个 RAM 管理员用户：

```bash
# 创建 RAM 用户
aliyun ram CreateUser --UserName admin

# 为新用户创建 AccessKey
aliyun ram CreateAccessKey --UserName admin

# 绑定 AdministratorAccess 策略
aliyun ram AttachPolicyToUser \
  --PolicyType System \
  --PolicyName AdministratorAccess \
  --UserName admin

# 开启 RAM 用户的控制台登录权限
aliyun ram CreateLoginProfile \
  --UserName admin \
  --Password 'YourStrongPassword123!' \
  --MFABindRequired true
```

把 AccessKey 输出存到安全地方（比如密码管理器，别贴便利贴上）。然后重新配置 CLI 使用 RAM 用户：

```bash
aliyun configure set \
  --profile admin \
  --mode AK \
  --access-key-id <RAM_USER_AK_ID> \
  --access-key-secret <RAM_USER_AK_SECRET> \
  --region cn-hangzhou

# 设为默认 profile
aliyun configure set --profile admin
```

从这步开始，除非你要管账单或者从锁死状态恢复，否则别再碰根账号凭证。

### 第五步：设置账单预警

去 **费用中心** > **消费预警** 设个预算。学习期我建议设 100 CNY/月。免费额度够大多数实验折腾，但要是忘了关 ECS 实例或者丢着一台 GPU VM 不管，一夜之间就能把你的预算烧光。

```bash
# 快速检查：这个月我花了多少？
aliyun bssopenapi QueryAccountBalance
```

### 第六步：了解免费额度

阿里云新账号的免费额度（有效期 12 个月）包括：

- **ECS**：1 台实例（1 vCPU, 1 GB RAM）可用 3 个月——够本系列用了
- **OSS**：5 GB 存储，5 GB  outbound 流量 / 月
- **RDS**：1 台微实例可用 1 个月
- **Function Compute**：100 万次请求 + 400,000 GB-秒 / 月
- **SLS**：500 MB / 天 摄入
- **各类 AI 服务**：每个模型免费额度（DashScope, PAI 等）

具体 offerings 每季度会变。去 [免费试用页面](https://free.aliyun.com/) 看最新详情。

## 区域与可用区

阿里云在国内的区域数量比任何云厂商都多。这很关键，因为国内的网络拓扑决定了，深圳用户连北京服务器，延迟肯定比连深圳服务器高得多——防火墙加物理距离，双重影响。

![阿里云全球区域与可用区](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_region_map.png)

### 中国区

| Region ID | Location | Good for |
|---|---|---|
| `cn-hangzhou` | 杭州，浙江 | 大多数服务的默认区。阿里总部。服务覆盖最全。 |
| `cn-shanghai` | 上海 | 金融服务，AI/ML（GPU 库存最足）。 |
| `cn-beijing` | 北京 | 政府，媒体，华北用户。 |
| `cn-shenzhen` | 深圳，广东 | 华南，毗邻香港。 |
| `cn-zhangjiakou` | 张家口，河北 | 冷存储，批处理（最便宜）。 |
| `cn-hohhot` | 呼和浩特，内蒙古 | 大数据，成本敏感型负载。 |
| `cn-wulanchabu` | 乌兰察布，内蒙古 | AI 训练，大型 GPU 集群。 |
| `cn-chengdu` | 成都，四川 | 西南覆盖。 |
| `cn-guangzhou` | 广州，广东 | 华南替代深圳的选择。 |
| `cn-nanjing` | 南京，江苏 | 华东冗余。 |
| `cn-fuzhou` | 福州，福建 | 跨海峡（毗邻台湾）。 |
| `cn-heyuan` | 河源，广东 | 华南成本优化。 |

### 国际区

| Region ID | Location | Good for |
|---|---|---|
| `ap-southeast-1` | 新加坡 | 东南亚，连国内延迟最低的非国内区。 |
| `ap-northeast-1` | 东京 | 日本，韩国用户。 |
| `ap-south-1` | 孟买 | 印度，南亚。 |
| `ap-southeast-5` | 雅加达 | 印尼。 |
| `ap-southeast-3` | 吉隆坡 | 马来西亚。 |
| `eu-central-1` | 法兰克福 | 欧洲。 |
| `eu-west-1` | 伦敦 | 英国。 |
| `us-east-1` | 弗吉尼亚 | 美东。 |
| `us-west-1` | 硅谷 | 美西。 |
| `ap-southeast-2` | 悉尼 | 澳洲，大洋洲。 |
| `me-east-1` | 迪拜 | 中东。 |

### 怎么选

这是我的决策树，都是从生产环境踩坑踩出来的：

1. **用户在哪？** 选离他们最近的区域。延迟压倒一切。
2. **需要符合国内数据驻留法律吗？** 如果是，选中国大陆区域。`cn-*` 区域的数据没有明确的跨境传输批准不能出境。
3. **需要 GPU 吗？** 上海和杭州 GPU 库存最足。乌兰察布 GPU 定价最便宜，但连终端用户网络最差。
4. **成本敏感的批处理负载？** 张家口和呼和浩特计算资源便宜 15-30%。
5. **默认选啥？** 国内选 `cn-hangzhou`，国际选 `ap-southeast-1`。

> **实战建议：** 每个区域都有多个可用区（AZ），标记为 `a`, `b`, `c` 等。只要是重要业务，至少跨两个 AZ 部署。AZ 故障虽少见，但确实会发生——我在 18 个月内见过 `cn-hangzhou-h` 两次降级。

## 理解计费模式

阿里云的计费模式比 AWS 灵活，但要是你不熟悉，也会觉得更晕。一共有四种购买模式，你都得搞清楚。

![阿里云计费模式对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_billing_comparison.png)

### 按量付费（PAYG, 按量付费）

字面意思，用多少付多少，按秒计费（计算）或按 GB 计费（存储/流量）。这是默认模式，也是你学习期间应该用的。无承诺，无预付，随时停。

缺点：单位成本最贵。`cn-hangzhou` 的一台 ECS `ecs.c7.large`（2 vCPU, 4 GB）大概价格：
- 按量付费：~0.35 CNY/小时（24/7 运行约 252 CNY/月）
- 包年包月：~125 CNY/月（1 年承诺）

为了灵活性，你得付 50% 的溢价。

### 包年包月（Subscription, 包年包月）

你承诺用 1 个月、3 个月、6 个月、1 年、2 年或 3 年。承诺越久，折扣越深。3 年的 ECS 承诺比按量付费便宜 40-60%。但你需要预付（或按月自动续费），而且即使你停用实例，钱照样扣。

适用场景：你知道至少会跑一年的生产服务器。

### 抢占式实例（Preemptible / Spot Instances, 抢占式实例）

概念跟 AWS Spot Instances 一样。你竞价使用闲置容量，折扣 10-90%。实例可能被回收，但会有 5 分钟预警。阿里云的 Spot 市场竞争没 AWS 那么激烈，这意味着折扣往往更深，中断也没那么频繁——至少我在 `cn-shanghai` 的经验是这样。

适用场景：批处理、可 checkpoint 的训练任务、CI  runners。

```bash
# 启动一台抢占式实例
aliyun ecs RunInstances \
  --RegionId cn-hangzhou \
  --InstanceType ecs.c7.large \
  --SpotStrategy SpotAsPriceGo \
  --SpotPriceLimit 0.2 \
  --ImageId aliyun_3_x64_20G_alibase_20240819.vhd \
  --SecurityGroupId sg-bp1xxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxx
```

### 节省计划与资源包

**节省计划** 跟 AWS Savings Plans 类似——你承诺每小时消费一定金额，就能在所有计算资源（ECS, ECI, ACK）上获得折扣，不管实例类型或区域。比包年包月灵活，但要求你清楚自己的基线消费。

**资源包**（资源包）是针对特定服务的预付 bundle。比如买 500 GB OSS 存储用一年，打 7 折。买 1000 万次 Function Compute 调用，打 6 折。摸清用量模式后再买划算，学习期间别买。

### 账单控制台

`usercenter2.aliyun.com` 是搞懂你花钱去向的地方。关键页面：

- **账单** > **账单概览**：按产品划分的月度汇总
- **账单** > **账单明细**：行级详情（实例级）
- **成本管理** > **成本分析**：按产品、区域、标签分解
- **预算** > **预算预警**：设置消费阈值

我每周都会查。我在生产账号上吃过最大的账单亏，就是一个忘在测试 VPC 里的 NAT Gateway——15 CNY/天，而我根本没在用那个资源。要是第一天就设好标签和预算预警，本来能避开的。
## 你的第一台 ECS 实例

背景铺垫到此为止，咱们来部署点东西。

![阿里云免费额度一览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_free_tier.png)

我们要启动一台运行 Alibaba Cloud Linux 3 的 ECS 实例，SSH 连上去，装个 nginx，然后托管一个测试页面。大概 15 分钟搞定。

### 前置条件

- 已完成实名的阿里云账号（前面已搞定）
- 已在 `aliyun` CLI 中配置好 Access Key 的 RAM 管理员用户（前面已搞定）
- 选定的地域（我用 `cn-hangzhou`）

### 第一步：创建 VPC 和 VSwitch

每台 ECS 实例都得活在 VPC 里。你需要一个 VPC 和至少一个 VSwitch（子网）。

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

### 第二步：创建安全组

安全组其实就是虚拟防火墙。咱们得放行 SSH（22 端口）仅给自己的 IP，HTTP（80 端口）开放给全网。

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

### 第三步：创建密钥对

ECS 千万别用密码认证。始终使用密钥对。

```bash
# Create key pair
aliyun ecs CreateKeyPair \
  --RegionId cn-hangzhou \
  --KeyPairName "fullstack-key" \
  | jq -r '.PrivateKeyBody' > ~/.ssh/fullstack-key.pem

chmod 600 ~/.ssh/fullstack-key.pem
```

### 第四步：启动实例

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

这几个参数得说一下：

- **`ecs.c7.large`**：2 vCPU，4 GB 内存。拿来学习够用，价格也不心疼。`c7` 系列是 Intel Ice Lake 架构，通用负载的默认好选择。
- **`cloud_essd`**：增强型 SSD。新实例默认就是这个，只要涉及磁盘读写，选它准没错。老的 `cloud_efficiency`（高效云盘）虽然便宜点，但慢得明显。
- **`InternetMaxBandwidthOut 5`**：5 Mbps 公网带宽。设成非零值会自动分配公网 IP，不用单独搞 EIP。学习够用；生产环境建议用 EIP 或 SLB。
- **`PostPaid`**：按量付费。用完停掉实例，计算资源就不再计费（磁盘还得付点小钱）。

### 第五步：获取公网 IP 并 SSH 登录

```bash
# Wait for the instance to start (usually 30-60 seconds)
aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0] | "\(.Status) \(.PublicIpAddress.IpAddress[0])"'
```

等到状态显示 `Running` 并且看到 IP 后：

```bash
# SSH in
ECS_IP=$(aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

ssh -i ~/.ssh/fullstack-key.pem root@$ECS_IP
```

你应该能看到 Alibaba Cloud Linux 3 的欢迎 banner。恭喜，你有了一台运行中的云服务器。

### Terraform 方案

跟着做到这儿，你可能会想：“这么多命令式 CLI 操作，以后根本没法可靠复现”——没错。同样的事，用 Terraform 是这样：

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

三条命令搞定，不用敲十二遍。状态有人跟踪，配置进了版本控制。用完 `terraform destroy` 一键销毁。我们在 [第 7 篇](/zh/aliyun-fullstack/07-terraform-iac/) 会深入讲 Terraform，如果想深挖，还有个完整的 [Terraform for AI Agents series](/zh/terraform-agents/01-why-terraform-for-agents/)。

## 首次部署检查清单

ECS 实例跑起来了。现在让它干点正事。

![从注册到访问的首次部署流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_deployment_flow.png)

### 安装必备包

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

### 托管个测试页

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

### 验证公网访问

在本地机器上：

```bash
curl http://$ECS_IP
```

应该能看到 HTML 内容。如果超时，检查：
1. 安全组规则 —— 80 端口开了吗？
2. nginx 跑着吗？（实例上 `systemctl status nginx`）
3. 公网 IP 对吗？（`aliyun ecs DescribeInstances` 确认一下）

### 配置基础监控

ECS 实例默认开启云监控。但默认 agent 未必装好了：

```bash
# On the ECS instance — install CloudMonitor agent
ARGUS_VERSION=3.5.8
curl -sL "https://cms-agent-cn-hangzhou.oss-cn-hangzhou.aliyuncs.com/cms-go-agent/${ARGUS_VERSION}/CmsGoAgent.linux-amd64.tar.gz" \
  | tar xzf - -C /usr/local/

/usr/local/CmsGoAgent/CmsGoAgent start
/usr/local/CmsGoAgent/CmsGoAgent status
```

现在去云监控控制台，几分钟内应该就能看到实例的 CPU、内存、磁盘和网络指标。

### 用完记得清理

实验做完，别忘了停掉或释放实例。运行中的 `ecs.c7.large` 大概 0.35 元/小时。放着不管一个月就是 252 元。

```bash
# Stop the instance (you still pay for the disk, ~0.5 CNY/day for 40 GB ESSD)
aliyun ecs StopInstance --InstanceId "$INSTANCE_ID"

# Or release it entirely (all data is lost)
aliyun ecs DeleteInstance --InstanceId "$INSTANCE_ID" --Force true

# If using Terraform, just:
terraform destroy
```

## 我们要构建的架构

这篇文章是阿里云全栈系列十二篇中的第一篇。完整路线图如下：

![12 篇系列文章路线图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/01-ecosystem-map/01_architecture_roadmap.png)

```
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

每篇都基于前一篇的内容。到第 12 篇时，你将拥有一个运行在阿里云上的完整生产级栈，包含合适的网络、安全、监控和 CI/CD。

其中有些主题本博客已经有专门的深度系列了：

- **PAI (articles 1-5)**：如果想全面了解 GPU 笔记本、分布式训练和模型服务，请看 [PAI series](/zh/aliyun-pai/01-platform-overview/)。
- **Bailian and DashScope (articles 1-5)**：针对 LLM API、多模态模型、TTS 和视频生成，请看 [Bailian series](/zh/aliyun-bailian/01-platform-overview/)。
- **Terraform (articles 1-8)**：针对 AI 代理系统的基础设施即代码模式，请看 [Terraform for AI Agents series](/zh/terraform-agents/01-why-terraform-for-agents/)。

这个全栈系列是广度优先的巡礼。那些系列是深度优先的深挖。两者互补——每到一个主题，我会指引你去看对应的深度文章。
## 核心要点

1. **阿里云和 AWS 的服务映射差不多是 1:1 的。** 只要你熟悉 AWS，阿里云 80% 的内容你其实已经掌握了。剩下 20% 无非是产品命名、控制台布局以及一些中国特有的功能。直接把文中的服务映射表当成你的 Rosetta Stone 对照着看就行。

2. **账号注册多了个步骤。** 实名认证是强制的，得花 1-3 天。别等到要部署了才想起来，提前搞定。注册完立刻创建一个 RAM 管理员用户，日常操作千万别用 root 账号。

3. **选 Region 比在 AWS 上更关键。** 国内网络拓扑复杂，不同 Region 到用户的延迟差异很明显。国内业务默认选 `cn-hangzhou`，国际业务选 `ap-southeast-1`。另外注意数据合规，国内数据必须留在境内。

4. **学习阶段用按量付费，生产环境再转包年包月。** 两者差价能达到 40-60%。别急着锁定包年包月，等负载稳定了再说。务必设置账单告警——一个被遗忘的 NAT Gateway 或者闲置的 GPU 实例，会悄无声息地掏空你的账户。

5. **从第一天就开始用 Terraform。** 上面的 CLI 演示只是为了教学，实际生产中千万别命令式地创建基础设施。本系列里的每个资源都会提供对应的 Terraform 代码。未来的你一定会感谢现在的决定。

下一篇：[文章 02 — ECS 深挖](/zh/aliyun-fullstack/02-ecs-deep-dive/)，我们会深入探讨实例选型、镜像管理、磁盘性能特征，以及对生产负载至关重要的部署策略。