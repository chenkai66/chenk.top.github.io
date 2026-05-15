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
series_title: "阿里云全栈实战"
series_order: 1
series_total: 12
description: "阿里云产品生态实战指南。将每个核心服务映射到 AWS/Azure/GCP 对标产品，从零开通账号，部署第一台 ECS 实例——一小时搞定。"
disableNunjucks: true
translationKey: "aliyun-fullstack-1"
---
刚接触阿里云的第一周，我彻底迷失在产品名称的海洋里：ECS、SLB、SLS、RDS、OSS、NAS、PAI、ARMS、ACK、FC、CDN、WAF、RAM、KMS、ROS、CloudMonitor、EventBridge、PolarDB、Lindorm、AnalyticDB、MaxCompute、DataWorks、Flink、DashScope、Bailian、OpenSearch……每个控制台页面都链接到三四个我没听过的产品，文档默认你已经了解一切，英文翻译有时直译、有时意译，偶尔干脆缺失。这正是我当初最需要的指南——不用浪费整个周末点击控制台、硬啃那些只教你怎么开关功能却从不解释产品本质的翻译文档。

本文就是一张全景地图：本文阿里云生态完整映射到你熟悉的 AWS/Azure/GCP 服务，从零开通账号、厘清计费模型（避免账单惊吓），最终部署一台可运行的 ECS 实例。文中提到的每一项服务，要么已在生产环境使用，要么经过审慎评估后明确弃用，绝不空谈理论。

---

## 为什么选阿里云？

从 AWS 过来的人通常会问：为什么要用阿里云？

![阿里云产品家族树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_product_families.png)

原因有三，按实际重要性降序排列：

**1. 中国。** 如果你的用户在中国，就必须使用中国本土的基础设施。AWS 中国（由光环新网/西云数据运营）虽然存在，但它是独立实体，需单独注册账号，仅提供部分服务，且新功能上线更慢；Azure 中国（由世纪互联运营）也有同样限制；GCP 则根本没有中国大陆区域。阿里云是真正的原生云——淘宝、支付宝、天猫、饿了么都跑在上面。它拥有七个中国大陆地域、亚洲最大的数据中心规模，所有服务上线首日即可使用，无需通过持有 ICP 牌照的合作伙伴。

**2. 市场地位。** 阿里云是亚太地区最大的云服务商，全球排名第三，仅次于 AWS 和 Azure。据 Gartner 2025 年数据，它占据中国云市场约 35% 的份额，全球约 5%。这一点之所以重要，原因和你在西方选择 AWS 一样：生态系统（第三方集成、社区支持、人才储备、厂商合作）总是追随市场领导者。

**3. 独特服务。** 部分阿里云产品在其他云上没有直接对标，或明显领先：

- **DashScope / Bailian** —— Qwen 模型家族（大语言模型、多模态、语音合成、视频生成）仅在此原生提供。你当然可以在任何云上自托管 Qwen，但带有按 Token 计费和中文优化的托管 API 是阿里云独有。
- **PolarDB** —— 一款真正不同于 Aurora 的云原生分布式数据库。它在页级别分离计算与存储，并在同一产品中同时支持 PostgreSQL、MySQL 和分布式（分片）模式。
- **MaxCompute / DataWorks** —— 支撑阿里巴巴内部数据分析的数据仓库与 ETL 栈。AWS 上没有任何组合能匹配这两者之间的深度集成。
- **支付宝 / 淘宝生态集成** —— 如果你为中国电商开发（小程序、支付流程、商家工具），原生集成能节省数月开发时间。

如果你的情况完全不适用——比如用户主要在美国、不需要中文大模型、也无意拓展亚洲市场——那继续用 AWS 完全合理，我绝不会强行说服你。但只要上述任意一条成立，阿里云就值得系统学习，这也是我们撰写本系列的原因。

## 服务地图：阿里云 vs AWS vs Azure vs GCP

本文最有价值的部分，就是这张“罗塞塔石碑”：将每个阿里云产品精准对应到你已知的服务。描述力求简洁——每项一句话——因为你不需要长篇大论才能理解“OSS 就是 S3”。

![阿里云服务地图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_service_map.png)

### 计算

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **ECS** (Elastic Compute Service) | EC2 | Virtual Machines | Compute Engine | 虚拟机，基础中的基础。 |
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | 无服务器函数，事件驱动。 |
| **ACK** (Container Service for Kubernetes) | EKS | AKS | GKE | 托管 Kubernetes。 |
| **SAE** (Serverless App Engine) | App Runner | Container Apps | Cloud Run | 无服务器容器，无需管理 K8s。 |
| **ECI** (Elastic Container Instance) | Fargate | Container Instances | Cloud Run Jobs | 无服务器容器实例，无需集群。 |
| **Batch Compute** | AWS Batch | Azure Batch | Cloud Batch | 托管批作业调度。 |

### 存储

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **OSS** (Object Storage Service) | S3 | Blob Storage | Cloud Storage | 对象存储，存文件、取文件。 |
| **NAS** (Network Attached Storage) | EFS | Azure Files | Filestore | POSIX 兼容的共享文件系统。 |
| **CPFS** (Cloud Parallel File System) | FSx for Lustre | — | — | 高性能并行文件系统，适用于 HPC/AI。 |
| **Block Storage** (EBS equivalent) | EBS | Managed Disks | Persistent Disk | 挂载到 ECS 实例的块存储。 |
| **Tablestore** | DynamoDB | Table Storage | Bigtable | 宽列 NoSQL，适合时间序列场景。 |

### 数据库

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **RDS** | RDS | Azure SQL / Azure DB | Cloud SQL | 托管 MySQL、PostgreSQL、SQL Server、MariaDB。 |
| **PolarDB** | Aurora | — | AlloyDB | 云原生分布式关系数据库。 |
| **Lindorm** | DynamoDB + Timestream | Cosmos DB | Bigtable + BigTable | 多模型数据库（宽列、时间序列、搜索）。 |
| **AnalyticDB (ADB)** | Redshift | Synapse | BigQuery | 云数据仓库，列式存储，MPP 架构。 |
| **Tair** (formerly Redis Enterprise) | ElastiCache | Azure Cache for Redis | Memorystore | 托管 Redis 兼容内存存储。 |
| **MongoDB** (ApsaraDB for MongoDB) | DocumentDB | Cosmos DB (Mongo API) | — | 托管 MongoDB。 |

### 网络

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **VPC** | VPC | Virtual Network | VPC | 虚拟私有网络，基础设施基石。 |
| **SLB** (Server Load Balancer) | ALB / NLB | Load Balancer | Cloud Load Balancing | 四层 + 七层负载均衡。 |
| **ALB** (Application Load Balancer) | ALB | Application Gateway | — | 七层负载均衡（较新产品）。 |
| **CDN** | CloudFront | Azure CDN | Cloud CDN | 内容分发网络。 |
| **DCDN** (Dynamic CDN) | CloudFront + Global Accelerator | Front Door | — | 动态内容加速 + 边缘计算。 |
| **CEN** (Cloud Enterprise Network) | Transit Gateway | Virtual WAN | Cloud Interconnect | 多 VPC、多地域网络骨干。 |
| **NAT Gateway** | NAT Gateway | NAT Gateway | Cloud NAT | 私有实例访问公网的出口。 |
| **EIP** (Elastic IP) | Elastic IP | Public IP | Static IP | 静态公网 IP 地址。 |
| **PrivateLink** | PrivateLink | Private Link | Private Service Connect | 私有端点访问服务，无需经过公网。 |

### AI 与机器学习

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **PAI** (Platform for AI) | SageMaker | Azure ML | Vertex AI | 完整 ML 平台：Notebook、训练、推理。 |
| **DashScope** | Bedrock | Azure OpenAI Service | Vertex AI (Gemini) | 模型 API 网关（Qwen、万相、Embeddings）。 |
| **Bailian (百炼)** | Bedrock console | Azure AI Studio | — | 模型管理控制台，RAG 构建器，微调 UI。 |
| **OpenSearch** | OpenSearch Service | Azure AI Search | Vertex AI Search | 托管搜索服务，支持 AI 排序与检索。 |

### 安全与身份

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **RAM** (Resource Access Management) | IAM | Entra ID (Azure AD) | Cloud IAM | 用户、角色、策略管理。 |
| **KMS** (Key Management Service) | KMS | Key Vault | Cloud KMS | 加密密钥管理。 |
| **WAF** (Web Application Firewall) | WAF | Azure WAF | Cloud Armor | Web 应用防火墙。 |
| **Security Center** | GuardDuty + Inspector | Defender for Cloud | Security Command Center | 威胁检测与漏洞扫描。 |
| **SSL Certificates** | ACM | App Service Certificates | Certificate Manager | 托管 TLS/SSL 证书。 |
| **ActionTrail** | CloudTrail | Activity Log | Cloud Audit Logs | API 审计日志。 |

### 可观测性

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **SLS** (Simple Log Service) | CloudWatch Logs | Azure Monitor Logs | Cloud Logging | 日志收集、索引与分析。 |
| **CloudMonitor** | CloudWatch Metrics | Azure Monitor | Cloud Monitoring | 基础设施指标与告警。 |
| **ARMS** | X-Ray + CloudWatch APM | Application Insights | Cloud Trace | 应用性能监控与分布式追踪。 |
| **Grafana Service** | Amazon Managed Grafana | Azure Managed Grafana | — | 托管 Grafana 仪表盘。 |

### 无服务器与事件驱动

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **Function Compute (FC)** | Lambda | Azure Functions | Cloud Functions | 无服务器函数。 |
| **EventBridge** | EventBridge | Event Grid | Eventarc | 事件总线，支持路由与过滤。 |
| **Message Queue (RocketMQ)** | SQS + SNS | Service Bus | Pub/Sub | 分布式消息队列（阿里自研）。 |
| **Kafka (ApsaraDB for Kafka)** | MSK | Event Hubs (Kafka) | Managed Kafka | 托管 Apache Kafka。 |

### 基础设施即代码

| 阿里云 | AWS | Azure | GCP | 功能说明 |
|---|---|---|---|---|
| **Terraform** (alicloud provider) | Terraform (aws provider) | Terraform (azurerm) | Terraform (google) | 使用官方 Provider 的 HashiCorp Terraform。 |
| **ROS** (Resource Orchestration Service) | CloudFormation | ARM/Bicep | Deployment Manager | 原生基于模板的 IaC。 |
| **Pulumi** (alicloud provider) | Pulumi (aws) | Pulumi (azure) | Pulumi (gcp) | 支持 Python/TS/Go 的编程式 IaC。 |

产品确实繁多，但你不必死记硬背这张表。把它加入书签，下次在文档中看到陌生缩写时再回来查即可。最关键的心智模型是：**阿里云几乎为每个 AWS 服务都提供了接近 1:1 的对应产品**，外加一些独特优势（如 PolarDB、DashScope、MaxCompute），也存在少量短板（例如缺乏像 AWS Organizations 那样成熟的多账号治理方案，且英文 SDK 文档通常比中文版滞后数周甚至数月）。

## 账号搭建：从零到控制台

注册阿里云账号不像 AWS 那么直接。有个额外步骤，几乎每次都会让初次接触的海外开发者栽跟头：**实名认证**（实名认证）。

### 第一步：注册

访问 [aliyun.com](https://www.alibabacloud.com/) 完成注册。国际手机号和邮箱均可使用。国际站（alibabacloud.com）支持信用卡支付；中国站（aliyun.com）则需绑定支付宝或中国大陆银行卡。

> **该选哪个站？** 如果你的业务面向中国大陆用户，请使用中国站（aliyun.com）；若面向国际用户，则用国际站（alibabacloud.com）。两者账号独立、账单独立、地域也基本隔离。你可以同时拥有两个账号，但建议选定一个作为主账号。本系列默认采用中国站，因为完整的产品目录仅在此提供。

### 第二步：实名认证

中国法规要求所有云服务必须完成实名认证，这是监管强制要求，而非阿里云自行设定。AWS 中国同样如此。流程如下：

- **个人账号**：上传身份证件照片（中国大陆居民用身份证，外籍人士用护照），审核通常需 1–3 个工作日。
- **企业账号**：需上传营业执照、法人身份证，有时还需银行打款验证，耗时约 3–5 个工作日。

如果只是学习用途，个人账号完全足够，后续可随时升级为企业账号。两者主要区别在于发票类型（个人 vs 企业）和消费额度上限。

### 第三步：启用 MFA

这一步务必立即执行。控制台 `home.console.aliyun.com` 掌控着你的账单、数据和基础设施权限。请为根账号启用 MFA：

1. 进入 **账号管理** > **安全设置**
2. 启用 **虚拟 MFA 设备**
3. 使用认证器应用（如 Google Authenticator、1Password 等）扫描二维码
4. 输入两个连续生成的验证码完成确认

### 第四步：创建 RAM 管理员用户

切勿使用根账号进行日常操作。这与 AWS 的最佳实践一致——根账号仅用于账单管理和紧急恢复，其余所有操作均应通过 RAM 完成。

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

务必将 AccessKey 信息保存在安全位置（推荐密码管理器，切勿写在便利贴上）。随后重新配置 CLI，改用 RAM 用户：

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

从此刻起，除非处理账单或从锁定状态恢复，否则永远不要再使用根账号凭证。

### 第五步：设置账单预警

前往 **费用中心** > **消费预警**，设定月度预算。建议学习阶段设为 100 CNY/月。阿里云免费额度足以支撑大多数实验，但若忘记关闭 ECS 实例或遗留一台 GPU 虚拟机，一夜之间就可能耗尽预算。

```bash
# 快速检查：这个月我花了多少？
aliyun bssopenapi QueryAccountBalance
```

### 第六步：了解免费额度

阿里云为新账号提供为期 12 个月的免费试用，包含：

- **ECS**：1 台实例（1 vCPU, 1 GB RAM），可用 3 个月——足够完成本系列实验
- **OSS**：5 GB 存储空间 + 5 GB 出站流量 / 月
- **RDS**：1 台微型实例，可用 1 个月
- **Function Compute**：100 万次请求 + 40 万 GB-秒 / 月
- **SLS**：每日 500 MB 日志摄入量
- **各类 AI 服务**：各模型均有免费调用额度（如 DashScope、PAI 等）

具体权益每季度可能调整，请访问 [免费试用页面](https://free.aliyun.com/) 查看最新详情。

## 区域与可用区

阿里云在中国大陆的地域数量超过任何其他云厂商。这一点至关重要，因为中国互联网的拓扑结构决定了：深圳用户访问北京服务器的延迟，远高于访问深圳本地服务器——物理距离叠加网络策略，双重影响体验。

![阿里云全球区域与可用区](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_region_map.png)

### 中国区

| Region ID | Location | Good for |
|---|---|---|
| `cn-hangzhou` | 杭州，浙江 | 默认地域，阿里总部所在地，服务覆盖最全。 |
| `cn-shanghai` | 上海 | 金融业务、AI/ML（GPU 库存最充足）。 |
| `cn-beijing` | 北京 | 政府、媒体及华北用户。 |
| `cn-shenzhen` | 深圳，广东 | 华南地区，毗邻香港。 |
| `cn-zhangjiakou` | 张家口，河北 | 冷数据存储、批处理任务（价格最低）。 |
| `cn-hohhot` | 呼和浩特，内蒙古 | 大数据处理、成本敏感型负载。 |
| `cn-wulanchabu` | 乌兰察布，内蒙古 | AI 训练、大型 GPU 集群。 |
| `cn-chengdu` | 成都，四川 | 西南地区覆盖。 |
| `cn-guangzhou` | 广州，广东 | 华南地区替代深圳的选择。 |
| `cn-nanjing` | 南京，江苏 | 华东地区冗余部署。 |
| `cn-fuzhou` | 福州，福建 | 面向海峡对岸（毗邻台湾）。 |
| `cn-heyuan` | 河源，广东 | 华南地区成本优化选项。 |

### 国际区

| Region ID | Location | Good for |
|---|---|---|
| `ap-southeast-1` | 新加坡 | 东南亚，且是离中国大陆延迟最低的非境内区域。 |
| `ap-northeast-1` | 东京 | 日本、韩国用户。 |
| `ap-south-1` | 孟买 | 印度及南亚市场。 |
| `ap-southeast-5` | 雅加达 | 印尼市场。 |
| `ap-southeast-3` | 吉隆坡 | 马来西亚市场。 |
| `eu-central-1` | 法兰克福 | 欧洲市场。 |
| `eu-west-1` | 伦敦 | 英国市场。 |
| `us-east-1` | 弗吉尼亚 | 美国东海岸。 |
| `us-west-1` | 硅谷 | 美国西海岸。 |
| `ap-southeast-2` | 悉尼 | 澳大利亚及大洋洲。 |
| `me-east-1` | 迪拜 | 中东市场。 |

### 如何选择

以下决策树源于真实生产经验：

1. **用户在哪里？** 优先选择离用户最近的地域，延迟压倒一切。
2. **是否需遵守中国数据驻留法规？** 若是，必须选择中国大陆地域（`cn-*`）。这些区域的数据未经跨境审批不得出境。
3. **是否需要 GPU？** 上海和杭州 GPU 库存最充足；乌兰察布价格最低，但网络延迟较高。
4. **是否为成本敏感的批处理任务？** 张家口和呼和浩特的计算资源便宜 15–30%。
5. **默认选择？** 中国大陆业务选 `cn-hangzhou`，国际业务选 `ap-southeast-1`。

> **实战建议：** 每个地域包含多个可用区（AZ），标记为 `a`、`b`、`c` 等。任何关键业务都应至少跨两个 AZ 部署。AZ 故障虽罕见，但确实会发生——我在 18 个月内就遇到过 `cn-hangzhou-h` 两次服务降级。

## 理解计费模式

阿里云的计费模式比 AWS 更灵活，但也更容易让人困惑。共有四种购买方式，你必须全部了解。

![阿里云计费模式对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_billing_comparison.png)

### 按量付费（PAYG, 按量付费）

顾名思义，用多少付多少，计算资源按秒计费，存储和流量按 GB 计费。这是默认模式，也是学习阶段的最佳选择：无需承诺、无需预付、随时可停。

缺点是单价最高。以 `cn-hangzhou` 的 ECS `ecs.c7.large`（2 vCPU, 4 GB）为例：
- 按量付费：约 0.35 CNY/小时（24/7 运行约 252 CNY/月）
- 包年包月：约 125 CNY/月（1 年承诺）

为灵活性付出的溢价高达 50%。

### 包年包月（Subscription, 包年包月）

承诺使用 1 个月至 3 年不等，承诺期越长，折扣越大。3 年期 ECS 实例比按量付费便宜 40–60%。但需预付费用（或开启自动续费），即使停用实例，费用仍会照常扣除。

适用场景：确定会长期运行（至少一年）的生产服务器。

### 抢占式实例（Preemptible / Spot Instances, 抢占式实例）

概念与 AWS Spot Instances 相同：以 10–90% 的折扣竞价使用闲置资源，但可能被提前 5 分钟回收。阿里云的抢占市场不如 AWS 激烈，因此折扣往往更深、中断频率更低——至少我在 `cn-shanghai` 的体验如此。

适用场景：批处理任务、支持 checkpoint 的训练作业、CI/CD 流水线。

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

**节省计划** 类似 AWS Savings Plans：承诺每小时消费一定金额，即可在所有计算资源（ECS、ECI、ACK）上享受折扣，不受实例类型或地域限制。比包年包月更灵活，但需准确预估基线用量。

**资源包**（资源包）是针对特定服务的预付套餐。例如：500 GB OSS 存储包年享 7 折，1000 万次 Function Compute 调用享 6 折。建议在用量稳定后再购买，学习阶段无需考虑。

### 账单控制台

账单管理入口为 `usercenter2.aliyun.com`，关键页面包括：

- **账单** > **账单概览**：按产品划分的月度汇总
- **账单** > **账单明细**：实例级别的详细消费记录
- **成本管理** > **成本分析**：按产品、地域、标签拆解成本
- **预算** > **预算预警**：设置消费阈值告警

我每周都会查看。生产环境中最大的意外支出，曾是一个遗忘在测试 VPC 中的 NAT Gateway——每天默默消耗 15 CNY，而我根本没在使用它。若从第一天就启用标签和预算告警，本可避免这一损失。

## 你的第一台 ECS 实例

背景介绍到此为止，现在动手部署！

![阿里云免费额度一览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_free_tier.png)

本文启动一台运行 Alibaba Cloud Linux 3 的 ECS 实例，通过 SSH 登录，安装 nginx，并托管一个测试页面。整个过程约需 15 分钟。

### 前置条件

- 已完成实名认证的阿里云账号（前文已配置）
- 已在 `aliyun` CLI 中配置 RAM 管理员用户 AccessKey（前文已配置）
- 已选定地域（本文使用 `cn-hangzhou`）

### 第一步：创建 VPC 和 VSwitch

每台 ECS 实例必须位于 VPC 内。你需要先创建一个 VPC 和至少一个 VSwitch（子网）。

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

安全组相当于虚拟防火墙。我们需放行 SSH（端口 22）仅限你的 IP，HTTP（端口 80）对全网开放。

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

ECS 切勿使用密码登录，务必使用密钥对。

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

参数说明：

- **`ecs.c7.large`**：2 vCPU，4 GB 内存，学习足够，成本可控。`c7` 系列为 Intel Ice Lake 架构，通用负载的理想选择。
- **`cloud_essd`**：增强型 SSD，新实例默认选项，涉及磁盘读写的场景首选。旧版 `cloud_efficiency`（高效云盘）虽便宜，但性能明显落后。
- **`InternetMaxBandwidthOut 5`**：5 Mbps 公网带宽。设为非零值会自动分配公网 IP，无需额外申请 EIP。学习环境足够；生产环境建议使用 EIP 或 SLB。
- **`PostPaid`**：按量付费。停止实例后计算资源不再计费（磁盘仍需少量费用）。

### 第五步：获取公网 IP 并 SSH 登录

```bash
# Wait for the instance to start (usually 30-60 seconds)
aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0] | "\(.Status) \(.PublicIpAddress.IpAddress[0])"'
```

当实例状态变为 `Running` 并显示公网 IP 后：

```bash
# SSH in
ECS_IP=$(aliyun ecs DescribeInstances \
  --RegionId cn-hangzhou \
  --InstanceIds "['$INSTANCE_ID']" \
  | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

ssh -i ~/.ssh/fullstack-key.pem root@$ECS_IP
```

你应该能看到 Alibaba Cloud Linux 3 的欢迎横幅。恭喜！你已成功启动一台云服务器。

### Terraform 方案

如果你觉得“这么多命令式 CLI 操作根本无法可靠复现”——你说得对。同样的操作，用 Terraform 只需：

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

三条命令替代十二步操作，状态自动跟踪，配置可版本控制，用完执行 `terraform destroy` 即可一键清理。本文在 [第 7 篇](/zh/aliyun-fullstack/07-observability) 深入讲解 Terraform，若想深入探索，还可参考完整的 [Terraform for AI Agents series](/zh/terraform-agents/01-why-terraform-for-agents/)。

## 首次部署检查清单

ECS 实例已运行，现在让它真正发挥作用。

![从注册到访问的首次部署流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_deployment_flow.png)

### 安装必备软件包

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

### 部署测试页面

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

在本地终端执行：

```bash
curl http://$ECS_IP
```

应能看到 HTML 内容。若连接超时，请检查：
1. 安全组是否开放了 80 端口？
2. nginx 是否正在运行？（在实例上执行 `systemctl status nginx`）
3. 公网 IP 是否正确？（通过 `aliyun ecs DescribeInstances` 确认）

### 启用基础监控

ECS 实例默认已接入云监控，但监控代理可能未安装：

```bash
# On the ECS instance — install CloudMonitor agent
ARGUS_VERSION=3.5.8
curl -sL "https://cms-agent-cn-hangzhou.oss-cn-hangzhou.aliyuncs.com/cms-go-agent/${ARGUS_VERSION}/CmsGoAgent.linux-amd64.tar.gz" \
  | tar xzf - -C /usr/local/

/usr/local/CmsGoAgent/CmsGoAgent start
/usr/local/CmsGoAgent/CmsGoAgent status
```

随后进入云监控控制台，几分钟内即可看到 CPU、内存、磁盘和网络等指标。

### 实验结束后及时清理

切勿忘记在实验结束后停止或释放实例。一台运行中的 `ecs.c7.large` 约 0.35 元/小时，若持续运行，月费用将达 252 元。

```bash
# Stop the instance (you still pay for the disk, ~0.5 CNY/day for 40 GB ESSD)
aliyun ecs StopInstance --InstanceId "$INSTANCE_ID"

# Or release it entirely (all data is lost)
aliyun ecs DeleteInstance --InstanceId "$INSTANCE_ID" --Force true

# If using Terraform, just:
terraform destroy
```

## 我们要构建的架构

本文是阿里云全栈系列的开篇，共十二篇文章。完整路线图如下：

![12 篇系列文章路线图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/01-ecosystem-map/01_architecture_roadmap.png)

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

每篇文章都以前一篇为基础。到第 12 篇结束时，你将拥有一个完整的生产级阿里云架构，涵盖网络、安全、可观测性和 CI/CD。

部分主题在本博客已有深度系列：

- **PAI (articles 1–5)**：如需全面了解 GPU Notebook、分布式训练和模型服务，请参阅 [PAI series](/zh/aliyun-pai/01-platform-overview/)。
- **Bailian and DashScope (articles 1–5)**：关于 LLM API、多模态模型、TTS 和视频生成，请参阅 [Bailian series](/zh/aliyun-bailian/01-platform-overview/)。
- **Terraform (articles 1–8)**：针对 AI Agent 系统的基础设施即代码实践，请参阅 [Terraform for AI Agents series](/zh/terraform-agents/01-why-terraform-for-agents/)。

本全栈系列是广度优先的导览，上述系列则是深度优先的钻研。两者互补——每讲到一个主题，我会指引你阅读对应的深度文章。

## 总结

1. **阿里云与 AWS 的服务基本一一对应。** 若你熟悉 AWS，其实已掌握阿里云 80% 的内容。剩余 20% 主要是命名差异、控制台布局和中国特有功能。直接将本文的服务对照表当作你的“罗塞塔石碑”即可。

2. **账号注册多一个关键步骤。** 实名认证是强制要求，需 1–3 天完成。务必提前启动，不要等到部署时才处理。注册后立即创建 RAM 管理员用户，日常操作切勿使用根账号。

3. **地域选择比在 AWS 上更重要。** 中国网络拓扑复杂，不同地域到用户的延迟差异显著。中国大陆业务默认选 `cn-hangzhou`，国际业务选 `ap-southeast-1`。同时注意数据合规要求：中国大陆数据必须留在境内。

4. **学习用按量付费，生产用包年包月。** 两者价差可达 40–60%。负载稳定前不要盲目承诺长期合约。务必设置账单告警——一个被遗忘的 NAT Gateway 或闲置 GPU 实例，会悄无声息地掏空你的账户。

5. **从第一天就使用 Terraform。** 上述 CLI 演示仅为教学目的，实际生产中绝不应命令式创建基础设施。本系列所有资源均提供 Terraform 实现。未来的你一定会感谢现在的决定。

下一篇：[文章 02 — ECS 深挖](/zh/aliyun-fullstack/02-ecs-compute)，本文深入探讨实例选型、镜像管理、磁盘性能特性，以及对生产负载至关重要的部署策略。
