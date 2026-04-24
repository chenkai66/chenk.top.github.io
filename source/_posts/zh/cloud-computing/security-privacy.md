---
title: "云安全与隐私保护"
date: 2024-06-26 09:00:00
tags:
  - 云计算
  - 云安全
  - IAM
  - 数据加密
  - 零信任
  - 合规
categories: 云计算
series:
  name: "云计算"
  part: 5
  total: 8
lang: zh-CN
mathjax: false
description: "工程师视角的云安全实战：共担责任、可扩展的 IAM、静态/传输/使用中加密、零信任、合规框架，以及一份可以反复演练的事件响应流程。"
disableNunjucks: true
series_order: 6
---

2019 年 Capital One 泄露了一亿条客户数据。攻击链很短：一个配置错误的 WAF 允许了对 EC2 元数据端点的服务端请求伪造（SSRF），元数据端点交出了 IAM 临时凭证，而那个 IAM 角色对账户里所有 S3 存储桶都拥有 `s3:*` 权限。一处错配、一个权限过宽的角色、一条没人写过的规则。账单（不算法律费用）：超过 8000 万美元。

之后几乎每一起公开的云安全事件都是同一种形态。不是零日漏洞、也不是国家级恶意软件——是配置错误，没人发现，直到数据已经出现在 Pastebin 上。云安全工程师的工作因此并不在于发明新的密码学，而在于系统性地消除"小错误演变成 8000 万美元事故"的条件。

本文从共担责任契约一路讲到事件响应，每一层都附上代码、架构图、失效模式和可演练的运行手册。

## 你将学到

- 共担责任模型，以及它如何在 IaaS / PaaS / SaaS 之间漂移
- 可扩展的 IAM：身份、组、角色、策略，以及那些能通过审计的实战模式
- 静态、传输中、使用中三种数据状态的加密——包括几乎所有人都会犯错的细节
- DDoS 防护与 WAF，以及如何在不误杀正常用户的前提下做限流
- 把零信任落实为一组具体控制项，而非厂商口号
- 五大合规框架（SOC 2、HIPAA、GDPR、PCI DSS、ISO 27001）的真实要求对比
- 一个可以在事故来临前反复演练的事件响应循环

## 前置知识

- 网络基础（TCP、TLS、DNS、防火墙）
- 至少熟悉一家云厂商的 IAM 控制台
- 建议先阅读本系列前 4 篇

---

## 1. 共担责任，把话说清楚

![共担责任模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/security-privacy/fig1_shared_responsibility.png)

每家云厂商都会发布共担责任模型。他们之所以发布，是因为每次事故复盘的第一句争论都一样：**这事到底归谁守。**答案取决于你用的是哪一层服务。

| 层级       | IaaS（EC2） | PaaS（App Engine） | SaaS（Gmail） |
| ---------- | ----------- | ------------------ | ------------- |
| 数据与访问 | **你**      | **你**             | **你**        |
| 应用代码   | **你**      | **你**             | 厂商          |
| 运行时/OS  | **你**      | 厂商               | 厂商          |
| 虚拟化     | 厂商        | 厂商               | 厂商          |
| 网络基础设施 | 厂商      | 厂商               | 厂商          |
| 物理       | 厂商        | 厂商               | 厂商          |

图里很容易被忽略的三件事。第一，**数据与访问控制永远是你的责任**——即便用 SaaS，谁能进你的 Salesforce 租户也是你说了算。第二，**厂商服务的配置也是你的责任**——AWS 给你 S3，但桶策略、加密设置、公开访问拦截开关是你的事，而且这些默认值在很长一段时间里都是"开放"。第三，**只有当你"按规矩"用厂商服务时，厂商那部分责任才生效**；自己在 EC2 上跑一套数据库，打补丁、备份、HA 又全部回到你头上。

一句更有用的概括：**厂商守底座；你能配的，必须正确地配。**

## 2. 你真正会遇到的威胁

Verizon DBIR 和 Mandiant M-Trends 报告每年的云事故 Top 原因高度一致，按发生频次排序大致如下：

- **对象存储配置错误。**公开的 S3 / GCS 桶，里面是客户 PII、源代码、备份。
- **凭证泄露或长期不轮换。**API 密钥提交进 GitHub、烤进 AMI、躺在 Slack 历史里。
- **IAM 权限过宽。**只需要某一个桶的 `s3:GetObject`，结果在控制台向导里点了 `s3:*`，对账户里所有桶生效。
- **应用未打补丁。**库、容器镜像、操作系统包里的公开 CVE。
- **供应链攻击。**恶意依赖（`event-stream`）、被污染的基础镜像、抢注的同名包。
- **会话令牌被盗。**MFA 验证码被钓鱼，或浏览器侧的 token 被窃取后异地重放。

这份名单上没有任何针对 AES 或 TLS 的密码学攻击。防御方的杠杆点在 IAM 卫生、配置管理、检测能力——而不在自己发明新算法。

## 3. IAM：唯一不能搞砸的子系统

![IAM 模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/security-privacy/fig2_iam_model.png)

### 3.1 模型

成熟的 IAM 系统都有同样的四块原语：

- **Identity（身份）**——人（Alice）或机器（CI Runner）。身份通过密码 + MFA、证书、Workload Identity 令牌等方式证明自己。
- **Group（组）**——一组身份的静态集合（"工程组"、"财务组"）。组只是寻址工具，本身不持有任何权限。
- **Role（角色）**——一组命名的权限。身份或组可以**扮演**一个角色；AWS 上每个角色还有一份**信任策略（trust policy）**，声明谁有资格扮演它。
- **Policy（策略）**——真正的授权语句：允许 / 拒绝、对哪些 Action、针对哪些资源、在什么条件下。

箭头方向很关键：**身份 -> 组 -> 角色 -> 策略 -> 操作**。权限只向前流动。审计任何一次访问，都可以从资源端反向走完这条链。

### 3.2 六条经得起实战的规则

1. **最小权限，再迭代。**从零开始，只授予恰好够完成任务的权限。如果某个合理的操作真的失败了，**精准**放宽——不要 `s3:*`。
2. **凡是跑代码的地方，都用角色，不用长期用户。**EC2 实例 Profile、Lambda 执行角色、EKS 的 IRSA、GKE 的 Workload Identity——它们都给出短期、自动轮换的临时凭证。
3. **每个人类身份必须 MFA。**不要给"多人共用的服务账号"开例外——这种账号本来就不该存在。
4. **轮换、撤销、过期。**90 天没换的 API 密钥就是异味。员工一离职，他的凭证就开始倒计时。
5. **权限边界 + SCP。**给团队自己创建角色的自由，但用一份组织级策略**封住口子**，让危险组合永远无法生效。
6. **每次授权和每次使用都要有日志。**CloudTrail、Cloud Audit Logs、Azure Activity Log——集中存储、防篡改、至少保留一年。

### 3.3 一份真正最小权限的策略

只对一个桶只读，限制企业网段，并且要求当前会话**确实**完成了 MFA：

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "ReadOnlyFromOfficeWithMfa",
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::reports-prod",
      "arn:aws:s3:::reports-prod/*"
    ],
    "Condition": {
      "IpAddress":  { "aws:SourceIp": "203.0.113.0/24" },
      "Bool":       { "aws:MultiFactorAuthPresent": "true" },
      "NumericLessThan": { "aws:MultiFactorAuthAge": "3600" }
    }
  }]
}
```

`MultiFactorAuthAge` 这个条件被严重低估——它强制 MFA 证据必须**新鲜**，而不是"他六小时前进控制台时 MFA 过一次"。

### 3.4 组织级护栏

绑在组织根节点上的 SCP 不能被任何账号级角色覆盖。把它用在"爆炸半径"决策上：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyOutsideApprovedRegions",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": ["us-east-1", "us-west-2", "eu-west-1"]
        }
      }
    },
    {
      "Sid": "DenyRootUser",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:PrincipalType": "Root" }
      }
    }
  ]
}
```

这条策略一次性拒绝了 root 用户的任何动作、以及在你不运营的 region 里的任何动作。两类事故被一并堵死：数据驻留违规，以及攻击者在偏远 region 起 GPU 挖矿。

### 3.5 反复出现的 IAM 错误

- 任何非 Deny 语句里出现 `"Action": "*"` 或 `"Resource": "*"`。几乎一定权限过宽。
- `iam:PassRole` 配 `Resource: "*"`——这是一行就能完成的特权提升。
- 长期 Access Key 烤进 AMI、容器镜像，或者塞进提交到 Git 的 `.env`。
- "临时"管理员授权没设过期时间，结果变成永久。
- 多服务共用一个 Service Account，导致没人有权也没人敢轮换。

## 4. 加密：静态、传输中、使用中

![数据三种状态](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/security-privacy/fig3_encryption_layers.png)

数据有三种状态，每一种都需要不同的保护机制，并且各自针对的威胁不同。

### 4.1 静态

威胁：有人把硬盘、快照、备份带磁带走；或者一个本不该看到原始字节的进程读到了它。防御：用你**真正控制**密钥的对称加密。

- 应用层加密用 **AES-256-GCM**。
- 整盘 / 卷加密（EBS、PD）用 **AES-XTS**。
- 托管数据库用 **TDE（透明数据加密）**。
- 对象存储默认**服务端加密**。AWS 上记得开 `BucketKeyEnabled: true`，KMS 调用费降约 99%。

用云 KMS 或 HSM 托管密钥。**不要**自己实现密钥轮换。

```python
import boto3

kms = boto3.client("kms")

# 信封加密：KMS 加密一个小的数据密钥；
# 你在本地用这个数据密钥加密真正的负载。
def encrypt_blob(plaintext: bytes, key_id: str) -> dict:
    resp = kms.generate_data_key(KeyId=key_id, KeySpec="AES_256")
    plaintext_dk, encrypted_dk = resp["Plaintext"], resp["CiphertextBlob"]

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os
    nonce = os.urandom(12)
    ciphertext = AESGCM(plaintext_dk).encrypt(nonce, plaintext, associated_data=None)

    # 丢弃明文数据密钥；只持久化加密后的版本。
    return {"nonce": nonce, "ciphertext": ciphertext, "wrapped_key": encrypted_dk}
```

每一项"透明"的云加密能力，背后基本都是这个信封模式。理解它你就能把成本（每个数据密钥一次 KMS 调用，不是按字节计费）和密钥轮换（重新封装数据密钥，而不是重新加密 PB 级数据）想清楚。

### 4.2 传输中

威胁：链路上的中间人、被攻陷的网络设备做包捕获、网格里的恶意 sidecar。防御：**正确**使用 TLS。

```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;

    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers on;

    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1h;
    ssl_session_tickets off;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options    "nosniff"  always;
    add_header X-Frame-Options           "DENY"     always;
}
```

三个常被忽略的细节：

- **在负载均衡层面禁用 TLS 1.0 / 1.1。**所有现行合规体系都已禁止，且存在已知弱点。
- **HSTS 配长 `max-age`**——这是抵御 SSL-strip 降级攻击的唯一有效手段。
- **服务间 mTLS。**集群内服务 A 必须拒绝服务 B，除非 B 出示由你内部 CA 签发的证书。Service Mesh（Istio、Linkerd）让这变成一行 YAML。

### 4.3 使用中

最难的一类。数据在内存里被解密以便处理，意味着同一台机器上权限足够的进程能读到。今天可用的三类缓解：

- **机密 VM**：AWS Nitro Enclaves、GCP Confidential VMs、Azure Confidential Computing。Hypervisor 看不到客户机内存，并通过硬件根度量进行远程证明。
- **硬件 enclave**：Intel SGX、AMD SEV-SNP。进程内一块小的可信区域，明文数据只在里面出现。
- **同态加密 / 安全多方计算**。数学上漂亮，但目前比明文计算慢 1000 倍到 100 万倍——只在很窄的场景（隐私集合求交、某些聚合）落地。

对大多数团队，"使用中"保护其实是间接达成的：**收紧明文处理面、隔离它、审计它**。

## 5. DDoS 防护与 WAF

### 5.1 三类攻击

| 类别 | 手法 | 例子 | 打到哪里 |
|------|------|------|----------|
| 流量型 | 打满带宽 | UDP / DNS 反射放大 | 网络管道 |
| 协议型 | 耗尽连接状态表 | SYN 洪水、ACK 洪水 | LB、操作系统 |
| 应用层 | 强迫服务做昂贵的工作 | HTTP 洪水、Slowloris、GraphQL 炸弹 | 应用服务器、数据库 |

真实攻击三种叠加。真实防御也得分层。

### 5.2 防御栈

- **边缘 / 网络层**：AWS Shield Advanced、Cloud Armor、Cloudflare。在流量打进 VPC 之前吸收 L3/L4 洪水。
- **CDN**：缓存静态资源，把对外暴露面扩大很多倍，攻击被稀释。
- **WAF**：拦截 SQLi / XSS / RCE 模式、恶意爬虫，必要时做地理拦截。先用托管规则集打底，再叠加针对自己应用的窄规则。
- **限流**：按 IP、按 token、按路由。能抓凭证撞库、爬数据、慢烧型应用层攻击。
- **应用本身**：输入校验、参数化查询、查询代价上限。**WAF 不是你的输入校验器。**

### 5.3 一份能直接上线的 WAF 规则

```json
{
  "Rules": [
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 10,
      "OverrideAction": { "None": {} },
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "VisibilityConfig": { "SampledRequestsEnabled": true,
                            "CloudWatchMetricsEnabled": true,
                            "MetricName": "common" }
    },
    {
      "Name": "RateLimitPerIp",
      "Priority": 20,
      "Action": { "Block": {} },
      "Statement": {
        "RateBasedStatement": { "Limit": 2000, "AggregateKeyType": "IP" }
      },
      "VisibilityConfig": { "SampledRequestsEnabled": true,
                            "CloudWatchMetricsEnabled": true,
                            "MetricName": "ratelimit" }
    }
  ]
}
```

`Limit` 要仔细调。太严会把走 CGNAT 的自家 App 一并拦掉。**先用 `Count` 模式跑一周再切到 `Block`**。

## 6. 安全日志与检测

### 6.1 应该记什么

按优先级排序的五类：

1. **认证**——登录、失败、MFA 挑战、密码重置。
2. **授权**——每一次允许和每一次拒绝，连同做出决策的策略。
3. **数据访问**——对敏感资源的读写（PII 字段、KMS 密钥、密钥库）。
4. **配置变更**——IAM、安全组、KMS、IaC apply。
5. **网络**——VPC Flow Logs、DNS 查询日志。

### 6.2 一条 CloudTrail 事件，逐字段读

```json
{
  "eventTime": "2024-01-15T14:30:00Z",
  "eventSource": "s3.amazonaws.com",
  "eventName": "GetObject",
  "userIdentity": {
    "type": "AssumedRole",
    "arn": "arn:aws:sts::111122223333:assumed-role/AnalystRole/Alice",
    "sessionContext": { "mfaAuthenticated": "true" }
  },
  "sourceIPAddress": "203.0.113.12",
  "userAgent": "aws-sdk-python/1.28",
  "requestParameters": { "bucketName": "reports-prod", "key": "q4-revenue.csv" },
  "additionalEventData": { "bytesTransferredOut": 38421 }
}
```

SIEM 规则真正会消费的字段：`userIdentity.arn`、`mfaAuthenticated`、`sourceIPAddress`、`eventName`、`requestParameters` 与响应状态。值得告警的模式：

- 来自非运营国家的 `ConsoleLogin`。
- 自动化白名单之外的 `iam:CreateUser` / `iam:CreateAccessKey`。
- 某个角色突然出现 `s3:GetObject` 调用量暴涨。
- 任何 `kms:Disable*` / `kms:ScheduleKeyDeletion`。
- 之前从未启过 GPU 实例的角色突然 `ec2:RunInstances` 一台 GPU。

### 6.3 管道与保留

- 集中到 SIEM（Security Hub、Splunk、Elastic Security、Chronicle）。
- 日志桶加密，并对所有人——包括安全团队自己的角色——拒绝 `s3:DeleteObject`。**可被改写的日志不构成证据。**
- 90 天热存，1–7 年冷存，按行业合规要求。
- **在用到之前**就把仪表盘搭好；凌晨三点的事故不是学 Lucene 语法的好时机。

## 7. 零信任，落到具体动作

![零信任架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/security-privacy/fig4_zero_trust.png)

"零信任"是个被市场化的术语，但内核很简单：**网络位置不授予任何特权**。你在公司 VPN 里、在生产 VPC 里、在集群网络里——这些事实本身都不构成授权。

实现它的具体控制：

- **每个内部应用前面架一个身份感知代理。**别再说"它只在内网，不需要鉴权"。Google 的 BeyondCorp 是开山之作；Cloudflare Access、AWS Verified Access、Tailscale 提供开箱即用的版本。
- **设备状态作为认证因素。**PDP 检查：笔记本是否加密、是否在 MDM、补丁是否最新、序列号是否在你的资产登记里？任何一项不满足，无论凭证多新，请求都会被拒或被隔离。
- **服务间 mTLS，配 SPIFFE / SPIRE。**每个工作负载有一个加密身份（SPIFFE ID），证书短期有效，策略声明谁能调谁。
- **持续评估。**会话不是"12 小时有效"，而是持续根据信号（IP 变了、地理位置变了、设备脱管）重新评估。可疑会话被降级或撤销。
- **微分段。**一个 Pod 被攻陷不应能横向访问整个集群。NetworkPolicy 把东西向流量限制在声明过的依赖之内。

心智模型：每一次请求都跑过一个 PDP，被问"谁、做什么、从哪里、什么设备、什么上下文、对哪个资源——允许还是拒绝？"，答案被记录，每一次。

## 8. 合规框架

![合规框架](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/security-privacy/fig5_compliance_frameworks.png)

合规不等于安全，但它强制你建立基线控制和留痕。覆盖大多数工程团队义务的五个框架：

| 框架 | 适用范围 | 标志性要求 |
|------|---------|-----------|
| **SOC 2** | 美国服务型组织 | 信任服务标准：安全、可用、处理完整性、机密、隐私。年度审计。 |
| **HIPAA** | 美国医疗 PHI | 与子处理者签 BAA、加密无处不在、6 年审计留存、违规通报。 |
| **GDPR** | 欧盟个人数据，处理地不限 | 合法依据、数据最小化、访问 / 删除权、72 小时违规通报、DPIA。 |
| **PCI DSS** | 支付卡数据 | 网络分段、加密存储、季度扫描、年度渗透测试。 |
| **ISO 27001** | 国际通用 ISMS | 文档化的信息安全管理体系、风险评估、附录 A 控制项、认证审计。 |

### GDPR 工程清单

- 数据资产盘点：每个接触个人数据的系统，都要写明合法依据。
- 主体访问请求接口：30 天内交付该主体的所有数据。
- 删除接口：通过按租户密钥做加密销毁，等价于备份的"密码学粉碎"，是被接受的做法。
- 72 小时违规通报：在用到之前就把通讯路径排练好。
- 任何对高风险数据（位置、健康、生物识别、画像）的新处理活动做 DPIA。
- 处理活动记录（GDPR 第 30 条）——从你的数据资产盘点里直接生成，不要手动维护。

五个框架的共同模式：你**不可能**在最后阶段把它们贴上去。把控制项埋进基础设施代码、审计日志、发布流水线里，审计就退化为文书工作，而不是生死劫。

## 9. 事件响应：那个循环

NIST 的循环（准备 -> 检测 -> 遏制 -> 消除 -> 恢复 -> 复盘）方向正确但偏理论。决定你的事故是 30 分钟的脚注还是 30 天的灾难，实战中只看三件事：

1. **运行手册被排练过。**每季度桌面演练，每年至少一次实战级 game day。
2. **能快速且具备取证能力地隔离。**销毁前先快照；重启前先收内存。
3. **写无指责的事后复盘，行动项真的会落地。**否则同一个事故明年还会再来一遍。

### 一个可复用的遏制脚本

```python
"""被入侵的 EC2 实例：取证快照、隔离、停机。"""
import boto3
from datetime import datetime

ec2 = boto3.client("ec2")

def quarantine(instance_id: str, reason: str) -> dict:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = [{"Key": "incident",  "Value": reason},
           {"Key": "timestamp", "Value": timestamp}]

    # 1. 在做任何其他事情之前，先给每一块挂载卷打快照。
    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    snapshots = []
    for bdm in inst["BlockDeviceMappings"]:
        snap = ec2.create_snapshot(
            VolumeId=bdm["Ebs"]["VolumeId"],
            Description=f"forensic-{instance_id}-{reason}-{timestamp}",
            TagSpecifications=[{"ResourceType": "snapshot", "Tags": tag}],
        )
        snapshots.append(snap["SnapshotId"])

    # 2. 把安全组替换成"什么都不放行"。
    isolation_sg = ec2.create_security_group(
        GroupName=f"quarantine-{instance_id}-{timestamp}",
        Description=f"Incident isolation: {reason}",
        VpcId=inst["VpcId"],
    )["GroupId"]
    ec2.modify_instance_attribute(InstanceId=instance_id, Groups=[isolation_sg])

    # 3. 解绑 IAM 实例 Profile，让凭证无法继续被复用。
    profile = inst.get("IamInstanceProfile", {}).get("Arn")
    if profile:
        association_id = next(
            a["AssociationId"]
            for a in ec2.describe_iam_instance_profile_associations(
                Filters=[{"Name": "instance-id", "Values": [instance_id]}])["IamInstanceProfileAssociations"]
        )
        ec2.disassociate_iam_instance_profile(AssociationId=association_id)

    # 4. 停机（不要 terminate——terminate 会销毁证据）。
    ec2.stop_instances(InstanceIds=[instance_id])

    return {"instance": instance_id, "snapshots": snapshots, "isolation_sg": isolation_sg}
```

顺序很重要：**快照 -> 隔离 -> 撤凭证 -> 停机**。颠倒顺序，就可能在打快照的几秒内被持续外传，或者在 OS 关机序列里丢掉证据。

## 10. 写进 IaC 的安全基线

如果一个疲惫的工程师能在周五下午 `terraform apply` 出一个公开 S3 桶，你的安全基线就一文不值。把基线编进模块和策略即代码（OPA / Sentinel / Checkov）：

```hcl
resource "aws_s3_bucket" "data" {
  bucket = "reports-${var.environment}"

  tags = {
    DataClassification = "confidential"
    Owner              = "analytics"
    Compliance         = "soc2,hipaa"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.reports.arn
    }
    bucket_key_enabled = true   # 把 KMS 调用费用降约 99%
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "data" {
  bucket        = aws_s3_bucket.data.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "reports/"
}
```

CI 里跑一份 Checkov 策略，拒绝任何不带这四个伴随资源的 `aws_s3_bucket`。错误在进生产之前就被拦下。

## 11. 工程师的起飞前检查单

**身份**
- [ ] 全员 MFA，包括 break-glass 账号（验证码放在密封信封里，不是 Slack 里）。
- [ ] 工作负载没有长期 Access Key；用实例 Profile / IRSA / Workload Identity。
- [ ] 季度复审：谁还有管理员权限？还需要吗？
- [ ] SCP / 组织策略已经禁用 root，已经禁用未使用的 region。

**加密**
- [ ] 每个存储服务的"默认加密"开关都在打开。
- [ ] 敏感数据用客户托管 KMS 密钥，开启年度自动轮换。
- [ ] 每一个 LB 都强制 TLS 1.2+，HSTS 已加入预加载。
- [ ] 威胁模型需要时，服务间走 mTLS。

**网络**
- [ ] 安全组默认拒绝；开端口必须在 code review 中说明。
- [ ] 公开应用前面有 WAF；先 count 模式跑一周再切 block。
- [ ] 生产入口开启 DDoS 防护（Shield Advanced / Cloud Armor）。
- [ ] VPC Flow Logs、DNS 查询日志开启并送往 SIEM。

**检测**
- [ ] CloudTrail / Cloud Audit Logs 写入独立账号下的 write-once 桶。
- [ ] 高优先级模式（root 使用、密钥创建、新 region、KMS 禁用）的检测规则到位。
- [ ] On-call 轮值对 critical 告警有响应。
- [ ] 日志热存 90 天，按合规拉长冷存。

**响应**
- [ ] 事件手册写好并每季度演练。
- [ ] 遏制脚本对一台测试实例跑过。
- [ ] 复盘共享，行动项跟踪到关闭。
- [ ] 客户 / 监管方沟通模板预先经法务批准。

清单的目的不是让你勾完之后觉得安全。每一项没勾的，都是攻击者或合规审计员**最先发现**的已知配置。

---

## 系列导航

| 篇 | 主题 |
|----|------|
| 1 | [基础与架构体系](/zh/cloud-computing-fundamentals/) |
| 2 | [虚拟化技术深度解析](/zh/cloud-computing-virtualization/) |
| 3 | [存储系统与分布式架构](/zh/cloud-computing-storage-systems/) |
| 4 | [网络架构与 SDN](/zh/cloud-computing-networking-sdn/) |
| **5** | **安全与隐私保护（当前）** |
| 6 | [运维与 DevOps 实践](/zh/cloud-computing-operations-devops/) |
| 7 | [云原生与容器技术](/zh/cloud-computing-cloud-native-containers/) |
| 8 | [多云与混合架构](/zh/cloud-computing-multi-cloud-hybrid/) |
