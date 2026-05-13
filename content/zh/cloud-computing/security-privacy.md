---
title: "云计算（六）：云安全与隐私保护"
date: 2023-05-07 09:00:00
tags:
  - Cloud Computing
  - Cloud Security
  - IAM
  - Encryption
  - Zero Trust
  - Compliance
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "工程师视角的云安全实战：共担责任、可扩展的 IAM、静态/传输/使用中加密、零信任、合规框架，以及一份可以反复演练的事件响应流程。"
disableNunjucks: true
series_order: 6
translationKey: "cloud-computing-6"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/illustration_1.png)

2019 年，Capital One 泄露了一亿条客户记录。整个攻击链非常简短：一个配置错误的 WAF 允许攻击者对 EC2 元数据端点发起服务端请求伪造（SSRF），该端点随即返回了 IAM 临时凭证，而该 IAM 角色对账户内所有 S3 存储桶都拥有通配符权限 `s3:*`。一处配置失误、一个权限过宽的角色、一条安全团队漏写的防护规则——仅此而已。事件造成的直接经济损失（不含法律费用）超过 8000 万美元。

此后几乎每一起公开的云安全事件都呈现出相同的模式：并非零日漏洞，也不是国家级恶意软件，而是无人察觉的配置错误，直到数据早已出现在 Pastebin 上。因此，云安全工程师的核心任务，并非钻研密码学奇技，而是系统性地消除那些将微小失误放大为数千万美元灾难的条件。

本文将从共享责任模型一路讲到事件响应，每一层都配有代码、架构图、失效模式和可演练的运行手册。

## 你将学到什么
- 共享责任模型及其在 IaaS / PaaS / SaaS 中的差异
- 可扩展的 IAM 设计：身份、组、角色、策略，以及经得起审计考验的最佳实践
- 静态、传输中和使用中数据的加密方案——尤其聚焦于几乎人人都会忽略的关键细节
- DDoS 防护与 Web 应用防火墙（WAF）的实战配置，以及如何限流而不误伤合法用户
- 零信任架构的具体落地：一套可执行的控制措施，而非厂商口号
- 合规框架对比（SOC 2、HIPAA、GDPR、PCI DSS、ISO 27001）：它们真正要求你做什么
- 一套可提前演练的事件响应闭环流程

## 前置知识
- 基础网络知识（TCP、TLS、DNS、防火墙）
- 熟悉至少一家云厂商的 IAM 控制台
- 建议先阅读本系列第 1–4 篇

---

## 1. 共享责任，说透本质

![共担责任模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/fig1_shared_responsibility.png)

云厂商都会发布共享责任模型，原因很简单：每次事故复盘，第一句话总是“这到底该谁负责？”答案取决于你使用的服务层级。

| 层级             | IaaS（EC2）   | PaaS（App Engine） | SaaS（Gmail） |
| ---------------- | ------------- | ------------------ | ------------- |
| 数据与访问权限   | **你**        | **你**             | **你**        |
| 应用代码         | **你**        | **你**             | 厂商          |
| 运行时 / OS      | **你**        | 厂商               | 厂商          |
| 虚拟化层         | 厂商          | 厂商               | 厂商          |
| 网络基础设施     | 厂商          | 厂商               | 厂商          |
| 物理设施         | 厂商          | 厂商               | 厂商          |

但图表常遗漏三点关键事实：  
第一，**数据和访问控制始终归你**——即便是 SaaS，谁能进入你的 Salesforce 租户，只有你能决定。  
第二，**厂商服务的配置也由你负责**——AWS 提供 S3，但桶策略、加密设置、公共访问阻断开关都需你手动配置，而这些功能默认曾长期处于“开放”状态。  
第三，**厂商的责任仅在你正确使用其服务时才生效**；若你在 EC2 上自建数据库，补丁、备份和高可用性就全落回你肩上。

更直白的说法是：“厂商保障底层基座，凡是你能配置的，就必须配对。”

## 2. 你真正会遭遇的威胁

Verizon DBIR 与 Mandiant M-Trends 报告年复一年地指向相同的云安全事件根源。按发生频率大致排序如下：

- **对象存储配置错误**：公开的 S3 / GCS 桶中存放客户 PII、源码或备份。
- **凭证泄露或长期未轮换**：API 密钥被提交到 GitHub、硬编码进 AMI，或残留在 Slack 聊天记录中。
- **IAM 权限过度宽松**：某个角色本只需 `s3:GetObject` 访问单个桶，却因控制台向导图省事，直接授予 `s3:*` 全桶权限。
- **应用未及时打补丁**：库、容器镜像或 OS 包中的已知 CVE。
- **供应链投毒**：恶意依赖（如 `event-stream`）、污染的基础镜像、仿冒拼写错误的软件包。
- **会话令牌被盗**：钓鱼获取 MFA 验证码，或浏览器端窃取 token 后异地重放。

注意：针对 AES 或 TLS 的密码学攻击从未出现在这份清单上。防御者的杠杆在于 IAM 卫生、配置管理与检测能力，而非发明新密码算法。

## 3. IAM：不容有失的核心子系统

![IAM 架构图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/fig2_iam_model.png)

### 3.1 模型基础

成熟的 IAM 系统都包含四个基本原语：

- **身份（Identity）**：人（如 Alice）或机器（如 CI Runner），通过密码 + MFA、证书或工作负载身份令牌证明自己。
- **组（Group）**：身份的静态集合（如“工程师”“财务”），仅为管理便利，本身不持有权限。
- **角色（Role）**：命名的权限包。身份或组可“扮演”角色；在 AWS 中，角色自带“信任策略”，声明谁可扮演它。
- **策略（Policy）**：具体的权限语句：允许/拒绝哪些操作、作用于哪些资源、满足何种条件。

权限流向至关重要：**身份 → 组 → 角色 → 策略 → 操作**。权限只能单向流动。审计任意访问行为时，可从资源反向追溯整条链。

### 3.2 六条经得起实战检验的规则

1. **最小权限起步，逐步迭代**：从零开始，仅授予任务所需的最小权限。若合法操作失败，精准放宽——绝不使用 `s3:*`。
2. **所有运行代码的实体一律使用角色，禁用长期凭证**：EC2 实例配置文件、Lambda 执行角色、EKS 的 IRSA、GKE 的 Workload Identity——均提供自动轮换的短期凭证。
3. **所有人类身份强制启用 MFA**：不存在“多人共用的服务账号”的例外——这类账号本就不该存在。
4. **轮换、撤销、设过期**：超过 90 天的 API 密钥是危险信号；员工离职那一刻，其凭证倒计时即已启动。
5. **使用权限边界与 SCP**：允许团队自主创建角色，但通过组织级策略限制危险组合。
6. **记录每一次授权与使用**：CloudTrail、Cloud Audit Logs、Azure Activity Log——集中存储、防篡改、保留至少一年。

### 3.3 一份真正的最小权限策略

仅允许对单一存储桶只读访问，限制为企业 CIDR 范围，且要求当前会话已完成 MFA 验证：

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

其中 `MultiFactorAuthAge` 条件常被忽视——它强制 MFA 验证必须“新鲜”，而非“六小时前登录时验证过一次”。

### 3.4 组织级防护护栏

在组织根节点应用的服务控制策略（SCP）无法被任何账户级角色覆盖，适用于控制爆炸半径：

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

该策略禁止 root 用户执行任何操作，并阻止在非运营区域执行动作。两者共同防范数据驻留违规及攻击者在冷门区域启动 GPU 挖矿等行为。

### 3.5 反复出现的 IAM 错误

- 在非 Deny 语句中使用 `"Action": "*"` 或 `"Resource": "*"`——几乎必然导致权限过宽。
- `iam:PassRole` 配合 `Resource: "*"`——单行代码即可实现权限提升。
- 长期 Access Key 被硬编码进 AMI、容器镜像，或随 `.env` 文件提交至 Git。
- “临时”管理员权限未设过期，最终变成永久授权。
- 多个服务共享同一 Service Account，导致无人敢轮换凭证。

## 4. 加密：静态、传输中与使用中

![数据的三种状态](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/fig3_encryption_layers.png)

数据有三种状态，每种需不同防护机制，对应不同威胁。

### 4.1 静态数据

威胁：硬盘、快照或备份磁带被盗，或授权进程意外读取你不希望暴露的原始字节。防御：使用你实际控制的密钥进行对称加密。

- 应用层加密：**AES-256-GCM**
- 全盘/卷加密（EBS、持久盘）：**AES-XTS**
- 托管数据库：**TDE（透明数据加密）**
- 对象存储：默认启用**服务端加密**；在 AWS 上设置 `BucketKeyEnabled: true` 可降低约 99% 的 KMS 请求成本。

使用云 KMS 或 HSM 支持的密钥库，**切勿自行实现密钥轮换**。

```python
import boto3

kms = boto3.client("kms")

# 信封加密：KMS 加密一个小的数据密钥；
# 然后用这个数据密钥在本地加密实际的数据负载。
def encrypt_blob(plaintext: bytes, key_id: str) -> dict:
    resp = kms.generate_data_key(KeyId=key_id, KeySpec="AES_256")
    plaintext_dk, encrypted_dk = resp["Plaintext"], resp["CiphertextBlob"]

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os
    nonce = os.urandom(12)
    ciphertext = AESGCM(plaintext_dk).encrypt(nonce, plaintext, associated_data=None)

    # 丢弃明文数据密钥；只保存加密后的版本。
    return {"nonce": nonce, "ciphertext": ciphertext, "wrapped_key": encrypted_dk}
```

所有“透明”云加密功能底层都采用信封加密模式。理解它有助于评估成本（每数据密钥一次 KMS 调用，而非每字节）和轮换策略（重包装数据密钥，而非重加密 PB 级数据）。

### 4.2 传输中数据

威胁：链路中间人、被攻陷的网络设备抓包、恶意服务网格 sidecar。防御：**正确使用 TLS**。

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

- **在负载均衡器禁用 TLS 1.0 和 1.1**：所有现行合规标准均已禁止，且存在已知弱点。
- **HSTS 设置长 `max-age`**：这是唯一能抵御 SSL-strip 降级攻击的手段。
- **服务间启用 mTLS**：集群内部，服务 A 应拒绝未出示内部 CA 证书的服务 B。服务网格（Istio、Linkerd）可将此简化为一行 YAML。

### 4.3 使用中数据

最棘手的场景：数据在内存解密后处理，意味着同主机上高权限进程可能读取它。现有三种缓解方案：

- **机密虚拟机**（AWS Nitro Enclaves、GCP Confidential VMs、Azure Confidential Computing）：Hypervisor 无法读取客户机内存，且可通过硬件根信任远程验证。
- **硬件安全区**（Intel SGX、AMD SEV-SNP）：进程内小型可信区域，用于存放解密数据。
- **同态加密 / 安全多方计算**：数学优美，但性能比明文慢 1000 至 100 万倍，仅适用于特定场景（如隐私集合求交、特定聚合）。

对大多数团队而言，“使用中”保护靠间接实现：**最小化明文处理面、严格隔离、加强审计**。

## 5. DDoS 防护与 Web 应用防火墙

### 5.1 三类攻击

| 类型       | 机制                     | 示例                     | 影响点               |
|------------|--------------------------|--------------------------|----------------------|
| 流量型     | 耗尽带宽                 | UDP / DNS 放大           | 网络管道             |
| 协议型     | 耗尽连接状态表           | SYN 洪水、ACK 洪水       | 负载均衡器、操作系统 |
| 应用层     | 强制执行高成本操作       | HTTP 洪水、Slowloris、GraphQL 炸弹 | 应用服务器、数据库   |

真实攻击者常三者并用，真实防御需分层堆叠。

### 5.2 防御堆栈

- **边缘 / 网络层**：AWS Shield Advanced、Cloud Armor、Cloudflare——在流量进入 VPC 前吸收 L3/L4 洪水。
- **CDN**：缓存静态资源，扩大暴露面以稀释攻击。
- **WAF**：拦截 SQLi、XSS、RCE、恶意机器人，必要时地理封锁。以托管规则集为基线，再叠加应用专属规则。
- **限流**：按 IP、token、路由限制，应对撞库、爬虫和慢速应用层攻击。
- **应用层**：输入验证、参数化查询、查询成本上限。**WAF 不是输入验证的替代品**。

### 5.3 今日即可部署的 WAF 规则集

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
    },
    {
      "Name": "BlockKnownBadCountries",
      "Priority": 30,
      "Action": { "Block": {} },
      "Statement": {
        "GeoMatchStatement": { "CountryCodes": ["KP", "IR"] }
      },
      "VisibilityConfig": { "SampledRequestsEnabled": true,
                            "CloudWatchMetricsEnabled": true,
                            "MetricName": "geo" }
    }
  ]
}
```

谨慎调整 `Limit`。过严可能误拦 CGNAT 后的自家移动 App。建议先以 `Count` 模式运行一周，再切换至 `Block`。

## 6. 安全日志与检测

### 6.1 应记录的内容

按优先级排序的五类日志：

1. **认证**：登录、失败、MFA 挑战、密码重置。
2. **授权**：每次授权与拒绝，附决策所用策略。
3. **数据访问**：对敏感资源（PII 字段、KMS 密钥、密钥库）的读写。
4. **配置变更**：IAM、安全组、KMS、基础设施即代码（IaC）变更。
5. **网络**：VPC Flow Logs、DNS 查询日志。

### 6.2 解析一条 CloudTrail 事件

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
  "responseElements": null,
  "additionalEventData": { "bytesTransferredOut": 38421 }
}
```

SIEM 规则实际消费的字段：`userIdentity.arn`、`mfaAuthenticated`、`sourceIPAddress`、`eventName`、`requestParameters` 及响应状态。值得告警的模式包括：

- 来自非运营国家的 `ConsoleLogin`
- 非自动化流程外的 `iam:CreateUser` 或 `iam:CreateAccessKey`
- 某角色突然激增 `s3:GetObject` 调用
- 任何 `kms:Disable*` 或 `kms:ScheduleKeyDeletion`
- 从未使用 GPU 的角色调用 `ec2:RunInstances` 启动 GPU 实例

### 6.3 日志管道与保留策略

- 集中至 SIEM（Security Hub、Splunk、Elastic Security、Chronicle）
- 日志桶加密，并禁止所有人（含安全团队角色）执行 `s3:DeleteObject`——**可篡改的日志不是证据**
- 热数据保留 90 天，冷数据依行业合规要求保留 1–7 年
- 提前构建仪表盘；凌晨 3 点应急时不该才学 Lucene 语法

## 7. 零信任，具体落地

![零信任架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/fig4_zero_trust.png)

“零信任”虽是营销术语，但核心原则简单：**网络位置不赋予特权**。身处公司 VPN、生产 VPC 或集群网络，并不自动获得授权。

具体控制措施包括：

- **每个内部应用前部署身份感知代理**：告别“只在内网，无需认证”的旧思维。Google BeyondCorp 开创此范式，Cloudflare Access、AWS Verified Access、Tailscale 提供开箱即用方案。
- **设备状态作为认证因子**：策略决策点（PDP）检查：设备是否加密？是否注册 MDM？补丁是否最新？序列号是否在资产库？不符则拒或隔离，无论凭证是否有效。
- **服务间 mTLS + SPIFFE/SPIRE**：每个工作负载拥有加密身份（SPIFFE ID）、短期证书，策略明确允许的调用关系。
- **持续评估**：会话非“12 小时有效”，而是持续基于信号（IP 变更、地理位置变化、设备不合规）重新评估。高风险会话被降级或撤销。
- **微分段**：被攻破的 Pod 无法自由访问集群其余部分。NetworkPolicy 将东西向流量限制于声明的服务依赖。

心智模型：每次请求都经策略决策点（PDP）判断——“谁？做什么？从哪来？用什么设备？在什么上下文？针对此资源——允许还是拒绝？”且每次决策均被记录。

## 8. 合规框架

![合规框架](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/security-privacy/fig5_compliance_frameworks.png)

合规不等于安全，但能强制建立基线控制与文档链。五大框架覆盖多数工程团队义务：

| 框架        | 范围                          | 核心要求 |
|-------------|-------------------------------|----------|
| **SOC 2**   | 美国服务组织                  | 信任服务准则：安全、可用性、处理完整性、保密性、隐私。年度审计。 |
| **HIPAA**   | 美国医疗 PHI                  | 与子处理器签 BAA、全程加密、六年审计留存、泄露通知。 |
| **GDPR**    | 欧盟个人数据（全球处理）      | 合法依据、数据最小化、主体访问/删除权、72 小时泄露通知、DPIA。 |
| **PCI DSS** | 支付卡数据                    | 网络分段、加密存储、季度扫描、年度渗透测试。 |
| **ISO 27001**| 国际 ISMS                    | 文档化信息安全管理体系、风险评估、附录 A 控制、认证审计。 |

### GDPR 工程清单

- **数据盘点**：记录所有处理个人数据的系统及合法依据。
- **主体访问请求端点**：30 天内导出某人全部数据。
- **删除端点**：通过租户级密钥对备份进行密码学粉碎，合规认可。
- **72 小时泄露通知**：提前演练沟通路径。
- **高风险处理的 DPIA**：位置、健康、生物识别、画像等新处理需评估。
- **处理活动记录（第 30 条）**：从数据盘点自动生成，勿手工维护。

五大框架共性：**无法事后补救**。将控制嵌入基础设施代码、审计日志与部署流水线，审计才能成为文书工作，而非生存危机。

## 9. 事件响应：闭环实践

NIST 循环（准备 → 检测 → 遏制 → 清除 → 恢复 → 复盘）理论正确但偏学术。实践中，决定事故是 30 分钟插曲还是 30 天灾难的，是以下三点：

1. **运行手册已演练**：每季度桌面推演，每年至少一次实战“游戏日”。
2. **能快速隔离且保留取证**：先快照，再销毁；先取内存，再重启。
3. **无责复盘报告附可落地行动项**：否则明年还会重演。

### 可复用的遏制脚本

```python
"""被入侵的 EC2 实例：生成取证快照、隔离环境、停止实例。"""
import boto3
from datetime import datetime

ec2 = boto3.client("ec2")

def quarantine(instance_id: str, reason: str) -> dict:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = [{"Key": "incident",  "Value": reason},
           {"Key": "timestamp", "Value": timestamp}]

    # 1. 在做任何其他操作之前，先为所有挂载的卷创建快照。
    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    snapshots = []
    for bdm in inst["BlockDeviceMappings"]:
        snap = ec2.create_snapshot(
            VolumeId=bdm["Ebs"]["VolumeId"],
            Description=f"forensic-{instance_id}-{reason}-{timestamp}",
            TagSpecifications=[{"ResourceType": "snapshot", "Tags": tag}],
        )
        snapshots.append(snap["SnapshotId"])

    # 2. 替换安全组，确保实例完全隔离。
    isolation_sg = ec2.create_security_group(
        GroupName=f"quarantine-{instance_id}-{timestamp}",
        Description=f"Incident isolation: {reason}",
        VpcId=inst["VpcId"],
    )["GroupId"]
    ec2.modify_instance_attribute(InstanceId=instance_id, Groups=[isolation_sg])

    # 3. 解绑 IAM 实例 Profile，防止凭证被滥用。
    profile = inst.get("IamInstanceProfile", {}).get("Arn")
    if profile:
        association_id = next(
            a["AssociationId"]
            for a in ec2.describe_iam_instance_profile_associations(
                Filters=[{"Name": "instance-id", "Values": [instance_id]}])["IamInstanceProfileAssociations"]
        )
        ec2.disassociate_iam_instance_profile(AssociationId=association_id)

    # 4. 停止实例（不要直接终止——终止会丢失证据）。
    ec2.stop_instances(InstanceIds=[instance_id])

    return {"instance": instance_id, "snapshots": snapshots, "isolation_sg": isolation_sg}
```

顺序至关重要：**快照 → 隔离 → 撤销凭证 → 停止**。颠倒顺序可能导致快照期间数据外泄，或关机时丢失证据。

## 10. 安全基线固化到基础设施即代码中

再完美的安全基线，也挡不住疲惫工程师周五下午随手 `terraform apply` 出一个公开 S3 桶。解决方案：将基线编码为模块与策略即代码（OPA / Sentinel / Checkov）：

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
    bucket_key_enabled = true   # 减少约 99% 的 KMS 调用成本
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

CI 中的 Checkov 策略会拒绝任何未配套这四个附属资源的 `aws_s3_bucket`。错误在抵达生产前即被拦截。

## 11. 工程师起飞前检查清单

**身份**
- [ ] 所有人类身份强制 MFA，包括应急账号（密钥封存信封，不在 Slack）
- [ ] 工作负载禁用长期 Access Key，改用实例配置文件 / IRSA / Workload Identity
- [ ] 每季度审查：谁有管理员权限？是否仍需？
- [ ] SCP / 组织策略禁止 root 使用，禁用非运营区域

**加密**
- [ ] 所有存储服务默认开启加密
- [ ] 敏感数据使用客户托管 KMS 密钥，每年轮换
- [ ] 负载均衡器强制 TLS 1.2+，HSTS 预加载
- [ ] 威胁模型要求时，内部服务间启用 mTLS

**网络**
- [ ] 安全组默认拒绝；开放端口需代码评审说明
- [ ] 所有公网应用前置 WAF，先 Count 模式运行一周再 Block
- [ ] 生产端点启用 DDoS 防护（Shield Advanced / Cloud Armor）
- [ ] VPC Flow Logs 与 DNS 查询日志启用并送入 SIEM

**检测**
- [ ] CloudTrail / Cloud Audit Logs 写入独立账户的防删桶
- [ ] 高优模式检测规则（root 使用、密钥创建、新区域、KMS 禁用）
- [ ] On-call 轮值对关键告警有响应机制
- [ ] 日志热存 90 天，冷存依合规延长

**响应**
- [ ] 事件手册每季度演练
- [ ] 遏制自动化在测试实例验证
- [ ] 复盘公开，行动项跟踪至闭环
- [ ] 客户 / 监管沟通模板法务预批

清单目的并非勾选后安心，而是提醒：**每个未勾选项，都是攻击者或审计员最先发现的已知风险**。
