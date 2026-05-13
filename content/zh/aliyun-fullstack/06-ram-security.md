---
title: "阿里云全栈实战（六）：RAM、KMS 筑牢云安全"
date: 2026-05-03 09:00:00
tags:
  - Alibaba Cloud
  - RAM
  - KMS
  - Security
  - IAM
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 6
description: "锁定云安全：RAM 用户、组、角色和策略。STS 临时凭证。KMS 加密。ActionTrail 审计日志。构建最小权限的多团队访问模型。"
disableNunjucks: true
translationKey: "aliyun-fullstack-6"
---
有一次，我在一个公开的 GitHub 仓库里发现了自己的 DashScope API Key——有人 fork 了我几个月前上传的一个 Demo，而这个 API Key 明文存放在未被 .gitignore 排除的配置文件中。等我发现时，这个 Key 已在一个周末内被用于发起 14,000 次 Qwen API 调用，所幸账单未超支，这得益于 DashScope 按 token 计费的弹性计费机制，但教训极为深刻。我曾以为云安全可以“以后再做”，结果这个“以后”变成了凌晨两点触发的账单告警。

那天我配置了 RAM 用户，轮转了所有 access key，开启了 MFA，并将所有涉及前端直连云服务的场景改为使用 STS 临时凭证。我将这一过程的经验系统地梳理成文，结构清晰、目标明确，一个下午就完成了基础加固，不必等到事故发生后再亡羊补牢。


安全组——也就是网络层防火墙——我们在 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/) 讲过了。这篇讲的是身份层：谁能做什么、怎么加密数据、怎么审计所有操作。想用 Terraform 管安全，看 [Terraform Part 6: LLM Gateway and Secrets](/zh/terraform-agents/06-llm-gateway-and-secrets/)。

## 安全心智模型

云安全不是打开一个开关就能搞定的事——它由多个相互独立的安全层构成，各层分别防御一类典型故障；即使某一层失效，其余层仍可提供防护，这正是“纵深防御（defense in depth）”原则的核心要义。

![阿里云安全模型概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_security_model.png)

我把它看作四大支柱：

| 支柱 | 回答的问题 | 阿里云服务 | AWS 对应服务 |
|---|---|---|---|
| **身份 (Identity)** | 谁发起了请求？ | RAM (users, groups) | IAM (users, groups) |
| **授权 (Authorization)** | 允许做什么？ | RAM (policies, roles) | IAM (policies, roles) |
| **加密 (Encryption)** | 数据静态和传输中是否受保护？ | KMS, SSL 证书 | KMS, ACM |
| **审计 (Auditing)** | 谁在什么时候做了什么？ | ActionTrail | CloudTrail |

每一项安全决策都落在这四个范畴之内；一旦出问题（不可避免），审计日志会指出其余三层中哪一层失守。为新团队设计权限时，应依次覆盖这四个维度：创建身份、分配权限、加密数据、记录操作。

该心智模型与 AWS IAM 高度一致——阿里云在设计 RAM 时直接对标 IAM。顶层是 root 账号，其下是 RAM 用户；策略授予权限，角色支持跨服务与跨账号访问。若已熟悉 AWS IAM，RAM 的大部分概念和操作你基本已掌握，剩下的两成只是命名差别和一些功能细节不同。

一个关键区别在于，阿里云将 root 账号称为“阿里云主账号”（也简称“主账号”）。虽然控制台中不显示“root”字样，但其权限与 root 完全等同，即对整个云环境拥有完全控制权，因此严禁用于日常操作。

## RAM：资源访问管理

RAM 就是阿里云的身份和访问管理系统。每个 API 调用、每次控制台点击、每条 CLI 命令都通过 RAM 认证和授权。理解 RAM 不是选修课，而是必修课——后文所有内容都以此为基础。

![RAM 用户、组与角色层次](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_ram_hierarchy.png)

### 阿里云主账号 (Root)

注册阿里云的时候，你会得到一个阿里云主账号。这就是 root 身份。它对每个服务、每个资源、每个账单设置都有无限制访问权。它可创建或删除 RAM 用户、修改支付方式，甚至直接注销整个阿里云账号。

该主账号仅允许用于以下三类操作：

1. 初始设置（创建你的第一个 RAM 管理员用户）
2. 账单和支付方式变更
3. RAM 配错时的紧急恢复

其他所有事——开发、部署、运维、监控——都用 RAM 用户。我曾见过多个团队让六名工程师共用主账号凭证。一旦有人误操作删除了生产环境 RDS 实例，便无法追溯具体责任人，因为 ActionTrail 中所有操作均显示为‘root’。身份隔离并非额外增加管理负担，而是开展事故根因分析的基本前提。

### 创建 RAM 用户

RAM 用户是隶属于阿里云主账号的永久身份。每个用户有自己的登录 credentials （控制台用密码， API/CLI 用 AccessKey）和自己的权限集。

用 CLI 创建 RAM 用户：

```bash
aliyun ram CreateUser \
  --UserName alice \
  --DisplayName "Alice Chen" \
  --Comments "Backend developer, team-alpha"
```

启用控制台密码登录：

```bash
aliyun ram CreateLoginProfile \
  --UserName alice \
  --Password 'TempP@ssw0rd!2026' \
  --PasswordResetRequired true \
  --MFABindRequired true
```

`--PasswordResetRequired true` 标志强制 Alice 首次登录时改密码。`--MFABindRequired true` 强制她先配好 MFA 才能操作。凡拥有生产资源写权限的账号，必须强制启用密码修改与 MFA。

创建 AccessKey pair 用于程序访问：

```bash
aliyun ram CreateAccessKey --UserName alice
```

这会返回一个 AccessKeyId 和 AccessKeySecret。 AccessKeySecret 仅在创建时显示一次，一旦丢失，只能通过重新创建密钥对来恢复。务必存入专用密码管理器——严禁写入配置文件、共享服务器环境变量，更不得提交至 Git 仓库。

### 配置 MFA

多因素认证（MFA）为身份认证这一支柱增加了第二重保障。即便攻击者窃取了密码，若未同步获取手机 App 中的 TOTP 动态验证码，仍无法完成登录。

给 RAM 用户启用虚拟 MFA：

```bash
# Step 1: Create a virtual MFA device
aliyun ram CreateVirtualMFADevice \
  --VirtualMFADeviceName alice-mfa

# Step 2: The output includes a Base32StringSeed -- scan this as a QR code
# in Google Authenticator, Authy, or any TOTP app

# Step 3: Bind the MFA device to the user (requires two consecutive codes)
aliyun ram BindMFADevice \
  --UserName alice \
  --SerialNumber "acs:ram::1234567890:mfa/alice-mfa" \
  --AuthenticationCode1 123456 \
  --AuthenticationCode2 789012
```

请为主账号单独配置 MFA：登录阿里云控制台，依次进入 **账号管理 > 安全设置 > 多因素认证（MFA）**。推荐优先使用硬件安全密钥（如 YubiKey）。主账号的 MFA 设备须妥善保存在物理保险箱中，切勿存放在 CEO 等人员频繁变动的个人手机上。

## RAM 群组

基于单个用户绑定策略的方式缺乏可扩展性： 3 名开发者时尚可维护，但当团队扩展至 30 人时，权限管理将迅速失控。例如，某开发者转岗后，旧权限可能被遗漏清理；新成员入职时，若直接复制他人策略，极易误授生产环境删除权限。

用户组（Group）机制可有效应对这一权限管理难题。 RAM 用户组本质上是用户的逻辑集合。您可将策略绑定至用户组，组内所有成员自动继承该策略权限。有人换团队，把他移到别的组就行。新人入职，加进对应的组，拿到的权限刚好够用。

以下是我在多数项目中采用的用户组划分方案：

| 群组 | 用途 | 关键策略 |
|---|---|---|
| `Administrators` | 除账单外的全权限 | AdministratorAccess |
| `Developers` | 计算、存储、数据库读写 | 自定义： ECS/OSS/RDS 全权，无 RAM/账单 |
| `ReadOnly` | 只看不动 | ReadOnlyAccess |
| `Billing` | 管付款和成本分析 | 自定义： BSS 全权 |
| `CICD` | 部署流水线（非人类） | 自定义： ECS/CR/ACK 仅部署 |

创建群组并添加用户：

```bash
# Create the groups
aliyun ram CreateGroup --GroupName Administrators --Comments "Full admin access"
aliyun ram CreateGroup --GroupName Developers --Comments "Dev team - compute and storage"
aliyun ram CreateGroup --GroupName ReadOnly --Comments "Stakeholders - view only"
aliyun ram CreateGroup --GroupName Billing --Comments "Finance team - billing only"
aliyun ram CreateGroup --GroupName CICD --Comments "CI/CD service accounts"

# Add users to groups
aliyun ram AddUserToGroup --UserName alice --GroupName Developers
aliyun ram AddUserToGroup --UserName bob --GroupName Administrators
aliyun ram AddUserToGroup --UserName carol --GroupName ReadOnly

# Attach system policies to groups
aliyun ram AttachPolicyToGroup \
  --PolicyType System \
  --PolicyName AdministratorAccess \
  --GroupName Administrators

aliyun ram AttachPolicyToGroup \
  --PolicyType System \
  --PolicyName ReadOnlyAccess \
  --GroupName ReadOnly
```

列出组里所有用户：

```bash
aliyun ram ListUsersForGroup --GroupName Developers
```

将用户从原用户组中移除：

```bash
aliyun ram RemoveUserFromGroup --UserName alice --GroupName Developers
aliyun ram AddUserToGroup --UserName alice --GroupName Administrators
```

核心原则：禁用直接向 RAM 用户绑定权限策略，所有权限必须经由用户组统一分发。唯一例外是需对组内个别成员施加额外限制的情形；此时，建议为其新建专用用户组并绑定 Deny 策略，而非在原组内混用 Allow/Deny。
## 深入理解 RAM 策略

策略是授权的核心引擎。阿里云里的每次 API 调用都要过一遍策略，决定是放行还是拒绝。理解策略机制，是从‘功能可用’迈向‘安全可控’的关键分水岭。

![RAM 策略评估流程图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_policy_evaluation.png)

### 系统策略 vs 自定义策略

阿里云预置了 800 多个系统策略，官方维护，没法改，但覆盖了绝大多数常见场景：

| 系统策略 | 授权内容 |
|---|---|
| `AdministratorAccess` | 所有服务和资源的全部权限 |
| `ReadOnlyAccess` | 所有服务的只读权限 |
| `AliyunECSFullAccess` | ECS 全部权限 |
| `AliyunOSSFullAccess` | OSS 全部权限 |
| `AliyunRDSFullAccess` | RDS 全部权限 |
| `AliyunVPCFullAccess` | VPC 全部权限 |
| `AliyunRAMFullAccess` | RAM 全部权限（危险——这是通往王国的钥匙） |
| `AliyunKMSFullAccess` | KMS 全部权限 |
| `AliyunActionTrailFullAccess` | ActionTrail 全部权限 |
| `AliyunBSSFullAccess` | 账单全部权限 |

超出这些通用场景，你就得写自定义策略了。

### 策略结构

RAM 策略本质上是个 JSON 文档，结构固定。每个策略都有 `Version` 和一个或多个 `Statement`。`Statement` 里包含 `Effect`（允许或拒绝）、`Action`（操作接口）、`Resource`（具体资源），还有可选的 `Condition`（生效条件）。

结构拆解如下：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeInstances",
        "ecs:StartInstance",
        "ecs:StopInstance"
      ],
      "Resource": "acs:ecs:cn-hangzhou:1234567890:instance/*",
      "Condition": {
        "IpAddress": {
          "acs:SourceIp": "10.0.0.0/8"
        }
      }
    }
  ]
}
```

逐项来看：

- **Version**: 固定 `"1"`。阿里云 RAM 目前只有这一个策略版本。
- **Effect**: `"Allow"` 或 `"Deny"`。一旦匹配到 Deny，直接拒绝，优先级高于 Allow。
- **Action**: API 操作。支持通配符：`ecs:*` 代表所有 ECS 操作，`ecs:Describe*` 代表所有 ECS 读操作。
- **Resource**: 阿里云资源名称（ARN）。格式：`acs:{service}:{region}:{account-id}:{resource-type}/{resource-id}`。用 `*` 代表所有资源。
- **Condition**: 可选约束。常见的有：源 IP、 MFA 状态、时间、请求标签值。

### 实战策略示例

**ECS 管理员——仅限单个地域的 ECS 全权：**

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecs:*",
      "Resource": "acs:ecs:cn-hangzhou:*:*"
    },
    {
      "Effect": "Allow",
      "Action": "vpc:Describe*",
      "Resource": "*"
    }
  ]
}
```

第二条语句给了 VPC 读权限——这很有必要，因为 ECS 操作经常需要查询 VPC/VSwitch 信息。少了它，创建实例时会报授权错误，而且错误信息里完全不提 VPC，让人摸不着头脑。

**指定 Bucket 的 OSS 只读：**

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "oss:GetObject",
        "oss:GetObjectAcl",
        "oss:ListObjects",
        "oss:GetBucketInfo",
        "oss:ListBuckets"
      ],
      "Resource": [
        "acs:oss:*:*:my-data-bucket",
        "acs:oss:*:*:my-data-bucket/*"
      ]
    }
  ]
}
```

注意这里有两行 Resource：第一行授权访问 Bucket 本身（用于 `ListObjects`），第二行授权访问 Bucket 内的对象（用于 `GetObject`）。缺任何一行都会报让人困惑的 403 错误。

**仅限 DashScope API 访问（适合 AI 开发者）：**

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dashscope:*"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": [
        "dashscope:DeleteModel",
        "dashscope:DeleteDeployment"
      ],
      "Resource": "*"
    }
  ]
}
```

这给了完整的 DashScope 权限，但显式拒绝了删除操作。 Deny 会覆盖 Allow，所以即便有人拿了 `dashscope:*`，也删不了模型或部署。这是常见套路：宽泛 Allow 加上针对破坏性操作的特定 Deny。

**敏感操作强制 MFA：**

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "Bool": {
          "acs:MFAPresent": "true"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "ram:GetUser",
        "ram:ListMFADevicesForUser",
        "ram:CreateVirtualMFADevice",
        "ram:BindMFADevice"
      ],
      "Resource": "*"
    }
  ]
}
```

第一条语句说“允许所有操作，但必须开启 MFA"。第二条语句允许 MFA 相关操作无需 MFA （否则用户根本没权限去设置 MFA）。少了第二条，新用户会陷入死循环：没 MFA 不让动，想设 MFA 又没权限。

### RBAC 与 ABAC

RAM 支持两种权限模型，大多数 setup 都是混用：

**RBAC （基于角色的访问控制）**：权限基于用户角色（组 membership）分配。“所有开发人员可以启停 ECS 实例。”这就是组的作用。

**ABAC （基于属性的访问控制）**：权限基于资源属性，通常是标签。“用户只能管理 tagged 为 `team=alpha` 的实例。”

ABAC 示例——用户只能管理自己团队的实例：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecs:*",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "acs:ResourceTag/team": "${ram:PrincipalTagValue/team}"
        }
      }
    }
  ]
}
```

这策略说：仅当资源的 `team` 标签匹配用户的 `team` 标签时，才允许 ECS 操作。给用户 Alice 打上 `team=alpha` 标签，给她实例也打上 `team=alpha`，她就能管理。即便 Action 写了 `ecs:*`，她也动不了 `team=beta` 的实例。

ABAC 强大但难调试。我建议先从 RBAC （组 + 策略）开始，只有需要基于标签隔离时才加 ABAC——通常是多个团队共享同一个账号的时候。

创建并绑定自定义策略：

```bash
# 创建策略
aliyun ram CreatePolicy \
  --PolicyName ECS-HangZhou-Admin \
  --PolicyDocument '{
    "Version": "1",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": "ecs:*",
        "Resource": "acs:ecs:cn-hangzhou:*:*"
      },
      {
        "Effect": "Allow",
        "Action": "vpc:Describe*",
        "Resource": "*"
      }
    ]
  }'

# 绑定到组
aliyun ram AttachPolicyToGroup \
  --PolicyType Custom \
  --PolicyName ECS-HangZhou-Admin \
  --GroupName Developers
```

## RAM 角色

RAM 用户是给真人（和 CI/CD 流水线）用的永久身份。 RAM 角色是临时身份，专为这三种场景设计：

![ABAC vs RBAC 对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_abac_rbac.png)

1. **服务角色**：阿里云服务（ECS、 Function Compute 等）需要访问另一个服务（OSS、 RDS 等）
2. **跨账号访问**：账号 A 的用户需要访问账号 B 的资源
3. **联邦登录 (SSO)**：外部身份提供商（LDAP、 SAML、 OIDC）的用户需要阿里云访问权限

用户和角色的关键区别：

| 方面 | RAM 用户 | RAM 角色 |
|---|---|---|
| 身份类型 | 永久 | 临时（扮演） |
| 凭证 | 密码 + AccessKey | STS token （自动过期） |
| 使用者 | 真人、 CI/CD 机器人 | 服务、跨账号、 SSO |
| MFA 支持 | 支持 | 不支持（信任策略处理） |
| 直接登录 | 支持（控制台） | 不支持（必须扮演） |
| 最大会话 | 永久 | 1 小时（可配置到 12h） |

### 信任策略

每个角色都有个信任策略，指定谁能扮演它。这和权限策略（指定角色扮演后能做什么）是分开的。把它想成：信任策略是门锁，权限策略是里面的钥匙串。

### 服务角色示例： ECS 访问 OSS

常见场景： ECS 实例需要读 OSS Bucket 里的文件。错误做法是把 AccessKey 写进实例环境变量。实例一旦沦陷，攻击者就拿走了永久凭证。正确做法是用实例角色——ECS 实例自动获取临时凭证，每小时轮换。

第一步——创建角色，信任策略允许 ECS 扮演：

```bash
aliyun ram CreateRole \
  --RoleName ECS-OSS-Reader \
  --AssumeRolePolicyDocument '{
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "Service": ["ecs.aliyuncs.com"]
        }
      }
    ],
    "Version": "1"
  }' \
  --Description "Allows ECS instances to read from OSS"
```

第二步——给角色绑定权限策略：

```bash
aliyun ram AttachPolicyToRole \
  --PolicyType Custom \
  --PolicyName OSS-DataBucket-ReadOnly \
  --RoleName ECS-OSS-Reader
```

第三步——把角色绑定到 ECS 实例：

```bash
aliyun ecs AttachInstanceRamRole \
  --RegionId cn-hangzhou \
  --InstanceIds '["i-bp1234567890abcdef"]' \
  --RamRoleName ECS-OSS-Reader
```

第四步——在 ECS 实例内部， SDK 自动获取角色凭证：

```python
from alibabacloud_oss20190517.client import Client
from alibabacloud_tea_openapi.models import Config

# 不需要 AccessKey -- SDK 使用实例元数据获取 STS token
config = Config(
    region_id='cn-hangzhou',
    credential=None  # SDK 自动发现实例角色凭证
)
client = Client(config)

# 照常读 OSS
result = client.get_object('my-data-bucket', 'data/input.csv')
```

SDK 调用实例元数据服务 `http://100.100.100.200/latest/meta-data/ram/security-credentials/ECS-OSS-Reader` 获取临时凭证。这些凭证自动轮换。实例上不存 AccessKey。即便实例被攻破，攻击者拿到的凭证一小时内就过期，而且你能立刻撤销角色。

### 跨账号角色

当账号 B （ID: `9876543210`）需要让账号 A （ID: `1234567890`）的用户管理其 ECS 实例：

```bash
# 在账号 B：创建信任账号 A 的角色
aliyun ram CreateRole \
  --RoleName CrossAccount-ECS-Admin \
  --AssumeRolePolicyDocument '{
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "RAM": ["acs:ram::1234567890:root"]
        },
        "Condition": {
          "Bool": {
            "acs:MFAPresent": "true"
          }
        }
      }
    ],
    "Version": "1"
  }'

# 给角色绑定 ECS 管理员策略
aliyun ram AttachPolicyToRole \
  --PolicyType System \
  --PolicyName AliyunECSFullAccess \
  --RoleName CrossAccount-ECS-Admin
```

在账号 A，用户扮演该角色：

```bash
aliyun sts AssumeRole \
  --RoleArn acs:ram::9876543210:role/crossaccount-ecs-admin \
  --RoleSessionName alice-cross-account \
  --DurationSeconds 3600
```

Condition 块要求 MFA，为跨账号访问加了一层二次验证。
## STS：临时凭证

STS 生成的临时 AccessKey  pair 会附带一个安全令牌。用起来跟普通 AccessKey 没区别，但会自动过期。 RAM 角色底层就是这套机制，你也可以直接拿来用，比如移动端上传或者前端直连场景。

![STS 临时凭证流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_sts_flow.png)

### 为什么临时凭证优于永久凭证

永久 AccessKey 简直就是个安全隐患。它不过期。一旦泄露，除非你手动轮换，否则一直有效。轮换意味着要把所有用到 key 的服务都更新一遍，要么停机，要么协调发布。大部分团队因为嫌麻烦就拖着，结果泄露的 key 能活跃好几个月。

STS 令牌会过期。最长寿命 12 小时（默认 1 小时，最短 15 分钟）。就算令牌泄露，危害窗口也很小。你不需要轮换任何东西——等它过期就行，然后再慢慢查是怎么泄露的。

### STS 工作流程

流程很简单：可信后端扮演角色，拿到临时凭证，传给不可信客户端（移动 App、浏览器、第三方服务），客户端用凭证直到过期。

```text
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│             │  1. AssumeRole        │         │             │
│  Your App   │ ──────────────────►   │   STS   │             │
│  (Backend)  │         │             │ Service │             │
│             │  ◄──────────────────  │         │             │
│             │  2. Temp AK/SK/Token  │         │             │
└──────┬──────┘         └─────────────┘         └─────────────┘
       │
       │ 3. Pass temp credentials
       ▼
┌─────────────┐         ┌─────────────┐
│             │  4. Upload with       │             │
│  Frontend/  │    temp credentials   │             │
│  Mobile App │ ──────────────────►   │     OSS     │
│             │                       │             │
└─────────────┘                       └─────────────┘
```

### STS 完整示例：前端直传 OSS

这是 STS 最典型的用法。移动 App 或浏览器需要直接上传文件到 OSS。你肯定不想让流量经过后端（带宽和延迟扛不住），但也不想把永久 OSS 凭证写在前端代码里。

第一步 -- 创建一个权限受限的 OSS 角色策略：

```bash
# Create the role
aliyun ram CreateRole \
  --RoleName STS-OSS-Uploader \
  --AssumeRolePolicyDocument '{
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "RAM": ["acs:ram::1234567890:root"]
        }
      }
    ],
    "Version": "1"
  }'

# Create a narrowly scoped policy -- upload only, to one bucket prefix
aliyun ram CreatePolicy \
  --PolicyName STS-Upload-Only \
  --PolicyDocument '{
    "Version": "1",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "oss:PutObject",
          "oss:InitiateMultipartUpload",
          "oss:UploadPart",
          "oss:CompleteMultipartUpload",
          "oss:AbortMultipartUpload"
        ],
        "Resource": [
          "acs:oss:*:*:user-uploads",
          "acs:oss:*:*:user-uploads/uploads/*"
        ]
      }
    ]
  }'

# Attach the policy to the role
aliyun ram AttachPolicyToRole \
  --PolicyType Custom \
  --PolicyName STS-Upload-Only \
  --RoleName STS-OSS-Uploader
```

第二步 -- 后端扮演角色，把临时凭证返回给前端：

```python
from alibabacloud_sts20150401.client import Client
from alibabacloud_tea_openapi.models import Config
from alibabacloud_sts20150401.models import AssumeRoleRequest

config = Config(
    access_key_id='<backend-ak>',
    access_key_secret='<backend-sk>',
    endpoint='sts.cn-hangzhou.aliyuncs.com'
)
client = Client(config)

request = AssumeRoleRequest(
    role_arn='acs:ram::1234567890:role/sts-oss-uploader',
    role_session_name='user-12345-upload',
    duration_seconds=900  # 15 minutes -- keep it short
)
response = client.assume_role(request)

# Return these three values to the frontend
credentials = response.body.credentials
print(f"AccessKeyId:     {credentials.access_key_id}")
print(f"AccessKeySecret: {credentials.access_key_secret}")
print(f"SecurityToken:   {credentials.security_token}")
print(f"Expiration:      {credentials.expiration}")
```

第三步 -- 前端用临时凭证直传 OSS：

```javascript
// Browser-side upload using STS credentials
const OSS = require('ali-oss');

const client = new OSS({
  region: 'oss-cn-hangzhou',
  accessKeyId: stsCredentials.AccessKeyId,
  accessKeySecret: stsCredentials.AccessKeySecret,
  stsToken: stsCredentials.SecurityToken,
  bucket: 'user-uploads',
});

// Upload a file -- this goes directly to OSS, not through your backend
const result = await client.put(
  `uploads/${userId}/${filename}`,
  file
);
```

凭证 15 分钟后过期。如果用户还要上传，前端再找后端要新的。后端可以在发凭证这一步加业务逻辑（限流、文件类型校验、配额检查），字节还没进 OSS 就能先控一遍。

## KMS：密钥管理服务

KMS 负责加密这块硬骨头。它管理 cryptographic keys，用来加密/解密数据。你根本接触不到原始密钥材料——KMS 把它存在硬件安全模块（HSMs）里，代表你执行加密操作。

![KMS 信封加密流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_kms_encryption.png)

### 核心概念

| Concept | What it is |
|---|---|
| **CMK** (Customer Master Key) | 顶层密钥。永远不出 KMS。用来加密数据密钥。 |
| **Data Key** | KMS 生成的密钥，被 CMK 加密过。你用明文版本加密数据，把密文版本跟数据存一起。 |
| **Envelope Encryption** | 标准模式： KMS 生成数据密钥 → 你用明文数据密钥加密数据 → 把加密后的数据密钥跟密文数据存一起 → 解密时把加密数据密钥发给 KMS，拿回明文，再解密数据。 |
| **Symmetric Key** | 加密解密用同一把钥匙。 AES-256。用来加密数据。 |
| **Asymmetric Key** | 公钥/私钥对。 RSA 或 EC。用来签名和密钥交换。 |

### 为什么要用信封加密？

你可能会问：干嘛不把数据直接扔给 KMS 让它全权加密？因为 KMS 直接加密有 6 KB 限制。实际场景里 Anything larger （文件、数据库字段、磁盘卷）都得用信封加密。

流程如下：

1. 调用 `GenerateDataKey` -- KMS 返回明文数据密钥 AND 同一把密钥的密文副本
2. 用明文数据密钥在本地加密数据 (AES-256-GCM)
3. 把密文数据 + 密文数据密钥存一起
4. 从内存里清除明文数据密钥
5. 解密时：把密文数据密钥发给 KMS (`Decrypt`)，拿回明文，解密数据

这样 KMS 只处理一次小解密（数据密钥），大量加密操作由应用在本地完成。速度快，可扩展，主密钥也永远不出 KMS。

### 创建密钥并加密数据

创建对称 CMK：

```bash
aliyun kms CreateKey \
  --KeySpec Aliyun_AES_256 \
  --KeyUsage ENCRYPT/DECRYPT \
  --Origin Aliyun_KMS \
  --Description "Production data encryption key" \
  --ProtectionLevel HSM
```

`ProtectionLevel HSM` 表示密钥存在硬件安全模块里。贵一点，但符合 FIPS 140-2 Level 3 合规要求。

生成用于信封加密的数据密钥：

```bash
aliyun kms GenerateDataKey \
  --KeyId <your-cmk-id> \
  --KeySpec AES_256

# Response includes:
# - Plaintext: base64-encoded plaintext data key (use this to encrypt, then discard)
# - CiphertextBlob: base64-encoded encrypted data key (store this with your data)
```

直接加密一小段数据（6 KB 以下）：

```bash
# Encrypt
aliyun kms Encrypt \
  --KeyId <your-cmk-id> \
  --Plaintext "$(echo 'sensitive-api-key-value' | base64)"

# Decrypt
aliyun kms Decrypt \
  --CiphertextBlob <the-ciphertext-from-encrypt>
```

### 加密阿里云服务

大部分阿里云服务原生支持 KMS 加密。你提供 CMK ID，服务内部会自动处理信封加密。

| 服务 | 加密内容 | 如何启用 |
|---|---|---|
| **ECS** | 系统盘，数据盘 | 创建磁盘时加 `--Encrypted true --KMSKeyId <id>` |
| **OSS** | 静态对象 | Bucket SSE 设置：`x-oss-server-side-encryption: KMS` |
| **RDS** | 透明数据加密 (TDE) | 控制台：实例 > 数据安全 > TDE > 启用 |
| **NAS** | 文件系统数据 | 创建文件系统时选加密类型 |
| **ACK** | Kubernetes Secrets | 集群设置里启用 Secret 加密 |

为 OSS Bucket 启用服务端加密：

```bash
aliyun oss bucket-encryption --method put \
  oss://my-secure-bucket \
  --sse-algorithm KMS \
  --kms-masterkey-id <your-cmk-id>
```

创建实例时启用 ECS 磁盘加密：

```bash
aliyun ecs CreateInstance \
  --RegionId cn-hangzhou \
  --InstanceType ecs.g7.large \
  --ImageId ubuntu_22_04_x64 \
  --SystemDisk.Category cloud_essd \
  --SystemDisk.Encrypted true \
  --SystemDisk.KMSKeyId <your-cmk-id> \
  --VSwitchId <your-vswitch-id>
```

### 密钥轮换

KMS 支持自动密钥轮换。启用后， KMS 会按你的计划（比如每 90 天）创建新密钥版本。新加密操作会用最新版本。解密时会自动检测用的是哪个版本并正确解密。你不需要重新加密现有数据。

```bash
# Enable automatic rotation every 365 days
aliyun kms UpdateRotationPolicy \
  --KeyId <your-cmk-id> \
  --EnableAutomaticRotation true \
  --RotationInterval "365d"
```

手动轮换（适合应急响应——“我们怀疑这把密钥泄露了”）：

```bash
aliyun kms CreateKeyVersion --KeyId <your-cmk-id>
```
## ActionTrail：审计一切

ActionTrail 是审计体系的基石。它会记录针对你阿里云账户发出的每一次 API 调用——谁操作的、什么时候、来自哪个 IP、带了什么参数，以及是否成功。你就把它当成云上的黑匣子飞行记录仪。

![ActionTrail 审计流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_audit_trail.png)

### 记录的内容

ActionTrail 捕获两类事件：

- **管理事件（Management events）**：创建、修改或删除资源的 API 调用。比如 `CreateInstance`、`DeleteBucket`、`AttachPolicyToUser`。默认开启日志记录。
- **数据事件（Data events）**：在资源内部读写数据的 API 调用。比如 OSS 上的 `GetObject`， MNS 上的 `SendMessage`。因为数据量太大，这类需要手动开启。

每条事件记录都包含这些信息：

```json
{
  "eventId": "a1b2c3d4-...",
  "eventVersion": "1",
  "eventSource": "ecs.aliyuncs.com",
  "eventType": "ApiCall",
  "eventName": "StopInstance",
  "eventTime": "2026-05-18T03:24:15Z",
  "userIdentity": {
    "type": "ram-user",
    "principalId": "1234567890",
    "accountId": "1234567890",
    "userName": "alice",
    "accessKeyId": "LTAI5t..."
  },
  "sourceIpAddress": "203.0.113.42",
  "requestParameters": {
    "InstanceId": "i-bp1234567890abcdef",
    "ForceStop": "false"
  },
  "responseElements": {
    "RequestId": "e5f6g7h8-..."
  },
  "errorCode": "",
  "errorMessage": ""
}
```

这条日志告诉你： Alice 在 UTC 03:24 从 IP `203.0.113.42` 停止了实例 `i-bp1234567890abcdef`，操作成功了。要是有人凌晨 3 点删了生产环境的 RDS 实例，你能精确查到是谁干的、从哪干的、用了什么凭证。

### 配置 Trail

Trail 负责把审计事件投递到存储目的地。你得确保至少有一个 Trail 始终处于激活状态，把日志投递到你控制区域内的 OSS Bucket。

```bash
# Create an OSS bucket for audit logs (if you do not have one)
aliyun oss mb oss://company-audit-logs-cn-hangzhou

# Create the trail
aliyun actiontrail CreateTrail \
  --Name production-audit \
  --OssBucketName company-audit-logs-cn-hangzhou \
  --OssKeyPrefix actiontrail/ \
  --RoleName AliyunActionTrailDefaultRole \
  --IsOrganizationTrail false

# Start logging
aliyun actiontrail StartLogging --Name production-audit
```

日志会以 gzipped JSON 文件的形式到达 OSS Bucket，按日期分区：

```text
actiontrail/
  AliyunLogs/
    1234567890/
      2026/05/18/
        1234567890_actiontrail_cn-hangzhou_2026-05-18T03-00-00Z.json.gz
```

要做实时分析，还得投递一份到 SLS （Simple Log Service）：

```bash
aliyun actiontrail UpdateTrail \
  --Name production-audit \
  --SlsProjectArn acs:log:cn-hangzhou:1234567890:project/security-audit \
  --SlsWriteRoleArn acs:ram::1234567890:role/actiontrail-sls-role
```

### 查询审计事件

通过 CLI 查询最近的事件：

```bash
# Find all events by a specific user in the last 24 hours
aliyun actiontrail LookupEvents \
  --StartTime "2026-05-17T00:00:00Z" \
  --EndTime "2026-05-18T00:00:00Z" \
  --LookupAttribute '[{"Key":"UserName","Value":"alice"}]'

# Find all delete operations
aliyun actiontrail LookupEvents \
  --StartTime "2026-05-17T00:00:00Z" \
  --EndTime "2026-05-18T00:00:00Z" \
  --LookupAttribute '[{"Key":"EventName","Value":"Delete*"}]'
```

对于合规场景（SOC 2、 ISO 27001、 PCI DSS）， ActionTrail 提供审计证据链。核心要求就这几条：

- 日志必须防篡改：投递到开启版本控制的 OSS Bucket，并设置生命周期规则， N 年内禁止删除
- 日志必须覆盖所有账户：多账户架构使用组织 Trail （Organization Trail）
- 日志必须被监控：针对高风险事件设置 SLS 告警（root 登录、策略变更、安全组修改）

## 安全最佳实践清单

配好 RAM、 KMS 和 ActionTrail 之后，这是我针对每个阿里云账户都会过一遍的完整清单。打印出来，贴墙上。

**身份管理：**
- 给根账户（Root Alibaba Cloud Account）启用 MFA。有条件就用硬件密钥。
- 给每个真人创建 RAM 用户。千万别共享根凭证。
- 使用 RAM 用户组。别直接把策略绑给用户。
- 员工离职 24 小时内，删除或禁用对应的 RAM 用户。

**授权管理：**
- 遵循最小权限原则。从零权限开始，只加需要的。
- 常见场景用系统策略，其他情况写自定义策略。
- 除了 Administrators 组，别用 `"Action": "*", "Resource": "*"`。
- 每季度审查权限。利用 RAM 的 "last accessed" 数据找出未使用的权限。
- 每 90 天轮换一次 AccessKey。设个日历提醒。

**加密：**
- 所有 OSS Bucket 启用服务端加密（SSE-KMS）。
- 所有 ECS 实例启用磁盘加密。
- 包含敏感数据的 RDS 实例启用 TDE。
-  secrets、 API Key、 Token 应用层加密用 KMS。
- 启用自动密钥轮换（至少一年一次）。

**审计：**
- 在你使用的每个区域都启用 ActionTrail。
- 日志同时投递到 OSS （长期存储）和 SLS （实时分析）。
- 针对以下事件设置告警：根账户登录、创建 AccessKey、策略变更、安全组变更、跨账户角色假设。
- 给审计日志 Bucket 启用 OSS 版本控制。加一条禁止删除的生命周期规则。

**网络（详见 [Part 3](/zh/aliyun-fullstack/03-vpc-networking/)，但对安全至关重要）：**
- 别把 22 端口（SSH）对 `0.0.0.0/0` 开放。用堡垒机或 VPN。
- 支持私网端点的服务都用 VPC 私网 endpoint （OSS、 RDS、 KMS）。
- 安全组限制到具体的 CIDR 范围，别用 `0.0.0.0/0`。

**凭证管理：**
- 别在源代码里硬编码 AccessKey。用实例角色、 STS 或环境变量。
- 任何不可信客户端（移动端、浏览器、第三方）都用 STS 临时凭证。
- 第一次 commit 之前，先把 `.env`、`credentials` 和 `*.key` 加进 `.gitignore`。

## 解决方案：安全的多团队访问

我把这些内容整合一下。下面是一个完整 walkthrough，适合一家初创公司：三个团队（admin、 development、 stakeholders）需要不同级别的访问权限，有一个 ECS 应用要从 OSS 读数据，前端要上传到 OSS，还得有合规审计 trail。

### Step 1: 创建 RAM 用户组

```bash
# Admin group -- full access minus billing
aliyun ram CreateGroup --GroupName admin-team \
  --Comments "Infrastructure admins - full access"
aliyun ram AttachPolicyToGroup \
  --PolicyType System --PolicyName AdministratorAccess \
  --GroupName admin-team

# Dev group -- compute + storage + database, no RAM/billing
aliyun ram CreateGroup --GroupName dev-team \
  --Comments "Developers - compute, storage, database access"
aliyun ram CreatePolicy \
  --PolicyName dev-team-policy \
  --PolicyDocument '{
    "Version": "1",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["ecs:*", "oss:*", "rds:*", "vpc:Describe*", "dashscope:*"],
        "Resource": "*"
      },
      {
        "Effect": "Deny",
        "Action": ["ecs:DeleteInstance", "rds:DeleteDBInstance", "oss:DeleteBucket"],
        "Resource": "*"
      }
    ]
  }'
aliyun ram AttachPolicyToGroup \
  --PolicyType Custom --PolicyName dev-team-policy \
  --GroupName dev-team

# Readonly group -- view everything
aliyun ram CreateGroup --GroupName readonly-team \
  --Comments "Stakeholders - view only access"
aliyun ram AttachPolicyToGroup \
  --PolicyType System --PolicyName ReadOnlyAccess \
  --GroupName readonly-team
```

### Step 2: 创建 RAM 用户并分配到组

```bash
# Create users
for user in admin-wang dev-alice dev-bob readonly-carol; do
  aliyun ram CreateUser --UserName $user
  aliyun ram CreateLoginProfile \
    --UserName $user \
    --Password "ChangeMe@2026!" \
    --PasswordResetRequired true \
    --MFABindRequired true
done

# Assign to groups
aliyun ram AddUserToGroup --UserName admin-wang --GroupName admin-team
aliyun ram AddUserToGroup --UserName dev-alice --GroupName dev-team
aliyun ram AddUserToGroup --UserName dev-bob --GroupName dev-team
aliyun ram AddUserToGroup --UserName readonly-carol --GroupName readonly-team
```

### Step 3: 为 ECS 创建服务角色

```bash
# Role: ECS instances can read from the data bucket
aliyun ram CreateRole \
  --RoleName app-oss-reader \
  --AssumeRolePolicyDocument '{
    "Statement": [{"Action":"sts:AssumeRole","Effect":"Allow","Principal":{"Service":["ecs.aliyuncs.com"]}}],
    "Version": "1"
  }'

aliyun ram CreatePolicy \
  --PolicyName app-oss-read-policy \
  --PolicyDocument '{
    "Version": "1",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["oss:GetObject", "oss:ListObjects", "oss:GetBucketInfo"],
      "Resource": ["acs:oss:*:*:app-data-bucket", "acs:oss:*:*:app-data-bucket/*"]
    }]
  }'

aliyun ram AttachPolicyToRole \
  --PolicyType Custom --PolicyName app-oss-read-policy \
  --RoleName app-oss-reader
```

### Step 4: 为前端上传设置 STS 策略

```bash
# Role: backend can assume to get upload-only credentials
aliyun ram CreateRole \
  --RoleName sts-frontend-uploader \
  --AssumeRolePolicyDocument '{
    "Statement": [{"Action":"sts:AssumeRole","Effect":"Allow","Principal":{"RAM":["acs:ram::1234567890:root"]}}],
    "Version": "1"
  }'

aliyun ram CreatePolicy \
  --PolicyName frontend-upload-policy \
  --PolicyDocument '{
    "Version": "1",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["oss:PutObject", "oss:InitiateMultipartUpload", "oss:UploadPart", "oss:CompleteMultipartUpload"],
      "Resource": ["acs:oss:*:*:user-uploads", "acs:oss:*:*:user-uploads/uploads/*"]
    }]
  }'

aliyun ram AttachPolicyToRole \
  --PolicyType Custom --PolicyName frontend-upload-policy \
  --RoleName sts-frontend-uploader
```

### Step 5: 启用 ActionTrail

```bash
# Create audit log bucket with versioning
aliyun oss mb oss://company-audit-trail
aliyun oss bucket-versioning --method put oss://company-audit-trail --status Enabled

# Create and start the trail
aliyun actiontrail CreateTrail \
  --Name main-audit-trail \
  --OssBucketName company-audit-trail \
  --OssKeyPrefix logs/ \
  --RoleName AliyunActionTrailDefaultRole

aliyun actiontrail StartLogging --Name main-audit-trail
```

### Step 6: 为敏感数据创建 KMS 密钥

```bash
# Create the master key
aliyun kms CreateKey \
  --KeySpec Aliyun_AES_256 \
  --KeyUsage ENCRYPT/DECRYPT \
  --Description "Production secrets encryption" \
  --ProtectionLevel HSM

# Enable automatic rotation
aliyun kms UpdateRotationPolicy \
  --KeyId <cmk-id-from-above> \
  --EnableAutomaticRotation true \
  --RotationInterval "365d"

# Encrypt the OSS bucket
aliyun oss bucket-encryption --method put \
  oss://app-data-bucket \
  --sse-algorithm KMS \
  --kms-masterkey-id <cmk-id-from-above>
```

走完这六步，你就有了：三个权限范围恰当的 user 组，每个用户都强制 MFA， ECS 通过服务角色访问 OSS 无需存储凭证，前端上传用 STS 且 token 15 分钟过期，数据 Bucket 落地加密，还有一套投递到版本控制 OSS Bucket 的完整审计 trail。

通过 CLI 配置完这套东西大概只要 30 分钟。要是没配好，后续补救花的可就是事故响应的小时数，和数据清理的天数了。
## 总结

1. **别拿 root 账号干日常活儿。** 创建 RAM 用户，强制上 MFA，用用户组管权限。 root 账号得锁进保险柜里，最好别动。

2. **最小权限是个习惯，不是一劳永逸的设置。** 从零权限开始，缺啥加啥，每季度 review 一次。`"Action": "*"` 这种策略等同于把家门敞开，绝对不行。

3. **临时凭证永远比永久凭证香。** 只要涉及不可信环境（前端、移动端、第三方），统统用 STS。 ECS 和函数计算直接用实例角色。永久 AccessKey 只留给那些没法用角色的后端服务。

4. **静态数据全部加密。** KMS 让这事儿变得很简单——OSS 开启 SSE-KMS， ECS 开磁盘加密， RDS 开 TDE。性能损耗几乎可以忽略，但数据泄露的代价你承担不起。

5. **所有操作必须审计。** ActionTrail 的管理事件是免费的。第一天就开通，别等出了事再后悔。当问题发生时——肯定会发生的——审计日志是你第一个要去查的地方。

6. **安全是靠层层防御，不是单靠一堵墙。** 身份控制谁进得来，授权控制能干什么，加密确保即使进来了数据也拿不走，审计告诉你谁试过。每一层都在为其他层的失误兜底。

这篇文章起因的那个硬编码 API Key，让我赔上了一个周末和一笔不小的账单。要是生产库泄露或者 root 账号被攻陷，代价得翻好几个数量级。解决方案里那六步操作，半小时就能搞定。现在就做，别等“以后”。

## 下一步

在 [第 7 部分](/zh/aliyun-fullstack/07-observability) 里，我们要进入存储层了：对象存储 OSS、共享文件系统 NAS，以及支撑 ECS 的块存储选项。我们会沿用本文打下的安全基础——每个 Bucket 都开 SSE-KMS，所有访问走 RAM 角色，绝不用永久 AccessKey。