---
title: "Alibaba Cloud Full Stack (6): RAM, KMS, and Cloud Security"
date: 2026-05-03 09:00:00
tags:
  - Alibaba Cloud
  - RAM
  - KMS
  - Security
  - IAM
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 6
description: "Lock down your cloud: RAM users, groups, roles, and policies. STS for temporary credentials. KMS for encryption. ActionTrail for audit logging. Build a secure multi-team access model with least privilege."
disableNunjucks: true
translationKey: "aliyun-fullstack-6"
---

I once found a DashScope API key hardcoded in a public GitHub repo. It was mine. Someone had forked a demo I pushed months earlier, and the key was sitting in a config file I forgot to gitignore. By the time I noticed, the key had been used to generate 14,000 Qwen API calls in a single weekend. The bill was not catastrophic -- DashScope per-token pricing is forgiving -- but the lesson was. I had treated cloud security as something I would figure out later. "Later" arrived as a billing alert at 2 AM on a Sunday.

That was the day I set up RAM users, rotated every access key, enabled MFA, and started using STS for anything that touches a frontend. This article is everything I learned in the process, structured so you can do it in an afternoon instead of learning it from an incident.

![RAM and Security](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/cover.png)

Security groups -- the network-layer firewall -- are covered in [Part 3](/en/aliyun-fullstack/03-vpc-networking/). This article is about the identity layer: who can do what, how to encrypt data, and how to audit everything. For Terraform-managed security, see [Terraform Part 6: LLM Gateway and Secrets](/en/terraform-agents/06-llm-gateway-and-secrets/).

## The Security Mental Model

Cloud security is not a single feature you turn on. It is a stack of independent layers, each covering a different failure mode. Miss one layer and the others still protect you -- that is the principle of defense in depth.

![Alibaba Cloud security model overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_security_model.png)

I think about it as four pillars:

| Pillar | Question it answers | Alibaba Cloud service | AWS equivalent |
|---|---|---|---|
| **Identity** | Who is making this request? | RAM (users, groups) | IAM (users, groups) |
| **Authorization** | What are they allowed to do? | RAM (policies, roles) | IAM (policies, roles) |
| **Encryption** | Is the data protected at rest and in transit? | KMS, SSL certificates | KMS, ACM |
| **Auditing** | Who did what, and when? | ActionTrail | CloudTrail |

Every security decision you make falls into one of these four buckets. When something goes wrong -- and it will -- the audit trail tells you which of the other three failed. When you design access for a new team, you walk through all four: create identities, assign permissions, encrypt their data, and log their actions.

The mental model maps cleanly to AWS IAM, which is intentional. Alibaba Cloud built RAM as a near-equivalent to AWS IAM, with the same conceptual hierarchy: root account at the top, RAM users underneath, policies granting permissions, roles for cross-service and cross-account access. If you have used AWS IAM, you already know 80% of what RAM does. The remaining 20% is naming differences and a few features that work slightly differently.

One critical difference: Alibaba Cloud's root account is called the "Alibaba Cloud Account" or sometimes the "primary account." It is not called "root" in the console, but functionally it is the same thing -- an all-powerful identity that should never be used for daily work.

## RAM: Resource Access Management

RAM is the identity and access management system for Alibaba Cloud. Every API call, every console click, every CLI command is authenticated and authorized through RAM. Understanding RAM is not optional -- it is the foundation that everything else in this article builds on.

![RAM user, group, and role hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_ram_hierarchy.png)

### The Alibaba Cloud Account (Root)

When you sign up for Alibaba Cloud, you get an Alibaba Cloud Account. This is the root identity. It has unrestricted access to every service, every resource, every billing setting. It can create and delete RAM users, change payment methods, close the account entirely.

You should use this account for exactly three things:

1. Initial setup (creating your first RAM admin user)
2. Billing and payment method changes
3. Emergency recovery when RAM is misconfigured

For everything else -- development, deployment, operations, monitoring -- use RAM users. I have seen teams where six engineers all share the root account credentials. One person accidentally deletes a production RDS instance, and nobody can figure out who did it because ActionTrail shows "root" for every action. Separate identities are not bureaucracy; they are how you debug incidents.

### Creating RAM Users

A RAM user is a permanent identity within your Alibaba Cloud Account. Each user gets their own login credentials (password for console, AccessKey for API/CLI) and their own set of permissions.

Create a RAM user via CLI:

```bash
aliyun ram CreateUser \
  --UserName alice \
  --DisplayName "Alice Chen" \
  --Comments "Backend developer, team-alpha"
```

Enable console login with a password:

```bash
aliyun ram CreateLoginProfile \
  --UserName alice \
  --Password 'TempP@ssw0rd!2026' \
  --PasswordResetRequired true \
  --MFABindRequired true
```

The `--PasswordResetRequired true` flag forces Alice to change her password on first login. The `--MFABindRequired true` flag forces MFA setup before she can do anything. Both are non-negotiable for any account that has write access to production resources.

Create an AccessKey pair for programmatic access:

```bash
aliyun ram CreateAccessKey --UserName alice
```

This returns an AccessKeyId and AccessKeySecret. The secret is shown exactly once -- if you lose it, you have to create a new key pair. Store it in a password manager, not in a config file, not in an environment variable on a shared server, and absolutely not in a Git repository.

### Setting Up MFA

Multi-factor authentication adds a second layer to the identity pillar. Even if someone steals a password, they cannot log in without the TOTP code from a phone app.

Enable virtual MFA for a RAM user:

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

For the root account specifically, go to the console: **Account Management > Security Settings > MFA**. Use a hardware key if you have one. The root MFA device should be stored in a safe, not on the CEO's phone that they replace every year.

## RAM Groups

Managing permissions per user does not scale. When you have 3 developers, attaching policies to each user is fine. When you have 30, it becomes a maintenance nightmare. One developer changes teams and you forget to remove their old permissions. Another developer joins and you copy-paste policies from someone else, accidentally granting production delete access to a junior hire.

Groups solve this. A RAM group is a container for users. You attach policies to the group, and every user in the group inherits those policies. When someone changes teams, you move them between groups. When a new hire starts, you add them to the right group and they get exactly the permissions they need.

Here is the group structure I use for most projects:

| Group | Purpose | Key policies |
|---|---|---|
| `Administrators` | Full access to everything except billing | AdministratorAccess |
| `Developers` | Read/write to compute, storage, database | Custom: ECS/OSS/RDS full, no RAM/billing |
| `ReadOnly` | View everything, change nothing | ReadOnlyAccess |
| `Billing` | Manage payments and cost analysis | Custom: BSS full access |
| `CICD` | Deploy pipelines (not humans) | Custom: ECS/CR/ACK deploy-only |

Create groups and add users:

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

List all users in a group:

```bash
aliyun ram ListUsersForGroup --GroupName Developers
```

Remove a user when they change teams:

```bash
aliyun ram RemoveUserFromGroup --UserName alice --GroupName Developers
aliyun ram AddUserToGroup --UserName alice --GroupName Administrators
```

The key discipline: never attach policies directly to users. Always go through groups. The one exception is deny policies for specific users who need restricted access within their group -- but even that is better handled with a separate group.

## RAM Policies Deep Dive

Policies are the authorization engine. Every API call in Alibaba Cloud is evaluated against the caller's attached policies to decide: allow or deny. Understanding how policies work is the difference between "it works" and "it works securely."

![RAM policy evaluation flowchart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_policy_evaluation.png)

### System Policies vs Custom Policies

Alibaba Cloud provides over 800 system policies -- pre-built permission sets maintained by Alibaba Cloud. You cannot modify them, but they cover the most common scenarios:

| System policy | What it grants |
|---|---|
| `AdministratorAccess` | Full access to all services and resources |
| `ReadOnlyAccess` | Read-only access to all services |
| `AliyunECSFullAccess` | Full access to ECS |
| `AliyunOSSFullAccess` | Full access to OSS |
| `AliyunRDSFullAccess` | Full access to RDS |
| `AliyunVPCFullAccess` | Full access to VPC |
| `AliyunRAMFullAccess` | Full access to RAM (dangerous -- this is the keys to the kingdom) |
| `AliyunKMSFullAccess` | Full access to KMS |
| `AliyunActionTrailFullAccess` | Full access to ActionTrail |
| `AliyunBSSFullAccess` | Full access to billing |

For anything beyond these broad strokes, you need custom policies.

### Policy Structure

A RAM policy is a JSON document with a specific structure. Every policy has a Version and one or more Statements. Each Statement has an Effect (Allow or Deny), an Action (what API operations), a Resource (which specific resources), and optionally a Condition (when the rule applies).

Here is the anatomy:

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

Breaking this down:

- **Version**: Always `"1"`. Alibaba Cloud RAM currently has only one policy version.
- **Effect**: `"Allow"` or `"Deny"`. Deny always wins over Allow when both match.
- **Action**: The API operations. Supports wildcards: `ecs:*` means all ECS operations, `ecs:Describe*` means all ECS read operations.
- **Resource**: The Alibaba Cloud Resource Name (ARN). Format: `acs:{service}:{region}:{account-id}:{resource-type}/{resource-id}`. Use `*` for all resources.
- **Condition**: Optional constraints. Common ones: source IP, MFA present, time of day, request tag values.

### Real Policy Examples

**ECS administrator -- full ECS access in one region only:**

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

The second statement grants VPC read access -- necessary because ECS operations often need to query VPC/VSwitch information. Without it, creating instances fails with an authorization error that does not mention VPC at all, which is confusing.

**OSS read-only for a specific bucket:**

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

Note the two Resource lines: the first grants access to the bucket itself (for `ListObjects`), the second grants access to objects within the bucket (for `GetObject`). Missing either one produces confusing 403 errors.

**DashScope API access only (for AI developers):**

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

This grants full DashScope access but explicitly denies deletion operations. The Deny overrides the Allow, so even if someone has `dashscope:*`, they cannot delete models or deployments. This is a common pattern: broad Allow plus targeted Deny for destructive operations.

**Require MFA for sensitive operations:**

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

The first statement says "allow everything, but only if MFA is active." The second statement allows MFA-related actions without MFA (so the user can actually set up MFA in the first place). Without the second statement, a new user would be locked in a catch-22: cannot do anything without MFA, cannot set up MFA because that requires permissions.

### RBAC vs ABAC

RAM supports two permission models, and most setups use both:

**RBAC (Role-Based Access Control)**: Permissions are assigned based on the user's role (group membership). "All Developers can start/stop ECS instances." This is what groups provide.

**ABAC (Attribute-Based Access Control)**: Permissions are assigned based on resource attributes, typically tags. "Users can only manage instances tagged with `team=alpha`."

ABAC example -- users can only manage their own team's instances:

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

This policy says: allow ECS operations only when the resource's `team` tag matches the user's `team` tag. Tag user Alice with `team=alpha`, tag her instances with `team=alpha`, and she can manage them. She cannot touch instances tagged `team=beta`, even though the Action says `ecs:*`.

ABAC is powerful but harder to debug. I recommend starting with RBAC (groups + policies) and adding ABAC only when you need tag-based isolation -- typically when multiple teams share the same account.

Create and attach a custom policy:

```bash
# Create the policy
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

# Attach to a group
aliyun ram AttachPolicyToGroup \
  --PolicyType Custom \
  --PolicyName ECS-HangZhou-Admin \
  --GroupName Developers
```

## RAM Roles

RAM users are permanent identities for humans (and CI/CD pipelines). RAM roles are temporary identities designed for three scenarios:

![ABAC vs RBAC comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_abac_rbac.png)

1. **Service roles**: An Alibaba Cloud service (ECS, Function Compute, etc.) needs to access another service (OSS, RDS, etc.)
2. **Cross-account access**: A user in Account A needs to access resources in Account B
3. **Federated login (SSO)**: Users from an external identity provider (LDAP, SAML, OIDC) need Alibaba Cloud access

The key difference between a user and a role:

| Aspect | RAM User | RAM Role |
|---|---|---|
| Identity type | Permanent | Temporary (assumed) |
| Credentials | Password + AccessKey | STS token (auto-expires) |
| Who uses it | Humans, CI/CD bots | Services, cross-account, SSO |
| MFA support | Yes | No (trust policy handles it) |
| Direct login | Yes (console) | No (must be assumed) |
| Max session | Permanent | 1 hour (configurable to 12h) |

### Trust Policy

Every role has a trust policy that specifies who can assume it. This is separate from the permission policy (which specifies what the role can do once assumed). Think of it as: the trust policy is the door lock, the permission policy is the key ring inside.

### Service Role Example: ECS Accessing OSS

A common scenario: your ECS instance needs to read files from an OSS bucket. The wrong way to do this is to put an AccessKey in the instance's environment variables. If the instance is compromised, the attacker has permanent credentials. The right way is an instance role -- the ECS instance automatically gets temporary credentials that rotate every hour.

Step 1 -- Create the role with a trust policy allowing ECS to assume it:

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

Step 2 -- Attach a permission policy to the role:

```bash
aliyun ram AttachPolicyToRole \
  --PolicyType Custom \
  --PolicyName OSS-DataBucket-ReadOnly \
  --RoleName ECS-OSS-Reader
```

Step 3 -- Attach the role to an ECS instance:

```bash
aliyun ecs AttachInstanceRamRole \
  --RegionId cn-hangzhou \
  --InstanceIds '["i-bp1234567890abcdef"]' \
  --RamRoleName ECS-OSS-Reader
```

Step 4 -- Inside the ECS instance, the SDK automatically picks up the role credentials:

```python
from alibabacloud_oss20190517.client import Client
from alibabacloud_tea_openapi.models import Config

# No AccessKey needed -- SDK uses instance metadata to get STS tokens
config = Config(
    region_id='cn-hangzhou',
    credential=None  # SDK auto-discovers instance role credentials
)
client = Client(config)

# Read from OSS as usual
result = client.get_object('my-data-bucket', 'data/input.csv')
```

The SDK calls the instance metadata service at `http://100.100.100.200/latest/meta-data/ram/security-credentials/ECS-OSS-Reader` to get temporary credentials. These credentials rotate automatically. No AccessKey is stored on the instance. If the instance is compromised, the attacker gets credentials that expire within the hour, and you can revoke the role immediately.

### Cross-Account Role

When Account B (ID: `9876543210`) needs to let users from Account A (ID: `1234567890`) manage its ECS instances:

```bash
# In Account B: create a role trusting Account A
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

# Attach ECS admin policy to the role
aliyun ram AttachPolicyToRole \
  --PolicyType System \
  --PolicyName AliyunECSFullAccess \
  --RoleName CrossAccount-ECS-Admin
```

In Account A, the user assumes the role:

```bash
aliyun sts AssumeRole \
  --RoleArn acs:ram::9876543210:role/crossaccount-ecs-admin \
  --RoleSessionName alice-cross-account \
  --DurationSeconds 3600
```

The Condition block requires MFA, adding a second verification layer for cross-account access.

## STS: Temporary Credentials

Security Token Service generates temporary AccessKey pairs with an attached security token. They work exactly like regular AccessKeys but expire automatically. This is the mechanism behind RAM roles, and you can also use it directly for scenarios like mobile uploads and frontend access.

![STS temporary credential flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_sts_flow.png)

### Why Temporary Beats Permanent

Permanent AccessKeys are a liability. They do not expire. If leaked, they remain valid until you manually rotate them. Rotation means updating every service that uses the key, which means downtime or a coordinated deployment. Most teams put off rotation because it is painful, which means leaked keys stay active for months.

STS tokens expire. The maximum lifetime is 12 hours (default 1 hour, minimum 15 minutes). If a token is leaked, the damage window is small. You do not need to rotate anything -- just wait for it to expire, then investigate how it leaked.

### The STS Workflow

The flow is: trusted backend assumes a role, gets temporary credentials, passes them to the untrusted client (mobile app, browser, third-party service), client uses the credentials until they expire.

```
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

### Complete STS Example: Frontend Upload to OSS

This is the most common STS use case. A mobile app or browser needs to upload files directly to OSS. You do not want the upload to go through your backend (bandwidth and latency), but you also do not want permanent OSS credentials in the frontend.

Step 1 -- Create a role with a narrowly scoped OSS policy:

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

Step 2 -- Backend assumes the role and returns temporary credentials to the frontend:

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

Step 3 -- Frontend uses the temporary credentials to upload directly to OSS:

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

The credentials expire after 15 minutes. If the user needs to upload more files, the frontend requests new credentials from your backend. The backend can add business logic (rate limiting, file type validation, quota checking) at the credential-issuance step, before any bytes reach OSS.

## KMS: Key Management Service

KMS handles the encryption pillar. It manages cryptographic keys and uses them to encrypt/decrypt data. You never see the raw key material -- KMS keeps it in hardware security modules (HSMs) and performs cryptographic operations on your behalf.

![KMS envelope encryption flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_kms_encryption.png)

### Key Concepts

| Concept | What it is |
|---|---|
| **CMK** (Customer Master Key) | The top-level key. Never leaves KMS. Used to encrypt data keys. |
| **Data Key** | A key generated by KMS, encrypted under a CMK. You use the plaintext version to encrypt your data, store the encrypted version alongside it. |
| **Envelope Encryption** | The pattern: KMS generates a data key → you encrypt data with the plaintext data key → you store the encrypted data key with the encrypted data → to decrypt, you send the encrypted data key to KMS, get the plaintext back, decrypt your data. |
| **Symmetric Key** | Same key for encrypt and decrypt. AES-256. Used for data encryption. |
| **Asymmetric Key** | Public/private pair. RSA or EC. Used for signatures and key exchange. |

### Why Envelope Encryption?

You might ask: why not just send my data to KMS and let it encrypt everything directly? Because KMS has a 6 KB limit on direct encryption. For anything larger (which is everything in practice -- files, database fields, disk volumes), you use envelope encryption.

The flow:

1. Call `GenerateDataKey` -- KMS returns a plaintext data key AND an encrypted copy of the same key
2. Use the plaintext data key to encrypt your data locally (AES-256-GCM)
3. Store the encrypted data + the encrypted data key together
4. Discard the plaintext data key from memory
5. To decrypt: send the encrypted data key to KMS (`Decrypt`), get the plaintext back, decrypt your data

This way, KMS handles one small decryption (the data key), and your application handles the bulk encryption locally. Fast, scalable, and the master key never leaves KMS.

### Creating Keys and Encrypting Data

Create a symmetric CMK:

```bash
aliyun kms CreateKey \
  --KeySpec Aliyun_AES_256 \
  --KeyUsage ENCRYPT/DECRYPT \
  --Origin Aliyun_KMS \
  --Description "Production data encryption key" \
  --ProtectionLevel HSM
```

The `ProtectionLevel HSM` means the key is stored in a hardware security module. It costs more but provides FIPS 140-2 Level 3 compliance.

Generate a data key for envelope encryption:

```bash
aliyun kms GenerateDataKey \
  --KeyId <your-cmk-id> \
  --KeySpec AES_256

# Response includes:
# - Plaintext: base64-encoded plaintext data key (use this to encrypt, then discard)
# - CiphertextBlob: base64-encoded encrypted data key (store this with your data)
```

Encrypt a small piece of data directly (under 6 KB):

```bash
# Encrypt
aliyun kms Encrypt \
  --KeyId <your-cmk-id> \
  --Plaintext "$(echo 'sensitive-api-key-value' | base64)"

# Decrypt
aliyun kms Decrypt \
  --CiphertextBlob <the-ciphertext-from-encrypt>
```

### Encrypting Alibaba Cloud Services

Most Alibaba Cloud services support KMS encryption natively. You provide your CMK ID and the service handles envelope encryption internally.

| Service | What gets encrypted | How to enable |
|---|---|---|
| **ECS** | System disk, data disks | `--Encrypted true --KMSKeyId <id>` at disk creation |
| **OSS** | Objects at rest | Bucket SSE setting: `x-oss-server-side-encryption: KMS` |
| **RDS** | Transparent Data Encryption (TDE) | Console: Instance > Data Security > TDE > Enable |
| **NAS** | File system data | Encryption type at filesystem creation |
| **ACK** | Kubernetes Secrets | Enable Secret encryption in cluster settings |

Enable server-side encryption for an OSS bucket:

```bash
aliyun oss bucket-encryption --method put \
  oss://my-secure-bucket \
  --sse-algorithm KMS \
  --kms-masterkey-id <your-cmk-id>
```

Enable ECS disk encryption when creating an instance:

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

### Key Rotation

KMS supports automatic key rotation. When you enable it, KMS creates a new key version on your schedule (e.g., every 90 days). New encryption operations use the latest version. Decryption automatically detects which version was used and decrypts correctly. You do not need to re-encrypt existing data.

```bash
# Enable automatic rotation every 365 days
aliyun kms UpdateRotationPolicy \
  --KeyId <your-cmk-id> \
  --EnableAutomaticRotation true \
  --RotationInterval "365d"
```

For manual rotation (useful for incident response -- "we think this key might be compromised"):

```bash
aliyun kms CreateKeyVersion --KeyId <your-cmk-id>
```

## ActionTrail: Audit Everything

ActionTrail is the auditing pillar. It records every API call made against your Alibaba Cloud account -- who did it, when, from what IP, with what parameters, and whether it succeeded. Think of it as the black box flight recorder for your cloud.

![ActionTrail audit pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/06-ram-security/06_audit_trail.png)

### What Gets Logged

ActionTrail captures two categories of events:

- **Management events**: API calls that create, modify, or delete resources. `CreateInstance`, `DeleteBucket`, `AttachPolicyToUser`. These are logged by default.
- **Data events**: API calls that read or write data within resources. `GetObject` on OSS, `SendMessage` on MNS. These are opt-in because of volume.

Each event record includes:

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

This tells you: Alice stopped instance `i-bp1234567890abcdef` at 03:24 UTC from IP `203.0.113.42`, and it succeeded. If someone deletes a production RDS instance at 3 AM, you can find exactly who did it, from where, and what credentials they used.

### Setting Up a Trail

A trail delivers audit events to a storage destination. You should have at least one trail active at all times, delivering to an OSS bucket in a region you control.

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

The logs arrive in the OSS bucket as gzipped JSON files, partitioned by date:

```
actiontrail/
  AliyunLogs/
    1234567890/
      2026/05/18/
        1234567890_actiontrail_cn-hangzhou_2026-05-18T03-00-00Z.json.gz
```

For real-time analysis, also deliver to SLS (Simple Log Service):

```bash
aliyun actiontrail UpdateTrail \
  --Name production-audit \
  --SlsProjectArn acs:log:cn-hangzhou:1234567890:project/security-audit \
  --SlsWriteRoleArn acs:ram::1234567890:role/actiontrail-sls-role
```

### Querying Audit Events

Look up recent events via CLI:

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

For compliance use cases (SOC 2, ISO 27001, PCI DSS), ActionTrail provides the audit evidence trail. The key requirements are:

- Logs must be tamper-proof: deliver to an OSS bucket with versioning enabled and a lifecycle rule that prevents deletion for N years
- Logs must cover all accounts: use organization trails for multi-account setups
- Logs must be monitored: set up SLS alerts for high-risk events (root login, policy changes, security group modifications)

## Security Best Practices Checklist

After setting up RAM, KMS, and ActionTrail, here is the full checklist I run through for every Alibaba Cloud account. Print this and tape it to your wall.

**Identity:**
- Enable MFA on the root Alibaba Cloud Account. Use a hardware key if possible.
- Create RAM users for every human. Never share the root credentials.
- Use RAM groups. Never attach policies directly to users.
- Delete or deactivate unused RAM users within 24 hours of offboarding.

**Authorization:**
- Follow least privilege. Start with zero permissions and add only what is needed.
- Use system policies for common scenarios, custom policies for everything else.
- Never use `"Action": "*", "Resource": "*"` except for the Administrators group.
- Review permissions quarterly. Use RAM's "last accessed" data to find unused permissions.
- Rotate AccessKeys every 90 days. Set calendar reminders.

**Encryption:**
- Enable server-side encryption (SSE-KMS) on all OSS buckets.
- Enable disk encryption on all ECS instances.
- Enable TDE on RDS instances containing sensitive data.
- Use KMS for application-level encryption of secrets, API keys, tokens.
- Enable automatic key rotation (annually at minimum).

**Auditing:**
- Enable ActionTrail in every region you use.
- Deliver logs to both OSS (long-term storage) and SLS (real-time analysis).
- Set up alerts for: root account login, AccessKey creation, policy changes, security group changes, cross-account role assumptions.
- Enable OSS versioning on the audit log bucket. Add a lifecycle rule preventing deletion.

**Network (covered in [Part 3](/en/aliyun-fullstack/03-vpc-networking/) but critical for security):**
- Never open port 22 (SSH) to `0.0.0.0/0`. Use bastion hosts or VPN.
- Use VPC private endpoints for services that support them (OSS, RDS, KMS).
- Restrict security groups to specific CIDR ranges, not `0.0.0.0/0`.

**Credentials:**
- Never hardcode AccessKeys in source code. Use instance roles, STS, or environment variables.
- Use STS temporary credentials for any untrusted client (mobile, browser, third-party).
- Add `.env`, `credentials`, and `*.key` to your `.gitignore` before the first commit.

## Solution: Secure Multi-Team Access

Let me put it all together. Here is a complete walkthrough for a startup with three teams (admin, development, stakeholders) that need different levels of access, an ECS application that reads from OSS, a frontend that uploads to OSS, and an audit trail for compliance.

### Step 1: Create the RAM Groups

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

### Step 2: Create RAM Users and Assign to Groups

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

### Step 3: Create Service Role for ECS

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

### Step 4: Set Up STS Policy for Frontend Upload

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

### Step 5: Enable ActionTrail

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

### Step 6: Create KMS Key for Sensitive Data

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

After these six steps, you have: three groups with properly scoped permissions, MFA enforced on every user, a service role so ECS accesses OSS without stored credentials, STS for frontend uploads with 15-minute token expiry, encryption at rest for your data bucket, and a complete audit trail delivered to a versioned OSS bucket.

The total time to set this up is about 30 minutes via CLI. The total time to recover from not setting it up is measured in incident-response hours and compromised-data cleanup days.

## Key Takeaways

1. **Never use the root account for daily work.** Create RAM users, enforce MFA, use groups for permission management. The root account should be locked in a metaphorical safe.

2. **Least privilege is a practice, not a one-time setup.** Start with zero permissions, add what is needed, review quarterly. The `"Action": "*"` policy is the security equivalent of leaving your front door open.

3. **Temporary credentials beat permanent credentials every time.** Use STS for anything that touches an untrusted environment (frontend, mobile, third-party). Use instance roles for ECS and Function Compute. Reserve permanent AccessKeys for backend services that cannot use roles.

4. **Encrypt everything at rest.** KMS makes this trivial -- enable SSE-KMS on OSS, disk encryption on ECS, TDE on RDS. The performance overhead is negligible. The cost of a data breach is not.

5. **Audit everything, always.** ActionTrail is free for management events. Enable it on day one, not after the first incident. When something goes wrong -- and it will -- the audit trail is the first thing you reach for.

6. **Security is layers, not a single wall.** Identity controls who gets in. Authorization controls what they can do. Encryption protects data even if someone gets through. Auditing tells you when someone tried. Each layer compensates for failures in the others.

The hardcoded API key that kicked off this article cost me a weekend and a modest bill. A production database leak or a compromised root account costs orders of magnitude more. The six steps in the solution section are 30 minutes of work. Do them now, not "later."

## What's Next

In [Part 7](/en/aliyun-fullstack/07-oss-storage/), we move to the storage layer: OSS for object storage, NAS for shared filesystems, and the block storage options that back your ECS instances. We will build on the security foundations from this article -- every bucket gets SSE-KMS, every access goes through RAM roles, and nothing gets a permanent AccessKey.
