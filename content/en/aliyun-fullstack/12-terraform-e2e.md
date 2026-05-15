---
title: "Alibaba Cloud Full Stack (12): End-to-End — One Terraform Apply for Everything"
date: 2026-05-09 09:00:00
tags:
  - Alibaba Cloud
  - Terraform
  - Infrastructure as Code
  - CI/CD
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 12
series_total: 12
description: "The grand finale: codify everything from Parts 1-11 into Terraform modules. VPC, ECS, RDS, OSS, RAM, SLS, Function Compute — all provisioned with one terraform apply. Plus CI/CD with GitHub Actions and cost optimization."
disableNunjucks: true
translationKey: "aliyun-fullstack-12"
---

Eleven articles. Dozens of CLI commands. Hundreds of manual steps. Now we throw all of that away and rebuild the entire stack with a single `terraform apply`. This is why infrastructure-as-code exists.

Over the past eleven parts of this series, we have clicked through consoles, typed `aliyun` CLI commands, and manually configured everything from VPCs to Function Compute triggers. It worked. We learned every resource intimately because we built each one by hand. But if I asked you right now to recreate that entire stack in a new region — the VPC with its three tiers and two availability zones, the ECS instance with its cloud-init script, the RDS MySQL HA setup, the OSS bucket with lifecycle rules, the RAM policies, the SLS log pipeline, the Function Compute event processing — you would need at least a full day of careful work. And you would inevitably miss something. A security group rule. A backup policy. A CORS configuration.

Infrastructure-as-code eliminates that problem entirely. You describe what you want in declarative configuration files, and the tool figures out how to get there. The entire stack we built across eleven articles becomes a single repository of `.tf` files that anyone on your team can read, review, modify, and apply.

This is the grand finale. We will take everything from Parts 1 through 11 and encode it into Terraform modules. By the end of this article, you will have a complete, production-grade Terraform project that provisions your entire Alibaba Cloud infrastructure with one command.



---

## Why Infrastructure as Code?

If you're reading the twelfth article in this series, you probably don't need to be convinced. But let me lay out the case precisely, as these are the arguments you'll use to convince your team to invest the time.

### The progression

Every team goes through a similar evolution.

**Stage 1: Manual (console clicks).** You click through the web console. It works for one person managing a few resources. It falls apart when you need to recreate something, explain what you did, or hand the environment to a teammate.

**Stage 2: Scripts (CLI commands).** You write shell scripts that call `aliyun ecs CreateInstance` and similar commands. Better than clicking, but the scripts are imperative — they describe the steps, not the desired end state. If you run the script twice, it either fails (resource already exists) or creates duplicates. You end up writing increasingly complex logic to handle idempotency, and you have reinvented a bad version of Terraform.

**Stage 3: Infrastructure as Code (declarative).** You describe the desired state: "I want a VPC with CIDR 10.0.0.0/16 and three VSwitches." The tool compares the desired state to the actual state and figures out the minimum set of API calls to reconcile them. Run it once, run it a hundred times — the result is the same.

### The five pillars

Infrastructure-as-code provides five benefits that manual provisioning cannot:

| Pillar | What it means | Why it matters |
|---|---|---|
| **Reproducibility** | Same code = same infrastructure, every time | Spin up identical staging/production environments |
| **Version control** | Infrastructure changes tracked in Git | Who changed what, when, and why — with full diff history |
| **Collaboration** | Pull requests for infrastructure changes | Code review for infrastructure, just like application code |
| **Disaster recovery** | Recreate entire stack from code in minutes | Region goes down? Redeploy to a new region with one command |
| **Cost tracking** | Infrastructure defined in code can be cost-estimated | Know what a change will cost before applying it |

### Terraform vs Alibaba Cloud ROS

Alibaba Cloud has its own IaC service: Resource Orchestration Service (ROS). It uses JSON or YAML templates and is tightly integrated with the Alibaba Cloud console. It's free. So why do I recommend Terraform instead?

| Criteria | Terraform | ROS |
|---|---|---|
| **Multi-cloud** | Yes — AWS, Azure, GCP, 3000+ providers | Alibaba Cloud only |
| **State management** | Local or remote (S3, OSS, Consul, etc.) | Managed by ROS service |
| **Community** | Massive — modules, examples, Stack Overflow | Small, mostly Chinese-language |
| **Language** | HCL (purpose-built, readable) | JSON/YAML (verbose, error-prone) |
| **Module ecosystem** | Terraform Registry with thousands of modules | Limited |
| **Learning investment** | Transferable to any cloud | Alibaba-specific |
| **Drift detection** | `terraform plan` shows exact diff | Limited drift detection |
| **Import existing resources** | `terraform import` | Supported but less mature |

ROS is fine if your entire world is Alibaba Cloud and you want zero additional tooling. But if you work across clouds, value community support, or want skills that transfer to your next job, Terraform is the clear choice. The Alibaba Cloud Terraform provider (`alicloud`) is actively maintained and covers effectively every service we have used in this series.

For a much deeper treatment of Terraform itself, see our [Terraform series](/en/terraform-agents/01-why-terraform-for-agents/) — eight articles covering every aspect from first principles to advanced patterns. This article assumes you know the basics and focuses on applying Terraform to the specific Alibaba Cloud stack we have built.

## Architecture Overview

Here is the complete stack we are codifying. Every component corresponds to something we built by hand in a previous article:

![Full-stack architecture on Alibaba Cloud](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_full_architecture.png)

```text
┌─────────────────────────────────────────────────────────────┐
│                        Alibaba Cloud                        │
│                                                             │
│  ┌──────────────────── VPC (10.0.0.0/16) ─────────────────┐ │
│  │                                                         │ │
│  │  ┌─── AZ-A ───┐         ┌─── AZ-B ───┐                │ │
│  │  │ vsw-pub-a  │         │ vsw-pub-b  │  Public Tier    │ │
│  │  │ 10.0.1.0/24│         │ 10.0.2.0/24│                 │ │
│  │  └─────┬──────┘         └──────┬─────┘                 │ │
│  │        │                       │                        │ │
│  │  ┌─────┴──────┐         ┌──────┴─────┐                │ │
│  │  │ vsw-app-a  │         │ vsw-app-b  │  App Tier       │ │
│  │  │10.0.10.0/24│         │10.0.11.0/24│                 │ │
│  │  │  ┌─────┐   │         │            │                 │ │
│  │  │  │ ECS │   │         │            │                 │ │
│  │  └──┴─────┴───┘         └────────────┘                 │ │
│  │        │                       │                        │ │
│  │  ┌─────┴──────┐         ┌──────┴─────┐                │ │
│  │  │ vsw-data-a │         │ vsw-data-b │  Data Tier      │ │
│  │  │10.0.20.0/24│         │10.0.21.0/24│                 │ │
│  │  │  ┌─────┐   │         │  ┌─────┐   │                │ │
│  │  │  │ RDS │◄──┼─────────┼──│ RDS │   │                │ │
│  │  │  │(pri)│   │         │  │(sec)│   │                 │ │
│  │  └──┴─────┴───┘         └──┴─────┴───┘                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─── OSS ───┐  ┌── RAM ──┐  ┌── SLS ──┐  ┌──── FC ────┐  │
│  │  Media    │  │ Users   │  │ Logs    │  │ Event      │  │
│  │  Storage  │  │ Roles   │  │ Alerts  │  │ Processing │  │
│  │  CDN      │  │ Policies│  │ Dashb.  │  │ OSS Trigger│  │
│  └───────────┘  └─────────┘  └─────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The resources and where we built them:

| Resource | Series Part | Terraform Module |
|---|---|---|
| VPC, VSwitches, Security Groups, NAT | [Part 3: VPC Networking](/en/aliyun-fullstack/03-vpc-networking/) | `modules/network` |
| ECS instance, cloud-init, key pair | [Part 2: ECS Compute](/en/aliyun-fullstack/02-ecs-compute/) | `modules/compute` |
| RDS MySQL HA, backups | [Part 5: RDS Database](/en/aliyun-fullstack/05-rds-database/) | `modules/database` |
| OSS bucket, lifecycle, CORS, CDN | [Part 4: OSS Storage](/en/aliyun-fullstack/04-oss-storage/) | `modules/storage` |
| RAM users, roles, policies, KMS | [Part 6: RAM Security](/en/aliyun-fullstack/06-ram-security/) | `modules/security` |
| SLS project, logstore, alerts | [Part 7: SLS Observability](/en/aliyun-fullstack/07-observability) | `modules/monitoring` |
| Function Compute, OSS trigger | [Part 8: Serverless](/en/aliyun-fullstack/08-serverless/) | `modules/serverless` |

For LLM and ML deployment covered in [Part 10](/en/aliyun-fullstack/10-bailian-llm/) and [Part 11](/en/aliyun-fullstack/11-pai-ml-platform/), Terraform support is more limited — DashScope and PAI model deployment are typically done through their own SDKs. We will note where Terraform coverage ends.

## Project Structure

A well-organized Terraform project is the difference between a codebase your team maintains for years and one that becomes a "don't touch it, it works" liability within months. Here is the structure:

![Terraform module layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_module_layout.png)

```text
aliyun-fullstack-terraform/
├── main.tf                    # Root module: composes all child modules
├── variables.tf               # Input variables for the root module
├── outputs.tf                 # Output values (IPs, endpoints, etc.)
├── versions.tf                # Provider and Terraform version constraints
├── backend.tf                 # Remote state configuration (OSS)
├── terraform.tfvars           # Variable values (gitignored)
├── terraform.tfvars.example   # Example variable values (committed)
│
├── modules/
│   ├── network/
│   │   ├── main.tf            # VPC, VSwitches, Security Groups, NAT
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── compute/
│   │   ├── main.tf            # ECS, cloud-init, key pair
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── cloud-init.yaml    # Cloud-init template
│   │
│   ├── database/
│   │   ├── main.tf            # RDS MySQL HA
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── storage/
│   │   ├── main.tf            # OSS, lifecycle, CORS
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── security/
│   │   ├── main.tf            # RAM users, roles, policies, KMS
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   ├── monitoring/
│   │   ├── main.tf            # SLS project, logstore, alerts
│   │   ├── variables.tf
│   │   └── outputs.tf
│   │
│   └── serverless/
│       ├── main.tf            # Function Compute, triggers
│       ├── variables.tf
│       ├── outputs.tf
│       └── functions/         # Function source code
│           └── image-resize/
│               └── index.py
│
├── environments/
│   ├── dev.tfvars             # Dev environment overrides
│   ├── staging.tfvars         # Staging environment overrides
│   └── prod.tfvars            # Production environment overrides
│
└── .github/
    └── workflows/
        └── terraform.yml      # CI/CD pipeline
```

Key decisions in this layout:

- **One root module, many child modules.** Each child module is self-contained and reusable. You can use `modules/network` in a different project without dragging along the database module.
- **Environment separation via `.tfvars` files.** Not separate directories, not workspaces (which share state backends and cause accidents). Just different variable files: `terraform apply -var-file=environments/prod.tfvars`.
- **State per environment.** Each environment gets its own state file in a separate OSS path. We configure this in the backend.
- **Secrets stay out of version control.** `terraform.tfvars` is gitignored. `terraform.tfvars.example` shows the structure without values.

## Provider and State Setup

Before any module code, we need two things: the Alibaba Cloud provider configuration and a remote state backend.

![Terraform state management flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_state_management.png)

### versions.tf

```hcl
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.230.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
  }
}

provider "alicloud" {
  region     = var.region
  access_key = var.access_key
  secret_key = var.secret_key
}
```

Pin the provider version. The `alicloud` provider is updated frequently and breaking changes happen. The `~>` constraint allows patch updates (1.230.x) but not minor version bumps.

Never hardcode credentials in the provider block. In practice, you would use environment variables (`ALICLOUD_ACCESS_KEY`, `ALICLOUD_SECRET_KEY`) or an Alibaba Cloud credentials file. We use variables here for explicitness.

### backend.tf

```hcl
terraform {
  backend "oss" {
    bucket              = "my-terraform-state"
    prefix              = "fullstack/prod"
    region              = "cn-hangzhou"
    tablestore_endpoint = "https://terraform-lock.cn-hangzhou.ots.aliyuncs.com"
    tablestore_table    = "terraform_lock"
  }
}
```

Remote state in OSS with locking via TableStore. This is critical for team environments — without locking, two people running `terraform apply` simultaneously will corrupt the state file. TableStore provides a distributed lock that prevents concurrent state modifications.

![Remote state backend topology with OSS storage and Tablestore lock](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_state_backend.png)

To create the state bucket and lock table (a one-time bootstrap step):

```bash
# Create the state bucket
aliyun oss mb oss://my-terraform-state --region cn-hangzhou

# Enable versioning (so you can recover from state corruption)
aliyun oss bucket-versioning --method put oss://my-terraform-state --versioning-configuration '{"Status":"Enabled"}'

# Create the TableStore instance and lock table
aliyun ots create-instance --instance-name terraform-lock --region cn-hangzhou
aliyun ots create-table \
  --instance-name terraform-lock \
  --table-meta '{"TableName":"terraform_lock","PrimaryKey":[{"Name":"LockID","Type":"STRING"}]}'
```

### variables.tf (root module)

```hcl
# --- Provider ---
variable "region" {
  description = "Alibaba Cloud region"
  type        = string
  default     = "cn-hangzhou"
}

variable "access_key" {
  description = "Alibaba Cloud Access Key ID"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "Alibaba Cloud Access Key Secret"
  type        = string
  sensitive   = true
}

# --- Common ---
variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "fullstack"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# --- Network ---
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# --- Compute ---
variable "ecs_instance_type" {
  description = "ECS instance type"
  type        = string
  default     = "ecs.g7.large"
}

variable "ecs_password" {
  description = "ECS instance root password"
  type        = string
  sensitive   = true
}

# --- Database ---
variable "rds_instance_type" {
  description = "RDS instance type"
  type        = string
  default     = "rds.mysql.s2.large"
}

variable "db_password" {
  description = "RDS database password"
  type        = string
  sensitive   = true
}
```

Notice the `sensitive = true` on credentials. Terraform will mask these values in plan output and state logs.

## Module: Network

The network module provisions the VPC architecture from [Part 3](/en/aliyun-fullstack/03-vpc-networking/) — a 3-tier, 2-AZ layout with security groups, a NAT Gateway for outbound traffic, and an EIP for public access.

![Network module resources](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_network_module.png)

### modules/network/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}
```

### modules/network/main.tf

```hcl
# --- Data sources ---
data "alicloud_zones" "available" {
  available_resource_creation = "VSwitch"
}

locals {
  az_a = data.alicloud_zones.available.zones[0].id
  az_b = data.alicloud_zones.available.zones[1].id
  name_prefix = "${var.project_name}-${var.environment}"
}

# --- VPC ---
resource "alicloud_vpc" "main" {
  vpc_name   = "${local.name_prefix}-vpc"
  cidr_block = var.vpc_cidr

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- VSwitches (3 tiers x 2 AZs = 6 VSwitches) ---
resource "alicloud_vswitch" "public_a" {
  vswitch_name = "${local.name_prefix}-public-a"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.1.0/24"
  zone_id      = local.az_a
}

resource "alicloud_vswitch" "public_b" {
  vswitch_name = "${local.name_prefix}-public-b"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.2.0/24"
  zone_id      = local.az_b
}

resource "alicloud_vswitch" "app_a" {
  vswitch_name = "${local.name_prefix}-app-a"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.10.0/24"
  zone_id      = local.az_a
}

resource "alicloud_vswitch" "app_b" {
  vswitch_name = "${local.name_prefix}-app-b"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.11.0/24"
  zone_id      = local.az_b
}

resource "alicloud_vswitch" "data_a" {
  vswitch_name = "${local.name_prefix}-data-a"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.20.0/24"
  zone_id      = local.az_a
}

resource "alicloud_vswitch" "data_b" {
  vswitch_name = "${local.name_prefix}-data-b"
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.21.0/24"
  zone_id      = local.az_b
}

# --- Security Groups ---
resource "alicloud_security_group" "web" {
  name        = "${local.name_prefix}-sg-web"
  vpc_id      = alicloud_vpc.main.id
  description = "Security group for web tier"
}

resource "alicloud_security_group_rule" "web_http" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "80/80"
  cidr_ip           = "0.0.0.0/0"
  security_group_id = alicloud_security_group.web.id
}

resource "alicloud_security_group_rule" "web_https" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "443/443"
  cidr_ip           = "0.0.0.0/0"
  security_group_id = alicloud_security_group.web.id
}

resource "alicloud_security_group" "app" {
  name        = "${local.name_prefix}-sg-app"
  vpc_id      = alicloud_vpc.main.id
  description = "Security group for app tier"
}

resource "alicloud_security_group_rule" "app_ssh" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "22/22"
  cidr_ip           = var.vpc_cidr
  security_group_id = alicloud_security_group.app.id
}

resource "alicloud_security_group_rule" "app_from_web" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "8080/8080"
  cidr_ip           = "10.0.1.0/24"
  security_group_id = alicloud_security_group.app.id
}

resource "alicloud_security_group" "data" {
  name        = "${local.name_prefix}-sg-data"
  vpc_id      = alicloud_vpc.main.id
  description = "Security group for data tier"
}

resource "alicloud_security_group_rule" "data_mysql" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "3306/3306"
  cidr_ip           = "10.0.10.0/23"
  security_group_id = alicloud_security_group.data.id
}

# --- NAT Gateway ---
resource "alicloud_nat_gateway" "main" {
  vpc_id           = alicloud_vpc.main.id
  nat_gateway_name = "${local.name_prefix}-nat"
  payment_type     = "PayAsYouGo"
  vswitch_id       = alicloud_vswitch.public_a.id
  nat_type         = "Enhanced"
}

resource "alicloud_eip_address" "nat" {
  address_name         = "${local.name_prefix}-eip-nat"
  bandwidth            = "5"
  internet_charge_type = "PayByTraffic"
  payment_type         = "PayAsYouGo"
}

resource "alicloud_eip_association" "nat" {
  allocation_id = alicloud_eip_address.nat.id
  instance_id   = alicloud_nat_gateway.main.id
  instance_type = "Nat"
}

resource "alicloud_snat_entry" "app_a" {
  snat_table_id     = alicloud_nat_gateway.main.snat_table_ids
  source_vswitch_id = alicloud_vswitch.app_a.id
  snat_ip           = alicloud_eip_address.nat.ip_address
}

resource "alicloud_snat_entry" "app_b" {
  snat_table_id     = alicloud_nat_gateway.main.snat_table_ids
  source_vswitch_id = alicloud_vswitch.app_b.id
  snat_ip           = alicloud_eip_address.nat.ip_address
}
```

### modules/network/outputs.tf

```hcl
output "vpc_id" {
  value = alicloud_vpc.main.id
}

output "public_vswitch_ids" {
  value = [alicloud_vswitch.public_a.id, alicloud_vswitch.public_b.id]
}

output "app_vswitch_ids" {
  value = [alicloud_vswitch.app_a.id, alicloud_vswitch.app_b.id]
}

output "data_vswitch_ids" {
  value = [alicloud_vswitch.data_a.id, alicloud_vswitch.data_b.id]
}

output "web_security_group_id" {
  value = alicloud_security_group.web.id
}

output "app_security_group_id" {
  value = alicloud_security_group.app.id
}

output "data_security_group_id" {
  value = alicloud_security_group.data.id
}

output "nat_eip" {
  value = alicloud_eip_address.nat.ip_address
}
```

Notice the security group rules. SSH is only allowed from within the VPC (`10.0.0.0/16`), not from the public internet. The app port (8080) is only accessible from the public subnet. MySQL is only accessible from the app subnets. This is the three-tier isolation we designed in [Part 3](/en/aliyun-fullstack/03-vpc-networking/), now codified and repeatable.

## Module: Compute

The compute module creates the ECS instance we configured in [Part 2](/en/aliyun-fullstack/02-ecs-compute/), including a cloud-init script for automated bootstrapping.

### modules/compute/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "vswitch_id" {
  type        = string
  description = "VSwitch for the ECS instance (app tier)"
}

variable "security_group_id" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "ecs.g7.large"
}

variable "password" {
  type      = string
  sensitive = true
}
```

### modules/compute/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

data "alicloud_images" "ubuntu" {
  name_regex  = "^ubuntu_22"
  most_recent = true
  owners      = "system"
}

# --- Key Pair ---
resource "tls_private_key" "ecs" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "alicloud_ecs_key_pair" "main" {
  key_pair_name = "${local.name_prefix}-keypair"
  public_key    = tls_private_key.ecs.public_key_openssh
}

# --- ECS Instance ---
resource "alicloud_instance" "app" {
  instance_name = "${local.name_prefix}-app"
  host_name     = "${local.name_prefix}-app"

  image_id      = data.alicloud_images.ubuntu.images[0].id
  instance_type = var.instance_type
  vswitch_id    = var.vswitch_id
  security_groups = [var.security_group_id]
  key_name      = alicloud_ecs_key_pair.main.key_pair_name

  system_disk_category = "cloud_essd"
  system_disk_size     = 40

  internet_max_bandwidth_out = 0  # No public IP; use EIP

  user_data = base64encode(templatefile("${path.module}/cloud-init.yaml", {
    hostname    = "${local.name_prefix}-app"
    environment = var.environment
  }))

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Role        = "app-server"
  }
}

# --- EIP for the app server ---
resource "alicloud_eip_address" "app" {
  address_name         = "${local.name_prefix}-eip-app"
  bandwidth            = "10"
  internet_charge_type = "PayByTraffic"
  payment_type         = "PayAsYouGo"
}

resource "alicloud_eip_association" "app" {
  allocation_id = alicloud_eip_address.app.id
  instance_id   = alicloud_instance.app.id
  instance_type = "EcsInstance"
}
```

### modules/compute/cloud-init.yaml

```yaml
#cloud-config
hostname: ${hostname}
package_update: true
package_upgrade: true

packages:
  - docker.io
  - docker-compose
  - nginx
  - certbot
  - python3-certbot-nginx
  - fail2ban
  - ufw

runcmd:
  # Enable and start Docker
  - systemctl enable docker
  - systemctl start docker

  # Configure firewall
  - ufw default deny incoming
  - ufw default allow outgoing
  - ufw allow 22/tcp
  - ufw allow 80/tcp
  - ufw allow 443/tcp
  - ufw --force enable

  # Enable fail2ban
  - systemctl enable fail2ban
  - systemctl start fail2ban

  # Log the completion
  - echo "cloud-init complete for ${environment}" >> /var/log/cloud-init-custom.log
```

### modules/compute/outputs.tf

```hcl
output "instance_id" {
  value = alicloud_instance.app.id
}

output "private_ip" {
  value = alicloud_instance.app.private_ip
}

output "public_ip" {
  value = alicloud_eip_address.app.ip_address
}

output "private_key_pem" {
  value     = tls_private_key.ecs.private_key_pem
  sensitive = true
}
```

## Module: Database

The database module creates the RDS MySQL HA instance from [Part 5](/en/aliyun-fullstack/05-rds-database/), including a read replica, automated backups, and a dedicated account.

### modules/database/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "vswitch_id_primary" {
  type        = string
  description = "VSwitch for the primary RDS instance (data tier, AZ-A)"
}

variable "vswitch_id_secondary" {
  type        = string
  description = "VSwitch for the secondary RDS instance (data tier, AZ-B)"
}

variable "security_group_id" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "rds.mysql.s2.large"
}

variable "db_name" {
  type    = string
  default = "app_production"
}

variable "db_password" {
  type      = string
  sensitive = true
}
```

### modules/database/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --- RDS MySQL Instance (Primary) ---
resource "alicloud_db_instance" "primary" {
  engine               = "MySQL"
  engine_version       = "8.0"
  instance_type        = var.instance_type
  instance_storage     = 50
  instance_name        = "${local.name_prefix}-rds-primary"
  vswitch_id           = var.vswitch_id_primary
  security_group_ids   = [var.security_group_id]
  instance_charge_type = "Postpaid"

  db_instance_storage_type = "cloud_essd"

  # High Availability
  zone_id               = null  # auto-select based on vswitch
  zone_id_slave_a       = null

  # Security
  security_ips = ["10.0.10.0/24", "10.0.11.0/24"]
  ssl_action   = "Open"

  # Maintenance window
  maintain_time = "02:00Z-06:00Z"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Role        = "database-primary"
  }
}

# --- Database ---
resource "alicloud_db_database" "main" {
  instance_id = alicloud_db_instance.primary.id
  name        = var.db_name
  character_set = "utf8mb4"
}

# --- Database Account ---
resource "alicloud_rds_account" "app" {
  db_instance_id   = alicloud_db_instance.primary.id
  account_name     = "app_user"
  account_password = var.db_password
  account_type     = "Normal"
}

resource "alicloud_db_account_privilege" "app" {
  instance_id  = alicloud_db_instance.primary.id
  account_name = alicloud_rds_account.app.account_name
  privilege    = "ReadWrite"
  db_names     = [alicloud_db_database.main.name]
}

# --- Backup Policy ---
resource "alicloud_db_backup_policy" "main" {
  instance_id               = alicloud_db_instance.primary.id
  preferred_backup_time     = "02:00Z-03:00Z"
  preferred_backup_period   = ["Monday", "Wednesday", "Friday"]
  backup_retention_period   = 7
  log_backup_retention_period = 7
  enable_backup_log         = true
}

# --- Read-Only Instance ---
resource "alicloud_db_readonly_instance" "replica" {
  master_db_instance_id = alicloud_db_instance.primary.id
  engine_version        = "8.0"
  instance_type         = var.instance_type
  instance_storage      = 50
  instance_name         = "${local.name_prefix}-rds-readonly"
  vswitch_id            = var.vswitch_id_secondary
  zone_id               = null

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Role        = "database-readonly"
  }
}
```

### modules/database/outputs.tf

```hcl
output "primary_connection_string" {
  value = alicloud_db_instance.primary.connection_string
}

output "primary_port" {
  value = alicloud_db_instance.primary.port
}

output "readonly_connection_string" {
  value = alicloud_db_readonly_instance.replica.connection_string
}

output "database_name" {
  value = alicloud_db_database.main.name
}
```

## Module: Storage

The OSS bucket from [Part 4](/en/aliyun-fullstack/04-oss-storage/) with lifecycle policies, CORS rules, and CDN acceleration.

### modules/storage/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# Random suffix to ensure bucket name uniqueness
resource "random_string" "bucket_suffix" {
  length  = 6
  special = false
  upper   = false
}

# --- OSS Bucket ---
resource "alicloud_oss_bucket" "media" {
  bucket = "${local.name_prefix}-media-${random_string.bucket_suffix.result}"
  acl    = "private"

  # Versioning
  versioning {
    status = "Enabled"
  }

  # Server-side encryption
  server_side_encryption_rule {
    sse_algorithm = "AES256"
  }

  # Lifecycle rules
  lifecycle_rule {
    id      = "archive-old-files"
    enabled = true

    transition {
      days          = 90
      storage_class = "IA"
    }

    transition {
      days          = 180
      storage_class = "Archive"
    }
  }

  lifecycle_rule {
    id      = "cleanup-temp"
    prefix  = "tmp/"
    enabled = true

    expiration {
      days = 7
    }
  }

  # CORS rules
  cors_rule {
    allowed_origins = ["https://*.${var.domain}"]
    allowed_methods = ["GET", "PUT", "POST"]
    allowed_headers = ["*"]
    max_age_seconds = 3600
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Bucket policy: only allow access from VPC ---
resource "alicloud_oss_bucket_policy" "vpc_only" {
  bucket = alicloud_oss_bucket.media.id
  policy = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect    = "Allow"
        Action    = ["oss:GetObject"]
        Principal = ["*"]
        Resource  = ["acs:oss:*:*:${alicloud_oss_bucket.media.id}/*"]
        Condition = {
          StringEquals = {
            "acs:SourceVpc" = [var.vpc_id]
          }
        }
      }
    ]
  })
}
```

### modules/storage/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "domain" {
  type    = string
  default = "example.com"
}
```

### modules/storage/outputs.tf

```hcl
output "bucket_name" {
  value = alicloud_oss_bucket.media.id
}

output "bucket_domain" {
  value = alicloud_oss_bucket.media.extranet_endpoint
}

output "bucket_internal_domain" {
  value = alicloud_oss_bucket.media.intranet_endpoint
}
```

## Module: Security

The RAM module from [Part 6](/en/aliyun-fullstack/06-ram-security/) — users, groups, roles, and policies following the principle of least privilege. Plus a KMS key for encryption.

### modules/security/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --- RAM Group for Developers ---
resource "alicloud_ram_group" "developers" {
  name  = "${local.name_prefix}-developers"
  force = true
}

# --- RAM Policy: ECS Read-Only ---
resource "alicloud_ram_policy" "ecs_readonly" {
  policy_name = "${local.name_prefix}-ecs-readonly"
  policy_document = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:Describe*", "ecs:List*"]
        Resource = ["*"]
      }
    ]
  })
}

resource "alicloud_ram_group_policy_attachment" "dev_ecs" {
  policy_name = alicloud_ram_policy.ecs_readonly.policy_name
  policy_type = "Custom"
  group_name  = alicloud_ram_group.developers.name
}

# --- RAM Role: ECS Instance Role (for EC2-like instance profiles) ---
resource "alicloud_ram_role" "ecs_role" {
  name     = "${local.name_prefix}-ecs-role"
  document = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = ["ecs.aliyuncs.com"]
        }
      }
    ]
  })
}

# --- RAM Policy: Allow ECS to access OSS ---
resource "alicloud_ram_policy" "ecs_oss_access" {
  policy_name = "${local.name_prefix}-ecs-oss-access"
  policy_document = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["oss:GetObject", "oss:PutObject", "oss:ListBucket"]
        Resource = [
          "acs:oss:*:*:${var.oss_bucket_name}",
          "acs:oss:*:*:${var.oss_bucket_name}/*"
        ]
      }
    ]
  })
}

resource "alicloud_ram_role_policy_attachment" "ecs_oss" {
  policy_name = alicloud_ram_policy.ecs_oss_access.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.ecs_role.name
}

# --- RAM Policy: Allow ECS to write SLS logs ---
resource "alicloud_ram_policy" "ecs_sls_write" {
  policy_name = "${local.name_prefix}-ecs-sls-write"
  policy_document = jsonencode({
    Version = "1"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["log:PostLogStoreLogs"]
        Resource = ["acs:log:*:*:project/${var.sls_project_name}/*"]
      }
    ]
  })
}

resource "alicloud_ram_role_policy_attachment" "ecs_sls" {
  policy_name = alicloud_ram_policy.ecs_sls_write.policy_name
  policy_type = "Custom"
  role_name   = alicloud_ram_role.ecs_role.name
}

# --- KMS Key for encryption ---
resource "alicloud_kms_key" "main" {
  description         = "${local.name_prefix} master encryption key"
  key_usage           = "ENCRYPT/DECRYPT"
  pending_window_in_days = 7
  automatic_rotation  = "Enabled"
  rotation_interval   = "365d"
}

resource "alicloud_kms_alias" "main" {
  alias_name = "alias/${local.name_prefix}-master-key"
  key_id     = alicloud_kms_key.main.id
}
```

### modules/security/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "oss_bucket_name" {
  type = string
}

variable "sls_project_name" {
  type    = string
  default = ""
}
```

### modules/security/outputs.tf

```hcl
output "ecs_role_name" {
  value = alicloud_ram_role.ecs_role.name
}

output "ecs_role_arn" {
  value = alicloud_ram_role.ecs_role.arn
}

output "kms_key_id" {
  value = alicloud_kms_key.main.id
}

output "developer_group_name" {
  value = alicloud_ram_group.developers.name
}
```

## Module: Monitoring

SLS (Simple Log Service) project and logstore for centralized logging, plus CloudMonitor alert rules. This codifies the observability layer from [Part 7](/en/aliyun-fullstack/07-observability/).

### modules/monitoring/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --- SLS Project ---
resource "alicloud_log_project" "main" {
  project_name = "${local.name_prefix}-logs"
  description  = "Log project for ${var.project_name} ${var.environment}"
}

# --- SLS Logstores ---
resource "alicloud_log_store" "app" {
  project_name          = alicloud_log_project.main.project_name
  logstore_name         = "app-logs"
  retention_period      = 30
  shard_count           = 2
  auto_split            = true
  max_split_shard_count = 8
  append_meta           = true
  enable_web_tracking   = false
}

resource "alicloud_log_store" "access" {
  project_name          = alicloud_log_project.main.project_name
  logstore_name         = "access-logs"
  retention_period      = 90
  shard_count           = 2
  auto_split            = true
  max_split_shard_count = 16
}

resource "alicloud_log_store" "error" {
  project_name          = alicloud_log_project.main.project_name
  logstore_name         = "error-logs"
  retention_period      = 180
  shard_count           = 1
  auto_split            = true
  max_split_shard_count = 4
}

# --- Log Index (required for querying) ---
resource "alicloud_log_store_index" "app" {
  project  = alicloud_log_project.main.project_name
  logstore = alicloud_log_store.app.logstore_name

  full_text {
    case_sensitive = false
    token          = ", '\";=()[]{}?@&<>/:\n\t\r"
  }

  field_search {
    name             = "level"
    type             = "text"
    case_sensitive   = false
    include_chinese  = false
    token            = ", '\";=()[]{}?@&<>/:\n\t\r"
    enable_analytics = true
  }

  field_search {
    name             = "request_time"
    type             = "double"
    enable_analytics = true
  }
}

# --- CloudMonitor Alert: High CPU ---
resource "alicloud_cms_alarm" "high_cpu" {
  name    = "${local.name_prefix}-high-cpu"
  project = "acs_ecs_dashboard"
  metric  = "CPUUtilization"
  period  = 300

  dimensions = {
    instanceId = var.ecs_instance_id
  }

  escalations_critical {
    statistics          = "Average"
    comparison_operator = ">="
    threshold           = "90"
    times               = 3
  }

  contact_groups = [var.alert_contact_group]
  enabled        = true
}

# --- CloudMonitor Alert: High Memory ---
resource "alicloud_cms_alarm" "high_memory" {
  name    = "${local.name_prefix}-high-memory"
  project = "acs_ecs_dashboard"
  metric  = "memory_usedutilization"
  period  = 300

  dimensions = {
    instanceId = var.ecs_instance_id
  }

  escalations_critical {
    statistics          = "Average"
    comparison_operator = ">="
    threshold           = "90"
    times               = 3
  }

  contact_groups = [var.alert_contact_group]
  enabled        = true
}

# --- CloudMonitor Alert: RDS CPU ---
resource "alicloud_cms_alarm" "rds_cpu" {
  name    = "${local.name_prefix}-rds-high-cpu"
  project = "acs_rds_dashboard"
  metric  = "CpuUsage"
  period  = 300

  dimensions = {
    instanceId = var.rds_instance_id
  }

  escalations_critical {
    statistics          = "Average"
    comparison_operator = ">="
    threshold           = "85"
    times               = 3
  }

  contact_groups = [var.alert_contact_group]
  enabled        = true
}
```

### modules/monitoring/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "ecs_instance_id" {
  type = string
}

variable "rds_instance_id" {
  type    = string
  default = ""
}

variable "alert_contact_group" {
  type    = string
  default = "default-contact-group"
}
```

### modules/monitoring/outputs.tf

```hcl
output "sls_project_name" {
  value = alicloud_log_project.main.project_name
}

output "sls_endpoint" {
  value = alicloud_log_project.main.project_name
}

output "app_logstore" {
  value = alicloud_log_store.app.logstore_name
}

output "access_logstore" {
  value = alicloud_log_store.access.logstore_name
}
```

## Module: Serverless

Function Compute service with an OSS trigger — exactly what we built in [Part 8](/en/aliyun-fullstack/08-serverless/). When a file is uploaded to the media bucket, a function automatically generates a thumbnail.

### modules/serverless/main.tf

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --- Function Compute Service ---
resource "alicloud_fc_service" "main" {
  name        = "${local.name_prefix}-fc-service"
  description = "Function Compute service for ${var.project_name}"

  role = var.fc_role_arn

  log_config {
    project  = var.sls_project_name
    logstore = var.sls_logstore_name
  }

  vpc_config {
    vswitch_ids         = var.vswitch_ids
    security_group_id   = var.security_group_id
  }

  internet_access = false
}

# --- Function: Image Resize ---
data "archive_file" "image_resize" {
  type        = "zip"
  source_dir  = "${path.module}/functions/image-resize"
  output_path = "${path.module}/functions/image-resize.zip"
}

resource "alicloud_fc_function" "image_resize" {
  service     = alicloud_fc_service.main.name
  name        = "image-resize"
  description = "Resize images uploaded to OSS"
  runtime     = "python3.10"
  handler     = "index.handler"
  memory_size = 512
  timeout     = 60

  filename = data.archive_file.image_resize.output_path

  environment_variables = {
    DEST_BUCKET = var.oss_bucket_name
    DEST_PREFIX = "thumbnails/"
    MAX_WIDTH   = "800"
    MAX_HEIGHT  = "600"
  }
}

# --- OSS Trigger ---
resource "alicloud_fc_trigger" "oss_upload" {
  service  = alicloud_fc_service.main.name
  function = alicloud_fc_function.image_resize.name
  name     = "oss-upload-trigger"
  type     = "oss"
  role     = var.fc_trigger_role_arn

  config = jsonencode({
    events    = ["oss:ObjectCreated:PutObject", "oss:ObjectCreated:PostObject"]
    filter = {
      key = {
        prefix = "uploads/"
        suffix = ""
      }
    }
  })

  source_arn = "acs:oss:*:*:${var.oss_bucket_name}"
}
```

### modules/serverless/functions/image-resize/index.py

```python
import json
import oss2
import os
from io import BytesIO

def handler(event, context):
    """
    Triggered by OSS upload. Resizes images and stores thumbnails.
    """
    evt = json.loads(event)
    bucket_name = evt["events"][0]["oss"]["bucket"]["name"]
    object_key = evt["events"][0]["oss"]["object"]["key"]

    creds = context.credentials
    auth = oss2.StsAuth(
        creds.access_key_id,
        creds.access_key_secret,
        creds.security_token
    )

    endpoint = f"https://oss-{context.region}-internal.aliyuncs.com"
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    # Download the original image
    result = bucket.get_object(object_key)
    img_data = result.read()

    # Generate thumbnail key
    dest_prefix = os.environ.get("DEST_PREFIX", "thumbnails/")
    filename = object_key.split("/")[-1]
    dest_key = f"{dest_prefix}{filename}"

    # For production, use Pillow for actual resizing.
    # This example passes through the original as a placeholder.
    bucket.put_object(dest_key, img_data)

    return {"statusCode": 200, "body": f"Processed {object_key} -> {dest_key}"}
```

### modules/serverless/variables.tf

```hcl
variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "fc_role_arn" {
  type = string
}

variable "fc_trigger_role_arn" {
  type = string
}

variable "sls_project_name" {
  type = string
}

variable "sls_logstore_name" {
  type = string
}

variable "vswitch_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "oss_bucket_name" {
  type = string
}
```

### modules/serverless/outputs.tf

```hcl
output "service_name" {
  value = alicloud_fc_service.main.name
}

output "function_name" {
  value = alicloud_fc_function.image_resize.name
}

output "trigger_name" {
  value = alicloud_fc_trigger.oss_upload.name
}
```

## Putting It All Together

The root `main.tf` composes all seven modules, wiring outputs from one module as inputs to another. This is the file where you see how everything connects.

### main.tf (root module)

![Inter-module data flow showing how outputs feed into inputs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_module_deps.png)

```hcl
# --- Network ---
module "network" {
  source = "./modules/network"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  vpc_cidr     = var.vpc_cidr
}

# --- Compute ---
module "compute" {
  source = "./modules/compute"

  project_name      = var.project_name
  environment       = var.environment
  vpc_id            = module.network.vpc_id
  vswitch_id        = module.network.app_vswitch_ids[0]
  security_group_id = module.network.app_security_group_id
  instance_type     = var.ecs_instance_type
  password          = var.ecs_password
}

# --- Database ---
module "database" {
  source = "./modules/database"

  project_name          = var.project_name
  environment           = var.environment
  vpc_id                = module.network.vpc_id
  vswitch_id_primary    = module.network.data_vswitch_ids[0]
  vswitch_id_secondary  = module.network.data_vswitch_ids[1]
  security_group_id     = module.network.data_security_group_id
  instance_type         = var.rds_instance_type
  db_password           = var.db_password
}

# --- Storage ---
module "storage" {
  source = "./modules/storage"

  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.network.vpc_id
}

# --- Security ---
module "security" {
  source = "./modules/security"

  project_name     = var.project_name
  environment      = var.environment
  oss_bucket_name  = module.storage.bucket_name
  sls_project_name = module.monitoring.sls_project_name
}

# --- Monitoring ---
module "monitoring" {
  source = "./modules/monitoring"

  project_name    = var.project_name
  environment     = var.environment
  ecs_instance_id = module.compute.instance_id
  rds_instance_id = module.database.primary_connection_string
}

# --- Serverless ---
module "serverless" {
  source = "./modules/serverless"

  project_name        = var.project_name
  environment         = var.environment
  fc_role_arn         = module.security.ecs_role_arn
  fc_trigger_role_arn = module.security.ecs_role_arn
  sls_project_name    = module.monitoring.sls_project_name
  sls_logstore_name   = module.monitoring.app_logstore
  vswitch_ids         = module.network.app_vswitch_ids
  security_group_id   = module.network.app_security_group_id
  oss_bucket_name     = module.storage.bucket_name
}
```

### outputs.tf (root module)

```hcl
# --- Endpoints ---
output "app_public_ip" {
  description = "Public IP of the application server"
  value       = module.compute.public_ip
}

output "app_private_ip" {
  description = "Private IP of the application server"
  value       = module.compute.private_ip
}

output "rds_endpoint" {
  description = "RDS primary connection string"
  value       = module.database.primary_connection_string
}

output "rds_readonly_endpoint" {
  description = "RDS read-only connection string"
  value       = module.database.readonly_connection_string
}

output "oss_bucket" {
  description = "OSS bucket name"
  value       = module.storage.bucket_name
}

output "oss_internal_endpoint" {
  description = "OSS internal endpoint (use from within VPC)"
  value       = module.storage.bucket_internal_domain
}

output "sls_project" {
  description = "SLS project name"
  value       = module.monitoring.sls_project_name
}

output "fc_service" {
  description = "Function Compute service name"
  value       = module.serverless.service_name
}

output "nat_eip" {
  description = "NAT Gateway public IP"
  value       = module.network.nat_eip
}

# --- SSH ---
output "ssh_private_key" {
  description = "SSH private key (save to file, chmod 600)"
  value       = module.compute.private_key_pem
  sensitive   = true
}

output "ssh_command" {
  description = "SSH command to connect to the app server"
  value       = "ssh -i key.pem root@${module.compute.public_ip}"
}
```

### terraform.tfvars.example

![Multi-environment strategy: dev, staging, prod from one codebase](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_multi_env.png)

```hcl
# Copy this to terraform.tfvars and fill in your values
# DO NOT commit terraform.tfvars to version control

region       = "cn-hangzhou"
access_key   = "your-access-key-id"
secret_key   = "your-access-key-secret"

project_name = "fullstack"
environment  = "prod"

vpc_cidr          = "10.0.0.0/16"
ecs_instance_type = "ecs.g7.large"
ecs_password      = "YourSecurePassword123!"
rds_instance_type = "rds.mysql.s2.large"
db_password       = "YourDBPassword456!"
```

### Running It

![Terraform workflow lifecycle from init to destroy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_workflow_lifecycle.png)

```bash
# Initialize Terraform (download providers, configure backend)
terraform init

# Preview what will be created
terraform plan -var-file=environments/prod.tfvars

# Apply (create everything)
terraform apply -var-file=environments/prod.tfvars

# See what was created
terraform output

# Get the SSH key
terraform output -raw ssh_private_key > key.pem
chmod 600 key.pem

# Connect to your server
$(terraform output -raw ssh_command)
```

The `terraform plan` output will show you every resource that will be created before you commit. This is the single most important feature of Terraform — you always know what is about to happen. In my experience, roughly one in three plans catches something I did not intend, saving me from creating resources in the wrong AZ, opening the wrong port, or provisioning the wrong instance type.

A typical plan output for our full stack looks like this:

```text
Plan: 42 to add, 0 to change, 0 to destroy.

Changes to Outputs:
  + app_public_ip       = (known after apply)
  + oss_bucket          = (known after apply)
  + rds_endpoint        = (known after apply)
  + ssh_command          = (known after apply)
```

Forty-two resources from a single command. That is the power of infrastructure-as-code.

## CI/CD with GitHub Actions

Manual `terraform apply` from your laptop is fine for personal projects. For teams, you need automation: run `terraform plan` on every pull request so reviewers can see the infrastructure diff, and run `terraform apply` automatically when the PR merges to main.

![CI/CD pipeline with GitHub Actions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_cicd_flow.png)

### .github/workflows/terraform.yml

```yaml
name: "Terraform"

on:
  push:
    branches: [main]
    paths:
      - "**.tf"
      - "**.tfvars"
      - ".github/workflows/terraform.yml"
  pull_request:
    branches: [main]
    paths:
      - "**.tf"
      - "**.tfvars"

permissions:
  contents: read
  pull-requests: write

env:
  TF_VERSION: "1.7.0"
  ALICLOUD_ACCESS_KEY: ${{ secrets.ALICLOUD_ACCESS_KEY }}
  ALICLOUD_SECRET_KEY: ${{ secrets.ALICLOUD_SECRET_KEY }}
  ALICLOUD_REGION: "cn-hangzhou"

jobs:
  terraform-plan:
    name: "Plan"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init -input=false

      - name: Terraform Format Check
        run: terraform fmt -check -recursive

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        id: plan
        run: |
          terraform plan -no-color -input=false \
            -var-file=environments/prod.tfvars \
            -out=tfplan \
            2>&1 | tee plan_output.txt
        continue-on-error: true

      - name: Comment PR with Plan
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('plan_output.txt', 'utf8');
            const truncated = plan.length > 60000
              ? plan.substring(0, 60000) + '\n... (truncated)'
              : plan;

            const output = `### Terraform Plan
            \`\`\`
            ${truncated}
            \`\`\`

            *Pushed by: @${{ github.actor }}*`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: output
            });

      - name: Plan Status
        if: steps.plan.outcome == 'failure'
        run: exit 1

  terraform-apply:
    name: "Apply"
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init -input=false

      - name: Terraform Apply
        run: |
          terraform apply -auto-approve -input=false \
            -var-file=environments/prod.tfvars

      - name: Terraform Output
        run: terraform output -no-color
```

### Secrets management

![Secret management with RAM OIDC federation and KMS encryption](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_secrets_ram_kms.png)

Store these secrets in your GitHub repository settings (Settings > Secrets and variables > Actions):

| Secret Name | Value | Source |
|---|---|---|
| `ALICLOUD_ACCESS_KEY` | Your RAM user's Access Key ID | [Part 6: RAM Security](/en/aliyun-fullstack/06-ram-security/) |
| `ALICLOUD_SECRET_KEY` | Your RAM user's Access Key Secret | [Part 6: RAM Security](/en/aliyun-fullstack/06-ram-security/) |

Never use your root account credentials in CI/CD. Create a dedicated RAM user with only the permissions needed for Terraform (or better, use OIDC federation to assume a RAM role without long-lived credentials — but that is an advanced topic for another day).

The workflow does five things:

1. **Format check** — Ensures all `.tf` files are properly formatted. Fail fast on style issues.
2. **Validate** — Checks syntax and internal consistency without accessing any APIs.
3. **Plan** — Shows what would change. Posts the plan as a PR comment so reviewers can see the infrastructure diff.
4. **Apply** — Only runs on merge to main. Applies the changes automatically.
5. **Output** — Shows the resulting endpoints and IPs in the workflow log.

This gives you the same code review workflow for infrastructure that you have for application code. Someone proposes a change, the plan shows what will happen, the team reviews, and the merge triggers the apply.

## Cost Estimation and Optimization

Infrastructure-as-code makes cost optimization possible at the planning stage — before you spend anything. The open-source tool `infracost` reads your Terraform files and estimates monthly costs.

![Cost optimization strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_cost_optimization.png)

### Installing and running infracost

```bash
# Install infracost
brew install infracost

# Register for a free API key
infracost auth login

# Generate cost breakdown
infracost breakdown --path=. --terraform-var-file=environments/prod.tfvars
```

A typical output for our full stack:

```text
Project: aliyun-fullstack-terraform

 Name                                     Monthly Qty  Unit         Monthly Cost

 module.compute.alicloud_instance.app
 ├─ Instance (ecs.g7.large, linux)            730  hours           $86.14
 └─ System disk (cloud_essd, 40GB)             40  GB               $4.00

 module.database.alicloud_db_instance.primary
 ├─ Instance (rds.mysql.s2.large)             730  hours          $142.35
 └─ Storage (cloud_essd, 50GB)                 50  GB               $8.50

 module.database.alicloud_db_readonly_instance.replica
 ├─ Instance (rds.mysql.s2.large)             730  hours          $142.35
 └─ Storage (cloud_essd, 50GB)                 50  GB               $8.50

 module.network.alicloud_nat_gateway.main
 └─ NAT Gateway (Enhanced)                   730  hours           $13.14

 module.network.alicloud_eip_address.nat
 └─ EIP (PayByTraffic)                         1  month            $3.36

 module.compute.alicloud_eip_address.app
 └─ EIP (PayByTraffic)                         1  month            $3.36

 module.storage.alicloud_oss_bucket.media
 └─ Standard storage                          50  GB               $1.20
    (estimate, actual varies with usage)

 module.monitoring.alicloud_log_project.main
 └─ SLS (estimate)                             1  month            $0.00
    (free tier covers most dev/small prod usage)

 module.serverless.alicloud_fc_function.image_resize
 └─ Function Compute                          1  month            $0.00
    (free tier: 1M requests + 400K GB-s/month)

 OVERALL TOTAL                                                   $412.90
```

### Optimization strategies

These are the same strategies from every cloud, but Terraform makes them easy to implement because you change a parameter in code rather than clicking through a console:

**1. Preemptible (Spot) instances for non-critical workloads.**

```hcl
resource "alicloud_instance" "worker" {
  # ... other config ...
  spot_strategy    = "SpotAsPriceGo"
  spot_price_limit = 0  # Accept market price (typically 70-90% off)
}
```

Preemptible instances on Alibaba Cloud cost up to 90% less but can be reclaimed with 5 minutes notice. Use them for batch processing, dev/test environments, and stateless workers.

**2. Subscription (reserved) instances for predictable workloads.**

For the production database that runs 24/7, switching from pay-as-you-go to a 1-year subscription saves roughly 40%:

```hcl
resource "alicloud_db_instance" "primary" {
  # ... other config ...
  instance_charge_type = "Prepaid"
  period               = 12   # 1 year
  auto_renew           = true
  auto_renew_period    = 12
}
```

**3. Right-sizing based on actual usage.**

After running for a month, check CloudMonitor metrics:

```bash
# Check average CPU utilization over the past 7 days
aliyun cms DescribeMetricLast \
  --Namespace acs_ecs_dashboard \
  --MetricName CPUUtilization \
  --Dimensions '[{"instanceId":"i-xxxx"}]' \
  --Period 86400
```

If your `ecs.g7.large` (2 vCPU, 8 GiB) averages 15% CPU and 30% memory, downgrade to `ecs.g7.small` (1 vCPU, 4 GiB) and cut costs in half. With Terraform, this is a one-line change:

```hcl
# In environments/prod.tfvars
ecs_instance_type = "ecs.g7.small"  # was: ecs.g7.large
```

**4. Cost comparison table: before and after optimization.**

| Resource | Before (Pay-as-you-go) | After (Optimized) | Savings |
|---|---|---|---|
| ECS (app server) | $86.14/mo | $43.07/mo (right-sized) | 50% |
| RDS Primary | $142.35/mo | $85.41/mo (1yr subscription) | 40% |
| RDS Replica | $142.35/mo | $85.41/mo (1yr subscription) | 40% |
| NAT Gateway | $13.14/mo | $13.14/mo | 0% |
| EIPs (x2) | $6.72/mo | $6.72/mo | 0% |
| OSS | $1.20/mo | $1.20/mo | 0% |
| SLS + FC | $0.00/mo | $0.00/mo (free tier) | 0% |
| **Total** | **$412.90/mo** | **$234.95/mo** | **43%** |

That is nearly $2,200 saved per year just from right-sizing and switching to subscriptions. And because the changes are in version-controlled `.tfvars` files, you have a clear audit trail of when and why you made each optimization decision.

## Teardown and Cleanup

When you are done with an environment — perhaps tearing down a staging stack after testing — Terraform makes it a single command:

![Safe teardown flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_destroy_flow.png)

```bash
# Preview what will be destroyed
terraform plan -destroy -var-file=environments/staging.tfvars

# Destroy everything
terraform destroy -var-file=environments/staging.tfvars
```

This is the inverse of `terraform apply`. It reads the state, determines all managed resources, and deletes them in the correct dependency order. RDS read replicas are deleted before primary instances. SNAT entries are removed before NAT gateways. VSwitches are emptied before the VPC is deleted.

Two warnings about `terraform destroy`:

![Dependency-aware teardown order in reverse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_teardown_order.png)

1. **It is irreversible.** RDS data, OSS objects, SLS logs — gone. Terraform will prompt for confirmation, but once you type `yes`, there is no undo button.
2. **Some resources resist deletion.** OSS buckets must be empty before deletion. RDS instances with `deletion_protection = true` will block the destroy. These are safety features. If you are tearing down for real, you may need to empty buckets manually or set `force_destroy = true` on the bucket resource.

```hcl
# Only set this if you truly want terraform destroy to delete non-empty buckets
resource "alicloud_oss_bucket" "media" {
  # ... other config ...
  force_destroy = var.environment != "prod"  # Never force-destroy production
}
```

## Common Pitfalls

![Drift detection: when actual cloud state diverges from code](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/12-terraform-e2e/12_drift_detection.png)

After running Terraform against Alibaba Cloud for over a year, here are the issues I hit most often:

**1. Provider version drift.** The `alicloud` provider ships updates weekly. Pin your version and upgrade deliberately. An unplanned provider upgrade once changed the default value of a security group rule attribute and opened a port I thought was closed.

**2. State file corruption.** Always use remote state with locking. If two people apply simultaneously without locking, the state file can become inconsistent. Recovery involves `terraform state pull`, manual editing, and `terraform state push` — unpleasant work you never want to do.

**3. Resource name uniqueness.** Some Alibaba Cloud resources require globally unique names (like OSS bucket names). Always include a random suffix or your account/project prefix.

**4. Quota limits.** Every Alibaba Cloud account has default quotas (e.g., 20 ECS instances per region, 5 VPCs per region). If `terraform apply` fails partway through because you hit a quota, you end up with a partially-created infrastructure. Check your quotas before applying.

```bash
# Check ECS quota
aliyun ecs DescribeAccountAttributes --RegionId cn-hangzhou
```

**5. Import existing resources.** If you already have manually-created resources and want to bring them under Terraform management, use `terraform import`:

```bash
# Import an existing VPC
terraform import module.network.alicloud_vpc.main vpc-bp1xxxxxxxxxxxxx

# Import an existing ECS instance
terraform import module.compute.alicloud_instance.app i-bp1xxxxxxxxxxxxx
```

After importing, run `terraform plan` to see if your configuration matches the actual resource. Expect to iterate a few times until the plan shows no changes.

## Summary

This article — and this series — boils down to a few principles:

**1. Infrastructure belongs in code.** Every resource we built by hand in Parts 1 through 11 is now a declarative `.tf` file. The code is the documentation, the runbook, and the disaster recovery plan all in one.

**2. Modules are the unit of reuse.** Each module (network, compute, database, storage, security, monitoring, serverless) is self-contained and independently testable. You can use the network module in your next project without touching the database module.

**3. Environments are just variables.** Dev, staging, and production use the same code with different `.tfvars` files. No more "production has that one security group rule we added manually six months ago and nobody remembers why."

**4. CI/CD closes the loop.** Infrastructure changes follow the same pull request workflow as application code: propose, review, merge, apply. No more "who changed the security group?" mysteries.

**5. Cost optimization is a code change.** Right-sizing, reserved instances, spot instances — these are parameter changes in `.tfvars` files, reviewable and auditable like any other code change.

We have covered a lot of ground in twelve articles. Starting from the [ecosystem map in Part 1](/en/aliyun-fullstack/01-ecosystem-map/), through [compute](/en/aliyun-fullstack/02-ecs-compute/), [networking](/en/aliyun-fullstack/03-vpc-networking/), [storage](/en/aliyun-fullstack/04-oss-storage/), [databases](/en/aliyun-fullstack/05-rds-database/), [security](/en/aliyun-fullstack/06-ram-security/), observability, [serverless](/en/aliyun-fullstack/08-serverless/), containers, [LLMs](/en/aliyun-fullstack/10-bailian-llm/), [ML platforms](/en/aliyun-fullstack/11-pai-ml-platform/), and now infrastructure-as-code — you have the complete toolkit for building production systems on Alibaba Cloud.

The code from this article is a starting point, not a finished product. Fork it, adapt it to your architecture, add the resources you need, remove the ones you do not. The point is never the specific configuration — it is the practice of treating infrastructure as seriously as you treat your application code.

One `terraform apply`. Everything you need. That is the way it should be.

---

This is Part 12 (the final article) of the **Alibaba Cloud Full Stack** series. For an even deeper dive into Terraform itself — modules, workspaces, testing, CI/CD patterns, and multi-cloud architectures — see the [Terraform for Agents series](/en/terraform-agents/01-why-terraform-for-agents/).
