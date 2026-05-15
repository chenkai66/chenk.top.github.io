---
title: "Alibaba Cloud Full Stack (2): ECS — Compute That Actually Makes Sense"
date: 2026-04-29 09:00:00
tags:
  - Alibaba Cloud
  - ECS
  - Compute
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 2
series_total: 12
description: "Everything you need to know about ECS: instance families (g8, c8, r8, GPU), pricing models, cloud-init automation, security groups, and key pairs. We deploy a production-ready app server from scratch."
disableNunjucks: true
translationKey: "aliyun-fullstack-2"
---

The first ECS instance I ever launched was wildly over-provisioned. I picked the biggest instance I could find — an `ecs.r6.8xlarge` with 32 vCPUs and 256 GiB RAM — to run a Flask app that served maybe 20 requests per minute. I burned through credits in a week, panicked, learned how to downsize online, and discovered my app ran perfectly on a 2-vCPU box costing 94% less. Right-sizing matters more than raw power, and understanding the compute layer is the single most useful thing you can learn about any cloud platform.

This article is the complete guide to Elastic Compute Service. We start from what ECS actually is, move through instance families and pricing models, then build a production-ready app server from scratch using the CLI. By the end, you will have enough working knowledge to provision, secure, and operate ECS instances for real workloads.


---

## What ECS actually is

Elastic Compute Service is Alibaba Cloud's virtual machine product. If you have used AWS EC2, Azure VMs, or GCP Compute Engine, ECS is the direct equivalent. You get a virtual server running Linux or Windows, connected to a virtual network, with block storage attached. You control it over SSH or RDP, and you pay by the hour or by the month.

But ECS is not just "a VM." It is a composition of six building blocks, and understanding each one separately saves a lot of confusion:

| Component | What it is | AWS equivalent |
|---|---|---|
| **Instance** | The virtual machine itself — vCPUs, RAM, local NVMe | EC2 instance |
| **Image** | The OS template used to boot the instance | AMI |
| **Block Storage (disk)** | Network-attached persistent storage — system disk + data disks | EBS |
| **Security Group** | Stateful firewall rules attached to the instance's network interface | Security Group |
| **VPC / VSwitch** | The virtual network and subnet the instance lives in | VPC / Subnet |
| **ENI** | Elastic Network Interface — the virtual NIC | ENI |

When you "create an ECS instance" in the console, you are actually configuring all six at once. The console bundles them to reduce friction, but they are separate resources with independent lifecycles. You can detach a disk from one instance and attach it to another. You can move an ENI between instances. Security groups are shared across instances. Understanding this decomposition is what separates someone who uses ECS from someone who operates it.

### The ECS lifecycle

Every instance passes through a well-defined state machine:

```text
                  ┌──────────┐
         create   │          │  start
    ────────────► │ Stopped  │ ──────────┐
                  │          │           │
                  └────┬─────┘           ▼
                       │          ┌──────────┐
                 delete│          │          │
                       │          │ Starting │
                       ▼          │          │
                  ┌──────────┐    └────┬─────┘
                  │          │         │
                  │ Released │         ▼
                  │          │    ┌──────────┐
                  └──────────┘    │          │  stop
                                  │ Running  │ ──────┐
                                  │          │       │
                                  └────┬─────┘       ▼
                                       │       ┌──────────┐
                                       │       │          │
                                       └───────│ Stopping │
                                               │          │
                                               └──────────┘
```

The states that matter in practice:

- **Stopped**: No compute charges on pay-as-you-go (disk and IP charges continue). This is the state you want for dev instances overnight.
- **Running**: The instance is up and billing. CPU/memory/network metering is active.
- **Stopping**: Brief transition. On a `ForceStop` this can take up to 60 seconds. Graceful stop sends ACPI shutdown to the OS and waits.
- **Released**: Gone. The instance, its system disk, and its local disks are permanently deleted. Data disks survive only if you configured them with `DeleteWithInstance = false`.

One thing that trips people up: a Stopped instance still holds its private IP and its elastic IP association. You are not releasing network resources by stopping. If you want to actually free everything, you release the instance.

### ECS vs EC2: what's different

If you're coming from AWS, these are the meaningful differences:

1. **No instance store volumes** in the EC2 sense. ECS local NVMe disks exist on specific instance families (i-series), but most workloads use cloud disks exclusively.
2. **Security groups are VPC-scoped**, not region-scoped. You cannot share a security group across VPCs.
3. **Metadata endpoint** is `http://100.100.100.200/latest/meta-data/` instead of `169.254.169.254`. Cloud-init works the same way, but if you're porting scripts, update the URL.
4. **Chinese regions** have separate infrastructure. `cn-hangzhou` and `us-east-1` are not just different availability zones — they are completely independent control planes with separate accounts and billing.

## Instance family deep dive

Instance families are the core abstraction for hardware specialization. The naming convention is:

![ECS instance family comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_instance_families.png)

```yaml
ecs.{family}{generation}.{size}

Examples:
  ecs.g8i.large      →  General Purpose, gen 8 (Intel), large
  ecs.c8y.xlarge     →  Compute Optimized, gen 8 (Yitian ARM), xlarge
  ecs.r7.2xlarge     →  Memory Optimized, gen 7, 2xlarge
  ecs.gn7i.xlarge    →  GPU (NVIDIA), gen 7 (Intel), xlarge
```

The suffix letter after the generation number indicates the processor:

- No suffix or `i` = Intel Xeon
- `a` = AMD EPYC
- `y` = Alibaba Yitian 710 (ARM)

Here is the instance family reference you will actually use:

| Family | Type | vCPU : Memory | Processor | Network (Gbps) | Best for |
|---|---|---|---|---|---|
| **g7** | General Purpose | 1:4 | Intel Xeon (Ice Lake) | Up to 25 | Web servers, mid-tier APIs |
| **g8i** | General Purpose | 1:4 | Intel Xeon (Sapphire Rapids) | Up to 40 | General workloads, latest gen |
| **g8y** | General Purpose | 1:4 | Yitian 710 (ARM) | Up to 40 | Cost-efficient ARM workloads |
| **c7** | Compute Optimized | 1:2 | Intel Xeon (Ice Lake) | Up to 25 | High-CPU: encoding, CI/CD |
| **c8i** | Compute Optimized | 1:2 | Intel Xeon (Sapphire Rapids) | Up to 40 | Batch processing, game servers |
| **c8y** | Compute Optimized | 1:2 | Yitian 710 (ARM) | Up to 40 | ARM CI/CD, build farms |
| **r7** | Memory Optimized | 1:8 | Intel Xeon (Ice Lake) | Up to 25 | Databases, in-memory caches |
| **r8i** | Memory Optimized | 1:8 | Intel Xeon (Sapphire Rapids) | Up to 40 | Redis, Elasticsearch |
| **r8y** | Memory Optimized | 1:8 | Yitian 710 (ARM) | Up to 40 | Cost-efficient memory workloads |
| **gn7i** | GPU | Varies | A10 (24 GB) | Up to 32 | ML inference, fine-tuning |
| **gn7** | GPU | Varies | V100 (16/32 GB) | Up to 25 | ML training, HPC |
| **gn6v** | GPU | Varies | V100 (16 GB) | Up to 5 | Budget ML, dev inference |
| **t6** | Burstable | 1:1/1:2/1:4 | Intel Xeon | Up to 1.2 | Dev/test, micro workloads |
| **ebmg7** | Bare Metal | 1:4 | Intel Xeon (Ice Lake) | Up to 65 | High-performance, no hypervisor |
| **ebmc7** | Bare Metal | 1:2 | Intel Xeon (Ice Lake) | Up to 65 | Dedicated hardware, compliance |

### Sizes within a family

Each family offers sizes that double resources at each step:

| Size | vCPU | Memory (g-family, 1:4 ratio) |
|---|---|---|
| small | 1 | 2 GiB |
| large | 2 | 8 GiB |
| xlarge | 4 | 16 GiB |
| 2xlarge | 8 | 32 GiB |
| 4xlarge | 16 | 64 GiB |
| 8xlarge | 32 | 128 GiB |
| 16xlarge | 64 | 256 GiB |

> **Practical note on Yitian 710 (ARM):** The `*y` families are 20-30% cheaper than their Intel equivalents for the same specs. If your workload runs on Linux and you are not dependent on x86-specific binaries, always try the ARM variant first. Most Python/Node/Go/Java workloads just work. Docker images need to be multi-arch (`linux/arm64`), which is a one-line change in your Dockerfile build.

### Burstable instances: the trap and the fix

The `t6` family deserves special mention because it trips up almost everyone. Burstable instances accumulate CPU credits when idle and spend them when busy. Once you run out of credits, your CPU is throttled to a baseline — typically 10-20% of a vCPU.

This is perfect for a dev/test box that sits idle most of the day and occasionally runs a build. It is terrible for any workload with sustained CPU usage. I have personally seen production databases on `t6.large` instances run fine for weeks, then suddenly crater during a traffic spike because the credit balance hit zero.

The rule: if your average CPU exceeds the baseline (check the product page for your specific size), you need a non-burstable instance. Period.

## Choosing the right size

Decision-making here is simpler than it looks. Start with the workload, not the instance:

![ECS instance specification comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_instance_specs.png)

| Workload | Recommended start | Why |
|---|---|---|
| Static site / reverse proxy | `ecs.c7.large` (2 vCPU, 4 GiB) | CPU-bound, barely needs memory |
| REST API backend (Node/Python) | `ecs.g7.xlarge` (4 vCPU, 16 GiB) | Balanced — some CPU for JSON serialization, memory for connection pools |
| PostgreSQL / MySQL | `ecs.r7.2xlarge` (8 vCPU, 64 GiB) | Memory-heavy for buffer pool. Consider RDS instead. |
| Redis / Memcached | `ecs.r7.xlarge` (4 vCPU, 32 GiB) | All about memory. Again, consider managed Redis. |
| CI/CD runner | `ecs.c8y.xlarge` (4 vCPU, 8 GiB) | CPU-bound compilation. ARM is fine for most builds. |
| ML inference (LLM) | `ecs.gn7i.xlarge` (4 vCPU, 1x A10) | GPU for matrix ops, moderate CPU for pre/post processing |
| ML training | `ecs.gn7.8xlarge` (32 vCPU, 8x V100) | Multi-GPU for distributed training |
| Dev/test throwaway | `ecs.t6.large` (2 vCPU, 4 GiB) | Cheap, burstable, stop it at night |

The golden rule: **start small, monitor for one week, then resize.** ECS supports online instance type changes for most families — stop the instance, change the type, start it again. The whole process takes under two minutes. Over-provisioning from day one is burning money on speculation.

## Pricing models explained

ECS offers four ways to pay, and choosing the right one can cut your bill by 80%. Here they are, from most expensive to least:

![ECS pricing model comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_pricing_models.png)

### Pay-as-you-go (PAYG)

Billed per second, minimum one-minute granularity. When you stop the instance, compute charges stop (disk and IP continue). This is the default and the most expensive per-hour, but it has zero commitment and zero waste — you pay only for what you use.

Best for: dev/test, spiky workloads, instances that run a few hours per day.

### Subscription (prepaid)

Commit to 1 month, 3 months, 6 months, 1 year, 2 years, or 3 years. Discounts range from ~15% (1 month) to ~50% (3 years) compared to PAYG. You pay upfront. The instance runs whether you use it or not.

Best for: production workloads with predictable, steady utilization.

### Preemptible instances (spot)

Same hardware as PAYG, but prices float based on supply and demand — typically 70-90% cheaper. The catch: Alibaba Cloud can reclaim your instance with a 5-minute warning when demand spikes. You get interrupted.

Best for: stateless batch processing, CI/CD, distributed ML training (with checkpointing), any workload that can handle interruption.

### Savings plans and reserved instances

Savings Plans let you commit to a spending amount (e.g., 100 CNY/hour) across any instance family in a region, at 30-60% discount. Reserved Instances are similar but locked to a specific instance type and AZ.

Best for: large, well-understood fleets where you know your baseline usage.

### Pricing comparison

Here is a real cost comparison for `ecs.c7.large` (2 vCPU, 4 GiB) in `cn-beijing`, running 24/7 for one month (prices approximate, check current rates):

| Model | Hourly (CNY) | Monthly (CNY) | Savings vs PAYG |
|---|---|---|---|
| Pay-as-you-go | 0.68 | ~490 | — |
| Subscription (1 month) | — | ~415 | 15% |
| Subscription (1 year) | — | ~310 | 37% |
| Subscription (3 years) | — | ~245 | 50% |
| Preemptible (avg) | 0.10 | ~72 | 85% |
| Savings Plan (1 year) | — | ~295 | 40% |

The preemptible price is not a typo. If your workload tolerates interruption, spot instances are almost free. I run all CI/CD on spot and have been interrupted maybe three times in a year — always during major Chinese holiday shopping events when demand peaks.

> **The hybrid strategy I actually use:** Production runs on Subscription (1-year). Dev/test runs on PAYG with auto-stop scripts at midnight. Batch jobs run on Preemptible with a fallback to PAYG if spot capacity is unavailable. This cuts the overall bill by about 45% compared to all-PAYG.

## Creating an ECS instance step by step

### Prerequisites

Before you create an instance, you need:

1. **A VPC and VSwitch** in your target region. We cover VPC setup in detail in [Part 3](/en/aliyun-fullstack/03-vpc-networking/), but for this walkthrough, I will assume you have a VPC with a VSwitch in `cn-beijing-h`.
2. **A security group** in that VPC (we create one below).
3. **A key pair** for SSH access (we create one below).
4. **The Alibaba Cloud CLI (`aliyun`)** installed and configured. Run `aliyun configure` if you haven't already.

### Console walkthrough (quick version)

The console path is: **ECS Console** > **Instances** > **Create Instance**.

1. **Billing**: Select Pay-As-You-Go for now.
2. **Region**: `China (Beijing)`, Zone H.
3. **Instance Type**: Search for `ecs.c7.large`. Select it.
4. **Image**: Alibaba Cloud Linux 3.2104 LTS 64-bit (the default). This is CentOS-compatible and free.
5. **Storage**: System disk = 40 GiB ESSD PL0. No data disk for now.
6. **Networking**: Select your VPC, your VSwitch in zone H. Assign a public IP (1 Mbps is fine for SSH; use an SLB/ALB for production traffic).
7. **Security Group**: Select or create one that allows TCP 22 from your IP.
8. **Login**: Key Pair (not password).
9. **Advanced**: Paste your cloud-init script in User Data (we write one below).
10. **Create**.

That is seven clicks and three dropdowns. But if you are going to create more than one instance, or if you want reproducibility, use the CLI.

### CLI walkthrough (the real way)

First, let's create the supporting resources. If you already have a VPC and security group, skip ahead to the instance creation.

**Create a security group:**

```bash
# Create security group
aliyun ecs CreateSecurityGroup \
  --RegionId cn-beijing \
  --VpcId vpc-2ze1234567890abcdef \
  --SecurityGroupName "app-server-sg" \
  --SecurityGroupType normal \
  --Description "Security group for app servers"
```

Save the `SecurityGroupId` from the response — you will need it.

**Create a key pair:**

```bash
# Create key pair — the private key is returned ONLY ONCE
aliyun ecs CreateKeyPair \
  --RegionId cn-beijing \
  --KeyPairName "app-server-key" | tee keypair-response.json

# Extract and save the private key
cat keypair-response.json | jq -r '.PrivateKeyBody' > ~/.ssh/app-server-key.pem
chmod 600 ~/.ssh/app-server-key.pem
rm keypair-response.json
```

**Create the ECS instance:**

```bash
aliyun ecs CreateInstance \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --InstanceType ecs.c7.large \
  --ImageId aliyun_3_x64_20G_alibase_20240528.vhd \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --VSwitchId vsw-2ze1234567890abcdef \
  --InstanceName "app-server-01" \
  --HostName "app-server-01" \
  --KeyPairName "app-server-key" \
  --SystemDisk.Category cloud_essd \
  --SystemDisk.Size 40 \
  --SystemDisk.PerformanceLevel PL0 \
  --InternetMaxBandwidthOut 5 \
  --InternetChargeType PayByTraffic \
  --InstanceChargeType PostPaid \
  --SpotStrategy NoSpot \
  --UserData "$(base64 -w0 cloud-init.yaml)" \
  --Description "Production app server" \
  --Tag.1.Key Environment \
  --Tag.1.Value production \
  --Tag.2.Key App \
  --Tag.2.Value web-server
```

Note: `--ImageId` changes as new images are released. To find the latest Alibaba Cloud Linux 3 image:

```bash
aliyun ecs DescribeImages \
  --RegionId cn-beijing \
  --OSType linux \
  --ImageOwnerAlias system \
  --ImageName "Alibaba Cloud Linux 3*" \
  --PageSize 1 \
  --SortKey CreatedOn \
  | jq '.Images.Image[0] | {ImageId, ImageName, CreationTime}'
```

After creation, the instance is in `Stopped` state. Start it:

```bash
aliyun ecs StartInstance --InstanceId i-2ze1234567890abcdef
```

Wait about 30 seconds, then SSH in:

```bash
ssh -i ~/.ssh/app-server-key.pem root@<public-ip>
```

## Cloud-init: automate everything from boot

Nobody should SSH into a fresh instance and manually install packages. Cloud-init runs on first boot and configures the instance automatically. Every ECS image ships with cloud-init pre-installed.

![Cloud-init boot sequence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_cloud_init.png)

You pass your cloud-init configuration as `UserData` when creating the instance. It must be base64-encoded. Here is a comprehensive `cloud-init.yaml` that sets up a production-ready app server:

```yaml
#cloud-config

# --- System ---
timezone: Asia/Shanghai
locale: en_US.UTF-8

# --- Package management ---
package_update: true
package_upgrade: true
packages:
  - nginx
  - python3
  - python3-pip
  - python3-venv
  - git
  - curl
  - htop
  - jq
  - supervisor
  - certbot
  - python3-certbot-nginx
  - fail2ban

# --- Users ---
users:
  - name: deploy
    groups: [sudo, www-data]
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ... deploy@workstation

# --- SSH hardening ---
ssh_pwauth: false
disable_root: false

write_files:
  - path: /etc/ssh/sshd_config.d/hardened.conf
    content: |
      PermitRootLogin prohibit-password
      PasswordAuthentication no
      ChallengeResponseAuthentication no
      MaxAuthTries 3
      ClientAliveInterval 300
      ClientAliveCountMax 2

  - path: /opt/app/deploy.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      set -euxo pipefail
      cd /opt/app
      python3 -m venv venv
      source venv/bin/activate
      pip install --upgrade pip
      pip install flask gunicorn
      
      cat > app.py << 'PYEOF'
      from flask import Flask, jsonify
      import os, socket
      
      app = Flask(__name__)
      
      @app.route("/health")
      def health():
          return jsonify(status="ok", host=socket.gethostname())
      
      @app.route("/")
      def index():
          return jsonify(message="Hello from ECS", instance=os.uname().nodename)
      PYEOF
      
      cat > /etc/supervisor/conf.d/app.conf << 'SUPEOF'
      [program:app]
      command=/opt/app/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
      directory=/opt/app
      user=deploy
      autostart=true
      autorestart=true
      stderr_logfile=/var/log/app/error.log
      stdout_logfile=/var/log/app/access.log
      SUPEOF
      
      mkdir -p /var/log/app
      chown deploy:deploy /var/log/app
      supervisorctl reread
      supervisorctl update

  - path: /etc/nginx/sites-available/app
    content: |
      server {
          listen 80;
          server_name _;
          
          location / {
              proxy_pass http://127.0.0.1:8000;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header X-Forwarded-Proto $scheme;
          }
          
          location /health {
              proxy_pass http://127.0.0.1:8000/health;
              access_log off;
          }
      }

# --- Run commands on first boot ---
runcmd:
  # Enable and start fail2ban
  - systemctl enable fail2ban
  - systemctl start fail2ban

  # Configure nginx
  - ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app
  - rm -f /etc/nginx/sites-enabled/default
  - nginx -t && systemctl reload nginx

  # Deploy the application
  - /opt/app/deploy.sh

  # Restart SSH with hardened config
  - systemctl restart sshd

  # Log completion
  - echo "cloud-init complete at $(date)" >> /var/log/cloud-init-complete.log
```

After the instance boots, cloud-init processes this file in stages: set timezone, install packages, create users, write files, run commands. The whole process takes 2-3 minutes on an `ecs.c7.large`.

To verify cloud-init completed successfully:

```bash
# Check cloud-init status
cloud-init status --long

# If something failed, check the log
tail -100 /var/log/cloud-init-output.log

# Verify the app is running
curl http://localhost/health
```

> **Debugging tip:** The most common cloud-init failure is a YAML indentation error. Validate your config locally with `cloud-init schema --config-file cloud-init.yaml` before base64-encoding it. Also, `UserData` has a 16 KiB limit — if your script is complex, have cloud-init pull a script from OSS instead.

## Security groups: your first firewall

A security group is a stateful firewall at the ENI level. Every packet entering or leaving an ECS instance is evaluated against its security group rules. If no rule matches, the packet is **dropped** — default deny.

![Security group inbound and outbound rules](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_security_groups.png)

"Stateful" means that if you allow inbound TCP 80, the response packets are automatically allowed out. You do not need a matching outbound rule for return traffic.

### Common security group patterns

**Web server (public-facing):**

```bash
# Allow HTTP from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow HTTP"

# Allow HTTPS from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow HTTPS"

# Allow SSH from your office IP only
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2ze1234567890abcdef \
  --IpProtocol tcp \
  --PortRange 22/22 \
  --SourceCidrIp 203.0.113.50/32 \
  --Policy accept \
  --Priority 1 \
  --Description "Allow SSH from office"
```

**Database server (internal only):**

```bash
# Allow PostgreSQL from app server security group
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId sg-2zeDBSERVERSG \
  --IpProtocol tcp \
  --PortRange 5432/5432 \
  --SourceGroupId sg-2zeAPPSERVERSG \
  --Policy accept \
  --Priority 1 \
  --Description "Allow PostgreSQL from app servers"
```

Notice the last example uses `--SourceGroupId` instead of `--SourceCidrIp`. This means "allow traffic from any instance in that security group." This is the right way to do internal service communication — you never hardcode IPs.

### Rules I apply to every security group

1. **Never open SSH (22) to `0.0.0.0/0`.** Use your office CIDR, or better, use a bastion host.
2. **Never open database ports to the internet.** PostgreSQL on 5432, MySQL on 3306, Redis on 6379 — these should only be reachable from your app security group.
3. **Use descriptions on every rule.** Six months from now, you will not remember why port 8443 is open. The description tells you.
4. **Review rules quarterly.** Security groups accumulate stale rules like barnacles. Set a calendar reminder.

## Key pairs and SSH access

Passwords are bad for SSH. They are brute-forceable, they encourage password reuse, and they cannot be rotated without logging into the instance. Key pairs are the standard.

### Creating and using key pairs

We created a key pair earlier. Here is the complete SSH workflow:

```bash
# Your SSH config (~/.ssh/config)
Host app-server
    HostName 47.94.xxx.xxx
    User root
    IdentityFile ~/.ssh/app-server-key.pem
    StrictHostKeyChecking accept-new

Host bastion
    HostName 47.94.yyy.yyy
    User deploy
    IdentityFile ~/.ssh/bastion-key.pem
    StrictHostKeyChecking accept-new

# Access internal servers through a bastion host
Host app-internal
    HostName 172.16.1.10
    User deploy
    IdentityFile ~/.ssh/app-server-key.pem
    ProxyJump bastion
```

With this config:

```bash
# Direct access to public instance
ssh app-server

# Access internal instance via bastion (two-hop)
ssh app-internal

# Copy files through bastion
scp -o ProxyJump=bastion localfile.tar.gz app-internal:/opt/app/
```

The `ProxyJump` directive is the modern replacement for SSH tunneling through bastion hosts. It establishes the SSH connection through `bastion` transparently — you do not need to SSH to the bastion first, then SSH to the internal host. One command, two hops, no exposed internal IPs.

### Key rotation

To rotate keys without downtime:

1. Generate a new key pair in the ECS console or CLI.
2. Add the new public key to `~/.ssh/authorized_keys` on the target instance (or use cloud-init to manage keys via the API).
3. Test login with the new key.
4. Remove the old public key from `authorized_keys`.
5. Delete the old key pair from ECS.

For fleets, manage SSH keys through cloud-init or an Ansible playbook — never manually edit `authorized_keys` on 20 instances.

## Disks and storage

Every ECS instance has at least one disk: the system disk, which holds the OS. You can attach up to 16 additional data disks. All disks are network-attached block storage — they persist independently of the instance lifecycle (if configured correctly).

![ESSD disk type IOPS and throughput comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_disk_types.png)

### Disk types

| Type | IOPS (max) | Throughput (max) | Latency | Best for |
|---|---|---|---|---|
| **ESSD PL0** | 10,000 | 180 MB/s | 0.2-0.5 ms | System disks, light workloads |
| **ESSD PL1** | 50,000 | 350 MB/s | 0.1-0.3 ms | General production, databases |
| **ESSD PL2** | 100,000 | 750 MB/s | 0.1-0.3 ms | High-IOPS databases |
| **ESSD PL3** | 1,000,000 | 4,000 MB/s | 0.1-0.3 ms | Extreme performance, OLTP |
| **Standard SSD** | 25,000 | 300 MB/s | 0.5-2 ms | Legacy, non-critical |
| **Ultra Disk** | 5,000 | 140 MB/s | 1-3 ms | Cold storage, archives |

The performance level (PL0 through PL3) is the single most important storage decision you make. A database on PL0 will hit the 10,000 IOPS ceiling and queue operations; the same database on PL1 has 5x the headroom. The price difference between PL0 and PL1 is about 2x — still cheap compared to the compute cost.

> **My default:** PL0 for system disks (the OS does not need high IOPS), PL1 for data disks running databases or anything with fsync. If you are not sure, start with PL1 — the cost difference for a 100 GiB disk is about 40 CNY/month.

### Expanding disks online

ECS supports online disk expansion — you can grow a disk without stopping the instance or unmounting the filesystem:

```bash
# Step 1: Expand the disk via API (does not touch the filesystem)
aliyun ecs ResizeDisk \
  --DiskId d-2ze1234567890abcdef \
  --NewSize 100 \
  --Type online

# Step 2: SSH into the instance and grow the filesystem
# For ext4:
sudo growpart /dev/vda 1
sudo resize2fs /dev/vda1

# For xfs:
sudo growpart /dev/vda 1
sudo xfs_growfs /
```

You can only expand, never shrink. Plan your initial size conservatively and grow as needed — this is cloud, not a physical server where disk replacement requires a maintenance window.

### Snapshots

Snapshots are point-in-time copies of a disk. They are incremental (only changed blocks are stored) and crash-consistent. Use them for:

1. **Backup before risky operations.** `aliyun ecs CreateSnapshot --DiskId d-xxx` before you run that database migration.
2. **Creating images.** Snapshot a configured instance's system disk, then create a custom image from it. Every new instance boots fully configured in 30 seconds instead of waiting 3 minutes for cloud-init.
3. **Disaster recovery.** Set up automatic snapshot policies — daily snapshots retained for 7 days is a reasonable starting point.

```bash
# Create an on-demand snapshot
aliyun ecs CreateSnapshot \
  --DiskId d-2ze1234567890abcdef \
  --SnapshotName "pre-migration-backup" \
  --Description "Before v2.0 database migration" \
  --RetentionDays 30

# Create an automatic snapshot policy (daily at 02:00, keep 7 days)
aliyun ecs CreateAutoSnapshotPolicy \
  --RegionId cn-beijing \
  --autoSnapshotPolicyName "daily-7d" \
  --timePoints '["2"]' \
  --repeatWeekdays '["1","2","3","4","5","6","7"]' \
  --retentionDays 7

# Apply the policy to a disk
aliyun ecs ApplyAutoSnapshotPolicy \
  --autoSnapshotPolicyId sp-2ze1234567890abcdef \
  --diskIds '["d-2ze1234567890abcdef"]'
```

## Monitoring and maintenance

### CloudMonitor metrics

![ECS instance lifecycle states](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/02-ecs-compute/02_lifecycle.png)

Every ECS instance automatically reports metrics to CloudMonitor. The ones you should watch:

- **CPUUtilization**: Sustained >80% means you need to scale up or out.
- **MemoryUsedUtilization**: Sustained >85% means you are approaching OOM territory. Note: this metric requires the CloudMonitor agent (installed by default on Alibaba Cloud Linux).
- **DiskReadIOPS / DiskWriteIOPS**: Compare against your disk's PL limit. If you are consistently at 80% of the limit, upgrade the PL or add a disk.
- **IntranetInRate / IntranetOutRate**: Network throughput. If you are hitting the instance family's limit, you need a bigger instance.
- **vm.TcpConnectionCount**: Connection count. Useful for detecting connection leaks.

### Setting up alerts

```bash
# Alert when CPU > 90% for 5 minutes
aliyun cms PutResourceMetricRule \
  --RuleId "ecs-cpu-high" \
  --RuleName "ECS CPU High" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "CPUUtilization" \
  --Resources '[{"instanceId":"i-2ze1234567890abcdef"}]' \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 90 \
  --Escalations.Critical.Times 3 \
  --Period 60
```

### Scheduled maintenance and live migration

Alibaba Cloud periodically maintains the physical infrastructure. When your instance's host needs maintenance, you receive a notification (email, SMS, or console alert) with a scheduled window — typically 2-4 weeks away.

For most instance families, Alibaba Cloud performs **live migration**: your instance is moved to another physical host with near-zero downtime (a brief pause of 10-100ms during the final memory copy). You do not need to do anything.

For instances that cannot be live-migrated (bare metal, GPU instances with VGPU), you get a maintenance window and need to restart the instance yourself. Set up a process to check for pending maintenance events:

```bash
# Check for pending system events
aliyun ecs DescribeInstancesFullStatus \
  --RegionId cn-beijing \
  --InstanceId.1 i-2ze1234567890abcdef \
  | jq '.InstanceFullStatusSet.InstanceFullStatusType[] | .ScheduledSystemEventSet'
```

## Solution: production-ready app server

Let's put everything together. Here is the complete sequence to go from nothing to a running, secured, monitored Flask application on ECS — all via CLI.

### Step 1: Create the VPC and VSwitch

```bash
# Create VPC
VPC_ID=$(aliyun vpc CreateVpc \
  --RegionId cn-beijing \
  --CidrBlock 172.16.0.0/12 \
  --VpcName "prod-vpc" \
  | jq -r '.VpcId')

echo "VPC: $VPC_ID"

# Wait for VPC to be available
sleep 5

# Create VSwitch in zone H
VSWITCH_ID=$(aliyun vpc CreateVSwitch \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --VpcId "$VPC_ID" \
  --CidrBlock 172.16.1.0/24 \
  --VSwitchName "prod-app-subnet" \
  | jq -r '.VSwitchId')

echo "VSwitch: $VSWITCH_ID"
```

### Step 2: Create security group with rules

```bash
# Create security group
SG_ID=$(aliyun ecs CreateSecurityGroup \
  --RegionId cn-beijing \
  --VpcId "$VPC_ID" \
  --SecurityGroupName "prod-app-sg" \
  | jq -r '.SecurityGroupId')

echo "Security Group: $SG_ID"

# Allow HTTP
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0 --Policy accept --Priority 1

# Allow HTTPS
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0 --Policy accept --Priority 1

# Allow SSH from specific IP (replace with yours)
aliyun ecs AuthorizeSecurityGroup \
  --RegionId cn-beijing \
  --SecurityGroupId "$SG_ID" \
  --IpProtocol tcp --PortRange 22/22 \
  --SourceCidrIp "$(curl -s ifconfig.me)/32" --Policy accept --Priority 1
```

### Step 3: Create key pair

```bash
aliyun ecs CreateKeyPair \
  --RegionId cn-beijing \
  --KeyPairName "prod-app-key" \
  | jq -r '.PrivateKeyBody' > ~/.ssh/prod-app-key.pem

chmod 600 ~/.ssh/prod-app-key.pem
```

### Step 4: Find the latest image

```bash
IMAGE_ID=$(aliyun ecs DescribeImages \
  --RegionId cn-beijing \
  --OSType linux \
  --ImageOwnerAlias system \
  --ImageName "Alibaba Cloud Linux 3*" \
  --PageSize 1 \
  | jq -r '.Images.Image[0].ImageId')

echo "Image: $IMAGE_ID"
```

### Step 5: Create and start the instance

```bash
# Create instance with cloud-init user data
INSTANCE_ID=$(aliyun ecs CreateInstance \
  --RegionId cn-beijing \
  --ZoneId cn-beijing-h \
  --InstanceType ecs.c7.large \
  --ImageId "$IMAGE_ID" \
  --SecurityGroupId "$SG_ID" \
  --VSwitchId "$VSWITCH_ID" \
  --InstanceName "prod-app-01" \
  --HostName "prod-app-01" \
  --KeyPairName "prod-app-key" \
  --SystemDisk.Category cloud_essd \
  --SystemDisk.Size 40 \
  --SystemDisk.PerformanceLevel PL0 \
  --InternetMaxBandwidthOut 10 \
  --InternetChargeType PayByTraffic \
  --InstanceChargeType PostPaid \
  --UserData "$(base64 -w0 cloud-init.yaml)" \
  | jq -r '.InstanceId')

echo "Instance: $INSTANCE_ID"

# Start the instance
aliyun ecs StartInstance --InstanceId "$INSTANCE_ID"

# Wait for it to be running
echo "Waiting for instance to start..."
sleep 30

# Get the public IP
PUBLIC_IP=$(aliyun ecs DescribeInstances \
  --RegionId cn-beijing \
  --InstanceIds "[\"$INSTANCE_ID\"]" \
  | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

echo "Public IP: $PUBLIC_IP"
```

### Step 6: Verify the deployment

```bash
# Wait for cloud-init to finish (usually 2-3 min)
sleep 120

# Test the health endpoint
curl -s "http://$PUBLIC_IP/health" | jq .
# Expected: {"host":"prod-app-01","status":"ok"}

# SSH in and check logs
ssh -i ~/.ssh/prod-app-key.pem root@"$PUBLIC_IP" \
  "cloud-init status --long && supervisorctl status"
```

### Step 7: Set up HTTPS with certbot

After pointing your domain's DNS A record to the public IP:

```bash
ssh -i ~/.ssh/prod-app-key.pem root@"$PUBLIC_IP" << 'EOF'
  certbot --nginx -d app.example.com \
    --non-interactive --agree-tos \
    --email admin@example.com

  # Verify auto-renewal
  certbot renew --dry-run

  # Set up cron for renewal
  echo "0 3 * * * certbot renew --quiet" | crontab -
EOF
```

### Step 8: Set up monitoring

```bash
# Create alert for CPU
aliyun cms PutResourceMetricRule \
  --RuleId "prod-app-01-cpu" \
  --RuleName "Prod App CPU High" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "CPUUtilization" \
  --Resources "[{\"instanceId\":\"$INSTANCE_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Average" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 90 \
  --Escalations.Critical.Times 3 \
  --Period 60

# Create alert for disk usage
aliyun cms PutResourceMetricRule \
  --RuleId "prod-app-01-disk" \
  --RuleName "Prod App Disk Full" \
  --Namespace "acs_ecs_dashboard" \
  --MetricName "diskusage_utilization" \
  --Resources "[{\"instanceId\":\"$INSTANCE_ID\"}]" \
  --ContactGroups '["ops-team"]' \
  --Escalations.Critical.Statistics "Maximum" \
  --Escalations.Critical.ComparisonOperator "GreaterThanThreshold" \
  --Escalations.Critical.Threshold 85 \
  --Escalations.Critical.Times 1 \
  --Period 300
```

That is a complete production deployment: VPC, security group, key pair, ECS instance with cloud-init automation, nginx reverse proxy, Flask app under supervisor, HTTPS with auto-renewal, and monitoring alerts. From zero to production in about 5 minutes of API calls and 3 minutes of cloud-init execution.

## Summary

**Start small, resize later.** ECS supports online instance type changes. Do not guess what you need — measure and adjust.

**Use the right instance family.** General Purpose (g-series) is the default. Switch to Compute (c-series) for CPU-bound, Memory (r-series) for data-heavy, or GPU (gn-series) for ML workloads. Try ARM (y-suffix) for 20-30% cost savings.

**Mix pricing models.** Subscription for steady production, PAYG for dev/test, Preemptible for batch. A hybrid strategy typically saves 40-50% over all-PAYG.

**Automate from boot.** Cloud-init should bring every instance to production-ready without manual SSH. If you find yourself SSHing in to install things, your cloud-init is incomplete.

**Security groups are not optional.** Default deny, allow only what you need, use security group references (not IPs) for internal traffic, review quarterly.

**Snapshots are your safety net.** Automatic daily snapshots cost almost nothing and have saved me from disaster more than once.

For infrastructure-as-code approach to ECS, see our [Terraform series, Part 4: Compute](/en/terraform-agents/04-compute-for-agent-runtime/), which covers the same concepts as Terraform resources. Next up in this series, [Part 3](/en/aliyun-fullstack/03-vpc-networking/) dives deep into VPC, VSwitches, route tables, and NAT gateways — the networking layer that connects all your ECS instances.
