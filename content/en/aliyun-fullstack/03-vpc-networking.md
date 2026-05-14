---
title: "Alibaba Cloud Full Stack (3): VPC, SLB, and the Network Layer"
date: 2026-04-30 09:00:00
tags:
  - Alibaba Cloud
  - VPC
  - SLB
  - Networking
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 3
description: "Build a production network from scratch: VPC architecture, CIDR planning, VSwitches across availability zones, security groups as stateful firewalls, SLB for load balancing, NAT Gateway for outbound traffic, and EIP for public access."
disableNunjucks: true
translationKey: "aliyun-fullstack-3"
---

Every outage I have debugged in the cloud ultimately traced back to networking. Bad CIDR planning that ran out of IPs six months in. Missing routes that silently dropped traffic between tiers. Security groups that were either wide open (hello, port 22 to `0.0.0.0/0`) or so locked down that health checks failed and the load balancer kept draining healthy instances. Getting the network layer right is the single most important thing you can do before deploying anything else, and it is the single most painful thing to fix retroactively because changing a VPC CIDR means recreating everything inside it.


We set up the basic VPC in [Part 1](/en/aliyun-fullstack/01-ecosystem-map/) — now we are going deep. By the end of this article you will have a production-grade multi-AZ network with isolated tiers, proper security boundaries, internet access via NAT, and load balancing via SLB. The ECS instances that go into these subnets are covered in [Part 2](/en/aliyun-fullstack/02-ecs-compute/). For the Terraform approach to VPC setup, see [Terraform Part 3: VPC and Security Baseline](/en/terraform-agents/03-vpc-and-security-baseline/).

---

## What Is a VPC?

A Virtual Private Cloud is your own isolated network segment on Alibaba Cloud. Think of it as a private data center network that you define entirely in software: you pick the IP range, you carve it into subnets, you write the firewall rules, you decide what can reach the internet and what stays internal. Nothing gets in or out unless you explicitly allow it.

![VPC architecture overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_vpc_architecture.png)

If you have used AWS, the mental model is almost identical. Alibaba Cloud's VPC is functionally equivalent to AWS VPC, with different naming:

| Alibaba Cloud | AWS | What it does |
|---|---|---|
| VPC | VPC | The top-level isolated network |
| VSwitch | Subnet | A subnet tied to one availability zone |
| Route Table | Route Table | Routing rules for traffic direction |
| Security Group | Security Group | Stateful instance-level firewall |
| Network ACL | Network ACL | Stateless subnet-level firewall |
| EIP | Elastic IP | Static public IP address |
| NAT Gateway | NAT Gateway | Internet access for private subnets |
| SLB (CLB/ALB/NLB) | ELB (CLB/ALB/NLB) | Load balancing |

Before VPC existed, Alibaba Cloud had a "Classic Network" where all instances in a region shared a flat network. Classic Network is deprecated and no longer available for new accounts. If you encounter it in legacy documentation, ignore it. Everything runs in VPC now.

The key components you will work with:

- **VPC** — The container. One VPC per region (you can have multiple). Defined by a CIDR block.
- **VSwitch** — A subnet within the VPC, bound to exactly one availability zone. This is where instances actually live.
- **Route Table** — Controls where traffic goes. Every VPC has a system route table; you can create custom ones.
- **Security Group** — A stateful firewall attached to instances. Rules allow traffic; anything not explicitly allowed is denied inbound.
- **Network ACL** — An optional stateless firewall at the VSwitch level. Most setups skip this and rely on security groups.

One VPC cannot span regions. If you need cross-region connectivity, that is where Cloud Enterprise Network (CEN) comes in, which we cover later in this article.

## CIDR Planning: Get This Right from Day One

CIDR (Classless Inter-Domain Routing) notation defines your IP address space. Get this wrong and you will spend a very unpleasant weekend migrating everything to a new VPC.

![CIDR planning guide for VPC subnets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_cidr_planning.png)

The notation works like this: `10.0.0.0/16` means "all IPs from `10.0.0.0` to `10.0.255.255`." The number after the slash is the prefix length — how many bits are fixed. The remaining bits are yours to allocate.

| CIDR | Prefix bits | Available IPs | Typical use |
|---|---|---|---|
| /8 | 8 | 16,777,216 | Enterprise-wide (10.0.0.0/8) |
| /16 | 16 | 65,536 | One VPC |
| /20 | 20 | 4,096 | Large subnet |
| /24 | 24 | 256 (254 usable) | Standard subnet |
| /28 | 28 | 16 (14 usable) | Tiny subnet (NAT, bastion) |

Alibaba Cloud VPCs support CIDR blocks from /8 to /24 using the private ranges: `10.0.0.0/8`, `172.16.0.0/12`, or `192.168.0.0/16`. VSwitches can go down to /29 (8 IPs, 6 usable — Alibaba Cloud reserves the first two and last one in each VSwitch).

The golden rule: **plan for 10x your current need**. If you think you need 50 IPs, allocate a /24 (254 usable). If you think you need 500 IPs, allocate a /20 (4,094 usable). Subnets cannot be resized after creation — you would have to create a new VSwitch and migrate instances.

Here is my standard planning table for a 3-tier architecture across 2 availability zones:

| Tier | AZ-A VSwitch | AZ-A CIDR | AZ-B VSwitch | AZ-B CIDR | Usable IPs per VSwitch |
|---|---|---|---|---|---|
| Public (web/ALB) | vsw-public-a | 10.0.1.0/24 | vsw-public-b | 10.0.2.0/24 | 251 |
| Private App | vsw-app-a | 10.0.10.0/24 | vsw-app-b | 10.0.11.0/24 | 251 |
| Private Data | vsw-data-a | 10.0.20.0/24 | vsw-data-b | 10.0.21.0/24 | 251 |

The VPC itself gets `10.0.0.0/16`, giving us 65,534 usable IPs and room to add more subnets later without running out of address space. I deliberately leave gaps between the tier ranges (1.x, 10.x, 20.x) so that when you inevitably add a fourth tier — maybe a cache layer at 10.0.30.0/24 — it slots in cleanly without renumbering.

> **Note:** If your VPCs will ever need to peer with each other (via CEN or VPN), their CIDR blocks must not overlap. Plan a scheme like VPC-prod = 10.0.0.0/16, VPC-staging = 10.1.0.0/16, VPC-dev = 10.2.0.0/16.

## VSwitches: Subnets Across Availability Zones

A VSwitch is a subnet, and every VSwitch lives in exactly one availability zone. You cannot stretch a VSwitch across zones. This is by design — it means a zone failure only takes out instances in VSwitches assigned to that zone, not your entire tier.

![Multi-AZ VSwitch topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_vswitch_layout.png)

The pattern is one VSwitch per AZ per tier. For a 3-tier setup across 2 AZs, that is 6 VSwitches. For 3 AZs, that is 9. I have never needed more than 3 AZs for a single application.

First, create the VPC:

```bash
aliyun vpc CreateVpc \
  --RegionId cn-hangzhou \
  --CidrBlock 10.0.0.0/16 \
  --VpcName prod-vpc \
  --Description "Production VPC for 3-tier web application"
```

Note the `VpcId` from the response. Then create VSwitches for each tier and zone:

```bash
# Public tier - AZ A
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.1.0/24 \
  --VSwitchName prod-public-a \
  --Description "Public subnet in AZ-A for ALB and bastion"

# Public tier - AZ B
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.2.0/24 \
  --VSwitchName prod-public-b \
  --Description "Public subnet in AZ-B for ALB and bastion"

# App tier - AZ A
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.10.0/24 \
  --VSwitchName prod-app-a \
  --Description "Private app subnet in AZ-A"

# App tier - AZ B
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.11.0/24 \
  --VSwitchName prod-app-b \
  --Description "Private app subnet in AZ-B"

# Data tier - AZ A
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.20.0/24 \
  --VSwitchName prod-data-a \
  --Description "Private data subnet in AZ-A for RDS/Redis"

# Data tier - AZ B
aliyun vpc CreateVSwitch \
  --VpcId vpc-bp1xxxxxxxxx \
  --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.21.0/24 \
  --VSwitchName prod-data-b \
  --Description "Private data subnet in AZ-B for RDS/Redis"
```

You can verify the layout:

```bash
aliyun vpc DescribeVSwitches \
  --VpcId vpc-bp1xxxxxxxxx \
  --output cols=VSwitchId,VSwitchName,ZoneId,CidrBlock,AvailableIpAddressCount
```

A few things worth knowing:

- The `ZoneId` must match a real AZ in your region. Run `aliyun ecs DescribeZones --RegionId cn-hangzhou` to list available zones.
- Not all instance types are available in all zones. Check before committing to a zone layout.
- You can add VSwitches to an existing VPC at any time, as long as the CIDR does not overlap with existing VSwitches. This is why leaving gaps in the CIDR plan matters.

## Route Tables

Every VPC comes with a system route table that you cannot delete. It contains one entry you care about: a local route that automatically enables all VSwitches within the VPC to communicate with each other. This is implicit — you will not see it in the console, but traffic between `10.0.1.0/24` and `10.0.20.0/24` just works.

![Route table decision flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_route_table.png)

The system route table also handles the default route (`0.0.0.0/0`), which initially goes nowhere. To give instances internet access, you point this default route at a NAT Gateway or an internet-facing router.

For most setups, the system route table is enough. You create custom route tables when you need different routing behavior for different VSwitches — for example, if your public VSwitches should route `0.0.0.0/0` to an internet gateway but your private VSwitches should route it to a NAT Gateway:

```bash
# Create a custom route table
aliyun vpc CreateRouteTable \
  --VpcId vpc-bp1xxxxxxxxx \
  --RouteTableName prod-private-rt \
  --Description "Route table for private subnets via NAT"

# Add a default route pointing to NAT Gateway
aliyun vpc CreateRouteEntry \
  --RouteTableId vtb-bp1xxxxxxxxx \
  --DestinationCidrBlock 0.0.0.0/0 \
  --NextHopType NatGateway \
  --NextHopId ngw-bp1xxxxxxxxx

# Associate with private VSwitches
aliyun vpc AssociateRouteTable \
  --RouteTableId vtb-bp1xxxxxxxxx \
  --VSwitchId vsw-bp1xxxxxxxxx
```

When a VSwitch is associated with a custom route table, that table takes precedence over the system route table. The local route (intra-VPC traffic) is always inherited and cannot be overridden.

One route table detail that trips people up: Alibaba Cloud evaluates routes by longest prefix match. A route for `10.0.10.0/24` is more specific than `10.0.0.0/16`, which is more specific than `0.0.0.0/0`. Traffic always follows the most specific matching route.

## Security Groups Deep Dive

Security groups are where most people either under-invest (leaving everything open) or over-invest (creating 200 rules that nobody can audit). The right answer is a small number of groups with clear, intent-revealing names.

A security group is a stateful firewall at the instance level. "Stateful" means if you allow inbound traffic on port 80, the response traffic is automatically allowed out — you do not need a separate outbound rule for return packets.

Default behavior:

- **Inbound**: deny all (nothing can reach your instance unless you add a rule)
- **Outbound**: allow all (your instance can reach anything, including the internet if routing permits)

This default is reasonable for getting started but too permissive on the outbound side for production. A compromised instance with unrestricted outbound can exfiltrate data or join a botnet.

### Rule anatomy

Each rule has:

- **Direction**: Inbound or Outbound
- **Protocol**: TCP, UDP, ICMP, GRE, or All
- **Port range**: e.g., 80/80 (single port), 1/65535 (all ports), 443/443
- **Source/Destination**: A CIDR block (`0.0.0.0/0`) or another security group ID
- **Priority**: 1 (highest) to 100 (lowest). Lower number wins.
- **Action**: Allow or Drop

The priority system is powerful. You can set a broad "deny all" at priority 100, then punch specific holes at priority 1. Alibaba Cloud evaluates rules from highest priority (1) to lowest (100) and applies the first match.

### Enterprise vs Basic security groups

Alibaba Cloud offers two types:

| Feature | Basic Security Group | Enterprise Security Group |
|---|---|---|
| Max rules | 200 | 200 |
| Max instances | 2,000 | 65,535 |
| Allow rules referencing other SGs | Yes | No |
| Default inbound between members | Configurable | Isolated |
| Performance | Standard | Higher throughput |

For most setups, Basic security groups are the right choice because they support security-group-to-security-group references. Enterprise security groups are for large-scale deployments (thousands of instances) where you need higher network throughput and are willing to use CIDR-based rules exclusively.

### The three-group pattern

Here is the pattern I use for every 3-tier application:

```bash
# Create security groups
aliyun ecs CreateSecurityGroup \
  --VpcId vpc-bp1xxxxxxxxx \
  --SecurityGroupName prod-web-sg \
  --Description "Web tier: ALB and public-facing instances"

aliyun ecs CreateSecurityGroup \
  --VpcId vpc-bp1xxxxxxxxx \
  --SecurityGroupName prod-app-sg \
  --Description "App tier: backend services"

aliyun ecs CreateSecurityGroup \
  --VpcId vpc-bp1xxxxxxxxx \
  --SecurityGroupName prod-data-sg \
  --Description "Data tier: RDS, Redis, Elasticsearch"
```

Now add rules. The web tier accepts HTTP/HTTPS from anywhere and SSH from a bastion CIDR:

```bash
# Web tier: allow HTTP from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-web-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0 \
  --Priority 1 \
  --Description "HTTP from internet"

# Web tier: allow HTTPS from anywhere
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-web-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0 \
  --Priority 1 \
  --Description "HTTPS from internet"

# Web tier: SSH from bastion only
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-web-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 22/22 \
  --SourceCidrIp 10.0.1.0/24 \
  --Priority 1 \
  --Description "SSH from bastion subnet only"
```

The app tier only accepts traffic from the web security group:

```bash
# App tier: accept traffic from web tier on port 8080
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-app-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 8080/8080 \
  --SourceGroupId sg-web-xxxxxxxxx \
  --Priority 1 \
  --Description "App port from web tier SG"

# App tier: accept health checks from web tier
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-app-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 8081/8081 \
  --SourceGroupId sg-web-xxxxxxxxx \
  --Priority 1 \
  --Description "Health check port from web tier SG"
```

The data tier only accepts connections from the app tier:

```bash
# Data tier: MySQL from app tier
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-data-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 3306/3306 \
  --SourceGroupId sg-app-xxxxxxxxx \
  --Priority 1 \
  --Description "MySQL from app tier SG"

# Data tier: Redis from app tier
aliyun ecs AuthorizeSecurityGroup \
  --SecurityGroupId sg-data-xxxxxxxxx \
  --IpProtocol tcp \
  --PortRange 6379/6379 \
  --SourceGroupId sg-app-xxxxxxxxx \
  --Priority 1 \
  --Description "Redis from app tier SG"
```

This creates a clean chain: internet --> web-sg --> app-sg --> data-sg. No tier can be reached directly from a tier that is not adjacent. A compromised web server cannot directly query the database.

The complete rule set in table form:

| Security Group | Direction | Protocol | Port | Source / Dest | Priority | Description |
|---|---|---|---|---|---|---|
| prod-web-sg | Inbound | TCP | 80 | 0.0.0.0/0 | 1 | HTTP |
| prod-web-sg | Inbound | TCP | 443 | 0.0.0.0/0 | 1 | HTTPS |
| prod-web-sg | Inbound | TCP | 22 | 10.0.1.0/24 | 1 | SSH from bastion |
| prod-app-sg | Inbound | TCP | 8080 | sg-web-xxx | 1 | App traffic from web |
| prod-app-sg | Inbound | TCP | 8081 | sg-web-xxx | 1 | Health check |
| prod-data-sg | Inbound | TCP | 3306 | sg-app-xxx | 1 | MySQL from app |
| prod-data-sg | Inbound | TCP | 6379 | sg-app-xxx | 1 | Redis from app |

## Elastic IP (EIP)

An Elastic IP is a static public IP address that you own independently of any instance. You can allocate it, bind it to an ECS instance or NAT Gateway, unbind it, and rebind it to something else. The IP stays yours until you release it.

When to use EIP:

- **Bastion / jump hosts** that need a fixed public IP for SSH access
- **Small deployments** (1-2 instances) where a load balancer is overkill
- **NAT Gateway** requires an EIP for outbound internet access
- **Services that need a stable IP** for whitelisting by partners

When NOT to use EIP:

- **Web applications serving real traffic** — use SLB instead (it handles failover)
- **Every instance** — if you are binding EIPs to 10 instances, you need a load balancer

There are two billing modes:

| Billing mode | How it works | Best for |
|---|---|---|
| Pay-By-Traffic | Pay per GB transferred | Bursty workloads, dev/test |
| Pay-By-Bandwidth | Pay for reserved Mbps | Steady, predictable traffic |

Pay-By-Traffic is almost always right for getting started. You only pay for what you use, and the per-GB rate (roughly 0.12 USD/GB for outbound) is reasonable until you hit sustained high bandwidth.

Creating and binding an EIP:

```bash
# Allocate an EIP
aliyun vpc AllocateEipAddress \
  --RegionId cn-hangzhou \
  --Bandwidth 100 \
  --InternetChargeType PayByTraffic \
  --InstanceChargeType PostPaid \
  --ISP BGP

# Bind to an ECS instance
aliyun vpc AssociateEipAddress \
  --AllocationId eip-bp1xxxxxxxxx \
  --InstanceId i-bp1xxxxxxxxx \
  --InstanceType EcsInstance

# Later, unbind it
aliyun vpc UnassociateEipAddress \
  --AllocationId eip-bp1xxxxxxxxx \
  --InstanceId i-bp1xxxxxxxxx \
  --InstanceType EcsInstance
```

A common gotcha: an ECS instance can only have one EIP bound at a time via the primary ENI (Elastic Network Interface). If you need multiple public IPs on one instance, create secondary ENIs and bind EIPs to those. But if you need multiple public IPs, you almost certainly need an SLB instead.

## NAT Gateway: Internet Access for Private Subnets

Instances in private subnets (the app and data tiers from our plan) have no public IP. They cannot reach the internet by default. But they often need to — pulling Docker images, calling external APIs, downloading security patches. NAT Gateway solves this.

![NAT Gateway architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_nat_gateway.png)

NAT Gateway sits in a public subnet and provides two functions:

**SNAT (Source NAT)** — outbound. Private instances send traffic to the NAT Gateway, which replaces the source IP with its own EIP and forwards the request to the internet. The response comes back to the NAT Gateway, which routes it back to the originating private instance. The private instance never gets a public IP.

**DNAT (Destination NAT)** — inbound. Maps a specific port on the NAT Gateway's EIP to a specific private instance and port. Incoming traffic to `EIP:8080` gets forwarded to `10.0.10.5:8080`. This is useful for one-off services that need to be reachable from the internet but do not warrant an SLB.

When to use which:

| Need | Solution |
|---|---|
| Private instances need to reach the internet (pull updates, call APIs) | SNAT via NAT Gateway |
| One specific service needs to be reachable from outside | DNAT or EIP |
| Multiple instances serving the same service need to be reachable | SLB (not NAT) |
| A single dev/test instance needs a public IP | EIP directly |

Creating a NAT Gateway:

```bash
# Create an Enhanced NAT Gateway
aliyun vpc CreateNatGateway \
  --RegionId cn-hangzhou \
  --VpcId vpc-bp1xxxxxxxxx \
  --VSwitchId vsw-public-a-xxxxxxxxx \
  --NatGatewayName prod-nat \
  --NatType Enhanced \
  --InternetChargeType PayByLcu \
  --Description "NAT for private subnet outbound access"

# Allocate an EIP for the NAT Gateway
aliyun vpc AllocateEipAddress \
  --RegionId cn-hangzhou \
  --Bandwidth 200 \
  --InternetChargeType PayByTraffic

# Bind EIP to NAT Gateway
aliyun vpc AssociateEipAddress \
  --AllocationId eip-bp1yyyyyyyyy \
  --InstanceId ngw-bp1xxxxxxxxx \
  --InstanceType Nat
```

Now create SNAT entries so private subnets can reach the internet:

```bash
# SNAT for app tier AZ-A
aliyun vpc CreateSnatEntry \
  --SnatTableId stb-bp1xxxxxxxxx \
  --SourceVSwitchId vsw-app-a-xxxxxxxxx \
  --SnatIp 47.xxx.xxx.xxx \
  --SnatEntryName "app-a-outbound"

# SNAT for app tier AZ-B
aliyun vpc CreateSnatEntry \
  --SnatTableId stb-bp1xxxxxxxxx \
  --SourceVSwitchId vsw-app-b-xxxxxxxxx \
  --SnatIp 47.xxx.xxx.xxx \
  --SnatEntryName "app-b-outbound"

# SNAT for data tier AZ-A (for package updates)
aliyun vpc CreateSnatEntry \
  --SnatTableId stb-bp1xxxxxxxxx \
  --SourceVSwitchId vsw-data-a-xxxxxxxxx \
  --SnatIp 47.xxx.xxx.xxx \
  --SnatEntryName "data-a-outbound"

# SNAT for data tier AZ-B
aliyun vpc CreateSnatEntry \
  --SnatTableId stb-bp1xxxxxxxxx \
  --SourceVSwitchId vsw-data-b-xxxxxxxxx \
  --SnatIp 47.xxx.xxx.xxx \
  --SnatEntryName "data-b-outbound"
```

The `SnatIp` is the EIP address you bound to the NAT Gateway. All outbound traffic from these VSwitches will appear to come from this IP. This is useful for whitelisting — if an external API needs to whitelist your IP, you give them the NAT EIP.

Always use the Enhanced NAT type. The "Normal" type is the legacy version with lower throughput, no support for multi-EIP SNAT, and no integration with newer services. Enhanced NAT is billed by LCU (Logical Connection Unit) which scales with your actual usage.

## SLB: Server Load Balancer

Server Load Balancer distributes incoming traffic across multiple backend instances. It is the front door for any service that needs to be highly available, and it is what separates "I have two servers" from "I have a production deployment."

![SLB Layer 4 vs Layer 7 comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_slb_comparison.png)

Alibaba Cloud has three SLB products, and the naming is initially confusing:

| Product | Full Name | Layer | Protocol | Best for |
|---|---|---|---|---|
| CLB | Classic Load Balancer | 4 and 7 | TCP/UDP/HTTP/HTTPS | Legacy, still widely used |
| ALB | Application Load Balancer | 7 | HTTP/HTTPS/gRPC | Modern web apps, content-based routing |
| NLB | Network Load Balancer | 4 | TCP/UDP/TLS | High-performance TCP, gaming, IoT |

**CLB** is the original. It works at both Layer 4 (TCP/UDP forwarding) and Layer 7 (HTTP/HTTPS with host/path routing). It is battle-tested and handles most use cases. If the docs or a tutorial says "SLB" without qualification, they mean CLB.

**ALB** is the modern Layer 7 option. It supports content-based routing (route `/api/*` to one backend, `/static/*` to another), gRPC, WebSocket, and has better integration with WAF and DCDN. For new HTTP/HTTPS workloads, ALB is the right choice.

**NLB** is for raw TCP/UDP. It passes through packets without terminating the connection, preserving the client's source IP. Use it for database proxies, game servers, or anything that needs maximum throughput with minimum latency at Layer 4.

For a typical web application, ALB is what you want:

```bash
# Create an ALB instance
aliyun alb CreateLoadBalancer \
  --LoadBalancerName prod-web-alb \
  --VpcId vpc-bp1xxxxxxxxx \
  --AddressType Internet \
  --LoadBalancerEdition Standard \
  --LoadBalancerBillingConfig '{"PayType":"PostPay"}' \
  --ZoneMappings '[
    {"ZoneId":"cn-hangzhou-h","VSwitchId":"vsw-public-a-xxxxxxxxx"},
    {"ZoneId":"cn-hangzhou-i","VSwitchId":"vsw-public-b-xxxxxxxxx"}
  ]'
```

The ALB spans two AZs. If one zone goes down, the ALB in the surviving zone keeps serving traffic. This is the whole point of multi-AZ — and the ALB must know about at least two zones to provide this.

Next, create a server group and add backend instances:

```bash
# Create a server group
aliyun alb CreateServerGroup \
  --ServerGroupName prod-app-servers \
  --VpcId vpc-bp1xxxxxxxxx \
  --Protocol HTTP \
  --Scheduler Wrr \
  --HealthCheckConfig '{
    "HealthCheckEnabled": true,
    "HealthCheckProtocol": "HTTP",
    "HealthCheckPath": "/health",
    "HealthCheckCodes": ["http_2xx"],
    "HealthCheckInterval": 5,
    "HealthyThreshold": 3,
    "UnhealthyThreshold": 3,
    "HealthCheckTimeout": 3
  }'

# Add backend servers
aliyun alb AddServersToServerGroup \
  --ServerGroupId sgp-xxxxxxxxx \
  --Servers '[
    {"ServerId":"i-bp1aaaaaaa","ServerIp":"10.0.10.5","ServerType":"Ecs","Port":8080,"Weight":100},
    {"ServerId":"i-bp1bbbbbbb","ServerIp":"10.0.11.5","ServerType":"Ecs","Port":8080,"Weight":100}
  ]'
```

The `Scheduler` field controls how traffic is distributed:

- **Wrr** (Weighted Round Robin) — distributes requests proportionally to server weights. Set weight 100 on all servers for equal distribution, or 200 on a bigger instance.
- **Wlc** (Weighted Least Connections) — sends new requests to the server with the fewest active connections, adjusted by weight. Better when request durations vary.
- **Sch** (Source-IP Hash) — same client IP always goes to the same backend. Useful for stateful apps, but defeats the purpose of load balancing if traffic is skewed.

Finally, create a listener:

```bash
# Create an HTTP listener on port 80
aliyun alb CreateListener \
  --ListenerProtocol HTTP \
  --ListenerPort 80 \
  --LoadBalancerId alb-xxxxxxxxx \
  --DefaultActions '[{
    "Type": "ForwardGroup",
    "ForwardGroupConfig": {
      "ServerGroupTuples": [{"ServerGroupId":"sgp-xxxxxxxxx"}]
    }
  }]'
```

For production, you would add an HTTPS listener on port 443 with a TLS certificate, and redirect port 80 to 443. But the HTTP listener above is enough to verify that traffic is flowing.

### Health checks matter more than you think

The health check configuration deserves deliberate thought. With the settings above:

- Every 5 seconds, the ALB sends `GET /health` to each backend
- If a backend returns a non-2xx response 3 times in a row (`UnhealthyThreshold: 3`), it is removed from the pool
- Once removed, it must return 2xx 3 times in a row (`HealthyThreshold: 3`) to be added back
- Each health check times out after 3 seconds

This means a failing backend is removed after roughly 15 seconds (3 checks * 5 second interval). A recovering backend is added back after another 15 seconds. Tune these values based on your application's startup time — if your app takes 30 seconds to warm up, set a higher HealthyThreshold or longer interval.

The `/health` endpoint in your application should check real dependencies. A health check that just returns `200 OK` without verifying database connectivity tells you nothing useful. At minimum, check that the process is running and the main datastore is reachable.

## CEN: Cross-Region Networking

Cloud Enterprise Network connects VPCs across different regions into a single private network. Instead of routing traffic over the public internet between your Beijing and Singapore deployments, CEN provides dedicated Alibaba Cloud backbone links with predictable latency and encryption.

The architecture is hub-and-spoke:

1. **CEN Instance** — the global container. Free to create.
2. **Transit Router (TR)** — regional hub, one per region. This is the thing that actually routes traffic. Costs money.
3. **VPC Attachments** — connect each VPC to its regional Transit Router.
4. **Inter-region connections** — link Transit Routers across regions. Billed by bandwidth.

When you need CEN:

- **Multi-region deployment** — app servers in cn-hangzhou, disaster recovery in cn-shanghai
- **Global presence** — China + Southeast Asia + Europe
- **Shared services** — a central VPC with logging/monitoring that all regional VPCs send data to
- **Hybrid cloud** — connecting on-premises data centers via VPN or Express Connect, then extending to cloud VPCs

When you do NOT need CEN:

- Single-region deployment (all your VSwitches are already in the same VPC)
- Two VPCs in the same region that just need a few ports open (use VPC peering instead — simpler and cheaper)

A basic CEN setup between two regions:

```bash
# Create CEN instance
aliyun cbn CreateCen \
  --Name prod-cen \
  --Description "Cross-region backbone"

# Create Transit Router in cn-hangzhou
aliyun cbn CreateTransitRouter \
  --CenId cen-xxxxxxxxx \
  --RegionId cn-hangzhou \
  --TransitRouterName tr-hangzhou

# Attach VPC to Transit Router
aliyun cbn CreateTransitRouterVpcAttachment \
  --TransitRouterId tr-xxxxxxxxx \
  --VpcId vpc-hangzhou-xxxxxxxxx \
  --ZoneMappings '[
    {"ZoneId":"cn-hangzhou-h","VSwitchId":"vsw-xxxxxxxxx"},
    {"ZoneId":"cn-hangzhou-i","VSwitchId":"vsw-yyyyyyyyy"}
  ]'

# Repeat for cn-shanghai region, then create inter-region connection
aliyun cbn CreateTransitRouterPeerAttachment \
  --CenId cen-xxxxxxxxx \
  --TransitRouterId tr-hangzhou-xxx \
  --PeerTransitRouterId tr-shanghai-xxx \
  --RegionId cn-hangzhou \
  --PeerTransitRouterRegionId cn-shanghai \
  --Bandwidth 100 \
  --BandwidthType DataTransfer
```

CEN pricing is primarily based on inter-region bandwidth. Intra-region traffic through the Transit Router is free (or very cheap). Cross-border traffic (e.g., cn-hangzhou to ap-southeast-1) costs more than domestic traffic. Budget 0.06-0.15 USD/GB for domestic inter-region, 0.08-0.20 USD/GB for cross-border, depending on direction.

## Solution: Multi-AZ Production Network

Let us put everything together. Here is a complete, production-ready network setup for a 3-tier web application across 2 availability zones in cn-hangzhou.

![Complete network topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_network_topology.png)

The target architecture:

- VPC: `10.0.0.0/16`
- 6 VSwitches (public/app/data across 2 AZs)
- 3 security groups (web/app/data with chained rules)
- 1 NAT Gateway with SNAT for private subnets
- 1 ALB for the web tier

### Step 1: Create the VPC

```bash
VPC_ID=$(aliyun vpc CreateVpc \
  --RegionId cn-hangzhou \
  --CidrBlock 10.0.0.0/16 \
  --VpcName prod-vpc \
  --Description "Production 3-tier VPC" \
  --output cols=VpcId --rows | tail -1)

echo "VPC created: $VPC_ID"

# Wait for VPC to become available
aliyun vpc DescribeVpcs \
  --VpcId $VPC_ID \
  --output cols=Status
```

### Step 2: Create all 6 VSwitches

```bash
# Public subnets
VSW_PUB_A=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.1.0/24 --VSwitchName prod-public-a \
  --output cols=VSwitchId --rows | tail -1)

VSW_PUB_B=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.2.0/24 --VSwitchName prod-public-b \
  --output cols=VSwitchId --rows | tail -1)

# App subnets
VSW_APP_A=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.10.0/24 --VSwitchName prod-app-a \
  --output cols=VSwitchId --rows | tail -1)

VSW_APP_B=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.11.0/24 --VSwitchName prod-app-b \
  --output cols=VSwitchId --rows | tail -1)

# Data subnets
VSW_DATA_A=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-h \
  --CidrBlock 10.0.20.0/24 --VSwitchName prod-data-a \
  --output cols=VSwitchId --rows | tail -1)

VSW_DATA_B=$(aliyun vpc CreateVSwitch \
  --VpcId $VPC_ID --ZoneId cn-hangzhou-i \
  --CidrBlock 10.0.21.0/24 --VSwitchName prod-data-b \
  --output cols=VSwitchId --rows | tail -1)
```

### Step 3: Create security groups with chained rules

```bash
# Create the three groups
SG_WEB=$(aliyun ecs CreateSecurityGroup \
  --VpcId $VPC_ID --SecurityGroupName prod-web-sg \
  --output cols=SecurityGroupId --rows | tail -1)

SG_APP=$(aliyun ecs CreateSecurityGroup \
  --VpcId $VPC_ID --SecurityGroupName prod-app-sg \
  --output cols=SecurityGroupId --rows | tail -1)

SG_DATA=$(aliyun ecs CreateSecurityGroup \
  --VpcId $VPC_ID --SecurityGroupName prod-data-sg \
  --output cols=SecurityGroupId --rows | tail -1)

# Web tier rules
aliyun ecs AuthorizeSecurityGroup --SecurityGroupId $SG_WEB \
  --IpProtocol tcp --PortRange 80/80 --SourceCidrIp 0.0.0.0/0 --Priority 1
aliyun ecs AuthorizeSecurityGroup --SecurityGroupId $SG_WEB \
  --IpProtocol tcp --PortRange 443/443 --SourceCidrIp 0.0.0.0/0 --Priority 1

# App tier: only from web SG
aliyun ecs AuthorizeSecurityGroup --SecurityGroupId $SG_APP \
  --IpProtocol tcp --PortRange 8080/8080 --SourceGroupId $SG_WEB --Priority 1

# Data tier: only from app SG
aliyun ecs AuthorizeSecurityGroup --SecurityGroupId $SG_DATA \
  --IpProtocol tcp --PortRange 3306/3306 --SourceGroupId $SG_APP --Priority 1
aliyun ecs AuthorizeSecurityGroup --SecurityGroupId $SG_DATA \
  --IpProtocol tcp --PortRange 6379/6379 --SourceGroupId $SG_APP --Priority 1
```

### Step 4: Set up NAT Gateway for private subnet outbound

```bash
# Create NAT Gateway in public subnet
NAT_ID=$(aliyun vpc CreateNatGateway \
  --RegionId cn-hangzhou --VpcId $VPC_ID \
  --VSwitchId $VSW_PUB_A --NatGatewayName prod-nat \
  --NatType Enhanced --InternetChargeType PayByLcu \
  --output cols=NatGatewayId --rows | tail -1)

# Allocate and bind EIP
EIP_ID=$(aliyun vpc AllocateEipAddress \
  --RegionId cn-hangzhou --Bandwidth 200 \
  --InternetChargeType PayByTraffic \
  --output cols=AllocationId --rows | tail -1)

EIP_IP=$(aliyun vpc DescribeEipAddresses \
  --AllocationId $EIP_ID \
  --output cols=IpAddress --rows | tail -1)

aliyun vpc AssociateEipAddress \
  --AllocationId $EIP_ID --InstanceId $NAT_ID --InstanceType Nat

# Get SNAT table ID
SNAT_TABLE=$(aliyun vpc DescribeNatGateways \
  --NatGatewayId $NAT_ID \
  --output cols=SnatTableIds --rows | tail -1)

# Create SNAT entries for all private subnets
for VSW in $VSW_APP_A $VSW_APP_B $VSW_DATA_A $VSW_DATA_B; do
  aliyun vpc CreateSnatEntry \
    --SnatTableId $SNAT_TABLE \
    --SourceVSwitchId $VSW \
    --SnatIp $EIP_IP
done
```

### Step 5: Create ALB for the web tier

```bash
# Create ALB spanning both AZs
ALB_ID=$(aliyun alb CreateLoadBalancer \
  --LoadBalancerName prod-web-alb \
  --VpcId $VPC_ID \
  --AddressType Internet \
  --LoadBalancerEdition Standard \
  --LoadBalancerBillingConfig '{"PayType":"PostPay"}' \
  --ZoneMappings "[
    {\"ZoneId\":\"cn-hangzhou-h\",\"VSwitchId\":\"$VSW_PUB_A\"},
    {\"ZoneId\":\"cn-hangzhou-i\",\"VSwitchId\":\"$VSW_PUB_B\"}
  ]" \
  --output cols=LoadBalancerId --rows | tail -1)

# Create server group with health checks
SGP_ID=$(aliyun alb CreateServerGroup \
  --ServerGroupName prod-app-backend \
  --VpcId $VPC_ID \
  --Protocol HTTP \
  --Scheduler Wrr \
  --HealthCheckConfig '{
    "HealthCheckEnabled":true,
    "HealthCheckProtocol":"HTTP",
    "HealthCheckPath":"/health",
    "HealthCheckCodes":["http_2xx"],
    "HealthCheckInterval":5,
    "HealthyThreshold":3,
    "UnhealthyThreshold":3,
    "HealthCheckTimeout":3
  }' \
  --output cols=ServerGroupId --rows | tail -1)

# Create HTTP listener
aliyun alb CreateListener \
  --ListenerProtocol HTTP \
  --ListenerPort 80 \
  --LoadBalancerId $ALB_ID \
  --DefaultActions "[{
    \"Type\":\"ForwardGroup\",
    \"ForwardGroupConfig\":{
      \"ServerGroupTuples\":[{\"ServerGroupId\":\"$SGP_ID\"}]
    }
  }]"

echo "ALB DNS: $(aliyun alb GetLoadBalancerAttribute \
  --LoadBalancerId $ALB_ID \
  --output cols=DNSName --rows | tail -1)"
```

### Step 6: Verify connectivity

After launching instances into the subnets (covered in Part 2), verify:

```bash
# From an app-tier instance, test outbound via NAT
ssh bastion "ssh app-server-a 'curl -s ifconfig.me'"
# Should return the NAT EIP address

# From the internet, test ALB
curl -I http://<ALB_DNS_NAME>/health
# Should return HTTP 200

# From app tier, test data tier connectivity
ssh bastion "ssh app-server-a 'mysql -h 10.0.20.5 -u app -p -e \"SELECT 1\"'"
# Should succeed

# From web tier, verify data tier is NOT reachable
ssh bastion "ssh web-server-a 'nc -zv 10.0.20.5 3306'"
# Should fail (connection refused / timeout)
```

That last test is the most important. If the web tier can reach the data tier directly, your security group chain is broken.

## Summary

**CIDR planning is permanent.** You cannot resize a VPC or VSwitch after creation. Over-allocate now. A /16 VPC with /24 VSwitches gives you room for decades of growth. The cost of unused IP space is zero.

**One VSwitch per AZ per tier.** This is the fundamental pattern. It survives zone failures, it maps cleanly to security groups, and it makes CIDR planning obvious.

**Security groups reference security groups, not CIDRs.** Using `SourceGroupId` instead of `SourceCidrIp` means your rules survive IP changes, auto-scaling events, and instance replacements. The chain is always intact.

**NAT Gateway is for outbound. SLB is for inbound.** Do not use DNAT for services that need to handle real traffic — use an ALB or NLB. NAT Gateway's DNAT is for one-off port mappings, not production load balancing.

**ALB over CLB for new HTTP workloads.** CLB still works and will be supported for years, but ALB has better routing, better health check integration, and native gRPC support.

**Test the negative case.** After setting up security groups, verify that disallowed traffic is actually blocked. A security group that allows everything is worse than no security group because it gives false confidence.

## What's Next

The network is the foundation. With VPC, VSwitches, security groups, NAT, and SLB in place, everything else we deploy has a predictable, secure place to land. In the next article, we move up the stack to managed databases — RDS for relational data, Redis for caching, and the replication and backup strategies that keep your data alive when hardware fails.
