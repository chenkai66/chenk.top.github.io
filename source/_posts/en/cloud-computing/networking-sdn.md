---
title: "Cloud Network Architecture and SDN"
date: 2024-09-04 09:00:00
tags:
  - Cloud Computing
  - VPC
  - Cloud Networking
  - Load Balancing
  - SDN
categories: Cloud Computing
series:
  name: "Cloud Computing"
  part: 4
  total: 8
lang: en
mathjax: false
description: "VPC, load balancers, CDN, SDN/NFV, and BGP -- a deep tour of cloud networking from packet to planet, with the production knobs that matter."
---

A cloud platform is, in the end, a network with computers attached. The compute layer scales by adding boxes; the storage layer scales by adding disks; the *network* layer is what makes those boxes and disks behave as a single coherent system. Get the network right and the rest of the stack feels effortless. Get it wrong -- a missing route, a 5-tuple mismatch on a security group, an under-provisioned load balancer -- and the whole platform goes dark.

This article maps the cloud-networking stack from the packet up: how a VPC actually carves an isolated network out of shared infrastructure, what changes when load balancers move from L4 to L7, how a CDN turns geography into latency savings, why SDN reshaped the data centre, and how BGP stitches it all together across regions.

## What you will learn

1. **VPC internals** -- subnets, route tables, gateways, endpoints, and the encapsulation that makes them isolated
2. **Load balancing** -- L4 vs L7, algorithms, health checks, sticky sessions, draining
3. **CDN architecture** -- edge PoPs, cache hierarchies, TLS termination, dynamic acceleration
4. **SDN** -- control / data plane split, OpenFlow / P4Runtime, NFV and service chains
5. **VPN, Direct Connect, and Transit Gateway** -- when to choose which connectivity model
6. **Security** -- security groups vs NACLs, flow logs, zero-trust micro-segmentation
7. **BGP and global routing** -- AS-paths, ECMP, anycast, multi-region failover

## Prerequisites

- IP addressing and CIDR notation
- Familiarity with at least one cloud console (AWS / GCP / Aliyun)
- Parts 1-3 of this series

---

## 1. Virtual Private Cloud (VPC)

A VPC is a software-defined slice of the cloud provider's physical network that *behaves* like your own private datacentre: you choose the IP space, draw the subnets, install the gateways and write the firewall rules. Behind the scenes the provider implements isolation through **VXLAN** (or a proprietary equivalent) -- every packet carries a tenant tag so two customers can both use 10.0.0.0/16 without ever seeing each other's traffic.

![Multi-AZ VPC architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/networking-sdn/fig1_vpc_architecture.png)

### 1.1 Anatomy of a production VPC

The diagram above is a textbook 3-tier deployment. The pieces:

| Component | Layer | What it does | Free? |
|-----------|-------|--------------|-------|
| **VPC** | network | The 10.0.0.0/16 envelope. One per environment per region. | yes |
| **Subnet** | network | A CIDR carved out of the VPC, pinned to one AZ. Public/Private/Isolated by *route table*, not by name. | yes |
| **Internet Gateway (IGW)** | edge | Bidirectional NAT-free path to the public internet. One per VPC. | yes |
| **NAT Gateway** | egress | Lets *private* instances reach the internet outbound; blocks inbound. Per-AZ for HA. | hourly + GB |
| **Route Table** | control | Maps destination CIDR -> target (IGW, NAT, VPCe, peering, TGW). One per subnet. | yes |
| **Security Group** | instance FW | **Stateful**, allow-only, attached to ENIs. | yes |
| **Network ACL** | subnet FW | **Stateless**, allow + deny, ordered. Belt-and-braces with SGs. | yes |
| **VPC Endpoint** | data plane | Private path from VPC to a cloud service (S3, DynamoDB, Secrets Manager, ...). | gateway free / interface hourly |
| **Transit Gateway** | hub | N-to-N VPC + VPN + Direct Connect interconnect. | hourly + GB |

**The "public vs private subnet" distinction is purely about routes.** A subnet whose route table contains `0.0.0.0/0 -> igw-...` is public; a subnet whose default route is `0.0.0.0/0 -> nat-...` is private; a subnet with no internet route at all is isolated. This sounds trivial but is the source of half of all "I cannot reach the internet from my Lambda" tickets.

### 1.2 Terraform: a real, multi-AZ VPC

```hcl
locals {
  azs = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "prod-vpc" }
}

# One public + one private subnet per AZ
resource "aws_subnet" "public" {
  for_each                = toset(local.azs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, index(local.azs, each.key))
  availability_zone       = each.key
  map_public_ip_on_launch = true
  tags = { Name = "public-${each.key}", Tier = "public" }
}

resource "aws_subnet" "private" {
  for_each          = toset(local.azs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, index(local.azs, each.key) + 10)
  availability_zone = each.key
  tags = { Name = "private-${each.key}", Tier = "private" }
}

resource "aws_internet_gateway" "main" { vpc_id = aws_vpc.main.id }

# One NAT GW per AZ for HA (a single NAT GW is a per-AZ SPOF)
resource "aws_eip"         "nat" { for_each = aws_subnet.public }
resource "aws_nat_gateway" "nat" {
  for_each      = aws_subnet.public
  allocation_id = aws_eip.nat[each.key].id
  subnet_id     = each.value.id
}

# Routes
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0"  gateway_id = aws_internet_gateway.main.id }
}

resource "aws_route_table" "private" {
  for_each = aws_subnet.private
  vpc_id   = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[each.key].id
  }
}

# Gateway endpoint for S3 -- avoids NAT GW data-processing fees
resource "aws_vpc_endpoint" "s3" {
  vpc_id          = aws_vpc.main.id
  service_name    = "com.amazonaws.us-east-1.s3"
  route_table_ids = [for rt in aws_route_table.private : rt.id]
}
```

Three production-relevant choices encoded above:

- **One NAT GW per AZ.** A single NAT GW is a per-AZ failure boundary; a region-wide NAT GW is a region-wide outage waiting to happen.
- **Gateway endpoint for S3 / DynamoDB.** Without it, every byte to S3 from a private subnet flows through the NAT GW and bills at $0.045/GB. The endpoint is free.
- **`map_public_ip_on_launch` only on public subnets.** Stops you from accidentally giving a private DB a public IP.

### 1.3 Connecting VPCs: peering, TGW, PrivateLink

| Pattern | Topology | Best for | Caveats |
|---------|----------|----------|---------|
| **VPC peering** | point-to-point | 2-5 VPCs, simple intra-region | non-transitive: A↔B and B↔C does not give A↔C |
| **Transit Gateway** | hub-and-spoke | 5+ VPCs, VPN + DX integration, central inspection | hourly + per-GB charge |
| **PrivateLink** | service exposure | one VPC publishes a service to many consumer VPCs | one-way; no need to peer entire networks |
| **Cloud WAN** | global mesh | multi-region, multi-account meshes | newest, simplest at scale, costs accordingly |

The decision driver is mostly the *topology* you need, not the bandwidth -- all of these can saturate 10s of Gbps when sized correctly.

---

## 2. Load Balancing

A load balancer turns a fleet into a service. It hides individual instance failures, spreads load, terminates TLS, and -- in its L7 form -- lets you reshape traffic by path, host or header without touching the apps.

![L4 vs L7 load balancers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/networking-sdn/fig3_lb_l4_l7.png)

### 2.1 L4 vs L7

| Feature | Network LB (L4) | Application LB (L7) |
|---------|-----------------|---------------------|
| Operates on | TCP / UDP / TLS | HTTP / HTTPS / gRPC |
| Routing decision | 5-tuple (src IP/port, dst IP/port, protocol) | path, host, header, cookie, JWT claim |
| Latency added | tens of µs | low ms |
| Throughput per LB | 100s of Gbps | 10s of Gbps |
| TLS termination | optional (passthrough or terminate) | almost always terminate |
| WebSocket / HTTP/2 / gRPC | passthrough only | first-class |
| Source IP preservation | yes (with proxy protocol) | via `X-Forwarded-For` |
| Best for | Game servers, IoT, MQTT, low-level TCP | Web apps, microservices, public APIs |

A practical pattern in modern stacks: **L4 (NLB) -> L7 (Envoy/ALB) -> services**. The L4 layer absorbs DDoS and gives you a stable anycast IP; the L7 layer does smart routing and authentication. Each layer does one thing well.

### 2.2 Algorithms

```
Round robin           A, B, C, A, B, C, ...
Weighted RR           A(3), B(1)  ->  A, A, A, B, A, A, A, B, ...
Least connections     pick the backend with fewest in-flight requests (best for variable-cost requests)
Least response time   pick the backend with the lowest EWMA of response latency
Power of two choices  pick 2 backends at random; route to the lighter one (90% of "perfect" load balance, O(1) cost)
Consistent hash       hash(client IP / path) -> backend (sticky for cache locality)
Maglev hash           Google's bounded-disruption consistent hash (used in Cloud Load Balancing)
```

For most stateless web traffic, *power-of-two-choices* (P2C) is empirically near-optimal and trivial to implement. For cache-sensitive workloads (anything in front of a key-value store), use a consistent or Maglev hash so the same key keeps hitting the same backend.

### 2.3 ALB with path + host routing (Terraform)

```hcl
resource "aws_lb" "app" {
  name                             = "app-alb"
  load_balancer_type               = "application"
  security_groups                  = [aws_security_group.alb.id]
  subnets                          = [for s in aws_subnet.public : s.id]
  enable_cross_zone_load_balancing = true
  enable_deletion_protection       = true
  idle_timeout                     = 60
  drop_invalid_header_fields       = true     # mitigate request smuggling
}

resource "aws_lb_target_group" "web" {
  name        = "web-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"      # pods/Fargate-friendly

  health_check {
    path                = "/healthz"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200-299"
  }

  deregistration_delay = 30   # match graceful shutdown window
  stickiness {
    type            = "app_cookie"
    cookie_name     = "JSESSIONID"
    cookie_duration = 3600
    enabled         = true
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.main.arn
  default_action { type = "forward"  target_group_arn = aws_lb_target_group.web.arn }
}

# Path-based routing: /api/* to a different target group
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100
  action       { type = "forward"  target_group_arn = aws_lb_target_group.api.arn }
  condition    { path_pattern { values = ["/api/*"] } }
}
```

The non-obvious knobs:

- `deregistration_delay` should match your application's graceful-shutdown window. The default 300 s blocks deploys; setting it lower than your in-flight request timeout drops connections.
- `drop_invalid_header_fields = true` mitigates HTTP request-smuggling attacks (CVE-2019-18860 class).
- `enable_cross_zone_load_balancing` ensures even spread when AZs have unequal target counts. Off by default on NLB, on by default on ALB.

### 2.4 Health checks that do not lie

A health check that hits `/` and runs a database query *does* tell you when the database is down -- but it also marks the entire fleet unhealthy on the first DB blip, taking the whole site offline. The pattern that survives production:

- **`/livez`** -- "the process is alive" (no dependencies). LB checks this.
- **`/readyz`** -- "this instance can serve traffic right now" (cache warmed, DB pool open). Kubernetes checks this.
- Application metrics, traces, and logs check the rest -- not the load balancer.

---

## 3. Content Delivery Networks (CDN)

A CDN is geography turned into a cache. Static content is replicated to edge PoPs near users; the origin (often an S3 bucket) is hit only on a cache miss. Latency drops from 200 ms to under 30 ms; origin egress drops by 10x or more.

![CDN edge distribution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/networking-sdn/fig4_cdn_edge.png)

### 3.1 What actually happens on a request

1. **DNS** -- the user resolves `cdn.example.com` to an anycast IP that lands at the closest healthy PoP.
2. **TLS termination** -- the TLS handshake completes at the edge (much shorter RTT) and the PoP holds a session ticket.
3. **Cache lookup** -- the PoP keys on URL + Vary headers. On HIT, return immediately.
4. **MISS path** -- the PoP fetches from a *parent* (regional cache) which may itself fetch from origin (the hierarchical/tiered cache reduces origin load even for unique content).
5. **Cache fill** -- the PoP stores the response per its TTL (`Cache-Control: max-age`, `s-maxage`, `stale-while-revalidate`).

### 3.2 Cache headers that work

```nginx
# Static, content-addressed assets (hash in filename) -- cache forever
location ~* \.[a-f0-9]{8,}\.(js|css|woff2|png|jpg|svg)$ {
    add_header Cache-Control "public, max-age=31536000, immutable";
}

# API responses -- short TTL, allow stale on origin trouble
location /api/products/ {
    add_header Cache-Control "public, max-age=60, s-maxage=300, stale-while-revalidate=86400";
}

# User-specific responses -- never cache
location /api/me {
    add_header Cache-Control "no-store";
}
```

Three rules of thumb:

- **`immutable`** is the single biggest win on static assets (no revalidation traffic at all).
- **`stale-while-revalidate`** lets the CDN serve a stale copy while it asynchronously refreshes -- origin pain becomes invisible to users.
- **Vary on the bare minimum.** `Vary: Accept-Encoding` is fine; `Vary: User-Agent` blows up the cache key space and your hit ratio.

### 3.3 Beyond static: dynamic acceleration

Modern CDNs also accelerate *uncacheable* traffic. The mechanisms:

- **TLS / TCP termination at the edge** cuts the client-side handshake from `4 × RTT_to_origin` to `4 × RTT_to_edge`.
- **Pre-warmed long-haul connections** between PoPs and origin reuse keepalive (no re-handshake per request).
- **Anycast DNS + BGP optimisation** routes the client to the lowest-latency PoP, not just the closest by mileage.

The combination -- CloudFront's "CloudFront Functions", Aliyun DCDN, Cloudflare Workers -- moves the *line* between "static" and "dynamic" further into the application than most teams realise.

---

## 4. Software-Defined Networking (SDN)

Traditional networks bound the *control* (routing, ACLs, QoS) tightly to each device. SDN separates the **control plane** (decisions, programmable from above) from the **data plane** (line-rate forwarding). The result is a network that you can manage like software: declarative, version-controlled, observable.

![SDN control plane vs data plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/networking-sdn/fig2_sdn_planes.png)

### 4.1 The split

| Plane | Responsibility | Implementation |
|-------|---------------|----------------|
| **Control plane** | Build the topology, compute paths, enforce policy, expose APIs | Logically centralised controller (ONOS, OpenDaylight, hyperscaler proprietary) |
| **Data plane** | Match -> action on every packet at line rate | ASIC switches with a flow table, or eBPF/XDP/DPDK in software |
| **Southbound API** | Controller pushes flow rules to switches | OpenFlow 1.3+, NETCONF/YANG, gNMI, P4Runtime |
| **Northbound API** | Apps program the controller | REST / gRPC, GraphQL |

Inside a hyperscaler's region, **everything** is SDN: Hyper-V Virtual Switch, Open vSwitch, Cisco ACI, AWS's "Mapping Service" + "Hyperplane" implement what you experience as a VPC. Your security-group rules are compiled into ACL entries pushed to the host's vswitch on the path your packet takes -- not enforced at the destination instance.

### 4.2 Why operators love it

- **Centralised intent.** "All traffic from PCI-tagged subnets must pass through inspection" is one policy, applied everywhere, instead of 200 device configs.
- **Traffic engineering.** A controller seeing the whole topology can route around congestion that distributed routing protocols (OSPF, IS-IS) react to with a 30-second delay.
- **Programmable packet processing.** P4 lets the data plane parse new protocols (e.g. SRv6, custom in-band telemetry) without an ASIC respin.
- **Fast failure recovery.** SDN-precomputed alternate paths + BFD failure detection reduces MTTR to milliseconds.

### 4.3 Network Functions Virtualisation (NFV)

NFV is SDN's sibling: *replace dedicated network appliances with software running on commodity x86 servers*. A firewall, a load balancer, a WAN optimiser -- each becomes a VNF that you can spin up, scale, and chain like any other workload.

```
Traditional:   [Router HW] -> [Firewall HW] -> [LB HW] -> [WAN-opt HW]
NFV:           x86 server: [Router VNF] -> [Firewall VNF] -> [LB VNF] -> [WAN-opt VNF]
```

A representative VNF -- HAProxy as the load balancer in a service chain:

```
frontend public_https
    bind *:443 ssl crt /etc/haproxy/star.example.com.pem alpn h2,http/1.1
    http-request set-header X-Forwarded-Proto https
    default_backend api_pool

backend api_pool
    balance               leastconn
    option                httpchk GET /healthz HTTP/1.1\r\nHost:\ api.example.com
    http-check expect     status 200
    default-server        check inter 2s fall 3 rise 2 maxconn 256
    server api1 10.0.10.11:8080
    server api2 10.0.11.11:8080
    server api3 10.0.12.11:8080
```

**Service chaining** (sometimes called *service function chaining*, SFC) is the topology by which traffic walks through the VNFs in order. In a Kubernetes mesh, this is what an Envoy sidecar + mesh policy does for east-west traffic.

---

## 5. Hybrid connectivity: VPN, Direct Connect, Transit Gateway

Most enterprises live in hybrid topologies: cloud workloads need to reach on-prem databases, identity providers, or partner networks. There are three layers of connectivity tooling.

### 5.1 VPN vs Direct Connect

| Feature | Site-to-site VPN | Direct Connect / Express Connect / Cloud Interconnect |
|---------|------------------|--------------------------------------------------------|
| Path | Encrypted tunnel over the public internet | Dedicated physical fibre to a provider POP |
| Bandwidth | up to ~10 Gbit/s, jitter-prone | 1 / 10 / 100 Gbit/s, deterministic |
| Latency | variable (tens of ms + jitter) | low, single-digit ms within metro, stable |
| Setup time | minutes | weeks (carrier provisioning) |
| Cost | hourly + per-GB egress | port hourly + per-GB egress (cheaper at scale) |
| Encryption | IPsec mandatory | optional MACsec; usually you run IPsec on top for confidentiality |
| Best for | low / variable volume, dev environments | production data flows, low-latency sync, regulatory needs |

A common layered approach: **DX as the primary** with a **VPN as the encrypted backup** that activates on DX failure. Both terminate on the same Virtual Private Gateway / Transit Gateway.

### 5.2 Transit Gateway

A Transit Gateway (TGW) is the regional hub: every VPC, every VPN, every DX connection attaches once and gains reachability to everything else, governed by *route tables on the TGW itself*. It replaces the N²-peering problem with N attachments.

```hcl
resource "aws_ec2_transit_gateway" "main" {
  description                     = "prod-tgw"
  default_route_table_association = "disable"
  default_route_table_propagation = "disable"
}

resource "aws_ec2_transit_gateway_vpc_attachment" "prod" {
  transit_gateway_id = aws_ec2_transit_gateway.main.id
  vpc_id             = aws_vpc.prod.id
  subnet_ids         = [for s in aws_subnet.tgw_private : s.id]   # one /28 per AZ
}

# Two route tables: prod and shared services
resource "aws_ec2_transit_gateway_route_table" "prod"   { transit_gateway_id = aws_ec2_transit_gateway.main.id }
resource "aws_ec2_transit_gateway_route_table" "shared" { transit_gateway_id = aws_ec2_transit_gateway.main.id }

# Prod can reach shared, shared can reach prod, prod cannot reach prod
resource "aws_ec2_transit_gateway_route" "prod_to_shared" {
  destination_cidr_block         = "10.10.0.0/16"
  transit_gateway_route_table_id = aws_ec2_transit_gateway_route_table.prod.id
  transit_gateway_attachment_id  = aws_ec2_transit_gateway_vpc_attachment.shared.id
}
```

The pattern -- **disable default association/propagation, then write the routes you want** -- is how you turn a TGW from a flat-mesh footgun into an enforceable segmentation boundary.

---

## 6. Security in the network

### 6.1 Security Groups vs Network ACLs

| Feature | Security Group | Network ACL |
|---------|---------------|-------------|
| Attached to | ENI / instance | Subnet |
| State | **Stateful** (return traffic auto-allowed) | **Stateless** (must allow both directions) |
| Rules | Allow only | Allow + Deny |
| Default | Deny all inbound, allow all outbound | Allow all both ways |
| Rule evaluation | All rules union | First match wins (numeric order) |
| Quota (AWS) | 60 inbound + 60 outbound, 5 SGs per ENI | 20 inbound + 20 outbound rules per NACL |
| Best for | App-to-app authorisation | Blast-radius limits, deny-listing IPs |

The robust pattern is **SG-to-SG references**: instead of opening port 3306 from a CIDR, open port 3306 from `aws_security_group.app.id`. The rule remains correct as the app fleet scales and IPs churn; auditors see *intent*, not infrastructure.

```hcl
resource "aws_security_group" "db" {
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]   # not a CIDR
    description     = "MySQL from app tier"
  }
  egress {
    from_port = 0  to_port = 0  protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### 6.2 VPC Flow Logs

Flow Logs record every accepted/rejected 5-tuple to S3 or CloudWatch. Three queries you will run within a week of enabling them:

```sql
-- 1. Top REJECTed source IPs in the last hour (probably scanners / misconfigured clients)
SELECT srcaddr, COUNT(*) c
FROM "vpc_flow_logs"
WHERE action = 'REJECT' AND start_time > now() - interval '1' hour
GROUP BY srcaddr ORDER BY c DESC LIMIT 20;

-- 2. Egress to unexpected destinations (data exfiltration hunting)
SELECT dstaddr, dstport, SUM(bytes) bytes_out
FROM "vpc_flow_logs"
WHERE direction = 'egress' AND dstaddr NOT LIKE '10.%'
GROUP BY dstaddr, dstport ORDER BY bytes_out DESC LIMIT 20;

-- 3. Cross-AZ traffic (often unintentional, always billed)
SELECT srcsubnet, dstsubnet, SUM(bytes) bytes
FROM "vpc_flow_logs"
WHERE az_id_src != az_id_dst
GROUP BY srcsubnet, dstsubnet ORDER BY bytes DESC LIMIT 20;
```

### 6.3 Zero trust at the network layer

The "soft chewy interior" model -- a hard perimeter, trusted internal network -- is gone. Modern designs assume the network is hostile and enforce identity per request. Practical building blocks:

- **mTLS everywhere** (service-to-service via a mesh: Istio, Linkerd, App Mesh).
- **Short-lived workload identity** (SPIFFE/SPIRE, IAM Roles for Service Accounts, GCP Workload Identity).
- **Per-flow policy** evaluated by the SDN (security groups + Cilium NetworkPolicies + service-mesh L7 RBAC layered together).

---

## 7. BGP and global routing

When a request leaves your region and crosses the public internet -- or when you fail over from one region to another -- the system that gets it there is **BGP**, the Border Gateway Protocol. BGP is the routing protocol *between* autonomous systems (ASes); within a region your provider runs OSPF or IS-IS, but the moment you step out, BGP is in charge.

![BGP across multiple regions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/networking-sdn/fig5_bgp_multi_region.png)

### 7.1 The route-selection algorithm (simplified)

Given multiple paths to the same prefix, BGP picks one by walking down a long tiebreak list. The first five rungs cover 99% of cases:

1. **Highest LOCAL_PREF** -- "internal preference"; how *we* want to leave our network. Operator-set.
2. **Shortest AS_PATH** -- fewer ASes to traverse.
3. **Lowest MED** -- "I'd prefer you enter my AS this way"; honoured between peers, not always between providers.
4. **eBGP > iBGP** -- prefer routes learned from external peers over internal.
5. **Lowest IGP cost to the next hop** -- shortest *internal* path to the chosen exit.

### 7.2 ECMP and anycast

- **ECMP (Equal-Cost Multi-Path)** -- when several paths tie on all the above, hash the 5-tuple of each flow and spread them across the equal-cost neighbours. ECMP is how a 100 Gbit/s logical link is built from ten 10 Gbit/s physical links, and how a multi-region anycast service spreads load across its PoPs.
- **Anycast** -- announce the *same* IP from many locations; BGP delivers each user to the topologically closest one. Anycast is the foundation of every CDN, every DNS root, every public DoH resolver.

### 7.3 Multi-region failover

A common pattern for global services:

- Primary region serves writes; secondary regions serve reads.
- The DNS name is anycast or a Route 53 / Aliyun GTM **failover record set** keyed off health checks.
- Database replication is async cross-region; on failover the standby is promoted (RPO of seconds to minutes).
- On recovery, traffic is *manually* shifted back -- automatic flap-back has burned almost everyone who has tried it.

The BGP details rarely intrude on the application -- but when something is mis-announced upstream (the [Facebook 2021 outage](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) or any of the hundreds of route leaks per year) every layer above goes dark. RPKI route-origin validation and a careful prefix list at the ingress edge are mandatory hygiene at any meaningful scale.

---

## 8. Troubleshooting in production

### 8.1 The triage order

When traffic is broken, walk *up* from the wire:

1. **Reachability** -- can the source ARP/ND the gateway? `ip neigh`, `arping`. If not, it is L2 / SG / NACL.
2. **Routing** -- does the route table have a path? `aws ec2 describe-route-tables`, `ip route`.
3. **Filters** -- is the SG / NACL allowing the 5-tuple? Check both directions if NACL is involved.
4. **DNS** -- is the name resolving to the IP you think? `dig +short`, beware of split-horizon DNS.
5. **TLS** -- `openssl s_client -connect host:443 -servername host`. Half of "API down" is a stale cert.
6. **Application** -- `curl -vvv https://...`. Log shows `200 OK`? It's not the network.

### 8.2 The Swiss-army CLI

```bash
# Where are packets actually going?
mtr -rwn -c 100 example.com           # combine traceroute + ping with stats
ss -tunap                              # what is listening / connected and to whom
ss -i                                  # per-socket congestion window, RTT
tcpdump -ni any -s 96 'port 443 and tcp[tcpflags] & (tcp-syn|tcp-fin) != 0'   # SYN/FIN summary
nft list ruleset                       # current netfilter rules (modern iptables)

# DNS
dig +trace example.com
dig @8.8.8.8 example.com               # bypass local resolver to isolate split-horizon

# Path MTU
ip -6 route get 2606:4700::6810:84e5    # confirm next hop + MTU
ping -M do -s 1472 8.8.8.8              # detect black-holed PMTUD
```

### 8.3 The five most expensive misconfigurations

| Symptom | Usual cause | Fix |
|---------|-------------|-----|
| Lambda in VPC cannot reach the internet | Default route missing or NAT GW down | `0.0.0.0/0 -> nat-...` on the function's subnet |
| Sudden NAT GW bill spike | App pulled large object from S3 over NAT instead of via Gateway Endpoint | Add `aws_vpc_endpoint.s3` |
| Random TCP resets after seconds | SG / NACL allows new flows but stateful tracking dropped (idle timeout) | Tune keepalive, increase `tcp-keepalive-time` |
| Cross-AZ data transfer cost dwarfs compute | LB cross-zone off, or services not AZ-aware | Enable cross-zone LB; topology-aware routing in K8s |
| Asymmetric routing on multi-homed VPC | SG stateful tracking sees only one direction | Route both directions through the same TGW attachment |

---

## 9. Common questions

**Q: Should we run our own load balancer (HAProxy / NGINX) or use the cloud LB?**

Use the cloud LB unless you have a specific reason not to (custom routing logic, sticky on a header the cloud LB does not support, tight latency budget). Cloud LBs are HA, auto-scaling, integrated with TLS / WAF / IAM; you get back the engineer who would have run HAProxy.

**Q: NLB or ALB?**

ALB for HTTP/gRPC services (path/host routing, OIDC auth, redirects). NLB for non-HTTP TCP/UDP, for static-IP requirements (each NLB has a per-AZ EIP), or when you need extreme PPS.

**Q: How many subnets per VPC?**

At least one public + one private + one DB subnet per AZ. A 3-AZ VPC therefore has 9 subnets. More if you separate by tier (web/app/db/mgmt) or by sensitivity (PCI/non-PCI). Keep `/16` per VPC and `/22`-`/24` per subnet so you have room to grow.

**Q: When does my workload need its own SDN controller (vs the cloud's)?**

Almost never on the cloud -- you are renting the controller and you cannot replace it. On-prem with OpenStack or VMware NSX, yes. Edge / 5G operators who build their own DC fabric, yes. Most application teams should not even think about SDN below the VPC abstraction.

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Architecture](/en/cloud-computing-fundamentals/) |
| 2 | [Virtualization Technology Deep Dive](/en/cloud-computing-virtualization/) |
| 3 | [Storage Systems and Distributed Architecture](/en/cloud-computing-storage-systems/) |
| **4** | **Network Architecture and SDN (you are here)** |
| 5 | [Security and Privacy Protection](/en/cloud-computing-security-privacy/) |
| 6 | [Operations and DevOps Practices](/en/cloud-computing-operations-devops/) |
| 7 | [Cloud-Native and Container Technologies](/en/cloud-computing-cloud-native-containers/) |
| 8 | [Multi-Cloud and Hybrid Architecture](/en/cloud-computing-multi-cloud-hybrid/) |
