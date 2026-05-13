---
title: "阿里云全栈实战（三）：VPC、SLB 构建网络基石"
date: 2026-04-30 09:00:00
tags:
  - Alibaba Cloud
  - VPC
  - SLB
  - Networking
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 3
description: "从零开始构建生产网络：VPC 架构、CIDR 规划、跨可用区 VSwitch、安全组防火墙、SLB 负载均衡、NAT 网关出站流量、EIP 公网访问。"
disableNunjucks: true
translationKey: "aliyun-fullstack-3"
---
我在云上排查过的每一次故障，追根溯源最后都指向了网络。要么是 CIDR 规划没做好，半年后 IP 不够用了；要么是路由缺失，流量在层级间静默丢弃；要么是安全组配置极端——要么对 `0.0.0.0/0` 开放了 22 端口（哈喽，黑客朋友），要么锁得太死导致健康检查失败，负载均衡不停剔除健康实例。网络层是所有部署的前提，必须最先规划和落地；但一旦需要调整，补救成本极高——修改 VPC 的 CIDR 段意味着要重建其下的所有资源。

[第一篇](/zh/aliyun-fullstack/01-ecosystem-map/) 我们梳理了阿里云生态，[第二篇](/zh/aliyun-fullstack/02-ecs-compute/) 学会了启动 ECS 实例——但 ECS 不能裸跑在公网上。本文带你搭建一个生产级多可用区网络：层级隔离的子网、边界清晰的安全组、通过 NAT 网关访问外网的私有子网、SLB 负载均衡公网流量。如果想直接用 Terraform 一键拉起这套网络，参考 [Terraform 实战（三）：VPC 与安全基线](/zh/terraform-agents/03-vpc-and-security-baseline/)。

## 什么是 VPC？

虚拟私有云（VPC）是你在阿里云上独占的网络段，可以理解为一个纯软件定义的私有数据中心网络。你可以指定 IP 地址段、划分子网、配置防火墙规则，并控制哪些实例可访问互联网或仅限内网通信。默认情况下，所有入站和出站流量都被拒绝，只有显式允许的流量才能通过。

![VPC 架构概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_vpc_architecture.png)

如果你熟悉 AWS，阿里云 VPC 的心智模型与其基本一致，功能完全等价，只是术语有所不同。

| Alibaba Cloud | AWS | What it does |
|---|---|---|
| VPC | VPC | 最高级别的隔离网络 |
| VSwitch | Subnet | 绑定到一个可用区的子网 |
| Route Table | Route Table | 用于流量导向的路由规则 |
| Security Group | Security Group | 有状态的实例级防火墙 |
| Network ACL | Network ACL | 无状态的子网级防火墙 |
| EIP | Elastic IP | 静态公网 IP 地址 |
| NAT Gateway | NAT Gateway | 私有子网的互联网访问 |
| SLB (CLB/ALB/NLB) | ELB (CLB/ALB/NLB) | 负载均衡 |

在 VPC 出现之前，阿里云有个“经典网络”，区域内所有实例共享一个扁平网络。经典网络已被弃用，新账号无法创建；如果在旧文档里看到，直接忽略。现在所有业务都跑在 VPC 里。

你需要配置的核心组件包括：

- **VPC** —— 容器。每个区域一个 VPC（你可以建多个）。由 CIDR 块定义。
- **VSwitch** —— VPC 内的子网，绑定到一个可用区。实例实际就住在这里。
- **Route Table** —— 控制流量去向。每个 VPC 都有系统路由表，也可以创建自定义的。
- **Security Group** —— 绑定在实例上的有状态防火墙。规则允许流量；任何未显式允许的入站流量都会被拒绝。
- **Network ACL** —— VSwitch 级别的无状态防火墙，可选。大多数架构会跳过这个，直接用安全组。

一个 VPC 不能跨区域。如果需要跨区域连通，就得用云企业网（CEN），本文后面会讲。

## CIDR 规划：从第一天起就做好

CIDR（无类别域间路由）用于定义 VPC 的 IP 地址空间。如果这一步规划失误，将不得不重建整个 VPC 并迁移所有资源。

![VPC 子网 CIDR 规划指南](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_cidr_planning.png)

记法逻辑是这样：`10.0.0.0/16` 意味着“从 `10.0.0.0` 到 `10.0.255.255` 的所有 IP”。斜杠后面的数字是前缀长度——有多少位是固定的。剩下的位归你分配。

| CIDR | 前缀位 | 可用 IP | 典型用途 |
|---|---|---|---|
| /8 | 8 | 16,777,216 | Enterprise-wide (10.0.0.0/8) |
| /16 | 16 | 65,536 | One VPC |
| /20 | 20 | 4,096 | Large subnet |
| /24 | 24 | 256 (254 usable) | Standard subnet |
| /28 | 28 | 16 (14 usable) | Tiny subnet (NAT, bastion) |

阿里云 VPC 支持使用私有范围 `10.0.0.0/8`、`172.16.0.0/12` 或 `192.168.0.0/16` 内的 /8 到 /24 CIDR 块。VSwitch 支持的最小网段为 /29（共 8 个 IP 地址，其中 6 个可用——阿里云为每个 VSwitch 保留首两个和末一个地址）。

黄金法则是**按当前需求的 10 倍规划**。例如，如果需要 50 个 IP，分配一个 /24（254 个可用）；如果需要 500 个 IP，分配一个 /20（4,094 个可用）。子网创建后不能调整大小，需新建 VSwitch 并迁移实例。

这是我针对 2 个可用区 3 层架构的标准规划表：

| 层级 | AZ-A 子网 | AZ-A CIDR | AZ-B 子网 | AZ-B CIDR | 每个子网可用 IP 数 |
|---|---|---|---|---|---|
| 公网 (web/ALB) | vsw-public-a | 10.0.1.0/24 | vsw-public-b | 10.0.2.0/24 | 251 |
| Private App | vsw-app-a | 10.0.10.0/24 | vsw-app-b | 10.0.11.0/24 | 251 |
| Private Data | vsw-data-a | 10.0.20.0/24 | vsw-data-b | 10.0.21.0/24 | 251 |

VPC 本身用 `10.0.0.0/16`，给我们 65,534 个可用 IP，以后加子网也不怕地址空间不够。我故意在层级范围之间留了空隙（1.x, 10.x, 20.x），这样当你迟早要添加第四层——比如 10.0.30.0/24 的缓存层——它能干净地插进去，不用重新编号。

> **Note:** 如果你的 VPC 将来需要彼此对等（通过 CEN 或 VPN），它们的 CIDR 块不能重叠。规划一个方案，比如 VPC-prod = 10.0.0.0/16, VPC-staging = 10.1.0.0/16, VPC-dev = 10.2.0.0/16。

## VSwitch：跨可用区的子网

VSwitch 即子网，每个 VSwitch 仅隶属于一个可用区，不能拉伸到多个可用区。这是设计使然：单个可用区发生故障时，仅影响该可用区内的 VSwitch 所承载的实例，不会波及其他可用区。

![多可用区 VSwitch 拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_vswitch_layout.png)

典型模式是：每一层、每一个可用区各部署一个 VSwitch。2 个可用区 3 层架构需要 6 个 VSwitch，3 个可用区则需要 9 个。单个应用通常不需要超过 3 个可用区。

首先创建 VPC：

```bash
aliyun vpc CreateVpc \
  --RegionId cn-hangzhou \
  --CidrBlock 10.0.0.0/16 \
  --VpcName prod-vpc \
  --Description "Production VPC for 3-tier web application"
```

记下响应中的 `VpcId`。然后为每个层级和可用区创建 VSwitch：

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

你可以验证布局：

```bash
aliyun vpc DescribeVSwitches \
  --VpcId vpc-bp1xxxxxxxxx \
  --output cols=VSwitchId,VSwitchName,ZoneId,CidrBlock,AvailableIpAddressCount
```

需注意以下几点：

- `ZoneId` 必须匹配你区域里的真实可用区。运行 `aliyun ecs DescribeZones --RegionId cn-hangzhou` 列出可用区。
- 并非所有实例类型在所有可用区均有供应。定可用区布局前先查一下。
- 你可以随时给现有 VPC 添加 VSwitch，只要 CIDR 不和现有 VSwitch 重叠。这也是 CIDR 规划时预留地址间隙的关键原因。

## 路由表

每个 VPC 自带一个系统路由表，删不掉。里面有一条你关心的条目：本地路由，自动启用 VPC 内所有 VSwitch 互相通信。这是隐式的——控制台里你看不到，但 `10.0.1.0/24` 和 `10.0.20.0/24` 之间的流量就是能通。

![路由表决策流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_route_table.png)

系统路由表也处理默认路由（`0.0.0.0/0`），起初它哪儿也不通。要给实例互联网访问权，你得把这条默认路由指向 NAT 网关或者面向互联网的路由器。

大多数配置系统路由表就够了。当你需要不同 VSwitch 有不同的路由行为时，才创建自定义路由表——比如，公共 VSwitch 应该把 `0.0.0.0/0` 路由到互联网网关，但私有 VSwitch 应该路由到 NAT 网关：

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

当 VSwitch 关联到自定义路由表时，该表优先级高于系统路由表。本地路由（VPC 内流量）始终继承，无法覆盖。

需特别注意一个路由匹配机制：阿里云采用最长前缀匹配原则。`10.0.10.0/24` 的路由比 `10.0.0.0/16` 更具体，`10.0.0.0/16` 又比 `0.0.0.0/0` 更具体。流量总是走最具体的匹配路由。

## 安全组深度解析

安全组配置常见两种偏差：一是过度宽松（如全端口开放），二是过度复杂（规则数百条，难以维护）。最佳实践其实是：只建少量安全组，名字要起得望文生义。

安全组是作用于 ECS 实例的有状态防火墙。所谓“有状态”，是指当某条入站规则允许流量（例如 80 端口的 HTTP 请求）进入后，其对应的响应流量（如 TCP ACK 或 HTTP 响应）将自动被允许出站，无需单独配置返回路径。

默认行为是这样的：

- **入站（Inbound）**：拒绝所有（除非你加规则，否则谁也连不上你的实例）
- **出站（Outbound）**：允许所有（你的实例可以访问任何地方，只要路由允许，包括互联网）

该默认配置便于快速上手，但在生产环境中，出站流量完全放行存在安全风险。实例一旦被攻破，unrestricted outbound 会让它轻易窃取数据或者加入僵尸网络。

### 规则构成

每条规则包含以下要素：

- **方向**：入站或出站
- **协议**：TCP、UDP、ICMP、GRE 或 All
- **端口范围**：比如 80/80（单端口）、1/65535（全端口）、443/443
- **源/目的**：CIDR 块（`0.0.0.0/0`）或者另一个安全组 ID
- **优先级**：1（最高）到 100（最低）。数字越小优先级越高。
- **动作**：允许或丢弃

该优先级机制十分灵活：你可在优先级 100 设置一条宽泛的“拒绝所有”规则，再在更高优先级（如 1）添加精确的允许规则。阿里云会按优先级从高到低（1 到 100）评估规则，命中第一条就停止。

### 企业型 vs 基础型安全组

阿里云提供两种类型：

| 特性 | 基础安全组 (Basic) | 企业安全组 (Enterprise) |
|---|---|---|
| 最大规则数 | 200 | 200 |
| 最大实例数 | 2,000 | 65,535 |
| 允许引用其他安全组 | 是 | 否 |
| 成员间默认入站 | 可配置 | 隔离 |
| 性能 | 标准 | 更高吞吐量 |

大多数场景下，基础安全组是更好的选择，因为它支持安全组互引。企业安全组适合大规模部署（成千上万个实例），当你需要更高网络吞吐量且愿意只用 CIDR 规则时才考虑它。

### 三组模式

我自己所有三层架构应用都用这套模式：

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

接下来加规则。Web 层接受来自任意地方的 HTTP/HTTPS，以及来自堡垒机 CIDR 的 SSH：

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

App 层只接受来自 Web 安全组的流量：

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

数据层只接受来自 App 层的连接：

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

这就形成了一条干净的链条：internet → web-sg → app-sg → data-sg。任何一层都无法被非相邻层直接访问。就算 Web 服务器被攻破，攻击者也无法直接查询数据库。

完整的规则集如下表：

| 安全组 | 方向 | 协议 | 端口 | 源/目标 | 优先级 | 描述 |
|---|---|---|---|---|---|---|
| prod-web-sg | Inbound | TCP | 80 | 0.0.0.0/0 | 1 | HTTP |
| prod-web-sg | Inbound | TCP | 443 | 0.0.0.0/0 | 1 | HTTPS |
| prod-web-sg | 入站 | TCP | 22 | 10.0.1.0/24 | 1 | 从堡垒机 SSH 访问 |
| prod-app-sg | 入站 | TCP | 8080 | sg-web-xxx | 1 | 从 web 的应用流量 |
| prod-app-sg | 入站 | TCP | 8081 | sg-web-xxx | 1 | 健康检查 |
| prod-data-sg | 入站 | TCP | 3306 | sg-app-xxx | 1 | 从应用访问 MySQL |
| prod-data-sg | 入站 | TCP | 6379 | sg-app-xxx | 1 | 从应用访问 Redis |

## 弹性公网 IP (EIP)

EIP 是一个你独立拥有的静态公网 IP，不绑定在任何实例上。你可以分配它，绑定到 ECS 实例或 NAT 网关，解绑，再绑定到别处。直到你释放它，这个 IP 才归别人。

什么时候用 EIP：

- **堡垒机 / 跳板机**：需要固定公网 IP 以便 SSH 访问
- **小型部署**（1-2 台实例）：上负载均衡有点杀鸡用牛刀
- **NAT 网关**：需要 EIP 才能访问互联网
- **需要稳定 IP 的服务**：方便合作伙伴做白名单

什么时候别用 EIP：

- **承载真实流量的 Web 应用**：用 SLB 代替（它能处理故障转移）
- **每台实例都绑**：如果你要给 10 台实例绑 EIP，说明你需要负载均衡

计费模式有两种：

| 计费模式 | 工作原理 | 适用场景 |
|---|---|---|
| 按流量计费 (Pay-By-Traffic) | 按传输 GB 数付费 | 波动负载，开发/测试 |
| 按带宽计费 (Pay-By-Bandwidth) | 按预留 Mbps 付费 | 稳定、可预测的流量 |

起步阶段选按流量计费几乎没错。用多少付多少，outbound 每 GB 大概 0.12 美元，直到你遇到持续高带宽之前，这个价格都很合理。

创建并绑定 EIP：

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

这里有个坑：一台 ECS 实例通过主 ENI（弹性网卡）一次只能绑定一个 EIP。如果你需要一台实例多个公网 IP，得创建辅助 ENI 并把 EIP 绑上去。但如果你真需要多个公网 IP，大概率你应该用 SLB 而不是折腾这个。

## NAT 网关：私有子网的出口

私有子网里的实例（比如我们规划里的 App 层和数据层）没有公网 IP，默认上不了网。但它们往往需要上网——拉 Docker 镜像、调用外部 API、下载安全补丁。NAT 网关就是解决这个问题的。

![NAT 网关架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_nat_gateway.png)

NAT 网关部署在公共子网，提供两个功能：

**SNAT (Source NAT)** —— 出站。私有实例把流量发给 NAT 网关，网关把源 IP 换成自己的 EIP 再转发到互联网。响应回到 NAT 网关，再路由回发起请求的私有实例。私有实例始终不需要公网 IP。

**DNAT (Destination NAT)** —— 入站。把 NAT 网关 EIP 上的特定端口映射到特定私有实例和端口。发往 `EIP:8080` 的流量会被转发到 `10.0.10.5:8080`。这对那些需要从互联网访问但不值得上 SLB 的一次性服务很有用。

怎么选：

| 需求 | 解决方案 |
|---|---|
| 私有实例需要访问互联网（拉更新、调 API） | 通过 NAT 网关做 SNAT |
| 某个特定服务需要从外部访问 | DNAT 或 EIP |
| 多个实例提供相同服务且需要被访问 | SLB（别用 NAT） |
| 单台开发/测试实例需要公网 IP | 直接绑 EIP |

创建 NAT 网关：

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

现在创建 SNAT 条目，让私有子网能上网：

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

`SnatIp` 就是你绑给 NAT 网关的那个 EIP 地址。这些 VSwitch 的所有出站流量看起来都来自这个 IP。这对白名单很有用——如果外部 API 要求 whitelist 你的 IP，直接给 NAT 的 EIP 就行。

务必使用 Enhanced NAT 类型。“Normal” 类型是旧版，吞吐量低，不支持多 EIP SNAT，也无法集成新服务。Enhanced NAT 按 LCU (Logical Connection Unit) 计费，随实际用量弹性伸缩。

## SLB：服务器负载均衡器

Server Load Balancer 就是把流量分摊到多台后端实例上。任何想要高可用的服务，它都是必经之门。有了它，你才算从“我有两台服务器”进化到了“我有生产环境部署”。

![SLB 第 4 层与第 7 层对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_slb_comparison.png)

阿里云有三个 SLB 产品，名字一开始容易让人晕：

| 产品 | 全称 | 层级 | 协议 | 适用场景 |
|---|---|---|---|---|
| CLB | Classic Load Balancer | 4 和 7 | TCP/UDP/HTTP/HTTPS | 旧版架构，仍广泛使用 |
| ALB | Application Load Balancer | 7 | HTTP/HTTPS/gRPC | 现代 Web 应用，基于内容路由 |
| NLB | Network Load Balancer | 4 | TCP/UDP/TLS | 高性能 TCP，游戏，IoT |

**CLB** 是元老级的产品。四层（TCP/UDP 转发）和七层（HTTP/HTTPS 带 host/path 路由）都能做。久经沙场，大多数场景都能扛。如果文档或教程里只提“SLB”没加限定词，指的就是 CLB。

**ALB** 是现代化的七层方案。支持基于内容的路由（比如 `/api/*` 走一组后端，`/static/*` 走另一组），支持 gRPC、WebSocket，跟 WAF 和 DCDN 集成得也更好。新的 HTTP/HTTPS 负载，选 ALB 准没错。

**NLB** 专攻纯 TCP/UDP。它不终止连接，直接透传数据包，保留客户端源 IP。数据库代理、游戏服务器，或者任何需要在四层实现最大吞吐、最低延迟的场景，都用它。

典型的 Web 应用，直接上 ALB：

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

这个 ALB 实例跨了两个可用区（AZ）。就算一个区挂了，另一个区的 ALB 还能继续扛流量。这就是多可用区部署的意义——ALB 必须至少感知两个可用区，才能提供这种保障。

接下来建服务器组，把后端实例加进去：

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

`Scheduler` 字段决定流量怎么分：

- **Wrr** (Weighted Round Robin) —— 按权重轮询。权重都一样就是平均分配，大实例给高点权重。
- **Wlc** (Weighted Least Connections) —— 最少连接数加权。请求耗时波动大的场景更合适。
- **Sch** (Source-IP Hash) —— 源 IP 哈希。同一客户端 IP 永远指向同一后端。适合有状态应用，但流量倾斜时会失去负载均衡的意义。

最后，创建监听：

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

生产环境肯定要在 443 端口配 HTTPS 监听并上传 TLS 证书，再把 80 跳转过去。不过上面这个 HTTP 监听足够用来验证流量是否通了。

### 健康检查比你想的更重要

健康检查的配置得仔细琢磨。按上面的设置：

- 每隔 5 秒，ALB 向每个后端发 `GET /health`
- 如果后端连续 3 次返回非 2xx 状态码（`UnhealthyThreshold: 3`），ALB 会把它从池子里踢出去
- 踢出去后，必须连续 3 次返回 2xx（`HealthyThreshold: 3`）才能加回来
- 每次检查超时时间 3 秒

这意味着故障后端大概 15 秒后（3 次检查 * 5 秒间隔）被移除。恢复的后端再过 15 秒加回来。根据应用启动时间调整这些值——如果应用预热要 30 秒，就把 `HealthyThreshold` 调大或者间隔拉长。

应用里的 `/health` 端点得检查真实依赖。那种不验数据库连通性直接返回 `200 OK` 的健康检查没什么用。起码要确认进程活着，主数据存储可达。

## CEN：跨区域网络

Cloud Enterprise Network 把不同地域的 VPC 连成一张私网。北京和新加坡的部署之间，不用走公网，CEN 提供阿里云专用的骨干链路，延迟可控，还带加密。

架构是中心辐射型（Hub-and-spoke）：

1. **CEN Instance** —— 全局容器。创建免费。
2. **Transit Router (TR)** —— 地域枢纽，每个地域一个。真正路由流量的东西。收费。
3. **VPC Attachments** —— 把每个 VPC 连到地域 TR 上。
4. **Inter-region connections** —— 跨地域连接 TR。按带宽计费。

什么时候需要 CEN：

- **多地域部署** —— 杭州跑应用，上海做灾备
- **全球覆盖** —— 中国 + 东南亚 + 欧洲
- **共享服务** —— 中央 VPC 做日志/监控，各地域 VPC 都往这发数据
- **混合云** —— 通过 VPN 或专线连本地 IDC，再扩展到云 VPC

什么时候不需要 CEN：

- 单地域部署（VSwitch 都在同一个 VPC 里）
- 同地域两个 VPC 只需开几个端口（用 VPC 对等连接更简单便宜）

两个地域之间的基础 CEN setup：

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

CEN 计费主要看跨地域带宽。同地域流量经过 Transit Router 免费（或极便宜）。跨境流量（比如 cn-hangzhou 到 ap-southeast-1）比国内流量贵。预算大概按国内跨地域 0.06–0.15 USD/GB，跨境 0.08–0.20 USD/GB 估算，具体看方向。

## 解决方案：多可用区生产网络

咱们把前面的碎片拼起来。这是一套能直接上生产环境的网络架构，跑在 cn-hangzhou 的两个可用区上，典型的三层 Web 应用。

![完整网络拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/03-vpc-networking/03_network_topology.png)

目标架构长这样：

- VPC: `10.0.0.0/16`
- 6 个 VSwitch（public/app/data 分布在 2 个 AZ）
- 3 个安全组（web/app/data 采用链式规则）
- 1 个 NAT Gateway 配合 SNAT 供私有子网出网
- 1 个 ALB 负责 Web 层流量

### 步骤 1：创建 VPC

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

### 步骤 2：创建所有 6 个 VSwitch

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

### 步骤 3：创建带有链式规则的安全组

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

### 步骤 4：为私有子网出站设置 NAT Gateway

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

### 步骤 5：为 Web 层创建 ALB

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

### 步骤 6：验证连接性

实例启动并放入子网后（这部分会在第二部分讲），咱们得验证一下连通性：

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

最后那个测试最关键。要是 Web 层能直连 Data 层，说明你的安全组链式规则失效了。

## 关键要点

**CIDR 规划一旦定稿就无法更改。** VPC 或 VSwitch 创建后没法调整大小。现在就把 IP 空间预留充足。一个 /16 的 VPC 配上 /24 的 VSwitch，足够你折腾几十年。unused IP 不要钱，但不够用会很麻烦。

**每个可用区、每一层对应一个 VSwitch。** 这是基本模式。它能扛住可用区故障，跟安全组映射清晰，CIDR 规划也一目了然。

**安全组引用安全组，别引用 CIDR。** 用 `SourceGroupId` 代替 `SourceCidrIp`，你的规则就能扛住 IP 变动、弹性伸缩和实例替换。这条链永远断不了。

**NAT Gateway 只管出，SLB 只管进。** 别用 DNAT 去扛真实业务流量，那是 ALB 或 NLB 的活儿。NAT Gateway 的 DNAT 只适合做临时的端口映射，不适合生产负载均衡。

**新 HTTP 业务首选 ALB 而不是 CLB。** CLB 还能用，也会支持很多年，但 ALB 的路由能力更强，健康检查集成更好，还原生支持 gRPC。

**一定要测试“不通”的情况。** 安全组配完后，得验证被禁止的流量是不是真被拦住了。一个放行所有的安全组比没有安全组更糟糕，因为它给你虚假的安全感。

## 下一步

网络是地基。VPC、VSwitch、安全组、NAT 和 SLB 就位后，我们部署的任何东西都有了 predictable、secure 的落脚点。下一篇咱们往上走，聊托管数据库——关系型数据用 RDS，缓存用 Redis，还有那些能在硬件故障时保住数据的复制和备份策略。
