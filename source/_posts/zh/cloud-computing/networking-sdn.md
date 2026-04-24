---
title: "云网络架构与 SDN"
date: 2024-09-04 09:00:00
tags:
  - 云计算
  - VPC
  - 云网络
  - 负载均衡
  - SDN
categories: 云计算
series:
  name: "云计算"
  part: 4
  total: 8
lang: zh-CN
mathjax: false
description: "VPC、负载均衡、CDN、SDN/NFV、BGP —— 从一个数据包到全球互联，深入云网络的关键旋钮。"
---

云平台说到底，就是「一张网络 + 接在网络上的若干计算」。计算靠加节点扩，存储靠加盘扩，*网络* 才是让这些节点和盘表现得像一个统一系统的那一层。网络做对了，整个栈让人觉得轻盈；网络做错了 —— 一条少加的路由、安全组的 5 元组对不上、负载均衡规格不够 —— 整个平台直接黑屏。

本文从一个数据包开始，自下而上把云网络栈走一遍：VPC 是怎么把共享物理网络切出一块隔离网络的、L4 升到 L7 时究竟变了什么、CDN 怎么把地理变成延迟节省、SDN 为什么重塑了数据中心、BGP 又怎么把这一切跨地域缝在一起。

## 你将学到

1. **VPC 的内部** —— 子网、路由表、网关、端点，以及让它们彼此隔离的封装技术
2. **负载均衡** —— L4 vs L7、算法、健康检查、会话保持、连接排空
3. **CDN 架构** —— 边缘 PoP、缓存层级、TLS 终结、动态加速
4. **SDN** —— 控制 / 数据平面分离、OpenFlow / P4Runtime、NFV 与服务链
5. **VPN、专线、TGW** —— 三种连接模型如何选
6. **网络层安全** —— 安全组 vs NACL、流日志、零信任微隔离
7. **BGP 与全球路由** —— AS-Path、ECMP、Anycast、跨地域故障切换

## 前置知识

- IP 寻址与 CIDR
- 至少熟悉一个云控制台（AWS / GCP / 阿里云）
- 本系列前 3 篇

---

## 1. 虚拟私有云（VPC）

VPC 是云厂商在物理网络上为你切出的一块软件定义切片，从外面看像你自己的私有数据中心：你定义 IP 空间、画子网、装网关、写防火墙规则。背后云厂商靠 **VXLAN**（或自研对等技术）实现隔离 —— 每个数据包都带租户标签，所以两个客户都能用 10.0.0.0/16 而互不可见。

![多 AZ VPC 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig1_vpc_architecture.png)

### 1.1 一个生产级 VPC 的解剖

上图就是一个标准的三层部署。各部件：

| 组件 | 层级 | 作用 | 是否免费 |
|------|------|------|---------|
| **VPC** | 网络 | 10.0.0.0/16 这个壳。每环境每地域一个。 | 是 |
| **子网** | 网络 | 从 VPC 切出的 CIDR，绑定到一个 AZ。公有/私有/隔离由 *路由表* 决定，不是名字。 | 是 |
| **Internet 网关 IGW** | 边缘 | 双向、无 NAT 的公网通道。每个 VPC 一个。 | 是 |
| **NAT 网关** | 出口 | 让 *私有* 实例能出网；阻止入站。生产按 AZ 部署。 | 按小时 + GB |
| **路由表** | 控制 | CIDR -> 目标（IGW、NAT、VPCe、对等、TGW）。每子网一张。 | 是 |
| **安全组** | 实例防火墙 | **有状态**、仅允许、绑在 ENI 上 | 是 |
| **网络 ACL** | 子网防火墙 | **无状态**、允许 + 拒绝、有序 | 是 |
| **VPC 端点** | 数据面 | VPC 私有访问云服务（S3、DynamoDB、Secrets Manager…） | 网关型免费 / 接口型按小时 |
| **Transit Gateway** | 中枢 | N 对 N 的 VPC + VPN + 专线互联 | 按小时 + GB |

**「公有 vs 私有子网」的本质纯粹是路由。** 路由表里有 `0.0.0.0/0 -> igw-...` 的就是公有；默认路由是 `0.0.0.0/0 -> nat-...` 的就是私有；完全没有出网路由的就是隔离。听起来 trivial，但「我的 Lambda 上不了网」工单一半源于此。

### 1.2 Terraform：一个真正的多 AZ VPC

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

# 每个 AZ 一个公有 + 一个私有子网
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

# 每 AZ 一个 NAT GW 才是 HA（单 NAT GW 是 AZ 级单点）
resource "aws_eip"         "nat" { for_each = aws_subnet.public }
resource "aws_nat_gateway" "nat" {
  for_each      = aws_subnet.public
  allocation_id = aws_eip.nat[each.key].id
  subnet_id     = each.value.id
}

# 路由
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

# S3 网关端点 —— 避免 NAT GW 的数据处理费
resource "aws_vpc_endpoint" "s3" {
  vpc_id          = aws_vpc.main.id
  service_name    = "com.amazonaws.us-east-1.s3"
  route_table_ids = [for rt in aws_route_table.private : rt.id]
}
```

上面隐含了三个生产相关的选择：

- **每 AZ 一个 NAT GW。** 单 NAT GW 是 AZ 级故障域；全地域共用一个就是等着挨故障。
- **S3 / DynamoDB 用网关端点。** 不开端点的话，私有子网到 S3 的每字节都要走 NAT GW，按 $0.045/GB 计费。端点免费。
- **`map_public_ip_on_launch` 只在公有子网开。** 防止你一不小心给私有数据库分了公网 IP。

### 1.3 VPC 互联：Peering、TGW、PrivateLink

| 模式 | 拓扑 | 适用 | 注意 |
|------|------|------|------|
| **VPC 对等** | 点对点 | 2-5 个 VPC、地域内简单互联 | 不传递：A↔B、B↔C 不等于 A↔C |
| **Transit Gateway** | 星型 | 5+ VPC、VPN + 专线集中、统一审计 | 按小时 + GB |
| **PrivateLink** | 服务暴露 | 一个 VPC 把服务暴露给多个消费方 VPC | 单向；不需要把整个网络对等 |
| **Cloud WAN** | 全球网格 | 多地域、多账号网格 | 最新、规模化最简单，价格也相应 |

决策驱动通常是 *拓扑* 而不是带宽 —— 几种方案规格够大都能跑满几十 Gbps。

---

## 2. 负载均衡

负载均衡器把一组实例变成一个服务。它隐藏单实例故障、分摊负载、终结 TLS；在 L7 形态下，还能基于路径、Host、Header 重塑流量而不动应用本身。

![L4 vs L7 负载均衡器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig3_lb_l4_l7.png)

### 2.1 L4 vs L7

| 特性 | 网络 LB（L4） | 应用 LB（L7） |
|------|--------------|--------------|
| 工作于 | TCP / UDP / TLS | HTTP / HTTPS / gRPC |
| 路由依据 | 5 元组（源 IP/端口、目标 IP/端口、协议） | 路径、Host、Header、Cookie、JWT 字段 |
| 引入延迟 | 几十 µs | 低 ms |
| 单 LB 吞吐 | 几百 Gbps | 几十 Gbps |
| TLS 终结 | 可选（直通或终结） | 几乎都终结 |
| WebSocket / HTTP/2 / gRPC | 仅直通 | 一等公民 |
| 源 IP 保留 | 是（配 Proxy Protocol） | 通过 `X-Forwarded-For` |
| 适用 | 游戏、IoT、MQTT、低层 TCP | Web 应用、微服务、公开 API |

现代栈里常见的实战组合：**L4（NLB） -> L7（Envoy/ALB） -> 服务**。L4 层吃 DDoS、提供稳定的 Anycast IP；L7 层做智能路由和鉴权。每层只做一件事。

### 2.2 算法

```
轮询              A, B, C, A, B, C, ...
加权轮询          A(3), B(1)  ->  A, A, A, B, A, A, A, B, ...
最少连接          挑当前未完成请求最少的后端（适合开销不均的请求）
最短响应时间      挑响应延迟 EWMA 最低的后端
两选一（P2C）     随机抽 2 个后端，挑负载更轻的（90% 的「完美均衡」效果，O(1) 开销）
一致性哈希        hash(客户端 IP / 路径) -> 后端（缓存亲和性）
Maglev 哈希       Google 的有界扰动一致性哈希（Cloud LB 在用）
```

无状态 Web 流量一般用 *两选一（P2C）* 就接近最优、实现简单。缓存敏感（任何挂在 KV 前的服务）用一致性哈希或 Maglev，让同一 Key 反复打到同一后端。

### 2.3 ALB 路径 + 主机路由（Terraform）

```hcl
resource "aws_lb" "app" {
  name                             = "app-alb"
  load_balancer_type               = "application"
  security_groups                  = [aws_security_group.alb.id]
  subnets                          = [for s in aws_subnet.public : s.id]
  enable_cross_zone_load_balancing = true
  enable_deletion_protection       = true
  idle_timeout                     = 60
  drop_invalid_header_fields       = true     # 缓解请求走私
}

resource "aws_lb_target_group" "web" {
  name        = "web-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"      # 兼容 Pod / Fargate

  health_check {
    path                = "/healthz"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200-299"
  }

  deregistration_delay = 30   # 与优雅关闭窗口对齐
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

# 路径路由：/api/* 转到另一 TG
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100
  action       { type = "forward"  target_group_arn = aws_lb_target_group.api.arn }
  condition    { path_pattern { values = ["/api/*"] } }
}
```

不那么明显但很关键的旋钮：

- `deregistration_delay` 必须与应用的优雅关闭窗口对齐。默认 300 s 会让发布卡住；设得短于在途请求超时会断连。
- `drop_invalid_header_fields = true` 缓解 HTTP 请求走私（CVE-2019-18860 这一类）。
- `enable_cross_zone_load_balancing` 在 AZ 后端数量不均时保证均匀。NLB 默认关、ALB 默认开。

### 2.4 不会撒谎的健康检查

打 `/` 同时跑数据库查询 *的确* 能告诉你数据库挂了 —— 但数据库一抖，整个 fleet 就被标 unhealthy，整站下线。能扛生产的模式：

- **`/livez`** —— 「进程还活着」（不查依赖）。LB 打这个。
- **`/readyz`** —— 「现在能服务流量」（缓存预热完、DB 连接池打开）。Kubernetes 打这个。
- 应用的指标、追踪、日志负责其余 —— 不是负载均衡器。

---

## 3. 内容分发网络（CDN）

CDN 是「把地理变成缓存」。静态内容复制到接近用户的边缘 PoP；只有缓存未命中才回源（通常是 S3）。延迟从 200 ms 降到 30 ms 以下；源站出网带宽降 10 倍以上。

![CDN 边缘分布](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig4_cdn_edge.png)

### 3.1 一次请求实际发生了什么

1. **DNS** —— 用户解析 `cdn.example.com` 到 Anycast IP，落到最近的健康 PoP。
2. **TLS 终结** —— TLS 握手在边缘完成（RTT 短得多），PoP 拿到 session ticket。
3. **缓存查找** —— PoP 用 URL + Vary Header 作为 Key。命中就立即返回。
4. **未命中路径** —— PoP 向 *父级*（地域级缓存）请求；父级再视情况回源。这种分级（tiered）缓存即使是冷内容也能减少源站压力。
5. **缓存填充** —— PoP 按 TTL 缓存（`Cache-Control: max-age`、`s-maxage`、`stale-while-revalidate`）。

### 3.2 真正能用的缓存头

```nginx
# 内容寻址的静态资产（文件名带 hash）—— 永久缓存
location ~* \.[a-f0-9]{8,}\.(js|css|woff2|png|jpg|svg)$ {
    add_header Cache-Control "public, max-age=31536000, immutable";
}

# API 响应 —— 短 TTL，源站抖动时允许返回旧内容
location /api/products/ {
    add_header Cache-Control "public, max-age=60, s-maxage=300, stale-while-revalidate=86400";
}

# 用户特定响应 —— 永不缓存
location /api/me {
    add_header Cache-Control "no-store";
}
```

三条拇指法则：

- **`immutable`** 是静态资源最大的单点优化（连重新校验流量都没了）。
- **`stale-while-revalidate`** 让 CDN 用旧内容应付，同时异步刷 —— 源站抖动对用户不可见。
- **Vary 越窄越好。** `Vary: Accept-Encoding` 没问题；`Vary: User-Agent` 把 Key 空间炸成天文数字，命中率跌到地板。

### 3.3 不止静态：动态加速

现代 CDN 也加速 *不可缓存* 的流量，机制是：

- **TLS / TCP 在边缘终结**，把客户端握手从 `4 × RTT_to_origin` 砍到 `4 × RTT_to_edge`。
- **PoP 与源站之间预热的长连接** 复用 keepalive（每请求都不再握手）。
- **Anycast DNS + BGP 优化** 把客户端送到延迟最低的 PoP，而不只是地图上最近的。

CloudFront Functions、阿里云 DCDN、Cloudflare Workers 等组合，把「静态 vs 动态」的分界线推得比多数团队意识到的更靠近应用。

---

## 4. 软件定义网络（SDN）

传统网络把 *控制*（路由、ACL、QoS）紧紧绑在每台设备上。SDN 把 **控制平面**（决策、可由上层编程）和 **数据平面**（线速转发）分离，结果是一张可以像软件一样管理的网络：声明式、版本化、可观测。

![SDN 控制平面 vs 数据平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig2_sdn_planes.png)

### 4.1 二分

| 平面 | 职责 | 实现 |
|------|------|------|
| **控制平面** | 构建拓扑、计算路径、执行策略、对外提供 API | 逻辑集中的控制器（ONOS、OpenDaylight、超大规模厂商自研） |
| **数据平面** | 在每个数据包上线速「匹配 -> 动作」 | 带流表的 ASIC 交换机；或软件方案（eBPF/XDP/DPDK） |
| **南向接口** | 控制器把流表下发到交换机 | OpenFlow 1.3+、NETCONF/YANG、gNMI、P4Runtime |
| **北向接口** | 应用编程控制器 | REST / gRPC、GraphQL |

在超大规模云厂商的地域内，**一切都是 SDN**：Hyper-V Virtual Switch、Open vSwitch、Cisco ACI、AWS 的 "Mapping Service" + "Hyperplane" 共同实现了你所体验到的 VPC。你写的安全组规则会被编译成 ACL 表项，下发到数据包要走的宿主机 vSwitch 上 —— 而不是在目的实例处再执行。

### 4.2 运维侧为什么爱它

- **意图集中。** 「PCI 标签的子网流量必须经过审计」是一条策略，到处生效，而不是 200 份设备配置。
- **流量工程。** 看到全局拓扑的控制器能实时绕开拥塞；分布式路由协议（OSPF、IS-IS）要 30 秒才反应过来。
- **可编程数据面。** P4 让数据平面解析新协议（SRv6、自定义带内遥测），不用换 ASIC。
- **快速故障收敛。** SDN 预先计算好的备份路径 + BFD 检测把 MTTR 压到毫秒级。

### 4.3 网络功能虚拟化（NFV）

NFV 是 SDN 的姊妹概念：*用商用 x86 服务器上的软件，替代专用网络硬件设备*。一台防火墙、一台负载均衡、一台 WAN 加速器，每一个都变成可弹性拉起、按需扩缩、可串成链的 VNF。

```
传统：    [路由器硬件] -> [防火墙硬件] -> [LB 硬件] -> [WAN-opt 硬件]
NFV：    x86 服务器：[路由器 VNF] -> [防火墙 VNF] -> [LB VNF] -> [WAN-opt VNF]
```

一个有代表性的 VNF —— HAProxy 在服务链中作为负载均衡器：

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

**服务链**（Service Function Chaining，SFC）就是流量按顺序穿过 VNF 的拓扑。在 Kubernetes 网格里，Envoy Sidecar + 网格策略对东西向流量做的就是这件事。

---

## 5. 混合云连接：VPN、专线、Transit Gateway

多数企业生活在混合拓扑里：云上工作负载要访问本地数据库、IdP、合作伙伴网络。三层连接工具：

### 5.1 VPN vs 专线

| 特性 | 站点 VPN | 专线 / 高速通道 |
|------|---------|----------------|
| 路径 | 公网上的加密隧道 | 到运营商 PoP 的专用物理光纤 |
| 带宽 | 至多约 10 Gbit/s，抖动大 | 1 / 10 / 100 Gbit/s，可预期 |
| 延迟 | 不稳定（几十 ms + 抖动） | 同城稳定个位数 ms |
| 开通时间 | 分钟 | 周（运营商资源） |
| 成本 | 按小时 + 出网 GB | 端口按小时 + 出网 GB（规模上更便宜） |
| 加密 | IPsec 必备 | MACsec 可选；通常仍上层叠 IPsec |
| 适用 | 流量小或波动、开发环境 | 生产数据流、低延迟同步、合规 |

常见分层：**专线为主**，**VPN 作为加密备份**，专线挂了 VPN 自动接管。两者终结于同一台 VPN Gateway / TGW。

### 5.2 Transit Gateway

TGW 是地域级中枢：每个 VPC、每条 VPN、每条专线接一次，就能与其余所有互联，由 *TGW 自身的路由表* 控制谁能到谁。把 N² 对等问题变成 N 个挂载。

```hcl
resource "aws_ec2_transit_gateway" "main" {
  description                     = "prod-tgw"
  default_route_table_association = "disable"
  default_route_table_propagation = "disable"
}

resource "aws_ec2_transit_gateway_vpc_attachment" "prod" {
  transit_gateway_id = aws_ec2_transit_gateway.main.id
  vpc_id             = aws_vpc.prod.id
  subnet_ids         = [for s in aws_subnet.tgw_private : s.id]   # 每 AZ 一个 /28
}

# 两张路由表：prod 与共享服务
resource "aws_ec2_transit_gateway_route_table" "prod"   { transit_gateway_id = aws_ec2_transit_gateway.main.id }
resource "aws_ec2_transit_gateway_route_table" "shared" { transit_gateway_id = aws_ec2_transit_gateway.main.id }

# prod 能到 shared，shared 能到 prod，prod 不能到 prod
resource "aws_ec2_transit_gateway_route" "prod_to_shared" {
  destination_cidr_block         = "10.10.0.0/16"
  transit_gateway_route_table_id = aws_ec2_transit_gateway_route_table.prod.id
  transit_gateway_attachment_id  = aws_ec2_transit_gateway_vpc_attachment.shared.id
}
```

**关掉默认关联与传播，再显式写出你要的路由** —— 这样才能把 TGW 从「全连通自爆器」变成可执行的隔离边界。

---

## 6. 网络层安全

### 6.1 安全组 vs 网络 ACL

| 特性 | 安全组 | 网络 ACL |
|------|--------|---------|
| 绑在 | ENI / 实例 | 子网 |
| 状态 | **有状态**（返回流量自动允许） | **无状态**（必须显式允许双向） |
| 规则 | 仅允许 | 允许 + 拒绝 |
| 默认 | 入站全拒，出站全允许 | 双向全允许 |
| 评估方式 | 所有规则取并集 | 第一条匹配生效（按编号顺序） |
| 配额（AWS） | 入站 60 + 出站 60 条；ENI 上 5 个 SG | 每 NACL 入站 20 + 出站 20 条 |
| 适用 | 应用到应用的授权 | 限制爆炸半径、IP 黑名单 |

最稳健的写法是 **SG 引用 SG**：与其放开端口 3306 给某个 CIDR，不如放开端口 3306 给 `aws_security_group.app.id`。应用 fleet 扩缩、IP 漂移，规则始终正确；审计看到的是 *意图*，不是基础设施。

```hcl
resource "aws_security_group" "db" {
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]   # 不是 CIDR
    description     = "MySQL from app tier"
  }
  egress {
    from_port = 0  to_port = 0  protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### 6.2 VPC Flow Logs

Flow Logs 把所有被允许 / 被拒绝的 5 元组记录到 S3 或 CloudWatch。开了之后一周内你会跑这三个查询：

```sql
-- 1. 最近一小时被 REJECT 的源 IP TopN（多半是扫描器或客户端配置错）
SELECT srcaddr, COUNT(*) c
FROM "vpc_flow_logs"
WHERE action = 'REJECT' AND start_time > now() - interval '1' hour
GROUP BY srcaddr ORDER BY c DESC LIMIT 20;

-- 2. 出网到不该去的目标（数据外泄排查）
SELECT dstaddr, dstport, SUM(bytes) bytes_out
FROM "vpc_flow_logs"
WHERE direction = 'egress' AND dstaddr NOT LIKE '10.%'
GROUP BY dstaddr, dstport ORDER BY bytes_out DESC LIMIT 20;

-- 3. 跨 AZ 流量（多半是无意的，但永远要计费）
SELECT srcsubnet, dstsubnet, SUM(bytes) bytes
FROM "vpc_flow_logs"
WHERE az_id_src != az_id_dst
GROUP BY srcsubnet, dstsubnet ORDER BY bytes DESC LIMIT 20;
```

### 6.3 网络层零信任

「外硬内软」的旧模型 —— 边界硬、内网可信 —— 已经过时了。现代设计假设网络是敌对的，按请求执行身份。落地组件：

- **到处 mTLS**（服务网格做：Istio、Linkerd、App Mesh）。
- **短期工作负载身份**（SPIFFE/SPIRE、IAM Roles for Service Accounts、GCP Workload Identity）。
- **逐流策略**，由 SDN 评估（安全组 + Cilium NetworkPolicy + 网格的 L7 RBAC 叠加）。

---

## 7. BGP 与全球路由

请求一旦离开你的地域、踏上公网，或者你从一个地域故障切换到另一个地域，把它送到目的地的就是 **BGP**（Border Gateway Protocol）。BGP 是 *自治系统（AS）之间* 的路由协议；地域内你的云厂商跑 OSPF 或 IS-IS，但只要踏出地域，就是 BGP 说了算。

![跨地域 BGP 路由](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig5_bgp_multi_region.png)

### 7.1 路由选择算法（简化版）

到同一前缀有多条路径时，BGP 按一个长长的 tiebreak 列表往下走。前 5 项覆盖了 99% 的场景：

1. **最高 LOCAL_PREF** —— 「内部偏好」；*我们* 想从哪里出去。运营商手设。
2. **最短 AS_PATH** —— 经过更少的 AS。
3. **最低 MED** —— 「我希望你这样进我的 AS」；对等之间通常会尊重，跨厂商不一定。
4. **eBGP > iBGP** —— 优先外部对等学到的路由。
5. **到下一跳的最低 IGP 代价** —— *内部* 到达出口的最短路径。

### 7.2 ECMP 与 Anycast

- **ECMP（等价多路径）** —— 多条路径在以上各项都打平时，按每条流的 5 元组哈希，分散到等价的下一跳。ECMP 是你把 100 Gbit/s 逻辑链路拆成 10 条 10 Gbit/s 物理链路的方法，也是多地域 Anycast 服务把负载分到各 PoP 的方法。
- **Anycast** —— 从多地通告 *同一* IP；BGP 把每个用户送到拓扑上最近的那一处。Anycast 是每个 CDN、每个 DNS 根、每个公开 DoH 解析器的基础。

### 7.3 跨地域故障切换

全球服务的常见模式：

- 主地域承担写；其余地域承担读。
- DNS 名通过 Anycast，或挂在 Route 53 / 阿里云 GTM 的 **故障切换记录集** 上，由健康检查触发。
- 数据库异步跨地域复制；故障切换时把备库提主（RPO 秒到分钟）。
- 恢复后 *人工* 把流量切回 —— 自动「flap-back」让所有试过的人都被烧过。

BGP 的细节通常不会侵入到应用层 —— 但一旦上游误通告（[Facebook 2021 大宕机](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/)，或者每年几百次的 route leak 之一），上面所有层一起黑屏。任何有规模的网络，RPKI 路由源校验 + 入口边缘的前缀过滤都是必备卫生。

---

## 8. 生产排障

### 8.1 排查顺序

流量挂了，自下而上 *沿线* 走：

1. **可达性** —— 源能否 ARP/ND 到网关？`ip neigh`、`arping`。不行就是 L2 / SG / NACL。
2. **路由** —— 路由表里有路径吗？`aws ec2 describe-route-tables`、`ip route`。
3. **过滤** —— SG / NACL 是否放行这个 5 元组？涉及 NACL 时双向都要查。
4. **DNS** —— 名字是否解析到你以为的 IP？`dig +short`，注意分裂解析。
5. **TLS** —— `openssl s_client -connect host:443 -servername host`。「API 挂了」一半是证书过期。
6. **应用** —— `curl -vvv https://...`。日志里写着 `200 OK`？不是网络的事。

### 8.2 排障瑞士军刀

```bash
# 包到底走到哪了？
mtr -rwn -c 100 example.com           # traceroute + ping 合体，带统计
ss -tunap                              # 当前监听 / 已连接的 socket
ss -i                                  # 每 socket 的拥塞窗口、RTT
tcpdump -ni any -s 96 'port 443 and tcp[tcpflags] & (tcp-syn|tcp-fin) != 0'   # 只抓 SYN/FIN
nft list ruleset                       # 当前 netfilter 规则（现代 iptables）

# DNS
dig +trace example.com
dig @8.8.8.8 example.com               # 绕过本地 resolver，排除分裂解析

# 路径 MTU
ip -6 route get 2606:4700::6810:84e5    # 确认下一跳与 MTU
ping -M do -s 1472 8.8.8.8              # 排查 PMTUD 黑洞
```

### 8.3 五个最贵的配置错误

| 现象 | 常见原因 | 修复 |
|------|---------|------|
| 在 VPC 里的 Lambda 上不了网 | 默认路由缺失，或 NAT GW 挂了 | 给函数所在子网加 `0.0.0.0/0 -> nat-...` |
| NAT GW 账单突然飙升 | 应用从 S3 拉了大对象但没走 Gateway Endpoint | 加 `aws_vpc_endpoint.s3` |
| TCP 几秒后随机重置 | SG / NACL 允许新流，但状态跟踪过期掉了 | 调 keepalive，加 `tcp-keepalive-time` |
| 跨 AZ 数据传输费比计算费还贵 | 跨区 LB 没开，或服务没考虑 AZ 拓扑 | 开启跨区 LB；K8s 用拓扑感知路由 |
| 多归属 VPC 上的非对称路由 | SG 状态跟踪只看到一向流量 | 双向都走同一个 TGW 挂载 |

---

## 9. 常见问题

**问：自建负载均衡（HAProxy / NGINX）还是用云上 LB？**

没有特殊原因（自定义路由、云 LB 不支持的某个 Header 粘性、极端低延迟）就用云上 LB。云 LB 自身高可用、自动扩缩、与 TLS / WAF / IAM 集成；省下一个本来要去运维 HAProxy 的工程师。

**问：NLB 还是 ALB？**

HTTP/gRPC 服务用 ALB（路径/Host 路由、OIDC 鉴权、跳转）。非 HTTP 的 TCP/UDP、需要静态 IP（每个 NLB 在每 AZ 给一个 EIP）、或要极高 PPS 时用 NLB。

**问：一个 VPC 里要切几个子网？**

每 AZ 至少一个公有 + 一个私有 + 一个数据库子网。3 AZ 的 VPC 就是 9 个子网。如果按层（Web/App/DB/Mgmt）或敏感度（PCI/非 PCI）再切就更多。VPC 留 `/16`，子网用 `/22`-`/24`，留出扩容空间。

**问：什么时候我的工作负载需要自己的 SDN 控制器（而不是云上的）？**

云上几乎从来不用 —— 控制器是租来的，你也换不掉。本地 OpenStack 或 VMware NSX 才用。自建 DC fabric 的边缘 / 5G 运营商才用。绝大多数应用团队不需要在 VPC 抽象之下去想 SDN。

---

## 系列导航

| 篇 | 主题 |
|----|------|
| 1 | [基础与架构体系](/zh/cloud-computing-fundamentals/) |
| 2 | [虚拟化技术深度解析](/zh/cloud-computing-virtualization/) |
| 3 | [存储系统与分布式架构](/zh/cloud-computing-storage-systems/) |
| **4** | **网络架构与 SDN（当前）** |
| 5 | [安全与隐私保护](/zh/cloud-computing-security-privacy/) |
| 6 | [运维与 DevOps 实践](/zh/cloud-computing-operations-devops/) |
| 7 | [云原生与容器技术](/zh/cloud-computing-cloud-native-containers/) |
| 8 | [多云与混合架构](/zh/cloud-computing-multi-cloud-hybrid/) |
