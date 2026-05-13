---
title: "云计算（五）：云网络架构与 SDN"
date: 2023-04-18 09:00:00
tags:
  - Cloud Computing
  - VPC
  - Cloud Networking
  - Load Balancing
  - SDN
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "VPC、负载均衡、CDN、SDN/NFV、BGP —— 从一个数据包到全球互联，深入云网络的关键旋钮。"
disableNunjucks: true
series_order: 5
translationKey: "cloud-computing-5"
polished_by_qwen_max: true
---
云平台本质上是一张网络，上面连接着若干计算节点。计算层通过增加服务器实现扩展，存储层通过增加磁盘实现扩展，而**网络层**则将这些资源整合成一个统一、连贯的系统。网络配置得当，整个技术栈运转如丝般顺滑；一旦出错——比如漏配一条路由、安全组中的五元组不匹配，或负载均衡器容量不足——整个平台可能瞬间瘫痪。

本文将从数据包的视角出发，自底向上梳理云网络架构：VPC 如何在共享基础设施上构建隔离网络，负载均衡器从 L4 升级到 L7 带来了哪些变化，CDN 如何将地理位置优势转化为延迟红利，SDN 为何彻底重塑了数据中心网络，以及 BGP 如何将这一切无缝连接至全球。

## 你会学到

1. **VPC 内部机制** —— 子网、路由表、网关、端点，以及实现隔离的封装技术  
2. **负载均衡** —— L4 与 L7 的区别、调度算法、健康检查、会话保持、连接排空  
3. **CDN 架构** —— 边缘 PoP、缓存层级、TLS 终结、动态内容加速  
4. **SDN（软件定义网络）** —— 控制平面与数据平面分离、OpenFlow / P4Runtime、NFV 与服务链  
5. **混合云连接方案** —— VPN、专线（Direct Connect）、Transit Gateway，何时选用哪种  
6. **网络安全** —— 安全组 vs 网络 ACL、流日志、零信任微隔离  
7. **BGP 与全球路由** —— AS 路径、ECMP、Anycast、多区域故障切换  

## 前置知识

- 掌握 IP 地址规划与 CIDR 表示法  
- 至少熟悉一个主流云平台控制台（如 AWS、GCP 或阿里云）  
- 已阅读本系列前 3 篇文章  

---

## 1. 虚拟私有云（VPC）

VPC 是云服务商在物理网络之上构建的一块软件定义网络区域，其行为**如同你专属的私有数据中心**：你可以自由选择 IP 地址空间、划分子网、部署网关，并编写防火墙规则。在底层，云厂商通过 **VXLAN**（或其专有等效技术）实现租户隔离——每个数据包都携带租户标识，因此即使两个客户同时使用 `10.0.0.0/16`，彼此的流量也完全不可见。

![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/illustration_1.png)

### 1.1 生产级 VPC 的组成结构

上图展示了一个典型的三层架构部署。各组件说明如下：

| 组件 | 所属层级 | 功能说明 | 是否免费 |
|------|----------|----------|----------|
| **VPC** | 网络层 | 定义 `10.0.0.0/16` 的地址空间边界。通常每个环境、每个区域一个。 | 是 |
| **子网** | 网络层 | 从 VPC 中划分出的 CIDR 块，绑定到单个可用区（AZ）。所谓“公有”“私有”或“隔离”，完全由*路由表*决定，而非名称。 | 是 |
| **Internet 网关（IGW）** | 边缘层 | 提供双向、无 NAT 的公网访问通道。每个 VPC 只能有一个。 | 是 |
| **NAT 网关** | 出口层 | 允许*私有子网*中的实例主动访问互联网，但阻止外部入站连接。为高可用，建议每个 AZ 部署一个。 | 按小时计费 + 流量费用 |
| **路由表** | 控制层 | 将目标 CIDR 映射到具体出口（如 IGW、NAT、VPC 端点、对等连接、TGW）。每个子网关联一张路由表。 | 是 |
| **安全组** | 实例防火墙 | **有状态**、仅支持“允许”规则，直接绑定到弹性网卡（ENI）。 | 是 |
| **网络 ACL** | 子网防火墙 | **无状态**，支持“允许”和“拒绝”规则，按顺序匹配。通常与安全组配合使用，形成双重防护。 | 是 |
| **VPC 端点** | 数据平面 | 提供 VPC 到云服务（如 S3、DynamoDB、Secrets Manager 等）的私有连接路径。 | 网关型免费 / 接口型按小时计费 |
| **Transit Gateway（TGW）** | 中枢层 | 支持 N 对 N 的 VPC、VPN 和专线互联，作为中心枢纽。 | 按小时计费 + 流量费用 |

**“公有子网”与“私有子网”的本质区别，完全取决于路由表配置。**  
- 若子网路由表包含 `0.0.0.0/0 -> igw-...`，则为公有子网；  
- 若默认路由为 `0.0.0.0/0 -> nat-...`，则为私有子网；  
- 若完全没有互联网出口路由，则为隔离子网。  

这看似简单，却是“我的 Lambda 为什么无法访问互联网”这类工单的罪魁祸首。

### 1.2 Terraform：一个多 AZ 的生产级 VPC 示例

![多 AZ VPC 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig1_vpc_architecture.png)

上述配置体现了三个关键的生产实践：

- **每个 AZ 部署一个 NAT 网关**：单个 NAT 网关构成 AZ 级故障域；若整个区域共用一个，则可能引发区域性中断。  
- **为 S3 和 DynamoDB 配置网关端点**：否则，私有子网访问 S3 的所有流量都会经过 NAT 网关，产生高达 $0.045/GB 的费用；而端点本身是免费的。  
- **仅在公有子网上启用 `map_public_ip_on_launch`**：避免意外给数据库等私有资源分配公网 IP。

### 1.3 VPC 互联方案：对等连接、TGW 与 PrivateLink

| 方案 | 拓扑结构 | 适用场景 | 注意事项 |
|------|----------|----------|----------|
| **VPC 对等连接** | 点对点 | 2–5 个 VPC 的简单区域内互联 | **非传递性**：A↔B 且 B↔C，并不意味着 A↔C |
| **Transit Gateway** | 星型枢纽 | 5 个以上 VPC、需集成 VPN/专线、需集中审计 | 按小时 + 流量计费 |
| **PrivateLink** | 服务暴露 | 一个 VPC 向多个消费者 VPC 提供服务 | 单向连接；无需打通整个网络 |
| **Cloud WAN** | 全球网状 | 多区域、多账号的大规模网状互联 | 最新方案，大规模下最简洁，成本相应更高 |

选择的关键通常在于所需**网络拓扑**，而非带宽——只要合理规划，上述方案均可支持数十 Gbps 的吞吐。

---

## 2. 负载均衡

负载均衡器将一组后端实例抽象为一个统一服务。它不仅能屏蔽单点故障、均衡流量，还能终结 TLS；在 L7 模式下，甚至能基于路径、主机名或请求头对流量进行精细化调度，而无需改动应用代码。

### 2.1 L4 与 L7 负载均衡对比

| 特性 | 网络负载均衡（L4） | 应用负载均衡（L7） |
|------|-------------------|-------------------|
| 工作协议 | TCP / UDP / TLS | HTTP / HTTPS / gRPC |
| 路由依据 | 五元组（源/目的 IP+端口+协议） | 路径、主机名、Header、Cookie、JWT 声明 |
| 引入延迟 | 数十微秒 | 数毫秒 |
| 单 LB 吞吐 | 数百 Gbps | 数十 Gbps |
| TLS 终结 | 可选（透传或终结） | 几乎总是终结 |
| WebSocket / HTTP/2 / gRPC | 仅支持透传 | 原生支持 |
| 源 IP 保留 | 支持（需 Proxy Protocol） | 通过 `X-Forwarded-For` 传递 |
| 适用场景 | 游戏服务器、IoT、MQTT、底层 TCP 服务 | Web 应用、微服务、公开 API |

现代架构中常见的组合模式是：**L4（NLB） → L7（Envoy/ALB） → 服务**。L4 层吸收 DDoS 攻击并提供稳定的 Anycast IP；L7 层负责智能路由与认证。每一层各司其职，发挥所长。

### 2.2 负载均衡算法

![负载均衡算法对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig_lb_algorithms_zh.png)

对于大多数无状态 Web 流量，**“二选一”（Power-of-Two-Choices, P2C）** 算法在实践中接近最优，且实现简单。而对于缓存敏感型负载（如键值存储前端），应使用一致性哈希或 Maglev 哈希，确保相同 Key 始终命中同一后端。

### 2.3 基于路径与主机名的 ALB 配置（Terraform）

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

  deregistration_delay = 30   # 对齐优雅关闭窗口
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

几个容易被忽视的关键配置：

- `deregistration_delay` 应与应用的优雅停机窗口对齐。默认 300 秒会拖慢发布流程；若设得比请求超时还短，则会导致连接中断。  
- `drop_invalid_header_fields = true` 可有效缓解 HTTP 请求走私类攻击（如 CVE-2019-18860）。  
- `enable_cross_zone_load_balancing` 能在各 AZ 后端数量不均时保证流量均匀分布。NLB 默认关闭，ALB 默认开启。

### 2.4 不会“撒谎”的健康检查

如果健康检查接口 `/` 会执行数据库查询，那它确实能在数据库宕机时发出警报——但也会因一次短暂的数据库抖动，将整个服务集群标记为不健康，导致全站不可用。真正经得起生产考验的做法是：

- **`/livez`** —— 仅检查“进程是否存活”（不依赖任何外部服务），由负载均衡器调用。  
- **`/readyz`** —— 检查“当前实例是否准备好处理请求”（如缓存已预热、数据库连接池已就绪），由 Kubernetes 调用。  
- 其余监控（指标、链路追踪、日志）交由专门的可观测性系统处理，而非负载均衡器。

---

## 3. 内容分发网络（CDN）

CDN 的核心思想是**将地理距离转化为缓存优势**。静态内容被复制到靠近用户的边缘 PoP（Point of Presence）；只有缓存未命中时，才会回源（通常是 S3 存储桶）。此举可将延迟从 200 毫秒降至 30 毫秒以内，同时将源站出口流量减少 10 倍以上。

![CDN 边缘分布](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig4_cdn_edge.png)

### 3.1 一次 CDN 请求的实际流程

1. **DNS 解析** —— 用户将 `cdn.example.com` 解析为一个 Anycast IP，该请求被路由至地理上最近且健康的 PoP。  
2. **TLS 终结** —— TLS 握手在边缘完成（RTT 更短），PoP 保存会话票据以供复用。  
3. **缓存查找** —— PoP 根据 URL 和 `Vary` 头进行缓存键匹配。命中则立即返回。  
4. **未命中处理** —— PoP 向上级缓存（如区域级缓存）请求内容，后者可能再回源。这种分层缓存架构即使面对唯一内容，也能显著减轻源站压力。  
5. **缓存填充** —— PoP 根据响应中的 TTL（如 `Cache-Control: max-age`、`s-maxage`、`stale-while-revalidate`）存储内容。

### 3.2 实战有效的缓存头配置

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

三条经验法则：

- **`immutable`** 是静态资源的最大优化项，可彻底消除客户端的重新验证请求。  
- **`stale-while-revalidate`** 允许 CDN 在后台异步刷新缓存的同时，向前端用户提供稍旧的内容，使源站抖动对用户完全透明。  
- **`Vary` 头应尽可能精简**。`Vary: Accept-Encoding` 是合理的；但 `Vary: User-Agent` 会指数级膨胀缓存键空间，严重拉低命中率。

### 3.3 超越静态：动态内容加速

现代 CDN 同样能加速**不可缓存的动态流量**，主要依靠以下机制：

- **在边缘终结 TLS/TCP**：将客户端握手延迟从 `4 × RTT_to_origin` 缩短至 `4 × RTT_to_edge`。  
- **PoP 与源站间预热的长连接**：复用 keepalive 连接，避免每次请求都重新握手。  
- **Anycast DNS + BGP 优化**：将用户导向延迟最低的 PoP，而非单纯地理距离最近的节点。

借助 CloudFront Functions、阿里云 DCDN、Cloudflare Workers 等能力，静态与动态内容的传统边界正不断向应用层内移——这一趋势往往超出多数团队的认知。

---

## 4. 软件定义网络（SDN）

传统网络将**控制逻辑**（如路由、ACL、QoS）紧密耦合在每台设备上。SDN 则将**控制平面**（负责决策，可被上层编程）与**数据平面**（负责线速转发）彻底分离。其结果是：网络可以像软件一样被管理——声明式、可版本控制、可观测。

![SDN 控制平面 vs 数据平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig2_sdn_planes.png)

### 4.1 控制平面与数据平面分离

| 平面 | 职责 | 典型实现 |
|------|------|----------|
| **控制平面** | 构建网络拓扑、计算转发路径、执行策略、暴露 API | 逻辑集中的控制器（如 ONOS、OpenDaylight，或超大规模云厂商自研系统） |
| **数据平面** | 对每个数据包执行“匹配 → 动作”操作，达到线速转发 | 带流表的 ASIC 交换机，或基于 eBPF/XDP/DPDK 的软件方案 |
| **南向接口** | 控制器向数据平面设备下发流表规则 | OpenFlow 1.3+、NETCONF/YANG、gNMI、P4Runtime |
| **北向接口** | 应用程序对控制器进行编程 | REST / gRPC、GraphQL |

在超大规模云厂商的单个区域内，**一切皆 SDN**：Hyper-V Virtual Switch、Open vSwitch、Cisco ACI，以及 AWS 的 “Mapping Service” + “Hyperplane”，共同实现了你所感知的 VPC。你的安全组规则会被编译成 ACL 条目，并精准推送到数据包路径上的宿主机虚拟交换机——而非在目标实例处执行。

### 4.2 运维人员为何青睐 SDN

- **意图驱动的集中策略**：例如“所有 PCI 子网流量必须经过审计”只需一条策略，全局生效，无需在 200 台设备上分别配置。  
- **精细化流量工程**：控制器掌握全网拓扑，可实时绕开拥塞链路；而传统协议（如 OSPF、IS-IS）通常需 30 秒才能响应。  
- **可编程的数据平面**：P4 语言允许数据平面解析新协议（如 SRv6、自定义带内遥测），无需更换 ASIC 芯片。  
- **毫秒级故障恢复**：SDN 预先计算备用路径，结合 BFD 快速检测，将平均修复时间（MTTR）压缩至毫秒级。

### 4.3 网络功能虚拟化（NFV）

NFV 是 SDN 的“孪生兄弟”：**用运行在通用 x86 服务器上的软件，替代专用网络硬件**。防火墙、负载均衡器、WAN 优化器等，都变成了可弹性伸缩、灵活编排的虚拟网络功能（VNF）。

```text
传统：    [路由器硬件] -> [防火墙硬件] -> [LB 硬件] -> [WAN-opt 硬件]
NFV：    x86 服务器：[路由器 VNF] -> [防火墙 VNF] -> [LB VNF] -> [WAN-opt VNF]
```

一个典型的 VNF 示例——在服务链中使用 HAProxy 作为负载均衡器：

```text
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

**服务链**（Service Function Chaining, SFC）定义了流量依次穿过 VNF 的顺序。在 Kubernetes 服务网格中，Envoy Sidecar 配合网格策略，正是通过这种方式处理东西向流量。

---

## 5. 混合云连接：VPN、专线与 Transit Gateway

大多数企业采用混合架构：云上工作负载需要访问本地数据中心的数据库、身份提供商或合作伙伴网络。为此，云平台提供了三层连接工具。

### 5.1 VPN 与专线对比

| 特性 | 站点到站点 VPN | 专线（Direct Connect / Express Connect / Cloud Interconnect） |
|------|----------------|-------------------------------------------------------------|
| 传输路径 | 公网上的加密隧道 | 专用光纤直连至云厂商接入点（PoP） |
| 带宽 | 最高约 10 Gbit/s，抖动较大 | 1 / 10 / 100 Gbit/s，稳定可靠 |
| 延迟 | 波动大（数十毫秒 + 抖动） | 同城内低至个位数毫秒，极其稳定 |
| 开通时间 | 分钟级 | 数周（依赖运营商资源调配） |
| 成本 | 按小时 + 出向流量计费 | 端口按小时 + 出向流量计费（大规模下更划算） |
| 加密 | 必须使用 IPsec | MACsec 可选；通常仍叠加 IPsec 以保障机密性 |
| 适用场景 | 低流量、波动性大或开发测试环境 | 生产数据同步、低延迟要求、合规性场景 |

常见做法是：**以专线为主链路，VPN 作为加密备份**。当专线中断时，流量自动切换至 VPN。两者通常终结于同一个虚拟私有网关（Virtual Private Gateway）或 Transit Gateway。

### 5.2 Transit Gateway（TGW）

TGW 是区域级网络中枢：每个 VPC、每条 VPN、每条专线只需挂载一次，即可通过 TGW 自身的**路由表**实现互通。它将传统的 N² 对等连接问题简化为 N 个挂载点。

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

# 两张路由表：prod 和共享服务
resource "aws_ec2_transit_gateway_route_table" "prod"   { transit_gateway_id = aws_ec2_transit_gateway.main.id }
resource "aws_ec2_transit_gateway_route_table" "shared" { transit_gateway_id = aws_ec2_transit_gateway.main.id }

# prod 到 shared，shared 到 prod，prod 不到 prod
resource "aws_ec2_transit_gateway_route" "prod_to_shared" {
  destination_cidr_block         = "10.10.0.0/16"
  transit_gateway_route_table_id = aws_ec2_transit_gateway_route_table.prod.id
  transit_gateway_attachment_id  = aws_ec2_transit_gateway_vpc_attachment.shared.id
}
```

**关键实践：禁用默认的关联与传播，显式编写所需路由**。唯有如此，才能将 TGW 从“全连通炸弹”转变为可强制执行的网络隔离边界。

---

## 6. 网络安全

### 6.1 安全组 vs 网络 ACL

| 特性 | 安全组 | 网络 ACL |
|------|--------|----------|
| 绑定对象 | 弹性网卡（ENI）/ 实例 | 子网 |
| 状态性 | **有状态**（返回流量自动放行） | **无状态**（必须显式允许双向流量） |
| 规则类型 | 仅支持“允许” | 支持“允许”和“拒绝” |
| 默认策略 | 入站全拒，出站全允 | 双向全允 |
| 规则评估 | 所有规则取并集 | 按编号顺序，首条匹配即生效 |
| 配额（AWS） | 入站 60 + 出站 60 条；每个 ENI 最多 5 个安全组 | 每个 NACL 入站 20 + 出站 20 条 |
| 适用场景 | 应用间细粒度授权 | 限制爆炸半径、IP 黑名单 |

最佳实践是使用 **安全组引用安全组**：不要对 CIDR 开放 3306 端口，而是开放给 `aws_security_group.app.id`。这样，无论应用实例如何扩缩容或 IP 如何变化，规则始终有效；审计人员看到的是业务意图，而非底层基础设施细节。

```hcl
resource "aws_security_group" "db" {
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]   # 不用 CIDR
    description     = "MySQL from app tier"
  }
  egress {
    from_port = 0  to_port = 0  protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### 6.2 VPC 流日志（Flow Logs）

流日志会将所有被允许或拒绝的五元组记录到 S3 或 CloudWatch。启用后一周内，你大概率会运行以下三类查询：

```sql
-- 1. 最近一小时被 REJECT 的源 IP TopN（多为扫描器或配置错误）
SELECT srcaddr, COUNT(*) c
FROM "vpc_flow_logs"
WHERE action = 'REJECT' AND start_time > now() - interval '1' hour
GROUP BY srcaddr ORDER BY c DESC LIMIT 20;

-- 2. 出网到异常目标（排查数据外泄）
SELECT dstaddr, dstport, SUM(bytes) bytes_out
FROM "vpc_flow_logs"
WHERE direction = 'egress' AND dstaddr NOT LIKE '10.%'
GROUP BY dstaddr, dstport ORDER BY bytes_out DESC LIMIT 20;

-- 3. 跨 AZ 流量（通常无意，但计费不可避免）
SELECT srcsubnet, dstsubnet, SUM(bytes) bytes
FROM "vpc_flow_logs"
WHERE az_id_src != az_id_dst
GROUP BY srcsubnet, dstsubnet ORDER BY bytes DESC LIMIT 20;
```

### 6.3 网络层的零信任架构

“外硬内软”的传统安全模型已然过时。现代架构假设网络本身是敌对的，需在每次请求时验证身份。关键构建模块包括：

- **全面启用 mTLS**（通过服务网格实现，如 Istio、Linkerd、App Mesh）；  
- **使用短期工作负载身份**（如 SPIFFE/SPIRE、IAM Roles for Service Accounts、GCP Workload Identity）；  
- **逐流策略执行**，由 SDN 层统一评估（安全组 + Cilium NetworkPolicy + 服务网格 L7 RBAC 多层叠加）。

---

## 7. BGP 与全球路由

当请求离开你的区域、穿越公共互联网，或在多区域间故障切换时，背后驱动这一切的是 **BGP（边界网关协议）**。BGP 是自治系统（AS）之间的路由协议；在云厂商区域内，通常使用 OSPF 或 IS-IS，但一旦跨出区域边界，BGP 便成为主导。

![跨地域 BGP 路由](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig5_bgp_multi_region.png)

### 7.1 BGP 路由选择算法（简化版）

当存在多条到达同一前缀的路径时，BGP 会按优先级顺序进行筛选。前五条规则覆盖了 99% 的实际场景：

1. **最高 LOCAL_PREF** —— 内部偏好值，决定“我们希望如何离开自己的网络”，由运维人员设定。  
2. **最短 AS_PATH** —— 经过的自治系统数量最少。  
3. **最低 MED（Multi-Exit Discriminator）** —— “建议你从这个入口进入我的 AS”，通常在对等体之间被尊重。  
4. **eBGP 优于 iBGP** —— 优先选择从外部对等体学到的路由。  
5. **到下一跳的 IGP 开销最低** —— 在内部网络中，选择到达出口的最短路径。

### 7.2 ECMP 与 Anycast

- **ECMP（等价多路径）**：当多条路径在上述所有条件上完全相同时，BGP 会对每个流的五元组进行哈希，并将流量分散到这些等价路径上。ECMP 既用于将 10 条 10 Gbit/s 物理链路聚合成 100 Gbit/s 逻辑链路，也用于将多区域 Anycast 服务的流量均衡分发至各 PoP。  
- **Anycast**：从多个地理位置**通告相同的 IP 地址**，BGP 会将每个用户自动路由至拓扑上最近的节点。Anycast 是所有 CDN、DNS 根服务器、公共 DoH 解析器的基石。

### 7.3 多区域故障切换

全球服务的典型模式如下：

- 主区域处理写请求，其他区域处理读请求；  
- DNS 名称通过 Anycast 或 Route 53 / 阿里云 GTM 的**故障切换记录集**提供服务，由健康检查触发切换；  
- 数据库采用异步跨区域复制，故障时将备库提升为主库（RPO 通常为数秒至数分钟）；  
- 故障恢复后，**手动**将流量切回主区域——自动回切（flap-back）几乎让所有尝试者吃过亏。

虽然 BGP 细节通常不会侵入应用层，但一旦上游发生误通告（如 [Facebook 2021 年大宕机](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) 或每年数百起的路由泄露事件），上层所有服务都会瞬间瘫痪。因此，在任何具备规模的网络中，**RPKI 路由源验证**和**入口处的严格前缀过滤**都是必不可少的基础防护措施。

---

## 8. 生产环境排障指南

### 8.1 排查顺序

当流量异常时，应**自底向上**逐层排查：

1. **可达性** —— 源主机能否通过 ARP/ND 解析到网关？使用 `ip neigh` 或 `arping`。若不能，问题出在 L2 层、安全组或网络 ACL。  
2. **路由** —— 路由表是否存在有效路径？使用 `aws ec2 describe-route-tables` 或 `ip route`。  
3. **过滤规则** —— 安全组或网络 ACL 是否放行了该五元组？若涉及 NACL，需检查双向规则。  
4. **DNS** —— 域名是否解析到了预期 IP？使用 `dig +short`，注意分裂 DNS（split-horizon）的影响。  
5. **TLS** —— 使用 `openssl s_client -connect host:443 -servername host` 测试。一半所谓的“API 不可用”其实是证书过期所致。  
6. **应用层** —— 使用 `curl -vvv https://...` 直接测试。若日志显示 `200 OK`，那问题就不在网络层。

### 8.2 排障瑞士军刀命令集

```bash
# 包走到哪了？
mtr -rwn -c 100 example.com           # traceroute + ping 合体，带统计
ss -tunap                              # 查看监听和连接的 socket
ss -i                                  # 每个 socket 的拥塞窗口、RTT
tcpdump -ni any -s 96 'port 443 and tcp[tcpflags] & (tcp-syn|tcp-fin) != 0'   # 只抓 SYN/FIN
nft list ruleset                       # 当前 netfilter 规则（现代 iptables）

# DNS
dig +trace example.com
dig @8.8.8.8 example.com               # 绕过本地 resolver，排查分裂解析

# 路径 MTU
ip -6 route get 2606:4700::6810:84e5    # 确认下一跳与 MTU
ping -M do -s 1472 8.8.8.8              # 排查 PMTUD 黑洞
```

### 8.3 五大“天价”配置错误

| 现象 | 常见原因 | 修复方案 |
|------|---------|----------|
| VPC 内的 Lambda 无法访问互联网 | 子网缺少默认路由，或 NAT 网关宕机 | 在 Lambda 所在子网添加 `0.0.0.0/0 -> nat-...` 路由 |
| NAT 网关账单突然飙升 | 应用从 S3 下载大文件时未使用网关端点，流量经 NAT 网关出站 | 添加 `aws_vpc_endpoint.s3` |
| TCP 连接在数秒后随机重置 | 安全组/NACL 允许新建连接，但状态跟踪因空闲超时被清除 | 调整应用 keepalive 设置，或增大 `tcp-keepalive-time` |
| 跨 AZ 数据传输费用远超计算成本 | 负载均衡器未开启跨区均衡，或服务未实现 AZ 感知 | 启用跨区负载均衡；在 K8s 中使用拓扑感知路由 |
| 多归属 VPC 出现非对称路由 | 安全组的状态跟踪只看到单向流量 | 确保往返流量均通过同一个 TGW 挂载点 |

---

## 9. 常见问题

**问：我们应该自建负载均衡器（如 HAProxy / NGINX），还是直接使用云厂商的 LB？**

除非有特殊需求（如云 LB 不支持的自定义路由逻辑、特定 Header 的会话保持、或极低延迟要求），否则**强烈建议使用云厂商的负载均衡器**。它们天然具备高可用、自动扩缩容能力，并深度集成 TLS、WAF、IAM 等服务，还能省下一个专职维护 HAProxy 的工程师。

**问：该选 NLB 还是 ALB？**

- **ALB** 适用于 HTTP/gRPC 服务，支持基于路径/主机名的路由、OIDC 认证、重定向等功能。  
- **NLB** 适用于非 HTTP 的 TCP/UDP 流量、需要静态 IP（每个 AZ 分配一个 EIP）、或对每秒包数（PPS）有极高要求的场景。

**问：一个 VPC 中应该划分多少个子网？**

至少为每个 AZ 配置一个公有子网、一个私有子网和一个数据库子网。因此，一个 3 AZ 的 VPC 至少需要 9 个子网。若按应用层级（Web/App/DB/Mgmt）或安全等级（PCI/非 PCI）进一步隔离，子网数量还会增加。建议 VPC 使用 `/16` 地址空间，子网使用 `/22` 至 `/24`，为未来扩容预留充足空间。

**问：什么情况下我的工作负载需要自建 SDN 控制器（而非使用云厂商提供的）？**

在公有云上，**几乎永远不需要**——你租用的是云厂商的 SDN 控制器，无法替换。只有在以下场景才需考虑：  
- 本地部署 OpenStack 或 VMware NSX；  
- 边缘计算或 5G 运营商自建数据中心网络架构。  
对绝大多数应用团队而言，不应也不必关心 VPC 抽象之下的 SDN 实现细节。
