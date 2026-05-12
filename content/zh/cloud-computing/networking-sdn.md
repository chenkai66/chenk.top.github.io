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
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/illustration_1.png)

云平台说到底，就是一张网络加上接在网络上的若干计算。计算靠增加节点扩展，存储靠增加磁盘扩展，而网络才是让这些节点和磁盘表现得像一个统一系统的关键。网络配置正确时，整个技术栈显得轻盈流畅；但若出错——如漏配路由、安全组五元组不匹配或负载均衡规格不足——整个平台可能直接不可用。

本文从一个数据包开始，自下而上遍历云网络栈： VPC 如何将共享物理网络切分为隔离网络、 L4 到 L7 的变化、 CDN 如何通过地理位置优化延迟、 SDN 为何重塑数据中心以及 BGP 如何将这一切跨地域连接起来。

## 我会学到

1. **VPC 内部** —— 子网、路由表、网关、端点，封装实现隔离
2. **负载均衡** —— L4/L7、算法、健康检查、会话保持、连接排空
3. **CDN 架构** —— 边缘 PoP、缓存层级、 TLS 终结、动态加速
4. **SDN** —— 控制/数据平面分离、 OpenFlow/P4Runtime、 NFV 与服务链
5. **VPN、专线、 TGW** —— 连接模型选择
6. **网络层安全** —— 安全组/NACL、流日志、零信任微隔离
7. **BGP 与全球路由** —— AS-Path、 ECMP、 Anycast、跨地域切换
## 前置知识

- IP 寻址与 CIDR
- 至少熟悉一个云控制台（AWS / GCP / 阿里云）
- 本系列前 3 篇

---

## 1. 虚拟私有云（VPC）

VPC 是云厂商基于物理网络构建的软件定义网络区域，逻辑上等效于一个私有数据中心。用户可以定义 IP 空间、划分子网、配置网关并编写防火墙规则。背后依靠 **VXLAN** 或类似技术实现隔离——每个数据包带有租户标签，即使两个客户使用相同的 10.0.0.0/16 也不会互相干扰。

![多 AZ VPC 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig1_vpc_architecture.png)

### 1.1 生产级 VPC 的组成

上图是典型的三层架构。组件如下：

| 组件 | 层级 | 功能 | 是否免费 |
|------|------|------|---------|
| **VPC** | 网络 | 10.0.0.0/16 的外壳。每环境每地域一个。 | 是 |
| **子网** | 网络 | 从 VPC 切出的 CIDR，绑定到一个 AZ。公有/私有/隔离由 *路由表* 决定。 | 是 |
| **Internet 网关 IGW** | 边缘 | 双向无 NAT 的公网通道。每个 VPC 一个。 | 是 |
| **NAT 网关** | 出口 | 让私有实例能出网，阻止入站。按 AZ 部署。 | 按小时 + GB |
| **路由表** | 控制 | CIDR -> 目标（IGW、 NAT、 VPCe、对等、 TGW）。每子网一张。 | 是 |
| **安全组** | 实例防火墙 | **有状态**、仅允许、绑在 ENI 上。 | 是 |
| **网络 ACL** | 子网防火墙 | **无状态**、允许 + 拒绝、有序。 | 是 |
| **VPC 端点** | 数据面 | 私有访问云服务（S3、 DynamoDB、 Secrets Manager…）。 | 网关型免费 / 接口型按小时 |
| **Transit Gateway** | 中枢 | N 对 N 的 VPC + VPN + 专线互联。 | 按小时 + GB |

**公有子网与私有子网的本质区别在于路由配置。** 路由表里有 `0.0.0.0/0 -> igw-...` 是公有；默认路由是 `0.0.0.0/0 -> nat-...` 是私有；没有出网路由就是隔离。听起来简单，但实际上很多「Lambda 上不了网」的问题都出在这里。

### 1.2 Terraform：多 AZ VPC 示例

![负载均衡算法对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig_lb_algorithms_zh.png)

对于无状态 Web 流量， P2C （Pick Two from Cloud）负载均衡策略已足够：实现简单，且性能接近最优。缓存敏感场景（KV 前端）用一致性哈希或 Maglev，保证同 Key 打同一后端。

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

关键但不明显的配置：

- `deregistration_delay` 必须对齐优雅关闭窗口。默认 300 秒卡发布；比在途请求超时短会断连。
- `drop_invalid_header_fields = true` 缓解 HTTP 请求走私（CVE-2019-18860 类）。
- `enable_cross_zone_load_balancing` 保证 AZ 不均时均匀分布。 NLB 默认关， ALB 默认开。

### 2.4 不撒谎的健康检查

打 `/` 跑数据库查询 *确实* 能发现数据库挂了 —— 但数据库一抖，整个 fleet 就被标 unhealthy，整站下线。能扛生产的模式：

- **`/livez`** —— 「进程还活着」（不查依赖）。 LB 打这个。
- **`/readyz`** —— 「现在能服务流量」（缓存预热完、 DB 连接池打开）。 Kubernetes 打这个。
- 应用指标、追踪、日志负责其余 —— 不是负载均衡器。
## 3. 内容分发网络（CDN）

CDN 的核心思想是将地理距离转化为缓存层级。静态内容复制到用户附近的边缘 PoP，缓存未命中才回源（通常是 S3）。延迟从 200 ms 降到 30 ms 以内，源站出网带宽减少 10 倍以上。

![CDN 边缘分布](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig4_cdn_edge.png)

### 3.1 一次请求实际发生了什么

1. **DNS** —— 用户解析 `cdn.example.com` 到 Anycast IP，落到最近的健康 PoP。
2. **TLS 终结** —— TLS 握手在边缘完成， RTT 更短， PoP 拿到 session ticket。
3. **缓存查找** —— PoP 用 URL + Vary Header 查找。命中就直接返回。
4. **未命中路径** —— PoP 向父级（地域级缓存）请求，父级视情况回源。分级缓存减少源站压力。
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

三条经验：

- **`immutable`** 是静态资源的最大优化，彻底消除重新校验流量。
- **`stale-while-revalidate`** 让 CDN 用旧内容应对，同时异步刷新，源站抖动对用户透明。
- **Vary 越窄越好。** `Vary: Accept-Encoding` 可以；`Vary: User-Agent` 把 Key 空间炸开，命中率跌到谷底。

### 3.3 不止静态：动态加速

现代 CDN 也加速 *不可缓存* 的流量，机制如下：

- **TLS / TCP 在边缘终结**，客户端握手从 `4 × RTT_to_origin` 减到 `4 × RTT_to_edge`。
- **PoP 与源站之间预热的长连接** 复用 keepalive，每请求无需重新握手。
- **Anycast DNS + BGP 优化** 把客户端送到延迟最低的 PoP，而非地图上最近的。

借助 CloudFront Functions、阿里云 DCDN、 Cloudflare Workers 等能力，「静态内容」与「动态内容」的传统边界正不断向应用层收窄——这一趋势往往超出多数团队的认知。
## 4. 软件定义网络（SDN）

传统网络把 *控制*（路由、 ACL、 QoS）绑死在设备上。 SDN 分离 **控制平面**（决策、可编程）和 **数据平面**（线速转发）。结果是网络能像软件一样管理：声明式、版本化、可观测。

![SDN 控制平面 vs 数据平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig2_sdn_planes.png)

### 4.1 二分

| 平面 | 职责 | 实现 |
|------|------|------|
| **控制平面** | 构建拓扑、计算路径、执行策略、提供 API | 逻辑集中控制器（ONOS、 OpenDaylight、超大规模厂商自研） |
| **数据平面** | 每包线速「匹配 -> 动作」 | 带流表的 ASIC 交换机；或软件方案（eBPF/XDP/DPDK） |
| **南向接口** | 控制器下发流表到交换机 | OpenFlow 1.3+、 NETCONF/YANG、 gNMI、 P4Runtime |
| **北向接口** | 应用编程控制器 | REST / gRPC、 GraphQL |

在超大规模云厂商地域内，**全是 SDN**： Hyper-V Virtual Switch、 Open vSwitch、 Cisco ACI、 AWS 的 "Mapping Service" + "Hyperplane" 共同实现 VPC。安全组规则编译成 ACL 表项，下发到宿主机 vSwitch 上 —— 不在目的实例处执行。

### 4.2 运维侧为什么爱它

- **意图集中。** 「PCI 子网流量必须审计」一条策略全局生效，不用写 200 份配置。
- **流量工程。** 控制器看到全网拓扑，实时绕开拥塞； OSPF、 IS-IS 要 30 秒才反应。
- **可编程数据面。** P4 支持新协议（SRv6、带内遥测），不用换 ASIC。
- **快速故障收敛。** SDN 预算备份路径 + BFD 检测， MTTR 压到毫秒级。

### 4.3 网络功能虚拟化（NFV）

NFV 是 SDN 的姊妹概念：*用 x86 服务器上的软件替代专用硬件*。防火墙、负载均衡、 WAN 加速器，都变成 VNF，弹性拉起、扩缩、串链。

```
传统：    [路由器硬件] -> [防火墙硬件] -> [LB 硬件] -> [WAN-opt 硬件]
NFV：    x86 服务器：[路由器 VNF] -> [防火墙 VNF] -> [LB VNF] -> [WAN-opt VNF]
```

一个典型 VNF —— HAProxy 在服务链中做负载均衡：

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

**服务链**（Service Function Chaining， SFC）是流量按顺序穿过 VNF 的拓扑。在 Kubernetes 网格里， Envoy Sidecar + 策略对东西向流量做的就是这个。
## 5. 混合云连接： VPN、专线、 Transit Gateway

多数企业用混合拓扑：云上负载要访问本地数据库、 IdP 或伙伴网络。分三层工具。

### 5.1 VPN vs 专线

| 特性 | 站点 VPN | 专线 / 高速通道 |
|------|---------|----------------|
| 路径 | 公网加密隧道 | 专用光纤到运营商 PoP |
| 带宽 | 最高约 10 Gbit/s，抖动大 | 1 / 10 / 100 Gbit/s，稳定 |
| 延迟 | 不稳定（几十 ms + 抖动） | 同城个位数 ms，稳定 |
| 开通时间 | 分钟级 | 周级（运营商资源） |
| 成本 | 按小时 + 出网 GB | 端口按小时 + 出网 GB （规模更便宜） |
| 加密 | IPsec 必须 | MACsec 可选；通常叠加 IPsec |
| 适用 | 小流量或波动、开发环境 | 生产数据流、低延迟同步、合规 |

常见做法：**专线为主**，**VPN 为加密备份**，专线断了 VPN 自动接管。两者终结于同一台 VPN Gateway / TGW。

### 5.2 Transit Gateway

TGW 是区域中枢：每个 VPC、每条 VPN、每条专线挂一次，就能互通，由 *TGW 自身路由表* 控制访问。把 N² 对等连接简化为 N 个挂载。

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

**关掉默认关联与传播，显式写路由** —— 这样才能把 TGW 从「全连通炸弹」变成可控的隔离边界。
## 6. 网络层安全

### 6.1 安全组 vs 网络 ACL

| 特性 | 安全组 | 网络 ACL |
|------|--------|---------|
| 绑定对象 | ENI / 实例 | 子网 |
| 状态 | **有状态**（返回流量自动允许） | **无状态**（双向需显式允许） |
| 规则类型 | 仅允许 | 允许 + 拒绝 |
| 默认规则 | 入站全拒，出站全允 | 双向全允 |
| 规则评估 | 所有规则取并集 | 首条匹配生效（按编号顺序） |
| 配额（AWS） | 入站 60 + 出站 60； ENI 上限 5 个 SG | 每 NACL 入站 20 + 出站 20 |
| 适用场景 | 应用间授权 | 限制爆炸半径、 IP 黑名单 |

推荐写法是 **SG 引用 SG**：不放开端口 3306 给 CIDR，而是给 `aws_security_group.app.id`。应用扩缩或 IP 漂移时，规则始终有效；审计看到的是意图，不是基础设施。

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

### 6.2 VPC Flow Logs

Flow Logs 记录所有允许/拒绝的 5 元组，存到 S3 或 CloudWatch。启用后一周内通常会跑这三个查询：

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

### 6.3 网络层零信任

「外硬内软」模型已过时。现代设计假设网络敌对，按请求验证身份。落地组件包括：

- **全面 mTLS**（服务网格： Istio、 Linkerd、 App Mesh）。
- **短期工作负载身份**（SPIFFE/SPIRE、 IAM Roles for Service Accounts、 GCP Workload Identity）。
- **逐流策略**，由 SDN 评估（安全组 + Cilium NetworkPolicy + 网格 L7 RBAC 叠加）。
## 7. BGP 与全球路由

请求离开地域进入公网，或者跨地域切换时，靠的是 **BGP**（Border Gateway Protocol）。地域内跑 OSPF 或 IS-IS，但跨出去就是 BGP 主导。

![跨地域 BGP 路由](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/networking-sdn/fig5_bgp_multi_region.png)

### 7.1 路由选择算法（简化版）

到同一前缀有多条路径时， BGP 按 tiebreak 列表依次筛选。前 5 条规则覆盖了 99% 的场景：

1. **最高 LOCAL_PREF** —— 内部偏好，决定出口方向，运营商配置。
2. **最短 AS_PATH** —— 经过的 AS 数量最少。
3. **最低 MED** —— 希望外部流量从特定入口进，对等间通常遵守。
4. **eBGP > iBGP** —— 外部路由优先于内部。
5. **最低 IGP 代价到下一跳** —— 内部到达出口的最短路径。

### 7.2 ECMP 与 Anycast

- **ECMP （等价多路径）** —— 多条路径打平时，按流的 5 元组哈希分散到等价邻居。 ECMP 用于把 100 Gbit/s 逻辑链路拆成 10 条 10 Gbit/s 物理链路，也用于多地域 Anycast 分流。
- **Anycast** —— 同一 IP 从多地通告， BGP 把用户送到拓扑最近的节点。 Anycast 是 CDN、 DNS 根、 DoH 解析器的基础。

### 7.3 跨地域故障切换

全球服务常见模式：

- 主地域写，其他地域读。
- DNS 名用 Anycast 或 Route 53 / 阿里云 GTM 的 **故障切换记录集**，健康检查触发。
- 数据库异步复制，故障时备库提主（RPO 秒到分钟）。
- 恢复后人工切回流量，自动 flap-back 几乎都翻车。

BGP 细节一般不侵入应用层，但上游误通告（[Facebook 2021 宕机](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) 或每年几百次 route leak）会让上层全黑屏。 RPKI 路由校验 + 入口前缀过滤是规模网络的必备措施。
## 8. 生产排障

### 8.1 排查顺序

流量挂了，从底层往上查：

1. **可达性** —— 源能 ARP/ND 到网关吗？`ip neigh`、`arping`。不行就是 L2 / SG / NACL。
2. **路由** —— 路由表有路径吗？`aws ec2 describe-route-tables`、`ip route`。
3. **过滤** —— SG / NACL 放行 5 元组了吗？涉及 NACL 就查双向。
4. **DNS** —— 名字解析到目标 IP 了吗？`dig +short`，小心分裂解析。
5. **TLS** —— `openssl s_client -connect host:443 -servername host`。「API 挂了」一半是证书问题。
6. **应用** —— `curl -vvv https://...`。日志里有 `200 OK`？不是网络问题。

### 8.2 排障瑞士军刀

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

### 8.3 五个最贵的配置错误

| 现象 | 常见原因 | 修复 |
|------|---------|------|
| VPC 内 Lambda 上不了网 | 默认路由缺失或 NAT GW 挂了 | 子网加 `0.0.0.0/0 -> nat-...` |
| NAT GW 账单飙升 | 应用拉 S3 大对象没走 Gateway Endpoint | 加 `aws_vpc_endpoint.s3` |
| TCP 随机重置 | SG / NACL 允许新流但状态跟踪超时 | 调 keepalive，改 `tcp-keepalive-time` |
| 跨 AZ 数据传输费高 | 跨区 LB 关闭或服务未考虑 AZ 拓扑 | 开启跨区 LB； K8s 用拓扑感知路由 |
| 多归属 VPC 非对称路由 | SG 状态跟踪只看到单向流量 | 双向都走同一个 TGW 挂载 |

---
## 9. 常见问题

**问：自建负载均衡（HAProxy / NGINX）还是用云上 LB？**

除非有特殊需求（自定义路由、云 LB 不支持的 Header 粘性、极低延迟），否则直接用云 LB。云 LB 高可用、自动扩缩，集成 TLS / WAF / IAM，省下一个运维 HAProxy 的人。

**问： NLB 还是 ALB？**

HTTP/gRPC 服务选 ALB （路径/Host 路由、 OIDC 鉴权、跳转）。非 HTTP 的 TCP/UDP、需要静态 IP （每 AZ 一个 EIP）、或极高 PPS 时选 NLB。

**问：一个 VPC 里要切几个子网？**

每 AZ 至少一个公有、一个私有、一个数据库子网。 3 AZ 就是 9 个子网。按层（Web/App/DB/Mgmt）或敏感度（PCI/非 PCI）再细分就更多。 VPC 保留 `/16`，子网用 `/22`-`/24`，留足扩容空间。

**问：什么时候我的工作负载需要自己的 SDN 控制器（而不是云上的）？**

云上几乎不用 —— 控制器是租的，换不掉。本地 OpenStack 或 VMware NSX 才用。边缘 / 5G 自建 DC fabric 的运营商才用。多数应用团队不用考虑 VPC 抽象之下的 SDN。
