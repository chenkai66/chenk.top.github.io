---
title: "系统设计（二）：DNS、CDN 与负载均衡——请求旅程的前三跳"
date: 2025-07-12 09:00:00
tags:
  - System Design
  - Networking
  - Load Balancing
  - CDN
categories: System Design
series: system-design
lang: zh
description: "每个 Web 请求都始于 DNS 解析，可能途经 CDN 边缘节点，最终抵达负载均衡器，才到达你的应用。理解这三跳机制，是构建快速、可靠、全球分布式系统的基石。"
disableNunjucks: true
series_order: 2
translationKey: "system-design-2"
---
2017 年，一家主流云服务商因一条配置错误的 DNS 记录，导致互联网大面积中断数小时——成千上万个网站无法访问，并非因为服务器宕机，而是负责将域名翻译为 IP 地址的系统出了问题。这次事故尖锐地提醒我们：那些被视作理所当然的基础设施——DNS、CDN 和负载均衡器——正是整个系统赖以运转的地基。

用户发起的每个 HTTP 请求，在抵达你的应用代码前，至少会经过其中两个系统。一旦任一环节失效或性能不佳，下游的所有优化都将毫无意义。

## DNS 解析

域名系统（Domain Name System，DNS）是一个分布式、分层的数据库，用于将人类可读的域名映射为 IP 地址。当用户在浏览器中输入 `photos.example.com` 时，一系列 DNS 查询便已悄然启动，远早于你的任何一行应用代码执行。

![DNS 解析流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-dns-resolution.png)

### 解析流程

DNS 解析包含两种查询模式：递归查询（recursive）和迭代查询（iterative）。

**递归查询** 是浏览器的行为：它向一个递归解析器（通常是 ISP 提供的 DNS 服务器，或公共解析器如 `8.8.8.8`）发送请求，并期望直接获得最终答案；所有中间查询工作都由该解析器完成。

**迭代查询** 则是递归解析器内部的操作：它依次向权威服务器链发起查询，每台服务器要么返回答案，要么指引它去查询更具体的下一级服务器。

以 `photos.example.com` 为例，完整的解析链如下：

1. 浏览器检查本地 DNS 缓存  
2. 操作系统检查其 DNS 缓存（先查 `/etc/hosts`，再查系统解析器缓存）  
3. 查询被发送至配置的递归解析器  
4. 解析器向根域名服务器询问：“谁负责 `.com` 域？”  
5. 根服务器返回 `.com` 顶级域（TLD）服务器的地址  
6. 解析器向 `.com` TLD 服务器询问：“谁负责 `example.com`？”  
7. TLD 服务器返回 `example.com` 的权威域名服务器地址  
8. 解析器向该权威服务器询问：“`photos.example.com` 的 IP 是什么？”  
9. 权威服务器返回 IP 地址  
10. 解析器缓存结果并返回给客户端  

对于未缓存的查询，整条链路通常耗时 20–120ms；若命中缓存，则可在 1ms 内完成。

### DNS 记录类型

| 记录类型 | 用途 | 示例 |
|------------|---------|---------|
| A | 将域名映射到 IPv4 地址 | `photos.example.com → 93.184.216.34` |
| AAAA | 将域名映射到 IPv6 地址 | `photos.example.com → 2606:2800:220:1:...` |
| CNAME | 将一个域名设为另一个域名的别名 | `www.example.com → example.com` |
| MX | 邮件交换服务器 | `example.com → mail.example.com (priority 10)` |
| NS | 权威域名服务器 | `example.com → ns1.example.com` |
| TXT | 存储任意文本（用于 SPF、DKIM、域名验证等） | `example.com → "v=spf1 include:..."` |
| SRV | 指定服务位置（主机 + 端口） | `_sip._tcp.example.com → sipserver.example.com:5060` |
| PTR | 反向解析（IP → 域名） | `34.216.184.93 → photos.example.com` |

### TTL 与缓存

每条 DNS 记录都带有 TTL（Time To Live，单位为秒）。解析器在缓存记录时，会严格遵守该 TTL，超时后才会重新查询。

```text
; 示例 DNS 区域文件条目
photos.example.com.   300   IN  A      93.184.216.34
photos.example.com.   300   IN  A      93.184.216.35
cdn.example.com.      3600  IN  CNAME  d111111abcdef8.cloudfront.net.
```

TTL 的权衡如下：
- **短 TTL（30–300 秒）**：故障转移更快，但会增加 DNS 查询量和解析器负载；
- **长 TTL（3600–86400 秒）**：减少查询次数，提升性能，但故障恢复更慢；
- **迁移期间**：建议提前数天将 TTL 调低（如 60 秒），完成切换后再调高 TTL。

### 基于 DNS 的负载均衡

DNS 可通过返回不同的 IP 地址，将流量分发到多个服务器。

**轮询 DNS（Round Robin DNS）**：返回多条 A 记录，客户端通常选择第一条。这种方式简单，但缺乏健康检查——即使某台服务器已宕机，DNS 仍会返回其 IP。

```text
photos.example.com.  300  IN  A  10.0.1.1
photos.example.com.  300  IN  A  10.0.1.2
photos.example.com.  300  IN  A  10.0.1.3
```

**加权 DNS（Weighted DNS）**：按预设概率返回不同记录。AWS Route 53 等服务支持此功能。

**地理 DNS（GeoDNS）**：根据解析器的地理位置返回不同的 IP。例如，东京用户获得东京数据中心的 IP，伦敦用户则获得法兰克福数据中心的 IP。

GeoDNS 是全局负载均衡的基础，但也存在明显局限：
- 定位依据是解析器的 IP，而非用户真实 IP（使用 VPN 的用户可能被导向错误区域）；
- DNS 缓存导致配置变更传播缓慢；
- 若不结合具备健康检查能力的 DNS 服务，无法实时感知后端状态。

## 内容分发网络（CDN）

CDN 是一个全球分布的代理服务器网络，能将内容缓存到靠近终端用户的边缘节点。当悉尼的用户请求一张位于弗吉尼亚服务器上的图片时，CDN 会直接从悉尼的边缘节点提供该资源。

![CDN 边缘缓存拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-cdn-topology.png)

### CDN 缓存机制

基本流程如下：

1. 用户请求 `https://photos.example.com/img/abc123.jpg`  
2. DNS 将 `photos.example.com` 解析为最近的 CDN 边缘节点（通过 GeoDNS 或 Anycast）  
3. 边缘节点检查本地缓存是否存在该对象  
4. **缓存命中（Cache hit）**：直接返回对象（延迟：5–20ms）  
5. **缓存未命中（Cache miss）**：边缘节点向源站（origin server）拉取内容，缓存后再返回给用户  

### 源站拉取（Origin Pull） vs 源站推送（Origin Push）

**源站拉取（懒加载）**：CDN 在首次请求（缓存未命中）时从源站拉取内容并缓存。这是大多数 CDN 的默认模式。

优势：
- 配置简单——只需将 DNS 指向 CDN；
- 仅缓存实际被请求的内容；
- 无需预热缓存。

劣势：
- 每个对象的首次请求较慢（需回源）；
- 源站需承担缓存未命中的流量压力；
- 热门对象缓存过期时容易引发“惊群效应”（thundering herd）。

**源站推送（主动式）**：你主动将内容上传至 CDN 的存储系统。适用于大文件、视频内容和软件下载。

优势：
- 用户无缓存未命中延迟；
- 源站完全不参与内容分发；
- 更适合传输成本高昂的大文件。

劣势：
- 需集成 CDN 提供的上传 API；
- 缓存生命周期需显式管理；
- CDN 侧会产生额外存储费用。

### CDN 缓存失效策略

CDN 缓存失效向来是个难题。常见策略包括：

**TTL 驱动**：在响应头中设置 `Cache-Control`。

```text
Cache-Control: public, max-age=31536000  # 不变资源：1 年
Cache-Control: public, max-age=300       # 半动态内容：5 分钟
Cache-Control: no-store                  # 绝对不缓存（全动态内容）
```

**版本化 URL**：在 URL 中附加版本号或哈希值。内容更新即 URL 变更，旧缓存自然失效。

```text
/static/app.a1b2c3d4.js    # 文件名含哈希
/img/photo.jpg?v=20250712  # 查询参数带版本
```

**Purge API**：多数 CDN 提供显式清除缓存的 API。应谨慎使用——全球生效通常需要 5–30 秒。

```bash
# CloudFront 缓存清除示例
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/img/abc123.jpg" "/api/feed/*"
```

### CDN 何时增益、何时拖累

**CDN 有益的场景**：
- 内容静态或半静态（如图片、CSS、JS、视频）；
- 用户地理分布广泛；
- 读写比高；
- 内容被大量用户共享（如同一张图服务百万用户）。

**CDN 有害的场景**：
- 内容高度个性化（如用户仪表盘、账户页）；
- 内容频繁变更（如实时数据）；
- 内容访问极稀疏（长尾内容，缓存命中率低）；
- 对强一致性有严苛要求（CDN 可能返回陈旧数据）。

### CDN 架构

主流 CDN 提供商在全球数十个国家运营数百个接入点（Points of Presence, PoPs）。每个 PoP 包含：

- **边缘服务器（Edge servers）**：缓存并提供内容，处理 TLS 终止；
- **区域缓存（Mid-tier caches）**：容量更大的中间层缓存，位于边缘与源站之间，显著降低源站压力；
- **路由基础设施**：通过 Anycast IP 或 GeoDNS，将用户导向最近的 PoP。

分层缓存架构至关重要。如果没有区域缓存，每个边缘节点都会独立回源拉取未命中内容；而引入区域缓存后，同一区域内所有边缘节点的未命中请求，只需一次回源即可满足。

## 第 4 层负载均衡（Layer 4 Load Balancing）

![DNS 解析过程，消息通过层级结构传输](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/02-dns-resolution-journey-message-traveling-through-hierarchica.jpg)

第 4 层负载均衡器工作在传输层（TCP/UDP），仅依据 IP 地址和端口号做路由决策，不会解析应用层的数据内容。

![L4 与 L7 负载均衡对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-l4-vs-l7.png)

### 工作原理

第 4 层负载均衡器接收 TCP 连接，选择一个后端服务器，并转发原始 TCP 数据包。它不解析 HTTP 头、URL 或 Cookie，因此性能极高——每秒可处理数百万连接，且延迟开销极小。

### 负载均衡算法

**轮询（Round Robin）**：按顺序将连接分发给各后端。简单、无状态，适用于所有服务器容量相同、请求处理时间相近的场景。

**加权轮询（Weighted Round Robin）**：根据服务器容量分配权重。权重为 3 的服务器获得的连接数是权重为 1 的服务器的 3 倍。

**最少连接（Least Connections）**：将新连接发往当前活跃连接数最少的后端。当请求处理时间差异较大时，效果优于轮询。

**IP 哈希（IP Hash）**：对客户端 IP 做哈希运算，确定性地选择后端。可确保同一客户端始终访问同一服务器（简易版会话保持），但在 NAT 共享 IP 的场景下会失效。

**随机（Random）**：随机选择后端。看似简单，实则高效；当后端数量足够多时，随机分布接近均匀。

**二选一最优（Power of Two Choices）**：随机选取两个后端，再将请求发往连接数更少的那个。仅需极少状态，就能实现近似最优的负载分布。

### 第 4 层在实践中的应用

Linux IPVS（IP Virtual Server）是广泛使用的内核级第 4 层负载均衡器：

```bash
# 安装并配置 IPVS
ipvsadm -A -t 10.0.0.1:80 -s rr           # 添加虚拟服务，启用轮询算法
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.1:80 -m  # 添加后端（NAT 模式）
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.2:80 -m  # 添加后端
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.3:80 -m  # 添加后端
```

云厂商也提供托管式的第 4 层负载均衡器，例如 AWS Network Load Balancer、GCP Network Load Balancer 和 Azure Load Balancer。

## 第 7 层负载均衡（Layer 7 Load Balancing）

第 7 层负载均衡器工作在应用层，能够解析 HTTP 请求，并基于 URL、Header、Cookie 甚至请求体内容进行路由决策。相比第 4 层，它支持更复杂的路由逻辑，但代价是更高的延迟和更低的吞吐量。

### 路由能力

**基于 URL 的路由**：按 URL 路径将请求分发至不同的后端池。

```text
/api/*        → API 服务器池
/static/*     → 静态文件服务器池
/ws/*         → WebSocket 服务器池
```

**基于 Header 的路由**：根据 HTTP Header 字段进行路由。

```text
Host: api.example.com     → API 服务器
Host: www.example.com     → Web 服务器
X-API-Version: v2         → V2 API 服务器
```

**基于 Cookie 的路由**：利用会话 Cookie 实现粘性会话（sticky sessions）。

**基于方法的路由**：将 GET 请求路由至只读副本，POST/PUT/DELETE 请求路由至写入服务器。

### Nginx 作为第 7 层负载均衡器

Nginx 是最广泛使用的第 7 层负载均衡器之一。以下是一个生产级配置示例：

```nginx
# 定义后端服务器池
upstream api_servers {
    least_conn;                    # 使用最少连接算法

    server 10.0.1.1:8080 weight=3;  # 高容量服务器
    server 10.0.1.2:8080 weight=2;
    server 10.0.1.3:8080 weight=1;

    # 健康检查参数
    server 10.0.1.4:8080 backup;   # 仅当其他全部宕机时启用

    keepalive 64;                  # 与后端保持长连接
}

upstream static_servers {
    server 10.0.2.1:80;
    server 10.0.2.2:80;
}

upstream websocket_servers {
    ip_hash;                       # WebSocket 需要粘性会话
    server 10.0.3.1:8080;
    server 10.0.3.2:8080;
}

server {
    listen 443 ssl http2;
    server_name photos.example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # 安全头
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options nosniff;

    # API 请求
    location /api/ {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 10s;

        # 故障重试
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 2;
    }

    # 静态文件
    location /static/ {
        proxy_pass http://static_servers;
        proxy_cache static_cache;
        proxy_cache_valid 200 1d;
        proxy_cache_valid 404 1m;
        add_header X-Cache-Status $upstream_cache_status;
    }

    # WebSocket 连接
    location /ws/ {
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;  # WebSocket 保持 24 小时活跃
    }

    # 健康检查端点
    location /health {
        access_log off;
        return 200 "OK";
        add_header Content-Type text/plain;
    }
}
```

该配置展示了：
- 按 URL 路径划分不同后端池；
- 各池采用不同的负载均衡算法（如 `least_conn`、`ip_hash`）；
- 支持异构服务器的加权配置；
- 对 WebSocket 的连接升级头支持；
- 静态内容的代理缓存；
- 通过 `proxy_next_upstream` 实现自动故障转移。

## 健康检查（Health Checks）

负载均衡器必须探测后端的健康状态，并停止向异常节点转发流量。主要有两类方式。

![健康检查机制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-health-checks.png)

### 主动健康检查（Active Health Checks）

负载均衡器周期性地向各后端发送探针请求，并根据响应判断其健康状况。

```nginx
# Nginx Plus 主动健康检查（商业版）
upstream api_servers {
    zone api_servers 64k;
    server 10.0.1.1:8080;
    server 10.0.1.2:8080;
    server 10.0.1.3:8080;
}

server {
    location /api/ {
        proxy_pass http://api_servers;
        health_check interval=5s fails=3 passes=2;
        # 每 5 秒检查一次
        # 连续失败 3 次标记为不健康
        # 连续成功 2 次标记为健康
    }
}
```

开源版 Nginx 依赖真实流量进行被动检查，而 HAProxy 在免费版本中就支持主动健康检查：

```text
backend api_servers
    option httpchk GET /health
    http-check expect status 200

    server srv1 10.0.1.1:8080 check inter 3s fall 3 rise 2
    server srv2 10.0.1.2:8080 check inter 3s fall 3 rise 2
    server srv3 10.0.1.3:8080 check inter 3s fall 3 rise 2
```

### 被动健康检查（Passive Health Checks）

负载均衡器监控真实的用户请求流量。如果某后端返回过多错误或超时，就会被标记为不健康。

优势：
- 无需额外探针流量；
- 能检测真实请求处理中的失败（不仅限于健康检查端点）。

劣势：
- 需要真实流量才能触发检测（空闲的后端会一直显示为健康）；
- 用户会直接遭遇那些触发检测的失败请求。

### 优雅降级（Graceful Degradation）

当后端被判定为不健康时，应执行以下步骤：

1. **立即移出轮询队列**——停止分发新请求；
2. **排空现有连接**——允许正在进行的请求完成（可配置超时时间）；
3. **透明重试**——若请求是幂等的，可自动在另一后端重试；
4. **恢复后重新加入**——经过连续成功的健康检查后，再将其重新纳入服务。

```nginx
# Nginx 被动健康检查与重试
upstream api_servers {
    server 10.0.1.1:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.2:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.3:8080 max_fails=3 fail_timeout=30s;
}
```

该配置表示：若服务器在 30 秒内连续失败 3 次，即被标记为不可用，并在 30 秒后尝试恢复。

## 全局服务器负载均衡（GSLB）

![负载均衡器流量控制器将请求导向健康的服务器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/02-load-balancer-traffic-controller-directing-requests-to-healt.jpg)

GSLB（Global Server Load Balancing）能在多个地理区域间分发流量，结合 DNS 路由与健康检查，将用户导向最近且健康的区域数据中心。

![全局服务器负载均衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-gslb.png)

### GSLB 架构

典型的 GSLB 部署包含：

1. **DNS 层**：通过 GeoDNS 或 Anycast 将用户导向最近的区域；
2. **区域负载均衡器**：每个区域部署自己的第 4/7 层负载均衡器；
3. **健康监控**：全局健康检查器持续监控所有区域的状态；
4. **故障转移逻辑**：当某区域宕机时，更新 DNS 记录以重定向流量。

多区域部署的数据流示例如下：

```text
东京用户
  → 查询 photos.example.com 的 DNS
  → GeoDNS 返回东京负载均衡器 IP
  → 东京第 7 层 LB 路由至东京应用服务器
  → 应用服务器读取东京数据库副本

若东京区域宕机：
  → 健康检查器发现故障
  → DNS 更新为返回新加坡 LB 的 IP
  → 东京用户被导向新加坡（延迟升高，但服务可用）
```

### 故障转移时效性

区域级故障转移的速度取决于 DNS TTL：

```text
TTL = 300s（5 分钟）：用户最多可能向宕机区域发送请求达 5 分钟  
TTL = 60s（1 分钟）：故障转移更快，但 DNS 查询量增加 60 倍  
TTL = 10s：近乎即时故障转移，但 DNS 查询量极高  
```

正因如此，许多大规模系统对关键服务采用 Anycast 而非 GeoDNS。Anycast 通过 BGP 协议在多个地点广播同一个 IP 地址，网络路由会自动将数据包导向最近的健康节点。这种机制下的故障转移发生在网络层（秒级），而非 DNS 层（分钟级）。

## 对比：第 4 层 vs 第 7 层负载均衡器

| 特性 | 第 4 层（传输层） | 第 7 层（应用层） |
|---------|--------------------|-----------------------|
| 工作层级 | TCP/UDP 数据包 | HTTP 请求 |
| 路由依据 | IP + 端口 | URL、Header、Cookie、请求体 |
| 性能 | 极高（百万级连接/秒） | 高（十万级请求/秒） |
| TLS 终止 | 透传或终止 | 通常终止 |
| 内容检查 | 否 | 是 |
| 基于 URL 的路由 | 否 | 是 |
| 粘性会话 | 仅 IP Hash | Cookie、Header 或 URL 方式 |
| WebSocket 支持 | 透明（纯 TCP） | 需显式支持 |
| 成本 | 较低（逻辑简单） | 较高（计算密集） |
| 典型用例 | TCP 服务、数据库、高吞吐场景 | HTTP API、Web 应用、微服务 |
| 示例 | AWS NLB、IPVS、LVS | Nginx、HAProxy、AWS ALB、Envoy |

实践中，许多架构会同时使用两层：

```text
Internet
  → 第 4 层 LB（NLB）——处理 TLS 终止与原始 TCP 分发
    → 第 7 层 LB（Nginx/Envoy）——处理 HTTP 路由与应用逻辑
      → 应用服务器
```

第 4 层负载均衡器提供高吞吐和 DDoS 防护，第 7 层负载均衡器提供智能路由。二者协同，既能应对海量流量，又能处理复杂业务逻辑。

## 全链路串联

一位伦敦用户访问你的照片分享应用，其请求会依次经过以下组件：

1. **浏览器 DNS 缓存**：检查是否已缓存（< 1ms）  
2. **操作系统 DNS 解析器**：检查系统级缓存（< 1ms）  
3. **递归 DNS 解析器**：若未缓存，则查询权威服务器（20–100ms）  
4. **GeoDNS**：返回最近的 CDN 边缘节点或负载均衡器的 IP  
5. **CDN 边缘节点**：检查请求资源是否在缓存中  
   - **缓存命中**：直接从伦敦边缘节点返回（约 5ms）  
   - **缓存未命中**：回源拉取、缓存、返回（首次约 100–200ms）  
6. **第 4 层负载均衡器**：将 TCP 连接分发至一台第 7 层 LB  
7. **第 7 层负载均衡器**：解析 HTTP 请求，路由至对应的后端池  
8. **应用服务器**：处理请求并返回响应  

对于已缓存的静态资源，总延迟仅为 5–20ms；而对于未缓存的 API 调用，总延迟则在 50–200ms 之间（取决于地理距离和后端处理时间）。这两者之间的巨大差距，正是 CDN 与缓存策略至关重要的原因。

## 下一步

DNS、CDN 与负载均衡将请求送达你的应用。但该请求本身应为何种形态？下一篇文章将探讨 API 设计——REST、gRPC 与 GraphQL——以及决定协议选型的关键权衡。
