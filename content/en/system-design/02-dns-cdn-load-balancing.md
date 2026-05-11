---
title: "System Design (2): DNS, CDN, and Load Balancing — The First Three Hops"
date: 2025-07-12 09:00:00
tags:
  - System Design
  - Networking
  - Load Balancing
  - CDN
categories:
  - System Design
series: system-design
lang: en
description: "Every web request begins with DNS resolution, may traverse a CDN edge, and lands on a load balancer before reaching your application. Understanding these three hops is essential to building systems that are fast, reliable, and globally distributed."
disableNunjucks: true
series_order: 2
translationKey: "system-design-2"
---

In 2017, a single misconfigured DNS record at a major cloud provider took down a significant portion of the internet for several hours. Thousands of websites became unreachable — not because their servers were down, but because the system that translates domain names into IP addresses stopped working correctly. The incident was a stark reminder that the infrastructure we take for granted — DNS, CDN, load balancers — is the foundation everything else rests on.

Every HTTP request your users make passes through at least two of these systems before it reaches your application code. If any of them fails or performs poorly, nothing downstream matters.

## DNS Resolution

The Domain Name System is a distributed, hierarchical database that maps human-readable domain names to IP addresses. When a user types `photos.example.com` into their browser, a cascade of lookups happens before a single byte of your application code executes.

![DNS resolution flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-dns-resolution.png)


### The Resolution Process

DNS resolution involves two query styles: recursive and iterative.

**Recursive resolution** is what your browser does. It sends a query to a recursive resolver (typically your ISP's DNS server or a public resolver like 8.8.8.8) and expects a complete answer. The resolver does all the work.

**Iterative resolution** is what the recursive resolver does internally. It queries a chain of authoritative servers, each of which either provides the answer or refers the resolver to a more specific server.

The full chain for resolving `photos.example.com`:

1. Browser checks its local DNS cache
2. OS checks its DNS cache (`/etc/hosts`, then system resolver cache)
3. Query goes to the configured recursive resolver
4. Resolver queries a root nameserver: "Who handles `.com`?"
5. Root nameserver responds with the `.com` TLD nameserver addresses
6. Resolver queries the `.com` TLD nameserver: "Who handles `example.com`?"
7. TLD nameserver responds with the authoritative nameserver for `example.com`
8. Resolver queries the authoritative nameserver: "What is the IP for `photos.example.com`?"
9. Authoritative nameserver responds with the IP address
10. Resolver caches the result and returns it to the client

This entire chain typically completes in 20-120ms for uncached queries. Cached queries resolve in under 1ms.

### DNS Record Types

| Record Type | Purpose | Example |
|------------|---------|---------|
| A | Maps name to IPv4 address | `photos.example.com → 93.184.216.34` |
| AAAA | Maps name to IPv6 address | `photos.example.com → 2606:2800:220:1:...` |
| CNAME | Alias one name to another | `www.example.com → example.com` |
| MX | Mail exchange servers | `example.com → mail.example.com (priority 10)` |
| NS | Authoritative nameservers | `example.com → ns1.example.com` |
| TXT | Arbitrary text (SPF, DKIM, verification) | `example.com → "v=spf1 include:..."` |
| SRV | Service location (host + port) | `_sip._tcp.example.com → sipserver.example.com:5060` |
| PTR | Reverse lookup (IP to name) | `34.216.184.93 → photos.example.com` |

### TTL and Caching

Every DNS record has a TTL (Time To Live) in seconds. When a resolver caches a record, it honors the TTL before re-querying.

```
; Example DNS zone file entries
photos.example.com.   300   IN  A      93.184.216.34
photos.example.com.   300   IN  A      93.184.216.35
cdn.example.com.      3600  IN  CNAME  d111111abcdef8.cloudfront.net.
```

TTL trade-offs:
- **Short TTL (30-300 seconds)**: Faster failover, more DNS queries, higher resolver load
- **Long TTL (3600-86400 seconds)**: Fewer queries, slower failover, better performance
- **During migration**: Set TTL low (60s) days before the change, then switch records, then raise TTL back

### DNS-Based Load Balancing

DNS can distribute traffic across multiple servers by returning different IP addresses.

**Round Robin DNS**: Return multiple A records. Clients pick one (usually the first). Simple but provides no health checking — DNS will happily return the IP of a dead server.

```
photos.example.com.  300  IN  A  10.0.1.1
photos.example.com.  300  IN  A  10.0.1.2
photos.example.com.  300  IN  A  10.0.1.3
```

**Weighted DNS**: Return different records with different probabilities. AWS Route 53 and similar services support this.

**GeoDNS**: Return different IP addresses based on the geographic location of the resolver. A user in Tokyo gets the IP of your Tokyo datacenter; a user in London gets your Frankfurt datacenter.

GeoDNS is the foundation of global load balancing, but it has limitations:
- Location is determined by the resolver's IP, not the user's IP (VPN users get wrong results)
- DNS caching means changes propagate slowly
- No real-time health awareness unless combined with health-checking DNS services

## Content Delivery Networks

A CDN is a globally distributed network of proxy servers that cache content close to end users. When a user in Sydney requests an image from your server in Virginia, the CDN serves it from an edge server in Sydney instead.

### How CDN Caching Works

The basic flow:

![CDN edge caching topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-cdn-topology.png)


1. User requests `https://photos.example.com/img/abc123.jpg`
2. DNS resolves `photos.example.com` to the nearest CDN edge server (via GeoDNS or anycast)
3. Edge server checks its local cache for the object
4. **Cache hit**: Return the object directly (latency: 5-20ms)
5. **Cache miss**: Edge server fetches from the origin server, caches it, returns to user

### Origin Pull vs Origin Push

**Origin Pull (lazy loading)**: The CDN fetches content from your origin server on the first request (cache miss), then caches it. This is the default model for most CDNs.

Advantages:
- Simple setup — just point DNS to CDN
- Only caches content that is actually requested
- No need to pre-populate the cache

Disadvantages:
- First request for each object is slow (origin fetch)
- Origin server must handle cache miss traffic
- Thundering herd on cache expiration of popular objects

**Origin Push (proactive)**: You upload content directly to the CDN's storage. Used for large files, video content, and software downloads.

Advantages:
- No cache miss latency for users
- Origin server never needs to serve the content
- Better for large files that are expensive to transfer

Disadvantages:
- Requires integration with CDN's upload API
- You manage cache lifecycle explicitly
- Storage costs on the CDN side

### Cache Invalidation at the CDN

CDN cache invalidation is notoriously difficult. Common strategies:

**TTL-based**: Set Cache-Control headers on your responses.

```
Cache-Control: public, max-age=31536000  # 1 year for immutable assets
Cache-Control: public, max-age=300       # 5 minutes for semi-dynamic content
Cache-Control: no-store                  # Never cache (fully dynamic)
```

**Versioned URLs**: Append a version or hash to the URL. When content changes, the URL changes, so old cached versions are never served.

```
/static/app.a1b2c3d4.js    # Hash in filename
/img/photo.jpg?v=20250712  # Version query parameter
```

**Purge API**: Most CDNs offer an API to explicitly invalidate cached objects. Use this sparingly — it can take 5-30 seconds to propagate globally.

```bash
# CloudFront invalidation
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/img/abc123.jpg" "/api/feed/*"
```

### When CDN Helps and When It Hurts

**CDN helps when**:
- Content is static or semi-static (images, CSS, JS, videos)
- Users are geographically distributed
- Read-to-write ratio is high
- Content is shared across many users (same image served to millions)

**CDN hurts when**:
- Content is personalized per user (user dashboards, account pages)
- Content changes very frequently (real-time data)
- Content is accessed very rarely (long-tail content with low hit rates)
- You need strong consistency (CDN caches may serve stale data)

### CDN Architecture

A major CDN provider operates hundreds of Points of Presence (PoPs) across dozens of countries. Each PoP contains:

- **Edge servers**: Cache and serve content, handle TLS termination
- **Regional caches (mid-tier)**: Larger caches that sit between edge servers and origin, reducing origin load
- **Routing infrastructure**: Anycast IP addresses or GeoDNS to direct users to nearest PoP

The tiered caching architecture is important. Without mid-tier caches, every edge server would independently fetch cache misses from the origin. With mid-tier caches, only one fetch per region reaches the origin.

## Layer 4 Load Balancing


![Dns resolution journey message traveling through hierarchica](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/02-dns-resolution-journey-message-traveling-through-hierarchica.jpg)

Layer 4 load balancers operate at the transport layer (TCP/UDP). They make routing decisions based on IP addresses and port numbers without inspecting the application-layer payload.

![L4 vs L7 load balancing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-l4-vs-l7.png)


### How It Works

A Layer 4 load balancer receives a TCP connection, selects a backend server, and forwards the raw TCP packets. It does not parse HTTP headers, URLs, or cookies. This makes it extremely fast — it can handle millions of connections per second with minimal latency overhead.

### Load Balancing Algorithms

**Round Robin**: Distribute connections sequentially across backends. Simple, stateless, works well when all servers have equal capacity and all requests have similar cost.

**Weighted Round Robin**: Assign weights to backends based on capacity. A server with weight 3 gets 3x the connections of a server with weight 1.

**Least Connections**: Send new connections to the backend with the fewest active connections. Better than round robin when request processing times vary widely.

**IP Hash**: Hash the client IP to deterministically select a backend. Ensures the same client always reaches the same server (poor man's session affinity). Breaks when clients are behind a NAT that shares one IP.

**Random**: Randomly select a backend. Surprisingly effective and very simple. With enough backends, random selection approximates even distribution.

**Power of Two Choices**: Randomly pick two backends, then send the request to the one with fewer connections. Provides near-optimal distribution with minimal state.

### Layer 4 in Practice

Linux IPVS (IP Virtual Server) is a widely-used kernel-level Layer 4 load balancer:

```bash
# Install and configure IPVS
ipvsadm -A -t 10.0.0.1:80 -s rr           # Add virtual service with round robin
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.1:80 -m  # Add backend (masquerading/NAT mode)
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.2:80 -m  # Add backend
ipvsadm -a -t 10.0.0.1:80 -r 10.0.1.3:80 -m  # Add backend
```

Cloud providers offer managed Layer 4 load balancers: AWS Network Load Balancer, GCP Network Load Balancer, Azure Load Balancer.

## Layer 7 Load Balancing

Layer 7 load balancers operate at the application layer. They parse HTTP requests and make routing decisions based on URLs, headers, cookies, and request content. This enables far more sophisticated routing than Layer 4, at the cost of higher latency and lower throughput.

### Routing Capabilities

**URL-based routing**: Route requests to different backend pools based on the URL path.

```
/api/*        → API server pool
/static/*     → Static file server pool
/ws/*         → WebSocket server pool
```

**Header-based routing**: Route based on HTTP headers.

```
Host: api.example.com     → API servers
Host: www.example.com     → Web servers
X-API-Version: v2         → V2 API servers
```

**Cookie-based routing**: Route based on session cookies for sticky sessions.

**Method-based routing**: Route GET requests to read replicas, POST/PUT/DELETE to write servers.

### Nginx as a Layer 7 Load Balancer

Nginx is one of the most widely-used Layer 7 load balancers. Here is a production-grade configuration.

```nginx
# Define backend server pools
upstream api_servers {
    least_conn;                    # Use least connections algorithm

    server 10.0.1.1:8080 weight=3;  # Higher capacity server
    server 10.0.1.2:8080 weight=2;
    server 10.0.1.3:8080 weight=1;

    # Health check parameters
    server 10.0.1.4:8080 backup;   # Only used when all others are down

    keepalive 64;                  # Persistent connections to backends
}

upstream static_servers {
    server 10.0.2.1:80;
    server 10.0.2.2:80;
}

upstream websocket_servers {
    ip_hash;                       # Sticky sessions for WebSocket
    server 10.0.3.1:8080;
    server 10.0.3.2:8080;
}

server {
    listen 443 ssl http2;
    server_name photos.example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options nosniff;

    # API requests
    location /api/ {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 10s;

        # Retry on failure
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 2;
    }

    # Static files
    location /static/ {
        proxy_pass http://static_servers;
        proxy_cache static_cache;
        proxy_cache_valid 200 1d;
        proxy_cache_valid 404 1m;
        add_header X-Cache-Status $upstream_cache_status;
    }

    # WebSocket connections
    location /ws/ {
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;  # Keep WebSocket alive for 24h
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "OK";
        add_header Content-Type text/plain;
    }
}
```

This configuration demonstrates:
- Different backend pools for different URL paths
- Per-pool load balancing algorithms (least_conn, ip_hash)
- Weighted backends for heterogeneous server capacities
- WebSocket support with connection upgrade headers
- Proxy caching for static content
- Automatic failover with `proxy_next_upstream`

## Health Checks

Load balancers must detect unhealthy backends and stop sending traffic to them. There are two approaches.

![Health check mechanisms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-health-checks.png)


### Active Health Checks

The load balancer periodically sends probe requests to each backend and evaluates the response.

```nginx
# Nginx Plus active health checks (commercial version)
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
        # Check every 5 seconds
        # Mark unhealthy after 3 failures
        # Mark healthy after 2 successes
    }
}
```

For open-source Nginx, health checks rely on real traffic (passive checks). HAProxy offers active health checks in its free version:

```
backend api_servers
    option httpchk GET /health
    http-check expect status 200

    server srv1 10.0.1.1:8080 check inter 3s fall 3 rise 2
    server srv2 10.0.1.2:8080 check inter 3s fall 3 rise 2
    server srv3 10.0.1.3:8080 check inter 3s fall 3 rise 2
```

### Passive Health Checks

The load balancer monitors actual request traffic. If a backend returns too many errors or times out, it is marked unhealthy.

Advantages:
- No extra probe traffic
- Detects failures in real request handling (not just health endpoint)

Disadvantages:
- Requires real traffic to detect failures (idle backends appear healthy)
- Users experience the failed requests that trigger the detection

### Graceful Degradation

When a backend is detected as unhealthy:

1. **Remove from rotation immediately** — stop sending new requests
2. **Drain existing connections** — let in-flight requests complete (configurable timeout)
3. **Retry on another backend** — if the request is idempotent, retry transparently
4. **Re-add when healthy** — after consecutive successful health checks

```nginx
# Nginx passive health check with retry
upstream api_servers {
    server 10.0.1.1:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.2:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.3:8080 max_fails=3 fail_timeout=30s;
}
```

This configuration marks a server as unavailable after 3 failures within 30 seconds, and tries it again after 30 seconds.

## Global Server Load Balancing (GSLB)


![Load balancer traffic controller directing requests to healt](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/system-design/02-load-balancer-traffic-controller-directing-requests-to-healt.jpg)

GSLB distributes traffic across multiple geographic regions. It combines DNS-based routing with health checking to direct users to the closest healthy datacenter.

![Global server load balancing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/system-design/02-gslb.png)


### GSLB Architecture

A typical GSLB setup:

1. **DNS layer**: GeoDNS or anycast routing directs users to the nearest region
2. **Regional load balancers**: Each region has its own Layer 4/7 load balancers
3. **Health monitoring**: A global health checker monitors all regions
4. **Failover logic**: If a region goes down, DNS is updated to redirect traffic

The data flow for a multi-region deployment:

```
User in Tokyo
  → DNS query for photos.example.com
  → GeoDNS returns IP of Tokyo load balancer
  → Tokyo L7 LB routes to Tokyo app server
  → App server reads from Tokyo database replica

If Tokyo region fails:
  → Health checker detects failure
  → DNS updated to return Singapore LB IP
  → Tokyo users directed to Singapore (higher latency, but available)
```

### Failover Timing

The speed of regional failover depends on DNS TTL:

```
TTL = 300s (5 min): Users may hit dead region for up to 5 minutes
TTL = 60s (1 min):  Faster failover, but 60x more DNS queries
TTL = 10s:          Near-instant failover, very high DNS query volume
```

This is why many large-scale systems use anycast instead of GeoDNS for critical services. With anycast, the same IP address is advertised from multiple locations via BGP. Network routing automatically directs packets to the nearest healthy location, with failover happening at the network layer (seconds) rather than the DNS layer (minutes).

## Comparison: Layer 4 vs Layer 7 Load Balancers

| Feature | Layer 4 (Transport) | Layer 7 (Application) |
|---------|--------------------|-----------------------|
| Operates on | TCP/UDP packets | HTTP requests |
| Routing decisions | IP + port | URL, headers, cookies, body |
| Performance | Very high (millions of conn/sec) | High (hundreds of thousands req/sec) |
| TLS termination | Pass-through or terminate | Typically terminates |
| Content inspection | No | Yes |
| URL-based routing | No | Yes |
| Sticky sessions | IP hash only | Cookie, header, or URL-based |
| WebSocket support | Transparent (just TCP) | Requires explicit support |
| Cost | Lower (simpler logic) | Higher (more processing) |
| Use cases | TCP services, databases, high-volume | HTTP APIs, web apps, microservices |
| Examples | AWS NLB, IPVS, LVS | Nginx, HAProxy, AWS ALB, Envoy |

In practice, many architectures use both layers:

```
Internet
  → Layer 4 LB (NLB) — handles TLS termination and raw TCP distribution
    → Layer 7 LB (Nginx/Envoy) — handles HTTP routing and application logic
      → Application servers
```

The Layer 4 balancer provides high throughput and DDoS protection. The Layer 7 balancer provides intelligent routing. Together they handle both volume and complexity.

## Putting It All Together

A request from a user in London to your photo-sharing application traverses these components:

1. **Browser DNS cache**: Check for cached resolution (< 1ms)
2. **OS DNS resolver**: Check system cache (< 1ms)
3. **Recursive DNS resolver**: Query authoritative servers if not cached (20-100ms)
4. **GeoDNS**: Return IP of nearest CDN edge or load balancer
5. **CDN edge**: Check cache for the requested resource
   - **Cache hit**: Return directly from edge server in London (5ms)
   - **Cache miss**: Fetch from origin, cache, return (100-200ms first time)
6. **Layer 4 load balancer**: Distribute TCP connection to a Layer 7 LB
7. **Layer 7 load balancer**: Parse HTTP request, route to appropriate backend pool
8. **Application server**: Process the request and return response

For a cached static asset, total latency is 5-20ms. For an uncached API call, total latency is 50-200ms depending on geography and backend processing time. The gap between these two numbers is why CDN and caching strategy matter so much.

## What's Next

DNS, CDN, and load balancing get the request to your application. But what shape should that request take? The next article covers API design — REST, gRPC, and GraphQL — and the trade-offs that determine which protocol fits your system.
