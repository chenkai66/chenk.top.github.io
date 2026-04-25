---
title: "Computer Fundamentals: Network, Power, and Troubleshooting"
date: 2022-12-24 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: "Computer Fundamentals"
series_order: 5
series_total: 6
lang: en
mathjax: false
description: "Deep dive into the NIC pipeline (PHY/MAC/DMA), TCP handshake, network topologies, PSU efficiency curves, datacenter PUE, component power hierarchy, and UPS architecture."
disableNunjucks: true
---

Why does the gigabit NIC on your motherboard sometimes negotiate down to 100 Mbps? Why does a brand-new build with a 650 W "Gold" PSU randomly reboot under heavy GPU load? Why does the room next to the server rack always feel warm? These are the everyday consequences of two systems that most people never look at: **the network I/O pipeline** and **the power-and-cooling chain** that keeps the silicon alive.

This is the finale of the Computer Fundamentals series. Instead of repeating component spec tables, we will follow the data and the energy: from the copper pair on the wall, through the PHY, MAC, and DMA engine into RAM; and from the wall socket, through the PSU's conversion stages, into the 12 V rails powering the GPU. Along the way we will look at how a datacenter quantifies waste with PUE, and how a UPS keeps the lights on when the grid falters.

# Series Navigation

**Computer Fundamentals Deep Dive Series** (6 parts):

1. CPU and Computing Core
2. Memory and High-Speed Cache
3. Storage Systems
4. Motherboard, Graphics, and Expansion
5. **Network, Power, and Troubleshooting** *(this article)*
6. Deep Dive: Cross-Cutting Topics

# Part 1: How a NIC Actually Works

A network interface card is not just "the port on the back of the case". Inside that port is a small, dedicated subsystem that converts electrical pulses on a twisted pair into packets that the kernel can route — at line rate, with almost no help from the CPU.

![NIC architecture: PHY, MAC, DMA, and offload features](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig1_nic_architecture.png)

## The three layers inside the silicon

| Stage | What it does | What can go wrong |
|-------|-------------|-------------------|
| **PHY** (Physical Layer) | SerDes, signal conditioning, magnetics. Encodes 1000BASE-T over 4 twisted pairs at 125 MHz. | Bad cabling (CAT5 instead of CAT5e+), kinked patch leads, EMI from a PSU sitting next to the cable. |
| **MAC** (Media Access Control) | Frame assembly, CRC, MAC-address filtering, flow control (PAUSE frames), jumbo frames. | Mismatched MTU between switch and NIC; flow-control storms from a misbehaving peer. |
| **DMA + Ring Buffers** | Transmit and receive descriptor rings; the NIC writes frames straight into a kernel-allocated `skb` (Linux) or `mbuf` (BSD). MSI-X interrupts wake the right CPU core. | Ring overflow (`ifconfig RX-DRP`) when a single core can't drain fast enough — fix with RSS multi-queue. |

Above that sit the **offload engines** that make 25/40/100 GbE practical at all:

- **Checksum offload** — TCP/IP checksums computed in hardware.
- **TSO / LRO** — Transmit Segmentation Offload and Large Receive Offload move per-packet work into silicon. The CPU hands the NIC a 64 KB blob; the NIC produces 44 wire-sized frames.
- **RSS** (Receive Side Scaling) — distributes incoming flows across multiple CPU cores via a Toeplitz hash. Without this, a single 25 GbE flow can saturate one core and stall the whole machine.
- **VLAN tagging** and **SR-IOV** — the latter exposes a single physical NIC as many virtual functions, one per VM, bypassing the hypervisor's software switch.

## The CAT5 vs CAT5e gotcha

The most common "gigabit only runs at 100 Mbps" cause is cabling, not configuration. Gigabit Ethernet (1000BASE-T) needs all four pairs in the cable; the older CAT5 spec only guaranteed two. Patch leads sold cheap on marketplaces often quietly downgrade. The fix is mechanical, not in software:

```bash
# Linux — confirm the negotiated speed
ethtool eth0 | grep -E "Speed|Duplex"
# Speed: 100Mb/s    <-- your cable is the problem
# Duplex: Full

# Force renegotiation after replacing the cable
ethtool -r eth0
```

## Wired NIC standards (2024 reality)

| Standard | Line rate | Real throughput | Where you find it |
|---------|-----------|----------------|------------------|
| **100 Mbps** (Fast Ethernet) | 100 Mb/s | ~12 MB/s | Legacy IoT, old switches |
| **1 Gbps** (Gigabit) | 1 Gb/s | ~118 MB/s | The default everywhere |
| **2.5 Gbps** | 2.5 Gb/s | ~295 MB/s | High-end consumer boards, Wi-Fi 6 backhaul |
| **10 Gbps** (10GBASE-T / SFP+) | 10 Gb/s | ~1.18 GB/s | NAS, workstations, server access ports |
| **25/40/100 Gbps** | up to 100 Gb/s | up to 12.5 GB/s | Datacenter spine and leaf, all SFP28/QSFP28 fiber |

## Wi-Fi standards

| Standard | Bands | Theoretical max | Year |
|---------|-------|----------------|------|
| Wi-Fi 4 (802.11n) | 2.4 / 5 GHz | 600 Mb/s | 2009 |
| Wi-Fi 5 (802.11ac) | 5 GHz | 3.5 Gb/s | 2013 |
| **Wi-Fi 6 (802.11ax)** | 2.4 / 5 GHz | 9.6 Gb/s | 2019 |
| Wi-Fi 6E | 2.4 / 5 / 6 GHz | 9.6 Gb/s, less congested 6 GHz | 2021 |
| Wi-Fi 7 (802.11be) | 2.4 / 5 / 6 GHz | 46 Gb/s, MLO | 2024 |

For a 2024 build, **Wi-Fi 6 or 6E** is the sweet spot: most APs and clients support it, and the 6 GHz band is still relatively quiet.

# Part 2: TCP, the Conversation You Never See

Every web request, every SSH session, every database query begins with a tiny, three-message ritual. Understanding it is the difference between guessing and diagnosing when a connection times out.

![TCP three-way handshake with state transitions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig2_tcp_handshake.png)

## Why three messages, not two?

You might think "client says hello, server says hello back, done". The reason TCP needs a third message is that it must confirm **bidirectional** capability:

1. **SYN** (client → server): "I want to talk; my starting sequence number is `x`."
2. **SYN-ACK** (server → client): "I heard you (`ack = x+1`); my starting sequence number is `y`."
3. **ACK** (client → server): "I heard you too (`ack = y+1`); we are connected."

After step 2, the **server** knows the client can send. After step 3, both ends know both directions work. Drop step 3 and the server is stuck in the `SYN_RCVD` state — exactly the half-open condition exploited by the classic SYN-flood DoS.

## Watching it happen

```bash
# Capture a single handshake on port 443
sudo tcpdump -ni any 'tcp port 443 and (tcp-syn|tcp-ack) != 0' -c 3
```

The three flags-only packets between your machine and the destination are the handshake. After that the conversation switches to a stream of data segments, each carrying its own sequence number for in-order delivery and retransmission.

## Connection-establishment cost

Every new TCP connection spends one round-trip time (RTT) on the handshake before any payload moves. For a server in the same datacenter (RTT = 0.5 ms) that is invisible. For a transatlantic API call (RTT = 100 ms) it is the dominant latency cost. This is why HTTP/2 multiplexes many requests over one connection, and why HTTP/3 (QUIC) folds the TLS handshake into the same round trip.

# Part 3: Network Topologies — Cost vs Resilience

How do you wire 4, 40, or 4000 nodes together? Six classic answers, each with its own bandwidth and failure properties:

![Bus, ring, star, mesh, tree, fat-tree topologies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig3_network_topologies.png)

| Topology | Wires | Failure behavior | Where used today |
|---------|-------|------------------|------------------|
| **Bus** | One shared backbone | One break = total outage | Historical (10BASE2 coax); industrial CAN bus |
| **Ring** | One per neighbor | Token can route around a single break | Token Ring (legacy), SONET, FDDI |
| **Star** | One per node to a hub | Hub failure = total outage | **Today's LAN standard** — every Ethernet switch is a star core |
| **Full Mesh** | n*(n-1)/2 | Maximum redundancy | Small core router groups; rarely used at scale |
| **Tree** | Hierarchical | Upper-layer failure = whole subtree gone | Classic enterprise three-tier (core / aggregation / access) |
| **Fat-Tree** (Clos) | Equal bandwidth at every layer | Many parallel paths; ECMP load balancing | **Hyperscale datacenters** — AWS, Google, Azure all use this |

The fat-tree is the architectural insight that makes hyperscale possible. In a classic tree, the top of the hierarchy is *oversubscribed*: 48 servers feeding into a single 10 Gb uplink contend for 1/48 of their nominal bandwidth. A fat-tree fixes this by making every layer have the same aggregate capacity — you scale by adding more spines, not by buying ever-fatter individual switches.

# Part 4: Network Addressing — The Concepts That Actually Matter

## 127.0.0.1 — the loopback address

127.0.0.1 is the address your machine uses to talk to itself. Packets sent here never touch a physical NIC; they short-circuit inside the kernel's network stack. This is what makes `curl http://localhost:8080` work even with the cable unplugged. The whole `127.0.0.0/8` range is loopback — `127.0.0.1`, `127.5.6.7`, all of it.

## 0.0.0.0 — "any interface"

0.0.0.0 has two roles depending on context:

- As a **bind address** (server side): "listen on every interface I have — eth0, wlan0, lo, all of them." This is what `app.run(host='0.0.0.0')` means in Flask.
- As a **route target**: "the default route", written `0.0.0.0/0` in routing tables.

The classic mistake is treating 0.0.0.0 as something you can `curl` or `ping`. You cannot — it is not a destination, only a wildcard meaning "anywhere".

| Bind to | Reachable from | Typical use |
|---------|---------------|-------------|
| `127.0.0.1` | Same machine only | Local development, security-sensitive services |
| `192.168.1.100` (specific IP) | That NIC only | Multi-homed host serving different content per interface |
| `0.0.0.0` | Any interface | Production servers |

## Ports — service multiplexing

A single IP address can host thousands of services simultaneously because the **port number** (16 bits, 0–65535) selects which one. The first 1024 are "well-known":

| Port | Protocol | Service |
|------|----------|---------|
| 22 | TCP | SSH |
| 53 | UDP | DNS |
| 80 | TCP | HTTP |
| 443 | TCP | HTTPS |
| 3306 | TCP | MySQL |
| 6379 | TCP | Redis |
| 27017 | TCP | MongoDB |

The classic puzzle "ping works but the website doesn't" is solved by remembering that `ping` uses **ICMP** (no ports involved), while HTTP needs a process actively bound to TCP/80. The host is reachable; the service simply is not running, or is bound to `127.0.0.1` instead of `0.0.0.0`.

```bash
# Check which port the service is listening on
ss -tnlp | grep nginx
# LISTEN 0  511   0.0.0.0:80   ...   <-- correct
# LISTEN 0  511   127.0.0.1:80 ...   <-- only local can reach it
```

## NAT — how 4 billion addresses serve 30 billion devices

IPv4 has roughly 4.2 billion addresses. There are far more devices than that. **Network Address Translation** is the trick that makes the math work: every device in your home shares a single public IP, with the router rewriting source addresses on the way out and reverse-mapping responses on the way back.

```
Internal: 192.168.1.100:54321  --SNAT-->  PublicIP:60001  -->  8.8.8.8:53
Internal: 192.168.1.101:33333  --SNAT-->  PublicIP:60002  -->  1.1.1.1:443
```

The router keeps a translation table in memory; entries time out after a few minutes of inactivity. This is why long-idle SSH sessions sometimes silently die — the NAT table forgot them.

## DNS — the address book

Computers route by IP, but humans remember names. DNS is the recursive lookup chain that resolves `www.example.com` to `93.184.216.34`:

1. Browser cache → 2. OS cache (`/etc/hosts` first) → 3. Recursive resolver (your ISP, or `1.1.1.1`) → 4. Root → `.com` TLD → `example.com` authoritative → returns the A record.

CDNs exploit this by returning **different IPs for the same name** depending on the resolver's geography — that is why `www.netflix.com` resolves to a server in your city, not in California.

# Part 5: Virtual Machine Network Modes

When you run a VM under VMware, VirtualBox, or KVM, the hypervisor has to decide how the guest's virtual NIC plugs into the rest of the world. There are three classic options.

## Bridged — VM is a first-class citizen on the LAN

The hypervisor exposes a virtual switch that bridges the guest's NIC directly onto the physical network. The VM gets a DHCP lease from your home router (`192.168.1.x`), the same as the host. Other LAN devices can ping it directly.

Use when: you are running a test web server that colleagues on the same network need to reach.

## NAT — VM lives behind a private virtual router

The hypervisor itself acts as a small router. The VM gets an address in a separate subnet (`192.168.182.x` for VMware), and outbound traffic is double-NAT'd: first by the hypervisor to the host's IP, then by the home router to the public IP.

Use when: you want isolation but still need internet access. **This is the right default for almost every dev VM.**

## Host-only — total isolation

A virtual cable between guest and host with nothing else attached. No internet, no LAN. Useful for malware analysis, air-gapped testing, or running a database that should literally never accept outside connections.

| Feature | Bridged | NAT | Host-only |
|---------|---------|-----|-----------|
| VM IP subnet | Same as host | Separate virtual subnet | Isolated |
| Reachable from LAN | Yes | No (needs port forward) | No |
| Internet access | Yes | Yes | No |
| Consumes a LAN IP | Yes | No | No |
| Typical use | Test servers | Daily development | Malware sandbox |

# Part 6: Network Troubleshooting — Bottom Up

Network problems break down into seven layers, but you debug them in the opposite order from how they're drawn: start at the bottom (cables, link, IP) and work up (TCP, TLS, application).

```
Layer 7  Application      curl works? wrong URL? auth?
Layer 6  Presentation     TLS cert valid? SNI right?
Layer 4  Transport        port open? firewall? service bound to 0.0.0.0?
Layer 3  Network          can I ping the gateway? routing table sane?
Layer 2  Data link        link light on the switch? duplex correct?
Layer 1  Physical         cable plugged in? CAT5e? not crushed?
```

## Case 1: ping works, the website does not

The host responds to ICMP, so layers 1–3 are fine. The break is at layer 4 or above.

```bash
# Is anything bound to port 80?
ss -tnlp | grep ':80 '

# Can I reach it from the network?
nc -zv example.com 80

# What does the service log say?
journalctl -u nginx -n 50
```

Most common causes: nginx isn't running; nginx is running but bound to `127.0.0.1`; a firewall rule blocks port 80; the AWS security group hasn't allowed inbound HTTP.

## Case 2: domain does not resolve

```bash
ping 8.8.8.8                  # confirms IP-level connectivity
dig www.example.com           # try DNS directly
cat /etc/resolv.conf          # is the resolver list sane?
```

If `8.8.8.8` works but DNS does not, your configured resolver is broken. Drop in `1.1.1.1` or `8.8.8.8` and try again.

## Case 3: VM has no internet (NAT mode)

```bash
ip addr           # did DHCP assign an IP in the virtual subnet?
ip route          # is the default route the virtual gateway (e.g. .2)?
ping 192.168.182.2  # virtual NAT gateway
ping 8.8.8.8        # external by IP — bypasses DNS
ping example.com    # if this fails but the line above works -> DNS
```

The most common fixes are: restart the VMware NAT service on the host, or replace `/etc/resolv.conf` with `nameserver 8.8.8.8`.

## The hosts file — local DNS override

`/etc/hosts` (or `C:\Windows\System32\drivers\etc\hosts`) is checked **before** any DNS query. It is a flat text file mapping IPs to names:

```
127.0.0.1     dev.example.com
127.0.0.1     api.example.com
0.0.0.0       ads.tracker.example
```

The first two lines route local development domains to your machine. The third is the simplest ad-blocker: requests to those names go to `0.0.0.0` and silently fail. After editing, flush the OS DNS cache:

```bash
# macOS
sudo dscacheutil -flushcache && sudo killall -HUP mDNSResponder
# Windows
ipconfig /flushdns
# Linux (systemd-resolved)
sudo systemd-resolve --flush-caches
```

# Part 7: PSU — The Most Underestimated Component

The power supply is the only component that touches every other component. A flaky PSU produces symptoms that look like RAM faults, GPU faults, motherboard faults — anything but power. And yet most build guides treat it as a wattage number.

## What "80 PLUS" actually measures

A PSU's nameplate wattage tells you what it can deliver to the system. Its **efficiency** tells you how much it wastes converting wall AC to the 12 V / 5 V / 3.3 V rails. The "80 PLUS" certification grades that conversion at four load points.

![80 PLUS efficiency curves across certification levels](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig4_psu_efficiency.png)

A few non-obvious things fall out of these curves:

- **Efficiency peaks at 40–70 % load.** Run a 200 W system on a 1000 W PSU and you sit at 20 % load — the inefficient end of the curve. Sizing too big costs you money, not just the upfront price.
- **Titanium is only meaningfully better at the extremes.** At 50 % load Gold (90 %) and Titanium (94 %) differ by 4 percentage points. At 10 % load — typical for a server idle overnight — Titanium pulls ahead.
- **The dollar value is real but bounded.** A 400 W PC running 8 h/day with $0.12/kWh electricity wastes about 56 W more on a "white" 80 PLUS than on Gold. That is roughly $50/year. Gold typically costs $20–30 more, and lasts 7–10 years; the math is easy.

## Sizing a PSU — the right way

The standard rule of thumb is **(CPU TDP + GPU TDP + 100 W) × 1.3**. The 1.3 multiplier covers transient spikes (modern GPUs can briefly draw 30 % above their nominal TDP), efficiency loss, and headroom for landing in the sweet-spot region of the curve.

| Build class | Typical sustained draw | Recommended PSU |
|-------------|----------------------|----------------|
| Office (no dGPU) | 80–120 W | 350–450 W, Bronze |
| Light gaming (RTX 3050) | 200–280 W | 500 W, Gold |
| Mainstream (i5-13600K + RTX 4070) | 350–450 W | **650 W, Gold** ✅ |
| High-end (i9 + RTX 4080) | 550–700 W | 850 W, Gold/Platinum |
| Workstation (i9 + RTX 4090) | 800–1100 W | 1200 W, Platinum |

## Modular vs non-modular

Three flavors:

- **Non-modular** — all cables permanently attached. Cheapest, ugliest cable management.
- **Semi-modular** — the always-needed 24-pin and CPU 8-pin are fixed; PCIe and SATA cables detach. **Best balance for most builds.**
- **Fully modular** — every cable detaches. Pay $20–30 more for cleaner SFF builds and easier replacements.

A safety note that comes up often: **never mix modular cables between PSU brands or models.** The PSU-side connector pinout is not standardized; using an EVGA cable on a Corsair PSU can short the 12 V rail through the 5 V circuit and destroy the entire system.

# Part 8: Component Power Hierarchy

When you understand which components actually dominate the power budget, sizing decisions become obvious.

![Idle vs full-load power for each component](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig6_power_hierarchy.png)

The two takeaways:

1. **The CPU and GPU together are 80–95 % of peak draw.** Everything else — RAM, SSDs, motherboard, fans — is rounding error. Size the PSU around CPU TDP + GPU TDP and add 100 W for the rest.
2. **Idle vs load ratio is enormous, especially for GPUs.** An RTX 4090 idles at ~18 W and peaks at 450 W — a 25× swing. This is why fan noise correlates with workload, and why undervolting a GPU has a much bigger impact than undervolting a CPU.

A useful diagnostic when a system reboots under load: if it always happens during specific game scenes or rendering, suspect the PSU's transient response, not its average wattage. Modern GPUs spike to 1.3–1.5× TDP for milliseconds; a borderline PSU can hold the average but trip the over-current protection on the spike.

# Part 9: Datacenter PUE — Where Every Watt Goes

A single server room is just a bigger version of the same problem: how much of the power going through the meter actually reaches the silicon? The industry metric is **PUE** (Power Usage Effectiveness):

$$ \text{PUE} = \frac{\text{Total facility power}}{\text{IT equipment power}} $$

A PUE of 1.0 is the theoretical perfect: every watt drawn from the grid ends up doing computation. Reality looks like this:

![Datacenter PUE breakdown and operator comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig5_datacenter_pue.png)

The breakdown of a typical PUE = 1.5 datacenter:

- **67 %** to IT equipment — servers, storage, networking. The thing you actually wanted.
- **25 %** to cooling — chillers, CRAC units, fans. The bigger the heat load, the bigger this slice.
- **5 %** to power conversion — UPS losses, PDU losses, transformer losses.
- **3 %** to lighting, security, and miscellaneous building load.

Hyperscalers have driven this dramatically lower. Google's 2024 fleet-wide trailing-12-month PUE is **1.10**, and the best-in-class facilities (immersion cooling, no chillers) approach **1.03**. The improvements come from:

- **Higher inlet temperatures** — running the cold aisle at 27 °C instead of 18 °C halves the cooling load.
- **Free cooling** — using outside air (or seawater, in northern facilities) instead of compression chillers most of the year.
- **Better airflow management** — hot/cold aisle containment, blanking panels, sealed floor tiles.
- **Higher-voltage distribution** — 415 V three-phase to the rack instead of 208 V cuts conversion losses.
- **Liquid and immersion cooling** — eliminating air as a heat-transfer medium for the densest GPU racks.

PUE is also why the geographic location of a hyperscale datacenter matters: Iceland, Oregon, and northern Sweden enable nearly free cooling for most of the year.

# Part 10: UPS — Keeping the Lights On

A single grid blink — even 30 milliseconds — is enough to crash an unprotected server. The Uninterruptible Power Supply is the energy buffer that bridges the gap between grid failure and the diesel generator coming online.

![Online UPS architecture with bypass, battery, and generator paths](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/05-network-power/fig7_ups_backup.png)

## Three UPS topologies

- **Standby (offline)** — load is normally fed straight from the wall. On a sag or outage, a relay flips to the inverter. Transfer time: 4–10 ms. Cheap; OK for a single home PC.
- **Line-interactive** — adds an autotransformer that can correct moderate sag/swell without going to battery. Transfer time: 2–4 ms. Common in small server rooms.
- **Online (double-conversion)** — load is *always* on the inverter. AC → DC (rectifier) → DC bus → DC → AC (inverter). Battery floats on the DC bus. **Transfer time: 0 ms** — the load never even notices the grid is gone. Standard for datacenters.

## The full power chain in a real datacenter

1. **Utility AC** comes in from the grid (with a backup feed from a separate substation if you are paranoid).
2. **ATS** (Automatic Transfer Switch) sits between utility and generator.
3. **Diesel generator** spins up in 10–30 seconds when the ATS senses utility loss.
4. **Online UPS** carries the load during those 10–30 seconds. Modern designs use lithium-ion (3× the cycle life of VRLA, half the footprint).
5. **PDU** (Power Distribution Unit) splits a high-current feed into per-rack circuits.
6. **Server PSUs** — usually two, on independent feeds (A side / B side), so a single UPS or PDU failure does not take the rack down.

This is why a hyperscale site has **multiple layers of N+1 redundancy**: utility, generator, UPS, PDU, server PSU. A single failure at any layer is invisible. Two simultaneous failures at the same layer are survivable. The only true outages happen when the failures cascade across layers — almost always due to operator error, not equipment failure.

## How long does the battery last?

Runtime is a function of **load** and **battery bank size**, not of UPS rating:

- 5 kVA UPS at full 5 kW load with a small internal battery: 5–10 minutes.
- The same UPS at half load: ~25 minutes (battery discharge is non-linear).
- External battery cabinets can extend this to hours, but are usually sized only for "long enough for the generator to start and stabilize" — typically 15 minutes is the design target.

# Part 11: Common Hardware Faults — A Triage Guide

After ten years of fixing other people's machines, the same handful of failure modes account for nearly every problem.

## Boot failures

| Symptom | First thing to try | If that fails |
|--------|------------------|--------------|
| No power at all | Wall outlet, PSU rocker switch, 24-pin seated | Paperclip-test the PSU (jump green to any black) |
| Fans spin, no display | **Reseat the RAM** — fixes ~80 % of these | Clear CMOS, try one stick at a time, try integrated graphics |
| POST beeps | Look up the beep code in the motherboard manual | Different boards mean different codes |
| Boots, then BSOD/kernel panic immediately | Check CPU temperature (could be an unmounted cooler) | MemTest86 overnight, SMART check on the boot drive |
| Boots, then reboots under load | PSU undersized or aging; GPU power spikes | Test with a known-good higher-wattage PSU |

The "reseat the RAM" reflex feels superstitious but is grounded in real physics: thermal cycling slowly walks DIMMs out of their slots, and the gold contacts oxidize. Removing and reinserting wipes the contact patch and restores a low-resistance connection.

## Storage and GPU

- **SSD suddenly slow** — check free space (under 10 % free triggers garbage-collection thrashing on most consumer drives), confirm TRIM is active, look at the SMART "Wear Leveling Count" attribute.
- **GPU artifacts** — usually overheating or power-delivery instability. Reseat the PCIe power connectors; underclock the memory by 200 MHz to test.
- **GPU not detected** — wrong PCIe slot (use the top x16, not a chipset-attached slot), missing power connector, or a dead riser cable in fancy cases.
- **M.2 NVMe missing** — motherboard's M.2 slot may share lanes with a SATA port; check the manual for "shared lane" warnings.

## Network

- **Disconnects** — check link partner, replace the patch cable, update the NIC driver.
- **Wi-Fi slow** — switch from 2.4 GHz to 5 GHz, change channel (use a Wi-Fi analyzer to find a quiet one), check for microwave/Bluetooth interference.
- **Cannot get DHCP** — confirm the link light, check `ip addr` for an APIPA address (`169.254.x.x`), restart the DHCP client.

# Series Recap

What this series covered, in five sentences:

1. The CPU is the bottleneck for serial work; the GPU for parallel work; everything else is feeding them.
2. Memory hierarchy (registers → L1/L2/L3 → DRAM → NVMe → HDD) spans seven orders of magnitude in latency, and good code respects locality.
3. Storage is bandwidth and IOPS *and* endurance — the right answer depends on the workload.
4. Buses (PCIe), interfaces (USB, Thunderbolt), and the motherboard's VRM define what the silicon can actually deliver.
5. Networking and power are the two systems most builders never look at, and the two that produce the weirdest failure modes.

If you take one thing away: **the bottleneck is rarely where you first look**. Profile, measure, and trust the numbers — not the spec sheet.

The deep-dive part (Part 6) returns to the questions this series only sketched: cache-coherence protocols, NUMA, virtualization at the silicon level, and the energy-efficiency frontier.
