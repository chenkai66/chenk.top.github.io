---
title: "Computer Fundamentals: Memory and Cache Systems"
date: 2022-10-22 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: "Computer Fundamentals"
series_order: 2
series_total: 6
lang: en
mathjax: false
description: "How memory really works: hierarchy, DRAM vs SRAM, virtual memory, the TLB, ECC, NUMA, and channel scaling — explained from circuits to system design."
disableNunjucks: true
---

A CPU core can complete a multiplication in roughly **0.3 ns**. A spinning hard disk needs **10 ms** to seat its head over a sector. Between those two numbers sits a factor of about **30 million**. Every line of memory engineering — caches, DRAM cells, page tables, TLBs, ECC, NUMA, channels — is a coordinated answer to that single, brutal asymmetry.

This is part 2 of the **Computer Fundamentals Deep Dive**. We will not stop at "DDR is fast and RAM is volatile". We will trace a single load instruction from the CPU pipeline through the L1, L2, L3 caches, the TLB, the page table, the memory controller, the channels, and finally the DRAM cells themselves — and look at what each layer is actually doing, and why.

# Series Navigation

**Computer Fundamentals Deep Dive** (6 parts):

1. CPU & Computing Core
2. **→ Memory & Cache Systems** ← *you are here*
3. Storage Systems (HDD, SSD, NVMe, RAID)
4. Motherboard, GPU & Expansion (PCIe, USB, BIOS)
5. Network, Power & Practical Troubleshooting
6. Putting It Together: A System-Level Deep Dive

# 1. The Memory Hierarchy: Why a Single "RAM" Is Not Enough

If a single technology could be both fast and cheap, computers would have one big block of it and we would be done. No such technology exists. SRAM is fast but expensive and large per bit. DRAM is dense but slow. NAND flash is dense and persistent but slower still. Spinning disks are cheap and huge but mechanical. The memory hierarchy is the engineering compromise.

![Memory hierarchy: speed vs capacity trade-off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig1_memory_hierarchy.png)

A useful rule of thumb when you read this pyramid: **each level is roughly 10x larger and 10x slower than the one above it**. A CPU register is ~0.3 ns away. L1 is ~1 ns. L2 is ~4 ns. L3 is ~15 ns. DRAM is ~100 ns. NVMe SSD is ~100 µs. A spinning HDD is ~10 ms.

Translate those numbers into something tangible. If a register access takes one second, then:

| Layer  | Real time | If 1 register access = 1 second |
|--------|-----------|----------------------------------|
| Register | 0.3 ns | 1 second |
| L1 cache | 1 ns   | 3 seconds |
| L2 cache | 4 ns   | 13 seconds |
| L3 cache | 15 ns  | 50 seconds |
| DRAM     | 100 ns | 5 minutes |
| NVMe SSD | 100 µs | 4 days |
| HDD      | 10 ms  | **1 year** |

Every cache miss that goes to DRAM is the difference between a 1-second decision and a 5-minute coffee break. Every page fault to disk is the difference between 1 second and **a year**. This is why "just add RAM" or "just buy a faster SSD" is not the right mental model — what matters is **where in the hierarchy your working set actually lives**.

# 2. DRAM vs SRAM: Why Your CPU Has Tiny Caches

Both DRAM and SRAM store bits with transistors and voltages. The difference is in the cell — and the cell determines everything else: density, cost, speed, power, even why DRAM has to be refreshed.

![DRAM vs SRAM cell structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig2_dram_vs_sram.png)

**DRAM = 1 transistor + 1 capacitor (1T1C).** A bit is stored as charge on a tiny capacitor. To write, you raise the word line, drive the bit line, and let charge in or out. To read, you raise the word line and a sense amplifier detects whether the capacitor was charged. A DRAM cell is microscopic — about 6 F² in modern processes, where F is the minimum feature size. That is why a single DDR5 chip can hold 16 Gb in a few square millimetres.

The price of that density is two ugly facts:

1. **Capacitors leak.** The stored charge bleeds away in milliseconds. Every cell has to be **refreshed** about every 64 ms — read and re-written, just to remember what it already knew. Refresh costs both bandwidth and power.
2. **Reading is destructive.** The sense amplifier draws charge off the capacitor to detect the value, so the value has to be written back after every read.

**SRAM = 6 transistors in a cross-coupled latch (6T).** A bit is stored as the stable state of two inverters feeding each other. As long as power is on, the latch holds — no refresh, no destructive read, sub-nanosecond access. The cost is **20× the area per bit**. That is why your CPU has 64 KB of L1 cache per core and not 64 GB.

This single trade-off explains the entire memory hierarchy. CPUs use SRAM where it has to be fast (registers, L1, L2, L3). DRAM lives one level out, where density wins. Flash and disk live further out still, where persistence and capacity dominate.

# 3. The CPU Cache: Three Levels of Bridge

Even DRAM at 100 ns is **300×** slower than the CPU. To hide that latency, modern CPUs put SRAM on-die and serve hot data from there. The cache is not a single thing — it is a hierarchy in miniature.

| Level | Per-core / shared | Typical size | Latency | Built from |
|-------|-------------------|--------------|---------|------------|
| L1 (split into L1-I and L1-D) | per core | 32-64 KB each | ~1 ns / 4 cycles | SRAM |
| L2 | per core (or per pair) | 256 KB - 2 MB | ~4 ns / 12 cycles | SRAM |
| L3 (LLC) | shared across all cores | 8 MB - 96 MB | ~15 ns / 40 cycles | SRAM |
| DRAM | external | 8-128 GB | ~100 ns | DRAM |

The hit rates compound. With 95% L1 hit, 85% L2 hit on misses, 70% L3 hit on remaining misses:

```
average latency
  = 0.95   * 1 ns
  + 0.05  * 0.85 * 4 ns
  + 0.05  * 0.15 * 0.70 * 15 ns
  + 0.05  * 0.15 * 0.30 * 100 ns
  ~ 1.4 ns
```

That is **~70× faster** than going to DRAM every time. The whole point of the hierarchy is to make 99% of accesses look like L1 access while still pretending you have 32 GB of memory.

**Why three levels and not two or four?** Each extra level adds a tag-check stage in the pipeline. Two levels leave too big a gap (L2 to DRAM = 25× jump). Four levels add complexity without much win. Three is the empirically tuned answer for current technology — though server chips with **96 MB of L3 via 3D V-Cache** are pushing on this boundary.

# 4. Virtual Memory: Every Process Lives in Its Own Address Space

So far we have talked about physical addresses — actual coordinates in DRAM. But **no modern process ever sees a physical address**. Programs run in a virtual address space, and the hardware translates virtual to physical on every single memory access.

![Virtual to physical address translation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig3_virtual_memory.png)

The translation works in fixed-size chunks called **pages** (typically 4 KB). A 64-bit virtual address is split into:

- the **virtual page number (VPN)** — which page of the program?
- the **offset** — which byte inside that page?

The OS maintains a **page table** for each process, mapping VPN → PFN (physical frame number). On every load and store, the hardware:

1. Splits the address.
2. Walks the page table to find the matching PFN.
3. Concatenates PFN + offset to get the physical address.
4. Issues that physical address to the memory controller.

This buys us four enormous things at once:

- **Isolation**: process A cannot accidentally (or maliciously) read process B's memory because their page tables map to different physical frames.
- **Illusion of large memory**: the page table can mark a page as "not present", forcing a page fault that the OS handles by loading the page from disk — this is called swapping.
- **Sharing**: two processes can share a library by mapping the same physical frames into both page tables.
- **Permissions**: each page table entry has read/write/execute bits enforced by hardware, which is the foundation of OS security.

The cost: every memory access now requires walking a page table, which itself lives in memory. On x86-64 the page table is 4 levels deep, so a TLB miss can cost up to 4 extra DRAM reads — hundreds of nanoseconds. Which is exactly why the TLB exists.

# 5. The TLB: The Cache You Have Never Heard Of

The Translation Lookaside Buffer is a tiny, fully-associative SRAM cache of recent VPN → PFN translations. Every CPU has one (often two: L1 TLB and L2 TLB), with somewhere between 64 and 1024 entries.

![TLB hit and miss paths](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig4_tlb.png)

When the CPU issues a virtual address:

- **TLB hit (~99% of the time)**: translation comes back in 1 cycle. The CPU goes straight to DRAM/cache. The cost is essentially zero.
- **TLB miss (~1%)**: hardware (or, on some architectures, software) walks the page table. On x86-64 that is up to four memory reads — 100-400 ns. The result is then inserted into the TLB so the next access is fast.

The TLB is why virtual memory is practical at all. At 99% hit-rate, the average translation cost is ~1 ns; without the TLB it would be ~100 ns on every single load. Most workloads never even notice virtual memory exists. But TLB misses are also why workloads with **large random working sets** (some databases, graph algorithms, certain ML inference patterns) can be much slower than their cache-miss numbers suggest. **Huge pages** (2 MB or 1 GB instead of 4 KB) exist mainly to relieve TLB pressure: one TLB entry now covers 512× more memory.

# 6. Memory Channels: Why Two Sticks Beat One Big Stick

DRAM is connected to the CPU through **memory channels**. A channel is an independent 64-bit data path with its own command/address bus. A modern desktop CPU has 2 channels; a server CPU has 4, 8 or 12.

![Memory channels: parallel data lanes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig7_memory_channels.png)

Each channel can be doing a transfer at the same time as every other channel. So peak bandwidth scales nearly linearly with channel count:

| Configuration | Channels | Per-channel rate | Aggregate bandwidth |
|---------------|----------|------------------|---------------------|
| Single-channel DDR4-3200 | 1 | 25.6 GB/s | 25.6 GB/s |
| Dual-channel DDR4-3200 | 2 | 25.6 GB/s | **51.2 GB/s** |
| Quad-channel DDR5-4800 | 4 | 38.4 GB/s | **153 GB/s** |
| 12-channel DDR5-4800 (Sapphire Rapids) | 12 | 38.4 GB/s | **460 GB/s** |

Latency does not improve — a single load is still ~100 ns. What improves is **how many independent loads can be in flight at once**. That is precisely what matters for:

- Multi-core workloads (each core wants its own memory traffic),
- GPU-like memory-bound code (rendering, video, scientific compute),
- Anything that streams large arrays.

This is also the technical reason "**2 × 8 GB beats 1 × 16 GB**": a single stick can only fill one channel. The total capacity is the same, the bandwidth is half. On modern Intel/AMD desktops you typically populate slots **A2 + B2** to get dual channel.

# 7. ECC Memory: When Cosmic Rays Become a Bug Report

DRAM is reliable but not perfect. Cells get hit by alpha particles from package materials, by neutrons from cosmic rays, by electromagnetic noise. Google's large-scale field study (2009) found that DRAM error rates are far higher than vendors had publicly stated — on the order of **one correctable error per gigabyte per year**, with some modules orders of magnitude worse.

For a gaming PC, an undetected bit flip means a rare crash. For a database with billions of rows, it means **silent data corruption that propagates into backups**.

![ECC memory: detect and correct bit flips](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig5_ecc_correction.png)

ECC memory adds extra check bits — typically 8 ECC bits per 64 data bits — and a Hamming SEC-DED code:

- **SEC** (Single Error Correction): any single-bit flip is detected and silently corrected. The CPU sees clean data.
- **DED** (Double Error Detection): any double-bit flip is detected (but not corrected) and reported, so the OS can panic instead of returning bad data.

The trade-offs are real:

- **Cost**: ~12% more silicon (72 bits to store 64), and ECC DIMMs are 30-50% more expensive at retail.
- **Performance**: a small latency hit for the encode/decode (negligible on modern controllers).
- **Compatibility**: most consumer CPUs disable ECC (Intel Core, most Ryzen). Xeon, EPYC, Threadripper Pro, and Apple Silicon support it.

**Rule of thumb**: if losing a bit would be a bug report rather than an inconvenience, you want ECC. Servers, NAS, workstations doing finance/CAD/scientific work — yes. A pure gaming desktop — usually no.

DDR5 introduces **on-die ECC** which protects against errors *inside the chip*, but that is not the same as full system-level ECC: errors on the bus between chip and CPU still need traditional ECC DIMMs.

# 8. NUMA: When Memory Has a Postcode

Once you put two CPU sockets on a board (or two chiplets in a package), memory becomes **non-uniform**. Each socket has its own memory controller and its own local bank of DRAM. Accessing your local memory is fast; accessing the other socket's memory has to traverse the inter-socket interconnect (Intel UPI, AMD Infinity Fabric).

![NUMA topology and access asymmetry](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/02-memory/fig6_numa.png)

Local access on a typical 2-socket server is ~80 ns. Remote access through UPI is ~140 ns — **about 1.7× slower**, and bandwidth across the link is limited. On 4-socket and 8-socket systems the worst-case ratio can climb above 3×.

What this means in practice:

- **The OS is NUMA-aware.** Linux's scheduler tries to keep a thread on the same socket as the memory it allocates, and tries to allocate new memory on the local node (`numactl`, `mbind`, `set_mempolicy`).
- **Allocators are NUMA-aware.** glibc/jemalloc/tcmalloc do per-thread arenas to keep allocations local.
- **Databases are NUMA-aware.** Postgres, MySQL, ClickHouse, Spark all expose pinning/binding options.
- **Ignore NUMA and you can lose 30-60% throughput** on a workload that should otherwise scale linearly with cores.

Even on a single-socket modern CPU you can hit "NUMA-like" effects: AMD's chiplet (CCD) design means cores in one CCD see slower L3 access to data cached in another CCD. The architectural lesson generalises — **memory has a topology**, and good systems software respects it.

# 9. DDR Generations: What Actually Changes Each Step

Each DDR generation roughly doubles peak bandwidth and trims voltage. The interesting story is *how*.

| Gen   | Year | Per-pin rate | Peak per channel | Voltage | Key architectural change |
|-------|------|--------------|------------------|---------|---------------------------|
| DDR   | 2000 | 200-400 MT/s | 1.6-3.2 GB/s     | 2.5 V   | Transfer on both clock edges |
| DDR2  | 2003 | 400-800 MT/s | 3.2-6.4 GB/s     | 1.8 V   | 4n prefetch (internal bus 2× I/O) |
| DDR3  | 2007 | 0.8-2.1 GT/s | 6.4-17 GB/s      | 1.5 V   | 8n prefetch, fly-by topology |
| DDR4  | 2014 | 1.6-3.2 GT/s | 12.8-25.6 GB/s   | 1.2 V   | Bank groups for parallelism |
| DDR5  | 2020 | 4.8-8.4 GT/s | 38.4-67.2 GB/s   | 1.1 V   | **Two 32-bit subchannels per DIMM**, on-die ECC, on-DIMM PMIC |

The single most important change in DDR5 is splitting each 64-bit DIMM into **two independent 32-bit subchannels**. Effectively, what looks like dual-channel from the outside is now quad-channel internally. That is why DDR5 helps multi-core workloads disproportionately: the controller can issue twice as many independent requests per DIMM.

DDR6 (in standardisation) is targeting 8.8-17.6 GT/s and four subchannels per DIMM.

# 10. Q & A

### Q1. If caches make memory access "average ~1 ns", why does memory speed still matter?

Because the average hides the tail. A cache miss to DRAM still costs ~100 ns, and any workload that doesn't fit in L3 — large databases, big arrays, ML models — generates these misses constantly. Faster RAM (higher MT/s, lower CL) reduces the cost of those misses, and dual-channel doubles the rate at which misses can be serviced. The 5-15% real-world FPS improvement from DDR4-3200 → DDR4-3600 is almost entirely about reducing miss penalty.

### Q2. Should I prioritise frequency or CL timing?

Look at the actual nanosecond latency: `ns = CL × 2000 / MT/s`. DDR4-3200 CL16 and DDR4-3600 CL18 are both exactly 10 ns first-word latency, so the 3600 kit wins on bandwidth at no latency cost. But DDR4-3600 CL16 is 8.9 ns and clearly superior to both. As a working rule, **frequency improvements compound with bandwidth, latency improvements compound with cache-miss-heavy workloads**. For most users, DDR4-3600 CL16 or DDR5-6000 CL30 is the sweet spot.

### Q3. Why doesn't DDR5 feel "twice as fast" even though bandwidth doubled?

Because most consumer workloads are **not bandwidth-bound** — they are latency-bound or compute-bound. Games hit a CPU bottleneck or a GPU bottleneck long before they saturate a single-channel DDR4-3200 link. DDR5 shines on (1) multi-core productivity (compilation, encoding, simulation), (2) integrated GPUs that share system memory, and (3) future workloads as core counts keep growing. If you are bandwidth-bound today, you already know it from profiling.

### Q4. What actually happens when I "run out of RAM"?

The kernel does three things, in increasing severity:

1. **Reclaim page cache** — the OS uses any free RAM as a disk cache. When pressure rises, it drops the oldest cached file pages first. Cost: future disk reads will miss the cache.
2. **Swap out anonymous pages** — pages that aren't backed by a file (program heap, anonymous mmap) get written to a swap file or partition. Future access triggers a major page fault that has to read the page back from disk. Cost: ~10 ms each on an NVMe SSD, and your latency tail explodes.
3. **OOM killer** — if even swap can't keep up, the kernel picks a process and kills it. On Linux, the choice is roughly proportional to memory used.

This is why "8 GB is fine, I have a fast SSD" is a half-truth: SSDs help once you are already swapping, but they cannot hide the fact that you fell out of the DRAM tier.

### Q5. Why is cache coherence such a big deal in multi-core systems?

Each core has its own L1/L2 cache, but they share the same logical view of memory. If core 0 writes `x = 1` to its L1 and core 1 has its own cached copy of `x = 0`, the system has to detect this and invalidate or update other copies. The standard protocol is **MESI** (Modified, Exclusive, Shared, Invalid), implemented in hardware and tracked by the L3/directory.

The performance cliff is **false sharing**: two threads write to two different variables that happen to live in the same 64-byte cache line. Every write on one core invalidates the other core's copy, even though logically there is no shared data. The fix is to pad hot per-thread state out to a full cache line. This single change can give 3-10× speedup on contended counters and lock-free queues.

### Q6. What is "memory ordering" and why should I care?

Modern CPUs reorder loads and stores aggressively to keep pipelines busy. As long as a single thread's view stays consistent, the hardware is free to do almost anything. But **across threads**, what one core sees is not necessarily the order another core wrote things. x86 has a relatively strong model (TSO — total store order, with a single store-buffer-induced exception). ARM and PowerPC are much weaker. This is why concurrent code uses **memory barriers**, **atomics**, and **acquire/release semantics** — to force the hardware to publish writes in the order the algorithm requires. Almost every "works on my x86 laptop, breaks on ARM server" bug is really a memory-ordering bug.

### Q7. Why do servers have so many memory channels?

Server CPUs are usually bandwidth-bound. A 96-core EPYC running an in-memory database is constantly pulling fresh data from DRAM; a single channel would starve it. Twelve channels of DDR5-4800 give ~460 GB/s — and even that is often the bottleneck. The same logic applies to GPUs, which is why HBM (high-bandwidth memory) stacked next to the GPU die exists: it provides ~3 TB/s by stacking 8-12 DRAM dies on a wide silicon interposer.

### Q8. How do I diagnose whether a slow program is memory-bound?

Three quick signals on Linux:

1. `perf stat -e cache-misses,cache-references,LLC-loads,LLC-load-misses ./prog` — high LLC miss rate (> 10% of LLC loads) usually means DRAM-bound.
2. `perf stat -e dTLB-load-misses,iTLB-load-misses` — high TLB misses suggest huge pages will help.
3. Top-down profiling with `toplev` (`pmu-tools`) — directly attributes pipeline stalls to "Backend / Memory Bound", "Frontend Bound", "Bad Speculation", or "Retiring", which tells you whether the bottleneck is even memory at all.

If the bottleneck is genuinely memory bandwidth, the fixes are: blocking/tiling the algorithm to fit in cache, NUMA pinning, huge pages, prefetch hints, or in extreme cases switching to a more memory-friendly data layout (struct-of-arrays vs array-of-structs).

# 11. Summary

- The memory hierarchy exists because **no single technology is both fast and dense**. SRAM is fast and expensive (caches). DRAM is dense and slower (main memory). Flash and disk are denser still and persistent.
- A single load instruction quietly traverses **registers → L1 → L2 → L3 → memory controller → channel → DRAM rank/bank/row/column**, with translation through the **TLB and page table** layered on top.
- **Caches** turn average DRAM latency from 100 ns into ~1 ns by exploiting temporal and spatial locality. Hit rates compound across L1/L2/L3.
- **Virtual memory** gives every process its own address space, with hardware-enforced isolation, sharing, and on-demand paging. The **TLB** is what makes that affordable.
- **DDR generations** keep doubling bandwidth and trimming voltage. **DDR5's** real innovation is two subchannels per DIMM, on-die ECC, and on-DIMM power management.
- **Channels** scale bandwidth almost linearly. Always populate dual-channel; for serious work, populate every channel your board offers.
- **ECC** turns silent corruption into a logged event. Essential for servers, optional for desktops.
- **NUMA** means memory has a topology. Local access is fast; remote access is slower. Modern OSes and runtimes take this seriously, and so should you on multi-socket or chiplet systems.

# What's Next?

In **Computer Fundamentals (3): Storage Systems**, we will follow the data one more level out: the controller and FTL inside an SSD, the SLC/MLC/TLC/QLC trade-off, NVMe queues and PCIe lanes, RAID and erasure coding, and why the storage stack is currently the most rapidly evolving layer of the system. Stay tuned.
