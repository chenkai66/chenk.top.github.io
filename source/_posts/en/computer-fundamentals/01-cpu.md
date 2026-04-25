---
title: "Computer Fundamentals: CPU and the Computing Core"
date: 2022-10-01 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: "Computer Fundamentals"
series_order: 1
series_total: 6
lang: en
mathjax: false
description: "Understand data units (bits and bytes), CPU architecture, the 5-stage pipeline, cache hierarchy, branch prediction, out-of-order execution, multi-core and SMT, Intel vs AMD differences, and how to choose the right processor."
disableNunjucks: true
---

Why does your 100 Mbps internet only download at about 12 MB/s? Why does a "1 TB" hard drive show only 931 GB in Windows? Why does a 32-bit system top out around 3.2 GB of usable RAM? And what *actually* happens, cycle by cycle, when the CPU runs your code?

This is part 1 of the **Computer Fundamentals** series. We start from bits and bytes, then go down into the CPU itself: pipelines, caches, branch prediction, out-of-order execution, multiple cores, and SMT. By the end you should be able to read a CPU spec sheet — or a perf profile — and know what each number is paying for.

# Series Navigation

**Computer Fundamentals** (6 parts):

1. **CPU and the Computing Core** — *you are here*
2. Memory and the Cache Hierarchy
3. Storage Systems (HDD, SSD, NVMe, RAID)
4. Motherboard, GPU, and Expansion Buses
5. Networking, Power, and Cooling
6. Deep Dive: Putting it All Together

# Three Real-World Puzzles

**Puzzle 1 — the bandwidth mystery.** Your ISP sells you "100 Mbps" service but Chrome's download manager peaks at about 12 MB/s. Are you being shortchanged?

**Puzzle 2 — the missing 69 GB.** You buy a 1 TB drive. Windows says 931 GB. Defective?

**Puzzle 3 — the FPS upset.** Your friend's 14-core i5-13600K runs CS2 at ~450 FPS. Your shiny 16-core R9 7950X gets ~420. More cores, fewer frames — why?

All three answers live in this article.

# Part 1 — Data Units: The Metric System of Computing

## Bit and Byte

A **bit** is a single binary digit: 0 or 1. Think of one switch: off or on. Eight switches together can encode 2^8 = 256 different patterns — exactly enough for the original ASCII alphabet of letters, digits and punctuation. That is why a **byte** is 8 bits.

The "Mbps vs MB/s" trap: ISPs sell bandwidth in **bits per second** (Mbps). Browsers display **bytes per second** (MB/s). To convert, divide by 8.

| You see                     | Meaning                       |
|-----------------------------|-------------------------------|
| 100 Mbps                    | 100 million **bits** / second |
| 12.5 MB/s (theoretical max) | 12.5 million **bytes** / second |
| 10–11 MB/s (typical actual) | After TCP overhead and congestion |

So a "100M" line really does deliver about 12 MB/s. No one is cheating you — the units are just different.

## The 1024 vs 1000 Mismatch

Computers count in powers of two; the closest power of 2 to 1000 is 2^10 = 1024. So internally, 1 KB = 1024 B, 1 MB = 1024 KB, and so on. Disk manufacturers, however, use the SI prefixes where 1 KB = 1000 B. That single mismatch is the entire reason a "1 TB" disk shows up as 931 GB.

| Unit (binary, computer) | Bytes               | Unit (decimal, vendor) | Bytes               |
|-------------------------|---------------------|------------------------|---------------------|
| 1 KiB                   | 1,024               | 1 KB                   | 1,000               |
| 1 MiB                   | 1,048,576           | 1 MB                   | 1,000,000           |
| 1 GiB                   | 1,073,741,824       | 1 GB                   | 1,000,000,000       |
| 1 TiB                   | 1,099,511,627,776   | 1 TB                   | 1,000,000,000,000   |

Quick estimate: **actual capacity ≈ advertised capacity × 0.931**.

| Advertised | Actual usable | Apparent loss |
|------------|---------------|---------------|
| 256 GB     | 238 GiB       | 7.0%          |
| 1 TB       | 931 GiB       | 6.9%          |
| 2 TB       | 1,863 GiB     | 6.9%          |
| 4 TB       | 3,726 GiB     | 6.9%          |

RAM does not "shrink" because RAM modules are sized in binary units to begin with: an 8 GB stick is exactly 8 × 2^30 bytes.

# Part 2 — Inside a CPU Core

## What a CPU Actually Is

A CPU's job is brutally simple to state and absurdly complicated to implement: **fetch the next instruction, decode it, execute it, and write back the result, billions of times per second**. Everything else — pipelines, caches, branch predictors — exists to make that loop run faster without changing what the program means.

A modern core has four kinds of hardware inside:

- **Front-end (control)**: fetch, decode, branch predictor.
- **Compute**: integer ALUs, floating-point and SIMD units, load/store unit.
- **Register file**: a tiny, fast scratchpad (16 general-purpose registers in x86-64, plus AVX-512 vectors).
- **Caches**: L1 instruction + L1 data (per core), L2 (per core), L3 (shared).

![Figure 1. CPU core block diagram: front-end, compute, registers, and the cache hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig1_cpu_architecture.png)

The front-end *prepares* work; the compute units *do* the work; the register file holds the operands the compute units will read; the caches keep the working set close enough to feed the whole assembly line at full speed.

## The 5-Stage Pipeline

The classic teaching pipeline has five stages:

1. **IF** — Instruction Fetch (read the next instruction from L1I)
2. **ID** — Instruction Decode (figure out what it is, read register operands)
3. **EX** — Execute (run the ALU)
4. **MEM** — Memory access (loads / stores)
5. **WB** — Write Back (commit the result to a register)

Without overlap, each instruction would take 5 cycles, and 5 instructions would take 25 cycles. With pipelining, every cycle a *new* instruction enters IF while the previous one moves to ID, etc. After a 4-cycle warm-up the pipeline is full and one instruction completes every cycle — the **ideal IPC (instructions per cycle) is 1.0**.

![Figure 2. Five instructions overlap across nine cycles instead of running serially in twenty-five](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig2_pipeline_stages.png)

Real x86 cores like Intel Golden Cove or AMD Zen 4 are far wider — 6 to 8 instructions decoded per cycle, with 10+ pipeline stages — but the idea is unchanged: **keep every stage busy every cycle**. Anything that creates a *bubble* (an empty stage) costs throughput. The next four sections are all about preventing or hiding those bubbles.

## The Memory Hierarchy

If main memory were as fast as the registers, none of the caching machinery would exist. It is not. There are roughly **eight orders of magnitude** between a register access and a hard-disk seek.

![Figure 3. Memory hierarchy: latency in nanoseconds, log scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig3_cache_hierarchy.png)

| Level         | Latency  | Capacity     | Built from          |
|---------------|----------|--------------|---------------------|
| Register      | ~0.3 ns  | bytes        | flip-flops in the core |
| L1 cache      | ~1 ns    | 32–64 KB     | SRAM, 6 transistors/bit |
| L2 cache      | ~4 ns    | 256 KB–1 MB  | SRAM (denser, slower) |
| L3 cache      | ~15 ns   | 8–64 MB      | SRAM (shared across cores) |
| DRAM (DDR4/5) | ~90 ns   | 8–128 GB     | 1 transistor + 1 capacitor / bit |
| SSD (NVMe)    | ~100 µs  | 0.5–8 TB     | NAND flash         |
| HDD           | ~10 ms   | 1–20 TB      | spinning rust       |

**A useful mental model.** Scale CPU clock cycles up to "1 second of human thinking" (roughly 1 ns → 1 s):

- L1 hit: instant, you remember it.
- L3 hit: 15 seconds, walk to the next desk.
- DRAM: 90 seconds, walk to the next room.
- SSD: a day and a half.
- HDD: 4 months.

This is why a single L3 miss costs a CPU dozens of *wasted* cycles, and why a cache-friendly inner loop can be 10× faster than a cache-hostile one even when both contain the same arithmetic.

## The Three Kinds of Cache Miss

Caches are not magic; they fail in three distinct ways. Knowing which kind of miss you have tells you how to fix it.

![Figure 4. The 3C taxonomy: compulsory, capacity, conflict](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig4_cache_misses.png)

- **Compulsory (cold) miss**: the first time a block is touched, the cache has never seen it. Unavoidable on the first pass; *prefetching* can hide the latency.
- **Capacity miss**: the working set is larger than the cache. A fully-associative, perfectly-managed cache would *still* miss. Fixes: shrink the working set (blocking / tiling), use a bigger cache level.
- **Conflict miss**: a low-associativity cache has multiple hot blocks fighting for the same set. Two arrays whose addresses differ by exactly one cache-set's worth of bytes will repeatedly evict each other. Fixes: change the data layout, pad arrays, or rely on higher-associativity caches.

Modern L1 caches are typically 8-way associative and L2/L3 even more, so pure conflict misses are less common than they were on early processors — but they still appear in tight loops over power-of-two-strided arrays.

## Branch Prediction

Conditional branches are the pipeline's nightmare: by the time `if (x > 0)` *resolves* in EX, three or four later instructions have already been fetched down one path. If the guess was wrong, all that in-flight work is squashed and refetched — a misprediction in a modern core costs **15–20 cycles**.

The classic trick is the **2-bit saturating counter**, one per branch in the predictor table:

![Figure 5. 2-bit predictor state machine, and prediction accuracy across predictor generations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig5_branch_prediction.png)

The state machine has the nice property that **one anomalous outcome doesn't immediately flip the prediction** — only two consecutive wrong outcomes do. That alone gets you to ~93% accuracy on typical workloads. Modern predictors (TAGE, perceptron-based) push this above 98% by also looking at the *history* of recent branches, which captures correlations like "if the outer loop took its branch, the inner loop usually does too."

If you've ever seen `[[likely]]` / `[[unlikely]]` annotations in C++20 or `__builtin_expect` in GCC, that's you helping the compiler help the predictor.

## Out-of-Order Execution

Even with great branch prediction, the pipeline still stalls on data: a `load` from DRAM can't deliver its value for ~90 ns (~300 cycles at 3 GHz), and any instruction that *uses* that value must wait. An in-order machine waits with it; everything behind also waits.

An **out-of-order (OoO)** core does something cleverer: it buffers dozens or hundreds of decoded instructions in a *reorder buffer*, and dispatches any instruction whose operands are ready, regardless of program order. As long as the *visible result* (architectural register state, memory) ends up the same, the hardware can shuffle execution however it wants.

![Figure 6. The same five-instruction stream, in-order vs out-of-order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig6_ooo_execution.png)

In the example, instructions I3 and I4 are independent of the load. An in-order core stalls them behind I2; an OoO core slips them into the cycles where the load is in flight. The result: 6 cycles instead of 8, on a tiny example. On real workloads with large reorder windows (the Apple M3's window is reportedly ~600 micro-ops), the savings are dramatic and *automatic* — no programmer effort.

OoO is also why **micro-benchmarks lie**: if your loop body has independent instructions, the CPU will execute them in parallel and your measurement is the *throughput* of the bottleneck unit, not the latency of any one operation.

## Multi-Core and SMT (Hyperthreading)

There are two ways to add more parallelism without making each core faster:

1. **More cores.** Replicate everything: front-end, ALUs, registers, L1, L2. Share only L3 and the memory controller.
2. **SMT (Intel calls it Hyperthreading).** Within one physical core, keep two architectural register sets and let two software threads share the *same* execution units.

![Figure 7. Multi-core layout (left) and how SMT fills idle execution slots (right)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/01-cpu/fig7_multicore_smt.png)

**Key intuition for SMT:** when one thread stalls (cache miss, branch mispredict), its execution units sit idle. SMT lets a second thread immediately use those slots. The typical real-world gain is **+20–30% throughput** on a mixed workload — *not* 2×, because the two threads share one set of ALUs, one L1 cache, and one set of execution ports. On purely compute-bound code that already saturates the units (a tight matrix multiply, say), SMT can even *slow you down* by adding cache pressure, which is why HPC workloads often disable it.

This finally answers Puzzle 3 from the opening. CS2 at 450 FPS is not bottlenecked by core count — at that frame rate, the engine's main thread is the long pole. The i5-13600K runs that thread at higher per-cycle performance than the R9 7950X (newer P-core, higher boost clock, larger L2 per core). The 7950X's extra cores sit idle. **More cores only helps if your work decomposes into more parallel threads.**

# Part 3 — Choosing a CPU

## Intel vs AMD, Today

The headline trade-offs in the current generation:

| Aspect                | Intel (Core 13/14th gen) | AMD (Ryzen 7000 / 9000)        |
|-----------------------|--------------------------|--------------------------------|
| Single-thread peak    | very high (P-cores boost to ~6 GHz) | very high (Zen 4/5 IPC) |
| Many-thread throughput | E-cores add wide MT     | high core count, all symmetric |
| Process node          | Intel 7 / Intel 4        | TSMC 5 nm / 4 nm               |
| Platform longevity    | LGA 1700 (12/13/14)      | AM5 (promised through 2027+)   |
| Power efficiency      | weaker at top of stack   | generally better                |
| Best at               | gaming, mixed workloads  | content creation, compilation   |

Picking by workload:

- **Gaming**: single-thread matters most. A mid-tier Intel i5/i7 or a Ryzen 7 with high boost is the sweet spot.
- **Video editing, 3D rendering, large compiles**: more cores win. Ryzen 9 or Threadripper.
- **Office, web, light dev**: any modern 4–6 core CPU is overkill in the good way.
- **Servers**: AMD EPYC (up to 96+ cores, 12-channel DDR5, more PCIe lanes) currently leads on density and price/perf; Intel Xeon still wins on certain accelerator-heavy workloads (AMX, QAT).

## 32-bit vs 64-bit and the 4 GB Wall

A 32-bit address bus can name 2^32 = 4 GiB of memory locations. A 32-bit OS, however, has to fit *both* RAM *and* memory-mapped I/O (graphics aperture, BIOS, PCIe device registers) into that same 4 GiB. The top ~0.5–1 GiB is reserved for MMIO, leaving ~3.0–3.5 GiB for actual RAM. This is the "I installed 4 GB and Windows shows 3.25 GB" phenomenon — and it is also why the only real fix is to move to a 64-bit OS, where the 2^64 ≈ 16 EiB address space ends the conversation forever.

## ECC, Multi-Socket, and Why Server Clocks Are Lower

Server CPUs make different trade-offs because their workload is different:

- **ECC memory** detects and corrects single-bit errors. At server scale (thousands of DIMMs running 24/7) bit flips happen often enough to matter; ECC is non-negotiable in finance, scientific computing, and anything safety-critical.
- **Multi-socket** lets two or four CPUs share one coherent memory image — useful for very large databases and virtualization hosts.
- **More PCIe lanes** (64–128 vs ~20 on desktop) feed many GPUs, NVMe drives, and 100 GbE NICs.
- **Lower clock speeds** (all-core ~3 GHz instead of single-core 5.8 GHz) trade peak frequency for dramatically better power efficiency and reliability under sustained load. A desktop chip is a sprinter; a server chip is a marathon runner.

# Quick-Reference Cheat Sheet

- Bandwidth in **bits**, file size in **bytes** — divide by 8.
- Disks use base 1000, OSes use base 1024 — multiply advertised TB by 0.931.
- The CPU pipeline ideal is **1 instruction per cycle**; everything fancy (caches, branch prediction, OoO) exists to defend that ideal.
- Memory latency spans **eight orders of magnitude**. Cache locality is usually the single biggest performance lever.
- A misprediction costs ~15–20 cycles. A DRAM miss costs ~300. Predictor and cache work hide them.
- SMT typically buys +20–30%, not 2×.
- More cores only helps work that parallelizes; gaming usually doesn't.

# What's Next

Part 2 — **Memory and the Cache Hierarchy** — goes deeper:

- DDR generations from DDR2 to DDR5: what actually got faster?
- Dual-channel and quad-channel: real measured benchmarks.
- Cache associativity, replacement policies, and how to measure cache misses with `perf`.
- Memory troubleshooting: black screens, blue screens, MemTest86.

**Thinking question for the gap:** if the CPU already has L1/L2/L3 cache, why do we need DRAM at all? (Hint: capacity per dollar, and what SRAM costs to build.)

# Further Reading

- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* — the canonical text.
- Bryant & O'Hallaron, *Computer Systems: A Programmer's Perspective* (CSAPP) — the canonical undergrad text.
- Agner Fog's optimization manuals — practical x86 microarchitecture details.
- Intel ® 64 and IA-32 Architectures Software Developer's Manual.
- AMD64 Architecture Programmer's Manual.
- WikiChip — concise, accurate die diagrams of every modern microarchitecture.
