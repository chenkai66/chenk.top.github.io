---
title: "Computer Fundamentals (3): Storage Systems (HDD vs SSD)"
date: 2022-11-12 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: computer-fundamentals
series_order: 3
series_total: 6
lang: en
mathjax: false
description: "Compare HDD and SSD working principles, SATA vs NVMe interface speeds, SLC/MLC/TLC/QLC lifespan, SSD optimization (4K alignment, TRIM), and RAID array configurations."
disableNunjucks: true
translationKey: "computer-fundamentals-3"
---
![Chapter concept illustration](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/illustration_1.jpg)

Why can a single SSD swap "resurrect" a five-year-old laptop? Why does a TLC drive rated for only 1 000 P/E cycles still last more than a decade for normal users? Why does a brand-new SSD that benchmarks at 3 500 MB/s sometimes collapse to 50 MB/s after a few weeks? This third instalment of the **Computer Fundamentals Deep Dive Series** answers those questions from first principles. We will look at how rotating magnetic platters compare with charge-trap NAND cells, how the bandwidth of an interface (SATA, PCIe Gen 3/4/5) interacts with the parallelism of a protocol (AHCI vs NVMe), how RAID levels trade capacity for fault tolerance, how a file system organises bytes on a raw block device, and how to keep all of this fast and safe in production.

## Series Navigation

**Computer Fundamentals Deep Dive Series** (6 parts)

1. CPU & Computing Core
2. Memory & High-Speed Cache
3. **→ Storage Systems** (HDD vs SSD, interfaces, RAID, file systems, tiering) ← *You are here*
4. Motherboard, Graphics & Expansion
5. Network, Power & Practical Troubleshooting
6. Deep-Dive Appendix

## Part 1 — HDD vs SSD: Two Very Different Physics

The defining difference between a hard-disk drive and a solid-state drive is not "spinning vs not spinning" — it is **how a single bit is stored and how long it takes to reach it**. An HDD stores each bit as the magnetisation direction of a tiny grain on a spinning platter; reading it requires a mechanical seek, then waiting for the platter to rotate the right sector under the head. An SSD stores each bit as the threshold voltage of a floating-gate or charge-trap transistor inside a NAND cell; reading it is a purely electrical operation that completes in microseconds.

![HDD vs SSD physical anatomy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig1_hdd_vs_ssd.png)

**Why the structural difference matters.** Inside the HDD on the left of the figure, every random read costs at least one *seek* (typically 4–10 ms) plus *rotational latency* (4 ms average at 7 200 RPM). Sequential reads are limited by how much data passes under the head per revolution — about 200 MB/s on the outer tracks of a modern 24 TB drive, less on the inner tracks. Inside the SSD on the right, the same read is fanned out to many NAND dies in parallel by the controller. The drive's *sustained* bandwidth is therefore set by how many dies the controller can talk to simultaneously, not by any moving part.

| Property | HDD (7 200 RPM) | SATA SSD | NVMe Gen 4 SSD |
| --- | --- | --- | --- |
| Bit storage medium | Magnetic grains on a platter | NAND cell (charge trap) | NAND cell (charge trap) |
| Random read latency | ~10 ms | ~100 µs | ~30 µs |
| Sequential read | 100–200 MB/s | 550 MB/s | 7 000 MB/s |
| 4 KiB random IOPS | ~120 | ~95 000 | ~1 000 000 |
| Idle / active power | 5–10 W | 2–4 W | 4–7 W |
| Shock tolerance (operating) | ~30 G | ~1 500 G | ~1 500 G |
| $ per TB (2026) | ~15 | ~60 | ~110 |
| Typical capacity ceiling | 24 TB | 8 TB | 8 TB |

The HDD still wins on two dimensions: **raw capacity per dollar** and **the ability to recover data forensically** after a failure (because the magnetic encoding survives a controller death). The SSD wins on every other dimension that humans actually feel — boot time, app launch, level loading, large-file copy, and battery life.

## Why an SSD upgrade is the single biggest perceptual win

A user does not "feel" sequential bandwidth; they feel **random read latency** and **queue depth**. Booting Windows touches roughly 30 000 small files; launching Chrome touches another 4 000. On an HDD each of those touches pays a full seek; on an NVMe SSD they overlap inside thousands of in-flight commands. That is why moving from an HDD to *any* SSD typically cuts boot time by 4–7× — a far larger user-visible jump than doubling RAM or upgrading a CPU generation.

## Part 2 — Storage Performance: Three Numbers That Matter

Almost every storage decision can be made by looking at three numbers — **throughput**, **IOPS**, and **latency** — at the right percentile. Average values lie; the long tail is what users notice.

![Throughput, IOPS, and latency on a log scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig2_performance.png)

- **Throughput (MB/s)** matters when you are moving large contiguous files: video edits, backups, game-asset patches. Notice how each interface generation (SATA → PCIe 3 → 4 → 5) roughly doubles the previous ceiling.
- **IOPS** matters for everything else: launching apps, compiling code, running a database, booting an OS. The HDD-to-SSD jump here is **three orders of magnitude** — 120 vs 95 000 — which is why even a budget SATA SSD feels transformative.
- **Latency** matters under load. A modern NVMe Gen 4 drive serves a 4 KiB random read in ~30 µs; the same operation on a 7 200 RPM HDD costs ~10 ms — a 300× gap, and one that gets *worse* on the HDD when many requests queue up.

A useful rule of thumb: *if your workload's queue depth is 1, latency dominates; if the queue is deep, IOPS and throughput dominate.* Consumer workloads are almost always queue-depth-1, which is why NVMe Gen 5 — whose advantage is mostly at QD32+ — is a marginal upgrade for a desktop user but a large one for a database server.

## Reading a vendor spec sheet honestly

Vendor numbers (e.g. "7 000 MB/s read, 1 000 000 IOPS") are measured with **128 KiB sequential** transfers and **QD256** workloads on a fresh drive with the SLC cache empty. Real desktop workloads see roughly:

- 50–70 % of the headline sequential number on sustained writes (after the SLC cache fills).
- 5–15 % of the headline IOPS at QD1.
- 2–4× the headline latency once the drive is more than 70 % full.

Always check the *sustained write speed past the SLC cache* and the *latency at QD1* — those are the numbers that determine how the drive feels.

## Part 3 — SSD Internals: NAND, FTL, and Why TLC "Only" 1 000 P/E Cycles Lasts Forever

A NAND cell is a transistor whose threshold voltage can be programmed. SLC stores one bit (two voltage levels), MLC two bits (four levels), TLC three (eight), QLC four (sixteen). Packing more levels into the same physical cell roughly halves cost per gigabyte but **quarters endurance and doubles read latency** at every step.

| NAND type | Bits / cell | Voltage levels | P/E cycles | Relative cost / GB | Where it ships today |
| --- | --- | --- | --- | --- | --- |
| **SLC** | 1 | 2 | ~100 000 | 6× | Industrial / boot SSD caches |
| **MLC (eMLC)** | 2 | 4 | 3 000–10 000 | 3× | Enterprise (rare in consumer) |
| **TLC** | 3 | 8 | 1 000–3 000 | 1.0× | Mainstream consumer & datacentre |
| **QLC** | 4 | 16 | 500–1 000 | 0.7× | Budget bulk storage |

## The lifespan calculation that always surprises people

The endurance figure looks alarming until you do the arithmetic for a real workload. A 1 TB TLC SSD with 1 000 P/E cycles can write 1 PB before wear-out. A heavy desktop user writes ~50 GB per day; a typical user writes 5–15 GB per day:

![RAID Level Decision Tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig_raid_decision_en.png)

**Picking the right interface.**

![Storage Media Selection Guide](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig_ssd_decision_en.png)

## Mnemonic

> **HDD spins, SSD switches** — physics decides latency before any benchmark runs.
> **Bus sets bandwidth, protocol sets parallelism** — SATA caps at 0.5 GB/s, NVMe scales to 12.
> **TLC is forever for humans** — 1 PB endurance dwarfs any consumer write rate.
> **Align, TRIM, leave room** — three free settings that double sustained write life.
> **RAID is uptime, backups are recoverability** — never confuse the two.
> **Hot bytes on fast media, cold bytes on cheap media** — tier or pay.

**Next in the series:** Motherboard, Graphics & Expansion — chipsets, VRMs, PCIe lane allocation, GPU architecture, and BIOS configuration.
