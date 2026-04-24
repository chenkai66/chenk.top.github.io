---
title: "Computer Fundamentals: Storage Systems (HDD vs SSD)"
date: 2024-04-29 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: "Computer Fundamentals"
series_order: 3
series_total: 6
lang: en
mathjax: false
description: "Compare HDD and SSD working principles, SATA vs NVMe interface speeds, SLC/MLC/TLC/QLC lifespan, SSD optimization (4K alignment, TRIM), and RAID array configurations."
disableNunjucks: true
---

Why can a single SSD swap "resurrect" a five-year-old laptop? Why does a TLC drive rated for only 1 000 P/E cycles still last more than a decade for normal users? Why does a brand-new SSD that benchmarks at 3 500 MB/s sometimes collapse to 50 MB/s after a few weeks? This third instalment of the **Computer Fundamentals Deep Dive Series** answers those questions from first principles. We will look at how rotating magnetic platters compare with charge-trap NAND cells, how the bandwidth of an interface (SATA, PCIe Gen 3/4/5) interacts with the parallelism of a protocol (AHCI vs NVMe), how RAID levels trade capacity for fault tolerance, how a file system organises bytes on a raw block device, and how to keep all of this fast and safe in production.

# Series Navigation

**Computer Fundamentals Deep Dive Series** (6 parts)

1. CPU & Computing Core
2. Memory & High-Speed Cache
3. **→ Storage Systems** (HDD vs SSD, interfaces, RAID, file systems, tiering) ← *You are here*
4. Motherboard, Graphics & Expansion
5. Network, Power & Practical Troubleshooting
6. Deep-Dive Appendix

# Part 1 — HDD vs SSD: Two Very Different Physics

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

# Part 2 — Storage Performance: Three Numbers That Matter

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

# Part 3 — SSD Internals: NAND, FTL, and Why TLC "Only" 1 000 P/E Cycles Lasts Forever

A NAND cell is a transistor whose threshold voltage can be programmed. SLC stores one bit (two voltage levels), MLC two bits (four levels), TLC three (eight), QLC four (sixteen). Packing more levels into the same physical cell roughly halves cost per gigabyte but **quarters endurance and doubles read latency** at every step.

| NAND type | Bits / cell | Voltage levels | P/E cycles | Relative cost / GB | Where it ships today |
| --- | --- | --- | --- | --- | --- |
| **SLC** | 1 | 2 | ~100 000 | 6× | Industrial / boot SSD caches |
| **MLC (eMLC)** | 2 | 4 | 3 000–10 000 | 3× | Enterprise (rare in consumer) |
| **TLC** | 3 | 8 | 1 000–3 000 | 1.0× | Mainstream consumer & datacentre |
| **QLC** | 4 | 16 | 500–1 000 | 0.7× | Budget bulk storage |

## The lifespan calculation that always surprises people

The endurance figure looks alarming until you do the arithmetic for a real workload. A 1 TB TLC SSD with 1 000 P/E cycles can write 1 PB before wear-out. A heavy desktop user writes ~50 GB per day; a typical user writes 5–15 GB per day:

```
Heavy user:   1 PB ÷ (50 GB/day × 365)  ≈ 55 years
Typical user: 1 PB ÷ (15 GB/day × 365)  ≈ 183 years
```

Even better, the SSD controller's **wear-levelling** spreads writes evenly across all NAND blocks, and **write amplification** (the ratio of NAND writes to host writes) on a healthy drive is 1.1–1.5×, not 10× as it was on early SSDs. The practical limit for almost everybody is therefore "the drive becomes obsolete before it wears out." The exception is sustained heavy writes — video capture, log servers, blockchain validators — where you should pick a TLC drive with a TBW rating > 600 and monitor SMART's `Wear Leveling Count` quarterly.

## Why a fresh SSD is fast and a full SSD is slow

The controller writes to the drive using a small **SLC cache** (a region of TLC/QLC running in 1-bit-per-cell mode for speed). When the cache is full the drive falls back to native TLC/QLC speeds, which can be 5–10× slower for QLC. As the drive fills, two things happen at once: the SLC cache shrinks (because it was carved from free TLC pages), and **garbage collection** has fewer empty blocks to consolidate, so write amplification rises. This is why **keeping ≥ 15 % free** is not optional advice — it preserves both the SLC cache and the garbage collector's working room.

# Part 4 — Interfaces and Protocols: SATA AHCI vs NVMe over PCIe

Throughput is set by the **physical bus**; latency and IOPS are set by the **protocol** sitting on top of it. SATA + AHCI was designed in 2003 around the assumption that storage was slow and serial; NVMe was designed in 2011 around the assumption that storage is fast and massively parallel. The architectural gap is enormous.

![NVMe vs SATA architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig5_nvme_vs_sata.png)

| Aspect | SATA + AHCI | NVMe over PCIe |
| --- | --- | --- |
| Command queues | 1 | up to 65 535 |
| Queue depth | 32 | up to 65 536 per queue |
| MSI-X interrupts | 1 | one per queue |
| Per-command overhead | ~6 µs | < 3 µs |
| Bus | SATA III, 6 Gb/s, half-duplex | PCIe Gen 3/4/5 ×4, full-duplex |
| Useful bandwidth | ~550 MB/s | 3.5 / 7 / 12 GB/s |

Notice that NVMe's parallelism (65 535 queues) is what lets a modern SSD saturate a million IOPS — every CPU core can drive its own private queue without locking, which AHCI's single command queue made structurally impossible.

![Storage interface evolution, 1986 → 2022](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig6_interface_evolution.png)

The bandwidth log-plot above tells a clean story. From PATA-133 (1986) to SATA III (2009) the storage interface gained roughly 4× in 23 years. From SATA III to NVMe Gen 5 (2022) it gained another **27×** in just 13 years — once the protocol was no longer the bottleneck, every PCIe generation translated directly into storage bandwidth. The pattern that matters: *interface bandwidth doubles roughly every two PCIe generations (≈4 years), and SSD controllers track it within ~12 months.*

## When NVMe actually beats SATA in practice

For a queue-depth-1 desktop workload (booting, opening apps), NVMe Gen 3 is about 30–40 % faster than SATA — noticeable but not transformative. The big wins arrive when:

- Queue depth ≥ 4 (compiling, virtualisation, databases, video editing scratch).
- Multiple processes contending for the drive simultaneously.
- Single-file transfers larger than the SLC cache (typically > 50 GB).

For a pure boot drive on a budget build, a SATA SSD captures roughly 80 % of the perceptual benefit at 50 % of the cost.

# Part 5 — RAID: Trading Capacity for Fault Tolerance

A RAID array combines several physical disks into one logical volume. Different RAID **levels** make different trade-offs between three knobs: **capacity utilisation**, **performance**, and **fault tolerance**. The five levels you will meet in practice:

![RAID levels — striping, mirroring, parity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig3_raid_levels.png)

| Level | Min disks | Usable capacity | Tolerates | Best for |
| --- | --- | --- | --- | --- |
| **RAID 0** — striping | 2 | 100 % | 0 disk failures (any failure = total loss) | Scratch, video edit cache, ephemeral data |
| **RAID 1** — mirroring | 2 | 50 % | 1 disk failure | Boot drive, small NAS, simple reliability |
| **RAID 5** — single parity | 3 | (n − 1)/n | 1 disk failure | File servers, balanced workloads |
| **RAID 6** — dual parity | 4 | (n − 2)/n | 2 disk failures | Large arrays (≥ 6 disks) where rebuild times are dangerous |
| **RAID 10** — mirror of stripes | 4 | 50 % | 1 per mirror pair (often 2 total) | Databases, VMs, mission-critical workloads |

## The RAID 5 rebuild problem

When a disk in a RAID 5 array fails, every other disk must be read end-to-end to reconstruct the missing data on the replacement. On a modern 16 TB HDD that takes 18–30 hours of constant heavy I/O — during which the surviving disks are most likely to fail (they are the same age, the same model, from the same batch). The probability of a second failure during rebuild for an array of seven 16 TB drives is empirically 1–3 %. This is why **RAID 6** (which tolerates two simultaneous failures) is the norm for any HDD array with capacity-per-drive > 4 TB. SSD arrays rebuild much faster (because of the bandwidth) and so RAID 5 remains acceptable on flash.

## RAID is not a backup

RAID protects against *disk* failure. It does not protect against:

- File deletion (the array faithfully replicates the deletion).
- Ransomware (the array replicates the encrypted blocks).
- Filesystem corruption, controller failure, fire, theft, lightning.

You still need backups. RAID buys you *uptime*; backups buy you *recoverability*.

# Part 6 — How a File System Turns Blocks into Files

A raw block device only knows how to read and write fixed-size sectors at logical block addresses. A **file system** is the layer that maps human concepts (files, directories, permissions, timestamps) onto those sectors. Almost every modern file system — ext4, XFS, NTFS, APFS, ZFS — uses the same architectural pieces.

![ext4-style file system layout — inodes, blocks, journaling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig4_filesystem.png)

The on-disk layout sketched above has three core pieces:

1. **Superblock and group descriptors** — a small region near LBA 0 holding "what is this file system, how big, where do the bitmaps and inode tables live."
2. **Inode table** — one fixed-size record per file (typically 256 B in ext4) containing the file's mode, owner, sizes, timestamps, and a list of pointers to its data blocks. The classic ext-family scheme uses **12 direct pointers** (covering the first 48 KiB of a file at 4 KiB blocks), one **indirect** pointer (a block of pointers, +4 MiB), one **double-indirect** (+4 GiB), and one **triple-indirect** (+4 TiB). Modern ext4 instead uses **extents** — `(start, length)` ranges — which scale better for large files.
3. **Data blocks** — the actual file content, allocated from the free space tracked by the *block bitmap*.

The names you see in `ls -li` — that integer next to each filename — is the inode number. A directory is just a small file whose contents are a list of `(filename, inode-number)` pairs. Hard links are simply two directory entries pointing at the same inode, which is why deleting a file does not free its blocks until the link count reaches zero.

## Why journaling exists

Suppose you write a 1 MiB file and the power fails halfway through. Without a journal, the file system might have updated the inode (size = 1 MiB) but not yet written the data blocks (which now contain garbage from a previous file), or updated the bitmap (marking blocks "in use") but not the inode (so the blocks leak). After reboot, `fsck` would have to scan the entire disk to find these inconsistencies — minutes on a small disk, hours on a multi-TB one.

A **journal** (or "write-ahead log") solves this by writing every metadata change to a small dedicated region *before* applying it to the main file system. The five-step sequence shown in the figure — *begin → write metadata to journal → commit → checkpoint to final location → free journal space* — guarantees that after a crash, replaying the journal returns the file system to a consistent state in seconds. ext4, NTFS, XFS and APFS all journal metadata; ZFS and Btrfs go further with **copy-on-write**, which makes every write atomic by writing to new blocks and only swapping the root pointer at the end.

# Part 7 — Storage Tiering: Putting Hot Data on Fast Bytes

Storage cost spans **eight orders of magnitude** — from DRAM at \$5/GB·month down to S3 Glacier Deep Archive at \$0.001/GB·month. No real workload accesses all of its data with the same frequency; a typical access pattern is "5 % of the data sees 80 % of the requests." **Tiering** is the practice of placing each byte on the cheapest medium that meets its latency requirement.

![Storage tiering — hot, warm, cool, cold](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig7_tiering.png)

| Tier | Typical media | Read latency | Cost (USD/GB·month) | Workload share |
| --- | --- | --- | --- | --- |
| **Hot** | DRAM, Optane, NVMe SSD | < 100 µs | 0.05 – 5 | ~5 % |
| **Warm** | SATA SSD, 15K SAS | 100 µs – 1 ms | 0.04 – 0.06 | ~15 % |
| **Cool** | 7.2K HDD (RAID 6) | 5 – 10 ms | 0.015 | ~30 % |
| **Cold** | S3 Standard-IA, Glacier | seconds – hours | 0.004 – 0.023 | ~50 % |

The right-hand chart in the figure plots latency vs cost on log–log axes; the points trace a near-linear frontier — every ten-fold reduction in latency costs roughly ten-fold more per gigabyte. Cloud platforms (AWS S3 Lifecycle, GCS Object Lifecycle, Azure Blob Tiers) automate the tiering by moving objects between classes on access-time rules. On a single workstation the same idea applies manually: keep the OS and active project on NVMe, the steam library on SATA SSD or HDD, and the archive on a NAS or cloud bucket.

# Part 8 — Operating Your Storage in Production

## The four optimisations every SSD wants

1. **4 KiB partition alignment.** Every modern partition table created since Windows 7 / Linux 2008 starts at sector 2048 (1 MiB offset), which is aligned to any reasonable NAND page size. Misalignment costs 30–35 % on small writes because each logical write straddles two physical pages.
2. **TRIM enabled.** When the OS deletes a file, TRIM tells the SSD which logical blocks are now garbage so they can be erased proactively. Without TRIM the controller has no idea which pages are stale and must do read-modify-write on them when reuse comes; sustained write speed can collapse from 500 to 150 MB/s within months. Verify with `fsutil behavior query DisableDeleteNotify` (Windows, expects `0`) or `lsblk --discard` (Linux, `DISC-MAX > 0`).
3. **Over-provisioning of 10–15 %.** Leaving 100 GB of a 1 TB SSD unpartitioned (or simply unused) dramatically improves the controller's freedom to do garbage collection in the background, stabilising sustained write performance and adding 20–30 % to wear life.
4. **Never defragment an SSD.** Random reads cost the same as sequential reads on flash, so defragmentation only burns write cycles for no benefit. Windows 8+ correctly disables defrag on SSDs and runs `Optimize-Volume -ReTrim` instead.

## Monitoring with SMART

SMART (Self-Monitoring, Analysis and Reporting Technology) exposes per-drive health attributes. The five worth alerting on:

| Attribute | What it means | Warn / critical |
| --- | --- | --- |
| `Reallocated_Sector_Ct` | Bad sectors remapped to spares | > 10 / > 100 |
| `Wear_Leveling_Count` (SSD) | Remaining endurance | < 20 % / < 10 % |
| `Current_Pending_Sector` | Sectors waiting to be remapped | > 0 / > 10 |
| `Reported_Uncorrect` | Errors ECC could not fix | > 0 / > 5 |
| `Temperature_Celsius` | Drive temperature | > 60 °C / > 70 °C |

```bash
# Linux
sudo apt install smartmontools
sudo smartctl -a /dev/nvme0n1            # full attribute dump
sudo smartctl -H /dev/nvme0n1            # quick PASSED/FAILED

# macOS
diskutil info disk0 | grep "SMART Status"

# Windows (PowerShell)
Get-PhysicalDisk | Get-StorageReliabilityCounter
```

A reallocated-sector count that climbs week-on-week is the single most reliable early-failure signal — any upward trend is "back up tonight, replace this month."

## The 3-2-1 backup rule

For data you cannot recreate (photos, source code, documents, database state):

- **3 copies** of the data (one live, two backups).
- On **2 different media** (e.g. local SSD + external HDD, or NAS + cloud).
- With **1 copy off-site** (cloud, or a drive at a different physical location).

For business-critical data, extend to **3-2-1-1-0**: add an *air-gapped* copy (offline, immune to ransomware) and require **0 errors** on a quarterly restore drill. Every backup strategy is theoretical until you have actually restored a file from it under time pressure.

# Decision Cheat Sheet

**Picking the right drive.**

| Use case | Recommended | Reason |
| --- | --- | --- |
| OS / boot | NVMe Gen 3 TLC, 500 GB – 1 TB | Best perceptual win; cache-friendly |
| Gaming | NVMe Gen 3 / 4 TLC, 1–2 TB | Asset streaming, level loads |
| Video editing | NVMe Gen 4 TLC, 2–4 TB + HDD archive | Sustained sequential writes |
| Photo / media archive | 7.2K HDD, 4–16 TB | Cost per TB |
| NAS / file server | HDD RAID 6 + SSD cache | Capacity + redundancy |
| Database / VM host | Enterprise NVMe, RAID 10 | Low-latency random I/O |
| Laptop | NVMe (the only sane choice) | Power, shock, silence |

**Picking the right RAID level.**

```
Need redundancy?
├─ No  → RAID 0     (fastest, zero protection — scratch only)
└─ Yes → How many disks?
         ├─ 2       → RAID 1   (mirror — simplest)
         ├─ 3–5 SSD → RAID 5   (single parity, balanced)
         ├─ 4+ HDD  → RAID 6   (dual parity — safer rebuild)
         └─ 4+      → RAID 10  (best perf + protection, 50 % overhead)
```

**Picking the right interface.**

```
> 4 TB single drive needed?
├─ Yes → HDD (cost dominates)
└─ No  → SSD
         ├─ Old motherboard, no M.2  → SATA SSD
         ├─ Boot/games on QD1 desktop → NVMe Gen 3 (sweet spot)
         ├─ Pro workstation, deep queues → NVMe Gen 4
         └─ Server / future-proofing  → NVMe Gen 5
```

# Mnemonic

> **HDD spins, SSD switches** — physics decides latency before any benchmark runs.
> **Bus sets bandwidth, protocol sets parallelism** — SATA caps at 0.5 GB/s, NVMe scales to 12.
> **TLC is forever for humans** — 1 PB endurance dwarfs any consumer write rate.
> **Align, TRIM, leave room** — three free settings that double sustained write life.
> **RAID is uptime, backups are recoverability** — never confuse the two.
> **Hot bytes on fast media, cold bytes on cheap media** — tier or pay.

**Next in the series:** Motherboard, Graphics & Expansion — chipsets, VRMs, PCIe lane allocation, GPU architecture, and BIOS configuration.
