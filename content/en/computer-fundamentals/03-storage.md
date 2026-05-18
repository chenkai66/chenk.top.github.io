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
![Computer Fundamentals (3): Storage Systems (HDD vs SSD) — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/illustration_1.png)

Why can a single SSD swap "resurrect" a five-year-old laptop? Why does a TLC drive rated for only 1 000 P/E cycles still last more than a decade for normal users? Why does a brand-new SSD that benchmarks at 3 500 MB/s sometimes collapse to 50 MB/s after a few weeks? This third instalment of the **Computer Fundamentals Deep Dive Series** answers those questions from first principles. We will look at how rotating magnetic platters compare with charge-trap NAND cells, how the bandwidth of an interface (SATA, PCIe Gen 3/4/5) interacts with the parallelism of a protocol (AHCI vs NVMe), how RAID levels trade capacity for fault tolerance, how a file system organises bytes on a raw block device, and how to keep all of this fast and safe in production.

---

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

### Why an SSD upgrade is the single biggest perceptual win

A user does not "feel" sequential bandwidth; they feel **random read latency** and **queue depth**. Booting Windows touches roughly 30 000 small files; launching Chrome touches another 4 000. On an HDD each of those touches pays a full seek; on an NVMe SSD they overlap inside thousands of in-flight commands. That is why moving from an HDD to *any* SSD typically cuts boot time by 4–7× — a far larger user-visible jump than doubling RAM or upgrading a CPU generation.

### Inside an HDD: platters, heads, and zones

A hard disk drive packs multiple circular platters on a single spindle, spinning at 5 400, 7 200, or 10 000 RPM. Each platter surface has its own read/write head on an actuator arm. The data surface is divided into concentric tracks, and each track is split into sectors (historically 512 bytes, modern drives use 4 KiB "Advanced Format" sectors).

**Seeking.** The actuator arm swings the head to the correct track. Full-stroke seek (innermost to outermost track) takes ~15 ms; a short seek across a few thousand tracks takes ~2 ms. Average seek (random) is roughly the midpoint: 4–8 ms.

**Rotational latency.** Once on the right track, the head must wait for the correct sector to rotate underneath it. At 7 200 RPM the platter completes one rotation every 8.33 ms, so average rotational latency is half that: ~4.17 ms. At 5 400 RPM it rises to ~5.56 ms.

**Zone Bit Recording (ZBR).** Outer tracks are physically longer, so they contain more sectors per track. A modern drive divides the platter into ~15–30 zones; the outermost zone delivers ~250 MB/s sequential, while the innermost may drop to ~100 MB/s. This is why a half-full HDD seems slower — the operating system filled the fast outer zones first.

**Shingled Magnetic Recording (SMR).** Some high-capacity consumer drives overlap write tracks like roof shingles to squeeze more tracks per platter. The trade-off: random writes require reading an entire band (30–50 MB), rewriting it, and flushing it back. This makes SMR drives unsuitable for NAS or database use, and explains the sudden write-speed cliff that surprises users who chose the cheapest 8 TB drive.

### Inside an SSD: dies, planes, blocks, and pages

An SSD controller sits between the host interface (SATA or NVMe) and a bank of NAND flash memory chips. Each NAND chip contains one or more **dies**, each die contains **planes** (typically 2 or 4), each plane contains **blocks** (thousands), and each block contains **pages** (128–512 pages, each 4–16 KiB).

The critical asymmetry: **you can read or write individual pages, but you can only erase entire blocks.** A block is typically 4–16 MB. This asymmetry drives nearly every piece of SSD firmware engineering: the Flash Translation Layer, garbage collection, wear leveling, and over-provisioning all exist to work around this page/block mismatch.

**Parallelism.** The controller can issue reads to multiple dies simultaneously — this is called *channel interleaving* and *die interleaving*. A consumer SSD typically has 4–8 NAND channels; an enterprise drive has 8–16. More channels = more parallelism = higher throughput. This is why a 256 GB SSD is often measurably slower than a 1 TB model of the same series: fewer dies means less parallelism for the controller to exploit.

## Part 2 — Storage Performance: Three Numbers That Matter

Almost every storage decision can be made by looking at three numbers — **throughput**, **IOPS**, and **latency** — at the right percentile. Average values lie; the long tail is what users notice.

![Throughput, IOPS, and latency on a log scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig2_performance.png)

- **Throughput (MB/s)** matters when you are moving large contiguous files: video edits, backups, game-asset patches. Notice how each interface generation (SATA → PCIe 3 → 4 → 5) roughly doubles the previous ceiling.
- **IOPS** matters for everything else: launching apps, compiling code, running a database, booting an OS. The HDD-to-SSD jump here is **three orders of magnitude** — 120 vs 95 000 — which is why even a budget SATA SSD feels transformative.
- **Latency** matters under load. A modern NVMe Gen 4 drive serves a 4 KiB random read in ~30 µs; the same operation on a 7 200 RPM HDD costs ~10 ms — a 300× gap, and one that gets *worse* on the HDD when many requests queue up.

A useful rule of thumb: *if your workload's queue depth is 1, latency dominates; if the queue is deep, IOPS and throughput dominate.* Consumer workloads are almost always queue-depth-1, which is why NVMe Gen 5 — whose advantage is mostly at QD32+ — is a marginal upgrade for a desktop user but a large one for a database server.

### Reading a vendor spec sheet honestly

Vendor numbers (e.g. "7 000 MB/s read, 1 000 000 IOPS") are measured with **128 KiB sequential** transfers and **QD256** workloads on a fresh drive with the SLC cache empty. Real desktop workloads see roughly:

- 50–70 % of the headline sequential number on sustained writes (after the SLC cache fills).
- 5–15 % of the headline IOPS at QD1.
- 2–4× the headline latency once the drive is more than 70 % full.

Always check the *sustained write speed past the SLC cache* and the *latency at QD1* — those are the numbers that determine how the drive feels.

## Part 3 — Interfaces and Protocols: SATA, NVMe, and Why They Are Not the Same Axis

Two separate things determine how fast a drive talks to the CPU: the **physical bus** (the wires) and the **protocol** (the language). Conflating them is the most common storage misconception.

| Bus | Max bandwidth | Typical protocol | Queue depth limit |
| --- | --- | --- | --- |
| SATA III | 600 MB/s | AHCI | 32 |
| PCIe 3.0 ×4 | 3 940 MB/s | NVMe | 65 535 |
| PCIe 4.0 ×4 | 7 880 MB/s | NVMe | 65 535 |
| PCIe 5.0 ×4 | 15 760 MB/s | NVMe | 65 535 |

**SATA + AHCI.** The SATA cable provides 600 MB/s of half-duplex bandwidth. AHCI (Advanced Host Controller Interface) was designed in 2004 for hard disks; it manages a single command queue of 32 entries. For an HDD that processes one command at a time, 32 is plenty. For an SSD that can handle thousands of in-flight reads, 32 is a choking bottleneck.

**NVMe.** NVMe (Non-Volatile Memory Express) was designed from scratch for NAND-based storage over PCIe. It supports 65 535 queues, each with 65 536 entries. It talks directly to the CPU over PCIe without the legacy AHCI layer. The result: lower software latency (2.8 µs vs 6 µs per command), vastly higher parallelism, and bandwidth that scales with the PCIe generation.

**Why a SATA SSD still feels fast.** Even though a SATA SSD is capped at 550 MB/s sequential, its random read latency (~100 µs) is still 100× better than an HDD (~10 ms). Most consumer workloads are latency-bound, not throughput-bound. A SATA SSD is a perfectly valid upgrade for a laptop that has no M.2 slot.

### Physical form factors

| Form factor | Connector | Bus | Use case |
| --- | --- | --- | --- |
| 2.5-inch SATA | SATA data + power | SATA III | Upgrade for older laptops/desktops |
| M.2 2280 (B+M key) | M.2 slot | SATA III | Compact laptops, budget builds |
| M.2 2280 (M key) | M.2 slot | PCIe ×4 | Modern desktop/laptop (NVMe) |
| U.2 | U.2 connector | PCIe ×4 | Enterprise / server 2.5-inch NVMe |
| E1.S / E3.S | EDSFF connector | PCIe ×4 | Next-gen datacentre |
| AIC (Add-in Card) | PCIe slot | PCIe ×4/×8/×16 | High-perf server, workstation |

The form factor does *not* determine the protocol. An M.2 slot can carry SATA or NVMe — check the key notch and the motherboard spec. Plugging a SATA M.2 drive into an NVMe-only M.2 slot (or vice versa) results in a drive that is not detected.

## Part 4 — SSD Internals: NAND, FTL, and Why TLC "Only" 1 000 P/E Cycles Lasts Forever

A NAND cell is a transistor whose threshold voltage can be programmed. SLC stores one bit (two voltage levels), MLC two bits (four levels), TLC three (eight), QLC four (sixteen). Packing more levels into the same physical cell roughly halves cost per gigabyte but **quarters endurance and doubles read latency** at every step.

| NAND type | Bits / cell | Voltage levels | P/E cycles | Relative cost / GB | Where it ships today |
| --- | --- | --- | --- | --- | --- |
| **SLC** | 1 | 2 | ~100 000 | 6× | Industrial / boot SSD caches |
| **MLC (eMLC)** | 2 | 4 | 3 000–10 000 | 3× | Enterprise (rare in consumer) |
| **TLC** | 3 | 8 | 1 000–3 000 | 1.0× | Mainstream consumer & datacentre |
| **QLC** | 4 | 16 | 500–1 000 | 0.7× | Budget bulk storage |

### The lifespan calculation that always surprises people

The endurance figure looks alarming until you do the arithmetic for a real workload. A 1 TB TLC SSD with 1 000 P/E cycles can write 1 PB before wear-out. A heavy desktop user writes ~50 GB per day; a typical user writes 5–15 GB per day:

| Usage profile | Daily writes | Time to exhaust 1 PB TBW | Time to exhaust 600 TBW (rated) |
| --- | --- | --- | --- |
| Light office / web | 5 GB | 548 years | 329 years |
| Developer (compiles, VMs) | 30 GB | 91 years | 55 years |
| Heavy creative (video editing) | 100 GB | 27 years | 16 years |
| Database server | 500 GB | 5.5 years | 3.3 years |

Even the "heavy creative" profile exceeds any realistic drive lifetime by a wide margin. The drive's electronics or firmware will fail long before the NAND wears out. This is why TLC is the correct default for everything except the database-server profile, where enterprise MLC or Intel Optane (RIP) made sense.

### The Flash Translation Layer (FTL)

The FTL is the firmware layer that translates logical block addresses (LBAs) from the host into physical page locations on the NAND. It exists because of the read-write-erase asymmetry: you can write to a clean page, but you cannot overwrite a dirty page — you must erase the entire block first.

The FTL maintains a **mapping table** (logical page → physical page) in DRAM. When you write LBA 1000, the FTL does not overwrite the old physical page at LBA 1000. Instead, it writes to a new clean page, updates the mapping table, and marks the old page as *stale*. This is called **log-structured writing** and it is the same trick that log-structured file systems (LFS) and LSM-tree databases use.

**Why the mapping table needs DRAM.** A 1 TB drive with 4 KiB pages has 256 million entries. At 4 bytes each, the mapping table is 1 GB. This is why SSDs ship with DRAM (typically 1 GB per 1 TB) — and why DRAM-less SSDs are measurably slower for random workloads: they must fetch mapping entries from NAND, adding a read latency to every address translation.

**HMB (Host Memory Buffer).** Budget NVMe drives without onboard DRAM can borrow a small chunk (32–64 MB) of the host's system RAM to cache the hot portion of the mapping table. This largely closes the gap for consumer workloads where the working set of LBAs is small.

### Garbage collection

When the FTL runs low on clean blocks, it must reclaim blocks that contain a mix of valid and stale pages. The process: read all valid pages from a partially-stale block, write them to a new clean block, then erase the old block.

This has two consequences:

1. **Write amplification (WA).** The drive writes more data to NAND than the host actually wrote. WA of 2× means every 1 GB from the host becomes 2 GB on NAND. High WA wastes endurance and reduces sustained write speed.

2. **Background activity.** GC runs in the background during idle time, which is why leaving an SSD powered on and idle periodically is healthy. Drives that are always under load (24/7 database) have to GC under load, which causes latency spikes.

### TRIM: the OS telling the SSD what's deleted

When you delete a file, the file system marks the LBAs as free but does not tell the drive. The FTL thinks those pages are still valid and will copy them during garbage collection — pointless work that increases write amplification.

**TRIM** (called UNMAP in SCSI, DEALLOCATE in NVMe) is the command the OS sends to the drive to say "these LBAs are no longer in use." The FTL can then mark those pages as stale immediately, so the next garbage collection pass skips them.

**Practical impact.** Without TRIM, a drive that has been in use for months will have a severely inflated valid-page count. GC runs more often, copies more data, and sustained write speed drops — sometimes dramatically. This is the "new SSD that gets slower after a few weeks" phenomenon. Enabling TRIM (it's on by default in modern Windows, Linux, and macOS) is free performance.

```bash
# Linux: verify TRIM is enabled for a mounted filesystem
lsblk --discard
# The DISC-GRAN and DISC-MAX columns should be non-zero

# Manually trigger TRIM on all mounted filesystems
sudo fstrim -av
```

### Wear leveling

The FTL distributes writes evenly across all NAND blocks so that no single block wears out early. Two types:

- **Dynamic wear leveling** moves *new writes* to the least-worn blocks. Simple but does not help with blocks that were written once (e.g. your OS partition) and never touched again — those blocks age at zero while others wear out.
- **Static wear leveling** periodically moves cold (rarely-written) data to more-worn blocks and moves hot data to less-worn blocks. This equalises wear across the entire drive, including blocks that hold static data. All modern SSDs do this.

### Over-provisioning

Consumer SSDs reserve 7–28 % of the raw NAND capacity as invisible spare area. A "1 TB" drive actually has 1 024 GiB or more of raw NAND but presents 931 GiB to the host. The spare area serves three purposes:

1. **Clean blocks for GC.** The FTL always has somewhere to write during garbage collection without stalling the host.
2. **Replacement blocks.** When a block fails (the error rate exceeds correction), the FTL retires it and maps in a spare.
3. **Write amplification reduction.** More spare blocks means GC can be pickier about which blocks to reclaim, reducing unnecessary data movement.

Enterprise SSDs often ship with 28 % over-provisioning; consumer drives use 7 %. You can artificially increase OP by leaving unpartitioned space at the end of the drive — 10–20 % is a common recommendation for write-heavy workloads.

### The SLC cache trick

Most modern TLC and QLC SSDs use a portion of their NAND in SLC mode (one bit per cell instead of three or four). This "SLC cache" delivers high burst write speeds: a TLC cell written in SLC mode is 3× faster and lasts 30–100× longer.

The catch: SLC mode uses 3× the physical space. A 1 TB TLC drive with a 100 GB dynamic SLC cache is really using 300 GB of physical NAND for that cache. Once the cache fills, the drive must flush (fold) SLC pages back into TLC, and new writes go directly to TLC at the much slower native speed. This is the "SSD write speed cliff" that shows up in sustained write benchmarks around the 30–120 GB mark.

| Cache type | Size behaviour | Pro | Con |
| --- | --- | --- | --- |
| **Static SLC cache** | Fixed (e.g. 12 GB) | Predictable; no folding surprise | Small; fills fast on large writes |
| **Dynamic SLC cache** | Grows with free space | Large initial burst | Shrinks as drive fills; folding causes latency |
| **Full-drive SLC** | Entire drive in SLC | Fastest writes always | Halves usable capacity; some drives do this when very empty |

## Part 5 — 4K Alignment and SSD Optimization

### What 4K alignment is and why it matters

NAND pages are typically 4 KiB (some newer NAND uses 8 KiB or 16 KiB pages). If the file system's allocation unit (cluster) is not aligned to the NAND page boundary, a single 4 KiB logical write can span two physical pages, doubling the write amplification and halving IOPS for small random writes.

Modern operating systems (Windows 7+ and every modern Linux distro) align partitions to 1 MiB boundaries by default, which guarantees 4K alignment. The problem arises with:

- Drives cloned from older systems (Windows XP used 63-sector offsets)
- Manually partitioned drives with non-standard alignment
- Virtual disk images on misaligned host partitions

```bash
# Linux: check alignment
sudo parted /dev/sda align-check optimal 1
# Output: "1 aligned" means correct

# Windows (PowerShell):
Get-Partition | Format-Table DiskNumber, PartitionNumber, Offset
# Offset should be divisible by 4096 (and ideally by 1048576)
```

### Five SSD optimization practices

| Practice | What to do | Why |
| --- | --- | --- |
| **Enable TRIM** | Verify `discard` mount option or `fstrim.timer` | Prevents GC from copying deleted data |
| **Keep 10-20% free** | Don't fill past 80% | Gives GC room to work without stalling writes |
| **Disable defragmentation** | Turn off scheduled defrag for SSDs | Defrag creates pointless writes; SSDs have no seek penalty |
| **Enable AHCI/NVMe mode** | Check BIOS — not IDE/legacy mode | IDE mode disables TRIM and NCQ |
| **Update firmware** | Check manufacturer's tool periodically | FTL bugs are common in first-year firmware |

### Monitoring SSD health

Every SSD exposes **SMART attributes** that tell you how much life remains:

```bash
# Linux: install smartmontools
sudo smartctl -a /dev/nvme0n1

# Key attributes to watch:
# - Percentage Used:         how much of rated endurance is consumed
# - Available Spare:         remaining spare blocks (should be >5%)
# - Media and Data Integrity Errors:  should be 0
# - Data Units Written:     total bytes written (for your own WA calculation)
```

On Windows, CrystalDiskInfo provides a friendly GUI for the same data.

## Part 6 — RAID: Trading Disks for Fault Tolerance

RAID (Redundant Array of Independent Disks) combines multiple physical drives into a single logical volume. The levels differ in how they distribute data and parity.

![RAID Level Decision Tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig_raid_decision_en.png)

| Level | Min disks | Usable capacity | Fault tolerance | Read speed | Write speed | Use case |
| --- | --- | --- | --- | --- | --- | --- |
| **RAID 0** | 2 | 100% | 0 drives | N× | N× | Scratch space, temp data |
| **RAID 1** | 2 | 50% | 1 drive | 2× read | 1× | OS drive, boot mirror |
| **RAID 5** | 3 | (N-1)/N | 1 drive | (N-1)× | Moderate | General file server |
| **RAID 6** | 4 | (N-2)/N | 2 drives | (N-2)× | Lower | Large arrays (>4 drives) |
| **RAID 10** | 4 | 50% | 1 per mirror | N× read | (N/2)× | Database, VM storage |

### RAID 0: striping for speed, zero safety

Data is split across drives in fixed-size stripes. If any drive fails, the entire array is lost. Use only for data you can regenerate (build caches, render farms, game installs).

### RAID 1: simple mirroring

Every write goes to both drives. Either drive can serve reads. Usable capacity is 50%. The simplest and most reliable form of redundancy. For a two-drive desktop or a server boot volume, RAID 1 is often the best answer.

### RAID 5: distributed parity

Data and parity are striped across all drives. One drive's worth of capacity holds parity; the rest holds data. Can survive one drive failure. The rebuild process reads every surviving drive, which is where RAID 5 becomes risky with large drives: rebuilding a 16 TB drive takes 12–24 hours, during which a second failure destroys the array. For arrays with drives larger than 4 TB, RAID 6 is the safer choice.

### RAID 6: double parity

Like RAID 5 but with two parity blocks per stripe. Survives two simultaneous drive failures. The write penalty is higher (each write touches two parity blocks), but for large arrays this is the minimum acceptable level of protection.

### RAID 10: mirrors of stripes

Pairs of mirrored drives are striped together. Combines the speed of RAID 0 with the fault tolerance of RAID 1. Can survive one failure in each mirror pair. The best choice for write-heavy workloads (databases, virtual machines) where both performance and safety matter.

### Hardware vs software RAID

| Type | Controller | CPU overhead | Boot support | Cost |
| --- | --- | --- | --- | --- |
| Hardware RAID | Dedicated card with battery-backed cache | None | Yes | $200–$2000 |
| Software RAID (mdadm, ZFS) | OS kernel | Moderate | Yes (with config) | Free |
| Fake RAID (motherboard) | BIOS + driver | High | Limited | Free |

Modern CPUs are fast enough that software RAID (Linux mdadm, ZFS, Windows Storage Spaces) is a perfectly valid choice for most servers. The battery-backed cache on hardware RAID controllers is the one feature software RAID cannot replicate — it protects against data loss during a power failure mid-write.

### RAID is not backup

This deserves its own heading because the misconception is so common. RAID protects against **drive failure**. It does not protect against:

- Accidental deletion (the delete propagates to all mirrors instantly)
- Ransomware (the encryption propagates to all mirrors)
- Controller failure (the array metadata may be unreadable by a different controller)
- Fire, theft, flood (all drives are in the same machine)

**Backup** means a copy on different media, in a different location, with versioning. RAID and backup serve different functions; you need both.

## Part 7 — File Systems: How Bytes Become Files

A file system is the software layer that maps filenames and directory structures onto a flat sequence of blocks on a storage device. The choice of file system affects performance, reliability, and feature set.

| File system | OS | Max file size | Max volume | Journaling | COW | Use case |
| --- | --- | --- | --- | --- | --- | --- |
| **NTFS** | Windows | 16 EB | 256 TB | Yes | No | Windows system drive |
| **ext4** | Linux | 16 TB | 1 EB | Yes | No | Linux default, general purpose |
| **XFS** | Linux | 8 EB | 8 EB | Yes | No | Large files, high throughput |
| **Btrfs** | Linux | 16 EB | 16 EB | Yes | Yes | Snapshots, checksums, flexible |
| **ZFS** | Linux/BSD | 16 EB | 256 ZB | Yes | Yes | Enterprise, self-healing, RAIDZ |
| **APFS** | macOS | 8 EB | 8 EB | Yes | Yes | macOS/iOS, SSD-optimized |
| **FAT32** | Any | 4 GB | 2 TB | No | No | USB drives, cross-platform compat |
| **exFAT** | Any | 16 EB | 128 PB | No | No | SD cards, large USB drives |

### Journaling

A journaling file system writes a log (journal) of pending changes before committing them to the main data structures. If power is lost mid-write, the journal allows the file system to replay or discard the incomplete operation on next mount — avoiding the multi-hour `fsck` that older file systems required.

**Metadata-only journaling** (ext4 default) journals file system metadata (directory entries, inode tables) but not file data. This protects against structural corruption but not against partial file writes.

**Full journaling** (ext4 with `data=journal`) journals both metadata and data. Safer but slower, because every byte is written twice: once to the journal, once to its final location.

### Copy-on-Write (COW)

COW file systems (ZFS, Btrfs, APFS) never overwrite data in place. Instead, they write the new version to a new location and atomically update the pointer. This enables:

- **Free snapshots.** A snapshot is just a frozen pointer tree; the data is shared until modified.
- **Self-healing.** Every block is checksummed. If a read returns corrupted data and a mirror or parity copy exists, the file system silently repairs it.
- **Atomic multi-file updates.** Entire transactions can succeed or fail as a unit.

The trade-off: COW file systems fragment more aggressively under random write workloads, and their metadata overhead is higher. ZFS mitigates this with its ARC cache and large record sizes; Btrfs mitigates it with autodefrag.

## Part 8 — Storage Tiering: Hot, Warm, Cold

Not all data is accessed equally. Storage tiering places frequently-accessed (hot) data on fast, expensive media and rarely-accessed (cold) data on slow, cheap media.

| Tier | Media | Latency | Cost / TB | Example data |
| --- | --- | --- | --- | --- |
| **Hot** | NVMe SSD | ~30 µs | $110 | Active databases, working set |
| **Warm** | SATA SSD | ~100 µs | $60 | Logs, recent backups, dev environments |
| **Cold** | HDD | ~10 ms | $15 | Archives, compliance, old media |
| **Archive** | Tape / cloud cold | seconds–hours | $3 | Legal holds, regulatory, disaster recovery |

### Automated tiering in practice

Modern storage arrays (NetApp, Pure Storage, Ceph) and cloud object stores (S3 Intelligent-Tiering, Alibaba Cloud OSS lifecycle rules) automatically move objects between tiers based on access patterns.

For a home or small-server setup, the simplest tiering is:

1. OS + apps on NVMe SSD (256 GB – 1 TB).
2. Active project files on SATA SSD (1–4 TB).
3. Media library, backups, and archives on HDD (4–16 TB).
4. Off-site backup on cloud cold storage (unlimited).

This four-tier layout costs less than putting everything on NVMe and performs better than putting everything on HDD.

## Part 9 — Choosing the Right Storage: A Decision Guide

![Storage Media Selection Guide](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/03-storage/fig_ssd_decision_en.png)

### Desktop / laptop buyer's checklist

1. **Does your machine have an M.2 NVMe slot?** → Get a PCIe Gen 4 NVMe SSD. Gen 5 is a marginal consumer upgrade and costs 2× more. Gen 3 is fine if Gen 4 isn't supported.
2. **No M.2 slot?** → Get a 2.5-inch SATA SSD. Still a massive upgrade from HDD.
3. **Need bulk storage for media?** → Add an internal HDD or an external USB 3.2 HDD enclosure.
4. **How much capacity?** → 512 GB minimum for OS + apps; 1 TB is the sweet spot in 2026 at ~$60; 2 TB if you do video editing or game hoarding.

### Server / NAS buyer's checklist

1. **Primary storage for databases or VMs?** → NVMe SSD in RAID 10.
2. **General file server?** → SATA SSD in RAID 5/6, or HDD in RAID 6 if capacity > 20 TB.
3. **Backup target?** → HDD in RAID 6 or JBOD with ZFS RAIDZ2.
4. **Write-heavy workload?** → Check the drive's sustained write speed past SLC cache. Enterprise drives with power-loss protection. Avoid QLC and SMR.

## Mnemonic

> **HDD spins, SSD switches** — physics decides latency before any benchmark runs.
> **Bus sets bandwidth, protocol sets parallelism** — SATA caps at 0.5 GB/s, NVMe scales to 12.
> **TLC is forever for humans** — 1 PB endurance dwarfs any consumer write rate.
> **Align, TRIM, leave room** — three free settings that double sustained write life.
> **RAID is uptime, backups are recoverability** — never confuse the two.
> **Hot bytes on fast media, cold bytes on cheap media** — tier or pay.

**Next in the series:** Motherboard, Graphics & Expansion — chipsets, VRMs, PCIe lane allocation, GPU architecture, and BIOS configuration.
