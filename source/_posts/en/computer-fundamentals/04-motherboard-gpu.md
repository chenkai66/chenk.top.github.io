---
title: "Computer Fundamentals: Motherboard, Graphics, and Expansion"
date: 2024-05-06 09:00:00
tags:
  - Computer Hardware
  - System Architecture
categories: Computer Fundamentals
series: "Computer Fundamentals"
series_order: 4
series_total: 6
lang: en
mathjax: false
description: "Read a motherboard floor plan, understand PCIe lane allocation across CPU and chipset, follow the SIMT execution model that makes GPUs fast, compare DDR/GDDR/HBM memory, and pick the right display interface."
disableNunjucks: true
---

A modern desktop motherboard is an unusually honest object. Every important design decision — how many PCIe lanes the CPU exposes, which slots are wired straight to the CPU and which tunnel through the chipset, how the VRM is sized to feed a 250 W processor, why the second long PCIe slot only runs at ×4 — is laid out in plain copper on the PCB. If you can read the board, you can predict almost every performance cliff a user will hit. This fourth instalment of the **Computer Fundamentals Deep Dive Series** teaches that reading skill, then turns it inward to the GPU, where the same lesson applies in miniature: a GPU is a chip whose entire architecture exists to keep thousands of arithmetic lanes fed with data, and almost everything else — caches, schedulers, tensor cores, HBM stacks — is in service of that goal.

# Series Navigation

**Computer Fundamentals Deep Dive Series** (6 parts)

1. CPU & Computing Core
2. Memory & High-Speed Cache
3. Storage Systems
4. **→ Motherboard, Graphics & Expansion** (PCIe, GPU, displays, chipset) ← *You are here*
5. Network, Power & Practical Troubleshooting
6. Deep-Dive Appendix

# Part 1 — Reading a Motherboard

The motherboard is not really one bus; it is **two bus domains stitched together by a single high-speed link**. The CPU has a small, fast budget of PCIe lanes and DDR channels that it owns directly. Everything else — extra storage slots, USB controllers, SATA ports, the chipset's own PCIe — hangs off a second domain run by the chipset (Intel calls it the PCH; AMD calls it the FCH). The two domains talk through one link: Intel's **DMI 4.0 ×8** carries roughly 16 GB/s in each direction, AMD uses a comparable PCIe ×4 link. That single link is the most important number on the board, because every device on the chipset side fights for it.

![Motherboard floor plan with CPU-direct vs chipset domains called out](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig1_motherboard_layout.png)

Five regions repay study:

- **CPU socket and VRM.** LGA 1700 and AM5 sockets pull up to 250 W under full load. The vertical strip of MOSFETs and chokes immediately around the socket is the **VRM**, which converts the 12 V coming from the EPS 8-pin connector into the 1.0–1.4 V the CPU core actually wants. A board with too few phases for its CPU is the most common cause of "throttles under sustained load" reviews.
- **DIMM slots.** Always two channels on consumer boards (slots A1/A2 + B1/B2). Populating only A2 + B2 (the slots furthest from the CPU on most boards) gives dual-channel; populating A1 + A2 silently halves your bandwidth. DDR5 modules carry a tiny on-DIMM voltage regulator, which is why DDR5 boards have fewer memory phases than DDR4 boards did.
- **The first long PCIe slot.** Wired directly to the CPU's PCIe root complex; on Z790/X670E it is PCIe **5.0 ×16**. This is where the GPU goes — full bandwidth, no chipset hop.
- **M.2_1.** Almost always also CPU-direct (PCIe ×4). This is your fastest NVMe slot.
- **The chipset and everything below it.** The second long PCIe slot, M.2_2, the SATA ports, the rear USB stack, Wi-Fi, audio — all of it sits on the chipset side of the DMI link.

Two practical consequences follow immediately. First, **the second physical ×16 slot is rarely a real ×16**: on B760/B660 it is wired ×4 through the chipset and shares bandwidth with USB and SATA. Plug a second GPU there and it will run, but throttled. Second, **populating M.2_2 often disables two SATA ports** because the chipset multiplexes those lanes; the manual will tell you which ports drop.

## How to verify the slot you actually got

You bought a board advertised as "PCIe 5.0 ×16". You want to know it's actually delivering ×16. On Windows the easiest check is **GPU-Z**: the *Bus Interface* field reports negotiated width and generation as `PCIe x16 5.0 @ x16 5.0`. On Linux, `lspci -vv | grep -i lnk` shows `LnkCap` (what the slot can do) versus `LnkSta` (what it negotiated). If `LnkSta` is narrower or older than `LnkCap`, something downstream — riser cable, BIOS setting, slot misuse — is forcing a downgrade.

# Part 2 — PCIe Generations & Lane Allocation

PCIe is the universal expansion bus. Two parameters matter: the **generation** (which sets per-lane bandwidth) and the **width** (how many lanes are aggregated). Each generation almost exactly doubles the previous one's per-lane rate, so a Gen 4 ×4 NVMe SSD has the same theoretical bandwidth as a Gen 3 ×8 device.

![PCIe 2.0 → 5.0 bandwidth at ×1, ×4, ×16 plus real device requirements](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig2_pcie_generations.png)

| Generation | Year | Per-lane (GB/s) | ×4 (NVMe) | ×16 (GPU) |
| --- | --- | --- | --- | --- |
| PCIe 2.0 | 2007 | 0.5 | 2 | 8 |
| PCIe 3.0 | 2010 | 1.0 | 4 | 16 |
| PCIe 4.0 | 2017 | 2.0 | 8 | 32 |
| PCIe 5.0 | 2022 | 4.0 | 16 | 64 |
| PCIe 6.0 | 2025 | 8.0 | 32 | 128 |

The headline numbers are per-direction; PCIe is full-duplex, so the aggregate is double again. The figures above use the per-direction convention because that's what GPU-Z and `lspci` report.

**The crucial insight is that consumer GPUs have not yet caught up with PCIe 4.0 ×16.** An RTX 4090 averages roughly 22 GB/s on a sustained game workload — well under the 32 GB/s ceiling of Gen 4. PCIe 5.0 first matters for **NVMe SSDs** (where a single drive can saturate ×4 Gen 4 at 7 GB/s and Gen 5 doubles the headroom), not for GPUs. If you are building today and have to choose, pay for Gen 5 NVMe support before you pay for Gen 5 GPU support.

## CPU lanes are scarcer than chipset lanes

A typical consumer CPU exposes **20 PCIe 5.0 lanes**: 16 for the GPU slot, 4 for the primary M.2. That is the entire CPU-direct budget. Everything else lives behind the chipset, which then re-fans out additional PCIe 4.0 lanes — often 20 or more on Z790 — but they all share the DMI link back to the CPU. The arithmetic is brutal: a Z790 chipset can publish 20 downstream lanes while only having 16 GB/s upstream. On a quiet system this is invisible; the moment you simultaneously stress two NVMe drives and a 10 GbE NIC behind the chipset, you hit the DMI ceiling and they all slow down together.

This is why workstation platforms (Threadripper, Xeon-W) cost so much: they expose 64+ PCIe lanes from the CPU itself, with no shared chokepoint.

## What different devices actually need

| Device | Practical width | Sustained bandwidth | Where it should go |
| --- | --- | --- | --- |
| Modern GPU (RTX 4090 / RX 7900 XTX) | ×16 | 18–22 GB/s | First long slot, CPU-direct |
| NVMe Gen 5 SSD | ×4 | 12 GB/s | M.2_1, CPU-direct |
| NVMe Gen 4 SSD | ×4 | 7 GB/s | Any M.2 ×4 slot |
| 10 GbE / 25 GbE NIC | ×4 (×8 for 25G) | 1.25–3.1 GB/s | Any open ×4+ slot |
| Capture card / USB 4 add-in | ×4 | up to 5 GB/s | Open ×4+ slot |
| Sound card, USB 2 hub | ×1 | <0.5 GB/s | Any ×1 slot |

The takeaway: **for a single-GPU + dual-NVMe gaming build, even a B660/B650 board has plenty of bandwidth**. The reason to step up to Z790/X670E is not GPU performance — it is the second CPU-direct M.2 slot, the higher VRM phase count for K-series CPUs, and richer rear I/O (USB 3.2 Gen 2×2, 2.5 GbE).

# Part 3 — From Northbridge to PCH: How the Chipset Got Smaller

The motherboard you can buy in 2025 has roughly half as many big chips on it as the one you could buy in 2005. That is not a cost-cutting story; it is the story of integration absorbing the high-speed parts of the chipset into the CPU package, leaving the chipset to do only the slow-and-numerous work.

![Legacy northbridge/southbridge architecture vs modern PCH](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig7_chipset_evolution.png)

In the legacy design (left), the **Front-Side Bus (FSB)** connected the CPU to the **northbridge**, which housed the memory controller and the AGP/PCIe ×16 link. The **southbridge** sat below the northbridge on a slower hub-link and ran USB, SATA, PCI and audio. Memory bandwidth was bottlenecked by the FSB, and the northbridge ran hot enough to need its own heatsink.

In the modern design (right), the CPU package contains the **integrated memory controller (IMC)**, the **PCIe root complex**, the iGPU, and a chunk of L3 cache. RAM and the GPU now talk *directly* to the CPU. The chipset (Intel PCH, AMD FCH) is downgraded to a "fan-out" role: it takes the DMI ×8 link upstream and turns it into a pile of PCIe 4.0 ×1/×4 lanes, SATA ports, USB controllers and Wi-Fi connectivity. It runs cool enough that it often has no heatsink at all, just a thin metal cover for aesthetics.

The architectural payoff is that **CPU-attached devices** (GPU, primary NVMe, RAM) get full bandwidth and the lowest possible latency, while a second tier of slower-but-numerous devices coexists peacefully behind the chipset. The cost is the bandwidth ceiling on that second tier, which is why we keep coming back to "the DMI link is the most important number on the board."

# Part 4 — Inside the GPU: SIMT, Warps and Streaming Multiprocessors

A CPU and a GPU are both "processors", but they are designed against opposite optimisation targets. A CPU is built to **finish a single thread as quickly as possible** — caches, branch predictors, out-of-order execution, deep pipelines all serve that goal. A GPU is built to **keep thousands of arithmetic lanes busy at all times**, even if any individual lane sits idle waiting for memory. The architectural model that delivers this is **SIMT — Single Instruction, Multiple Threads**.

![GPU die: shared L2, eight Streaming Multiprocessors, warp scheduler, CUDA / Tensor / RT cores](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig3_gpu_simt.png)

The hierarchy reads from the bottom up:

1. **A CUDA lane** (NVIDIA) or **stream-processor lane** (AMD) is one floating-point ALU. It executes one element of an arithmetic operation per cycle.
2. **A warp** is 32 lanes (NVIDIA) or 32–64 lanes (AMD "wavefront") **executing the same instruction in lock-step** on different data. This is the SIMT trick: you only need one instruction decoder per 32 ALUs.
3. **A Streaming Multiprocessor (SM)** holds the resources to keep many warps in flight at once: register file, L1/shared memory, and a warp scheduler that picks a *ready* warp every cycle. While one warp is stalled on memory, the scheduler runs another. This is **latency hiding by parallelism**, and it is the GPU's actual super-power.
4. **The GPU die** ties 40–144 SMs together with a shared L2 cache (32–96 MB on modern chips) and a wide bus to off-chip memory.

Two specialised core types appear inside each modern SM:

- **Tensor cores** (NVIDIA) / **AI accelerators** (AMD) execute small matrix-multiply-and-accumulate operations in one shot. A Tensor core can do a 4×4 FP16 matmul per cycle, which is why DLSS, Stable Diffusion and LLM inference are so much faster on RTX cards than on CUDA cores alone.
- **RT cores** accelerate ray-triangle and ray-box intersection tests in hardware. Without them, ray tracing falls back to general shaders and runs an order of magnitude slower.

The numbers on a GPU spec sheet — "5 888 CUDA cores" on an RTX 4070, "16 384 cores" on an RTX 4090 — are just SMs × lanes-per-SM. What you should care about more is the SM count, because that's how many *independent* schedulers can hide memory latency.

# Part 5 — CPU vs GPU: When to Use Which

Once you understand the SIMT model, the CPU-vs-GPU question stops being "which is faster?" and becomes "which kind of work is this?" Two parameters decide:

- **Independence.** Can the work be split into many items that don't talk to each other? Pixel shading, matrix multiply, dense neural-network inference all qualify. Tree traversal, branchy game logic, single-file compression do not.
- **Volume.** Is there enough work to amortise the GPU's launch overhead? A CUDA kernel launch costs roughly **5–50 µs** end-to-end. If your job runs in less than 1 ms total on a CPU, sending it to a GPU will *slow it down*.

![CPU vs GPU schematic, plus a wall-clock crossover curve as work scales up](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig4_cpu_vs_gpu.png)

The right-hand chart shows the crossover. Below ~250 independent work items, an 8-core CPU finishes faster because launch overhead dominates the GPU's wall-clock. Above that, the GPU's parallelism wins decisively, and by the time you have tens of thousands of items the GPU is two orders of magnitude faster. Real workloads sit at very different points on this curve:

| Workload | Items per job | Best processor | Reason |
| --- | --- | --- | --- |
| Compile a single C++ file | 1 (per file) | CPU | Branchy, sequential, small |
| Render one game frame | 2 M pixels × shader ops | GPU | Massive uniform parallelism |
| SQLite point query | 1 row | CPU | Latency-bound, branchy |
| Train a neural network batch | millions of FMAs | GPU (Tensor cores) | Dense matmul |
| Sort 10 K integers | 10 K | Either, similar | Below crossover, CPU often faster |
| Encode 4K H.265 video | per-frame | GPU (fixed-function NVENC) | Dedicated hardware, not even shaders |

The honest summary: **CPUs win on small, branchy, latency-sensitive work; GPUs win on large, uniform, throughput-sensitive work; and dedicated fixed-function blocks (video encoders, display engines, NICs) win on anything narrow enough to put in silicon.** The skill of system design is putting each piece of work where it belongs.

# Part 6 — Memory Bandwidth: DDR vs GDDR vs HBM

A GPU with 18 000 ALUs is useless if you can't feed it. That is why a high-end GPU spends as much die area on memory controllers and cache as it does on compute, and why memory technology forks in two directions for CPUs and GPUs.

![DDR, GDDR and HBM bandwidths on a log scale, plus a topology comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig5_memory_bandwidth.png)

The three families optimise for different things:

- **DDR (DDR4, DDR5)** prioritises capacity, modular DIMMs and per-bit cost. CPUs use it because most CPU workloads are latency-sensitive and want lots of it: 64–192 GB is normal. Bandwidth tops out around **50–100 GB/s** for a dual-channel desktop, **300–500 GB/s** for a 12-channel server.
- **GDDR (GDDR6, GDDR6X)** is DDR's pin-compatible-but-faster cousin, soldered directly to the GPU board with very wide buses (256-bit on RTX 4070, 384-bit on RTX 4090). Per pin it runs at 16–24 Gb/s, which gives **450–1 000 GB/s** of aggregate bandwidth. The capacity ceiling is much lower (typically 12–24 GB) because you can only ring so many chips around the GPU.
- **HBM (HBM2e, HBM3, HBM3e)** stacks 8–16 DRAM dies on top of one another, connects them through silicon-through-vias, and places the whole stack right next to the GPU on a silicon interposer. Each stack exposes a **1024-bit bus**, and a GPU typically uses 4–6 stacks. Aggregate bandwidth reaches **2–5 TB/s** on H100/MI300, at the cost of much more expensive packaging and limited per-stack capacity.

The topology picture in the figure is the easiest way to remember why the bandwidth gap is so large: DDR has to drive long, lossy PCB traces, so each pin can only switch slowly. HBM signals travel a few millimetres across the interposer, so each pin can switch fast and there are 1 024 of them per stack. Bandwidth equals frequency times width, and HBM wins on both.

Practical rule: **gaming GPUs use GDDR; AI accelerators use HBM**. The crossover happens around enterprise inference cards, where models that don't fit in 24 GB of GDDR push you into HBM territory whether you want it or not.

# Part 7 — Display Interfaces: DP, HDMI, USB-C

You have three modern choices for getting pixels from the GPU to a panel. They are not interchangeable; each was designed against different constraints, and the cable you happened to grab from the drawer is often the bottleneck.

![Effective bandwidth of common display interfaces, plus a "best port for the job" matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/computer-fundamentals/04-motherboard-gpu/fig6_display_interfaces.png)

| Interface | Effective payload | Headline mode | Where it shines |
| --- | --- | --- | --- |
| HDMI 2.0 | 14.4 Gb/s | 4K@60 Hz HDR10 | TVs, consoles, projectors |
| DisplayPort 1.4 | 25.9 Gb/s | 4K@120 Hz / 8K@60 Hz w/ DSC | High-refresh PC monitors |
| HDMI 2.1 | 42.6 Gb/s | 4K@144 Hz, 8K@60 Hz, eARC, VRR | Next-gen TVs, PS5/XSX |
| USB-C / Thunderbolt 4 (DP-Alt) | up to 40 Gb/s | 4K@144 + 100 W power + USB | Single-cable laptop docks |
| DisplayPort 2.1 (UHBR 20) | 77.4 Gb/s | 4K@240 Hz uncompressed, 8K@120 Hz | 2025+ flagship monitors |

Three rules cover 95 % of real choices:

1. **TV → HDMI 2.1.** ARC/eARC sends audio back to your soundbar over the same cable, VRR eliminates console screen-tear, and ALLM auto-switches to game mode. DisplayPort cannot do these things.
2. **High-refresh PC monitor → DisplayPort.** Higher per-lane rate, native multi-stream transport (one cable, two daisy-chained monitors), and broader G-Sync/FreeSync support. Most gaming monitors ship with one HDMI 2.1 and three DP inputs precisely because of this.
3. **Laptop dock → USB-C / Thunderbolt 4.** One cable carries DP 1.4 video, USB 3 data, and up to 100 W of power. This is the only configuration where the cable's own active electronics matter — buy a certified Thunderbolt cable, not a phone-charger USB-C cable.

The single most common mistake is using an old HDMI cable with a new HDMI 2.1 port and wondering why 4K@120 Hz isn't available. The cable is the limit; the spec is **HDMI Ultra High Speed Certified**, which is printed on the packaging if it's real.

# Part 8 — Discrete vs Integrated GPU: When the Choice Actually Matters

An integrated GPU (iGPU) lives inside the CPU package and shares system DDR memory with the CPU cores. A discrete GPU (dGPU) is a separate card with its own GDDR memory on its own PCIe bus. The performance gap is set almost entirely by **memory bandwidth and shader count**, both of which the dGPU has in vast excess.

| Workload | iGPU (Intel UHD 770 / AMD 780M) | dGPU (RTX 4060) |
| --- | --- | --- |
| Office, browsing, video playback | smooth ✅ | overkill |
| 4K H.264/HEVC decode | hardware accelerated ✅ | hardware accelerated |
| LoL / CS2 / Valorant 1080p | 60–120 FPS ✅ | 200–400 FPS |
| Cyberpunk 2077 1080p high | 12–18 FPS ❌ | 75 FPS ✅ |
| Stable Diffusion XL | unusably slow | 1.5 s/iter ✅ |
| 4K editing in DaVinci Resolve | timeline lag | smooth ✅ |

The decision is binary. **If the workload has nothing to do with 3D rendering or AI, an iGPU is enough**, and you should not pay for a discrete card. **If the workload involves AAA games, ray tracing, machine learning, or 3D content creation, no iGPU will be enough** — even the best AMD 780M iGPU only matches a five-year-old GTX 1650.

## The "I plugged my monitor into the motherboard" mistake

If you have a discrete GPU installed, **the monitor cable must go into the GPU's outputs, not the motherboard's**. If you plug into the motherboard, the system silently routes rendering through the iGPU, leaving the dGPU idle. The give-away is GPU-Z's *GPU Load* sitting at 0–5 % in a game where it should be 90+ %.

The reason is that monitor outputs are physically wired to whichever GPU owns them; the motherboard's HDMI is wired to the iGPU. Some BIOSes can route iGPU output through the dGPU (Intel's "iGPU Multi-Monitor" plus DDU-style Optimus), but this is a laptop pattern that is fragile on desktops. The simple rule is: dGPU installed → cable into the dGPU.

# Part 9 — VRM, Power Connectors and the Limits of Sustained Performance

A modern K-series Intel CPU pulls up to **253 W** at PL2 boost; a Ryzen 9 7950X pulls up to **230 W** at PPT; an RTX 4090 pulls **450 W** sustained and spikes to 600 W. None of this is delivered by the 24-pin ATX connector alone. The motherboard has to convert 12 V from the PSU into the 1.0–1.4 V the CPU wants, while the GPU takes its own 12 V directly through a separate set of cables. The component that does the CPU conversion is the **VRM** (Voltage Regulator Module), and its phase count tells you how much sustained power it can deliver without overheating.

| CPU | TDP / PL2 | Recommended VRM | Typical board class |
| --- | --- | --- | --- |
| i3 / Ryzen 3 | ≤65 W | 6+2 phase | H610 / A620 |
| i5 / Ryzen 5 (non-K) | 65–125 W | 10+2 phase | B760 / B650 |
| i7-13700K / R7 7700X | 125–180 W | 14+2 phase | mid Z790 / X670 |
| i9-13900K / R9 7950X (OC) | 250–300 W | 18+2 phase, heatsinked | flagship Z790 / X670E |

Pairing a flagship CPU with an entry-level VRM produces the most insidious failure mode in PC building: it boots, it benchmarks fine for thirty seconds, and then it slowly throttles as the VRM MOSFETs cross 100 °C. Reviewers measure VRM temperature explicitly for this reason; it's the only number that distinguishes a "looks similar on paper" $130 board from a $250 board.

GPU power is simpler because it bypasses the motherboard entirely. The connector standard tells you the budget:

| GPU power | Connector | Examples |
| --- | --- | --- |
| ≤75 W | none (PCIe slot only) | GTX 1650 |
| 75–150 W | one 6-pin | RTX 3050 |
| 150–225 W | one 8-pin | RTX 4060 Ti |
| 225–300 W | two 8-pin | RTX 4070 Ti |
| ≥300 W | one 12VHPWR (16-pin) | RTX 4080 / 4090 |

Forgetting to plug in a required GPU power cable produces either a no-boot beep code or a boot with the GPU artificially capped to 75 W — slow enough that some games crash on startup. Always count the connectors before powering on.

# Part 10 — BIOS Settings That Actually Matter

The BIOS exposes hundreds of toggles; four of them produce all the user-visible performance gain.

- **XMP / EXPO (Intel / AMD).** RAM is sold at, say, "DDR5-6000" but defaults to its JEDEC speed (DDR5-4800) until you enable the on-DIMM overclock profile. Turning on XMP/EXPO is free 5–15 % in CPU-bound games. If the system fails to POST, clear CMOS and try the next-lower profile or manually drop the frequency by one step.
- **Resizable BAR / Smart Access Memory.** Lets the CPU map the GPU's full VRAM into address space at once instead of in 256 MB windows. Worth 2–8 % FPS in games with large texture working sets (Cyberpunk 2077, Forza). Requires "Above 4G Decoding" + "Re-Size BAR" both enabled, and a supported GPU (RTX 30/40, RX 6000/7000).
- **CPU power limits (PL1/PL2 on Intel, PPT/TDC/EDC on AMD).** Default-locked on most boards, which means the CPU only sustains its boost for 28–56 seconds before dropping to base clock. Removing the limit (or setting both PL1 and PL2 to the same value) lets the CPU boost indefinitely as long as cooling holds — usually worth 10–20 % in long compiles and renders.
- **Fan curves.** The default curve on most boards is silence-first, which lets temps drift to throttling. A curve that hits 80 % fan speed at 75 °C costs a few dB of noise and earns you sustained boost clocks instead of thermal throttling.

Other BIOS settings (Secure Boot, virtualisation, SATA mode) are correctness-critical but performance-neutral. Enable virtualisation if you'll touch Docker, WSL2, VMware or Android Studio; otherwise leave the defaults alone.

# Summary

The motherboard is two domains glued by one link: a small CPU-direct domain that runs the GPU, primary NVMe and RAM at full speed, and a chipset domain that fans out everything else through a shared DMI bottleneck. Reading the PCB tells you which slot is which.

PCIe per-lane bandwidth doubles every generation, so a Gen 4 ×4 NVMe is as fast as a Gen 3 ×8. Consumer GPUs have not yet outgrown PCIe 4.0 ×16; Gen 5 first matters for SSDs.

GPUs win by parallelism: the SIMT model runs 32 lanes of one instruction per warp, and SMs hide memory latency by switching warps every cycle. CPUs win on small, branchy, latency-bound work; GPUs win on large, uniform, throughput-bound work. Choose by workload, not by reflex.

Memory technology forks: DDR for CPUs (capacity, latency), GDDR for gaming GPUs (bandwidth, modest capacity), HBM for AI accelerators (bandwidth, on-package). Display interfaces fork too: HDMI 2.1 for TVs and consoles, DisplayPort for high-refresh PC monitors, USB-C/Thunderbolt for laptop docks.

If you have a discrete GPU, the monitor cable goes into the GPU. The VRM phase count must match the CPU's sustained TDP or the system throttles silently. XMP/EXPO and Resizable BAR are free performance; everything else in the BIOS is a tuning exercise, not a magic bullet.

**Next: Part 5 — Network, Power & Practical Troubleshooting.**
