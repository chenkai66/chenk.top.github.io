---
title: "Virtualization Technology Deep Dive"
date: 2023-02-20 09:00:00
tags:
  - Cloud Computing
  - Virtualization
  - KVM
  - VMware
  - Docker
categories: Cloud Computing
series:
  name: "Cloud Computing"
  part: 2
  total: 8
lang: en
mathjax: false
description: "A hands-on guide to virtualization -- hypervisors (VMware, KVM, Xen, Hyper-V), storage and network virtualization, performance tuning, and container comparison."
disableNunjucks: true
series_order: 2
---
Without virtualization, there is no cloud. Every EC2 instance, every Lambda invocation, every Kubernetes pod ultimately stands on the same trick: lying convincingly to an operating system about the hardware underneath it. This article walks the full stack -- from the CPU instructions that make the trick cheap, through the four hypervisors that dominate the market, to the production-grade tuning knobs that decide whether your VMs run at 70 % or 99 % of bare metal.

## What You Will Learn

- How CPU virtualization actually works (rings, VT-x, EPT) and why Type 1 and Type 2 hypervisors exist
- Hands-on configuration of VMware ESXi, KVM, Xen, and Hyper-V with the right defaults for production
- Storage virtualization with LVM and ZFS, and why disk format choice can change IOPS by 4x
- Network virtualization: VLANs, VXLAN overlays, Open vSwitch and SR-IOV
- Performance tuning: CPU pinning, NUMA, huge pages, virtio, vhost-net
- Live migration internals, nested virtualization, and GPU sharing (vGPU / MIG)
- Security hardening, isolation primitives, and a troubleshooting playbook

## Prerequisites

- Comfortable on a Linux command line
- Basic OS concepts (processes, kernel vs userspace, page tables)
- Part 1 of this series ([Cloud Computing Fundamentals](/en/cloud-computing-fundamentals/))

---

## 1. Virtualization Fundamentals

Virtualization creates virtual versions of hardware resources -- CPU, memory, disks, NICs -- so that multiple operating systems can each believe they own a whole machine. The component that maintains the illusion is the **hypervisor**, also called the Virtual Machine Monitor (VMM).

### 1.1 Why It Took Hardware Help

x86 was not originally virtualizable. The architecture exposes 17 sensitive instructions (e.g. `POPF`, `SGDT`) that change global state but do **not** trap when executed in user mode -- which means a naive hypervisor cannot intercept them. Two workarounds emerged in the early 2000s:

- **Binary translation** (VMware): rewrite guest kernel code on the fly to replace dangerous instructions with calls into the hypervisor. Clever but slow and complex.
- **Para-virtualization** (Xen): modify the guest OS to call the hypervisor directly via "hypercalls". Fast but only works for cooperating guests (Linux, BSD).

In 2005-2006 Intel VT-x and AMD-V added a new CPU mode -- **VMX root** for the hypervisor and **VMX non-root** for the guest -- with hardware-managed transitions (`VMENTER` / `VMEXIT`). Suddenly any unmodified OS could run at near-native speed. A few years later, Extended Page Tables (EPT) and Nested Page Tables (NPT) eliminated the second-biggest cost -- shadow page table maintenance -- by giving the MMU two-level translation in hardware.

This is the moment virtualization became cheap enough to build a public cloud on.

### 1.2 Type 1 vs Type 2 Hypervisors

![Type 1 (Bare-Metal) vs Type 2 (Hosted) Hypervisor Architectures](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig1_hypervisor_types.png)

| | Type 1 (bare-metal) | Type 2 (hosted) |
|---|---|---|
| Runs on | Hardware directly | A host OS |
| Examples | VMware ESXi, KVM, Xen, Hyper-V | VirtualBox, VMware Workstation, Parallels |
| Overhead | Minimal | Adds host-OS scheduling layer |
| Use case | Production clouds, data centers | Dev laptops, training labs |

KVM is a slightly weird case: it is a kernel module that turns Linux *into* a Type 1 hypervisor -- the host kernel and the hypervisor are the same kernel.

### 1.3 Key Concepts

- **Hypervisor (VMM):** the layer that schedules vCPUs onto pCPUs, allocates memory, and traps privileged guest operations.
- **Guest OS:** the OS running inside a VM, unaware (or barely aware) it is virtualized.
- **vCPU:** a thread on the host scheduler that runs guest code in VMX non-root mode.
- **Resource overcommit:** allocating more virtual resources than physically exist. Safe within limits because guests rarely peak together. Typical safe ratios: CPU 4:1-8:1, memory 1.5:1-2:1.
- **Ballooning:** a guest driver that returns idle memory to the host on demand, enabling memory overcommit.

### 1.4 Historical Milestones

| Year | Event |
|------|-------|
| 1972 | IBM VM/370 -- first commercial OS-level virtualization |
| 1999 | VMware Workstation ships, virtualizing x86 via binary translation |
| 2003 | Xen 1.0 introduces para-virtualization on Linux |
| 2005 | Intel VT-x and AMD-V bring hardware-assisted virtualization to x86 |
| 2006 | KVM merged into Linux 2.6.20 |
| 2008 | EPT / NPT eliminate shadow page table cost |
| 2013 | Docker launches; containers become the second wave |
| 2018 | Firecracker (microVMs) enables sub-second VM boot for serverless |

## 2. Types of Virtualization

| Property | Full virt (BT) | Para-virt | HW-assisted | Containers |
|----------|---------------|-----------|-------------|------------|
| Guest OS modified? | No | Yes | No | N/A |
| CPU overhead | High | Medium | ~1-3 % | Near zero |
| Isolation | Strong | Strong | Strong | Process-level |
| Boot time | 30-60 s | 30-60 s | 20-45 s | < 1 s |
| Image size | GB | GB | GB | tens of MB |
| Example | Early VMware | Xen PV | KVM, ESXi 6+ | Docker, containerd |

### 2.1 Containers vs VMs: a Different Isolation Boundary

![VM vs container resource isolation: each VM ships its own kernel, containers share the host kernel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig2_vm_vs_container.png)

The single most important diagram in this article. A VM virtualizes the **hardware** -- so each guest brings its own kernel. A container virtualizes the **OS** -- it is just a Linux process bundled with namespaces (PID, mount, network, UTS, IPC, user) and cgroups for resource limits. Same kernel for everyone.

Consequences:

- A container that triggers a kernel bug can compromise the host. A VM that triggers a kernel bug only crashes itself.
- Containers cannot run a different OS (no Windows containers on a Linux host kernel, despite the marketing).
- Containers boot in milliseconds because there is no kernel to initialize.

This is why production workloads often run **containers inside VMs**: the VM gives you a security boundary, the container gives you density and speed.

### 2.2 Startup Latency and Memory Cost

![Cold-start latency and idle memory footprint: containers vs MicroVMs vs VMs (log scale)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig3_startup_and_memory.png)

The numbers above are why "serverless" works on Firecracker microVMs (~125 ms cold start, ~30 MB overhead) and not on traditional KVM/QEMU VMs. Two orders of magnitude in startup time and memory completely change the economics of bursty workloads.

## 3. Hypervisor Choice

![KVM vs Xen vs VMware ESXi vs Hyper-V across six dimensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig4_hypervisor_matrix.png)

There is no single best hypervisor -- the right choice depends on what you already run, who runs it, and what you can spend on licensing.

### 3.1 KVM (Kernel-based Virtual Machine)

KVM turns Linux itself into a Type 1 hypervisor. It is open-source, ships in every major distro, and powers OpenStack, Proxmox, Amazon EC2 (Nitro), Google Cloud, and most of Alibaba Cloud's ECS fleet.

**Install on Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients \
                    bridge-utils virt-manager

sudo usermod -aG libvirt,kvm $USER

# Verify hardware support and that the kernel module is loaded
egrep -c '(vmx|svm)' /proc/cpuinfo   # > 0 means VT-x/AMD-V available
lsmod | grep kvm
virsh list --all
```

**Create a VM:**

```bash
virt-install \
  --name ubuntu-server \
  --ram 4096 \
  --vcpus 4 \
  --disk path=/var/lib/libvirt/images/ubuntu-server.qcow2,size=50,bus=virtio,cache=none,io=native \
  --os-variant ubuntu22.04 \
  --network bridge=virbr0,model=virtio \
  --graphics none \
  --console pty,target_type=serial \
  --location 'http://archive.ubuntu.com/ubuntu/dists/jammy/main/installer-amd64/' \
  --extra-args 'console=ttyS0,115200n8 serial'
```

**Common `virsh` commands:**

```bash
virsh list --all                # list VMs
virsh start <vm>                # start
virsh shutdown <vm>             # graceful shutdown
virsh destroy <vm>              # force stop (kill -9 equivalent)
virsh suspend <vm>              # freeze in memory
virsh snapshot-create-as <vm> snap1
virsh domstats <vm>             # live statistics
```

### 3.2 VMware ESXi

ESXi is a Type 1 hypervisor that runs directly on the server. It is the de facto enterprise standard and pairs with vCenter for cluster management.

The default VM hardware uses emulated SCSI (LSI Logic) and an emulated 1 Gbps NIC (E1000). This is fine for a Windows installer but cripples production workloads. Switch to paravirtual:

```ini
# Optimised .vmx fragment
virtualHW.version = "19"
numvcpus = "4"
cpuid.coresPerSocket = "2"
memory = "8192"

# PVSCSI: 3-4x the IOPS of LSI Logic
scsi0.virtualDev   = "pvscsi"
scsi0:0.fileName   = "Ubuntu-Server-22.04.vmdk"

# VMXNET3: ~9.5 Gbps vs E1000's 1 Gbps
ethernet0.virtualDev = "vmxnet3"
```

| Setting | Default | Optimised | Win |
|---------|---------|-----------|-----|
| SCSI controller | LSI Logic (~80 K IOPS) | PVSCSI (~300 K+ IOPS) | 3-4x IOPS |
| NIC | E1000 (~1 Gbps) | VMXNET3 (~9.5 Gbps) | ~10x throughput |
| CPU overhead | Higher (emulation) | Lower (paravirtual) | 30-50 % less CPU |

Both PVSCSI and VMXNET3 require the VMware Tools driver in the guest. Install with the legacy controllers, switch after first boot.

```bash
# Verify in the guest after switch-over
lsmod | grep vmw_pvscsi
ethtool -i eth0    # should show "vmxnet3"
```

### 3.3 Xen

Xen is the original open-source Type 1 hypervisor and powered the first generation of AWS EC2. It supports both PV and HVM (hardware-assisted) modes. Today it is most common in security-focused stacks (Qubes OS) and a handful of legacy clouds.

```bash
sudo apt install -y xen-hypervisor-amd64 xen-tools
sudo update-grub && sudo reboot

sudo xen-create-image \
  --hostname=debian-pv --memory=1024mb --vcpus=2 \
  --lvm=vg0 --size=20gb --dist=bookworm

sudo xl create /etc/xen/debian-pv.cfg
sudo xl list
```

### 3.4 Microsoft Hyper-V

Hyper-V is bundled with Windows Server and the Windows desktop SKUs. The right choice when most of your guests are Windows or your team lives in PowerShell.

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All

New-VM -Name "WindowsServer2022" `
  -MemoryStartupBytes 4GB -Generation 2 `
  -NewVHDPath "C:\VMs\WindowsServer2022.vhdx" `
  -NewVHDSizeBytes 100GB -SwitchName "Default Switch"

Set-VMProcessor -VMName "WindowsServer2022" -Count 4
Set-VMMemory    -VMName "WindowsServer2022" `
                -DynamicMemoryEnabled $true `
                -MinimumBytes 2GB -MaximumBytes 8GB
Start-VM -Name  "WindowsServer2022"
```

### 3.5 Choosing

| Criterion | KVM | VMware ESXi | Hyper-V | Xen |
|-----------|-----|-------------|---------|-----|
| Cost | Free | $$$ per socket | Bundled with WS | Free |
| Management | virsh, OpenStack | vSphere (excellent GUI) | PowerShell, Windows Admin Center | xl, XenCenter |
| Performance | Excellent | Excellent | Very good | Excellent |
| Best fit | Linux clouds, OpenStack | Enterprise data centers | Microsoft shops | Security niches, legacy |

## 4. Storage Virtualization

### 4.1 LVM (Logical Volume Manager)

LVM abstracts block devices into flexible, resizable logical volumes. The mental model: physical volumes (`pv`) -> volume groups (`vg`) -> logical volumes (`lv`).

```bash
sudo pvcreate /dev/sdb1 /dev/sdc1
sudo vgcreate vg_storage /dev/sdb1 /dev/sdc1
sudo lvcreate -L 100G -n lv_data vg_storage
sudo mkfs.ext4 /dev/vg_storage/lv_data
sudo mount /dev/vg_storage/lv_data /mnt/data

# Online expand (no unmount, no downtime)
sudo lvextend -L +50G /dev/vg_storage/lv_data
sudo resize2fs   /dev/vg_storage/lv_data

# Copy-on-write snapshot
sudo lvcreate -L 10G -s -n lv_data_snap /dev/vg_storage/lv_data
```

### 4.2 ZFS

ZFS combines volume management and filesystem with built-in checksums, compression, snapshots, and send/receive replication. The cost is RAM (rule of thumb: 1 GB per TB for the ARC).

```bash
sudo zpool create tank raidz /dev/sdb /dev/sdc /dev/sdd

sudo zfs create tank/data
sudo zfs set compression=lz4 tank/data        # ~30 % space savings, near-free CPU
sudo zfs set atime=off       tank/data        # avoid write amplification

sudo zfs snapshot tank/data@2025-01-01
sudo zfs send tank/data@2025-01-01 | ssh backup-host \
     sudo zfs receive backup-pool/data
```

### 4.3 Disk Format and I/O Path

The format you pick for the virtual disk image -- and the cache mode you give QEMU -- often matters more than which CPU you bought.

```xml
<!-- libvirt: production-grade disk for KVM -->
<disk type='file' device='disk'>
  <driver name='qemu' type='raw' cache='none' io='native' discard='unmap'/>
  <source file='/var/lib/libvirt/images/db.raw'/>
  <target dev='vda' bus='virtio'/>
</disk>
```

```bash
# Use a low-overhead I/O scheduler under the host
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
```

| Format | Performance | Features | Best for |
|--------|------------|----------|----------|
| Raw | Best | None | Production DBs, latency-critical |
| QCOW2 | Very good | Snapshots, compression, thin | General use, dev/test |
| VMDK | Good | VMware ecosystem | VMware shops |

`cache=none` + `io=native` bypasses the host page cache and uses asynchronous direct I/O -- the right default for any guest with its own filesystem cache (i.e. all of them). `cache=writeback` is faster but loses data on host crash; only use it for ephemeral workloads.

## 5. Network Virtualization

### 5.1 VLANs

A VLAN tag (802.1Q) carves one physical network into many isolated broadcast domains.

```bash
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100
sudo ip link set eth0.100 up
```

Limit: 4 094 VLAN IDs is plenty for a single rack, nowhere near enough for a public cloud.

### 5.2 VXLAN

VXLAN tunnels Layer-2 Ethernet frames inside UDP packets, giving you 16 million logical networks (24-bit VNI) over any IP fabric. This is the workhorse of multi-tenant cloud networks and Kubernetes overlays (Flannel, Calico VXLAN mode).

```text
VM A (10.1.1.10) -- VTEP1 [VNI 100] --IP fabric-- VTEP2 [VNI 100] -- VM B (10.1.1.20)
```

```bash
sudo ip link add vxlan100 type vxlan id 100 dstport 4789 \
     group 239.1.1.1 dev eth0
sudo ip addr add 10.1.1.1/24 dev vxlan100
sudo ip link set vxlan100 up
```

### 5.3 Open vSwitch

OVS is a programmable virtual switch with OpenFlow support, used by OpenStack Neutron, Open Virtual Network (OVN), and many SDN stacks.

```bash
sudo apt install -y openvswitch-switch
sudo ovs-vsctl add-br br0
sudo ovs-vsctl add-port br0 eth0
sudo ovs-vsctl add-port br0 vnet0

sudo ovs-ofctl add-flow br0 "in_port=1,actions=output:2"
sudo ovs-vsctl show
sudo ovs-ofctl dump-flows br0
```

### 5.4 SR-IOV: Bypass the Hypervisor

SR-IOV (Single Root I/O Virtualization) lets a NIC expose Virtual Functions (VFs) that are mapped directly into VMs, bypassing the host's network stack entirely. Latency drops from ~30 µs (virtio) to ~3 µs (SR-IOV); throughput becomes line-rate.

```bash
echo 4 | sudo tee /sys/class/net/eth0/device/sriov_numvfs
ip link show eth0
# eth0: ... vf 0 MAC ...  vf 1 MAC ...  vf 2 MAC ...  vf 3 MAC ...
```

The trade-off: live migration becomes harder (the VM is bound to a specific physical NIC) and you can only have as many VMs on a NIC as it has VFs.

## 6. Performance Optimization

### 6.1 CPU: Pinning, NUMA, Topology

On any multi-socket host, the worst case is a vCPU that wakes up on socket A but its memory lives on socket B -- every cache line is a cross-socket round trip. Fix this with **CPU pinning** plus **NUMA pinning**:

```bash
# Pin each vCPU to a specific physical core
virsh vcpupin ubuntu-server 0 0
virsh vcpupin ubuntu-server 1 1
virsh vcpupin ubuntu-server 2 2
virsh vcpupin ubuntu-server 3 3

# Force memory allocation onto the same NUMA node
virsh numatune ubuntu-server --nodeset 0 --mode strict
```

Expose the real CPU topology so the guest scheduler can make good decisions:

```xml
<cpu mode='host-passthrough' check='none'>
  <topology sockets='1' cores='4' threads='2'/>
  <cache mode='passthrough'/>
</cpu>
```

`mode='host-passthrough'` exposes every CPU flag (AVX-512, AES-NI, etc.) but breaks live migration to dissimilar hosts; use `host-model` if you need migration across a heterogeneous fleet.

### 6.2 Memory: Huge Pages and Ballooning

Each TLB entry maps one page. With 4 KB pages, walking 1 GB of RAM hits 262 144 entries; with 2 MB huge pages, just 512. For databases and JVMs the speedup is 5-15 %.

```bash
# Allocate 2 048 huge pages * 2 MB = 4 GB
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages

# Tell libvirt to back the VM with huge pages
# <memoryBacking><hugepages/></memoryBacking>
```

Ballooning gives memory back dynamically:

```bash
virsh setmaxmem ubuntu-server 8G --config
virsh setmem    ubuntu-server 2G --live   # shrink without reboot
```

### 6.3 I/O: virtio Everywhere

Always use virtio devices in KVM guests. They are paravirtual -- the guest knows it is in a VM and uses ring buffers shared with the hypervisor instead of MMIO emulation. `vhost-net` moves the network ring processing into the kernel, eliminating one userspace round trip per packet.

```xml
<interface type='bridge'>
  <model type='virtio'/>
  <driver name='vhost' queues='4'/>   <!-- multi-queue scales with vCPUs -->
</interface>
```

### 6.4 Tuning Checklist

- [ ] VT-x / AMD-V enabled in BIOS
- [ ] virtio (KVM) or PVSCSI/VMXNET3 (VMware) for disk and NIC
- [ ] CPU pinning + NUMA pinning for any VM with > 8 vCPUs
- [ ] Host I/O scheduler = `none` (NVMe) or `mq-deadline` (SATA)
- [ ] Disk format = raw, `cache=none`, `io=native` for production
- [ ] Huge pages backing memory-intensive guests
- [ ] SR-IOV for network-intensive VMs that do not need live migration
- [ ] Multi-queue virtio-net with queue count = vCPU count
- [ ] CPU governor = `performance` on the host

## 7. Live Migration: Moving a Running VM

![Pre-copy live migration: full image, then iterative dirty-page rounds, then sub-100ms cutover](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig5_live_migration.png)

Live migration is the single feature that turned VMs from "another way to run a server" into a fluid pool. The dominant algorithm is **pre-copy**:

1. **Initial copy.** Send the entire guest memory image to the destination while the VM keeps running on the source.
2. **Dirty-page rounds.** The source tracks pages dirtied during step 1 and re-sends them. Repeat until the dirty rate falls below network bandwidth, or you hit the round limit.
3. **Stop & switch.** Pause the source VM, send the last dirty pages and CPU state, resume on the destination. Pause time is typically < 100 ms -- TCP connections and user sessions survive.

```bash
# KVM live migration over TLS
virsh migrate --live --persistent --undefinesource \
              --copy-storage-all \
              ubuntu-server qemu+tls://dest-host/system
```

If the guest is dirtying memory faster than the link can move it (a busy in-memory database, for example), pre-copy never converges. The fallbacks are **post-copy** (switch first, page-fault memory across the network -- lower total downtime, higher tail latency) and **auto-converge** (throttle the guest CPU until the dirty rate drops).

Requirements:

- Shared storage, **or** `--copy-storage-all` (slower)
- Compatible CPU features on source and destination (use `host-model`, not `host-passthrough`)
- A network fast enough that dirty rate < bandwidth -- 10 Gbps is the practical minimum for production VMs

## 8. Nested Virtualization

![Nested virtualization stack: L0 hypervisor, L1 guest hypervisor, L2 guest VM, with throughput cost per level](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig6_nested_virtualization.png)

Nested virtualization lets a guest VM itself be a hypervisor. Real uses:

- Running a hypervisor lab inside a cloud VM
- CI pipelines that spin up VMs as part of their build
- WSL 2 on Windows (Hyper-V running Linux running containers)
- Confidential computing (one hypervisor inside another for defence in depth)

Enable on KVM:

```bash
# Intel
echo "options kvm-intel nested=1" | sudo tee /etc/modprobe.d/kvm.conf
sudo modprobe -r kvm_intel && sudo modprobe kvm_intel
cat /sys/module/kvm_intel/parameters/nested        # should print Y

# AMD
echo "options kvm-amd nested=1" | sudo tee /etc/modprobe.d/kvm.conf
```

Then expose `vmx` (or `svm`) to the L1 guest:

```xml
<cpu mode='host-passthrough'/>   <!-- simplest way to leak the flag -->
```

Cost: each nesting level adds VM-exit hops. CPU-bound workloads typically lose 5-10 % at L1 and 25-40 % at L2; I/O-bound workloads can lose much more without paravirt drivers all the way down.

## 9. GPU Virtualization

![Four ways to share a GPU: time-slicing, vGPU, MIG, and full passthrough](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/cloud-computing/virtualization/fig7_gpu_virtualization.png)

GPUs were designed for one process to own the whole device. Sharing them across VMs requires picking a point on the density-vs-isolation spectrum:

| Mode | How | Isolation | Density | Notes |
|------|-----|-----------|---------|-------|
| **Time-slicing** | Hypervisor round-robins context | Weak (no perf isolation) | High | Cheapest, used in many ML training clusters |
| **vGPU** (NVIDIA GRID) | Mediated passthrough, virtual GPU per VM | Medium | Medium | Hard QoS for memory; requires NVIDIA license |
| **MIG** (A100 / H100) | Hardware partitions GPU into up to 7 instances | Strong (separate SMs, L2, memory) | Fixed shapes (1g.5gb, 2g.10gb, ...) | True isolation, no fractional sharing within a slice |
| **Passthrough** (VFIO) | One whole GPU bound to one VM | Native | 1 VM per GPU | Bare-metal speed, no sharing at all |

A typical AI cloud uses MIG for inference (many small slices, hard QoS) and passthrough for training (one job, the whole device).

```bash
# VFIO passthrough on Linux
echo "vfio-pci" | sudo tee /etc/modules-load.d/vfio.conf
echo "options vfio-pci ids=10de:2204" | sudo tee /etc/modprobe.d/vfio.conf
# Then in libvirt XML:
# <hostdev mode='subsystem' type='pci' managed='yes'>
#   <source><address bus='0x01' slot='0x00' function='0x0'/></source>
# </hostdev>
```

## 10. Security and Isolation

### 10.1 Threat Model

The fundamental promise of a hypervisor is that one tenant cannot read or affect another. Real-world breaks have come from:

- **VM escape** via emulated device bugs (Venom, 2015 -- floppy drive emulation in QEMU)
- **Side channels** on shared CPU caches (L1TF, MDS, Spectre v2)
- **Hypervisor kernel bugs** in the management plane

Defenses are layered, not perfect.

### 10.2 Hypervisor Hardening

```bash
# Smallest possible attack surface
sudo systemctl disable --now avahi-daemon cups bluetooth

# Firewall: management plane only, restricted source
sudo ufw enable
sudo ufw allow from 192.168.1.0/24 to any port 22
sudo ufw allow from 192.168.1.0/24 to any port 16509   # libvirt

# libvirt over TLS only
# /etc/libvirt/libvirtd.conf
# listen_tls = 1
# listen_tcp = 0
# auth_tls   = "sasl"
```

Enable CPU side-channel mitigations on the host (`spectre_v2=on`, `l1tf=full`) and keep microcode current.

### 10.3 Guest Best Practices

- Minimal install (no GUI, no `cups`, no `apt-listchanges`)
- Automatic security updates (`unattended-upgrades`)
- Full-disk encryption for sensitive data (LUKS in the guest)
- Per-trust-tier VLANs -- never put PCI workloads on the same broadcast domain as developer VMs
- Keep guest agents (`qemu-guest-agent`, `vmtoolsd`, `Hyper-V Integration Services`) up to date

### 10.4 Confidential Computing

The newest layer: AMD SEV-SNP, Intel TDX, and ARM CCA encrypt guest memory with a key the hypervisor cannot see. Even a fully compromised host cannot read tenant data. Available today on Azure Confidential VMs, GCP Confidential VMs, and Alibaba Cloud ECS gN8v.

## 11. Troubleshooting Playbook

| Symptom | First check | Likely fix |
|---------|------------|-----------|
| VM won't start | `journalctl -u libvirtd`, `qemu-img check vm.qcow2` | Disk corruption, permissions, SELinux/AppArmor labels |
| Slow CPU | `virt-top`, `mpstat -P ALL` on host | CPU pinning, governor = performance, check %steal |
| Slow disk | `iostat -xz 1`, guest `iotop` | Switch to virtio + `cache=none io=native`, raw format |
| Network packet loss | `ethtool -S`, `tc -s qdisc` | Multi-queue virtio, increase ring buffer, check vhost CPU |
| Live migration stalls | `virsh domjobinfo <vm>` | Dirty rate too high -- enable auto-converge or use post-copy |
| Memory pressure | `free -h`, `cat /proc/meminfo`, `numastat` | Add huge pages, fix NUMA pinning, reduce overcommit |
| Random VM crash | `dmesg`, `/var/log/libvirt/qemu/<vm>.log` | EDAC errors -> bad RAM; otherwise check kernel + microcode |
| `KVM: entry failed, hardware error 0x80000021` | `dmesg | grep KVM` | Disable nested or update microcode; check VT-x state |

## 12. Resource Sizing Guidelines

| Resource | Light | Medium | Heavy |
|----------|-------|--------|-------|
| vCPUs | 1-2 | 2-4 | 4-8+ |
| Memory | 2 GB | 4-8 GB | 16+ GB |
| OS disk | 20 GB | 30-50 GB | 50-100 GB |
| Network | 1 Gbps | 1-10 Gbps | 10+ Gbps (SR-IOV) |

**Safe overcommit ratios:**

- CPU: 4:1 to 8:1 — drop to 2:1 or 1:1 for latency-sensitive workloads
- Memory: 1.5:1 to 2:1 — never overcommit memory for databases
- Disk: thin provisioning OK if you monitor capacity

## 13. Case Studies

### 13.1 Enterprise Data Center Consolidation

A financial services company consolidated 200 physical servers onto 20 VMware ESXi hosts backed by a shared NVMe SAN. Result: 90 % rack reduction, 60 % cost saving, deployment time from weeks to hours. The bigger win was operational -- a single vCenter cluster replaced four ticket queues.

### 13.2 HPC Research Cluster

A research institute ran tightly-coupled MPI jobs on KVM with `host-passthrough` CPU, NUMA pinning, 1 GB huge pages, and SR-IOV InfiniBand. Sustained 95-98 % of bare-metal throughput while gaining the ability to snapshot whole experiments and ship them between sites.

### 13.3 Public Cloud Compute Plane

Hyperscale clouds (AWS Nitro, Alibaba ECS Shenlong) push device emulation off the CPU entirely onto custom DPUs, leaving 100 % of the host CPU for guests and reducing the host's attack surface to a thin KVM. Performance overhead in the single digits, security boundary smaller than any traditional ESXi host.

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Architecture](/en/cloud-computing-fundamentals/) |
| **2** | **Virtualization Technology Deep Dive (you are here)** |
| 3 | [Storage Systems and Distributed Architecture](/en/cloud-computing-storage-systems/) |
| 4 | [Network Architecture and SDN](/en/cloud-computing-networking-sdn/) |
| 5 | [Security and Privacy Protection](/en/cloud-computing-security-privacy/) |
| 6 | [Operations and DevOps Practices](/en/cloud-computing-operations-devops/) |
| 7 | [Cloud-Native and Container Technologies](/en/cloud-computing-cloud-native-containers/) |
| 8 | [Multi-Cloud and Hybrid Architecture](/en/cloud-computing-multi-cloud-hybrid/) |
