---
title: "云计算（二）：虚拟化技术深度解析"
date: 2023-02-20 09:00:00
tags:
  - Cloud Computing
  - Virtualization
  - KVM
  - VMware
  - Docker
categories: 云计算
series: cloud-computing
lang: zh
mathjax: false
description: "虚拟化实战指南 -- Hypervisor（VMware、KVM、Xen、Hyper-V）、存储与网络虚拟化、性能调优以及容器对比。"
disableNunjucks: true
series_order: 2
translationKey: "cloud-computing-2"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/illustration_1.png)

没有虚拟化就没有云计算。每一个 EC2 实例、每一次 Lambda 调用、每一个 Kubernetes Pod，本质上都依赖同一个把戏：**让操作系统对底层硬件深信不疑地撒谎**。本文将从 CPU 指令层（使这个把戏变得廉价的硬件支持）讲到主流四大 Hypervisor，再到生产级调优——决定你的虚拟机性能是达到裸机的 70% 还是 99%。

## 你将学到的内容
- 深入理解 CPU 虚拟化的运行机制（保护环、VT-x、EPT）及 Type 1 和 Type 2 Hypervisor 的各自意义
- 实战演练：如何正确配置 VMware ESXi、KVM、Xen 和 Hyper-V，并设置适合生产环境的默认参数
- 存储虚拟化技术解析：LVM 和 ZFS 的应用，以及磁盘格式选择对 IOPS 性能的影响
- 网络虚拟化全貌：VLAN 划分、VXLAN 隧道封装、Open vSwitch 的使用及 SR-IOV 的优势
- 性能优化技巧：CPU 绑核、NUMA 亲和性、大页内存、virtio 驱动与 vhost-net 的实践
- 深入探讨在线热迁移的实现原理、嵌套虚拟化的应用场景，以及 GPU 共享技术（vGPU / MIG）的使用方法
- 安全加固策略、隔离机制的核心概念，以及一份实用的故障排查指南
## 准备知识

- 能够熟练使用 Linux 命令行
- 掌握操作系统的基本概念（如进程、内核与用户空间的区别、页表机制）
- 最好先阅读本系列的第一篇文章（[云计算基础](/zh/cloud-computing/fundamentals/)）

---
## 1. 虚拟化基础

虚拟化技术通过为硬件资源（如 CPU、内存、磁盘和网卡）创建虚拟版本，让多个操作系统能够各自“以为”自己独占整台机器。而实现这一假象的核心组件，就是 **Hypervisor**，也被称为虚拟机监视器（VMM）。

### 1.1 为什么需要硬件支持？

早期的 x86 架构并不是为虚拟化设计的。它包含 17 条敏感指令（例如 `POPF` 和 `SGDT`），这些指令会修改全局状态，但在用户模式下执行时却**不会触发异常**——这意味着一个简单的 Hypervisor 根本无法拦截它们。为了解决这个问题，2000 年代初期出现了两种主要方法：

- **二进制翻译**（VMware 的做法）：在运行时动态改写 Guest 内核代码，将危险指令替换为对 Hypervisor 的调用。这种方法虽然巧妙，但性能开销大，实现复杂。
- **半虚拟化**（Xen 的方案）：直接修改 Guest 操作系统，让它通过“hypercall”主动与 Hypervisor 通信。这种方式效率高，但仅适用于愿意配合的操作系统（如 Linux 和 BSD）。

直到 2005-2006 年，Intel VT-x 和 AMD-V 技术引入了一种全新的 CPU 模式：Hypervisor 运行在 **VMX root** 模式，而 Guest 则运行在 **VMX non-root** 模式。硬件负责管理两者的切换（通过 `VMENTER` 和 `VMEXIT` 指令）。这使得任何未经修改的操作系统都能以接近原生的速度运行。几年后，**扩展页表（EPT）** 和 **嵌套页表（NPT）** 的出现进一步解决了第二大性能瓶颈——影子页表的维护问题，将两级地址翻译完全交由硬件完成。

正是这些技术突破，让虚拟化的成本大幅降低，从而为公有云的构建奠定了基础。

### 1.2 Type 1 和 Type 2 Hypervisor 的对比

![Type 1（裸金属型）与 Type 2（宿主型）Hypervisor 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig1_hypervisor_types.png)

| | Type 1（裸金属型） | Type 2（宿主型） |
|---|---|---|
| 运行环境 | 直接运行在硬件上 | 运行在宿主操作系统之上 |
| 典型代表 | VMware ESXi、KVM、Xen、Hyper-V | VirtualBox、VMware Workstation、Parallels |
| 性能开销 | 极低 | 额外增加宿主 OS 的调度层 |
| 使用场景 | 生产环境、数据中心 | 开发测试、个人实验 |

KVM 是一个特例：它是一个内核模块，能够将 Linux 本身转变为 Type 1 Hypervisor——也就是说，宿主内核和 Hypervisor 实际上是同一个内核。

### 1.3 核心概念

- **Hypervisor（VMM）**：负责将虚拟 CPU（vCPU）调度到物理 CPU（pCPU）、分配内存，并拦截 Guest 的特权操作。
- **客户机操作系统（Guest OS）**：运行在虚拟机中的操作系统，通常对虚拟化过程无感知或仅有少量感知。
- **vCPU**：宿主机调度器中的一个线程，用于在 VMX non-root 模式下执行 Guest 的代码。
- **资源超分（Resource Overcommit）**：分配的虚拟资源总量超过实际物理资源。在合理范围内是安全的，因为 Guest 很少同时达到峰值负载。常见安全比例：CPU 4:1~8:1，内存 1.5:1~2:1。
- **气球驱动（Ballooning）**：一种运行在 Guest 内部的驱动程序，可以按需回收空闲内存并返还给宿主机，从而支持内存超分。

### 1.4 发展历程中的重要节点

| 年份 | 事件 |
|------|------|
| 1972 | IBM VM/370——首个商用 OS 级虚拟化技术诞生 |
| 1999 | VMware Workstation 发布，通过二进制翻译实现 x86 虚拟化 |
| 2003 | Xen 1.0 推出，在 Linux 上首次引入半虚拟化 |
| 2005 | Intel VT-x 和 AMD-V 带来硬件辅助虚拟化，彻底改变 x86 虚拟化格局 |
| 2006 | KVM 被合并到 Linux 内核 2.6.20 版本 |
| 2008 | EPT/NPT 技术消除影子页表开销，大幅提升性能 |
| 2013 | Docker 发布，容器技术成为虚拟化的第二波浪潮 |
| 2018 | Firecracker（轻量级虚拟机）实现毫秒级启动，为 Serverless 提供技术支持 |
## 2. 虚拟化类型

| 特性 | 全虚拟化（BT） | 半虚拟化 | 硬件辅助虚拟化 | 容器 |
|------|---------------|---------|---------------|------|
| 是否需要修改 Guest OS | 否 | 是 | 否 | 不适用 |
| CPU 开销 | 高 | 中等 | ~1-3% | 几乎为零 |
| 隔离性 | 强 | 强 | 强 | 进程级别 |
| 启动时间 | 30-60 秒 | 30-60 秒 | 20-45 秒 | < 1 秒 |
| 镜像大小 | GB 级别 | GB 级别 | GB 级别 | 数十 MB |
| 示例 | 早期 VMware | Xen PV | KVM、ESXi 6+ | Docker、containerd |

### 2.1 容器与虚拟机：隔离边界的本质差异

![虚拟机与容器的资源隔离对比：每个虚拟机自带内核，而容器共享宿主机内核](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig2_vm_vs_container.png)

这张图是本文的核心。**虚拟机虚拟的是硬件层**，因此每个虚拟机都需要携带自己的操作系统内核；而**容器虚拟的是操作系统层**，本质上是一个普通的 Linux 进程，通过命名空间（如 PID、mount、network、UTS、IPC、user 等）实现隔离，并利用 cgroups 限制资源使用。所有容器共享同一个宿主机内核。

这种设计带来了几个关键影响：

- 如果容器触发了内核漏洞，可能会危及宿主机；而虚拟机触发内核漏洞只会导致自身崩溃。
- 容器无法运行不同内核的操作系统（例如，在 Linux 宿主机上运行所谓的“Windows 容器”只是营销噱头，实际并不可行）。
- 容器启动速度极快，通常在毫秒级别，因为它无需初始化内核。

正因为如此，生产环境中常见的做法是**在虚拟机中运行容器**：虚拟机提供了安全隔离边界，而容器则提升了部署密度和启动速度。

### 2.2 启动延迟与内存开销

![冷启动延迟与空闲内存占用对比：容器 vs MicroVM vs 虚拟机（对数坐标）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig3_startup_and_memory.png)

上图揭示了为什么 Serverless 架构会选择 Firecracker MicroVM（冷启动约 125 毫秒，内存开销约 30 MB），而不是传统的 KVM/QEMU 虚拟机。启动时间和内存占用相差**两个数量级**，这对突发型工作负载的成本模型产生了根本性的影响。
## 3. Hypervisor 选型

![KVM、Xen、VMware ESXi、Hyper-V 在六个维度上的能力对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig4_hypervisor_matrix.png)

没有所谓“最好的” Hypervisor，选择哪种方案取决于你的现有技术栈、运维团队的熟悉程度以及预算对授权成本的接受度。

### 3.1 KVM（Kernel-based Virtual Machine）

KVM 将 Linux 内核直接转变为 Type 1 Hypervisor。它开源、预装在主流发行版中，并为 OpenStack、Proxmox、Amazon EC2（Nitro）、Google Cloud 以及阿里云 ECS 的大部分实例提供底层支持。

**在 Ubuntu/Debian 上安装：**

```bash
sudo apt update
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients \
                    bridge-utils virt-manager

sudo usermod -aG libvirt,kvm $USER

# 检查硬件虚拟化支持和内核模块加载情况
egrep -c '(vmx|svm)' /proc/cpuinfo   # > 0 表示支持 VT-x/AMD-V
lsmod | grep kvm
virsh list --all
```

**创建虚拟机：**

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

**常用 `virsh` 命令：**

```bash
virsh list --all                # 查看所有虚拟机
virsh start <vm>                # 启动虚拟机
virsh shutdown <vm>             # 优雅关机
virsh destroy <vm>              # 强制关闭（相当于 kill -9）
virsh suspend <vm>              # 挂起到内存
virsh snapshot-create-as <vm> snap1
virsh domstats <vm>             # 实时性能统计
```

### 3.2 VMware ESXi

ESXi 是直接运行在裸金属上的 Type 1 Hypervisor，是企业级虚拟化的事实标准，通常与 vCenter 配合进行集群管理。

默认的虚拟硬件配置使用模拟 SCSI 控制器（LSI Logic）和千兆网卡（E1000）。这种配置适合安装 Windows 系统，但在生产环境中会成为瓶颈。建议切换到半虚拟化设备：

```ini
# 优化后的 .vmx 配置片段
virtualHW.version = "19"
numvcpus = "4"
cpuid.coresPerSocket = "2"
memory = "8192"

# PVSCSI：IOPS 是 LSI Logic 的 3-4 倍
scsi0.virtualDev   = "pvscsi"
scsi0:0.fileName   = "Ubuntu-Server-22.04.vmdk"

# VMXNET3：吞吐量约 9.5 Gbps，远超 E1000 的 1 Gbps
ethernet0.virtualDev = "vmxnet3"
```

| 配置项 | 默认值 | 优化后 | 提升效果 |
|--------|--------|--------|----------|
| SCSI 控制器 | LSI Logic（约 80K IOPS） | PVSCSI（约 300K+ IOPS） | IOPS 提升 3-4 倍 |
| 网卡 | E1000（约 1 Gbps） | VMXNET3（约 9.5 Gbps） | 吞吐量提升近 10 倍 |
| CPU 开销 | 较高（模拟模式） | 较低（半虚拟化） | CPU 消耗减少 30-50% |

PVSCSI 和 VMXNET3 都需要 Guest 安装 VMware Tools 驱动。建议先用默认控制器完成系统安装，启动后再切换。

```bash
# 切换后在 Guest 中验证
lsmod | grep vmw_pvscsi
ethtool -i eth0    # 应输出 "vmxnet3"
```

### 3.3 Xen

Xen 是最早的开源 Type 1 Hypervisor，曾为第一代 AWS EC2 提供底层支持。它同时支持 PV（半虚拟化）和 HVM（硬件辅助虚拟化）模式。如今主要应用于安全敏感场景（如 Qubes OS）以及一些遗留云平台。

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

Hyper-V 内置于 Windows Server 和桌面版 Windows 中。如果你的虚拟机大多是 Windows 系统，或者团队习惯使用 PowerShell，那么 Hyper-V 是一个不错的选择。

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

### 3.5 选型建议

| 维度 | KVM | VMware ESXi | Hyper-V | Xen |
|------|-----|-------------|---------|-----|
| 成本 | 免费 | 按 socket 授权（昂贵） | 随 Windows Server 附带 | 免费 |
| 管理工具 | virsh、OpenStack | vSphere（强大的 GUI） | PowerShell、Windows Admin Center | xl、XenCenter |
| 性能 | 优秀 | 优秀 | 良好 | 优秀 |
| 适用场景 | Linux 云平台、OpenStack | 企业数据中心 | 微软生态 | 安全场景、遗留系统 |
## 4. 存储虚拟化

### 4.1 LVM（逻辑卷管理器）

LVM 的核心思想是将块设备抽象为灵活且可动态调整的逻辑卷。它的逻辑结构可以简单理解为：物理卷（`pv`）→ 卷组（`vg`）→ 逻辑卷（`lv`）。这种分层设计让存储管理更加灵活。

```bash
# 创建物理卷
sudo pvcreate /dev/sdb1 /dev/sdc1
# 将物理卷加入卷组
sudo vgcreate vg_storage /dev/sdb1 /dev/sdc1
# 在卷组中创建逻辑卷
sudo lvcreate -L 100G -n lv_data vg_storage
# 格式化逻辑卷为 ext4 文件系统
sudo mkfs.ext4 /dev/vg_storage/lv_data
# 挂载逻辑卷到指定目录
sudo mount /dev/vg_storage/lv_data /mnt/data

# 在线扩容逻辑卷，无需卸载或停机
sudo lvextend -L +50G /dev/vg_storage/lv_data
sudo resize2fs /dev/vg_storage/lv_data

# 创建写时复制快照
sudo lvcreate -L 10G -s -n lv_data_snap /dev/vg_storage/lv_data
```

### 4.2 ZFS

ZFS 是一种集卷管理和文件系统于一体的解决方案，内置了数据校验、压缩、快照以及 send/receive 复制功能。不过，它的性能对内存需求较高，通常建议每 TB 数据分配 1 GB 内存给 ARC 缓存。

```bash
# 创建 ZFS 存储池，使用 RAID-Z 配置
sudo zpool create tank raidz /dev/sdb /dev/sdc /dev/sdd

# 创建 ZFS 文件系统
sudo zfs create tank/data
# 启用 lz4 压缩，节省约 30% 空间，CPU 开销极低
sudo zfs set compression=lz4 tank/data
# 关闭 atime 更新，减少写放大问题
sudo zfs set atime=off tank/data

# 创建快照
sudo zfs snapshot tank/data@2025-01-01
# 将快照发送到远程备份主机
sudo zfs send tank/data@2025-01-01 | ssh backup-host \
     sudo zfs receive backup-pool/data
```

### 4.3 磁盘格式与 I/O 路径

在虚拟化环境中，选择合适的虚拟磁盘格式以及 QEMU 的缓存模式，往往比 CPU 的选型更能影响整体性能。

```xml
<!-- libvirt：适用于 KVM 的生产级磁盘配置 -->
<disk type='file' device='disk'>
  <driver name='qemu' type='raw' cache='none' io='native' discard='unmap'/>
  <source file='/var/lib/libvirt/images/db.raw'/>
  <target dev='vda' bus='virtio'/>
</disk>
```

```bash
# 在宿主机上设置低开销的 I/O 调度器
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
```

| 格式   | 性能   | 特性                     | 适用场景               |
|--------|--------|--------------------------|------------------------|
| Raw    | 最佳   | 无                      | 生产环境数据库、延迟敏感任务 |
| QCOW2  | 很好   | 快照、压缩、精简配置      | 开发测试、通用场景       |
| VMDK   | 良好   | VMware 生态兼容          | VMware 环境             |

`cache=none` 和 `io=native` 的组合会绕过宿主机的页缓存，直接使用异步 I/O，这是大多数带有自身文件系统缓存的虚拟机的最佳默认配置。而 `cache=writeback` 虽然性能更高，但在宿主机崩溃时可能导致数据丢失，因此仅适用于临时性工作负载。
## 5. 网络虚拟化

### 5.1 VLAN 技术

VLAN（802.1Q 标签）能够将一个物理网络划分为多个独立的广播域，从而实现流量隔离。

```bash
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100
sudo ip link set eth0.100 up
```

局限性：单个交换机支持的 4094 个 VLAN ID 对于小型机柜环境来说绰绰有余，但在公有云场景下远远不够用。

### 5.2 VXLAN 技术

VXLAN 通过 UDP 封装二层以太网帧，能够在任意 IP 网络上提供多达 1600 万个逻辑网络（基于 24 位 VNI）。这项技术是多租户云网络和 Kubernetes 覆盖网络（如 Flannel 和 Calico 的 VXLAN 模式）的核心支柱。

```text
VM A (10.1.1.10) -- VTEP1 [VNI 100] --IP 网络-- VTEP2 [VNI 100] -- VM B (10.1.1.20)
```

```bash
sudo ip link add vxlan100 type vxlan id 100 dstport 4789 \
     group 239.1.1.1 dev eth0
sudo ip addr add 10.1.1.1/24 dev vxlan100
sudo ip link set vxlan100 up
```

### 5.3 Open vSwitch（OVS）

Open vSwitch 是一款支持 OpenFlow 协议的可编程虚拟交换机，广泛应用于 OpenStack Neutron、OVN 以及各类 SDN 解决方案中。

```bash
sudo apt install -y openvswitch-switch
sudo ovs-vsctl add-br br0
sudo ovs-vsctl add-port br0 eth0
sudo ovs-vsctl add-port br0 vnet0

sudo ovs-ofctl add-flow br0 "in_port=1,actions=output:2"
sudo ovs-vsctl show
sudo ovs-ofctl dump-flows br0
```

### 5.4 SR-IOV：绕过 Hypervisor 的高效方案

SR-IOV（单根 I/O 虚拟化）允许物理网卡直接暴露多个虚拟功能（VF），并将这些 VF 映射到虚拟机中，完全绕过宿主机的网络栈。这种方式可以将延迟从约 30 微秒（virtio）降低到约 3 微秒（SR-IOV），同时实现接近线速的吞吐性能。

```bash
echo 4 | sudo tee /sys/class/net/eth0/device/sriov_numvfs
ip link show eth0
# eth0: ... vf 0 MAC ...  vf 1 MAC ...  vf 2 MAC ...  vf 3 MAC ...
```

权衡点：使用 SR-IOV 后，虚拟机的热迁移变得困难（因为 VM 绑定了特定的物理网卡），并且单张网卡支持的虚拟机数量受限于其 VF 数量。
## 6. 性能优化

### 6.1 CPU：绑核、NUMA 与拓扑优化

在多路服务器上，最糟糕的情况是 vCPU 在一个插槽（socket A）上被唤醒，但其内存却位于另一个插槽（socket B）。这种情况下，每一条缓存行（cache line）都需要跨插槽传输，导致性能大幅下降。解决这个问题的关键是 **CPU 绑核** 和 **NUMA 节点绑定**：

```bash
# 将每个 vCPU 绑定到特定的物理核心
virsh vcpupin ubuntu-server 0 0
virsh vcpupin ubuntu-server 1 1
virsh vcpupin ubuntu-server 2 2
virsh vcpupin ubuntu-server 3 3

# 强制将内存分配到同一个 NUMA 节点
virsh numatune ubuntu-server --nodeset 0 --mode strict
```

为了让虚拟机内的调度器能够做出更优的决策，需要将真实的 CPU 拓扑结构暴露给 Guest：

```xml
<cpu mode='host-passthrough' check='none'>
  <topology sockets='1' cores='4' threads='2'/>
  <cache mode='passthrough'/>
</cpu>
```

`mode='host-passthrough'` 会暴露所有的 CPU 特性（如 AVX-512、AES-NI 等），但会导致无法在异构主机之间进行热迁移。如果需要支持跨异构集群的迁移，建议使用 `host-model`。

### 6.2 内存：大页与动态调整

TLB（Translation Lookaside Buffer）表项用于映射内存页。如果使用 4 KB 的小页，遍历 1 GB 内存需要处理 262,144 个表项；而使用 2 MB 的大页时，只需处理 512 个表项。对于数据库和 JVM 等内存密集型应用，启用大页通常可以带来 5%-15% 的性能提升。

```bash
# 分配 2048 个 2 MB 大页，总计 4 GB
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages

# 配置 libvirt 使用大页为虚拟机提供内存支持
# <memoryBacking><hugepages/></memoryBacking>
```

通过气球驱动（ballooning），可以动态回收虚拟机的内存资源：

```bash
virsh setmaxmem ubuntu-server 8G --config
virsh setmem    ubuntu-server 2G --live   # 动态缩减内存，无需重启
```

### 6.3 I/O：全面采用 virtio

在 KVM 虚拟化环境中，始终推荐使用 virtio 设备。virtio 是一种半虚拟化技术，Guest 知道自己运行在虚拟机中，并通过与 Hypervisor 共享的环形缓冲区（ring buffer）进行通信，避免了 MMIO 模拟带来的开销。`vhost-net` 进一步将网络数据包的处理移到内核空间，从而减少每次数据包处理时的用户态切换。

```xml
<interface type='bridge'>
  <model type='virtio'/>
  <driver name='vhost' queues='4'/>   <!-- 多队列设计，随 vCPU 数量扩展 -->
</interface>
```

### 6.4 调优检查清单

- [ ] BIOS 中启用 VT-x / AMD-V
- [ ] 磁盘和网卡选择 virtio（KVM）或 PVSCSI/VMXNET3（VMware）
- [ ] 对于超过 8 个 vCPU 的虚拟机，必须启用 CPU 绑核和 NUMA 绑定
- [ ] 宿主机 I/O 调度器：NVMe 使用 `none`，SATA 使用 `mq-deadline`
- [ ] 生产环境磁盘配置：raw 格式、`cache=none`、`io=native`
- [ ] 内存密集型虚拟机使用大页内存
- [ ] 对于网络密集型且不需要热迁移的虚拟机，启用 SR-IOV
- [ ] 配置 virtio-net 多队列，队列数等于 vCPU 数量
- [ ] 宿主机 CPU 调度策略设置为 `performance`
## 7. 在线热迁移：让运行中的虚拟机“搬家”

![预拷贝热迁移：完整内存镜像 → 多轮脏页同步 → 亚 100 毫秒切换](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig5_live_migration.png)

热迁移是虚拟机技术从“另一种服务器部署方式”进化为“动态资源池”的关键功能。目前最常用的算法是**预拷贝（pre-copy）**，其核心思想是通过多次迭代逐步完成内存的迁移，尽量减少停机时间。

### 预拷贝的工作原理

1. **初次传输**  
   将整个 Guest 的内存镜像发送到目标主机，同时源主机上的虚拟机继续保持运行。这一步的目标是尽可能快地将大部分内存数据传输过去。

2. **脏页追踪与同步**  
   在初次传输过程中，Guest 可能会修改部分内存页（称为“脏页”）。源主机会记录这些被修改的页面，并在后续的迭代中重新发送它们。这个过程会不断重复，直到脏页生成的速度低于网络传输带宽，或者达到预设的迭代次数上限。

3. **最终切换**  
   当剩余的脏页数量足够少时，源主机暂停虚拟机，将最后一批脏页和 CPU 状态发送到目标主机，然后在目标主机上恢复运行。这一阶段的停机时间通常小于 100 毫秒，因此 TCP 连接和用户会话基本不会受到影响。

```bash
# 使用 TLS 进行 KVM 热迁移
virsh migrate --live --persistent --undefinesource \
              --copy-storage-all \
              ubuntu-server qemu+tls://dest-host/system
```

### 特殊场景下的应对策略

如果 Guest 修改内存的速度超过了网络链路的传输能力（例如运行一个高频写入的内存数据库），预拷贝可能永远无法完成。针对这种情况，有两种常见的备选方案：

- **后拷贝（post-copy）**  
  先快速切换虚拟机到目标主机，然后按需通过网络加载缺失的内存页。这种方式的优点是总停机时间更短，但缺点是可能会引入较高的尾延迟（tail latency），因为缺页中断会导致性能波动。

- **自动收敛（auto-converge）**  
  动态限制 Guest 的 CPU 性能，降低内存修改的速度，从而确保迁移能够顺利完成。这种方法适用于对停机时间敏感但可以容忍一定性能下降的场景。

### 实施热迁移的前提条件

要成功执行热迁移，需要满足以下几个关键条件：

- **存储配置**  
  源主机和目标主机之间需要共享存储，或者通过 `--copy-storage-all` 参数显式复制存储内容（不过这种方式会显著增加迁移时间）。

- **CPU 兼容性**  
  源主机和目标主机的 CPU 特性必须兼容。建议使用 `host-model` 模式，而不是 `host-passthrough`，以避免因 CPU 差异导致迁移失败。

- **网络性能**  
  网络带宽必须足够高，以确保脏页生成速度低于传输速度。对于生产环境中的虚拟机，10 Gbps 是一个实际可行的最低要求。
## 8. 嵌套虚拟化

![嵌套虚拟化架构：L0 Hypervisor → L1 Guest 中的 Hypervisor → L2 Guest VM，以及每层的性能损耗](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig6_nested_virtualization.png)

嵌套虚拟化是一种让虚拟机（Guest VM）本身充当虚拟化平台（Hypervisor）的技术。它的实际应用场景包括：

- 在云端虚拟机中搭建虚拟化实验环境
- 在持续集成（CI）流水线中动态创建虚拟机以完成构建任务
- WSL 2（Windows 上通过 Hyper-V 运行 Linux，并在其内部运行容器）
- 机密计算（通过多层 Hypervisor 实现更深层次的安全防护）

在 KVM 中启用嵌套虚拟化的步骤如下：

```bash
# Intel 平台
echo "options kvm-intel nested=1" | sudo tee /etc/modprobe.d/kvm.conf
sudo modprobe -r kvm_intel && sudo modprobe kvm_intel
cat /sys/module/kvm_intel/parameters/nested        # 应输出 Y

# AMD 平台
echo "options kvm-amd nested=1" | sudo tee /etc/modprobe.d/kvm.conf
```

接下来需要将 CPU 的虚拟化特性（`vmx` 或 `svm`）传递给第一层虚拟机（L1 Guest）：

```xml
<cpu mode='host-passthrough'/>   <!-- 最直接的方式：完整透传硬件特性 -->
```

性能开销：每增加一层嵌套，都会引入额外的 VM-exit 跳转。对于 CPU 密集型任务，第一层虚拟机（L1）通常会有 5-10% 的性能损失，而第二层虚拟机（L2）的损失可能达到 25-40%；而对于 I/O 密集型任务，如果没有在每一层都使用半虚拟化驱动，性能下降可能会更加显著。
## 9. GPU 虚拟化

![GPU 共享的四种方式：时间片轮转、vGPU、MIG 和直通](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig7_gpu_virtualization.png)

GPU 最初的设计目标是让单个进程独占整个设备。然而，要在多个虚拟机（VM）之间共享 GPU，就需要在“资源密度”和“隔离性”之间找到一个平衡点。

| 模式         | 实现方式                     | 隔离性                  | 密度       | 备注                                   |
|--------------|-----------------------------|------------------------|-----------|---------------------------------------|
| **时间片轮转** | Hypervisor 轮流切换上下文     | 弱（无性能隔离）         | 高         | 成本最低，广泛应用于机器学习训练集群   |
| **vGPU**（NVIDIA GRID） | 中介直通，每个 VM 分配一个虚拟 GPU | 中等（显存硬隔离）       | 中等       | 显存 QoS 较难保障，需 NVIDIA 授权     |
| **MIG**（A100 / H100） | 硬件将 GPU 划分为最多 7 个独立实例 | 强（独立 SM、L2 缓存、显存） | 固定规格（如 1g.5gb、2g.10gb 等） | 完全隔离，切片内无法进一步细分        |
| **直通**（VFIO） | 整张物理 GPU 绑定到单个 VM    | 原生性能                | 每张卡仅支持 1 个 VM | 性能接近裸机，完全不共享             |

在实际应用中，典型的 AI 云平台通常会采用 MIG 技术来满足推理任务的需求（需要多个小切片且要求严格的 QoS），而训练任务则倾向于使用直通模式（独占整张 GPU，追求极致性能）。

```bash
# 在 Linux 上配置 VFIO 直通
echo "vfio-pci" | sudo tee /etc/modules-load.d/vfio.conf
echo "options vfio-pci ids=10de:2204" | sudo tee /etc/modprobe.d/vfio.conf
# 在 libvirt XML 配置中添加以下内容：
# <hostdev mode='subsystem' type='pci' managed='yes'>
#   <source><address bus='0x01' slot='0x00' function='0x0'/></source>
# </hostdev>
```
## 10. 安全与隔离

### 10.1 威胁模型

Hypervisor 的核心承诺是确保租户之间无法相互读取或干扰。然而，现实中的安全漏洞主要来自以下几个方面：

- **虚拟机逃逸**：通过模拟设备的漏洞实现（例如 2015 年的 Venom 漏洞，源于 QEMU 的软盘驱动模拟）。
- **侧信道攻击**：利用共享 CPU 缓存引发的安全问题，如 L1TF、MDS 和 Spectre v2。
- **管理平面漏洞**：Hypervisor 内核中的缺陷可能导致安全隐患。

防御措施需要多层叠加，但并不存在绝对的安全。

### 10.2 Hypervisor 加固策略

```bash
# 最小化攻击面
sudo systemctl disable --now avahi-daemon cups bluetooth

# 配置防火墙：仅允许管理平面流量，并限制来源
sudo ufw enable
sudo ufw allow from 192.168.1.0/24 to any port 22
sudo ufw allow from 192.168.1.0/24 to any port 16509   # libvirt

# 仅启用 TLS 的 libvirt 配置
# /etc/libvirt/libvirtd.conf
# listen_tls = 1
# listen_tcp = 0
# auth_tls   = "sasl"
```

在宿主机上启用 CPU 侧信道缓解措施（`spectre_v2=on`、`l1tf=full`），并保持微码更新以应对最新威胁。

### 10.3 虚拟机最佳实践

- **最小化安装**：避免安装不必要的组件，如 GUI、`cups` 和 `apt-listchanges`。
- **自动更新**：启用 `unattended-upgrades` 以确保及时应用安全补丁。
- **数据加密**：对敏感数据使用 LUKS 在虚拟机内实现全盘加密。
- **网络隔离**：按信任级别划分 VLAN，切勿将 PCI 工作负载与开发用虚拟机置于同一广播域。
- **代理更新**：定期更新虚拟机代理程序，如 `qemu-guest-agent`、`vmtoolsd` 和 Hyper-V Integration Services。

### 10.4 机密计算

这是最新的安全技术层：AMD SEV-SNP、Intel TDX 和 ARM CCA 使用 Hypervisor 无法访问的密钥对虚拟机内存进行加密。即使宿主机完全被攻破，也无法读取租户的数据。目前，这项技术已在 Azure Confidential VMs、GCP Confidential VMs 和阿里云 ECS gN8v 实例中提供支持。
## 11. 故障排查指南

| 现象 | 初步检查 | 可能的解决方法 |
|------|----------|----------------|
| 虚拟机无法启动 | 检查 `journalctl -u libvirtd` 和 `qemu-img check vm.qcow2` | 检查磁盘是否损坏、权限配置是否正确，以及 SELinux/AppArmor 的标签设置 |
| CPU 性能低下 | 使用 `virt-top` 查看虚拟机资源占用，宿主机上运行 `mpstat -P ALL` | 尝试绑定 CPU 核心，将调度器模式设为 performance，并检查 %steal 是否过高 |
| 磁盘性能差 | 在宿主机上运行 `iostat -xz 1`，虚拟机内使用 `iotop` | 切换到 virtio 驱动，启用 `cache=none io=native` 参数，或将磁盘格式改为 raw |
| 网络丢包 | 使用 `ethtool -S` 查看网卡状态，`tc -s qdisc` 检查队列规则 | 启用多队列 virtio，增大 ring buffer 大小，并检查 vhost 所在 CPU 的负载 |
| 热迁移卡住 | 运行 `virsh domjobinfo <vm>` 查看迁移状态 | 如果脏页率过高，可以尝试启用 auto-converge 或切换到 post-copy 模式 |
| 内存压力大 | 使用 `free -h`、`cat /proc/meminfo` 和 `numastat` 分析内存使用情况 | 添加大页支持，优化 NUMA 绑定策略，减少内存超分比例 |
| 虚拟机随机崩溃 | 查看 `dmesg` 和 `/var/log/libvirt/qemu/<vm>.log` 日志 | 如果发现 EDAC 错误，可能是内存条故障；否则需要检查内核版本和微码更新 |
| `KVM: entry failed, hardware error 0x80000021` | 运行 `dmesg | grep KVM` 查看相关日志 | 关闭嵌套虚拟化或更新 CPU 微码，并确认 VT-x 功能是否正常启用 |
## 12. 资源规格参考

| 资源 | 轻量 | 中等 | 重型 |
|------|------|------|------|
| vCPU | 1-2 | 2-4 | 4-8+ |
| 内存 | 2 GB | 4-8 GB | 16+ GB |
| 系统盘 | 20 GB | 30-50 GB | 50-100 GB |
| 网络 | 1 Gbps | 1-10 Gbps | 10+ Gbps（SR-IOV） |

**安全超分比：**

- CPU：4:1 ~ 8:1——延迟敏感型负载降到 2:1 甚至 1:1
- 内存：1.5:1 ~ 2:1——数据库**绝不**做内存超分
- 磁盘：监控容量到位的话，精简配置可接受

## 13. 实践案例

### 13.1 企业数据中心整合

一家金融服务公司将 200 台物理服务器整合到 20 台 VMware ESXi 主机上，后端存储采用共享的 NVMe SAN。效果显著：机柜占用减少了 90%，成本节省了 60%，部署时间从原来的数周缩短到几个小时。更重要的是，运维效率大幅提升——原来需要维护四个工单队列，现在只需管理一个 vCenter 集群即可。

### 13.2 HPC 科研集群

某研究机构在 KVM 虚拟化平台上运行高耦合的 MPI 任务，通过 `host-passthrough` 模式直通 CPU、绑定 NUMA 节点、启用 1 GB 大页内存，并结合 SR-IOV InfiniBand 网络优化。最终性能达到裸机吞吐量的 95%-98%，同时还具备了实验快照功能，能够将整个实验环境打包并在不同站点之间迁移。

### 13.3 公有云计算面

超大规模云服务商（如 AWS Nitro 和阿里云神龙 ECS）通过自研 DPU 将设备模拟完全从 CPU 卸载，让宿主机的全部 CPU 资源都可用于 Guest 实例，同时大幅缩小了宿主机的攻击面，仅保留一个极简的 KVM 层作为安全边界。这种设计使得性能损耗控制在个位数百分比以内，而安全性则远超传统 ESXi 主机。
## 14. 阿里云上的那些坑

前面提到的内容在各大云平台上基本通用，但有些细节确实是阿里云 ECS 独有的“惊喜”。以下是我亲身经历、被账单教育过的教训：

- **ECS 实例规格族和底层虚拟化技术的差异。** 新一代 ECS 实例运行在阿里云的“神龙”架构上，这种架构类似于 AWS 的 Nitro 系统，通过 DPU 将 I/O 完全卸载，性能强劲。而老一代的 `n4` 和 `mn4` 系列仍然使用软件虚拟化（hypervisor），对于 I/O 密集型任务表现明显逊色。如果生产环境涉及高网络吞吐或频繁磁盘操作，建议至少选择 `g7`、`c7` 或 `r7` 系列，或者更新的实例类型。**多花的钱，可能还比不上一个周末排查“为什么 p99 延迟周二又飙高”的成本**。
  
- **突发性能型 t 系列的“隐藏成本”。** `ecs.t6` 和 `ecs.t5` 实例依赖 CPU 积分机制。刚开始用的时候一切正常，第二天却发现 `top` 显示 CPU 占用 100%，实际性能却只有单核的 20% —— 因为积分已经耗尽了。**如果你的工作负载平均 CPU 使用率超过 20%，千万别选 t 系列实例**，否则性能瓶颈会让你抓狂。

- **系统盘容量只能扩容，不能缩容。** 如果一不小心分配了一个 200 GB 的系统盘，那这台实例的整个生命周期都会背着这个“大包袱”。如果需要缩小系统盘容量，唯一的办法是从一个小容量镜像重新创建实例。

- **快照计费按存储量而非数量。** 快照费用是按照每 GB 每月来计算的，而不是按快照的数量。如果你对一块 500 GB 的磁盘每天创建一次快照，账单会迅速膨胀。建议使用增量快照（阿里云默认支持），并根据实际需求调整保留策略，避免不必要的开销。

- **维护期间的热迁移并非完全无感知。** 当物理主机需要维护时，阿里云会提前通过邮件通知具体的维护窗口时间。在迁移过程中，虚拟机可能会暂停约 10 秒钟。**对于有状态的服务（如数据库、长连接 TCP 应用）来说，必须确保应用程序能够正确处理这种短暂的中断**，否则可能会引发问题。
## 15. Region/AZ 选址：成本与延迟的权衡

选择 Region 是一个看似简单却影响深远的决策，它的成本最低，但一旦出问题，波及范围可能是最大的。以下是我在实际项目中观察到的一些数据：

| 选项 | 国内用户延迟 | 东南亚用户延迟 | 备注 |
|---|---|---|---|
| `cn-hangzhou` | 20-40 ms | 60-90 ms | 默认 Region，服务种类最全，资源池最大 |
| `cn-shanghai` | 20-40 ms | 60-90 ms | 历史故障较少，但部分特殊 SKU 不一定有货 |
| `cn-shenzhen` | 30-50 ms | 40-60 ms | 港澳和东南亚业务的首选 |
| `cn-hongkong` | 50-80 ms | 30-50 ms | 位于 GFW 之外，适合全球化业务，无需 ICP 备案 |
| `ap-southeast-1`（新加坡） | 80-120 ms | 10-30 ms | 东南亚的标准枢纽，服务覆盖全面 |

以下是几个用“学费”换来的教训，分享给各位：

- **AZ 级别的故障确实会发生。** 多 AZ 部署不是过度设计，而是确保系统在硬件故障时依然可用的基本要求。而且，SLA 合同通常也明确要求多 AZ 架构。
- **跨 Region 的延迟既昂贵又不对称。** 比如，杭州到新加坡的延迟大约是 80 ms，而杭州到法兰克福则高达 280 ms。如果你的系统依赖同步复制，这段延迟会直接体现在写入路径上，影响性能。
- **带宽费用因 Region 而异。** 国内 Region 内的流量费用相对较低，但跨境流量（尤其是从国内到国际 Region 的出向流量）往往是很多企业账单中最昂贵的部分。

这些经验虽然简单，但在实际架构设计中至关重要。希望对大家有所帮助！
## 16. 如何选择 VM、容器和函数计算

“VM 已经过时了”这种说法是不对的。实际情况是，不同的场景需要不同的技术选型，以下是常见的选项及其适用场景：

- **裸金属 ECS（神龙裸金属实例）。** 如果你的工作负载需要 GPU 直通、完整的 PCIe 访问权限，或者你需要在上面运行其他 hypervisor，那么裸金属实例就是你的菜。特别是那些对延迟极其敏感的数据库工作负载，通常都会倾向于选择裸金属。
- **普通 ECS（虚拟机）。** 这是最通用的选择。它提供了十年如一日的稳定接口，完整的 root 权限支持，几乎可以运行任何操作系统镜像。按申请的资源配置付费，即使空闲也会计费。
- **容器服务（ACK 或 Serverless 容器）。** 如果你的应用符合 12-factor 规范，并且你不想操心底层主机的管理问题，那就用容器服务。ACK Pro 提供托管的控制平面，而 ECI（弹性容器实例）则让你按秒计费，完全不用管理节点。“无节点”的模式其实比很多人想象中更常用，也更适合现代应用。
- **函数计算（FC）。** 如果你的工作负载是事件驱动型、突发性强且难以预测，那么函数计算会是一个不错的选择。比如一个每天随机触发 10 次的任务，在 FC 上可能只需花费几分钱，而在 ECS 上却可能因为闲置时间浪费掉 $30/月的成本。不过需要注意的是，函数冷启动的时间通常在 100–500 毫秒之间——对于 webhook 这类后台任务来说完全可以接受，但如果是面向用户的请求，可能会让人觉得卡顿。

大多数企业实际可行的迁移路径通常是这样的：先在 ECS 上快速搭建原型，然后通过 ACK 将其投入生产环境，最后将一些定时任务或批处理任务剥离到 FC 上运行。而一开始就直接上 Kubernetes，往往会因为复杂度过高而导致项目延期半年甚至更久。