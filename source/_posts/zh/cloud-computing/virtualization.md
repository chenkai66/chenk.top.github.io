---
title: "虚拟化技术深度解析"
date: 2024-09-02 09:00:00
tags:
  - 云计算
  - 虚拟化
  - KVM
  - VMware
  - Docker
categories: 云计算
series:
  name: "云计算"
  part: 2
  total: 8
lang: zh-CN
mathjax: false
description: "虚拟化实战指南 -- Hypervisor（VMware、KVM、Xen、Hyper-V）、存储与网络虚拟化、性能调优以及容器对比。"
---
没有虚拟化就没有云计算。每一个 EC2 实例、每一次 Lambda 调用、每一个 Kubernetes Pod，本质上都依赖同一个把戏：**让操作系统对底层硬件深信不疑地撒谎**。本文从 CPU 指令层（让这个把戏变便宜的硬件支持），一直走到主流四大 Hypervisor，再到生产级调优——决定你的虚拟机到底跑在裸机性能的 70% 还是 99%。

## 你将学到

- CPU 虚拟化的本质（保护环、VT-x、EPT），以及 Type 1 / Type 2 Hypervisor 为什么并存
- VMware ESXi、KVM、Xen、Hyper-V 的实操配置和生产级默认值
- LVM 与 ZFS 存储虚拟化，以及磁盘格式选择如何把 IOPS 拉开 4 倍
- 网络虚拟化：VLAN、VXLAN 隧道、Open vSwitch、SR-IOV
- 性能调优：CPU 绑核、NUMA、大页、virtio、vhost-net
- 在线热迁移原理、嵌套虚拟化、GPU 共享（vGPU / MIG）
- 安全加固、隔离原语，以及一份故障排查手册

## 前置知识

- 熟悉 Linux 命令行
- 了解 OS 基础概念（进程、内核态/用户态、页表）
- 建议先阅读本系列第 1 篇（[云计算基础](/zh/cloud-computing-fundamentals/)）

---

## 1. 虚拟化基础

虚拟化为 CPU、内存、磁盘、网卡等硬件资源创建虚拟版本，让多个操作系统都以为自己独占一台机器。维持这个假象的组件叫做 **Hypervisor**，也称 VMM（虚拟机监视器）。

### 1.1 为什么离不开硬件辅助

x86 最初不可虚拟化。架构里有 17 条**敏感指令**（比如 `POPF`、`SGDT`），它们会修改全局状态，但在用户态执行时**不会触发陷入**——这意味着朴素的 Hypervisor 根本拦不住它们。2000 年代早期出现了两种绕开办法：

- **二进制翻译**（VMware）：在 Guest 内核代码运行时即时改写，把危险指令换成对 Hypervisor 的调用。聪明，但慢且复杂。
- **半虚拟化**（Xen）：修改 Guest OS，让它通过 hypercall 主动调用 Hypervisor。快，但只对配合的 OS（Linux、BSD）有效。

2005-2006 年，Intel VT-x 和 AMD-V 引入了一种新的 CPU 模式——Hypervisor 跑在 **VMX root**，Guest 跑在 **VMX non-root**——并由硬件管理两者切换（`VMENTER` / `VMEXIT`）。一夜之间，任何未修改的 OS 都能以接近原生速度运行。又过了几年，**EPT/NPT（扩展页表/嵌套页表）**消除了第二大开销——影子页表维护——把两级地址翻译彻底交给 MMU 做。

这就是虚拟化变得便宜到足以撑起公有云的那一刻。

### 1.2 Type 1 vs Type 2 Hypervisor

![Type 1（裸金属）vs Type 2（宿主型）Hypervisor 架构对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig1_hypervisor_types.png)

| | Type 1（裸金属） | Type 2（宿主型） |
|---|---|---|
| 运行位置 | 直接跑在硬件上 | 跑在宿主 OS 上 |
| 代表 | VMware ESXi、KVM、Xen、Hyper-V | VirtualBox、VMware Workstation、Parallels |
| 开销 | 极低 | 多一层宿主 OS 调度 |
| 适用 | 生产云、数据中心 | 开发笔记本、培训实验环境 |

KVM 是个特殊存在：它是把 Linux **变成** Type 1 Hypervisor 的内核模块——宿主内核和 Hypervisor 是同一个内核。

### 1.3 核心概念

- **Hypervisor（VMM）**：调度 vCPU 到 pCPU、分配内存、拦截 Guest 特权操作的那一层。
- **客户机操作系统（Guest OS）**：跑在 VM 里、对自己被虚拟化无感（或几乎无感）的 OS。
- **vCPU**：宿主调度器里的一个线程，在 VMX non-root 模式下执行 Guest 代码。
- **资源超分（overcommit）**：分配的虚拟资源总量超过物理资源。在合理范围内安全，因为 Guest 很少同时打到峰值。安全比例：CPU 4:1~8:1，内存 1.5:1~2:1。
- **气球驱动（ballooning）**：Guest 内的驱动按需把空闲内存还给宿主，使内存超分成为可能。

### 1.4 发展里程碑

| 年份 | 事件 |
|------|------|
| 1972 | IBM VM/370——第一个商用 OS 级虚拟化 |
| 1999 | VMware Workstation 通过二进制翻译虚拟化 x86 |
| 2003 | Xen 1.0 在 Linux 上引入半虚拟化 |
| 2005 | Intel VT-x 和 AMD-V 把硬件辅助虚拟化带到 x86 |
| 2006 | KVM 合入 Linux 2.6.20 |
| 2008 | EPT / NPT 消除影子页表开销 |
| 2013 | Docker 发布，容器成为第二波浪潮 |
| 2018 | Firecracker（microVM）让 VM 启动进入毫秒级，撑起 Serverless |

## 2. 虚拟化类型

| 维度 | 全虚拟化（BT） | 半虚拟化 | 硬件辅助 | 容器 |
|------|---------------|---------|---------|------|
| 是否修改 Guest OS | 否 | 是 | 否 | 不适用 |
| CPU 开销 | 高 | 中 | ~1-3% | 接近 0 |
| 隔离性 | 强 | 强 | 强 | 进程级 |
| 启动时间 | 30-60 秒 | 30-60 秒 | 20-45 秒 | < 1 秒 |
| 镜像大小 | GB 级 | GB 级 | GB 级 | 几十 MB |
| 代表 | 早期 VMware | Xen PV | KVM、ESXi 6+ | Docker、containerd |

### 2.1 容器 vs 虚拟机：完全不同的隔离边界

![VM 与容器的资源隔离边界：每个 VM 自带内核，容器共享宿主内核](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig2_vm_vs_container.png)

这是本文最重要的一张图。**虚拟机虚拟的是硬件**——所以每个 Guest 都要带上自己的内核。**容器虚拟的是 OS**——它不过是一个 Linux 进程，被命名空间（PID、mount、network、UTS、IPC、user）隔离起来，再用 cgroups 限制资源用量。所有容器共享一个内核。

由此带来的后果：

- 触发内核 bug 的容器可能逃逸到宿主；触发内核 bug 的 VM 只会把自己搞崩。
- 容器无法运行不同的 OS（Linux 宿主内核上"跑 Windows 容器"是营销话术，实际不行）。
- 容器毫秒级启动，因为没有内核要初始化。

这就是为什么生产环境常见的部署是 **VM 里跑容器**：VM 提供安全边界，容器提供密度和速度。

### 2.2 启动延迟与内存开销

![启动延迟与空闲内存占用：容器、microVM、VM（对数坐标）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig3_startup_and_memory.png)

上图就是 Serverless 必须建立在 Firecracker microVM（约 125 ms 冷启动、约 30 MB 开销）而不是传统 KVM/QEMU VM 上的根本原因。启动时间和内存开销差**两个数量级**，对突发型工作负载的经济模型完全不同。

## 3. Hypervisor 选型

![KVM、Xen、VMware ESXi、Hyper-V 在六个维度上的能力雷达图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig4_hypervisor_matrix.png)

没有"最佳" Hypervisor——选什么取决于你已有的栈、谁来运维、能接受什么样的授权成本。

### 3.1 KVM（Kernel-based Virtual Machine）

KVM 把 Linux 本身变成 Type 1 Hypervisor。开源、所有主流发行版自带，支撑着 OpenStack、Proxmox、Amazon EC2（Nitro）、Google Cloud，以及阿里云 ECS 大部分实例族。

**Ubuntu/Debian 安装：**

```bash
sudo apt update
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients \
                    bridge-utils virt-manager

sudo usermod -aG libvirt,kvm $USER

# 验证硬件支持和内核模块
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
virsh list --all                # 列出所有 VM
virsh start <vm>                # 启动
virsh shutdown <vm>             # 优雅关机
virsh destroy <vm>              # 强制停止（相当于 kill -9）
virsh suspend <vm>              # 挂起到内存
virsh snapshot-create-as <vm> snap1
virsh domstats <vm>             # 实时统计
```

### 3.2 VMware ESXi

ESXi 是直接跑在服务器硬件上的 Type 1 Hypervisor，是企业事实标准，配合 vCenter 做集群管理。

VM 默认硬件用的是模拟 SCSI（LSI Logic）和模拟千兆网卡（E1000）。装个 Windows 安装盘没问题，但跑生产负载就是瓶颈。改用半虚拟化设备：

```ini
# 优化后的 .vmx 片段
virtualHW.version = "19"
numvcpus = "4"
cpuid.coresPerSocket = "2"
memory = "8192"

# PVSCSI：IOPS 是 LSI Logic 的 3-4 倍
scsi0.virtualDev   = "pvscsi"
scsi0:0.fileName   = "Ubuntu-Server-22.04.vmdk"

# VMXNET3：约 9.5 Gbps vs E1000 的 1 Gbps
ethernet0.virtualDev = "vmxnet3"
```

| 配置 | 默认 | 优化后 | 收益 |
|------|------|--------|------|
| SCSI 控制器 | LSI Logic（约 80K IOPS） | PVSCSI（约 300K+ IOPS） | 3-4 倍 IOPS |
| 网卡 | E1000（约 1 Gbps） | VMXNET3（约 9.5 Gbps） | 约 10 倍吞吐 |
| CPU 开销 | 较高（模拟） | 较低（半虚拟化） | CPU 消耗降低 30-50% |

PVSCSI 和 VMXNET3 都需要 Guest 安装 VMware Tools 驱动。安装 OS 时先用旧控制器，启动后切换。

```bash
# Guest 切换后验证
lsmod | grep vmw_pvscsi
ethtool -i eth0    # 应输出 "vmxnet3"
```

### 3.3 Xen

Xen 是开源 Type 1 Hypervisor 的鼻祖，第一代 AWS EC2 就跑在它之上。同时支持 PV 和 HVM（硬件辅助）模式。如今主要在安全敏感栈（Qubes OS）和少数遗留云上还能见到。

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

Hyper-V 随 Windows Server 和桌面版 Windows 附带。当大多数 Guest 是 Windows，或团队习惯 PowerShell 时是首选。

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

### 3.5 选型决策

| 维度 | KVM | VMware ESXi | Hyper-V | Xen |
|------|-----|-------------|---------|-----|
| 成本 | 免费 | 按 socket 授权（$$$） | 随 Windows Server 附带 | 免费 |
| 管理 | virsh、OpenStack | vSphere（最强 GUI） | PowerShell、Windows Admin Center | xl、XenCenter |
| 性能 | 优秀 | 优秀 | 很好 | 优秀 |
| 适用场景 | Linux 云、OpenStack | 企业数据中心 | 微软技术栈 | 安全场景、遗留系统 |

## 4. 存储虚拟化

### 4.1 LVM（逻辑卷管理器）

LVM 把块设备抽象成可灵活调整大小的逻辑卷。心智模型：物理卷（pv）→ 卷组（vg）→ 逻辑卷（lv）。

```bash
sudo pvcreate /dev/sdb1 /dev/sdc1
sudo vgcreate vg_storage /dev/sdb1 /dev/sdc1
sudo lvcreate -L 100G -n lv_data vg_storage
sudo mkfs.ext4 /dev/vg_storage/lv_data
sudo mount /dev/vg_storage/lv_data /mnt/data

# 在线扩容（不卸载、不停机）
sudo lvextend -L +50G /dev/vg_storage/lv_data
sudo resize2fs   /dev/vg_storage/lv_data

# 写时复制快照
sudo lvcreate -L 10G -s -n lv_data_snap /dev/vg_storage/lv_data
```

### 4.2 ZFS

ZFS 把卷管理和文件系统合二为一，内置校验、压缩、快照和 send/receive 复制。代价是 RAM（经验值：每 TB 1 GB ARC）。

```bash
sudo zpool create tank raidz /dev/sdb /dev/sdc /dev/sdd

sudo zfs create tank/data
sudo zfs set compression=lz4 tank/data        # ~30% 空间节省，CPU 几乎免费
sudo zfs set atime=off       tank/data        # 减少写放大

sudo zfs snapshot tank/data@2025-01-01
sudo zfs send tank/data@2025-01-01 | ssh backup-host \
     sudo zfs receive backup-pool/data
```

### 4.3 磁盘格式与 I/O 路径

虚拟磁盘镜像的格式选择，以及给 QEMU 的 cache 模式，往往比 CPU 选型更影响性能。

```xml
<!-- libvirt：KVM 生产级磁盘配置 -->
<disk type='file' device='disk'>
  <driver name='qemu' type='raw' cache='none' io='native' discard='unmap'/>
  <source file='/var/lib/libvirt/images/db.raw'/>
  <target dev='vda' bus='virtio'/>
</disk>
```

```bash
# 宿主使用低开销 I/O 调度器
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
```

| 格式 | 性能 | 特性 | 适用 |
|------|------|------|------|
| Raw | 最佳 | 无 | 生产 DB、延迟敏感 |
| QCOW2 | 很好 | 快照、压缩、精简配置 | 通用、开发测试 |
| VMDK | 良好 | VMware 生态 | VMware 环境 |

`cache=none` + `io=native` 绕过宿主 page cache，使用异步 direct I/O——这是给所有自带文件系统缓存的 Guest（也就是所有 Guest）应当采用的默认值。`cache=writeback` 更快，但宿主崩溃会丢数据，只用于易失性场景。

## 5. 网络虚拟化

### 5.1 VLAN

VLAN 标签（802.1Q）把一张物理网络切成多个隔离的广播域。

```bash
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100
sudo ip link set eth0.100 up
```

限制：4094 个 VLAN ID 对单机柜够用，对公有云远远不够。

### 5.2 VXLAN

VXLAN 把二层以太网帧封装在 UDP 包里，提供 1600 万逻辑网络（24 位 VNI），跑在任意 IP 网络之上。这是多租户云网络和 Kubernetes 覆盖网络（Flannel、Calico VXLAN 模式）的主力。

```text
VM A (10.1.1.10) -- VTEP1 [VNI 100] --IP 网络-- VTEP2 [VNI 100] -- VM B (10.1.1.20)
```

```bash
sudo ip link add vxlan100 type vxlan id 100 dstport 4789 \
     group 239.1.1.1 dev eth0
sudo ip addr add 10.1.1.1/24 dev vxlan100
sudo ip link set vxlan100 up
```

### 5.3 Open vSwitch

OVS 是支持 OpenFlow 的可编程虚拟交换机，被 OpenStack Neutron、OVN 和大量 SDN 栈使用。

```bash
sudo apt install -y openvswitch-switch
sudo ovs-vsctl add-br br0
sudo ovs-vsctl add-port br0 eth0
sudo ovs-vsctl add-port br0 vnet0

sudo ovs-ofctl add-flow br0 "in_port=1,actions=output:2"
sudo ovs-vsctl show
sudo ovs-ofctl dump-flows br0
```

### 5.4 SR-IOV：跳过 Hypervisor

SR-IOV（单根 I/O 虚拟化）让网卡向外暴露多个 VF（虚拟功能），直接映射到 VM，**完全绕开宿主网络栈**。延迟从 ~30 µs（virtio）降到 ~3 µs（SR-IOV），吞吐打满线速。

```bash
echo 4 | sudo tee /sys/class/net/eth0/device/sriov_numvfs
ip link show eth0
# eth0: ... vf 0 MAC ...  vf 1 MAC ...  vf 2 MAC ...  vf 3 MAC ...
```

代价：热迁移变难（VM 绑死了具体物理网卡），且单网卡上 VM 数受限于 VF 数。

## 6. 性能优化

### 6.1 CPU：绑核、NUMA、拓扑

在多路服务器上，最坏的情况是 vCPU 在 socket A 唤醒，但内存在 socket B——每条 cache line 都要跨 socket 传一次。解决方法是 **CPU 绑核 + NUMA 绑定**：

```bash
# 每个 vCPU 绑到一个物理核
virsh vcpupin ubuntu-server 0 0
virsh vcpupin ubuntu-server 1 1
virsh vcpupin ubuntu-server 2 2
virsh vcpupin ubuntu-server 3 3

# 强制内存分配在同一个 NUMA 节点
virsh numatune ubuntu-server --nodeset 0 --mode strict
```

把真实 CPU 拓扑暴露给 Guest，让 Guest 调度器能做出正确决策：

```xml
<cpu mode='host-passthrough' check='none'>
  <topology sockets='1' cores='4' threads='2'/>
  <cache mode='passthrough'/>
</cpu>
```

`mode='host-passthrough'` 暴露所有 CPU 特性（AVX-512、AES-NI 等），但破坏跨异构主机的热迁移；如果需要跨异构集群迁移，改用 `host-model`。

### 6.2 内存：大页与气球

每条 TLB 表项映射一页。4 KB 页时遍历 1 GB 内存要 262144 项；2 MB 大页时只要 512 项。数据库和 JVM 通常因此快 5-15%。

```bash
# 分配 2048 个大页 * 2 MB = 4 GB
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages

# 让 libvirt 用大页给 VM
# <memoryBacking><hugepages/></memoryBacking>
```

气球驱动支持动态归还内存：

```bash
virsh setmaxmem ubuntu-server 8G --config
virsh setmem    ubuntu-server 2G --live   # 在线缩容
```

### 6.3 I/O：处处用 virtio

KVM Guest 一律用 virtio。它是半虚拟化的——Guest 知道自己在 VM 里，使用与 Hypervisor 共享的环形缓冲区，而不是 MMIO 模拟。`vhost-net` 把网络环处理移进内核，砍掉每个数据包一次用户态切换。

```xml
<interface type='bridge'>
  <model type='virtio'/>
  <driver name='vhost' queues='4'/>   <!-- 多队列随 vCPU 扩展 -->
</interface>
```

### 6.4 调优清单

- [ ] BIOS 里启用 VT-x / AMD-V
- [ ] 磁盘和网卡用 virtio（KVM）或 PVSCSI/VMXNET3（VMware）
- [ ] 超过 8 vCPU 的 VM 必须 CPU 绑核 + NUMA 绑定
- [ ] 宿主 I/O 调度器：NVMe 用 `none`，SATA 用 `mq-deadline`
- [ ] 生产磁盘：raw 格式、`cache=none`、`io=native`
- [ ] 内存密集 Guest 用大页内存
- [ ] 网络密集且不需要热迁移的 VM 用 SR-IOV
- [ ] virtio-net 多队列，队列数 = vCPU 数
- [ ] 宿主 CPU 调度策略 = `performance`

## 7. 在线热迁移：把运行中的 VM 搬走

![预拷贝热迁移：完整镜像 → 多轮脏页同步 → 亚 100 ms 切换](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig5_live_migration.png)

热迁移是把 VM 从"另一种装服务器的方式"变成"流动资源池"的关键功能。主流算法是**预拷贝（pre-copy）**：

1. **首轮拷贝。** 把整个 Guest 内存镜像传到目标主机，源端 VM 继续运行。
2. **多轮脏页同步。** 源端追踪第 1 步期间被改写的页面，重新发送。重复，直到脏页率低于网络带宽，或达到轮数上限。
3. **停机切换。** 暂停源端 VM，发送最后一批脏页和 CPU 状态，在目标端恢复执行。停机时间通常 < 100 ms——TCP 连接和用户会话都不会断。

```bash
# 通过 TLS 进行 KVM 热迁移
virsh migrate --live --persistent --undefinesource \
              --copy-storage-all \
              ubuntu-server qemu+tls://dest-host/system
```

如果 Guest 写脏内存的速度超过链路传输速度（比如一个忙碌的内存数据库），预拷贝永远不会收敛。备选方案是 **post-copy**（先切换，按需通过网络拉缺页——总停机时间更短，但尾延迟更高）和 **auto-converge**（限制 Guest CPU 直到脏页率下降）。

前提：

- 共享存储，**或** 加 `--copy-storage-all`（更慢）
- 源和目标 CPU 特性兼容（用 `host-model`，不要用 `host-passthrough`）
- 网络足够快，使脏页率 < 带宽——10 Gbps 是生产 VM 的下限

## 8. 嵌套虚拟化

![嵌套虚拟化栈：L0 Hypervisor → L1 Guest 内的 Hypervisor → L2 Guest VM，以及每层的吞吐损失](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig6_nested_virtualization.png)

嵌套虚拟化让 Guest VM 自己也能当 Hypervisor。真实用途：

- 在云 VM 里跑 Hypervisor 实验
- CI 流水线里临时拉起 VM 做构建
- WSL 2（Windows 上的 Hyper-V 跑 Linux 跑容器）
- 机密计算（一层 Hypervisor 套另一层做纵深防御）

KVM 启用嵌套：

```bash
# Intel
echo "options kvm-intel nested=1" | sudo tee /etc/modprobe.d/kvm.conf
sudo modprobe -r kvm_intel && sudo modprobe kvm_intel
cat /sys/module/kvm_intel/parameters/nested        # 应输出 Y

# AMD
echo "options kvm-amd nested=1" | sudo tee /etc/modprobe.d/kvm.conf
```

然后把 `vmx`（或 `svm`）特性透传给 L1 Guest：

```xml
<cpu mode='host-passthrough'/>   <!-- 最简单的方式：完整透传 -->
```

代价：每多一层嵌套就多一次 VM-exit。CPU 密集型负载在 L1 通常损失 5-10%，到 L2 损失 25-40%；I/O 密集型如果没有半虚拟化驱动一路传下去，损失会大得多。

## 9. GPU 虚拟化

![GPU 共享的四种方式：时间片、vGPU、MIG、直通](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/cloud-computing/virtualization/fig7_gpu_virtualization.png)

GPU 设计之初就是给一个进程独占的。要把它分给多个 VM，必须在"密度 vs 隔离"的谱系上选一个点：

| 模式 | 原理 | 隔离性 | 密度 | 备注 |
|------|------|-------|------|------|
| **时间片** | Hypervisor 轮转上下文 | 弱（无性能隔离） | 高 | 最便宜，很多训练集群在用 |
| **vGPU**（NVIDIA GRID） | 中介直通，每 VM 一个虚拟 GPU 切片 | 中（显存硬隔离） | 中 | 需要 NVIDIA 授权 |
| **MIG**（A100 / H100） | 硬件把 GPU 切成最多 7 个独立实例 | 强（独立 SM、L2、显存） | 固定形状（1g.5gb、2g.10gb…） | 真隔离，切片内不能再分 |
| **直通**（VFIO） | 整张物理 GPU 绑给一个 VM | 原生 | 1 VM/卡 | 裸机速度，零共享 |

典型 AI 云的做法：推理用 MIG（多个小切片，硬 QoS），训练用直通（一个任务独占整张卡）。

```bash
# Linux 上的 VFIO 直通
echo "vfio-pci" | sudo tee /etc/modules-load.d/vfio.conf
echo "options vfio-pci ids=10de:2204" | sudo tee /etc/modprobe.d/vfio.conf
# libvirt XML 中：
# <hostdev mode='subsystem' type='pci' managed='yes'>
#   <source><address bus='0x01' slot='0x00' function='0x0'/></source>
# </hostdev>
```

## 10. 安全与隔离

### 10.1 威胁模型

Hypervisor 给租户的根本承诺是：一个租户不能读取或影响另一个。现实中被攻破的途径包括：

- **VM 逃逸**——通过模拟设备的 bug（Venom，2015 年，QEMU 软驱模拟）
- **侧信道**——共享 CPU 缓存导致的 L1TF、MDS、Spectre v2
- **管理面 Hypervisor 内核 bug**

防御是分层的，没有银弹。

### 10.2 Hypervisor 加固

```bash
# 攻击面最小化
sudo systemctl disable --now avahi-daemon cups bluetooth

# 防火墙：仅允许管理面流量，限制源
sudo ufw enable
sudo ufw allow from 192.168.1.0/24 to any port 22
sudo ufw allow from 192.168.1.0/24 to any port 16509   # libvirt

# libvirt 仅 TLS
# /etc/libvirt/libvirtd.conf
# listen_tls = 1
# listen_tcp = 0
# auth_tls   = "sasl"
```

启用宿主侧信道缓解（`spectre_v2=on`、`l1tf=full`），保持微码更新。

### 10.3 Guest 最佳实践

- 最小化安装（无 GUI、无 `cups`、无 `apt-listchanges`）
- 自动安全更新（`unattended-upgrades`）
- 敏感数据在 Guest 内做全盘加密（LUKS）
- 按信任等级划分 VLAN——绝不要把 PCI 工作负载和开发 VM 放在同一广播域
- Guest 代理（`qemu-guest-agent`、`vmtoolsd`、Hyper-V Integration Services）保持最新

### 10.4 机密计算

最新的一层：AMD SEV-SNP、Intel TDX、ARM CCA 用 Hypervisor 看不到的密钥加密 Guest 内存。即便宿主完全沦陷，也读不到租户数据。今天可在 Azure Confidential VMs、GCP Confidential VMs、阿里云 ECS gN8v 上买到。

## 11. 故障排查手册

| 现象 | 先看哪里 | 常见处置 |
|------|---------|---------|
| VM 启动失败 | `journalctl -u libvirtd`、`qemu-img check vm.qcow2` | 镜像损坏、权限、SELinux/AppArmor 标签 |
| CPU 慢 | `virt-top`、宿主 `mpstat -P ALL` | CPU 绑核、调度策略改 performance、看 %steal |
| 磁盘慢 | `iostat -xz 1`、Guest `iotop` | 改 virtio + `cache=none io=native`、raw 格式 |
| 网络丢包 | `ethtool -S`、`tc -s qdisc` | 多队列 virtio、加大 ring buffer、看 vhost CPU |
| 热迁移卡住 | `virsh domjobinfo <vm>` | 脏页率太高——开启 auto-converge 或换 post-copy |
| 内存压力 | `free -h`、`/proc/meminfo`、`numastat` | 加大页、修 NUMA 绑定、降低超分 |
| VM 随机崩溃 | `dmesg`、`/var/log/libvirt/qemu/<vm>.log` | EDAC 报错 → 内存条坏；否则查内核 + 微码 |
| `KVM: entry failed, hardware error 0x80000021` | `dmesg | grep KVM` | 关掉嵌套或更新微码；检查 VT-x 状态 |

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

某金融服务公司把 200 台物理服务器整合到 20 台 VMware ESXi 主机上，后端是共享 NVMe SAN。结果：机柜空间减少 90%，成本节省 60%，部署时间从数周缩到数小时。更大的收益是运维侧——一个 vCenter 集群替掉了四个工单队列。

### 13.2 HPC 科研集群

某科研院所在 KVM 上跑紧耦合 MPI 任务，用 `host-passthrough`、NUMA 绑定、1 GB 大页、SR-IOV InfiniBand。可持续达到裸机 95-98% 的吞吐，同时获得整实验快照和跨站点搬迁的能力。

### 13.3 公有云计算面

超大规模云（AWS Nitro、阿里云神龙）把设备模拟完全卸载到自研 DPU 上，让 100% 的宿主 CPU 留给 Guest，同时把宿主攻击面缩小成一个极薄的 KVM。性能损失个位数百分点，安全边界比传统 ESXi 主机更小。

---

## 系列导航

| 篇 | 主题 |
|----|------|
| 1 | [基础与架构体系](/zh/cloud-computing-fundamentals/) |
| **2** | **虚拟化技术深度解析（当前）** |
| 3 | [存储系统与分布式架构](/zh/cloud-computing-storage-systems/) |
| 4 | [网络架构与 SDN](/zh/cloud-computing-networking-sdn/) |
| 5 | [安全与隐私保护](/zh/cloud-computing-security-privacy/) |
| 6 | [运维与 DevOps 实践](/zh/cloud-computing-operations-devops/) |
| 7 | [云原生与容器技术](/zh/cloud-computing-cloud-native-containers/) |
| 8 | [多云与混合架构](/zh/cloud-computing-multi-cloud-hybrid/) |
