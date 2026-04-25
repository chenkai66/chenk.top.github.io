---
title: "阿里云 PAI 实战（一）：平台总览与产品家族地图"
date: 2026-04-25 09:00:00
tags:
  - 阿里云 PAI
  - 机器学习
  - DSW
  - DLC
  - EAS
categories: 阿里云 PAI
lang: zh-CN
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 1
description: "2026 年阿里云 PAI 到底是什么、你真正会用到的五个子产品（DSW、DLC、EAS、Designer、QuickStart），它们和 ECS、OSS 是什么关系，以及一个干净的账号/区域/工作空间初始化流程。"
disableNunjucks: true
---

只要你的团队在阿里云上训练或者部署模型，迟早要进 PAI 控制台。PAI 是个伞形产品，下面挂着真正干活的几个子产品——一个 Notebook 服务、一个分布式训练服务、一个模型推理服务，外加几层 GUI/快速部署的封装。在某 AI Marketing 平台跑了一年半真实 LLM 业务之后，这个系列就是我希望第一次上线 EAS 之前有人塞给我的那本书。

第一篇先把地图画清楚，代码量刻意压到最低——后面四篇是真正的深水区。这一篇的目标是：当我后面提到 "DLC 任务" 或 "EAS 服务" 时，你已经知道它们各自属于哪一类。

## PAI 是什么，又不是什么

PAI（Platform for AI，阿里云机器学习平台）是**一组共享同一控制台、同一账号体系、同一 OSS 存储层、同一 Python SDK 的产品集合**，不是单一服务。`pai.console.aliyun.com` 这个控制台只是入口，每个图标点进去都是一个独立子产品，有自己的配额、价格、资源模型。

我用得最顺的一个心智模型：

- **PAI 是商场**。
- **DSW、DLC、EAS、Designer、QuickStart、Studio** 是商场里的工坊。
- **ECS、OSS、NAS** 是真正的硅片和字节。PAI 只是在替你编排它们。

把 PAI 当作 "在 GPU ECS 之上做了一层封装" 来理解就对了。你启动一个 DSW 实例，背后真的有一台 `ecs.gn7i-c8g1.2xlarge`（或者你选的别的型号）在某个机房启动；EAS 做扩容，真实的 GPU Pod 就拉起来。用 PAI 而不是裸 ECS 的理由很简单：CUDA/PyTorch 镜像别人替你烤好了，OSS 自动挂上，监控 dashboard 直接给你，按秒计费。

## 你真正会用到的五个子产品

一年半下来，我真正花过钱的就这五个：

| 产品 | 解决什么问题 | 什么时候用 |
|---|---|---|
| **PAI-DSW**（Data Science Workshop） | 带 GPU 的云上 Jupyter，OSS 已挂载，镜像预烤好 | 交互式开发、调试、小规模训练 |
| **PAI-DLC**（Deep Learning Container） | 托管集群上的分布式训练任务 | 多卡/多机 SFT、预训练、大规模评测 |
| **PAI-EAS**（Elastic Algorithm Service） | 模型推理服务，自动扩缩、蓝绿、流量切分 | 生产推理 endpoint |
| **PAI-Designer** | 拖拽式流程编排（前身 PAI-Studio） | ETL→训练→评估，需要交付给非程序员的链路 |
| **PAI-QuickStart** | 模型库一键部署 | 10 分钟内评估一个新的开源模型 |

此外还有 **PAI-Studio**、用于超大集群的 **PAI-灵骏**、做推理加速的 **PAI-Blade**，以及一堆其他东西，但除非你做 1000+ 卡的预训练或者 ASIC 优化，第一天可以全部忽略。

子产品的拆分对应着 ML 生命周期：

```
        DSW          DLC           EAS
         |            |             |
       [探索]      [大规模训练]   [推理]
         \           |            /
          \          |           /
           +---- OSS / NAS -----+
                    |
                  ECS GPUs
```

Designer 和 QuickStart 是正交的——它们坐在上面，最终生成的任务还是跑在 DLC/EAS 这套底座上。

## PAI 和 ECS、OSS 的关系

这是从纯云主机背景过来的人最容易踩坑的地方。三条铁律：

1. **PAI 不持有你的数据。** 数据集、checkpoint、模型工件全部存在 **OSS**（或者用 **NAS** 提供 POSIX 语义）。DSW 或 DLC 实例死掉时，你没写到 OSS 的东西就消失了。系统盘存在，但要当 `/tmp` 用。
2. **PAI 持有计算资源。** 你不会自己去开 GPU ECS 给 PAI 任务用——PAI 维护一个池子，你声明 "我要 1 张 `ecs.gn7e-c12g1.3xlarge`"，按秒计费。
3. **PAI 用你的账号但走自己的 RAM 角色。** 给 PAI 授权访问 OSS 时，你挂的是服务关联角色 `AliyunPAIAccessingOSSRole`，让 PAI 的计算节点不需要长期 AK 也能读你的 bucket。这一步不能跳——跳了 DLC 任务会在 `data_loader` 那一刻 403。

> **真实经验：** "PAI 坏了" 工单里出现频率最高的，就是 PAI 和 OSS 之间的权限问题。在调你的训练脚本之前，先在 DSW 终端里跑一句 `oss ls oss://your-bucket/`。这一步不通就先去修角色，别去改代码。

## 账号、区域、工作空间

入门只需要三件事，按顺序：

1. **一个 aliyun.com 账号**，且完成实名认证——任何 GPU 资源都需要。国际站账号可用，但杭州/上海的 GPU 库存最稳定。
2. **一个区域。** 选一个就一直用。PAI 资源、OSS bucket、GPU 都按区域隔离，跨区域既贵又慢。国内生产我默认 `cn-shanghai`，海外用 `ap-southeast-1`（新加坡）。
3. **一个工作空间（Workspace）。** 工作空间是 PAI 的租户单元——配额、数据集、模型注册表、IAM 都挂在它上面。基本上一定要至少两个：`dev`（人类在 DSW 里乱搞）和 `prod`（DLC 任务和 EAS 服务）。跨空间的权限处理略麻烦，但第一次有实习生误重启线上 endpoint 时你就会感谢这个隔离。

```bash
# 这些都是按区域隔离的
aliyun configure set --region cn-shanghai
```

## 两条路：控制台 vs SDK

和百炼一样，PAI 上的每件事都有两种做法。控制台适合一次性操作和查看状态；要写 CI 的话，SDK 才是正经路。

Python SDK 一个包搞定：

```bash
pip install alibabacloud-pai-python-sdk
```

一个 "hello PAI"——列出你的工作空间：

```python
import os
from pai.session import setup_default_session

sess = setup_default_session(
    access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
    access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
    region_id="cn-shanghai",
)

for ws in sess.workspace_api.list().items:
    print(ws.id, ws.name)
```

这段如果能打印出至少一个 workspace ID，说明你的账号、区域、凭证都配对了，可以去看第二篇。

> **真实经验：** SDK 用子账号跑，附上最小权限的 RAM 策略。永远别用主账号 AK——一旦 AK 进了 git 历史，立刻轮换。阿里云的泄露 Key 检测是有的，但远没 GitHub 自家的那么快。

## 计费一句话讲清

DSW 运行期间**按实例秒**计费，外加可选的持久化磁盘费用。DLC 也是按秒，另外有 30%-50% 折扣的抢占式 GPU 池子，前提是你的任务能 checkpoint。EAS 按副本秒数计费再加少量按调用次数，自动扩缩的 `min_replicas` 才是成本大头，请求量本身一般是次要的。Designer 和 QuickStart 自身不收费，但它们启动的 DLC/EAS 资源照常计费。新账号有几百块的免费额度，跟完整个系列绰绰有余。

## 下一篇

第二篇是 **PAI-DSW** 端到端：选机型、镜像目录、OSS-FUSE 挂载、以及一个跑得通的 CIFAR-10 ResNet notebook。第三篇 **PAI-DLC** 分布式训练——一个真实的 8 卡 LLM SFT 任务。第四篇 **PAI-EAS** 模型部署，包括我至少踩过三次的冷启动陷阱。第五篇是 **Designer 和 QuickStart** 在 "我就想赶紧上线" 场景下的诚实对比。

如果只能读一篇，读第四篇——EAS 是生产成本花费最多、文档最薄的地方。
