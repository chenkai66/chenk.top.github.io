---
title: "阿里云 PAI（一）：平台概览与产品地图"
date: 2026-03-05 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - DSW
  - DLC
  - EAS
categories: 阿里云 PAI
lang: zh
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 1
description: "基于官方文档梳理 2026 年阿里云 PAI 的四层服务架构、你真正会用到的五个子产品（DSW、DLC、EAS、Designer、Model Gallery），它们和 ECS、OSS 的关系，以及一套干净的账号/区域/工作空间初始化流程。"
disableNunjucks: true
translationKey: "aliyun-pai-1"
---
只要你的团队在阿里云上训练或部署模型，迟早会用到 PAI 控制台。PAI 是一个平台级整合层（umbrella），真正干活的是它底下的核心产品：笔记本（DSW）、分布式训练服务（DLC）、模型推理服务（EAS），以及两个面向快速交付的 GUI 层——Designer 和 Model Gallery。我在一个 AI 营销平台上用 PAI 跑了大约十八个月的真实 LLM 负载，写下这个系列，就是希望你在部署第一个 endpoint 前，能少踩些我踩过的坑。

本文是整个系列的“地形图”，刻意少放代码——深度实操留到第 2 到第 5 篇。目标很简单：让你下次听到“DLC job”或“EAS endpoint”时，心里清楚它们到底指什么。

![阿里云PAI（1）：平台概览及产品家族图 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_1.png)

## PAI 是什么，不是什么

官方文档称，**Platform for AI (PAI)** 是“阿里云覆盖全生命周期的 AI 开发平台，涵盖数据标注、模型开发、训练和部署”。控制台 `pai.console.aliyun.com` 只是一个入口，PAI 本身其实是一组紧密关联的产品家族，它们共享账号体系、基于 OSS 的存储层，以及统一的 Python SDK。

我最常用的心智模型是：

- **PAI 是店面。**
- **DSW、DLC、EAS、Designer、Model Gallery** 是店里的工作台。
- **ECS、OSS、NAS、CPFS** 才是算力和数据真正存放的地方——PAI 只是替你调度它们。

官方“服务架构”图将 PAI 描述为四层堆栈：

![PAI四层服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig1_pai_4layer_architecture.png)

建议从下往上看：最底层是基础设施层，包括 CPU、GPU、RDMA 网络和 ACK（阿里云 Kubernetes 服务）。往上一层，*灵骏*（Lingjun）提供高密度 AI 算力，而*通用算力*则基于 ECS 提供日常 GPU 资源池。再往上是平台与工具层——你日常打交道最多的地方，包括 PyTorch、Megatron、DeepSpeed，以及 PAI 自研的优化工具（TorchAcc、BladeLLM、EasyCkpt、AIMaster）和可见产品（DSW、DLC、EAS、Designer、FeatureStore、iTAG）。应用层则将 PAI 与阿里 MaaS 生态（如 ModelScope、百炼/DashScope、Model Studio）打通。最顶层的业务层，主要是面向行业的解决方案。

相比直接使用裸 ECS，PAI 的优势在于：预装 CUDA 和 PyTorch 镜像、自动挂载你的 OSS Bucket、提供监控仪表盘，并支持按秒计费。

## 你实际会用到的五个子产品

经过一年半的生产实践，我只为以下五个组件付过钱——它们直接来自官方“核心组件”表格：

| 组件 | 官方定义 | 适用场景 |
|---|---|---|
| **DSW** (Data Science Workshop) | 云端 IDE，集成 Jupyter / VSCode / 终端，预装 PyTorch 和 TensorFlow 镜像，支持 GPU 实例 | 交互式开发、调试、小规模训练 |
| **DLC** (Deep Learning Containers) | 基于 Kubernetes 的训练服务，支持 Megatron、DeepSpeed、PyTorch、TF、Slurm、Ray、MPI、XGBoost，无需自行搭建集群 | 多 GPU / 多节点 SFT、预训练、大规模评估 |
| **EAS** (Elastic Algorithm Service) | 在线推理服务，支持自动扩缩容、灰度发布、流量拆分和镜像回放 | 生产环境的推理 endpoint |
| **Designer** | 提供 140+ 内置算法组件，支持拖拽式编排、JSON 导出，并可在 DataWorks 中调度 | 交给非编码人员的 ETL → 训练 → 评估流程 |
| **Model Gallery** | 封装 DLC + EAS，实现对目录中开源模型的零代码部署与微调 | 10 分钟内快速试用 Qwen / DeepSeek / Llama 等模型 |

此外还有 iTAG（数据标注）、PAI-Lingjun（超大规模集群）、PAI-Blade / BladeLLM（推理优化）和 FeatureStore。除非你要做千卡级预训练或构建推荐系统，否则初期完全可以忽略它们。

这些产品的划分清晰对应机器学习的生命周期：

![PAI子产品在机器学习生命周期中的位置](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig2_pai_subproducts_lifecycle.png)

Designer 和 Model Gallery 功能正交——它们都位于上层，但底层任务最终仍由 DLC 或 EAS 执行。

## PAI 与 ECS、OSS 的关系

这一点最容易让习惯直接操作云虚拟机（ECS）的用户困惑。记住三条铁律：

1. **PAI 从不拥有你的数据。** 数据集、检查点（checkpoint）和模型产物（artifact）全部存放在 **OSS** 中（若需 POSIX 语义可用 **NAS**，追求 HPC 级吞吐则用 **CPFS**）。一旦 DSW 或 DLC 实例终止，未写入 OSS 的数据将永久丢失。虽然实例自带“系统盘”，但请把它当作 `/tmp` 使用。
2. **PAI 掌管计算资源。** 你无需手动创建 GPU ECS 实例来跑 PAI 任务。PAI 管理一个资源池，你只需申请 `1 * ecs.gn7i-c8g1.2xlarge`，系统便按实际分配时长（精确到秒）计费。
3. **PAI 共享你的账号，但使用独立的 RAM 角色。** 当你授权 PAI 访问 OSS 时，实际上是附加了一个服务关联角色（`AliyunPAIAccessingOSSRole`），使 PAI 的计算节点能在不暴露长期 AK 的前提下读取你的 Bucket。千万别跳过这一步——否则 DLC 任务会在 `data_loader` 阶段因 403 错误直接失败。

> **实战建议：** 最常见的“PAI 出问题”工单，其实都是 PAI 与 OSS 之间的权限配置错误。在排查训练脚本前，先在 DSW 终端执行 `oss ls oss://your-bucket/`。如果这步失败，优先修复角色权限，而不是改代码。

## 账号、地域与工作空间

起步必须按顺序完成三件事：

1. **一个 aliyun.com 账号**，并完成实名认证（实名认证）——这是使用任何 GPU 资源的前提。国际账号在多数区域可用，但杭州、上海和北京的 GPU 库存最充足。
2. **选定一个地域**，并坚持使用。PAI 资源、OSS Bucket 和 ECS GPU 均为地域级隔离，跨地域传输不仅产生费用，还会增加延迟。国内生产环境我默认选 `cn-shanghai`；国际业务则倾向 `ap-southeast-1`（新加坡）。
3. **创建工作空间（Workspace）**。按官方说法，工作空间是 PAI 的租户基本单元，用于管理配额、数据集、模型注册表和 IAM 权限绑定。你几乎总是需要至少两个：一个 `dev` 工作空间供人工在 DSW 中实验，一个 `prod` 工作空间专用于运行 DLC 任务和 EAS endpoint。跨工作空间的权限配置相当繁琐，但第一次有实习生误重启生产 endpoint 时，你就会感谢这份隔离。

![PAI租户：账号、区域、工作空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig3_pai_account_workspace.png)

## 两条路径：控制台 vs SDK

和百炼（Bailian）一样，PAI 为所有操作提供了两种方式：控制台适合一次性任务和状态查看，而 SDK 才是你集成到 CI/CD 流水线中的正确选择。

Python SDK 只需安装一个包：

```bash
pip install alibabacloud-pai-python-sdk
```

来个“Hello PAI”——列出你的工作空间：

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

如果这段代码能成功打印出至少一个工作空间 ID，说明你的账号、地域和凭证配置无误，可以放心进入第 2 篇。

> **实战建议：** SDK 操作务必使用子账号，并为其配置最小权限的 RAM Policy。切勿使用主账号 Access Key；一旦 AK 泄露到 Git 历史中，请立即轮换。阿里云的密钥泄露检测机制尚可，但远不如 GitHub 敏捷。

## 一段话讲清计费模型

官方文档列出了五种计费方式：**按量付费**、**包年包月**（预付）、**资源包**（DSW 预付额度）、**节省计划**（承诺用量享折扣），以及 **按推理时长付费**（EAS Serverless 模式，无空闲副本成本）。具体来看：DSW 按实例运行时长（秒）计费；DLC 同样按秒计费，且提供独立的 Spot/抢占式 GPU 配额，若任务支持断点续训，成本可降低 30%–50%；EAS 则按副本运行秒数 + 每百万次请求的小额费用计价，其中自动扩缩容设定的最小副本数往往是成本大头。Designer 和 Model Gallery 本身不收费——它们只是触发 DLC/EAS 资源，后者照常计费。新账号还享有小额免费额度，足够完整体验本系列所有内容。

## Designer 工作流底层到底长什么样

![阿里云PAI（1）：平台概览和产品家族图 — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_2.png)

Designer 的画布操作看似像搭积木，但导出的其实是一份扁平化的 JSON 文档，平台可直接重放执行。第一次我需要调试一个卡住的流水线时，从画布菜单（`...` → *Export workflow*）导出了这份 JSON，瞬间揭开了它的“魔法”面纱。结构大致如下：

```json
{
  "workflowName": "user-segmentation-weekly",
  "workspaceId": "ws-1xxxx",
  "globalParams": {
    "outputProject": "marketing_dwd",
    "samplingRatio": 0.05
  },
  "nodes": [
    {
      "id": "n1",
      "type": "ReadODPSTable",
      "params": {"projectName": "marketing_dwd", "tableName": "user_features_p"},
      "position": {"x": 60, "y": 80}
    },
    {
      "id": "n2",
      "type": "Split",
      "params": {"trainRatio": 0.8},
      "inputs": [{"from": "n1", "port": "out"}]
    },
    {
      "id": "n3",
      "type": "KMeans",
      "params": {"k": 12, "maxIter": 50, "seed": 42},
      "inputs": [{"from": "n2", "port": "train"}],
      "compute": {"engine": "MaxCompute", "instanceQuota": "general"}
    }
  ],
  "edges": [
    {"from": "n1.out", "to": "n2.in"},
    {"from": "n2.train", "to": "n3.in"}
  ],
  "schedule": {"cron": "0 3 * * 1", "retries": 2, "timeoutMin": 60}
}
```

这里有三点至关重要。首先，每个节点都明确指定了 `compute.engine`：SQL 类组件走 `MaxCompute`，Python / PyAlink 节点走 `DLC`，流处理则用 `Flink`。Designer 并不会智能选择最便宜的执行引擎——它完全依据组件类型决定，而任何自定义 Python 节点默认都会启动一个 DLC Pod，无论上游 MaxCompute 步骤耗时 30 秒还是 30 分钟，这笔费用都算在你头上。其次，这份 JSON 是 *幂等* 且 *可 diff* 的——我会把它和依赖的 SQL 一起存入 Git，代码审查时就像 Review Terraform Plan 一样直观。最后，同一份 JSON 也能通过 SDK 提交（`pai.designer.submit_workflow(json_path)`），这正是我在 CI 中自动化运行流水线的方式，全程无需打开控制台。

更妙的是隐藏的升级路径：一旦你手握这份 JSON，“我想把它转成代码”就不再是推倒重来。你可以从 Python DAG 动态生成它，进行版本管理、静态检查，甚至通过 Mock 输入对单个节点做单元测试。画布本质上只是这份 JSON 的可视化前端，而非独立工具。

## 什么时候 Designer 比手写代码快，什么时候恰恰相反

我见过团队花一周时间开发一套自定义 DLC + PySpark 流水线，结果同样的任务在 Designer 画布上 90 分钟就能搞定；也见过资深工程师硬要在 Designer 里实现实时特征工程，结果和平台死磕两周。现实远比宣传材料更极端：

| 判断信号 | Designer 更优 | 自定义代码更优 |
|---|---|---|
| 源数据是否在 MaxCompute？ | 是（避免跨系统拉取开销） | 否（浪费 Designer 最强集成优势） |
| 流水线是否以表格 ETL + 传统 ML 为主？ | 是 | 否（涉及 LLM、强化学习或自定义 CUDA → 选 DLC） |
| 负责人是否为数据分析师 / BI 工程师？ | 是 | 否（工程师通常反感图形界面） |
| 是否只需 Cron + 自动重试？ | 是 | 否（需自定义触发逻辑或多阶段审批 → 用 Airflow / DataWorks 代码） |
| 单节点自定义 Python 代码是否超过 100 行？ | 否（画布沦为 `.py` 文件的包装器） | 是（直接写脚本更高效） |
| 是否需要流式处理或亚秒级 SLA？ | 否（Designer 本质是批处理） | 是 |

举个生产实例：我曾将一个推荐特征流水线迁移到 Designer，它在 MaxCompute Spot 实例上仅用 22 分钟，日均成本约 6 元；而此前基于 DLC 的 PySpark 版本在 4 节点集群上跑了 41 分钟，成本高达 38 元。成本相差六倍，关键在于数据从未离开 MaxCompute。反向案例也有：一次尝试用 Designer 微调 Qwen LoRA，因需自定义训练循环，最终在一个 Python 节点里塞进 600 行胶水代码，调试极其痛苦——画布把完整的 stack trace 藏在小小的 `view logs` 链接背后。我们下午就重写为标准 DLC 任务，团队立刻轻松不少。

最大的误区，就是把 Designer 当成“万能解”（产品经理常这么想）或“完全无用”（资深工程师容易这么认为）。它其实是一个高度特化的工具——专为“MaxCompute 规模下的表格类机器学习，且负责人非程序员”的场景而生。在这个范畴内，它确实是最佳选择。

## Designer 藏在背后的 DLC 后端

这点我花了比预期更久才搞明白：当你在 Designer 画布上点击包含 Python 节点的 *Run* 按钮时，平台其实在后台悄悄启动了一个 DLC 任务——和你在第 3 篇通过 SDK 提交的任务完全一致。但画布对此毫无提示，你得手动钻进 *Operations → Recent runs → View backend job* 才能找到真相。这意味着：

- **Python 节点按 DLC 计费**，哪怕你的画布看起来“全是 MaxCompute”。一个不小心加了 `PyAlink` 步骤，在单节点 `ecs.gn6i-c8g1.2xlarge` 上运行，每小时就要烧掉 4–5 元。
- **容器镜像版本至关重要。** Designer 会将 Python 节点锁定到特定镜像（通常是 `pai-designer:latest` 或工作空间中指定的版本）。如果你的代码依赖 `vllm==0.8.2`，但镜像只带 `0.6.0`，任务会在 import 阶段失败，而错误堆栈只会出现在底层 DLC 日志中，画布界面完全不显示。我就曾为此浪费半天时间。
- **配额会在产品间互相挤占。** 如果 DLC 配额紧张，一个需要 Python Pod 的 Designer 运行可能会抢走真正训练任务的资源，反之亦然——因为它们共享同一个工作空间的配额池。解决办法是：在工作空间中单独设置一个“designer-only”资源配额，上限设为 4 CPU + 16 GB，防止失控的画布吃光你的训练预算。

一旦你意识到 Designer 本质上只是个生成 MaxCompute SQL + DLC Pod 提交请求的 UI，整个产品就不再神秘——必要时，你也能干净利落地绕过它。

## 接下来写什么

第 2 篇将深入 **PAI-DSW**：如何选择合适的 GPU 实例、理解镜像目录、挂载 OSS-FUSE，并提供一个可直接运行的 MNIST Notebook（就是官方 Quick Start 里的那个）。第 3 篇聚焦 **PAI-DLC** 分布式训练，展示一个带 AIMaster 容错能力的真实多 GPU 任务。第 4 篇剖析 **PAI-EAS** 模型服务，重点揭露那个多次坑我的冷启动陷阱。第 5 篇则坦诚对比 **Designer 与 Model Gallery**，帮你判断在“I just want to ship something”的场景下该选谁。

如果只能读一篇，选第 4 篇——EAS 是生产环境中花钱最多的地方，也是官方文档最薄弱的一环。
