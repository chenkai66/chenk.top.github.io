---
title: "阿里云 PAI 实战（一）：平台总览与产品家族地图"
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
如果你的团队在阿里云上训练或部署模型，那么迟早会用到 PAI 控制台。PAI 是一个综合性平台，下面包含了几个核心的工作组件：一个 Notebook 服务、一个分布式训练服务、一个模型推理服务，还有一些基于 GUI 的快速部署工具。我们在某个 AI 营销平台上运行了大约一年半的真实 LLM 工作负载后，我深刻体会到，这个系列文章正是我希望在第一次上线 EAS 之前就有人能递到我手里的指南。

这篇是开篇，主要讲整体布局，代码内容很少，深度技术细节会放在第 2 到第 5 篇中展开。这篇文章的目标是让你对整个体系有一个清晰的认识，这样当后面提到“DLC 任务”或者“EAS 端点”时，你能立刻明白它们在整个架构中的位置。

![阿里云 PAI 实战（一）：平台总览与产品家族地图 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_1.jpg)
## PAI 是什么，又不是什么

根据官方文档的描述，**人工智能平台 PAI** 是“阿里云面向 AI 全生命周期的一站式开发平台，覆盖数据标注、模型开发、训练到部署的全流程”。虽然可以通过 `pai.console.aliyun.com` 进入控制台，但 PAI 并不仅仅是一个单一的产品，而是一个由多个相关服务组成的**家族**，它们共享统一的账号体系、基于 OSS 的存储层，以及一个通用的 Python SDK。

我个人觉得最容易理解的方式是这样类比：

- **PAI 是一家大型商场。**
- **DSW、DLC、EAS、Designer、Model Gallery** 是商场里的不同工作间。
- **ECS、OSS、NAS、CPFS** 则是真正存放硬件和数据的地方，而 PAI 的作用就是帮你把这些资源整合起来，按需调配。

官方文档中的“服务架构”部分将 PAI 的整体设计描绘为一个四层结构：

![PAI 四层服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig1_pai_4layer_architecture.png)

从下往上来看，最底层是基础设施层，也就是硬件核心——CPU、GPU、RDMA 高速网络，背后依托的是 ACK Kubernetes 集群。在这之上，*灵骏* 提供了超高密度的 AI 计算能力，而通用计算则提供了基于 ECS 的常规 GPU 资源池。再往上是平台与工具层，这是工程师每天打交道的地方：PyTorch、Megatron、DeepSpeed 等主流框架，加上 PAI 提供的优化工具（如 TorchAcc、BladeLLM、EasyCkpt、AIMaster），以及直接可见的产品模块（如 DSW、DLC、EAS、Designer、FeatureStore、iTAG）。应用层则是 PAI 与阿里巴巴 MaaS 生态对接的桥梁，包括 ModelScope、百炼/DashScope 和 Model Studio。最顶层的业务层，则是用来展示行业解决方案的营销材料。

那么问题来了，为什么不用裸 ECS 呢？因为 PAI 已经帮你做好了 CUDA 和 PyTorch 的镜像预装，自动挂载了 OSS 存储桶，还提供了一个现成的监控面板，并且支持按秒计费，省去了很多繁琐的配置工作。
## 实际工作中常用的五个子产品

在生产环境中摸爬滚打了一年半，我真正用到并付费的核心组件，其实只有以下这些，它们直接来自官方的“核心组件”表格：

| 组件 | 官方描述 | 使用场景 |
|---|---|---|
| **DSW**（数据科学工作台） | 云端集成开发环境，支持 Jupyter、VSCode 和终端，预装 PyTorch 和 TensorFlow 镜像，提供 GPU 实例 | 交互式开发、调试、小规模模型训练 |
| **DLC**（深度学习容器） | 基于 Kubernetes 的分布式训练框架，支持 Megatron、DeepSpeed、PyTorch、TensorFlow、Slurm、Ray、MPI、XGBoost，无需手动搭建集群 | 多 GPU 或多节点的 SFT（监督微调）、预训练、大规模评估 |
| **EAS**（弹性算法服务） | 提供自动扩缩容、灰度发布、流量分流和镜像功能的在线推理服务 | 生产环境中的推理服务端点 |
| **Designer**（可视化建模工具） | 内置 140+ 种算法组件，支持拖拽式构建流水线，可导出 JSON 文件，并能在 DataWorks 中调度 | 非开发人员也能轻松上手的 ETL → 训练 → 评估流程 |
| **Model Gallery**（模型广场） | 封装了 DLC 和 EAS，支持零代码部署和微调开源模型 | 快速体验 Qwen、DeepSeek、Llama 等开源模型，10 分钟完成评估 |

除此之外，还有一些其他组件，比如 **iTAG**（数据标注工具）、**PAI-Lingjun**（超大规模集群管理）、**PAI-Blade / BladeLLM**（推理优化工具）以及 **FeatureStore**（特征存储）。不过，除非你是在进行上千张 GPU 的大规模预训练，或者构建推荐系统，这些组件在初期完全可以先放一放。

这些子产品的划分与机器学习生命周期高度契合：

![PAI 子产品与 ML 生命周期](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig2_pai_subproducts_lifecycle.png)

Designer 和 Model Gallery 是两个独立的高层工具，它们最终生成的任务依然运行在底层的 DLC 和 EAS 平台上。
## PAI 与 ECS 和 OSS 的关系

如果你是从传统云服务器（ECS）背景转过来的，这里可能会让你有点困惑。以下是三条关键原则，帮你理清 PAI 和 ECS、OSS 的关系：

1. **PAI 不会管理你的数据。** 所有的数据集、训练过程中的 checkpoint 以及模型文件都存储在 **OSS** 中（如果需要 POSIX 兼容性，可以选择 **NAS**；如果追求高性能计算的吞吐量，则可以用 **CPFS**）。一旦 DSW 或 DLC 实例被释放，所有没有保存到 OSS 的内容都会丢失。虽然实例有一个“系统盘”，但你可以把它当作临时目录 `/tmp` 来使用。
   
2. **PAI 负责计算资源的调度。** 你不需要自己去创建 GPU 类型的 ECS 实例来运行 PAI 的任务。PAI 会维护一个资源池，你只需要告诉它需要多少算力，比如申请 `1 * ecs.gn7i-c8g1.2xlarge`，然后按实际使用的秒数计费即可。

3. **PAI 使用你的账号，但依赖自己的 RAM 角色。** 当你需要让 PAI 访问 OSS 时，实际上是通过绑定一个服务关联角色 `AliyunPAIAccessingOSSRole` 来实现的。这样，PAI 的计算节点就可以直接读取你的 OSS 存储桶，而无需长期有效的 AccessKey（AK）。这一步非常重要——如果跳过，你的 DLC 任务很可能会在加载数据（`data_loader`）时因权限不足（403 错误）而失败。

> **实战建议：** 在日常支持中，最常见的“PAI 出问题了”的工单，往往是因为 PAI 和 OSS 之间的权限配置不正确。在排查训练代码之前，建议先在 DSW 的终端里执行 `oss ls oss://your-bucket/` 命令。如果这个命令报错，优先检查并修复 RAM 角色的配置，而不是急着改代码。
## 账号、区域与工作空间

在开始之前，你需要按顺序准备好以下三样东西：

1. **一个 aliyun.com 账号**，并且完成实名认证（实名认证是使用 GPU 资源的必要条件）。国际站账号在大多数区域都可以正常使用，但如果你需要更稳定的 GPU 库存，推荐选择杭州、上海或北京这些区域。
2. **选定一个区域并固定下来。** PAI 资源、OSS 存储桶以及 ECS 上的 GPU 都是基于区域划分的，跨区域传输不仅会产生额外费用，还会增加延迟。对于国内生产环境，我通常选择 `cn-shanghai`；如果是国际环境，则优先考虑 `ap-southeast-1`（新加坡）。
3. **创建工作空间。** 根据官方文档的描述，工作空间是 PAI 的核心租户单元，用于管理配额、数据集、模型注册表以及 IAM 权限绑定等内容。一般建议至少创建两个工作空间：一个是供开发人员调试使用的 `dev` 工作空间，主要用于 DSW 交互式开发；另一个是用于部署 DLC 任务和 EAS 端点的 `prod` 工作空间。虽然跨工作空间的权限配置稍显复杂，但当你第一次遇到实习生误操作重启了生产环境的服务端点时，就会发现这种隔离设计是多么值得。

![PAI 租户模型：账号 - 区域 - 工作空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig3_pai_account_workspace.png)
## 两种方式：控制台与 SDK

和百炼类似，PAI 提供了两种操作路径。控制台适合用来查看状态或执行一次性任务，而 SDK 则是集成到 CI/CD 流程中的首选工具。

Python SDK 只需安装一个包：

```bash
pip install alibabacloud-pai-python-sdk
```

一个简单的“Hello PAI”示例——列出你的所有工作空间：

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

如果能成功打印出至少一个工作空间的 ID，说明你的账号、区域和凭证配置都没问题，可以继续阅读下一篇内容。

> **实战建议：** 使用子账号并绑定精细化的 RAM 策略来操作 SDK，切勿使用主账号的 AccessKey。如果发现 AccessKey 被意外提交到 git 历史记录中，请立即轮换密钥。阿里云的密钥泄露检测机制虽然可用，但响应速度还达不到 GitHub 那样的级别。
## 计费模式一句话概括

文档中列出了五种计费方式：**按量付费**、**包年包月**（预付费）、**资源包**（DSW 预付费额度）、**节省计划**（承诺使用以获取折扣）以及**按推理时长计费**（EAS Serverless，无需为空闲副本买单）。DSW 按实例运行时间秒级计费；DLC 同样按秒计费，并且支持 Spot/抢占式 GPU 实例，价格大约便宜 30%-50%，但前提是你的任务能够支持断点续训；EAS 则按照副本运行时间秒级计费，外加极低的每百万次请求费用——不过由于自动扩缩容机制的存在，“最小副本数”的配置往往是成本的主要来源。Designer 和 Model Gallery 本身不收费，但它们调用的 DLC 或 EAS 资源会正常计费。新用户账号享有数百元的免费额度，足够完整跑完本系列教程。
## Designer 工作流背后的真正模样

![阿里云 PAI 实战（一）：平台总览与产品家族地图 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_2.jpg)

在 Designer 的画布上搭建工作流，感觉就像拼乐高积木一样直观。然而，当你导出工作流时，会发现它实际上是一个扁平的 JSON 文件，平台通过这个文件来回放整个任务流程。记得第一次排查一个卡住的流水线时，我从画布的“…”菜单中选择 *导出工作流*，拿到了对应的 JSON 文件。那一刻，Designer 的“魔法”面纱被揭开，露出了它的本质。以下是 JSON 文件的大致结构：

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

这里有几个关键点需要特别注意。

第一，每个节点都明确指定了 `compute.engine`，即计算引擎的选择。SQL 类型的组件默认使用 `MaxCompute`，Python 或 PyAlink 类型的组件则运行在 `DLC` 上，而流式任务则依赖 `Flink`。需要注意的是，Designer 并不会自动选择最便宜的计算引擎，而是根据组件类型来决定。例如，任何自定义的 Python 节点都会默认分配一个 DLC Pod，无论上游的 MaxCompute 步骤耗时 30 秒还是 30 分钟，DLC Pod 的计费时间都不会改变。

第二，这份 JSON 文件具备**幂等性**和**可对比性**。我会将它与相关的 SQL 脚本一起存入 git 仓库，并像审查 Terraform 计划一样，对每次改动进行代码审查。这种方式不仅提高了工作流的透明度，还让团队协作更加高效。

第三，同样的 JSON 文件可以通过 SDK 提交到平台执行。具体来说，调用 `pai.designer.submit_workflow(json_path)` 就可以完成任务提交。这也是我在 CI 环境中运行流水线的方式——完全不需要打开控制台。

隐藏的升级路径在于：一旦你掌握了 JSON 文件，迁移工作流到代码中就不再是痛苦的重写过程。你可以通过 Python DAG 自动生成 JSON 文件，对其进行版本管理、代码检查，甚至通过模拟输入对单个节点进行单元测试。换句话说，画布只是这份 JSON 文件的可视化界面，而不是一个独立的工具。
## Designer 比写代码快的情况，以及不适用的场景

我见过有些团队花了一周时间搭建自定义的 DLC + PySpark 流水线，结果做的事情其实用 Designer 画布 90 分钟就能搞定。我也见过相反的情况——一位资深工程师试图在 Designer 中实现实时特征工程，结果和平台较劲了两周才勉强完成。这种分界线比市场宣传中描绘的要更加清晰：

| 特征 | Designer 更适合 | 自定义代码更合适 |
|---|---|---|
| 数据源是否在 MaxCompute？ | 是（避免跨系统数据拉取） | 否（Designer 的核心优势无法发挥） |
| 流水线主要是表格 ETL 和经典机器学习？ | 是 | 否（涉及 LLM、RL 或自定义 CUDA 的任务更适合 DLC） |
| 负责人是数据分析师或 BI 工程师？ | 是 | 否（工程师通常不喜欢用画布操作） |
| 定时任务 + 重试机制够用吗？ | 是 | 否（需要自定义触发器或多阶段审批 → 使用 Airflow / DataWorks 编写代码） |
| 单节点自定义 Python 脚本超过 100 行？ | 否（画布变成了 `.py` 文件的包装器） | 是（直接编写脚本更高效） |
| 是否需要流式处理或亚秒级 SLA？ | 否（Designer 更适合批处理任务） | 是 |

举个实际的例子：我曾经把一条推荐特征流水线**迁移到** Designer 上运行，使用 MaxCompute 抢占式资源，每天运行一次，耗时 22 分钟，成本约 6 元；而之前的 PySpark + DLC 实现方式，在 4 节点集群上运行需要 41 分钟，每天的成本约为 38 元。之所以能便宜 6 倍，原因很简单——数据完全不用离开 MaxCompute。当然，反过来的迁移也发生过：有一次我们尝试用 Designer 来跑 Qwen LoRA，但因为需要自定义训练循环，最终变成了在一个 Python 节点里塞了 600 行胶水代码，调试起来痛苦不堪，因为画布会把堆栈信息隐藏在一个不起眼的 `查看日志` 链接后面。后来我们花了一个下午把它改成了普通的 DLC 提交，团队的心情顿时好了很多。

真正的陷阱在于，把 Designer 当成**永远正确**（产品经理常这么想）或者**永远错误**（资深工程师常这么认为）。它是一个为特定问题设计的工具——适用于 MaxCompute 规模的表格机器学习，并且由非开发人员负责的场景。在这种场景下，它确实是最佳选择。
## Designer 背后隐藏的 DLC

这件事我花了比预期更长的时间才搞明白。在 Designer 画布上点击 *运行*，如果包含一个 Python 节点，平台实际上会在后台启动一个 DLC 任务——和你在第三章通过 SDK 提交的任务完全一样。然而，画布界面上并不会明确告诉你这一点，你需要进入 *运维 → 最近运行 → 查看后端任务* 才能找到相关线索。这背后有几个值得注意的地方：

- **Python 节点按 DLC 价格计费**，即使你的画布看起来像是“纯 MaxCompute”的工作流。比如，一个单独的 `PyAlink` 步骤运行在一个 `ecs.gn6i-c8g1.2xlarge` 实例上时，每小时的费用大约是 4-5 元人民币，只要画布在运行就会持续计费。
- **容器镜像版本至关重要。** Designer 的 Python 节点会绑定到特定的镜像（通常是 `pai-designer:latest` 或者工作空间设置中的某个版本标签）。如果你的自定义代码中 `import vllm==0.8.2`，但镜像自带的是 `0.6.0`，那么运行时会在导入模块时失败。而这个错误不会直接显示在画布界面上，只能通过底层的 DLC 日志排查。我就曾经因为这个问题浪费了半天时间。
- **资源配额在产品间是共享的。** 如果你的 DLC 配额本来就紧张，Designer 运行时需要启动 Python pod 可能会导致真正的 DLC 训练任务资源不足，反之亦然。这是因为它们共享同一个工作空间的配额池。解决办法是：为 Designer 单独配置一个资源配额，限制在比如 4 核 CPU 和 16 GB 内存，这样可以避免一个失控的画布耗尽你的训练预算。

一旦你意识到 Designer 不过是一个用来生成 MaxCompute SQL 和提交 DLC pod 的用户界面，整个产品的运作逻辑就变得清晰了——在必要时，你也可以轻松绕过它来实现更灵活的操作。
## 接下来的内容

第二篇将深入探讨 **PAI-DSW** 的端到端实践：如何选择合适的 GPU 实例、镜像目录的使用方法、OSS-FUSE 挂载技巧，以及如何跑通官方 Quick Start 中提供的 MNIST 示例。第三篇聚焦于 **PAI-DLC** 分布式训练——通过一个真实的多 GPU 任务，展示 AIMaster 的容错能力。第四篇则围绕 **PAI-EAS** 模型服务展开，特别会提到那个让我踩坑无数的冷启动问题。第五篇是对 **Designer 和 Model Gallery** 的坦诚对比，帮助你判断在“我只想快速交付点东西”的场景下，哪种工具更适合。

如果只选一篇来读，建议直接看第四篇——EAS 是生产环境中花钱最多的地方，同时也是文档支持最薄弱的部分。