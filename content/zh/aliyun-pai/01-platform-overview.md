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
只要你的团队在阿里云上训模型或者做服务，迟早得进 PAI 控制台。PAI 是把大伞，底下坐着的才是干活的主力——笔记本产品、分布式训练服务、模型推理服务，再加上顶层的几个 GUI 和快速部署层。在一个 AI 营销平台上跑了十八个月真实的 LLM 负载后，这篇系列文章就是我希望在上线第一个 endpoint 之前能拿到的实战指南。

第一篇先摸清路子。代码会故意少放点——深潜内容在第 2 到第 5 篇。这里的目的是，等到后面我提到"DLC job"或者"EAS endpoint"时，你心里已经清楚它们属于哪个筐了。

![Aliyun PAI (1): Platform Overview and the Product Family Map — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_1.png)

## PAI 是什么，不是什么

官方文档里写，**Platform for AI (PAI)** 是“阿里云覆盖全生命周期的 AI 开发平台：数据标注、模型开发、训练和部署”。控制台 `pai.console.aliyun.com` 是个入口，但 PAI 本身是一*家族*相关产品，它们共用一套账号模型、基于 OSS 的存储层，以及同一个 Python SDK。

对我最管用的心智模型是这样的：

- **PAI 是店面。**
- **DSW, DLC, EAS, Designer, Model Gallery** 是店里的操作台。
- **ECS, OSS, NAS, CPFS** 才是算力和字节真正存放的地方。PAI 只是代表你去编排它们。

官方的“服务架构”主题把它拆解成了四层栈：

![PAI four-layer service architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig1_pai_4layer_architecture.png)

从下往上读。基础设施层是硅片——CPU、GPU、RDMA fabric，底下撑着 ACK Kubernetes。在这之上，*灵骏*（Lingjun）给你超高密度的 AI 算力，*通用算力*给你日常的 ECS-backed GPU 资源池。平台与工具层是你每天待的地方：PyTorch / Megatron / DeepSpeed，加上 PAI 的优化玩具（TorchAcc, BladeLLM, EasyCkpt, AIMaster），还有可见的产品（DSW, DLC, EAS, Designer, FeatureStore, iTAG）。应用层是 PAI 怎么插进阿里剩下的 MaaS 世界（ModelScope, Bailian/DashScope, Model Studio）。业务层则是给行业用例看的营销幻灯片。

用 PAI 而不是裸 ECS 的理由在于，它预置好了 CUDA / PyTorch 镜像，帮你挂载 OSS bucket，给你指标仪表盘，而且按秒计费。

## 你实际会碰到的五个子产品

摸爬滚打一年半的生产环境工作后，我只为下面这些付过钱，直接摘自官方的“核心组件”表：

| 组件 | 官方定义 | 什么时候用它 |
|---|---|---|
| **DSW** (Data Science Workshop) | 基于云的 IDE，含 Jupyter / VSCode / 终端，预配 PyTorch 和 TensorFlow 镜像，GPU 实例 | 交互式开发、调试、小规模训练 |
| **DLC** (Deep Learning Containers) | 基于 Kubernetes 的训练，支持 Megatron, DeepSpeed, PyTorch, TF, Slurm, Ray, MPI, XGBoost — 无需集群 setup | 多 GPU / 多节点 SFT、预训练、大规模评测 |
| **EAS** (Elastic Algorithm Service) | 在线推理，支持自动扩缩容、灰度发布、流量拆分、镜像 | 生产环境推理 endpoint |
| **Designer** | 140+ 内置算法组件，拖拽式 pipeline，可导出 JSON，可在 DataWorks 调度 | 交给非编码人员的 ETL → 训练 → 评测流 |
| **Model Gallery** | 封装 DLC + EAS 实现零代码部署和微调目录中的开源模型 | 10 分钟内评估一个 Qwen / DeepSeek / Llama 模型 |

还有 **iTAG**（数据标注）、**PAI-Lingjun** 针对超大规模集群、**PAI-Blade / BladeLLM** 用于推理优化，以及 **FeatureStore**，但除非你要搞 >1000 GPU 的预训练或者构建推荐系统，第一天可以先忽略它们。

产品划分 cleanly 映射到 ML 生命周期：

![PAI sub-products on the ML lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig2_pai_subproducts_lifecycle.png)

Designer 和 Model Gallery 是正交的——它们坐在顶层，生成的 job 最终跑在同样的 DLC / EAS 底座上。

## PAI 跟 ECS 和 OSS 的关系

这点最容易坑到纯云 VM 背景的同学。三条铁律：

1. **PAI 从不拥有你的数据。** 数据集、checkpoint 和模型 artifact 都存在 **OSS** 里（或者 **NAS** 用于 POSIX 语义，或者 **CPFS** 用于 HPC 风格的吞吐）。当 DSW 或 DLC 实例挂掉时，你没写进 OSS 的东西就没了。有个“系统盘”，但把它当成 `/tmp` 对待。
2. **PAI 拥有算力。** 你不要自己去 provision GPU ECS 实例给 PAI  workload。PAI 管理一个资源池，你申请 `1 * ecs.gn7i-c8g1.2xlarge`，然后按分配秒数计费。
3. **PAI 共用你的账号但用自己的 RAM 角色。** 当你授权 PAI 访问 OSS 时，你是附加了一个服务关联角色（`AliyunPAIAccessingOSSRole`），这样 PAI 的算力就能读你的 bucket 而无需长活的 AK 对。别跳过这步——不然你的 DLC job 会在 `data_loader` 阶段报 403 失败。

> **实战建议：** 最常见的"PAI 坏了”工单其实是 PAI 和 OSS 之间的权限问题。在调试训练脚本之前，先在 DSW 终端里跑一下 `oss ls oss://your-bucket/`。如果这步失败，修角色，别修代码。

## 账号、地域、工作空间

起步得按顺序搞定三件事：

1. **一个 aliyun.com 账号** 完成实名认证——任何 GPU 资源都必需。国际账号在大多数地域能用，但杭州、上海和北京的 GPU 库存最足。
2. **一个地域。** 选一个然后别变。PAI 资源、OSS bucket 和 ECS GPU 都是地域 scoped 的，跨地域流量既花钱又加延迟。国内生产我默认 `cn-shanghai`；国际选 `ap-southeast-1`（新加坡）。
3. **一个工作空间。** 按文档说法，工作空间是 PAI 的租户基本单元——它 holding 配额、数据集、模型注册表和 IAM 绑定。你几乎总需要至少两个：一个 `dev` 工作空间让人类在 DSW 里折腾，一个 `prod` 工作空间放 DLC job 和 EAS endpoint。跨工作空间权限配置挺麻烦，但第一次有实习生误重启了服务 endpoint 时，你就知道隔离的价值了。

![PAI tenancy: account, region, workspace](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/fig3_pai_account_workspace.png)

## 两条路：控制台 vs SDK

跟百炼一样，PAI 也给你两种方式做所有事。控制台适合一次性操作和检查状态；SDK 才是你放进 CI 里发货的东西。

Python SDK 就一个包：

```bash
pip install alibabacloud-pai-python-sdk
```

来个"hello PAI"——列出你的工作空间：

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

如果这能打印出至少一个工作空间 ID，说明你的账号、地域和凭证连线正确，可以进入第 2 篇了。

> **实战建议：** SDK 工作用子账号配 scoped RAM policy。千万别用根账号 access key——而且如果你的 AK 对出现在任何 git 历史里，立刻轮换。阿里云的漏钥检测还行，但还没到 GitHub 那种速度。

## 一段话讲清计费模型

文档列了五种计费方式：**按量付费**、**包年包月**（预付）、**资源包**（DSW 预付配额）、**节省计划**（承诺折扣），以及**按推理时长付费**（EAS serverless — 无空闲 replica 成本）。DSW 按实例运行秒数计费，DLC 按秒计费且有单独的 spot/preemptible GPU 配额，如果你的 job 能 checkpoint 大概能便宜 30-50%，EAS 按 replica 秒数加上每百万请求的小额收费，自动扩缩容的最小 replica 数主导成本。Designer 和 Model Gallery 本身不收费——它们 spawn 的 DLC/EAS 资源正常计费。新账号有个小的免费额度，足够跟完整个系列。

## Designer 工作流底层到底长什么样

![Aliyun PAI (1): Platform Overview and the Product Family Map — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/01-platform-overview/illustration_2.png)

Designer 界面上像搭乐高，但导出的是一个平台可以重放的扁平 JSON 文档。第一次我得调试卡住的 pipeline 时，我从画布导出了 JSON（`...` 菜单 → *Export workflow*），它看起来就不那么魔法了。结构大致如下：

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

这里有三点很重要。第一，每个节点都有明确的 `compute.engine` —— SQL 形状的组件用 `MaxCompute`，Python / PyAlink 用 `DLC`，流式处理用 `Flink`。Designer 不会 magically 挑最便宜的那个；它基于组件类型挑选，任何自定义 Python 节点默认跑在一个 DLC pod 上，不管上游 MaxCompute 步骤花了 30 秒还是 30 分钟，这都算你的钱。第二，JSON 是 *幂等* 且 *可 diff* 的——我把我们的存在 git 里，跟它依赖的 SQL 放一起，代码 review 变更的方式跟我 review Terraform plan 一样。第三，同样的 JSON 可以通过 SDK 提交（`pai.designer.submit_workflow(json_path)`），这就是我怎么在 CI 里跑 pipeline 而不用开控制台。

隐藏的升级路径：一旦你有了 JSON，“我想把这个变成代码”就不再是重写。你可以从 Python DAG 生成它，版本化，lint 它，甚至通过 mock 输入单元测试单个节点。画布只是在这个文件之上的 UI，不是独立的工具。
## 什么时候 Designer 比手写代码快，什么时候恰恰相反

我见过团队为了一个本来 90 分钟能在 Designer 画布上搞定的任务，花了一周去写 custom DLC + PySpark pipeline；也见过反过来的情况——资深工程师非要在 Designer 里做实时特征工程，跟平台死磕了两周。实际情况比市场宣传的要极端得多：

| Signal | Designer wins | Custom code wins |
|---|---|---|
| 源数据在 MaxCompute 里？ | 是（省去跨系统拉取） | 否（浪费了 Designer 最强的绑定能力） |
| 流水线主要是表格 ETL + 经典 ML？ | 是 | 否（LLM, RL, 自定义 CUDA → DLC） |
| 负责人是数据分析师 / BI 工程师？ | 是 | 否（工程师会讨厌画布） |
| Cron + 重试够用了？ | 是 | 否（自定义触发、多阶段审批 → Airflow / DataWorks 代码） |
| 单个节点自定义 Python > 100 行？ | 否（画布变成了 `.py` 的包装器） | 是（直接写脚本） |
| 需要流式 / 亚秒级 SLA？ | 否（Designer 是批处理形状） | 是 |

说个生产环境的具体数字：我之前把一个推荐特征 pipeline 迁移*到* Designer，在 MaxCompute spot 实例上跑 22 min，每天成本 ~6 RMB；之前 PySpark-on-DLC 版本在 4-node cluster 上要跑 41 min，成本 ~38 RMB。便宜了六倍，因为数据压根不用出 MaxCompute。反过来也有案例：有个“用 Designer 跑 Qwen LoRA"的尝试，因为需要自定义 training loop，最后在一个 Python 节点里塞了 600 lines of glue code，调试起来痛不欲生，因为画布把 stack traces 藏在一个小小的 `view logs` 链接后面。我们下午把它重写成正常的 DLC 提交，团队反而更开心了。

这里的坑就是把 Designer 当成*万能药*（PM 常这么想）或者*垃圾*（资深工程师常这么想）。它就是为解决特定类问题而生的工具——MaxCompute 规模下的表格 ML，且负责人不怎么写代码——在这个范畴里，它确实是最佳选择。

## Designer 藏在背后的 DLC 后端

这点我搞明白花的时间比预期长。在 Designer 画布上点击带 Python 节点的 *Run*，平台背后其实启动了一个 DLC 任务——跟第 3 章里通过 SDK 提交看到的一样。画布上这点展示得并不清楚；你得深挖 *Operations → Recent runs → View backend job* 才能找到。影响如下：

- **Python 节点按 DLC 收费**，哪怕你的画布看起来像是“纯 MaxCompute"。一个不小心加了 `PyAlink` 步骤，跑在 1-node `ecs.gn6i-c8g1.2xlarge` 上，画布运行期间每小时大概烧 4-5 RMB。
- **容器镜像版本很关键。** Designer 把 Python 节点绑定在特定镜像上（通常是 `pai-designer:latest` 或者工作空间设置里的版本标签）。如果你自定义 Python 代码里 import 了 `vllm==0.8.2` 但镜像自带 `0.6.0`，运行会在 import 阶段失败，stack trace 不会显示在画布上——只在底层的 DLC log 里。这点我曾经坑过半天。
- **配额会在产品间泄露。** 如果 DLC 配额紧张，一个需要 Python pod 的 Designer 运行可能会抢走真正 DLC 训练任务的资源，反之亦然。它们共享同一个 workspace quota pool。解决办法：在工作空间里设一个单独的"designer-only"资源配额，上限设个 4 CPU + 16 GB 就行，防止画布 runaway 吃光你的训练预算。

一旦你知道 Designer 只是个用来 emit MaxCompute SQL + DLC pod 提交的 UI，整个产品就没那么神秘了——需要的时候你也可以干净利落地绕过它。

## 接下来写什么

第 2 篇是 **PAI-DSW** 端到端实战：怎么选 GPU instance，image catalog，OSS-FUSE 挂载，还有一个能跑的 MNIST notebook（就是官方 Quick Start 里那个）。第 3 篇是 **PAI-DLC** 分布式训练——真正的多 GPU 任务，带 AIMaster 容错。第 4 篇是 **PAI-EAS** 模型服务，包括那个坑过我好几次的 cold-start 陷阱。第 5 篇是 **Designer vs Model Gallery** 的诚实对比，针对"I just want to ship something"的场景。

如果只读一篇，读第 4 篇——EAS 是生产环境花钱最多的地方，也是文档最薄的地方。