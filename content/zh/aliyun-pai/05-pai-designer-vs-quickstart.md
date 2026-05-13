---
title: "阿里云 PAI（五）：Designer vs Model Gallery"
date: 2026-03-09 09:00:00
tags:
  - Aliyun PAI
  - PAI-Designer
  - Model Gallery
  - Low-Code
categories: 阿里云 PAI
lang: zh
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 5
description: "PAI-Designer 处理表格 ML 流水线，Model Gallery 一键部署/微调开源模型。一份诚实的决策矩阵：什么时候该跳过 SDK、让 GUI 帮你交付。"
disableNunjucks: true
translationKey: "aliyun-pai-5"
---
前四篇文章介绍了底层原语——DSW、DLC、EAS——以及如何用 Python 对它们进行编排；本文则聚焦于两个 GUI 产品：**PAI-Designer**（拖拽式表格 pipeline 构建）和 **Model Gallery**（零代码部署与微调开源模型）。这两个工具将底层原语封装为开箱即用的解决方案，专为不想写 Python 的用户设计。尽管资深工程师未必会首选它们，但在两种特定场景下，它们恰恰是最合适的选择。

![阿里云PAI (5)：Designer与Model Gallery — 当图形界面真正发挥作用时 — 视觉](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/illustration_1.png)

## Designer —— 拖拽式 Pipeline 编排工具

根据官方文档，Designer 通过工作流实现建模与模型调试，用户只需像搭积木一样拖拽不同组件，即可构建完整的 AI 开发流程。其核心功能包括 140 多个内置算法组件、支持导出 JSON、可接入 DataWorks 调度，以及允许在节点中嵌入自定义 SQL、Python 或 PyAlink 脚本。

它的优势在于：

- **MaxCompute 规模的表格机器学习**：Designer 与 MaxCompute 深度绑定。如果你的训练数据是 MaxCompute 上一张包含两亿行的分区表，那么 Designer 的 source / split / encode / train 等内置组件会直接在 MaxCompute 内部执行，而非通过网络传输到一个 Python Pod。你只需为 MaxCompute 计算付费，无需承担 GPU 实例空转等待数据的开销。
- **交接给非编码背景的分析师**：推荐、用户流失预测、风控等团队常有精通业务但不会写 Python 的专家。Designer 的画布对他们而言清晰可读，甚至能自主修改并长期维护。
- **内置模板丰富**：文档提供了商品推荐、新闻分类、金融风控、雾霾预测、心脏病预测、农业贷款、人口普查分析等开箱即用的案例。即便你最终替换了其中近半数的节点，这些模板依然是极佳的起点。
- **支持调度离线运行**：将工作流导出为 JSON，交给 DataWorks，即可获得带重试机制的日级或小时级定时任务。

它的局限也很明确：

- **不适用于任何 LLM 相关任务**：Designer 的强项在于特征工程与经典机器学习，而非编写自定义的 PyTorch 训练循环。
- **无法处理自定义 CUDA 工作、新颖损失函数等“算法即核心”的场景**。

对于表格类工作负载，我会优先使用 Designer 替代原本在 DLC 和 SQL 中构建的方案；其他所有情况，则仍选择在 DLC 中训练自定义模型。

## Model Gallery —— 零代码 MaaS 捷径

Model Gallery 是对 DLC + EAS 的封装，让非 MLOps 用户仅需几次点击，就能完成开源模型的微调与部署。官方文档指出，它封装了 PAI-DLC 和 PAI-EAS，为开源大语言模型的训练与部署提供了一套零代码解决方案。

Quick Start 示例完整走通了 Qwen3-0.6B 的端到端流程：

1. 在 Model Gallery 中搜索 “Qwen3-0.6B” → 点击 **Deploy**。
2. 全程采用默认配置：默认 GPU 类型、默认 vLLM 镜像等 → 点击 **OK**。
3. 约 5 分钟后，状态变为 `Running`。
4. 点击 **View Call Information** → 获取 `Internet Endpoint` 和 token。
5. 将其接入 Cherry Studio（或 Claude Code MCP，或使用 OpenAI 兼容 base URL 的 Python SDK），即可开始聊天。

在微调方面，文档提供了一个物流信息抽取的例子：上传一个 JSON 数据集，从下拉菜单中选择 LoRA 超参数，系统便会自动提交一个 DLC 任务。Quick Start 特别强调了 **蒸馏模式**——用大教师模型（Qwen3-235B）标注数据，再让小学生模型（Qwen3-0.6B）从中学习。这一模式值得牢记，据我所知，这是目前性价比最高的微调方法。

Gallery 的优势体现在：

- **10 分钟内快速评估新模型**：DeepSeek-V3 发布当天，我们团队在续杯咖啡的时间里就完成了部署并开始对话。如果还要手动配置 OSS Bucket、安全组和 SSL 证书，仅靠 `vllm serve` 几乎不可能做到这一点。
- **面向非技术干系人的演示**：点击 → 获取端点 → 接入 Cherry Studio 聊天 → 直接用于董事会汇报。
- **一键 LoRA 微调**：对于大多数领域适配任务，Gallery 自动选择的默认超参（学习率、轮数、LoRA rank）与最优值的差距通常在 5% 以内。

Gallery 的短板包括：

- **不支持自定义模型架构**：一旦你修改了模型代码，就必须回到 DSW + DLC。
- **无法满足严苛的延迟要求**：Gallery 的默认服务配置虽合理，但未经优化。若需 p99 延迟低于 100ms，你必须手动编写 EAS 部署配置，并调整批处理参数。
- **不支持隔离网络或跨区域部署**：Gallery 默认假设“在当前区域部署”。

![PAI-Designer 画布](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig1_designer_canvas.png)

## 何时选择哪个？

这是我长期验证有效的决策矩阵：

![Model Gallery 管道](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig2_modelgallery_pipeline.png)

总结一条经验法则：**在需求允许的前提下，尽可能从技术栈的高层开始**。许多团队在第一天就过度工程化——明明 Model Gallery 部署已足够，却非要构建一套自定义的 DLC + EAS 流水线。应优先优化“首 token 时间”，待真实流量和指标到位后，再针对性重构。

## 实战案例：Designer 如何胜过自定义代码

我曾遇到一个真实需求：市场部希望每周基于 MaxCompute 中一张 6000 万行的表运行用户分群。数据科学家的第一反应是：用 PySpark + scikit-learn 在 DLC 中跑任务，代码存 OSS，通过 SLS 回调 EventBridge 实现调度——预计耗时三天。

而 Designer 方案只需：source 节点 → sample → encode → KMeans → 写回 MaxCompute。导出 JSON 后交由 DataWorks 调度，全程仅用两小时（含向市场 PM 解释的会议时间）。输出结果完全一致，成本减半（无需 GPU Pod），维护量更是降至十分之一。

## 实战案例：Model Gallery 如何节省一周时间

我们需要验证 Qwen3-Coder 是否足以替代内部基于 `qwen-plus` 的代码审查机器人。在 Gallery 出现前，流程是：阅读 vLLM 文档、搭建 EAS 部署、编写 OpenAI 兼容桥接层、交付团队——至少两天。而使用 Gallery 后：搜索 → 部署 → 将端点接入现有客户端 → 午饭前搞定。我们得以聚焦核心问题（模型是否更好？），而非底层基建。

## 具体的决策树

![决策矩阵](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig3_decision_matrix.png)

上述矩阵是启发式规则；当队友问“这事该用 Designer / Gallery / DLC / EAS 哪个？”时，我实际遵循的是以下决策树：

![阿里云PAI (5)：Designer与Model Gallery — 当图形界面真正发挥作用时](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/illustration_2.png)

其中有几个不太明显的判断依据：

- **数据是否驻留在 MaxCompute 是选择 Designer 的最强信号**。关键不在于数据量大小或算法类型，而在于*数据所在位置*。用 `pyodps` 从 MaxCompute 拉取 5000 万行到 Python Pod 既慢又贵；而通过 Designer 组件直接在 MaxCompute 内运行模型则又快又省。如果你在设计文档中写下“首先需要导出这张表”，那你就已经输掉了“不该用 Designer”的论点。
- **“快速评估”正是 Gallery 的强项，即使后续要自托管也值得**。花 10 分钟在 Gallery 中确认 Qwen3-Coder 足够好，再花两天编写优化版 EAS 部署，远比先花两天写完 EAS 才发现模型不行更高效。
- **EAS 分支永远不会回退到 Gallery**。Gallery 自动生成的服务配置*尚可*（中位延迟稳定，无意外），但任何生产级服务都需要手写 EAS spec——锁定 vLLM 批处理参数、选择 GPU 类型、配置自动扩缩容。Gallery 适用于开发阶段，手写 EAS 才是生产之选。

我最常看到的错误是：“为了快速上线，先用 Gallery”，结果在账单暴涨或延迟问题凸显后，三个月才仓促迁移。一旦服务面向真实用户，就应立即规划迁移到手写 EAS。

## 各用例成本对比

以下数据来自我实际运行的工作负载，已归一化为月度人民币成本。具体价格因区域和季节而异，但*比例关系*才是关键。

**用例 A：基于 6000 万行 MaxCompute 表的每周用户分群**

| 方法 | 构建时间 | 单次运行成本 | 月成本（4 次） | 维护成本 |
|---|---|---|---|---|
| Designer | 2 小时 | ~6 元（MaxCompute 竞价实例） | ~24 元 | 可忽略 |
| DLC + PySpark | 3 天 | ~38 元（4 节点集群，运行 40 分钟） | ~152 元 | 每季度需一个工程师日更新镜像 |
| 手写 EMR 作业 | 1 周 | ~45 元 | ~180 元 | 需多个工程师日 |

在此场景下，Designer 在所有维度全面胜出。无需争论。

**用例 B：10 分钟内评估新开源 LLM（刚发布的 Qwen3-VL-7B）**

| 方法 | 评估耗时 | 评估期间成本 | 放弃成本 | 上线成本 |
|---|---|---|---|---|
| Model Gallery | 15 分钟（部署）+ 实际评估时间 | ~5 元/小时 × 评估时长 | 0（删除服务即可） | 需通过 EAS 重新部署用于生产 |
| 手写 EAS | 1–2 天（研究 vLLM、挂载权重、调试） | ~5 元/小时 ×（评估+设置时间） | ~50 元沉没成本 | 已就绪 |
| DSW Notebook | 1 小时（下载模型、运行推理循环） | ~5 元/小时 × 评估时长 | 0 | 无法从 DSW 提供服务 |

Gallery 在“评估速度”上胜出。若放弃，无后续负担；若上线，反正需通过 EAS 重部署（因 Gallery 默认配置不适合生产）。手写 EAS 仅在你*确定会上线*且不介意前期投入时才有优势。

**用例 C：生产级 LLM 聊天端点，平均 5 QPS，峰值 30 QPS**

| 方法 | 设置耗时 | 月成本 | p99 延迟 | 备注 |
|---|---|---|---|---|
| Model Gallery（默认） | 5 分钟 | ~14,000 元 | ~2.5 秒 | 默认 min_replicas=2，未调优批处理 |
| 手写 EAS（优化） | 1–2 天 | ~10,500 元 | ~1.2 秒 | 调优 vLLM、配置定时扩缩容、权重预置 |
| Bailian 托管 Qwen-Plus | 0（纯 API） | 波动，通常 ~3–8 元/百万 tokens | ~1.5 秒 | 使用他人 GPU，运维由对方负责 |

这正是每次“是否自托管？”规划会上应有的讨论。若月请求量在 5 万–10 万之间，Bailian 在成本和运维负担上占优；一旦超过 100 万请求/月或有数据驻留要求，自托管 EAS 才更具优势。

**用例 D：在 5000 条 JSON 样本上微调 Qwen3-7B**

| 方法 | 设置耗时 | 运行成本 | 效果 |
|---|---|---|---|
| Model Gallery 默认 LoRA | 10 分钟 | ~80 元（单 A100，约 3 小时） | 对多数任务，效果在最优值的 5% 以内 |
| DLC 自定义 Megatron + LoRA | 2–3 天 | ~60–100 元 | 可调至最优，适用于 >5 万样本场景 |
| DSW 手动运行 | 半天 | ~80 元 | 效果同 Gallery，但更易调试 |

除非有特殊理由需精细调整训练循环，否则 Gallery 仍是首选。其“默认 LoRA 超参数”出奇地好——经我多次 benchmark，与手工调优配置的差距始终很小。

四个用例共同揭示了一个规律：**Designer 和 Gallery 在“快速产出可用结果”上占优，在大规模下的成本优化或质量极限上略逊一筹**。建议用它们完成前 80% 的工作；仅当有明确证据表明必要时，再迁移到手写的 DLC / EAS 处理最后 20%。

![PAI 产品决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig_pai_decision_en.png)

## 突破瓶颈：从 Designer/Gallery 迁移到原生 PAI 资源

终有一天，你会因某个服务的需求超出 GUI 能力而需要迁移。以下是我在实践中验证过的路径：

**从 Designer 画布迁移到原生 DLC + DataWorks**。痛苦的做法是“用 PySpark 从头重写”；高效做法是“导出 JSON，将每个 Python 节点替换为对应的 DLC 提交任务，保留 MaxCompute 步骤作为 DataWorks SQL 节点”。具体步骤：

1. 从 Designer 右上角菜单导出工作流 JSON。
2. 识别仅依赖 MaxCompute 的子图，将其节点一对一转换为 DataWorks 中的 ODPS SQL 节点。
3. 找出需自定义逻辑的 Python 节点，为每个节点编写 `train.py` 并通过 `TrainingJob` SDK 提交（参见第三篇）。
4. 通过 DataWorks 的“Shell 节点 + PAI SDK 调用”触发 DLC 任务。
5. 待 DataWorks 版本稳定运行两周后，再删除原 Designer 工作流。

典型 Designer 画布的迁移耗时 1–2 天。最大收益是可观测性——DataWorks 提供分阶段日志、重试机制和正式告警，这是 Designer 画布无法比拟的。

**从 Gallery 部署迁移到手写 EAS**。这是我最常执行的迁移，正确顺序如下：

1. 在 Gallery 服务详情页点击 *View configuration*，查看底层 EAS 服务 spec（镜像、模型路径、命令、资源、自动扩缩容等）。
2. 将该 spec 复制到代码仓库的 YAML 文件中，使其成为可版本控制的标准 EAS 部署。
3. 用该 YAML 部署一个新 EAS 服务（命名需区分），验证其行为与 Gallery 服务一致。
4. 逐步调优：将权重烘焙进自定义镜像、将扩缩容指标改为 `concurrent_requests`、添加定时扩缩容、配置服务组流量镜像——每次改动对应一个 PR。
5. 通过服务组（参见第四篇模式）将流量从 Gallery 切换至手写版本。
6. 删除 Gallery 部署。

核心原则是：不要重写，而是*导出后重构*。Designer 和 Gallery 生成的配置本质上都是合法的 PAI 原语——并无魔法。应将它们视为起始模板，而非困住你的牢笼。

**反向迁移（原生 → Gallery）虽罕见但可行**。若你为一个本质是标准开源模型的场景构建了复杂的自定义 EAS 部署，可将其简化回 Gallery 以降低运维复杂度。我曾将一个膨胀至 600 行 YAML 的 DeepSeek 自定义部署，简化为 Gallery 服务 + 一个小型 EAS 侧代理（处理 Gallery 无法表达的部分）。团队无人怀念那堆 YAML。

## 接下来是什么

本系列至此完结。回顾如下：

- **第一篇** — PAI 是什么，以及各组件如何协同。
- **第二篇** — DSW 用于开发。
- **第三篇** — DLC 用于训练。
- **第四篇** — EAS 用于生产服务。
- **第五篇** — Designer / Model Gallery 适用于 GUI 确实更合适的场景。

配套的 **Aliyun Bailian** 系列将涵盖 DashScope、Qwen、Wanxiang 和 Qwen-TTS——这是构建在相同 PAI-EAS 基础设施之上的*托管* MaaS 层。许多团队两者并用：当需要在自有 GPU 上运行自有模型时选择 PAI；当只需通过 API Key 调用他人模型时选择 Bailian。按你所需控制的粒度来选择即可。
