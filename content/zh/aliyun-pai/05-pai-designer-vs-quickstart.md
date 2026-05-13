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
前四篇介绍了底层原语 DSW、DLC、EAS 以及如何用 Python 编排；这篇则聚焦于两个 GUI 产品：**PAI-Designer**（拖拽式表格 pipeline 构建）和 **Model Gallery**（零代码部署与微调开源模型）。这两个工具将原语封装成开箱即用的交付能力，专为不写代码的用户设计。尽管资深工程师不常首选这些工具，但在某些关键场景下，它们是唯一正解。

![Aliyun PAI (5): Designer vs Model Gallery — When the GUIs Actually Earn Their Keep — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/illustration_1.png)

## Designer —— 拖拽式 Pipeline 编排工具

Designer 基于工作流实现建模与调试，用户通过拖拽组件即可构建 AI 开发流程，内置 140 多个算法组件，支持导出 JSON 和接入 DataWorks 调度，节点还可以嵌入自定义 SQL、Python 或 PyAlink 脚本。

![PAI-Designer canvas](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig1_designer_canvas.png)

它的强项：
- **MaxCompute 规模的表格机器学习**：与 MaxCompute 深度集成，当训练数据是 MaxCompute 上 2 亿行分区表时，source、split、encode、train 等内置组件直接在 MaxCompute 内执行，免去网络传输至 Python pod 的开销，只需支付 MaxCompute 计算费用，无闲置 GPU 实例成本。
- **交接给不写代码的分析师**：推荐、流失、风控团队常有懂业务但不懂 Python 的专家。Designer 画布直观易懂，业务人员能自主修改并长期维护。
- **内置模板**：文档列了商品推荐、新闻分类、金融风控、雾霾预测、心脏病预测、农业贷款、census 分析等开箱即用案例。即使你最终替换了近半数节点，这些模板仍是极佳的起点。
- **调度离线运行**：工作流导出 JSON，交给 DataWorks，搞定带重试的日/小时 cron。

它的弱项：
- **所有 LLM 相关任务**：Designer 擅长特征工程与经典 ML，但不支持自定义 PyTorch 训练循环。
- **自定义 CUDA 工作、新颖 loss、任何“算法即核心”的场景**。

对于表格负载，我会用 Designer 替代 DLC + SQL；其他情况下，在 DLC 中运行自定义训练模型。

## Model Gallery —— 零代码 MaaS 捷径

Model Gallery 将 DLC + EAS 封装为零代码界面，非 MLOps 用户只需点击约六次即可完成开源模型的微调与部署。官方文档称其封装了 Platform for AI (PAI)-DLC 和 PAI-EAS，提供高效部署与训练开源大语言模型的零代码方案。

![Model Gallery pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig2_modelgallery_pipeline.png)

我们通过 Quick Start 完成了 Qwen3-0.6B 的全流程部署：

1. 在 Model Gallery 搜索 "Qwen3-0.6B" → 点击 **Deploy**。
2. 默认 GPU 类型，默认 vLLM 镜像，全程默认 → **OK**。
3. 大概 5 分钟后状态变为 `Running`。
4. **View Call Information** → 拿到 `Internet Endpoint` 和 token。
5. 接入 Cherry Studio （或 Claude Code MCP，或用 OpenAI 兼容 base URL 的 Python SDK）开始聊天。

微调方面，文档以物流信息提取为例：输入一个 JSON 数据集，从下拉菜单选择 LoRA 超参数，系统会自动提交一个 DLC 任务。Quick Start 特别提到 **蒸馏模式** —— 用大模型（Qwen3-235B）标注数据，让小模型（Qwen3-0.6B）跟随学习。这一模式值得关注，据我所知，这是目前性价比最高的微调方案。

Gallery 的强项：

- **10 分钟内评估新模型。** DeepSeek-V3 发布时，我们团队喝完一杯咖啡的时间就部署好聊上了。如果还要设 OSS bucket、安全组、 SSL cert，光靠 `vllm serve` 做不到这点。
- **给非工程利益相关者演示。** 点击 → 端点 → Cherry Studio 聊天 → 董事会汇报。
- **一键 LoRA 微调。** 大多数领域适配工作， Gallery 选的默认值（LR, epochs, LoRA rank）跟最优解差距在 5% 以内。

Gallery 的弱项：

- 自定义架构。改过模型代码就得用 DSW + DLC。
- 严格延迟目标。 Gallery 的默认服务配置基本可用，但未经过精细调优。需要 <100ms p99 就得自己写 EAS 部署配 batching。
- 隔离网或跨区部署。 Gallery 假设“在你所在的区域部署”。

## 什么时候选哪个

这个决策矩阵我一直沿用：

![Decision matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig3_decision_matrix.png)

总结一条经验法则：**在需求允许的情况下，尽量从栈的高层开始**。许多团队一开始就过度工程化，明明可以用 Model Gallery 部署，却非要自定义 DLC + EAS pipeline。应先优化首 token 时间，有了真实流量和指标后再重构下沉。

## 实战案例：什么时候 Designer 胜过自定义代码

一个真实工单：市场部想要基于 MaxCompute 里 6000 万行的表做每周用户分群。数据科学家的第一反应是 DLC 任务跑 PySpark + scikit-learn，代码放 OSS，通过 SLS-callback-to-EventBridge 调度。三天工作量。

Designer 版本： source 节点 → sample → encode → KMeans → 写回 MaxCompute。导出 JSON，在 DataWorks 调度。两小时搞定，包括跟市场 PM 解释的会议。输出表一样，成本减半（不用 GPU pod），维护量只有十分之一。

## 实战案例：什么时候 Model Gallery 省了一周时间

我们需要测试 Qwen3-Coder 是否足以替换内部基于 `qwen-plus` 的代码审查 bot。用 Gallery 之前流程是：读 vLLM 文档，搭 EAS 部署，写 OpenAI 兼容桥接，交给团队。用 Gallery 之后：搜索 → 部署 → 端点接入现有客户端 → 午饭前搞定。我们可以专注于真正的问题（模型是否更好），而不是底层基建活儿。

## 具体的决策树

![Aliyun PAI (5): Designer vs Model Gallery — When the GUIs Actually Earn Their Keep — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/illustration_2.png)

上面的矩阵是经验法则；这是队友问“这事儿该用 Designer / Gallery / DLC / EAS 哪个？”时我实际跑的决策树。

![PAI Product Decision Tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/05-pai-designer-vs-quickstart/fig_pai_decision_en.png)

几个不那么明显的分支：

- **数据就在 MaxCompute 里是选 Designer 的最强信号。** 不是看大小，不是看算法——是看*数据在哪*。用 `pyodps` 拉 5000 万行从 MaxCompute 到 Python pod 又慢又贵；在 MaxCompute 内部跑 Designer 组件又快又便宜。如果设计文档里写“先得导出表”，那你不用 Designer 的理由已经输了。
- **“快速评估”是 Gallery 强项，哪怕以后要自托管。** 花 10 分钟在 Gallery 确认 Qwen3-Coder 够用，再花 2 天写优化 EAS，比先花 2 天写 EAS 发现模型不行要快得多。
- **EAS 分支永远不回 Gallery。** Gallery 自动部署的服务配置*还行*（中位延迟，没惊喜），但生产级服务你得手写 EAS spec——锁定 vLLM batching 参数、选 GPU 类型、配 autoscaling。 Gallery 用于开发，手写 EAS 用于生产。

我最常见的错误：为了“快点上线”在 Gallery 上待得太久，直到账单来了或者延迟开始要命了，三个月后才 frantic 重部署。一旦服务面向真实用户，就应规划迁移到手写 EAS 部署方案。

## 各用例成本对比

数字来自我跑过的真实负载， normalized 到月度人民币。 exact 价格随区域和季节波动；*比例*才是关键。

**用例 A：基于 6000 万行 MaxCompute 表的每周用户分群。**

| Approach | Time-to-build | Per-run cost | Monthly cost (4 runs) | Maintenance |
|---|---|---|---|---|
| Designer | 2 hours | ~6 RMB (MaxCompute spot) | ~24 RMB | Negligible |
| DLC + PySpark | 3 days | ~38 RMB (4-node cluster, 40 min) | ~152 RMB | One engineer-day per quarter for image bumps |
| Hand-written EMR job | 1 week | ~45 RMB | ~180 RMB | Multiple engineer-days |

Designer 在所有维度上都赢。别争了。

**用例 B： 10 分钟评估新开源 LLM （刚发布的 Qwen3-VL-7B）。**

| Approach | Time-to-eval | Cost during eval | Cost if abandoned | Cost if shipped |
|---|---|---|---|---|
| Model Gallery | 15 min (deploy) + actual eval time | ~5 RMB/h × eval hours | 0 (delete service) | re-deploy via EAS for prod |
| Hand-rolled EAS | 1-2 days (figure out vLLM, mount weights, debug) | ~5 RMB/h × eval + setup time | ~50 RMB sunk | already there |
| DSW notebook | 1 hour (download model, run inference loop) | ~5 RMB/h × eval hours | 0 | nope, can't serve from DSW |

Gallery 赢在评估时间。如果放弃，没有后续工作；如果上线，反正你要通过 EAS 重部署，因为 Gallery 默认值不适合生产。手写 EAS 只有在你*确定*要上线且不介意前期成本时才赢。

**用例 C：生产级 LLM 聊天端点，平均 5 QPS，峰值 30 QPS。**

| Approach | Setup | Monthly cost | Latency p99 | Notes |
|---|---|---|---|---|
| Model Gallery (default) | 5 min | ~14,000 RMB | ~2.5 s | min_replicas=2 default, no batching tuning |
| EAS hand-written (optimized) | 1-2 days | ~10,500 RMB | ~1.2 s | tuned vLLM, scheduled scaling, weights baked |
| Bailian managed Qwen-Plus | 0 (it's an API) | varies — typically ~3-8 RMB per 1M tokens | ~1.5 s | someone else's GPU, someone else's problem |

这是每次“要不要自托管？”规划会议上该发生的对话。如果用量是每月 5 万 -10 万请求， Bailian 在成本和运维负担上赢。超过每月 100 万请求或有数据驻留要求，自托管 EAS 领先。

**用例 D：在 5000 条 JSON 数据集上微调 Qwen3-7B。**

| Approach | Setup | Run cost | Quality |
|---|---|---|---|
| Model Gallery default LoRA | 10 min | ~80 RMB (single A100, ~3 h) | Within 5% of optimal for most tasks |
| DLC custom Megatron + LoRA | 2-3 days | ~60-100 RMB | Tunable to optimal, worth it for >50K examples |
| DSW manual run | 一半 day | ~80 RMB | Same as Gallery, more inspectable |

除非你有特定理由调整循环，否则 Gallery 这里也赢。 Gallery 里的“默认 LoRA 超参数”出奇的好——我跟手写调优配置 benchmark 过，差距 一致地 很小。

这四个用例的模式：**Designer 和 Gallery 赢在“尽快跑出东西”，在成本优化或大规模质量上略输**。前 80% 工作用它们，有证据表明需要时再 breakout 到手写 DLC / EAS 做最后 20%。
## 突破瓶颈：从 Designer/Gallery 转向原生 PAI 资源

迟早有一天，你会发现某个服务的 GUI 界面不够用了。我自己实际跑通过的迁移路径有以下几条：

**从 Designer 画布到原生 DLC + DataWorks。** 最痛苦的做法是“用 PySpark 从头重写”；最省事的办法是“导出 JSON，把每个 Python 节点替换成对应的 DLC 提交， MaxCompute 步骤保留为 DataWorks SQL 节点”。步骤如下：

1. 从 Designer 右上角菜单导出 workflow JSON。
2. 找出只用 MaxCompute 的子图，把这些节点挪进 DataWorks 工作流里当 ODPS SQL 节点（基本上是一对一转换）。
3. 找出需要自定义逻辑的 Python 节点；每个节点写一个 `train.py` 并用 `TrainingJob` SDK 提交（参考第 3 章）。
4. 用"Shell 节点 + pai SDK 调用”把 DataWorks 工作流和 DLC 任务连起来。
5. 等 DataWorks 版本稳定运行 2 周后，再删除 Designer 工作流。

迁移一个典型的 Designer 画布大概需要 1-2 天。好处是可观测性——DataWorks 能提供分阶段日志、重试机制和正规告警，这点 Designer 画布做不到。

**从 Gallery 部署到手写 EAS。** 这是我做得最多的迁移，正确的顺序是这样的：

1. 在 Gallery 服务详情页点击 *View configuration*，它能显示底层的 EAS 服务 spec （镜像、模型路径、命令、资源、自动伸缩）。
2. 把这个 spec 复制到 repo 里的 YAML 文件。现在它就是个普通的 EAS 部署，可以版本控制了。
3. 用这个 YAML 部署一个新的 EAS 服务（名字分开）。验证它和 Gallery 那个服务表现一致。
4. 调优：把权重打进自定义镜像，指标换成 `concurrent_requests`，加上定时伸缩，配置服务组的流量镜像。每次改动都是一个 PR。
5. 通过服务组把流量从 Gallery 迁移到手写版本（参考第 4 章的模式）。
6. 删除 Gallery 部署。

模式很简单：别重写，*导出然后重构*。 Designer 和 Gallery 生成的配置都是合法的 PAI 原语——没什么魔法。把它们当作起始模板，而不是把你困住的终点。

**反向迁移（原生 → Gallery）很少见，但也可行。** 如果你为一个本质上是开源模型且配置标准的场景构建了自定义 EAS 部署，可以把它缩回 Gallery 部署以减少运维面。我做过一次：一个自定义 DeepSeek 部署膨胀到了 600 行 YAML，后来简化回 Gallery 服务，加上一个小的 EAS 侧代理来处理 Gallery 表达不了的部分。团队里没人怀念那堆 YAML。

## 接下来是什么

系列文章到此为止。回顾一下：

- **Article 1** — PAI 是什么以及各部分如何衔接。
- **Article 2** — DSW 用于开发。
- **Article 3** — DLC 用于训练。
- **Article 4** — EAS 用于生产服务。
- **Article 5** — Designer / Model Gallery 适用于 GUI 正确的场景。

配套的 **Aliyun Bailian** 系列涵盖 DashScope、 Qwen、 Wanxiang 和 Qwen-TTS —— 这是构建在上述 PAI-EAS 基础设施之上的*托管* MaaS 层。很多团队两者都用：需要自己的模型跑在自己的 GPU 上时用 PAI，需要通过 API 密钥调用别人的模型时用 Bailian。按你需要控制什么来选择。