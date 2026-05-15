---
title: "阿里云 PAI（四）：EAS 部署与冷启动真相"
date: 2026-03-08 09:00:00
tags:
  - Aliyun PAI
  - PAI-EAS
  - Model Serving
  - Inference
  - Auto Scaling
categories: 阿里云 PAI
lang: zh
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 4
series_total: 5
description: "PAI-EAS 端到端：基于镜像 + OSS 挂权重的部署方式、三种推理模式、不让账单爆炸的扩缩容配置，以及用服务组做灰度发布。配上来自官方 Quick Start 的 vLLM Qwen3 完整部署示例。"
disableNunjucks: true
translationKey: "aliyun-pai-4"
---
钱主要花在 EAS 上：DSW 开发每月只需几百元，DLC 训练属于脉冲式消费，而 EAS 则是 24/7 持续计费——服务一旦进入 Running 状态，费用便持续产生。自动伸缩配置中的 `min_replicas`（最小副本数）是整个平台最关键的杠杆。这篇文章汇总了我在部署首个生产端点前最希望掌握的关键信息。

![Aliyun PAI (4): PAI-EAS — 模型服务、冷启动与 TPS 谎言 — 视觉图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/illustration_1.png)

---

## 文档里的 EAS 是什么

官方《EAS 概述》将其定义为：“将训练好的模型部署为在线推理服务或 AI Web 应用，支持异构资源、自动伸缩、一键压测、灰度发布与实时监控。”核心要点如下：

- 它是**容器化服务层**：模型存放在 OSS 中，代码打包在容器镜像里；EAS 启动时拉取镜像、挂载 OSS、执行启动命令，然后监听端口。
- 它是**按副本数自动伸缩**：并非 Serverless 函数模型（有个重要例外，见下文）；每个副本都是真实的 GPU Pod，启动耗时 30–120 秒，必须据此规划资源和伸缩策略。

## 请求链路

![EAS 请求链路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig1_eas_request_path.png)

文档指出，基于运行时镜像的部署包含以下四个关键组件：

1. **运行时镜像**：只读模板，包含操作系统、CUDA、Python 及依赖项。可使用官方镜像（如 `vllm:0.11.2-mows0.5.1`、`pytorch:...`），也可自行构建并推送到 ACR。
2. **代码和模型**：*不包含在镜像中*，而是存储在 OSS 或 NAS 上。这种解耦设计允许你在不重建镜像的情况下更新模型权重。
3. **存储挂载**：启动时，EAS 通过 FUSE 将你指定的 OSS 路径挂载到容器内的目录，例如 `/mnt/data/`。
4. **运行命令**：容器启动后执行的第一条命令，通常是启动 HTTP 服务（如 `vllm serve /mnt/data/Qwen/Qwen3-0.6B`）。

> **实战建议**：从项目第一天起就将 `/mnt/data/` 写入代码路径，切勿硬编码模型路径为 `/workspace/models/`。这样只需修改一行配置，就能无缝切换本地开发环境与 EAS 生产环境，避免大规模代码重构。

## 三种推理模式

![EAS 推理模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig2_eas_inference_modes.png)

文档列出了三种推理模式，选错会白白浪费成本或增加延迟。一个实用的判断准则如下：

- **实时同步**：适用于聊天机器人、RAG 检索、广告排序、搜索等场景，这类服务对 p99 延迟敏感。
- **异步**：适用于耗时超过 5 秒的任务，如图像生成、视频生成、PDF 批量 OCR。其内置队列会根据任务积压自动扩缩容，非常适合此类工作负载。
- **批量**：适用于可等待数分钟的任务，如夜间 Embedding 计算、语音转录。此时可选用抢占式实例，成本直接减半。

## 真正的 Quick Start

官方 Quick Start 使用 vLLM 部署 Qwen3-0.6B，控制台操作流程如下：

1. **部署方式**：基于镜像部署。
2. **镜像**：`vllm:0.11.2-mows0.5.1`（官方 EAS 镜像；注意 vLLM ≥ 0.8.5 才支持 OpenAI 兼容接口）。
3. **模型位置**：OSS 路径 `oss://your-bucket/models/`，挂载到容器内 `/mnt/data/`。
4. **启动命令**：`vllm serve /mnt/data/Qwen/Qwen3-0___6B`。
5. **计算资源**：`ecs.gn7i-c16g1.4xlarge`（1 × A10 GPU）。
6. **点击部署**：约 5 分钟后状态变为 `Running`。

随后你会获得一个 OpenAI 兼容的推理端点 URL，可直接调用：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EAS_TOKEN"],          # the token from "View Call Information"
    base_url="https://YOUR-ENDPOINT.cn-shanghai.pai-eas.aliyuncs.com/v1",
)

resp = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is EAS in one sentence?"}],
)
print(resp.choices[0].message.content)
```

若返回有效文本，说明端点已成功上线，你可以自信地向同事展示成果了。

## 自动伸缩的正确姿势

这部分文档并未充分强调：默认的自动伸缩策略（基于请求速率，最小副本数为 1）极易引发两类问题——冷启动导致高延迟，或低负载时资源闲置推高账单。

![EAS 自动伸缩 — 副本数跟踪 QPS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig3_eas_autoscaling.png)

真正关键的三个配置项：

- **`min_replicas`**：生产环境切勿设为 0。一个 7B 模型的 vLLM 容器冷启动需 60–120 秒，而用户通常在 5 秒内就会放弃。我习惯设为 2：一个保障高可用，一个提供冗余。异步服务可设为 0，依靠队列缓冲。
- **`max_replicas`**：这是你的预算刹车。计算公式为 `(单副本 p99 QPS) × 2`。若不清楚单副本性能，请务必运行一键压测（文档中“服务压测”部分有说明）。
- **伸缩指标**：默认为 `qps`，但对 LLM 服务而言这是误导性指标。应改用 `concurrent_requests`（或 vLLM 的 `running` 指标）。因为长文本生成虽只算一次请求，却会长时间占用 GPU 资源，QPS 无法反映真实负载。

> **实战建议**：我在 PAI 上见过最离谱的浪费案例：一个服务配置了 `max_replicas=50` 和 `min_replicas=10`，但实际 off-peak 流量仅 0.5 QPS。结果 5 台空闲 A10 GPU 24/7 运行了整整两个月。节假日前务必检查周六晚间的监控仪表盘。

## 灰度、蓝绿与流量镜像

EAS 通过**服务组**（Service Group）实现这些功能：一个路由前端指向多个服务版本，并按比例分配流量。同一机制也支持**流量镜像**——将真实请求复制一份发送给候选版本，但丢弃其响应，用户完全无感。这是在生产环境中安全验证新模型的最佳方式。

![EAS 服务组 — 灰度与镜像](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig4_eas_canary_release.png)

每次模型切换，我都会采用渐进策略：前 24 小时先以 90/10 分流，随后调整为 50/50，最后全量切至新版本。若任一阶段的成功率或 p99 延迟出现恶化，立即回滚——服务组调整流量权重仅需几秒钟。

## 压测——真的要做

文档专门有一节介绍“一键压测”功能。请务必使用它：系统会自动提升 QPS，绘制副本扩容曲线，并告诉你单副本的饱和点。这个数值是你配置自动伸缩策略的基础。未经过压测就上线，是导致“模型在下午 3 点高峰崩溃”这类事故的最常见原因。

## 天大坑

文档角落藏着一句警告：“如果 EAS 服务连续 180 天处于非 Running 状态，系统将自动删除该服务。”建议设置日历提醒。我就曾因负责团队解散、无人续费，导致一个服务配置被自动清理。恢复过程花了整整一个下午，反复排查哪个 `vllm` 版本对应哪套模型权重。

## 冷启动优化，按效果排序

![Aliyun PAI (4): PAI-EAS — 模型服务、冷启动与 TPS 谎言 — 视觉图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/illustration_2.png)

冷启动是 EAS 最棘手的实际问题。一个 vLLM Qwen3-7B 容器从调度器选中到返回首个 Token 需要 60–120 秒——仅模型加载就占 30–60 秒。若自动伸缩在此期间新增副本，首批用户请求几乎必然超时。

以下是我实际投产过的优化方案，按效果从高到低排序：

**1. 预构建容器，将权重打入镜像（节省 30–60 秒）**  
默认流程在容器启动时从 OSS 下载模型。改为将模型直接嵌入镜像——Qwen3-7B 增加 14 GB 层大小完全可接受，因为 EAS 会在节点级别缓存镜像。虽然新节点首次启动仍慢，但同一节点后续启动可跳过 OSS 拉取。代价是每次更新权重都需重建镜像（一个 Dockerfile + CI 任务，约 10 分钟构建 + 5 分钟推送）。

**2. 预热 Ping（节省 5–15 秒 CUDA / kernel 初始化时间）**  
vLLM 容器在 HTTP 服务启动后即视为“运行中”，但首个真实请求会触发特定 batch shape 的 CUDA kernel JIT 编译。可通过合成请求预热：

```python
# In your container start script, after vllm serve is healthy:
import requests, time
def warmup():
    for _ in range(3):
        requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "Qwen3-7B",
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 4,
        })
    print("warmup done")

# Run once before EAS health check passes
threading.Thread(target=warmup, daemon=True).start()
```

EAS 在健康检查返回 200 之前不会路由真实流量，因此应让健康检查在预热完成后才返回成功。这会增加 5–15 秒的“可见”冷启动时间，但能消除前 1–3 个真实请求的延迟惩罚。

**3. 通过共享 NAS 预加载权重（节省 20–40 秒）**  
不从 OSS 下载，而是挂载一个已包含权重的 NAS 卷。NAS 的读取带宽比 OSS-FUSE 更稳定，模型加载时间可从 30–60 秒降至 10–20 秒。仅当你管理大量小副本时值得采用；单个大副本用 OSS-FUSE 已足够。

**4. 始终保持最小副本在线（100% 消除冷启动，代价为副本 × 24 小时费用）**  
这是最直接的手段。设 `min_replicas=2`，用户永远遇不到冷启动。按 A10 价格估算：~5 RMB/h × 24 × 30 ≈ 3600 RMB/月/副本。若服务收入影响远超此成本，显然值得；若是低流量内部工具，则过于奢侈——可考虑下一项。

**5. 定时伸缩（在可预测低谷期节省冷启动成本）**  
EAS 支持基于时间窗口的伸缩规则：

```yaml
autoscaling:
  rules:
    - name: business-hours
      cron: "0 9 * * MON-FRI"
      min_replicas: 3
      max_replicas: 20
    - name: off-hours
      cron: "0 19 * * MON-FRI"
      min_replicas: 1
      max_replicas: 5
    - name: weekends
      cron: "0 0 * * SAT,SUN"
      min_replicas: 0
      max_replicas: 3
```

我在 B2B 服务中广泛使用此策略。对于典型的中国大陆工作时段流量模式，可节省约 40% 的副本小时数，且对用户无任何延迟影响。

**6. 异步推理模式（绕过冷启动问题）**  
对于能容忍数秒排队延迟的工作负载（如图像生成、长文本生成），使用异步模式。其队列基于积压任务而非 QPS 扩容，因此即使冷启动耗时 90 秒，用户看到的也只是 90 秒排队时间，而非 5 秒超时。成本相同，但体验截然不同。

我实际用于生产 LLM 服务的组合策略是：预构建含权重的容器 + 启动脚本中加入 warmup ping + 工作时间 `min_replicas=2`、夜间降至 `1` + 所有耗时 >3 秒的推理走异步模式。

## 自动伸缩策略：CPU、请求速率还是自定义指标？

默认伸缩器基于 QPS 触发，但这对 LLM 服务并不合适，而文档并未解释清楚原因。我们稍作分析：

vLLM 通过 paged-attention batching 处理请求，吞吐量取决于两个因素：一是**并发请求数**（越高 GPU 利用率越好，直至达到 batch 上限），二是**每个请求的生成长度**（越长占用 GPU 时间越久）。而 QPS（每秒请求数）与这两者均无直接关联。

EAS 提供三种伸缩指标，适用场景各不相同：

**`qps`（默认）**  
按请求到达速率伸缩。适用于同步、固定成本的接口（如图像分类、Embedding 生成），**不适用于**生成长度可变的场景。

**`concurrent_requests`**  
按当前正在处理的请求数伸缩。适用于 LLM 聊天、RAG 接口等可明确设定单副本目标并发数的场景。具体数值建议：运行一键压测，找到 p99 延迟开始显著上升的并发水位，将目标值设为其 70%。

**自定义指标（CloudMonitor）**  
可基于任意自定义指标伸缩。我常用以下两种：
- `vllm_running_requests_avg`：vLLM 内部“正在解码”的请求数，比 EAS 的 `concurrent_requests` 更精准，因为它排除了已排队但尚未开始解码的请求。
- `gpu_memory_pct`：当 KV Cache 成为瓶颈时（如长上下文场景），可在 GPU 内存利用率达 75% 时触发扩容。

以下是 Qwen3-7B 聊天服务的一个实战配置示例：

```yaml
autoscaling:
  metric: concurrent_requests
  target: 12              # found via stress test: p99 spikes above 16
  min_replicas: 2
  max_replicas: 10
  scale_up_stabilization_window: 60s
  scale_down_stabilization_window: 600s   # slow scale-down
```

注意其中**非对称的稳定窗口**：扩容要快（阈值突破后 60 秒内响应），避免用户排队；缩容要慢（持续低负载 10 分钟后再释放副本），防止流量波动导致频繁扩缩。默认的对称策略会产生过多不必要的 churn。

**我不推荐的指标：CPU**  
EAS 虽支持 `cpu` 作为伸缩依据，但 vLLM 是 GPU 密集型应用，无论负载高低，CPU 利用率始终维持在 5–15%。基于 CPU 伸缩要么永不触发，要么因偶然的内存分配峰值误触发，与实际服务能力毫无关系。

## 真正可用的蓝绿部署 + 流量调度

服务组提供了基础能力，但要用好还需一套规范流程。我每次更换模型都遵循以下步骤：

**步骤 0：将候选模型部署为新服务，设 `min_replicas=2`**  
使用相同镜像、硬件和 OSS 路径，仅指向新权重。暂不加入服务组。

**步骤 1：私有流量 Sanity Check**  
直接调用候选服务的独立端点（绕过服务组），使用 50–200 条已标注的 golden prompt 进行验证。若失败，就无需浪费服务组资源测试一个明显有问题的模型。

**步骤 2：镜像 5% 生产流量，持续 1 小时**  
EAS 通过服务组路由中的 `mirror_weight: 5` 字段实现流量镜像：真实请求被复制给候选服务，响应被丢弃，不影响用户。借此可离线对比新旧模型的 p99 延迟、错误率及回复质量差异。

```python
# Service group config (illustrative):
service_group = {
    "name": "qwen-chat-prod",
    "routes": [
        {"service": "qwen3-7b-v23", "weight": 100, "mirror_weight": 0},
        {"service": "qwen3-7b-v24", "weight": 0,   "mirror_weight": 5},
    ],
}
```

**步骤 3：分阶段切流，24 小时内完成 5% → 25% → 50% → 100%**  
每阶段至少维持 1 小时，监控成功率、p99 及定性反馈。一旦指标异常，立即将权重降回 0——服务组更新通常在 10 秒内生效。

**步骤 4：保留旧服务至少 48 小时再下线**  
万一在迁移后第 36 小时需要回滚，“将旧服务权重设回 100%”是最快恢复手段，远胜于从 OSS 重新部署。

此外，服务组还支持更精细的流量拆分：按 user-agent（先在单一客户端测试）、按 region（如先在 cn-shanghai 灰度，再推 cn-hangzhou）、按 request-header（区分内部测试与公开用户）。我曾利用 user-agent 拆分，将新模型优先推送给团队的 Cherry Studio 会话，成功捕获了标准评测集未能发现的 bug。

## 单次推理的成本算术

我为每个 EAS 服务都维护一张核心成本表。单次推理成本主要由副本小时数（replica-hours）主导，而非按请求计费。以一个 Qwen3-7B 聊天服务为例（业务时段约 5 QPS）：

| 组件 | 数量 | 月成本 |
|---|---|---|
| 最小副本基线 (2 × A10, 24/7) | 2 × 5 RMB/h × 720 h | ~7,200 RMB |
| 突发副本 (平均 9 小时 × 22 天多 3 个) | 3 × 5 × 198 h | ~2,970 RMB |
| 按请求费 (5 QPS × 86400 × 22) | 9.5 M req × 0.0001 RMB | ~950 RMB |
| OSS 带宽 (冷启动加载模型) | 14 GB × 30 冷启动 × 0.5 RMB/GB | ~210 RMB |
| **总计** | | **~11,330 RMB** |

这张表揭示了三个关键认知，我花了很长时间才真正内化：

1. **最小副本成本占主导**：非业务时段将 `min_replicas` 从 2 降至 1，可省约 3,600 RMB/月；降至 0 可再省同等金额，但会引入 60–120 秒冷启动。决策应基于 SLA，而非“求稳”直觉。
2. **LLM 场景下按请求费可忽略**：该费用仅在高 QPS 分类任务（如 1000+ QPS）中显著；对于聊天或生成类服务，完全可以忽略。
3. **规模化后冷启动带宽成本不可忽视**：30 次冷启动 × 14 GB = 420 GB OSS 读流量，按跨区 0.5 RMB/GB 计算，已是实打实的支出。将权重打入镜像（见上文冷启动章节）可使此项成本趋近于零。

我评估任何新服务时都使用以下公式：

```text
monthly_cost = min_replicas × replica_price_per_hour × 720
             + avg_burst_replicas × replica_price_per_hour × business_hours_month
             + total_requests × per_request_fee
             + cold_starts × model_size_gb × oss_read_price
```

**务必在部署前填入实际数值**。现实中不乏因默认 `min_replicas=10` 导致“小型”部署意外产生 $2k/月账单的案例。控制台点击 Deploy 时会显示价格预估——记得仔细查看。

## 下一步

第 5 篇将为本系列收尾，我会坦诚探讨 **Designer** 与 **Model Gallery** ——这两个零/低代码界面。尽管大多数工程师不会首选它们，但在特定场景下，它们确实能创造巨大价值，甚至成为显而易见的最佳选择。
