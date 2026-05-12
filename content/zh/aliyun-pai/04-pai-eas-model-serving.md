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
description: "PAI-EAS 端到端：基于镜像 + OSS 挂权重的部署方式、三种推理模式、不让账单爆炸的扩缩容配置，以及用服务组做灰度发布。配上来自官方 Quick Start 的 vLLM Qwen3 完整部署示例。"
disableNunjucks: true
translationKey: "aliyun-pai-4"
---
钱主要花在 EAS 上：DSW 开发每月只需几百元，DLC 训练属于脉冲式消费，而 EAS 则是 24/7 持续计费——服务一旦进入 Running 状态，费用便持续产生。自动配置里的 `minimum replica count` 这一行，是整个平台杠杆最高的旋钮。这篇文章汇总了我在部署首个生产端点前最希望掌握的关键信息。

![Aliyun PAI (4): PAI-EAS — 模型服务、冷启动与 TPS 谎言 — 视觉图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/illustration_1.png)

## 文档里的 EAS 是什么

官方《EAS 概述》将其定义为：“将训练好的模型部署为在线推理服务或 AI Web 应用，支持异构资源、自动伸缩、一键压测、灰度发布与实时监控。”——核心要点如下：

- 它是**容器化服务层**——模型在 OSS 里，代码在容器镜像里。EAS 启动时拉镜像、挂载 OSS、运行启动命令，然后监听端口。
- 它是**按副本数自动伸缩**——不是 Serverless 函数模型（有个重要例外，见下文）。副本是真实的 GPU Pod，启动需要 30-120 秒，需据此规划资源配额和伸缩策略。

## 请求链路

![EAS 请求链路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/fig1_eas_request_path.png)

文档指出，运行时镜像部署包含以下四个关键部件：

1. **运行时镜像** —— 只读模板，包含 OS、 CUDA、 Python 和依赖。用官方的（`vllm:0.11.2-mows0.5.1`，`pytorch:...`）或者推自己的到 ACR。
2. **代码和模型** —— *不在镜像里*，它们存在 OSS/NAS。解耦后，更新权重不用重新构建镜像。
3. **存储挂载** —— 启动时， EAS 通过 FUSE 把你指定的 OSS 路径挂载到容器内的目录，比如 `/mnt/data/`。
4. **运行命令** —— 容器启动后的第一条命令。通常是启动 HTTP 服务（`vllm serve /mnt/data/Qwen/Qwen3-0.6B`）。

> **实战建议：** 从第一天起就把 `/mnt/data/` 写进代码路径。别让模型路径硬编码成 `/workspace/models/`。这样即可通过修改一行配置，完成从本地开发到 EAS 的迁移，无需重构代码。

## 三种推理模式

文档列了三种模式。选错模式会增加成本或提高延迟。

![EAS 推理模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/fig2_eas_inference_modes.png)

一个实用的启发式规则如下：

- **实时同步** —— 聊天机器人、RAG 检索、广告排序、搜索。你关心 p99 延迟。
- **异步** —— 任何耗时 5 秒以上的任务，如图像生成、视频生成、PDF 批量 OCR。内置队列会根据任务积压量自动伸缩副本，这是处理此类工作负载的合理方式。
- **批量** —— 任何可以等几分钟的任务，如夜间 Embedding、语音转录。使用抢占式实例，账单减半。

## 真正的 Quick Start

官方 Quick Start 是用 vLLM 部署 Qwen3-0.6B，控制台操作流程如下：

1. **方式：** 基于镜像部署。
2. **镜像：** `vllm:0.11.2-mows0.5.1`（官方 EAS 镜像 — vLLM ≥ 0.8.5 才支持 OpenAI 兼容聊天）。
3. **模型：** OSS，`oss://your-bucket/models/`，挂载路径 `/mnt/data/`。
4. **命令：** `vllm serve /mnt/data/Qwen/Qwen3-0___6B`。
5. **资源：** `ecs.gn7i-c16g1.4xlarge`（1 × A10）。
6. **点击部署。** 大约 5 分钟变 `Running`。

然后你就能拿到控制台提供的 OpenAI 兼容端点 URL，并调用它：

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

若收到有效响应，说明端点已成功上线，可以立即进行功能验证。

## 自动伸缩的正确姿势

这部分文档没有讲透，默认自动伸缩策略（按请求速率伸缩，最小副本数为 1）容易导致两类问题：冷启动延迟引发用户投诉，低负载时资源闲置推高费用。

![EAS 自动伸缩 — 副本数跟踪 QPS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/fig3_eas_autoscaling.png)

真正重要的三个设置：

- **`min_replicas`** —— 生产环境中，最小副本数切勿设为 0。 7B vLLM 容器的冷启动要 60-120 秒；用户 5 秒就放弃了。我通常将最小副本数设为 2：一个用于高可用保障，另一个提供冗余容量。异步服务可以设 0，靠队列顶着。
- **`max_replicas`** —— 预算刹车。计算公式：`(p99_qps_per_replica) * 2`。如果不知道单副本 QPS，跑一键压测。文档在“服务压测”部分讲了这点。
- **伸缩指标** —— 默认是 `qps`。 LLM 服务要换成 `concurrent_requests`（或者 vLLM 的 `running` 指标）。 QPS 容易产生误导，因为长文本生成类请求的 GPU 占用时间远高于请求频次所反映的负载。

> **实战建议：** 我在 PAI 上见过最严重的浪费，是一个 `max_replicas=50` 的自动伸缩器，配了 `min_replicas=10`，结果服务 Off-peak 只有 0.5 QPS。 5 台 idle A10， 24/7，跑了两个月。节前务必检查周六晚间的监控仪表盘。

## 灰度、蓝绿与流量镜像

EAS 通过**服务组**实现这一功能：一个路由前端指向多个服务版本，按百分比分流。同样的原语支持**流量镜像**——真实流量拷贝一份发给候选版本，但响应丢弃，用户无感知。这是在生产流量上测试新模型最安全的方式。

![EAS 服务组 — 灰度与镜像](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/fig4_eas_canary_release.png)

任何模型切换，前 24 小时我用 90/10 分流，然后 50/50，最后 0/100。如果任何一步的成功率或 p99 指标恶化，立刻回滚——服务组改流量权重只需几秒钟。

## 压测——真的要做

文档有一整节介绍一键压测器。使用它，可以自动提升 QPS，绘制副本扩容曲线，并给出单副本的饱和点。这个数字是你构建自动伸缩器的基础。未经过压测就上线，是导致“模型在下午 3 点流量峰值时宕机”这类工单的最常见原因。

## 180 天大坑

文档角落里藏着一句话：“如果 EAS 服务连续 180 天处于非 Running 状态，系统会自动删除服务。”建议设置日历提醒。我曾因负责团队解散、无人续费，导致一个服务配置被自动删除。恢复花了一个下午重新二分查找哪个 `vllm` 版本对应哪个权重。

## 冷启动优化，按效果排序

![Aliyun PAI (4): PAI-EAS — 模型服务、冷启动与 TPS 谎言 — 视觉图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/04-pai-eas-model-serving/illustration_2.png)

冷启动是 EAS 最实际的痛点。一个 vLLM Qwen3-7B 容器从调度器选中到首 Token 服务需要 60-120 秒 —— 光模型加载就要 30-60 秒。如果自动伸缩器在负载下需要加副本，窗口期内的首批用户请求会超时。

我实际上线过的优化方案，按效果排序：

**1. 预构建容器，权重打进去（省 30-60 秒）。** 默认流程是容器启动时从 OSS 下载模型。改成把模型 bake 进镜像 —— Qwen3-7B 加 14 GB 到层里没问题，因为 EAS 会按节点缓存镜像。新节点首次启动一样慢；但同一节点后续启动完全跳过 OSS 拉取。代价：每次权重变动都要重 build 镜像（一个 Dockerfile + CI 任务，大概 10 分钟 build， 5 分钟 push）。

**2. 预热 Ping （省 5-15 秒 CUDA / kernel 初始化）。** vLLM 容器 HTTP 服务起来就算"running"了，但第一个真实请求会触发该 Batch Shape 的 CUDA 内核 JIT 编译。用合成请求预热：

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

EAS 在健康检查返回 200 之前不会把真实流量路由给副本，因此要将健康检查改为预热完成后才返回。这会增加 5-15 秒的可见冷启动时间，但消除了前 1-3 个真实请求的 5-15 秒延迟惩罚。

**3. 通过共享 NAS 预加载权重（省 20-40 秒）。** 不从 OSS 下载，挂载一个已经有权重的 NAS 卷。 NAS 读取带宽比 OSS-FUSE 稳定，模型加载步骤（"loading model weights from /mnt/models/..."）从 30-60 秒降到 10-20 秒。只有管理很多小副本时才划算 —— 单个大副本， OSS-FUSE 路径没问题。

**4. 永远温热的最小副本（省 100% 冷启动，花费 副本 × 24 小时）。** 最粗暴的工具。`min_replicas=2`，首个用户永远看不到冷启动。按 A10 价格算：~5 RMB/h × 24 × 30 = ~3600 RMB/月 每个永远温热的副本。对于收入影响超过这条线的服务，显而易见。对于低流量内部工具，肉疼 —— 用下一项。

**5. 定时伸缩（在可预测的低谷期省冷启动成本）。** EAS 支持时间窗口伸缩规则：

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

我在流量可预测的 B2B 服务上用这个模式。典型的中国大陆工作时间模式下，大概省 40% 的副本小时数，用户无感知延迟影响。

**6. 异步推理模式（绕过问题）。** 对于能容忍几秒队列时间的工作负载（如图像生成、长文本生成），使用异步模式。队列根据积压而非 QPS 伸缩副本，因此在 surge 期间 90 秒的冷启动只会导致用户看到 90 秒的队列延迟，而不是 5 秒超时。花费相同，但用户体验完全不同。

我实际上线生产 LLM 服务的配置：预构建带权重的容器，启动脚本里加 warmup ping，工作时间 `min_replicas=2` 夜间降到 `1`，任何耗时 >3 秒的推理都用异步模式。
## 自动伸缩策略： CPU、请求速率还是自定义指标

默认伸缩器是靠 QPS 触发的。做 LLM 服务的话，这指标其实不对路，文档里也没讲清楚缘由。咱们稍微推导一下。

vLLM 副本靠 paged-attention batching 来处理请求。吞吐量主要看两个因素：一是*并发*请求数（并发越高 GPU 利用率越好，当然得在 batch 限制内），二是每个请求的*生成长度*（越长占 GPU 时间越久）。 QPS——也就是每秒启动的请求数——跟这两个核心因素都没啥直接关系。

EAS 暴露了三种伸缩指标，各有各的适用场景：

**`qps`（默认）**。按请求到达速率伸缩。适合：同步、固定成本的接口（比如图像分类、 embedding）。不适合：任何生成长度可变的场景。

**`concurrent_requests`**。按任意时刻正在处理中的请求数伸缩。适合： LLM 聊天、 RAG 接口，或者任何你能指定每副本目标并发数的场景。具体数值怎么定：跑一遍一键压测，找到 p99 延迟开始爬升的那个并发水位，把目标值设为该水位的 70%。

**自定义指标（CloudMonitor）**。按你发布的任意指标伸缩。我自己常用这两个：
- `vllm_running_requests_avg` — vLLM 内部“正在解码”的计数，比 EAS 侧的 `concurrent_requests` 更准，因为它排除了那些排了队但还没开始解码的请求。
- `gpu_memory_pct` — 当 KV cache 压力成为瓶颈时（比如长上下文负载）。内存利用率到 75% 就扩容。

一个 Qwen3-7B 聊天服务的实战配置：

```yaml
autoscaling:
  metric: concurrent_requests
  target: 12              # found via stress test: p99 spikes above 16
  min_replicas: 2
  max_replicas: 10
  scale_up_stabilization_window: 60s
  scale_down_stabilization_window: 600s   # slow scale-down
```

这种非对称的稳定窗口很关键。扩容要快（超过阈值 60 秒内动作），别让用户排队；缩容要慢（持续低负载 10 分钟后再移除副本），避免流量波动时实例反复启停。默认配置是对称的，会产生太多不必要的 churn。

**我不推荐的指标： CPU。** EAS 支持 `cpu` 作为指标，但 vLLM 是 GPU-bound 的，不管负载多少 CPU 利用率都停在 5-15%。按 CPU 伸缩要么永远不触发，要么因为某个内存分配峰值触发，但这跟实际服务能力毫无关系。

## 真正可用的蓝绿部署 + 流量调度

服务组给了你基础原语；要想用得好，得有点纪律。我每次换模型都跑这个流程：

**步骤 0：把候选模型部署为一个新服务，设 `min_replicas=2`。** 镜像、硬件、 OSS 路径都一样，只是指向新权重。先别把它放进服务组。

**步骤 1：用私有流量做 Sanity Check。** 直接调候选服务的接口（不走服务组），用固定 eval 集——50-200 个你已经 golden-labeled 的 prompt。如果这步挂了，就别浪费服务组路由的 churn 在一个坏模型上。

**步骤 2：镜像 5% 生产流量跑 1 小时。** 镜像会把真实用户请求复制给候选服务，丢弃响应，让你能离线对比候选和基线的回复。 EAS 通过服务组路由上的 `mirror_weight: 5` 字段实现。盯着 p99 延迟、错误率，以及（如果你记录响应的话）跟基线的定性差异。

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

**步骤 3：线上切流， 24 小时内按 5% / 25% / 50% / 100% 推进。** 每步至少保持 1 小时，监控成功率、 p99 以及每路由的定性检查。一旦有任何波动，直接把权重降回 0——服务组更新不到 10 秒。

**步骤 4：下线旧服务。** 100% 迁移后至少留 48 小时再删。如果在第 36 小时需要回滚，“把旧服务权重设回 100"是最快的恢复手段——比从 OSS 重新部署快多了。

流量调度原语还能做更有趣的拆分：按 user-agent （先在单个客户端测试）、按 region （先在 cn-shanghai 灰度再推 cn-hangzhou）、按 request-header 值（内部测试 vs 公开）。全都在服务组路由规则里配。我用过 user-agent 拆分，先把新模型推给我团队的 Cherry Studio 会话，这能抓到 eval set 漏掉的 bug。

## 单次推理的成本算术

我为每个 EAS 服务都维护着一张最有用的表格。单次推理成本主要由 replica-hours 主导，而不是按请求计费。举个例子，一个 Qwen3-7B 聊天接口，业务时段大概 5 QPS：

| 组件 | 数量 | 月成本 |
|---|---|---|
| 最小副本基线 (2 × A10, 24/7) | 2 × 5 RMB/h × 720 h | ~7,200 RMB |
| 突发副本 (平均 9 小时 × 22 天多 3 个) | 3 × 5 × 198 h | ~2,970 RMB |
| 按请求费 (5 QPS × 86400 × 22) | 9.5 M req × 0.0001 RMB | ~950 RMB |
| OSS 带宽 (冷启动加载模型) | 14 GB × 30 冷启动 × 0.5 RMB/GB | ~210 RMB |
| **总计** | | **~11,330 RMB** |

这张表里有三个观察点，我花了不少时间才彻底内化：

1. **最小副本行占主导。** 非业务时段把 `min_replicas` 从 2 降到 1 能省 ~3,600 RMB/月。降到 0 能再省 ~3,600 RMB/月 *但* 会引入 60-120 秒冷启动。根据 SLA 来做 trade-off，别凭感觉求稳。
2. **LLM 负载下按请求费可忽略。** 这对高 QPS 分类（1000+ QPS）很重要，那里按百万计费可能占大头。对于聊天/生成，直接忽略。
3. **规模化下冷启动带宽不可忽视。** 30 次冷启动每次 14 GB 就是 420 GB 的 OSS 读流量——按跨区 0.5 RMB/GB 算，这是实打实的钱。把权重 bake 进镜像（见上文 Cold Start 章节），这行成本就近乎归零。

我评估新服务时都用这个公式：

```
monthly_cost = min_replicas × replica_price_per_hour × 720
             + avg_burst_replicas × replica_price_per_hour × business_hours_month
             + total_requests × per_request_fee
             + cold_starts × model_size_gb × oss_read_price
```

部署前*先*把数字填进去。因为有人默认 `min_replicas=10` 导致一个“小”部署变成意外 $2k/月的情况时有发生。控制台点 Deploy 时会显示估价——记得看一眼。

## 下一篇预告

第 5 篇收尾，我会诚实聊聊 **Designer** 和 **Model Gallery** —— 这两个零/低代码界面。大多数工程师不会首选它们，但用对了能值回票价，而且确实有一类特定任务，它们显然是正确答案。