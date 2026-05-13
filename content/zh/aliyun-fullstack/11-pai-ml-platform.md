---
title: "阿里云全栈实战（十一）：PAI 打造机器学习平台"
date: 2026-05-08 09:00:00
tags:
  - Alibaba Cloud
  - PAI
  - Machine Learning
  - DSW
  - EAS
  - Cloud Computing
categories: Cloud Computing
lang: zh
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 11
description: "阿里云完整ML平台：PAI-DSW笔记本、PAI-DLC分布式训练、PAI-EAS模型服务、Designer可视化工作流、Model Gallery。端到端训练和部署自定义模型。"
disableNunjucks: true
translationKey: "aliyun-fullstack-11"
---
单卡跑模型很有趣，但要稳定支撑每秒 1000 个请求才是从实验到生产的真正跨越；PAI 同时覆盖这两类场景。

PAI （Platform for AI）是阿里云的托管式机器学习平台——严格来说，并非单一产品，而是五个独立子产品共享同一控制台的集合。 Notebook 用于交互式探索，分布式训练支撑规模化训练，模型服务承载生产部署，可视化流水线面向拖拽式建模用户，模型库提供开源模型的一键部署能力。历经十八个月的真实 LLM 负载验证，各组件表现不一：EAS 表现优秀，Designer 基本可用，理清协同机制后，整体效能显著超越各组件能力的简单叠加。

本文将对 PAI 进行广度优先的概览。如果你想要深度优先的处理——比如实例选型策略、 DLC 抢占式实例生存指南、 EAS 冷启动缓解——那边有个专门的 [PAI 系列](/zh/aliyun-pai/01-platform-overview/)，五篇文章深挖每个子产品。本文聚焦核心目标：帮助你快速理解 PAI 是什么、各组件的适用场景，以及如何完成端到端的模型训练与部署。


## PAI 平台概览

PAI 全称 Platform for AI，其命名体现了通用性，覆盖从交互式实验到生产服务的整个机器学习生命周期。其他云上最接近的对标是 AWS SageMaker、Azure Machine Learning 和 GCP Vertex AI，但这种对比仅具参考性。 SageMaker 将 Notebook、训练和推理端点整合为相对一体化的体验；而 PAI 采用模块化设计，各子产品具备独立的资源模型、计费方式和 SDK 接口，支持按需选用。

![PAI 平台组件概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_pai_components.png)

你实际会接触到的五个组件如下：

| 组件 | 功能 | SageMaker 对应项 |
|---|---|---|
| **PAI-DSW** | 云上的 Jupyter/VSCode，带 GPU、预建镜像、 OSS 挂载 | SageMaker Studio / Notebook Instances |
| **PAI-DLC** | 托管式分布式训练任务（多卡、多节点） | SageMaker Training Jobs |
| **PAI-EAS** | 模型服务，支持自动扩缩容、蓝绿部署、流量拆分 | SageMaker Endpoints |
| **PAI-Designer** | 拖拽式可视化 ML 流水线构建器 | SageMaker Pipelines (visual mode) |
| **PAI-QuickStart** | 从模型库一键部署开源模型 | SageMaker JumpStart |

一种直观的理解方式是：代码通过 DSW → DLC → EAS 流程逐步走向生产就绪，而 Designer 和 QuickStart 提供了跳过中间环节的快捷路径。

```text
    DSW              DLC              EAS
     |                |                |
 [explore]     [train at scale]    [serve]
     \               |               /
      \              |              /
       +------  OSS / NAS  --------+
                     |
                 GPU ECS pool
```

PAI 不持有用户数据，数据集、checkpoint 和模型 artifact 均存储在 OSS 或 NAS 中；它为你编排 GPU 计算——DSW Notebook 启动时，会启动一台真实的 GPU ECS 实例；EAS 端点扩容时，会启动真实的 GPU pod。相比直接使用裸 ECS，PAI 的优势在于预装 CUDA/PyTorch 等镜像、自动挂载存储、内置监控仪表盘，并且按秒计费（而非按小时）。

### PAI 与 SageMaker：主要区别

如果你是从 AWS 过来的，这些区别可能会让你感到困扰或惊喜：

| Aspect | PAI | SageMaker |
|---|---|---|
| **Pricing model** | 按实际 GPU 实例秒计费（你能看到 ECS SKU） | 抽象的 "ml.p3.2xlarge" 定价，通常更高 |
| **Container freedom** | DLC 和 EAS 支持完整 Docker 镜像；自带任何框架 | 对框架和入口点更有主见（限制更多） |
| **GPU availability** | cn-shanghai 和 cn-hangzhou 库存最足；有 A100/H800 | us-east-1, us-west-2 可用性更好 |
| **Spot training** | DLC 支持抢占式实例，约 40% 折扣 | SageMaker 托管抢占式训练，折扣类似 |
| **Model gallery** | Qwen 系列，中国开源模型， plus 国际模型 | JumpStart 国际模型选择更广 |
| **SDK maturity** | Python SDK 功能可用但文档落后于中文版 | 成熟 SDK，文档 extensive |

最实质的区别在于： PAI 直接暴露底层 ECS 实例规格。选 DSW 实例时，你选的是 `ecs.gn7i-c8g1.2xlarge`（1x A10, 24 GB）。提交 DLC 任务时，你指定具体的 GPU SKU。这种透明度让成本估算变得直白——你用 ECS 的价格计算器也能算 PAI。

## PAI-DSW：交互式笔记本

大多数 ML 工作在 PAI 上都是从 DSW（Data Science Workshop）开始的。DSW 是运行在 GPU ECS 实例上的 JupyterLab 和浏览器版 VSCode，由 PAI 管理，其卖点在于跳过 CUDA/cuDNN/PyTorch 的安装步骤，大约 90 秒即可获得一个可用的 GPU 环境。

![DSW 笔记本工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_dsw_workflow.png)

### 何时使用 DSW

- 交互式探索、 EDA、画图
- 用 `pdb` 和真 GPU 调试模型
- 几小时内的单卡训练
- 写最终提交给 DLC 的训练脚本
- 部署到 EAS 前迭代推理代码

不要用 DSW 进行多卡训练（这是 DLC 的职责）、超过 8 小时的无人值守任务（空闲时会关闭）或生产推理（这是 EAS 的职责）。

### GPU 实例选项

| Instance | GPU | VRAM | vCPU | RAM | Best for |
|---|---|---|---|---|---|
| `ecs.gn7i-c8g1.2xlarge` | 1x A10 | 24 GB | 8 | 30 GB | 原型验证、微调 7B 模型 |
| `ecs.gn7e-c12g1.3xlarge` | 1x A100 40 GB | 40 GB | 12 | 93 GB | 13B 模型、更大规模的微调 |
| `ecs.gn8v-c8g1.2xlarge` | 1x H800 80 GB | 80 GB | 8 | 188 GB | 70B 推理 (int4)，贵 |
| `ecs.g7.xlarge` | None (CPU) | - | 4 | 16 GB | EDA、数据预处理，不需要 GPU |

我遵循的模式是：数据 prep 和 EDA 先用 CPU 实例，真要到调用 `.cuda()` 的时候再切 GPU 实例。 PAI 允许你 stop 一个 DSW 实例，然后用不同 SKU restart 它。

> **成本陷阱：** 设好自动 shutdown 计时器。每个 DSW 实例都有个“空闲 shutdown"旋钮——默认 1 小时。开发工作我推到 30 分钟。周一早上发现忘了关的 A100 实例计了一周末费，这种亏我可不想再吃第二次。

### 预构建镜像

DSW 自带镜像，这样你就不用花 20 分钟在 `pip install torch` 上：

| Image | Contents | Use case |
|---|---|---|
| `pytorch2.3-gpu-py310-cu124` | PyTorch 2.3, CUDA 12.4, Python 3.10, Transformers | 通用深度学习 |
| `tensorflow2.15-gpu-py310-cu121` | TensorFlow 2.15, CUDA 12.1, Keras | 基于 TF 的工作流 |
| `modelscope1.17-py310-cu124` | ModelScope SDK, Qwen 支持， DashScope 客户端 | 阿里模型生态 |
| `custom` | 来自 ACR 的自有 Docker 镜像 | 完全控制 |

### 创建 DSW 实例

通过控制台：**PAI Console** > **DSW** > **Create Instance** > 选 region、实例类型、镜像、存储。通过 SDK：

```python
from pai.session import setup_default_session
from pai.workspace import Workspace

# Configure session
setup_default_session(region_id="cn-shanghai")

# List available instance types for DSW
import json
from alibabacloud_paistudio20220112.client import Client
from alibabacloud_tea_openapi.models import Config

config = Config(
    access_key_id="<YOUR_AK>",
    access_key_secret="<YOUR_SK>",
    region_id="cn-shanghai",
    endpoint="pai.cn-shanghai.aliyuncs.com"
)
client = Client(config)

# Create a DSW instance
from alibabacloud_paistudio20220112.models import CreateInstanceRequest

request = CreateInstanceRequest(
    instance_name="dev-notebook",
    ecs_spec="ecs.gn7i-c8g1.2xlarge",
    image_id="pytorch2.3-gpu-py310-cu124",
    workspace_id="<YOUR_WORKSPACE_ID>",
    datasets=[{
        "dataset_id": "<OSS_DATASET_ID>",
        "mount_path": "/mnt/data"
    }]
)
response = client.create_instance(request)
print(f"Instance ID: {response.body.instance_id}")
```

### 存储：关键部分

存储是关键部分。新 PAI 用户常见的错误是：训练几小时后，实例重启会导致所有数据丢失。DSW 实例有一个系统盘，在重启时会被重置。因此，你需要将所有重要数据存放在以下位置：

1. **OSS** -- 在 `/mnt/data` 挂载 OSS bucket。从这里读训练数据，往这里写 checkpoint。
2. **NAS** -- 挂载 NAS 文件系统获取 POSIX 语义。更适合随机访问负载（许多小文件）。
3. **持久盘** -- 实例 restart 后依然存在的云盘。限 500 GB，只能挂一个实例。

```bash
# Inside a DSW terminal -- verify OSS mount
ls /mnt/data/
# Should show your OSS bucket contents

# Save a checkpoint to OSS (it persists)
python train.py --output_dir /mnt/data/checkpoints/run-001/

# Bad: saving to /root/ (will be lost on restart)
python train.py --output_dir /root/checkpoints/  # DON'T
```

想要完整的 DSW 深挖——镜像选择、 SSH 隧道、 GPU 显存 profiling——去看 [PAI Part 2](/zh/aliyun-pai/02-pai-dsw-notebook/)。
## PAI-DLC：分布式训练

当单卡无法满足需求时，可以考虑使用 DLC (Deep Learning Container)。 DLC 是一个托管的批处理系统，你只需提供容器镜像、命令、资源规格和数据挂载点。 DLC 会将任务调度到 GPU 集群上，处理节点间的网络（如果有 RDMA 就使用 RDMA），运行你的代码，流式输出日志，并在完成后自动清理。

![DLC 分布式训练模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_dlc_distributed.png)

![PAI 分布式训练流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_training_pipeline.png)

### 何时从 DSW 转到 DLC

何时应从 DSW 切换到 DLC？满足以下任一条件即可：
- 你需要不止一张 GPU
- 任务需要无人值守运行超过 4 小时
- 你想要节点间的 RDMA/NCCL 来加速梯度同步
- 你想用抢占式实例（spot instances）省钱
- 你在跑超参数搜索

切换过程通常很无痛，因为 DLC 接受的 Docker 镜像跟你之前在 DSW 里用的一样。

### 支持的框架

| Framework | Job type | Use case |
|---|---|---|
| **PyTorch DDP** | PyTorchJob | 标准分布式训练，默认选项 |
| **DeepSpeed** | PyTorchJob | 大模型的 ZeRO 优化 |
| **Megatron-LM** | PyTorchJob | 预训练的张量/流水线并行 |
| **TensorFlow** | TFJob | TF 分布式策略 |
| **Horovod** | MPIJob | Ring-allreduce （遗留方案，大多已被 DDP 取代） |
| **Custom** | ElasticBatchJob | 任意框架，你自己管理分布式初始化 |

现在的活儿，`PyTorchJob` 能覆盖 95% 的场景。 DLC 会自动在每个容器里设置 `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, 和 `LOCAL_RANK` 环境变量，所以 `torchrun` 和 `torch.distributed.init_process_group` 拿来就能用。

### 提交训练任务

下面是一个完整的 DLC 任务配置。我们用 2 个节点共 8 张卡，微调一个 Qwen-2.5-7B 模型：

```yaml
# dlc_job.yaml
apiVersion: training.pai.alibaba.com/v1
kind: PyTorchJob
metadata:
  name: qwen25-7b-sft-001
  namespace: default
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: pai-image-pytorch:2.3-gpu-py310-cu124
              command:
                - torchrun
                - --nproc_per_node=4
                - --nnodes=2
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=$(MASTER_PORT)
                - train_sft.py
                - --model_name_or_path=/mnt/data/models/Qwen2.5-7B
                - --data_path=/mnt/data/datasets/sft-v1/
                - --output_dir=/mnt/data/checkpoints/qwen25-7b-sft-001/
                - --num_train_epochs=3
                - --per_device_train_batch_size=4
                - --gradient_accumulation_steps=4
                - --learning_rate=2e-5
                - --bf16=true
                - --deepspeed=ds_config.json
              resources:
                limits:
                  nvidia.com/gpu: 4
                  cpu: "32"
                  memory: 256Gi
              volumeMounts:
                - name: data
                  mountPath: /mnt/data
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: oss-pvc
    Worker:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: pai-image-pytorch:2.3-gpu-py310-cu124
              command:
                - torchrun
                - --nproc_per_node=4
                - --nnodes=2
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=$(MASTER_PORT)
                - train_sft.py
                - --model_name_or_path=/mnt/data/models/Qwen2.5-7B
                - --data_path=/mnt/data/datasets/sft-v1/
                - --output_dir=/mnt/data/checkpoints/qwen25-7b-sft-001/
                - --num_train_epochs=3
                - --per_device_train_batch_size=4
                - --gradient_accumulation_steps=4
                - --learning_rate=2e-5
                - --bf16=true
                - --deepspeed=ds_config.json
              resources:
                limits:
                  nvidia.com/gpu: 4
                  cpu: "32"
                  memory: 256Gi
              volumeMounts:
                - name: data
                  mountPath: /mnt/data
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: oss-pvc
```

用 SDK 提交：

```python
from pai.estimator import Estimator

est = Estimator(
    image_uri="pai-image-pytorch:2.3-gpu-py310-cu124",
    command="torchrun --nproc_per_node=4 train_sft.py",
    instance_type="ecs.gn7i-c16g1.4xlarge",  # 4x A10 per node
    instance_count=2,                          # 2 nodes = 8 GPUs total
    source_dir="./training_code/",
    hyperparameters={
        "model_name_or_path": "/mnt/data/models/Qwen2.5-7B",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "bf16": True,
    },
    input_channels={
        "data": "oss://my-bucket/datasets/sft-v1/",
        "model": "oss://my-bucket/models/Qwen2.5-7B/"
    },
    output_path="oss://my-bucket/checkpoints/",
    spot_instance=True,  # Use spot for ~40% savings
)

est.fit()
```

### 使用竞价实例节省成本

DLC 支持抢占式实例（spot instances），大约可以节省 40% 的成本。代价是任务可能随时被中断，但会提前 5 分钟通知。解决方法是频繁保存 checkpoint，并在重启后从最新的 checkpoint 恢复。

```python
# In your training script -- checkpoint every 500 steps
if global_step % 500 == 0:
    save_path = f"/mnt/data/checkpoints/step-{global_step}/"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(optimizer.state_dict(), f"{save_path}/optimizer.pt")
    torch.save(lr_scheduler.state_dict(), f"{save_path}/scheduler.pt")
    print(f"Checkpoint saved at step {global_step}")

# On resume -- find the latest checkpoint
import glob
checkpoints = sorted(glob.glob("/mnt/data/checkpoints/step-*"))
if checkpoints:
    latest = checkpoints[-1]
    model = AutoModelForCausalLM.from_pretrained(latest)
    print(f"Resumed from {latest}")
```

想要完整的 DLC 治疗方案——RDMA 配置、 DeepSpeed ZeRO  configs、抢占式实例中断处理——请看 [PAI Part 3](/zh/aliyun-pai/03-pai-dlc-distributed-training/)。

## PAI-EAS：模型服务

EAS (Elastic Algorithm Service) 才是 PAI 真正的价值所在。 DSW 笔记本每小时只需几块钱。 DLC 任务是一次性投入。 EAS 端点需要 24/7 在线，能够应对流量峰值，闲时自动缩容，支持蓝绿部署，并且在市场团队上午 10 点向百万用户发送推送时保持稳定。

![EAS 模型服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_eas_serving.png)

### 部署模式

EAS 提供两种模式：

**Image mode** -- 你推一个 Docker 镜像，里面爱装什么 HTTP 服务器都行（FastAPI, Triton, vLLM）。 EAS 负责跑容器、路由流量、扩缩副本。容器里的一切你自己管。这是跑 LLM 服务的正确姿势。

**Processor mode** -- 你写个 Python 类，实现 `initialize()` 和 `process()` 方法。 EAS 提供 HTTP 服务器和路由。代码少，控制权也少。跑跑 scikit-learn 模型和轻量级分类器还行。

### 支持的服务框架

| Framework | Best for | GPU required |
|---|---|---|
| **vLLM** | LLM 推理 (Qwen, LLaMA, Mistral) | Yes |
| **Triton Inference Server** | 多模型服务、批处理、集成流水线 | Optional |
| **TensorFlow Serving** | TF SavedModel 格式 | Optional |
| **TorchServe** | 带自定义 handler 的 PyTorch 模型 | Optional |
| **ONNX Runtime** | 跨框架优化推理 | Optional |
| **Custom Docker** | 其他任何情况 | Your choice |

### 创建 EAS 服务

下面演示怎么用 Image 模式部署一个微调后的 Qwen-7B 模型，搭配 vLLM：

```python
# deploy_model.py
from pai.model import Model, InferenceSpec, container_serving_spec
from pai.session import setup_default_session

setup_default_session(region_id="cn-shanghai")

# Define the serving spec
inf_spec = container_serving_spec(
    image_uri="pai-image-vllm:0.6-cu124-py310",
    command=(
        "python -m vllm.entrypoints.openai.api_server "
        "--model /mnt/model "
        "--served-model-name qwen-7b-prod "
        "--port 8000 --host 0.0.0.0 "
        "--max-model-len 4096 "
        "--gpu-memory-utilization 0.9 "
        "--tensor-parallel-size 1"
    ),
    port=8000,
    health_check="/health",
)

model = Model(
    model_data="oss://my-bucket/checkpoints/qwen25-7b-sft-001/checkpoint-final/",
    inference_spec=inf_spec,
)

predictor = model.deploy(
    service_name="qwen-7b-prod",
    instance_type="ecs.gn7i-c8g1.2xlarge",  # 1x A10
    instance_count=1,
    options={
        "metadata.rpc.keepalive": 10000,
        "metadata.min_replica": 1,
        "metadata.max_replica": 4,
        "metadata.auto_scale.enabled": True,
        "metadata.auto_scale.metric": "QPS",
        "metadata.auto_scale.target": 50,
    }
)

print(f"Endpoint: {predictor.endpoint}")
print(f"Token: {predictor.access_token}")
```

### 测试端点

部署好后， EAS 会给你一个 HTTPS 端点和 access token：

```bash
# Test with curl
curl -X POST "https://your-service-id.cn-shanghai.pai-eas.aliyuncs.com/api/predict" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b-prod",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain gradient descent in one paragraph."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

```python
# Test with Python
import requests

endpoint = "https://your-service-id.cn-shanghai.pai-eas.aliyuncs.com/api/predict"
token = "<ACCESS_TOKEN>"

response = requests.post(
    endpoint,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen-7b-prod",
        "messages": [
            {"role": "user", "content": "What is PAI-EAS?"}
        ],
        "max_tokens": 128
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### 自动扩展

EAS 的自动扩缩容才是省钱的关键。你配置一个目标指标， EAS 会自动增减副本：

| Metric | When to use |
|---|---|
| **QPS** (queries per second) | 每次请求成本可预测的 API 端点 |
| **GPU utilization** | 计算密集型推理（图像生成、大 LLM） |
| **Pending requests** | 请求延迟波动极大的负载 |

```json
{
  "metadata.auto_scale.enabled": true,
  "metadata.auto_scale.metric": "QPS",
  "metadata.auto_scale.target": 50,
  "metadata.min_replica": 1,
  "metadata.max_replica": 8,
  "metadata.auto_scale.scale_in_cooldown": 300,
  "metadata.auto_scale.scale_out_cooldown": 60
}
```

不对称的冷却时间很重要：扩容要快（60 秒），因为你在丢请求；缩容要慢（300 秒），因为你不想在正常流量波动时在 1 到 4 个副本之间反复横跳。

### 蓝绿部署和 A/B 测试

EAS 支持在服务版本间拆分流量。部署一个新版本跟旧版本并存，然后逐渐切换流量：

```bash
# Deploy v2 alongside v1
pai eas update qwen-7b-prod \
  --canary-image "pai-image-vllm:0.7-cu124-py310" \
  --canary-model "oss://my-bucket/checkpoints/qwen25-7b-sft-002/" \
  --canary-weight 10   # 10% of traffic goes to v2

# Monitor metrics, then increase if v2 looks good
pai eas update qwen-7b-prod --canary-weight 50

# Full rollover
pai eas update qwen-7b-prod --canary-weight 100

# Or rollback
pai eas update qwen-7b-prod --canary-weight 0
```

想要完整的 EAS 深度解析——冷启动缓解、 warm pool  sizing、 TPS  dashboard 里的坑——请看 [PAI Part 4](/zh/aliyun-pai/04-pai-eas-model-serving/)。
## Model Gallery 模型广场

Model Gallery 其实就是 PAI 的模型中心。这里收录了一系列预训练模型，你可以一键部署到 EAS，或者把它们当作微调的起点。你可以把它当成 Hugging Face Hub 用，但它深度集成了 PAI 的计算和推理基础设施，省心很多。

![适用于机器学习工作负载的 GPU 实例对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_gpu_comparison.png)

### 可用模型

广场里既收录了阿里自家的模型，也有社区热门的开源模型：

| Category | Models | Notes |
|---|---|---|
| **Qwen family** | Qwen2.5-7B/14B/32B/72B, Qwen2.5-Coder, Qwen2.5-Math | 亲儿子，支持最好 |
| **LLaMA family** | LLaMA-3.1-8B/70B, LLaMA-3.2-1B/3B | 社区维护镜像 |
| **Stable Diffusion** | SDXL, SD 3.5, FLUX | 图像生成 |
| **Whisper** | whisper-large-v3 | 语音转文本 |
| **Embedding models** | GTE-Qwen2, BGE-M3 | RAG 检索 |
| **Specialized** | ChatGLM, Yi, Baichuan, DeepSeek | 中文优化 |

### 一键部署

在 Model Gallery 控制台操作非常简单：

1. 浏览或搜索模型
2. 点击 **Deploy**
3. 选择实例规格（广场会根据模型大小推荐）
4. 配置自动伸缩参数
5. 点击 **Create Service**

后台已经预配好了 Docker 镜像，自动从 ModelScope 或 OSS 下载模型，连 serving 命令都配好了（LLM 用 vLLM，视觉模型用 Triton），健康检查也一并设置妥当。通常不到 3 分钟，一个 Qwen2.5-7B 的 endpoint 就 ready 了。

### 基于模型广场微调

广场也支持微调。选一个基座模型，指向你在 OSS 上的数据集，广场会生成一个 DLC 训练任务，默认参数都配得很合理：

```python
# Programmatic fine-tuning from Model Gallery
from pai.model import RegisteredModel

# Get the model from gallery
base_model = RegisteredModel(
    model_name="Qwen2.5-7B",
    model_provider="pai"
)

# Fine-tune with your data
training_job = base_model.fine_tune(
    training_data="oss://my-bucket/datasets/sft-v1/",
    instance_type="ecs.gn7i-c16g1.4xlarge",  # 4x A10
    instance_count=1,
    hyperparameters={
        "epochs": 3,
        "learning_rate": 2e-5,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
    output_path="oss://my-bucket/fine-tuned/qwen-7b-custom/"
)

training_job.wait()
print(f"Fine-tuned model: {training_job.output_path}")
```

默认采用 LoRA （Low-Rank Adaptation）而不是全量微调，这对大多数场景来说是最优解——GPU 耗时能省 10 倍，而针对特定任务的 adapter 效果差异几乎可以忽略不计。

## Designer：可视化 ML 工作流

PAI-Designer （前身是 PAI-Studio）是一个拖拽式的 ML 流水线构建工具。你可以通过可视化方式连接组件：数据源、预处理、特征工程、算法、评估、部署。每个组件都是一个容器，运行在托管计算资源上。

### 什么时候用 Designer

Designer 适合以下场景：
- **表格数据 ML** -- 结构化数据的分类、回归、聚类。内置算法（XGBoost、 LightGBM、逻辑回归、 k-means）覆盖了 80% 的传统 ML 需求。
- **非程序员** -- 数据分析师和业务用户，他们擅长思考流水线逻辑，但不写 Python 代码。
- **可复现实验** -- 每次流水线运行都有版本记录和日志。你可以对比同一数据集上不同超参的运行 A 和运行 B。
- **ETL + 训练 + 评估 + 部署** 作为一个可调度的整体单元。

Designer 不适合以下场景：
- **深度学习** -- 内置的神经网络组件功能有限。写代码请用 DSW，训练请用 DLC。
- **LLM 负载** -- 原生不支持 Transformer 训练或推理服务。
- **复杂自定义逻辑** -- 如果预处理需要 200 行 Python 代码，在 Designer 里写代码组件会比直接用脚本更折磨。

### 内置算法

| Category | Algorithms |
|---|---|
| **分类** | XGBoost, LightGBM, 随机森林, 逻辑回归, SVM, KNN |
| **回归** | XGBoost, LightGBM, 线性回归, GBDT |
| **Clustering** | K-Means, DBSCAN |
| **NLP** | 文本分类（基于 BERT），分词，TF-IDF |
| **推荐** | 协同过滤, ALS |
| **特征工程** | 归一化, 独热编码, 特征哈希, PCA |
| **评估** | AUC, 准确率, RMSE, 混淆矩阵, 提升图 |

### Designer 还是写代码？我的决策树

1. 模型是 Transformer 或 Diffusion 吗？ **写代码**（DSW/DLC）。
2. 数据是表格且小于 100 GB 吗？ **Designer** 是强候选。
3. 流水线需要定时运行（比如每天重训）吗？ **Designer** -- 原生支持调度。
4. 维护流水线的人是数据科学家还是业务分析师？如果是分析师，**Designer**。
5. 这是一次性实验吗？ **写代码** -- 迭代更快。

关于 Designer 和 QuickStart 的对比，详见 [PAI Part 5](/zh/aliyun-pai/05-pai-designer-vs-quickstart/)。

## PAI + OSS + DashScope 集成

PAI 不是孤立存在的。它连接 OSS 做存储，连接 DashScope 调用模型 API，并接入阿里云生态的网络、安全和监控体系。搞清楚数据流转能少踩很多坑。

![PAI 完整集成流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-fullstack/11-pai-ml-platform/11_integration_flow.png)

### 完整数据流转

```text
  [Your local machine]
        |
        | (upload training data)
        v
  [OSS bucket: my-bucket/datasets/]
        |
        | (mount as /mnt/data in DSW/DLC)
        v
  [PAI-DSW: explore data, write training script]
        |
        | (submit job to DLC)
        v
  [PAI-DLC: distributed training, 8x GPU]
        |
        | (write checkpoints to OSS)
        v
  [OSS bucket: my-bucket/checkpoints/]
        |
        | (deploy to EAS)
        v
  [PAI-EAS: model serving endpoint]
        |
        | (HTTPS inference API)
        v
  [Your application / users]
```

### 从 OSS 读取训练数据

OSS 是 PAI 负载的主要数据存储。你把数据集上传到 OSS，在 DSW/DLC 中挂载，然后把结果写回去。关于模型 artifacts 的 bucket 配置和生命周期策略，详见 [Part 4: OSS Storage](/zh/aliyun-fullstack/04-oss-storage/)。

```python
# Upload training data to OSS (from your local machine)
import oss2

auth = oss2.Auth("<AK>", "<SK>")
bucket = oss2.Bucket(auth, "https://oss-cn-shanghai.aliyuncs.com", "my-bucket")

# Upload a dataset directory
import os
local_dir = "./datasets/sft-v1/"
for root, dirs, files in os.walk(local_dir):
    for fname in files:
        local_path = os.path.join(root, fname)
        oss_key = f"datasets/sft-v1/{os.path.relpath(local_path, local_dir)}"
        bucket.put_object_from_file(oss_key, local_path)
        print(f"Uploaded: {oss_key}")
```

```bash
# Or use ossutil (faster for large uploads)
ossutil cp -r ./datasets/sft-v1/ oss://my-bucket/datasets/sft-v1/ \
  --parallel 10 \
  --part-size 104857600
```

### 在 PAI 流水线中调用 DashScope

DashScope （Qwen API 网关，详见 [Part 10](/zh/aliyun-fullstack/10-bailian-llm/)）可以在 PAI 负载内部调用，用于数据标注、合成数据生成或 Embedding 计算等任务：

```python
# Inside a DSW notebook or DLC job -- call DashScope for embeddings
from dashscope import TextEmbedding

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using DashScope API from within PAI."""
    response = TextEmbedding.call(
        model="text-embedding-v3",
        input=texts,
        dimension=1024,
    )
    return [item["embedding"] for item in response.output["embeddings"]]

# Generate embeddings for your training data
import json
with open("/mnt/data/datasets/corpus.jsonl") as f:
    texts = [json.loads(line)["text"] for line in f]

# Batch process
batch_size = 25
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = get_embeddings(batch)
    all_embeddings.extend(embeddings)
    print(f"Processed {i + len(batch)}/{len(texts)}")
```

### 针对 Checkpoint 的 OSS 生命周期策略

训练任务会产生大量 Checkpoint。一个 7B 模型每个 Checkpoint 大约 14 GB。设置 OSS 生命周期规则可以自动清理旧文件：

```bash
# Set lifecycle rule: delete checkpoints older than 30 days
aliyun oss bucket-lifecycle --method put \
  --bucket my-bucket \
  --lifecycle-config '{
    "Rule": [{
      "ID": "cleanup-old-checkpoints",
      "Prefix": "checkpoints/",
      "Status": "Enabled",
      "Expiration": {"Days": 30},
      "Tags": [{"Key": "type", "Value": "intermediate-checkpoint"}]
    }]
  }'
```
## 解决方案：端到端的训练与部署

直接上全流程：从原始数据到生产环境的推理接口。我们要拿自定义的 Q&A 数据集去微调 Qwen2.5-7B 模型，然后部署成 REST API。这套打法我在 AI4Marketing 平台的生产环境里一直在用。

### 第一步：准备数据集

把数据整理成 JSONL 格式，得符合 Qwen 期待的 chat template：

```python
# prepare_dataset.py
import json

raw_data = [
    {
        "instruction": "What is the return policy for electronics?",
        "output": "Electronics can be returned within 15 days of purchase..."
    },
    {
        "instruction": "How do I track my order?",
        "output": "You can track your order by logging into your account..."
    },
    # ... thousands more
]

# Convert to chat format
with open("sft_data.jsonl", "w") as f:
    for item in raw_data:
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful customer service assistant."},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Prepared {len(raw_data)} training examples")
```

上传到 OSS：

```bash
ossutil cp sft_data.jsonl oss://my-bucket/datasets/customer-service/sft_data.jsonl
```

### 第二步：在 DSW 里探索

启动一个带 GPU 的 DSW 实例，挂载 OSS Bucket，先看看数据：

```python
# In a DSW Jupyter notebook
import json
from collections import Counter

# Load and inspect the dataset
with open("/mnt/data/datasets/customer-service/sft_data.jsonl") as f:
    data = [json.loads(line) for line in f]

print(f"Total examples: {len(data)}")
print(f"Avg user message length: {sum(len(d['messages'][1]['content']) for d in data) / len(data):.0f} chars")
print(f"Avg assistant message length: {sum(len(d['messages'][2]['content']) for d in data) / len(data):.0f} chars")

# Check for quality issues
short_responses = [d for d in data if len(d["messages"][2]["content"]) < 20]
print(f"Short responses (<20 chars): {len(short_responses)}")

# Quick test: load the base model and generate a response WITHOUT fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/mnt/data/models/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test the base model on a customer service question
messages = [
    {"role": "system", "content": "You are a helpful customer service assistant."},
    {"role": "user", "content": "What is the return policy for electronics?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print("Base model response:")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

基座模型给出的回答会比较泛。微调之后，它才能基于你们具体的业务政策给出回答。

### 第三步：编写训练脚本

```python
# train_sft.py
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def preprocess(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text, truncation=True, max_length=2048, padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True
        ),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

### 第四步：提交 DLC 训练任务

```python
# submit_training.py
from pai.estimator import Estimator

estimator = Estimator(
    image_uri="pai-image-pytorch:2.3-gpu-py310-cu124",
    command=(
        "pip install peft datasets && "
        "python train_sft.py "
        "--model_name_or_path /mnt/model/Qwen2.5-7B "
        "--data_path /mnt/data/sft_data.jsonl "
        "--output_dir /mnt/output/qwen-7b-cs/ "
        "--num_train_epochs 3 "
        "--per_device_train_batch_size 4 "
        "--gradient_accumulation_steps 4 "
        "--learning_rate 2e-5 "
        "--bf16 "
        "--lora_rank 16 "
        "--lora_alpha 32"
    ),
    source_dir="./",
    instance_type="ecs.gn7i-c16g1.4xlarge",  # 4x A10
    instance_count=1,
    input_channels={
        "data": "oss://my-bucket/datasets/customer-service/",
        "model": "oss://my-bucket/models/Qwen2.5-7B/",
    },
    output_path="oss://my-bucket/checkpoints/",
    spot_instance=True,
)

estimator.fit()
print(f"Training complete. Model at: {estimator.model_data}")
```

### 第五步：部署到 EAS

```python
# deploy_service.py
from pai.model import Model, container_serving_spec

inf_spec = container_serving_spec(
    image_uri="pai-image-vllm:0.6-cu124-py310",
    command=(
        "python -m vllm.entrypoints.openai.api_server "
        "--model /mnt/model "
        "--served-model-name customer-service "
        "--port 8000 --host 0.0.0.0 "
        "--max-model-len 2048 "
        "--gpu-memory-utilization 0.9"
    ),
    port=8000,
    health_check="/health",
)

model = Model(
    model_data="oss://my-bucket/checkpoints/qwen-7b-cs/",
    inference_spec=inf_spec,
)

predictor = model.deploy(
    service_name="customer-service-v1",
    instance_type="ecs.gn7i-c8g1.2xlarge",  # 1x A10
    instance_count=1,
    options={
        "metadata.min_replica": 1,
        "metadata.max_replica": 4,
        "metadata.auto_scale.enabled": True,
        "metadata.auto_scale.metric": "QPS",
        "metadata.auto_scale.target": 30,
    }
)

print(f"Service deployed!")
print(f"Endpoint: {predictor.endpoint}")
print(f"Token: {predictor.access_token}")
```

### 第六步：测试推理接口

```bash
# Quick smoke test
curl -X POST "${ENDPOINT}/v1/chat/completions" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "customer-service",
    "messages": [
      {"role": "system", "content": "You are a helpful customer service assistant."},
      {"role": "user", "content": "I want to return a laptop I bought 10 days ago."}
    ],
    "max_tokens": 256,
    "temperature": 0.3
  }' | jq .
```

```python
# Production client with error handling
import requests
import time

class CustomerServiceClient:
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })

    def ask(self, question: str, max_retries: int = 3) -> str:
        payload = {
            "model": "customer-service",
            "messages": [
                {"role": "system", "content": "You are a helpful customer service assistant."},
                {"role": "user", "content": question}
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }

        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=30
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

# Usage
client = CustomerServiceClient(
    endpoint="https://your-service.cn-shanghai.pai-eas.aliyuncs.com",
    token="<ACCESS_TOKEN>"
)
answer = client.ask("How do I track my order?")
print(answer)
```

### 第七步：配置自动伸缩与监控

```bash
# Monitor the service
pai eas list-services --region cn-shanghai

# Check service metrics
pai eas describe customer-service-v1 --region cn-shanghai

# View real-time logs
pai eas logs customer-service-v1 --region cn-shanghai --tail 100
```

EAS 集成了云监控用于告警。建议配置以下几项：
- **QPS 激增** -- 自动伸缩应该能扛住，但如果达到最大副本数得告警
- **错误率** -- 任何持续超过 1% 的 5xx 错误率都需要排查
- **延迟 p99** -- LLM 推理延迟是双峰分布的； p50 可能只有 200 ms，但 p99 可能高达 3 秒

### 成本算笔账

对于一个 7B 模型和 10,000 条训练样本，这套端到端流程的成本大概是这样：

| 步骤 | 资源 | 时长 | 预估成本 (CNY) |
|---|---|---|---|
| DSW 探索 | 1x A10 | 2 hours | ~30 |
| DLC 训练 (spot) | 4x A10 | 3 hours | ~180 (after spot discount) |
| EAS 服务 (1 副本) | 1x A10 | per month | ~2,200/month |
| EAS 服务 (自动伸缩 1-4) | 1-4x A10 | per month | ~2,200-8,800/month |
| OSS 存储 | 50 GB | per month | ~6/month |

推理服务的成本是大头。如果流量波动大， EAS 支持从 0 到 N 的自动伸缩（scale-to-zero），但这会带来 2-5 分钟的冷启动延迟。对于需要即时响应的生产服务，建议保持 `min_replica=1`。
## 核心要点

**PAI 其实是五个产品，不是一个。** 别混着用。 DSW 写 Notebook， DLC 跑训练， EAS 做服务， Designer 搞可视化流水线， QuickStart 一键部署模型。动手前先想清楚，到底哪个能解决你的问题。

**数据存在 OSS，别存在 PAI 里。** 所有的 checkpoint、数据集、模型 artifact 都必须落 OSS。 PAI 的计算资源是临时的——DSW 实例重启或者 DLC 任务跑完，没存进 OSS 的东西直接就没了。

**DSW 起步， DLC 训练， EAS 服务。** 代码成熟度是从左往右走的。先在 DSW 里交互式写训练脚本，再把多 GPU 任务提交给 DLC，最后把最终 checkpoint 部署到 EAS。全程同一个 Docker 镜像就能打通。

**EAS 自动伸缩是控制成本的关键杠杆。** GPU 闲置 20 小时和扛着 1000 QPS 的成本是一样的。配置自动伸缩时要用非对称冷却策略——扩容要快，缩容要慢。

**训练直接用 Spot 实例。** DLC 的 Spot 实例便宜大概 40%。勤打 checkpoint （每 500 步一次），就算任务被抢占也不会丢进度。

**Model Gallery 是捷径。** 如果你只需要 Qwen 或 LLaMA 的 endpoint，又不需要定制训练， Model Gallery 能让你 5 分钟内从零走到服务上线。在投入完整训练流水线之前，先用它做个评估。

想深入了解每个子产品，[PAI 系列](/zh/aliyun-pai/01-platform-overview/) 共有五篇文章：[DSW notebooks](/zh/aliyun-pai/02-pai-dsw-notebook/)、[DLC distributed training](/zh/aliyun-pai/03-pai-dlc-distributed-training/)、[EAS model serving](/zh/aliyun-pai/04-pai-eas-model-serving/) 以及 [Designer vs QuickStart](/zh/aliyun-pai/05-pai-designer-vs-quickstart/)。如果不想管理基础设施只想调 LLM API，去看 [Part 10: Bailian and DashScope](/zh/aliyun-fullstack/10-bailian-llm/)。

下一篇：[Article 12 -- Putting It All Together](/zh/aliyun-fullstack/12-production-architecture/)，我们会把本系列的所有内容组装成一个完整的生产架构。