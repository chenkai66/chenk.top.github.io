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
description: "阿里云完整 ML 平台：PAI-DSW 笔记本、PAI-DLC 分布式训练、PAI-EAS 模型服务、Designer 可视化工作流、Model Gallery。端到端训练和部署自定义模型。"
disableNunjucks: true
translationKey: "aliyun-fullstack-11"
---
单卡跑模型很有趣，但要稳定支撑每秒 1000 个请求，才是从实验迈向产品的关键一步。PAI 正好覆盖了这两个阶段。

PAI（Platform for AI）是阿里云的托管式机器学习平台。严格来说，它并非单一产品，而是五个独立子产品共享同一控制台的集合：Notebook 用于交互式探索，分布式训练服务支撑规模化训练，模型服务平台承载生产部署，可视化流水线面向偏好拖拽操作的用户，模型库则提供开源模型的一键部署能力。经过十八个月的真实 LLM 负载验证，各组件表现不一——EAS 表现优秀，Designer 基本够用；但一旦理清它们之间的协同机制，整体效能远超各部分之和。

本文将对 PAI 进行广度优先的概览。如果你想要深度优先的内容——比如实例选型策略、DLC 抢占式实例生存指南、EAS 冷启动缓解方案——可以参考专门的 [PAI 系列](/zh/aliyun-pai/01-platform-overview/)，其中五篇文章分别深入剖析每个子产品。本文的目标很明确：帮你快速理解 PAI 是什么、各组件适用场景，以及如何完成端到端的模型训练与部署。

## PAI 平台概览

PAI 全称 Platform for AI，名字虽泛，却名副其实——它覆盖了从交互式实验到生产服务的完整机器学习生命周期。其他云厂商中最接近的对标产品是 AWS SageMaker、Azure Machine Learning 和 GCP Vertex AI，但这种对比仅具参考意义。SageMaker 将 Notebook、训练和推理端点整合成相对一体化的体验；而 PAI 更加模块化，每个子产品都有独立的资源模型、计费方式和 SDK 接口，你可以按需单独使用任意一个。

![PAI 平台组件概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_pai_components.png)

你实际会用到的五个核心组件如下：

| 组件 | 功能 | SageMaker 对应项 |
|---|---|---|
| **PAI-DSW** | 云上的 Jupyter/VSCode，带 GPU、预建镜像、OSS 挂载 | SageMaker Studio / Notebook Instances |
| **PAI-DLC** | 托管式分布式训练任务（多卡、多节点） | SageMaker Training Jobs |
| **PAI-EAS** | 模型服务，支持自动扩缩容、蓝绿部署、流量拆分 | SageMaker Endpoints |
| **PAI-Designer** | 拖拽式可视化 ML 流水线构建器 | SageMaker Pipelines (visual mode) |
| **PAI-QuickStart** | 从模型库一键部署开源模型 | SageMaker JumpStart |

一种直观的理解方式是：代码沿着 DSW → DLC → EAS 的路径逐步走向生产就绪，而 Designer 和 QuickStart 则提供了跳过中间环节的快捷路径。

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

PAI 从不持有你的数据。数据集、checkpoint 和模型 artifact 都存储在 OSS 或 NAS 中。PAI 只负责为你编排 GPU 计算资源——当你启动一个 DSW Notebook 时，背后会拉起一台真实的 GPU ECS 实例；当 EAS 端点扩容时，也会启动真实的 GPU Pod。相比直接使用裸 ECS，PAI 的优势在于预装了 CUDA/PyTorch 等环境、自动挂载你的存储、提供监控仪表盘，并且按秒计费（而非按小时）。

### PAI 与 SageMaker：关键差异

如果你是从 AWS 过来的，以下几点可能会让你感到惊喜或需要适应：

| Aspect | PAI | SageMaker |
|---|---|---|
| **Pricing model** | 按实际 GPU 实例秒计费（你能看到具体的 ECS SKU） | 抽象的 "ml.p3.2xlarge" 定价，通常更高 |
| **Container freedom** | DLC 和 EAS 支持完整的 Docker 镜像，可自由引入任何框架 | 对框架和入口点限制较多，更“有主见” |
| **GPU availability** | cn-shanghai 和 cn-hangzhou 库存最足；A100/H800 可用 | us-east-1、us-west-2 可用性更好 |
| **Spot training** | DLC 支持抢占式实例，折扣约 40% | SageMaker 托管抢占式训练，折扣类似 |
| **Model gallery** | Qwen 系列、中国开源模型，以及部分国际模型 | JumpStart 的国际模型选择更广 |
| **SDK maturity** | Python SDK 功能可用，但文档更新滞后于中文版 | SDK 成熟，文档详尽 |

最实质的区别在于：PAI 直接暴露底层 ECS 实例规格。选 DSW 实例时，你看到的是 `ecs.gn7i-c8g1.2xlarge`（1×A10，24 GB）；提交 DLC 任务时，你也需指定具体的 GPU SKU。这种透明度让成本估算变得简单直接——你完全可以沿用 ECS 的价格计算器来估算 PAI 开销。

## PAI-DSW：交互式笔记本

大多数 ML 工作在 PAI 上都始于 DSW（Data Science Workshop）。它是在 GPU ECS 实例上运行的 JupyterLab 和浏览器版 VSCode，由 PAI 托管。其核心卖点是：跳过繁琐的 CUDA、cuDNN 和 PyTorch 安装过程，90 秒内即可获得一个可用的 GPU 开发环境。

![DSW 笔记本工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_dsw_workflow.png)

### 何时使用 DSW

- 交互式探索、EDA、绘图
- 使用 `pdb` 和真实 GPU 调试模型
- 几小时内完成的单卡训练
- 编写最终将提交给 DLC 的训练脚本
- 在部署到 EAS 前迭代推理逻辑

不要用 DSW 进行多卡训练（这是 DLC 的职责）、超过 8 小时的无人值守任务（空闲会自动关机），或用于生产推理（这是 EAS 的领域）。

### GPU 实例选项

| Instance | GPU | VRAM | vCPU | RAM | Best for |
|---|---|---|---|---|---|
| `ecs.gn7i-c8g1.2xlarge` | 1×A10 | 24 GB | 8 | 30 GB | 原型验证、微调 7B 模型 |
| `ecs.gn7e-c12g1.3xlarge` | 1×A100 40 GB | 40 GB | 12 | 93 GB | 13B 模型、更大规模微调 |
| `ecs.gn8v-c8g1.2xlarge` | 1×H800 80 GB | 80 GB | 8 | 188 GB | 70B 推理（int4），价格较高 |
| `ecs.g7.xlarge` | None (CPU) | - | 4 | 16 GB | EDA、数据预处理，无需 GPU |

我的习惯是：先用 CPU 实例做数据准备和 EDA，直到真正需要调用 `.cuda()` 时才切换到 GPU 实例。PAI 允许你停止 DSW 实例后，以不同规格重新启动。

> **成本陷阱：** 务必设置自动关机时间。每个 DSW 实例都有“空闲关机”选项，默认 1 小时。我通常设为 30 分钟。周一早上发现一台被遗忘的 A100 实例整个周末都在计费？这种教训，我不想再经历第二次。

### 预构建镜像

DSW 提供预装镜像，省去你花 20 分钟执行 `pip install torch` 的麻烦：

| Image | Contents | Use case |
|---|---|---|
| `pytorch2.3-gpu-py310-cu124` | PyTorch 2.3, CUDA 12.4, Python 3.10, Transformers | 通用深度学习 |
| `tensorflow2.15-gpu-py310-cu121` | TensorFlow 2.15, CUDA 12.1, Keras | 基于 TF 的工作流 |
| `modelscope1.17-py310-cu124` | ModelScope SDK、Qwen 支持、DashScope 客户端 | 阿里模型生态 |
| `custom` | 来自 ACR 的自有 Docker 镜像 | 完全控制 |

### 创建 DSW 实例

通过控制台：**PAI Console** > **DSW** > **Create Instance** > 选择地域、实例类型、镜像和存储。通过 SDK：

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

新用户最常见的错误是：辛苦训练数小时，实例重启后所有成果付诸东流。DSW 实例的系统盘会在重启时重置，因此所有需要保留的数据必须存放在以下位置：

1. **OSS** —— 挂载 OSS Bucket 到 `/mnt/data`，从此处读取训练数据，也将 checkpoint 写入此处。
2. **NAS** —— 挂载 NAS 文件系统以获得 POSIX 语义，更适合随机访问场景（如大量小文件）。
3. **持久盘** —— 一块随实例重启仍保留的云盘，最大 500 GB，仅能绑定一个实例。

```bash
# Inside a DSW terminal -- verify OSS mount
ls /mnt/data/
# Should show your OSS bucket contents

# Save a checkpoint to OSS (it persists)
python train.py --output_dir /mnt/data/checkpoints/run-001/

# Bad: saving to /root/ (will be lost on restart)
python train.py --output_dir /root/checkpoints/  # DON'T
```

若想深入了解 DSW —— 包括镜像选择、SSH 隧道、GPU 显存分析等 —— 请参阅 [PAI Part 2](/zh/aliyun-pai/02-pai-dsw-notebook/)。

## PAI-DLC：分布式训练

当单卡无法满足需求时，就该转向 DLC（Deep Learning Container）了。它是一个托管式批处理系统：你只需提供容器镜像、命令、资源规格和数据挂载点，DLC 便会将任务调度到 GPU 集群上，配置节点间网络（支持 RDMA 时会启用）、运行代码、实时输出日志，并在任务完成后自动清理资源。

![DLC 分布式训练模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_dlc_distributed.png)

![PAI 分布式训练流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_training_pipeline.png)

### 何时从 DSW 转到 DLC

满足以下任一条件，就该切换到 DLC：
- 需要超过一张 GPU
- 任务需无人值守运行超过 4 小时
- 希望利用 RDMA/NCCL 加速跨节点梯度同步
- 想通过抢占式实例节省成本
- 正在进行超参数搜索

迁移过程通常很顺畅，因为 DLC 可直接复用你在 DSW 中使用的 Docker 镜像。

### 支持的框架

| Framework | Job type | Use case |
|---|---|---|
| **PyTorch DDP** | PyTorchJob | 标准分布式训练，默认选择 |
| **DeepSpeed** | PyTorchJob | 大模型的 ZeRO 优化 |
| **Megatron-LM** | PyTorchJob | 预训练中的张量/流水线并行 |
| **TensorFlow** | TFJob | TensorFlow 分布式策略 |
| **Horovod** | MPIJob | Ring-allreduce（已逐渐被 DDP 取代） |
| **Custom** | ElasticBatchJob | 任意框架，自行管理分布式初始化 |

对于现代工作负载，`PyTorchJob` 覆盖了约 95% 的场景。DLC 会自动在每个容器中设置 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK` 和 `LOCAL_RANK` 环境变量，因此 `torchrun` 和 `torch.distributed.init_process_group` 开箱即用。

### 提交训练任务

以下是一个完整的 DLC 任务配置，用于在 2 个节点共 8 张 GPU 上微调 Qwen-2.5-7B 模型：

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

通过 SDK 提交：

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

### 使用抢占式实例节省成本

DLC 支持抢占式实例，折扣约 40%。代价是任务可能被中断，但会提前 5 分钟通知。应对方法是频繁保存 checkpoint，并在重启后从最新 checkpoint 恢复。

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

若想全面掌握 DLC —— 包括 RDMA 配置、DeepSpeed ZeRO 设置、抢占式中断处理等 —— 请参阅 [PAI Part 3](/zh/aliyun-pai/03-pai-dlc-distributed-training/)。

## PAI-EAS：模型服务

EAS（Elastic Algorithm Service）才是 PAI 真正体现价值的地方。DSW Notebook 每小时只需几元，DLC 任务是一次性支出，而 EAS 端点需要 7×24 小时在线：它必须能应对流量峰值、在低谷期自动缩容、支持蓝绿部署，并且在市场团队上午 10 点向百万用户推送消息时依然稳如泰山。

![EAS 模型服务架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_eas_serving.png)

### 部署模式

EAS 提供两种部署模式：

**Image mode** —— 你推送一个包含任意 HTTP 服务器（如 FastAPI、Triton、vLLM）的 Docker 镜像。EAS 负责运行容器、路由流量、扩缩副本，容器内部完全由你掌控。这是 LLM 服务的首选方式。

**Processor mode** —— 你只需编写一个包含 `initialize()` 和 `process()` 方法的 Python 类，EAS 会提供 HTTP 服务器和路由逻辑。代码量少，但控制力也弱，适合 scikit-learn 模型或轻量级分类器。

### 支持的服务框架

| Framework | Best for | GPU required |
|---|---|---|
| **vLLM** | LLM 推理（Qwen、LLaMA、Mistral） | Yes |
| **Triton Inference Server** | 多模型服务、批处理、集成流水线 | Optional |
| **TensorFlow Serving** | TF SavedModel 格式 | Optional |
| **TorchServe** | 带自定义 handler 的 PyTorch 模型 | Optional |
| **ONNX Runtime** | 跨框架优化推理 | Optional |
| **Custom Docker** | 其他任何场景 | Your choice |

### 创建 EAS 服务

以下是如何使用 Image 模式，通过 vLLM 部署一个微调后的 Qwen-7B 模型：

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

部署完成后，EAS 会提供一个 HTTPS 端点和访问令牌：

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

### 自动扩缩容

EAS 的自动扩缩容是控制成本的关键。你只需配置目标指标，EAS 便会自动调整副本数量：

| Metric | When to use |
|---|---|
| **QPS**（每秒查询数） | 请求成本可预测的 API 端点 |
| **GPU 利用率** | 计算密集型推理（如图像生成、大 LLM） |
| **待处理请求数** | 请求延迟波动极大的工作负载 |

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

非对称的冷却时间至关重要：扩容要快（60 秒），因为你正在丢弃请求；缩容要慢（300 秒），避免在正常流量波动时于 1 到 4 个副本之间反复震荡。

### 蓝绿部署与 A/B 测试

EAS 支持在不同服务版本间分配流量。你可以将新模型与旧版本并行部署，并逐步转移流量：

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

若想深入理解 EAS —— 包括冷启动缓解、warm pool 调优、TPS 仪表盘的“陷阱”等 —— 请参阅 [PAI Part 4](/zh/aliyun-pai/04-pai-eas-model-serving/)。

## Model Gallery 模型广场

Model Gallery 是 PAI 的模型中心，收录了一系列预训练模型，支持一键部署到 EAS，或作为微调起点。你可以把它看作 Hugging Face Hub，但它深度集成了 PAI 的计算与服务基础设施，使用更便捷。

![适用于机器学习工作负载的 GPU 实例对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_gpu_comparison.png)

### 可用模型

广场同时收录了阿里自研模型和热门开源模型：

| Category | Models | Notes |
|---|---|---|
| **Qwen family** | Qwen2.5-7B/14B/32B/72B、Qwen2.5-Coder、Qwen2.5-Math | 阿里亲儿子，支持最佳 |
| **LLaMA family** | LLaMA-3.1-8B/70B、LLaMA-3.2-1B/3B | 社区维护镜像 |
| **Stable Diffusion** | SDXL、SD 3.5、FLUX | 图像生成 |
| **Whisper** | whisper-large-v3 | 语音转文本 |
| **Embedding models** | GTE-Qwen2、BGE-M3 | RAG 检索 |
| **Specialized** | ChatGLM、Yi、Baichuan、DeepSeek | 中文优化 |

### 一键部署

在 Model Gallery 控制台操作非常简单：

1. 浏览或搜索模型
2. 点击 **Deploy**
3. 选择实例类型（系统会根据模型大小推荐）
4. 配置自动伸缩参数
5. 点击 **Create Service**

后台已预配置好 Docker 镜像、从 ModelScope/OSS 下载模型、服务启动命令（LLM 用 vLLM，视觉模型用 Triton）以及健康检查。通常不到 3 分钟，Qwen2.5-7B 的端点即可就绪。

### 基于模型广场微调

广场也支持微调。选择基座模型，指向 OSS 中的数据集，系统会自动生成一个 DLC 训练任务，并配好合理默认值：

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

默认采用 LoRA（Low-Rank Adaptation）而非全量微调，这对大多数场景是最优解——GPU 耗时可减少 10 倍，而任务特定 adapter 的效果差异几乎可忽略。

## Designer：可视化 ML 工作流

PAI-Designer（原 PAI-Studio）是一个拖拽式 ML 流水线构建工具。你可以通过可视化方式连接组件：数据源、预处理、特征工程、算法、评估、部署。每个组件都是一个容器，运行在托管计算资源上。

### 何时使用 Designer

Designer 适用于以下场景：
- **表格数据 ML** —— 结构化数据的分类、回归、聚类。内置算法（XGBoost、LightGBM、逻辑回归、k-means）覆盖了 80% 的传统 ML 需求。
- **非程序员** —— 数据分析师和业务用户，擅长流程思维但不写 Python。
- **可复现实验** —— 每次运行都有版本记录和日志，便于对比不同超参的效果。
- **ETL + 训练 + 评估 + 部署** 作为一个可调度的整体单元。

Designer 不适用于：
- **深度学习** —— 内置神经网络组件功能有限。建议在 DSW 编写代码，在 DLC 训练。
- **LLM 工作负载** —— 无原生 Transformer 训练或服务支持。
- **复杂自定义逻辑** —— 若预处理需 200 行 Python 代码，在 Designer 中反而更痛苦。

### 内置算法

| Category | Algorithms |
|---|---|
| **分类** | XGBoost、LightGBM、随机森林、逻辑回归、SVM、KNN |
| **回归** | XGBoost、LightGBM、线性回归、GBDT |
| **Clustering** | K-Means、DBSCAN |
| **NLP** | 文本分类（基于 BERT）、分词、TF-IDF |
| **推荐** | 协同过滤、ALS |
| **特征工程** | 归一化、独热编码、特征哈希、PCA |
| **评估** | AUC、准确率、RMSE、混淆矩阵、提升图 |

### Designer 还是写代码？我的决策树

1. 模型是 Transformer 或 Diffusion 吗？→ **写代码**（DSW/DLC）。
2. 数据是表格且小于 100 GB 吗？→ **Designer** 是强候选。
3. 流水线需定时运行（如每日重训）吗？→ **Designer**（原生支持调度）。
4. 维护者是数据科学家还是业务分析师？若是后者，→ **Designer**。
5. 这是一次性实验吗？→ **写代码**（迭代更快）。

关于 Designer 与 QuickStart 的对比，详见 [PAI Part 5](/zh/aliyun-pai/05-pai-designer-vs-quickstart/)。

## PAI + OSS + DashScope 集成

PAI 并非孤立存在。它与 OSS 对接存储，与 DashScope 联动调用模型 API，并融入阿里云的网络、安全与监控体系。理清数据流转路径，能大幅减少调试成本。

![PAI 完整集成流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_integration_flow.png)

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

OSS 是 PAI 工作负载的主要数据存储。你将数据集上传至 OSS，在 DSW/DLC 中挂载，并将结果写回。关于模型 artifacts 的 Bucket 配置与生命周期策略，详见 [Part 4: OSS Storage](/zh/aliyun-fullstack/04-oss-storage/)。

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

DashScope（Qwen API 网关，详见 [Part 10](/zh/aliyun-fullstack/10-bailian-llm/)）可在 PAI 任务内部调用，用于数据标注、合成数据生成或 Embedding 计算等任务：

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

训练任务会产生大量 checkpoint。一个 7B 模型每个 checkpoint 约 14 GB。建议设置 OSS 生命周期规则，自动清理旧文件：

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

下面展示完整流程：从原始数据到生产级推理接口。我们将使用自定义 Q&A 数据集微调 Qwen2.5-7B 模型，并部署为 REST API。这套方案已在 AI4Marketing 平台的生产环境中验证。

### 第一步：准备数据集

将数据整理为 JSONL 格式，符合 Qwen 所需的 chat template：

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

上传至 OSS：

```bash
ossutil cp sft_data.jsonl oss://my-bucket/datasets/customer-service/sft_data.jsonl
```

### 第二步：在 DSW 中探索

启动带 GPU 的 DSW 实例，挂载 OSS Bucket，初步查看数据：

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

基座模型的回答通常较泛化，微调后才能基于具体业务政策给出精准答案。

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

EAS 与云监控集成，建议设置以下告警：
- **QPS 激增** —— 自动伸缩应能应对，但若达到最大副本数需告警
- **错误率** —— 持续高于 1% 的 5xx 错误需立即排查
- **延迟 p99** —— LLM 推理延迟呈双峰分布；p50 可能仅 200 ms，但 p99 可达 3 秒

### 成本概算

对于 7B 模型和 10,000 条训练样本，端到端流程的成本大致如下：

| 步骤 | 资源 | 时长 | 预估成本 (CNY) |
|---|---|---|---|
| DSW 探索 | 1×A10 | 2 hours | ~30 |
| DLC 训练 (spot) | 4×A10 | 3 hours | ~180（含抢占折扣） |
| EAS 服务 (1 副本) | 1×A10 | per month | ~2,200/月 |
| EAS 服务 (自动伸缩 1–4) | 1–4×A10 | per month | ~2,200–8,800/月 |
| OSS 存储 | 50 GB | per month | ~6/月 |

推理服务成本占主导。若流量波动大，EAS 支持 scale-to-zero（从 0 扩容），但会带来 2–5 分钟冷启动延迟。对于需即时响应的生产服务，建议保持 `min_replica=1`。

## 总结

**PAI 实际上是五个产品，而非一个。** DSW 用于 Notebook，DLC 用于训练，EAS 用于服务，Designer 用于可视化流水线，QuickStart 用于一键部署。动手前先明确问题，再选择合适组件。

**数据必须存于 OSS，而非 PAI。** 所有 checkpoint、数据集和模型 artifact 都应落盘 OSS。PAI 的计算资源是临时的——DSW 实例重启或 DLC 任务结束，未存入 OSS 的数据将永久丢失。

**从 DSW 起步，经 DLC 训练，最终由 EAS 服务。** 代码成熟度沿此路径演进：先在 DSW 交互式编写训练脚本，再提交多 GPU 任务至 DLC，最后将最终 checkpoint 部署到 EAS。全程可复用同一 Docker 镜像。

**EAS 自动伸缩是控制成本的核心杠杆。** GPU 闲置 20 小时与承载 1000 QPS 的成本相同。配置自动伸缩时，务必采用非对称冷却策略——扩容要快，缩容要慢。

**训练务必使用抢占式实例。** DLC 的抢占实例便宜约 40%。每 500 步保存一次 checkpoint，即使任务被中断也不会丢失进度。

**Model Gallery 是快速通道。** 若只需 Qwen 或 LLaMA 的推理端点且无需定制训练，Model Gallery 能在 5 分钟内完成从零到上线。在投入完整训练流水线前，先用它快速评估效果。

若想深入每个子产品，[PAI 系列](/zh/aliyun-pai/01-platform-overview/) 包含五篇文章：[DSW notebooks](/zh/aliyun-pai/02-pai-dsw-notebook/)、[DLC distributed training](/zh/aliyun-pai/03-pai-dlc-distributed-training/)、[EAS model serving](/zh/aliyun-pai/04-pai-eas-model-serving/) 以及 [Designer vs QuickStart](/zh/aliyun-pai/05-pai-designer-vs-quickstart/)。若不想管理基础设施，只想调用 LLM API，请参阅 [Part 10: Bailian and DashScope](/zh/aliyun-fullstack/10-bailian-llm/)。

下一篇：[Article 12 — Putting It All Together](/zh/aliyun-fullstack/12-terraform-e2e)，我们将整合本系列所有内容，构建一套完整的生产级架构。
