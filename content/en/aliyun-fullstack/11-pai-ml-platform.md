---
title: "Alibaba Cloud Full Stack (11): PAI — The ML Platform"
date: 2026-05-08 09:00:00
tags:
  - Alibaba Cloud
  - PAI
  - Machine Learning
  - DSW
  - EAS
  - Cloud Computing
categories: Cloud Computing
lang: en
mathjax: false
series: aliyun-fullstack
series_title: "Alibaba Cloud Full Stack"
series_order: 11
description: "The complete ML platform on Alibaba Cloud: PAI-DSW for notebooks, PAI-DLC for distributed training, PAI-EAS for model serving, Designer for visual workflows, and Model Gallery. Train and deploy a custom model end-to-end."
disableNunjucks: true
translationKey: "aliyun-fullstack-11"
---

Training a model on a single GPU is fun. Deploying it to handle 1000 requests per second without failing is what separates experiments from products. PAI handles both.

PAI (Platform for AI) is Alibaba Cloud's managed ML platform. It's not just one product; it's five products in a trench coat, sharing a console. These include a notebook environment for exploration, a distributed training service for scale, a model serving platform for production, a visual pipeline designer for those who prefer dragging boxes, and a model gallery for one-click deployment of open-source models. After eighteen months of running real LLM workloads on it, I can say that the individual components range from excellent (EAS) to good enough (Designer). The whole platform is genuinely greater than the sum of its parts once you understand how they connect.

This article is the breadth-first tour. If you want the depth-first treatment — instance selection strategies, DLC spot preemption survival, EAS cold-start mitigation — there is a dedicated [PAI series](/en/aliyun-pai/01-platform-overview/) with five articles that go deep on each sub-product. Here we cover enough to understand what PAI is, when to reach for each component, and how to train and deploy a model end-to-end.


## PAI platform overview

![PAI platform component overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_pai_components.png)

PAI stands for Platform for AI. The name is generic because the product is broad — it covers the entire ML lifecycle from interactive experimentation to production serving. The closest equivalents on other clouds are AWS SageMaker, Azure Machine Learning, and GCP Vertex AI. However, the comparison is only approximate. SageMaker bundles notebooks, training, and endpoints into a relatively monolithic experience. PAI is more modular, with each sub-product having its own resource model, pricing, and SDK surface, and you can use any one of them independently.

The five components you will actually touch:

| Component | What it does | SageMaker equivalent |
|---|---|---|
| **PAI-DSW** | Cloud Jupyter/VSCode with GPU, pre-built images, OSS mount | SageMaker Studio / Notebook Instances |
| **PAI-DLC** | Managed distributed training jobs (multi-GPU, multi-node) | SageMaker Training Jobs |
| **PAI-EAS** | Model serving with autoscaling, blue/green, traffic split | SageMaker Endpoints |
| **PAI-Designer** | Drag-and-drop visual ML pipeline builder | SageMaker Pipelines (visual mode) |
| **PAI-QuickStart** | One-click deploy of open-source models from a gallery | SageMaker JumpStart |

The mental model that works best for me: code matures from left to right through DSW, DLC, and EAS, while Designer and QuickStart are shortcuts that skip part of that journey.

```
    DSW              DLC              EAS
     |                |                |
 [explore]     [train at scale]    [serve]
     \               |               /
      \              |              /
       +------  OSS / NAS  --------+
                     |
                 GPU ECS pool
```

PAI never owns your data. Datasets, checkpoints, and model artifacts live in OSS or NAS. PAI orchestrates GPU compute for you — when a DSW notebook starts, a real GPU ECS instance boots; when an EAS endpoint scales out, real GPU pods come up. The reason to use PAI instead of raw ECS is that it pre-bakes CUDA/PyTorch images, mounts your storage, provides metrics dashboards, and bills per second rather than per hour.

### PAI vs SageMaker: the meaningful differences

If you're coming from AWS, here are the things that might trip you up or delight you:

| Aspect | PAI | SageMaker |
|---|---|---|
| **Pricing model** | Per-second billing on actual GPU instances (you see the ECS SKU) | Abstracted "ml.p3.2xlarge" pricing, often higher |
| **Container freedom** | Full Docker image support in DLC and EAS; bring any framework | More opinionated about frameworks and entry points |
| **GPU availability** | Best stock in cn-shanghai and cn-hangzhou; A100/H800 available | Better availability in us-east-1, us-west-2 |
| **Spot training** | DLC supports spot instances at ~40% discount | SageMaker Managed Spot Training, similar discount |
| **Model gallery** | Qwen family, Chinese open-source models, plus international models | JumpStart has broader international model selection |
| **SDK maturity** | Python SDK is functional but docs lag behind Chinese version | Mature SDK, extensive documentation |

The biggest practical difference: PAI exposes the underlying ECS instance types directly. When you pick a DSW instance, you choose `ecs.gn7i-c8g1.2xlarge` (1x A10, 24 GB). When you submit a DLC job, you specify the exact GPU SKU. This transparency makes cost estimation straightforward — the same price calculators you use for ECS work for PAI.

## PAI-DSW: interactive notebooks

DSW (Data Science Workshop) is where most ML work begins on PAI. It's JupyterLab and VSCode-in-browser running on a GPU ECS instance managed by PAI. The pitch: skip the CUDA/cuDNN/PyTorch installation and get a working GPU box in about 90 seconds.

![DSW notebook workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_dsw_workflow.png)

### When to use DSW

- Interactive exploration, EDA, plotting
- Debugging a model with `pdb` and a real GPU
- Single-GPU training runs under a few hours
- Writing the training script you will eventually submit to DLC
- Iterating on inference code before deploying to EAS

Do not use DSW for multi-GPU training (use DLC), unattended jobs longer than 8 hours (idle shutdown will terminate them), or production inference (use EAS).

### GPU instance options

| Instance | GPU | VRAM | vCPU | RAM | Best for |
|---|---|---|---|---|---|
| `ecs.gn7i-c8g1.2xlarge` | 1x A10 | 24 GB | 8 | 30 GB | Prototyping, fine-tuning 7B models |
| `ecs.gn7e-c12g1.3xlarge` | 1x A100 40 GB | 40 GB | 12 | 93 GB | 13B models, larger fine-tunes |
| `ecs.gn8v-c8g1.2xlarge` | 1x H800 80 GB | 80 GB | 8 | 188 GB | 70B inference (int4), expensive |
| `ecs.g7.xlarge` | None (CPU) | - | 4 | 16 GB | EDA, data preprocessing, no GPU needed |

The pattern I follow: start on a CPU instance for data prep and EDA, switch to a GPU instance only when you actually call `.cuda()`. PAI lets you stop a DSW instance and restart it with a different SKU.

> **Cost trap:** Set the auto-shutdown timer. Every DSW instance has an "idle shutdown" knob — default 1 hour. I push it to 30 minutes for dev work. The number of times I have come in on Monday morning to find a forgotten A100 instance billing all weekend is not something I am proud of.

### Pre-built images

DSW ships images so you do not spend 20 minutes on `pip install torch`:

| Image | Contents | Use case |
|---|---|---|
| `pytorch2.3-gpu-py310-cu124` | PyTorch 2.3, CUDA 12.4, Python 3.10, Transformers | General deep learning |
| `tensorflow2.15-gpu-py310-cu121` | TensorFlow 2.15, CUDA 12.1, Keras | TF-based workflows |
| `modelscope1.17-py310-cu124` | ModelScope SDK, Qwen support, DashScope client | Alibaba model ecosystem |
| `custom` | Your own Docker image from ACR | Full control |

### Creating a DSW instance

Through the console: **PAI Console** > **DSW** > **Create Instance** > pick region, instance type, image, storage. Through the SDK:

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

### Storage: the critical part

The number one mistake new PAI users make: training for hours, then losing everything when the instance restarts. DSW instances have a system disk that resets on restart. Everything you want to keep must go to:

1. **OSS** — Mount an OSS bucket at `/mnt/data`. Read training data from here, write checkpoints here.
2. **NAS** — Mount a NAS filesystem for POSIX semantics. Better for random-access workloads (many small files).
3. **Persistent disk** — A cloud disk that survives instance restarts. Limited to 500 GB, attached to one instance.

```bash
# Inside a DSW terminal -- verify OSS mount
ls /mnt/data/
# Should show your OSS bucket contents

# Save a checkpoint to OSS (it persists)
python train.py --output_dir /mnt/data/checkpoints/run-001/

# Bad: saving to /root/ (will be lost on restart)
python train.py --output_dir /root/checkpoints/  # DON'T
```

For the full DSW deep dive — image selection, SSH tunneling, GPU memory profiling — see [PAI Part 2](/en/aliyun-pai/02-pai-dsw-notebook/).

## PAI-DLC: distributed training

DLC (Deep Learning Container) is where you go when a single GPU is not enough. It is a managed batch-job system: you hand it a container image, a command, a resource spec, and data mounts. DLC schedules the job onto a GPU cluster, sets up inter-node networking (RDMA where available), runs your code, streams logs, and tears down when done.

![DLC distributed training patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_dlc_distributed.png)

![PAI distributed training pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_training_pipeline.png)

### When to move from DSW to DLC

Move to DLC when any of these is true:
- You need more than one GPU
- The run will take more than 4 hours unattended
- You want RDMA/NCCL across nodes for faster gradient sync
- You want spot instances for cost savings
- You are running a hyperparameter sweep

The transition is usually painless because DLC accepts the same Docker image you used in DSW.

### Supported frameworks

| Framework | Job type | Use case |
|---|---|---|
| **PyTorch DDP** | PyTorchJob | Standard distributed training, the default |
| **DeepSpeed** | PyTorchJob | ZeRO optimization for large models |
| **Megatron-LM** | PyTorchJob | Tensor/pipeline parallelism for pretraining |
| **TensorFlow** | TFJob | TF distributed strategies |
| **Horovod** | MPIJob | Ring-allreduce (legacy, mostly replaced by DDP) |
| **Custom** | ElasticBatchJob | Any framework, you manage distributed init |

For modern work, `PyTorchJob` covers 95% of cases. DLC sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` environment variables on each container, so `torchrun` and `torch.distributed.init_process_group` work out of the box.

### Submitting a training job

Here is a complete DLC job specification. This fine-tunes a Qwen-2.5-7B model on a custom dataset using 8 GPUs across 2 nodes:

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

Submit with the SDK:

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

### Spot instances for cost savings

DLC supports spot (preemptible) instances at roughly 40% discount. The catch: your job can be interrupted with 5 minutes of warning. The fix: checkpoint frequently and resume from the latest checkpoint on restart.

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

For the full DLC treatment — RDMA configuration, DeepSpeed ZeRO configs, spot preemption handling — see [PAI Part 3](/en/aliyun-pai/03-pai-dlc-distributed-training/).

## PAI-EAS: model serving

EAS (Elastic Algorithm Service) is where PAI earns its keep. A DSW notebook costs a few yuan per hour while you are at your desk. A DLC job is a one-time spend. An EAS endpoint sits there 24/7, and it needs to handle traffic spikes, scale down during quiet hours, support blue/green deployments, and not fall over when your marketing team sends a push notification to a million users at 10 AM.

![EAS model serving architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_eas_serving.png)

### Deployment modes

EAS offers two modes:

**Image mode** — You push a Docker image with whatever HTTP server you want (FastAPI, Triton, vLLM). EAS runs the container, routes traffic, scales replicas. You own everything inside the container. This is the right choice for LLM serving.

**Processor mode** — You write a Python class with `initialize()` and `process()` methods. EAS provides the HTTP server and routing. Less code, less control. Fine for scikit-learn models and lightweight classifiers.

### Supported serving frameworks

| Framework | Best for | GPU required |
|---|---|---|
| **vLLM** | LLM inference (Qwen, LLaMA, Mistral) | Yes |
| **Triton Inference Server** | Multi-model serving, batching, ensemble pipelines | Optional |
| **TensorFlow Serving** | TF SavedModel format | Optional |
| **TorchServe** | PyTorch models with custom handlers | Optional |
| **ONNX Runtime** | Cross-framework optimized inference | Optional |
| **Custom Docker** | Anything else | Your choice |

### Creating an EAS service

Here is how to deploy a fine-tuned Qwen-7B model using vLLM in Image mode:

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

### Testing the endpoint

Once deployed, EAS gives you an HTTPS endpoint and an access token:

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

### Auto-scaling

EAS auto-scaling is where the cost savings happen. You configure a target metric and EAS adds or removes replicas automatically:

| Metric | When to use |
|---|---|
| **QPS** (queries per second) | API endpoints with predictable per-request cost |
| **GPU utilization** | Compute-heavy inference (image generation, large LLMs) |
| **Pending requests** | Workloads with highly variable request latency |

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

The asymmetric cooldowns matter: scale out fast (60 seconds) because you are dropping requests, scale in slow (300 seconds) because you do not want to thrash between 1 and 4 replicas during normal traffic fluctuation.

### Blue/green and A/B testing

EAS supports traffic splitting between service versions. Deploy a new model version alongside the old one and gradually shift traffic:

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

For the complete EAS deep dive — cold-start mitigation, warm pool sizing, the TPS dashboard lie — see [PAI Part 4](/en/aliyun-pai/04-pai-eas-model-serving/).

## Model Gallery

Model Gallery is PAI's model hub. It is a curated catalog of pre-trained models that you can deploy to EAS with one click or use as a starting point for fine-tuning. Think of it as Hugging Face Hub, but integrated with PAI's compute and serving infrastructure.

![GPU instance comparison for ML workloads](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_gpu_comparison.png)

### Available models

The gallery includes both Alibaba's own models and popular open-source models:

| Category | Models | Notes |
|---|---|---|
| **Qwen family** | Qwen2.5-7B/14B/32B/72B, Qwen2.5-Coder, Qwen2.5-Math | First-party, best support |
| **LLaMA family** | LLaMA-3.1-8B/70B, LLaMA-3.2-1B/3B | Community-maintained images |
| **Stable Diffusion** | SDXL, SD 3.5, FLUX | Image generation |
| **Whisper** | whisper-large-v3 | Speech-to-text |
| **Embedding models** | GTE-Qwen2, BGE-M3 | RAG retrieval |
| **Specialized** | ChatGLM, Yi, Baichuan, DeepSeek | Chinese-language optimized |

### One-click deployment

From the Model Gallery console:

1. Browse or search for a model
2. Click **Deploy**
3. Select instance type (the gallery recommends one based on model size)
4. Choose auto-scaling parameters
5. Click **Create Service**

The gallery pre-configures the Docker image, the model download from ModelScope/OSS, the serving command (vLLM for LLMs, Triton for vision models), and the health check. A Qwen2.5-7B endpoint is typically ready in under 3 minutes.

### Fine-tuning from Model Gallery

The gallery also supports fine-tuning. Select a base model, point it at your dataset in OSS, and the gallery generates a DLC training job with sensible defaults:

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

The gallery's fine-tuning defaults use LoRA (Low-Rank Adaptation) rather than full fine-tuning, which is the right default for most use cases — it is 10x cheaper in GPU-hours and the quality difference for task-specific adapters is negligible.

## Designer: visual ML workflows

PAI-Designer (formerly PAI-Studio) is the drag-and-drop ML pipeline builder. You connect components visually: data source, preprocessing, feature engineering, algorithm, evaluation, deployment. Each component is a container that runs on managed compute.

### When to use Designer

Designer makes sense for:
- **Tabular ML** — classification, regression, clustering on structured data. The built-in algorithms (XGBoost, LightGBM, logistic regression, k-means) cover 80% of traditional ML.
- **Non-coders** — data analysts and business users who can think in pipelines but do not write Python.
- **Reproducible experiments** — every pipeline run is versioned and logged. You can compare run A vs run B on the same dataset with different hyperparameters.
- **ETL + train + eval + deploy** as a single schedulable unit.

Designer does not make sense for:
- **Deep learning** — the built-in neural network components are limited. Write code in DSW, train in DLC.
- **LLM workloads** — no native support for transformer training or serving.
- **Complex custom logic** — if your preprocessing needs 200 lines of Python, a code component in Designer is more painful than just using a script.

### Built-in algorithms

| Category | Algorithms |
|---|---|
| **Classification** | XGBoost, LightGBM, Random Forest, Logistic Regression, SVM, KNN |
| **Regression** | XGBoost, LightGBM, Linear Regression, GBDT |
| **Clustering** | K-Means, DBSCAN |
| **NLP** | Text classification (BERT-based), tokenization, TF-IDF |
| **Recommendation** | Collaborative filtering, ALS |
| **Feature engineering** | Normalization, one-hot encoding, feature hashing, PCA |
| **Evaluation** | AUC, accuracy, RMSE, confusion matrix, lift chart |

### Designer vs code: my decision tree

1. Is the model a transformer or diffusion model? **Code** (DSW/DLC).
2. Is the data tabular and under 100 GB? **Designer** is a strong candidate.
3. Does the pipeline need to run on a schedule (daily retrain)? **Designer** — it has native scheduling.
4. Will the person maintaining this pipeline be a data scientist or a business analyst? If analyst, **Designer**.
5. Is this a one-off experiment? **Code** — faster to iterate.

For the Designer vs QuickStart comparison, see [PAI Part 5](/en/aliyun-pai/05-pai-designer-vs-quickstart/).

## PAI + OSS + DashScope integration

PAI does not exist in isolation. It connects to OSS for storage, to DashScope for model APIs, and to the broader Alibaba Cloud ecosystem for networking, security, and monitoring. Understanding the data flow saves a lot of debugging.

![Full PAI integration flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-fullstack/11-pai-ml-platform/11_integration_flow.png)

### The complete data flow

```
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

### Reading training data from OSS

OSS is the primary data store for PAI workloads. You upload datasets to OSS, mount them in DSW/DLC, and write results back. For model artifacts, see [Part 4: OSS Storage](/en/aliyun-fullstack/04-oss-storage/) for bucket configuration and lifecycle policies.

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

### Using DashScope models in PAI pipelines

DashScope (the Qwen API gateway, covered in [Part 10](/en/aliyun-fullstack/10-bailian-llm/)) can be called from within PAI workloads for tasks like data labeling, synthetic data generation, or embedding computation:

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

### OSS lifecycle policies for checkpoints

Training jobs generate a lot of checkpoints. A 7B model produces ~14 GB per checkpoint. Set OSS lifecycle rules to automatically clean up old checkpoints:

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

## Solution: train and deploy end-to-end

Here is the complete walkthrough: from raw dataset to production inference endpoint. We will fine-tune a Qwen2.5-7B model on a custom Q&A dataset and deploy it as a REST API. This is the pattern I use in production for the AI4Marketing platform.

### Step 1: Prepare the dataset

Format your data as JSONL with the chat template that Qwen expects:

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

Upload to OSS:

```bash
ossutil cp sft_data.jsonl oss://my-bucket/datasets/customer-service/sft_data.jsonl
```

### Step 2: Explore in DSW

Start a DSW instance with a GPU, mount the OSS bucket, and explore the data:

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

The base model will give a generic answer. After fine-tuning, it should give answers grounded in your specific policies.

### Step 3: Write the training script

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

### Step 4: Submit DLC training job

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

### Step 5: Deploy to EAS

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

### Step 6: Test the inference endpoint

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

### Step 7: Set up auto-scaling and monitoring

```bash
# Monitor the service
pai eas list-services --region cn-shanghai

# Check service metrics
pai eas describe customer-service-v1 --region cn-shanghai

# View real-time logs
pai eas logs customer-service-v1 --region cn-shanghai --tail 100
```

EAS integrates with CloudMonitor for alerting. Set up alerts for:
- **QPS spike** — auto-scaling should handle it, but alert if max replicas are reached
- **Error rate** — any sustained 5xx rate above 1% needs investigation
- **Latency p99** — LLM inference latency is bimodal; p50 might be 200 ms while p99 is 3 seconds

### Cost summary

Here is what this end-to-end workflow costs for a 7B model with 10,000 training examples:

| Step | Resource | Duration | Approximate cost (CNY) |
|---|---|---|---|
| DSW exploration | 1x A10 | 2 hours | ~30 |
| DLC training (spot) | 4x A10 | 3 hours | ~180 (after spot discount) |
| EAS serving (1 replica) | 1x A10 | per month | ~2,200/month |
| EAS serving (auto-scale 1-4) | 1-4x A10 | per month | ~2,200-8,800/month |
| OSS storage | 50 GB | per month | ~6/month |

The serving cost dominates. If your traffic is bursty, auto-scaling from 0 to N replicas (scale-to-zero) is available in EAS but adds cold-start latency of 2-5 minutes. For production services that need instant response, keep `min_replica=1`.

## Key takeaways

**PAI is five products, not one.** DSW for notebooks, DLC for training, EAS for serving, Designer for visual pipelines, QuickStart for one-click model deployment. Understand which one solves your problem before reaching for it.

**Data lives in OSS, not in PAI.** Every checkpoint, dataset, and model artifact should be in OSS. PAI compute is ephemeral — if your DSW instance restarts or your DLC job finishes, anything not in OSS is gone.

**Start in DSW, train in DLC, serve in EAS.** Code matures left to right. Write your training script interactively in DSW, submit the multi-GPU job to DLC, deploy the final checkpoint to EAS. The same Docker image works across all three.

**EAS auto-scaling is the cost lever.** A GPU sitting idle 20 hours a day costs the same as one serving 1000 QPS. Configure auto-scaling with asymmetric cooldowns — scale out fast, scale in slow.

**Use spot instances for training.** DLC spot instances are ~40% cheaper. Checkpoint frequently (every 500 steps) and your job survives preemption without losing progress.

**Model Gallery is the fast path.** If you need a Qwen or LLaMA endpoint and do not need custom training, Model Gallery gets you from zero to serving in under 5 minutes. Use it for evaluation before committing to a full training pipeline.

For the full depth on each sub-product, the [PAI series](/en/aliyun-pai/01-platform-overview/) has five articles: [DSW notebooks](/en/aliyun-pai/02-pai-dsw-notebook/), [DLC distributed training](/en/aliyun-pai/03-pai-dlc-distributed-training/), [EAS model serving](/en/aliyun-pai/04-pai-eas-model-serving/), and [Designer vs QuickStart](/en/aliyun-pai/05-pai-designer-vs-quickstart/). For LLM APIs without managing your own infrastructure, see [Part 10: Bailian and DashScope](/en/aliyun-fullstack/10-bailian-llm/).

Next up: [Article 12 — Putting It All Together](/en/aliyun-fullstack/12-production-architecture/), where we assemble a complete production architecture using everything from this series.
