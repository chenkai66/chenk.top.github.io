---
title: "自然语言处理（八）：模型微调与PEFT"
date: 2025-11-05 09:00:00
tags:
  - NLP
  - PEFT
  - LoRA
  - LLM
  - 微调
categories: 自然语言处理
series: nlp
lang: zh-CN
mathjax: true
description: "深入参数高效微调：LoRA 为什么用低秩更新就够、QLoRA 把 7B 模型塞进 6GB 显存的内存账本、Adapter 与 Prefix-Tuning 的取舍，以及生产环境怎么选。"
disableNunjucks: true
series_order: 8
translationKey: "nlp-8"
polished_by_qwen_max: true
---
2020 年，想要微调一个 70 亿参数的语言模型可不是件小事：得准备八张 A100 显卡、花上几天时间，还得有一位懂得如何调试梯度检查点的工程师全程盯着。到了 2024 年，研究生用一台笔记本电脑就能搞定。从那时到如今，这一路的变化几乎全靠两篇论文铺平道路——胡等人在 ICLR 2022 提出的 LoRA，以及 Dettmers 等人在 NeurIPS 2023 发表的 QLoRA。

这不仅仅是工程上的进步。参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）彻底改变了“拥有一个模型”的含义。过去每个任务都需要一份完整的权重文件，现在只需要一个冻结的基础模型，再加上一个存放小适配器的目录，每个适配器只有几十兆大小。切换任务变成了加载不同的适配器；支持 N 个领域的服务从 O(N) 的复杂度降到了 O(1) 基础模型加 N 个轻量级适配器。

本文将从第一性原理出发重新梳理 PEFT。我们会先探讨全量微调能解决哪些问题、又有哪些局限，然后推导 LoRA 的低秩假设，拆解 QLoRA 是如何通过内存优化把 7B 参数模型塞进 6GB 显存的，并最终落到实际工程中的选择：该用哪种方法？秩设为多少？改哪些模块？

<!-- wanx-hero -->
![自然语言处理（八）：模型微调与PEFT — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/fine-tuning-peft/illustration_1.jpg)
## 你将学到
的内容

- **为什么** 在大语言模型（LLM）时代，全量微调是一种资源浪费——过参数化现象与内在秩假设的启示  
- **LoRA 的工作原理**：分解公式 $\Delta W = BA$ 的含义、为何 $B$ 的初始值设为零、以及 $\alpha/r$ 缩放如何影响有效学习率  
- **QLoRA 的核心技术**：NF4 量化方法、双重量化技术、分页优化器的设计，以及内存占用的精确计算方式  
- **Adapter 和 Prefix-Tuning 的应用**：它们在 Transformer 模块中的具体位置、在哪些场景下表现优异、又在哪些情况下存在不足  
- **实际工程中的决策点**：如何选择合适的秩、目标模块的选择策略、多 LoRA 服务的实现方法、以及通过指令微调和基于人类反馈的强化学习（RLHF）进行模型对齐的最佳实践
## 前置知识

<!-- wanx-mid -->
![自然语言处理（八）：模型微调与 PEFT —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/illustration_2.jpg)

- 熟悉 Transformer 架构（详见 [第 4 部分](/en/nlp/attention-transformer/)）
- 了解 GPT 风格的解码器原理（参考 [第 6 部分](/en/nlp/gpt-generative-models/)）
- 掌握 PyTorch 的基本用法，并对 GPU 内存的核心概念（如优化器状态、激活值和梯度）有一定理解

---
## 1. 为什么不直接全量微调？
### 显存开销的现实

全量微调意味着所有参数都需要更新，每个梯度都要存储下来，而优化器（通常是 AdamW）还会为每个参数额外维护两个 fp32 的状态缓冲区。以一个 7B 参数的模型为例，在混合精度训练的情况下，单步显存占用大致如下：

| 组件 | 每参数字节数 | 7B 模型 |
|------|-------------|---------|
| 权重（fp16） | 2 | 14 GB |
| 梯度（fp16） | 2 | 14 GB |
| AdamW 状态（fp32 m + v） | 8 | 56 GB |
| 激活值（取决于序列长度和 batch 大小） | — | 8–20 GB |
| **总计** | — | **约 95 GB** |

换句话说，还没写一行训练代码，至少需要两张 A100-80GB 显卡才能跑起来。这清楚地说明了为什么 PEFT 不仅仅是一个“锦上添花”的优化手段——对大多数从业者来说，它是能够实际操作 7B 以上大模型的唯一选择。

![PEFT 各方法在 7B 基座上的可训练参数量，以及服务 N 个任务时的磁盘占用对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%与PEFT/fig1_full_vs_peft.png)

### 内在秩假设的启示

更深层次的原因来自 Aghajanyan 等人（2020）以及 LoRA 论文的核心观点：**微调引入的权重变化通常具有非常低的内在秩**。如果预训练模型的调整本质上局限在一个低维子空间中，那么训练完整的 $d \times k$ 矩阵不仅成本高昂，还选错了搜索空间——应该直接在低秩流形中寻找解。

实验表明，对 175B 参数的模型进行下游任务微调时，只需训练大约 200 个参数方向即可达到与全量微调相当的效果（Aghajanyan et al., 2020）。这一发现正是 LoRA 方法的理论基础。

### 冻结微调——效果有限的基线方法

在 PEFT 出现之前，最简单的节省资源的方法是冻结模型主体，仅训练分类头，或者只解冻顶部几层。例如：

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.transformer.h[-2:].parameters():
    param.requires_grad = True
```

这种方法在基于 BERT 类编码器的分类任务中还能勉强应付，但在生成任务中就显得力不从心：被训练的层数太少，无法充分调整上游表示的方向。相比之下，PEFT 方法通过在整个网络中插入少量可训练参数，既保持了参数总数极低，又显著提升了模型的表现能力。
## 2. LoRA：低秩适配
### 分解原理

对于一个冻结的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 通过两个“瘦矩阵”的乘积来学习一个加性更新：

$$W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\, B A,
\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k).$$

前向传播的计算变为 $h = x W_0^\top + \frac{\alpha}{r} (x A^\top) B^\top$。原始计算 $x W_0^\top$ 保持不变，而 LoRA 的分支则引入了一个秩为 $r$ 的修正。

![LoRA 将权重更新分解为两个瘦矩阵，其中 $B$ 初始化为零，$A$ 初始化为小随机值。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig2_lora_decomp.png)

### 参数统计

假设 $d = k = 4096$（LLaMA-7B 的隐藏层维度），且 $r = 8$：

- 原始 $W_0$：$d \cdot k = 16{,}777{,}216$ 个参数
- LoRA 的 $A, B$：$r \cdot (d + k) = 65{,}536$ 个参数
- 减少：**每个矩阵的可训练参数减少 256 倍**

如果将 LoRA 应用到 32 层中的所有四个注意力投影上，总共只需训练约 800 万个参数——占 70 亿总参数的 0.12%。

### 两个关键设计点

**初始化不对称性。** $B$ 初始化为零，而 $A$ 初始化为小高斯分布。因此在训练的第一步，$BA = 0$，模型的行为与预训练版本完全一致。这种设计让训练从一个已知的良好起点开始，无需额外的 warmup 操作。

**$\alpha/r$ 缩放因子。** 这个因子绝不仅仅是装饰性的。如果固定 $\alpha$ 并增大 $r$，更新的幅度会随之增加（更多方向、更大的组合范数），从而迫使重新调整学习率。$\alpha/r$ 因子将两者解耦：只需一次设定 $\alpha$（通常取 $\alpha = 2r$），通过调整 $r$ 来控制容量，有效步长基本保持不变。

### 最简实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """为冻结的 nn.Linear 添加一个可训练的低秩更新。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        # A: (r, in_features) — 小高斯初始化
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B: (out_features, r) — 零初始化，分支初始不激活
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 冻结路径
        out = self.base(x)
        # 低秩分支：x @ A^T @ B^T，缩放后相加
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """将 LoRA 增量合并到基座权重中，推理时无额外开销。"""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta
        self.lora_B.zero_()
```

`merge()` 方法是 LoRA 在推理时零成本的关键：训练完成后，将 $\Delta W$ 合并到 $W_0$ 中，模型结构与原始模型完全一致——没有额外的矩阵乘法，也没有延迟。

### 应用场景

Hu 等人发现，将 LoRA 应用于**查询和值投影**（`q_proj`、`v_proj`）即可获得大部分收益。当前生产环境的默认设置更广——四个注意力投影都用，有时还包括 FFN：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.124
```

再加入 `gate_proj`、`up_proj`、`down_proj`（LLaMA 的 FFN 三件套），通常在基准测试中只能提升不到一个点，但可训练参数量是原来的 3 倍。代码生成任务值得，分类任务基本不划算。
## 3. QLoRA：把基座模型压缩到 4 比特

LoRA 解决了可训练参数的问题，但冻结的权重仍然以 fp16 格式存储，占据了显存的大头。QLoRA 的目标正是解决这个问题。

![QLoRA 将 4 比特量化的基础模型与 bf16 的 LoRA 适配器结合；梯度仅通过绿色的小路径流动。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig5_qlora.png)

### QLoRA 的三大创新点

**1. NF4（NormalFloat 4 比特）量化**  
传统的 4 比特整数量化效率不高，因为大语言模型的权重分布并不均匀，而是接近高斯分布。NF4 根据标准正态分布选择了 16 个等概率的量化级别，从信息论的角度实现了最优覆盖。每个块通常包含 64 个权重，并配有一个 fp16 的缩放因子。

**2. 双重量化**  
每个块都需要一个缩放常数。对于 7B 参数的模型，如果块大小为 64，则需要约 1.1 亿个缩放常数，占用约 0.44 GB 内存。QLoRA 对这些常数本身进行 8 比特量化，每参数节省约 0.4 比特。虽然单个参数的节省看似微不足道，但在 7B 规模下却意义重大。

**3. 分页优化器**  
长序列会导致激活内存激增。Paged AdamW 借助 NVIDIA Unified Memory 将优化器状态分页到 CPU 内存中，需要时再加载回 GPU。这种设计牺牲了一部分吞吐量，但能够训练原本会因显存不足而无法运行的长序列。

### 配置方法

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4 比特
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 对量化常数再次量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # 将 LayerNorm 等转换为 fp32

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
```

### 代价与收益

相比 fp16 LoRA，QLoRA 在大多数基准测试中的性能损失约为 1–2 个百分点，但基本恢复了全量微调的效果。而显存的节省则非常显著：

![按组件拆分的训练显存，以及不同模型规模下的峰值显存——QLoRA 让 70B 模型跑在单张 A100 上。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig6_memory.png)

对于 70B 参数的模型，全量微调需要近 1 TB 显存，这基本只有顶级实验室才能负担得起。QLoRA 将其压缩到单张 80 GB 的 GPU 上，直接决定了一个项目是“只能研究”还是“可以落地”。
## 4. Adapter：最早的 PEFT 方法
早在 2019 年，Houlsby 等人就提出了 Adapter，这比 LoRA 的出现早了两年。它的核心思想非常直观：在 Transformer 每个子层的后面插入一个小型的瓶颈模块。

$$\text{Adapter}(x) = x + W_{\text{up}}\, \sigma(W_{\text{down}}\, x), \qquad
W_{\text{down}} \in \mathbb{R}^{m \times d},\ W_{\text{up}} \in \mathbb{R}^{d \times m},\ m \ll d.$$

其中，$W_{\text{up}}$ 被初始化为零，这意味着 Adapter 初始状态下等同于恒等映射——这一技巧与 LoRA 中 $B$ 的初始化方式如出一辙。

![Adapter 插入在注意力和 FFN 子层之后；每个 Adapter 都是一个包含降维、非线性变换和升维的瓶颈结构，并带有残差连接。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig3_adapter.png)

```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))
```

### Adapter 和 LoRA 的对比：谁更适合你的场景？

| | **Adapter** | **LoRA** |
|---|---|---|
| 位置 | 嵌入在残差路径**内部** | 附加在权重矩阵**旁边** |
| 推理开销 | 每个模块增加一次矩阵乘法（约 5–10% 的延迟） | 调用 `merge()` 后无额外开销 |
| 可组合性 | 支持多个 Adapter 的串联 | 直接叠加 LoRA 的增量 |
| 多任务服务 | 每个任务对应一个 Adapter | 每个任务对应一个 LoRA，支持批量混合 |
| 典型大小 | 占基础模型的 0.5–3% | 占基础模型的 0.1–1% |
| 适用场景 | 推理图稳定的情况 | 对延迟要求严格的 LLM 服务 |

由于推理过程中零开销的优势，LoRA 已经占据了 Adapter 的大部分市场份额。然而，在需要显式模块化组合的场景中，Adapter 依然具有独特的优势。例如，在多语言任务中，AdapterFusion（Pfeiffer 等，2021）可以在推理时动态地组合多个针对特定语言的 Adapter，从而实现更灵活的多语言支持。
## 5. 基于提示的 PEFT：调整输入，而非权重
一种截然不同的思路：模型参数完全冻结，转而学习**该给模型喂什么数据**。

![三种基于提示的 PEFT 方法——输入层的 soft prompt、每层注入的 KV 前缀、以及深层提示覆盖所有层。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig4_prefix_prompt.png)

### Prompt-Tuning（Lester et al., 2021）

在输入序列前添加 $n$ 个可训练的嵌入向量，这就是整个方法的核心思想。简单却有效。当使用 11B 参数的 T5 模型时，Prompt-Tuning 在 SuperGLUE 基准上的表现能够媲美全量微调——但这一结论**仅适用于大规模模型**。对于小于 1B 参数的模型，Prompt-Tuning 的效果则大打折扣。

### Prefix-Tuning（Li & Liang, 2021）

与 Prompt-Tuning 思路类似，但将这一方法应用到每一层 Transformer 的注意力机制中：通过学习键和值的前缀矩阵 $P_K, P_V$，在计算注意力时将它们拼接到原始键值对之前：

$$\text{Attention}(Q,\ [P_K; K],\ [P_V; V]).$$

由于每一层都有独立的前缀，模型可以比单一输入提示更深入地调整其内部计算逻辑。

### P-Tuning v2（Liu et al., 2022）

摒弃了 Prefix-Tuning 中使用的重参数化 MLP（因为它在不同规模模型上的表现不够稳定），改为统一应用深层提示。这是首个在**所有规模模型**（包括小型模型）上都能达到全量微调水平的提示类方法，尤其在 NLU 任务中表现出色。

**这些方法适合什么时候用？**  
1. 当你需要极致的参数效率时——每个任务只需占用几 KB 的存储空间。  
2. 当你希望在一个服务化的模型背后挂载多个提示，灵活应对不同任务需求。  

不过，在生成类任务中，这类提示方法的表现通常不如 LoRA。
## 6. 方法怎么选

![PEFT 方法在参数效率与质量之间的权衡，以及 LoRA 秩对不同任务类型的影响。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%与PEFT/fig7_perf_vs_params.png)

以下是一个实用的决策流程：

```
                    ┌─ 追求极致性能？ → 全量微调（如果显存允许）
                    │
显存是否充足？  ────┤
                    │  是 → 使用 LoRA r=16 修改 q/k/v/o
                    └─ 否 → 使用 QLoRA r=16（NF4 + 分页 AdamW）

一个基础模型服务多种任务？ → LoRA（支持合并和切换）

生成任务对延迟要求严格？ → 合并后的 LoRA 推理  
编码器 NLU 任务希望参数最少？ → P-Tuning v2 或 BitFit  
模块化或组合式场景？ → Adapter（搭配 AdapterFusion 使用）
```

### 如何选择秩 $r$

建议从 $r = 16$ 开始。从上图右侧可以看出，大多数任务的收益递减拐点位于 $r = 8$ 到 $r = 32$ 之间：

- **简单分类任务**：$r = 8$ 基本够用  
- **代码生成、逻辑推理**：$r = 32$ 或 $r = 64$ 效果更佳  
- **领域适配任务**（如医疗、法律）：通常 $r = 16$ 到 $r = 32$ 比较合适  

如果不确定最佳值，可以尝试 $r \in \{8, 16, 32\}$，选择能够达到验证集目标指标的最小值即可。
## 7. 对齐：指令微调与 RLHF
PEFT 是手段，对齐才是目的。现代大语言模型的后训练过程通常分为两个关键阶段：

**有监督指令微调**  
通过 `(instruction, response)` 数据对基础模型进行微调，使其能够理解并准确响应人类编写的指令。在这一阶段，数据质量远比数量重要：1K 到 10K 条精心设计的样本，往往能胜过 10 万条普通的众包数据（Zhou 等人在 2023 年的 LIMA 论文中展示了仅用 1K 条高质量样本，就能让一个 65B 参数的模型表现得非常出色）。

```python
def format_example(ex):
    if ex["input"]:
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}")
    return (f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}")
```

**RLHF**（Ouyang 等人 2022 年的 InstructGPT 论文）  
在完成指令微调后，基于人类偏好数据训练一个奖励模型，并使用 PPO 算法优化策略以适配该奖励模型。奖励模型的损失函数采用 Bradley–Terry 偏好损失，公式如下：

$$\mathcal{L}_{\text{RM}} = -\log \sigma\bigl(r_\theta(x, y_{\text{chosen}}) - r_\theta(x, y_{\text{rejected}})\bigr).$$

```python
class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[:, -1, :]
        return self.head(last).squeeze(-1)
```

关于强化学习的具体内容，可以参考 [RL 第 12 部分：RLHF 与 LLM 应用](/en/reinforcement-learning/12-rlhf-and-llm-applications/)。此外，DPO（Rafailov 等人 2023 年提出）作为一种更简单的替代方法，直接省略了显式的奖励模型，近年来也受到了广泛关注。
## 8. 端到端方案

```python
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch

MODEL = "meta-llama/Llama-2-7b-hf"

# 1. 加载 4-bit 模型
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 2. 在注意力机制的投影层应用 LoRA
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
# 可训练参数：8,388,608 || 总参数：3,508,801,536 || 可训练占比：0.239

# 3. 使用 SFTTrainer 进行训练（自动处理数据格式和批处理）
ds = load_dataset("yahma/alpaca-cleaned", split="train")
args = TrainingArguments(
    output_dir="./llama-qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    bf16=True,
    optim="paged_adamw_8bit",     # QLoRA 的分页优化器
    logging_steps=10, save_steps=500,
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=ds,
    tokenizer=tokenizer, max_seq_length=1024,
    dataset_text_field="text",
)
trainer.train()
model.save_pretrained("./llama-qlora-adapter")  # 磁盘占用约 80 MB
```

### 多 LoRA 推理支持

由于 LoRA 适配器体积小巧且支持叠加，可以同时在内存中加载数十个适配器，并根据请求动态切换：

```python
# 将多个适配器加载到同一个基础模型中
model.load_adapter("./adapter-medical", adapter_name="med")
model.load_adapter("./adapter-legal",   adapter_name="legal")

model.set_adapter("med")
out_a = model.generate(...)

model.set_adapter("legal")
out_b = model.generate(...)
```

像 vLLM 和 S-LoRA 这样的框架进一步优化了这一过程：通过堆叠 LoRA 增量，在单次前向传播中批量处理针对不同适配器的请求。显存中只需保留一个 7B 的基础模型，即可高效服务数百个微调模型。

---
## 常见问题
### 什么情况下适合全量微调？

如果你有充足的计算资源、$\geq$10 万条高质量样本，并且对每个小数点的性能提升都斤斤计较——比如基座模型厂商在推出旗舰级指令模型时。对于其他人来说，LoRA 或 QLoRA 已经足够。

### LoRA 的秩该如何选择？

从 16 开始尝试。如果发现欠拟合（训练损失停滞在较高水平），可以提高到 32 或 64。如果数据量较少且出现过拟合，则可以降到 4–8。对于分类任务，8 通常已经绰绰有余。

### 应该调整哪些模块？

`q_proj` 和 `v_proj` 能带来大约 80% 的效果提升。再加上 `k_proj` 和 `o_proj` 就能覆盖剩余的部分。只有在生成任务占比较高且预算允许的情况下，才需要加入 FFN 的三个模块（`gate/up/down_proj`），因为这会让可训练参数增加 3 倍。

### LoRA 会影响推理速度吗？

merge 后完全不会影响。但在 merge 之前，每层会多一次额外的小矩阵乘法操作——虽然可以忽略不计，但确实存在。

### QLoRA 的质量下降明显吗？

在标准基准测试中，相比 fp16 的 LoRA，QLoRA 通常会有 1–2 个点的下降，基本在误差范围内。对于绝大多数从业者来说，显存节省的价值远远超过这点性能损失。

### 需要多少指令数据？

LIMA 的研究表明，仅用 1K 条精心挑选的样本，就能将一个强大的基座模型训练成一个连贯的助手。实际应用中，1K–10K 条高质量样本是一个合理的下限；数据的质量远比数量重要。

### PEFT 方法可以组合使用吗？

可以。LoRA + Prompt-Tuning 是一种有文献支持的组合方式，而 QLoRA 本身就是一个方法栈（4 比特基座 + LoRA + 分页优化器）。不过，在同一个模型中同时使用 Adapter 和 LoRA 则较为少见。
## 9. 一份具体的微调配方（7B 上的 LoRA）

以下是我亲测有效并成功上线的超参数配置，真正对效果有影响。

**数据规模**  
在 7B 模型上使用 LoRA 时，1000 到 5000 条高质量的指令-响应对就能带来显著收益。如果样本数量少于 500 条，不如直接用 few-shot 提示来得高效。而当数据量超过 20000 条时，除非任务极其复杂，否则投入产出比会大幅下降，属于“花了钱却没赚到分”。

**rank `r`**  
默认值设为 `r=8` 即可。对于代码生成、多步推理等复杂任务，可以将 `r` 提升到 `r=16` 或 `r=32`。增加 `r` 会让参数量翻倍，但训练时间几乎不会显著增加。不过，`r` 超过 64 通常不会有额外收益，反而容易导致过拟合。

**alpha**  
设置 `alpha = 2 * r`，例如当 `r=8` 时，`alpha=16`。`alpha/r` 是 LoRA 路径上的等效学习率缩放因子，将其固定为 2 是一个稳妥的选择。

**目标模块**  
如果显存允许，建议将 LoRA 同时应用到 `q_proj, k_proj, v_proj, o_proj` 和 `gate_proj, up_proj, down_proj` 上。原始论文中仅对注意力模块（attention-only）进行微调，在指令调优场景下会损失不少潜在收益。实际测试表明，结合注意力和 MLP 的方法在 Open LLM Leaderboard 的任务上能额外提升 2 到 4 分。

**学习率**  
LoRA 的初始学习率推荐设为 `2e-4`。相比之下，全量微调通常使用 `2e-5`，而 LoRA 可以承受更高的学习率，因为只有少量参数在更新。建议搭配 cosine 学习率调度，并设置 3% 的 warmup 比例。

**batch size 和梯度累积**  
等效 batch size 控制在 32 到 64 条序列之间为宜。在单张 A100 80GB 显卡上，序列长度为 2048 时，micro-batch 通常只能容纳 2 到 4 条序列，因此需要通过梯度累积达到目标 batch size。

**epoch 数**  
跑 1 到 3 轮即可。观察验证集损失（eval loss），通常在第 2 轮时就会降到最低点。对于小规模指令数据集，跑到 5 轮以上几乎一定会过拟合。

**序列打包**  
将多个短样本打包成一条 2048-token 的序列，并正确设置 attention mask。相比传统的 padding 方法，这种方式能将吞吐量提升 2 到 3 倍。目前 `transformers` 的 SFTTrainer 已原生支持该功能。

**训练成本估算**  
使用 5000 条样本、7B 基座模型、单张 A100 显卡，整轮训练大约需要 2 到 4 小时，租卡成本约为 $8 到 $12。
## 10. LoRA 悄悄翻车的地方

以下是我在实际项目中见过的三种让团队头疼的失败模式，分享给大家以供参考。

**失败 1：稀有 token 长尾部分的灾难性遗忘**  
虽然 LoRA 在保护基座模型方面比全量微调更胜一筹，但它仍然会对输出分布产生一定影响。举个例子，当我们对一个代码模型进行指令微调后，尽管英文指令的理解能力显著提升，但像 Erlang、Haskell 这类稀有编程语言的困惑度（perplexity）却飙升了 30% 以上。解决这个问题的办法是，在微调数据中加入 5%-10% 的“基座风格”数据作为回放样本，帮助模型更好地保留对稀有 token 的理解。

**失败 2：非英语输出质量下降的隐忧**  
当使用以英文为主的 SFT 数据集进行 LoRA 和指令微调时，即使基座模型本身是多语言的，中文、韩文、日文等语言的输出质量也会显著下降。这是因为 LoRA 更倾向于学习符合英文特点的回答模式。为了解决这个问题，建议在 SFT 数据集中至少加入 20% 的目标语言样本，确保模型在多语言场景下的表现更加均衡。

**失败 3：合并后的 LoRA 并不总是等价于推理时的 LoRA**  
当你将 LoRA 权重合并到基座模型中以准备部署（`model.merge_and_unload()`），在 4-bit 量化的情况下，合并后的模型性能可能会明显低于未合并的版本。原因在于，合并操作扰动了基座权重，而量化过程中的重新取整进一步放大了这些变化。针对这一问题，有两种解决方案：一是推理时保持 LoRA 分离（会稍微增加一点延迟），二是在合并后使用预留的数据集进行量化校准，从而缓解性能损失。
## 参考文献

1. Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS.
3. Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML.
4. Li, X. L. & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL.
5. Lester, B. et al. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP.
6. Liu, X. et al. (2022). *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally*. ACL.
7. Aghajanyan, A. et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ACL 2021.
8. Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
9. Rafailov, R. et al. (2023). *Direct Preference Optimization*. NeurIPS.
10. Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment*. NeurIPS.
