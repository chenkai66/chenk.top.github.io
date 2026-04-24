---
title: "自然语言处理（八）：模型微调与PEFT"
date: 2024-06-08 09:00:00
tags:
  - NLP
  - PEFT
  - LoRA
  - LLM
  - 微调
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 8
  total: 12
lang: zh-CN
mathjax: true
description: "深入参数高效微调：LoRA 为什么用低秩更新就够、QLoRA 把 7B 模型塞进 6GB 显存的内存账本、Adapter 与 Prefix-Tuning 的取舍，以及生产环境怎么选。"
disableNunjucks: true
---

2020 年微调一个 70 亿参数的语言模型还得排预算：八张 A100、几天时间，再加一个会调梯度检查点的工程师。2024 年，研究生在笔记本上就能跑。中间这段距离，几乎完全由 Hu 等人 2022 年的 LoRA 论文，以及 Dettmers 等人 2023 年的 QLoRA 论文铺平。

但这不只是工程优化。参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）重新定义了"拥有一个模型"这件事。过去每个任务都得有一份完整的权重文件，现在只需要一个冻结的基座，加上一个目录里几十兆的小适配器。换任务变成了换文件，N 个领域的服务从 O(N) 变成 O(1) 个基座加 N 个 ε。

本文从第一性原理重建 PEFT。先回答全量微调能解决什么、不能解决什么，再推导 LoRA 的低秩假设，把 QLoRA 让 7B 模型塞进 6GB 显存的算账过程拆开，最后落到工程选择：用哪种方法、秩多少、改哪些模块。

## 你将学到

- **为什么** LLM 时代全量微调是浪费——过参数化与"内在秩"假说
- **LoRA 的机制**：分解 $\Delta W = BA$、为什么 $B$ 初始化为零、$\alpha/r$ 缩放怎么影响有效学习率
- **QLoRA**：NF4 量化、双重量化、分页优化器，以及确切的内存账本
- **Adapter 与 Prefix-Tuning**：它们在 Transformer 块里的位置，什么时候赢、什么时候输
- **工程选择**：秩怎么挑、改哪些模块、多 LoRA 服务、指令微调与 RLHF 对齐

## 前置知识

- Transformer 架构（[第 4 部分](/zh/自然语言处理-四-注意力机制与Transformer/)）
- GPT 风格解码器（[第 6 部分](/zh/自然语言处理-六-GPT与生成式语言模型/)）
- PyTorch 基础与 GPU 内存直觉（优化器状态、激活、梯度）

---

## 1. 为什么不直接全量微调？

### 一笔账

全量微调意味着所有参数都要算梯度，AdamW 还要为每个参数额外存两个 fp32 状态。一个 7B 模型在混合精度下，单步训练的显存账是这样：

| 项目 | 每参数字节 | 7B 模型 |
|------|-----------|---------|
| 权重（fp16） | 2 | 14 GB |
| 梯度（fp16） | 2 | 14 GB |
| AdamW 状态（fp32 m + v） | 8 | 56 GB |
| 激活（依赖序列长度、batch） | — | 8–20 GB |
| **合计** | — | **约 95 GB** |

也就是说在写下第一行训练代码之前，你至少要两张 A100-80G。这笔账解释了为什么 PEFT 不是"锦上添花的优化"——对绝大多数从业者来说，它是能碰 7B 以上模型的**唯一**方式。

![PEFT 各方法在 7B 基座上的可训练参数量，以及服务 N 个任务时的磁盘占用对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig1_full_vs_peft.png)

### 内在秩假说

还有一层更深的理由。Aghajanyan 等人（2020）和 LoRA 原论文都指出：**微调引入的权重变化具有非常低的内在秩**。如果你需要对预训练模型做的那个改动本来就生活在一个低维子空间里，那么训练完整的 $d \times k$ 矩阵不仅贵，**还在用错误的假设空间搜索**——你应该直接在低秩流形上找。

实证上，对 175B 模型在下游任务上的微调，可以仅用大约 200 个参数空间方向就达到匹配（Aghajanyan et al., 2020）。这就是 LoRA 想法的钥匙。

### 冻结微调——一个弱基线

PEFT 之前最朴素的省钱办法，是冻住主体只训分类头，或者只解冻顶上几层：

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.transformer.h[-2:].parameters():
    param.requires_grad = True
```

这套方法在 BERT 类编码器上做分类还行，做生成就不行：被训练的层数太少，没法把上游表示重新引导到任务需要的方向。PEFT 方法不一样——它在网络**各处**插入可训参数，但总数仍然极小。

---

## 2. LoRA：低秩适配

### 分解形式

对一个冻结的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 学一个加性更新，参数化成两个瘦矩阵的乘积：

$$
W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\, B A,
\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k).
$$

前向变成 $h = x W_0^\top + \frac{\alpha}{r} (x A^\top) B^\top$。原来的 $x W_0^\top$ 一动不动，LoRA 分支加一个秩为 $r$ 的修正。

![LoRA 把权重更新分解成两个瘦矩阵，$B$ 初始化为零，$A$ 初始化为小高斯。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig2_lora_decomp.png)

### 数一下参数

取 $d = k = 4096$（LLaMA-7B 的隐层维度），$r = 8$：

- 原始 $W_0$：$d \cdot k = 16{,}777{,}216$ 个参数
- LoRA 的 $A, B$：$r \cdot (d + k) = 65{,}536$ 个参数
- 缩减：**每个矩阵的可训参数减少 256 倍**

把 LoRA 应用到 32 层每层的 4 个注意力投影上，可训参数大约 800 万——占 7B 总量的 0.12%。

### 两个不能省的设计

**初始化的不对称性。** $B$ 初始化为零，$A$ 用小高斯。所以训练第一步时 $BA = 0$，模型的前向输出和预训练版完全一致。训练从一个已知良好的工作点开始，不需要折腾 warmup。

**$\alpha/r$ 这个比例。** 它不是装饰性常数。如果固定 $\alpha$、增大 $r$，更新的范数会随之变大（方向更多、合成模长更长），逼你重新调学习率。$\alpha/r$ 缩放把这两件事解耦：$\alpha$ 一次定下（典型取 $\alpha = 2r$），扫 $r$ 找容量，有效步长基本不变。

### 一个最小实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """把一个冻结的 nn.Linear 包上一个可训的低秩更新。"""

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
        # B: (out_features, r) — 零初始化，让分支起步时不发挥作用
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 冻结分支
        out = self.base(x)
        # 低秩分支：x @ A^T @ B^T 再缩放
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """把 LoRA 增量折回基座权重——推理时零额外开销。"""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta
        self.lora_B.zero_()
```

`merge()` 这个方法是 LoRA 在推理时不花钱的关键：训完之后把 $\Delta W$ 折进 $W_0$，模型在结构上就和原始模型一模一样了——没有多余的矩阵乘，没有额外延迟。

### 改哪些模块

Hu 等人在原论文里发现，把 LoRA 加到**查询和值投影**（`q_proj`、`v_proj`）就能拿到大部分收益。生产环境现在的默认要更广一点——四个注意力投影都加，有时候连 FFN 也一起：

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

再把 `gate_proj`、`up_proj`、`down_proj`（LLaMA 的 FFN 三件套）加上，通常在 benchmark 上多换不到一个点，但可训参数量是原来的 3 倍。代码生成任务值得，分类任务基本不值。

---

## 3. QLoRA：把基座压到 4 比特

LoRA 解决的是**可训**参数的问题。**冻结**的那部分权重还在 fp16 里趴着，仍然是显存大头。QLoRA 攻击的就是这块。

![QLoRA 把基座放进 4 比特，LoRA 适配器留在 bf16；梯度只走绿色那条小路径。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig5_qlora.png)

### QLoRA 的三个创新

**1. NF4（NormalFloat 4-bit）量化。** 朴素的 4 比特整数量化浪费分辨率，因为 LLM 的权重并不均匀分布——它们近似服从高斯。NF4 选 16 个分位数，使每一档在标准正态下等概率，理论上是最优编码。块大小通常是 64，每块配一个 fp16 缩放常数。

**2. 双重量化。** 每个块都需要一个缩放常数；对 7B 模型，块大小 64 意味着大约 1.1 亿个常数 × 4 字节 = 0.44 GB 的额外开销。QLoRA 把这些**常数本身**也量化到 8 比特，每参数省下大约 0.4 比特——单个看不多，7B 规模上有意义。

**3. 分页优化器。** 长序列会让激活内存突然飙升。Paged AdamW 通过 NVIDIA Unified Memory 把优化器状态分页到 CPU 内存，需要时再换回来。牺牲一点吞吐，换来训练原本会 OOM 的长序列的能力。

### 配置

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4 比特
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 把量化常数本身再量化一次
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # 把 LayerNorm 等转回 fp32

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
```

### 代价与收益

QLoRA 相对 fp16 LoRA，在大部分 benchmark 上掉 1–2 个百分点，但能恢复全量微调的绝大部分性能。换来的显存收益是颠覆性的：

![训练显存按组件拆分，以及不同模型规模下的峰值显存——QLoRA 让 70B 进了单张 A100。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig6_memory.png)

70B 模型全量微调大约要一 TB 显存（基本上只有前沿实验室在做）。QLoRA 把它压进了一张 80 GB 卡。这是"只能搞研究"和"你的团队能落地"之间的差别。

---

## 4. Adapter：最早的 PEFT

Houlsby 等人 2019 年比 LoRA 早两年提出 Adapter。思路是：在 Transformer 每个子层后插入一个小瓶颈模块。

$$
\text{Adapter}(x) = x + W_{\text{up}}\, \sigma(W_{\text{down}}\, x), \qquad
W_{\text{down}} \in \mathbb{R}^{m \times d},\ W_{\text{up}} \in \mathbb{R}^{d \times m},\ m \ll d.
$$

$W_{\text{up}}$ 初始化为零，让 Adapter 起步是恒等映射——和 LoRA 的 $B$ 同样的把戏。

![Adapter 插在注意力和 FFN 子层之后；每个 Adapter 是降维-非线性-升维的瓶颈结构，外加残差。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig3_adapter.png)

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

### Adapter 还是 LoRA

| | **Adapter** | **LoRA** |
|---|---|---|
| 位置 | 在残差路径**之内** | 在权重矩阵**旁边** |
| 推理开销 | 每块多一次矩阵乘（约 5–10% 延迟） | merge 后为零 |
| 可组合性 | 多 Adapter 串叠 | LoRA 增量直接相加 |
| 多任务服务 | 一任务一个 Adapter | 一任务一个 LoRA，可以 batch 混跑 |
| 典型大小 | 基座的 0.5–3% | 0.1–1% |
| 适合场景 | 推理图稳定的设置 | 对延迟敏感的 LLM 服务 |

LoRA 抢走了 Adapter 的大部分市场，主要是因为推理零开销这条性质。Adapter 现在仍有的位置是显式模块化组合的场景——比如多语言里 AdapterFusion（Pfeiffer et al., 2021）在推理时把多个语言专用 Adapter 融合。

---

## 5. 基于提示的 PEFT：不调权重，调输入

一个完全不同的思路：把模型整个冻住，只学**喂给它什么**。

![三种基于提示的 PEFT——输入处的 soft prompt、每层注入 KV 前缀、深层提示遍布所有层。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig4_prefix_prompt.png)

### Prompt-Tuning（Lester et al., 2021）

在输入序列前面拼上 $n$ 个可训练的嵌入向量。就这么简单。在 11B 的 T5 上，Prompt-Tuning 在 SuperGLUE 上能匹配全量微调——但**这个结论只在足够大的模型上成立**。1B 以下的模型上，Prompt-Tuning 差得远。

### Prefix-Tuning（Li & Liang, 2021）

同样的想法但应用到每一层注意力：学一组键和值前缀矩阵 $P_K, P_V$，在注意力计算前拼上去：

$$
\text{Attention}(Q,\ [P_K; K],\ [P_V; V]).
$$

每层都有自己的前缀，相比单一输入提示能更深地改写模型的计算。

### P-Tuning v2（Liu et al., 2022）

去掉了 Prefix-Tuning 用的重参数化 MLP（在不同模型规模下不稳定），让深层提示均匀分布到所有层。它是第一个**在所有规模上**（包括小模型）都能在 NLU 任务上匹配全量微调的提示类方法。

**什么时候用提示类。**（a）需要极致的参数效率——每个任务只占几 KB；（b）想用一个服务化模型背后挂很多个提示。生成类任务上通常打不过 LoRA。

---

## 6. 怎么选

![PEFT 方法在参数效率/质量平面上的位置，以及 LoRA 秩对不同任务类型的影响。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E4%B8%8EPEFT/fig7_perf_vs_params.png)

一个实用的决策树：

```
                    ┌─ 要顶级分数？     → 全量微调（如果你养得起）
                    │
显存够吗？  ────────┤
                    │  够 → LoRA r=16 加在 q/k/v/o
                    └─ 不够 → QLoRA r=16（NF4 + 分页 AdamW）

要从一个基座服务多个任务？  → LoRA（可合并、可换）

延迟敏感的生成任务？        → LoRA 训完合并
编码器 NLU 任务、参数极小？  → P-Tuning v2 或 BitFit
模块化、组合式？            → Adapter（配 AdapterFusion）
```

### 秩怎么选

从 $r = 16$ 开始。从上面右图能看到，大多数任务的边际收益拐点在 $r = 8$ 到 $r = 32$ 之间：

- **简单分类**到 $r = 8$ 就饱和
- **代码生成、推理**到 $r = 32$ 或 $r = 64$ 还能继续涨
- **领域适配**（医疗、法律）通常 $r = 16$–$32$ 比较合适

不确定就扫 $r \in \{8, 16, 32\}$，挑能在留出集上达到目标指标的最小那个。

---

## 7. 对齐：指令微调与 RLHF

PEFT 是杠杆，对齐通常才是你拉这根杠杆的理由。现代 LLM 后训练有两个阶段：

**有监督指令微调。** 用 `（指令，回答）` 对去微调基座，让它学会遵循人类写的提示。质量比数量重要：1K–10K 精挑细选的样本经常打败 10 万条众包数据（Zhou 等人 2023 年的 LIMA 论文，1K 条样本就让 65B 模型有了相当的助手能力）。

```python
def format_example(ex):
    if ex["input"]:
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}")
    return (f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}")
```

**RLHF**（Ouyang 等人 2022 年的 InstructGPT 论文）。SFT 之后，用人类的偏好对训练一个奖励模型，再用 PPO 把策略往这个奖励上推。Bradley–Terry 偏好损失：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\bigl(r_\theta(x, y_{\text{chosen}}) - r_\theta(x, y_{\text{rejected}})\bigr).
$$

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

RL 那一侧的展开见 [RL 第 12 部分：RLHF 与 LLM 应用](/en/reinforcement-learning-12-rlhf-and-llm-applications/)。DPO（Rafailov et al., 2023）是现在很流行的简化替代，不再需要显式的奖励模型。

---

## 8. 端到端配方

```python
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch

MODEL = "meta-llama/Llama-2-7b-hf"

# 1. 4 比特基座
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

# 2. 注意力投影上加 LoRA
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 3,508,801,536 || trainable%: 0.239

# 3. 用 SFTTrainer 训练（自动处理格式化和 collator）
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
model.save_pretrained("./llama-qlora-adapter")  # 磁盘上约 80 MB
```

### 多 LoRA 服务

LoRA 适配器小且可加，所以可以同时把几十个挂在内存里，按请求切换：

```python
# 把多个 adapter 加载到同一个基座
model.load_adapter("./adapter-medical", adapter_name="med")
model.load_adapter("./adapter-legal",   adapter_name="legal")

model.set_adapter("med")
out_a = model.generate(...)

model.set_adapter("legal")
out_b = model.generate(...)
```

vLLM、S-LoRA 这类框架更进一步：把不同 adapter 的请求 batch 在一次前向里跑，靠堆叠 LoRA 增量实现。一张卡上一个 7B 基座，背后服务几百个微调模型。

---

## 常见问题

**什么时候值得做全量微调？** 算力充裕、$\geq$10 万条高质量样本、每个百分点都重要时——比如基座厂商发布旗舰指令模型。其他场景：LoRA 或 QLoRA。

**LoRA 秩怎么挑？** 从 16 起步。欠拟合（训练损失停在很高位置）就升到 32 或 64。过拟合且数据少就降到 4–8。分类任务 8 通常够。

**改哪些模块？** `q_proj` 和 `v_proj` 给你 80% 的收益。再加 `k_proj` 和 `o_proj` 拿剩下的。FFN 三件套（`gate/up/down_proj`）只在生成密集且预算允许的任务上加，可训参数会变成 3 倍。

**LoRA 影响推理吗？** merge 之后不影响。merge 之前每个被包起来的层多一次很小的矩阵乘——能忽略但非零。

**QLoRA 掉多少质量？** 标准 benchmark 上相比 fp16 LoRA 通常掉 1–2 个点，常常在噪声范围内。对绝大多数从业者来说，省下的显存值得。

**指令数据要多少？** LIMA 显示 1K 条精心挑选的样本就能从一个强基座得到一个连贯的助手。实操底线 1K–10K 条高质量样本；质量比数量重要得多。

**能组合 PEFT 方法吗？** 能——LoRA + Prompt-Tuning 是有论文的组合，QLoRA 本身就是栈（4 比特基座 + LoRA + 分页优化器）。Adapter + LoRA 同模型里则不常见。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 7 | 提示工程与 ICL | [<- 上一篇](/zh/自然语言处理-七-提示工程与In-Context-Learning/) |
| **8** | **模型微调与 PEFT（本文）** | |
| 9 | 大语言模型架构深度解析 | [下一篇 ->](/zh/自然语言处理-九-大语言模型架构深度解析/) |

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
