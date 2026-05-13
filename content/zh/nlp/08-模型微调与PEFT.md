---
title: "自然语言处理（八）：模型微调与 PEFT"
date: 2025-11-05 09:00:00
tags:
  - NLP
  - PEFT
  - LoRA
  - LLM
  - 微调
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "深入参数高效微调：LoRA 为什么用低秩更新就够、QLoRA 把 7B 模型塞进 6GB 显存的内存账本、Adapter 与 Prefix-Tuning 的取舍，以及生产环境怎么选。"
disableNunjucks: true
series_order: 8
translationKey: "nlp-8"
polished_by_qwen_max: true
---
2020 年，微调一个 70 亿参数的语言模型需要八张 A100 显卡、几天时间以及一位懂得调试梯度检查点的工程师；而到了 2024 年，研究生只需一台笔记本电脑即可完成。这一变化主要归功于两篇论文：胡等人在 ICLR 2022 提出的 LoRA 和 Dettmers 等人在 NeurIPS 2023 发表的 QLoRA。

这不仅是工程层面的进步，更是一次范式转变——参数高效微调（PEFT）重新定义了“拥有一个模型”的含义。过去每个任务都需要一份完整的权重文件，而现在只需一个冻结的基础模型和一个存放小适配器的目录，每个适配器只有几十兆大小。切换任务只需加载对应适配器；支持 N 个领域时，服务开销从 O(N) 降至 O(1)（单份基础模型）+ O(N)（N 个轻量适配器）。

本文将回归第一性原理，系统梳理 PEFT 的核心思想及其演进逻辑。首先探讨全量微调能解决的问题及其局限，然后推导 LoRA 的低秩假设，拆解 QLoRA 如何通过内存优化将 7B 参数模型塞进 6GB 显存，并最终讨论实际工程中的选择：使用哪种方法、设置多大的秩、修改哪些模块。

<!-- wanx-hero -->
![自然语言处理（八）：模型微调与PEFT — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/fine-tuning-peft/illustration_1.png)
## 你将学到什么

- **为什么** 在大语言模型（LLM）时代，全量微调是一种资源浪费——过参数化现象与内在低秩性的启示  
- **LoRA 的工作原理**：分解公式 $\Delta W = BA$ 的含义、为何 $B$ 的初始值设为零、以及 $\alpha/r$ 缩放如何影响有效学习率  
- **QLoRA 的核心技术**： NF4 量化原理、双重量化机制、分页优化器的设计思路，以及显存占用的精确估算方法  
- **Adapter 和 Prefix-Tuning 的应用**：它们在 Transformer 模块中的具体位置、在哪些场景下表现优异、又在哪些情况下存在不足  
- **实际工程中的决策点**：如何选择合适的秩、目标模块的选择策略、多 LoRA 服务的实现方法、以及通过指令微调和基于人类反馈的强化学习（RLHF）进行模型对齐的最佳实践
## 前置知识

<!-- wanx-mid -->
![自然语言处理（八）：模型微调与 PEFT —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/illustration_2.png)

- 熟悉 Transformer 架构（详见 [第 4 部分](/zh/nlp/04-注意力机制与transformer)）
- 了解 GPT 风格的解码器原理（参考 [第 6 部分](/zh/nlp/06-gpt与生成式语言模型)）
- 掌握 PyTorch 的基本用法，并对 GPU 内存的核心概念（如优化器状态、激活值和梯度）有一定理解

---
## 1. 为什么不直接全量微调？
### 显存开销的现实
全量微调意味着所有参数都需要更新，每个梯度都要存储下来，而优化器（通常是 AdamW）还会为每个参数额外维护两个 fp32 的状态缓冲区。以 7B 参数的模型为例，在混合精度训练下，单步显存占用大致如下：

| 组件 | 每参数字节数 | 7B 模型 |
|------|-------------|---------|
| 权重（fp16） | 2 | 14 GB |
| 梯度（fp16） | 2 | 14 GB |
| AdamW 状态（fp32 m + v） | 8 | 56 GB |
| 激活值（取决于序列长度和 batch 大小） | — | 8–20 GB |
| **总计** | — | **约 95 GB** |

换句话说，还没写一行训练代码，至少需要两张 A100-80GB 显卡才能跑起来。这清楚地说明了为什么 PEFT 不仅仅是一个“锦上添花”的优化手段——对大多数从业者来说，它是能够实际操作 7B 以上大模型的唯一选择。

![PEFT 各方法在 7B 基座上的可训练参数量，以及服务 N 个任务时的磁盘占用对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-模型微调与PEFT/fig1_full_vs_peft.png)

### 内在秩假设的启示

更深层次的原因来自 Aghajanyan 等人（2020）以及 LoRA 论文的核心观点：**微调引入的权重变化通常具有非常低的内在秩**。如果预训练模型的调整本质上局限在一个低维子空间中，那么训练完整的 $d \times k$ 矩阵不仅成本高昂，还选错了搜索空间——应该直接在低秩流形中寻找解。

实验表明，对 175B 参数的模型进行下游任务微调时，只需训练大约 200 个参数方向即可达到与全量微调相当的效果（Aghajanyan et al., 2020）。这一发现正是 LoRA 方法的理论基础。

### 冻结微调——效果有限的基线方法

在 PEFT 出现之前，最简单的节省资源的方法是冻结模型主体，仅训练分类头，或者只解冻顶部几层。例如：

![微调策略决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/fine-tuning-peft/fig_finetuning_decision_zh.png)

### 如何选择秩 $r$

建议从 $r = 16$ 开始。从上图右侧可以看出，大多数任务的收益递减拐点位于 $r = 8$ 到 $r = 32$ 之间：

- **简单分类任务**：$r = 8$ 基本够用  
- **代码生成、逻辑推理**：$r = 32$ 或 $r = 64$ 效果更佳  
- **领域适配任务**（如医疗、法律）：通常 $r = 16$ 到 $r = 32$ 比较合适  

如果不确定最佳值，可以尝试 $r \in \{8, 16, 32\}$，选择能够达到验证集目标指标的最小值即可。
## 7. 对齐：指令微调与 RLHF
PEFT 是手段，对齐才是目的。现代大语言模型的后训练过程通常分为两个关键阶段：

**有监督指令微调**  
通过 `(instruction, response)` 数据对基础模型进行微调，使其能够理解并准确响应人类编写的指令。在这一阶段，数据质量远重于数量： 1K–10K 条精心构造的样本，常优于 10 万条普通众包数据。（Zhou 等人在 2023 年的 LIMA 论文中展示了仅用 1K 条高质量样本，就能让一个 65B 参数的模型表现得非常出色）。

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

关于强化学习的具体内容，可以参考 [RL 第 12 部分：RLHF 与 LLM 应用](/zh/reinforcement-learning/12-rlhf与大语言模型应用)。此外， DPO （Rafailov 等人 2023 年提出）作为一种更简单的替代方法，直接省略了显式的奖励模型，近年来也受到了广泛关注。
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

如果你有充足的计算资源、$\geq$10 万条高质量样本，并且对每个小数点的性能提升都斤斤计较——比如基座模型厂商在推出旗舰级指令模型时。对绝大多数研究者与工程师而言， LoRA 或 QLoRA 已完全满足需求。

### LoRA 的秩该如何选择？

从 16 开始尝试。如果发现欠拟合（训练损失停滞在较高水平），可以提高到 32 或 64。如果数据量较少且出现过拟合，则可以降到 4–8。对于分类任务， 8 通常已经绰绰有余。

### 应该调整哪些模块？

`q_proj` 和 `v_proj` 能带来大约 80% 的效果提升。再加上 `k_proj` 和 `o_proj` 就能覆盖剩余的部分。只有在生成任务占比较高且预算允许的情况下，才需要加入 FFN 的三个模块（`gate/up/down_proj`），因为这会让可训练参数增加 3 倍。

### LoRA 会影响推理速度吗？

merge 后完全不会影响。但在 merge 之前，每层会多一次额外的小矩阵乘法操作——虽然可以忽略不计，但确实存在。

### QLoRA 的质量下降明显吗？

在标准基准测试中，相比 fp16 的 LoRA， QLoRA 通常会有 1–2 个点的下降，基本在误差范围内。对于绝大多数从业者来说，显存节省的价值远远超过这点性能损失。

### 需要多少指令数据？

LIMA 的研究表明，仅用 1K 条精心挑选的样本，就能将一个强大的基座模型训练成一个连贯的助手。实际应用中， 1K–10K 条高质量样本是一个合理的下限；数据的质量远比数量重要。

### PEFT 方法可以组合使用吗？

可以。 LoRA + Prompt-Tuning 是一种有文献支持的组合方式，而 QLoRA 本身就是一个方法栈（4 比特基座 + LoRA + 分页优化器）。不过，在同一个模型中同时使用 Adapter 和 LoRA 则较为少见。
## 9. 一份具体的微调配方（7B 上的 LoRA）

以下是我亲测有效并成功上线的超参数配置，真正对效果有影响。

**数据规模**  
在 7B 模型上使用 LoRA 时， 1000 到 5000 条高质量的指令-响应对就能带来显著收益。如果样本数量少于 500 条，不如直接用 few-shot 提示来得高效。而当数据量超过 20000 条时，除非任务极其复杂，否则投入产出比会大幅下降，属于“花了钱却没赚到分”。

**rank `r`**  
默认值设为 `r=8` 即可。对于代码生成、多步推理等复杂任务，可以将 `r` 提升到 `r=16` 或 `r=32`。增加 `r` 会让参数量翻倍，但训练时间几乎不会显著增加。不过，`r` 超过 64 通常不会有额外收益，反而容易导致过拟合。

**alpha**  
设置 `alpha = 2 * r`，例如当 `r=8` 时，`alpha=16`。`alpha/r` 是 LoRA 路径上的等效学习率缩放因子，将其固定为 2 是一个稳妥的选择。

**目标模块**  
如果显存允许，建议将 LoRA 同时应用到 `q_proj, k_proj, v_proj, o_proj` 和 `gate_proj, up_proj, down_proj` 上。原始论文中仅对注意力模块（attention-only）进行微调，在指令调优场景下会损失不少潜在收益。实际测试表明，结合注意力和 MLP 的方法在 Open LLM Leaderboard 的任务上能额外提升 2 到 4 分。

**学习率**  
LoRA 的初始学习率推荐设为 `2e-4`。相比之下，全量微调通常使用 `2e-5`，而 LoRA 可以承受更高的学习率，因为只有少量参数在更新。建议搭配 cosine 学习率调度，并设置 3% 的 warmup 比例。

**batch size 和梯度累积**  
等效 batch size 控制在 32 到 64 条序列之间为宜。在单张 A100 80GB 显卡上，序列长度为 2048 时， micro-batch 通常只能容纳 2 到 4 条序列，因此需要通过梯度累积达到目标 batch size。

**epoch 数**  
跑 1 到 3 轮即可。观察验证集损失（eval loss），通常在第 2 轮时就会降到最低点。对于小规模指令数据集，跑到 5 轮以上几乎一定会过拟合。

**序列打包**  
将多个短样本打包成一条 2048-token 的序列，并正确设置 attention mask。相比传统的 padding 方法，这种方式能将吞吐量提升 2 到 3 倍。目前 `transformers` 的 SFTTrainer 已原生支持该功能。

**训练成本估算**  
使用 5000 条样本、 7B 基座模型、单张 A100 显卡，整轮训练大约需要 2 到 4 小时，租卡成本约为 $8 到 $12。
## 10. LoRA 悄悄翻车的地方

以下是我在实际项目中见过的三种让团队头疼的失败模式，分享给大家以供参考。

**失败 1：稀有 token 长尾部分的灾难性遗忘**  
虽然 LoRA 在保护基座模型方面比全量微调更胜一筹，但它仍然会对输出分布产生一定影响。举个例子，当我们对一个代码模型进行指令微调后，尽管英文指令的理解能力显著提升，但像 Erlang、 Haskell 这类稀有编程语言的困惑度（perplexity）却飙升了 30% 以上。解决这个问题的办法是，在微调数据中加入 5%-10% 的“基座风格”数据作为回放样本，帮助模型更好地保留对稀有 token 的理解。

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
