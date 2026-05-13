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
2020 年，微调一个 70 亿参数的语言模型还是一项需要专门预算的工程：八张 A100 显卡、几天时间，外加一位懂得调试梯度检查点的工程师；而到了 2024 年，一名研究生用一台笔记本电脑就能完成。从这两个世界之间的鸿沟，几乎完全被两篇论文填平——胡等人（Hu et al.）在 ICLR 2022 提出的 LoRA，以及 Dettmers 等人在 NeurIPS 2023 发表的 QLoRA。

这不仅是工程上的飞跃，更是一次范式重构：参数高效微调（PEFT）彻底改变了“拥有一个模型”的含义。过去，每个任务都需要一份完整的二进制权重文件；如今，你只需保留一个冻结的基础模型，再搭配一个存放小型适配器的小目录，每个适配器仅几十兆字节。切换任务？只需加载对应的适配器；服务 N 个领域？开销变为 O(1) 的基础模型加上 N 个 ε 级别的轻量适配器。

本文将从第一性原理出发，系统重建 PEFT 的逻辑脉络。我们会先审视全量微调能回答什么问题、又遗漏了哪些关键挑战，继而推导 LoRA 的低秩假设，拆解 QLoRA 如何通过精巧的内存设计将 7B 模型塞进 6 GB 显存，并最终落脚于实践决策：选哪种方法、设多大秩、改哪些模块。

<!-- wanx-hero -->
![自然语言处理（八）：模型微调与PEFT — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/illustration_1.png)
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
全量微调意味着所有参数都被解冻，每个梯度都要存储，而优化器（通常是 AdamW）还会为每个参数额外维护两个 fp32 的状态缓冲区。以 7B 模型在混合精度训练下的单步显存占用为例：

| 组件 | 每参数字节数 | 7B 模型 |
|------|-------------|---------|
| 权重（fp16） | 2 | 14 GB |
| 梯度（fp16） | 2 | 14 GB |
| AdamW 状态（fp32 m + v） | 8 | 56 GB |
| 激活值（取决于序列长度和 batch 大小） | — | 8–20 GB |
| **总计** | — | **约 95 GB** |

这意味着，在写任何训练代码之前，你就至少需要两张 A100-80GB 显卡。这张账单清楚地说明：PEFT 不是什么“锦上添花”的优化，而是绝大多数从业者能够触碰 7B+ 模型的唯一途径。

![PEFT 各方法在 7B 基座上的可训练参数量，以及服务 N 个任务时的磁盘占用对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/08-模型微调与PEFT/fig1_full_vs_peft.png)

### 内在秩假设的启示

更深层的原因来自 Aghajanyan 等人（2020）及 LoRA 论文的核心洞见：**微调引入的权重更新具有极低的内在秩**。如果预训练模型所需的调整本质上局限于一个低维子空间，那么训练一个完整的 $d \times k$ 矩阵不仅昂贵，更是错误的假设空间——你应该直接在低秩流形中搜索解。

实证表明，对 175B 模型进行下游任务微调时，仅需训练约 200 个参数方向，就能达到与全量微调相当的效果（Aghajanyan et al., 2020）。这正是 LoRA 的理论钥匙。

### 冻结微调——效果有限的基线方法

在 PEFT 出现前，最简单的节省成本方式是冻结模型主体，只训练分类头，或仅解冻顶部几层。例如：

### 如何选择秩 $r$

建议从 $r = 16$ 开始。从上图右侧可见，大多数任务的收益拐点落在 $r = 8$ 到 $r = 32$ 之间：

- **简单分类任务**：$r = 8$ 基本饱和  
- **代码生成、逻辑推理**：可受益于 $r = 32$ 甚至 $r = 64$  
- **领域适配任务**（如医疗、法律）：通常 $r = 16$–$32$ 更合适  

若不确定，可在 $r \in \{8, 16, 32\}$ 中做小范围搜索，选择能闭合验证集目标指标差距的最小值即可。
## 7. 对齐：指令微调与 RLHF
PEFT 是杠杆，而对齐才是你通常想撬动的目标。现代大语言模型的后训练流程主要由两个阶段主导：

**有监督指令微调**。在 `(instruction, response)` 数据对上微调基础模型，使其学会遵循人类编写的提示。这一阶段，质量远胜数量：1K–10K 条精心筛选的样本，往往优于 10 万条众包数据。（Zhou 等人在 2023 年的 LIMA 论文中甚至证明，仅用 1K 条高质量样本，就能让一个 65B 模型表现出色。）

```python
def format_example(ex):
    if ex["input"]:
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}")
    return (f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}")
```

**RLHF**（Ouyang 等人 2022 年的 InstructGPT 论文）。在 SFT 之后，基于人类偏好对训练奖励模型，并用 PPO 算法优化策略以匹配该奖励。其 Bradley–Terry 偏好损失如下：

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

关于强化学习部分的细节，可参阅 [RL 第 12 部分：RLHF 与 LLM 应用](/zh/reinforcement-learning/12-rlhf与大语言模型应用)。如今，DPO（Rafailov et al., 2023）作为一种更简洁的替代方案，跳过了显式奖励模型，也日益流行。
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

由于 LoRA 适配器体积小且可叠加，你可以同时将数十个适配器驻留在内存中，并按请求动态切换：

```python
# 将多个适配器加载到同一个基础模型中
model.load_adapter("./adapter-medical", adapter_name="med")
model.load_adapter("./adapter-legal",   adapter_name="legal")

model.set_adapter("med")
out_a = model.generate(...)

model.set_adapter("legal")
out_b = model.generate(...)
```

像 vLLM 和 S-LoRA 这类框架更进一步：通过堆叠 LoRA 增量，在单次前向传播中批量处理指向不同适配器的请求。显存中只需一份 7B 基础模型，即可高效服务数百个微调版本。

---
## 常见问题
### 什么情况下适合全量微调？

当你拥有充足算力、≥10 万条高质量样本，且对每个小数点的性能提升都锱铢必较时——比如基座模型厂商发布旗舰级指令模型。对其他人而言：LoRA 或 QLoRA 足矣。

### LoRA 的秩该如何选择？

从 16 开始。若出现欠拟合（训练损失高位平台），可增至 32 或 64；若数据少且过拟合，则降至 4–8。分类任务通常 8 就够了。

### 应该调整哪些模块？

`q_proj` 和 `v_proj` 能带来约 80% 的收益。加上 `k_proj` 和 `o_proj` 可覆盖剩余部分。只有在生成任务占主导且显存允许时，才考虑加入 FFN 三件套（`gate/up/down_proj`），因为这会使可训练参数增加三倍。

### LoRA 会影响推理速度吗？

合并后完全不影响。未合并前，每层会多一次微小的矩阵乘法——虽可忽略，但确实存在。

### QLoRA 的质量下降明显吗？

在标准基准上，相比 fp16 LoRA，QLoRA 通常仅下降 1–2 个点，常在噪声范围内。对绝大多数人来说，显存节省的价值远超这点损失。

### 需要多少指令数据？

LIMA 表明，1K 条精心挑选的样本就足以让强基座模型变成连贯助手。实践中，1K–10K 条高质量样本是合理下限；质量远比数量重要。

### PEFT 方法可以组合使用吗？

可以。LoRA + Prompt Tuning 是有文献支持的组合，QLoRA 本身也是技术栈（4-bit 基座 + LoRA + 分页优化器）。但在同一模型中同时使用 Adapter 和 LoRA 则较为罕见。
## 9. 一份具体的微调配方（7B 上的 LoRA）

以下是我实际使用并上线的超参数配置，真正影响效果的关键项。

**数据规模**。在 7B 基座上用 LoRA 时，1,000–5,000 条高质量指令-响应对就能获得大部分收益。少于 500 条时，few-shot 提示更划算；超过 20,000 条则大概率陷入边际效益递减，除非任务极其复杂。

**Rank `r`**。默认 `r=8`，但对代码生成、多步推理等难题，可提升至 `r=16` 或 `r=32`。`r` 翻倍会使参数量翻倍，但训练时间几乎不变。超过 `r=64` 很少有帮助，反而易过拟合。

**Alpha**。设 `alpha = 2 * r`，即 `r=8` → `alpha=16`。`alpha/r` 是 LoRA 路径上的有效学习率缩放因子，保持为 2 是可靠默认。

**目标模块**。若显存允许，建议同时对 `q_proj, k_proj, v_proj, o_proj` 和 `gate_proj, up_proj, down_proj` 应用 LoRA。仅微调注意力层（原始 LoRA 设置）在指令调优中会损失不少潜力。实测表明，注意力+MLP 组合在 Open LLM Leaderboard 任务上可额外提升 2–4 分。

**学习率**。LoRA 推荐从 `2e-4` 开始。全量微调通常用 `2e-5`，而 LoRA 可承受高 10 倍的学习率，因为仅少量参数更新。配合 cosine 调度和 3% warmup。

**Batch size 和梯度累积**。等效 batch size 控制在 32–64 条序列。单张 A100 80GB、序列长 2048 时，micro-batch 通常为 2–4，需累积达标。

**Epochs**。1–3 轮足矣。验证损失通常在第 2 轮触底，超过 5 轮在小数据集上几乎必然过拟合。

**序列打包**。将多个短样本打包成一条 2048-token 序列，并正确设置 attention mask。相比 padding，吞吐量可提升 2–3 倍。`transformers` 的 SFTTrainer 已原生支持。

在 5K 样本、7B 基座、单张 A100 上，整轮训练约需 2–4 小时，租卡成本约 $8–12。
## 10. LoRA 悄悄翻车的地方

以下是我在项目中见过的三种让团队踩坑的失败模式。

**失败 1：稀有 token 长尾的灾难性遗忘**。LoRA 虽比全量微调更能保护基座模型，但仍会扰动输出分布。曾有一次，对代码模型做指令微调后，英文指令遵循能力提升，但 Erlang、Haskell 等稀有语言的困惑度却飙升 30% 以上。解决方法：在微调数据中混入 5%–10% 的基座风格数据作为回放。

**失败 2：非英语输出的质量隐性下滑**。在英文为主的 SFT 数据上做 LoRA 指令微调，即使基座是多语言的，中文、韩文、日文输出质量也会下降——因为 LoRA 学到了“英文形状”的回答。对策：SFT 数据中至少包含 20% 的目标语言样本。

**失败 3：合并后的 LoRA ≠ 推理时的 LoRA**。当你调用 `model.merge_and_unload()` 将 LoRA 合并进基座模型用于部署时，在 4-bit 量化设置下，合并后的模型性能可能明显劣于未合并版本。原因在于：合并扰动了基座权重，而量化过程的重新取整放大了误差。解决办法：要么推理时保持 LoRA 分离（轻微延迟代价），要么在合并后用预留集做量化校准。
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
