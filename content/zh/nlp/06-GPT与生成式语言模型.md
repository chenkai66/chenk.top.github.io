---
title: "自然语言处理（六）：GPT 与生成式语言模型"
date: 2025-10-26 09:00:00
tags:
  - NLP
  - GPT
  - 深度学习
  - 语言模型
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从 GPT-1 到 GPT-4：理解自回归语言建模、解码策略（贪心、束搜索、top-k、top-p）、上下文学习，并用 HuggingFace 构建聊天机器人。"
disableNunjucks: true
series_order: 6
series_total: 12
translationKey: "nlp-6"
polished_by_qwen_max: true
---
当你向 ChatGPT 提问，看到一段流畅的多段落回答逐 token 流式生成时，你其实正在见证一个看似简单却威力巨大的循环：把“到目前为止的所有内容”喂给 Transformer 解码器，观察它输出的词汇表概率分布，从中挑一个 token 追加到末尾，然后重复——这便是自回归语言模型的全部逻辑。真正神奇的并非这个循环本身，而是当你把循环背后的网络扩展到数千亿参数，并用近乎整个互联网的数据训练后，它所展现出的能力。

如果说 BERT（第五篇）是“理解”之王，那 GPT 就是“生成”之王。本文将带你完整走过 GPT 的演进脉络，拆解自回归解码的机制，让你真切感受到不同解码策略带来的差异，并最终教你如何在自己的笔记本电脑上跑起一个能用的聊天机器人。

<!-- wanx-hero -->
![自然语言处理（六）：GPT与生成式语言模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/illustration_1.png)

---

## 你将学到什么
![自然语言处理（六）：GPT与生成式语言模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/illustration_2.png)

- 仅含解码器的 Transformer 如何通过“预测下一个 token”实现通用人工智能
- **因果（掩码）自注意力**的关键作用——这是 GPT 与 BERT 的根本区别
- 四种经典解码策略（贪心、束搜索、top-k、top-p）及**温度**参数的影响
- **上下文学习**（零样本/少样本）为何无需任何梯度更新就能生效
- 实证发现的**缩放定律**与神秘的**涌现能力**现象
- 如何评估生成文本（BLEU、ROUGE、困惑度），以及各指标的失效场景
- 基于 GPT-2 和 HuggingFace 构建可运行的多轮对话聊天机器人

**前置知识**：第 4 篇（Transformer 架构）、第 5 篇（预训练与 BERT）。

---

## 仅解码器 Transformer

<!-- wanx-mid -->
![NLP (6): GPT 和生成式语言模型 —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/illustration_2.png)

![Decoder-only 架构与因果掩码](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig1_decoder_only_arch.png)

原始 Transformer（Vaswani et al., 2017）包含编码器和解码器两部分：前者读取源句，后者生成目标句。BERT 只保留了编码器，而 GPT 则只保留了**解码器**，并做了一个关键改动：每一层自注意力都使用**因果掩码**，使得位置 $i$ 只能关注 $1, 2, \ldots, i$，绝不能窥探未来。

为什么这一设计如此重要？因为它让训练和推理完全一致。训练时模型虽能看到整句话，但掩码会遮住未来，迫使每个位置仅凭过去预测下一个词——这正是推理时的真实场景。没有训练/推理不匹配，也无需切换“生成模式”：训练时计算损失的前向传播，与推理时生成下一 token 的过程完全相同。

### 前向传播，一气呵成

给定输入 tokens $x_1, \ldots, x_t$：

1. **嵌入**：$h^0_i = E_{\text{tok}}(x_i) + E_{\text{pos}}(i)$
2. **重复 $L$ 次**（一个 Transformer 块）：
$$\tilde h = h + \text{MaskedMHA}(\text{LN}(h)), \quad h \leftarrow \tilde h + \text{FFN}(\text{LN}(\tilde h))$$
3. **投影**：logits $z_i = W_o\,h^L_i$，随后 $P(x_{i+1}\mid x_{\le i}) = \text{softmax}(z_i)$。

### 因果掩码，具体呈现

掩码注意力仍是缩放点积形式，只是在 softmax 前对禁止位置加上 $-\infty$：
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right) V,
\quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$-\infty$ 经 softmax 后变为 0，因此未来 token 完全无贡献。上图右侧面板直观展示了这一点：每行是一个查询位置，每列是一个键，只有下三角（过去）是活跃的。

### 训练目标

最大化语料库的对数似然——即对序列中每个位置的交叉熵求和：
$$\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_1, \ldots, x_{i-1})$$
仅凭这一个损失函数，在 TB 级文本上训练，便是 GPT 的全部故事。

---

## 自回归生成，步步为营

![自回归生成：逐个生成 token](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig2_autoregressive_step.png)

推理时，模型逐 token 生成。每一步包含三件事：

1. 对当前序列执行**前向传播**，仅获取**最后一个位置**的 logits。
2. 将 logits 转为词汇表上的概率分布（可选地引入温度调节）。
3. **选择**下一个 token（确定性或采样），追加后重复，直至遇到 `<eos>` 或达到长度上限。

上图底栏展示了提示 `"The cat sat on the"` 后的下一 token 分布：`mat` 以约 42% 的概率胜出，但大量概率质量也分布在合理选项上（如 `floor`、`couch`、`sofa` 等）。你的模型是每次都输出相同内容，还是带来惊喜，完全取决于**如何从该分布中采样**——这将在[第 4 节](#GPT-家族：5-年，参数增长近万倍)详述。

> **关于速度的实用提示**。朴素地生成 $T$ 个 token 需要 $T$ 次前向传播，其成本随序列长度平方增长——难以承受。实际实现均采用 **KV 缓存**：过去 token 的 keys 和 values 被缓存并复用，新 token 仅需与缓存向量做注意力计算。这使生成总 FLOPs 从 $O(T^3)$ 降至 $O(T^2)$，也是现实系统可用的根本原因。

---

## GPT 家族：5 年，参数增长近万倍

![GPT-1 到 GPT-4 的演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig4_gpt_evolution.png)

GPT 系列讲述了一个简单想法被不断放大后的故事。

### GPT-1（2018 年 6 月）——概念验证

- 12 层解码器，**1.17 亿**参数，BooksCorpus（约 4.5 GB）。
- 方法：先用语言建模目标预训练，再为每个下游任务**微调**一个小任务头。
- 启示：一个通用 Transformer 解码器，仅靠原始文本预训练，就能在广泛基准上击败手工设计的任务专用架构。

### GPT-2（2019 年 2 月）——零样本能力浮现

- **15 亿**参数，WebText（约 40 GB，来自 Reddit 出站链接）。
- 新思路：完全跳过微调。只需在提示中**描述任务**，模型往往就能完成。

```text
Translate to French:
English: The cat sat on the mat.
French:
```

GPT-2 能生成尚可的翻译，尽管从未见过平行语料。它在网页上接触了足够多的双语文本，学会了“English: ... / French: ...”这类模式。

### GPT-3（2020 年 5 月）——规模改变游戏规则

- **1750 亿**参数（约 GPT-2 的 100 倍），约 570 GB 精选文本，训练算力达 $3.14\times 10^{23}$ FLOPs。
- 核心发现：**少样本上下文学习**。在提示中放入少量输入-输出示例，模型即可即时掌握任务，**无需任何梯度更新**。
- 许多能力（三位数算术、代码补全、多步推理）在 GPT-2 规模几乎不存在，却在 GPT-3 规模突然出现——这正是[第 7 节](#涌现能力)将讨论的**涌现**现象。

### GPT-4（2023 年 3 月）——多模态与指令优化

- 架构与参数量未公开（传闻为万亿级稀疏专家模型）。
- 支持文本与图像输入。
- 通过 **RLHF**（基于人类反馈的强化学习）大幅优化安全性、有用性与指令遵循能力。
- 律师考试达前 10%，AP 考试得 5 分，并能处理早期模型无法胜任的长程推理。

| 模型 | 参数量 | 训练数据 | 标志性能力 |
|------|--------|----------|------------|
| GPT-1 | 1.17 亿 | 4.5 GB | 预训练 + 微调有效 |
| GPT-2 | 15 亿 | 40 GB | 零样本来自提示 |
| GPT-3 | 1750 亿 | 570 GB | 少样本上下文学习 |
| GPT-4 | （未公开） | （未公开） | 多模态、强推理、RLHF |

---

## 解码策略

解码策略的选择能让生成内容从**重复呆板**变为**惊喜且切题**，全程无需重训任何权重。下文将在[第 2 节](#仅解码器-Transformer)的**同一**下一 token 分布上，可视化四种经典策略。

![贪心、top-k、top-p 和温度对同一分布的影响](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig5_sampling_strategies.png)

### 贪心解码

始终选择最高概率 token：$x_t = \arg\max_w P(w \mid x_{<t})$。

```python
def greedy_decode(model, tokenizer, prompt, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**优点**：确定性强、速度快。**缺点**：易陷入退化循环（如 `"the the the"`），生成内容枯燥可预测。仅适用于需可复现性的调试场景。

### 束搜索

每步维护 top-$k$ 条部分序列，按**累计**对数概率排序（可选长度惩罚以避免偏好短序列）。

```python
def beam_search(model, tokenizer, prompt, beam=5, max_new=100, lp=0.6):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    beams = [(ids, 0.0)]
    for _ in range(max_new):
        cands = []
        for seq, score in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id:
                cands.append((seq, score)); continue
            logp = torch.log_softmax(model(seq).logits[:, -1, :], dim=-1)
            top_lp, top_id = logp.topk(beam, dim=-1)
            for p, idx in zip(top_lp[0], top_id[0]):
                new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                cands.append((new_seq, score + p.item()))
        cands.sort(key=lambda x: x[1] / (x[0].size(1) ** lp), reverse=True)
        beams = cands[:beam]
    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)
```

**优点**：似然高于贪心，是机器翻译与摘要的主力。**缺点**：在开放式生成中易产出**平淡泛化**的输出——即“**束搜索诅咒**”（最高似然句子往往最无聊）。

### Top-$k$ 采样

仅从 $k$ 个最高概率 token 中采样（重归一化使其和为 1）：

```python
def top_k_sample(model, tokenizer, prompt, k=50, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        top_logits, top_ids = logits.topk(k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        nxt = top_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**注意**：$k$ 固定。若模型极自信（一 token 占 95%），top-$k$ 仍考虑 $k$ 个选项，可能选出荒谬结果；若模型在数百 token 上真不确定，$k=50$ 又太局限。

### Top-$p$（核采样）

选取累积概率 $\ge p$ 的最小 token 集合——即**核（nucleus）**——并从中采样。核大小会**自适应**模型置信度。

```python
def top_p_sample(model, tokenizer, prompt, p=0.9, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        sorted_logits, sorted_ids = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep = cum <= p
        keep[..., 0] = True               # 至少保留 top-1
        keep[..., 1:] = keep[..., :-1].clone() | (cum[..., :-1] <= p)
        sorted_logits[~keep] = -float("inf")
        nucleus_probs = torch.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(nucleus_probs, 1)
        nxt = sorted_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

上图 (c) 中核含 5 个 token，因前 5 个已覆盖 85% 概率。换一分布，核可能是 1 个（极自信）或 200 个（极平坦）。正是这种自适应性使 top-$p$ 成为开放式生成的默认选择。

### 温度

温度 $T$ 在 softmax **前**对 logits 缩放：
$$P_T(w) = \text{softmax}(z / T)$$
上图 (d) 展示了同一 logits 在 $T = 0.5$ 与 $T = 1.5$ 下的效果。直观上，$T$ 控制分布的“尖锐度”：

- $T \to 0$：分布塌缩至 argmax 的 one-hot（等价贪心）。
- $T = 1$：原始分布（不变）。
- $T \to \infty$：分布趋于均匀。

经验法则：**top-$p = 0.9$ 配 $T = 0.7$–$0.9$** 是多数聊天/创意写作应用的默认配置。

### 策略速查表

| 策略   | 多样性 | 质量        | 速度  | 最佳场景                |
|--------|--------|-------------|-------|-------------------------|
| 贪心   | 无     | 低          | 快    | 可复现调试              |
| 束搜索 | 低     | 高似然      | 慢    | 翻译、摘要              |
| Top-$k$| 中     | 中-高       | 中    | 通用生成                |
| Top-$p$| 中-高  | 高          | 中    | 聊天、故事、对话        |

---

## 缩放定律：涌现前的可预测性

![缩放定律：损失随规模幂律下降](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig3_scaling_laws.png)

2020 年 Kaplan 等人发现，测试损失随三个量——模型参数 $N$、数据集大小 $D$、训练算力 $C$——呈**清晰幂律**下降，前提是三者均非瓶颈：
$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$
实测指数极小（$\alpha_N \approx 0.076$，$\alpha_C \approx 0.050$）。在双对数坐标下呈直线，故上图两面板均为线性。两大推论：

1. **可预测大模型表现**：从 $\le 1$ B 参数实验即可预估 1750 亿模型的损失。这使 GPT-3 的巨额投入变得经济可行——团队在花数百万美元买 GPU 前，就大致知道损失曲线会落在何处。
2. **存在不可逾越的损失下限**（虚线）——即数据本身的熵。再多缩放也无法突破。

2022 年 **Chinchilla** 论文进一步完善：固定算力下，GPT-3 **训练严重不足**。算力最优配方约为**每参数 20 个 token**——因此用 1.4 T tokens 训练的 70 B 模型（Chinchilla）在同算力下优于用 300 B tokens 训练的 175 B 模型（GPT-3）。现代开源模型（LLaMA、Mistral、Qwen）均采用 Chinchilla 式缩放。

---

## 涌现能力

![涌现能力：随规模出现的尖锐相变](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig6_emergent_capabilities.png)

缩放定律表明**整体损失**平滑下降，但许多**单项任务**并非如此——它们在数量级尺度上表现如随机，随后突然跃升至高准确率。

图中展示典型形态：情感分类（蓝）从小规模起平滑提升；少样本上下文学习、三位数算术、思维链推理（紫、橙、绿）在临界规模前近乎随机，之后陡升。Wei 等（2022）在 **BIG-Bench** 基准中记录了 137 个此类任务。

涌现是否“真实”，抑或仅是测量方式（如离散任务的精确匹配准确率）所致，仍有争议——Schaeffer 等（2023）表明改用连续指标可使部分曲线变平滑。但操作事实不变：小模型**无法**完成任务；参数在 $10^{10}$ 至 $10^{11}$ 间某处，它就能完成。规划路线时务必考虑此点。

---

## 上下文学习

![少样本上下文学习](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig7_in_context_learning.png)

大型 GPT 模型最令人惊讶的特性之一：仅通过在提示中展示示例，就能教会它新任务。无需反向传播、优化器或参数更新——模型纯靠“阅读”适应。

### 零样本 vs. 少样本

```text
# 零样本
将以下英文句子翻译成法语：
The bird flew in the sky.
法语：

# 少样本
英译法。
English: The cat sat on the mat. -> French: Le chat s'est assis sur le tapis.
English: The dog ran in the park. -> French: Le chien a couru dans le parc.
English: The bird flew in the sky. -> French:
```

少样本提示为模型提供可模仿的**格式**。实证表明，多数任务 2–5 个示例已足够；超过约 8 个后收益递减。

### 为何有效？

机制仍是研究热点，主流解释包括：

- **大规模模式匹配**：预训练暴露于海量输入-输出格式，提示激活匹配模板。
- **隐式梯度下降**：Garg 等（2022）等表明，在玩具线性回归设定中，Transformer 可在其前向传播中对上下文示例实现**单步梯度下降**。模型越大，步数越多。
- **贝叶斯隐任务推断**：Xie 等（2022）将 ICL 视为从示例推断隐藏“任务”变量，再据此生成。
- **规模相变**：ICL 仅在数十亿参数以上有效——与涌现故事一致。

### 实用提示工程配方

```python
def few_shot_prompt(task, examples, query):
    parts = [task, ""]
    for inp, out in examples:
        parts.append(f"Input: {inp}\nOutput: {out}\n")
    parts.append(f"Input: {query}\nOutput:")
    return "\n".join(parts)
```

几条经验证有效的经验法则：

1. **明确说明任务**：在开头用直白语言陈述任务。
2. **选代表性示例**：覆盖测试时预期的多样性。
3. **保持格式一致**：分隔符、大小写、标签统一。
4. **推理任务用思维链**：要求“逐步思考”或在少样本中包含完整推理。这在数学与逻辑基准上显著提升准确率。

---

## 评估生成文本

没有完美指标，但知道该用哪个近似指标及其盲点至关重要。

### BLEU（翻译）

衡量生成文本对参考文本的 n-gram 精确率，带长度惩罚：
$$\text{BLEU} = \text{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu(generated: str, reference: str) -> float:
    return sentence_bleu([reference.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method1)
```

与人工评判在**翻译**上强相关；在开放式生成上弱相关。

### ROUGE（摘要）

衡量 n-gram **召回率**——参考文本中有多少出现在生成中：

```python
from rouge_score import rouge_scorer
def rouge(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in s.items()}
```

### 困惑度（内在指标）

模型对保留文本的“惊讶度”。越低越好：
$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i\mid w_{<i})\right)$$
```python
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**enc, labels=enc.input_ids).loss
    return torch.exp(loss).item()
```

困惑度是训练中有用的**内在**信号，但**低困惑度不等于好聊天机器人**——模型可能精于预测常见模式，但仍会幻觉。

### 各任务推荐指标

| 任务                  | 推荐                     |
|-----------------------|--------------------------|
| 机器翻译              | BLEU、chrF、COMET（学习式）|
| 摘要                  | ROUGE、BERTScore         |
| 开放式对话            | 人工评估 + 多样性（distinct-n）|
| 指令遵循              | LLM-as-judge + 人工评估  |
| 语言建模              | 困惑度                   |

对任何面向用户的场景，**人工评估仍是金标准**。

---

## 用 GPT-2 构建聊天机器人

GPT-2 足够小，可在 CPU 运行；又足够大，能生成有趣内容。HuggingFace `transformers` 库几乎屏蔽了所有样板代码。

### 单轮对话

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class ChatBot:
    def __init__(self, name: str = "gpt2") -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def respond(self, user: str, max_new: int = 100,
                temperature: float = 0.8, top_p: float = 0.9) -> str:
        prompt = f"User: {user}\nAssistant:"
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Assistant:")[-1].strip()

print(ChatBot().respond("What is machine learning?"))
```

### 多轮对话（带滚动历史）

```python
class MultiTurnBot(ChatBot):
    def __init__(self, name: str = "gpt2") -> None:
        super().__init__(name)
        self.history: list[str] = []

    def respond(self, user: str, max_turns: int = 5, **gen_kwargs) -> str:
        # 只保留最近几轮对话，确保总长度不超过 1024 token 的上下文限制
        recent = self.history[-2 * max_turns:]
        prompt = "\n".join(recent + [f"User: {user}", "Assistant:"])
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=gen_kwargs.get("max_new", 100),
                do_sample=True,
                temperature=gen_kwargs.get("temperature", 0.8),
                top_p=gen_kwargs.get("top_p", 0.9),
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        reply = self.tokenizer.decode(out[0], skip_special_tokens=True)
        reply = reply.split("Assistant:")[-1].split("User:")[0].strip()
        self.history += [f"User: {user}", f"Assistant: {reply}"]
        return reply
```

### Pipeline API 一行实现

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("User: What is deep learning?\nAssistant:",
          max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)[0]["generated_text"])
```

> GPT-2 **不会**像 ChatGPT——它未经指令微调或 RLHF。要获得类似行为，需用指令微调开源模型（如 `meta-llama/Llama-3-8B-Instruct`、`Qwen/Qwen2.5-7B-Instruct`），或自行用指令数据集微调 GPT-2。

---

## GPT vs. BERT，一表看懂

| 维度        | BERT                                      | GPT                                         |
|-------------|-------------------------------------------|---------------------------------------------|
| 架构        | 编码器，双向自注意力                      | 解码器，**因果**自注意力                    |
| 预训练      | 掩码 LM + 下一句预测                      | 因果语言建模                                |
| 上下文      | 同时见左右                                | 仅见左侧（过去）                            |
| 擅长        | 分类、NER、问答、检索                     | 生成、对话、代码、创意文本                  |
| 适配        | 每任务微调任务头                          | 提示（零/少样本）或微调                     |
| 缩放效应    | 约 10 亿参数后收益递减                    | 持续提升；涌现能力出现                      |

一个有用的心智模型：**BERT 是搜索引擎，GPT 是作家**。需输入向量表示时用 BERT 家族；需生成文本时用 GPT 家族。

---

## 必须坦诚的局限

- **幻觉**：对未知内容，默认生成流畅但错误的文本。缓解方案：检索增强（第十篇）、工具调用、校准拒绝训练。
- **上下文长度**：GPT-2 为 1024，GPT-3 为 2048，GPT-4 达 32K 后至 128K；如今 1M 已常见。但长上下文昂贵，且中间信息易被遗忘（“迷失中间”）。
- **算力成本**：1750 亿参数模型单次前向需数百 GB GPU 显存。模型生命周期中，推理成本远超训练。
- **训练数据偏见**：模型复现训练数据中的统计模式——包括刻板印象。
- **可控性有限**：对风格、格式或事实性的硬约束需额外机制（系统提示、约束解码、RLHF、后过滤）。

---

## 总结

- GPT 模型本质是**仅含解码器的 Transformer 加因果掩码**，目标为预测下一 token。简洁即是精髓。
- **规模是乘数**：参数、数据、算力共同推动损失以幂律下降，并在临界点触发能力跃迁。
- **上下文学习**让单一固定模型通过提示处理数千任务——多数场景无需微调。
- **解码策略是免费杠杆**：聊天与创意写作首选 **top-$p$ + 温度 0.7–0.9**；翻译与可复现场景用贪心或束搜索。
- **BERT 理解，GPT 生成**。二者覆盖现代 NLP 全谱，第七至十二篇将深入探讨如何高效驾驭 GPT 类模型（提示、微调、RAG、多模态等）。
