---
title: "自然语言处理（六）：GPT与生成式语言模型"
date: 2024-06-06 09:00:00
tags:
  - NLP
  - GPT
  - 深度学习
  - 语言模型
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 6
  total: 12
lang: zh-CN
mathjax: true
description: "从GPT-1到GPT-4：理解自回归语言建模、解码策略（贪心、束搜索、top-k、top-p）、上下文学习，并用HuggingFace构建聊天机器人。"
disableNunjucks: true
---

当你向 ChatGPT 提一个问题，看到一段流畅的多段回答一个 token 接一个 token 流式涌出时，你目睹的其实是一个朴素到惊人的循环：把"目前为止的所有内容"喂给一个 Transformer 解码器，看它输出的词表概率分布，挑一个 token，拼到末尾，再循环。这就是自回归语言模型干的全部事情。神奇的不是这个循环，而是当你把循环背后的网络放大到几千亿参数、用半个互联网训练时，会发生什么。

如果说 BERT（第 5 篇）是"理解之王"，GPT 就是"生成之王"。本文会把 GPT 的整条家谱讲清楚，把自回归解码的机制掰开来看，让你对"采样策略到底意味着什么"有切身感受，最后落到一个能在你笔记本上跑起来的聊天机器人。

## 你将学到什么

- 一个**仅解码器（decoder-only）** 的 Transformer 是怎么把"预测下一个 token"变成通用 AI 的
- **因果（掩码）自注意力**——把 GPT 和 BERT 区分开的那一处关键设计
- 四种主流解码策略（贪心、束搜索、top-k、top-p）以及**温度**的作用
- 为什么**上下文学习**（zero/few-shot）能在不更新任何参数的前提下生效
- **缩放定律**与"涌现能力"这一奇怪现象
- 如何评估生成文本（BLEU、ROUGE、困惑度），以及它们各自在哪里失灵
- 用 HuggingFace 在 GPT-2 上搭一个能跑的多轮聊天机器人

**前置阅读**：第 4 篇（Transformer 架构）、第 5 篇（预训练与 BERT）。

---

## 1. 仅解码器 Transformer

![Decoder-only 架构与因果掩码](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig1_decoder_only_arch.png)

最初的 Transformer（Vaswani 等，2017）由两半组成：编码器读源句子、解码器写目标句子。BERT 只保留了编码器；GPT 只保留**解码器**，并且做了一个关键改动——每一层自注意力都加上**因果掩码**，使得位置 $i$ 只能注意到 $1,2,\ldots,i$，永远看不到未来。

为什么这一处设计如此重要？因为它让训练和推理变得**完全一致**。训练时模型一次性看到整段文本，但掩码屏蔽了未来，所以每个位置都被迫"只用过去预测下一个"——这正是它在生成时面临的处境。没有训练/推理的不匹配，也不需要单独的"生成模式"：训练时计算损失的那次前向传播，就是推理时计算下一个 token 分布的那次前向传播。

### 一口气讲完前向传播

给定输入 $x_1,\ldots,x_t$：

1. **嵌入**：$h^0_i = E_{\text{tok}}(x_i) + E_{\text{pos}}(i)$
2. **重复 $L$ 次**（一个 Transformer block）：

$$
\tilde h = h + \text{MaskedMHA}(\text{LN}(h)),\quad h \leftarrow \tilde h + \text{FFN}(\text{LN}(\tilde h))
$$

3. **投影**：logits $z_i = W_o\,h^L_i$，再 $P(x_{i+1}\mid x_{\le i}) = \text{softmax}(z_i)$。

### 因果掩码，具体长什么样

掩码注意力跟普通的缩放点积一模一样，只不过在 softmax 之前给被禁止的位置加上 $-\infty$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right) V,
\quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$-\infty$ 经过 softmax 变成 $0$，所以未来位置的贡献为零。上面右图把这件事可视化了：每一行是一个查询位置，每一列是被注意的位置，只有下三角（过去）是亮的。

### 训练目标

最大化语料的对数似然——也就是把每个位置的交叉熵加起来：

$$
\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_1, \ldots, x_{i-1})
$$

把这一个损失函数应用到 TB 量级的文本上，就是 GPT 的全部训练故事。

---

## 2. 自回归生成，一步一步看

![自回归生成：一次一个 token](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig2_autoregressive_step.png)

推理时模型每次只生成一个 token，每一步发生三件事：

1. **前向传播**当前序列，只取**最后一个位置**的 logits。
2. **把 logits 转成词表上的概率分布**（可选地除以温度）。
3. **挑一个 token**（确定地或采样地），拼到末尾，重复直到撞上 `<eos>` 或长度上限。

下面那张柱状图展示了 prompt `"The cat sat on the"` 之后的下一 token 分布：`mat` 以约 42% 的概率领先，但还有不少概率分散在合理的备选上（`floor`、`couch`、`sofa`…）。模型每次会输出同一句话还是会给你惊喜，**完全取决于你怎么从这个分布里采样**——这正是第 4 节要讲的。

> **一个关于速度的工程细节**。朴素地生成 $T$ 个 token 意味着要跑 $T$ 次前向传播，每次的代价随序列长度二次增长——根本跑不起来。所有真实实现都会用 **KV cache**：过去 token 的 keys 和 values 算一次就缓存下来，新 token 只需要和缓存做注意力。这把生成的总 FLOPs 从 $O(T^3)$ 降到 $O(T^2)$，是真实系统能用的根本原因。

---

## 3. GPT 家族：5 年，约 10000 倍参数增长

![GPT-1 到 GPT-4 的演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig4_gpt_evolution.png)

GPT 系列讲的是一个故事：把一个小想法的"规模"旋钮拧到底会发生什么。

### GPT-1（2018 年 6 月）——概念验证

- 12 层解码器、**1.17 亿**参数，BooksCorpus（约 4.5 GB）。
- 范式：先用语言建模目标预训练，再为每个下游任务微调一个小的任务头。
- 结论：一个通用的 Transformer 解码器，在原始文本上预训练之后，能在一系列基准上击败手工设计的任务专用架构。

### GPT-2（2019 年 2 月）——零样本能力浮现

- **15 亿**参数，WebText（约 40 GB，从 Reddit 出站链接抓取）。
- 新想法：干脆不微调。如果你只在 prompt 里**描述**任务，模型经常就能把它做出来。

```text
Translate to French:
English: The cat sat on the mat.
French:
```

GPT-2 能给出过得去的翻译，尽管它从未在平行语料上训练过——它在网上看过足够多的双语段落，把"English: … / French: …"这种**模式**吸收下来了。

### GPT-3（2020 年 5 月）——规模改变游戏规则

- **1750 亿**参数（约为 GPT-2 的 100 倍），约 570 GB 精筛文本，约 $3.14\times 10^{23}$ FLOPs 的训练算力。
- 标志性发现：**少样本上下文学习**——在 prompt 里塞几个输入-输出样例，模型就能即时学会任务，**不更新一个参数**。
- 许多能力（三位数算术、代码补全、多步推理）在 GPT-2 规模上几乎是零，到 GPT-3 规模突然出现——这正是我们第 7 节要讲的**涌现**现象。

### GPT-4（2023 年 3 月）——多模态 + 指令微调

- 架构和参数量未公开（坊间猜测是万亿参数级别的 MoE）。
- 同时接收文本和图像。
- 用 **RLHF**（基于人类反馈的强化学习）大量打磨了安全性、有用性和指令遵循。
- 律考能上 90 分位、AP 考试拿 5 分、能处理早期模型搞不定的长程推理。

| 模型 | 参数量 | 训练数据 | 标志能力 |
|------|--------|----------|----------|
| GPT-1 | 1.17 亿 | 4.5 GB | 预训练 + 微调跑通 |
| GPT-2 | 15 亿 | 40 GB | 从 prompt 做零样本 |
| GPT-3 | 1750 亿 | 570 GB | 少样本上下文学习 |
| GPT-4 | （未公开） | （未公开） | 多模态、强推理、RLHF |

---

## 4. 解码策略

解码策略这一个旋钮，能让生成结果从"机械重复、乏味"变成"出人意料、贴题"——一个权重都不用改。下面我们用第 2 节那个**同一个**下一 token 分布，把四种主流策略并排可视化。

![贪心、top-k、top-p 与温度对同一分布的效果](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig5_sampling_strategies.png)

### 4.1 贪心解码

每一步选概率最高的那个：$x_t = \arg\max_w P(w \mid x_{<t})$。

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

**优点**：确定性、快。**缺点**：容易陷入退化循环（`"the the the"`），输出乏味、可预测。一般只在需要可复现的调试场景里用。

### 4.2 束搜索（Beam Search）

每一步同时维护 $k$ 条最优部分序列，按**累计**对数概率排序（通常加一个长度惩罚，避免偏爱短序列）。

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

**优点**：似然比贪心高，是机器翻译和摘要的主力。**缺点**：在开放式生成里容易给出**平淡、保守**的句子——这一现象叫"束搜索诅咒"（似然最高的句子往往是最无趣的）。

### 4.3 Top-$k$ 采样

只从概率最高的 $k$ 个 token 里采样（采样前对它们的概率重新归一化）：

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

**问题**：$k$ 是固定的。模型很自信时（一个 token 占了 95% 的概率），top-$k$ 还是会考虑 $k$ 个候选，可能采到傻东西；模型在几百个 token 上都不确定时，$k=50$ 又太苛刻。

### 4.4 Top-$p$（核采样）

挑出累计概率 $\ge p$ 的最小 token 集合——也就是**核（nucleus）**——然后从中采样。核的大小**自适应**地随模型置信度变化。

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

上图的 (c) 子图里，核里有 5 个 token，因为前 5 个就覆盖了 85% 的概率。换一个分布，核可能只有 1 个 token（模型非常自信）或 200 个（分布很平）。这种自适应性是 top-$p$ 成为开放式生成默认选择的根本原因。

### 4.5 温度

温度 $T$ 在 softmax **之前**对 logits 做缩放：

$$
P_T(w) = \text{softmax}(z / T)
$$

上图的 (d) 子图把 $T = 0.5$ 和 $T = 1.5$ 在同一组 logits 上画在一起。直觉上，$T$ 控制分布的"尖锐程度"：

- $T \to 0$：分布塌缩到 argmax 上的 one-hot（等价于贪心）。
- $T = 1$：原始分布（不变）。
- $T \to \infty$：分布趋向均匀。

经验法则：**top-$p = 0.9$ 配合 $T = 0.7$–$0.9$** 是大多数对话和创意写作场景的默认配置。

### 策略速查表

| 策略 | 多样性 | 质量 | 速度 | 最佳场景 |
|------|--------|------|------|----------|
| 贪心 | 无 | 低 | 快 | 可复现调试 |
| 束搜索 | 低 | 似然高 | 慢 | 翻译、摘要 |
| Top-$k$ | 中 | 中-高 | 中 | 通用生成 |
| Top-$p$ | 中-高 | 高 | 中 | 对话、故事、创意写作 |

---

## 5. 缩放定律：涌现之前的可预测性

![缩放定律：损失随规模幂律下降](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig3_scaling_laws.png)

2020 年 Kaplan 等发现，测试损失随三个量呈**干净的幂律**下降——模型参数 $N$、数据集规模 $D$、训练算力 $C$——只要三者都不是瓶颈：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

经验指数小得吓人（$\alpha_N \approx 0.076$、$\alpha_C \approx 0.050$）。在对数-对数坐标上这就是直线，所以上图两个面板都是线性的。两个推论：

1. **可预测性**。你可以从 $\le 1$ B 参数的实验外推出 1750 亿参数模型的损失。这正是当年敢押注 GPT-3 的经济基础——团队在烧掉几百万美元 GPU 费用**之前**就大致知道损失曲线会落在哪。
2. **存在不可约损失下限**（图中虚线）——也就是数据自身的熵。再怎么放大也跨不过去。

2022 年的 **Chinchilla** 论文又把这幅图修正了一次：在固定算力预算下，GPT-3 其实**训练量严重不足**。算力最优的配比大约是**每个参数对应 20 个 token**——所以一个 70 B 模型用 1.4 T tokens 训练（Chinchilla），在同等算力下能打过用 300 B tokens 训练的 175 B 模型（GPT-3）。今天的开源主流模型（LLaMA、Mistral、Qwen）都遵循 Chinchilla 风格的配比。

---

## 6. 涌现能力

![涌现能力：随规模出现的尖锐相变](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig6_emergent_capabilities.png)

缩放定律说**整体损失**平滑下降，但很多**单个任务**并不平滑——它们在好几个数量级的规模上停在随机水平，然后突然蹿到高准确率。

上图给了定性形状：情感分类（蓝）从小规模就平滑提升；少样本上下文学习、三位数算术、思维链推理（紫、橙、绿）在临界规模之前几乎是随机水平，到了临界点忽然起飞。Wei 等（2022）在 **BIG-Bench** 基准上整理了 137 个这样的任务。

涌现到底"真存在"还是测量方式（用离散任务的精确匹配准确率）造成的假象，学界仍有争议——Schaeffer 等（2023）证明换成连续指标后，部分涌现曲线会变平滑。但操作层面上的事实没变：小规模时模型**做不到**这件事，参数规模到了 $10^{10}$–$10^{11}$ 之间的某处时它**能**了。规划路线图时务必把这点考虑进去。

---

## 7. 上下文学习

![少样本上下文学习](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT%E4%B8%8E%E7%94%9F%E6%88%90%E5%BC%8F%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/fig7_in_context_learning.png)

GPT 类大模型最让人吃惊的性质大概是：你能**只在 prompt 里给几个例子**就教会它新任务。没有反向传播、没有优化器、没有参数更新——模型纯粹通过"读到的内容"来适应。

### 零样本 vs. 少样本

```text
# 零样本
请把下面这句英文翻译成法语：
The bird flew in the sky.
法语：

# 少样本
英译法。
English: The cat sat on the mat. -> French: Le chat s'est assis sur le tapis.
English: The dog ran in the park. -> French: Le chien a couru dans le parc.
English: The bird flew in the sky. -> French:
```

少样本 prompt 给了模型一个**格式**去模仿。经验上，2–5 个例子对大多数任务足够；超过 8 个之后收益就平了。

### 它为什么能 work？

机制目前仍是活跃研究方向，主流解释有：

- **大规模模式匹配**。互联网预训练让模型见过无数种输入-输出格式，prompt 激活了对应的模板。
- **隐式梯度下降**。Garg 等（2022）证明，在线性回归这类玩具场景下，Transformer 在前向传播中可以对 in-context 示例**实现一步梯度下降**。模型越大，能模拟的步数越多。
- **贝叶斯隐任务推断**。Xie 等（2022）把 ICL 解释为先从示例里推断出一个隐藏的"任务变量"，再以它为条件做生成。
- **规模相变**。ICL 只在几十亿参数以上才稳定生效——和涌现的故事是一致的。

### 实战 prompt 工程

```python
def few_shot_prompt(task, examples, query):
    parts = [task, ""]
    for inp, out in examples:
        parts.append(f"Input: {inp}\nOutput: {out}\n")
    parts.append(f"Input: {query}\nOutput:")
    return "\n".join(parts)
```

几条经常奏效的经验：

1. **明确**。在最上面用一句大白话说清任务。
2. **挑有代表性的样例**。覆盖你预期会遇到的输入分布。
3. **格式严格一致**。同样的分隔符、同样的大小写、同样的标签。
4. **推理类任务上用思维链**。让模型"一步一步想"，或在少样本示例里直接写出推理过程。这在数学、逻辑类基准上常常带来巨幅提升。

---

## 8. 评估生成文本

没有完美的指标——但知道在什么场景该用哪个近似指标、它会漏掉什么，是必备能力。

### BLEU（翻译）

衡量生成相对参考的 n-gram 精确率，外加一个长度惩罚：

$$
\text{BLEU} = \text{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu(generated: str, reference: str) -> float:
    return sentence_bleu([reference.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method1)
```

在**翻译**上和人工评分相关性强；在开放式生成上则相关性较弱。

### ROUGE（摘要）

衡量 n-gram **召回率**——参考里有多少出现在了生成里：

```python
from rouge_score import rouge_scorer
def rouge(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in s.items()}
```

### 困惑度（内在指标）

模型对留出文本"有多惊讶"，越低越好：

$$
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i\mid w_{<i})\right)
$$

```python
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**enc, labels=enc.input_ids).loss
    return torch.exp(loss).item()
```

困惑度是训练期间有用的**内在**信号，但**低困惑度 ≠ 好的聊天机器人**——模型可能只是擅长预测常见模式，照样会胡说八道。

### 哪个任务用哪个指标？

| 任务 | 推荐指标 |
|------|----------|
| 机器翻译 | BLEU、chrF、COMET（学习式） |
| 文本摘要 | ROUGE、BERTScore |
| 开放式对话 | 人工评估 + 多样性（distinct-n） |
| 指令遵循 | LLM-as-judge + 人工评估 |
| 语言建模 | 困惑度 |

任何面向真实用户的场景，**人工评估仍是金标准**。

---

## 9. 用 GPT-2 搭一个聊天机器人

GPT-2 小到能在 CPU 上跑，又大到足够有趣。HuggingFace 的 `transformers` 库把几乎所有样板代码都藏起来了。

### 单轮版

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

### 多轮版（带滚动历史）

```python
class MultiTurnBot(ChatBot):
    def __init__(self, name: str = "gpt2") -> None:
        super().__init__(name)
        self.history: list[str] = []

    def respond(self, user: str, max_turns: int = 5, **gen_kwargs) -> str:
        # 只保留最近 N 轮，以免超过 1024 token 的上下文窗口
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

### 一行版（pipeline API）

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("User: What is deep learning?\nAssistant:",
          max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)[0]["generated_text"])
```

> GPT-2 的回答**不会**像 ChatGPT——它既没有指令微调，也没有 RLHF。要拿到接近 ChatGPT 的体验，要么换一个开源指令模型（如 `meta-llama/Llama-3-8B-Instruct`、`Qwen/Qwen2.5-7B-Instruct`），要么自己在指令数据集上把 GPT-2 微调一遍。

---

## 10. GPT vs. BERT，一表对照

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | 编码器，双向自注意力 | 解码器，**因果**自注意力 |
| 预训练 | Masked LM + Next Sentence Prediction | 因果语言建模 |
| 上下文 | 双向（看左右） | 单向（只看左侧） |
| 拿手 | 分类、NER、问答、检索 | 生成、对话、代码、创作 |
| 适配方式 | 加任务头微调 | Prompt（zero/few-shot）或微调 |
| 规模故事 | 约 1 B 参数后收益递减 | 收益持续，并伴随能力涌现 |

一个有用的心智模型：**BERT 像搜索引擎、GPT 像写作者**。需要把输入压成向量时用 BERT 系；需要产出文本时用 GPT 系。

---

## 11. 必须诚实承认的局限

- **幻觉**。对自己不知道的东西，生成流畅但事实错误的内容是模型的默认行为。缓解手段：检索增强（第 10 篇）、工具调用、校准过的拒答训练。
- **上下文长度**。GPT-2 1024、GPT-3 2048、GPT-4 32K 又扩到 128K，今天 1M 也开始普及。但长上下文成本高，而且**中段内容**的召回会下降（"lost in the middle"）。
- **算力成本**。1750 亿参数模型一次前向传播就要几百 GB GPU 显存，推理总成本在模型生命周期里远超训练成本。
- **训练数据偏见**。模型会照搬训练数据里的统计模式，包括刻板印象。
- **可控性有限**。要硬性约束风格、格式或事实性，必须叠加额外机制（system prompt、约束解码、RLHF、后置过滤）。

---

## 12. 核心要点

- GPT 模型本质就是一个**带因果掩码的仅解码器 Transformer**，被训练来"预测下一个 token"。这种朴素本身就是 point。
- **规模是放大器**：参数、数据、算力联手让损失幂律下降，并在此之上叠加突如其来的能力跃迁。
- **上下文学习**让一个固定模型靠 prompt 就能处理上千种任务——大多数场景都不需要微调。
- **解码策略**是免费的杠杆：对话和创作首选 **top-$p$ + 温度 0.7–0.9**；翻译和可复现性场景选贪心或束搜索。
- **BERT 理解、GPT 生成**——两者合起来覆盖了现代 NLP 的全部光谱。第 7–12 篇会继续深入 GPT 类模型的实战玩法（提示工程、微调、RAG、多模态…）。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 1 | NLP 入门与文本预处理 | [阅读](/zh/自然语言处理-一-NLP入门与文本预处理/) |
| 2 | 词向量与语言模型 | [阅读](/zh/自然语言处理-二-词向量与语言模型/) |
| 3 | RNN 与序列建模 | [阅读](/zh/自然语言处理-三-RNN与序列建模/) |
| 4 | 注意力机制与 Transformer | [阅读](/zh/自然语言处理-四-注意力机制与Transformer/) |
| 5 | BERT 与预训练模型 | [上一篇](/zh/自然语言处理-五-BERT与预训练模型/) |
| **6** | **GPT 与生成式语言模型（本文）** | |
| 7 | 提示工程与 In-Context Learning | [下一篇](/zh/自然语言处理-七-提示工程与In-Context-Learning/) |
| 8 | 模型微调与 PEFT | [阅读](/zh/自然语言处理-八-模型微调与PEFT/) |
