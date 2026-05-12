---
title: "大模型工程（六）：长上下文与 RoPE、YaRN"
date: 2026-04-01 09:00:00
tags:
  - LLM
  - long-context
  - RoPE
  - yarn
  - alibi
  - attention-sinks
categories: 大模型工程
series: llm-engineering
series_order: 6
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "RoPE 怎么编码位置、为什么朴素扩展会崩、NTK-aware 和 YaRN 缩放、ALiBi vs RoPE、流式生成的 attention sinks，以及 1M 上下文承诺为什么常在检索测试上崩盘。"
translationKey: "llm-engineering-6"
---
""1M 上下文"算是大模型圈最注水的指标之一：模型能处理 1M token 属于架构能力，但能否真正利用第 80 万位的信息回答问题，则是对行为能力的考验——这难度要大得多。本章将介绍位置编码的数学原理、将上下文扩展到训练长度之外的工程技巧，以及为什么大多数长上下文能力在“大海捞针”测试中表现不佳。

大模型长上下文的发展史大概分三幕。第一幕（2017-2021）：模型训练长度卡在 512-2048 token，因为 attention 是 $O(n^2)$ 的，显存吃不消。第二幕（2022-2023）：高效的 attention 算子（如 FlashAttention [Dao et al., 2022]）使长序列训练成为可能；后处理式的上下文扩展技术（如 Position Interpolation、NTK-aware scaling 和 YaRN）则允许将预训练 checkpoint 的上下文长度从 4K 扩展到 32K 甚至更高。第三幕（2024-2026）：原生长上下文训练（如 Llama 3.1 的 128K、Gemini 的 1-2M 和 Claude 的 200K）成为标配，但能 attend 的上下文和有用的上下文之间仍存在差距——这正是本章的重点。

![LLM Engineering (6): Long Context — RoPE, YaRN, Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_1.png)

## 位置信息不是免费的

Self-attention 对排列是不变的，没有位置信号，模型无法区分“猫坐在垫子上”和“垫子坐在猫上”。注入位置信息主要有三种方案：

1. **Sinusoidal absolute**（原始 Transformer）：在第 1 层之前把位置的 $\sin/\cos$ 函数加到 token embedding 上。
2. **Learned absolute**：为每个绝对索引学习一个位置 embedding，直到最大长度。GPT-2, BERT 在用。
3. **Rotary (RoPE)**：在每个 attention 层*内部*，按与位置成比例的角度旋转 Q 和 K 向量。LLaMA, Qwen, Mistral, DeepSeek 都在用。

RoPE 已成为主流——截至 2026 年，所有主流大模型都采用它作为位置编码，既在每层内部注入位置信号以增强鲁棒性，又让相对位置信息自然浮现于 Q·Kᵀ 点积中，而这正是 attention 机制的本质需求。

第四个方案 **ALiBi** (attention with linear bias) 在 2022 年左右曾有激烈的竞争，但最终失败了；本章后面会将其作为最有趣的替代路径进行讲解。第五个 **xPos** (Sun et al., 2022) 是 RoPE 的改进版，加了个依赖长度的 decay 让外推更稳；DeepSeek 和少数现代模型内部在用，但核心思想还是 RoPE 加工程打磨。

## RoPE：数学原理

![fig1: RoPE rotation visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig1_rope_rotation.png)


对于位置 $m$ 和 $n$ 的 query 向量 $q$ 和 key 向量 $k$，RoPE 用旋转矩阵 $R(m\theta)$ 和 $R(n\theta)$ 分别乘它们，其中 $\theta$ 取决于维度 ([Su et al., 2021][su-rope])：

$$\theta_i = b^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1$$

其中 $b$ 是 **rope base**（默认 10,000）。每对维度 $(2i, 2i+1)$ 以频率 $\theta_i$ 旋转。低索引对旋转快（携带细粒度位置），高索引对旋转慢（携带粗粒度位置）。

关键性质：旋转后，点积 $q \cdot k$ 只取决于*相对*位置 $m - n$：

$$\langle R(m\theta) q, R(n\theta) k \rangle = \langle q, R((n-m)\theta) k \rangle.$$

这就是 RoPE 能外推的原因：模型看到的不是绝对位置 50K，而是相对于当前 token 的 $-3$ 偏移，这在训练期间它见过几十亿次了。

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# precompute cos/sin tables
def rope_cos_sin(seq_len, dim, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
```

几乎每个现代 attention 算子都内联处理 RoPE；概念上就是这么回事。

### 波长与每个维度携带的信息

每个频率 $\theta_i$ 对应一个波长 $\lambda_i = 2\pi / \theta_i$ —— 也就是那对维度完成一次完整旋转所需的 token 位置数。设 base $b=10000$ 且 head_dim $d=128$：

- $i=0$: $\lambda_0 = 2\pi \approx 6.3$ tokens。这对维度编码非常局部的位置。
- $i=32$: $\lambda_{32} = 2\pi \cdot 10000^{32/64} \approx 628$ tokens。中等范围。
- $i=63$: $\lambda_{63} = 2\pi \cdot 10000 \approx 62832$ tokens。最长波长是 62K token；如果你的训练上下文是 4K，这个维度在训练期间连一次完整旋转都没走完。

这就是扩展问题的根源。波长*远长于*训练上下文的维度，只见过极小部分的旋转范围。当我们让模型在远超训练长度的位置 attend 时，这些维度就进入了模型从未见过的旋转区域。高频（短波长）维度编码局部位置，它们没问题——它们在训练上下文中循环了很多次，模型熟悉它们的全范围。

这一洞见催生了现代上下文扩展方法的核心思想：避免对所有维度进行等比例缩放，而应依据各维度对应波长的长短进行差异化缩放。

## 为什么 naive 扩展会崩

![fig2: YaRN frequency rescaling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig2_yarn_freq_adjustment.png)


![fig4: position interpolation strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig4_position_interpolation.png)


用 `rope_base=10000` 和 `max_position=4096` 训练模型。试着在位置 32768 用它。会发生什么？

最低频维度每 token 旋转 $10000^{-1} \approx 10^{-4}$ 弧度。到位置 32768 时它们旋转了 $\sim 3.3$ 弧度——超过了 $\pi$，意味着它们绕过了训练期间见过的任何状态。点积几何现在处于模型从未学过如何解释的区域。质量崩塌。

两种有效的扩展策略：

**Position interpolation (PI)** ([Chen et al., 2023][chen-pi])：按比例 $s = L_{\text{new}} / L_{\text{train}}$ 缩小位置。位置 32768 在 $s=8$ 时变成有效位置 4096，模型见过这个。继续微调几百步能修复小的分布偏移。PI 方法虽有效，但会轻微损害短上下文性能——因为每个位置现在对应一个模型在训练中从未见过的分数旋转角度。原始 PI 论文报告，仅需 1000 步微调即可将 LLaMA 的上下文长度从 2K 扩展至 32K，且性能下降极小；这标志着后处理式上下文扩展正式进入实用工程阶段。

**NTK-aware scaling** (2023 年中通过 [r/LocalLLaMA 社区引入][ntk-aware])：不是均匀缩放位置，而是缩放 rope base，让低频维度保持在训练范围内，而高频维度（携带局部信息）受干扰最小。新 base 是 $b' = b \cdot s^{d/(d-2)}$。直觉符合上面的波长分析：训练期间已经循环多次的维度不需要缩放； barely 旋转的维度需要。

**YaRN** ([Peng et al., 2024][peng-yarn])：目前最佳。结合 NTK-aware 缩放与基于维度波长是否短于训练上下文的每维度插值强度。加上温度校正（按 $\sqrt{\log s}$ 重缩放 attention logits 以补偿 attending 到固定 query 的 token 数量增加）。YaRN 用极少的继续训练（几百步）将 LLaMA-2-7B 从 4K 扩展到 128K，且几乎无短上下文质量下降（困惑度与基座模型相差 0.1 以内）。

```python
# YaRN-style rope_base for context extension
import math
def yarn_base(orig_base, orig_ctx, target_ctx, alpha=1, beta=32):
    ratio = target_ctx / orig_ctx
    return orig_base * (ratio ** (alpha / (alpha - beta)))
# example: 4K → 32K
print(yarn_base(10000, 4096, 32768))  # ~1.6e8
```

在实践中，当前主流开源模型（如 Qwen3、LLaMA-3、Mistral）通常已在发布时完成上下文扩展。通常不需要重新扩展；你读技术报告知道他们用了哪种方法以及实际限制是什么就行。

## LongRoPE 与基于搜索的缩放

YaRN 不错，但假设一个 band 内所有维度用统一的缩放公式。**LongRoPE** ([Ding et al., 2024][ding-longrope]) 更进一步：把每维度缩放因子当作搜索问题。对几千个候选缩放向量（每个长度 $d/2$，每个 RoPE 频率对一个因子）进行进化搜索，在小校准集上评估每个（长上下文文本的困惑度），保留赢家。LongRoPE 将 LLaMA-2-7B 扩展到 2M token，RULER 分数与原生长上下文模型相当，且无需大量继续预训练。

教训是合适的缩放形状不是固定的——它取决于模型、数据分布和目标长度。每维度搜索后微调是目前已知表达力最强的扩展策略。

实际影响：当你在模型卡片上看到“上下文扩展到 1M"时，问问用了哪种方法。PI, NTK-aware 和 YaRN 本质上是确定性的。LongRoPE 涉及搜索过程；发布的 checkpoint 编码了发现的缩放因子。部署成本一样（只是初始化 cos/sin 表不同），但在高扩展比例下的质量有明显差异。

## ALiBi：更简单的替代方案

ALiBi ([Press et al., 2022][press-alibi]) 完全跳过位置旋转，给 attention 分数加一个线性 bias：

$$\text{attn}_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d}} - m_h \cdot |i - j|\right)$$

其中 $m_h$ 是每头斜率，随头索引几何变化（通常 $H$ 个头对应 $m_h = 2^{-8h/H}$）。越近的 token 获得越高 attention；bias 在训练时固定，无需旋转。

优点：外推到比训练见过的更长上下文无需微调。ALiBi 论文展示了 train-at-1024 / test-at-2048 无质量下降。缺点：大多数实验发现 ALiBi 在需要精确长程检索的任务上表现不如 RoPE——线性 decay bias 意味着无论相关性如何，非常远的 token 获得的 attention 呈指数级减少。

ALiBi 用在 BLOOM, MPT 和一些研究模型中。RoPE 赢了生产环境之战。例外是混合 Mamba-attention 模型中 ALiBi 可能更便宜，以及一些"attention sink"实现中为了给流式稳定性给 RoPE 加 ALiBi 风格的 decay 项。

一个细微的观察：ALiBi 有效是因为语言对大多数 token 确实存在大致对数的距离 - 相关性关系，log-attention 空间中的线性 bias 近似了这一点。它在检索上输给 RoPE 的原因是检索任务恰恰需要*反*单调 attention——相关 token 可能是上下文中*最远*的那个，而不是最近的。RoPE 能学习任意 attention 模式；ALiBi 有个内置的距离先验。
## Attention sinks：流式处理的 hack

![fig3: sliding-window attention with sinks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig3_sliding_window.png)


Xiao 等人（[2024, "StreamingLLM"][xiao-sinks]）发现个怪现象：你要是用 **windowed attention**（每个 token 只关注前 $w$ 个 token），一旦解码超过窗口限制，质量在第 $w+1$ 位直接崩盘。不是慢慢衰减，是断崖式下跌。

原因很简单：attention softmax 非得把概率质量分配出去，哪怕没 key 匹配 query。训练好的模型会把这部分质量路由到 **序列开头的几个 token** 上——它们成了 "attention sinks"。把这些截掉，softmax 分布就乱套了。

解决办法简单得有点可笑：永远在每个 attention 窗口里保留最开头的 4-8 个 token。配合滑动窗口，模型就能解码任意长度的序列而不崩。

```python
def streaming_attention(q, k_cache, v_cache, sink_size=4, window=4096):
    # Keep first sink_size tokens + last window tokens
    sink_k, sink_v = k_cache[:, :, :sink_size, :], v_cache[:, :, :sink_size, :]
    win_k,  win_v  = k_cache[:, :, -window:, :], v_cache[:, :, -window:, :]
    k = torch.cat([sink_k, win_k], dim=2)
    v = torch.cat([sink_v, win_v], dim=2)
    return F.scaled_dot_product_attention(q, k, v)
```

从 Mistral-7B-v0.2 开始，还有 Qwen3 以及大多数生产级的长上下文模型，内部都用 sinks + sliding windows。这就是 "1M 上下文能用" 和 "1M 上下文却 hallucinates" 的区别。

### 为什么 softmax 非得要个 sink

数学原因是结构性的。Softmax 输出的是 key 上的概率分布，构造上概率和就得为 1。要是没 key 跟 query 真正相关（比如模型在处理 "filler" token），softmax 照样得产出分布。去哪呢？训练好的模型学会把概率质量 dump 到几个特定位置——通常是开头那几个靠近 BOS 的 token，毕竟每个训练样本里都有它们。这些 token 充当了 "空操作" 目标。

这对流式处理影响很大。要是滑动窗口滑过了这些靠近 BOS 的 token，你就把学好的 空操作 目标给删了。Softmax 被迫把质量放到看起来真正相关的 key 上，attention 模式扭曲，质量直接崩盘。永远保留 sinks 成本几乎为零（4-8 个 KV 条目），但保住了学到的动态特性。

另一篇相关但不同的论文，**Massive Activations**（Sun et al., 2024），指出 sink 行为是更广泛模式的一部分：训练好的 transformer 中少量特征激活承载了 disproportionate 的重要性。剪枝这些激活会毁掉模型。Sinks 就是这种现象在 attention 侧的表现。

## 生产环境中的 Sliding window attention

Mistral 7B（2023 年发布）是第一个标配 **sliding window attention**（SWA）的流行模型：每个 token 只关注前 $w=4096$ 个 token。内存和计算量随 $w$ 线性下降，而不是随 $n$ 二次方下降，这让 32K 上下文变得很便宜。

感受野依然在增长：在第 $\ell$ 层，token 通过 $\ell$ 跳大小为 $w$ 的路径关注，所以有效感受野是 $\ell \cdot w$。对于 32 层、$w=4096$ 的 Mistral 7B，第 32 层的感受野是 131K token，远超名义上的上下文窗口。远距离 token 的信息确实能传播，但是通过中间层间接传播的。

实践中，SWA 配合 attention sinks 构成了大多数生产级长上下文行为。纯 dense attention 超过 32K 很少见；成本 - 质量曲线倾向于 SWA + sinks + 扩展技术（YaRN/LongRoPE）来处理长尾。

## 大海捞针：唯一诚实的基准测试

![LLM Engineering (6): Long Context — RoPE, YaRN, Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_2.png)


![fig5: RULER scores by context length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig5_ruler_scores.png)


架构让你能 attend 到 N 个 token。但你是否真的利用了位置 N 的信息是另一回事。标准测试（Greg Kamradt 2023 年首创）：在长上下文的位置 $p$ 插入一个事实（"The magic number is 7392"），然后问 "What is the magic number?"

"Needle in a haystack" 给了个可分享的可视化（位置 vs 上下文长度的红绿热力图），但这测试太弱了。单事实检索对现代模型来说太简单。**RULER**（[Hsieh et al., 2024][hsieh-ruler]）是现代替代品：13 个任务类别，控制上下文长度，包括多 needle 检索、跨隐藏事实的多跳推理、变量追踪和聚合任务。RULER 分数告诉你模型在声称的长度上是否 *真的* 有用，而不只是能否复述字符串。

来自 RULER 的真实数据（2024 年发布 + 我在后续模型上的复现）：

| Model | Claimed ctx | RULER 16K | RULER 64K | RULER 256K |
|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8K | 90.3 | 31.4 | 0.0 |
| LLaMA-3.1-8B (YaRN-128K) | 128K | 89.7 | 78.2 | 38.5 |
| Qwen3-32B | 128K | 95.1 | 91.2 | 81.8 |
| GPT-4o | 128K | 95.5 | 91.8 | 84.2 |
| Claude-4.5-Sonnet | 200K | 96.7 | 93.5 | 88.4 |
| Gemini-3-Pro | 1M | 97.2 | 95.0 | 92.1 |

规律很明显：声称的上下文通常是工作上下文的 2-4 倍。预训练就支持长上下文的架构（Gemini, Claude）比事后扩展的更稳。生产环境中，除非厂商发布 RULER 结果，否则假设工作上下文是声称值的一半。

更痛苦的测试：**multi-needle**。隐藏 $k$ 个事实，让模型检索全部 $k$ 个。大多数模型在 $k = 5$ 时就直接崩了，哪怕上下文只有 32K。**基于 needle 的推理**（把不同位置的 3 个事实串起来）更难——前沿模型在 128K 维持 ~80 %，但在 256K 掉到 <50 %。

## 中间迷失（Lost in the Middle）

另一个记录详尽的长上下文失败模式：**Lost in the Middle**（[Liu et al., 2023][liu-lostmiddle]）。即使模型 *能* attend 到所有位置，它也 disproportionately 加权上下文开头和结尾的 token。长上下文中间 60% 被系统性低估。

Liu 等人的关键实验：在多文档 QA prompt 中，把单个相关文档放在 19 个干扰项之间的不同位置。位置 1（开头）和位置 20（结尾）的准确率是 75-80 %；位置 10（中间）掉到 50-55 %。每个测试模型（GPT-3.5, Claude-1, MPT 等）都出现了 U 型曲线，2026 年的模型依然存在，虽然没那么严重。

原因：attention 被训练时的分布偏向开头（大多数文档的重要 framing 在顶部），又被自回归解码的 recency bias 偏向结尾。中间被挤占了。

实用修复方案：

- **把问题放在长上下文的结尾**，而不是开头。Lost-in-the-middle 的 attention 偏向结尾，所以模型读完文档后再 "思考" 问题。Empirically：准确率提高 5-15 %。
- **把最重要的上下文放在文档列表的开头或结尾**，而不是中间，如果你能控制顺序的话。
- **重排序很重要。** 即使检索返回了正确文档，把它放在 10 个中的第 5 位也不如放在第 1 位。RAG pipeline（第 8 章）应该对检索到的 chunk 排序，最相关的放前面。

## 原生长上下文 vs 事后扩展

到了 2026 年，长上下文 landscape 已经分化。有些模型把长上下文作为头等大事进行预训练；有些则是事后扩展。

原生长上下文模型（Llama 3.1 的 128K, Gemini 2.5/3 的 1-2M, Claude 4.5 的 200K）在预训练中穿插长文档训练数据。模型原生学会位置依赖行为；RULER 分数在整个上下文范围内都很高。

事后扩展模型（社区 LLaMA-2 → 32K 用 YaRN, Qwen2.5 → 128K 用扩展）从短上下文 base 开始，应用 YaRN/LongRoPE plus 少量长上下文数据继续预训练。RULER 分数在适度扩展（2-4 倍）时不错，激进扩展（16 倍+）则退化。

成本差异是真实的。从头预训练 128K 每个 token 成本大约是 4K 预训练的 2-4 倍（长序列下 attention 主导计算，即使有 FlashAttention；KV cache 内存限制 batch size）。事后扩展基本免费（几千步 fine-tuning）。如果你是花 1 亿美元做训练的基础实验室，原生长上下文的 2-4 倍溢价是合理的。如果你是发开源权重的小实验室，事后扩展是唯一可行路径。

## RAG vs 长上下文：真正的权衡

最常见的应用问题："我应该把文档塞进长上下文，还是做 RAG？" 2026 年的答案比两年前更 nuanced。

满足以下条件就 Stuff into context：
- 文档适合你的模型 <128K tokens。
- 检索质量难工程化（流动 query，没法好 chunking）。
- 延迟预算容忍 5-30s prefill。
- 成本预算容忍每 query ~$0.30（200K-token prompt 按 $1.50/Mtok 算）。

满足以下条件就用 RAG（第 8 章）：
- 文档语料库远大于上下文窗口。
- 延迟必须 <1s。
- 每 query 成本必须是几分钱。
- 你需要来源归属。

长上下文适合需要整体阅读的任务——整个 repo 的代码 review，长会议总结，正确 chunk 不可预测的多文档 QA。其他情况 RAG 在成本上领先 1-2 个数量级。

混合方法（RAG 找候选，长上下文合成）是大多数非 trivial 工作负载的生产 sweet spot。Anthropic 的 "Contextual Retrieval"（2024）和 Microsoft 的 GraphRAG（2024）都是这种混合主题的变体；第 8 章会详细讲。

## 生产建议

- **部署前务必用 needle 测试检查实际工作负载下的工作上下文。** 厂商引用的数字是最佳情况。
- **服务重复输入时使用 Prompt-cache。** 100K-token system prompt prefill 成本 $1.50，缓存后只要 $0.05（vLLM enable_prefix_caching, OpenAI/Anthropic prompt caching APIs）。
- **把问题放在长上下文的结尾**，而不是开头。Lost-in-the-middle（Liu et al., 2023）显示 attention disproportionately 加权长上下文的开头和结尾。问题放结尾准确率高 5-15 %。
- **别信任 50K+ 位置做算术。** 哪怕 Claude-4.5，输入数据靠后时多步数学错误也更多。如果能把计算上下文移近问题就移近。
- **如果你自己 roll 长上下文模型，用 sliding window + sinks。** 成本 - 质量权衡无敌。
- **关注 prefill 延迟曲线。** 单 GPU 上 TTFT 大致随 prompt 长度线性扩展，但在 TP 上更陡（all-reduces 主导），跨节点 setup 也是。TP=2 H100 上 200K-token prompt prefill 要 6-8 秒；用户会注意到。
- **在你的领域上做基准测试。** RULER 上 90 分的模型，在你的医疗记录任务上可能只有 70 分，因为医疗文本分布跟 RULER 的合成 needle 不同。领域特定 eval 是唯一诚实的信号。

## 总结与下一章

RoPE 让长上下文变得可行；YaRN 把它扩展到了训练长度之外；sinks 让流式处理稳定；但工作上下文总是小于声称的上下文。在你的工作负载上测试。对于大多数生产任务，RAG 赢在成本。长上下文赢在整体阅读和短运行的交互式工作流。

下一章：**function calling 和 tool use**。JSON schema vs free-form，并行 tool calls，错误恢复，以及真正有效的 agent-loop 模式。
## 参考资料

- [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.][su-rope]
- [Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023.][chen-pi]
- [Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," ICLR 2024.][peng-yarn]
- [Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.][ding-longrope]
- [Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)," ICLR 2022.][press-alibi]
- [Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.][xiao-sinks]
- [Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," 2023.][liu-lostmiddle]
- [Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?" 2024.][hsieh-ruler]
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.][dao-flashattention]
- [NTK-aware scaling: r/LocalLLaMA bowtied_handbasket post (2023)][ntk-aware] —— 这项技术的社区起源。
- Sun et al., "Massive Activations in Large Language Models," 2024.

[su-rope]: https://arxiv.org/abs/2104.09864
[chen-pi]: https://arxiv.org/abs/2306.15595
[peng-yarn]: https://arxiv.org/abs/2309.00071
[ding-longrope]: https://arxiv.org/abs/2402.13753
[press-alibi]: https://arxiv.org/abs/2108.12409
[xiao-sinks]: https://arxiv.org/abs/2309.17453
[liu-lostmiddle]: https://arxiv.org/abs/2307.03172
[hsieh-ruler]: https://arxiv.org/abs/2404.06654
[dao-flashattention]: https://arxiv.org/abs/2205.14135
[ntk-aware]: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/