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
“1M token 上下文”堪称大模型领域最被夸大的指标之一。模型能处理 1M tokens，这反映的是架构能力；但能否真正利用第 80 万位的信息来回答问题，则考验的是行为能力——后者要难得多。本章将深入探讨位置编码的数学原理、将上下文扩展至训练长度之外的工程技巧，并解释为何大多数长上下文模型在“大海捞针”测试中表现不佳。

大模型长上下文的发展大致可分为三幕。第一幕（2017–2021）：模型训练长度被限制在 512–2048 tokens，因为注意力机制的时间复杂度为 $O(n^2)$，硬件资源难以支撑更长序列。第二幕（2022–2023）：高效注意力算子（如 FlashAttention [Dao et al., 2022]）使长序列训练变得可行，而事后上下文扩展技术（如 Position Interpolation、NTK-aware scaling 和 YaRN）则让从业者能将预训练好的 checkpoint 从 4K 扩展至 32K 甚至更长。第三幕（2024–2026）：原生长上下文训练（如 Llama 3.1 的 128K、Gemini 的 1–2M、Claude 的 200K）成为行业标配，但“可处理”的上下文与“真正有用”的上下文之间仍存在显著差距——而这正是本章的核心议题。

![LLM 工程（6）：长上下文 — RoPE、YaRN、Sinks — 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_1.png)

## 位置信息不是免费的

自注意力机制对输入顺序是不变的。若无位置信号，模型无法区分“猫坐在垫子上”和“垫子坐在猫上”。目前主流的位置注入方案有三种：

1. **正弦绝对位置编码**（原始 Transformer）：在第一层前，将位置的 $\sin/\cos$ 函数加到 token embedding 上。
2. **可学习绝对位置编码**：为每个绝对位置索引学习一个嵌入向量，上限为最大长度。GPT-2 和 BERT 均采用此方法。
3. **旋转位置编码（RoPE）**：在每一层注意力内部，按与位置成比例的角度旋转查询（Q）和键（K）向量。LLaMA、Qwen、Mistral 和 DeepSeek 等现代模型均采用此方案。

RoPE 已成为事实标准——截至 2026 年，所有主流大模型都使用它作为位置编码。原因有二：其一，它在每一层都注入位置信号，提供更强的位置感知；其二，相对位置信息自然地从 Q·K 点积中涌现，而这正是注意力机制真正需要的。

第四种方案 **ALiBi**（带线性偏置的注意力）曾在 2022 年左右激烈竞争，但最终落败；我们将在本章后文将其作为最有趣的替代路径进行探讨。第五种方案 **xPos**（Sun et al., 2022）是对 RoPE 的改进，通过引入长度相关的衰减因子提升外推稳定性。DeepSeek 等少数现代模型采用了该方法，但其核心仍是 RoPE 加上一些工程优化。

## RoPE：数学原理

![图1：RoPE 旋转可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig1_rope_rotation.png)

对于位于位置 $m$ 和 $n$ 的查询向量 $q$ 与键向量 $k$，RoPE 分别将其乘以旋转矩阵 $R(m\theta)$ 和 $R(n\theta)$，其中 $\theta$ 依赖于维度（[Su et al., 2021][su-rope]）：

$$
\theta_i = b^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1
$$

其中 $b$ 为 **rope base**（默认值为 10,000）。每对维度 $(2i, 2i+1)$ 以频率 $\theta_i$ 旋转：低索引对旋转快，承载细粒度位置信息；高索引对旋转慢，承载粗粒度位置信息。

关键性质在于：旋转后的点积 $q \cdot k$ 仅依赖于 *相对位置* $m - n$：

$$
\langle R(m\theta) q, R(n\theta) k \rangle = \langle q, R((n-m)\theta) k \rangle.
$$

这正是 RoPE 能有效外推的原因：模型看到的并非绝对位置 50K，而是相对于当前 token 的 $-3$ 偏移——这种模式在训练中已出现数十亿次。

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

几乎所有现代注意力算子都会内联处理 RoPE，上述即为其核心思想。

### 波长与各维度承载的信息

每个频率 $\theta_i$ 对应一个波长 $\lambda_i = 2\pi / \theta_i$，即该维度对完成一次完整旋转所需的 token 数。设 rope base $b=10000$，head_dim $d=128$：

- $i=0$：$\lambda_0 = 2\pi \approx 6.3$ tokens，编码极局部的位置；
- $i=32$：$\lambda_{32} = 2\pi \cdot 10000^{32/64} \approx 628$ tokens，覆盖中等范围；
- $i=63$：$\lambda_{63} = 2\pi \cdot 10000 \approx 62832$ tokens，最长波长达 62K tokens。

若训练上下文仅为 4K，则最高频维度在整个训练过程中连一次完整旋转都未完成。

这正是上下文扩展问题的根源：那些波长远超训练长度的维度，仅见过其旋转范围的一小部分。当模型被要求处理远超训练长度的位置时，这些维度便进入了从未见过的旋转区域。相比之下，高频（短波长）维度因在训练中反复循环，模型对其全范围已非常熟悉。

这一洞察催生了现代扩展方法的核心理念：**不应等比例缩放所有维度，而应根据波长进行差异化处理**。

## 为什么朴素扩展会失效

![图2：YaRN 频率重缩放](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig2_yarn_freq_adjustment.png)

![图4：位置插值策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig4_position_interpolation.png)

假设用 `rope_base=10000` 和 `max_position=4096` 训练一个模型，然后尝试在位置 32768 使用它，会发生什么？

最低频维度每 token 旋转约 $10^{-4}$ 弧度，到位置 32768 时累计旋转约 3.3 弧度——已超过 $\pi$，意味着其旋转状态已超出训练期间所见的任何范围。此时点积的几何结构进入模型从未学习过的区域，性能急剧下降。

目前有两种行之有效的扩展策略：

**位置插值（Position Interpolation, PI）**（[Chen et al., 2023][chen-pi]）：将新位置按比例 $s = L_{\text{new}} / L_{\text{train}}$ 压缩。例如，位置 32768 在 $s=8$ 下等效于位置 4096，模型对此已有经验。再辅以几百步微调即可修正微小的分布偏移。PI 虽有效，但会轻微损害短上下文性能——因为每个位置现在对应一个训练中从未出现过的分数旋转角度。原始论文报告称，仅需 1000 步微调即可将 LLaMA 从 2K 扩展至 32K，且性能损失极小。这一突破使事后上下文扩展真正成为工程可行选项。

**NTK-aware 缩放**（2023 年中由 [r/LocalLLaMA 社区提出][ntk-aware]）：不统一缩放位置，而是调整 rope base，使低频维度保持在训练范围内，同时最小化对高频维度（承载局部信息）的干扰。新 base 设为 $b' = b \cdot s^{d/(d-2)}$。其直觉与前述波长分析一致：训练中已充分循环的维度无需缩放，而几乎未旋转的维度则需重点调整。

**YaRN**（[Peng et al., 2024][peng-yarn]）：当前最优方案。它结合了 NTK-aware 缩放，并根据各维度波长是否短于训练上下文动态调整插值强度。此外还引入温度校正——将注意力 logits 按 $\sqrt{\log s}$ 重缩放，以补偿因 token 数量增加而导致的注意力稀释。YaRN 仅需几百步继续训练，就能将 LLaMA-2-7B 从 4K 扩展至 128K，且短上下文性能几乎无损（困惑度与基座模型相差不超过 0.1）。

```python
# YaRN-style rope_base for context extension
import math
def yarn_base(orig_base, orig_ctx, target_ctx, alpha=1, beta=32):
    ratio = target_ctx / orig_ctx
    return orig_base * (ratio ** (alpha / (alpha - beta)))
# example: 4K → 32K
print(yarn_base(10000, 4096, 32768))  # ~1.6e8
```

实践中，你部署的开源模型（如 Qwen3、LLaMA-3、Mistral）通常已在发布时完成上下文扩展。你一般无需自行重新扩展，只需阅读技术报告，了解其采用的方法及实际能力边界即可。

## LongRoPE 与基于搜索的缩放

YaRN 虽好，但仍假设同一频段内所有维度采用统一缩放公式。**LongRoPE**（[Ding et al., 2024][ding-longrope]）更进一步：将各维度的缩放因子视为一个搜索问题。通过进化算法在数千个候选缩放向量（每个长度为 $d/2$，对应每对 RoPE 频率）中搜索，在小型校准集（长上下文文本的困惑度）上评估性能，保留最优解。LongRoPE 成功将 LLaMA-2-7B 扩展至 2M tokens，其 RULER 分数与原生长上下文模型相当，且几乎无需额外预训练。

这说明：**最优缩放形态并非固定，而是依赖于具体模型、数据分布和目标长度**。目前表达力最强的扩展策略，便是“逐维度搜索 + 微调”。

实际影响在于：当你看到模型卡宣称“上下文扩展至 1M”时，应追问其采用的方法。PI、NTK-aware 和 YaRN 本质上是确定性的；而 LongRoPE 涉及搜索过程，发布的 checkpoint 中已编码了所发现的缩放因子。尽管部署成本相同（仅需初始化不同的 cos/sin 表），但在高扩展比下的质量差异显著。

## ALiBi：更简单的替代方案

ALiBi（[Press et al., 2022][press-alibi]）完全跳过位置旋转，转而在注意力得分中加入线性偏置：

$$
\text{attn}_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d}} - m_h \cdot |i - j|\right)
$$

其中 $m_h$ 为每头斜率，通常按头索引几何衰减（如 $H$ 个头时 $m_h = 2^{-8h/H}$）。越近的 token 获得越高注意力；该偏置在训练时固定，无需旋转操作。

优点：无需微调即可外推至远超训练长度的上下文。ALiBi 论文展示了在 1024 长度训练、2048 长度测试时性能无损。缺点：多数实验表明，ALiBi 在需精确长程检索的任务上逊于 RoPE——因其线性衰减偏置导致极远 token 的注意力呈指数级下降，无论其相关性如何。

ALiBi 被用于 BLOOM、MPT 及部分研究模型，但在生产环境中 RoPE 已胜出。例外情况包括：混合 Mamba-注意力模型中 ALiBi 计算更省；以及某些“注意力汇”实现中，为提升流式稳定性而在 RoPE 上叠加 ALiBi 风格的衰减项。

一个微妙观察：ALiBi 之所以有效，是因为语言中大多数 token 的相关性确实随距离呈近似对数衰减，而 log-attention 空间中的线性偏置恰好能拟合这一规律。但它在检索任务上败给 RoPE，正是因为检索恰恰需要 *反单调* 的注意力——相关 token 往往是上下文中 *最远* 的那个，而非最近的。RoPE 能学习任意注意力模式，而 ALiBi 内置了距离先验。

## Attention Sinks：流式处理的巧妙 hack

![图3：带 sinks 的滑动窗口注意力机制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig3_sliding_window.png)

Xiao 等人（[2024, "StreamingLLM"][xiao-sinks]）发现一个奇特现象：若使用 **滑动窗口注意力**（每个 token 仅关注前 $w$ 个 token），一旦解码超出窗口，性能会在位置 $w+1$ 处断崖式崩溃——并非缓慢衰减，而是瞬间崩塌。

原因在于：注意力 softmax 必须将概率质量分配出去，即使没有 key 与 query 匹配。训练好的模型会将这部分“多余”质量路由至 **序列开头的几个 token**——它们成为“注意力汇”（attention sinks）。一旦截断这些 token，softmax 分布便失去锚点，变得混乱。

解决方案出奇简单：在每个注意力窗口中永久保留最开始的 4–8 个 token。配合滑动窗口机制，模型即可无限解码长序列而不崩溃。

```python
def streaming_attention(q, k_cache, v_cache, sink_size=4, window=4096):
    # Keep first sink_size tokens + last window tokens
    sink_k, sink_v = k_cache[:, :, :sink_size, :], v_cache[:, :, :sink_size, :]
    win_k,  win_v  = k_cache[:, :, -window:, :], v_cache[:, :, -window:, :]
    k = torch.cat([sink_k, win_k], dim=2)
    v = torch.cat([sink_v, win_v], dim=2)
    return F.scaled_dot_product_attention(q, k, v)
```

从 Mistral-7B-v0.2 起，Qwen3 及大多数生产级长上下文模型均在内部采用“sinks + 滑动窗口”组合。这正是“1M 上下文可用”与“1M 上下文却产生幻觉”的关键区别。

### 为何 softmax 必须有“汇”

数学根源在于 softmax 的结构性约束：其输出必须是概率分布，总和恒为 1。当 query 与所有 key 均无实质相关性（如处理填充 token）时，softmax 仍需生成分布。训练模型学会将这部分质量“倾倒”到特定位置——通常是紧邻 BOS 的开头几个 token，因为它们出现在每一个训练样本中，充当了“空操作”目标。

这对流式推理影响深远：若滑动窗口移过了这些 BOS 邻近 token，就等于移除了模型习得的“安全出口”。softmax 被迫将质量分配给看似相关的 key，扭曲注意力模式，导致性能骤降。永久保留 sinks 几乎无额外开销（仅需 4–8 个 KV 条目），却能维持模型学到的动态行为。

另一篇相关但不同的论文 **Massive Activations**（Sun et al., 2024）指出，sink 行为是更广泛现象的一部分：在训练好的 Transformer 中，极少数特征激活承载了不成比例的重要性，剪枝这些激活会摧毁模型。注意力汇正是这一现象在注意力机制中的体现。

## 生产环境中的滑动窗口注意力

Mistral 7B（2023 年发布）是首个标配 **滑动窗口注意力**（SWA）的流行模型：每个 token 仅关注前 $w=4096$ 个 token。内存与计算开销随 $w$ 线性增长，而非随序列长度 $n$ 平方增长，使得 32K 上下文变得经济可行。

尽管如此，感受野仍在逐层扩大：在第 $\ell$ 层，token 可通过 $\ell$ 跳、每跳 $w$ 的路径间接关注更远位置，有效感受野达 $\ell \cdot w$。以 32 层、$w=4096$ 的 Mistral 7B 为例，第 32 层的感受野可达 131K tokens，远超名义上下文窗口。远距离信息虽能传播，但需经中间层中转。

实践中，SWA 与 attention sinks 的组合构成了大多数生产级长上下文系统的基础。纯稠密注意力在 32K 以上极为罕见；成本-质量权衡更倾向于 SWA + sinks + 扩展技术（如 YaRN/LongRoPE）来处理长尾场景。

## 大海捞针：唯一诚实的基准测试

![LLM 工程（6）：长上下文 — RoPE, YaRN, Sinks — 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_2.png)

![图5：不同上下文长度的 RULER 分数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig5_ruler_scores.png)

架构允许模型处理 N 个 token，但能否真正利用位置 N 的信息则是另一回事。标准测试（由 Greg Kamradt 于 2023 年首创）：在长上下文的位置 $p$ 插入一句事实（如 “The magic number is 7392”），然后提问 “What is the magic number?”。

“大海捞针”测试虽提供了直观的热力图（位置 vs 上下文长度），但过于简单——单事实检索对现代模型而言轻而易举。**RULER**（[Hsieh et al., 2024][hsieh-ruler]）是更全面的替代方案：包含 13 类任务，涵盖多 needle 检索、跨隐藏事实的多跳推理、变量追踪与聚合等，严格控制上下文长度。RULER 分数能真实反映模型在宣称长度上的实用性，而非仅能否复述字符串。

以下是 RULER 的实测数据（2024 年发布 + 笔者在后续模型上的复现）：

| 模型 | 声称上下文 | RULER 16K | RULER 64K | RULER 256K |
|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8K | 90.3 | 31.4 | 0.0 |
| LLaMA-3.1-8B (YaRN-128K) | 128K | 89.7 | 78.2 | 38.5 |
| Qwen3-32B | 128K | 95.1 | 91.2 | 81.8 |
| GPT-4o | 128K | 95.5 | 91.8 | 84.2 |
| Claude-4.5-Sonnet | 200K | 96.7 | 93.5 | 88.4 |
| Gemini-3-Pro | 1M | 97.2 | 95.0 | 92.1 |

规律清晰可见：**宣称上下文通常是有效工作上下文的 2–4 倍**。原生支持长上下文的模型（如 Gemini、Claude）表现远优于事后扩展模型。在生产环境中，除非厂商公开 RULER 结果，否则应假设实际有效上下文仅为宣称值的一半。

更严苛的测试是 **多 needle 检索**：隐藏 $k$ 个事实并要求全部召回。多数模型在 $k=5$ 时即大幅下滑，即便上下文仅 32K。**基于 needle 的推理**（串联不同位置的多个事实）难度更高——前沿模型在 128K 下准确率约 80%，但在 256K 下跌至不足 50%。

## 中间迷失（Lost in the Middle）

另一类被充分记录的长上下文失效模式是 **中间迷失**（[Liu et al., 2023][liu-lostmiddle]）：即使模型能访问所有位置，其注意力仍显著偏向上下文的开头与结尾，中间 60% 的内容被系统性忽视。

Liu 等人的关键实验：在含 19 个干扰文档的多文档 QA 任务中，将唯一相关文档置于不同位置。结果：位置 1（开头）和位置 20（结尾）的准确率达 75–80%；而位置 10（中间）骤降至 50–55%。这种 U 型曲线在所有测试模型（GPT-3.5、Claude-1、MPT 等）中均存在，2026 年的模型虽有所改善，但仍未根除。

原因有二：训练数据分布使模型偏向开头（重要信息常位于文档顶部）；自回归解码的近期偏好又强化了对结尾的关注。中间区域因此被双重挤压。

实用修复建议：

- **将问题置于长上下文末尾，而非开头**。因注意力天然偏向结尾，模型可在读完所有文档后再“思考”问题，实测可提升 5–15% 准确率。
- **若可控制顺序，将最关键内容放在文档列表的首尾**，避开中间区域。
- **重排序至关重要**。即使检索返回正确文档，将其置于 10 个 chunk 的第 5 位也不如放在第 1 位。RAG 流水线（第八章）应对检索结果排序，确保最相关 chunk 优先。

## 原生长上下文 vs 事后扩展

截至 2026 年，长上下文生态已明显分化：一类模型将长上下文作为预训练核心目标；另一类则依赖事后扩展。

原生长上下文模型（如 Llama 3.1 的 128K、Gemini 2.5/3 的 1–2M、Claude 4.5 的 200K）在预训练中持续混入长文档数据，使其原生掌握位置依赖行为，RULER 分数在整个上下文范围内保持高位。

事后扩展模型（如社区版 LLaMA-2 → 32K via YaRN、Qwen2.5 → 128K via 扩展）从短上下文基座出发，应用 YaRN/LongRoPE 并辅以少量长上下文继续预训练。其 RULER 分数在适度扩展（2–4 倍）时表现良好，但在激进扩展（16 倍以上）时显著退化。

成本差异真实存在：从头预训练 128K 上下文，每 token 成本约为 4K 预训练的 2–4 倍（长序列下注意力计算主导开销，即便使用 FlashAttention；KV 缓存也限制 batch size）。而事后扩展几乎免费（仅需数千步微调）。对投入上亿美元训练的基础模型实验室而言，原生长上下文的溢价合理；但对发布开源模型的小团队，事后扩展是唯一可行路径。

## RAG vs 长上下文：真正的权衡

最常见的应用问题是：“应将文档塞入长上下文，还是采用 RAG？” 到 2026 年，答案已比两年前更为精细。

**适合直接塞入上下文的情况**：
- 文档总长 <128K tokens（适配你的模型）；
- 检索难以工程化（如查询语义流动、缺乏有效分块策略）；
- 延迟预算可容忍 5–30 秒 prefill；
- 成本可接受约 $0.30/查询（按 200K-token prompt、$1.50/Mtok 计算）。

**应采用 RAG 的情况**（见第八章）：
- 文档库远超上下文窗口；
- 延迟必须 <1 秒；
- 单次查询成本需控制在几分钱；
- 需要来源追溯。

长上下文在需整体理解的任务中占优——如整个代码仓库的审查、长会议纪要总结、或正确 chunk 难以预判的多文档问答。其余场景，RAG 在成本上领先 1–2 个数量级。

**混合方案**（RAG 检索候选，长上下文合成答案）已成为多数非平凡工作负载的生产最优解。Anthropic 的 “Contextual Retrieval”（2024）与微软的 GraphRAG（2024）均属此类变体，第八章将详述。

## 生产建议

- **部署前务必用 needle 测试验证实际工作负载下的有效上下文**。厂商宣称数字均为理想情况。
- **对重复输入启用 Prompt 缓存**。一个 100K-token 系统提示的 prefill 成本为 $1.50，缓存后可降至 $0.05（vLLM 的 enable_prefix_caching 或 OpenAI/Anthropic 的 prompt caching API）。
- **将问题置于长上下文末尾**。据 Liu et al. (2023)，“中间迷失”现象表明模型更关注首尾，问题放结尾可提升 5–15% 准确率。
- **避免在 50K+ 位置进行算术运算**。即便是 Claude-4.5，在输入数据靠后时多步计算错误率也显著上升。若可能，将计算相关内容移近问题位置。
- **若自研长上下文模型，请采用 sliding window + sinks 组合**。其成本-质量比无可匹敌。
- **监控 prefill 延迟曲线**。单 GPU 上 TTFT 大致随 prompt 长度线性增长，但在张量并行（TP）或跨节点设置下更陡峭（all-reduce 主导）。例如，TP=2 的 H100 上处理 200K-token prompt 需 6–8 秒 prefill，用户会明显感知。
- **在你的领域数据上做基准测试**。RULER 得分 90 的模型，在医疗记录任务上可能仅得 70 分——因医疗文本分布与 RULER 的合成 needle 不同。领域特定评估才是唯一可靠信号。

## 总结与下一章

RoPE 使长上下文变得可行；YaRN 将其扩展至训练长度之外；sinks 保障了流式推理的稳定性。但无论如何，**有效工作上下文始终小于宣称值**。务必在你的实际负载上测试。对多数生产任务，RAG 在成本上胜出；长上下文则在整体阅读与短交互式工作流中占优。

下一章：**函数调用与工具使用**。涵盖 JSON schema vs 自由格式、并行工具调用、错误恢复，以及真正有效的智能体循环模式。
## 参考文献

- [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.][su-rope]
- [Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023.][chen-pi]
- [Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," ICLR 2024.][peng-yarn]
- [Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.][ding-longrope]
- [Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)," ICLR 2022.][press-alibi]
- [Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.][xiao-sinks]
- [Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," 2023.][liu-lostmiddle]
- [Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?" 2024.][hsieh-ruler]
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.][dao-flashattention]
- [NTK-aware scaling: r/LocalLLaMA bowtied_handbasket post (2023)][ntk-aware] —— 该技术的社区起源。
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
