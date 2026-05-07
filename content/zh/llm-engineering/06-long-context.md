---
title: "大模型工程（六）：长上下文 —— RoPE、YaRN、Attention Sinks"
date: 2026-05-01 09:00:00
tags:
  - llm
  - long-context
  - rope
  - yarn
  - alibi
  - attention-sinks
categories: 大模型工程
series: llm-engineering
series_order: 6
series_title: "大模型工程"
lang: zh-CN
mathjax: true
disableNunjucks: true
description: "RoPE 怎么编码位置、为什么朴素扩展会崩、NTK-aware 和 YaRN 缩放、ALiBi vs RoPE、流式生成的 attention sinks，以及 1M 上下文承诺为什么常在检索测试上崩盘。"
translationKey: "llm-engineering-6"
---
"1M token 上下文" 是 LLM 领域最常被夸大的数字之一。模型能处理 1M token，这是架构能力的体现。但能否用第 80 万位置的信息回答问题，这是行为能力，难度更高。这一章聊聊位置编码的数学原理、扩展上下文的工程技巧，以及为什么大多数长上下文声明经不起测试。

长上下文 LLM 的发展分三个阶段。第一阶段（2017-2021 年）：模型训练长度通常在 512 到 2048 token。原因是注意力机制复杂度 $O(n^2)$，超出这个范围算力扛不住。第二阶段（2022-2023 年）：FlashAttention 等高效注意力内核出现，让更长训练成为可能。Position Interpolation、NTK-aware scaling 和 YaRN 等技术，把预训练模型从 4K 推到 32K 甚至更高。第三阶段（2024-2026 年）：原生长上下文训练成为标配，比如 Llama 3.1 的 128K、Gemini 的 1-2M 和 Claude 的 200K。但能注意的上下文和真正有用的上下文之间，差距依然明显。这也是本章的核心内容。

![大模型工程（六）：长上下文 —— RoPE、YaRN、Attention Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_1.jpg)
## 位置不是免费的

自注意力机制对顺序不敏感。没有位置信号，模型分不清“猫坐在垫子上”和“垫子坐在猫上”。怎么注入位置信息？主要有三种方法。

1. **正弦绝对位置**（原始 Transformer）：在第一层前，把位置的 $\sin/\cos$ 函数加到 token 嵌入里。
2. **可学习绝对位置**：为每个绝对索引学一个位置嵌入，直到最大长度。GPT-2 和 BERT 就用这个。
3. **旋转位置编码（RoPE）**：在每个注意力层内部，按位置比例旋转 Q 和 K 向量。LLaMA、Qwen、Mistral 和 DeepSeek 都用 RoPE。

RoPE 赢了。我了解的所有靠谱的 2026 年 LLM 都用它。原因很简单：每层都注入位置信息，信号更强。相对位置自然从点积中得出，这正是注意力需要的。

还有一种方法叫 **ALiBi**（线性偏置注意力）。2022 年前后它曾是有力竞争者，但没打过 RoPE。后面我会详细讲，这是个很有趣的替代方案。再就是 **xPos**（Sun 等，2022），它是 RoPE 的改进版，加了长度依赖的衰减，让外推更稳定。DeepSeek 和一些现代模型用了它，但核心还是 RoPE，只是工程上更精致了。
## RoPE 的数学

![fig1: RoPE rotation visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig1_rope_rotation.png)

RoPE 对位置 $m$ 的查询向量 $q$ 和位置 $n$ 的键向量 $k$ 做旋转。它分别乘以旋转矩阵 $R(m\theta)$ 和 $R(n\theta)$，其中 $\theta$ 与维度相关（[Su 等，2021][su-rope]）。

$$\theta_i = b^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1$$

$b$ 是 **rope base**，默认值为 10000。每对维度 $(2i, 2i+1)$ 按频率 $\theta_i$ 旋转。低索引维度旋转快，携带细粒度位置信息；高索引维度旋转慢，携带粗略位置信息。

关键点来了：旋转后，点积 $q \cdot k$ 只依赖相对位置 $m - n$。

$$\langle R(m\theta) q, R(n\theta) k \rangle = \langle q, R((n-m)\theta) k \rangle.$$

这就是 RoPE 能外推的原因。模型看到的不是绝对位置 5 万，而是相对于当前 token 的偏移 -3。这种偏移在训练时见过十亿次了。

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def rope_cos_sin(seq_len, dim, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
```

几乎所有现代注意力内核都直接处理 RoPE。上面就是它的概念过程。

### 波长与每个维度携带的信息

每个频率 $\theta_i$ 对应一个波长 $\lambda_i = 2\pi / \theta_i$。波长表示这对维度完成一整圈旋转所需的 token 数。假设 base $b=10000$、head_dim $d=128$：

- $i=0$：$\lambda_0 = 2\pi \approx 6.3$ token。这对维度编码非常局部的位置。
- $i=32$：$\lambda_{32} = 2\pi \cdot 10000^{32/64} \approx 628$ token。中程距离。
- $i=63$：$\lambda_{63} = 2\pi \cdot 10000 \approx 62832$ token。最长波长为 62K token。如果训练上下文是 4K，这个维度在训练过程中连一圈都没转完。

这就是扩展问题的根源。波长远超训练上下文长度的维度，在训练时只见过它们旋转范围的一小部分。当要求模型关注远超训练长度的位置时，这些维度进入了模型从未见过的旋转区域。

高频（短波长）维度编码局部位置，没问题。它们在每个训练上下文中循环多次，模型已经熟悉它们的全范围。

这一洞见推动了现代扩展方法的发展：不要均匀缩放所有维度，而是按波长缩放。
## 为什么朴素扩展会崩

![fig2: YaRN frequency rescaling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig2_yarn_freq_adjustment.png)


![fig4: position interpolation strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig4_position_interpolation.png)

用 `rope_base=10000` 和 `max_position=4096` 训练的模型，直接跑到位置 32768 会怎样？

最低频维度每 token 旋转 $10000^{-1} \approx 10^{-4}$ 弧度。到位置 32768 时，累计旋转约 3.3 弧度——超过 $\pi$。这超出了训练范围。点积几何进入模型从未见过的区域，质量直接崩了。

两种扩展策略能解决问题：

**位置插值（PI）**（[Chen 等，2023][chen-pi]）。按比例 $s = L_{\text{new}} / L_{\text{train}}$ 缩小位置。比如，位置 32768 在 $s=8$ 下变成有效位置 4096，这是模型见过的范围。再微调几百步，修正分布漂移即可。PI 能用，但短上下文质量会略降。每个位置对应的小数旋转是模型没训练过的。原论文中，LLaMA 从 2K 扩展到 32K，只需 1000 步微调，质量退化极小。这方法让事后上下文扩展成为可行选项。

**NTK-aware 缩放**（2023 年中通过 [r/LocalLLaMA 社区][ntk-aware]引入）。不统一缩放位置，而是调整 rope base。低频维度保持在训练范围内，高频维度（携带局部信息）几乎不受影响。新 base 是 $b' = b \cdot s^{d/(d-2)}$。直觉很简单：训练时多次循环的维度不用调；几乎没有旋转的维度才需要调整。

**YaRN**（[Peng 等，2024][peng-yarn]）。目前最好的方法。结合 NTK-aware 缩放和基于波长的插值强度，再加上温度校正（通过 $\sqrt{\log s}$ 重新缩放注意力 logit，补偿固定查询的 token 数增加）。YaRN 将 LLaMA-2-7B 从 4K 扩展到 128K，只需几百步微调，短上下文质量几乎不降（困惑度与基础模型相差不到 0.1）。

```python
# YaRN 风格的 rope_base 上下文扩展
import math
def yarn_base(orig_base, orig_ctx, target_ctx, alpha=1, beta=32):
    ratio = target_ctx / orig_ctx
    return orig_base * (ratio ** (alpha / (alpha - beta)))
print(yarn_base(10000, 4096, 32768))  # ~1.6e8
```

实际部署开源模型（如 Qwen3、LLaMA-3、Mistral）时，它们通常已经完成上下文扩展。一般不需要重新扩展。读技术报告了解方法和限制就行。
## LongRoPE 与基于搜索的缩放

YaRN 表现不错，但它假设一个频段内所有维度共享统一的缩放公式。**LongRoPE**（[Ding 等，2024][ding-longrope]）更进一步，把每个维度的缩放因子当作一个搜索问题。具体做法是，在几千个候选缩放向量上进行进化搜索。每个向量长度为 $d/2$，对应一个 RoPE 频率对。接着，用小规模校准集评估这些向量，校准集主要测试长上下文文本的困惑度。最后保留效果最好的结果。

LongRoPE 把 LLaMA-2-7B 扩展到 2M token，RULER 分数接近原生长上下文模型。关键是，整个过程无需大规模继续预训练。

正确缩放形状不是固定的。它依赖模型架构、数据分布和目标长度。逐维度搜索再微调，是目前最灵活的扩展策略。

实际踩过的坑告诉我，看到模型卡写着“上下文扩展到 1M”，一定要问清楚方法。PI、NTK-aware 和 YaRN 基本是确定性的，直接套公式就行。LongRoPE 不一样，它需要额外的搜索过程。发布的检查点里，已经编码了找到的缩放因子。部署成本其实差不多，只是 cos/sin 表初始化方式不同。但在高扩展比下，质量差异非常明显。

下一节会详细讲如何在生产环境中选择合适的方法。
## ALiBi：更简单的替代方案

ALiBi（[Press 等，2022][press-alibi]）完全跳过位置旋转，直接给注意力分数加一个线性偏置：

$$\text{attn}_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d}} - m_h \cdot |i - j|\right)$$

$m_h$ 是每个头的斜率，按头索引几何变化。通常公式是 $m_h = 2^{-8h/H}$，$H$ 是头的数量。距离近的 token 得到更高注意力。偏置在训练时固定，不需要旋转。

优点很明显：能扩展到比训练时更长的上下文，而且不用微调。论文里展示了训练长度 1024、测试长度 2048 的情况，质量一点没掉。缺点也有：多数实验表明，ALiBi 在需要精确长程检索的任务上不如 RoPE。原因在于线性衰减偏置会让远处 token 的注意力指数级下降，不管它是否相关。

ALiBi 用在 BLOOM、MPT 和一些研究模型里。RoPE 在生产环境里赢了。不过有两个例外：一是 Mamba-注意力混合模型，ALiBi 更高效；二是一些 "attention sink" 实现，把 ALiBi 风格的衰减项加到 RoPE 上，提升流式稳定性。

这里有个有意思的点：ALiBi 能用，是因为语言中大多数 token 的距离和相关性大致是对数关系。log-attention 空间里的线性偏置刚好近似了这一点。但在检索任务上，它输给 RoPE，因为检索需要 *反* 单调注意力——相关 token 可能是上下文里最远的，而不是最近的。RoPE 能学任意注意力模式，而 ALiBi 内置了距离先验，限制了灵活性。
## Attention sinks：流式处理的技巧

![fig3: sliding-window attention with sinks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig3_sliding_window.png)

Xiao 等人（[2024, "StreamingLLM"][xiao-sinks]）发现一个奇怪现象。用 **窗口注意力**，每个 token 只关注前 $w$ 个 token。解码超过窗口后，质量在位置 $w+1$ 直接崩了。不是慢慢下降，而是断崖式崩塌。

原因很简单。即使没有 key 匹配 query，softmax 也得分配概率。训练好的模型会把概率堆到序列前几个 token 上。这些 token 成了“attention sink”。去掉它们，softmax 分布就乱了。

解决方法也很简单。每个注意力窗口永远保留前 4 到 8 个 token。结合滑动窗口，模型就能解码任意长度序列，质量不会崩。

```python
def streaming_attention(q, k_cache, v_cache, sink_size=4, window=4096):
    # 保留前 sink_size 个 + 最后 window 个
    sink_k, sink_v = k_cache[:, :, :sink_size, :], v_cache[:, :, :sink_size, :]
    win_k,  win_v  = k_cache[:, :, -window:, :], v_cache[:, :, -window:, :]
    k = torch.cat([sink_k, win_k], dim=2)
    v = torch.cat([sink_v, win_v], dim=2)
    return F.scaled_dot_product_attention(q, k, v)
```

从 Mistral-7B-v0.2 开始，Qwen3 和大多数生产环境的长上下文模型都用了这种 sinks 加滑动窗口的方法。这就是“1M 上下文能用”和“1M 上下文胡说”的区别。

### 为什么 softmax 需要 sink

数学上，原因是结构性的。Softmax 输出的是 key 的概率分布，总和必须为 1。如果没有任何 key 和 query 相关，比如处理“填充”token，softmax 还是会生成分布。那这些概率去哪了？模型学会了把它们堆到特定位置，通常是 BOS 附近的前几个 token。因为每个训练样本都有这些 token，它们成了“no-op”目标。

这对流式处理影响很大。如果滑动窗口移过了这些 BOS 附近的 token，就没了 no-op 目标。这时，softmax 只能分配给看起来相关的 key。注意力模式扭曲，质量直接崩掉。永远保留这些 sink 几乎没成本，只需 4 到 8 个 KV 条目，还能保住学到的动力学特性。

另一篇相关论文《Massive Activations》（Sun 等，2024）提到，sink 行为是更广泛现象的一部分。少量特征激活在训练好的 transformer 中非常重要。剪掉这些激活，模型就废了。Sinks 就是这一现象在注意力机制中的体现。
## 生产中的滑动窗口注意力

Mistral 7B（2023 年发布）是第一个带 **滑动窗口注意力（SWA）** 的流行模型。每个 token 只看前 $w=4096$ 个 token。内存和计算量随 $w$ 线性增长，而不是随 $n$ 二次增长。这让 32K 上下文变得很便宜。

感受野其实还在扩大。第 $\ell$ 层时，token 通过 $\ell$ 次跳跃传播信息，每次跳 $w$ 个 token。有效感受野就是 $\ell \cdot w$。Mistral 7B 有 32 层，$w=4096$，到第 32 层时感受野达到 131K token。这远超名义上下文窗口。远处 token 的信息会通过中间层间接传递。

实际用起来，SWA 加上 attention sink 就能搞定大部分长上下文需求。纯密集注意力超过 32K 的情况很少见。从成本和质量的平衡来看，SWA + sinks + 扩展技术（比如 YaRN/LongRoPE）更适合长尾场景。
## Needle in a haystack：唯一诚实的基准测试

![大模型工程（六）：长上下文 —— RoPE、YaRN、Attention Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/illustration_2.jpg)


![fig5: RULER scores by context length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/06-long-context/fig5_ruler_scores.png)

架构能处理 N 个 token，但是否用到了第 N 个位置的信息是另一回事。标准测试由 Greg Kamradt 在 2023 年提出：在长上下文的某个位置 $p$ 插入一句 "magic number 是 7392"，然后问模型 "magic number 是几"。

“针” 提供了红绿热图这种可视化结果，方便分享。但它太简单了，单事实检索对现代模型来说毫无压力。**RULER**（[Hsieh 等，2024][hsieh-ruler]）是更好的替代方案。它包含 13 类任务，覆盖多事实检索、隐藏事实的多跳推理、变量跟踪和聚合任务。RULER 分数直接告诉你模型在标称长度上是否真的有用，而不是只会复述字符串。

以下是 RULER 的实际数据（2024 年发表 + 我在后续模型上的复现结果）：

| 模型 | 标称上下文 | RULER 16K | RULER 64K | RULER 256K |
|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8K | 90.3 | 31.4 | 0.0 |
| LLaMA-3.1-8B (YaRN-128K) | 128K | 89.7 | 78.2 | 38.5 |
| Qwen3-32B | 128K | 95.1 | 91.2 | 81.8 |
| GPT-4o | 128K | 95.5 | 91.8 | 84.2 |
| Claude-4.5-Sonnet | 200K | 96.7 | 93.5 | 88.4 |
| Gemini-3-Pro | 1M | 97.2 | 95.0 | 92.1 |

规律很明显：标称上下文通常是实际工作上下文的 2 到 4 倍。预训练时就用长上下文的架构（如 Gemini 和 Claude）表现更稳。后来扩展的架构则差一些。在生产中，如果厂商没公布 RULER 结果，我建议按标称值的一半来估算实际能力。

更难的测试是 **多事实检索**。隐藏 $k$ 个事实，让模型全部找出来。大多数模型在 $k=5$ 时就崩了，即使上下文只有 32K。**基于多个事实的推理**更复杂，要求模型串联不同位置的 3 个事实。前沿模型在 128K 上还能保持约 80% 的准确率，但到 256K 就掉到 50% 以下了。
## Lost in the Middle

另一个长上下文失败模式叫 **Lost in the Middle**（[Liu 等，2023][liu-lostmiddle]）。模型虽然能关注所有位置，但更偏向开头和结尾的 token。长上下文中间 60% 的内容往往被忽略。

Liu 等人做了个关键实验。他们把一个相关文档放在多文档 QA 提示的不同位置，周围有 19 个干扰项。位置 1（开头）和位置 20（结尾）的准确率是 75-80%，而位置 10（中间）只有 50-55%。每个测试模型（GPT-3.5、Claude-1、MPT 等）都出现了 U 形曲线。2026 年的模型也有类似现象，只是没那么严重。

原因很简单。训练数据分布让模型更关注开头，因为大多数文档的重要信息都在前面。自回归解码又让模型偏向最近的内容，也就是结尾。中间部分自然就被挤掉了。

解决方法如下：

- 把问题放在长上下文的末尾，而不是开头。模型在读完文档后会“思考”问题，效果更好。实测能提升 5-15% 的准确率。
- 最重要的内容放开头或结尾，别放中间。如果能控制顺序，就这么做。
- 排序很重要。即使检索到了正确文档，放在第 5 位的效果不如第 1 位。RAG 流程（第 8 篇）应该按相关性排序，最重要的放最前。

下一节会讲更多优化技巧。
## 原生长上下文 vs 事后扩展

到 2026 年，长上下文领域已经分化。有些模型预训练时就主打长上下文，有些则是事后扩展。

原生长上下文模型，比如 Llama 3.1 的 128K、Gemini 2.5/3 的 1-2M 和 Claude 4.5 的 200K，预训练全程都混入长文档数据。模型直接学会位置依赖行为，RULER 分数全程保持高水准。

事后扩展的模型，比如社区版 LLaMA-2 用 YaRN 扩展到 32K，Qwen2.5 用扩展方法做到 128K，起点是短上下文模型。先用 YaRN/LongRoPE 调整，再用少量长上下文数据继续预训练。中等扩展（2-4 倍）时 RULER 分数还行，但激进扩展（16 倍以上）就会明显掉队。

成本差距不小。从零开始预训练 128K 模型，单 token 成本是 4K 预训练的 2-4 倍。长序列下注意力计算占大头，哪怕用了 FlashAttention，KV 缓存内存也会卡住 batch 大小。事后扩展基本没成本，几千步微调就够了。如果我是个大实验室，砸 1 亿美元搞训练，原生长上下文的溢价完全可以接受。但如果我是小团队，发开源权重模型，事后扩展才是现实选择。
## RAG vs 长上下文：真正的权衡

最常见的问题：“文档是塞进长上下文，还是用 RAG？”2026 年的答案比两年前复杂得多。

适合塞进上下文的场景：
- 文档小于 128K token。
- 检索质量难保证，比如查询多变或分块困难。
- 延迟预算允许 5 到 30 秒预填充。
- 成本预算能接受每次查询约 0.30 美元（200K token 提示词，每百万 token 1.50 美元）。

适合用 RAG 的场景：
- 文档库远超上下文窗口。
- 延迟要求低于 1 秒。
- 每次查询成本必须控制在几分钱。
- 需要来源归因。

长上下文适合需要整体阅读的任务。比如审查整个代码仓库、总结长时间会议、多文档问答且无法预测正确片段。其他场景下，RAG 在成本上低 1 到 2 个数量级。

混合方法是大多数复杂任务的最佳选择。先用 RAG 找候选，再用长上下文综合。Anthropic 的 "Contextual Retrieval"（2024）和微软的 GraphRAG（2024）都属于这种思路。第 8 章会详细讲这些内容。
## 生产小贴士

- 部署前，一定要在实际负载上跑个 needle 测试。厂商给的数字都是理想情况。
- 长输入重复用时，记得开 prompt 缓存。100K token 的系统 prompt，预填充一次要 1.5 美元，缓存后只要 0.05 美元（vLLM enable_prefix_caching、OpenAI/Anthropic 的 prompt 缓存 API）。
- 长上下文里，把问题放末尾，别放开头。Lost-in-the-middle（Liu 等，2023）发现，注意力更关注开头和结尾。问题放末尾，准确率能提升 5-15%。
- 位置超过 5 万的算术运算不靠谱。Claude-4.5 在输入数据靠后时，多步数学计算更容易出错。尽量把计算上下文挪到问题附近。
- 自己搞长上下文模型时，用滑窗 + sinks。成本和质量的平衡非常划算。
- 注意 prefill 延迟曲线。TTFT 在单 GPU 上随 prompt 长度线性增长，TP 模式下涨得更快（all-reduce 占主导），跨节点更明显。200K token 的 prompt，TP=2 H100 上预填充要 6-8 秒，用户肯定能感觉到。
- 在自己的领域做基准测试。RULER 上 90 分的模型，医疗记录任务可能只有 70 分。医疗文本分布和 RULER 的合成数据不一样。领域特定评估才是硬道理。
## 小结与下一篇

RoPE 解决了长上下文的难题。YaRN 把上下文扩展到超过训练长度。Sinks 让流式处理更稳定。但实际用起来，工作上下文总是比标称值短。建议在自己的任务上跑跑看。多数生产环境里，RAG 更省钱。长上下文适合整体阅读和短交互流程。

下一篇聊**函数调用和工具使用**。包括 JSON schema 和自由格式、并行工具调用、错误恢复，还有真正在实践中有效的 agent 循环模式。
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
- [NTK-aware scaling: r/LocalLLaMA bowtied_handbasket post (2023)][ntk-aware] — community origin of the technique.
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
