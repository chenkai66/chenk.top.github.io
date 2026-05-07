---
title: "大模型工程（一）：从 Transformer 到 MoE 的架构演化"
date: 2026-04-26 09:00:00
tags:
  - llm
  - transformer
  - moe
  - architecture
  - mamba
categories: 大模型工程
series: llm-engineering
series_order: 1
series_title: "大模型工程"
lang: zh-CN
mathjax: true
disableNunjucks: true
description: "MHA、GQA、MQA 的取舍，Mixtral 与 Qwen3-MoE 的稀疏路由，滑动窗口注意力，以及 Mamba、RWKV 这条非注意力路径——每条路的代价和适用场景。"
translationKey: "llm-engineering-1"
---
2017 年的 Transformer 方框图，到 2026 年依然是所有生产级 LLM 的骨架。但里面几乎每一块都被换了、稀疏化了，或者特化了。这个系列会从头到尾讲现代 LLM 工程栈：架构、训练、推理、检索、评估、安全、部署。第一篇聚焦方框本身。

2026 年的注意力机制长什么样？MoE 如何打破参数量和 FLOPs 的绑定关系？Mamba 和 RWKV 这些非注意力模型又在哪些场景下胜过 Transformer？

默认你已经熟悉原始 Transformer。不熟的话先看 [NLP 系列第 4 篇](/zh/nlp/04-注意力机制与transformer/)。这一篇只讲*改了什么*。

![大模型工程（一）：从 Transformer 到 MoE 的架构演化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/illustration_1.jpg)
## 改了哪些、为什么改

![fig5: architecture timeline 2017-2026](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig5_timeline.png)

现代解码器层，比如 LLaMA-3、Qwen3、Mistral、DeepSeek-V3、Yi，结构大致如下：

```python
# 单层伪代码
def layer(x, kv_cache):
    h = x + attention(rms_norm(x), kv_cache)   # pre-norm + RMSNorm
    h = h + ffn_or_moe(rms_norm(h))             # SwiGLU FFN，可能是路由的
    return h
```

相比 2017 年的 "Attention Is All You Need" [Vaswani et al., 2017]，主要有 5 处改动。

---

**Pre-norm 替代 Post-norm**  
梯度直接走残差恒等路径，完全不需要 warmup。原始的 Post-norm Transformer 必须小心调整学习率 warmup（大约 1 万步）。否则前几次梯度更新就会让 norm-then-residual 路径崩掉。Pre-norm 由 GPT-2 推广开来，[Xiong et al., 2020] 的分析证明它从第 0 步开始就能稳定训练。2020 年之后，所有生产环境中的 LLM 都采用了 Pre-norm。

---

**RMSNorm 替代 LayerNorm**  
不再减去均值，只保留均方根除法。每层少了一次 reduction 操作。[Zhang & Sennrich, 2019] 研究表明，RMSNorm 在 Transformer FFN 上的效果与 LayerNorm 相当，但运行速度能快 7%-64%。T5 和整个 LLaMA 系列都采用了 RMSNorm。到 2026 年，只有少数老旧架构还在坚持使用均值中心化。

---

**SwiGLU 替代 GELU**  
引入门控 FFN，Perplexity 提升约 2%-3%。FFN 的 FLOPs 增加 50%，但这笔账算得过来。[Shazeer, 2020] 系统性测试了各种 GLU 变体，SwiGLU 几乎在所有基准测试中都胜出。惯例是将 FFN 的中间维度缩小到原来的 2/3，因为 GLU 将投影数量从 2 个增加到 3 个，缩放中间维度可以保持总参数量不变。

---

**RoPE 替代正弦位置编码**  
[Su et al., 2021] 提出了旋转位置编码（RoPE），通过在二维子空间中旋转 Q 和 K 向量来编码相对位置。2026 年，所有支持长上下文的模型（128K-1M token）都依赖 RoPE 加上下文扩展技巧（如 NTK scaling、YaRN）。第 6 章会详细讨论这一点。

---

**GQA / MQA 替代 MHA**  
KV 缓存显著缩小，质量几乎无损。在长上下文场景中，这是生死攸关的改进。

---

密集 FFN 逐渐被稀疏 MoE（Mixture of Experts）取代。这是过去三年最大的架构变化，也是本章一半内容的重点。

补充一点历史背景：这些技术的名字（Pre-norm、RMSNorm、SwiGLU、RoPE、GQA）看似构成了一条清晰的演进路线，但每一项在首次提出后都经历了至少一年的争论。Pre-norm 和 Post-norm 的争论从 2018 年持续到 2020 年。RoPE、ALiBi 和 NoPE [Press et al., 2022; Kazemnejad et al., 2023] 的争论则贯穿了 2022-2024 年——甚至在某些极端长度外推的研究中，ALiBi 至今仍占优。2026 年所谓的“定型”架构之所以成为主流，是因为它能上线，而不是因为数学上已经完美收官。
## 注意力数学，认真讲

[Vaswani et al., 2017] 提出了 scaled dot-product attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

这个公式里有三个关键点值得细说。

**为什么分母是 $\sqrt{d_k}$？**  
$Q$ 和 $K$ 是通过线性层对单位方差输入投影得到的。初始化时，每列方差为 1。点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个单位方差项的和，方差就是 $d_k$，标准差是 $\sqrt{d_k}$。如果不除以 $\sqrt{d_k}$，点积值会随 $d_k$ 增大而膨胀。这会导致 softmax 饱和到某一行，梯度消失。除以 $\sqrt{d_k}$ 能让 logits 保持单位方差，无论头维度多大。这是 Transformer 论文里最重要的细节。没有它，模型根本训不动。

**为什么用 softmax？**  
Softmax 可微、归一化到概率单纯形，还能在某个 logit 占主导时聚焦得很尖锐。但它也是瓶颈。Linear attention [Katharopoulos et al., 2020] 把 $\text{softmax}(QK^\top)V$ 替换为 $(\phi(Q)\phi(K)^\top)V$，其中 $\phi$ 是特征映射函数。利用结合律，先算 $\phi(K)^\top V$，复杂度从 $O(n^2 d)$ 降到 $O(n d^2)$。听起来很美，但实测发现，linear attention 稳定地损失 2-5 个困惑度点。原因很简单：隐式核跟语言特性不匹配。[Schlag et al., 2021] 的研究最清楚，证明 linear attention 类似固定容量的联想记忆，远早于 scaled-dot-product 达到饱和。

**为什么 FlashAttention 不只是优化？**  
[Dao et al., 2022] 指出，vanilla attention 的 $O(n^2)$ 内存开销不是来自 FLOPs，而是来自物化 $n \times n$ 的分数矩阵。FlashAttention 把 $Q$、$K$、$V$ 分块成适合 SRAM 的小块，在每个块上在线计算数值稳定的 softmax 归一化，完全避免写入完整分数矩阵到 HBM。结果是：训练快了 2-4 倍，内存占用从 $O(n^2)$ 降到 $O(n)$。FlashAttention-2 [Dao, 2023] 重新组织计算，更好地重叠矩阵乘法和归约操作。FlashAttention-3 [Shah et al., 2024] 为 H100 添加了异步 warp-specialized 调度。到 2026 年，所有主流 LLM 训练框架（PyTorch SDPA、JAX TPU attention、vLLM、SGLang）都调用了基于 FlashAttention 的内核。

FlashAttention 核心的 online softmax 技巧值得细品。标准 softmax 需要两次遍历：一次找最大值 $\max$，一次算 $\sum e^{x - \max}$。FlashAttention 维护一个 $(m_t, \ell_t)$ 对——当前最大值和当前指数和——实现单次遍历更新。新块到来时，如果发现新的最大值 $m_{t+1} > m_t$，就缩放现有和：$\ell_{t+1} = e^{m_t - m_{t+1}} \ell_t + e^{x - m_{t+1}}$。这正是数值稳定流式统计的经典技巧，用在了注意力机制的 softmax 上。输出结果在 fp32 累加精度范围内与非分块参考实现完全一致。
## GQA、MQA、MHA：KV 缓存的真实代价

![fig1: MHA → GQA → MQA head sharing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig1_attention_heads.png)

MHA 把每个 token 映射到 $h$ 组独立的 Q、K、V，每组维度是 $d_{\text{head}}$。MQA 保留 $h$ 组 Q，但 K 和 V 只有一份，所有头共享。GQA 则介于两者之间：分成 $g$ 组，每组共享一份 K/V，覆盖 $h/g$ 个头。

长上下文场景下，KV 缓存的内存占用最关键。以 LLaMA-3-70B 为例，配置为 $h=64$、$d_{\text{head}}=128$、$L=80$ 层、FP16：

$$\text{每 token KV 字节数} = 2 \cdot L \cdot 2 \cdot h_{\text{kv}} \cdot d_{\text{head}}$$

在 32K 上下文的情况下：

| 变体   | $h_{\text{kv}}$ | KV / token | KV / 32K |
|--------|------------------|------------|----------|
| MHA    | 64               | 32 KB      | 1.0 GB   |
| GQA-8  | 8                | 4 KB       | 128 MB   |
| MQA    | 1                | 0.5 KB     | 16 MB    |

到了 2026 年，GQA-8 成为主流选择。根据 GQA 论文 [Ainslie et al., 2023]，大多数任务的质量损失不到 0.5%。KV 缓存缩小了 8 倍。MQA 曾在早期模型（如 PaLM、Falcon-7B）中尝试过，但在大规模场景下质量损失太大，尤其是长上下文对头多样性依赖较强时。

以下是 LLaMA-3 风格的 GQA 实现：

```python
import torch
import torch.nn.functional as F

class GQA(torch.nn.Module):
    def __init__(self, d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128):
        super().__init__()
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, k_cache=None, v_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        # 将 K/V 扩展到与 Q 的头数匹配（每组 n_heads/n_kv 个头共享一份 K/V）
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
```

代码中的 `repeat_interleave` 只是一个逻辑示意。实际的 kernel（如 FlashAttention-2、FlexAttention）会在 SRAM tile 内部直接广播，不会真正生成重复的 K/V。

**为什么偏偏是 GQA-8？** 分组数的选择基于实验，而非理论推导。Ainslie 等人在 T5-XXL 上测试了 $g \in \{1, 2, 4, 8, 16, 64\}$，发现从 $g=8$ 到 $g=4$ 质量明显下降，从 $g=2$ 到 $g=1$（即 MQA）又进一步下降。当 $g=8$ 时，验证集困惑度与 MHA 相差仅 0.05——在运行噪声范围内；而 $g=1$ 时相差 0.4——下游任务能明显感知。LLaMA-2-70B 直接采用了这篇论文的 $g=8$，LLaMA-3 和 Qwen3 也沿用了这一选择。8 并不是什么神奇数字，而是实验曲线上的拐点。

**多查询隐空间注意力（MLA）。** DeepSeek-V2 [DeepSeek-AI, 2024] 提出了另一种节省 KV 的思路。不再在头之间复制 K/V，而是将 token 投影到一个小的 *隐空间* 维度 $d_c \ll h \cdot d_{\text{head}}$（通常 $d_c = 512$），缓存只存储这个隐空间向量，在计算注意力时再投影回每头的 K 和 V。每 token 每层的 KV 缓存降到大约 $d_c$ 字节——比 MQA 还小——而注意力质量与 MHA 持平。代价是增加了注意力计算时的开销，但对于内存受限的长上下文推理来说，这种权衡非常划算。DeepSeek-V3 已在生产环境中使用 MLA。到 2026 年，任何新出的 100B+ 参数规模的长上下文模型，MLA 都是默认架构选择。

**通过量化压缩 KV 缓存。** 除了架构选择，还可以用 INT8 或 INT4 存储 KV 缓存。KIVI [Liu et al., 2024] 和 FP8-KV 表明，INT4 KV 配合逐通道非对称量化在大多数基准测试中几乎无损，内存占用减少 4 倍。结合 GQA-8，相比 MHA-FP16 总共可节省 32 倍的 KV 缓存。vLLM、SGLang 和 TensorRT-LLM 在 2026 年都提供了 INT8/FP8 KV 缓存作为可选项。
## 滑动窗口注意力

Mistral-7B 引入了 **滑动窗口注意力（SWA）** [Jiang et al., 2023]，窗口大小为 4096。每个 token 只关注前 $w$ 个 token。层数增加时，感受野线性扩展。第 $L$ 层第 $n$ 个 token 能看到 $L \cdot w$ 个 token。一个 32 层、$w=4096$ 的模型，有效感受野可达 13 万 token。

SWA 把 KV 缓存限制在 $w$ 个 token 内，和上下文长度无关。但模型需要学会用层叠的感受野。这导致 SWA 模型在“大海捞针”这类长程检索任务上表现不如全注意力模型。到 2026 年，多数长上下文模型会结合 SWA 和 **attention sinks** [Xiao et al., 2024]（比如 Mistral-Large 和 Qwen3）。这些模型每层固定保留前 4 到 8 个 token 在注意力窗口中，显著提升稳定性。第 6 章会详细讲。

Attention sink 是现代 LLM 中最奇怪的现象之一。Xiao 等人发现，流式推理时（固定窗口滑动并丢弃超出的 token），几千个 token 后会出现灾难性发散，困惑度飙升到几千。解决方法很简单：永远别丢前 4 个 token。保留这些 sink，即使推理几百万个 token，困惑度也能保持平稳。原因在于 softmax 输出必须归一化到 1。训练时，“多余”的注意力如果没有明确目标，就会集中在前几个 token 上。丢掉这些 token，就等于移除了注意力的“垃圾场”，分布会失控。现在的模型预训练时会显式保留这些 sink，避免这种问题。
## Mixture of Experts：参数更多，FLOPs 不变

![大模型工程（一）：从 Transformer 到 MoE 的架构演化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/illustration_2.jpg)


![fig3: sparse MoE vs dense compute](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig3_sparse_vs_dense.png)


![fig2: MoE top-2 routing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig2_moe_routing.png)

MoE 的核心思想很简单。用 $E$ 个 FFN 替代密集 FFN，再加一个小型路由器。每个 token 只选 $k$ 个专家。总参数量随 $E$ 增长，但每个 token 的 FLOPs 只和 $k$ 相关。

MoE 的历史比现代浪潮早得多。[Jacobs et al., 1991] 首次提出“混合专家”作为集成方法。[Shazeer et al., 2017] 将其扩展到 137B 参数的语言模型，使用稀疏 top-$k$ 门控。这篇论文是当前稀疏 MoE 设计的鼻祖。GShard [Lepikhin et al., 2020] 和 Switch Transformer [Fedus et al., 2022] 在 TPU 上实现了专家并行化。2024-2026 年这一波（Mixtral、DeepSeek、Qwen3）是第三代。特点是 top-$k$ 路由、无辅助损失负载均衡、细粒度专家设计以及共享专家。

Mixtral-8x7B 是典型例子。每层有 8 个专家，采用 top-2 路由，共 32 层。总参数量 46.7B（不是 56B，因为注意力层共享）。每个 token 激活参数量 12.9B。我只需支付 12.9B FLOPs 的计算成本，就能获得 46.7B 参数模型的建模能力。参数与计算的比例达到 3.6 倍。

Qwen3-235B-A22B（2025 年底发布）将比例进一步提升。总参数量 235B，激活参数量 22B，比例达到 10.7 倍。DeepSeek-V3 [DeepSeek-AI, 2024] 更夸张，总参数 671B，激活参数 37B，比例高达 18 倍。这是目前生产级模型中的最高记录。

一个最简化的 MoE FFN 实现如下：

```python
class MoEBlock(torch.nn.Module):
    def __init__(self, d_model, d_ffn, n_experts, top_k):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, n_experts, bias=False)
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model, d_ffn) for _ in range(n_experts)
        ])
        self.top_k = top_k

    def forward(self, x):                        # x: [B, T, d]
        scores = self.gate(x)                    # [B, T, E]
        topk_w, topk_i = scores.topk(self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_i[..., k]
            w   = topk_w[..., k:k+1]
            for e in range(len(self.experts)):
                mask = (idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out
```

这段代码虽然简单，但在实际生产中隐藏了两个关键问题。

**负载均衡。** 如果路由器总是选择同一个专家，那相当于浪费多余参数，最终还是密集模型。解决办法是引入辅助损失，惩罚不均衡路由。Switch Transformer 使用公式 $\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_e f_e \cdot p_e$，其中 $f_e$ 是路由到专家 $e$ 的 token 占比，$p_e$ 是路由概率均值。DeepSeek-V3 抛弃了辅助损失，改用 **无辅助损失** 的均衡方案 [Wang et al., 2024]。在 gate logits 上为每个专家添加偏置 $b_e$，通过梯度下降更新，使 $f_e$ 自然均衡。这种方法更简洁，且不影响模型质量。

**专家并行。** 当专家数量达到 256 个（如 DeepSeek-V3），单张 GPU 根本装不下。专家会被分片到多张 GPU 上，token 通过 all-to-all 操作路由。all-to-all 的延迟是主要瓶颈。DeepSeek 的 [DeepEP](https://github.com/deepseek-ai/DeepEP) 在 NVLink 上能达到 156 GB/s 的速度，接近硬件极限。

MoE 的痛点在于显存占用。显存与总参数量挂钩，即使只有 $k$ 个专家被激活也无法改变。比如 DeepSeek-V3，单张 80 GB H100 显卡根本跑不动，需要约 700 GB 权重内存。激活参数等价性只是针对 *计算* 的说法，和内存无关。
## MoE 数学：路由、容量与均衡的细节

路由器是一个线性映射 $g(x) = W_g x \in \mathbb{R}^E$，后面接一个 top-$k$ 子集上的 softmax。注意，这里必须是严格的 top-$k$，而不是“选所有超过某个阈值的专家”。原因很简单：每个 token 的 FLOPs 必须固定，否则内核没法正常工作。Mixtral 用 $k=2$，DeepSeek-V3 在 256 个专家中选 $k=8$，外加 1 个始终开启的共享专家。Qwen3-MoE 则在 128 个专家中选 $k=8$。

Switch Transformer 的辅助损失分为两部分。设 $T$ 是一批 token，$E$ 是专家数量。定义 $f_e = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}[\arg\max_e g(x_t) = e]$，表示 top-1 分配给专家 $e$ 的 token 比例。再定义 $p_e = \frac{1}{|T|} \sum_{t \in T} \text{softmax}(g(x_t))_e$，表示专家 $e$ 的平均路由概率。辅助损失公式如下：

$$\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot p_e.$$

当所有 $f_e = 1/E$ 且 $p_e = 1/E$ 时，损失最小。设计很巧妙：单独用 $f_e$ 会有不可微问题，单独用 $p_e$ 虽然可微，但无法惩罚硬分配不均衡。两者相乘后，路由器既能学习软概率均衡，也能实现硬分配均衡。

**专家容量** 是单个专家在一批数据中能接收的最大 token 数量。如果容量是 $c \cdot |T| \cdot k / E$（容量因子 $c$），那么路由到已满专家的 token 会被丢弃——这些 token 绕过 FFN，贡献为零。Switch Transformer 用 $c=1.25$。小 $c$ 节省内存和带宽，但丢弃更多 token；大 $c$ 避免丢弃，但浪费容量。现代训练方法（如 DeepSeek-V3 和 Qwen3）通常在训练时用 $c=1.0$，配合无辅助损失的均衡策略。推理时则设 $c=\infty$，因为 batch 较小，几乎不会触及容量限制。

[Wang et al., 2024] 提出了一种无辅助损失的方法，DeepSeek-V3 采用了它。这种方法在 top-$k$ 之前给路由 logits 加一个每专家偏置：

$$g'_e(x) = g_e(x) + b_e,\qquad b_e \leftarrow b_e - \eta \cdot \text{sign}(f_e - 1/E)$$

每批数据处理完后，更新偏置，将 token 推向利用率低的专家。最终用于专家输出凸组合的 softmax 权重来自未加偏置的 $g_e(x)$——只有分配过程用 $g'_e$。这种方法解耦了“选哪个专家”和“权重多少”，避免了辅助损失与模型自然路由偏好冲突带来的质量损失。
## Mixtral vs Qwen3-MoE vs DeepSeek-V3 架构对比

三种稀疏 MoE 设计，参数与计算权衡各有不同：

| 属性 | Mixtral 8x7B | Qwen3-235B-A22B | DeepSeek-V3 |
|---|---|---|---|
| 总参数 | 46.7B | 235B | 671B |
| 激活参数 | 12.9B | 22B | 37B |
| 稀疏比 | 3.6× | 10.7× | 18.1× |
| 层数 | 32 | 94 | 61 |
| 每层专家数 | 8 | 128 | 256 + 1 共享 |
| Top-$k$ | 2 | 8 | 8 |
| 单专家大小（FFN inner） | 14336 | 1536 | 2048 |
| Attention | GQA-8 | GQA-8 | MLA |
| 均衡 | aux loss | aux-loss-free | aux-loss-free |
| Tokenizer 词表 | 32K | 152K | 129K |

趋势很明显：专家更多、更小，激活的 top-$k$ 占比更低。Mixtral 的“8 个大专家选 2”是最早的稀疏设计。DeepSeek-V3 的“256 个小专家选 8 + 1 个常驻”则是当前前沿。细粒度专家分工更精细。共享专家处理通用模式，无需路由，为特化任务节省资源。

DeepSeek-V2 的“共享专家”概念值得聊聊。没有共享专家时，高频无趣模式（如英文功能词或常见代码结构）会和稀有特化模式争抢路由资源。路由器不得不学习冗余映射。有了共享专家后，普适模式交给它处理，路由专家专注特化任务。实验表明，这种方法让路由熵减半，决策更明确，下游任务质量平均提升 1-2%。

Mixtral 的“8 专家 top-2”设计部分为了优化单节点 8 卡推理效率。每个 GPU 负责一个专家，top-2 意味着每个 token 在 all-to-all 通信中激活两张卡。DeepSeek 的“256 专家 top-8”需要更复杂的专家并行策略，但负载分布更均匀，稀疏比更高。两种架构反映了不同的推理部署假设。
## 混合架构：Jamba、Zamba、Samba

纯 Attention 复杂度是 $O(n^2)$，纯 Mamba 是 $O(n)$，但复制任务表现差。解决办法？混合。

**Jamba** [Lieber et al., 2024] 是首个大规模部署的混合架构。它的模块交替设计：7 层 Mamba 加 1 层 Attention，循环往复。以 Jamba-1.5-Large 为例，32 层中 4 层是 Attention，28 层是 Mamba。再加上 MoE，FFN 用稀疏结构，16 个专家，top-2 路由。最终效果不错：总参数 398B，每 token 激活 94B，上下文窗口支持 256K。长上下文推理速度比同级密集 Transformer 快近 5 倍。

**Zamba** [Glorioso et al., 2024] 在多个 Mamba 层间插入一个共享 Attention 块。这种设计摊薄了参数开销——不需要 N 个独立 Attention 层，所有层引用同一个块。Zamba-7B-v2 使用 Mamba-2 层，网络不同深度共享一个 Attention 块。这种方式节省 30% 参数，但计算量略增（共享块运行 N 次）。

**Samba** [Ren et al., 2024] 是最激进的混合架构。Mamba 和滑动窗口 Attention 交替排列，比例 1:1。3.8B 的 Samba 模型在多数基准测试中与 Phi-3-3.8B 持平，还能扩展到 100 万 token 上下文长度。这是纯 Transformer 即使用 RoPE 扩展技巧也难做到的。

从这三个架构能总结出一点经验：少量（10%-50%）Attention 层足以弥补纯 Mamba 在复制/查找任务上的短板，同时保留线性时间优势。比例仍有争议。复制密集型任务（比如 in-context learning 新模式）需要更多 Attention 层；一般语言建模任务，少一点也够用。
## 状态空间模型：Mamba 与线性时间路线

![fig4: state-space vs attention complexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig4_mamba_vs_attention.png)

Attention 的复杂度是 $O(n^2)$，跟序列长度成平方关系。这些年，线性时间的替代方案层出不穷。比如 linear attention [Katharopoulos et al., 2020]、Performer [Choromanski et al., 2021]、Linformer 和 Reformer。可惜，这些方法都没能站稳脚跟。规模一大，它们的表现都不如传统 Attention。

直到 Mamba 出现。Mamba [Gu & Dao, 2023] 是一种 **selective state-space model**。每一层维护一个固定大小的隐状态 $h_t \in \mathbb{R}^N$，并通过递推公式更新：

$$h_t = \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t, \quad y_t = C_t \, h_t.$$

“selective” 的关键在于，$A$、$B$、$C$ 都由输入 $x_t$ 动态计算得出。这是之前的状态空间模型（比如 S4 [Gu et al., 2022]）欠缺的部分。S4 的动力学是时间不变的，无法实现基于内容的记忆功能。Mamba 做到了。

Mamba-2 [Dao & Gu, 2024] 进一步优化了 $\bar{A}_t$，将其简化为标量乘单位矩阵。这样一来，递推公式可以表示为结构化的矩阵乘法（State Space Duality，SSD）。在 GPU 上运行效率极高。参数量达到 2.7B 时，Mamba-2 的困惑度与 Pythia-2.8B 持平。推理速度快了 5 倍，每个 token 的内存占用固定，完全不需要 KV cache。

不过，Mamba 并不是 Transformer 的终结者。Jamba 论文 [Lieber et al., 2024] 和后续的一些混合模型（比如 Zamba、Samba、Falcon-Mamba）发现，纯 Mamba 在 in-context learning 和 copy 任务上表现不佳。凡是需要从序列中精确复制某个 token 的任务，Mamba 都显得力不从心。解决办法是在多层 Mamba 中插入少量注意力层。Jamba 大约每 7 层 Mamba 配 1 层注意力，而 Samba 则是 1:1 的比例。

Mamba 在 copy 任务上的瓶颈在于隐状态维度 $N$ 是固定的，通常为 64-128。如果要复制 5000 步之前的 token，模型必须将相关 token 压缩进隐状态，并且在接下来的 5000 步中一直携带它，还不能被覆盖。而 Attention 完全避开了这个问题——每一步都会重新计算所有过去 token 的相似度。[Jelassi et al., 2024] 形式化证明了：当序列长度超过隐状态大小的线性比例时，Mamba 无法解决关联召回问题，而 Attention 可以。

到 2026 年，长上下文领域的实际最优解是混合架构。通用 LLM 市场仍然由纯 Transformer（如 Qwen3、GPT-4o、Claude-4.5）主导。但在超过 256K 的上下文窗口场景下，Mamba-注意力混合模型（如 Jamba-1.5-Large、Falcon3-Mamba）以 5-10 倍更低的推理成本展现出竞争力。
## RWKV：第三条路

RWKV [Peng et al., 2023] 是一种循环网络，目标是训练时像 Transformer 一样并行计算。它有两个核心模块：time-mixing（带指数衰减的线性注意力）和 channel-mixing（门控 FFN）。RWKV-7（2025）引入了“Goose”，一种可学习的动态状态演化机制。这个改进几乎抹平了和注意力机制的质量差距。

我提 RWKV 只是为了内容完整。过去 12 个月里，所有推出非 Transformer LLM 的团队，都选择了 Mamba-2 混合架构。RWKV 社区小，工具链也不够强。生产环境优先用注意力机制。长上下文场景可以试试 Mamba 混合架构。RWKV 更适合做研究。
## 实算：70B 模型 32K 上下文的 KV cache 与 FLOPs

单步解码时，attention 块到底在干嘛？我以 LLaMA-3-70B 为例，GQA-8，$h=64$，$d_{\text{head}}=128$，$L=80$，$d_{\text{model}}=8192$，词表 $V=128256$，FFN 内层维度 $d_{\text{ffn}}=28672$（SwiGLU 的三投影结构，等价于约 57K 标准 FFN）。

**每层参数量**

- Attention QKVO 投影：公式是 $d_{\text{model}} \cdot (n_{\text{heads}} \cdot d_{\text{head}} + 2 \cdot n_{\text{kv}} \cdot d_{\text{head}} + d_{\text{model}})$。代入后得 $8192 \cdot (8192 + 2048 + 8192) = 152 \text{M}$。
- FFN SwiGLU（gate + up + down）：公式是 $3 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$。计算结果是 $3 \cdot 8192 \cdot 28672 = 705 \text{M}$。
- RMSNorm × 2：忽略不计，约 32K。
- 单层总计：约 857M。

80 层总共是 68.6B。再加上 embeddings（$V \cdot d_{\text{model}} = 1.05 \text{B}$，共享权重）和输出 norm，总计约 70B。✓

**32K 单 token 的 KV cache**

根据公式：$2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 = 327{,}680$ 字节每 token（FP16 精度）。换算下来就是 4 KB。32K tokens 总共需要 128 MB。✓

**单解码 token 的 FLOPs**

- Attention QKV 投影：公式是 $2 \cdot d_{\text{model}} \cdot (n_{\text{heads}} + 2 n_{\text{kv}}) \cdot d_{\text{head}}$。计算结果是 $2 \cdot 8192 \cdot 80 \cdot 128 \approx 168 \text{M}$。
- Attention 计算：一个新 query 对比 32K 缓存的 K/V。公式是 $4 \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot 32{,}768$。结果是 $1.07 \text{G}$。
- O 投影：公式是 $2 \cdot d_{\text{model}}^2$。计算结果是 $134 \text{M}$。
- FFN：公式是 $2 \cdot 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$。结果是 $1.41 \text{G}$。
- 单层总计：约 2.78 GFLOPs。
- 80 层总计：约 222 GFLOPs 每 token。

H100 的有效 FP16 吞吐约为 2 TFLOPs/W。理论上纯计算需要 110 ms 每 token。但实际解码是 *内存瓶颈*，不是算力瓶颈。真正卡脖子的是从 HBM 中读取 70B 参数，带宽为 3.35 TB/s，硬底限是约 21 ms 每 token（一次解码需要访问每个权重一次）。生产环境中，我会用 batching、KV 量化和 speculative decoding 来缓解这个瓶颈（第 5 章）。
## 生产真相：前沿实验室真正交付的内容

架构论文只讲了上线模型的 1%，剩下的 99% 是工程细节。以下是每个前沿实验室都在做，但不会写在模型卡片上的三件事。

**定制 CUDA kernel 融合 Attention 和 FFN。** vLLM 和 SGLang 都为 LLaMA 系列手写了优化过的 kernel。这些 kernel 把 RMSNorm、QKV 投影、RoPE、Attention 和输出投影融合成少数几个调用。普通 PyTorch 实现每层每 token 要启动 20 个 kernel，而融合后只需 2 到 3 个。80 层、64-batch 解码时，kernel 启动开销可能拖垮小模型性能。NVIDIA 的 TensorRT-LLM 更进一步，直接从计算图 IR 动态编译出针对硬件优化的 kernel。

**FP8 Attention 计算。** H100 和 B200 GPU 原生支持 FP8（E4M3 / E5M2）tensor core。FlashAttention-3 [Shah et al., 2024] 在 QK 矩阵乘法中用 FP8 计算，FP32 累加结果，吞吐量比 FP16 提升一倍。标准测试中，精度损失小于 0.1% 困惑度。生产系统（如 GPT-4o 和 Claude-4.5）在 prefill 和 decode 阶段都用了 FP8。FP8 训练（NVIDIA Transformer Engine，[Micikevicius et al., 2022]）正成为新预训练任务的标准。

**逐层学习率缩放和权重共享。** LLaMA-3 论文 [Dubey et al., 2024] 提到，attention 输出投影和 FFN 下采样投影初始化方差更小，避免训练初期不稳定。大多数生产脚本还会共享 embedding 层和 LM head 权重，减少 1-2% 参数量。单看改动很小，但叠加起来显著提升训练稳定性和效果。

“Mixtral 8x7B 是一个模型、一个架构”其实是简化说法。实际部署的 Mistral API 用了分页注意力、小型草稿模型推测解码、FP8 KV 缓存等推理技巧。这些不属于架构本身，却直接影响质量和延迟。你看 benchmark 数据时，看到的是完整部署方案，不只是架构本身。
## 常见踩坑

这些架构里，我遇到过 5 个常见问题。

**1. 硬编码 $h_{\text{kv}} = 1$ 假设 MQA。**  
有人写 7B 模型的训练脚本，直接抄教程代码，默认用了 MQA。结果模型质量比基线差了 3-5 个 perplexity 点。改成 GQA-8 后问题解决。如果从头实现，GQA-8 是更稳妥的选择。

**2. 忘记在缓存中跨头共享 K/V。**  
有些 FlashAttention 封装代码，即使在 GQA 模型中，KV 缓存也分配成 $[B, T, n_{\text{heads}}, d_{\text{head}}]$，运行时再对 K/V 做 `repeat_interleave`。这样在 GQA-8 模型上会浪费 8 倍内存。正确做法是把缓存设计为 $[B, T, n_{\text{kv}}, d_{\text{head}}]$，广播交给 attention kernel 内部处理。

**3. RMSNorm 的 $\epsilon$ 太小。**  
PyTorch 默认 $\epsilon = 10^{-6}$。FP16 训练时容易下溢，建议用 $10^{-5}$，BF16 则保持 $10^{-6}$。我曾花两天追查一个步数到 12000 时发散的 bug，最后发现就是这个原因。

**4. MoE 路由器梯度被静默清零。**  
自己实现 MoE 时容易犯错：只有被选中的专家输出会回传梯度到路由器。如果你计算 router gate，然后对 top-$k$ 重新 softmax，梯度路径必须使用 *原始* logits，而不是只用 top-$k$ 子集。否则路由器学不会避开表现差的专家。Mixtral 和 DeepSeek-V3 的参考实现都处理得很好，但很多从零实现的代码却常出错。

**5. SwiGLU FFN 的 inner dim 没调整。**  
标准 Transformer FFN 的 inner dim 是 $4 \cdot d_{\text{model}}$。直接换成 SwiGLU，投影数量会变成三倍，参数量也会膨胀 3 倍。通常做法是把 inner dim 缩小到原来的 2/3（大约 $2.67 \cdot d_{\text{model}}$），以保持总参数量基本不变。一些 LLaMA 开源分支没注意这点，导致最终参数量和预期不符。
## 研究前沿 2024-2026

"GQA + MoE Transformer" 共识之后，接下来有哪些新方向？

**Differential attention** [Ye et al., 2024] 提出了一种新方法：用不同参数生成两份注意力图，然后相减。实测发现，这种方法能有效抑制注意力噪声，提升长上下文检索效果。一些 2025-2026 年发布的模型已经开始采用。

**注意力原生稀疏化。** Native Sparse Attention（NSA）等研究直接训练模型，让它天然关注稀疏的键子集，而不是事后优化稀疏性。[Yuan et al., 2025] 的研究表明，在长上下文基准测试中，原生稀疏注意力可以用更低的计算量和 KV 成本达到与密集注意力相当的效果。

**Linear attention 再次崛起。** 2024-2025 年有多篇论文（Gated Linear Attention、RetNet [Sun et al., 2023]、TransNormerLLM）通过引入门控和衰减机制，大幅缩小了与全注意力的质量差距。虽然能否在生产环境中完全取代 softmax attention 还有待验证，但这是过去 5 年里差距最小的一次。

**Diffusion 语言模型。** [Lou et al., 2024]（SEDD）和 [Sahoo et al., 2024]（MDLM）证明，离散扩散模型在文本生成上的困惑度可以媲美自回归模型。2025 年发布的 Mercury Coder 声称，1000-token 的生成延迟不到 50 毫秒，比任何同长度的自回归模型都快。不过，这种质量优势能否在更大规模上保持，目前还不清楚。但它是近年来非自回归路线最有潜力的竞争者。

**推理时算力扩展。** [Snell et al., 2024] 发现，增加推理阶段的计算量（比如 chain-of-thought、self-consistency、MCTS）可以在相同质量下替代部分预训练计算量。o1 / DeepSeek-R1 / Claude-thinking 系列模型认真践行了这一点：一个 32B 参数的 thinking 模型，通过投入 10 倍的推理算力，能在复杂推理任务上超越 70B 参数的非 thinking 模型。这虽然不是架构层面的变化，但却重新定义了架构与成本之间的权衡关系。
## 什么场景用什么

| 场景 | 架构 | 理由 |
|---|---|---|
| 通用对话、代码 | Dense Transformer (LLaMA-3, Qwen3-Dense) | 参数质量高，工具链成熟 |
| 成本敏感的推理 | MoE (Mixtral, DeepSeek-V3, Qwen3-MoE) | 参数与计算量比达到 3-10 倍 |
| 超长上下文（256K+），低延迟 | Hybrid Mamba-attention (Jamba) | 每个 token 的内存占用固定 |
| 边缘设备（< 4 GB） | 量化小模型 (Qwen3-1.7B INT4, Phi-4-mini) | 内存受限，MoE 不适用 |
| 推理密集型任务（数学、代码） | Dense Transformer + thinking RL | 质量随推理算力提升 |

选架构不是为了挑最好的，而是挑最符合部署条件的。多卡推理时，计算量是瓶颈，MoE 更合适。单卡推理时，显存有限，Dense 占优。上下文长度是瓶颈时，Hybrid 是更好的选择。
## 小结与下一篇

现代 LLM 还是 Transformer，但每个模块都被重新设计过。为了稳定性，加了 pre-norm 和 RMSNorm。为了质量，用了 SwiGLU 和 RoPE。为了降低推理成本，引入了 GQA 和滑动窗口。为了参数效率，实现了 MoE。纯非注意力模型（比如 Mamba 和 RWKV）通用性稍差，但在长上下文混合架构里表现不错。

下一篇讲 Tokenization。为什么 CJK 字符的 token 比英文贵 2-3 倍？BPE 在字节流上到底干了什么？聊天模板的 token 怎么影响模型行为？这是个容易被忽略的部分，但跳过的人最后都会后悔。
## 参考文献

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.
- Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *NeurIPS*.
- Lepikhin, D., Lee, H., Xu, Y., et al. (2020). GShard: Scaling giant models with conditional computation and automatic sharding. *[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)*.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive Transformers with linear attention. *ICML*.
- Xiong, R., Yang, Y., He, D., et al. (2020). On layer normalization in the Transformer architecture. *ICML*.
- Shazeer, N. (2020). GLU variants improve Transformer. *[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)*.
- Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. *[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)*.
- Choromanski, K., Likhosherstov, V., Dohan, D., et al. (2021). Rethinking attention with Performers. *ICLR*.
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers are secretly fast weight programmers. *ICML*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*.
- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*.
- Press, O., Smith, N., & Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation (ALiBi). *ICLR*.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces (S4). *ICLR*.
- Micikevicius, P., Stosic, D., Burgess, N., et al. (2022). FP8 formats for deep learning. *[arXiv:2209.05433](https://arxiv.org/abs/2209.05433)*.
- Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023). GQA: Training generalized multi-query Transformer models from multi-head checkpoints. *EMNLP*.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)*.
- Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)*.
- Jiang, A., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. *[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)*.
- Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer era. *EMNLP Findings*.
- Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive network: A successor to Transformer for large language models (RetNet). *[arXiv:2307.08621](https://arxiv.org/abs/2307.08621)*.
- Kazemnejad, A., Padhi, I., Ramamurthy, K., et al. (2023). The impact of positional encoding on length generalization in Transformers. *NeurIPS*.
- Jiang, A., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of experts. *[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)*.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *ICML*.
- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient streaming language models with attention sinks. *ICLR*.
- DeepSeek-AI. (2024). DeepSeek-V3 technical report. *[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)*.
- DeepSeek-AI. (2024). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. *[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)*.
- Lieber, O., Lenz, B., Bata, H., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. *[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)*.
- Glorioso, P., Anthony, Q., Tokpanov, Y., et al. (2024). Zamba: A compact 7B SSM hybrid model. *[arXiv:2405.16712](https://arxiv.org/abs/2405.16712)*.
- Ren, L., Liu, Y., Lu, Y., et al. (2024). Samba: Simple hybrid state space models for efficient unlimited context language modeling. *[arXiv:2406.07522](https://arxiv.org/abs/2406.07522)*.
- Wang, L., Gao, H., Zhao, C., et al. (2024). Auxiliary-loss-free load balancing strategy for mixture-of-experts. *[arXiv:2408.15664](https://arxiv.org/abs/2408.15664)*.
- Liu, Z., Yuan, J., Jin, H., et al. (2024). KIVI: A tuning-free asymmetric 2bit quantization for KV cache. *ICML*.
- Shah, J., Bikshandi, G., Zhang, Y., et al. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *NeurIPS*.
- Jelassi, S., Brandfonbrener, D., Kakade, S., & Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. *ICML*.
- Ye, T., Dong, L., Xia, Y., et al. (2024). Differential Transformer. *[arXiv:2410.05258](https://arxiv.org/abs/2410.05258)*.
- Lou, A., Meng, C., & Ermon, S. (2024). Discrete diffusion modeling by estimating the ratios of the data distribution (SEDD). *ICML*.
- Sahoo, S., Arriola, M., Schiff, Y., et al. (2024). Simple and effective masked diffusion language models (MDLM). *NeurIPS*.
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *[arXiv:2408.03314](https://arxiv.org/abs/2408.03314)*.
- Dubey, A., Jauhri, A., Pandey, A., et al. (2024). The Llama 3 herd of models. *[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)*.
- Yuan, J., Gao, H., Dai, D., et al. (2025). Native sparse attention: Hardware-aligned and natively trainable sparse attention. *[arXiv:2502.11089](https://arxiv.org/abs/2502.11089)*.
- Jacobs, R., Jordan, M., Nowlan, S., & Hinton, G. (1991). Adaptive mixtures of local experts. *Neural Computation* 3(1):79-87.
