---
title: "大模型工程（一）：Transformer 到 MoE"
date: 2026-03-27 09:00:00
tags:
  - LLM
  - Transformer
  - moe
  - architecture
  - mamba
categories: 大模型工程
series: llm-engineering
series_order: 1
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "MHA、GQA、MQA 的取舍，Mixtral 与 Qwen3-MoE 的稀疏路由，滑动窗口注意力，以及 Mamba、RWKV 这条非注意力路径——每条路的代价和适用场景。"
translationKey: "llm-engineering-1"
---
2017 年的 Transformer 块，到了 2026 年依然是所有生产级 LLM 的轮廓，但内部零件几乎全换了——有的被彻底替换，有的被稀疏化，有的演变为专用模块。本系列教程将端到端覆盖现代大语言模型技术栈：架构、训练、推理、检索增强、评估、安全与部署。第一章咱们就聊这个块本身：2026 年模型中的注意力机制有何演进，MoE 如何解耦参数量与计算量（FLOPs），以及非注意力架构（如 Mamba、RWKV）在哪些任务或场景下展现出对 Transformer 的优势。

我默认你已经熟悉原始 Transformer 块。如果不熟，[NLP 系列第 4 部分](/zh/nlp/attention-transformer/) 里有讲。本章只讲*现在有什么不同*。

![LLM Engineering (1): Architectures from Transformer to MoE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/illustration_1.png)

## 变了什么，为什么变

![fig5: architecture timeline 2017-2026](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig5_timeline.png)


现代的 Decoder 块——LLaMA-3、Qwen3、Mistral、DeepSeek-V3、Yi——长这样：

```python
# pseudocode for a single decoder layer
def layer(x, kv_cache):
    h = x + attention(rms_norm(x), kv_cache)   # pre-norm + RMSNorm
    h = h + ffn_or_moe(rms_norm(h))             # SwiGLU FFN, sometimes routed
    return h
```

相较于《Attention Is All You Need》[Vaswani et al., 2017] 中的原始设计，现代 Decoder 模块主要存在五处关键改进：

1. **Pre-norm** 替代 post-norm —— 梯度流过干净的残差恒等路径，不需要 warmup。原始 post-norm Transformer 需要精心设计学习率预热（约 10K 步），否则初始几次梯度更新便可能导致 norm-then-residual 路径不稳定甚至崩溃。Pre-norm 最早由 GPT-2 采用并推广，后经 [Xiong et al., 2020] 严格理论分析，证实其可消除学习率预热需求，实现从训练初始阶段起的稳定收敛。2020 年以后的生产级 LLM 全用 pre-norm。
2. **RMSNorm** 替代 LayerNorm —— 去掉均值，只留 RMS 除数。每层少一次归约操作。[Zhang & Sennrich, 2019] 证明 RMSNorm 在 Transformer FFN 上能达到 LayerNorm 的质量，墙钟时间快 ~7-64%。T5 和整个 LLaMA  lineage 都采用了它；到了 2026 年，只有少数遗留架构还在做均值中心化。
3. **SwiGLU** 替代 GELU —— 门控 FFN，困惑度赢 ~2-3%，虽使 FFN 计算量增加约 50% FLOPs，但带来的困惑度下降（约 2–3%）使其整体收益显著。[Shazeer, 2020] 的 "GLU variants" 论文系统扫了一遍门控激活函数；SwiGLU（Swish  gating）在几乎所有基准上都赢了。实践中通常将 FFN 内部隐藏层维度缩减约 2/3，以维持 FFN 总参数量基本不变——这是因为 SwiGLU 引入了三次线性投影（而非 GELU 的两次）。
4. **RoPE** 替代正弦位置编码 —— [Su et al., 2021] 提出旋转嵌入，通过在 2D 子空间旋转 Q 和 K 向量来编码相对位置。结合上下文扩展技术（如 NTK-aware scaling 和 YaRN），RoPE 已成为 2026 年支持 128K–1M token 长上下文的主流模型的核心位置编码方案。第 6 章会深入讲这个。
5. **GQA / MQA** 替代 MHA —— 更小的 KV cache，质量不变。长上下文场景下至关重要。

密集型前馈网络（Dense FFN）正日益被稀疏混合专家（MoE）架构取代——这已成为过去三年最重大的架构演进，也是本章重点讨论内容之一。

需注意一个历史事实：尽管 pre-norm、RMSNorm、SwiGLU、RoPE、GQA 等术语看似构成一条线性演进脉络，但每项技术自首篇论文发表后均经历了至少一年的实证争议与社区验证。Pre-norm 对战 post-norm 的争论持续到 2018-2020 年。RoPE 对战 ALiBi 对战 NoPE [Press et al., 2022; Kazemnejad et al., 2023] 的争论持续到 2022-2024 年——而且有些研究显示 ALiBi 在极端长度外推上依然胜出。2026 年所谓‘定型’的架构，并非源于数学上的终极完备性，而是因其工程可行性与线上交付能力已获充分验证。

## 注意力数学，正经讲

来自 [Vaswani et al., 2017] 的缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

这个公式里有三件事，值得咱们深挖一下，不能只停留在标准的一行解释上。

**为什么分母是 $\sqrt{d_k}$。** $Q$ 和 $K$ 是单位方差输入通过线性层的投影，初始化保证每列都是单位方差。点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 然后是 $d_k$ 个单位方差项的和，所以它的方差是 $d_k$，标准差是 $\sqrt{d_k}$。如果没有这个除数，点积会随着 $d_k$ 增大而增大，softmax 会在某一行饱和，穿过 softmax 的梯度就会消失（one-hot 输入处 softmax 的雅可比矩阵为零）。除以 $\sqrt{d_k}$ 能让 softmax 前的 logits 保持单位方差，不管 head 维度多大。这是整篇 Transformer 论文里最重要的细节——没有它，模型根本训不起来。

**为什么用 softmax 而不是别的。** Softmax 可微，归一化到概率单纯形，当某个 logit 占主导时能给出尖锐的聚焦。它也是瓶颈所在。线性注意力 [Katharopoulos et al., 2020] 用 $(\phi(Q)\phi(K)^\top)V$ 替换 $\text{softmax}(QK^\top)V$（$\phi$ 为特征映射），然后利用结合律先算 $\phi(K)^\top V$ —— 这把成本从 $O(n^2 d)$ 变成了 $O(n d^2)$。线性注意力听起来很棒，直到你去实测。它 一致地 丢掉 2-5 个困惑度点，因为隐式核跟语言不匹配。最干净的研究 [Schlag et al., 2021] 显示，线性注意力表现像固定容量的关联记忆，早在缩放点积饱和之前就饱和了。

**为什么 FlashAttention 不只是个优化。** [Dao et al., 2022] 表明， vanilla 注意力的 $O(n^2)$ 显存占用不是来自 FLOPs —— 而是来自实例化 $n \times n$ 分数矩阵。FlashAttention 把 $Q$、$K$、$V$ 切成适合 SRAM 的块，每块计算带在线数值稳定归一化的 softmax，从不把完整分数矩阵写回 HBM。结果：训练时墙钟时间加速 2-4 倍，显存从 $O(n^2)$ 降到 $O(n)$。FlashAttention-2 [Dao, 2023] 重组了工作流以更好地重叠 matmul 和归约；FlashAttention-3 [Shah et al., 2024] 为 H100 添加了异步 warp 专用调度。2026 年每个生产级 LLM 训练框架（PyTorch SDPA、JAX TPU attention、vLLM、SGLang）都在调用 FlashAttention 衍生的 kernel。

FlashAttention 核心的在线 softmax 技巧值得理解。标准 softmax 需要遍历行两次：一次求 $\max$，一次求 $\sum e^{x - \max}$。FlashAttention 通过维护一个运行中的 $(m_t, \ell_t)$ 对——运行最大值和运行指数和——在新块到达时更新它们，从而一次遍历完成。当你看到一个新的最大值 $m_{t+1} > m_t$，你重新缩放现有的和：$\ell_{t+1} = e^{m_t - m_{t+1}} \ell_t + e^{x - m_{t+1}}$。这和数值稳定流式统计里用的技巧一样，只是应用到了注意力内部的 softmax 上。在 fp32 累加的精度范围内，输出与未分块的参考实现 位精确一致。

## GQA, MQA, MHA：KV cache 的真实成本

![fig1: MHA → GQA → MQA head sharing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig1_attention_heads.png)


多头注意力（MHA）把每个 token 投影到 $h$ 个独立的 Q、K、V 向量，维度为 $d_{\text{head}}$。多查询注意力（MQA）投影 $h$ 个 Q 向量，但只有**一个** K 和一个 V，跨 head 共享。分组查询注意力（GQA）是中间方案：$g$ 组，每组在 $h/g$ 个 heads 间共享一个 K/V。

到了长上下文场景，KV cache 显存占用才是关键。对于 LLaMA-3-70B，$h=64$，$d_{\text{head}}=128$，$L=80$ 层，FP16：

$$\text{KV bytes per token} = 2 \cdot L \cdot 2 \cdot h_{\text{kv}} \cdot d_{\text{head}}$$

对于 32K token 上下文：

| Variant | $h_{\text{kv}}$ | KV / token | KV / 32K context |
|---|---|---|---|
| MHA | 64 | 32 KB | 1.0 GB |
| GQA-8 | 8 | 4 KB | 128 MB |
| MQA | 1 | 0.5 KB | 16 MB |

GQA-8 是 2026 年的主流选择。它保留了 MHA 几乎全部的质量（GQA 论文 [Ainslie et al., 2023] 报告大多数任务退化不到 0.5%），同时给出 8 倍 KV cache 缩减。MQA 在早期模型（PaLM、Falcon-7B）里试过，但在大规模下质量损失太多，尤其是长上下文场景下 head 多样性很重要。

一个 LLaMA-3 风格的带 GQA 注意力块：

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
        # repeat K, V to match Q heads (n_heads / n_kv groups share)
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
```

`repeat_interleave` 只是概念上的——实际 kernel（FlashAttention-2、FlexAttention）在 SRAM 块内部处理广播，不会实例化重复的 K/V。

**为什么偏偏是 GQA-8。** 组数的选择是经验的，不是理论的。Ainslie 等人在 T5-XXL 上扫了 $g \in \{1, 2, 4, 8, 16, 64\}$，发现质量在 $g=8$ 和 $g=4$ 之间急剧下降，然后在 $g=2$ 和 $g=1$（MQA）之间再次下降。在 $g=8$ 时，相对于完整 MHA 的验证困惑度差异是 0.05——在运行间噪声范围内。在 $g=1$ 时是 0.4——大到足以在下流任务中注意到。LLaMA-2-70B 直接从这篇论文选了 $g=8$；LLaMA-3 保留了它；Qwen3 也保留了它。8 不是什么魔法数字——它只是实验曲线上的拐点。

**多查询 latent 注意力（MLA）。** DeepSeek-V2 [DeepSeek-AI, 2024] 引入了另一种节省 KV 的思路。不是跨 heads 复制 K/V，而是把 token 投影到一个小的 *latent* 维度 $d_c \ll h \cdot d_{\text{head}}$（通常 $d_c = 512$），只 cache 那个 latent 向量，然后在运行时重新投影到每头的 K 和 V。KV cache 降到每层每 token 大约 $d_c$ 字节——比 MQA 还小——同时注意力质量匹配 MHA。代价是注意力时刻计算量更大，但对于显存受限的长上下文推理，这个取舍是划算的。DeepSeek-V3 在生产中交付了 MLA。截至 2026 年，MLA 是开放前沿上任何瞄准长上下文的新 100B+ 模型的架构选择。

**通过量化压缩 KV-cache。** 跟架构选择正交，你可以用 INT8 或 INT4 存储 KV cache 而不是 FP16。KIVI [Liu et al., 2024] 和 FP8-KV 显示，带每通道不对称量化的 INT4 KV 在大多数基准上 几乎无损，显存缩减到原来的 1/4。结合 GQA-8，相比 MHA-FP16 你能得到 32 倍总 KV 缩减。vLLM、SGLang 和 TensorRT-LLM 在 2026 年都提供 INT8/FP8 KV cache 作为选项。
## Sliding window attention

Mistral-7B 引入了 **sliding window attention (SWA)**，窗口大小 4096 [Jiang et al., 2023]。每个 token 只关注前 $w$ 个 token。感受野依然随深度线性增长——第 $L$ 层位置 $n$ 的 token 能回看 $L \cdot w$ 个 token——所以一个 32 层模型配合 $w=4096$，有效感受野能达到 131K token。

SWA 把 KV cache 限制在 $w$ 个 token，不管上下文多长。坑在于：模型还得学会利用这种分层感受野，这意味着 SWA 模型在需要精确长程检索的任务上（比如 "needle in a haystack"）往往不如全注意力模型。大多数 2026 年的长上下文模型会把 SWA 和 **attention sinks** [Xiao et al., 2024] 结合起来（比如 Mistral-Large, Qwen3）——在每层的注意力窗口里保留前 4-8 个 token，这能极大地稳定长上下文行为。第 6 章会细聊这个。

attention sink 现象是现代 LLM 里最奇怪的实证观察之一。Xiao 等人发现，流式推理（滑动固定窗口向前，丢弃掉出窗口的 token）在几千 token 后会灾难性发散——perplexity 飙升到几千。修复方法简单得离谱：永远不要驱逐前 4 个 token。只要保留 sinks，perplexity 即使在百万 token 级别也能保持平稳。机制上的解释是 softmax 的和总是 1，所以任何没有 meaningful target 的“多余”注意力质量，在训练期间都会被倾倒到前几个 token 上。驱逐这些 token 就等于拆掉了倾倒点，注意力分布直接爆炸。现代模型预训练时都会显式保留 sinks 来避免这种脆弱性。

## Mixture of experts: more parameters, same FLOPs

![LLM Engineering (1): Architectures from Transformer to MoE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/illustration_2.png)


![fig3: sparse MoE vs dense compute](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig3_sparse_vs_dense.png)


![fig2: MoE top-2 routing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig2_moe_routing.png)


MoE 的核心 trick：把 dense FFN 换成 $E$ 个 FFN（即"experts"），再加一个小的 router 每个 token 选 $k$ 个。总参数量随 $E$ 缩放；单 token FLOPs 随 $k$ 缩放。

MoE 在深度学习里的谱系比现代浪潮早得多。[Jacobs et al., 1991] 最早把 "mixture of experts" 作为 ensemble 想法提出。[Shazeer et al., 2017] 把它扩展到 137B 参数量的语言模型，用了 sparse top-$k$ gating——那篇论文是所有当前 sparse MoE 设计的直接祖先。GShard [Lepikhin et al., 2020] 和 Switch Transformer [Fedus et al., 2022] 在 TPU 上实现了专家并行产品化。2024-2026 这波（Mixtral, DeepSeek, Qwen3）是第三代：top-$k$ 路由配合 auxiliary-loss-free 负载均衡、细粒度专家（更多、更小的专家）以及 shared-expert 设计。

Mixtral-8x7B 是教科书般的案例。每层 8 个专家，top-2 路由，32 层。总参数 46.7B（不是 56B——注意力部分是共享的）。单 token 激活参数 12.9B。所以你只花了 12.9B FLOPs 的算力，就拥有了 46.7B 模型的建模能力。参数 - 算力比率达到 3.6×。

Qwen3-235B-A22B（2025 年底发布）把这个推得更远：总参数 235B，激活 22B，比率 10.7×。DeepSeek-V3 [DeepSeek-AI, 2024] 总参数 671B / 激活 37B，比率 18×——这是目前已发布模型中最高的比率。

一个最小化的 MoE FFN 长这样：

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
            idx = topk_i[..., k]                 # [B, T]
            w   = topk_w[..., k:k+1]             # [B, T, 1]
            for e in range(len(self.experts)):
                mask = (idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out
```

这段朴素代码隐藏了两个在生产环境至关重要的问题：

**负载均衡。** 如果 router 总是选同一个专家，那你就得到了一个带浪费参数的 dense 模型。修复方法是加一个 auxiliary loss 惩罚不平衡的路由——Switch Transformer 的 $\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_e f_e \cdot p_e$，其中 $f_e$ 是路由到专家 $e$ 的 token 比例，$p_e$ 是 router 概率的平均值。DeepSeek-V3 抛弃了 aux loss，改用 **auxiliary-loss-free** 平衡方案 [Wang et al., 2024]：给 gate logits 加一个每专家 bias $b_e$，通过梯度下降更新它以均衡 $f_e$。更干净，没有质量惩罚。

**专家并行。** 有了 256 个专家（DeepSeek-V3），你没法把它们全塞进一张 GPU。专家被分片到多张 GPU 上，token 通过 all-to-all 路由过去。all-to-all 延迟是瓶颈——DeepSeek 的 [DeepEP](https://github.com/deepseek-ai/DeepEP) 在 NVLink 上能跑到 156 GB/s，这已经接近硬件天花板了。

MoE 尖锐的一面在于：总 VRAM 随总参数量缩放，即使只有 $k$ 个被激活。你没法在单张 80 GB H100 上跑 DeepSeek-V3——你需要约 700 GB 的权重显存。Active-param-equivalence 是个 *算力*  claim，不是显存 claim。

## MoE math: routing, capacity, and balancing in detail

Router 是一个线性映射 $g(x) = W_g x \in \mathbb{R}^E$，接着在 top-$k$ 子集上做 softmax。必须是 Top-$k$——不是“选所有超过阈值的专家”，因为单 token FLOPs 必须是确定的，kernel 才能工作。Mixtral 用 $k=2$，DeepSeek-V3 用 256 个专家里选 $k=8$ 再加 1 个常开共享专家，Qwen3-MoE 用 128 个里选 $k=8$。

Switch Transformer auxiliary loss 分解为两个因子。设 $T$ 为 token batch，$E$ 为专家数量。定义 $f_e = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}[\arg\max_e g(x_t) = e]$（top-1 路由到专家 $e$ 的 token 比例）和 $p_e = \frac{1}{|T|} \sum_{t \in T} \text{softmax}(g(x_t))_e$（专家 $e$ 的平均 router 概率）。Loss 是

$$\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot p_e.$$

当所有 $e$ 都满足 $f_e = 1/E$ 且 $p_e = 1/E$ 时最小化。巧妙的地方在于同时用了这两个量：$f_e$ 单独有不可微的 arg-max，$p_e$ 单独可微但不惩罚硬不平衡。相乘让 router 学会在软概率和实际硬分配上都保持平衡。

**专家容量** 是一个 batch 中任何一个专家能接收的最大 token 数。如果容量是 $c \cdot |T| \cdot k / E$（容量因子 $c$），路由到满员专家的 token 会被 *丢弃*——它们绕过 FFN，贡献为零。Switch Transformer 用 $c=1.25$。更低的 $c$ 节省显存和 all-to-all 带宽但丢弃更多 token；更高的 $c$ 避免丢弃但浪费容量。现代训练（DeepSeek-V3, Qwen3）通常在训练时用 $c=1.0$ 配合 aux-loss-free 平衡，推理时则用 $c=\infty$（这没问题，因为 batch size 够小，很少触达容量上限）。

[Wang et al., 2024] 提出的 auxiliary-loss-free 方法被 DeepSeek-V3 采用，用加在 routing logits 上 top-$k$ 之前的每专家 bias 替换了 aux loss：

$$g'_e(x) = g_e(x) + b_e,\qquad b_e \leftarrow b_e - \eta \cdot \text{sign}(f_e - 1/E)$$

Bias 在每个 batch 后更新，把 token 推向利用率不足的专家。用于专家输出凸组合的 softmax 权重依然来自无 bias 的 $g_e(x)$——只有分配过程用 $g'_e$。这把“选哪个专家”和“强度多大”解耦了，避免了 aux-loss 在对抗模型自然路由偏好时带来的质量惩罚。

## Mixtral vs Qwen3-MoE vs DeepSeek-V3 architecture comparison

三种 sparse-MoE 设计，三种不同的参数 - 算力权衡：

| Property | Mixtral 8x7B | Qwen3-235B-A22B | DeepSeek-V3 |
|---|---|---|---|
| Total params | 46.7B | 235B | 671B |
| Active params | 12.9B | 22B | 37B |
| Sparsity ratio | 3.6× | 10.7× | 18.1× |
| Layers | 32 | 94 | 61 |
| Experts per layer | 8 | 128 | 256 + 1 shared |
| Top-$k$ | 2 | 8 | 8 |
| Expert size (FFN inner) | 14336 | 1536 | 2048 |
| Attention | GQA-8 | GQA-8 | MLA |
| Balancing | aux loss | aux-loss-free | aux-loss-free |
| Tokenizer vocab | 32K | 152K | 129K |

趋势很明显：更多、更小的专家，相对于总数更少的 top-$k$ 激活。Mixtral 的"8 个大专家，选 2 个”是 OG sparse 设计。DeepSeek-V3 的"256 个小专家，选 8 个 + 1 个常开”是当前前沿。直觉上，细粒度专家允许更多的 specialization；常开共享专家捕捉不需要路由的通用模式（把路由专家的容量留给真正需要 specialized 的工作）。

DeepSeek-V2 提出的"shared expert"想法值得单列一段。没有它，频繁出现的无聊模式（英语功能词、常见代码惯用法）会和罕见的专用模式竞争路由槽位，router 不得不为每个模式学习一个浪费的映射。有了它，共享专家吞掉通用模式，路由专家做 specialization。实证表明这减半了路由熵（路由更果断），同时平均提升下游质量 1-2%。

Mixtral 的"8 专家，top-2"部分是为了单 8-GPU 节点上的推理效率选的——每 GPU 一个专家，top-2 意味着每个 token 在 all-to-all 中激活 2 张 GPU。DeepSeek 的"256 专家，top-8"需要更复杂的专家并行，但负载分布更好，拿到了更高的 sparsity ratio。这些架构编码了不同的推理部署假设。

## Hybrid architectures: Jamba, Zamba, Samba

纯 attention 是 $O(n^2)$。纯 state-space (Mamba) 是 $O(n)$ 但在复制风格任务上表现差。自然的答案是混合。

**Jamba** [Lieber et al., 2024] 是第一个广泛部署的混合架构。Jamba block 交替结构：7 层 Mamba，1 层 attention，循环。在 Jamba-1.5-Large 的 32 层中，4 层是 attention，28 层是 Mamba。上面再加 MoE——FFN 是 sparse 的，16 个专家，top-2 路由。结果：总参数 398B，单 token 激活 94B，256K 上下文窗口，长上下文推理速度比可比 dense Transformer 快约 5 倍。

**Zamba** [Glorioso et al., 2024] 在许多 Mamba block 之间 穿插一个共享 attention block。共享 attention 摊销了参数成本——不是 N 层 attention，而是 N 次引用同一个 attention block。Zamba-7B-v2 使用 Mamba-2 层，网络中多个深度共享一层 attention。这种模式节省了 30% 参数，代价是稍微多一点算力（共享 block 运行 N 次）。

**Samba** [Ren et al., 2024] 是最激进的混合：Mamba 和 sliding-window attention 1:1 交替。3.8B 的 Samba 模型声称在大多数 benchmark 上匹配 Phi-3-3.8B，同时能干净地外推到 1M token 上下文——这是纯 Transformer 即使有 RoPE 扩展 trick 也难以做到的。

这三者的实证教训：**一小部分（10-50%）attention 层足以恢复纯 Mamba 在 copy/lookup 任务上丢失的能力**，同时保留大部分线性时间优势。确切比例仍有争议。对于 copy-heavy 任务（新模式的 in-context learning），更多 attention 有帮助。对于一般语言建模，少一点 attention 没问题。
## 状态空间模型：Mamba 与线性时间替代方案

![fig4: state-space vs attention complexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/01-architectures/fig4_mamba_vs_attention.png)

Attention 的计算复杂度是序列长度的 $O(n^2)$。线性时间的替代方案其实一直都有：linear attention [Katharopoulos et al., 2020]、Performer [Choromanski et al., 2021]、Linformer、Reformer。但这些方案都没成气候——规模一大，效果全都不如 vanilla attention。

直到 Mamba 出现。Mamba [Gu & Dao, 2023] 是一种**选择性状态空间模型**：每一层维护一个固定大小的隐藏状态 $h_t \in \mathbb{R}^N$，并通过递归更新：

$$h_t = \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t, \quad y_t = C_t \, h_t.$$

关键在于“选择性”：$A$、$B$、$C$ 是*依赖输入*的，由 $x_t$ 计算得出。这才是 missing piece——早期的 SSM（比如 S4 [Gu et al., 2022]） dynamics 是时不变的，做不到基于内容的记忆。Mamba 可以。

Mamba-2 [Dao & Gu, 2024] 把 $\bar{A}_t$ 变成了标量乘单位矩阵，这让递归可以表达为结构化矩阵乘法（State Space Duality, SSD），在 GPU 上跑得非常高效。2.7B 参数量的模型， perplexity 能匹敌 Pythia-2.8B，推理速度快 5 倍，而且每个 token 的内存占用是常数——根本不需要 KV cache。

别误会，Mamba 不是来消灭 Transformer 的。Jamba 论文 [Lieber et al., 2024] 以及随后的几个混合架构（Zamba, Samba, Falcon-Mamba）都发现，**纯 Mamba 在 in-context learning 和 copy 任务上表现不佳**——具体来说，任何需要从序列早期查找确切 token 的任务都吃力。解决办法是在多层 Mamba 中穿插少量 attention 层。Jamba 大概是 7 层 Mamba 配 1 层 attention；Samba 是 1:1。

Mamba 搞不定复制任务的机理原因在于，它的隐藏状态维度 $N$ 是固定的（通常 64-128）。要想 copy 5000 个位置之前的 token，模型必须把相关 token 压缩并路由进隐藏状态，然后在 5000 步内携带它而不被覆盖。Attention 避开了这个问题，它在每一步都重新计算与所有过去 token 的相似度。[Jelassi et al., 2024] 从理论上证明了，对于超过隐藏状态大小比例的序列长度，Mamba 无法解决 associative-recall 问题，而 attention 可以。

到了 2026 年，混合架构是长上下文场景的实际 SOTA。纯 Transformer（Qwen3, GPT-4o, Claude-4.5）依然主导通用 LLM 市场，但对于 >256K 的上下文窗口，混合 Mamba-attention 模型（Jamba-1.5-Large, Falcon3-Mamba）在推理成本降低 5-10 倍的情况下，竞争力相当。

## RWKV：第三条路

RWKV [Peng et al., 2023] 是一种循环网络，设计目标是在训练时能像 Transformer 一样并行化。它使用时序混合块（带指数衰减的 linear attention）和通道混合块（gated FFN）。RWKV-7（2025）引入了"Goose"——一种学习到的动态状态演化机制，缩小了大部分与 attention 的质量差距。

我把 RWKV 写在这儿主要是为了完整性。说实话，过去 12 个月里我见过的每个上线非 Transformer LLM 的团队，选的都是 Mamba-2 混合架构。RWKV 的社区更小，工具链也更弱。如果你要做生产环境，默认选 attention；针对特定的长上下文负载可以考虑 Mamba 混合；把 RWKV 当作研究方案就好。

## 实战演练：70B 模型在 32K 上下文下的 KV cache 与 FLOPs

咱们来算笔账，看看一个 attention block 在单次解码步骤中到底干了什么。以 LLaMA-3-70B 为例，GQA-8，$h=64$，$d_{\text{head}}=128$，$L=80$，$d_{\text{model}}=8192$，词表 $V=128256$，FFN 内部维度 $d_{\text{ffn}}=28672$（SwiGLU 三重投影等效于 ~57K 的标准 FFN）。

**单层参数量。**

- Attention QKVO 投影：$d_{\text{model}} \cdot (n_{\text{heads}} \cdot d_{\text{head}} + 2 \cdot n_{\text{kv}} \cdot d_{\text{head}} + d_{\text{model}}) = 8192 \cdot (8192 + 2048 + 8192) = 152 \text{M}$
- FFN SwiGLU (gate + up + down)：$3 \cdot d_{\text{model}} \cdot d_{\text{ffn}} = 3 \cdot 8192 \cdot 28672 = 705 \text{M}$
- RMSNorm × 2：忽略不计 (~32K)
- 单层总计：~857M

乘以 80 层 = 68.6B。加上 embeddings（$V \cdot d_{\text{model}} = 1.05 \text{B}$， tied），再加 output norm，总计 ~70B。✓

**32K 上下文下的每 token KV cache。**根据上面的公式：$2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 = 327{,}680$ bytes per token at FP16，即 4 KB。32K tokens 就是 128 MB。✓

**每解码 token 的 FLOPs。**

- Attention QKV 投影：$2 \cdot d_{\text{model}} \cdot (n_{\text{heads}} + 2 n_{\text{kv}}) \cdot d_{\text{head}} = 2 \cdot 8192 \cdot 80 \cdot 128 \approx 168 \text{M}$
- Attention 计算（一个新 query 对抗 32K 缓存的 K/V）：$4 \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot 32{,}768 \approx 1.07 \text{G}$
- O 投影：$2 \cdot d_{\text{model}}^2 \approx 134 \text{M}$
- FFN：$2 \cdot 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}} \approx 1.41 \text{G}$
- 单层：~2.78 GFLOPs
- 80 层：每解码 token ~222 GFLOPs

在 H100 上，FP16 有效吞吐量约 2 TFLOPs/W，纯计算需要 ~110 ms 每 token——但解码是*内存受限*的，不是计算受限。真正的瓶颈是以 3.35 TB/s 的速度从 HBM 读取 70B 参数权重，这给出了每 token ~21 ms 的硬下限（一次解码需要触碰每个权重一次）。生产环境 Serving 通过 batching、KV 量化和 speculative decoding 来对抗这个问题（第 5 章详述）。

## 生产环境真相：前沿实验室到底交付了什么

架构论文只描述了上线模型中公开的 1%。剩下 99% 都是基建。每个前沿实验室都在做，但你不会在 model card 里看到的三件事：

**针对 attention + FFN 融合的自定义 CUDA kernels。**vLLM 和 SGLang 都发布了针对 LLaMA 家族手工调优的 kernels，把 RMSNorm + QKV 投影 + RoPE + attention + 输出投影融合到少数几个 kernel launch 中。 naive 的 PyTorch graph 每层每 token 会 launch 约 20 个 kernels；融合 kernel 只 launch 2-3 个。在 80 层和 64-batch 解码下，光是 kernel-launch 开销就能主导小模型的耗时。NVIDIA 的 TensorRT-LLM 走得更远，直接从 graph IR JIT 编译出针对特定架构的 kernel。

**FP8 attention 计算。**H100 和 B200 拥有原生 FP8 (E4M3 / E5M2) tensor cores。FlashAttention-3 [Shah et al., 2024] 在 FP8 中运行 QK matmul 并用 FP32 累加，吞吐量比 FP16 attention 翻倍。标准基准测试上的精度损失 < 0.1% perplexity；生产系统（GPT-4o, Claude-4.5）在 prefill 和 decode 阶段都使用 FP8。使用 FP8 训练（NVIDIA 的 Transformer Engine, [Micikevicius et al., 2022]）正成为新预训练运行的标准。

**逐层 LR 缩放和权重 tying。**LLaMA-3 论文 [Dubey et al., 2024] 透露，attention 输出投影和 FFN down 投影的初始化方差比其他层小，以避免训练早期不稳定。大多数生产训练脚本还会 tying  embedding 和 LM head 矩阵，减少 1-2% 的总参数量。这些 tweaks 单独看很小，但叠加起来对训练稳定性和质量差异影响明显。

事实上，"Mixtral 8x7B" 是一个模型一种架构的说法部分是个虚构。部署的 Mistral API 运行 Mixtral 时用了多种推理时 trick（paged attention、通过小 draft 模型进行 speculative decoding、FP8 KV cache），这些不属于架构本身，但确实影响质量和延迟。当你看到基准测试数字时，你看到的是部署效果，不仅仅是架构。

## 常见坑点

这五个坑我都见过别人踩。

**1. 硬编码 $h_{\text{kv}} = 1$ 假设是 MQA。**有个 7B 模型的自定义训练脚本复制了教程示例代码， assumed 是 MQA。结果质量比 baseline 差了 3-5 个 perplexity 点。解决办法是切换到 GQA-8。如果你从零开始，GQA-8 是安全的默认值。

**2. 忘记在 cache 中跨头共享 K/V。**我见过一些 FlashAttention 封装，即使对于 GQA 模型也把 KV cache 分配为 $[B, T, n_{\text{heads}}, d_{\text{head}}]$，然后在运行时对缓存的 K/V 做 `repeat_interleave`。这在 GQA-8 模型上浪费了 8 倍内存。Cache 应该是 $[B, T, n_{\text{kv}}, d_{\text{head}}]$，broadcast 应该发生在 attention kernel 内部。

**3. RMSNorm $\epsilon$ 太小。**PyTorch 默认 $\epsilon = 10^{-6}$。对于 FP16 训练这可能会 underflow。FP16 用 $\epsilon = 10^{-5}$，BF16 用 $10^{-6}$。我们曾追过一个 step-12000 发散的 bug 追了两天，结果就是因为这个。

**4. MoE router 梯度被静默清零。**实现自定义 MoE 时的常见错误：只有被选中的 expert 输出才将梯度流向 router。如果你计算了 router gate 然后在 top-$k$ 上重新 softmax，必须在梯度路径中使用*原始* logits，而不仅仅是 top-$k$ 子集。否则 router 永远学不会*不*选坏的 expert。Mixtral 和 DeepSeek-V3 的参考实现都正确做到了这一点；从头写的实现往往没有。

**5. SwiGLU FFN 内部维度未调整。**标准 Transformer FFN 内部维度是 $4 \cdot d_{\text{model}}$。 naive 地换成 SwiGLU 会使投影数量 triples，参数量爆炸 3 倍。惯例是将内部维度缩小 2/3（即 $\approx 2.67 \cdot d_{\text{model}}$），以保持 FFN 总参数量大致相等。几个 LLaMA 的开源 fork 打破了这条规则，最终参数量与预期不符。

## 2024-2026 研究前沿

当前的 "MoE Transformer with GQA" 共识之后是什么：

**Differential attention** [Ye et al., 2024] 减去两个具有不同参数的 attention maps。经验表明这能抑制 attention 噪声并改善长上下文检索。已经出现在一些 2025-2026 模型发布中。

**Attention 中的原生稀疏性。**Native Sparse Attention (NSA) 及类似工作训练模型原生地 attend 到 key 的稀疏子集，而不是事后 retrofit 稀疏性。[Yuan et al., 2025] 表明原生稀疏 attention 可以在长上下文基准上以低得多的计算和 KV 成本匹敌 dense attention。

**Linear attention 回归。**几篇 2024-2025 论文（Gated Linear Attention, RetNet [Sun et al., 2023], TransNormerLLM）通过添加 gating 和 decay 缩小了与 full attention 的质量差距。它们是否真的能在生产规模上取代 softmax attention 还有待观察，但差距是 5 年来最小的。

**Diffusion 语言模型。**[Lou et al., 2024] (SEDD) 和 [Sahoo et al., 2024] (MDLM) 表明离散 diffusion 可以在文本上匹敌 autoregressive perplexity。Mercury Coder 于 2025 年发布，声称通过 diffusion 实现 1000-token 生成的 sub-50ms 延迟，这比任何 autoregressive 模型在该长度下能达都要快。质量在前线规模上是否保持尚不清楚，但这是多年来最可信的非 autoregressive 竞争者。

**测试时计算缩放。**[Snell et al., 2024] 表明增加推理时计算（chain-of-thought, self-consistency, MCTS）可以在相同质量下替代预训练计算。o1 / DeepSeek-R1 / Claude-thinking 系列模型认真对待这一点：一个 32B 参数的 thinking 模型可以通过花费 10 倍以上的推理计算，在困难推理任务上胜过 70B 参数的 non-thinking 模型。这不是架构变化，但它改变了架构与成本的权衡。
## 什么时候该用什么架构

| 场景 | 架构 | 原因 |
|---|---|---|
| 通用聊天、代码 | Dense Transformer (LLaMA-3, Qwen3-Dense) | 单位参数质量最高，工具链成熟 |
| 成本敏感型服务 | MoE (Mixtral, DeepSeek-V3, Qwen3-MoE) | 参数量与 FLOPs 比达到 3-10× |
| 256K+ 上下文，低延迟 | Hybrid Mamba-attention (Jamba) | 每 token 内存占用恒定 |
| 边缘端推理 < 4 GB | Quantized small dense (Qwen3-1.7B INT4, Phi-4-mini) | 显存受限，MoE 帮不上忙 |
| 重推理任务（数学、代码） | Dense Transformer + thinking RL | 质量随推理算力扩展 |

选架构这事儿，很少是单纯挑个“最好”的。关键是看哪个架构的约束条件跟你的服务场景能对上。MoE 在你 GPU 管够、主要受限于 FLOPs 时胜出。单卡跑的时候显存总量是天花板，Dense 更香。上下文长度卡脖子的时候，混合架构赢面大。

## 总结与下一章

现在的 LLM 底子确实还是 Transformer，但块内部已经被重新打磨了一遍。为了稳定性（pre-norm, RMSNorm），为了质量（SwiGLU, RoPE），为了推理成本（GQA, sliding window），还有为了参数效率（MoE）。纯非注意力模型（Mamba, RWKV）通用表现差点意思，但跟注意力机制 hybrid 一下，长上下文场景就能打。

下一章我打算往下挖一层：**tokenization**。为什么 CJK token 成本比英文高 2-3 倍，BPE 在字节流上到底干了什么，chat template token 又是怎么固化成模型行为的。这层大家都爱跳过，最后往往又得回来补课。

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