---
title: "大模型工程（二）：Tokenization 深度解析"
date: 2026-03-28 09:00:00
tags:
  - LLM
  - tokenization
  - bpe
  - sentencepiece
  - chat-template
categories: 大模型工程
series: llm-engineering
series_order: 2
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "BPE、SentencePiece、WordPiece 的差别，byte-level fallback，CJK token 膨胀问题，扩词表的真实代价，以及悄悄塑造每个模型行为的 chat template 特殊 token。"
translationKey: "llm-engineering-2"
---
分词层往往被大家忽视，却是我在生产环境中调试最多的地方——静默的质量下降、异常的成本激增、模型无法正确执行指令（通常源于 chat template 格式错误）。我希望在发布多语言产品前能彻底掌握这一章的内容。

![LLM Engineering (2): Tokenization Deep Dive — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/illustration_1.png)

## 分词器到底在做什么

分词器（tokenizer）做的事很简单：把字符串映射成一串整数 ID，反过来也能把 ID 还原成字符串。这两个方向都是确定性的，但通常不是双射——`tokenizer.decode(tokenizer.encode(s))` 往返一遍可能会丢失空格、标准化 Unicode 或者合并重复标点，具体取决于算法。

主流方案主要有三类：

- **WordPiece** [Schuster & Nakajima, 2012; Wu et al., 2016] — BERT 在用。从前向后贪心最长匹配。用 `##` 标记子词 continuation。
- **BPE (Byte Pair Encoding)** [Sennrich et al., 2016] — GPT-2/3/4、 LLaMA、 Qwen、 DeepSeek 都在用。迭代合并出现频率最高的相邻 pair，直到达到词表大小。原始 BPE 算法来自 Philip Gage 1994 年关于数据压缩的论文； Sennrich 等人把它适配到了 NMT 上。
- **Unigram (SentencePiece)** [Kudo, 2018; Kudo & Richardson, 2018] — T5、 mBART、 XLM-R 在用。从一个巨大的候选词表开始，通过 EM 算法剪枝，最大化 unigram 语言模型下的序列 likelihood。

到了 2026 年，几乎所有大型生成式 LLM 都用 **byte-level BPE** [Radford et al., 2019] 或者 SentencePiece BPE 变体。"byte-level" 是关键，它让分词器对 Unicode 安全：字母表是 256 个字节，而不是 ~15 万个 Unicode 码点，所以任何编码的任何字符串都能分词，不会出现 `<UNK>`。

这里有个细节值得厘清： BPE-on-characters （Sennrich 原始 formulation）的字母表是训练数据中出现的所有 distinct Unicode 字符——多语言数据通常有几万个，长尾部分会被丢弃到 `<UNK>`。 BPE-on-bytes （GPT-2 formulation）的字母表固定为 256 个字节。任何 UTF-8 字符串都能分解成字节并成功分词。代价是，一个没能合并成 token 的汉字会消耗 3 个字节（3 个 tokens），而不是 1 个字符（1 个 token）。这就是我们后面要拆解的 CJK token bloat 的根源。

## 手动演示字节流上的 BPE

![fig1: BPE merge tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig1_bpe_tree.png)

下面用一个 tiny corpus 把算法具象化：

```python
from collections import Counter

def get_pairs(word):
    return [(word[i], word[i+1]) for i in range(len(word) - 1)]

def merge(words, pair):
    new = []
    for w in words:
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i+1]) == pair:
                out.append(w[i] + w[i+1])
                i += 2
            else:
                out.append(w[i])
                i += 1
        new.append(out)
    return new

corpus = ["lower", "lowest", "newer", "wider"]
words  = [list(w) + ["</w>"] for w in corpus]
merges = []
for _ in range(8):
    pair_counts = Counter()
    for w in words:
        pair_counts.update(get_pairs(w))
    best = pair_counts.most_common(1)[0][0]
    merges.append(best)
    words = merge(words, best)
print(merges)
# [('e', 'r'), ('er', '</w>'), ('l', 'o'), ('lo', 'w'), ...]
```

真实世界的 BPE 把 `list(w)` 换成字节序列 `w.encode("utf-8")`，语料库也大得多（trillions of tokens），但算法逻辑就是这个循环跑几十万次。

byte-level 这部分很重要： GPT-2 将 256 个字节重新映射到一个可打印的 Unicode 子集，便于人工检查，但其编码字母表仍固定为 256 个符号。这意味着*没有任何输入会分词失败*。不存在 out-of-vocabulary 失败模式。

## Tiktoken： OpenAI 的实际实现

OpenAI 的 tiktoken 库是 byte-level BPE 的生产级参考。它的实现选择和 naive BPE 不同，这些差异在大规模下很重要。

**Pre-tokenization regex。** 在 BPE 合并之前， tiktoken 用正则把文本切成 chunks：

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

这会按空格、数字（限制在 3 位以内以避免 "9999999" → 1 token 的怪异情况）、缩写和非字母数字字符串进行分割。 BPE 合并只发生在这些 chunk *内部*。这防止了跨词边界合并，否则会产生损害泛化能力的无意义多词 token。

**Rust 实现， BPE 缓存。** 热路径是 encode-by-pre-token：先用正则分割输入字符串，然后对每个 chunk 用缓存的 merge 表应用 BPE 合并。 Tiktoken 的 Rust 核心对 pre-tokens 做 hash 并缓存合并结果——重复出现的常见词（如 "the", "and"）编码复杂度为 O(1)。吞吐量在单核上英文约 ~3 MB/s， CJK 约 ~1.5 MB/s。

**词表文件格式。** Tiktoken 把 vocab 存为 base64 编码的字节串， key 是 token ID。要想 inspect `cl100k_base` 里 token 12345 到底代表什么，解码 base64 就能看到原始字节。很多 token 是不可打印的（空格串、部分 CJK 字符）； vocab 是字节级而不是字符级。

cl100k_base 分词器（GPT-3.5/4）有 100,256 个 tokens。 o200k_base 分词器（GPT-4o, GPT-4o-mini, o1）有 199,997 个。从 100K 扩展到 200K 是为了多语言覆盖——o200k 合并了更多的 CJK 字符序列，把英文的 tokens-per-word 从 1.3 降到了 0.93。

## 词表大小：成本与质量的旋钮

![fig4: vocab size vs perplexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig4_vocab_perplexity.png)

词表大小 $V$ 是预训练时选的超参数。它在 **embedding 表 FLOPs 和内存** 与 **每文档序列长度** 之间做权衡。

对于一个 $d_{\text{model}} = 4096$ 的 7B 级别模型，具体数字如下：

| Vocab | Embedding params | Tied LM head | Total embed | Token / English word |
|---|---|---|---|---|
| GPT-2 | 50,257 | tied | 206 M | 1.3 |
| LLaMA-2 | 32,000 | untied | 262 M | 1.4 |
| Qwen2.5 | 151,936 | tied | 622 M | 0.95 |
| GPT-4o | 200,000 | tied | 819 M | 0.93 |
| Qwen3 | 152,064 | tied | 623 M | 0.95 |

词表越大，每篇文档所需的 token 数越少，因此按 token 计费时推理成本更低；但 embedding 层参数和 softmax 计算开销也会增大。大多数现代模型的词表规模最终确定在 100K–200K 区间。 Qwen / GPT-4o 这个范围在多语言覆盖和避免 embedding 表主导之间找到了 sweet spot。

词表变大会有什么变化？合并操作次数增加，所以更长的常见子词（整个稀有词、常见短语）会变成单个 token。对于英文，超过 ~50K 后边际收益下降很快。对于 CJK，直到至少 200K 都还有收益。

词表大小和质量之间的实证 scaling 关系已有研究。[Tao et al., 2024] 发现对于固定训练预算，词表大小有一个最优值，且该最优值*随模型大小增长*：小模型偏好小词表（< 32K），大模型偏好大词表（> 100K）。直觉是：大模型能摊销 embedding 参数成本，并且更受益于序列长度压缩。他们推导出了一个 scaling law $V_{\text{opt}} \propto N^{0.65}$，其中 $N$ 是非 embedding 参数。对于 70B 模型，最优值在 200K 左右；对于 1B 模型，则在 32K 左右。

## Llama 3 分词器的 128K 词表决策

LLaMA-2 用的是 32K SentencePiece BPE 分词器，主要在英文上训练。 LLaMA-3 跳到了 128K （具体是 128,256），以 tiktoken 的 cl100k_base 为起点，在 Meta 的预训练 mix 上重新训练。 Llama 3 论文 [Dubey et al., 2024] 给出了理由：

- 多语言覆盖比 LLaMA-2 分词器多 4 倍
- 在整个预训练语料库上平均序列长度缩短 ~15 %
- 序列缩短节省的计算量 outweigh 了 7B 以上模型的 embedding 表成本

对于 70B 参数规模的模型， embedding 表（128K × 8192 = 1.05B 参数，权重绑定）仅占总参数量的 1.5%，影响可忽略。对于 1B 模型，同样的 embedding 会占参数的 100 %，这就是为什么小型开源模型（Phi-4-mini, Qwen3-0.5B）通常用较小的词表（50-64K）以保持 embedding 与非 embedding 的比例合理。

LLaMA-3 还在词表末尾加了 28 个保留特殊 token （`<|reserved_special_token_0|>` 到 `<|reserved_special_token_27|>`）。这些是留给未来 fine-tuning 需求的占位符——Meta 的 Llama-3-Instruct 用了其中几个作为 chat template 标记，而无需重新训练 embedding 表。这是一个值得复制的模式：在建词表时预留几百个 token 给下游专用。

## CJK token 膨胀问题

![LLM Engineering (2): Tokenization Deep Dive — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/illustration_2.png)

![fig2: CJK token bloat by language](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig2_cjk_bloat.png)

我在开发一款中文产品时，曾因这一问题遭遇严重困扰。 GPT-3.5 的分词器（cl100k_base, 100K 词表）把英文字符串 "Hello, how are you?" 分成了 6 个 tokens。中文等价物 "你好，你今天怎么样？" 分成了 17 个 tokens。

三个因素叠加：

1. **UTF-8 编码成本。** 典型汉字占 3 个 UTF-8 字节， ASCII 占 1 个。合并前，这在字节级别就已经是 3 倍的 token 劣势。
2. **合并频率。** BPE 合并频率最高的 pair。中文预训练数据历史上只占 OpenAI 语料库的一小部分，所以字符级 pair 合并得较少。
3. **无词边界。** 英文 BPE 受益于 leading-space 约定（`Ġhello`）；中文没有这种标记。

同一 prompt 在不同分词器上的成本，针对这句中文 "你好，请帮我用 Python 写一个快速排序"：

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 47 |
| cl100k_base (GPT-3.5/4) | 24 |
| o200k_base (GPT-4o) | 18 |
| LLaMA-2 (32K) | 39 |
| LLaMA-3 (128K) | 21 |
| Qwen2.5 (152K) | 14 |

如果你按 token 计费（确实如此——每个 API 都这样），这意味着**对于同样的中文 workload， Qwen2.5 比 GPT-4o 每 token 便宜 1.7 倍，这还没算每 token 单价的差异**。 Qwen3 达到了 13。对于中文重度产品，选那个在中文上训练过的分词器，而不是那个跟着 hype 模型来的。

你可以用五行代码自己测：

```python
import tiktoken
from transformers import AutoTokenizer

text = "你好，请帮我用 Python 写一个快速排序"
print("o200k:", len(tiktoken.get_encoding("o200k_base").encode(text)))
print("qwen3:", len(AutoTokenizer.from_pretrained("Qwen/Qwen3-7B").encode(text)))
print("llama3:", len(AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B").encode(text)))
```

在选模型前，用你的真实 prompt 跑一下这个。决策结果经常和 "谁在 MMLU 上最强" 不一样。
## 实战案例：算算中文产品的成本账

拿个典型的中文客服机器人场景来说：用户消息平均 80 个汉字，模型回复平均 200 个汉字，系统 Prompt 平均 600 个汉字。每天 100 万轮对话。

| Tokenizer | Tokens / convo | Daily tokens | Monthly cost @ $1/1M |
|---|---|---|---|
| cl100k_base | (600 + 80 + 200) × 24/14 ≈ 1509 | 1.51 B | $45,300 |
| o200k_base | (600 + 80 + 200) × 18/14 ≈ 1131 | 1.13 B | $33,930 |
| Qwen3 | 880 × 13/14 ≈ 817 | 0.82 B | $24,510 |

（换算系数基于上面测得的 CJK 字符 token 比率。）同样的负载，同样的 token 单价，光选个 Tokenizer，月账单就能从 4.5 万刀降到 2.4 万刀。量再大一点，一年省下来的就是七位数。

反过来逻辑也一样：如果一个产品的用户 100% 说英语，用 LLaMA-3 的 128K 词表会比 LLaMA-2 的 32K 吐出*更多* token，因为大词表能合并更多多词短语（比如 "in order to" 变成 1 个 token 而不是 4 个）。英语上的提升没中文那么夸张（5-15% vs 2-3 倍），但也是实打实的。

## 为代码定制的 Tokenizer

代码是另一种方言，有自己的 Tokenization 需求。缩进（4 空格、 8 空格、 Tab）、括号密度、标识符长度分布，跟自然文本完全不同。通用 Tokenizer 在网页文本上训练，对常见代码模式的合并效果很差。

**StarCoder** [Li et al., 2023] 用了 49,152 词的 Tokenizer，专门在 The Stack （6 TB 代码语料）上训练。对比一个典型的 Python 文件（1 KB）：

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 384 |
| cl100k_base | 287 |
| StarCoder | 218 |
| DeepSeek-Coder | 215 |

StarCoder 比 cl100k 少了 24% 的 token，因为它合并了常见模式：`def `、`    return `、`import `、`if __name__ == "__main__":`。 DeepSeek-Coder 的 Tokenizer 走得更远，直接用显式的缩进 token——4 个空格是一个 token， 8 个空格是另一个。代码预训练模型几乎都用这类专用 Tokenizer，因为成本节省会复利：代码占预训练数据的 8-15%，而且推理阶段代码负载的 token 消耗很大。

**FIM (fill-in-the-middle) tokens。** 代码补全模型需要在光标处插入，而不是只能追加。 FIM 训练 [Bavarian et al., 2022] 用了三个特殊 token——`
## 字符级和视觉 Tokenizers 怎么样？

最近有两个技术动向值得了解，不过都还没成为主流。

**纯字符级/字节级方案**：代表工作是 ByT5、 MEGABYTE、 MAMBA-byte。干脆跳过 BPE 合并步骤，直接在原始字节上训练。好处是简单，没有分词 bug，也没有中文词汇膨胀的问题。代价是序列长度：一段中文如果用 BPE 是 100 个 token，变成字节就是 600 个。就算用了亚二次方注意力的优化，成本依然很高。 MEGABYTE [Yu et al., 2023] 的做法是分层处理字节块。很有前景，但在通用规模上还没法跟主流方案竞争。

**视觉 Tokenizers**：把文本渲染成图片，再用视觉方式分词。代表工作是 Donut [Kim et al., 2022] 和 PIX2STRUCT [Lee et al., 2023]。主打的是统一文本 - 视觉建模，让你不用再纠结选哪种分词器。目前还不适合通用 LLM 负载，但在 OCR 密集的场景里很有意思。

展望 2026 年的生产环境： byte-level BPE 依然是默认选项。真正有趣的工作集中在更好的合并算法（greedy 还是 marginal-likelihood）以及针对文档的动态词表选择。

## 2024-2026 研究前沿

在 byte-level BPE 成为共识之后，接下来会发生什么：

**学习型分词（Learned tokenization）。** [Yu et al., 2024] (SpaceByte) 展示了可以把分词器和语言模型一起端到端训练，让模型自己决定子词单元。效果能匹配 BPE，而且不需要手写正则规则。虽然还没上线生产，但已经对 BPE 构成了 credible threat。

**动态词表（Dynamic vocabularies）。** 针对文档的词表适配——比如分词器知道自己在处理 Python 代码，就切换到针对代码优化的合并表。早期的工作（[Provilkov et al., 2020] BPE-dropout, [Kudo, 2018] subword regularization）探索了训练时的随机分词； 2024-2025 这一波则在探索推理时的确定性但上下文感知的分词。

**多模态 Tokenizers （Multimodal tokenizers）。** 处理文本、图像、音频和视频的模型需要统一分词方案。目前的主流做法（LLaVA, Qwen-VL, Gemini）是用独立的视觉 encoder 处理图像，然后把 vision tokens 插入文本流。下一代会不会转向所有模态统一的分词器还是个未知数。[Team, 2024] (Chameleon) 展示了早期融合的全模态分词是可行的。

## 核心建议与后续

分词这东西，平时无感，一出问题就要命：选错分词器，中文负载成本直接翻倍；跳过 chat template，指令遵循能力下降；盲目扩展词表，微调后的模型反而不如基座。所以在选模型前，务必用你的实际 prompt 测一下 token 数量。一定要通过 `apply_chat_template` 使用模型官方的 chat template。除非你有几千亿的训练 token 可以挥霍，否则别随便扩展词表。

下一章：**大规模预训练（pretraining at scale）**。数据混合、去重、那个安静的 200B token 悬崖（大多数开放数据集到这里就不再有提升了）、 FSDP 对比 ZeRO-3，以及当你把 LLaMA 风格的训练任务从 8 张卡扩展到 8000 张卡时，到底会出什么乱子。

## 参考文献

- Schuster, M., & Nakajima, K. (2012). Japanese and Korean voice search. *ICASSP*.
- Wu, Y., Schuster, M., Chen, Z., et al. (2016). Google's neural machine translation system. *[arXiv:1609.08144](https://arxiv.org/abs/1609.08144)*.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL*.
- Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. *ACL*.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer. *EMNLP demo*.
- Radford, A., Wu, J., Child, R., et al. (2019). Language models are unsupervised multitask learners (GPT-2). *OpenAI*.
- Provilkov, I., Emelianenko, D., & Voita, E. (2020). BPE-Dropout: Simple and effective subword regularization. *ACL*.
- Hewitt, J. (2021). Initializing new word embeddings for pretrained language models. *blog post*.
- Bavarian, M., Jun, H., Tezak, N., et al. (2022). Efficient training of language models to fill in the middle (FIM). *[arXiv:2207.14255](https://arxiv.org/abs/2207.14255)*.
- Xue, L., Barua, A., Constant, N., et al. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models. *TACL*.
- Kim, G., Hong, T., Yim, M., et al. (2022). OCR-free document understanding Transformer (Donut). *ECCV*.
- Scao, T., Fan, A., Akiki, C., et al. (2022). BLOOM: A 176B-parameter open-access multilingual language model. *[arXiv:2211.05100](https://arxiv.org/abs/2211.05100)*.
- Cui, Y., Yang, Z., & Yao, X. (2023). Efficient and effective text encoding for Chinese LLaMA and Alpaca. *[arXiv:2304.08177](https://arxiv.org/abs/2304.08177)*.
- Li, R., Allal, L., Zi, Y., et al. (2023). StarCoder: May the source be with you. *[arXiv:2305.06161](https://arxiv.org/abs/2305.06161)*.
- Lee, K., Joshi, M., Turc, I., et al. (2023). Pix2Struct: Screenshot parsing as pretraining for visual language understanding. *ICML*.
- Yu, L., Simig, D., Flaherty, C., et al. (2023). MEGABYTE: Predicting million-byte sequences with multiscale Transformers. *NeurIPS*.
- Faysse, M., Fernandes, P., Guerreiro, N., et al. (2024). CroissantLLM: A truly bilingual French-English language model. *[arXiv:2402.00786](https://arxiv.org/abs/2402.00786)*.
- Tao, C., Liu, Q., Dou, L., et al. (2024). Scaling laws with vocabulary: Larger models deserve larger vocabularies. *NeurIPS*.
- Wang, J., Gangavarapu, T., Yan, J., & Sabuncu, M. (2024). MambaByte: Token-free selective state space model. *COLM*.
- Yu, L., Simig, D., Wang, X., et al. (2024). SpaceByte: Towards deleting tokenization from large language modeling. *NeurIPS*.
- Team, Chameleon. (2024). Chameleon: Mixed-modal early-fusion foundation models. *[arXiv:2405.09818](https://arxiv.org/abs/2405.09818)*.
- Dubey, A., Jauhri, A., Pandey, A., et al. (2024). The Llama 3 herd of models. *[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)*.