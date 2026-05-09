---
title: "大模型工程（二）：Tokenization 深挖"
date: 2026-04-27 09:00:00
tags:
  - llm
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
Tokenization 是人人都会跳过的部分，也是我在线上环境调试最多的一层。质量悄悄下降、成本突然飙升、模型不听指令，全是因为有人搞错了 chat template。这些问题我都踩过坑，血泪经验告诉我，这些细节太重要了。

这篇内容，是我在上线多语言产品前希望自己早就明白的所有东西。

![大模型工程（二）：Tokenization 深挖 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/illustration_1.png)
## Tokenizer 到底在做什么

Tokenizer 把字符串转成整数 ID 列表，也能反向还原。两个方向都确定，但通常不是一一对应。跑一圈 `tokenizer.decode(tokenizer.encode(s))`，可能丢空格、规范 Unicode 或合并重复标点，具体看算法。

目前主要有三种方法：

- **WordPiece** [Schuster & Nakajima, 2012; Wu et al., 2016]——BERT 用的就是它。从前向后贪心匹配最长子串，子词续接部分用 `##` 标记。
- **BPE（Byte Pair Encoding）** [Sennrich et al., 2016]——GPT-2/3/4、LLaMA、Qwen 和 DeepSeek 都用它。核心是不断合并最高频的相邻字符对，直到词表大小达标。最初来自 Philip Gage 1994 年的数据压缩论文，后来被 Sennrich 等人改造用于神经机器翻译（NMT）。
- **Unigram（SentencePiece）** [Kudo, 2018; Kudo & Richardson, 2018]——T5、mBART 和 XLM-R 用它。先生成超大候选词表，再用 EM 算法根据 unigram 模型剪枝，最大化序列似然。

到 2026 年，几乎所有大型生成式 LLM 都会用 **byte-level BPE** [Radford et al., 2019] 或 SentencePiece 的 BPE 变体。"byte-level" 是关键：字母表只有 256 个字节，而不是约 15 万 Unicode 码位。任何编码的字符串都能分词，完全不用 `<UNK>`。

这里有个重要区别：BPE-on-characters（Sennrich 原版）的字母表是训练数据中的所有 Unicode 字符，通常是几万个，尾部低频字符丢弃并标记为 `<UNK>`。BPE-on-bytes（GPT-2 版本）固定为 256 字节，就这么简单。任何 UTF-8 字符串都能分解成字节并成功分词。代价是，单个未合并的汉字占 3 字节（3 token），而不是 1 字符（1 token）。这就是后面要分析的 CJK token 膨胀问题的根源。
## 手动计算 BPE on bytes

![fig1: BPE 合并树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig1_bpe_tree.png)

用一个小语料库演示 BPE 算法的过程：

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

实际应用中，BPE 不会直接用 `list(w)`，而是将字符串转为字节序列 `w.encode("utf-8")`。语料库规模通常达到万亿 token，但核心算法还是这个循环，只是需要运行几十万次。

byte-level 的设计很关键。GPT-2 把 256 个字节重新映射到可打印的 Unicode 子集，方便人类查看。编码表仍然是 256 个符号。这样一来，任何输入都能成功分词，完全避免了 OOV 问题。
## Tiktoken：OpenAI 的实际做法

OpenAI 的 tiktoken 库是 byte-level BPE 的生产级实现。它的设计和朴素 BPE 方法有显著区别，这些差异在大规模场景下尤为重要。

**预分词正则表达式**  
BPE 合并前，tiktoken 用一个正则表达式将文本切分成小块。正则如下：

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

这个正则按空格、数字（最多 3 位，避免 "9999999" 被当成单个 token）、缩写、非字母数字字符切分。BPE 合并只在小块内部进行。这样能避免跨单词边界合并，防止生成无意义的多词 token，影响模型泛化能力。

**Rust 实现与 BPE 缓存**  
性能关键路径是按预分词编码。先用正则切分输入字符串，再对每个小块用缓存的合并表做 BPE 合并。tiktoken 的 Rust 核心会对预分词内容哈希，并缓存合并结果。高频词（如 "the"、"and"）编码速度可达 O(1)。单核处理英文约 3 MB/s，CJK 文本约 1.5 MB/s。

**词表文件格式**  
tiktoken 将词表存储为 base64 编码的字节串，以 token ID 为键。想查看 `cl100k_base` 中 token 12345 是什么，解码 base64 就能看到原始字节。很多 token 不可打印（如空白字符或部分 CJK 字符），因为词表基于字节，而非字符。

cl100k_base tokenizer（用于 GPT-3.5/4）包含 100,256 个 token。o200k_base tokenizer（用于 GPT-4o、GPT-4o-mini、o1）包含 199,997 个 token。从 10 万扩展到 20 万，主要驱动力是多语言支持。o200k 合并了更多 CJK 字符序列，将英文的 token-per-word 比例从 1.3 降到 0.93。
## 词表大小：成本与质量的调节旋钮

![fig4: 词表大小 vs perplexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig4_vocab_perplexity.png)

词表大小 $V$ 是预训练时定下的超参数。它用来平衡嵌入表的计算量、内存占用和每篇文档的序列长度。

以一个 7B 级模型为例，假设 $d_{\text{model}} = 4096$，数据如下：

| 词表 | Embedding 参数 | LM head tied | 总嵌入 | 每英文词 token |
|---|---|---|---|---|
| GPT-2 | 50,257 | tied | 206 M | 1.3 |
| LLaMA-2 | 32,000 | untied | 262 M | 1.4 |
| Qwen2.5 | 151,936 | tied | 622 M | 0.95 |
| GPT-4o | 200,000 | tied | 819 M | 0.93 |
| Qwen3 | 152,064 | tied | 623 M | 0.95 |

词表越大，文档需要的 token 越少。按 token 计费时，推理成本更低。但嵌入层和 softmax 的规模会变大。现代模型的词表通常在 100K 到 200K 之间。Qwen 和 GPT-4o 的词表范围兼顾了多语言支持和嵌入表占比。

词表增大时，合并操作次数增加。更长的常见子词（如低频词或常用短语）会被分配到单个 token。对英语来说，词表超过 50K 后边际收益下降很快。对 CJK 语言，即使到 200K 仍有明显收益。

关于词表大小和模型质量的关系，[Tao et al., 2024] 做了研究。他们发现，在固定训练预算下，词表大小有一个最优值。这个值随模型规模增大而增大。小模型适合小词表（< 32K），大模型适合大词表（> 100K）。原因很简单：大模型能分摊嵌入参数的成本，并从序列长度压缩中获益更多。他们推导出一个公式：$V_{\text{opt}} \propto N^{0.65}$，其中 $N$ 是非嵌入参数的数量。对于 70B 的模型，最优词表约为 200K；对于 1B 的模型，约为 32K。
## Llama 3 选择 128K 词表的原因

LLaMA-2 用的是 32K 的 SentencePiece BPE tokenizer，主要训练数据是英文。到了 LLaMA-3，直接跳到 128K（具体是 128,256）。它以 tiktoken 的 cl100k_base 为基础，在 Meta 的预训练数据混合集上重新训练。Llama 3 的论文 [Dubey et al., 2024] 提了几个关键理由。

多语言覆盖范围比 LLaMA-2 扩大了 4 倍。预训练语料中，序列长度平均缩短了 15%。在 7B 参数以上的模型里，序列变短省下的计算量，超过了嵌入表增加的成本。

70B 参数的模型中，嵌入表（128K × 8192 = 1.05B 参数，tied）只占总参数的 1.5%，几乎可以忽略。但对 1B 参数的模型来说，同样的嵌入表会占到总参数的 100%。小型开源模型（比如 Phi-4-mini、Qwen3-0.5B）通常用更小的词表（50-64K），保持嵌入和非嵌入参数比例合理。

LLaMA-3 还在词表末尾加了 28 个保留的特殊 token（`<|reserved_special_token_0|>` 到 `<|reserved_special_token_27|>`）。这些是为未来微调预留的占位符。Meta 的 Llama-3-Instruct 就用其中几个作为聊天模板标记，完全不用重新训练嵌入表。这种做法值得借鉴：构建词表时预留几百个 token，方便后续特定任务使用。
## CJK Token 膨胀问题

![大模型工程（二）：分词器，决定你 API 账单一半的隐形选择 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/illustration_2.png)


![fig2: 各语言 CJK token 膨胀对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig2_cjk_bloat.png)

这个问题让我在中文产品上踩了大坑。GPT-3.5 的 tokenizer（cl100k_base，10 万词表）把英文句子 "Hello, how are you?" 分成 6 个 token。换成中文 "你好，你今天怎么样？"，直接飙到 17 个。

原因很简单：

1. **UTF-8 编码开销大。** 普通汉字占 3 字节 UTF-8，ASCII 只占 1 字节。还没开始合并，字节级别就差了 3 倍。
2. **合并频率低。** BPE 算法优先合并高频字符对。但中文在 OpenAI 的训练语料中占比小，字符级别的合并自然少。
3. **没有词边界。** 英文 BPE 得益于前导空格标记（比如 `Ġhello`），中文完全没有这种优势。

同一句话 "你好，请帮我用 Python 写一个快速排序"，不同 tokenizer 的分词结果如下：

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 47 |
| cl100k_base (GPT-3.5/4) | 24 |
| o200k_base (GPT-4o) | 18 |
| LLaMA-2 (32K) | 39 |
| LLaMA-3 (128K) | 21 |
| Qwen2.5 (152K) | 14 |

按 token 计费的话（所有 API 都这么算），同样的中文任务，Qwen2.5 的 token 数量比 GPT-4o 少 1.7 倍，还不算单价差异。Qwen3 更是降到 13。如果你的产品以中文为主，选 tokenizer 时别盲目跟风热门模型，要看它是不是在中文数据上训练过的。

自己动手测一下很简单，5 行代码搞定：

```python
import tiktoken
from transformers import AutoTokenizer

text = "你好，请帮我用 Python 写一个快速排序"
print("o200k:", len(tiktoken.get_encoding("o200k_base").encode(text)))
print("qwen3:", len(AutoTokenizer.from_pretrained("Qwen/Qwen3-7B").encode(text)))
print("llama3:", len(AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B").encode(text)))
```

选模型之前，先用你的实际 prompt 测试一遍。结果往往和 "MMLU 谁更强" 不一样。
## 实际案例：一个中文产品的成本计算

看一个典型的中文客服机器人负载。用户消息平均 80 字，模型回复平均 200 字，系统提示词平均 600 字。每天处理 100 万次对话。

| Tokenizer | 单次对话 token 数 | 每日 token 总数 | 每月成本 @ $1/1M |
|---|---|---|---|
| cl100k_base | (600 + 80 + 200) × 24/14 ≈ 1509 | 1.51 B | $45,300 |
| o200k_base | (600 + 80 + 200) × 18/14 ≈ 1131 | 1.13 B | $33,930 |
| Qwen3 | 880 × 13/14 ≈ 817 | 0.82 B | $24,510 |

换算因子基于前面测得的每个 CJK 字符对应的 token 比例。同样的负载，同样的单 token 价格，仅 tokenizer 不同，每月成本就能从 $45K 降到 $24K。规模更大时，一年能省下七位数。

反过来也成立。一个完全面向英文用户的产品，LLaMA-3 的 128K 词表比 LLaMA-2 的 32K 词表生成更多 token。更大的词表合并了更多多词短语，比如 "in order to" 从 4 个 token 变成 1 个。对英文来说，提升幅度较小（5%-15%），远不如中文（2-3 倍）。但优化是实打实的。
## 代码专用 tokenizer

代码是一种独特的“方言”，分词需求和普通文本完全不同。缩进用 4 个空格、8 个空格还是 Tab，括号密度，标识符长度分布，这些都和自然语言差别很大。通用 tokenizer 通常在网页文本上训练，对代码常见模式的处理不够好。

**StarCoder** [Li et al., 2023] 的 tokenizer 专门针对代码训练，词汇表大小为 49,152。它基于 The Stack（一个 6 TB 的代码语料库）训练。以下是对一个典型 Python 文件（1 KB）的分词结果对比：

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 384 |
| cl100k_base | 287 |
| StarCoder | 218 |
| DeepSeek-Coder | 215 |

StarCoder 比 cl100k_base 少用了 24% 的 token。原因很简单，它合并了常见的代码模式，比如 `def `、`    return `、`import ` 和 `if __name__ == "__main__":`。DeepSeek-Coder 更进一步，为缩进引入了显式 token——4 个空格是一个 token，8 个空格是另一个。几乎所有代码预训练模型都会用这种专用 tokenizer。代码占预训练数据的 8%-15%，推理时对 token 的消耗又特别大，节省开销的效果会不断累积。

**FIM（填空式补全）token**

代码补全模型需要在光标位置插入内容，而不是简单追加。FIM 训练 [Bavarian et al., 2022] 引入了三个特殊 token——`
## 扩展词表：通常不建议

![fig3: tokenizer 对比表](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig3_tokenizer_table.png)

很多人觉得：“LLaMA-3 微调中文，tokenizer 不行，加 5 万中文 token 就好了。” 这种想法基本是错的。

给预训练模型加 token 会带来三个问题：

1. embedding 表要新增行，随机初始化或用子词嵌入平均值。
2. LM head 也要新增行（如果 untied），同样是随机的。
3. 模型需要通过微调学会用这些新 token。

新嵌入要赶上原有 token 的水平，需要多得多的数据。几篇论文（[Cui et al., 2023] Chinese-LLaMA、[Faysse et al., 2024] CroissantLLM）提到，扩展词表后需要 500 亿到 2000 亿 token 的持续预训练才能融合。数据不够的话，分词效率高了，但生成质量反而更差。

实际点的做法是选一个 tokenizer 从一开始就针对目标语言训练过的模型。中文用 Qwen3，东南亚语言用 Sea-LION 或 Sailor，非洲语言用 BLOOM [Scao et al., 2022]。只有找不到合适模型时才考虑扩词表。

如果实在躲不开（比如给基础模型加医学术语），可以试试这几招：

**用子词分解均值初始化新嵌入。** 比如 "tachycardia"，用现有 BPE 分片（"tachy"、"card"、"ia"）的嵌入均值初始化。这样起点更合理。[Hewitt, 2021] 研究表明，这种方法比随机初始化好 3-5 个困惑度点，只需少量训练。

**warmup 阶段冻结其他部分。** 先只训练新嵌入几百步，再解冻整个模型。避免其他部分为了补偿随机嵌入而破坏已有知识。

**嵌入层也用 LoRA。** PEFT 支持对 `embed_tokens` 应用 LoRA。结合 warmup 技巧，数据成本低，效果却不错。
## 不用分词器的方法：ByT5、MEGABYTE、MAMBA-byte

两条值得关注但还没火起来的技术路线。

**ByT5** [Xue et al., 2022] 直接在 UTF-8 字节上训练 T5，完全不用分词器。词表只有 256 个字节加 3 个特殊标记。序列长度比 mT5 长得多，但 ByT5-large 在多语言基准测试中表现相当。代价是序列变长了。一段中文如果用 BPE 分成 100 个 token，换成字节就是 600 个。训练时每步速度慢，但每个字节的样本效率更高。推理时生成回复也更慢。ByT5 在多语言 NER 和噪声文本任务中找到了位置，这些任务里分词器不匹配对质量影响很大。

**MEGABYTE** [Yu et al., 2023] 用分层方法处理长字节序列。一个叫 "patch" 的模型先把字节分成大小为 8 到 16 的组，每组跑一个小 Transformer。再用一个 "global" Transformer 处理这些组的表示。这种方法把复杂度从单个字节降到 patch 级别，接近 O(n²)。这让字节级处理在长度达到 1M 的序列时也能用。

**MAMBA-byte** [Wang et al., 2024] 把 Mamba 状态空间架构直接用在字节流上。线性时间递推让字节级训练在大规模下变得可行。据报道，MAMBA-byte 350M 在 ARC-Easy 和 PIQA 上的表现和用 BPE 分词的基线模型相当。但在需要精确到词级推理的任务（比如 MMLU 和 HumanEval）上表现会差一些。

到 2026 年，byte-level BPE 还是主流选择。有意思的研究方向是更好的合并算法（贪心算法 vs 边际似然）以及按文档动态选择词表。
## Chat template 是 tokenizer 的状态

![fig5: chat template token 边界](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/02-tokenization/fig5_chat_template.png)

Chat template 是一段 Jinja 模板，用来把消息列表转成可分词的字符串。以 LLaMA-3 为例：

```jinja
{% for m in messages %}
<|start_header_id|>{{ m.role }}<|end_header_id|>

{{ m.content }}<|eot_id|>
{% endfor %}
```

`<|start_header_id|>`、`<|end_header_id|>`、`<|eot_id|>` 和 `<|begin_of_text|>` 都是特殊 token。每个只占一个 ID。它们记录在 `added_tokens.json` 文件中，也是模型训练时用来识别对话边界的标记。

生产环境里我踩过两个坑。

**坑一：直接拼接普通字符串当 prompt。** 有人写成这样：

```python
prompt = f"User: {user_msg}\nAssistant:"
out = model.generate(tokenizer.encode(prompt))
```

代码能跑，但效果差。模型没见过这种格式，指令遵循和安全性都会崩。原因是缺少对话标记。一定要用 `tokenizer.apply_chat_template(messages)` 构建 prompt。

**坑二：渲染模板时设置 `add_special_tokens=True`。** 这样会重复添加 BOS 标记。模型看到 `<|begin_of_text|><|begin_of_text|>` 就懵了。正确做法是用 `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`，它会自动判断是否需要加特殊标记。

Chat template 还决定了 function calling 能不能用。Qwen3 的模板会在 JSON 工具调用周围加上 `
## 常见踩坑

我在 tokenization 上踩过的 5 个大坑，分享给大家。

**1. 训练和推理用不同的 tokenizer。** 训练时用 HuggingFace 的 `tokenizer.json`，部署时为了“快”换了 `tiktoken`。BPE 合并规则看似一样，但特殊 token 处理可能不同。结果是模型生成了特殊 token，推理层没去掉；或者推理层请求了模型没见过的 token。记住，训练和推理必须用 *完全相同的* tokenizer 文件。

**2. 输入空格规范化问题。** 一些基于 SentencePiece 的 tokenizer 会把前导空格替换成 `▁`（U+2581）。用 `tokenizer.decode(tokenizer.encode(s))` 转一圈，结果可能多出或少掉空格，具体看实现。链式 pipeline 中尤其麻烦：模型 A 的输出传给模型 B，空格偏移会导致质量下降。

**3. 把 token 当成字符。** 设置 `max_tokens=200` 时，很多人以为是限制“200 个字符”。实际上，Qwen3 处理中文时，200 token ≈ 200-260 字；处理英文时，200 token ≈ 800-1200 字。长度预算要按模型的 token 算，别按字符或字节。

**4. 在特殊 token 中间截断。** 把 100K token 的 prompt 截短到 32K，如果按字节截断，可能把 `<|im_end|>`
## 生产环境中的现实：serving 栈里的 tokenizer

在生产环境中，tokenizer 是每个请求的起点和终点。关键路径很简单：

1. 收到请求 → tokenize prompt → 把 token ID 交给调度器  
2. 生成下一个 token ID  
3. detokenize 成文本 → 流式返回给客户端  

实际实现受三个约束影响。

**Tokenization 吞吐量**  
vLLM 和 SGLang 都用 Rust 写的 tokenizer（HuggingFace 的 `tokenizers` 库）。相比 Python 的 `transformers` AutoTokenizer，性能提升约 10 倍。假设每秒 1000 次请求，prompt 包含 500 个 token，Python tokenize 会占满一个 CPU 核心。Rust 则轻松搞定，只用一小部分资源。

**流式 detokenize**  
每次生成一个 token 并 detokenize 时，需要缓冲。BPE token 可能是不完整的 UTF-8 序列，尤其是 CJK 字符。vLLM 的 detokenizer 维护一个近期 token 缓冲区，尝试解码后，只输出到上一个完整 UTF-8 边界。这会让流式输出延迟增加 1 到 2 个 token。

**Tokenizer 缓存**  
很多 serving 系统会对固定不变的系统 prompt 缓存 tokenize 结果。Anthropic 的 prompt caching 功能就是例子——每个请求的前 1000 个 token 预先 tokenize，相同前缀的请求直接复用。结合 KV-cache 前缀共享技术，聊天应用中长系统 prompt 的首 token 生成时间可以减少 50% 到 70%。

下一节会聊我在生产环境踩过的 tokenizer 坑。
## 字符级和视觉 tokenizer 呢？

最近有两个方向值得关注，虽然还没火起来。

**字符级 / 纯字节级**：比如 ByT5、MEGABYTE 和 MAMBA-byte。这些方法直接跳过 BPE 合并步骤，在原始字节流上训练。好处很明显：简单，没有分词错误，也不会因为 CJK 字符膨胀。但问题也不少。序列长度会显著增加，一段 100 个 BPE token 的中文段落，换成字节级就是 600 字节。即使用了亚二次复杂度的注意力机制，计算开销依然很大。MEGABYTE [Yu et al., 2023] 提出分层处理字节块的办法，缓解了这个问题。潜力不错，但目前还打不过主流方法。

**视觉 tokenizer**：把文本渲染成图像，再按图像分词。代表工作有 Donut [Kim et al., 2022] 和 PIX2STRUCT [Lee et al., 2023]。它们统一了文本和视觉建模，摆脱了对特定 tokenizer 的依赖。目前不适合通用大语言模型的工作负载，但在 OCR 领域很有吸引力。

展望 2026 年，byte-level BPE 还是生产环境的默认选择。真正有趣的研究集中在改进合并算法（比如贪心算法 vs 边际似然算法）以及针对每个文档动态选择词汇表。
## 研究前沿 2024-2026

byte-level BPE 共识之后，接下来会有什么新方向？

**学习型分词。** [Yu et al., 2024]（SpaceByte）展示了如何端到端训练分词器和语言模型。模型自己决定子词单元，效果不输 BPE，还不用手动设计正则表达式。虽然还没真正在生产环境跑过，但已经对 BPE 构成了威胁。

**动态词表。** 分词器可以根据文档类型调整词汇表。比如处理 Python 代码时，切换到针对代码优化的合并表。早期工作（[Provilkov et al., 2020] BPE-dropout、[Kudo, 2018] subword regularization）在训练阶段探索了随机分词。2024-2025 年的研究重点转向推理阶段的上下文感知分词方法，确定性更强。

**多模态分词器。** 处理文本、图像、音频和视频的模型需要统一的分词方案。目前主流方法（LLaVA、Qwen-VL、Gemini）用独立视觉编码器分词图像，再将视觉 token 插入文本流。下一代是否采用全模态统一的分词器仍是未知数。不过，[Team, 2024]（Chameleon）证明了早融合的全模态分词方法可行。
## 小结与下一篇

Tokenization 平时看不见，但出问题就很头疼。选错 tokenizer，CJK 工作负载成本翻倍。跳过 chat template，指令跟随能力下降。盲目扩展词表，微调模型可能还不如基础版。挑模型前，先在实际 prompt 上统计 token 数。用 `apply_chat_template` 调用官方 chat template。没有几千亿 token 的训练预算，别动词表。

下一篇聊**大规模预训练**。数据混合、去重、200B token 的隐形悬崖，公开数据集到这里基本失效。FSDP 和 ZeRO-3 怎么选？从 8 张 GPU 扩到 8000 张，LLaMA 风格训练会踩哪些坑？接着看吧。
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
