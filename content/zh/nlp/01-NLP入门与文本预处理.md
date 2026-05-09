---
title: "自然语言处理（一）：NLP入门与文本预处理"
date: 2025-10-01 09:00:00
tags:
  - NLP
  - 深度学习
  - 文本预处理
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从第一性原理出发的 NLP 入门：梳理四个时代的脉络，亲手搭建从清洗到向量化的完整流水线，把分词、TF-IDF、n-gram 与分布式表示背后的数学讲清楚。"
disableNunjucks: true
series_order: 1
translationKey: "nlp-1"
polished_by_qwen_max: true
---
每当你向通义千问提问、让 GitHub Copilot 帮你补全代码，或者浏览 Google 翻译的结果时，其实都在使用一套凝聚了七十年技术积累的系统。自然语言处理（NLP）是一门教会机器如何阅读、评分、转换和生成人类语言的学科。令人惊讶的是，如今这套复杂的现代技术体系，其底层仍然依赖于几十年前发明的一些基础预处理方法。

作为本系列的第一篇文章，本文主要完成两个目标。首先，它会为你绘制一幅“地图”：介绍这个领域的历史渊源、当前的研究范围，以及我们今天使用的工具为何是现在的模样。其次，它会带你亲手搭建一个扎实的基础层——包括数据清洗、分词、标准化和特征提取，并提供可以直接复用的代码。读完这篇文章后，你不仅能获得一条可复用的预处理流水线，还能更清楚地理解每个步骤的实际作用，以及它们在什么情况下可能无意中破坏了有价值的信息。

![NLP 应用全景图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig1_applications_landscape.png)


<!-- wanx-hero -->
![自然语言处理（一）：NLP入门与文本预处理 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/introduction-and-preprocessing/illustration_1.png)
## 你将学到什么
<!-- wanx-mid -->
![自然语言处理（一）：NLP入门与文本预处理 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/introduction-and-preprocessing/illustration_2.png)

- 自然语言处理（NLP）的四大范式，以及每次技术变革背后的深层原因
- 分词领域的核心术语：字符、词、子词，为什么 BPE 能够脱颖而出
- 如何利用 NLTK、spaCy 和 scikit-learn 构建一条灵活可配置的文本预处理流水线
- Bag-of-Words 和 TF-IDF 的数学原理，以及如何解读生成的矩阵数据
- Zipf 定律、n-gram 语言模型，以及 one-hot 向量为何在实际应用中力不从心
- 一张实用的决策表，帮助你判断何时需要执行（或跳过）哪些预处理步骤

**前置要求**：熟练掌握 Python 编程，对 NumPy 和 pandas 有基本了解，无需任何 NLP 相关背景知识。
## 1. NLP 的四个时代

自然语言处理（NLP）的发展并非一帆风顺，而是经历了几次跳跃式突破。每一次飞跃都源于一种全新的语言表示方式。了解这段历史能帮你更精准地选择工具：在狭窄的表单填写任务中，规则系统依然碾压神经网络；搜索排序的核心依然是统计方法；而嵌入表示则几乎横扫了其他所有领域。

### 1.1 符号主义时代（1950年代 — 1980年代末）

早期的研究者把语言当作逻辑问题来解决。比如，1966 年的 ELIZA 系统通过手工编写的正则表达式匹配用户输入，并对捕获的内容进行重组输出；1970 年的 SHRDLU 则借助手写文法解析“积木世界”中的指令。这些系统在其特定领域内表现得非常精确，但一旦超出范围就完全失效——一个同义词或拼写错误就能让它们崩溃。回头看，教训显而易见：人类语言的表面形式千变万化，靠人工枚举根本无法穷尽。

### 1.2 统计革命（1990年代）

真正的转折点在于一个简单却深刻的洞见：不用再费力编写规则，直接从数据中估计概率就行。最经典的例子是 bigram 模型：

$$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$$
就是这么一个公式，撑起了 IBM 的统计机器翻译、第一代真正可用的语音识别系统，以及概率词性标注器。隐马尔可夫模型（HMM）将这套思想扩展到隐状态，概率上下文无关文法（PCFG）则进一步覆盖了句法分析。虽然特征仍然需要人工设计，但规则已经能够自动学习了。

### 1.3 深度学习时代（2013 — 2016）

2013 年，Mikolov 等人提出的 Word2Vec 展示了一个令人惊叹的现象：训练一个小型神经网络预测上下文词，得到的词向量竟然自带“算术”能力——
$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$
从此，词不再是孤立的标识符，而是住进了一个连续空间，相似度可以用余弦距离轻松计算。随后，RNN 和 LSTM 登场，让模型能够沿着序列传递上下文，终于学会了利用顺序信息，而不仅仅是依赖词袋统计。

### 1.4 Transformer 革命（2017 — 至今）

2017 年，《Attention Is All You Need》用自注意力机制彻底取代了循环结构：
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
这一改变带来了两个关键的工程优势。首先，序列各位置可以完全并行计算，训练规模取决于 GPU 内存，而不是序列长度。其次，任意两个 token 都能直接互相注意，长程依赖问题迎刃而解。BERT、GPT 以及今天的所有大模型，都是它的直系后代。

| 时代         | 时间          | 核心思想           | 被什么瓶颈打破       |
|--------------|---------------|--------------------|----------------------|
| 符号主义     | 1950 — 1980s  | 手写规则与文法      | 表面形式无法穷举      |
| 统计学习     | 1990s — 2010s | 从语料估概率        | 人工特征工程触顶      |
| 深度学习     | 2013 — 2016   | 端到端学稠密表示    | 循环结构慢、不并行    |
| Transformer  | 2017 — 现在   | 全序列自注意力      | （仍在探索中）        |

**洞察**：每次范式更替都解决了上一代的瓶颈，但并没有抛弃底层技术。今天的 LLM 用的 tokenizer 依然是统计学习的产物；你的检索系统底层很可能还挂着一个 TF-IDF 兜底。
## 2. NLP 现在用在哪儿

| 领域 | 典型应用 |
|---|---|
| **文本分类** | 情感分析、垃圾邮件检测、意图识别与路由 |
| **信息抽取** | 命名实体识别、关系抽取、知识图谱构建 |
| **生成任务** | 机器翻译、文本摘要、代码生成 |
| **对话 AI** | ChatGPT、Claude、语音助手 |
| **搜索与分析** | 语义搜索、主题建模、RAG |

上面的图表将这些应用场景归纳为六个主要类别。可以发现，几乎每个类别最终都需要用到向量——而这正是预处理的核心目标：将文本转化为向量形式。
## 3. 预处理流水线概览

在将原始文本交给模型之前，必须先将其转化为数值特征。标准的预处理流程通常包括六个步骤，每一步都是一种权衡：为了获得更规整的数据结构，我们不得不舍弃部分信息。

![文本预处理流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig2_preprocessing_pipeline.png)

```
原始文本
  -> 清洗      （去除 HTML 标签、URL、邮箱地址和无意义字符）
  -> 分词      （切分为单词或子词单元）
  -> 规范化    （转为小写、还原词形，必要时提取词干）
  -> 停用词过滤（根据需要移除 "the"、"is"、"at" 等高频无意义词）
  -> 向量化    （使用 BoW、TF-IDF 或词嵌入表示）
  -> 模型
```

一个常见的误区是盲目套用所有预处理步骤，认为越多越好。实际上，正确的思路是：每一步的目标是去掉那些下游任务无法处理的噪音，同时尽可能保留其他有用的信息。在后续的每个环节中，我们会反复探讨这种权衡的重要性。

### 3.1 环境准备

```bash
pip install nltk spacy scikit-learn matplotlib numpy pandas beautifulsoup4
python -m spacy download en_core_web_sm
```

```python
import nltk
for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)
```
## 4. 第一步——文本清洗
从网页抓取的文本通常夹杂着 HTML 标签、URL 链接，以及各种控制字符。清洗的目的是去掉这些显而易见的“噪音”，同时尽量不破坏语义。

```python
import re

def clean_text(text: str) -> str:
    """去除 HTML 标签、URL、邮箱地址和非字母字符，统一空白符。"""
    text = re.sub(r'<[^>]+>', '', text)              # 去掉 HTML 标签
    text = re.sub(r'http\S+|www\.\S+', '', text)     # 去掉 URL
    text = re.sub(r'\S+@\S+', '', text)              # 去掉邮箱
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # 只保留字母和空格
    text = re.sub(r'\s+', ' ', text).strip()         # 统一空白符
    return text

raw = """<p>Check out https://example.com for info!</p>
Contact info@test.com. Price: $29.99"""
print(clean_text(raw))
# Check out for info Contact Price
```

**激进清洗的代价**。上面的函数会顺便删掉数字和标点符号。对于主题建模任务来说，这没什么问题，因为数字通常是无关紧要的干扰信息；但如果是以下场景，这种清洗方式就不合适了：

- **情感分析**——像 `!!!` 和 `?!` 这样的标点符号往往承载着强烈的情感信号。
- **命名实体识别**——比如 "Apple Inc."，句点和大小写都是不可或缺的部分。
- **金融 NLP**——像 `$29.99` 这样的价格信息才是你真正需要提取的关键内容。

因此，清洗规则一定要根据具体任务量身定制，不能一刀切地套用通用逻辑。

**性能优化小贴士**。如果你需要处理数百万篇文档，建议提前编译正则表达式，避免重复编译带来的性能开销：

```python
HTML_RE = re.compile(r'<[^>]+>')
URL_RE  = re.compile(r'http\S+|www\.\S+')
text = URL_RE.sub('', HTML_RE.sub('', text))
```

当面对复杂的 HTML（例如残缺的标签或嵌入的脚本）时，正则表达式的局限性就会显现出来。这时，使用专门的解析器会更加可靠：

```python
from bs4 import BeautifulSoup
text = BeautifulSoup(html_text, 'html.parser').get_text(' ', strip=True)
```
## 5. 第二步——分词
分词是将文本切分成模型能够处理的最小单元的过程。选择的粒度——字符、单词还是子词——会直接影响词表的大小、序列长度，以及模型对未见过词汇的处理能力。

![同一输入的三种分词策略对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig4_tokenization_variants.png)

### 5.1 单词级分词

```python
# 简单粗暴：遇到缩写和标点就出问题
"Don't split can't".split()
# ["Don't", 'split', "can't"]

from nltk.tokenize import word_tokenize
tokens = word_tokenize("Dr. Smith earned $150,000 in 2023! Isn't that amazing?")
# ['Dr.', 'Smith', 'earned', '$', '150,000', 'in', '2023', '!',
#  'Is', "n't", 'that', 'amazing', '?']
```

NLTK 将 `Dr.` 视为一个整体 token，标点符号单独切分，而缩写 `Isn't` 被拆成 `Is` 和 `n't`。这些规则实际上是硬编码的英语语言习惯，这也正是为什么单词级分词在跨语言场景中显得非常脆弱。如果换成中文，我会更倾向于使用 jieba、THULAC 或 HanLP 这些基于词典或统计模型的工具。

### 5.2 句子级分词

```python
from nltk.tokenize import sent_tokenize
text = "Dr. Johnson works at A.I. Corp. He earned his Ph.D. in 2010."
sent_tokenize(text)
# ['Dr. Johnson works at A.I. Corp.', 'He earned his Ph.D. in 2010.']
```

NLTK 的 Punkt 模型通过数据学习哪些句号表示句子结束，哪些只是缩写的组成部分。

### 5.3 子词分词（BPE）

现代模型如 GPT、BERT、Llama 和 Claude 都不再以单词为单位进行分词，而是采用**子词分词**，几乎都基于字节对编码（Byte-Pair Encoding，BPE）。其核心逻辑非常直观：

1. 初始词表只包含单个字符。
2. 统计语料库中所有相邻字符对的出现频率。
3. 合并最高频的一对字符，生成一个新的符号。
4. 重复上述过程，直到词表达到目标规模（通常在 3 万到 10 万之间）。

```
语料: "low" x5, "lower" x2, "newest" x6, "widest" x3
初始:  l o w  /  l o w e r  /  n e w e s t  /  w i d e s t

合并 1: (e, s) -> es        # 在 "newest" 和 "widest" 中频繁出现
合并 2: (es, t) -> est
合并 3: (l, o) -> lo
...
```

为什么 BPE 在实际应用中如此重要？

- **罕见词可分解**——例如，`unbelievable` 被拆分为 `un + believ + able`，每部分都在其他地方出现过。
- **词表可控**——五万规模的子词表足以覆盖任意英文文本和大部分代码。
- **跨语言通用**——只要训练语料是多语言的，同一套分词器可以同时支持英文、法文和中文。

下面是一个可运行的最小实现：

```python
from collections import defaultdict

def get_stats(vocab):
    """统计相邻符号对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    return {w.replace(bigram, replacement): f for w, f in vocab.items()}

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

for step in range(5):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"合并 {step + 1}: {best} -> {''.join(best)}")
```

在生产环境中，直接使用 Hugging Face 的 `tokenizers` 库即可，它通过统一的 API 支持 GPT 风格的 BPE、BERT 的 WordPiece 和 SentencePiece。
## 6. 第三步——标准化
处理

标准化的目的是将同一个词的不同形式归一化为统一的表达，这样可以有效缩减词汇表规模，提升匹配效率。但需要注意的是，这种操作会丢失部分信息，因此在实际应用中需要根据具体场景权衡利弊，避免盲目使用。

### 6.1 转小写处理

```python
"Apple Inc. sells apples in APPLE stores".lower()
# "apple inc. sells apples in apple stores"
```

将文本统一转为小写对搜索和主题建模任务非常有帮助。然而，这种做法可能会对命名实体识别造成干扰。例如，公司名 `Apple` 和水果名 `apple` 在小写后就无法区分了。此外，大小写通常用于表示强调的任务（如标题或专有名词）也会受到影响。

### 6.2 词干提取与词形还原

**词干提取（stemming）** 是通过规则直接去掉词尾的一种方法，速度快但相对粗糙，有时还会产生错误结果：

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for w in ['running', 'easily', 'connection']:
    print(f"{w} -> {stemmer.stem(w)}")
# running -> run, easily -> easili, connection -> connect
```

像 `easili` 这样的结果显然不是一个合法的单词。Porter 词干提取器的设计目标是优化匹配效果，而不是保证输出的可读性。

**词形还原（lemmatization）** 则结合词典和词性标注信息，返回一个真正的词典原形：

```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("The geese were running and swimming better than the mice")
for token in doc:
    print(f"{token.text:10} -> {token.lemma_:10} ({token.pos_})")
# geese      -> goose      (NOUN)
# were       -> be         (AUX)
# running    -> run        (VERB)
# swimming   -> swim       (VERB)
# better     -> well       (ADV)
# mice       -> mouse      (NOUN)
```

| 维度       | 词干提取          | 词形还原          |
|------------|-------------------|-------------------|
| 速度       | 微秒级            | 毫秒级（需词性标注） |
| 输出结果   | 可能不是合法单词  | 必定是词典原形    |
| 准确率     | 较低              | 较高              |
| 适用场景   | 搜索 / 信息检索   | 自然语言理解 / 问答 |

我的建议是：除非你正在处理对延迟要求极高的高吞吐量检索系统，否则默认选择词形还原更为稳妥。
## 7. 第四步——停用词与 Zipf 定律
停用词是指那些高频但语义信息较少的词汇，比如英文中的 `the`、`is`、`at`，或者中文里的“的”“了”“是”。这些词虽然常见，但在具体任务中往往贡献不大。去掉它们后，词汇表规模可以减少大约三分之一，同时让信息更加集中在真正有意义的内容词上。

为什么少数几个词会占据主导地位？这背后的原因是 Zipf 定律：在自然语言中，一个词的出现频率大致与其排名成反比。
$$f(\text{rank}) \propto \frac{1}{\text{rank}}$$
![Zipf 分布：头部由停用词主导，尾部是大量的低频词](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig5_zipf_distribution.png)

排名前十的词，通常就能占到整个语料中 25% 到 30% 的 token。这就是分布的“头部”，主要由停用词构成。而“长尾”部分——那些成千上万只出现一两次的词——才是语义最丰富的区域，但也是模型最难处理的地方。正因如此，子词分词方法才显得尤为重要。

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
text = "The quick brown fox jumps over the lazy dog"
filtered = [w for w in word_tokenize(text.lower()) if w not in stop_words]
# ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

那么，什么时候应该去掉停用词呢？

- **去**——适用于词袋模型、主题建模、搜索倒排索引等场景。
- **不去**——情感分析（例如 `not good` 和 `good` 意思完全不同）、问答系统（虚词承载提问语气）、以及任何能够自己学习 token 权重的深度学习模型。
## 8. 第五步——从 Token 到向量
模型需要数字来工作。两个经典方法——词袋（Bag-of-Words）和 TF-IDF——依然是大多数检索系统的核心，也是任何新任务的基准。

### 8.1 One-hot 表示 vs 分布式表示

在讲 BoW 之前，先看看为什么最简单的编码方式会失败。One-hot 编码给每个词分配一个唯一索引，对应位置是 1，其他全是 0。任意两个词的向量都是正交的，这意味着这种编码完全无法表达词与词之间的相似性。

![One-hot 丢掉了语义；学到的嵌入把它捡了回来](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig6_onehot_vs_distributed.png)

分布式表示——我们会在第二篇里训练——把意义压缩到稠密向量中，相关词彼此靠近。BoW 和 TF-IDF 是中间状态：每个词仍然独占一个维度，但填的是频率，而不是简单的标记。

### 8.2 词袋模型

把每篇文档表示成词频向量，忽略顺序：

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

docs = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love deep learning and machine learning",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()))
```

```
   amazing  and  deep  is  learning  love  machine
0        0    0     0   0         1     1        1
1        1    0     0   1         1     0        1
2        0    1     1   0         2     1        1
```

致命缺陷：`dog bites man` 和 `man bites dog` 的向量一模一样。词袋模型完全丢掉了顺序。

### 8.3 TF-IDF

TF-IDF 给那些“在本文档中频繁、在整个语料中罕见”的词加权——这是个简单但有效的启发式规则：“对当前文档重要，但不是通用词”。
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$$$\text{IDF}(t) = \log\!\frac{1 + N}{1 + \text{df}(t)} + 1$$
其中 $N$ 是文档总数，$\text{df}(t)$ 是包含词 $t$ 的文档数。`+1` 是平滑项，确保某个词在所有文档中都出现（或都不出现）时 IDF 仍有定义。

![同一组玩具语料下，词袋计数与 TF-IDF 加权的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig3_bow_vs_tfidf.png)

上图并列展示了两个矩阵。可以看到，`learning` 出现在每篇文档中，TF-IDF 把它压低了；而像 `vision` 这种只出现在某一篇中的词，TF-IDF 把它抬高了。这正是搜索排序需要的行为。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

docs = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning",
    "Computer vision uses deep learning techniques",
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

for i, doc in enumerate(docs):
    top = df.iloc[i].sort_values(ascending=False).head(3)
    print(f"文档 {i + 1}: {dict(top.round(3))}")
```

**生产级 TF-IDF**。默认参数在真实语料上几乎没法用，至少调这几个参数：

```python
tfidf = TfidfVectorizer(
    max_features=5_000,    # 限制词表大小
    min_df=2,              # 去掉只出现一次的词
    max_df=0.8,            # 去掉出现在 80% 以上文档中的词（相当于停用词）
    ngram_range=(1, 2),    # 一元 + 二元，捕捉短语
    sublinear_tf=True,     # 对 TF 取对数，抑制重复
    stop_words='english',
)
```
## 9. 第六步——n-gram 语言模型
在完成分词（tokenization）之后，下一步可以尝试建模这些词之间的前后关系。n-gram 模型的核心思想是将一个句子分解为一系列条件概率的连乘形式：
$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

简单来说，bigram 模型只考虑前一个词的上下文，trigram 则会看前两个词，依此类推。

![n-gram 的滑动窗口、bigram 公式，以及困惑度与稀疏性之间的权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP入门与文本预处理/fig7_ngram_language_models.png)

这里有一个非常明显的权衡点：

- **更大的 n** 能够捕捉到更多的上下文信息，从而降低困惑度（perplexity）。困惑度可以粗略理解为模型预测时的“平均分支数”，数值越低说明模型的表现越好。
- 然而，**更大的 n** 也会导致参数量急剧膨胀，同时面临数据稀疏的问题。假设词表大小为 $V$，那么一个 trigram 模型最多可能需要 $V^3$ 个参数，但其中大部分参数对应的上下文组合在训练数据中根本不会出现。这就是统计 NLP 中经典的**稀疏性问题**，也是该领域长期以来的核心痛点。

为了缓解这个问题，研究者提出了各种平滑技术（如 Laplace 平滑和 Kneser-Ney 平滑），通过重新分配概率质量，为那些未见过的 n-gram 分配一定的概率值。而现代神经语言模型则完全绕开了这一难题——它们通过嵌入（embeddings）的方式让不同上下文共享参数，这种方法不仅高效，还为后续的深度学习方法铺平了道路，这也是我们将在第二部分深入探讨的内容。
## 10. 一个可复用的预处理类

将上述步骤整合成一个可以直接嵌入项目的实用工具：

```python
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """灵活配置的英文文本预处理流水线。"""

    def __init__(self, use_lemmatization: bool = True,
                 remove_stopwords: bool = True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords

        if use_lemmatization:
            # 加载 spaCy 的小型英语模型，禁用解析器和命名实体识别以提高效率
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        else:
            # 使用 NLTK 的 Porter 词干提取器
            self.stemmer = PorterStemmer()

        if remove_stopwords:
            # 获取英文停用词集合
            self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        # 转为小写
        text = text.lower()
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除 URL 链接
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # 移除电子邮件地址
        text = re.sub(r'\S+@\S+', '', text)
        # 移除非字母字符
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 合并多余空格并去除首尾空白
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_and_normalize(self, text: str) -> list[str]:
        if self.use_lemmatization:
            # 使用 spaCy 进行词形还原
            doc = self.nlp(text)
            tokens = [t.lemma_ for t in doc if not t.is_space]
        else:
            # 使用 NLTK 进行词干提取
            tokens = [self.stemmer.stem(t) for t in word_tokenize(text)]

        if self.remove_stopwords:
            # 移除停用词
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def preprocess(self, text: str) -> str:
        # 清洗、分词并归一化后重新拼接为字符串
        return ' '.join(self.tokenize_and_normalize(self.clean(text)))

    def preprocess_corpus(self, texts: list[str]) -> list[str]:
        # 对文本集合中的每条文本进行预处理
        return [self.preprocess(t) for t in texts]

# 初始化预处理器，启用词形还原并移除停用词
pre = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

# 示例文本集合
texts = [
    "Natural Language Processing (NLP) is amazing! Visit https://example.com",
    "Machine learning models are trained on large datasets.",
    "Deep learning has revolutionized computer vision and NLP.",
]

# 打印原始文本与处理后的结果对比
for orig, proc in zip(texts, pre.preprocess_corpus(texts)):
    print(f"原文：  {orig}")
    print(f"处理后：{proc}\n")
```
## 11. 端到端示例：一个极简的垃圾邮件分类器
将各个模块整合起来，构建一个可用的分类器。如果想进行真实场景的实验，建议使用 SMS Spam Collection 数据集，或者从 Kaggle 上选择任意一个垃圾邮件数据集。下面的代码特意写得非常短小，目的是确保它能在任何环境中运行。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

texts = [
    "Congratulations! You've won a $1000 gift card. Call now!",
    "Hey, are we still meeting for dinner tonight?",
    "URGENT: Your account will be closed. Click here immediately!",
    "Can you send me the project report by EOD?",
    "Get rich quick! Amazing investment opportunity!",
    "Don't forget to pick up milk on your way home",
    "You have been selected for a free cruise. Reply YES",
    "Meeting moved to 3pm tomorrow in conference room B",
    "Lose 20 pounds in 2 weeks with this miracle pill!",
    "Thanks for your help with the presentation yesterday",
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=垃圾邮件, 0=正常邮件

# 保留停用词——像 "free"、"now"、"you" 这样的词往往是垃圾邮件的重要特征
pre = TextPreprocessor(use_lemmatization=True, remove_stopwords=False)
processed = pre.preprocess_corpus(texts)

vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
X = vectorizer.fit_transform(processed)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

new_msgs = ["Can you review my code?", "FREE MONEY!!! Click now!!!"]
new_vecs = vectorizer.transform(pre.preprocess_corpus(new_msgs))
for msg, pred in zip(new_msgs, model.predict(new_vecs)):
    print(f"[{'垃圾邮件' if pred else '正常邮件'}] {msg}")
```

这个例子的重点并不是在区区十条样本上的准确率，而是整个处理流程的设计思路。如果你换成 5000 条短信数据，同样的代码几乎不用额外调整，就能轻松达到约 97% 的准确率。这正是经典 NLP 技术栈的魅力所在：代码简单明了，逻辑清晰易懂，即使没有 GPU 的加持，也很难被轻易超越。
## 12. 决策表：不同任务需要哪些预处理步骤
| 任务 | 分词方式 | 标准化处理 | 停用词处理 | 特征表示 |
|---|---|---|---|---|
| 搜索 / 信息检索 | 词级别 | 提取词干 | 移除 | TF-IDF |
| 情感分析 | 词级别 / 子词级别 | 还原词形 | **保留** | TF-IDF 或词嵌入 |
| 主题建模 | 词级别 | 还原词形 | 移除 | BoW 或 TF-IDF |
| 机器翻译 | 子词（BPE） | 尽量少做 | 保留 | 词嵌入 |
| 命名实体识别 | 词级别 | 不做处理 | 保留 | 词嵌入 + 上下文信息 |
| 现代大模型 | 子词（BPE） | 不做处理 | 保留 | 学习到的嵌入 |

**经验总结**。模型的能力越强，数据量越大，预处理的工作就越应该简化。深度学习模型能够自己学习标准化规则，过于激进的预处理反而会抹掉一些对模型有用的信息。传统机器学习方法依赖精心设计的特征工程，而像 LLM 这样的现代大模型则更倾向于直接使用原始文本进行训练。
## 核心要点
- 预处理方法因任务而异。比如，搜索引擎通常需要激进的文本规范化，而神经网络模型更倾向于使用原始文本。
- 子词分词技术（如 BPE、WordPiece、SentencePiece）已成为当前的标准选择，因为它既能限制词表规模，又能有效应对未登录词问题。
- TF-IDF 仍然是一个可靠的基准方法。如果一个简单的 TF-IDF 加逻辑回归模型都能超越你的复杂模型，那说明你的模型可能存在问题。
- 齐普夫定律（Zipf's law）揭示了为什么移除停用词对传统模型有益，同时也解释了为什么处理低频词会如此困难。
- 简单即高效。过度预处理可能会削弱表示学习的效果，因此务必通过实验来验证每一步的必要性。