---
title: "自然语言处理（一）：NLP入门与文本预处理"
date: 2025-08-15 09:00:00
tags:
  - NLP
  - 深度学习
  - 文本预处理
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 1
  total: 12
lang: zh-CN
mathjax: true
description: "从第一性原理出发的 NLP 入门：梳理四个时代的脉络，亲手搭建从清洗到向量化的完整流水线，把分词、TF-IDF、n-gram 与分布式表示背后的数学讲清楚。"
disableNunjucks: true
series_order: 1
---

每次你用通义千问问问题、让 GitHub Copilot 补全一行代码，或者打开 Google 翻译——你都在调用一套花了七十年才搭起来的技术栈。自然语言处理（NLP）研究的就是怎么让机器读、评分、改写和生成人类语言。有意思的是，现代这套体系底层很大一部分，仍然依赖于几十年前发明的那一小撮预处理工具。

本系列的第一篇做两件事。第一，画地图：这门学科从哪儿来、今天覆盖什么、为什么工具长成现在这副样子。第二，把最底层那一层——清洗、分词、标准化、特征提取——亲手搭起来，代码可以直接拿去用。读完之后你会有一条可复用的预处理流水线，更重要的是，你会知道每一步什么时候该用、什么时候反而是在悄悄毁掉信号。

![NLP 应用全景图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig1_applications_landscape.png)

## 你将学到什么

- NLP 的四个范式，以及每一次替换背后的技术原因
- 把"分词"这件事说清楚：字符级、词级、子词级，以及 BPE 为什么赢
- 用 NLTK、spaCy、scikit-learn 搭一条可配置的预处理流水线
- BoW 和 TF-IDF 背后的数学，以及怎么读懂这两个矩阵
- Zipf 定律、n-gram 语言模型，以及 one-hot 表示为什么不够用
- 一张速查表：什么任务该做哪些预处理、什么任务该跳过

**前置要求**：能熟练写 Python，对 NumPy / pandas 有基本印象，不需要任何 NLP 背景。

---

## 1. NLP 的四个时代

NLP 不是平稳地往前走的，它是跳着走的，每一跳都源于一种新的语言表示方法。把这条线理清楚，挑工具的时候就有了直觉：规则系统在窄领域的填表场景里仍然吊打神经网络；统计方法在搜索排序里依然是主力；嵌入表示则统治了其余几乎所有场景。

### 1.1 符号主义时代（1950s — 1980s 末）

早期系统把语言当成逻辑题。1966 年的 ELIZA 用手写正则匹配用户输入再把捕获组重新拼回去；1970 年的 SHRDLU 用一份手写文法解析"积木世界"里的指令。这些系统在自己的小天地里精度很高，一旦走出去就立刻崩溃——换个同义词，打个错字，规则就失效了。事后总结，教训是：人类语言的表面形式太多，没有谁能靠枚举写完。

### 1.2 统计革命（1990s）

转折点是一个朴素但锋利的发现：你不用写规则，让机器从数据里估概率就行。最经典的就是 bigram 模型：

$$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$$

仅这一个公式就支撑起了 IBM 的统计机器翻译、第一代真正能用的语音识别、以及概率词性标注。隐马尔可夫模型（HMM）把同样的思想推广到隐状态，概率上下文无关文法（PCFG）把它推广到句法。特征还得人工设计，但规则是学出来的。

### 1.3 深度学习时代（2013 — 2016）

2013 年 Mikolov 等人提出的 Word2Vec 展示了一个惊人的现象：训练一个小型神经网络去预测上下文词，得到的词向量竟然带有"做算术"的能力——

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

词从此不再是孤立的 ID，而是住在一片连续空间里，相似度一个余弦距离就能算。RNN、LSTM 紧接着登场，让模型能沿着序列把上下文穿起来，终于学到了顺序，而不只是一袋词的统计。

### 1.4 Transformer 革命（2017 — 至今）

2017 年的《Attention Is All You Need》用自注意力替换掉了循环结构：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

带来两个工程上至关重要的后果。第一，模型在序列各位置上是完全并行的，训练规模随 GPU 内存而非序列长度扩展。第二，任意两个 token 都能直接互相注意，长程依赖问题被一次性解决。BERT、GPT 以及今天的所有大模型，都是它的直系后裔。

| 时代 | 时间 | 核心思想 | 被什么瓶颈打破 |
|---|---|---|---|
| 符号主义 | 1950 — 1980s | 手写规则与文法 | 表面形式无法穷举 |
| 统计学习 | 1990s — 2010s | 从语料估概率 | 人工特征工程触顶 |
| 深度学习 | 2013 — 2016 | 端到端学稠密表示 | 循环结构慢、不并行 |
| Transformer | 2017 — 现在 | 全序列自注意力 | （仍在探索中） |

**洞察**：每一次范式更替都解决了上一代的瓶颈，但并没有把上一层扔掉。今天 LLM 用的分词器仍是统计学习的产物；你的检索系统底层很可能仍然挂着一个 TF-IDF 兜底。

---

## 2. NLP 现在用在哪儿

| 领域 | 典型应用 |
|---|---|
| **文本分类** | 情感分析、垃圾邮件、意图路由 |
| **信息抽取** | 命名实体、关系抽取、知识图谱 |
| **生成** | 翻译、摘要、代码生成 |
| **对话 AI** | ChatGPT、通义千问、智能客服 |
| **搜索与分析** | 语义搜索、主题建模、RAG |

上面那张图把这些应用归成了六个簇。注意几乎所有簇最终都要消费一个向量——而把文本变成向量，正是预处理流水线的工作。

---

## 3. 预处理流水线总览

进任何模型之前，原始文本得先变成数值特征。标准流水线分六步，每一步都是一次"用信息换规整"的取舍。

![文本预处理流水线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig2_preprocessing_pipeline.png)

```
原始文本
  -> 清洗      （去掉 HTML、URL、邮箱、垃圾字符）
  -> 分词      （切成词或子词）
  -> 标准化    （小写、词形还原，必要时词干提取）
  -> 停用词    （酌情去掉 "the"、"is"、"at"）
  -> 向量化    （BoW、TF-IDF 或词嵌入）
  -> 模型
```

最常见的错误是反射性地把所有步骤都做一遍。正确的思路是：每一步只去掉下游处理不了的噪音，其他信息一律保留。后面每一步我们都会回到这条原则。

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

---

## 4. 第一步——文本清洗

网页上抓回来的文本往往裹着 HTML、塞满 URL、夹着各种控制字符。清洗就是把这些显而易见的噪音抹掉，同时不动语义。

```python
import re

def clean_text(text: str) -> str:
    """去 HTML、URL、邮箱、非字母字符；折叠空白。"""
    text = re.sub(r'<[^>]+>', '', text)              # HTML 标签
    text = re.sub(r'http\S+|www\.\S+', '', text)     # URL
    text = re.sub(r'\S+@\S+', '', text)              # 邮箱
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # 只留字母和空格
    text = re.sub(r'\s+', ' ', text).strip()         # 折叠空白
    return text

raw = """<p>Check out https://example.com for info!</p>
Contact info@test.com. Price: $29.99"""
print(clean_text(raw))
# Check out for info Contact Price
```

**激进清洗的代价**。上面这段函数顺手把数字和标点也删了。做主题建模的时候这没问题，因为数字往往是噪音；但对下面这些任务就是错的：

- **情感分析**——`!!!` 和 `?!` 本身就是情感信号。
- **命名实体识别**——"Apple Inc." 离不开句点和大写。
- **金融 NLP**——`$29.99` 本身就是你要的内容。

所以不要用一套通用清洗器走天下，规则要贴着任务定。

**性能小贴士**。要处理几百万篇文档时，把正则编译一次再复用：

```python
HTML_RE = re.compile(r'<[^>]+>')
URL_RE  = re.compile(r'http\S+|www\.\S+')
text = URL_RE.sub('', HTML_RE.sub('', text))
```

碰到野生 HTML（残缺标签、内嵌脚本）时，正则就力不从心了，得请出真正的解析器：

```python
from bs4 import BeautifulSoup
text = BeautifulSoup(html_text, 'html.parser').get_text(' ', strip=True)
```

---

## 5. 第二步——分词

分词把文本切成模型最终看到的那种最小单元。切的粒度——字符、词、子词——决定了词表大小、序列长度，以及模型遇到没见过的词时优雅程度。

![同一句输入下三种分词策略对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig4_tokenization_variants.png)

### 5.1 词级分词

```python
# 最朴素的做法——遇到缩写和标点就出错
"Don't split can't".split()
# ["Don't", 'split', "can't"]

from nltk.tokenize import word_tokenize
tokens = word_tokenize("Dr. Smith earned $150,000 in 2023! Isn't that amazing?")
# ['Dr.', 'Smith', 'earned', '$', '150,000', 'in', '2023', '!',
#  'Is', "n't", 'that', 'amazing', '?']
```

NLTK 把 `Dr.` 当成一个 token，把标点单独切出来，把缩写 `Isn't` 拆成 `Is` + `n't`。这每一个判断都是写死的英文惯例——所以词级分词跨语言时很脆。中文里你会换上 jieba、THULAC、HanLP 这类按词典+模型来切分的工具。

### 5.2 句子分词

```python
from nltk.tokenize import sent_tokenize
text = "Dr. Johnson works at A.I. Corp. He earned his Ph.D. in 2010."
sent_tokenize(text)
# ['Dr. Johnson works at A.I. Corp.', 'He earned his Ph.D. in 2010.']
```

NLTK 的 Punkt 模型是从数据里学的：哪些句号是真句末，哪些只是缩写。

### 5.3 子词分词（BPE）

GPT、BERT、Llama、Claude 这些现代模型都不按"词"来切，而是用**子词分词**，几乎都是字节对编码（Byte-Pair Encoding，BPE）的某个变体。算法逻辑很短：

1. 初始词表只包含单个字符。
2. 在语料里数所有相邻字符对的出现次数。
3. 把最高频的那一对合并成新符号。
4. 重复，直到词表达到目标大小（常见是 3 万到 10 万）。

```
语料: "low" x5, "lower" x2, "newest" x6, "widest" x3
初始:  l o w  /  l o w e r  /  n e w e s t  /  w i d e s t

合并 1: (e, s) -> es        # 在 "newest"、"widest" 里都频繁
合并 2: (es, t) -> est
合并 3: (l, o) -> lo
...
```

为什么 BPE 在工程上重要：

- **罕见词能被拆开**——`unbelievable` 切成 `un + believ + able`，每一块都在别处见过。
- **词表有上限**——五万规模的子词表足以覆盖任意英文文本和大部分代码。
- **跨语言迁移**——只要训练语料是多语言的，同一套分词器可以同时处理英文、法文、中文。

下面是一个能跑通的最小实现：

```python
from collections import defaultdict

def get_stats(vocab):
    """统计相邻符号对的频率。"""
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

生产环境直接用 Hugging Face 的 `tokenizers` 库，一套 API 同时覆盖 GPT 风格的 BPE、BERT 的 WordPiece 和 SentencePiece。

---

## 6. 第三步——标准化

标准化把同一个词的各种表面写法归并成一个，词表更小，匹配更准。代价是丢信息，所以要主动权衡，别一上来就全套加上。

### 6.1 小写化

```python
"Apple Inc. sells apples in APPLE stores".lower()
# "apple inc. sells apples in apple stores"
```

小写化对搜索和主题建模有帮助。但它会伤命名实体识别（公司 `Apple` 和水果 `apple` 揉到一起），也会伤任何把大小写当强调信号的任务。

### 6.2 词干提取 vs 词形还原

**词干提取（stemming）** 用确定性规则砍后缀。快、糙、有时是错的：

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for w in ['running', 'easily', 'connection']:
    print(f"{w} -> {stemmer.stem(w)}")
# running -> run, easily -> easili, connection -> connect
```

`easili` 不是个真单词——Porter 词干提取只在乎能不能匹配，不在乎结果好不好看。

**词形还原（lemmatization）** 借助词典加词性信息，返回真正的词典原形：

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

| 维度 | 词干提取 | 词形还原 |
|---|---|---|
| 速度 | 微秒级 | 毫秒级（含词性标注） |
| 输出 | 可能不是真词 | 一定是词典原形 |
| 准确率 | 较低 | 较高 |
| 适用场景 | 搜索 / 信息检索 | 自然语言理解 / 问答 |

一个比较实用的默认值：默认用词形还原，除非你在跑高吞吐的检索系统，延迟预算非常紧。

---

## 7. 第四步——停用词与 Zipf 定律

停用词是一类高频但语义少的封闭类词，比如英文的 `the`、`is`、`at`，中文的"的"、"了"、"是"。去掉它们，词表能小三分之一，信号也就更集中在内容词上。

为什么少数几个词能霸榜？因为 Zipf 定律：在自然语言语料里，一个词的频率大致和它的排名成反比。

$$f(\text{rank}) \propto \frac{1}{\text{rank}}$$

![Zipf 分布：头部由停用词主导，尾部是大量的低频词](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig5_zipf_distribution.png)

光是排名前十的那几个词，就能占到整个语料 25% — 30% 的 token。这就是分布的"头部"，绝大多数是停用词。"长尾"那一头——成千上万只出现一两次的词——才是语义最浓的地方，但也是模型最吃力的地方，正因为如此子词分词才显得格外有用。

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
text = "The quick brown fox jumps over the lazy dog"
filtered = [w for w in word_tokenize(text.lower()) if w not in stop_words]
# ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

什么时候去停用词：

- **去**——词袋模型、主题建模、搜索倒排索引。
- **不去**——情感分析（`not good` 和 `good` 差着十万八千里）、问答（虚词承担提问语气）、任何能自学习权重的深度模型。

---

## 8. 第五步——把 token 变成向量

模型只认数字。两个经典编码——词袋（Bag-of-Words）和 TF-IDF——至今仍是大部分检索系统的根基，也是任何新任务都该先跑一遍的基线。

### 8.1 One-hot 表示 vs 分布式表示

在讲 BoW 之前，先看一眼为什么最朴素的编码不够用。One-hot 给每个词分配一个唯一的索引，对应位置打 1，其余全是 0。任何两个词的向量都是正交的——也就是说，编码本身根本不带相似度信息。

![One-hot 丢掉了语义；学到的嵌入把它捡了回来](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig6_onehot_vs_distributed.png)

分布式表示——我们会在第二篇里亲手训出来——把含义压进一个稠密向量里，相关的词彼此靠近。BoW 和 TF-IDF 介于两者之间：每个词仍然占一个维度，但维度上填的不再是 0/1 而是频率。

### 8.2 词袋模型

把每篇文档表示成词频向量，词序完全丢掉：

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

致命局限：`dog bites man` 和 `man bites dog` 的向量是一样的。词袋把语序整个丢了。

### 8.3 TF-IDF

TF-IDF 给那种"在本文档里频繁、在整个语料里罕见"的词加权——这是一个朴素但好用的"对当前文档重要、对所有文档不通用"的启发式：

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

$$\text{IDF}(t) = \log\!\frac{1 + N}{1 + \text{df}(t)} + 1$$

其中 $N$ 是文档总数，$\text{df}(t)$ 是包含词 $t$ 的文档数。`+1` 是平滑项，保证某个词在所有文档里都出现（或都不出现）时 IDF 仍有定义。

![同一组玩具语料下，词袋计数与 TF-IDF 加权的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig3_bow_vs_tfidf.png)

上图把两个矩阵并排放着。可以看到：`learning` 在每篇文档里都出现，TF-IDF 把它压低了；而 `vision` 只在某一篇里出现，TF-IDF 反而把它抬上来。这正是搜索排序里你想要的那种行为。

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

**生产级 TF-IDF**。默认参数在真实语料上几乎活不下来，至少调这几个旋钮：

```python
tfidf = TfidfVectorizer(
    max_features=5_000,    # 词表上限
    min_df=2,              # 只出现一次的丢掉
    max_df=0.8,            # 超过 80% 文档出现的当成事实上的停用词
    ngram_range=(1, 2),    # 一元 + 二元，能捕到短语
    sublinear_tf=True,     # 对 TF 取对数，抑制重复
    stop_words='english',
)
```

---

## 9. 第六步——n-gram 语言模型

切完 token 之后，你还可以建模"它们怎么互相跟随"。n-gram 模型把一句话拆成一串条件概率的乘积：

$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

bigram 用一个词做上下文，trigram 用两个，依此类推。

![n-gram 滑窗、bigram 公式，以及困惑度与稀疏性的取舍](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/01-NLP%E5%85%A5%E9%97%A8%E4%B8%8E%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86/fig7_ngram_language_models.png)

这里的取舍很尖锐：

- **n 越大**，上下文越多，困惑度（perplexity，可以粗略理解成模型有效"分支数"，越低越好）就越低。
- **n 越大**，参数量也炸，绝大多数上下文又遇不到。$V$ 是词表大小，trigram 模型最多有 $V^3$ 个参数，其中绝大多数训练时一次都没见过。这就是**稀疏性问题**，是统计 NLP 时代的核心痛点。

平滑技术（Laplace 平滑、Kneser-Ney 平滑）通过把概率质量重新分一些给"没见过的 n-gram"来打补丁。现代神经语言模型干脆绕开这个问题——通过嵌入把上下文之间的参数共享起来，这就是通往第二篇的桥。

---

## 10. 一个可复用的预处理类

把上面这些步骤拼成一个能直接拖进项目的小工具：

```python
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """可配置的英文文本预处理流水线。"""

    def __init__(self, use_lemmatization: bool = True,
                 remove_stopwords: bool = True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords

        if use_lemmatization:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        else:
            self.stemmer = PorterStemmer()

        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_and_normalize(self, text: str) -> list[str]:
        if self.use_lemmatization:
            doc = self.nlp(text)
            tokens = [t.lemma_ for t in doc if not t.is_space]
        else:
            tokens = [self.stemmer.stem(t) for t in word_tokenize(text)]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def preprocess(self, text: str) -> str:
        return ' '.join(self.tokenize_and_normalize(self.clean(text)))

    def preprocess_corpus(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]


pre = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
texts = [
    "Natural Language Processing (NLP) is amazing! Visit https://example.com",
    "Machine learning models are trained on large datasets.",
    "Deep learning has revolutionized computer vision and NLP.",
]
for orig, proc in zip(texts, pre.preprocess_corpus(texts)):
    print(f"原文：  {orig}")
    print(f"处理后：{proc}\n")
```

---

## 11. 端到端示例：一个最小的垃圾邮件分类器

把所有部件串起来。真要做实验，建议用 SMS Spam Collection 或 Kaggle 上任意一个垃圾邮件数据集；下面这段写得很小，目的是哪儿都能跑。

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
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=垃圾, 0=正常

# 不去停用词——"free"、"now"、"you" 本身就是垃圾邮件信号
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

这例子的重点不是十条样本上的准确率——而是流水线的形状。把数据换成 5000 条 SMS，同样这套代码能跑到大约 97% 的准确率，几乎不需要再调参。这就是经典 NLP 栈的力量：短、透明，并且在没有 GPU 的情况下出奇地难被打败。

---

## 12. 速查表：什么任务做哪些预处理

| 任务 | 分词 | 标准化 | 停用词 | 特征 |
|---|---|---|---|---|
| 搜索 / 信息检索 | 词级 | 词干提取 | 去除 | TF-IDF |
| 情感分析 | 词级 / 子词 | 词形还原 | **保留** | TF-IDF 或词嵌入 |
| 主题建模 | 词级 | 词形还原 | 去除 | BoW 或 TF-IDF |
| 机器翻译 | 子词（BPE） | 最少处理 | 保留 | 词嵌入 |
| 命名实体识别 | 词级 | 不做 | 保留 | 词嵌入 + 上下文 |
| 现代大模型 | 子词（BPE） | 不做 | 保留 | 学到的嵌入 |

**经验法则**。模型容量越大、数据越多，预处理就该做得越少。深度模型会自己学标准化，过度预处理会把它本来能用的信号毁掉；经典 ML 反而靠精心的特征工程吃饭；LLM 偏爱原始文本。

---

## 核心要点

- 预处理是任务相关的：搜索要狠的标准化，神经网络要原始文本。
- 子词分词（BPE、WordPiece、SentencePiece）已经是现代默认值，因为它把词表卡死并且能处理没见过的词。
- TF-IDF 仍然是该有的基线。如果 TF-IDF + 逻辑回归基线能打过你的花哨模型，那花哨模型一定哪里坏了。
- Zipf 定律解释了为什么停用词去除对经典模型有用，也解释了为什么长尾词难学。
- 少即是多。过度预处理对学表示的模型有害，永远要靠实验衡量。

---

## 延伸阅读

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) —— Jurafsky & Martin 的免费在线教材，本领域的标准参考
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— Transformer 原始论文
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— Sennrich 等人，把 BPE 引入 NLP 的论文
- [spaCy 语言学特征文档](https://spacy.io/usage/linguistic-features)
- [scikit-learn 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

## 系列导航

| 部分 | 主题 | 链接 |
|---|---|---|
| **1** | **NLP 入门与文本预处理（本文）** | |
| 2 | 词向量与语言模型 | [下一篇 -->](/zh/自然语言处理-二-词向量与语言模型/) |
| 3 | RNN 与序列建模 | [阅读 -->](/zh/自然语言处理-三-RNN与序列建模/) |
| 4 | 注意力机制与 Transformer | [阅读 -->](/zh/自然语言处理-四-注意力机制与Transformer/) |
| 5 | BERT 与预训练模型 | [阅读 -->](/zh/自然语言处理-五-BERT与预训练模型/) |
| 6 | GPT 与生成式语言模型 | [阅读 -->](/zh/自然语言处理-六-GPT与生成式语言模型/) |
