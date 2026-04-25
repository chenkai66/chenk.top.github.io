---
title: "自然语言处理（五）：BERT与预训练模型"
date: 2025-08-31 09:00:00
tags:
  - NLP
  - BERT
  - 深度学习
  - 迁移学习
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 5
  total: 12
lang: zh-CN
mathjax: true
description: "BERT 如何让双向预训练成为 NLP 的默认范式：从架构到 80/10/10 掩码规则，到微调技巧，再到 RoBERTa/ALBERT/ELECTRA 全家桶，并附完整 HuggingFace 代码。"
disableNunjucks: true
series_order: 5
---

2018 年 10 月，Google 发布 BERT，一口气在 11 个 NLP 基准上刷新了纪录。配方却出奇地简单：拿一个 Transformer 编码器，让它根据左右两侧的上下文去预测被随机遮盖的词；预训练完成之后，再把同一个模型针对各种下游任务做一次轻量的微调。在 BERT 之前，每个任务都要从头训练一个模型；在 BERT 之后，"先预训练，再微调"成了整个领域的默认思路。

如果你在过去几年用过情感分析 API、能理解意图的搜索引擎、或是稍微靠谱一点的客服机器人，背后大概率就是 BERT 或它的某个后代在干活。

## 这一篇你会学到

- 预训练范式的演进：Word2Vec → ELMo → GPT-1 → BERT
- BERT 的架构：以 WordPiece 输入的双向 Transformer 编码器
- 掩码语言建模（MLM）与下一句预测（NSP），以及为什么是 80/10/10 这个比例
- 把 BERT 微调到分类、NER、问答、句对任务上的具体做法
- BERT 家族：RoBERTa、ALBERT、ELECTRA 各自的取舍
- 真实可用的微调配方（学习率、warmup、梯度累积）
- 一份可以直接 copy-paste 的完整 HuggingFace 流水线

**前置知识**：第 4 部分（Transformer 架构）以及基本的 PyTorch 使用经验。

---

## 预训练-微调范式是怎么火起来的

在 BERT 之前，每个 NLP 任务都从一个随机初始化的模型开始，用自己那点标注数据从头训练。这件事昂贵（算力）、浪费（任务之间不共享知识），而且很脆（数据少模型就抖）。整个领域是怎么从这个泥潭里爬出来的，可以串起 4 个里程碑式的工作。

### 一段简短的演进史

**Word2Vec（2013）**：从无标注文本里学静态词向量。问题在于，"bank"在 *river bank* 和 *bank account* 里得到的向量完全一样——上下文没法改变一个词的含义。

**ELMo（2018 年初）**：用双向 LSTM 产生上下文相关的向量，最后把每一层的隐藏状态加权求和：

$$
\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \, h_{k,j}
$$

其中 $h_{k,j}$ 是第 $j$ 层在位置 $k$ 的隐藏状态，$s_j$ 是可学习的 softmax 权重。ELMo 用结果证明了"上下文相关的预训练表示"对几乎所有下游任务都有显著提升——但它仍然是 RNN，训练慢、难以并行。

**GPT-1（2018 年 6 月）**：第一次把 Transformer 用于规模化预训练，目标函数是从左到右的标准语言模型：

$$
P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})
$$

GPT-1 已经很强，但它是单向的：读到 "the bank is closed" 时，模型在 "bank" 这个位置上还没看到 "closed"，就没法用 "closed" 来消除"银行 / 河岸"的歧义。

**BERT（2018 年 10 月）**：突破点是把预训练目标改造成允许每个位置同时看到双向上下文。这一个改动直接带来了横扫所有任务的提升。

### 为什么这个范式重要

![预训练一次，微调到各种任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig4_finetune_pipeline.png)

整个流水线分两步：

1. **预训练**：在海量无标注文本（书、维基百科、整个互联网）上用自监督任务训练。这一步贵，但只做一次。
2. **微调**：在每个下游任务上加一个小小的"任务头"，用很小的学习率端到端微调几轮。

收益是看得见的：

- **数据效率高**：底座已经"懂"语法和不少常识，每个任务只需几百到几千条标注就能跑出像样的结果。
- **通用**：同一个底座既能做分类，也能做标注、抽取、句对任务。
- **基线极强**：朴素的 BERT 微调常常就能超过过去一整套定制化架构。

---

## BERT 的架构

BERT 就是原始 Transformer 的**编码器**那一半，堆 12 或 24 层。没有解码器、没有 causal mask、不做自回归生成——它就是一摞双向自注意力层，把 token 序列变成一串"懂上下文"的向量。

![BERT 双向编码器与输入嵌入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig1_bert_architecture.png)

### 输入表示：三个嵌入相加

每个 token 的输入向量，是三个学到的同维度嵌入相加：

$$
\text{Input}_i = E^{\text{tok}}_{w_i} + E^{\text{seg}}_{s_i} + E^{\text{pos}}_{i}
$$

- **Token 嵌入**：WordPiece 子词的 id，词表大小 30K。
- **Segment 嵌入**：第一句用 $E_A$，第二句用 $E_B$。这让 BERT 不改架构就能处理 NLI、QA 这类句对任务。
- **Position 嵌入**：从 0 到 511 每个绝对位置一个**可学习**向量。（与原始 Transformer 的正余弦位置不同，BERT 自己学。）

两个特殊 token 是整套协议的关键：

- `[CLS]` 放在每个输入开头。所有层走完之后，它的隐藏状态被当作整段序列的"池化摘要"，喂给分类头。
- `[SEP]` 用来分隔句子 A 和 B，也用来标记输入结束。

### 双向自注意力

每一层编码器内部，多头自注意力让每个 token 都能看到所有其他 token：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

关键在于：$Q$、$K$、$V$ 都来自同一段输入（自注意力），并且**没有 causal mask**（双向）。所以位置 3 上的 "bank"，在一次前向中就能同时整合左边的 "river" 和右边的 "is closed"。

### 两种规格

原文给了两个一直沿用至今的配置：

|             | BERT-Base | BERT-Large |
| ----------- | --------- | ---------- |
| 层数        | 12        | 24         |
| 隐藏维度    | 768       | 1024       |
| 注意力头数  | 12        | 16         |
| 参数量      | 1.1 亿    | 3.4 亿     |

BERT-Base 在一张消费级 GPU 上就能推理。BERT-Large 是 2018 年那一批 SOTA 纪录的主力。

---

## 预训练目标

BERT 的预训练把两个自监督任务叠在一起。第一个出名，第二个后来被证明可有可无。

### 掩码语言建模（MLM）

![MLM 的 80/10/10 损坏规则](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig2_mlm_corruption.png)

对每条输入序列，随机选 15% 的位置。被选中的位置，按下面的概率分别处理：

- **80%** 的概率换成 `[MASK]`；
- **10%** 的概率换成词表里**随机**的一个 token；
- **10%** 的概率**保持原样**。

模型要做的事，是在每个被选中的位置上把**原来的**词预测回来：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid \tilde{x})
$$

其中 $\mathcal{M}$ 是被选中的位置集合，$\tilde{x}$ 是被损坏后的输入。

**为什么是 80/10/10？** 这个比例不是随便定的，它专门用来避开两个失败模式：

- 如果只用 `[MASK]`，模型在微调时永远见不到 `[MASK]`（下游真实输入不会有），训练-推理就出现了分布失配。
- 如果只用随机 token，模型会变得不敢相信任何输入，反而过度依赖远处信息。
- 留 10% 不变，是逼模型即使看到一个"看上去没问题"的词，也不要偷懒直接把它复制出去——必须真的用上下文判断。

MLM 是让 BERT"干净地"做到双向的关键：要从左右两侧预测被遮的词，编码器就必须在每个位置都融合整段上下文。

### 下一句预测（NSP）

![NSP：正例 vs 负例](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig3_nsp.png)

加 NSP 的初衷，是让 BERT 学到句间关系，方便迁移到 NLI、QA。每条训练样本把两个句子打包成 `[CLS] A [SEP] B [SEP]`，标签靠抛硬币决定：

- 50% 的样本里，B 真的是语料中紧跟 A 的下一句（标签 `IsNext`）；
- 50% 的样本里，B 是从其他文档随机抽来的句子（标签 `NotNext`）。

最后一层的 `[CLS]` 向量过一个 Linear+softmax 头，预测这个标签：

$$
P(\text{IsNext}) = \text{softmax}(W \, h_{\text{[CLS]}} + b)
$$

预训练总损失就是 MLM 损失加 NSP 损失。

> 这里有一段后来被打脸的历史：RoBERTa、ALBERT 等后续工作发现 NSP 帮助微乎其微，去掉甚至替换掉它反而**更好**。后面变体那一节会再聊。

### 预训练语料

BERT 的训练数据是 **BooksCorpus**（约 8 亿词）+ **英文维基百科**（约 25 亿词），合计约 33 亿词。从 2026 年的视角看这点量可以忽略不计——现代 LLM 动辄上万亿 token——但在当年已经足够把基线拉到一个新高度。

---

## WordPiece 分词

![WordPiece 子词分词](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig6_wordpiece.png)

BERT 不直接处理整词，用的是 **WordPiece** 子词方案。这是在两个极端之间找到的折中：

- 整词词表想覆盖真实语料就得几百万词，而且推理时永远会碰到 OOV；
- 纯字符词表很小，但模型每次都得从字符重新拼出"词义"，效率太低。

WordPiece 的做法，是贪心地合并那些"合并后能最大化训练语料似然"的字符对，最终留下大约 30K 个 token 作为词表。分词时，把每个词切成词表里能匹配上的最长片段；位于词内部的片段统一加 `##` 前缀作为"延续"标记：

```
playing       -> play  ##ing
unbelievable  -> un    ##bel  ##iev  ##able
transformer   -> transform  ##er
Tokyo2024     -> tokyo  ##20  ##24
```

这样既保证不会出现 OOV（任何东西都能拆到已知子词，最坏退化到单字符），又保留了常用词作为单 token 的效率。

---

## 把 BERT 微调到下游任务

微调的核心思想是：**底座几乎不变，换个头就能换任务**。

### 文本分类

整句级标签（情感、垃圾邮件、意图识别）的做法是：把输入过一遍 BERT，取最终的 `[CLS]` 向量，再过一个线性层。

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2,
)

text = "I love this movie!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
print(f"预测: {prediction.item()}")  # 0 或 1
```

底层发生的事：

1. 分词，前后加上 `[CLS]` 和 `[SEP]`；
2. 过 12 层 Transformer 编码器；
3. 取 `[CLS]` 的隐藏状态（BERT-Base 是 768 维）；
4. 过一个 768 → num_labels 的线性层。

### 命名实体识别（NER）

token 级任务用每个位置的向量，而不是 `[CLS]`：

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=9,  # 例如 PER/ORG/LOC/MISC 的 BIO 标签 + O
)

text = "Barack Obama was born in Hawaii"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: 标签 {pred.item()}")
```

一个细节：WordPiece 会把词切碎（"Hawaii" 也许整词不切，"Tokyo2024" 会被切成 3 个片段）。在把预测还原成"词级实体"时，惯用的做法是**只取每个词第一个子词的预测**，忽略后面的 `##` 续接片段。

### 抽取式问答

SQuAD 风格的 QA 是从上下文中预测答案 span 的起止位置：

```python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

question = "Where was Barack Obama born?"
context = "Barack Obama was born in Hawaii in 1961."

inputs = tokenizer(question, context, return_tensors='pt',
                   padding=True, truncation=True)
outputs = model(**inputs)
start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits)
answer = tokenizer.decode(inputs['input_ids'][0][start:end + 1])
print(f"答案: {answer}")
```

两个头都是建在每个 token 向量上的线性层，分别输出每个位置的 start logit 和 end logit。预测的 span 就是在 start ≤ end 的约束下，使 start + end logits 之和最大的那段连续区间。

### 句对分类（NLI、释义判定）

把两个句子用 `[SEP]` 拼起来，再用 `[CLS]` 头：

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3,  # 蕴含 / 中立 / 矛盾
)

premise = "A man is playing guitar"
hypothesis = "Someone is making music"

inputs = tokenizer(premise, hypothesis, return_tensors='pt',
                   padding=True, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
```

注意架构上几乎没变：同一个 `BertForSequenceClassification` 类，单句和句对分类只靠分词方式的区别就能切换。

---

## 真正能跑通的微调配方

用几千条标注微调一个 1.1 亿参数的模型，跟从头训练根本是两回事。从头训 ResNet 那一套默认值放到这里，会把预训练权重直接冲烂。

### 学习率小一点，配上权重衰减

预训练权重已经落在一个不错的盆地里，你要做的是轻轻挪动，而不是推土机式地碾过去。常见做法是 AdamW + 分组参数：bias 和 LayerNorm 不做权重衰减，其他参数衰减 0.01：

```python
from torch.optim import AdamW

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
```

### 先 warmup 再线性衰减

即使学习率已经很小，第一步直接打到一个刚刚加上的随机初始化的头上仍然太猛。Warmup 是把 LR 在前 ~10% 步内线性升上去，之后再线性降回 0：

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
```

### 显存不够就梯度累积

GPU 装不下 batch size 32，就把它拆成多个小 batch，把梯度累积起来再统一更新：

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 默认配方

懒得想就先按下面这套来，大多数论文也是这么干的：

| 设置             | 推荐值                                                         |
| ---------------- | -------------------------------------------------------------- |
| 学习率           | 2e-5 ~ 5e-5                                                    |
| Batch size       | 16-32（显存不够就梯度累积）                                    |
| 训练轮数         | 2-4（BERT 微调收敛很快，更多轮反而容易过拟合）                 |
| Warmup           | 总步数的 10%                                                   |
| 最大序列长度     | 128-512（越短越快，挑刚好能装下输入的最小值）                  |
| 优化器           | AdamW，权重衰减 0.01；bias 与 LayerNorm 设为 0                 |

---

## 一份完整的 HuggingFace 流水线

把上面这些拼到一起，下面是一份在 IMDB 情感数据集上端到端微调的代码：

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from datasets import load_dataset

# 1. 加载数据和模型
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2,
)

# 2. 分词
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True,
                     padding=True, max_length=512)

tokenized = dataset.map(preprocess, batched=True)

# 3. 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 4. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()

# 5. 评估
print(trainer.evaluate())
```

在一张现代 GPU 上跑几个小时，就能把 IMDB 准确率拉到 92-94% 左右——而在 BERT 之前，这个数字是要靠多年特征工程才能逼近的。

---

## 当年这个跨度有多大？看一眼 GLUE

要感受 BERT 当时的冲击，看一下原文里 8 个 GLUE 任务的成绩：

![BERT 与之前 SOTA 在 GLUE 上的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig7_glue_benchmark.png)

灰色是 BERT 之前的最佳模型，蓝色是 BERT-Base，紫色是 BERT-Large。在 CoLA（语言学可接受性）和 RTE（小数据量的文本蕴含）这种结构上比较难的任务上，**绝对提升达到了两位数**。一个预训练模型，加几个 epoch 的微调和一个微小的头，同时干翻了过去多年针对每个任务专门设计的架构。

---

## BERT 家族：RoBERTa、ALBERT、ELECTRA

BERT 是起点，不是终点。两年内涌现出一批从不同角度改进 BERT 的变体。

![BERT vs RoBERTa vs ALBERT vs ELECTRA](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT%E4%B8%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/fig5_variants_comparison.png)

### RoBERTa（Facebook，2019）：把训练流程做对就行

Liu 等人发现 BERT 其实是被**严重训练不足**的。RoBERTa 一行架构都没改，只调了配方：

- **去掉 NSP**：没用，甚至还可能有害。
- **动态掩码**：每个 epoch 重新生成掩码，而不是预处理时一次性固定下来。
- **更大的 batch**：从 256 提到 8K。
- **多得多的数据**：在 BooksCorpus + Wikipedia 之外加上 CC-News、OpenWebText、Stories（约 1600 亿 token，对比 BERT 的 33 亿）。

仅这些就让 GLUE 涨了 2-3 分，而架构跟 BERT-Large 一模一样。

### ALBERT（Google，2019）：把参数压一压

ALBERT 用更少的**独立**参数达到接近的效果：

- **嵌入分解**：把 $V \times H$ 的嵌入矩阵拆成 $V \times E$ 和 $E \times H$，其中 $E \ll H$。token 嵌入活在低维空间里，再投到隐藏维度。
- **跨层参数共享**：所有 Transformer 层共享同一套权重，深度不再随参数量线性放大。
- **句子顺序预测（SOP）** 取代 NSP：给两句**相邻**的句子，判断它们的顺序是否被颠倒过。这比"是否随机配对"难得多，信号也更有用。

ALBERT-xxlarge 大约 2.35 亿独立参数（对比 BERT-Large 的 3.4 亿），但在 GLUE 上能持平甚至超过。

### ELECTRA（Google，2020）：每个 token 都参与训练

MLM 只在 15% 的位置上有损失，剩下 85% 的位置纯属"路过"，浪费。ELECTRA 把 MLM 替换成**替换 token 检测（RTD）**：

1. 一个小型的生成器（一个迷你 MLM）把一些 token 换成"看起来像但不是原版"的词；
2. 一个更大的判别器逐 token 地判断：这个 token 是原文的，还是被换过的？
3. 训练完之后，只保留判别器。

因为判别器在**每个**位置上都有损失，ELECTRA 用更少的算力就能达到 BERT 级别——ELECTRA-Small 只用四分之一的训练时间就赶上了 BERT-Base。

### 对比表

| 模型    | 关键创新                                   | 参数量            | GLUE 相对提升 |
| ------- | ------------------------------------------ | ----------------- | ------------- |
| BERT    | 双向 MLM + NSP                             | 1.1 亿 / 3.4 亿   | 基线          |
| RoBERTa | 去 NSP、动态掩码、海量数据                 | 1.1 亿 / 3.55 亿  | +2 ~ +3       |
| ALBERT  | 嵌入分解 + 跨层参数共享                    | 0.12 亿 ~ 2.35 亿 | +1 ~ +2       |
| ELECTRA | RTD（100% token 都参与训练）               | 0.14 亿 ~ 3.35 亿 | +2 ~ +3       |

挑选其实是在选**配方**而不是选**架构**：四个模型都是编码器-only Transformer，骨架几乎一致。

---

## BERT 做不到的事

知道它的边界跟知道它的能力一样重要。

- **成本**：1.1 亿到 3.4 亿参数 + 二次方注意力，让实时推理变得不太舒服，需要靠蒸馏（DistilBERT、TinyBERT）或量化来压缩。
- **不能生成**：BERT 是双向 encoder-only，没法做合理的自回归解码。要生成文本就得用 GPT 系列的 decoder 模型——下一篇会讲。
- **512 token 上限**：位置嵌入只学到 0-511。处理长文档要靠滑窗、层级聚合，或者直接换架构（Longformer、BigBird）。
- **以英语为中心**：原始 BERT 只在英文上训练。多语言 BERT（mBERT）覆盖了 100+ 语言，但在每一个具体语言上都打不过那门语言的专用模型（BERT-Chinese、CamemBERT 等）。

---

## 常见问题

### 为什么用 `[CLS]` 做分类？

它放在每条输入的位置 0，注意力天然会让它在最后一层聚合整段序列的信息。预训练时的 NSP 目标也在显式训练 `[CLS]` 充当"序列摘要"。

### 选 BERT 还是 RoBERTa？

想要更高的分数、又有几个小时多余的 GPU 时间，就选 RoBERTa；想要最大的生态、最多的教程和最丰富的 checkpoint，BERT 仍然是最稳的基线。

### 变体怎么选？

通用基线选 BERT，要冲分用 RoBERTa，参数量受限（移动端、嵌入式）选 ALBERT，训练算力是瓶颈就用 ELECTRA。

### 中文怎么办？

中文有比 mBERT 更专业的选择：BERT-wwm-ext、MacBERT 是常用基线。直接拿 `bert-base-chinese` 起步通常也够用，关键看任务对中文细粒度（字 vs 词）有多敏感。

### 能拿来做句向量吗？

直接对 BERT token 向量做平均得到的句向量效果一般。要做相似度检索，请用 Sentence-BERT（在 BERT 上加一层 Siamese 对比训练得到的变体）。

---

## 核心要点

- BERT 用**双向预训练**（通过 MLM）让每个 token 在一次前向中就能看到完整上下文。
- 80/10/10 的掩码比例是为了避免训练-推理分布失配，并强迫模型即使输入"看起来正常"也要利用上下文。
- **预训练-微调**范式让一次昂贵的预训练能摊销到无数下游任务上；微调只需要一个很小的头、一个很小的学习率和几个 epoch。
- **RoBERTa、ALBERT、ELECTRA** 共同说明了：配方（数据、掩码方式、参数共享、训练目标）和架构同样重要。
- BERT 擅长**理解**任务，但不会**生成**——生成任务要看下一篇 GPT。

---

## 系列导航

| 部分  | 主题                                          | 链接                                                          |
| ----- | --------------------------------------------- | ------------------------------------------------------------- |
| 3     | RNN 与序列建模                                | [阅读](/zh/自然语言处理-三-RNN与序列建模/)                    |
| 4     | 注意力机制与 Transformer                      | [上一篇](/zh/自然语言处理-四-注意力机制与Transformer/)        |
| **5** | **BERT 与预训练模型（本文）**                 |                                                               |
| 6     | GPT 与生成式语言模型                          | [下一篇](/zh/自然语言处理-六-GPT与生成式语言模型/)            |
| 7     | 提示工程与 In-Context Learning                | [阅读](/zh/自然语言处理-七-提示工程与In-Context-Learning/)    |
