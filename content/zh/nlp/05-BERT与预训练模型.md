---
title: "自然语言处理（五）：BERT 与预训练模型"
date: 2025-10-21 09:00:00
tags:
  - NLP
  - BERT
  - 深度学习
  - 迁移学习
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "BERT 如何让双向预训练成为 NLP 的默认范式：从架构到 80/10/10 掩码规则，到微调技巧，再到 RoBERTa/ALBERT/ELECTRA 全家桶，并附完整 HuggingFace 代码。"
disableNunjucks: true
series_order: 5
translationKey: "nlp-5"
polished_by_qwen_max: true
---
2018 年 10 月，Google 推出了 BERT，一举刷新了 11 项 NLP 基准测试的记录。方法出人意料地简洁：仅需一个 Transformer 编码器，通过让模型根据双向上下文预测被随机遮盖的词进行预训练，再在同一模型上针对下游任务进行微调。在 BERT 出现之前，每个任务都需要从零开始训练一个专属模型；BERT 的出现彻底改变了这一局面，“一次预训练、多次微调”迅速成为该领域的标准范式。

如果你近几年接触过情感分析 API、能理解用户意图的搜索引擎或智能客服机器人，那么背后的核心技术很可能是 BERT 或其改进版本。


<!-- wanx-hero -->
![自然语言处理（五）：BERT与预训练模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/bert-pretrained-models/illustration_1.png)
## 这一篇你会学到

- 预训练技术的发展历程：从 Word2Vec 到 ELMo，再到 GPT-1 和 BERT  
- BERT 的核心架构：基于 WordPiece 分词的双向 Transformer 编码器  
- 掩码语言模型（MLM）与下一句预测（NSP）的工作原理，以及为什么采用 80/10/10 的掩码比例  
- 如何针对分类、命名实体识别（NER）、问答（QA）和句子对任务微调 BERT  
- BERT 系列模型： RoBERTa、 ALBERT、 ELECTRA，以及它们各自适用的场景  
- 微调中的实用技巧：学习率的选择、 warmup 策略、梯度累积等最佳实践  
- 一段可以直接复用的完整 HuggingFace 流水线代码  

**前置要求**：第 4 部分（Transformer 架构）以及 PyTorch 基础知识
## 预训练-微调范式的崛起

<!-- wanx-mid -->
![自然语言处理（五）：BERT与预训练模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/bert-pretrained-models/illustration_2.png)

在 BERT 出现之前，每个 NLP 任务都得从零开始：模型随机初始化后，用专门标注的数据集进行训练。这种方式不仅计算成本高，还存在明显弊端——不同任务之间无法共享知识，且由于标注数据量有限，模型的稳定性也难以保证。此后，预训练技术历经四个关键发展阶段，最终重塑了这一格局。

### 简短的发展历程

**Word2Vec （2013）**  
通过无标注文本学习静态词向量。这种方法的问题在于，同一个词无论出现在什么上下文中，其向量表示都是一样的。例如，“bank” 在 *river bank* 和 *bank account* 中对应相同的向量表示，无法反映上下文对词义的区分作用。

**ELMo （2018 年初）**  
引入了基于双向 LSTM 的上下文相关词向量。它通过加权求和每一层的隐藏状态生成最终的词表示：

$$\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \, h_{k,j}$$

其中 $h_{k,j}$ 是第 $j$ 层在位置 $k$ 的隐藏状态，$s_j$ 是可学习的 softmax 权重。 ELMo 证明了上下文相关的表示能够显著提升几乎所有下游任务的效果。然而，它的核心仍然是 RNN，训练速度慢，且难以并行化。

**GPT-1 （2018 年 6 月）**  
首次将 Transformer 应用于大规模预训练，目标函数是一个从左到右的语言模型：

$$P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})$$

GPT-1 表现不俗，但其语言建模是单向的：以句子 "the bank is closed" 为例，模型在预测 "bank" 时，仅能利用其左侧上下文（即 "the"），无法获取右侧的 "is closed" 等信息，因而难以消解歧义。

**BERT （2018 年 10 月）**  
真正的突破在于： BERT 将预训练目标改为双向上下文建模，使每个词的表征都能融合其左右两侧的全部上下文信息。仅此一项设计变更，便显著提升了各类下游任务的性能。

### 为什么这种范式如此重要

![一次预训练，多种任务微调](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT与预训练模型/fig4_finetune_pipeline.png)

整个流程分为两个阶段：

1. **预训练**  
在海量无标注文本（如书籍、维基百科、网页等）上，使用自监督任务训练模型。该步骤计算开销较大，但仅需执行一次。
2. **微调**  
针对每个具体的下游任务，添加一个小型的任务头，然后用小学习率进行端到端的微调。

这种范式的优势显而易见：

- **高效利用数据**  
预训练模型已经掌握了语法和大量常识性知识，因此每个任务通常只需要几百到几千条标注数据即可取得不错的效果。
- **广泛的适用性**  
同一个预训练模型可以应对分类、标注、抽取、句对匹配等多种任务。
- **强大的基线性能**  
即便是标准的 BERT 微调，其性能也往往优于以往为特定任务精心设计的复杂架构。
## BERT 的架构
BERT 是原始 Transformer 模型的编码器部分，通过堆叠 12 层或 24 层构建而成。它没有解码器，也不涉及因果掩码或自回归生成。简单来说， BERT 就是一个双向自注意力层的堆栈，能够将一段 token 序列转化为带有上下文信息的向量序列。

![BERT 双向编码器与输入嵌入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT与预训练模型/fig1_bert_architecture.png)

### 输入表示：三种嵌入相加

每个 token 的输入由三个同维度的嵌入相加得到：

$$\text{Input}_i = E^{\text{tok}}_{w_i} + E^{\text{seg}}_{s_i} + E^{\text{pos}}_{i}$$

- **Token 嵌入**：这是 WordPiece 分词后的子词 ID，来源于一个包含 30K 词汇的词表。
- **Segment 嵌入**：第一句的 token 使用 $E_A$，第二句的 token 使用 $E_B$。这种设计让 BERT 能够在不改变架构的情况下处理句子对任务，例如自然语言推理（NLI）和问答（QA）。
- **Position 嵌入**：为每个绝对位置（从 0 到 511）学习一个独立的向量。（不同于原始 Transformer 使用正弦波位置编码， BERT 的位置信息是通过训练学到的。）

两个特殊 token 在输入中扮演了重要角色：

- `[CLS]`：位于每个输入序列的开头。经过所有编码层后，它的隐藏状态被视为整个序列的全局摘要，并作为分类任务的输入。
- `[SEP]`：用于分隔句子 A 和句子 B，同时也标记输入序列的结束。

### 双向自注意力机制

在每层编码器内部，多头自注意力机制允许每个 token 关注序列中的其他所有 token：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

关键点在于：$Q$、$K$ 和 $V$ 都来自同一输入序列（即自注意力），并且没有因果掩码（因此是双向的）。这意味着，比如位置 3 上的 "bank"，可以在一次前向传播中同时整合左侧的 "river" 和右侧的 "is closed"。

### 两种配置

原论文提出了两种配置，至今仍是业界的标准参考：

|             | BERT-Base | BERT-Large |
| ----------- | --------- | ---------- |
| 层数        | 12        | 24         |
| 隐藏维度    | 768       | 1024       |
| 注意力头数  | 12        | 16         |
| 参数量      | 1.1 亿    | 3.4 亿     |

BERT-Base 可以在单张消费级 GPU 上完成推理任务，而 BERT-Large 则是 2018 年多项记录的主力模型。
## 预训练目标
BERT 的预训练结合了两项自监督任务。第一项任务声名远扬，而第二项任务后来被证明可有可无。

### 掩码语言建模（MLM）

![MLM 的 80/10/10 损坏规则](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig2_mlm_corruption.png)

对于每个输入序列，随机选择其中 15% 的 token 位置进行处理。在每个选中的位置上：

- **80% 的概率**：将该 token 替换为 `[MASK]`；
- **10% 的概率**：将其替换为词表中的一个随机 token；
- **10% 的概率**：保持原样不变。

模型的训练目标是预测这些位置上的**原始 token**，通过最小化以下损失函数来实现：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid \tilde{x})$$

其中，$\mathcal{M}$ 表示被掩码的位置集合，$\tilde{x}$ 是经过损坏的输入。

**为什么采用 80/10/10 的比例？** 这个设计是为了避免两种潜在问题：

- 如果只用 `[MASK]`，模型在微调时不会遇到 `[MASK]`（下游任务中没有掩码），导致训练和推理之间的分布不一致。
- 如果只用随机 token，模型会对输入失去信任，从而过度依赖远处的信息。
- 留下 10% 的 token 不变，则是为了迫使模型即使面对看似正常的 token，也不能简单地复制——必须依靠上下文信息来判断。

MLM 是 BERT 实现双向编码的核心：从左右两侧预测被遮挡的词，要求编码器在每个位置都融合整个序列的信息。

### 下一句预测（NSP）

![NSP：正例 vs 负例](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig3_nsp.png)

加入 NSP 的目的是让 BERT 学会句子对之间的语义关系，以便迁移到 NLI 和 QA 等任务。每个训练样本包含两个句子 `[CLS] A [SEP] B [SEP]`，标签通过随机决定：

- **50% 的情况**： B 是语料中紧跟 A 的下一句（标签 `IsNext`）；
- **50% 的情况**： B 是从其他文档中随机抽取的句子（标签 `NotNext`）。

最后一层的 `[CLS]` 向量通过一个 Linear+softmax 头预测标签：

$$P(\text{IsNext}) = \text{softmax}(W \, h_{\text{[CLS]}} + b)$$

预训练的总损失是 MLM 损失和 NSP 损失的加和。

> 这里有个后来被推翻的设计：后续工作（如 RoBERTa、 ALBERT）发现 NSP 的作用微乎其微，去掉或替换它反而**更好**。这个问题我们会在变体部分再详细讨论。

### 预训练语料

BERT 的训练数据来自 **BooksCorpus**（约 8 亿词）和 **英文维基百科**（约 25 亿词），总计约 33 亿词。以 2026 年的标准来看，这个数据量很小——现代 LLM 动辄训练在万亿级 token 上——但在当时已经足够刷新基准线了。
## WordPiece 分词
![WordPiece 子词分词](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT与预训练模型/fig6_wordpiece.png)

BERT 并不直接处理完整的单词，而是采用了一种名为 **WordPiece** 的子词分词方法。这种方法在两种极端之间找到了一个巧妙的平衡：

- 如果使用整词词表，为了覆盖真实语料库中的词汇，往往需要几百万个词条，但即便如此，在推理阶段仍然可能遇到未登录词（OOV）的问题。
- 而如果完全基于字符构建词表，虽然词表规模可以很小，但模型每次都需要从零开始重新拼凑词义，这无疑会增加计算负担。

WordPiece 的解决思路是：通过贪心算法不断合并字符对，优先选择那些合并后能够最大程度提升训练语料似然性的组合，最终生成一个包含 30K 个词条的词表。在实际分词时，它会将每个单词拆分为词表中最长的匹配片段；对于单词内部的片段，则会在前面加上 `##` 标记，表示这是单词的延续部分：

```text
playing       -> play  ##ing
unbelievable  -> un    ##bel  ##iev  ##able
transformer   -> transform  ##er
Tokyo2024     -> tokyo  ##20  ##24
```

这种设计既确保了不会出现未登录词（任何内容都可以分解为已知的子词，最差情况下也能退化到单个字符），又保留了高频词汇作为单一 token 的高效性，从而兼顾了性能和灵活性。
## 微调 BERT 用于下游任务
微调的核心思想其实很简单：同一个骨干网络可以适配几乎所有任务，只需更换“头”部分即可。

### 文本分类

如果需要对句子级别的标签进行预测（例如情感分析、垃圾邮件检测或意图识别），只需要将输入送入 BERT，提取最终的 `[CLS]` 向量，并通过一个线性层映射到目标类别即可。

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
print(f"预测: {prediction.item()}")  # 输出 0 或 1
```

具体流程如下：

1. 对输入文本进行分词，并在开头和结尾分别添加 `[CLS]` 和 `[SEP]`。
2. 输入经过 12 层 Transformer 编码器。
3. 提取 `[CLS]` 的隐藏状态（对于 BERT-Base 来说，是 768 维向量）。
4. 添加一个线性层，将 768 维向量映射到类别数。

### 命名实体识别（NER）

对于 token 级别的任务（如命名实体识别），使用每个 token 的向量，而不是 `[CLS]`。

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=9,  # 比如 PER/ORG/LOC/MISC 的 BIO 标签 + O
)

text = "Barack Obama was born in Hawaii"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: 标签 {pred.item()}")
```

需要注意的是， WordPiece 分词可能会将某些单词切分为多个片段。例如，“Hawaii”可能保持完整，而“Tokyo2024”会被拆成三个子词。在还原到词级别时，通常只保留每个单词第一个子词的预测结果，忽略后续的 `##` 片段。

### 抽取式问答

对于 SQuAD 风格的问答任务，模型需要从上下文中预测答案片段的起始和结束位置。

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

这里有两个输出头，它们分别是基于每个 token 向量的线性层，用于生成每个位置的 start logit 和 end logit。最终预测的答案片段是满足 start ≤ end 条件下，使 start + end logits 最大的连续区间。

### 句对分类（NLI、释义判定）

对于句对分类任务（如自然语言推理或释义判定），将两个句子用 `[SEP]` 拼接起来，并使用 `[CLS]` 头进行分类。

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

可以看到，架构改动非常小。同一个 `BertForSequenceClassification` 类，单句分类和句对分类只需调整分词方式即可实现切换。
## 真正有效的微调方法

用几千条标注数据去微调一个 1.1 亿参数的模型，和从零开始训练完全是两码事。 ResNet 那套从头训练的默认参数，放到微调场景下会直接把预训练权重搞砸。

### 学习率要小，记得加权重衰减

预训练权重已经处于一个不错的“舒适区”，我们的目标是轻轻推它一把，而不是粗暴地推倒重来。常用的做法是用 AdamW，并对参数分组： bias 和 LayerNorm 参数不加权重衰减，其他参数设置为 0.01：

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

### 先 warmup，再线性衰减

学习率再小，如果一开始就作用到新加入的随机初始化层上，还是会显得太猛。 Warmup 的作用是在前 10% 的训练步数里逐步提升学习率，然后通过线性调度慢慢降回零：

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
```

### 显存不够就用梯度累积

如果 GPU 装不下 batch size 32，可以把大 batch 拆成几个小 batch，分别计算梯度后再统一更新：

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

### 默认配置

如果不确定怎么调，就先按这套来，大多数论文都这么干：

| 设置             | 推荐值                                                         |
| ---------------- | -------------------------------------------------------------- |
| 学习率           | 2e-5 ~ 5e-5                                                    |
| Batch size       | 16-32 （显存不够就用梯度累积）                                  |
| 训练轮数         | 2-4 （BERT 微调很快，多训容易过拟合）                           |
| Warmup           | 总步数的 10%                                                   |
| 最大序列长度     | 128-512 （越短越快，选能装下输入的最小值）                      |
| 优化器           | AdamW，权重衰减 0.01； bias 和 LayerNorm 不衰减                 |

---
## 一份完整的 HuggingFace 流水线

接下来，我们将所有步骤串联起来，展示如何在 IMDB 情感分析数据集上实现端到端的微调流程：

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from datasets import load_dataset

# 1. 加载数据与模型
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2,
)

# 2. 数据预处理
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True,
                     padding=True, max_length=512)

tokenized = dataset.map(preprocess, batched=True)

# 3. 配置训练参数
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

# 4. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()

# 5. 评估模型
print(trainer.evaluate())
```

使用一张现代 GPU，这个流程只需几个小时就能完成训练，并在 IMDB 数据集上达到约 92%-94% 的准确率。而在 BERT 出现之前，这样的性能需要研究者耗费数年时间设计手工特征才能勉强实现。
## 当年这个跨度有多大？看一眼 GLUE

要明白为什么 BERT 的出现让整个领域为之震动，不妨看看它在原论文中展示的 GLUE 基准测试中的八项任务表现。

![BERT 与之前 SOTA 在 GLUE 上的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT与预训练模型/fig7_glue_benchmark.png)

图中灰色柱状图代表之前的任务专用最佳模型，蓝色和紫色分别对应 BERT-Base 和 BERT-Large。在像 CoLA （语言学可接受性判断）和 RTE （小规模文本蕴含推理）这样结构复杂、难度较高的任务上， BERT 的提升幅度竟然达到了两位数。更令人惊叹的是，这个单一的预训练模型，仅需微调几个 epoch，并附加一个极简的分类头，就能同时在所有任务上超越那些经过多年精心设计的专用架构。
## BERT 家族： RoBERTa、 ALBERT、 ELECTRA
BERT 并不是终点，而是一个起点。在短短两年时间里，围绕 BERT 衍生出了一系列改进版本，它们从不同角度对原始模型进行了优化。

![BERT 与 RoBERTa 与 ALBERT 与 ELECTRA](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/05-BERT与预训练模型/fig5_variants_comparison.png)

### RoBERTa （Facebook， 2019）：把训练做到位

Liu 等人的研究表明， BERT 的训练并不充分。 RoBERTa 没有改变模型结构，而是通过调整训练策略显著提升了性能：

- **去掉 NSP**： NSP 不仅没有帮助，甚至可能拖累效果。
- **动态掩码**：每次训练时重新生成掩码，而不是在预处理阶段固定一个静态掩码。
- **增大 batch size**：将 batch size 从 256 提升到 8K。
- **大幅增加数据量**：在 BooksCorpus 和 Wikipedia 的基础上，新增了 CC-News、 OpenWebText 和 Stories 数据集，总 token 数达到约 1600 亿（相比之下， BERT 只有 33 亿）。

最终， RoBERTa 在完全沿用 BERT-Large 架构的情况下， GLUE 分数提升了 2-3 分。

### ALBERT （Google， 2019）：减少参数规模

ALBERT 通过创新设计，在显著减少独立参数数量的同时，依然保持了竞争力：

- **嵌入分解**：将原本 $V \times H$ 的嵌入矩阵拆分为 $V \times E$ 和 $E \times H$，其中 $E \ll H$。 Token 嵌入先映射到低维空间，再投影到隐藏层维度。
- **跨层参数共享**：所有 Transformer 层共享同一套权重，从而避免了深度增加带来的参数膨胀。
- **句子顺序预测（SOP）**：取代 NSP，任务变为判断两句话的顺序是否被调换。相比检测随机配对，这一任务更具挑战性，也提供了更有价值的训练信号。

ALBERT-xxlarge 的独立参数量仅为约 2.35 亿（相比之下， BERT-Large 为 3.4 亿），但在 GLUE 上的表现却持平甚至超越了 BERT。

### ELECTRA （Google， 2020）：让每个 token 都发挥作用

MLM 只在 15% 的位置计算损失，其余 85% 的 token 被浪费了。 ELECTRA 提出了**替换 Token 检测（RTD）**，彻底改变了这一局面：

1. 使用一个小生成器（一个轻量级 MLM）替换部分 Token。
2. 判别器逐 Token 判断：这个 Token 是原始的，还是被生成器替换过的？
3. 训练完成后，丢弃生成器，只保留判别器。

由于判别器在每个位置都有损失， ELECTRA 以更少的计算量达到了与 BERT 相当的效果——ELECTRA-Small 只需四分之一的训练时间，就能追平 BERT-Base。

### 对比表

| 模型    | 核心创新                                   | 参数量            | GLUE 提升幅度 |
| ------- | ------------------------------------------ | ----------------- | ------------- |
| BERT    | 双向 MLM + NSP                             | 1.1 亿 / 3.4 亿   | 基线          |
| RoBERTa | 去掉 NSP、动态掩码、海量数据               | 1.1 亿 / 3.55 亿  | +2 ~ +3       |
| ALBERT  | 嵌入分解 + 跨层参数共享                    | 0.12 亿 ~ 2.35 亿 | +1 ~ +2       |
| ELECTRA | RTD （100% Token 都参与训练）               | 0.14 亿 ~ 3.35 亿 | +2 ~ +3       |

选择这些模型更像是在挑选配方，而不是架构：四个模型都是编码器-only Transformer，整体结构非常相似。
## BERT 做不到的事

搞清楚 BERT 能做什么很重要，但了解它的局限性同样关键。

- **算力开销大**： BERT 参数量高达 1.1 亿到 3.4 亿，再加上注意力机制的复杂度是二次方增长，直接用原始模型做实时推理会非常吃力。如果不想办法优化，比如使用蒸馏版模型（DistilBERT、 TinyBERT）或者量化技术，性能瓶颈会让人头疼。
- **无法生成文本**： BERT 是一个纯 encoder 架构，采用双向注意力机制，因此它天生就不适合用来生成文本。想实现自回归式的文本生成？那得换用 GPT 这类 decoder 模型——具体内容我们会在第六部分深入探讨。
- **最多支持 512 个 token**： BERT 的位置编码只学习了 0 到 511 的范围，处理长文档时就会捉襟见肘。要应对这种情况，要么采用滑动窗口的方式分段处理，要么通过层级聚合提取信息，要么干脆换个架构，比如 Longformer 或 BigBird。
- **偏向英语**：最初的 BERT 模型完全基于英语数据训练，虽然后来推出了多语言版（mBERT），覆盖了 100 多种语言，但在具体语言上的表现往往不如专门针对该语言优化的模型（例如 BERT-Chinese、 CamemBERT 等）。
## 常见问题
### 为什么用 `[CLS]` 做分类？

`[CLS]` 被固定放在每个输入序列的第 0 位，这种设计让注意力机制能够自然地帮助它在最后一层聚合整个序列的信息。此外，在预训练阶段， NSP （下一句预测）任务也让 `[CLS]` 学会了如何充当整个序列的总结表示。

### BERT 还是 RoBERTa？

如果你追求更高的性能，并且愿意多花几个小时的 GPU 时间， RoBERTa 是更好的选择。但如果你更看重生态系统的完善程度、教程资源的丰富性以及可用 checkpoint 的多样性，那么 BERT 依然是最稳妥的基线模型。

### 怎么选择变体？

对于通用任务， BERT 是一个可靠的起点；如果目标是提升精度， RoBERTa 更适合；如果参数量是一个关键限制（比如移动端或嵌入式场景）， ALBERT 是不错的选择；而当训练算力成为瓶颈时， ELECTRA 则是一个高效的替代方案。

### 非英语语言怎么办？

对于多语言任务， mBERT 或 XLM-RoBERTa 是常用的基线模型。但如果需要针对特定语言的最佳效果，建议使用专门优化的 checkpoint：例如中文可以用 BERT-wwm-ext 或 MacBERT，法语推荐 CamemBERT，荷兰语则有 BERTje，依此类推。

### 能用来生成句向量吗？

直接对 BERT 的 token 向量取平均值，得到的句向量质量通常不尽如人意。如果需要计算句子相似度，建议使用 Sentence-BERT，这是通过 Siamese 对比损失微调后的变体，专为相似度任务设计，效果显著优于原始 BERT。
## 核心要点
- BERT 通过 Masked Language Modeling （MLM）实现了**双向预训练**，让每个 token 在单次前向传播中就能感知到完整的上下文信息。
- 80/10/10 的掩码分配比例经过精心设计，不仅避免了训练和测试之间的分布差异，还迫使模型在输入看似未被遮盖的情况下依然依赖上下文信息。
- **先预训练再微调**的范式非常高效：一次成本高昂的预训练可以服务于所有下游任务；而微调阶段只需添加一个轻量级的分类头，设置较小的学习率，并用少量 epoch 即可完成。
- **RoBERTa、 ALBERT 和 ELECTRA** 的研究表明，数据质量、掩码策略、参数共享方式以及训练目标这些“配方”，与模型架构本身同样重要。
- BERT 在**理解类任务**上表现卓越，但无法生成文本。如果需要生成能力，就要看 GPT （第六部分）。