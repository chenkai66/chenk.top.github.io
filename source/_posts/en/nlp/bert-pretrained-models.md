---
title: "NLP Part 5: BERT and Pretrained Models"
date: 2025-08-31 09:00:00
tags:
  - NLP
  - BERT
  - Deep Learning
  - Transfer Learning
categories: Natural Language Processing
series:
  name: "Natural Language Processing"
  part: 5
  total: 12
lang: en
mathjax: true
description: "How BERT made bidirectional pretraining the default in NLP. We unpack the architecture, the 80/10/10 masking rule, fine-tuning recipes, and the RoBERTa/ALBERT/ELECTRA family with HuggingFace code."
disableNunjucks: true
series_order: 5
---

In October 2018, Google released BERT and broke eleven NLP benchmarks at once. The recipe is almost embarrassingly simple: take a Transformer encoder, train it to predict words that have been randomly hidden using both left and right context, and then fine-tune the same pretrained model for whatever downstream task you have. Before BERT, every task came with its own from-scratch model. After BERT, "pretrain once, fine-tune everywhere" became the default mental model for the entire field.

If you have used a sentiment-analysis API, a search engine that understands intent, or a customer-support bot in the last few years, there is a very good chance BERT or one of its descendants is doing the heavy lifting underneath.

## What you will learn

- How pretraining evolved: Word2Vec to ELMo to GPT-1 to BERT
- BERT's architecture: a bidirectional Transformer encoder with WordPiece input
- Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), and why the 80/10/10 mask split exists
- Fine-tuning BERT for classification, NER, QA and sentence-pair tasks
- The BERT family: RoBERTa, ALBERT, ELECTRA, and when to pick each
- Practical fine-tuning recipes (learning rate, warmup, gradient accumulation)
- A complete HuggingFace pipeline you can copy-paste

**Prerequisites:** Part 4 (Transformer architecture) and basic PyTorch.

---

## The rise of pretrain-then-finetune

Before BERT, every NLP task started from a freshly initialized model trained on its own labeled dataset. That was expensive (compute), wasteful (no knowledge sharing across tasks), and brittle (small datasets gave shaky models). The story of how the field escaped this trap runs through four landmark systems.

### A short evolution

**Word2Vec (2013).** Static word embeddings learned from raw text. The same vector represented "bank" in *river bank* and in *bank account* -- there was no way for context to change a word's meaning.

**ELMo (early 2018).** A bidirectional LSTM produced context-dependent vectors by combining hidden states from every layer:

$$
\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \, h_{k,j}
$$

where $h_{k,j}$ is the hidden state of layer $j$ at token position $k$ and $s_j$ are learned softmax weights. ELMo proved that contextual representations dramatically improve almost every downstream task -- but it was still RNN-based, so training was slow and hard to parallelize.

**GPT-1 (June 2018).** The first system to scale a Transformer through pretraining. It used a left-to-right language model:

$$
P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})
$$

GPT-1 was strong but unidirectional: when reading "the bank is closed," it could not use "closed" to disambiguate "bank," because at the position of "bank" the model has not yet seen "closed."

**BERT (October 2018).** The breakthrough: change the pretraining objective so every token can attend to its full context, in *both* directions, simultaneously. That single decision unlocked an across-the-board jump in benchmark scores.

### Why the paradigm shift matters

![Pretrain once, fine-tune for many tasks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig4_finetune_pipeline.png)

The pretrain-then-finetune pipeline has two stages:

1. **Pretrain** on massive unlabeled text (books, Wikipedia, the web) using a self-supervised objective. This is expensive but you do it once.
2. **Fine-tune** on a small labeled dataset for each downstream task by adding a tiny head and training end-to-end with a small learning rate.

The wins are concrete:

- **Data-efficient.** The backbone already "knows" syntax and a lot of world knowledge, so you usually need only hundreds to a few thousand labeled examples per task.
- **Universal.** The same backbone serves classification, tagging, span extraction, and pairwise tasks.
- **Strong baselines.** Plain BERT fine-tuning routinely beats the bespoke architectures it replaced.

---

## BERT's architecture

BERT is the **encoder** half of the original Transformer, repeated 12 or 24 times. There is no decoder, no causal mask, and no autoregressive generation -- just a bidirectional stack of self-attention layers that turn a sequence of tokens into a sequence of contextual vectors.

![BERT bidirectional encoder and input embeddings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig1_bert_architecture.png)

### Input representation: three embeddings, summed

For every token, BERT adds three learned embeddings of the same dimension:

$$
\text{Input}_i = E^{\text{tok}}_{w_i} + E^{\text{seg}}_{s_i} + E^{\text{pos}}_{i}
$$

- **Token embedding** -- the WordPiece sub-token id, drawn from a 30K vocabulary.
- **Segment embedding** -- $E_A$ for tokens belonging to the first sentence, $E_B$ for the second. This lets BERT model sentence-pair tasks (NLI, QA) without any architectural change.
- **Position embedding** -- a *learned* vector for each absolute position from 0 to 511. (Unlike the original Transformer's sinusoidal positions, BERT learns its own.)

Two special tokens carry most of the protocol:

- `[CLS]` is prepended to every input. After all layers, its hidden state is treated as a pooled summary of the whole sequence and is the input to classification heads.
- `[SEP]` separates sentence A from sentence B and marks the end of the input.

### Bidirectional self-attention

Inside every encoder layer, multi-head self-attention lets every token look at every other token:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Crucially, $Q$, $K$, and $V$ all come from the same input sequence (self-attention) and there is **no causal mask** (bidirectional). So the representation of "bank" at position 3 can simultaneously incorporate "river" on its left and "is closed" on its right within a single forward pass.

### Two sizes

The original paper released two configurations that are still the reference points today:

|                 | BERT-Base | BERT-Large |
| --------------- | --------- | ---------- |
| Layers          | 12        | 24         |
| Hidden size     | 768       | 1024       |
| Attention heads | 12        | 16         |
| Parameters      | 110M      | 340M       |

BERT-Base fits on a single consumer GPU for inference. BERT-Large was the workhorse that set most of the 2018 records.

---

## Pretraining objectives

BERT's pretraining combines two self-supervised tasks. The first is the famous one; the second turned out to be optional.

### Masked Language Modeling (MLM)

![MLM 80/10/10 corruption rule](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig2_mlm_corruption.png)

For each input sequence, randomly select 15% of the token positions. At each chosen position:

- with probability **80%**, replace the token with `[MASK]`,
- with probability **10%**, replace it with a *random* vocabulary token,
- with probability **10%**, leave it unchanged.

The model is trained to predict the *original* token at every chosen position by minimising

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid \tilde{x})
$$

where $\mathcal{M}$ is the set of masked positions and $\tilde{x}$ is the corrupted input.

**Why the 80/10/10 mix?** It is engineered to prevent two failure modes:

- If you only used `[MASK]`, the model would never see `[MASK]` during fine-tuning (downstream inputs have no masks), creating a train/test mismatch.
- If you only replaced tokens with random ones, the model could not trust any input token and would underuse local information.
- Leaving 10% unchanged forces the model to use context even when the surface form looks correct -- otherwise it could learn the shortcut "if a token is not weird-looking, just copy it."

The MLM objective is what makes BERT bidirectional in a clean way: predicting the masked word from *both sides* requires the encoder to fuse information from the entire sequence at every position.

### Next Sentence Prediction (NSP)

![NSP: positive vs negative pairs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig3_nsp.png)

NSP was added so that BERT could learn sentence-pair semantics for tasks like NLI and QA. Each training example packs two sentences `[CLS] A [SEP] B [SEP]`, with the label generated by a coin flip:

- 50% of the time, B is the *actual* sentence that followed A in the corpus (label `IsNext`).
- 50% of the time, B is a *random* sentence from a different document (label `NotNext`).

A linear-plus-softmax head reads the final `[CLS]` vector and predicts the label:

$$
P(\text{IsNext}) = \text{softmax}(W \, h_{\text{[CLS]}} + b)
$$

The total pretraining loss is just the sum of the MLM and NSP losses.

> A footnote that aged badly: subsequent work (RoBERTa, ALBERT) found NSP contributes very little, and removing or replacing it actually *helps*. We will return to this in the variants section.

### Pretraining corpus

BERT was trained on **BooksCorpus** (about 800M words) and **English Wikipedia** (about 2.5B words), totalling roughly 3.3B words. By 2026 standards that is tiny -- modern LLMs train on trillions of tokens -- but it was already enough to set a new bar.

---

## WordPiece tokenization

![WordPiece subword tokenization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig6_wordpiece.png)

BERT does not work with whole words. Instead, it uses **WordPiece**, a subword scheme that strikes a balance between two extremes:

- Whole-word vocabularies need millions of entries to cover any real corpus and still hit out-of-vocabulary tokens at inference time.
- Pure character vocabularies are tiny but force the model to reconstruct word meaning from scratch every sequence.

WordPiece picks a 30K-token vocabulary by greedily merging the character pairs whose merger most increases the likelihood of the training corpus. At tokenization time, it segments each word into the longest pieces it has in the vocabulary; pieces inside a word are prefixed with `##` to mark them as continuations:

```
playing       -> play  ##ing
unbelievable  -> un    ##bel  ##iev  ##able
transformer   -> transform  ##er
Tokyo2024     -> tokyo  ##20  ##24
```

This guarantees there is no out-of-vocabulary token (everything decomposes into known pieces, ultimately into single characters), while keeping common words as single tokens for efficiency.

---

## Fine-tuning BERT for downstream tasks

The big idea behind fine-tuning is that the same backbone serves nearly every task; only the head differs.

### Text classification

For a sentence-level label (sentiment, spam, intent), feed the input through BERT and project the final `[CLS]` vector through a linear layer:

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
print(f"Prediction: {prediction.item()}")  # 0 or 1
```

Internally:

1. Tokenize and add `[CLS]` and `[SEP]`.
2. Pass through 12 encoder layers.
3. Take the `[CLS]` hidden state (768-dim for BERT-Base).
4. Apply a single linear layer mapping 768 -> num_labels.

### Named entity recognition (NER)

Token-level tasks use the per-token vectors instead of `[CLS]`:

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=9,  # e.g. BIO tags for PER/ORG/LOC/MISC + O
)

text = "Barack Obama was born in Hawaii"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: tag {pred.item()}")
```

A subtlety: WordPiece can split a word ("Hawaii" might stay whole, but "Tokyo2024" splits into three pieces). When converting back to word-level entities, you typically take the prediction at the *first* sub-token of each word and ignore the continuation pieces.

### Question answering (extractive)

For SQuAD-style QA, the model predicts the start and end positions of the answer span inside the context:

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
print(f"Answer: {answer}")
```

The two heads are linear layers on top of every token vector, producing one start logit and one end logit per position. The predicted span is the contiguous range that maximizes start + end logits subject to start <= end.

### Sentence-pair classification (NLI, paraphrase)

Pack the two sentences with a `[SEP]` between them and use the `[CLS]` head:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3,  # entailment / neutral / contradiction
)

premise = "A man is playing guitar"
hypothesis = "Someone is making music"

inputs = tokenizer(premise, hypothesis, return_tensors='pt',
                   padding=True, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
```

Notice how minimal the architectural change is: the same `BertForSequenceClassification` class handles both single-sentence and sentence-pair classification just by tokenizing differently.

---

## Fine-tuning recipes that actually work

Fine-tuning a 110M-parameter model with a few thousand labels is fundamentally different from training from scratch. The defaults that work for ResNet from scratch will overshoot and destroy the pretrained weights here.

### Use a small learning rate, with weight decay

The pretrained weights live in a good basin; you want to nudge them, not bulldoze them. The standard trick is AdamW with separate parameter groups so that bias and LayerNorm parameters are *not* decayed:

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

### Warm up, then linearly decay

Even a tiny learning rate is too large if applied to the very first step on a freshly added head. Warmup ramps the LR up over the first ~10% of steps, then a linear schedule cools it back to zero:

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
```

### Gradient accumulation if memory is tight

When you cannot fit batch size 32 on your GPU, simulate it by accumulating gradients across several smaller forward/backward passes:

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

### Default recipe

When in doubt, start here -- it is what most papers use:

| Setting             | Recommended                                                     |
| ------------------- | --------------------------------------------------------------- |
| Learning rate       | 2e-5 to 5e-5                                                    |
| Batch size          | 16-32 (use gradient accumulation if needed)                     |
| Epochs              | 2-4 (BERT fine-tunes fast; more epochs risk overfitting)        |
| Warmup              | 10% of total steps                                              |
| Max sequence length | 128-512 (shorter is faster; pick the smallest that fits inputs) |
| Optimizer           | AdamW with weight decay 0.01 on weights, 0.0 on bias/LayerNorm  |

---

## A complete HuggingFace pipeline

Putting it all together, here is an end-to-end fine-tuning pipeline on the IMDB sentiment dataset:

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from datasets import load_dataset

# 1. Load data and model
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2,
)

# 2. Tokenize
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True,
                     padding=True, max_length=512)

tokenized = dataset.map(preprocess, batched=True)

# 3. Training config
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

# 4. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()

# 5. Evaluate
print(trainer.evaluate())
```

On a single modern GPU this trains in a few hours and reaches around 92-94% accuracy on IMDB -- a number that took years of hand-engineered features to hit before BERT.

---

## How big was the jump? GLUE in 2018

To appreciate why BERT shocked the field, look at the eight-task GLUE benchmark from the original paper.

![BERT vs prior SOTA on GLUE](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig7_glue_benchmark.png)

The bars compare the previous task-specific best (gray) with BERT-Base (blue) and BERT-Large (purple). On structurally hard tasks like CoLA (linguistic acceptability) and the small RTE (textual entailment) dataset, the absolute gain was *double-digit*. A single pretrained model, fine-tuned with a few epochs and a tiny head, beat years of bespoke architectures simultaneously on every task.

---

## The BERT family: RoBERTa, ALBERT, ELECTRA

BERT was the start, not the end. Within two years a small family of variants improved on it along orthogonal axes.

![BERT vs RoBERTa vs ALBERT vs ELECTRA](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/bert-pretrained-models/fig5_variants_comparison.png)

### RoBERTa (Facebook, 2019): just train it properly

Liu et al. showed that BERT was significantly *under*-trained. Without changing the architecture, RoBERTa changes the recipe:

- **Drop NSP.** It does not help and may hurt.
- **Dynamic masking.** Generate a fresh mask for each epoch instead of reusing a static one created during preprocessing.
- **Bigger batches.** 8K instead of 256.
- **Much more data.** Add CC-News, OpenWebText, and Stories on top of BooksCorpus + Wikipedia (~160B tokens vs BERT's ~3.3B).

The result is a 2-3 GLUE point jump using exactly the BERT-Large architecture.

### ALBERT (Google, 2019): squeeze the parameters

ALBERT achieves competitive scores with far fewer *unique* parameters:

- **Factorized embeddings.** Rather than a single $V \times H$ embedding matrix, decompose it into $V \times E$ and $E \times H$ with $E \ll H$. Token embeddings live in a small space and are projected up to the hidden size.
- **Cross-layer parameter sharing.** All Transformer layers share the same weights, so depth no longer multiplies parameter count.
- **Sentence Order Prediction (SOP)** replaces NSP with a harder task: given two adjacent sentences, decide whether their order has been swapped. This is harder than detecting random pairs and turns out to be a more useful signal.

ALBERT-xxlarge has roughly 235M unique parameters (compare with BERT-Large's 340M) but matches or beats it on GLUE.

### ELECTRA (Google, 2020): use every token

MLM only computes a loss at 15% of positions, which is wasteful. ELECTRA replaces MLM with **Replaced Token Detection (RTD)**:

1. A small generator (a tiny MLM) plausibly replaces some tokens.
2. A larger discriminator examines every token and decides: was this the original, or did the generator swap it in?
3. We throw away the generator and keep only the discriminator.

Because the discriminator gets a loss at *every* position, ELECTRA reaches BERT-quality scores with much less compute -- ELECTRA-Small matches BERT-Base while training in a quarter of the time.

### Comparison

| Model   | Key innovation                            | Parameters        | Relative GLUE |
| ------- | ----------------------------------------- | ----------------- | ------------- |
| BERT    | Bidirectional MLM + NSP                   | 110M / 340M       | Baseline      |
| RoBERTa | Drop NSP, dynamic masking, far more data  | 110M / 355M       | +2-3          |
| ALBERT  | Factorized embeddings + parameter sharing | 12M to 235M       | +1-2          |
| ELECTRA | RTD (loss on 100% of tokens)              | 14M to 335M       | +2-3          |

Picking among them is a recipe choice, not an architecture one: all four are encoder-only Transformers with very similar shape.

---

## What BERT cannot do

It is just as important to know BERT's limits.

- **Cost.** 110-340M parameters and quadratic attention make real-time inference uncomfortable without distillation (DistilBERT, TinyBERT) or quantization.
- **No generation.** BERT is encoder-only with bidirectional attention. There is no sensible way to autoregressively decode text from it. For generation you need GPT-style decoder models -- the topic of Part 6.
- **512-token ceiling.** Position embeddings are learned for positions 0-511. Long documents need sliding windows, hierarchical aggregation, or a different architecture (Longformer, BigBird).
- **English-centric.** The original BERT was trained on English text only. Multilingual BERT covers 100+ languages but underperforms language-specific models (BERT-Chinese, CamemBERT, etc.) on their target language.

---

## Common questions

**Why use `[CLS]` for classification?** It is placed at position 0 of every input, so attention naturally lets it aggregate information from the whole sequence by the final layer. The pretraining NSP objective also conditions `[CLS]` to act as a sequence summary.

**BERT or RoBERTa?** If you want the highest score and have a few extra GPU-hours, RoBERTa. If you want the largest ecosystem, the most tutorials, and the most checkpoints to choose from, BERT remains the safest baseline.

**How do I pick a variant?** Use BERT for general baselines, RoBERTa for pushing accuracy, ALBERT when parameter count matters (mobile, embedded), ELECTRA when training compute is the bottleneck.

**What about non-English languages?** Use mBERT or XLM-RoBERTa as multilingual baselines. For best per-language performance, use a dedicated checkpoint -- BERT-wwm-ext or MacBERT for Chinese, CamemBERT for French, BERTje for Dutch, and so on.

**Can BERT be used for sentence embeddings?** Naively averaging BERT token vectors gives mediocre sentence embeddings. Use Sentence-BERT (a fine-tuned variant trained with a Siamese contrastive loss) when you need similarity scoring.

---

## Key takeaways

- BERT introduced **bidirectional pretraining** via Masked Language Modeling, letting every token see its full context in a single forward pass.
- The 80/10/10 mask split is engineered to avoid train-test mismatch and to force the model to use context even when the input looks unmasked.
- The **pretrain-then-finetune** paradigm means one expensive pretraining run amortizes across every downstream task; fine-tuning needs only a tiny head, a small learning rate, and a few epochs.
- **RoBERTa, ALBERT, and ELECTRA** show that the recipe (data, masking, parameter sharing, training objective) matters as much as the architecture.
- BERT excels at **understanding** tasks but cannot generate text. For that we turn to GPT (Part 6).

---

## Series Navigation

| Part  | Topic                                           | Link                                                   |
| ----- | ----------------------------------------------- | ------------------------------------------------------ |
| 1     | Introduction and Text Preprocessing             | [Read](/en/nlp-introduction-and-preprocessing/)        |
| 2     | Word Embeddings and Language Models             | [Read](/en/nlp-word-embeddings-lm/)                    |
| 3     | RNN and Sequence Modeling                       | [Read](/en/nlp-rnn-sequence-modeling/)                 |
| 4     | Attention Mechanism and Transformer             | [Previous](/en/nlp-attention-transformer/)             |
| **5** | **BERT and Pretrained Models (this article)**   |                                                        |
| 6     | GPT and Generative Models                       | [Read next](/en/nlp-gpt-generative-models/)            |
