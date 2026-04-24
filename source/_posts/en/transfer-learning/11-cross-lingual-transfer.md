---
title: "Transfer Learning (11): Cross-Lingual Transfer"
date: 2024-08-11 09:00:00
categories:
  - Transfer Learning
  - Machine Learning
tags:
  - Cross-Lingual
  - Multilingual Models
  - mBERT
  - XLM-R
  - Transfer Learning
series:
  name: "Transfer Learning"
  order: 11
  total: 12
lang: en
mathjax: true
description: "Derive cross-lingual transfer from bilingual word-embedding alignment to multilingual pretraining (mBERT, XLM-R). Covers zero-shot transfer, translate-train vs translate-test, pivot strategies, subword anchors, the high/low-resource gap, and cross-lingual prompts."
disableNunjucks: true
---

English has the labels. The world has 7,000+ languages. Cross-lingual transfer is what lets a sentiment classifier trained only on English IMDB reviews score Spanish tweets, what makes a question-answering model fine-tuned on SQuAD answer Hindi questions, and what allows a model that has never seen a single labeled Swahili sentence to do passable Swahili NER.

This post derives why that is even possible. We start from the bilingual-embedding alignment that motivated the field, walk through the multilingual pretraining recipe (mBERT, XLM-R) that made parallel data optional, and end with the practical playbook -- zero-shot vs translate-train vs translate-test, when to pick which, and where the wheels come off.

## What You Will Learn

- The shared-semantic-space hypothesis and the Procrustes alignment that makes it concrete
- Why mBERT transfers across languages with no parallel corpus, and what the subword vocabulary has to do with it
- XLM-R's three changes (data, sampling, scale) and what each one buys you
- The translate-train / translate-test / zero-shot trade-off and how to choose
- Pivot strategies for language pairs that have no direct labeled data
- Where the transfer gap comes from -- script, family, resource size -- and what to do about it

## Prerequisites

- BERT pretraining (MLM, [CLS] pooling) -- see Part 2
- Transfer-learning fundamentals -- Parts 1-6
- Word embeddings: dense vectors, cosine similarity

---

## Problem Setup

Let $\ell_s$ be a **source** language with abundant labeled data $\mathcal{D}_s = \{(x_i^{(s)}, y_i)\}$, and $\ell_t$ a **target** language for which we have no labels. The classical zero-shot cross-lingual objective is

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim \mathcal{D}_s}\,\mathcal{L}\!\left(f_\theta(x), y\right),
\qquad \text{evaluate on } \mathcal{D}_t.
$$

The number we report is the **transfer gap**

$$
\Delta(\ell_s \to \ell_t) = \operatorname{Acc}(\ell_s) - \operatorname{Acc}(\ell_t).
$$

A perfectly language-agnostic representation has $\Delta = 0$. In practice $\Delta$ ranges from a few points (English -> German) to twenty-plus (English -> Swahili), and most of this post is about why.

---

## The Shared Semantic Space

The hypothesis behind cross-lingual transfer is simple to state and surprisingly hard to prove:

> Different languages encode the same underlying concepts; with the right encoder $f_\theta$, semantically equivalent inputs land at nearby points in $\mathbb{R}^d$.

Formally, for sentences $s^{(\ell_1)}$ and $s^{(\ell_2)}$ with the same meaning,

$$
\bigl\| f_\theta\!\left(s^{(\ell_1)}\right) - f_\theta\!\left(s^{(\ell_2)}\right) \bigr\| \approx 0.
$$

![Multilingual embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig1_embedding_space.png)

The figure above is the cartoon: each cluster is one concept, and the three markers inside it are the same word in three languages. If your encoder achieves something like this, a classifier head trained on the English markers will fire on the Chinese and French ones too -- you never had to translate.

### Bilingual word-embedding alignment

Before multilingual Transformers, this hypothesis was tested on **static word embeddings**. Train word2vec on English, train it again on French, and you get two unrelated embedding matrices $\mathbf{X}_s, \mathbf{X}_t \in \mathbb{R}^{n \times d}$ (rows aligned via a small bilingual dictionary). The question is: does there exist a single linear map taking one space onto the other?

The answer turned out to be yes, and the map should be **orthogonal** -- rotations and reflections preserve dot products, hence preserve the geometric relationships word2vec learned. This is the **Orthogonal Procrustes** problem:

$$
\mathbf{W}^* = \arg\min_{\mathbf{W}} \bigl\|\mathbf{X}_s\mathbf{W} - \mathbf{X}_t\bigr\|_F^2
\quad\text{s.t.}\quad \mathbf{W}^\top \mathbf{W} = \mathbf{I}.
$$

It has a closed-form solution. Compute the SVD $\mathbf{X}_t^\top \mathbf{X}_s = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, then $\mathbf{W}^* = \mathbf{V}\mathbf{U}^\top$. Conneau et al. (2018) later showed you can drop the dictionary entirely and recover $\mathbf{W}$ adversarially, which is what made unsupervised bilingual lexicons possible.

The Procrustes story matters because **multilingual Transformers can be read as doing this alignment implicitly, end-to-end, in a much higher-dimensional space**.

---

## Multilingual Pretraining

![mBERT vs XLM-R](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig2_xlmr_architecture.png)

### mBERT -- the "it just works" baseline

Devlin et al.'s original recipe: take BERT-base, train it on the concatenation of Wikipedia in 104 languages with a single shared 110K-WordPiece vocabulary, masked-language-modeling objective only, **no parallel data, no language IDs**.

The mystery is why this works at all. Pires et al. (2019) and Wu & Dredze (2020) traced it to three mechanisms:

1. **Anchor tokens.** Numbers, punctuation, named entities, and English loanwords (`COVID`, `OK`, `Internet`) appear identically across many languages. When the same token shows up in many language contexts, its embedding is forced toward a language-agnostic centroid.
2. **Subword overlap.** WordPiece splits cognates into shared pieces. `international` (en) and `internationale` (fr) share `inter` + `national`. Two languages that share thousands of subwords are indirectly forced into the same embedding subspace.
3. **Deep parameter sharing.** All 12 layers are shared across all 104 languages. The cheapest representation for the model -- in the bits-per-token sense -- is one that re-uses circuits across languages, so it does.

Strikingly, *removing the anchor tokens at training time only drops cross-lingual NER accuracy by ~5 points*. The bigger driver is subword overlap.

### XLM-R -- scale, balance, more scale

Conneau et al. (2020) asked: what if we throw away the cute design choices and just scale? XLM-R changes three things relative to mBERT:

| Knob              | mBERT                 | XLM-R                       |
|-------------------|-----------------------|-----------------------------|
| Corpus            | Wikipedia (~13 GB)    | CommonCrawl (2.5 TB, ~200x) |
| Vocabulary        | WordPiece, 110K       | SentencePiece, 250K         |
| Sampling          | exponential, ad-hoc   | $p_\ell \propto n_\ell^{0.7}$ |
| Parameters        | 110M                  | 270M (base) / 550M (large)  |

The sampling change matters more than it looks. Without it, the model sees English orders of magnitude more often than Swahili and the Swahili representations rot. With $\alpha = 0.7$ the high-resource languages still dominate, but the tail gets enough budget to learn useful subword statistics. XLM-R-large beats mBERT by roughly **+10 average XNLI accuracy** -- driven mostly by gains on the long-tail languages.

### Why subwords are the hidden hero

![Subword tokenization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig5_subword_tokenization.png)

Look at how the same concept tokenizes across four languages. `inter` and `national` appear in the English, French, and German splits and share embeddings; only Chinese, with no script overlap, gets disjoint tokens. Cross-lingual transfer is essentially "free" between languages that share script and morphology, and progressively harder as that overlap drops. Concretely:

- en -> de transfer is easy: shared Latin script, ~30% subword overlap on Wikipedia.
- en -> zh requires the model to learn the alignment from context alone -- harder, and the gap shows.
- en -> sw is bottlenecked by *how much Swahili the model saw at all*, more than by the alignment.

---

## Where the Transfer Gap Comes From

![Zero-shot NER across 10 languages](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig3_zero_shot_ner.png)

The figure plots zero-shot NER F1 on WikiAnn. Three patterns show up consistently:

1. **Same family beats different family.** German, Dutch, Spanish, French (Indo-European, Latin script) all sit near 80 F1 with XLM-R-base. Arabic and Chinese sit ten points lower despite having more pretraining data than Dutch.
2. **Resource matters, but with diminishing returns.** Going from 1 GB of Swahili pretraining text to 50 GB of Arabic moves the needle, but going from 50 GB to 200 GB of German is almost a no-op.
3. **The oracle gap is bimodal.** For high-resource Indo-European languages, the in-language supervised oracle is only 5-10 points above zero-shot XLM-R-large. For low-resource non-Latin languages it can be 15-25.

The corpus-size view makes the resource effect explicit:

![Pretraining corpus vs zero-shot accuracy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig6_resource_curve.png)

The log-linear fit captures the high-resource plateau and the low-resource cliff. The green points show what balanced sampling buys: almost nothing for English, **+5-7 points for Swahili and Yoruba**.

---

## Transfer Strategies in Practice

There are three working recipes, and you should know all three before picking one.

![Translate-train vs translate-test vs zero-shot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig7_translate_vs_align.png)

### 1. Zero-shot direct transfer

Fine-tune XLM-R on English labels, deploy it on the target language as-is. The model's encoder is already aligned, so the classifier head -- which only sees `[CLS]` representations -- generalizes.

- Pros: cheapest by far, single model serves all languages, no MT dependency.
- Cons: leaves the most accuracy on the table when target is far from source.
- Use when: you support many target languages, or MT for the pair is unreliable.

### 2. Translate-train

Machine-translate your English training set into the target language(s), then fine-tune on the translated data (often jointly with the original English).

- Pros: typically **+2-5 accuracy points** vs zero-shot, especially for low-resource targets.
- Cons: cost scales linearly with the number of targets; degrades when MT is bad; you ship a model per language unless you train jointly.
- Use when: you have a small number of high-value target languages and decent MT.

### 3. Translate-test

Keep the English-fine-tuned model. At inference time, translate the target-language input back to English and run the English model.

- Pros: only one model to maintain; no per-target training cost.
- Cons: latency penalty (an MT call per request), and translation errors compound with model errors.
- Use when: you have to add a new target language and re-training isn't an option.

The bottom-bar grid in the figure summarizes the trade-off. In practice teams ship a **zero-shot baseline first**, then translate-train the top three target languages by traffic, then keep translate-test as a fallback for the long tail.

### Pivot strategies for hard language pairs

![Pivot strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig4_pivot_strategies.png)

For pairs where neither direction has decent MT (e.g., Yoruba -> Hausa), pivot through a high-resource language. The three patterns above generalize most of what's deployed in industry:

- **Source-pivot (translate-test through English).** Yoruba input -> English MT -> English model. Works as long as Yoruba->English MT is at least serviceable.
- **Target-pivot (translate-train through English).** Take English labels, MT them to Hausa, train. Works when English->Hausa MT is good.
- **Multi-source ensemble.** Train one classifier per source language (English, Spanish, German), average their predictions on the target. Often the strongest option when you have several source languages with different inductive biases.

---

## Cross-Lingual Prompting and Code-Switching

Two ideas that show up in modern systems and are worth knowing.

**Language-agnostic continuous prompts.** Instead of writing a prompt template per language, prepend $m$ learnable prefix vectors $\mathbf{P} = [\mathbf{p}_1, \dots, \mathbf{p}_m]$ to every input and train them on multiple languages simultaneously:

$$
\min_{\mathbf{P}} \; \sum_{\ell} \mathbb{E}_{(x,y)\sim \mathcal{D}_\ell}\,\mathcal{L}\!\left(f_\theta\!\left([\mathbf{P}; x^{(\ell)}]\right), y\right).
$$

The frozen encoder stays cross-lingual; the prompt learns a task-specific anchor that doesn't favor any single language.

**Code-switching augmentation.** During training, randomly replace a fraction of source-language words with their target-language translations while preserving syntax. This teaches the encoder to treat a token's *meaning* as more important than its *language*, and reliably improves robustness on code-mixed benchmarks (GLUECoS, LinCE) by 5-10 points.

---

## Implementation Sketch

A minimal zero-shot cross-lingual classifier on top of mBERT or XLM-R looks like this:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossLingualClassifier(nn.Module):
    """[CLS]-pooled classifier on a frozen-or-finetuned multilingual encoder."""

    def __init__(self, model_name: str = "xlm-roberta-base",
                 num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]      # [CLS] vector
        return self.head(cls)


# Train: feed English (input_ids, labels), backprop through encoder + head.
# Evaluate: same model, same tokenizer, but feed Chinese / French inputs.
# The encoder's shared subword space and aligned [CLS] do the heavy lifting.
```

Practical notes that matter more than the code:

- Use the **same tokenizer at train and test** -- mismatched tokenization is the most common silent bug.
- For zero-shot, **don't freeze the encoder**: full fine-tuning on the source language consistently beats frozen + linear probe by 3-5 points.
- For translate-train, mix English and translated data 1:1 in the same batch; pure-translated training degrades on the source language without helping the target.

---

## Q&A

**Q1: mBERT has no parallel corpus -- why does cross-lingual work?**
Three forces line up: anchor tokens (numbers, punctuation, loanwords) shared across languages, subword overlap on cognates that wires the embedding tables together, and deep parameter sharing that makes language-agnostic features the cheapest representation. Removing any one of these degrades transfer; removing all three breaks it.

**Q2: How do I pick a source language?**
Default to whichever language has the most labels (almost always English). If that gives a poor transfer gap, try a closer-family source: for a Spanish target, fine-tuning on Italian or Portuguese labeled data often beats English. For a Swahili target, English remains the right call simply because no other language has the labels.

**Q3: XLM-R or mBERT?**
XLM-R, unless you are inference-bound on a small device. XLM-R-base (270M) costs roughly 1.5x mBERT (110M) at inference and is 5-10 points better on every long-tail language. XLM-R-large is 3x and another 2-4 points on top.

**Q4: Where are the theoretical limits?**
Transfer quality is bounded above by the mutual information between source and target distributions in the encoder's representation space. Empirically: same-family Indo-European pairs cap out with a 3-5 point gap; cross-family pairs (English -> Chinese, English -> Arabic) plateau around 8-12; very low-resource targets (Yoruba, Swahili) are limited by how much of the language the encoder ever saw, not by the alignment.

**Q5: Translate-train always wins on accuracy -- why ever ship zero-shot?**
Cost. Translate-train requires per-language MT (latency at training time, but you also need to maintain the MT system) and you typically end up shipping a model per target language. Zero-shot is one model serving all 100 languages. The right answer is usually a hybrid: zero-shot baseline + translate-train on the top traffic languages.

**Q6: My target is a low-resource language -- what's the order of operations?**
(a) Try XLM-R-large zero-shot. (b) If accuracy is unacceptable, do **adaptive pretraining**: continue MLM on monolingual target-language text for a few epochs before fine-tuning. (c) Add translate-train on whatever MT pair is least bad. (d) If you can collect even 100 labeled target-language examples, mix them in -- few-shot data is disproportionately effective on top of zero-shot transfer.

---

## References

- Conneau, A., Lample, G., Ranzato, M.A., et al. (2018). Word translation without parallel data. *ICLR*.
- Devlin, J., Chang, M.W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. *NAACL*.
- Pires, T., Schlinger, E., Garrette, D. (2019). How multilingual is multilingual BERT? *ACL*.
- Wu, S., Dredze, M. (2020). Are all languages created equal in multilingual BERT? *RepL4NLP*.
- Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale (XLM-R). *ACL*.
- Hu, J., Ruder, S., Siddhant, A., et al. (2020). XTREME: A massively multilingual multi-task benchmark. *ICML*.
- Lauscher, A., Ravishankar, V., Vulic, I., Glavas, G. (2020). From zero to hero: On the limitations of zero-shot language transfer. *EMNLP*.

---

## Series Navigation

- Previous: [Part 10 -- Continual Learning](/en/transfer-learning-10-continual-learning/)
- Next: [Part 12 -- Industrial Applications](/en/transfer-learning-12-industrial-applications-and-best-practices/)
- [View all 12 parts in this series](/tags/Transfer-Learning/)
