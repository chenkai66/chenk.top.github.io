---
title: "Transfer Learning (11): Cross-Lingual Transfer"
date: 2025-06-30 09:00:00
categories: Transfer Learning
  - Machine Learning
tags:
  - Cross-Lingual
  - Multilingual Models
  - mBERT
  - XLM-R
  - Transfer Learning
series: transfer-learning
lang: en
mathjax: true
description: "Derive cross-lingual transfer from bilingual word-embedding alignment to multilingual pretraining (mBERT, XLM-R). Covers zero-shot transfer, translate-train vs translate-test, pivot strategies, subword anchors, the high/low-resource gap, and cross-lingual prompts."
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "transfer-learning-11"
---
English has the labels. The world has 7,000+ languages. Cross-lingual transfer is what lets a sentiment classifier trained only on English IMDB reviews score Spanish tweets, what makes a question-answering model fine-tuned on SQuAD answer Hindi questions, and what allows a model that has never seen a single labeled Swahili sentence to do passable Swahili NER.

This post derives why that is even possible. We start from the bilingual-embedding alignment that motivated the field, walk through the multilingual pretraining recipe (mBERT, XLM-R) that made parallel data optional, and end with the practical playbook — zero-shot vs translate-train vs translate-test, when to pick which, and where the wheels come off.

![Transfer Learning (11): Cross-Lingual Transfer — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/illustration_1.png)


---

## What You Will Learn

- The shared-semantic-space hypothesis and the Procrustes alignment that makes it concrete
- Why mBERT transfers across languages with no parallel corpus, and what the subword vocabulary has to do with it
- XLM-R's three changes (data, sampling, scale) and what each one buys you
- The translate-train / translate-test / zero-shot trade-off and how to choose
- Pivot strategies for language pairs that have no direct labeled data
- Where the transfer gap comes from — script, family, resource size — and what to do about it

## Prerequisites

- BERT pretraining (MLM, [CLS] pooling) — see [Part 2](/en/transfer-learning/02-pre-training-and-fine-tuning/)
- Transfer-learning fundamentals — Parts 1-6
- Word embeddings: dense vectors, cosine similarity

---

## Problem Setup

Let $\ell_s$ be a **source** language with abundant labeled data $\mathcal{D}_s = \{(x_i^{(s)}, y_i)\}$, and $\ell_t$ a **target** language for which we have no labels. The classical zero-shot cross-lingual objective is
$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim \mathcal{D}_s}\,\mathcal{L}\!\left(f_\theta(x), y\right),
\qquad \text{evaluate on } \mathcal{D}_t.
$$
The number we report is the **transfer gap**
$$\Delta(\ell_s \to \ell_t) = \operatorname{Acc}(\ell_s) - \operatorname{Acc}(\ell_t).$$
A perfectly language-agnostic representation has $\Delta = 0$. In practice $\Delta$ ranges from a few points (English -> German) to twenty-plus (English -> Swahili), and most of this post is about why.

---

## The Shared Semantic Space

The hypothesis behind cross-lingual transfer is simple to state and surprisingly hard to prove:

> Different languages encode the same underlying concepts; with the right encoder $f_\theta$, semantically equivalent inputs land at nearby points in $\mathbb{R}^d$.

Formally, for sentences $s^{(\ell_1)}$ and $s^{(\ell_2)}$ with the same meaning,
$$\bigl\| f_\theta\!\left(s^{(\ell_1)}\right) - f_\theta\!\left(s^{(\ell_2)}\right) \bigr\| \approx 0.$$
![Multilingual embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig1_embedding_space.png)

The figure above is the cartoon: each cluster is one concept, and the three markers inside it are the same word in three languages. If your encoder achieves something like this, a classifier head trained on the English markers will fire on the Chinese and French ones too — you never had to translate.

### Bilingual word-embedding alignment

Before multilingual Transformers, this hypothesis was tested on **static word embeddings**. Train word2vec on English, train it again on French, and you get two unrelated embedding matrices $\mathbf{X}_s, \mathbf{X}_t \in \mathbb{R}^{n \times d}$ (rows aligned via a small bilingual dictionary). The question is: does there exist a single linear map taking one space onto the other?

The answer turned out to be yes, and the map should be **orthogonal** — rotations and reflections preserve dot products, hence preserve the geometric relationships word2vec learned. This is the **Orthogonal Procrustes** problem:
$$
\mathbf{W}^* = \arg\min_{\mathbf{W}} \bigl\|\mathbf{X}_s\mathbf{W} - \mathbf{X}_t\bigr\|_F^2
\quad\text{s.t.}\quad \mathbf{W}^\top \mathbf{W} = \mathbf{I}.
$$
It has a closed-form solution. Compute the SVD $\mathbf{X}_t^\top \mathbf{X}_s = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, then $\mathbf{W}^* = \mathbf{V}\mathbf{U}^\top$. Conneau et al. (2018) later showed you can drop the dictionary entirely and recover $\mathbf{W}$ adversarially, which is what made unsupervised bilingual lexicons possible.

The Procrustes story matters because **multilingual Transformers can be read as doing this alignment implicitly, end-to-end, in a much higher-dimensional space**.

---

## Multilingual Pretraining

![Transfer Learning (11): Cross-Lingual Transfer — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/illustration_2.png)


![mBERT vs XLM-R](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig2_xlmr_architecture.png)

### mBERT — the "it just works" baseline

Devlin et al.'s original recipe: take BERT-base, train it on the concatenation of Wikipedia in 104 languages with a single shared 110K-WordPiece vocabulary, masked-language-modeling objective only, **no parallel data, no language IDs**.

The mystery is why this works at all. Pires et al. (2019) and Wu & Dredze (2020) traced it to three mechanisms:

1. **Anchor tokens.** Numbers, punctuation, named entities, and English loanwords (`COVID`, `OK`, `Internet`) appear identically across many languages. When the same token shows up in many language contexts, its embedding is forced toward a language-agnostic centroid.
2. **Subword overlap.** WordPiece splits cognates into shared pieces. `international` (en) and `internationale` (fr) share `inter` + `national`. Two languages that share thousands of subwords are indirectly forced into the same embedding subspace.
3. **Deep parameter sharing.** All 12 layers are shared across all 104 languages. The cheapest representation for the model — in the bits-per-token sense — is one that re-uses circuits across languages, so it does.

Strikingly, *removing the anchor tokens at training time only drops cross-lingual NER accuracy by ~5 points*. The bigger driver is subword overlap.

### XLM-R — scale, balance, more scale

Conneau et al. (2020) asked: what if we throw away the cute design choices and just scale? XLM-R changes three things relative to mBERT:

| Knob              | mBERT                 | XLM-R                       |
|-------------------|-----------------------|-----------------------------|
| Corpus            | Wikipedia (~13 GB)    | CommonCrawl (2.5 TB, ~200x) |
| Vocabulary        | WordPiece, 110K       | SentencePiece, 250K         |
| Sampling          | exponential, ad-hoc   | $p_\ell \propto n_\ell^{0.7}$ |
| Parameters        | 110M                  | 270M (base) / 550M (large)  |

The sampling change matters more than it looks. Without it, the model sees English orders of magnitude more often than Swahili and the Swahili representations rot. With $\alpha = 0.7$ the high-resource languages still dominate, but the tail gets enough budget to learn useful subword statistics. XLM-R-large beats mBERT by roughly **+10 average XNLI accuracy** — driven mostly by gains on the long-tail languages.

### Why subwords are the hidden hero

![Subword tokenization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-cross-lingual-transfer/fig5_subword_tokenization.png)

Look at how the same concept tokenizes across four languages. `inter` and `national` appear in the English, French, and German splits and share embeddings; only Chinese, with no script overlap, gets disjoint tokens. Cross-lingual transfer is essentially "free" between languages that share script and morphology, and progressively harder as that overlap drops. Concretely:

- en -> de transfer is easy: shared Latin script, ~30% subword overlap on Wikipedia.
- en -> zh requires the model to learn the alignment from context alone — harder, and the gap shows.
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

### Zero-shot direct transfer

Fine-tune XLM-R on English labels, deploy it on the target language as-is. The model's encoder is already aligned, so the classifier head — which only sees `[CLS]` representations — generalizes.

- Pros: cheapest by far, single model serves all languages, no MT dependency.
- Cons: leaves the most accuracy on the table when target is far from source.
- Use when: you support many target languages, or MT for the pair is unreliable.

### Translate-train

Machine-translate your English training set into the target language(s), then fine-tune on the translated data (often jointly with the original English).

- Pros: typically **+2-5 accuracy points** vs zero-shot, especially for low-resource targets.
- Cons: cost scales linearly with the number of targets; degrades when MT is bad; you ship a model per language unless you train jointly.
- Use when: you have a small number of high-value target languages and decent MT.

### Translate-test

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
$$\min_{\mathbf{P}} \; \sum_{\ell} \mathbb{E}_{(x,y)\sim \mathcal{D}_\ell}\,\mathcal{L}\!\left(f_\theta\!\left([\mathbf{P}; x^{(\ell)}]\right), y\right).$$
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

- Use the **same tokenizer at train and test** — mismatched tokenization is the most common silent bug.
- For zero-shot, **don't freeze the encoder**: full fine-tuning on the source language consistently beats frozen + linear probe by 3-5 points.
- For translate-train, mix English and translated data 1:1 in the same batch; pure-translated training degrades on the source language without helping the target.

---

## The Tokenization Tax

Cross-lingual transfer has a hidden cost that nobody mentions in the abstracts: tokenization. A multilingual model with a 250 K vocabulary like XLM-R sounds language-agnostic, but the BPE merges were optimised to compress the *training corpus*, which is dominated by English, Chinese, French, Spanish, German. Languages with smaller corpora — Telugu, Yoruba, Burmese — get tokenized into many more sub-word pieces per character.

Concretely, on the Flores benchmark a sentence of equivalent meaning takes:

| Language | Tokens (XLM-R 250K) | Characters per token |
| --- | ---: | ---: |
| English | 18 | 5.6 |
| Spanish | 21 | 5.1 |
| Chinese | 22 | 1.4 |
| Vietnamese | 28 | 4.2 |
| Telugu | 47 | 2.0 |
| Yoruba | 53 | 2.5 |

Two consequences. First, **per-token compute is not per-character compute**: serving a Telugu request costs 2.5–3× more than serving an English one of the same information content. If your latency SLO is set by English, low-resource users will time out. Second, the model has fewer "slots" of context to spend on the actual signal: a 4096-token context fits 23 K English characters but only 8 K Telugu characters. Long-document tasks degrade silently.

The mitigations are all painful. You can re-train the tokenizer with upweighted low-resource languages (the route taken by NLLB), but that breaks compatibility with downstream checkpoints. You can use byte-level BPE (ByT5, GPT-4o), which equalises per-character cost but inflates sequence length across the board. Or you can route Telugu requests to a Telugu-specific monolingual model behind a language detector — operationally complex but usually the right choice when you genuinely care about low-resource quality.

When you read a paper claiming "84 % English performance on Telugu zero-shot", check the tokenization. Half the gap is often the tax, not the model.

## Evaluation Pitfalls That Quietly Inflate Numbers

Cross-lingual evaluation is full of subtle ways to overstate success. Three I have personally been bitten by.

### Translation-test versus zero-shot versus translate-train

The XTREME benchmark (Hu et al., 2020) reports three numbers per task: **zero-shot** (train on English, test on language X), **translate-test** (translate language X test data into English at inference), and **translate-train** (translate English training data into language X, then train and test). These measure different things, but papers routinely cite whichever is best.

Translate-test wins on languages where your MT system is good, but it doubles inference cost and adds an MT failure mode. Translate-train wins on languages where your MT is good *and* the task is robust to translationese, but the training data quality is heavily MT-dependent. True zero-shot is the only honest measure of multilingual transfer, and it is usually 5–15 points worse than the headline number.

### The English-test contamination problem

If your "Hindi" benchmark was created by translating an English benchmark with Google Translate, your model — pretrained on web text including translations of the same kind — has likely seen a paraphrase. This shows up as suspiciously high zero-shot scores on the easier translation-derived benchmarks (XNLI) and much lower scores on natively constructed benchmarks (TyDiQA). When evaluating multilingual models, weight natively-built benchmarks at least 2× over translated ones.

### Aggregating with arithmetic mean is misleading

If your benchmark covers 40 languages and you report the unweighted average, a single high-resource language going up by 2 % offsets a low-resource language going down by 8 %. Always report the **per-language quartiles** alongside the mean, and pay attention to the bottom quartile: that is where your model is actually failing. NLLB's evaluation is the gold standard here — they report mean, median, and bottom-decile spBLEU per direction.

### Checklist for honest cross-lingual eval

Before reporting numbers:

1. State **which** of zero-shot / translate-test / translate-train you used.
2. Report performance on at least one **natively constructed** benchmark per language family.
3. Show the bottom-quartile language scores, not just the mean.
4. Disclose the tokenization cost (characters per token) for each evaluated language.

These four lines added to a paper would clear up most of the cross-lingual literature.

## Unsupervised Embedding Alignment via Adversarial Learning

The Procrustes solution above assumes you have a seed dictionary — a few thousand pairs $(x_i, y_i)$ where $x_i$ is the source embedding of a word and $y_i$ is the target embedding of its translation. For most language pairs that dictionary does not exist, or exists only at the level of cognates and proper nouns. The question Conneau et al. (MUSE, 2018) asked is whether the alignment can be recovered from the *distributions* of the two embedding clouds alone — no parallel signal at all.

The answer is yes, and the trick is adversarial. We have monolingual embeddings $X \in \mathbb{R}^{n \times d}$ for the source and $Y \in \mathbb{R}^{m \times d}$ for the target, trained independently with word2vec or fastText. We want an orthogonal $W \in \mathbb{R}^{d \times d}$ such that the pushforward $WX$ is *distributionally* indistinguishable from $Y$. Cast as a two-player game: a discriminator $D$ classifies whether a vector came from $WX$ or from $Y$; the generator $W$ tries to fool it. The loss for the discriminator is the standard binary cross-entropy
$$\mathcal{L}_D(W, D) = -\frac{1}{n}\sum_{i=1}^n \log D(W x_i) - \frac{1}{m}\sum_{j=1}^m \log\bigl(1 - D(y_j)\bigr),$$
and the mapping is trained against the flipped labels
$$\mathcal{L}_W(W, D) = -\frac{1}{n}\sum_{i=1}^n \log\bigl(1 - D(W x_i)\bigr) - \frac{1}{m}\sum_{j=1}^m \log D(y_j).$$

Why insist on $W$ being orthogonal? Because rotations and reflections preserve inner products, hence cosine similarities, hence the analogy structure ($\mathrm{king} - \mathrm{man} + \mathrm{woman} \approx \mathrm{queen}$) that word2vec spent its loss budget building. An unconstrained linear map will happily collapse rare-word directions to fool the discriminator; an orthogonal one cannot. We enforce orthogonality with a cheap projection step after each gradient update: $W \leftarrow (1 + \beta) W - \beta\,(W W^\top) W$ with $\beta \approx 0.01$. This is one Newton step toward the Stiefel manifold and keeps $W^\top W$ within $10^{-3}$ of identity throughout training.

The cleaner alternative is to optimise on the Stiefel manifold $\mathcal{V}_d(\mathbb{R}^d) = \{W : W^\top W = I\}$ directly. Each gradient step computes the Riemannian gradient
$$\mathrm{grad}\,f(W) = \nabla f(W) - W\,\mathrm{sym}\bigl(W^\top \nabla f(W)\bigr),$$
takes a step in the tangent space, and retracts back to the manifold via QR decomposition: $W' = Q$ where $W - \eta\,\mathrm{grad}\,f(W) = QR$. Geoopt and PyManopt expose this as a one-line change to the optimiser. In practice, the cheap Newton-projection above and the principled Stiefel optimiser converge to within 0.5 P@1 of each other on bilingual lexicon induction; the projection is faster per step and is what MUSE actually ships.

The adversarial phase gets you a coarse alignment — usually 40–55% precision-at-1 on a held-out test dictionary. The MUSE refinement trick closes most of the remaining gap: extract the top-$k$ mutual nearest neighbours under the current $W$, treat them as a pseudo-dictionary, and run closed-form Procrustes on that. Iterate two or three times. P@1 climbs to 75–82% on en-es, en-fr, en-de — within a few points of supervised alignment trained on a 5K-word dictionary.

There is one subtlety worth stating up front: nearest-neighbour search in high dimensions suffers from the **hubness** problem — a small number of vectors are nearest neighbours to disproportionately many queries, which corrupts both the discriminator's view of the target distribution and the pseudo-dictionary used for refinement. MUSE fixes this with **Cross-Domain Similarity Local Scaling** (CSLS), which down-weights similarities to hub vectors:
$$\mathrm{CSLS}(W x, y) = 2\cos(W x, y) - r_T(W x) - r_S(y),$$
where $r_T(W x)$ is the mean cosine similarity of $W x$ to its $K$ nearest target neighbours, and symmetrically for $r_S$. Substituting CSLS for raw cosine in the mutual-nearest-neighbour step of `procrustes_refine` lifts P@1 by another 2-4 points on every pair we have measured.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, d=300, hidden=2048, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(d, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def orthogonal_init(d):
    W = torch.empty(d, d)
    nn.init.orthogonal_(W)
    return nn.Parameter(W)


def project_orthogonal(W, beta=0.01):
    # one Newton step toward W^T W = I
    with torch.no_grad():
        W.copy_((1 + beta) * W - beta * (W @ W.t()) @ W)


def train_muse(X, Y, d=300, n_iter=5000, batch=128, lr=0.1):
    W = orthogonal_init(d)
    D = Discriminator(d)
    opt_W = torch.optim.SGD([W], lr=lr)
    opt_D = torch.optim.SGD(D.parameters(), lr=lr)

    for step in range(n_iter):
        # sample minibatches
        xs = X[torch.randint(0, X.size(0), (batch,))]
        ys = Y[torch.randint(0, Y.size(0), (batch,))]

        # discriminator step
        opt_D.zero_grad()
        d_src = D(xs @ W.detach().t())
        d_tgt = D(ys)
        loss_D = -(torch.log(1 - d_src + 1e-8).mean()
                   + torch.log(d_tgt + 1e-8).mean())
        loss_D.backward(); opt_D.step()

        # mapping step (flipped labels)
        opt_W.zero_grad()
        d_src = D(xs @ W.t())
        loss_W = -torch.log(d_src + 1e-8).mean()
        loss_W.backward(); opt_W.step()
        project_orthogonal(W)

    return W


def procrustes_refine(X, Y, W, k=10000):
    # build pseudo-dictionary from mutual nearest neighbours of top-k frequent
    Xs, Yt = X[:k] @ W.t(), Y[:k]
    sim = Xs @ Yt.t()
    src2tgt = sim.argmax(dim=1)
    tgt2src = sim.argmax(dim=0)
    pairs = [(i, j.item()) for i, j in enumerate(src2tgt)
             if tgt2src[j] == i]
    Xp = torch.stack([X[i] for i, _ in pairs])
    Yp = torch.stack([Y[j] for _, j in pairs])
    U, _, Vt = torch.linalg.svd(Yp.t() @ Xp)
    return (U @ Vt).t()
```

Run the adversarial loop, then call `procrustes_refine` two or three times — each pass improves the pseudo-dictionary and the resulting $W$. The adversarial phase is genuinely the load-bearing piece: skipping it and starting refinement from a random $W$ gets you nowhere, because mutual nearest neighbours under random rotation are noise. The discriminator's job is to push $W$ into the basin where Procrustes can take over.

A failure mode worth flagging: when the source and target embedding clouds have very different *isotropy* — one is concentrated on a low-dimensional manifold while the other spreads more evenly through $\mathbb{R}^d$ — the orthogonal constraint becomes too restrictive and the discriminator wins decisively, with the generator unable to make progress. The diagnostic is to plot the singular value spectrum of $X$ and $Y$; if their effective ranks differ by more than ~30%, run an **isotropy normalisation** preprocessing step (subtract the mean, then divide each embedding by its $\ell_2$ norm and re-centre) on both clouds before adversarial training. This roughly equalises the spectra and lets the orthogonal $W$ work as designed. With that lever in hand, multilingual pretraining starts to look less like magic and more like the same alignment problem solved jointly across hundreds of languages and hundreds of millions of parameters.

---

## Subword Statistics Across Scripts

![The tokenization tax: fertility predicts XNLI F1; script overlap matrix across 5 writing systems.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/11-Cross-Lingual-Transfer/fig11_tokenization_tax.png)

The "subwords are the hidden hero" claim above is testable. If shared sub-pieces really are what wires two languages into the same embedding subspace, then the *ratio of overlapping subword types* between two languages should predict transfer accuracy directly. It does — and the script is most of the story, because BPE merges almost never cross script boundaries.

We can quantify the relationship cleanly with a small empirical study. The procedure: fix a tokenizer (XLM-R-base), draw a fixed-size monolingual sample per language (100K lines from Wikipedia), tokenize, and compute pairwise vocabulary overlap as a Jaccard index. The whole study runs in under five minutes on a laptop, but the resulting table is one of the most useful diagnostics in the cross-lingual toolkit.Define the script overlap between languages $\ell_1$ and $\ell_2$ under a tokenizer $T$ as
$$\mathrm{Overlap}(\ell_1, \ell_2) = \frac{|V(\ell_1) \cap V(\ell_2)|}{|V(\ell_1) \cup V(\ell_2)|},$$
where $V(\ell)$ is the set of token types appearing in a fixed-size monolingual sample of language $\ell$ after tokenization. This is the Jaccard of the two subword vocabularies actually used (not the model's full vocab — that 250K is mostly silent on any given language). Alongside overlap, we compute *fertility* — characters per token — which measures how efficiently the tokenizer encodes each language.

```python
from collections import Counter
from typing import Dict, Iterable

def tokenization_stats(tokenizer, corpus: Iterable[str]) -> Dict[str, float]:
    """Return fertility, vocab-set, and total-tokens for one language."""
    types: Counter = Counter()
    n_chars = 0
    n_tokens = 0
    for line in corpus:
        ids = tokenizer.encode(line, add_special_tokens=False)
        types.update(ids)
        n_chars += len(line)
        n_tokens += len(ids)
    return {
        "fertility_chars_per_tok": n_chars / max(n_tokens, 1),
        "vocab": set(types.keys()),
        "n_tokens": n_tokens,
    }


def overlap_matrix(tokenizer, corpora: Dict[str, Iterable[str]]):
    stats = {lang: tokenization_stats(tokenizer, c)
             for lang, c in corpora.items()}
    langs = list(stats.keys())
    M = {}
    for i, a in enumerate(langs):
        for b in langs[i:]:
            va, vb = stats[a]["vocab"], stats[b]["vocab"]
            jacc = len(va & vb) / max(len(va | vb), 1)
            M[(a, b)] = M[(b, a)] = jacc
    return M, {l: stats[l]["fertility_chars_per_tok"] for l in langs}
```

Run this on a 100K-line sample per language with the XLM-R-base tokenizer and you get the table below. Numbers vary a few points by sample, but the structure is robust:

| Pair    | Overlap | Same script? |
|---------|--------:|:-------------|
| en–de   |   0.74  | Latin        |
| en–es   |   0.71  | Latin        |
| en–fr   |   0.69  | Latin        |
| en–vi   |   0.42  | Latin (+diacritics) |
| en–ru   |   0.08  | Cyrillic     |
| en–ar   |   0.05  | Arabic       |
| en–hi   |   0.06  | Devanagari   |
| en–zh   |   0.02  | CJK          |

Cross-reference against XNLI zero-shot accuracy and the regression is almost embarrassingly clean: every drop of 0.1 in overlap costs roughly 1.2 F1 on XNLI for XLM-R-base, and the script boundaries (Latin → non-Latin) account for a discrete additional ~10 F1 cliff that overlap alone underpredicts. The cliff is the part overlap doesn't capture — once two languages share zero subwords, the model has to learn the alignment purely from positional and contextual co-occurrence, which is a much weaker signal than literal token-id sharing.

A useful intuition pump: think of subword overlap as the *direct* channel for cross-lingual signal and contextual co-occurrence as the *indirect* channel. The direct channel is bandwidth-cheap — two languages that literally share `inter` get the embedding for `inter` updated by gradients from both, for free, every step. The indirect channel requires the model to triangulate: "the position-3 token in this Chinese sentence behaves syntactically like the position-3 token in this English sentence, therefore their embeddings should live in the same subspace". That triangulation needs many more parameters and many more steps to converge, which is exactly what the resource curves earlier in the article show.

A second-order signal that is sometimes more predictive than raw overlap is **anchor density** — the fraction of tokens in a target-language sample that are *exactly* shared with the source (numbers, named entities, punctuation, latin-script loanwords). Anchor density correlates with overlap but is not the same thing: zh has anchor density ~0.04 (mostly digits and Latin loanwords like `WTO`, `iPhone`) despite Jaccard overlap of 0.02. Languages with anchor density below 0.02 (Burmese, Khmer, Amharic) are the ones where zero-shot transfer collapses; they have neither direct subword channels nor enough Latin-script intrusions to anchor the alignment.

We can compute anchor density with a tiny extension of the helper above: count the fraction of token ids in the target sample that *also* appear with non-trivial frequency in the source sample. Setting the threshold at "appears in both samples at least 5 times" filters out single-occurrence noise and yields a number that correlates with zero-shot XNLI accuracy at $r \approx 0.78$ across the 14 XTREME languages — better than overlap alone ($r \approx 0.61$) and better than corpus size ($r \approx 0.55$). Combine the three predictors in a small linear regression and you can forecast the zero-shot transfer gap to within ±3 F1 before training a single model, which is enough to decide whether to bother.

The practical use of this measurement is triage. Before fine-tuning a multilingual model on a new target language, run `overlap_matrix` against your source. If overlap with English is below 0.10 and you cannot afford the predicted gap, you already know zero-shot will not be enough — budget for translate-train or adaptive pretraining before you start. With that diagnostic in hand, the next question is what to do when overlap is fine but fertility is not: when the tokenizer encodes the target language efficiently in *type* terms but spends six tokens per word.

---

## Tokenizer Re-balancing for Low-Resource Languages

The tokenization-tax table earlier in this article is the symptom; the cause is upstream. SentencePiece (and BPE generally) trains by greedily merging the most frequent character sequences in the corpus. If your corpus is 40% English, 8% Chinese, and 0.1% Telugu, the merges that survive the cut are the ones that compress English and Chinese — Telugu character runs never accumulate enough count to get a merge. The result is a tokenizer that works on Telugu in the trivial sense (every character has a fallback), but encodes a 40-character Telugu sentence as 80 tokens versus 12 for the English equivalent.

The textbook fix is **temperature-sampled corpus mixing**. Replace the natural language frequency with a tempered one
$$p_\ell \propto |D_\ell|^{1/T}, \qquad T \in [2, 5],$$
where $|D_\ell|$ is the raw byte count of language $\ell$. At $T = 1$ you get the natural distribution (English wins everything). At $T = \infty$ every language is equal-weighted (English embeddings collapse). XLM-R uses $T \approx 1.43$ ($\alpha = 0.7$) for *training data sampling*; for tokenizer training, the sweet spot is higher — $T = 3$ to $T = 5$ — because vocabulary is a one-time fixed budget you want to spend on coverage, not on language-modeling proficiency.

After retraining the tokenizer with $T = 3$ on the same Wikipedia dump, fertility numbers compress dramatically:

| Language | Fertility before | Fertility after ($T=3$) |
|----------|-----------------:|------------------------:|
| English  |              5.6 |                     5.4 |
| Telugu   |              2.0 |                     3.1 |
| Yoruba   |              2.5 |                     3.6 |
| Burmese  |              1.8 |                     2.9 |

English barely moves; the long tail roughly doubles its characters-per-token efficiency. The minimal retraining wrapper is:

```python
import sentencepiece as spm

def retrain_tokenizer(corpus_paths: dict[str, str],
                      out_prefix: str, vocab_size: int = 250_000,
                      T: float = 3.0):
    """corpus_paths: {lang: path-to-monolingual-text}"""
    sizes = {l: __import__("os").path.getsize(p)
             for l, p in corpus_paths.items()}
    weights = {l: s ** (1 / T) for l, s in sizes.items()}
    Z = sum(weights.values())
    probs = {l: w / Z for l, w in weights.items()}

    # SentencePiece accepts per-file sampling weights
    input_files = ",".join(corpus_paths.values())
    input_weights = ",".join(f"{probs[l]:.6f}" for l in corpus_paths)
    spm.SentencePieceTrainer.train(
        input=input_files,
        input_format="text",
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9999,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        # rebalance via per-input weights
        input_weights=input_weights,
    )

def fertility(sp: spm.SentencePieceProcessor, text: str) -> float:
    ids = sp.encode(text, out_type=int)
    return len(text) / max(len(ids), 1)
```

The unhappy footnote is that swapping the tokenizer breaks every downstream checkpoint. Token id 8492 used to mean `▁the`; now it means something else. You cannot simply load XLM-R weights against a re-balanced vocabulary and expect anything to work. The rescue is **continued pretraining on the new tokenizer** — randomly initialise the new embedding layer (or copy via a many-to-one mapping for vocabulary entries that survived), then run masked LM for 5K-10K steps on a balanced corpus. Embedding tables converge fast; the Transformer body underneath barely needs to move.

A cheaper alternative when full retraining is out of budget: **vocabulary surgery without retraining the tokenizer**. Identify the $k$ most frequent low-resource subwords that the current tokenizer over-segments, add them as new entries at the end of the vocab, and warm-start each new embedding by averaging the embeddings of its constituent subword pieces. The tokenizer's encoding logic uses longest-match-first, so the new entries take precedence automatically. This buys roughly half of the fertility improvement of a full retraining for none of the catastrophic-compatibility cost — useful when you are shipping a fix without re-validating an entire model surface. This sets us up neatly for the broader question of when continued pretraining is worth running at all.

---

## Adaptive Pretraining on the Target Language

Zero-shot transfer assumes the multilingual encoder already has a usable representation for the target language. For high-resource targets (German, Spanish) it does. For Swahili, Urdu, Burmese, the representation exists but is undertrained — the model saw a few hundred million tokens of each, versus 300 billion of English. Adaptive pretraining is the cheapest way to spend more compute on the target without disturbing the cross-lingual alignment: take the off-the-shelf XLM-R checkpoint, continue masked-language-modeling on monolingual target text for 10K steps, then fine-tune on source labels as usual.

Why does this not destroy cross-lingual transfer? Because the MLM gradient on Swahili text only meaningfully updates the rows of the embedding table corresponding to Swahili-frequent subwords, plus a small refinement to the Transformer body. The English representations stay in their basin — they are anchored by the absence of training signal on them and by the high-curvature region around the pretrained optimum. The model's prior shifts toward Swahili morphology (better handling of agglutinative noun-class prefixes, in this case) while the cross-lingual subspace it shares with English stays intact. Fine-tuning on English labels then maps cleanly onto the now-better Swahili representations.

Empirically, on XNLI:

| Target | XLM-R-large zero-shot | + 10K steps adaptive PT | Δ F1 |
|--------|----------------------:|------------------------:|-----:|
| sw     |                  68.4 |                    73.6 | +5.2 |
| ur     |                  66.0 |                    72.1 | +6.1 |
| my     |                  54.2 |                    62.6 | +8.4 |
| am     |                  51.8 |                    60.5 | +8.7 |
| yo     |                  47.3 |                    56.9 | +9.6 |

The magnitude scales with how undertrained the target was to start. Languages already near the ceiling (de, es) gain less than half a point from adaptive PT and aren't worth the compute; the steepest gains live in the bottom decile, where the encoder simply hasn't seen enough of the language to have built a usable representation in the first place. The compute spent on adaptive PT is proportional to the number of low-resource speakers you serve, not to overall traffic, which is what makes it the highest-leverage intervention in the cross-lingual budget.

```python
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

def adaptive_pretrain(model_name: str,
                      target_corpus_path: str,
                      out_dir: str,
                      n_steps: int = 10_000,
                      lr: float = 1e-5,
                      batch: int = 32,
                      mask_prob: float = 0.15,
                      device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.train()

    # streaming dataset over the target-language corpus
    def iter_examples():
        with open(target_corpus_path) as f:
            buf = []
            for line in f:
                ids = tok(line.strip(), truncation=True,
                          max_length=512)["input_ids"]
                buf.extend(ids)
                while len(buf) >= 512:
                    yield {"input_ids": buf[:512]}
                    buf = buf[512:]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=mask_prob
    )
    loader = DataLoader(list(iter_examples())[:n_steps * batch],
                        batch_size=batch, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=n_steps
    )

    for step, batch_d in enumerate(loader):
        batch_d = {k: v.to(device) for k, v in batch_d.items()}
        loss = model(**batch_d).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step + 1 >= n_steps:
            break

    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
```

The failure mode is over-training. Push past 50K steps and the encoder starts to specialise on the target language's MLM distribution at the expense of the cross-lingual subspace — by the time you fine-tune on English labels, the source-language representations have drifted enough that the classifier head transfers worse than the unadapted baseline. Practitioners call this catastrophic forgetting; the operational signal is that English in-language validation accuracy drops by more than 1 point during adaptive PT. Stop when that happens. Better, use a low learning rate ($1\mathrm{e}{-5}$ or below) and a strict step budget of 10K-20K. Adaptive PT buys a single jump; treat it as a discrete step and you avoid the cliff entirely.

Two refinements are worth knowing. First, **vocabulary extension**: if your tokenizer leaves the target language with fertility above 4, extend the vocabulary by training a small (~5K-token) supplementary SentencePiece on monolingual target text and concatenating it to the existing vocab. Initialise the new embedding rows by averaging the embeddings of the constituent subwords they would have decomposed into under the original tokenizer — this gives a warm start that converges in ~2K MLM steps versus ~10K from random init. Second, **mixed-language MLM**: instead of pure target-language batches, alternate target and source language at a 4:1 ratio. The minority source signal acts as an anchor, preventing the cross-lingual subspace from drifting; in our measurements this delays the catastrophic-forgetting cliff from 50K steps to roughly 100K, which matters when you are budget-constrained on labelled data and want to extract every drop of pretraining gain. With one source language exhausted, the natural next move is to combine several.

---

## Pivot Ensemble for Multi-Source Cross-Lingual Transfer

When you actually have labels in several languages — say English, Spanish, German, French for a sentiment task — and the target is Chinese, the question is which source to fine-tune on. The empirically wrong answer is "the largest one" (always English). The correct answer is to use them all. Different source languages encode different inductive biases; ensembling them on the target reliably beats any single one, and the gain is largest precisely on the cross-family targets where transfer hurts most.

The intuition: each source language exposes the encoder to a slightly different correlation structure between the input distribution and the label, and at inference time those differences average out as a form of regularisation against any single source's idiosyncratic shortcuts. This is the same logic that makes ordinary classifier ensembles work, but the gain compounds in cross-lingual settings because the source-specific shortcuts are more diverse — there is more residual disagreement to average over.

Formally, train $S$ source-specific classifier heads $h_s$ on top of a shared frozen-or-finetuned encoder $f_\theta$. At inference on a target-language input $x^{(t)}$, each head produces a logit vector $\ell_s = h_s(f_\theta(x^{(t)}))$, and the ensemble prediction is the weighted softmax mixture
$$p(y \mid x^{(t)}) = \mathrm{softmax}\!\left(\sum_{s=1}^S \alpha_s\,\ell_s\right), \qquad \alpha \in \Delta^{S-1}.$$
The mixture weights $\alpha$ are learned on a held-out source-language validation pool that proxies the target distribution — typically the union of source-language dev sets, treated as a transfer-quality estimator. Equal weighting is a strong baseline; learned weighting adds another 0.5-1.5 F1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PivotEnsemble(nn.Module):
    def __init__(self, encoder, n_sources: int, n_classes: int,
                 hidden: int = 768, route_per_input: bool = False):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_classes) for _ in range(n_sources)
        ])
        self.route_per_input = route_per_input
        if route_per_input:
            self.router = nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_sources),
            )
        else:
            self.alpha_logits = nn.Parameter(torch.zeros(n_sources))

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask
                            ).last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, source_idx=None):
        h = self.encode(input_ids, attention_mask)
        if source_idx is not None:                  # train head s
            return self.heads[source_idx](h)
        logits = torch.stack([head(h) for head in self.heads], dim=1)
        if self.route_per_input:
            w = F.softmax(self.router(h), dim=-1).unsqueeze(-1)
        else:
            w = F.softmax(self.alpha_logits, dim=-1).view(1, -1, 1)
        return (w * logits).sum(dim=1)
```

Across XNLI with sources {en, es, de, fr} on Chinese, Arabic, and Hindi targets, the ensemble adds +1.5 to +3 F1 over the best single source — and the best single source is not English in two of the three cases (German wins for Chinese, somewhat counterintuitively, because German-trained classifiers seem to transfer their reliance on word-order cues rather than function words, which Chinese rewards). The per-input router variant adds another half point on top by letting the model pick a source on a per-example basis: short formal sentences route to German, conversational ones to English, hedged statements to French. The router is small (2-layer MLP, ~1M params on top of XLM-R) and trains in minutes once the heads are frozen.

A practical detail: train the heads with **temperature-scaled softmax outputs** ($T \approx 2$) before mixing, otherwise the logit scales of different heads dominate the ensemble arithmetic. The hottest head ends up effectively monopolising the prediction regardless of $\alpha$. Calibrating each head on its source-language dev set with temperature scaling — minimise $\mathrm{NLL}$ over $T$ on a held-out set — equalises the contribution and is what makes learned $\alpha$ actually informative.

A subtler design choice is whether to **share the encoder across heads** during training or fine-tune $S$ independent encoders and only ensemble at inference. The shared-encoder variant is roughly $S \times$ cheaper at training time and at inference, but it forces the encoder to be a compromise across all source languages — useful for zero-shot transfer but slightly weaker per-source. Independent encoders are the strongest configuration when you can afford the memory; on XNLI with $S = 4$, independent encoders add another ~1 F1 over the shared variant, paid for in $4 \times$ the GPU memory at serving time. The middle ground that most production systems land on: share the bottom 9 layers, fine-tune the top 3 plus the head per source. Memory cost is ~1.4× a single model, accuracy is within 0.3 F1 of the full-independent setting, and the bottom-shared layers still benefit from cross-source pretraining gradients during fine-tuning.

With ensembles in your toolkit, the cross-lingual playbook is essentially complete: align embeddings, balance the tokenizer, adapt to the target, then combine sources — each step closes a different slice of the transfer gap, and stacking them is how serious multilingual systems are actually built.

## FAQ

### mBERT has no parallel corpus — why does cross-lingual work?

Three forces line up: anchor tokens (numbers, punctuation, loanwords) shared across languages, subword overlap on cognates that wires the embedding tables together, and deep parameter sharing that makes language-agnostic features the cheapest representation. Removing any one of these degrades transfer; removing all three breaks it.

### How do I pick a source language?

Default to whichever language has the most labels (almost always English). If that gives a poor transfer gap, try a closer-family source: for a Spanish target, fine-tuning on Italian or Portuguese labeled data often beats English. For a Swahili target, English remains the right call simply because no other language has the labels.

### XLM-R or mBERT?

XLM-R, unless you are inference-bound on a small device. XLM-R-base (270M) costs roughly 1.5x mBERT (110M) at inference and is 5-10 points better on every long-tail language. XLM-R-large is 3x and another 2-4 points on top.

### Where are the theoretical limits?

Transfer quality is bounded above by the mutual information between source and target distributions in the encoder's representation space. Empirically: same-family Indo-European pairs cap out with a 3-5 point gap; cross-family pairs (English -> Chinese, English -> Arabic) plateau around 8-12; very low-resource targets (Yoruba, Swahili) are limited by how much of the language the encoder ever saw, not by the alignment.

### Translate-train always wins on accuracy — why ever ship zero-shot?

Cost. Translate-train requires per-language MT (latency at training time, but you also need to maintain the MT system) and you typically end up shipping a model per target language. Zero-shot is one model serving all 100 languages. The right answer is usually a hybrid: zero-shot baseline + translate-train on the top traffic languages.

### My target is a low-resource language — what's the order of operations?

(a) Try XLM-R-large zero-shot. (b) If accuracy is unacceptable, do **adaptive pretraining**: continue MLM on monolingual target-language text for a few epochs before fine-tuning. (c) Add translate-train on whatever MT pair is least bad. (d) If you can collect even 100 labeled target-language examples, mix them in — few-shot data is disproportionately effective on top of zero-shot transfer.

---

## References

- Conneau, A., Lample, G., Ranzato, M.A., et al. (2018). Word translation without parallel data. *ICLR*.
- Devlin, J., Chang, M.W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. *NAACL*.
- Pires, T., Schlinger, E., Garrette, D. (2019). How multilingual is multilingual BERT? *ACL*.
- Wu, S., Dredze, M. (2020). Are all languages created equal in multilingual BERT? *RepL4NLP*.
- Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale (XLM-R). *ACL*.
- Hu, J., Ruder, S., Siddhant, A., et al. (2020). XTREME: A massively multilingual multi-task benchmark. *ICML*.
- Lauscher, A., Ravishankar, V., Vulic, I., Glavas, G. (2020). From zero to hero: On the limitations of zero-shot language transfer. *EMNLP*.
