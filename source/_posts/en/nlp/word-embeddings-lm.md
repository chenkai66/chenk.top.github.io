---
title: "NLP Part 2: Word Embeddings and Language Models"
date: 2024-06-02 09:00:00
tags:
  - NLP
  - Word Embeddings
  - Deep Learning
categories: Natural Language Processing
series:
  name: "Natural Language Processing"
  part: 2
  total: 12
lang: en
mathjax: true
description: "Understand how Word2Vec, GloVe, and FastText turn words into vectors that capture meaning. Learn the math, train your own embeddings with Gensim, and connect embeddings to language models."
disableNunjucks: true
---

For decades, machines treated "king" and "queen" as unrelated symbols -- nothing more than two distinct slots in a vocabulary list. Then a single idea changed everything: what if every word lived in a continuous space, and meaning was just a *direction*? Once that idea took hold, models could compute

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

and the entire trajectory of NLP turned toward representation learning. This article walks through that turn -- from the failure of one-hot vectors, to Word2Vec's shallow networks, to the global statistics that GloVe exploits, to the subword n-grams that let FastText handle words it has never seen -- and finally connects embeddings to the language models that gave rise to them.

## What You Will Learn

- Why one-hot encoding fails and how dense embeddings fix it
- Skip-gram and CBOW: two ways to learn from local context
- Negative sampling: the trick that makes training feasible
- GloVe: learning from global co-occurrence statistics
- FastText: handling rare and out-of-vocabulary words with subwords
- How embeddings connect to language models and why neural LMs scale
- How to train, evaluate, and visualize embeddings with Gensim

**Prerequisites**: Part 1 (text preprocessing basics), basic linear algebra (dot products, matrix multiplication), and a working knowledge of softmax and stochastic gradient descent.

---

## From Sparse to Dense: Why Embeddings Matter

### The Problem with One-Hot Encoding

Given a vocabulary of $V$ words, one-hot encoding maps each word to a $V$-dimensional indicator vector:

$$\text{cat} = [1, 0, 0, 0, \ldots], \quad \text{dog} = [0, 1, 0, 0, \ldots]$$

Three properties make this representation a dead end:

- **Sparsity.** With $V = 50{,}000$, every vector is 99.998% zeros. Storage and compute are wasted on emptiness.
- **No similarity.** The dot product between any two distinct one-hot vectors is exactly zero. The model literally cannot tell that "cat" is closer to "dog" than to "quantum"; from its point of view, all non-identical words are equally different.
- **No generalisation.** A linear classifier on top of one-hot input has $V$ independent weights per class. Knowing that "movie" is positive teaches the model nothing about "film".

### The Embedding Solution

A word embedding maps each word to a dense vector of $d \ll V$ dimensions (typically 100--300):

$$\text{cat} = [0.21, -0.34, 0.78, \ldots], \quad \text{dog} = [0.18, -0.29, 0.71, \ldots]$$

Now $\text{cat} \cdot \text{dog} \gg \text{cat} \cdot \text{quantum}$, parameters are shared across semantically related words, and downstream models suddenly need orders of magnitude fewer examples to generalise. The challenge becomes: where do the numbers in those vectors come from?

### The Distributional Hypothesis

Every embedding method in this article rests on one observation, attributed to the linguist J. R. Firth: **"You shall know a word by the company it keeps."** Concretely, two words that habitually appear in similar contexts probably have similar meanings.

- "The **cat** sat on the mat" vs. "The **dog** sat on the mat"
- "The **king** ruled the kingdom" vs. "The **queen** ruled the kingdom"

If we can train any model to predict context from a word -- or vice versa -- the parameters that succeed at this prediction task will have absorbed distributional regularities. The embeddings emerge as a *byproduct* of the prediction.

The payoff of taking this hypothesis seriously is shown below: trained embeddings encode meaningful relations as nearly constant directions in space, which is why the famous analogy arithmetic works.

![Word analogies form constant directions in embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig2_word_analogy.png)

---

## Word2Vec: Learning from Local Context

Word2Vec (Mikolov et al., 2013) was the first method to learn high-quality embeddings cheaply on billions of tokens. It comes in two flavours -- **Skip-gram** and **CBOW** -- both implemented as one-hidden-layer neural networks with no non-linearity. The simplicity is the point.

### Skip-gram: Predict Context from Target

Given a target word, predict each surrounding context word inside a fixed window. For "the quick **brown** fox jumps" with window size 2, the target "brown" generates four positive training pairs:

```
(brown, the)   (brown, quick)   (brown, fox)   (brown, jumps)
```

The network has three layers: a one-hot input, an embedding lookup matrix $W \in \mathbb{R}^{V \times d}$, and an output matrix $W' \in \mathbb{R}^{d \times V}$ followed by softmax. Because the input is one-hot, the embedding layer reduces to a row lookup -- a constant-time operation.

![Skip-gram architecture: one-hot input, embedding lookup, softmax over the vocabulary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig1_skipgram_architecture.png)

The training objective averages log-probabilities of the true context words:

$$J = \frac{1}{T}\sum_{t=1}^{T} \sum_{-m \le j \le m,\, j \neq 0} \log P(w_{t+j} \mid w_t)$$

with

$$P(c \mid w) = \frac{\exp(\mathbf{v}_w^\top \mathbf{v}'_c)}{\sum_{i=1}^{V} \exp(\mathbf{v}_w^\top \mathbf{v}'_i)}.$$

Notice the asymmetry: $\mathbf{v}_w$ is read from the input matrix $W$ ("input" or "target" embedding), while $\mathbf{v}'_c$ is read from the output matrix $W'$ ("context" embedding). After training, most pipelines keep only $W$ -- but as we will see, GloVe argues that combining the two matrices is even better.

**Why it works.** If "cat" and "dog" both predict "sat", "mat", and "runs" in their contexts, gradient descent must push $\mathbf{v}_{\text{cat}}$ and $\mathbf{v}_{\text{dog}}$ in similar directions -- otherwise they cannot produce similar output distributions. Distributional similarity is *forced* into geometric similarity.

### CBOW: Predict Target from Context

CBOW reverses the roles: average the context embeddings, then predict the centre word.

$$\mathbf{h} = \frac{1}{2m} \sum_{j=-m,\, j\neq 0}^{m} \mathbf{v}_{w_{t+j}}, \qquad P(w_t \mid \text{context}) = \mathrm{softmax}(W'\mathbf{h}).$$

Practical differences:

- **Skip-gram** generates $2m$ training examples per centre word, so it sees rare words $2m$ times as often. It is slower per epoch but better on infrequent vocabulary.
- **CBOW** smooths over context (the average attenuates noise) and trains faster. It tends to produce slightly better embeddings for high-frequency words.

For most general-purpose embeddings on web-scale corpora, skip-gram with negative sampling is the default.

### The Softmax Bottleneck

Both architectures have the same wall: the softmax denominator sums over the entire vocabulary $V$. With $V = 100{,}000$ and billions of training pairs, a literal softmax is hopeless -- each gradient step costs $O(Vd)$. Word2Vec's first decisive trick is to never compute that softmax.

### Negative Sampling: The Key Trick

Instead of asking "which of the $V$ words is the true context?", negative sampling asks the easier binary question: "is this (word, context) pair real, or am I lying to you?". For each positive pair $(w, c)$, draw $k$ random "negative" words from a noise distribution $P_n$ and minimise

$$J = \log \sigma(\mathbf{v}_w^\top \mathbf{v}'_c) + \sum_{i=1}^{k} \mathbb{E}_{n_i \sim P_n}\!\left[\log \sigma(-\mathbf{v}_w^\top \mathbf{v}'_{n_i})\right]$$

where $\sigma$ is the sigmoid. Geometrically, the gradient pulls the target and the true context together while pushing the target away from $k$ unrelated words.

![Negative sampling: pull one positive context closer, push k random negatives away](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig6_negative_sampling.png)

Two details that matter in practice:

- **Noise distribution.** Word2Vec samples negatives from $P_n(w) \propto f(w)^{0.75}$, where $f(w)$ is the unigram frequency. The 0.75 exponent flattens the distribution: without it, "the" would be picked $\approx$100x more often than "zebra", and the gradient would be dominated by uninformative high-frequency negatives. With it, rare words show up as negatives often enough to provide useful signal.
- **Speed gain.** With $k = 5$--$15$ negatives, each step costs $O((k+1)d)$ instead of $O(Vd)$. For $V = 10^5$ and $k = 10$, that is roughly four orders of magnitude faster -- enough to train on billions of tokens on a single machine.

A separate trick, **subsampling**, also drops a fraction of frequent tokens during training. Together with negative sampling, these two heuristics are why Word2Vec runs in hours rather than weeks.

---

## GloVe: Global Matrix Factorization

Word2Vec scans the corpus one window at a time and never sees the global picture. **GloVe** (Pennington et al., 2014) takes the opposite approach: build the entire word-by-word co-occurrence matrix once, then factorise it.

### Why Co-occurrence Ratios?

Consider two probe words and the target words "ice" and "steam":

| Probe word | $P(\text{word} \mid \text{ice})$ | $P(\text{word} \mid \text{steam})$ | Ratio |
|---|---|---|---|
| solid | high | low | $\gg 1$ |
| gas | low | high | $\ll 1$ |
| water | high | high | $\approx 1$ |
| fashion | low | low | $\approx 1$ |

Raw probabilities mix two signals: how related the probe is to either word, and how common the probe is in general. The *ratio* removes the second signal and isolates the first. GloVe argues that embeddings should encode these ratios directly.

### The GloVe Objective

Let $X_{ij}$ be the number of times word $j$ appears in the context of word $i$. GloVe seeks word vectors $\mathbf{w}_i$ and context vectors $\tilde{\mathbf{w}}_j$ (plus biases) such that

$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \;\approx\; \log X_{ij}.$$

The full loss is a weighted least-squares regression on the log-counts:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2,$$

with

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

(Pennington et al. use $x_{\max} = 100$, $\alpha = 0.75$.) The weighting function does two jobs: it caps the influence of extremely frequent pairs (so "the the" cannot dominate the loss), and it down-weights rare pairs whose counts are statistically noisy.

This is literally a low-rank factorisation: a $V \times V$ matrix of log-counts is approximated by the product of a $V \times d$ word matrix and a $d \times V$ context matrix.

![GloVe approximates the log co-occurrence matrix by the product of a word and a context embedding matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig4_glove_factorization.png)

The published GloVe vectors actually use $\mathbf{w}_i + \tilde{\mathbf{w}}_i$ -- the sum of input and context vectors -- which empirically works slightly better than either alone.

### GloVe vs. Word2Vec

| Aspect | Word2Vec | GloVe |
|---|---|---|
| View of the corpus | Local windows (online) | Global co-occurrence matrix |
| Optimisation | SGD with negative sampling | Weighted least squares (AdaGrad in practice) |
| Memory | Low (no global stats stored) | Co-occurrence matrix can be huge |
| Strengths | Streams data; robust on small corpora | Slight edge on analogy tasks; deterministic |

In practice the two produce embeddings of comparable quality, and the choice often comes down to whichever pre-trained set already exists for your language and domain.

---

## FastText: Subword Embeddings

Word2Vec and GloVe assign one vector per *word*, which means they fail in two situations every real system runs into: words they have never seen during training, and morphologically rich languages where the same root spawns dozens of inflected forms. **FastText** (Bojanowski et al., 2017) fixes both by representing each word as a *bag of character n-grams*.

### How It Works

For the word "where", FastText pads it with boundary markers `<` and `>` and extracts every character n-gram of length 3 to 6, plus the full word as a special token:

- 3-grams: `<wh`, `whe`, `her`, `ere`, `re>`
- 4-grams: `<whe`, `wher`, `here`, `ere>`
- 5-grams, 6-grams: similar
- Full word: `<where>`

Each n-gram has its own embedding $\mathbf{z}_g$. The word vector is the sum:

$$\mathbf{v}_w = \sum_{g \in G(w)} \mathbf{z}_g$$

Training otherwise looks identical to Word2Vec skip-gram with negative sampling -- only the input "embedding" is replaced with this sum.

![FastText: every word is the sum of its character n-gram embeddings, which gives free OOV support](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig7_subword_fasttext.png)

### Why Subwords Pay Off

- **Out-of-vocabulary words.** Encountering "wherever" at inference time is no problem: it shares n-grams (`whe`, `here`, `ere`...) with "where" and "ever", so its summed vector is meaningful even though the word was never trained.
- **Morphology.** In Turkish, German, or Finnish, a single noun can take dozens of inflected forms. FastText shares parameters across them via shared n-grams, so rare inflections inherit information from common ones.
- **Typos.** A misspelled "teh" still shares n-grams with "the", so it lands roughly in the right neighbourhood.

The cost is bigger models (you need an embedding per n-gram, not per word) and slightly slower training. In English with a closed vocabulary the gains are modest; in morphologically rich languages they are dramatic.

| Scenario | Use FastText | Use Word2Vec / GloVe |
|---|---|---|
| Morphologically rich languages (German, Turkish, Finnish) | Yes | No |
| User-generated text full of typos and slang | Yes | No |
| Need OOV handling at inference | Yes | No |
| English with a fixed clean vocabulary | Either | Yes (smaller model) |
| Tightest possible memory budget | No | Yes |

---

## Visualising the Result

If embeddings really capture distributional semantics, projecting them down to two dimensions should reveal semantic clusters even though the projection itself knows nothing about word meaning. With t-SNE or UMAP on a few hundred trained vectors, you reliably see exactly that: animals huddle together, countries form their own region, royalty terms cluster in another corner.

![t-SNE projection: words from the same semantic field land in the same neighbourhood](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig3_tsne_clusters.png)

This is one of the most useful debugging tools in practice. If your trained embeddings do *not* show coherent clusters for obviously related words, something is wrong with your data, your tokenisation, or your hyperparameters -- long before you bother running a downstream evaluation.

---

## Language Models and Embeddings

A language model assigns a probability to a sequence of words. Training one well *requires* sharing strength across similar contexts -- and that sharing is exactly what embeddings provide.

### N-gram Language Models

A classical n-gram model estimates the next-word distribution by counting:

$$P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = \frac{\text{count}(w_{t-n+1}, \ldots, w_t)}{\text{count}(w_{t-n+1}, \ldots, w_{t-1})}.$$

The trouble is data sparsity. With a 50k vocabulary, a 4-gram model has $50{,}000^4 \approx 6 \times 10^{18}$ possible contexts -- the vast majority of which never appear in any corpus. Decades of clever smoothing techniques (Kneser-Ney, Witten-Bell, modified back-off) chip away at this problem, but they cannot share information across *similar* contexts: "the cat ate" and "the kitten ate" remain unrelated bins.

### Neural Language Models

The Bengio et al. (2003) neural language model fixed the sparsity problem in one move: replace counting with a parametric function whose first layer is a word embedding lookup.

$$\mathbf{h} = \tanh\left(W_h \cdot [\mathbf{v}_{w_{t-n+1}}; \ldots; \mathbf{v}_{w_{t-1}}] + \mathbf{b}_h\right), \quad P(w_t \mid \text{context}) = \mathrm{softmax}(W_o \mathbf{h} + \mathbf{b}_o).$$

Now "the cat ate" and "the kitten ate" produce *almost the same* hidden state because $\mathbf{v}_{\text{cat}} \approx \mathbf{v}_{\text{kitten}}$, and the model generalises to combinations it has never seen.

The empirical consequence is dramatic. As corpus size grows, n-gram perplexity flattens out -- there are simply not enough counts to keep improving. Neural LMs keep getting better, and Transformer LMs better still:

![Perplexity vs. corpus size: n-gram models plateau, neural and Transformer LMs keep improving](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/word-embeddings-lm/fig5_lm_perplexity.png)

This is the punch line: **embeddings are what let language models scale.** Without them, every new context is a new bin to estimate; with them, every new context borrows strength from the geometry the model has already learned.

### Word2Vec as a Simplified LM

In retrospect Word2Vec is essentially a stripped-down neural language model:

- **Skip-gram** is the LM that predicts each surrounding word independently from a single target.
- **CBOW** is the LM that predicts the centre word from a bag of surrounding words.

The simplifications -- shallow network, no non-linearity, no word order, negative sampling instead of full softmax -- are all in service of one goal: train fast enough on enough data for the geometry to settle. The full LM lives one floor up.

### Static vs. Contextual Embeddings (Preview)

Word2Vec, GloVe, and FastText all produce **static embeddings**: "bank" has a single vector, regardless of whether it appears next to "river" or "account". This is a known limitation. The next wave of models -- ELMo, BERT, GPT -- produce **contextual embeddings**, where every occurrence of a word gets a different vector based on its surroundings. We will cover those in Parts 5 and 6.

---

## Practical Training with Gensim

### Install

```bash
pip install gensim numpy matplotlib scikit-learn
```

### Training Word2Vec

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
    "the quick brown fox jumps over the lazy dog",
    "a cat and a dog are playing in the garden",
]
tokenized = [simple_preprocess(s) for s in sentences]

# Skip-gram with negative sampling
model = Word2Vec(
    sentences=tokenized,
    vector_size=100,   # embedding dimension d
    window=5,          # context window radius m
    min_count=1,       # drop words seen fewer times
    sg=1,              # 1 = skip-gram, 0 = CBOW
    negative=5,        # k negative samples per positive
    ns_exponent=0.75,  # the famous 0.75 exponent
    epochs=100,
    workers=4,
    seed=42,
)

print(f"shape: {model.wv['cat'].shape}")            # (100,)
print(model.wv.most_similar('cat', topn=3))
print(f"cat-dog similarity: {model.wv.similarity('cat', 'dog'):.4f}")
```

This toy corpus is far too small to produce useful embeddings -- treat the snippet as an API tour, not a training recipe. For real work, train on at least tens of millions of tokens, or load pre-trained vectors.

### Training FastText

The Gensim API is intentionally near-identical:

```python
from gensim.models import FastText

model_ft = FastText(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    min_n=3,           # smallest character n-gram
    max_n=6,           # largest character n-gram
    epochs=100,
    seed=42,
)

# OOV "kitty" never appeared in the corpus, but FastText still gives a vector
print(model_ft.wv['kitty'].shape)
```

### Loading Pre-trained Embeddings

For production, pre-trained vectors trained on billions of tokens almost always beat anything you train yourself on a small in-domain corpus:

```python
import gensim.downloader as api

# Google News Word2Vec (3M words, 300-dim)
w2v = api.load('word2vec-google-news-300')

# GloVe trained on Wikipedia + Gigaword (400K words, 300-dim)
glove = api.load('glove-wiki-gigaword-300')

# FastText with subword info (1M words, 300-dim)
fasttext = api.load('fasttext-wiki-news-subwords-300')

print(w2v.most_similar('computer', topn=5))
```

### Testing Analogies

```python
def analogy(model, a, b, c):
    """Solve a : b :: c : ?"""
    try:
        result = model.most_similar(positive=[b, c], negative=[a], topn=1)
        return result[0][0]
    except KeyError:
        return "OOV"

print(analogy(w2v, 'man', 'woman', 'king'))     # -> queen
print(analogy(w2v, 'France', 'Paris', 'Italy')) # -> Rome
```

### Visualising with t-SNE

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

words = list(w2v.index_to_key[:200])
vectors = np.array([w2v[w] for w in words])

projection = TSNE(n_components=2, random_state=42, perplexity=20).fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(projection[:, 0], projection[:, 1], s=12, alpha=0.6)
for i, word in enumerate(words):
    plt.annotate(word, projection[i], fontsize=8)
plt.title('Word embeddings (t-SNE)')
plt.tight_layout()
plt.savefig('embeddings_tsne.png', dpi=150)
```

---

## Evaluating Embeddings

Once you have a set of vectors, three classes of evaluation tell you whether they are any good.

### Intrinsic 1: Analogy Tasks

The Google Analogy Dataset contains 19,544 questions of the form "Athens is to Greece as Beijing is to ?". A well-trained 300-dim Word2Vec or GloVe model on a billion-word corpus typically scores 60--80% top-1 accuracy. This is a strong sanity check, though analogy accuracy is known to be somewhat brittle to dataset choice.

### Intrinsic 2: Word Similarity

Datasets like WordSim-353 and SimLex-999 collect human similarity ratings for word pairs. You compute cosine similarity for each pair under your embeddings and report the Spearman rank correlation $\rho$ against human judgments. Good embeddings reach $\rho \approx 0.6$--$0.8$ depending on the dataset.

### Extrinsic: Downstream Tasks

The only evaluation that ultimately matters: do the embeddings improve real tasks like sentiment classification, NER, or retrieval? Pre-trained embeddings typically lift accuracy by 2--10 percentage points when labelled data is limited -- the smaller your task-specific dataset, the larger the win.

### Quick Sanity Check

Before any of the above, just inspect nearest neighbours by hand. For "cat" you expect "dog", "kitten", "feline", maybe "pet". If you see "the", "and", "is" -- a classic symptom of forgetting to subsample frequent words -- you have a problem long before any benchmark will catch it.

---

## Choosing the Embedding Dimension

| Dimension $d$ | When to use |
|---|---|
| 50 -- 100 | Small datasets (under 1M tokens), simple downstream models |
| 100 -- 300 | Medium datasets, general-purpose embeddings, the sweet spot in practice |
| 300 -- 1000 | Large datasets (over 1B tokens), quality-critical applications |

The returns diminish quickly: going from 50 to 100 dimensions buys a lot, going from 300 to 600 buys almost nothing for most tasks. Start with 100 or 300 and only revisit if your downstream evaluation says you need more.

---

## Key Takeaways

- **Embeddings encode distributional semantics.** Words that share contexts share geometric neighbourhoods, and that neighbourhood structure is what gives downstream models their generalisation.
- **Word2Vec, GloVe, and FastText are three answers to the same question.** Word2Vec scans local windows; GloVe factorises the global co-occurrence matrix; FastText decomposes words into character n-grams. They produce embeddings of comparable quality through different routes.
- **Negative sampling makes training feasible** by replacing the full softmax with binary classification against $k$ random negatives, with frequencies smoothed by the $f(w)^{0.75}$ noise distribution.
- **Embeddings let language models scale.** Without them, n-gram counts plateau as the corpus grows; with them, neural LMs keep improving because each new context borrows strength from learned geometry.
- **Static embeddings have a hard limit.** "Bank" has one vector regardless of whether it sits next to "river" or "account". The next two wavefronts -- ELMo / BERT / GPT in Parts 5--6 -- replace static vectors with context-dependent ones.

Word embeddings cracked open neural NLP. Once words could be added, subtracted, and clustered like real vectors, every downstream architecture -- from RNNs to Transformers -- became possible. We pick up that thread in the next article with sequence modelling.

---

## Series Navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Text Preprocessing | [Previous](/en/nlp-introduction-and-preprocessing/) |
| **2** | **Word Embeddings and Language Models (this article)** | |
| 3 | RNN and Sequence Modeling | [Read next](/en/nlp-rnn-sequence-modeling/) |
| 4 | Attention Mechanism and Transformer | [Read](/en/nlp-attention-transformer/) |
| 5 | BERT and Pretrained Models | [Read](/en/nlp-bert-pretrained-models/) |
| 6 | GPT and Generative Models | [Read](/en/nlp-gpt-generative-models/) |
