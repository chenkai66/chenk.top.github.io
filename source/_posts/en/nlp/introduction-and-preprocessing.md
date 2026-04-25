---
title: "NLP Part 1: Introduction and Text Preprocessing"
date: 2025-10-01 09:00:00
tags:
  - NLP
  - Deep Learning
  - Text Preprocessing
categories: Natural Language Processing
series:
  name: "Natural Language Processing"
  part: 1
  total: 12
lang: en
mathjax: true
description: "A first-principles introduction to NLP and text preprocessing. We trace the four eras of the field, build the cleaning to vectorization pipeline by hand, and unpack the math behind tokenization, TF-IDF, n-grams, and distributed representations."
disableNunjucks: true
series_order: 1
---

Every time you ask Claude a question, autocomplete a sentence in Gmail, or read a Google Translate page, you are touching a stack that took seventy years to assemble. Natural Language Processing is the discipline that taught machines to read, score, transform, and write human language -- and the surprising thing is how much of the modern stack still rests on a small set of preprocessing primitives invented decades ago.

This first article in the series does two things. First, it draws the map: where the field came from, what it covers today, and why the tools we use look the way they do. Second, it builds the foundational layer -- cleaning, tokenization, normalization, and feature extraction -- with code that you can lift directly into a project. By the end you will have a reusable preprocessing pipeline and, more importantly, a sense of when each step helps and when it quietly destroys signal.

![NLP application landscape](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig1_applications_landscape.png)

## What You Will Learn

- The four paradigms of NLP and the technical reason each one displaced the previous
- A precise vocabulary for tokenization: characters, words, subwords, and why BPE won
- How to build a configurable preprocessing pipeline in Python with NLTK, spaCy, and scikit-learn
- The math behind Bag-of-Words and TF-IDF, and how to read the resulting matrices
- Zipf's law, n-gram language models, and why one-hot vectors fail
- A decision table for when to apply (or skip) each preprocessing step

**Prerequisites**: Comfortable Python, light familiarity with NumPy and pandas, no prior NLP exposure required.

---

## 1. Four Eras of NLP

NLP did not advance smoothly. It moved in jumps, each driven by a new representation of language. Knowing the sequence helps you reach for the right tool: rule systems still beat neural nets for narrow form-filling, statistical methods still drive search ranking, and embeddings dominate everything else.

### 1.1 Symbolic Era (1950s -- late 1980s)

Early systems treated language as a logic problem. ELIZA (1966) matched user input against hand-crafted regex patterns and rephrased the captured groups; SHRDLU (1970) parsed instructions about a blocks world using a hand-written grammar. These systems were precise within their domain and completely brittle outside it -- a synonym or a typo broke them. The lesson, in hindsight, is that language has too many surface forms for any human to enumerate.

### 1.2 Statistical Revolution (1990s)

The turning point was the realization that you do not need to write rules; you can estimate probabilities from data. The bigram model is the canonical example:

$$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$$

This single formula powered IBM's statistical machine translation, the first viable speech recognizers, and probabilistic part-of-speech taggers. Hidden Markov Models extended the same idea to latent state, and probabilistic context-free grammars handled syntax. Features were still hand-engineered, but the rules were learned.

### 1.3 Deep Learning Era (2013 -- 2016)

Word2Vec (Mikolov et al., 2013) showed that a tiny neural network trained to predict context words produces vectors with a remarkable property: semantic relationships become arithmetic.

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

For the first time, words were no longer atomic identifiers. They lived in a continuous space where similarity was a cosine away. RNNs and LSTMs followed, letting models thread context through a sequence and finally learn from order, not just bag-of-tokens counts.

### 1.4 Transformer Revolution (2017 -- present)

The 2017 paper "Attention Is All You Need" replaced recurrence with self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Two practical consequences mattered. First, the model is fully parallel across positions, so training scales with GPU memory rather than sequence length. Second, every token can attend directly to every other token, which finally solved the long-range dependency problem. BERT, GPT, and every modern LLM are direct descendants.

| Era | Years | Core idea | What broke it |
|---|---|---|---|
| Symbolic | 1950 -- 1980s | Hand-written rules and grammars | Cannot enumerate surface forms |
| Statistical | 1990s -- 2010s | Estimate probabilities from corpora | Hand-engineered features hit a ceiling |
| Deep learning | 2013 -- 2016 | Learn dense representations end-to-end | Recurrence is sequential, slow to train |
| Transformer | 2017 -- now | Self-attention over the whole sequence | (Still being explored) |

**Insight**: each shift solved the previous era's bottleneck without throwing away the layer below. Even today, an LLM tokenizer is a statistical artifact, and your retrieval system probably uses TF-IDF as a fallback.

---

## 2. Where NLP Shows Up Today

| Domain | Examples |
|---|---|
| **Text classification** | Sentiment, spam, intent routing |
| **Information extraction** | Named entities, relations, knowledge graphs |
| **Generation** | Translation, summarization, code |
| **Conversational AI** | ChatGPT, Claude, voice assistants |
| **Search and analysis** | Semantic search, topic modeling, RAG |

The figure above arranges these into six clusters. Notice that almost every cluster ultimately consumes a vector -- which is exactly what preprocessing produces.

---

## 3. The Preprocessing Pipeline at a Glance

Before any model, raw text has to become numerical features. The standard pipeline has six stages, and each one is a deliberate choice that trades information for regularity.

![Text preprocessing pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig2_preprocessing_pipeline.png)

```
Raw text
  -> Cleaning      (strip HTML, URLs, emails, junk characters)
  -> Tokenization  (split into words / subwords)
  -> Normalization (lowercase, lemmatize, optionally stem)
  -> Stopword pass (drop "the", "is", "at" if they hurt)
  -> Vectorization (BoW, TF-IDF, or embeddings)
  -> Model
```

A common mistake is to apply every step by reflex. The right framing is: each stage should remove noise that downstream cannot handle and preserve everything else. We will revisit this trade-off at every step.

### 3.1 Environment Setup

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

## 4. Step 1 -- Text Cleaning

Web text comes wrapped in HTML, peppered with URLs, and littered with control characters. Cleaning removes the obvious noise without touching meaning.

```python
import re

def clean_text(text: str) -> str:
    """Strip HTML, URLs, emails, non-letters; collapse whitespace."""
    text = re.sub(r'<[^>]+>', '', text)              # HTML tags
    text = re.sub(r'http\S+|www\.\S+', '', text)     # URLs
    text = re.sub(r'\S+@\S+', '', text)              # emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # keep letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()         # normalize whitespace
    return text

raw = """<p>Check out https://example.com for info!</p>
Contact info@test.com. Price: $29.99"""
print(clean_text(raw))
# Check out for info Contact Price
```

**The aggressive cleaning trade-off**. The function above also deletes digits and punctuation. That is fine for topic modeling, where numbers add noise, but it is wrong for:

- **Sentiment analysis** -- `!!!` and `?!` carry emotion.
- **Named entity recognition** -- "Apple Inc." needs the period and the capitalization.
- **Financial NLP** -- `$29.99` is the actual signal you care about.

Always tailor the regex set to the task; do not apply a one-size-fits-all cleaner.

**Performance tip**. Compile patterns once if you process millions of documents:

```python
HTML_RE = re.compile(r'<[^>]+>')
URL_RE  = re.compile(r'http\S+|www\.\S+')
text = URL_RE.sub('', HTML_RE.sub('', text))
```

For HTML in the wild (malformed tags, embedded scripts), regex is fragile. Reach for a parser:

```python
from bs4 import BeautifulSoup
text = BeautifulSoup(html_text, 'html.parser').get_text(' ', strip=True)
```

---

## 5. Step 2 -- Tokenization

Tokenization splits text into the atomic units a model will see. The boundary you choose -- characters, words, subwords -- determines vocabulary size, sequence length, and how gracefully the model handles words it has never seen.

![Three tokenization strategies for the same input](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig4_tokenization_variants.png)

### 5.1 Word Tokenization

```python
# Naive: breaks on contractions and punctuation
"Don't split can't".split()
# ["Don't", 'split', "can't"]

from nltk.tokenize import word_tokenize
tokens = word_tokenize("Dr. Smith earned $150,000 in 2023! Isn't that amazing?")
# ['Dr.', 'Smith', 'earned', '$', '150,000', 'in', '2023', '!',
#  'Is', "n't", 'that', 'amazing', '?']
```

NLTK keeps `Dr.` as one token, separates punctuation, and splits the contraction `Isn't` into `Is` + `n't`. Each of those decisions is a hard-coded English convention -- which is exactly why word tokenization is brittle across languages.

### 5.2 Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize
text = "Dr. Johnson works at A.I. Corp. He earned his Ph.D. in 2010."
sent_tokenize(text)
# ['Dr. Johnson works at A.I. Corp.', 'He earned his Ph.D. in 2010.']
```

NLTK's Punkt model learns from data which periods end sentences and which mark abbreviations.

### 5.3 Subword Tokenization (BPE)

Modern models -- GPT, BERT, Llama, Claude -- do not tokenize on words. They use **subword tokenization**, almost always a variant of Byte-Pair Encoding (BPE):

1. Start with a vocabulary of individual characters.
2. Count adjacent character pair frequencies across the corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the vocabulary reaches a target size (commonly 30k -- 100k).

```
Corpus: "low" x5, "lower" x2, "newest" x6, "widest" x3
Initial:  l o w  /  l o w e r  /  n e w e s t  /  w i d e s t

Merge 1: (e, s) -> es        # frequent in "newest", "widest"
Merge 2: (es, t) -> est
Merge 3: (l, o) -> lo
...
```

Why BPE matters in practice:

- **Rare words decompose** -- `unbelievable` becomes `un + believ + able`, all of which have appeared elsewhere.
- **Vocabulary stays bounded** -- a 50k subword vocabulary covers any English text and most code.
- **Cross-lingual transfer** -- the same tokenizer handles English, French, and Mandarin if trained on a multilingual corpus.

Here is a minimal, runnable implementation:

```python
from collections import defaultdict

def get_stats(vocab):
    """Count frequency of adjacent symbol pairs."""
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
    print(f"merge {step + 1}: {best} -> {''.join(best)}")
```

For production, use Hugging Face's `tokenizers` library -- it ships GPT-style BPE, BERT WordPiece, and SentencePiece behind a unified API.

---

## 6. Step 3 -- Normalization

Normalization collapses surface variants of the same word into a single form, which shrinks vocabulary and improves matching. It also throws information away, so apply it deliberately.

### 6.1 Lowercasing

```python
"Apple Inc. sells apples in APPLE stores".lower()
# "apple inc. sells apples in apple stores"
```

Lowercasing helps search and topic modeling. It hurts named-entity recognition (`Apple` the company collapses with `apple` the fruit) and any task where capitalization signals emphasis.

### 6.2 Stemming vs Lemmatization

**Stemming** chops suffixes with deterministic rules. Fast, crude, sometimes wrong:

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for w in ['running', 'easily', 'connection']:
    print(f"{w} -> {stemmer.stem(w)}")
# running -> run, easily -> easili, connection -> connect
```

`easili` is not a word -- the Porter stemmer optimizes for matching, not for legibility.

**Lemmatization** uses a dictionary plus part-of-speech information to return the actual lemma:

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

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Speed | Microseconds | Milliseconds (POS-tagged) |
| Output | May not be a real word | Always a dictionary form |
| Accuracy | Lower | Higher |
| Best for | Search and IR | NLU and QA |

A useful default: use lemmatization unless you are running a high-throughput retrieval system where the latency budget is tight.

---

## 7. Step 4 -- Stopwords and Zipf's Law

Stopwords are common closed-class words such as `the`, `is`, `at` that carry little task-specific meaning. Removing them shrinks vocabulary by roughly a third and concentrates signal in content words.

The reason a small set of words dominates is Zipf's law: in any natural-language corpus, a word's frequency is roughly inversely proportional to its rank.

$$f(\text{rank}) \propto \frac{1}{\text{rank}}$$

![Zipf distribution: head dominated by stopwords, long tail of rare words](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig5_zipf_distribution.png)

The top ten words alone often account for 25 -- 30% of all tokens. That is the head of the distribution, and it is mostly stopwords. The tail -- thousands of words appearing once or twice -- is where most semantic content lives, but it is also where models struggle and where subword tokenization earns its keep.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
text = "The quick brown fox jumps over the lazy dog"
filtered = [w for w in word_tokenize(text.lower()) if w not in stop_words]
# ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

When to remove stopwords:

- **Yes** -- bag-of-words and topic models, search indexing.
- **No** -- sentiment (`not good` is not the same as `good`), QA (function words carry the question), any deep model that learns to weight tokens itself.

---

## 8. Step 5 -- From Tokens to Vectors

A model needs numbers. Two classical encodings -- Bag-of-Words and TF-IDF -- still anchor most retrieval systems and are the right baseline for any new task.

### 8.1 One-hot vs Distributed Representations

Before we get to BoW, it helps to see why naive encodings fail. A one-hot vector assigns each word a unique index, with a 1 in that position and 0 everywhere else. Every pair of words is orthogonal, which means the encoding carries zero similarity information.

![One-hot encoding loses semantics; learned embeddings recover them](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig6_onehot_vs_distributed.png)

Distributed representations -- which we will build in Part 2 -- pack meaning into dense vectors where related words sit near each other. BoW and TF-IDF are a halfway step: each word still gets its own dimension, but the value in that dimension is a frequency, not just a marker.

### 8.2 Bag of Words

Represent each document as a vector of word counts, ignoring order:

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

The fatal limitation: `dog bites man` and `man bites dog` produce identical vectors. BoW discards order entirely.

### 8.3 TF-IDF

TF-IDF up-weights words that are frequent in a document but rare in the corpus -- a heuristic for "important to this document, but not generic":

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

$$\text{IDF}(t) = \log\!\frac{1 + N}{1 + \text{df}(t)} + 1$$

where $N$ is the number of documents and $\text{df}(t)$ is the number of documents containing term $t$. The `+1` smoothing keeps the IDF defined when a term appears in every document (or in none).

![Bag of Words counts versus TF-IDF weights on the same toy corpus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig3_bow_vs_tfidf.png)

The figure above shows both matrices side by side. Notice how `learning` -- present in every document -- gets weighted down by TF-IDF, while a word like `vision` that is unique to one document gets lifted. That is exactly the ranking behavior you want for search.

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
    print(f"Doc {i + 1}: {dict(top.round(3))}")
```

**Production-grade TF-IDF**. The defaults rarely survive contact with a real corpus. Tune at least these knobs:

```python
tfidf = TfidfVectorizer(
    max_features=5_000,    # cap vocabulary
    min_df=2,              # drop hapax legomena
    max_df=0.8,            # drop terms in 80%+ of docs (effective stopwords)
    ngram_range=(1, 2),    # unigrams + bigrams capture short phrases
    sublinear_tf=True,     # log-scale TF, dampens repetition
    stop_words='english',
)
```

---

## 9. Step 6 -- N-gram Language Models

Once you have tokens, you can also model how they follow each other. An n-gram model factors a sentence into a chain of conditional probabilities:

$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

A bigram model uses one word of context, a trigram uses two, and so on.

![N-gram windows, the bigram formula, and the perplexity vs sparsity trade-off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/introduction-and-preprocessing/fig7_ngram_language_models.png)

The trade-off is sharp:

- **Larger n** captures more context, which lowers perplexity (perplexity is roughly the model's effective branching factor -- lower is better).
- **Larger n** also explodes the parameter count and starves on rare contexts. With $V$ vocabulary, a trigram model has up to $V^3$ parameters, most of which see zero training examples. This is the **sparsity problem**, the central pain point of statistical NLP.

Smoothing techniques (Laplace, Kneser-Ney) patch the holes by redistributing probability mass to unseen n-grams. Modern neural language models sidestep the issue entirely by sharing parameters across contexts via embeddings -- which is the bridge to Part 2.

---

## 10. A Reusable Preprocessing Class

Putting the steps together into something you can drop into a project:

```python
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Configurable English text preprocessing pipeline."""

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
    print(f"original:  {orig}")
    print(f"processed: {proc}\n")
```

---

## 11. End-to-End Example: a Minimal Spam Classifier

Combining everything into a working classifier. Use the SMS Spam Collection or any Kaggle spam dataset for real experiments; the snippet below is intentionally tiny so it runs anywhere.

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
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=spam, 0=ham

# Keep stopwords -- "free", "now", "you" are spam signals.
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
    print(f"[{'SPAM' if pred else 'HAM'}] {msg}")
```

The point of the example is not the accuracy on ten samples -- it is the shape of the pipeline. Swap in 5,000 SMS messages and the same code reaches roughly 97% accuracy with no further engineering. That is the strength of the classical NLP stack: short, transparent, and remarkably hard to beat without a GPU.

---

## 12. Decision Table: Which Steps for Which Task

| Task | Tokenization | Normalization | Stopwords | Features |
|---|---|---|---|---|
| Search / IR | Word | Stem | Remove | TF-IDF |
| Sentiment | Word / subword | Lemma | **Keep** | TF-IDF or embeddings |
| Topic modeling | Word | Lemma | Remove | BoW or TF-IDF |
| Machine translation | Subword (BPE) | Minimal | Keep | Embeddings |
| NER | Word | None | Keep | Embeddings + context |
| Modern LLMs | Subword (BPE) | None | Keep | Learned embeddings |

**Rule of thumb**. The more model capacity and data you have, the less preprocessing you should do. Deep models learn their own normalization; aggressive preprocessing destroys signal they could have used. Classical ML benefits from careful feature engineering; LLMs prefer raw text.

---

## Key Takeaways

- Preprocessing is task-specific. Search wants aggressive normalization; neural models want raw text.
- Subword tokenization (BPE, WordPiece, SentencePiece) is the modern default because it bounds vocabulary and handles unseen words.
- TF-IDF remains the right baseline. If a TF-IDF + logistic regression baseline beats your fancy model, the fancy model is broken.
- Zipf's law explains why stopword removal helps classical models and why long-tail words are hard.
- Less is often more. Over-preprocessing hurts representation learning. Always measure.

---

## Further Reading

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) -- Jurafsky and Martin, free online textbook (the canonical reference)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- the Transformer paper
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) -- Sennrich et al., the BPE-for-NLP paper
- [spaCy linguistic features](https://spacy.io/usage/linguistic-features)
- [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

## Series Navigation

| Part | Topic | Link |
|---|---|---|
| **1** | **Introduction and Text Preprocessing (this article)** | |
| 2 | Word Embeddings and Language Models | [Read next -->](/en/nlp-word-embeddings-lm/) |
| 3 | RNN and Sequence Modeling | [Read -->](/en/nlp-rnn-sequence-modeling/) |
| 4 | Attention Mechanism and Transformer | [Read -->](/en/nlp-attention-transformer/) |
| 5 | BERT and Pretrained Models | [Read -->](/en/nlp-bert-pretrained-models/) |
| 6 | GPT and Generative Models | [Read -->](/en/nlp-gpt-generative-models/) |
