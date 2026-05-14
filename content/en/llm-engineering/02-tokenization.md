---
title: "LLM Engineering (2): Tokenization Deep Dive"
date: 2026-03-28 09:00:00
tags:
  - LLM
  - tokenization
  - bpe
  - sentencepiece
  - chat-template
categories: LLM Engineering
series: llm-engineering
series_order: 2
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "BPE vs SentencePiece vs WordPiece, byte-level fallback, the CJK token-bloat problem, vocabulary expansion costs, and the chat-template tokens that silently shape every model's behavior."
translationKey: "llm-engineering-2"
---

Tokenization is the layer everyone skips. It's also the layer where I've debugged the most production bugs — silent quality regressions, mysterious cost spikes, models refusing to follow instructions because someone formatted the chat template wrong. This chapter is everything I wish I'd internalized before shipping a multilingual product.

![LLM Engineering (2): Tokenization Deep Dive — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/illustration_1.png)

---

## What a tokenizer actually does

A tokenizer maps a string to a list of integer IDs. Reverse maps IDs back to a string. Both directions are deterministic but not bijective in general — round-tripping `tokenizer.decode(tokenizer.encode(s))` can lose whitespace, normalize Unicode, or collapse repeated punctuation, depending on the algorithm.

Three families dominate:

- **WordPiece** [Schuster & Nakajima, 2012; Wu et al., 2016] — used by BERT. Greedy longest-match starting from the front. Marks subword continuations with `##`.
- **BPE (Byte Pair Encoding)** [Sennrich et al., 2016] — used by GPT-2/3/4, LLaMA, Qwen, DeepSeek. Iteratively merges the most frequent adjacent pair until vocab size is reached. The original BPE algorithm comes from a 1994 paper by Philip Gage on data compression; Sennrich et al. adapted it for NMT.
- **Unigram (SentencePiece)** [Kudo, 2018; Kudo & Richardson, 2018] — used by T5, mBART, XLM-R. Starts with a huge candidate vocab, prunes by EM to maximize sequence likelihood under a unigram language model.

In 2026 essentially all large generative LLMs use **byte-level BPE** [Radford et al., 2019] or a SentencePiece BPE variant. The "byte-level" part is what makes the tokenizer Unicode-safe: the alphabet is the 256 bytes, not the ~150K Unicode code points, so any string in any encoding can be tokenized without an `<UNK>` token.

A subtle distinction worth pinning down: BPE-on-characters (the original Sennrich formulation) has an alphabet of all distinct Unicode characters seen in training — typically tens of thousands for multilingual data, with the long tail dropped to `<UNK>`. BPE-on-bytes (the GPT-2 formulation) has an alphabet of 256 bytes, period. Any UTF-8 string decomposes into bytes and tokenizes without failure. The cost is that a single Chinese character that doesn't make it into a merged token consumes 3 bytes (3 tokens) instead of 1 character (1 token). This is the root cause of the CJK token bloat we'll dissect later.

## BPE on a byte stream, by hand

![fig1: BPE merge tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/fig1_bpe_tree.png)


Here's BPE on a tiny corpus to make the algorithm concrete:

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

Real-world BPE replaces `list(w)` with the byte sequence `w.encode("utf-8")` and uses a much bigger corpus (trillions of tokens), but the algorithm is exactly this loop run a few hundred thousand times.

The byte-level part matters: GPT-2 famously remaps the 256 bytes to a printable Unicode subset so they're easier to inspect, but the encoding alphabet is still 256 symbols. That means *no input can ever fail to tokenize*. There is no out-of-vocabulary failure mode.

## Tiktoken: how OpenAI actually does it

OpenAI's tiktoken library is the production reference for byte-level BPE. The implementation choices it makes are different from naive BPE in ways that matter at scale.

**Pre-tokenization regex.** Before BPE merges, tiktoken splits text into chunks using a regex like:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

This splits on whitespace, numbers (capped at 3 digits to limit "9999999" → 1 token weirdness), contractions, and runs of non-alphanumeric characters. BPE merges only happen *inside* these chunks. This prevents merges across word boundaries, which would otherwise produce nonsensical multi-word tokens that hurt generalization.

**Rust implementation, BPE caching.** The hot path is encode-by-pre-token: split the input string by regex, then for each chunk apply BPE merges using a cached merge table. Tiktoken's Rust core hashes pre-tokens and caches merge results — repeated common words (e.g., "the", "and") encode in O(1). Throughput is ~3 MB/s on a single core for English, ~1.5 MB/s for CJK.

**Vocabulary file format.** Tiktoken stores vocab as base64-encoded byte strings keyed by token ID. To inspect what token 12345 actually represents in `cl100k_base`, decode the base64 and you'll see the raw bytes. Many tokens are unprintable (whitespace runs, partial CJK characters); the vocab is byte-level not character-level.

The cl100k_base tokenizer (GPT-3.5/4) has 100,256 tokens. The o200k_base tokenizer (GPT-4o, GPT-4o-mini, o1) has 199,997. The expansion from 100K to 200K was driven by multilingual coverage — o200k merges many more CJK character sequences and brings English tokens-per-word from 1.3 down to 0.93.

## Vocabulary size: a cost-quality knob

![fig4: vocab size vs perplexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/fig4_vocab_perplexity.png)


Vocab size $V$ is a hyperparameter chosen at pretraining time. It trades **embedding-table FLOPs and memory** against **sequence length per document**.

Concrete numbers for a 7B-class model with $d_{\text{model}} = 4096$:

| Vocab | Embedding params | Tied LM head | Total embed | Token / English word |
|---|---|---|---|---|
| GPT-2 | 50,257 | tied | 206 M | 1.3 |
| LLaMA-2 | 32,000 | untied | 262 M | 1.4 |
| Qwen2.5 | 151,936 | tied | 622 M | 0.95 |
| GPT-4o | 200,000 | tied | 819 M | 0.93 |
| Qwen3 | 152,064 | tied | 623 M | 0.95 |

Bigger vocab → fewer tokens per document → cheaper inference at prompt-billing time, but bigger embeddings and softmax. Most modern models settled around 100-200K. The Qwen / GPT-4o range hits a sweet spot for multilingual coverage without making the embedding table dominate.

What changes when vocab grows? The number of merge operations increases, so longer common subwords (whole rare words, common phrases) get a single token. For English, the marginal benefit drops fast past ~50K. For CJK, it keeps paying off until at least 200K.

The empirical scaling between vocab size and quality has been studied. [Tao et al., 2024] found that for a fixed training budget, vocab size has an optimum that *grows with model size*: small models prefer small vocabs (< 32K), large models prefer larger (> 100K). The intuition: large models can amortize the embedding parameter cost, and benefit more from the sequence-length compression. They derived a scaling law $V_{\text{opt}} \propto N^{0.65}$ where $N$ is non-embedding params. For a 70B model the optimum is around 200K; for a 1B model around 32K.

## The Llama 3 tokenizer's 128K vocab decision

LLaMA-2 used a 32K SentencePiece BPE tokenizer trained primarily on English. LLaMA-3 jumped to 128K (specifically 128,256) using tiktoken's cl100k_base as starting point, retrained on Meta's pretraining mix. The Llama 3 paper [Dubey et al., 2024] gives the rationale:

- 4× more multilingual coverage than the LLaMA-2 tokenizer
- ~15 % shorter sequences on average across the pretraining corpus
- Compute saved from shorter sequences outweighs embedding-table cost above 7B

For a 70B model, the embedding table (128K × 8192 = 1.05B params, tied) is 1.5 % of total params — a rounding error. For a 1B model the same embedding would be 100 % of params, which is why small open models (Phi-4-mini, Qwen3-0.5B) often use smaller vocabs (50-64K) to keep the embedding-to-non-embedding ratio sane.

LLaMA-3 also added 28 reserved special tokens (`<|reserved_special_token_0|>` through `<|reserved_special_token_27|>`) at the end of the vocab. These are placeholders for future fine-tuning needs — Meta's Llama-3-Instruct uses several of them for chat template markers without retraining the embedding table. This is a pattern worth copying: reserve a few hundred tokens at vocab build time for downstream specialization.

## The CJK token-bloat problem

![LLM Engineering (2): Tokenization Deep Dive — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/illustration_2.png)


![fig2: CJK token bloat by language](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/fig2_cjk_bloat.png)


This one bit me hard on a Chinese product. GPT-3.5's tokenizer (cl100k_base, 100K vocab) tokenizes the English string "Hello, how are you?" as 6 tokens. The Chinese equivalent "你好，你今天怎么样？" tokenizes as 17 tokens.

Three factors compound:

1. **UTF-8 encoding cost.** A typical Chinese character takes 3 UTF-8 bytes vs 1 for ASCII. Pre-merge, that's already a 3× token disadvantage at the byte level.
2. **Merge frequency.** BPE merges the most frequent pairs. Chinese pretraining data was historically a small fraction of OpenAI's corpus, so character-level pairs got merged less.
3. **No word boundaries.** English BPE benefits from the leading-space convention (`Ġhello`); Chinese has no such marker.

Cost of the same prompt across tokenizers, on the Chinese sentence "你好，请帮我用 Python 写一个快速排序":

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 47 |
| cl100k_base (GPT-3.5/4) | 24 |
| o200k_base (GPT-4o) | 18 |
| LLaMA-2 (32K) | 39 |
| LLaMA-3 (128K) | 21 |
| Qwen2.5 (152K) | 14 |

If you bill per token (you do — every API does), this means **for the same Chinese workload, Qwen2.5 is 1.7x cheaper than GPT-4o per token, before any per-token price difference**. Qwen3 hits 13. For a Chinese-heavy product, choose the tokenizer that was trained on Chinese, not just the one that came with the hyped model.

You can measure this yourself in five lines:

```python
import tiktoken
from transformers import AutoTokenizer

text = "你好，请帮我用 Python 写一个快速排序"
print("o200k:", len(tiktoken.get_encoding("o200k_base").encode(text)))
print("qwen3:", len(AutoTokenizer.from_pretrained("Qwen/Qwen3-7B").encode(text)))
print("llama3:", len(AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B").encode(text)))
```

Run this on your real prompts before picking a model. The decision frequently comes out different from "which is best on MMLU".

## Worked example: the cost arithmetic on a real Chinese product

Take a typical Chinese support-bot workload: avg user message 80 Chinese characters, avg model response 200 Chinese characters, avg system prompt 600 Chinese characters. 1M conversations per day.

| Tokenizer | Tokens / convo | Daily tokens | Monthly cost @ $1/1M |
|---|---|---|---|
| cl100k_base | (600 + 80 + 200) × 24/14 ≈ 1509 | 1.51 B | $45,300 |
| o200k_base | (600 + 80 + 200) × 18/14 ≈ 1131 | 1.13 B | $33,930 |
| Qwen3 | 880 × 13/14 ≈ 817 | 0.82 B | $24,510 |

(Conversion factor based on the relative tokens-per-CJK-character measured above.) The same workload, same prices per token, swing from $45K to $24K monthly purely on tokenizer choice. At higher volumes it becomes 7-figure savings annually.

The same logic in reverse: a product whose users are 100 % English speakers gets *more* tokens out of LLaMA-3's 128K vocab than out of LLaMA-2's 32K, because the bigger vocab merges more multi-word phrases (e.g., "in order to" becomes 1 token instead of 4). The improvement on English is much smaller (5-15 %) than on Chinese (2-3×), but it's still real.

## Code-tuned tokenizers

Code is its own dialect with its own tokenization needs. Indentation (4 spaces, 8 spaces, tabs), bracket density, and identifier length distributions are all different from prose. A general-purpose tokenizer trained on web text under-merges common code patterns.

**StarCoder** [Li et al., 2023] uses a 49,152-token tokenizer trained specifically on The Stack, a 6 TB code corpus. Comparison on a typical Python file (1 KB):

| Tokenizer | Tokens |
|---|---|
| GPT-2 | 384 |
| cl100k_base | 287 |
| StarCoder | 218 |
| DeepSeek-Coder | 215 |

StarCoder's 24 % token-count reduction over cl100k comes from merging common patterns: `def `, `    return `, `import `, `if __name__ == "__main__":`. DeepSeek-Coder's tokenizer goes further with explicit indentation tokens — 4 spaces is one token, 8 spaces another. Code-pretrained models almost always use these specialized tokenizers because the cost saving compounds: code is 8-15 % of pretraining data, and code workloads are token-heavy at inference.

**FIM (fill-in-the-middle) tokens.** Code-completion models need to insert at the cursor, not just append. FIM training [Bavarian et al., 2022] uses three special tokens — `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` — to teach the model to predict the middle given prefix and suffix. Every modern code LLM (GitHub Copilot, Codeium, Cursor's local models) uses FIM. The chat template for code completion has nothing to do with conversational chat templates; it's a different post-training regime entirely.

## Vocabulary expansion: don't, usually

![fig3: tokenizer comparison table](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/fig3_tokenizer_table.png)


A common impulse: "I want to fine-tune LLaMA-3 on Chinese, the tokenizer is bad for Chinese, let me add 50K Chinese tokens." This is almost always wrong.

Adding tokens to a pretrained model means:

1. New rows in the embedding table — initialized from scratch or from averaged subword embeddings.
2. New rows in the LM head (if untied) — similarly random.
3. The model has to learn to use these new tokens through fine-tuning.

The new embeddings need orders of magnitude more training data to reach parity with the original tokens. Several papers ([Cui et al., 2023] Chinese-LLaMA, [Faysse et al., 2024] CroissantLLM) report 50-200B tokens of continued pretraining needed to fully integrate an expanded vocabulary. Below that, you get a model that tokenizes Chinese efficiently but generates lower-quality Chinese than the original.

The pragmatic alternative: use a base model whose tokenizer was trained on your target language from the start. Qwen3 for Chinese, Sea-LION or Sailor for SE Asian languages, BLOOM [Scao et al., 2022] for African languages. Vocab expansion is for when no such model exists.

When vocab expansion is unavoidable (e.g., adding domain-specific medical terminology to a base model with no overlap), a few practices help:

**Initialize new embeddings as the mean of their subword decomposition.** For a new token "tachycardia", initialize its embedding as the mean of the embeddings of its existing BPE pieces ("tachy", "card", "ia"). This gets you partway to a sensible starting point. [Hewitt, 2021] showed this beats random init by 3-5 perplexity points after a small amount of training.

**Freeze the rest of the model for a warmup phase.** Train only the new embeddings for a few hundred steps before unfreezing the full model. This avoids the rest of the model trying to compensate for the random new embeddings, which corrupts existing knowledge.

**Use LoRA for the embedding layer too.** PEFT supports LoRA on `embed_tokens`. Combined with the warmup trick, this gets you most of the benefit of vocab expansion at a fraction of the data cost.

## Tokenizer-free approaches: ByT5, MEGABYTE, MAMBA-byte

Two recent threads worth knowing about, neither yet mainstream.

**ByT5** [Xue et al., 2022] trains T5 directly on UTF-8 bytes — no tokenizer at all. The vocab is 256 (the bytes) plus 3 special tokens. ByT5-large matches mT5-large on multilingual benchmarks despite the much longer sequences. The cost is sequence length: a Chinese paragraph that's 100 BPE tokens becomes 600 bytes. Training is slower per step but more sample-efficient per byte; inference is slower per response. ByT5 found a niche in multilingual NER and noisy-text tasks where tokenizer mismatches dominate quality.

**MEGABYTE** [Yu et al., 2023] handles long byte sequences hierarchically: a "patch" model groups bytes into patches of size 8-16 and runs a small Transformer per patch, then a "global" Transformer over patch representations. This recovers near-O(n²) cost over the patches rather than over individual bytes, making byte-level practical for sequences up to ~1M.

**MAMBA-byte** [Wang et al., 2024] applies the Mamba state-space architecture directly to byte streams. The linear-time recurrence makes byte-level training tractable at scale. MAMBA-byte 350M was reported to match BPE-tokenized baselines on ARC-Easy and PIQA, but degrades on tasks requiring precise word-level reasoning (MMLU, HumanEval).

For 2026 production: byte-level BPE remains the default. The interesting work is on better merge algorithms (greedy vs marginal-likelihood) and dynamic vocab selection per document.

## Chat templates are tokenizer state

![fig5: chat template token boundaries](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/02-tokenization/fig5_chat_template.png)


A chat template is a Jinja string that converts a list of messages into a single tokenizable string. For LLaMA-3:

```jinja
{% for m in messages %}
<|start_header_id|>{{ m.role }}<|end_header_id|>

{{ m.content }}<|eot_id|>
{% endfor %}
```

The four highlighted tokens — `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, plus `<|begin_of_text|>` — are *single special tokens*, each encoded to one ID. They are part of the tokenizer's `added_tokens.json`. They are also the tokens the model was post-trained to recognize as turn boundaries.

Two production failure modes I've seen:

**Failure 1: building the prompt as a plain string.** Someone writes:

```python
prompt = f"User: {user_msg}\nAssistant:"
out = model.generate(tokenizer.encode(prompt))
```

This works. Sort of. The model has never seen that format during post-training. It might generate, but instruction-following and safety behaviors are degraded because the conversation tokens that gate its trained behavior aren't there. Always use `tokenizer.apply_chat_template(messages)`.

**Failure 2: tokenizing the rendered template with `add_special_tokens=True`.** This double-adds the BOS token. The model sees `<|begin_of_text|><|begin_of_text|>` and gets weird. Use `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` — it knows whether to add specials.

The chat template is also what makes function calling work. Qwen3's chat template includes `<tool_call>...</tool_call>` tags around JSON tool calls, and the model is trained to emit them. Mistral uses `[TOOL_CALLS]`. OpenAI's format uses no special tokens at all and does it via JSON in the message payload. [Chapter 7](/en/llm-engineering/07-function-calling/) covers function calling end to end.

## Common Pitfalls

The five tokenization gotchas I've personally lost the most time to:

**1. Mixing tokenizers between training and inference.** Trained on `tokenizer.json` from HuggingFace, served with `tiktoken` for "speed." The vocabularies don't match exactly — even with the same BPE merge rules, special token handling differs. Result: model emits special tokens that the inference layer doesn't strip, or the inference layer requests tokens the model never saw. Always use the *exact same* tokenizer file for training and inference.

**2. Whitespace normalization on the input.** Several tokenizers (SentencePiece-based) replace leading spaces with `▁` (U+2581). If you `tokenizer.decode(tokenizer.encode(s))`, the result has extra spaces or no spaces depending on the variant. This shows up in chained pipelines: "summarize this output of model A" passed back to model B can have shifted whitespace and degrade quality.

**3. Treating tokens as characters.** A common mistake: setting `max_tokens=200` thinking it means "200 characters". For Chinese on Qwen3, 200 tokens ≈ 200-260 characters; for English, 200 tokens ≈ 800-1200 characters. Length budgets must be reasoned about in the model's tokens, not characters or bytes.

**4. Truncating inside a special token.** Cutting a 100K-token prompt to 32K can split `<|im_end|>` mid-token if the truncation logic operates on bytes. The model sees a corrupted special token and either ignores it or generates nonsense. Always truncate at token boundaries, never at byte/char boundaries inside a tokenized stream.

**5. Stop-token mismatches.** OpenAI-compatible APIs accept `stop=["</s>"]`, but `</s>` may or may not be in the model's vocab as a single token. If it's BPE'd into multiple tokens, the stop-string detection has to scan the decoded text after each generation step, which is slow and racy at the streaming boundary. Modern serving frameworks (vLLM, SGLang) handle this with EOS-token-ID matching plus rolling string detection, but custom servers often get it wrong.

## Production reality: tokenizer in the serving stack

A serving framework treats the tokenizer as the first and last hop of every request. The hot path is:

1. Receive request → tokenize prompt → push token IDs to scheduler
2. Generate next token IDs
3. Detokenize to text → stream back to client

Three production constraints shape the implementation:

**Tokenization throughput.** vLLM and SGLang use Rust tokenizers (the `tokenizers` library from HuggingFace, written in Rust) for ~10× speedup over the Python `transformers` AutoTokenizer. At 1000 RPS with 500-token prompts, Python tokenization can saturate a CPU core; Rust handles it on a fraction of one.

**Streaming detokenization.** Generating one token at a time and detokenizing each requires buffering, because BPE tokens can be partial UTF-8 sequences (especially for CJK). vLLM's detokenizer keeps a buffer of recent tokens, attempts to decode, and only emits text up to the last complete UTF-8 boundary. This adds 1-2 token of latency to streaming output.

**Tokenizer caching.** Many serving systems cache tokenization results for system prompts that don't change. Anthropic's prompt caching feature is partly this — the first ~1000 tokens of every request are pre-tokenized and re-used across requests with the same prefix. Combined with KV-cache prefix sharing, this can drop time-to-first-token by 50-70 % for chat applications with long system prompts.

## What about character-level and visual tokenizers?

Two recent threads worth knowing about, neither yet mainstream.

**Character-level / byte-level pure**: ByT5, MEGABYTE, MAMBA-byte. Skip the BPE merge step entirely; train on raw bytes. Simpler, no tokenization bugs, no CJK bloat. The cost is sequence length: a Chinese paragraph that's 100 BPE tokens becomes 600 bytes. Even with sub-quadratic attention this is expensive. MEGABYTE [Yu et al., 2023] handled this by chunking bytes hierarchically. Promising but not yet competitive at general-purpose scale.

**Visual tokenizers**: render text as images, tokenize visually. Donut [Kim et al., 2022], PIX2STRUCT [Lee et al., 2023]. The pitch is unified text-vision modeling and freedom from tokenizer choice. Not yet ready for general LLM workloads but interesting for OCR-heavy domains.

For 2026 production: byte-level BPE remains the default. The interesting work is on better merge algorithms (greedy vs marginal-likelihood) and dynamic vocab selection per document.

## Research frontier 2024-2026

What's coming after the byte-level BPE consensus:

**Learned tokenization.** [Yu et al., 2024] (SpaceByte) showed you can train the tokenizer end-to-end with the language model, letting the model decide its own subword units. Quality matches BPE, no hand-crafted regex. Not yet in production but a credible threat to BPE.

**Dynamic vocabularies.** Per-document vocab adaptation — e.g., a tokenizer that knows it's processing Python code and switches to a code-tuned merge table. Initial work ([Provilkov et al., 2020] BPE-dropout, [Kudo, 2018] subword regularization) explored stochastic tokenization at training time; the 2024-2025 wave explores deterministic but context-aware tokenization at inference.

**Multimodal tokenizers.** Models that handle text, images, audio, and video need a unified tokenization scheme. The current dominant approach (LLaVA, Qwen-VL, Gemini) tokenizes images with a separate vision encoder and inserts vision tokens into the text stream. Whether the next generation moves to a unified tokenizer for all modalities is an open question. [Team, 2024] (Chameleon) showed early-fusion all-modality tokenization can work.

## What's Next

Tokenization is invisible until it bites: 2x cost on CJK workloads if you pick the wrong tokenizer; instruction-following degradation if you skip the chat template; a fine-tuned model that's worse than the base if you naively expanded the vocab. Always measure tokens on your actual prompts before picking a model. Always use the model's official chat template via `apply_chat_template`. Don't expand vocabularies unless you have hundreds of billions of training tokens to spare.

Next chapter: **pretraining at scale**. Data mixture, deduplication, the quiet 200B-token cliff where most open data sets stop helping, FSDP vs ZeRO-3, and what actually goes wrong when you scale a LLaMA-style training run from 8 GPUs to 8000.

## References

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
