---
title: "LLM Engineering (10): Evaluation"
date: 2026-04-05 09:00:00
tags:
  - LLM
  - evaluation
  - benchmarks
  - llm-as-judge
  - ab-testing
categories: LLM Engineering
series: llm-engineering
series_order: 10
series_title: "LLM Engineering"
lang: en
mathjax: false
disableNunjucks: true
description: "Why MMLU is broken, the contamination problem, LLM-as-judge biases, position-bias mitigation, calibration, and the A/B testing patterns that actually catch regressions in production."
translationKey: "llm-engineering-10"
---

Evaluation is the part of the LLM stack where everyone has opinions but no one is confident. The leaderboards are gamed, the public benchmarks are contaminated, and most teams I've worked with had no eval set when I joined. This chapter covers what evaluation actually tells you, what the benchmarks hide, the LLM-as-judge biases that go unaddressed, the calibration metrics most teams skip, and the production patterns that catch regressions before customers notice.

The chapter has a slightly different flavor from the others in this series. Most evaluation problems are not technical — they are *epistemic*. The question "is model A better than model B" is a hypothesis-testing question, and the field's collective track record at running clean experiments is poor. The literature I cite below is not a leaderboard; it's a collection of failure-mode papers that should make any practitioner more cautious.

![LLM Engineering (10): Evaluation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/illustration_1.png)

---

## Why public benchmarks lie

![fig1: benchmark contamination over time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig1_benchmark_contamination.png)


Here are some issues with the standard benchmark suite (MMLU, GSM8K, HumanEval, ARC, HellaSwag, etc.):

**Contamination.** Most public benchmarks are accessible via web crawl. Models trained on CommonCrawl have seen the questions and often the answers. Phi-3 was caught with literal MMLU questions in its training data; it's hard to believe other labs are cleaner. Zhou et al. (2023, *Don't Make Your LLM an Evaluation Benchmark Cheater*) measured the effect systematically — even small amounts of test-set leakage produce 10-30 percentage point gains, indistinguishable from "legitimate" capability improvements. The paper's most uncomfortable finding: random sub-sampling of CommonCrawl already includes 5-10 % of the popular benchmark questions. Avoiding contamination requires active filtering, which most labs claim and few document.

A related paper by Sainz et al. (2023, *NLP Evaluation in Trouble*) surveyed over 250 recent benchmark releases and found that about 40% had detectable leakage in at least one major foundation model's training corpus. The leakage rate has increased with the size of pre-training corpora.

**Saturation.** MMLU was designed to be hard for 2020 models. Top models in 2026 score 88-92 % — they're competing in the noise band. A 1.5-point improvement on MMLU may be a real improvement or may be sampling variance from a 50-question pull on the boundary cases. The benchmark stopped discriminating capability years ago. A simple statistical argument: with 14K questions and a model at 90 % accuracy, the standard error on the score is roughly $\sqrt{0.9 \cdot 0.1 / 14000} \approx 0.25$ percentage points; a 1-point claim that doesn't beat ~0.7 points is within noise.

**Multiple-choice bias.** MMLU is multiple-choice with 4 options. A model that always picks "B" gets 25 %. Models trained heavily on multi-choice data have an advantage that doesn't transfer to free-form output. Worse, position bias inside multi-choice prompts is real — Wang et al. (2023, *Large Language Models are not Fair Evaluators*) showed that shuffling answer positions changes accuracy by 2-8 points on most models.

**English-only.** MMLU, GSM8K, HumanEval are all English. Models excellent in Chinese or French may score modestly on the canonical leaderboard but win their actual deployment.

**Format coupling.** GSM8K answers are extracted via regex looking for "the answer is X". Models that don't follow that exact format get marked wrong even when correct. A 2024 audit by HuggingFace's evaluation team estimated 5-15 % of "wrong" answers on GSM8K are correct content with non-canonical formatting.

The 2024-2026 generation of benchmarks (MMLU-Pro, GPQA, BIG-Bench Hard, RULER, SWE-bench, LiveBench, ArenaHard, IFEval) try to address these but each has its own gaming problem within months of release.

## What MMLU actually measures (still)

Despite all the criticism, MMLU isn't useless. It measures something — broadly, "did the model see and retain a lot of college-level knowledge during pretraining?" That correlates loosely with general capability. A model scoring 50 on MMLU is genuinely worse than one scoring 80; the gap from 88 to 92 is mostly noise.

Use MMLU as a coarse filter ("is this model in the right zip code?"). Don't use it to compare two models in the 85+ range.

## The benchmark zoo: what to use when

A pragmatic guide to which benchmark answers which question:

- **General knowledge / coarse capability**: MMLU (use only as a filter), MMLU-Pro (still meaningful at the frontier).
- **Reasoning under uncertainty**: GPQA Diamond (Rein et al., 2023) — graduate-level science questions designed to resist LLM-friendly tricks. Top-2026 models score 60-75 %; humans with PhDs in the relevant field score ~65 %. This benchmark is genuinely hard.
- **Math, easier**: GSM8K (Cobbe et al., 2021, *Training Verifiers to Solve Math Word Problems*) — 8.5K grade-school word problems. Saturated by frontier models (>95 %) but still useful for smaller models.
- **Math, harder**: MATH (Hendrycks et al., 2021, *Measuring Mathematical Problem Solving*) — 12.5K competition problems from AMC/AIME-level. Still discriminative at the frontier (top models 85-92 %). Subsumed in 2025 by AIME-2025 and Putnam-2025 for actual competition rigor.
- **Code, contained**: HumanEval (Chen et al., 2021, *Evaluating Large Language Models Trained on Code*) — 164 hand-written Python problems. Saturated. MBPP (Austin et al., 2021) is similar but slightly broader.
- **Code, realistic**: SWE-bench (Jimenez et al., 2023) — real GitHub issues from 12 popular Python repos that the model must resolve by editing the repo. Multi-file, requires understanding existing code. Top models in 2026 hit 50-60 % on SWE-bench Verified.
- **Code, broad**: BigCodeBench (Zhuo et al., 2024) — 1140 tasks across 723 functions and 139 libraries. Resists the GitHub contamination that affects HumanEval clones.
- **Instruction following**: IFEval (Zhou et al., 2023) — verifies whether the model satisfies explicit constraints ("respond in exactly 3 paragraphs, each starting with a question"). Programmatic verification, low ambiguity.
- **Long context**: RULER (Hsieh et al., 2024) — needle-in-haystack at multiple difficulties up to 128K tokens. Most "1M context" claims fall apart at 32K when measured here.
- **Hallucination**: SimpleQA (OpenAI, 2024) — 4326 short-answer factual questions designed to expose confabulation. Frontier models score 30-55 %; lower-tier models near 10 %.
- **Pairwise human preference**: Chatbot Arena / ArenaHard (lmsys) — millions of crowdsourced A-vs-B votes on real chat interactions.

For production work the relevant question is rarely "which model wins MMLU" and almost always "which model wins on my workload." The benchmark zoo is useful mainly for *eliminating obviously bad candidates* before you spend money on a custom eval.

## LiveBench, ArenaHard, and dynamic benchmarks

The 2024-2025 response to contamination was **dynamic benchmarks**: questions generated or curated freshly each month, never published before testing.

- **LiveBench** (released late 2024, refreshed monthly): math, coding, reasoning, language, instruction-following questions sourced from recent (post-cutoff) papers and competitions. Hard to contaminate.
- **ArenaHard / Chatbot Arena** (lmsys): human pairwise preferences from real chat traffic. Slow to update but resistant to gaming because humans choose. The Arena's published leaderboard suffers a different bug — *style preference* dominates *correctness preference* for many users, so the rankings reward warm, well-formatted, slightly verbose answers over correct terse ones.
- **SimpleQA** (OpenAI, 2024): factual questions with short specific answers. Designed to expose hallucination — a model that confabulates loses.

These are better signal than MMLU in 2026 but still have problems. ArenaHard heavily weights what users find pleasant ("nice formatting", "warmth"), which can decouple from correctness. LiveBench's question selection process is itself gameable if you know the curation criteria. *No public benchmark survives contact with sustained optimization pressure*.

## The contamination defense: hold-out evals

For production work, you should not be evaluating on public benchmarks. Build your own eval set:

1. Sample 100-500 representative inputs from real (or synthetic-but-realistic) production traffic.
2. Hand-write or hand-curate gold-standard outputs.
3. Never share this set publicly. Never put it in any prompt that hits an external API (vendors might learn from your traffic).
4. Re-evaluate every model change against this set.

The investment is real (a 200-question hand-labeled set is 1-2 person-days) but pays back the first time it catches a regression that public benchmarks didn't.

A subtle point: **how you split the eval set matters as much as what's in it**. A common mistake is sampling 200 questions uniformly from production traffic, when production is 80 % easy and 20 % hard. Your eval becomes dominated by easy questions where every model scores 95 %. Stratify: aim for 30-40 % easy, 30-40 % medium, 30-40 % hard, where "hard" is sourced from your error log. The hard slice is the part that discriminates models.

## LLM-as-judge: the dominant pattern and its failure modes

![LLM Engineering (10): Evaluation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/illustration_2.png)


![fig5: human vs auto-eval correlation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig5_human_vs_auto.png)


![fig2: LLM-judge position bias](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig2_judge_position_bias.png)


For free-form outputs (most production tasks), exact match doesn't work. You need a way to score "is this answer good." The dominant pattern is **LLM-as-judge** (Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*): a strong model scores outputs from the model under test.

Basic pattern:

```python
JUDGE_PROMPT = """You are evaluating an AI assistant's response.

Question: {question}

Reference answer (for grading reference; the assistant did not see this):
{reference}

Assistant's answer:
{candidate}

Rate the assistant's answer on a 1-5 scale:
1 = factually wrong or off-topic
2 = partially correct but missing key information
3 = correct but poorly explained
4 = correct and well explained
5 = correct, well explained, and adds useful related information

Output only the integer score."""

def judge(question, reference, candidate, model="claude-4-5-sonnet-20250901"):
    score = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}]
    )
    return int(score.content[0].text.strip())
```

LLM-as-judge correlates with human judgment 70-85 % depending on task. It's much cheaper than human eval and scales to thousands of examples per day. But it has biases the Zheng paper enumerated and subsequent work refined:

- **Position bias**: when comparing two answers, judges prefer the one shown first (or, for some models, second). Zheng measured 30-65 % of pairwise judgments flip when the order is swapped on GPT-4-as-judge — far above chance. Mitigation: present both orderings, average.
- **Length bias**: longer answers get higher scores even when shorter is better. Dubois et al. (2024, *Length-Controlled AlpacaEval*) measured a +3-7 point lift just from doubling answer length with no quality change. Mitigation: explicitly tell the judge "do not reward length unless useful," or use length-controlled scoring (subtract a length term, or normalize against a length-matched baseline).
- **Self-preference**: GPT-4 judges prefer GPT-4 outputs; Claude prefers Claude. Panickssery et al. (2024, *LLM Evaluators Recognize and Favor Their Own Generations*) showed this is a self-recognition phenomenon — the judge model can identify its own outputs and rates them higher. Mitigation: use a different judge family than the model under test, or use a panel of judges from different families and majority-vote.
- **Format bias**: structured (Markdown, headers) outputs score higher than equivalently good prose. Mitigation: normalize formatting before scoring.
- **Sycophancy / verbosity-as-rigor**: judges sometimes side with whichever answer is more confident or more thorough-looking even when wrong. Mitigation: include a reference answer when possible, and tell the judge to weight correctness over presentation.

For pairwise comparison (model A vs model B), the bias-corrected approach is:

```python
def pairwise_judge(question, answer_a, answer_b, judge_model):
    # First order: A first
    score_ab = judge(question, answer_a, answer_b, judge_model)
    # Reverse order: B first
    score_ba = judge(question, answer_b, answer_a, judge_model)
    # Reverse the second score and combine
    if score_ab == "A" and score_ba == "B": return "A"
    if score_ab == "B" and score_ba == "A": return "B"
    return "tie"
```

Zheng et al. showed swap-consistent judgments correlate ~85 % with human; un-swap-consistent ones are essentially noise. Throwing them out is the easiest correction.

**Pointwise vs pairwise.** The two judging paradigms have different failure modes. Pointwise (rate each answer 1-5 independently) is cheaper but suffers from rating drift across batches and judges' implicit anchoring on the most recent example. Pairwise (which is better, A or B?) is more reliable per comparison but quadratic in the number of models. The standard production compromise: pointwise for triage (filter out clearly bad candidates), pairwise for the final 2-3 candidates that matter.

## Code and math: program-based eval

For verifiable domains, skip the judge:

- **Code**: run the candidate against a test suite. HumanEval, MBPP, SWE-bench all do this. Pass@k measures the fraction of $k$ samples that pass.
- **Math**: extract the final numerical answer, compare to ground truth.
- **JSON output**: validate against schema, check key presence.
- **Tool calls**: simulate the tool, check the call's effect.

Program-based eval is unbiased, free to compute, and instant. Where you can use it, use it. The frontier reasoning models (o-series, Qwen3-Reasoning, DeepSeek-R1) all use program-based eval as their primary signal during RLVR (chapter 4).

A useful framing: every domain that admits a programmatic verifier becomes a *training signal* in RLVR, which means model capability on that domain grows roughly with verifier quality. Math and code grow faster than prose because they have better verifiers. If you're building an in-house benchmark for a verifiable task, you're not just measuring — you're producing a potential training signal that an open-source community will eventually pay you back for.

## A/B testing in production

![fig4: A/B test power calculation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig4_ab_power.png)


Eval sets catch regressions before deploy. A/B testing catches what the eval set missed.

Production A/B: route 5-10 % of traffic to the new model variant, compare outcome metrics:

- **Engagement**: did the user continue the conversation?
- **Resolution**: did the user mark the conversation resolved?
- **Latency**: did p50/p95 worsen?
- **Cost**: tokens per turn, $ per resolved request.
- **Satisfaction**: thumbs up/down rate.
- **Escalation**: did the user ask for a human?

Surface metrics like "engagement" can mislead — a model that goes off-topic in interesting ways drives more turns but solves fewer problems. Pair turn count with resolution rate.

For statistical significance: 10K traffic is enough to detect a 2 % shift in a binary metric at 95 % confidence. The relevant power calculation:
$$n \approx \frac{16 \cdot p (1-p)}{\delta^2}$$
where $p$ is the baseline rate and $\delta$ is the minimum detectable effect. For $p = 0.5$, $\delta = 0.02$, that's $n \approx 10{,}000$ per arm; for $\delta = 0.005$ (catching half-percent shifts), $n \approx 160{,}000$. Run experiments at least 7 days to wash out day-of-week effects. Stratify by traffic segment if your traffic is heterogeneous (free vs paid, mobile vs desktop, language).

Two production patterns worth knowing:

- **Latency-aware quality metrics**. A new model that is 1 % more accurate but 200 ms slower may net out negative because user dropoff scales with latency. Combine quality and latency into a single objective (e.g., quality - $\alpha$ · latency_seconds, for an $\alpha$ calibrated against historical drop-off curves) before declaring a winner.
- **Sequential A/B with early stopping**. Frequentist t-tests assume a fixed sample size; if you peek and stop early, your false-positive rate inflates. Use a sequential test (mSPRT, or simply Bonferroni-correct over the number of times you peek) to keep the false-positive rate honest while still getting fast decisions.

## Eval set maintenance

The eval set you build today will be obsolete in 6 months. Production traffic shifts, customer use cases evolve, and your model's failure modes are not the failure modes of next quarter's model.

Maintenance routine that works:

- **Weekly**: sample 50 production calls, mark anomalies, add interesting failures to the eval set.
- **Monthly**: re-run all production prompts against the eval set, plot quality trend.
- **Quarterly**: retire eval items that all current models pass at 100 % (no longer informative). Add new items from recent failure modes.
- **On every model change**: full re-evaluation, no exceptions. If the eval is too slow to run on every change, automate it and parallelize.

A useful invariant: the *interquartile range* of model accuracy across your eval set should not collapse to zero. If every model scores 99 % on every question, you've over-saturated and the eval is no longer doing work. Cull the easy questions, source new hard ones.

## Calibration: a metric people skip

![fig3: calibration plot (ECE)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig3_calibration.png)


A model is **calibrated** if its stated confidence matches actual accuracy. A model that says "I'm 90 % confident" should be right 90 % of the time. Most LLMs are systematically overconfident — they'll say "I'm sure" and be wrong 30 % of the time.

For high-stakes deployments (medical, legal, financial), calibration matters as much as accuracy. Measure with the Expected Calibration Error (Guo et al., 2017, *On Calibration of Modern Neural Networks*):
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$
Bin predictions by confidence into $M$ buckets, compute |accuracy - average confidence| per bucket, weighted by bucket size. A reliability diagram plots accuracy on the y-axis against confidence on the x-axis; a perfectly calibrated model lies on the diagonal.

The Guo paper showed that *accuracy improvements often come at the cost of calibration* — the techniques (deep networks, label smoothing, temperature scaling) that boost accuracy on classification benchmarks tend to harm calibration unless explicitly corrected. The same lesson transferred to LLMs: post-training (RLHF, DPO) reliably degrades calibration, because the reward model rewards confident-sounding answers more than hedged ones.

Frontier models in 2026 still have ECE in the 5-15 % range — meaningful overconfidence. RLHF tends to *worsen* calibration (the model learns that confident-sounding answers get higher rewards). For deployments where the model needs to know what it doesn't know, you might need to:

1. **Undo some post-training assertiveness** via SFT on "I don't know" examples sourced from cases where the base model was uncertain but the post-trained model became confident.
2. **Use temperature scaling on logprobs** at inference time. Calibrate the temperature on a held-out set; this is cheap and often shaves ECE in half.
3. **Sample multiple completions and use the entropy of answers as a confidence proxy.** Self-consistency at $N=10$ on a question — if 9/10 agree, high confidence; if 5/10/0/0/etc spread, low confidence.
4. **Fit a calibration model** on top of (model_logprob, prompt_features) → empirical_correctness on a labeled sample.

For RAG systems specifically, *retrieval confidence* (top-k similarity scores, reranker scores) is a more honest confidence signal than the LLM's own generated tokens. Pipe it through.

## Long-form evaluation: what the benchmarks miss

Most public benchmarks evaluate short answers. Production tasks often involve long-form generation (summaries, drafts, code blocks of 500+ lines). The benchmarks that target this:

- **AlpacaEval / AlpacaEval 2** (Li et al., 2023): 805 instructions; LLM-as-judge against a reference (text-davinci-003 originally, later GPT-4). Length-controlled version (Dubois et al., 2024) corrects for the length bias problem above.
- **MT-Bench** (Zheng et al., 2023): 80 multi-turn conversations across 8 categories; LLM-as-judge with a chain-of-thought rationale before the score.
- **Arena Hard** (lmsys, 2024): 500 hard questions distilled from real Chatbot Arena traffic; LLM-as-judge with strong correlation to human Arena rankings.

For internal long-form eval, the practical pattern is:

1. Curate 50-200 prompts that span your real workload.
2. Define a 5-10 dimension rubric (accuracy, completeness, format compliance, tone, hallucination rate, etc.).
3. Have a strong judge model rate each dimension separately, with a one-line rationale.
4. Aggregate to a composite score, but always look at per-dimension breakdown when deciding to ship.

The per-dimension view is critical. A model that gains 5 points on completeness while losing 3 on hallucination is not strictly an improvement; whether to ship depends on which dimension matters more for your product.

## Eval pipeline as code

A mature evaluation system has the same operational hygiene as a CI pipeline:

```text
[eval_set.jsonl in version control]
     ↓
[runner: parallelize across N candidates × M questions]
     ↓
[grader: program-based + LLM-as-judge + optional human spot-check]
     ↓
[result store: per-run scores, broken down by question and dimension]
     ↓
[diff view: candidate vs baseline, highlight regressions]
     ↓
[gate: block deployment if regression > threshold]
```

Tools that work in 2026: **Promptfoo** (simple YAML-driven, good for local iteration), **Inspect AI** (UK AISI's framework, used for safety evals), **OpenAI evals**, **DeepEval**, and roll-your-own with pytest if your taste runs that way. The framework matters less than having one; the worst pattern is a notebook that runs evals once when someone remembers to.

## What's Next

Public benchmarks are mostly noise above the 80-mark and many are contaminated. Build a hand-curated eval set on your real traffic, stratified by difficulty, and treat it like load-bearing code. LLM-as-judge works with bias correction (swap order, length-control, use a different judge family, panel-vote). Use program-based eval whenever the task is verifiable. A/B test in production for what the offline eval misses, and use sequential tests when you peek. Maintain the eval set continuously and cull saturated questions. Don't forget calibration — confidence matters as much as accuracy in many deployments, and post-training reliably degrades it. For long-form, score by dimension before composing.

Next chapter: **safety and alignment**. Refusal behaviors, the RLHF objective, sycophancy, red-teaming methodology, hallucination metrics, and constitutional AI.

## References

- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685
- Zhou, K. et al. (2023). *Don't Make Your LLM an Evaluation Benchmark Cheater*. https://arxiv.org/abs/2311.01964
- Sainz, O. et al. (2023). *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*. EMNLP 2023 Findings. https://arxiv.org/abs/2310.18018
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML 2017. https://arxiv.org/abs/1706.04599
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. https://arxiv.org/abs/2110.14168
- Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS 2021 Datasets. https://arxiv.org/abs/2103.03874
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding* (MMLU). ICLR 2021. https://arxiv.org/abs/2009.03300
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code* (HumanEval). https://arxiv.org/abs/2107.03374
- Austin, J. et al. (2021). *Program Synthesis with Large Language Models* (MBPP). https://arxiv.org/abs/2108.07732
- Jimenez, C. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* https://arxiv.org/abs/2310.06770
- Zhuo, T. Y. et al. (2024). *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions*. https://arxiv.org/abs/2406.15877
- Rein, D. et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*. https://arxiv.org/abs/2311.12022
- Hsieh, C.-P. et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* https://arxiv.org/abs/2404.06654
- Zhou, J. et al. (2023). *Instruction-Following Evaluation for Large Language Models* (IFEval). https://arxiv.org/abs/2311.07911
- Li, X. et al. (2023). *AlpacaEval: An Automatic Evaluator of Instruction-following Models*. https://github.com/tatsu-lab/alpaca_eval
- Dubois, Y. et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*. https://arxiv.org/abs/2404.04475
- Wang, P. et al. (2023). *Large Language Models are not Fair Evaluators*. ACL 2024. https://arxiv.org/abs/2305.17926
- Panickssery, A. et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. https://arxiv.org/abs/2404.13076
- OpenAI (2024). *Introducing SimpleQA*. https://openai.com/index/introducing-simpleqa/
- LMSYS (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. https://arxiv.org/abs/2403.04132
- White, J. et al. (2024). *LiveBench: A Challenging, Contamination-Free LLM Benchmark*. https://livebench.ai/
- UK AISI (2024). *Inspect: An OSS framework for large language model evaluations*. https://inspect.ai-safety-institute.org.uk/
