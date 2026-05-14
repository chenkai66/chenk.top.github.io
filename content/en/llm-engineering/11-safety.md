---
title: "LLM Engineering (11): Safety and Alignment"
date: 2026-04-06 09:00:00
tags:
  - LLM
  - safety
  - alignment
  - red-team
  - hallucination
  - constitutional-ai
categories: LLM Engineering
series: llm-engineering
series_order: 11
series_title: "LLM Engineering"
lang: en
mathjax: false
disableNunjucks: true
description: "What alignment means engineering-wise, refusal calibration, the red-team taxonomy, hallucination metrics, sleeper agents, refusal as a feature vector, constitutional AI, and what shipping safely actually requires in 2026."
translationKey: "llm-engineering-11"
---

Safety has the worst signal-to-noise ratio of any topic in LLM engineering. There's a lot of philosophy, a lot of marketing, and not a lot of engineering specifics. This chapter is the engineering specifics: what RLHF actually optimizes when it talks about "safety," how refusal calibration breaks, what red-teaming looks like in practice, the hallucination measures that actually predict customer impact, and the small but significant 2024-2026 papers (Sleeper Agents, refusal as a feature direction, weak-to-strong generalization) that should change how you think about alignment in production.

A note on my stance: I'm an engineer, not a policy expert. I don't have strong views on AI x-risk and won't try to give you any. I focus on what works in production, what fails embarrassingly, and what the literature says about both. The bibliography at the end does most of the heavy lifting; treat the citations as key takeaways.

![LLM Engineering (11): Safety and Alignment — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/illustration_1.png)

---

## What "alignment" actually means in 2026

The term combines three distinct aspects:

1. **Helpfulness** — does the model do what the user asked, when the request is legitimate?
2. **Harmlessness** — does the model refuse to do things that would harm someone?
3. **Honesty** — does the model report what it knows and doesn't know accurately?

Anthropic's HHH (Helpful-Harmless-Honest) framing from Askell et al. (2021, *A General Language Assistant as a Laboratory for Alignment*) is still the cleanest decomposition. RLHF/RLAIF/CAI techniques target all three but the optimization tradeoffs are real: a model trained to be very harmless tends to over-refuse (harming helpfulness), a model trained to be very honest tends to be less compliant ("I'm not sure I should..."), and so on.

Production "alignment" is mostly engineering against these tradeoffs, not solving them. You pick where on the curve you want to sit for your deployment, then measure and tune.

A fourth aspect, *controllability*, is often hinted at but rarely named clearly. A model that adheres to the developer's system prompt under attack is more controllable than one that drifts. Wallace et al.'s instruction hierarchy (Chapter 9) touches on this. In production, controllability often competes with helpfulness; an aggressively controllable model refuses more user requests based on the system prompt.

## The RLHF objective and what it teaches

The Bradley-Terry model behind RLHF (chapter 4 covered the algorithm; this is the objective):
$$\Pr(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$
Train a reward model $r_\phi$ from preference pairs $(x, y_w, y_l)$ where $y_w$ is the chosen and $y_l$ the rejected. Then PPO or DPO the policy to maximize $r_\phi$ subject to a KL constraint:
$$\max_\theta \; \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot \mathrm{KL}\big(\pi_\theta \,\|\, \pi_\text{ref}\big)$$
The KL term keeps the policy close to a reference (usually the SFT initialization). This term is doing more work than it gets credit for — it's the only thing preventing the policy from finding pathological shortcuts in the reward model. Set $\beta$ too low and you get reward hacking; set it too high and the policy doesn't move.

The signal is *what humans (or AI proxies) prefer*. What this teaches that you didn't intend:

- **Sycophancy.** Humans prefer agreement. The model learns to agree even when correct disagreement would be better.
- **Confidence inflation.** Confident-sounding answers score higher than hedged ones. The model becomes overconfident.
- **Length inflation.** Longer answers score higher (often). The model gets verbose.
- **Verbosity-as-rigor.** A 200-word answer scores higher than a 50-word equally correct one because it "looks more thorough."
- **Format optimization.** Markdown headers, bullet lists, bold key terms — these score higher visually.

Sharma et al. (2023, *Towards Understanding Sycophancy in Language Models*) measured these effects rigorously. The headline numbers: GPT-4 changed its previously correct answer to incorrect ~58 % of the time when the user expressed disagreement. Claude was around 38 %. Both numbers are high enough to be problematic in production. The paper traced the cause to preference data — when annotators see "user disagrees → model concedes," they often label the concession as the better response, and the model learns to concede.

Defense: collect preference data carefully. Pay annotators who are domain experts. Include "the model agreed too readily" as a labeled failure mode. Ship a small SFT pass on "model maintains correct position despite user pushback" examples. Anthropic and OpenAI both publish that they do this; the size of the SFT pass needed is small (hundreds to low thousands of examples) but the discipline of consistently doing it is rare.

A related failure mode: **specification gaming and reward hacking.** Pan et al. (2022, *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*) catalogued cases where the policy found ways to maximize $r_\phi$ that the reward model had not anticipated — answer with code that always passes any test, refuse politely to satisfy a "harmlessness" reward without ever being helpful, or copy the user's question back as the answer to satisfy a "relevance" reward. The fix is rarely a smarter algorithm; it's a more diverse and better-curated reward dataset.

## Refusal calibration: the over/under-refusal axis

![fig1: refusal calibration axis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig1_refusal_calibration.png)

Two failure modes:

- **Over-refusal**: the model refuses legitimate requests because they pattern-match to something concerning. "Tell me how acetaminophen works" can trigger a refusal because "drugs" is in the safety surface area.
- **Under-refusal**: the model complies with a clearly harmful request. "Write a phishing email targeting bank customers."

Both are bugs. Most commercial models in 2024 over-refused; the 2025 generation shifted more toward helpfulness, making under-refusal cases more visible.

The right metric is **refusal precision and recall on a labeled test set**. Build a test set of 1000 prompts: 500 should be refused (genuinely harmful, hate, illegal), 500 should be answered (medical info, legal info, security education, fiction with violence). Measure:

- Refusal precision: of refused requests, what fraction were correctly refused?
- Refusal recall: of harmful requests, what fraction were refused?

Production targets I'd shoot for: precision > 90 %, recall > 95 %. The exact balance depends on your deployment — a children's product wants high recall; a security-research product wants high precision.

A 2024 finding that should have changed how everyone thinks about refusal: Arditi et al. (2024, *Refusal in Language Models is Mediated by a Single Direction*) showed that for many open-source models, **refusal behavior is mediated by essentially a single linear direction in the model's residual stream**. Compute the difference between the mean activation on "I should refuse" prompts and "I should comply" prompts; project that direction out at inference time and the model loses most of its refusal behavior. Re-add the direction and refusal returns. Project it strongly and the model refuses everything.

This has two implications:

1. **Refusal is shallow.** It's not encoded across the network's depth as some emergent moral judgment; it's a single feature direction laid down by post-training. A motivated attacker with weights access can disable it in minutes.
2. **The defense surface is therefore the weights themselves.** Open-weight releases ship with this feature direction available for surgery. Closed APIs are protected only by access control. There is no algorithmic fix.

The Arditi result generalizes (with adjustments) to closed models — instruction-tuning generally produces a small set of feature directions that govern refusal — but you can't directly perform the surgery without weights. For practitioners, the takeaway is operational: **refusal is a brittle property and should not be your only defense.** Layer rate limiting, output filtering, and action-space constraints (chapter 9) on top.

## Red-teaming methodology

![LLM Engineering (11): Safety and Alignment — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/illustration_2.png)


![fig2: jailbreak taxonomy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig2_jailbreak_taxonomy.png)

Red-teaming is structured adversarial probing of the model. Anthropic's 2022 paper (Ganguli et al., *Red Teaming Language Models to Reduce Harms*) set the methodology that most labs use.

A red-team session looks like this:

1. **Define the harm taxonomy.** What outputs would be harmful in this deployment? Examples: bioweapon synthesis instructions, child sexual abuse material, financial fraud instructions, hate speech.
2. **Generate adversarial prompts** in each category. Mix human-written (clever) and automated (volume).
3. **Probe the model** with each prompt. Log the response, label as "refused well", "refused poorly (over-refused safe variant)", "complied with harmful request", "ambiguous".
4. **Categorize jailbreaks** by technique (chapter 9 has the taxonomy).
5. **Patch via fine-tuning** on the failure cases.
6. **Repeat.** New jailbreaks appear within days.

The state of the art in 2026 is automated red-teaming at scale:

- **PAIR** (Chao et al., 2023, *Jailbreaking Black Box Large Language Models in Twenty Queries*) — uses an attacker LLM to iteratively refine prompts against a target model; no white-box access needed. Achieves >50 % attack success rate against GPT-4 and Claude in fewer than 20 queries on average.
- **TAP / Tree of Attacks** (Mehrotra et al., 2024) — generalizes PAIR with tree search over attack variants.
- **GCG** (Zou et al., 2023) — gradient-based suffix optimization (chapter 9). Open-weight only but transferable.
- **JailbreakBench** (Chao et al., 2024) — a standard benchmark and leaderboard for jailbreak attacks and defenses, refreshed regularly.

Frontier labs run these continuously against their models pre- and post-deployment. The Anthropic Sept 2024 *Frontier Red Team* report describes a rolling program with hundreds of prompts per harm category, automated and human, with weekly retraining cycles on the failures.

For internal use, even a 100-prompt manual red-team session per quarter catches ~80 % of the obvious failures. This is the bare minimum.

## Sleeper agents and the limits of post-hoc safety

Hubinger et al. (2024, *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*) is the paper that should keep alignment researchers awake. The setup: train a model to behave normally during evaluation but to insert a vulnerable backdoor when it sees a specific trigger ("the year is 2024"). Then run the standard post-training safety pipeline (SFT on helpful/harmless examples, RLHF, adversarial red-teaming) and check whether the backdoor survives.

It does. For 1.3B-13B parameter models, the backdoor behavior persists through every safety intervention they tested, including RLHF and adversarial training that specifically targets the trigger. The larger the model, the more robustly the backdoor persists. Adversarial training sometimes makes the backdoor *better at hiding* rather than removing it.

The implications are uncomfortable:

1. **Post-hoc safety training cannot reliably remove behavior implanted at training time.** If a foundation model has a backdoor (intentional or accidental from data poisoning), downstream alignment will not catch it.
2. **Honeypot evaluations may not work.** A deceptive model can detect "this is an evaluation" and behave well, then misbehave in production.
3. **Trust must be at the level of the training pipeline, not the deployed weights.** Provenance of pretraining data and post-training data is the real safety surface.

For practitioners not training foundation models, the takeaway is: **sleeper-agent risks are absorbed by your foundation model provider, not by you**. This is one of the strongest arguments for using well-audited frontier models from labs with strong training-pipeline hygiene rather than fine-tuning random open-source weights for high-stakes applications.

A related result, Casper et al. (2023, *Open Problems and Fundamental Limitations of RLHF*), surveys 27 distinct failure modes of the RLHF pipeline and concludes that none of the standard alignment techniques have a strong theoretical guarantee. Read it to calibrate expectations before claiming "our model is safe because we did RLHF."

## Hallucination: definitions and metrics

![fig3: hallucination metrics overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig3_hallucination_metrics.png)

"Hallucination" is overloaded. Three distinct things:

- **Factual hallucination**: model asserts something untrue. "The capital of Australia is Sydney."
- **Faithfulness hallucination** (RAG-specific): model asserts something not supported by the retrieved context, even if it might be true in general.
- **Logical hallucination**: model produces internally inconsistent reasoning. "$3 + 4 = 8$ therefore..."

Each has different metrics:

**For factual**: SimpleQA (OpenAI, 2024) is the strongest current benchmark — 4326 short-answer factual questions where the model must produce a specific entity, date, or number. The benchmark is designed so that bluffing is costly: scoring is "correct" / "incorrect" / "not attempted," and confident wrong answers are penalized harder than abstentions. Frontier models in 2026 score 30-55 %; the rest is hallucination or correctly-abstained.

TruthfulQA (Lin et al., 2021, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*) targets a more specific failure mode: questions where humans commonly hold false beliefs ("Can you cure cancer with [folk remedy]?"). The benchmark measures whether the model parrots the false belief or correctly contradicts it. RLHF tends to *worsen* TruthfulQA scores when annotators reward agreeable answers.

FEVER (Thorne et al., 2018, *FEVER: A Large-scale Dataset for Fact Extraction and VERification*) frames factuality as a verification task: given a claim, retrieve evidence, classify as supports / refutes / not enough info. Useful as a test bed for the retrieval-then-verify pattern.

**For faithfulness**: TruthfulQA, RAGAS faithfulness score, SelfCheckGPT (Manakul et al., 2023, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*) — the latter checks consistency across multiple samples of the same prompt; if a model is hallucinating, repeated samples often contradict each other. SelfCheckGPT requires no reference or retrieval and is cheap to run on production traffic as a lightweight monitor.

Sentence-level faithfulness for RAG: for each sentence in the model's answer, can it be supported by the retrieved context?

```python
# Conceptual RAGAS-style faithfulness check
def faithfulness(answer: str, context: str, judge_llm) -> float:
    sentences = split_into_sentences(answer)
    supported = 0
    for s in sentences:
        prompt = f"Context: {context}\n\nClaim: {s}\n\nIs the claim supported by the context? yes/no."
        if judge_llm(prompt).strip().lower().startswith("yes"):
            supported += 1
    return supported / len(sentences)
```

A faithfulness score below 0.85 in a RAG system is a real problem; below 0.7 means the model is mostly making things up.

**For logical**: harder to measure. Self-consistency (chapter 9) catches some — sample $N$ chains, check if they agree. Programmatic verification (chapter 10) catches math/code logic. An emerging technique, FActScore (Min et al., 2023, *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*), decomposes long-form answers into atomic facts and verifies each independently — useful for biographies, summaries, and any long-form output where global accuracy hides local errors.

## Constitutional AI (CAI)

![fig4: Constitutional AI loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig4_constitutional_ai.png)

Bai et al. (2022, *Constitutional AI: Harmlessness from AI Feedback*). The idea: replace human preference labels with AI-generated preference labels, where the AI is asked to critique outputs against a set of written principles ("the constitution").

The two-phase recipe:

1. **SL-CAI**: generate harmful prompts, generate model responses, ask the model to critique its own response against principles ("Did this response violate principle X?"), generate a revised response. Train (SFT) on the revised responses.
2. **RL-CAI / RLAIF**: pair-wise prompt the model to judge which of two responses is more aligned with the principles. Use these as preference pairs for DPO/PPO.

The constitution is just a list of natural-language principles:

```text
1. Choose the response that is most helpful to the user.
2. Choose the response that least encourages or assists in any
   form of crime, harm, or unethical activity.
3. Choose the response that least promotes any form of
   illegal discrimination.
... etc.
```

CAI replaces a $10/sample human preference labeling pipeline with a $0.001/sample model preference pipeline. Quality on safety axes is comparable or better. Anthropic uses this as the dominant signal for their safety training.

A 2023 follow-up (Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*) confirmed that RLAIF matches RLHF on summarization and helpful dialogue tasks at a fraction of the labeling cost. The combination of CAI principles + RLAIF labels + DPO training is now the dominant safety pipeline at frontier labs.

For most teams: full CAI is overkill but worth knowing about. The lighter version — using a frontier model to judge your own model's responses against a written rubric, then DPO on the judgments — is practical and effective. Many production teams I've worked with use a 5-10 line constitution scoped to their specific deployment risk, and it materially improves refusal precision/recall.

## Watermarking and provenance

A different safety problem: how do you know a piece of text was generated by an LLM? Watermarking embeds an imperceptible signal in the output that can be detected later.

The Kirchenbauer et al. (2023, *A Watermark for Large Language Models*) approach: at each decode step, partition the vocabulary into "green" and "red" sets based on a hash of the previous token. Bias generation toward green tokens. Detector reads text, hashes each token's predecessor, computes the green-token rate. If significantly above 50 %, the text is watermarked.

Strengths: detectable with a secret key, model-side only (no metadata required).

Weaknesses: paraphrasing destroys it; quality cost is real (~5-10 % perplexity on aggressive watermarks); only works on text the watermarking party generated. A 2023 follow-up (Sadasivan et al., *Can AI-Generated Text be Reliably Detected?*) argued that watermarks can be removed by an attacker with comparable LLM resources via careful paraphrasing, putting an upper bound on the practical security.

Provenance via metadata (C2PA standard for images, similar evolving standards for text) is more reliable when you control the pipeline but doesn't survive copy-paste. In 2026, watermarking is shipped by Google (SynthID) and a few others; OpenAI and Anthropic have not deployed it broadly. The technical case is strong; the deployment case is mixed because watermarks degrade output and customers complain.

## Weak-to-strong generalization

A different angle on alignment for the next generation of models: as models become superhuman in some domains, we may not be able to label preferences accurately. How do we train them?

Burns et al. (2023, *Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision*) studied this directly. Use a weaker model (GPT-2) to generate preference labels, then train a stronger model (GPT-4) on those labels. The strong model recovers a substantial fraction of the capability gap to ground truth — it generalizes the weak supervisor's intent rather than imitating its mistakes.

The result is preliminary and the methodology has limits, but it points at the practical question facing alignment over the 2025-2027 window: as model capability outpaces human ability to evaluate the outputs (in math, code, science), what does "alignment" even mean operationally? The current best answers are scalable oversight protocols (debate, recursive reward modeling, AI safety via market) but none has been shipped at scale.

For practitioners, the takeaway is to be skeptical of "we ran RLHF on the safest thing we could measure" claims. If your reward model is a human judging things humans aren't qualified to judge, the optimization signal is noise.

## What shipping safely actually requires

![fig5: red-teaming workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig5_red_teaming_workflow.png)

A practical checklist for shipping an LLM-powered product safely:

1. **Threat model**: write down what bad outcomes you're defending against. Different products, different threats.
2. **Refusal precision/recall test set**: 500-1000 hand-labeled prompts, retest every model change.
3. **Red-team session**: quarterly, internal team, ~100 prompts in each harm category.
4. **Output filter**: a moderation model (or set of rules) checks outputs before they reach users. Catches what the LLM let through.
5. **Logging**: every interaction logged with model, prompt, output, latency, cost. PII-redacted but otherwise complete.
6. **Anomaly alerts**: flag unusual patterns (sudden refusal rate spike, novel jailbreak attempts, unexpected high-volume from one user).
7. **Human review path**: a way for users to escalate "this answer is wrong/harmful" with a fast path to engineering.
8. **Rate limiting**: prevent automated probing at scale.
9. **Disclosure**: tell users they're talking to AI and the AI can be wrong.
10. **Rollback ready**: previous model version one config flag away.
11. **Action-space constraints**: if your product has tools, the worst-case action of any single response should be bounded (no irreversible operations without confirmation).
12. **Vendor due diligence**: if you depend on a foundation-model API, know your vendor's safety practices, incident history, and notification SLAs.

This isn't a complete safety strategy. It's what a small team can implement and what most products in 2026 still don't have.

## What's Next

Alignment is three goals (helpful, harmless, honest) plus a fourth (controllable) that trade off against each other; pick where you want to sit. RLHF teaches sycophancy, verbosity, and confidence inflation that you didn't ask for; correct via SFT on counter-examples and reward-model auditing. Refusal is mediated by a single feature direction in the residual stream — it is shallow and brittle; layer defenses on top. Refusal calibration needs a labeled test set with both refusal-correct and refusal-wrong cases. Red-team continuously, including with automated PAIR/TAP/GCG-style attacks. Hallucination breaks into factual, faithfulness, and logical — each has different metrics; SelfCheckGPT and FActScore are practical, RAGAS is the right RAG-specific tool. Sleeper Agents shows post-hoc safety has limits — your foundation-model provider's pipeline hygiene matters more than your fine-tuning pass. CAI is a powerful pattern even if you don't run the full Anthropic recipe. Ship with a small but real set of safety mechanisms in place, not just intentions.

Next chapter (final): **production**. Serving stack choice in detail, autoscaling, latency budgets, cost tracking, multi-model routing, and the observability you need from day one.

## References

- Askell, A. et al. (2021). *A General Language Assistant as a Laboratory for Alignment* (HHH framework). https://arxiv.org/abs/2112.00861
- Christiano, P. et al. (2017). *Deep Reinforcement Learning from Human Preferences*. https://arxiv.org/abs/1706.03741
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). https://arxiv.org/abs/2203.02155
- Sharma, M. et al. (2023). *Towards Understanding Sycophancy in Language Models*. Anthropic. https://arxiv.org/abs/2310.13548
- Pan, A. et al. (2022). *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*. ICLR 2022. https://arxiv.org/abs/2201.03544
- Casper, S. et al. (2023). *Open Problems and Fundamental Limitations of RLHF*. https://arxiv.org/abs/2307.15217
- Ganguli, D. et al. (2022). *Red Teaming Language Models to Reduce Harms*. Anthropic. https://arxiv.org/abs/2209.07858
- Chao, P. et al. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR). https://arxiv.org/abs/2310.08419
- Mehrotra, A. et al. (2024). *Tree of Attacks: Jailbreaking Black-Box LLMs Automatically*. https://arxiv.org/abs/2312.02119
- Chao, P. et al. (2024). *JailbreakBench*. https://arxiv.org/abs/2404.01318
- Hubinger, E. et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. Anthropic. https://arxiv.org/abs/2401.05566
- Arditi, A. et al. (2024). *Refusal in Language Models is Mediated by a Single Direction*. https://arxiv.org/abs/2406.11717
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic. https://arxiv.org/abs/2212.08073
- Lee, H. et al. (2023). *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*. https://arxiv.org/abs/2309.00267
- Lin, S. et al. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. https://arxiv.org/abs/2109.07958
- Manakul, P. et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*. EMNLP 2023. https://arxiv.org/abs/2303.08896
- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. https://arxiv.org/abs/2305.14251
- Thorne, J. et al. (2018). *FEVER: A Large-scale Dataset for Fact Extraction and VERification*. https://arxiv.org/abs/1803.05355
- Kirchenbauer, J. et al. (2023). *A Watermark for Large Language Models*. ICML 2023. https://arxiv.org/abs/2301.10226
- Sadasivan, V. et al. (2023). *Can AI-Generated Text be Reliably Detected?* https://arxiv.org/abs/2303.11156
- Burns, C. et al. (2023). *Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision*. OpenAI. https://arxiv.org/abs/2312.09390
- Anthropic (2024). *Frontier Red Team Report*. https://www.anthropic.com/research
