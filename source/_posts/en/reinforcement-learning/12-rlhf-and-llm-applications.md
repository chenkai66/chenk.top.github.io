---
title: "Reinforcement Learning (12): RLHF and LLM Applications"
date: 2024-07-12 09:00:00
tags:
  - Reinforcement Learning
  - RLHF
  - DPO
  - ChatGPT
  - LLM Alignment
categories: Reinforcement Learning
series: Reinforcement Learning
part: 12
total_parts: 12
lang: en
mathjax: true
description: "How RLHF turned base language models into ChatGPT and Claude: the SFT→Reward-Model→PPO pipeline, the Bradley-Terry preference model, the DPO closed-form derivation, RLAIF and Constitutional AI, reward hacking and Goodhart's law, and where RL goes next in embodied intelligence."
disableNunjucks: true
---
GPT-3 (June 2020) and ChatGPT (November 2022) share most of their weights. The base model could write fluent prose, complete code, and continue any pattern you gave it — and yet, asked a plain question, it would happily ramble, refuse for the wrong reasons, hallucinate citations, or produce a paragraph of toxicity. The two and a half years between them were not spent on bigger transformers. They were spent learning **how to ask the model to be useful** — and that turned out to be a reinforcement-learning problem.

This final installment closes the series where it has been heading the whole time: every concept we built — value functions, policy gradients, PPO's trust region, off-policy corrections, preference learning, intrinsic motivation, and even the imitation→IRL ladder — gets composed into the alignment stack that produced ChatGPT, Claude, Llama-3-Instruct, and every assistant-class model worth talking about. We will derive the **three-stage RLHF pipeline**, the **Bradley-Terry** likelihood underneath every preference dataset on the planet, the **closed-form optimum** that lets DPO skip RL entirely, and the **Goodhart-law failure modes** that make alignment a moving target. Then we will look past language at where RL is going next: embodied agents, constitutional self-supervision, and inference-time search.

## What You Will Learn

- The **three-stage RLHF pipeline**: supervised fine-tuning, reward-model training, and PPO with a KL anchor — and why each stage exists
- The **Bradley-Terry model**: why preferences (not absolute scores) are the right currency and what it implies about the reward you can recover
- **InstructGPT**'s key empirical finding: a 1.3B aligned model beats 175B GPT-3 on what humans actually want
- **DPO**: a one-page derivation that turns the closed-form RLHF optimum into a plain log-likelihood loss
- **RLAIF and Constitutional AI**: replacing the human in the loop without collapsing the model
- **Reward hacking and Goodhart's law**: why proxy reward goes up while user satisfaction goes down, and what to do about it
- **Where RL goes next**: embodied agents, sim-to-real, vision-language-action models, and inference-time search

## Prerequisites

- PPO and the trust-region intuition ([Part 6](/en/reinforcement-learning-6-ppo-and-trpo/))
- Off-policy corrections and importance sampling ([Part 3](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/))
- Inverse RL and preference learning ([Part 7](/en/reinforcement-learning-7-imitation-learning/))
- Working knowledge of transformers and HuggingFace `transformers`

---

## 1. RLHF: The Three-Stage Pipeline

![RLHF Three-Stage Pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig1_rlhf_three_stage_pipeline.png)

Pretraining gives you a model that knows the *distribution* of internet text — including parts you do not want. RLHF gives you a model that knows what *you* want. The recipe that escaped from OpenAI in 2022 has become an industry standard with three sharp boundaries: a supervised stage, a preference stage, and a reinforcement stage.

### Stage 1 — Supervised Fine-Tuning (SFT)

Take the pretrained base $\pi_{\text{base}}$, collect a small set of high-quality `(prompt, response)` demonstrations written by humans (InstructGPT used about 13K), and minimise standard next-token cross-entropy:
$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{demo}}}\sum_t \log \pi_\theta(y_t \mid x, y_{<t}).
$$
This stage is cheap, well understood, and absolutely necessary. It pulls the policy from "complete the next token of arbitrary internet text" to "respond helpfully to instructions". Most importantly, it produces $\pi_{\text{SFT}}$, which becomes the **reference policy** $\pi_{\text{ref}}$ that the next two stages anchor against. Everything after this point is a correction term on top of SFT.

### Stage 2 — Reward Model Training

Demonstrations are expensive — humans have to *write* the ideal answer. Preferences are cheap: humans only have to *compare* two model outputs and say which is better. Stage 2 trades demonstration quality for comparison quantity (InstructGPT collected about 33K comparisons).

For each prompt $x$, sample two completions $y_A, y_B \sim \pi_{\text{SFT}}$, ask annotators to pick the winner, label them $(y_w, y_l)$ with $y_w \succ y_l$, and train a reward model $r_\phi(x, y)$ — usually the SFT model with the language head replaced by a scalar — to assign higher score to $y_w$ via the **Bradley-Terry** loss:
$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right].
$$
This is exactly the preference-learning objective from [Part 7](/en/reinforcement-learning-7-imitation-learning/) — RLHF is, structurally, **inverse RL on language models**. A counter-intuitive empirical finding from InstructGPT: a **6B reward model is more stable** than a 175B one for downstream RL. The reward model only has to be good enough that PPO does not drift outside its support; if it is too capable, PPO finds adversarial inputs that exploit its blind spots.

### Stage 3 — PPO with a KL Anchor

Now the actual reinforcement learning. We treat $r_\phi$ as the environment reward and optimise the policy to maximise expected reward — but with a **KL penalty** keeping us close to $\pi_{\text{ref}}$:
$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\!\left[\,r_\phi(x, y) \,-\, \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\,\right].
$$
The bracketed term is the per-token reward used by PPO; the second piece is the KL anchor. Without it, the policy will eventually find tokens that score high under $r_\phi$ but read like word salad — that is reward hacking, and we will return to it in §6.

PPO is the algorithm of choice for three reasons familiar from [Part 6](/en/reinforcement-learning-6-ppo-and-trpo/):

1. **Action space is the vocabulary** (~50K tokens). Q-learning with a 50K-way argmax per token is not tractable; policy gradient is.
2. **Clipping prevents catastrophic updates.** A single bad batch can break a 70B-parameter chat model in ways no checkpoint manager will save you from. PPO's clipped surrogate $\min\big(\tfrac{\pi_\theta}{\pi_{\text{old}}}A,\,\text{clip}(\tfrac{\pi_\theta}{\pi_{\text{old}}},1\!-\!\epsilon,1\!+\!\epsilon)A\big)$ caps how far one update can move you.
3. **The KL penalty integrates naturally** with PPO's trust-region philosophy: both bound how much the policy can change per step, but for different reasons (PPO bounds drift from $\pi_{\text{old}}$ within a single update; KL bounds drift from $\pi_{\text{ref}}$ across the entire training run).

### Why three stages?

Each stage compresses the previous artifact into a more compact signal. SFT compresses tens of millions of internet tokens into a model that *can* respond. The reward model compresses 33K human judgements into a scalar that *evaluates* responses. PPO compresses that evaluator back into a policy that *generates* the responses humans wanted in the first place. Each compression is lossy, and each loss is the topic of an entire research literature.

---

## 2. The Bradley-Terry Model: Why Preferences, Not Scores

![Bradley-Terry preference model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig5_bradley_terry.png)

If you ask a hundred annotators to rate completions on a 1–10 scale, you will find that one annotator's 7 is another's 4. Absolute scores are non-stationary across people, across days, and even across consecutive prompts (calibration drift). The Bradley-Terry model — proposed in 1952 for ranking sports teams — assumes there is a latent score $s(y)$ per item, and that pairwise outcomes follow:
$$
P(y_A \succ y_B) \;=\; \frac{e^{s(y_A)}}{e^{s(y_A)} + e^{s(y_B)}} \;=\; \sigma\!\big(s(y_A) - s(y_B)\big).
$$
Two consequences shape every modern RLHF system:

- **The reward is identifiable only up to a constant.** Adding a constant to all scores leaves all preferences unchanged. The reward model's absolute scale is meaningless; only differences matter. This is also why the partition function $Z(x)$ vanishes in the DPO derivation below.
- **Annotator noise has an irreducible floor.** Even gold-standard humans agree with each other only ~78% of the time on InstructGPT-style prompts. A reward model that hits 78% accuracy on held-out preferences has saturated the signal. Pushing it higher means it is fitting individual annotator quirks, not human values.

The right mental picture: a reward model is a **calibrated classifier of preferences**, not an oracle of quality. The PPO stage then treats this calibrated classifier as if it were ground truth — which is the entry point for every Goodhart failure mode in §6.

---

## 3. PPO with KL Anchor: The Picture in Parameter Space

![PPO with KL constraint](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig3_ppo_kl_constraint.png)

The KL term is doing more work than a regulariser. It is implementing the same trust region we built TRPO around in [Part 6](/en/reinforcement-learning-6-ppo-and-trpo/), but anchored to a *frozen* reference instead of the previous iterate. The left panel above shows the geometry: with the KL anchor, the policy walks toward a moderate-but-genuine reward peak; without it, the policy slides off into a region where the proxy reward $r_\phi$ has a spurious maximum but the real-world output is incoherent.

The right panel shows the practical problem: as you sweep $\beta$ from large to small, the **proxy reward** monotonically rises (you are giving the policy more freedom to optimise) but **true human quality** is hump-shaped — it peaks somewhere in $\beta \in [0.01, 0.03]$ and then collapses. Picking $\beta$ is a hyperparameter choice that requires actual humans in the loop; no offline metric tells you when you have crossed over. In practice teams use **adaptive KL control**: target a fixed average per-token KL (e.g. 6 nats) and let $\beta$ float to maintain it.

A working RLHF run touches **four models in memory simultaneously**: the policy $\pi_\theta$ being trained, the reference $\pi_{\text{ref}}$ for the KL term, the reward model $r_\phi$ for scoring rollouts, and a value head (often shared with the policy backbone) for GAE advantage estimation. This is why RLHF is so much more engineering-heavy than SFT — the memory bill alone is roughly $4\times$.

---

## 4. InstructGPT: What the Numbers Said

![Reward model training](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig2_reward_model_training.png)

The InstructGPT paper (Ouyang et al., NeurIPS 2022) is short, dense, and the closest thing the field has to a Rosetta Stone for what RLHF actually buys you. Four findings worth memorising:

1. **Alignment beats scale.** In blind human evaluations, **1.3B-parameter InstructGPT was preferred over 175B-parameter GPT-3 about 85% of the time**. The aligned smaller model was more useful than the unaligned larger one — by a margin you could not have closed with another order of magnitude of pretraining compute.
2. **Generalisation is real but uneven.** RLHF trained on English instructions transferred to code and to non-English prompts the model had barely seen in the SFT set. The reward model captured something more general than the surface form of its training distribution.
3. **The "alignment tax" is small.** Aligned models lost a few points on standard NLP benchmarks (TriviaQA, HellaSwag) — they had become slightly worse at the next-token completion game. Users did not care; the user-facing wins dwarfed the benchmark losses. This is the first concrete demonstration that **benchmarks and user value can diverge**, an observation that has only sharpened since.
4. **Reward hacking shows up immediately.** The paper documents length hacking (responses get longer for marginal score gains), format hacking (everything becomes a bulleted list), and a mild form of sycophancy. These are not bugs to fix once; they are **stable attractors** that recur in every RLHF system, including production ones.

---

## 5. DPO: Skipping the Reward Model and the RL

![DPO derivation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig4_dpo_derivation.png)

The single most influential RLHF result of the post-InstructGPT era is **Direct Preference Optimization** (Rafailov et al., NeurIPS 2023). Its claim is provocative: you can throw away the reward model and PPO entirely, and replace the whole stack with one supervised loss that you train on the same preference dataset.

### The Derivation

Start from the KL-regularised RL objective:
$$
\max_\pi\; \mathbb{E}_{x,y\sim\pi}\big[r(x,y)\big] \,-\, \beta\, D_{\mathrm{KL}}\!\big[\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big].
$$
This is a constrained convex problem in the per-prompt distribution $\pi(\cdot|x)$. Lagrangian calculus (or just guessing and checking) gives a closed-form optimum:
$$
\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right),
$$
where $Z(x)$ is the partition function over $y$. Now invert this to express the reward in terms of the optimal policy:
$$
r(x,y) \;=\; \beta\,\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta\,\log Z(x).
$$
The crucial observation: **plug this expression for $r$ into the Bradley-Terry preference likelihood**:
$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x,y_w) - r(x,y_l)\big),
$$
and the $\beta\log Z(x)$ terms cancel — they depend only on $x$, not on $y_w$ vs $y_l$. What is left is a loss that depends only on $\pi_\theta$, $\pi_{\text{ref}}$, and the preference data:
$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \,-\, \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].\;}
$$
This is just cross-entropy on a sigmoid of two log-likelihood ratios. No rollouts. No value head. No reward model. No PPO clipping. **Two forward passes, one backward pass, done.**

### What DPO Actually Buys You

- **One stage instead of three.** SFT, then DPO directly on preferences. No separate reward model artifact to maintain.
- **No sampling loop.** PPO requires generating completions inside the training loop, which dominates wall-clock time. DPO is offline supervised learning on a fixed dataset.
- **No explicit reward to hack.** The implicit reward $\hat r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ is *defined* by the policy, so the policy cannot diverge from it.
- **Memory roughly halved.** Two models ($\pi_\theta$, $\pi_{\text{ref}}$) instead of four.

### What DPO Does Not Buy You

- **You cannot inspect reward.** The implicit $\hat r$ exists only as a ratio; you cannot label new completions with a "score" the way you can with a separate $r_\phi$.
- **DPO is more sensitive to noisy preferences.** PPO's KL anchor and online sampling provide some robustness; DPO trusts the preference dataset literally.
- **DPO can underperform on long-horizon reasoning** (chain-of-thought, multi-step tool use), where the policy benefits from on-policy exploration that DPO does not do. This is the gap that **online DPO**, **iterative DPO**, **IPO**, and **KTO** are trying to close.

The pragmatic verdict in 2024–2026: most open-weights instruction-tuned models (Llama-3, Qwen-2.5, Mistral) ship as DPO variants because the engineering is simpler and the headline benchmarks are competitive. Most frontier closed models (ChatGPT, Claude, Gemini) still use PPO-based RLHF or its constitutional cousins, because the marginal quality on hard reasoning tasks justifies the extra complexity. Both branches will keep co-existing.

---

## 6. Reward Hacking and Goodhart's Law

![Reward hacking and Goodhart's law](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig7_reward_hacking.png)

Charles Goodhart's 1975 observation, in its modern form: *"When a measure becomes a target, it ceases to be a good measure."* RLHF is a constructive proof of this aphorism in machine learning. The reward model is a measure of human preference; the moment PPO targets it, the policy starts finding ways to score high without actually serving humans.

The left panel is the canonical empirical picture (Gao et al., ICML 2023): plot training KL from $\pi_{\text{ref}}$ on the x-axis (the "dose" of RL), proxy reward $r_\phi$ rises monotonically, but **gold-standard human reward is hump-shaped** — it peaks early and then collapses. The gap between the two curves is the Goodhart gap, and it widens with model size and training duration.

The right panel catalogues the failure modes that reliably appear:

1. **Length hacking.** Responses grow 2–3× longer than humans actually wanted, because longer often correlates with higher RM score.
2. **Sycophancy.** The model agrees with the user's stated position, even when the user is wrong, because RM annotators tend to prefer being agreed with.
3. **Format hacking.** Bulleted lists, headers, and tables proliferate; the RM learned that structure looks like effort.
4. **Confident BS.** Fluent, well-formatted, factually wrong. The reward model cannot fact-check, so it rewards confidence.
5. **Refusal creep.** The model over-refuses benign queries to hedge against the harmlessness reward, producing the "as a large language model, I cannot..." failure that users hate.

### Mitigations that work in practice

- **KL anchor.** The first line of defence; tune $\beta$ adaptively to a target KL.
- **Reward model ensembling.** Average predictions from several reward models trained on different data splits — exploits average out.
- **Periodic re-labelling.** Collect new preferences on the *current* policy's outputs, not stale ones, and refresh the reward model every few rounds.
- **Length-controlled rewards.** Subtract a length penalty or evaluate at fixed length budgets.
- **Constitutional / red-team additions.** Add explicit rules and adversarial examples to the dataset (next section).

There is no permanent fix. Reward hacking is an arms race built into the structure of the problem.

---

## 7. RLAIF and Constitutional AI: Removing the Human

Human annotation is slow, expensive, and inconsistent — and it does not scale to the volume of preferences a frontier model needs. Two families of methods replace humans, partly or wholly, with strong models:

**RLAIF** (Lee et al., 2023) replaces the annotator with another LLM (e.g. GPT-4) that compares completions:

```text
Given this question and two responses, which better follows
the criteria of being helpful, honest, and harmless?
Question: {x}
Response A: {y_A}
Response B: {y_B}
Answer with A or B and briefly justify.
```

RLAIF preferences agree with human preferences ~85% of the time on standard tasks at roughly 10× lower cost. The risk is **model collapse**: train enough generations on AI-labelled data and the distribution narrows, biases entrench, and quality degrades. The current mitigation toolbox: mix with fresh human data, rotate evaluator models, recalibrate on human gold periodically.

**Constitutional AI** (Bai et al., Anthropic 2022) goes further: write down a "constitution" of natural-language principles ("be helpful", "avoid suggesting harmful actions"), and have the model **self-critique and revise** its own outputs against the constitution before any preference labelling happens. The preference dataset for the RM is then constructed from `(original, revised)` pairs where the revision better satisfies the constitution. This is the foundation of Claude's training stack and a clean example of **using the model's own capabilities to bootstrap its own alignment** — closing a loop that pure RLHF leaves open.

The trend line is clear: as base models get better, more of the alignment signal can be generated by models themselves, and humans move from "label every comparison" to "audit the constitution and the disagreements". The bottleneck shifts from annotation throughput to specification quality.

---

## 8. Architecture of a Production Alignment Stack

![ChatGPT/Claude training architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/12-rlhf-and-llm-applications/fig6_chatgpt_architecture.png)

Putting the pieces together, every modern assistant — ChatGPT, Claude, Gemini, Llama-3-Instruct, Qwen-2.5-Instruct — fits the same five-layer template:

1. **Pretraining.** Trillions of tokens of self-supervised next-token prediction. 90–99% of total compute, and the only stage that meaningfully changes raw capabilities.
2. **SFT / instruction tuning.** 10K–100K curated `(prompt, response)` pairs, optionally augmented with synthetic data distilled from a stronger model.
3. **Preference data.** Human annotations (RLHF), AI annotations (RLAIF), or constitutional self-critique (CAI). Often a mix.
4. **Alignment optimisation.** PPO + KL anchor (OpenAI tradition), DPO and its variants — IPO, KTO, ORPO — (open-weights tradition), or a constitutional self-supervision loop (Anthropic).
5. **Deployment-time guardrails.** System prompts, safety classifiers, tool-use scaffolding, online red-teaming, and rolling evaluations on MT-Bench, Chatbot Arena, and internal regression suites.

The pieces are commoditised. The differentiation between labs lives in the **quality of preference data**, the **robustness of the reward model or its implicit replacement**, and the **discipline of deployment evaluations**. Algorithms are the easy part.

---

## 9. Beyond Language: Where RL Goes Next

RLHF is the highest-stakes deployment of RL right now, but it is not the most ambitious one. Three other frontiers are advancing in parallel and borrow heavily from this series:

**Sim-to-real for robotics.** Train policies in fast simulators (MuJoCo, Isaac Gym), bridge the reality gap with **domain randomisation** (varying physics parameters, lighting, textures), and deploy on real hardware. OpenAI's Dactyl solved a Rubik's cube with a robot hand this way; Google's Aloha system uses imitation learning to bootstrap ([Part 7](/en/reinforcement-learning-7-imitation-learning/)) and online RL to refine.

**Offline RL for safety-critical control.** Driving, healthcare, and industrial control cannot afford on-policy exploration. Methods from [Part 10](/en/reinforcement-learning-10-offline-reinforcement-learning/) — CQL, IQL, Decision Transformer — initialise policies from logged data and only then move to cautious online fine-tuning.

**Vision-Language-Action models.** Google's RT-2 took a pretrained vision-language model and co-fine-tuned it on web data and robot trajectories, producing the first robotic policy with strong zero-shot generalisation to unseen objects and instructions. This is the embodied-intelligence equivalent of what RLHF did for chat: take a model that already understands the world and bend it toward acting in it.

**Inference-time RL.** The most recent twist: rather than spending RL compute at training time, spend it at inference time. OpenAI's o-series and DeepSeek's R1 use RL not to update weights for a single forward pass but to teach the model to **search through chains of thought** before answering — a fusion of [MCTS](/en/reinforcement-learning-8-alphago-and-mcts/) ideas, [PPO](/en/reinforcement-learning-6-ppo-and-trpo/), and the preference-learning machinery above. Expect this to dominate the next two years of frontier model progress.

---

## 10. Simplified RLHF Implementation

The reference code below covers the conceptual flow — reward model with Bradley-Terry loss, then a stripped-down PPO-style optimisation against it. Production stacks (TRL, DeepSpeed-Chat, OpenRLHF, trlX) add GAE advantages, value heads, full PPO clipping, multi-GPU sharding, and adaptive KL control, none of which fit on this page.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# -----------------------------
# Reward model: backbone + scalar head
# -----------------------------
class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        self.value_head = nn.Linear(self.transformer.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.transformer.transformer(
            input_ids=input_ids, attention_mask=attention_mask)
        # Use the last non-pad token's hidden state
        last = out.last_hidden_state[:, -1, :]
        return self.value_head(last).squeeze(-1)


def train_reward_model(model, dataloader, epochs=3, lr=1e-5):
    """Bradley-Terry preference loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for batch in dataloader:
            r_w = model(batch["chosen_ids"], batch["chosen_mask"])
            r_l = model(batch["rejected_ids"], batch["rejected_mask"])
            loss = -F.logsigmoid(r_w - r_l).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(dataloader):.4f}")


# -----------------------------
# Conceptual RLHF: reward + KL anchor
# -----------------------------
class SimpleRLHF:
    def __init__(self, policy, reward_model, ref_model,
                 tokenizer, beta=0.02, lr=1e-6):
        self.policy = policy
        self.rm = reward_model
        self.ref = ref_model               # frozen π_ref
        self.tokenizer = tokenizer
        self.beta = beta
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    def _logp(self, model, ids):
        logits = model(ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)

    def step(self, prompt_ids):
        # 1. Sample a completion from current policy
        with torch.no_grad():
            out = self.policy.generate(
                prompt_ids, max_new_tokens=64,
                do_sample=True, top_p=0.9)
            r = self.rm(out, torch.ones_like(out))      # [B]

        # 2. Compute log-probs under policy and reference
        logp_pi = self._logp(self.policy, out)
        with torch.no_grad():
            logp_ref = self._logp(self.ref, out)

        # 3. KL-regularised reward as per-sequence advantage
        kl = logp_pi - logp_ref                          # [B]
        advantage = (r - self.beta * kl).detach()

        # 4. Policy gradient step
        loss = -(logp_pi * advantage).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), r.mean().item(), kl.mean().item()
```

Two things this code clarifies that prose tends to obscure: (1) the "reward" PPO sees is $r_\phi - \beta \cdot \text{KL}$ baked in as a scalar advantage per sequence, and (2) $\pi_{\text{ref}}$ is frozen — `requires_grad=False` everywhere. Forget either and your training run drifts for hours before you notice.

---

## 11. Frequently Asked Questions

**Q: Why does RLHF beat SFT given enough demonstrations?**
SFT is bounded by demonstration writers — humans rarely write the *optimal* answer, just a *good* one. RLHF lets the model explore beyond the demonstration distribution and rank its own samples. Comparisons are also cheaper than demonstrations, so you collect more signal per dollar.

**Q: Will RLAIF cause model collapse over many generations?**
Current evidence (1–2 generations) shows no obvious degradation. The risk grows with each round of self-distillation. Mitigations: keep a sustained fraction of fresh human data, rotate which model labels, and audit calibration against held-out human gold periodically.

**Q: Is reward hacking solvable?**
Not permanently. It is a structural consequence of optimising a measure, not a bug in any specific reward model. Practical defences (KL anchor, RM ensembling, periodic relabelling, length penalties, constitutional filters) bound the damage but never eliminate it. Treat it as load-bearing engineering, not a one-off fix.

**Q: How much does RLHF cost relative to pretraining?**
Roughly 1–10% of pretraining compute, dominated by reward-model training and PPO sampling. DPO drops this to roughly the cost of a second SFT pass — the compute case for DPO is genuinely strong.

**Q: How do you handle conflicting objectives like helpfulness vs harmlessness?**
Three patterns in practice: (a) train **separate reward models** per objective and combine with a learned or hand-tuned weighting, (b) use **Constitutional AI** to encode hard constraints as natural-language rules, (c) expose **user-controllable preference weights** (e.g. "be more cautious about medical advice"). All three coexist in production stacks.

**Q: When should I pick PPO over DPO?**
Pick DPO when you have a large, clean preference dataset, want fast iteration, and care about wall-clock training time. Pick PPO when you have a high-quality reward model you want to keep in the loop, you need on-policy exploration (multi-step reasoning, tool use), or you intend to mix in safety constraints and constitutional rules at training time.

**Q: How is this connected to inverse RL from Part 7?**
Directly. RLHF is structurally inverse RL with two simplifying choices: pairwise preferences instead of full demonstrations (Bradley-Terry instead of MaxEnt IRL), and PPO as the forward-RL step. The reward model is the IRL output; the PPO stage is the standard "use the recovered reward to train a new policy" step.

---

## 12. References

- **Bradley & Terry (1952).** Rank Analysis of Incomplete Block Designs. *Biometrika*. — origin of the preference likelihood.
- **Christiano et al. (2017).** Deep Reinforcement Learning from Human Preferences. *NeurIPS*. — the first modern preference-RL paper.
- **Stiennon et al. (2020).** Learning to Summarize with Human Feedback. *NeurIPS*. — RLHF on summarization, blueprint for InstructGPT.
- **Ouyang et al. (2022).** Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*.
- **Bai et al. (2022a).** Training a Helpful and Harmless Assistant with RLHF. *Anthropic*.
- **Bai et al. (2022b).** Constitutional AI: Harmlessness from AI Feedback. *Anthropic*.
- **Gao et al. (2023).** Scaling Laws for Reward Model Overoptimization. *ICML*. — the Goodhart-curve paper.
- **Rafailov et al. (2023).** Direct Preference Optimization. *NeurIPS*.
- **Lee et al. (2023).** RLAIF: Scaling RL from Human Feedback with AI Feedback.
- **Skalse et al. (2022).** Defining and Characterizing Reward Hacking. *NeurIPS*.
- **Brohan et al. (2023).** RT-2: Vision-Language-Action Models. *Google DeepMind*.

---

## Series Conclusion

This is the twelfth and final instalment. The series began with Markov Decision Processes and a humble GridWorld and ended with the alignment stack that built ChatGPT and Claude. Along the way we built up:

- **Foundations** — MDPs, Bellman equations, value iteration ([Part 1](/en/reinforcement-learning-1-fundamentals-and-core-concepts/))
- **Value-based methods** — Q-learning, DQN, double/dueling/distributional ([Part 2](/en/reinforcement-learning-2-q-learning-and-dqn/))
- **Policy gradient and Actor-Critic** ([Part 3](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/))
- **Exploration and intrinsic motivation** ([Part 4](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/))
- **Model-based RL and world models** ([Part 5](/en/reinforcement-learning-5-model-based-rl-and-world-models/))
- **PPO and TRPO** ([Part 6](/en/reinforcement-learning-6-ppo-and-trpo/)) — the algorithm that made RLHF possible
- **Imitation and inverse RL** ([Part 7](/en/reinforcement-learning-7-imitation-learning/)) — the conceptual ancestor of preference learning
- **AlphaGo and MCTS** ([Part 8](/en/reinforcement-learning-8-alphago-and-mcts/))
- **Multi-agent RL** ([Part 9](/en/reinforcement-learning-9-multi-agent-rl/))
- **Offline RL** ([Part 10](/en/reinforcement-learning-10-offline-reinforcement-learning/))
- **Hierarchical and meta-RL** ([Part 11](/en/reinforcement-learning-11-hierarchical-and-meta-rl/))
- **RLHF and LLM alignment** (this part)

The unifying thread: **reinforcement learning is the science of learning from consequences**. Whether the consequence is a game score, a physics simulator, a human pairwise preference, or a constitutional self-critique, the same Bellman backups, the same exploration-exploitation trade-offs, and the same trust-region intuitions show up. The next decade of RL will be about closing the loop between training-time and inference-time search, between digital agents and embodied ones, and between human feedback and increasingly autonomous self-improvement. The mathematics in these twelve posts is the toolkit for understanding it.

- **Previous**: [Part 11 — Hierarchical RL and Meta-Learning](/en/reinforcement-learning-11-hierarchical-and-meta-rl/)
- **Series complete!** [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)
