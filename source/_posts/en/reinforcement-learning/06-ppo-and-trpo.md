---
title: "Reinforcement Learning (6): PPO and TRPO -- Trust Region Policy Optimization"
date: 2025-07-05 09:00:00
tags:
  - Reinforcement Learning
  - PPO
  - TRPO
  - Policy Optimization
  - Trust Region
  - RLHF
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 6
  total: 12
lang: en
mathjax: true
description: "Why PPO became the most widely used RL algorithm -- from TRPO's theoretical foundations through natural gradients to PPO's elegant clipping mechanism, plus its role in RLHF for large language models."
disableNunjucks: true
series_order: 6
---
Policy gradients (Part 3) optimise the policy directly, sidestepping discrete `argmax` operators and naturally handling stochastic strategies. They have one fatal flaw: **a single overlong step can destroy the policy**, and because the data distribution is *coupled* to the policy, recovery is nearly impossible.

**Trust-region methods** make this concrete: bound the change in *behaviour*, not in parameters, at every update. TRPO does it through a hard KL constraint and a second-order solver. PPO mimics the same effect with one line of clipped arithmetic. The cheaper trick won: PPO trains OpenAI Five, ChatGPT's RLHF stage, almost every modern robotics policy, and remains the workhorse of applied deep RL.

## What you will learn

- Why vanilla policy gradient is **catastrophically** unstable, with a worked example
- **Importance sampling** as the bridge that lets us reuse off-policy data
- **TRPO**: monotonic-improvement bound, natural gradient, conjugate gradient, line search
- **PPO-Clip** and **PPO-Penalty**: two ways to approximate a trust region with first-order optimisation
- **PPO inside RLHF**: how it aligns ChatGPT-class models, and where DPO/IPO/KTO fit
- A practical hyperparameter and debugging guide that mirrors what you'd find in `cleanrl` or Stable Baselines 3

**Prerequisites:** [Part 3](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) (REINFORCE, Actor-Critic, advantage). Familiarity with KL divergence and Fisher information helps but is not required -- both are introduced in context.

---

## Why policy updates are unstable

### A pathological example

Say the current policy at state $s$ assigns $\pi(a_1|s) = 0.9$ and $\pi(a_2|s) = 0.1$. By bad luck we sample $a_2$ and the environment hands back a reward of $+100$. The REINFORCE estimator multiplies the score $\nabla_\theta \log \pi(a_2|s) = -1/\pi(a_2|s)\cdot\nabla_\theta\pi(a_2|s)$ by $+100$, then takes a gradient step. After one update the probability of $a_2$ may jump from $0.1$ to $0.7$ even though, on expectation, $a_2$ might be far worse than $a_1$.

Three pathologies are at play here:

1. **Variance explosion.** The score function carries a $1/\pi$ factor, so rare actions induce gigantic gradient magnitudes -- the same reason "off-policy" REINFORCE is unstable.
2. **Distribution shift.** The next batch of trajectories is collected by the *new* policy. If the new policy is bad, every datapoint we collect from it confirms a worse signal -- a feedback loop.
3. **Irreversibility.** A supervised model only loses *fit* on a bad step; an RL agent loses *data* on a bad step, and policy collapse can take an order of magnitude more samples to undo than to cause.

### Parameter space lies; policy space tells the truth

Consider two Gaussian policies $\pi_1 = \mathcal{N}(0, 0.01)$ and $\pi_2 = \mathcal{N}(0, 10)$. The Euclidean distance between their parameters (mean and log-std) is small, yet $\pi_1$ is essentially deterministic at zero while $\pi_2$ is nearly uniform across the action range. The induced behaviours -- and hence the rewards -- are completely different.

The lesson: **a fixed step in parameter space can cause an unbounded change in policy behaviour.** Any safety guarantee must therefore live in *distribution space*. The Kullback-Leibler divergence $D_{KL}(\pi_{\text{old}} \| \pi_\theta)$ is the natural metric.

![Trust region: many small KL-bounded steps stay safe; one large parameter step falls off the cliff](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig1_trust_region.png)

The figure makes the geometry concrete: the green-to-red surface is a hypothetical $J(\theta)$ landscape with a narrow ridge of high reward and a sharp cliff. Vanilla policy gradient (left) takes a giant step along the gradient direction and lands on the cliff. TRPO (right) constrains every step to a KL ball around the current iterate; the trajectory hugs the ridge and converges to a safe optimum.

---

## Importance sampling: the bridge to off-policy data

On-policy methods throw away every batch after a single gradient step, because the data distribution shifts. **Importance sampling** lets us re-use a batch by reweighting:

$$\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\!\left[\tfrac{q(x)}{p(x)}\, f(x)\right]$$

Plugging the old and new policies into the policy-gradient objective gives the **surrogate objective**:

$$L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right]$$

The probability ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ is the central object of every algorithm in this post.

Two facts to keep in mind:

- At $\theta = \theta_{\text{old}}$, $L^{\text{IS}}$ has the **same value and gradient** as the true policy-gradient objective $J(\theta)$ -- so it is locally faithful.
- Far from $\theta_{\text{old}}$, the variance of the IS estimator grows roughly as $\exp(2\,D_{KL})$ (figure below). Tiny KL = trustworthy reuse; large KL = noisy garbage.

![Importance sampling ratio distribution and variance growth versus KL](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig5_importance_sampling.png)

The left panel histograms the ratio $r_t$ as the new and old policies drift apart -- the distribution stays tight inside the green PPO clip zone $[0.8, 1.2]$ for $D_{KL}\!\le\!0.02$, and develops a heavy right tail beyond $D_{KL}\!=\!0.05$. The right panel plots the variance of the IS estimator on a log scale; the orange shaded region is the variance "saved" by clipping. This is the quantitative argument for keeping every update inside a small trust region.

---

## TRPO: trust region policy optimization

### The monotonic-improvement bound

Schulman et al. (2015) showed -- generalising Kakade & Langford (2002) -- that the true return of the new policy can be bounded below by the surrogate plus a KL penalty:

$$J(\pi_{\text{new}}) \;\geq\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) \;-\; C \cdot D_{KL}^{\max}\!\left(\pi_{\text{old}} \,\|\, \pi_{\text{new}}\right)$$

where $C = 4\varepsilon\gamma/(1-\gamma)^2$ depends on the maximum advantage magnitude $\varepsilon$ and the discount factor $\gamma$. The corollary is striking: **as long as we improve the surrogate while keeping $D_{KL}^{\max}$ small, monotonic policy improvement is guaranteed.**

In practice the constant $C$ is too pessimistic to be useful; TRPO replaces the penalty with a hard constraint and tunes it empirically.

### The constrained problem

$$\max_\theta \;\; \mathbb{E}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right] \quad\text{s.t.}\quad \bar{D}_{KL}\!\left(\pi_{\text{old}} \,\|\, \pi_\theta\right) \leq \delta$$

with $\delta \approx 0.01$. The mean KL is used in place of the maximum because it is much cheaper to estimate from samples.

### Natural gradient: the right notion of "small"

Standard SGD chooses the direction that maximises the linearised objective subject to a Euclidean ball $\|\Delta\theta\|^2 \le c$. **Natural gradient** changes the metric: it constrains the *KL ball* in distribution space, which is locally a quadratic form with the **Fisher information matrix** as its Hessian:

$$D_{KL}(\pi_\theta \,\|\, \pi_{\theta+\Delta\theta}) \;\approx\; \tfrac{1}{2}\,\Delta\theta^\top F\,\Delta\theta, \qquad F = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta\,\nabla_\theta \log \pi_\theta^\top\right]$$

Solving the constrained problem yields the natural gradient update $\Delta\theta \propto F^{-1}\nabla J$. It is the steepest ascent direction in **policy space**, not parameter space, which is exactly what we want.

### Implementation: conjugate gradient + line search

For a network with millions of parameters, $F$ has $\sim 10^{12}$ entries -- forming it explicitly is impossible. TRPO sidesteps this with two tricks:

1. **Conjugate gradient** to solve $Fx = g$ using only Fisher-vector products $Fv$ (cheap to compute via two `autograd` passes, no matrix is materialised).
2. **Backtracking line search** along the natural-gradient direction, halving the step until the KL constraint is satisfied *and* the surrogate improves -- this preserves the monotonicity guarantee even when the quadratic approximation is loose.

```python
def conjugate_gradient(fisher_vector_product, b, n_steps=10, tol=1e-10):
    """Solve F x = b without materialising F. Costs n_steps * (Fv) ops."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = r.dot(r)
    for _ in range(n_steps):
        Ap = fisher_vector_product(p)
        alpha = rdotr / (p.dot(Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

# After CG: step length is fixed by the KL budget delta
step = torch.sqrt(2 * delta / (x.dot(fisher_vector_product(x)) + 1e-8)) * x
# Backtracking line search ensures both KL <= delta AND surrogate improves.
```

**Strengths.** Theoretically sound monotonic improvement; very stable on hard control tasks (Humanoid, Ant) where vanilla PG diverges.

**Weaknesses.** ~300 lines of careful code, ~10-20 Hessian-vector products per update, almost no benefit from running multiple epochs over a batch (CG already used the curvature information). Hard to scale to distributed training because conjugate gradient requires per-step synchronisation.

---

## PPO: keeping 90% of the benefit at 20% of the complexity

In 2017 Schulman and colleagues asked: *can we get TRPO-like stability using only first-order optimisation?* The answer was PPO, and it has dominated the field ever since.

### PPO-Clip: the central trick

Define the clipped surrogate:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}\!\left(r_t(\theta),\,1\!-\!\varepsilon,\,1\!+\!\varepsilon\right)\hat{A}_t\Big)\right]$$

with $\varepsilon \approx 0.2$. The construction is asymmetric on purpose: the `min` makes the objective **pessimistic**, picking whichever of the clipped and unclipped quantities is smaller.

![PPO clipped surrogate split by sign of advantage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig2_ppo_clipping.png)

The two cases (and a one-line summary of each) are:

- **$\hat{A}>0$ (a good action).** The unclipped surrogate keeps growing as $r_t \to \infty$; the clip caps the reward for $r_t > 1+\varepsilon$. The `min` then *takes the cap*, so the gradient drops to zero past the cap. Translation: *don't push the probability of a sampled good action above $(1+\varepsilon)\pi_{\text{old}}$ on the strength of one minibatch.*
- **$\hat{A}<0$ (a bad action).** Symmetric: the clip caps how negative the loss can get for $r_t < 1-\varepsilon$, so once the new policy has reduced the probability by enough, the gradient stops. *Don't crush a single bad-luck action.*

The deeper question is: *why use `min`, not `max`?* If we always take whichever surrogate is **larger**, we'd reward huge ratios -- the optimiser would happily blow past the trust region. The `min` bakes in the trust-region intuition without ever computing a KL.

### PPO-Penalty: the adaptive cousin

A second variant -- less popular but useful in some domains -- adds an explicit KL penalty and adapts its coefficient:

$$L^{\text{KL}}(\theta) = \mathbb{E}\!\left[r_t(\theta)\,\hat{A}_t\right] \;-\; \beta\,\mathbb{E}\!\left[D_{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)\right]$$

with $\beta$ adjusted after every iteration: doubled when measured KL exceeds $1.5\,\delta_{\text{target}}$, halved when below $\delta_{\text{target}}/1.5$.

![Adaptive KL penalty: objective shape and beta schedule](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig3_kl_penalty.png)

The left panel shows how $\beta$ shapes the objective: $\beta=0$ recovers the unconstrained surrogate (and its instability); large $\beta$ makes the optimum hug $\theta_{\text{old}}$. The right panel shows a real adaptive schedule -- $\beta$ moves over orders of magnitude on a log axis to keep the observed KL near its target. Adaptive KL is what early InstructGPT used internally before clip-style PPO became the default for RLHF in libraries like `trl`.

### Surrogate landscape: why clip wins in practice

![Surrogate vs true objective: unclipped misleads, clipped stays honest](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig6_surrogate_landscape.png)

This figure is the visual punchline. The black curve is the *true* return $J(\theta)$ along a 1D slice -- it has a peak followed by a sharp drop into a low-reward region. The orange dashed line is the unclipped IS surrogate, which keeps climbing and would lure SGD to step into the cliff. The blue line is the PPO-clipped surrogate: inside the trust region (green band) it tracks the true objective; outside, it flattens out, removing the gradient that would otherwise carry us off the cliff.

### A complete PPO implementation

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    """Shared trunk, separate actor and critic heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        x = self.shared(state)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, eps_clip=0.2, k_epochs=10, c_vf=0.5, c_ent=0.01):
        self.gamma, self.lam = gamma, lam
        self.eps_clip, self.k_epochs = eps_clip, k_epochs
        self.c_vf, self.c_ent = c_vf, c_ent
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Generalised Advantage Estimation (Schulman et al., 2016).

        lam=0  -> 1-step TD  (low variance, biased)
        lam=1  -> Monte-Carlo (unbiased, high variance)
        lam=0.95 is the universal default.
        """
        advantages = []
        gae = 0.0
        values = list(values) + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages)

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        with torch.no_grad():
            values = self.policy(states)[1].squeeze().numpy()
            next_val = self.policy(torch.FloatTensor(next_state).unsqueeze(0))[1].item()

        advantages = self.compute_gae(rewards, values, dones, next_val)
        returns = advantages + torch.FloatTensor(values)
        # Per-batch advantage normalisation -- cheap, vital for stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            probs, vals = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            vals = vals.squeeze()

            # Ratios in log space for numerical safety.
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(vals, returns)
            loss = policy_loss + self.c_vf * value_loss - self.c_ent * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
```

A handful of lines beyond a vanilla actor-critic, and you have an algorithm that solves CartPole in 100 episodes and learns a humanoid to walk in $\sim$10 million steps.

---

## TRPO vs PPO: head to head

| Aspect | TRPO | PPO-Clip |
|--------|------|----------|
| Solver | Conjugate gradient + line search (2nd order) | Adam (1st order) |
| KL constraint | Hard, exactly $\le \delta$ | Soft, via per-sample clip |
| Updates per batch | 1 | 3-10 epochs |
| Hessian-vector products | $\sim$10-20 | 0 |
| Code complexity | $\sim$300 LOC | $\sim$100 LOC |
| Theoretical guarantee | Monotonic improvement (under bound) | None formally |
| GPU/distributed friendliness | Poor (CG needs sync) | Excellent |
| Hyperparameter sensitivity | $\delta$ matters | $\varepsilon = 0.2$ works almost always |

PPO's edge in practice comes from a *combination* of advantages, not the clip alone:

- **Multiple epochs per batch** squeeze more learning from the same trajectories.
- **Entropy bonus** keeps exploration alive without bespoke noise schedules.
- **No backward-pass overhead per step** -- one Adam call per epoch.
- **Trivially parallelisable** -- you can run 64 actors in parallel collecting rollouts.

![PPO matches or beats TRPO at a fraction of the cost](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig4_benchmark.png)

The MuJoCo curves (left) and Atari bars (right) show the empirical picture published in the original PPO paper and reproduced by Engstrom et al. (2020): PPO-Clip dominates A2C across the board and edges past TRPO on most tasks while running roughly 10$\times$ faster per wall-clock second.

---

## PPO inside RLHF

PPO's most consequential application is **Reinforcement Learning from Human Feedback** -- the algorithm that turned GPT-3 into ChatGPT.

### The three-stage pipeline

1. **Supervised Fine-Tuning (SFT).** Fine-tune a pretrained LLM on high-quality human demonstrations. Output: $\pi_{\text{ref}}$, the reference model.
2. **Reward modelling.** Collect human preference data of the form *"response A is better than response B"*. Train a reward model $R_\phi(x, y)$ via a Bradley-Terry pairwise loss.
3. **PPO fine-tuning.** Treat the LLM as a policy, the prompt $x$ as state, the response $y$ as action, and $R_\phi(x, y)$ as reward. Run PPO.

### The RLHF objective

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\,y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x, y) - \beta\, D_{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)\right]$$

Two trust regions are at work:

- The **PPO clip** prevents per-update collapse, exactly as in classical RL.
- The **KL-to-reference penalty** $\beta D_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ prevents *long-run drift* from the SFT model. Without it, PPO finds **reward hacks** -- responses that score high under the reward model but are gibberish, hostile, or sycophantic. The KL penalty is the alignment tax that keeps generations recognisably human-written.

### Why RLHF is harder than CartPole

- Each "episode" is one generation; compute per sample is enormous (forward+backward on a 70B model).
- The reward model is *also* learnt, and noisy. A small clip range and a substantial KL coefficient ($\beta \in [0.01, 0.2]$) are usually needed.
- Action space is the entire vocabulary (~50K tokens) over hundreds of timesteps. The advantage estimator must be very low-variance -- this is why PPO with GAE and a large batch is preferred to vanilla policy gradient.

### Where DPO, IPO and KTO fit

The complexity of running PPO at scale -- four model copies (policy, ref, reward, value), multiple GPUs syncing -- spurred a wave of *direct* preference learning methods:

- **DPO** (Rafailov et al., 2023) reparameterises the optimal RLHF policy in closed form and trains with a *supervised* contrastive loss -- no reward model, no rollouts.
- **IPO** patches DPO's tendency to overfit on confidently labelled pairs.
- **KTO** uses single-response signed feedback ("good"/"bad") instead of pairs.

Where direct methods shine: simpler infra, lower variance. Where PPO still wins: when you need an actual *reward signal* (online learning, tool-using agents, code/math environments where a verifier provides reward), PPO remains the only general-purpose solution.

---

## Practical tuning guide

### Hyperparameter cheat sheet

| Parameter | Typical range | Notes |
|-----------|---------------|-------|
| Learning rate | $1\text{e-}4$ to $3\text{e-}4$ | Sometimes annealed linearly to 0 over training |
| Clip $\varepsilon$ | 0.1 -- 0.3 | 0.2 works in 90% of cases |
| GAE $\lambda$ | 0.9 -- 0.99 | 0.95 default; closer to 1 when value function is unreliable |
| Discount $\gamma$ | 0.99 -- 0.999 | Longer-horizon tasks need larger $\gamma$ |
| Rollout batch | 2048 -- 8192 (single env: longer) | Larger = lower-variance advantage |
| PPO epochs | 3 -- 10 | More than 15 reliably overfits |
| Mini-batch size | 64 -- 256 | Independent of rollout batch |
| Entropy coef | 0.0 -- 0.01 | Atari needs more (~0.01); MuJoCo often 0 |
| Value loss coef | 0.5 -- 1.0 | Higher when critic is hard to learn |
| Grad clip norm | 0.5 -- 1.0 | A "safety belt" for outlier batches |

![Hyperparameter sensitivity: clip range, learning rate, and the epochs x batch grid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/fig7_hyperparameter.png)

The sensitivity plot shows three classic patterns. Clip $\varepsilon$ has a **wide, flat optimum** -- you really do not have to tune it. Learning rate is **narrower** -- a factor of 3 misstep costs 5-10% of return. The epochs $\times$ batch heatmap reveals an interaction: too many epochs on a small batch overfits to current data and hurts the policy's KL budget.

### Debugging checklist

- **Approximate KL** between old and new policy: should hover around 0.01-0.02 per update. > 0.05 means clip is failing to constrain you -- lower the LR or shrink mini-batches. (Beware the common "one-sample" KL estimator $\frac{1}{2}(\log r)^2$; the unbiased Schulman estimator $r - 1 - \log r$ is preferred.)
- **Clip fraction**: the share of samples that hit the clip boundary. 10-30% is healthy. 0% means you are not using your trust region; > 50% means you are training on the boundary.
- **Entropy**: should decay smoothly. A sudden drop to near-zero is *premature convergence* -- raise the entropy coefficient.
- **Explained variance** of the value function: $1 - \mathrm{Var}(R - V)/\mathrm{Var}(R)$. Should rise above 0.5 within a few hundred updates; if stuck near 0, your critic is broken.
- **Advantage normalisation**: do it per *minibatch*, not per epoch. Forgetting the per-batch normalisation is the single most common implementation bug.

### Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Reward grows then crashes | KL too aggressive, clip ineffective | Smaller LR; fewer epochs per batch |
| Reward never moves | Advantage normalised across batch=0 reward | Check rewards aren't constant; widen exploration |
| Entropy collapses to 0 in 50 steps | No entropy bonus; greedy from start | Add `c_ent` $\ge 0.005$, raise temperature |
| Critic predictions explode | No reward normalisation; large unbounded rewards | Clip or normalise rewards (running stats) |
| Works on CartPole, fails on continuous control | Discrete-style network for Gaussian policy | Use `tanh` squashing, learnable log-std |

---

## Summary

Trust-region methods rescued policy gradients from their most embarrassing failure mode -- one bad step erases an hour of training. Two ideas carry the field:

- **TRPO** turns the policy-improvement bound of Kakade-Langford into an algorithm: optimise the surrogate, hard-constrain the KL, solve via natural gradient. Theoretically beautiful, operationally heavy.
- **PPO** trades the hard constraint for a clipped surrogate plus first-order optimisation. It loses the formal monotonic-improvement guarantee but gains everything that matters in practice: simple code, multi-epoch updates, parallel rollouts, and robust defaults.

The deeper engineering lesson is universal: **a simple, locally-honest approximation often beats a complex, globally-correct one** -- especially when the "correct" method is two orders of magnitude more expensive per step. PPO's victory mirrors Adam's victory over second-order optimisers in deep learning.

PPO's reach now extends well beyond classical RL. Every major aligned LLM trained in the past three years -- ChatGPT, Claude, Gemini, the Llama-2 chat models -- ran a PPO loop somewhere in its post-training pipeline. The clip is, quite literally, what keeps modern AI assistants from going feral on their reward model.

---

## References

- Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). Trust Region Policy Optimization. *ICML*. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Kakade, S., & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. *ICML*.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. *ICLR*. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Huang, S., Dossa, R., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. (2022). The 37 Implementation Details of PPO. *ICLR Blog Post*. [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Core Concepts](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) |
| 2 | [Q-Learning and DQN](/en/reinforcement-learning-2-q-learning-and-dqn/) |
| 3 | [Policy Gradient and Actor-Critic](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) |
| 4 | [Exploration and Curiosity-Driven Learning](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) |
| 5 | [Model-Based RL and World Models](/en/reinforcement-learning-5-model-based-rl-and-world-models/) |
| **6** | **PPO and TRPO (you are here)** |
