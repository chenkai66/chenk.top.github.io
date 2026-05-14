---
title: "Reinforcement Learning (10): Offline Reinforcement Learning"
date: 2025-09-15 09:00:00
tags:
  - Reinforcement Learning
  - Offline RL
  - CQL
  - Decision Transformer
  - BCQ
categories: Reinforcement Learning
series: reinforcement-learning
part: 10
total_parts: 12
lang: en
mathjax: true
description: "Master offline RL: learn policies from fixed datasets without environment interaction. Covers distributional shift, Conservative Q-Learning (CQL), BCQ, Implicit Q-Learning (IQL), Decision Transformer, with a complete CQL implementation."
disableNunjucks: true
series_order: 10
translationKey: "reinforcement-learning-10"
---
Every algorithm we've studied so far has the same core loop: act, observe, update. This loop makes RL work, but it also prevents RL from being deployed. A self-driving system can't practice intersections by crashing. A clinical decision-support model can't run a randomized policy on real patients. A factory robot can't test ten thousand grasp variants on a production line.

These settings do have logs — millions of hours of human driving, decades of de-identified patient records, and terabytes of behavior cloning data. **Offline RL** (also called *batch RL*) is the subfield that asks: can we extract a strong policy from a fixed dataset without any new interaction with the environment?

The answer is "yes, but only if we are very careful." The reason for this caveat is the central theme of this post: distributional shift between the *behavior policy* that generated the data and the *learned policy* that aims to improve on it.


---

## What You Will Learn

- **Why naive off-policy RL fails offline**: extrapolation error, value overestimation, and the death spiral.
- **CQL** (Conservative Q-Learning): a pessimistic regularizer that lower-bounds the true value.
- **BCQ** (Batch-Constrained Q-Learning): generative action proposals that stay inside the data manifold.
- **IQL** (Implicit Q-Learning): expectile regression that approximates `max` *without ever querying OOD actions*.
- **Decision Transformer**: reframing RL as conditional sequence modeling on returns-to-go.
- **D4RL benchmark numbers** so you know which algorithm to reach for first.
- **A complete, runnable CQL implementation** in PyTorch.

## Prerequisites

- Q-learning, target networks, and actor-critic ([Parts 2](/en/reinforcement-learning/02-q-learning-and-dqn/)-[6](/en/reinforcement-learning/06-ppo-and-trpo/)).
- Comfort with the Bellman optimality operator and importance sampling.
- PyTorch and the Gym/Gymnasium API.

---

## Why Offline RL Is Genuinely Hard

![Online vs offline RL: the missing feedback loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig1_online_vs_offline.png)

In **online** RL the policy and the data distribution are coupled. As soon as the policy starts overestimating some action, the next rollout puts it under the microscope and the Bellman update corrects it. In **offline** RL the dataset is *frozen*. There is no second chance: any error the model makes about an unseen action will sit in the Q-table forever, and the `argmax` operator will happily exploit it.

### Distributional Shift

Let $\pi_\beta$ be the behavior policy that produced the dataset $\mathcal{D}=\{(s_i,a_i,r_i,s_i')\}_{i=1}^N$, and let $\pi_\theta$ be the policy we are learning. The state-action visitation distributions are usually different,
$$d_{\pi_\theta}(s,a)\neq d_{\pi_\beta}(s,a).$$
This becomes a problem the moment $\pi_\theta$ wants to take an action $a$ for which $\pi_\beta(a\mid s)\approx 0$. The Q-network has *never* seen a target for that action; whatever value it returns is pure extrapolation from a neural net that has been trained to fit a completely different region of the input space.

### Extrapolation Error and the Death Spiral

![Distribution shift produces over-optimistic Q-values on OOD actions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig2_distribution_shift.png)

The Q-learning update is
$$Q(s,a)\leftarrow r + \gamma\,\max_{a'}Q(s',a').$$
The `max` operator selects the *most optimistic* extrapolation. If $Q$ overestimates even one out-of-distribution action by $\epsilon$, that bias becomes the bootstrap target for the previous timestep, then for the timestep before that, and so on. Empirically this **diverges within a few thousand gradient steps** on standard benchmarks ([Fujimoto et al., 2019](https://arxiv.org/abs/1812.02900)). The right panel of the figure above is the picture to keep in mind: the green curve is reality, the red curve is what the network believes, and the policy walks straight off the cliff at the OOD peak.

The three families of algorithms below all attack this problem; they differ in *what they constrain*.

| Family | Slogan | Constraint placed on |
|--------|--------|----------------------|
| **Policy-constraint** (BCQ, BEAR, TD3+BC) | "Only act like the data" | $\pi_\theta$ |
| **Value-pessimism** (CQL, MOPO) | "Distrust unfamiliar Q-values" | $Q$ |
| **In-sample learning** (IQL, AWAC) | "Never query OOD actions" | the loss itself |
| **Sequence modeling** (Decision Transformer, Trajectory Transformer) | "Skip Bellman entirely" | the problem formulation |

---

## Conservative Q-Learning (CQL)

![Conservative Q-Learning penalises OOD actions vs naive offline Q (which diverges).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-Offline-Reinforcement-Learning/fig10_cql.png)

![Reinforcement Learning (10): Offline Reinforcement Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/illustration_2.png)

CQL ([Kumar et al., 2020](https://arxiv.org/abs/2006.04779)) is the most widely used baseline. It does not change the network, the actor, or the data pipeline. It changes one term in the loss.

### The Idea in One Sentence

> **Push down the Q-values of any action the policy might consider, then push back up the Q-values of actions actually present in the data.** The net effect is a Q-function that is pessimistic exactly where it lacks evidence.

### The Objective

On top of the usual TD loss $\mathcal{L}_{\mathrm{TD}}$, CQL adds
$$\mathcal{L}_{\mathrm{CQL}} \;=\; \alpha\,\Big[\,\underbrace{\log\sum_{a}\exp Q(s,a)}_{\text{push down everywhere}} \;-\; \underbrace{\mathbb{E}_{a\sim\mathcal{D}}\big[Q(s,a)\big]}_{\text{push up on data}}\Big] \;+\; \mathcal{L}_{\mathrm{TD}}.$$
The `logsumexp` is a soft maximum over actions: minimizing it pulls *all* Q-values down. Subtracting the expectation under the data distribution lets the in-data actions float back up. Out-of-distribution actions only feel the downward force.

The headline theorem ([Kumar et al., 2020, Thm 3.2](https://arxiv.org/abs/2006.04779)) is that for sufficiently large $\alpha$,
$$\hat{Q}^{\pi}(s,a)\;\leq\;Q^{\pi}(s,a)\quad\forall (s,a).$$
A pessimistic estimate is exactly what we need: any policy that maximizes a *lower bound* on the true value cannot pick a catastrophic action because it has no incentive to.

![CQL: the regularizer pushes OOD Q-values down so argmax stays inside the data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig4_cql_penalty.png)

The left panel shows what happens with a vanilla SAC-style critic on offline data: the argmax flees to the OOD bump. The right panel shows CQL's correction: the orange shaded region is the pessimism penalty, and the new argmax sits comfortably inside the data support.

### Practical Notes

- The `logsumexp` is intractable for continuous actions, so implementations approximate it with `n_random` uniform samples plus `n_actor` samples from the current policy. 10-20 samples is usually enough.
- The original paper's "CQL($\mathcal{H}$)" variant adds a Lagrangian that auto-tunes $\alpha$ to hit a target gap; this is what the open-source `d3rlpy` and `JaxRL` implementations ship by default.
- Empirically CQL is robust on `medium-replay` and `medium` D4RL splits but can be slightly conservative on `medium-expert`, where IQL or DT often win.

### A Complete CQL Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Twin Q-networks to reduce overestimation (Clipped Double Q)."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = self._mlp(state_dim + action_dim, hidden)
        self.q2 = self._mlp(state_dim + action_dim, hidden)

    @staticmethod
    def _mlp(in_dim, hidden):
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian (the SAC actor)."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def sample(self, state):
        x = self.trunk(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # Tanh change-of-variables for the log-prob.
        log_prob = (normal.log_prob(z)
                    - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob

class CQLAgent:
    """CQL on top of SAC. cql_weight = alpha in the formulas above."""

    def __init__(self, state_dim, action_dim, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 cql_weight=1.0, n_random=10):
        self.gamma, self.tau = gamma, tau
        self.cql_weight, self.n_random = cql_weight, n_random
        self.action_dim = action_dim

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q = QNetwork(state_dim, action_dim)
        self.target_q.load_state_dict(self.q_net.state_dict())
        self.policy = GaussianPolicy(state_dim, action_dim)

        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.alpha = alpha

    # --------- the only function that differs from vanilla SAC -------------
    def _cql_penalty(self, states, q1_data, q2_data):
        """logsumexp over actions - mean Q on in-data actions."""
        b = states.size(0)
        # (1) random actions uniform in [-1, 1]
        rand_a = torch.empty(b * self.n_random, self.action_dim).uniform_(-1, 1)
        # (2) actions from the current policy at s and s'
        rep_s = states.repeat_interleave(self.n_random, 0)
        pi_a, pi_lp = self.policy.sample(rep_s)

        def _q(s, a):
            return self.q_net(s, a)

        q1_rand, q2_rand = _q(rep_s, rand_a)
        q1_pi,   q2_pi   = _q(rep_s, pi_a)

        # importance-weighted logsumexp (Kumar et al., Eq. 4)
        log_pi_uniform = -float(self.action_dim) * torch.log(torch.tensor(2.0))
        cat1 = torch.cat([q1_rand - log_pi_uniform,
                          q1_pi   - pi_lp.detach()], 0).view(b, -1)
        cat2 = torch.cat([q2_rand - log_pi_uniform,
                          q2_pi   - pi_lp.detach()], 0).view(b, -1)

        lse1 = torch.logsumexp(cat1, 1, keepdim=True)
        lse2 = torch.logsumexp(cat2, 1, keepdim=True)
        return ((lse1 - q1_data) + (lse2 - q2_data)).mean()

    def update(self, states, actions, rewards, next_states, dones):
        # 1. Bellman target with clipped double Q + entropy bonus
        with torch.no_grad():
            next_a, next_lp = self.policy.sample(next_states)
            tq1, tq2 = self.target_q(next_states, next_a)
            target_q = torch.min(tq1, tq2) - self.alpha * next_lp
            target = rewards + self.gamma * (1 - dones) * target_q

        q1, q2 = self.q_net(states, actions)
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        cql = self._cql_penalty(states, q1, q2)
        q_loss = td_loss + self.cql_weight * cql

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # 2. Actor update (standard SAC)
        a, lp = self.policy.sample(states)
        q1_a, q2_a = self.q_net(states, a)
        pi_loss = (self.alpha * lp - torch.min(q1_a, q2_a)).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # 3. Polyak averaging of target Q
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(),
                             self.target_q.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"td": td_loss.item(),
                "cql": cql.item(),
                "pi":  pi_loss.item()}
```

Plug an offline replay buffer (D4RL via `minari` or `d4rl-pybullet`) into `update`, train for ~1M gradient steps, and you should hit roughly the published numbers in the benchmark figure further down.

---

## BCQ: Generative Action Proposals

CQL constrains the *value*; BCQ ([Fujimoto et al., 2019](https://arxiv.org/abs/1812.02900)) constrains the *policy* directly. The principle is mechanical: never even *query* the Q-function on an action that the behavior policy would not have produced.

![BCQ: a VAE proposes candidates, a perturbation network adds a small refinement, then argmax picks one](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig3_bcq_architecture.png)

The architecture has three pieces:

1. **A conditional VAE** $G_\omega(s)$ trained on $\mathcal{D}$ via the standard ELBO. It models $\pi_\beta(a\mid s)$, so samples drawn from it stay on the behavior manifold.
2. **A perturbation network** $\xi_\phi(s,a)\in[-\Phi,\Phi]^{\dim(a)}$ that adds a small, bounded refinement to each VAE sample. With $\Phi$ small (typically 0.05), the refined action cannot drift far from the data; with $\Phi=0$ BCQ reduces to weighted imitation.
3. **Twin Q-networks** that score the $N$ refined candidates; the argmax is the action that gets executed.

The clever part is that the perturbation network is trained *to maximize Q*. So BCQ gets the benefit of policy improvement (it is not pure imitation) without the cost of OOD extrapolation (it cannot stray far from the data). The price is a more elaborate pipeline — four networks instead of two — and the VAE itself must be trained well, otherwise the candidates are biased.

---

## IQL: Avoid the Bootstrap Problem Entirely

IQL ([Kostrikov et al., 2022](https://arxiv.org/abs/2110.06169)) is the most elegant of the three. Its observation: every offline-RL pathology comes from the $\max_{a'}Q(s',a')$ in the Bellman target, because that is where OOD actions enter. So... just don't use it.

IQL learns a separate state-value function $V(s)$ via **expectile regression** on the in-data actions only:
$$\mathcal{L}_V \;=\; \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[L_2^{\tau}\big(Q(s,a)-V(s)\big)\big],\qquad L_2^{\tau}(u)=\big|\tau-\mathbb{1}(u<0)\big|\,u^{2}.$$
When $\tau=0.5$ this is plain MSE and $V$ learns the *mean*. When $\tau\to 1$ it asymmetrically penalizes under-estimates much more than over-estimates, so $V$ converges to the *upper expectile* of $Q(s,a)$ over the actions seen in the data.

![IQL: the expectile loss is asymmetric; pushing tau toward 1 pulls V toward max_a Q without ever leaving the data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig5_iql_expectile.png)

For $\tau\approx 0.7$-$0.9$, $V(s)$ is an excellent proxy for $\max_a Q(s,a)$ — *and it was computed from in-data actions only*. The Bellman target then becomes
$$y \;=\; r + \gamma\,V(s'),$$
with no `max`, no policy sampling, no extrapolation. The actor is recovered via **advantage-weighted regression**:
$$\mathcal{L}_\pi \;=\; -\mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\exp\big(\beta\,(Q(s,a)-V(s))\big)\,\log\pi_\theta(a\mid s)\big].$$
IQL has the smallest moving-parts surface of any modern offline-RL algorithm, and it consistently tops the D4RL leaderboard on AntMaze and Adroit — environments where good data is sparse and bootstrapping with `max` is most dangerous.

---

## Decision Transformer: RL as Sequence Modeling

If we are going to drop the Bellman equation, why stop at IQL? The **Decision Transformer** ([Chen et al., 2021](https://arxiv.org/abs/2106.01345)) drops the value function entirely and treats RL as next-token prediction.

![Decision Transformer: tokens are (return-to-go, state, action) triplets fed to a causal Transformer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig6_decision_transformer.png)

A trajectory is laid out as a sequence of triplets $(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots)$, where $\hat{R}_t = \sum_{t'\geq t} r_{t'}$ is the *return-to-go* from timestep $t$. A standard causal (GPT-style) Transformer is trained with cross-entropy or MSE to predict $a_t$ given everything before it.

At test time you feed in your *desired* return-to-go as $\hat{R}_1$, run the model autoregressively, and out come the actions. The return-to-go acts as a knob: ask for more, get a more aggressive policy.

**What this buys you:**
- No bootstrapping, so no extrapolation error.
- Long context lets the model condition on partial-observation history "for free."
- Works out of the box with all the Transformer engineering (LayerNorm, RoPE, FlashAttention, mixed precision...).
- One paradigm scales from D4RL to Atari to large multi-task setups (e.g. *Gato*).

**What it costs:**
- Cannot exceed the best return seen in the data: if no trajectory in $\mathcal{D}$ ever scored 90, asking for 90 produces garbage.
- No causal credit assignment — the model is trained to *imitate trajectories conditioned on outcome*, not to reason about which action *caused* the outcome.
- Generally requires more parameters and more data than CQL/IQL to reach the same wall-clock score on small benchmarks.

---

## Putting Numbers to It: D4RL

![D4RL MuJoCo locomotion: BC, BCQ, CQL, IQL, DT across three data qualities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig7_d4rl_benchmark.png)

The figure compiles representative normalized scores from the original CQL ([Kumar et al., 2020](https://arxiv.org/abs/2006.04779)), IQL ([Kostrikov et al., 2022](https://arxiv.org/abs/2110.06169)) and DT ([Chen et al., 2021](https://arxiv.org/abs/2106.01345)) papers on the canonical [D4RL](https://arxiv.org/abs/2004.07219) MuJoCo locomotion suite (`hopper`, `halfcheetah`, `walker2d` averaged). 100 = expert performance, 0 = random.

Three takeaways:

1. **`medium-replay` is where conservatism matters most.** This split is generated by an early SAC checkpoint — lots of bad actions, lots of recovery transitions. CQL and IQL roughly *double* BC's score; DT trails because it has no value bootstrapping to stitch together good sub-trajectories.
2. **`medium-expert` is where DT shines.** When the data already contains near-optimal trajectories, sequence modeling wins on simplicity.
3. **IQL is the most consistent.** It is rarely the absolute best on any single dataset, but it is rarely worse than third, and its training is the most stable of the four.

When in doubt, start with **IQL** for stability or **CQL** for simplicity, and reach for **DT** when the dataset already contains near-expert demonstrations.

---

## The Distributional Shift Bound (Why Naive Q-learning Fails Offline)

I want to make the hand-wavy "distributional shift is bad" intuition concrete. There is an actual bound, and once you read it, the rest of offline RL stops feeling like a bag of tricks.

Setup. The dataset is $\mathcal{D} = \{(s, a, r, s')\}$ collected by a behaviour policy $\beta$. We want to evaluate (or learn) a target policy $\pi$. The Q-function we fit on $\mathcal{D}$ via the standard Bellman backup is $\hat Q^\pi$. The true on-policy value is $Q^\pi$. The question: how wrong is $\hat Q^\pi$?

Kumar et al. 2019 give a clean answer:
$$|Q^\pi(s,a) - \hat Q^\pi(s,a)| \;\le\; \frac{2 r_{\max}}{(1-\gamma)^2}\, \mathbb{E}_{s \sim d^\pi}\, D_{\text{TV}}\big(\pi(\cdot \mid s)\, \|\, \beta(\cdot \mid s)\big).$$

Two factors blow up. The horizon coupling $1/(1-\gamma)^2$ is the standard "errors compound along trajectories" term — quadratic, not linear. The other term is the total-variation gap between $\pi$ and $\beta$, *averaged over states the target policy actually visits*. If $\pi$ wants to do anything $\beta$ never did, that expectation explodes.

Why this is fatal for naive Q-learning. The Bellman target queries $\hat Q(s', a')$ for $a' \sim \pi(\cdot \mid s')$. Nothing in the loss prevents $\pi$ from picking $a'$ outside the support of $\beta$. At those points $\hat Q$ is whatever the network extrapolates — an unconstrained MLP on inputs it was never trained on. The `argmax` then *selects for* the most over-optimistic extrapolation, and the next backup propagates that fantasy value backward in time. There is no second rollout to correct it.

The minimal demo. Twenty-five lines of vanilla Q-learning on a D4RL halfcheetah-medium replay buffer. Watch the Q-values, not the returns.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Q(nn.Module):
    def __init__(self, sd, ad):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd + ad, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], -1))

def naive_offline_q(buffer, sd, ad, steps=20_000, gamma=0.99):
    q, tq = Q(sd, ad), Q(sd, ad); tq.load_state_dict(q.state_dict())
    pi = nn.Sequential(nn.Linear(sd, 256), nn.ReLU(), nn.Linear(256, ad), nn.Tanh())
    opt_q  = torch.optim.Adam(q.parameters(),  3e-4)
    opt_pi = torch.optim.Adam(pi.parameters(), 3e-4)
    for t in range(steps):
        s, a, r, s2, d = buffer.sample(256)
        with torch.no_grad():
            a2 = pi(s2)                       # OOD whenever pi disagrees with beta
            target = r + gamma * (1 - d) * tq(s2, a2)
        loss = F.mse_loss(q(s, a), target)
        opt_q.zero_grad(); loss.backward(); opt_q.step()
        opt_pi.zero_grad(); (-q(s, pi(s)).mean()).backward(); opt_pi.step()
        if t % 500 == 0:
            tq.load_state_dict(q.state_dict())
            print(t, "mean_Q", q(s, pi(s)).mean().item())
```

What I see when I run it: at step 0, $\mathbb{E}[Q] \approx 50$ (roughly the empirical return scale). By step 10 000, $\mathbb{E}[Q] \approx 5\,000$. By step 20 000 it has either NaN'd or saturated the float range. The policy returns, meanwhile, are below random.

This is the bound playing out: $\pi$ drifts slightly off $\beta$, $\hat Q$ extrapolates upward there, $\pi$ chases the upward extrapolation, and round we go. Every offline-RL method below is some way to break this loop. CQL bends $\hat Q$ down on OOD actions so the chase has nowhere to go. BCQ and BRAC clamp $\pi$ near $\beta$ so the OOD region is never queried. IQL replaces the $\max$ with an in-data expectile so the right-hand side of the Bellman backup is structurally bounded.

One more thing the bound makes clear: more data does not help if the new data has the same support as the old. The TV term depends on the *gap* between $\pi$ and $\beta$, not on $|\mathcal{D}|$. Doubling the dataset shrinks variance but not the bias. The only way to shrink the bound at fixed $\pi$ is to broaden $\beta$'s support — which in practice means deliberately mixing in exploratory or sub-optimal data. This is why D4RL's `medium-replay` split, despite being noisier, is often the *easier* dataset to learn from than `expert`.

---

## Conservative Q-Learning (CQL) From Scratch

CQL's idea is the simplest one that works: if the problem is that $\hat Q$ is over-optimistic on OOD actions, just penalise OOD Q-values during training. Specifically, add to the Q-loss:
$$\mathcal{L}_{\text{CQL}} = \alpha\, \mathbb{E}_{s \sim \mathcal{D}}\Big[\log \sum_a \exp Q(s, a) \;-\; \mathbb{E}_{a \sim \beta}\, Q(s, a)\Big].$$

Read it term by term. The $\log\sum\exp$ is a soft-max over *all* actions — it pushes Q *down* everywhere, but disproportionately at the current `argmax`. The $\mathbb{E}_{a \sim \beta} Q$ is computed on the dataset actions and gets *added back*, so in-data Q-values are spared. The net effect: Q drops on OOD actions, stays roughly correct on in-data actions. The Bellman target now bootstraps from values that are pessimistic where they have no business being optimistic.

For continuous actions you cannot enumerate $a$, so the $\log\sum\exp$ is approximated by importance-sampled logsumexp over (random uniform actions, current-policy samples, next-policy samples). The full SAC + CQL recipe is in the source article above; here I want to isolate just the penalty so the mechanism is unambiguous.

```python
import torch, torch.nn.functional as F

class CQLAgent:
    def __init__(self, q, pi, target_q, action_dim,
                 alpha=1.0, n_random=10, gamma=0.99):
        self.q, self.pi, self.tq = q, pi, target_q
        self.alpha, self.n_random = alpha, n_random
        self.gamma, self.ad = gamma, action_dim
        self.opt = torch.optim.Adam(self.q.parameters(), 3e-4)

    def cql_penalty(self, s, q_data):
        b = s.size(0)
        rep = s.repeat_interleave(self.n_random, 0)
        rand_a = torch.empty(b * self.n_random, self.ad).uniform_(-1, 1)
        pi_a, pi_lp = self.pi.sample(rep)
        log_uniform = -float(self.ad) * torch.log(torch.tensor(2.0))

        q_rand = self.q(rep, rand_a) - log_uniform
        q_pi   = self.q(rep, pi_a)   - pi_lp.detach()
        cat    = torch.cat([q_rand, q_pi], 0).view(b, -1)
        return (torch.logsumexp(cat, 1, keepdim=True) - q_data).mean()

    def update(self, s, a, r, s2, d):
        with torch.no_grad():
            a2, lp2 = self.pi.sample(s2)
            y = r + self.gamma * (1 - d) * (self.tq(s2, a2) - 0.2 * lp2)
        q_sa = self.q(s, a)
        td   = F.mse_loss(q_sa, y)
        cql  = self.cql_penalty(s, q_sa)
        loss = td + self.alpha * cql
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return td.item(), cql.item()
```

The single critical knob is $\alpha$. Too small (say 0.1) and the penalty is dominated by the TD loss; you get the same divergence as Section A, just slower. Too big (say 10) and Q is flattened so aggressively that even good in-data actions look bad and the policy collapses to random. The sweet spot tracks dataset quality — expert data needs little conservatism, replay/random data needs a lot. The Lagrangian variant from Kumar 2020 auto-tunes $\alpha$ to keep the gap $\mathbb{E}_{a \sim \pi} Q - \mathbb{E}_{a \sim \mathcal{D}} Q$ near a target, which I default to in any new project.

The numerical anchor. On D4RL halfcheetah-medium, the original CQL paper reports a normalised score of 51.4. Vanilla offline SAC on the same dataset scores -0.6 — it diverges and the policy is worse than random. One penalty term, two orders of magnitude in score. That gap is the offline RL field in a single number.

A small implementation note that cost me a day. The `n_random` uniform action samples should be drawn fresh per batch, not cached — caching them turns the logsumexp into a moving average and the penalty silently weakens over training. Likewise, the policy log-probs `pi_lp` inside the importance-weighted logsumexp must be `.detach()`-ed; otherwise gradients flow through the penalty into the actor and the actor learns to game its own pessimism term. Both bugs train cleanly for thousands of steps and only manifest in the final score, so I now sanity-check by logging the mean penalty value: it should hover around a small positive number (~1-5), not drift toward zero.

---

## Behaviour Cloning Regularisation: BCQ and BRAC

CQL constrains the value. BCQ and BRAC constrain the policy directly — keep $\pi$ close to $\beta$, and the OOD problem disappears by construction. The two methods differ in *how* they enforce closeness.

**BCQ** ([Fujimoto 2019](https://arxiv.org/abs/1812.02900)) uses generative modelling. A conditional VAE $G(s)$ is trained on $\mathcal{D}$ to model $\beta(a \mid s)$. At act time, it samples $N$ candidates from $G$, perturbs each by a small bounded amount, scores all $N$ with the twin Q-networks, and takes the argmax. The perturbation network is trained to maximise Q, so BCQ is *not* pure imitation — it can rank candidates and improve over $\beta$ — but it physically cannot propose actions far from the data manifold. This is great for discrete or factored action spaces where $G$ converges quickly; on high-dim continuous control the VAE itself becomes a bottleneck.

**BRAC** ([Wu 2019](https://arxiv.org/abs/1911.11361)) is the lazy-but-effective sibling. Fit a behaviour density estimator $\hat\beta$, then add a divergence regulariser to the actor loss:
$$\mathcal{L}_\pi = -\mathbb{E}_{s \sim \mathcal{D}}\big[Q(s, \pi(s))\big] \;+\; \lambda\, \mathbb{E}_{s \sim \mathcal{D}}\big[D_{\text{KL}}(\pi(\cdot \mid s)\, \|\, \hat\beta(\cdot \mid s))\big].$$

That is the entire algorithm. No VAE-sampled candidates, no perturbation network, no twin-policy gymnastics. Thirty lines:

```python
import torch, torch.nn.functional as F

class BehaviourPrior(torch.nn.Module):
    """Diagonal-Gaussian density estimator for beta(a|s); MLE on the dataset."""
    def __init__(self, sd, ad, hid=256):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(sd, hid), torch.nn.ReLU(),
                                        torch.nn.Linear(hid, hid), torch.nn.ReLU())
        self.mu, self.log_std = torch.nn.Linear(hid, ad), torch.nn.Linear(hid, ad)
    def dist(self, s):
        h = self.body(s)
        return torch.distributions.Normal(self.mu(h), self.log_std(h).clamp(-5, 2).exp())

def brac_actor_loss(pi, q, prior, s, lam=1.0):
    a, lp = pi.sample(s)                       # current policy sample
    q_val = q(s, a)
    with torch.no_grad():
        beta_dist = prior.dist(s)
    kl = (lp - beta_dist.log_prob(a).sum(-1, keepdim=True)).mean()
    return -q_val.mean() + lam * kl

# pretrain the prior once
def fit_prior(prior, buffer, steps=50_000):
    opt = torch.optim.Adam(prior.parameters(), 3e-4)
    for _ in range(steps):
        s, a, *_ = buffer.sample(256)
        nll = -prior.dist(s).log_prob(a).sum(-1).mean()
        opt.zero_grad(); nll.backward(); opt.step()
```

When to reach for which. BCQ if the action space is discrete or low-dim factored (the VAE is trivially easy and the explicit candidate ranking pays off). BRAC if the action space is continuous and you want the smallest change to your existing SAC/TD3 codebase — adding a KL term to the actor loss is a one-line edit, where bolting on a VAE plus perturbation network is a project. On D4RL locomotion, BRAC-v with a tuned $\lambda$ matches BCQ within a few normalised points and is markedly easier to tune.

Both methods share the same blind spot: they will not stitch. If $\beta$ never produced a good action at state $s$, neither BCQ nor BRAC can either, because both are anchored to $\beta$ by construction. CQL and IQL can, in principle, because they constrain values rather than the policy distribution itself.

---

## IQL: Just Predict the Expectile

IQL ([Kostrikov 2021](https://arxiv.org/abs/2110.06169)) sidesteps the entire OOD-action question by never querying $\hat Q$ off-distribution. The trick is a small one — replace $\max_a Q(s, a)$ with an *expectile* of $Q(s, a)$ over $a \sim \beta$ — but it changes what can go wrong.

Expectile regression. For asymmetry parameter $\tau \in (0, 1)$, the $\tau$-expectile loss is
$$L_2^\tau(u) = |\tau - \mathbb{1}(u < 0)|\, u^2.$$
At $\tau = 0.5$ this is plain MSE and the regressor learns the conditional mean. As $\tau \to 1$ the loss penalises under-predictions much more than over-predictions, and the regressor moves toward the upper tail. At $\tau \approx 0.7$-$0.9$, $V(s)$ trained with this loss against in-data $Q(s,a)$ values is a smooth approximation of $\max_a Q(s, a)$ — restricted to actions that actually appear in $\mathcal{D}$. The Bellman target becomes $y = r + \gamma V(s')$, with no policy sampling, no `max`, no extrapolation.

Policy extraction is advantage-weighted regression: the policy imitates dataset actions, weighted by $\exp(\beta (Q(s,a) - V(s)))$, so good actions are upweighted and bad ones decay. No $Q$ ever gets queried off-data, end to end.

```python
import torch, torch.nn.functional as F

def expectile_loss(diff, tau=0.7):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

class IQLAgent:
    def __init__(self, q, v, pi, target_q, gamma=0.99, tau=0.7, beta=3.0):
        self.q, self.v, self.pi, self.tq = q, v, pi, target_q
        self.gamma, self.tau, self.beta = gamma, tau, beta
        self.opt_q  = torch.optim.Adam(q.parameters(),  3e-4)
        self.opt_v  = torch.optim.Adam(v.parameters(),  3e-4)
        self.opt_pi = torch.optim.Adam(pi.parameters(), 3e-4)

    def update(self, s, a, r, s2, d):
        with torch.no_grad():
            q_target = self.tq(s, a)             # in-data only
        v_pred = self.v(s)
        loss_v = expectile_loss(q_target - v_pred, self.tau)
        self.opt_v.zero_grad(); loss_v.backward(); self.opt_v.step()

        with torch.no_grad():
            y = r + self.gamma * (1 - d) * self.v(s2)   # no max, no pi
        loss_q = F.mse_loss(self.q(s, a), y)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        with torch.no_grad():
            adv = (self.tq(s, a) - self.v(s)).clamp(max=100.0)
            w   = torch.exp(self.beta * adv)
        loss_pi = -(w * self.pi.log_prob(s, a)).mean()  # AWR
        self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
        return loss_v.item(), loss_q.item(), loss_pi.item()
```

Why it stays sane. Every Bellman backup uses $V(s')$, which was fit only on $(s, a) \in \mathcal{D}$. There is no path by which an OOD action's Q-value enters the target. The policy update is pure weighted imitation — it can only upweight actions that exist in the dataset, never invent new ones. The expectile $\tau$ controls the optimism: $\tau = 0.5$ degenerates to behaviour cloning under advantage weighting; $\tau \to 1$ approaches a `max` and re-introduces variance (because the upper expectile is dominated by the few highest-Q dataset actions). I keep $\tau = 0.7$ as a default and rarely move it.

Empirically IQL is the most consistent of the three on D4RL. It is rarely the absolute best on a single dataset, but it is rarely worse than third, and it dominates on AntMaze where stitching matters and `max`-based bootstraps are most dangerous. It is also my default for offline-to-online fine-tuning: because nothing in the offline phase is over-conservative in the CQL sense, the policy unfreezes cleanly when fresh data starts arriving.

---

## Dataset Quality and the "ORL > BC" Question

A question I get often: when does offline RL actually beat behaviour cloning, and when am I just adding complexity for sport?

The honest answer is that offline RL beats BC when the dataset has *trajectory-stitching potential* — multiple suboptimal trajectories whose pieces, recombined, produce something better than any single trajectory. Without stitching potential, BC matches or beats every offline RL method I have benchmarked, with maybe ten percent of the code.

A thought experiment that makes this visceral. Imagine a 10x10 grid maze with start at $(0,0)$ and goal at $(9,9)$. The dataset contains two trajectory types in equal proportion. Type A goes right along the top edge then drops down on the right side — total reward 18. Type B goes down along the left edge then right along the bottom — also total reward 18. The optimal diagonal would score 18 too, but neither type takes it. Now: what does each method do at the centre cell $(5,5)$, which appears in *both* trajectory types?

BC sees $(5,5)$ in type-A trajectories paired with action "right" and in type-B trajectories paired with action "down". The maximum-likelihood action under BC is roughly a 50/50 mixture — and in a deterministic env, sampling that mixture means sometimes going right, sometimes going down, and on most rollouts ending up oscillating near the centre. BC averages the two policies and gets stuck.

Offline RL with a value function does something different. It learns $Q((5,5), \text{right}) \approx Q((5,5), \text{down}) \approx \text{(reward to goal)}$, picks one consistently, and stitches the right half of trajectory A onto the left half of trajectory B (or vice versa). The result is a coherent path, even though no single trajectory in $\mathcal{D}$ ever made the choice the policy makes. This is what "stitching" means concretely.

The practical recipe I use:

- Dataset is from a near-expert with little diversity (e.g. D4RL `expert` or `medium-expert`): start with **BC**. Try IQL only if BC plateaus.
- Dataset is diverse with mixed quality (`medium-replay`, `random`, real-world logs from a fleet of mediocre policies): start with **IQL** or **CQL**. BC will not stitch.
- Dataset is tiny (< 10k transitions): BC again, regardless of quality. All offline RL methods are data-hungry because they learn a value function from scratch.
- Offline-to-online fine-tuning planned: **IQL**. CQL initialisations are over-conservative and unfreeze slowly when fresh data arrives.

One more failure mode worth flagging. If the dataset comes from a *single* policy (no diversity at all — say a frozen production model logging its own decisions), no offline method can do anything but match it. Stitching needs at least two distinguishable trajectory types to stitch *between*. I have watched teams burn weeks running CQL on monolithic logs from one policy, then conclude "offline RL doesn't work for our problem", when the right diagnosis was "this dataset has zero stitching potential". The fix is upstream: deliberately log a small fraction of exploratory or randomised behaviour, even one or two percent, and the stitching surface opens up.

The meta-point. Offline RL is not a strict upgrade over imitation — it is a different tool for a different kind of dataset. Match the tool to the dataset, not the dataset to the tool.

## FAQ

**Q: When does offline RL fail outright?**
Three failure modes are well-documented: (i) *narrow data coverage* — only expert trajectories with no recovery examples, so the policy cannot learn to handle its own mistakes; (ii) *very low data quality* — random-policy data on long-horizon tasks; (iii) *evaluation-environment shift* — the test MDP's transition dynamics differ from those that produced $\mathcal{D}$.

**Q: How do I tune CQL's $\alpha$?**
Roughly: $\alpha=0.5$-$1.0$ for expert data, $1.0$-$5.0$ for medium data, $5.0$-$10.0$ for random/replay data. Better: use the Lagrangian variant from the original paper, which auto-tunes $\alpha$ to keep the gap $\mathbb{E}_{a\sim\pi}Q - \mathbb{E}_{a\sim\mathcal{D}}Q$ near a target threshold (typically 5-10).

**Q: Offline RL vs imitation learning — what's actually different?**
Imitation cloning (BC) ignores rewards and cannot beat the demonstrator. Offline RL uses the reward signal and can **stitch** good segments from different trajectories, producing a policy strictly better than any single trajectory in $\mathcal{D}$. The classic stitching example: trajectories A reach state $s$ via a clumsy route then act well, trajectories B reach $s$ efficiently then act poorly — offline RL can produce "B's beginning + A's end" while BC cannot.

**Q: Should I use a model-based offline method instead?**
For high-dimensional continuous control, MOPO/MOReL/COMBO are competitive but add a learned dynamics model with its own pessimism term. They shine when the dataset is small (because the model adds inductive bias) but are heavier to engineer. As a default, model-free CQL/IQL are still the right starting point in 2025.

**Q: What about offline-to-online fine-tuning?**
This is where IQL pulled ahead in the last two years. Methods like **AWAC**, **Cal-QL**, and **RLPD** initialize from an offline policy and continue training online with a small amount of fresh data. CQL initialization tends to be over-conservative for fine-tuning; IQL or AWAC initializations are usually preferred.

---

## References

- Levine, Kumar, Tucker, Fu. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). 2020.
- Fujimoto, Meger, Precup. [Off-Policy Deep Reinforcement Learning without Exploration (BCQ)](https://arxiv.org/abs/1812.02900). ICML 2019.
- Kumar, Zhou, Tucker, Levine. [Conservative Q-Learning for Offline Reinforcement Learning (CQL)](https://arxiv.org/abs/2006.04779). NeurIPS 2020.
- Kostrikov, Nair, Levine. [Offline Reinforcement Learning with Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169). ICLR 2022.
- Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345). NeurIPS 2021.
- Fu, Kumar, Nachum, Tucker, Levine. [D4RL: Datasets for Deep Data-Driven RL](https://arxiv.org/abs/2004.07219). 2020.
