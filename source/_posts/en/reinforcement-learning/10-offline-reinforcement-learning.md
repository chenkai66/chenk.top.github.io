---
title: "Reinforcement Learning (10): Offline Reinforcement Learning"
date: 2024-07-10 09:00:00
tags:
  - Reinforcement Learning
  - Offline RL
  - CQL
  - Decision Transformer
  - BCQ
categories: Reinforcement Learning
series: Reinforcement Learning
part: 10
total_parts: 12
lang: en
mathjax: true
description: "Master offline RL: learn policies from fixed datasets without environment interaction. Covers distributional shift, Conservative Q-Learning (CQL), BCQ, Implicit Q-Learning (IQL), Decision Transformer, with a complete CQL implementation."
disableNunjucks: true
---

Every algorithm we have studied so far has the same loop at its core: act, observe, update. That loop is what makes RL work, but it is also what stops RL from being deployed. A self-driving stack cannot rehearse intersections by crashing into them. A clinical decision-support model cannot run a randomized policy on actual patients. A factory robot cannot try ten thousand grasp variants on a production line.

What these settings *do* have is logs -- millions of hours of human driving, decades of de-identified patient records, terabytes of behavior cloning data. **Offline RL** (also called *batch RL*) is the subfield that asks: can we squeeze a strong policy out of a fixed dataset, with **zero new interaction** with the environment?

The answer is "yes, but only if we are very careful," and the reason for the caveat is the central theme of this post: distributional shift between the *behavior policy* that produced the data and the *learned policy* that wants to improve on it.

## What You Will Learn

- **Why naive off-policy RL fails offline**: extrapolation error, value overestimation, and the death spiral.
- **CQL** (Conservative Q-Learning): a pessimistic regularizer that lower-bounds the true value.
- **BCQ** (Batch-Constrained Q-Learning): generative action proposals that stay inside the data manifold.
- **IQL** (Implicit Q-Learning): expectile regression that approximates `max` *without ever querying OOD actions*.
- **Decision Transformer**: reframing RL as conditional sequence modeling on returns-to-go.
- **D4RL benchmark numbers** so you know which algorithm to reach for first.
- **A complete, runnable CQL implementation** in PyTorch.

## Prerequisites

- Q-learning, target networks, and actor-critic ([Parts 2](/en/reinforcement-learning-2-q-learning-and-dqn/)-[6](/en/reinforcement-learning-6-ppo-and-trpo/)).
- Comfort with the Bellman optimality operator and importance sampling.
- PyTorch and the Gym/Gymnasium API.

---

## 1. Why Offline RL Is Genuinely Hard

![Online vs offline RL: the missing feedback loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig1_online_vs_offline.png)

In **online** RL the policy and the data distribution are coupled. As soon as the policy starts overestimating some action, the next rollout puts it under the microscope and the Bellman update corrects it. In **offline** RL the dataset is *frozen*. There is no second chance: any error the model makes about an unseen action will sit in the Q-table forever, and the `argmax` operator will happily exploit it.

### 1.1 Distributional Shift

Let $\pi_\beta$ be the behavior policy that produced the dataset $\mathcal{D}=\{(s_i,a_i,r_i,s_i')\}_{i=1}^N$, and let $\pi_\theta$ be the policy we are learning. The state-action visitation distributions are usually different,

$$d_{\pi_\theta}(s,a)\neq d_{\pi_\beta}(s,a).$$

This becomes a problem the moment $\pi_\theta$ wants to take an action $a$ for which $\pi_\beta(a\mid s)\approx 0$. The Q-network has *never* seen a target for that action; whatever value it returns is pure extrapolation from a neural net that has been trained to fit a completely different region of the input space.

### 1.2 Extrapolation Error and the Death Spiral

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

## 2. Conservative Q-Learning (CQL)

CQL ([Kumar et al., 2020](https://arxiv.org/abs/2006.04779)) is the most widely used baseline. It does not change the network, the actor, or the data pipeline. It changes one term in the loss.

### 2.1 The Idea in One Sentence

> **Push down the Q-values of any action the policy might consider, then push back up the Q-values of actions actually present in the data.** The net effect is a Q-function that is pessimistic exactly where it lacks evidence.

### 2.2 The Objective

On top of the usual TD loss $\mathcal{L}_{\mathrm{TD}}$, CQL adds

$$\mathcal{L}_{\mathrm{CQL}} \;=\; \alpha\,\Big[\,\underbrace{\log\sum_{a}\exp Q(s,a)}_{\text{push down everywhere}} \;-\; \underbrace{\mathbb{E}_{a\sim\mathcal{D}}\big[Q(s,a)\big]}_{\text{push up on data}}\Big] \;+\; \mathcal{L}_{\mathrm{TD}}.$$

The `logsumexp` is a soft maximum over actions: minimizing it pulls *all* Q-values down. Subtracting the expectation under the data distribution lets the in-data actions float back up. Out-of-distribution actions only feel the downward force.

The headline theorem ([Kumar et al., 2020, Thm 3.2](https://arxiv.org/abs/2006.04779)) is that for sufficiently large $\alpha$,

$$\hat{Q}^{\pi}(s,a)\;\leq\;Q^{\pi}(s,a)\quad\forall (s,a).$$

A pessimistic estimate is exactly what we need: any policy that maximizes a *lower bound* on the true value cannot pick a catastrophic action because it has no incentive to.

![CQL: the regularizer pushes OOD Q-values down so argmax stays inside the data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig4_cql_penalty.png)

The left panel shows what happens with a vanilla SAC-style critic on offline data: the argmax flees to the OOD bump. The right panel shows CQL's correction: the orange shaded region is the pessimism penalty, and the new argmax sits comfortably inside the data support.

### 2.3 Practical Notes

- The `logsumexp` is intractable for continuous actions, so implementations approximate it with `n_random` uniform samples plus `n_actor` samples from the current policy. 10-20 samples is usually enough.
- The original paper's "CQL($\mathcal{H}$)" variant adds a Lagrangian that auto-tunes $\alpha$ to hit a target gap; this is what the open-source `d3rlpy` and `JaxRL` implementations ship by default.
- Empirically CQL is robust on `medium-replay` and `medium` D4RL splits but can be slightly conservative on `medium-expert`, where IQL or DT often win.

### 2.4 A Complete CQL Implementation

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

## 3. BCQ: Generative Action Proposals

CQL constrains the *value*; BCQ ([Fujimoto et al., 2019](https://arxiv.org/abs/1812.02900)) constrains the *policy* directly. The principle is mechanical: never even *query* the Q-function on an action that the behavior policy would not have produced.

![BCQ: a VAE proposes candidates, a perturbation network adds a small refinement, then argmax picks one](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig3_bcq_architecture.png)

The architecture has three pieces:

1. **A conditional VAE** $G_\omega(s)$ trained on $\mathcal{D}$ via the standard ELBO. It models $\pi_\beta(a\mid s)$, so samples drawn from it stay on the behavior manifold.
2. **A perturbation network** $\xi_\phi(s,a)\in[-\Phi,\Phi]^{\dim(a)}$ that adds a small, bounded refinement to each VAE sample. With $\Phi$ small (typically 0.05), the refined action cannot drift far from the data; with $\Phi=0$ BCQ reduces to weighted imitation.
3. **Twin Q-networks** that score the $N$ refined candidates; the argmax is the action that gets executed.

The clever part is that the perturbation network is trained *to maximize Q*. So BCQ gets the benefit of policy improvement (it is not pure imitation) without the cost of OOD extrapolation (it cannot stray far from the data). The price is a more elaborate pipeline -- four networks instead of two -- and the VAE itself must be trained well, otherwise the candidates are biased.

---

## 4. IQL: Avoid the Bootstrap Problem Entirely

IQL ([Kostrikov et al., 2022](https://arxiv.org/abs/2110.06169)) is the most elegant of the three. Its observation: every offline-RL pathology comes from the $\max_{a'}Q(s',a')$ in the Bellman target, because that is where OOD actions enter. So... just don't use it.

IQL learns a separate state-value function $V(s)$ via **expectile regression** on the in-data actions only:

$$\mathcal{L}_V \;=\; \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[L_2^{\tau}\big(Q(s,a)-V(s)\big)\big],\qquad L_2^{\tau}(u)=\big|\tau-\mathbb{1}(u<0)\big|\,u^{2}.$$

When $\tau=0.5$ this is plain MSE and $V$ learns the *mean*. When $\tau\to 1$ it asymmetrically penalizes under-estimates much more than over-estimates, so $V$ converges to the *upper expectile* of $Q(s,a)$ over the actions seen in the data.

![IQL: the expectile loss is asymmetric; pushing tau toward 1 pulls V toward max_a Q without ever leaving the data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig5_iql_expectile.png)

For $\tau\approx 0.7$-$0.9$, $V(s)$ is an excellent proxy for $\max_a Q(s,a)$ -- *and it was computed from in-data actions only*. The Bellman target then becomes

$$y \;=\; r + \gamma\,V(s'),$$

with no `max`, no policy sampling, no extrapolation. The actor is recovered via **advantage-weighted regression**:

$$\mathcal{L}_\pi \;=\; -\mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\exp\big(\beta\,(Q(s,a)-V(s))\big)\,\log\pi_\theta(a\mid s)\big].$$

IQL has the smallest moving-parts surface of any modern offline-RL algorithm, and it consistently tops the D4RL leaderboard on AntMaze and Adroit -- environments where good data is sparse and bootstrapping with `max` is most dangerous.

---

## 5. Decision Transformer: RL as Sequence Modeling

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
- No causal credit assignment -- the model is trained to *imitate trajectories conditioned on outcome*, not to reason about which action *caused* the outcome.
- Generally requires more parameters and more data than CQL/IQL to reach the same wall-clock score on small benchmarks.

---

## 6. Putting Numbers to It: D4RL

![D4RL MuJoCo locomotion: BC, BCQ, CQL, IQL, DT across three data qualities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/10-offline-reinforcement-learning/fig7_d4rl_benchmark.png)

The figure compiles representative normalized scores from the original CQL ([Kumar et al., 2020](https://arxiv.org/abs/2006.04779)), IQL ([Kostrikov et al., 2022](https://arxiv.org/abs/2110.06169)) and DT ([Chen et al., 2021](https://arxiv.org/abs/2106.01345)) papers on the canonical [D4RL](https://arxiv.org/abs/2004.07219) MuJoCo locomotion suite (`hopper`, `halfcheetah`, `walker2d` averaged). 100 = expert performance, 0 = random.

Three takeaways:

1. **`medium-replay` is where conservatism matters most.** This split is generated by an early SAC checkpoint -- lots of bad actions, lots of recovery transitions. CQL and IQL roughly *double* BC's score; DT trails because it has no value bootstrapping to stitch together good sub-trajectories.
2. **`medium-expert` is where DT shines.** When the data already contains near-optimal trajectories, sequence modeling wins on simplicity.
3. **IQL is the most consistent.** It is rarely the absolute best on any single dataset, but it is rarely worse than third, and its training is the most stable of the four.

When in doubt, start with **IQL** for stability or **CQL** for simplicity, and reach for **DT** when the dataset already contains near-expert demonstrations.

---

## 7. Frequently Asked Questions

**Q: When does offline RL fail outright?**
Three failure modes are well-documented: (i) *narrow data coverage* -- only expert trajectories with no recovery examples, so the policy cannot learn to handle its own mistakes; (ii) *very low data quality* -- random-policy data on long-horizon tasks; (iii) *evaluation-environment shift* -- the test MDP's transition dynamics differ from those that produced $\mathcal{D}$.

**Q: How do I tune CQL's $\alpha$?**
Roughly: $\alpha=0.5$-$1.0$ for expert data, $1.0$-$5.0$ for medium data, $5.0$-$10.0$ for random/replay data. Better: use the Lagrangian variant from the original paper, which auto-tunes $\alpha$ to keep the gap $\mathbb{E}_{a\sim\pi}Q - \mathbb{E}_{a\sim\mathcal{D}}Q$ near a target threshold (typically 5-10).

**Q: Offline RL vs imitation learning -- what's actually different?**
Imitation cloning (BC) ignores rewards and cannot beat the demonstrator. Offline RL uses the reward signal and can **stitch** good segments from different trajectories, producing a policy strictly better than any single trajectory in $\mathcal{D}$. The classic stitching example: trajectories A reach state $s$ via a clumsy route then act well, trajectories B reach $s$ efficiently then act poorly -- offline RL can produce "B's beginning + A's end" while BC cannot.

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

---

## Series Navigation

- **Previous**: [Part 9 -- Multi-Agent RL](/en/reinforcement-learning-9-multi-agent-rl/)
- **Next**: [Part 11 -- Hierarchical RL and Meta-Learning](/en/reinforcement-learning-11-hierarchical-and-meta-rl/)
- [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)
