---
title: "Reinforcement Learning (11): Hierarchical RL and Meta-Learning"
date: 2024-07-11 09:00:00
tags:
  - Reinforcement Learning
  - Hierarchical RL
  - Meta-Learning
  - MAML
  - Options Framework
categories: Reinforcement Learning
series: Reinforcement Learning
part: 11
total_parts: 12
lang: en
mathjax: true
description: "A deep dive into hierarchical RL (Options, MAXQ, Feudal Networks, goal-conditioned policies) and meta-RL (MAML, FOMAML, RL^2). Covers temporal abstraction, semi-MDPs, manager-worker architectures, second-order meta-gradients and recurrent meta-learners, with annotated PyTorch implementations."
---

Standard RL treats every problem as a flat sequence of atomic decisions: observe state, pick an action, receive a reward, repeat. That works when the horizon is short and rewards are dense, but it breaks down on the kind of tasks humans solve effortlessly. "Make breakfast" is not one decision; it is a tree of subtasks --- *brew coffee*, *fry eggs*, *toast bread*, *plate it up* --- each of which is itself a small policy. **Hierarchical RL (HRL)** lets agents reason and act at multiple timescales by treating macro-actions as first-class citizens.

A second weakness of standard RL is that every new task is learned from scratch. A bicycle rider becomes a motorcycle rider in an afternoon, not in 10 million environment steps. **Meta-RL** attacks this by training across a *distribution* of tasks so that adaptation to a held-out task takes a handful of episodes --- or even a single forward pass through a recurrent network.

This post unifies the two ideas: hierarchy buys temporal abstraction, meta-learning buys task abstraction. Both reduce the effective dimensionality of the learning problem, and the modern frontier (FuN, HIRO, MAML, RL$^2$) combines them aggressively.

## What you will learn

- **Options framework** --- semi-Markov decision processes and intra-option Q-learning
- **MAXQ** --- value-function decomposition along a task hierarchy
- **Feudal RL** (FuN, HIRO) --- continuous subgoals with manager-worker architectures
- **Goal-conditioned RL** --- universal value functions and HER
- **MAML / FOMAML** --- learning an initialisation that adapts in a few gradient steps
- **RL$^2$** --- folding the inner RL algorithm into an RNN's hidden state
- **Working code** --- intra-option Q-learning in Four Rooms and a MAML policy gradient on 2-D navigation

## Prerequisites

- Q-learning, policy gradients and value functions ([Parts 1--6](/en/reinforcement-learning-01-fundamentals-and-core-concepts/))
- Comfort with RNN unrolling and second-order autodiff
- PyTorch

---

## 1. Hierarchy: the Options framework

### Why temporal abstraction matters

A flat policy makes one decision per environment step, so its credit-assignment path on a horizon-$T$ task has length $T$ and its exploration tree has $|\mathcal{A}|^T$ leaves. Both grow exponentially. A hierarchy with average macro-action length $\bar k$ shrinks the decision count to $T/\bar k$ and the exploration tree to $|\mathcal{O}|^{T/\bar k}$, where $\mathcal{O}$ is a (typically small) set of options. The figure below contrasts the two views.

![Options Framework: timeline view and the (I, π, β) triple](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig1_options_framework.png)

Beyond the asymptotic argument, hierarchies offer three practical wins:

1. **Modularity** --- options trained on one task transfer to others (e.g. *go-to-door* generalises across navigation problems).
2. **Interpretability** --- the high-level policy is small enough to inspect, and decisions are taken at semantically meaningful checkpoints.
3. **Reward shaping** --- subgoals provide dense intrinsic rewards even when extrinsic reward is sparse.

### The option triple

Following Sutton, Precup & Singh (1999), an **option** is a temporally extended action

$$
o = \langle \mathcal{I}, \pi_o, \beta \rangle,
$$

where $\mathcal{I} \subseteq \mathcal{S}$ is the set of states from which the option may be initiated, $\pi_o(a \mid s)$ is the option's internal policy and $\beta(s) \in [0, 1]$ is its termination probability. Once options are introduced, the underlying MDP becomes a **semi-MDP**: the high-level policy $\mu(o \mid s)$ chooses an option, the option runs until $\beta$ fires, and only then does the next high-level decision happen.

### Intra-option Q-learning

The naive way to learn $Q(s, o)$ is to wait until the option terminates, observe the cumulative reward and bootstrap. This is wasteful: an option may run for dozens of steps and we throw away the intermediate transitions. **Intra-option Q-learning** (Sutton et al., 1999) updates $Q(s, o)$ at *every* step, exploiting the fact that the same transition $(s, a, r, s')$ is consistent with every option that would have taken action $a$ in state $s$:

$$
Q(s, o) \leftarrow Q(s, o) + \alpha \big[r + \gamma U(s', o) - Q(s, o)\big],
$$

with the **continuation value**

$$
U(s', o) = (1 - \beta(s'))\, Q(s', o) + \beta(s')\, \max_{o'} Q(s', o').
$$

The continuation value is the elegant piece: if the option keeps going we keep its Q-value, otherwise we hand control back to the high-level policy and take the best alternative.

```python
import numpy as np
from collections import defaultdict


class IntraOptionQLearning:
    """Intra-option Q-learning over a discrete option set.

    Every primitive transition updates Q(s, o) for the option currently
    in control, plus the option's flat Q-table that defines its
    internal policy.
    """

    def __init__(self, env, options, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.env, self.options = env, options
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = defaultdict(lambda: np.zeros(len(options)))     # Q(s, o)
        self.Q_prim = defaultdict(lambda: np.zeros(env.n_actions))  # primitive

    def _select_option(self, state):
        available = [i for i, o in enumerate(self.options)
                     if o.can_initiate(state)]
        if not available:
            return None
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available))
        q = self.Q[state][available]
        return int(available[np.argmax(q)])

    def train_episode(self, max_steps=1000):
        state = self.env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            oid = self._select_option(state)
            if oid is None:
                break
            option = self.options[oid]

            while not option.should_terminate(state):
                action = option.act(state, self.epsilon)
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Continuation value U(s', o)
                if option.should_terminate(next_state):
                    U = np.max(self.Q[next_state])
                else:
                    U = self.Q[next_state][oid]

                # High-level update
                self.Q[state][oid] += self.alpha * (
                    reward + self.gamma * U - self.Q[state][oid])

                # Internal policy update (flat Q over primitives)
                td = (reward + self.gamma
                      * np.max(self.Q_prim[next_state])
                      - self.Q_prim[state][action])
                self.Q_prim[state][action] += self.alpha * td
                option.policy[state] = int(np.argmax(self.Q_prim[state]))

                state = next_state
                if done:
                    return total_reward
        return total_reward
```

In the canonical Four Rooms benchmark, intra-option Q-learning with four hand-crafted "go through doorway" options converges 3--5$\times$ faster than flat Q-learning, and the learned options transfer cleanly to new goal locations.

### MAXQ: value decomposition along a task tree

Where Options leaves the hierarchy *implicit* in the option set, **MAXQ** (Dietterich, 2000) makes it explicit. The agent is given a directed acyclic graph of subtasks; for each composite task $i$ and each child $a$ the value decomposes as

$$
Q_i(s, a) = V_a(s) + C_i(s, a),
$$

where $V_a(s)$ is the value of completing the *child* subtask and $C_i(s, a)$ is the **completion function** --- the value of finishing the parent task once the child returns. Because $V_a$ depends only on $a$, it can be reused across every parent that invokes $a$, which is where the sample-efficiency win comes from.

![MAXQ-style hierarchy with shared primitives](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig6_task_decomposition.png)

MAXQ pays for that win with **recursive optimality** rather than global optimality: the policy is optimal *given the decomposition*. If your task graph cannot express the truly optimal solution, MAXQ will not find it.

---

## 2. Feudal RL: continuous subgoals with manager-worker

Discrete options scale poorly: in continuous-control or pixel-input domains we cannot enumerate a useful option set by hand. **Feudal Networks** (FuN, Vezhnevets et al., 2017) and **HIRO** (Nachum et al., 2018) replace the discrete option set with a *continuous goal vector* produced by a high-level Manager.

![Feudal RL: Manager sets goals every c steps, Worker executes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig2_feudal_architecture.png)

The Manager sees state $s_t$ and emits a goal $g_t \in \mathbb{R}^d$ every $c$ steps (FuN uses $c = 10$). The Worker is a goal-conditioned policy $\pi_\phi(a \mid s, g)$ that earns an **intrinsic reward** for moving in the direction of the goal:

$$
r^{\text{int}}_t = \cos\!\big(s_{t+c} - s_t,\, g_t\big),
$$

while the Manager is trained on the *extrinsic* environment reward. This decoupling is what makes Feudal architectures so attractive: the Worker learns motor control on a dense, geometric reward, and the Manager focuses on long-horizon credit assignment with a much smaller effective horizon ($T/c$).

### HIRO's relabelling trick

Feudal training has a chicken-and-egg problem: the Worker is non-stationary (because it is still learning), so old goals stored in the replay buffer no longer correspond to what the Worker actually achieves. HIRO fixes this with **subgoal relabelling**: when sampling a transition $(s_t, g_t, a_{t:t+c}, s_{t+c})$, replace $g_t$ with the goal that maximises the Worker's likelihood of producing the action sequence we actually saw,

$$
\tilde g_t = \arg\max_{g} \log \pi_\phi(a_{t:t+c} \mid s_{t:t+c},\, g).
$$

This keeps the Worker's training data on-policy with respect to its current parameters and dramatically stabilises off-policy learning of the Manager.

---

## 3. Goal-conditioned RL and HER

Goal-conditioned policies $\pi(a \mid s, g)$ deserve a section of their own, because they are the bridge between hierarchy and meta-learning. The same network can pursue many goals; the goal $g$ is just an additional input.

![Goal-conditioned policy and 4 trajectories from one network](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig3_goal_conditioned.png)

The classical formalisation is **Universal Value Function Approximators** (UVFA, Schaul et al., 2015): learn $V(s, g)$ or $Q(s, a, g)$ instead of $V(s), Q(s, a)$. Without further tricks UVFAs suffer badly from sparse rewards: most goals are never reached during exploration, so the reward signal is essentially zero.

**Hindsight Experience Replay** (HER, Andrychowicz et al., 2017) is the standard fix. After running an episode that *failed* to reach goal $g$, we relabel the trajectory with a goal it *did* reach (typically the final state), turning a failure into a successful demonstration for a different task:

$$
(s_t, a_t, r, s_{t+1}, g) \;\longrightarrow\; (s_t, a_t, r', s_{t+1},\, g' = s_T).
$$

Combined with off-policy methods (DDPG, SAC), HER turns sparse-reward goal reaching from "essentially impossible" into "routine".

---

## 4. Meta-RL: learning to learn

Meta-RL assumes a *task distribution* $p(\mathcal{T})$ rather than a single MDP. At meta-train time the agent sees many tasks $\mathcal{T}_i \sim p(\mathcal{T})$; at meta-test time it sees a fresh task and must adapt with as few interactions as possible.

![Task distribution and adaptation curves on a held-out task](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig5_meta_rl_distribution.png)

The two dominant families are:

- **Optimisation-based** (MAML, Reptile, ANIL): adaptation = a few gradient steps on the test task.
- **Recurrent / context-based** (RL$^2$, PEARL): adaptation = updating the hidden state of an RNN as data arrives, no gradients needed at test time.

### MAML --- learning a good initialisation

The meta-objective of **Model-Agnostic Meta-Learning** (MAML, Finn et al., 2017) is to find parameters $\theta$ such that one (or a few) inner-loop SGD steps on any task $\mathcal{T}_i$ produce strong performance:

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta),
\qquad
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{i} \mathcal{L}_{\mathcal{T}_i}(\theta_i').
$$

The outer gradient differentiates *through* the inner update, so it contains a Hessian term $\nabla^2 \mathcal{L}$. That is what makes MAML expensive --- and what motivates **FOMAML**, which simply ignores the second-order term. Empirically FOMAML is roughly $10\times$ faster per outer step and loses less than 5% of the final return.

![MAML in parameter space and as a two-loop algorithm](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig4_maml_inner_outer_loop.png)

The picture on the left is the most useful intuition: meta-training does *not* try to find an initialisation that is good on any single task. It finds an initialisation that sits in a "sweet spot" from which one gradient step lands close to each task-specific optimum.

```python
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    """Diagonal-Gaussian policy for continuous control."""

    def __init__(self, state_dim=2, action_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def sample(self, state):
        h = self.trunk(state)
        mean = torch.tanh(self.mean(h))
        std = self.log_std.exp().clamp(1e-3, 1.0)
        dist = torch.distributions.Normal(mean, std)
        a = dist.rsample()
        return a, dist.log_prob(a).sum(-1)


class MAML:
    """First- or second-order MAML for REINFORCE-style policy gradient."""

    def __init__(self, policy, inner_lr=0.1, outer_lr=1e-3,
                 first_order=False):
        self.policy = policy
        self.inner_lr = inner_lr
        self.first_order = first_order
        self.opt = torch.optim.Adam(policy.parameters(), lr=outer_lr)

    def _rollout(self, env, max_steps=50):
        state = env.reset()
        rewards, log_probs = [], []
        for _ in range(max_steps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            a, lp = self.policy.sample(s)
            state, r, done = env.step(a.detach().numpy()[0])
            rewards.append(r)
            log_probs.append(lp)
            if done:
                break
        return rewards, log_probs

    @staticmethod
    def _returns(rewards, gamma=0.99):
        G, out = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            out.insert(0, G)
        out = torch.as_tensor(out, dtype=torch.float32)
        return (out - out.mean()) / (out.std() + 1e-8)

    def _loss(self, rewards, log_probs):
        returns = self._returns(rewards)
        return -(torch.stack(log_probs) * returns).mean()

    def adapt(self, env):
        """One inner step on `env` returning the adapted parameter list."""
        rewards, log_probs = self._rollout(env)
        loss = self._loss(rewards, log_probs)
        grads = torch.autograd.grad(
            loss, self.policy.parameters(),
            create_graph=not self.first_order,
        )
        return [p - self.inner_lr * g
                for p, g in zip(self.policy.parameters(), grads)]

    def meta_step(self, task_envs):
        meta_loss = 0.0
        for env in task_envs:
            adapted = self.adapt(env)
            # Functional forward pass with adapted parameters
            backup = [p.detach().clone() for p in self.policy.parameters()]
            for p, a in zip(self.policy.parameters(), adapted):
                p.data = a.data
            rewards, log_probs = self._rollout(env)
            meta_loss = meta_loss + self._loss(rewards, log_probs)
            for p, b in zip(self.policy.parameters(), backup):
                p.data = b
        meta_loss = meta_loss / len(task_envs)

        self.opt.zero_grad()
        meta_loss.backward()
        self.opt.step()
        return meta_loss.item()
```

The implementation above swaps adapted weights into the network for the meta-evaluation rollout rather than relying on a functional forward pass --- it is the simplest readable version, and good enough for small networks. For research-scale MAML you want a proper functional API such as `higher` or `torch.func.functional_call`.

### RL$^2$ --- folding the algorithm into an RNN

**RL$^2$** (Duan et al., 2016; Wang et al., 2016) takes a different route: keep the agent's parameters fixed at meta-test time and let the RNN's hidden state do the adaptation. The recurrent policy receives the augmented input

$$
x_t = (s_t,\, a_{t-1},\, r_{t-1},\, d_{t-1}),
$$

i.e. state plus *the previous action, reward and done-flag*. Across episodes within a meta-trial the hidden state $h_t$ accumulates information about the current task --- effectively performing Bayesian belief updates implicitly. At meta-train time the outer optimiser (PPO or A2C) shapes the RNN weights so that this implicit "inner algorithm" is sample-efficient on $p(\mathcal{T})$.

![RL² unrolled across a meta-trial spanning multiple episodes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig7_rl_squared.png)

RL$^2$ has two attractive properties: (i) zero gradient computation at test time, so adaptation is one forward pass per step; (ii) the adaptation procedure is *learned*, so it can in principle exceed any hand-designed optimiser on the training distribution. The price is a hard credit-assignment problem during meta-training --- the gradient must flow through hundreds of recurrent steps spanning multiple episodes.

---

## 5. Frequently asked questions

**Why does the Options framework actually accelerate learning?**
Three compounding effects: (i) the effective horizon shrinks from $T$ to roughly $T/\bar k$; (ii) the high-level branching factor $|\mathcal{O}|$ is usually much smaller than $|\mathcal{A}|$; (iii) intra-option learning means every primitive transition contributes to the value of *every* compatible option, not just the one in control.

**Hierarchically optimal vs globally optimal --- what's the difference?**
A hierarchically optimal policy is the best policy that respects the given decomposition (option set or task graph). A globally optimal policy is the best policy on the underlying flat MDP. They diverge whenever the decomposition cannot express the optimum --- e.g. an option that always runs for at least 10 steps cannot implement a policy that needs to switch behaviour every 3 steps.

**Why does MAML need second-order gradients, and how much does FOMAML lose?**
The meta-loss is $\mathcal{L}_{\mathcal{T}_i}(\theta_i')$ where $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$. Differentiating with respect to $\theta$ yields $(I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i})\, \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(\theta_i')$, which contains the Hessian. FOMAML drops the $-\alpha \nabla^2 \mathcal{L}$ factor, treating $\theta_i'$ as a stop-gradient. The original MAML paper reports <5% performance loss on the standard few-shot benchmarks.

**When should I use MAML vs RL$^2$?**
Use MAML when you can afford gradient computation at adaptation time and your task distribution is broad enough that a fixed policy cannot do well on all of it. Use RL$^2$ when the task family is narrow and exploitative (e.g. multi-armed bandits with shifting arms), or when test-time gradients are infeasible. PEARL --- which infers a task embedding with a separate encoder --- is often a happy medium.

**Why is HER specific to off-policy algorithms?**
HER changes the goal that a transition was collected for, which violates on-policy assumptions: the action distribution that produced the data was conditioned on the original goal, not the relabelled one. Off-policy algorithms (DQN, DDPG, SAC) only require that we know $\pi(a \mid s, g')$ at update time, which we do, so the relabelling is harmless.

---

## Series Navigation

- **Previous**: [Part 10 -- Offline RL](/en/reinforcement-learning-10-offline-reinforcement-learning/)
- **Next**: [Part 12 -- RLHF and LLM Applications](/en/reinforcement-learning-12-rlhf-and-llm-applications/)
- [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)

## References

1. Sutton, Precup & Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence, 1999.
2. Dietterich. *Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition.* JAIR, 2000.
3. Vezhnevets et al. *FeUdal Networks for Hierarchical Reinforcement Learning.* ICML 2017. arXiv:1703.01161.
4. Nachum et al. *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS 2018. arXiv:1805.08296.
5. Schaul et al. *Universal Value Function Approximators.* ICML 2015.
6. Andrychowicz et al. *Hindsight Experience Replay.* NeurIPS 2017. arXiv:1707.01495.
7. Finn, Abbeel & Levine. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML 2017. arXiv:1703.03400.
8. Duan et al. *RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning.* 2016. arXiv:1611.02779.
9. Wang et al. *Learning to reinforcement learn.* 2016. arXiv:1611.05763.
10. Rakelly et al. *Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL).* ICML 2019.
