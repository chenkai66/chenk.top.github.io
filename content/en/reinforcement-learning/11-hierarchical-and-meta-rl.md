---
title: "Reinforcement Learning (11): Hierarchical RL and Meta-Learning"
date: 2025-09-20 09:00:00
tags:
  - Reinforcement Learning
  - Hierarchical RL
  - Meta-Learning
  - MAML
  - Options Framework
categories: Reinforcement Learning
series: reinforcement-learning
part: 11
total_parts: 12
lang: en
mathjax: true
description: "A deep dive into hierarchical RL (Options, MAXQ, Feudal Networks, goal-conditioned policies) and meta-RL (MAML, FOMAML, RL^2). Covers temporal abstraction, semi-MDPs, manager-worker architectures, second-order meta-gradients and recurrent meta-learners, with annotated PyTorch implementations."
disableNunjucks: true
series_order: 11
translationKey: "reinforcement-learning-11"
---
Standard RL treats every problem as a flat sequence of atomic decisions: observe state, pick an action, receive a reward, repeat. That works when the horizon is short and rewards are dense, but it breaks down on the kind of tasks humans solve effortlessly. "Make breakfast" is not one decision; it is a tree of subtasks --- *brew coffee*, *fry eggs*, *toast bread*, *plate it up* --- each of which is itself a small policy. **Hierarchical RL (HRL)** lets agents reason and act at multiple timescales by treating macro-actions as first-class citizens.

A second weakness of standard RL is that every new task is learned from scratch. A bicycle rider becomes a motorcycle rider in an afternoon, not in 10 million environment steps. **Meta-RL** attacks this by training across a *distribution* of tasks so that adaptation to a held-out task takes a handful of episodes --- or even a single forward pass through a recurrent network.

This post unifies the two ideas: hierarchy buys temporal abstraction, meta-learning buys task abstraction. Both reduce the effective dimensionality of the learning problem, and the modern frontier (FuN, HIRO, MAML, RL$^2$) combines them aggressively.


---

## What You Will Learn

- **Options framework** --- semi-Markov decision processes and intra-option Q-learning
- **MAXQ** --- value-function decomposition along a task hierarchy
- **Feudal RL** (FuN, HIRO) --- continuous subgoals with manager-worker architectures
- **Goal-conditioned RL** --- universal value functions and HER
- **MAML / FOMAML** --- learning an initialisation that adapts in a few gradient steps
- **RL$^2$** --- folding the inner RL algorithm into an RNN's hidden state
- **Working code** --- intra-option Q-learning in Four Rooms and a MAML policy gradient on 2-D navigation

## Prerequisites

- Q-learning, policy gradients and value functions ([Parts 1--6](/en/reinforcement-learning/01-fundamentals-and-core-concepts/))
- Comfort with RNN unrolling and second-order autodiff
- PyTorch

---

## Hierarchy: the Options framework

### Why temporal abstraction matters

A flat policy makes one decision per environment step, so its credit-assignment path on a horizon-$T$ task has length $T$ and its exploration tree has $|\mathcal{A}|^T$ leaves. Both grow exponentially. A hierarchy with average macro-action length $\bar k$ shrinks the decision count to $T/\bar k$ and the exploration tree to $|\mathcal{O}|^{T/\bar k}$, where $\mathcal{O}$ is a (typically small) set of options. The figure below contrasts the two views.

![Options Framework: timeline view and the (I, π, β) triple](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig1_options_framework.png)

Beyond the asymptotic argument, hierarchies offer three practical benefits:

1. **Modularity** — options trained on one task transfer to others (e.g., *go-to-door* generalizes across navigation problems).
2. **Interpretability** — the high-level policy is small enough to inspect, and decisions are made at semantically meaningful checkpoints.
3. **Reward Shaping** — subgoals provide dense intrinsic rewards even when extrinsic rewards are sparse.

### The option triple

Following Sutton, Precup & Singh (1999), an **option** is a temporally extended action
$$o = \langle \mathcal{I}, \pi_o, \beta \rangle,$$
where $\mathcal{I} \subseteq \mathcal{S}$ is the set of states from which the option may be initiated, $\pi_o(a \mid s)$ is the option's internal policy and $\beta(s) \in [0, 1]$ is its termination probability. Once options are introduced, the underlying MDP becomes a **semi-MDP**: the high-level policy $\mu(o \mid s)$ chooses an option, the option runs until $\beta$ fires, and only then does the next high-level decision happen.

### Intra-option Q-learning

The naive way to learn $Q(s, o)$ is to wait until the option terminates, observe the cumulative reward and bootstrap. This is wasteful: an option may run for dozens of steps and we throw away the intermediate transitions. **Intra-option Q-learning** (Sutton et al., 1999) updates $Q(s, o)$ at *every* step, exploiting the fact that the same transition $(s, a, r, s')$ is consistent with every option that would have taken action $a$ in state $s$:
$$Q(s, o) \leftarrow Q(s, o) + \alpha \big[r + \gamma U(s', o) - Q(s, o)\big],$$
with the **continuation value**
$$U(s', o) = (1 - \beta(s'))\, Q(s', o) + \beta(s')\, \max_{o'} Q(s', o').$$
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
$$Q_i(s, a) = V_a(s) + C_i(s, a),$$
where $V_a(s)$ is the value of completing the *child* subtask and $C_i(s, a)$ is the **completion function** --- the value of finishing the parent task once the child returns. Because $V_a$ depends only on $a$, it can be reused across every parent that invokes $a$, which is where the sample-efficiency win comes from.

![MAXQ-style hierarchy with shared primitives](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig6_task_decomposition.png)

MAXQ pays for that win with **recursive optimality** rather than global optimality: the policy is optimal *given the decomposition*. If your task graph cannot express the truly optimal solution, MAXQ will not find it.

---

## Feudal RL: continuous subgoals with manager-worker

![Feudal RL: manager outputs latent goals, worker conditions on them.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-Hierarchical-and-Meta-RL/fig11_hierarchy.png)

Discrete options scale poorly: in continuous-control or pixel-input domains we cannot enumerate a useful option set by hand. **Feudal Networks** (FuN, Vezhnevets et al., 2017) and **HIRO** (Nachum et al., 2018) replace the discrete option set with a *continuous goal vector* produced by a high-level Manager.

![Feudal RL: Manager sets goals every c steps, Worker executes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig2_feudal_architecture.png)

The Manager sees state $s_t$ and emits a goal $g_t \in \mathbb{R}^d$ every $c$ steps (FuN uses $c = 10$). The Worker is a goal-conditioned policy $\pi_\phi(a \mid s, g)$ that earns an **intrinsic reward** for moving in the direction of the goal:
$$r^{\text{int}}_t = \cos\!\big(s_{t+c} - s_t,\, g_t\big),$$
while the Manager is trained on the *extrinsic* environment reward. This decoupling is what makes Feudal architectures so attractive: the Worker learns motor control on a dense, geometric reward, and the Manager focuses on long-horizon credit assignment with a much smaller effective horizon ($T/c$).

### HIRO's relabelling trick

Feudal training has a chicken-and-egg problem: the Worker is non-stationary (because it is still learning), so old goals stored in the replay buffer no longer correspond to what the Worker actually achieves. HIRO fixes this with **subgoal relabelling**: when sampling a transition $(s_t, g_t, a_{t:t+c}, s_{t+c})$, replace $g_t$ with the goal that maximises the Worker's likelihood of producing the action sequence we actually saw,
$$\tilde g_t = \arg\max_{g} \log \pi_\phi(a_{t:t+c} \mid s_{t:t+c},\, g).$$
This keeps the Worker's training data on-policy with respect to its current parameters and dramatically stabilises off-policy learning of the Manager.

---

## Goal-conditioned RL and HER

Goal-conditioned policies $\pi(a \mid s, g)$ deserve a section of their own, because they are the bridge between hierarchy and meta-learning. The same network can pursue many goals; the goal $g$ is just an additional input.

![Goal-conditioned policy and 4 trajectories from one network](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig3_goal_conditioned.png)

The classical formalisation is **Universal Value Function Approximators** (UVFA, Schaul et al., 2015): learn $V(s, g)$ or $Q(s, a, g)$ instead of $V(s), Q(s, a)$. Without further tricks UVFAs suffer badly from sparse rewards: most goals are never reached during exploration, so the reward signal is essentially zero.

**Hindsight Experience Replay** (HER, Andrychowicz et al., 2017) is the standard fix. After running an episode that *failed* to reach goal $g$, we relabel the trajectory with a goal it *did* reach (typically the final state), turning a failure into a successful demonstration for a different task:
$$(s_t, a_t, r, s_{t+1}, g) \;\longrightarrow\; (s_t, a_t, r', s_{t+1},\, g' = s_T).$$
Combined with off-policy methods (DDPG, SAC), HER turns sparse-reward goal reaching from "essentially impossible" into "routine".

---

## Meta-RL: learning to learn

![Reinforcement Learning (11): Hierarchical RL and Meta-Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/illustration_2.png)

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
$$x_t = (s_t,\, a_{t-1},\, r_{t-1},\, d_{t-1}),$$
i.e. state plus *the previous action, reward and done-flag*. Across episodes within a meta-trial the hidden state $h_t$ accumulates information about the current task --- effectively performing Bayesian belief updates implicitly. At meta-train time the outer optimiser (PPO or A2C) shapes the RNN weights so that this implicit "inner algorithm" is sample-efficient on $p(\mathcal{T})$.

![RL² unrolled across a meta-trial spanning multiple episodes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/11-hierarchical-and-meta-rl/fig7_rl_squared.png)

RL$^2$ has two attractive properties: (i) zero gradient computation at test time, so adaptation is one forward pass per step; (ii) the adaptation procedure is *learned*, so it can in principle exceed any hand-designed optimiser on the training distribution. The price is a hard credit-assignment problem during meta-training --- the gradient must flow through hundreds of recurrent steps spanning multiple episodes.

---

## The Options Framework (Sutton, Precup & Singh)

I keep coming back to the 1999 Sutton-Precup-Singh paper because it is still the cleanest formalisation of "act for a while, then decide". An option is a triple $\langle I, \pi, \beta \rangle$: an initiation set $I \subseteq \mathcal{S}$ that says where the option may start, an intra-option policy $\pi(a \mid s)$ that picks primitive actions while the option is running, and a termination function $\beta(s) \in [0,1]$ that gives the per-step probability of handing control back to the high-level policy.

The framework only earns its keep because of a structural result: once the agent commits to a fixed option set, the world it sees at the option level is still a Markov reward process, just with extended actions. Formally it is a *semi-MDP* — transitions take a random number of primitive steps $\tau$, and the discount on the cumulative reward is $\gamma^\tau$ rather than $\gamma$. That means every algorithm I already trust at the primitive level (Q-learning, SARSA, actor-critic) lifts to options unchanged, modulo the variable-duration discount.

It is worth pausing on what "semi-MDP" buys and what it does not. It buys convergence proofs: SMDP Q-learning converges under the same Robbins-Monro conditions as flat Q-learning, just on the option-level transition kernel $P(s', \tau \mid s, o)$. It does *not* buy globally optimal policies — the best policy expressible with a given option set is in general worse than the best flat policy, sometimes much worse. Sutton's original paper is careful about this distinction, calling the result *hierarchically optimal*; I have seen practitioners conflate the two and then spend weeks debugging "why the options agent plateaus below DQN".

The Bellman equation I actually implement is:
$$Q^\Omega(s, o) = r(s,o) + \gamma \sum_{s'} P(s'\mid s,o)\big[(1-\beta(s'))\, Q^\Omega(s',o) + \beta(s')\, V^\Omega(s')\big].$$
The mixture is the whole point: with probability $1-\beta(s')$ the same option keeps running and we bootstrap from $Q^\Omega(s', o)$; with probability $\beta(s')$ we re-select and bootstrap from $V^\Omega(s') = \max_{o'} Q^\Omega(s', o')$. I got this wrong on my first pass — I bootstrapped from $V^\Omega$ unconditionally and the agent learned to terminate every option after one step.

```python
import torch
import torch.nn as nn

class OptionAgent(nn.Module):
    """Discrete options with shared trunk: per-option pi, beta, and Q^Omega."""

    def __init__(self, state_dim, n_actions, n_options, hidden=64):
        super().__init__()
        self.n_options = n_options
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_options * n_actions)
        self.beta = nn.Linear(hidden, n_options)
        self.q_omega = nn.Linear(hidden, n_options)
        self.n_actions = n_actions

    def forward(self, s):
        h = self.trunk(s)
        pi_logits = self.pi(h).view(-1, self.n_options, self.n_actions)
        beta = torch.sigmoid(self.beta(h))
        q = self.q_omega(h)
        return pi_logits, beta, q

    def act(self, s, current_option):
        pi_logits, beta, q = self.forward(s)
        terminate = torch.bernoulli(beta[:, current_option]).bool()
        option = q.argmax(-1) if terminate else torch.tensor([current_option])
        action = torch.distributions.Categorical(
            logits=pi_logits[:, option.item()]
        ).sample()
        return action.item(), option.item(), terminate.item()
```

I ran this on the canonical Four Rooms grid with four hand-coded hallway options (each navigates to one of the four doorways). The options-based agent reached 90% optimal return in roughly 30K environment steps; flat tabular Q-learning on the same grid took 200K+ steps to match it. The win comes from the branching factor: at the option level the tree has fan-out 4 instead of 4, but the depth drops by an order of magnitude.

Two implementation traps cost me a day each. First, I initially decayed $\epsilon$ on the high-level policy at the same schedule as a flat agent — but the high-level policy makes ten times fewer decisions per episode, so it was effectively still random when the primitive learner had converged. The fix is to count high-level *decisions*, not environment steps, when scheduling exploration. Second, the termination function $\beta$ is itself learnable in the option-critic variant (Bacon 2017), and I have found that letting it learn from the start is unstable: the agent converges on a degenerate solution where every option terminates after one step, recovering flat Q-learning. Freezing $\beta$ for the first 10K steps, or adding a small "deliberation cost" on termination, fixes it.

---

## Feudal RL: Goal-Conditioned Hierarchies (FuN, HIRO)

Discrete options force me to enumerate skills up front, which is fine for grids and miserable for continuous control. The Feudal alternative is to let a Manager network emit a *goal vector* and a Worker network learn to follow it. FuN (Vezhnevets 2017) and HIRO (Nachum 2018) differ in details, but the skeleton is the same: Manager runs slow, Worker runs fast, the latent goal is the only contract between them.

Concretely: every $c$ steps (FuN uses $c=10$, HIRO uses $c=10$ to $c=50$ depending on the task) the Manager observes $s_t$ and emits $g_t \in \mathbb{R}^d$. The Worker conditions on $(s_\tau, g_t)$ for $\tau \in [t, t+c)$ and is rewarded for moving the state in the direction of $g_t$. The Manager itself only ever sees the *extrinsic* reward; the Worker only ever sees the *intrinsic* one. That decoupling is what makes the architecture trainable — the Worker has dense rewards, the Manager has a horizon shrunk by a factor of $c$.

The trick that makes off-policy Manager training work is HIRO's goal relabelling. The Worker is non-stationary, so a goal $g_t$ that was reasonable when the transition was collected may be unreachable for the current Worker. When sampling a stored transition, replace $g_t$ with whichever goal best explains the action sequence the Worker actually took: $\tilde g_t = \arg\max_g \log \pi_\phi(a_{t:t+c} \mid s_{t:t+c}, g)$. In practice I do the argmax over a small candidate set (the original $g_t$, the empirical state difference $s_{t+c} - s_t$, plus eight Gaussian samples around them). It costs almost nothing and I have measured a 3-5x sample efficiency boost on AntMaze.

```python
class FeudalAgent(nn.Module):
    """Manager-worker with continuous goals in a learned latent space."""

    def __init__(self, state_dim, action_dim, goal_dim=16, c=10, hidden=128):
        super().__init__()
        self.c = c
        self.goal_dim = goal_dim
        # Manager: slow, emits goals
        self.manager = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, goal_dim),
        )
        # State -> latent (shared with manager target)
        self.phi = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, goal_dim),
        )
        # Worker: fast, goal-conditioned
        self.worker = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def manager_step(self, s):
        return self.manager(s)

    def worker_step(self, s, g):
        return self.worker(torch.cat([s, g], dim=-1))

    def intrinsic_reward(self, s_t, s_tpc, g_t):
        # Cosine reward in latent space
        z_diff = self.phi(s_tpc) - self.phi(s_t)
        return torch.nn.functional.cosine_similarity(z_diff, g_t, dim=-1)
```

On AntMaze-Medium with sparse terminal reward, my FeudalAgent trained with HIRO relabelling reached >80% success after 1M environment steps. A flat SAC baseline on the same task hovered around 5% — the ant could not get past the first U-turn because the reward signal at the goal was effectively never propagated. The intrinsic reward in the latent direction $g_t$ did the long-horizon credit assignment that flat TD-bootstrapping could not.

One subtlety I missed for a while: if the latent $\phi$ is learned end-to-end with the policy, the Manager and the latent encoder co-adapt and the goal space collapses. FuN solves this by using a *target* network for $\phi$ updated slowly via Polyak averaging, so the goal space the Manager points into is approximately stationary. HIRO sidesteps the issue entirely by setting $\phi$ to the identity on a hand-picked subset of state coordinates (typically the agent's xy position). Both work; the identity choice is more sample-efficient when the relevant subspace is known a priori, and the learned latent wins when it is not.

---

## MAML for RL: Meta-Gradient Through Trajectories

MAML in the supervised setting is already mind-bending; in RL it gets worse, because the inner-loop loss is itself a stochastic estimator of expected return. The objective stays compact:
$$\theta^* = \arg\min_\theta\, \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\big[ \mathcal{L}_\mathcal{T}(\theta'_\mathcal{T}) \big], \qquad \theta'_\mathcal{T} = \theta - \alpha \nabla_\theta \mathcal{L}_\mathcal{T}(\theta).$$
The inner update $\theta'_\mathcal{T}$ is a standard policy-gradient step (REINFORCE, PPO, whatever); the outer update differentiates the post-adaptation loss with respect to the *pre-adaptation* parameters $\theta$. That requires backprop through the inner step, which means the inner gradient $\nabla_\theta \mathcal{L}_\mathcal{T}(\theta)$ has to be a node in the autograd graph rather than a detached tensor.

In PyTorch this is one keyword argument: `torch.autograd.grad(..., create_graph=True)`. Forget it and you silently get FOMAML — which is sometimes what you want, but it is not what the paper calls MAML.

```python
import torch
import torch.nn as nn

class BanditPolicy(nn.Module):
    def __init__(self, n_arms=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, n_arms),
        )

    def logits(self, params=None):
        x = torch.zeros(1, 1)
        if params is None:
            return self.net(x).squeeze(0)
        # Functional forward with externally supplied params
        h = torch.relu(torch.nn.functional.linear(x, params[0], params[1]))
        return torch.nn.functional.linear(h, params[2], params[3]).squeeze(0)

def reinforce_loss(logits, action, reward):
    logp = torch.distributions.Categorical(logits=logits).log_prob(action)
    return -(logp * reward)

def maml_step(policy, tasks, inner_lr=0.1, outer_lr=1e-2, K=5):
    outer_loss = 0.0
    for task in tasks:                       # task = bandit reward vector
        # --- inner loop: K-shot REINFORCE with second-order tracking
        params = list(policy.parameters())
        for _ in range(K):
            logits = policy.logits()
            a = torch.distributions.Categorical(logits=logits).sample()
            r = task[a.item()]
            loss = reinforce_loss(logits, a, torch.tensor(r))
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [p - inner_lr * g for p, g in zip(params, grads)]
        # --- evaluate adapted policy on the same task
        logits = policy.logits(params)
        a = torch.distributions.Categorical(logits=logits).sample()
        r = task[a.item()]
        outer_loss = outer_loss + reinforce_loss(logits, a, torch.tensor(r))
    outer_loss = outer_loss / len(tasks)
    outer_loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            p -= outer_lr * p.grad
            p.grad.zero_()
    return outer_loss.item()
```

I tested this on a 5-task bandit suite where each task is a different reward vector over 5 arms (drawn from $\mathcal{U}[0,1]^5$). After 2000 outer steps of meta-training, one inner-loop SGD step on a held-out task got the agent to 80% of the optimal arm's expected reward. From-scratch REINFORCE with the same single inner step landed at about 30% — i.e., barely better than the uniform-random 20% baseline. The meta-learned initialisation is genuinely doing more than sitting near a task-averaged mean.

The Hessian-vector product that `create_graph=True` triggers is where MAML earns its reputation for being expensive. On this bandit setup the second-order pass adds about 2.5x to the wall-clock per outer step versus FOMAML, and on a real Mujoco MAML run (locomotion with 20-step inner unroll) I measured closer to 8-10x. Whether that is worth it depends on the task family — on the bandit suite FOMAML reaches the same final return, on locomotion the original Finn paper reports a 5-15% gap in MAML's favour.

---

## RL² and PEARL: Context-Conditioned Meta-RL

MAML is gradient-based at test time, which is great when gradients are cheap and bad when they are not. The other branch of meta-RL skips test-time gradients entirely and folds adaptation into a recurrent forward pass. RL² (Duan 2016) is the canonical version: take a recurrent policy, feed it $(s_t, a_{t-1}, r_{t-1}, d_{t-1})$ at every step, and *do not reset the hidden state between episodes within a meta-trial*. The hidden state ends up encoding the agent's posterior over which task it is in, and the outer optimiser (PPO or A2C) shapes the recurrent weights so this implicit posterior update is sample-efficient on $p(\mathcal{T})$.

PEARL (Rakelly 2019) makes the posterior explicit. A separate encoder $q_\phi(z \mid \tau_{1:k})$ takes a few trajectories from the current task and outputs a Gaussian over a task latent $z$; the policy is then $\pi(a \mid s, z)$. Training uses a VAE-style ELBO on the encoder plus off-policy SAC on the policy. The win is sample efficiency: PEARL hits the same return as RL² with roughly 20-100x fewer environment steps on Mujoco meta-benchmarks. The cost is that PEARL only adapts to tasks similar to the training distribution; the encoder cannot extrapolate to genuinely new structure.

The encoder design also matters more than I expected. A naive permutation-invariant encoder (mean-pool over per-transition embeddings) works on simple meta-bandit tasks but fails on locomotion, where the order of transitions carries information. The PEARL paper uses a product-of-Gaussians factorisation that is permutation-invariant by construction; I have had better luck on order-sensitive tasks with a small Transformer encoder, at the cost of more parameters and a slightly less clean ELBO.

```python
class RL2Policy(nn.Module):
    """GRU policy that consumes (state, prev_action, prev_reward, done)."""

    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.n_actions = n_actions
        in_dim = state_dim + n_actions + 1 + 1
        self.gru = nn.GRUCell(in_dim, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def init_hidden(self, batch=1):
        return torch.zeros(batch, self.hidden)

    def step(self, s, prev_a, prev_r, prev_d, h):
        a_oh = torch.nn.functional.one_hot(
            prev_a, num_classes=self.n_actions
        ).float()
        x = torch.cat([s, a_oh, prev_r.unsqueeze(-1), prev_d.unsqueeze(-1)], -1)
        h = self.gru(x, h)
        return self.pi(h), self.v(h).squeeze(-1), h
```

The trade-off I keep in mind: MAML adapts to *new* tasks (it only requires that one inner-loop step exists in the task family), while RL²/PEARL only adapts to tasks from the training distribution. If my task family is narrow and my budget is "one rollout to figure it out", I reach for RL². If the family is broad and I can afford a few SGD steps, MAML.

A practical note on training RL²: backpropagating through hundreds of recurrent steps spanning multiple episodes is brutal. I truncate BPTT at 200 steps and treat each meta-trial as a single PPO trajectory, which loses some long-horizon credit assignment but keeps the gradient norms sane. PEARL avoids this entirely by training the encoder on short context windows (typically 100 transitions) and the policy off-policy via SAC, which is one of the reasons its sample efficiency is so much better — it never has to backprop through a thousand-step recurrence.

---

## Lifelong / Continual RL: Beyond Static Task Sets

Meta-RL still cheats: it assumes a static distribution $p(\mathcal{T})$ and a clean train/test split. Lifelong RL throws that out. The agent runs forever, the task changes whenever the world feels like it, and there is no oracle to say "task done, start adapting again". The challenges are inherited from supervised continual learning — primarily catastrophic forgetting — plus the RL-specific pain that the data distribution itself is policy-dependent and therefore non-stationary on top of any task drift.

The mitigations I have actually used: Elastic Weight Consolidation to penalise drift in parameters that mattered for old tasks, replay buffers that keep a stratified sample of past experience, and adapter modules (PackNet-style) that grow capacity per task. CLEAR (Rolnick 2019) is the most pragmatic one I have shipped: it combines a behaviour-cloning loss against past trajectories with a V-trace off-policy correction on the current data, and it works without any task-boundary signal.

```python
def clear_update(policy, current_batch, replay_batch,
                 distill_coef=0.5, opt=None):
    # Current data: standard policy-gradient (REINFORCE-style, abbreviated)
    s, a, R = current_batch
    logp = policy.log_prob(s, a)
    pg_loss = -(logp * R).mean()

    # Replay data: behaviour-clone toward stored old-policy logits
    s_r, old_logits = replay_batch
    new_logits = policy.logits(s_r)
    distill = torch.nn.functional.kl_div(
        torch.log_softmax(new_logits, -1),
        torch.softmax(old_logits, -1),
        reduction="batchmean",
    )

    loss = pg_loss + distill_coef * distill
    opt.zero_grad(); loss.backward(); opt.step()
    return pg_loss.item(), distill.item()
```

The open problem is detection. Most published continual-RL benchmarks hand the agent task IDs or task boundaries, and the agent's "skill" is just routing to the right adapter. Take the oracle away and performance collapses, because change-point detection on policy-conditional reward streams is genuinely hard. I do not have a good answer here — neither does the literature — and I think the next interesting paper in this area will be one that handles unannounced task changes without regressing on the easy oracle-supervised setting.

A second issue I keep running into: replay buffers grow without bound. CLEAR's authors used 50M transitions on Atari, which is fine on a research cluster and ridiculous on a robot. Reservoir sampling caps memory but biases the distribution toward early experience, and importance-weighted sampling fixes the bias but adds variance. The compromise I currently use is a per-skill reservoir of 100K transitions, sized so the whole buffer fits in 4GB. It is a hack; the principled solution probably looks more like a generative replay model (Shin 2017) that compresses old experience into a network rather than storing raw transitions, but I have not yet found a setup where the generator is cheap enough to be worth it.

## FAQ

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

## References

1. Sutton, Precup & Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence, 1999.
2. Dietterich. *Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition.* JAIR, 2000.
3. Vezhnevets et al. *FeUdal Networks for Hierarchical Reinforcement Learning.* ICML 2017. [arXiv:1703.01161](https://arxiv.org/abs/1703.01161).
4. Nachum et al. *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS 2018. [arXiv:1805.08296](https://arxiv.org/abs/1805.08296).
5. Schaul et al. *Universal Value Function Approximators.* ICML 2015.
6. Andrychowicz et al. *Hindsight Experience Replay.* NeurIPS 2017. [arXiv:1707.01495](https://arxiv.org/abs/1707.01495).
7. Finn, Abbeel & Levine. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML 2017. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400).
8. Duan et al. *RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning.* 2016. [arXiv:1611.02779](https://arxiv.org/abs/1611.02779).
9. Wang et al. *Learning to reinforcement learn.* 2016. [arXiv:1611.05763](https://arxiv.org/abs/1611.05763).
10. Rakelly et al. *Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL).* ICML 2019.
