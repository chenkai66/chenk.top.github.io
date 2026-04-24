---
title: "Reinforcement Learning (2): Q-Learning and Deep Q-Networks (DQN)"
date: 2025-06-19 09:00:00
tags:
  - Reinforcement Learning
  - Q-Learning
  - DQN
  - Deep Q-Network
  - Experience Replay
  - Target Network
  - Rainbow
  - Value-Based Methods
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 2
  total: 12
lang: en
mathjax: true
description: "How DQN combined neural networks with Q-Learning to master Atari games -- covering experience replay, target networks, Double DQN, Dueling DQN, Prioritized Experience Replay, and Rainbow."
disableNunjucks: true
series_order: 2
---

In December 2013, a small DeepMind team uploaded a paper to arXiv with a striking claim: a single neural network, trained from raw pixels and the score, learned to play seven Atari games -- and beat the previous best on six of them. No game-specific features. No hand-coded heuristics. The same architecture for Pong, Breakout, and Space Invaders. The algorithm was **Deep Q-Network (DQN)**, and it kicked off the deep reinforcement learning era.

DQN was not invented from scratch. It is **Q-Learning** -- a 1989 tabular algorithm by Chris Watkins -- with a neural network in place of the lookup table, plus two engineering tricks that keep training from blowing up. This article explains exactly which problems those tricks solve, walks through a complete PyTorch implementation, and surveys the variants that turned DQN from an Atari demo into a workhorse of modern RL.

## What you will learn

- Why tabular Q-Learning collapses in high-dimensional state spaces
- The **Deadly Triad** -- the three ingredients that make naive deep RL diverge
- DQN's two innovations: **experience replay** and **target networks** -- and which Triad failure each one fixes
- A complete, runnable Atari **DQN agent** in PyTorch
- The post-DQN family: **Double DQN**, **Dueling DQN**, **Prioritized Experience Replay**, and **Rainbow**

**Prerequisites:** [Part 1](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) for MDPs, the Bellman equations, and the temporal-difference (TD) intuition.

---

## Q-Learning Foundations

### The Bellman optimality equation, reread

Recall from Part 1 that for the optimal policy $\pi^*$, the action-value function satisfies the Bellman optimality equation:

$$
Q^*(s, a) \;=\; \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\Big]
$$

Read it as a contract: the value of taking action $a$ in state $s$ today equals the immediate reward plus the discounted best future you can achieve from $s'$. Once you know $Q^*$, the optimal policy is just $\pi^*(s) = \arg\max_a Q^*(s, a)$ -- no planning, no search, just a table lookup.

So the entire problem reduces to: **estimate $Q^*$**. Q-Learning is one way.

### The Q-Learning update rule

After taking action $A_t$ in state $S_t$, observing reward $r_t$ and next state $S_{t+1}$, Q-Learning (Watkins, 1989) updates the table entry for the visited cell:

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha \underbrace{\Big[r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big]}_{\text{TD error } \delta_t}
$$

The bracketed quantity is the **TD error** -- the gap between a one-step bootstrapped estimate and the current value. Positive $\delta_t$ means we underestimated and should nudge up; negative means the opposite.

Two properties make Q-Learning special:

1. **Off-policy.** The target uses $\max_{a'}$ -- the greedy action -- regardless of which action the agent actually picks next. The behaviour policy (typically epsilon-greedy) can explore freely; the target policy stays greedy. This decoupling is the source of Q-Learning's power and, later, its instability.
2. **Convergence guarantee.** Watkins and Dayan (1992) proved that if every state-action pair is visited infinitely often and the learning rate satisfies the Robbins-Monro conditions ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$), then $Q(s, a) \to Q^*(s, a)$ with probability one.

A trained Q-table is a small but informative object. The figure below shows one for a 4x4 gridworld with a goal, a pit, and a -0.04 step cost. Each cell carries four numbers (one per action); the arrow marks the greedy choice that you would extract via $\arg\max_a Q(s, a)$.

![Q-Table on a 4x4 Gridworld -- four Q-values and the greedy arrow per cell](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig1_qtable_gridworld.png)

### A worked example: Cliff Walking

Sutton & Barto's Cliff Walking is the standard sanity check for Q-Learning. The agent walks on a 4 x 12 grid; the bottom row between start and goal is a cliff that costs -100 and resets the episode. Every other step costs -1, so the optimal route is a one-step-above-the-cliff shortcut with return -13.

```python
import numpy as np


class CliffWalkingEnv:
    """4x12 grid. Start: (3,0), Goal: (3,11), Cliff: (3,1)-(3,10)."""

    def __init__(self):
        self.height, self.width = 4, 12
        self.start, self.goal = (3, 0), (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0:   r = max(0, r - 1)       # up
        elif action == 1: c = min(11, c + 1)      # right
        elif action == 2: r = min(3, r + 1)       # down
        elif action == 3: c = max(0, c - 1)       # left

        ns = (r, c)
        if ns in self.cliff:
            self.state = self.start
            return self.start, -100, False
        if ns == self.goal:
            self.state = ns
            return ns, 0, True
        self.state = ns
        return ns, -1, False


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {(i, j): np.zeros(4) for i in range(4) for j in range(12)}
    history = []
    for _ in range(episodes):
        state, total, steps = env.reset(), 0, 0
        while steps < 1000:
            action = (np.random.randint(4) if np.random.rand() < epsilon
                      else int(np.argmax(Q[state])))
            ns, reward, done = env.step(action)
            total += reward
            # Q-Learning update -- target uses the GREEDY next action
            td_target = reward + gamma * np.max(Q[ns])
            Q[state][action] += alpha * (td_target - Q[state][action])
            state = ns
            steps += 1
            if done:
                break
        history.append(total)
    return Q, history
```

The interesting thing is what happens when you sweep $\varepsilon$. Too greedy and the agent never finds the optimal route. Too exploratory and it keeps falling off the cliff.

![Q-Learning on Cliff Walking under three exploration rates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig2_cliff_walking_curves.png)

The 0.10 curve climbs fast and stabilises near the optimum. The 0.50 curve plateaus far from optimal because the agent keeps choosing random (often fatal) actions. The 0.01 curve eventually converges but takes longer because exploration is so sparse early on. This trade-off recurs everywhere in RL.

---

## Why Tables Are Not Enough

Cliff Walking has 48 states -- a Q-table is 192 floats. Now consider Atari Breakout. The DQN paper preprocesses the screen into 84 x 84 grayscale frames and stacks the last four to give the agent a sense of motion. The state is a vector in $\{0, \ldots, 255\}^{84 \times 84 \times 4}$, a space of size $256^{28224}$. There are not enough atoms in the universe.

Two failures follow immediately:

1. **Storage is impossible.** No table can hold a Q-value per state.
2. **Visit count is zero.** Q-Learning's convergence theorem requires every state-action pair to be visited infinitely often. The agent will never see the *exact* same 84x84x4 array twice.

The fix is to replace the table with a **parametric function approximator**: a function $Q(s, a; \theta)$ with parameters $\theta$ -- in DQN, a convolutional network -- that *generalises* across visually similar states. Two screens that differ only in noise or brick colour should map to similar Q-values.

### The Deadly Triad

Generalisation comes at a price. Sutton & Barto identify three properties that, **together**, can make a value-based RL algorithm diverge. They call this the **Deadly Triad**:

1. **Bootstrapping.** The TD target $r + \gamma \max_{a'} Q(s', a'; \theta)$ uses the network's own current estimate as ground truth.
2. **Function approximation.** Updating $\theta$ to fix the Q-value at one state shifts the Q-values at every nearby state too -- you cannot edit one cell of a "neural table" in isolation.
3. **Off-policy data.** The transitions used for updates do not come from the policy whose value we are trying to learn.

Each pair is fine; tabular Q-Learning is off-policy + bootstrapping but has no function approximation. Linear function approximation + on-policy is fine. But all three at once is genuinely unsafe -- you can construct toy MDPs where Q-values diverge to infinity.

DQN's two innovations target the off-policy and bootstrapping legs of the Triad respectively.

---

## DQN's Core Innovations

### Experience replay: breaking temporal correlation

In supervised learning, mini-batch SGD assumes the data are i.i.d. -- you shuffle and sample. In RL, consecutive transitions $(s_t, a_t, r_t, s_{t+1})$ and $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ are wildly correlated: the next state is literally a function of the current one. Train on this stream directly and the gradient updates oscillate, the network forgets old experiences as soon as it leaves them, and convergence stalls.

DQN's fix is the **replay buffer**: store every observed transition in a large FIFO queue (typically 1 million entries) and, at each gradient step, draw a random mini-batch from it.

![Experience replay buffer: stream in, sample random mini-batches out](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig4_replay_buffer.png)

This buys three benefits:

- **Decorrelated mini-batches.** A random sample of 32 transitions is approximately i.i.d., satisfying the assumption SGD relies on.
- **Sample efficiency.** Each transition is replayed many times -- DeepMind reports roughly an order-of-magnitude improvement in sample efficiency.
- **Smoother distribution shift.** The buffer mixes data from many recent policies, so the training distribution evolves slowly even when the policy changes quickly.

```python
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s_next), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
```

### Target network: stop chasing your own tail

Even with replay, the TD target $r + \gamma \max_{a'} Q(s', a'; \theta)$ moves every time we update $\theta$ -- the target is a function of the parameters we are optimising. The geometry of the loss surface changes underfoot at every step. Empirically, this causes oscillation and divergence.

DQN keeps **two copies** of the network:

- The **online network** $Q(\cdot; \theta)$, updated every step by gradient descent.
- The **target network** $Q(\cdot; \theta^-)$, a frozen copy that the online network is periodically reset to.

The loss is computed against the frozen copy:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big]
$$

Every $C$ steps (10,000 in the Nature paper), copy: $\theta^- \leftarrow \theta$. Between copies the target is *constant*, turning each $C$-step window into a near-supervised regression problem.

![Target network: a lagging copy of the online weights stabilises the TD target](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig5_target_network.png)

A continuous variant -- the **Polyak (soft) update** $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$ with $\tau \approx 0.005$ -- is used by DDPG and SAC and works equally well.

---

## A Complete Atari DQN

The DQN architecture is small by 2026 standards: three convolutional layers and two fully-connected layers, around 1.7M parameters. The convolutions look at local patches of the 84x84 screen; the FC layers map the flattened feature map to one Q-value per action.

![DQN architecture: 4 stacked frames pass through three conv layers, an FC layer, and a per-action output head](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig3_dqn_architecture.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class DQN(nn.Module):
    """Nature-DQN architecture. Input: (batch, 4, 84, 84) uint8 frames /255."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> (B, 32, 20, 20)
        x = F.relu(self.conv2(x))   # -> (B, 64, 9, 9)
        x = F.relu(self.conv3(x))   # -> (B, 64, 7, 7)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, n_actions: int, device: str = "cuda"):
        self.n_actions, self.device = n_actions, device

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=2.5e-4)
        self.memory = ReplayBuffer(capacity=100_000)

        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon, self.epsilon_end, self.epsilon_decay = 1.0, 0.01, 0.995
        self.target_update_freq = 10_000
        self.steps_done = 0

    def select_action(self, state, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.policy_net(x).argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s_next, d = self.memory.sample(self.batch_size)
        s        = torch.as_tensor(s,      dtype=torch.float32, device=self.device)
        a        = torch.as_tensor(a,      dtype=torch.long,    device=self.device)
        r        = torch.as_tensor(r,      dtype=torch.float32, device=self.device)
        s_next   = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        d        = torch.as_tensor(d,      dtype=torch.float32, device=self.device)

        # Q(s, a) from the online network
        q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Bootstrap target from the FROZEN network -- no gradient flows here
        with torch.no_grad():
            q_next = self.target_net(s_next).max(dim=1).values
            q_target = r + (1.0 - d) * self.gamma * q_next

        # Huber loss is more robust than MSE to occasional large TD errors
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        return loss.item()
```

A few practical notes you will not find in the paper:

- Use **Huber loss** (`smooth_l1_loss`), not pure MSE -- a few wild TD errors will not throw the optimiser if the gradient is clipped to magnitude 1.
- **Gradient clipping** to 10 (or norm 1) prevents the rare large update from destabilising everything.
- **Frame-skip 4 with max-pool over the last two frames** handles Atari's flickering sprites; this is environment preprocessing, not part of the agent.
- On Pong, this agent reaches near-optimal play (mean episode reward $\approx +21$) in roughly 200-300 episodes on a single GPU.

---

## DQN Variants: From Double to Rainbow

The original DQN was published in 2013 (workshop) and 2015 (*Nature*). In the years that followed, a procession of papers each plugged one specific weakness. Eventually they were assembled into a single agent -- Rainbow -- whose components compose almost additively.

### Double DQN: removing the maximisation bias

Even with a perfect target network, the $\max_{a'}$ operator induces a systematic upward bias. The reason is short and elegant. Suppose the true Q-values are $Q^*(s, a)$ and the network's estimates are $Q^*(s, a) + \varepsilon_a$ with zero-mean noise. Then:

$$
\mathbb{E}\Big[\max_a \big(Q^*(s, a) + \varepsilon_a\big)\Big] \;\geq\; \max_a Q^*(s, a)
$$

The max preferentially picks whichever action got a *positive* error. Through bootstrapping, this overestimation seeps into every other state's target. Empirically, vanilla DQN's predicted Q-values drift far above the true return.

van Hasselt et al. (2016) proposed **Double DQN**: decouple the action *selection* from the action *evaluation*. The online network picks the next action; the target network evaluates it.

$$
y_t = r_t + \gamma\, Q\big(s_{t+1},\; \arg\max_{a'} Q(s_{t+1}, a'; \theta);\; \theta^-\big)
$$

Errors in the two networks are partially independent, so the systematic bias largely cancels.

![Double DQN tracks the true Q-value; vanilla DQN inflates it](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig6_double_vs_vanilla.png)

The code change is two lines:

```python
with torch.no_grad():
    # SELECT next action with the ONLINE network
    next_actions = self.policy_net(s_next).argmax(dim=1, keepdim=True)
    # EVALUATE that action with the TARGET network
    q_next = self.target_net(s_next).gather(1, next_actions).squeeze(1)
    q_target = r + (1.0 - d) * self.gamma * q_next
```

### Dueling DQN: separating "how good is here?" from "how good is this action?"

In a great many Atari frames, the choice of action barely matters -- the ball is far away, the enemies are off-screen, you cannot influence anything for a few frames. Lumping state value and action advantage into one Q-head wastes capacity learning the same $V(s)$ over and over.

Wang et al. (2016) factor the head into two streams:

$$
Q(s, a) \;=\; V(s) + \Big(A(s, a) - \tfrac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Big)
$$

- $V(s)$ -- "How good is it to be in this state?" -- is action-independent.
- $A(s, a)$ -- "How much better is action $a$ than average?" -- is action-specific.

The mean-subtraction trick is what makes the decomposition identifiable; without it $V$ and $A$ could absorb arbitrary constants from each other.

```python
class DuelingDQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.value_fc, self.value_head = nn.Linear(64 * 7 * 7, 512), nn.Linear(512, 1)
        self.adv_fc,   self.adv_head   = nn.Linear(64 * 7 * 7, 512), nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        v = self.value_head(F.relu(self.value_fc(x)))      # (B, 1)
        a = self.adv_head(F.relu(self.adv_fc(x)))          # (B, |A|)
        return v + a - a.mean(dim=1, keepdim=True)
```

Dueling pairs especially well with Double DQN -- the two changes attack different weaknesses and stack cleanly.

### Prioritized Experience Replay

Uniform sampling treats every transition as equally informative, but that is plainly false: a transition where the prediction was already correct teaches almost nothing, while one with a large TD error contains a strong learning signal.

Prioritized Experience Replay (Schaul et al., 2016) samples transition $i$ with probability proportional to a power of its absolute TD error:

$$
p_i \;\propto\; \big(|\delta_i| + \varepsilon\big)^\alpha
$$

The exponent $\alpha \in [0, 1]$ trades off prioritisation strength ($\alpha = 0$ recovers uniform sampling). Because non-uniform sampling biases the gradient, each sample is reweighted by an importance-sampling correction:

$$
w_i = \Big(\tfrac{1}{N \cdot p_i}\Big)^\beta
$$

with $\beta$ annealed from 0.4 toward 1.0 across training -- low $\beta$ early when the bias hurts less, full correction late when stability matters more.

### Multi-step targets, distributional RL, NoisyNets

Three more pieces complete Rainbow:

- **n-step returns** ($n=3$ in Rainbow) replace the one-step bootstrap with a partial Monte-Carlo target, reducing bias at the cost of variance:

  $$
  y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)
  $$

- **Distributional RL (C51)**, from Bellemare et al. (2017), learns the entire distribution of returns rather than just the mean. The categorical projection onto a fixed support of 51 atoms gives the method its name. The full distribution carries information that the mean discards, especially in stochastic environments.

- **NoisyNet** replaces epsilon-greedy with parameterised noise injected directly into the FC layer weights. Exploration is thus state-dependent and learned, rather than a hand-tuned schedule.

### Rainbow

Hessel et al. (2018) put all six together (DQN base + Double + Dueling + PER + n-step + distributional + NoisyNet) and ran a careful ablation. Every single component contributed; their combination beat each individual variant by a clear margin.

![Median human-normalised score on the Atari-57 benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/fig7_atari_benchmark.png)

The numbers above are median scores normalised so that the human baseline = 100% and a random agent = 0%. DQN alone is below human. Each subsequent improvement adds tens of percentage points; Rainbow more than doubles human performance on the median game.

---

## Practical Notes

### A starter hyperparameter table

| Hyperparameter        | Typical value          | Comment                                         |
| --------------------- | ---------------------- | ----------------------------------------------- |
| Replay capacity       | 100K - 1M transitions  | Larger is better, RAM-bound                     |
| Mini-batch size       | 32 - 128               | 32 is the Nature default                        |
| Learning rate         | 1e-4 to 3e-4           | Adam, 2.5e-4 is a safe starting point           |
| Discount $\gamma$     | 0.99                   | Lower for short horizons                        |
| $\varepsilon$ schedule| 1.0 -> 0.01            | Linear over ~1M frames                          |
| Target update period $C$ | 10K steps           | Or soft update with $\tau \approx 0.005$        |
| Frame skip            | 4                      | Atari preprocessing                             |
| Gradient clipping     | norm 10                | Use Huber loss as well                          |

### A debugging checklist

1. **Solve CartPole first.** A correctly implemented DQN solves CartPole in under 200 episodes on CPU. If yours cannot, the bug is in your code, not your hyperparameters.
2. **Watch the Q-value magnitudes.** They should rise and then plateau near a sensible scale (for Atari, single or double digits). If they drift past 1000 you have a divergence bug.
3. **Track the TD error.** It should fall and stay bounded. A monotonic upward trend means the target is escaping.
4. **Inspect the action distribution.** If the agent only ever takes one action, the network's output is collapsing -- check for dead ReLUs, broken initialisation, or a runaway learning rate.

### When DQN, when something else?

| Aspect              | DQN family                       | Policy gradient (PPO, SAC, ...) |
| ------------------- | -------------------------------- | ------------------------------- |
| Action space        | Discrete only                    | Discrete and continuous         |
| Sample efficiency   | High (replay reuses data)        | Lower (PPO is on-policy)        |
| Stability           | Needs the target network + tricks| Generally easier to tune        |
| Best fit            | Atari, board games, discrete control | Robotics, locomotion, continuous control |

DQN's hard limitation is the discrete action space: $\arg\max_a Q(s, a)$ is trivial over five buttons but infeasible over a real-valued joint angle. The next article picks up exactly there.

---

## Summary

DQN's contribution was as much engineering as it was theory. The neural network in place of the table is the obvious change; the experience replay buffer and the target network are the two ideas that turn an unstable algorithm into one that actually trains. Together they neutralise the off-policy and bootstrapping legs of the Deadly Triad enough to get useful learning out of a non-linear function approximator.

The post-DQN variants each add a clean fix to a clean failure: Double DQN removes overestimation bias, Dueling separates state value from action advantage, Prioritized Replay focuses learning on surprising transitions, and Rainbow shows that these fixes compose. The same building blocks survive into modern agents -- you will see replay buffers and target networks in SAC, in offline RL, and in many recent LLM-RL hybrids.

**Next:** [Part 3](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) introduces **policy gradient** methods and **actor-critic** architectures -- the family that handles continuous actions and underlies PPO, SAC, and the policy half of every modern algorithm.

---

## References

- Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. *NIPS Deep Learning Workshop*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep RL with Double Q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling Network Architectures for Deep RL. *ICML*.
- Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR*.
- Bellemare, M., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
- Fortunato, M., et al. (2018). Noisy Networks for Exploration. *ICLR*.
- Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. -- Chapter 11 for the Deadly Triad.

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Core Concepts](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) |
| **2** | **Q-Learning and DQN (you are here)** |
| 3 | [Policy Gradient and Actor-Critic](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) |
| 4 | [Exploration and Curiosity-Driven Learning](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) |
| 5 | [Model-Based RL and World Models](/en/reinforcement-learning-5-model-based-rl-and-world-models/) |
| 6 | [PPO and TRPO](/en/reinforcement-learning-6-ppo-and-trpo/) |
