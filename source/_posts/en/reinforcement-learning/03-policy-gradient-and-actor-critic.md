---
title: "Reinforcement Learning (3): Policy Gradient and Actor-Critic Methods"
date: 2024-07-03 09:00:00
tags:
  - Reinforcement Learning
  - Policy Gradient
  - REINFORCE
  - Actor-Critic
  - A3C
  - DDPG
  - TD3
  - SAC
  - PPO
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 3
  total: 12
lang: en
mathjax: true
description: "From REINFORCE to SAC -- how policy gradient methods directly optimize policies, naturally handle continuous actions, and power modern algorithms like PPO, TD3, and SAC."
disableNunjucks: true
---

DQN proved that deep RL can master Atari, but it has a hard ceiling: it only works in **discrete action spaces**. Ask it to control a robot arm with seven continuous joint angles and it falls apart -- you would have to solve an inner optimisation problem every time you choose an action.

**Policy gradient methods** take a fundamentally different route. Instead of learning a value function and *deriving* a policy from it, they **directly optimise the policy**. That single change opens the door to continuous actions, stochastic strategies, and problems where the optimal play is itself random (think rock-paper-scissors).

## What You Will Learn

- Why policy gradients exist, and what the **Policy Gradient Theorem** actually says
- **REINFORCE**: the simplest policy-gradient algorithm and why its variance is so painful
- The **Actor-Critic** architecture and how the **advantage function** $A = Q - V$ shrinks variance
- **GAE($\lambda$)**: one knob for the bias-variance trade-off
- **DDPG / TD3 / SAC** for continuous control
- A practical **algorithm-selection guide** grounded in current industrial usage

**Prerequisites:** [Part 1](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) (MDPs, value functions, TD learning) and [Part 2](/en/reinforcement-learning-2-q-learning-and-dqn/) (DQN, target networks, replay buffers).

---

## 1. Why Policy Gradients?

DQN learns $Q(s,a)$ and acts greedily: $\pi(s) = \arg\max_a Q(s,a)$. That indirect recipe creates four pain points:

1. **Discrete actions only.** Computing $\arg\max$ over a continuous space is itself a non-trivial optimisation, repeated at every environment step.
2. **No stochastic policies.** Greedy policies are deterministic. But in matching-pennies-style games the **optimal** policy is genuinely random.
3. **Error amplification.** Q-value approximation errors get amplified by the $\max$ operator -- the overestimation bias we fixed (partially) with Double DQN in Part 2.
4. **Ad-hoc exploration.** $\epsilon$-greedy is a hack: it has no principled reason for the noise it injects.

Policy gradient methods sidestep all four by **parameterising the policy directly** as $\pi_\theta(a|s)$:

- For **discrete** actions: a linear layer followed by a softmax produces a categorical distribution.
- For **continuous** actions: the network outputs the parameters of a Gaussian (mean $\mu_\theta(s)$ and log-std $\log\sigma_\theta(s)$), often squashed through $\tanh$ to bound the action.

![Discrete categorical vs continuous tanh-squashed Gaussian policy heads](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig6_action_policies.png)

Either way, the loss machinery is identical: pick an action by sampling from $\pi_\theta$, then push the parameters in a direction that makes good actions more likely.

### 1.1 The Policy Gradient Theorem

Let $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$ be the expected return of the trajectories produced by $\pi_\theta$. We want $\nabla_\theta J(\theta)$ so we can do gradient ascent.

The **Policy Gradient Theorem** (Sutton et al., 2000) gives a clean, sample-able form:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\Big[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\;\cdot\;Q^{\pi_\theta}(s_t, a_t)\Big]
$$

Three things are worth pausing on:

- $\nabla_\theta \log \pi_\theta(a|s)$ is called the **score function**. It is the gradient that, on its own, would make action $a$ slightly *more* likely.
- $Q^\pi(s,a)$ acts as a **scalar weight** on that direction: good actions amplify the score, bad actions invert it.
- The **environment dynamics $P(s'|s,a)$ disappear** from the formula. We never need to know them; sampled trajectories are enough. This is what makes the method **model-free**.

Visually, the theorem says "shift probability mass toward actions whose realised return was high":

![Policy gradient as a score-function update on the action distribution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig1_policy_gradient_theorem.png)

The left panel shows $\pi_\theta(a|s)$ before and after one (large, illustrative) update -- mass migrates toward the reward bump. The right panel shows the update direction itself: the score function multiplied by the reward and the current density. Where the product is positive we increase $\pi(a)$; where it is negative we decrease it.

### 1.2 The Variance Problem and the Baseline Trick

The raw estimator above is **unbiased**, but its variance is horrible. $Q^\pi$ can be hundreds or thousands; one lucky episode can shove $\theta$ in any direction.

A simple identity rescues us. For **any function** $b(s)$ that does *not* depend on the action,

$$
\mathbb{E}_{a \sim \pi_\theta}\!\big[\,\nabla_\theta \log \pi_\theta(a|s)\,\cdot\,b(s)\,\big] \;=\; 0,
$$

so subtracting $b(s)$ from $Q$ leaves the gradient unchanged in expectation but can drastically shrink variance. The optimal choice is the state-value function $V^\pi(s)$, which gives us the **advantage function**:

$$
A^\pi(s,a) \;=\; Q^\pi(s,a) - V^\pi(s).
$$

"Advantage" is exactly what it sounds like: how much better is action $a$ than the average action in state $s$? Centred around zero, the gradient signal stops swinging wildly.

![Q(s,a), the value baseline V(s), and the resulting advantage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig4_advantage_decomposition.png)

Green bars on the right are actions worth reinforcing; red bars are actions to suppress. The greedy DQN view ("pick the biggest $Q$") becomes a continuous, signed update: "lift the policy in proportion to how above-average each action turned out to be."

---

## 2. REINFORCE: Monte Carlo Policy Gradient

**REINFORCE** (Williams, 1992) is the textbook starting point. It uses the *actual* discounted return $G_t$ as a Monte Carlo estimate of $Q^\pi(s_t, a_t)$.

### 2.1 Algorithm

1. Roll out one full trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T)$ under $\pi_\theta$.
2. Compute the discounted return for every step: $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$.
3. Estimate the gradient: $\hat g = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,(G_t - b(s_t))$.
4. Ascend: $\theta \leftarrow \theta + \alpha\,\hat g$.

That is the whole algorithm. The simplicity is the point.

### 2.2 REINFORCE with a Learned Baseline on CartPole

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def reinforce_with_baseline(env_name="CartPole-v1", episodes=1000,
                            gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    value = ValueNetwork(state_dim)
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = optim.Adam(value.parameters(), lr=1e-3)

    history = []

    for ep in range(episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []

        while True:
            st = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(st)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value(st))

            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            if done:
                break

        # Discounted returns G_t (Monte-Carlo estimate of Q).
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        log_probs = torch.stack(log_probs)
        values = torch.cat(values)

        # Advantage = G_t - V_phi(s_t)  (baseline subtraction)
        advantages = returns - values.detach()

        # Policy: gradient ascent on log pi * advantage  -> minimise the negation.
        policy_loss = -(log_probs * advantages).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # Value: regress V_phi onto the empirical returns.
        value_loss = F.mse_loss(values, returns)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        history.append(sum(rewards))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1:4d}  avg100 = "
                  f"{np.mean(history[-100:]):6.1f}")

    return policy, history
```

This typically solves CartPole in 100--200 episodes. On harder tasks, though, REINFORCE shows its weakness fast.

**Strengths:** simple, unbiased, action-space agnostic.
**Weaknesses:** very high gradient variance, every trajectory is used exactly once (no off-policy reuse), and updates only happen at episode boundaries.

---

## 3. Actor-Critic: Replacing Returns with TD Estimates

REINFORCE waits until the end of the episode to compute $G_t$. That return contains noise from *every* future state, action, and reward. Can we do better?

**Actor-Critic** says: train a second network -- a **critic** $V_\phi(s)$ -- and use it to bootstrap the gradient signal.

- **Actor** $\pi_\theta(a|s)$: decides what to do.
- **Critic** $V_\phi(s)$: scores how good a state is.

The crucial substitution is the **TD-error advantage**:

$$
\hat A_t \;=\; r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t).
$$

This depends on **only one step** of randomness instead of the entire tail of the trajectory. The variance collapses; in exchange we accept some bias from the imperfect critic.

The two networks usually share a backbone:

![Actor-Critic shared-backbone architecture and update flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig3_actor_critic_architecture.png)

The TD error $\delta_t$ does double duty: it acts as the **advantage** weighting the actor's gradient, and as the **target** for the critic's regression loss.

### 3.1 How Big a Deal Is the Variance Reduction?

A simulated comparison on the same set of trajectories: the orange line uses raw Monte Carlo returns, the blue line uses the advantage produced by a learned baseline.

![Per-step gradient signal: REINFORCE vs A2C](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig2_variance_reduction.png)

Same trajectories, same expected gradient. The right panel shows the practical payoff: variance shrinks by an order of magnitude, which is exactly why training curves of A2C/PPO look so much smoother than REINFORCE.

### 3.2 A2C in Code

```python
class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic."""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)   # policy head
        self.critic = nn.Linear(hidden, 1)            # value head

    def forward(self, state):
        h = self.trunk(state)
        return F.softmax(self.actor(h), dim=-1), self.critic(h).squeeze(-1)
```

A3C (Mnih et al., 2016) parallelised this across asynchronous workers. Modern practice prefers the synchronous version **A2C**: collect rollouts from $N$ environments in lockstep, then take one combined gradient step. Same idea, much friendlier to GPUs.

### 3.3 GAE: A Dial Between TD and Monte Carlo

One-step TD has low variance but high bias; Monte Carlo has the opposite. **Generalised Advantage Estimation** (Schulman et al., 2016) interpolates between them with a single hyperparameter $\lambda \in [0, 1]$:

$$
\hat A_t^{\text{GAE}(\lambda)} \;=\; \sum_{k=0}^{\infty} (\gamma\lambda)^k\,\delta_{t+k},
\quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

$\lambda = 0$ recovers the one-step TD advantage; $\lambda = 1$ recovers the full Monte Carlo return (minus the baseline). Practitioners almost always pick something in $[0.9, 0.97]$.

![GAE bias-variance trade-off and n-step return weighting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig5_gae_lambda_sweep.png)

The left panel shows the qualitative trade-off; the right panel shows what the weighting actually does -- larger $\lambda$ spreads the credit across many future TD errors, smaller $\lambda$ trusts only the immediate one. PPO's defaults ($\lambda = 0.95$, $\gamma = 0.99$) live near the sweet spot.

---

## 4. Continuous Control: DDPG and TD3

For continuous actions like joint torques the policy is naturally Gaussian: $a \sim \mathcal{N}(\mu_\theta(s),\,\sigma_\theta(s))$. But sampling injects noise that hurts precise control. **Deterministic policies** $a = \mu_\theta(s)$ avoid that noise -- and admit a particularly clean gradient.

### 4.1 DDPG: Deep Deterministic Policy Gradient

DDPG (Lillicrap et al., 2016) couples DQN-style stability tricks with an Actor-Critic structure:

- **Replay buffer + target networks** (inherited from DQN).
- The **deterministic policy gradient** (Silver et al., 2014):

$$
\nabla_\theta J \;=\; \mathbb{E}_{s \sim \rho^\beta}\!\Big[\,\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;\nabla_\theta \mu_\theta(s)\,\Big].
$$

Read it from right to left: shift $\theta$ in whatever direction $\mu_\theta(s)$ moves, weighted by how steeply $Q$ rises in $a$ at that point. Pure chain rule.

Exploration is added externally as action noise: $a_t = \mu_\theta(s_t) + \mathcal{N}(0,\sigma)$.

### 4.2 TD3: Three Tricks That Stabilise DDPG

DDPG inherits DQN's overestimation bias and is famously brittle. **TD3** (Fujimoto et al., 2018) fixes it with three independent ideas, each useful on its own:

1. **Clipped double Q-learning.** Train two critics and use the *minimum* of their target predictions: $y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde a')$.
2. **Delayed policy updates.** Update the actor only every $d$ critic updates (typically $d=2$). Lets the critic settle before the actor pulls the rug.
3. **Target policy smoothing.** Add clipped noise to the target action: $\tilde a' = \mu_{\theta'}(s') + \mathrm{clip}(\epsilon, -c, c)$, $\epsilon \sim \mathcal{N}(0, \sigma)$. This forces the critic to be smooth in $a$, so the actor cannot exploit narrow Q-spikes.

```python
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.total_it = 0

    def train(self, replay_buffer, batch_size=256,
              gamma=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_delay=2):
        self.total_it += 1
        state, action, reward, next_state, done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # (3) Target policy smoothing.
            noise = (torch.randn_like(action) * policy_noise) \
                .clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise) \
                .clamp(-self.max_action, self.max_action)

            # (1) Clipped double-Q target.
            target_q = torch.min(
                self.critic_1_target(next_state, next_action),
                self.critic_2_target(next_state, next_action),
            )
            target_q = reward + (1 - done) * gamma * target_q

        critic_loss = (
            F.mse_loss(self.critic_1(state, action), target_q)
            + F.mse_loss(self.critic_2(state, action), target_q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # (2) Delayed actor + target updates.
        if self.total_it % policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            # ... same Polyak averaging for both critics.
```

These three changes turn DDPG from "sometimes works after careful tuning" into a competitive, reproducible baseline.

---

## 5. SAC: Maximum Entropy RL

Even TD3 has a failure mode: the policy can collapse to a narrow distribution and stop exploring. **Soft Actor-Critic** (Haarnoja et al., 2018) attacks the problem at its root by **changing the objective**:

$$
J(\pi) \;=\; \mathbb{E}\!\Big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal H[\pi(\cdot|s_t)]\big)\Big].
$$

The entropy term $\mathcal H[\pi]$ pays the agent for being uncertain. The temperature $\alpha$ controls the trade-off. The Bellman backups are modified to match: the "soft" Q-target adds an expected log-policy term.

Three engineering details make SAC the workhorse it is:

- **Automatic temperature tuning.** $\alpha$ is itself learned by gradient descent against a target-entropy constraint, so you do not have to guess.
- **Stochastic squashed-Gaussian policy.** The actor outputs $(\mu, \log\sigma)$; samples are passed through $\tanh$ and the log-prob is corrected by the change-of-variables Jacobian.
- **Twin critics with clipped double-Q**, off-policy replay, and soft target updates -- the familiar TD3 stack.

In practice, SAC matches or beats TD3 on MuJoCo benchmarks while being noticeably less sensitive to hyperparameters. For continuous control on real hardware, SAC is the default starting point in many labs.

---

## 6. Why Does This Even Work? Climbing a Noisy Hill in $\theta$-Space

It is worth zooming out. Every algorithm in this article is a special case of one idea: **stochastic gradient ascent on $J(\theta)$ in policy parameter space**. The "stochastic" is doing a lot of work -- our gradient estimates are noisy, sometimes wildly so, and the loss surface itself is non-convex.

![Stochastic gradient ascent on the policy return surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/fig7_policy_optimization_landscape.png)

A few things this picture makes obvious:

- The path is **jagged**, not smooth. Variance reduction (baseline, advantage, GAE) buys us a less jagged path, not a different destination.
- **Local plateaus exist.** Entropy regularisation (SAC) and stochastic policies are partly defences against getting stuck on them.
- The landscape itself **changes as $\theta$ changes** -- the data distribution is on-policy. This is why off-policy methods (DDPG, TD3, SAC) need careful corrections, and why PPO clips the update step in the next instalment.

---

## 7. Algorithm Selection Guide

| Situation | Recommended | Why |
|-----------|-------------|-----|
| Discrete actions, fast prototyping | **PPO** | Stable, simple, well-supported by every library |
| Continuous actions, expensive samples | **SAC** or **TD3** | Off-policy, high sample efficiency |
| Continuous actions, abundant simulation | **PPO** | Smoother training curves, easy to scale across environments |
| Need a stochastic policy by design | **SAC** | Maximum-entropy framework gives you one for free |
| Sparse rewards | **SAC** | Entropy keeps the policy exploring long enough to find them |
| Learning the field, building intuition | **REINFORCE -> A2C -> PPO** | Each step adds exactly one new idea on top of the last |

**What major labs actually use, as of 2026:**

- OpenAI (Dota 2, ChatGPT RLHF): **PPO**
- DeepMind (continuous control research): **SAC** and variants
- Berkeley robotics: **SAC** for real-world manipulation
- TD3 remains the standard reference baseline in continuous-control benchmarks

---

## 8. Summary

Policy-gradient methods opened RL to the world beyond discrete actions:

- **REINFORCE** showed that policies can be optimised directly via gradient ascent on expected return -- conceptually clean, practically noisy.
- **Actor-Critic + advantage** traded a touch of bias for an order-of-magnitude variance reduction, making training tractable.
- **GAE($\lambda$)** turned the bias-variance choice into a single tunable knob.
- **DDPG / TD3** brought off-policy efficiency to continuous control with deterministic policies and DQN-style stabilisation.
- **SAC** added entropy regularisation and became the go-to method for continuous control.
- **PPO** -- the subject of [Part 6](/en/reinforcement-learning-6-ppo-and-trpo/) -- simplified trust-region ideas into a clipped surrogate that became the industry workhorse.

All of these methods are **model-free**: they learn from interaction without ever building an explicit model of the environment. That generality is their strength, and the millions of samples they consume is their weakness.

**Next up:** [Part 4](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) tackles the **exploration problem** -- how do agents discover rewards in the first place when the environment provides almost no feedback?

---

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. *ICML*.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *ICML*.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: off-policy maximum entropy deep RL. *ICML*.

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Core Concepts](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) |
| 2 | [Q-Learning and DQN](/en/reinforcement-learning-2-q-learning-and-dqn/) |
| **3** | **Policy Gradient and Actor-Critic (you are here)** |
| 4 | [Exploration and Curiosity-Driven Learning](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) |
| 5 | [Model-Based RL and World Models](/en/reinforcement-learning-5-model-based-rl-and-world-models/) |
| 6 | [PPO and TRPO](/en/reinforcement-learning-6-ppo-and-trpo/) |
