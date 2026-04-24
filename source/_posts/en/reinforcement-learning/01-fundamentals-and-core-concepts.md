---
title: "Reinforcement Learning (1): Fundamentals and Core Concepts"
date: 2025-06-15 09:00:00
tags:
  - Reinforcement Learning
  - MDP
  - Bellman Equation
  - Dynamic Programming
  - Q-Learning
  - Temporal Difference Learning
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 1
  total: 12
lang: en
mathjax: true
description: "A beginner-friendly guide to the mathematical foundations of reinforcement learning -- MDPs, Bellman equations, dynamic programming, Monte Carlo methods, and temporal difference learning -- with working Python code, all explained through the analogy of learning to ride a bicycle."
disableNunjucks: true
---

The first time you sat on a bicycle, nobody handed you a manual that said *"if your tilt angle exceeds 7.4 degrees, apply 12% counter-steer."* You wobbled, you over-corrected, you fell, you got back on. After a few hundred attempts your body simply *knew* what to do, even though you could not put it into words.

That trial-feedback-improvement loop is not just how we learn to ride bikes. It is how AlphaGo learned to defeat the world Go champion, how Boston Dynamics robots learn to walk, and how recommendation systems quietly improve every time you click. They all share one mathematical framework called **reinforcement learning** (RL).

This article builds RL from the ground up. We will use the bicycle as our running analogy and translate every piece of intuition into the math that powers modern algorithms.

## What You Will Learn

- The **Markov Decision Process** (MDP) -- the mathematical skeleton of every RL problem
- **Bellman equations** and why they make value functions tractable
- **Dynamic programming** for environments where the rules are known
- **Monte Carlo methods** for learning purely from experience
- **Temporal difference (TD) learning** -- the bridge between DP and MC that powers DQN, PPO, and beyond
- Working **Python implementations** you can run on your laptop

**Prerequisites:** Basic probability and a little Python. Familiarity with supervised learning helps but is not required.

---

## The Bicycle Loop

![Agent and environment in a closed feedback loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig1_agent_environment_loop.png)

Picture yourself on a bicycle for the first time. At every instant, three things are happening in a tight loop:

1. **You observe**: tilt angle, speed, where the curb is.
2. **You act**: lean a little, steer a little, pedal a little.
3. **The world responds**: the bike either steadies, drifts further off balance, or drops you on the pavement.

That third step gives you a *signal* -- a small reward when you stay upright, a sharp punishment when you fall. Over many trials, your brain assembles a **policy**: a mapping from "what I am sensing right now" to "what I should do next." The policy is never written down; it is etched into your reflexes by the loop itself.

Reinforcement learning is the mathematical formalism of this loop. The "you" in the diagram is called the **agent**; the bicycle plus road is the **environment**; the lean and steer commands are **actions**; the tilt and speed readouts are **states**; and the don't-fall feeling is the **reward**. Everything else in this article is just careful bookkeeping on top of this picture.

---

## Markov Decision Process: The Mathematical Foundation

![A small MDP with three states for the bicycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig2_mdp_framework.png)

Formally, the bicycle loop is a **Markov Decision Process (MDP)**, a five-tuple

$$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle.$$

The figure above shows a deliberately tiny MDP: just three states (*Balanced*, *Wobbling*, *Fallen*), a few actions, and the transition probabilities and rewards labelled directly on the edges. Every real RL problem -- robot control, Go, large-language-model fine-tuning -- is a (much larger) instance of this same structure.

### The Five Components

**State space** $\mathcal{S}$: every situation the agent can find itself in. For a bicycle: tilt angle, angular velocity, forward speed, road curvature. States can be discrete (board positions in chess) or continuous (joint angles of a robot).

**Action space** $\mathcal{A}$: everything the agent can do. Discrete (`{lean-left, lean-right, hold}`) or continuous (apply a torque of 0.37 Nm to the handlebars).

**Transition probability** $P(s' \mid s, a)$: the probability of landing in state $s'$ after taking action $a$ in state $s$. This is the environment's *physics*:

$$P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a),\qquad \sum_{s'} P(s' \mid s, a) = 1.$$

A perfectly balanced bicycle is *not* a deterministic system: a gust of wind, a pebble, or a slightly uneven pedal stroke can each push you to a different next state. The transition probability captures all of that uncertainty.

**Reward function** $R(s, a, s')$: the immediate payoff for the transition $s \xrightarrow{a} s'$. Rewards are the agent's **only** learning signal. Get them wrong and the agent will obediently optimise the wrong thing -- a phenomenon known as *reward hacking*.

**Discount factor** $\gamma \in [0, 1)$: how much the agent cares about *future* reward versus *immediate* reward. We will see this is far more than a numerical convenience.

### The Markov Property

The defining assumption of an MDP is delightfully simple: **the future depends only on the present, not on how you got here.**

$$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = P(S_{t+1} \mid S_t, A_t).$$

For the bicycle this looks suspicious -- surely *which way I was leaning a moment ago* matters? It does, but the trick is to *fold history into the state itself*. If we redefine the state as `(tilt, angular velocity, speed)` instead of just `tilt`, the Markov property holds again. In Atari, DeepMind famously stacked the last 4 frames into the state for exactly this reason.

### Policy: From States to Actions

A **policy** $\pi$ is the agent's strategy -- the function that turns observations into decisions:

- **Deterministic**: $a = \pi(s)$. Always the same action for the same state.
- **Stochastic**: $\pi(a \mid s) = \Pr(A_t = a \mid S_t = s)$. A probability distribution over actions.

Stochastic policies matter for two reasons: they let the agent **explore** new behaviours, and they are the natural output of neural networks (a softmax over actions).

### Return and Value Functions

The agent's goal is not to maximise *the next* reward but the *cumulative discounted return* from time $t$:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}.$$

Why discount? Three independent reasons all point the same way:

- **Mathematical**: when $|r| \le R_{\max}$ the geometric sum stays finite, $|G_t| \le R_{\max} / (1 - \gamma)$. Without it, infinite-horizon tasks would blow up.
- **Cognitive**: a reward today is worth more than the same reward tomorrow -- there is uncertainty between you and tomorrow.
- **Operational**: without discount, an agent could rationally do *nothing forever* and still claim infinite return. Discounting forces it to *get on with it*.

We then define two value functions, one for states and one for state-action pairs:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].$$

In words: $V^\pi(s)$ is "how good is it to be here, if I follow $\pi$?", and $Q^\pi(s, a)$ is "how good is it to take *this specific action* here, then follow $\pi$?". They are linked by

$$V^\pi(s) = \sum_a \pi(a \mid s) \, Q^\pi(s, a),\qquad Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V^\pi(s')\right].$$

### Bellman Equations: The Recursive Heart of RL

![Bellman backup tree: today's value built from tomorrow's](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig3_bellman_recursion.png)

Value functions have a beautiful recursive structure. This is the single most important idea in RL theory -- once it clicks, every algorithm in the rest of the series will feel like a variation on a theme.

**Bellman expectation equation** (for any policy $\pi$):

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V^\pi(s')\right].$$

Read it out loud: *the value of being here equals the expected immediate reward plus the discounted value of where I land next*. The tree in the figure is exactly this equation drawn out -- the root is the current state, the middle layer is the actions weighted by $\pi$, the leaves are the next states weighted by $P$, and the rewards live on the arrows.

**Bellman optimality equation** (for the best possible policy $\pi^*$):

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V^*(s')\right],$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right].$$

The change is small but everything: the expectation over $\pi$ is replaced by a $\max$. Once you have $Q^*$, the optimal policy falls out trivially:

$$\pi^*(s) = \arg\max_{a} Q^*(s, a).$$

### A Numerical Example

Let us pin this down with a small two-state MDP, $\{s_1, s_2\}$, one action $a_1$ that we always pick, and $\gamma = 0.9$.

| State | Action | Next State | Prob | Reward |
|-------|--------|------------|------|--------|
| $s_1$ | $a_1$ | $s_1$ | 0.5 | 5 |
| $s_1$ | $a_1$ | $s_2$ | 0.5 | 10 |
| $s_2$ | $a_1$ | $s_1$ | 0.7 | 2 |
| $s_2$ | $a_1$ | $s_2$ | 0.3 | 8 |

Plug into Bellman:

$$V(s_1) = 0.5\,[5 + 0.9 V(s_1)] + 0.5\,[10 + 0.9 V(s_2)],$$

$$V(s_2) = 0.7\,[2 + 0.9 V(s_1)] + 0.3\,[8 + 0.9 V(s_2)].$$

Rearranging gives a linear system

$$0.55\,V(s_1) - 0.45\,V(s_2) = 7.5,\qquad -0.63\,V(s_1) + 0.73\,V(s_2) = 3.8,$$

with solution $V(s_1) \approx 52.3$ and $V(s_2) \approx 50.4$. The values are large because the rewards keep coming forever and $\gamma$ is close to 1; that is the geometric-series effect at work.

---

## Dynamic Programming: When You Know the Rules

When the environment model ($P$ and $R$) is fully known, **dynamic programming (DP)** computes the optimal policy *exactly*. There are no samples, no noise -- just deterministic iteration on the Bellman equation.

### Policy Evaluation

Given a fixed policy $\pi$, repeatedly apply the Bellman expectation operator:

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V_k(s')\right].$$

Start with $V_0 \equiv 0$ and iterate. The Bellman operator is a $\gamma$-contraction in the sup-norm, so $V_k \to V^\pi$ exponentially fast.

### Policy Improvement

Given $V^\pi$, build a *greedier* policy:

$$\pi'(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V^\pi(s')\right].$$

The **policy improvement theorem** guarantees $V^{\pi'}(s) \ge V^\pi(s)$ for every state. You never get worse by being greedier with respect to a correct value function.

### Policy Iteration

Alternate between the two:

1. Start with any policy $\pi_0$.
2. **Evaluate**: compute $V^{\pi_k}$.
3. **Improve**: build the greedy $\pi_{k+1}$.
4. If $\pi_{k+1} = \pi_k$, stop -- we have hit a fixed point, which is the optimal policy.

### Value Iteration

Why fully evaluate before improving? Skip it. Iterate the optimality equation directly:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a)\!\left[R(s, a, s') + \gamma V_k(s')\right].$$

Each sweep contracts the error by a factor of $\gamma$. The figure below shows what the converged value function and the resulting greedy policy look like on a small grid world.

![Value heatmap and greedy policy on GridWorld](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig5_value_vs_policy.png)

The two panels are inseparable halves of the same answer: the heatmap tells you *how good is here?*, and the arrows tell you *where should I go next?*. The arrows always point uphill on the heatmap, which is exactly what "greedy with respect to $V$" means.

### Code: GridWorld with Value Iteration

```python
import numpy as np


class GridWorld:
    """Agent moves from start to goal; obstacles are impassable."""

    def __init__(self, grid_size=(5, 5), start=(0, 0),
                 goal=(4, 4), obstacles=None):
        self.height, self.width = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.actions = [0, 1, 2, 3]                        # U, D, L, R
        self.action_names = ['U', 'D', 'L', 'R']
        self.action_deltas = {0: (-1, 0), 1: (1, 0),
                              2: (0, -1), 3: (0, 1)}
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.action_deltas[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        if (0 <= nr < self.height and 0 <= nc < self.width
                and (nr, nc) not in self.obstacles):
            self.state = (nr, nc)
        if self.state == self.goal:
            return self.state, 10.0, True
        return self.state, -1.0, False

    def get_transitions(self, state, action):
        """Deterministic transitions, returned as [(next, prob, reward)]."""
        old = self.state
        self.state = state
        ns, r, _ = self.step(action)
        self.state = old
        return [(ns, 1.0, r)]


def value_iteration(env, gamma=0.9, theta=1e-6):
    """Solve the Bellman optimality equation by repeated max-backups."""
    V = np.zeros((env.height, env.width))
    for iteration in range(1000):
        delta = 0
        for r in range(env.height):
            for c in range(env.width):
                s = (r, c)
                if s == env.goal or s in env.obstacles:
                    continue
                old_v = V[r, c]
                V[r, c] = max(
                    sum(p * (rew + gamma * V[ns])
                        for ns, p, rew in env.get_transitions(s, a))
                    for a in env.actions
                )
                delta = max(delta, abs(old_v - V[r, c]))
        if delta < theta:
            print(f"Converged after {iteration + 1} sweeps")
            break

    policy = np.zeros((env.height, env.width), dtype=int)
    for r in range(env.height):
        for c in range(env.width):
            s = (r, c)
            if s == env.goal or s in env.obstacles:
                continue
            q = [sum(p * (rew + gamma * V[ns])
                     for ns, p, rew in env.get_transitions(s, a))
                 for a in env.actions]
            policy[r, c] = int(np.argmax(q))
    return V, policy


env = GridWorld(grid_size=(5, 5), start=(0, 0), goal=(4, 4),
                obstacles=[(1, 1), (2, 2), (3, 1)])
V, policy = value_iteration(env)
print(np.round(V, 1))
```

DP is exact and elegant, but it has two crippling limitations. First, it requires the model: in real life we rarely know $P$ and $R$ in closed form. Second, it sweeps over the entire state space on every iteration -- impossible when the state space is the set of all $84 \times 84$ Atari frames. The next two sections fix both issues.

---

## Monte Carlo Methods: Learning Without a Model

When we do not know $P$ and $R$, we must learn from experience. The most direct way is also the oldest in statistics: **average a lot of samples**.

### Core Idea

A value function is, by definition, an *expected* return. Replace the expectation with a sample mean:

$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)},$$

where $G_t^{(i)}$ is the return observed in the $i$-th episode that passed through $s$. As $N \to \infty$ this is unbiased and consistent.

The recipe is brutally simple:

1. Run the policy from start to terminal state, recording $s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T$.
2. For each state visited, compute the return from that point onwards: $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$.
3. Update the running mean.

### First-Visit vs Every-Visit MC

Within a single episode, the same state may appear multiple times. We have two options:

- **First-visit MC**: only use the return from the *first* occurrence.
- **Every-visit MC**: use returns from *every* occurrence.

Both converge to $V^\pi$ as the number of episodes grows. First-visit is the easier of the two to prove unbiased and is the default choice in textbooks.

### MC Control: Finding Better Policies

For *control* (i.e. searching for the optimal policy), we estimate $Q(s, a)$ instead of $V(s)$ and improve greedily. To make sure every state-action pair gets visited, we use an **$\varepsilon$-greedy** policy:

$$\pi(a \mid s) = \begin{cases}
1 - \varepsilon + \varepsilon / |\mathcal{A}| & \text{if } a = \arg\max_{a'} Q(s, a'), \\
\varepsilon / |\mathcal{A}| & \text{otherwise.}
\end{cases}$$

That tiny $\varepsilon$ is the engine of *exploration* -- without it, the agent might lock onto a mediocre action and never discover the better one.

### Code: MC Policy Evaluation and Control

```python
import numpy as np
from collections import defaultdict


def mc_policy_evaluation(env, policy_fn, num_episodes=10000, gamma=0.9):
    """First-Visit Monte Carlo policy evaluation."""
    returns = defaultdict(list)
    V = defaultdict(float)
    for _ in range(num_episodes):
        trajectory, state, done = [], env.reset(), False
        while not done:
            action = policy_fn(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        G, visited = 0.0, set()
        for state, _, reward in reversed(trajectory):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                returns[state].append(G)
                V[state] = float(np.mean(returns[state]))
    return V


def mc_control(env, num_episodes=10000, gamma=0.9, epsilon=0.1):
    """On-policy MC control with epsilon-greedy exploration."""
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    returns = defaultdict(list)
    for _ in range(num_episodes):
        trajectory, state, done = [], env.reset(), False
        while not done:
            if np.random.random() < epsilon:
                action = int(np.random.choice(env.actions))
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        G, visited = 0.0, set()
        for state, action, reward in reversed(trajectory):
            G = reward + gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = float(np.mean(returns[(state, action)]))
    return Q
```

**Pros**: model-free, conceptually transparent, unbiased.

**Cons**: needs *complete* episodes (so non-terminating tasks are out), high variance because the entire return depends on a long noisy trajectory, and no online updates.

---

## Temporal Difference Learning: The Best of Both Worlds

![Smoothed return curves under three discount factors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig4_episode_rewards.png)

**Temporal difference (TD) learning** is the conceptual centrepiece of modern RL. It combines MC's model-free spirit with DP's bootstrapping in a single line of code, and it is the reason DQN, A3C, PPO, and SAC all exist.

### TD(0): One-Step Updates

The update rule:

$$V(S_t) \leftarrow V(S_t) + \alpha\,\big[\,r_t + \gamma V(S_{t+1}) - V(S_t)\,\big].$$

The bracketed quantity is the **TD error**:

$$\delta_t = r_t + \gamma V(S_{t+1}) - V(S_t).$$

It is the difference between *what just happened* ($r_t + \gamma V(S_{t+1})$) and *what we expected* ($V(S_t)$). The agent nudges its estimate towards reality, by an amount controlled by the learning rate $\alpha$.

Three things are special about this rule:

| Method | Update target | Bias / variance | Online? |
|--------|---------------|-----------------|---------|
| Monte Carlo | $G_t$ (actual return) | Unbiased / high variance | No |
| TD(0) | $r_t + \gamma V(S_{t+1})$ | Biased initially / low variance | Yes |
| DP backup | $\mathbb{E}[r_t + \gamma V(S_{t+1})]$ | No noise, requires model | Yes |

The figure above shows how the choice of $\gamma$ alone reshapes how fast a Q-learning agent (a TD method) climbs out of the early-training swamp on a small grid task. Lower $\gamma$ propagates credit more locally and learns faster initially; higher $\gamma$ is slower to converge but values long-term success more.

### Sarsa: On-Policy TD Control

Sarsa updates $Q$ using the action the agent *actually takes* next:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\,\big[\,r_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\,\big].$$

The name spells out the quintuple it depends on: $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$.

### Q-Learning: Off-Policy TD Control

Q-learning (Watkins, 1989) replaces the actual next action with the *best* next action:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\,\big[\,r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\,\big].$$

The single-character difference -- $A_{t+1}$ versus $\max_{a'}$ -- changes everything. Sarsa evaluates the policy it follows (**on-policy**); Q-learning evaluates the *greedy* policy regardless of what it actually does (**off-policy**). Q-learning can therefore learn the optimal policy *while* exploring randomly.

### Sarsa vs Q-Learning: Cliff Walking

The classic illustration is *cliff walking*:

```
S . . . . . . G
C C C C C C C C
```

The agent starts at `S`, must reach `G`, and falls into the cliff `C` if it ever steps below the top row.

- **Sarsa** *factors in* the risk of an exploratory step pushing it off the cliff. It learns a safer path that hugs the top edge of the grid.
- **Q-learning** evaluates the greedy policy (which never explores), so it learns the optimal path right along the cliff edge -- but during *training* it falls in much more often.

This is the cleanest possible illustration of the on-policy/off-policy trade-off: Sarsa is conservative, Q-learning is asymptotically optimal but more reckless during learning.

### TD($\lambda$) and Eligibility Traces

TD(0) only looks one step ahead. **TD($\lambda$)** blends multi-step returns:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)},$$

where $G_t^{(n)}$ is the $n$-step return. The interpolation parameter $\lambda \in [0, 1]$ lets you smoothly trade between TD(0) ($\lambda = 0$) and Monte Carlo ($\lambda = 1$).

**Eligibility traces** implement this efficiently by maintaining a "credit memory" $e_t(s)$ that decays exponentially:

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}[S_t = s], \qquad V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s)\quad\text{for all } s.$$

Recently visited states get more credit when a reward arrives, propagating information backwards through the entire trajectory in a single step.

### Code: Sarsa and Q-Learning

```python
import numpy as np


def _eps_greedy(Q, state, actions, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.choice(actions))
    return int(np.argmax(Q[state]))


def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Sarsa: on-policy TD control."""
    Q = {(r, c): np.zeros(len(env.actions))
         for r in range(env.height) for c in range(env.width)}
    for _ in range(num_episodes):
        state = env.reset()
        action = _eps_greedy(Q, state, env.actions, epsilon)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = _eps_greedy(Q, next_state, env.actions, epsilon)
            target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (target - Q[state][action])
            state, action = next_state, next_action
    return Q


def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Q-Learning: off-policy TD control."""
    Q = {(r, c): np.zeros(len(env.actions))
         for r in range(env.height) for c in range(env.width)}
    for _ in range(num_episodes):
        state, done = env.reset(), False
        while not done:
            action = _eps_greedy(Q, state, env.actions, epsilon)
            next_state, reward, done = env.step(action)
            target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
    return Q
```

---

## Two Themes That Run Through Everything

Two ideas appear and reappear in every algorithm above. It is worth pulling them out so they are easy to recognise later in the series.

### Exploration vs Exploitation

![Cumulative regret of fixed epsilon policies on a 10-armed bandit](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig6_explore_vs_exploit.png)

Every learning agent faces a permanent dilemma: *use what I know, or test what I don't?* If you only ever pick the action you currently believe is best, you may never discover a better one. If you constantly randomise, you learn a lot but earn very little.

The figure above runs a classic 10-armed bandit experiment: each "arm" is a slot machine with an unknown average payoff, and the agent has 1000 pulls. Cumulative *regret* is the gap between the reward of the best arm and the reward you actually collected.

- **$\varepsilon = 0$** (pure greedy) often locks onto an arm that *seemed* good after a few pulls but isn't actually the best -- regret grows linearly forever.
- **$\varepsilon = 0.3$** keeps wasting pulls on bad arms.
- **$\varepsilon \in [0.01, 0.1]$** sits in the sweet spot.

This same trade-off shows up dressed in different clothes throughout the series: $\varepsilon$-greedy in DQN, entropy bonuses in PPO, intrinsic motivation in Part 4, Thompson sampling in Bayesian RL. The *names* differ, the *problem* doesn't.

### The Discount Factor as a Planning Horizon

![How gamma controls the agent's planning horizon](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/01-fundamentals-and-core-concepts/fig7_discount_effect.png)

The discount factor $\gamma$ is more than a numerical knob. It quietly defines *how far into the future* the agent thinks. Two ways to see it:

- **Reward weighting (left panel)**: a reward $k$ steps away is worth $\gamma^k$ of an immediate reward. With $\gamma = 0.5$, a reward 10 steps out is already worth less than 0.1%; with $\gamma = 0.99$, it still carries 90%.
- **Effective horizon (right panel)**: the rough number of future steps that meaningfully contribute to $G_t$ is $1 / (1 - \gamma)$. So $\gamma = 0.9$ means "I plan about 10 steps ahead," $\gamma = 0.99$ means "100 steps," and $\gamma = 0.999$ means "1000 steps." Notice the y-axis is logarithmic -- pushing $\gamma$ from 0.99 to 0.999 is a *tenfold* expansion of the planning horizon, not a 1% tweak.

Practical consequence: $\gamma$ should match the time scale of your task. Game-playing agents often use $\gamma \approx 0.99$. Recommendation systems that care about decade-long lifetime value might push it higher. Robot reflex controllers might use $\gamma = 0.9$ or lower because anything more than a second is irrelevant.

---

## Choosing the Right Method

A quick decision guide:

| Situation | Recommended method | Why |
|-----------|--------------------|-----|
| Model is known | Dynamic programming | Exact, no sampling needed |
| Short episodes, no model | Monte Carlo | Unbiased, conceptually simple |
| Long or continuing tasks | TD (Q-learning / Sarsa) | Online, low variance |
| Need optimal policy | Q-learning | Off-policy, converges to $Q^*$ |
| Safety matters | Sarsa | Factors exploration risk into learning |
| Long credit-assignment chains | TD($\lambda$) / Sarsa($\lambda$) | Eligibility traces propagate fast |

**Three sentences to memorise**:

> **Value = immediate reward + discounted future value.**
>
> **Policy improvement = greedily prefer high-value actions.**
>
> **Learning = use experience to correct value estimates.**

---

## Summary

This chapter built the foundation that the rest of the series stands on:

- **MDPs** formalise the agent-environment loop: states, actions, transitions, rewards, discount.
- **Bellman equations** give value functions a recursive structure that turns long-horizon planning into local arithmetic.
- **Dynamic programming** solves an MDP exactly when the model is known.
- **Monte Carlo methods** learn from complete episodes without a model.
- **Temporal difference methods** combine the best of both worlds: model-free *and* online.
- **Exploration vs exploitation** and the **discount factor** are the two hidden levers that show up in every algorithm to come.

All of these methods assume small, discrete state and action spaces where you can store one value per cell of a table. When state spaces explode -- think $256^{84 \times 84}$ Atari frames -- tabular methods break down completely.

**Next up:** [Part 2](/en/reinforcement-learning-2-q-learning-and-dqn/) introduces **Deep Q-Networks (DQN)** -- using neural networks to approximate $Q$, with experience replay and target networks to keep training stable. That is the bridge from textbook RL to the algorithms that beat humans at Atari.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9-44.
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Silver, D. (2015). UCL Course on Reinforcement Learning.

---

## Series Navigation

| Part | Topic |
|------|-------|
| **1** | **Fundamentals and Core Concepts (you are here)** |
| 2 | [Q-Learning and DQN](/en/reinforcement-learning-2-q-learning-and-dqn/) |
| 3 | [Policy Gradient and Actor-Critic](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) |
| 4 | [Exploration and Curiosity-Driven Learning](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) |
| 5 | [Model-Based RL and World Models](/en/reinforcement-learning-5-model-based-rl-and-world-models/) |
| 6 | [PPO and TRPO](/en/reinforcement-learning-6-ppo-and-trpo/) |
