---
title: "Reinforcement Learning (5): Model-Based RL and World Models"
date: 2025-08-21 09:00:00
tags:
  - Reinforcement Learning
  - Model-Based RL
  - World Models
  - Dyna
  - MBPO
  - Dreamer
  - MuZero
  - PlaNet
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 5
  total: 12
lang: en
mathjax: true
description: "From Dyna and MBPO to World Models, Dreamer, and MuZero -- how learning a model lets agents plan in imagination and reach expert performance with 10-100x fewer real interactions."
disableNunjucks: true
series_order: 5
---

Every algorithm we have covered so far -- DQN, REINFORCE, A2C, PPO, SAC -- is **model-free**: the agent treats the environment as a black box, throws actions at it, and updates its policy from the rewards that come back. The approach works, but it is profligate. DQN needs roughly **10 million frames** to master Atari Pong. OpenAI Five trained on Dota 2 for the equivalent of **~45,000 years** of self-play. AlphaStar consumed years of StarCraft for a single agent.

Humans clearly do not learn this way. A chess player imagines positions a few moves deep and prunes obvious blunders; a child learns "cliffs are bad" once, by inference, not by falling. Both rely on an internal **model** of how the world responds to actions, and they spend most of their cognitive budget *in that model*, not in the world.

**Model-Based RL (MBRL)** formalises this idea: learn an approximate dynamics model$\hat{P}(s'\mid s, a)$and reward model$\hat{R}(s, a)$, then use them as a cheap simulator for planning, policy improvement, or value estimation. The payoff, on tasks where it works, is a **10-100x reduction in real-environment samples** -- the difference between a robot that needs three months of physical interaction and one that needs an afternoon.

This article traces the modern lineage:  Dyna (1990) -> MBPO (2019) -> World Models (2018) -> Dreamer (2020-23) -> MuZero (2020). Each method rests on a single sharp idea, and the seven figures in this post visualise those ideas one at a time.

## What You Will Learn

- The exact trade-offs that make model-based RL win or lose
- **Dyna-Q**: the original blueprint for mixing real and imagined updates
- **MBPO**: why short-horizon imagination is the sweet spot
- **MPC** as a pure planning loop with a learned model
- **World Models** (V/M/C): compressing pixels into a latent dream space
- **Dreamer / RSSM**: end-to-end latent imagination with a recurrent + stochastic state
- **MuZero**: planning without ever predicting an observation

**Prerequisites:** [Parts 1-3](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) (MDPs, value functions, policy gradients, Actor-Critic).

---

## 1. Two Paradigms, One Goal

![Model-free vs model-based control loops](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig1_mf_vs_mb_loops.png)

In model-free RL the only loop is *act -> observe -> learn*. In model-based RL we insert a second loop: *learn a model -> plan inside the model -> improve the policy*. Real-world interaction now amortises across thousands of imagined updates.

### The Trade-off

|                    | Model-Free                              | Model-Based                                          |
| ------------------ | --------------------------------------- | ---------------------------------------------------- |
| **What is learnt** | A policy / value function only          | A model $\hat{P},\hat{R}$ **and** a policy / value   |
| **Sample cost**    | High -- each gradient step uses a real interaction | Low -- one real step yields many imagined updates  |
| **Compute cost**   | Lower per step                          | Higher (model fitting + planning)                    |
| **Asymptote**      | Limited only by exploration             | Limited by **model bias**                            |
| **Transfer**       | Tied to the trained reward              | Same model can be reused for new tasks               |
| **Failure mode**   | Slow learning                           | Compounding model error -> hallucinated optima       |

### Sample Efficiency in Practice

| Algorithm     | Family       | Benchmark              | Steps to expert |
| ------------- | ------------ | ---------------------- | --------------- |
| DQN           | Model-free   | Atari Pong             | ~10M frames     |
| PPO           | Model-free   | MuJoCo HalfCheetah     | ~1-2M steps     |
| SAC           | Model-free   | MuJoCo HalfCheetah     | ~600K steps     |
| **MBPO**      | Model-based  | MuJoCo HalfCheetah     | **~80-100K**    |
| **Dreamer**   | Model-based  | DMControl Walker       | **~100K**       |
| **DreamerV3** | Model-based  | Minecraft (diamonds)   | first algorithm to do it from scratch |

The gap is roughly **one order of magnitude** in continuous control and even larger when the simulator is expensive (real robots, slow physics).

### When to Reach for Model-Based

Good fit:

- Real interaction is expensive: robotics, autonomous driving, drug discovery, dialogue systems with users in the loop.
- Dynamics are *learnable* with reasonable data: smooth physics, board games, structured environments.
- You will face **multiple downstream tasks**, so the model amortises across them.

Poor fit:

- A free, fast, high-fidelity simulator already exists (Atari *is* the simulator).
- Dynamics are highly stochastic or adversarial (financial markets, social interaction).
- The state space is so high-dimensional that no model fits in the data budget.

---

## 2. Dyna-Q: The Original Blueprint

![Dyna-Q flow diagram and convergence curves](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig3_dyna_q_flow.png)

Sutton's **Dyna** (1990) is the first system to articulate the model-based loop in its purest form. Each real transition is consumed three times:

1. **Direct learning** -- update Q with the real$(s,a,r,s')$,
2. **Model learning** -- store the transition in a tabular model$M(s,a)\to(r,s')$,
3. **Planning** -- sample$n$previously-seen$(s,a)$pairs, query the model, and apply$n$additional Q-updates from these *imagined* transitions.

The convergence plot on the right shows the consequence on a deterministic GridWorld: increasing$n$from 0 (vanilla Q-Learning) to 50 collapses convergence by an order of magnitude in episodes -- because every real step now triggers 51 Bellman updates instead of 1.

### Reference Implementation

```python
import numpy as np


class DynaQ:
    """Tabular Dyna-Q: real learning + planning from a remembered model."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, planning_steps=10):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}                    # (s, a) -> (r, s')
        self.visited = []                  # ordered list for sampling
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def _q_update(self, s, a, r, s_next):
        target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def learn(self, s, a, r, s_next):
        # 1. direct RL
        self._q_update(s, a, r, s_next)
        # 2. model learning (deterministic env: just memorise)
        if (s, a) not in self.model:
            self.visited.append((s, a))
        self.model[(s, a)] = (r, s_next)
        # 3. planning -- replay imagined transitions
        for _ in range(self.planning_steps):
            sp, ap = self.visited[np.random.randint(len(self.visited))]
            rp, sp_next = self.model[(sp, ap)]
            self._q_update(sp, ap, rp, sp_next)
```

### What Dyna Teaches Us, and Where It Breaks

Dyna isolates the core insight: **a learned model lets you spend compute instead of samples**. The trade-off it surfaces -- and that every modern method inherits -- is that planning on a wrong model injects bias straight into the value function. In tabular deterministic worlds this is invisible; with a neural-network model and a long horizon, errors compound exponentially. The rest of this article is essentially a sequence of clever answers to that one problem.

---

## 3. MBPO: Keep Your Imagination Short

![MBPO short branched rollouts and model error growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig4_mbpo_short_rollouts.png)

**Model-Based Policy Optimization** (Janner et al., NeurIPS 2019) is the cleanest modern instantiation of Dyna for continuous control. Its title insight is in two words: **short rollouts**.

The right-hand panel shows the issue. Cumulative state-prediction error grows roughly geometrically with rollout length$k$. By$k = 20$even an ensemble of 5 dynamics models has drifted far enough to be useless for credit assignment. The MBPO answer is to **branch only 1-5 steps** off real states (left panel), then hand the resulting transitions to SAC for the long-horizon credit assignment that model-free methods do well.

### Algorithm

1. Roll the current policy in the real environment, append to$\mathcal{D}_{\text{real}}$.
2. Fit an **ensemble of 5** probabilistic dynamics models$f_\theta(s,a)\to(s',r)$on$\mathcal{D}_{\text{real}}$.
3. Repeatedly sample initial states from$\mathcal{D}_{\text{real}}$, branch a $k$-step rollout in a randomly chosen ensemble member, and append imagined transitions to$\mathcal{D}_{\text{model}}$.
4. Train SAC on a mixture of$\mathcal{D}_{\text{real}}$and$\mathcal{D}_{\text{model}}$.

The ensemble matters: averaging or random sampling across members both *regularises* predictions (they disagree most where data is sparse) and supplies *epistemic uncertainty* the policy can implicitly avoid.

### Sketch

```python
import numpy as np
import torch
import torch.nn as nn


class EnsembleDynamics(nn.Module):
    """5 probabilistic dynamics models predicting (delta_s, reward)."""

    def __init__(self, state_dim, action_dim, hidden=256, n=5):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * (state_dim + 1)),  # mean + log_std
            ) for _ in range(n)
        ])
        self.state_dim = state_dim

    def sample(self, s, a):
        idx = np.random.randint(len(self.heads))
        x = torch.cat([s, a], dim=-1)
        out = self.heads[idx](x)
        mu, log_std = out.chunk(2, dim=-1)
        eps = torch.randn_like(mu) * log_std.exp()
        delta = (mu + eps)[..., : self.state_dim]
        reward = (mu + eps)[..., self.state_dim:]
        return s + delta, reward
```

### Result

On MuJoCo HalfCheetah, MBPO reaches ~10,000 return in **~100K** environment steps where SAC needs ~1M and PPO ~1.6M. The empirically-optimal rollout length is **$k=1$**for most tasks; longer rollouts hurt because compounding error overwhelms the additional credit-assignment depth.

---

## 4. Pure Planning: Model Predictive Control

![Model Predictive Control: shoot, score, execute first action, repeat](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig5_mpc_planning.png)

If the model is good enough, you can skip "policy" altogether and **plan from scratch at every step**. **Model Predictive Control** (MPC) is the workhorse of classical control engineering, and a learned dynamics model slots straight into it.

The loop:

1. Sample$N$candidate action sequences$a_{t:t+H}$(uniform, Gaussian, or from a CEM/iCEM proposal distribution).
2. Roll each one forward $H$ steps in the **learned** model and score by predicted return.
3. **Execute only the first action** of the best sequence.
4. Observe the real next state and re-plan.

The figure shows 12 candidate trajectories (grey), the best one (green), and the single highlighted action that actually gets sent to the actuator. Crucially, executing one step at a time means the model only has to be locally accurate -- compounding error never gets a chance to wreck a long open-loop plan.

MPC is the dominant choice when **the cost of a mistake is high** (real robots, surgery, autonomous driving). It is also the bridge between learned models and the rest of the planning literature: PETS, PlaNet, TD-MPC, and Dreamer's policy improvement loop all reduce to "MPC inside a learned model" in some form.

---

## 5. World Models: Dreaming in a Latent Space

![World Model V/M/C architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig2_world_model_vmc.png)

MBPO works because MuJoCo states are 11-23 dimensional. Predicting the next$84\times 84\times 3$Atari frame, by contrast, is hopelessly hard -- and most of those pixels (sky, scoreboard) are irrelevant to control. **World Models** (Ha & Schmidhuber, 2018) propose a different shape:

> Compress observations into a small latent code, then learn dynamics *in that latent space*.

Three components, drawn left-to-right above:

- **V (Vision)** -- a Variational Autoencoder maps each frame$o_t$to a ~32-dimensional latent$z_t$. The reconstruction loss forces$z_t$to retain enough information about the scene.
- **M (Memory)** -- a Mixture-Density-Network RNN models$P(z_{t+1}\mid z_t, a_t, h_t)$, where$h_t$is the recurrent state. M *is* the world model.
- **C (Controller)** -- a deliberately tiny linear policy maps$(z_t, h_t)\to a_t$. On CarRacing it has just **867 parameters** versus DQN's 1.7M.

### Why It Works -- and Why It Was Surprising

The controller can be trained **entirely in dreams**: roll out M from a sampled$z$, get pseudo-trajectories, evolve C with CMA-ES, and never touch the real environment until evaluation. The 867-parameter controller scores near-human on CarRacing-v0. The deeper lesson, which all of Dreamer / DreamerV3 / TD-MPC inherit, is that *learning a useful representation is most of the problem*: once V and M are in place, control is almost trivial.

---

## 6. Dreamer: End-to-End Latent Imagination

![Dreamer RSSM latent dynamics across three time steps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig7_dreamer_latent.png)

World Models trains V, M, C in three separate phases, which means the VAE optimises for pixel reconstruction rather than for what the controller actually needs. **Dreamer** (Hafner et al., ICLR 2020; DreamerV2 2021; DreamerV3 2023) trains the entire stack jointly and adds a key architectural piece: the **Recurrent State-Space Model (RSSM)**.

### RSSM in One Picture

The figure shows three time steps. At each step the latent state has two parts:

-$h_t$**(deterministic)** -- a GRU hidden state carrying long-range memory:$h_t = \mathrm{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$.
-$z_t$**(stochastic)** -- a small categorical or Gaussian latent sampled from a prior$p(z_t\mid h_t)$at imagination time, or a posterior$q(z_t\mid h_t, o_t)$at training time.

This separation matters. The deterministic$h$ remembers, while the stochastic$z$ models genuinely uncertain dynamics (a Pong ball going off-screen, a Minecraft chest hiding random loot). Heads on$(h_t, z_t)$predict reward, value, and (during training) the observation -- so any decoder loss flows back through the dynamics and into the representation.

### Behaviour Learning Happens Entirely in Imagination

Once the world model is fitted on real data, Dreamer trains the actor and critic by:

1. Sampling a batch of real$(h_t, z_t)$as starting points.
2. Rolling the **prior** dynamics 15 steps forward, sampling actions from the actor.
3. Bootstrapping a value target through the imagined trajectory and updating the actor by reparameterised policy gradient.

No real interaction during this step. A single batch on the real buffer fuels thousands of imagined gradient updates -- the same Dyna idea, but now in a learned latent space.

### Results

- **DMControl Walker:** ~900 return in 100K steps, where SAC needs ~1M.
- **Atari:** DreamerV2 matches IQN/Rainbow on the 55-game suite *while running on a single GPU*.
- **Minecraft (DreamerV3, 2023):** the first algorithm to collect diamonds from scratch, with no demonstrations and no per-task hyperparameter tuning.

DreamerV3's claim to fame is robustness: the same model-based agent, with the same hyperparameters, beats specialised baselines across more than 150 tasks spanning DMControl, Atari, Crafter, and Minecraft.

---

## 7. MuZero: Plan Without Predicting Pixels

The thread running through World Models and Dreamer is "predict observations". MuZero (Schrittwieser et al., *Nature* 2020) noticed that for **planning**, you do not actually need observations -- you need value, policy, and reward. Everything else is a means to that end.

MuZero learns three small networks operating on an abstract hidden state:

- **Representation:**$s_0 = h(o_0)$-- encode the real observation into a hidden state.
- **Dynamics:**$s_{k+1}, r_{k+1} = g(s_k, a_k)$-- transition purely in hidden space.
- **Prediction:**$p_k, v_k = f(s_k)$-- emit policy logits and value.

The hidden states$s_k$do not have to correspond to anything in the real environment; they only have to be useful for the Monte Carlo Tree Search that plans on top of them.

### Training Loss

For a trajectory unrolled $K$ steps with MCTS targets$z^v, z^p$and observed rewards$z^r$:

$$
\mathcal{L} = \sum_{k=0}^{K} \Big[ \ell^p(p_k, z_k^p) + \ell^v(v_k, z_k^v) + \ell^r(r_k, z_k^r)\Big].
$$

Crucially, no reconstruction term ever appears. The model is *implicit* -- it is whatever makes the MCTS targets self-consistent.

### Results

A single algorithm with a single hyperparameter set achieves:

- **Go, chess, shogi:** matches or exceeds AlphaZero -- *without being given the game rules*.
- **Atari 57:** new SOTA over R2D2.
- **MuZero Reanalyse / Sampled MuZero / EfficientZero (2021):** human-level Atari in 2 hours of game time.

MuZero is the cleanest demonstration of a deep principle: **your model only has to be as faithful as your downstream use of it requires**.

---

## 8. The Big Picture: Sample Efficiency

![Sample efficiency on MuJoCo HalfCheetah and steps-to-target bar chart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/fig6_sample_efficiency.png)

Stacking the methods on one plot makes the central claim concrete. On HalfCheetah, MBPO and Dreamer reach a score that takes SAC ~600K steps and PPO ~1.6M to match -- in **80-150K** real steps. The shape of every model-based curve is the same: a slow start (the model is being learned) followed by a sharp climb once imagined updates start carrying useful gradients.

That said, the plot also shows an honest limitation. Model-based curves do not always **exceed** model-free asymptotes; they reach the same level much faster. When samples are cheap, model-free methods often win by simplicity. When samples are expensive -- which is the empirically interesting regime -- model-based wins by a wide margin.

---

## 9. Choosing the Right Tool

| Scenario                                          | Method        | Why                                         |
| ------------------------------------------------- | ------------- | ------------------------------------------- |
| Small discrete environment, fast iteration        | Dyna-Q        | Tabular, trivially correct, instant payoff  |
| Continuous control, moderate-dim states           | MBPO          | Short rollouts + SAC give 10x sample gain   |
| Real robot, expensive interaction                 | MPC + ensemble dynamics (PETS/iCEM) | Plan locally, never trust long open-loop   |
| Pixel observations, limited budget                | Dreamer / DreamerV3 | Latent dynamics handle high-dim sensors  |
| Perfect-information games / discrete planning     | MuZero        | Implicit model + MCTS, no rules required    |
| You already have a free, fast simulator           | PPO / SAC     | No model error to worry about               |

---

## 10. Open Problems

Three frontiers are particularly active:

1. **Model error in the long tail.** All current methods either keep horizons short (MBPO, MPC) or hide the problem in latent space (Dreamer). Neither scales gracefully to tasks needing 1000-step credit assignment with photorealistic dynamics.
2. **Stochastic and multi-modal worlds.** RSSM's stochastic$z$is a step, but predicting genuinely multi-modal futures (driving, dialogue) remains hard.
3. **World models for foundation agents.** Recent work (Genie, Sora-as-world-model, V-JEPA) treats large generative video models as the dynamics component; whether a single pretrained world model can transfer across tasks the way LLMs do is an open and consequential question.

---

## Summary

Model-based RL is the family of methods that **spend compute to save samples**:

- **Dyna** introduced the loop -- mix real and imagined updates to amortise interaction.
- **MBPO** showed that *short* imagined rollouts beat long ones, because model error compounds.
- **MPC** treats the model as a one-step-ahead simulator and replans every step.
- **World Models** moved dynamics learning into a compressed latent space, making pixels tractable.
- **Dreamer / RSSM** trains representation, dynamics, and policy jointly and learns behaviour entirely in imagination.
- **MuZero** dropped reconstruction altogether: the model just has to be self-consistent under MCTS.

The unifying lesson is that **what you predict should match how you use the prediction**. Predict pixels if pixels matter; predict$(r, v, p)$if those are all that the planner consumes. That principle is what makes the modern wave -- DreamerV3, EfficientZero, TD-MPC2 -- finally feel general.

**Next up:** [Part 6](/en/reinforcement-learning-6-ppo-and-trpo/) dives into **PPO and TRPO** -- the trust-region policy gradient methods that quietly power industrial RL, from robotic manipulation to ChatGPT's RLHF.

---

## References

- Sutton, R. S. (1990). *Integrated architectures for learning, planning, and reacting based on approximating dynamic programming*. ICML.
- Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). *When to trust your model: model-based policy optimization*. NeurIPS. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). *Deep reinforcement learning in a handful of trials using probabilistic dynamics models* (PETS). NeurIPS. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)
- Ha, D., & Schmidhuber, J. (2018). *World models*. NeurIPS. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to control: learning behaviors by latent imagination* (Dreamer). ICLR. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with discrete world models* (DreamerV2). ICLR. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- Hafner, D., et al. (2023). *Mastering diverse domains through world models* (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Schrittwieser, J., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model* (MuZero). *Nature*. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- Ye, W., Liu, S., Kurutach, T., Abbeel, P., & Gao, Y. (2021). *Mastering Atari games with limited data* (EfficientZero). NeurIPS. [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)
- Hansen, N., Wang, X., & Su, H. (2022/2024). *Temporal difference learning for model predictive control* (TD-MPC / TD-MPC2). [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

---

## Series Navigation

| Part   | Topic                                                                                                |
| ------ | ---------------------------------------------------------------------------------------------------- |
| 1      | [Fundamentals and Core Concepts](/en/reinforcement-learning-1-fundamentals-and-core-concepts/)       |
| 2      | [Q-Learning and DQN](/en/reinforcement-learning-2-q-learning-and-dqn/)                               |
| 3      | [Policy Gradient and Actor-Critic](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/)   |
| 4      | [Exploration and Curiosity-Driven Learning](/en/reinforcement-learning-4-exploration-and-curiosity-driven-learning/) |
| **5**  | **Model-Based RL and World Models (you are here)**                                                   |
| 6      | [PPO and TRPO](/en/reinforcement-learning-6-ppo-and-trpo/)                                           |
