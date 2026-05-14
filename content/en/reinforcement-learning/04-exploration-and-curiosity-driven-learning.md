---
title: "Reinforcement Learning (4): Exploration Strategies and Curiosity-Driven Learning"
date: 2025-08-16 09:00:00
tags:
  - Reinforcement Learning
  - Exploration Strategies
  - Intrinsic Reward
  - Curiosity-Driven Learning
  - ICM
  - RND
  - NGU
  - Count-Based Methods
categories: Reinforcement Learning
series: reinforcement-learning
lang: en
mathjax: true
description: "How do RL agents discover rewards when the environment gives almost no feedback? From count-based methods to ICM, RND, and NGU -- the science of curiosity-driven exploration."
disableNunjucks: true
series_order: 4
translationKey: "reinforcement-learning-4"
---
Drop a fresh agent into Montezuma's Revenge. To score a single point, it must walk to the right, jump over a skull, climb a rope, leap to a platform, and grab a key — roughly **a hundred precise actions in a row**. Until the key is collected, the reward signal is always zero.

A textbook DQN with $\varepsilon=0.1$ exploration has, by a generous estimate, a $0.1^{100} \approx 10^{-100}$ chance of stumbling onto that key by accident. Unsurprisingly, vanilla DQN scores **0** on this game. Not "low" — literally zero, every episode, for the entire training run.

This is the **sparse-reward problem**, and it exposes an uncomfortable truth: a deep RL algorithm is only as good as its exploration strategy. Even the finest Bellman backup is useless if the agent never observes a non-zero reward. This chapter explores the path from blind random exploration to **curiosity-driven learning** — algorithms that generate their own rewards for discovering new things.

![Reinforcement Learning (4): Exploration Strategies and Curiosity-Driven Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/illustration_1.png)


---

## What You Will Learn

- Why $\varepsilon$-greedy, Boltzmann, and even UCB collapse in high-dimensional environments
- **Count-based** methods and pseudo-counts via density models
- **ICM** (Intrinsic Curiosity Module): prediction error in a learned feature space
- **RND** (Random Network Distillation): the simplest curiosity signal that actually works at scale
- **NGU** (Never Give Up): episodic memory for tasks that punish forgetting
- Practical PPO + curiosity recipes, hyper-parameters, and failure modes

**Prerequisites:** [Part 1-3](/en/reinforcement-learning/01-fundamentals-and-core-concepts/) (MDPs, DQN, policy gradients, PPO basics).

---

## Why exploration is so hard

### Classical schedules and what they actually look like

Every introductory RL course starts with **$\varepsilon$-greedy**: with probability $\varepsilon$ pick a uniformly random action, otherwise pick the greedy one. The hard part is not the formula — it is the *schedule*: how should $\varepsilon$ decay as training progresses?

![Epsilon-greedy decay schedules and induced action probabilities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig1_epsilon_greedy_decay.png)

The figure above shows three popular schedules (linear, exponential, piecewise step) on the left and, on the right, the actual action distribution under linear decay with four available actions. Three observations are worth noting:

1. **Different schedules give very different "exploration budgets."** Exponential decay spends most of its random actions in the first 20k steps; linear decay spreads them more evenly; step schedules behave like crude curricula.
2. **Even at $\varepsilon = 0.05$ a quarter of the random actions still go to the greedy choice** ($1 - \varepsilon + \varepsilon/|\mathcal{A}|$), which surprises people who expect 5% noise to mean 5% off-policy behaviour.
3. **None of these curves consider the state.** Exploration is purely a function of the training step. This is the central weakness we will address.

Mathematically:
$$
\pi_\varepsilon(a \mid s) = \begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & a = \arg\max_{a'} Q(s, a') \\[4pt]
\dfrac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$
### Boltzmann (softmax) exploration: a marginal upgrade

Instead of an all-or-nothing random kick, **Boltzmann** exploration weights actions by their Q-values.
$$\pi_\tau(a \mid s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}.$$
The temperature $\tau$ replaces $\varepsilon$ as the exploration knob. As $\tau \to 0$ the policy becomes greedy; as $\tau \to \infty$ it becomes uniform.

![Boltzmann action distribution and policy entropy for several temperatures](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig2_boltzmann_softmax.png)

The right panel plots policy entropy $H(\pi_\tau) = -\sum_a \pi_\tau(a) \log \pi_\tau(a)$ against $\tau$. The curve flattens to the maximum value $\ln |\mathcal{A}|$ surprisingly fast: by $\tau = 2$ you are already above 90 % of maximum entropy. This is exactly the **entropy bonus** trick used inside PPO and SAC — it is Boltzmann exploration, repackaged as a regulariser on the policy network.

But Boltzmann shares $\varepsilon$-greedy's fatal flaw: it spreads probability based on the agent's *current Q estimates*, not on any notion of how much it has actually visited each region of the state space. Two states the agent has never seen still get the same softmax over the same untrained Q-values.

### UCB: the principled approach that does not scale

For multi-armed bandits, the classical **UCB1** rule is provably near-optimal.
$$a_t = \arg\max_a \left[ \hat Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right].$$
The first term *exploits*; the second *explores* — arms pulled fewer times receive a larger uncertainty bonus.

![UCB1 score decomposition and arm-pull statistics over a 5-arm bandit](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig3_ucb_bandit.png)

Watch what happens in the figure above. At $t = 50$ the bonus dominates (orange bars), all arms look attractive, and pulls are spread across the bandit. By $t = 1000$ the uncertainty bonus has shrunk for the optimal arm 3 (it has been pulled hundreds of times) but stays high for the rest, so the algorithm settles into a *near-greedy regime* on arm 3 while still occasionally checking the others. This is exploration done right — and it is *guided by data*, not by a pre-baked schedule.

So why do we not just use UCB everywhere? **Because $N(s,a)$ is meaningless in high-dimensional state spaces.** In Atari each frame is $84 \times 84 \times 4 = 28{,}224$ pixels; the agent will essentially never see the same state twice. So $N(s, a) = 1$ for almost every encountered state-action pair, and the bonus becomes a useless constant.

### Thompson sampling: posterior beliefs over rewards

A close cousin of UCB is **Thompson sampling**: maintain a posterior over each arm's reward parameter, sample one possible world from that posterior, and pick the action that is best in that sampled world. For Bernoulli arms with a Beta prior the update is delightfully simple — on success increment $\alpha$, on failure increment $\beta$.

![Beta posteriors for three Bernoulli arms after 10, 50, and 300 pulls](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig4_thompson_sampling.png)

The figure shows posteriors sharpening around the true reward rates (dashed). After 10 pulls the algorithm has barely committed; by 300 pulls the posterior on the best arm (arm 2, $\mu = 0.75$) is a tight spike, while losing arms still carry enough variance to keep them in the running. Thompson sampling tends to match or beat UCB in practice and is widely used inside contextual-bandit recommender systems. As an exploration strategy in deep RL, however, it suffers the same generalisation problem as UCB: maintaining a posterior over a $10^{60}$-dimensional state-action space is intractable, and *Bayesian* deep RL approaches (e.g. **Bootstrapped DQN**, **Bayes by Backprop**) only partially recover its benefits.

### Four reasons exploration is hard in deep RL

Putting the pieces together, the difficulty of exploration in real environments compounds along four axes:

1. **Sparse rewards.** Hundreds or thousands of correct actions before any external signal arrives. Random exploration cannot find a needle in this haystack.
2. **Combinatorial state spaces.** $256^{28224}$ possible Atari frames; you cannot count visits if you never see the same thing twice.
3. **Local optima.** Small early rewards (a coin in a dead-end corridor) can permanently distract the agent from the much larger reward behind a hard-to-find door.
4. **The noisy-TV problem.** A naive novelty signal mistakes random pixel noise for "new and interesting." Drop a TV showing static into the room and the agent will sit and watch it forever.

The conceptual leap of the modern era is to stop computing exploration as a *function of step number* and start computing it as a *function of the agent's own experience*. Babies do this naturally: they play with new toys, not old ones.

---

## The curiosity blueprint: intrinsic rewards

![Reinforcement Learning (4): Exploration Strategies and Curiosity-Driven Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/illustration_2.png)

Every modern method we will discuss adds a **intrinsic reward** $r^{\text{int}}_t$ on top of the environment's external reward:
$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t.$$
The agent is then trained on $r^{\text{total}}$ with whatever RL algorithm you like (DQN, PPO, IMPALA, ...). All the design effort goes into a single question:

> **How do we compute $r^{\text{int}}_t$ so it is large for genuinely novel experiences and small for stale ones, without requiring us to count pixels?**

We now look at three increasingly elegant answers.

---

## Count-based methods: when counting still works

The cleanest definition of novelty is "states I have visited fewer times." For tabular MDPs this gives the MBIE-EB bonus
$$r^{\text{int}}(s) = \frac{\beta}{\sqrt{N(s)}},$$
which has matching theoretical guarantees to UCB. The headache is that $N(s) = 0$ for almost every state in pixel-based environments.

Bellemare et al. (2016) replaced the literal count with a **pseudo-count** derived from a density model $\rho(s)$:
$$\hat N(s) = \frac{\rho(s)\bigl(1 - \rho_{\text{new}}(s)\bigr)}{\rho_{\text{new}}(s) - \rho(s)},$$
where $\rho_{\text{new}}$ is the model's density after training on one extra observation of $s$. The construction pretends $\rho$ is the empirical distribution of a giant counter and inverts it. With a strong density model (PixelCNN, neural autoregressive models) this delivers the first non-trivial scores on Montezuma's Revenge — but the model is expensive to train, fragile, and the noisy-TV problem is still wide open: random pixel noise looks low-density, hence high-novelty, hence highly rewarded. Modern systems abandoned pseudo-counts for the more robust prediction-error methods we cover next.

---

## ICM: curiosity through prediction error

![Intrinsic Curiosity Module — features encoder + inverse + forward dynamics with prediction-error bonus.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-Exploration-and-Curiosity-Driven-Learning/fig04_icm.png)

**Intrinsic Curiosity Module (ICM)** (Pathak et al., ICML 2017) replaces "have I seen this before?" with "can I predict what happens next?". The intuition:

- If the environment's response to my action is *predictable*, I already understand it — low intrinsic reward.
- If the response *surprises* me, I have found something I do not yet model — high intrinsic reward.

But predicting raw pixels is a bad idea: TV static is unpredictable yet useless. ICM's brilliant move is to predict in a **learned feature space** $\phi$ that only encodes information *the agent's actions can affect*.

![ICM and RND architectures side by side](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig5_icm_rnd_architecture.png)

The left half of the figure above shows the three components:

1. **Encoder $\phi$** (CNN) maps $s_t$ to a feature vector $\phi(s_t)$.
2. **Forward model $\hat f$** predicts the next features from current features and action: $\hat\phi_{t+1} = \hat f(\phi(s_t), a_t)$. Its squared error
   $$
   r^{\text{int}}_t = \eta \,\bigl\| \hat\phi_{t+1} - \phi(s_{t+1}) \bigr\|^2$$   *is* the intrinsic reward.
3. **Inverse model $g$** predicts the action from a pair of consecutive features: $\hat a_t = g(\phi(s_t), \phi(s_{t+1}))$. Its loss flows back into the encoder $\phi$.

Step 3 is the magic ingredient. The inverse model can only succeed if $\phi$ retains information that *changes between $s_t$ and $s_{t+1}$ as a function of the agent's action*. Static background pixels, TV noise, and other action-independent distractions get filtered out of $\phi$ because they do not help predict $a_t$. Once they are gone, the forward model cannot mistake them for novelty.

### Reference implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    """Intrinsic Curiosity Module (Pathak et al., 2017)."""

    def __init__(self, obs_channels, action_dim, feature_dim=256, eta=0.1):
        super().__init__()
        self.eta = eta

        # Feature encoder (Atari-style CNN).
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
        )

        # Forward model: (phi_t, a_t) -> phi_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Inverse model: (phi_t, phi_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs, next_obs, action_onehot):
        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)

        # Forward prediction error == intrinsic reward.
        phi_next_pred = self.forward_model(
            torch.cat([phi, action_onehot], dim=1))
        forward_err = F.mse_loss(
            phi_next_pred, phi_next.detach(), reduction="none").sum(dim=1)
        intrinsic_reward = self.eta * forward_err

        # Inverse-model loss trains the encoder to keep only
        # action-relevant information.
        action_pred = self.inverse_model(torch.cat([phi, phi_next], dim=1))
        inverse_loss = F.cross_entropy(action_pred, action_onehot.argmax(1))

        return intrinsic_reward, forward_err.mean(), inverse_loss
```

### What you can expect

On Montezuma's Revenge, ICM + A3C reaches roughly **6,600** points within 25 M frames (vanilla DQN sits at 0 forever). Even more striking, an agent trained with **zero external reward** — only ICM's intrinsic signal — learns to navigate the first few rooms, dodge enemies, and pick up keys, simply because doing those things keeps its forward model on its toes.

### Where ICM falls down

- **Stochastic environments.** If a slot machine in the environment has a genuinely random outcome, the forward model can never predict it, and the agent fixates on pulling the lever. The inverse-model trick filters action-independent randomness, but action-*dependent* randomness still bites.
- **Compute.** Three networks (encoder, forward, inverse) on top of your policy; expect roughly 2x training cost compared to vanilla PPO.

---

## RND: a startlingly simple alternative

**Random Network Distillation** (Burda et al., ICLR 2019) replaces the entire forward/inverse machinery with a single observation:

> *Distil a fixed random network. Wherever the predictor still has high error, the agent has not been there enough.*

Concretely, RND keeps two networks (right half of the architecture figure above):

- A **target network $f$**, with random weights, **frozen** for the entire training run.
- A **predictor network $\hat f$**, trained by gradient descent to minimise $\| \hat f(s) - f(s) \|^2$ on observed states.

Intrinsic reward is the predictor's residual:
$$
r^{\text{int}}(s) = \bigl\| \hat f(s) - f(s) \bigr\|^2.
$$
For a state the predictor has seen many times, training has driven the loss to near zero — low reward. For a novel state, the predictor has never been trained there, so its output is random and the residual is large — high reward. The frozen target acts like a deterministic hash function: structurally similar states map to similar targets, so generalisation comes for free.

The same trick neutralises the noisy-TV problem. Random static frames are visually different but **structurally similar** in the eyes of a random CNN; the predictor learns to match them after a handful of updates and the reward decays to zero. ICM and RND therefore handle "noisy TV" by completely different mechanisms — ICM by filtering the *features*, RND by exploiting the *consistency of a random map*.

### Reference implementation

```python
class RND(nn.Module):
    """Random Network Distillation (Burda et al., 2019)."""

    def __init__(self, obs_channels, output_dim=512):
        super().__init__()

        # Target network: random weights, NEVER updated.
        self.target = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, output_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network: trained to match the target.
        self.predictor = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # Running statistics for reward normalisation.
        self.r_mean, self.r_std = 0.0, 1.0

    def compute_reward(self, obs):
        with torch.no_grad():
            r = (self.predictor(obs) - self.target(obs)).pow(2).sum(dim=1)
        return r

    def update(self, obs):
        loss = F.mse_loss(self.predictor(obs), self.target(obs).detach())
        return loss

    def normalise(self, r):
        # Critical: keep intrinsic reward on a stable scale.
        self.r_mean = 0.99 * self.r_mean + 0.01 * r.mean().item()
        self.r_std = max(0.99 * self.r_std + 0.01 * r.std().item(), 1e-8)
        return (r - self.r_mean) / self.r_std
```

### Headline numbers

- **Montezuma's Revenge**: 8,152 average score — the first algorithm to surpass the human expert benchmark of 7,385.
- **Pitfall**: 70.4 (previous best: exactly 0).
- RND was, for a year or two, the strongest reported method on every "hard exploration" Atari game.

The two non-obvious tricks that make RND work: (i) **two value heads** in the policy, one for $r^{\text{ext}}$ and one for $r^{\text{int}}$, with separate discount factors; (ii) **per-environment normalisation** of intrinsic rewards using a running estimate of their standard deviation. Without (ii) the intrinsic-reward scale grows wildly, drowns out the external reward, and training collapses.

---

## NGU: never give up on a state

RND has a quiet assumption: once a state's predictor error has been driven to zero, that state is never novel again. Most of the time this is what we want. But two important cases break it:

- **Key-door tasks.** The agent needs to *re-pick-up* a key every time it dies and respawns. After the first pickup, RND considers the key location boring; the agent never goes back for it.
- **Backtracking.** After exploring the right half of a level, the agent must return through the start and head left. The start has been visited thousands of times; RND sees no reason to ever go through it again.

**Never Give Up (NGU)** (Badia et al., ICLR 2020) repairs this by combining two novelty signals:
$$
r^{\text{int}}_t = r^{\text{episodic}}_t \cdot \min\bigl(r^{\text{lifetime}}_t,\; L\bigr).
$$
- **Episodic novelty** $r^{\text{episodic}}$. Maintain an episodic memory of state embeddings *for the current episode only*. The reward is large when the current state is far (in embedding space) from anything in this memory. Crucially, the memory **resets on each new episode**, so even a state that has been visited millions of times across training still feels novel within a fresh run.
- **Lifetime novelty** $r^{\text{lifetime}}$. The classic RND signal, capped at $L$ to prevent run-away.

The multiplication insists on both: a state must be locally novel *and* globally not yet exhausted to score high. NGU also introduces a **family of policies** with different exploration coefficients trained in parallel, and a **directed exploration** scheme on top — but the episodic-lifetime decomposition is the heart of the idea.

The successor, **Agent57** (Badia et al., 2020), bolts NGU onto a meta-controller that picks which exploration policy to use at each moment. It is the first single algorithm to surpass the human baseline on **all 57 Atari games**.

### Highlight scores

| Game | DQN (ε-greedy) | RND | NGU | Human expert |
|------|---------------:|----:|----:|-------------:|
| Montezuma's Revenge | 0 | 8,152 | 11,000+ | 7,385 |
| Pitfall! | 0 | 70 | 5,000+ | 6,464 |
| Private Eye | 0 | 8,800 | 69,000 | 69,571 |

---

## Visualising the gap: random vs curious agents

It is worth seeing, on a tiny problem, just how different "random" and "curious" exploration look.

![Visit-count heatmaps for ε-greedy and curiosity-driven agents on a 25x25 grid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig7_trajectory_comparison.png)

Both panels show 1,500 steps of an agent starting at the white star in a 25 x 25 GridWorld. The blue agent is a uniform random walker; the purple agent samples its next move with probability proportional to $1/\sqrt{N(s')}$ over neighbours — the simplest possible count-based curiosity reward. After the same number of steps, the random walker has covered only **65.6 %** of the grid and revisited its favourite cell **19** times; the curious agent has covered **80 %** with a maximum revisit count of **11**. The right panel makes the same point with a sorted log-scale visit-count distribution: curiosity flattens the head of the distribution (no over-visited cells) and lifts the tail (no neglected cells).

Now multiply that effect over 100 million Atari frames in a state space the size of an exoplanet and you understand why curiosity is the difference between scoring zero and scoring eleven thousand.

![Score curves and time-to-first-reward on Montezuma's Revenge](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig6_sparse_reward_montezuma.png)

The figure above tells the same story at scale. Vanilla DQN stays at zero forever — it never finds a single reward. Each successive generation of curiosity (count-based -> ICM -> RND -> NGU) shaves frames off the time-to-first-reward and lifts the asymptotic score. NGU is the first to clear the human-expert dashed line.

---

## Practical recipes: PPO + curiosity in production

### Hyper-parameter starting points

| Parameter | Suggested start | Notes |
|-----------|-----------------|-------|
| Intrinsic coefficient $\beta$ | 0.01 | Push to 0.1 if rewards are extremely sparse |
| Reward normalisation | **always on**, separately for $r^{\text{ext}}$ and $r^{\text{int}}$ | Without this, RND/ICM can dwarf the task reward |
| Discount $\gamma_{\text{int}}$ | 0.99 | Often *lower* than $\gamma_{\text{ext}}$ to encourage local exploration |
| Two value heads | yes | One head per reward stream is consistently better than mixing |
| Frame skip / stack | 4 / 4 | Atari standard |
| Gradient clipping | 0.5 | ICM/RND gradients can be heavy-tailed |

### What progress should look like

On Montezuma's Revenge with PPO + RND on a single modern GPU:

| Frames | Expected milestone |
|--------|-------------------|
| 10 M | Consistently clears the first room (~100 pts) |
| 30 M | Picks up the key, opens the door (~400 pts) |
| 100 M | Average return 6,000+, occasional 8,000+ |

If you are stuck below the first milestone for 20 M frames, the most common culprits (in order) are: forgetting to normalise $r^{\text{int}}$, using a single value head, or setting $\beta$ too low.

### Choosing a method

| Setting | Recommended exploration | Reason |
|---------|------------------------|--------|
| Dense rewards, low-dim observations | $\varepsilon$-greedy with linear decay | Cheap, sufficient |
| Continuous control (MuJoCo, robotics) | SAC entropy bonus or parameter noise | Smooth action spaces want smooth exploration |
| Sparse-reward Atari, single hard game | **PPO + RND** | Best simplicity / performance ratio |
| Sparse-reward tasks needing revisits | **NGU** | Episodic memory handles "do this thing again" |
| All-of-Atari, research-grade results | Agent57 / Go-Explore | State-of-the-art, much more engineering |

---

## Count-Based Exploration with Hash Codes

Tabular MDPs make exploration feel solved. Maintain a counter $N(s,a)$, add a UCB-style bonus $r^+(s,a) = \beta / \sqrt{N(s,a)}$, and you get a regret bound straight out of the multi-armed bandit textbook. The catch is that "tabular" means the agent can recognise the same state twice — which falls apart the moment $s$ is a $210 \times 160 \times 3$ Atari frame.

I tried storing raw observations as dict keys once. After 50k steps the dict had 50k entries because no two pixel arrays were ever bit-identical. The bonus collapsed to a constant $\beta / 1$ and the agent behaved like vanilla $\varepsilon$-greedy with extra steps.

The fix from Tang et al. (2017) is **SimHash**: project the state down to $k$ bits using a fixed random matrix, and count the resulting bit-strings. Two perceptually similar states map to the same bucket; noise gets averaged out. With $k=64$ the bucket space is $2^{64}$, which sounds large but the *occupied* buckets stay manageable because the projection groups visually similar states.

```python
import numpy as np
from collections import defaultdict

class SimHashCounter:
    """Count-based exploration via random-projection hashing (Tang 2017)."""

    def __init__(self, obs_dim, k=64, beta=0.1):
        # A fixed Gaussian projection matrix. Never updated.
        self.A = np.random.randn(k, obs_dim).astype(np.float32)
        self.counts = defaultdict(int)
        self.beta = beta

    def hash(self, obs):
        # Sign of the projection -> a k-bit binary code.
        bits = np.sign(self.A @ obs.flatten()).astype(np.int8)
        return bits.tobytes()  # hashable

    def observe(self, obs):
        h = self.hash(obs)
        self.counts[h] += 1
        return self.counts[h]

    def bonus(self, obs):
        n = self.counts[self.hash(obs)]
        return self.beta / np.sqrt(max(n, 1))
```

Drop this in front of any DQN agent and add `bonus(obs)` to the environment reward. On Montezuma's Revenge, Tang et al. report a jump from **0** to roughly **2,500** within 50M frames — not state of the art, but the first sign of life for a method that fits in thirty lines.

Two failure modes I have hit personally. First, the bonus dominates if you forget to normalise: $\beta = 0.1$ on a game where the real reward is once-per-1000-steps means curiosity *is* the policy gradient. Second, the projection matrix wants to see whitened observations; raw pixel intensities in $[0, 255]$ make every hash collide on the high-magnitude dimensions. Divide by 255 before the projection, or use a learned encoder upstream.

The choice of $k$ matters less than I expected. Below $k = 32$ the bucket space is too coarse and distinct rooms in Montezuma collide; above $k = 256$ every state is unique and the counts never accumulate. Anywhere in between works, with a slight bias toward the lower end on simpler environments.

---

## ICM: Intrinsic Curiosity Module From Scratch

Counts work but feel crude. The next step up — Pathak et al. (2017) — is to train two small networks alongside the policy and use prediction error as the bonus. The package is called the **Intrinsic Curiosity Module (ICM)** and it has three pieces.

A feature encoder $\phi: \mathcal{S} \to \mathbb{R}^d$ maps observations into an embedding. An **inverse model** $g(\phi(s_t), \phi(s_{t+1})) \to \hat a_t$ tries to recover the action that connected two consecutive states. A **forward model** $f(\phi(s_t), a_t) \to \hat\phi_{t+1}$ predicts the next embedding given the current one and the action taken.

The intrinsic reward is the forward model's residual:
$$
r^i_t = \eta \, \bigl\| \hat\phi_{t+1} - \phi(s_{t+1}) \bigr\|^2.
$$
The clever bit is *why* this resists the noisy-TV problem. Because $\phi$ is trained only by the inverse-model loss, gradients flow into the encoder only when a feature dimension helps predict the action. Random pixel noise — by definition — cannot help predict the agent's action, so it is squeezed out of $\phi$. The forward model never has to predict it, and the agent never gets fascinated by static.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    """Intrinsic Curiosity Module (Pathak 2017)."""

    def __init__(self, obs_dim, n_actions, feat_dim=128, eta=0.5):
        super().__init__()
        self.eta = eta
        self.n_actions = n_actions

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, feat_dim),
        )
        self.inverse = nn.Sequential(
            nn.Linear(2 * feat_dim, 256), nn.ELU(),
            nn.Linear(256, n_actions),
        )
        self.forward_net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 256), nn.ELU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, s, s_next, a):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        a_onehot = F.one_hot(a, self.n_actions).float()

        # Forward model: predict next features from current + action.
        phi_next_hat = self.forward_net(torch.cat([phi, a_onehot], dim=1))
        forward_err = (phi_next_hat - phi_next.detach()).pow(2).sum(dim=1)

        # Intrinsic reward (no gradient back to policy through this path).
        r_int = self.eta * forward_err.detach()

        # Inverse model: predict action from features.
        a_logits = self.inverse(torch.cat([phi, phi_next], dim=1))
        inv_loss = F.cross_entropy(a_logits, a)

        return r_int, forward_err.mean(), inv_loss
```

Wiring this into PPO is mechanical. Each rollout step, call `r_int, fwd_loss, inv_loss = icm(s, s_next, a)`, add `r_int` to the environment reward before computing advantages, and add `fwd_loss + inv_loss` to the policy's optimiser as an auxiliary loss. The convention from the paper is $\eta = 0.5$ and a 0.2/0.8 split between forward and inverse losses.

On VizDoom's MyWayHome (sparse-reward navigation), extrinsic-only PPO scores 0 for the entire training budget. ICM-PPO solves the level in ~5M frames. The same module on Mario reaches the end of world 1-1 with no extrinsic reward at all — moving forward keeps surprising the forward model, so the agent learns to move forward.

A debugging note. The intrinsic reward should *decay* over training as the forward model improves. If it stays flat, either your forward model is undercapacity, or your encoder $\phi$ is collapsing to a constant — both inverse-model and forward-model losses going to near-zero is the giveaway. The fix is to add a small reconstruction loss on $\phi$ to prevent collapse, or to widen the inverse model so it actually has to use the features.

---

## Random Network Distillation (RND): The Stable Curiosity

ICM works, but training a forward dynamics model in a stochastic environment is delicate. Pathak's own follow-up admits the inverse-model trick doesn't fully neutralise action-dependent randomness. Burda et al. (2018) replace the entire dynamics machinery with one observation: you don't need to predict the *next* state, you just need a moving target that decays as states get visited.

Take a randomly initialised network $f$, freeze it forever, and train a second network $\hat f$ to match its output on observed states. The bonus is the squared residual:
$$
r^i(s) = \bigl\| \hat f(s) - f(s) \bigr\|^2.
$$
For frequently-visited $s$, the predictor has converged and the residual is small. For a novel $s$, the predictor has never received gradient there and produces something close to its random initialisation, far from the target.

The key advantage over ICM is that $f$ is *fixed*. There is no Bellman-style chicken-and-egg between predicting and the target moving; the target is a deterministic function of $s$. Stochastic transitions don't matter because $f$ doesn't see transitions, only states.

```python
class RND(nn.Module):
    """Random Network Distillation (Burda 2018) — minimal version."""

    def __init__(self, obs_dim, hid=256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, hid),
        )
        # Running stats for reward normalisation.
        self.r_rms_mean, self.r_rms_var, self.r_rms_n = 0.0, 1.0, 1e-4

    def bonus(self, obs):
        with torch.no_grad():
            r = (self.predictor(obs) - self.target(obs)).pow(2).sum(-1)
        return r

    def loss(self, obs):
        return F.mse_loss(self.predictor(obs), self.target(obs).detach())

    def normalise(self, r):
        # Welford-style running variance, then divide by stddev.
        batch_mean = r.mean().item()
        self.r_rms_mean = 0.99 * self.r_rms_mean + 0.01 * batch_mean
        self.r_rms_var = 0.99 * self.r_rms_var + 0.01 * r.var().item()
        return r / max(self.r_rms_var ** 0.5, 1e-8)
```

Two non-negotiable hyperparameters from the paper. Normalise the intrinsic reward by a running standard deviation — without this the bonus scale drifts by orders of magnitude over training and either drowns the extrinsic reward or vanishes. Use **two value heads** in the policy, one for $r^{\text{ext}}$ and one for $r^{\text{int}}$, with separate discount factors ($\gamma_{\text{int}} = 0.99$, $\gamma_{\text{ext}} = 0.999$). Mixing them into one head hurts on every game I have tried.

On Montezuma's Revenge, RND was the first method to beat the human expert benchmark of 7,385, averaging 8,152 and occasionally clearing 11k. The implementation is simpler than ICM and trains faster because there is no inverse model to backprop through.

---

## Information Gain and VIME

Curiosity bonuses come in two intellectual flavours. ICM and RND measure novelty as *prediction error*. The other tradition — going back to Bayesian active learning — measures novelty as *information gain*: how much does observing this transition update my beliefs about the world?

VIME (Houthooft et al., 2016) makes this precise. Maintain a Bayesian posterior $p(\theta \mid h_t)$ over the parameters of a dynamics model, and reward the agent for transitions that move that posterior:
$$
r^i_t = D_{\mathrm{KL}}\bigl( p(\theta \mid h_t, a_t, s_{t+1}) \,\|\, p(\theta \mid h_t) \bigr).
$$
A transition that the model already explains well has near-zero KL — no information gained. A surprising transition shifts the posterior and pays a bonus. This is the RL analogue of querying the most informative point in active learning.

The original implementation uses Bayes-by-Backprop, which is fiddly. Pathak et al. (2019) showed that **ensemble disagreement** — train $K$ dynamics models on different bootstrap samples and use their predictive variance — is just as effective and an order of magnitude cheaper.

```python
class EnsembleDisagreement(nn.Module):
    """Information-gain bonus via ensemble variance (Pathak 2019)."""

    def __init__(self, obs_dim, act_dim, n_models=5, hid=128):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + act_dim, hid), nn.ELU(),
                nn.Linear(hid, obs_dim),
            ) for _ in range(n_models)
        ])

    def bonus(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        preds = torch.stack([m(sa) for m in self.models], dim=0)  # [K, B, D]
        return preds.var(dim=0).mean(dim=-1)  # variance across ensemble

    def loss(self, s, a, s_next):
        sa = torch.cat([s, a], dim=-1)
        return sum(F.mse_loss(m(sa), s_next) for m in self.models) / len(self.models)
```

Each ensemble member is trained on a different random subset of the replay buffer; disagreement between them is high in regions of state space the agent has explored sparsely. The bonus has a clean Bayesian interpretation as approximate posterior variance, and unlike RND it scales with the *dynamics* model's uncertainty rather than a feature predictor's, which can matter for tasks where the relevant novelty is in the transition structure rather than the marginal state distribution.

The connection to active learning is more than aesthetic. Both VIME and ensemble disagreement reduce to "act to maximise expected information gain about $\theta$", which is the standard Bayesian-experimental-design objective. The RL twist is that the agent must commit to a policy rather than picking individual queries, so the bonus must be propagated through Bellman backups rather than acted on greedily. In practice this means you treat $r^i$ as a normal reward and let the value function aggregate it across the trajectory.

---

## When Exploration Hurts: Reward Hacking

Curiosity bonuses are not free. I have shipped agents that scored higher on intrinsic reward than the previous best while doing strictly worse on the actual task. Three failure modes recur often enough to deserve names.

**Noisy TV.** The original ICM paper introduces this scenario: place a screen of random static somewhere in the environment. A naive forward model can never predict the next frame of static, so its error stays maximal forever. The agent learns to plant itself in front of the screen and stare. ICM's inverse-model trick mitigates this for action-independent noise; RND mitigates it because a random CNN maps all noise frames to nearby points and the predictor learns the average quickly. Neither is bulletproof against action-*dependent* stochasticity.

**Bonus reward farming.** Subtler and more annoying. The agent discovers a class of states with a slightly higher residual — say, mid-jump frames where the lighting differs by one byte — and learns to bounce in place. The intrinsic reward stream looks healthy on the dashboard. The extrinsic reward goes to zero. I have only ever caught this by visualising trajectories.

**Drowning the task signal.** With $\eta$ too high, the policy gradient is dominated by intrinsic returns and the extrinsic reward becomes a rounding error. The fix is either to anneal $\eta$ over training, or to normalise both reward streams to unit variance and tune their weights as a *ratio* rather than absolute scales.

A cheap monitoring rule that catches all three:

```python
def curiosity_health_check(r_int_buf, r_ext_buf, window=10_000):
    """Log r_int / r_ext ratio; warn if intrinsic dominates."""
    r_int_mean = sum(r_int_buf[-window:]) / window
    r_ext_mean = sum(r_ext_buf[-window:]) / window + 1e-8
    ratio = r_int_mean / r_ext_mean
    if ratio > 5.0:
        print(f"[WARN] r_int/r_ext = {ratio:.1f} — intrinsic is taking over")
    if r_ext_mean < 1e-6 and len(r_ext_buf) > 5 * window:
        print("[WARN] extrinsic stream silent for 5 windows — check env wiring")
    return ratio
```

The practical recipe I have settled on: always normalise $r^i$ by a running standard deviation, log the ratio above every 10k steps, and start with $\eta = 0.01$ (RND) or $\eta = 0.5$ on the *normalised* intrinsic reward (ICM). If the agent stops making extrinsic progress for 20% of the training budget, halve $\eta$ before changing anything else. Curiosity is a hyperparameter regime, not a free lunch.

## Summary and what comes next

Exploration is the bottleneck that separates toy reinforcement learning from anything resembling general intelligence. Random exploration scales catastrophically badly — not because the math is wrong, but because the universe of possible states is unimaginably larger than what uniform sampling can cover.

The unifying idea behind every modern advance:

> **Treat curiosity as a learnable reward.** Define "novelty" operationally — as low visit count, high prediction error, large distillation residual, or distance from episodic memory — and let the agent maximise it alongside the task reward.

We saw four concrete instantiations:

- **Count-based / pseudo-count** — elegant in tabular MDPs, fragile in pixels.
- **ICM** — prediction error in a learned, action-relevant feature space.
- **RND** — distillation error of a random network; embarrassingly simple, embarrassingly effective.
- **NGU / Agent57** — episodic and lifetime novelty multiplied together for tasks that punish forgetting.

The exploration problem is far from closed. Current methods still need on the order of $10^8$-$10^9$ environment frames; the human brain solves Montezuma's Revenge in a few hours. Active research directions include skill discovery, language-grounded exploration, and learning exploration policies from human demonstrations.

**Coming up next:** [Part 5](/en/reinforcement-learning/05-model-based-rl-and-world-models/) introduces **Model-Based RL and World Models** — learning a differentiable simulator of the environment so the agent can "dream" thousands of imaginary trajectories per real interaction.

---

## References

- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. *ICLR*. [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
- Badia, A. P., Sprechmann, P., Vitvitskyi, A., et al. (2020). Never Give Up: learning directed exploration strategies. *ICLR*. [arXiv:2002.06038](https://arxiv.org/abs/2002.06038)
- Badia, A. P., Piot, B., Kapturowski, S., et al. (2020). Agent57: outperforming the Atari human benchmark. *ICML*. [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)
- Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying count-based exploration and intrinsic motivation. *NeurIPS*. [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J. (2021). First return, then explore. *Nature*, 590, 580-586. [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multi-armed bandit problem. *Machine Learning*, 47, 235-256.
