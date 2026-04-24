---
title: "Reinforcement Learning (4): Exploration Strategies and Curiosity-Driven Learning"
date: 2025-06-27 09:00:00
tags:
  - Reinforcement Learning
  - Exploration Strategies
  - Intrinsic Reward
  - Curiosity-Driven Learning
  - ICM
  - RND
  - NGU
  - Count-Based Methods
categories:
  - Reinforcement Learning
series:
  name: "Reinforcement Learning"
  part: 4
  total: 12
lang: en
mathjax: true
description: "How do RL agents discover rewards when the environment gives almost no feedback? From count-based methods to ICM, RND, and NGU -- the science of curiosity-driven exploration."
disableNunjucks: true
---

Drop a fresh agent into Montezuma's Revenge. To score a single point it must walk to the right, jump a skull, climb a rope, leap to a platform, and grab a key -- roughly **a hundred precise actions in a row**. Until that key is collected, every reward signal is exactly zero.

A textbook DQN with $\varepsilon=0.1$ exploration has, by a generous estimate, a $0.1^{100} \approx 10^{-100}$ chance of stumbling onto that key by accident. Unsurprisingly, vanilla DQN scores **0** on this game. Not "low" -- literally zero, every episode, for the entire training run.

This is the **sparse-reward problem**, and it exposes an uncomfortable truth: a deep RL algorithm is only as good as its exploration strategy. Even the finest Bellman backup is useless if the agent never observes a non-zero reward to back up. This chapter walks the path from blind random exploration to **curiosity-driven learning** -- algorithms that manufacture their own rewards for discovering anything new.

## What you will learn

- Why $\varepsilon$-greedy, Boltzmann, and even UCB collapse in high-dimensional environments
- **Count-based** methods and pseudo-counts via density models
- **ICM** (Intrinsic Curiosity Module): prediction error in a learned feature space
- **RND** (Random Network Distillation): the simplest curiosity signal that actually works at scale
- **NGU** (Never Give Up): episodic memory for tasks that punish forgetting
- Practical PPO + curiosity recipes, hyper-parameters, and failure modes

**Prerequisites:** [Part 1-3](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) (MDPs, DQN, policy gradients, PPO basics).

---

## 1. Why exploration is so hard

### 1.1 Classical schedules and what they actually look like

Every introductory RL course starts with **$\varepsilon$-greedy**: with probability $\varepsilon$ pick a uniformly random action, otherwise pick the greedy one. The hard part is not the formula -- it is the *schedule*: how should $\varepsilon$ decay as training progresses?

![Epsilon-greedy decay schedules and induced action probabilities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig1_epsilon_greedy_decay.png)

The figure above shows three popular schedules (linear, exponential, piecewise step) on the left, and on the right what the actual action distribution looks like under linear decay with four available actions. Three observations are worth internalising:

1. **Different schedules give very different "exploration budgets."** Exponential decay spends most of its random actions in the first 20k steps; linear decay spreads them more evenly; step schedules behave like crude curricula.
2. **Even at $\varepsilon = 0.05$ a quarter of the random actions still go to the greedy choice** ($1 - \varepsilon + \varepsilon/|\mathcal{A}|$), which surprises people who expect 5% noise to mean 5% off-policy behaviour.
3. **Nothing in any of these curves looks at the state.** Exploration is purely a function of training step. That is the central weakness we are about to attack.

Mathematically:

$$
\pi_\varepsilon(a \mid s) = \begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & a = \arg\max_{a'} Q(s, a') \\[4pt]
\dfrac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

### 1.2 Boltzmann (softmax) exploration: a marginal upgrade

Instead of an all-or-nothing random kick, **Boltzmann** weights actions by their Q-values:

$$
\pi_\tau(a \mid s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}.
$$

The temperature $\tau$ replaces $\varepsilon$ as the exploration knob. As $\tau \to 0$ the policy becomes greedy; as $\tau \to \infty$ it becomes uniform.

![Boltzmann action distribution and policy entropy for several temperatures](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig2_boltzmann_softmax.png)

The right panel plots policy entropy $H(\pi_\tau) = -\sum_a \pi_\tau(a) \log \pi_\tau(a)$ against $\tau$. The curve flattens to the maximum value $\ln |\mathcal{A}|$ surprisingly fast: by $\tau = 2$ you are already above 90 % of maximum entropy. This is exactly the **entropy bonus** trick used inside PPO and SAC -- it is Boltzmann exploration, repackaged as a regulariser on the policy network.

But Boltzmann shares $\varepsilon$-greedy's fatal flaw: it spreads probability based on the agent's *current Q estimates*, not on any notion of how much it has actually visited each region of the state space. Two states the agent has never seen still get the same softmax over the same untrained Q-values.

### 1.3 UCB: the principled approach that does not scale

For multi-armed bandits, the classical **UCB1** rule is provably near-optimal:

$$
a_t = \arg\max_a \left[ \hat Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right].
$$

The first term *exploits*; the second *explores* -- arms pulled fewer times get a larger uncertainty bonus.

![UCB1 score decomposition and arm-pull statistics over a 5-arm bandit](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig3_ucb_bandit.png)

Watch what happens in the figure above. At $t = 50$ the bonus dominates (orange bars), all arms look attractive, and pulls are spread across the bandit. By $t = 1000$ the uncertainty bonus has shrunk for the optimal arm 3 (it has been pulled hundreds of times) but stays high for the rest, so the algorithm settles into a *near-greedy regime* on arm 3 while still occasionally checking the others. This is exploration done right -- and it is *guided by data*, not by a pre-baked schedule.

So why do we not just use UCB everywhere? **Because $N(s,a)$ is meaningless in high-dimensional state spaces.** In Atari each frame is $84 \times 84 \times 4 = 28{,}224$ pixels; the agent will essentially never see the same state twice. So $N(s, a) = 1$ for almost every encountered state-action pair, and the bonus becomes a useless constant.

### 1.4 Thompson sampling: posterior beliefs over rewards

A close cousin of UCB is **Thompson sampling**: maintain a posterior over each arm's reward parameter, sample one possible world from that posterior, and pick the action that is best in that sampled world. For Bernoulli arms with a Beta prior the update is delightfully simple -- on success increment $\alpha$, on failure increment $\beta$.

![Beta posteriors for three Bernoulli arms after 10, 50, and 300 pulls](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig4_thompson_sampling.png)

The figure shows posteriors sharpening around the true reward rates (dashed). After 10 pulls the algorithm has barely committed; by 300 pulls the posterior on the best arm (arm 2, $\mu = 0.75$) is a tight spike, while losing arms still carry enough variance to keep them in the running. Thompson sampling tends to match or beat UCB in practice and is widely used inside contextual-bandit recommender systems. As an exploration strategy in deep RL, however, it suffers the same generalisation problem as UCB: maintaining a posterior over a $10^{60}$-dimensional state-action space is intractable, and *Bayesian* deep RL approaches (e.g. **Bootstrapped DQN**, **Bayes by Backprop**) only partially recover its benefits.

### 1.5 Four reasons exploration is hard in deep RL

Putting the pieces together, the difficulty of exploration in real environments compounds along four axes:

1. **Sparse rewards.** Hundreds or thousands of correct actions before any external signal arrives. Random exploration cannot find a needle in this haystack.
2. **Combinatorial state spaces.** $256^{28224}$ possible Atari frames; you cannot count visits if you never see the same thing twice.
3. **Local optima.** Small early rewards (a coin in a dead-end corridor) can permanently distract the agent from the much larger reward behind a hard-to-find door.
4. **The noisy-TV problem.** A naive novelty signal mistakes random pixel noise for "new and interesting." Drop a TV showing static into the room and the agent will sit and watch it forever.

The conceptual leap of the modern era is to stop computing exploration as a *function of step number* and start computing it as a *function of the agent's own experience*. Babies do this naturally: they play with new toys, not old ones.

---

## 2. The curiosity blueprint: intrinsic rewards

Every modern method we will discuss adds a **intrinsic reward** $r^{\text{int}}_t$ on top of the environment's external reward:

$$
r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t.
$$

The agent is then trained on $r^{\text{total}}$ with whatever RL algorithm you like (DQN, PPO, IMPALA, ...). All the design effort goes into a single question:

> **How do we compute $r^{\text{int}}_t$ so it is large for genuinely novel experiences and small for stale ones, without requiring us to count pixels?**

We now look at three increasingly elegant answers.

---

## 3. Count-based methods: when counting still works

The cleanest definition of novelty is "states I have visited fewer times." For tabular MDPs this gives the MBIE-EB bonus

$$
r^{\text{int}}(s) = \frac{\beta}{\sqrt{N(s)}},
$$

which has matching theoretical guarantees to UCB. The headache is that $N(s) = 0$ for almost every state in pixel-based environments.

Bellemare et al. (2016) replaced the literal count with a **pseudo-count** derived from a density model $\rho(s)$:

$$
\hat N(s) = \frac{\rho(s)\bigl(1 - \rho_{\text{new}}(s)\bigr)}{\rho_{\text{new}}(s) - \rho(s)},
$$

where $\rho_{\text{new}}$ is the model's density after training on one extra observation of $s$. The construction pretends $\rho$ is the empirical distribution of a giant counter and inverts it. With a strong density model (PixelCNN, neural autoregressive models) this delivers the first non-trivial scores on Montezuma's Revenge -- but the model is expensive to train, fragile, and the noisy-TV problem is still wide open: random pixel noise looks low-density, hence high-novelty, hence highly rewarded. Modern systems abandoned pseudo-counts for the more robust prediction-error methods we cover next.

---

## 4. ICM: curiosity through prediction error

**Intrinsic Curiosity Module (ICM)** (Pathak et al., ICML 2017) replaces "have I seen this before?" with "can I predict what happens next?". The intuition:

- If the environment's response to my action is *predictable*, I already understand it -- low intrinsic reward.
- If the response *surprises* me, I have found something I do not yet model -- high intrinsic reward.

But predicting raw pixels is a bad idea: TV static is unpredictable yet useless. ICM's brilliant move is to predict in a **learned feature space** $\phi$ that only encodes information *the agent's actions can affect*.

![ICM and RND architectures side by side](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig5_icm_rnd_architecture.png)

The left half of the figure above shows the three components:

1. **Encoder $\phi$** (CNN) maps $s_t$ to a feature vector $\phi(s_t)$.
2. **Forward model $\hat f$** predicts the next features from current features and action: $\hat\phi_{t+1} = \hat f(\phi(s_t), a_t)$. Its squared error
   $$r^{\text{int}}_t = \eta \,\bigl\| \hat\phi_{t+1} - \phi(s_{t+1}) \bigr\|^2$$
   *is* the intrinsic reward.
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

On Montezuma's Revenge, ICM + A3C reaches roughly **6,600** points within 25 M frames (vanilla DQN sits at 0 forever). Even more striking, an agent trained with **zero external reward** -- only ICM's intrinsic signal -- learns to navigate the first few rooms, dodge enemies, and pick up keys, simply because doing those things keeps its forward model on its toes.

### Where ICM falls down

- **Stochastic environments.** If a slot machine in the environment has a genuinely random outcome, the forward model can never predict it, and the agent fixates on pulling the lever. The inverse-model trick filters action-independent randomness, but action-*dependent* randomness still bites.
- **Compute.** Three networks (encoder, forward, inverse) on top of your policy; expect roughly 2x training cost compared to vanilla PPO.

---

## 5. RND: a startlingly simple alternative

**Random Network Distillation** (Burda et al., ICLR 2019) replaces the entire forward/inverse machinery with a single observation:

> *Distil a fixed random network. Wherever the predictor still has high error, the agent has not been there enough.*

Concretely, RND keeps two networks (right half of the architecture figure above):

- A **target network $f$**, with random weights, **frozen** for the entire training run.
- A **predictor network $\hat f$**, trained by gradient descent to minimise $\| \hat f(s) - f(s) \|^2$ on observed states.

Intrinsic reward is the predictor's residual:

$$
r^{\text{int}}(s) = \bigl\| \hat f(s) - f(s) \bigr\|^2.
$$

For a state the predictor has seen many times, training has driven the loss to near zero -- low reward. For a novel state, the predictor has never been trained there, so its output is random and the residual is large -- high reward. The frozen target acts like a deterministic hash function: structurally similar states map to similar targets, so generalisation comes for free.

The same trick neutralises the noisy-TV problem. Random static frames are visually different but **structurally similar** in the eyes of a random CNN; the predictor learns to match them after a handful of updates and the reward decays to zero. ICM and RND therefore handle "noisy TV" by completely different mechanisms -- ICM by filtering the *features*, RND by exploiting the *consistency of a random map*.

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

- **Montezuma's Revenge**: 8,152 average score -- the first algorithm to surpass the human expert benchmark of 7,385.
- **Pitfall**: 70.4 (previous best: exactly 0).
- RND was, for a year or two, the strongest reported method on every "hard exploration" Atari game.

The two non-obvious tricks that make RND work: (i) **two value heads** in the policy, one for $r^{\text{ext}}$ and one for $r^{\text{int}}$, with separate discount factors; (ii) **per-environment normalisation** of intrinsic rewards using a running estimate of their standard deviation. Without (ii) the intrinsic-reward scale grows wildly, drowns out the external reward, and training collapses.

---

## 6. NGU: never give up on a state

RND has a quiet assumption: once a state's predictor error has been driven to zero, that state is never novel again. Most of the time this is what we want. But two important cases break it:

- **Key-door tasks.** The agent needs to *re-pick-up* a key every time it dies and respawns. After the first pickup, RND considers the key location boring; the agent never goes back for it.
- **Backtracking.** After exploring the right half of a level, the agent must return through the start and head left. The start has been visited thousands of times; RND sees no reason to ever go through it again.

**Never Give Up (NGU)** (Badia et al., ICLR 2020) repairs this by combining two novelty signals:

$$
r^{\text{int}}_t = r^{\text{episodic}}_t \cdot \min\bigl(r^{\text{lifetime}}_t,\; L\bigr).
$$

- **Episodic novelty** $r^{\text{episodic}}$. Maintain an episodic memory of state embeddings *for the current episode only*. The reward is large when the current state is far (in embedding space) from anything in this memory. Crucially, the memory **resets on each new episode**, so even a state that has been visited millions of times across training still feels novel within a fresh run.
- **Lifetime novelty** $r^{\text{lifetime}}$. The classic RND signal, capped at $L$ to prevent run-away.

The multiplication insists on both: a state must be locally novel *and* globally not yet exhausted to score high. NGU also introduces a **family of policies** with different exploration coefficients trained in parallel, and a **directed exploration** scheme on top -- but the episodic-lifetime decomposition is the heart of the idea.

The successor, **Agent57** (Badia et al., 2020), bolts NGU onto a meta-controller that picks which exploration policy to use at each moment. It is the first single algorithm to surpass the human baseline on **all 57 Atari games**.

### Highlight scores

| Game | DQN (ε-greedy) | RND | NGU | Human expert |
|------|---------------:|----:|----:|-------------:|
| Montezuma's Revenge | 0 | 8,152 | 11,000+ | 7,385 |
| Pitfall! | 0 | 70 | 5,000+ | 6,464 |
| Private Eye | 0 | 8,800 | 69,000 | 69,571 |

---

## 7. Visualising the gap: random vs curious agents

It is worth seeing, on a tiny problem, just how different "random" and "curious" exploration look.

![Visit-count heatmaps for ε-greedy and curiosity-driven agents on a 25x25 grid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig7_trajectory_comparison.png)

Both panels show 1,500 steps of an agent starting at the white star in a 25 x 25 GridWorld. The blue agent is a uniform random walker; the purple agent samples its next move with probability proportional to $1/\sqrt{N(s')}$ over neighbours -- the simplest possible count-based curiosity reward. After the same number of steps, the random walker has covered only **65.6 %** of the grid and revisited its favourite cell **19** times; the curious agent has covered **80 %** with a maximum revisit count of **11**. The right panel makes the same point with a sorted log-scale visit-count distribution: curiosity flattens the head of the distribution (no over-visited cells) and lifts the tail (no neglected cells).

Now multiply that effect over 100 million Atari frames in a state space the size of an exoplanet and you understand why curiosity is the difference between scoring zero and scoring eleven thousand.

![Score curves and time-to-first-reward on Montezuma's Revenge](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/04-exploration-and-curiosity-driven-learning/fig6_sparse_reward_montezuma.png)

The figure above tells the same story at scale. Vanilla DQN stays at zero forever -- it never finds a single reward. Each successive generation of curiosity (count-based -> ICM -> RND -> NGU) shaves frames off the time-to-first-reward and lifts the asymptotic score. NGU is the first to clear the human-expert dashed line.

---

## 8. Practical recipes: PPO + curiosity in production

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

## 9. Summary and what comes next

Exploration is the bottleneck that separates toy reinforcement learning from anything resembling general intelligence. Random exploration scales catastrophically badly -- not because the math is wrong, but because the universe of possible states is unimaginably larger than what uniform sampling can cover.

The unifying idea behind every modern advance:

> **Treat curiosity as a learnable reward.** Define "novelty" operationally -- as low visit count, high prediction error, large distillation residual, or distance from episodic memory -- and let the agent maximise it alongside the task reward.

We saw four concrete instantiations:

- **Count-based / pseudo-count** -- elegant in tabular MDPs, fragile in pixels.
- **ICM** -- prediction error in a learned, action-relevant feature space.
- **RND** -- distillation error of a random network; embarrassingly simple, embarrassingly effective.
- **NGU / Agent57** -- episodic and lifetime novelty multiplied together for tasks that punish forgetting.

The exploration problem is far from closed. Current methods still need on the order of $10^8$-$10^9$ environment frames; the human brain solves Montezuma's Revenge in a few hours. Active research directions include skill discovery, language-grounded exploration, and learning exploration policies from human demonstrations.

**Coming up next:** [Part 5](/en/reinforcement-learning-5-model-based-rl-and-world-models/) introduces **Model-Based RL and World Models** -- learning a differentiable simulator of the environment so the agent can "dream" thousands of imaginary trajectories per real interaction.

---

## References

- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. *ICLR*. [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
- Badia, A. P., Sprechmann, P., Vitvitskyi, A., et al. (2020). Never Give Up: learning directed exploration strategies. *ICLR*. [arXiv:2002.06038](https://arxiv.org/abs/2002.06038)
- Badia, A. P., Piot, B., Kapturowski, S., et al. (2020). Agent57: outperforming the Atari human benchmark. *ICML*. [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)
- Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying count-based exploration and intrinsic motivation. *NeurIPS*. [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J. (2021). First return, then explore. *Nature*, 590, 580-586. [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multi-armed bandit problem. *Machine Learning*, 47, 235-256.

---

## Series Navigation

| Part | Topic |
|------|-------|
| 1 | [Fundamentals and Core Concepts](/en/reinforcement-learning-1-fundamentals-and-core-concepts/) |
| 2 | [Q-Learning and DQN](/en/reinforcement-learning-2-q-learning-and-dqn/) |
| 3 | [Policy Gradient and Actor-Critic](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/) |
| **4** | **Exploration and Curiosity-Driven Learning (you are here)** |
| 5 | [Model-Based RL and World Models](/en/reinforcement-learning-5-model-based-rl-and-world-models/) |
| 6 | [PPO and TRPO](/en/reinforcement-learning-6-ppo-and-trpo/) |
