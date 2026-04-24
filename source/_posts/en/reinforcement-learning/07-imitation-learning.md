---
title: "Reinforcement Learning (7): Imitation Learning and Inverse RL"
date: 2024-07-07 09:00:00
tags:
  - Reinforcement Learning
  - Imitation Learning
  - Behavioral Cloning
  - Inverse RL
  - GAIL
categories: Reinforcement Learning
series: Reinforcement Learning
part: 7
total_parts: 12
lang: en
mathjax: true
description: "A practical, theory-grounded tour of imitation learning: behavioral cloning and its quadratic compounding error, DAgger and the no-regret reduction, MaxEnt inverse RL for recovering reward functions, and adversarial methods (GAIL, AIRL). Includes runnable PyTorch code, a method-selection ladder, and seven publication-quality figures."
disableNunjucks: true
---

Every algorithm in the previous chapters assumed access to a reward function. In practice, _designing_ that reward is often the hardest part of an RL project. Try writing one paragraph that captures "drive like a careful human", "fold a shirt the way a tailor would", or "summarise this document the way an expert editor would". You can _show_ those behaviours far more easily than you can _specify_ them.

Imitation learning takes that intuition seriously: instead of optimising a hand-engineered scalar, it learns from expert demonstrations $\mathcal{D} = \{(s_t, a_t)\}$. This chapter walks the four canonical methods -- behavioral cloning, DAgger, maximum-entropy IRL, and GAIL/AIRL -- not as isolated tricks but as a single ladder where each rung relaxes one assumption and pays for it with new structure.

## What you will learn

- **Behavioral cloning (BC)**: imitation as supervised learning, why it works on short tasks, and the precise reason it breaks on long ones.
- **DAgger**: how interactive relabelling turns BC's quadratic error into a linear one, with the no-regret theorem behind it.
- **Maximum-entropy IRL**: recovering an interpretable reward whose optimum reproduces the demonstrations.
- **GAIL and AIRL**: matching expert _occupancy measures_ end-to-end through adversarial training.
- **A selection rubric**: which method to reach for given expert availability, environment access, and need for transfer.

**Prerequisites**: policy gradients ([Part 5](/en/reinforcement-learning-5-policy-gradients/)) and PPO ([Part 6](/en/reinforcement-learning-6-advanced-policy-optimization/)). PyTorch fundamentals are assumed for the code snippets.

---

## 1. Problem setting

Given expert demonstrations

$$
\mathcal{D} = \{(s_1, a_1), (s_2, a_2), \ldots, (s_N, a_N)\},
$$

the goal is to learn a policy $\pi_\theta$ whose behaviour is close to that of an unknown expert policy $\pi^*$. We never observe $\pi^*$ directly -- only its samples -- and there is no reward signal. Sometimes we can also _query_ the expert on new states (DAgger); often we cannot.

| Aspect | Reinforcement learning | Imitation learning |
|---|---|---|
| Signal | Reward $r(s, a)$ | Expert dataset $\mathcal{D}$ |
| Interaction | Required (must explore) | Sometimes optional (offline BC) |
| Objective | Maximise $\mathbb{E}[\sum \gamma^t r_t]$ | Match expert behaviour distribution |
| Failure mode | Reward hacking, exploration | Distribution shift, mode collapse |
| Typical use | Game-playing, robotics, RLHF | Driving, surgery, language style |

The methods we cover trade these axes against each other. The five-rung ladder is summarised below; the rest of the chapter unpacks each rung.

![Imitation learning method hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig6_method_hierarchy.png)

---

## 2. Behavioral cloning

The simplest imitation algorithm is also the most-deployed one: treat $\mathcal{D}$ as a supervised dataset and minimise

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{(s,a)\sim \mathcal{D}}\big[ \ell\big(\pi_\theta(s),\, a\big) \big],
$$

where $\ell$ is cross-entropy for discrete actions or MSE / negative log-likelihood for continuous ones. The historical first instance is **ALVINN** (Pomerleau, 1989), which trained a fully connected network to steer a van from camera images.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class BehavioralCloning:
    """Supervised imitation: minimise loss between policy and expert actions."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256),
                 lr=1e-3, continuous=False, dropout=0.1):
        self.continuous = continuous
        layers, prev = [], state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        if continuous:
            layers += [nn.Linear(prev, action_dim), nn.Tanh()]
            self.criterion = nn.MSELoss()
        else:
            layers += [nn.Linear(prev, action_dim)]
            self.criterion = nn.CrossEntropyLoss()
        self.policy = nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.state_mean = self.state_std = None

    def fit(self, states, actions, epochs=100, batch_size=64, val_frac=0.1):
        # Standardise states; large input scales destabilise BC.
        self.state_mean = states.mean(0)
        self.state_std = states.std(0) + 1e-8
        S = (states - self.state_mean) / self.state_std

        n_val = int(len(S) * val_frac)
        idx = np.random.permutation(len(S))
        S_tr, S_val = S[idx[n_val:]], S[idx[:n_val]]
        A_tr, A_val = actions[idx[n_val:]], actions[idx[:n_val]]

        S_tr = torch.as_tensor(S_tr, dtype=torch.float32)
        S_val = torch.as_tensor(S_val, dtype=torch.float32)
        if self.continuous:
            A_tr = torch.as_tensor(A_tr, dtype=torch.float32)
            A_val = torch.as_tensor(A_val, dtype=torch.float32)
        else:
            A_tr = torch.as_tensor(A_tr, dtype=torch.long)
            A_val = torch.as_tensor(A_val, dtype=torch.long)

        loader = DataLoader(TensorDataset(S_tr, A_tr),
                            batch_size=batch_size, shuffle=True)
        best, best_state = float("inf"), None
        for epoch in range(epochs):
            self.policy.train()
            for s, a in loader:
                loss = self.criterion(self.policy(s), a)
                self.optim.zero_grad(); loss.backward(); self.optim.step()
            self.policy.eval()
            with torch.no_grad():
                v = self.criterion(self.policy(S_val), A_val).item()
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in
                                       self.policy.state_dict().items()}
        self.policy.load_state_dict(best_state)

    def act(self, state):
        s = (state - self.state_mean) / self.state_std
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            out = self.policy(s)
        return out.squeeze().numpy() if self.continuous else out.argmax(-1).item()
```

Three implementation details matter much more than they look:

1. **Standardise inputs.** BC is a small supervised model -- unscaled features dominate the loss surface and produce overconfident actions in rare states.
2. **Early-stopping on a held-out validation set.** Long training overfits the expert's noise, which makes the next problem worse, not better.
3. **Action representation.** For continuous control, predicting Gaussian parameters with a NLL loss outperforms MSE on Tanh outputs whenever the expert is multimodal.

### 2.1 Why BC fails on long horizons

BC is trained under the expert's state distribution $d_{\pi^*}$ but, at deployment, it visits states drawn from its _own_ distribution $d_{\pi_\theta}$. Because the policy is imperfect, those distributions diverge with every step.

![BC vs DAgger state distributions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig1_bc_vs_dagger.png)

The classical bound formalises the cascade. If $\pi_\theta$ has expected per-step error $\varepsilon$ on the expert distribution, then over a $T$-step rollout the worst-case total error is

$$
J(\pi^*) - J(\pi_\theta) \;\le\; \mathcal{O}\!\left( \varepsilon \, T^2 \right),
$$

quadratic in horizon (Ross & Bagnell, 2010). Even a 99% accurate policy fails on a 200-step task, because the 1% probability of a mistake compounds and the policy lands in states the expert never visited -- where it has no useful training signal.

You can see the cascade clearly when you roll the trained policy forward:

![Expert vs learner trajectories](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig2_expert_trajectory_comparison.png)

Early in the episode the BC learner stays on the expert corridor. Around the midpoint a small error appears, the policy enters off-distribution states, and from there it drifts steadily into the hazard zone. The same compounding effect appears in the per-step error curve below.

![Compounding error in BC](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig5_compounding_error.png)

---

## 3. DAgger: dataset aggregation

DAgger (Ross, Gordon & Bagnell, 2011) breaks the cascade by collecting expert labels _on the states the learner actually visits_. Each iteration adds a new slice of $(s, a^*)$ pairs sampled from $d_{\pi_\theta}$ and retrains on the aggregated dataset.

**Algorithm.**

1. Train $\pi_1$ on initial expert demonstrations.
2. For $i = 1, \ldots, N$:
   - Mix $\pi_i$ with the expert under a schedule $\beta_i \to 0$ to roll out states.
   - Query the expert for the correct action $a^*$ at every visited state.
   - Aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, a^*)\}$.
   - Retrain $\pi_{i+1}$ on all of $\mathcal{D}$.

**No-regret guarantee.** Treat the sequence of policies as a no-regret online learner against the loss family induced by the visited distributions. Then

$$
J(\pi^*) - J(\pi_{\hat{i}}) \;\le\; \mathcal{O}\!\left( \varepsilon \, T \right) + \mathcal{O}\!\left( \tfrac{T \sqrt{\log N}}{\sqrt{N}} \right),
$$

i.e. _linear_ in horizon. The right way to read this is: every additional iteration buys back the recovery information that BC discarded.

```python
class DAgger:
    """DAgger = BC + iterative relabelling on learner-visited states."""

    def __init__(self, state_dim, action_dim, **bc_kwargs):
        self.bc = BehavioralCloning(state_dim, action_dim, **bc_kwargs)
        self.S, self.A = [], []

    def train(self, env, expert, n_iters=10,
              n_init=50, n_per_iter=20):
        # Iteration 0: pure expert rollouts (warm start)
        s0, a0 = self._collect(env, expert, n_init, beta=1.0)
        self.S += s0; self.A += a0
        self.bc.fit(np.array(self.S), np.array(self.A))

        for i in range(1, n_iters + 1):
            beta = max(0.0, 1.0 - i / n_iters)   # fade expert from 1 -> 0
            s, a = self._collect(env, expert, n_per_iter, beta=beta)
            self.S += s; self.A += a
            self.bc.fit(np.array(self.S), np.array(self.A))

    def _collect(self, env, expert, n_episodes, beta):
        states, actions = [], []
        for _ in range(n_episodes):
            s, done = env.reset(), False
            while not done:
                # Mixed action choice; expert label is always recorded.
                a_play = expert(s) if np.random.rand() < beta else self.bc.act(s)
                a_label = expert(s)
                states.append(s); actions.append(a_label)
                s, _, done, _ = env.step(a_play)
        return states, actions
```

**When DAgger applies.** DAgger needs an expert that can be queried _at run-time_. That is realistic when:

- the expert is an algorithmic planner (e.g. an MPC controller, a search-based oracle);
- the expert is a more capable model you want to distil (e.g. teacher-student RL, model-based oracle, language-model self-distillation);
- a human is in the loop and willing to label batches.

It does _not_ apply when the only demonstrations are a static log -- e.g. a recorded driving dataset. For that regime, jump to GAIL.

---

## 4. Inverse reinforcement learning

BC and DAgger both _imitate the action_. IRL asks a deeper question: **why** is that action good? It posits that the expert is (approximately) optimising some unknown reward $r^*$, recovers a candidate $\hat r$ from the demonstrations, and then runs standard RL under $\hat r$.

Why bother going through reward? Two reasons:

- **Interpretability.** $\hat r$ tells you _what the expert cares about_, which is auditable and editable.
- **Transfer.** A reward function survives changes to dynamics, embodiment, and start-state distribution that would break a behaviour-level model.

![Inverse RL recovers reward from behaviour](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig3_irl_recovery.png)

### 4.1 Maximum-entropy IRL

A purely "match the expert's value" objective is ill-posed -- many rewards explain the same behaviour. **Maximum-entropy IRL** (Ziebart et al., 2008) breaks the tie by requiring the recovered policy to maximise $r$ subject to maximum entropy. Concretely, the expert's distribution over trajectories $\tau$ takes the Boltzmann form

$$
p_\theta(\tau) \;\propto\; \exp\!\left( \sum_t r_\theta(s_t, a_t) \right).
$$

Maximising the log-likelihood of the demonstrations gives the elegant gradient

$$
\nabla_\theta \mathcal{L}(\theta) \;=\; \mathbb{E}_{\tau \sim \pi^*}\!\left[\nabla_\theta r_\theta(\tau)\right] \;-\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta r_\theta(\tau)\right].
$$

Read this as a contrastive update: _push reward up where the expert goes, push it down where the current policy goes_. At convergence, the two expectations match and the policy reproduces the expert occupancy.

The cost is the second expectation. Computing $\mathbb{E}_{\pi_\theta}$ requires solving an RL problem (or doing trajectory sampling) at every reward update -- a nested loop that limited classical IRL to small grid worlds. **Guided cost learning** (Finn, Levine & Abbeel, 2016) replaces the inner RL solve with sampled importance-weighted trajectories, scaling MaxEnt IRL to continuous control.

### 4.2 Reward ambiguity

Even with the max-entropy regulariser, $\hat r$ is recovered up to **shaping invariances**: adding a potential function $\Phi(s') - \Phi(s)$ leaves the optimal policy unchanged but changes $r$. The recovered reward is therefore a useful _ranking_ over states, not an absolute scale. Adversarial inverse RL (§5.2) explicitly disentangles the shaping component.

---

## 5. Adversarial imitation: GAIL and AIRL

The IRL inner loop is expensive. **GAIL** (Ho & Ermon, 2016) noticed that for imitation we don't actually need $r$ -- we only need the policy whose state-action _occupancy_ matches the expert's. So GAIL replaces "recover reward, then re-solve RL" with a single adversarial game.

A discriminator $D_\phi(s, a)$ tries to tell expert pairs from policy pairs; the policy tries to fool it. The minimax objective is

$$
\min_\theta \max_\phi \;\;
\mathbb{E}_{(s,a)\sim \pi^*}\!\left[\log D_\phi(s, a)\right]
+ \mathbb{E}_{(s,a)\sim \pi_\theta}\!\left[\log\big(1 - D_\phi(s, a)\big)\right]
- \lambda H(\pi_\theta).
$$

The entropy term $\lambda H(\pi_\theta)$ stabilises the generator and discourages mode collapse. At the saddle point, $\pi_\theta$'s occupancy measure equals $\pi^*$'s -- which is exactly what BC was trying to do but without the supervision-vs-rollout mismatch.

![GAIL discriminator-generator architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig4_gail_architecture.png)

In practice, the discriminator's score doubles as the per-step reward fed to a PPO update of the policy:

$$
\hat r(s, a) \;=\; -\log\!\big(1 - D_\phi(s, a)\big) \quad\text{or}\quad \hat r(s, a) \;=\; -\log D_\phi(s, a).
$$

Both forms appear in the literature; the first is the reward used in the original GAIL paper, the second has lower variance early in training.

```python
class GAIL:
    """Adversarial imitation via a learned discriminator reward."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, continuous=False):
        self.continuous = continuous
        in_dim = state_dim + action_dim
        self.discriminator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(),
                                           lr=3e-4)
        # Policy / value networks: any standard PPO setup works (omitted).

    def _sa(self, s, a):
        if not self.continuous:
            a = torch.nn.functional.one_hot(a.long(), self.action_dim).float()
        return torch.cat([s, a], dim=-1)

    def reward(self, states, actions):
        """Per-step imitation reward used by the policy learner."""
        with torch.no_grad():
            logits = self.discriminator(self._sa(states, actions))
            d = torch.sigmoid(logits)
            r = -torch.log(1.0 - d + 1e-8)   # original GAIL reward
        return r.squeeze(-1)

    def update_discriminator(self, expert_sa, policy_sa):
        # Binary cross entropy with labels: expert=0, policy=1.
        e_logits = self.discriminator(expert_sa)
        p_logits = self.discriminator(policy_sa)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        loss = bce(e_logits, torch.zeros_like(e_logits)) + \
               bce(p_logits, torch.ones_like(p_logits))
        self.disc_optim.zero_grad(); loss.backward(); self.disc_optim.step()
        return loss.item()
```

**Tuning notes.** Three failure modes dominate GAIL in practice:

- _Discriminator wins too quickly._ Gradient through the policy reward vanishes. Mitigations: clip discriminator logits, use spectral normalisation, or enforce a Lipschitz constraint (WGAIL).
- _Reward signal collapses._ As $D \to 0$ on policy samples, the reward drifts to zero. Reward normalisation per batch (running mean / std) restores learning.
- _Policy mode-collapses._ Increase the entropy bonus $\lambda$ or use a maximum-entropy actor (SAC-style) as the generator.

### 5.1 What does "matching occupancy" actually mean?

GAIL is not minimising an action-prediction loss; it is minimising the Jensen-Shannon divergence between the expert occupancy $\rho_{\pi^*}(s, a)$ and the learner occupancy $\rho_{\pi_\theta}(s, a)$. That is a much stronger objective than BC -- it is _aware_ of the rollout distribution. The price: it requires environment interaction during training (to sample $\rho_{\pi_\theta}$), so it does not work in fully offline settings without modification.

### 5.2 AIRL: disentangling reward from shaping

The discriminator GAIL learns is great for imitation but is _not_ a reusable reward. **AIRL** (Fu, Luo & Levine, 2018) restructures the discriminator as

$$
D_\phi(s, a, s') \;=\; \frac{\exp\big(f_\phi(s, a, s')\big)}{\exp\big(f_\phi(s, a, s')\big) + \pi_\theta(a \mid s)},
\qquad
f_\phi(s, a, s') \;=\; r_\psi(s) + \gamma \Phi_\xi(s') - \Phi_\xi(s),
$$

so that $r_\psi$ is the _state-only_ reward and $\Phi_\xi$ absorbs the shaping. Training with this parameterisation yields a recovered reward $\hat r = r_\psi$ that transfers across dynamics changes -- the empirical headline result of the AIRL paper.

---

## 6. Sample efficiency: imitation vs RL

The strongest practical argument for imitation is sample efficiency. A few thousand expert demonstrations can substitute for tens of millions of environment interactions, especially in high-dimensional control.

![Sample efficiency: imitation vs RL](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/07-imitation-learning/fig7_sample_efficiency.png)

The pattern in the left panel is robust across robotics benchmarks: BC saturates below the expert (it cannot exceed its training data), DAgger climbs once it sees recovery states, GAIL eventually matches the expert, and pure RL takes much longer to reach the same level. The right panel makes the trade-off concrete: methods that consume more expert data tend to need fewer environment steps, and vice versa.

This is also why **imitation pre-training + RL fine-tuning** is the dominant recipe in modern systems -- AlphaStar, robotic manipulation, and InstructGPT all start with imitation and then improve with RL.

---

## 7. Method selection

| Method | Needs interactive expert? | Needs env interaction? | Sample efficiency | Interpretable reward? | Typical regime |
|---|---|---|---|---|---|
| BC | no | no | high (offline) | no | short tasks, abundant logs |
| DAgger | **yes** | yes | medium-high | no | algorithmic / model expert available |
| MaxEnt IRL | no | yes (inner loop) | low | **yes** | small state spaces, need transfer |
| GAIL | no | yes | medium | no | high-dim continuous control |
| AIRL | no | yes | medium | yes (state reward) | want to transfer reward across tasks |

A short decision rule:

- **Static log of demonstrations, short horizon** → start with **BC**. Add early stopping and standardisation; consider an ensemble for uncertainty.
- **You can _query_ the expert at training time** → **DAgger**. The linear-error guarantee is the cheapest improvement you will ever buy.
- **You need to _understand_ what the expert wants, or to transfer it to a different environment** → **MaxEnt IRL** for small problems, **AIRL** for continuous control.
- **High-dimensional continuous control with a static dataset and an environment you can interact with** → **GAIL**. Budget for adversarial-training instability.

---

## 8. Frequently asked questions

**Can imitation learning exceed the expert?**
Pure imitation cannot, by construction -- the optimal imitator matches the expert. The standard fix is _imitation as initialisation_: start from the BC/GAIL policy and fine-tune with RL on whatever reward you _can_ specify (or with RLHF). Most large-scale systems use exactly this two-stage recipe.

**How do I handle noisy or suboptimal expert demonstrations?**
Three lines of defence: (i) weight samples by an estimated quality score (e.g. return, human preference); (ii) replace cross-entropy with a robust loss (Huber, log-cosh) to discount outliers; (iii) use offline RL on the demonstrations -- methods such as IQL or CQL handle suboptimal trajectories gracefully.

**The expert is multimodal -- different actions in the same state. What now?**
Standard MSE-BC averages the modes and produces dangerous in-between actions (the classic "drive into the obstacle because half the demos go left and half go right" failure). Use a Mixture Density Network, a discrete-action quantised policy, a conditional VAE, or a diffusion policy. Diffusion policies (Chi et al., 2023) currently dominate on multimodal manipulation tasks.

**Can BC handle distribution shift if I just collect more data?**
Up to a point. More data shrinks $\varepsilon$ but does not change the $T^2$ exponent. Once your horizon is long enough that the policy ever leaves the support of $\mathcal{D}$, you need either DAgger-style relabelling, GAIL-style occupancy matching, or _conservative_ offline RL (CQL, IQL) that explicitly penalises out-of-support actions.

**Is RLHF a kind of imitation learning?**
Partly. The supervised fine-tuning stage of RLHF is BC on human-written demonstrations. The preference-modelling stage is closer to an IRL variant -- you learn a reward model from comparisons, then optimise it with PPO. We unpack the full pipeline in [Part 12](/en/reinforcement-learning-12-rlhf-and-llm-applications/).

---

## References

- Pomerleau, D. (1989). _ALVINN: An Autonomous Land Vehicle in a Neural Network_. NIPS.
- Ross, S., Gordon, G., & Bagnell, J. A. (2011). _A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning_. AISTATS. arXiv:1011.0686.
- Ziebart, B., Maas, A., Bagnell, J. A., & Dey, A. (2008). _Maximum Entropy Inverse Reinforcement Learning_. AAAI.
- Finn, C., Levine, S., & Abbeel, P. (2016). _Guided Cost Learning_. ICML. arXiv:1603.00448.
- Ho, J., & Ermon, S. (2016). _Generative Adversarial Imitation Learning_. NeurIPS. arXiv:1606.03476.
- Fu, J., Luo, K., & Levine, S. (2018). _Learning Robust Rewards with Adversarial Inverse Reinforcement Learning_. ICLR. arXiv:1710.11248.
- Chi, C. et al. (2023). _Diffusion Policy: Visuomotor Policy Learning via Action Diffusion_. RSS. arXiv:2303.04137.

---

## Series navigation

- **Previous**: [Part 6 -- Advanced Policy Optimization](/en/reinforcement-learning-6-advanced-policy-optimization/)
- **Next**: [Part 8 -- AlphaGo and Monte Carlo Tree Search](/en/reinforcement-learning-8-alphago-and-mcts/)
- [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)
