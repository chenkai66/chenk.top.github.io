---
title: "Reinforcement Learning (8): AlphaGo and Monte Carlo Tree Search"
date: 2025-07-13 09:00:00
tags:
  - Reinforcement Learning
  - AlphaGo
  - MCTS
  - AlphaZero
  - MuZero
categories: Reinforcement Learning
series: Reinforcement Learning
part: 8
total_parts: 12
lang: en
mathjax: true
description: "From MCTS to AlphaGo, AlphaGo Zero, AlphaZero, and MuZero. Understand UCT exploration-exploitation, self-play training, and planning with learned models. Includes a complete AlphaZero implementation for Gomoku."
disableNunjucks: true
series_order: 8
---

In March 2016, AlphaGo defeated world Go champion Lee Sedol 4–1 in Seoul. The result was not just a sporting upset; it was the moment a 60-year programme in artificial intelligence — beating the world's best at Go — concluded a full decade ahead of most published predictions. Go has roughly $10^{170}$ legal positions, more than the number of atoms in the observable universe. No amount of brute-force search will ever crack it. AlphaGo's victory came from a different idea: let a deep network supply the *intuition* about which moves look promising, and let Monte Carlo Tree Search (MCTS) supply the *deliberation* that verifies and sharpens that intuition.

Eighteen months later, AlphaGo Zero learned the game from nothing but the rules and three days of self-play, and crushed the Lee-Sedol version 100–0. AlphaZero generalised the same recipe to chess and shogi. MuZero went further and learned the rules themselves. This chapter traces the full evolution — the algorithm, the mathematics, and a working implementation you can actually train.

## What You Will Learn

- **MCTS Foundations**: the four-phase loop, UCT exploration–exploitation balance, asymptotic optimality
- **AlphaGo (2016)**: three-stage training (SL policy, RL policy, value network) and how MCTS fuses them
- **AlphaGo Zero (2017)**: self-play from scratch, a single dual-head network, no rollouts
- **AlphaZero (2017)**: the same algorithm applied to chess, shogi, and Go
- **MuZero (2019)**: planning in a *learned* latent space without environment rules
- **Complete code**: AlphaZero for Gomoku — environment, network, MCTS, and self-play loop

## Prerequisites

- Deep RL fundamentals (policy gradients, value functions) — see [Part 3](/en/reinforcement-learning-3-policy-gradient-and-actor-critic/)
- Convolutional neural networks
- Game trees are helpful but not assumed

---

## 1. Monte Carlo Tree Search

Classical game-tree search (minimax + alpha–beta) needs an evaluation function and a manageable branching factor. Chess has both. Go has neither: branching factor around 250, no concise evaluation function, no useful heuristics. MCTS sidesteps this by *sampling* rather than enumerating, and by *focusing* its samples on the most promising parts of the tree.

![MCTS Four Phases](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig1_mcts_four_phases.png)

A single MCTS *simulation* repeats four phases:

1. **Selection** — starting from the root, walk down the tree by repeatedly choosing the child that maximises a search criterion (UCT, below). Stop on reaching a node that has at least one untried action — a *leaf* of the current tree, not necessarily of the game.
2. **Expansion** — pick one untried action at the leaf and add the resulting state as a new child.
3. **Simulation (rollout)** — from the new child, play the game out to a terminal state using a fast policy: random in vanilla MCTS, a small network in classical AlphaGo, replaced entirely by a value network in AlphaGo Zero.
4. **Backpropagation** — propagate the outcome up the path, incrementing visit counts $N$ and accumulating values $W$ at every node touched.

After a fixed simulation budget (AlphaGo Zero uses 800 per move), the algorithm returns the action whose root child has the most visits — *not* the highest mean value. Visit count is a more robust statistic because it integrates the search's own self-correction.

### 1.1 UCT: balancing exploration and exploitation

![UCB1 Exploration and Exploitation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig2_ucb_exploration.png)

The selection rule is the *Upper Confidence bound for Trees* (UCT), Kocsis & Szepesvári (2006). At a node $s$, choose the child action $a$ that maximises

$$\text{UCT}(s, a) = \underbrace{\frac{W(s, a)}{N(s, a)}}_{\text{exploitation}} \;+\; \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{exploration}}.$$

The first term is the empirical mean value — a node that has won often gets visited more. The second is an Auer–Cesa-Bianchi–Fischer confidence bound: small visit counts inflate it, encouraging the search to *try* less-visited children. As $N(s,a) \to \infty$ the bonus shrinks, and the rule converges to greedy exploitation. UCT is **asymptotically optimal**: in the limit of infinite simulations the visit distribution concentrates on the optimal action.

In AlphaGo's PUCT variant the bonus is also weighted by the network's prior $p(a\mid s)$:

$$\text{PUCT}(s, a) = Q(s, a) \;+\; c_{\text{puct}} \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}.$$

The prior tells the search *where to look first*; the visit count tells it *where to stop looking*.

---

## 2. AlphaGo (2016): Networks Meet Search

![AlphaGo Architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig3_alphago_architecture.png)

The original AlphaGo trained in three stages:

**Stage 1 — Supervised policy $p_\sigma$.** A 13-layer CNN was trained on 30 million positions from KGS expert games to predict the human move. It reached 57% top-1 accuracy and was the first big jump over prior work (~44%).

**Stage 2 — RL policy $p_\rho$.** Initialise $p_\rho \leftarrow p_\sigma$, then improve via REINFORCE-style self-play against an opponent randomly sampled from earlier checkpoints. The RL policy beat the SL policy 80% of the time. *But* — and this is the surprise — using $p_\rho$ inside MCTS hurt: it had collapsed onto a few stylistic preferences and lost the diversity that gives the search useful priors. Production AlphaGo therefore used $p_\sigma$ for priors.

**Stage 3 — Value network $v_\theta$.** A separate CNN was regressed on game outcomes. To avoid overfitting from highly correlated within-game positions, only *one* position per self-play game was used, yielding 30 M independent (state, outcome) pairs.

At play time, MCTS combined the two networks. The leaf evaluation mixed the value-network estimate with a fast random rollout:

$$V(s_L) \;=\; (1 - \lambda)\, v_\theta(s_L) \;+\; \lambda\, z_L, \qquad \lambda = 0.5.$$

Why mix at all? In 2016 the value network was strong but not perfect — rollouts averaged out its systematic errors. By 2017 networks had improved enough that the rollout term hurt more than it helped, and AlphaGo Zero dropped it entirely.

---

## 3. AlphaGo Zero (2017): Tabula Rasa

![AlphaGo Zero Self-Play Loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig4_zero_self_play_loop.png)

AlphaGo Zero is the same idea, *simpler*. Three changes — each of which would have looked dangerous on its own:

1. **No human data.** The agent starts from random initialisation and learns from self-play only. Every previous move-prediction system above 50% accuracy was thrown away.
2. **Single dual-head network** $f_\theta(s) = (\mathbf{p}, v)$. One residual tower with two heads — a policy head outputting a distribution over moves and a scalar value head — replaces the policy and value networks of AlphaGo.
3. **No rollouts.** The leaf is evaluated *only* by the value head. The fast rollout policy is gone.

The training loop is a tight closed cycle (see figure):

1. **Self-play.** The current best network plays against itself with MCTS (800 simulations per move), producing trajectories of $(s_t, \boldsymbol{\pi}_t, z_T)$ where $\boldsymbol{\pi}_t$ is the *MCTS visit-count distribution* (a sharper, slower target than the raw network policy) and $z_T \in \{-1, +1\}$ is the final game result from the player-to-move's perspective.
2. **Train.** Update $\theta$ by minimising $\mathcal{L}(\theta) \;=\; (z - v)^2 \;-\; \boldsymbol{\pi}^\top \log \mathbf{p} \;+\; c\,\|\theta\|^2,$ a squared-error value loss, a cross-entropy policy loss, and weight decay.
3. **Evaluate.** A new network challenges the current best. Only if it wins more than 55% of 400 games does it become the new generator of self-play games.

The genius is in the *labels*. The MCTS visit distribution $\boldsymbol{\pi}$ is *strictly better than the raw network policy that produced it* — search has refined the priors. Training $\mathbf{p}$ to imitate $\boldsymbol{\pi}$ distils the search's improvement back into the network. This is policy iteration where the policy improvement step is MCTS itself. Each new network plays self-play that yields targets a little stronger than itself. The process bootstraps without external supervision because the *opponent improves at the same rate as the learner* — automatic curriculum learning.

After **3 days** of training on 4 TPUs, AlphaGo Zero defeated the Lee-Sedol AlphaGo 100–0. After 40 days it surpassed AlphaGo Master, the version that had beaten Ke Jie.

---

## 4. AlphaZero and MuZero

![Algorithm Evolution Timeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig5_evolution_timeline.png)

**AlphaZero** (Dec 2017) showed that the AlphaGo Zero algorithm was not Go-specific. With only the game-specific state encoding swapped out — and a few tweaks: no win-rate gating between generations, draws allowed (chess has many) — the same code beat Stockfish 8 in chess (28 wins, 0 losses, 72 draws) and Elmo in shogi after roughly **9 hours** of training on TPUs. It also beat AlphaGo Zero at Go.

**MuZero** (Nov 2019) went one step further: drop the requirement that the agent know the rules of the game. MuZero learns three functions jointly:

- **Representation** $h_\theta : o_{\le t} \mapsto s_t^0$ — encodes observation history into an initial hidden state.
- **Dynamics** $g_\theta : (s_t^k, a_{t+k}) \mapsto (s_t^{k+1}, r_t^{k+1})$ — predicts next hidden state and reward.
- **Prediction** $f_\theta : s_t^k \mapsto (\mathbf{p}_t^k, v_t^k)$ — predicts policy and value.

MCTS unfolds entirely in **latent space**. There is no environment simulator inside the search — only the learned dynamics function. The hidden state is not required to reconstruct the observation; it only has to be useful for predicting *reward, value, and policy*. With this looser objective, MuZero matches AlphaZero on board games and surpasses model-free methods (R2D2, Ape-X) on Atari, where no rule-based simulator exists.

### 4.1 Elo over time

![Elo Progression](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig6_elo_progression.png)

The left panel compares peak Elo across the family. The right panel shows Zero's training trajectory: it surpasses the Lee-Sedol version after 3 days, AlphaGo Master after about 21 days, and saturates near 5200 Elo. For context, 9-dan human professionals are around 3500–3700.

### 4.2 How much does search actually help?

![Search Budget vs Strength](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/fig7_search_vs_strength.png)

Search and the network are complementary — the network produces priors; search refines them. The left panel shows that doubling MCTS simulations adds roughly constant Elo (a logarithmic relationship), with no plateau in sight up to 12 800 simulations. The right panel shows the multiplicative gain from neural priors: vanilla MCTS with random rollouts saturates much earlier than MCTS guided by a network. Each component is needed; neither alone is competitive.

---

## 5. Complete Implementation: AlphaZero for Gomoku

Gomoku ("five in a row") on a 9×9 board is a useful test bed: rules fit in 30 lines, branching factor is around 60, and decent play emerges from a few thousand self-play games on a single GPU.

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class GomokuEnv:
    def __init__(self, size=9):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.get_state()

    def get_state(self):
        # 3-channel state: own stones, opponent stones, side-to-move plane
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0] = (self.board == self.current_player)
        state[1] = (self.board == -self.current_player)
        state[2] = self.current_player
        return state

    def legal_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        self.board[action] = self.current_player
        if self._check_win(action):
            self.done = True
            self.winner = self.current_player
            return self.get_state(), self.winner, True
        if len(self.legal_actions()) == 0:
            self.done = True
            return self.get_state(), 0, True
        self.current_player *= -1
        return self.get_state(), 0, False

    def _check_win(self, last_move):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.board[last_move]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = last_move[0] + sign * dr, last_move[1] + sign * dc
                while (0 <= r < self.size and 0 <= c < self.size
                       and self.board[r, c] == player):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 5:
                return True
        return False


class PolicyValueNet(nn.Module):
    """Tiny dual-head network: a shared CNN trunk + policy head + value head."""
    def __init__(self, board_size=9, channels=128):
        super().__init__()
        self.size = board_size
        self.shared = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size ** 2, board_size ** 2),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        shared = self.shared(x)
        # log-softmax for numerically stable cross-entropy with MCTS targets
        policy = torch.log_softmax(self.policy_head(shared), dim=1)
        value = self.value_head(shared)
        return policy, value


class MCTSNode:
    def __init__(self, prior, parent=None):
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

    def select_child(self, c_puct=1.0):
        # PUCT: Q + c * P * sqrt(parent N) / (1 + child N)
        best_score, best_action, best_child = -float('inf'), None, None
        for action, child in self.children.items():
            u = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.value() + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(prior, parent=self)

    def backup(self, value):
        # value is from the perspective of the player-to-move at this node
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # flip sign: parent's perspective is the opponent's
            self.parent.backup(-value)


class MCTS:
    def __init__(self, model, num_simulations=400, c_puct=1.0):
        self.model = model
        self.num_sims = num_simulations
        self.c_puct = c_puct

    @torch.no_grad()
    def search(self, env):
        root = MCTSNode(prior=0)
        # Initialise root with network priors over legal actions
        state = env.get_state()
        legal = env.legal_actions()
        log_probs, _ = self.model(torch.FloatTensor(state).unsqueeze(0))
        probs = torch.exp(log_probs).squeeze().numpy()
        action_probs = np.array([probs[a[0] * env.size + a[1]] for a in legal])
        action_probs /= action_probs.sum()
        root.expand(legal, action_probs)

        for _ in range(self.num_sims):
            node = root
            env_copy = self._copy_env(env)
            path = [node]

            # Selection: walk down until we hit an unexpanded node or terminal
            while node.children and not env_copy.done:
                action, node = node.select_child(self.c_puct)
                path.append(node)
                env_copy.step(action)

            # Expansion + evaluation
            if not env_copy.done:
                state = env_copy.get_state()
                legal = env_copy.legal_actions()
                log_probs, value = self.model(
                    torch.FloatTensor(state).unsqueeze(0))
                probs = torch.exp(log_probs).squeeze().numpy()
                ap = np.array([probs[a[0] * env_copy.size + a[1]] for a in legal])
                ap /= ap.sum()
                node.expand(legal, ap)
                v = value.item()
            else:
                # Terminal: ground-truth outcome from the leaf player's view
                v = env_copy.winner * env_copy.current_player

            # Backup along the visited path (sign flips at each level)
            for n in reversed(path):
                n.backup(v)
                v = -v

        # Return visit-count distribution as the MCTS-improved policy
        visits = np.zeros(env.size ** 2)
        for action, child in root.children.items():
            visits[action[0] * env.size + action[1]] = child.visit_count
        return visits / visits.sum()

    def _copy_env(self, env):
        new = GomokuEnv(env.size)
        new.board = env.board.copy()
        new.current_player = env.current_player
        new.done = env.done
        new.winner = env.winner
        return new
```

The training loop generates self-play games, stores $(s_t, \boldsymbol{\pi}_t, z_T)$ triples, and trains the network with the AlphaGo Zero loss. On a 9×9 board with 400 simulations, the network reaches non-trivial play (beats random + greedy heuristics) after roughly 50 self-play iterations on a single consumer GPU. Two practical tips: (i) inject Dirichlet noise into the root priors during self-play to keep exploration alive; (ii) use a *temperature* on the visit distribution — temperature 1 for the first ~10 moves of self-play, then near-greedy — which stops the data being collected from a single deterministic line of play.

---

## Frequently Asked Questions

**Q: Why does AlphaGo Zero not need rollouts?**
By 2017 deeper residual networks, more self-play data, and a unified policy/value head produced a value function that was on average more accurate than mixing the network with noisy random rollouts. The DeepMind ablations are explicit: pure value evaluation beat the mixture, so the rollout was dropped.

**Q: Can self-play get stuck in a degenerate equilibrium?**
For two-player zero-sum perfect-information games, *fictitious self-play* converges to a Nash equilibrium (Brown, 1951; Heinrich & Silver, 2016). MCTS adds optimistic exploration on top, which prevents premature mode collapse. In imperfect-information games (poker) or cooperative games this is not guaranteed and a population of opponents (PSRO, AlphaStar's league) is needed.

**Q: Why use the *visit-count* distribution as the policy target, not the empirical $Q$?**
Visit counts are robust: a child with very few visits can have a noisy mean $Q$, but cannot have many visits unless the search consistently chose it. Cross-entropy against $\boldsymbol{\pi}$ also gives a meaningful gradient even for actions the search rarely tried, which a hard argmax target would not.

**Q: Can MCTS handle continuous action spaces?**
Not directly — UCT and PUCT assume a finite action set. Extensions like *Progressive Widening* gradually add sampled actions as a node is visited more often, and recent work (e.g., Sampled MuZero, 2021) handles continuous and structured action spaces. For pure continuous control, model-free methods such as SAC and PPO remain more practical.

**Q: Why 800 simulations per move? Could you train with 1?**
You can. With 1 simulation per move the visit-count target *is* the network policy, no improvement happens, and training stalls. At 800 simulations the search target is meaningfully sharper than the network — that gap is what the network learns from. Diminishing returns kick in around the low thousands; AlphaZero used 800 for Go and chess, MuZero kept the same number.

---

## References

- Silver et al., **Mastering the game of Go with deep neural networks and tree search**, *Nature* 529, 2016.
- Silver et al., **Mastering the game of Go without human knowledge**, *Nature* 550, 2017 (AlphaGo Zero).
- Silver et al., **A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play**, *Science* 362, 2018 (AlphaZero).
- Schrittwieser et al., **Mastering Atari, Go, chess and shogi by planning with a learned model**, *Nature* 588, 2020 (MuZero).
- Kocsis & Szepesvári, **Bandit based Monte-Carlo Planning**, *ECML* 2006 (UCT).
- Auer, Cesa-Bianchi & Fischer, **Finite-time Analysis of the Multiarmed Bandit Problem**, *Machine Learning* 47, 2002 (UCB1).
- Browne et al., **A Survey of Monte Carlo Tree Search Methods**, *IEEE TCIAIG* 4, 2012.

---

## Series Navigation

- **Previous**: [Part 7 — Imitation Learning](/en/reinforcement-learning-7-imitation-learning/)
- **Next**: [Part 9 — Multi-Agent RL](/en/reinforcement-learning-9-multi-agent-rl/)
- [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)
