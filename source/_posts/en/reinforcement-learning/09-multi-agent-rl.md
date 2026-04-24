---
title: "Reinforcement Learning (9): Multi-Agent Reinforcement Learning"
date: 2024-07-09 09:00:00
tags:
  - Reinforcement Learning
  - Multi-Agent
  - QMIX
  - MADDPG
  - Game Theory
categories: Reinforcement Learning
series: Reinforcement Learning
part: 9
total_parts: 12
lang: en
mathjax: true
description: "A working tour of multi-agent RL: Markov games, the non-stationarity and credit-assignment problems, CTDE, value decomposition (VDN, QMIX), counterfactual baselines (COMA), MADDPG, communication topologies, and the league-training pipeline behind AlphaStar and OpenAI Five — with a runnable QMIX mixer in PyTorch."
---

Single-agent RL rests on one quiet but enormous assumption: the environment is stationary. The transition kernel does not change while the agent learns. The moment a second learner shares the world, that assumption collapses. Each agent now sees an environment whose dynamics shift as its peers update, rewards become entangled across agents, and the joint action space explodes combinatorially. These are not engineering nuisances. They are the reason multi-agent RL needs its own algorithms instead of just *running DQN n times in parallel*.

The payoff for solving this is large. AlphaStar reached Grandmaster on the StarCraft II ladder, OpenAI Five won against the world champions in Dota 2, and cooperative MARL is increasingly the workhorse for warehouse robotics, traffic-signal control, and multi-LLM agent systems. This chapter builds the conceptual backbone — Markov games, CTDE, value decomposition, counterfactual credit — and lands on a clean QMIX mixer you can drop into your own training loop.

## What you will learn

- The four hard problems unique to MARL: **non-stationarity**, **credit assignment**, **partial observability**, and **scalability**
- The game-theoretic vocabulary you actually need: Markov games, Nash equilibria, Pareto fronts, social dilemmas
- **CTDE** — centralised training, decentralised execution — and why it dominates
- **Value decomposition**: VDN, QMIX, and the monotonicity constraint that makes greedy execution sound
- **Multi-agent actor-critic**: MADDPG (centralised critic per agent) and **COMA** (counterfactual baselines)
- **Communication**: when broadcast, sparse, and attention-based message passing each pay off
- The engineering of **AlphaStar's league** and **OpenAI Five's** scale-out training

## Prerequisites

- Q-learning and DQN (Part 2)
- Actor-critic and policy gradients (Part 3, Part 6)
- Comfort with expectations and conditional probabilities; we keep formal game theory minimal

---

## 1. Why MARL is genuinely harder than single-agent RL

![Cooperative, competitive, and mixed-motive multi-agent regimes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig1_scenarios.png)

The reward structure decides almost everything else. In **cooperative** games every agent shares the team reward — coordinated soccer plays, swarm logistics, StarCraft micro-management. **Competitive** games are zero-sum: one agent's gain is another's loss, and the right tool is some form of self-play. The interesting middle ground is **mixed-motive** (general-sum) games — markets, traffic, negotiation, congestion control — where short-term self-interest can torpedo long-term collective welfare (the prisoner's dilemma is the canonical 2×2 case, and the figure shows it as the third payoff matrix).

Within any of these regimes, four obstacles break the single-agent toolbox.

**Non-stationarity.** From agent $i$'s point of view the transition is $P(s' \mid s, a_i, a_{-i})$, where $a_{-i}$ denotes everyone else's actions. As the others' policies $\pi_{-i}$ keep updating during training, the *effective* transition kernel that $i$ sees keeps drifting. Off-policy methods like DQN, which assume a fixed data-generating distribution, can diverge or oscillate.

**Credit assignment.** When the team gets a single reward $r$ for an episode-long collaboration, who actually earned it? A naive shared gradient encourages free-riding: an agent that does nothing receives the same credit as one that did the heavy lifting. Counterfactual baselines (COMA, Section 5) and value decomposition (Section 4) are the two principled answers.

**Partial observability.** Each agent sees only its own observation $o_i$. There is no Markov state available at execution time. Recurrent networks plus belief-state-style representations are typical, but the deeper fix is to architect training so that *the centralised critic* sees more than execution does.

**Scalability.** With $n$ agents and $|A|$ actions each, the joint action space is $|A|^n$. Without exploiting structure, even modest cooperative tasks become intractable.

The dominant resolution to all four — and the conceptual backbone of the chapter — is **CTDE**.

---

## 2. CTDE: train with everything, execute with almost nothing

![Centralized training with decentralized execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig2_ctde.png)

The idea is asymmetric. During training we let the algorithm peek at the full state and at every agent's action — a *centralised* critic, mixer, or value function. At deployment we keep only the per-agent policies that condition on local observations. The execution interface is decentralised; the optimisation interface is not.

CTDE works because the things that make MARL hard (non-stationarity, credit assignment, partial observability) are properties of the *learning* phase, while the constraints that make MARL practical (limited bandwidth, asynchronous control, privacy) are properties of the *execution* phase. CTDE puts each problem in its own corner. Almost every modern cooperative MARL algorithm — VDN, QMIX, MADDPG, COMA, MAPPO — is a CTDE method; they differ only in *what* the centralised object is and *how* it is decomposed.

---

## 3. Markov games and Nash equilibria, just enough to be useful

A **Markov game** generalises the MDP to $n$ agents:

$$
\langle \mathcal{N}, \mathcal{S}, \{A_i\}_{i\in\mathcal{N}}, P, \{r_i\}_{i\in\mathcal{N}}, \gamma \rangle.
$$

Each agent $i$ has its own action set $A_i$ and its own reward $r_i(s, a_1, \ldots, a_n)$. A **Nash equilibrium** is a joint policy $\pi^* = (\pi_1^*, \ldots, \pi_n^*)$ where no single agent can improve unilaterally:

$$
V_i(\pi_i^*, \pi_{-i}^*) \;\geq\; V_i(\pi_i, \pi_{-i}^*) \qquad \forall \pi_i,\, \forall i.
$$

Two facts make Nash equilibria a slippery target. First, they are typically not unique; second, they are not always Pareto optimal. In the *fully cooperative* case, where $r_1 = \cdots = r_n$, the picture simplifies dramatically — the problem reduces to maximising a single team return, and Nash equilibrium and Pareto optimality collapse onto the same set. That is why most of the algorithmic machinery in this chapter targets the cooperative setting first; competitive and mixed-motive cases borrow from it but layer additional ideas (self-play, opponent modelling, regularisation) on top.

---

## 4. Value decomposition: VDN and QMIX

![QMIX architecture and monotonic factorisation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig3_qmix.png)

Value decomposition is a CTDE recipe for the cooperative case. We learn a per-agent value function $Q_i(o_i, a_i)$ for execution and a *combined* $Q_\text{tot}(s, \mathbf{a})$ for training, then we constrain the relationship between them so that decentralised greedy action selection on $Q_i$ is equivalent to joint greedy selection on $Q_\text{tot}$. This is the **Individual-Global-Max (IGM)** property:

$$
\arg\max_{\mathbf{a}} Q_\text{tot}(s, \mathbf{a}) \;=\; \big(\arg\max_{a_1} Q_1(o_1, a_1),\; \ldots,\; \arg\max_{a_n} Q_n(o_n, a_n)\big).
$$

### VDN: additive decomposition

[Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) take the simplest sufficient condition:

$$
Q_\text{tot}(s, \mathbf{a}) \;=\; \sum_{i=1}^{n} Q_i(o_i, a_i).
$$

Sums are monotonic in each summand, so IGM holds trivially. The cost is expressivity: VDN cannot represent any non-additive interaction between agents.

### QMIX: monotonic mixing

[QMIX](https://arxiv.org/abs/1803.11485) keeps IGM but generalises *additive* to *monotonic*. A neural **mixing network** combines the $Q_i$ into $Q_\text{tot}$ subject to

$$
\frac{\partial Q_\text{tot}}{\partial Q_i} \;\geq\; 0 \qquad \forall i.
$$

The constraint is enforced by *constructing* the mixer's weights to be non-negative — by passing them through an absolute value or a softplus. Crucially, the weights themselves are produced by a **hypernetwork** conditioned on the global state $s$, so different states can reweight agents differently (and the mixer can express interactions that pure summation cannot). The right panel of the figure shows the resulting $Q_\text{tot}$ surface: monotone in each $Q_i$, so the joint argmax aligns with the pair of per-agent argmaxes.

```python
import torch
import torch.nn as nn


class QMixer(nn.Module):
    """QMIX mixing network with state-conditioned positive weights."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Hypernetwork: state -> mixing weights (kept >= 0 via abs)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        # Final scalar bias is unconstrained
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """agent_qs: (B, n_agents)   state: (B, state_dim)   ->  (B, 1)"""
        B = agent_qs.size(0)
        agent_qs = agent_qs.view(B, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.hidden_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(B, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, 1)
```

Train it exactly like a DQN target — sample transitions, build the TD target on $Q_\text{tot}$, backprop through the mixer into all per-agent networks. At execution time, every agent runs its own $Q_i$ and never touches the mixer.

A practical note: QMIX cannot represent every IGM-decomposable function (its monotonicity is *sufficient* but not *necessary*). [QTRAN](https://arxiv.org/abs/1905.05408) and [QPLEX](https://arxiv.org/abs/2008.01062) push this frontier, at the cost of more complex losses and engineering.

---

## 5. Multi-agent actor-critic: MADDPG and COMA

### MADDPG: a centralised critic per agent

![MADDPG centralised critic, decentralised actor](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig5_maddpg.png)

[MADDPG](https://arxiv.org/abs/1706.02275) is the natural CTDE extension of DDPG. Each agent $i$ has

- an **actor** $\mu_i(o_i)$ that conditions only on its local observation, and
- a **critic** $Q_i(s, a_1, \ldots, a_n)$ that conditions on the full state and *every* agent's action.

The actor is updated with the deterministic policy gradient through its own critic; the critic is updated with a TD target that uses target actors of all agents:

$$
y \;=\; r_i \;+\; \gamma \, Q_i^{\text{tgt}}\!\left(s', \mu_1^{\text{tgt}}(o_1'), \ldots, \mu_n^{\text{tgt}}(o_n')\right).
$$

Because the critic conditions on $a_{-i}$, the environment looks stationary *to the critic* — its inputs include the very thing that was changing under it. The actor remains decentralised, so deployment is unchanged. MADDPG works in cooperative, competitive, and mixed-motive settings; the replay buffer simply stores joint trajectories.

### COMA: counterfactual credit assignment

![COMA counterfactual baseline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig4_coma.png)

[COMA](https://arxiv.org/abs/1705.08926) attacks the credit-assignment problem head-on. It uses a single centralised critic $Q(s, \mathbf{a})$ and, for each agent $i$, a **counterfactual baseline** that holds everyone else's actions fixed and averages out $i$'s action under its own policy:

$$
A_i(s, \mathbf{a}) \;=\; Q(s, \mathbf{a}) \;-\; \sum_{a_i'} \pi_i(a_i' \mid o_i) \, Q\!\left(s, (a_i', a_{-i})\right).
$$

Read this carefully. The first term is the value of what actually happened. The second is the *expected* value if agent $i$ had sampled some other action *while everyone else did exactly what they did*. The difference is precisely $i$'s marginal contribution. When the team reward is shared, this is the right object to do policy gradient on — it removes the noise from $a_{-i}$ without introducing bias. The figure shows the four-action case: only the chosen action receives a non-trivial advantage.

The trick that makes COMA tractable is that the centralised critic outputs a vector of $Q$-values for all candidate actions of agent $i$ at once, so the baseline is a single dot product against the policy probabilities — no extra environment rollouts needed.

---

## 6. Communication: how much should agents tell each other?

![Communication topologies: broadcast, sparse, attention](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig6_communication.png)

Sometimes decentralised execution is too restrictive — agents really do need to share information at test time. The design space is a spectrum.

**Broadcast** — every agent sends a fixed-size message to every other agent. Conceptually clean, but message volume grows as $O(n^2)$ per step, which is fatal beyond a few dozen agents.

**Sparse / k-NN** — each agent only talks to its $k$ nearest neighbours (in space, in a graph, in role). Linear in $n$ for fixed $k$. CommNet, IC3Net and TarMAC all live near this end.

**Attention** — soft, learned routing. Each agent issues a query, peers respond with keys, and message weights are computed by softmax. The cost is $O(n^2)$ in attention scores but messages can be heavily pruned (top-$k$ attention) and the structure is end-to-end differentiable. This is the dominant choice in modern MARL and the natural bridge to multi-LLM-agent systems, where attention-style routing maps directly onto tool-use and agent orchestration patterns.

The right knob is task-dependent: for tightly-coupled robotics teams, dense communication is fine; for thousands of microservices or vehicles, sparse or attention-pruned schemes are the only feasible designs.

---

## 7. Scaling MARL: AlphaStar, OpenAI Five, and league training

![AlphaStar league training and flagship MARL systems](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig7_league.png)

Both [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) (StarCraft II) and [OpenAI Five](https://arxiv.org/abs/1912.06680) (Dota 2) succeed by treating *training-population design* as a first-class engineering concern. Naive self-play tends to cycle — agent A learns to beat B, B learns to beat A's new policy and forgets why it used to beat A's old policy, and so on. **AlphaStar's league** explicitly decouples three populations:

- **Main agents** — the agents that will actually be deployed; they are the optimisation target.
- **Main exploiters** — separate agents whose only job is to find weaknesses in the main agents, forcing them to round out their game.
- **League exploiters** — periodically reset agents that maintain *strategic diversity* so the league does not collapse onto a narrow style.

A pool of *frozen historical checkpoints* is matched against current learners with prioritised fictitious self-play (PFSP), which biases match-making toward opponents the current learner is just barely beating. Together these mechanisms prevent the catastrophic forgetting that breaks naive self-play.

OpenAI Five takes a different cut. Each Dota hero is controlled by an independent LSTM policy with a shared parameter set; the team-level coordination problem is handed off to a hand-engineered shared reward and an enormous compute budget — roughly *180 years of self-play per day*. The lesson from both systems is the same: at industrial scale, the *training curriculum* matters at least as much as the per-agent algorithm.

These ideas are migrating outside games. RLHF (Part 12) and tool-use agent orchestration both increasingly look like multi-population training — a population of policies, a population of evaluators, and prioritised match-making between them.

---

## Frequently asked questions

**Why does QMIX need the monotonicity constraint?**
So that decentralised greedy execution is consistent with centralised joint optimisation. If $\partial Q_\text{tot}/\partial Q_i$ could go negative, an agent acting greedily on its own $Q_i$ might *lower* the team value. Monotonicity makes the per-agent argmax and the joint argmax coincide.

**When do MARL systems get stuck in suboptimal equilibria?**
Whenever the optimal joint action requires *coordinated* exploration — both agents must simultaneously try the risky move for it to look good. If random exploration almost never produces that joint action, the system converges to a safer but Pareto-dominated equilibrium. The fixes are joint exploration (committed-exploration schedules), explicit communication, and opponent modelling.

**How does sample complexity scale with the number of agents?**
For cooperative tasks where value decomposition holds, sample complexity grows roughly linearly in $n$ — that is the entire point of QMIX. Without decomposition, learning $Q(s, \mathbf{a})$ over the joint action space is exponential in $n$.

**Independent learners vs CTDE — is CTDE always worth it?**
For very small populations or when global state is genuinely unavailable, independent Q-learning can work and is much simpler to implement. As soon as credit assignment matters or the team is more than 3–4 agents, CTDE methods pull ahead and the gap widens fast.

**Does any of this transfer to multi-LLM-agent systems?**
Yes. Centralised critics map onto outer evaluators, decentralised actors onto sub-agents, attention-based communication onto tool routing, and league training onto the population-of-evaluators pattern that is becoming standard in agentic RLHF.

---

## References

- Sunehag et al., *Value-Decomposition Networks for Cooperative Multi-Agent Learning*, AAMAS 2018. [arXiv:1706.05296](https://arxiv.org/abs/1706.05296)
- Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*, ICML 2018. [arXiv:1803.11485](https://arxiv.org/abs/1803.11485)
- Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), NeurIPS 2017. [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients* (COMA), AAAI 2018. [arXiv:1705.08926](https://arxiv.org/abs/1705.08926)
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575, 2019.
- OpenAI et al., *Dota 2 with Large Scale Deep Reinforcement Learning*, 2019. [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO), NeurIPS 2022. [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)

---

## Series Navigation

- **Previous**: [Part 8 — AlphaGo and MCTS](/en/reinforcement-learning-8-alphago-and-mcts/)
- **Next**: [Part 10 — Offline Reinforcement Learning](/en/reinforcement-learning-10-offline-reinforcement-learning/)
- [View all 12 parts in the RL series](/tags/Reinforcement-Learning/)
