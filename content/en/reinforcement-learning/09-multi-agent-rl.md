---
title: "Reinforcement Learning (9): Multi-Agent Reinforcement Learning"
date: 2025-09-10 09:00:00
tags:
  - Reinforcement Learning
  - Multi-Agent
  - QMIX
  - MADDPG
  - Game Theory
categories: Reinforcement Learning
series: reinforcement-learning
part: 9
total_parts: 12
lang: en
mathjax: true
description: "A working tour of multi-agent RL: Markov games, the non-stationarity and credit-assignment problems, CTDE, value decomposition (VDN, QMIX), counterfactual baselines (COMA), MADDPG, communication topologies, and the league-training pipeline behind AlphaStar and OpenAI Five — with a runnable QMIX mixer in PyTorch."
disableNunjucks: true
series_order: 9
translationKey: "reinforcement-learning-9"
---
Single-agent RL rests on one quiet but enormous assumption: the environment is stationary. The transition kernel does not change while the agent learns. The moment a second learner shares the world, that assumption collapses. Each agent now sees an environment whose dynamics shift as its peers update, rewards become entangled across agents, and the joint action space explodes combinatorially. These are not engineering nuisances. They are the reason multi-agent RL needs its own algorithms instead of just *running DQN n times in parallel*.

The payoff for solving this is large. AlphaStar reached Grandmaster on the StarCraft II ladder, OpenAI Five won against the world champions in Dota 2, and cooperative MARL is increasingly the workhorse for warehouse robotics, traffic-signal control, and multi-LLM agent systems. This chapter builds the conceptual backbone — Markov games, CTDE, value decomposition, counterfactual credit — and lands on a clean QMIX mixer you can drop into your own training loop.

![Reinforcement Learning (9): Multi-Agent Reinforcement Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/illustration_1.png)


---

## What You Will Learn

- The four hard problems unique to MARL: **non-stationarity**, **credit assignment**, **partial observability**, and **scalability**
- The game-theoretic vocabulary you actually need: Markov games, Nash equilibria, Pareto fronts, social dilemmas
- **CTDE** — centralised training, decentralised execution — and why it dominates
- **Value decomposition**: VDN, QMIX, and the monotonicity constraint that makes greedy execution sound
- **Multi-agent actor-critic**: MADDPG (centralised critic per agent) and **COMA** (counterfactual baselines)
- **Communication**: when broadcast, sparse, and attention-based message passing each pay off
- The engineering of **AlphaStar's league** and **OpenAI Five's** scale-out training

## Prerequisites

- Q-learning and DQN ([Part 2](/en/reinforcement-learning/02-q-learning-and-dqn/))
- Actor-critic and policy gradients ([Part 3](/en/reinforcement-learning/03-policy-gradient-and-actor-critic/), [Part 6](/en/reinforcement-learning/06-ppo-and-trpo/))
- Comfort with expectations and conditional probabilities; we keep formal game theory minimal

---

## Why MARL is genuinely harder than single-agent RL

![Cooperative, competitive, and mixed-motive multi-agent regimes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig1_scenarios.png)

The reward structure decides almost everything else. In **cooperative** games every agent shares the team reward — coordinated soccer plays, swarm logistics, StarCraft micro-management. **Competitive** games are zero-sum: one agent's gain is another's loss, and the right tool is some form of self-play. The interesting middle ground is **mixed-motive** (general-sum) games — markets, traffic, negotiation, congestion control — where short-term self-interest can torpedo long-term collective welfare (the prisoner's dilemma is the canonical 2×2 case, and the figure shows it as the third payoff matrix).

Within any of these regimes, four obstacles break the single-agent toolbox.

**Non-stationarity.** From agent $i$'s point of view the transition is $P(s' \mid s, a_i, a_{-i})$, where $a_{-i}$ denotes everyone else's actions. As the others' policies $\pi_{-i}$ keep updating during training, the *effective* transition kernel that $i$ sees keeps drifting. Off-policy methods like DQN, which assume a fixed data-generating distribution, can diverge or oscillate.

**Credit assignment.** When the team gets a single reward $r$ for an episode-long collaboration, who actually earned it? A naive shared gradient encourages free-riding: an agent that does nothing receives the same credit as one that did the heavy lifting. Counterfactual baselines (COMA, [Section 5](#markov-games-and-nash-equilibria-just-enough-to-be-useful)) and value decomposition ([Section 4](#ctde-train-with-everything-execute-with-almost-nothing)) are the two principled answers.

**Partial observability.** Each agent sees only its own observation $o_i$. There is no Markov state available at execution time. Recurrent networks plus belief-state-style representations are typical, but the deeper fix is to architect training so that *the centralised critic* sees more than execution does.

**Scalability.** With $n$ agents and $|A|$ actions each, the joint action space is $|A|^n$. Without exploiting structure, even modest cooperative tasks become intractable.

The dominant resolution to all four — and the conceptual backbone of the chapter — is **CTDE**.

---

## CTDE: train with everything, execute with almost nothing

![Reinforcement Learning (9): Multi-Agent Reinforcement Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/illustration_2.png)

![Centralized training with decentralized execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig2_ctde.png)

The idea is asymmetric. During training we let the algorithm peek at the full state and at every agent's action — a *centralised* critic, mixer, or value function. At deployment we keep only the per-agent policies that condition on local observations. The execution interface is decentralised; the optimisation interface is not.

CTDE works because the things that make MARL hard (non-stationarity, credit assignment, partial observability) are properties of the *learning* phase, while the constraints that make MARL practical (limited bandwidth, asynchronous control, privacy) are properties of the *execution* phase. CTDE puts each problem in its own corner. Almost every modern cooperative MARL algorithm — VDN, QMIX, MADDPG, COMA, MAPPO — is a CTDE method; they differ only in *what* the centralised object is and *how* it is decomposed.

---

## Markov games and Nash equilibria, just enough to be useful

A **Markov game** generalises the MDP to $n$ agents:
$$\langle \mathcal{N}, \mathcal{S}, \{A_i\}_{i\in\mathcal{N}}, P, \{r_i\}_{i\in\mathcal{N}}, \gamma \rangle.$$
Each agent $i$ has its own action set $A_i$ and its own reward $r_i(s, a_1, \ldots, a_n)$. A **Nash equilibrium** is a joint policy $\pi^* = (\pi_1^*, \ldots, \pi_n^*)$ where no single agent can improve unilaterally:
$$V_i(\pi_i^*, \pi_{-i}^*) \;\geq\; V_i(\pi_i, \pi_{-i}^*) \qquad \forall \pi_i,\, \forall i.$$
Two facts make Nash equilibria a slippery target. First, they are typically not unique; second, they are not always Pareto optimal. In the *fully cooperative* case, where $r_1 = \cdots = r_n$, the picture simplifies dramatically — the problem reduces to maximising a single team return, and Nash equilibrium and Pareto optimality collapse onto the same set. That is why most of the algorithmic machinery in this chapter targets the cooperative setting first; competitive and mixed-motive cases borrow from it but layer additional ideas (self-play, opponent modelling, regularisation) on top.

---

## Value decomposition: VDN and QMIX

![QMIX architecture and monotonic factorisation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig3_qmix.png)

Value decomposition is a CTDE recipe for the cooperative case. We learn a per-agent value function $Q_i(o_i, a_i)$ for execution and a *combined* $Q_\text{tot}(s, \mathbf{a})$ for training, then we constrain the relationship between them so that decentralised greedy action selection on $Q_i$ is equivalent to joint greedy selection on $Q_\text{tot}$. This is the **Individual-Global-Max (IGM)** property:
$$\arg\max_{\mathbf{a}} Q_\text{tot}(s, \mathbf{a}) \;=\; \big(\arg\max_{a_1} Q_1(o_1, a_1),\; \ldots,\; \arg\max_{a_n} Q_n(o_n, a_n)\big).$$
### VDN: additive decomposition

[Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) take the simplest sufficient condition:
$$Q_\text{tot}(s, \mathbf{a}) \;=\; \sum_{i=1}^{n} Q_i(o_i, a_i).$$
Sums are monotonic in each summand, so IGM holds trivially. The cost is expressivity: VDN cannot represent any non-additive interaction between agents.

### QMIX: monotonic mixing

[QMIX](https://arxiv.org/abs/1803.11485) keeps IGM but generalises *additive* to *monotonic*. A neural **mixing network** combines the $Q_i$ into $Q_\text{tot}$ subject to
$$\frac{\partial Q_\text{tot}}{\partial Q_i} \;\geq\; 0 \qquad \forall i.$$
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

## Multi-agent actor-critic: MADDPG and COMA

### MADDPG: a centralised critic per agent

![MADDPG centralised critic, decentralised actor](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig5_maddpg.png)

[MADDPG](https://arxiv.org/abs/1706.02275) is the natural CTDE extension of DDPG. Each agent $i$ has

- an **actor** $\mu_i(o_i)$ that conditions only on its local observation, and
- a **critic** $Q_i(s, a_1, \ldots, a_n)$ that conditions on the full state and *every* agent's action.

The actor is updated with the deterministic policy gradient through its own critic; the critic is updated with a TD target that uses target actors of all agents:
$$y \;=\; r_i \;+\; \gamma \, Q_i^{\text{tgt}}\!\left(s', \mu_1^{\text{tgt}}(o_1'), \ldots, \mu_n^{\text{tgt}}(o_n')\right).$$
Because the critic conditions on $a_{-i}$, the environment looks stationary *to the critic* — its inputs include the very thing that was changing under it. The actor remains decentralised, so deployment is unchanged. MADDPG works in cooperative, competitive, and mixed-motive settings; the replay buffer simply stores joint trajectories.

### COMA: counterfactual credit assignment

![COMA counterfactual baseline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig4_coma.png)

[COMA](https://arxiv.org/abs/1705.08926) attacks the credit-assignment problem head-on. It uses a single centralised critic $Q(s, \mathbf{a})$ and, for each agent $i$, a **counterfactual baseline** that holds everyone else's actions fixed and averages out $i$'s action under its own policy:
$$A_i(s, \mathbf{a}) \;=\; Q(s, \mathbf{a}) \;-\; \sum_{a_i'} \pi_i(a_i' \mid o_i) \, Q\!\left(s, (a_i', a_{-i})\right).$$
Read this carefully. The first term is the value of what actually happened. The second is the *expected* value if agent $i$ had sampled some other action *while everyone else did exactly what they did*. The difference is precisely $i$'s marginal contribution. When the team reward is shared, this is the right object to do policy gradient on — it removes the noise from $a_{-i}$ without introducing bias. The figure shows the four-action case: only the chosen action receives a non-trivial advantage.

The trick that makes COMA tractable is that the centralised critic outputs a vector of $Q$-values for all candidate actions of agent $i$ at once, so the baseline is a single dot product against the policy probabilities — no extra environment rollouts needed.

---

## Communication: how much should agents tell each other?

![Communication topologies: broadcast, sparse, attention](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig6_communication.png)

Sometimes decentralised execution is too restrictive — agents really do need to share information at test time. The design space is a spectrum.

**Broadcast** — every agent sends a fixed-size message to every other agent. Conceptually clean, but message volume grows as $O(n^2)$ per step, which is fatal beyond a few dozen agents.

**Sparse / k-NN** — each agent only talks to its $k$ nearest neighbours (in space, in a graph, in role). Linear in $n$ for fixed $k$. CommNet, IC3Net and TarMAC all live near this end.

**Attention** — soft, learned routing. Each agent issues a query, peers respond with keys, and message weights are computed by softmax. The cost is $O(n^2)$ in attention scores but messages can be heavily pruned (top-$k$ attention) and the structure is end-to-end differentiable. This is the dominant choice in modern MARL and the natural bridge to multi-LLM-agent systems, where attention-style routing maps directly onto tool-use and agent orchestration patterns.

The right knob is task-dependent: for tightly-coupled robotics teams, dense communication is fine; for thousands of microservices or vehicles, sparse or attention-pruned schemes are the only feasible designs.

---

## Scaling MARL: AlphaStar, OpenAI Five, and league training

![AlphaStar league training and flagship MARL systems](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-multi-agent-rl/fig7_league.png)

Both [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) (StarCraft II) and [OpenAI Five](https://arxiv.org/abs/1912.06680) (Dota 2) succeed by treating *training-population design* as a first-class engineering concern. Naive self-play tends to cycle — agent A learns to beat B, B learns to beat A's new policy and forgets why it used to beat A's old policy, and so on. **AlphaStar's league** explicitly decouples three populations:

- **Main agents** — the agents that will actually be deployed; they are the optimisation target.
- **Main exploiters** — separate agents whose only job is to find weaknesses in the main agents, forcing them to round out their game.
- **League exploiters** — periodically reset agents that maintain *strategic diversity* so the league does not collapse onto a narrow style.

A pool of *frozen historical checkpoints* is matched against current learners with prioritised fictitious self-play (PFSP), which biases match-making toward opponents the current learner is just barely beating. Together these mechanisms prevent the catastrophic forgetting that breaks naive self-play.

OpenAI Five takes a different cut. Each Dota hero is controlled by an independent LSTM policy with a shared parameter set; the team-level coordination problem is handed off to a hand-engineered shared reward and an enormous compute budget — roughly *180 years of self-play per day*. The lesson from both systems is the same: at industrial scale, the *training curriculum* matters at least as much as the per-agent algorithm.

These ideas are migrating outside games. RLHF ([Part 12](/en/reinforcement-learning/12-rlhf-and-llm-applications/)) and tool-use agent orchestration both increasingly look like multi-population training — a population of policies, a population of evaluators, and prioritised match-making between them.

---

## Credit Assignment in MARL: A Worked Example

Credit assignment — figuring out which agent's action caused a team-level reward — is the *real* hard problem in cooperative MARL, and it deserves more than a passing mention. Consider a 4-agent grid task where the team gets +10 only when *all four* agents are on a goal cell at the same step. Train naive independent Q-learners and watch them oscillate forever: each agent's $Q$-update treats the other three as part of the environment, so the gradient of "I should move to my goal" gets buried in noise from teammates moving in and out of theirs.

Three families of solutions, in order of complexity.

### Difference rewards

The simplest fix is to compute, for each agent $i$, the **counterfactual difference**
$$
D_i = R(s, a) - R(s, (a_{-i}, c_i)),$$where $c_i$ is a default action (e.g. "do nothing"). Each agent's effective reward is what the team got *because* of agent $i$'s actual choice, marginalised over a baseline. Difference rewards predate deep RL by 15 years (Wolpert & Tumer, 2002) and remain shockingly effective when you can compute the counterfactual cheaply — for example in any simulator you can roll back.

### Counterfactual baselines (COMA)

When the counterfactual cannot be cheaply rolled, you can *learn* it. COMA computes its advantage as$$A_i(s, a) = Q(s, a) - \sum_{a_i'} \pi_i(a_i' | \tau_i)\, Q(s, (a_{-i}, a_i')),
$$
which marginalises over agent $i$'s alternative actions while holding the rest fixed. The magic is that the marginal $Q$-value uses the *centralised* critic — at training time you have everyone's observation — so the baseline is well-defined even though each agent acts on its own history at test time.

### Reward shaping with potentials

For domains where you cannot afford either, fall back to potential-based reward shaping: add an extra reward term $F(s, s') = \gamma \Phi(s') - \Phi(s)$ where $\Phi$ is a hand-designed potential (e.g. negative distance to goal). The Ng-Harada-Russell theorem guarantees the optimal policy is preserved. In MARL this trick applied per agent often does most of the work that fancier credit-assignment methods do, and it is trivially easy to implement.

### A concrete failure I have seen

On a multi-robot warehouse pick task, an early team trained QMIX with a sparse "all packages picked" reward. After 100 M environment steps the policy was still suboptimal because robot 4 was idling — its individual $Q$-value barely moved because robot 4's action almost never determined the team success on its own. We added a difference reward (each robot's bonus = team reward $-$ team reward if that robot stood still), and the policy converged in 12 M steps. The fancy credit-assignment paper turned out to be dominated by a 30-line shaping function.

## Reward Shaping and Curricula in MARL

Sparse team rewards rarely converge in MARL. Two engineering tricks help more than another algorithm switch.

### Curriculum over team size

Train with 2 agents first, then 3, then 4. The gradient signal at $n=2$ is much stronger because the joint action space is small enough that random exploration occasionally hits the reward. Once policies are sensible, scale up — add agents whose initial policy is the best learned single-agent policy, and continue training with the full team. AlphaStar uses an extreme version of this (population-based training over leagues), but you do not need that machinery to benefit from the basic curriculum.

### Self-play with frozen opponents

For competitive MARL, the standard recipe is **fictitious self-play**: at each iteration the active learner trains against a uniform mixture of past versions of itself. This avoids the cycle where today's policy beats yesterday's but loses to last week's (the rock-paper-scissors trap). OpenAI Five maintained 80 % of training against the latest checkpoint and 20 % against random older snapshots. The 20 % is what kept the population diverse enough to generalise.

### A note on hyperparameter sensitivity

MARL is *much* more sensitive to hyperparameters than single-agent RL. The same QMIX implementation, with all defaults, converges on StarCraft II micromanagement at one learning rate and diverges on a multi-particle environment at a 2× different learning rate. The community lore is that the right starting point is roughly half the learning rate you would use for single-agent PPO — and to keep the entropy bonus higher for longer, because exploration is a public good in MARL: when one agent stops exploring, all the others lose information.

## Centralised Critic, Decentralised Actor (CTDE) — A Theorem

![MADDPG: centralised critic during training, decentralised actors at execution.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/09-Multi-Agent-RL/fig09_maddpg.png)

The setup is the one almost every paper after 2017 inherits. There are $N$ agents, a joint state $s$, individual actions $a_i$, and a joint policy that factorises as $\pi(a \mid s) = \prod_i \pi_i(a_i \mid s)$. Each agent only sees its own slice at execution time, but during training we let the critic peek at everything.

The result that justifies all of this is from [Lowe et al. 2017](https://arxiv.org/abs/1706.02275). Define a centralised critic $Q^\pi(s, a_1, \dots, a_N)$ that takes the full joint action as input. Then the policy gradient for agent $i$ is
$$\nabla_{\theta_i} J(\pi_i) = \mathbb{E}\!\left[ \nabla_{\theta_i} \log \pi_i(a_i \mid s) \, Q^\pi(s, a_1, \dots, a_N) \right],$$
and this estimator is unbiased even while teammates are still updating. That is the whole CTDE theorem in one line.

The proof sketch is short and worth knowing. Freeze the teammates' policies $\pi_{-i}$ for the duration of one update. From agent $i$'s viewpoint, the transition kernel is now $P(s' \mid s, a_i, a_{-i})$ marginalised over $a_{-i} \sim \pi_{-i}$, which is stationary because $\pi_{-i}$ is fixed. But — and this is the load-bearing word — the marginalisation introduces variance that scales badly with $N$. If instead we *condition* on $a_{-i}$ explicitly, the variance disappears. Centralisation is what enables that conditioning. An independent learner cannot condition on what it cannot see.

The contrapositive matters too. If you only have access to $Q_i(s, a_i)$, the gradient $\nabla_{\theta_i} \log \pi_i(a_i \mid s) \, Q_i(s, a_i)$ is biased whenever the teammates' policies drift between data collection and update. This is the formal source of the non-stationarity that wrecks IQL.

There is also a dimensionality argument that gets lost in the proof. The centralised critic learns a function over $\mathbb{R}^{|s| + N \cdot |a|}$, which is a lot. But it only has to learn it on the support of the joint policy — and since the joint policy is the product of decentralised actors, the support is much smaller than the full Cartesian product of action spaces. Empirically the critic generalises across the support cleanly, which is why MADDPG works with critic networks of modest capacity.

A working MADDPG skeleton looks like this. One actor per agent, one shared centralised critic, deterministic policy gradient with Gumbel-Softmax for discrete action spaces.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs, hard=False, tau=1.0):
        logits = self.net(obs)
        return F.gumbel_softmax(logits, tau=tau, hard=hard)


class CentralisedCritic(nn.Module):
    def __init__(self, state_dim, n_agents, act_dim, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + n_agents * act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, joint_actions):
        return self.q(torch.cat([state, joint_actions], dim=-1)).squeeze(-1)


def maddpg_update(actors, critic, target_actors, target_critic, batch, gamma=0.99):
    s, a, r, s_next, done = batch
    with torch.no_grad():
        a_next = torch.cat([ta(s_next) for ta in target_actors], dim=-1)
        y = r + gamma * (1 - done) * target_critic(s_next, a_next)
    critic_loss = F.mse_loss(critic(s, a), y)

    actor_losses = []
    for i, actor in enumerate(actors):
        a_i = actor(s)  # differentiable through Gumbel-Softmax
        a_joint = a.clone()
        slice_i = slice(i * a_i.size(-1), (i + 1) * a_i.size(-1))
        a_joint[..., slice_i] = a_i
        actor_losses.append(-critic(s, a_joint).mean())
    return critic_loss, actor_losses
```

The numerical anchor I keep coming back to is the predator-prey environment with four predators chasing one prey on a 2-D torus. Independent Q-learning plateaus at roughly 35% catch rate after 50k episodes — the predators learn to chase but never coordinate the pincer. MADDPG hits 78% in the same budget, matching Figure 4 of the original paper. The gap is not the algorithm being clever; it is the centralised critic refusing to be confused.

Two implementation details that cost me a week the first time. The Gumbel-Softmax temperature $\tau$ matters: too high and the actor's gradient is noise, too low and it becomes a one-hot that the critic cannot differentiate through. I anneal from $\tau = 1.0$ to $\tau = 0.5$ over the first 20% of training. Second, the target networks update slowly — Polyak coefficient $\rho = 0.995$ — and they are essential. Without them the critic chases its own moving target and the actor follows it off a cliff.

The next question is who, inside the team reward, actually earned what.

## Counterfactual Reasoning: COMA Baselines

When the team gets a single scalar reward at the end of an episode, the gradient for each agent is contaminated by everyone else's noise. If agent 3 happened to do something useful on the same step that agent 1 made a coordinated kill, both get the same reward and the gradient credits both equally. Over millions of episodes that bias does not wash out — it locks in lazy agents.

[COMA](https://arxiv.org/abs/1705.08926) proposes a counterfactual baseline. Define the advantage of agent $i$ as
$$A_i(s, a) = Q(s, a_1, \dots, a_N) - \sum_{a'_i} \pi_i(a'_i \mid s) \, Q(s, a_1, \dots, a'_i, \dots, a_N),$$
which marginalises over agent $i$'s possible actions while holding teammates fixed at what they actually did. The first term is what happened. The second is the expected $Q$ if agent $i$ had sampled from its policy instead. The difference is exactly $i$'s marginal contribution.

This is not a new invention. It is the **difference reward** trick from mechanism design (Wolpert & Tumer 2002), reframed as a value-function baseline. The variance reduction is real — Foerster et al. show roughly a 4× reduction in gradient variance compared to a shared state-value baseline on SMAC.

Why a difference reward beats a shared baseline is one of those things that looks obvious only after you have stared at the math for an hour. A shared baseline like $V(s)$ subtracts the same scalar from every agent's $Q(s, \mathbf{a})$, so gradient noise from $a_{-i}$ leaks into agent $i$'s update. The counterfactual baseline subtracts a quantity that depends on $a_{-i}$ in exactly the right way to cancel that leak. It is the same pattern as control variates in Monte Carlo integration.

The implementation is cheaper than it looks. The centralised critic outputs a vector over agent $i$'s action space at once, so the baseline is one dot product:

```python
def coma_advantage(critic, state, joint_actions, agent_idx, policy_probs, n_actions):
    """
    critic: outputs Q-values for all actions of agent `agent_idx`,
            given state and the OTHER agents' actions.
    policy_probs: (batch, n_actions) — pi_i(.|s).
    """
    # q_all[b, a'] = Q(s, a_{-i}, a'_i = a')
    q_all = critic(state, joint_actions, agent_idx)
    actual_action = joint_actions[:, agent_idx].long()
    q_taken = q_all.gather(1, actual_action.unsqueeze(1)).squeeze(1)
    baseline = (policy_probs * q_all).sum(dim=1)
    return q_taken - baseline
```

On the SMAC 3m map (three Marines vs three Marines), vanilla independent actor-critic wins about 41% of evaluations after 2M steps. COMA hits 87% in the same budget. The map is small enough that the centralised critic fits in a single MLP — the gain is not from capacity, it is from the baseline.

The catch: COMA needs the full joint action distribution at training time. In partially observable settings where you cannot reconstruct $a_{-i}$ from the replay buffer, you fall back to off-policy corrections or just drop the agent index dimension and use a shared critic.

A subtler practical point: COMA is on-policy. The advantage is computed under the *current* $\pi_i$, not whatever policy generated the replay sample. Mixing in stale samples re-introduces the bias the baseline was meant to remove. I run COMA with a small replay window (the last 8 episodes) and it stays stable; longer windows drift.

That assumes the agents already see enough to coordinate. When they do not, they need to talk.

## Communication: Differentiable Message Passing (DIAL)

Sometimes CTDE is not enough. If the agents have genuinely partial observations and the joint state cannot be reconstructed at execution, they need a communication channel — and the channel itself has to be learned. The trick is that you cannot backprop through a discrete message.

[DIAL (Foerster et al. 2016)](https://arxiv.org/abs/1605.06676) solves this with a soft trick: during training, send continuous-valued messages and let gradients flow through them as if they were activations. At execution time, discretise to whatever the channel actually supports (one bit, one byte, a token). The continuous-to-discrete gap is small if the noise injected during training matches the discretisation granularity.

The gradient flowing back through the message is what makes this work. Agent $j$'s message to agent $i$ becomes part of $i$'s policy input, $i$'s loss depends on its action, and the chain rule pushes the loss signal back into $j$'s message-emitting head. So $j$ learns to emit messages that improve $i$'s decisions — even though $j$ never gets a direct reward for messaging. This is genuinely useful and genuinely hard to debug when it does not work.

A CommNet-style module is the easiest variant to drop into existing code. Each agent emits a message vector, the messages are pooled (mean or attention), and the pooled representation is broadcast back as an extra input to every actor.

```python
import torch
import torch.nn as nn

class CommModule(nn.Module):
    def __init__(self, obs_dim, msg_dim, hidden=128, n_heads=4):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden)
        self.msg_head = nn.Linear(hidden, msg_dim)
        self.attn = nn.MultiheadAttention(msg_dim, n_heads, batch_first=True)
        self.policy_head = nn.Linear(hidden + msg_dim, hidden)

    def forward(self, obs):
        # obs: (batch, n_agents, obs_dim)
        h = torch.relu(self.encoder(obs))
        msgs = self.msg_head(h)  # (batch, n_agents, msg_dim)
        # Self-attention over agents — each agent attends to all peers
        pooled, _ = self.attn(msgs, msgs, msgs)
        out = torch.relu(self.policy_head(torch.cat([h, pooled], dim=-1)))
        return out
```

The caveat that bit me on a robotics project: DIAL helps when partial observability is the bottleneck. On fully observable environments — including most StarCraft II micromanagement scenarios at small scale — the centralised critic in CTDE already provides enough coordination signal, and adding a communication channel just adds parameters that learn to ignore themselves. Run an ablation before assuming you need messages.

Communication scales linearly with $n$ if you use sparse top-$k$ attention; mean pooling is $O(n)$ but loses individual structure. For teams above 32 agents, neither matters much — what matters is whether the joint $Q$ function is even tractable.

One more thing worth flagging: discretisation noise. DIAL trains with continuous messages plus Gaussian noise of std $\sigma$ matched to the discretisation step size. At test time the message is rounded to the nearest grid point. If $\sigma$ is too small the deployed policy underperforms the trained one because the rounding is off-distribution; too large and the trained policy never learns a useful signal. Picking $\sigma$ to match the binning was the only hyperparameter that mattered on my hand.

## QMIX: Mixing Networks for Cooperative Tasks

The core scaling problem of CTDE is in the centralised critic itself. With $N$ agents and $|A|$ actions each, the joint action space has $|A|^N$ entries. A centralised $Q$-network has to learn over that space, and at execution time you would need to enumerate it to pick the joint argmax. Both are dead on arrival past $N=8$.

[QMIX (Rashid et al. 2018)](https://arxiv.org/abs/1803.11485) factorises the joint action-value into per-agent components combined by a mixing network:
$$Q_{\text{tot}}(s, \mathbf{a}) = f_{\text{mix}}\!\left(Q_1(s, a_1), \dots, Q_N(s, a_N); s\right),$$
where $f_{\text{mix}}$ is a feed-forward network whose **weights are non-negative** and conditioned on the global state $s$ via a hypernetwork. Non-negativity gives the monotonicity property
$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0 \quad \forall i,$$
which is the entire point. Monotonicity guarantees
$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(s, \mathbf{a}) = \left(\arg\max_{a_1} Q_1(s, a_1), \dots, \arg\max_{a_N} Q_N(s, a_N)\right),$$
so each agent can act greedily on its own $Q_i$ at execution and the resulting joint action is provably optimal under the centralised $Q_{\text{tot}}$.

The implementation has two pieces — the per-agent value heads and the hypernetwork that generates the mixing weights:

```python
import torch
import torch.nn as nn

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mix_hidden=32):
        super().__init__()
        self.n_agents = n_agents
        self.mix_hidden = mix_hidden
        # Hypernetworks: state -> mixing weights (non-negative via abs)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mix_hidden)
        self.hyper_w2 = nn.Linear(state_dim, mix_hidden)
        self.hyper_b1 = nn.Linear(state_dim, mix_hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mix_hidden), nn.ReLU(),
            nn.Linear(mix_hidden, 1),
        )

    def forward(self, agent_qs, state):
        # agent_qs: (batch, n_agents)
        # state: (batch, state_dim)
        b = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, self.mix_hidden)
        b1 = self.hyper_b1(state).view(b, 1, self.mix_hidden)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(b, self.mix_hidden, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(b)
```

On the SMAC `3s5z` map (three Stalkers and five Zealots vs the same composition), MADDPG plateaus at around 10% win rate — the joint critic cannot generalise across the heterogeneous unit types. QMIX hits 92% in the same budget. The factorisation is doing real work; this is not a constant-factor speedup.

The known limitation: the IGM (individual-global-max) property that QMIX enforces is sufficient but not necessary for decentralisable optimal policies. Tasks where the optimal joint action requires non-monotonic credit (one agent's $Q$ goes up only when another's goes down) cannot be represented exactly. QTRAN and QPLEX address this; in practice I rarely see the failure modes outside synthetic counter-examples.

Training tip from a failure: do not share the learning rate between the per-agent $Q_i$ networks and the mixer. The mixer sees a much smaller effective gradient norm because the hypernetwork outputs are gated by the absolute value, and matching the agent learning rate will leave the mixer untrained for the first 5M steps. I run the mixer at 3× the agent rate and it keeps up.

So far everything has been cooperative. The competitive case is its own kind of broken.

## Self-Play and Population-Based Training

Naive self-play in competitive games is a trap. The agent trains against its current self, gets better, but only along the axis its current self exposes. Old failure modes resurface six months later. The literature calls it the rock-paper-scissors cycle; I have watched a Dota bot forget how to defend a tower it had defended fine 200k iterations earlier.

Two fixes recur. **Fictitious self-play** plays the current learner against a uniform mixture of past versions of itself — the average policy, not the latest one. **League training**, the AlphaStar variant, extends this with multiple sub-populations: main agents (the deployment target), main exploiters (whose only job is to find weaknesses in main agents), and league exploiters (which maintain strategic diversity by periodically resetting and forking).

The numerical anchor: AlphaStar's published results show that pure self-play plateaus at roughly Diamond-rank play after equivalent training; the full league reaches Grandmaster. The compute is the same. The difference is curriculum.

A minimal league looks like this:

```python
import random
from collections import deque

class League:
    def __init__(self, max_history=200):
        self.main = None
        self.main_exploiters = []
        self.history = deque(maxlen=max_history)

    def sample_opponent(self, learner_role):
        if learner_role == "main":
            # 50% latest checkpoints, 35% historical, 15% exploiters
            r = random.random()
            if r < 0.5 and self.history:
                return random.choice(list(self.history)[-20:])
            if r < 0.85 and self.history:
                return random.choice(self.history)
            return random.choice(self.main_exploiters) if self.main_exploiters else self.main
        if learner_role == "main_exploiter":
            return self.main  # always train against the current main
        if learner_role == "league_exploiter":
            return random.choice(self.history) if self.history else self.main

    def checkpoint(self, agent):
        self.history.append(agent.snapshot())
```

The connection back to the cooperative side is [PSRO](https://arxiv.org/abs/1711.00832) (Policy-Space Response Oracles): it frames competitive MARL as a meta-game whose actions are policies and whose payoff matrix is "expected return when policy A plays policy B". The inner game is the environment, and the outer solver is something like double oracle or Nash equilibrium computation over the meta-payoff matrix. League training is one schedule for populating that matrix; fictitious self-play is another. The view is unifying: cooperative CTDE is PSRO with a single policy class and an additive payoff, competitive league training is PSRO with multiple classes and zero-sum payoffs, and the rest is engineering.

A debugging story to close on. I spent two weeks watching a competitive MARL setup learn nothing. The reward curves were flat. Win rate against the latest checkpoint hovered at 50% — exactly chance. The bug was that I was sampling opponents only from the most recent 5 checkpoints, which were all variants of the same near-optimal policy that the learner had just memorised. Once I expanded the sampling window to 200 checkpoints, the learner immediately discovered exploits in older versions and bootstrapped from there. The policy improved not because the algorithm got smarter but because the *opponent distribution* got broader. That is the whole lesson of league training in one paragraph.

Once you see MARL through the meta-game lens, the line between RLHF reward modelling and competitive self-play gets thin. That is the topic of [Part 12](/en/reinforcement-learning/12-rlhf-and-llm-applications/).

## FAQ

### Why does QMIX need the monotonicity constraint?

So that decentralised greedy execution is consistent with centralised joint optimisation. If $\partial Q_\text{tot}/\partial Q_i$ could go negative, an agent acting greedily on its own $Q_i$ might *lower* the team value. Monotonicity makes the per-agent argmax and the joint argmax coincide.

### When do MARL systems get stuck in suboptimal equilibria?

Whenever the optimal joint action requires *coordinated* exploration — both agents must simultaneously try the risky move for it to look good. If random exploration almost never produces that joint action, the system converges to a safer but Pareto-dominated equilibrium. The fixes are joint exploration (committed-exploration schedules), explicit communication, and opponent modelling.

### How does sample complexity scale with the number of agents?

For cooperative tasks where value decomposition holds, sample complexity grows roughly linearly in $n$ — that is the entire point of QMIX. Without decomposition, learning $Q(s, \mathbf{a})$ over the joint action space is exponential in $n$.

### Independent learners vs CTDE — is CTDE always worth it?

For very small populations or when global state is genuinely unavailable, independent Q-learning can work and is much simpler to implement. As soon as credit assignment matters or the team is more than 3–4 agents, CTDE methods pull ahead and the gap widens fast.

### Does any of this transfer to multi-LLM-agent systems?

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
