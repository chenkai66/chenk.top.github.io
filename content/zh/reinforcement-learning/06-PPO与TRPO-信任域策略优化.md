---
title: "强化学习（六）：PPO 与 TRPO —— 信任域策略优化"
date: 2025-08-26 09:00:00
tags:
  - Reinforcement Learning
  - PPO
  - TRPO
  - Policy Optimization
  - Trust Region
  - RLHF
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "PPO 和 TRPO 的完整推导：从策略优化的不稳定性到信任域约束，PPO 的裁剪技巧，以及 PPO 在 RLHF 中的关键角色。"
disableNunjucks: true
series_order: 6
series_total: 12
translationKey: "reinforcement-learning-6"
---
策略梯度（第三部分）直接优化策略，绕开了离散的 `argmax` 操作，还能自然处理随机策略。但它存在一个致命缺陷：**一次过大的更新就可能彻底摧毁策略**。更糟的是，由于数据分布与策略紧密耦合，一旦崩溃，几乎无法恢复。

**信任域方法**精准地抓住了问题核心：每次更新应限制*行为*的变化，而非参数的变化。TRPO 通过硬性的 KL 散度约束和二阶优化器实现了这一点；而 PPO 则仅用一行带裁剪（clip）的简单算术，就模拟出了类似效果。最终，这个更简单的技巧胜出了——PPO 不仅训练出了 OpenAI Five，还支撑了 ChatGPT 的 RLHF 阶段，并成为几乎所有现代机器人策略的基石，至今仍是应用深度强化学习领域的主力算法。
![强化学习（六）：PPO 与 TRPO：信任域策略优化 — 章节概览图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/illustration_1.png)


---

## 你将学到什么

- 为什么朴素策略梯度会**灾难性地**不稳定，并附带一个详细示例
- **重要性采样**如何作为桥梁，让我们能够复用离策略数据
- **TRPO**：单调改进下界、自然梯度、共轭梯度法与线搜索
- **PPO-Clip** 与 **PPO-Penalty**：两种用一阶优化近似信任域的方法
- **PPO 在 RLHF 中的应用**：它如何对齐 ChatGPT 级别的模型，以及 DPO/IPO/KTO 的定位
- 一份实用的超参数调优与调试指南，内容贴近 `cleanrl` 或 Stable Baselines 3 的实践

**前置知识**：[第三篇](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)（REINFORCE、Actor-Critic、优势函数）。熟悉 KL 散度和 Fisher 信息会有帮助，但非必需——文中会在上下文中介绍它们。

---

## 策略更新为什么会不稳定

### 一个病态示例

假设当前策略在状态 $s$ 下，$\pi(a_1|s) = 0.9$，$\pi(a_2|s) = 0.1$。不幸的是，我们采样到了动作 $a_2$，而环境恰好返回了 $+100$ 的奖励。REINFORCE 估计器会将得分函数 $\nabla_\theta \log \pi(a_2|s) = -1/\pi(a_2|s)\cdot\nabla_\theta\pi(a_2|s)$ 乘以 $+100$，然后执行一次梯度更新。结果，$a_2$ 的概率可能从 $0.1$ 一步跳到 $0.7$，尽管从期望来看，$a_2$ 可能远不如 $a_1$。

这里同时存在三个病态问题：

1. **方差爆炸**。得分函数包含 $1/\pi$ 因子，因此罕见动作会引发巨大的梯度幅值——这也是“离策略” REINFORCE 不稳定的原因。
2. **分布偏移**。下一批轨迹由*新策略*收集。如果新策略变差，所有新数据都会强化这个错误信号，形成恶性循环。
3. **不可逆性**。监督学习模型在糟糕的更新后只是暂时失去拟合能力；而强化学习智能体在糟糕的更新后会丢失*数据*本身，策略崩溃所需的修正样本量，往往比造成崩溃所需的样本量高出一个数量级。

### 参数空间会骗人，策略空间才说真话

考虑两个高斯策略 $\pi_1 = \mathcal{N}(0, 0.01)$ 和 $\pi_2 = \mathcal{N}(0, 10)$。它们的参数（均值和对数标准差）在欧氏距离上很接近，但行为却天差地别：$\pi_1$ 几乎在零处确定性输出，而 $\pi_2$ 的动作则近乎均匀地分布在整片动作空间。由此产生的行为——以及回报——完全不同。

教训是：**在参数空间中固定的步长，可能导致策略行为发生无界的剧变**。因此，任何安全保证都必须建立在*分布空间*中。Kullback-Leibler 散度 $D_{KL}(\pi_{\text{old}} \| \pi_\theta)$ 是最自然的度量。

![信任域：许多个 KL 受限的小步保持安全；一次大的参数步会跌下悬崖](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig1_trust_region.png)

这张图清晰地展示了其几何含义：绿色到红色的曲面是一个假想的 $J(\theta)$ 景观，其中有一条狭窄的高回报山脊和一道陡峭的悬崖。朴素策略梯度（左）沿着梯度方向迈出一大步，直接坠入悬崖；而 TRPO（右）将每一步都约束在当前策略周围的 KL 球内，其轨迹紧贴山脊，最终收敛到一个安全的最优解。

---

## 重要性采样：连接离策略数据的桥梁

在策略方法在单次梯度更新后就会丢弃数据，因为数据分布发生了偏移。**重要性采样**通过重新加权，使我们能够复用同一批数据：
$$\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\!\left[\tfrac{q(x)}{p(x)}\, f(x)\right]$$
将旧策略和新策略代入策略梯度目标，就得到了**代理目标**：
$$L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right]$$
其中概率比 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ 是本文所有算法的核心对象。

需要记住两点：

- 当 $\theta = \theta_{\text{old}}$ 时，$L^{\text{IS}}$ 与真实策略梯度目标 $J(\theta)$ 具有**相同的值和梯度**——因此它在局部是忠实的。
- 当远离 $\theta_{\text{old}}$ 时，重要性采样（IS）估计器的方差大致按 $\exp(2\,D_{KL})$ 增长（见下图）。KL 很小 = 复用可靠；KL 很大 = 噪声垃圾。

![重要性采样比的分布以及方差随 KL 的增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig5_importance_sampling.png)

左图展示了当新旧策略逐渐偏离时，比率 $r_t$ 的直方图分布——当 $D_{KL}\!\le\!0.02$ 时，分布紧密集中在绿色的 PPO 裁剪区间 $[0.8, 1.2]$ 内；而当 $D_{KL}\!=\!0.05$ 时，右侧出现了明显的重尾。右图以对数尺度绘制了 IS 估计器的方差；橙色阴影区域表示裁剪所“节省”下来的方差。这正是将每次更新限制在小信任域内的定量依据。

---

## TRPO：信任域策略优化

### 单调改进下界

Schulman 等人（2015）在 Kakade & Langford（2002）工作的基础上证明，新策略的真实回报可以被代理目标加上一个 KL 惩罚项所下界：
$$J(\pi_{\text{new}}) \;\geq\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) \;-\; C \cdot D_{KL}^{\max}\!\left(\pi_{\text{old}} \,\|\, \pi_{\text{new}}\right)$$
其中 $C = 4\varepsilon\gamma/(1-\gamma)^2$，依赖于最大优势幅值 $\varepsilon$ 和折扣因子 $\gamma$。其推论令人震惊：**只要我们在提升代理目标的同时保持 $D_{KL}^{\max}$ 足够小，就能保证策略单调改进**。

实践中，常数 $C$ 过于悲观而难以使用；TRPO 将惩罚项替换为硬约束，并通过实验调整。

### 带约束的优化问题
$$\max_\theta \;\; \mathbb{E}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right] \quad\text{s.t.}\quad \bar{D}_{KL}\!\left(\pi_{\text{old}} \,\|\, \pi_\theta\right) \leq \delta$$
通常取 $\delta \approx 0.01$。这里使用平均 KL 而非最大 KL，因为前者更容易从样本中估计。

### 自然梯度：何为“小”

标准 SGD 在欧氏球 $\|\Delta\theta\|^2 \le c$ 内选择使线性化目标最大的方向。**自然梯度**则改变了度量方式：它约束的是分布空间中的 *KL 球*，在局部可近似为一个二次型，其 Hessian 即为 **Fisher 信息矩阵**：
$$D_{KL}(\pi_\theta \,\|\, \pi_{\theta+\Delta\theta}) \;\approx\; \tfrac{1}{2}\,\Delta\theta^\top F\,\Delta\theta, \qquad F = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta\,\nabla_\theta \log \pi_\theta^\top\right]$$
求解该约束问题可得自然梯度更新 $\Delta\theta \propto F^{-1}\nabla J$。这是**策略空间**中的最速上升方向，而非参数空间中的方向——这正是我们想要的。

### 实现：共轭梯度 + 线搜索

对于拥有数百万参数的网络，$F$ 的元素数量高达 $\sim 10^{12}$——显式构造它完全不可行。TRPO 通过两个技巧绕过此问题：

1. **共轭梯度法**求解 $Fx = g$，仅需 Fisher-向量乘积 $Fv$（通过两次 `autograd` 反向传播即可高效计算，无需实例化矩阵）。
2. **回溯线搜索**沿自然梯度方向进行，不断减半步长，直到满足 KL 约束*且*代理目标有所提升——即使二次近似不够精确，也能保留单调性保证。

```python
def conjugate_gradient(fisher_vector_product, b, n_steps=10, tol=1e-10):
    """求解 F x = b，无需显式构造 F；代价为 n_steps 次 (Fv) 操作。"""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = r.dot(r)
    for _ in range(n_steps):
        Ap = fisher_vector_product(p)
        alpha = rdotr / (p.dot(Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

# 共轭梯度结束后，由 KL 预算 delta 唯一确定步长
step = torch.sqrt(2 * delta / (x.dot(fisher_vector_product(x)) + 1e-8)) * x
# 然后做回溯线搜索，确保 KL <= delta 且代理目标改善
```

**优点**：理论上具有单调改进保证；在 Humanoid、Ant 等复杂控制任务上极其稳定，而朴素 PG 方法在此类任务上容易发散。

**缺点**：约 300 行精心编写的代码；每次更新需约 10–20 次 Hessian-向量乘积；对同一批数据运行多轮 epoch 几乎无额外收益（CG 已利用曲率信息）；难以扩展至分布式训练，因为共轭梯度需要每步同步。

---

## PPO：用两成复杂度，拿到九成收益

![强化学习（六）：PPO 与 TRPO：信任域策略优化 — 章节小结图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/06-ppo-and-trpo/illustration_2.png)

2017 年，Schulman 及其同事提出了一个问题：*能否仅用一阶优化方法，获得类似 TRPO 的稳定性？* 答案就是 PPO，自此它便主导了该领域。

### PPO-Clip：核心技巧

定义裁剪后的代理目标：
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}\!\left(r_t(\theta),\,1\!-\!\varepsilon,\,1\!+\!\varepsilon\right)\hat{A}_t\Big)\right]$$
其中 $\varepsilon \approx 0.2$。该设计故意不对称：`min` 使目标变得**悲观**，总是选取裁剪值与未裁剪值中较小的一个。

![按优势符号拆分的 PPO 裁剪代理目标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig2_ppo_clipping.png)

两种情况及其简要总结如下：

- **$\hat{A}>0$（好动作）**。未裁剪的代理目标会随 $r_t \to \infty$ 持续增长；裁剪操作在 $r_t > 1+\varepsilon$ 时封顶奖励。`min` 会*取封顶值*，因此超过该点后梯度降为零。翻译过来就是：*不要仅凭一个 minibatch 的好运，就把某个好动作的概率推高到 $(1+\varepsilon)\pi_{\text{old}}$ 以上。*
- **$\hat{A}<0$（坏动作）**。对称地，裁剪操作限制了 $r_t < 1-\varepsilon$ 时损失能负到多深，因此一旦新策略将概率压低到足够程度，梯度就会停止。*不要因为一次坏运气，就把某个动作彻底扼杀。*

更深层的问题是：*为何用 `min` 而非 `max`？* 如果总是取较大的那个代理值，优化器会乐于奖励巨大的比率——从而轻易突破信任域。`min` 在不计算任何 KL 的情况下，就将信任域的直觉融入了目标函数。

### PPO-Penalty：自适应变体

另一种变体（虽不流行但在某些领域有用）显式添加 KL 惩罚项并自适应调整其系数：
$$L^{\text{KL}}(\theta) = \mathbb{E}\!\left[r_t(\theta)\,\hat{A}_t\right] \;-\; \beta\,\mathbb{E}\!\left[D_{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)\right]$$
其中 $\beta$ 在每次迭代后调整：若实测 KL 超过 $1.5\,\delta_{\text{target}}$ 则翻倍，低于 $\delta_{\text{target}}/1.5$ 则减半。

![自适应 KL 罚项：目标形状与 beta 的调度过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig3_kl_penalty.png)

左图展示了 $\beta$ 如何塑造目标：$\beta=0$ 退化为无约束代理目标（及其不稳定性）；大的 $\beta$ 使最优解紧贴 $\theta_{\text{old}}$。右图展示了一个真实的自适应调度过程——$\beta$ 在对数轴上跨越多个数量级变化，以将观测到的 KL 维持在目标值附近。早期 InstructGPT 内部使用的就是这种自适应 KL 方法，后来 clip 式 PPO 成为 `trl` 等库中 RLHF 的默认选择。

### 代理目标景观：为何裁剪在实践中胜出

![代理目标 vs 真实目标：未裁剪会误导优化器，裁剪能保持诚实](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig6_surrogate_landscape.png)

这张图是视觉上的核心结论。黑线是沿一维切片的*真实*回报 $J(\theta)$——它先有一个峰值，随后急剧跌入低回报区域。橙色虚线是未裁剪的 IS 代理目标，它持续攀升，会诱使 SGD 踏入悬崖。蓝线是 PPO 裁剪后的代理目标：在信任域内（绿色带），它紧贴真实目标；在域外，它被压平，消除了会将我们带下悬崖的梯度。

### 一个完整的 PPO 实现

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    """共享底座，独立的 actor / critic 头。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        x = self.shared(state)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, eps_clip=0.2, k_epochs=10, c_vf=0.5, c_ent=0.01):
        self.gamma, self.lam = gamma, lam
        self.eps_clip, self.k_epochs = eps_clip, k_epochs
        self.c_vf, self.c_ent = c_vf, c_ent
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()

    def compute_gae(self, rewards, values, dones, next_value):
        """广义优势估计 GAE（Schulman 等，2016）。"""

        advantages = []
        gae = 0.0
        values = list(values) + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages)

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        with torch.no_grad():
            values = self.policy(states)[1].squeeze().numpy()
            next_val = self.policy(torch.FloatTensor(next_state).unsqueeze(0))[1].item()

        advantages = self.compute_gae(rewards, values, dones, next_val)
        returns = advantages + torch.FloatTensor(values)
        # 按批归一化优势：成本极低，对稳定性至关重要
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            probs, vals = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            vals = vals.squeeze()

            # 在对数空间算比率，数值更稳
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(vals, returns)
            loss = policy_loss + self.c_vf * value_loss - self.c_ent * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
```

在朴素 actor-critic 基础上仅增加寥寥数行，你就能得到一个百回合内解决 CartPole、约千万步学会 Humanoid 行走的算法。

---

## TRPO 与 PPO 正面对比

| 维度 | TRPO | PPO-Clip |
|------|------|----------|
| 优化器 | 共轭梯度 + 线搜索（二阶） | Adam（一阶） |
| KL 约束 | 硬约束，严格 $\le \delta$ | 软约束，通过逐样本裁剪实现 |
| 每批数据更新次数 | 1 | 3–10 epochs |
| Hessian-向量乘积 | $\sim$10–20 | 0 |
| 代码复杂度 | $\sim$300 行 | $\sim$100 行 |
| 理论保证 | 单调改进（在下界条件下） | 无形式化保证 |
| GPU/分布式友好性 | 差（CG 需要同步） | 优秀 |
| 超参数敏感性 | $\delta$ 需仔细调整 | $\varepsilon = 0.2$ 几乎总是有效 |

PPO 在实践中的优势源于*多种因素的结合*，而不仅仅是裁剪：

- **每批数据多轮更新**能从相同轨迹中榨取更多学习信号。
- **熵奖励**在无需专门噪声调度的情况下维持探索。
- **每步无反向传播开销**——每个 epoch 只需一次 Adam 调用。
- **极易并行化**——你可以并行运行 64 个 actor 收集 rollout。

![PPO 在远低于 TRPO 的成本下，效果持平甚至更好](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig4_benchmark.png)

MuJoCo 曲线（左）和 Atari 条形图（右）展示了原始 PPO 论文发表并由 Engstrom 等人（2020）复现的经验结果：PPO-Clip 在所有任务上全面超越 A2C，并在大多数任务上小幅领先 TRPO，同时每墙钟秒的运行速度大约快 10 倍。

---

## PPO 在 RLHF 中的角色

PPO 最具影响力的应用是 **基于人类反馈的强化学习（RLHF）**——正是该算法将 GPT-3 转变为 ChatGPT。

### 三阶段流程

1. **监督微调（SFT）**。在高质量人类示范数据上微调预训练 LLM。输出：参考模型 $\pi_{\text{ref}}$。
2. **奖励建模**。收集人类偏好数据（如“回答 A 优于回答 B”），通过 Bradley-Terry 成对损失训练奖励模型 $R_\phi(x, y)$。
3. **PPO 微调**。将 LLM 视为策略，提示 $x$ 为状态，响应 $y$ 为动作，$R_\phi(x, y)$ 为奖励。运行 PPO。

### RLHF 目标函数
$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\,y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x, y) - \beta\, D_{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)\right]$$
这里有两个信任域在起作用：

- **PPO 裁剪**防止每次更新时策略崩溃，作用与经典 RL 完全相同。
- **参考模型的 KL 惩罚项** $\beta D_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ 防止策略*长期漂移*出 SFT 模型。没有它，PPO 会找到**奖励黑客**——即在奖励模型下得分很高，但内容却是胡言乱语、充满敌意或阿谀奉承的响应。KL 惩罚就是保持生成内容“类人”的对齐税。

### 为何 RLHF 比 CartPole 更难

- 每个“episode”就是一次生成；单样本计算开销巨大（70B 模型的前向+反向传播）。
- 奖励模型本身也是学习得到的，且带有噪声。通常需要较小的裁剪范围和较大的 KL 系数（$\beta \in [0.01, 0.2]$）。
- 动作空间是整个词表（约 5 万 token），跨越数百个时间步。优势估计器必须方差极低——这就是为何 PPO 结合 GAE 和大批量优于朴素策略梯度。

### DPO、IPO 和 KTO 的定位

大规模运行 PPO 的复杂性（四个模型副本：策略、参考、奖励、价值；多 GPU 同步）催生了一波*直接偏好学习*方法：

- **DPO**（Rafailov 等，2023）将最优 RLHF 策略重参数化为闭式解，并通过*监督*对比损失训练——无需奖励模型，也无需 rollout。
- **IPO** 修复了 DPO 在高置信度标注对上的过拟合倾向。
- **KTO** 使用单响应符号反馈（“好”/“坏”）而非成对比较。

直接方法的优势在于：基础设施更简单，方差更低。而 PPO 仍占优的场景是：当你需要真实的*奖励信号*时（在线学习、使用工具的智能体、代码/数学环境中由验证器提供奖励），PPO 仍是唯一的通用解决方案。

---

## 实用调优指南

### 超参数速查表

| 参数 | 典型范围 | 备注 |
|------|---------|------|
| 学习率 | $1\text{e-}4$ 到 $3\text{e-}4$ | 有时在训练过程中线性退火至 0 |
| 裁剪 $\varepsilon$ | 0.1 — 0.3 | 0.2 在 90% 的情况下有效 |
| GAE $\lambda$ | 0.9 — 0.99 | 默认 0.95；当价值函数不可靠时更接近 1 |
| 折扣 $\gamma$ | 0.99 — 0.999 | 长视野任务需要更大的 $\gamma$ |
| Rollout 批量 | 2048 — 8192（单环境：更长） | 越大 = 优势方差越低 |
| PPO epochs | 3 — 10 | 超过 15 通常会过拟合 |
| Mini-batch 大小 | 64 — 256 | 与 rollout 批量无关 |
| 熵系数 | 0.0 — 0.01 | Atari 需要更高（~0.01）；MuJoCo 常为 0 |
| 价值损失系数 | 0.5 — 1.0 | 当 critic 难学时调高 |
| 梯度裁剪范数 | 0.5 — 1.0 | 异常 batch 的“安全带” |

![超参敏感度：裁剪范围、学习率、epoch×batch 网格](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig7_hyperparameter.png)

敏感性图展示了三种经典模式。裁剪 $\varepsilon$ 具有**宽而平坦的最优区间**——你真的不必调整它。学习率的最优区间**更窄**——3 倍的偏差会导致 5–10% 的回报损失。epochs × batch 的热力图揭示了交互效应：小批量上过多 epochs 会过拟合当前数据，损害策略的 KL 预算。

### 调试清单

- **新旧策略间的近似 KL**：每次更新应徘徊在 0.01–0.02。> 0.05 意味着裁剪未能有效约束——降低学习率或缩小 mini-batch。（注意常见的“单样本” KL 估计器 $\frac{1}{2}(\log r)^2$；推荐使用无偏的 Schulman 估计器 $r - 1 - \log r$。）
- **裁剪比例**：触及裁剪边界的样本占比。10–30% 是健康的。0% 意味着你未利用信任域；> 50% 意味着你在边界上训练。
- **熵**：应平滑衰减。若在几十步内骤降至近零，即为*过早收敛*——提高熵系数。
- **价值函数的解释方差**：$1 - \mathrm{Var}(R - V)/\mathrm{Var}(R)$。应在几百次更新内升至 0.5 以上；若卡在 0 附近，说明你的 critic 有问题。
- **优势归一化**：应在每个*minibatch* 内进行，而非每个 epoch。忘记 per-batch 归一化是 PPO 实现中最常见的 bug。

### 常见故障模式

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| 奖励先升后崩 | KL 过激进，裁剪无效 | 降低学习率；减少每批 epochs |
| 奖励始终不动 | 优势在 batch 内归一化为 0 | 检查奖励是否恒定；扩大探索 |
| 50 步内熵塌陷至 0 | 无熵奖励；从开始就贪婪 | 添加 `c_ent` $\ge 0.005$，提高温度 |
| Critic 预测爆炸 | 无奖励归一化；奖励无界 | 裁剪或归一化奖励（滑动统计） |
| CartPole 可行，连续控制失败 | 高斯策略使用了离散风格网络 | 使用 `tanh` 压缩，学习可调 log-std |

---

## 总结

信任域方法将策略梯度从其最尴尬的失败模式中拯救了出来——一次糟糕的更新就能抹去一小时的训练成果。两大思想推动了该领域的发展：

- **TRPO** 将 Kakade-Langford 的策略改进下界转化为算法：优化代理目标，硬约束 KL，通过自然梯度求解。理论优美，但工程实现沉重。
- **PPO** 用裁剪后的代理目标加一阶优化，替代了硬约束。它放弃了形式化的单调改进保证，却换来了实践中真正重要的东西：代码简单、支持多 epoch 更新、可并行 rollout，以及稳健的默认参数。

更深层的工程启示具有普适性：**一个简单且局部诚实的近似，往往胜过复杂但全局正确的方案**——尤其是当“正确”方法的单步代价高出两个数量级时。PPO 对 TRPO 的胜利，正如 Adam 在深度学习中对二阶优化器的胜利。

如今，PPO 的影响早已超越经典 RL。过去三年训练的每一款主流对齐 LLM——ChatGPT、Claude、Gemini、Llama-2 chat 模型——都在其后训练流程中运行了 PPO 循环。毫不夸张地说，**正是这个裁剪机制，让现代 AI 助手不至于在奖励模型面前失控**。

---

## 参考文献

- Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). Trust Region Policy Optimization. *ICML*. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Kakade, S., & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. *ICML*.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. *ICLR*. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Huang, S., Dossa, R., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. (2022). The 37 Implementation Details of PPO. *ICLR Blog Post*. [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
