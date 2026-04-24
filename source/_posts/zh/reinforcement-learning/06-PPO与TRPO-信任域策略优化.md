---
title: "强化学习（六）：PPO与TRPO：信任域策略优化"
date: 2024-07-06 09:00:00
tags:
  - 强化学习
  - PPO
  - TRPO
  - 信任域
categories: 强化学习
series:
  name: "强化学习"
  part: 6
  total: 12
lang: zh-CN
mathjax: true
description: "PPO和TRPO的完整推导：从策略优化的不稳定性到信任域约束，PPO的裁剪技巧，以及PPO在RLHF中的关键角色。"
disableNunjucks: true
---
策略梯度（参见第三篇）直接对策略本身求导，绕开了离散 `argmax`，能自然处理连续动作和随机策略。但它有一个致命缺陷——**走错一步就可能毁掉整个策略**，而且因为采样分布跟着策略一起变，回头几乎不可能。

**信任域方法**把这件事说得很直白：每次更新限制的是"行为"的变化幅度，而不是"参数"的变化幅度。TRPO 通过硬性 KL 约束加二阶优化器实现；PPO 用一行带 clip 的算式近似出同样的效果。**廉价的近似版赢了**——PPO 训练了 OpenAI Five，撑起了 ChatGPT 的 RLHF 阶段，几乎所有现代机器人策略都在用它，至今仍是工业 RL 的"默认算法"。

## 你将学到什么

- 朴素策略梯度为什么会**灾难性地不稳定**——附一个具体例子
- **重要性采样**：让我们能复用旧策略数据的桥梁
- **TRPO**：单调改进的下界证明、自然梯度、共轭梯度法、回溯线搜索
- **PPO-Clip** 与 **PPO-Penalty**：用一阶优化近似信任域的两条路
- **PPO 在 RLHF 中的角色**：怎么对齐 ChatGPT 级别的模型，DPO/IPO/KTO 又卡在哪里
- 一份能在 `cleanrl` 或 Stable Baselines 3 里直接用的调参与排错指南

**前置知识**：[第三篇](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/)（REINFORCE、Actor-Critic、优势函数）。会一些 KL 散度和 Fisher 信息更好，但不是必需，文中会按需展开。

---

## 策略更新为什么会不稳定

### 一个病态的小例子

设当前策略在状态 $s$ 下，$\pi(a_1|s) = 0.9$，$\pi(a_2|s) = 0.1$。运气不好，我们采样到 $a_2$，环境恰好返回了奖励 $+100$。REINFORCE 估计器会把 $\nabla_\theta \log \pi(a_2|s)$ 乘上 $+100$ 再做一步梯度上升。一次更新之后，$a_2$ 的概率可能从 $0.1$ 蹦到 $0.7$——尽管 $a_2$ 的真实期望回报很可能远不如 $a_1$。

背后藏着三个独立的问题：

1. **方差爆炸**：score 函数里有 $1/\pi$ 这一项，越罕见的动作梯度越大——这正是离策略 REINFORCE 不稳定的根源。
2. **分布漂移**：下一批轨迹是用更新后的*新策略*采的。如果新策略变差了，从它身上采到的每一条数据都在强化这个糟糕的信号——一个负反馈循环。
3. **不可逆性**：监督学习一步走错只是损失变差；RL 走错一步会丢掉整个数据分布，事后纠正所需的样本量往往是出错代价的一个数量级。

### 参数空间会骗人，策略空间不会

考虑两个高斯策略 $\pi_1 = \mathcal{N}(0, 0.01)$ 和 $\pi_2 = \mathcal{N}(0, 10)$。它们的参数（均值与对数标准差）在欧氏距离意义下非常接近，但行为完全不同：$\pi_1$ 几乎确定性地输出 0，而 $\pi_2$ 把动作几乎均匀地撒在整个值域上。

教训很清楚：**参数空间里走一固定步长，可能在策略行为上引发任意大的变化**。任何"安全"保证都只能建立在*分布空间*里，而 KL 散度 $D_{KL}(\pi_{\text{old}} \| \pi_\theta)$ 是天然的度量。

![信任域：许多个 KL 受限的小步保持安全；一次大的参数步会跌下悬崖](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig1_trust_region.png)

图把这层几何讲得很直白：绿到红的曲面是一个假想的 $J(\theta)$ 地形，山脊狭窄、悬崖陡峭。朴素策略梯度（左图）沿梯度方向迈了一大步，正好踩到悬崖上；TRPO（右图）每一步都被约束在当前策略的 KL 球内，轨迹紧贴山脊，最终稳稳收敛到附近的最优解。

---

## 重要性采样：通往离策略数据的桥

朴素的在策略方法每批数据只用一次梯度就丢——分布变了。**重要性采样**让我们能"加权复用"旧数据：

$$\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\!\left[\tfrac{q(x)}{p(x)}\, f(x)\right]$$

把旧策略当 $p$、新策略当 $q$ 代回策略梯度目标，就得到**代理目标**：

$$L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right]$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ 是**概率比**，本文之后所有算法的核心都围绕它展开。

两条要记住的事实：

- 在 $\theta = \theta_{\text{old}}$ 处，$L^{\text{IS}}$ 与真实策略梯度目标 $J(\theta)$ 的**值和梯度都相等**——所以它在局部是可信的。
- 离 $\theta_{\text{old}}$ 越远，IS 估计器的方差大致按 $\exp(2\,D_{KL})$ 增长（见下图）。KL 极小，复用极可信；KL 一大，估计就成了噪声。

![重要性采样比的分布以及方差随 KL 的增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig5_importance_sampling.png)

左图按不同的策略距离画出 $r_t$ 的分布：当 $D_{KL}\!\le\!0.02$ 时，分布紧紧落在 PPO 的绿色裁剪区间 $[0.8, 1.2]$ 内；一旦 $D_{KL}\!=\!0.05$，右尾就会显著拉长。右图把 IS 方差画成 KL 的函数（对数纵轴），橙色阴影正是被裁剪"省下来"的方差。这是把每次更新约束在小信任域里的定量理由。

---

## TRPO：信任域策略优化

### 单调改进下界

Schulman 等人（2015）在 Kakade & Langford（2002）的基础上证明：新策略的真实回报可以用代理目标加 KL 罚项来下界：

$$J(\pi_{\text{new}}) \;\geq\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) \;-\; C \cdot D_{KL}^{\max}\!\left(\pi_{\text{old}} \,\|\, \pi_{\text{new}}\right)$$

其中 $C = 4\varepsilon\gamma/(1-\gamma)^2$，$\varepsilon$ 是优势函数的最大幅度，$\gamma$ 是折扣因子。推论很惊艳：**只要在保持 $D_{KL}^{\max}$ 不大的前提下提升代理目标，就能保证策略单调不退**。

实际中常数 $C$ 太悲观，没法直接用；TRPO 干脆把"罚项"换成"硬约束"，由经验决定大小。

### 受约束的优化问题

$$\max_\theta \;\; \mathbb{E}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right] \quad\text{s.t.}\quad \bar{D}_{KL}\!\left(\pi_{\text{old}} \,\|\, \pi_\theta\right) \leq \delta$$

通常 $\delta \approx 0.01$。这里用的是**平均 KL**而不是最大 KL，原因很现实：平均 KL 用样本估计起来便宜得多。

### 自然梯度：定义合适的"小"

普通 SGD 在欧氏球 $\|\Delta\theta\|^2 \le c$ 里挑使线性化目标最大的方向；**自然梯度**换了度量，把这个球换成"KL 球"。在 $\theta$ 附近，KL 散度二阶展开正是一个二次型，其 Hessian 就是**Fisher 信息矩阵**：

$$D_{KL}(\pi_\theta \,\|\, \pi_{\theta+\Delta\theta}) \;\approx\; \tfrac{1}{2}\,\Delta\theta^\top F\,\Delta\theta, \qquad F = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta\,\nabla_\theta \log \pi_\theta^\top\right]$$

求解这个受约束的优化得到自然梯度更新 $\Delta\theta \propto F^{-1}\nabla J$。它沿的是**策略空间**里的最陡上升方向，而不是参数空间里的——正是我们想要的。

### 实现：共轭梯度 + 线搜索

百万级参数下，$F$ 有 $\sim 10^{12}$ 项，根本存不下。TRPO 用两个技巧绕开：

1. **共轭梯度**求解 $Fx = g$，全程只需要 Fisher–向量乘积 $Fv$（用两次 `autograd` 反向就能算出来），永远不显式构造 $F$。
2. **回溯线搜索**：沿自然梯度方向逐步缩小步长，直到 KL 约束满足且代理目标确实改善——这一步把"二阶近似不准"的风险也兜住了，是单调性保证能在实际中近似成立的关键。

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

**优点**：理论上单调改进；在 Humanoid、Ant 这类朴素 PG 容易发散的任务上非常稳定。

**缺点**：核心代码 $\sim$300 行，每次更新要算 10–20 次 Hessian-向量乘积；多 epoch 复用一批数据收益很小（CG 已经把曲率信息榨干）；分布式训练不友好（CG 每步都要同步）。

---

## PPO：花两成代价拿走九成收益

2017 年 Schulman 团队提了一个问题：*能不能只用一阶优化器，得到接近 TRPO 的稳定性？* 答案就是 PPO，从此它统治了这个领域。

### PPO-Clip：核心套路

定义裁剪后的代理目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}\!\left(r_t(\theta),\,1\!-\!\varepsilon,\,1\!+\!\varepsilon\right)\hat{A}_t\Big)\right]$$

通常 $\varepsilon \approx 0.2$。这个设计有意造成不对称：`min` 会让目标变得**悲观**——总是取裁剪与未裁剪两者中较小的那个。

![按优势符号拆分的 PPO 裁剪代理目标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig2_ppo_clipping.png)

两种情况，每种一句话总结：

- **$\hat{A}>0$（好动作）**：未裁剪版的目标随 $r_t \to \infty$ 一直增大；裁剪版在 $r_t > 1+\varepsilon$ 时把奖励"封顶"。`min` 会取这个封顶值，于是越过封顶后梯度归零。一句话：*别因为一个 minibatch 的好运气，就把这个动作的概率推到 $(1+\varepsilon)\pi_{\text{old}}$ 以上。*
- **$\hat{A}<0$（坏动作）**：对称——裁剪版限制了 $r_t < 1-\varepsilon$ 时损失能负到多深，所以新策略一旦把概率压低到一定程度，梯度就停了。*别因为一次倒霉就把某个动作彻底压死。*

更深一层的问题是：*为什么用 `min` 而不是 `max`？* 如果总是取较大的那个，优化器会奖励大比率，反而会把信任域踩破。`min` 把信任域的精神写进了目标里，全程不需要算 KL。

### PPO-Penalty：自适应版本

第二种变体不那么常见，但在某些场景里好用：在目标里直接加 KL 罚项，并自适应调整系数：

$$L^{\text{KL}}(\theta) = \mathbb{E}\!\left[r_t(\theta)\,\hat{A}_t\right] \;-\; \beta\,\mathbb{E}\!\left[D_{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)\right]$$

每次更新后调整 $\beta$：观测到的 KL 高于 $1.5\,\delta_{\text{target}}$ 时翻倍，低于 $\delta_{\text{target}}/1.5$ 时减半。

![自适应 KL 罚项：目标形状与 beta 的调度过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig3_kl_penalty.png)

左图展示了 $\beta$ 如何塑造目标：$\beta=0$ 退化为无约束代理目标（连同它的不稳定）；$\beta$ 很大时最优解被"拉"回 $\theta_{\text{old}}$。右图是真实的自适应过程——为了把观测 KL 维持在目标值附近，$\beta$ 会跨数量级变动（注意纵轴是对数刻度）。早期的 InstructGPT 内部用的就是 KL 自适应版，后来在 `trl` 等 RLHF 库中被 clip 版取代。

### 代理目标地形：为什么裁剪在实践里更稳

![代理目标 vs 真实目标：未裁剪会误导优化器，裁剪能保持诚实](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig6_surrogate_landscape.png)

这张图就是直接的视觉总结。黑线是沿一维切片的*真实*回报 $J(\theta)$，先有一个峰，紧接着是一个掉进低谷的悬崖。橙色虚线是未裁剪的 IS 代理目标，它一路上升，会把 SGD 引到悬崖里。蓝线是 PPO 裁剪后的目标：信任域内（绿色带）紧贴真实曲线；信任域外它被"压平"了，恰好擦掉了那段会把我们带下悬崖的梯度。

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
        """广义优势估计 GAE（Schulman 等，2016）。

        lam=0  -> 单步 TD（方差小，偏差大）
        lam=1  -> 蒙特卡洛回报（无偏，方差大）
        lam=0.95 是几乎所有任务通用的默认值。
        """
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

在普通 actor-critic 之外只多了几行关键代码，就能跑出一个百回合解决 CartPole、千万步学会 Humanoid 行走的算法。

---

## TRPO 与 PPO 正面对比

| 维度 | TRPO | PPO-Clip |
|------|------|----------|
| 优化器 | 共轭梯度 + 线搜索（二阶） | Adam（一阶） |
| KL 约束 | 硬约束，严格 $\le \delta$ | 软约束，逐样本 clip |
| 每批数据更新次数 | 1 | 3–10 epochs |
| Hessian-向量乘积 | $\sim$10–20 | 0 |
| 核心代码量 | $\sim$300 行 | $\sim$100 行 |
| 理论保证 | 单调改进（在下界范围内） | 形式上没有 |
| GPU / 分布式友好度 | 差（CG 每步要同步） | 优秀 |
| 超参敏感性 | $\delta$ 必须调 | $\varepsilon = 0.2$ 几乎万用 |

PPO 的实战优势从来不只来自 clip 这一招，而是**多个因素叠加**：

- **每批多 epoch 更新**——同一批轨迹榨出更多学习信号。
- **熵奖励**让探索保持活跃，不需要专门的噪声调度。
- **每步无需反向曲率信息**——一次 Adam 调用搞定一个 epoch。
- **天然可并行**——可以同时跑 64 个 actor 收 rollout。

![PPO 在远低于 TRPO 的成本下，效果持平甚至更好](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig4_benchmark.png)

左边是 MuJoCo 训练曲线，右边是 Atari 上的人类归一化得分——这正是原始 PPO 论文中给出、后来又被 Engstrom 等（2020）系统复现的实证图景：PPO-Clip 全面压制 A2C，在大多数任务上略胜 TRPO，**而且每秒钟挂钟时间能多跑约 10 倍的步数**。

---

## PPO 在 RLHF 中的角色

PPO 最具历史意义的应用是 **RLHF（基于人类反馈的强化学习）**——把 GPT-3 调教成 ChatGPT 的那个算法。

### 三阶段流程

1. **监督微调（SFT）**：用高质量人类示范微调预训练 LLM，得到参考模型 $\pi_{\text{ref}}$。
2. **奖励模型训练**：收集"回答 A 比 B 好"形式的偏好数据，用 Bradley-Terry 成对损失训练奖励模型 $R_\phi(x, y)$。
3. **PPO 微调**：把 LLM 视作策略，prompt $x$ 视作状态，回答 $y$ 视作动作，$R_\phi(x, y)$ 视作奖励，跑 PPO。

### RLHF 的目标函数

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\,y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x, y) - \beta\, D_{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)\right]$$

里面其实有**两层信任域**：

- **PPO 的 clip** 阻止单步崩溃，跟经典 RL 里完全一样。
- **对参考模型的 KL 罚项** $\beta D_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ 阻止*长期漂移*。一旦没有它，PPO 就会找到**奖励模型的漏洞**——回答在奖励模型下分很高，但其实是胡言乱语、攻击性内容或舔狗式回应。这一项是"对齐税"，让生成的内容仍然像人写的。

### 为什么 RLHF 比 CartPole 难

- 一个"episode"就是一次生成；单条样本的算力开销巨大（70B 模型的前向 + 反向）。
- 奖励模型本身也是学出来的，含噪。所以裁剪范围常取小一点，KL 系数取得相对大（$\beta \in [0.01, 0.2]$）。
- 动作空间是整个词表（约 5 万 token），序列长达数百步。优势估计必须方差极低——这就是为什么大家都用 GAE + 大批量 PPO，而非朴素策略梯度。

### DPO、IPO、KTO 在哪一格

把 PPO 拉到大模型规模上做工程实在太麻烦——四份模型副本（policy、ref、reward、value）、多卡同步——于是冒出一波**直接偏好学习**方法：

- **DPO**（Rafailov 等，2023）将"最优 RLHF 策略"重参数化为闭式，再用一个*监督式*对比损失训练——不需要奖励模型，也不需要做 rollout。
- **IPO** 修补了 DPO 在高置信度偏好对上的过拟合倾向。
- **KTO** 用单回答的有符号反馈（"好"/"坏"），不再需要成对比较。

直接方法的优势：基础设施简单、方差低。PPO 仍胜出的场景：**当你需要真正的奖励信号时**（在线学习、调用工具的智能体、有可验证奖励的代码/数学环境），PPO 仍然是唯一的通用解。

---

## 调参与排错指南

### 超参速查表

| 参数 | 常见范围 | 备注 |
|------|---------|------|
| 学习率 | $1\text{e-}4$ 到 $3\text{e-}4$ | 训练后期常线性退火到 0 |
| 裁剪 $\varepsilon$ | 0.1 – 0.3 | 0.2 在 9 成场景里直接能用 |
| GAE $\lambda$ | 0.9 – 0.99 | 默认 0.95；价值网络不可靠时往 1 调 |
| 折扣 $\gamma$ | 0.99 – 0.999 | 长时间任务取更接近 1 的值 |
| Rollout 批量 | 2048 – 8192（单环境时拉长） | 越大优势估计方差越小 |
| PPO epochs | 3 – 10 | 超过 15 几乎必然过拟合 |
| Mini-batch | 64 – 256 | 与 rollout 批量解耦 |
| 熵系数 | 0.0 – 0.01 | Atari 需要更大（约 0.01）；MuJoCo 常取 0 |
| 价值损失系数 | 0.5 – 1.0 | critic 难学时调高 |
| 梯度裁剪范数 | 0.5 – 1.0 | 防异常 batch 的"安全带" |

![超参敏感度：裁剪范围、学习率、epoch×batch 网格](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO%E4%B8%8ETRPO-%E4%BF%A1%E4%BB%BB%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96/fig7_hyperparameter.png)

这张图概括了三种典型模式。裁剪 $\varepsilon$ 的最优区间**又宽又平**——基本不必调；学习率的最优区间**窄得多**——偏离 3 倍就会损失 5%–10% 的回报；epoch × batch 的热力图揭示了一个交互效应——小批量上多 epoch 很容易过拟合当前数据，把 KL 预算挥霍掉。

### 排错清单

- **新旧策略的近似 KL**：每次更新应在 0.01–0.02 之间。> 0.05 说明 clip 没拉住，应当降学习率或缩小 mini-batch。（注意常见的"单样本"估计 $\frac{1}{2}(\log r)^2$ 有偏；推荐用 Schulman 的无偏估计 $r - 1 - \log r$。）
- **clip 触发率**：被裁剪到边界的样本比例。10%–30% 是健康的；0% 说明根本没在用信任域；> 50% 说明你一直在边界上训练。
- **熵**：应平滑下降。如果几十步内骤降到 0，那是*过早收敛*，调高熵系数。
- **价值函数解释方差**：$1 - \mathrm{Var}(R - V)/\mathrm{Var}(R)$，几百次更新内应升过 0.5；如果一直贴近 0，critic 多半坏了。
- **优势归一化**：要在每个*minibatch*里做，不是每个 epoch。漏掉这一步是 PPO 实现里最常见的 bug。

### 常见故障

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| 奖励先涨后崩 | KL 步子过大，clip 没拉住 | 降学习率；减少每批 epoch 数 |
| 奖励一直不动 | 优势在 batch 内归一化为 0 | 检查奖励是否常数；扩大探索 |
| 50 步内熵塌成 0 | 没加熵奖励，从一开始就贪心 | 加 `c_ent` $\ge 0.005$，提高温度 |
| critic 预测爆炸 | 没归一化奖励；奖励无界 | 用滑动统计裁剪/归一化奖励 |
| CartPole 能跑，连续控制不行 | Gaussian 策略却用了离散网络结构 | 加 tanh 压缩、可学的 log-std |

---

## 小结

信任域方法把策略梯度最尴尬的失败模式——一步走错，半小时白训——彻底解决了。两条核心思想撑起整个领域：

- **TRPO** 把 Kakade-Langford 的策略改进下界落地为算法：优化代理目标、硬约束 KL、用自然梯度求解。理论漂亮，工程沉重。
- **PPO** 把硬约束换成裁剪后的代理目标，再加一阶优化器——丢掉了形式上的单调改进保证，但拿回了实际中真正重要的一切：代码简洁、多 epoch 复用数据、可并行 rollout、默认值开箱即用。

更深的工程教训是普适的：**一个简单、局部诚实的近似，往往胜过复杂、全局正确的方案**——尤其当"正确"方案的单步代价高出两个数量级。PPO 战胜 TRPO，与 Adam 在深度学习里战胜二阶优化器是同一种胜利。

PPO 的影响力早已溢出经典 RL 的范畴。过去三年里训练出的所有主流对齐 LLM——ChatGPT、Claude、Gemini、Llama-2 chat——都在后训练阶段跑过 PPO。某种意义上，**正是这个 clip，让现代 AI 助手不至于在自己的奖励模型面前彻底失控。**

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

---

## 系列导航

| 部分 | 主题 |
|------|------|
| 1 | [基础与核心概念](/zh/强化学习-一-基础与核心概念/) |
| 2 | [Q-Learning与深度Q网络](/zh/强化学习-二-Q-Learning与深度Q网络/) |
| 3 | [Policy Gradient与Actor-Critic](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/) |
| 4 | [探索策略与好奇心驱动学习](/zh/强化学习-四-探索策略与好奇心驱动学习/) |
| 5 | [Model-Based 强化学习与世界模型](/zh/强化学习-五-Model-Based强化学习与世界模型/) |
| **6** | **PPO 与 TRPO（本文）** |
