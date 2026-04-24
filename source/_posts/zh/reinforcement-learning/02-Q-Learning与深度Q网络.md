---
title: "强化学习（二）：Q-Learning 与深度 Q 网络（DQN）"
date: 2025-06-19 09:00:00
tags:
  - 强化学习
  - Q-Learning
  - DQN
  - 经验回放
  - 目标网络
  - Rainbow
  - 价值方法
categories: 强化学习
series:
  name: "强化学习"
  part: 2
  total: 12
lang: zh-CN
mathjax: true
description: "DQN 如何结合神经网络与 Q-Learning 玩转 Atari——经验回放、目标网络、Double DQN、Dueling DQN、优先经验回放与 Rainbow。"
disableNunjucks: true
series_order: 2
---

2013 年 12 月，DeepMind 一支不大的团队在 arXiv 上挂出了一篇短论文，里面有一个相当扎眼的结论：同一个神经网络，仅仅以原始像素和分数为输入，就学会了七款 Atari 游戏，并在其中六款上刷新了当时的最好成绩。没有针对游戏专门设计的特征，没有手写的启发式规则，Pong、Breakout、Space Invaders 共用同一套架构。这套算法叫**深度 Q 网络（Deep Q-Network, DQN）**，它正式拉开了深度强化学习时代的序幕。

DQN 并不是凭空造出来的。它就是 Watkins 在 1989 年提出的**表格 Q-Learning**，把查表换成了神经网络，再加上两个让训练不至于崩掉的工程技巧。本文要讲清楚的是：这两个技巧到底解决了哪两个具体的问题，PyTorch 里完整怎么写，以及 DQN 之后那些把它从 Atari 演示推上工业舞台的变体。

## 你将学到什么

- 为什么表格型 Q-Learning 在高维状态空间里会彻底失效
- **致命三角（Deadly Triad）**——让朴素深度 RL 发散的三个要素
- DQN 的两大创新：**经验回放**与**目标网络**——分别针对三角的哪一条边
- 一份可直接运行的 Atari **DQN** PyTorch 实现
- DQN 之后的家族：**Double DQN**、**Dueling DQN**、**优先经验回放**与 **Rainbow**

**前置知识**：[第 1 部分](/zh/强化学习-一-基础与核心概念/) 中的 MDP、Bellman 方程与时序差分（TD）思想。

---

## Q-Learning 基础

### 重新读一遍 Bellman 最优方程

回顾第 1 部分：对于最优策略 $\pi^*$，动作价值函数满足 Bellman 最优方程：

$$
Q^*(s, a) \;=\; \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\Big]
$$

可以把它当成一份"契约"来读：在状态 $s$ 下采取动作 $a$ 的价值，等于即时奖励，加上折扣后所能取得的最好未来。一旦得到 $Q^*$，最优策略就退化成一次查表：$\pi^*(s) = \arg\max_a Q^*(s, a)$，不需要规划，也不需要搜索。

于是整个问题被压缩成一句话：**估计 $Q^*$**。Q-Learning 是众多解法之一。

### Q-Learning 的更新规则

智能体在状态 $S_t$ 下执行动作 $A_t$，观察到奖励 $r_t$ 与下一状态 $S_{t+1}$ 后，Q-Learning（Watkins, 1989）只更新当前访问的那一格：

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha \underbrace{\Big[r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big]}_{\text{TD 误差 } \delta_t}
$$

方括号里的量是 **TD 误差**——它衡量的是"一步自举出来的目标"与"当前估值"之间的差距。$\delta_t > 0$ 说明现在低估了，应该向上调；反之则向下调。

Q-Learning 有两条值得记住的性质：

1. **离策略（off-policy）**。目标里用的是 $\max_{a'}$，也就是贪心动作，与智能体下一步*实际*选什么毫无关系。行为策略（通常是 $\varepsilon$-greedy）可以放心探索，目标策略始终保持贪心。这种解耦既是 Q-Learning 的杀手锏，也是它后来不稳定的根源。
2. **收敛性保证**。Watkins 与 Dayan（1992）证明：如果每个状态-动作对都被访问无穷多次，且学习率满足 Robbins-Monro 条件（$\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$），那么 $Q(s, a)$ 以概率 1 收敛到 $Q^*(s, a)$。

训出来的 Q 表本身就是一个相当直观的对象。下图展示了一张 4x4 网格世界的 Q 表，目标格 +1，陷阱 -1，每走一步罚 -0.04。每一格里有四个数字（对应四个动作），箭头指向 $\arg\max_a Q(s, a)$ 给出的贪心选择。

![4x4 网格世界上的 Q 表——每格四个 Q 值与一个贪心箭头](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig1_qtable_gridworld.png)

### 走一遍 Cliff Walking

Sutton 与 Barto 教科书里的 Cliff Walking（悬崖行走）几乎是 Q-Learning 的标配实验。智能体在一张 4x12 的网格上行走，起点和终点之间的整排底格是悬崖：踩进去罚 -100，然后被弹回起点。其它每走一步罚 -1，所以最优路线就是**贴着悬崖上沿走一格**，回报是 -13。

```python
import numpy as np


class CliffWalkingEnv:
    """4x12 网格。起点 (3,0)，终点 (3,11)，悬崖 (3,1)-(3,10)。"""

    def __init__(self):
        self.height, self.width = 4, 12
        self.start, self.goal = (3, 0), (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0:   r = max(0, r - 1)       # 上
        elif action == 1: c = min(11, c + 1)      # 右
        elif action == 2: r = min(3, r + 1)       # 下
        elif action == 3: c = max(0, c - 1)       # 左

        ns = (r, c)
        if ns in self.cliff:
            self.state = self.start
            return self.start, -100, False
        if ns == self.goal:
            self.state = ns
            return ns, 0, True
        self.state = ns
        return ns, -1, False


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {(i, j): np.zeros(4) for i in range(4) for j in range(12)}
    history = []
    for _ in range(episodes):
        state, total, steps = env.reset(), 0, 0
        while steps < 1000:
            action = (np.random.randint(4) if np.random.rand() < epsilon
                      else int(np.argmax(Q[state])))
            ns, reward, done = env.step(action)
            total += reward
            # Q-Learning 更新——目标里使用的是"贪心"动作
            td_target = reward + gamma * np.max(Q[ns])
            Q[state][action] += alpha * (td_target - Q[state][action])
            state = ns
            steps += 1
            if done:
                break
        history.append(total)
    return Q, history
```

更有意思的是把 $\varepsilon$ 调成不同值会怎样：太贪心，根本找不到那条贴边的最优路；太爱探索，又会反复掉下悬崖。

![不同探索率下的 Cliff Walking 学习曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig2_cliff_walking_curves.png)

$\varepsilon = 0.10$ 那条曲线收敛得最干净；$\varepsilon = 0.50$ 卡在远低于最优的水平上，因为太多步是随机的（很容易致命）；$\varepsilon = 0.01$ 最终也能到达最优，只不过早期探索太稀疏，收敛慢一截。这种"探索 vs 利用"的权衡在整个 RL 里反复出现。

---

## 为什么表格不够用

Cliff Walking 才 48 个状态，一张 Q 表 192 个浮点数就够。换成 Atari 的 Breakout：DQN 论文把屏幕预处理成 84x84 灰度图，再叠最近四帧来给智能体一点"运动感"，状态变成 $\{0, \ldots, 255\}^{84 \times 84 \times 4}$ 中的一个向量，可能取值数量大约是 $256^{28224}$。宇宙里的原子都不够数。

由此立即得出两个失败：

1. **存不下**。任何表都装不下"每个状态对应一个 Q 值"。
2. **访问次数为零**。Q-Learning 收敛性证明要求每个状态-动作对被访问无穷多次。智能体根本不会两次看到*完全相同*的 84x84x4 数组。

解决方案是把"表"换成**带参数的函数近似器**：用 $Q(s, a; \theta)$ 这样一个参数化的函数（DQN 中是一个卷积网络）让模型在视觉上相似的状态之间**泛化**——只是噪声不同、砖块颜色不同的两帧屏幕，应该映射到接近的 Q 值。

### 致命三角

泛化是有代价的。Sutton 与 Barto 指出，在价值方法中，下面三个性质如果**同时**出现，算法就有可能发散，他们称之为**致命三角**：

1. **自举（bootstrapping）**：TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 把网络自己当下的估值当成"真值"。
2. **函数近似**：调整 $\theta$ 去修一个状态的 Q 值时，会同时把其它相邻状态的 Q 值也带偏——你没法像编辑表格一样只改一格。
3. **离策略数据**：用来更新的轨迹不是来自当前正在评估的策略。

任意两条单独都不致命：表格 Q-Learning 是"自举 + 离策略"，没有函数近似；线性近似 + 在策略也没问题。但三者凑齐之后，确实可以构造出 Q 值发散到无穷的玩具 MDP。

DQN 的两个创新分别针对其中"离策略"与"自举"两条边。

---

## DQN 的两大创新

### 经验回放：打破时间相关性

监督学习里，mini-batch SGD 的前提是 i.i.d. 假设——你把数据洗一下再采样。RL 里相邻的两步 $(s_t, a_t, r_t, s_{t+1})$ 与 $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ 高度相关：下一个状态本身就是上一个状态的函数。直接拿这串数据去训，梯度会在相关样本上来回震荡，旧的经验一离开屏幕马上被忘掉，收敛迟迟不发生。

DQN 的解法是**经验回放缓冲区（replay buffer）**：把每一条转移都放进一个很大的 FIFO 队列（一般 100 万条），每次梯度更新时从中随机抽 mini-batch。

![经验回放缓冲区：流式写入，随机抽取 mini-batch](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig4_replay_buffer.png)

它带来三个好处：

- **mini-batch 解相关**。32 条随机样本近似独立同分布，正好满足 SGD 的假设。
- **样本利用率高**。每条经验会被反复使用，DeepMind 报告的样本效率提升大约一个数量级。
- **平滑分布漂移**。缓冲区里混着许多近期策略产生的数据，即使策略变化很快，训练分布也变得平缓。

```python
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s_next), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
```

### 目标网络：别再追自己的尾巴

哪怕加了回放，TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 依然会随着 $\theta$ 的每次更新而变化——目标本身是优化变量的函数，损失曲面在每一步脚下都在变形。实测下来，这会带来振荡甚至发散。

DQN 的做法是同时维护**两份**网络：

- **在线网络** $Q(\cdot; \theta)$：每一步都用梯度下降更新。
- **目标网络** $Q(\cdot; \theta^-)$：在线网络的一份"冻结副本"，每隔一段时间被同步覆盖一次。

损失针对的是这份冻结副本：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big]
$$

每隔 $C$ 步（Nature 论文里是 10 000）执行一次同步：$\theta^- \leftarrow \theta$。在两次同步之间，目标是*常量*，每个 $C$ 步窗口都接近一个标准的有监督回归问题。

![目标网络：一份滞后的副本，把 TD 目标稳住](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig5_target_network.png)

也有一种连续的变体——**Polyak 软更新** $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$，$\tau$ 取 0.005 左右，DDPG 与 SAC 都用它，效果同样好。

---

## 完整的 Atari DQN

按 2026 年的标准来看，DQN 的架构并不大：3 个卷积层加 2 个全连接层，参数量约 170 万。卷积负责看 84x84 屏幕的局部纹理，全连接把展平后的特征图映射成"每个动作一个 Q 值"。

![DQN 架构：4 帧叠加输入 → 三层卷积 → FC 512 → 每个动作一个 Q 值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig3_dqn_architecture.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class DQN(nn.Module):
    """Nature-DQN 架构。输入 (batch, 4, 84, 84)，uint8 像素 / 255。"""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> (B, 32, 20, 20)
        x = F.relu(self.conv2(x))   # -> (B, 64, 9, 9)
        x = F.relu(self.conv3(x))   # -> (B, 64, 7, 7)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, n_actions: int, device: str = "cuda"):
        self.n_actions, self.device = n_actions, device

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=2.5e-4)
        self.memory = ReplayBuffer(capacity=100_000)

        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon, self.epsilon_end, self.epsilon_decay = 1.0, 0.01, 0.995
        self.target_update_freq = 10_000
        self.steps_done = 0

    def select_action(self, state, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.policy_net(x).argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s_next, d = self.memory.sample(self.batch_size)
        s        = torch.as_tensor(s,      dtype=torch.float32, device=self.device)
        a        = torch.as_tensor(a,      dtype=torch.long,    device=self.device)
        r        = torch.as_tensor(r,      dtype=torch.float32, device=self.device)
        s_next   = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        d        = torch.as_tensor(d,      dtype=torch.float32, device=self.device)

        # Q(s, a) 来自在线网络
        q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # 自举目标来自冻结的目标网络——这里不让梯度回传
        with torch.no_grad():
            q_next = self.target_net(s_next).max(dim=1).values
            q_target = r + (1.0 - d) * self.gamma * q_next

        # Huber loss 对偶尔出现的大 TD 误差比 MSE 更稳
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        return loss.item()
```

几条论文里看不到、但实战中很重要的细节：

- 用 **Huber loss**（`smooth_l1_loss`）而不是纯 MSE——配合梯度裁剪，偶尔出现的大 TD 误差不会把优化器搞崩。
- **梯度裁剪**到范数 10（或 1）能挡住罕见的大更新带来的连锁震荡。
- Atari 还要在环境侧做 **frame-skip 4 + 后两帧 max-pool**，处理画面闪烁；这是预处理而非智能体的一部分。
- 在 Pong 上，单 GPU 跑 200-300 个回合就能基本达到上限（平均回报 $\approx +21$）。

---

## DQN 变体：从 Double 到 Rainbow

DQN 最初的两版分别在 2013 年（workshop）和 2015 年（*Nature*）发表。之后几年，一系列论文每篇都精准修补一个具体的弱点；最后这些改进被整合成一个叫 **Rainbow** 的统一智能体，各组件几乎可加。

### Double DQN：消除最大化偏差

哪怕目标网络做得再好，$\max_{a'}$ 这个算子本身就引入一个系统性的"高估"。理由很短，也很优雅：假设真实 Q 值是 $Q^*(s, a)$，网络估值是 $Q^*(s, a) + \varepsilon_a$，噪声 $\varepsilon_a$ 期望为零，那么

$$
\mathbb{E}\Big[\max_a \big(Q^*(s, a) + \varepsilon_a\big)\Big] \;\geq\; \max_a Q^*(s, a)
$$

max 算子总是优先选中那个**正向噪声**最大的动作。通过自举，这种高估会沿着 TD 链路一路渗到所有状态。实测中，原版 DQN 的预测 Q 值会显著高于真实回报。

van Hasselt 等人（2016）提出 **Double DQN**：把动作的*选择*和*评估*拆给两个网络。在线网络选下一步动作，目标网络评估它的价值。

$$
y_t = r_t + \gamma\, Q\big(s_{t+1},\; \arg\max_{a'} Q(s_{t+1}, a'; \theta);\; \theta^-\big)
$$

两个网络的噪声部分独立，系统性偏差被基本抵消。

![Double DQN 紧贴真实 Q 值；原版 DQN 显著高估](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig6_double_vs_vanilla.png)

代码改动只有两行：

```python
with torch.no_grad():
    # 用"在线网络"选下一个动作
    next_actions = self.policy_net(s_next).argmax(dim=1, keepdim=True)
    # 用"目标网络"评估这个动作的价值
    q_next = self.target_net(s_next).gather(1, next_actions).squeeze(1)
    q_target = r + (1.0 - d) * self.gamma * q_next
```

### Dueling DQN：把"在哪里"和"做什么"拆开

很多 Atari 帧里，做哪个动作其实根本不重要——球还远着呢、敌人还没出现，几帧之内你怎么按都没区别。如果把状态价值和动作优势揉成一个 Q 头，就会浪费很多容量去反复学同一个 $V(s)$。

Wang 等人（2016）把输出头拆成两支：

$$
Q(s, a) \;=\; V(s) + \Big(A(s, a) - \tfrac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Big)
$$

- $V(s)$ —— "处于这个状态有多好？" 与动作无关。
- $A(s, a)$ —— "在这里选 $a$ 比平均水平好多少？" 与动作有关。

减去均值这一步是为了让分解**可识别**，不然 $V$ 与 $A$ 之间可以互相吸收任意常数。

```python
class DuelingDQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.value_fc, self.value_head = nn.Linear(64 * 7 * 7, 512), nn.Linear(512, 1)
        self.adv_fc,   self.adv_head   = nn.Linear(64 * 7 * 7, 512), nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        v = self.value_head(F.relu(self.value_fc(x)))      # (B, 1)
        a = self.adv_head(F.relu(self.adv_fc(x)))          # (B, |A|)
        return v + a - a.mean(dim=1, keepdim=True)
```

Dueling 与 Double DQN 配合得特别好——两者攻击的是不同的弱点，效果几乎可加。

### 优先经验回放（PER）

均匀采样把每一条转移都当成"同等重要"，但这显然不对：预测已经准的转移基本学不到东西，而 TD 误差大的转移正是信号最强的样本。

优先经验回放（Schaul 等人，2016）按照 TD 误差绝对值的某个幂次来采样：

$$
p_i \;\propto\; \big(|\delta_i| + \varepsilon\big)^\alpha
$$

指数 $\alpha \in [0, 1]$ 控制"优先程度"，$\alpha = 0$ 退化为均匀。非均匀采样会让梯度估计有偏，需要用重要性采样权重纠正：

$$
w_i = \Big(\tfrac{1}{N \cdot p_i}\Big)^\beta
$$

$\beta$ 在训练过程中从 0.4 退火到 1.0——前期偏差影响小，后期需要严格修正以保证稳定性。

### 多步回报、分布式 RL、NoisyNet

Rainbow 还集齐了三块拼图：

- **n 步回报**（Rainbow 用 $n=3$）：把单步自举换成局部 Monte-Carlo 目标，以方差换偏差：

  $$
  y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)
  $$

- **分布式 RL（C51）**，Bellemare 等人（2017）：不再只学回报的均值，而学整个回报分布。通过把分布投影到 51 个固定原子上得名 C51。完整分布带有的信息（尤其在随机环境中）远多于一个均值。

- **NoisyNet**：用注入到 FC 权重里的可学习噪声替代 $\varepsilon$-greedy 的探索。探索从此变成"状态相关、可学习"的，而不是手调的衰减表。

### Rainbow

Hessel 等人（2018）把六块（DQN 基础 + Double + Dueling + PER + n 步 + 分布式 + NoisyNet）拼到一起，并做了完整的 ablation。每个组件单独都能带来收益；合起来后的成绩明显高于任何单一变体。

![Atari-57 基准上的人类标准化中位分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning%E4%B8%8E%E6%B7%B1%E5%BA%A6Q%E7%BD%91%E7%BB%9C/fig7_atari_benchmark.png)

上图把得分按"人类 = 100%、随机智能体 = 0%"标准化。原版 DQN 中位分还低于人类；每加一项改进就往上推几十个百分点；Rainbow 在中位游戏上把人类水平翻了一倍多。

---

## 实战要点

### 一份起步用的超参数表

| 超参数                      | 常用取值                | 备注                                       |
| --------------------------- | ----------------------- | ------------------------------------------ |
| 回放缓冲区容量              | 100K - 1M 转移          | 越大越好，受限于内存                       |
| mini-batch 大小             | 32 - 128                | Atari 默认 32                              |
| 学习率                      | 1e-4 ~ 3e-4             | Adam，2.5e-4 是稳妥的起点                  |
| 折扣 $\gamma$               | 0.99                    | 短时距任务可以小一些                       |
| $\varepsilon$ 衰减          | 1.0 → 0.01              | 在 ~1M 帧上线性衰减                        |
| 目标网络同步周期 $C$        | 10K 步                  | 或软更新 $\tau \approx 0.005$              |
| Frame skip                  | 4                       | Atari 预处理                               |
| 梯度裁剪                    | 范数 10                 | 配合 Huber loss                            |

### 调试清单

1. **先在 CartPole 上跑通**。一个写对的 DQN，单 CPU 不到 200 个回合就能解决 CartPole。如果你的不行，问题大概率在代码里，不是超参数。
2. **盯着 Q 值的量级**。它应该上升然后稳定在合理范围（Atari 上一般是个位或两位数）。如果飘到 1000 以上，基本是发散。
3. **跟踪 TD 误差**。它应该下降并保持有界。一直单调上升说明目标在跑路。
4. **看动作分布**。如果只输出某一个动作，多半是网络坍缩了——检查 ReLU 是否大面积"死掉"、初始化是否出错、学习率是不是过大。

### DQN 还是别的算法？

| 维度          | DQN 家族                          | 策略梯度（PPO / SAC 等）             |
| ------------- | --------------------------------- | ------------------------------------ |
| 动作空间      | 仅离散                            | 离散 + 连续                          |
| 样本效率      | 高（回放可复用）                  | 较低（PPO 是在线策略）               |
| 稳定性        | 需要目标网络等一系列技巧          | 整体更易调                           |
| 最适场景      | Atari、棋类、离散控制             | 机器人、运动控制、连续控制           |

DQN 一个硬伤是**只能处理离散动作**：在五个按键上做 $\arg\max$ 很轻松，在一个实数关节角度上就成了优化问题。下一篇正是从这里接上。

---

## 小结

DQN 的贡献，工程的成分丝毫不少于理论。"用神经网络替换 Q 表"是显而易见的一步；真正把它从一个不稳定的算法变成可训练的工程系统的，是经验回放与目标网络这两个工程化想法。它们正好把致命三角中的"离策略"和"自举"两条边各自压住到能用的程度。

DQN 之后的每个变体也都是"针对某个具体失败提出某个具体修补"：Double 抹掉高估偏差，Dueling 拆开状态价值与动作优势，PER 把学习算力聚焦在"出乎意料"的转移上，Rainbow 则证明这些修补可以叠加。这些组件后来一路活到了今天——SAC、离线 RL，乃至许多 LLM-RL 杂交方法里，都还能看到经验回放和目标网络的身影。

**下一篇**：[第 3 部分](/zh/强化学习-三-策略梯度与actor-critic方法/) 进入**策略梯度**与 **Actor-Critic** 架构——能处理连续动作的那一大类算法，PPO、SAC 与现代 RL 算法的"策略侧"都从这里展开。

---

## 参考文献

- Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. *NIPS Deep Learning Workshop*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep RL with Double Q-learning. *AAAI*.
- Wang, Z., et al. (2016). Dueling Network Architectures for Deep RL. *ICML*.
- Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR*.
- Bellemare, M., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
- Fortunato, M., et al. (2018). Noisy Networks for Exploration. *ICLR*.
- Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*（第 2 版）. MIT Press. —— 第 11 章讲致命三角。

---

## 系列导航

| 部分 | 主题 |
|------|------|
| 1 | [基础与核心概念](/zh/强化学习-一-基础与核心概念/) |
| **2** | **Q-Learning 与深度 Q 网络（本文）** |
| 3 | [Policy Gradient 与 Actor-Critic 方法](/zh/强化学习-三-策略梯度与actor-critic方法/) |
| 4 | [探索策略与好奇心驱动学习](/zh/强化学习-四-探索策略与好奇心驱动学习/) |
| 5 | [Model-Based 强化学习与世界模型](/zh/强化学习-五-model-based强化学习与世界模型/) |
| 6 | [PPO 与 TRPO](/zh/强化学习-六-ppo与trpo-信任域策略优化/) |
