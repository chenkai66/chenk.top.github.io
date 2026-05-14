---
title: "强化学习（二）：Q-Learning 与深度 Q 网络（DQN）"
date: 2025-08-06 09:00:00
tags:
  - Reinforcement Learning
  - Q-Learning
  - DQN
  - Deep Q-Network
  - Experience Replay
  - Target Network
  - Rainbow
  - Value-Based Methods
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "DQN 如何结合神经网络与 Q-Learning 玩转 Atari——经验回放、目标网络、Double DQN、Dueling DQN、优先经验回放与 Rainbow。"
disableNunjucks: true
series_order: 2
translationKey: "reinforcement-learning-2"
---
2013 年 12 月，DeepMind 的一个小团队在 arXiv 上发布了一篇论文，提出了一个令人震撼的成果：一个神经网络仅凭原始像素和游戏得分，就学会了玩七款 Atari 游戏，并在其中六款上超越了此前的最佳表现。没有针对特定游戏设计的特征，也没有手工编写的启发式规则——Pong、Breakout 和 Space Invaders 全都使用同一套架构。这个算法就是 **Deep Q-Network（DQN）**，它正式拉开了深度强化学习时代的序幕。

DQN 并非凭空创造，而是对 Chris Watkins 在 1989 年提出的 **Q-Learning** 算法的扩展：用神经网络替代了查表，并辅以两项关键工程技巧，防止训练过程失控。本文将详细解释这两项技巧各自解决了什么问题，手把手带你用 PyTorch 实现一个完整的 DQN 智能体，并梳理那些让 DQN 从 Atari 演示蜕变为现代强化学习主力的各种改进版本。
![强化学习（二）：Q-Learning 与深度 Q 网络（DQN） — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/illustration_1.png)


---

## 你将学到什么

- 为什么表格型 Q-Learning 在高维状态空间中会失效
- **致命三角（Deadly Triad）**——导致朴素深度强化学习发散的三个关键因素
- DQN 的两项创新：**经验回放**和**目标网络**——它们分别缓解了致命三角中的哪一环
- 一个完整、可运行的 Atari **DQN 智能体**的 PyTorch 实现
- DQN 的后续演进：**Double DQN**、**Dueling DQN**、**优先经验回放**和 **Rainbow**

**前置知识**：需掌握 [第 1 部分](/zh/reinforcement-learning/01-基础与核心概念/) 中关于 MDP、Bellman 方程以及时序差分（TD）的基本直觉。

---

## Q-Learning 基础

### 再读 Bellman 最优方程

回顾第 1 部分，对于最优策略 $\pi^*$，其动作价值函数满足 Bellman 最优方程：
$$Q^*(s, a) \;=\; \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\Big]$$
可以将其理解为一份契约：在状态 $s$ 下执行动作 $a$ 的价值，等于即时奖励加上折扣后的最佳未来收益。一旦我们得到了 $Q^*$，最优策略就很简单：$\pi^*(s) = \arg\max_a Q^*(s, a)$——无需规划，也无需搜索，只需查表取最大值。

于是整个问题就归结为：**估计 $Q^*$**。Q-Learning 正是实现这一目标的一种方法。

### Q-Learning 更新规则

当智能体在状态 $S_t$ 执行动作 $A_t$，观察到奖励 $r_t$ 和下一状态 $S_{t+1}$ 后，Q-Learning（Watkins, 1989）会更新该状态-动作对的估值：
$$Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha \underbrace{\Big[r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big]}_{\text{TD 误差 } \delta_t}$$
方括号内的量即 **TD 误差**，表示一步自举估计与当前估值之间的差距。若 $\delta_t > 0$，说明低估了，应上调；反之则下调。

Q-Learning 有两个关键特性：

1. **离策略（Off-policy）**：目标使用的是 $\max_{a'}$（即贪心动作），与智能体实际采取的动作无关。行为策略（通常为 $\varepsilon$-greedy）可自由探索，而目标策略始终保持贪心。这种解耦赋予了 Q-Learning 强大的灵活性，但也埋下了不稳定性的种子。
2. **收敛性保证**：Watkins 和 Dayan（1992）证明，只要每个状态-动作对被无限次访问，且学习率满足 Robbins-Monro 条件（$\sum_t \alpha_t = \infty$，$\sum_t \alpha_t^2 < \infty$），那么 $Q(s, a)$ 将以概率 1 收敛到 $Q^*(s, a)$。

一个训练好的 Q 表虽小却信息丰富。下图展示了一个 4×4 网格世界的 Q 表：目标格奖励 +1，陷阱惩罚 -1，每步代价 -0.04。每个格子包含四个数值（对应四个动作），箭头标出了通过 $\arg\max_a Q(s, a)$ 得到的贪心动作。

![4x4 网格世界上的 Q 表——每格四个 Q 值与一个贪心箭头](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig1_qtable_gridworld.png)

### 实例：Cliff Walking

Sutton 与 Barto 教科书中的 Cliff Walking 几乎是 Q-Learning 的标准测试环境。智能体在一个 4×12 的网格上移动，起点与终点之间的底行是悬崖：踩入即扣 -100 分并重置回合；其他每步代价为 -1。因此最优路径是紧贴悬崖上方走一行，总回报为 -13。

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

有趣的是调整 $\varepsilon$ 的效果：过于贪心，智能体永远找不到那条贴边捷径；过度探索，又会反复坠崖。

![不同探索率下的 Cliff Walking 学习曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig2_cliff_walking_curves.png)

$\varepsilon = 0.10$ 的曲线快速上升并稳定在接近最优的水平；$\varepsilon = 0.50$ 则早早停滞在远低于最优的位置，因为随机动作（常导致坠崖）太多；$\varepsilon = 0.01$ 虽最终收敛，但因早期探索不足而耗时更长。这种“探索 vs 利用”的权衡贯穿整个强化学习领域。

---

## 为什么表格不够用

Cliff Walking 仅有 48 个状态，Q 表只需存储 192 个浮点数。但换成 Atari 的 Breakout 就完全不同了。DQN 论文将屏幕预处理为 84×84 的灰度帧，并堆叠最近四帧以感知运动。此时状态空间为 $\{0, \ldots, 255\}^{84 \times 84 \times 4}$，总状态数高达 $256^{28224}$——宇宙中的原子数量都远远不够。

这直接导致两个根本问题：

1. **无法存储**：没有任何表格能容纳每个状态对应的 Q 值。
2. **无法充分访问**：Q-Learning 的收敛性要求每个状态-动作对被无限次访问，但智能体几乎不可能两次看到完全相同的 84×84×4 像素数组。

解决方案是用**参数化函数近似器**替代查表：即使用一个带参数 $\theta$ 的函数 $Q(s, a; \theta)$（在 DQN 中是一个卷积网络），对视觉上相似的状态进行泛化。例如，两帧画面若仅在噪声或砖块颜色上有微小差异，应映射到相近的 Q 值。

### 致命三角

泛化并非没有代价。Sutton 与 Barto 指出，以下三个特性**同时存在**时，会导致基于价值的强化学习算法发散，他们称之为 **致命三角（Deadly Triad）**：

1. **自举（Bootstrapping）**：TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 将网络自身的当前估计当作“真值”。
2. **函数近似（Function approximation）**：更新 $\theta$ 以修正某个状态的 Q 值时，会连带改变邻近所有状态的估值——你无法像修改表格那样只改一个单元格。
3. **离策略数据（Off-policy data）**：用于更新的数据并非来自当前正在评估的策略。

任意两者组合尚可接受：表格 Q-Learning 是“自举 + 离策略”但无函数近似；线性近似配合在策略训练也稳定。但三者齐聚时，算法可能彻底失控——甚至能在简单 MDP 中构造出 Q 值发散至无穷的例子。

DQN 的两项创新，正是分别针对“离策略”和“自举”这两个环节。

---

## DQN 的两大创新

![强化学习（二）：Q-Learning 与深度 Q 网络（DQN） — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/02-q-learning-and-dqn/illustration_2.png)

### 经验回放：打破时间相关性

在监督学习中，mini-batch SGD 假设数据独立同分布（i.i.d.），因此可以打乱后采样。但在强化学习中，连续的状态转移 $(s_t, a_t, r_t, s_{t+1})$ 与 $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ 高度相关——下一状态直接由当前状态决定。若直接用此序列训练，梯度更新会在相关样本间剧烈震荡，旧经验刚离开视野就被遗忘，收敛变得极其困难。

DQN 引入了 **经验回放缓冲区（replay buffer）**：将每次交互得到的转移存入一个大型 FIFO 队列（通常容量为 100 万条），并在每次训练时从中随机抽取 mini-batch。

![经验回放缓冲区：流式写入，随机抽取 mini-batch](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig4_replay_buffer.png)

这一设计带来三大好处：

- **解相关的 mini-batch**：随机抽取的 32 条转移近似 i.i.d.，满足 SGD 的基本假设。
- **提升样本效率**：每条经验可被多次复用，DeepMind 报告称样本效率提升约一个数量级。
- **平滑分布偏移**：缓冲区混合了多个近期策略产生的数据，即使策略快速变化，训练分布也能缓慢演化。

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

### 目标网络：别追自己的尾巴

即便有了经验回放，TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 仍会随 $\theta$ 的更新而变动——目标本身是优化变量的函数，导致损失曲面在每一步都在变形。实践中，这常引发振荡甚至发散。

DQN 的对策是维护**两个网络副本**：

- **在线网络（online network）** $Q(\cdot; \theta)$：每步通过梯度下降更新。
- **目标网络（target network）** $Q(\cdot; \theta^-)$：一个冻结副本，定期从在线网络同步参数。

损失函数基于目标网络计算：
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big]$$
每隔 $C$ 步（Nature 论文中为 10,000 步），执行一次硬同步：$\theta^- \leftarrow \theta$。在此期间，目标保持恒定，使每个 $C$ 步窗口近似于一个有监督的回归问题。

![目标网络：一份滞后的副本，稳住 TD 目标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig5_target_network.png)

另一种连续更新方式是 **Polyak 软更新（soft update）**：$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$，其中 $\tau \approx 0.005$。DDPG 和 SAC 均采用此法，效果同样出色。

## 完整的 Atari DQN

以 2026 年的标准看，DQN 架构相当小巧：三层卷积加两层全连接，参数量约 170 万。卷积层提取 84×84 屏幕的局部特征，全连接层将展平后的特征映射到每个动作的 Q 值。

![DQN 架构：4 帧叠加输入 → 三层卷积 → FC 512 → 每个动作一个 Q 值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig3_dqn_architecture.png)

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

        # 在线网络计算 Q(s, a)
        q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # 冻结目标网络提供自举目标——这里不回传梯度
        with torch.no_grad():
            q_next = self.target_net(s_next).max(dim=1).values
            q_target = r + (1.0 - d) * self.gamma * q_next

        # Huber loss 对偶尔的大 TD 误差更鲁棒
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

一些论文未提及但实战中至关重要的细节：

- 使用 **Huber loss**（`smooth_l1_loss`）而非纯 MSE——配合梯度裁剪，可避免少数极端 TD 误差破坏优化过程。
- **梯度裁剪**至范数 10（或 1），防止罕见的大更新引发训练崩溃。
- Atari 环境需进行 **frame-skip 4 并对最后两帧做 max-pool**，以应对画面闪烁问题。这是环境预处理，不属于智能体本身。
- 在 Pong 上，单 GPU 训练约 200–300 回合即可达到接近最优表现（平均回合奖励 $\approx +21$）。

## DQN 变体：从 Double 到 Rainbow

原始 DQN 分别于 2013 年（workshop）和 2015 年（*Nature*）发表。此后数年，一系列论文陆续针对其具体弱点提出改进。最终，这些技术被整合为一个统一框架——**Rainbow**，其各组件效果近乎可叠加。

### Double DQN：消除最大化偏差

即使目标网络设计完美，$\max_{a'}$ 操作本身也会引入系统性高估偏差。原因简洁而深刻：假设真实 Q 值为 $Q^*(s, a)$，网络估计为 $Q^*(s, a) + \varepsilon_a$，其中噪声 $\varepsilon_a$ 均值为零。那么：
$$\mathbb{E}\Big[\max_a \big(Q^*(s, a) + \varepsilon_a\big)\Big] \;\geq\; \max_a Q^*(s, a)$$
max 操作倾向于选择带有**正向误差**的动作。通过自举机制，这种高估会逐步污染所有状态的目标值。实践中，原版 DQN 的预测 Q 值常显著高于真实回报。

van Hasselt 等人（2016）提出 **Double DQN**：将动作的**选择**与**评估**解耦。在线网络选择动作，目标网络评估其价值：
$$y_t = r_t + \gamma\, Q\big(s_{t+1},\; \arg\max_{a'} Q(s_{t+1}, a'; \theta);\; \theta^-\big)$$
由于两个网络的误差部分独立，系统性偏差得以大幅抵消。

![Double DQN 紧贴真实 Q 值；原版 DQN 显著高估](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig6_double_vs_vanilla.png)

代码改动仅需两行：

```python
with torch.no_grad():
    # 用在线网络选择下一个动作
    next_actions = self.policy_net(s_next).argmax(dim=1, keepdim=True)
    # 用目标网络评估该动作的价值
    q_next = self.target_net(s_next).gather(1, next_actions).squeeze(1)
    q_target = r + (1.0 - d) * self.gamma * q_next
```

### Dueling DQN：分离“状态有多好”与“动作有多好”

在许多 Atari 帧中，动作选择其实无关紧要——球尚未飞近、敌人仍在屏幕外，几帧内任何操作都无法影响结果。若将状态价值与动作优势混在同一 Q 头中，会浪费大量容量反复学习相同的 $V(s)$。

Wang 等人（2016）将输出头拆分为两条流：
$$Q(s, a) \;=\; V(s) + \Big(A(s, a) - \tfrac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Big)$$
- $V(s)$ —— “当前状态有多好？” 与动作无关。
- $A(s, a)$ —— “动作 $a$ 比平均水平好多少？” 与动作相关。

减去均值的操作确保分解唯一可识别；否则 $V$ 与 $A$ 可互相吸收任意常数。

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

Dueling 与 Double DQN 配合极佳——二者解决不同问题，效果可干净叠加。

### 优先经验回放（PER）

均匀采样假设所有转移同等重要，但这显然不成立：预测已准确的转移几乎无学习价值，而 TD 误差大的转移则蕴含强信号。

优先经验回放（Schaul 等人，2016）按 TD 误差绝对值的幂次进行采样：
$$p_i \;\propto\; \big(|\delta_i| + \varepsilon\big)^\alpha$$
指数 $\alpha \in [0, 1]$ 控制优先级强度（$\alpha = 0$ 退化为均匀采样）。由于非均匀采样会引入偏差，需用重要性采样权重校正：
$$w_i = \Big(\tfrac{1}{N \cdot p_i}\Big)^\beta$$
其中 $\beta$ 从 0.4 逐渐增至 1.0——早期容忍偏差，后期强调稳定性。

### 多步回报、分布式 RL、NoisyNet

Rainbow 还整合了三项关键技术：

- **n 步回报**（Rainbow 中 $n=3$）：用部分 Monte-Carlo 目标替代单步自举，在增加方差的同时降低偏差：
  $$
  y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)
  $$
- **分布式 RL（C51）**（Bellemare 等人，2017）：不再仅学习回报均值，而是建模整个回报分布。通过将分布投影到 51 个固定“原子”上得名 C51。完整分布保留了均值丢失的信息，尤其在随机环境中更为关键。

- **NoisyNet**：用注入全连接层权重的可学习噪声替代 $\varepsilon$-greedy 探索。探索由此变为状态相关且可学习，无需手动调整衰减表。

### 彩虹（Rainbow）

Hessel 等人（2018）将六项改进（DQN 基础 + Double + Dueling + PER + n 步 + 分布式 + NoisyNet）整合，并进行了严谨的消融实验。每个组件均贡献显著，组合后性能明显优于任一单独变体。

![Atari-57 基准上的人类标准化中位分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig7_atari_benchmark.png)

图中得分按“人类 = 100%、随机智能体 = 0%”标准化。原版 DQN 中位分低于人类；每项改进提升数十个百分点；Rainbow 在中位游戏上的表现超过人类一倍以上。

## 实战要点

### 超参数参考表

| 超参数              | 常用值                  | 说明                                   |
| ------------------- | ----------------------- | -------------------------------------- |
| 回放缓冲区容量      | 100K - 1M 转移          | 越大越好，受限于内存                   |
| mini-batch 大小     | 32 - 128                | 32 是 Nature 论文默认值                |
| 学习率              | 1e-4 到 3e-4            | Adam 优化器，2.5e-4 是稳妥起点         |
| 折扣 $\gamma$       | 0.99                    | 短视界任务可适当调低                   |
| $\varepsilon$ 衰减  | 1.0 → 0.01              | 在约 1M 帧内线性衰减                   |
| 目标网络更新周期 $C$| 10K 步                  | 或使用软更新，$\tau \approx 0.005$    |
| Frame skip          | 4                       | Atari 标准预处理                       |
| 梯度裁剪            | 范数 10                 | 建议搭配 Huber loss 使用               |

### 调试清单

1. **先搞定 CartPole**：正确实现的 DQN 应在单 CPU 上 200 回合内解决 CartPole。若失败，问题大概率在代码而非超参。
2. **监控 Q 值幅度**：Q 值应先上升后稳定在合理范围（Atari 上通常为个位或两位数）。若突破 1000，基本可判定发散。
3. **跟踪 TD 误差**：应逐渐下降并保持有界。若单调上升，说明目标值失控。
4. **检查动作分布**：若智能体始终选择同一动作，可能是网络输出坍缩——排查 ReLU 死亡、初始化错误或学习率过高。

### 何时用 DQN，何时选其他？

| 维度          | DQN 家族                | 策略梯度（PPO / SAC 等）         |
| ------------- | ----------------------- | -------------------------------- |
| 动作空间      | 仅支持离散              | 支持离散和连续                   |
| 样本效率      | 高（回放机制复用数据）  | 较低（PPO 为在策略）             |
| 稳定性        | 需目标网络等技巧        | 整体更易调参                     |
| 最佳适用场景  | Atari、棋类、离散控制   | 机器人、运动控制、连续控制       |

DQN 的硬伤在于仅适用于离散动作空间：对五个按键做 $\arg\max_a Q(s, a)$ 很简单，但在实值关节角度上则不可行。下一篇将正好从这里展开。

## 总结

DQN 的贡献，工程与理论各占一半。用神经网络替代 Q 表是最直观的改动，但真正让算法从不稳定走向实用的，是经验回放缓冲区和目标网络这两项设计。它们有效压制了“致命三角”中的离策略与自举问题，使得非线性函数逼近器能够稳定学习。

后续的 DQN 变体均针对具体缺陷提出精准修复：Double DQN 消除高估偏差，Dueling DQN 分离状态价值与动作优势，优先经验回放聚焦于意外转移，而 Rainbow 则证明这些改进可有效叠加。这些核心组件至今仍活跃于现代算法中——无论是 SAC、离线 RL，还是众多 LLM-RL 混合方法，都能看到经验回放与目标网络的身影。

**下一篇**：[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法) 将介绍 **策略梯度** 方法与 **Actor-Critic** 架构——这是处理连续动作的核心家族，也是 PPO、SAC 及所有现代算法中“策略侧”的基础。

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
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. -- Chapter 11 for the Deadly Triad.
