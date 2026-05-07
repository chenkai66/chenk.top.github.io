---
title: "强化学习（二）：Q-Learning 与深度 Q 网络（DQN）"
date: 2025-08-06 09:00:00
tags:
  - 强化学习
  - Q-Learning
  - DQN
  - 经验回放
  - 目标网络
  - Rainbow
  - 价值方法
categories: 强化学习
series: reinforcement-learning
lang: zh-CN
mathjax: true
description: "DQN 如何结合神经网络与 Q-Learning 玩转 Atari——经验回放、目标网络、Double DQN、Dueling DQN、优先经验回放与 Rainbow。"
disableNunjucks: true
series_order: 2
translationKey: "reinforcement-learning-2"
---
2013 年 12 月，DeepMind 的一个小团队在 arXiv 上发布了一篇论文，提出了一个令人震撼的成果：一个神经网络，直接从原始像素和得分学习，掌握了七款 Atari 游戏，并在其中六款中超越了之前的最佳表现。没有针对游戏设计的特定特征，也没有手工编写的启发式规则，Pong、Breakout 和 Space Invaders 都用的是同一套架构。这个算法叫 **Deep Q-Network (DQN)**，它开启了深度强化学习的新时代。

DQN 并不是凭空创造的。它是 Watkins 在 1989 年提出的 **Q-Learning** 算法的升级版，把查表替换成了神经网络，同时引入了两个工程技巧，避免了训练过程失控。我会详细解释这两个技巧解决了什么问题，手把手带你用 PyTorch 实现一个完整的 DQN，并梳理那些让 DQN 从 Atari 演示变成现代强化学习主力的各种改进版本。
![强化学习（二）：Q-Learning 与深度 Q 网络（DQN） — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-q-learning-and-dqn/illustration_1.jpg)

## 你将学到什么

- 表格型 Q-Learning 在高维状态空间中为什么会失效
- **致命三角（Deadly Triad）**——导致朴素深度强化学习发散的三个关键因素
- DQN 的两项创新：**经验回放**和**目标网络**——它们分别解决了致命三角中的哪个问题
- 一个完整的、可运行的 Atari **DQN 智能体**的 PyTorch 实现
- DQN 之后的改进版本：**Double DQN**、**Dueling DQN**、**优先经验回放**和 **Rainbow**

**前置知识**：需要了解 [第 1 部分](/zh/reinforcement-learning/01-基础与核心概念/) 中的 MDP、Bellman 方程以及时序差分（TD）的基本思想。

---
## Q-Learning 基础

### 再读 Bellman 最优方程

回顾第 1 部分，对于最优策略 $\pi^*$，动作价值函数满足 Bellman 最优方程：

$$Q^*(s, a) \;=\; \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\Big]$$

可以把它看成一份契约：在状态 $s$ 下执行动作 $a$ 的价值，等于即时奖励加上折扣后的最佳未来收益。一旦有了 $Q^*$，最优策略就是查表取最大值：$\pi^*(s) = \arg\max_a Q^*(s, a)$。不需要规划，也不需要搜索。

问题的核心就变成了：**估算 $Q^*$**。Q-Learning 是一种方法。

### Q-Learning 更新规则

智能体在状态 $S_t$ 执行动作 $A_t$ 后，观察到奖励 $r_t$ 和下一状态 $S_{t+1}$。Q-Learning（Watkins, 1989）只更新当前访问的状态-动作对：

$$Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha \underbrace{\Big[r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big]}_{\text{TD 误差 } \delta_t}$$

方括号中的量是 **TD 误差**，表示一步自举估计值与当前估值之间的差距。如果 $\delta_t > 0$，说明低估了，需要上调；反之则下调。

Q-Learning 有两个重要特性：

1. **离策略**。目标使用的是 $\max_{a'}$，也就是贪心动作，与智能体下一步实际选择的动作无关。行为策略（通常是 $\varepsilon$-greedy）可以自由探索，而目标策略始终保持贪心。这种解耦是 Q-Learning 的优势，也是它不稳定性的根源。
2. **收敛性保证**。Watkins 和 Dayan（1992）证明：如果每个状态-动作对都被访问无穷多次，并且学习率满足 Robbins-Monro 条件（$\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$），那么 $Q(s, a)$ 会以概率 1 收敛到 $Q^*(s, a)$。

训练好的 Q 表是一个直观的对象。下图展示了一个 4x4 网格世界的 Q 表，目标格奖励 +1，陷阱惩罚 -1，每走一步扣 -0.04。每个格子有四个数字（对应四个动作），箭头指向 $\arg\max_a Q(s, a)$ 给出的贪心选择。

![4x4 网格世界上的 Q 表——每格四个 Q 值与一个贪心箭头](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig1_qtable_gridworld.png)

### 实例：Cliff Walking

Sutton 和 Barto 教科书中的 Cliff Walking 几乎是 Q-Learning 的标配实验。智能体在一个 4x12 的网格上行走，起点和终点之间的底边是悬崖：踩进去扣 -100，然后被弹回起点。其他每走一步扣 -1，所以最优路线是**贴着悬崖上沿走一格**，回报为 -13。

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

有趣的是调整 $\varepsilon$ 的效果：太贪心，智能体找不到那条贴边的最优路；太爱探索，又会反复掉下悬崖。

![不同探索率下的 Cliff Walking 学习曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig2_cliff_walking_curves.png)

$\varepsilon = 0.10$ 的曲线收敛最快，稳定接近最优值；$\varepsilon = 0.50$ 卡在远低于最优的水平，因为太多步是随机的（容易致命）；$\varepsilon = 0.01$ 最终也能达到最优，但早期探索太少，收敛慢得多。这种"探索 vs 利用"的权衡贯穿整个强化学习领域。

---
## 为什么表格不够用

Cliff Walking 只有 48 个状态，Q 表存 192 个浮点数就够了。但换成 Atari 的 Breakout 就不一样了。DQN 论文把屏幕预处理成 84x84 的灰度图，再叠加最近的四帧画面，让智能体感知运动。这样一来，状态变成了 $\{0, \ldots, 255\}^{84 \times 84 \times 4}$ 中的一个向量，可能的状态数量高达 $256^{28224}$。宇宙里的原子都不够用来存储这些状态。

这直接导致两个问题：

1. **存不下**。任何表都装不下每个状态对应的 Q 值。
2. **访问不到**。Q-Learning 的收敛性要求每个状态-动作对被访问无穷多次。但智能体根本不可能两次看到完全相同的 84x84x4 数组。

解决办法是用**带参数的函数近似器**代替表格。具体来说，就是用一个参数化的函数 $Q(s, a; \theta)$（在 DQN 中是一个卷积网络）来泛化视觉上相似的状态。比如，两帧画面如果只是噪声不同或者砖块颜色稍有变化，就应该映射到接近的 Q 值。

### 致命三角

泛化是有代价的。Sutton 和 Barto 指出，在基于价值的强化学习中，三个特性同时出现时会导致算法发散，他们称之为**致命三角**：

1. **自举（Bootstrapping）**：TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 把网络当前的估值当作"真值"。
2. **函数近似**：调整 $\theta$ 修正某个状态的 Q 值时，会连带影响附近所有状态的 Q 值——你没法像改表格那样只改一个格子。
3. **离策略数据**：用于更新的数据不是来自当前正在评估的策略。

任意两条单独存在都没问题：表格 Q-Learning 是"自举 + 离策略"，但没有函数近似；线性近似加在策略也没问题。但如果三条同时出现，就真的不安全了——可以构造出 Q 值发散到无穷的简单 MDP。

DQN 的两项创新分别针对致命三角中的"离策略"和"自举"两条边。

---
## DQN 的两大创新

![强化学习（二）：Q-Learning 与深度 Q 网络（DQN） — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-q-learning-and-dqn/illustration_2.jpg)

### 经验回放：打破时间相关性

监督学习中，mini-batch SGD 假设数据是独立同分布的（i.i.d.），也就是可以打乱顺序再采样。但在强化学习里，连续的状态转移 $(s_t, a_t, r_t, s_{t+1})$ 和 $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ 高度相关——下一个状态直接由当前状态决定。如果直接用这些数据训练，梯度更新会在相关样本间来回震荡，旧经验刚离开视野就被遗忘，收敛变得遥遥无期。

DQN 提出了一个解决方案：**经验回放缓冲区**。它是一个大容量的 FIFO 队列（通常能存 100 万条数据）。每次智能体与环境交互后，都会把状态转移存入缓冲区。训练时，随机从中抽取 mini-batch。

![经验回放缓冲区：流式写入，随机抽取 mini-batch](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig4_replay_buffer.png)

这个设计带来了三个好处：

- **解相关 mini-batch**。随机抽取的 32 条样本近似独立同分布，正好满足 SGD 的假设。
- **提高样本利用率**。每条经验会被反复使用，DeepMind 报告称样本效率提升了约 10 倍。
- **平滑分布变化**。缓冲区混合了多个近期策略产生的数据，即使策略快速变化，训练分布也能保持相对稳定。

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

即使有了经验回放，TD 目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 还是会随着 $\theta$ 的更新而变化。目标本身是优化变量的函数，损失曲面在每一步都在变形。实际测试发现，这会导致振荡甚至发散。

DQN 的解决办法是维护两个网络：

- **在线网络** $Q(\cdot; \theta)$：每一步都通过梯度下降更新。
- **目标网络** $Q(\cdot; \theta^-)$：在线网络的一份“冻结副本”，每隔一段时间同步一次。

损失计算基于目标网络：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big]$$

每隔 $C$ 步（Nature 论文中是 10,000 步），执行一次同步：$\theta^- \leftarrow \theta$。两次同步之间，目标网络保持不变，每个 $C$ 步窗口就像一个标准的有监督回归问题。

![目标网络：一份滞后的副本，稳住 TD 目标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig5_target_network.png)

还有一种连续版本的更新方法——**Polyak 软更新**：$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$，其中 $\tau$ 通常取 0.005。DDPG 和 SAC 都用这种方法，效果同样出色。
## 完整的 Atari DQN

以 2026 年的标准来看，DQN 的架构非常简单：3 个卷积层加 2 个全连接层，参数量大约 170 万。卷积层负责提取 84x84 屏幕的局部特征，全连接层则将展平后的特征映射到每个动作对应的 Q 值。

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

实战中一些论文没提到但很重要的细节：

- 用 **Huber loss**（`smooth_l1_loss`）代替纯 MSE。配合梯度裁剪，偶尔的大 TD 误差不会让优化器崩溃。
- **梯度裁剪**限制范数到 10（或 1），避免罕见的大更新导致训练不稳定。
- Atari 环境需要做 **frame-skip 4 和后两帧 max-pool**，解决画面闪烁问题。这是环境预处理，不是智能体的一部分。
- 在 Pong 上，单 GPU 跑 200-300 回合就能接近最优表现（平均回报 $\approx +21$）。
## DQN 变体：从 Double 到 Rainbow

DQN 最早的版本分别在 2013 年（workshop）和 2015 年（*Nature*）发布。随后几年，研究者们陆续发表了一系列论文，每篇都针对一个具体问题进行改进。最终，这些改进被整合到一个统一的智能体中，称为 **Rainbow**，它的各个组件几乎可以线性叠加。

### Double DQN：消除最大化偏差

即使目标网络设计得再完美，$\max_{a'}$ 操作本身也会引入系统性的高估偏差。原因很简单也很优雅：假设真实 Q 值是 $Q^*(s, a)$，而网络估计值为 $Q^*(s, a) + \varepsilon_a$，其中噪声 $\varepsilon_a$ 的期望为零。那么：

$$\mathbb{E}\Big[\max_a \big(Q^*(s, a) + \varepsilon_a\big)\Big] \;\geq\; \max_a Q^*(s, a)$$

max 操作总是倾向于选择那些带有**正向误差**的动作。通过自举机制，这种高估会逐渐传播到所有状态的目标值中。实际测试中，原版 DQN 的预测 Q 值往往会显著高于真实的回报。

van Hasselt 等人（2016）提出了 **Double DQN**：将动作的选择和评估分开。在线网络负责选择下一步动作，目标网络负责评估这个动作的价值。

$$y_t = r_t + \gamma\, Q\big(s_{t+1},\; \arg\max_{a'} Q(s_{t+1}, a'; \theta);\; \theta^-\big)$$

两个网络的误差部分独立，因此系统性偏差基本被抵消。

![Double DQN 紧贴真实 Q 值；原版 DQN 显著高估](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig6_double_vs_vanilla.png)

代码改动只有两行：

```python
with torch.no_grad():
    # 用在线网络选择下一个动作
    next_actions = self.policy_net(s_next).argmax(dim=1, keepdim=True)
    # 用目标网络评估该动作的价值
    q_next = self.target_net(s_next).gather(1, next_actions).squeeze(1)
    q_target = r + (1.0 - d) * self.gamma * q_next
```

### Dueling DQN：分离“状态有多好”和“动作有多好”

在很多 Atari 游戏帧中，选择哪个动作其实无关紧要——球还没飞过来、敌人还没出现，几帧内无论如何操作都不会有影响。如果把状态价值和动作优势混在一个 Q 头里，就会浪费大量容量去反复学习同一个 $V(s)$。

Wang 等人（2016）将输出头拆分为两条流：

$$Q(s, a) \;=\; V(s) + \Big(A(s, a) - \tfrac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Big)$$

- $V(s)$ —— “当前状态有多好？” 与动作无关。
- $A(s, a)$ —— “选择动作 $a$ 比平均水平好多少？” 与动作相关。

减去均值的操作是为了让分解唯一可识别，否则 $V$ 和 $A$ 之间可以互相吸收任意常数。

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

Dueling DQN 和 Double DQN 配合得很好——两者解决的是不同问题，效果可以叠加。

### 优先经验回放（PER）

均匀采样假定每条转移数据的重要性相同，但这显然不对：预测已经准确的转移几乎没有学习价值，而 TD 误差大的转移则包含更强的学习信号。

优先经验回放（Schaul 等人，2016）按照 TD 误差绝对值的某个幂次来采样：

$$p_i \;\propto\; \big(|\delta_i| + \varepsilon\big)^\alpha$$

指数 $\alpha \in [0, 1]$ 控制优先级强度，$\alpha = 0$ 时退化为均匀采样。由于非均匀采样会导致梯度估计有偏，需要用重要性采样权重进行修正：

$$w_i = \Big(\tfrac{1}{N \cdot p_i}\Big)^\beta$$

$\beta$ 在训练过程中从 0.4 逐渐增加到 1.0——前期偏差影响较小，后期需要严格修正以保证稳定性。

### 多步回报、分布式 RL、NoisyNet

Rainbow 还集成了三项改进：

- **n 步回报**（Rainbow 中 $n=3$）：用局部 Monte-Carlo 目标替代单步自举，以方差换取偏差的降低：

  $$
  y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)
  $$

- **分布式 RL（C51）**，Bellemare 等人（2017）：不再只学习回报的均值，而是学习整个回报分布。通过将分布投影到 51 个固定原子上得名 C51。完整分布包含的信息（尤其在随机环境中）远多于单一均值。

- **NoisyNet**：用注入到全连接层权重中的可学习噪声替代 $\varepsilon$-greedy 探索。探索从此变成状态相关且可学习的，而不是依赖手动调参的衰减表。

### Rainbow

Hessel 等人（2018）将六项改进（DQN 基础 + Double + Dueling + PER + n 步 + 分布式 + NoisyNet）整合在一起，并进行了详细的消融实验。每个组件单独都能带来提升，组合后的性能明显优于任何单一变体。

![Atari-57 基准上的人类标准化中位分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/02-Q-Learning与深度Q网络/fig7_atari_benchmark.png)

图中得分按“人类 = 100%、随机智能体 = 0%”标准化。原版 DQN 的中位分低于人类水平；每加一项改进就提升几十个百分点；Rainbow 在中位游戏上的表现比人类高出一倍多。
## 实战要点

### 超参数参考表

| 超参数              | 常用值                  | 说明                                   |
| ------------------- | ----------------------- | -------------------------------------- |
| 回放缓冲区容量      | 100K - 1M 转移          | 越大越好，但受限于内存                 |
| mini-batch 大小     | 32 - 128                | 32 是 Atari 的默认值                   |
| 学习率              | 1e-4 到 3e-4            | Adam 优化器，2.5e-4 是个稳妥的起点     |
| 折扣 $\gamma$       | 0.99                    | 短时任务可以调低                      |
| $\varepsilon$ 衰减  | 1.0 → 0.01              | 在约 1M 帧内线性衰减                   |
| 目标网络更新周期 $C$| 10K 步                  | 或使用软更新，$\tau \approx 0.005$    |
| Frame skip          | 4                       | Atari 预处理标准                       |
| 梯度裁剪            | 范数 10                 | 推荐搭配 Huber loss 使用               |

### 调试清单

1. **先搞定 CartPole**。一个正确的 DQN 实现，在单 CPU 上不到 200 个回合就能解决 CartPole。如果搞不定，问题多半出在代码上，而不是超参数。
2. **关注 Q 值的范围**。Q 值应该先上升，然后稳定在一个合理区间（Atari 上通常是单位数或两位数）。如果超过 1000，基本可以断定是发散了。
3. **监控 TD 误差**。TD 误差应该逐渐下降并保持稳定。如果一直单调上升，说明目标值失控了。
4. **检查动作分布**。如果智能体总是选择同一个动作，可能是网络输出坍缩了——检查 ReLU 是否大面积失效、初始化是否正确、学习率是否过高。

### 什么时候用 DQN，什么时候用别的？

| 维度          | DQN 家族                | 策略梯度（PPO / SAC 等）         |
| ------------- | ----------------------- | -------------------------------- |
| 动作空间      | 仅支持离散              | 支持离散和连续                   |
| 样本效率      | 高（回放机制复用数据）  | 较低（PPO 是在线策略）           |
| 稳定性        | 需要目标网络等技巧      | 整体更容易调参                   |
| 最佳适用场景  | Atari、棋类、离散控制   | 机器人、运动控制、连续控制       |

DQN 的硬伤在于只能处理离散动作：对五个按键做 $\arg\max_a Q(s, a)$ 很简单，但在实数值的关节角度上就变得不可行。下一篇正好从这里展开讨论。
## 小结

DQN 的贡献，工程和理论各占一半。用神经网络代替 Q 表是最直观的改动，但真正让这个不稳定的算法变得可用的是经验回放池和目标网络这两个关键设计。它们成功压制了“致命三角”中的离策略和自举问题，使得非线性函数逼近器能够有效学习。

DQN 之后的各种改进版本，都是针对具体问题提出具体解决方案：Double DQN 消除了高估偏差，Dueling DQN 分离了状态价值和动作优势，Prioritized Replay 把学习重点放在意外的转移上，而 Rainbow 则证明这些改进可以叠加使用。这些核心组件一直沿用至今——无论是 SAC、离线 RL，还是许多现代 LLM-RL 混合方法，都能看到经验回放池和目标网络的身影。

**下一篇**：[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法) 将介绍**策略梯度**方法和 **Actor-Critic** 架构——这是处理连续动作的核心算法家族，也是 PPO、SAC 以及所有现代算法中“策略侧”的基础。

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
