---
title: "强化学习（五）：Model-Based强化学习与世界模型"
date: 2025-08-21 09:00:00
tags:
  - 强化学习
  - Model-Based RL
  - 世界模型
  - Dyna
  - MBPO
  - Dreamer
  - MuZero
  - PlaNet
categories:
  - 强化学习
series: reinforcement-learning
lang: zh-CN
mathjax: true
description: "从 Dyna、MBPO 到 World Models、Dreamer 和 MuZero——学一个环境模型，让智能体在想象中规划，把样本效率提高 10-100 倍。"
disableNunjucks: true
series_order: 5
translationKey: "reinforcement-learning-5"
---
到目前为止，我介绍的所有算法——DQN、REINFORCE、A2C、PPO、SAC——都属于 **Model-Free** 类型。智能体把环境当作黑盒，输入动作，接收奖励，更新策略，完全不去理解环境的内部机制。这种方法虽然有效，但效率极低。比如，DQN 需要 **1000 万帧**才能在 Atari Pong 上达到精通；OpenAI Five 在 Dota 2 上自我对弈的时间相当于 **45000 年**；AlphaStar 则消耗了数年的 StarCraft 数据来训练一个智能体。

人类显然不是这样学习的。下棋的人会提前推演几步，排除明显糟糕的走法；小孩掉一次悬崖就知道"危险"，而不是靠反复摔下去学会。这两种情况都依赖于一个内部的 **模型**，用来预测世界如何响应动作。更重要的是，大部分认知资源都花在模型里，而不是真实环境中。

**Model-Based RL（基于模型的强化学习，MBRL）** 就是把这个思路系统化：先学习一个近似的动态模型 $\hat{P}(s'\mid s,a)$ 和奖励模型 $\hat{R}(s,a)$，然后用它们作为廉价的模拟器，进行规划、策略改进或价值估计。在适用的任务上，这种方法能带来巨大的回报——**真实环境样本需求量减少 10 到 100 倍**。这就像让一个机器人从需要三个月物理交互，缩短到只需要一下午。

这篇文章梳理了 MBRL 的现代发展脉络：Dyna（1990）-> MBPO（2019）-> World Models（2018）-> Dreamer（2020-23）-> MuZero（2020）。每种方法背后都有一个核心思想，而本文的 7 张图逐一展示了这些思想。
![强化学习（五）：Model-Based强化学习与世界模型 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-model-based-rl-and-world-models/illustration_1.jpg)

## 你将学到什么

- Model-Based 强化学习赢在哪里，又输在何处，精确的权衡点是什么
- **Dyna-Q**：第一个把真实经验与想象经验结合起来更新的经典方法
- **MBPO**：为什么短时间范围的想象是最优选择
- **MPC**：用学到的模型做纯粹的规划循环
- **World Models（V/M/C）**：把像素压缩到一个潜在的“梦境”空间
- **Dreamer / RSSM**：端到端的潜在空间想象，结合了循环结构和随机状态
- **MuZero**：完全不预测观测，直接进行规划

**前置知识：**[第 1-3 部分](/zh/reinforcement-learning/01-基础与核心概念/)（MDP、价值函数、策略梯度、Actor-Critic）。

---
## 一、两种范式，一个目标

![Model-free 与 model-based 控制循环对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig1_mf_vs_mb_loops.png)

Model-Free RL 只有一个循环：*执行 -> 观察 -> 学习*。Model-Based RL 则多了一个循环：*学习模型 -> 在模型中规划 -> 改进策略*。一次真实交互可以分摊到成千上万次想象更新中，效率立刻提升。

### 核心权衡

|              | Model-Free                            | Model-Based                                   |
| ------------ | ------------------------------------- | --------------------------------------------- |
| **学什么**   | 策略 / 价值函数                       | 模型 $\hat{P},\hat{R}$ 和策略 / 价值函数      |
| **样本成本** | 高，每次梯度更新需要一次真实交互       | 低，一次真实步可生成多次想象更新               |
| **算力成本** | 单步较低                               | 较高（模型拟合 + 规划）                        |
| **渐近性能** | 仅受探索能力限制                       | 受模型偏差限制                                 |
| **迁移性**   | 绑定于训练时的奖励                     | 同一模型可复用于新任务                         |
| **失败模式** | 学习速度慢                             | 模型误差累积导致虚假最优解                     |

### 实际中的样本效率

| 算法           | 类别        | 基准测试                | 达到专家水平所需步数   |
| -------------- | ----------- | ----------------------- | ---------------------- |
| DQN            | Model-Free  | Atari Pong              | 约 1000 万帧           |
| PPO            | Model-Free  | MuJoCo HalfCheetah      | 约 100 万-200 万步     |
| SAC            | Model-Free  | MuJoCo HalfCheetah      | 约 60 万步             |
| **MBPO**       | Model-Based | MuJoCo HalfCheetah      | **约 8 万-10 万步**    |
| **Dreamer**    | Model-Based | DMControl Walker        | **约 10 万步**         |
| **DreamerV3**  | Model-Based | Minecraft（挖钻石）     | 首个从零完成的算法     |

差距大约是一个数量级，在连续控制任务中尤为明显。如果模拟器昂贵（如真实机器人或慢速物理仿真），优势更加突出。

### 什么时候选择 Model-Based

适合场景：

- 真实交互成本高：机器人、自动驾驶、药物发现、有用户参与的对话系统。
- 动力学规律可学习：物理规则平滑、棋类游戏、结构化环境。
- 多个下游任务共享同一环境，模型可以复用。

不适合场景：

- 已有免费、快速且高保真的模拟器（例如 Atari 本身就是模拟器）。
- 动力学高度随机或对抗性强（如金融市场、社交互动）。
- 状态空间维度太高，模型无法在数据预算内拟合。
## 二、Dyna-Q：最初的蓝图

![Dyna-Q 数据流和收敛曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig3_dyna_q_flow.png)

Sutton 在 1990 年提出的 **Dyna** 是第一个清晰阐述 model-based 循环的系统。每次真实的状态转移会被用三次：

1. **直接学习** —— 用真实的 $(s,a,r,s')$ 更新 Q 值；
2. **模型学习** —— 把转移存入表格化模型 $M(s,a)\to(r,s')$；
3. **规划** —— 随机采样 $n$ 条之前见过的 $(s,a)$，查询模型，用这些“想象”的转移再做 $n$ 次 Q 值更新。

右边的收敛曲线展示了结果：在确定性的 GridWorld 中，把规划步数从 0（普通的 Q-Learning）增加到 50，收敛所需的 episode 数量减少了一个数量级——因为每次真实交互现在触发了 51 次 Bellman 更新，而不是 1 次。

### 参考实现

```python
import numpy as np

class DynaQ:
    """表格化 Dyna-Q：直接学习 + 基于记忆模型的规划。"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, planning_steps=10):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}                    # (s, a) -> (r, s')
        self.visited = []                  # 用于采样的有序列表
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def _q_update(self, s, a, r, s_next):
        target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def learn(self, s, a, r, s_next):
        # 1. 直接 RL
        self._q_update(s, a, r, s_next)
        # 2. 模型学习（确定性环境直接记忆即可）
        if (s, a) not in self.model:
            self.visited.append((s, a))
        self.model[(s, a)] = (r, s_next)
        # 3. 规划：回放想象中的转移
        for _ in range(self.planning_steps):
            sp, ap = self.visited[np.random.randint(len(self.visited))]
            rp, sp_next = self.model[(sp, ap)]
            self._q_update(sp, ap, rp, sp_next)
```

### Dyna 的启示与局限

Dyna 提炼出了一个核心洞见：**学到一个模型，可以用算力替代样本**。它同时也揭示了一个所有现代方法都绕不开的问题：在错误的模型上规划，会直接将偏差注入价值函数。在表格化的确定性世界中，这个问题并不明显；但换成神经网络模型和长 horizon，误差会指数级累积。本文后续内容本质上就是针对这个问题的一系列巧妙解答。
## 三、MBPO：让想象短一些

![MBPO 短分支 rollout 与模型误差随长度增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based弱化学习与世界模型/fig4_mbpo_short_rollouts.png)

Janner 等人在 NeurIPS 2019 提出的 **Model-Based Policy Optimization（MBPO）**，是 Dyna 思想在连续控制领域最简洁的现代实现。它的核心洞见就两个字：**短 rollout**。

右图清楚地说明了问题。状态预测的累积误差随着 rollout 长度 $k$ 大致呈几何级数增长。当 $k = 20$ 时，即使是一个包含 5 个动态模型的 ensemble，误差也已经大到无法用于信用分配。MBPO 的解决办法很简单：从真实状态出发，只扩展 **1 到 5 步**（左图），然后将生成的转移交给 SAC 去完成长程信用分配——这正是无模型方法擅长的事情。

### 算法

1. 用当前策略在真实环境中采样，把数据加入 $\mathcal{D}_{\text{real}}$。
2. 在 $\mathcal{D}_{\text{real}}$ 上训练一个由 **5 个网络**组成的概率动态模型 ensemble $f_\theta(s,a)\to(s',r)$。
3. 反复从 $\mathcal{D}_{\text{real}}$ 中采样初始状态，在随机选择的 ensemble 成员中进行 $k$ 步 rollout，将生成的转移加入 $\mathcal{D}_{\text{model}}$。
4. 使用 $\mathcal{D}_{\text{real}}$ 和 $\mathcal{D}_{\text{model}}$ 的混合数据训练 SAC。

Ensemble 很关键：成员之间的分歧不仅起到了正则化作用（数据稀疏的地方分歧最大），还隐式提供了认知不确定性，帮助策略避开不可靠区域。

### 简化代码

```python
import numpy as np
import torch
import torch.nn as nn

class EnsembleDynamics(nn.Module):
    """5 个概率动态模型，预测 (delta_s, reward)。"""

    def __init__(self, state_dim, action_dim, hidden=256, n=5):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * (state_dim + 1)),  # 均值 + log_std
            ) for _ in range(n)
        ])
        self.state_dim = state_dim

    def sample(self, s, a):
        idx = np.random.randint(len(self.heads))
        x = torch.cat([s, a], dim=-1)
        out = self.heads[idx](x)
        mu, log_std = out.chunk(2, dim=-1)
        eps = torch.randn_like(mu) * log_std.exp()
        delta = (mu + eps)[..., : self.state_dim]
        reward = (mu + eps)[..., self.state_dim:]
        return s + delta, reward
```

### 结果

在 MuJoCo HalfCheetah 环境中，MBPO 仅用 **约 10 万步**真实交互就达到了约 10000 的回报，而 SAC 需要约 100 万步，PPO 则需要约 160 万步。实验表明，最佳的 rollout 长度通常是 $k=1$；更长的 rollout 会导致累积误差迅速放大，反而拖累了性能。

---
## 四、纯规划：模型预测控制（MPC）

![模型预测控制：采样 -> 评分 -> 执行第一步 -> 重新规划](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig5_mpc_planning.png)

如果模型足够好，完全可以不用策略，每一步都重新规划。模型预测控制（MPC）是经典控制工程的核心方法，直接套用学习到的动态模型就能在现代基于模型的强化学习中大显身手。

流程如下：

1. 采样$N$条候选动作序列$a_{t:t+H}$，可以是均匀分布、高斯分布，或者从 CEM/iCEM 提议分布中采样。
2. 用学到的模型将每条序列向前推$H$步，根据预测回报打分。
3. 找到最优序列后，只执行它的第一个动作。
4. 观察真实环境的下一状态，然后重新规划。

图中展示了 12 条候选轨迹（灰色）、最优的一条（绿色），以及被高亮显示并实际发送给执行器的那个动作。关键在于，每次只执行一步，模型只需要局部准确——累积误差根本没有机会破坏长程开环计划。

在错误代价极高的场景中，比如真实机器人、手术和自动驾驶，MPC 几乎是首选方案。它也是连接学习模型与经典规划文献的桥梁：PETS、PlaNet、TD-MPC 和 Dreamer 的策略改进循环，归根结底都可以看作是在学习模型中运行 MPC。

---
## 五、World Models：在潜空间里做梦

![强化学习（五）：Model-Based强化学习与世界模型 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-model-based-rl-and-world-models/illustration_2.jpg)


![World Model V/M/C 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig2_world_model_vmc.png)

MBPO 能成功，是因为 MuJoCo 的状态只有 11 到 23 维。但要预测下一帧 Atari（$84\times 84\times 3$像素）就难如登天了——而且大部分像素（比如天空、计分板）跟控制完全无关。**World Models**（Ha & Schmidhuber, 2018）提出了另一种思路：

> 把观察压缩成一个很小的潜变量编码，然后直接在潜空间里学习动力学。

从左到右，三个组件分别是：

- **V（Vision）** —— 用变分自编码器把每一帧$o_t$压缩成约 32 维的潜变量$z_t$。重建损失确保$z_t$保留了足够的场景信息。
- **M（Memory）** —— 用混合密度网络 RNN 建模$P(z_{t+1}\mid z_t, a_t, h_t)$，其中$h_t$是 RNN 的隐状态。M 就是世界模型本身。
- **C（Controller）** —— 一个故意设计得很小的线性策略，将$(z_t, h_t)$映射为$a_t$。在 CarRacing 上，它只有 **867 个参数**，而 DQN 有 170 万。

### 它为什么有效——以及为什么令人惊讶

控制器可以**完全在梦里训练**：从采样的$z$开始，用 M 生成伪轨迹，再用 CMA-ES 优化 C，直到评估阶段才接触真实环境。这个 867 参数的控制器在 CarRacing-v0 上达到了接近人类的分数。更深层的启示是，**学到一个有用的表示，已经解决了大半问题**。一旦 V 和 M 准备好，控制部分几乎就是小事一桩。这正是 Dreamer、DreamerV3 和 TD-MPC 都继承的核心思想。

---
## 六、Dreamer：端到端的潜空间想象

![Dreamer RSSM 在三个时间步上的潜空间动力学](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig7_dreamer_latent.png)

World Models 分别训练 V、M、C 三个模块，VAE 的优化目标是像素重建，而不是控制器真正需要的特征。**Dreamer**（Hafner 等人，ICLR 2020；DreamerV2 2021；DreamerV3 2023）则联合训练整个系统，并引入了一个关键架构：**循环状态空间模型（RSSM）**。

### 一图看懂 RSSM

这张图展示了三个时间步。每个时间步的潜状态分为两部分：

-$h_t$**（确定性）** —— GRU 的隐状态，负责长程记忆：$h_t = \mathrm{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$。  
-$z_t$**（随机性）** —— 一个小的离散类别或高斯潜变量，想象时从先验$p(z_t\mid h_t)$采样，训练时从后验$q(z_t\mid h_t, o_t)$采样。

这种分离很重要。确定性的$h$负责记住过去，随机的$z$负责建模真正的不确定性，比如 Pong 中飞出屏幕的球，或者 Minecraft 中箱子随机掉落的物品。所有预测头（reward、value、训练时还有 observation）都接在$(h_t, z_t)$上，因此解码器的损失会反向传播到动力学和表示中。

### 行为完全在想象中学

世界模型拟合好真实数据后，Dreamer 用以下方式训练 actor 和 critic：

1. 从真实 buffer 中采一批$(h_t, z_t)$作为起点；
2. 用 **prior** 动力学向前推 15 步，每步用 actor 采样动作；
3. 在想象轨迹上计算价值目标，用重参数化的策略梯度更新 actor。

这一步完全不需要真实交互。一个 batch 的真实数据可以生成成千上万次想象梯度更新——还是 Dyna 的思想，但现在跑在学习到的潜空间里。

### 结果

- **DMControl Walker：** 10 万步达到 ~900 分，而 SAC 需要 100 万步。
- **Atari：** DreamerV2 在 55 个游戏上追平 IQN/Rainbow，**而且只用单卡 GPU**。
- **Minecraft（DreamerV3，2023）：** 第一个能从零挖到钻石的算法，没有演示数据，也不需要逐任务调参。

DreamerV3 的亮点是**鲁棒性**：同一套 model-based 智能体，同一套超参数，在 150 多个任务（DMControl、Atari、Crafter、Minecraft）上击败了为单任务专门调参的 baseline。

---
## 七、MuZero：无需预测像素即可规划

World Models 和 Dreamer 的核心思想是“预测观测”。但 MuZero（Schrittwieser 等人，*Nature* 2020）发现，做**规划**时其实不需要观测数据，真正需要的是价值、策略和奖励。其他的一切都只是实现这些目标的手段。

MuZero 学习了三个运行在抽象隐藏状态上的小型网络：

- **表示网络：**$s_0 = h(o_0)$，将真实观测编码为隐藏状态。
- **动态网络：**$s_{k+1}, r_{k+1} = g(s_k, a_k)$，完全在隐藏状态空间中完成状态转移。
- **预测网络：**$p_k, v_k = f(s_k)$，输出策略 logits 和价值。

隐藏状态$s_k$不需要与真实环境中的任何事物对应，只要能支持在其上运行的蒙特卡洛树搜索（MCTS）即可。

### 训练损失

对于一条展开 $K$ 步的轨迹，MCTS 提供目标值$z^v, z^p$，实际观测到的奖励为$z^r$，损失函数定义如下：

$$\mathcal{L} = \sum_{k=0}^{K} \Big[\ell^p(p_k, z_k^p) + \ell^v(v_k, z_k^v) + \ell^r(r_k, z_k^r)\Big].$$

关键点在于，这里没有任何重建项。模型是**隐式的**——只要能让 MCTS 的目标自洽，就足够了。

### 实验结果

使用同一套算法和超参数，MuZero 达成了以下成果：

- **围棋、国际象棋、将棋：**性能追平甚至超越 AlphaZero ——**且无需提供游戏规则**。
- **Atari 57：**超越 R2D2，刷新 SOTA。
- **MuZero Reanalyse / Sampled MuZero / EfficientZero（2021）：**在 Atari 游戏中，仅用 2 小时游戏时间就达到了人类水平。

MuZero 最清晰地展示了这样一个深刻的原则：**模型只需忠实到满足下游任务需求的程度即可。**

---
## 八、大局观：样本效率

![MuJoCo HalfCheetah 上的样本效率曲线和达标步数对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig6_sample_efficiency.png)

把几种方法画在一张图上，结论就很清楚了。在 HalfCheetah 环境中，MBPO 和 Dreamer 用 **8 万到 15 万**真实步就能达到 SAC 需要 60 万步、PPO 需要 160 万步才能追上的分数。每条基于模型的方法曲线形状都差不多：开始时增长缓慢（因为模型还在学习），等想象更新开始提供有用的梯度后，曲线迅速攀升。

不过，这张图也老实交代了一个局限性：基于模型的方法不一定能**超越**无模型方法的渐近水平，但它们能更快达到相同的水平。当样本获取成本低时，无模型方法往往因为简单而占优；当样本昂贵时——这才是实际中更值得关注的情况——基于模型的方法优势非常明显。
## 九、如何选择合适的工具

| 场景                                   | 方法                              | 理由                                              |
| -------------------------------------- | --------------------------------- | ------------------------------------------------- |
| 小型离散环境，需要快速迭代             | Dyna-Q                            | 表格化方法，简单直接，效果立竿见影                |
| 连续控制，状态维度中等                 | MBPO                              | 短时 rollout 加上 SAC，样本效率提升 10 倍         |
| 真实机器人，交互成本高                 | MPC + ensemble 动力学（PETS/iCEM）| 局部规划，避免依赖长程开环控制                    |
| 像素观测，预算有限                     | Dreamer / DreamerV3               | 潜空间动力学轻松处理高维传感器数据                |
| 完全信息棋类 / 离散规划                | MuZero                            | 隐式模型结合 MCTS，无需规则也能工作                |
| 已有快速且准确的免费模拟器             | PPO / SAC                         | 不用担心模型误差问题                              |

---
## 十、开放问题

目前有三个方向特别活跃：

1. **长尾中的模型误差。** 现有的方法要么缩短时间范围（MBPO、MPC），要么把问题隐藏在潜在空间里（Dreamer）。但这些方法都无法很好地扩展到需要 1000 步信用分配且具备逼真动态特性的任务。
2. **随机性和多模态世界。** RSSM 引入了随机变量$z$，这是个进步，但要预测真正多模态的未来（比如驾驶或对话场景）仍然非常困难。
3. **面向基础智能体的世界模型。** 最近的一些工作（如 Genie、将 Sora 用作世界模型、V-JEPA）尝试把大型生成视频模型当作动力学组件。但一个预训练好的世界模型是否能像 LLM 那样跨任务迁移，仍然是一个悬而未决但意义重大的问题。

---
## 总结

Model-Based RL 是一类**用计算换样本**的方法：

- **Dyna** 提出了一个循环——混合真实交互和想象更新，分摊交互成本。
- **MBPO** 证明了**短**想象轨迹比长轨迹更有效，因为模型误差会不断累积。
- **MPC** 把模型当作单步预测器，每一步都重新规划。
- **World Models** 将动力学学习搬到压缩的隐空间中，让像素级环境变得可控。
- **Dreamer / RSSM** 联合训练表示、动力学和策略，完全在想象中学习行为。
- **MuZero** 直接放弃了重建：模型只需要在 MCTS 下保持自洽。

核心教训是：**预测的内容要和使用方式匹配**。如果任务依赖像素，就预测像素；如果规划只用$(r, v, p)$，那就只预测这些。这个原则正是让 DreamerV3、EfficientZero、TD-MPC2 等现代方法显得通用的关键。

**下一篇：**[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/)深入 **PPO 和 TRPO**——支撑工业级强化学习（从机器人操作到 ChatGPT 的 RLHF）的信任域策略梯度方法。

---
## 参考文献

- Sutton, R. S. (1990). *Integrated architectures for learning, planning, and reacting based on approximating dynamic programming*. ICML.
- Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). *When to trust your model: model-based policy optimization*. NeurIPS. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). *Deep reinforcement learning in a handful of trials using probabilistic dynamics models* (PETS). NeurIPS. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)
- Ha, D., & Schmidhuber, J. (2018). *World models*. NeurIPS. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to control: learning behaviors by latent imagination* (Dreamer). ICLR. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with discrete world models* (DreamerV2). ICLR. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- Hafner, D., et al. (2023). *Mastering diverse domains through world models* (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Schrittwieser, J., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model* (MuZero). *Nature*. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- Ye, W., Liu, S., Kurutach, T., Abbeel, P., & Gao, Y. (2021). *Mastering Atari games with limited data* (EfficientZero). NeurIPS. [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)
- Hansen, N., Wang, X., & Su, H. (2022/2024). *Temporal difference learning for model predictive control* (TD-MPC / TD-MPC2). [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
