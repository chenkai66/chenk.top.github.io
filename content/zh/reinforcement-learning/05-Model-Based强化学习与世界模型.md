---
title: "强化学习（五）：Model-Based 强化学习与世界模型"
date: 2025-08-21 09:00:00
tags:
  - Reinforcement Learning
  - Model-Based RL
  - World Models
  - Dyna
  - MBPO
  - Dreamer
  - MuZero
  - PlaNet
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "从 Dyna、MBPO 到 World Models、Dreamer 和 MuZero——学一个环境模型，让智能体在想象中规划，把样本效率提高 10-100 倍。"
disableNunjucks: true
series_order: 5
series_total: 12
translationKey: "reinforcement-learning-5"
---
到目前为止，我们介绍的所有算法——DQN、REINFORCE、A2C、PPO、SAC——都属于 **Model-Free**（无模型）类型。智能体将环境视为黑盒，不断尝试动作并根据返回的奖励更新策略，完全不关心环境内部如何运作。这种方法确实有效，但代价高昂：DQN 需要大约 **1000 万帧**才能掌握 Atari Pong；OpenAI Five 在 Dota 2 上的训练量相当于 **约 4.5 万年**的自我对弈；AlphaStar 则消耗了数年的 StarCraft 对局数据来训练单个智能体。

人类显然不是这样学习的。棋手会向前推演几步，主动排除明显失误；小孩只需一次观察或推理就能明白“悬崖危险”，而不需要真的摔下去。两者都依赖一个内在的 **模型**——即对世界如何响应动作的预测机制，并且大部分认知资源都花在**模型内部**的推理上，而非真实世界中反复试错。

**Model-Based RL（基于模型的强化学习，MBRL）** 正是将这一思想形式化：先学习一个近似的动态模型 $\hat{P}(s'\mid s, a)$ 和奖励模型 $\hat{R}(s, a)$，再将它们作为廉价的模拟器，用于规划、策略改进或价值估计。在适用的任务上，这种方法能将真实环境交互所需的样本量减少 **10 到 100 倍**——这意味着机器人的物理训练时间可以从三个月缩短到一个下午。

本文梳理了 MBRL 的现代发展脉络：Dyna（1990）→ MBPO（2019）→ World Models（2018）→ Dreamer（2020–23）→ MuZero（2020）。每种方法都围绕一个核心洞见展开，文中的七幅图将逐一可视化这些思想。
![强化学习（五）：Model-Based 强化学习与世界模型 — 章节概览图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/illustration_1.png)


---

## 你将学到什么

- Model-Based RL 成功或失败背后的精确权衡
- **Dyna-Q**：首个融合真实经验与想象经验进行更新的经典框架
- **MBPO**：为何短时程的“想象”才是最佳选择
- **MPC**：仅依赖学习模型的纯规划循环
- **World Models（V/M/C）**：将像素压缩进一个低维的“梦境”潜空间
- **Dreamer / RSSM**：端到端的潜空间想象，结合循环结构与随机状态
- **MuZero**：无需预测观测即可完成规划

**前置知识：**[第 1–3 部分](/zh/reinforcement-learning/01-基础与核心概念/)（MDP、价值函数、策略梯度、Actor-Critic）。

---

## 一、两种范式，一个目标

![Model-free 与 model-based 控制循环对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig1_mf_vs_mb_loops.png)

在无模型强化学习中，唯一的学习循环是 *执行 → 观察 → 学习*。而在基于模型的方法中，我们插入了第二个循环：*学习模型 → 在模型内规划 → 改进策略*。一次真实交互现在可以支撑成千上万次“想象”中的更新，大幅摊薄样本成本。

### 核心权衡

|                    | Model-Free                              | Model-Based                                          |
| ------------------ | --------------------------------------- | ---------------------------------------------------- |
| **学什么**         | 仅策略 / 价值函数                       | 模型 $\hat{P}, \hat{R}$ **和** 策略 / 价值函数        |
| **样本成本**       | 高 —— 每次梯度更新需一次真实交互        | 低 —— 一次真实步可生成多次想象更新                   |
| **计算成本**       | 单步较低                                | 较高（需拟合模型 + 规划）                            |
| **渐近性能上限**   | 仅受探索能力限制                        | 受**模型偏差**限制                                   |
| **任务迁移性**     | 绑定于训练时的奖励函数                  | 同一模型可用于多个新任务                             |
| **典型失败模式**   | 学习缓慢                                | 模型误差累积 → 产生幻觉式最优解                      |

### 实际中的样本效率

| 算法           | 类别        | 基准测试                | 达到专家水平所需步数 |
| -------------- | ----------- | ----------------------- | -------------------- |
| DQN            | Model-Free  | Atari Pong              | ~1000 万帧           |
| PPO            | Model-Free  | MuJoCo HalfCheetah      | ~100–200 万步        |
| SAC            | Model-Free  | MuJoCo HalfCheetah      | ~60 万步             |
| **MBPO**       | Model-Based | MuJoCo HalfCheetah      | **~8–10 万步**       |
| **Dreamer**    | Model-Based | DMControl Walker        | **~10 万步**         |
| **DreamerV3**  | Model-Based | Minecraft（挖钻石）     | 首个从零完成的算法   |

差距约为 **一个数量级**，在连续控制任务中尤为显著；当模拟器本身昂贵（如真实机器人或慢速物理仿真）时，优势更为突出。

### 何时选用 Model-Based 方法？

**适合场景：**

- 真实交互成本高：机器人、自动驾驶、药物发现、含用户反馈的对话系统。
- 动力学规律**可学习**：平滑物理、棋类游戏、结构化环境。
- 面临**多个下游任务**，模型可在任务间复用。

**不适合场景：**

- 已有免费、快速且高保真的模拟器（Atari 本身就是模拟器）。
- 动力学高度随机或对抗性强（如金融市场、社交互动）。
- 状态空间维度极高，模型无法在有限数据下有效拟合。

---

## 二、Dyna-Q：最初的蓝图

![Dyna-Q 数据流和收敛曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig3_dyna_q_flow.png)

Sutton 于 1990 年提出的 **Dyna** 是首个清晰阐述 Model-Based 循环的系统。每次真实的状态转移会被使用三次：

1. **直接学习** —— 用真实的 $(s, a, r, s')$ 更新 Q 值；
2. **模型学习** —— 将转移存入表格模型 $M(s, a) \to (r, s')$；
3. **规划** —— 随机采样 $n$ 个曾见过的 $(s, a)$ 对，查询模型，并基于这些“想象”的转移额外执行 $n$ 次 Q 更新。

右侧的收敛曲线展示了其效果：在确定性 GridWorld 中，将规划步数 $n$ 从 0（即普通 Q-Learning）增至 50，收敛所需的 episode 数量骤降一个数量级——因为每次真实交互现在触发了 51 次 Bellman 更新，而非仅 1 次。

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

Dyna 揭示了核心思想：**用计算换样本**。但它也暴露了一个所有现代方法都必须面对的问题：在错误模型上规划会直接将偏差注入价值函数。在表格化、确定性环境中，这几乎不可见；但一旦使用神经网络建模并在长时程任务中 rollout，误差会指数级累积。本文后续内容本质上就是对这一问题的一系列巧妙回应。

---

## 三、MBPO：让想象短一些

![MBPO 短分支 rollout 与模型误差随长度增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig4_mbpo_short_rollouts.png)

**Model-Based Policy Optimization（MBPO）**（Janner et al., NeurIPS 2019）是 Dyna 思想在连续控制中最简洁的现代实现。其核心洞见只有两个字：**短 rollout**。

右图清晰展示了问题所在：状态预测的累积误差随 rollout 长度 $k$ 近似几何增长。当 $k = 20$ 时，即使使用由 5 个动力学模型组成的集成（ensemble），预测结果也已严重偏离，无法用于信用分配。MBPO 的解决方案是：仅从真实状态出发**扩展 1–5 步**（左图），然后将生成的转移交给 SAC 处理——这正是无模型方法擅长的长时程信用分配。

### 算法流程

1. 用当前策略在真实环境中采样，存入 $\mathcal{D}_{\text{real}}$。
2. 在 $\mathcal{D}_{\text{real}}$ 上拟合一个由 **5 个**概率动力学模型组成的集成 $f_\theta(s, a) \to (s', r)$。
3. 反复从 $\mathcal{D}_{\text{real}}$ 中采样初始状态，在随机选择的集成成员中进行 $k$ 步 rollout，并将想象出的转移加入 $\mathcal{D}_{\text{model}}$。
4. 在 $\mathcal{D}_{\text{real}}$ 与 $\mathcal{D}_{\text{model}}$ 的混合数据上训练 SAC。

集成至关重要：成员间的分歧不仅起到**正则化**作用（在数据稀疏区域分歧最大），还隐式提供了**认知不确定性**，使策略能自动避开不可靠区域。

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

### 实验结果

在 MuJoCo HalfCheetah 上，MBPO 仅用 **~10 万步**真实交互就达到约 10,000 的回报，而 SAC 需 ~100 万步，PPO 需 ~160 万步。实验发现，大多数任务的最优 rollout 长度为 **$k=1$**；更长的 rollout 反而有害，因为累积误差迅速压倒了额外的信用分配深度。

---

## 四、纯规划：模型预测控制（MPC）

![模型预测控制：采样 -> 评分 -> 执行第一步 -> 重新规划](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig5_mpc_planning.png)

如果模型足够可靠，甚至可以完全跳过“策略”学习，**每一步都从头规划**。**模型预测控制（MPC）** 是经典控制工程的主力方法，而学习到的动力学模型可直接嵌入其中。

其循环如下：

1. 采样 $N$ 条候选动作序列 $a_{t:t+H}$（均匀、高斯，或来自 CEM/iCEM 提议分布）。
2. 在**学习到的模型**中将每条序列前推 $H$ 步，并按预测回报打分。
3. **仅执行最优序列的第一个动作**。
4. 观测真实下一状态，并重新规划。

图中展示了 12 条候选轨迹（灰色）、最优轨迹（绿色），以及实际发送给执行器的那个高亮动作。关键在于：每次只执行一步，意味着模型只需**局部准确**——累积误差根本没有机会破坏长时程开环计划。

MPC 是**高风险场景**（如真实机器人、手术、自动驾驶）的首选。它也是连接学习模型与经典规划文献的桥梁：PETS、PlaNet、TD-MPC 和 Dreamer 的策略改进循环，本质上都是“在学习模型中运行 MPC”的变体。

---

## 五、World Models：在潜空间里做梦

![强化学习（五）：Model-Based 强化学习与世界模型 — 章节小结图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/05-model-based-rl-and-world-models/illustration_2.png)


![World Model V/M/C 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig2_world_model_vmc.png)

MBPO 之所以有效，是因为 MuJoCo 状态仅有 11–23 维。但要预测下一帧 $84 \times 84 \times 3$ 的 Atari 图像则困难得多——而且大部分像素（如天空、计分板）与控制无关。**World Models**（Ha & Schmidhuber, 2018）提出了另一种思路：

> 将观测压缩为低维潜码，然后直接在该潜空间中学习动力学。

如上图从左至右所示，包含三个组件：

- **V（Vision）** —— 变分自编码器（VAE）将每帧 $o_t$ 映射为约 32 维潜变量 $z_t$，重建损失迫使 $z_t$ 保留足够场景信息。
- **M（Memory）** —— 混合密度网络 RNN 建模 $P(z_{t+1} \mid z_t, a_t, h_t)$，其中 $h_t$ 为 RNN 隐状态。M 本身即为世界模型。
- **C（Controller）** —— 一个刻意设计得极小的线性策略，将 $(z_t, h_t)$ 映射为 $a_t$。在 CarRacing 上，它仅有 **867 个参数**，远少于 DQN 的 170 万。

### 为何有效——以及为何令人惊讶

控制器可**完全在“梦境”中训练**：从采样的 $z$ 开始，用 M 生成伪轨迹，通过 CMA-ES 优化 C，直到评估阶段才接触真实环境。这个仅 867 参数的控制器在 CarRacing-v0 上达到了接近人类的水平。更深层的启示是：**学习有用表示已解决大半问题**——一旦 V 和 M 就位，控制几乎变得微不足道。这一思想被 Dreamer / DreamerV3 / TD-MPC 全面继承。

---

## 六、Dreamer：端到端的潜空间想象

![Dreamer RSSM 在三个时间步上的潜空间动力学](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig7_dreamer_latent.png)

World Models 分阶段训练 V、M、C，导致 VAE 优化目标是像素重建，而非控制器真正需要的信息。**Dreamer**（Hafner et al., ICLR 2020；DreamerV2 2021；DreamerV3 2023）则端到端联合训练整个系统，并引入关键架构：**循环状态空间模型（RSSM）**。

### RSSM 一图解

图中展示三个时间步。每个时刻的潜状态分为两部分：

- $h_t$ **（确定性）** —— GRU 隐状态，承载长程记忆：$h_t = \mathrm{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$。
- $z_t$ **（随机性）** —— 小型离散或高斯潜变量，想象时从先验 $p(z_t \mid h_t)$ 采样，训练时从后验 $q(z_t \mid h_t, o_t)$ 采样。

这种分离至关重要：确定性 $h$ 负责记忆，随机性 $z$ 建模真实不确定性（如 Pong 中飞出屏幕的球，或 Minecraft 中箱子的随机掉落物）。接在 $(h_t, z_t)$ 上的预测头（reward、value，训练时还有 observation）使解码损失能反向传播至动力学与表示模块。

### 行为学习完全在想象中进行

世界模型拟合真实数据后，Dreamer 按以下方式训练 actor 与 critic：

1. 从真实 buffer 中采样一批 $(h_t, z_t)$ 作为起点；
2. 使用 **prior** 动力学前推 15 步，每步由 actor 采样动作；
3. 在想象轨迹上构建价值目标，并通过重参数化策略梯度更新 actor。

此过程无需任何真实交互。一个真实 batch 可驱动成千上万次想象梯度更新——仍是 Dyna 的思想，但如今运行在学习到的潜空间中。

### 实验结果

- **DMControl Walker**：10 万步达 ~900 分，SAC 需 ~100 万步。
- **Atari**：DreamerV2 在 55 游戏套件上媲美 IQN/Rainbow，**且仅用单 GPU**。
- **Minecraft（DreamerV3, 2023）**：首个从零挖到钻石的算法，无需演示，也无需逐任务调参。

DreamerV3 的亮点在于**鲁棒性**：同一套基于模型的智能体、同一组超参数，在超过 150 个任务（涵盖 DMControl、Atari、Crafter、Minecraft）上击败了专为单任务调优的基线。

---

## 七、MuZero：无需预测像素即可规划

World Models 与 Dreamer 的主线是“预测观测”。但 MuZero（Schrittwieser et al., *Nature* 2020）指出：对于**规划**而言，你其实不需要观测——只需要价值、策略和奖励。其余一切只是达成此目标的手段。

MuZero 学习三个在抽象隐状态上操作的小型网络：

- **表示网络**：$s_0 = h(o_0)$ —— 将真实观测编码为隐状态。
- **动力学网络**：$s_{k+1}, r_{k+1} = g(s_k, a_k)$ —— 完全在隐空间中转移。
- **预测网络**：$p_k, v_k = f(s_k)$ —— 输出策略 logits 与价值。

隐状态 $s_k$ 无需对应真实世界的任何实体，只要能支持其上的蒙特卡洛树搜索（MCTS）即可。

### 训练损失

对一条展开 $K$ 步的轨迹，设 MCTS 目标为 $z^v, z^p$，观测奖励为 $z^r$，损失为：
$$
\mathcal{L} = \sum_{k=0}^{K} \Big[ \ell^p(p_k, z_k^p) + \ell^v(v_k, z_k^v) + \ell^r(r_k, z_k^r) \Big].
$$
关键在于：**从未出现重建项**。模型是**隐式的**——只要能使 MCTS 目标自洽即可。

### 实验结果

单一算法、统一超参数即实现：

- **围棋、国际象棋、将棋**：媲美或超越 AlphaZero —— **且无需提供游戏规则**。
- **Atari 57**：刷新 R2D2 的 SOTA。
- **MuZero Reanalyse / Sampled MuZero / EfficientZero（2021）**：仅用 2 小时游戏时间即达人类水平。

MuZero 最清晰地诠释了一个深刻原则：**你的模型只需忠实到满足下游任务需求的程度**。

---

## 八、大局观：样本效率

![MuJoCo HalfCheetah 上的样本效率曲线和达标步数对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based强化学习与世界模型/fig6_sample_efficiency.png)

将各方法置于同一图表，核心结论一目了然。在 HalfCheetah 上，MBPO 与 Dreamer 仅用 **8–15 万**真实步就达到了 SAC（~60 万步）和 PPO（~160 万步）的水平。所有基于模型的方法曲线形状相似：初期增长缓慢（模型尚在学习），一旦想象更新开始提供有效梯度，性能便迅速攀升。

但该图也坦诚揭示了一个局限：基于模型的方法通常**无法超越**无模型方法的渐近性能上限，只是更快抵达同一水平。当样本廉价时，无模型方法因简单而胜出；但当样本昂贵时——这正是实践中更值得关注的情形——基于模型的方法优势巨大。

---

## 九、如何选择合适的工具

| 场景                                   | 方法                              | 理由                                              |
| -------------------------------------- | --------------------------------- | ------------------------------------------------- |
| 小型离散环境，需快速迭代               | Dyna-Q                            | 表格化、绝对正确、见效快                          |
| 连续控制，状态维度中等                 | MBPO                              | 短 rollout + SAC，样本效率提升 10 倍              |
| 真实机器人，交互成本高                 | MPC + ensemble 动力学（PETS/iCEM）| 局部规划，绝不信任长时程开环                      |
| 像素观测，预算有限                     | Dreamer / DreamerV3               | 潜空间动力学轻松应对高维传感器                    |
| 完全信息棋类 / 离散规划                | MuZero                            | 隐式模型 + MCTS，无需规则                         |
| 已有免费、快速、高保真模拟器           | PPO / SAC                         | 无需担忧模型误差                                  |

---

## 十、开放问题

当前有三个前沿方向尤为活跃：

1. **长尾任务中的模型误差**。现有方法要么限制时程（MBPO、MPC），要么将问题藏于潜空间（Dreamer），均难以优雅扩展至需 1000 步信用分配且具逼真动力学的任务。
2. **随机性与多模态世界**。RSSM 的随机 $z$ 是进步，但预测真正多模态未来（如驾驶、对话）仍极具挑战。
3. **面向基础智能体的世界模型**。近期工作（Genie、Sora-as-world-model、V-JEPA）尝试将大型生成视频模型作为动力学组件；一个预训练世界模型能否像 LLM 那样跨任务迁移，仍是悬而未决但影响深远的问题。

---

## 总结

Model-Based RL 是一类**用计算换样本**的方法：

- **Dyna** 引入循环——混合真实与想象更新，摊薄交互成本。
- **MBPO** 证明**短**想象轨迹优于长轨迹，因模型误差会累积。
- **MPC** 将模型视为单步预测器，每步重新规划。
- **World Models** 将动力学移至压缩潜空间，使像素环境变得可控。
- **Dreamer / RSSM** 联合训练表示、动力学与策略，行为完全在想象中学习。
- **MuZero** 彻底放弃重建：模型只需在 MCTS 下自洽。

统一教训是：**预测内容应匹配使用方式**。若任务依赖像素，就预测像素；若规划仅需 $(r, v, p)$，那就只预测这些。正是这一原则，使 DreamerV3、EfficientZero、TD-MPC2 等现代方法展现出真正的通用性。

**下一篇：**[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化)深入 **PPO 和 TRPO**——这些信任域策略梯度方法默默支撑着工业级 RL，从机器人操控到 ChatGPT 的 RLHF。

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
