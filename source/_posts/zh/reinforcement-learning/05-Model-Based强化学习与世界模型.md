---
title: "强化学习（五）：Model-Based强化学习与世界模型"
date: 2025-07-01 09:00:00
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
series:
  name: "强化学习"
  part: 5
  total: 12
lang: zh-CN
mathjax: true
description: "从 Dyna、MBPO 到 World Models、Dreamer 和 MuZero——学一个环境模型，让智能体在想象中规划，把样本效率提高 10-100 倍。"
disableNunjucks: true
series_order: 5
---

到目前为止介绍过的所有算法——DQN、REINFORCE、A2C、PPO、SAC——本质上都是 **Model-Free** 的：智能体把环境当成黑盒，扔动作、收奖励、更新策略，从来不去理解环境到底是怎么运作的。这条路走得通，但极其奢侈：DQN 在 Atari Pong 上要 **1000 万帧**才能精通，OpenAI Five 在 Dota 2 上自我对弈了相当于 **45000 年**的游戏时间，AlphaStar 也是按"年"来烧 StarCraft 的样本。

人显然不是这么学的。下棋的高手会在脑子里推演几步、剪掉明显的臭棋；小孩学会"悬崖危险"是靠推理，不是靠摔下去一次。两者依靠的都是对世界响应方式的内部**模型**，而且大部分认知预算花在"模型里"，而不是"世界里"。

**Model-Based RL（基于模型的强化学习，MBRL）**就是把这件事形式化：先学一个近似的动态模型$\hat{P}(s'\mid s,a)$和奖励模型$\hat{R}(s,a)$，再把它当成一个便宜的模拟器，用来规划、提升策略、估计价值。在它能 work 的任务上，回报非常可观——**真实环境样本量降低 10 到 100 倍**——这就是"机器人需要在物理世界里跑三个月"和"跑一下午就够了"之间的差距。

本文按时间和思想脉络梳理现代 MBRL：Dyna（1990）-> MBPO（2019）-> World Models（2018）-> Dreamer（2020-23）-> MuZero（2020）。每种方法背后都是一个非常锋利的想法，下面 7 张图就是一图一想法。

## 你将学到什么

- Model-Based 何时赢、何时输的精确权衡
- **Dyna-Q**：真实经验和想象经验混合学习的祖师爷
- **MBPO**：为什么"短"想象才是甜区
- **MPC**：纯粹靠规划，不学策略
- **World Models（V/M/C）**：把像素压成潜空间里的"梦"
- **Dreamer / RSSM**：端到端的潜空间想象，确定性 + 随机状态
- **MuZero**：根本不预测观察的规划

**前置知识：**[第 1-3 部分](/zh/强化学习-一-基础与核心概念/)（MDP、价值函数、策略梯度、Actor-Critic）。

---

## 一、两种范式，同一个目标

![Model-free 与 model-based 控制循环对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig1_mf_vs_mb_loops.png)

Model-Free RL 只有一个循环：*执行 -> 观察 -> 学习*。Model-Based 在中间塞进了第二个循环：*学模型 -> 在模型里规划 -> 改进策略*。一次真实交互摊销到成千上万次想象更新里，账就立刻好看了。

### 核心权衡

|              | Model-Free                            | Model-Based                                   |
| ------------ | ------------------------------------- | --------------------------------------------- |
| **学什么**   | 只学策略 / 价值函数                   | 学模型 $\hat{P},\hat{R}$ **和** 策略 / 价值   |
| **样本成本** | 高，每次梯度都要消耗一次真实交互      | 低，一次真实步可以衍生很多想象更新            |
| **算力成本** | 单步更便宜                            | 更贵（要拟合模型 + 规划）                     |
| **渐近性能** | 只受探索瓶颈限制                      | 受 **模型偏差**限制                           |
| **迁移性**   | 与训练时的奖励绑定                    | 同一个模型可以复用到新任务                    |
| **失败模式** | 学得慢                                | 误差累积 -> 在"梦里的最优"上自我催眠           |

### 现实中的样本效率

| 算法           | 类别        | 基准                        | 达到专家水平的步数   |
| -------------- | ----------- | --------------------------- | -------------------- |
| DQN            | Model-Free  | Atari Pong                  | 约 1000 万帧         |
| PPO            | Model-Free  | MuJoCo HalfCheetah          | 约 100 万-200 万步   |
| SAC            | Model-Free  | MuJoCo HalfCheetah          | 约 60 万步           |
| **MBPO**       | Model-Based | MuJoCo HalfCheetah          | **约 8 万-10 万步**  |
| **Dreamer**    | Model-Based | DMControl Walker            | **约 10 万步**       |
| **DreamerV3**  | Model-Based | Minecraft（挖钻石）         | 史上首个从零完成     |

差距大约是 **一个数量级**，模拟器越贵（真实机器人、慢物理仿真），优势越夸张。

### 什么时候上 Model-Based

合适的场景：

- 真实交互昂贵：机器人、自动驾驶、药物发现、有真人参与的对话系统。
- 动力学**可学**：物理光滑、棋类、结构化环境。
- 同一个环境上有**多个下游任务**，模型可以摊销复用。

不合适的场景：

- 已经有又快又准的免费模拟器（Atari 自己**就是**模拟器）。
- 高度随机或对抗（金融市场、社交博弈）。
- 状态空间巨大到模型没法在数据预算内拟合。

---

## 二、Dyna-Q：最早的蓝图

![Dyna-Q 数据流和收敛曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig3_dyna_q_flow.png)

Sutton 1990 年提出的 **Dyna** 是第一个把 model-based 循环讲清楚的系统。它把每次真实转移用了三遍：

1. **直接学习** —— 用真实$(s,a,r,s')$更新 Q；
2. **学模型** —— 把这条转移存进表格化模型$M(s,a)\to(r,s')$；
3. **规划** —— 从历史$(s,a)$里随机采样$n$条，查模型得到想象转移，再做$n$次 Q 更新。

右边的收敛曲线说明了后果：在确定性 GridWorld 上，规划步数从 0（普通 Q-Learning）拉到 50，收敛所需 episode 数掉了一个数量级——因为每次真实交互现在触发 51 次 Bellman 更新而不是 1 次。

### 参考实现

```python
import numpy as np


class DynaQ:
    """表格 Dyna-Q：直接学习 + 从记忆模型里规划。"""

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
        # 2. 学模型（确定性环境直接记忆即可）
        if (s, a) not in self.model:
            self.visited.append((s, a))
        self.model[(s, a)] = (r, s_next)
        # 3. 规划：回放想象转移
        for _ in range(self.planning_steps):
            sp, ap = self.visited[np.random.randint(len(self.visited))]
            rp, sp_next = self.model[(sp, ap)]
            self._q_update(sp, ap, rp, sp_next)
```

### Dyna 教会我们什么，又卡在哪里

Dyna 把核心洞察隔离了出来：**学到一个模型，等于用算力换样本**。它同时也把后续所有方法都要面对的麻烦摆上了台面：在错误的模型上做规划，会把偏差直接打进价值函数。表格化的确定性世界看不到这件事，但换成神经网络模型 + 长 horizon，误差会指数级累积。本文剩下的内容，本质上就是一连串针对这个问题的聪明回答。

---

## 三、MBPO：让你的想象力短一点

![MBPO 短分支 rollout 与模型误差随长度增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig4_mbpo_short_rollouts.png)

Janner 等人在 NeurIPS 2019 提出的 **Model-Based Policy Optimization（MBPO）**，是 Dyna 思想在连续控制上最干净的现代版本。它的标题级洞察就两个字：**短想象**。

右图说明了为什么：状态预测的累积误差大致随 rollout 长度$k$几何增长。到$k = 20$，即使是 5 个模型组成的 ensemble 也已经飘到没法用了。MBPO 的回答是：只在真实状态上**branch 1-5 步**（左图），然后把这些短想象转移交给 SAC 去做长程的信用分配——这正是 model-free 算法擅长的事情。

### 算法

1. 拿当前策略在真实环境跑一段，加进$\mathcal{D}_{\text{real}}$；
2. 在$\mathcal{D}_{\text{real}}$上拟合一个 **5 个网络**的概率动态模型 ensemble$f_\theta(s,a)\to(s',r)$；
3. 反复从$\mathcal{D}_{\text{real}}$里采初始状态，从 ensemble 里随机挑一个成员 branch$k$步，把想象转移加进$\mathcal{D}_{\text{model}}$；
4. SAC 在$\mathcal{D}_{\text{real}}$和$\mathcal{D}_{\text{model}}$的混合数据上训练。

Ensemble 是关键：成员之间的分歧既起到**正则**作用（数据稀疏的地方分歧最大），又隐式地提供**认知不确定性**让策略避开。

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

在 MuJoCo HalfCheetah 上，MBPO 用 **约 10 万**真实步达到 ~10000 的回报，而 SAC 要 ~100 万、PPO 要 ~160 万。经验上的最佳 rollout 长度是$k=1$，再长就被累积误差吞掉了。

---

## 四、纯规划：模型预测控制（MPC）

![模型预测控制：采样 -> 评分 -> 执行第一步 -> 重新规划](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig5_mpc_planning.png)

如果模型够好，其实可以连"策略"都不要，**每一步都重新从零规划一次**。**模型预测控制（MPC）**是经典控制工程里的主力，把学习到的动态模型塞进去就是现代 MBRL 里非常常用的一招。

循环：

1. 采样$N$条候选动作序列$a_{t:t+H}$（均匀、高斯，或者从 CEM/iCEM 提议分布里采）；
2. 把每条都用**学到的模型**前推$H$步，按预测回报打分；
3. **只执行**最优序列的**第一个动作**；
4. 观察真实下一状态，重新规划。

图里画了 12 条候选轨迹（灰色）、最优一条（绿色），以及那唯一被高亮、真正发给执行机构的动作。关键点是：每次只执行一步，意味着模型只需要**局部准确**——长程开环里的累积误差根本没机会爆发。

MPC 在 **错误代价很高**的场景（真实机器人、手术、自动驾驶）几乎是默认选择。它也是连接学习模型和经典规划文献的桥梁：PETS、PlaNet、TD-MPC 以及 Dreamer 的策略改进环，归结起来都是某种形式的"在学习模型里跑 MPC"。

---

## 五、World Models：在潜空间里做梦

![World Model V/M/C 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig2_world_model_vmc.png)

MBPO 之所以好用，是因为 MuJoCo 的状态只有十几到二十几维。可是要预测下一帧 Atari（$84\times 84\times 3$像素），就完全是另一回事——而且其中大多数像素（天空、计分板）跟控制毫无关系。**World Models**（Ha & Schmidhuber, 2018）给出了完全不同的形态：

> 把观察压成一个很小的潜空间编码，然后**在潜空间里学动力学**。

三个组件，左到右就是上图：

- **V（Vision）** —— 一个变分自编码器把每一帧$o_t$压到约 32 维的潜变量$z_t$。重建损失保证$z_t$里包含足够的场景信息。
- **M（Memory）** —— 一个混合密度网络 RNN 建模$P(z_{t+1}\mid z_t, a_t, h_t)$，其中$h_t$是 RNN 的隐状态。M **就是**世界模型。
- **C（Controller）** —— 一个故意做得很小的线性策略，把$(z_t, h_t)$映射成$a_t$。在 CarRacing 上它只有 **867 个参数**，而 DQN 是 170 万。

### 它为什么有效——以及为什么这很惊人

控制器可以**完全在梦里**训练：从一个采样的$z$开始 rollout M，得到伪轨迹，用 CMA-ES 进化 C，直到评估之前都不碰真实环境。这个 867 参数的控制器在 CarRacing-v0 上拿到了接近人类的分数。更深一层的教训——也是 Dreamer / DreamerV3 / TD-MPC 都继承的——是**学到一个有用的表示，本身就是问题的大半**：V 和 M 摆好之后，控制几乎是 trivial 的。

---

## 六、Dreamer：端到端的潜空间想象

![Dreamer RSSM 在三个时间步上的潜空间动力学](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig7_dreamer_latent.png)

World Models 把 V、M、C 分三阶段训练，意味着 VAE 是为像素重建优化的，而不是为控制器真正需要的特征优化的。**Dreamer**（Hafner 等人，ICLR 2020；DreamerV2 2021；DreamerV3 2023）把整个系统联合训练，并加了一块关键的架构：**循环状态空间模型（RSSM）**。

### 一图看懂 RSSM

图里画了三个时间步。每一步的潜状态都有两部分：

-$h_t$**（确定性）** —— 一个 GRU 隐状态，承担长程记忆：$h_t = \mathrm{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$。
-$z_t$**（随机）** —— 一个小的离散类别或高斯潜变量，想象时从先验$p(z_t\mid h_t)$采样，训练时从后验$q(z_t\mid h_t, o_t)$采样。

这种分离很重要。确定性的$h$负责记住过去，随机的$z$负责建模真正不确定的动力学（Pong 球飞出屏幕、Minecraft 箱子里随机的战利品）。所有 head（reward、value、训练时还有 observation）都接在$(h_t, z_t)$上——所以解码器损失会反传到动力学和表示里。

### 行为完全在想象中学

世界模型在真实数据上拟合好之后，Dreamer 用下面的方式训练 actor 和 critic：

1. 从真实 buffer 里采一批$(h_t, z_t)$作为起点；
2. 用 **prior** 动力学前推 15 步，每步用 actor 采样动作；
3. 在想象轨迹上 bootstrap 出价值目标，用 reparameterised 策略梯度更新 actor。

这一步**完全没有真实交互**。一个 batch 真实数据可以喂出成千上万次想象梯度更新——还是 Dyna 的思想，但现在跑在学习到的潜空间里。

### 结果

- **DMControl Walker：** 10 万步达到 ~900 分，SAC 要 100 万步。
- **Atari：** DreamerV2 在 55 个游戏上追平 IQN/Rainbow，**而且只要单卡 GPU**。
- **Minecraft（DreamerV3，2023）：** 第一个能从零挖到钻石的算法，没有演示数据，没有逐任务调参。

DreamerV3 真正的卖点是**鲁棒性**：同一套 model-based 智能体，同一套超参数，在 150 多个任务（DMControl、Atari、Crafter、Minecraft）上击败了为单任务专门调参的 baseline。

---

## 七、MuZero：不预测像素也能规划

World Models 和 Dreamer 这条线的共同前提是"预测观察"。MuZero（Schrittwieser 等人，*Nature* 2020）注意到：要做**规划**，你其实不需要观察——你需要的是价值、策略、奖励。其它都是手段。

MuZero 学三个小网络，全部跑在抽象的隐藏状态上：

- **表示函数：**$s_0 = h(o_0)$，把真实观察编码成隐状态。
- **动力学函数：**$s_{k+1}, r_{k+1} = g(s_k, a_k)$，纯粹在隐状态空间里转移。
- **预测函数：**$p_k, v_k = f(s_k)$，输出策略 logits 和价值。

隐状态$s_k$**不需要**对应真实环境里的任何东西，只要在它之上跑的蒙特卡洛树搜索（MCTS）能用就行。

### 损失函数

把一条轨迹展开$K$步，MCTS 给出$z^v, z^p$，真实奖励是$z^r$，损失就是：

$$
\mathcal{L} = \sum_{k=0}^{K} \Big[\ell^p(p_k, z_k^p) + \ell^v(v_k, z_k^v) + \ell^r(r_k, z_k^r)\Big].
$$

注意，**没有任何重建项**。这个模型是**隐式**的——只要让 MCTS 的目标自洽，它就够了。

### 结果

同一套算法、同一套超参数：

- **围棋、国际象棋、将棋：**追平或超过 AlphaZero ——**而且不需要被告知规则**。
- **Atari 57：**刷新 R2D2 的 SOTA。
- **MuZero Reanalyse / Sampled MuZero / EfficientZero（2021）：**在 Atari 上用 2 小时游戏时间达到人类水平。

MuZero 把一个深刻的原则演示得最干净：**模型只需要忠实到下游用法所要求的程度就够了。**

---

## 八、大局观：样本效率

![MuJoCo HalfCheetah 上的样本效率曲线和达标步数对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/05-Model-Based%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/fig6_sample_efficiency.png)

把这些方法叠在一张图上，结论就具体了。在 HalfCheetah 上，MBPO 和 Dreamer 用 **8 万-15 万**真实步达到的分数，SAC 要 60 万、PPO 要 160 万才能追上。每条 model-based 曲线都是同样的形状：先慢一段（在学模型），等想象更新开始携带有用梯度后立刻陡升。

不过图里也诚实地展示了一个限制：model-based 不一定能**超过** model-free 的渐近水平，它是**更快**地达到同一水平。样本便宜的时候，model-free 凭简洁性常常更优；样本昂贵的时候——这才是经验上更有趣的设定——model-based 优势巨大。

---

## 九、怎么选

| 场景                                   | 方法                              | 理由                                              |
| -------------------------------------- | --------------------------------- | ------------------------------------------------- |
| 小型离散环境，要快速迭代               | Dyna-Q                            | 表格化，立即就有效果                              |
| 连续控制，状态维度中等                 | MBPO                              | 短 rollout + SAC，10 倍样本效率                   |
| 真实机器人，交互成本高                 | MPC + ensemble 动态（PETS/iCEM）  | 局部规划，永不信任长程开环                        |
| 像素观察，预算有限                     | Dreamer / DreamerV3               | 潜空间动力学应付高维传感器                        |
| 完全信息棋类 / 离散规划                | MuZero                            | 隐式模型 + MCTS，连规则都不用给                   |
| 已经有又快又准的免费模拟器             | PPO / SAC                         | 没有模型误差要操心                                |

---

## 十、开放问题

三个方向目前最活跃：

1. **长程模型误差。**目前所有方法要么把 horizon 砍短（MBPO、MPC），要么把问题藏进潜空间（Dreamer）。两者都没有优雅地扩展到那种需要 1000 步信用分配 + 高保真动力学的任务。
2. **随机和多模态世界。**RSSM 的随机$z$是一步，但要预测真正多模态的未来（驾驶、对话），仍然很难。
3. **基础智能体的世界模型。**最近的工作（Genie、把 Sora 当世界模型用、V-JEPA）把大型生成式视频模型当作动力学组件；一个预训练好的世界模型能否像 LLM 那样跨任务迁移，是一个开放且重要的问题。

---

## 总结

Model-Based RL 是一族**用算力换样本**的方法：

- **Dyna** 引入了循环——真实和想象更新混合，把交互摊薄。
- **MBPO** 证明了**短**想象比长想象更好用，因为模型误差会累积。
- **MPC** 把模型当成一步预测器，每一步都重新规划。
- **World Models** 把动力学学习搬进压缩潜空间，让"像素环境"变得可处理。
- **Dreamer / RSSM** 把表示、动力学、策略联合训练，行为完全在想象中学习。
- **MuZero** 干脆把重建丢掉了：模型只要在 MCTS 下自洽就行。

贯穿始终的一课是：**你预测什么，应该跟你怎么用这个预测匹配**。像素重要就预测像素；下游只用$(r, v, p)$，那就只预测它们。这个原则正是让 DreamerV3、EfficientZero、TD-MPC2 这一波终于显得"通用"的关键。

**下一篇：**[第 6 部分](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)深入 **PPO 与 TRPO**——支撑工业级 RL（机器人操作、ChatGPT 的 RLHF）的信任域策略梯度方法。

---

## 参考文献

- Sutton, R. S. (1990). *Integrated architectures for learning, planning, and reacting based on approximating dynamic programming*. ICML.
- Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). *When to trust your model: model-based policy optimization*. NeurIPS. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). *Deep reinforcement learning in a handful of trials using probabilistic dynamics models*（PETS）. NeurIPS. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)
- Ha, D., & Schmidhuber, J. (2018). *World models*. NeurIPS. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to control: learning behaviors by latent imagination*（Dreamer）. ICLR. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with discrete world models*（DreamerV2）. ICLR. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- Hafner, D., et al. (2023). *Mastering diverse domains through world models*（DreamerV3）. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Schrittwieser, J., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model*（MuZero）. *Nature*. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- Ye, W., Liu, S., Kurutach, T., Abbeel, P., & Gao, Y. (2021). *Mastering Atari games with limited data*（EfficientZero）. NeurIPS. [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)
- Hansen, N., Wang, X., & Su, H. (2022/2024). *Temporal difference learning for model predictive control*（TD-MPC / TD-MPC2）. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

---

## 系列导航

| 部分   | 主题                                                                          |
| ------ | ----------------------------------------------------------------------------- |
| 1      | [基础与核心概念](/zh/强化学习-一-基础与核心概念/)                             |
| 2      | [Q-Learning 与深度 Q 网络](/zh/强化学习-二-Q-Learning与深度Q网络/)            |
| 3      | [Policy Gradient 与 Actor-Critic](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/) |
| 4      | [探索策略与好奇心驱动学习](/zh/强化学习-四-探索策略与好奇心驱动学习/)         |
| **5**  | **Model-Based 强化学习与世界模型（本文）**                                    |
| 6      | [PPO 与 TRPO](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)                      |
