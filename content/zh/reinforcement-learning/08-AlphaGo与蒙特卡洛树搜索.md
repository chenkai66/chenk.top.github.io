---
title: "强化学习（八）：AlphaGo 与蒙特卡洛树搜索"
date: 2025-09-05 09:00:00
tags:
  - Reinforcement Learning
  - AlphaGo
  - MCTS
  - AlphaZero
  - MuZero
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "从 MCTS 到 AlphaGo、AlphaGo Zero、AlphaZero 与 MuZero：UCT 探索-利用、自我对弈训练、在学到的世界模型里规划。附五子棋上的 AlphaZero 完整实现。"
disableNunjucks: true
series_order: 8
translationKey: "reinforcement-learning-8"
---
2016 年 3 月，AlphaGo 在首尔以 4 比 1 击败了围棋世界冠军李世石。这不仅是一场体育赛事的爆冷，更标志着人工智能领域一个长达 60 年的目标——让机器击败人类顶尖围棋选手——比大多数预测提前整整十年达成。围棋的合法局面数约为 $10^{170}$，远超可观测宇宙中的原子总数。无论多少暴力搜索都无法破解它。AlphaGo 的胜利源于一种全新思路：用深度神经网络提供“哪些着法值得尝试”的直觉，再由蒙特卡洛树搜索（MCTS）进行深思熟虑式的推演，验证并精炼这种直觉。

仅仅 18 个月后，AlphaGo Zero 仅凭游戏规则和三天的自我对弈，就从零开始学会了围棋，并以 100 比 0 的战绩彻底碾压了当年战胜李世石的版本。AlphaZero 将同一套方法推广至国际象棋和将棋，而 MuZero 更进一步，连游戏规则都不再需要预先提供。本章将完整追溯这一技术路线的演进历程——涵盖算法设计、数学原理，以及一个可实际训练的完整实现。
![强化学习（八）：AlphaGo与蒙特卡洛树搜索 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/illustration_1.png)

## 你将学到什么
- **MCTS 基础**：四阶段循环、UCT 探索与利用平衡、渐近最优性
- **AlphaGo （2016）**：三阶段训练（监督策略、强化策略、价值网络），以及 MCTS 如何融合它们
- **AlphaGo Zero （2017）**：从零开始的自我对弈、单一双头网络、完全摒弃 rollout
- **AlphaZero （2017）**：同一算法在国际象棋、将棋和围棋上的通用化
- **MuZero （2019）**：在学习得到的隐空间中进行规划，无需环境规则
- **完整代码**：五子棋版 AlphaZero——包含环境、网络、MCTS 和自我对弈循环

## 前置知识

- 深度强化学习基础（策略梯度、价值函数）——可参考[第 3 篇](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)
- 卷积神经网络
- 了解博弈树会有帮助，但并非必需

---

## 1. 蒙特卡洛树搜索

经典博弈树搜索（minimax + alpha–beta 剪枝）依赖一个评估函数和可控的分支因子。国际象棋两者兼备，而围棋则不然：其分支因子高达 250，既缺乏简洁的评估函数，也缺少有效的启发式规则。MCTS 巧妙地绕开了这些限制——它不枚举所有可能，而是通过采样，并将计算资源聚焦于树中最富潜力的部分。

![MCTS 四个阶段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig1_mcts_four_phases.png)

一次 MCTS *仿真*包含四个阶段：

1. **选择**——从根节点出发，依据某个搜索准则（如下文的 UCT）不断向下选择子节点，直到抵达一个尚有未尝试动作的节点。该节点是当前搜索树的“叶子”，但未必是游戏终局。
2. **扩展**——在该叶子节点上选取一个尚未尝试的动作，并将对应的新状态作为子节点加入树中。
3. **模拟（rollout）**——从新节点开始，快速将对局推进至终局。朴素 MCTS 使用随机策略；早期 AlphaGo 则采用一个小型网络；而 AlphaGo Zero 完全摒弃了 rollout，仅依赖价值网络进行评估。
4. **回传**——将模拟结果沿路径向上回传，更新路径上每个节点的访问次数 $N$ 和累计价值 $W$。

在固定仿真预算耗尽后（AlphaGo Zero 每步使用 800 次仿真），算法会选择根节点下访问次数最多的子动作作为最终决策——而非平均价值最高的那个。访问次数是一个更稳健的统计量，因为它已内化了搜索过程中的自我修正能力。

### 1.1 UCT：探索与利用的平衡

![UCB1 探索与利用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig2_ucb_exploration.png)

选择阶段采用的准则是 *Upper Confidence bound for Trees*（UCT），由 Kocsis 与 Szepesvári（2006）提出。在节点 $s$ 处，选择使下式最大的动作 $a$：

$$
\text{UCT}(s, a) = \underbrace{\frac{W(s, a)}{N(s, a)}}_{\text{利用}} \;+\; \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{探索}}.
$$
第一项是经验均值——胜率高的节点会被优先访问；第二项源自 Auer–Cesa-Bianchi–Fischer 的置信上界：访问次数越少，该项越大，从而鼓励搜索尝试那些较少探索的子节点。当 $N(s,a) \to \infty$ 时，探索项逐渐衰减，准则收敛至纯贪心策略。UCT 具有**渐近最优性**：在无限次仿真的极限下，访问分布会集中于最优动作。

AlphaGo 采用的是 PUCT 变体，其探索项还加权了网络提供的先验概率 $p(a\mid s)$：
$$
\text{PUCT}(s, a) = Q(s, a) \;+\; c_{\text{puct}} \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}.
$$
直观理解：先验告诉搜索“该先看哪里”，而访问次数则告诉它“何时可以停止关注”。

---

## 2. AlphaGo （2016）：网络遇上搜索

![强化学习（八）：AlphaGo与蒙特卡洛树搜索 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/08-alphago-and-mcts/illustration_2.png)

![AlphaGo 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig3_alphago_architecture.png)

最初的 AlphaGo 采用三阶段训练流程：

**第一阶段——监督策略 $p_\sigma$**。研究人员利用 KGS 平台上 3000 万个人类专家对局局面，训练了一个 13 层 CNN 来预测人类落子。其 top-1 准确率达到 57%，显著超越此前约 44% 的最佳水平。

**第二阶段——强化策略 $p_\rho$**。以 $p_\sigma$ 初始化 $p_\rho$，随后通过 REINFORCE 风格的自我对弈进行优化，对手从历史检查点中随机采样。该强化策略对监督策略的胜率达 80%。但令人意外的是，若将其直接用于 MCTS 的先验，性能反而下降——因为它过度收敛于少数几种风格，丧失了多样性，而多样性恰恰是搜索所需的有效先验。因此，正式版 AlphaGo 在 MCTS 中仍使用 $p_\sigma$ 提供先验。

**第三阶段——价值网络 $v_\theta$**。一个独立的 CNN 被训练用于回归对局胜负结果。为避免同一盘棋内高度相关的局面导致过拟合，每盘自我对弈仅采样一个局面，最终获得 3000 万组独立的（状态，结果）样本。

实战对弈时，MCTS 融合了这两个网络。叶节点的评估值是价值网络估计与快速 rollout 结果的加权平均：
$$
V(s_L) \;=\; (1 - \lambda)\, v_\theta(s_L) \;+\; \lambda\, z_L, \qquad \lambda = 0.5.
$$

为何要混合？2016 年时，价值网络虽强但尚不完美，rollout 能有效平滑其系统性误差。到 2017 年，随着网络性能大幅提升，rollout 反而引入噪声，拖累整体表现，因此 AlphaGo Zero 彻底移除了这一项。

---

## 3. AlphaGo Zero （2017）：从零开始

![AlphaGo Zero 自我对弈循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig4_zero_self_play_loop.png)

AlphaGo Zero 沿用了相同的核心思想，但设计更为简洁。它做了三项看似激进的改动：

1. **完全摒弃人类数据**。智能体从随机初始化出发，仅通过自我对弈学习。此前所有依赖人类棋谱、准确率超过 50% 的系统被全部抛弃。
2. **采用单一双头网络** $f_\theta(s) = (\mathbf{p}, v)$。一个共享的残差网络塔输出两个头：策略头给出动作概率分布，价值头输出标量胜率，取代了 AlphaGo 中分离的策略与价值网络。
3. **彻底取消 rollout**。叶节点仅由价值头评估，不再进行任何快速模拟。

其训练流程构成一个紧密闭环（见图示）：

1. **自我对弈**。当前最优网络使用 MCTS（每步 800 次仿真）与自身对弈，生成轨迹 $(s_t, \boldsymbol{\pi}_t, z_T)$，其中 $\boldsymbol{\pi}_t$ 是 MCTS 的访问次数分布（比原始网络策略更锐利、更可靠），$z_T \in \{-1, +1\}$ 是从当前行棋方视角记录的终局结果。
2. **训练**。通过最小化损失函数更新参数 $\theta$：
$$
\mathcal{L}(\theta) \;=\; (z - v)^2 \; - \; \boldsymbol{\pi}^\top \log \mathbf{p} \; + \; c\,\|\theta\|^2,
$$
包含价值的均方误差、策略的交叉熵损失及权重衰减。
3. **评估**。新网络挑战当前最优网络，仅当在 400 局比赛中胜率超过 55% 时，才接替成为新的自我对弈生成器。

这套设计的精妙之处在于*标签本身*。MCTS 生成的访问分布 $\boldsymbol{\pi}$ *严格优于*生成它的原始网络策略——搜索过程已对其先验进行了精炼。训练策略头 $\mathbf{p}$ 去拟合 $\boldsymbol{\pi}$，相当于将搜索带来的提升“蒸馏”回网络。这是一种策略迭代，其中策略改进步骤由 MCTS 自身完成。每一代新网络通过自我对弈产生的训练目标，都略强于其自身能力。整个过程无需外部监督，因为对手与学习者同步进化，自然形成了自动课程学习机制。

仅用 4 块 TPU 训练 **3 天**，AlphaGo Zero 便以 100–0 击败了曾战胜李世石的 AlphaGo。训练 40 天后，它又超越了曾击败柯洁的 AlphaGo Master。

## 4. AlphaZero 与 MuZero

![算法演进时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig5_evolution_timeline.png)

**AlphaZero**（2017 年 12 月）证明了 AlphaGo Zero 的算法并非围棋专属。只需替换游戏特定的状态编码，并做少量调整（如取消代际胜率门槛、允许和棋——国际象棋中常见），同一套代码在 TPU 上训练约 **9 小时**后，便在国际象棋中以 28 胜、0 负、72 和的成绩击败 Stockfish 8，在将棋中战胜 Elmo，并在围棋上超越 AlphaGo Zero。

**MuZero**（2019 年 11 月）更进一步：它甚至不需要预知游戏规则。MuZero 联合学习三个函数：

- **表征函数** $h_\theta : o_{\le t} \mapsto s_t^0$ —— 将观测历史编码为初始隐状态。
- **动力学函数** $g_\theta : (s_t^k, a_{t+k}) \mapsto (s_t^{k+1}, r_t^{k+1})$ —— 预测下一隐状态与即时奖励。
- **预测函数** $f_\theta : s_t^k \mapsto (\mathbf{p}_t^k, v_t^k)$ —— 从隐状态输出策略与价值。

MCTS 完全在**隐空间**中展开。搜索内部不再调用环境模拟器，仅依赖学习到的动力学函数。隐状态无需重建原始观测，只需足以预测奖励、价值和策略即可。得益于这一更宽松的目标，MuZero 在棋类游戏中媲美 AlphaZero，在 Atari 游戏上则超越了 R2D2、Ape-X 等无模型方法——而 Atari 正是缺乏规则模拟器的典型场景。

### 4.1 Elo 随时间的变化

![Elo 演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig6_elo_progression.png)

左图对比了各代系统的峰值 Elo 分数。右图展示了 AlphaGo Zero 的训练轨迹：3 天超越李世石版 AlphaGo，约 21 天超越 AlphaGo Master，最终在 5200 Elo 左右趋于饱和。作为参照，人类九段职业棋手的 Elo 通常在 3500–3700 区间。

### 4.2 搜索到底有多大作用？

![搜索预算与棋力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig7_search_vs_strength.png)

搜索与网络相辅相成：网络提供先验，搜索加以精炼。左图显示，MCTS 仿真次数每翻一倍，Elo 提升幅度大致恒定（呈对数关系），即便在 12800 次仿真下仍未见平台期。右图则揭示了神经先验的乘数效应：纯随机 rollout 的朴素 MCTS 很早就陷入停滞，而由网络引导的 MCTS 则能持续进步。二者缺一不可，单独使用任一组件都无法达到竞技水平。

## 5. 完整实现：五子棋上的 AlphaZero

9×9 的五子棋（“五子连珠”）是一个理想的试验场：规则仅需约 30 行代码即可实现，分支因子约为 60，且在单张消费级 GPU 上经过数千局自我对弈后，即可展现出可观的棋力。

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class GomokuEnv:
    def __init__(self, size=9):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.get_state()

    def get_state(self):
        # 三通道状态：己方棋子、对方棋子、当前出手方
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0] = (self.board == self.current_player)
        state[1] = (self.board == -self.current_player)
        state[2] = self.current_player
        return state

    def legal_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        self.board[action] = self.current_player
        if self._check_win(action):
            self.done = True
            self.winner = self.current_player
            return self.get_state(), self.winner, True
        if len(self.legal_actions()) == 0:
            self.done = True
            return self.get_state(), 0, True
        self.current_player *= -1
        return self.get_state(), 0, False

    def _check_win(self, last_move):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.board[last_move]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = last_move[0] + sign * dr, last_move[1] + sign * dc
                while (0 <= r < self.size and 0 <= c < self.size
                       and self.board[r, c] == player):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 5:
                return True
        return False

class PolicyValueNet(nn.Module):
    """简单的双头网络：共享 CNN 主干 + 策略头 + 价值头"""
    def __init__(self, board_size=9, channels=128):
        super().__init__()
        self.size = board_size
        self.shared = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size ** 2, board_size ** 2),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        shared = self.shared(x)
        # log-softmax 配合 MCTS 目标做交叉熵更稳定
        policy = torch.log_softmax(self.policy_head(shared), dim=1)
        value = self.value_head(shared)
        return policy, value

class MCTSNode:
    def __init__(self, prior, parent=None):
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

    def select_child(self, c_puct=1.0):
        # PUCT: Q + c * P * sqrt(parent N) / (1 + child N)
        best_score, best_action, best_child = -float('inf'), None, None
        for action, child in self.children.items():
            u = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.value() + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(prior, parent=self)

    def backup(self, value):
        # value 是当前节点出手方视角下的价值
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # 上一层是对手视角，符号要翻转
            self.parent.backup(-value)

class MCTS:
    def __init__(self, model, num_simulations=400, c_puct=1.0):
        self.model = model
        self.num_sims = num_simulations
        self.c_puct = c_puct

    @torch.no_grad()
    def search(self, env):
        root = MCTSNode(prior=0)
        # 用网络在合法着法上的概率初始化根节点
        state = env.get_state()
        legal = env.legal_actions()
        log_probs, _ = self.model(torch.FloatTensor(state).unsqueeze(0))
        probs = torch.exp(log_probs).squeeze().numpy()
        action_probs = np.array([probs[a[0] * env.size + a[1]] for a in legal])
        action_probs /= action_probs.sum()
        root.expand(legal, action_probs)

        for _ in range(self.num_sims):
            node = root
            env_copy = self._copy_env(env)
            path = [node]

            # 选择阶段：一直往下走，直到遇到未扩展节点或终局
            while node.children and not env_copy.done:
                action, node = node.select_child(self.c_puct)
                path.append(node)
                env_copy.step(action)

            # 扩展 + 评估
            if not env_copy.done:
                state = env_copy.get_state()
                legal = env_copy.legal_actions()
                log_probs, value = self.model(
                    torch.FloatTensor(state).unsqueeze(0))
                probs = torch.exp(log_probs).squeeze().numpy()
                ap = np.array([probs[a[0] * env_copy.size + a[1]] for a in legal])
                ap /= ap.sum()
                node.expand(legal, ap)
                v = value.item()
            else:
                # 终局：用真实结果作为叶节点价值（出手方视角）
                v = env_copy.winner * env_copy.current_player

            # 沿路径回传，每层翻一次符号
            for n in reversed(path):
                n.backup(v)
                v = -v

        # 返回访问次数归一化的分布——MCTS 改进过的策略
        visits = np.zeros(env.size ** 2)
        for action, child in root.children.items():
            visits[action[0] * env.size + action[1]] = child.visit_count
        return visits / visits.sum()

    def _copy_env(self, env):
        new = GomokuEnv(env.size)
        new.board = env.board.copy()
        new.current_player = env.current_player
        new.done = env.done
        new.winner = env.winner
        return new
```

训练循环十分简洁：持续生成自我对弈对局，存储 $(s_t, \boldsymbol{\pi}_t, z_T)$ 三元组，并使用 AlphaGo Zero 的损失函数训练网络。在 9×9 棋盘上，每步执行 400 次仿真，单卡训练约 50 轮自我对弈迭代后，网络便能稳定击败随机策略与贪心启发式。两条实用建议：(i) 在自我对弈时，向根节点先验注入 Dirichlet 噪声以维持探索；(ii) 对访问分布使用*温度*参数——前约 10 步设为温度 1，之后趋近于贪心，以避免收集的数据过度集中于单一确定性走法。

---

## 常见问题

### 为什么 AlphaGo Zero 不需要 rollout？

到 2017 年，更深的残差网络、更丰富的自我对弈数据，以及统一的策略-价值双头架构，使得纯价值网络的评估精度已超过“网络估计 + 随机 rollout”的混合方案。DeepMind 的消融实验明确证实：纯价值评估效果更佳，因此 rollout 被果断舍弃。

### 自我对弈会不会陷入退化的均衡？

在两人零和、完美信息博弈中，*虚拟自我对弈*（fictitious play, Brown 1951; Heinrich & Silver 2016）可收敛至纳什均衡。MCTS 引入的乐观探索机制进一步防止了过早陷入单一模式。但在非完美信息博弈（如扑克）或合作博弈中，这一性质无法保证，此时需借助对手种群（如 PSRO 或 AlphaStar 的“联赛”机制）。

### 为什么用访问次数分布作为策略目标，而不是经验 $Q$？

访问次数更稳健：一个子节点若访问极少，其 $Q$ 值可能受噪声干扰严重；但若搜索未持续选择它，其访问次数也不可能高。此外，对 $\boldsymbol{\pi}$ 使用交叉熵损失，即使对极少被尝试的动作也能提供有意义的梯度，而硬性 argmax 目标则无法做到这一点。

### MCTS 能处理连续动作空间吗？

不能直接处理——UCT 和 PUCT 均假设动作集有限。扩展方法如 *Progressive Widening* 会在节点访问增多时逐步采样新动作加入。近年工作（如 Sampled MuZero, 2021）已能处理连续及结构化动作空间。但在纯连续控制任务中，SAC、PPO 等无模型方法仍更为实用。

### 为什么每步要跑 800 次仿真？只用 1 次可以吗？

理论上可行，但训练会停滞。若每步仅 1 次仿真，访问分布就等同于网络策略本身，无法提供任何改进信号。而 800 次仿真下，搜索目标显著优于网络当前能力——正是这一差距驱动了学习。仿真次数超过数千后收益递减；AlphaZero 在围棋和国际象棋中均采用 800 次，MuZero 也沿用了该设定。

## 参考文献

- Silver et al., **Mastering the game of Go with deep neural networks and tree search**, *Nature* 529, 2016.
- Silver et al., **Mastering the game of Go without human knowledge**, *Nature* 550, 2017 (AlphaGo Zero).
- Silver et al., **A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play**, *Science* 362, 2018 (AlphaZero).
- Schrittwieser et al., **Mastering Atari, Go, chess and shogi by planning with a learned model**, *Nature* 588, 2020 (MuZero).
- Kocsis & Szepesvári, **Bandit based Monte-Carlo Planning**, *ECML* 2006 (UCT).
- Auer, Cesa-Bianchi & Fischer, **Finite-time Analysis of the Multiarmed Bandit Problem**, *Machine Learning* 47, 2002 (UCB1).
- Browne et al., **A Survey of Monte Carlo Tree Search Methods**, *IEEE TCIAIG* 4, 2012.
