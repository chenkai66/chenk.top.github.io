---
title: "强化学习（八）：AlphaGo与蒙特卡洛树搜索"
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
2016 年 3 月，AlphaGo 在首尔以 4 比 1 击败了围棋世界冠军李世石。这不仅是体育界的一次震撼事件，更是人工智能领域一个长达 60 年的课题——"让机器击败人类顶尖围棋选手"——提前十年画上了句号。围棋的合法局面数量大约是 $10^{170}$，比可观测宇宙中的原子总数还多。靠暴力搜索根本不可能破解。AlphaGo 的胜利源于一种全新的思路：用深度网络提供"哪些着法可能有戏"的*直觉*，再用蒙特卡洛树搜索（MCTS）进行*推演*，验证并优化这种直觉。

仅仅 18 个月后，AlphaGo Zero 完全不依赖任何人类棋谱，仅凭规则和三天的自我对弈，就以 100 比 0 的战绩碾压了当年战胜李世石的版本。AlphaZero 将同样的方法推广到了国际象棋和将棋。MuZero 更进一步，连规则都不需要输入。这一章完整追溯了这条技术路线的演变过程——从算法到数学原理，再到一份可以直接训练的实现代码。
![强化学习（八）：AlphaGo与蒙特卡洛树搜索 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-alphago-and-mcts/illustration_1.png)

## 你将学到什么

- **MCTS 基础**：四阶段循环、UCT 探索与利用平衡、渐近最优性
- **AlphaGo（2016）**：三阶段训练，包括监督策略、强化策略和价值网络，以及 MCTS 如何融合它们
- **AlphaGo Zero（2017）**：从零开始自我对弈，使用单一双头网络，完全去掉 rollout
- **AlphaZero（2017）**：同一算法扩展到国际象棋、将棋和围棋
- **MuZero（2019）**：在*学习得到的*隐空间中进行规划，无需依赖环境规则
- **完整代码**：五子棋版 AlphaZero——包含环境、网络、MCTS 和自我对弈循环
## 前置知识

- 深度强化学习基础（策略梯度、价值函数）——可以参考[第 3 篇](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)
- 卷积神经网络
- 了解博弈树会有帮助，但不是必须的

---
## 1. 蒙特卡洛树搜索

经典博弈树搜索（minimax + alpha-beta 剪枝）需要两样东西：一个评估函数和一个可控的分支因子。国际象棋两者都有，围棋却一样都没有——分支因子高达 250，没有简洁的评估函数，也没有有效的启发式方法。MCTS 绕开了这些问题：它用*采样*代替枚举，把计算资源集中在树中最有潜力的部分。

![MCTS 四个阶段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig1_mcts_four_phases.png)

一次 MCTS *仿真*包含四个步骤：

1. **选择**——从根节点开始，按照某个搜索准则（比如 UCT）一路向下选择子节点，直到遇到一个还有未尝试动作的节点。这是当前树中的叶子节点，但不一定是游戏的终局。
2. **扩展**——在叶子节点上挑一个未尝试的动作，将对应的状态作为新子节点加入树中。
3. **模拟**——从新子节点出发，快速推进到终局。朴素的 MCTS 使用随机策略，早期 AlphaGo 用一个小网络，而 AlphaGo Zero 完全用价值网络替代了这一步。
4. **回传**——将模拟结果沿路径回传，路径上的每个节点访问次数 $N$ 加一，累计价值 $W$ 更新胜负信息。

完成固定的仿真预算后（AlphaGo Zero 每步跑 800 次），算法会选择访问次数最多的根节点子节点作为最终动作，而不是平均价值最高的那个。访问次数是一个更稳健的统计量，因为它已经包含了搜索过程中的自我修正。

### 1.1 UCT：探索与利用的平衡

![UCB1 探索与利用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig2_ucb_exploration.png)

选择阶段用的准则是 *Upper Confidence bound for Trees*（UCT），由 Kocsis 和 Szepesvári 在 2006 年提出。在节点 $s$ 处，选择使下式最大的动作 $a$：

$$\text{UCT}(s, a) = \underbrace{\frac{W(s, a)}{N(s, a)}}_{\text{利用}} \;+\; \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{探索}}.$$
第一项是经验均值，赢得多的节点会被优先选择。第二项是 Auer–Cesa-Bianchi–Fischer 的置信上界：访问次数越少，这一项越大，搜索会倾向于尝试那些较少访问的子节点。当 $N(s,a) \to \infty$ 时，探索项逐渐缩小，规则收敛到贪心利用。UCT 是**渐近最优的**：仿真次数趋于无穷时，访问分布会集中到最优动作上。

AlphaGo 使用的是 PUCT 变体，探索项还加入了网络给出的先验 $p(a\mid s)$：
$$\text{PUCT}(s, a) = Q(s, a) \;+\; c_{\text{puct}} \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}.$$
直观来说：先验告诉搜索*先看哪里*，访问次数告诉它*什么时候停下*。

---
## 2. AlphaGo（2016）：网络遇上搜索

![强化学习（八）：AlphaGo与蒙特卡洛树搜索 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-alphago-and-mcts/illustration_2.png)

![AlphaGo 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig3_alphago_architecture.png)

最早的 AlphaGo 训练分三个阶段：

**第一阶段——监督策略 $p_\sigma$。** 我用 KGS 网站上专家对局的 3000 万个局面，训练了一个 13 层 CNN，目标是预测人类下一步的动作。Top-1 准确率达到 57%，比之前最好的结果（约 44%）提升了一大截。

**第二阶段——强化策略 $p_\rho$。** 先用 $p_\sigma$ 初始化 $p_\rho$，然后通过 REINFORCE 风格的自我对弈训练。对手从之前的检查点中随机采样，避免单一化。强化策略 $p_\rho$ 对监督策略 $p_\sigma$ 的胜率达到 80%。但有趣的是，直接把 $p_\rho$ 用作 MCTS 的先验反而效果更差。原因是它收敛到了几种特定风格上，失去了多样性，而多样性正是搜索需要的有用先验。所以最终上线版本中，MCTS 的先验仍然来自 $p_\sigma$。

**第三阶段——价值网络 $v_\theta$。** 我单独训练了一个 CNN，用来回归对局结果。为了避免同一盘棋中相邻局面高度相关导致过拟合，每盘自我对弈只采样一个局面，最终得到 3000 万对独立的（状态, 结果）样本。

对弈时，MCTS 把两个网络结合起来。叶节点的评估是价值网络估计和一次快速 rollout 的混合：
$$V(s_L) \;=\; (1 - \lambda)\, v_\theta(s_L) \;+\; \lambda\, z_L, \qquad \lambda = 0.5.$$

为什么要混合？2016 年时，价值网络虽然很强，但还不够完美，rollout 能平滑掉它的系统性偏差。到了 2017 年，网络性能大幅提升，rollout 反而成了噪声源。于是 AlphaGo Zero 完全去掉了这一项。

---
## 3. AlphaGo Zero（2017）：从零开始

![AlphaGo Zero 自我对弈循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig4_zero_self_play_loop.png)

AlphaGo Zero 的核心思想和之前一样，但更简洁。它做了三项改动，每一项单独看都挺大胆：

1. **完全不用人类数据。** 智能体从随机初始化起步，只靠自我对弈学习。以前那些预测人类走法准确率超过 50% 的系统，全都被抛弃了。
2. **单一双头网络** $f_\theta(s) = (\mathbf{p}, v)$。一个残差塔输出两个头：策略头给出动作分布，价值头输出标量胜率。原来的策略网络和价值网络被合并成一个。
3. **去掉 rollout。** 叶节点只用价值头评估，快速 rollout 策略彻底取消。

训练流程是一个紧密的闭环（见上图）：

1. **自我对弈。** 当前最强网络用 MCTS 和自己对弈（每步 800 次模拟），生成 $(s_t, \boldsymbol{\pi}_t, z_T)$ 数据。其中 $\boldsymbol{\pi}_t$ 是 MCTS 访问次数分布，比原始网络策略更锐利、更慢但更准；$z_T \in \{-1, +1\}$ 是终局结果，从当前行动方视角记录。
2. **训练。** 更新参数 $\theta$，目标是最小化损失函数：$\mathcal{L}(\theta) \;=\; (z - v)^2 \;-\; \boldsymbol{\pi}^\top \log \mathbf{p} \;+\; c\,\|\theta\|^2,$ 包括价值的均方误差、策略的交叉熵损失以及权重衰减。
3. **评估。** 新网络挑战当前最强网络，只有在 400 局中胜率超过 55%，才能成为新的自我对弈生成器。

这套设计的亮点在于*标签*。MCTS 生成的访问分布 $\boldsymbol{\pi}$ *严格优于*原始网络策略——搜索过程把先验知识打磨得更精细。让 $\mathbf{p}$ 去拟合 $\boldsymbol{\pi}$，相当于把搜索带来的提升蒸馏回网络。这是一种策略迭代，策略改进的步骤由 MCTS 完成。每一代新网络生成的自我对弈数据，都会比它自己稍微强一点。整个过程不需要外部监督，因为对手和学习者同步进步，自然形成了课程学习。

用 4 块 TPU 训练 **3 天**，AlphaGo Zero 就以 100–0 击败了战胜李世石的那个版本。训练 40 天后，它超越了曾击败柯洁的 AlphaGo Master。
## 4. AlphaZero 与 MuZero

![算法演进时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig5_evolution_timeline.png)

**AlphaZero**（2017 年 12 月）证明了 AlphaGo Zero 的算法并不局限于围棋。只需要替换掉棋类特定的状态编码，再做些小调整——比如去掉世代间的胜率限制、允许和棋（国际象棋中常见）——同一套代码在 TPU 上训练约 **9 小时**后，就在国际象棋上以 28 胜 0 负 72 和的成绩击败了 Stockfish 8，在将棋上战胜了 Elmo，还在围棋上超越了 AlphaGo Zero。

**MuZero**（2019 年 11 月）更进一步：完全不需要告诉智能体游戏规则。MuZero 同时学习三个函数：

- **表征** $h_\theta : o_{\le t} \mapsto s_t^0$ —— 把观测历史编码成初始隐状态。
- **动力学** $g_\theta : (s_t^k, a_{t+k}) \mapsto (s_t^{k+1}, r_t^{k+1})$ —— 预测下一个隐状态和奖励。
- **预测** $f_\theta : s_t^k \mapsto (\mathbf{p}_t^k, v_t^k)$ —— 从隐状态输出策略和价值。

MCTS 完全在**隐空间**中展开。搜索过程中没有环境模拟器，只有学到的动力学函数。隐状态不需要还原观测，只要能用来预测奖励、价值和策略就够了。目标宽松了，效果却更好：MuZero 在棋类游戏中追平了 AlphaZero，在 Atari 游戏上超过了 R2D2 和 Ape-X 等无模型方法——而 Atari 没有规则模拟器可用。

### 4.1 Elo 随时间的变化

![Elo 演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig6_elo_progression.png)

左图对比了几代 AlphaGo 的峰值 Elo。右图展示了 AlphaGo Zero 的训练轨迹：3 天超过李世石版本，约 21 天超过 Master 版本，最终在 5200 Elo 左右趋于稳定。作为参考，人类 9 段棋手的 Elo 大约在 3500–3700 之间。

### 4.2 搜索到底有多大作用？

![搜索预算与棋力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo与蒙特卡洛树搜索/fig7_search_vs_strength.png)

搜索和网络是相辅相成的：网络提供先验，搜索优化它。左图显示，每次把 MCTS 仿真次数翻倍，Elo 提升量基本固定（呈对数关系），直到 12800 次仿真也没有饱和迹象。右图展示了神经先验的乘数效应：纯随机 rollout 的朴素 MCTS 很早就停滞不前，而结合网络先验的 MCTS 则能持续提升棋力。两者缺一不可，单独使用任何一项都无法达到竞争力。
## 5. 完整实现：五子棋上的 AlphaZero

9×9 的五子棋是个不错的试验场。规则简单，30 行代码就能写完；分支因子大约是 60；单张消费级 GPU 跑几千局自我对弈，就能看到不错的棋力。

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

训练循环很简单：不断生成自我对弈数据，把 $(s_t, \boldsymbol{\pi}_t, z_T)$ 三元组喂给网络，按 AlphaGo Zero 的损失函数更新参数。在 9×9 棋盘上，每步跑 400 次仿真，单卡跑 50 轮自我对弈迭代，就能看到不错的棋力——能稳赢随机策略和贪心启发式。两条实战经验分享一下：

1. 自我对弈时，给根节点先验加点 Dirichlet 噪声，保持探索性。
2. 对访问分布加个温度参数：前 10 步左右用温度 1，之后接近贪心。这样能避免数据全都来自单一确定性的走法。

---
## 常见问题

### 为什么 AlphaGo Zero 不需要 rollout？

2017 年的时候，更深的残差网络、更多的自我对弈数据，再加上统一的策略-价值头，让价值函数的精度超过了“网络估计 + 随机 rollout”的混合方法。DeepMind 的消融实验也明确表明，纯价值评估的效果更好，所以 rollout 就被去掉了。

### 自我对弈会不会陷入退化的均衡？

在两人零和完美信息博弈中，*虚拟自我对弈*（fictitious play, Brown 1951; Heinrich & Silver 2016）能收敛到 Nash 均衡。MCTS 加了一层乐观探索，防止过早坍塌到某种模式。如果是不完美信息博弈（比如扑克）或者合作博弈，这种性质就不成立了，需要用对手种群（比如 PSRO 或 AlphaStar 的“联赛”）来解决。

### 为什么用访问次数分布作为策略目标，而不是经验 $Q$？

访问次数更稳健：一个子节点如果访问很少，它的 $Q$ 值可能噪声很大；但如果搜索没有持续选择它，它的访问次数也不可能多。另外，对 $\boldsymbol{\pi}$ 做交叉熵时，即使某些动作很少被搜索尝试，也能得到有意义的梯度，而硬性 argmax 目标做不到这一点。

### MCTS 能处理连续动作空间吗？

不能直接处理——UCT 和 PUCT 都假设动作集是有限的。扩展方法比如 *Progressive Widening*，会在节点访问次数增加时逐步采样新动作加入。近年的工作如 Sampled MuZero（2021）已经能处理连续和结构化动作空间。但在纯连续控制问题中，model-free 方法（比如 SAC 和 PPO）还是更实用。

### 为什么每步要跑 800 次仿真？只用 1 次可以吗？

理论上可以用 1 次，但训练会卡住。每步只跑 1 次仿真时，访问分布就等于网络的策略本身，没有任何提升信息，训练没法推进。800 次仿真时，搜索目标比网络本身锐利得多，这个差距正是网络学习的内容。仿真次数超过几千后，收益就开始递减了——AlphaZero 在围棋和国际象棋上都用了 800 次，MuZero 也沿用了这个数字。
## 参考文献

- Silver et al., **Mastering the game of Go with deep neural networks and tree search**, *Nature* 529, 2016.
- Silver et al., **Mastering the game of Go without human knowledge**, *Nature* 550, 2017 (AlphaGo Zero).
- Silver et al., **A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play**, *Science* 362, 2018 (AlphaZero).
- Schrittwieser et al., **Mastering Atari, Go, chess and shogi by planning with a learned model**, *Nature* 588, 2020 (MuZero).
- Kocsis & Szepesvári, **Bandit based Monte-Carlo Planning**, *ECML* 2006 (UCT).
- Auer, Cesa-Bianchi & Fischer, **Finite-time Analysis of the Multiarmed Bandit Problem**, *Machine Learning* 47, 2002 (UCB1).
- Browne et al., **A Survey of Monte Carlo Tree Search Methods**, *IEEE TCIAIG* 4, 2012.
