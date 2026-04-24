---
title: "强化学习（八）：AlphaGo与蒙特卡洛树搜索"
date: 2024-07-08 09:00:00
tags:
  - 强化学习
  - AlphaGo
  - MCTS
  - AlphaZero
  - MuZero
categories: 强化学习
series:
  name: "强化学习"
  part: 8
  total: 12
lang: zh-CN
mathjax: true
description: "从 MCTS 到 AlphaGo、AlphaGo Zero、AlphaZero 与 MuZero：UCT 探索-利用、自我对弈训练、在学到的世界模型里规划。附五子棋上的 AlphaZero 完整实现。"
---

2016 年 3 月，AlphaGo 在首尔以 4–1 击败了围棋世界冠军李世石。这不只是一场体育新闻——它给"让机器在围棋上击败人类顶尖棋手"这个延续了 60 年的人工智能命题画上了句号，比绝大多数学界预测早了整整十年。围棋约有 $10^{170}$ 种合法局面，比可观测宇宙的原子总数还多，纯靠暴力搜索没有任何机会。AlphaGo 的胜利来自一个不一样的思路：让深度网络给出"哪些着法看起来不错"的*直觉*，再让蒙特卡洛树搜索（MCTS）来*推演*，去验证、修正这种直觉。

十八个月后，AlphaGo Zero 不看任何人类棋谱，仅靠规则和三天的自我对弈，就把当年战胜李世石的版本打成了 100–0。AlphaZero 把同一套算法搬到了国际象棋和将棋。MuZero 更进一步——连规则都不需要给。本文沿着这条路线把算法、数学和一份能跑起来的实现讲清楚。

## 你将学到什么

- **MCTS 基础**：四阶段循环、UCT 的探索-利用平衡、渐近最优性
- **AlphaGo（2016）**：三阶段训练（监督策略、强化策略、价值网络），以及 MCTS 如何把它们组合起来
- **AlphaGo Zero（2017）**：从零开始的自我对弈、单一双头网络、不再需要 rollout
- **AlphaZero（2017）**：同一套算法迁移到国际象棋、将棋
- **MuZero（2019）**：在*学到的*隐空间里做规划，不依赖环境规则
- **完整代码**：五子棋上的 AlphaZero——环境、网络、MCTS、自我对弈循环

## 前置知识

- 深度强化学习基础（策略梯度、价值函数）——见[第 3 篇](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/)
- 卷积神经网络
- 博弈树有了解最好，没有也可以

---

## 1. 蒙特卡洛树搜索

经典博弈树搜索（minimax + alpha-beta 剪枝）需要两样东西：一个评估函数，一个可控的分支因子。国际象棋两者都有，围棋两者都没有——分支因子约 250，没有简洁的局面评估，没有好用的启发式。MCTS 绕过了这两个难题：用*采样*代替枚举，并且把采样集中在树里最有希望的部分。

![MCTS 四个阶段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig1_mcts_four_phases.png)

一次 MCTS *仿真*由四步组成：

1. **选择（Selection）**——从根节点出发，按某个搜索准则（下面要讲的 UCT）一路向下选子节点，直到走到一个还有"未尝试着法"的节点。这是当前*树*里的叶子，不一定是博弈意义上的终局。
2. **扩展（Expansion）**——挑一个未尝试的着法，把对应的局面作为新子节点加进树里。
3. **模拟（Simulation, rollout）**——从新子节点出发把对局快速走到终局。最朴素的 MCTS 用随机走子，早期 AlphaGo 用一个轻量网络，AlphaGo Zero 干脆完全不要这一步、用价值网络代替。
4. **回传（Backpropagation）**——把这次仿真的结果沿路径回传，路径上每个节点的访问次数 $N$ 加一、累计价值 $W$ 加上对应胜负。

跑完固定的仿真预算（AlphaGo Zero 是每步 800 次）后，最终落子选*访问次数最多*的根节点子节点，而不是平均价值最高的。访问次数是个更稳健的统计量——它已经把搜索过程里的反复试探与修正消化进去了。

### 1.1 UCT：探索与利用的平衡

![UCB1 探索与利用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig2_ucb_exploration.png)

选择阶段用的准则叫 *Upper Confidence bound for Trees*（UCT），来自 Kocsis 与 Szepesvári 在 2006 年的工作。在节点 $s$ 处，选使下式最大的子动作 $a$：

$$\text{UCT}(s, a) = \underbrace{\frac{W(s, a)}{N(s, a)}}_{\text{利用}} \;+\; \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{探索}}.$$

第一项是经验均值——赢得多的节点会被选得多。第二项是 Auer–Cesa-Bianchi–Fischer 的置信上界：访问次数越少，这一项越大，搜索就被推着去*尝试*那些访问较少的子节点。当 $N(s,a) \to \infty$ 时探索项缩为零，整条规则收敛到贪心利用。UCT 是**渐近最优的**：仿真次数趋于无穷时，访问分布会集中到最优动作上。

AlphaGo 用的是 PUCT 变体——把网络给的先验 $p(a\mid s)$ 也乘进探索项里：

$$\text{PUCT}(s, a) = Q(s, a) \;+\; c_{\text{puct}} \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}.$$

直观上：先验告诉搜索*先看哪里*，访问次数告诉它*什么时候停下*。

---

## 2. AlphaGo（2016）：网络遇上搜索

![AlphaGo 架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig3_alphago_architecture.png)

最早的 AlphaGo 训练分三个阶段：

**第一阶段——监督策略 $p_\sigma$。** 用 KGS 网站上专家对局的 3000 万个局面训练一个 13 层 CNN，预测人类下一步会怎么走。Top-1 准确率 57%，比此前最好的工作（约 44%）大幅提升。

**第二阶段——强化策略 $p_\rho$。** 把 $p_\rho$ 用 $p_\sigma$ 初始化，再用 REINFORCE 风格的策略梯度做自我对弈训练，对手从过去的若干检查点里随机采样，避免对手单一。$p_\rho$ 对 $p_\sigma$ 的胜率约 80%。但有个反直觉的现象：把 $p_\rho$ 直接放进 MCTS 当先验反而更差——它已经在风格上"收敛"到几种偏好上去了，丢了多样性。所以最终上线版本里，搜索的先验仍然来自 $p_\sigma$。

**第三阶段——价值网络 $v_\theta$。** 单独训一个 CNN 去回归对局结果。同一盘棋里相邻局面高度相关，直接拿全部局面训练会严重过拟合，于是每盘自我对弈*只采样一个*局面，最终得到 3000 万对独立的（局面, 结果）样本。

对弈时，MCTS 把两张网络糅合起来。叶节点的评估是价值网络估计与一次快速 rollout 的混合：

$$V(s_L) \;=\; (1 - \lambda)\, v_\theta(s_L) \;+\; \lambda\, z_L, \qquad \lambda = 0.5.$$

为什么要混？2016 年那时价值网络强但还不到位，rollout 能在统计意义上抹掉它的系统性偏差。到了 2017 年网络足够强，rollout 反倒成了噪声源——AlphaGo Zero 干脆把这一项去掉了。

---

## 3. AlphaGo Zero（2017）：从零开始

![AlphaGo Zero 自我对弈循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig4_zero_self_play_loop.png)

AlphaGo Zero 用的是同一个想法，但*更简洁*。三处改动，每一条单看都挺冒险：

1. **不用任何人类数据。** 从随机初始化开始，全靠自我对弈学习。此前所有"预测人类下一步准确率超过 50%"的成果，统统作废。
2. **单一双头网络** $f_\theta(s) = (\mathbf{p}, v)$。一座残差塔，输出两个头：策略头给出着法分布，价值头给出标量胜率。原来的策略网络和价值网络合并成了一个。
3. **不再 rollout。** 叶节点*只*由价值头评估，快速 rollout 策略整个去掉。

训练流程是一个紧密咬合的闭环（见上图）：

1. **自我对弈。** 当前最优网络与自己用 MCTS 对弈（每步 800 次仿真），生成 $(s_t, \boldsymbol{\pi}_t, z_T)$ 三元组。这里 $\boldsymbol{\pi}_t$ 是 *MCTS 访问次数*归一化后的分布——比网络原始策略更锐利、更慢但更准——$z_T \in \{-1, +1\}$ 是终局结果，从当前出招方的视角写下。
2. **训练。** 用以下损失更新参数 $\theta$：$$\mathcal{L}(\theta) \;=\; (z - v)^2 \;-\; \boldsymbol{\pi}^\top \log \mathbf{p} \;+\; c\,\|\theta\|^2,$$分别是价值的均方误差、策略的交叉熵、以及权重衰减。
3. **评估。** 新网络与当前最优网络对弈，胜率超过 55%（共 400 局）才能取代之，成为下一轮自我对弈的生成方。

这套设计的精妙之处在*标签*。MCTS 给出的访问分布 $\boldsymbol{\pi}$ *严格强于*生成它的那张网络的策略——搜索把先验"打磨"了一遍。让 $\mathbf{p}$ 去逼近 $\boldsymbol{\pi}$，等于把搜索带来的提升*蒸馏*回网络。这是一种把策略改进步骤交给 MCTS 来做的策略迭代。每一代新网络生成的自我对弈数据，都会比它自己稍微强一点，整个过程不需要外部监督就能滚起来——因为*对手始终和学习者一起进步*，自然形成了课程学习。

在 4 块 TPU 上训练 **3 天**，AlphaGo Zero 把战胜李世石的那个版本打成了 100–0。训练 40 天后超过了曾击败柯洁的 AlphaGo Master。

---

## 4. AlphaZero 与 MuZero

![算法演进时间线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig5_evolution_timeline.png)

**AlphaZero**（2017 年 12 月）证明了 AlphaGo Zero 算法并非围棋专属。只换掉与具体棋种相关的状态编码，外加几处小改动——例如取消世代之间的胜率门槛、允许和棋（国际象棋常出现）——同一份代码就在国际象棋上把 Stockfish 8 打成了 28 胜 0 负 72 和（约 **9 小时**训练，TPU），在将棋上超过 Elmo，在围棋上超过 AlphaGo Zero。

**MuZero**（2019 年 11 月）再进一步：连规则都不告诉算法。MuZero 联合学三个函数：

- **表征** $h_\theta : o_{\le t} \mapsto s_t^0$——把观测历史编码为初始隐状态。
- **动力学** $g_\theta : (s_t^k, a_{t+k}) \mapsto (s_t^{k+1}, r_t^{k+1})$——预测下一隐状态和奖励。
- **预测** $f_\theta : s_t^k \mapsto (\mathbf{p}_t^k, v_t^k)$——从隐状态输出策略与价值。

MCTS 整个发生在**隐空间**里，搜索过程中没有任何环境模拟器，只有学到的动力学函数。隐状态*不需要*能还原观测，它只要对预测奖励、价值、策略有用就够了。这个目标比"重建观测"宽松得多，正因如此 MuZero 在棋类上追平了 AlphaZero，又在 Atari 上超过了 R2D2、Ape-X 等 model-free 方法——而 Atari 是没有规则模拟器可用的。

### 4.1 Elo 演进

![Elo 演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig6_elo_progression.png)

左图比较了几代 AlphaGo 的峰值 Elo。右图给出了 AlphaGo Zero 训练过程中 Elo 随时间的变化：3 天超过李世石版本，约 21 天超过 Master 版本，最终在约 5200 Elo 附近饱和。作为参考，人类 9 段棋手的 Elo 大致在 3500–3700 区间。

### 4.2 搜索究竟带来多少提升？

![搜索预算与棋力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/08-AlphaGo%E4%B8%8E%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2/fig7_search_vs_strength.png)

搜索和网络是互补的：网络给先验，搜索把它精炼。左图显示，每翻倍一次 MCTS 仿真次数，Elo 增量大致恒定（呈对数关系），到 12800 仿真也看不到饱和迹象。右图显示了神经先验带来的乘数效应——纯随机 rollout 的朴素 MCTS 很早就饱和，加上网络先验后能压出更大的棋力。两者缺一不可，单用任一项都不够强。

---

## 5. 完整实现：五子棋上的 AlphaZero

9×9 的五子棋是一个很合适的练手场：规则 30 行写完，分支因子约 60，单卡跑几千局自我对弈就能看到像样的棋力。

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
    """简单的双头网络：共享 CNN 主干 + 策略头 + 价值头。"""
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
        # 用 log-softmax，配合 MCTS 目标做交叉熵更稳定
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
            # 上一层是对手视角，符号要翻
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

训练循环就是不断生成自我对弈数据、把 $(s_t, \boldsymbol{\pi}_t, z_T)$ 三元组喂给网络、按 AlphaGo Zero 损失函数更新参数。在 9×9 棋盘、每步 400 次仿真的设置下，单卡上跑 50 轮自我对弈迭代就能看到不平凡的棋力（能稳定击败随机策略和贪心启发式）。两条实战经验：(i) 自我对弈时给根节点先验加 Dirichlet 噪声，保持探索；(ii) 对访问分布加*温度*——前 10 步左右用温度 1，之后接近贪心——避免数据全部来自单一确定性的对局走法。

---

## 常见问题

**Q：为什么 AlphaGo Zero 不再需要 rollout？**
到 2017 年，更深的残差网络、更多的自我对弈数据、统一的策略-价值头使得价值函数本身的精度已经超过了"网络估计 + 随机 rollout"的混合。DeepMind 的消融实验是明确的——纯价值评估更好，rollout 就被去掉了。

**Q：自我对弈会不会陷入退化的均衡？**
对于两人零和完美信息博弈，*虚拟自我对弈*（fictitious play, Brown 1951; Heinrich & Silver 2016）能收敛到 Nash 均衡。MCTS 在上面叠了一层乐观探索，进一步避免提前坍塌到某种风格。如果换成不完美信息博弈（扑克）或合作博弈，这条性质就不再成立了，需要维护一个对手种群（如 PSRO、AlphaStar 的"联赛"）。

**Q：为什么用*访问次数*分布作为策略目标，而不是经验 $Q$？**
访问次数稳健：访问很少的子节点 $Q$ 估计噪声很大，但不可能在没被搜索看好的情况下被反复访问。再者，对 $\boldsymbol{\pi}$ 做交叉熵能给那些搜索很少试过的动作也留下有意义的梯度，硬性 argmax 目标做不到这一点。

**Q：MCTS 能处理连续动作空间吗？**
不能直接处理——UCT 和 PUCT 都假设动作集是有限的。扩展方法包括 *Progressive Widening*（节点访问够多时陆续采样新动作加入），近年也有 Sampled MuZero（2021）这类工作。纯连续控制问题里，model-free 方法（SAC、PPO）依然更好用。

**Q：为什么是每步 800 次仿真？只用 1 次行不行？**
理论上行，但训不动。每步只跑 1 次仿真时，访问分布*就等于*网络的策略本身，没有任何提升信息，训练原地打转。800 次的时候搜索目标比网络本身锐利得多，正是这道差距给了网络可学的东西。继续往上的边际收益从几千以后就开始衰减——AlphaZero 在围棋和国际象棋上都用 800，MuZero 也沿用了这个数字。

---

## 参考文献

- Silver et al., **Mastering the game of Go with deep neural networks and tree search**, *Nature* 529, 2016.
- Silver et al., **Mastering the game of Go without human knowledge**, *Nature* 550, 2017（AlphaGo Zero）。
- Silver et al., **A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play**, *Science* 362, 2018（AlphaZero）。
- Schrittwieser et al., **Mastering Atari, Go, chess and shogi by planning with a learned model**, *Nature* 588, 2020（MuZero）。
- Kocsis & Szepesvári, **Bandit based Monte-Carlo Planning**, *ECML* 2006（UCT 原始论文）。
- Auer, Cesa-Bianchi & Fischer, **Finite-time Analysis of the Multiarmed Bandit Problem**, *Machine Learning* 47, 2002（UCB1）。
- Browne et al., **A Survey of Monte Carlo Tree Search Methods**, *IEEE TCIAIG* 4, 2012。

---

## 系列导航

- **上一篇**：[第 7 篇 — 模仿学习与逆强化学习](/zh/强化学习-七-模仿学习与逆强化学习/)
- **下一篇**：[第 9 篇 — 多智能体强化学习](/zh/强化学习-九-多智能体强化学习/)
- [查看强化学习系列全部 12 篇](/tags/强化学习/)
