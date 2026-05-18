---
title: '优化理论（三）：梯度下降族——从 SGD 到 AdamW'
date: 2022-09-16 09:00:00
tags:
  - ML
  - Optimization
  - Deep Learning
  - LLM
categories: Algorithm
lang: zh
mathjax: true
description: "一篇文章串起 GD → SGD → Momentum → NAG → AdaGrad → RMSProp → Adam → AdamW，再到 Lion / Sophia / Schedule-Free 的完整脉络：每一步解决了前一步的什么痛点？为什么大模型几乎都在用 AdamW？以及 2023 年之后我们究竟走到了哪里。"
disableNunjucks: true
translationKey: "optim-03"
series: optimization-theory
series_order: 3
series_total: 12
aliases:
  - /zh/standalone/优化算法的演进-从梯度下降到adam/
---
为什么“调学习率是一门艺术”成了 ResNet 的梗，而每篇现代 LLM 论文却只是简单写下 “AdamW, $\beta_1{=}0.9, \beta_2{=}0.95, \mathrm{wd}{=}0.1$” 就翻篇了？这并非偶然——这是 **三十余年优化器演化的终点**。

本文沿着一条主线完整梳理这段演化史：每一步的出现，都是因为前一步存在 **某个具体缺陷**。最终我们会介绍三种真正进入 2023 年后大模型工具箱的方向：Lion、Sophia 和 Schedule-Free。


---

为什么训练 ResNet 的时候“调学习率是一门艺术”是个梗，可几乎每篇 LLM 论文写到优化器都只是淡淡一句 “AdamW, $\beta_1{=}0.9, \beta_2{=}0.95, \mathrm{wd}{=}0.1$” 就翻篇了？这不是巧合，是**三十多年优化器演化的终点**。

我在做研究和读代码的过程中走过这条线一遍：每一步新优化器的出现，都是因为前一步在某个具体场景里栽了跟头。本文我把这条主线串起来，从最朴素的梯度下降一路讲到 2023 年之后才进入大模型工具箱的几个新方法（Lion、Sophia、Schedule-Free）。

读这篇文章不需要你能背出 Adam 的更新公式——只需要你愿意一步步问“上一个版本不够好在哪里”。

## 你将学到什么

- 为什么 GD 在病态损失曲面上会“之字形”前进，以及动量如何从物理角度解决这个问题
- Nesterov “前瞻”与经典动量之间的精确数学区别
- 为什么 AdaGrad 对稀疏特征效果极佳，却又为何在深度网络中最终“窒息”
- RMSProp 如何通过一行改动（指数移动平均）拯救了 AdaGrad
- Adam 如何将动量与 RMSProp 结合，以及为何偏差校正至关重要
- AdamW vs Adam：一旦在分母中引入自适应缩放，“L2 == weight decay” 就不再成立
- Lion / Sophia / Schedule-Free：三种在 AdamW 之后真正实现扩展的方向

## 前置知识

- 基础微积分（梯度、Hessian、泰勒展开）
- 一些神经网络训练经验（任意框架）

---

## 演化脉络一览

| Year | Algorithm | Specific problem it fixed |
|---|---|---|
| 1847 | GD | Formalized "step along the negative gradient" |
| 1951 | SGD | Datasets too big for full-batch gradients |
| 1964 | Momentum | GD zig-zags in narrow valleys |
| 1983 | NAG | Plain momentum overshoots near minima |
| 2011 | AdaGrad | Sparse features need per-coordinate LRs |
| 2012 | RMSProp | AdaGrad's denominator suffocates the LR |
| 2014 | Adam | Combine direction (momentum) and scale (RMSProp) |
| 2017 | AdamW | Adam + L2 != Adam + weight decay |
| 2023 | Lion | Drop the second moment; use sign of momentum |
| 2023 | Sophia | Cheap diagonal-Hessian preconditioner |
| 2024 | Schedule-Free | Stop needing to know the total step count |

以下各节按此顺序展开。

## Gradient descent (GD)：起源

给定一个可微损失函数 $J(\theta)$，最简单的更新方式是
$$\theta_{t+1} = \theta_t - \eta\,\nabla J(\theta_t).$$
**收敛性**：若 $J$ 是凸函数且梯度满足 $L$-Lipschitz 条件，则当 $\eta \le 1/L$ 时，能保证（次）线性收敛到全局最小值。

**催生后续所有方法的根本弱点**：

- 在 **病态曲率**（Hessian 条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 很大）下，迭代次数随 $\kappa$ 线性增长。一维直觉：$f(\theta)=\frac{1}{2}H\theta^2$ 给出 $\theta_{t+1}=(1-\eta H)\theta_t$，稳定当且仅当 $\eta < 2/H$。你的步长 **受限于最陡方向的曲率**。
- 当最陡方向（$\lambda_{\max}$）与最平缓方向（$\lambda_{\min}$）相差几个数量级时，你在平缓方向几乎不动，却在陡峭方向来回震荡。这就是 **窄谷问题** —— 见下图 Fig 1 左侧面板。

## SGD：噪声的代价与红利

当数据集无法装入内存时，你用小批量估计替代全梯度：
$$g_t = \nabla J(\theta_t) + \xi_t,\qquad \mathbb{E}[\xi_t]=0.$$
噪声 $\xi_t$ 既是诅咒也是祝福：
- **诅咒**：稍大的步长会被噪声放大，导致发散。
- **祝福**：噪声有助于 **逃离尖锐局部极小值** —— 后来 Keskar 等人将其与“平坦极小值”的泛化理论联系起来。

**Fig 1 中间面板**：SGD 在同一山谷中的轨迹比 GD 更“毛糙”，但平均而言仍流向谷底。

## Momentum：赋予优化器惯性

心理模型：把 $\theta$ 想象成 **沿山谷滚动的球**。GD 是一只“无质量的虫子”——每一步只看到局部斜率，因此在狭窄方向来回弹跳。给虫子加上质量和惯性，**惯性会沿山谷长轴累积，而垂直方向的震荡相互抵消**。
$$v_t = \gamma v_{t-1} + \eta\,g_t,\qquad \theta_{t+1} = \theta_t - v_t.$$
典型 $\gamma = 0.9$ —— 对过去梯度进行几何加权，有效记忆长度约为 $1/(1-\gamma) = 10$ 步。

**关键洞见**：动量 **将有效步长放大了约 $1/(1-\gamma)$ 倍**。因此当你开启动量时，必须 **缩小** 之前不用动量时的学习率。这是最常见的新手陷阱。

**Fig 1 右侧面板**：同样的山谷，但动量路径被“拉直”——垂直震荡抵消，纵向速度累积。

![GD / SGD / Momentum trajectories on an ill-conditioned quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig1_gd_sgd_momentum_contour.png)

## Nesterov accelerated gradient (NAG)：先看再跳

经典动量在接近最小值时会 **过冲**：它在当前点计算梯度，因此只有在多走一步后才意识到“哎呀，走太远了”。

NAG 只改了一行：
$$v_t = \gamma v_{t-1} + \eta\,\nabla J(\theta_t - \gamma v_{t-1}),\qquad \theta_{t+1} = \theta_t - v_t.$$
**唯一区别在于梯度计算的位置**：经典动量在 $\theta_t$ 处计算，NAG 在 **前瞻点** $\theta_t - \gamma v_{t-1}$ 处计算——即“仅靠动量步我会走到哪里”。

**为何有效**：这是一种单步前瞻。如果斜率即将变平，NAG 能提前察觉并减速；反之亦然。Nesterov (1983) 证明这能将凸光滑优化的收敛速度从 $O(1/t)$ 加速到 $O(1/t^2)$。

![NAG: lookahead gradient evaluation reduces overshoot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig2_nesterov_lookahead.png)

## AdaGrad：每个坐标拥有自己的学习率

到 2011 年，NLP 领域已被 **稀疏特征** 淹没——比如 word2vec 中，一个罕见词可能在百万样本中只出现 5 次。若对所有参数使用同一个 $\eta$：

- 罕见词参数：梯度小，但相同的 $\eta$ 要么太大（直接杀死它们），要么太小（永远学不到）。
- 常见词参数：梯度大且频繁，更希望 **更小的步长**。

Duchi 提出了 AdaGrad——基于每个坐标的梯度历史进行 **逐坐标自适应**：
$$G_t = G_{t-1} + g_t^2 \quad(\text{element-wise})$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t.$$
**直觉**：累积的 $g^2$ 越大 → 分母越大 → 有效步长越小。罕见但突然变大的坐标 → 分母小 → 有效步长大。**学习率按频率自动分配**。

**致命缺陷**：$G_t$ 是一个 **单调递增的和**。在深度网络中训练数十万步后，分母会使所有有效学习率趋近于零。模型“窒息”。Fig 3 右侧面板清晰展示了这一点。

![AdaGrad: shrinks the steep direction automatically, but every per-coord LR decays monotonically](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig3_adagrad_per_coord_lr.png)

## RMSProp：用 EMA 替代累积和

在 2012 年 Coursera 课程幻灯片中，Hinton **只改了一处** 就拯救了 AdaGrad：将累积和 $\sum g_t^2$ 替换为指数移动平均：
$$E[g^2]_t = \rho\,E[g^2]_{t-1} + (1-\rho)\,g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}\,g_t.$$
典型 $\rho = 0.9$ —— “大致记住最近 10 步的梯度幅度”。

**关键区别**：
- **AdaGrad**：$G_t$ 只增不减 → 学习率只减不增（不可逆）。
- **RMSProp**：$E[g^2]_t$ 是有限窗口平均 → 当 **梯度幅度变化** 时，分母随之调整 → 学习率可以 **重新放大**。

**Fig 4 右侧面板** 直接展示了这一点：在第 60 步梯度幅度骤降。AdaGrad 的有效学习率持续下降；RMSProp 的有效学习率 **回升** 以匹配新状态。

![RMSProp (EMA) vs AdaGrad (cumulative) under non-stationary gradient magnitude](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig4_rmsprop_moving_average.png)

## Adam：将动量与 RMSProp 缝合

此时两条线索均已成熟：
- **动量** 提供良好的 **方向**。
- **RMSProp** 提供良好的 **逐坐标缩放**。

Kingma & Ba (2014) 直接将它们拼接起来：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t \quad\text{(1st moment = momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \quad\text{(2nd moment = RMSProp)}$$
**偏差校正** —— 最被低估的细节。由于 $m_0=v_0=0$，前几步中 $m_t$ 和 $v_t$ 都严重偏向零。修正方法：
$$\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
默认值：$\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$。

**为何 $\beta_2$ 远大于 $\beta_1$**：方差估计比均值估计更嘈杂，需要更长的平均窗口。$1/(1-0.999) = 1000$ 步——这正是 Adam 通常需要约 1000 步预热才能让 $v_t$ “热起来”的原因。

![Adam dataflow: momentum branch + RMSProp branch -> bias correction -> adaptive update](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig5_adam_combined.png)

## AdamW：存活十年的 weight-decay bug

向损失函数添加 L2 正则项 $\frac{\lambda}{2}\|\theta\|^2$ 会在梯度中增加一项 $\lambda\theta$。在 **SGD** 中，这等价于每步将权重乘以 $(1-\eta\lambda)$ —— 即经典的“weight decay”。

但 Loshchilov & Hutter (2017) 发现，在 **Adam** 中这两种操作 **不再等价**。原因很直接：Adam 将梯度除以 $\sqrt{\hat v_t}$。如果你将 $\lambda\theta$ 折入梯度，**它也会被 $\sqrt{\hat v_t}$ 除** —— 这意味着 **梯度历史大的参数获得更少的 weight decay**，而这与正则化的目标背道而驰。

AdamW 的修复方法是将 weight decay **移出梯度**，直接作用于参数：
$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\,\theta_t.$$
效果：在相同 $\lambda$ 和 LR 下，AdamW 在 ImageNet/Transformer 上的泛化差距明显小于 Adam+L2。这就是为何 **2018 年后所有大模型预训练默认使用 AdamW**。

![AdamW (decoupled) vs Adam+L2 (coupled): where weight decay enters the update](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig6_adamw_vs_adam.png)

## 2023 年后的前沿：三种经受住规模考验的方向

在 AdamW 主导约 6 年后，自 2023 年起有三种方向真正 **在大规模上被验证有效**。

### Lion (Google, 2023)：只保留符号

由 AutoML 程序搜索发现；更新只保留 **符号**：
$$m_t = \beta_2 m_{t-1} + (1-\beta_2)\,g_t$$

$$\theta_{t+1} = \theta_t - \eta\,\mathrm{sign}\bigl(\beta_1 m_{t-1} + (1-\beta_1)\,g_t\bigr).$$
**显著特性**：
- **优化器状态减半**：无需 $v_t$ —— 对百亿参数模型而言能省下真金白银。
- **恒定更新幅度 $\eta$**：因为 sign 返回 $\pm 1$。因此 Lion 的 LR 必须比 AdamW **小约 10 倍**，wd 则需大 10 倍。
- 在 ViT 和 LLM 预训练中，以更快的实际耗时匹配或略微超越 AdamW。

### Sophia (Stanford, 2023)：廉价二阶方法

Sophia 将廉价的对角 Hessian 估计插入分母：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$

$$h_t \approx \mathrm{diag}(H_t) \quad\text{(Hutchinson estimate, every } k \text{ steps)}$$

$$\theta_{t+1} = \theta_t - \eta\,\mathrm{clip}\!\left(\frac{m_t}{\max(\gamma h_t,\,\varepsilon)},\,1\right).$$
**核心技巧**：
- 使用 $\mathrm{diag}(H)$ 而非 $g^2$ 作为分母 —— 这才是 **真实的曲率**。
- `clip` 至关重要：在非凸损失中 $h_t$ 可能为负，裁剪可保证更新有界。
- Hessian 探针每 $k$ 步运行一次，因此摊销成本适中。

报告结果：在 GPT-2 规模下，达到相同困惑度所需的实际耗时大约减半。

### Schedule-Free (Meta, 2024)：抛弃调度

学习率调度（cosine、WSD 等）都有一个烦人之处：**你必须提前知道总步数**。研究中通常不知道，因此承诺一个调度会束缚手脚。

Schedule-Free AdamW 用 **迭代平均** 替代调度：
$$y_t = (1-\beta) z_t + \beta x_t \quad\text{(point at which the gradient is taken)}$$

$$z_{t+1} = z_t - \eta\,\nabla J(y_t)$$

$$x_{t+1} = (1-c_t)\,x_t + c_t\,z_{t+1} \quad\text{(returned "averaged" parameters)}$$
结果：在 **无显式衰减** 的情况下匹配 cosine 调度的最终性能，并可在 **训练中途扩展** 而无需重新设计。

![Lion / Sophia / Schedule-Free: the three post-AdamW directions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/optimizer-evolution-gd-to-adam/fig7_modern_optimizers.png)

## 选择指南

| Setting | Recommendation | Why |
|---|---|---|
| Convex / simple regression | GD or SGD-momentum | Strong theory, easy tuning |
| CV baseline | SGD + Nesterov + cosine | Historically the best optimum on ResNet/CNN |
| Transformer / LLM pretraining | **AdamW** + warmup + cosine/WSD | Industry default; close to free lunch |
| Memory-constrained large models | **Lion** | Saves the 1st-moment-equivalent state (~1/3 memory) |
| Research, unknown training length | **Schedule-Free AdamW** | Extend mid-run, no schedule redesign |
| Chasing wall-clock SOTA | **Sophia** | 2nd-order acceleration, but engineering cost |

## 最常被忽略的五个事实

1. **开启动量时，降低 LR**。动量将有效步长放大了约 $1/(1-\gamma)$ 倍。当 $\gamma=0.9$ 时约为 10 倍。
2. **Adam 的 $\beta_2 = 0.999$ 意味着约 1000 步预热**，因为在此之前 $v_t$ 尚未“热起来”。
3. **AdamW 的 wd 与 LR 解耦**。当 LR 调度器衰减 LR 时，wd **不会** 随之衰减。这是与旧 SGD+L2 工作流的根本区别。
4. **Lion 的 LR 必须比 AdamW 小约 10 倍**。直接复制 AdamW 的 `3e-4` 会立即发散。
5. **二阶方法曾被认为“永久不实用”并非因为效果差，而是因为 Hessian 计算太昂贵**。Sophia 通过结合 $\mathrm{diag}(H)$ 与廉价的 Hutchinson 估计打破了这一壁垒。

## 优化器状态内存：数学隐藏的成本

动量、AdaGrad、RMSProp 和 Adam 的简洁推导从未提及 VRAM。但在生产环境中，这是主要约束。对于 fp16 下含 $P$ 个可训练参数的模型：

| Optimizer | Per-param state (fp32) | For 7 B params | For 70 B params |
|---|---|---|---|
| SGD | 0 | 0 | 0 |
| SGD + momentum | 4 bytes | 28 GB | 280 GB |
| Adam / AdamW | 8 bytes ($m, v$) | 56 GB | 560 GB |
| Lion | 4 bytes ($m$ only) | 28 GB | 280 GB |
| Sophia | 8 bytes ($m, h$) | 56 GB | 560 GB |
| Adafactor | ~2 bytes (factored) | 14 GB | 140 GB |
| 8-bit AdamW (`bnb`) | 2 bytes | 14 GB | 140 GB |

正是这张表决定了你实际能运行哪个优化器。一个 fp16 的 7B 模型权重占 14GB，梯度占 14GB；AdamW 再增加 56GB，超过单张 A100 80GB 的容量。而使用 8-bit AdamW 或 Adafactor 则能轻松容纳。这就是为何 LLM 预训练文献对优化器状态量化如此痴迷，而传统 ML 从未有过：算法本身没问题，瓶颈在内存。

实用经验法则：若优化器状态超过模型权重内存的 1.5 倍，你很可能需要分片优化器（ZeRO-1）、量化（`bitsandbytes`）或改用状态更轻的算法（Lion、Adafactor）。具体选择取决于瓶颈是单卡内存还是集群总内存。


![Optimizer state memory across optimizers, and total VRAM at 7B / 70B scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/03-gradient-descent-family/fig8_optimizer_memory.png)

## 混合精度：优化器看到的是 fp32

一个让首次预训练者吃亏的微妙点：即使其余训练使用 fp16/bf16，优化器几乎总是在 fp32 下运行。原因是 Adam 的指数移动平均累积数千步。当 $\beta_2 = 0.999$ 时，$v_t$ 是一个求和，其中最小贡献者仅为最大者的 $10^{-3}$，很容易低于 fp16 的可表示范围（约 $6 \times 10^{-5}$ 到 $6 \times 10^4$）。

标准做法是：

1. 前向 + 反向传播使用 fp16/bf16。梯度为 fp16/bf16。
2. 在优化器步骤前将梯度转为 fp32。
3. 优化器状态（$m, v$，权重）以 fp32 存储。
4. 更新后，将权重转回 fp16/bf16 用于下一次前向传播。

PyTorch 的 `torch.amp` 和 DeepSpeed 会自动处理。这也解释了为何上表中的“内存成本”以 fp32 字节计——即使模型是 fp16，优化器仍为每个参数支付 fp32 成本。

bf16 略有不同：其动态范围足够宽，有时可全程保留 bf16 梯度，但所有值得使用的实现中优化器状态仍是 fp32。在大规模 LLM 预训练中，这是不可妥协的。


![Mixed-precision training data flow: where each tensor lives in fp16/bf16 vs fp32](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/03-gradient-descent-family/fig9_mixed_precision.png)

## 各优化器的学习率合理范围（Transformer 基线）

这是我启动 Transformer 类任务时首先尝试的区间。将其视为贝叶斯先验，而非最终值。

| Optimizer | LR range | Notes |
|---|---|---|
| SGD + momentum | 0.01 – 0.5 | Linear warmup ~5 % of total steps |
| AdamW | 1e-4 – 6e-4 | Most LLMs land at 3e-4 ± 1.5x |
| Lion | 1e-5 – 6e-5 | Roughly 10× smaller than AdamW |
| Sophia | 1e-4 – 4e-4 | Also lower than AdamW |
| Schedule-Free AdamW | 1e-4 – 6e-4 | Same as AdamW; the *schedule* is what changes |
| Adafactor | rel. step 0.01 – 0.05 | Uses relative step; literal LR is meaningless |

这些只是初试先验。实际数值取决于 batch size（SGD 遵循线性缩放规则，Adam 遵循类似 $\sqrt{}$ 的缩放规则）、预热长度和数据集噪声。但若初始值超出这些区间，几乎总是配置错误，而非新发现。

## 总结

三十余年的优化器演化可浓缩为两句话：

- **从 GD 到 Adam**：先解决 **方向** 问题（动量），再解决 **缩放** 问题（AdaGrad / RMSProp），然后合并二者并修正偏差（Adam）。
- **Adam 之后**：算法改进让位于 **正则化细节**（AdamW）、**内存效率**（Lion）、**二阶信息**（Sophia）和 **调度自由**（Schedule-Free）。

若你只记住一点：**LLM 时代的默认仍是 AdamW + warmup + cosine/WSD + gradient clipping**。除非你有明确瓶颈（内存、实际耗时、调度灵活性），否则任何声称超越 AdamW 的论文都应在你自己的任务上复现基线后再做决定。

## 参考文献

- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014) — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017) — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Symbolic Discovery of Optimization Algorithms / Lion (Chen et al., 2023) — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training (Liu et al., 2023) — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
- The Road Less Scheduled / Schedule-Free (Defazio et al., 2024) — [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- For the **learning-rate side** of training: [Learning Rate: From Basics to Large-Scale Training](./04-learning-rate-schedules/)