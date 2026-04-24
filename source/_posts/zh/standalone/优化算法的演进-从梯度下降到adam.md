---
title: "优化算法的演进：从梯度下降到 Adam（再到 2025 之后）"
date: 2025-02-01 09:00:00
tags:
  - ML
  - Optimization
  - Deep Learning
  - LLM
categories: Algorithm
lang: zh-CN
mathjax: true
description: "一篇文章串起 GD → SGD → Momentum → NAG → AdaGrad → RMSProp → Adam → AdamW，再到 Lion / Sophia / Schedule-Free 的完整脉络：每一步解决了前一步的什么痛点？为什么大模型几乎都在用 AdamW？以及 2023 年之后我们究竟走到了哪里。"
---

为什么训练 ResNet 时大家都说"调 LR 是手艺活"，到了 GPT/LLaMA 这一代，几乎所有论文却清一色地写 "AdamW，$\beta_1{=}0.9, \beta_2{=}0.95, \mathrm{wd}{=}0.1$"？这不是巧合——它是**优化器三十年演进**的最终收敛点。

本文沿一条主线把这条演进路径讲清楚：每一步动了什么、为什么动，以及**它解决了上一代算法的哪个具体毛病**。文末把 2023 年以后真正在大模型里被反复验证的三条路（Lion / Sophia / Schedule-Free）也讲透。

## 你将学到

- 为什么 GD 在病态曲面上"之"字形震荡，动量如何在物理上修复它
- Nesterov 的"先看一眼再走"和经典动量在数学上差在哪里
- AdaGrad 为什么在稀疏特征上是杀手锏，又为什么在深度网络里"慢慢窒息"
- RMSProp 用一行代码（指数滑动平均）救活了 AdaGrad
- Adam 怎么把动量和 RMSProp 拼在一起，为什么需要"偏差修正"
- AdamW vs Adam：为什么"L2 正则等价于 weight decay"在 Adam 里**不再成立**
- Lion / Sophia / Schedule-Free：2023+ 真正进入主流训练栈的三条路

## 前置知识

- 基础微积分（梯度、Hessian、Taylor 展开）
- 训练过任意一个神经网络（任何框架都行）

---

# 演进总览

| 时间 | 算法 | 解决的核心问题 |
|---|---|---|
| 1847 | GD | 把"沿负梯度走"写成迭代格式 |
| 1951 | SGD | 数据量大到无法算全量梯度 |
| 1964 | Momentum | GD 在窄长山谷里之字形震荡 |
| 1983 | NAG | 动量在转弯处过冲（overshoot） |
| 2011 | AdaGrad | 稀疏特征上每个坐标该有自己的步长 |
| 2012 | RMSProp | AdaGrad 的累积分母把 LR "饿死" |
| 2014 | Adam | 把"方向（动量）"和"尺度（RMSProp）"合二为一 |
| 2017 | AdamW | Adam + L2 ≠ Adam + weight decay |
| 2023 | Lion | 只用一阶信息但**只看符号**，省掉 $v_t$ |
| 2023 | Sophia | 把 Hessian 对角线塞进分母（廉价二阶） |
| 2024 | Schedule-Free | 不再需要"事先知道总步数" |

下文按这个顺序展开。

# 1. 梯度下降（GD）：原点

给定可微损失 $J(\theta)$，最朴素的更新是

$$
\theta_{t+1} = \theta_t - \eta\,\nabla J(\theta_t)
$$

**收敛性**：若 $J$ 凸且梯度 $L$-Lipschitz，取 $\eta \le 1/L$ 即可线性/次线性收敛到全局最优。

**致命弱点（为后面所有改进埋下伏笔）**：

- 在**病态曲率**（Hessian 条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 大）下，要走到底需要的迭代数随 $\kappa$ 线性增长。一维直觉：$f(\theta)=\frac{1}{2}H\theta^2$ 给出 $\theta_{t+1}=(1-\eta H)\theta_t$，稳定要求 $\eta < 2/H$，即步长被**最陡方向的曲率**死死卡住。
- 一旦最陡方向（$\lambda_{\max}$）和最缓方向（$\lambda_{\min}$）差几个数量级，沿缓方向几乎不动，沿陡方向却来回弹——这就是**"窄长山谷"问题**，下面 Fig 1 左面板能直观看到。

# 2. SGD：噪声的代价与红利

数据量上去之后，全量梯度算不动。SGD 用 mini-batch 估计：

$$
g_t = \nabla J(\theta_t) + \xi_t,\quad \mathbb{E}[\xi_t]=0
$$

噪声 $\xi_t$ 既是诅咒也是祝福：
- **诅咒**：步子大一点就被噪声放大成发散
- **祝福**：噪声有助于**逃离尖锐局部极小**——这后来被 Keskar 等人证明与泛化直接相关（小 batch 的"flat minima"假说）

**Fig 1 中间面板**：SGD 在同一窄谷里的轨迹比 GD 更"毛糙"，但平均方向仍然朝着谷底。

# 3. 动量（Momentum）：把"惯性"还给优化器

直觉：把参数想成**滑下山谷的小球**。GD 等价于"无质量的虫子"——每步只看当前坡度，于是窄谷里左右乱晃。给虫子加点质量，惯性就会**沿着山谷长方向累积**，在垂直窄方向自动相互抵消。

更新规则：

$$
v_t = \gamma v_{t-1} + \eta\,g_t,\qquad \theta_{t+1} = \theta_t - v_t
$$

通常 $\gamma = 0.9$，相当于把过去梯度按几何级数加权——有效"记忆"长度约 $1/(1-\gamma) = 10$ 步。

**关键洞察**：动量会把**有效步长放大**约 $1/(1-\gamma)$ 倍，所以加了动量之后，原来不带动量时能用的 $\eta$ 必须**对应缩小**。这是新手最常踩的坑。

**Fig 1 右面板**：同样的窄谷，动量法的轨迹被"拉直"——垂直方向的振荡互相抵消，水平方向速度持续累积。

![GD / SGD / Momentum 在病态二次型上的轨迹对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig1_gd_sgd_momentum_contour.png)

# 4. Nesterov 加速梯度（NAG）：先看一眼再走

经典动量在**靠近极小值时容易过冲**：因为它用的是当前点的梯度，等下一步发现"哎呀我冲过头了"，下一步才开始调头。

NAG 的修正只有一行：

$$
v_t = \gamma v_{t-1} + \eta\,\nabla J(\theta_t - \gamma v_{t-1}),\qquad \theta_{t+1} = \theta_t - v_t
$$

**区别只在梯度求值的位置**：经典动量在 $\theta_t$ 求梯度，NAG 在"如果只走动量那一步会到的地方" $\theta_t - \gamma v_{t-1}$ 求梯度。

**为什么有效**：相当于带了一步**前瞻**。如果前方坡度变缓，NAG 的梯度提前感知到，自动减速；反之提前加速。Nesterov 在 1983 年证明对凸光滑函数能从 $O(1/t)$ 加速到 $O(1/t^2)$。

![NAG 用前瞻梯度抑制过冲](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig2_nesterov_lookahead.png)

# 5. AdaGrad：每个坐标自己的学习率

到了 2011，NLP 里大量稀疏特征——比如 word2vec 里某个罕见词只在百万样本里出现 5 次。如果对所有参数用同一个 $\eta$：

- 罕见词的参数：每次都用同样的小梯度，$\eta$ 取大了会把它震飞，取小了它永远学不动
- 高频词的参数：梯度大、出现频繁，需要的步长反而小

Duchi 提出 AdaGrad，让**每个坐标根据自己的历史梯度自适应**：

$$
G_t = G_{t-1} + g_t^2 \quad(\text{逐元素})
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t
$$

**直觉**：累计梯度平方大的坐标 → 分母大 → 有效步长小。罕见但梯度突然变大的坐标 → 分母小 → 有效步长大。**自动按"频率"分配 LR**。

**致命问题**：$G_t$ 是**单调递增**的累加和。深度网络要训几十万步，分母最终把所有有效 LR 都压到接近 0，模型"窒息"——这正是 Fig 3 右面板想表达的。

![AdaGrad：自动收缩陡方向，但所有坐标的 LR 单调衰减](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig3_adagrad_per_coord_lr.png)

# 6. RMSProp：把累加和换成指数滑动平均

Hinton 在 2012 年的 Coursera 课件里**只改了一个字**就救活了 AdaGrad：把"累加" $\sum g_t^2$ 换成"指数滑动平均"

$$
E[g^2]_t = \rho\,E[g^2]_{t-1} + (1-\rho)\,g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}\,g_t
$$

通常 $\rho = 0.9$，等价于"只记住最近约 10 步的梯度规模"。

**关键差别**：
- **AdaGrad**：$G_t$ 永远变大 → LR 永远变小（不可逆）
- **RMSProp**：$E[g^2]_t$ 是有限窗口平均 → 当**梯度规模发生变化**时，分母会跟着变 → LR **可以重新放大**

**Fig 4 右面板**直接展示：在 step 60 处梯度规模骤降，AdaGrad 的有效 LR 继续往下走，RMSProp 的有效 LR 立刻**爬升回来**以匹配新规模。

![RMSProp（EMA）vs AdaGrad（累积和）在非平稳梯度规模下](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig4_rmsprop_moving_average.png)

# 7. Adam：把动量和 RMSProp 拼起来

到这里，两条优化主线都成熟了：
- **动量**给的是**好方向**
- **RMSProp** 给的是**好尺度（per-coordinate scaling）**

Kingma & Ba（2014）的 Adam 把它们直接合二为一：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t \quad\text{(一阶矩，即动量)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \quad\text{(二阶矩，即 RMSProp)}
$$

**偏差修正**——这是新手最常忽略的细节：因为 $m_0=v_0=0$，前几步 $m_t, v_t$ 严重偏向 0。修正：

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

默认配置：$\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$。

**为什么 $\beta_2$ 远大于 $\beta_1$**？因为方差估计 $v_t$ 比均值估计 $m_t$ 噪声大得多，需要更长的窗口去平滑。$1/(1-0.999) = 1000$ 步——这就是为什么 Adam 在前 1000 步左右常常需要预热。

![Adam 数据流：动量分支 + RMSProp 分支 → 偏差修正 → 自适应更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig5_adam_combined.png)

# 8. AdamW：被忽略十年的 weight decay

L2 正则 $\frac{\lambda}{2}\|\theta\|^2$ 加到损失里，会在梯度里多出一项 $\lambda\theta$。在**SGD** 里这等价于直接把权重乘以 $(1-\eta\lambda)$——也就是经典的 weight decay。

但 Loshchilov & Hutter（2017）发现：在 **Adam** 里这两件事**不再等价**。原因很直接：Adam 把梯度除以 $\sqrt{\hat v_t}$。如果你把 $\lambda\theta$ 也塞进梯度，它**也被除以** $\sqrt{\hat v_t}$——也就是说**梯度大的参数 weight decay 被自动调小**，这恰恰和正则化想做的事**反着来**。

AdamW 的修正只是把 weight decay **从梯度里抠出来**，直接乘到参数上：

$$
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\,\theta_t
$$

效果：在同样的 $\lambda$ 和 LR 下，AdamW 在 ImageNet/Transformer 上的 generalization gap 显著小于 Adam+L2。这就是为什么 2018 之后**所有大模型预训练默认都是 AdamW**。

![AdamW（解耦）vs Adam+L2（耦合）：weight decay 进入更新公式的位置不同](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig6_adamw_vs_adam.png)

# 9. 2023+ 之后的三条路

AdamW 当了 6 年的"统治者"之后，2023 起出现三个**真正在大模型上被反复验证**的方向。

## 9.1 Lion（Google, 2023）：只看符号

由 AutoML 程序搜索发现的优化器，update 只取**符号**：

$$
m_t = \beta_2 m_{t-1} + (1-\beta_2)\,g_t
$$

$$
\theta_{t+1} = \theta_t - \eta\,\mathrm{sign}\bigl(\beta_1 m_{t-1} + (1-\beta_1)\,g_t\bigr)
$$

**关键性质**：
- **省一半显存**：不用维护 $v_t$（二阶矩），对千亿参数模型是真金白银的省
- **更新幅度恒定 $\eta$**：因为 sign 输出 $\pm 1$。所以 Lion 的 LR **比 AdamW 小约 10 倍**，wd 大约 10 倍
- 在 ViT 和 LLM 预训练上和 AdamW 打平甚至略胜，且训练更快

## 9.2 Sophia（Stanford, 2023）：廉价二阶

Sophia 把对角 Hessian 的廉价估计塞到分母里：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t
$$

$$
h_t \approx \mathrm{diag}(H_t) \quad\text{(Hutchinson 估计，每 } k \text{ 步算一次)}
$$

$$
\theta_{t+1} = \theta_t - \eta\,\mathrm{clip}\!\left(\frac{m_t}{\max(\gamma h_t,\,\varepsilon)},\,1\right)
$$

**核心 trick**：
- 用 $\mathrm{diag}(H)$ 而不是 $g^2$ 当分母——这才是**真正的曲率**
- `clip` 是关键：非凸下 $h_t$ 可能为负，clip 把更新幅度框住保证稳定
- Hessian probe 不是每步都做，所以平均开销可控

报告显示在 GPT-2 规模上能把达到同样困惑度的 wall-clock 减半。

## 9.3 Schedule-Free（Meta, 2024）：摆脱"必须知道总步数"

LR 调度（cosine、WSD 等）的麻烦在于：**必须事先知道总步数**。研究阶段你完全不知道要训多久，提前订调度等于自我束缚。

Schedule-Free AdamW 的核心思想是把"调度"用**迭代平均**替代：

$$
y_t = (1-\beta) z_t + \beta x_t \quad\text{(用于求梯度)}
$$

$$
z_{t+1} = z_t - \eta\,\nabla J(y_t)
$$

$$
x_{t+1} = (1-c_t)\,x_t + c_t\,z_{t+1} \quad\text{(返回的"平均"参数)}
$$

效果：在不需要任何 LR 衰减的前提下，能匹配 cosine schedule 的最终性能；并且**训练中途随时延长不需要重新设计调度**。

![Lion / Sophia / Schedule-Free：2023+ 的三条主流方向](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B-%E4%BB%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%88%B0adam/fig7_modern_optimizers.png)

# 10. 选择指南（实战）

| 场景 | 推荐 | 原因 |
|---|---|---|
| 凸优化 / 简单回归 | GD 或 SGD-momentum | 理论保证强、调参简单 |
| 计算机视觉 baseline | SGD + Nesterov + cosine | 历史上对 ResNet/CNN 收敛点最好 |
| Transformer / LLM 预训练 | **AdamW** + warmup + cosine/WSD | 业界默认，几乎是 free lunch |
| 显存吃紧的大模型 | **Lion** | 节省一阶矩对应的显存（约 1/3） |
| 研究阶段、训练长度未定 | **Schedule-Free AdamW** | 中途可延长，不用重设调度 |
| 想冲 SOTA wall-clock | **Sophia** | 二阶信息加速，但需要工程投入 |

# 11. 五条最容易被忽略的事实

1. **加了动量必须降 LR**：动量把有效步长放大约 $1/(1-\gamma)$ 倍。$\gamma=0.9$ 大约对应 ×10。
2. **Adam 的 $\beta_2 = 0.999$ 决定了你需要 ~1000 步预热**：因为 $v_t$ 在前 1000 步还没"热起来"。
3. **AdamW 的 wd 是和 LR 解耦的**：所以 LR scheduler 衰减 LR 的时候，wd 不会跟着衰减。这是和老的 SGD+L2 工作流的根本差别。
4. **Lion 的 LR 必须比 AdamW 小约一个数量级**：直接照搬 AdamW 的 3e-4 会立即发散。
5. **二阶方法在深度学习里"看起来"一直没普及**：不是因为不好，而是 Hessian 的存储/计算曾经太贵。Sophia 把 $\mathrm{diag}(H)$ 和 Hutchinson 估计组合起来后，这个壁垒第一次被打穿。

# 总结

优化器三十年的演进可以浓缩成两句话：

- **从 GD 到 Adam**：先解决"方向"问题（动量），再解决"尺度"问题（AdaGrad/RMSProp），最后把两者合并并修偏差（Adam）；
- **从 Adam 到 AdamW 之后**：算法层的改进让位给**正则化的细节**（AdamW）、**显存效率**（Lion）、**二阶信息**（Sophia）和**调度自由**（Schedule-Free）。

如果你只能记住一件事：**LLM 时代的默认依然是 AdamW + warmup + cosine/WSD + 梯度裁剪**。在你有非常具体的瓶颈（显存 / wall-clock / 调度灵活性）之前，所有"超过 AdamW"的论文都建议先在自己的任务上**复现一遍 baseline** 再下结论。

# 延伸阅读

- Adam: A Method for Stochastic Optimization（Kingma & Ba, 2014）— [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Decoupled Weight Decay Regularization（Loshchilov & Hutter, 2017）— [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Symbolic Discovery of Optimization Algorithms / Lion（Chen et al., 2023）— [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training（Liu et al., 2023）— [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
- The Road Less Scheduled / Schedule-Free（Defazio et al., 2024）— [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- 进一步看**学习率调度本身**：[学习率：从入门到大模型训练的终极指南](../学习率-从入门到大模型训练的终极指南-2026/)
