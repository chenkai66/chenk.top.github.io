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
aliases:
  - /zh/standalone/优化算法的演进-从梯度下降到adam/
---
为什么训练 ResNet 时大家都说“调 LR 是一门艺术”，而到了 GPT/LLaMA 这一代，几乎所有论文却清一色地写 “AdamW，$\beta_1{=}0.9, \beta_2{=}0.95, \mathrm{wd}{=}0.1$” 就直接翻篇？这绝非偶然——它正是**优化器历经三十年演进后的终点**。
\n本文将沿着一条清晰主线，完整梳理这条演化路径：每一步的出现，都是为了修复前一代算法的**某个具体缺陷**。最后，我们还会深入剖析三条真正于 2023 年后在大模型训练中得到规模化验证的新方向：Lion、Sophia 和 Schedule-Free。

## 你将学到

- 为什么 GD 在病态损失曲面上会“之”字形震荡，以及动量如何从物理直觉上解决这个问题
- Nesterov 的“前瞻”与经典动量在数学上的精确区别
- AdaGrad 为何在稀疏特征上表现出色，又为何在深度网络中最终“窒息”
- RMSProp 如何仅凭一行改动（引入指数滑动平均）就拯救了 AdaGrad
- Adam 如何将动量（方向）与 RMSProp（尺度）巧妙融合，以及为何偏差修正至关重要
- AdamW 与 Adam 的关键差异：一旦引入自适应缩放，“L2 正则等价于 weight decay”这一等式便不再成立
- Lion / Sophia / Schedule-Free：三条真正进入 2023 年后大模型工具箱的前沿方向

## 前置知识

- 基础微积分（梯度、Hessian、泰勒展开）
- 有过训练神经网络的经验（任何框架均可）

---

## 演进总览

| 时间 | 算法 | 解决的核心问题 |
|---|---|---|
| 1847 | GD | 将“沿负梯度方向前进”形式化为迭代公式 |
| 1951 | SGD | 数据集过大，无法计算全批量梯度 |
| 1964 | Momentum | GD 在狭窄山谷中来回震荡 |
| 1983 | NAG | 经典动量在接近极小值时容易过冲 |
| 2011 | AdaGrad | 稀疏特征需要每个坐标拥有独立的学习率 |
| 2012 | RMSProp | AdaGrad 的累积分母会“饿死”学习率 |
| 2014 | Adam | 融合方向（动量）与尺度（RMSProp） |
| 2017 | AdamW | Adam + L2 ≠ Adam + weight decay |
| 2023 | Lion | 舍弃二阶矩，仅使用动量的符号 |
| 2023 | Sophia | 引入廉价的对角 Hessian 预条件器 |
| 2024 | Schedule-Free | 不再需要预先知道总训练步数 |
\n下文将严格遵循此顺序展开。

## 1. 梯度下降（GD）：原点
\n给定一个可微损失函数 $J(\theta)$，最朴素的更新规则为：

$$
\theta_{t+1} = \theta_t - \eta\,\nabla J(\theta_t).
$$

**收敛性**：若 $J$ 是凸函数且其梯度满足 $L$-Lipschitz 条件，则当 $\eta \le 1/L$ 时，算法能以（次）线性速率收敛至全局最小值。

**致命弱点（催生后续所有改进的根源）**：

- 在**病态曲率**（Hessian 条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 很大）的情形下，所需迭代次数随 $\kappa$ 线性增长。一维情形下的直观理解是：对于 $f(\theta)=\frac{1}{2}H\theta^2$，有 $\theta_{t+1}=(1-\eta H)\theta_t$，其稳定条件为 $\eta < 2/H$。这意味着你的步长被**最陡峭方向的曲率**所限制。
- 当最陡方向（$\lambda_{\max}$）与最平缓方向（$\lambda_{\min}$）相差数个数量级时，你在平缓方向几乎寸步难行，却在陡峭方向反复弹跳。这就是著名的**狭窄山谷问题**——如图 1 左侧面板所示。

## 2. SGD：噪声的代价与红利
\n当数据集大到无法装入内存时，我们用小批量估计替代全梯度：

$$\ng_t = \nabla J(\theta_t) + \xi_t,\qquad \mathbb{E}[\xi_t]=0.
$$
\n这里的噪声 $\xi_t$ 既是诅咒也是祝福：
- **诅咒**：稍大的步长会被噪声放大，导致发散。
- **祝福**：噪声有助于**逃离尖锐的局部极小值**——Keskar 等人后来将此现象与“平坦极小值”泛化理论联系起来。

**图 1 中间面板**显示：SGD 在同一山谷中的轨迹比 GD 更“毛糙”，但其平均趋势仍指向谷底。

## 3. 动量（Momentum）：赋予优化器惯性
\n心理模型：将 $\theta$ 视为**滚下山谷的小球**。GD 相当于一只“无质量的虫子”——每一步只感知当前位置的坡度，因此在狭窄方向上来回振荡。若赋予虫子质量与惯性，速度便会**沿山谷长轴持续累积**，而垂直于山谷的振荡则相互抵消。
\n更新规则为：

$$\nv_t = \gamma v_{t-1} + \eta\,g_t,\qquad \theta_{t+1} = \theta_t - v_t.
$$
\n通常取 $\gamma = 0.9$，相当于对历史梯度进行几何加权，有效记忆长度约为 $1/(1-\gamma) = 10$ 步。

**关键洞见**：动量会将**有效步长放大**约 $1/(1-\gamma)$ 倍。因此，启用动量后，必须相应**缩小**此前未使用动量时的学习率。这是新手最常见的陷阱。

**图 1 右侧面板**展示了动量的效果：在相同山谷中，其路径被“拉直”——垂直振荡相互抵消，纵向速度持续累积。

![GD / SGD / Momentum 在病态二次型上的轨迹对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig1_gd_sgd_momentum_contour.png)

## 4. Nesterov 加速梯度（NAG）：先看一眼再走
\n经典动量在接近极小值时容易**过冲**：它在当前位置计算梯度，因此只有在“冲过头”之后的下一步才能意识到错误并开始减速。
\nNAG 仅修改一行代码：

$$\nv_t = \gamma v_{t-1} + \eta\,\nabla J(\theta_t - \gamma v_{t-1}),\qquad \theta_{t+1} = \theta_t - v_t.
$$

**唯一区别在于梯度的计算位置**：经典动量在 $\theta_t$ 处求梯度，而 NAG 在**前瞻点** $\theta_t - \gamma v_{t-1}$（即“仅靠动量一步会到达的位置”）处求梯度。

**为何有效**：这是一种单步前瞻机制。若前方坡度即将变缓，NAG 能提前感知并减速；反之亦然。Nesterov（1983）证明，该方法可将凸光滑优化的收敛速率从 $O(1/t)$ 加速至 $O(1/t^2)$。

![NAG 用前瞻梯度抑制过冲](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig2_nesterov_lookahead.png)

## 5. AdaGrad：每个坐标拥有自己的学习率
\n到 2011 年，NLP 领域已深陷**稀疏特征**的海洋——例如在 word2vec 中，一个罕见词可能在百万样本中仅出现 5 次。若对所有参数使用统一的学习率 $\eta$：

- 罕见词参数：梯度小，若 $\eta$ 过大则会被“震飞”，过小则永远无法学习。
- 高频词参数：梯度大且频繁，反而希望使用**更小**的步长。
\nDuchi 提出 AdaGrad，基于每个坐标的梯度历史进行**逐坐标自适应**：

$$\nG_t = G_{t-1} + g_t^2 \quad(\text{逐元素})
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t.
$$

**直观理解**：累计梯度平方 $g^2$ 越大，分母越大，有效步长越小；罕见但突然出现大梯度的坐标，分母小，有效步长大。**学习率根据频率自动分配**。

**致命缺陷**：$G_t$ 是一个**单调递增的累加和**。在深度网络中训练数十万步后，分母会将所有有效学习率压向零，导致模型“窒息”。图 3 右侧面板清晰地展示了这一点。

![AdaGrad：自动收缩陡方向，但所有坐标的 LR 单调衰减](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig3_adagrad_per_coord_lr.png)

## 6. RMSProp：将累加和替换为指数滑动平均
\nHinton 在 2012 年的 Coursera 课件中**仅做了一处修改**便拯救了 AdaGrad：将累积和 $\sum g_t^2$ 替换为指数滑动平均：

$$\nE[g^2]_t = \rho\,E[g^2]_{t-1} + (1-\rho)\,g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}\,g_t.
$$
\n通常取 $\rho = 0.9$，相当于“记住最近约 10 步的梯度幅度”。

**关键区别**：
- **AdaGrad**：$G_t$ 只增不减 → 学习率持续衰减（不可逆）。
- **RMSProp**：$E[g^2]_t$ 是有限窗口的平均值 → 当**梯度幅度发生变化**时，分母随之调整 → 学习率**可以重新增大**。

**图 4 右侧面板**直接展示了这一效果：在第 60 步，梯度幅度骤降。AdaGrad 的有效学习率继续下降，而 RMSProp 的有效学习率则**迅速回升**以适应新的梯度规模。

![RMSProp（EMA）vs AdaGrad（累积和）在非平稳梯度规模下](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig4_rmsprop_moving_average.png)

## 7. Adam：融合动量与 RMSProp
\n至此，两条优化主线均已成熟：
- **动量**提供了良好的**方向**。
- **RMSProp**提供了良好的**逐坐标尺度**。
\nKingma & Ba（2014）将二者直接拼接，提出了 Adam：

$$\nm_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t \quad\text{(一阶矩 = 动量)}
$$

$$\nv_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \quad\text{(二阶矩 = RMSProp)}
$$

**偏差修正**——这是最常被忽视的细节。由于 $m_0=v_0=0$，初始几步的 $m_t$ 和 $v_t$ 严重偏向零。修正方法为：

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$
\n默认参数：$\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$。

**为何 $\beta_2$ 远大于 $\beta_1$**？因为方差估计比均值估计噪声更大，需要更长的平均窗口。$1/(1-0.999) = 1000$ 步——这也正是 Adam 通常需要约 1000 步预热的原因，以便 $v_t$ “充分热身”。

![Adam 数据流：动量分支 + RMSProp 分支 → 偏差修正 → 自适应更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig5_adam_combined.png)

## 8. AdamW：被忽视十年的 weight decay 问题
\n在损失函数中加入 L2 正则项 $\frac{\lambda}{2}\|\theta\|^2$ 会在梯度中增加 $\lambda\theta$ 项。在**SGD**中，这等价于每步将权重乘以 $(1-\eta\lambda)$——即经典的“weight decay”。
\n但 Loshchilov & Hutter（2017）发现，在**Adam**中，这两种操作**不再等价**。原因很直接：Adam 将梯度除以 $\sqrt{\hat v_t}$。若将 $\lambda\theta$ 并入梯度，它**同样会被除以 $\sqrt{\hat v_t}$**——这意味着**历史梯度较大的参数会受到更弱的 weight decay**，而这恰恰与正则化的目标背道而驰。
\nAdamW 的解决方案是将 weight decay **从梯度中分离**，直接作用于参数：

$$
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\,\theta_t.
$$
\n效果显著：在相同的 $\lambda$ 和学习率下，AdamW 在 ImageNet/Transformer 上的泛化差距明显小于 Adam+L2。这正是为何**2018 年后所有大模型预训练都默认采用 AdamW**。

![AdamW（解耦）vs Adam+L2（耦合）：weight decay 进入更新公式的位置不同](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig6_adamw_vs_adam.png)

## 9. 2023 年后的前沿：三条经受住规模化考验的方向
\n在 AdamW 主导约 6 年后，自 2023 年起，有三个方向真正**在大规模模型上证明了自身价值**。

## 9.1 Lion（Google, 2023）：仅保留符号
\n由 AutoML 程序搜索发现，其更新规则**仅保留符号**：

$$\nm_t = \beta_2 m_{t-1} + (1-\beta_2)\,g_t
$$

$$
\theta_{t+1} = \theta_t - \eta\,\mathrm{sign}\bigl(\beta_1 m_{t-1} + (1-\beta_1)\,g_t\bigr).
$$

**显著特性**：
- **优化器状态减半**：无需维护 $v_t$——这对千亿参数模型意味着真金白银的节省。
- **更新幅度恒为 $\eta$**：因为 sign 函数输出 $\pm 1$。因此 Lion 的学习率必须**比 AdamW 小约 10 倍**，而 weight decay 则需大 10 倍。
- 在 ViT 和 LLM 预训练任务上，性能与 AdamW 持平甚至略优，且训练速度更快。

## 9.2 Sophia（Stanford, 2023）：廉价的二阶信息
\nSophia 将廉价的对角 Hessian 估计引入分母：

$$\nm_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t
$$

$$\nh_t \approx \mathrm{diag}(H_t) \quad\text{(每 } k \text{ 步使用 Hutchinson 估计)}
$$

$$
\theta_{t+1} = \theta_t - \eta\,\mathrm{clip}\!\left(\frac{m_t}{\max(\gamma h_t,\,\varepsilon)},\,1\right).
$$

**核心技巧**：
- 使用 $\mathrm{diag}(H)$ 而非 $g^2$ 作为分母——这才是**真实的曲率**。
- `clip` 至关重要：在非凸损失中 $h_t$ 可能为负，裁剪操作确保更新有界。
- Hessian 探测并非每步执行，因此摊销成本可控。
\n报告结果表明：在 GPT-2 规模上，达到相同困惑度所需的训练时间大约减半。

## 9.3 Schedule-Free（Meta, 2024）：摆脱调度依赖
\n学习率调度（如 cosine、WSD 等）有一个共同痛点：**必须预先知道总训练步数**。而在研究阶段，我们通常无法确定训练时长，提前绑定调度方案会束缚手脚。
\nSchedule-Free AdamW 用**迭代平均**替代显式调度：

$$\ny_t = (1-\beta) z_t + \beta x_t \quad\text{(用于计算梯度的点)}
$$

$$\nz_{t+1} = z_t - \eta\,\nabla J(y_t)
$$

$$\nx_{t+1} = (1-c_t)\,x_t + c_t\,z_{t+1} \quad\text{(返回的“平均”参数)}
$$
\n效果显著：在**无需任何显式衰减**的情况下，即可匹配 cosine 调度的最终性能，并且**可在训练中途任意延长**，无需重新设计调度策略。

![Lion / Sophia / Schedule-Free：2023+ 的三条主流方向](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/优化算法的演进-从梯度下降到adam/fig7_modern_optimizers.png)

## 10. 选择指南

| 场景 | 推荐 | 原因 |
|---|---|---|
| 凸优化 / 简单回归 | GD 或 SGD-momentum | 理论保障强，调参简单 |
| 计算机视觉基线 | SGD + Nesterov + cosine | 在 ResNet/CNN 上历史最优 |
| Transformer / LLM 预训练 | **AdamW** + warmup + cosine/WSD | 业界默认，近乎免费的午餐 |
| 显存受限的大模型 | **Lion** | 节省一阶矩状态（约 1/3 显存） |
| 研究阶段、训练长度未知 | **Schedule-Free AdamW** | 可中途延长，无需重设调度 |
| 追求训练速度 SOTA | **Sophia** | 利用二阶信息加速，但工程成本较高 |

## 11. 最常被忽略的五个事实

1. **启用动量后务必降低学习率**。动量会将有效步长放大 $1/(1-\gamma)$ 倍。当 $\gamma=0.9$ 时，放大倍数约为 10 倍。
2. **Adam 的 $\beta_2 = 0.999$ 意味着约需 1000 步预热**，因为在此之前的 $v_t$ 尚未“热身”完毕。
3. **AdamW 的 weight decay 与学习率解耦**。当学习率调度器衰减学习率时，weight decay **不会随之衰减**。这是与传统 SGD+L2 工作流的根本区别。
4. **Lion 的学习率必须比 AdamW 小约一个数量级**。直接套用 AdamW 的 `3e-4` 会导致立即发散。
5. **二阶方法曾被认为“永久不实用”**，并非因其效果不佳，而是因为 Hessian 的计算与存储成本过高。Sophia 通过结合 $\mathrm{diag}(H)$ 与廉价的 Hutchinson 估计，首次打破了这一壁垒。

## 12. 优化器状态显存：数学公式隐藏的成本
\n动量、AdaGrad、RMSProp 和 Adam 的优雅推导从不提及显存。然而在生产环境中，这往往是首要约束。对于一个含 $P$ 个可训练参数的 fp16 模型：

| 优化器 | 每参数状态（fp32） | 7 B 参数 | 70 B 参数 |
|---|---|---|---|
| SGD | 0 | 0 | 0 |
| SGD + momentum | 4 字节 | 28 GB | 280 GB |
| Adam / AdamW | 8 字节（$m, v$） | 56 GB | 560 GB |
| Lion | 4 字节（仅 $m$） | 28 GB | 280 GB |
| Sophia | 8 字节（$m, h$） | 56 GB | 560 GB |
| Adafactor | ~2 字节（分解） | 14 GB | 140 GB |
| 8-bit AdamW (`bnb`) | 2 字节 | 14 GB | 140 GB |
\n这张表才真正决定了你能运行哪个优化器。一个 fp16 的 7B 模型，权重占 14 GB，梯度占 14 GB，而 AdamW 还需额外 56 GB，轻松超过单张 A100 80 GB 的显存上限。同一模型若使用 8-bit AdamW 或 Adafactor 则能轻松容纳。这也解释了为何 LLM 预训练社区对优化器状态量化有着古典机器学习时代从未有过的执念：算法本身没问题，瓶颈在于显存。
\n一个实用经验法则：若优化器状态超过模型权重显存的 1.5 倍，你很可能需要在以下三者中选择其一——对优化器进行分片（ZeRO-1）、量化（`bitsandbytes`），或切换至状态更轻量的算法（如 Lion、Adafactor）。具体选择取决于你的瓶颈是单卡显存还是集群总显存。


![各优化器的状态显存对比，以及在 7B / 70B 规模下的总显存占用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/03-gradient-descent-family/fig8_optimizer_memory.png)

## 13. 混合精度：优化器眼中的 fp32 世界
\n一个常令初次预训练者措手不及的细节是：即便训练的其余部分使用 fp16/bf16，优化器几乎总是运行在 fp32 中。原因在于 Adam 的指数滑动平均需在数千步上累积。以 $\beta_2 = 0.999$ 为例，$v_t$ 中最小贡献项仅为最大项的 $10^{-3}$，已低于 fp16 的可表示范围（约 $6 \times 10^{-5}$ 至 $6 \times 10^4$）。
\n标准流程如下：

1. 前向与反向传播使用 fp16/bf16，梯度为 fp16/bf16。
2. 在优化器步骤前，将梯度**转换为 fp32**。
3. 优化器状态（$m, v$、权重）以 fp32 存储。
4. 更新后，将权重转回 fp16/bf16 用于下一次前向传播。
\nPyTorch 的 `torch.amp` 和 DeepSpeed 均自动执行此流程。这也解释了前述“显存成本”表为何以 fp32 字节计算——即使模型为 fp16，优化器仍为每个参数支付 fp32 的开销。
\nbf16 略有不同：其动态范围足够宽，有时可全程保持梯度为 bf16，但在所有值得使用的实现中，优化器状态仍为 fp32。在大规模 LLM 预训练中，这一点无可妥协。


![混合精度训练的数据流：哪些张量在 fp16/bf16，哪些必须在 fp32](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/03-gradient-descent-family/fig9_mixed_precision.png)

## 14. 各优化器在 Transformer 基线上的学习率合理范围
\n以下是我启动 Transformer 类任务时首先尝试的学习率区间。请将其视为贝叶斯先验，而非最终答案。

| 优化器 | LR 范围 | 备注 |
|---|---|---|
| SGD + momentum | 0.01 – 0.5 | 线性 warmup，约占总步数的 5% |
| AdamW | 1e-4 – 6e-4 | 大多数 LLM 落在 3e-4 ± 1.5 倍 |
| Lion | 1e-5 – 6e-5 | 约为 AdamW 的 1/10 |
| Sophia | 1e-4 – 4e-4 | 也略低于 AdamW |
| Schedule-Free AdamW | 1e-4 – 6e-4 | 与 AdamW 相同；变化的是**调度策略** |
| Adafactor | 相对步长 0.01 – 0.05 | 使用相对步长，绝对 LR 无意义 |
\n这些仅是初始先验。实际最优值取决于 batch size（SGD 遵循线性缩放律，Adam 约为 $\sqrt{}$ 缩放律）、warmup 长度及数据噪声。但若初始值超出上述范围，几乎可以肯定是配置错误，而非重大发现。

## 7. 总结
\n三十年的优化器演进可浓缩为两句话：

- **从 GD 到 Adam**：先解决**方向**问题（动量），再解决**尺度**问题（AdaGrad/RMSProp），最后将二者融合并修正偏差（Adam）。
- **从 Adam 开始**：算法层面的改进让位于**正则化细节**（AdamW）、**显存效率**（Lion）、**二阶信息**（Sophia）和**调度自由**（Schedule-Free）。
\n若你只记住一点：**LLM 时代的默认配置仍是 AdamW + warmup + cosine/WSD + 梯度裁剪**。在你面临明确瓶颈（显存、训练时长或调度灵活性）之前，任何声称超越 AdamW 的论文，都应在你的任务上**复现基线**后再做定论。

## 延伸阅读

- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014) — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017) — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Symbolic Discovery of Optimization Algorithms / Lion (Chen et al., 2023) — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training (Liu et al., 2023) — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
- The Road Less Scheduled / Schedule-Free (Defazio et al., 2024) — [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- 关于**学习率调度**的深入探讨：[学习率：从入门到大模型训练的终极指南](./04-learning-rate-schedules/)
