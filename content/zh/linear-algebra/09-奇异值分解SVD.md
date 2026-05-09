---
title: "线性代数（九）：奇异值分解 SVD"
date: 2025-02-26 09:00:00
tags:
  - Linear Algebra
  - SVD
  - Singular Value Decomposition
  - PCA
  - Image Compression
  - Dimensionality Reduction
description: "SVD 被誉为线性代数的皇冠明珠：它能分解任意矩阵，不限于方阵或对称矩阵。从图像压缩到推荐系统，从人脸识别到基因分析，SVD 无处不在。"
categories: 线性代数
series: linear-algebra
lang: zh
mathjax: true
disableNunjucks: true
series_order: 9
translationKey: "linear-algebra-9"
polished_by_qwen_max: true
---
![线性代数（九）：奇异值分解 SVD — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/illustration_1.png)

## 一、为什么 SVD 配得上“皇冠”二字

第 8 章的谱定理给出了 $A = Q\Lambda Q^{\!\top}$，简洁优美，但有个硬性限制：$A$ 必须是对称矩阵。现实中的矩阵大多不对称，甚至不是方阵：

- 一张 $1920 \times 1080$ 的图片；
- 用户—电影评分矩阵（百万行，数千列）；
- 文档—词项矩阵（NLP 中文档数 × 词汇量）；
- 基因表达矩阵（生物信息学）。

**奇异值分解（SVD）** 能处理所有这些情况。对**任意** $m \times n$ 矩阵 $A$，
$$A = U\,\Sigma\,V^{\!\top}.$$这是线性代数里最强大、最通用的分解方法。

### 一个摄影类比

把照片看成像素矩阵，SVD 告诉我三件事：

1. **任何照片都可以拆解为若干“基础层”的叠加**。
2. **这些层按重要性排序**：第一层抓主结构，第二层抓次要细节，第三层抓更精细的部分。
3. **只保留前几层就能还原大部分图像信息**。

这就像乐队录音由主唱、吉他、贝斯、鼓叠加而成：去掉背景和声，歌曲还能听；去掉主唱，整首歌就垮了。SVD 把这种直觉量化了：奇异值精确衡量了每一层的贡献。

### 本章学习目标

- 掌握 SVD 的定义及其**三步几何意义**：旋转 → 拉伸 → 旋转。
- 学会通过 $A^{\!\top}\!A$ 和 $AA^{\!\top}$ 计算奇异值和奇异向量。
- 从 $U$ 和 $V$ 直接读出**四个基本子空间**。
- 理解**低秩逼近**与 Eckart–Young 定理——数据压缩背后的最优性原理。
- 掌握**伪逆**：一种适用于不可逆或非方阵的“最佳逆”。
- 发现 PCA 其实就是 SVD 的另一种形式。
- 应用领域：图像压缩、推荐系统、潜在语义分析、信号去噪、特征脸。

### 前置知识

- 特征值与特征向量（第 6 章）
- 正交矩阵与投影（第 7 章）
- 对称矩阵与谱定理（第 8 章）

---
## 二、SVD 的定义

### 基本定理

**SVD 定理**：任何 $m \times n$ 的实矩阵 $A$ 都可以分解为：$$A = U\,\Sigma\,V^{\!\top}$$其中：

- $U \in \mathbb{R}^{m\times m}$ 是正交矩阵，列向量是**左奇异向量** $u_1, \ldots, u_m$；
- $V \in \mathbb{R}^{n\times n}$ 是正交矩阵，列向量是**右奇异向量** $v_1, \ldots, v_n$；
- $\Sigma \in \mathbb{R}^{m\times n}$ 是对角矩阵，主对角线上是**奇异值** $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$，其余位置全为零。

实际应用中，尤其是处理高瘦矩阵时，常用"经济型"形式，只保留 $r = \operatorname{rank}(A)$ 个非零奇异值：$$A = U_r\,\Sigma_r\,V_r^{\!\top},\qquad U_r\in\mathbb{R}^{m\times r},\ \Sigma_r\in\mathbb{R}^{r\times r},\ V_r\in\mathbb{R}^{n\times r}.$$
奇异值有三个重要特性：

- 它们永远是非负实数（特征值可能是负数或复数）。
- 按惯例从大到小排列。
- 任何矩阵都有 SVD。这是特征分解做不到的，也是 SVD 的核心优势。

### 几何意义：三步完成变换

分解 $A = U\Sigma V^{\!\top}$ 可以看作一个三步过程，从右往左依次作用：

1. **旋转** $V^{\!\top}$：把输入对齐到 $A$ 的自然方向。
2. **拉伸** $\Sigma$：第 $i$ 个坐标乘以 $\sigma_i$。
3. **旋转** $U$：把结果转到输出空间的最终位置。

用揉面团来比喻：先把面团转到合适的角度（$V^{\!\top}$），再用擀面杖压扁拉长（$\Sigma$），最后把擀好的面饼摆到需要的方向（$U$）。

![SVD 三步：先旋转 V^T，再拉伸 Σ，最后旋转 U](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig1_svd_geometry.png)

图中跟踪了一个单位圆的变化过程。两次旋转保持了正交基的性质，只有中间的对角阵 $\Sigma$ 改变了形状，把圆变成了半轴长度为 $\sigma_1, \sigma_2$ 的椭圆。

### 单位圆变椭圆

还可以用一幅图浓缩整个过程：左边是单位圆，右边是 $A$ 作用后的椭圆。

![右奇异向量 v_i 被映射为 σ_i u_i，正好是椭圆的主轴](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig2_circle_to_ellipse.png)

看图要点：

- **右奇异向量** $v_1, v_2$：输入侧的正交方向，被映射到椭圆的主轴上。
- **左奇异向量** $u_1, u_2$：输出侧主轴的正交方向。
- **奇异值** $\sigma_1, \sigma_2$：椭圆的半轴长度。

一句话总结：$$A\,v_i \;=\; \sigma_i\, u_i.$$这就是 SVD 的核心公式，所有结论都从这里展开。

### 外积展开

SVD 还有一种等价写法，就是把矩阵表示成秩 1 矩阵的加权和：$$A = \sigma_1 u_1 v_1^{\!\top} + \sigma_2 u_2 v_2^{\!\top} + \cdots + \sigma_r u_r v_r^{\!\top}.$$每个 $u_i v_i^{\!\top}$ 是秩为 1 的矩阵，奇异值是权重。这种视角是低秩逼近的关键：保留权重大的项，舍弃权重小的项。
## 三、SVD 的计算

### 桥梁：$A^{\!\top}\!A$ 和 $AA^{\!\top}$

两个对称矩阵的特征向量分别给出 $V$ 和 $U$。把 $A = U\Sigma V^{\!\top}$ 从两边展开：$$A^{\!\top}\!A = V\,\Sigma^{\!\top}\!\Sigma\,V^{\!\top}, \qquad AA^{\!\top} = U\,\Sigma\Sigma^{\!\top}\,U^{\!\top}.$$
$A^{\!\top}\!A$ 和 $AA^{\!\top}$ 都是对称半正定矩阵，因此可以用谱定理分析。从谱分解的角度看：

- $V$ 的列是 $A^{\!\top}\!A$ 的正交特征向量（称为**右奇异向量**）。
- $U$ 的列是 $AA^{\!\top}$ 的正交特征向量（称为**左奇异向量**）。
- $\sigma_i = \sqrt{\lambda_i}$，其中 $\lambda_i$ 是这两个矩阵共有的非负特征值。

**为什么用 $A^{\!\top}\!A$？** 可以理解为 $A$ "作用两次"：先通过 $A$ 前进，再通过 $A^{\!\top}$ 回退。这一来回会让某个方向被放大 $\sigma^2$ 倍，所以 $A^{\!\top}\!A$ 的特征值正好是奇异值的平方。

### 计算步骤

假设 $A$ 是 $m \times n$ 矩阵，且 $m \ge n$：

1. 构造 $A^{\!\top}\!A$，计算其特征值 $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$ 和对应的正交单位特征向量 $v_1, \ldots, v_n$。这些向量组成 $V$ 的列。
2. 设 $\sigma_i = \sqrt{\lambda_i}$。
3. 对每个 $\sigma_i > 0$，定义 $u_i = A v_i / \sigma_i$。这些向量构成 $U$ 的前 $r$ 列。
4. 如果 $r < m$，用 Gram–Schmidt 方法将 $\{u_1, \ldots, u_r\}$ 扩展为 $\mathbb{R}^m$ 的正交基，补齐 $U$。

实际数值计算中，没人会这么算 —— 显式构造 $A^{\!\top}\!A$ 会让条件数平方化，导致精度损失。生产代码通常使用双对角化结合 QR 算法或分治法。上面的推导只是为了说明原理。

### 一个手算例子

设 $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$：
$$A^{\!\top}\!A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}, \qquad \det(A^{\!\top}\!A - \lambda I) = \lambda^2 - 3\lambda + 1.$$
解得 $\lambda = \frac{3 \pm \sqrt{5}}{2}$，于是$$\sigma_1 = \sqrt{\tfrac{3+\sqrt 5}{2}} \approx 1.618, \qquad \sigma_2 = \sqrt{\tfrac{3-\sqrt 5}{2}} \approx 0.618.$$
求出 $A^{\!\top}\!A$ 的特征向量得到 $V$，再通过 $u_i = A v_i / \sigma_i$ 得到 $U$。验证一下：$\sigma_1 \sigma_2 = 1 = |\det A|$，结果自洽。

### 特征值 vs 奇异值：直观对比

对称矩阵的特征值和奇异值（最多差正负号）是一回事。但对一般的 $A$，两者完全不同，对比起来很有启发性。

![特征向量不正交，奇异向量正交；特征值描述不变方向，奇异值描述拉伸](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig3_eig_vs_svd.png)

| | 特征向量 | 奇异向量 |
|---|---|---|
| 定义 | $A x = \lambda x$ | $A v = \sigma u$，且 $u, v$ 各自正交 |
| 正交性 | 一般不正交 | $V$ 和 $U$ 都是正交矩阵 |
| 取值 | $\lambda_i \in \mathbb{C}$ | $\sigma_i \in \mathbb{R}_{\ge 0}$ |
| 几何含义 | 不变方向，按 $\lambda$ 缩放 | 输入正交基映射到输出正交基，按 $\sigma$ 缩放 |
| 存在性 | 方阵，且未必可对角化 | 任意矩阵都存在 |

图中同一个非对称 $A$ 的特征向量呈斜角，而 SVD 在输入和输出两侧都给出了垂直的方向。
## SVD 与四个基本子空间

SVD 能最清晰地揭示矩阵的结构。假设 $A \in \mathbb{R}^{m \times n}$，秩为 $r$：

| 子空间 | 维度 | SVD 提供的正交基 | 所在空间 |
|---|---|---|---|
| 行空间 $\mathcal{C}(A^{\!\top})$ | $r$ | $v_1, \ldots, v_r$ | $\mathbb{R}^n$ |
| 零空间 $\mathcal{N}(A)$ | $n - r$ | $v_{r+1}, \ldots, v_n$ | $\mathbb{R}^n$ |
| 列空间 $\mathcal{C}(A)$ | $r$ | $u_1, \ldots, u_r$ | $\mathbb{R}^m$ |
| 左零空间 $\mathcal{N}(A^{\!\top})$ | $m - r$ | $u_{r+1}, \ldots, u_m$ | $\mathbb{R}^m$ |

行空间的正交基 $\{v_1, \ldots, v_r\}$ 映射到列空间的正交基 $\{u_1, \ldots, u_r\}$，每个方向按 $\sigma_i$ 拉伸。零空间的所有内容都被压缩到零点。这就是矩阵 $A$ 的全部作用。

---
## 五、低秩逼近：压缩背后的定理

![线性代数（九）：奇异值分解 SVD — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/illustration_2.png)

### Eckart–Young 定理

把外积展开截断到前 $k$ 项：$$A_k = \sigma_1 u_1 v_1^{\!\top} + \cdots + \sigma_k u_k v_k^{\!\top}.$$
**定理（Eckart–Young, 1936）**：在所有秩不超过 $k$ 的矩阵 $B$ 中，$$\|A - A_k\|_F \;=\; \min_{\operatorname{rank}(B) \le k} \|A - B\|_F \;=\; \sqrt{\sigma_{k+1}^{\,2} + \cdots + \sigma_r^{\,2}}.$$用算子（2-）范数衡量时，$\|A - A_k\|_2 = \sigma_{k+1}$。

所以，$A_k$ 不仅是一个低秩近似，而且是**理论上最优**的。任何其他秩为 $k$ 的矩阵都无法比它更接近原矩阵。

**类比 MP3**：MP3 压缩会丢弃人耳难以察觉的高频成分。SVD 截断对矩阵做了类似的事：去掉能量最小的部分，保留最重要的信息。

### 一层一层叠加

把每一层单独画出来更容易理解。每个 $\sigma_i u_i v_i^{\!\top}$ 都是一张秩为 1 的图像，逐步叠加后越来越接近原图。

![秩 1 图层依次相加重建图像；前几层占据了大部分能量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig4_low_rank_blocks.png)

上排是前三层秩 1 图像（红色为正，蓝色为负）以及原图；下排是叠加 1 层、2 层、3 层后的结果，右侧是奇异值柱状图。仅仅三层就能看出图像的主要结构。

### 能量视角

矩阵的"能量"可以用 Frobenius 范数的平方表示：$$\|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2.$$秩 $k$ 近似保留的能量比例为：$$\text{能量比例} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2}.$$对于大多数自然数据，奇异值衰减得很快。比如一张 $1000 \times 1000$ 的照片，前 50 个奇异值通常能保留 95% 的能量。

### 图像压缩的实际效果

存储一个秩 $k$ 近似需要保存 $k$ 个奇异值，再加上 $U$ 和 $V$ 的前 $k$ 列：$$\text{需存数字数} = k\,(m + n + 1).$$以 $500 \times 500$ 图像为例，取 $k = 50$：原图需要 $250{,}000$ 个数字，压缩后只需 $50{,}050$ 个 —— 压缩了 5 倍，但肉眼几乎看不出差异。

![原图与 k=5, 20, 50 的对比；奇异值谱与累积能量曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig5_image_compression.png)

左下角的对数坐标奇异值衰减曲线是应用 SVD 时最关键的诊断工具。它告诉我可以大胆截断到什么程度而不会让质量崩塌。

```python
import numpy as np

def compress(img, k):
    """对 2D 数组做秩 k SVD 近似"""
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# 诊断：每个 k 保留多少能量？
U, s, Vt = np.linalg.svd(img, full_matrices=False)
for k in [5, 20, 50, 100]:
    energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
    print(f"k={k:>3}: 能量保留 {energy:.1f}%")
```
## 六、伪逆

### 逆矩阵不存在时怎么办？

解方程 $A x = b$ 时，通常希望得到 $x = A^{-1} b$。但 $A^{-1}$ 只有在 $A$ 是方阵且满秩时才存在。这时，**Moore–Penrose 伪逆** $A^{+}$ 就成了一个通用的“最佳替代方案”。

### 用 SVD 定义伪逆

如果 $A = U \Sigma V^{\!\top}$，那么$$A^{+} = V \Sigma^{+} U^{\!\top},\qquad
\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i, & \sigma_i > 0,\\ 0, & \sigma_i = 0,\end{cases}$$其中 $\Sigma^{+}$ 被转置为 $n \times m$ 的形状。如果 $A$ 可逆，直接有 $A^{+} = A^{-1}$。

### 伪逆的作用

对任意 $b$，$\hat x = A^{+} b$ 满足以下两点：

1. **最小二乘解**：让 $\|A x - b\|_2$ 最小；
2. 在所有最小二乘解中，$\|x\|_2$ 最小。

两种常见情况：

- **超定**（$m > n$）：通常没有精确解，$A^{+} b$ 提供最小二乘拟合；
- **欠定**（$m < n$）：解有无穷多个，$A^{+} b$ 选范数最小的那个。

![伪逆几何：A^+ b 是 b 在 col(A) 上的投影，残差与 col(A) 正交](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig6_pseudoinverse.png)

左图是最小二乘直线拟合的经典例子；右图揭示了“最小二乘”背后的几何原理 —— $b$ 投影到 $A$ 的列空间，残差与列空间正交。这种正交性正是**正规方程**的核心，而 SVD 直接给出了结果。

```python
import numpy as np

# 用伪逆实现最小二乘拟合 y = a x + b
x = np.linspace(-2, 4, 25)
y = 1.2 * x + 0.4 + np.random.default_rng(0).normal(0, 1, 25)

A = np.column_stack([x, np.ones_like(x)])
coef = np.linalg.pinv(A) @ y          # 内部使用 SVD
print(f"斜率={coef[0]:.3f}  截距={coef[1]:.3f}")
```
## 七、用 SVD 实现 PCA

### 关系

PCA 就是披着统计外衣的 SVD。

先把数据矩阵 $X \in \mathbb{R}^{n\times p}$ 中心化，也就是每列减去均值，记为 $X_c$。然后对它做 SVD 分解：$$X_c = U \Sigma V^{\!\top}.$$
结果如下：

- **主成分方向**：就是 $V$ 的列（右奇异向量）。这些方向是方差最大的正交轴。
- **主成分得分**：计算公式是 $X_c V = U \Sigma$。这是数据在新基下的表示。
- **第 $i$ 个主成分的方差**：等于 $\sigma_i^2 / (n - 1)$。
- **降到 $k$ 维**：取前 $k$ 个主成分，公式是 $X_k = U_k \Sigma_k$。

### 为什么 PCA 有效

第一主成分的方向，是在单位向量 $w$ 中让 $\operatorname{Var}(X_c w)$ 最大化的方向。简单推导一下，这个目标可以写成 $w^{\!\top}\!(X_c^{\!\top} X_c)\,w / (n-1)$。最大化这个表达式的结果，正好是 $X_c^{\!\top} X_c$ 的最大特征向量，也就是右奇异向量 $v_1$。

![中心化数据 + 主成分轴；PC1 方向上的得分直方图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig7_pca_via_svd.png)

虚线椭圆是 1-$\sigma$ 高斯拟合。两个箭头是按标准差 $\sigma_i / \sqrt{n-1}$ 缩放的主成分轴。把数据投影到 PC1 上，从 2 维降到 1 维，同时保留了绝大部分方差。

```python
import numpy as np

def pca(X, k):
    """用 SVD 实现 PCA，返回 (得分, 主成分方向, 解释方差比例)。"""
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                    # k x p
    scores = Xc @ components.T             # n x k
    var = (s ** 2) / (X.shape[0] - 1)
    return scores, components, (var / var.sum())[:k]
```
## 八、推荐系统

### 问题背景

Netflix、亚马逊和Spotify都面临同一个核心问题：**怎么预测用户对没接触过的物品的评分？** 用户—物品评分矩阵 $R \in \mathbb{R}^{m\times n}$ 规模庞大，但大部分数据缺失。

### 矩阵分解

我的建模假设是，评分由少数几个**潜在因子**决定。比如对电影来说，这些因子可能是“动作强度”、“浪漫指数”、“幽默程度”或者“艺术深度”。基于这个假设，可以写出：$$R \approx U_k \Sigma_k V_k^{\!\top}.$$

- $U_k \Sigma_k$ 的每一行表示一个用户的偏好向量。
- $V_k$ 的每一行表示一个物品的特征向量。
- 预测评分就是这两个向量的点积。

具体做法是：先用已知的评分数据拟合出 $U_k$ 和 $V_k$，再用它们计算未知评分。这种方法正是Netflix Prize（2006–2009）冠军方案的核心思想。

---
## 九、SVD 与特征分解的对比

| 性质 | 特征分解 | SVD |
|---|---|---|
| 适用对象 | 方阵（通常对称） | 任意矩阵 |
| 形式 | $A = P\Lambda P^{-1}$ | $A = U\Sigma V^{\!\top}$ |
| 取值 | 特征值，可正可负也可复数 | 奇异值，非负实数 |
| 向量 | 特征向量，不一定正交 | 奇异向量，始终正交 |
| 几何意义 | 不变方向 + 缩放 | 旋转 + 拉伸 + 再旋转 |
| 是否总是存在 | 否 | 是 |

### 为什么 SVD 被称为“皇冠明珠”

- **普适性强**：无论方阵还是非方阵，满秩还是降秩，都能用。
- **数值稳定**：计算上非常可靠，是求秩、判断条件数和最小二乘问题的黄金标准。
- **理论最优**：提供低秩逼近的最优解（Eckart–Young 定理保证）。
- **洞察深刻**：一次分解就能揭示矩阵的秩、四个子空间以及算子范数。
- **应用广泛**：图像压缩、NLP、推荐系统、去噪、控制系统、统计分析等领域都离不开它。

---
## 十、其他应用

### 潜在语义分析（LSA）

先构建一个文档—词项矩阵，行代表文档，列代表词汇，矩阵中的值是 TF-IDF 分数。对这个矩阵做 SVD 分解，保留前 $k$ 个分量。右奇异向量可以看作“潜在主题”，而文档之间的相似度就变成了低维空间中的余弦相似度。

### 信号去噪

假设信号是低秩的，噪声是满秩的。对观测数据做 SVD 分解后，保留较大的奇异值，丢弃较小的奇异值，然后重构信号。这种方法从天文图像处理到地震数据分析都在用，非常实用。

### 特征脸（Eigenfaces）

把一组对齐的人脸图像输入 PCA，得到的主成分就是所谓的“特征脸”。这些特征脸构成了人脸的典型表示。任何新的人脸都可以用特征脸的线性组合来表示，识别过程就简化为比较系数向量的距离。
## Python 实现

### 用特征分解手动实现 SVD

```python
import numpy as np

def svd_via_eigen(A):
    """通过 A^T A 的特征分解来概念性地计算 SVD。
    实际项目中请用 np.linalg.svd，这里仅用于教学。"""
    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)            # 特征值按升序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[idx], V[:, idx]

    sigma = np.sqrt(np.maximum(eigvals, 0.0))
    r = int((sigma > 1e-10).sum())
    U = (A @ V[:, :r]) / sigma[:r]              # 按列广播
    return U, sigma[:r], V[:, :r].T

A = np.array([[3.0, 2.0], [2.0, 3.0]])
U, s, Vt = svd_via_eigen(A)
print("奇异值:", s)
print("重建:\n", U @ np.diag(s) @ Vt)
```

### 图像压缩演示

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_compression(img, ks):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("原图"); axes[0].axis("off")
    for ax, k in zip(axes[1:], ks):
        rec = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
        energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
        ax.imshow(rec, cmap="gray")
        ax.set_title(f"k={k}  ({energy:.0f}%)"); ax.axis("off")
    plt.tight_layout(); plt.show()

# plot_compression(your_grayscale_image, [5, 20, 50, 100])
```
## 练习题

### 基础练习

1. 奇异值为什么总是非负的？而特征值却可以是负数或复数？
2. 手动计算矩阵 $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ 的 SVD。
3. 如果矩阵 $A$ 是 $3 \times 5$ 的，完整 SVD 中 $U$、$\Sigma$、$V^{\!\top}$ 的形状分别是什么？经济型 SVD 中呢？

### 深入思考

4. 证明矩阵 $A$ 的秩 $\operatorname{rank}(A)$ 等于非零奇异值的数量。
5. 证明矩阵的 Frobenius 范数平方满足 $\|A\|_F^2 = \sigma_1^2 + \cdots + \sigma_r^2$。
6. 如果 $Q$ 是正交矩阵，它的奇异值是什么？原因是什么？
7. 证明 $U_r U_r^{\!\top}$ 是列空间的投影矩阵，$V_r V_r^{\!\top}$ 是行空间的投影矩阵。
8. 证明 Eckart–Young 定理的算子范数形式：$\|A - A_k\|_2 = \sigma_{k+1}$。

### 编程挑战

9. **压缩曲线**  
   加载一张灰度图像，计算其 SVD。对 $k \in \{5, 20, 50, 100\}$，绘制秩为 $k$ 的重建图像，并在对数坐标下绘制奇异值衰减曲线。
   
10. **Iris 数据集上的 PCA**  
    对 Iris 数据集应用 PCA，绘制前两个主成分的散点图（按物种着色），并报告各主成分的解释方差比例。

11. **简单推荐系统**  
    构造一个 $5 \times 10$ 的评分矩阵，其中部分值缺失。用每行的均值填充缺失值，进行秩为 3 的 SVD 分解，观察预测评分的结果是否合理。
## 十三、本章小结

| 概念 | 核心公式 | 直观理解 |
|---|---|---|
| SVD | $A = U\Sigma V^{\!\top}$ | 旋转 + 拉伸 + 再旋转 |
| 奇异值 | $\sigma_i \ge 0$ | 椭圆主轴的拉伸倍数 |
| 外积形式 | $A = \sum_i \sigma_i u_i v_i^{\!\top}$ | 秩为 1 的分量加权叠加 |
| 低秩逼近 | $A_k = U_k \Sigma_k V_k^{\!\top}$ | 最优的秩 $k$ 矩阵（Eckart–Young） |
| 伪逆 | $A^{+} = V \Sigma^{+} U^{\!\top}$ | 最小范数的最小二乘解 |
| PCA | 对中心化后的 $X$ 做 SVD | 最大方差方向就是右奇异向量 |

---
## 参考资料

- **Strang, G.** (2019). *Introduction to Linear Algebra* (5th ed.), 第 7 章。
- **Trefethen, L. N. & Bau, D.** (1997). *Numerical Linear Algebra*. SIAM.
- **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins.
- **Eckart, C. & Young, G.** (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*, 1(3).
- **Hastie, T., Tibshirani, R. & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer.
- **Koren, Y., Bell, R. & Volinsky, C.** (2009). "Matrix Factorization Techniques for Recommender Systems." *Computer*, 42(8).
- **3Blue1Brown**. *Essence of Linear Algebra* 系列。
