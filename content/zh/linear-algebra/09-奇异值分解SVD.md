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

第 8 章的谱定理给出了 $A = Q\Lambda Q^{\!\top}$，形式简洁优美，但有个硬性限制：**仅适用于对称矩阵**。而现实中遇到的矩阵大多不对称，甚至根本不是方阵：

- 一张 $1920 \times 1080$ 的照片；
- Netflix 风格的用户—电影评分矩阵（百万行，数千列）；
- NLP 中的文档—词项矩阵（文档数 × 词汇量）；
- 生物信息学中的基因表达矩阵。

**奇异值分解（Singular Value Decomposition, SVD）** 却能处理所有这些情况。对于**任意** $m \times n$ 矩阵 $A$，都有

$$A = U\,\Sigma\,V^{\!\top}.$$

这堪称线性代数中最强大、适用范围最广的矩阵分解。

### 一个摄影类比

把一张照片看作像素强度矩阵，SVD 同时揭示了三点：

1. **任何照片都能分解为若干“基础图层”的叠加**；
2. **这些图层按重要性排序**——第一层捕捉整体结构，第二层呈现次要细节，第三层刻画更精细的纹理；
3. **仅保留前几层就能还原图像的绝大部分信息**。

这就像乐队录音：主唱、吉他、贝斯、鼓各自独立。去掉背景和声，歌曲依然成立；但若去掉主唱，整首歌就垮了。SVD 将这种直觉精确化：奇异值 $\sigma_i$ 正好量化了每一层的贡献程度。

### 本章学习目标

- 掌握 SVD 的定义及其**三步几何意义**（旋转 → 拉伸 → 再旋转）；
- 学会通过 $A^{\!\top}\!A$ 和 $AA^{\!\top}$ 计算奇异值与奇异向量；
- 从 $U$ 和 $V$ 直接读出**四个基本子空间**；
- 理解**低秩逼近**与 Eckart–Young 定理——这是数据压缩背后的最优性保证；
- 掌握**伪逆**：为不可逆或非方阵提供通用的“最佳逆”；
- 揭示**PCA 实质上是 SVD 的一种特例**；
- 应用场景包括：图像压缩、推荐系统、潜在语义分析、信号去噪、特征脸等。

### 前置知识

- 特征值与特征向量（第 6 章）
- 正交矩阵与投影（第 7 章）
- 对称矩阵与谱定理（第 8 章）

---

## 二、SVD 的定义

### 基本定理

**SVD 定理**：任意 $m \times n$ 实矩阵 $A$ 都可分解为

$$A = U\,\Sigma\,V^{\!\top},$$

其中

- $U \in \mathbb{R}^{m\times m}$ 是正交矩阵，其列向量为**左奇异向量** $u_1, \ldots, u_m$；
- $V \in \mathbb{R}^{n\times n}$ 是正交矩阵，其列向量为**右奇异向量** $v_1, \ldots, v_n$；
- $\Sigma \in \mathbb{R}^{m\times n}$ 是对角矩阵，主对角线上为**奇异值** $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$，其余元素为零。

实际应用中（尤其对高瘦矩阵），常采用“经济型”形式，仅保留 $r = \operatorname{rank}(A)$ 个非零奇异值：

$$A = U_r\,\Sigma_r\,V_r^{\!\top},\qquad U_r \in \mathbb{R}^{m\times r},\ \Sigma_r \in \mathbb{R}^{r\times r},\ V_r \in \mathbb{R}^{n\times r}.$$

奇异值之所以特殊，在于三点：

- 它们**恒为非负实数**（而特征值可为负或复数）；
- **按惯例降序排列**；
- **SVD 对任意矩阵都存在**——这正是特征分解所缺乏的普适性，也是 SVD 成为“万能工具”的关键。

### 几何意义：三步完成变换

分解式 $A = U\Sigma V^{\!\top}$ 蕴含清晰的几何图像。从右向左解读，对任意向量应用 $A$ 相当于：

1. **旋转** $V^{\!\top}$：将输入坐标系对齐到 $A$ 的“自然输入方向”；
2. **拉伸** $\Sigma$：沿第 $i$ 个轴缩放 $\sigma_i$ 倍；
3. **旋转** $U$：将结果转至输出空间的最终朝向。

想象揉面团：先将面团转到合适角度（$V^{\!\top}$），再用擀面杖压扁拉长（$\Sigma$），最后把面饼摆到所需位置（$U$）。

![SVD 三步：先旋转 V^T，再拉伸 Σ，最后旋转 U](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig1_svd_geometry.png)

图中追踪了单位圆在每一步的变化。两次旋转保持正交基不变，只有中间的对角矩阵 $\Sigma$ 改变形状，将圆变为半轴长为 $\sigma_1$ 和 $\sigma_2$ 的椭圆。

### 单位圆变椭圆

另一种更紧凑的视角：左侧是输入单位圆，右侧是其在 $A$ 作用下的像——一个椭圆。

![右奇异向量 v_i 被映射为 σ_i u_i，正好是椭圆的主轴](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig2_circle_to_ellipse.png)

图中关键信息：

- **右奇异向量** $v_1, v_2$ 是输入侧的正交方向，被 $A$ 映射到输出椭圆的主轴上；
- **左奇异向量** $u_1, u_2$ 是这些主轴在输出空间中的正交方向；
- **奇异值** $\sigma_1, \sigma_2$ 正是椭圆半轴的长度。

一句话概括：

$$A\,v_i \;=\; \sigma_i\, u_i.$$

这就是 SVD 的核心等式，一切性质皆由此衍生。

### 外积展开

SVD 还可等价地写成秩-1 矩阵的加权和：

$$A = \sigma_1 u_1 v_1^{\!\top} + \sigma_2 u_2 v_2^{\!\top} + \cdots + \sigma_r u_r v_r^{\!\top}.$$

每个 $u_i v_i^{\!\top}$ 是秩为 1 的矩阵，奇异值为其权重。这一视角是低秩逼近的关键：保留大权重项，舍弃小权重项。

---

## 三、SVD 的计算

### 桥梁：$A^{\!\top}\!A$ 与 $AA^{\!\top}$

两个对称矩阵的特征向量分别给出 $V$ 和 $U$。将 $A = U\Sigma V^{\!\top}$ 左右相乘可得：

$$A^{\!\top}\!A = V\,\Sigma^{\!\top}\!\Sigma\,V^{\!\top}, \qquad AA^{\!\top} = U\,\Sigma\Sigma^{\!\top}\,U^{\!\top}.$$

$A^{\!\top}\!A$ 与 $AA^{\!\top}$ 均为**对称半正定矩阵**，故可应用谱定理。由此可知：

- $V$ 的列 = $A^{\!\top}\!A$ 的标准正交特征向量（即**右奇异向量**）；
- $U$ 的列 = $AA^{\!\top}$ 的标准正交特征向量（即**左奇异向量**）；
- $\sigma_i = \sqrt{\lambda_i}$，其中 $\lambda_i$ 是这两个矩阵共有的非负特征值。

**为何用 $A^{\!\top}\!A$？** 可理解为 $A$ “往返一次”：先经 $A$ 前进，再由 $A^{\!\top}$ 返回。该过程将某方向放大 $\sigma^2$ 倍，因此 $A^{\!\top}\!A$ 的特征值恰为奇异值的平方。

### 分步计算流程

给定 $m \times n$ 矩阵 $A$（设 $m \ge n$）：

1. 构造 $A^{\!\top}\!A$，求其（降序）特征值 $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$ 及对应的标准正交特征向量 $v_1, \ldots, v_n$，组成 $V$；
2. 令 $\sigma_i = \sqrt{\lambda_i}$；
3. 对每个 $\sigma_i > 0$，定义 $u_i = A v_i / \sigma_i$，作为 $U$ 的前 $r$ 列；
4. 若 $r < m$，用 Gram–Schmidt 等方法将 $\{u_1, \ldots, u_r\}$ 扩展为 $\mathbb{R}^m$ 的标准正交基，补全 $U$。

实际数值计算中，**不会**采用此法——构造 $A^{\!\top}\!A$ 会使条件数平方化，导致精度严重损失。工业级代码通常使用双对角化结合 QR 算法或分治法。上述推导仅为概念说明。

### 手算示例

设 $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$，则

$$A^{\!\top}\!A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}, \qquad \det(A^{\!\top}\!A - \lambda I) = \lambda^2 - 3\lambda + 1.$$

解得 $\lambda = \frac{3 \pm \sqrt{5}}{2}$，故

$$\sigma_1 = \sqrt{\tfrac{3+\sqrt{5}}{2}} \approx 1.618, \qquad \sigma_2 = \sqrt{\tfrac{3-\sqrt{5}}{2}} \approx 0.618.$$

求出 $A^{\!\top}\!A$ 的特征向量得 $V$，再由 $u_i = A v_i / \sigma_i$ 得 $U$。（验证：$\sigma_1 \sigma_2 = 1 = |\det A|$，结果自洽。）

### 特征值 vs 奇异值：直观对比

对称矩阵的特征值与奇异值（最多差符号）一致。但对一般矩阵，二者截然不同，对比极具启发性。

![特征向量不正交，奇异向量正交；特征值描述不变方向，奇异值描述拉伸](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig3_eig_vs_svd.png)

| | 特征向量 | 奇异向量 |
|---|---|---|
| 定义方式 | $A x = \lambda x$ | $A v = \sigma u$，且 $u, v$ 各自正交 |
| 正交性 | 一般不正交 | $V$ 和 $U$ 均为正交矩阵 |
| 取值范围 | $\lambda_i \in \mathbb{C}$ | $\sigma_i \in \mathbb{R}_{\ge 0}$ |
| 几何含义 | 不变方向，按 $\lambda$ 缩放 | 输入正交基映射到输出正交基，按 $\sigma$ 缩放 |
| 存在性 | 仅限方阵，且未必可对角化 | **任意矩阵均存在** |

图中同一非对称矩阵 $A$ 的特征向量呈斜角，而 SVD 在输入与输出两侧均选取了相互垂直的方向。

---

## 四、SVD 与四个基本子空间

SVD 为矩阵结构提供了最清晰的刻画。设 $A \in \mathbb{R}^{m \times n}$，秩为 $r$：

| 子空间 | 维度 | SVD 提供的标准正交基 | 所属空间 |
|---|---|---|---|
| 行空间 $\mathcal{C}(A^{\!\top})$ | $r$ | $v_1, \ldots, v_r$ | $\mathbb{R}^n$ |
| 零空间 $\mathcal{N}(A)$ | $n - r$ | $v_{r+1}, \ldots, v_n$ | $\mathbb{R}^n$ |
| 列空间 $\mathcal{C}(A)$ | $r$ | $u_1, \ldots, u_r$ | $\mathbb{R}^m$ |
| 左零空间 $\mathcal{N}(A^{\!\top})$ | $m - r$ | $u_{r+1}, \ldots, u_m$ | $\mathbb{R}^m$ |

行空间的标准正交基 $\{v_1, \ldots, v_r\}$ 被 $A$ 映射为列空间的标准正交基 $\{u_1, \ldots, u_r\}$，各方向按 $\sigma_i$ 拉伸；零空间中所有向量均被映射为零。这便是矩阵 $A$ 的全部作用。

---

## 五、低秩逼近：压缩背后的定理

![线性代数（九）：奇异值分解 SVD — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/illustration_2.png)

### Eckart–Young 定理

将外积展开截断至前 $k$ 项：

$$A_k = \sigma_1 u_1 v_1^{\!\top} + \cdots + \sigma_k u_k v_k^{\!\top}.$$

**定理（Eckart–Young, 1936）**：在所有秩不超过 $k$ 的矩阵 $B$ 中，

$$\|A - A_k\|_F \;=\; \min_{\operatorname{rank}(B) \le k} \|A - B\|_F \;=\; \sqrt{\sigma_{k+1}^{\,2} + \cdots + \sigma_r^{\,2}}.$$

在算子（2-）范数下，有 $\|A - A_k\|_2 = \sigma_{k+1}$。

因此，$A_k$ 不仅是一个低秩近似，更是**理论上最优**的——没有任何秩-$k$ 矩阵能比它更接近原矩阵。

**MP3 类比**：MP3 压缩舍弃人耳难以察觉的高频成分；SVD 截断则舍弃矩阵中“能量”最低的成分，保留主导信息。

### 图层叠加效果

将每一层单独可视化有助于理解。每个 $\sigma_i u_i v_i^{\!\top}$ 是一张秩-1 图像，逐步叠加后趋近原图。

![秩 1 图层依次相加重建图像；前几层占据了大部分能量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig4_low_rank_blocks.png)

上排展示前三层秩-1 图像（红色为正，蓝色为负）及原图；下排为叠加 1、2、3 层后的累积效果，右侧附奇异值柱状图。仅三层已能还原图像的主要结构。

### 能量视角

定义矩阵“能量”为其 Frobenius 范数的平方：

$$\|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2.$$

秩-$k$ 近似保留的能量比例为

$$\text{能量保留率} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2}.$$

对大多数自然数据，奇异值衰减迅速：一张 $1000 \times 1000$ 的照片，前 50 个奇异值常可保留 95% 以上的能量。

### 图像压缩的实际收益

存储秩-$k$ 近似需保存 $k$ 个奇异值及 $U$、$V$ 的前 $k$ 列，总计

$$\text{存储量} = k\,(m + n + 1).$$

以 $500 \times 500$ 图像为例，取 $k = 50$：原图需 $250{,}000$ 个数，压缩后仅需 $50{,}050$ 个——压缩率达 5 倍，且视觉损失通常难以察觉。

![原图与 k=5, 20, 50 的对比；奇异值谱与累积能量曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig5_image_compression.png)

左下角的对数尺度奇异值衰减曲线是应用 SVD 时最关键的诊断工具：它直观指示了可在不显著损失质量的前提下截断到何种程度。

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

### 当逆矩阵不存在时

求解 $A x = b$ 时，理想解为 $x = A^{-1} b$，但 $A^{-1}$ 仅在 $A$ 为满秩方阵时存在。此时，**Moore–Penrose 伪逆** $A^{+}$ 提供了一个通用的“最佳替代”。

### 通过 SVD 定义伪逆

若 $A = U \Sigma V^{\!\top}$，则
$$A^{+} = V \Sigma^{+} U^{\!\top}, \qquad
\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0,\\ 0 & \sigma_i = 0,\end{cases}$$
且 $\Sigma^{+}$ 转置为 $n \times m$ 形状。当 $A$ 可逆时，$A^{+} = A^{-1}$。

### 伪逆的作用

对任意 $b$，$\hat{x} = A^{+} b$ 满足：

1. 是**最小二乘解**：使 $\|A x - b\|_2$ 最小；
2. 在所有最小二乘解中，具有**最小范数** $\|x\|_2$。

两种典型情形：

- **超定系统**（$m > n$）：通常无精确解，$A^{+} b$ 给出最小二乘拟合；
- **欠定系统**（$m < n$）：解无穷多，$A^{+} b$ 选取范数最小者。

![伪逆几何：A^+ b 是 b 在 col(A) 上的投影，残差与 col(A) 正交](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig6_pseudoinverse.png)

左图展示经典的最小二乘直线拟合；右图揭示其几何本质：$b$ 被正交投影到 $A$ 的列空间，残差与之垂直。这种正交性正是**正规方程**的核心，而 SVD 自动满足该条件。

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

### 二者关系

主成分分析（PCA）本质上就是披着统计外衣的 SVD。

将数据矩阵 $X \in \mathbb{R}^{n\times p}$ 按列中心化（每列均值为零），记为 $X_c$，对其做 SVD：

$$X_c = U \Sigma V^{\!\top}.$$

则有：

- **主成分方向** = $V$ 的列（右奇异向量），即最大方差的正交轴；
- **主成分得分** = $X_c V = U \Sigma$，即数据在新基下的坐标；
- **第 $i$ 主成分的方差** = $\sigma_i^2 / (n - 1)$；
- **降至 $k$ 维**：取 $X_k = U_k \Sigma_k$。

### 为何 PCA 有效

第一主成分方向是使 $\operatorname{Var}(X_c w)$ 最大的单位向量 $w$。简单推导可知，该目标等价于最大化 $w^{\!\top}(X_c^{\!\top} X_c) w / (n-1)$，其解恰为 $X_c^{\!\top} X_c$ 的最大特征向量——即右奇异向量 $v_1$。

![中心化数据 + 主成分轴；PC1 方向上的得分直方图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-奇异值分解SVD/fig7_pca_via_svd.png)

虚线椭圆为 1-$\sigma$ 高斯拟合，两箭头为主成分轴，长度正比于标准差 $\sigma_i / \sqrt{n-1}$。将数据投影到 PC1 上，从 2D 降至 1D，同时保留了绝大部分方差。

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

### 问题设定

Netflix、亚马逊、Spotify 等平台面临共同挑战：**如何预测用户对未接触物品的评分？** 用户—物品评分矩阵 $R \in \mathbb{R}^{m\times n}$ 规模庞大且极度稀疏。

### 矩阵分解思想

建模假设：评分由少量**潜在因子**驱动——对电影而言，可能是“动作强度”、“浪漫指数”、“幽默感”或“艺术深度”。于是有

$$R \approx U_k \Sigma_k V_k^{\!\top}.$$

- $U_k \Sigma_k$ 的行向量表示用户偏好；
- $V_k$ 的行向量表示物品特征；
- 预测评分即二者点积。

具体做法：利用已知评分拟合 $U_k$ 与 $V_k$，再预测缺失项。此方法正是 **Netflix Prize（2006–2009）** 冠军方案的核心。

---

## 九、SVD 与特征分解对比

| 性质 | 特征分解 | SVD |
|---|---|---|
| 适用对象 | 方阵（通常对称） | **任意矩阵** |
| 分解形式 | $A = P\Lambda P^{-1}$ | $A = U\Sigma V^{\!\top}$ |
| 数值特性 | 特征值可为负或复数 | 奇异值恒为非负实数 |
| 向量性质 | 特征向量未必正交 | 奇异向量始终正交 |
| 几何解释 | 不变方向 + 缩放 | 旋转 + 拉伸 + 再旋转 |
| 是否总存在 | 否 | **是** |

### 为何 SVD 被誉为“皇冠明珠”

- **普适性**：无论方阵与否、满秩与否，皆可应用；
- **数值稳定性**：计算鲁棒，是秩、条件数、最小二乘问题的黄金标准；
- **理论最优性**：Eckart–Young 定理保证低秩逼近的最优性；
- **结构洞察力**：一次性揭示秩、四个子空间及算子范数；
- **广泛应用**：图像压缩、NLP、推荐系统、去噪、控制理论、统计分析等领域不可或缺。

---

## 十、其他应用

### 潜在语义分析（LSA）

构建文档—词项矩阵（行=文档，列=词汇，元素=TF-IDF 值），对其做 SVD 并保留前 $k$ 个分量。右奇异向量可视为“潜在主题”，文档相似度转化为低维空间中的余弦相似度。

### 信号去噪

假设信号低秩、噪声满秩。对观测数据做 SVD，保留大奇异值、舍弃小奇异值后重构。此法广泛应用于天文图像清理、地震数据处理等领域。

### 特征脸（Eigenfaces）

对对齐的人脸图像集做 PCA，所得主成分即“特征脸”，构成人脸典型表示基。新人脸可表为特征脸的线性组合，识别简化为比较系数向量。

---

## Python 实现

### 通过特征分解手动实现 SVD

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

---

## 练习题

### 基础题

1. 解释为何奇异值恒为非负，而特征值可为负或复数。
2. 手动计算 $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ 的 SVD。
3. 若 $A$ 为 $3 \times 5$ 矩阵，完整 SVD 与经济型 SVD 中 $U$、$\Sigma$、$V^{\!\top}$ 的形状分别为何？

### 进阶题

4. 证明 $\operatorname{rank}(A)$ 等于非零奇异值的个数。
5. 证明 $\|A\|_F^2 = \sigma_1^2 + \cdots + \sigma_r^2$。
6. 若 $Q$ 为正交矩阵，其奇异值为何？说明理由。
7. 证明 $U_r U_r^{\!\top}$ 为列空间投影矩阵，$V_r V_r^{\!\top}$ 为行空间投影矩阵。
8. 证明 Eckart–Young 定理的算子范数形式：$\|A - A_k\|_2 = \sigma_{k+1}$。

### 编程题

9. **压缩曲线**：加载灰度图像，计算 SVD，绘制 $k \in \{5, 20, 50, 100\}$ 的重建图，并在对数坐标下绘制奇异值衰减曲线。
10. **Iris 数据集上的 PCA**：对 Iris 数据做 PCA，绘制前两主成分散点图（按物种着色），报告解释方差比。
11. **简易推荐系统**：构造 $5 \times 10$ 评分矩阵（部分缺失），用行均值填充后做秩-3 SVD，观察预测评分合理性。

---

## 十三、总结

| 概念 | 核心公式 | 直观理解 |
|---|---|---|
| SVD | $A = U\Sigma V^{\!\top}$ | 旋转 → 拉伸 → 再旋转 |
| 奇异值 | $\sigma_i \ge 0$ | 椭圆主轴的拉伸倍数 |
| 外积形式 | $A = \sum_i \sigma_i u_i v_i^{\!\top}$ | 秩-1 分量的加权叠加 |
| 低秩逼近 | $A_k = U_k \Sigma_k V_k^{\!\top}$ | 最优秩-$k$ 矩阵（Eckart–Young） |
| 伪逆 | $A^{+} = V \Sigma^{+} U^{\!\top}$ | 最小范数的最小二乘解 |
| PCA | 对中心化 $X$ 做 SVD | 最大方差方向 = 右奇异向量 |

---

## 参考文献

- Strang, G. (2019). *Introduction to Linear Algebra*, 第 7 章。
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*（第 4 版）. Johns Hopkins.
- Eckart, C. & Young, G. (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*, 1(3).
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Koren, Y., Bell, R. & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." *Computer*, 42(8).
- 3Blue1Brown. *Essence of Linear Algebra* 系列。
