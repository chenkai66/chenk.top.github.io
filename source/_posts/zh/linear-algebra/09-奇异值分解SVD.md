---
title: "奇异值分解 SVD"
date: 2025-02-16 09:00:00
tags:
  - 线性代数
  - SVD
  - 奇异值分解
  - PCA
  - 降维
  - 图像压缩
description: "SVD 被誉为线性代数的皇冠明珠：它能分解任意矩阵，不限于方阵或对称矩阵。从图像压缩到推荐系统，从人脸识别到基因分析，SVD 无处不在。"
categories: 线性代数
series:
  name: "线性代数"
  part: 9
  total: 18
lang: zh-CN
mathjax: true
disableNunjucks: true
series_order: 9
---

## 一、为什么 SVD 配得上"皇冠"二字

[第 8 章](/zh/线性代数-八-对称矩阵与二次型/)的谱定理告诉我们 $A = Q\Lambda Q^{\!\top}$，干净漂亮，但**有一个硬性前提：$A$ 必须对称**。现实里大多数矩阵既不对称，甚至不是方阵：

- 一张 $1920 \times 1080$ 的图像；
- 用户—电影评分矩阵（百万级用户、数千部电影）；
- 文档—词项矩阵（NLP 中文档数 × 词汇量）；
- 基因表达矩阵（生物信息学）。

**奇异值分解（SVD）** 一视同仁地处理它们。对**任意** $m \times n$ 矩阵 $A$，
$$A = U\,\Sigma\,V^{\!\top}.$$
这是线性代数中最强大、最普适的分解，没有之一。

### 一个生活类比

把照片当成像素矩阵，SVD 同时告诉你三件事：

1. **任何照片都可以分解为若干"基础图层"的叠加**；
2. **这些图层按重要性排序** —— 第一层抓主结构，第二层抓次要细节，依此类推；
3. **只保留前几层就能还原大部分信息**。

就像乐队录音由主唱、吉他、贝斯、鼓四条轨道叠加：去掉背景和声没什么人察觉，去掉主唱整首歌就废了。SVD 把这种"重要性"量化为奇异值。

### 本章学习目标

- SVD 的定义和**三步几何意义**：旋转 → 拉伸 → 旋转；
- 通过 $A^{\!\top}\!A$ 与 $AA^{\!\top}$ 求出奇异值与奇异向量；
- 从 $U, V$ 直接读出**四个基本子空间**；
- **低秩逼近**与 Eckart–Young 定理 —— 数据压缩背后的最优性；
- **伪逆**：处理不可逆/非方阵的通用"最佳逆"；
- **PCA 就是 SVD**；
- 应用：图像压缩、推荐系统、潜在语义分析、信号去噪、特征脸。

### 前置知识

- 特征值与特征向量（第 6 章）
- 正交矩阵与投影（第 7 章）
- 对称矩阵与谱定理（第 8 章）

---

## 二、SVD 的定义

### 基本定理

**SVD 定理**：任意 $m \times n$ 实矩阵 $A$ 都可以分解为
$$A = U\,\Sigma\,V^{\!\top},$$
其中

- $U \in \mathbb{R}^{m\times m}$ 正交，列是**左奇异向量** $u_1, \ldots, u_m$；
- $V \in \mathbb{R}^{n\times n}$ 正交，列是**右奇异向量** $v_1, \ldots, v_n$；
- $\Sigma \in \mathbb{R}^{m\times n}$ 主对角线上是**奇异值** $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$，其余位置全为 0。

实际计算更常用"经济型"，只保留 $r = \operatorname{rank}(A)$ 个非零奇异值：
$$A = U_r\,\Sigma_r\,V_r^{\!\top},\qquad U_r\in\mathbb{R}^{m\times r},\ \Sigma_r\in\mathbb{R}^{r\times r},\ V_r\in\mathbb{R}^{n\times r}.$$

奇异值有三个关键性质：

- **永远是非负实数**（特征值可能是负数甚至复数）；
- **降序排列**是惯例；
- **任何矩阵都有 SVD**。这正是特征分解所欠缺的，也是 SVD"普适"的根源。

### 几何意义：把变换拆成三步

读 $A = U\Sigma V^{\!\top}$ 时，从右往左作用：

1. **旋转** $V^{\!\top}$：把输入对齐到 $A$ 的"自然输入方向"；
2. **拉伸** $\Sigma$：第 $i$ 个坐标乘以 $\sigma_i$；
3. **旋转** $U$：把结果转到输出空间的最终位置。

**揉面团**的比喻：先把面团转到顺手的角度（$V^{\!\top}$），再用擀面杖压扁拉长（$\Sigma$），最后把擀好的面饼摆到需要的方向（$U$）。

![SVD 三步：先旋转 V^T，再拉伸 Σ，最后旋转 U](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig1_svd_geometry.png)

图中跟踪一个单位圆经过每一步的变化：两次正交旋转保持正交基不变，只有中间这一步对角阵 $\Sigma$ 真正改变了形状，把圆变成了半轴长度为 $\sigma_1, \sigma_2$ 的椭圆。

### 单位圆变椭圆

把同样的故事浓缩到一对输入—输出图里：左边是单位圆，右边是 $A$ 作用之后的椭圆。

![右奇异向量 v_i 被映射为 σ_i u_i，正好是椭圆的主轴](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig2_circle_to_ellipse.png)

读图要点：

- **右奇异向量** $v_1, v_2$：输入侧的正交方向，恰好被映射到椭圆的主轴上；
- **左奇异向量** $u_1, u_2$：输出侧主轴的正交方向；
- **奇异值** $\sigma_1, \sigma_2$：椭圆的半轴长度。

一句话总结：
$$A\,v_i \;=\; \sigma_i\, u_i.$$
后面所有结论都从这条等式自然流出。

### 外积展开

SVD 还有一个等价写法 —— **秩 1 矩阵的加权和**：
$$A = \sigma_1 u_1 v_1^{\!\top} + \sigma_2 u_2 v_2^{\!\top} + \cdots + \sigma_r u_r v_r^{\!\top}.$$
每个 $u_i v_i^{\!\top}$ 都是秩为 1 的"基础积木"，奇异值是权重。这正是低秩逼近的关键视角：保留权重大的，扔掉权重小的。

---

## 三、SVD 的计算

### 桥梁：$A^{\!\top}\!A$ 与 $AA^{\!\top}$

$V$ 与 $U$ 来自两个对称矩阵的特征向量。把 $A = U\Sigma V^{\!\top}$ 双向乘开：
$$A^{\!\top}\!A = V\,\Sigma^{\!\top}\!\Sigma\,V^{\!\top},\qquad AA^{\!\top} = U\,\Sigma\Sigma^{\!\top}\,U^{\!\top}.$$

$A^{\!\top}\!A$ 与 $AA^{\!\top}$ 都是**对称半正定矩阵**，谱定理可用。把上式当作各自的谱分解读：

- $V$ 的列 = $A^{\!\top}\!A$ 的正交特征向量（**右奇异向量**）；
- $U$ 的列 = $AA^{\!\top}$ 的正交特征向量（**左奇异向量**）；
- $\sigma_i = \sqrt{\lambda_i}$，其中 $\lambda_i$ 是这两个乘积**共有**的非负特征值。

**为什么是 $A^{\!\top}\!A$？** 把它理解成"$A$ 作用两次"：先正向走 $A$，再反向走 $A^{\!\top}$。这一来一回让某方向被放大了 $\sigma^2$ 倍 —— 所以 $A^{\!\top}\!A$ 的特征值正好是奇异值的平方。

### 计算步骤

设 $A$ 是 $m \times n$ 矩阵，$m \ge n$：

1. 计算 $A^{\!\top}\!A$，求其降序排列的特征值 $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$ 与正交单位特征向量 $v_1, \ldots, v_n$。这 $n$ 个向量组成 $V$。
2. 取 $\sigma_i = \sqrt{\lambda_i}$。
3. 对每个 $\sigma_i > 0$，令 $u_i = A v_i / \sigma_i$，得到 $U$ 的前 $r$ 列。
4. 若 $r < m$，用 Gram–Schmidt 把 $\{u_1, \ldots, u_r\}$ 扩成 $\mathbb{R}^m$ 的正交单位基，补齐 $U$。

数值实现中**没人**真这么算 —— 显式构造 $A^{\!\top}\!A$ 会让条件数平方化，损失精度。生产代码用双对角化加 QR 算法或分治法。上面的推导只为讲清楚原理。

### 一个手算例子

取 $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$：

$$A^{\!\top}\!A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix},\qquad \det(A^{\!\top}\!A - \lambda I) = \lambda^2 - 3\lambda + 1.$$

解出 $\lambda = \frac{3 \pm \sqrt 5}{2}$，于是
$$\sigma_1 = \sqrt{\tfrac{3+\sqrt 5}{2}} \approx 1.618,\qquad \sigma_2 = \sqrt{\tfrac{3-\sqrt 5}{2}} \approx 0.618.$$

求 $A^{\!\top}\!A$ 的特征向量得到 $V$，再令 $u_i = A v_i / \sigma_i$ 得到 $U$。验证一下：$\sigma_1 \sigma_2 = 1 = |\det A|$，自洽。

### 特征值 vs 奇异值：直观对比

对称矩阵的特征值与奇异值（最多差正负号）是同一个东西。一旦 $A$ 不对称，两者就分道扬镳，对比非常有启发。

![特征向量不正交，奇异向量正交；特征值描述不变方向，奇异值描述拉伸](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig3_eig_vs_svd.png)

| | 特征向量 | 奇异向量 |
|---|---|---|
| 定义 | $A x = \lambda x$ | $A v = \sigma u$，且 $u, v$ 各自正交 |
| 正交性 | 一般不正交 | $V$ 与 $U$ 都是正交矩阵 |
| 取值 | $\lambda_i \in \mathbb{C}$ | $\sigma_i \in \mathbb{R}_{\ge 0}$ |
| 几何含义 | 不变方向，按 $\lambda$ 缩放 | 输入正交基映射到输出正交基，按 $\sigma$ 缩放 |
| 存在性 | 方阵，且未必可对角化 | 任意矩阵都存在 |

图中同一个非对称 $A$ 的特征向量呈斜角，而 SVD 在输入和输出两侧都给出了垂直对。

---

## 四、SVD 与四个基本子空间

SVD 提供了矩阵结构最干净的全景图。设 $A \in \mathbb{R}^{m\times n}$，秩为 $r$：

| 子空间 | 维度 | SVD 给出的正交基 | 所在空间 |
|---|---|---|---|
| 行空间 $\mathcal{C}(A^{\!\top})$ | $r$ | $v_1, \ldots, v_r$ | $\mathbb{R}^n$ |
| 零空间 $\mathcal{N}(A)$ | $n - r$ | $v_{r+1}, \ldots, v_n$ | $\mathbb{R}^n$ |
| 列空间 $\mathcal{C}(A)$ | $r$ | $u_1, \ldots, u_r$ | $\mathbb{R}^m$ |
| 左零空间 $\mathcal{N}(A^{\!\top})$ | $m - r$ | $u_{r+1}, \ldots, u_m$ | $\mathbb{R}^m$ |

行空间的正交基 $\{v_1, \ldots, v_r\}$ 被映射到列空间的正交基 $\{u_1, \ldots, u_r\}$，每个方向上拉伸 $\sigma_i$ 倍；零空间整体被压扁到原点。**这就是 $A$ 全部的作用**。

---

## 五、低秩逼近：压缩背后的定理

### Eckart–Young 定理

把外积展开截断到前 $k$ 项：
$$A_k = \sigma_1 u_1 v_1^{\!\top} + \cdots + \sigma_k u_k v_k^{\!\top}.$$

**定理（Eckart–Young, 1936）**：在所有秩不超过 $k$ 的矩阵 $B$ 中，
$$\|A - A_k\|_F \;=\; \min_{\operatorname{rank}(B) \le k} \|A - B\|_F \;=\; \sqrt{\sigma_{k+1}^{\,2} + \cdots + \sigma_r^{\,2}}.$$
在算子（2-）范数下也有 $\|A - A_k\|_2 = \sigma_{k+1}$。

也就是说，$A_k$ 不是"一个"低秩近似 —— 它是**可证最优**的近似，没有任何秩为 $k$ 的矩阵能更接近 $A$。

**类比 MP3**：MP3 丢掉人耳不敏感的高频成分。SVD 截断对矩阵做了同样的事 —— 丢掉"能量"最小的分量，保留响亮的。

### 一层一层叠

把外积"积木"画出来更直观：每个 $\sigma_i u_i v_i^{\!\top}$ 单独是一张秩 1 图像，部分和会越来越接近原图。

![秩 1 图层依次相加重建图像；前几层占据了大部分能量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig4_low_rank_blocks.png)

上排是前三个秩 1 图层本身（红正蓝负）以及原图；下排是累计前 1, 2, 3 层的重建结果，右下是奇异值柱状图。仅仅三层就已经看出图像主结构。

### 能量视角

矩阵的"能量"用 Frobenius 范数平方衡量：
$$\|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2.$$
秩 $k$ 近似保留的能量比例：
$$\text{能量比例} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2}.$$
对绝大多数自然数据，奇异值衰减很快：一张 $1000 \times 1000$ 的照片，前 50 个奇异值往往就能装下 95% 的能量。

### 图像压缩到底省多少

存储一份秩 $k$ 近似要保存 $k$ 个奇异值加上 $U, V$ 的前 $k$ 列：
$$\text{需存数字数} = k\,(m + n + 1).$$
$500 \times 500$ 图像 + $k = 50$：原图 $250{,}000$ 个数字，压缩后 $50{,}050$ 个 —— 5 倍压缩，肉眼几乎看不出区别。

![原图与 k=5, 20, 50 的对比；奇异值谱与累积能量曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig5_image_compression.png)

左下角对数坐标的奇异值衰减曲线，是应用 SVD 时**最重要的诊断图**：它告诉你能积极截断到多少不至于崩。

```python
import numpy as np

def compress(img, k):
    """对 2D 数组做秩 k SVD 近似。"""
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# 诊断：每个 k 保留多少能量？
U, s, Vt = np.linalg.svd(img, full_matrices=False)
for k in [5, 20, 50, 100]:
    energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
    print(f"k={k:>3}: 能量保留 {energy:.1f}%")
```

---

## 六、伪逆

### 当逆矩阵不存在时

解 $A x = b$ 时我们想要 $x = A^{-1} b$，但 $A^{-1}$ 只在方阵满秩时才存在。**Moore–Penrose 伪逆** $A^{+}$ 是一个普适的"最佳替代品"。

### 用 SVD 定义

若 $A = U \Sigma V^{\!\top}$，则
$$A^{+} = V \Sigma^{+} U^{\!\top},\qquad
\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i, & \sigma_i > 0,\\ 0, & \sigma_i = 0,\end{cases}$$
并把 $\Sigma^{+}$ 转成 $n \times m$ 的形状。当 $A$ 可逆时 $A^{+} = A^{-1}$。

### $A^{+}$ 在做什么

对任意 $b$，$\hat x = A^{+} b$ 同时满足：

1. **最小二乘解**：使 $\|A x - b\|_2$ 最小；
2. 在所有最小二乘解中，**范数最小**的那个。

两种典型情形：

- **超定**（$m > n$）：通常无精确解，$A^{+} b$ 给最小二乘拟合；
- **欠定**（$m < n$）：解有无穷多，$A^{+} b$ 选最短的那个。

![伪逆几何：A^+ b 是 b 在 col(A) 上的投影，残差与 col(A) 正交](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig6_pseudoinverse.png)

左图是经典最小二乘直线拟合；右图揭示了"最小二乘"得名的几何原因 —— $b$ 投影到 $A$ 的列空间，残差与列空间正交。这种正交性正是**正规方程**的内容，SVD 让它白白送上门来。

```python
import numpy as np

# 用伪逆做最小二乘拟合 y = a x + b
x = np.linspace(-2, 4, 25)
y = 1.2 * x + 0.4 + np.random.default_rng(0).normal(0, 1, 25)

A = np.column_stack([x, np.ones_like(x)])
coef = np.linalg.pinv(A) @ y          # 内部用 SVD
print(f"斜率={coef[0]:.3f}  截距={coef[1]:.3f}")
```

---

## 七、用 SVD 做 PCA

### 关系

主成分分析其实是穿了统计外衣的 SVD。

把数据矩阵 $X \in \mathbb{R}^{n\times p}$ 中心化（每列减去均值），记为 $X_c$。对它做 SVD：
$$X_c = U \Sigma V^{\!\top}.$$
则：

- **主成分方向** = $V$ 的列（右奇异向量），是方差最大的正交方向；
- **主成分得分** = $X_c V = U \Sigma$，是数据在新基下的坐标；
- **第 $i$ 个主成分的方差** = $\sigma_i^2 / (n - 1)$；
- **降到 $k$ 维**：$X_k = U_k \Sigma_k$。

### 为什么 PCA 有效

第一主成分要在单位向量 $w$ 中最大化 $\operatorname{Var}(X_c w)$。展开后等于 $w^{\!\top}\!(X_c^{\!\top} X_c)\,w / (n-1)$，其最大化方向正是 $X_c^{\!\top} X_c$ 的最大特征向量 —— 也就是最大右奇异向量 $v_1$。

![中心化数据 + 主成分轴；PC1 方向上的得分直方图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/linear-algebra/09-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3SVD/fig7_pca_via_svd.png)

虚线椭圆是 1-$\sigma$ 高斯拟合，两个箭头是按标准差 $\sigma_i / \sqrt{n-1}$ 缩放的主成分轴。把数据投影到 PC1 上，从 2 维压到 1 维但保留了大部分方差。

```python
import numpy as np

def pca(X, k):
    """通过 SVD 实现 PCA，返回 (得分, 主成分方向, 解释方差比例)。"""
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                    # k x p
    scores = Xc @ components.T             # n x k
    var = (s ** 2) / (X.shape[0] - 1)
    return scores, components, (var / var.sum())[:k]
```

---

## 八、推荐系统

### 问题背景

Netflix、亚马逊、淘宝面对同一个核心问题：**如何预测用户对没看过的物品的喜好？** 用户—物品评分矩阵 $R \in \mathbb{R}^{m\times n}$ 巨大且严重缺失。

### 矩阵分解

建模假设是：评分由少数**潜在因子**驱动 —— 对电影来说，可能是"动作浓度"、"浪漫程度"、"幽默感"、"艺术深度"。于是
$$R \approx U_k \Sigma_k V_k^{\!\top}.$$

- $U_k \Sigma_k$ 的一行：某个用户的口味向量；
- $V_k$ 的一行：某部电影的特征向量；
- 预测评分 = 两者的点积。

只用观测到的评分去拟合 $U_k, V_k$，再读出剩余条目 —— 这正是 Netflix Prize（2006–2009）冠军方法的核心思想。

---

## 九、SVD vs 特征分解

| 性质 | 特征分解 | SVD |
|---|---|---|
| 适用对象 | 方阵（通常对称） | 任意矩阵 |
| 形式 | $A = P\Lambda P^{-1}$ | $A = U\Sigma V^{\!\top}$ |
| 取值 | 特征值，可负可复 | 奇异值，非负实数 |
| 向量 | 特征向量，未必正交 | 奇异向量，永远正交 |
| 几何含义 | 不变方向 + 缩放 | 旋转 + 拉伸 + 旋转 |
| 是否总存在 | 否 | 是 |

### 凭什么称作"皇冠明珠"

- **普适**：任意矩阵都能分解；
- **稳定**：数值上极其稳健 —— 求秩、判病态、做最小二乘的事实标准；
- **最优**：低秩逼近有可证的最优性（Eckart–Young）；
- **洞察力**：一次性给出秩、四个子空间、算子范数；
- **实用**：图像压缩、NLP、推荐系统、去噪、控制、统计 —— 都靠它。

---

## 十、其他应用

### 潜在语义分析（LSA）

构造文档—词项矩阵（行：文档，列：词，值：TF-IDF），做 SVD 后取前 $k$ 个分量。右奇异向量充当"潜在主题"，文档相似度变成低维空间里的余弦相似度。

### 信号去噪

模型：低秩信号 + 满秩噪声。对观测做 SVD，保留大奇异值（信号）、丢掉小奇异值（噪声），重构。从天文图像清理到地震数据处理都靠这个套路。

### 特征脸（Eigenfaces）

对一组对齐的人脸图像做 PCA，主成分就是"特征脸"。任何新面孔可表示为特征脸的线性组合，识别就化为系数向量的距离比较。

---

## 十一、Python 实现

### 用特征分解手写 SVD

```python
import numpy as np

def svd_via_eigen(A):
    """通过 A^T A 的特征分解概念性地计算 SVD。
    生产环境请用 np.linalg.svd —— 这只用于教学。"""
    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)            # 升序
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

## 十二、练习题

### 入门

1. 解释为什么奇异值永远非负，而特征值可以是负数或复数。
2. 手算 $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ 的 SVD。
3. 设 $A$ 是 $3 \times 5$ 矩阵，写出完整 SVD 中 $U, \Sigma, V^{\!\top}$ 的形状；经济型 SVD 中又是什么形状？

### 进阶

4. 证明 $\operatorname{rank}(A)$ 等于非零奇异值的个数。
5. 证明 $\|A\|_F^2 = \sigma_1^2 + \cdots + \sigma_r^2$。
6. 若 $Q$ 是正交矩阵，它的奇异值是什么？为什么？
7. 证明 $U_r U_r^{\!\top}$ 是到列空间的投影矩阵，$V_r V_r^{\!\top}$ 是到行空间的投影矩阵。
8. 证明 Eckart–Young 在算子范数下的版本：$\|A - A_k\|_2 = \sigma_{k+1}$。

### 编程实战

9. **压缩曲线**：选一张灰度图，对 $k \in \{5, 20, 50, 100\}$ 画出秩 $k$ 重建图，并在对数坐标下画奇异值衰减曲线。
10. **PCA on Iris**：对 Iris 数据做 PCA，画出前两个主成分的散点图（按品种着色），并报告解释方差比例。
11. **小推荐系统**：构造一个 $5 \times 10$ 的评分矩阵（含缺失），用行均值填充缺失值，做秩 3 SVD，观察预测评分的合理性。

---

## 十三、本章小结

| 概念 | 关键公式 | 直觉 |
|---|---|---|
| SVD | $A = U\Sigma V^{\!\top}$ | 旋转 + 拉伸 + 旋转 |
| 奇异值 | $\sigma_i \ge 0$ | 椭圆主轴的拉伸因子 |
| 外积形式 | $A = \sum_i \sigma_i u_i v_i^{\!\top}$ | 秩 1 积木的加权和 |
| 低秩逼近 | $A_k = U_k \Sigma_k V_k^{\!\top}$ | 最优秩 $k$ 矩阵（Eckart–Young） |
| 伪逆 | $A^{+} = V \Sigma^{+} U^{\!\top}$ | 最小范数最小二乘解 |
| PCA | 中心化 $X$ 的 SVD | 最大方差方向 = 右奇异向量 |

---

## 参考资料

- **Strang, G.** (2019). *Introduction to Linear Algebra* (5th ed.), 第 7 章。
- **Trefethen, L. N. & Bau, D.** (1997). *Numerical Linear Algebra*. SIAM.
- **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins.
- **Eckart, C. & Young, G.** (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*, 1(3).
- **Hastie, T., Tibshirani, R. & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer.
- **Koren, Y., Bell, R. & Volinsky, C.** (2009). "Matrix Factorization Techniques for Recommender Systems." *Computer*, 42(8).
- **3Blue1Brown**. *Essence of Linear Algebra* 系列。

---

## 系列导航

- **上一篇：** [第 8 章 -- 对称矩阵与二次型](/zh/线性代数-八-对称矩阵与二次型/)
- **下一篇：** [第 10 章 -- 矩阵范数与条件数](/zh/线性代数-十-矩阵范数与条件数/)
- **系列：** 线性代数的本质（第 9 篇，共 18 篇）
