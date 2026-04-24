---
title: "变分自编码器 (VAE)：从直觉到实现与调试"
date: 2024-07-24 09:00:00
tags:
  - ML
  - Deep Learning
  - Generative Models
categories: Algorithm
lang: zh-CN
mathjax: true
description: "从零用 PyTorch 构建 VAE。涵盖 ELBO 目标函数、重参数化技巧、后验坍塌修复、beta-VAE，以及完整的训练流水线。"
disableNunjucks: true
---

普通自编码器只能压缩与重建，变分自编码器（VAE）则要有用得多——它学到的是一个**平滑、有结构的潜在空间**，你可以从里面**采样**，凭空生成新的数据。把编码器从"输出一个向量"改成"输出一个分布"，仅这一步，模型就从一个花哨的压缩器升级成了带可优化似然下界的生成模型。

本文走完整条路径：先讲清楚自编码器为什么不能生成；再从 ELBO 推导出损失函数；接着拆解为什么必须用重参数化技巧让梯度走得通；然后给出可以直接跑的 PyTorch 实现；最后逐一分析每种你**一定会遇到**的失败模式，并给出对应的修复手段。

## 你将学到什么

- 自编码器的潜在空间为什么不能采样，VAE 改了哪一步
- ELBO 目标函数：重建项与 KL 项是怎么从一个似然下界中自然落出来的
- 重参数化技巧：为什么直接采样会切断梯度，`mu + sigma * eps` 又是怎么救场的
- 一份完整的 PyTorch 实现：编码器、解码器、损失、训练循环、采样、插值
- 你迟早会遇到的失败模式：后验坍塌、采样模糊、NaN 梯度——含诊断与修复
- 常用变体：beta-VAE、条件 VAE（CVAE）、层次化 VAE
- 什么时候用 VAE，什么时候改用 GAN 或扩散模型

## 前置知识

- PyTorch 基础（`nn.Module`、前向/反向传播、优化器）
- 概率基础（均值、方差、高斯分布）
- 完整训练过神经网络的经验

---

# 为什么需要 VAE：自编码器与生成模型

![VAE 架构：编码器输出 (mu, sigma)，重参数化采样后送入解码器，KL 把 q 拉向先验](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig1_vae_architecture.png)

## 自编码器基线

自编码器由编码器 $f_\phi$ 和解码器 $g_\theta$ 组成。它把输入 $x$ 压成编码 $z = f_\phi(x)$，再把它重建成 $\hat{x} = g_\theta(z)$。训练目标就是重建误差：

$$
\mathcal{L}_{\text{AE}}(x) = \|x - g_\theta(f_\phi(x))\|^2.
$$

用作压缩或去噪没问题，但它给你的是一个**确定性、无结构**的潜在空间，具体表现为：

- **没法采样。** 解码器只见过编码器输出的那些点。在空间里随便撒一个 $z$，大概率落进解码器从没拜访过的"洞"里，吐出来一坨垃圾。
- **插值很脆弱。** 两张视觉相近的图可能映射到很远的位置；两张毫无关系的图反而挨在一起。在潜在空间里走一条直线，中间多半是无意义的过渡。
- **没有概率解释。** 没有先验、没有似然，更没法谈"这个生成样本到底有多可能"。

## VAE 的改动：把潜在变量概率化

VAE 把"确定性的编码"换成"一个分布"。编码器输出的是高斯分布的参数：

$$
q_\phi(z \mid x) = \mathcal{N}\!\left(\mu_\phi(x),\, \sigma^2_\phi(x)\, I\right).
$$

解码器定义似然 $p_\theta(x \mid z)$，再加上一个固定的先验 $p(z) = \mathcal{N}(0, I)$。

仅这一改动，就带来了三件好事：

1. **能采样了。** 从 $p(z)$ 抽一个 $z$，喂给解码器，就得到一张全新的 $\hat{x}$。
2. **插值变平滑。** 相似输入对应的编码分布会**互相重叠**，于是潜在空间里的直线对应的就是一段连贯的渐变。
3. **结构被强制约束。** 损失中的 KL 项会主动把聚合后的后验拉向先验，让整个空间被均匀填满。

# ELBO 目标函数：损失从哪里来

## 推导下界

我们要最大化数据的对数似然 $\log p_\theta(x)$，但它涉及对 $z$ 的积分，不可解。这时引入变分后验 $q_\phi(z \mid x)$，配合 Jensen 不等式，可以得到：

$$
\log p_\theta(x) \;\geq\; \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log p_\theta(x \mid z)\right] - D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

右边就是大名鼎鼎的**证据下界（ELBO）**。最大化它，等价于同时做两件事：

- **重建项** $\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]$：解码器要能从编码器抽出的 $z$ 里把 $x$ 还原回来。
- **KL 项** $D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))$：编码器不能想跑哪儿跑哪儿，必须紧贴着先验。

写代码时我们最小化的是负 ELBO。当编码器是高斯、先验是标准正态时，KL 项有逐维度的闭式解：

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu, \sigma^2)\,\|\,\mathcal{N}(0,1)\right)
= \tfrac{1}{2}\!\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right).
$$

## 为什么 KL 项是关键的承重墙

去掉 KL 正则，立刻会出两个极端：

- **尖峰加空隙。** 编码器把每个 $x$ 映到一个孤立的尖刺上，方差小得可怜。整个空间就是真空里的一串针——重建好看，生成无用。
- **解码器无视 $z$。** 解码器太强时会出现反方向的崩坏（也就是后面要讲的"后验坍塌"）。

KL 项强制不同��入的 $q_\phi(z\mid x)$ **重叠**并**覆盖先验**。这种重叠正是平滑插值的来源，也正是先验能当作采样器使用的前提。

# 重参数化技巧

## 问题：梯度穿不过随机采样

要估算重建项，得先从 $q_\phi(z \mid x)$ 抽样 $z$ 再跑解码器。问题是采样是个**随机节点**，反向传播没法穿过"抽一个随机数"这一步——直接采样的话，编码器参数 $\phi$ 的梯度根本无定义。

![朴素采样 vs. 重参数化版本：把随机的 epsilon 移到参数图之外，梯度才能顺着 mu 和 sigma 流回去](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig2_reparameterization.png)

## 解决：把随机性挪到参数路径之外

把采样改写成**参数的确定性函数**加上**外部噪声**：

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

这下 $\epsilon$ 与参数无关，从 $\phi$ 到 $z$ 是一条全程可微的通路，梯度可以干干净净地顺着 $\mu$ 和 $\sigma$ 走。

```python
def reparameterize(mu, logvar):
    """以可微的方式从 N(mu, sigma^2) 采样 z。

    Args:
        mu:     (B, latent_dim) 编码器均值
        logvar: (B, latent_dim) 编码器对数方差
    Returns:
        z:      (B, latent_dim) 采样结果
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

**为什么预测 `logvar` 而不是 `sigma`？** 两个理由，都是数值安全相关：

- `logvar` 不受约束，任意实数都合法；直接预测 `sigma` 必须额外保证 $\sigma > 0$（比如套个 `softplus`），又多了一个出错点。
- KL 闭式解里本来就要用 $\log \sigma^2$。直接预测它，可以避免 `log(exp(...))` 来回折腾导致下溢。

# 完整的 PyTorch 实现

下面这份代码是经典的"MNIST VAE"：全连接的编解码器、20 维潜空间、伯努利解码器。故意写得很短，方便从头读到尾。

## 网络结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1       = nn.Linear(input_dim, hidden_dim)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))   # 像素概率，落在 [0, 1]


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

## 损失函数

```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """负 ELBO = 重建项 + beta * KL。

    返回的是 batch 上的**总和**而不是均值，因为 KL 闭式解本身也是求和。
    日志输出时按需除以 batch_size 即可换成"每张图"的数值。
    """
    # 伯努利重建（适用于二值化 MNIST）；
    # 连续值数据用 F.mse_loss(reduction='sum')。
    recon = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL(N(mu, sigma^2) || N(0, I))，对 batch 与潜在维度求和。
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta * kl, recon, kl
```

把两项分别返回，多花一行代码值回票价：分别监控这两项是排查问题最有用的工具，没有之一。

## 训练循环

```python
def train_vae(model, loader, optimizer, epochs=20, device="cuda",
              warmup_epochs=10):
    model.to(device).train()
    for epoch in range(1, epochs + 1):
        # KL 退火：beta 在前 warmup_epochs 轮里从 0 线性升到 1。
        beta = min(1.0, epoch / warmup_epochs)
        ep_total = ep_recon = ep_kl = 0.0
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            recon_x, mu, logvar = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_total += loss.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()

        n = len(loader.dataset)
        print(f"epoch {epoch:3d}  beta={beta:.2f}  "
              f"loss={ep_total/n:.2f}  recon={ep_recon/n:.2f}  "
              f"kl={ep_kl/n:.2f}")
```

这循环里有三件事在实战中**不要省**：梯度裁剪、KL 退火、两项损失分别打印。读完下一节你就明白为什么。

# 你迟早会撞上的失败模式

下面这四种坑几乎人人会踩。每一节给出"症状—根因—修复"。

## 坑 1：后验坍塌

![健康 VAE vs. 坍塌 VAE 的逐维 KL 对比，以及解码器在坍塌时输出的"模糊均值图"](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig6_posterior_collapse.png)

**症状。** KL 项在前几轮直接掉到接近零，再也起不来；不管输入是什么，重建图都长得像训练集的"平均脸"；潜在空间随便走，输出毫无变化。

**根因。** 解码器太强，光靠 $\mu$ 就能重建；或者训练初期 KL 的压力盖过了重建信号。编码器发现只要把每个输入都映到 $\mu \approx 0,\ \sigma \approx 1$，就能用最便宜的方式满足 KL 项——也就是把 $x$ 直接忽略了。

**有效的修复手段，按"够用程度"排序：**

1. **KL 退火。** 让 $\beta$ 从 0 出发，在前 10–20 轮里线性升到 1（上面的训练循环就是这么做的）。给重建一段时间扎根，再让 KL 上桌。
2. **Free bits。** 对那些 KL 已经足够小的维度停止施压，免得优化器把它们直接打死。
   ```python
   kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
   kl_loss    = torch.sum(torch.clamp(kl_per_dim, min=free_bits))
   ```
3. **削弱解码器。** 减小隐藏维、加 dropout；如果是自回归解码器，前几轮先把自回归捷径砍掉。

## 坑 2：采样质量差

**症状。** 训练数据的重建看起来不错，但从 $z \sim \mathcal{N}(0, I)$ 采样出来的图就乱七八糟。

**根因。** 聚合后验 $\frac{1}{N}\sum_i q_\phi(z\mid x_i)$ 没有真正贴合先验 $p(z)$，先验里有大片解码器从没训练过的"洞"。

**修复。** 加大 $\beta$（1.5–4 比较常见）；扩大潜在维度（$20 \to 64$）；训得更久；换更强的解码器（图像任务用卷积）。如果到这一步样本质量还不够"以假乱真"，那基本就是在告诉你：换 GAN 或扩散模型。

## 坑 3：重建模糊

**症状。** 重建认得出，但平滑、缺细节；损失停在一个不上不下的位置。

**根因。** 像素独立的似然（伯努利或逐像素高斯）只惩罚单像素误差，最优解就是条件均值——而在不确定性下，均值天然就是糊的。

**修复。** 加感知损失（VGG 特征、LPIPS）；自然图像换成离散 logistic 混合似然；或者改用层次化 VAE，让高层潜变量管全局结构、低层潜变量管细节。

## 坑 4：NaN 损失与梯度爆炸

**症状。** 跑几百步以后损失变成 `NaN`，或者梯度范数突然飙升好几个数量级。

**根因。** 几乎一定是这几个里的某一个：`logvar` 没设上界，KL 里 `exp(logvar)` 溢出；BCE 的输入跑出了 `[0, 1]`；学习率过大；某个 batch 里重参数化的 $\sigma$ 下溢成 0。

**修复：**

```python
logvar = torch.clamp(logvar, min=-10, max=10)        # 给 exp(logvar) 兜底
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 优先 AdamW，lr 取 1e-3 ~ 3e-4，weight_decay 取 1e-5 量级
```

# 几个常用变体

## Beta-VAE：显式追求解耦

把 $\beta$ 调到大于 1，鼓励潜在维度去捕获**相互独立**的变化因子。ELBO 变成：

$$
\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)] - \beta \cdot D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z)),
$$

做解耦实验时 $\beta \in [2, 10]$ 比较典型。代价也很直白：$\beta$ 越大，重建越糊。

## 条件 VAE：可控生成

把编码器和解码器都条件化在标签 $y$ 上（数字类别、属性向量都行）：

$$
q_\phi(z \mid x, y), \qquad p_\theta(x \mid z, y).
$$

代码层面，把 one-hot 的 $y$ 拼到编码器输入上，再在送入解码器前拼到 $z$ 上即可。要生成指定类别，只需采样 $z \sim \mathcal{N}(0, I)$，再附上想要的标签。

## 层次化 VAE：多尺度潜变量

把潜变量堆成 $z_1, z_2, \ldots, z_L$，让 $z_{l-1}$ 在 $z_l$ 的条件下生成。下层潜变量管局部细节，上层管全局语义。现代变体（NVAE、Very Deep VAE）就是靠这一招把样本质量逼近扩散模型的。

# 实战经验

## 1. 输入归一化要匹配你用的似然

BCE 期望像素落在 $[0, 1]$；高斯似然（MSE）则在零均值数据上更稳。

```python
# MNIST：ToTensor() 已经给到 [0, 1]
transform = transforms.ToTensor()

# 连续数据：标准化到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

## 2. 潜在维度从小开始

MNIST 这种规模，`latent_dim = 20` 是个合理起点。太小，重建被瓶颈卡死；太大，更容易后验坍塌，训练也变慢。

## 3. 永远把重建项与 KL 项分开打印

![训练过程中的 ELBO 分解：KL 退火让 KL 项干净地爬上来，重建项稳步下降](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig5_elbo_decomposition.png)

健康的训练曲线长这样：重建项单调下降，KL 稳定在一个非平凡的值（每个活跃维度几个 nat）。如果 KL 一直贴在零，你正在后验坍塌；如果 KL 爆掉，重建项在被忽视。

## 4. 把潜在空间画出来

二维潜空间（或更高维做 PCA/t-SNE 之后），按类别给 $\mu(x)$ 上色画散点图。你应该看到的是**有重叠但仍可分**的若干类簇，并且大致铺满先验。

![MNIST 上的 VAE 潜空间：十个类形成相互重叠的平滑簇，大致填满先验](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig3_latent_scatter.png)

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_latent_space(model, loader, device="cuda"):
    model.eval()
    mus, labels = [], []
    for x, y in loader:
        mu, _ = model.encoder(x.view(-1, 784).to(device))
        mus.append(mu.cpu()); labels.append(y)
    mus    = torch.cat(mus).numpy()
    labels = torch.cat(labels).numpy()
    plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10",
                s=8, alpha=0.7)
    plt.colorbar(); plt.title("VAE 潜在空间 (2D)"); plt.show()
```

## 5. 既要采样，也要在样本之间走

下面这两个工具函数你会反复用到：

```python
@torch.no_grad()
def sample_vae(model, n=16, device="cuda"):
    model.eval()
    z = torch.randn(n, model.encoder.fc_mu.out_features).to(device)
    return model.decoder(z).cpu().view(-1, 28, 28)


@torch.no_grad()
def interpolate(model, x1, x2, steps=10, device="cuda"):
    model.eval()
    mu1, _ = model.encoder(x1.view(1, -1).to(device))
    mu2, _ = model.encoder(x2.view(1, -1).to(device))
    ts = torch.linspace(0, 1, steps + 1, device=device).view(-1, 1)
    z  = (1 - ts) * mu1 + ts * mu2
    return model.decoder(z).cpu().view(-1, 28, 28)
```

一段干净的潜在插值，是证明你的 VAE "真的训出来了"的最有说服力的可视化。

![潜在空间插值：在两个数字编码之间走一条直线，每一步都解码出一张图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig4_latent_interpolation.png)

# VAE 与其他生成模型的对比

![潜空间几何对比，以及 AE / VAE / GAN 在五个能力维度上的表现](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8-vae-%E8%AF%A6%E8%A7%A3/fig7_model_comparison.png)

| 模型 | 潜在空间 | 训练 | 生成质量 | 可解释性 |
|---|---|---|---|---|
| **VAE** | 显式、平滑 | 稳定（ELBO） | 凑合，偏糊 | 高（常常解耦） |
| **GAN** | 隐式 | 不稳定（对抗） | 锐利、逼真 | 低（常见模式坍塌） |
| **扩散模型** | 隐式（按步） | 稳定（去噪） | 当前最佳 | 中（迭代采样） |
| **自回归** | 无 | 稳定（似然） | 高，但慢 | 低（顺序生成） |

**适合用 VAE 的场景：** 下游任务需要显式潜在表示；想要训练稳定、不搞对抗博弈；关心解耦或可解释性。

**别用 VAE 的场景：** 目标是照片级真实（用 GAN 或扩散）；只要序列上的似然（用自回归）。

# 总结：VAE 五步走

1. **编码器**输出 $\mu_\phi(x)$ 与 $\log \sigma^2_\phi(x)$，不再是单个确定的编码。
2. **重参数化：** $z = \mu + \sigma \odot \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$——梯度顺利通过。
3. **解码器**从 $z$ 重建 $\hat{x}$。
4. **损失 = 负 ELBO：** 重建 + $\beta$ · KL。
5. **采样：** $z \sim \mathcal{N}(0, I) \to$ 解码器 $\to \hat{x}$。

**最影响结果的几个超参：** 潜在维度（从 20 起步）、$\beta$（默认 1.0，追解耦时调大）、学习率（Adam 1e-3，配 10–20 轮 KL 退火，梯度裁剪 1.0）。

**早晚会踩的坑：** 后验坍塌（用 KL 退火或 free bits）、采样模糊（加大潜在维度或感知损失）、NaN 损失（钳制 `logvar`、裁剪梯度）。

# 延伸阅读

- Kingma, D.P. & Welling, M. (2013). [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114). VAE 的原始论文。
- Higgins, I. et al. (2017). [*beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/forum?id=Sy2fzU9gl).
- Doersch, C. (2016). [*Tutorial on Variational Autoencoders*](https://arxiv.org/abs/1606.05908). 推导细致的入门教程。
- Sohn, K., Lee, H. & Yan, X. (2015). [*Learning Structured Output Representation using Deep Conditional Generative Models*](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html). CVAE 论文。
- Vahdat, A. & Kautz, J. (2020). [*NVAE: A Deep Hierarchical Variational Autoencoder*](https://arxiv.org/abs/2007.03898). 把 VAE 样本质量推到能与 GAN 对抗的层次化版本。
- Kingma, D.P. & Welling, M. (2019). [*An Introduction to Variational Autoencoders*](https://arxiv.org/abs/1906.02691). 全面的综述专著。
