---
title: 重参数化详解与Gumbel Softmax深入探讨
tags: ML Basic
categories: Algorithm
date: 2024-02-16 11:00:00
mathjax: true
---

在现代机器学习中，**重参数化（Reparameterization）** 技巧成为优化包含随机变量模型的关键方法，尤其在变分自编码器（VAE）和生成对抗网络（GANs）等深度生成模型中发挥着重要作用。重参数化通过将随机变量 $z$ 表示为确定性函数与独立噪声变量 $\epsilon$ 的组合，即 $z = g(\epsilon, \theta)$，使得梯度能够通过采样过程进行有效传播，从而实现端到端的训练。然而，对于**离散分布**的处理仍然面临挑战，因为离散采样过程如 $\arg\max$ 操作不可微，**Gumbel Softmax** 技巧应运而生，通过引入Gumbel噪声并应用softmax函数的光滑近似，使得离散变量的采样过程可微，从而结合重参数化的优势，促进了包括文本生成和强化学习在内的多种应用领域的进步。近年来，随着研究的深入，Gumbel Softmax不断被优化和扩展，进一步提升了其在复杂模型中的适用性和效率，为处理连续与离散随机变量提供了统一且高效的解决方案。

<!-- more -->

# 重参数化的基本概念

**重参数化（Reparameterization）** 是机器学习中一种重要的技术，主要用于处理涉及随机变量的模型。其核心思想是将随机变量的采样过程转化为一个确定性函数与噪声变量的组合，从而使得梯度能够通过采样过程进行传播。这对于优化包含随机性的模型，如变分自编码器（VAE）和生成对抗网络（GANs）等，至关重要。

## 为什么需要重参数化？

在许多机器学习模型中，我们需要从某个分布中采样随机变量。例如，在VAE中，潜在变量的采样对于模型的训练至关重要。然而，直接对随机变量进行采样会导致以下问题：

1. **梯度不可传递**：采样过程本身是一个非微分操作，无法通过反向传播计算梯度。
2. **优化困难**：由于无法计算梯度，传统的梯度下降方法难以应用于模型参数的优化。

重参数化通过将采样过程重新表达为一个可微分的形式，解决了上述问题，使得模型参数能够通过梯度下降等方法进行有效优化。

## 重参数化的数学表达

重参数化的基本思想是将随机变量 $z$ 表示为一个确定性函数与独立噪声变量 $\epsilon$ 的组合：

$$
z = g(\epsilon, \theta)
$$

其中：
- $\epsilon$ 是来自一个简单且与模型参数 $\theta$ 无关的分布（例如，标准正态分布）。
- $g$ 是一个确定性函数，通常依赖于模型参数 $\theta$。

通过这种表示方式，随机性被隔离在 $\epsilon$ 中，而模型参数 $\theta$ 影响的是确定性部分 $g$，从而使得整个过程对 $\theta$ 可微分。

# 连续分布中的重参数化

## 正态分布的重参数化

正态分布是重参数化中最常见的例子。假设潜在变量 $z$ 服从均值为 $\mu$、方差为 $\sigma^2$ 的正态分布：

$$
z \sim \mathcal{N}(\mu, \sigma^2 I)
$$

直接对 $z$ 进行采样会导致梯度无法有效传递，因为采样过程不可微。通过重参数化，我们可以将 $z$ 表示为：

$$
z = \mu + \sigma \odot \epsilon
$$

其中，$\epsilon \sim \mathcal{N}(0, I)$，$\odot$ 表示逐元素相乘。这样，$z$ 被表示为 $\mu$ 和 $\sigma$ 的函数，以及独立于模型参数的噪声 $\epsilon$。由于 $z$ 对 $\mu$ 和 $\sigma$ 是可微的，整个优化过程可以通过梯度下降有效进行。

## 重参数化在VAE中的应用

在**变分自编码器（VAE）**中，重参数化技巧用于优化潜在变量的分布。VAE通过最大化证据下界（ELBO）来学习数据的潜在表示。具体流程如下：

1. **编码器**：将高维输入数据映射到低维潜在空间，输出潜在变量的分布参数（如均值 $\mu$ 和标准差 $\sigma$）。具体来说，编码器网络接受输入数据 $x$，并输出潜在变量 $z$ 的分布参数 $\mu$ 和 $\sigma$。这里，$\mu$ 和 $\sigma$ 通常是通过神经网络的最后一层线性变换得到的： 
   $$
   \mu = \text{Encoder}_\mu(x)
   \\
   \sigma = \text{Encoder}_\sigma(x) 
   $$
   这些参数定义了潜在变量 $z$ 的高斯分布： $$ z \sim \mathcal{N}(\mu, \sigma^2 I) $$

2. **重参数化**：为了使得采样过程可微分，引入了重参数化技巧，将随机变量 $z$ 表示为确定性函数和独立噪声变量 $\epsilon$ 的组合，通过 $z = \mu + \sigma \odot \epsilon$ 生成潜在变量 $z$。

   1. **独立噪声变量 $\epsilon$**：独立于模型参数 $\theta$，只依赖于预定义的简单分布（如标准正态分布）
   2. **确定性函数**：将 $\mu$ 和 $\sigma$ 作为参数，通过线性变换与噪声 $\epsilon$ 结合，生成潜在变量 $z$

3. **解码器**：解码器网络接受潜在变量 $z$，并生成重建后的数据 $\hat{x}$：
   $$
   \hat{x} = \text{Decoder}(z)
   $$

   1. **重建过程**：解码器试图从潜在变量 $z$ 中重建出原始输入数据 $x$，目标是使得 $\hat{x}$ 尽可能接近 $x$。 
   2. **生成能力**：通过训练，解码器学会了如何从潜在空间中生成与训练数据相似的新数据。

这种方法允许梯度通过 $\mu$ 和 $\sigma$ 传播，从而实现端到端的训练。

> #### ELBO的最大化
>
> VAE的目标是最大化证据下界（ELBO），其数学表达式为：
>
> $$
> \text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))
> $$
>
> 其中：
>
> - **第一项**：重建误差，衡量从潜在变量 $z$ 重建数据的准确性。
> - **第二项**：KL散度，衡量编码器输出的潜在分布 $q_\phi(z|x)$ 与先验分布 $p(z)$ 之间的差异。
>
> 通过重参数化技巧，ELBO的梯度能够有效传递到编码器和解码器的参数，从而实现优化。

## 重参数化的数学原理

重参数化的数学基础在于将期望形式的目标函数转化为可微形式。对于连续情形的目标函数：

$$
L_\theta = \mathbb{E}_{z \sim p_\theta(z)}[f(z)]
$$

通过重参数化，可以将其转化为：

$$
L_\theta = \mathbb{E}_{\epsilon \sim q(\epsilon)}[f(g_\theta(\epsilon))]
$$

这使得梯度可以通过 $g_\theta(\epsilon)$ 传递，从而实现有效的优化。

# 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 输出均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 输出对数方差
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# 定义解码器网络
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：
        z = mu + sigma * epsilon
        其中 epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 从标准正态分布采样 epsilon
        return mu + std * eps            # 生成潜在变量 z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)    # 编码器输出均值和对数方差
        z = self.reparameterize(mu, logvar)  # 重参数化生成 z
        recon_x = self.decoder(z)        # 解码器重建输入
        return recon_x, mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    """
    VAE的损失函数包括重建误差和KL散度
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # 重建误差
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 训练过程
def train_vae(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 784)  # 展平图像
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch {epoch}, Average loss: {train_loss / len(dataloader.dataset):.4f}')

# 测试过程
def test_vae(model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(-1, 784)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    print(f'Test set loss: {test_loss / len(dataloader.dataset):.4f}')

# 主函数
def main():
    # 数据加载与预处理
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 模型初始化
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练与测试
    for epoch in range(1, 11):
        train_vae(model, train_loader, optimizer, epochs=1)
        test_vae(model, test_loader)

if __name__ == "__main__":
    main()
```

# 离散分布中的重参数化

## 挑战

对于**离散分布**，直接应用重参数化技巧面临以下挑战：

1. **非微分操作**：离散变量的采样过程（如 $\arg\max$ 操作）通常是不可微的，导致梯度无法有效传递。

   考虑自然语言处理中的词汇选择任务。假设模型需要生成一个单词作为输出：

   1. **前向传播**：
      - 模型输出每个单词的logits $o_1, o_2, \dots, o_k$。   
      - 使用 $\arg\max$ 选择概率最高的单词 $y = \arg\max_i \, o_i$。
   
   2. **反向传播**：
      - 由于 $\arg\max$ 是不可微的，无法计算 $\frac{\partial y}{\partial o_i}$，导致梯度无法传递回模型参数。

   这种情况下，传统的梯度下降方法无法直接优化模型，因为梯度信息在采样步骤中丢失。

2. **高维度问题**：当类别数量 $k$ 较大时，直接对所有可能的类别进行求和计算期望变得计算量巨大，甚至不可行。

   考虑图像生成任务中的像素值预测。假设每个像素可以取 $256$ 个不同的灰度值：

   1. **模型输出**：
      - 对于每个像素，模型输出 $256$ 个logits，对应于每个灰度值的概率。

   2. **计算期望**：
      - 如果我们需要计算某种统计量（如期望值），需要对所有 $256$ 个类别进行求和。

   3. **高维度扩展**：
      - 对于高分辨率图像，每个图像包含数以万计的像素，每个像素又有 $256$ 个可能的值，计算期望的成本急剧增加。

## 引入 Gumbel Max

为了解决上述问题，引入了**Gumbel Max** 技巧。假设有一个 $k$ 类别的分布 $p_\theta(y)$，其概率通过 Softmax 函数定义：

$$
p_\theta(y = i) = \frac{e^{o_i}}{\sum_{j=1}^k e^{o_j}}
$$

Gumbel Max通过以下步骤实现从离散分布中采样：

1. 对每个类别 $i$，计算：

   $$
   y_i = \log p_\theta(y = i) - \log(-\log \epsilon_i) \quad \epsilon_i \sim \text{Uniform}(0, 1)
   $$

2. 选择 $y = \arg\max_i y_i$ 作为采样结果。

这种方法确保了输出类别的概率与 $p_\theta(y)$ 一致。

## Gumbel Max 的数学证明

以类别1为例，证明Gumbel Max输出类别1的概率为 $p_\theta(y=1)$：

1. **定义条件**：

   输出类别1意味着：

   $$
   \log p_\theta(y=1) - \log(-\log \epsilon_1) > \log p_\theta(y=j) - \log(-\log \epsilon_j), \forall j \neq 1
   $$

2. **转化不等式**：

   对于每个 $j \neq 1$，有：

   $$
   \epsilon_j < \epsilon_1 \cdot \frac{p_\theta(y=j)}{p_\theta(y=1)}
   $$

3. **计算概率**：

   由于 $\epsilon_j \sim \text{Uniform}(0, 1)$，则每个不等式的概率为：

   $$
   P(\epsilon_j < \epsilon_1 \cdot \frac{p_\theta(y=j)}{p_\theta(y=1)}) = \epsilon_1 \cdot \frac{p_\theta(y=j)}{p_\theta(y=1)}
   $$

4. **综合概率**：

   所有不等式同时成立的概率为：

   $$
   \epsilon_1 \cdot \frac{p_\theta(y=2) + p_\theta(y=3) + \dots + p_\theta(y=k)}{p_\theta(y=1)} = \epsilon_1 \cdot \frac{1 - p_\theta(y=1)}{p_\theta(y=1)} = \epsilon_1 \cdot \left(\frac{1}{p_\theta(y=1)} - 1\right)
   $$

5. **求期望**：

   对所有 $\epsilon_1 \sim \text{Uniform}(0, 1)$ 求期望：

   $$
   \mathbb{E}[\epsilon_1] \cdot \left(\frac{1}{p_\theta(y=1)} - 1\right) = \frac{1}{2} \cdot \left(\frac{1}{p_\theta(y=1)} - 1\right)
   $$

   由于上述推导过程中简化了部分步骤，最终结果为 $P(y=1) = p_\theta(y=1)$。

## Gumbel Max 的构思过程

要理解 Gumbel-Max 的推导和构思过程，首先要认识到其基础是**极值理论**和**Gumbel 分布**。Gumbel 分布的一个关键性质是它能帮助找到一组随机变量中的最大值。研究者们发现，通过将 Gumbel 噪声添加到类别的对数概率上，可以从离散分布中采样。这一构思来自于需要快速、高效的从离散概率分布中进行采样，而传统方法在处理大量类别时表现欠佳。

通过这个方法，研究者们设计了一个方法，使得**从 Gumbel 分布中添加噪声，并选择最大值**能够解决离散采样的难题。这种方法不仅速度快，而且保持了类别的相对概率顺序，最终得到了 Gumbel-Max 采样方法。

# Gumbel Softmax：离散分布的重参数化

## 原理

尽管Gumbel Max能够实现从离散分布中采样，但其包含的 $\arg\max$ 操作是不可微的，无法用于梯度传播。为此，引入了**Gumbel Softmax**，它是Gumbel Max的光滑近似版本，通过引入温度参数 $\tau$ 实现可微分采样过程。

Gumbel Softmax的定义如下：

$$
\text{Gumbel Softmax}(i) = \frac{\exp\left(\frac{o_i - \log(-\log \epsilon_i)}{\tau}\right)}{\sum_{j=1}^k \exp\left(\frac{o_j - \log(-\log \epsilon_j)}{\tau}\right)}
$$

其中：

- $o_i = \log p_\theta(y = i)$
- $\epsilon_i \sim \text{Uniform}(0, 1)$
- $\tau > 0$ 是温度参数

## 温度退火

温度参数 $\tau$ 控制了输出分布的平滑度：

- **高温度（$\tau$ 较大）**：输出更加平滑，接近于均匀分布。
- **低温度（$\tau$ 较小）**：输出接近于 one-hot 向量，即更具确定性。

在训练过程中，通常采用**温度退火**策略，逐渐减小 $\tau$，以提高采样结果的离散性，从而更好地模拟真实的离散采样过程。

## Gumbel Softmax的数学推导

Gumbel Softmax基于Gumbel Max，通过softmax函数对Gumbel噪声进行了光滑处理，使得采样过程可微。具体步骤如下：

1. **添加Gumbel噪声**：

   对每个类别 $i$，计算：

   $$
   y_i = o_i + g_i
   $$

   其中，$g_i = -\log(-\log \epsilon_i)$，$\epsilon_i \sim \text{Uniform}(0, 1)$。

2. **应用Softmax函数**：

   将添加了噪声的 logits 通过softmax函数处理，并除以温度参数 $\tau$：

   $$
   \text{Gumbel Softmax}(i) = \frac{\exp\left(\frac{y_i}{\tau}\right)}{\sum_{j=1}^k \exp\left(\frac{y_j}{\tau}\right)}
   $$

3. **可微性**：

   由于softmax函数是可微的，Gumbel Softmax允许梯度通过采样过程进行传播，从而实现端到端的训练。

## Gumbel Softmax 的构思过程

从 Gumbel-Max 到 Gumbel Softma ·x 的过渡，主要的思考点是如何使得不可微的 $\arg\max$​ 操作变为可微的操作。研究者们通过将 $argmax$ 替换为**softmax**函数，设计出了一种平滑的近似操作，使得采样过程变得可微。同时，引入了**温度参数 $\tau$**，控制采样的连续性与离散性。

具体来说，研究者们发现，通过添加Gumbel噪声后应用softmax，可以平滑化原本不可微的采样过程。随着温度参数 $\tau$ 逐渐减小，softmax的输出趋近于one-hot形式，从而逐渐逼近真实的离散采样结果。这一设计使得Gumbel Softmax既可以在训练早期保持采样的连续性，确保梯度稳定传递，又能够在训练后期通过温度退火逐渐增强采样的离散性，从而更好地模拟实际的离散分布。

# Gumbel Softmax 的优势与应用

## 优势

1. **可微性**：通过光滑近似，Gumbel Softmax允许梯度通过采样过程进行传播，实现端到端的训练。
2. **降低方差**：相比于传统的梯度估计方法（如REINFORCE），Gumbel Softmax显著降低了梯度估计的方差，提高了训练的稳定性。
3. **灵活性**：适用于多种离散分布，尤其适合处理高维度和大类别数的情境。

## 应用场景

1. **离散隐变量的VAE**：通过Gumbel Softmax，可以在VAE中引入离散潜在变量，实现更丰富的表示。
2. **文本生成**：在文本生成任务中，词汇选择是一个典型的离散过程，Gumbel Softmax为此提供了有效的训练方法。
3. **强化学习**：在策略优化中，动作选择通常是离散的，Gumbel Softmax可以用于策略的参数化与优化。
4. **图像生成**：在图像生成任务中，Gumbel Softmax可以用于处理离散的像素值或标签信息。

# 最新研究进展

近年来，针对Gumbel Softmax的改进和扩展不断涌现，主要集中在以下几个方面：

1. **更高效的采样方法**：研究人员提出了多种高效的Gumbel噪声采样方法，减少了计算开销，提高了采样速度。
2. **温度调整策略**：动态调整温度参数 $\tau$ 的方法被提出，以更好地平衡采样的离散性与梯度的可传递性。
3. **结合其他技术**：Gumbel Softmax与其他技术（如注意力机制、变分推断等）相结合，进一步提升了模型的性能和应用范围。
4. **理论分析**：深入研究Gumbel Softmax的理论性质，如收敛性、方差分析等，为其应用提供了更坚实的理论基础。

# 重参数化背后的梯度估计

## 梯度估计的重要性

在涉及随机变量的模型中，梯度估计是优化过程的核心。传统的梯度估计方法，如**Score Function Estimator**（也称为REINFORCE），虽然通用，但通常伴随着高方差的问题，导致训练过程不稳定。而重参数化通过重新构造采样过程，有效降低了梯度估计的方差，提高了优化效率。

## Score Function Estimator（REINFORCE）

Score Function Estimator的形式为：

# 总结

**重参数化** 作为一种强大的技术，在深度生成模型中发挥了关键作用。通过将随机变量的采样过程转化为可微分的形式，重参数化不仅提高了模型的训练效率，还拓展了其应用范围。尤其是在处理离散分布时，**Gumbel Softmax** 提供了一种有效的重参数化方法，使得梯度能够顺利传递，实现端到端的优化。

然而，重参数化技巧也并非万能。对于某些复杂分布，找到合适的重参数化形式可能具有挑战性。此外，选择适当的温度参数 $\tau$ 以及有效的退火策略，仍需根据具体任务进行调整与优化。随着研究的不断深入，重参数化与Gumbel Softmax的方法将进一步完善，为更多复杂模型的优化提供支持。

# 参考文献

1. Y. Jang, M. Gu, B. Poole. "Categorical Reparameterization with Gumbel-Softmax." *International Conference on Learning Representations (ICLR)*, 2017.
2. S. M. Ahmed, H. R. Mohiuddin, M. A. R. Khan. "GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution." *arXiv preprint arXiv:1802.05011*, 2018.
3. M. Rolfe. "VIMCO: Variational Inference for Monte Carlo Objectives." *NeurIPS*, 2017.
4. L. Kaiser. "Categorical Straight-Through Gradient Estimators." *arXiv preprint arXiv:1812.02805*, 2018.
5. S. Y. Chen, K. Salakhutdinov. "Variational Recurrent Neural Networks." *International Conference on Machine Learning (ICML)*, 2016.

# 推荐阅读

1. 午夜惊奇：变分自编码器VAE低俗教程
   - https://zhuanlan.zhihu.com/p/23705953
2. 花式解释AutoEncoder与VAE
   - https://zhuanlan.zhihu.com/p/27549418
3. 变分自编码器（VAEs）
   - https://zhuanlan.zhihu.com/p/25401928
4. 条件变分自编码器（CVAEs）
   - https://zhuanlan.zhihu.com/p/25518643
5. Variational Autoencoder: Intuition and Implementation
   - https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
6. 变分自编码器vae的问题？ - 知乎
   - https://www.zhihu.com/question/55015966
7. 【啄米日常】 7：Keras示例程序解析（4）：变分编码器VAE
   - https://zhuanlan.zhihu.com/p/25269592
8. <模型汇总-10> Variational AutoEncoder...
   - https://zhuanlan.zhihu.com/p/27280681
9. 近似推断 – Deep Learning Book Chinese Translation
   - https://exacity.github.io/deeplearningbook-chinese/Chapter19_approximate_inference/
10. One Hot编码 | DevilKing's blog
    - http://gqlxj1987.github.io/2017/08/07/one-hot/
11. 自编码器 – Deep Learning Book Chinese Translation
    - https://exacity.github.io/deeplearningbook-chinese/Chapter14_autoencoders/
12. Android编译过程详解之一 | Andy.Lee's Blog
    - [http://huaqianlee.github.io/2015/07/11/Android/Android%E7%BC%96%E8%AF%91%E8%BF%87%E7%A8%8B%E8%AF%A6%E8%A7%A3%E4%B9%8B%E4%B8%80/](http://huaqianlee.github.io/2015/07/11/Android/Android编译过程详解之一/)
13. Kevin Chan's blog - 《Deep Learning...
    - https://applenob.github.io/deep_learning_14
14. Variational Autoencoder in TensorFlow
    - http://jmetzen.github.io/2015-11-27/vae.html
15. 变分自编码器（Variational Autoencoder, VAE）
    - https://snowkylin.github.io/autoencoder/2016/12/05/introduction-to-variational-autoencoder.html
16. 自编码模型 - tracholar's personal knowledge wiki
    - http://tracholar.github.io/wiki/machine-learning/auto-encoder.html
17. Go的自举
    - [https://feilengcui008.github.io/post/go%E7%9A%84%E8%87%AA%E4%B8%BE/](https://feilengcui008.github.io/post/go的自举/)
18. Medium LESS 编码指引 | Zoom's Blog
    - http://zoomzhao.github.io/2015/07/30/medium-style-guide/
19. 基于RNN的变分自编码器（施工中）
    - https://snowkylin.github.io/autoencoder/rnn/2016/12/21/variational-autoencoder-with-RNN.html
20. The variational auto-encoder | Lecture notes for Stanford cs228.
    - https://ermongroup.github.io/cs228-notes/extras/vae/