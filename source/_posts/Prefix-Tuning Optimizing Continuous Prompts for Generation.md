---
title: Prefix Tuning —— Optimizing Continuous Prompts for Generation
tags: PEFT
categories: Paper
date: 2024-08-17 9:00:00
mathjax: true
---

在自然语言生成任务中，微调是使用大型预训练语言模型的常见方式。然而，微调需要修改所有模型参数，因此每个任务都需要存储一份完整的模型副本，这对存储和计算成本提出了很高的要求。本文提出了一种新的方法：Prefix-Tuning，它通过优化一个小的任务特定连续向量（称为前缀），而保持语言模型参数不变。与微调相比，Prefix-Tuning只需存储前缀，显著降低了存储需求。

<!-- more -->

# 背景介绍

轻量级微调（Lightweight Fine-Tuning）是一种针对大型预训练模型的优化技术，旨在减少存储和计算资源的需求，同时保持模型在特定任务上的高性能表现。随着自然语言处理模型规模的迅速扩大，如GPT-3具有1750亿个参数，传统的微调方法面临着存储和计算成本的巨大挑战。为了解决这个问题，研究者们提出了多种轻量级微调方法，包括适配器微调（Adapter-Tuning）、提示词微调（Prompting）、以及更近一步的前缀微调（Prefix-Tuning）。

在传统微调中，所有的模型参数都会根据下游任务进行调整，这需要为每个任务保存一个完整的模型副本，导致存储需求呈线性增长。而轻量级微调通过仅调整一小部分参数或外部组件，大大减少了存储需求。例如，**适配器微调（Adapter-Tuning）**在每个模型层之间插入小型的可训练模块（适配器），这些适配器能够捕捉任务特定的信息，而无需大规模更新模型的原始参数。这样的方法不仅减少了存储开销，还允许不同任务之间共享相同的预训练模型，使多任务学习更加高效。

**提示词微调（Prompting）**则更进一步，它完全不改变模型的原始参数，而是通过在输入前添加设计好的自然语言提示来引导模型生成所需的输出。这种方法的优势在于，它无需存储任何额外的参数，并且可以灵活地适应不同的任务。然而，其性能很大程度上依赖于提示的设计质量，且在处理超长上下文时效果有限。

**前缀微调（Prefix-Tuning）**是一种本文提出的方法，结合了适配器微调和提示词微调的优点。它通过在输入前增加一个可训练的连续向量（前缀），引导模型生成特定任务的输出。与提示词微调使用的离散自然语言提示不同，前缀微调使用的连续前缀是可学习的参数向量，这些向量不对应于实际的单词，而是直接影响模型的内部状态。这种方法既不需要大规模修改模型参数，又能够通过调整前缀灵活适应不同任务，展示了比传统微调和其他轻量级方法更优的存储效率和性能表现。

轻量级微调的核心思想是通过最小化参数调整来优化计算资源的利用，同时确保模型在各类任务上的适应性。这种方法为大规模语言模型的实际应用提供了新的可能，使得这些模型可以在更广泛的环境中以更低的成本被有效使用，从而推动了自然语言处理领域的进一步发展。

## 轻量级微调的现有方法的细节

- **Adapter-Tuning**: 通过在预训练语言模型层之间插入小的可训练模块（适配器），实现任务特定的调整。这种方法减少了模型参数的调整，降低了存储需求。在 Adapter-Tuning 中，适配器模块通常是一个小型的前馈神经网络（如全连接层），其输入是从原始模型的某一层提取的特征。通过调整这些适配器的参数，模型可以学习到特定任务所需的特征表示，同时保持原有预训练模型的参数不变。

  数学上，适配器的实现可以描述为：

  $$h^′=W_2(\text{ReLU}(W_1h+b_1))+b_2$$

  其中，$h$ 是从预训练模型中提取的特征，$W_1$ 和 $W_2$ 是适配器的可训练权重矩阵，$b_1$ 和 $b_2$ 是偏置项。适配器的输出 $h^′$ 将被送回原模型的后续层进行进一步处理。

  根据最新的研究，例如 UniPELT 框架结合了多种微调策略，包括适配器和提示调优（Prompt Tuning），实现了参数效率和模型性能的最佳平衡([Papers with Code](https://paperswithcode.com/paper/dynamic-adapter-meets-prompt-tuning-parameter))。此框架允许模型在多任务环境中有效传递，而无需大量重新训练基础模型参数。使用适配器的方法在不同的数据集（如 GLUE 基准、领域特定数据集和 SQuAD）上的表现证明了这种策略在减少训练参数数量的同时，依然保持了与全模型微调相当的性能([Papers with Code](https://paperswithcode.com/paper/parameter-efficient-fine-tuning-with-adapters))。

- **Prompting**: 不进行任何参数调整，而是通过在输入前添加自然语言提示（prompt）来引导预训练模型生成期望的输出。这种方法虽然不需要存储额外参数，但它的效果依赖于提示的设计质量，且在处理超长上下文时有一定局限。目前常见的 Prompting 策略包括：

  - **Manual Prompting**: 通过人工设计提示词，引导模型产生特定输出。这种方法的效果依赖于提示词的设计质量和模型的预训练知识。
  - **Automatic Prompting**: 通过自动化方法生成提示词，如使用遗传算法或梯度下降方法优化提示词，使模型在下游任务上表现更好。
  - **Soft Prompting**: 使用连续的向量表示替代离散的词汇提示，允许在提示词中引入更多的细微差别，从而提高模型的任务表现。

## Prefix-Tuning方法

Prefix-Tuning是一种新的轻量级微调方法，它借鉴了prompting的思想，通过在输入前添加一个可训练的连续向量（前缀）来引导模型生成输出。不同于prompting使用离散的提示词，Prefix-Tuning使用连续的前缀，这些前缀是自由参数，不对应于实际的单词。

### 方法详解

在Prefix-Tuning中，模型输入被修改为 $z = [\text{PREFIX}; x; y]$，其中 $\text{PREFIX}$ 是前缀，$x$ 是输入，$y$ 是输出。在生成过程中，Transformer模型可以将这些前缀看作“虚拟的token”进行处理。

假设我们有一个自回归语言模型 $p(y|x)$，模型的激活状态在每个时间步 $i$ 为 $h_i$。在Prefix-Tuning中，前缀是可训练的连续向量，初始化为一个参数矩阵 $P_\theta$：

$$
h_i =
\begin{cases}
P_\theta[i,:] & \text{if } i \in \text{Pidx} \\
\text{LM}_\phi(z_i, h_{<i}) & \text{otherwise}
\end{cases}
$$

其中，$\text{Pidx}$ 表示前缀的索引序列，$\text{LM}_\phi$ 表示语言模型，$\phi$ 是语言模型的参数，这些参数在训练过程中保持不变。

在传统微调方法中，模型参数 $\phi$ 是可训练的，优化目标是最大化条件概率 $p_\phi(y|x)$：

$$
\max_\phi \log p_\phi(y|x) = \sum_{i \in \text{Yidx}} \log p_\phi(z_i | h_{<i})
$$

在Prefix-Tuning中，模型参数 $\phi$ 保持不变，只优化前缀参数 $\theta$：

$$
\max_\theta \log p_\phi(y|x; \theta) = \sum_{i \in \text{Yidx}} \log p_\phi(z_i | h_{<i}; \theta)
$$

这种优化策略通过调整前缀来影响语言模型的激活状态 $h_i$，从而改变生成结果。

### 参数化策略

在Prefix-Tuning方法中，直接优化前缀矩阵$P_\theta$的参数可能导致训练不稳定。为了缓解这个问题，研究者们提出了一种重新参数化的策略，通过将一个小型的基础矩阵$P_\theta'$和一个前馈神经网络（$\text{MLP}_\theta$）组合起来生成最终的前缀矩阵$P_\theta$。这种策略不仅能提高训练的稳定性，还能更有效地调整前缀矩阵的参数。

#### 1. 直接优化前缀矩阵的挑战

直接优化$P_\theta$（即直接将前缀矩阵作为可训练参数）在实践中可能面临以下挑战：

- **高维度参数空间**：如果直接优化$P_\theta$，其维度可能会非常高，这使得优化过程更容易陷入局部最优，难以收敛到更好的解决方案。
- **不稳定的梯度**：在深度神经网络的训练过程中，梯度的稳定性至关重要。直接优化高维的前缀矩阵容易导致梯度不稳定，进而导致训练过程不稳定，甚至可能出现梯度爆炸或消失的问题。

#### 2. 重新参数化策略的具体实现

为了应对这些问题，Prefix-Tuning方法引入了一个重新参数化的策略，将前缀矩阵$P_\theta$分解为两个部分：一个小型的基础矩阵$P_\theta'$和一个前馈神经网络$\text{MLP}_\theta$。具体来说，这个过程包括以下几个步骤：

1. **初始化基础矩阵**：$P_\theta'$是一个小型的可训练参数矩阵，其初始值通常是随机生成的。这个矩阵的维度较小，相对容易优化。
2. **前馈神经网络变换**：使用一个前馈神经网络$\text{MLP}_\theta$对基础矩阵$P_\theta'$的每一行进行变换，生成最终的前缀矩阵$P_\theta$。公式表示如下：

$$
P_\theta[i,:] = \text{MLP}_\theta(P_\theta'[i,:])
$$

这里，$\text{MLP}_\theta$可以被视为一个函数映射，它接收一个小型向量作为输入，输出一个更大的向量。通过这种映射，我们可以更灵活地学习前缀矩阵的结构和特征。

**训练过程中的稳定性**：通过先学习一个小型的基础矩阵并使用前馈网络变换来生成高维度的前缀矩阵，这种方法有效减少了优化过程中的不稳定性。因为前馈网络的输出是连续且光滑的，这使得梯度更新更加稳定和可控。

**训练后的简化**：在训练完成后，前馈神经网络$\text{MLP}_\theta$的参数可以被丢弃，只需保留最终得到的前缀矩阵$P_\theta$。这样做的好处是进一步减少了存储需求，因为无需保存整个网络的参数，只需保存前缀矩阵即可。

# 示例代码

以下是使用PyTorch实现的一个Prefix-Tuning的简单示例，展示了如何在模型输入前添加一个可训练的前缀：

```python
import torch
import torch.nn as nn

# 定义一个简单的前缀模型
class PrefixModel(nn.Module):
    def __init__(self, model, prefix_length, hidden_size):
        super(PrefixModel, self).__init__()
        self.model = model  # 预训练的语言模型，例如GPT-2
        self.prefix_length = prefix_length
        self.prefix_embedding = nn.Parameter(torch.randn(prefix_length, hidden_size))

    def forward(self, input_ids):
        # 将前缀嵌入添加到输入前面
        prefix = self.prefix_embedding.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        input_embeds = self.model.transformer.wte(input_ids)
        input_embeds = torch.cat([prefix, input_embeds], dim=1)
        
        # 使用模型的前向传播计算输出
        outputs = self.model(inputs_embeds=input_embeds)
        return outputs

# 示例模型和输入
from transformers import GPT2Model

gpt2_model = GPT2Model.from_pretrained('gpt2')
prefix_model = PrefixModel(gpt2_model, prefix_length=10, hidden_size=768)

input_ids = torch.tensor([[50256, 50257, 50258]])
outputs = prefix_model(input_ids)
print(outputs)
```

在上面的代码中，我们定义了一个 `PrefixModel` 类，通过添加一个可训练的前缀嵌入到输入序列的前面，实现了Prefix-Tuning的基本功能。使用PyTorch的 `nn.Parameter` 定义前缀嵌入，使其在训练过程中可以被优化。

在代码 `prefix = self.prefix_embedding.unsqueeze(0).expand(input_ids.size(0), -1, -1)` 中，这一行执行了几个操作，以准备好前缀嵌入（prefix embeddings），从而能够与模型的输入嵌入（input embeddings）进行拼接。

1. **`self.prefix_embedding`**：这是一个张量，代表了可训练的前缀嵌入，其形状为 `(prefix_length, hidden_size)`。这个张量是初始化后存储在模型中的一个参数。
2. **`unsqueeze(0)`**：这个操作在张量的第0个位置（也就是最前面）增加一个新的维度，通常称为批量维度（batch dimension）。如果 `self.prefix_embedding` 原本的形状是 `(prefix_length, hidden_size)`，经过 `unsqueeze(0)` 处理后，它的形状变为 `(1, prefix_length, hidden_size)`。这样做是为了匹配模型输入的维度，因为模型输入通常包含一个批量维度，用来表示同时处理的输入样本数量。
3. **`expand(input_ids.size(0), -1, -1)`**：
   - `input_ids.size(0)` 获取输入张量 `input_ids` 的批量维度大小。如果 `input_ids` 的形状为 `(batch_size, sequence_length)`，那么 `input_ids.size(0)` 就是 `batch_size`。
   - `expand` 操作用于在指定的维度上复制张量，而不会实际复制内存中的数据。在这里，`-1` 表示“保持当前维度的大小”，而 `input_ids.size(0)` 替换了之前 `unsqueeze(0)` 的 `1`，有效地将前缀嵌入在批量维度上进行广播。
   - 因此，经过这个操作后，`prefix` 的形状变为 `(batch_size, prefix_length, hidden_size)`。这样，每个批量的输入都有自己的一组前缀嵌入，这些嵌入最初在所有批量中是相同的，但在训练过程中会有所不同。

在代码 `input_embeds = self.model.transformer.wte(input_ids)` 中，这一行的目的是通过将输入标识符（`input_ids`）转换为嵌入向量（embedding vectors），为Transformer模型的后续处理准备数据。

1. **`self.model.transformer.wte`**：
   - `self.model` 指的是一个预训练的语言模型，例如 GPT-2
   - `self.model.transformer` 代表这个模型的 Transformer 结构
   - `wte` 是 `word token embeddings` 的缩写。它是 Transformer 模型的一部分，专门用于将离散的输入标识符（即单词或子词的索引）转换为连续的向量表示。这样，模型能够处理这些输入并理解它们之间的语义关系。
2. **`input_ids`**：
   - `input_ids` 是一个包含输入标识符（通常是单词或子词的索引）的张量。每个输入标识符都是一个整数，表示特定词汇在模型词汇表中的位置。这个张量的形状通常是 `(batch_size, sequence_length)`，其中 `batch_size` 是同时处理的输入样本数，`sequence_length` 是每个输入样本的标识符序列长度。
3. **嵌入操作**：
   - `self.model.transformer.wte(input_ids)` 通过查找嵌入矩阵，将输入标识符转换为对应的嵌入向量。嵌入矩阵的每一行代表一个特定单词或子词的向量表示，这些向量是在预训练过程中学习到的，能够捕捉到单词或子词的语义特征。
   - 例如，如果模型词汇表的大小为 50,000，嵌入维度（hidden size）为 768，那么嵌入矩阵的形状就是 `(50000, 768)`。每个输入标识符通过嵌入操作后都会被转换为一个形状为 `(768,)` 的向量。
4. **结果**：
   - `input_embeds` 是一个形状为 `(batch_size, sequence_length, hidden_size)` 的张量，其中每个输入标识符都被转换为相应的嵌入向量。这些嵌入向量作为 Transformer 模型的输入，供其后续层（如自注意力机制和前馈神经网络）进行处理。

# Prefix-Tuning 优化

[THUMT Research Blog](https://thumtblog.github.io/2022/04/05/robust-prefix-tuning/) 发现前缀微调的鲁棒性存在不足。当输入数据被操控时，前缀容易被欺骗。例如，输入的稍微改变可能会使模型做出错误的分类。在此背景下，防御对抗攻击对于保持前缀微调的参数高效性是非常有必要的。目前在自然语言处理（NLP）领域，大致有四种防御方法：

1. **模型功能改进**（Li & Sethy, 2019; Jones et al., 2020）：对模型功能进行改进以提高其鲁棒性。
2. **鲁棒性认证**（Jia et al., 2019; Shi et al., 2020）：提供对模型鲁棒性的理论保障。
3. **对抗者检测**（Pruthi et al., 2019; Zhou et al., 2019）：检测并防御对抗性输入。
4. **对抗性训练**（Miyato et al., 2017; Miyato et al., 2019; Zhu et al., 2020; Yi et al., 2021）：通过对抗性样本训练模型提高其对抗性。

在对抗性攻击的背景下，前缀微调（Prefix-Tuning）面临的一个主要挑战是其鲁棒性不足。许多现有的防御方法，比如模型功能改进、鲁棒性认证、对抗检测和对抗性训练，通常需要对模型架构和参数进行修改，或是增加对抗者检测器的额外维护。这些方法需要额外的模型更新和存储，削弱了前缀微调的模块化特性。尽管对抗性训练在某种程度上可以增强鲁棒性，但它因训练时间漫长而不太适合前缀微调场景。

## 设计鲁棒前缀

为了提升前缀微调的鲁棒性，同时保持其效率和模块化，我们提出了一种新的方法。在这种方法中，我们在推理过程中微调一个附加前缀 \( $P_{\Psi'}$ \)，同时保持原始前缀 \( $P_{\theta}$ \) 固定不变。通过优化附加前缀 \( $P_{\Psi'}$ \)，我们希望在遭受攻击时能够纠正错误的激活状态，引导模型做出正确的预测。

具体来说，我们假设在第 \( $j$ \) 层的所有正确激活状态都位于一个流形 \( $M(j)$ \) 上。通过最小化错误激活的正交分量来调整附加前缀 \( $P_{\Psi'}$ \)。以下是这个过程的实现细节：

1. **构建正则流形**：通过主成分分析（PCA）来表征正确分类输入的层激活。
2. **更新附加前缀**：在每一层 \( $j$ \)，通过线性投影矩阵 \( $Q(j)$ \)，计算在输出位置 \( $o$ \) 处的激活状态 \( $h_o^{(j)}$ \)，并最小化损失函数
   $$
   \|h_o^{(j)} - h_o^{(j)} Q(j)\|^2
   $$
   从而在推理过程中更新 \( $P_{\Psi'}$ \)。

## 简易示例代码

```python
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# 简单神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)  # 二分类输出

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型和一些示例数据
model = SimpleModel()
correct_classified_inputs = torch.randn(100, 10)  # 假设我们有100个正确分类的输入

# 1. 收集激活状态
model.eval()  # 设置模型为评估模式
activations = []  # 用于存储所有层的激活状态
with torch.no_grad():
    for x in correct_classified_inputs:
        x = x.unsqueeze(0)  # 扩展维度，使其成为批量输入
        activation = model.fc1(x)
        activations.append(activation.numpy())

# 2. 主成分分析（PCA）
activations = torch.tensor(activations).squeeze(1).numpy()  # 转换为numpy格式以进行PCA
pca = PCA(n_components=10)  # 我们希望将激活状态降到10维
pca.fit(activations)

# 3. 线性投影矩阵 Q(j)
Q_j = pca.components_  # 这是从PCA得到的线性投影矩阵

# 使用 Q_j 将激活状态投影到流形上
def project_to_manifold(activation, Q_j):
    return torch.tensor(activation).matmul(torch.tensor(Q_j.T))

# 示例使用
new_activation = torch.randn(1, 50)  # 假设新的激活状态
projected_activation = project_to_manifold(new_activation, Q_j)
print(projected_activation)
```

