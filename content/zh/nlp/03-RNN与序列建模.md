---
title: "自然语言处理（三）：RNN与序列建模"
date: 2025-10-11 09:00:00
tags:
  - NLP
  - RNN
  - 深度学习
  - LSTM
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "RNN、LSTM、GRU 如何通过记忆处理序列。从第一性原理推导梯度消失，用 PyTorch 实现字符级文本生成器和 Seq2Seq 翻译器。"
disableNunjucks: true
series_order: 3
translationKey: "nlp-3"
polished_by_qwen_max: true
---
打开 Google 翻译、用滑动输入法打字或对着手机录一段备忘——这些日常操作背后，都离不开一个核心任务：按顺序处理一串 token，再生成另一串符号。前馈神经网络会把每个输入当作独立的个体来处理，但语言本质上是上下文关联的。例如，在句子“猫坐在垫子上”中，理解“垫子”的意思需要结合前面的所有词语。循环神经网络（RNN）通过维护一个隐藏状态来解决这个问题。每读入一个 token，隐藏状态就会更新一次，动态地汇总过去的信息，可视为网络的‘记忆’。

这篇文章将带你从头梳理循环神经网络家族的发展脉络。从最基础的 RNN 开始，分析它为什么无法记住超过十几个 token 的信息，接着看 LSTM 和 GRU 如何通过引入门控机制突破这一限制，最后用 PyTorch 实现一个英法翻译模型。读完本文，你会清晰理解 RNN 如何逐步演进为注意力机制（Attention）和 Transformer。


<!-- wanx-hero -->
![自然语言处理（三）：RNN与序列建模 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/rnn-sequence-modeling/illustration_1.png)
## 你将学到什么
![自然语言处理（三）：RNN与序列建模 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/rnn-sequence-modeling/illustration_2.png)

- RNN 是如何通过循环连接和参数共享来维持记忆的
- 从第一性原理推导梯度消失和梯度爆炸的原因
- LSTM 的三个门控机制（遗忘门、输入门、输出门）以及细胞状态如何解决长程依赖问题
- GRU 作为比 LSTM 更轻量的选择，以及在不同场景下如何取舍
- 双向 RNN 和堆叠 RNN 如何增强每个 token 的表示能力
- Seq2Seq 编码器-解码器架构的局限性，以及为什么注意力机制的引入是不可避免的
- 使用 PyTorch 实现文本生成和翻译的实际代码示例

**前置知识**：本系列第 1-2 部分（分词与词嵌入），以及基础 PyTorch 知识（张量操作、`nn.Module` 模块、训练循环）。
## 一、核心思想：循环与参数共享

![NLP (3): RNN 和序列建模 —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rnn-sequence-modeling/illustration_2.png)

![基础 RNN 在五个时间步展开，展示循环权重共享](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/rnn-sequence-modeling/fig1_unrolled_rnn.png)

在每个时间步 $t$， RNN 会接收当前输入 $x_t$ 和上一个隐藏状态 $h_{t-1}$，然后计算出新的隐藏状态和输出：

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b), \qquad y_t = W_y h_t + b_y.$$

从图中可以看出，每一步的箭头都是一样的，这表明 **矩阵 $W_h$、$W_x$、$W_y$ 在所有时间步上是共享的**。正是这一巧妙的设计，让 RNN 同时具备了三个重要特性：

- **跨位置泛化能力**：在第 3 个位置学到的模式可以直接应用到第 30 个位置，因为它们共享同一组权重。
- **参数数量恒定**：模型的大小与序列长度无关，处理 10 个 token 和 1 万个 token 的存储成本完全一致。
- **支持变长输入和输出**：公式中没有任何地方对序列长度 $T$ 做出限制，无论是 5 还是 500 都能轻松应对。

可以想象， RNN 就像是一个逐词阅读的“理解机器”，每读入一个词，它都会更新对句子的整体理解。而第 $t$ 步的隐藏状态，则是对从第一个词到当前词 $x_1, \dots, x_t$ 的固定长度的可学习总结。
## 二、梯度消失问题
![梯度范数随距离衰减；右侧用一个示例句展示长距离依赖](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig2_vanishing_gradient.png)

训练过程中，问题很快浮现。为了计算梯度，我们需要将网络在时间维度上展开，然后通过反向传播算法进行优化，这种方法被称为沿时间反向传播（Backpropagation Through Time, BPTT）。在第 $T$ 步的损失函数相对于第 $t$ 步隐藏状态的梯度，可以表示为一系列雅可比矩阵的连乘积：

$$\frac{\partial h_T}{\partial h_t} \;=\; \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}.$$

每个雅可比矩阵因子大约是 $W_h^{\top}\,\mathrm{diag}(\tanh'(\cdot))$。由于 $\tanh'$ 的值始终小于等于 1，这个因子的谱范数不会超过权重矩阵 $W_h$ 的最大奇异值，记作 $\lambda$。当我们将 $T-t$ 个这样的因子相乘时，结果会呈现出以下形式：

$$\left\| \frac{\partial h_T}{\partial h_t} \right\| \;\lesssim\; \lambda^{\,T-t}.$$

从这里可以看出两种典型情况，左图也清晰地展示了它们：

- 如果 $\lambda < 1$，梯度范数会以**指数速度衰减**。通常经过 10 到 20 步后，梯度数值上几乎归零。此时，优化器完全无法感知第 $t$ 个 token 对第 $T$ 步损失的影响，模型自然无法学习到这种长距离依赖关系。
- 如果 $\lambda > 1$，梯度则会**爆炸**——权重更新变得极其不稳定，训练可能在单步内就发散。

右图给出了一个具体例子：“那只猫，它坐在垫子上、还大声呼噜，**很**开心。”主语“猫”和谓语“很”之间隔了十个词。普通的 RNN 根本无法将梯度传递这么远的距离，因此永远学不会这种主谓一致的关系。

**实际中的解决方案是什么？** 梯度裁剪可以有效应对梯度爆炸问题——通过将全局梯度范数限制在某个阈值（比如 5.0）以内，避免更新失控。但裁剪对梯度消失无能为力。真正解决问题的关键在于重新设计循环结构，为梯度提供一条**不会衰减**的传播路径。 LSTM 和 GRU 正是通过引入这样的路径，成功缓解了这一问题。
## 三、长短期记忆网络（LSTM）
![LSTM 单元结构：遗忘门、输入门、输出门，以及加法式细胞状态高速公路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig3_lstm_gates.png)

LSTM （Hochreiter & Schmidhuber, 1997）引入了一种带门控机制的单元，替换了传统循环神经网络中的简单循环单元。这种新单元不仅维护隐藏状态 $h_t$，还额外保留了一条显式的长期记忆 $C_t$，用于存储跨越长时间步的信息。

### 三个关键门控机制

假设 $[h_{t-1}, x_t]$ 表示上一时刻的隐藏状态和当前输入的拼接结果，所有门控机制都基于这一共享输入进行操作。

**遗忘门**——决定哪些长期记忆需要被丢弃：

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f).$$

**输入门与候选值**——决定哪些新信息需要被写入：

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \qquad
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C).
$$

**细胞状态更新**——将旧记忆与新信息结合起来：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t.$$

**输出门**——决定暴露哪部分细胞状态作为新的隐藏状态：

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \qquad
h_t = o_t \odot \tanh(C_t).
$$

这里，$\sigma$ 是 sigmoid 函数（一种软性的 0–1 开关），$\odot$ 表示逐元素乘法。每个门的作用可以看作是一个可学习的位置相关决策：*忘掉这部分，写入那部分，露出这些内容*。

### 为什么这解决了梯度消失问题？

图中最核心的部分是顶端的细胞状态线。它的更新采用的是**加法**形式：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t.$$

对其求导后得到 $\partial C_t / \partial C_{t-1} = f_t$，这是一个逐元素的标量，取值范围在 $[0,1]$。当遗忘门的值接近 1 时，连乘项 $\prod f_k$ 也会保持接近 1——这意味着梯度可以通过细胞状态几乎无损地传递，即使跨越几百个时间步。相比之下，传统的 RNN 更新依赖于 $W_h$ 的**乘法**形式，容易导致梯度逐渐趋近于零。 LSTM 的关键改进在于用一个可学习且随时间变化的 $f_t$ 替代了全局共享的 $W_h$，仅此一点改动就让它能够有效建模长距离上下文。

### 传送带的比喻

可以把细胞状态想象成一条贯穿整个序列的传送带。遗忘门像是一个工人，负责从传送带上移除不再需要的物品；输入门则是另一个工人，负责将新物品放到传送带上；输出门则像是一扇窗口，决定了外部世界（网络的其他部分）当前能看到哪些内容。第 3 步放上去的物品，可以一路平稳地传送到第 300 步，而不会受到干扰。
## 四、门控循环单元（GRU）
![GRU 单元：重置门 + 更新门，结构比 LSTM 更简洁](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig4_gru_cell.png)

GRU （Cho 等人， 2014）继承了门控机制的核心思想，但对其设计进行了简化。它将遗忘门和输入门合并为一个**更新门**，取消了独立的细胞状态，直接在隐藏状态 $h_t$ 上进行操作：
$$
z_t = \sigma(W_z [h_{t-1}, x_t]), \qquad
r_t = \sigma(W_r [h_{t-1}, x_t]),
$$$$
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]), \qquad
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
$$
其中，**重置门** $r_t$ 控制过去信息对候选隐藏状态的影响程度；**更新门** $z_t$ 则在旧状态和新候选之间进行线性插值。当 $z_t \approx 0$ 时， GRU 会直接将上一时刻的状态 $h_{t-1}$ 复制到当前时刻，这种机制与 LSTM 中通过细胞状态保留梯度的“高速公路”效果完全一致。

### LSTM 和 GRU 的对比

| 对比维度       | LSTM                          | GRU                          |
|----------------|-------------------------------|------------------------------|
| 门的数量       | 3 个（遗忘门、输入门、输出门） | 2 个（重置门、更新门）        |
| 是否有独立细胞状态 | 有（$C_t$）                   | 无（仅使用 $h_t$）           |
| 参数量         | 约为基础 RNN 的 $4\times$     | 约为基础 RNN 的 $3\times$    |
| 长序列表现     | 在许多基准测试中略占优势       | 表现相当                     |
| 训练速度       | 较慢                          | 较快                         |

**经验法则**：优先尝试 GRU。它的训练速度更快，超参数更少，且在大多数任务上与 LSTM 的精度差距微乎其微。如果处理的是特别长的序列，或者任务明确需要更大的模型容量（例如某些语音处理任务），可以考虑切换到 LSTM。
## 五、双向 RNN
![双向 RNN：每个位置正向与反向状态拼接](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig5_bidirectional_rnn.png)

在许多任务中，未来的信息和过去的信息同样关键。举个例子：“他说这道菜**不**好吃”——如果模型只从左往右读，没看到“不”这个字，就可能错误地把“好吃”判定为正面情感。

双向 RNN （Schuster & Paliwal, 1997）通过同时运行两个独立的循环网络，将它们的状态结合起来：
$$
\overrightarrow{h}_t = \mathrm{RNN}_\text{fwd}(x_t, \overrightarrow{h}_{t-1}), \qquad
\overleftarrow{h}_t = \mathrm{RNN}_\text{bwd}(x_t, \overleftarrow{h}_{t+1}),
$$$$
h_t = \big[\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\big].
$$
这样一来，每个位置的表示都能同时捕捉到前后的上下文信息。

**适合的应用场景**：命名实体识别、词性标注、机器翻译中的编码器等任务。只要输入数据是完整的，就可以充分发挥双向 RNN 的优势。

**不适合的应用场景**：流式处理或自回归生成任务。因为反向循环依赖未来的 token，而在逐个生成的过程中，这些 token 尚未出现。
## 六、堆叠 RNN
网络深度确实能带来好处：通过堆叠多层 RNN，每一层都可以基于前一层每一步的输出进一步提炼特征：
$$
h_t^{(1)} = \mathrm{RNN}^{(1)}(x_t,\, h_{t-1}^{(1)}), \qquad
h_t^{(2)} = \mathrm{RNN}^{(2)}(h_t^{(1)},\, h_{t-1}^{(2)}).
$$
在实际应用中，较低层通常会学习到一些局部模式，例如字符 n-gram、词边界以及形态学特征；而较高层则更倾向于捕捉句法结构和长距离的语义信息。对于大多数 NLP 任务来说， 2 到 4 层的堆叠已经能够达到很好的效果；如果继续增加层数，残差连接就显得尤为重要，否则优化过程可能会变得不稳定。
## 七、序列到序列模型
![Seq2Seq 编码器-解码器架构中的固定大小上下文向量瓶颈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig6_seq2seq.png)

Seq2Seq 模型（Sutskever et al., 2014）是一种将输入序列转换为长度*不同*的输出序列的方法，最经典的例子就是机器翻译。它通过两个 RNN 来实现：

- **编码器**负责读取整个输入序列，并将其压缩成一个单一的上下文向量 $c = h_T^{\text{enc}}$。
- **解码器**则根据这个上下文向量 $c$ 和已经生成的部分输出，逐步生成目标序列的每个 token：
$$
s_t = \mathrm{RNN}_\text{dec}(y_{t-1}, s_{t-1}), \qquad
P(y_t \mid y_{<t}, x) = \mathrm{softmax}(W_o s_t).
$$

**瓶颈问题。** 整个输入序列——可能包含多达 50 个词的信息——都必须被压缩进一个固定大小的向量 $c$ 中。对于短句子来说，这还能应付；但对于长句子，编码器就会显得力不从心。 Sutskever 等人在原论文中发现，当输入序列长度超过 30 个 token 时， BLEU 分数会显著下降。有趣的是，将源语言句子倒序输入反而能提升效果，这一现象实际上暗示了瓶颈问题是性能下降的主要原因。

正是这个瓶颈问题直接推动了**注意力机制**的诞生，也就是第 4 部分的主题：与其让解码器依赖于单一的上下文向量 $c$，不如在每一步解码时，允许它动态地回顾编码器生成的*所有*隐藏状态。
## 八、 PyTorch 实现：字符级文本生成器
接下来，我们训练一个小型 LSTM 模型，让它能够逐字符生成文本。

### 数据准备

```python
import torch
import torch.nn as nn
import numpy as np

text = """Deep learning is a subset of machine learning that uses neural
networks with many layers. These networks can learn hierarchical
representations of data, making them powerful for tasks like image
recognition, natural language processing, and speech recognition."""

chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"词表大小: {vocab_size} 个字符")
```

### 模型设计

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
```

### 训练过程

```python
def train(model, text_data, epochs=100, seq_length=50, batch_size=16, lr=0.002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = [char_to_idx[ch] for ch in text_data]

    for epoch in range(epochs):
        model.train()
        hidden = model.init_hidden(batch_size, device)
        total_loss = 0
        n_batches = len(data) // (seq_length * batch_size)

        for _ in range(n_batches):
            starts = np.random.randint(0, len(data) - seq_length - 1, batch_size)
            inputs = torch.LongTensor([data[s:s+seq_length] for s in starts]).to(device)
            targets = torch.LongTensor([data[s+1:s+seq_length+1] for s in starts]).to(device)

            hidden = tuple(h.detach() for h in hidden)  # 截断 BPTT，避免梯度回传到整个语料
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 防止梯度爆炸
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"第 {epoch+1}/{epochs} 轮训练完成，当前损失: {total_loss/n_batches:.4f}")
```

在训练过程中，我们用到了两个 RNN 的常见技巧：**在每个 batch 之间分离隐藏状态**，以截断 BPTT （否则梯度会试图回传到整个训练数据）；以及**梯度裁剪**，防止梯度爆炸问题。

### 文本生成——温度控制创造性

```python
def generate(model, start_str, length=200, temperature=0.8):
    device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device)
    input_seq = [char_to_idx[ch] for ch in start_str]
    generated = start_str

    with torch.no_grad():
        for idx in input_seq[:-1]:  # 先通过初始字符串“预热”隐藏状态
            x = torch.LongTensor([[idx]]).to(device)
            _, hidden = model(x, hidden)

        x = torch.LongTensor([[input_seq[-1]]]).to(device)
        for _ in range(length):
            output, hidden = model(x, hidden)
            probs = torch.softmax(output.squeeze() / temperature, dim=0).cpu().numpy()
            char_idx = np.random.choice(len(probs), p=probs)
            generated += idx_to_char[char_idx]
            x = torch.LongTensor([[char_idx]]).to(device)

    return generated

model = CharRNN(vocab_size, hidden_size=128, num_layers=2)
train(model, text, epochs=100)
print(generate(model, "Deep learning", length=200))
```

**温度参数**用于调整 softmax 前的 logits 缩放比例：$P(w) = \mathrm{softmax}(\text{logits}/T)$。较低的温度（如 0.5 左右）会让概率分布更加尖锐，生成的文本趋于保守且重复性较高；而较高的温度（如 1.5 或更高）则会让分布更加平滑，生成的文本更具创造性但可能缺乏逻辑。通常情况下，$T=0.8$ 是一个不错的折中选择。
## 九、 PyTorch 实现：一个极简的 Seq2Seq 翻译器
我们实现了一个最基础的英法翻译器，目的是在引入注意力机制（第四部分）之前，先帮助大家理解编码器-解码器架构的基本数据流。

### 数据与词表

```python
import torch
import torch.nn as nn
import random

pairs = [
    ("hello", "bonjour"), ("good morning", "bon matin"),
    ("thank you", "merci"), ("goodbye", "au revoir"),
    ("how are you", "comment allez vous"),
    ("i love you", "je t aime"), ("welcome", "bienvenue"),
]

SOS, EOS = 0, 1

class Vocab:
    def __init__(self):
        self.word2idx = {"<SOS>": SOS, "<EOS>": EOS}
        self.idx2word = {SOS: "<SOS>", EOS: "<EOS>"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

src_vocab, tgt_vocab = Vocab(), Vocab()
for en, fr in pairs:
    src_vocab.add_sentence(en)
    tgt_vocab.add_sentence(fr)
```

### 编码器与解码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.lstm(self.embedding(x))

class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(self.embedding(x), hidden)
        return self.fc(out.squeeze(1)), hidden
```

### 训练（使用教师强制）

```python
def train_seq2seq(encoder, decoder, pairs, epochs=500, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = encoder.to(device), decoder.to(device)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(pairs)
        for src_sent, tgt_sent in pairs:
            src_ids = [src_vocab.word2idx[w] for w in src_sent.split()] + [EOS]
            tgt_ids = [tgt_vocab.word2idx[w] for w in tgt_sent.split()] + [EOS]

            src_t = torch.LongTensor([src_ids]).to(device)
            tgt_t = torch.LongTensor(tgt_ids).to(device)

            enc_opt.zero_grad(); dec_opt.zero_grad()

            _, hidden = encoder(src_t)                     # 上下文向量
            dec_input = torch.LongTensor([[SOS]]).to(device)
            loss = 0

            for i in range(len(tgt_ids)):
                output, hidden = decoder(dec_input, hidden)
                loss += criterion(output, tgt_t[i:i+1])
                dec_input = tgt_t[i:i+1].unsqueeze(0)      # 教师强制
            loss.backward()
            enc_opt.step(); dec_opt.step()
            total_loss += loss.item() / len(tgt_ids)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs):.4f}")
```

### 推理

```python
def translate(encoder, decoder, sentence):
    device = next(encoder.parameters()).device
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        src_ids = [src_vocab.word2idx.get(w, 0) for w in sentence.split()] + [EOS]
        _, hidden = encoder(torch.LongTensor([src_ids]).to(device))
        dec_input = torch.LongTensor([[SOS]]).to(device)
        words = []
        for _ in range(20):
            output, hidden = decoder(dec_input, hidden)
            token = output.argmax(dim=-1).item()
            if token == EOS:
                break
            words.append(tgt_vocab.idx2word[token])
            dec_input = torch.LongTensor([[token]]).to(device)
    return ' '.join(words)

hidden_size = 256
encoder = Encoder(src_vocab.n_words, hidden_size)
decoder = Decoder(hidden_size, tgt_vocab.n_words)
train_seq2seq(encoder, decoder, pairs)
for s in ["hello", "thank you", "good morning"]:
    print(f"{s} -> {translate(encoder, decoder, s)}")
```

**注意事项。** 这个实现非常简单，仅仅是为了演示编码器-解码器的数据流。它只能过拟合一个小短语表，使用的是贪婪解码，没有加入注意力机制或束搜索。如果要构建一个实用的翻译系统，还需要添加注意力机制（第四部分）、束搜索、子词（BPE）分词，以及用验证集实现早停功能。
## 十、它们的实际差距有多大

![损失曲线与不同序列长度下的精度对比，比较 RNN、LSTM、GRU](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN与序列建模/fig7_loss_curves.png)

上面两张图生动地展示了它们的差异。在处理长依赖任务时，普通的 RNN 训练到一定程度后损失值就会卡在一个较高的水平，再也降不下去；而 LSTM 和 GRU 则能持续优化——这说明“梯度高速公路”确实在发挥作用。随着序列长度增加， RNN 的预测精度迅速崩塌，而 LSTM 和 GRU 的性能下降则要平缓得多。此外，在大多数场景下， LSTM 和 GRU 的表现差距并不大，这也是为什么很多人会优先选择 GRU 作为默认模型的原因之一。
## 注意力机制预告

编码器会将所有信息压缩到一个向量 $c$ 中。然而，当处理较长的句子时，这种单一向量的形式往往会成为瓶颈，导致信息丢失。**注意力机制**的引入解决了这个问题：它允许解码器在每一步都“查阅”编码器的所有隐藏状态，并通过学习得到的权重来决定每个状态的重要性：
$$
\alpha_{tj} = \frac{\exp(\mathrm{score}(s_t, h_j))}{\sum_k \exp(\mathrm{score}(s_t, h_k))}, \qquad
c_t = \sum_j \alpha_{tj}\, h_j.
$$

这样一来，上下文向量就变成了编码器状态的*动态加权和*，能够根据时间步的变化灵活调整。这一突破性思想为从 RNN 到 Transformer 的演进铺平了道路，我们将在第 4 部分深入探讨。
## 常见问题
### 为什么基础 RNN 使用 tanh 而不是 ReLU？

tanh 的输出范围是 $[-1, 1]$，这使得隐藏状态在时间步之间能够保持在一个有限的范围内。而 ReLU 的正向输出没有上限，反复递归计算时很容易导致数值爆炸。 LSTM 则巧妙地结合了 sigmoid 和 tanh： sigmoid 用于门控（相当于一个软性的 0–1 开关），而 tanh 用于生成候选值（零中心分布，既能增加也能减少细胞状态的值）。

### 什么是 teacher forcing？它有哪些弊端？

在训练过程中，我们会将真实的上一个 token （即“正确答案”）直接作为解码器的输入，而不是使用模型自己预测的结果。这种方法在训练初期有助于稳定学习过程，但会导致训练和推理之间的不一致——推理阶段解码器必须依赖自身生成的（可能带噪声的）输出，而这些输出在训练中从未出现过。为了解决这个问题，常用的方法是 **scheduled sampling**：随着训练的推进，逐步提高使用模型自身预测结果的概率。

### 温度对生成过程有什么影响？

温度的作用是对 logits 进行缩放，然后再输入 softmax 函数：$P(w) = \mathrm{softmax}(\text{logits}/T)$。当温度较低（例如 0.5）时，概率分布会更加尖锐，生成结果偏向保守；而当温度较高（例如 1.5）时，分布会变得更加平滑，生成结果更具创造性，但同时也更容易出错。贪婪解码可以看作是温度趋近于零（$T \to 0$）时的极限情况。

### Transformer 出现后， RNN 是否还有存在的意义？

虽然 Transformer 在离线 NLP 基准测试中已经占据主导地位，但在一些特定场景下， RNN 仍然有其独特的优势：(i) 参数量有限的场景；(ii) 需要真正的流式或在线推理（无需重新计算整个前缀的注意力）；(iii) 每一步的内存占用固定，与序列长度无关。此外， RNN 在时间序列预测和端侧语音模型中依然广泛应用。从机制上看，注意力可以被理解为“如果用一个可学习的历史状态回顾机制替代 LSTM 的遗忘门，会发生什么？”因此，深入理解 RNN 是掌握 Transformer 的最佳切入点。
## 核心要点
- **RNN 通过循环连接和参数共享来处理序列数据**，每个时间步都使用相同的权重参数。
- **普通的 RNN 在处理长序列时表现不佳**，因为雅可比矩阵的连乘积 $\prod \partial h_{k+1} / \partial h_k$ 会呈指数级衰减或爆炸。
- **LSTM 引入了一条细胞状态的“高速公路”**，通过遗忘门、输入门和输出门进行控制，为梯度提供了一条不会消失的传播路径。
- **GRU 对 LSTM 进行了简化**，只保留两个门和一个状态，通常可以用比 LSTM 少约 25% 的参数达到相近的效果。
- **双向结构和堆叠式变体**分别扩展了单个位置的上下文范围，并增加了网络的深度。
- **Seq2Seq 编码器-解码器架构**可以实现序列到序列的映射，但受限于单一的上下文向量 $c$——这一瓶颈正是第 4 部分引入注意力机制的主要动机。