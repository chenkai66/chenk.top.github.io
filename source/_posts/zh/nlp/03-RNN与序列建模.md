---
title: "自然语言处理（三）：RNN与序列建模"
date: 2025-10-11 09:00:00
tags:
  - NLP
  - RNN
  - 深度学习
  - LSTM
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 3
  total: 12
lang: zh-CN
mathjax: true
description: "RNN、LSTM、GRU 如何通过记忆处理序列。从第一性原理推导梯度消失，用 PyTorch 实现字符级文本生成器和 Seq2Seq 翻译器。"
disableNunjucks: true
series_order: 3
---

打开翻译软件、用滑动键盘打字、对手机口述备忘——每一项功能背后，都需要一个模型按顺序消费一串 token，再产出另一串。前馈网络把每个输入当成孤立的样本，但语言天生就是**有顺序**的：要理解"猫坐在垫子上"里"垫子"的含义，你必须知道前面所有词的语境。循环神经网络（RNN）的解决方式是维护一个**隐藏状态**，每读一个 token 就更新一次。这个隐藏状态，就是网络对过去内容的"持续摘要"，也就是它的记忆。

本文从零开始把循环网络这一族架构串起来。先讲最朴素的 RNN，**推导**它为什么记不住超过十几个 token，再看 LSTM 和 GRU 用门控机制如何解围，最后用 PyTorch 跑通一个英法翻译器。读完后，你会理解从 RNN 走向注意力机制和 Transformer 的真正动因。

## 你将学到什么

- RNN 如何通过循环连接和参数共享维持记忆
- 从第一性原理推导梯度消失与梯度爆炸
- LSTM 的三个门（遗忘门、输入门、输出门）和细胞状态高速公路如何解决长距离依赖
- GRU 作为 LSTM 的精简版本，何时该选哪个
- 双向 RNN 和堆叠 RNN 如何丰富每个位置的表征
- Seq2Seq 编码器-解码器架构、它的瓶颈，以及为什么注意力是必然
- 文本生成与翻译的 PyTorch 实现

**前置知识**：本系列第 1-2 部分（分词与词嵌入），以及基础 PyTorch（张量、`nn.Module`、训练循环）。

---

## 一、核心思想：循环与参数共享

![基础 RNN 在 5 个时间步上展开，循环权重共享](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig1_unrolled_rnn.png)

在每个时间步 $t$，RNN 接收输入 $x_t$ 和上一时刻的隐藏状态 $h_{t-1}$，产出新的隐藏状态和输出：

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b), \qquad y_t = W_y h_t + b_y.
$$

整张图最关键的细节，就是同样的箭头在每个时间步重复出现——**矩阵 $W_h$、$W_x$、$W_y$ 在所有位置共用同一份**。这一个设计决策一次带来三个好处：

- **位置间泛化**：在第 3 个位置学到的模式，到第 30 个位置也能直接用，因为同一组权重见过两边。
- **参数量恒定**：模型大小与序列长度无关，10 个 token 和 1 万个 token 占用的*存储*完全一样。
- **支持变长**：方程里没有任何地方在乎 $T$ 是 5 还是 500。

形象地讲，可以把网络想成一次读一个词，每读一个词就更新它对整句话的"理解"。第 $t$ 步的隐藏状态，就是 $x_1, \dots, x_t$ 这一段历史的固定大小的可学习摘要。

---

## 二、梯度消失问题

![梯度范数随距离衰减；右侧用一个示例句展示长距离依赖](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig2_vanishing_gradient.png)

麻烦从训练开始。要算梯度，我们把网络沿时间维度*展开*再反向传播，这一过程叫"沿时间反向传播"（BPTT）。第 $T$ 步的损失对第 $t$ 步隐藏状态的梯度，是一长串雅可比矩阵的连乘：

$$
\frac{\partial h_T}{\partial h_t} \;=\; \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}.
$$

每个雅可比因子大致是 $W_h^{\top}\,\mathrm{diag}(\tanh'(\cdot))$。因为 $\tanh' \le 1$，这个因子的谱范数被 $W_h$ 的最大奇异值（记作 $\lambda$）封顶。把 $T-t$ 个这样的因子乘起来，就得到：

$$
\left\| \frac{\partial h_T}{\partial h_t} \right\| \;\lesssim\; \lambda^{\,T-t}.
$$

立刻冒出两种情况，左图也直接画出来了：

- 若 $\lambda < 1$，梯度范数**指数级衰减**。大约 10–20 步以外就在数值上归零，优化器根本看不出第 $t$ 个 token 对第 $T$ 步的损失有任何贡献，模型自然学不到这种依赖。
- 若 $\lambda > 1$，梯度则**指数级爆炸**——一次更新就能把权重炸飞，训练当场发散。

右图把这件事说具体了。"那只猫，它坐在垫子上、还大声呼噜，**很**开心"——主语"猫"和谓语"很"之间隔了十个词。基础 RNN 没办法把梯度传那么远，所以学不会这种主谓对应。

**实践中怎么办？** *梯度裁剪*（把全局梯度范数截在 5.0 之类的阈值）能解决爆炸，但对消失束手无策。真正的修复需要重新设计循环结构，让梯度有一条**不会缩水**的传播路径。LSTM 和 GRU 引入的，正是这条路径。

---

## 三、长短期记忆网络（LSTM）

![LSTM 单元结构：遗忘门、输入门、输出门，以及加法式细胞状态高速公路](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig3_lstm_gates.png)

LSTM（Hochreiter & Schmidhuber, 1997）把简单的循环单元换成一个**带门控的单元**，并显式维护一条长期记忆 $C_t$，与隐藏状态 $h_t$ 并行运转。

### 三个门

记 $[h_{t-1}, x_t]$ 为上一隐藏状态与当前输入的拼接，所有门共享这同一份输入。

**遗忘门**——决定从长期记忆里丢掉什么：

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f).$$

**输入门**与**候选值**——决定要写入什么新信息：

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \qquad
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C).
$$

**细胞状态更新**——把旧记忆与新内容结合：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t.$$

**输出门**——决定从细胞中提取哪部分作为新的隐藏状态：

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \qquad
h_t = o_t \odot \tanh(C_t).
$$

其中 $\sigma$ 是 sigmoid（一个 0–1 的软开关），$\odot$ 是逐元素乘法。每个门都是一次"看情况"的可学习决策：*这条丢、那条写、这块露出来*。

### 为什么这就解决了梯度消失

整张图最关键的就是顶端那条细胞状态线。它的更新是**加法**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t.$$

求导得到 $\partial C_t / \partial C_{t-1} = f_t$，是一个逐元素的标量、取值在 $[0,1]$ 之间。只要遗忘门保持在 1 附近，连乘 $\prod f_k$ 也就保持在 1 附近——梯度可以沿着细胞状态高速公路几乎无损地穿过去，哪怕跨越几百步。这跟基础 RNN 经过 $W_h$ 的*乘法*更新形成鲜明对比，后者一路相乘最后归零。LSTM 把"全局共享的 $W_h$"换成了"可学习、随时间变化的 $f_t$"，仅凭这一处改动，就让模型能建模长上下文。

### 传送带类比

把细胞状态想象成一条贯穿整个序列的传送带。遗忘门是个工人，负责*取下*不再需要的物品；输入门是另一个工人，负责*放上*新物品；输出门则是一扇窗，决定外部世界（网络的其他部分）此刻能看到什么。第 3 步放上去的物品，可以一路安稳地走到第 300 步。

---

## 四、门控循环单元（GRU）

![GRU 单元：重置门 + 更新门，结构比 LSTM 更简洁](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig4_gru_cell.png)

GRU（Cho 等人，2014）保留了门控思想但简化了设计。它把遗忘门和输入门合并成一个**更新门**，去掉了独立的细胞状态，直接在 $h_t$ 上工作：

$$
z_t = \sigma(W_z [h_{t-1}, x_t]), \qquad
r_t = \sigma(W_r [h_{t-1}, x_t]),
$$

$$
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]), \qquad
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
$$

**重置门** $r_t$ 控制计算候选值时让多少过去信息透进来。**更新门** $z_t$ 在旧状态和新候选之间做线性插值——当 $z_t \approx 0$ 时，GRU 就把 $h_{t-1}$ 原样复制到下一步，这跟 LSTM 用细胞状态高速公路保住梯度，是同一个套路。

### LSTM vs. GRU

| 对比维度 | LSTM | GRU |
|---------|------|-----|
| 门数量 | 3 个（遗忘、输入、输出） | 2 个（重置、更新） |
| 是否有独立细胞状态 | 有（$C_t$） | 无（只有 $h_t$） |
| 参数量 | 约为基础 RNN 的 $4\times$ | 约为基础 RNN 的 $3\times$（比 LSTM 少 25% 左右） |
| 长序列表现 | 在很多基准上略占优 | 相当 |
| 训练速度 | 较慢 | 较快 |

**经验法则**：先用 GRU。它训练更快、超参数更少，在大多数任务上和 LSTM 的精度差异都在噪声范围内。如果你的序列特别长，或者任务已知能从更大容量中获益（部分语音任务就是如此），再换成 LSTM。

---

## 五、双向 RNN

![双向 RNN：每个位置把正向与反向状态拼接起来](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig5_bidirectional_rnn.png)

很多任务里，未来的信息和过去同样重要。"他说这道菜**不**好吃"——如果不看到"不"，从左往右读的模型会把"好吃"判成正面情感。

双向 RNN（Schuster & Paliwal, 1997）跑两条独立的循环，再把它们的状态拼起来：

$$
\overrightarrow{h}_t = \mathrm{RNN}_\text{fwd}(x_t, \overrightarrow{h}_{t-1}), \qquad
\overleftarrow{h}_t = \mathrm{RNN}_\text{bwd}(x_t, \overleftarrow{h}_{t+1}),
$$

$$
h_t = \big[\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\big].
$$

每个位置的表征同时见到了双向的上下文。

**适用场景**：命名实体识别、词性标注、机器翻译的编码器——只要你能一次性拿到完整输入，都可以用。

**不适用场景**：流式或自回归生成。反向那一遍需要未来的 token，而生成时这些 token 根本还没产生。

---

## 六、堆叠 RNN

加深也有用：堆叠多层 RNN 能让每一层在前一层的逐步输出上继续构建：

$$
h_t^{(1)} = \mathrm{RNN}^{(1)}(x_t,\, h_{t-1}^{(1)}), \qquad
h_t^{(2)} = \mathrm{RNN}^{(2)}(h_t^{(1)},\, h_{t-1}^{(2)}).
$$

经验上，低层学局部模式（字符 n-gram、词边界、形态学），高层捕捉句法和更长距离的语义。大多数 NLP 任务里 2–4 层就够；再深就必须配残差连接，否则优化稳定性会出问题。

---

## 七、序列到序列模型

![Seq2Seq 编码器-解码器，固定大小的上下文向量构成瓶颈](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig6_seq2seq.png)

Seq2Seq 架构（Sutskever 等人，2014）把一个输入序列映射到长度*不同*的输出序列——典型应用就是机器翻译。它由两个 RNN 组成：

- **编码器**读入整段输入，把它压缩成一个上下文向量 $c = h_T^{\text{enc}}$。
- **解码器**逐 token 生成输出，每一步都以 $c$ 和已经产出的 token 为条件：

$$
s_t = \mathrm{RNN}_\text{dec}(y_{t-1}, s_{t-1}), \qquad
P(y_t \mid y_{<t}, x) = \mathrm{softmax}(W_o s_t).
$$

**瓶颈在哪里。** 整段输入——可能多达 50 个词——都得挤过 $c$ 这个固定大小的向量。短句子还好，长句子就装不下了。Sutskever 的原论文里发现，输入超过约 30 个 token 后 BLEU 分数迅速下跌；而*把源句倒过来*居然能改善结果（这本身就暗示了瓶颈才是真正的问题）。

这个瓶颈正是**注意力机制**的直接动因，也就是第 4 部分要讲的内容：与其逼解码器只靠一个 $c$ 过日子，不如让它在每一步都回头看*所有*编码器隐藏状态。

---

## 八、PyTorch 实现：字符级文本生成器

我们训练一个小型 LSTM，让它逐字符地生成文本。

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

### 模型

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

### 训练循环

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

            hidden = tuple(h.detach() for h in hidden)        # 截断 BPTT
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # 防梯度爆炸
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
```

注意两个 RNN 特有的小技巧：**在 batch 之间 detach 隐藏状态**以截断 BPTT（否则梯度会试图流回整段训练语料），以及**梯度裁剪**来抑制爆炸。

### 采样——温度控制创造性

```python
def generate(model, start_str, length=200, temperature=0.8):
    device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device)
    input_seq = [char_to_idx[ch] for ch in start_str]
    generated = start_str

    with torch.no_grad():
        for idx in input_seq[:-1]:                     # 先把隐藏状态"喂热"
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

**温度**在 softmax 之前对 logits 做缩放：$P(w) = \mathrm{softmax}(\text{logits}/T)$。低温（约 0.5）会让分布变尖、更倾向最可能的字符，输出保守且容易重复；高温（1.5 以上）让分布变平，输出更有创造性但也更容易胡言乱语。$T=0.8$ 是一个常见的折中。

---

## 九、PyTorch 实现：最小化的 Seq2Seq 翻译器

下面给出一个最简的英法翻译器，目的是在加入注意力（第 4 部分）之前，先把编码器-解码器的数据流搞清楚。

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

### 训练（含 teacher forcing）

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
                dec_input = tgt_t[i:i+1].unsqueeze(0)      # teacher forcing
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

**注意事项。** 这个最小化实现会过拟合一个极小的短语表，用的是贪婪解码，没有注意力也没有束搜索，仅供把编码器-解码器的数据流走通。真正的翻译系统需要补上：注意力（第 4 部分）、束搜索、子词（BPE）分词，以及用验证集做早停。

---

## 十、它们实际差距有多大

![损失曲线 + 不同序列长度下的精度对比，比较 RNN、LSTM、GRU](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/03-RNN%E4%B8%8E%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1/fig7_loss_curves.png)

上面这两幅图把整件事讲清楚了。在长依赖任务上，基础 RNN 的训练损失早早卡住，而 LSTM 和 GRU 持续下降——梯度高速公路确实在起作用。随着序列变长，基础 RNN 的精度断崖式下跌，LSTM 和 GRU 则平缓退化。LSTM 与 GRU 之间的差距在大多数场景里都很小，这也是为什么 GRU 是个合理的默认选项。

---

## 注意力机制预告

编码器把所有信息压进单一向量 $c$。对长句子来说，这个瓶颈会丢信息。**注意力机制**让解码器在每一步都能查看*所有*编码器隐藏状态，权重由学习得到：

$$
\alpha_{tj} = \frac{\exp(\mathrm{score}(s_t, h_j))}{\sum_k \exp(\mathrm{score}(s_t, h_k))}, \qquad
c_t = \sum_j \alpha_{tj}\, h_j.
$$

上下文向量从此变成了编码器状态的*随时间变化*的加权求和。这正是从 RNN 通往 Transformer 的桥梁，第 4 部分会详细讨论。

---

## 常见问题

### 为什么基础 RNN 用 tanh 而不是 ReLU？

tanh 输出在 $[-1, 1]$，能让隐藏状态在时间维度上保持有界。ReLU 在正方向无上界，循环作用下很容易指数级炸飞。LSTM 的门用 sigmoid（软 0–1 开关），候选值用 tanh（零中心，可以对细胞状态做加也可以做减）。

### 什么是 teacher forcing，它有什么副作用？

训练时我们把*真实*的上一步 token 作为解码器输入，而不是用模型自己的预测。这能让早期训练稳定下来，但带来了训练-推理不匹配——推理时解码器必须吃自己（带噪声的）输出，而它在训练里从没见过这种输入。常用的缓解方法是 **scheduled sampling**：训练过程中逐步提高"使用模型自身预测"的概率。

### 温度对生成的影响？

它在 softmax 前对 logits 做缩放：$P(w) = \mathrm{softmax}(\text{logits}/T)$。低温（0.5）让分布变尖、更保守；高温（1.5）让分布变平、更有创意但也更容易出错。贪婪解码相当于 $T \to 0$ 的极限。

### Transformer 之后 RNN 还有意义吗？

在离线 NLP 基准上 Transformer 已经全面占优，但 RNN 在以下场景仍然有用：(i) 参数预算非常紧；(ii) 真正的流式/在线推理（不需要每来一个 token 就重新对整个前缀做注意力）；(iii) 单步内存恒定、不随序列长度增长。它们在时间序列预测和端侧语音模型里依然常见。而且——注意力机制本身就可以理解为"如果把 LSTM 的遗忘门换成对全部历史状态的可学习回望，会怎样？"——理解 RNN 是理解 Transformer 最直接的路径。

---

## 核心要点

- **RNN 通过循环连接和参数共享处理带记忆的序列**——同一组权重作用于每一个时间步。
- **基础 RNN 在长序列上失败**，因为雅可比连乘 $\prod \partial h_{k+1} / \partial h_k$ 会指数级缩水（或爆炸）。
- **LSTM 引入加法式细胞状态高速公路**，由遗忘/输入/输出三个门控制，给梯度留出一条不消失的通路。
- **GRU 把 LSTM 简化**为两个门、一个状态，常常用比 LSTM 少 25% 的参数达到相当的精度。
- **双向和堆叠变体**分别拓宽了每个位置的上下文、加深了网络。
- **Seq2Seq 编码器-解码器**实现了序列到序列的映射，但被单一上下文向量 $c$ 卡住——这个瓶颈正是注意力机制（第 4 部分）的直接动因。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 1 | NLP 入门与文本预处理 | [<-- 阅读](/zh/自然语言处理-一-NLP入门与文本预处理/) |
| 2 | 词向量与语言模型 | [<-- 上一篇](/zh/自然语言处理-二-词向量与语言模型/) |
| **3** | **RNN 与序列建模（本文）** | |
| 4 | 注意力机制与 Transformer | [下一篇 -->](/zh/自然语言处理-四-注意力机制与Transformer/) |
| 5 | BERT 与预训练模型 | [阅读 -->](/zh/自然语言处理-五-BERT与预训练模型/) |
| 6 | GPT 与生成式语言模型 | [阅读 -->](/zh/自然语言处理-六-GPT与生成式语言模型/) |
