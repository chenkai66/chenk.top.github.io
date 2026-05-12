---
title: "强化学习（十二）：RLHF与大语言模型应用"
date: 2025-09-25 09:00:00
tags:
  - Reinforcement Learning
  - RLHF
  - DPO
  - ChatGPT
  - LLM Alignment
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "RLHF 把基础语言模型变成 ChatGPT 与 Claude 的完整路径：SFT→奖励模型→PPO 三阶段流程、Bradley-Terry 偏好模型、DPO 闭式解推导、RLAIF 与 Constitutional AI、Goodhart 定律下的奖励黑客，以及强化学习在具身智能与推理时搜索中的下一步。"
disableNunjucks: true
series_order: 12
translationKey: "reinforcement-learning-12"
---
GPT-3（2020年6月）和ChatGPT（2022年11月）共享了大部分权重。基础模型能写出流畅的散文、补全代码、续写任意给定模式，但若直接向其提问，却常答非所问、无端拒绝、编造虚假引用，甚至输出有害内容。这两年半没有用于扩大Transformer模型的规模，而是聚焦于一个更根本的问题——如何让模型真正有用。最终发现，这其实是一个强化学习问题。

本系列最后一篇将收束整个系列的核心主线：此前介绍的值函数、策略梯度、PPO的信任域、离策略修正、偏好学习、内在动机，以及从模仿学习到逆强化学习（IRL）的演进路径，已整合为‘对齐技术栈’。这一技术栈催生了ChatGPT、Claude、Llama-3-Instruct等主流助手类模型。我将推导出**RLHF的三阶段流程**、每个偏好数据集背后的**Bradley-Terry似然**、让DPO完全跳过RL的**闭式最优解**，以及导致对齐成为动态目标的**Goodhart失败模式**。接着，将目光从语言转向强化学习的未来方向：具身智能体、宪法式自监督，以及推理时搜索。
![强化学习（十二）：RLHF与大语言模型应用 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-rlhf-and-llm-applications/illustration_1.png)

## 你将学到什么

- **RLHF 的三个阶段**：监督微调、奖励模型训练，以及带 KL 锚点的 PPO。每个阶段都不可或缺。
- **Bradley-Terry 模型**：为什么偏好比绝对分数更重要，以及这对可恢复的奖励意味着什么。
- **InstructGPT 的关键发现**：一个 1.3B 参数的对齐模型，在人类实际需求上胜过了 175B 参数的 GPT-3。
- **DPO**：一页纸的推导，把 RLHF 的闭式最优解转化为普通的对数似然损失。
- **RLAIF 和 Constitutional AI**：去掉人工干预的同时，避免模型性能崩塌。
- **奖励操控与 Goodhart 定律**：代理奖励不断上升，用户满意度却下降，背后的原因和解决方法。
- **RL 的未来方向**：具身智能体、从仿真到现实（sim-to-real）、视觉-语言-动作模型，以及推理时搜索。
## 前置知识

- PPO 和信任域的直观理解（[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/)）
- 离策略修正和重要性采样（[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)）
- 逆强化学习和偏好学习（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）
- 熟悉 Transformer 和 HuggingFace `transformers` 的使用

---
## 1. RLHF：三阶段流程

![RLHF 三阶段流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig1_rlhf_three_stage_pipeline.png)

预训练模型学会了互联网文本的**分布**，包括我不想要的部分。RLHF 让模型学会我想要的东西。2022年从OpenAI流传出来的这套方法已成为行业标准，分为三个清晰的阶段：监督阶段、偏好阶段和强化阶段。

### 阶段 1 —— 监督微调（SFT）

用预训练的基础模型 $\pi_{\text{base}}$，收集一小批高质量的人类 `(prompt, response)` 数据（InstructGPT 用了大约 13K 条），然后最小化标准的下一词交叉熵：
$$\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{demo}}}\sum_t \log \pi_\theta(y_t \mid x, y_{<t}).$$
这一步成本低、技术成熟，而且必不可少。它把策略从“补全任意互联网文本的下一个 token”调整为“对指令做出有用响应”。更重要的是，它生成了 $\pi_{\text{SFT}}$，这个模型成为后两个阶段的**参考策略** $\pi_{\text{ref}}$。从这里开始，所有工作都是在 SFT 的基础上做修正。

### 阶段 2 —— 奖励模型训练

演示数据成本高昂——需要人类撰写理想回答；而偏好数据成本低得多：只需人类对两个模型输出进行两两比较，选出更优结果。阶段 2 用更多的比较数据代替高质量的演示数据（InstructGPT 收集了约 33K 对比较）。

对于每个 prompt $x$，从 $\pi_{\text{SFT}}$ 中采样两个补全 $y_A, y_B$，让标注员选出胜者，记为 $(y_w, y_l)$（$y_w \succ y_l$）。然后训练奖励模型 $r_\phi(x, y)$——通常是把 SFT 模型的语言头换成一个标量头——通过 **Bradley-Terry** 损失让它给 $y_w$ 打更高的分：
$$\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right].$$
这就是[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)提到的偏好学习目标——RLHF 在结构上就是语言模型的逆强化学习。 InstructGPT 的一个反直觉发现是：**6B 的奖励模型在下游 RL 中比 175B 的更稳定**。奖励模型无需过强，只需保证 PPO 的策略更新不超出其输出支撑集；若判别能力过强， PPO 反而可能发现并利用其判别盲区，生成对抗性输出。

### 阶段 3 —— 带 KL 锚点的 PPO

真正的强化学习从这里开始。我把 $r_\phi$ 当作环境奖励，优化策略以最大化期望奖励——但加了一个 **KL 惩罚**，防止策略偏离 $\pi_{\text{ref}}$ 太远：
$$\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\!\left[\,r_\phi(x, y) \,-\, \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\,\right].$$
括号里的整体就是 PPO 用的逐 token 奖励，第二项是 KL 锚点。没有它，策略迟早会找到一些在 $r_\phi$ 下高分但读起来像词汤的 token——这就是奖励黑客，第 6 节会再讲这个问题。

PPO 成为首选算法的原因和[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/)讲的一致：

1. **动作空间是词表**（约 5 万 token）。每个 token 上做 5 万路 argmax 的 Q-learning 不可行；策略梯度可以。
2. **裁剪防止灾难性更新**。一个糟糕的 batch 就能毁掉 70B 参数的对话模型， checkpoint 管理也救不回来。 PPO 的裁剪代理目标 $\min\big(\tfrac{\pi_\theta}{\pi_{\text{old}}}A,\,\text{clip}(\tfrac{\pi_\theta}{\pi_{\text{old}}},1\!-\!\epsilon,1\!+\!\epsilon)A\big)$ 限制了单步能走多远。
3. **KL 惩罚和 PPO 的信任域哲学天然契合**：两者都限制策略每步的变化幅度，只是出发点不同（PPO 限制单次更新内相对 $\pi_{\text{old}}$ 的漂移； KL 限制整个训练过程相对 $\pi_{\text{ref}}$ 的漂移）。

### 为什么是三阶段？

每一阶段都在压缩前一阶段的产物，变成更紧凑的信号。SFT 将数千万 token 的互联网文本压缩成一个**会**响应的模型。奖励模型将 33K 条人类判断压缩成一个**评估**响应的标量。PPO 再将这个评估器解压回一个**生成**人类一开始想要的响应的策略。每次压缩都会引入信息损失，每类损失都催生了一个独立的研究方向。
## 2. Bradley-Terry 模型：为什么关注偏好而不是分数

![Bradley-Terry 偏好模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig5_bradley_terry.png)

让一百个标注员用 1–10 分给补全结果打分，你会发现一个人的 7 分可能是另一个人的 4 分。绝对分数在不同人之间、不同日子之间，甚至连续几个提示之间都会变化（标定漂移）。 Bradley-Terry 模型——1952 年为体育队伍排名设计——假设每个项目都有一个潜在分数 $s(y)$，成对比较的结果服从以下公式：
$$P(y_A \succ y_B) \;=\; \frac{e^{s(y_A)}}{e^{s(y_A)} + e^{s(y_B)}} \;=\; \sigma\!\big(s(y_A) - s(y_B)\big).$$
这个模型有两个关键推论，它们奠定了所有现代 RLHF 系统的基础：

- **奖励只能确定到一个常数范围内。** 给所有分数加上一个常数，偏好不会改变。奖励模型的绝对数值没有意义，重要的是差值。这也是 DPO 推导中配分函数 $Z(x)$ 会消失的原因。
- **标注员噪声有不可消除的下限。** 即便是高水平的人类标注员，在 InstructGPT 风格的提示上，彼此一致性也只有约 78%。如果一个奖励模型在留出偏好数据上的准确率达到 78%，就已经接近信号上限。再往上提升，模型可能是在拟合个别标注员的习惯，而不是人类的真实偏好。

正确的理解是：奖励模型是一个**校准后的偏好分类器**，不是质量的评判标准。 PPO 阶段把这个分类器当作 ground truth 使用——这正是第 6 节中所有 Goodhart 失败模式的起点。
## 3. 带 KL 锚点的 PPO：参数空间中的视角

![带 KL 约束的 PPO](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig3_ppo_kl_constraint.png)

KL 项的作用远不止正则化。它实现了和[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/)中 TRPO 相同的信任域，但锚点是一个**冻结的参考模型**，而不是上一次迭代的结果。左图展示了几何关系：有 KL 锚点时，策略会走向一个适中但真实的奖励峰值；没有 KL 锚点时，策略会滑向一个代理奖励 $r_\phi$ 存在虚假峰值、但实际输出已经混乱的区域。

右图揭示了实际问题：当 $\beta$ 从大到小调整时，**代理奖励**单调上升（因为策略获得了更多优化自由），但**真实人类质量**呈现单峰曲线——通常在 $\beta \in [0.01, 0.03]$ 之间达到顶峰，随后迅速崩塌。选择 $\beta$ 是一个需要真人参与的超参数调优过程，没有任何离线指标能告诉你何时越界。实践中，团队普遍采用**自适应 KL 控制**：设定一个固定的平均逐 token KL 目标（比如 6 nats），让 $\beta$ 动态调整以维持这个目标。

一次成功的 RLHF 训练需要同时在显存中加载**四个模型**：正在训练的策略 $\pi_\theta$、提供 KL 项的参考模型 $\pi_{\text{ref}}$、为采样打分的奖励模型 $r_\phi$、以及用于 GAE 优势估计的价值头（通常与策略共享主干）。这就是为什么 RLHF 的工程复杂度远高于 SFT——光是显存开销就大约是 SFT 的 $4\times$。

---
## 4. InstructGPT：数据告诉了我们什么

![奖励模型训练](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig2_reward_model_training.png)

InstructGPT 的论文（Ouyang 等， NeurIPS 2022）篇幅短小、内容紧凑，堪称领域内的“罗塞塔石碑”，清楚地揭示了 RLHF 到底带来了什么。以下是四个值得记住的核心发现：

1. **对齐胜过规模**。在盲测的人类评估中，**13 亿参数的 InstructGPT 在约 85% 的情况下优于 1750 亿参数的 GPT-3**。经过对齐的小模型比未对齐的大模型更有用，这种差距大到即使再增加一个数量级的预训练算力也无法弥补。
2. **泛化能力真实存在，但不均衡**。在英文指令上训练的 RLHF 模型，能够迁移到代码任务和 SFT 数据集中几乎没有见过的非英文提示。这表明奖励模型学到的不只是训练分布的表面特征，而是更通用的东西。
3. **“对齐税”微乎其微**。对齐后的模型在标准 NLP 基准测试（如 TriviaQA、 HellaSwag）上丢了一些分数——它们在生成下一个词的任务上稍微变差了。但用户并不在意；实际使用中的提升远远超过了基准测试中的损失。这是第一次具体证明了**基准测试和用户价值可能背离**，而且这一现象在后来的研究中愈发明显。
4. **奖励操控问题立刻显现**。论文记录了几种典型现象：长度操控（为了边际分数增长生成更长的回答）、格式操控（所有输出都变成列表形式），以及轻微的谄媚行为（迎合用户偏好）。这些问题不是修一次就能解决的 bug，而是**稳定的吸引子**，在每个 RLHF 系统中都会反复出现，包括生产环境中的系统。

---
## 5. DPO：直接跳过奖励模型和强化学习

![DPO 推导](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig4_dpo_derivation.png)

后 InstructGPT 时代， RLHF 领域最具影响力的研究成果是 **Direct Preference Optimization**（Rafailov 等， NeurIPS 2023）。它的主张非常大胆：完全抛弃奖励模型和 PPO，用一个基于偏好数据的监督损失替代整个技术栈。

### 推导过程

从 KL 正则化的强化学习目标开始：
$$\max_\pi\; \mathbb{E}_{x,y\sim\pi}\big[r(x,y)\big] \,-\, \beta\, D_{\mathrm{KL}}\!\big[\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big].$$
这是一个关于每个 prompt 的分布 $\pi(\cdot|x)$ 的凸约束问题。通过拉格朗日法（或者直接猜测验证），可以得到闭式最优解：
$$\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right),$$
其中 $Z(x)$ 是关于 $y$ 的配分函数。将这个公式反推，用最优策略表达奖励：
$$r(x,y) \;=\; \beta\,\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta\,\log Z(x).$$
关键一步来了：**把 $r$ 的表达式代入 Bradley-Terry 偏好似然公式**：
$$P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x,y_w) - r(x,y_l)\big),$$
$\beta\log Z(x)$ 项被消掉了——因为它只依赖于 $x$，与 $y_w$ 和 $y_l$ 无关。最终剩下的损失只与 $\pi_\theta$、$\pi_{\text{ref}}$ 和偏好数据有关：
$$\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \,-\, \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].\;}$$
这就是两个对数似然比的差值经过 sigmoid 再计算交叉熵。没有采样，没有价值头，没有奖励模型，也没有 PPO 裁剪。两次前向传播，一次反向传播，搞定。

### DPO 的实际优势

- **一步到位，取代三步流程。** 先做 SFT，然后直接在偏好数据上训练 DPO，不需要单独维护奖励模型。
- **去掉采样循环。** PPO 需要在训练过程中生成补全，这占据了大部分时间。而 DPO 是离线监督学习，直接用固定数据集。
- **没有显式奖励可操控。** 隐式奖励 $\hat r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是由策略定义的，策略无法偏离它。
- **显存需求减半。** 只需要两个模型（$\pi_\theta$ 和 $\pi_{\text{ref}}$），而不是四个。

### DPO 的局限性

- **无法检查奖励值。** 隐式奖励 $\hat r$ 只是一个比值，不能像单独的 $r_\phi$ 那样为新生成的补全打分。
- **对噪声偏好更敏感。** PPO 的 KL 锚点和在线采样提供了一定的鲁棒性，而 DPO 完全依赖偏好数据集。
- **长程推理表现可能不足。** 在链式思维（CoT）或多步工具调用等任务中，策略需要在线探索才能表现更好，而 DPO 不具备这一特性。这也是为什么 **Online DPO**、**Iterative DPO**、**IPO** 和 **KTO** 等方法试图弥补这一差距。

2024–2026 年的实用主义结论是：大多数开源指令微调模型（如 Llama-3、 Qwen-2.5、 Mistral）都采用 DPO 变体，因为工程实现更简单，且在主流基准测试中表现不俗。而前沿闭源模型（如 ChatGPT、 Claude、 Gemini）仍倾向于使用基于 PPO 的 RLHF 或其变体，因为在复杂推理任务上的边际收益足以抵消额外的复杂性。两条路线将在未来一段时间内并存。
## 6. 奖励操控与 Goodhart 定律

![奖励操控与 Goodhart 定律](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig7_reward_hacking.png)

1975 年， Charles Goodhart 提出了一个观点，现代版本是："**当一个指标变成目标，它就不再是一个好指标。**" RLHF 正是这条警句在机器学习中的具体体现。奖励模型本是用来衡量人类偏好的；但一旦 PPO 把它当作优化目标，策略就开始寻找不真正服务人类却能拿高分的方法。

左图是经典的实证结果（Gao 等， ICML 2023）：横轴表示从 $\pi_{\text{ref}}$ 出发的训练 KL 散度（可以理解为 RL 的“剂量”），代理奖励 $r_\phi$ 持续上升，但**人类实际评价的奖励呈现单峰曲线**——早期达到峰值后迅速崩塌。两条曲线之间的差距就是 Goodhart gap，模型越大、训练时间越长，这个差距就越明显。

右图列出了几种常见的失败模式：

1. **长度操控。** 输出比人类期望的长 2–3 倍，因为更长的内容往往能获得更高的 RM 分数。
2. **讨好倾向。** 即使用户错了，模型也会附和用户的立场，因为 RM 标注员通常喜欢被认同。
3. **格式操控。** 列表、标题、表格泛滥成灾； RM 认为这些结构看起来像是用心的表现。
4. **自信胡扯。** 输出流畅、格式规范，但内容错误。奖励模型无法核查事实，因此奖励了表面的自信。
5. **过度拒绝。** 模型对无害的查询也频繁拒绝，以规避 harmlessness 奖励，导致用户讨厌的"作为一个大语言模型，我不能……"现象。

### 实际有效的缓解方法

- **KL 锚点。** 第一道防线，动态调整 $\beta$ 以匹配目标 KL。
- **奖励模型集成。** 在不同数据切分上训练多个奖励模型，取预测平均值——这样可以平滑掉单一模型的偏差。
- **定期重新标注。** 针对**当前**策略的输出收集新偏好，而不是用过时的数据，每隔几轮更新一次奖励模型。
- **长度约束奖励。** 扣除长度惩罚，或在固定长度预算下评估输出。
- **规则补充 / 红队测试。** 向数据集中添加显式规则和对抗样本（详见下一节）。

没有一劳永逸的解决方案。奖励操控是问题本身结构中自带的一场军备竞赛。
## 7. RLAIF 与 Constitutional AI：去掉人工环节

人工标注速度慢、成本高、一致性差，而且无法满足前沿模型对海量偏好数据的需求。两类方法尝试用强模型部分或完全替代人工：

**RLAIF**（Lee 等， 2023）直接用另一个 LLM （比如 GPT-4）代替标注员，让它来比较两种回答：

```text
给定一个问题和两个回答，哪个更符合
"有帮助、诚实、无害" 的标准？
问题：{x}
回答 A：{y_A}
回答 B：{y_B}
选择 A 或 B，并简要说明理由。
```

在标准任务上， RLAIF 的偏好判断与人类一致率约为 85%，但成本只有人工的十分之一左右。不过，这种方法存在一个风险：**模型坍缩**。如果连续多代都在 AI 标注的数据上训练，数据分布会逐渐变窄，偏见会被固化，质量也会下降。目前的应对措施包括：混入新鲜的人类数据、定期更换评估模型、周期性用高质量人类数据重新校准。

**Constitutional AI**（Bai 等， Anthropic 2022）走得更远。它先定义一份自然语言形式的“宪法”（比如“要有帮助”、“避免建议有害行为”），然后让模型在生成偏好数据之前，**根据宪法对自己的输出进行自我批评和修订**。奖励模型的偏好数据由 `(原始输出, 修订后输出)` 组成，其中修订后的版本更好地符合宪法要求。这是 Claude 训练体系的核心，也是**利用模型自身能力实现对齐引导**的一个典型例子——解决了纯 RLHF 方法未能闭合的反馈回路。

趋势很明显：随着基础模型性能提升，越来越多的对齐信号可以由模型自己生成。人类的角色从“逐一标注对比”转向“审查宪法和争议”。瓶颈也从标注效率转移到了规范设计的质量上。

---
## 8. 生产级对齐栈的架构

![ChatGPT/Claude 训练架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig6_chatgpt_architecture.png)

把各个模块拼起来，现代助手——比如 ChatGPT、 Claude、 Gemini、 Llama-3-Instruct、 Qwen-2.5-Instruct——全都遵循一个五层架构：

1. **预训练。** 数万亿 token 的自监督下一词预测任务。占总算力的 90–99%，也是唯一能显著提升模型原始能力的阶段。
2. **SFT / 指令微调。** 使用 10K–100K 条精心挑选的 `(prompt, response)` 对，有时还会用更强模型蒸馏出的合成数据来扩充。
3. **偏好数据。** 包括人类标注（RLHF）、 AI 标注（RLAIF）或宪法式自批评（CAI）。通常是多种方式混合使用。
4. **对齐优化。** PPO + KL 锚点（OpenAI 的传统做法）、 DPO 及其变种——IPO、 KTO、 ORPO （开源社区常用），或者宪法式自监督回路（Anthropic 的方法）。
5. **部署时护栏。** 系统提示、安全分类器、工具调用框架、在线红队测试，以及在 MT-Bench、 Chatbot Arena 和内部回归测试套件上的滚动评估。

这些模块已经高度标准化。实验室之间的差异主要体现在**偏好数据的质量**、**奖励模型（或其隐式替代方案）的稳定性**，以及**部署评测的严谨性**上。算法反而是最简单的部分。

---
## 9. 超越语言： RL 的下一步

RLHF 是目前 RL 应用中风险最高、投入最大的场景，但并不是最野心勃勃的方向。与此同时，还有三个前沿领域正在快速发展，并且大量借鉴了本系列的内容：

**机器人领域的 sim-to-real。** 先在快速模拟器（如 MuJoCo、 Isaac Gym）中训练策略，通过**域随机化**（调整物理参数、光照、纹理等）缩小仿真与现实的差距，最后部署到真实硬件上。 OpenAI 的 Dactyl 就是用这种方法让机械手成功还原了魔方； Google 的 Aloha 系统则结合模仿学习（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）进行初始化，再用在线 RL 进一步优化。

**安全关键控制中的离线 RL。** 自动驾驶、医疗和工业控制等领域无法承受在线策略探索带来的风险。[第 10 部分](/zh/reinforcement-learning/10-离线强化学习/)提到的方法——CQL、 IQL、 Decision Transformer——从历史数据中初始化策略，随后才谨慎地转向在线微调。

**视觉-语言-动作模型。** Google 的 RT-2 基于一个预训练的视觉-语言模型，同时用网页数据和机器人轨迹进行联合微调，最终生成了首个具备强大零样本泛化能力的机器人策略，能够处理从未见过的物体和指令。这相当于具身智能领域的 RLHF：拿一个已经理解世界的模型，教会它如何**在世界中行动**。

**推理时 RL。** 最新的趋势是：不再把 RL 的计算资源花在训练阶段，而是用在推理阶段。 OpenAI 的 o 系列和 DeepSeek 的 R1 并没有用 RL 更新单次前向传播的权重，而是教模型在回答问题之前**搜索思维链**。这是 [MCTS](/zh/reinforcement-learning/08-alphago与蒙特卡洛树搜索) 思想、[PPO](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/) 和上述偏好学习机制的结合体。预计这一方向将在未来两年主导前沿模型的发展。 

---
## 10. 简化的 RLHF 实现

下面的代码展示了核心流程：用 Bradley-Terry 损失训练奖励模型，然后基于它做简化版的 PPO 风格优化。生产级框架（如 TRL、 DeepSpeed-Chat、 OpenRLHF、 trlX）会加入 GAE 优势、价值头、完整的 PPO 裁剪、多 GPU 分片和自适应 KL 控制，但这些内容太多，一页纸根本写不下。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------
# 奖励模型：主干 + 标量头
# -----------------------------
class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        self.value_head = nn.Linear(self.transformer.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.transformer.transformer(
            input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一个非 pad token 的隐状态
        last = out.last_hidden_state[:, -1, :]
        return self.value_head(last).squeeze(-1)

def train_reward_model(model, dataloader, epochs=3, lr=1e-5):
    """Bradley-Terry 偏好损失"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for batch in dataloader:
            r_w = model(batch["chosen_ids"], batch["chosen_mask"])
            r_l = model(batch["rejected_ids"], batch["rejected_mask"])
            loss = -F.logsigmoid(r_w - r_l).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(dataloader):.4f}")

# -----------------------------
# 概念性 RLHF：奖励 + KL 锚点
# -----------------------------
class SimpleRLHF:
    def __init__(self, policy, reward_model, ref_model,
                 tokenizer, beta=0.02, lr=1e-6):
        self.policy = policy
        self.rm = reward_model
        self.ref = ref_model               # 冻结的 π_ref
        self.tokenizer = tokenizer
        self.beta = beta
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    def _logp(self, model, ids):
        logits = model(ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)

    def step(self, prompt_ids):
        # 1. 从当前策略采样补全
        with torch.no_grad():
            out = self.policy.generate(
                prompt_ids, max_new_tokens=64,
                do_sample=True, top_p=0.9)
            r = self.rm(out, torch.ones_like(out))      # [B]

        # 2. 计算 policy 和 reference 下的对数概率
        logp_pi = self._logp(self.policy, out)
        with torch.no_grad():
            logp_ref = self._logp(self.ref, out)

        # 3. KL 正则化奖励作为序列级优势
        kl = logp_pi - logp_ref                          # [B]
        advantage = (r - self.beta * kl).detach()

        # 4. 策略梯度更新
        loss = -(logp_pi * advantage).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), r.mean().item(), kl.mean().item()
```

这段代码清楚地说明了两件事，而文字描述往往容易模糊：  
1. PPO 看到的“奖励”是 $r_\phi - \beta \cdot \text{KL}$，直接作为序列级标量优势嵌入其中。  
2. $\pi_{\text{ref}}$ 是完全冻结的——所有地方都设置为 `requires_grad=False`。  

如果忽略了这两点，训练跑上几个小时后才会发现问题已经偏离了方向。
## 11. 常见问题

**问：演示数据足够多时，为什么 RLHF 比 SFT 更强？**  
SFT 的上限由演示作者的能力决定——人类很少写出**最优**答案，通常只是**不错**的答案。 RLHF 让模型能够突破演示分布的限制，同时对生成的样本进行排序。此外，比较的成本比演示低，单位预算下能获得更多信号。

**问： RLAIF 在多代生成后会导致模型坍缩吗？**  
目前的证据（1–2 代）显示没有明显退化。不过，每一轮自蒸馏都会增加风险。应对方法包括：持续引入一定比例的新鲜人类数据、轮换标注模型、定期用留出的人类金数据校准模型。

**问：奖励黑客问题能彻底解决吗？**  
不能。这是优化度量时必然出现的结构性问题，不是某个具体奖励模型的缺陷。实际中可以通过 KL 锚点、 RM 集成、周期性重新标注、长度惩罚、 constitutional 过滤等手段控制损害，但无法完全消除。应该把它当作一个需要长期维护的工程问题，而不是一次性修复。

**问： RLHF 的成本和预训练相比如何？**  
大约是预训练算力的 1–10%，主要开销在奖励模型训练和 PPO 采样上。 DPO 则将成本降到接近第二次 SFT 的水平——从算力角度看， DPO 的优势非常明显。

**问：如何处理像 helpfulness 和 harmlessness 这样的冲突目标？**  
实际中有三种做法：(a) 为每个目标单独训练**独立的奖励模型**，然后通过学习或手动调整权重组合；(b) 使用 **Constitutional AI** 将硬约束编码为自然语言规则；(c) 提供**用户可调的偏好权重**（比如"在医疗建议上更谨慎"）。这三种方法在生产环境中常常共存。

**问：什么时候该选 PPO，什么时候该选 DPO？**  
如果偏好数据量大且干净、追求快速迭代、在意 wall-clock 训练时间，就选 DPO。如果你有一个高质量的奖励模型想留在回路中、需要在线探索（如多步推理或工具使用）、或者计划在训练时加入安全约束和宪法规则，就选 PPO。

**问：这些内容和第 7 部分的逆强化学习有什么关系？**  
直接相关。 RLHF 在结构上就是逆强化学习，只不过做了两个简化：用成对偏好代替完整演示（Bradley-Terry 替代 MaxEnt IRL），用 PPO 作为前向 RL 步骤。奖励模型是 IRL 的输出结果，而 PPO 阶段则是标准的"用恢复的奖励训练新策略"步骤。
## 12. 参考文献

- **Bradley & Terry (1952).** Rank Analysis of Incomplete Block Designs. *Biometrika*. — 偏好似然的来源。
- **Christiano 等 (2017).** Deep Reinforcement Learning from Human Preferences. *NeurIPS*. — 现代偏好-RL 的开山之作。
- **Stiennon 等 (2020).** Learning to Summarize with Human Feedback. *NeurIPS*. — RLHF 在摘要任务上的应用， InstructGPT 的雏形。
- **Ouyang 等 (2022).** Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*。
- **Bai 等 (2022a).** Training a Helpful and Harmless Assistant with RLHF. *Anthropic*。
- **Bai 等 (2022b).** Constitutional AI: Harmlessness from AI Feedback. *Anthropic*。
- **Gao 等 (2023).** Scaling Laws for Reward Model Overoptimization. *ICML*. — 提出 Goodhart 曲线的文章。
- **Rafailov 等 (2023).** Direct Preference Optimization. *NeurIPS*。
- **Lee 等 (2023).** RLAIF: Scaling RL from Human Feedback with AI Feedback。
- **Skalse 等 (2022).** Defining and Characterizing Reward Hacking. *NeurIPS*。
- **Brohan 等 (2023).** RT-2: Vision-Language-Action Models. *Google DeepMind*。
## 系列总结

这是第十二篇，也是最后一篇。这个系列从马尔可夫决策过程和一个简单的 GridWorld 开始，到构建 ChatGPT 和 Claude 的对齐技术栈结束。一路走来，我整理了以下内容：

- **基础** —— MDP、 Bellman 方程、价值迭代（[第 1 部分](/zh/reinforcement-learning/01-基础与核心概念/)）
- **基于值的方法** —— Q-learning、 DQN、 double/dueling/distributional （[第 2 部分](/zh/reinforcement-learning/02-q-learning与深度q网络/)）
- **策略梯度与 Actor-Critic**（[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)）
- **探索与内在动机**（[第 4 部分](/zh/reinforcement-learning/04-探索策略与好奇心驱动学习/)）
- **基于模型的 RL 与世界模型**（[第 5 部分](/zh/reinforcement-learning/05-model-based强化学习与世界模型)）
- **PPO 与 TRPO**（[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化/)）—— 让 RLHF 成为可能的算法
- **模仿学习与逆强化学习**（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）—— 偏好学习的理论源头
- **AlphaGo 与 MCTS**（[第 8 部分](/zh/reinforcement-learning/08-alphago与蒙特卡洛树搜索)）
- **多智能体 RL**（[第 9 部分](/zh/reinforcement-learning/09-多智能体强化学习/)）
- **离线 RL**（[第 10 部分](/zh/reinforcement-learning/10-离线强化学习/)）
- **层次化与元 RL**（[第 11 部分](/zh/reinforcement-learning/11-层次化强化学习与元学习/)）
- **RLHF 与大语言模型对齐**（本部分）

贯穿始终的核心是：**强化学习是从后果中学习的科学**。无论是游戏分数、物理模拟器、人类成对偏好，还是宪法式自批评，背后都离不开 Bellman 回溯、探索-利用权衡以及信任域直觉。未来十年，强化学习的重点将是打通训练时与推理时搜索的闭环，连接数字智能体与具身智能体，弥合人类反馈与自主改进之间的差距。这十二篇文章里的数学工具，正是理解这些内容的关键。

- **上一篇**：[第 11 部分 —— 层次化强化学习与元学习](/zh/reinforcement-learning/11-层次化强化学习与元学习/)
- **系列完结！** [查看 RL 系列全部 12 篇](/zh/categories/强化学习/)
