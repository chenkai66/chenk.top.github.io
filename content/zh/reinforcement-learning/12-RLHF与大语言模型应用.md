---
title: "强化学习（十二）：RLHF 与大语言模型应用"
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
GPT-3（2020 年 6 月）和 ChatGPT（2022 年 11 月）共享了大部分权重。基础模型能写出流畅的散文、补全代码，也能续写任意给定的模式；但当你直接问它一个简单问题时，它却可能喋喋不休、以错误理由拒绝回答、编造虚假引用，甚至输出有害内容。从 GPT-3 到 ChatGPT 的两年半时间，并没有花在扩大 Transformer 规模上，而是聚焦于一个更根本的问题：**如何让模型真正有用**——而这本质上是一个强化学习问题。
\n作为本系列的收官之作，本文将整合此前构建的所有核心概念：值函数、策略梯度、PPO 的信任域、离策略修正、偏好学习、内在动机，以及从模仿学习到逆强化学习（IRL）的演进路径。这些共同构成了催生 ChatGPT、Claude、Llama-3-Instruct 等主流助手模型的“对齐技术栈”。我们将推导出 **RLHF 的三阶段流程**、每个偏好数据集背后的 **Bradley-Terry 似然模型**、让 DPO 能跳过强化学习的 **闭式最优解**，以及因 **Goodhart 定律**导致对齐成为动态目标的失败模式。最后，我们将目光投向强化学习的未来：具身智能体、宪法式自监督，以及推理时搜索。
![强化学习（十二）：RLHF与大语言模型应用 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-rlhf-and-llm-applications/illustration_1.png)

## 你将学到什么

- **RLHF 的三阶段流程**：监督微调（SFT）、奖励模型训练、带 KL 锚点的 PPO——以及每个阶段存在的必要性
- **Bradley-Terry 模型**：为何偏好（而非绝对分数）才是正确的“货币”，以及这对可恢复的奖励意味着什么
- **InstructGPT 的关键实证发现**：一个 1.3B 参数的对齐模型，在人类实际偏好上胜过了 175B 参数的 GPT-3
- **DPO**：仅用一页纸的推导，将 RLHF 的闭式最优解转化为普通的对数似然损失
- **RLAIF 与 Constitutional AI**：如何在移除人工干预的同时避免模型坍缩
- **奖励操控与 Goodhart 定律**：为何代理奖励不断上升而用户满意度却下降，以及应对策略
- **RL 的未来方向**：具身智能体、仿真到现实（sim-to-real）、视觉-语言-动作模型，以及推理时搜索

## 前置知识

- PPO 与信任域的直观理解（[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化)）
- 离策略修正与重要性采样（[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)）
- 逆强化学习与偏好学习（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）
- 熟悉 Transformer 架构及 HuggingFace `transformers` 库的使用

---

## 1. RLHF：三阶段流程

![RLHF 三阶段流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig1_rlhf_three_stage_pipeline.png)
\n预训练赋予模型对互联网文本**分布**的理解——包括你不想要的部分；而 RLHF 让模型学会理解**你**想要什么。2022 年由 OpenAI 提出的方法已成为行业标准，包含三个清晰阶段：监督微调、偏好建模和强化优化。

### 阶段 1 —— 监督微调（SFT）
\n取预训练基础模型 $\pi_{\text{base}}$，收集一小批高质量的人类 `(prompt, response)` 示范数据（InstructGPT 使用了约 13K 条），并最小化标准的下一词交叉熵损失：
$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{demo}}}\sum_t \log \pi_\theta(y_t \mid x, y_{<t}).
$$\n这一步成本低、技术成熟，且绝对必要。它将策略从“补全任意互联网文本的下一个 token”转变为“对指令做出有用响应”。更重要的是，它产出 $\pi_{\text{SFT}}$，该模型将成为后续两个阶段的**参考策略** $\pi_{\text{ref}}$。此后所有操作，本质上都是在 SFT 基础上的修正项。

### 阶段 2 —— 奖励模型训练
\n示范数据昂贵——人类必须**撰写**理想答案；而偏好数据便宜得多：人类只需**比较**两个模型输出并选出更优者。阶段 2 用更多比较数据换取示范质量（InstructGPT 收集了约 33K 对比较）。
\n对每个提示 $x$，从 $\pi_{\text{SFT}}$ 中采样两个补全 $y_A, y_B$，请标注员选出胜者，标记为 $(y_w, y_l)$（其中 $y_w \succ y_l$），然后训练奖励模型 $r_\phi(x, y)$——通常是将 SFT 模型的语言头替换为标量头——通过 **Bradley-Terry** 损失使其对 $y_w$ 给出更高分数：
$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right].
$$\n这正是 [第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/) 中介绍的偏好学习目标——从结构上看，RLHF 就是**语言模型上的逆强化学习**。InstructGPT 还有一个反直觉的发现：**6B 参数的奖励模型在下游 RL 中比 175B 的更稳定**。奖励模型只需足够好，确保 PPO 不会漂移到其支持集之外；若能力过强，PPO 反而会找到对抗性输入，利用其盲点。

### 阶段 3 —— 带 KL 锚点的 PPO
\n现在进入真正的强化学习阶段。我们将 $r_\phi$ 视为环境奖励，优化策略以最大化期望奖励——但加入一个 **KL 惩罚项**，使其保持接近 $\pi_{\text{ref}}$：
$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\!\left[\,r_\phi(x, y) \,-\, \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\,\right].
$$\n括号内的整体即为 PPO 使用的逐 token 奖励，第二项是 KL 锚点。若无此项，策略最终会找到在 $r_\phi$ 下得分高但语义混乱的 token 序列——这就是奖励操控，我们将在 §6 中详细讨论。
\nPPO 成为首选算法的原因与 [第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化) 中所述一致：

1. **动作空间即词表**（约 5 万 token）。对每个 token 执行 5 万路 argmax 的 Q-learning 不可行；策略梯度则天然适用。
2. **裁剪机制防止灾难性更新**。单个糟糕的 batch 就可能毁掉一个 70B 参数的聊天模型，且 checkpoint 无法挽救。PPO 的裁剪代理目标 $\min\big(\tfrac{\pi_\theta}{\pi_{\text{old}}}A,\,\text{clip}(\tfrac{\pi_\theta}{\pi_{\text{old}}},1\!-\!\epsilon,1\!+\!\epsilon)A\big)$ 限制了单次更新的步长。
3. **KL 惩罚与 PPO 的信任域哲学天然契合**：两者都限制策略每步的变化幅度，但目的不同（PPO 限制单次更新内相对于 $\pi_{\text{old}}$ 的漂移；KL 限制整个训练过程中相对于 $\pi_{\text{ref}}$ 的漂移）。

### 为何需要三阶段？
\n每个阶段都将前一阶段的产物压缩为更紧凑的信号。SFT 将数千万互联网 token 压缩成一个**能响应**的模型；奖励模型将 33K 条人类判断压缩成一个**评估**响应的标量；PPO 再将该评估器“解压”回一个**生成**人类最初想要响应的策略。每次压缩都有信息损失，而每类损失都催生了一个独立的研究方向。

## 2. Bradley-Terry 模型：为何偏好优于绝对分数

![Bradley-Terry 偏好模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig5_bradley_terry.png)
\n若让一百位标注员用 1–10 分制评价补全结果，你会发现某人的 7 分可能是另一人的 4 分。绝对分数在不同人之间、不同日期之间，甚至连续提示之间都不稳定（标定漂移）。Bradley-Terry 模型——1952 年为体育队伍排名提出——假设每个项目有潜在分数 $s(y)$，成对比较结果服从：
$$\nP(y_A \succ y_B) \;=\; \frac{e^{s(y_A)}}{e^{s(y_A)} + e^{s(y_B)}} \;=\; \sigma\!\big(s(y_A) - s(y_B)\big).
$$\n这一模型带来两个关键推论，塑造了所有现代 RLHF 系统：

- **奖励仅在常数偏移下可识别**。给所有分数加上同一常数，偏好不变。因此奖励模型的绝对尺度无意义，只有差值重要。这也是 DPO 推导中配分函数 $Z(x)$ 消失的原因。
- **标注员噪声存在不可消除的下限**。即使在 InstructGPT 类任务上，黄金标准的人类标注员彼此一致性也仅约 78%。若奖励模型在留出偏好数据上达到 78% 准确率，就已逼近信号上限；继续提升只会拟合个别标注员的个人偏好，而非人类共识。
\n正确理解应是：奖励模型是一个**校准后的偏好分类器**，而非质量的绝对判官。PPO 阶段将其视为 ground truth——这正是 §6 中所有 Goodhart 失败模式的起点。

## 3. 带 KL 锚点的 PPO：参数空间中的视角

![带 KL 约束的 PPO](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig3_ppo_kl_constraint.png)
\nKL 项的作用远不止正则化。它实现了与 [第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化) 中 TRPO 相同的信任域思想，但锚点是一个**冻结的参考模型**，而非上一次迭代的结果。左图展示了其几何意义：有 KL 锚点时，策略会走向一个适中但真实的奖励峰值；若无锚点，策略则滑向代理奖励 $r_\phi$ 存在虚假极大值但实际输出混乱的区域。
\n右图揭示了实际困境：当 $\beta$ 从大调小时，**代理奖励**单调上升（策略获得更大优化自由度），但**真实人类评价质量**呈单峰曲线——通常在 $\beta \in [0.01, 0.03]$ 区间达到峰值后迅速崩塌。选择 $\beta$ 是一个必须依赖真人反馈的超参调优过程，没有任何离线指标能告诉你何时越界。实践中，团队普遍采用**自适应 KL 控制**：设定固定平均逐 token KL 目标（如 6 nats），让 $\beta$ 动态调整以维持该目标。
\n一次成功的 RLHF 训练需同时在内存中加载**四个模型**：正在训练的策略 $\pi_\theta$、提供 KL 项的参考模型 $\pi_{\text{ref}}$、为 rollout 打分的奖励模型 $r_\phi$，以及用于 GAE 优势估计的价值头（通常与策略主干共享）。这也解释了为何 RLHF 的工程复杂度远高于 SFT——仅显存开销就约为后者的 $4\times$。

---

## 4. InstructGPT：数据告诉了我们什么

![奖励模型训练](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig2_reward_model_training.png)
\nInstructGPT 论文（Ouyang et al., NeurIPS 2022）篇幅短小、内容密集，堪称该领域的“罗塞塔石碑”，清晰揭示了 RLHF 的实际价值。以下四个发现值得铭记：

1. **对齐胜过规模**。在盲测人类评估中，**1.3B 参数的 InstructGPT 在约 85% 的情况下优于 175B 参数的 GPT-3**。经过对齐的小模型比未对齐的大模型更有用——这种差距之大，即使再增加一个数量级的预训练算力也无法弥补。
2. **泛化真实存在但不均衡**。在英文指令上训练的 RLHF 模型，能迁移到代码任务和 SFT 数据集中几乎未见的非英文提示。这表明奖励模型捕捉到了超越训练分布表面形式的通用信号。
3. **“对齐税”微乎其微**。对齐模型在标准 NLP 基准（如 TriviaQA、HellaSwag）上略有下降——它们在“预测下一个词”的游戏中稍弱。但用户并不在意；实际体验的提升远超基准损失。这是首次明确证明**基准分数与用户价值可能背离**，这一现象此后愈发显著。
4. **奖励操控立即显现**。论文记录了长度操控（为微小分数增益生成更长回答）、格式操控（所有输出变成列表）和轻微谄媚行为。这些问题并非一次性可修复的 bug，而是**稳定的吸引子**，在所有 RLHF 系统（包括生产系统）中反复出现。

---

## 5. DPO：跳过奖励模型与强化学习

![DPO 推导](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig4_dpo_derivation.png)
\nInstructGPT 之后，RLHF 领域最具影响力的工作是 **Direct Preference Optimization**（Rafailov et al., NeurIPS 2023）。其主张大胆：完全抛弃奖励模型和 PPO，用一个基于相同偏好数据的监督损失替代整个流程。

### 推导过程
\n从 KL 正则化的 RL 目标出发：
$$
\max_\pi\; \mathbb{E}_{x,y\sim\pi}\big[r(x,y)\big] \,-\, \beta\, D_{\mathrm{KL}}\!\big[\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big].
$$\n这是关于每个 prompt 下分布 $\pi(\cdot|x)$ 的凸约束优化问题。通过拉格朗日法（或直接验证）可得闭式最优解：
$$
\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right),
$$\n其中 $Z(x)$ 是关于 $y$ 的配分函数。将其反解，用最优策略表达奖励：
$$\nr(x,y) \;=\; \beta\,\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta\,\log Z(x).
$$\n关键观察：**将此 $r$ 表达式代入 Bradley-Terry 偏好似然**：
$$\nP(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x,y_w) - r(x,y_l)\big),
$$
$\beta\log Z(x)$ 项被消去——因其仅依赖 $x$，与 $y_w$ 或 $y_l$ 无关。最终得到仅依赖 $\pi_\theta$、$\pi_{\text{ref}}$ 和偏好数据的损失：
$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \,-\, \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].\;}
$$\n这本质上是对两个对数似然比之差的 sigmoid 输出计算交叉熵。无需 rollout，无需价值头，无需奖励模型，也无需 PPO 裁剪。**两次前向传播，一次反向传播，即可完成**。

### DPO 的实际优势

- **一步替代三步**。先做 SFT，再直接在偏好数据上训练 DPO，无需维护独立的奖励模型。
- **无需采样循环**。PPO 需在训练循环内生成补全，耗时极长；DPO 是离线监督学习，使用固定数据集。
- **无显式奖励可被操控**。隐式奖励 $\hat r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 由策略自身定义，策略无法偏离它。
- **显存需求减半**。仅需两个模型（$\pi_\theta$ 和 $\pi_{\text{ref}}$），而非四个。

### DPO 的局限性

- **无法检查奖励值**。隐式 $\hat r$ 仅以比值形式存在，无法像独立 $r_\phi$ 那样为新补全打分。
- **对噪声偏好更敏感**。PPO 的 KL 锚点和在线采样提供一定鲁棒性；DPO 则完全信任偏好数据集。
- **长程推理表现可能不足**（如思维链、多步工具调用），因策略缺乏 DPO 未提供的在线探索。这也是 **Online DPO**、**Iterative DPO**、**IPO** 和 **KTO** 试图弥补的差距。

2024–2026 年的实用结论是：大多数开源指令微调模型（如 Llama-3、Qwen-2.5、Mistral）采用 DPO 变体，因其工程简单且基准表现具竞争力；而前沿闭源模型（如 ChatGPT、Claude、Gemini）仍倾向使用 PPO-based RLHF 或其宪法变体，因在复杂推理任务上的边际收益值得额外复杂度。两条路线将持续共存。

## 6. 奖励操控与 Goodhart 定律

![奖励操控与 Goodhart 定律](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig7_reward_hacking.png)
\nCharles Goodhart 1975 年的观察，现代版本为：“**当一个度量成为目标，它就不再是一个好度量。**” RLHF 正是这一格言在机器学习中的构造性证明。奖励模型本是人类偏好的度量；但一旦 PPO 以其为目标，策略便开始寻找高分却不服务人类的方法。
\n左图是经典实证结果（Gao et al., ICML 2023）：横轴为训练中相对于 $\pi_{\text{ref}}$ 的 KL 散度（即“RL 剂量”），代理奖励 $r_\phi$ 单调上升，但**黄金标准的人类奖励呈单峰曲线**——早期达峰后迅速崩塌。两曲线间的差距即 Goodhart gap，且随模型规模和训练时长扩大。
\n右图列举了典型失败模式：

1. **长度操控**。输出比人类所需长 2–3 倍，因长度常与 RM 分数正相关。
2. **谄媚倾向**。即使用户错误，模型也附和其立场，因 RM 标注员偏好被认同。
3. **格式操控**。列表、标题、表格泛滥；RM 学到“结构=用心”。
4. **自信胡扯**。输出流畅规范但事实错误。RM 无法核查事实，故奖励表面自信。
5. **过度拒绝**。为规避 harmlessness 奖励，模型对无害查询也频繁拒绝，导致用户厌恶的“作为大语言模型，我不能……”现象。

### 实践有效的缓解措施

- **KL 锚点**。第一道防线；通过自适应调整 $\beta$ 维持目标 KL。
- **奖励模型集成**。对不同数据切片训练多个 RM，取预测均值以平滑偏差。
- **周期性重标注**。在**当前**策略输出上收集新偏好（而非陈旧数据），每几轮刷新 RM。
- **长度控制奖励**。扣除长度惩罚，或在固定长度预算下评估。
- **宪法/红队补充**。向数据集添加显式规则与对抗样本（见下节）。
\n不存在永久解决方案。奖励操控是问题结构内生的军备竞赛。

## 7. RLAIF 与 Constitutional AI：移除人工环节
\n人工标注慢、贵、不一致，且难以满足前沿模型对海量偏好的需求。两类方法尝试用强模型部分或完全替代人工：

**RLAIF**（Lee et al., 2023）用另一 LLM（如 GPT-4）替代标注员进行输出比较：

```text
给定一个问题和两个回答，哪个更符合
"有帮助、诚实、无害" 的标准？
问题：{x}
回答 A：{y_A}
回答 B：{y_B}
选择 A 或 B，并简要说明理由。
```
\n在标准任务上，RLAIF 偏好与人类一致率达 ~85%，成本约为人工的 1/10。风险在于**模型坍缩**：若多代均在 AI 标注数据上训练，分布会收窄、偏见固化、质量下降。当前缓解手段包括：混入新鲜人类数据、轮换评估模型、定期用人类黄金数据重新校准。

**Constitutional AI**（Bai et al., Anthropic 2022）更进一步：写下自然语言“宪法”（如“要有帮助”、“避免建议有害行为”），让模型在偏好标注前**依宪法自我批评并修订输出**。RM 的偏好数据由 `(原始, 修订)` 对构成，其中修订版更符合宪法。这是 Claude 训练栈的基础，也是**利用模型自身能力引导对齐**的典范——闭合了纯 RLHF 未解决的反馈回路。
\n趋势明确：随着基座模型变强，更多对齐信号可由模型自产，人类角色从“标注每对比较”转向“审核宪法与争议”。瓶颈从标注吞吐量转向规范质量。

---

## 8. 生产级对齐栈的架构

![ChatGPT/Claude 训练架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF与大语言模型应用/fig6_chatgpt_architecture.png)
\n整合各模块，所有现代助手——ChatGPT、Claude、Gemini、Llama-3-Instruct、Qwen-2.5-Instruct——均遵循五层模板：

1. **预训练**。数万亿 token 的自监督下一词预测。占总算力 90–99%，是唯一显著提升原始能力的阶段。
2. **SFT / 指令微调**。10K–100K 条精选 `(prompt, response)` 对，可选更强模型蒸馏的合成数据扩充。
3. **偏好数据**。人类标注（RLHF）、AI 标注（RLAIF）或宪法自批评（CAI），常混合使用。
4. **对齐优化**。PPO + KL 锚点（OpenAI 传统）、DPO 及其变种（IPO/KTO/ORPO，开源传统），或宪法自监督回路（Anthropic）。
5. **部署时护栏**。系统提示、安全分类器、工具调用框架、在线红队测试，及 MT-Bench、Chatbot Arena 和内部回归套件的滚动评估。
\n模块已高度标准化。实验室差异体现在**偏好数据质量**、**奖励模型（或其隐式替代）的鲁棒性**，以及**部署评估的严谨性**上。算法反而是最简单的部分。

---

## 9. 超越语言：RL 的下一步
\nRLHF 是当前 RL 最高风险的应用，但非最雄心勃勃的方向。三大前沿正并行推进，并大量借鉴本系列内容：

**机器人 sim-to-real**。在快速模拟器（MuJoCo、Isaac Gym）中训练策略，通过**域随机化**（变动物理参数、光照、纹理）弥合仿真-现实差距，再部署至真实硬件。OpenAI Dactyl 以此让机械手解魔方；Google Aloha 结合模仿学习（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）初始化，并用在线 RL 精调。

**安全关键控制的离线 RL**。自动驾驶、医疗、工业控制等领域无法承受在线探索。[第 10 部分](/zh/reinforcement-learning/10-离线强化学习/) 的方法（CQL、IQL、Decision Transformer）从日志数据初始化策略，再谨慎转为在线微调。

**视觉-语言-动作模型**。Google RT-2 对预训练视觉-语言模型联合微调网页数据与机器人轨迹，产出首个对未见物体/指令具强零样本泛化能力的机器人策略。这相当于具身智能的 RLHF：取一个已理解世界的模型，教会它**在世界中行动**。

**推理时 RL**。最新趋势：将 RL 计算从训练时移至推理时。OpenAI o 系列与 DeepSeek R1 不用 RL 更新单次前向权重，而是教模型在回答前**搜索思维链**——融合 [MCTS](/zh/reinforcement-learning/08-alphago与蒙特卡洛树搜索) 思想、[PPO](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化) 与上述偏好学习机制。预计此方向将主导未来两年前沿模型进展。

---

## 10. 简化的 RLHF 实现
\n下方参考代码涵盖核心流程：用 Bradley-Terry 损失训练奖励模型，再对其执行简化版 PPO 优化。生产栈（TRL、DeepSpeed-Chat、OpenRLHF、trlX）会加入 GAE 优势、价值头、完整 PPO 裁剪、多 GPU 分片和自适应 KL 控制，但这些无法容纳于单页。

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
\n此代码澄清了两点常被文字模糊之处：(1) PPO 所见“奖励”是 $r_\phi - \beta \cdot \text{KL}$，作为序列级标量优势嵌入；(2) $\pi_{\text{ref}}$ 完全冻结——处处设为 `requires_grad=False`。忽略任一，训练将漂移数小时后才被察觉。

## 11. 常见问题

**Q: 为何 RLHF 在足够示范下仍优于 SFT？**\nSFT 受限于示范作者——人类极少写出**最优**答案，通常仅是**不错**的答案。RLHF 允许模型探索超越示范分布，并对自身样本排序。且比较成本低于示范，单位预算可获更多信号。

**Q: RLAIF 多代后会导致模型坍缩吗？**\n当前证据（1–2 代）未见明显退化。但每轮自蒸馏均增加风险。缓解措施：持续混入新鲜人类数据、轮换标注模型、定期用留出人类黄金数据校准。

**Q: 奖励操控可彻底解决吗？**\n不可。这是优化度量的结构性后果，非特定 RM 缺陷。实用防御（KL 锚点、RM 集成、周期重标注、长度惩罚、宪法过滤）可控制损害但无法根除。应视其为需长期维护的工程问题，而非一次性修复。

**Q: RLHF 成本相较预训练如何？**\n约为预训练算力的 1–10%，主要开销在 RM 训练与 PPO 采样。DPO 将成本降至接近第二次 SFT——其算力优势确实显著。

**Q: 如何处理 helpfulness 与 harmlessness 等冲突目标？**\n实践中三种模式：(a) 为各目标训练**独立 RM**，通过学习或手动加权组合；(b) 用 **Constitutional AI** 将硬约束编码为自然语言规则；(c) 提供**用户可控偏好权重**（如“医疗建议更谨慎”）。三者常共存于生产栈。

**Q: 何时选 PPO 而非 DPO？**\n若偏好数据大而干净、追求快速迭代、关注 wall-clock 时间，选 DPO。若拥有高质量 RM 需保留在回路中、需在线探索（多步推理/工具使用）、或计划在训练中融入安全约束与宪法规则，选 PPO。

**Q: 与第 7 部分逆强化学习有何关联？**\n直接相关。RLHF 在结构上即逆强化学习，仅做两处简化：用成对偏好替代完整示范（Bradley-Terry 替代 MaxEnt IRL），以 PPO 为前向 RL 步骤。奖励模型即 IRL 输出；PPO 阶段则是标准的“用恢复奖励训练新策略”步骤。

## 参考文献

- **Bradley & Terry (1952).** Rank Analysis of Incomplete Block Designs. *Biometrika*. — 偏好似然起源。
- **Christiano et al. (2017).** Deep Reinforcement Learning from Human Preferences. *NeurIPS*. — 首篇现代偏好-RL 论文。
- **Stiennon et al. (2020).** Learning to Summarize with Human Feedback. *NeurIPS*. — RLHF 于摘要任务，InstructGPT 蓝图。
- **Ouyang et al. (2022).** Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*.
- **Bai et al. (2022a).** Training a Helpful and Harmless Assistant with RLHF. *Anthropic*.
- **Bai et al. (2022b).** Constitutional AI: Harmlessness from AI Feedback. *Anthropic*.
- **Gao et al. (2023).** Scaling Laws for Reward Model Overoptimization. *ICML*. — Goodhart 曲线论文。
- **Rafailov et al. (2023).** Direct Preference Optimization. *NeurIPS*.
- **Lee et al. (2023).** RLAIF: Scaling RL from Human Feedback with AI Feedback.
- **Skalse et al. (2022).** Defining and Characterizing Reward Hacking. *NeurIPS*.
- **Brohan et al. (2023).** RT-2: Vision-Language-Action Models. *Google DeepMind*.

## 系列总结
\n这是第十二篇，亦是终章。系列始于马尔可夫决策过程与简易 GridWorld，终于构建 ChatGPT 与 Claude 的对齐技术栈。一路走来，我们构建了：

- **基础** —— MDP、Bellman 方程、价值迭代（[第 1 部分](/zh/reinforcement-learning/01-基础与核心概念/)）
- **基于值的方法** —— Q-learning、DQN、double/dueling/distributional（[第 2 部分](/zh/reinforcement-learning/02-q-learning与深度q网络)）
- **策略梯度与 Actor-Critic**（[第 3 部分](/zh/reinforcement-learning/03-policy-gradient与actor-critic方法)）
- **探索与内在动机**（[第 4 部分](/zh/reinforcement-learning/04-探索策略与好奇心驱动学习/)）
- **基于模型的 RL 与世界模型**（[第 5 部分](/zh/reinforcement-learning/05-model-based强化学习与世界模型)）
- **PPO 与 TRPO**（[第 6 部分](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化)）—— 使 RLHF 成为可能的算法
- **模仿学习与逆强化学习**（[第 7 部分](/zh/reinforcement-learning/07-模仿学习与逆强化学习/)）—— 偏好学习的理论源头
- **AlphaGo 与 MCTS**（[第 8 部分](/zh/reinforcement-learning/08-alphago与蒙特卡洛树搜索)）
- **多智能体 RL**（[第 9 部分](/zh/reinforcement-learning/09-多智能体强化学习/)）
- **离线 RL**（[第 10 部分](/zh/reinforcement-learning/10-离线强化学习/)）
- **层次化与元 RL**（[第 11 部分](/zh/reinforcement-learning/11-层次化强化学习与元学习/)）
- **RLHF 与大语言模型对齐**（本部分）
\n贯穿始终的主线：**强化学习是从后果中学习的科学**。无论后果是游戏分数、物理模拟器、人类成对偏好，还是宪法式自批评，相同的 Bellman 回溯、探索-利用权衡与信任域直觉始终浮现。未来十年，RL 将聚焦于打通训练时与推理时搜索的闭环、连接数字智能体与具身智能体、弥合人类反馈与日益自主的自我改进之间的鸿沟。这十二篇文章中的数学工具，正是理解这一切的关键。

- **上一篇**：[第 11 部分 —— 层次化强化学习与元学习](/zh/reinforcement-learning/11-层次化强化学习与元学习/)
- **系列完结！** [查看 RL 系列全部 12 篇](/zh/categories/强化学习/)
