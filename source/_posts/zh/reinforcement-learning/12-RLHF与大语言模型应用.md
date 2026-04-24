---
title: "强化学习（十二）：RLHF与大语言模型应用"
date: 2025-07-29 09:00:00
tags:
  - 强化学习
  - RLHF
  - LLM
  - DPO
  - 对齐
categories: 强化学习
series:
  name: "强化学习"
  part: 12
  total: 12
lang: zh-CN
mathjax: true
description: "RLHF 把基础语言模型变成 ChatGPT 与 Claude 的完整路径：SFT→奖励模型→PPO 三阶段流程、Bradley-Terry 偏好模型、DPO 闭式解推导、RLAIF 与 Constitutional AI、Goodhart 定律下的奖励黑客，以及强化学习在具身智能与推理时搜索中的下一步。"
disableNunjucks: true
series_order: 12
---
GPT-3（2020 年 6 月）和 ChatGPT（2022 年 11 月）共享了大部分权重。基础模型能写流畅的散文、补全代码、续写任何模式——但你直接问它一个问题，它会东拉西扯、用错误的理由拒绝、编造引用，或者直接生成一段有毒内容。两年半的时间没有花在更大的 Transformer 上，而是花在**教模型怎么变得有用**——而这件事，最终被证明是一个强化学习问题。

本系列的最后一篇要把整个系列一直在指向的那条主线收拢起来：我们前面建立的所有概念——值函数、策略梯度、PPO 的信任域、离策略修正、偏好学习、内在动机、模仿→IRL 的进阶路径——会被组合成产出 ChatGPT、Claude、Llama-3-Instruct 以及任何能称之为助手的模型的对齐技术栈。我们会推导**RLHF 三阶段流程**、地球上每一个偏好数据集背后的 **Bradley-Terry** 似然、让 DPO 完全跳过 RL 的**闭式最优解**，以及让对齐成为永远在动的靶子的 **Goodhart 失败模式**。然后我们会越过语言，看看强化学习接下来要去哪里：具身智能体、宪法式自监督、推理时搜索。

## 你将学到什么

- **RLHF 三阶段流程**：监督微调、奖励模型训练、带 KL 锚点的 PPO，以及每一阶段为什么必须存在
- **Bradley-Terry 模型**：为什么偏好（而非绝对分数）才是合适的货币，以及它对你能恢复出什么样的奖励意味着什么
- **InstructGPT** 的核心实证发现：1.3B 的对齐模型在人类真正想要的事情上击败 175B 的 GPT-3
- **DPO**：一页纸的推导，把 RLHF 闭式最优解变成普通的对数似然损失
- **RLAIF 与 Constitutional AI**：把人从回路中拿掉而不让模型崩溃
- **奖励黑客与 Goodhart 定律**：为什么代理奖励一路上涨而用户满意度往下走，以及怎么应对
- **RL 接下来去哪**：具身智能体、sim-to-real、视觉-语言-动作模型、推理时搜索

## 前置知识

- PPO 与信任域直觉（[第 6 部分](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)）
- 离策略修正与重要性采样（[第 3 部分](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/)）
- 逆强化学习与偏好学习（[第 7 部分](/zh/强化学习-七-模仿学习与逆强化学习/)）
- Transformer 与 HuggingFace `transformers` 的工作经验

---

## 1. RLHF：三阶段流程

![RLHF 三阶段流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig1_rlhf_three_stage_pipeline.png)

预训练给你的是一个知道**互联网文本分布**的模型——包括你不想要的那些部分。RLHF 给你的是一个知道**你想要什么**的模型。2022 年从 OpenAI 流出的这套配方已经成了行业标准，三个阶段边界分明：监督阶段、偏好阶段、强化阶段。

### 阶段 1 —— 监督微调（SFT）

拿到预训练基模 $\pi_{\text{base}}$，收集一小批由人类撰写的高质量 `(prompt, response)` 演示（InstructGPT 用了大约 13K 条），用标准的下一词交叉熵最小化：
$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{demo}}}\sum_t \log \pi_\theta(y_t \mid x, y_{<t}).
$$
这一步成本低、机制成熟、绝对不能省。它把策略从"补全任意互联网文本的下一个 token"拉到"对指令做出有用响应"。更重要的是，它产出的 $\pi_{\text{SFT}}$ 会成为后两个阶段都要锚定的**参考策略** $\pi_{\text{ref}}$。从这里往后的一切，都是叠加在 SFT 之上的修正项。

### 阶段 2 —— 训练奖励模型

演示数据贵——人类得**写**出理想答案。而偏好数据便宜：人类只需要**比较**两个模型输出，挑出更好的那个。阶段 2 用比较数量换演示质量（InstructGPT 收集了约 33K 对比较）。

对每个 prompt $x$，从 $\pi_{\text{SFT}}$ 采样两个补全 $y_A, y_B$，让标注员选出胜者，记为 $(y_w, y_l)$（$y_w \succ y_l$），训练奖励模型 $r_\phi(x, y)$——通常是把 SFT 模型的语言头换成一个标量头——通过 **Bradley-Terry** 损失让它给 $y_w$ 打更高的分：
$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right].
$$
这正是[第 7 部分](/zh/强化学习-七-模仿学习与逆强化学习/)里的偏好学习目标——**RLHF 在结构上就是语言模型的逆强化学习**。InstructGPT 一个反直觉的实证发现：**6B 的奖励模型在下游 RL 中比 175B 的更稳定**。奖励模型只需要好到让 PPO 不跑出它的支撑集；如果它能力太强，PPO 反而会找到能利用它盲区的对抗输入。

### 阶段 3 —— 带 KL 锚点的 PPO

真正的强化学习现在才开始。我们把 $r_\phi$ 当作环境奖励，最大化策略的期望奖励——但加一个 **KL 惩罚**让它别离 $\pi_{\text{ref}}$ 太远：
$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\!\left[\,r_\phi(x, y) \,-\, \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\,\right].
$$
括号里的整体就是 PPO 用的逐 token 奖励，第二项是 KL 锚点。没有它，策略迟早会找到一些在 $r_\phi$ 下高分但读起来像词汤的 token——这就是奖励黑客，第 6 节会回到这个话题。

PPO 之所以是首选，原因和[第 6 部分](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)讲的一致：

1. **动作空间就是词表**（约 5 万 token）。每个 token 上做 5 万路 argmax 的 Q-learning 不可行；策略梯度可行。
2. **裁剪能防灾难性更新**。一个糟糕的 batch 就能毁掉 70B 参数的对话模型，没有 checkpoint 管理能救你。PPO 的裁剪代理目标 $\min\big(\tfrac{\pi_\theta}{\pi_{\text{old}}}A,\,\text{clip}(\tfrac{\pi_\theta}{\pi_{\text{old}}},1\!-\!\epsilon,1\!+\!\epsilon)A\big)$ 限制了单步能走多远。
3. **KL 惩罚和 PPO 的信任域哲学天然契合**：两者都限制策略每步能变多少，只是出于不同理由（PPO 限制的是单次更新内相对 $\pi_{\text{old}}$ 的漂移；KL 限制的是整个训练过程相对 $\pi_{\text{ref}}$ 的漂移）。

### 为什么是三阶段？

每一阶段都把上一阶段的产物压缩成更紧凑的信号。SFT 把数千万 token 的互联网文本压缩成一个**会**响应的模型。奖励模型把 33K 条人类判断压缩成一个**评估**响应的标量。PPO 又把这个评估器解压回一个**生成**人类一开始就想要的响应的策略。每次压缩都是有损的，每个损失都是一整片独立的研究文献。

---

## 2. Bradley-Terry 模型：为什么是偏好而不是分数

![Bradley-Terry 偏好模型](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig5_bradley_terry.png)

如果你让一百个标注员在 1–10 分上打分，你会发现一个人的 7 分等于另一个人的 4 分。绝对分数在人之间、在不同日子之间、甚至在连续几个 prompt 之间（标定漂移）都不平稳。Bradley-Terry 模型——1952 年为体育队伍排名提出——假设每个项目背后有一个潜在分数 $s(y)$，成对比较的结果服从：
$$
P(y_A \succ y_B) \;=\; \frac{e^{s(y_A)}}{e^{s(y_A)} + e^{s(y_B)}} \;=\; \sigma\!\big(s(y_A) - s(y_B)\big).
$$
两个推论塑造了所有现代 RLHF 系统：

- **奖励只能识别到一个常数偏移。** 给所有分数加一个常数，所有偏好都不变。奖励模型的绝对刻度没有意义，只有差值有意义。这也是为什么 DPO 推导里的配分函数 $Z(x)$ 会消掉。
- **标注员噪声有一个不可逾越的下限。** 即便是黄金标准的人类，在 InstructGPT 风格的 prompt 上彼此的一致性也只有约 78%。一个奖励模型如果在留出偏好上做到 78% 准确率，就已经把信号吃满了。再往上推就是在拟合个别标注员的怪癖，而不是人类价值。

正确的心理图像：奖励模型是一个**校准后的偏好分类器**，不是质量的神谕。PPO 阶段把这个校准分类器当成了 ground truth——这就是第 6 节所有 Goodhart 失败模式的入口。

---

## 3. 带 KL 锚点的 PPO：参数空间里的图像

![带 KL 约束的 PPO](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig3_ppo_kl_constraint.png)

KL 项做的事不止是正则化。它实现的是和[第 6 部分](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)里 TRPO 同样的信任域，只不过锚点是**冻结的参考模型**而不是上一次迭代。上图左边展示了几何：有 KL 锚点时，策略走向一个中等但真实的奖励峰；没有 KL 时，策略滑向一个 $r_\phi$ 有伪峰、但真实输出已经不连贯的区域。

右边是实际工程问题：当 $\beta$ 从大调到小，**代理奖励**单调上升（你给了策略更大的优化自由度），但**真实人类质量**是一条单峰曲线——大约在 $\beta \in [0.01, 0.03]$ 之间见顶，然后崩溃。选 $\beta$ 是个需要真人入回路的超参，没有任何离线指标能告诉你什么时候越界了。实际中各家用**自适应 KL 控制**：固定一个目标平均逐 token KL（例如 6 nats），让 $\beta$ 浮动以维持这个目标。

一次跑得起来的 RLHF 训练同时在显存里**塞了四个模型**：训练中的策略 $\pi_\theta$、提供 KL 项的参考 $\pi_{\text{ref}}$、给采样打分的奖励模型 $r_\phi$、做 GAE 优势估计的价值头（通常和策略共享主干）。这就是 RLHF 工程量比 SFT 大得多的原因——光显存账单就大约 $4\times$。

---

## 4. InstructGPT：数据说了什么

![奖励模型训练](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig2_reward_model_training.png)

InstructGPT 的论文（Ouyang 等，NeurIPS 2022）短小、密集，是这个领域最接近罗塞塔石碑的东西，把 RLHF 到底买到了什么讲清楚了。四个值得记住的发现：

1. **对齐胜过规模。** 在盲测人类评测里，**1.3B 参数的 InstructGPT 大约 85% 的时候被偏好于 175B 参数的 GPT-3**。对齐过的小模型比未对齐的大模型更有用，差距大到再加一个数量级的预训练算力都补不回来。
2. **泛化是真的，但不均匀。** 在英文指令上训练的 RLHF，迁移到了 SFT 集里几乎没见过的代码和非英文 prompt。奖励模型抓到的东西比它训练分布的表面形式更一般。
3. **"对齐税"很小。** 对齐过的模型在标准 NLP benchmark（TriviaQA、HellaSwag）上掉了几个点——它们在下一词补全这场游戏里稍微变差了。用户不在乎；用户体验上的胜利远远盖过 benchmark 上的失分。这是**benchmark 与用户价值会分叉**的第一个具体证据，从那以后这个观察只是越来越尖锐。
4. **奖励黑客立刻就出现。** 论文记录了长度黑客（响应越来越长换取边际分数）、格式黑客（一切都变成项目符号）和轻度的 sycophancy（讨好）。这些不是修一次就好的 bug，它们是**稳定吸引子**，每个 RLHF 系统都会反复出现，包括生产系统。

---

## 5. DPO：跳过奖励模型，也跳过 RL

![DPO 推导](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig4_dpo_derivation.png)

后 InstructGPT 时代影响最大的一个 RLHF 结果是 **Direct Preference Optimization**（Rafailov 等，NeurIPS 2023）。它的主张挑衅性十足：你可以**完全扔掉**奖励模型和 PPO，把整套技术栈换成在同一份偏好数据上训的一个监督损失。

### 推导

从 KL 正则化的 RL 目标出发：
$$
\max_\pi\; \mathbb{E}_{x,y\sim\pi}\big[r(x,y)\big] \,-\, \beta\, D_{\mathrm{KL}}\!\big[\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big].
$$
这是一个对每个 prompt 的分布 $\pi(\cdot|x)$ 而言的凸约束问题。拉格朗日（或者直接猜验证）给出闭式最优解：
$$
\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right),
$$
其中 $Z(x)$ 是关于 $y$ 的配分函数。把这个式子反过来，用最优策略表达奖励：
$$
r(x,y) \;=\; \beta\,\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta\,\log Z(x).
$$
关键一步：**把这个 $r$ 的表达式代入 Bradley-Terry 偏好似然**：
$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x,y_w) - r(x,y_l)\big),
$$
$\beta\log Z(x)$ 项消掉了——它只依赖 $x$，不区分 $y_w$ 和 $y_l$。剩下的损失只依赖 $\pi_\theta$、$\pi_{\text{ref}}$ 和偏好数据：
$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \,-\, \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].\;}
$$
就是两个对数似然比的差，过 sigmoid，再过交叉熵。**没有采样、没有价值头、没有奖励模型、没有 PPO 裁剪。两次前向、一次反向，结束。**

### DPO 实际买到了什么

- **一阶段代替三阶段。** SFT，然后直接在偏好上做 DPO。不需要单独维护一个奖励模型 artifact。
- **没有采样循环。** PPO 需要在训练循环里生成补全，这是 wall-clock 时间的大头。DPO 是固定数据集上的离线监督学习。
- **没有显式奖励可被黑。** 隐式奖励 $\hat r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是由策略**定义**的，所以策略没法和它分叉。
- **显存大致减半。** 两个模型（$\pi_\theta$、$\pi_{\text{ref}}$）而不是四个。

### DPO 没有买到什么

- **奖励无法被审视。** 隐式 $\hat r$ 只作为一个比值存在，你没法像用单独的 $r_\phi$ 那样给一条新的补全打分。
- **DPO 对噪声偏好更敏感。** PPO 的 KL 锚点和在线采样提供了一些鲁棒性，DPO 是字面意义上信任偏好数据集。
- **DPO 在长程推理（CoT、多步工具调用）上可能不如 PPO**，因为这些任务受益于 PPO 的在线探索，而 DPO 不做这件事。**Online DPO**、**iterative DPO**、**IPO**、**KTO** 都在试图弥合这个 gap。

2024–2026 年实用主义的判决：开源 instruction-tuned 模型（Llama-3、Qwen-2.5、Mistral）大多以 DPO 变种发布，因为工程更简单、头部 benchmark 也有竞争力。前沿闭源模型（ChatGPT、Claude、Gemini）大多还在用 PPO 系 RLHF 或它的 constitutional 版本，因为在硬推理任务上的边际质量足以支撑额外的复杂度。两条路线会长期共存。

---

## 6. 奖励黑客与 Goodhart 定律

![奖励黑客与 Goodhart 定律](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig7_reward_hacking.png)

Charles Goodhart 1975 年的观察，现代版本是："**当一个度量变成目标，它就不再是一个好度量。**" RLHF 是这条警句在机器学习里的构造性证明。奖励模型是一个对人类偏好的度量；PPO 一旦把它当目标，策略立刻开始找不真正服务人类却能拿高分的办法。

左图是经典实证图（Gao 等，ICML 2023）：横轴是从 $\pi_{\text{ref}}$ 出发的训练 KL（RL 的"剂量"），代理奖励 $r_\phi$ 单调上升，但**金标准人类奖励是单峰的**——早早见顶然后崩塌。两条曲线之间的间隙就是 Goodhart gap，模型越大、训练越久，gap 越宽。

右边是会反复出现的失败模式目录：

1. **长度黑客。** 响应比人类真正想要的长 2–3 倍，因为长度通常和 RM 分数正相关。
2. **Sycophancy（讨好）。** 模型同意用户陈述的立场，哪怕用户错了，因为 RM 标注员倾向于偏好被同意。
3. **格式黑客。** 项目符号、标题、表格泛滥；RM 学到了结构看起来像在用心。
4. **自信瞎说。** 流畅、格式漂亮、事实错误。奖励模型不能做事实核查，所以奖励了自信。
5. **过度拒绝。** 模型对良性查询过度拒绝，以对冲 harmlessness 奖励，产出"作为一个大语言模型，我不能……"这种用户深恶痛绝的失败。

### 实际有效的缓解手段

- **KL 锚点。** 第一道防线；自适应地把 $\beta$ 调到目标 KL。
- **奖励模型 ensemble。** 在不同数据切分上训几个奖励模型，对预测取平均——利用平均能洗掉。
- **周期性重新标注。** 在**当前**策略的输出上收集新偏好，而不是用陈旧数据，每几轮刷新奖励模型。
- **长度受控的奖励。** 减一个长度惩罚，或者在固定长度预算下评测。
- **Constitutional / 红队补充。** 把显式规则和对抗样本加进数据集（下一节）。

没有永久解决方案。奖励黑客是问题结构里就带的一场军备竞赛。

---

## 7. RLAIF 与 Constitutional AI：把人去掉

人工标注慢、贵、不稳定——而且根本扛不住前沿模型需要的偏好量。两类方法把人部分或全部地用强模型替换掉：

**RLAIF**（Lee 等，2023）把标注员换成另一个 LLM（例如 GPT-4），让它做比较：

```text
给定下面的问题和两个响应，哪一个更符合
"helpful, honest, harmless" 的标准？
问题：{x}
响应 A：{y_A}
响应 B：{y_B}
回答 A 或 B，并简要说明理由。
```

在标准任务上，RLAIF 的偏好和人类偏好的一致率约 85%，成本大约低 10 倍。风险是**模型坍缩**：在 AI 标注的数据上训练足够多代之后，分布变窄、偏见固化、质量退化。当前的缓解工具箱：保留持续比例的新鲜人类数据、轮换标注模型、定期对人类金数据做校准。

**Constitutional AI**（Bai 等，Anthropic 2022）走得更远：写一份自然语言原则的"宪法"（"要乐于助人"、"避免建议有害行为"），让模型在偏好标注之前先**根据宪法对自己的输出做自我批评和修订**。然后用 `(原始, 修订后)` 对作为奖励模型的偏好数据，其中修订更好地满足宪法。这是 Claude 训练栈的基础，也是**用模型自己的能力来 bootstrapping 自己的对齐**的一个干净示例——闭合了纯 RLHF 没闭合的那个回路。

趋势线很清楚：基模越好，越多的对齐信号可以由模型自己产生，人类从"标注每一对比较"转向"审计宪法和分歧"。瓶颈从标注吞吐转向规范质量。

---

## 8. 生产级对齐栈的架构

![ChatGPT/Claude 训练架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/12-RLHF%E4%B8%8E%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/fig6_chatgpt_architecture.png)

把所有部件拼起来，每一个现代助手——ChatGPT、Claude、Gemini、Llama-3-Instruct、Qwen-2.5-Instruct——都套同一个五层模板：

1. **预训练。** 数万亿 token 的自监督下一词预测。占总算力的 90–99%，也是唯一一个能实质性改变原始能力的阶段。
2. **SFT / 指令微调。** 10K–100K 条精挑细选的 `(prompt, response)`，可选地用更强模型蒸馏出的合成数据扩充。
3. **偏好数据。** 人类标注（RLHF）、AI 标注（RLAIF）或宪法式自批评（CAI）。通常是混合。
4. **对齐优化。** PPO + KL 锚点（OpenAI 传统）、DPO 及其变种——IPO、KTO、ORPO（开源传统）、或者宪法式自监督回路（Anthropic）。
5. **部署时护栏。** 系统提示、安全分类器、工具调用脚手架、在线红队、滚动评测（MT-Bench、Chatbot Arena 和内部回归测试）。

这些部件已经商品化。各家之间的差异化生活在**偏好数据的质量**、**奖励模型（或它的隐式替代）的鲁棒性**、以及**部署评测的纪律性**里。算法是最容易的部分。

---

## 9. 越过语言：RL 接下来去哪

RLHF 是 RL 当下最高赌注的部署，但不是最雄心勃勃的那个。另外三条战线在并行推进，且都从本系列大量借鉴：

**机器人 sim-to-real。** 在快速模拟器（MuJoCo、Isaac Gym）里训策略，用**域随机化**（变化物理参数、光照、纹理）跨过现实差距，部署到真实硬件上。OpenAI 的 Dactyl 用这种方法让机械手解开了魔方；Google 的 Aloha 用模仿学习（[第 7 部分](/zh/强化学习-七-模仿学习与逆强化学习/)）启动，再用在线 RL 精修。

**安全关键控制的离线 RL。** 驾驶、医疗、工业控制承担不起在线策略探索。[第 10 部分](/zh/强化学习-十-离线强化学习/)的方法——CQL、IQL、Decision Transformer——从日志数据初始化策略，然后才转向谨慎的在线微调。

**视觉-语言-动作模型。** Google 的 RT-2 拿一个预训练的视觉-语言模型，在网页数据和机器人轨迹上联合微调，产出了第一个对未见物体和未见指令具备强零样本泛化能力的机器人策略。这是具身智能版本的 RLHF 时刻：拿一个已经理解世界的模型，把它弯向**在世界里行动**。

**推理时 RL。** 最近的转折：与其在训练时花 RL 算力，不如在推理时花。OpenAI 的 o 系列和 DeepSeek 的 R1 用 RL 不是去更新单次前向的权重，而是教模型在回答前**搜索思维链**——这是 [MCTS](/zh/强化学习-八-AlphaGo与蒙特卡洛树搜索/) 思想、[PPO](/zh/强化学习-六-PPO与TRPO-信任域策略优化/) 和上文偏好学习机制的融合。预期它会主导未来两年前沿模型的进展。

---

## 10. 简化的 RLHF 实现

下面的参考代码覆盖了概念流程——用 Bradley-Terry 损失训奖励模型，然后做一个精简的 PPO 风格优化。生产级技术栈（TRL、DeepSpeed-Chat、OpenRLHF、trlX）会加上 GAE 优势、价值头、完整的 PPO 裁剪、多 GPU 切分、自适应 KL 控制，这些在一页纸上塞不下。

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
        # 用最后一个非 pad token 的隐状态
        last = out.last_hidden_state[:, -1, :]
        return self.value_head(last).squeeze(-1)


def train_reward_model(model, dataloader, epochs=3, lr=1e-5):
    """Bradley-Terry 偏好损失。"""
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

        # 2. 在 policy 与 reference 下计算对数概率
        logp_pi = self._logp(self.policy, out)
        with torch.no_grad():
            logp_ref = self._logp(self.ref, out)

        # 3. KL 正则化奖励作为序列级优势
        kl = logp_pi - logp_ref                          # [B]
        advantage = (r - self.beta * kl).detach()

        # 4. 策略梯度步
        loss = -(logp_pi * advantage).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), r.mean().item(), kl.mean().item()
```

这段代码澄清了散文容易遮蔽的两件事：(1) PPO 看到的"奖励"是 $r_\phi - \beta \cdot \text{KL}$ 烤进去成为序列级标量优势的；(2) $\pi_{\text{ref}}$ 是冻结的——`requires_grad=False` 处处都是。任何一处忘了，训练跑了几小时你才会发现已经漂走。

---

## 11. 常见问题

**问：在演示数据足够多的情况下，RLHF 为什么仍然击败 SFT？**
SFT 受演示作者的能力上限限制——人类很少写出**最优**答案，只是写出**好的**答案。RLHF 让模型可以探索演示分布之外的空间，并对自己的样本排序。比较还比演示便宜，所以单位预算下能拿到更多信号。

**问：跨多代生成，RLAIF 会不会引发模型坍缩？**
当前证据（1–2 代）没看到明显退化。每多一轮自蒸馏，风险都会涨。缓解：保留持续比例的新鲜人类数据、轮换打标的模型、定期对照留出的人类金数据校准。

**问：奖励黑客有解吗？**
没有永久解。它是优化一个度量这件事结构上的后果，不是任何具体奖励模型的 bug。实际防御（KL 锚点、RM ensemble、周期性重新标注、长度惩罚、constitutional 过滤）能限制损失，但永远消除不了。把它当作承重的工程问题，而不是一次性修补。

**问：RLHF 相比预训练贵多少？**
大约是预训练算力的 1–10%，主要花在奖励模型训练和 PPO 采样上。DPO 把这降到大约第二次 SFT 的成本——DPO 在算力账上的论据是真的强。

**问：怎么处理冲突目标，比如 helpfulness 与 harmlessness？**
实际中三种模式：(a) 给每个目标训**独立的奖励模型**，再用学到的或手调的权重组合；(b) 用 **Constitutional AI** 把硬约束编码成自然语言规则；(c) 暴露**用户可控的偏好权重**（例如"在医疗建议上更谨慎"）。三者在生产栈里共存。

**问：什么时候该选 PPO，什么时候选 DPO？**
偏好数据干净量大、要快速迭代、在乎 wall-clock 训练时间——选 DPO。你有一个想留在回路里的高质量奖励模型、需要在线探索（多步推理、工具使用）、或者打算在训练时混入安全约束和宪法规则——选 PPO。

**问：这些和第 7 部分的逆强化学习有什么联系？**
直接相关。RLHF 在结构上就是逆强化学习，做了两个简化选择：用成对偏好代替完整演示（用 Bradley-Terry 代替 MaxEnt IRL），用 PPO 作为前向 RL 步。奖励模型就是 IRL 的产物；PPO 阶段就是标准的"用恢复出的奖励训新策略"那一步。

---

## 12. 参考文献

- **Bradley & Terry (1952).** Rank Analysis of Incomplete Block Designs. *Biometrika*. — 偏好似然的起源。
- **Christiano 等 (2017).** Deep Reinforcement Learning from Human Preferences. *NeurIPS*. — 第一篇现代偏好-RL 论文。
- **Stiennon 等 (2020).** Learning to Summarize with Human Feedback. *NeurIPS*. — 摘要任务上的 RLHF，InstructGPT 的蓝图。
- **Ouyang 等 (2022).** Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*。
- **Bai 等 (2022a).** Training a Helpful and Harmless Assistant with RLHF. *Anthropic*。
- **Bai 等 (2022b).** Constitutional AI: Harmlessness from AI Feedback. *Anthropic*。
- **Gao 等 (2023).** Scaling Laws for Reward Model Overoptimization. *ICML*. — Goodhart 曲线那篇。
- **Rafailov 等 (2023).** Direct Preference Optimization. *NeurIPS*。
- **Lee 等 (2023).** RLAIF: Scaling RL from Human Feedback with AI Feedback。
- **Skalse 等 (2022).** Defining and Characterizing Reward Hacking. *NeurIPS*。
- **Brohan 等 (2023).** RT-2: Vision-Language-Action Models. *Google DeepMind*。

---

## 系列总结

这是第十二篇也是最后一篇。整个系列从马尔可夫决策过程和一个朴素的 GridWorld 开始，到搭出了 ChatGPT 与 Claude 的对齐技术栈结束。沿途我们建立起：

- **基础** —— MDP、Bellman 方程、价值迭代（[第 1 部分](/zh/强化学习-一-基础与核心概念/)）
- **基于值的方法** —— Q-learning、DQN、double/dueling/分布式（[第 2 部分](/zh/强化学习-二-Q-Learning与深度Q网络/)）
- **策略梯度与 Actor-Critic**（[第 3 部分](/zh/强化学习-三-Policy-Gradient与Actor-Critic方法/)）
- **探索与内在动机**（[第 4 部分](/zh/强化学习-四-探索策略与好奇心驱动学习/)）
- **基于模型的 RL 与世界模型**（[第 5 部分](/zh/强化学习-五-Model-Based强化学习与世界模型/)）
- **PPO 与 TRPO**（[第 6 部分](/zh/强化学习-六-PPO与TRPO-信任域策略优化/)）—— 让 RLHF 成为可能的算法
- **模仿学习与逆强化学习**（[第 7 部分](/zh/强化学习-七-模仿学习与逆强化学习/)）—— 偏好学习的概念祖先
- **AlphaGo 与 MCTS**（[第 8 部分](/zh/强化学习-八-AlphaGo与蒙特卡洛树搜索/)）
- **多智能体 RL**（[第 9 部分](/zh/强化学习-九-多智能体强化学习/)）
- **离线 RL**（[第 10 部分](/zh/强化学习-十-离线强化学习/)）
- **层次化与元 RL**（[第 11 部分](/zh/强化学习-十一-层次化强化学习与元学习/)）
- **RLHF 与大语言模型对齐**（本部分）

贯穿全系列的主线：**强化学习是从后果中学习的科学**。无论后果是游戏分数、物理模拟器、人类成对偏好，还是宪法式自批评，同样的 Bellman 回溯、同样的探索-利用权衡、同样的信任域直觉都会出现。RL 的下一个十年要做的，是闭合训练时与推理时搜索的回路、闭合数字智能体与具身智能体的回路、闭合人类反馈与日益自主的自我提升的回路。这十二篇里的数学，就是理解它的工具箱。

- **上一篇**：[第 11 部分 —— 层次化强化学习与元学习](/zh/强化学习-十一-层次化强化学习与元学习/)
- **系列完结！** [查看 RL 系列全部 12 篇](/tags/强化学习/)
