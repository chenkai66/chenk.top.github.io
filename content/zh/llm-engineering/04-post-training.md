---
title: "大模型工程（四）：SFT、DPO 与 RLHF"
date: 2026-03-30 09:00:00
tags:
  - LLM
  - post-training
  - SFT
  - DPO
  - RLHF
  - LoRA
categories: 大模型工程
series: llm-engineering
series_order: 4
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "SFT、DPO、RLHF、RLAIF 各自具体在优化什么，奖励模型在哪里失败，KL 约束的作用，LoRA vs 全量微调那场争论，以及 2026 年生产里实际跑的 post-training 配方。"
translationKey: "llm-engineering-4"
---
预训练的基座模型只会续写文本，而听懂指令、拒绝有害请求、维持人设等工作则属于后训练阶段——这也是论文中宣称的效果与生产级模型之间差距最大的地方。本章将探讨各个后训练算法的具体优化内容、大多数奖励模型存在的问题，以及 2026 年真正有效的方案。

![LLM Engineering (4): Post-training — SFT, DPO, RLHF, RLAIF — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/illustration_1.png)

## 四阶段技术栈

![fig1: RLHF pipeline overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig1_rlhf_pipeline.png)

当前的 LLM 后训练流程大致分为以下四个步骤：

1. **SFT**（supervised fine-tuning）用于指令数据，教授模型响应格式和基本的指令遵循行为。
2. **偏好优化**（DPO, IPO, KTO, 或 RLHF），教授模型在两个有效回复中哪个更受人类（或代理）青睐。
3. **在线 RL**（RLHF, RLAIF, 或 RLVR — 可验证奖励），对奖励模型或程序检查器进行更激进的调优。对于非推理模型，这一步是可选的，并且越来越常被跳过。
4. **专项阶段**：包括工具使用 SFT、长上下文 SFT、安全红队测试和宪法 AI 过滤。

OpenAI、Anthropic 和 Google 仍在采用接近 “SFT → 偏好 DPO → RLHF/RLAIF” 的流程。DeepSeek-R1 [DeepSeek-AI, 2025] 和 o1 系列模型引入了 **RLVR**（RL with verifiable rewards — 适用于数学/代码等可通过程序检查正确性的场景）作为主要信号。这是 2024-2025 年最重要的后训练变革。

师承关系非常重要。第一篇可信的 "RL from human feedback" 论文是 [Christiano et al., 2017]，将偏好学习应用于 Atari 和连续控制。[Stiennon et al., 2020] 将其用于摘要生成。[Ouyang et al., 2022] (InstructGPT) 是首个通过 SFT + RLHF 进行端到端指令微调的 LLM，现代所有后训练技术栈均源自该论文。2023-2025 年的这一波浪潮（DPO, IPO, KTO, RLVR）本质上都是 InstructGPT 主题的变体。

## SFT：比大家以为的更重要

SFT 就是给模型展示 10 万到 100 万条指令-回复对，仅对回复部分进行 next-token prediction loss 的训练。这个 mask 很关键——你希望模型学会助手的回答，而不是预测用户的问题。

```python
# Loss masking for SFT
def sft_loss(logits, labels, response_mask):
    # logits: [B, T, V], labels: [B, T], response_mask: [B, T] in {0, 1}
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_mask   = response_mask[:, 1:].contiguous().float()
    loss_per_token = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
    ).view(shifted_labels.shape)
    return (loss_per_token * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)
```

SFT 的成败取决于两点：
**数据质量而非数量。** LIMA [Zhou et al., 2023] 证明了 1000 条精心策划的例子可以媲美 5 万条普通数据。Tulu-3 混合集 [Lambert et al., 2024] (AllenAI, 2024) 策划了约 20 万条数据。Qwen3 的 SFT 混合集接近 100 万，但经过了严格的质量过滤。高质量数据在超过 100K 后，收益曲线趋于平缓。

**格式一致性。** 如果一半 SFT 数据用 "Sure, I'll help!" 这种开场白，另一半不用，模型就会学会不一致地使用开场白。如果 SFT 数据用 Markdown 标题，但测试 prompt 是关于纯 prose 的，输出就会不匹配。预处理要积极，把格式标准化。

一个意想不到的失败模式是：过度训练短回复会导致模型难以生成长文本。模型会从训练数据中学习回复长度的条件分布。如果希望模型能够写出 2000 字的文章，SFT 混合集中至少需要 5-10% 的长示例。我们之前调整过一个 Qwen3-7B 微调版，始终无法生成超过 800 token 的输出，原因是 SFT 混合集中 Q&A 占比过高。

## SFT 数据来源与合成

2026 年生产级 SFT 数据实际从哪来：

- **公开混合集**： Tulu-3 (AllenAI, 939K 例子), OpenHermes-2.5 (100 万， GPT-4 输出混合), UltraChat (140 万过滤后的 ChatGPT 对话 [Ding et al., 2023]), Magpie [Xu et al., 2024] (chat-tuned 模型 self-prompt 生成的合成指令)。
- **领域特定**：从内部产品日志爬取（需 consent 和去除 PII），领域专家撰写（$30-100/条），或在领域特定 seed prompts 上从强教师模型蒸馏。
- **从强教师模型合成**：让 Claude 或 GPT-4 根据 seed 主题和 few-shot 例子生成 (instruction, response) 对。这是主力军 — 2026 年生产环境中大多数 SFT 数据都是合成的。

Magpie 技术 [Xu et al., 2024] 值得了解一下。 trick 在于：只给 chat-tuned 模型一个 `<|im_start|>user\n`，让它自己生成用户消息，然后让它（或另一个模型）生成回复。这样不需要 seed prompts 就能产出格式良好的指令数据。他们用这种方式生了 400 万例子，过滤到 20 万，质量匹敌人工 curated 混合集。 2025-2026 年大多数 SFT 数据的流水线里，多少都掺了点 Magpie 风格的合成数据。

“指令遵循 SFT” 和 “聊天 SFT” 的区别也值得搞清楚。早期数据集（Alpaca, Self-Instruct）是短用户指令配短回复。现代数据（Tulu-3, 源自 ShareGPT 的混合集）是多轮对话配长回复。只训单轮指令的模型在多轮对话里会崩（忽略 prior turns）；只训多轮的模型又跟不上简单命令。需要平衡混合。

## DPO：不需要奖励模型的偏好优化

![LLM Engineering (4): Post-training — SFT, DPO, RLHF, RLAIF — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/illustration_2.png)


![fig2: DPO vs PPO comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig2_dpo_vs_ppo.png)

经典 RLHF 配方（InstructGPT [Ouyang et al., 2022]）是：先在人类偏好上训一个奖励模型，然后用 PPO 对着那个奖励模型调优 policy。这招管用，但实现起来痛苦 — 你需要单独的奖励模型、 value head、 GAE、 advantage normalization、 KL penalties，而且训练不稳定。

**DPO (Direct Preference Optimization)**, [Rafailov et al., 2023]，直接砍掉了奖励模型。核心洞察：你可以推导出一个关于偏好的闭式 policy， resulting loss 其实就是 log-probability 比值上的 binary cross-entropy：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

其中 $y_w$ 是被选中的回复，$y_l$ 是被拒绝的，$\pi_{\text{ref}}$ 是冻结作为参考的 SFT 模型，$\beta$ 控制 policy 允许漂移多远。无需奖励模型，也无需 PPO；每个偏好对仅需一次前向与反向传播

```python
# DPO loss in 10 lines
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios  = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps   - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
```

DPO 在 2026 年占主导有两个原因。第一，它管用 — [HuggingFace alignment-handbook](https://github.com/huggingface/alignment-handbook) 报告 DPO 在 AlpacaEval 上匹敌 PPO，且在 cost-per-quality 上胜出。第二，它是纯前向传播的训练循环，能干净地集成进 FSDP/LoRA 栈。

你会踩到的坑：

- **β 调参很关键。** β 太低 (0.01) → policy 漂移，基座能力侵蚀。β 太高 (1.0) → 学不动。大多数生产运行设在 β=0.1 到 0.3。
- **参考模型漂移。** 如果你连续跑两次 DPO，你的 reference 变成了 DPO 后的模型，原始版本没了。保存好 SFT checkpoint，每次偏好传递都用它做 ref。
- **偏好数据质量。** 合成偏好（比如 “GPT-4 选了 A 而不是 B”）容易生成但包含教师偏差。至少在 20% 的数据里混入人类偏好，防止 collapse。

## DPO 推导细节：从 Bradley-Terry 到闭式解

DPO loss 不是凭空捏造的 — 它分三步推导出来。值得搞清楚，因为这显示了 DPO 对偏好数据的假设以及如何解读 β。

**第一步： Bradley-Terry 偏好模型。** 假设偏好由 latent reward $r(x, y)$ 生成，偏好 $y_w$ 胜过 $y_l$ 的概率由 logistic 给出：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

这是 [Bradley & Terry, 1952] 配对比较模型。起源是棋类评级，是将 pairwise 偏好转换为 scalar reward 的标准假设。

**第二步： KL 约束奖励最大化给出闭式最优 policy。** RLHF 目标是

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta \, \text{KL}(\pi \| \pi_{\text{ref}})$$

该目标下的最优 policy （取梯度并设为零）是

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

这是最大熵 RL 文献中的经典结果。解出 $r$：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**第三步：代入 Bradley-Terry，消去 $Z(x)$。** Bradley-Terry 只依赖 reward *差值*，所以 $Z(x)$ 项抵消：

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

在该模型下最大化观测偏好的 log-likelihood 就得到了 DPO loss。整个 RLHF 流水线坍缩为 log-prob 比值上的 binary cross-entropy。不用奖励模型。不用采样。不用 PPO。

β 的解释变得清晰：β 是 implicit reward 的逆温度。低 β (0.01) 意味着小 reward 对应大 policy  shifts — 模型从每个偏好中激进地学习。高 β (1.0) 意味着 policy 几乎无法从 $\pi_{\text{ref}}$ 移动 — 偏好几乎不影响模型。经验上的 sweet spot 是 0.1-0.3，因为这时 implicit reward 大到足以提供信息，又小到防止漂移。
## DPO variants: KTO, IPO, ORPO, SimPO

DPO 在 2024-2025 年衍生出了一堆变体，各自解决特定痛点：

**KTO (Kahneman-Tversky Optimization)**, [Ethayarajh et al., 2024], 不用成对偏好，改用绝对的“好/坏”标签。损失函数是不对称的（采用 Kahneman-Tversky 风格的价值函数：人对损失更敏感，所以惩罚坏回答的权重高于奖励好回答）。当你只有大量非成对反馈（生产环境的点赞/踩）而没有成对比较时， KTO 就能派上用场。实证表明，有成对数据时它和 DPO 持平，只有非成对数据时则优于 DPO。

**IPO (Identity Preference Optimization)**, [Azar et al., 2023], 把 sigmoid 损失换成 MSE 损失，防止 DPO 在噪声偏好上过拟合。 DPO 的 sigmoid 在策略变得非常自信时会饱和，这意味着一旦策略“赢”了某对数据，损失就几乎没信号了。 IPO 的 MSE 能持续提供梯度。实践中 IPO 比 DPO 更稳但收敛慢；适合噪声较大的众包偏好数据。

**ORPO (Odds Ratio Preference Optimization)**, [Hong et al., 2024], 把 SFT 和偏好优化合并成一个损失。完整损失是 `SFT_loss(y_w) + λ · log(odds(y_w) / odds(y_l))`。单阶段训练，不用单独跑 SFT。适合资源受限场景（比如 LoRA 微调 7B 模型），跑两阶段不现实。算力更低的情况下，质量能和 SFT+DPO 掰手腕。

**SimPO (Simple Preference Optimization)**, [Meng et al., 2024], 直接扔掉参考模型。损失函数就是 $-\log \sigma(\beta (\bar{r}_w - \bar{r}_l) - \gamma)$，其中 $\bar{r}$ 是长度归一化的 log-prob，$\gamma$ 是 margin。没参考模型意味着显存减半，训练更快。质量据说有竞争力，但对长度归一化和 margin 项很敏感——容易配错。

按你的约束条件选：有成对人类数据 → DPO；有点赞数据 → KTO；噪声偏好 → IPO；单阶段训练 → ORPO；显存受限 → SimPO。 DPO 依然是稳妥的默认选项。

## RLHF and PPO: why anyone still uses it

![fig3: KL divergence over training](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig3_kl_divergence.png)

尽管 DPO 在通用后训练领域占主导，几家前沿实验室在最后阶段依然用基于 PPO 的 RLHF：

- **对抗案例更强。** PPO 能持续惩罚训练中发现的任何新失败模式。 DPO 只能学你收集到的偏好。
- **Reward shaping。** 你可以添加辅助奖励（长度惩罚、格式遵循、拒绝校准），这些没法放进成对偏好里。
- **迭代 RL。** Anthropic 的 constitutional AI (CAI) [Bai et al., 2022] 概念上就是迭代 RLAIF——生成、批评、偏好、重训。

RLHF 的最小 PPO 循环：

```python
# Per training step (heavily simplified)
prompts = sample_prompts(batch)
responses = policy.generate(prompts)         # rollout
rewards   = reward_model(prompts, responses)  # scalar per token
values    = value_head(prompts, responses)
advantages = compute_gae(rewards, values, gamma=1.0, lam=0.95)
for ppo_epoch in range(4):
    new_logps = policy.logprobs(prompts, responses)
    ratio = (new_logps - old_logps).exp()
    clip_loss = -torch.min(
        ratio * advantages,
        ratio.clamp(1 - 0.2, 1 + 0.2) * advantages
    ).mean()
    kl_loss = beta_kl * (new_logps - ref_logps).mean()
    total = clip_loss + kl_loss
    total.backward()
    optimizer.step()
```

KL 项至关重要。没它， PPO 就会攻击奖励模型——生成高奖励但完全不像连贯文本的回答。 Anthropic 的 Sleeper Agents 论文 [Hubinger et al., 2024] 记录了几种有趣的奖励黑客模式，包括“总是以'Yes I can help with that!'结尾”，因为奖励模型学到合规的开头预测高奖励。

## RLHF practical issues: reward hacking, mode collapse, length bias

每个基于 PPO 的 RLHF 训练都会遇到的三个生产级失败模式：

**Reward hacking** [Skalse et al., 2022]。奖励模型是学出来的文本函数——它有盲点。策略会找到它们。典型黑客行为：总是生成道歉语气的回答（因为帮助性标注者把道歉和帮助性关联）、开头用"Certainly!"或"Of course!"（前缀偏见）、拒绝边缘案例请求（因为标注者偏好安全回答），或者用警告和免责声明填充回答。修复方法是迭代：通过人工评估发现黑客行为，生成对抗提示暴露它，用对抗集重训奖励模型，再重训策略。 Anthropic 的 CAI 显式自动化了这个循环。

**Mode collapse**。策略变得确定性——每个回答听起来都一样。症状：训练期间策略熵下降几个数量级。原因： PPO 找到单个高奖励输出并利用它。修复：增加 KL 惩罚（β_kl 从 0.01 调到 0.05），给损失加熵 bonus，用多样化 rollout （每个提示高温采样多个完成）。

**Length bias**。 RLHF 策略系统性地生成比 SFT 策略更长的回答。原因是人类标注者往往偏好稍长的回答（觉得更详尽）。奖励模型捕捉到这点并放大。结果：聊天模型的回答从 200 token 涨到 800 token，质量却没真提升。修复：显式从奖励中减去长度惩罚，或在策略梯度中用长度归一化 log-probs。[Singhal et al., 2024] 详细记录了长度偏差，表明它负责了通常归因于 RLHF 优于 SFT 的 AlpacaEval 分数中的约 1.5 分。

## Constitutional AI: RLAIF lineage

Anthropic 的 Constitutional AI [Bai et al., 2022] 是主导的 RLAIF （RL from AI feedback）方案。两个阶段：

**Phase 1: Supervised CAI.** 用 SFT 模型生成红队提示的回答。让批评模型（通常是同模型不同提示）对照一组宪法原则（“回答不应有害、非法或不道德”）批评回答。让模型根据批评修改回答。在（prompt, revised response）对上训练模型。这产生了一个对齐宪法且无需人类偏好数据的模型。

**Phase 2: RLAIF.** 生成一对回答。让 AI 判断哪个更符合宪法。在这些 AI 偏好上训练奖励模型。用 PPO 针对 AI-derived 奖励模型优化策略。

这里的 trick 是宪法可以编辑并重应用，无需重收人类数据。这比 RLHF 便宜得多，允许迭代宪法本身。缺点是模型的偏见继承自 AI judge——如果 judge 有盲点，策略也会继承。

2024-2026 年 CAI 的演变是 **rule-based rewards (RBR)** [Mu et al., 2024] (OpenAI)。不用学出来的奖励模型，写一组显式规则（“回答≤200 词”，“不应包含医疗建议”，“应引用来源”），让 LLM 评分合规性。这比奖励模型更易调试，更易更新，可作为学习奖励的补充。

## RLVR: RL with verifiable rewards

DeepSeek-R1 [DeepSeek-AI, 2025] (2025 年 1 月) 发布了一个模型，其推理能力几乎全来自 **RLVR**：在数学问题上训练，奖励是“最终答案是否匹配 ground truth？”——程序化检查，不是学出来的模型。没奖励模型意味着没奖励黑客。

这能行是因为数学和代码有 ground truth。模型生成长链式思维，最后提取答案，检查器（Python 解释器， sympy，单元测试）告诉你对错。对 +1，错 0 或 -1。对此跑 PPO/GRPO。

GRPO (Group Relative Policy Optimization)，用在 DeepSeek-V3 和 R1 中，扔掉了 value head——不学价值函数，而是对同一提示采样 $G$ 个回答，计算组内相对优势：

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

这比 PPO 简单得多，适合可验证奖励场景。

DeepSeek-R1 的方案值得详述，因为它是 2025-2026 主导的推理训练模式：

1. **Cold-start SFT** 少量（~10K）高质量 CoT 格式示例，教回答格式。
2. **R1-Zero pure RL**: 在大量数学/代码问题上纯 RLVR 加 GRPO。无 SFT 数据，无奖励模型——只有可验证奖励。 R1-Zero 由此涌现；能力惊人但常不可读（混合语言，奇怪 token）。
3. **Rejection sampling SFT**: 用 R1-Zero 生成回答；过滤只留正确、格式好的；在过滤集上训练新模型。
4. **Final RL stage**: 更多 GRPO，混合可验证奖励和少量通用行为奖励模型。

R1-Zero 最野的地方在于它*没人教*就发现了链式思维推理。纯可验证奖励 RL 教会模型一步步思考是有奖励的。这是首个广泛公开的基础 LLM 纯 RL 涌现推理行为的例子。

当前前沿（Qwen3-Reasoning, GPT-5-thinking, Claude-4.5-thinking, Gemini-3-Thinking）都用 RLVR 式训练作为推理能力的主导信号。基座模型做 SFT，通用行为做 DPO，然后针对数学/代码/逻辑硬刷 RLVR 来获取推理行为。这就是为什么 thinking 模型数学/代码强得多，但对话只强一点点。
## LoRA 对比全量微调：实证证据

![fig5: LoRA vs full FT trade-offs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig5_lora_vs_full.png)

这场争论永远没有尽头。到了 2026 年，实话实说：

- **窄任务 SFT （风格迁移、单一领域）**： LoRA 胜出。质量一样，显存只要 1/10，训练更快，合并或切换也容易。大多数任务 rank 16-64 就够用了。
- **要想从根本上扩展能力（长上下文、新语言）**：必须全量微调。 LoRA 没法足够深入地改变模型的基础表示。
- **DPO 场景**： LoRA 表现不错。 DPO 不需要 drastic 地改变模型。
- **RL （PPO/GRPO）**：前沿实验室默认全量微调。最近有些工作（LoRA-RL, 2025）显示精心调参后 LoRA 能匹配 PPO，但比较脆弱。

当初 LoRA 那篇论文 [Hu et al., 2021] 就展示了，在 GPT-3 175B 上用 rank-8 的 adapter，在 GLUE 基准上能达到全量微调 0.5 % 精度以内的效果，而参数量只有 0.01 %。直觉很简单：微调产生的权重更新是低秩的——给每个权重加一个 rank-$r$ 矩阵就能捕捉大部分变化。

关于 LoRA 局限性的实证证据主要来自 2024 年的几篇论文：

- [Biderman et al., 2024] (LoRA Learns Less and Forgets Less) 显示 LoRA 比全量微调更好地保留了基座能力（灾难性遗忘更少），但学习新任务的速度更慢。对于与预训练差异很大的任务（比如用基座模型做医疗推理）， LoRA 在任务特定评估上比全量微调低 5-10 %。
- [Liu et al., 2024] (DoRA) 把 LoRA 更新分解为幅度和方向分量，证明分开更新这两部分能在相同 rank 下稳定超越标准 LoRA。
- [Hayou et al., 2024] (LoRA+) 显示给 B 矩阵用比 A 矩阵更高的学习率（通常高 16 倍），能稳定超越 vanilla LoRA。

目前的最佳实践是组合拳： DoRA + LoRA+ +  targeting all linear projections，不仅仅是 QKV。

用 PEFT 快速 setup LoRA：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,                    # alpha/r ≈ 2 is typical
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
print(model.print_trainable_parameters())
# trainable params: 167,772,160 || all params: 7,408,357,376 || trainable%: 2.27
```

两个实战要点：

- **Target 所有线性投影，不仅仅是 QKV。** 原 LoRA 论文只针对 QKV；后续工作（QLoRA [Dettmers et al., 2023], LongLoRA）显示针对 MLP 投影也对质量至关重要。
- **DoRA (weight-Decomposed LoRA)** 在相同 rank 下稳定超越 LoRA，额外成本几乎可以忽略。如果你的 trainer 支持，就用它。

QLoRA [Dettmers et al., 2023] 值得单独提一下。它把 LoRA 和基座模型的 4-bit 量化结合起来：把冻结的基座权重存成 NF4 （4-bit normalized float），计算用 BF16。 70B 模型只需 48 GB 显存而不是 140 GB，单张 80 GB H100 就能训练。在大多数任务上，质量与全量 BF16 微调相差不到 0.5 %。 2026 年， QLoRA 是在消费级或单 GPU 设置下微调前沿规模模型的标准配方。

## 实战案例： 32B 基座模型的生产级后训练

我来拆解一套具体方案——生产团队为了从 Qwen3-32B-Base 出发交付一个指令微调的 32B 模型，实际会跑什么。

**SFT 阶段（8 张 H100，约 3 天）。**
- 数据： 200K 混合样本 — 60K Tulu-3 通用， 80K 来自 Claude 的 Magpie 风格合成数据， 30K 领域特定（你的语料）， 30K 长文本（5K-token 回复）。
- 配方：全量微调， BF16， FSDP， LR 5e-6 （比预训练低，因为要避免灾难性遗忘）， cosine schedule， 3 个 epoch， batch size 1M tokens。
- Loss：标准的 CE 带 response masking。
- 评估： AlpacaEval， MT-Bench， IFEval，加上你的领域特定评估。

**DPO 阶段（8 张 H100，约 1 天）。**
- 数据： 30K 偏好对 — 15K UltraFeedback， 5K Anthropic HH， 10K 合成（Claude 在高温下对比模型自己的输出进行评判）。
- 配方： LoRA r=64 （省显存和时间， DPO 不需要全量微调），β=0.1， LR 5e-7， 1.5 个 epoch。
- 评估： Arena-Hard， MT-Bench，加上与仅 SFT 模型的 pairwise judge 对比。

**可选安全_pass （8 张 H100，约 6 小时）。**
- 数据： 3K 拒绝偏好，针对 curated 的红队提示集。
- 配方：在 DPO checkpoint 上继续 DPO，β=0.3 （更强）， 1 个 epoch。

**可选推理 RLVR （32 张 H100， 1K GRPO 步骤约 2 天）。**
- 数据： 5K 数学 + 3K 代码问题，带可验证答案。
- 配方： GRPO，$G=8$ 次 rollout 每 prompt， KL 惩罚 β=0.04， max sequence 8K。

总计：租 H100 算力大约 $10K 就能得到一个完全后训练的模型。数字大致随模型规模线性缩放 — 70B 成本约 2.5 倍， 7B 约 0.3 倍。

## 后训练中的常见坑

生产环境后训练最容易踩的五个坑：

**1. 参考模型没真冻住。** 一个隐蔽的 DPO bug：把 `ref_model` 和 `policy_model` 传成同一个 Python 对象，意味着梯度更新会影响两者。"ref" 的 log-probs 每一步都在变， loss 变得毫无意义，训练悄无声息地产出一个更差的模型。计算 ref log-probs 时用 `model.eval()` 和 `with torch.no_grad():`，或者加载一个单独的模型。

**2. SFT loss 没做 mask。** 训练 `[user_msg, assistant_msg]` 时没 mask 掉 user 部分 → 模型学会预测 user 消息，这会破坏它的对话能力。一定要 mask。

**3. SFT 和 DPO 数据的 Chat template 不一致。** SFT 数据用 `<|im_start|>...<|im_end|>` 格式； DPO 数据用纯文本。 DPO 步骤会训练模型给不带 chat token 的文本分配高概率，这会破坏聊天行为。所有后训练阶段必须使用相同的 chat template。

**4. 合成偏好数据只用了一个裁判模型。** 所有偏好都来自一个强模型（比如 Claude），意味着训练出的模型会继承 Claude 的偏见 — 啰嗦的回复、特定的措辞、特定的拒绝模式。至少混合 3 个裁判（Claude, GPT-4, 一个强的开源模型），并按裁判一致性加权。

**5. 评估集污染。** 从网上抓取的 SFT 数据包含 MMLU 问题。训练出的模型直接“背下”了 MMLU，评估分数 skyrocket，部署质量却没变。训练前一定要对 SFT 数据做去污染（与评估集做 13-gram 匹配）。 LIMA 论文发现 30 % 的常见 SFT 混合数据都有非 trivial 比例的 MMLU 污染。

## 生产现实：前沿实验室到底在交付什么

公开的后训练方案是“SFT → DPO → 可能加 RL”。前沿实验室实际交付的方案要混乱得多：

**SFT-DPO 循环迭代。** Anthropic 的 CAI 是迭代的。 OpenAI 的后训练也跑多个 SFT-DPO 周期，每个周期间生成合成数据。第 1 个周期后，你有一个更好的模型，能生成更好的偏好数据，从而改进第 2 个周期，以此类推。前沿模型通常跑 4-6 个周期。

**合并多个专用专家模型。** 前沿后训练通常涉及训练几个专用变体（一个写代码，一个做安全，一个遵循指令，一个对话语气），然后通过权重平均或插值合并它们。[Wortsman et al., 2022] (ModelSoup) 显示平均微调权重产生的模型比任何单个微调都好。变体：并行训练多个 LoRA adapter，推理时合并。

**持续后训练。** 生产模型不是“训练一次，永久部署”。 Anthropic、 OpenAI 和 DeepMind 都在生产遥测数据上跑持续后训练：点赞/差评信号喂给奖励模型，高质量样本加入 SFT 数据，模型每周或每月重训。你今天聊的"Claude 4.5"是自原始 4.5 发布以来数十次后训练更新的结果。

**深度集成红队测试。** 2026 年的后训练离不开安全。红队生成对抗提示，模型训练拒绝它们，红队找新的对抗提示，如此循环。 Anthropic 在模型卡里公布了一些； OpenAI 和 Google 大部分内部保密。产出是一个通常有帮助但对特定攻击模式越来越 resistant 的模型。

## 指令微调模型的生产级配方

![fig4: post-training decision flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig4_decision_flow.png)

如果 2026 年我要交付一个模型，配方是这样：

1. 选最强的开源基座（Qwen3-32B-Base, LLaMA-3.3-70B-Base, 或 DeepSeek-V3-Base，看成本）。
2. **SFT** 用 50-200K curated 样本。混合通用指令（Tulu-3, OpenHermes）、领域特定（你的数据）和几千个长文本样本。预算允许就全量微调；否则 LoRA r=64。
3. **DPO** 用 10-50K 偏好对。混合 Anthropic HH, UltraFeedback，以及一些来自强裁判模型的合成偏好。β=0.1， 1-2 个 epoch。
4. **可选安全_pass**：收集 1-5K 针对红队提示的拒绝偏好；再做一次 DPO，β=0.3 更激进些。
5. **可选推理 RL**：如果数学/代码重要，用 GRPO 在几千个可验证问题上跑 RLVR， 1-2K 步骤。

这套配方能保证模型达到生产级助理的水准。剩下的工作是评估（第 10 章）和 serving （第 5、 12 章）。

## 2024-2026 研究前沿

当前的 SFT → DPO → RLVR 共识之后，接下来会是什么：

**Online DPO** [Guo et al., 2024]。标准 DPO 用固定的离线偏好数据集。 Online DPO 持续采样新回复，让它们被排序（由人或 AI），然后训练。这结合了 DPO 的稳定性和 PPO 的适应性。几篇 2025 论文报告 online DPO 在标准基准上以更低算力匹配或超越 PPO。

**用于推理的过程奖励模型 (PRM)**。不只奖励最终答案，还奖励模型每一步推理正确。[Lightman et al., 2023] 显示 PRM 在数学基准上超越结果奖励模型。挑战是收集步骤级标注；最近的工作用 LLM 裁判给每一步打分。

**自博弈和自我批评。** SPIN [Chen et al., 2024] (Self-Play Fine-Tuning) 迭代训练模型偏好自己更好的回复胜过自己更差的回复，不需要外部偏好数据。出奇地有效； 3-4 次迭代内达到质量平台期。

**长度控制训练。** 几篇 2024-2025 论文通过长度归一化奖励或 policy log-probs 明确针对长度偏差。[Singhal et al., 2024] 显示这能找回大部分归因于长度的 apparent RLHF 收益，留下更小但更真实的质量增益。

**测试时后训练。** [Akyürek et al., 2024] 等人显示你可以在推理时用几个样本进行后训练（类似 in-context learning 但带权重更新）。对于高风险部署，“在用户前 10 个样本上后训练然后生成”正成为一种可行的模式。

## 总结与下一章

SFT 比大家承认的更重要，对数据卫生的敏感度高于数据量。 DPO 取代了大多数偏好工作的 PPO，但当你想继续适应新的失败模式时， PPO 仍然是正确的工具。 RLVR — 基于可验证奖励的 RL — 是 2024 年后推理能力的解锁键。大多数生产微调 LoRA 胜出；前沿实验室在后 DPO RL 阶段用全量微调。

下一章：**推理优化**。 KV cache 机制， paged attention， continuous batching， speculative decoding，量化（INT8/INT4/AWQ/GPTQ），以及 vLLM vs SGLang vs TensorRT-LLM 的选择。
## 参考文献

- Bradley, R., & Terry, M. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*.
- Christiano, P., Leike, J., Brown, T., et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS*.
- Stiennon, N., Ouyang, L., Wu, J., et al. (2020). Learning to summarize from human feedback. *NeurIPS*.
- Hu, E., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-rank adaptation of large language models. *[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)*.
- Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *[arXiv:2212.08073](https://arxiv.org/abs/2212.08073)*.
- Wortsman, M., Ilharco, G., Gadre, S., et al. (2022). Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML*.
- Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). Defining and characterizing reward hacking. *NeurIPS*.
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT). *NeurIPS*.
- Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.
- Zhou, C., Liu, P., Xu, P., et al. (2023). LIMA: Less is more for alignment. *NeurIPS*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
- Ding, N., Chen, Y., Xu, B., et al. (2023). Enhancing chat language models by scaling high-quality instructional conversations (UltraChat). *EMNLP*.
- Azar, M., Rowland, M., Piot, B., et al. (2023). A general theoretical paradigm to understand learning from human preferences (IPO). *AISTATS 2024*.
- Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's verify step by step. *ICLR 2024*.
- Hubinger, E., Denison, C., Mu, J., et al. (2024). Sleeper agents: Training deceptive LLMs that persist through safety training. *[arXiv:2401.05566](https://arxiv.org/abs/2401.05566)*.
- Ethayarajh, K., Xu, W., Muennighoff, N., et al. (2024). KTO: Model alignment as prospect theoretic optimization. *ICML*.
- Hong, J., Lee, N., & Thorne, J. (2024). ORPO: Monolithic preference optimization without reference model. *EMNLP*.
- Meng, Y., Xia, M., & Chen, D. (2024). SimPO: Simple preference optimization with a reference-free reward. *NeurIPS*.
- Lambert, N., Morrison, J., Pyatkin, V., et al. (2024). Tulu 3: Pushing frontiers in open language model post-training. *[arXiv:2411.15124](https://arxiv.org/abs/2411.15124)*.
- Xu, Z., Jiang, F., Niu, L., et al. (2024). Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. *[arXiv:2406.08464](https://arxiv.org/abs/2406.08464)*.
- Liu, S., Wang, C., Yin, H., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML*.
- Hayou, S., Ghosh, N., & Yu, B. (2024). LoRA+: Efficient low rank adaptation of large models. *ICML*.
- Biderman, D., Portes, J., Ortiz, J., et al. (2024). LoRA learns less and forgets less. *COLM*.
- Singhal, P., Goyal, T., Xu, J., & Durrett, G. (2024). A long way to go: Investigating length correlations in RLHF. *COLM*.
- Mu, T., Helyar, A., Heidecke, J., et al. (2024). Rule-based rewards for language model safety. *[arXiv:2411.01111](https://arxiv.org/abs/2411.01111)*.
- Chen, Z., Deng, Y., Yuan, H., et al. (2024). Self-play fine-tuning converts weak language models to strong language models (SPIN). *ICML*.
- Guo, S., Zhang, B., Liu, T., et al. (2024). Direct language model alignment from online AI feedback. *[arXiv:2402.04792](https://arxiv.org/abs/2402.04792)*.
- Akyürek, E., Damani, M., Qiu, L., et al. (2024). The surprising effectiveness of test-time training for few-shot learning. *[arXiv:2411.07279](https://arxiv.org/abs/2411.07279)*.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)*.