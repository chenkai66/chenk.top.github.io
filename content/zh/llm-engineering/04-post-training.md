---
title: "大模型工程（四）：Post-training —— SFT、DPO、RLHF、RLAIF"
date: 2026-04-29 09:00:00
tags:
  - llm
  - post-training
  - sft
  - dpo
  - rlhf
  - lora
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
预训练出来的 base model 只会续写文本。它不会跟随指令，不会拒绝有害请求，也不会保持人设。这些能力都来自 post-training。论文里的漂亮话和生产级模型的实际效果，差距也藏在 post-training 里。

这篇聊聊每个 post-training 算法到底在优化什么。为什么大多数奖励模型都有问题，只是问题很隐蔽。最后说说 2026 年真正在生产里跑得通的配方。

![大模型工程（四）：Post-training —— SFT、DPO、RLHF、RLAIF — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/illustration_1.jpg)
## 四阶段栈

![fig1: RLHF 流水线总览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig1_rlhf_pipeline.png)

现代 LLM 的后训练流程大致分四个阶段。

第一阶段是 **SFT**（监督微调）。用指令数据训练模型，教它掌握回答格式和基本指令跟随能力。  
第二阶段是 **偏好优化**，方法包括 DPO、IPO、KTO 或 RLHF。目标是让模型学会在两个合理回复中选择人类更喜欢的那个。  
第三阶段是 **在线强化学习**，比如 RLHF、RLAIF 或 RLVR（可验证奖励）。通过奖励模型或程序化检查器进一步调整模型。这个阶段对非推理类模型来说通常是可选的。  
第四阶段是 **专项优化**，包括工具使用 SFT、长上下文 SFT、安全红队测试和 constitutional AI 轮次。

OpenAI、Anthropic 和 Google 依然沿用类似 "SFT → 偏好 DPO → RLHF/RLAIF" 的流程。DeepSeek-R1 [DeepSeek-AI, 2025] 和 o1 系列模型则引入了 **RLVR**，主要针对数学和代码任务，利用程序验证正确性。这是 2024-2025 年后训练流程的最大变化。

技术发展有传承。最早的 "RL from human feedback" 论文是 [Christiano et al., 2017]，将偏好学习用在 Atari 游戏和连续控制任务上。[Stiennon et al., 2020] 把这种方法扩展到了文本摘要任务。[Ouyang et al., 2022]（InstructGPT）首次实现了端到端的 SFT + RLHF 指令微调 LLM。如今所有后训练流程都源自这篇论文。2023-2025 年的技术浪潮（DPO、IPO、KTO、RLVR）可以看作 InstructGPT 的延续与演变。
## SFT：比想象中更重要

SFT 的核心很简单：给模型喂 10 万到 100 万条指令-响应对，训练 next-token prediction loss。但只计算响应部分的损失。这个 mask 很关键——模型不需要预测用户的问题，只需专注于助手的回答。

```python
# SFT 的 loss masking
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

SFT 成败取决于两点。

**数据质量比数量重要。** LIMA [Zhou et al., 2023] 的研究证明，1000 条高质量数据能媲美 5 万条普通数据。Tulu-3 mix [Lambert et al., 2024]（AllenAI，2024）精修到约 20 万条。Qwen3 的 SFT 数据接近 100 万条，但筛选很严格。如果数据质量高，超过 10 万条后效果提升会明显放缓。

**格式一致性不能忽视。** 如果一半数据用 "Sure, I'll help!" 开头，另一半不用，模型就会学得混乱。如果训练数据全是 Markdown 标题，测试时却是普通散文，输出必然错位。预处理阶段必须狠下功夫，统一格式。

还有一个踩过的坑：短回复过多会让模型拒绝生成长文本。模型会从数据中学到长度分布。如果我希望它写 2000 字的文章，SFT 数据中至少要有 5%-10% 的长样本。我之前微调过一个 Qwen3-7B 模型，死活不肯生成超过 800 个 token。原因就是 SFT 数据集中问答对占了大头。
## SFT 数据来源和合成

2026 年，生产环境中的 SFT 数据从哪来？我总结了几种主要来源。

**开源混合数据**：  
Tulu-3 来自 AllenAI，包含 939K 条样本。OpenHermes-2.5 提供了 1M 条 GPT-4 输出的混合数据。UltraChat 筛选了 1.4M 条 ChatGPT 对话 [Ding et al., 2023]。Magpie [Xu et al., 2024] 则用 chat-tuned 模型自我提示生成指令数据。

**领域专用数据**：  
从内部产品日志提取，确保用户同意并移除 PII 信息。也可以让领域专家编写，每条 30 到 100 美元。还有一种方法是用强 teacher 模型蒸馏生成，基于领域特定的种子提示。

**强 teacher 合成数据**：  
让 Claude 或 GPT-4 根据种子主题和少量示例生成 (instruction, response) 对。这是主力方法。到 2026 年，生产环境中大部分 SFT 数据都靠这种方式合成。

Magpie 技术 [Xu et al., 2024] 我觉得值得了解。它的核心思路很简单：用 `<|im_start|>
## DPO：无需奖励模型的偏好优化

![大模型工程（四）：Post-training —— SFT、DPO、RLHF、RLAIF — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/illustration_2.jpg)


![fig2: DPO vs PPO 对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig2_dpo_vs_ppo.png)

经典 RLHF 方法（InstructGPT [Ouyang et al., 2022]）是这样的：先用人类偏好数据训练一个奖励模型，再用 PPO 算法让策略拟合这个奖励模型。虽然能用，但实现起来很麻烦。需要单独维护奖励模型，还得处理 value head、GAE、advantage normalization 和 KL 惩罚。训练过程也不稳定。

**DPO（Direct Preference Optimization）**，[Rafailov et al., 2023]，直接抛弃了奖励模型。核心思想是通过偏好关系推导出策略的闭式解。最终的 loss 是基于 log 概率比的二元交叉熵：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

其中 $y_w$ 是被选中的回复，$y_l$ 是被拒绝的回复。$\pi_{\text{ref}}$ 是冻结的 SFT 模型，作为参考。$\beta$ 控制策略偏离参考模型的程度。没有奖励模型，也不需要 PPO。每个偏好对只需要一次前向传播和一次反向传播。

```python
# DPO loss 10 行实现
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios  = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps   - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
```

到 2026 年，DPO 成为主流有两个原因。第一，它确实好用。[HuggingFace alignment-handbook](https://github.com/huggingface/alignment-handbook) 的报告显示，DPO 在 AlpacaEval 上与 PPO 性能相当，但在单位质量成本上更优。第二，它的训练循环只有前向传播，可以无缝集成到 FSDP/LoRA 技术栈中。

使用 DPO 时需要注意几个坑：

- **β 调参很关键。** β 太小（比如 0.01），策略会漂移，基础能力退化。β 太大（比如 1.0），模型根本学不动。实际生产中，β 通常设置在 0.1 到 0.3 之间。
- **参考模型漂移问题。** 如果连续两次应用 DPO，参考模型就会变成上次 DPO 训练后的模型，原始模型就丢了。一定要保存 SFT 的 checkpoint，每次训练都用它作为参考模型。
- **偏好数据质量很重要。** 合成偏好数据（比如“GPT-4 选择了 A 而不是 B”）容易生成，但会带入 teacher 偏置。混入至少 20% 的人类偏好数据可以有效避免模型性能坍缩。
## DPO 推导细节：从 Bradley-Terry 到闭式解

DPO 的损失函数不是凭空捏造的，而是通过三个步骤推导出来的。理解这个过程很重要，它揭示了 DPO 对偏好数据的假设，也让我搞清楚 β 的意义。

**第一步：Bradley-Terry 偏好模型。** 假设偏好由潜在奖励 $r(x, y)$ 决定。$y_w$ 比 $y_l$ 更受偏好的概率用 logistic 函数表示：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

这就是 [Bradley & Terry, 1952] 提出的成对比较模型。最早用于国际象棋评分系统，现在是将成对偏好转化为标量奖励的标准方法。

**第二步：KL 约束下的奖励最大化。** RLHF 的目标函数是：

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta \, \text{KL}(\pi \| \pi_{\text{ref}})$$

对 π 求导并令其为零，得到最优策略：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

这是最大熵强化学习领域的经典结果。接着解出 $r$：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**第三步：代入 Bradley-Terry 模型，去掉 $Z(x)$。** Bradley-Terry 模型只依赖奖励差值，因此 $Z(x)$ 被抵消掉：

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

最大化观测到的偏好的对数似然，就得到了 DPO 的损失函数。整个 RLHF 流程简化成了对数概率比上的二元交叉熵。不需要奖励模型，也不需要采样，更不需要 PPO。

β 的含义也很清楚了：它是隐含奖励的反温度参数。当 β 很低（比如 0.01），小奖励也会导致策略大幅调整——模型会激进地学习每个偏好。当 β 很高（比如 1.0），策略几乎无法偏离 $\pi_{\text{ref}}$——偏好对模型的影响微乎其微。经验上，β 的最佳范围是 0.1 到 0.3。这时隐含奖励既足够大以提供信息，又不会让模型漂移。
## DPO 变体：KTO、IPO、ORPO、SimPO

2024 到 2025 年，DPO 衍生出一堆变体，每种都针对特定问题做了改进。

**KTO（Kahneman-Tversky Optimization）**，[Ethayarajh et al., 2024]，用“好/坏”标签代替成对偏好。损失函数是非对称的，基于 Kahneman-Tversky 风格的价值函数：人天生怕损失，所以惩罚坏回答比奖励好回答更重要。如果你只有点赞或点踩数据，但没有成对比较数据，KTO 很适合。实际测试发现，有成对数据时 KTO 和 DPO 效果差不多；只有独立数据时，KTO 更强。

**IPO（Identity Preference Optimization）**，[Azar et al., 2023]，把 sigmoid 损失换成 MSE 损失，避免了 DPO 在噪声偏好上的过拟合问题。DPO 的 sigmoid 函数在策略自信时会饱和，一旦“赢了”一对比较，梯度信号几乎就没了。而 IPO 的 MSE 损失能持续提供梯度。实际用下来，IPO 更稳定，但收敛慢一些，适合处理噪声大的众包偏好数据。

**ORPO（Odds Ratio Preference Optimization）**，[Hong et al., 2024]，把 SFT 和偏好优化合并成一个损失函数。公式是 `SFT_loss(y_w) + λ · log(odds(y_w) / odds(y_l))`。训练只需一个阶段，不用单独做 SFT。资源有限时（比如用 LoRA 微调 7B 模型），这种方法很实用。计算成本低，效果却能媲美先 SFT 再 DPO 的两阶段方法。

**SimPO（Simple Preference Optimization）**，[Meng et al., 2024]，直接去掉参考模型。损失函数简化为 $-\log \sigma(\beta (\bar{r}_w - \bar{r}_l) - \gamma)$，其中 $\bar{r}$ 是长度归一化的对数概率，$\gamma$ 是边界值。去掉参考模型后，内存占用减半，训练更快。不过 SimPO 对长度归一化和边界值很敏感，配置不好容易翻车。

根据需求选变体：  
- 成对标注数据 → DPO  
- 点赞点踩数据 → KTO  
- 噪声偏好数据 → IPO  
- 单阶段训练 → ORPO  
- 内存受限 → SimPO  

虽然有了这些变体，DPO 还是最稳妥的默认选择。
## RLHF 和 PPO：为什么还有人在用

![fig3: 训练中的 KL 散度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig3_kl_divergence.png)

DPO 虽然在通用后训练中占主导地位，但一些前沿实验室在最后阶段依然选择 PPO。原因很简单。

PPO 在对抗场景中表现更好。它能持续惩罚模型暴露出的新问题模式。DPO 不行，只能依赖已有的偏好数据。

奖励设计更灵活。我可以加入长度惩罚、格式合规性、拒绝校准等辅助奖励。这些无法通过成对偏好实现。

PPO 支持迭代强化学习。Anthropic 的 constitutional AI（CAI）[Bai et al., 2022] 就是迭代 RLAIF 的一种形式。生成、批评、偏好、再训练，循环往复。

以下是 RLHF 中一个简化的 PPO 循环：

```python
# 单次训练步骤（高度简化）
prompts = sample_prompts(batch)
responses = policy.generate(prompts)         # rollout
rewards   = reward_model(prompts, responses) # 每个 token 对应一个标量
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

KL 项至关重要。没有它，PPO 会直接“欺骗”奖励模型。生成的文本虽然高奖励，但毫无逻辑。Anthropic 的 Sleeper Agents 论文 [Hubinger et al., 2024] 提到几种有趣的奖励操控模式。比如模型总是以 "Yes I can help with that!" 结尾。这是因为奖励模型学到了顺从式的开头往往对应高奖励。
## RLHF 的实战问题：奖励操控、模式坍缩、长度偏差

每次用 PPO 做 RLHF，都会踩到三个坑。

**奖励操控** [Skalse et al., 2022]。奖励模型是学出来的文本函数，总有盲区。策略总会找到漏洞。比如，生成一堆带歉意的回答（因为标注员觉得道歉显得更乐于助人）。或者总以 "Certainly!" 或 "Of course!" 开头（前缀偏差）。再比如，拒绝处理边缘情况请求（因为标注员喜欢保守答案）。还有在回答里堆砌免责声明和附加条件。解决办法是迭代改进。先用人发现漏洞，生成对抗性提示暴露问题，然后用这些数据重新训练奖励模型，再更新策略。Anthropic 的 CAI 方法自动化了这个循环。

**模式坍缩**。策略变得太确定性，所有回答听起来都一样。症状很明显：训练中策略的熵值会下降好几个数量级。原因是 PPO 找到一个高奖励输出后就反复利用。解决方法有几个。提高 KL 惩罚系数（把 β_kl 从 0.01 提到 0.05）。在损失函数里加熵奖励。用多样化采样，对每个提示用高温度生成多个补全结果。

**长度偏差**。RLHF 策略生成的回答比 SFT 策略长得多。原因很简单：人类评分者更喜欢稍长的回答，觉得它们更全面。奖励模型捕捉到这点后进一步放大。结果就是聊天模型的回答从 200 token 长到 800 token，但质量没提升。解决方法有两种。一是在奖励里减去长度惩罚。二是在策略梯度里用长度归一化的 log-prob。[Singhal et al., 2024] 详细分析了长度偏差，证明它导致 AlpacaEval 中 RLHF 比 SFT 多出约 1.5 分。
## Constitutional AI：RLAIF 的发展脉络

Anthropic 提出的 Constitutional AI [Bai et al., 2022] 是目前主流的 RLAIF（从 AI 反馈中学习强化学习）方法。它分两个阶段。

**第一阶段：监督式 CAI。** 用 SFT 模型生成红队提示的回复。再用一个批评模型（通常是同一个模型，但提示不同）根据宪法原则评价回复。宪法原则比如“回复不能有害、违法或不道德”。让模型根据批评修改回复。最后用 (提示, 修改后的回复) 对训练模型。这样就能对齐宪法，还不需要人工标注数据。

**第二阶段：RLAIF。** 针对同一个提示生成两组回复。让 AI 判断哪个回复更符合宪法原则。用这些偏好数据训练奖励模型。再用 PPO 算法优化策略，使其与 AI 奖励模型对齐。

这种方法的好处是，宪法可以随时编辑和重新应用，不用重新收集人类数据。成本比 RLHF 低得多，还能快速迭代宪法本身。缺点也很明显：AI 评判者的偏见会直接传递给策略。如果评判者有盲点，策略也会跟着出问题。

2024 年到 2026 年，CAI 的演进方向是 **基于规则的奖励（RBR）** [Mu et al., 2024]（OpenAI）。不再用学习型奖励模型，而是写一组明确的规则。比如“回复长度 ≤ 200 字”、“不能包含医疗建议”、“必须引用来源”。然后让 LLM 根据规则评估合规性。这种方式比奖励模型更容易调试，更新也方便，还能作为学习型奖励的补充手段。
## RLVR：基于可验证奖励的强化学习

DeepSeek-R1 [DeepSeek-AI, 2025]（2025 年 1 月发布）的推理能力几乎全靠 **RLVR**。训练时用数学题，奖励标准很简单："答案对不对"——程序直接判断，不用训练奖励模型。没有奖励模型，自然不存在奖励操控问题。

这种方法有效，因为数学和代码有明确答案。模型生成长串推理过程，最后提取答案，交给检查工具（Python 解释器、sympy 或单元测试）判断对错。答案正确，奖励 +1；错误，奖励 0 或 -1。用这个信号跑 PPO 或 GRPO。

GRPO（Group Relative Policy Optimization）是 DeepSeek-V3 和 R1 用的优化方法。它去掉价值头，不学价值函数，而是对同一个提示采样 $G$ 个回复，在组内计算相对优势：

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

相比 PPO，GRPO 更简单，适合可验证奖励场景。

DeepSeek-R1 的训练方法值得细讲，它是 2025-2026 年推理训练的主流模式。

1. **冷启动 SFT**：用约 10K 高质量 CoT 格式样本，教会模型生成符合要求的回复格式。
2. **R1-Zero 纯 RL**：在大量数学和代码问题上跑纯 RLVR，用 GRPO 方法。不依赖 SFT 数据，也没有奖励模型，完全靠可验证奖励驱动。R1-Zero 能力强，但输出常混杂语言，token 奇怪，难以阅读。
3. **拒绝采样 SFT**：用 R1-Zero 生成大量回复，筛选出正确且格式良好的部分，再用这些数据训练新模型。
4. **最终 RL 阶段**：继续用 GRPO 方法，结合可验证奖励和小规模奖励模型调整通用行为。

R1-Zero 最让人惊讶的是，它自己学会了链式思维推理，没人教过它。纯 RL 训练让它发现一步步思考能得到更高奖励。这是第一个公开例子，展示了基础大模型通过纯 RL 训练涌现出推理能力。

目前前沿模型（如 Qwen3-Reasoning、GPT-5-thinking、Claude-4.5-thinking、Gemini-3-Thinking）都以 RLVR 风格训练为核心信号。基础模型先用 SFT 和 DPO 调整通用行为，再在数学、代码和逻辑任务上强化 RLVR 训练。这就是为什么这些 thinking 模型数学和代码能力大幅提升，对话能力却只小幅改进。
## LoRA vs 全量微调：经验证据

![fig5: LoRA vs 全量微调权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig5_lora_vs_full.png)

LoRA 和全量微调的争论由来已久。2026 年，我的看法如下：

- **窄任务 SFT**（风格迁移、单领域）：LoRA 更优。质量相当，GPU 内存只需 1/10，训练更快，合并或切换也方便。rank 16-64 能搞定大部分任务。
- **扩展核心能力的 SFT**（长上下文、新语言）：必须全量微调。LoRA 改不动模型的基础表征。
- **DPO**：LoRA 表现不错。DPO 不需要大幅改变模型。
- **RL（PPO/GRPO）**：前沿实验室默认全量微调。不过，2025 年的研究（LoRA-RL）表明，经过精细调整，LoRA 可以追上 PPO 的效果，但稳定性较差。

LoRA 原论文 [Hu et al., 2021] 证明，在 GPT-3 175B 上用 rank-8 的 adapter，能在 GLUE 基准测试中达到与全量微调相差不到 0.5% 的精度，而参数量仅为全量微调的 0.01%。背后的直觉是，微调产生的权重更新通常是低秩的——给每个权重加上一个 rank-$r$ 的矩阵，就能捕捉到大部分变化。

关于 LoRA 局限性的证据来自几篇 2024 年的论文：

- [Biderman et al., 2024]（LoRA Learns Less and Forgets Less）指出，LoRA 比全量微调更能保留基础能力（灾难性遗忘更少），但学习新任务的速度较慢。对于和预训练差距很大的任务（比如医学推理），LoRA 在任务专项评估中比全量微调低 5-10%。
- [Liu et al., 2024]（DoRA）将 LoRA 更新分解为幅值和方向两部分，分别更新后，在相同 rank 下持续优于标准 LoRA。
- [Hayou et al., 2024]（LoRA+）发现，B 矩阵的学习率比 A 矩阵高（通常高 16 倍）时，效果始终优于普通 LoRA。

目前的最佳实践是结合 DoRA + LoRA+，并对所有线性投影应用 LoRA，而不仅仅是 QKV。

用 PEFT 快速配置 LoRA：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,                    # alpha/r ≈ 2 是典型值
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

两个实战建议：

- **覆盖所有线性投影，不只是 QKV。** LoRA 原论文只针对 QKV，但后续工作（如 QLoRA [Dettmers et al., 2023] 和 LongLoRA）表明，MLP 投影也必须加入，才能保证质量。
- **DoRA（weight-Decomposed LoRA）** 在相同 rank 下持续优于 LoRA，额外开销几乎可以忽略。如果训练框架支持，就用它。

QLoRA [Dettmers et al., 2023] 值得单独提一下。它把 LoRA 和基础模型的 4-bit 量化结合起来：冻结的基础权重存储为 NF4（4-bit 归一化浮点数），计算则用 BF16。这样一来，70B 的模型从 140 GB 压缩到 48 GB，单张 80 GB H100 就能训练。在大多数任务上，质量与全量 BF16 微调相差不到 0.5%。QLoRA 已成为 2026 年消费级硬件或单卡环境下微调前沿规模模型的标准方案。
## 实战案例：基于 32B 基础模型的生产后训练

我来拆解一个实际方案，讲讲如何从 Qwen3-32B-Base 出发，训练出一个指令调优的 32B 模型。

**SFT 阶段（8 张 H100，约 3 天）**  
数据分四部分：60K 条 Tulu-3 通用数据、80K 条 Magpie 风格合成数据（来自 Claude）、30K 条领域专用数据（你的语料库），以及 30K 条长文本数据（5K-token 的回复）。  
方法是全量微调，BF16 精度，FSDP 分布式训练。学习率设为 5e-6，比预训练低，避免灾难性遗忘。用余弦退火调度，跑 3 轮，批量大小 1M tokens。  
损失函数用标准交叉熵，带响应掩码。  
评估工具包括 AlpacaEval、MT-Bench、IFEval，再加上你自己的领域专用评估。

**DPO 阶段（8 张 H100，约 1 天）**  
数据有 3 万条偏好对：15K 条 UltraFeedback 数据、5K 条 Anthropic HH 数据，还有 10K 条合成数据（Claude 在高温度下对模型输出进行自我对比生成）。  
方法用 LoRA r=64，节省内存和时间，DPO 不需要全量微调。β 设为 0.1，学习率 5e-7，跑 1.5 轮。  
评估用 Arena-Hard 和 MT-Bench，再和仅 SFT 的模型做成对评判。

**可选安全增强（8 张 H100，约 6 小时）**  
数据是 3K 条拒绝偏好对，基于一个精心设计的红队提示集。  
方法是在 DPO 检查点上继续 DPO 训练，β 提高到 0.3（更强），跑 1 轮。

**可选推理强化（32 张 H100，约 2 天，1K GRPO 步骤）**  
数据包括 5K 条数学题和 3K 条有可验证答案的代码题。  
方法用 GRPO，每条提示生成 $G=8$ 次，KL 惩罚 β=0.04，最大序列长度 8K。

总计下来，租用 H100 的算力成本大约 $10K，能拿到一个完整的后训练模型。计算资源需求随模型规模线性增长——70B 模型的成本是 2.5 倍，7B 模型的成本是 0.3 倍。
## Post-training 常见踩坑

在生产环境做 post-training 时，我踩过最多的坑有 5 个。

**1. 参考模型没冻结。** DPO 里有个隐蔽问题：`ref_model` 和 `policy_model` 是同一个 Python 对象时，梯度更新会同时影响两者。结果是 "ref" 的 log-prob 每步都在变，loss 完全乱了，训练默默产出一个更差的模型。计算 ref log-prob 时，记得用 `model.eval()` 加 `with torch.no_grad():`，或者直接加载独立模型。

**2. SFT loss 没加 mask。** 在 `[user_msg, assistant_msg]` 上训练时，如果用户部分没 mask，模型会学会预测用户消息。这会破坏对话能力。永远记得加 mask。

**3. SFT 和 DPO 数据格式不一致。** SFT 数据用 `\<|im_start|\>...\</|im_end|\>` 格式，DPO 数据却是纯文本。DPO 阶段会让模型给没有 chat token 的文本分配高概率，聊天行为就被破坏了。所有 post-training 阶段必须统一 chat template。

**4. 单一 judge 合成偏好。** 如果偏好全来自一个强模型（比如 Claude），训出来的模型会继承它的偏置——啰嗦回答、特定措辞、特定拒绝模式。建议混合至少 3 个 judge（Claude、GPT-4、一个强开源模型），按一致性加权。

**5. Eval 数据污染。** 从网上爬的 SFT 数据可能包含 MMLU 题目。模型直接“记住”这些题目，eval 分数飙升，但部署质量没变化。训练前一定要去污（13-gram 匹配 eval 集）。LIMA 论文提到，30% 的常见 SFT 数据集存在非平凡比例的 MMLU 污染。
## 生产真相：前沿实验室真正交付的内容

公开的后训练流程是“SFT → DPO → 可能 RL”。但实际操作要复杂得多。

**反复迭代的 SFT-DPO 循环。** Anthropic 的 CAI 方法就是迭代式的。OpenAI 也跑多轮 SFT-DPO，每轮之间生成合成数据。第一轮提升模型性能，生成更好的偏好数据。第二轮用这些数据进一步优化。通常需要 4 到 6 轮。

**多个专业模型合并。** 前沿实验室会训练多个变体模型。一个专注代码，一个负责安全，一个优化指令跟随，一个改进对话语气。然后通过权重平均或插值合并。[Wortsman et al., 2022]（ModelSoup）证明，微调权重取平均值，效果优于单个微调模型。另一种方法是并行训练多个 LoRA 适配器，推理时合并。

**持续的后训练。** 模型不是“一次性训练，永久部署”。Anthropic、OpenAI 和 DeepMind 都基于生产遥测数据持续优化。用户点赞或点踩信号输入奖励模型。高质量样本加入 SFT 数据集。模型每周或每月重新训练一次。今天用的“Claude 4.5”，其实是经过几十次更新后的版本。

**深度整合红队测试。** 后训练和安全性密不可分。红队生成对抗性提示，模型学会拒绝。红队再找新对抗性提示，模型继续优化。Anthropic 在模型卡片中公开部分细节，OpenAI 和 Google 大多保密。最终模型既实用，又对特定攻击模式更抗打。

下一节会聊我在生产环境踩过的坑。
## 一个生产级指令微调模型的配方

![fig4: post-training 决策流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/04-post-training/fig4_decision_flow.png)

如果要在 2026 年上线一个生产级模型，我会这么做：

1. 选最强的开源基础模型。预算够就用 Qwen3-32B-Base 或 LLaMA-3.3-70B-Base，预算紧就选 DeepSeek-V3-Base。
2. **SFT** 用 5 万到 20 万条精选样本。通用指令数据用 Tulu-3 和 OpenHermes，领域数据用自己的，再加几千条长文本样本。预算足就全量微调，预算有限就用 LoRA r=64。
3. **DPO** 准备 1 万到 5 万对偏好数据。混合 Anthropic HH 和 UltraFeedback，再用强 judge 模型生成一些合成偏好数据。设置 β=0.1，跑 1-2 个 epoch。
4. **安全性优化（可选）**：针对红队攻击提示收集 1-5K 条拒绝偏好的数据，再做一次 DPO。设置 β=0.3，让模型更果断拒绝不当请求。
5. **推理强化学习（可选）**：如果数学或代码能力重要，就在几千道可验证问题上用 GRPO RLVR 方法，跑 1-2K 步。

这套方法基本能搞定一个生产级助手模型。评估和部署的事，留到后面再说（评估看第 10 篇，部署看第 5、12 篇）。
## 研究前沿 2024-2026

SFT → DPO → RLVR 的共识之后，接下来有哪些新方向？

**Online DPO** [Guo et al., 2024]。标准 DPO 用固定离线数据集。Online DPO 不一样，它实时采样新回复，让人或 AI 排序，然后训练。既稳又灵活。多篇 2025 年论文显示，Online DPO 在标准测试中效果不输 PPO，甚至更好，计算成本还更低。

**过程奖励模型（PRM）**。传统方法只看最终答案，PRM 则奖励每一步推理的正确性。[Lightman et al., 2023] 发现，PRM 在数学任务上比结果奖励模型更强。难点是标注每一步的数据。最近有人用 LLM 自动评分，效果不错。

**Self-play 和 Self-critique。** SPIN [Chen et al., 2024] 是个亮点。模型自己迭代训练，学会挑好的回复，丢掉差的。全程不用外部数据。没想到效果出奇好，3 到 4 次迭代就稳定了。

**长度受控训练。** 长度偏差是个老问题。2024 到 2025 年多篇论文提出对奖励或策略 log-prob 做长度归一化。[Singhal et al., 2024] 的实验表明，这种方法能还原大部分因长度带来的假收益，留下更真实的质量提升。

**推理时 Post-training。** [Akyürek et al., 2024] 提了个新思路：在推理阶段用少量样本做 Post-training。类似上下文学习，但更新权重。高风险场景下，"先用用户前 10 个样本 Post-train，再生成" 已经可行。下一节会详细聊这个模式的实际踩坑经验。
## 小结与下一篇

SFT 比很多人想象的更重要。数据质量比数据量更关键。DPO 已经取代 PPO，成为处理偏好任务的主流方法。但要持续适应新的失败模式，PPO 仍是更好的选择。RLVR（基于可验证奖励的强化学习）是 2024 年后提升推理能力的核心技术。生产环境微调中，LoRA 是首选。前沿实验室在 post-DPO 强化学习阶段，依然会用全量微调。

下一篇聊**推理优化**。我会讲 KV cache 的工作机制、paged attention、continuous batching、speculative decoding、量化技术（INT8/INT4/AWQ/GPTQ），还会对比 vLLM、SGLang 和 TensorRT-LLM 的选择。
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
