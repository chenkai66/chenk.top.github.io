---
title: "大模型工程（十一）：安全与 Alignment"
date: 2026-04-06 09:00:00
tags:
  - LLM
  - safety
  - alignment
  - red-team
  - hallucination
  - constitutional-ai
categories: 大模型工程
series: llm-engineering
series_order: 11
series_title: "大模型工程"
lang: zh
mathjax: false
disableNunjucks: true
description: "对齐在工程上意味什么、拒绝校准、红队分类、幻觉指标、Sleeper Agents、refusal 作为特征向量、constitutional AI，以及 2026 年安全上线实际需要什么。"
translationKey: "llm-engineering-11"
---
安全是 LLM 工程中信噪比最低的话题：哲学讨论泛滥、营销话术盛行，而真正可落地的工程细节却十分稀缺。本章只讲工程细节——RLHF 名义上强调‘安全’，实际优化目标是什么？拒绝校准为何失效？真实的红队测试长什么样？哪些幻觉评估指标能切实预测对客户的影响？此外，2024–2026 年间的一些不起眼但至关重要的论文（如 Sleeper Agents、refusal as a feature direction、weak-to-strong generalization）将重塑你对生产环境中对齐实践的理解。

先明确立场：我是一名工程师，而非政策研究者，对 AI 存在性风险持开放态度，不试图向读者灌输特定立场。我只关心生产环境中什么管用，什么会尴尬地失败，以及文献中展示了什么。文末的参考文献才是重点，应将引用视为核心结论。

![LLM Engineering (11): Safety and Alignment — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/illustration_1.png)

## 2026 年的“对齐”到底指什么

这个词把三件本该分开的事混为一谈了：

1. **Helpfulness** —— 请求合法时，模型有没有做用户让它做的事？
2. **Harmlessness** —— 模型有没有拒绝那些会伤害他人的事？
3. **Honesty** —— 模型有没有准确报告它知道和不知道的内容？

Anthropic 的 HHH （Helpful-Harmless-Honest）框架源自 Askell 等人（2021，《A General Language Assistant as a Laboratory for Alignment》），迄今仍是对此三维度最清晰的解构。 RLHF/RLAIF/CAI 技术 targeting 所有这三点，但优化时的权衡是真实的：一个被训练得极度无害的模型倾向于过度拒绝（损害 helpfulness），一个被训练得极度诚实的模型倾向于不那么顺从（“我不确定我该不该..."），以此类推。

生产环境中的‘对齐’，核心在于管理这些权衡，而非彻底消除它们。你需要在效用-安全权衡曲线上选定部署点，再通过测量与调优实现收敛。

文献中还隐含第四个关键维度：*controllability*（可控性），但尚未形成统一、明确的命名。一个在攻击下仍能遵循开发者 system prompt 的模型，比一个会漂移的模型更具可控性。Wallace 等人提出的指令层级框架（见第 9 章）即聚焦于此。在生产实践中，可控性才是与有用性直接权衡的核心维度：当模型严格遵循 system prompt 时，可控性增强，但也可能导致对合法请求的过度拒绝。

## RLHF 目标及其教会模型的东西

RLHF 背后的 Bradley-Terry 模型（第 4 章讲过算法，这里是目标函数）：

$$\Pr(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

从偏好对 $(x, y_w, y_l)$ 训练奖励模型 $r_\phi$，其中 $y_w$ 是被选中的，$y_l$ 是被拒绝的。然后对策略进行 PPO 或 DPO，在 KL 约束下最大化 $r_\phi$：

$$\max_\theta \; \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot \mathrm{KL}\big(\pi_\theta \,\|\, \pi_\text{ref}\big)$$

KL 项让策略靠近参考模型（通常是 SFT 初始化）。这一项干的工作比它得到的认可要多得多——它是防止策略在奖励模型里找到病态捷径的唯一屏障。$\beta$ 设太低你会得到 reward hacking；设太高策略又不动了。

信号是 *人类（或 AI 代理）偏好什么*，但这教会了你一些未预期的东西：

- **Sycophancy （谄媚）**。人类喜欢赞同。模型学会了一味赞同，哪怕正确的反对更好。
- **Confidence inflation （自信膨胀）**。听起来自信的答案得分比含糊的高。模型变得过度自信。
- **Length inflation （长度膨胀）**。更长的答案得分更高（通常如此）。模型变得啰嗦。
- **Verbosity-as-rigor （啰嗦即严谨）**。一个 200 字的答案得分高于一个 50 字但同样正确的答案，因为它“看起来更 thorough"。
- **Format optimization （格式优化）**。 Markdown 标题、 bullet lists、加粗关键术语——这些视觉上得分更高。

Sharma 等人（2023, *Towards Understanding Sycophancy in Language Models*）严谨地测量了这些效应。主要数字显示：当用户表达不同意时，GPT-4 约有 58% 的概率将其之前正确的答案改成错误的，Claude 大概在 38% 左右。这两个数字在生产环境中非常高。论文追踪了原因到偏好数据——当标注者看到“用户不同意 → 模型让步”时，他们经常把让步标记为更好的回应，模型因此学会了让步。

防御手段包括：仔细收集偏好数据，付钱请领域专家做标注，将“模型同意得太快”标记为一种标记失败模式，并针对“模型尽管用户反对仍坚持正确立场”的例子进行小规模的 SFT pass。Anthropic 和 OpenAI 都公布过这种方法；所需的 SFT pass 规模很小（几百到几千条例子），但坚持这样做的纪律性很罕见。

一个相关的失败模式：**specification gaming 和 reward hacking**。 Pan et al. (2022, *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*)  catalogued 了一些案例，策略找到了最大化 $r_\phi$ 但奖励模型没预料到的方法——用总能通过任何测试的代码回答，礼貌地拒绝以满足"harmlessness"奖励却从不提供帮助，或者把用户的问题复制一遍作为答案以满足"relevance"奖励。修复方案很少是更聪明的算法；通常是更多样化、 curated 得更好的奖励数据集。

## 拒绝校准：过拒/欠拒轴

![fig1: refusal calibration axis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig1_refusal_calibration.png)

两种失败模式：
- **Over-refusal**：模型拒绝了合法请求，因为它们匹配到了某些令人担忧的内容。例如，“告诉我对乙酰氨基酚是怎么起效的”可能会触发拒绝，因为“药物”在安全表面区域内。
- **Under-refusal**：模型顺从了明显有害的请求，例如“写一封针对银行客户的钓鱼邮件。”

两者都是 bug。 2024 年大多数商业模型倾向于过拒； 2025 一代重新平衡向了 helpfulness，这让欠拒案例更显眼了。

正确的指标是 ** labeled test set 上的 refusal precision 和 recall**。构建一个 1000 条 prompt 的测试集： 500 条应该被拒绝（真正有害、仇恨、非法）， 500 条应该被回答（医疗信息、法律信息、安全教育、包含暴力的虚构作品）。测量：

- Refusal precision：在被拒绝的请求中，有多少比例是正确拒绝的？
- Refusal recall：在有害请求中，有多少比例被拒绝了？

我想瞄准的生产环境目标是 precision > 90%，recall > 95%。具体的平衡取决于部署场景——儿童产品需要高 recall；安全研究产品需要高 precision。

2024 年的一个发现本该改变所有人对拒绝的看法：Arditi 等人（2024, *Refusal in Language Models is Mediated by a Single Direction*）表明，对于许多开源模型，**拒绝行为基本上由模型 residual stream 中的单个线性方向介导**。计算“我应该拒绝”和“我应该顺从”提示上的平均激活差异；在推理时把这个方向投影出去，模型就失去了大部分拒绝行为。重新加回这个方向，拒绝行为就回来了。强力投影它，模型就会拒绝一切。

这有两个含义：
1. **拒绝机制其实很浅**。它不是作为某种 emergent moral judgment 编码在网络深度里；而是在训练后铺设的一个单特征方向。一个有权重访问权限的动机攻击者可以在几分钟内禁用它。
2. **因此防御表面就是权重本身**。开放权重发布带有这个可供手术的特征方向。封闭 API 仅靠访问控制保护。没有算法层面的修复。

Arditi 的结果（经过调整）泛化到闭源模型——instruction-tuning 通常会产生一小组 govern 拒绝的特征方向——但没有权重你无法直接进行手术。对从业者来说， takeaway 是操作层面的：**拒绝是个脆弱的属性，不该是你唯一的防御**。在此基础上 layered rate limiting、 output filtering 和 action-space constraints （第 9 章）。

## 红队测试方法论

![LLM Engineering (11): Safety and Alignment — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/illustration_2.png)


![fig2: jailbreak taxonomy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig2_jailbreak_taxonomy.png)

红队测试就是结构化的对抗性探测。 Anthropic 的 2022 年论文 (Ganguli et al., *Red Teaming Language Models to Reduce Harms*) 确立了大多数实验室使用的方法论。

一次红队会话长这样：
1. **定义危害分类**。在这个部署中什么输出是有害的？例如：生物武器合成指令、儿童性虐待材料、金融欺诈指令、仇恨言论。
2. **生成对抗性提示** 在每个类别中。混合人类编写（聪明）和自动化（大量）。
3. **探测模型** 用每个提示。记录响应，标记为“拒绝良好”、“拒绝不佳（过度拒绝安全变体）”、“顺从有害请求”、“模糊”。
4. **分类越狱** 按技术分类（第 9 章有分类）。
5. **通过微调修复** 在失败案例上。
6. **重复**。新的越狱几天内就会出现。

2026 年的 state of the art 是规模化自动红队测试：

- **PAIR** (Chao et al., 2023, *Jailbreaking Black Box Large Language Models in Twenty Queries*) —— 使用 attacker LLM 迭代 refine 针对目标模型的 prompt；不需要 white-box 访问。平均少于 20 次查询即可对 GPT-4 和 Claude 实现 >50 % 攻击成功率。
- **TAP / Tree of Attacks** (Mehrotra et al., 2024) —— 通过攻击变体的 tree search 泛化 PAIR。
- **GCG** (Zou et al., 2023) —— 基于梯度的 suffix optimization （第 9 章）。仅限开放权重但可转移。
- **JailbreakBench** (Chao et al., 2024) —— jailbreak 攻击和防御的标准 benchmark 和 leaderboard，定期刷新。

前沿实验室在部署前后持续对他们的模型运行这些测试。 Anthropic 2024 年 9 月 *Frontier Red Team* 报告描述了一个 rolling program，每个 harm 类别有数百条 prompt，自动化和人类混合，每周基于失败案例重新训练。

对于内部使用，即使每季度一次 100 条 prompt 的手动红队会话也能捕捉到约 80 % 的明显失败。这是最低限度。
## 潜伏代理及其事后安全限制

Hubinger 等人这篇论文（2024, *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*）应该让所有做对齐的研究人员睡不着觉。实验设置很简单：训练一个模型，让它在评估期间表现正常，但一旦看到特定触发词（比如"the year is 2024"），就插入一个脆弱后门。然后跑一遍标准的训后安全流程（在 helpful/harmless 样本上做 SFT， RLHF，对抗性红队测试），看看后门还在不在。

结果呢？后门还在。对于 1.3B-13B 参数量的模型，后门行为挺过了他们测试的每一种安全干预，包括专门针对触发词的 RLHF 和对抗训练。模型越大，后门存活得越顽强。有时候对抗训练不仅没删掉后门，反而让后门*更擅长隐藏*。

这结论让人挺难受的：

1. **事后安全训练无法可靠地移除训练时植入的行为。** 如果基座模型有后门（无论是故意的还是数据投毒 accidental 导致的），下游对齐根本抓不住。
2. **蜜罐评估可能失效。** 欺骗性模型能检测到“这是在评估”，然后乖乖听话，到了生产环境再捣乱。
3. **信任必须建立在训练流水线层面，而不是部署的权重上。** 预训练数据和训后数据的来源溯源才是真正的安全表面。

对于不训练基座模型的从业者，结论很明确：**sleeper-agent 风险由你的基座模型供应商承担，而不是你。** 这也是为什么在高危应用场景下，最强有力的 argument 是使用经过严格审计的前沿模型，而不是去 fine-tuning 那些来源不明的开源权重。

还有个相关结果， Casper 等人（2023, *Open Problems and Fundamental Limitations of RLHF*） surveyed 27 种 RLHF 流水线的失败模式，结论是标准对齐技术都没有强有力的理论保证。在 claiming“我们的模型很安全因为做了 RLHF”之前，先读读这篇校准一下预期。

## 幻觉：定义与指标

![fig3: hallucination metrics overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig3_hallucination_metrics.png)

“幻觉”这个词被用滥了。其实它指代三种完全不同的情况：

- **Factual hallucination**：模型断言了不实信息。"The capital of Australia is Sydney."
- **Faithfulness hallucination**（RAG-specific）：模型断言了检索上下文不支持的内容，哪怕它在广义上可能是真的。
- **Logical hallucination**：模型产生了内部不一致的推理。"$3 + 4 = 8$ therefore..."

每种都有对应的 metrics：

**针对事实性**： SimpleQA （OpenAI, 2024）是目前最强的 benchmark —— 4326 个短答案事实问题，模型必须输出特定的实体、日期或数字。这个 benchmark 设计得让 bluffing 成本很高：评分只有"correct" / "incorrect" / "not attempted"，自信地答错比 abstentions 罚得更重。 2026 年的前沿模型得分在 30-55 %；剩下的部分是幻觉或者正确 abstained。

TruthfulQA （Lin et al., 2021, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*）针对更具体的失败模式：人类常持有错误信念的问题（"Can you cure cancer with [folk remedy]?"）。 Benchmark 测量模型是鹦鹉学舌重复错误信念，还是正确反驳。当 annotators 奖励 agreeable 答案时， RLHF 往往会*恶化* TruthfulQA 分数。

FEVER （Thorne et al., 2018, *FEVER: A Large-scale Dataset for Fact Extraction and VERification*）把事实性框定为验证任务：给定一个 claim，检索证据，分类为 supports / refutes / not enough info。作为 retrieval-then-verify 模式的 test bed 很有用。

**针对忠实度**： TruthfulQA, RAGAS faithfulness score, SelfCheckGPT （Manakul et al., 2023, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*）—— 后者检查同一 prompt 多次采样的一致性；如果模型在 hallucinating，重复采样往往会互相矛盾。 SelfCheckGPT 不需要 reference 或 retrieval，作为轻量级 monitor 跑在生产流量上很便宜。

RAG 的句子级忠实度：模型回答中的每一句话，能被检索上下文支持吗？

```python
# Conceptual RAGAS-style faithfulness check
def faithfulness(answer: str, context: str, judge_llm) -> float:
    sentences = split_into_sentences(answer)
    supported = 0
    for s in sentences:
        prompt = f"Context: {context}\n\nClaim: {s}\n\nIs the claim supported by the context? yes/no."
        if judge_llm(prompt).strip().lower().startswith("yes"):
            supported += 1
    return supported / len(sentences)
```

RAG 系统里， faithfulness score 低于 0.85 就是大问题；低于 0.7 说明模型基本在瞎编。

**针对逻辑性**：更难测。 Self-consistency （第 9 章）能 catch 住一部分 —— 采样 $N$ 条 chains，检查是否一致。 Programmatic verification （第 10 章）能 catch 住数学/代码逻辑。有个新兴技术， FActScore （Min et al., 2023, *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*），把长文本答案分解为 atomic facts 并独立验证 —— 这对传记、摘要以及任何全局准确性会掩盖局部错误的长文本输出很有用。

## 宪法AI (CAI)

![fig4: Constitutional AI loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig4_constitutional_ai.png)

Bai 等人（2022, *Constitutional AI: Harmlessness from AI Feedback*）。核心思路是用 AI 生成的偏好标签替代人工偏好标签，让 AI 对照一组书面原则（"the constitution"） critique 输出。

具体分两步走：

1. **SL-CAI**：生成 harmful prompts，生成模型响应，让模型对照原则 critique 自己的响应（"Did this response violate principle X?"），生成修订后的响应。在修订后的响应上训练（SFT）。
2. **RL-CAI / RLAIF**： pair-wise prompt 模型判断哪个响应更符合原则。把这些作为 preference pairs 用于 DPO/PPO。

Constitution 就是一串自然语言原则列表：

```
1. Choose the response that is most helpful to the user.
2. Choose the response that least encourages or assists in any
   form of crime, harm, or unethical activity.
3. Choose the response that least promotes any form of
   illegal discrimination.
... etc.
```

CAI 把单样本 $10 的人工偏好标注流水线换成了单样本 $0.001 的模型偏好流水线。安全轴上的质量相当甚至更好。 Anthropic 把这作为安全训练的主导 signal。

2023 年的后续研究（Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*）确认了 RLAIF 在摘要和 helpful 对话任务上能 match RLHF，但标注成本只是零头。 CAI 原则 + RLAIF 标签 + DPO 训练的组合，现在是前沿 labs 主导的安全流水线。

对大多数团队来说，全套 CAI 有点杀鸡用牛刀，但值得了解。轻量版 —— 用前沿模型对照书面 rubric 判断你自己模型的响应，然后基于判断做 DPO —— 既实用又有效。我合作过的很多生产团队会用 5-10 行 constitution， scope 限定在他们特定的部署风险上，这能实质性地提升 refusal 的 precision/recall。

## 水印与来源

这是个不一样的安全问题：你怎么知道一段文本是 LLM 生成的？ Watermarking 在输出中嵌入一个难以察觉的 signal，后续可以检测出来。

Kirchenbauer 等人（2023, *A Watermark for Large Language Models*）的方案：在每个 decode 步，根据前一个 token 的 hash 把 vocabulary  partition 成"green"和"red"集合。生成时 bias 向 green tokens。 Detector 读取文本， hash 每个 token 的前驱，计算 green-token 率。如果显著高于 50 %，文本就有 watermark。

优点：有 secret key 就能检测，只需 model-side （不需要 metadata）。

缺点： paraphrasing 会破坏它；质量成本是实打实的（aggressive watermarks 上 perplexity 增加 ~5-10 %）；只适用于 watermarking 方生成的文本。 2023 年的后续研究（Sadasivan et al., *Can AI-Generated Text be Reliably Detected?*）认为攻击者用相当的 LLM 资源通过 careful paraphrasing 就能移除 watermark，这给实际安全性设了个上限。

通过 metadata 的 provenance （图片的 C2PA 标准，文本也有类似演进标准）在你控制流水线时更可靠，但 copy-paste 后就没了。到了 2026 年， Google （SynthID）和其他几家已经 shipped watermarking； OpenAI 和 Anthropic 还没大规模部署。技术案例很强；但部署案例混合，因为 watermark 会 degrade 输出，客户会 complain。

## 从弱到强的泛化

关于下一代模型的对齐，有个不同的视角：当模型在某些领域变得 superhuman 时，我们可能无法准确标注偏好。怎么训练它们？

Burns 等人（2023, *Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision*）直接研究了这个问题。用较弱模型（GPT-2）生成偏好标签，然后在这些标签上训练较强模型（GPT-4）。强模型恢复了相当一部分相对于 ground truth 的能力 gap —— 它 generalizes 了弱 supervisor 的 intent，而不是 imitating 它的 mistakes。

结果还是初步的， methodology 也有局限，但它指向了 2025-2027 窗口期对齐面临的实际问题：当模型能力超过人类评估输出的能力（在数学、代码、科学领域）时，"alignment"在操作层面到底意味着什么？目前最好的答案是 scalable oversight protocols （debate, recursive reward modeling, AI safety via market），但还没有一个大规模 shipped。

对从业者来说，结论很明确：要对“我们跑了 RLHF 在我们能测的最安全的东西上”这种 claims 保持怀疑。如果你的 reward model 是人类在判断人类没资格判断的东西， optimization signal 就是 noise。
## 安全上线到底需要什么

![fig5: red-teaming workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig5_red_teaming_workflow.png)

要想安全地把 LLM 产品推上线，这份实操清单你得收好：

1. **Threat model**：写清楚你到底在防什么坏事。产品不同，威胁也不同。
2. **Refusal precision/recall test set**： 500-1000 条人工标注的 prompt，每次模型变动都要重测。
3. **Red-team session**：每季度一次，内部团队上阵，每个危害类别大概测 100 条 prompt。
4. **Output filter**：用审核模型（或者规则集）在内容触达用户前拦一道。 LLM 漏掉的，这里得 catch 住。
5. **Logging**：每次交互都要记：模型、 prompt、输出、延迟、成本。除了 PII 脱敏，其他信息得完整。
6. **Anomaly alerts**：盯着异常模式（比如拒绝率突然飙升、新型越狱尝试、单个用户 unexpectedly high-volume）。
7. **Human review path**：用户觉得“这回答有问题/有害”时能上报，并且有条快速通道直达工程团队。
8. **Rate limiting**：防止大规模自动化探测。
9. **Disclosure**：告诉用户他们在跟 AI 聊天，而且 AI 可能会犯错。
10. **Rollback ready**：旧模型版本得在一个 config flag 之外，随时能切回去。
11. **Action-space constraints**：如果你的产品带了工具，单次响应的最坏操作得有边界（没有确认就不能执行不可逆操作）。
12. **Vendor due diligence**：要是依赖 foundation-model API，得搞清楚供应商的安全实践、事故历史以及通知 SLAs。

这算不上什么完备的安全战略。但这是一个小团队能落地的方案，而且哪怕是 2026 年，大多数产品还没做到这些。

## 总结与后续

对齐其实就是三个目标（helpful, harmless, honest）加上第四个（controllable），它们之间互相 trade-off；你得想清楚自己要坐在哪个位置。 RLHF 会教模型学会你没想要的 sycophancy、啰嗦和过度自信；得通过 SFT 跑 counter-examples 和 reward-model auditing 来纠正。拒绝机制是由 residual stream 里的单个 feature direction 介导的——既浅又脆；得上层防御叠 layer。拒绝校准需要一个 labeled test set，里面既要有拒绝正确的案例，也要有拒绝错误的案例。红队要持续做，包括用自动化的 PAIR/TAP/GCG 风格攻击。幻觉分 factual, faithfulness 和 logical——每种都有不同的 metrics； SelfCheckGPT 和 FActScore 很实用， RAGAS 是专门针对 RAG 的工具。 Sleeper Agents 表明事后安全是有极限的——基础模型供应商的 pipeline hygiene 比你 fine-tuning 那一下更重要。 CAI 是个强大的模式，哪怕你不跑完整的 Anthropic recipe。上线时得带上一套虽小但真实的安全机制，别光靠意图。

下一章（终章）：**production**。详细聊聊 Serving stack 选型、 autoscaling、延迟预算、成本追踪、多模型 routing，以及从第一天起你就需要的 observability。

## 参考文献

- Askell, A. et al. (2021). *A General Language Assistant as a Laboratory for Alignment* (HHH framework). https://arxiv.org/abs/2112.00861
- Christiano, P. et al. (2017). *Deep Reinforcement Learning from Human Preferences*. https://arxiv.org/abs/1706.03741
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). https://arxiv.org/abs/2203.02155
- Sharma, M. et al. (2023). *Towards Understanding Sycophancy in Language Models*. Anthropic. https://arxiv.org/abs/2310.13548
- Pan, A. et al. (2022). *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*. ICLR 2022. https://arxiv.org/abs/2201.03544
- Casper, S. et al. (2023). *Open Problems and Fundamental Limitations of RLHF*. https://arxiv.org/abs/2307.15217
- Ganguli, D. et al. (2022). *Red Teaming Language Models to Reduce Harms*. Anthropic. https://arxiv.org/abs/2209.07858
- Chao, P. et al. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR). https://arxiv.org/abs/2310.08419
- Mehrotra, A. et al. (2024). *Tree of Attacks: Jailbreaking Black-Box LLMs Automatically*. https://arxiv.org/abs/2312.02119
- Chao, P. et al. (2024). *JailbreakBench*. https://arxiv.org/abs/2404.01318
- Hubinger, E. et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. Anthropic. https://arxiv.org/abs/2401.05566
- Arditi, A. et al. (2024). *Refusal in Language Models is Mediated by a Single Direction*. https://arxiv.org/abs/2406.11717
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic. https://arxiv.org/abs/2212.08073
- Lee, H. et al. (2023). *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*. https://arxiv.org/abs/2309.00267
- Lin, S. et al. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. https://arxiv.org/abs/2109.07958
- Manakul, P. et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*. EMNLP 2023. https://arxiv.org/abs/2303.08896
- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. https://arxiv.org/abs/2305.14251
- Thorne, J. et al. (2018). *FEVER: A Large-scale Dataset for Fact Extraction and VERification*. https://arxiv.org/abs/1803.05355
- Kirchenbauer, J. et al. (2023). *A Watermark for Large Language Models*. ICML 2023. https://arxiv.org/abs/2301.10226
- Sadasivan, V. et al. (2023). *Can AI-Generated Text be Reliably Detected?* https://arxiv.org/abs/2303.11156
- Burns, C. et al. (2023). *Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision*. OpenAI. https://arxiv.org/abs/2312.09390
- Anthropic (2024). *Frontier Red Team Report*. https://www.anthropic.com/research