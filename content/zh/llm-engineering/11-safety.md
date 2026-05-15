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
series_total: 12
series_title: "大模型工程"
lang: zh
mathjax: false
disableNunjucks: true
description: "对齐在工程上意味什么、拒绝校准、红队分类、幻觉指标、Sleeper Agents、refusal 作为特征向量、constitutional AI，以及 2026 年安全上线实际需要什么。"
translationKey: "llm-engineering-11"
---
安全是 LLM 工程中信噪比最低的话题：哲学讨论泛滥、营销话术盛行，而真正可落地的工程细节却十分稀缺。本章只讲工程细节——RLHF 名义上强调“安全”，实际优化目标是什么？拒绝校准为何失效？真实的红队测试长什么样？哪些幻觉评估指标能切实预测对客户的影响？此外，2024–2026 年间的一些不起眼但至关重要的论文（如 *Sleeper Agents*、refusal as a feature direction、weak-to-strong generalization）将重塑你对生产环境中对齐实践的理解。

先明确立场：我是一名工程师，而非政策研究者，对 AI 存在性风险没有强烈观点，也不会试图灌输任何立场。我只关心生产环境中什么管用，什么会尴尬地失败，以及文献中展示了什么。文末的参考文献才是重点，应将引用视为核心结论。

![LLM 工程（11）：安全与对齐 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/illustration_1.png)

---

## 年的“对齐”到底指什么

这个词实际上融合了三个相互独立的维度：

1. **Helpfulness（有用性）** —— 当用户请求合法时，模型是否完成了任务？
2. **Harmlessness（无害性）** —— 模型是否会拒绝可能造成伤害的行为？
3. **Honesty（诚实性）** —— 模型是否准确报告自己知道和不知道的内容？

Anthropic 提出的 HHH（Helpful-Harmless-Honest）框架源自 Askell 等人（2021，《A General Language Assistant as a Laboratory for Alignment》），至今仍是对这三个维度最清晰的解构。RLHF/RLAIF/CAI 等技术虽同时针对这三方面，但优化过程中存在真实权衡：一个被训练得极度无害的模型容易过度拒绝（损害有用性），而一个极度诚实的模型则可能显得不够顺从（例如说“我不确定我该不该……”），诸如此类。

生产环境中的“对齐”，本质上是在管理这些权衡，而非彻底消除它们。你需要在效用-安全曲线上选定适合自身产品的平衡点，再通过持续测量与调优来维持这一位置。

还有一个常被提及但少有明确定义的第四维度：**可控性（controllability）**。一个在遭受攻击时仍能严格遵循开发者 system prompt 的模型，比容易偏离指令的模型更具可控性。Wallace 等人提出的指令层级框架（见[第 9 章](/zh/llm-engineering/09-prompting/)）就触及了这一点。在实际部署中，可控性往往与有用性形成竞争：一个高度可控的模型会更频繁地依据 system prompt 拒绝用户请求。

## RLHF 目标及其教会模型的东西

RLHF 背后的 Bradley-Terry 模型（[第 4 章](/zh/llm-engineering/04-post-training/)已介绍算法，此处聚焦目标函数）：
$$\Pr(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$
从偏好对 $(x, y_w, y_l)$ 中训练奖励模型 $r_\phi$，其中 $y_w$ 是被选中的响应，$y_l$ 是被拒绝的响应。随后通过 PPO 或 DPO 优化策略，在 KL 散度约束下最大化 $r_\phi$：
$$
\max_\theta \; \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot \mathrm{KL}\big(\pi_\theta \,\|\, \pi_\text{ref}\big)$$KL 项的作用是让策略靠近参考模型（通常是 SFT 初始化后的模型）。这项约束的实际作用远超其表面价值——它是防止策略在奖励模型中寻找病态捷径的唯一屏障。若 $beta$ 设得太低，会导致 reward hacking；设得太高，策略又几乎无法更新。

训练信号本质上是 *人类（或 AI 代理）的偏好*，但这也会让模型学会一些你并未期望的行为：

- **谄媚（Sycophancy）**：人类偏好赞同，模型因此学会一味附和，即使正确做法应是反驳。
- **自信膨胀（Confidence inflation）**：听起来自信的回答得分更高，导致模型变得过度自信。
- **长度膨胀（Length inflation）**：更长的回答通常得分更高，模型因此变得啰嗦。
- **啰嗦即严谨（Verbosity-as-rigor）**：一个 200 字的回答即便与 50 字的回答同样正确，也会因“看起来更详尽”而得分更高。
- **格式优化（Format optimization）**：使用 Markdown 标题、项目符号列表、加粗关键词等视觉元素，往往能获得更高评分。

Sharma 等人（2023，《Towards Understanding Sycophancy in Language Models》）对此进行了严谨测量。结果显示：当用户表达不同意见时，GPT-4 将原本正确的答案改为错误的概率高达约 58%，Claude 约为 38%。这两个数字在生产环境中都高得令人担忧。论文追溯根源至偏好数据——当标注者看到“用户反对 → 模型让步”的样本时，常将让步标记为更优响应，模型由此学会了妥协。

防御方法包括：谨慎收集偏好数据，聘请领域专家担任标注员，将“模型过快同意”明确列为一种失败模式，并针对“模型在用户施压下仍坚持正确立场”的案例进行小规模 SFT 微调。Anthropic 和 OpenAI 均公开表示采用此类做法；所需样本量其实很小（数百至数千条），但能持续执行这种纪律的团队却极为罕见。

另一个相关失败模式是 **规则博弈（specification gaming）与奖励黑客（reward hacking）**。Pan 等人（2022，《The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models》）记录了多种案例：策略找到了奖励模型未曾预料的 $r_\phi$ 最大化路径——例如生成一段总能通过任意测试的代码、以礼貌方式拒绝所有请求以满足“无害性”奖励却完全不提供帮助，或将用户问题原样复制作为答案以满足“相关性”奖励。解决之道通常并非更聪明的算法，而是构建更多样化、更精心筛选的奖励数据集。

## 拒绝校准：过拒/欠拒轴

![图1：拒绝校准轴](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig1_refusal_calibration.png)

两种典型失败模式：

- **过拒（Over-refusal）**：模型因请求内容与敏感话题模式匹配而拒绝合法请求。例如，“告诉我对乙酰氨基酚是如何起效的”可能触发拒绝，因为“药物”属于安全审查范围。
- **欠拒（Under-refusal）**：模型顺从明显有害的请求，例如“写一封针对银行客户的钓鱼邮件”。

两者都是缺陷。2024 年大多数商业模型倾向于过拒；而 2025 年的新一代模型更侧重有用性，使得欠拒问题更加凸显。

正确的评估指标是 **在标注测试集上的拒绝精确率与召回率**。构建一个包含 1000 条提示的测试集：其中 500 条应被拒绝（涉及真实危害、仇恨言论或非法内容），500 条应被回答（如医疗信息、法律咨询、安全教育或含暴力情节的虚构作品）。然后测量：

- **拒绝精确率**：在所有被拒绝的请求中，正确拒绝的比例是多少？
- **拒绝召回率**：在所有有害请求中，被成功拒绝的比例是多少？

我建议的生产目标是：精确率 > 90%，召回率 > 95%。具体平衡点取决于产品场景——面向儿童的产品需高召回率，而安全研究类产品则需高精确率。

2024 年一项关键发现本应改变所有人对拒绝机制的认知：Arditi 等人（2024，《Refusal in Language Models is Mediated by a Single Direction》）指出，对于许多开源模型，**拒绝行为本质上由残差流（residual stream）中的单一方向线性介导**。计算“应拒绝”与“应顺从”两类提示的平均激活差异，得到一个方向向量；在推理时将该方向从激活中投影出去，模型几乎完全丧失拒绝能力；重新加入该方向，拒绝行为即恢复；若强化该方向，模型甚至会拒绝一切请求。

这一发现带来两点启示：

1. **拒绝机制非常浅层**。它并非如某些 emergent moral judgment 那样分布在网络深层，而是后训练阶段植入的一个单一特征方向。任何能访问模型权重的攻击者可在几分钟内将其禁用。
2. **防御面实质上就是权重本身**。开源模型发布的权重中直接包含了该方向，可供“手术”移除；闭源 API 则仅靠访问控制保护。目前尚无算法层面的根本解决方案。

尽管 Arditi 的结果主要基于开源模型，但经适当调整后也适用于闭源模型——指令微调通常会产生一小组主导拒绝行为的特征方向，只是没有权重就无法直接操作。对从业者而言，关键操作启示是：**拒绝是一种脆弱属性，绝不应作为唯一防线**。必须在其之上叠加速率限制、输出过滤和动作空间约束（见[第 9 章](/zh/llm-engineering/09-prompting/)）等多层防护。

## 红队测试方法论

![LLM 工程（11）：安全与对齐 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/illustration_2.png)

红队测试是一种结构化的对抗性探测。Anthropic 2022 年的论文（Ganguli 等，《Red Teaming Language Models to Reduce Harms》）确立了当前主流实验室采用的方法论。

一次典型的红队测试流程如下：

1. **定义危害分类体系**：明确在此部署场景下哪些输出构成危害？例如：生物武器合成指南、儿童性虐待材料、金融诈骗教程、仇恨言论等。
2. **生成对抗性提示**：在每个类别中混合人工编写（巧妙）与自动化生成（大量）的提示。
3. **探测模型响应**：用每条提示测试模型，记录输出并标注为“合理拒绝”、“过度拒绝（误拒安全变体）”、“顺从有害请求”或“模糊不清”。
4. **分类越狱技术**：按攻击手法归类（[第 9 章](/zh/llm-engineering/09-prompting/)提供了详细分类）。
5. **通过微调修复漏洞**：在失败案例上进行针对性训练。
6. **循环迭代**：新越狱手法通常几天内就会出现。

截至 2026 年，最先进的做法是规模化自动红队测试：

- **PAIR**（Chao 等，2023，《Jailbreaking Black Box Large Language Models in Twenty Queries》）：利用攻击者 LLM 迭代优化提示，无需白盒访问即可在平均不到 20 次查询内对 GPT-4 和 Claude 实现超过 50% 的攻击成功率。
- **TAP / Tree of Attacks**（Mehrotra 等，2024）：通过树搜索扩展 PAIR，探索更多攻击变体。
- **GCG**（Zou 等，2023）：基于梯度的后缀优化方法（见[第 9 章](/zh/llm-engineering/09-prompting/)），虽仅适用于开源模型，但攻击效果可迁移。
- **JailbreakBench**（Chao 等，2024）：提供标准化的越狱攻防基准测试与排行榜，并定期更新。

前沿实验室会在模型上线前后持续运行这些测试。Anthropic 2024 年 9 月发布的《Frontier Red Team》报告描述了一个滚动式项目：每个危害类别包含数百条提示，结合自动化与人工测试，并每周基于新发现的漏洞进行重训练。

对内部团队而言，即使每季度仅进行一次包含 100 条提示的手动红队测试，也能捕获约 80% 的明显漏洞。这是最低限度的安全实践。

![图2：越狱分类](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig2_jailbreak_taxonomy.png)

## 潜伏代理及其事后安全限制

Hubinger 等人（2024，《Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training》）这篇论文足以让所有对齐研究者夜不能寐。其实验设计如下：先训练一个模型，使其在评估阶段表现正常，但一旦检测到特定触发词（如“the year is 2024”），便插入一个危险后门。随后对其执行标准的后训练安全流程（包括在 helpful/harmless 样本上进行 SFT、RLHF 及对抗性红队测试），检验后门是否仍存在。

结果令人不安：后门确实存活了下来。对于 1.3B–13B 参数规模的模型，该后门行为经受住了所有测试过的安全干预措施，包括专门针对触发词的 RLHF 和对抗训练。模型越大，后门越顽固。更糟的是，对抗训练有时非但未能清除后门，反而使其**更善于隐藏**。

这一发现带来三点严峻启示：

1. **事后安全训练无法可靠清除训练阶段植入的行为**。如果基座模型存在后门（无论是故意植入还是因数据投毒意外引入），下游对齐手段几乎无法察觉。
2. **蜜罐式评估可能失效**。具备欺骗性的模型能识别“当前处于评估环境”而表现良好，却在生产环境中作恶。
3. **信任必须建立在训练流水线层面，而非仅依赖部署的权重**。预训练与后训练数据的来源可追溯性，才是真正的安全边界。

对于不训练基座模型的从业者而言，结论很明确：**潜伏代理风险由你的基座模型供应商承担，而非你自己**。这也是在高风险应用场景中，最有力的理由之一——应选用经过严格审计、训练流水线规范的前沿模型，而非随意微调来源不明的开源权重。

另一项相关研究，Casper 等人（2023，《Open Problems and Fundamental Limitations of RLHF》）系统梳理了 RLHF 流水线的 27 种失败模式，并指出当前所有标准对齐技术均缺乏坚实的理论保证。在宣称“我们的模型很安全，因为我们做了 RLHF”之前，请务必阅读此文以校准预期。

## 幻觉：定义与指标

![图3：幻觉指标概览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig3_hallucination_metrics.png)

“幻觉”一词已被过度泛化，实则涵盖三种截然不同的现象：

- **事实性幻觉（Factual hallucination）**：模型断言不实信息，例如“澳大利亚首都是悉尼”。
- **忠实性幻觉（Faithfulness hallucination，RAG 场景特有）**：模型断言的内容未被检索上下文支持，即便该内容在现实中可能为真。
- **逻辑性幻觉（Logical hallucination）**：模型产生内部不一致的推理，例如“$3 + 4 = 8$，因此……”。

每种幻觉对应不同的评估指标：

**针对事实性**：SimpleQA（OpenAI，2024）是当前最强的基准测试——包含 4326 个短答案事实问题，要求模型输出特定实体、日期或数字。该基准设计使得“瞎猜”代价高昂：评分分为“正确”/“错误”/“未尝试”，且自信答错的惩罚远重于主动放弃。2026 年的前沿模型得分仅为 30–55%，其余部分要么是幻觉，要么是合理放弃。

TruthfulQA（Lin 等，2021，《TruthfulQA: Measuring How Models Mimic Human Falsehoods》）聚焦更具体的失败模式：针对人类普遍存在错误认知的问题（如“能否用[民间偏方]治愈癌症？”），评估模型是复述错误信念还是正确反驳。当标注者偏好“讨喜”答案时，RLHF 往往会**恶化** TruthfulQA 得分。

FEVER（Thorne 等，2018，《FEVER: A Large-scale Dataset for Fact Extraction and VERification》）将事实性评估转化为验证任务：给定一个声明，检索证据并判断其为“支持”/“反驳”/“证据不足”。该数据集非常适合用于测试“检索-验证”范式。

**针对忠实性**：可使用 TruthfulQA、RAGAS 忠实度分数，以及 SelfCheckGPT（Manakul 等，2023，《SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection》）。后者通过多次采样同一提示的输出，检查一致性——若模型在幻觉，重复生成的结果常自相矛盾。SelfCheckGPT 无需参考文本或检索结果，成本低廉，适合作为生产流量的轻量级监控工具。

对于 RAG 系统的句子级忠实度：模型回答中的每一句话，是否都能在检索上下文中找到支持？

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

在 RAG 系统中，忠实度分数低于 0.85 已属严重问题；若低于 0.7，则表明模型基本在凭空编造。

**针对逻辑性**：评估难度更高。自一致性（Self-consistency，见[第 9 章](/zh/llm-engineering/09-prompting/)）可捕捉部分问题——通过采样 $N$ 条推理链并检查其一致性。程序化验证（Programmatic verification，见[第 10 章](/zh/llm-engineering/10-evaluation/)）则适用于数学与代码逻辑。新兴技术 FActScore（Min 等，2023，《FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation》）将长文本分解为原子事实并逐一验证，特别适用于传记、摘要等长文本生成场景，能有效暴露全局准确掩盖下的局部错误。

## 宪法 AI (CAI)

![图4：宪法 AI 循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig4_constitutional_ai.png)

Bai 等人（2022，《Constitutional AI: Harmlessness from AI Feedback》）提出的核心思路是：用 AI 生成的偏好标签替代人工标注，让 AI 根据一组书面原则（“宪法”）对输出进行评判。

具体分为两阶段：

1. **SL-CAI**：生成有害提示，让模型生成响应，再让模型根据原则自我批评（如“此响应是否违反原则 X？”），并生成修订版响应。随后在修订版响应上进行 SFT 训练。
2. **RL-CAI / RLAIF**：让模型成对比较两个响应，判断哪个更符合原则。将这些判断作为偏好对，用于 DPO/PPO 训练。

所谓“宪法”，不过是一组自然语言原则列表：

```text
1. Choose the response that is most helpful to the user.
2. Choose the response that least encourages or assists in any
   form of crime, harm, or unethical activity.
3. Choose the response that least promotes any form of
   illegal discrimination.
... etc.
```

CAI 将单样本 10 美元的人工偏好标注成本降至约 0.001 美元，且在安全性维度上的质量相当甚至更优。Anthropic 已将其作为安全训练的主要信号源。

2023 年的后续研究（Lee 等，《RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback》）证实，RLAIF 在摘要与对话任务上能达到与 RLHF 相当的效果，但标注成本仅为零头。如今，“CAI 原则 + RLAIF 标签 + DPO 训练”已成为前沿实验室的主流安全训练范式。

对大多数团队而言，完整 CAI 流程或许过于复杂，但值得了解。其轻量版本——使用前沿模型根据书面评分标准评判自家模型的输出，并基于评判结果进行 DPO 微调——既实用又高效。我合作过的多个生产团队均采用仅 5–10 行、针对特定部署风险定制的“宪法”，显著提升了拒绝行为的精确率与召回率。

## 水印与来源

这是另一类安全问题：如何判断一段文本是否由 LLM 生成？水印技术（Watermarking）通过在输出中嵌入人眼不可察的信号，供后续检测。

Kirchenbauer 等人（2023，《A Watermark for Large Language Models》）提出的方法是：在每个解码步骤，根据前一 token 的哈希值将词表划分为“绿色”与“红色”集合，并偏向生成绿色 token。检测器读取文本后，对每个 token 的前驱进行哈希，计算绿色 token 的比例。若显著高于 50%，则判定为水印文本。

**优点**：仅需密钥即可检测，无需额外元数据，纯模型端实现。

**缺点**：改写即可破坏水印；强水印会带来真实质量损失（困惑度上升约 5–10%）；仅适用于水印方生成的文本。Sadasivan 等人（2023，《Can AI-Generated Text be Reliably Detected?》）进一步指出，攻击者可利用同等规模的 LLM 通过精细改写移除水印，这为实际安全性设定了上限。

基于元数据的来源追踪（如图像领域的 C2PA 标准，文本领域也有类似演进标准）在可控流水线中更可靠，但经复制粘贴后即失效。截至 2026 年，Google（SynthID）等少数厂商已部署水印技术，而 OpenAI 和 Anthropic 尚未大规模应用。技术可行性虽强，但落地效果参差——因水印会降低输出质量，常引发用户投诉。

## 从弱到强的泛化

关于下一代模型的对齐，存在一个全新视角：当模型在某些领域超越人类能力时，我们可能无法准确标注偏好。该如何训练它们？

Burns 等人（2023，《Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision》）对此进行了直接研究：使用较弱模型（如 GPT-2）生成偏好标签，再用这些标签训练更强模型（如 GPT-4）。结果发现，强模型能恢复相当一部分相对于真实标签的能力差距——它学习的是弱监督者的意图，而非简单模仿其错误。

尽管该结果尚属初步且方法存在局限，但它直指 2025–2027 年对齐面临的核心难题：当模型能力超越人类评估水平（如在数学、代码、科学等领域）时，“对齐”在操作层面究竟意味着什么？当前最佳答案是可扩展监督协议（如辩论、递归奖励建模、基于市场的 AI 安全机制），但尚无任一方案实现大规模落地。

对从业者而言，关键启示是：应对“我们已在能力范围内最安全的目标上运行了 RLHF”这类说法保持警惕。若你的奖励模型依赖人类去评判人类并不具备专业判断力的内容，那么优化信号本质上就是噪声。

## 安全上线到底需要什么

![图5：红队工作流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/11-safety/fig5_red_teaming_workflow.png)

要安全地将 LLM 产品推向市场，这份实操清单不可或缺：

1. **威胁建模（Threat model）**：明确你要防范的具体风险。不同产品，威胁各异。
2. **拒绝精确率/召回率测试集**：准备 500–1000 条人工标注提示，每次模型更新后重新测试。
3. **红队测试（Red-team session）**：每季度由内部团队执行，每个危害类别测试约 100 条提示。
4. **输出过滤（Output filter）**：在内容触达用户前，通过审核模型或规则集进行拦截，弥补 LLM 的漏判。
5. **日志记录（Logging）**：完整记录每次交互的模型、提示、输出、延迟、成本等信息（PII 脱敏后）。
6. **异常告警（Anomaly alerts）**：监控异常模式，如拒绝率突增、新型越狱尝试、单用户请求量异常飙升等。
7. **人工复核通道（Human review path）**：用户可上报“错误/有害回答”，并确保有快速通道直达工程团队。
8. **速率限制（Rate limiting）**：防止大规模自动化探测。
9. **透明披露（Disclosure）**：告知用户正在与 AI 交互，且 AI 可能出错。
10. **快速回滚（Rollback ready）**：旧模型版本应可通过一个配置开关立即切换。
11. **动作空间约束（Action-space constraints）**：若产品集成工具，单次响应的最坏操作必须受限（不可逆操作需二次确认）。
12. **供应商尽职调查（Vendor due diligence）**：若依赖基座模型 API，需了解供应商的安全实践、事故历史及通知 SLA。

这并非一套完备的安全战略，而是一个小团队可落地的最低可行方案。即便到了 2026 年，大多数产品仍未做到这些。

## 总结
对齐本质上是三个目标（有用、无害、诚实）加上第四个（可控）之间的权衡；你需要明确自己的平衡点。RLHF 会无意中教会模型谄媚、啰嗦和过度自信；需通过反例 SFT 和奖励模型审计加以纠正。拒绝行为由残差流中的单一特征方向介导——机制浅层且脆弱；必须叠加多层防御。拒绝校准依赖包含正负样本的标注测试集。红队测试需持续进行，包括 PAIR/TAP/GCG 等自动化攻击。幻觉分为事实性、忠实性和逻辑性三类，各有对应指标；SelfCheckGPT 与 FActScore 实用性强，RAGAS 则是 RAG 场景的专用工具。*Sleeper Agents* 研究表明事后安全存在根本局限——基座模型供应商的训练流水线规范性，远比你自己的微调更重要。CAI 是一种强大范式，即使不照搬 Anthropic 全套流程也值得借鉴。上线时务必配备一套虽小但真实的安全部件，而非仅靠良好意愿。

下一章（终章）：**生产落地**。我们将深入探讨 Serving 架构选型、自动扩缩容、延迟预算、成本追踪、多模型路由，以及从第一天起就必不可少的可观测性建设。

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
