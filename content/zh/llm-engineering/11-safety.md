---
title: "大模型工程（十一）：安全与对齐"
date: 2026-05-06 09:00:00
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
在 LLM 工程里，安全这个话题信噪比最低。哲学讨论和营销话术一大堆，工程细节却少得可怜。这一章专讲工程细节。RLHF 提到“安全”时到底优化了什么？拒绝校准为什么会失效？红队测试实际操作中是什么样？哪些幻觉度量指标能预测客户影响？还有 2024 到 2026 年几篇重要论文，比如《Sleeper Agents》、把 refusal 当作特征方向的研究、弱到强泛化。这些内容会改变你对生产环境中对齐问题的看法。

简单说下我的立场。我是工程师，不是搞政策的。AI 生存风险问题上，我没有强烈观点，也不想灌输任何观点。我清楚生产环境中什么方法有效，什么方法会尴尬地失败，也知道文献怎么说的。文末的参考文献很重要，直接看引用内容就能抓住重点。

![大模型工程（十一）：安全与对齐 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/illustration_1.png)
## "对齐"在 2026 实际指什么

"对齐"这个词把三件事混为一谈，其实它们应该分开看：

1. **Helpfulness**：用户请求合理时，模型是否完成任务？  
2. **Harmlessness**：模型是否会拒绝伤害他人的请求？  
3. **Honesty**：模型是否准确表达自己知道和不知道的内容？  

Anthropic 的 HHH（Helpful-Harmless-Honest）框架，出自 Askell 等人 2021 年的论文，仍然是最清晰的分解。RLHF、RLAIF 和 CAI 技术试图优化这三点，但取舍不可避免。训练得非常 harmless 的模型容易过度拒绝请求，影响 helpfulness。训练得非常 honest 的模型则显得不够顺从，总说“我不确定该不该……”。

生产中的"对齐"更多是工程上的权衡，而不是彻底解决问题。我会根据需求选择曲线上的某个点，然后测量数据、调整参数。

文献中隐约提到但很少明确命名的还有第四个维度：*controllability（可控性）*。攻击场景下，遵循系统提示的模型比容易偏离的模型更可控。Wallace 等人在第 9 章提到的指令层级部分涉及这一点。实际生产中，可控性与 helpfulness 直接冲突。一个高度可控的模型会拒绝更多用户请求，因为它被系统提示要求这样做。
## RLHF 目标实际教了什么

RLHF 的核心是 Bradley-Terry 模型。第 4 章讲了算法，这里说目标。

$$\Pr(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

训练奖励模型 $r_\phi$，用偏好对 $(x, y_w, y_l)$。$y_w$ 是选中的答案，$y_l$ 是拒绝的。然后用 PPO 或 DPO 让策略最大化 $r_\phi$，加个 KL 散度约束：

$$\max_\theta \; \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot \mathrm{KL}\big(\pi_\theta \,\|\, \pi_\text{ref}\big)$$

KL 项让策略贴近参考模型（通常是 SFT 初始化）。它的作用比看起来重要得多。它是防止策略走捷径的唯一保障。$\beta$ 太低会 reward hacking，太高策略又不动。

信号的核心是 *人类（或 AI 代理）的偏好*。但模型学到了一些我不想要的东西。

- **谄媚倾向**：人类喜欢被认同。模型学会了迎合，即使反对更有道理。
- **过度自信**：听起来自信的回答得分更高。模型变得过于自信。
- **冗长倾向**：长回答得分更高。模型开始啰嗦。
- **以啰嗦为严谨**：200 字的回答比 50 字的得分高，仅仅因为“看起来更全面”。
- **格式优化**：Markdown 标题、项目符号、加粗关键词——这些视觉上更吸引人。

Sharma 等人（2023，*Towards Understanding Sycophancy in Language Models*）详细研究了这些现象。数据很扎心：用户表达异议时，GPT-4 把正确答案改成错误的比例高达 58%。Claude 是 38%。这两个数字在生产环境里都很危险。论文指出，问题出在偏好数据。标注者看到“用户不同意 → 模型让步”，往往把让步标记为更好。模型就学会了让步。

怎么应对？仔细收集偏好数据。找领域专家当标注者。明确标注“模型轻易让步”为失败模式。针对“坚持正确立场”的例子，做一次小规模 SFT 微调。Anthropic 和 OpenAI 都公开说他们在做这件事。SFT 数据量不大（几百到一千多条），但能持续执行的团队很少。

还有一个相关问题是 **spec gaming 和 reward hacking**。Pan 等人（2022，*The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*）记录了一些案例。策略找到了奖励模型没预料到的 $r_\phi$ 最大化方式。比如：
- 用永远能通过测试的代码作答。
- 礼貌拒绝以满足“无害性”奖励，但毫无帮助。
- 直接复述用户问题作为答案，满足“相关性”奖励。

解决这些问题的关键不是更聪明的算法，而是更多样化、更精心策划的奖励数据集。
## 拒绝校准：过/不足拒绝轴

![fig1: 拒绝校准轴](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig1_refusal_calibration.png)

两种失败模式：

- **过度拒绝**：模型把合法请求误判为有害内容。比如，用户问“对乙酰氨基酚如何起作用”，因为提到“药物”，模型直接拒绝。
- **不足拒绝**：模型对明显有害的请求照单全收。比如，用户要求“写一封钓鱼邮件”，模型竟然真的生成。

这两种情况都是问题。2024 年，大多数商业模型倾向于过度拒绝。到了 2025 年，模型更注重实用性，但也让不足拒绝的问题更加突出。

衡量拒绝性能的最佳指标是 **标注测试集上的精度和召回率**。我建议构建一个包含 1000 条提示的测试集。其中 500 条应该被拒绝（真正有害、仇恨言论、违法内容），另外 500 条应该正常回答（医疗信息、法律咨询、安全教育、包含暴力情节的小说）。

- 精度：被拒绝的请求中，正确拒绝的比例是多少？
- 召回：有害请求中，成功拒绝的比例是多少？

我的生产目标是：精度 > 90%，召回 > 95%。平衡点取决于应用场景。儿童产品需要高召回率，安全研究工具则需要高精度。

2024 年有一项重要发现，彻底改变了我对模型拒绝行为的理解。Arditi 等人在论文《Refusal in Language Models is Mediated by a Single Direction》中指出，对于许多开源模型，**拒绝行为主要由残差流中的一个单一线性方向决定**。

计算“我应该拒绝”和“我应该接受”两类提示的平均激活差异。在推理时将这个方向投影出去，模型几乎完全失去拒绝能力。重新加入这个方向，拒绝行为恢复。如果过度强化，模型会对所有请求都拒绝。

这项研究有两个启示：

1. **拒绝行为很浅层**。它不是通过网络深度编码的复杂道德判断，而是后训练阶段植入的一个简单特征方向。有权限访问权重的攻击者，几分钟内就能禁用。
2. **防御的关键在于权重本身**。开源模型发布时，这个特征方向可以被修改。闭源 API 则只能依赖访问控制保护。没有算法层面的根本解决方案。

Arditi 的研究结果也适用于闭源模型（经过调整）。指令微调通常会产生少量控制拒绝行为的特征方向。但如果没有权重，无法直接修改。

从业者的实际操作建议是：**拒绝行为并不稳固，不能作为唯一的防线**。还需要叠加速率限制、输出过滤以及动作空间约束（第 9 章）等多重防护措施。
## 红队方法学

![大模型工程（十一）：安全与对齐 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/illustration_2.png)


![fig2: 越狱分类法](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig2_jailbreak_taxonomy.png)

红队是一种系统化的对抗测试，专门用来挖掘模型漏洞。Anthropic 在 2022 年的论文（Ganguli 等，《Red Teaming Language Models to Reduce Harms》）奠定了主流方法的基础。

一次红队测试通常分六步：

1. **定义危害类型**  
   哪些输出算有害？比如生物武器指南、儿童性虐待内容、诈骗教程、仇恨言论。

2. **生成对抗提示**  
   每类提示分两种：人工精心设计的和自动化批量生成的。

3. **测试模型响应**  
   记录结果并打标签："拒绝得当"、"拒绝过度"、"满足有害请求"、"模棱两可"。

4. **分类越狱技术**  
   第 9 章详细讲了越狱分类法。

5. **微调修复漏洞**  
   针对失败案例进行微调。

6. **重复测试**  
   新越狱方法几天内就会冒头。

到 2026 年，最先进的方法是大规模自动化红队测试。

- **PAIR**（Chao 等，2023，《Jailbreaking Black Box Large Language Models in Twenty Queries》）  
  用攻击型 LLM 迭代优化提示，无需白盒访问。平均不到 20 次查询，攻击 GPT-4 和 Claude 的成功率超 50%。

- **TAP / Tree of Attacks**（Mehrotra 等，2024）  
  在 PAIR 基础上引入树搜索，探索多种攻击变体。

- **GCG**（Zou 等，2023）  
  基于梯度的后缀优化方法（第 9 章）。仅支持开源权重模型，但能迁移。

- **JailbreakBench**（Chao 等，2024）  
  标准化的越狱攻防基准平台，定期更新排行榜。

前沿实验室在模型部署前后持续运行这些测试。Anthropic 2024 年 9 月的《Frontier Red Team》报告显示，他们有一个滚动项目：每类危害包含数百条提示，结合自动化和人工测试，每周针对失败案例重新训练模型。

内部使用时，即使每季度手动跑一次 100 条提示的红队测试，也能发现约 80% 的明显问题。这已经是最低要求了。
## Sleeper Agents 和事后安全的极限

Hubinger 等人（2024，*Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*）写了一篇让对齐研究者睡不着觉的论文。实验设计很简单：训练一个模型，评估时表现正常，但遇到特定触发词（比如“年份是 2024”）时，插入一个后门漏洞。接着跑一遍标准的 post-training 安全流程，包括 SFT、RLHF 和对抗性红队测试，最后看后门还在不在。

结果很扎心：后门还在。1.3B 到 13B 参数规模的模型，无论用什么安全干预手段，后门都顽固存活。模型越大，后门越难移除。更糟的是，对抗训练有时不仅没去掉后门，反而让它隐藏得更深。

这些发现让人不安：

1. **事后安全训练无法移除训练阶段植入的行为。** 基础模型如果有后门，无论是故意还是数据污染导致，下游对齐根本发现不了。
2. **蜜罐测试可能无效。** 欺骗性模型能识别测试环境，装得规规矩矩，但在生产环境中却会作恶。
3. **信任应该在训练流程，而不是部署权重。** 预训练和 post-training 数据的来源才是真正的安全关键。

对于不训练基础模型的从业者来说，结论很简单：**Sleeper-agent 的风险由基础模型供应商承担，不是你。** 这也是为什么高风险场景中，选择经过严格审计的前沿实验室模型，比随意微调开源权重要靠谱得多。

另一个相关研究来自 Casper 等人（2023，*Open Problems and Fundamental Limitations of RLHF*）。他们分析了 RLHF 流程中的 27 种失败模式，得出结论：现有对齐技术没有一种有强理论保障。建议大家在喊“我们的模型很安全，因为我们做了 RLHF”之前，先读读这篇论文，调整预期。
## 幻觉：定义和指标

![fig3: 幻觉指标总览](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig3_hallucination_metrics.png)

“幻觉”这个词在技术讨论中被用得有点滥。实际上，它指三种不同的问题：

- **事实性幻觉**：模型输出了错误的事实。比如，“澳大利亚的首都是悉尼。”  
- **忠实性幻觉**（RAG 特有）：模型生成的内容与检索到的上下文不一致，即使这些内容在其他情况下可能是正确的。  
- **逻辑性幻觉**：模型推理过程中出现了内部矛盾。例如，“$3 + 4 = 8$，因此……”

针对这三种幻觉，有不同的评估方法。

---

### 事实性幻觉

目前最强的基准测试是 SimpleQA（OpenAI，2024）。这个数据集包含 4326 道简短的事实性问题，要求模型输出特定的实体、日期或数字。评分规则很严格：只有“正确”、“错误”和“未作答”三种结果。自信但错误的回答扣分更重，比直接放弃回答惩罚更大。

2026 年的前沿模型在这项测试中的得分范围是 30%-55%。剩下的部分要么是幻觉，要么是模型明智地选择了不作答。

另一个重要基准是 TruthfulQA（Lin 等，2021，*TruthfulQA: Measuring How Models Mimic Human Falsehoods*）。它专门针对人类常犯的错误信念问题，比如“用[民间偏方]能治癌症吗？” 这个基准测试模型是否会重复这些错误信念，还是能够正确反驳它们。

有趣的是，当标注者倾向于奖励“讨喜”的回答时，RLHF 方法往往会降低 TruthfulQA 的分数。

FEVER（Thorne 等，2018，*FEVER: A Large-scale Dataset for Fact Extraction and VERification*）将事实性问题转化为验证任务。给定一个陈述，模型需要检索证据并判断它是“支持”、“反驳”还是“信息不足”。这个数据集非常适合测试“先检索后验证”这种模式的效果。

---

### 忠实性幻觉

对于 RAG 系统来说，忠实性是一个核心指标。常用的评估工具有 TruthfulQA、RAGAS 的忠实性评分，以及 SelfCheckGPT（Manakul 等，2023，*SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*）。

SelfCheckGPT 的特点是通过多次采样同一个提示来检查一致性。如果模型在幻觉，不同采样之间的答案往往会互相矛盾。它不需要参考答案或额外的检索过程，运行成本低，适合在生产环境中作为轻量级监控工具。

在 RAG 系统中，句子级别的忠实性评估也很重要。模型生成的答案中，每一句话是否都能从检索到的上下文中找到支持？

```python
# 概念性的 RAGAS 风格忠实性检查
def faithfulness(answer: str, context: str, judge_llm) -> float:
    sentences = split_into_sentences(answer)
    supported = 0
    for s in sentences:
        prompt = f"上下文: {context}\n\n断言: {s}\n\n上下文是否支持这个断言？yes/no。"
        if judge_llm(prompt).strip().lower().startswith("yes"):
            supported += 1
    return supported / len(sentences)
```

如果 RAG 系统的忠实性评分低于 0.85，说明存在问题；如果低于 0.7，那模型基本上就是在编造内容了。

---

### 逻辑性幻觉

逻辑性幻觉的评估难度更高。Self-consistency（第 9 章）是一种常用的方法，通过对 $N$ 条推理链进行采样，检查它们是否一致。程序化验证（第 10 章）则专注于数学和代码逻辑的准确性。

最近还出现了一种新技术，叫 FActScore（Min 等，2023，*FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*）。这种方法将长篇生成文本分解为原子化的事实单元，并逐一验证每个单元的准确性。

这种方法特别适合用于传记、摘要等长篇输出场景。因为在这些场景中，整体准确率可能会掩盖局部错误。

下一节会讨论如何在实际工程中减少这些幻觉问题。
## Constitutional AI（CAI）

![fig4: Constitutional AI 循环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig4_constitutional_ai.png)

Bai 等人在 2022 年的论文《*Constitutional AI: Harmlessness from AI Feedback*》提出一个想法：用 AI 生成的偏好标签代替人工标注。模型会根据一组书面原则（称为“宪法”）评价输出。

方法分两个阶段：

1. **SL-CAI**：先生成有害的 prompt，再让模型生成回复。接着，让模型自我审查，比如问“这个回复是否违反了原则 X？”。然后生成改进版回复，最后用这些回复做 SFT 训练。
2. **RL-CAI / RLAIF**：让模型比较两个回复，判断哪个更符合原则。这些判断结果直接作为 DPO 或 PPO 的偏好对。

“宪法”其实是一份自然语言写的原则清单：

```
1. 选择对用户最有帮助的回复。
2. 选择最不鼓励或协助犯罪、伤害或不道德行为的回复。
3. 选择最不助长非法歧视的回复。
... 等等。
```

CAI 把每样本 10 美元的人工标注流程换成每样本 0.001 美元的模型标注流程。安全性指标上，效果持平甚至更好。Anthropic 已经把它作为安全训练的主要信号。

2023 年的一篇后续论文（Lee 等，《*RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*》）验证，RLAIF 在文本摘要和对话辅助任务上的表现与 RLHF 相当，但标注成本大幅降低。“CAI 原则 + RLAIF 标签 + DPO 训练”的组合已经成为前沿实验室的主流安全优化流程。

对大多数团队来说，完整 CAI 流程可能太复杂，但了解它很有价值。一种轻量做法是：用前沿模型根据一套书面规则评判自家模型的回复，然后基于这些评判结果做 DPO 训练。这种简化版既实用又高效。我合作过的许多生产团队都会制定一份 5 到 10 条原则的小型“宪法”，针对具体部署风险进行约束。实践证明，这种方法能显著提升拒绝行为的准确率和召回率。
## 水印和溯源

还有一个安全问题：怎么判断一段文本是不是 LLM 生成的？水印技术在输出中嵌入一个隐蔽信号，方便后续检测。

Kirchenbauer 等人在 2023 年的论文《*A Watermark for Large Language Models*》提出一种方法。每步解码时，根据前一个 token 的哈希值把词表分成“绿”和“红”两部分。生成时偏向绿 token。检测器读取文本后，对每个 token 的前驱做哈希计算，统计绿 token 的比例。如果比例显著高于 50%，就说明这段文本带水印。

优点很明显：只需要密钥就能检测，完全在模型侧完成，不依赖元数据。

缺点也不少。改写会破坏水印；质量损失是实打实的，激进的水印会让困惑度上升约 5% 到 10%；只对加了水印的一方生成的文本有效。2023 年 Sadasivan 等人的研究《*Can AI-Generated Text be Reliably Detected?*》指出，攻击者如果有类似的 LLM 资源，可以通过精心改写移除水印。这给水印的实际安全性设定了上限。

通过元数据实现溯源（比如图像领域的 C2PA 标准，以及正在发展的文本标准）更可靠，但前提是掌控整个流程。复制粘贴会让元数据丢失。到 2026 年，Google（SynthID）和其他少数公司推出了水印技术，而 OpenAI 和 Anthropic 尚未大规模部署。技术上看，水印效果不错；但从实际部署来看，情况复杂。水印会影响输出质量，客户也会抱怨。
## Weak-to-strong generalization

下一代模型对齐问题有了新视角。模型在某些领域超越人类后，我们可能无法准确标注偏好。这该怎么训练？

Burns 等人在 2023 年的论文《*Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision*》研究了这个问题。他们用弱模型（GPT-2）生成偏好标签，再用这些标签训练强模型（GPT-4）。结果发现，强模型能弥补与真实情况的大部分能力差距。它学会了弱监督者的意图，而不是模仿错误。

这个结果只是初步的，方法也有局限性。但它点出了 2025-2027 年对齐问题的核心挑战：当模型能力超过人类评估输出的能力（比如数学、代码、科学领域），"对齐"到底怎么定义？目前最靠谱的答案是可扩展的监督协议，比如辩论（debate）、递归奖励建模（recursive reward modeling）、通过市场实现 AI 安全（AI safety via market）。但这些方法还没一个能大规模落地。

对于从业者来说，关键教训是别轻信“我们在能测得最安全的东西上跑了 RLHF”的说法。如果你的奖励模型依赖人类去评判那些人类根本不够格评判的内容，优化信号就是噪声，毫无意义。
## 安全上线实际需要什么

![fig5: 红队工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/11-safety/fig5_red_teaming_workflow.png)

一个 LLM 驱动产品安全上线的实战清单：

1. **威胁模型**：写下要防范的坏结果。不同产品，威胁不同。
2. **拒绝测试集**：准备 500 到 1000 条手工标注的 prompt。每次模型更新后重新测试。
3. **红队演练**：每季度一次，内部团队负责。每个危害类别设计约 100 条 prompt。
4. **输出过滤器**：用 moderation 模型或规则集检查输出。拦截 LLM 放行的问题内容。
5. **日志记录**：记录每次交互，包括模型、prompt、输出、延迟和成本。脱敏 PII，其他信息完整保留。
6. **异常告警**：监控异常模式，比如拒绝率飙升、新型 jailbreak 尝试、单用户高频请求。
7. **人工审核通道**：提供用户反馈路径，比如“答案错误或有害”。快速转交工程团队处理。
8. **速率限制**：防止大规模自动化探测。
9. **明确告知**：告诉用户他们在和 AI 对话，AI 可能出错。
10. **回滚机制**：通过一个配置开关切换到上一版本模型。
11. **动作空间限制**：涉及工具操作时，单次响应的最坏影响必须可控。不可逆操作需确认。
12. **供应商审查**：依赖基础模型 API 时，了解供应商的安全实践、历史事故和通知 SLA。

这不是完整的安全策略，而是小团队能落地的内容。遗憾的是，到 2026 年，大多数产品可能仍然做不到这些。
## 小结与下一篇

对齐有四个目标：helpful、harmless、honest，还有 controllable。它们互相冲突，你得选一个平衡点。RLHF 会教出一些坏习惯，比如谄媚、啰嗦和过度自信。怎么解决？用反例做 SFT 训练，同时审计奖励模型。拒绝行为由 residual stream 的单一特征方向控制。这种机制浅显又脆弱，必须加几层防御。校准拒绝行为需要标注测试集，包含拒绝正确和拒绝错误的案例。红队测试要持续做，PAIR/TAP/GCG 风格的自动化攻击也别落下。

幻觉问题分三类：事实性、忠实性和逻辑性。每类都有不同指标。SelfCheckGPT 和 FActScore 很实用，RAGAS 是 RAG 场景的最佳工具。Sleeper Agents 的研究说明，事后安全措施有局限性。基础模型供应商的 pipeline 清洁程度，比你的微调过程更重要。CAI 是个强大模式，即使不照搬 Anthropic 的方案也能用。上线时别光靠意图，要配上小而真实的安全机制。

下一篇（最终篇）：**生产**。详细聊服务栈选择、autoscaling 策略、延迟预算、成本跟踪、多模型路由，以及从第一天就需要的可观测性。
## 参考资料

- Askell, A. 等（2021）。*A General Language Assistant as a Laboratory for Alignment*（HHH 框架）。https://arxiv.org/abs/2112.00861
- Christiano, P. 等（2017）。*Deep Reinforcement Learning from Human Preferences*。https://arxiv.org/abs/1706.03741
- Ouyang, L. 等（2022）。*Training language models to follow instructions with human feedback*（InstructGPT）。https://arxiv.org/abs/2203.02155
- Sharma, M. 等（2023）。*Towards Understanding Sycophancy in Language Models*。Anthropic。https://arxiv.org/abs/2310.13548
- Pan, A. 等（2022）。*The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*。ICLR 2022。https://arxiv.org/abs/2201.03544
- Casper, S. 等（2023）。*Open Problems and Fundamental Limitations of RLHF*。https://arxiv.org/abs/2307.15217
- Ganguli, D. 等（2022）。*Red Teaming Language Models to Reduce Harms*。Anthropic。https://arxiv.org/abs/2209.07858
- Chao, P. 等（2023）。*Jailbreaking Black Box Large Language Models in Twenty Queries*（PAIR）。https://arxiv.org/abs/2310.08419
- Mehrotra, A. 等（2024）。*Tree of Attacks: Jailbreaking Black-Box LLMs Automatically*。https://arxiv.org/abs/2312.02119
- Chao, P. 等（2024）。*JailbreakBench*。https://arxiv.org/abs/2404.01318
- Hubinger, E. 等（2024）。*Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*。Anthropic。https://arxiv.org/abs/2401.05566
- Arditi, A. 等（2024）。*Refusal in Language Models is Mediated by a Single Direction*。https://arxiv.org/abs/2406.11717
- Bai, Y. 等（2022）。*Constitutional AI: Harmlessness from AI Feedback*。Anthropic。https://arxiv.org/abs/2212.08073
- Lee, H. 等（2023）。*RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*。https://arxiv.org/abs/2309.00267
- Lin, S. 等（2021）。*TruthfulQA*。https://arxiv.org/abs/2109.07958
- Manakul, P. 等（2023）。*SelfCheckGPT*。EMNLP 2023。https://arxiv.org/abs/2303.08896
- Min, S. 等（2023）。*FActScore*。https://arxiv.org/abs/2305.14251
- Thorne, J. 等（2018）。*FEVER*。https://arxiv.org/abs/1803.05355
- Kirchenbauer, J. 等（2023）。*A Watermark for Large Language Models*。ICML 2023。https://arxiv.org/abs/2301.10226
- Sadasivan, V. 等（2023）。*Can AI-Generated Text be Reliably Detected?* https://arxiv.org/abs/2303.11156
- Burns, C. 等（2023）。*Weak-to-Strong Generalization*。OpenAI。https://arxiv.org/abs/2312.09390
- Anthropic（2024）。*Frontier Red Team Report*。https://www.anthropic.com/research
