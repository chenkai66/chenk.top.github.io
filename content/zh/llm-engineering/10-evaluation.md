---
title: "大模型工程（十）：LLM-as-Judge 与评估"
date: 2026-04-05 09:00:00
tags:
  - LLM
  - evaluation
  - benchmarks
  - llm-as-judge
  - ab-testing
categories: 大模型工程
series: llm-engineering
series_order: 10
series_title: "大模型工程"
lang: zh
mathjax: false
disableNunjucks: true
description: "为什么 MMLU 坏了、污染问题、LLM-as-judge 偏置、位置偏置缓解、校准、生产里真正能在客户之前抓到回归的 A/B 测试模式。"
translationKey: "llm-engineering-10"
---
评估是大模型技术栈中争议最多、信心最弱的一环——榜单被刷分、公开基准遭污染，我参与过的多数团队甚至在初期连自己的评估集都没有。本章将聚焦五个关键问题：评估真正能揭示什么、基准暗藏的陷阱、无人修复的 LLM-as-judge 偏差、多数团队忽略的校准指标，以及能在客户感知前捕获回归的生产级评估模式。

本章风格与系列其他章节略有不同：多数评估难题并非技术问题，而是**认知问题**。“模型 A 是否比模型 B 更好”本质上是一个假设检验问题，但整个领域在执行干净实验方面的集体记录相当糟糕。下文引用的文献并非排行榜，而是一组揭示失败模式的论文，理应让每位从业者保持审慎。

![LLM 工程（10）：评估 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/illustration_1.png)

## 为什么公开基准在撒谎

![图1：随时间变化的基准污染](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig1_benchmark_contamination.png)

标准基准套件（如 MMLU、GSM8K、HumanEval、ARC、HellaSwag 等）存在五大共性缺陷：

**污染。** 大部分公开基准都能通过网页爬取获取。使用 CommonCrawl 训练的模型很可能早已见过题目，甚至答案。Phi-3 就曾被发现其训练数据中直接包含 MMLU 原题；很难相信其他实验室更干净。Zhou et al. (2023, *Don't Make Your LLM an Evaluation Benchmark Cheater*) 系统性地测量了泄露的影响：即使只有少量测试集泄露，也能带来 10–30 个百分点的分数提升，这种提升与“真实”的能力进步几乎无法区分。该论文最令人不安的发现是：仅对 CommonCrawl 进行随机子采样，就已包含 5–10% 的热门基准题。避免污染需要主动过滤，但大多数实验室只是口头声称做到了，却鲜有文档佐证。

Sainz et al. (2023, *NLP Evaluation in trouble*) 调研了 250 多个近期发布的基准，发现约 40% 在至少一个主流基础模型的训练语料中存在可检测的泄露，且随着预训练语料规模扩大，泄露率还在上升。

**饱和。** MMLU 最初是为 2020 年的模型设计的难题，而到 2026 年，顶尖模型得分已达 88–92%——竞争已进入噪声主导区间。MMLU 上 1.5 分的提升，可能是真实进步，也可能只是从边界案例中抽取 50 题带来的采样方差。该基准早在几年前就失去了区分能力。一个简单的统计论证：在 14K 道题、模型准确率为 90% 的情况下，分数的标准误约为 $\sqrt{0.9 \cdot 0.1 / 14000} \approx 0.25$ 个百分点；任何低于约 0.7 分的提升都可能只是噪声。

**选择题偏差。** MMLU 是四选一的选择题，模型若始终选 “B” 也能拿到 25% 的分数。那些在大量多选题数据上训练的模型因此获得不公平优势，而这种优势无法迁移到自由形式的输出任务中。更糟的是，选项位置偏差真实存在——Wang et al. (2023, *Large Language Models are not Fair Evaluators*) 证明，在大多数模型上，仅打乱答案顺序就能使准确率波动 2–8 分。

**仅限英语。** MMLU、GSM8K 和 HumanEval 均为英文基准。一个在中文或法语上表现卓越的模型，可能在标准榜单上得分平平，却在其实际部署场景中大获成功。

**格式耦合。** GSM8K 的答案通过正则表达式提取 “the answer is X” 这一固定格式。即使内容正确，只要格式不符即被判错。HuggingFace 评估团队在 2024 年的一项审计估计，GSM8K 中 5–15% 的“错误”答案实际上是内容正确但格式非标准。

2024–2026 年推出的新一代基准（如 MMLU-Pro、GPQA、BIG-Bench Hard、RULER、SWE-bench、LiveBench、ArenaHard、IFEval）试图解决上述问题，但几乎每个都在发布后几个月内出现了新的刷榜手段。

## MMLU 到底测的是什么（至今为止）

尽管饱受批评，MMLU 并非毫无价值。它确实测出了某些东西——大致而言，“模型在预训练阶段是否接触并记住了大量大学水平的知识？”这与通用能力存在松散的相关性。MMLU 得分 50 的模型确实远逊于得分 80 的模型；但 88 到 92 分之间的差距，基本属于噪声范畴。

应将 MMLU 视为粗粒度过滤器（“该模型是否处于合理的能力区间？”），切勿用于比较两个得分均在 85 以上的模型。

## 基准动物园：什么时候用什么

一份实用指南，说明哪些基准适合回答哪些问题：

- **通用知识 / 粗略能力评估**：MMLU（仅作过滤器使用），MMLU-Pro（在能力前沿仍有意义）。
- **不确定性下的推理**：GPQA Diamond (Rein et al., 2023) —— 研究生级别的科学问题，专门设计以抵抗 LLM 友好的解题技巧。2026 年顶尖模型得分在 60–75%；相关领域的 PhD 人类专家得分约为 65%。这是一个真正具有挑战性的基准。
- **数学（较易）**：GSM8K (Cobbe et al., 2021, *Training Verifiers to Solve Math Word Problems*) —— 包含 8.5K 道小学应用题。虽已被前沿模型饱和（>95%），但对中小模型仍有参考价值。
- **数学（较难）**：MATH (Hendrycks et al., 2021, *Measuring Mathematical Problem Solving*) —— 12.5K 道 AMC/AIME 级别的竞赛题。在能力前沿仍具区分度（顶尖模型得分 85–92%）。到 2025 年，其地位已被 AIME-2025 和 Putnam-2025 取代，以体现真正的竞赛严谨性。
- **代码（封闭环境）**：HumanEval (Chen et al., 2021, *Evaluating Large Language Models Trained on Code*) —— 164 道手写 Python 题，已被饱和。MBPP (Austin et al., 2021) 类似，但覆盖范围略广。
- **代码（真实场景）**：SWE-bench (Jimenez et al., 2023) —— 来自 12 个热门 Python 仓库的真实 GitHub issue，要求模型通过编辑代码库来解决。涉及多文件，需理解现有代码结构。2026 年顶尖模型在 SWE-bench Verified 上的解决率约为 50–60%。
- **代码（广泛覆盖）**：BigCodeBench (Zhuo et al., 2024) —— 涵盖 1140 项任务，涉及 723 个函数和 139 个库，有效规避了影响 HumanEval 克隆版本的 GitHub 数据污染问题。
- **指令遵循**：IFEval (Zhou et al., 2023) —— 验证模型是否满足显式约束（如“用恰好三段回复，每段以问句开头”）。采用程序化验证，歧义极低。
- **长上下文**：RULER (Hsieh et al., 2024) —— 在高达 128K token 的上下文中进行多难度“大海捞针”测试。大多数宣称支持“1M 上下文”的模型，在此基准下于 32K 处就已失效。
- **幻觉检测**：SimpleQA (OpenAI, 2024) —— 包含 4326 道短答案事实题，专为暴露模型编造（confabulation）行为而设计。前沿模型得分仅 30–55%；低阶模型接近 10%。
- **成对人类偏好**：Chatbot Arena / ArenaHard (lmsys) —— 基于数百万条真实聊天交互的众包 A/B 投票。

在生产环境中，真正相关的问题几乎从来不是“谁赢了 MMLU”，而是“谁在我的实际工作负载上表现更好”。基准动物园的主要价值在于**在投入定制评估成本前，剔除明显不合格的候选模型**。

## LiveBench、ArenaHard 与动态基准

2024–2025 年应对数据污染的策略是推出**动态基准**：每月新鲜生成或精心筛选问题，确保在测试前从未公开。

- **LiveBench**（2024 年底发布，每月更新）：涵盖数学、编码、推理、语言和指令遵循，题目源自近期（模型训练截止日后）的论文与竞赛，难以被污染。
- **ArenaHard / Chatbot Arena** (lmsys)：基于真实聊天流量的人类成对偏好数据。更新较慢，但因由人类选择而难以被刷分。不过，Arena 公开排行榜存在另一问题——对许多用户而言，*风格偏好*（如温暖语气、良好排版）压倒了*正确性偏好*，导致排名更青睐详尽、温和但未必精准的回答，而非简洁正确的答案。
- **SimpleQA** (OpenAI, 2024)：提供简短明确的事实性问题，专为暴露幻觉设计——一旦模型编造答案，立即失分。

到 2026 年，这些动态基准提供的信号已优于 MMLU，但仍存在问题。ArenaHard 过度强调用户感知的舒适度（如“排版美观”、“语气亲切”），可能导致评分与正确性脱钩。LiveBench 的题目筛选流程本身，若被知晓标准，也可能被针对性优化。**没有任何公开基准能在持续的优化压力下长期保持有效性**。

## 防污染防御：Hold-out 评估集

在生产环境中，你不应依赖公开基准进行评估。务必构建自己的评估集：

1. 从真实（或合成但高度逼真）的生产流量中采样 100–500 个代表性输入。
2. 手动编写或精心审核黄金标准输出。
3. 切勿公开此评估集，也切勿将其放入任何调用外部 API 的提示中（供应商可能从你的流量中学习）。
4. 每次模型变更后，都必须在此集上重新评估。

这项投入是实实在在的（一个 200 题的手工标注集约需 1–2 人日），但只要它首次捕获到公开基准未能发现的回归，就已值回成本。

一个微妙但关键的点：**评估集的划分方式与其内容同等重要**。常见错误是从生产流量中均匀采样 200 个问题，而实际上生产流量中 80% 是简单问题，20% 是难题。结果评估集被简单问题主导，所有模型在此类问题上得分均超 95%，失去区分力。正确做法是分层采样：目标比例为 30–40% 简单、30–40% 中等、30–40% 困难，其中“困难”样本应来自你的错误日志。正是这部分难题，才能有效区分模型优劣。

## LLM-as-judge：主流模式与它的失效模式

![大语言模型工程（10）：评估 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/illustration_2.png)

![图5：人工评估与自动评估的相关性](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig5_human_vs_auto.png)

![图2：LLM 评判的位置偏差](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig2_judge_position_bias.png)

对于自由形式的输出（这也是大多数生产任务的形态），精确匹配（exact match）完全不适用。你需要一种方法来判断“这个回答好不好”。当前主流方案是 **LLM-as-judge**（Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*）：使用一个强模型对被测模型的输出进行评分。

基本模式如下：

```python
JUDGE_PROMPT = """You are evaluating an AI assistant's response.

Question: {question}

Reference answer (for grading reference; the assistant did not see this):
{reference}

Assistant's answer:
{candidate}

Rate the assistant's answer on a 1-5 scale:
1 = factually wrong or off-topic
2 = partially correct but missing key information
3 = correct but poorly explained
4 = correct and well explained
5 = correct, well explained, and adds useful related information

Output only the integer score."""

def judge(question, reference, candidate, model="claude-4-5-sonnet-20250901"):
    score = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}]
    )
    return int(score.content[0].text.strip())
```

LLM-as-judge 与人类判断的相关性在 70–85% 之间（具体取决于任务）。它比人工评估便宜得多，且可轻松扩展至每天处理数千样本。但 Zheng 的论文已列举其偏见，后续研究进一步细化了这些问题：

- **位置偏差（Position bias）**：在比较两个答案时，评判模型倾向于选择先出现的那个（某些模型则偏好后者）。Zheng 测量发现，在 GPT-4 作为评判者时，交换答案顺序会导致 30–65% 的成对判断结果翻转——远高于随机水平。缓解方法：同时呈现两种顺序，取平均分。
- **长度偏差（Length bias）**：更长的答案往往得分更高，即便简洁才是更优解。Dubois et al. (2024, *Length-Controlled AlpacaEval*) 发现，仅将答案长度加倍而不提升质量，分数就能提升 3–7 分。缓解方法：明确指示评判模型“除非长度有用，否则不要奖励”，或采用长度控制评分（如减去长度项，或与长度匹配的基线进行归一化）。
- **自我偏好（Self-preference）**：GPT-4 作为评判者时偏爱 GPT-4 的输出；Claude 则偏爱 Claude。Panickssery et al. (2024, *LLM Evaluators Recognize and Favor Their Own Generations*) 证明这是一种自我识别现象——评判模型能认出自己的生成结果并给予更高评分。缓解方法：使用与被测模型不同家族的模型作为评判者，或组建由不同家族模型组成的评审团，采用多数投票。
- **格式偏差（Format bias）**：结构化输出（如 Markdown、标题）比同等质量的纯文本得分更高。缓解方法：评分前统一格式。
- **谄媚倾向 / 冗长即严谨（Sycophancy / verbosity-as-rigor）**：评判模型有时会偏向那些看起来更自信或更详尽的答案，即使其内容错误。缓解方法：尽可能提供参考答案，并明确要求评判模型将正确性置于呈现形式之上。

对于成对比较（模型 A vs 模型 B），推荐的偏差校正方法是：

```python
def pairwise_judge(question, answer_a, answer_b, judge_model):
    # First order: A first
    score_ab = judge(question, answer_a, answer_b, judge_model)
    # Reverse order: B first
    score_ba = judge(question, answer_b, answer_a, judge_model)
    # Reverse the second score and combine
    if score_ab == "A" and score_ba == "B": return "A"
    if score_ab == "B" and score_ba == "A": return "B"
    return "tie"
```

Zheng et al. 证明，交换顺序后判断结果一致的样本与人类的相关性约为 85%；而不一致的判断基本等同于噪声。最简单的修正方法就是直接丢弃这些不一致的结果。

**逐点评分 vs 成对比较（Pointwise vs pairwise）**。这两种范式各有缺陷。逐点评分（独立为每个答案打 1–5 分）成本更低，但容易出现批次间的评分漂移，且评判者常会无意识地锚定在最近看到的样本上。成对比较（A 和 B 哪个更好？）单次更可靠，但模型数量增多时计算复杂度呈二次方增长。生产环境的标准折衷方案是：用逐点评分进行初筛（剔除明显差的候选），对最终 2–3 个关键候选使用成对比较。

## 代码与数学：基于程序的评估

对于可验证的领域，应跳过 LLM-as-judge：

- **代码**：将候选答案运行于测试套件。HumanEval、MBPP、SWE-bench 均采用此法。Pass@k 衡量 $k$ 个采样中通过测试的比例。
- **数学**：提取最终数值答案，与标准答案对比。
- **JSON 输出**：依据 Schema 验证，检查关键字段是否存在。
- **工具调用**：模拟工具执行，验证调用效果。

基于程序的评估无偏见、计算成本低、结果即时。凡可使用之处，务必优先采用。前沿推理模型（如 o-series、Qwen3-Reasoning、DeepSeek-R1）在 RLVR 阶段（见第四章）均以程序化评估为主要信号。

一个有用的视角是：任何能被程序化验证的领域，在 RLVR 中都会成为一种**训练信号**，这意味着模型在该领域能力的提升速度，大致与验证器的质量成正比。数学和代码之所以比自然语言文本进步更快，正是因为它们拥有更可靠的验证机制。如果你正在为可验证任务构建内部基准，你不仅是在测量性能，更是在生产一种潜在的训练信号——开源社区未来很可能会为此回馈你。

## 生产环境的 A/B 测试

![图4：A/B 测试的功效计算](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig4_ab_power.png)

评估集能在部署前拦截回归问题，而 A/B 测试则能捕捉评估集遗漏的盲点。

生产环境 A/B 测试的做法是：将 5–10% 的流量路由至新模型变体，并对比以下指标：

- **用户参与度（Engagement）**：用户是否继续对话？
- **问题解决率（Resolution）**：用户是否标记对话已解决？
- **延迟（Latency）**：p50/p95 延迟是否恶化？
- **成本（Cost）**：每轮对话消耗的 token 数、每个已解决问题的成本（美元）。
- **满意度（Satisfaction）**：点赞/点踩率。
- **升级请求（Escalation）**：用户是否要求转接人工？

像“参与度”这样的表面指标可能产生误导——一个模型若以有趣的方式跑题，可能引发更多对话轮次，但实际解决问题更少。应将对话轮次与解决率结合起来分析。

关于统计显著性：10K 样本足以在 95% 置信度下检测二元指标 2% 的变化。相关功效计算公式为：
$$n \approx \frac{16 \cdot p (1-p)}{\delta^2}$$
其中 $p$ 为基线率，$\delta$ 为最小可检测效应。当 $p = 0.5$、$\delta = 0.02$ 时，每组需约 $n \approx 10{,}000$ 样本；若要检测 0.5% 的微小变化（$\delta = 0.005$），则需约 $n \approx 160{,}000$。实验应至少运行 7 天以消除周内效应。若流量异构（如免费 vs 付费、移动端 vs 桌面端、不同语言），应按流量细分进行分层分析。

有两个值得借鉴的生产实践：

- **延迟感知的质量指标**：一个新模型若准确率提升 1% 但延迟增加 200 毫秒，净收益可能为负，因为用户流失率随延迟显著上升。应在宣布胜出前，将质量与延迟合并为单一目标（例如：quality - $\alpha$ · latency_seconds，其中 $\alpha$ 根据历史流失曲线校准）。
- **支持早停的序贯 A/B 测试**：经典 t 检验假设样本量固定；若中途查看结果并提前停止，假阳性率会膨胀。应采用序贯检验方法（如 mSPRT，或对查看次数进行 Bonferroni 校正），在保持假阳性率可控的同时实现快速决策。

## 评估集的维护

今天构建的评估集，半年后很可能已过时。生产流量在变，客户需求在演化，当前模型的失败模式也不会是下一季度模型的失败模式。

一套行之有效的维护流程如下：

- **每周**：采样 50 个生产调用，标记异常案例，将有趣的失败加入评估集。
- **每月**：用评估集重跑所有生产提示，绘制质量趋势图。
- **每季度**：淘汰所有当前模型均能 100% 通过的题目（已无信息量），并从近期失败模式中补充新题。
- **每次模型变更**：必须全量重评估，不得例外。若评估太慢，就自动化并行化。

一个有用的不变量是：评估集上模型准确率的**四分位距（interquartile range）** 不应坍缩至零。若所有模型在所有题目上都得 99 分，说明评估集已过度饱和，不再具备区分能力。此时应剔除简单题目，引入新的难题。

## 校准：一个常被忽略的指标

![图3：校准图（ECE）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/10-evaluation/fig3_calibration.png)

若模型声明的置信度与其实际准确率相匹配，则称其为**校准良好（calibrated）**。例如，一个声称“我有 90% 把握”的模型，应在 90% 的情况下正确。然而大多数 LLM 都系统性地过度自信——它们常说“我确定”，却有 30% 的概率出错。

在高风险场景（如医疗、法律、金融）中，校准的重要性不亚于准确率。可使用预期校准误差（Expected Calibration Error, ECE）进行衡量（Guo et al., 2017, *On Calibration of Modern Neural Networks*）：
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$
将预测按置信度划分为 $M$ 个桶（bucket），计算每个桶内 |准确率 - 平均置信度| 的绝对值，并按桶大小加权。可靠性图（reliability diagram）以置信度为横轴、准确率为纵轴；完美校准的模型应落在对角线上。

Guo 的论文指出，**准确率的提升往往以牺牲校准为代价**——那些在分类基准上提升准确率的技术（如深度网络、标签平滑、温度缩放），若不显式校正，通常会损害校准性能。这一结论同样适用于 LLM：后训练（RLHF、DPO）会系统性地**降低**校准度，因为奖励模型更青睐听起来自信的答案，而非留有余地的谨慎表述。

截至 2026 年，前沿模型的 ECE 仍在 5–15% 区间，表明存在显著的过度自信。RLHF 尤其会**恶化**校准（模型学到：自信的回答能获得更高奖励）。对于需要模型“知道自己不知道什么”的部署场景，你可能需要：

1. **通过 SFT 撤销部分后训练带来的断言性**：使用基座模型不确定但后训练模型变得自信的案例，构造“I don’t know”样本进行微调。
2. **在推理时对 logprobs 应用温度缩放**：在保留集上校准温度参数；此方法成本低廉，通常可将 ECE 降低一半。
3. **采样多个完成结果，并以答案熵作为置信度代理**：例如，在 $N=10$ 次采样中，若 9 次结果一致，则置信度高；若结果分散（如 5/10/0/0），则置信度低。
4. **拟合一个校准模型**：以 (model_logprob, prompt_features) 为输入，预测经验正确率（empirical_correctness），需在标注样本上训练。

对于 RAG 系统，**检索置信度**（如 top-k 相似度分数、重排序器分数）通常是比 LLM 自身生成 token 更可靠的置信信号，应加以利用。

## 长文本评估：基准测试遗漏了什么

大多数公开基准仅评估短答案，但生产任务常涉及长文本生成（如摘要、草稿、500 行以上的代码块）。针对此场景的基准包括：

- **AlpacaEval / AlpacaEval 2** (Li et al., 2023)：805 条指令；使用 LLM-as-judge 与参考答案（最初为 text-davinci-003，后改为 GPT-4）对比。Dubois et al. (2024) 提出的长度控制版本有效修正了前述长度偏差。
- **MT-Bench** (Zheng et al., 2023)：涵盖 8 个类别的 80 轮多轮对话；LLM-as-judge 在打分前会先生成链式推理（chain-of-thought）理由。
- **Arena Hard** (lmsys, 2024)：从真实 Chatbot Arena 流量中提炼出的 500 个难题；LLM-as-judge 评分与人类 Arena 排名高度相关。

对于内部长文本评估，推荐实践如下：

1. 精选 50–200 个覆盖真实工作负载的提示。
2. 定义一个 5–10 维的评分标准（如准确性、完整性、格式合规性、语气、幻觉率等）。
3. 使用强评判模型对每个维度单独打分，并附带一行理由。
4. 聚合为综合分数，但在决定是否上线时，务必查看各维度的详细 breakdown。

分维度视角至关重要。若一个模型在完整性上提升 5 分，却在幻觉率上恶化 3 分，这并非绝对改进；是否上线，取决于哪个维度对你的产品更重要。

## 把评估流水线写成代码

成熟的评估系统应具备与 CI 流水线同等的运维规范：

```text
[eval_set.jsonl in version control]
     ↓
[runner: parallelize across N candidates × M questions]
     ↓
[grader: program-based + LLM-as-judge + optional human spot-check]
     ↓
[result store: per-run scores, broken down by question and dimension]
     ↓
[diff view: candidate vs baseline, highlight regressions]
     ↓
[gate: block deployment if regression > threshold]
```

2026 年可用的工具包括：**Promptfoo**（YAML 驱动，适合本地迭代）、**Inspect AI**（英国 AISI 开发的框架，常用于安全评估）、**OpenAI evals**、**DeepEval**，或如果你偏好 DIY，也可基于 pytest 自行构建。选用哪个框架并不关键，关键在于**必须有一个**。最糟糕的做法是依赖一个 Notebook，仅在有人想起来时才手动运行一次评估。

## 总结

公开基准在 80 分以上大多已是噪声，且不少已被污染。应基于真实流量构建一套手工精筛的评估集，按难度分层，并视其为承重代码般维护。LLM-as-judge 可用，但必须校正偏差（交换顺序、控制长度、更换评判模型家族、采用评审团投票）。对于可验证任务，优先使用程序化评估。离线评估无法覆盖的场景，需通过生产环境 A/B 测试补足，并在中途查看时使用序贯检验。持续维护评估集，及时剔除饱和题目。切勿忽视校准——在许多部署场景中，置信度与准确率同等重要，而后训练往往会系统性地损害它。对于长文本任务，应先按维度评分，再合成总分。

下一章：**安全与对齐**。涵盖拒绝行为、RLHF 目标、谄媚倾向、红队测试方法论、幻觉指标，以及宪法 AI（constitutional AI）。

## 参考文献

- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685
- Zhou, K. et al. (2023). *Don't Make Your LLM an Evaluation Benchmark Cheater*. https://arxiv.org/abs/2311.01964
- Sainz, O. et al. (2023). *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*. EMNLP 2023 Findings. https://arxiv.org/abs/2310.18018
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML 2017. https://arxiv.org/abs/1706.04599
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. https://arxiv.org/abs/2110.14168
- Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS 2021 Datasets. https://arxiv.org/abs/2103.03874
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding* (MMLU). ICLR 2021. https://arxiv.org/abs/2009.03300
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code* (HumanEval). https://arxiv.org/abs/2107.03374
- Austin, J. et al. (2021). *Program Synthesis with Large Language Models* (MBPP). https://arxiv.org/abs/2108.07732
- Jimenez, C. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* https://arxiv.org/abs/2310.06770
- Zhuo, T. Y. et al. (2024). *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions*. https://arxiv.org/abs/2406.15877
- Rein, D. et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*. https://arxiv.org/abs/2311.12022
- Hsieh, C.-P. et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* https://arxiv.org/abs/2404.06654
- Zhou, J. et al. (2023). *Instruction-Following Evaluation for Large Language Models* (IFEval). https://arxiv.org/abs/2311.07911
- Li, X. et al. (2023). *AlpacaEval: An Automatic Evaluator of Instruction-following Models*. https://github.com/tatsu-lab/alpaca_eval
- Dubois, Y. et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*. https://arxiv.org/abs/2404.04475
- Wang, P. et al. (2023). *Large Language Models are not Fair Evaluators*. ACL 2024. https://arxiv.org/abs/2305.17926
- Panickssery, A. et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. https://arxiv.org/abs/2404.13076
- OpenAI (2024). *Introducing SimpleQA*. https://openai.com/index/introducing-simpleqa/
- LMSYS (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. https://arxiv.org/abs/2403.04132
- White, J. et al. (2024). *LiveBench: A Challenging, Contamination-Free LLM Benchmark*. https://livebench.ai/
- UK AISI (2024). *Inspect: An OSS framework for large language model evaluations*. https://inspect.ai-safety-institute.org.uk/
