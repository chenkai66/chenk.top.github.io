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
评估是大模型栈中争议最多、信心最弱的一环：榜单被刷、公开基准遭污染，我参与过的多数团队甚至没有自己的评估集。本章聚焦五个关键问题：评估真正能揭示什么、基准暗藏的陷阱、无人修复的 LLM-as-judge 偏差、多数团队忽略的校准指标，以及能在客户感知前捕获回归的生产级评估模式。

本章风格与系列其他章节不同：多数评估难题并非技术问题，而是认知问题。"模型 A 是否比模型 B 强"是个假设检验问题，但该领域的干净实验记录很差。以下引用的论文揭示了失败模式，理应让每位从业者保持审慎。

![LLM Engineering (10): Evaluation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/illustration_1.png)

## 为什么公开基准在撒谎

![fig1: benchmark contamination over time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig1_benchmark_contamination.png)


标准基准套件（如 MMLU、GSM8K、HumanEval、ARC、HellaSwag 等）存在五大共性缺陷：

**污染。** 大部分公开基准都能通过网页爬取获取。用 CommonCrawl 训练的模型见过题目，往往也见过答案。 Phi-3 被发现其训练数据中确实包含 MMLU 题目；很难相信其他实验室更干净。 Zhou et al. (2023, *Don't Make Your LLM an Evaluation Benchmark Cheater*) 系统性验证了泄露的影响：仅少量测试集泄露即可带来 10–30 个百分点的分数提升，与真实能力提升难以区分。这篇论文最令人不安的发现是：随机子采样 CommonCrawl 已经包含了 5-10% 的热门基准题。避免污染需要主动过滤，但大多数实验室只是口头承诺，实际文档中并未体现。

Sainz et al. (2023, *NLP Evaluation in trouble*) 对 250 余个近期基准进行调研，发现约 40% 在至少一个主流基础模型的训练语料中存在可检测泄露，泄露率大致随预训练语料规模增长。

**饱和。** MMLU 最初面向 2020 年模型设计，如今顶尖模型已达 88–92% 准确率——竞争已进入噪声主导区间。 MMLU 上 1.5 分的提升可能是真提升，也可能是边界案例上 50 题抽样的采样方差。这基准几年前就分不清能力了。简单的统计论证： 14K 题，模型准确率 90%，分数的标准误大约是 $\sqrt{0.9 \cdot 0.1 / 14000} \approx 0.25$ 个百分点；不到 ~0.7 分的 1 分宣称都在噪声里。

**选择题偏差。** MMLU 是 4 选 1 选择题。模型全选"B"也能拿 25%。经过大量多选题训练的模型有优势，但这优势无法泛化到自由生成输出。更糟糕的是，多选题提示内的位置偏差是真实的——Wang et al. (2023, *Large Language Models are not Fair Evaluators*) 展示了打乱答案位置能在大多数模型上改变 2-8 分的准确率。

**仅英语。** MMLU、 GSM8K 和 HumanEval 都是英文。中文或法文能力强的模型在标准榜单上分数一般，但在实际部署中表现更好。

**格式耦合。** GSM8K 答案通过正则表达式提取 "the answer is X"。不遵循确切格式就算错，即使内容正确。 HuggingFace 评估团队 2024 年审计估计 GSM8K 上 5-15% 的"错误"答案实际上是内容正确但格式非标准。

2024-2026 这一代基准（MMLU-Pro, GPQA, BIG-Bench Hard, RULER, SWE-bench, LiveBench, ArenaHard, IFEval）试图解决这些问题，但每个发布几个月内都出现了刷榜问题。

## MMLU 到底测的是什么（至今为止）

尽管被批评， MMLU 不是没用。它测的是东西—— broadly, "预训练时模型有没有看到并记住大量大学级别知识？" 这和通用能力 loosely 相关。 MMLU 跑 50 分的模型确实比跑 80 分的差；但 88 到 92 的差距 mostly 是噪声。

将 MMLU 视为粗粒度过滤器（‘该模型是否处于能力合理区间？’），切勿用于区分准确率 85% 以上的模型。

## 基准动物园：什么时候用什么

实用指南，哪个基准回答哪个问题：

- **通用知识 / 粗能力**： MMLU （仅当过滤器用）， MMLU-Pro （在前沿仍有意义）。
- **不确定性下的推理**： GPQA Diamond (Rein et al., 2023) —— 研究生级别科学题，设计用来抵抗 LLM 友好技巧。 2026 顶尖模型跑 60-75%；相关领域 PhD 人类跑 ~65%。这基准是真的难。
- **数学，较易**： GSM8K (Cobbe et al., 2021, *Training Verifiers to Solve Math Word Problems*) —— 8.5K 小学应用题。前沿模型饱和了 (>95%) 但对小模型仍有用。
- **数学，较难**： MATH (Hendrycks et al., 2021, *Measuring Mathematical Problem Solving*) —— 12.5K 竞赛题， AMC/AIME 级别。前沿仍有区分度（顶尖模型 85-92%）。 2025 年被 AIME-2025 和 Putnam-2025 取代以获得真正的竞赛严谨性。
- **代码，封闭**： HumanEval (Chen et al., 2021, *Evaluating Large Language Models Trained on Code*) —— 164 手写 Python 题。饱和。 MBPP (Austin et al., 2021) 类似但稍广。
- **代码，真实**： SWE-bench (Jimenez et al., 2023) —— 来自 12 个热门 Python  repo 的真实 GitHub issue，模型必须通过编辑 repo 解决。多文件，需要理解现有代码。 2026 顶尖模型在 SWE-bench Verified 上跑 50-60%。
- **代码，广泛**： BigCodeBench (Zhuo et al., 2024) —— 723 个函数和 139 个库 across 1140 任务。抵抗影响 HumanEval 克隆的 GitHub 污染。
- **指令遵循**： IFEval (Zhou et al., 2023) —— 验证模型是否满足显式约束（" exactly 3 段回复，每段以问题开头"）。程序化验证，低歧义。
- **长上下文**： RULER (Hsieh et al., 2024) —— 多种难度下的 needle-in-haystack，高达 128K tokens。大多数"1M 上下文"宣称在这里 32K 就崩了。
- **幻觉**： SimpleQA (OpenAI, 2024) —— 4326 短答案事实题，设计用来暴露 confabulation。前沿模型跑 30-55%；低阶模型接近 10%。
- **成对人类偏好**： Chatbot Arena / ArenaHard (lmsys) —— 真实聊天交互上数百万 crowdsourced A-vs-B 投票。

生产环境里，相关问题很少是"谁赢了 MMLU"，几乎是"谁在我的负载上赢了"。基准动物园 mainly 用于花钱做定制评估前 *剔除明显糟糕的候选者*。

## LiveBench, ArenaHard 和动态基准

2024-2025 对污染的回应是 **动态基准**：每月新鲜生成或 curated 问题，测试前从未发布。

- **LiveBench**（2024 年底发布，每月刷新）：数学、编码、推理、语言、指令遵循题， sourced from 近期（post-cutoff）论文和竞赛。难污染。
- **ArenaHard / Chatbot Arena** (lmsys)：真实聊天流量的人类成对偏好。更新慢但抗刷榜因为人类选择。 Arena 的公开榜单有个不同的 bug — *风格偏好* 压倒 *正确性偏好* 对许多用户，所以排名奖励温暖、格式好、稍啰嗦的回答，而不是正确但简洁的。
- **SimpleQA** (OpenAI, 2024)：事实题，短特定答案。设计用来暴露幻觉——模型 confabulates 就输。

2026 年这些信号比 MMLU 好，但仍有问题。 ArenaHard 权重用户觉得舒服的（" nice formatting", "warmth"），这可能跟正确性脱钩。 LiveBench 的选题过程本身如果你知道 curated 标准也可被游戏。*没有公开基准能在持续优化压力下存活*。

## 防污染防御： Hold-out 评估集

生产环境里，你不应该在公开基准上评估。建自己的 eval 集：

1. 从真实（或合成但逼真）生产流量采样 100-500 代表输入。
2. 手写或手 curate 金标输出。
3. 绝不公开这个集。绝不放进任何 hitting 外部 API 的 prompt （供应商可能从你的流量学习）。
4. 每次模型变更都重测这个集。

投入是实的（200 题手标集要 1-2 人天），但第一次抓住公开基准没发现的回归时就回本了。

微妙点：**怎么切分 eval 集和集里有什么一样重要**。常见错误是从生产流量均匀采样 200 题，但生产流量 80% 简单 20% 困难。你的 eval 被简单题主导，每个模型都跑 95%。分层：目标 30-40% 简单， 30-40% 中等， 30-40% 困难，其中"困难" sourced from 你的错误日志。困难切片才是区分模型的部分。
## LLM-as-judge：主流模式与它的失效模式

![LLM Engineering (10): Evaluation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/illustration_2.png)


![fig5: human vs auto-eval correlation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig5_human_vs_auto.png)


![fig2: LLM-judge position bias](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig2_judge_position_bias.png)


面对自由生成的输出（大多数生产任务都是这类）， exact match 根本行不通。你得有个办法给“这回答好不好”打分。目前主流的做法是 **LLM-as-judge**（Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*）：用一个强模型给被测模型的输出打分。

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

LLM-as-judge 与人类判断的相关性在 70-85% 之间，具体取决于任务。它比人工评估便宜得多，而且能扩展到每天处理成千上万个样本。但 Zheng 那篇论文列举了它的一些偏见，后续工作又进一步细化了这些问题：

- **Position bias**：比较两个答案时， judge 倾向于选排在前面那个（有些模型则是第二个）。 Zheng 测过，在 GPT-4-as-judge 上，交换顺序会导致 30-65% 的 pairwise 判断结果翻转——这远高于随机概率。解决办法：两种顺序都跑一遍，取平均。
- **Length bias**：答案越长得分越高，哪怕短一点更好。 Dubois et al. (2024, *Length-Controlled AlpacaEval*) 测过，仅仅把答案长度加倍而不改变质量，分数就能涨 3-7 分。解决办法：明确告诉 judge“除非有用，否则不要奖励长度”，或者用长度控制的评分（减去长度项，或针对长度匹配的基线做归一化）。
- **Self-preference**： GPT-4 当 judge 偏爱 GPT-4 的输出； Claude 偏爱 Claude。 Panickssery et al. (2024, *LLM Evaluators Recognize and Favor Their Own Generations*) 表明这是一种自我识别现象——judge 模型能认出自己的输出并给高分。解决办法：用与被测模型不同家族的模型当 judge，或者用不同家族的 judge 组成评审团，多数投票。
- **Format bias**：结构化输出（Markdown、标题）比同等质量的纯文本得分高。解决办法：评分前先归一化格式。
- **Sycophancy / verbosity-as-rigor**： judge 有时会偏向那个看起来更自信或更详尽的答案，哪怕它是错的。解决办法：尽可能提供参考答案，并告诉 judge 权重应放在正确性而非展示上。

对于 pairwise 比较（模型 A vs 模型 B），修正偏见的做法是：

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

Zheng et al. 表明，交换顺序后一致的判断与人类的相关性约为 85%；不一致的基本上就是噪声。直接丢弃这些不一致的结果是最简单的修正方法。

**Pointwise vs pairwise。** 这两种评判范式有不同的失效模式。 Pointwise （独立给每个答案打 1-5 分）更便宜，但 suffers from rating drift across batches，而且 judge 容易隐式地锚定在最近看到的例子上。 Pairwise （哪个更好， A 还是 B？）每次比较更可靠，但模型数量多了之后复杂度是二次方的。生产环境的标准折衷方案：用 pointwise 做初筛（过滤掉明显差的候选），对最后剩下的 2-3 个关键候选用 pairwise。

## 代码与数学：基于程序的评估

对于可验证的领域，跳过 judge：

- **代码**：拿候选答案跑测试套件。 HumanEval、 MBPP、 SWE-bench 都是这么做的。 Pass@k 衡量 $k$ 个样本中通过的比例。
- **数学**：提取最终数值答案，与 ground truth 对比。
- **JSON 输出**：针对 schema 验证，检查 key 是否存在。
- **工具调用**：模拟工具，检查调用的效果。

基于程序的评估没有偏见，计算免费，而且 instant。能用就用，别犹豫。前沿的 reasoning 模型（o-series, Qwen3-Reasoning, DeepSeek-R1）在 RLVR 期间（第 4 章）都主要用基于程序的评估作为信号。

有个很有用的视角：每个允许程序化验证的领域，在 RLVR 中都会变成一种 *训练信号*，这意味着模型在该领域的能力大致会随着验证器的质量增长。数学和代码比 prose 增长得快，就是因为它们有更好的验证器。如果你在为可验证任务构建内部基准，你不仅仅是在测量——你是在生产一个潜在的训练信号，开源社区迟早会为此回馈你。

## 生产环境的 A/B 测试

![fig4: A/B test power calculation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig4_ab_power.png)


评估集能在部署前拦住回归问题。 A/B 测试则能抓住评估集漏掉的东西。

生产环境 A/B：把 5-10% 的流量路由到新模型 variant，对比结果指标：

- **Engagement**：用户继续对话了吗？
- **Resolution**：用户标记对话已解决了吗？
- **Latency**： p50/p95 变差了吗？
- **Cost**：每 turn 的 tokens，每个已解决请求的 $。
- **Satisfaction**：点赞/点踩率。
- **Escalation**：用户要求转人工了吗？

像"engagement"这样的表面指标可能会误导——一个模型如果有趣地跑题，可能会带来更多 turns，但解决的问题更少。要把 turn count 和 resolution rate 结合起来看。

为了统计显著性： 10K 流量足以在 95% 置信度下检测到二元指标 2% 的偏移。相关的 power calculation 如下：

$$n \approx \frac{16 \cdot p (1-p)}{\delta^2}$$

其中 $p$ 是基线率，$\delta$ 是最小可检测效应。对于 $p = 0.5$，$\delta = 0.02$，那就是每臂 $n \approx 10{,}000$；对于 $\delta = 0.005$（捕捉半个百分点的偏移），$n \approx 160{,}000$。实验至少跑 7 天以消除星期几的影响。如果流量 heterogeneous （免费 vs 付费，移动 vs 桌面，语言），要按流量 segment 分层。

有两个生产模式值得了解：

- **感知延迟的质量指标**。一个新模型准确率高 1% 但慢 200 ms，净效果可能是负的，因为用户流失率随延迟缩放。在宣布赢家之前，把质量和延迟合并为单一目标（例如 quality - $\alpha$ · latency_seconds，其中 $\alpha$ 根据历史流失曲线校准）。
- **带早停的 Sequential A/B**。 Frequentist t-tests 假设固定样本量；如果你中途 peek 并早停，假阳性率会膨胀。用 sequential test （mSPRT，或者简单地对 peek 次数做 Bonferroni 校正）来保持假阳性率诚实，同时还能快速决策。

## 评估集的维护

你今天建的评估集，半年后就废了。生产流量在变，客户用例在演化，你模型的失效模式也不是下个季度模型的失效模式。

行之有效的维护 routine：

- **每周**：采样 50 个生产调用，标记异常，把有趣的失败案例加入评估集。
- **每月**：针对评估集重跑所有生产 prompt，绘制质量趋势。
- **每季度**：淘汰所有当前模型都能 100% 通过的评估项（不再具有信息量）。从最近的失效模式中添加新项。
- **每次模型变更**：全量重评估，绝无例外。如果评估跑得太慢，就自动化并并行化。

一个有用的不变量：你评估集上模型准确率的 *interquartile range* 不应坍缩为零。如果每个模型在每个问题上都得 99%，说明你过度饱和了，评估不再起作用。砍掉简单问题， sourcing 新的难题。

## 校准：一个常被忽略的指标

![fig3: calibration plot (ECE)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig3_calibration.png)


如果一个模型陈述的置信度与实际准确率匹配，它就是 **calibrated**。一个说“我有 90% 把握”的模型，应该 90% 的时候是对的。大多数 LLM 都系统性地过度自信——它们会说“我确定”，然后 30% 的时候是错的。

对于高风险部署（医疗、法律、金融），校准和准确率一样重要。用 Expected Calibration Error 测量（Guo et al., 2017, *On Calibration of Modern Neural Networks*）：

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

按置信度将预测分箱到 $M$ 个 bucket，计算每个 bucket 的 |accuracy - average confidence|，按 bucket 大小加权。 reliability diagram 在 y 轴 plot 准确率， x 轴 plot 置信度；完美校准的模型落在对角线上。

Guo 那篇论文表明 *准确率的提升往往以牺牲校准为代价*——那些在分类基准上提升准确率的技术（deep networks, label smoothing, temperature scaling）除非明确修正，否则往往会损害校准。同样的教训也适用于 LLM： post-training （RLHF, DPO）会可靠地 degrade 校准，因为 reward model 奖励听起来自信的答案多于那些留有余地的答案。

2026 年的前沿模型 ECE 仍在 5-15% 范围内——这意味着显著的过度自信。 RLHF 倾向于 *恶化* 校准（模型学到听起来自信的答案能获得更高奖励）。对于需要模型知道自己不知道什么的部署，你可能需要：

1. **通过 SFT 撤销部分 post-training 的断言性**，使用来自基座模型不确定但 post-trained 模型变得自信的案例中的 "I don't know" 样本。
2. **在推理时对 logprobs 使用 temperature scaling**。在 held-out 集上校准 temperature；这很便宜，通常能把 ECE 减半。
3. **采样多个 completions 并用答案的 entropy 作为置信度代理。** 一个问题上 $N=10$ 的 self-consistency——如果 9/10 一致，高置信度；如果 5/10/0/0 等分散，低置信度。
4. **拟合一个校准模型**，基于 (model_logprob, prompt_features) →  labeled sample 上的 empirical_correctness。

对于 RAG 系统，*retrieval confidence*（top-k 相似度分数， reranker 分数）是比 LLM 自己生成的 tokens 更诚实的置信度信号。把它传进去。
## 长文本评估：基准测试遗漏了什么

公共基准测试大多盯着短回答测。可真正落到生产环境，往往是长文本生成（比如总结、草稿、 500 行以上的代码块）。专门针对这块的基准测试有：

- **AlpacaEval / AlpacaEval 2** (Li et al., 2023): 805 条指令；用 LLM-as-judge 对比参考答案（最早是 text-davinci-003，后来换成 GPT-4）。 Length-controlled 版本 (Dubois et al., 2024) 修正了上面提到的长度偏差问题。
- **MT-Bench** (Zheng et al., 2023): 8 个类别共 80 轮多轮对话； LLM-as-judge 打分前会先生成 chain-of-thought 理由。
- **Arena Hard** (lmsys, 2024): 从真实 Chatbot Arena 流量中蒸馏出的 500 个硬问题； LLM-as-judge 打分，与人类 Arena 排名强相关。

内部搞长文本评估，一套实用的打法是：

1. 精选 50-200 个 prompt，覆盖你的真实 workload。
2. 定义一个 5-10 维度的评分标准（准确性、完整性、格式合规、语气、幻觉率等）。
3. 找个强的 judge 模型给每个维度单独打分，并附带一行理由。
4. 聚合出综合分数，但决定发版前，务必看分项 breakdown。

分项视角至关重要。如果一个模型完整性加了 5 分，幻觉却多了 3 分，这算不上严格的改进；发不发版，得看哪个维度对你的产品更关键。

## 把评估流水线写成代码

成熟的评估系统，运维 hygiene 得跟 CI 流水线一个级别：

```
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

2026 年还能打的工具：**Promptfoo**（简单的 YAML 驱动，适合本地迭代）、**Inspect AI**（UK AISI 的框架，常用于安全评估）、**OpenAI evals**、**DeepEval**，或者如果你偏好 DIY，直接用 pytest 写一套也行。框架选哪个没那么重要，重要的是你得有一个。最糟糕的模式是弄个 notebook，想起来才跑一次评估。

## 总结与下一章

公共基准测试分数过了 80 分大半都是噪音，还有不少被污染了。基于真实流量建一套手工 curated 的评估集，按难度分层，把它当成承重代码来维护。 LLM-as-judge 能用，但得做偏差校正（交换顺序、控制长度、换裁判模型家族、面板投票）。任务可验证时，优先用 program-based eval。离线评估覆盖不到的场景，上线做 A/B 测试， peek 的时候用 sequential tests。持续维护评估集，剔除饱和的问题。别忘了校准——在很多部署场景里，置信度和准确率一样重要，而 post-training 往往会 reliably degrade 它。长文本方面，先按维度打分，再合成总分。

下一章：**安全与对齐**。涵盖拒绝行为、 RLHF 目标、 sycophancy、红队测试方法论、幻觉指标，以及 constitutional AI。

## References

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