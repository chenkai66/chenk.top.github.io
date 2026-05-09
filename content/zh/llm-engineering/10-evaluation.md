---
title: "大模型工程（十）：评估"
date: 2026-05-05 09:00:00
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
评估是 LLM 技术栈里最让人头疼的部分。人人都有意见，但没人真有信心。排行榜被刷分，公开基准测试数据集被污染。我加入过的很多团队，压根没有评估集。

这一章聊聊评估到底能看出什么。基准测试藏着哪些坑？为什么没人解决 LLM 作为评判者的偏差？为什么大多数团队跳过校准指标？生产环境里怎么赶在用户之前发现模型退化？

这一章风格和其他章节有点不同。评估的问题大多不是技术问题，而是认识论问题。“模型 A 是否比模型 B 更好”是个假设检验问题。可惜，整个领域在设计干净实验上的表现很糟糕。

我引用的文献不是排行榜，而是一些失败模式的论文。这些内容应该让每个从业者都更谨慎。

![大模型工程（十）：评估 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/illustration_1.png)
## 为什么公开 benchmark 撒谎

![fig1: benchmark contamination over time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig1_benchmark_contamination.png)

标准基准测试套件（MMLU、GSM8K、HumanEval 等）问题不少。下面挨个聊聊。

**数据污染。** 大多数公开 benchmark 都能通过爬虫抓到。用 CommonCrawl 训练的模型，基本都见过题目，甚至答案。Phi-3 更是被发现训练数据里直接有 MMLU 原题。其他实验室的数据干净吗？我持怀疑态度。Zhou 等人 2023 年的论文《*Don't Make Your LLM an Evaluation Benchmark Cheater*》测了下影响——哪怕只有一点点泄露，分数也能涨 10-30 个百分点。这种提升和真实能力改进几乎分不出来。更扎心的是，随机抽 CommonCrawl 数据时，已经有 5-10% 的流行 benchmark 题目混进去了。想避免污染，得主动过滤。但大多数实验室只是嘴上说说，真正记录下来的很少。

Sainz 等人 2023 年的论文《*NLP Evaluation in trouble*》调查了 250 多个近期发布的 benchmark。结果发现，约 40% 的测试在至少一个主流基础模型的训练语料里有泄露。泄露率还随着预训练语料规模的增长而上升。

**饱和效应。** MMLU 最初设计是为了难倒 2020 年的模型。到了 2026 年，顶级模型得分已经飙到 88-92%。它们其实在噪声范围内竞争了。MMLU 上涨 1.5 个百分点，可能是能力进步，也可能是采样方差。毕竟，从边界案例中抽 50 道题，波动很正常。简单算一下：14K 道题，模型准确率 90%，分数的标准误差大约是 $\sqrt{0.9 \cdot 0.1 / 14000} \approx 0.25$ 个百分点。如果某个声称的 1 点提升没超过 ~0.7 点，那基本就是噪声。

**多选题偏差。** MMLU 是四选一的多选题。一个永远选 "B" 的模型能得 25% 的分数。那些在多选题数据上大量训练的模型，具备一种无法转移到自由形式输出的优势。更糟糕的是，多选题提示中的位置偏差确实存在。Wang 等人 2023 年的论文《*Large Language Models are not Fair Evaluators*》证明，打乱答案选项的位置会让大多数模型的准确率波动 2-8 个百分点。

**仅限英文。** MMLU、GSM8K、HumanEval 全部是英文测试。中文或法语表现出色的模型，在权威排行榜上可能得分平平，但在实际部署中却表现优异。

**格式依赖。** GSM8K 的答案提取依赖正则表达式匹配 "the answer is X"。如果模型没按这个格式输出，即使答案正确也会被判错。HuggingFace 评估团队在 2024 年的一项审计中估计，GSM8K 中 5-15% 的“错误”答案实际上是内容正确但格式不符合规范的结果。

2024-2026 年的新一代 benchmark（MMLU-Pro、GPQA、BIG-Bench Hard 等）试图解决这些问题。但每个测试发布几个月内都会出现新的刷分漏洞。
## MMLU 实际还在测什么

批评很多，但 MMLU 并非没用。它测的是模型预训练时有没有见过大量大学水平的知识，并记住了多少。这和通用能力有点关系。得分 50 的模型确实不如 80 的。88 到 92 的差距？基本是噪声。

MMLU 可以当粗筛工具，判断“这模型是不是在对的范围”。别用它比较 85 分以上的模型。
## Benchmark 动物园：什么时候用什么

一份实用指南，告诉你每个 benchmark 能回答什么问题。

- **通用知识 / 粗略能力**：MMLU 只用来筛选，MMLU-Pro 在前沿领域仍有意义。  
- **不确定性推理**：GPQA Diamond（Rein 等，2023），研究生级别的科学问题，专门防 LLM 取巧。顶级模型得分 60-75%，博士生约 65%。这玩意真难。  
- **数学，简单**：GSM8K（Cobbe 等，2021），8.5K 道小学应用题。前沿模型接近满分（>95%），小模型还能用它测。  
- **数学，困难**：MATH（Hendrycks 等，2021），12.5K 道 AMC/AIME 级竞赛题。顶级模型得分 85-92%。2025 年被 AIME-2025 和 Putnam-2025 替代，更严格。  

- **代码，封闭场景**：HumanEval（Chen 等，2021），164 道手写 Python 题。前沿模型已经饱和。MBPP（Austin 等，2021）类似，但范围稍广。  
- **代码，真实场景**：SWE-bench（Jimenez 等，2023），来自 12 个流行 Python 仓库的真实 GitHub 问题。模型需要修改代码库，涉及多文件操作。2026 年顶级模型在 Verified 上得分 50-60%。  
- **代码，广泛场景**：BigCodeBench（Zhuo 等，2024），1140 个任务，涵盖 723 函数和 139 库。避免了 HumanEval 克隆版的 GitHub 数据污染问题。  

- **指令遵循**：IFEval（Zhou 等，2023），验证模型是否满足明确约束，比如“恰好用 3 段回答”。程序化验证，歧义性低。  
- **长上下文**：RULER（Hsieh 等，2024），多难度“大海捞针”测试，最长支持 128K token。号称支持“1M 上下文”的模型，测到 32K 就崩了。  
- **幻觉检测**：SimpleQA（OpenAI，2024），4326 道简短事实问题，专门揭露胡编乱造。前沿模型得分 30-55%，低端模型仅 10%。  
- **成对人类偏好**：Chatbot Arena / ArenaHard（lmsys），百万级众包 A vs B 投票，基于真实聊天交互。  

实际生产中，很少有人关心“哪个模型在 MMLU 上表现最好”。大家真正想知道的是，“哪个模型在我的工作负载上更强”。Benchmark 动物园的主要作用是帮我在花钱做定制评估之前，快速淘汰明显不合格的候选模型。
## LiveBench、ArenaHard 和动态基准

2024 到 2025 年，业界针对数据污染问题提出了动态基准的解法。每月生成或筛选一批全新题目，测试前从未公开。

- **LiveBench**（2024 年底发布，每月更新）：题目涵盖数学、编程、推理、语言理解和指令跟随，全部来自 cutoff 后的论文和竞赛。实时更新，污染难度大。
- **ArenaHard / Chatbot Arena**（lmsys）：基于真实聊天流量中的人类两两偏好数据。更新慢，但抗刷分能力强。不过，排行榜有个问题——很多用户更看重风格而非正确性。语气友好、格式工整的答案往往排名更高，哪怕内容啰嗦甚至不够准确。
- **SimpleQA**（OpenAI，2024）：专注于简短明确的事实性问题，专门用来检测模型幻觉。胡编乱造的模型在这里无处藏身。

这些基准在 2026 年比 MMLU 更有参考价值，但问题也不少。ArenaHard 偏向“格式美观”和“语气友好”，容易脱离正确性。LiveBench 的策展标准一旦被摸透，也可能被针对性优化。一句话总结：**没有一个公开基准能扛住持续优化的压力**。
## 防污染：留出评测

生产环境里，别用公开 benchmark 做评估。自己建一个 eval set。

1. 从真实生产流量（或合成但接近真实的场景）抽 100 到 500 条代表性输入。  
2. 手动写答案，或者精心整理出标准输出。  
3. 这个集合不能公开，也不能放进调外部 API 的 prompt，防止厂商学走你的数据。  
4. 每次模型改了，都用这个集合重新跑一遍评估。

投入是实打实的。标注 200 道题要花 1 到 2 人天。但它第一次帮你抓到公开 benchmark 发现不了的性能回退时，你就赚了。

还有一个关键点：**怎么划分 eval set 和内容本身一样重要**。常见错误是从生产流量均匀采样 200 道题。但生产流量往往是 80% 简单，20% 困难。简单问题太多，每个模型都能拿 95% 的分，eval 就没意义了。正确做法是分层设计：简单、中等、困难各占 30%-40%。困难部分直接从错误日志里挑。困难问题才是区分模型能力的关键。
## LLM-as-judge：主流方法和它的坑

![大模型工程（十）：评估 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/illustration_2.png)

![fig5: human vs auto-eval correlation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig5_human_vs_auto.png)

![fig2: LLM-judge position bias](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig2_judge_position_bias.png)

自由格式输出没法用精确匹配来评估。我得想办法判断“这个答案好不好”。目前最常用的方法是 **LLM-as-judge**（Zheng 等，2023，*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*）。简单说，就是用一个强模型给被测模型的输出打分。

代码实现如下：

```python
JUDGE_PROMPT = """你在评估 AI 助手的回答。

问题：{question}

参考答案（仅用于评分参考；助手未见过）：
{reference}

助手的回答：
{candidate}

请按 1-5 分为助手的回答评分：
1 = 事实错误或跑题
2 = 部分正确但缺少关键信息
3 = 正确但解释不清
4 = 正确且解释清晰
5 = 正确、解释清晰，并补充了相关有用信息

只输出整数分数。"""

def judge(question, reference, candidate, model="claude-4-5-sonnet-20250901"):
    score = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}]
    )
    return int(score.content[0].text.strip())
```

LLM-as-judge 和人类判断的相关性在 70%-85% 之间，具体看任务类型。它比人工评估便宜得多，每天能处理上千个样本。但它有不少偏见，Zheng 的论文列了一些，后续研究又补充了不少。

- **位置偏见**：比较两个答案时，模型倾向于选先出现的答案（有些模型偏好后出现的）。Zheng 测试发现，GPT-4-as-judge 在交换顺序后，30%-65% 的判断会翻转——远高于随机概率。解决办法：同时呈现两种顺序，取平均值。
- **长度偏见**：更长的答案得分更高，即使短答案可能更好。Dubois 等（2024，*Length-Controlled AlpacaEval*）发现，仅将答案长度加倍，就能提升 3-7 分。解决办法：告诉模型“除非内容有用，否则别奖励长度”，或者用长度控制评分。
- **自我偏好**：GPT-4 偏好 GPT-4 的输出，Claude 偏好 Claude 的输出。Panickssery 等（2024，*LLM Evaluators Recognize and Favor Their Own Generations*）证明这是自我识别现象——模型能认出自己的输出并给高分。解决办法：换不同家族的判别模型，或者用多模型评审团投票。
- **格式偏见**：结构化输出（如 Markdown、标题）比同等质量的散文得分更高。解决办法：评分前统一格式。
- **谄媚效应 / 冗长即严谨**：模型有时偏向更自信或更详细的答案，即使这些答案是错的。解决办法：提供参考答案，告诉模型优先考虑正确性。

两两比较（模型 A 和模型 B）时，纠正偏见的方法如下：

```python
def pairwise_judge(question, answer_a, answer_b, judge_model):
    # 第一种顺序：A 在前
    score_ab = judge(question, answer_a, answer_b, judge_model)
    # 反序：B 在前
    score_ba = judge(question, answer_b, answer_a, judge_model)
    if score_ab == "A" and score_ba == "B": return "A"
    if score_ab == "B" and score_ba == "A": return "B"
    return "tie"
```

Zheng 等人指出，顺序一致的判断与人类评分的相关性约为 85%；顺序不一致的基本是噪声。直接丢弃这些噪声是最简单的修正方法。

**Pointwise 和 Pairwise**。这两种评分方式各有优劣。Pointwise 成本低，但容易受评分漂移和隐式锚定的影响。Pairwise 每次比较更可靠，但复杂度随模型数量平方增长。实际生产中，我通常用 Pointwise 初筛（过滤掉明显差的候选），再用 Pairwise 对最后 2-3 个重要候选做最终评估。
## 代码和数学：基于程序的评估

在可验证领域，直接跳过人工评判。

- **代码**：用测试套件跑候选代码。HumanEval、MBPP、SWE-bench 都这么做。Pass@k 衡量 $k$ 个样本中通过测试的比例。
- **数学**：提取最终数值答案，和标准答案对比。
- **JSON 输出**：校验是否符合 schema，检查关键字段是否存在。
- **工具调用**：模拟工具行为，验证调用结果。

基于程序的评估无偏见，计算成本低，响应速度快。能用就尽量用。前沿推理模型（o 系列、Qwen3-Reasoning、DeepSeek-R1）在 RLVR（第 4 章）中都把基于程序的评估作为主要信号。

一个有用的视角：任何能用程序验证的领域，都可以转化为 RLVR 的训练信号。模型能力提升速度取决于验证器质量。数学和代码进步比自然语言快，因为验证方法更可靠。如果你为某个可验证任务构建内部基准测试集，不只是在测量性能，还在创造潜在训练信号。开源社区迟早会因此回馈你。
## 生产 A/B 测试

![fig4: A/B test power calculation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig4_ab_power.png)

Eval 集能提前发现模型退化问题。A/B 测试则用来捕捉 Eval 集漏掉的问题。

生产环境的 A/B 测试很简单：把 5%-10% 的流量切给新模型变体，对比结果指标就行。

- **用户互动**：用户是否继续对话？  
- **问题解决**：用户是否标记对话已解决？  
- **响应延迟**：p50 或 p95 延迟有没有变差？  
- **成本**：每轮对话用多少 token？每次解决问题花多少钱？  
- **满意度**：用户点赞还是点踩？  
- **升级需求**：用户是否要求转接人工客服？

“用户互动”这种表面指标容易误导人。一个跑题但有趣的模型可能增加对话轮次，却解决不了实际问题。我会结合对话轮次和问题解决率一起看。

统计显著性怎么算？如果要在 95% 置信水平下检测二元指标 2% 的变化，1 万流量就够了。公式如下：

$$n \approx \frac{16 \cdot p (1-p)}{\delta^2}$$

$p$ 是基线比例，$\delta$ 是最小可检测效应。比如 $p = 0.5$、$\delta = 0.02$，每组需要约 1 万样本；如果 $\delta = 0.005$（检测 0.5% 的变化），就需要约 16 万样本。实验至少跑 7 天，避免星期周期影响。如果流量来源复杂（免费 vs 付费、移动端 vs 桌面端、不同语言），我会按流量分层分析。

再分享两个踩过的坑，值得记住：

- **延迟敏感的质量指标**。新模型准确率提高 1%，但响应时间慢了 200 毫秒，最终效果可能是负面的。用户流失率会随着延迟上升而增加。判定胜者之前，我会把质量和延迟整合成一个目标，比如 `quality - $\alpha$ · latency_seconds`，其中 $\alpha$ 根据历史流失曲线校准。
- **带早停机制的序列 A/B 测试**。传统 t 检验假设样本量固定。如果中途偷看数据并提前停止实验，假阳性率会升高。为了快速决策又控制假阳性率，我用序列检验方法，比如 mSPRT，或者简单对偷看次数做 Bonferroni 校正。
## Eval set 维护

Eval set 不会一劳永逸。生产流量会变，客户需求会变，模型的失败模式也会变。半年后，你今天的 eval set 很可能就过时了。

我的维护流程是这样的：

- **每周**：抽 50 条生产调用记录，标记异常，把有意思的失败加到 eval set。
- **每月**：用所有生产 prompt 跑一遍 eval set，画出质量趋势图。
- **每季度**：删掉所有模型都能 100% 通过的测试项，补充最近踩坑发现的新问题。
- **每次模型更新**：必须重新评估整个 eval set。如果太慢，就自动化并行处理。

一个简单判断标准：eval set 上模型准确率的 *四分位距* 不能为零。如果每个模型在每道题上都达到 99%，说明题目太简单了，eval set 失去了意义。这时要删掉容易的题，补充更难的案例。

下一节我会聊聊我在生产环境里踩过的几个大坑。
## Calibration：人们跳过的指标

![fig3: calibration plot (ECE)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/10-evaluation/fig3_calibration.png)

模型 **校准** 的意思是，置信度要和实际准确率一致。如果模型说“我有 90% 的把握”，那它就应该在 90% 的情况下是对的。但大多数 LLM 都过于自信。嘴上说“我很确定”，结果 30% 的时候是错的。

医疗、法律、金融这些高风险场景中，校准的重要性不亚于准确率。衡量校准程度可以用 Expected Calibration Error（Guo 等，2017，《On Calibration of Modern Neural Networks》）：

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

把预测按置信度分成 $M$ 个区间，每个区间计算 |准确率 - 平均置信度|，再按区间大小加权求和。可靠性图把准确率画在 y 轴，置信度画在 x 轴。完美校准的模型会落在对角线上。

Guo 的论文指出，提升准确率往往会牺牲校准效果。深度网络、label smoothing、temperature scaling 这些技术能提高分类任务的准确率，但如果不特别调整，通常会让校准变差。这个结论同样适用于 LLM。RLHF 和 DPO 这些后训练方法会让校准变得更糟，因为奖励模型更倾向于奖励听起来自信的答案，而不是模棱两可的回答。

到了 2026 年，前沿模型的 ECE 仍然在 5%-15% 之间。这说明它们还是过于自信。RLHF 甚至会让校准变得更差，因为模型学会了用自信的回答换取更高的奖励。如果部署场景要求模型知道自己不知道的东西，可以试试以下方法：

通过 SFT 撤回一些后训练带来的过度自信。用那些基模型不确定但后训练模型变得自信的案例，训练模型回答“我不知道”。

推理时对 logprobs 使用 temperature scaling。在验证集上调校温度参数，这种方法成本低，通常能把 ECE 减少一半。

采样多个补全结果，用答案的熵作为置信度的代理。比如对一个问题采样 $N=10$ 次，如果 9 次答案一致，那就认为置信度高；如果答案分布是 5/10 或 0/0 这种分散的情况，那就认为置信度低。

拟合一个校准模型。基于（model_logprob、prompt_features）→ 实际正确性，在标注数据上训练。

对于 RAG 系统来说，*检索置信度*（比如 top-k 相似度得分、reranker 得分）比 LLM 自己生成的 token 更能反映真实的置信度。把这个信号接入管道里就好。
## 长格式评估：benchmark 漏的

大多数公开 benchmark 只测短答案。但实际生产中，长格式生成更常见，比如总结、草稿、500 行以上的代码块。

几个针对长格式的 benchmark 值得关注：

- **AlpacaEval / AlpacaEval 2**（Li 等，2023）：805 条指令，用 LLM-as-judge 对比参考模型（最初是 text-davinci-003，后来换成 GPT-4）。Length-controlled 版本（Dubois 等，2024）解决了长度偏差问题。
- **MT-Bench**（Zheng 等，2023）：80 组多轮对话，覆盖 8 类任务。LLM-as-judge 在评分前会给出 chain-of-thought 的理由。
- **Arena Hard**（lmsys，2024）：从 Chatbot Arena 的真实流量中提炼出 500 道难题。LLM-as-judge 的结果和人类排名高度一致。

内部做长格式评估时，我的实践经验是这样的：

1. 收集 50 到 200 个提示词，覆盖实际工作负载。
2. 设计评分标准，包含 5 到 10 个维度，比如准确性、完整性、格式合规性、语气、幻觉率等。
3. 用一个强评判模型对每个维度单独打分，并附一句简短理由。
4. 汇总成综合分数，但决定是否上线时，必须看每个维度的详细表现。

按维度拆解很关键。一个模型在完整性上提高 5 分，但在幻觉问题上下降 3 分，这不一定算改进。最终是否上线，取决于哪个维度对你的产品更重要。
## Eval pipeline as code

成熟的评估系统，运维规范要像 CI 流水线一样清晰：

```
[eval_set.jsonl 放在版本控制中]
     ↓
[runner：对 N 个候选模型 × M 道题目并行处理]
     ↓
[grader：程序打分 + LLM-as-judge + 可选人工抽查]
     ↓
[结果存储：每次运行的分数，按题目和维度细分]
     ↓
[diff 视图：对比候选模型和基线，突出回归问题]
     ↓
[gate：回归超过阈值，阻止部署]
```

2026 年能用的工具有这些：**Promptfoo**，简单易用，YAML 驱动，适合本地迭代。**Inspect AI**，英国 AISI 的框架，专攻安全评估。还有 **OpenAI evals** 和 **DeepEval**。如果喜欢折腾，可以用 pytest 自己搭一个。有没有框架不重要，重要的是得有一个。最糟糕的做法是写个 notebook，等有人想起来才跑一次评估。这种踩过的坑，真在生产里跑过就知道多耽误事。
## 小结与下一篇

公开 benchmark 超过 80 分的，基本都是噪声，很多还被污染了。我建议从真实流量里手工整理一个评估集，按难度分层，像维护核心代码一样维护它。用 LLM-as-judge 时，记得校正偏差。可以调整顺序、控制长度、换 judge 家族，或者用多模型投票。任务能验证的话，优先用程序化评估方法。线下评估漏掉的部分，靠生产环境的 A/B 测试补上。如果需要提前看结果，记得用序列检验。评估集要持续更新，移除过时或太简单的问题。

别忘了校准。实际部署中，置信度和准确率一样重要。Post-training 往往会让置信度下降。长文本生成任务先按维度评分，再综合打分。

下一篇：**安全与对齐**。拒绝行为、RLHF 目标、谄媚问题、红队测试方法、幻觉指标以及 constitutional AI。
## 参考资料

- Zheng, L. 等（2023）。*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*。NeurIPS 2023。https://arxiv.org/abs/2306.05685
- Zhou, K. 等（2023）。*Don't Make Your LLM an Evaluation Benchmark Cheater*。https://arxiv.org/abs/2311.01964
- Sainz, O. 等（2023）。*NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*。EMNLP 2023 Findings。https://arxiv.org/abs/2310.18018
- Guo, C. 等（2017）。*On Calibration of Modern Neural Networks*。ICML 2017。https://arxiv.org/abs/1706.04599
- Cobbe, K. 等（2021）。*Training Verifiers to Solve Math Word Problems*。https://arxiv.org/abs/2110.14168
- Hendrycks, D. 等（2021）。*Measuring Mathematical Problem Solving With the MATH Dataset*。NeurIPS 2021 Datasets。https://arxiv.org/abs/2103.03874
- Hendrycks, D. 等（2020）。*Measuring Massive Multitask Language Understanding*（MMLU）。ICLR 2021。https://arxiv.org/abs/2009.03300
- Chen, M. 等（2021）。*Evaluating Large Language Models Trained on Code*（HumanEval）。https://arxiv.org/abs/2107.03374
- Austin, J. 等（2021）。*Program Synthesis with Large Language Models*（MBPP）。https://arxiv.org/abs/2108.07732
- Jimenez, C. 等（2023）。*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* https://arxiv.org/abs/2310.06770
- Zhuo, T. Y. 等（2024）。*BigCodeBench*。https://arxiv.org/abs/2406.15877
- Rein, D. 等（2023）。*GPQA: A Graduate-Level Google-Proof Q&A Benchmark*。https://arxiv.org/abs/2311.12022
- Hsieh, C.-P. 等（2024）。*RULER: What's the Real Context Size of Your Long-Context Language Models?* https://arxiv.org/abs/2404.06654
- Zhou, J. 等（2023）。*Instruction-Following Evaluation for Large Language Models*（IFEval）。https://arxiv.org/abs/2311.07911
- Li, X. 等（2023）。*AlpacaEval*。https://github.com/tatsu-lab/alpaca_eval
- Dubois, Y. 等（2024）。*Length-Controlled AlpacaEval*。https://arxiv.org/abs/2404.04475
- Wang, P. 等（2023）。*Large Language Models are not Fair Evaluators*。ACL 2024。https://arxiv.org/abs/2305.17926
- Panickssery, A. 等（2024）。*LLM Evaluators Recognize and Favor Their Own Generations*。https://arxiv.org/abs/2404.13076
- OpenAI（2024）。*Introducing SimpleQA*。https://openai.com/index/introducing-simpleqa/
- LMSYS（2024）。*Chatbot Arena*。https://arxiv.org/abs/2403.04132
- White, J. 等（2024）。*LiveBench*。https://livebench.ai/
- UK AISI（2024）。*Inspect: An OSS framework for large language model evaluations*。https://inspect.ai-safety-institute.org.uk/
