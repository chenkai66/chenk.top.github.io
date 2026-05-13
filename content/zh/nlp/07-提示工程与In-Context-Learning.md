---
title: "自然语言处理（七）：提示工程与In-Context Learning"
date: 2025-10-31 09:00:00
tags:
  - NLP
  - 提示工程
  - LLM
  - In-Context Learning
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从提示结构、思维链到 Self-Consistency 与 ReAct：一套关于 In-Context Learning 的工作原理、必须正面应对的方差问题，以及能扩展到生产系统的提示模式。"
disableNunjucks: true
series_order: 7
translationKey: "nlp-7"
polished_by_qwen_max: true
---
同一个模型，既可能给出精准而深刻的回答，也可能自信满满地“一本正经胡说八道”。关键在于你如何引导它，而不是模型的权重。简单输入“分析这段文本”通常只会得到泛泛而谈的总结；但在提示中明确角色、提供清晰示例并规定严格输出格式，则更可能得到一个结构化的 JSON，直接供下游解析器使用。**提示工程的核心是将这种从偶然到必然的差距转化为一套可重复、可操作的方法论。**

让这一切成为可能的机制叫做 In-Context Learning （上下文学习，简称 ICL）。当你在提示中加入几个示例时，模型并不会重新训练，而是通过这些示例调整其前向传播的条件分布，从而**推断出任务的具体要求**。理解 ICL 的能力和局限，是区分开发者是在与模型“较劲”还是在“驾驭”模型的关键。

本文是 NLP 系列的第七篇，假设你已经对 Transformer 解码器逐 token 生成的过程（第四篇）及自回归语言模型（第六篇）有所了解。内容基于已发表的研究成果，但需注意，提示工程领域的研究噪声较大，数据和结论高度依赖具体模型和数据集。因此，图中的柱状图仅作示意参考，不应视为基准测试的绝对结果。


<!-- wanx-hero -->
![自然语言处理（七）：提示工程与In-Context Learning — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/prompt-engineering-icl/illustration_1.png)
## 你将学到什么

![自然语言处理（七）：提示工程与In-Context Learning — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/prompt-engineering-icl/illustration_2.png)

- **提示的组成结构**：五个可以灵活组合的部分（system、指令、示例、查询、格式说明），每个部分分别能为你带来什么价值。
- **三种核心范式**：零样本（zero-shot）、少样本（few-shot）和思维链（chain-of-thought），它们各自适用的场景是什么，以及在 token 消耗上的代价如何。
- **ICL 的理论基础**：为什么一个未经训练的模型仍然能够通过提示中的示例“学习”，它实际上捕捉到了哪些关键信号。
- **方差问题**：仅因提示格式或顺序的不同，准确率可能会有多大波动，以及如何科学地评估这种变化。
- **Self-Consistency 方法**：通过对多条推理路径进行采样，将随机解码器转化为一种集成方法，从而提升结果的稳定性。
- **ReAct 框架**：将推理过程与工具调用交替结合，这是构建现代智能体（agents）的重要基石。
- **小型提示管理系统**：包括提示注册表、 A/B 测试工具、版本控制机制等，确保团队中的一组提示能够长期有效且易于维护。
## 前置知识

![NLP (7): Prompt Engineering and In-Context Learning — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/illustration_2.png)

- 需要对大语言模型有一定的认识，建议先阅读[第 6 部分：GPT 与生成式模型](/zh/nlp/06-gpt与生成式语言模型)。
- 掌握基础的 Python 知识，能够轻松读懂简短的代码片段。
- 拥有任意 LLM API 的访问权限，例如 OpenAI、 Anthropic，或者使用开源权重的模型。

---
## 1. 提示的组成部分

提示，简单来说，就是**一段模型用来生成结果的文本字符串**。至于其他内容——比如“系统”和“用户”的角色划分、函数描述、检索结果等——其实都是 API 在分词之前拼接成的一个完整序列。把提示看作一个由命名块组成的纯文本，是最直观也最实用的理解方式。

![结构化提示的组成](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-提示工程与In-Context-Learning/fig1_prompt_anatomy.png)

以下五个部分并非强制要求，但在实际生产环境中，提示通常会包含其中的一部分，顺序也大致如下：
1. **系统/角色定义**：设定角色性格、拒绝策略、语气风格以及长度限制等内容。这部分在多次请求中通常是固定的，因此可以很好地利用缓存。
2. **任务指令**：用一句简短的祈使句明确告诉模型要做什么。
3. **少样本示例**：提供输入到输出的示例对，这是 ICL（In-Context Learning）的核心信号。
4. **用户查询**：需要处理的实际输入内容。
5. **格式说明**：规定输出格式的 schema，例如 JSON、正则匹配标签或表格等。

一个实用的提示构造器实现如下：

```python
from dataclasses import dataclass, field

@dataclass
class Prompt:
    system: str = ""
    instruction: str = ""
    examples: list[tuple[str, str]] = field(default_factory=list)
    query: str = ""
    format_spec: str = ""

    def render(self) -> str:
        parts: list[str] = []
        if self.system:
            parts.append(f"[SYSTEM]\n{self.system}")
        if self.instruction:
            parts.append(f"[TASK]\n{self.instruction}")
        if self.examples:
            shots = "\n\n".join(
                f"Input: {x}\nOutput: {y}" for x, y in self.examples
            )
            parts.append(f"[EXAMPLES]\n{shots}")
        if self.format_spec:
            parts.append(f"[FORMAT]\n{self.format_spec}")
        if self.query:
            parts.append(f"[INPUT]\n{self.query}\nOutput:")
        return "\n\n".join(parts)
```

新手容易忽略的两个关键点：

- **顺序影响效果**。示例放在格式说明之后、用户查询之前，往往比放在开头效果更好。这是因为解码器存在“近因偏置”，越靠近查询的内容对生成结果的影响越大。
- **固定前缀，变化后缀**。将所有不变的内容（如系统定义、示例、格式说明）放在前面，方便 KV 缓存和提示缓存复用；而变化的部分（如用户查询）则放在最后。

### 四条经得起实践检验的原则

经过大量实际项目的锤炼，我仍然认为以下四条原则值得推荐：

1. **清晰胜过聪明**。与其写“分析这段文本”，不如明确为“将文本分类为 {正面， 负面， 中性} 之一，并以 JSON 格式返回”。你是在和模型可能的所有解读竞争，模糊只会让模型自行其是。
2. **具体带来确定性**。明确告诉模型“不要做什么”，并规定不确定时如何输出（例如“如果文档里找不到答案，请返回 `{\"answer\": null}`”）。模型对负向约束的遵守能力比想象中更强。
3. **上下文要完整**。如果回答需要某个模型可能不知道的定义，就直接在提示中带上。这比让模型猜错便宜得多。
4. **按需设定角色**。“你是一名资深安全审计员”确实能在代码评审任务中收紧输出分布，但它不是万能咒语——在通用任务中滥用只会徒增 token 消耗。
## 2. 零样本、少样本、思维链

这三种方法构成了所有其他技术的基础框架。

![三种提示范式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-提示工程与In-Context-Learning/fig2_prompting_paradigms.png)

### 零样本

直接描述任务，不提供任何示例。模型完全依赖预训练和指令微调阶段学到的知识来完成任务。

```python
zero_shot = """将句子的情感分类为 positive、negative 或 neutral。

句子：这部电影剧情精彩、表演出色。
情感："""
```

零样本适合那些模型已经非常熟悉的任务，比如情感分析、翻译或短文本摘要，尤其是当你对延迟和成本比较敏感时。它的缺点是输出格式不够稳定，可能会返回 "Positive sentiment"、"POSITIVE" 或一段分析文字。可以通过一句严格约束的指令固定格式，例如：“仅返回 {positive, negative, neutral} 中的一个词。”

### 少样本

少样本是在查询前提供 $k$ 个示例，这是 ICL 的经典场景。

```python
few_shot = """将句子的情感分类为 positive、negative 或 neutral：

句子：今天天气真好，阳光明媚。
情感：positive

句子：这个产品质量差、价格还贵。
情感：negative

句子：明天会下雨。
情感：neutral

句子：这家餐厅的服务令人印象深刻。
情感："""
```

这些示例实际上完成了三件事：

- **明确任务**：消除歧义。例如，“翻译”可能指音译、改写或重写，但通过两个示例就能清楚表达具体需求。
- **统一格式**：每个示例的输出部分相当于一个模板，模型会模仿这种格式。
- **限定标签范围**：示例中出现的标签集合会成为模型的实际输出词汇表，即使你没有显式列出所有标签。

一个反直觉的发现是：**标签本身没那么重要**。 Min 等人（2022）的研究表明，在很多任务中，随机打乱少样本示例中的正确标签对性能的影响很小——真正关键的是输入分布和标签空间。这并不是说标签完全无关紧要（在复杂任务上仍然重要），而是提醒我们不要过度纠结标签的准确性，而应更关注覆盖范围和格式。

### 思维链（Chain-of-Thought）

对于多步骤问题，让模型先写出推理过程，再给出答案。

```text
问题：一本书有 120 页。第一天读了 30 页，第二天读了第一天的两倍，
第三天读了剩下页数的一半。第三天读了多少页？

一步步推导：
1. 第一天：30 页。
2. 第二天：2 × 30 = 60 页。
3. 累计已读：30 + 60 = 90 页。
4. 剩下：120 − 90 = 30 页。
5. 第三天：30 / 2 = 15 页。

答案：15 页。
```

原理并不复杂，也没有什么神秘之处。每个生成的 token 都会改变下一个 token 的上下文条件。由于模型是自回归的，把中间状态（如 “累计已读： 90”）写出来后，后续 token 包括最终答案都能直接利用这些信息。如果不写出来，模型只能通过一次固定深度网络的前向计算得出所有中间结果。 CoT 实际上是用输出 token 的代价换取了更多的串行计算能力。

![思维链推理流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-提示工程与In-Context-Learning/fig3_cot_flow.png)

从概率角度可以更清晰地理解 CoT：对于问题 $x$ 和答案 $a$，通过对潜在推理链 $z$ 求边缘概率：
$$P(a \mid x) \;=\; \sum_z P(z \mid x)\, P(a \mid x, z).$$贪心 CoT 是选择一个 $z$ 并希望它是正确的。**Self-Consistency**（第 5 节）则是通过采样多个 $z$ 来近似求和。

CoT 在什么时候有用，什么时候反而拖后腿：

- **有用**：算术题、多跳问答、代码推理等涉及中间状态的任务。
- **持平或更糟**：简单分类、检索、单条事实查询。推理铺垫增加了 token 数量和延迟，还可能让模型自己绕进错误答案。
- **现代长上下文推理模型**（在 CoT 风格轨迹上专门训练的模型）已经内化了这种能力，显式 CoT 提示仍然有帮助，但提升幅度比 2022 年 GPT-3 那一代小得多。
## 3. ICL 的工作原理猜想

为什么在提示中加入示例会改变模型的行为？毕竟，模型的权重是固定的，并不会因为输入的变化而调整。虽然目前还没有一个统一的答案，但领域内逐渐形成了三种互补的解释，它们结合起来最接近于一种共识。

**1. 隐式的任务推断**  
预训练语言模型在训练过程中见过大量以文本形式表达的任务，比如问答对、代码注释对、翻译对等。在推理阶段，提示中的示例实际上起到了一种“任务线索”的作用，帮助模型推断接下来要完成的任务类型。从贝叶斯的角度来看，这些示例相当于对任务分布的后验更新（Xie et al., 2022）。通过少量示例，模型能够更清晰地估计 $P(\text{task} \mid \text{prompt})$，从而调整其行为。

**2. 注意力机制中的隐式梯度下降**  
一些研究（Akyurek et al., von Oswald et al., 2022-2023）表明，注意力机制可以在某些简单场景下模拟一步梯度下降的过程，针对的是提示中隐含的线性回归任务。尽管这种机制在复杂任务中的适用性有限，但它提供了一个直观的理解：注意力机制可能在快速适应新信息的过程中扮演了某种优化角色。

**3. 模式匹配与复制**  
最直接的解释是，模型通过归纳头（induction heads, Olsson et al., 2022）从上下文中复制前面出现过的模式。少样本提示为模型提供了一个可以直接模仿的模板，从而引导其生成符合预期的输出。

无论你更倾向于哪种解释，实际应用中的结论都是一致的：

- **示例越多越好，但边际收益递减明显**。前 2-4 条示例带来的提升最为显著，后续增加示例的效果则逐渐减弱。
- **分布覆盖比单个示例的精巧设计更重要**。相比那些单独设计得非常巧妙的示例，覆盖输入空间多样性的示例更能有效提升模型表现。
- **越靠后的示例影响越大**。提示中最后一条示例对模型的影响尤为突出。
- **选项位置偏置确实存在**。在多选题任务中，模型往往会系统性地偏向 A 选项或最近的选项，具体偏好取决于模型本身的设计。
## 4. 方差问题：必须正视的现实

在各种宣传材料中，几乎没人会告诉你一个令人不安的事实：**仅仅因为一些看似无关紧要的选择——比如示例的排列顺序、用 `Q:` 还是 `Input:`、答案是否加引号——提示的准确率就可能上下波动 10 到 30 个百分点。**

![提示对格式与顺序的敏感性](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig5_prompt_sensitivity.png)

Lu 等人（2022）将这种现象称为**顺序敏感性**。他们的研究表明，在分类任务中，同一个模型使用相同的示例集合，仅仅改变示例的排列顺序，性能可以从接近随机水平一路飙升到接近当前最优（SOTA）。 Sclar 等人（2024）进一步扩展了这一研究，提出了**格式敏感性**的概念——例如，将 `Q:/A:` 替换为 `Question:/Answer:`，准确率可能会出现两位数的波动。

这在实际操作中意味着什么？以下是几点建议：

- **评估提示时，务必使用至少 50 到 100 条独立样本。** 单个例子的结果毫无参考价值。
- **比较两个提示时，运行多个随机种子或不同的示例顺序。** 报告中位数，而不是最佳结果。
- **尽早固定提示格式，并将其视为一次有版本号的更新，而非随意调整的小改动。**

以下是一个简单的评估框架：

```python
import statistics, random
from typing import Callable

def evaluate(
    build_prompt: Callable[[str, list], str],
    model_call: Callable[[str], str],
    cases: list[dict],
    examples_pool: list[dict],
    k: int = 4,
    n_seeds: int = 5,
) -> dict:
    """通过 n_seeds 次不同的示例采样，返回准确率的中位数及其波动范围。"""
    accs = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        shots = rng.sample(examples_pool, k)
        correct = 0
        for case in cases:
            out = model_call(build_prompt(case["input"], shots))
            if out.strip() == case["expected"].strip():
                correct += 1
        accs.append(correct / len(cases))
    return {
        "median": statistics.median(accs),
        "min": min(accs),
        "max": max(accs),
        "spread": max(accs) - min(accs),
    }
```

如果 `spread > 0.05`，说明你的提示不够稳定，任何单次运行的结果都只是噪音。

### 示例数量到底需要多少？

![准确率随示例数量的变化曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/prompt-engineering-icl/fig4_shots_saturation.png)

无论任务类型如何，规律始终一致：最初的几条示例效果显著，收益通常在 $k \approx 4$ 到 $8$ 之间趋于饱和；而当 $k \approx 16$ 时，增加更多示例基本上只是浪费 token。这里总结两条实用经验：

- 对于类别数 $\le 5$ 的分类任务，$k = 2 \times \text{类别数}$ 是一个不错的默认值。
- 对于生成任务，$2$-$3$ 条高质量示例的效果几乎总是优于 $10$ 条平庸示例。

挑选示例时需注意两点：**输入侧要覆盖尽可能多样的分布**，**输出侧要清晰无歧义**。记住，模型会忠实复制你的格式，包括其中的错误。
## 5. Self-consistency：把解码器变成集成模型

单条 CoT 推理链在第二步就可能偏离正轨，错误会一路累积到最终结果。 Self-consistency （Wang 等， 2022）通过一个简单却强大的方法解决了这个问题：**对同一个问题生成 $k$ 条不同的推理路径，然后通过多数投票选出最可信的答案**。

![自洽性：多路径采样、多数投票](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-提示工程与In-Context-Learning/fig7_self_consistency.png)

具体来说，假设我们有一个问题 $x$，并从中采样了 $k$ 条推理路径 $z_1, \dots, z_k$，每条路径都会得出一个答案 $a_i$。最终答案 $\hat{a}$ 的计算公式为：  
$$\hat{a} \;=\; \arg\max_a \sum_{i=1}^{k} \mathbb{1}[a_i = a].$$  

这种方法可以看作是对边际分布 $\sum_z P(z \mid x)\, P(a \mid x, z)$ 的蒙特卡洛近似。其背后的原理是：**错误往往是分散的，而正确答案通常是收敛的**——错误的推理路径往往指向不同的错误答案，而正确的推理路径则会汇聚到同一个正确答案上。

```python
from collections import Counter

def self_consistency(
    model_call,
    prompt: str,
    k: int = 8,
    temperature: float = 0.7,
    extract_answer=lambda s: s.strip().split()[-1],
) -> dict:
    """采样 k 条 CoT 推理路径，并对最终答案进行多数投票"""
    answers = []
    for _ in range(k):
        out = model_call(prompt, temperature=temperature)
        answers.append(extract_answer(out))
    counts = Counter(answers)
    top, n = counts.most_common(1)[0]
    return {"answer": top, "confidence": n / k, "votes": dict(counts)}
```

在实际应用中，有两点需要特别注意：

- **温度参数至关重要**。建议将温度 $T$ 设置在 $[0.5, 0.9]$ 范围内。如果 $T = 0$，所有样本都会坍缩成同一条推理路径，集成模型也就失去了意义。
- **得票率反映了答案的置信度**。一个 5/5 全票通过的答案显然比 3/5 微弱多数的答案更值得信赖。把得票率暴露给下游用户，这是最直接且成本最低的可靠性指标之一。

**Tree of Thoughts**（Yao 等， 2023）进一步扩展了这一思路：不再独立采样线性推理路径，而是构建一棵推理树，探索部分推理步骤，剪掉低分分支，并通过搜索找到最优解。这种方法能力更强，但计算成本也更高，只有在 Self-consistency 方法达到瓶颈时才建议使用。
## 6. ReAct：推理 + 行动

Self-Consistency 的核心在于提升模型对已有知识的运用能力，而 **ReAct**（Yao 等， 2022）则更进一步，解决了一个更具挑战性的问题：当模型需要**外部信息或执行具体操作**时该如何应对？ ReAct 的输出模式通过交替使用三个关键模块来实现这一目标：

- **Thought （思考）**：以自由形式对当前状态进行分析和推理。
- **Action （行动）**：调用结构化工具，例如 `search("..."); calc("..."); read_file("...")`。
- **Observation （观察）**：工具返回的结果，这些结果会被反馈到下一轮输入中，供模型继续推理。

![ReAct：Thought / Action / Observation 的循环机制](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-提示工程与In-Context-Learning/fig6_react_pattern.png)

整个循环会在模型输出 `Answer:` 而非 `Action:` 时终止。如今流行的 Agent 框架（如 LangChain Agents、 OpenAI 的 function calling 和 Anthropic 的 tool use）本质上都是基于这一模式的变体——唯一的区别是，这些框架通常将结构化工具调用转移到 JSON 格式的 API 字段中，而不是从自由文本中解析。

以下是一个抓住核心逻辑的最小实现：

```python
import re
from typing import Callable

class ReActAgent:
    def __init__(self, model_call: Callable[[str], str],
                 tools: dict[str, Callable[[str], str]]):
        self.model_call = model_call
        self.tools = tools

    def _system(self) -> str:
        names = ", ".join(self.tools)
        return (
            "可用工具：" + names + "。\n"
            "请严格遵循以下格式：\n"
            "Thought: <推理内容>\n"
            "Action: <工具名>(<参数>)\n"
            "Observation: <工具返回结果>\n"
            "...重复上述步骤...\n"
            "Answer: <最终答案>"
        )

    def run(self, question: str, max_steps: int = 6) -> str:
        ctx = f"{self._system()}\n\n问题：{question}\n"
        for _ in range(max_steps):
            out = self.model_call(ctx, stop=["\nObservation:"])
            ctx += out
            if "Answer:" in out:
                return out.split("Answer:")[-1].strip()
            m = re.search(r"Action:\s*(\w+)\((.*?)\)", out)
            if not m:
                return "(无法解析 Action)"
            name, arg = m.group(1), m.group(2).strip()
            obs = self.tools[name](arg) if name in self.tools \
                  else f"未知工具：{name}"
            ctx += f"\nObservation: {obs}\n"
        return "(达到最大步数限制)"
```

在实际生产环境中，有三点需要特别注意：

- **停止标记（Stop tokens）**：在 `\nObservation:` 处停止生成，防止模型“脑补”工具的返回结果。
- **工具异常处理**：对工具调用进行封装，将**异常信息**作为观察结果返回。模型通常能够根据这些信息自行调整并纠正错误。
- **步数限制**：必须设置最大迭代次数。如果没有限制， Agent 可能会陷入无限循环，最终耗尽你的计算资源。
## 7. 构建提示系统
几个高质量的提示可以称为脚本，但能够经得起团队人员流动、模型升级以及长达四个月 A/B 测试考验的，才算是一个真正的系统。两者之间的差距，往往体现在以下三个习惯上。

### 将提示当作代码来管理

为提示添加版本号，进行代码评审，并将其存储在代码仓库中，而不是随意丢在共享文档里。以下是一个最简注册表的实现：

```python
from pathlib import Path

class PromptRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, str] = {}

    def get(self, name: str, version: str = "latest") -> str:
        key = f"{name}@{version}"
        if key not in self._cache:
            path = self.root / name / f"{version}.txt"
            self._cache[key] = path.read_text()
        return self._cache[key]
```

每个调用点都应标记 `(prompt_name, version)`。当修改提示时，记得更新版本号；老代码会继续使用旧提示，直到你手动完成迁移。

### 上线前必须进行评估

任何进入生产环境的提示，都需要准备好三样东西：

- 一个 **golden set**：包含 50 到 200 条输入及其期望输出（或者评分标准）。
- 一个自动评估器：可以是精确匹配、 JSON schema 校验，或者用 LLM 当裁判（裁判提示也需要固定版本）。
- 一个回归测试的 CI 步骤：运行当前提示和新提示，如果准确率下降，则拒绝合并。

```python
def regression_check(old_prompt: str, new_prompt: str,
                     cases, model_call, judge) -> bool:
    old_acc = mean(judge(case, model_call(old_prompt + case.input))
                   for case in cases)
    new_acc = mean(judge(case, model_call(new_prompt + case.input))
                   for case in cases)
    return new_acc >= old_acc - 0.01   # 允许 1% 的准确率下降
```

### 有目的地组合技术

最大的收益通常来自于技术的合理叠加：

- **角色 + 少样本 + 格式说明**：适用于所有结构化输出任务。
- **CoT + 自洽性**：适用于任何不能出错的推理任务。
- **ReAct + 检索**：适用于需要模型不具备的事实的任务。

但不要为了叠加而叠加。每增加一个模块都会消耗 token、增加延迟，并多一个可能出错的地方。

```python
def build_advanced(question: str, examples: list[dict],
                   *, role: str = "", use_cot: bool = True) -> str:
    parts: list[str] = []
    if role:
        parts.append(f"你是 {role}。")
    parts.append("请解决以下问题。")
    for ex in examples:
        parts.append(f"Q: {ex['question']}")
        if use_cot and "reasoning" in ex:
            parts.append(f"Reasoning: {ex['reasoning']}")
        parts.append(f"A: {ex['answer']}\n")
    if use_cot:
        parts.append("先一步步推理，再在以 'A:' 开头的一行里给出最终答案。")
    parts.append(f"Q: {question}\n")
    return "\n".join(parts)
```
## 实战：一个能上线的情感分类器

将前面提到的各个部分整合起来，完成一个简单但贴近实际的小任务。

```python
import json, re
from collections import Counter

SYSTEM = (
    "你是一个严谨的文本分类器。只返回合法的 JSON 格式结果，不要添加任何注释或说明。"
)

EXAMPLES = [
    ("这部电影太棒了，演员表现出色，剧情扣人心弦。",
     {"sentiment": "positive", "confidence": 0.95}),
    ("电池续航不到三小时，软件还频繁崩溃，体验极差。",
     {"sentiment": "negative", "confidence": 0.92}),
    ("包裹周二准时送达，没什么特别的问题。",
     {"sentiment": "neutral", "confidence": 0.88}),
]

FORMAT_SPEC = (
    '输出格式必须为：{"sentiment": "<positive|negative|neutral>", '
    '"confidence": <0.0-1.0>}'
)

def build_prompt(text: str) -> str:
    shots = "\n".join(
        f"Text: {x}\nOutput: {json.dumps(y, ensure_ascii=False)}"
        for x, y in EXAMPLES
    )
    return (
        f"[SYSTEM]\n{SYSTEM}\n\n"
        f"[FORMAT]\n{FORMAT_SPEC}\n\n"
        f"[EXAMPLES]\n{shots}\n\n"
        f"[INPUT]\nText: {text}\nOutput:"
    )

def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None

def classify(model_call, text: str, *, k: int = 5) -> dict:
    """通过 Self-Consistency 提升情感分类的稳定性。"""
    prompt = build_prompt(text)
    labels, confs = [], []
    for _ in range(k):
        parsed = parse(model_call(prompt, temperature=0.5))
        if parsed and "sentiment" in parsed:
            labels.append(parsed["sentiment"])
            confs.append(float(parsed.get("confidence", 0.5)))
    if not labels:
        return {"sentiment": "unknown", "confidence": 0.0, "votes": {}}
    counts = Counter(labels)
    top, n = counts.most_common(1)[0]
    return {"sentiment": top,
            "confidence": n / len(labels),
            "model_confidence": sum(confs) / len(confs),
            "votes": dict(counts)}
```

这段代码涵盖了文章中提到的所有关键点：一个明确的角色定义（system）、清晰的输出格式规范、三条多样化的少样本示例、一个能够容忍模型噪声的解析器，以及通过 Self-Consistency 抑制标签波动的逻辑。单独来看，每个部分都不复杂，但它们组合在一起，却能让整个流水线的准确率从 60% 提升到 90%。
## 常见问题
### 提示越长效果就越好吗？

并不是。超过一定长度后，额外的上下文不仅会干扰模型的理解，还会显著增加延迟和成本。建议从简短的提示开始，逐步添加那些在评估集上能够带来实际提升的内容。

### 示例数量应该如何选择？

对于分类任务，通常 2 到 5 个示例就够了；而对于生成任务， 1 到 3 个示例可能更合适。超过 8 个示例往往不会带来更多收益，反而可能因为格式不一致引入额外的方差。记住，一定要通过实验验证，不要凭直觉下结论。

### CoT 对所有任务都有效果吗？

并不是。 CoT （Chain of Thought）在多步骤推理任务中表现优异，比如数学问题、逻辑推导、代码生成以及多跳问答。但在简单的分类任务或事实查询中，它只会徒增噪声和 token 消耗，而不会带来任何实际好处。

### 温度参数该如何设置？

如果是需要确定性输出的任务（如分类、信息抽取），建议将温度设置为 $T \in [0.0, 0.2]$；对于需要平衡生成质量的任务，推荐 $T \in [0.5, 0.7]$；而在进行 Self-Consistency 采样时，可以选择 $T \in [0.7, 0.9]$——此时我们正是希望生成路径具备一定的多样性。

### 提示工程可以完全取代微调吗？

在很多场景下，提示工程确实可以替代传统的微调，特别是在指令跟随类任务中。然而，有些情况下仍然需要微调：(a) 在大规模应用中需要高度一致的行为，而提示漂移不可接受；(b) 需要一个更小、更高效的模型来在特定领域达到大模型的效果；(c) 当基础模型难以通过提示实现某些行为改变时。

### 示例顺序会影响结果吗？

影响比你想象的要大。近因效应是真实存在的——最后一条示例对模型的影响最大。因此，在设计提示时，务必尝试多种顺序排列，并以中位数结果作为参考。

### 是否需要担心 Prompt Injection？

当然需要。只要提示中包含来自外部的不受控文本（例如用户输入、检索到的文档或工具输出），就必须谨慎对待。这些内容应被视为数据，绝不能让它们篡改你的指令。这是一个独立且重要的主题，我们会在后续关于 Agent 和安全性的章节中详细探讨。