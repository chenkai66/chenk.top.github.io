---
title: "自然语言处理（七）：提示工程与In-Context Learning"
date: 2024-06-07 09:00:00
tags:
  - NLP
  - 提示工程
  - LLM
  - In-Context Learning
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 7
  total: 12
lang: zh-CN
mathjax: true
description: "从提示结构、思维链到 Self-Consistency 与 ReAct：一套关于 In-Context Learning 的工作原理、必须正面应对的方差问题，以及能扩展到生产系统的提示模式。"
disableNunjucks: true
---

同一个模型，可以给出一针见血的分析，也能一本正经地胡说八道。区别几乎从不在权重，而在你怎么问。一句"分析一下这段文本"换来的多半是泛泛的总结；同样的请求加上一个角色、两条干净的示例和一段严格的输出 schema，得到的就是下游解析器能直接消费的 JSON。**提示工程的意义，是把这种差距从"碰运气"变成可重复、可度量的工程实践。**

让这一切发生的机制叫做 In-Context Learning（上下文学习，ICL）。当你把示例塞进提示，模型并没有重新训练，它只是把这几条示例作为前缀，让一次前向计算的条件分布发生改变——本质上是从示例里**推断出任务**。理解 ICL 能做什么、不能做什么，决定了你是在和模型对抗，还是在驾驭它。

这一篇是 NLP 系列的第 7 部分，默认你已经大致清楚 Transformer 解码器是怎么逐 token 生成的（第 4 部分），以及自回归语言模型是什么（第 6 部分）。下文所有结论都尽量贴着已有研究，但要提前说明：提示工程领域的文献噪声相当大，很多数字都强烈依赖具体模型和数据集。图里的柱状图请当成"示意形状"看，而不是任何一个 benchmark 的精确数字。

## 你将学到什么

- **提示结构**：构成一个完整提示的五个可组合块（system、指令、示例、查询、格式说明）各自的用途。
- **三种范式**：零样本、少样本、思维链——分别适合什么场景，token 成本是多少。
- **ICL 的工作理论**：为什么"不训练"也能"学"，模型究竟在示例里捕捉了什么信号。
- **方差问题**：示例顺序和格式带来的准确率波动到底有多大，怎么测。
- **Self-Consistency**：用采样把一个随机解码器变成集成方法。
- **ReAct**：把推理和工具调用穿插起来，这是现代 Agent 的基本骨架。
- **小型提示系统**：注册表、A/B 评估、版本治理——让一组提示能在团队里活过半年。

## 前置知识

- 熟悉大语言模型——见[第 6 部分：GPT 与生成式模型](/zh/自然语言处理-六-GPT与生成式语言模型/)。
- 基础 Python，能看懂短代码。
- 任意一个 LLM API（OpenAI、Anthropic 或开源权重模型）。

---

## 1. 一个提示由什么组成

提示本质上是**一段模型要去条件化的文本字符串**。所谓的 system / user 角色、function 描述、检索结果，都只是 API 在 tokenize 之前拼成的一段结构化文本。把提示当成"带命名块的纯文本"是最干净的心智模型。

![结构化提示的解剖图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig1_prompt_anatomy.png)

下面五个块不是必须全部出现，但生产环境里的提示通常包含其中的一个子集，顺序也大致如此：

1. **System / 角色**：设定人格、拒答策略、语气、长度预算。请求间稳定，因此对 KV 缓存和 prompt cache 友好。
2. **任务指令**：用一句祈使句说明目标。
3. **少样本示例**：输入-输出示例对，是 ICL 的主要信号来源。
4. **用户查询**：要处理的实际输入。
5. **格式说明**：固定输出形态的 schema（JSON、可正则匹配的标签、表格等）。

一个务实的提示构造器：

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

两个新手常忽略的细节：

- **顺序很重要**。把示例放在格式说明之后、查询之前，往往比塞在最顶部要好——解码器有显著的"近因偏置"，越靠近查询的内容影响越大。
- **稳定前缀，可变后缀**。所有不变的内容（system、示例、格式）放在最前面，方便 KV 缓存或 prompt cache 复用；变化的输入放在最后。

### 四条经得起生产考验的原则

我做了不少提示之后，下面四条仍然是我会教别人的：

1. **清晰胜过聪明**。把"分析这段文本"换成"将文本分类为 {正面, 负面, 中性} 之一，并以 JSON 返回"。你在和模型脑中所有可能的解读竞争，含糊就是把话语权交给它。
2. **具体性买的是确定性**。把"做什么"和"不要做什么"都讲清楚，并明确不确定时怎么输出（"如果文档里找不到答案，请返回 `{\"answer\": null}`"）。模型对负向约束的遵守比想象中好。
3. **上下文要完整**。如果回答需要某个模型不一定知道的定义，就把它带上。比错答便宜得多。
4. **角色设定按需使用**。"你是一名资深安全审计员"在代码评审任务上确实能收紧输出分布，**但它不是咒语**——通用任务别套，徒增 token。

---

## 2. 零样本、少样本、思维链

这是其他所有技巧赖以构建的三种基线框架。

![三种提示范式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig2_prompting_paradigms.png)

### 零样本

只描述任务，不提供示例。模型完全靠预训练和指令微调期间见过的知识。

```python
zero_shot = """请将以下句子的情感分类为 positive、negative 或 neutral 之一。

句子：这部电影剧情精彩、表演出色。
情感："""
```

零样本适合那些模型本来就熟悉的任务（情感、翻译、短文本摘要），并且你在乎延迟和成本。它的弱点是**输出格式不稳定**——模型可能输出 "Positive sentiment"、"POSITIVE"，甚至是一段分析。用一句严格的格式约束钉死它（"只回复 {positive, negative, neutral} 中的一个单词"）。

### 少样本

少样本就是在查询前面放 $k$ 条示例，是 ICL 的教科书设定。

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

示例其实在做三件事：

- **任务识别**：消除歧义。"翻译"可能指音译、改写或重写，两条示例就能说清你要的是哪种。
- **格式对齐**：每条示例的输出侧就是模板，模型会照抄。
- **标签空间锚定**：示例里出现过的标签集合事实上变成了模型的输出词表，即使你从没显式列举过。

一个反直觉的发现：**示例里"标签是否正确"远没有想象中那么重要**。Min 等人（2022）发现把少样本示例的金标随机打乱，在不少任务上几乎不掉点——真正起作用的是**输入分布和标签空间**。这并不是说"标签不重要"，而是说不要在标签正确性上过度打磨；要在覆盖度和格式上下功夫。

### 思维链（Chain-of-Thought）

对于多步任务，让模型**先把推理过程写出来再给答案**。

```text
问题：一本书有 120 页。第一天读了 30 页，第二天读了第一天的两倍，
第三天读了剩下页数的一半。第三天读了多少页？

让我们一步步来想：
1. 第一天：30 页。
2. 第二天：2 × 30 = 60 页。
3. 累计已读：30 + 60 = 90 页。
4. 剩下：120 − 90 = 30 页。
5. 第三天：30 / 2 = 15 页。

答案：15 页。
```

机制并不神秘。每一个生成出来的 token 都会**改变下一个 token 的条件上下文**。模型是自回归的，把中间状态（"累计已读：90"）显式写出来，后面所有 token——包括最终答案——就能直接用上这个状态。如果不写出来，模型就只能在一次定深网络的前向中算出全部中间量。CoT 实际上是用输出 token 的成本，**买到了串行计算量**。

![思维链推理流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig3_cot_flow.png)

用概率视角看会更干净：对问题 $x$ 和答案 $a$，对潜在推理链 $z$ 做边际化：
$$P(a \mid x) \;=\; \sum_z P(z \mid x)\, P(a \mid x, z).$$
贪心 CoT 就是只采一个 $z$ 然后赌它对。**Self-Consistency**（第 5 节）是用采样去近似这个求和。

CoT 什么时候有用，什么时候反而拖后腿：

- **有用**：算术、多跳问答、代码推理，凡是有中间状态的任务。
- **持平或更糟**：简单分类、检索、单条事实查询。推理铺垫只是徒增 token 和延迟，还多了一次"被自己说服走错"的机会。
- **现代有"思考"模式的推理模型**（在 CoT 风格轨迹上专门训练过的长上下文推理模型）已经把这种行为内化了，显式 CoT 提示仍然有用，但提升幅度比 2022 年那一代 GPT-3 时代要小。

---

## 3. ICL 为什么会工作

往提示里塞几条示例，模型权重又没动，行为为什么变了？目前业界没有单一定论，但有三种互补的解释，合在一起最接近共识。

**1. 隐式任务推断**。预训练 LM 在训练数据里见过大量以文本形式表达的任务（问答对、代码-注释对、原文-译文对）。推理时，提示里的示例相当于做一次**贝叶斯后验更新**：让 $P(\text{task} \mid \text{prompt})$ 收敛到那个最匹配的"任务"。这是 Xie 等人（2022）的视角。

**2. 注意力内的隐式梯度下降**。一系列工作（Akyurek 等、von Oswald 等，2022-2023）证明，attention 层在某些玩具设置下可以实现**对提示中编码的线性回归任务做一步梯度下降**。机制结论只在 toy setting 严格成立，但"注意力在做某种快速适应"这个直觉非常有用。

**3. 模式匹配 + 复制**。最朴素的解释：induction head（Olsson 等，2022）会复制上下文中早些时候出现的模式。少样本提示给了模型一个可以复制的模式。

不管你更喜欢哪种解释，实际推论是相同的：

- **示例数量越多越好，但收益急速衰减**。大部分增益来自前 2-4 条。
- **覆盖度比正确率重要**。能覆盖输入空间的示例比单条很巧妙的示例更有用。
- **打平时近的赢**。提示中最后一条示例对模型影响最大。
- **位置偏置真实存在**。多选题任务上，模型常常系统性地偏向 A 选项或最近一个选项，具体看模型。

---

## 4. 方差问题：必须正面应对的事

下面这件事很少出现在宣传材料里，但它是真的：**仅仅改一改示例顺序、把 `Q:` 换成 `Input:`、给答案加不加引号，准确率就能波动 10-30 个点**。

![提示对格式与顺序的敏感度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig5_prompt_sensitivity.png)

Lu 等人（2022）把这个现象叫做**顺序敏感性**，他们证明在分类任务上，同一个模型、同一组示例，不同排列下的准确率能从接近随机一直跨到接近 SOTA。Sclar 等人（2024）把这个工作扩展到了**格式敏感性**——把 `Q:/A:` 换成 `Question:/Answer:`，准确率能差出两位数。

落到实处：

- **永远在至少 50-100 条留出样本上做评估**。单个轶事什么都说明不了。
- **比较两个提示时跑多个种子或多个示例顺序**。报中位数，不要报最好成绩。
- **格式尽早冻结**，每次改动当成一次有版本号的事件，而不是"顺手调一下"。

一个最小评估骨架：

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
    """跑 n_seeds 次不同的示例采样，返回准确率中位数与波动范围。"""
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

如果 `spread > 0.05`，你的提示就还不稳定，任何一次跑出来的数字都是噪音。

### 示例到底要给多少条

![准确率随示例数 k 的饱和曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig4_shots_saturation.png)

各类任务上的形状基本一致：前几条示例增益巨大，到 $k \approx 4$ 至 $8$ 收益饱和，超过 $k \approx 16$ 基本只是在烧 token。两条经验：

- 类别数 $\le 5$ 的分类任务，$k = 2 \times \text{类别数}$ 是不错的默认值。
- 生成任务上，$2$-$3$ 条高质量示例几乎总是吊打 $10$ 条平庸示例。

挑示例时记住两点：**输入侧覆盖广**、**输出侧干净无歧义**。模型会忠实复制你的格式，包括格式里的 bug。

---

## 5. Self-Consistency：把解码器变成集成

一条 CoT 链可能在第二步就走偏，错误一路传到结尾。Self-Consistency（Wang 等，2022）的解法只有一句话：**对同一个问题采样 $k$ 条不同的推理链，然后对最终答案多数投票**。

![Self-Consistency：多路径采样、多数投票](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig7_self_consistency.png)

形式上，对问题 $x$ 采 $k$ 条链 $z_1, \dots, z_k$，每条链给出答案 $a_i$：
$$\hat{a} \;=\; \arg\max_a \sum_{i=1}^{k} \mathbb{1}[a_i = a].$$

这是对边际 $\sum_z P(z \mid x)\, P(a \mid x, z)$ 的一次蒙特卡洛近似。它之所以有效，是因为**错误倾向于多样、正确倾向于收敛**——错的推理链往往各自走到不同的错答案上，对的推理链则会汇聚到同一个答案。

```python
from collections import Counter

def self_consistency(
    model_call,
    prompt: str,
    k: int = 8,
    temperature: float = 0.7,
    extract_answer=lambda s: s.strip().split()[-1],
) -> dict:
    """采样 k 条 CoT，对最终答案做多数投票。"""
    answers = []
    for _ in range(k):
        out = model_call(prompt, temperature=temperature)
        answers.append(extract_answer(out))
    counts = Counter(answers)
    top, n = counts.most_common(1)[0]
    return {"answer": top, "confidence": n / k, "votes": dict(counts)}
```

两点上线之后才会体会到的细节：

- **温度很关键**。建议 $T \in [0.5, 0.9]$。$T = 0$ 时所有样本会塌成同一条链，集成退化为单点。
- **得票率就是你的置信度信号**。5/5 一致和 3/5 险胜的可信度差很多。把这个数字暴露给下游消费者，这是最便宜的可靠性指标之一。

**Tree of Thoughts**（Yao 等，2023）是它的推广：不再独立采线性链，而是搜索一棵推理树，对低分分支剪枝。能力更强但成本也更高，等 Self-Consistency 真饱和了再上。

---

## 6. ReAct：推理 + 行动

Self-Consistency 提升的是"模型用已有知识能做到什么"。**ReAct**（Yao 等，2022）解决的是更难的问题：当模型需要**外部信息或外部动作**时怎么办？这个模式在输出里穿插三种块：

- **Thought（思考）**：用自然语言对当前状态做推理。
- **Action（行动）**：一次结构化工具调用，例如 `search("..."); calc("..."); read_file("...")`。
- **Observation（观察）**：工具的输出，作为下一轮的上下文喂回去。

![ReAct：Thought / Action / Observation 闭环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/07-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E4%B8%8EIn-Context-Learning/fig6_react_pattern.png)

循环在模型输出 `Answer:` 而不是 `Action:` 时终止。现代 Agent 框架（LangChain Agents、OpenAI 的 function calling、Anthropic 的 tool use）本质上都是这个模板的变体——只不过把"结构化工具调用"从自由文本里搬到了一个 JSON 字段里。

一个抓住要害的最小实现：

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
            "你可以调用以下工具：" + names + "。\n"
            "严格使用如下格式：\n"
            "Thought: <推理>\n"
            "Action: <tool>(<参数>)\n"
            "Observation: <工具结果>\n"
            "...重复...\n"
            "Answer: <最终答案>"
        )

    def run(self, question: str, max_steps: int = 6) -> str:
        ctx = f"{self._system()}\n\nQuestion: {question}\n"
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
        return "(达到最大步数)"
```

生产里真正会咬人的三件事：

- **Stop token**：在 `\nObservation:` 处停止生成，否则模型会自己幻觉出工具结果。
- **工具异常处理**：把工具调用包起来，把**异常信息**当成观察喂回去。模型经常能据此自我纠正。
- **步数预算**：必须给上限。没有预算的 Agent 会循环到把你账户烧穿。

---

## 7. 构建一个提示系统

几个写得好的提示是**脚本**；能撑过团队换人、模型升级、四个月 A/B 实验的，才叫**系统**。区别在三个习惯。

### 把提示当代码对待

加版本号、走代码评审、放进仓库，而不是堆在一份共享文档里。一个最小注册表：

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

每个调用点都打上 `(prompt_name, version)` 标签。改提示时升版本号，老代码继续用老提示直到你迁移它。

### 上线前必须有评估

每个进生产的提示，应该有三件配套：

- 一个 **golden set**：50-200 条输入和期望输出（或评分细则）。
- 一个**自动评估器**：精确匹配、JSON schema 校验，或是 LLM-as-judge（评判用的提示也要钉死版本）。
- 一个 **CI 回归门**：跑当前提示和新提示，准确率掉了就拒绝合入。

```python
def regression_check(old_prompt: str, new_prompt: str,
                     cases, model_call, judge) -> bool:
    old_acc = mean(judge(case, model_call(old_prompt + case.input))
                   for case in cases)
    new_acc = mean(judge(case, model_call(new_prompt + case.input))
                   for case in cases)
    return new_acc >= old_acc - 0.01   # 容忍 1pp 的回归预算
```

### 有意识地组合技巧

最大的增益通常来自堆叠：

- **角色 + 少样本 + 格式说明**：任何结构化输出任务的标准三件套。
- **CoT + Self-Consistency**：任何"答错代价高"的推理任务。
- **ReAct + 检索**：任何依赖模型不知道的事实的任务。

但不要为了堆而堆。每个块都要花 token、加延迟，并多一个出错的地方。

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

---

## 实战：一个能上线的情感分类器

把上面所有东西串起来，做一个不大但完整的小例子。

```python
import json, re
from collections import Counter

SYSTEM = (
    "你是一个严谨的文本分类器。只输出合法的 JSON，不要任何解释。"
)

EXAMPLES = [
    ("这部电影很棒，演技精湛、节奏紧凑。",
     {"sentiment": "positive", "confidence": 0.95}),
    ("电池只能撑三小时，App 还经常崩。",
     {"sentiment": "negative", "confidence": 0.92}),
    ("快递周二送到。",
     {"sentiment": "neutral", "confidence": 0.88}),
]

FORMAT_SPEC = (
    '严格输出：{"sentiment": "<positive|negative|neutral>", '
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
    """对情感分类器叠 Self-Consistency。"""
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

这段代码里能找到本文每一节的影子：一个角色（system）、一段格式说明、三条覆盖广的少样本示例、一个能容忍模型噪音的解析器，再用 Self-Consistency 抑制标签翻转。没什么花活，但把它们组合起来，就是一个 60% 流水线和一个 90% 流水线之间的差距。

---

## 常见问题

**提示是不是越长越好？**
不是。过了某个点，多余的上下文反而分散注意力，并把延迟和成本拉高。先短，再按评估集结果有针对性地加。

**示例给多少条合适？**
分类 2-5 条、生成 1-3 条。超过 8 条很少有用，反而经常因为格式问题引入额外方差。永远要在数据上验证，别凭感觉。

**CoT 在所有任务上都有效吗？**
不。它对多步推理（数学、逻辑、代码、多跳问答）效果突出，对简单分类或事实查询往往只是徒增噪声和 token。

**温度怎么调？**
确定性输出（分类、抽取）：$T \in [0.0, 0.2]$；平衡的生成：$T \in [0.5, 0.7]$；Self-Consistency 采样：$T \in [0.7, 0.9]$——你**就是要**路径多样性。

**提示工程能取代微调吗？**
能取代过去需要微调的相当大一部分场景，特别是 instruction following 类任务。需要微调的几种情形：(a) 大规模下要求行为高度一致，提示漂移不可接受；(b) 想用一个更小更便宜的模型在你的领域里追平大模型；(c) 基础模型抗拒的某种行为改造。

**示例顺序到底有多大影响？**
比应该的大很多。近因偏置是真实的——最后一条示例影响最大。比较两个提示一定要跑多个顺序、报中位数。

**要不要担心 Prompt Injection？**
要。任何时刻，只要你的提示拼进了不受你控制的文本（用户输入、检索回来的文档、工具输出），就要把这些当成数据，**绝不允许它们改写你的指令**。这是个独立话题，后面 Agent 与安全相关的内容会专门讨论。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 5 | BERT 与预训练模型 | [<-- 阅读](/zh/自然语言处理-五-BERT与预训练模型/) |
| 6 | GPT 与生成式语言模型 | [<-- 上一篇](/zh/自然语言处理-六-GPT与生成式语言模型/) |
| **7** | **提示工程与 In-Context Learning（本文）** | |
| 8 | 模型微调与 PEFT | [下一篇 -->](/zh/自然语言处理-八-模型微调与PEFT/) |
| 9 | 大语言模型架构深度解析 | [阅读 -->](/zh/自然语言处理-九-大语言模型架构深度解析/) |
