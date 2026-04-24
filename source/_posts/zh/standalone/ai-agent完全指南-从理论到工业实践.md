---
title: "AI Agent 完全指南：从理论到工业实践"
date: 2024-02-07 09:00:00
tags:
  - LLM
  - AI Agents
  - Applications
categories: Large Language Models
lang: zh-CN
description: "面向工程师的 AI Agent 实战指南：规划（CoT/ReAct/ToT）、记忆体系、工具调用、自我反思、多 Agent 协作、主流框架（LangChain、LangGraph、AutoGen、CrewAI）、评估方法与生产部署的全部坑。"
disableNunjucks: true
---

聊天机器人是用来回答问题的，Agent 是用来**把事情做完**的。同样一个大模型放在背后，前者只会输出文字，后者会去搜索、写代码、调 API、查数据库，并且不断迭代直到任务完成。差别不在模型，差别在外层包了什么——一个能保留状态的循环、一组工具、一个能审视自己输出的批评者。

这篇文章是这一思路的长版本。我们会自底向上把 Agent 的四大核心能力（规划、记忆、工具、反思）讲清楚，把主流框架的差别说明白，再讨论多 Agent 协作、评估方法，以及那些"演示时光鲜、上线后炸裂"的生产细节。

## 你会学到什么

- 为什么 Agent 不止是"GPT-4 套个 `while` 循环"
- 任何严肃的 Agent 都会实现的四大能力
- ReAct、Tree of Thoughts、Reflexion 各自值不值得
- 主流框架的取舍：LangChain、LangGraph、AutoGen、CrewAI、AutoGPT
- 真的能扩展的多 Agent 拓扑（以及看着炫但跑不动的那种）
- 一套能在 CI 里跑、能抓住真实回归的评估体系
- 上线 checklist：成本、安全、可观测性、常见失败模式

## 阅读前置

- 熟悉 Python，习惯 `requests`/`json` 那种风格的接口代码
- 至少用过一家大模型的 API（OpenAI、Anthropic、通义、智谱都行）
- 对 prompt、completion、上下文窗口有清晰的概念

---

## Agent 到底是什么

一个 AI Agent 是这样的系统：给定一个目标，由 LLM 自主决定下一步要做什么，通过工具执行该动作，观察结果，然后重复，直到目标达成或触发停止条件。

这句话里有四个关键词：**目标、决定、执行、观察**。普通的一次 LLM 调用一个都没有，它只是生成文本然后停下。Agent 在每一步都把新的上下文重新喂给模型，所以底层同一个 LLM 才能突然学会订机票、修 flaky test、操控浏览器。

### 从一次性生成到认知循环

从"请求-响应"切换到"循环"，是 Agent 的全部秘密：

```python
# 传统 LLM：一发，一中，结束
def traditional_llm(prompt: str) -> str:
    return llm.generate(prompt)


# Agent：带状态、带工具、带停止条件的循环
def agent_loop(goal: str) -> str:
    state = init_state(goal)
    for step in range(MAX_STEPS):
        observation = state.latest_observation()
        thought = llm.reason(observation, state.memory)
        action = llm.decide_action(thought, available_tools)
        result = execute(action)
        state.append(thought=thought, action=action, result=result)
        if state.goal_satisfied():
            return state.answer()
    return state.best_effort_answer()
```

让这个循环在生产里跑得通，靠三件事：跨步骤存活的记忆、能返回结构化结果的工具层、防止"热情过度的 LLM 一口气下了 200 份外卖"的停止规则。本文剩下的内容，全部是这三个主题的变奏。

![Agent 认知循环：感知、推理、行动、记忆](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig1_agent_loop.png)

### 为什么这件事重要：一个具体对比

设想任务："找出 2026 年三大机器学习顶会的 deadline，并且在每个 deadline 前两周提醒我。"

普通 LLM 单次调用，会返回训练时残留的（很可能过期的）记忆，并且**物理上不可能**帮你建日历事件。Agent 则会去搜当前的 deadline，解析结果，算出提醒日期，调日历 API，再把确认 ID 返回给你。底层 LLM 是同一个，**外层布线**才是差别所在。

![规则系统 vs LLM Agent](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig2_agent_vs_rule.png)

值得在上面这张图上停一下。规则系统是一棵硬编码的决策树：每加一个新意图，就多一根分支。LLM Agent 是一个通才内核加一组工具——**新行为通过新增工具描述加进来，而不是改控制流**。这个差别在维护成本上是数量级的。

### Agent 的五个组件

任何能用的 Agent 都有这同样五块。各家框架命名不同，职责完全一致。

**1. 大脑（LLM 内核）**：解释目标、生成计划、发出工具调用。模型选型基本决定了上限和成本。

**2. 规划器**：把目标拆成子目标。可以是隐式的（LLM 在 ReAct 里边想边规划），也可以是显式的（一个独立的 planner 输出任务 DAG）。

**3. 记忆**：工作记忆放当前轨迹；长期记忆跨会话保存事实、教训、向量。

**4. 工具**：Agent 能调用的一切——搜索、代码执行、SQL、HTTP API、文件 I/O，甚至其他 Agent。**给 LLM 看的 schema 比工具实现本身更影响效果。**

**5. 反思**：自我批评，能发现错误并触发带纠正信息的重试。

把第 3、4 块写成最小可用版本，几行就够：

```python
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., Any]

    def __call__(self, **kwargs) -> dict:
        try:
            return {"ok": True, "result": self.fn(**kwargs)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


@dataclass
class WorkingMemory:
    messages: list = field(default_factory=list)
    budget_tokens: int = 4000

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._evict()

    def _evict(self) -> None:
        # 粗估 4 字符 ≈ 1 token
        while sum(len(m["content"]) for m in self.messages) // 4 > self.budget_tokens \
                and len(self.messages) > 2:
            self.messages.pop(0)
```

生产系统在此之上加结构化日志、重试、可观测性，但内核就是这点东西。

---

## 能力一：规划

规划是把"帮我把这份 200 页报告总结一下并把要点发给团队"翻译成一串可执行步骤的能力。主流的规划方式有三种。

### Chain-of-Thought（CoT）

最简单的方式：在 prompt 里要求模型"把推理过程写出来"再给答案。CoT 在推理主要发生在模型内部时效果好——算术、逻辑题、结构化重写。它**本身不带工具调用**，只是把推理摊到更多 token 上。

CoT 是一个性价比很高的默认选项。先用它跑出一个基线，再考虑要不要升级。

### ReAct（Reason + Act）

ReAct 把"思考（Thought）"、"动作（Action）"、"观察（Observation）"三种步骤交错在一起。它是带工具 Agent 的事实标准，因为同一次 LLM 调用既决定怎么想，也决定怎么做。

```python
class ReActAgent:
    def __init__(self, llm, tools, max_steps=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        history = []
        for step in range(self.max_steps):
            prompt = self._build_prompt(task, history)
            response = self.llm.complete(prompt)
            parsed = self._parse(response)

            if parsed["type"] == "final_answer":
                return parsed["content"]

            tool = self.tools.get(parsed["tool"])
            if tool is None:
                obs = f"Unknown tool: {parsed['tool']}"
            else:
                result = tool(**parsed["args"])
                obs = result["result"] if result["ok"] else f"Error: {result['error']}"

            history.append({
                "thought": parsed.get("thought", ""),
                "action": f"{parsed['tool']}({parsed['args']})",
                "observation": obs,
            })
        return "Step budget exhausted"
```

一段典型的 ReAct trace 大致长这样，每一行对应一次 LLM 调用：

![ReAct trace：交错的推理与行动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig5_react_flow.png)

两个工程上的提醒。第一，**parser 是生产事故的最大来源**，能用 JSON mode 或 function calling 就别自己解析自由文本。第二，工具返回的大块内容**先压缩再喂回去**——一个原始 HTML 页面三步就能把上下文窗口撑爆。

### Tree of Thoughts（ToT）

ToT 把 CoT 推广成搜索树：每一步生成多个候选思考，打分，选最有希望的展开，必要时回溯。在带组合结构的问题上（24 点、创意写作、代码搜索）确实更强，但代价是 LLM 调用次数乘以分支因子 *N*。

经验法则：只有当 ReAct 反复失败、且你能说清楚为什么失败时再上 ToT。绝大多数生产任务，"ReAct + 重试"在性价比上完胜 ToT。

### 什么时候用哪种

| 模式 | 适用场景 | 成本 | 失败模式 |
|---|---|---|---|
| **CoT** | 闭式推理、不需要外部工具 | 1x | 需要事实时会幻觉 |
| **ReAct** | 带工具的任务：网页、代码、数据 | 3-15x | 死循环、parser 出错 |
| **ToT** | 搜索类问题、设计探索 | 10-50x | 慢、贵、难调 |

---

## 能力二：记忆

有没有记忆，是"帮你一次"和"和你共事几个月"的区别。值得分清的有四种记忆。

| 记忆类型 | 生命周期 | 典型存储 | 用途 |
|---|---|---|---|
| **工作记忆** | 单次任务内 | 上下文 buffer | 当前轨迹 |
| **实体记忆** | 按用户/项目 | KV 存储 + LLM 抽取 | 例如"这个用户偏好 Postgres" |
| **语义记忆** | 长期 | 向量库（FAISS、pgvector、Chroma） | 历史经验、文档 |
| **情景记忆** | 长期 | 结构化日志 + embedding | 历史 run 回放，用于学习 |

### 工作记忆：最不被重视、最先崩的那块

工作记忆其实就是你每一轮发给 LLM 的 messages 列表。它是大多数 Agent 里**工程化最薄弱、上量后最先出事**的部分。三件事很有用：

1. **旧轮次要总结，不要直接丢**。一段不断滚动更新的"目前已经确立的事实"摘要能保住连续性。
2. **scratchpad 和最终输出要分开**。模型的中间想法**它自己**应该看得到，但用户拿到的回答里要剥掉。
3. **按 token 预算淘汰，不要按消息条数淘汰**。一条 10K token 的工具输出能直接撑爆预算，而你还在勤勤恳恳地裁三行用户消息。

### 用向量检索做长期记忆

标准做法：把每段交互 embed，存"向量 + 原文 + 元数据"，新任务开始时按余弦相似度取 top-k。生产里用 Pinecone、Weaviate、Milvus 或 pgvector；本地原型，进程内 FAISS 完全够用。

```python
class VectorMemory:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store  # 任意带 add/query 的向量后端

    def remember(self, text: str, metadata: dict) -> None:
        vec = self.embedder.embed(text)
        self.store.add(vec, text=text, **metadata)

    def recall(self, query: str, k: int = 5) -> list[str]:
        vec = self.embedder.embed(query)
        hits = self.store.query(vec, k=k)
        return [h["text"] for h in hits]
```

要小心两种翻车方式。一种是**检索漂移**：检索到的内容主题相关但上下文不对，比如你想要本周的工单，它给你上个月的 bug report。另一种更隐蔽，叫**记忆中毒**：早期一条幻觉出来的"事实"被存进库，后续每一轮都把它检索出来用。前者用元数据过滤和时间衰减缓解；后者要在写入时带显式置信度，并定期审计记忆库。

### 主动遗忘

记忆不能无限增长。一个简单可用的保留分数：

```
score = 0.5 * importance + 0.3 * normalised_access_count - 0.2 * days_since_last_access
```

定期把分数最低的 10% 淘汰掉。**保留淘汰日志**——以后某天 Agent 行为开始变怪，你会想回头看看是不是那时候删错了什么。

---

## 能力三：工具调用

工具是 Agent 走出 LLM 的"幻觉世界"、真正去操作真实世界的接口。各家厂商的 pipeline 形状基本一致：声明工具 schema，让模型发出结构化调用，校验，执行，把结果喂回去。

![Function Calling 流程：schema、模型、执行、结果](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig3_tool_calling_pipeline.png)

### OpenAI 风格的 function calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]}
            },
            "required": ["city"]
        }
    }
}]


def run_turn(messages):
    resp = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = TOOL_REGISTRY[call.function.name](**args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
    # 第二轮：让模型把工具结果总结成自然语言
    return client.chat.completions.create(
        model="gpt-4o", messages=messages
    ).choices[0].message.content
```

Anthropic 的 `tool_use` 接口消息形状不同，语义一样。LangChain、LlamaIndex 和各家 Agent SDK 内部封的也都是其中之一。

### 工具设计：真正决定上限的环节

绝大多数所谓的"Agent 失败"，本质都是**工具设计失败**。三条铁律：

**做小、做幂等。** 一个 `update_user(...)` 带十二个可选字段，几乎是邀请模型把错的那个填上。拆成 `set_user_email`、`set_user_address` 这种小工具，每一个都更好描述、更好测、更好校验。

**返回结构化错误，不要抛异常。** `get_weather("Atlntis")` 调失败时，要返回 `{"error": "city_not_found", "suggestion": "Atlanta"}`，不要 500。模型能基于 suggestion 改下一步动作；它没法基于一段堆栈做任何事。

**描述里带例子。** "搜索 GitHub issues。例：`search_issues(repo='langchain-ai/langchain', label='bug', state='open')`"。两行例子能把边界查询的工具调用准确率翻倍。

### Computer Use 与下一代

Anthropic 在 2024 年 10 月放出的 Computer Use（以及类似系统）给 Agent 提供了截图和 `mouse_click(x, y)`、`type_text("...")` 这类原语。这一下解锁了所有有 UI 的应用，包括那些**根本没有 API 的**。代价是可靠性——单次点击 95% 准确率，10 步的任务只剩 60% 成功率。把 Computer Use 类 Agent 当研究级品，部署时要严格沙箱。

---

## 能力四：反思

反思是 Agent 内置的"批评家"。没有它，Agent 重试时会重复同样的错；有了它，Agent 在**单次 run 内**就能学习。

### Self-Refine

最小模板：生成、批评、再生成。

```python
def self_refine(llm, task: str, max_iters: int = 3) -> str:
    output = llm.complete(f"Task: {task}\nAnswer:")
    for _ in range(max_iters):
        critique = llm.complete(
            f"Task: {task}\nAnswer: {output}\n"
            "List concrete problems. End with SCORE: 0-10."
        )
        score = parse_score(critique)
        if score >= 8:
            return output
        output = llm.complete(
            f"Task: {task}\nPrevious answer: {output}\n"
            f"Critique: {critique}\nImproved answer:"
        )
    return output
```

Self-Refine 在文本生成（写作、摘要、code review）类任务上效果最好。带工具的任务上效果有限，因为批评家很难分清"动作错了"和"动作对了，但工具返回的数据本来就脏"。

### Reflexion：跨次尝试积累教训

Reflexion（Shinn 等，2023）在每次失败之后显式产出一条**口头化的"教训"**，并喂给下一次尝试。教训要短、具体、可操作——例如"SQL 查到 0 行时，先去掉 date 过滤器，再考虑换表"。几次重试之后，Reflexion Agent 经常能解掉同模型单发解不动的题。

Reflexion 最有效的场景：

- 有清晰的成败信号（测试通过、答案对得上）；
- 单次尝试相对便宜；
- 任务族足够一致，让教训能迁移。

### 陷阱

反思不是免费的。两个具体的失败模式：

- **谄媚式批评**：让生成答案的模型给自己打分，它通常会说"答案没问题"。要么换一个模型当批评家，要么换一个明显不同的人格 prompt。
- **反思死循环**：批评-改写循环可能收敛到一个**一致但错误**的不动点。设上限迭代次数，置信度不上升就回退到原始答案。

---

## 框架：选一个能长期居住的地方

框架生态变化很快，但每个工具在地图上的**位置**比较稳定。下图沿两条轴铺开：可组合度（横向构建复杂流程的能力）和典型 Agent 的自主度。

![Agent 框架与平台的定位图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig6_platform_landscape.png)

简单概括各家适用场景：

| 框架 | 是什么 | 什么时候选它 |
|---|---|---|
| **LangChain** | 组件库 | 你想要积木，不想要主见 |
| **LangGraph** | 基于 LangChain 的 DAG runtime | 你需要显式控制流、分支、循环 |
| **LlamaIndex** | RAG 优先的 Agent SDK | 工作主要是从文档里检索 |
| **AutoGen**（Microsoft） | 多 Agent 对话框架 | 你想让 Agent 之间互相说话 |
| **CrewAI** | 基于角色的多 Agent 编排 | 想要"专家小团队"语义开箱即用 |
| **Semantic Kernel** | 微软的 planner + skills SDK | 你住在 .NET / Azure 生态 |
| **OpenAI Assistants** | 托管 Agent runtime | 你愿意把状态和工具交给 OpenAI |
| **AWS Bedrock Agents** | AWS 原生托管 Agent | 你已经在 Bedrock 上 |
| **Dify / Flowise** | 无代码 Agent 搭建 | 让非工程师搭 flow |
| **AutoGPT / BabyAGI** | 自主任务循环 | 研究、探索、做 demo |

下面给一段 LangGraph 的小例子，因为它比大多数同行更能体现"现代 Agent"的形状：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator


class AgentState(TypedDict):
    messages: Annotated[Sequence[Any], operator.add]


def call_model(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def call_tools(state: AgentState) -> dict:
    last = state["messages"][-1]
    outputs = [TOOLS[c.name].invoke(c.args) for c in last.tool_calls]
    return {"messages": [ToolMessage(content=o, tool_call_id=c.id)
                          for o, c in zip(outputs, last.tool_calls)]}


def should_continue(state: AgentState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("model")
graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "model")
app = graph.compile()
```

图结构把分支和循环显式化，正好是 Agent 在生产里出问题、需要逐步排查时你最想要的形态。

### 关于"自主度光谱"的一句题外话

AutoGPT 那种追求开放目标的"全自主 Agent"，**看起来很神奇，能上线的极少**。同一个任务用 LangGraph 三个节点两条边写出来，在推上不那么好看，但在生产上要可观得多。今天大多数所谓"Agent 应用"，本质上更接近上面 LangGraph 的形态，而不是自主跑飞的 AutoGPT。

---

## 多 Agent 系统

单 Agent 在面对**长、异构**的任务时会撞上天花板：一个 prompt 装不下所有职责，轨迹长度突破上下文窗口，反思也开始失焦。多 Agent 系统的核心思路是**按专长拆**。

![三种多 Agent 协作模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig4_multi_agent_patterns.png)

### 模式一：层级（Manager + Workers）

一个 Manager Agent 拆解任务，把子任务派发给具体专家（Researcher、Coder、Analyst、Writer）。每个专家拥有自己的工具集，返回结构化结果。任务能干净拆开时效果好。要小心 **Manager 变成瓶颈**——如果每个结果都要回流到 Manager 才能决定下一步，你只是把"长上下文"换成了"长协调上下文"。

### 模式二：辩论（对抗式批评）

两个或更多 Agent 持不同立场互相辩论，再由一个评委 Agent（或投票）选出最终方案。在那种"质量难以直接验证"的任务上特别有用——法律论证、设计选型、复杂推理。在 GSM-Hard 这类基准上，3-5 轮辩论能稳定打过单 Agent 推理，代价是 token 用量翻几番。

### 模式三：流水线（专家串联）

MetaGPT 模式：PM → 架构师 → 工程师 → QA，每个 Agent 的输出是下一个的输入。**这是最"无聊"也最能上线**的模式。它好用是因为每次交接都产出一个**有名字的 artifact**（PRD、设计文档、代码、测试报告），天然就有 checkpoint 和回退点。

### 通信

多 Agent 系统的成败取决于**消息总线**。最小可用协议长这样：

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str        # "*" 表示广播
    type: str             # "request" | "response" | "broadcast"
    content: dict
    correlation_id: str   # 把响应和请求关联起来
    timestamp: datetime
```

**永远把每条消息记下来**。多 Agent 系统出问题时，故障几乎总是在"A 以为 B 说了什么"上——没消息日志，你只能靠猜。

---

## 评估

度量不到的东西，改进不了。Agent 评估比 LLM 评估更难：答案空间大，部分得分有意义。两种策略要一起用。

![评估框架：能力雷达 + 基准覆盖矩阵](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/ai-agent%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97-%E4%BB%8E%E7%90%86%E8%AE%BA%E5%88%B0%E5%B7%A5%E4%B8%9A%E5%AE%9E%E8%B7%B5/fig7_evaluation_framework.png)

### 公开基准

| 基准 | 测什么 | 备注 |
|---|---|---|
| **AgentBench** | 8 个环境（OS、DB、KG、Web…） | 覆盖广，GPT-4 领先 |
| **GAIA** | 真实助手任务，3 个难度等级 | 难。GPT-4 + 工具在 Level 2 之后也吃力 |
| **AgentBoard** | 细粒度能力打分 | 适合诊断，不只是排名 |
| **WebArena** | 真实网站里的 Web 导航 | 纯浏览器使用评估 |
| **SWE-bench** | 真实 GitHub issue 修复 | 代码 Agent 的硬指标 |
| **ToolBench** | 16k+ API 上的工具选择 | 测工具使用的广度 |

公开基准用来跟踪**通用能力**的版本演进。**不要**拿它们当你的验收测试——你的任务分布不是它们的。

### 内部 eval 套件

真正能抓住回归的，是你为自己的任务分布写的 eval。一个能跑得起来的形状：

```python
@dataclass
class EvalCase:
    id: str
    input: str
    rubric: dict          # 评分维度：正确性、格式等
    must_use_tool: list   # 必须调用的工具
    must_not_call: list   # 禁止的动作（例如 dry-run 时禁止 send_email）
    timeout_s: int
    expected_cost_max: float


def evaluate(agent, cases: list[EvalCase]) -> dict:
    results = []
    for case in cases:
        with timeout(case.timeout_s):
            trace = agent.run(case.input)
        score = score_with_rubric(trace.output, case.rubric)
        results.append({
            "id": case.id,
            "score": score,
            "tools_used": trace.tools,
            "cost": trace.cost,
            "violated_constraints": check_constraints(trace, case),
        })
    return aggregate(results)
```

三件事必须坚持：

1. **检查整条轨迹**，不只是最终输出。"Agent 有没有发它不该发的邮件"和"答案对不对"同样重要，甚至更重要。
2. **成本是一等公民**。0.04 美元做对的答案和 4.00 美元做对的答案不是同一回事。
3. **可复现**。固定模型版本、能固定的 seed、工具 mock、时间戳。否则昨天的回归测试今天碰巧通过，是糟糕的好消息。

---

## 生产细节

从能跑的 notebook 到能上生产的 Agent，距离通常比"零基础到能跑的 notebook"还要远。下面这份是短版的"会咬人"清单。

### 成本控制

LLM 调用塞进循环里，账单很快就长成需要开会才能说清楚的样子。三条实践：

- **分级路由**：分类、简单轮次用便宜模型，硬骨头才走贵的。
- **每个任务设硬预算**：超 `max_cost_usd` 就拒绝或降级。
- **激进缓存**：相同的工具调用、相同的子 prompt 应该命中缓存而不是 API。

```python
class BudgetedAgent:
    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent = 0.0

    def call(self, prompt, model="gpt-4o"):
        est = estimate_cost(prompt, model)
        if self.spent + est > self.budget:
            model = "gpt-4o-mini"  # 降级
        resp = client.complete(model=model, prompt=prompt)
        self.spent += actual_cost(resp)
        return resp
```

### 安全与沙箱

只要你的 Agent 能执行代码或 shell 命令，**它早晚会执行错的那一条**。按优先级的防御：

1. **沙箱里跑**：容器、无网络、无凭据、内存/CPU 限额、超时。
2. **工具和参数都用白名单**：永远不要让一个 LLM 输出的字符串不经过滤就变成 shell 命令。
3. **破坏性动作要审批关卡**：发邮件、写生产库、扣款——这些要走人工或严格策略，不能只看模型置信度。
4. **每一次工具调用都审计**：含输入、输出、以及产生该动作的那次 LLM 调用。

### 可观测性

没有遥测，你不可能搞懂 Agent 究竟干了什么。要采集的：

- 每一步的延迟、in/out token、成本、工具名。
- 完整轨迹（thought / action / observation）。
- 结果（成功 / 失败 / 超时 / 超预算）和原因。
- 稳定的 `task_id` 和 `parent_task_id`（多 Agent 必备）。

LangSmith、Langfuse、Helicone、Arize Phoenix 这一类工具能给你 Agent 版的 APM。**用其中一个**，自己从零搭至少够你做半年副业。

### 常见失败模式

| 现象 | 大概原因 | 第一刀 |
|---|---|---|
| 死循环 | 没停止条件 / 工具调用之间循环 | 硬步数上限 + 循环检测 |
| 工具参数总是错 | schema 烂或描述模糊 | 加例子、收紧 enum、拆工具 |
| 上下文爆掉 | 工具返回大块未压缩 | 工具输出超 N token 就摘要 |
| Agent 不听指令 | system prompt 又长又自相矛盾 | 缩短、去重、把细节挪进工具 |
| 自信但错 | 缺 grounding | 强制先做一次搜索或查库再回答 |
| 0.10 美元能跑、10 美元就崩 | 没设预算上限 | 上成本天花板 + 降级回退 |

---

## 一个完整的例子：数据分析 Agent

把上面所有模式压到一个**最小但真实**的 Agent 里：加载 CSV、清洗、做探索性分析、出图、生成报告。

```python
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 共享状态——生产环境里换成真正的存储
class State:
    raw: pd.DataFrame | None = None
    clean: pd.DataFrame | None = None
    charts: list[str] = []


@tool
def load_csv(path: str) -> str:
    """Load a CSV file into the analysis state."""
    State.raw = pd.read_csv(path)
    return f"Loaded {State.raw.shape[0]} rows x {State.raw.shape[1]} cols."


@tool
def clean() -> str:
    """Drop duplicates and impute missing values."""
    df = State.raw.drop_duplicates()
    num = df.select_dtypes("number").columns
    cat = df.select_dtypes("object").columns
    df[num] = df[num].fillna(df[num].mean())
    df[cat] = df[cat].fillna(df[cat].mode().iloc[0])
    State.clean = df
    return f"Cleaned. {len(df)} rows remain."


@tool
def describe() -> str:
    """Return summary statistics."""
    return State.clean.describe().to_string()


@tool
def plot(kind: str, x: str, y: str | None = None) -> str:
    """Create a plot. kind in {'hist','scatter','bar','box'}."""
    fig, ax = plt.subplots()
    df = State.clean
    if kind == "hist":
        ax.hist(df[x], bins=30)
    elif kind == "scatter" and y:
        ax.scatter(df[x], df[y])
    elif kind == "bar":
        df[x].value_counts().plot(kind="bar", ax=ax)
    elif kind == "box":
        df.boxplot(column=x, ax=ax)
    name = f"chart_{kind}_{x}.png"
    fig.savefig(name)
    plt.close(fig)
    State.charts.append(name)
    return f"Saved {name}"


tools = [load_csv, clean, describe, plot]
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst. Always: (1) load, (2) clean, "
     "(3) describe, (4) plot 2-3 informative charts, (5) summarise "
     "findings in 5 bullets. Never skip a step."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_functions_agent(
    ChatOpenAI(model="gpt-4o", temperature=0), tools, prompt
)
executor = AgentExecutor(agent=agent, tools=tools,
                          max_iterations=10, verbose=True)

executor.invoke({"input": "Analyse sales_data.csv"})
```

这个例子除了表面流程之外，还能体会三件事：

- **用单例 State 共享数据**，让 LLM 的活儿尽量小。模型决定"做什么"，**数据流在带外走**。
- **system prompt 把工作流写死**。Agent 可以在每一步内部选战术，但**不能跳过步骤**。
- **每个工具一句话描述 + 例子**。"Agent 质量"主要就长在这。

要把它推上生产，再叠加：成本天花板、每次工具调用的结构化日志、20-50 个有代表性 CSV 的 eval 套件、一个画图前必须先调的 schema 校验工具。这些事单看都不有趣，加起来都必要。

---

## 常见问题

**Agent 和聊天机器人的区别？**
聊天机器人是反应式的，Agent 是决策式的。同一个 LLM，外面套不套带工具的循环，决定它属于哪一种。

**CoT、ReAct、ToT 怎么选？**
默认 CoT；需要工具时升 ReAct；只有 ReAct 反复跑不通、且你**愿意付 10 倍成本**时才上 ToT。

**单 Agent 还是多 Agent？**
单 Agent 一直用，直到 prompt 维护不动、或者轨迹超出上下文预算。然后**沿着 artifact 边界拆**（PRD → 设计 → 代码 → 测试），不要沿着拍脑袋的角色边界拆。

**生产里怎么压制幻觉？**
所有事实都通过工具调用 grounding；非平凡论断要求出处；保留"我不知道"的退出路径并奖励它；离线采样标注幻觉——**度量不了的东西修不了**。

**最常见的失败模式？**
死循环、工具参数错、未压缩工具输出导致上下文爆、长任务的目标漂移、悄无声息的成本爆炸。每一个都有一行 fix，每一个都要等线上炸过一次才有人写。

**用托管平台还是自己搭？**
原型和低流量内部工具，用托管（OpenAI Assistants、Bedrock Agents 等）；需要控制路由、重试、可观测性、成本时，搭在 LangGraph、AutoGen 这种框架上。**不要从零自己造一个 Agent 框架**——那些"无聊的部分"深得出乎意料。

---

## 结语

关于 Agent 最重要的一件事是：**它是基础设施，不是魔法**。LLM 只是一块组件，让 Agent 真正有用的是外面那圈循环——拆解工作的规划器、跨步存活的记忆、把模型钉在现实里的工具层、在错误上线前抓出来的批评者。

值得记住的六条：

- **从简单开始**。三个好工具的 ReAct Agent，能打十二个平庸工具的多 Agent 系统。
- **工具是通往现实的接口**。工具设计的质量基本决定了 Agent 的质量。
- **记忆会腐烂**。提前规划淘汰、审计，以及"哪天不得不删一条中毒事实"的预案。
- **为失败做计划**。步数上限、成本上限、沙箱、重试、回退——全部要，并且**上线前**就要。
- **先度量再调优**。能在 CI 里跑的 eval 套件，是"持续改进"和"持续漂移"的分水岭。
- **上无聊的版本**。一个节点画得清清楚楚的 LangGraph，比任何 AutoGPT 风格的"全自主"演示活得都久。

**延伸阅读**

- [LangChain 文档](https://docs.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://www.crewai.com/)
- [AgentBench 论文](https://arxiv.org/abs/2308.03688)
- [GAIA 基准](https://arxiv.org/abs/2311.12983)
- [Reflexion 论文](https://arxiv.org/abs/2303.11366)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [OpenAI function calling 指南](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic tool use 指南](https://docs.anthropic.com/claude/docs/tool-use)
