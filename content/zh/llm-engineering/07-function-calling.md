---
title: "大模型工程（七）：Function Calling 实战"
date: 2026-04-02 09:00:00
tags:
  - LLM
  - function-calling
  - Tools
  - Agents
  - json-schema
categories: 大模型工程
series: llm-engineering
series_order: 7
series_title: "大模型工程"
lang: zh
mathjax: false
disableNunjucks: true
description: "JSON 模式 vs function 模式 vs 自由格式、并行工具调用、用文法保证结构化输出、错误恢复模式，以及在真实负载里活下来的 agent loop。"
translationKey: "llm-engineering-7"
---
函数调用是大语言模型（LLM）连接外部世界的关健接口，是 chat template、结构化输出内核与提示工程的交汇点。本章深入剖析底层机制，探讨哪些行为具备可依赖的确定性保证，以及哪些 agent-loop 模式能在真实生产负载下稳定运行。

技术渊源至关重要：LLM 的工具调用能力最早可追溯至 2022 年两篇几乎同期发表的论文——**MRKL Systems**（Karpas 等，AI21）提出神经符号模块间的专家路由机制；**ReAct**（[Yao 等，2022][yao-react]）则将思维链（Chain-of-Thought）推理与工具调用动作交替执行。**Toolformer** ([Schick et al., 2023][schick-toolformer]) 展示了工具使用的自监督教学，让模型在现有文本中插入工具调用标记来生成训练数据。到了 2024 年，所有前沿模型都围绕工具使用格式构建了后训练数据，工具调用也从“研究演示”变成了 API 功能。

![LLM Engineering (7): Function Calling and Tool Use — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/illustration_1.png)

## 函数调用到底是什么意思

API 提供“函数调用”能力时，其底层实现可能有以下几种形式：

1. **训练行为 + chat template 标记**。模型经后训练后，可在适当时机生成工具调用，并以特殊 token 进行包裹。例如： Qwen3 使用 `tool_call` 标签， Mistral 使用 `[TOOL_CALLS]`。
2. **JSON-mode 约束解码**。通过语法约束解码（grammar-constrained decoding）强制模型输出合法 JSON。模型本身未必针对该任务进行过专门训练，实际约束由解码器在生成阶段施加。
3. **Schema 引导的结构化输出**。在 JSON 模式基础上，进一步要求输出严格匹配预定义的 JSON Schema，包括函数名、参数类型等。
4. **自由形式 prompt**。在 system prompt 中声明‘请以 JSON 格式回复，包含 X 和 Y 字段’，依赖模型自觉遵守。这对能力强的模型依然有效，但没有任何保证。

实际生产系统中，这四类实现常混合采用：OpenAI 与 Anthropic 的 API 结合方案 1（后训练工具调用）与方案 3（JSON Schema 强制校验）；vLLM 和 SGLang 支持任意模型的方案 2（语法约束解码）与方案 3；自由形式（方案 4）则作为兜底策略。

一个关键却常被忽视的区别是工具调用采用 JSON 还是 XML 格式——OpenAI 默认 JSON；Anthropic 的 Claude 模型内部训练使用类 XML 的结构化输出，但对外 API 统一转为 JSON。Anthropic 团队于 2024 年公开指出，XML 标签更易于模型学习——其尖括号结构与预训练数据中标识特殊区域的模式高度一致，且流式解析部分 XML 内容比解析部分 JSON 更简单。实证表明，两种格式均能有效支持工具调用，具体选择主要影响下游解析工具链的设计。在模型内部，两者看起来都像是一串带有学习语法的 token 序列。

## 一个真实的函数调用请求

![fig1: tool-call request/response sequence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/fig1_tool_sequence.png)

Anthropic Claude API 示例：

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-4-5-sonnet-20250901",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    }],
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)

# response.content might be:
# [TextBlock(text="I'll check the weather in Tokyo."),
#  ToolUseBlock(id="toolu_xxx", name="get_weather",
#               input={"location": "Tokyo, Japan", "unit": "celsius"})]
```

模型返回一个结构化的 `ToolUseBlock`。你的代码执行工具，然后发送跟进消息：

```python
response2 = client.messages.create(
    model="claude-4-5-sonnet-20250901",
    max_tokens=1024,
    tools=[...],  # same tools
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_xxx",
            "content": "72°F, partly cloudy",
        }]},
    ],
)
```

现在对话里有了一个工具调用和一个工具结果；下一轮 assistant 回复就可以利用这些信息。这就是基本的循环逻辑。

## 工具定义的最佳实践

工具定义本质上是一种隐式的 prompt。模型在每次推理时都会将工具定义视为 system prompt 的一部分进行理解，其定义质量直接决定了模型能否准确选择工具、正确发起调用，以及在出错时有效恢复。以下实践已被验证能稳定生效：

**描述是 prompt 中被模型读得最多的文本。** 优质描述需涵盖三要素：(a) 一句话概括工具功能；(b) 明确适用与典型不适用场景；(c) 清晰说明返回结果。低质量描述往往仅机械复述函数名称。“搜索数据库”很糟糕；“搜索客户数据库以匹配给定条件的订单。当用户询问特定订单或订单历史时使用。返回最多 10 条最新匹配项。”这才是好的。

**参数描述比参数名更重要。** 模型能从 `location` 这个名字推断出它想要一个地点，但如果没有描述，它不知道格式应该是 "Tokyo"、"Tokyo, Japan" 还是 "JP/Tokyo"。务必包含格式示例。

**尽可能使用 enums。** 带有 `enum: ["celsius", "fahrenheit"]` 的 `unit` 参数比自由字符串 `unit` 难 misuse 得多。 Schema 约束可阻止模型生成非法值，例如 "Kelvin" 或 "C°"。

**对于复杂工具，应在描述中直接提供调用示例。** "示例：`transfer_money({from_account: 'A123', to_account: 'B456', amount: 100, currency: 'USD'})`" 比三段散文更有用。

**明确记录可能的错误响应格式。** “若账户不存在，则返回 HTTP 404 状态码；若调用者权限不足，则返回 HTTP 403 状态码。”这让模型能正确解释错误响应，并决定是重试还是升级处理。

**避免过度细化工具定义。** 实践中曾观察到，工具注册表包含多达 80 个工具时，模型选错工具的概率显著上升。将相关工具分组为较少数量的多态工具（例如，一个带有可选过滤器的 `query_orders`，而不是 `query_orders_by_id`、`query_orders_by_date`、`query_orders_by_customer` 等）。模型更擅长填充参数，而不是从长菜单中挑选。

## 并行工具调用

![fig3: parallel vs sequential tool execution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/fig3_parallel_vs_sequential.png)

到了 2026 年，所有前沿模型都支持并行工具调用——在一轮中 emit 多个 tool-use 块。如果用户问“东京和纽约的天气怎么样？”，模型会同时 emit 两个工具调用，你并行执行两者，然后把结果都传回去。

为什么这很重要：串行工具调用会累积延迟。一个 5 工具 agent 做 5 次串行调用，每次 200 ms，总延迟达到 1 秒。并行调用可以将其降至 200 ms。对于 agents（OpenClaw 第 7-12 章，任何涉及多个数据源的场景），这是“响应敏捷”和“令人沮丧”之间的区别。

要注意：并行工具调用只适用于彼此不依赖的工具。查询两个城市的天气是并行安全的。“搜索航班，然后预订最便宜的”就不行——第二个工具调用依赖第一个的结果。模型应该通过训练知道这一点，但并不总是如此。生产代码应该在并行运行之前验证并行调用确实是独立的。

依赖分析可能很微妙。如果两个工具都写入同一个外部资源（例如，对同一行的两个 `update_database` 调用），即使它们都不依赖对方的 *返回值*，并行运行也会引入 race conditions。更安全的模式是为工具声明副作用类别（只读、写入隔离、写入共享），仅在兼容类别内并行化。正因为这个原因，截至 2025 年底， Anthropic 的 Claude  emit 并行调用比 GPT 类模型更保守——当不确定依赖关系时，它倾向于串行。

## 基于语法的结构化输出

![fig2: schema-constrained decoding](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/fig2_schema_constrained_decode.png)

函数调用 API 保证 JSON 格式良好。但如果你想要 JSON 每次都必须匹配特定 schema，毫无例外呢？

vLLM 和 SGLang 都实现了 **grammar-constrained decoding**。在每个解码步骤， mask 输出分布，只保留语法下能继续构成有效字符串的 token。实现追溯到 **Outlines** ([Willard & Louf, 2023][willard-outlines])，它将 regex 和 JSON-schema 约束编译成有限状态机，在每一步 mask logits。更快的后继者包括 **XGrammar**（SGLang 使用）、 llama.cpp 的 GBNF 和微软的 **Guidance** 库。

```python
# vLLM with JSON schema constraint
from vllm.sampling_params import GuidedDecodingParams

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temp_c": {"type": "number"},
        "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rainy"]},
    },
    "required": ["city", "temp_c", "conditions"],
}

params = SamplingParams(max_tokens=200,
                       guided_decoding=GuidedDecodingParams(json=schema))
out = llm.generate(prompts, params)
# out is guaranteed parseable JSON matching the schema
```

这是最强的输出保证：不是“通常是 JSON"，不是“带有正确键的 JSON"，而是“完全符合 schema 的有效 JSON"。延迟成本很小（XGrammar 开销约 3-5%）。

### Token 级 masking：实际工作原理

内部实现上， grammar-constrained decoding 维护一个 **state machine** 来跟踪当前生成在语法中的位置。在每个解码步骤：

1. 模型在整个 vocabulary 上产生 logits （约 100K-150K tokens）。
2. 状态机计算从当前状态哪些 tokens 是有效 continuation。对于 JSON schema，这可能是：“在左大括号和键名之后，下一个 token 必须是 `:` 或空白。”大多数 tokens 无效。
3. 无效 tokens 的 logits 在 softmax 采样前被设为 `-inf`。
4. 采样的 token 将状态机推进到新状态。

挑战在于性能。 naive 实现每一步都重新计算有效 token mask，复杂度是 $O(\text{vocab\_size})$ —— 在单个 GPU 线程上每步约 0.5 ms，对于小模型来说相当于实际模型 forward pass 的时间。 Outlines 的洞察是预先计算每个语法状态的有效 token bitmap （使用 regex-to-FSM 编译）。 XGrammar 通过字节码风格的状态表示和增量 mask 更新进一步推进了这一点。

现代实现的解码步骤开销 <2%。编译成本（将 JSON schema 转为状态机）对于合理的 schema 通常 <100 ms，所以在请求时可以忽略不计。

有个细微的限制：语法约束影响 *结构*，不影响 *内容*。如果你的 schema 没有包含 `conditions` 的 `enum`，模型可以写 `"conditions": "elephant"` 并通过 schema 验证。 Schema 约束不能让输出变 *真*，只能让它们可 *解析*。
## 自由格式：当无法使用语法约束时

很多 API （大多数非 OpenAI/Anthropic 提供商、端侧推理、内部服务）都不支持语法约束解码。这时候退路只能是靠 Prompt 工程来搞结构化输出：

```python
prompt = """Output a JSON object with these keys, no markdown, no prose:
- "city" (string)
- "temp_c" (number)
- "conditions" (one of: "sunny", "cloudy", "rainy")

Query: What's the weather in Tokyo?
JSON:"""
```

能力强的模型（LLaMA-3.3-70B+, Qwen3-32B+）大概 95-99 % 都能听话。剩下那 1-5 % 的失败通常是这样：

- 给 JSON 包了一层 markdown 代码块。
- 加了个前言（"Sure, here's the JSON: ..."）。
- 多了个 trailing comma。
- 把 JSON 拆到了好几个段落里。

防御性解析能搞定大部分情况：

```python
import json, re

def parse_robust(text: str) -> dict:
    # 1. Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 2. Strip markdown fences
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Find first { ... last }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse: {text[:200]}")
```

我在生产环境里每个自由格式调用都跑这个解析器。失败率直接从 ~3 % 降到了 <0.1 %。

## 错误恢复模式

![fig4: error recovery patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/fig4_error_recovery.png)

工具总会挂的。 API 宕机、数据库 schema 不对、函数超时、参数 invalid。 Agent 得稳住，不能崩。

**模式 1：把错误当成工具结果返回。** 错误信息要作为 *工具结果内容* 返回，别抛异常：

```python
def execute_tool(name, args):
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
```

这样模型能在上下文里看到错误，决定是重试、换工具还是优雅放弃。你要是抛异常，就把模型踢出循环了。

**模式 2：校验逻辑写在工具里，别全指望 schema。** Schema 只管类型，业务逻辑管语义。比如 `transfer_money(from_account, to_account, amount)` 这个工具，得拒绝 `amount=-100` 并返回清晰错误 —— schema 可不知道你的业务规则。

**模式 3：限制重试次数。** 模型要是死磕同一个坏掉的工具，得强行终止循环。设个 `max_tool_calls=10` 上限，再加个 `max_consecutive_errors=3` 上限。我调试过的大多数 Agent 失控，都是模型陷入了“工具失败 → 原样重试”的死循环。

**模式 4：别把 stack trace 丢给模型。** Python stack trace 好几百个 token，小模型看了更晕。返回一句错误描述就行，完整 trace 单独记日志方便调试。

**模式 5：重试时带上理由。** 模型重试工具时，前面加一小段推理说明改了什么。“上次调用失败是因为日期格式不对，这次我用 ISO 8601"比直接重发新参数靠谱得多。有些 Agent 框架（LangGraph, CrewAI）内置了这个机制，出错后自动注入反思 Prompt。

**模式 6：搞不定就问人。** 遇到不可恢复的错误，别无限重试，该升级给用户了。“我试着往这个地址发邮件，但 SMTP 服务器返回‘收件人无效’，能麻烦您核对一下地址吗？”这种体验比静默重试 5 次最后报个通用错误好得多。

**模式 7：优雅放弃。** 要是 Agent 把合理的工具选项都试遍了，返回个带明确说明的部分答案（“我没拿到最新数据，所以这个答案基于昨天的缓存信息”）总比编个看着完整的答案强。

## Agent 循环

![LLM Engineering (7): Function Calling and Tool Use — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/illustration_2.png)


![fig5: agent loop control flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/fig5_agent_loop.png)

生产环境里最小可用的 Agent 循环：

```python
def run_agent(initial_message, tools, max_steps=20):
    messages = [{"role": "user", "content": initial_message}]
    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL, tools=tools, messages=messages, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return response  # final answer

        tool_results = []
        for tu in tool_uses:
            result = execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": str(result),
            })
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("Agent exceeded max_steps")
```

这代码够应付 80 % 的 Agent 场景了。剩下 20 % 得搞这些：

- **工具输出截断**：`read_file` 要是返回 100K tokens，上下文直接爆掉。截断到 ~10K 加个 "[truncated]" 标记。
- **记忆压缩**：第 15 步时对话已经 50K tokens 了；继续之前把旧步骤总结成一条消息。
- **子 Agent**：把复杂子任务（"research X"）拆给独立 Agent，它有自己对话历史，只返回最终总结。
- **工具调用流式输出**：工具调用生成时就 emit，别等完整响应。并行执行工具时能降低 TTFT。

## ReAct、 Voyager 与 Agent 循环的演进

上面那个最小循环其实是 **ReAct 模式** ([Yao et al., 2022][yao-react]) 的改良版。 ReAct  interleaved 三步： Thought （模型推理下一步动作）、 Action （工具调用）、 Observation （工具结果）。"Thought" 这一步很关键 —— 它给了模型一个明确的地方去规划和自我修正。现代 Agent 循环还在实现 ReAct，只不过 thought 隐含在模型的工具调用理由里了，不再单独成段。

**Voyager** ([Wang et al., 2023][wang-voyager]) 在 ReAct 基础上加了三点，专为长程代理设计：自动课程（Agent 根据已知信息自己选下一个子任务）、技能库（存下成功的工具使用模式复用）、带环境反馈的迭代 Prompt 循环。 Voyager 是在 Minecraft 里演示的，但这架构成了生产代码 Agent （Cursor, Cline, Claude Code）和研究型 Agent 的模板。

**Generative Agents** ([Park et al., 2023][park-generative]) 探索了相关方向：带持久记忆流和基于反思的记忆巩固的 Agent。 Park 的 Smallville 模拟显示， 25 个带简单反应行为 plus 记忆 + 反思的 Agent 能产生可信的 emergent 社交行为。这套记忆架构（基于 embedding 的检索 + 定时反思总结）现在是长运行 Agent 系统（比如 SWE-bench 解决方案和个人 AI 助手）的标准配置。

2024-2026 的演进： Agent 拿到了能返回丰富结构化数据的工具、递归子 Agent 调用、以及把明确的任务规划步骤和执行步骤分开。 OpenClaw 的 "Memory-Planning-Tool-Reflection" 架构（OpenClaw 系列第 7-12 章）就是这条演进线的一个具体实例。

## MCP：协议层

2024 年那会儿，每个框架都有自己的工具 spec 格式（LangChain tools, OpenAI function specs, Anthropic tool blocks 等等）。每个都得为每个框架重写一遍。 Anthropic 在 2024 年末发布了 **Model Context Protocol (MCP)**，这是个标准化 JSON-RPC 接口，用来连接 LLM 和工具服务器。

MCP 架构：**clients**（LLM 应用比如 Claude Desktop, Cursor, 自定义 Agent）通过 JSON-RPC 跟 **servers**（工具提供商 —— 文件系统、数据库、 API 等）对话。服务器暴露三个原语：**resources**（只读上下文比如文件内容）、**tools**（可调函数）、**prompts**（可复用 Prompt 模板）。协议负责发现、 schema 验证、流式响应和认证。

MCP 重要是因为它把工具开发和 Agent 框架选型解耦了。你要是给自己内部 API 写了个 MCP server，任何兼容 MCP 的 client 都能用，不用重写。到 2025 年中， MCP server 生态已经包括了 GitHub, GitLab, Slack, Postgres, Sentry, Linear 等几百个服务。这协议在工具集成领域的地位，就像 OpenAI 的 chat-completions schema 在推理 API 领域的地位一样：成了人人适配的事实标准。

有个实际观察： MCP servers 可以嵌入现有应用（你在代码库旁边写个小 Python server）或者作为独立服务运行。纯本地模式让 Claude Desktop 的文件系统访问变得安全 —— server 跑在用户机器上用他们的权限， LLM 没有直接磁盘访问权。这种本地性模型是 MCP 区别于旧式工具服务器协议的地方。

## 生产环境里的坑

**幻觉工具名。** 模型调用了一个不存在的工具。修复：验证 `tu.name in TOOLS` 返回工具未找到错误。有些模型（尤其是 7B 以下的小模型）会幻觉出跟真名 *差不多* 的工具 —— 比如 `get_weather_info` 而不是 `get_weather`。错误里建议最接近的匹配项会有帮助。

**幻觉工具参数。** 模型编了个 schema 里没有的 `force=True` 参数。 Schema 验证能抓到这个；返回清晰的 "parameter X not supported" 错误。

**工具结果错了还自信。** 搜索工具返回 "no results" 但模型还是幻觉出了答案。症状：模型用了工具结果但它的 claim 跟结果矛盾。防御：系统 Prompt 里加明确提醒（"要是工具返回无结果，就这么说"）。高风险用例的话，用另一个模型拿着工具转录事后验证答案。

**工具错误死循环。** 上面的模式 3。永远要设上限。

**延迟连锁反应。** 一个 30-second 工具调用会把用户请求卡住。每个工具调用都设超时。暴露 "这工具慢，要不要试着不用它？" 作为 fallback UX。

**Schema 漂移。** 工具实际返回形状变了（字段改名、加了新必填字段）但你给模型的 schema 定义没更新。模型按旧 schema 发请求，工具失败得莫名其妙。修复：在 dispatcher 里验证工具输出 against schema，把不匹配作为 version-skew 错误暴露。更好：从工具实现生成 schema 定义（比如 Pydantic 模型反射成 JSON schema），这样就漂不了。

**工具转录里 Token 预算耗尽。** 20 步 Agent 运行带着工具调用和结果，轻松到 50-100K tokens。模型撞到上下文限制，要么被截断要么开始丢失早期上下文。修复：实现记忆压缩（工具转录过期前总结早期内容）、长分支用子 Agent、监控每步 Token 用量。

**工具选择歧义。** 两个工具描述重叠（"search documents" vs "find documents"）导致模型 oscillate。修复：写 distinct、互斥的描述，或者合并成一个带消歧参数的工具。
## 核心要点与后续

函数调用这套玩法，其实就是训练行为加上 chat template，再看需不需要加上 grammar enforcement。约束力度能多强就多强，优先级很明确： schema-constrained decoding > JSON mode > prompted JSON。模型支持并发工具调用时就并行跑，有数据依赖再改串行。工具出错了也别直接抛异常，把错误信息当成 tool result 返回给模型，让它自己想办法恢复。还有几条铁律：循环必须封顶，工具调用必须加超时，输出太长必须截断。从思想脉络上看，是从 ReAct 演进到 Toolformer、 Voyager，再到现在的生产级 Agent；协议层面则正逐渐收敛到 MCP。

下一章咱们聊 **retrieval-augmented generation**。包括切分策略、 embedding 模型怎么选、混合检索（dense + sparse）、 reranking，还有实际落地时长上下文和 RAG 到底该怎么选。

## 参考文献

- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023.][yao-react]
- [Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," NeurIPS 2023.][schick-toolformer]
- [Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," 2023.][wang-voyager]
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," UIST 2023.][park-generative]
- [Willard & Louf, "Efficient Guided Generation for Large Language Models (Outlines)," 2023.][willard-outlines]
- [Anthropic, "Introducing the Model Context Protocol (MCP)," 2024.][mcp-spec]
- Karpas et al., "MRKL Systems," 2022 (AI21).
- [Microsoft Guidance library](https://github.com/guidance-ai/guidance)
- [JSONformer: structured output for any LLM](https://github.com/1rgs/jsonformer)
- [OpenAI function calling docs](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic tool use docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)

[yao-react]: https://arxiv.org/abs/2210.03629
[schick-toolformer]: https://arxiv.org/abs/2302.04761
[wang-voyager]: https://arxiv.org/abs/2305.16291
[park-generative]: https://arxiv.org/abs/2304.03442
[willard-outlines]: https://arxiv.org/abs/2307.09702
[mcp-spec]: https://modelcontextprotocol.io/