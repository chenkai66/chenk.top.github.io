---
title: "大模型工程（七）：Function Calling 与工具使用"
date: 2026-05-02 09:00:00
tags:
  - llm
  - function-calling
  - tools
  - agents
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
函数调用是 LLM 和外部世界的桥梁。聊天模板细节（第 2 篇）、结构化输出内核（第 5 篇）和提示工程（第 9 篇），都在这里交汇。我来聊聊底层机制、能依赖的保证，以及在真实负载中管用的 agent 循环模式。

工具使用作为 LLM 的能力，可以追溯到 2022 年两篇几乎同时发布的论文。**MRKL Systems**（Karpas 等，AI21）提出神经-符号模块间的专家路由。**ReAct**（[Yao 等，2022][yao-react]）把思维链推理和工具操作结合起来。到了 2023 年，**Toolformer**（[Schick 等，2023][schick-toolformer]）展示了自监督教学方法。模型通过在现有文本中插入工具调用标记生成训练数据。

2024 年，所有前沿模型的后训练数据都围绕工具使用格式构建。工具调用从“研究演示”变成了“API 特性”。

![大模型工程（七）：Function Calling 与工具使用 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/illustration_1.jpg)
## "Function calling" 实际指的是什么

API 提供 "function calling" 时，背后可能有几种机制。

1. **训练行为 + 聊天模板标记**  
模型经过后训练，会在适当时候生成工具调用，并用特殊标记包裹。比如，Qwen3 用 `tool_call` 标签，Mistral 用 `[TOOL_CALLS]`。

2. **JSON 模式约束解码**  
通过语法约束解码，强制输出合法 JSON。模型不一定专门训练过；解码器负责执行约束。

3. **Schema 引导的结构化输出**  
在 JSON 模式基础上，进一步要求 JSON 符合特定 schema，比如函数名和参数类型。

4. **自由格式提示**  
在系统提示中写 "请以 JSON 格式响应，键为 X、Y"，然后寄希望于模型。能力强的模型有时能奏效，但没有保证。

实际生产系统通常混合使用这四种方法。OpenAI 和 Anthropic 的 API 结合了 1 和 3。vLLM 和 SGLang 为任何模型实现了 2 和 3。自由格式（4）是其他方法不可用时的备选。

还有一个重要区别：**JSON 和 XML 工具格式**。  
OpenAI 默认用 JSON 工具调用；Anthropic Claude 内部按 XML 类似的结构化输出训练，但在 API 中暴露为 JSON。Anthropic 团队在 2024 年公开表示，XML 标签更容易让模型学习。尖括号结构与训练数据中标记特殊区域的方式一致，部分 XML 在流中也比部分 JSON 更容易解析。实际上两种格式都能工作，选择主要影响解析工具链。在模型内部，两者都是带有学到文法的 token 序列。
## 一次真实的 function-call 请求

![fig1: tool-call request/response sequence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/07-function-calling/fig1_tool_sequence.png)

Anthropic Claude API 示例：

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-4-5-sonnet-20250901",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "获取某个地点的当前天气。",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市和州，例如 San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    }],
    messages=[{"role": "user", "content": "东京的天气怎么样？"}],
)
# response.content 可能是：
# [TextBlock(text="我来查一下东京的天气。"),
#  ToolUseBlock(id="toolu_xxx", name="get_weather",
#               input={"location": "Tokyo, Japan", "unit": "celsius"})]
```

模型返回一个结构化的 `ToolUseBlock`。我执行工具后，发送一条后续消息：

```python
response2 = client.messages.create(
    model="claude-4-5-sonnet-20250901",
    max_tokens=1024,
    tools=[...],  # 同样的工具
    messages=[
        {"role": "user", "content": "东京的天气怎么样？"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_xxx",
            "content": "72°F，局部多云",
        }]},
    ],
)
```

会话中有了 tool call 和 tool result。下一轮 assistant 就能用这些信息了。这就是基本循环。
## 工具定义最佳实践

工具定义其实就是提示词。每次调用时，模型会把它当成系统提示的一部分读取。定义质量直接影响模型选工具、调用工具和错误恢复的能力。

**描述是提示中最常被读的部分。** 好的描述要包括三件事：工具功能一句话概括，什么时候用（什么时候不用），返回什么结果。“搜索数据库”太差了。好的例子是：“在客户数据库中搜索符合条件的订单。用户问具体订单或历史时用。返回最多 10 条最近记录。”

**参数描述比名字更重要。** 模型看到 `location` 能猜到需要地点，但不知道格式是 "Tokyo"、"Tokyo, Japan" 还是 "JP/Tokyo"。必须加格式示例。

**尽量用枚举。** `unit` 参数加 `enum: ["celsius", "fahrenheit"]` 比自由字符串好得多。Schema 约束能防止模型乱发明，比如 "Kelvin" 或 "C°"。

**复杂工具要加示例调用。** “Example: `transfer_money({from_account: 'A123', to_account: 'B456', amount: 100, currency: 'USD'})`” 比长篇大论更实用。

**写清楚错误格式。** 比如：“账户不存在返回 404，权限不足返回 403。” 这样模型能正确处理错误，决定重试还是上报。

**别定义太多工具。** 我见过一个注册表有 80 个工具，模型老是选错。把相关工具合并成少量多态工具更好。比如用一个带可选过滤器的 `query_orders`，而不是分开写 `query_orders_by_id`、`query_orders_by_date` 等。模型填参数比选菜单强多了。
## 并行工具调用

![fig3: 并行与串行工具执行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/07-function-calling/fig3_parallel_vs_sequential.png)

到 2026 年，所有前沿模型都会支持并行工具调用。简单说，就是一次输出多个工具调用块。比如用户问“东京和纽约天气如何？”，模型会同时发出两个调用。我并行执行它们，再把结果一起返回。

这很重要。串行调用会叠加延迟。一个代理调用 5 个工具，每个耗时 200 毫秒，总延迟就是 1 秒。换成并行，延迟直接降到 200 毫秒。对于多数据源的代理（比如 OpenClaw 第 7 到 12 章提到的场景），这是“流畅”和“卡顿”的区别。

但并行调用有个前提：工具之间不能有依赖关系。查两个城市的天气没问题。“搜索航班，再订最便宜的”就不行，因为第二个调用依赖第一个的结果。模型训练时应该学会这一点，但实际中不一定靠谱。生产环境里，我会先验证调用是否真的独立。

依赖分析有时很复杂。比如两个工具都写同一个外部资源（比如对同一行的两个 `update_database` 调用）。即使它们不依赖对方的返回值，并行运行也可能引发竞态条件。更安全的做法是给工具分类：只读、写隔离、写共享。只在兼容类别内并行化。Anthropic 的 Claude 在 2025 年底比 GPT 类模型更保守，原因就在这里。不确定依赖关系时，它会偏向串行执行。
## 用文法做结构化输出

![fig2: schema-constrained decoding](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/07-function-calling/fig2_schema_constrained_decode.png)

Function-calling API 能生成格式正确的 JSON。但我想让每次生成的 JSON 都严格符合某个特定 schema，怎么办？

vLLM 和 SGLang 都支持 **文法约束解码**。每一步解码时，只保留能生成合法字符串的 token。这个方法最早来自 **Outlines**（[Willard & Louf, 2023][willard-outlines]）。它把正则表达式和 JSON-schema 约束编译成有限状态机，用来掩码每一步的 logits。后来更快的实现包括 **XGrammar**（SGLang 使用）、llama.cpp 的 GBNF 和微软的 **Guidance** 库。

```python
# vLLM JSON schema 约束
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
# out 保证可解析且严格符合 schema
```

这是最强的输出保证：不是“通常是 JSON”，也不是“JSON 带正确键”，而是“完全符合 schema 的有效 JSON”。延迟开销很小，用 XGrammar 时仅 3-5%。

### Token 级掩码：实际原理

文法约束解码内部维护一个 **状态机**，跟踪当前生成位置。每步解码流程如下：

1. 模型在全词表（约 100K-150K token）上生成 logits。
2. 状态机计算当前状态下哪些 token 合法。比如，JSON schema 中“开括号和键名后，下一个 token 必须是 `:` 或空白”。多数 token 不合法。
3. 不合法 token 的 logits 在 softmax 前设为 `-inf`。
4. 采样到的 token 推进状态机到新状态。

性能是个难点。朴素实现每步重新计算合法 token 掩码，复杂度 $O(\text{vocab\_size})$。单 GPU 线程上每步耗时约 0.5 ms，接近小模型前向传递时间。Outlines 的创新点是预计算每个文法状态的合法 token 位图（通过正则表达式转有限状态机）。XGrammar 更进一步，采用字节码风格状态表示和增量掩码更新。

现代实现解码开销低于 2%。编译开销（将 JSON schema 转状态机）通常小于 100 ms，请求时可以忽略。

一个小问题：文法约束只管 *结构*，不管 *内容*。如果 schema 没定义 `conditions` 的 `enum`，模型可能生成 `"conditions": "elephant"` 并通过验证。Schema 约束只能保证输出 *可解析*，不能保证 *真实性*。
## 自由格式：没有文法时

很多 API 不支持文法约束解码。比如大多数非 OpenAI/Anthropic 提供商、设备端推理、内部服务。这时可以用 prompt 工程生成结构化输出：

```python
prompt = """输出 JSON 对象，包含以下键，不要用 markdown，也不要写成散文：
- "city"（字符串）
- "temp_c"（数字）
- "conditions"（"sunny"、"cloudy"、"rainy" 之一）

查询：东京天气如何？
JSON:"""
```

能力强的模型，比如 LLaMA-3.3-70B+ 和 Qwen3-32B+，95%-99% 的情况下都能正确输出。剩下的 1%-5% 失败情况主要有这些：

- 把 JSON 包在 markdown 代码块里。
- 加上前言，比如“好的，这是 JSON：...”。
- 多了一个尾逗号。
- JSON 跨多个段落。

防御性解析能解决大部分问题：

```python
import json, re

def parse_robust(text: str) -> dict:
    # 先直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 找到第一个 { 和最后一个 }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析: {text[:200]}")
```

每次自由格式结构化输出，我都用这个解析器。生产流程中的失败率从约 3% 降到 <0.1%。
## 错误恢复模式

工具会出问题。API 挂了，数据库返回意外 schema，函数超时，参数不合法。Agent 必须妥善处理。

**模式 1：错误当结果返回。** 不抛异常，直接把错误信息塞进工具结果：

```python
def execute_tool(name, args):
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
```

模型看到错误后，可以决定重试、换工具，或者优雅退出。抛异常会让模型掉线。

**模式 2：工具内验证业务逻辑。** Schema 只管类型，业务逻辑自己校验语义。比如 `transfer_money(from_account, to_account, amount)`，遇到 `amount=-100` 要明确拒绝。Schema 不懂业务规则。

**模式 3：限制重试次数。** 模型反复调用坏工具时，必须打断循环。我设过 `max_tool_calls=10` 和 `max_consecutive_errors=3`。踩过的坑告诉我，大多数 agent 跑飞都是因为 "工具失败 → 照样重试" 的死循环。

**模式 4：别暴露堆栈跟踪。** Python 堆栈动辄几百个 token，小模型根本看不懂。返回一句简单错误描述，完整堆栈单独记日志。

**模式 5：重试带上理由。** 模型重试时，先说明原因。比如："上次日期格式错了，这次改用 ISO 8601。" 这比瞎调参数更靠谱。LangGraph 和 CrewAI 这类框架会在错误后插入反思提示。

**模式 6：求助用户。** 遇到不可恢复的错误，别死磕，直接找用户。比如："我发邮件失败，SMTP 提示地址无效，能再确认下吗？" 比静默重试 5 次后报错强多了。

**模式 7：优雅放弃。** 工具都试过了还不行，就返回部分答案，带上说明。比如："最新数据没拿到，这是基于昨天缓存的结果。" 比硬编一个假答案靠谱。

下一节会讲如何设计工具接口。
## Agent 循环

![大模型工程（七）：Function calling、结构化输出与 Agent loop 的工程实操 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/07-function-calling/illustration_2.jpg)

生产环境里最简的 agent 循环代码如下：

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
            return response  # 最终答案

        tool_results = []
        for tu in tool_uses:
            result = execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": str(result),
            })
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("Agent 超过最大步数")
```

这段代码能搞定 80% 的 agent 场景。剩下的 20%，需要额外处理。

工具输出太长会炸上下文。比如 `read_file` 返回 10 万 token，必须截断到 1 万左右，加个 "[truncated]" 标记。

对话历史太长也有问题。第 15 步时可能已经有 5 万 token。继续之前，把早期内容总结成一条消息。

复杂任务可以拆给子 agent。比如“研究 X”，单独开一个 agent，只返回最终结果。

工具调用可以流式处理。不用等完整响应，边生成边发送。这样能降低并行执行的 TTFT。
## ReAct、Voyager 与 agent 循环的谱系

最简循环是对 **ReAct 模式**（[Yao 等，2022][yao-react]）的改进。ReAct 分三步走：思考、行动、观察。思考是模型推理下一步动作，行动是调用工具，观察是获取工具结果。这一步很关键，给模型留了规划和自我修正的空间。现代 agent 循环依然沿用 ReAct，但思考过程藏在工具调用的理由里，不再单独写成文本块。

**Voyager**（[Wang 等，2023][wang-voyager]）为长程任务扩展了 ReAct。它加了三个功能：自动课程，agent 根据已知选下一个子任务；技能库，存储并复用成功的工具使用模式；迭代提示循环，带环境反馈。Voyager 在 Minecraft 上展示过，后来成了生产代码 agent（如 Cursor、Cline、Claude Code）和研究 agent 的模板。

**Generative Agents**（[Park 等，2023][park-generative]）探索了另一个方向：持久记忆流和基于反思的记忆整合。Park 的 Smallville 模拟中，25 个 agent 只靠简单反应行为、记忆和反思，就产生了可信的社交行为。这种记忆架构（嵌入检索 + 定期反思总结）现在已经是 SWE-bench 解决方案和个人 AI 助手的标准配置。

2024-2026 年，agent 工具更强大了。工具能返回丰富结构化数据，支持递归调用子 agent，还把任务规划和执行分开。OpenClaw 的 "Memory-Planning-Tool-Reflection" 架构（OpenClaw 系列第 7-12 章）就是这个谱系的一个实例。
## MCP：协议层

2024 年，每个框架都有自己的 tool-spec 格式。比如 LangChain tools、OpenAI function specs、Anthropic tool blocks 等等。每种格式都要为不同框架重新实现。到了 2024 年底，Anthropic 推出了 **Model Context Protocol (MCP)**。这是一个标准化的 JSON-RPC 接口，用来连接 LLM 和工具服务器。

MCP 的架构很简单。**客户端**是 LLM 应用，比如 Claude Desktop、Cursor 或自定义代理。它们通过 JSON-RPC 和 **服务器**通信。服务器就是工具提供者，比如文件系统、数据库或 API。服务器暴露三种核心功能：**资源**（只读上下文，比如文件内容）、**工具**（可调用函数）和 **提示**（可复用的提示模板）。协议负责发现、模式验证、流式响应和认证。

MCP 的意义在于解耦。工具开发不再依赖特定的代理框架。如果我为内部 API 写一个 MCP 服务器，它就能兼容所有支持 MCP 的客户端，完全不用重写代码。到 2025 年年中，MCP 生态已经很丰富了。GitHub、GitLab、Slack、Postgres、Sentry、Linear 都加入了，还有几百个其他服务。这个协议已经成为工具集成的事实标准，就像 OpenAI 的 chat-completions 对推理 API 的意义一样。

实际用起来，MCP 服务器有两种部署方式。一种是嵌入现有应用，比如在代码库旁边写个小 Python 服务器。另一种是独立运行的服务。本地模式特别有意思。Claude Desktop 的文件系统访问之所以安全，就是因为服务器跑在用户机器上，用的是用户权限。LLM 没有直接磁盘访问权限。这种本地性模型让 MCP 和老一代工具协议拉开了差距。

下一节会讲 MCP 的具体实现细节。
## 生产中常见的问题

**虚构工具名称。** 模型调用了不存在的工具。解决方法很简单：验证 `tu.name in TOOLS`，返回工具未找到错误。小模型（尤其是 7B 以下）容易生成接近真实名称的假名，比如 `get_weather_info` 而不是 `get_weather`。在错误信息里建议最接近的匹配项，效果不错。

**虚构工具参数。** 模型会发明 schema 中没有的参数，比如 `force=True`。Schema 验证能抓到这个问题。直接返回清晰的错误："参数 X 不支持"。

**自信地给出错误结果。** 搜索工具返回 "无结果"，但模型还是凭空编了个答案。症状很明显：模型用了工具结果，但结论和结果矛盾。防御方法：在系统提示里加一句提醒，比如 "如果工具返回无结果，直接说明"。高风险场景下，可以用另一个模型对答案和工具日志进行后验证。

**陷入工具错误循环。** 参考前面提到的模式 3。记住，一定要设置上限。

**工具延迟级联。** 一个工具调用耗时 30 秒，用户的请求就卡住了。每个工具调用都得设超时。备用方案是提示用户："这个工具很慢，要不要不带它再试一次？"

**Schema 漂移。** 工具的实际返回结构变了，比如字段重命名或新增必填字段，但给模型的 schema 定义没更新。模型按旧 schema 发请求，工具就会失败，还不知道为啥。解决方法：在调度器里校验工具输出和 schema 的一致性，发现不匹配就报版本偏差错误。更好的办法是从工具实现自动生成 schema 定义，比如用 Pydantic 模型反射成 JSON schema，彻底避免漂移。

**工具脚本中的 token 预算耗尽。** 20 步 agent 运行、工具调用和结果，轻松吃掉 50-100K token。模型一到上下文限制，要么被截断，要么丢掉早期上下文。解决方法：实现记忆压缩，在早期工具脚本老化前总结一下；长分支交给子 agent 处理；每一步都要监控 token 使用量。

**工具选择模糊。** 两个工具描述重叠，比如 "搜索文档" 和 "查找文档"，模型就会摇摆不定。解决方法：写清楚互斥的描述，或者合并成一个工具，用参数来区分。

下一节会讲一些踩过的坑和血泪经验。
## 小结与下一篇

函数调用是训练行为、聊天模板和可选语法约束的结合。优先选择最强的保障方式：模式约束解码 > JSON 模式 > 提示生成 JSON。模型支持时，并行调用工具；有数据依赖时，串行调用。工具错误直接作为结果返回，让模型自行恢复。循环次数要设上限，工具调用必须加超时，大输出记得截断。

技术脉络从 ReAct 到 Toolformer，再到 Voyager，最终演进到现代生产代理。协议方面正逐步统一到 MCP。

下一篇聊**检索增强生成**。切分策略、嵌入模型选型、混合检索（密集 + 稀疏）、重排序，还有长上下文和 RAG 的实际权衡。
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
