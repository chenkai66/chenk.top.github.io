---
title: "AI Agents Complete Guide: From Theory to Industrial Practice"
date: 2024-02-11 09:00:00
tags:
  - LLM
  - AI Agents
  - Applications
categories: Large Language Models
lang: en
description: "A practitioner-grade guide to building AI agents: planning (CoT/ReAct/ToT), memory architectures, tool use, reflection, multi-agent patterns, frameworks (LangChain, LangGraph, AutoGen, CrewAI), evaluation, and production concerns."
disableNunjucks: true
---

A chatbot answers questions. An *agent* gets things done -- it browses, runs code, calls APIs, queries databases, and iterates until the job is finished. The same LLM sits behind both, but the wrapper is different: an agent runs inside a loop with tools, memory, and the ability to inspect its own work.

This guide is the long-form version of that idea. It covers the four core capabilities (planning, memory, tool use, reflection), the major framework families, multi-agent collaboration, evaluation, and the production concerns that decide whether an agent ships or quietly fails on a Tuesday afternoon.

## What you will learn

- Why an agent is more than "GPT-4 in a `while` loop"
- The four capabilities every serious agent implements
- ReAct, Tree of Thoughts, and Reflexion -- when each is worth the cost
- Framework trade-offs: LangChain, LangGraph, AutoGen, CrewAI, AutoGPT
- Multi-agent topologies that actually scale (and ones that look great in demos but break)
- An evaluation framework that catches real regressions
- Production checklist: cost, safety, observability, failure modes

## Prerequisites

- Comfortable with Python and the `requests`/`json` style of API code
- Familiar with at least one LLM API (OpenAI, Anthropic, etc.)
- A working mental model of prompts, completions, and context windows

---

## What an AI Agent actually is

An AI agent is a system in which an LLM, given a goal, autonomously decides what to do next, executes that action through tools, observes the result, and repeats until the goal is met or a stopping condition triggers.

That sentence has four load-bearing words: **goal**, **decide**, **execute**, **observe**. A single LLM call has none of them in any meaningful sense; it produces text and stops. An agent re-enters the model with new context every step, which is why the same underlying LLM can suddenly book a flight, fix a flaky test, or drive a browser.

### From one-shot generation to a cognitive loop

The shift from request-response to a loop is the entire story:

```python
# Traditional LLM: one shot, one answer
def traditional_llm(prompt: str) -> str:
    return llm.generate(prompt)


# Agent: a loop with state, tools, and a stop condition
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

Three things make this loop work in practice: a memory that survives across steps, a tool layer that returns structured results, and a stopping rule that prevents an enthusiastic LLM from ordering 200 pizzas. Everything else in this guide is variations on those three themes.

![Agent cognitive loop: perceive, reason, act, remember](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig1_agent_loop.png)

### Why this matters: a concrete contrast

Consider the goal "find the top three ML conference deadlines for 2026 and schedule reminders two weeks before each."

A standalone LLM call returns whatever it remembers from training -- often a year out of date -- and physically cannot create calendar events. An agent searches the web for current dates, parses the results, computes the reminder dates, calls a calendar API, and returns confirmation IDs. The LLM is the same; the wiring is what differs.

![Rule-based system vs LLM-based agent](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig2_agent_vs_rule.png)

The diagram above is worth pausing on. A rule-based system is a hard-coded decision tree: every new intent costs another branch. An LLM agent is a generalist core that *delegates specifics to tools*; new behaviour is added by adding a tool description, not by editing control flow.

### The five components of an agent

Every effective agent has the same five parts. The names vary across frameworks; the responsibilities do not.

**1. The brain (LLM core).** A reasoning engine that interprets goals, generates plans, and emits tool calls. Choice of model dominates capability and cost.

**2. The planner.** Decomposes goals into sub-goals. Can be implicit (the LLM plans on the fly inside ReAct) or explicit (a separate planner emits a DAG of tasks).

**3. Memory.** Working memory holds the live trajectory; long-term memory persists facts, lessons, and embeddings across runs.

**4. Tools.** Anything the agent can call: search, code execution, SQL, HTTP APIs, file I/O, other agents. The schema you give the LLM matters more than the tool itself.

**5. Reflection.** A self-critic that evaluates outputs and can trigger a retry with corrective context.

The minimal version of components 3 and 4 fits in a few lines:

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
        # Cheap heuristic: 4 chars per token
        while sum(len(m["content"]) for m in self.messages) // 4 > self.budget_tokens \
                and len(self.messages) > 2:
            self.messages.pop(0)
```

Production systems layer additional concerns on top -- structured logging, retries, telemetry -- but the kernel is exactly this.

---

## Capability 1: Planning

Planning is what turns "summarise this 200-page report and email the highlights to the team" into a sequence of steps the agent can actually execute. Three planning patterns dominate.

### Chain-of-Thought (CoT)

The simplest pattern: ask the model to verbalise its reasoning before producing the answer. CoT works well when reasoning is mostly internal -- arithmetic, logic puzzles, structured rewriting. It does *not* by itself enable tool use; it only spreads the reasoning over more tokens.

CoT is a cost-effective default. Use it as a baseline before reaching for anything fancier.

### ReAct (Reason + Act)

ReAct interleaves reasoning ("Thought") with tool calls ("Action") and tool outputs ("Observation"). It is the de facto standard for tool-using agents because the same LLM call decides both what to think and what to do.

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

A typical ReAct trace looks like the figure below. Each row corresponds to one LLM call.

![ReAct trace: interleaved reasoning and acting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig5_react_flow.png)

Two practical notes about ReAct. First, the parser is the source of most production bugs -- prefer JSON mode or function calling over free-form parsing. Second, observations from tools should be *summarised before being fed back* if they are large; raw HTML pages will blow your context window in three steps.

### Tree of Thoughts (ToT)

ToT generalises CoT to a search tree: at each step the model proposes several candidate thoughts, scores them, and either expands the most promising or backtracks. It is genuinely more capable on problems with combinatorial structure (Game of 24, creative writing, code search) but it costs *N* times more LLM calls, where *N* is the branching factor.

A reasonable rule of thumb: only reach for ToT when ReAct fails consistently and you can articulate why. For most production tasks, ReAct with retries beats ToT on cost-effectiveness.

### When to use which

| Pattern | Best for | Cost | Failure mode |
|---|---|---|---|
| **CoT** | Closed-form reasoning, no tools | 1x | Hallucinates if facts needed |
| **ReAct** | Tool-using tasks, web/code/data | 3-15x | Loops, parser errors |
| **ToT** | Search problems, design exploration | 10-50x | Slow, expensive, hard to debug |

---

## Capability 2: Memory

Memory is the difference between an agent that helps you once and an agent that learns your preferences over months. Four memory types are worth distinguishing.

| Memory type | Lifetime | Typical store | Use case |
|---|---|---|---|
| **Working** | Within one task | In-context buffer | Current trajectory |
| **Entity** | Per user / per project | Key-value + LLM extraction | "User prefers Postgres" |
| **Semantic** | Long-term | Vector DB (FAISS, pgvector, Chroma) | Past experiences, docs |
| **Episodic** | Long-term | Structured log + embeddings | Replay of past runs for learning |

### Working memory: the boring one that breaks first

Working memory is just the message list you send to the LLM each turn. It is the most under-engineered piece of most agents and the one that fails first under load. Three things help:

1. **Summarise old turns** rather than dropping them. A running summary of "what we have established so far" preserves continuity.
2. **Separate scratchpad from output**. The model's intermediate thoughts should be visible to *itself* but stripped from the user-facing answer.
3. **Token-budget eviction**, not message-count eviction. A single 10K-token tool output can blow your budget while you are still trimming three-line user messages.

### Long-term memory with vector search

The standard recipe: embed each interaction, store the embedding plus the original text, and retrieve top-k by cosine similarity when starting a new task. Production deployments use Pinecone, Weaviate, Milvus, or pgvector; for prototyping, FAISS in-process is fine.

```python
class VectorMemory:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store  # any vector backend with add/query

    def remember(self, text: str, metadata: dict) -> None:
        vec = self.embedder.embed(text)
        self.store.add(vec, text=text, **metadata)

    def recall(self, query: str, k: int = 5) -> list[str]:
        vec = self.embedder.embed(query)
        hits = self.store.query(vec, k=k)
        return [h["text"] for h in hits]
```

Two failure modes to watch for: **retrieval drift** (semantic search retrieves topically similar but contextually wrong items, e.g. last month's bug report when you want this week's) and **memory poisoning** (an early hallucinated fact gets stored and retrieved on every subsequent run). Mitigate the first with metadata filters and time decay; mitigate the second with explicit confidence labels and a periodic memory audit.

### Forgetting on purpose

Memory cannot grow without bound. A useful retention score combines importance (assigned at write time), access frequency, and recency:

```
score = 0.5 * importance + 0.3 * normalised_access_count - 0.2 * days_since_last_access
```

Drop the bottom 10% on a schedule. Keep the eviction log; you will want it when something starts behaving oddly.

---

## Capability 3: Tool use

Tools are how an agent leaves the LLM's hallucinated world and operates on the real one. The pipeline is the same regardless of provider: declare tool schemas, let the model emit a structured call, validate, execute, feed the result back.

![Function calling pipeline: schema, model, execution, result](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig3_tool_calling_pipeline.png)

### OpenAI-style function calling

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
    # second turn: model summarises tool output
    return client.chat.completions.create(
        model="gpt-4o", messages=messages
    ).choices[0].message.content
```

Anthropic's `tool_use` API has a different message shape but the same semantics. LangChain, LlamaIndex, and the agent SDKs all wrap one of these.

### Tool design: the part that actually matters

Most "agent failures" are tool-design failures in disguise. Three rules:

**Make tools idempotent and granular.** A `update_user(...)` tool with twelve optional fields invites the model to set the wrong one. Split it into `set_user_email`, `set_user_address`, etc. Each one is easier to describe, test, and validate.

**Return structured errors, not exceptions.** When `get_weather("Atlntis")` fails, return `{"error": "city_not_found", "suggestion": "Atlanta"}`, not a 500. The model can act on the suggestion; it cannot act on a stack trace.

**Include examples in the description.** "Search GitHub issues. Example: `search_issues(repo='langchain-ai/langchain', label='bug', state='open')`." A two-line example doubles tool-call accuracy on borderline queries.

### Computer use and the next frontier

Anthropic's *Computer Use* (October 2024) and similar systems give the agent screenshots and primitives like `mouse_click(x, y)` and `type_text("...")`. This unlocks any application with a UI, including those without an API. The trade-off is reliability -- a 95%-accurate click rate compounds to a 60% success rate over a 10-step task. Treat computer-use agents as research-grade and sandbox them aggressively.

---

## Capability 4: Reflection

Reflection is the agent's internal critic. Without it the agent makes the same mistake on retry; with it the agent learns inside a single run.

### Self-Refine

The minimal pattern: generate, critique, regenerate.

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

Self-Refine helps most on text generation tasks (writing, summarisation, code review). It helps less on tool-using tasks because the critic cannot easily distinguish "wrong action" from "correct action with bad data."

### Reflexion: lessons across attempts

Reflexion (Shinn et al., 2023) adds an explicit verbal *lesson* after each failed attempt and feeds it into the next attempt. The lesson is short, concrete, and actionable -- "When the SQL query returns zero rows, try removing the date filter before changing the table." Over a few retries, a Reflexion agent often solves problems that no single-shot agent of the same model can.

Reflexion is most effective when:
- There is a clear success signal (test passes, answer matches).
- Each attempt is cheap relative to the potential reward.
- The task family is consistent enough for lessons to transfer.

### Pitfalls

Reflection isn't free. Two specific failure modes:

- **Sycophantic critique.** The same model that produced the answer will often agree the answer is fine. Mitigate by using a different model (or a different prompt persona) as the critic.
- **Reflection loops.** Critique-rewrite cycles can converge to a fixed point that is wrong but consistent. Cap the number of iterations and fall back to the original output if confidence does not improve.

---

## Frameworks: choosing where to live

The framework landscape moves quickly, but the positions on the map are stable. The figure below sketches where each tool sits along two axes: how composable it is and how autonomous the resulting agents tend to be.

![Agent frameworks and platforms positioning map](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig6_platform_landscape.png)

A summary of what each is good for:

| Framework | What it is | Pick it when |
|---|---|---|
| **LangChain** | Toolkit of components | You want building blocks, not opinions |
| **LangGraph** | DAG runtime built on LangChain | You need explicit control flow, branches, loops |
| **LlamaIndex** | RAG-first agent SDK | The agent is mostly retrieving from documents |
| **AutoGen** (Microsoft) | Multi-agent conversation framework | You want agents that talk to each other |
| **CrewAI** | Role-based multi-agent orchestration | You want "team of specialists" semantics out of the box |
| **Semantic Kernel** | Microsoft's planner + skills SDK | You live in the .NET / Azure ecosystem |
| **OpenAI Assistants** | Hosted agent runtime | You want OpenAI to manage state and tools |
| **AWS Bedrock Agents** | Hosted, AWS-native | You are already on Bedrock |
| **Dify / Flowise** | No-code agent builders | Non-engineers will build flows |
| **AutoGPT / BabyAGI** | Autonomous task loops | Research, exploration, demos |

A short LangGraph example, because it captures the modern shape of an agent better than most:

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

The graph form makes branches and loops explicit, which is exactly what you want when an agent starts misbehaving in production and you need to step through what happened.

### A note on the autonomy spectrum

AutoGPT-style agents that pursue open-ended goals look magical and ship rarely. The same task expressed as a LangGraph with three nodes and two edges is less impressive on Twitter and dramatically more shippable. Most production "agents" today are closer to the right side of the LangGraph example than to the left side of AutoGPT.

---

## Multi-agent systems

A single agent hits ceilings on long, heterogeneous tasks: it has to keep too many concerns in one prompt, the trajectory grows past the context window, and reflection becomes muddled. Multi-agent systems split the work across specialists.

![Three multi-agent collaboration patterns](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig4_multi_agent_patterns.png)

### Pattern 1: Hierarchical (manager + workers)

A manager agent decomposes the task and dispatches sub-tasks to specialists (researcher, coder, analyst, writer). Each specialist owns its tool set and returns a structured result. Works well when the task fans out cleanly. Watch out for the manager becoming a bottleneck: if every result has to flow back through it for the next decision, you have replaced one long context with one long *coordination* context.

### Pattern 2: Debate (adversarial critique)

Two or more agents argue different positions; a judge agent (or a vote) picks the winner. Useful for tasks with a quality dimension that's hard to verify directly -- legal arguments, design choices, complex reasoning. Empirically, debate over 3-5 rounds beats single-agent reasoning by a meaningful margin on benchmarks like GSM-Hard, at the cost of multiplying token spend.

### Pattern 3: Pipeline (specialist chain)

The MetaGPT model: Product Manager -> Architect -> Engineer -> QA, each agent producing the input for the next. This is the most "boring" pattern and the one that ships. It works because each handoff produces a *named artifact* (PRD, design doc, code, test report), which gives you natural checkpoints and recovery points.

### Communication

Multi-agent systems live or die on the message bus. The minimum useful protocol:

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str        # or "*" for broadcast
    type: str             # "request" | "response" | "broadcast"
    content: dict
    correlation_id: str   # ties responses to requests
    timestamp: datetime
```

Always log every message. When a multi-agent system misbehaves, the failure is almost always in *what one agent thought another agent had said*; without the message log you are guessing.

---

## Evaluation

You cannot improve what you cannot measure. Agent evaluation is harder than LLM evaluation because the answer space is huge and partial credit matters. Two strategies, used together.

![Evaluation framework: capability profile and benchmark coverage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/ai-agents-complete-guide/fig7_evaluation_framework.png)

### Public benchmarks

| Benchmark | Tests | Notes |
|---|---|---|
| **AgentBench** | 8 environments (OS, DB, KG, web, etc.) | Broad coverage, GPT-4 leads |
| **GAIA** | Real-world assistant tasks, 3 difficulty levels | Hard. Even GPT-4 + tools struggles past Level 2 |
| **AgentBoard** | Fine-grained capability scoring | Useful for diagnosis, not just ranking |
| **WebArena** | Web navigation in realistic sites | Pure browser-use evaluation |
| **SWE-bench** | Resolve real GitHub issues | The bar for coding agents |
| **ToolBench** | Tool selection across 16k+ APIs | Tests breadth of tool use |

Use public benchmarks to track *general* capability of new model/agent versions. Do not use them as your acceptance test -- your task distribution is not theirs.

### Internal eval suites

The eval that actually catches regressions is one you write for your own task distribution. The shape that works:

```python
@dataclass
class EvalCase:
    id: str
    input: str
    rubric: dict          # fields to score: correctness, format, etc.
    must_use_tool: list   # required tools
    must_not_call: list   # forbidden actions (e.g. send_email in dry-run)
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

Three things to insist on:

1. **Trajectory-level checks**, not just output checks. "Did the agent send an email it should not have?" is at least as important as "is the answer correct?"
2. **Cost as a first-class metric.** A correct answer for $0.04 and a correct answer for $4.00 are not equivalent.
3. **Reproducibility.** Pin model versions, seeds (where supported), tool mocks, and timestamps. Otherwise yesterday's regression test passes today for the wrong reason.

---

## Production concerns

The gap between a working notebook and a production agent is usually larger than the gap between knowing nothing and the working notebook. The list below is the short version of what bites.

### Cost control

LLM calls inside a loop have a way of becoming the line item that requires a meeting. Three practices:

- **Tiered routing.** Cheap model for classification and easy turns; expensive model only for the hard ones.
- **Hard budget per task.** Reject or downgrade work that would exceed `max_cost_usd`.
- **Cache aggressively.** Identical tool calls and identical sub-prompts should hit a cache, not the API.

```python
class BudgetedAgent:
    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent = 0.0

    def call(self, prompt, model="gpt-4o"):
        est = estimate_cost(prompt, model)
        if self.spent + est > self.budget:
            model = "gpt-4o-mini"  # downgrade
        resp = client.complete(model=model, prompt=prompt)
        self.spent += actual_cost(resp)
        return resp
```

### Safety and sandboxing

If your agent can execute code or shell commands, it *will* try to execute the wrong one eventually. Defenses, in priority order:

1. **Run in a sandbox.** Container with no network, no credentials, capped memory and CPU, time limit.
2. **Allow-list tools and parameter values.** Never let an LLM-emitted string become a shell command unfiltered.
3. **Approval gates for destructive actions.** Sending email, writing to production databases, charging cards: these need a human or a strict policy check, not just a model's confidence.
4. **Audit log of every tool call.** With inputs, outputs, and the LLM call that produced the action.

### Observability

You will not understand what your agent did without telemetry. Track:

- Per-step latency, tokens in/out, cost, and tool name.
- Full trajectory with thoughts, actions, observations.
- Outcome (success / failure / timeout / budget) and reason.
- A stable `task_id` and `parent_task_id` for multi-agent runs.

Tools like LangSmith, Langfuse, Helicone, and Arize Phoenix give you the agent equivalent of APM dashboards. Use one of them. Building this from scratch is a six-month side quest you do not want.

### Common failure modes

| Symptom | Likely cause | First fix |
|---|---|---|
| Infinite loop | No stopping condition / loop in tool calls | Hard step cap + loop detection |
| Tool keeps getting wrong args | Bad schema or vague description | Add example, tighten enum, split tool |
| Context window overflow | Tool returns blob unchecked | Summarise tool outputs above N tokens |
| Agent ignores instructions | Long, contradictory system prompt | Shorten, deduplicate, move details to tools |
| Confident but wrong | Missing grounding | Force a search or DB call before the answer |
| Works at $0.10, breaks at $10 | No budget cap | Add cost ceiling and downgrade fallback |

---

## A worked example: the data-analysis agent

The patterns in this guide collapse into the following minimal but realistic agent, which loads a CSV, cleans it, runs exploratory analysis, generates plots, and writes a report.

```python
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Shared state -- in production this would be a real store
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

Three things this example illustrates beyond the obvious:

- **Shared state through a singleton** keeps the LLM's job small. The model decides *what* to do; data flow happens out-of-band.
- **The system prompt enumerates the workflow.** The agent is allowed to choose tactics within steps, not whether to skip steps.
- **Each tool has a one-line, example-rich description.** This is where most of the "agent quality" actually lives.

To productionise this, add the cost cap, structured logging of every tool call, an eval suite of 20-50 representative CSVs, and a pre-flight schema validation tool the agent must call before plotting. None of these are interesting; all of them are necessary.

---

## FAQ

**Agent vs chatbot?**
A chatbot reacts. An agent decides. The same LLM can be either, depending on whether you wrap it in a loop with tools.

**CoT vs ReAct vs ToT -- how do I choose?**
Default to CoT. Move to ReAct when you need tools. Reach for ToT only when ReAct fails consistently and you can afford 10x the calls.

**Single agent or multi-agent?**
Single agent until the prompt becomes unmanageable or the trajectory exceeds your context budget. Then split along *artifact boundaries* (PRD -> design -> code -> tests), not along arbitrary role boundaries.

**How do I stop hallucinations in production?**
Ground every fact through a tool call. Require citations for non-trivial claims. Add an "I don't know" exit path and reward it. Sample outputs offline and label hallucinations -- you cannot fix what you do not measure.

**What are the most common failure modes?**
Loops, tool argument errors, context overflow from unsummarised tool output, goal drift on long tasks, and silent cost blow-ups. Every one has a one-line fix that nobody puts in until it ships once.

**Should I use a hosted agent platform or build my own?**
Hosted (OpenAI Assistants, Bedrock Agents, etc.) for prototypes and low-traffic internal tools. Build on a framework like LangGraph or AutoGen when you need control over routing, retries, observability, or cost. Do not write your own agent framework from scratch unless you have a strong reason -- the boring parts are surprisingly deep.

---

## Conclusion

The most important thing to internalise about agents is that they are infrastructure, not magic. The LLM is a single component. What makes an agent useful is the loop around it: a planner that decomposes work, a memory that survives steps, a tool layer that grounds the model in reality, and a critic that catches mistakes before they ship.

Six things worth remembering:

- **Start simple.** A ReAct agent with three good tools beats a multi-agent system with twelve mediocre ones.
- **Tools are the interface to reality.** Tool design quality dominates agent quality.
- **Memory rots.** Plan for eviction, audit, and the day you have to delete a poisoned fact.
- **Plan for failure.** Step caps, cost caps, sandboxes, retries, fallbacks. All of them. Up front.
- **Measure before you tune.** An eval suite that runs in CI is the difference between an agent that improves and one that drifts.
- **Ship the boring version.** A LangGraph with explicit nodes will outlive the autonomous AutoGPT-style demo every time.

**Further reading**

- [LangChain documentation](https://docs.langchain.com/)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://www.crewai.com/)
- [AgentBench paper](https://arxiv.org/abs/2308.03688)
- [GAIA benchmark](https://arxiv.org/abs/2311.12983)
- [Reflexion paper](https://arxiv.org/abs/2303.11366)
- [ReAct paper](https://arxiv.org/abs/2210.03629)
- [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic tool use guide](https://docs.anthropic.com/claude/docs/tool-use)
