---
title: "NLP (12): Frontiers and Practical Applications"
date: 2025-08-27 09:00:00
tags:
  - NLP
  - LLM
  - Agents
  - Code Generation
  - Deployment
categories: Natural Language Processing
series: NLP
part: 12
total_parts: 12
lang: en
mathjax: true
description: "Series finale: agents and tool use (Function Calling, ReAct), code generation (Code Llama, Codex), long-context attention (Longformer, Infini-attention), reasoning models (o1, R1), safety and alignment, evaluation, and production deployment with FastAPI, vLLM and Docker."
disableNunjucks: true
---

We have spent eleven chapters climbing from raw text to multimodal foundation models. This twelfth and final chapter sits at the frontier and at the runway. It is where research stops being a paper and starts being a service: an LLM that calls tools, writes and debugs code, reasons through hundred-step problems, ingests a 200K-token contract, and serves a thousand concurrent users behind a FastAPI endpoint with p95 latency under 300 ms.

Capability brings new failure modes. Models hallucinate confidently, generate harmful content if pushed, leak training data, and cost a small fortune to run badly. So this chapter is split in two halves. The first half — agents, code generation, long context, reasoning — is the frontier of what models can do. The second half — safety, evaluation, deployment — is the engineering needed to put that frontier into production without burning users or budgets.

## What you will learn

- **Agents**: the Function Calling protocol and the ReAct reason-act loop, with worked Python.
- **Code generation**: where Codex / Code Llama / DeepSeek-Coder sit on HumanEval, and how a self-repair loop works.
- **Long context**: sliding-window, dilated and Infini-attention masks, and when to use which.
- **Reasoning models**: how o1 and DeepSeek-R1 trade test-time compute for accuracy, and why CoT is now policy-conditioned.
- **Safety**: hallucination taxonomy, RLHF / DPO / Constitutional AI alignment, content guardrails.
- **Evaluation**: capability, safety and efficiency benchmarks, and what each of them does *not* tell you.
- **Production**: a FastAPI + vLLM + Docker reference stack, observability, and concrete latency targets.

## Prerequisites

- The previous eleven chapters of this series; we lean especially on Parts 4 (Transformer), 6 (GPT), 8 (PEFT), 9 (LLM internals) and 10 (RAG).
- Comfort with Python, basic asyncio, and the rough shape of a Docker container.
- A passing familiarity with reinforcement learning helps for the alignment section; see [RL Part 12 — RLHF and LLM Applications](/en/reinforcement-learning-12-rlhf-and-llm-applications/) for the long version.

---

## 1. Agents and tool use

The single biggest capability jump after instruction-tuning was teaching models to *call functions*. A vanilla LLM is a frozen approximator: whatever it knew at training time is all it knows. An agentic LLM is a controller that can ask the world for facts, run code, query databases, and then continue generating. That changes the system from a clever autocomplete into a programmable executor.

![ReAct agent architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig1_agent_architecture.png)

### 1.1 Function Calling — the protocol

Function Calling, popularised by OpenAI in mid-2023 and now standard across Claude, Gemini, Llama-3 and Qwen, is a typed protocol layered on top of chat completion. The application declares tools as JSON Schema; the model is trained to either reply directly or emit a structured tool call. Five stages, no magic:

![Function Calling pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig2_tool_use_pipeline.png)

1. **Schema declaration** — the application registers each tool with a name, a docstring, and a JSON-Schema for arguments.
2. **Routing** — given the user message, the model decides whether to answer or call.
3. **Argument generation** — constrained decoding produces a JSON object that validates against the schema.
4. **Execution** — the *application*, not the model, runs the function, ideally in a sandbox.
5. **Integration** — the result is appended to the conversation as a `tool` message; the model then composes the final reply.

```python
import json
from openai import OpenAI

client = OpenAI()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city. Returns temperature in Celsius and a short condition string.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name in English"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["city"],
        },
    },
}]

def get_weather(city: str, unit: str = "celsius") -> dict:
    # Real implementation would call a weather API.
    return {"city": city, "temp": 25, "unit": unit, "condition": "sunny"}

def chat(user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = resp.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = get_weather(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return resp.choices[0].message.content
    return msg.content
```

Three details people get wrong. First, **let the model decide** whether to call (`tool_choice="auto"`); forcing a call when none is needed produces silly arguments. Second, **always sandbox execution** — the model can hallucinate a `delete_all_users()` call if your schema lets it. Third, the tool description is a *prompt*: rewrite it until the model picks the right tool reliably.

### 1.2 ReAct — reasoning + acting in a loop

Function Calling is a single-shot interface. **ReAct** (Yao et al., ICLR 2023, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)) generalises it into an iterative `Thought -> Action -> Observation` loop, so the model can decompose, branch, and re-plan. This is the architectural backbone of LangChain, AutoGPT, OpenAI's Assistants, Claude's tool-use mode and most production agents in 2025.

```python
import re
from dataclasses import dataclass
from typing import Callable

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]

REACT_PROMPT = """You are a careful reasoning agent. You have these tools:
{tool_block}

Use this exact format on every turn:
Thought: <one sentence reasoning>
Action: <tool name>
Action Input: <single-line argument string>

When done, instead emit:
Thought: <reasoning>
Action: Final Answer
Action Input: <answer to the user>

Question: {question}
{scratchpad}"""

class ReActAgent:
    def __init__(self, llm: Callable[[str], str], tools: list[Tool], max_steps: int = 8):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _render_tools(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def _parse(self, txt: str):
        m = re.search(r"Action:\s*(.+?)\s*\nAction Input:\s*(.+)", txt, re.S)
        if not m:
            raise ValueError(f"Cannot parse model output:\n{txt}")
        return m.group(1).strip(), m.group(2).strip()

    def run(self, question: str) -> str:
        scratch = ""
        for step in range(self.max_steps):
            prompt = REACT_PROMPT.format(
                tool_block=self._render_tools(),
                question=question,
                scratchpad=scratch,
            )
            out = self.llm(prompt)
            try:
                action, action_input = self._parse(out)
            except ValueError:
                return f"[malformed output at step {step}]\n{out}"

            if action == "Final Answer":
                return action_input
            if action not in self.tools:
                obs = f"Unknown tool '{action}'. Available: {list(self.tools)}"
            else:
                obs = self.tools[action].fn(action_input)

            scratch += f"\n{out}\nObservation: {obs}"
        return "[max_steps reached] " + scratch[-500:]
```

When to choose what. **Function Calling** is the right default: it is structured, the prompt is short, and one round-trip is cheap. **ReAct** earns its overhead when the task needs multi-step planning, branching on intermediate observations, or recovering from tool failures — research summarisation, data analysis, multi-hop QA, complex bookings. For everything in between, modern frameworks let you express either as a graph of Function Calling steps with explicit state, which is usually the most maintainable choice.

---

## 2. Code generation

Code is the application area where LLMs have most clearly crossed from "interesting demo" to "indispensable tool". GitHub reports that Copilot users accept roughly a third of suggestions and complete tasks ~55% faster on benchmark tasks. The technical recipe behind that shift is straightforward: pretrain on code, fine-tune on instructions, evaluate by *running* the output, and add a self-repair loop.

![Code generation pipeline and HumanEval pass@1](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig3_code_generation.png)

### 2.1 The pipeline

A modern code-LLM pipeline has five stages: an **intent** in natural language, an enriched **context** (open files, repository symbols, test stubs, retrieved API docs), a **decoder** trained on code, an **executor** that compiles and runs unit tests, and a **self-repair** step that feeds the failure back to the model. The repair loop is what turns 65% pass@1 into 85% pass@5. AlphaCode, Reflexion and Code Llama's instruction variants all use a version of it.

### 2.2 Models and benchmarks

The standard benchmark is **HumanEval** (Chen et al., [arXiv:2107.03374](https://arxiv.org/abs/2107.03374), 2021): 164 hand-written Python problems graded by execution. *pass@k* is the probability that at least one of $k$ samples passes all unit tests; $\mathrm{pass}@1$ is the strict, single-shot measure. Two notes of caution: HumanEval is short, single-file and Python-only, so it overestimates real-world performance; and it has been heavily contaminated, so 2024+ scores should be cross-checked against MBPP, LiveCodeBench, SWE-bench Verified or CRUXEval.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeLlamaInstruct:
    """Minimal wrapper around Code Llama 7B-Instruct."""

    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Instruct-hf"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate(self, instruction: str, max_new_tokens: int = 512) -> str:
        prompt = f"<s>[INST] {instruction.strip()} [/INST]"
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens,
                temperature=0.2, top_p=0.95, do_sample=True,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text.split("[/INST]")[-1].strip()
```

For Python-only work in 2025, DeepSeek-Coder-V2, Qwen2.5-Coder and GPT-4o sit at the top; for repo-scale tasks involving real PRs, SWE-bench Verified is a much better proxy than HumanEval and the leaderboard looks completely different — Claude 3.5 / 3.7 Sonnet, GPT-4o and Devin agents lead, often with execution agents on top of weaker base models.

---

## 3. Long-context modeling

Standard self-attention costs $O(n^2)$ in time *and* memory. Doubling context from 4K to 8K is barely noticeable; going from 32K to 128K is a memory cliff. Several families of techniques push past it; you usually combine two or three.

![Long-context attention masks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig4_long_context.png)

- **Sliding window** (Longformer, [Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)) — each query attends only to the last $w$ keys, dropping cost to $O(nw)$. Captures local structure; relies on stacked layers to propagate distant information.
- **Dilated + global tokens** (BigBird, Longformer-global) — sparse strided attention plus a few "global" tokens that everyone attends to. Good for QA where the question token must see the whole document.
- **Infini-attention** ([Munkhdalai et al., 2024](https://arxiv.org/abs/2404.07143)) — keep a small local window for precision *plus* a compressive memory that summarises everything older. Bounded memory, unbounded effective context.
- **Position-encoding extension** — RoPE base scaling, NTK-aware interpolation, YaRN, LongRoPE — extend a model trained at 4K to 128K with little or no fine-tuning by reparameterising the rotary frequencies.
- **Parameter-efficient long-context fine-tuning** — LongLoRA combines shifted sparse attention with LoRA so that a 7B model can be extended to 100K context on two A100s.

In practice, modern long-context LLMs combine RoPE extension during pretraining, sliding-window or grouped-query attention in the kernel, and continued training on long-document mixtures. At inference time, FlashAttention-2/3 and PagedAttention (vLLM) keep the constant factors sane.

---

## 4. Reasoning models

The 2024-2025 inflection point in NLP was **test-time compute**: instead of making the base model bigger, train it to *think for longer* before answering. OpenAI's o1 (Sep 2024) and DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948), Jan 2025) both do this: an internal chain-of-thought is generated, scored, sometimes rolled back, and only the final answer is returned to the user.

![Reasoning models — chain-of-thought and test-time scaling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig5_reasoning_models.png)

Two ideas drive the gain. First, **process supervision** — train a reward model to score *intermediate steps*, not just the final answer (Lightman et al., *Let's Verify Step by Step*, 2023). Second, **outcome RL on verifiable tasks** — for math and code where success is checkable, run large-scale RL with the verifier as the reward. DeepSeek-R1's "R1-Zero" recipe shows that reasoning emerges from pure RL on a strong base model, with no SFT step at all; aha moments and self-correction appear spontaneously.

The cost. Reasoning models are slow and token-hungry — a single AIME problem can consume 10K-100K reasoning tokens. They also hide their CoT (o1 charges for hidden reasoning tokens), which raises new evaluation and trust questions. The right design pattern in 2025 is a **router**: send easy queries to a fast model (GPT-4o, Claude Haiku), hard reasoning queries to o1 / R1 / Claude Sonnet thinking mode. The cost-quality difference is often 10-50x.

---

## 5. Safety, alignment and hallucinations

A model that is helpful but unsafe is unshippable. A model that is safe but unhelpful is worse than no model. Alignment is the engineering discipline that tries to land in the narrow strip in between.

### 5.1 The hallucination taxonomy

Hallucinations are not a single failure mode. The useful split (Huang et al., *Survey of Hallucination*, 2023) is:

- **Factuality** — the output contradicts a verifiable fact ("Marie Curie won the Fields Medal").
- **Faithfulness** — the output contradicts the *provided context* in a RAG or summarisation setting ("the contract says payment is due in 30 days" when it actually says 60).
- **Logical / arithmetic** — invalid reasoning steps that happen to look fluent.
- **Self-consistency** — the model contradicts an earlier statement in the same conversation.

Mitigations stack: **RAG** for factuality (Part 10), **constrained decoding** and JSON schemas for structural faithfulness, **self-consistency sampling** plus majority vote for arithmetic, **process supervision** and reasoning models for logical errors, **citation-required prompting** for verifiability, and **abstention training** ("say I don't know") as a last line of defence.

### 5.2 Alignment — RLHF, DPO, Constitutional AI

The dominant alignment recipe is still a three-stage pipeline: SFT on demonstrations, train a reward model on human preference pairs, then RL (PPO) against that reward. **DPO** (Rafailov et al., 2023) collapses the last two stages into a single supervised loss and is now the default in many open recipes because it is much simpler and cheaper. **Constitutional AI** (Bai et al., Anthropic, 2022) replaces most human labels with model-generated critiques against a written list of principles, making large-scale safety tuning tractable.

```python
class GuardedLLM:
    """Wrap any chat function with input + output safety classifiers."""

    def __init__(self, llm, in_filter, out_filter, refusal: str = "I cannot help with that request."):
        self.llm, self.inf, self.outf = llm, in_filter, out_filter
        self.refusal = refusal

    def chat(self, user_msg: str) -> str:
        if self.inf(user_msg).get("label") == "unsafe":
            return self.refusal
        reply = self.llm(user_msg)
        if self.outf(reply).get("label") == "unsafe":
            return self.refusal
        return reply
```

In production, the input filter usually catches jailbreak attempts and prompt injection (especially from RAG sources, which is now the dominant attack vector); the output filter catches PII, toxicity and policy violations. Both filters are themselves small classifiers (Llama-Guard-3, ShieldGemma, OpenAI Moderation) — cheaper and more predictable than asking the main model to police itself.

---

## 6. Evaluation

Benchmarks are how we lie to ourselves least. Three axes matter, and most teams under-invest in two of them.

| Axis | Public benchmarks | What it actually measures |
|---|---|---|
| Capability | MMLU, GPQA-Diamond, GSM8K, MATH, HumanEval, MBPP, SWE-bench, MT-Bench, Arena-Hard, C-Eval (zh) | Knowledge + reasoning + generation, mostly under contamination risk |
| Safety | TruthfulQA, ToxiGen, HarmBench, JailbreakBench, BBQ | Refusal behaviour, bias, jailbreak robustness |
| Efficiency | MLPerf-Inference, vLLM bench, MMLU-Pro tokens/answer | Throughput, p50/p95 latency, cost per million tokens |

Three habits that pay off. **Build a private eval set** of 100-500 prompts from your real users; it is the only number that correlates with launch outcomes. **Run pairwise comparison** (LLM-as-judge with chain-of-thought, plus periodic human spot-checks) — absolute scores drift, pairwise wins do not. **Track regressions per release** in a fixed harness (`lm-evaluation-harness`, `evalplus`, `inspect_ai`); a 2-point drop on your private eval should block a deploy even if MMLU went up.

---

## 7. Production deployment

This is where the pretty notebook meets reality. The reference stack we will sketch — FastAPI in front, vLLM in the middle, Docker + K8s wrapping the lot, Prometheus watching it all — is the same pattern used by most production teams in 2025, modulo religious preferences about Triton vs. TGI vs. TensorRT-LLM.

![Production deployment stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig6_production_deploy.png)

### 7.1 The serving layer

Forget `transformers.generate` for serving. It is fine for prototypes and disastrous in production: no continuous batching, no PagedAttention, naive KV-cache management. Use **vLLM** (the open-source default), **TensorRT-LLM** (NVIDIA, fastest on H100), or **TGI** (Hugging Face, simplest ops). All three implement continuous batching, paged KV-cache, prefix caching and speculative decoding.

```python
# vLLM as a library — production deployments usually run `vllm serve` instead.
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
)
params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
out = llm.generate(["Explain LoRA in two sentences."], params)
print(out[0].outputs[0].text)
```

### 7.2 The API layer

FastAPI is the path of least resistance: async, pydantic validation, OpenAPI for free, and it streams Server-Sent Events without ceremony. The non-obvious bit is **request hygiene** — strict input limits, request IDs, structured logging, and a fallback path when the GPU pool is saturated.

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio, time, uuid, logging

logger = logging.getLogger("nlp-api")
app = FastAPI(title="NLP Service", version="1.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False

async def _stream(prompt: str, max_tokens: int, temperature: float):
    # Replace with a real vLLM async generator.
    for tok in prompt.split():
        await asyncio.sleep(0.02)
        yield f"data: {tok} \n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        if req.stream:
            return StreamingResponse(
                _stream(req.message, req.max_tokens, req.temperature),
                media_type="text/event-stream",
                headers={"X-Request-Id": rid},
            )
        # Non-streaming path...
        reply = "stub"
        return {"id": rid, "reply": reply}
    except Exception as exc:
        logger.exception("rid=%s failed: %s", rid, exc)
        raise HTTPException(500, "internal error") from exc
    finally:
        logger.info("rid=%s latency_ms=%.1f", rid, (time.perf_counter() - t0) * 1000)

@app.get("/healthz")
async def health(): return {"status": "ok"}
```

### 7.3 Containerisation

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY app/ ./app/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

For Kubernetes, schedule the Pod onto a GPU node with `nvidia.com/gpu: 1`, set tight CPU/memory requests, configure a `livenessProbe` on `/healthz` and a `readinessProbe` that only flips to *ready* after the model is loaded (the most common deployment bug is taking traffic during the 30-90 s warm-up).

### 7.4 Observability and SLOs

You cannot fix what you do not measure. The minimum viable instrumentation:

- **Metrics** (Prometheus + Grafana): QPS, latency p50/p95/p99, GPU utilisation, VRAM, queue depth, cache hit rate, tokens-per-second.
- **Tracing** (OpenTelemetry): one span per request, with sub-spans for retrieval, generation, post-processing.
- **Logs** (Loki / ELK): structured JSON, every log line carries the request ID.
- **Errors** (Sentry): grouped by exception class, alert on rate spikes.

Sensible launch targets for a 7-8B model on a single A100 80GB with vLLM and bf16: **~1500-2500 tokens/s/GPU** aggregate throughput, **p95 first-token < 300 ms**, **p95 inter-token < 80 ms**, **~30 concurrent streams**. Quantising to 4-bit AWQ doubles throughput at the cost of 1-2 points on most benchmarks; ship a small private eval to confirm the trade is acceptable.

---

## 8. Frequently asked questions

**When should I pick Function Calling vs. ReAct vs. a graph framework?** Single tool, deterministic invocation: Function Calling. Multi-step tasks with branching, retries and intermediate inspection: ReAct or a graph framework like LangGraph. If you find yourself writing a state machine on top of ReAct strings, switch to a graph — it makes the control flow explicit and testable.

**Open-source model or hosted API?** Hosted APIs win on time-to-market, capability ceiling and the cost of *low* request volumes. Open-source self-hosting wins on data sovereignty, fine-tuning freedom and cost at high sustained throughput (the crossover is roughly 50-100 M tokens/day for a 7-13B model on owned hardware).

**Do I really need a reasoning model?** For routine chat, summarisation, classification and short code: no, a fast non-reasoning model is 10-50x cheaper and good enough. For competition math, complex code, multi-hop research and any "system 2" task: yes, the gap is qualitative. Route by query.

**Why does my agent loop forever?** Almost always one of: (1) tool descriptions are ambiguous and the model keeps re-trying the wrong one; (2) you forgot a `Final Answer` sentinel; (3) the observation is too long and pushes the original goal out of context. Cap iterations, log every step, and shrink observations to the relevant fields.

**What's the cheapest way to extend context?** If you control fine-tuning: RoPE base scaling + a small LongLoRA run. If not: chunk + RAG (Part 10). True 1M-token context is rarely the right answer — retrieval is cheaper, more debuggable, and usually more accurate.

---

## Series wrap-up

![The 12-chapter NLP journey](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/frontiers-applications/fig7_series_journey.png)

We started this series with whitespace tokenisation and ended with reasoning agents serving streamed tokens behind a load balancer. Looking at the map, four arcs are visible:

- **Foundations** (Parts 1-3) — text becomes vectors; sequences become hidden states.
- **Transformers** (Parts 4-6) — attention replaces recurrence; pretraining splits into BERT-style understanding and GPT-style generation.
- **Adaptation** (Parts 7-9) — prompting, PEFT and LLM internals turn pretrained models into tools we can shape.
- **Applications** (Parts 10-12) — RAG anchors models to knowledge, multimodality breaks the text-only ceiling, and agents + production engineering close the loop into a deployable system.

If you take three things from the whole series, take these. **Architecture matters less than data and objectives** — the Transformer is now table stakes; the differentiation is in pretraining mix, instruction data and evaluation. **Engineering discipline beats model size** — a careful 8B-model deployment with RAG, evals and guardrails will outperform a sloppy 70B service every time. **The frontier moves fast, but the fundamentals are stable** — embeddings, attention, fine-tuning, retrieval, alignment and serving will still be the right vocabulary in 2027, even when the specific model names are different.

Thank you for reading all twelve parts. Now go build something.

---

## References

- Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models**, ICLR 2023, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- Schick et al., **Toolformer: Language Models Can Teach Themselves to Use Tools**, NeurIPS 2023, [arXiv:2302.04761](https://arxiv.org/abs/2302.04761).
- Roziere et al., **Code Llama: Open Foundation Models for Code**, Meta AI, 2023, [arXiv:2308.12950](https://arxiv.org/abs/2308.12950).
- Chen et al., **Evaluating Large Language Models Trained on Code (HumanEval / Codex)**, 2021, [arXiv:2107.03374](https://arxiv.org/abs/2107.03374).
- Beltagy et al., **Longformer: The Long-Document Transformer**, 2020, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).
- Munkhdalai et al., **Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention**, 2024, [arXiv:2404.07143](https://arxiv.org/abs/2404.07143).
- Chen et al., **LongLoRA: Efficient Fine-tuning of Long-Context LLMs**, ICLR 2024, [arXiv:2309.12307](https://arxiv.org/abs/2309.12307).
- Lightman et al., **Let's Verify Step by Step**, 2023, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
- OpenAI, **Learning to Reason with LLMs (o1 system card)**, 2024.
- DeepSeek-AI, **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL**, 2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
- Rafailov et al., **Direct Preference Optimization**, NeurIPS 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
- Bai et al., **Constitutional AI: Harmlessness from AI Feedback**, Anthropic, 2022, [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).
- Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)**, SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- Dao, **FlashAttention-2**, 2023, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).

---

## Series navigation

- **Previous**: [Part 11 — Multimodal NLP](/en/nlp-multimodal-nlp/)
- **Series complete!** This concludes the 12-part NLP series.
- [View all 12 parts in the NLP series](/tags/NLP/)
